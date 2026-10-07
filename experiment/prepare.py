"""Train-only preprocessing, fixed median RBF width, validation-only lambda."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, cdist
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, NuSVR
from sklearn.metrics import accuracy_score, mean_squared_error

ROOT = Path('output')
SEED = 20261007
SOURCES = {
    'abalone': 'https://archive.ics.uci.edu/static/public/1/abalone.zip',
    'airfoil': 'https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip',
    'adult': 'https://archive.ics.uci.edu/static/public/2/adult.zip',
    'breast_cancer': 'https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip',
}


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def raw(dataset):
    directory = ROOT/'data/raw'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/(dataset+'.zip')
    if not path.exists():
        req = urllib.request.Request(SOURCES[dataset], headers={'User-Agent':'Mozilla/5.0 research experiment'})
        with urllib.request.urlopen(req, timeout=60) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.testzip()
        path.write_bytes(payload)
    with zipfile.ZipFile(path) as archive:
        name = {'abalone':'abalone.data','airfoil':'airfoil_self_noise.dat',
                'adult':'adult.data','breast_cancer':'wdbc.data'}[dataset]
        member = next(n for n in archive.namelist() if Path(n).name == name)
        content = archive.read(member)
    provenance = dict(url=SOURCES[dataset])
    if dataset == 'airfoil':
        table = np.loadtxt(io.BytesIO(content))
        return table[:,:-1], table[:,-1], [], provenance
    if dataset == 'breast_cancer':
        rows = [line.split(',') for line in content.decode().splitlines() if line.strip()]
        X = np.array([[float(v) for v in row[2:]] for row in rows])
        y = np.array([1 if row[1]=='M' else -1 for row in rows])
        return X,y,[],provenance
    rows = [list(map(str.strip,line.split(','))) for line in content.decode().splitlines() if line.strip()]
    if dataset == 'abalone':
        return np.array([row[:-1] for row in rows],dtype=object), np.array([float(row[-1]) for row in rows]), [0], provenance
    rows = [row for row in rows if '?' not in row]
    categories = [1,3,5,6,7,8,9,13]
    X = np.array([row[:-1] for row in rows],dtype=object)
    y = np.array([1 if row[-1].rstrip('.')=='>50K' else -1 for row in rows])
    provenance['missing_rows_removed'] = True
    provenance['official_test_not_used'] = 'A seeded split of adult.data is used throughout.'
    return X,y,categories,provenance


def prepare(dataset):
    output = ROOT/'data'/dataset
    if (output/'metadata.json').exists():
        return json.loads((output/'metadata.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    X,y,categories,provenance = raw(dataset)
    classification = dataset in ('adult','breast_cancer')
    indices = np.arange(len(y))
    train, held = train_test_split(indices, test_size=.3, random_state=SEED,
                                  stratify=y if classification else None)
    valid,test = train_test_split(held, test_size=.5, random_state=SEED+1,
                                 stratify=y[held] if classification else None)
    # Large dataset only: cap the training population before comparing any arm.
    if len(train)>2048:
        train,_ = train_test_split(train, train_size=2048, random_state=SEED+2,
                                  stratify=y[train] if classification else None)
    train,valid,test = map(np.sort,(train,valid,test))
    numeric = [i for i in range(X.shape[1]) if i not in categories]
    Z = X[:,numeric].astype(float)
    center,scale = Z[train].mean(0), Z[train].std(0)
    scale[scale==0] = 1
    features = [(Z-center)/scale]
    encodings = {}
    for column in categories:
        values = sorted(set(X[train,column]))
        encodings[str(column)] = values
        features.append(np.column_stack([X[:,column]==v for v in values]).astype(float))
    X = np.column_stack(features)
    target_mean,target_std = (0.,1.) if classification else (float(y[train].mean()),float(y[train].std()))
    y = (y-target_mean)/target_std
    sigma = float(np.median(pdist(X[train])))
    gamma = 1/(2*sigma*sigma)
    kernel = lambda rows: np.exp(-gamma*cdist(X[rows],X[train],metric='sqeuclidean'))
    K = kernel(train)
    np.fill_diagonal(K,1.)
    Kvalid,Ktest = kernel(valid),kernel(test)
    candidates = []
    models = {}
    for lam in (1e-4,1e-3,1e-2):
        model = SVC(C=1/(len(train)*lam),kernel='precomputed',tol=1e-6) if classification else NuSVR(C=1/(len(train)*lam),nu=.1,kernel='precomputed',tol=1e-6)
        model.fit(K,y[train])
        prediction = model.predict(Kvalid)
        metric = float(accuracy_score(y[valid],prediction)) if classification else float(np.sqrt(mean_squared_error(y[valid],prediction)))
        candidates.append(dict(lam=lam,validation_metric=metric,solver='SVC' if classification else 'NuSVR',fit_status=int(model.fit_status_)))
        models[lam] = model
    choice = (max if classification else min)(candidates,key=lambda c:c['validation_metric'])
    lam = choice['lam']
    model = models[lam]
    eig,U = eigh(K,check_finite=False,driver='evd')
    if eig.min() < -1e-10*eig.max():
        raise ArithmeticError('Kernel is not numerically PSD')
    eig = np.maximum(eig,0.)  # No positive eigenvalue truncation.
    root = np.sqrt(eig)
    B = np.ascontiguousarray(U*root)
    inverse = np.divide(1.,root,out=np.zeros_like(root),where=root>0)
    Bvalid = np.ascontiguousarray((Kvalid@U)*inverse)
    Btest = np.ascontiguousarray((Ktest@U)*inverse)
    dual = np.zeros(len(train))
    dual[model.support_] = model.dual_coef_[0]
    reference_x = B.T@dual
    arrays = dict(B=B,Bvalid=Bvalid,Btest=Btest,y=y[train],yvalid=y[valid],ytest=y[test],
                  train_rows=train,validation_rows=valid,test_rows=test,eigenvalues=eig,
                  reference_x=reference_x)
    for name,value in arrays.items():
        np.save(output/(name+'.npy'),value)
    metadata = dict(dataset=dataset,problem='svm' if classification else 'nusvr',source=provenance,
                    source_records=len(y),train=len(train),validation=len(valid),test=len(test),
                    input_features=X.shape[1],d=len(train),seed=SEED,train_cap=2048,
                    split='70/15/15 then training cap2048; stratified classification',
                    preprocessing='numeric train mean/std; train-only one-hot dictionary; standardized regression targets',
                    category_dictionary=encodings,target_mean=target_mean,target_std=target_std,
                    sigma=sigma,width_multiplier=1.,gamma=gamma,lam=lam,nu=.1,
                    lambda_selection=candidates,metric='accuracy' if classification else 'RMSE in standardized target units',
                    model='full RBF spectral factor; all positive eigenvalues retained; free intercept profiled',
                    factor_error=float(np.max(np.abs(B@B.T-K))),
                    effective_ranks={str(t):int(np.sum(eig>eig.max()*t)) for t in (1e-4,1e-6,1e-8)},
                    reference_intercept=float(model.intercept_[0]),
                    reference_test_metric=float(accuracy_score(y[test],model.predict(Ktest))) if classification else float(np.sqrt(mean_squared_error(y[test],model.predict(Ktest)))),
                    preparation_seconds=time.perf_counter()-started)
    write(output/'metadata.json',metadata)
    print(json.dumps(metadata,ensure_ascii=False),flush=True)
    return metadata


if __name__ == '__main__':
    for name in ('abalone','airfoil','adult','breast_cancer'):
        prepare(name)
