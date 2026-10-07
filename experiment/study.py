"""Small sequential study; configurations and all negative outputs are retained."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .runner import run,initialize
from .oracle import KernelProblem

ROOT = Path('output/results')
DATASETS = ('abalone','airfoil','adult','breast_cancer')
METHODS = ('subgradient','momentum','adagrad')


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def tuning():
    output = ROOT/'tuning'
    output.mkdir(parents=True,exist_ok=False)
    selections = {}
    for dataset in DATASETS:
        problem = KernelProblem(dataset)
        for method in METHODS:
            rates = (.1,1.,10.,100.,1000.) if method!='adagrad' else (.01,.03,.1,.3,1.)
            candidates = []
            for lr in rates:
                result = run(dataset,method,lr,'single_cut',output/f'{dataset}_{method}_{lr:g}',
                             records=500,eps_rel=(1e-8,),stop=False,score=False,problem=problem)
                candidates.append(dict(lr=lr,U=result['U'],gap=result['gap'],records=result['records'],
                                       path=str(output/f'{dataset}_{method}_{lr:g}')))
            selected = min(candidates,key=lambda c:c['U'])
            # One local refinement around the best coarse rate, by primal only.
            refinements = (selected['lr']/3, selected['lr']*3)
            for lr in refinements:
                if any(abs(np.log(lr/c['lr']))<1e-8 for c in candidates):
                    continue
                path = output/f'{dataset}_{method}_refine_{lr:g}'
                result = run(dataset,method,lr,'single_cut',path,records=500,eps_rel=(1e-8,),
                             stop=False,score=False,problem=problem)
                candidates.append(dict(lr=lr,U=result['U'],gap=result['gap'],records=result['records'],path=str(path)))
            selected = min(candidates,key=lambda c:c['U'])
            selections[f'{dataset}_{method}'] = dict(**selected,candidates=candidates,
                                                     criterion='minimum primal incumbent after500 exact FG; certificates/validation not used')
            write(ROOT/'rates.json',selections)
            print(json.dumps(dict(selected=f'{dataset}_{method}',**selected)),flush=True)
        del problem


def pilot():
    rates = json.loads((ROOT/'rates.json').read_text())
    output = ROOT/'pilot'
    output.mkdir(parents=True,exist_ok=False)
    selections = {}
    for dataset in DATASETS:
        problem = KernelProblem(dataset)
        initial = float(problem.oracle(torch.zeros(problem.d,dtype=torch.float64,device='cuda'))[0])
        base = np.sqrt(problem.lam*initial*.001)
        for method in METHODS:
            lr = rates[f'{dataset}_{method}']['lr']
            records = 2000
            for arm in ('full','single_cut'):
                result = run(dataset,method,lr,arm,output/f'{dataset}_{method}_{arm}',
                             records=records,eps_rel=(.001,),problem=problem)
            options = []
            # Three thresholds at one target, no Cartesian parameter sweep.
            for multiplier in (.03,.1,.3):
                tau = base*multiplier
                path = output/f'{dataset}_{method}_compressed_{multiplier:g}'
                result = run(dataset,method,lr,'compressed',path,records=records,eps_rel=(.001,),tau=tau,problem=problem)
                hit = result['first_hits'].get('0.001')
                options.append(dict(tau=tau,multiplier=multiplier,path=str(path),status=result['status'],
                                    gap=result['gap'],records=result['records'],k=result['k'],
                                    projected_seconds_at_1mib=(hit['seconds']+hit['bytes']/2**20) if hit else None))
            feasible = [c for c in options if c['projected_seconds_at_1mib'] is not None]
            selected = min(feasible,key=lambda c:c['projected_seconds_at_1mib']) if feasible else min(options,key=lambda c:c['gap'])
            selections[f'{dataset}_{method}'] = dict(**selected,candidates=options,
                                                     criterion='minimum modeled time at1MiB/s among certified candidates; otherwise smallest final gap')
            write(ROOT/'thresholds.json',selections)
            print(json.dumps(dict(selected_tau=f'{dataset}_{method}',**selected)),flush=True)
        del problem


def confirmation():
    rates = json.loads((ROOT/'rates.json').read_text())
    taus = json.loads((ROOT/'thresholds.json').read_text())
    output = ROOT/'confirmation'
    output.mkdir(parents=True,exist_ok=False)
    for repeat in range(3):
        for dataset in DATASETS:
            problem = KernelProblem(dataset)
            for method in METHODS:
                arms = ['full','compressed','single_cut']
                arms = arms[repeat:]+arms[:repeat]
                for arm in arms:
                    path = output/f'{dataset}_{method}_repeat{repeat}_{arm}'
                    result = run(dataset,method,rates[f'{dataset}_{method}']['lr'],arm,path,
                                 tau=taus[f'{dataset}_{method}']['tau'],bandwidth=1.,problem=problem)
            del problem


def sensitivity():
    rates = json.loads((ROOT/'rates.json').read_text())
    taus = json.loads((ROOT/'thresholds.json').read_text())
    output = ROOT/'sensitivity'
    output.mkdir(parents=True,exist_ok=False)
    for dataset in ('abalone','adult'):
        problem = KernelProblem(dataset)
        method = 'subgradient'  # Declared before results; no cherry-picked method.
        for arm in ('full','compressed','single_cut'):
            run(dataset,method,rates[f'{dataset}_{method}']['lr'],arm,output/f'{dataset}_{method}_{arm}',
                eps_rel=(.01,.001,.0001),tau=taus[f'{dataset}_{method}']['tau'],problem=problem)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',nargs='?',default='all',choices=('all','prepare','tuning','pilot','confirmation','sensitivity','results'))
    args = parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if args.stage in ('all','prepare'):
        from .prepare import prepare
        for dataset in DATASETS:
            prepare(dataset)
    if args.stage not in ('prepare','results'):
        initialize()
        for stage in ('tuning','pilot','confirmation','sensitivity') if args.stage=='all' else (args.stage,):
            globals()[stage]()
    if args.stage in ('all','results'):
        from .results import main
        main()
