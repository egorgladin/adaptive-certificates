"""Independent derivative, center, oracle, certificate and saved-output audits."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .certificate import historical_norm, solve_certificate
from .problem import Problem
from .volumetric import VolumetricCP


def tiny_checks():
    random = np.random.default_rng(31)
    problem = Problem(12, 4)
    optimum = problem.solution()
    assert abs(np.linalg.norm(optimum)-1) < 1e-12
    assert abs(problem.value(optimum)+.5) < 1e-12
    for _ in range(50):
        x, z = random.normal(size=(2, 12))*.2
        assert problem.value(z) >= problem.value(x)+problem.gradient(x) @ (z-x)-1e-12
    cp = VolumetricCP(3, center_tolerance=1e-7)
    cp.cut(np.array([1., .2, -.1]), cp.y.copy())
    y = cp.y+np.array([.05, .07, -.02])
    _, _, step, dec2, _ = cp.geometry(y)
    s = cp.b-cp.A @ y
    B = cp.A/s[:, None]
    H = B.T @ B
    inv = np.linalg.inv(H)
    T = B @ inv @ B.T
    gradient = B.T @ np.diag(T)
    hessian = 3*B.T @ (np.diag(T)[:,None]*B)-2*B.T @ ((T*T) @ B)
    h = 2e-5
    gradient_fd = np.empty(3); hessian_fd = np.empty((3,3))
    def raw_gradient(point):
        M = cp.A/(cp.b-cp.A @ point)[:,None]
        J = M @ np.linalg.inv(M.T @ M) @ M.T
        return M.T @ np.diag(J)
    for j in range(3):
        e = np.eye(3)[j]*h
        gradient_fd[j] = (cp.geometry(y+e,False)-cp.geometry(y-e,False))/(2*h)
        hessian_fd[:,j] = (raw_gradient(y+e)-raw_gradient(y-e))/(2*h)
    assert np.max(np.abs(gradient-gradient_fd)) < 1e-7
    assert np.max(np.abs(hessian-hessian_fd)) < 1e-6
    assert np.linalg.norm(hessian @ step-gradient) < 1e-10
    independent = minimize(lambda z: cp.geometry(z,False), cp.y,
                           jac=raw_gradient, method='BFGS', options={'gtol':1e-9})
    assert np.linalg.norm(independent.x-cp.y) < 1e-6
    # A nonzero historical profile ensures we test more than the exact-rank special case.
    records = {}
    for i in range(6):
        records[i] = {'v': random.normal(size=3), 'birth': i % 4, 'eta': .03*(i+1),
                      'scalar': .2*random.normal(), 'point': random.normal(size=3)*.2}
    C, output, info = solve_certificate(records, 3, 8)
    V = np.array([r['v'] for r in records.values()])
    b = np.array([r['scalar'] for r in records.values()])
    def objective(alpha):
        c = np.zeros(4)
        for weight, r in zip(alpha, records.values()):
            c[r['birth']] += weight*r['eta']
        return b @ alpha+2*historical_norm(V.T @ alpha,c)
    alternative = minimize(objective, np.full(6,1/6), method='SLSQP',
                           bounds=[(0,1)]*6, constraints=[{'type':'eq','fun':lambda a:a.sum()-1}],
                           options={'ftol':1e-12,'maxiter':1000})
    assert alternative.success
    assert abs(C-alternative.fun) < 1e-7
    assert abs(info['weights'].sum()-1) < 1e-14
    assert np.min(info['weights']) >= 0
    return {'oracle_support_checks':50, 'gradient_fd_max_error':float(np.max(abs(gradient-gradient_fd))),
            'hessian_fd_max_error':float(np.max(abs(hessian-hessian_fd))),
            'certificate_vs_independent_slsqp':float(abs(C-alternative.fun)),
            'center_vs_independent_bfgs':float(np.linalg.norm(independent.x-cp.y))}


def audit_grid(root):
    rows = []
    for path in sorted(Path(root).glob('d*/result.json')):
        row = json.loads(path.read_text())
        if row['status'] != 'certified':
            rows.append({'cell':path.parent.name,'status':row['status']}); continue
        problem = Problem(row['d'],row['k'],row['seed'])
        data = np.load(path.parent/'final_records.npz')
        a = data['weights']; m = row['m']
        assert abs(a.sum()-1) < 1e-12 and np.min(a) >= 0
        aggregate = data['V'][:,:m].T @ a
        profile = np.bincount(data['birth'], weights=data['eta']*a, minlength=m+1)
        replay = float(data['scalar'] @ a+2*historical_norm(aggregate,profile,m==row['d'])+1e-11)
        point_coordinates = data['points'].T @ a
        point = point_coordinates if row['arm']=='ambient' else data['basis'] @ point_coordinates[:m]
        gap = problem.value(point)+.5
        assert abs(replay-row['last_certificate']) < 1e-9
        assert np.linalg.norm(point-data['output']) < 1e-10
        assert np.linalg.norm(point) <= 2+1e-10
        assert gap <= replay+1e-9 and replay <= min(row['targets'])
        assert row['max_constraints_per_q'] <= 26+1e-8
        assert row['max_center_decrement'] <= 1.01e-5
        if row['arm']=='adaptive':
            assert m <= row['k'] and row['q'] <= 2*m
        traces = [json.loads(line) for line in (path.parent/'trace.jsonl').read_text().splitlines()]
        for endpoint, hit in row['hits'].items():
            first = next(t for t in traces if t['certificate'] <= float(endpoint))
            assert hit['calls'] == first['calls']
            assert hit['dual_gap'] <= float(endpoint)/4
        rows.append({'cell':path.parent.name,'status':'passed', 'actual_gap':gap,
                     'certificate':replay,'output_replay_error':float(np.linalg.norm(point-data['output']))})
    return {'cells':rows,'passed':sum(r['status']=='passed' for r in rows),'total':len(rows)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--grid')
    args = parser.parse_args()
    result = audit_grid(args.grid) if args.grid else tiny_checks()
    print(json.dumps(result, indent=2))
