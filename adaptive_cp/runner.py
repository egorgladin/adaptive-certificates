"""Serial fresh-process experiments, retaining every failure and trace."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

from .certificate import solve_certificate
from .problem import Problem
from .volumetric import VolumetricCP

ROOT = Path(__file__).resolve().parent
TAU = 1e-10


def run(d, k, arm, eps, seconds=600., max_calls=12000, seed=0):
    problem = Problem(d, k, seed)
    targets = sorted(set(eps), reverse=True)
    if arm == 'adaptive':
        basis = np.empty((d, 0)); m = 0; q = 1
    elif arm == 'ideal':
        basis = problem.U.copy(); m = q = k
    else:
        basis = None; m = q = d
    cp = VolumetricCP(q)
    records, hits, trace, phases = {}, {}, [], [q]
    calls = ball_cuts = total_deleted = total_newton = total_inserted = 0
    max_m = m
    max_eta = max_orthogonality = max_support_violation = 0.
    max_certificate_violation = max_dual_gap = 0.
    max_constraints_ratio = max_center_decrement = 0.
    min_insert_gain = np.inf
    last_point = np.zeros(d)
    certificate = None
    status = 'call_cap'
    error = None
    last_weights = np.empty(0)
    started = time.perf_counter()
    times = {'oracle': 0., 'basis': 0., 'primitive': 0., 'certificate': 0.}

    def cp_diagnostics():
        nonlocal max_constraints_ratio, max_center_decrement, min_insert_gain
        max_constraints_ratio = max(max_constraints_ratio, cp.max_constraints/cp.q)
        max_center_decrement = max(max_center_decrement, cp.max_center_decrement)
        min_insert_gain = min(min_insert_gain, cp.min_insert_barrier_gain)

    def retained_bytes():
        archive = sum(r['v'].nbytes+r['point'].nbytes+24 for r in records.values())
        return cp.state_bytes()+archive+(0 if basis is None else basis.nbytes)+8*d

    try:
        for update in range(30000):
            if time.perf_counter()-started >= seconds:
                status = 'time_cap'; break
            tick = time.perf_counter()
            y, removed = cp.prune()
            cp_diagnostics()
            for identifier in removed:
                records.pop(identifier, None)
            times['primitive'] += time.perf_counter()-tick
            if records:
                tick = time.perf_counter()
                certificate, average, solve_info = solve_certificate(records, m, d)
                last_weights = solve_info.pop('weights')
                times['certificate'] += time.perf_counter()-tick
                last_point = average.copy() if basis is None else basis @ average[:m]
                gap = problem.value(last_point)+.5  # Audit only; does not control queries/stop.
                max_certificate_violation = max(max_certificate_violation, gap-certificate)
                max_dual_gap = max(max_dual_gap, solve_info['dual_gap'])
                row = {
                    'calls': calls, 'certificate': certificate, 'actual_gap': gap,
                    'm': m, 'q': q, 'records': len(records), 'ball_cuts': ball_cuts,
                    'phase': len(phases), 'seconds': time.perf_counter()-started,
                    'retained_bytes': retained_bytes(), **solve_info,
                }
                trace.append(row)
                for target in targets:
                    if certificate <= target and str(target) not in hits:
                        hits[str(target)] = row.copy()
                if certificate <= min(targets):
                    status = 'certified'; break
            if np.linalg.norm(y) >= 2.:
                tick = time.perf_counter()
                cp.cut(y, y)
                ball_cuts += 1
                times['primitive'] += time.perf_counter()-tick
                continue
            if calls >= max_calls:
                status = 'call_cap'; break
            x = y.copy() if basis is None else basis @ y[:m]
            tick = time.perf_counter()
            g = problem.gradient(x)
            times['oracle'] += time.perf_counter()-tick
            calls += 1
            # Supporting inequality at the known minimizer is an independent audit.
            support = problem.value(x)+float(g @ (problem.solution()-x))+.5
            max_support_violation = max(max_support_violation, support)
            tick = time.perf_counter()
            if arm == 'adaptive':
                residual = g - basis @ (basis.T @ g)
                residual -= basis @ (basis.T @ residual)  # Reorthogonalization.
                if np.linalg.norm(residual) > TAU:
                    basis = np.column_stack((basis, residual/np.linalg.norm(residual)))
                    m += 1
                    max_m = max(max_m, m)
                if m > q:
                    total_deleted += cp.deleted; total_newton += cp.newton_steps
                    total_inserted += cp.inserted
                    q = min(2*q, d)
                    phases.append(q)
                    cp = VolumetricCP(q)
                    records = {}
                    times['basis'] += time.perf_counter()-tick
                    continue
            if basis is None:
                v = g.copy(); eta = 0.
            else:
                coordinates = basis.T @ g
                v = np.pad(coordinates, (0, q-m))
                eta = float(np.linalg.norm(g-basis @ coordinates))
                max_orthogonality = max(max_orthogonality,
                    float(np.linalg.norm(basis.T @ basis-np.eye(m), ord=2)))
            max_eta = max(max_eta, eta)
            times['basis'] += time.perf_counter()-tick
            if np.linalg.norm(v) <= 1e-14:
                certificate = 2*eta+1e-11
                last_point = x
                status = 'zero_slope'
                break
            records[calls] = {'v': v, 'eta': eta, 'birth': m,
                              'scalar': float(g @ x), 'point': y.copy()}
            tick = time.perf_counter()
            cp.cut(v, y, link=calls)
            times['primitive'] += time.perf_counter()-tick
        else:
            status = 'update_cap'
    except Exception:
        status = 'exception'
        error = traceback.format_exc()
    cp_diagnostics()
    result = {
        'd': d, 'k': k, 'arm': arm, 'seed': seed, 'targets': targets,
        'status': status, 'error': error, 'calls': calls, 'hits': hits,
        'last_certificate': certificate, 'last_actual_gap': problem.value(last_point)+.5,
        'solver_seconds': time.perf_counter()-started, 'time_categories': times,
        'm': m, 'q': q, 'phases': phases, 'ball_cuts': ball_cuts,
        'deleted': total_deleted+cp.deleted, 'inserted': total_inserted+cp.inserted,
        'newton_steps': total_newton+cp.newton_steps, 'max_eta': max_eta,
        'max_basis_orthogonality_error': max_orthogonality,
        'max_support_violation': max_support_violation,
        'max_certificate_violation': max_certificate_violation,
        'max_certificate_dual_gap': max_dual_gap,
        'max_constraints_per_q': max_constraints_ratio,
        'max_center_decrement': max_center_decrement,
        'min_insert_barrier_gain': None if not np.isfinite(min_insert_gain) else min_insert_gain,
        'retained_bytes': retained_bytes(),
        'configuration': {'R': 2., 'mu': 1., 'delta': 0., 'tau': TAU,
                          'G': np.sqrt(k)+2, 'deletion_threshold': .04,
                          'center_tolerance': 1e-5, 'certificate_tolerance': 1e-10,
                          'roundoff_allowance': 1e-11, 'max_calls': max_calls, 'seconds': seconds},
    }
    # Retain only the final localizer records for independent replay of the certificate.
    arrays = {'output': last_point, 'basis': np.empty((0, 0)) if basis is None else basis,
              'V': np.array([r['v'] for r in records.values()]),
              'points': np.array([r['point'] for r in records.values()]),
              'eta': np.array([r['eta'] for r in records.values()]),
              'birth': np.array([r['birth'] for r in records.values()]),
              'scalar': np.array([r['scalar'] for r in records.values()]),
              'weights': last_weights}
    return result, trace, arrays


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def worker(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result, trace, arrays = run(args.d, args.k, args.arm, args.eps, args.seconds, args.max_calls)
    write_json(output/'result.json', result)
    (output/'trace.jsonl').write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in trace))
    np.savez_compressed(output/'final_records.npz', **arrays)
    print(json.dumps({key: result[key] for key in ['d', 'k', 'arm', 'status', 'calls', 'solver_seconds',
                                                  'last_certificate', 'last_actual_gap']}), flush=True)
    return 0 if result['status'] == 'certified' else 1


def grid(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    import scipy, matplotlib, clarabel
    write_json(output/'environment.json', {
        'python': sys.version, 'platform': platform.platform(), 'cpu_count': os.cpu_count(),
        'numpy': np.__version__, 'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__,
        'clarabel': clarabel.__version__, 'threadpools': threadpool_info(),
    })
    # Main dimension grid; one tight trace gives all accuracy endpoints at d=256,k=8.
    cells = [(d, k, arm, [1e-4]) for d in (64, 256, 1024) for k in (2, 4, 8, 16)
             for arm in ('adaptive', 'ideal')]
    cells = [(d, k, arm, [1e-2, 1e-3, 1e-4, 1e-5, 1e-6] if (d,k) == (256,8) else eps)
             for d,k,arm,eps in cells]
    # Full-space baseline only at small dimensions; paired adaptive/ideal controls included.
    for d in (16, 32, 64):
        for arm in ('adaptive', 'ideal', 'ambient'):
            if not any((cd, ck, ca) == (d,4,arm) for cd,ck,ca,_ in cells):
                cells.append((d, 4, arm, [1e-4]))
    # Include an ambient accuracy curve at the largest feasible baseline dimension.
    cells = [(d,k,arm,[1e-2,1e-3,1e-4,1e-5,1e-6] if (d,k,arm)==(64,4,'ambient') else eps)
             for d,k,arm,eps in cells]
    write_json(output/'cells.json', [{'d': d, 'k': k, 'arm': arm, 'eps': eps} for d,k,arm,eps in cells])
    failed = []
    for d, k, arm, eps in cells:
        name = f'd{d}_k{k}_{arm}'
        command = [sys.executable, '-m', 'adaptive_cp.runner', 'worker', '--d', str(d),
                   '--k', str(k), '--arm', arm, '--output', str(output/name),
                   '--seconds', str(args.seconds), '--max-calls', str(args.max_calls),
                   '--eps', *map(str, eps)]
        completed = subprocess.run(command, cwd=ROOT.parent, env={**os.environ, 'OPENBLAS_NUM_THREADS':'1',
                                   'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}, capture_output=True, text=True)
        (output/(name+'.stdout')).write_text(completed.stdout)
        (output/(name+'.stderr')).write_text(completed.stderr)
        print(completed.stdout.strip() or f'{name}: worker failed ({completed.returncode})', flush=True)
        if completed.returncode:
            failed.append(name)
    if failed:
        raise RuntimeError('Failed workers: '+', '.join(failed))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['worker', 'grid'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--arm', choices=['adaptive','ideal','ambient'], default='adaptive')
    parser.add_argument('--eps', type=float, nargs='+', default=[1e-4])
    parser.add_argument('--seconds', type=float, default=600.)
    parser.add_argument('--max-calls', type=int, default=12000)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        return worker(args) if args.action == 'worker' else grid(args)


if __name__ == '__main__':
    raise SystemExit(main())
