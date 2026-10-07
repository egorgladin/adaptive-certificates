"""Historical, value-free ball certificate; weights from a direct SOCP."""
import clarabel
import numpy as np
from scipy import sparse


def historical_norm(v, c, full_rank=False):
    tail = 0. if full_rank else float(c[-1])
    for j in range(len(v)-1, -1, -1):
        tail = float(c[j]) + np.hypot(v[j], tail)
    return float(tail)


def solve_certificate(records, m, d, radius=2.):
    ids = list(records)
    n = len(ids)
    V = np.array([records[i]['v'][:m] for i in ids])
    scalars = np.array([records[i]['scalar'] for i in ids])
    eta = np.array([records[i]['eta'] for i in ids])
    birth = np.array([records[i]['birth'] for i in ids])
    profile = np.zeros((m+1, n))
    profile[birth, np.arange(n)] = eta
    nv = n+m+1
    objective = np.r_[scalars, radius, np.zeros(m)]
    blocks, rhs, cones = [], [], []
    equality = np.zeros((1, nv)); equality[0, :n] = 1.
    blocks.append(equality); rhs.append(np.ones(1)); cones.append(clarabel.ZeroConeT(1))
    nonnegative = np.zeros((n, nv)); nonnegative[:, :n] = -np.eye(n)
    blocks.append(nonnegative); rhs.append(np.zeros(n)); cones.append(clarabel.NonnegativeConeT(n))
    for j in range(m):
        block = np.zeros((3, nv))
        block[0, :n] = profile[j]; block[0, n+j] = -1.
        block[1, :n] = -V[:, j]
        block[2, n+j+1] = -1.
        blocks.append(block); rhs.append(np.zeros(3)); cones.append(clarabel.SecondOrderConeT(3))
    final = np.zeros((1, nv)); final[0, n+m] = -1.
    if m < d:
        final[0, :n] = profile[m]
    blocks.append(final); rhs.append(np.zeros(1)); cones.append(clarabel.NonnegativeConeT(1))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 200
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-10
    solver = clarabel.DefaultSolver(sparse.csc_matrix((nv, nv)), objective,
                                    sparse.csc_matrix(np.vstack(blocks)), np.concatenate(rhs),
                                    cones, settings)
    answer = solver.solve()
    if str(answer.status) not in ('Solved', 'AlmostSolved'):
        raise RuntimeError(f"certificate SOCP failed: {answer.status}")
    alpha = np.maximum(np.asarray(answer.x[:n]), 0.)
    alpha /= alpha.sum()
    v = V.T @ alpha
    c = profile @ alpha
    certificate = float(scalars @ alpha + radius * historical_norm(v, c, full_rank=(m == d)))
    # Budget for roundoff; this is a numerical audit, not interval certification.
    certificate += 1e-11
    dual_gap = float(certificate-answer.obj_val_dual)
    if answer.r_dual > 1e-7 or answer.r_prim > 1e-7:
        raise RuntimeError("certificate SOCP residuals exceed tolerance")
    point = sum(a * records[i]['point'] for a, i in zip(alpha, ids))
    return certificate, point, {
        'dual_gap': dual_gap, 'primal_residual': float(answer.r_prim),
        'dual_residual': float(answer.r_dual), 'status': str(answer.status),
        'max_eta': float(np.max(eta)), 'records': n, 'weights': alpha,
    }
