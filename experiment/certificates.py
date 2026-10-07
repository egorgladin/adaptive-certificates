"""Full QP and historical-residual SOCP certificates."""
import numpy as np
import torch
import clarabel
from scipy import sparse


def settings(accuracy=1e-9, seconds=60):
    s = clarabel.DefaultSettings()
    s.verbose = False
    s.max_threads = 1
    s.max_iter = 300
    s.time_limit = seconds
    s.tol_gap_abs = s.tol_gap_rel = s.tol_feas = accuracy
    return s


def simplex(w):
    w = np.maximum(np.asarray(w, dtype=np.float64), 0)
    if not np.all(np.isfinite(w)) or w.sum() <= 0:
        raise ArithmeticError("no feasible recovered weights")
    return w / w.sum()


def full_qp(G, b, lam):
    n = len(b)
    P = sparse.triu(sparse.csc_matrix(G / lam), format="csc")
    A = sparse.vstack((sparse.csc_matrix(np.ones((1, n))), -sparse.eye(n)), format="csc")
    solver = clarabel.DefaultSolver(P, -np.asarray(b), A, np.r_[1., np.zeros(n)],
                                   [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(n)], settings())
    result = solver.solve()
    w = simplex(result.x)
    grad = G @ w / lam - b
    return w, {"status": str(result.status), "iterations": result.iterations,
               "simplex_optimality_gap": float(w @ grad - grad.min()),
               "master_variables": n, "working_set": False}


def historical(v, c):
    u = float(c[-1])
    for j in range(len(v)-1, -1, -1):
        u = c[j] + np.hypot(v[j], u)
    return float(u)


def compressed_socp(V, b, eta, birth, lam):
    n, k = V.shape
    dim = n+k+1
    scale = np.sqrt(lam)
    P = sparse.csc_matrix(([1.], ([n], [n])), shape=(dim, dim))
    q = np.r_[-np.asarray(b), np.zeros(k+1)]
    blocks = [sparse.csc_matrix(np.r_[np.ones(n), np.zeros(k+1)][None, :]),
              -sparse.eye(dim, format="csc")]
    rhs = [np.ones(1), np.zeros(dim)]
    cones = [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(dim)]
    row = np.r_[eta*(birth == k)/scale, np.zeros(k+1)]
    row[n+k] = -1
    blocks.append(sparse.csc_matrix(row[None, :]))
    rhs.append(np.zeros(1))
    cones.append(clarabel.NonnegativeConeT(1))
    for j in range(k):
        block = np.zeros((3, dim))
        block[0, :n] = eta*(birth == j)/scale
        block[0, n+j] = -1
        block[1, :n] = -V[:, j]/scale
        block[2, n+j+1] = -1
        blocks.append(sparse.csc_matrix(block))
        rhs.append(np.zeros(3))
        cones.append(clarabel.SecondOrderConeT(3))
    solver = clarabel.DefaultSolver(P, q, sparse.vstack(blocks, format="csc"),
                                   np.concatenate(rhs), cones, settings())
    result = solver.solve()
    return simplex(result.x[:n]), {"status": str(result.status), "iterations": result.iterations,
                                  "master_variables": n, "working_set": False}


class FullArchive:
    def __init__(self, d, records, lam):
        self.d, self.lam = d, lam
        self.G = np.zeros((records, records))
        self.blocks, self.b, self.weights = [], [], np.empty(0)
        self.block_size = 32
        self.staging = torch.empty(d, dtype=torch.float64, pin_memory=True)
        self.wire_bytes = 0
        self.copies = 0

    def insert(self, a, b):
        self.staging.copy_(a, non_blocking=False)
        host = self.staging.numpy()
        n = len(self.b)
        for j, block in enumerate(self.blocks):
            start = j*self.block_size
            count = min(self.block_size, n-start)
            if count:
                row = block[:count] @ host
                self.G[n, start:start+count] = row
                self.G[start:start+count, n] = row
        self.G[n, n] = host @ host
        if n % self.block_size == 0:
            self.blocks.append(np.empty((self.block_size, self.d)))
        self.blocks[-1][n % self.block_size] = host
        self.b.append(b)
        self.wire_bytes += self.d*8
        self.copies += 1

    def evaluate(self, w):
        n = len(self.b)
        return float(np.asarray(self.b) @ w - w @ self.G[:n, :n] @ w/(2*self.lam))

    def solve(self):
        n = len(self.b)
        w, info = full_qp(self.G[:n, :n], np.asarray(self.b), self.lam)
        bound = self.evaluate(w)
        if len(self.weights):
            prev = np.pad(self.weights, (0, n-len(self.weights)))
            if self.evaluate(prev) > bound:
                w, bound = prev, self.evaluate(prev)
        self.weights = w
        return bound, info

    def direct_aggregate(self):
        n = len(self.b)
        a = np.zeros(self.d)
        for j, block in enumerate(self.blocks):
            start = j*self.block_size
            count = min(self.block_size, n-start)
            a += self.weights[start:start+count] @ block[:count]
        return a

    def numeric_bytes(self):
        return sum(b.nbytes for b in self.blocks)+self.G.nbytes+len(self.b)*8+self.weights.nbytes+self.d*8


class CompressedArchive:
    def __init__(self, d, records, lam, tau, capacity, budget):
        self.d, self.lam, self.tau = d, lam, tau
        self.capacity, self.budget, self.k = capacity, budget, 0
        self.Q = torch.empty((capacity, d), dtype=torch.float64, device="cuda")
        self.V = np.zeros((records, capacity))
        self.b, self.eta, self.birth, self.weights = [], [], [], np.empty(0)
        self.wire_bytes, self.copies = 0, 0
        self.peak_related = self.Q.numel()*8
        self.staging = torch.empty(capacity+1, dtype=torch.float64, pin_memory=True)


    def evaluate(self, w):
        n = len(self.b)
        v = w @ self.V[:n, :self.k]
        c = np.bincount(self.birth, weights=w*np.asarray(self.eta), minlength=self.k+1)
        return float(np.asarray(self.b) @ w-historical(v, c)**2/(2*self.lam))

    def solve(self):
        n = len(self.b)
        w, info = compressed_socp(self.V[:n, :self.k], self.b, np.asarray(self.eta),
                                  np.asarray(self.birth), self.lam)
        bound = self.evaluate(w)
        if len(self.weights):
            prev = np.pad(self.weights, (0, n-len(self.weights)))
            if self.evaluate(prev) > bound:
                w, bound = prev, self.evaluate(prev)
        self.weights = w
        return bound, info

    def numeric_bytes(self):
        return self.Q.numel()*8+self.V.nbytes+len(self.b)*24+self.weights.nbytes+(self.capacity+1)*8
