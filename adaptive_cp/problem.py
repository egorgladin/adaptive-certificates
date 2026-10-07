import numpy as np


class Problem:
    """Only this oracle owns U. Solvers receive ambient gradients."""

    def __init__(self, d, k, seed=0):
        self.d, self.k = d, k
        # Prefixes in k share an embedding at a given d.
        gaussian = np.random.default_rng(seed).standard_normal((d, min(d, 16)))
        basis, triangular = np.linalg.qr(gaussian)
        basis *= np.where(np.diag(triangular) >= 0, 1., -1.)
        self.U = np.ascontiguousarray(basis[:, :k])

    def gradient(self, x):
        z = self.U.T @ x
        index = int(np.argmax(z))
        return np.sqrt(self.k) * self.U[:, index] + x

    def value(self, x):
        return float(np.sqrt(self.k) * np.max(self.U.T @ x) + .5 * (x @ x))

    def solution(self):
        return -self.U.sum(axis=1) / np.sqrt(self.k)
