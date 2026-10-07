"""Central-cut Vaidya primitive, with minimum-leverage deletion at 0.04.

The exact-center update rule is Algorithm 1 / Lemma 4.1 in
https://arxiv.org/abs/2306.10096 (central-cut Anstreicher variant).
We use floating-point damped Newton centering, not the paper's bit rounding.
All three experimental arms call this same class.
"""
import numpy as np
from scipy.linalg import qr, solve_triangular, cho_factor, cho_solve


class CenteringError(RuntimeError):
    pass


class VolumetricCP:
    def __init__(self, q, radius=2., deletion_threshold=.04, center_tolerance=1e-5):
        self.q, self.radius = q, radius
        self.deletion_threshold = deletion_threshold
        self.center_tolerance = center_tolerance
        self.A = np.concatenate((np.eye(q), -np.eye(q)))
        self.b = np.full(2*q, radius)
        self.links = [None] * (2*q)
        self.y = np.zeros(q)
        self.newton_steps = self.deleted = self.inserted = 0
        self.max_constraints = 2*q
        self.max_center_decrement = 0.
        self.min_insert_barrier_gain = np.inf

    def geometry(self, y, derivatives=True):
        slack = self.b - self.A @ y
        if np.min(slack) <= 0:
            raise CenteringError("point is outside the strict localizer interior")
        scaled = self.A / slack[:, None]
        # QR avoids squaring the condition number of a thin localizer.
        W, triangular = qr(scaled, mode='economic', check_finite=False)
        value = float(np.log(np.abs(np.diag(triangular))).sum())
        if not derivatives:
            return value
        leverage = np.einsum('ij,ij->i', W, W)
        gram = W @ W.T
        grad = W.T @ leverage
        hessian = 3 * W.T @ (leverage[:, None] * W) - 2 * W.T @ ((gram*gram) @ W)
        hessian = .5 * (hessian + hessian.T)
        step_white = cho_solve(cho_factor(hessian, lower=True, check_finite=False), grad,
                               check_finite=False)
        step = solve_triangular(triangular, step_white, lower=False, check_finite=False)
        decrement2 = float(grad @ step_white)
        return value, leverage, step, max(0., decrement2), triangular

    def center(self):
        for iteration in range(100):
            value, leverage, step, decrement2, _ = self.geometry(self.y)
            if decrement2 <= self.center_tolerance**2:
                self.max_center_decrement = max(self.max_center_decrement, np.sqrt(decrement2))
                return self.y.copy(), leverage, value
            slack = self.b - self.A @ self.y
            change = -self.A @ step
            adverse = change > 0
            length = min(1., .99 * np.min(slack[adverse]/change[adverse])) if np.any(adverse) else 1.
            accepted = False
            for _ in range(40):
                trial = self.y - length * step
                try:
                    trial_value = self.geometry(trial, derivatives=False)
                except (CenteringError, np.linalg.LinAlgError):
                    trial_value = np.inf
                if trial_value <= value - .01 * length * decrement2:
                    accepted = True
                    break
                # Close to the center, log-det comparisons lose relative accuracy.
                if decrement2 < 1e-8 and trial_value <= value + 1e-11*(1+abs(value)):
                    _, _, _, trial_decrement2, _ = self.geometry(trial)
                    if trial_decrement2 < .9*decrement2 or trial_decrement2 <= self.center_tolerance**2:
                        accepted = True
                        break
                length *= .5
            if not accepted:
                raise CenteringError(f"Newton line search failed: decrement={np.sqrt(decrement2):g}")
            self.y = trial
            self.newton_steps += 1
        raise CenteringError(f"Newton centering exceeded 100 iterations; decrement={np.sqrt(decrement2):g}; step={np.linalg.norm(step):g}")

    def prune(self):
        """Delete one minimum-leverage constraint, then recenter; repeat."""
        removed = []
        while True:
            y, leverage, _ = self.center()
            index = int(np.argmin(leverage))
            if leverage[index] >= self.deletion_threshold:
                return y, removed
            if self.links[index] is not None:
                removed.append(self.links[index])
            self.A = np.delete(self.A, index, axis=0)
            self.b = np.delete(self.b, index)
            self.links.pop(index)
            self.deleted += 1

    def cut(self, normal, at, link=None):
        """Insert normal @ (z-at) <= 0, recovering a strict feasible start."""
        normal = np.asarray(normal, dtype=float)
        norm = np.linalg.norm(normal)
        if norm == 0:
            raise ValueError("a zero cut must be handled by the wrapper")
        normal = normal / norm
        old_value, _, _, _, triangular = self.geometry(self.y)
        # Start strictly inside the new halfspace, preserving old strict slacks.
        direction = -solve_triangular(triangular,
            solve_triangular(triangular.T, normal, lower=True, check_finite=False),
            lower=False, check_finite=False)
        slack = self.b - self.A @ self.y
        change = self.A @ direction
        adverse = change > 0
        length = .5 * np.min(slack[adverse]/change[adverse]) if np.any(adverse) else 1.
        start = self.y + length * direction
        rhs = float(normal @ at)
        if normal @ start >= rhs:
            raise CenteringError("could not enter central cut strictly")
        self.A = np.vstack((self.A, normal))
        self.b = np.append(self.b, rhs)
        self.links.append(link)
        self.y = start
        self.inserted += 1
        self.max_constraints = max(self.max_constraints, len(self.b))
        _, _, new_value = self.center()
        self.min_insert_barrier_gain = min(self.min_insert_barrier_gain, new_value-old_value)

    def state_bytes(self):
        return self.A.nbytes + self.b.nbytes + self.y.nbytes
