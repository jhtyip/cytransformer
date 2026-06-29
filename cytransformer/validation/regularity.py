"""
CYTools-free geometric checks for a triangulation given as index simplices over a
point set. The crux is `is_regular` (an LP); `is_fine`/`is_star` are trivial.

A triangulation T of a point set is REGULAR iff there is a height vector w such
that lifting each point p_i to (p_i, w_i) and taking the lower convex hull
projects back to exactly T. Equivalently, for every simplex sigma in T and every
point j not in sigma, point j must lift strictly ABOVE sigma's hyperplane:

        w_j  -  sum_{i in sigma} lambda_i * w_i   >  0,

where (lambda_i) are the barycentric coordinates of p_j with respect to sigma
(p_j = sum lambda_i p_i, sum lambda_i = 1). Affine height functions make every
such expression exactly 0, so they are always feasible at the boundary; T is
regular iff some w pushes all of them strictly positive. We test that by
maximizing a common margin eps (with w bounded to keep it finite): regular iff
the optimal eps > 0.
"""
import numpy as np
from scipy.optimize import linprog


def is_regular(points, simplices, tol=1e-7):
    points = np.asarray(points, dtype=float)
    n, d = points.shape
    simplices = [tuple(int(i) for i in s) for s in simplices]

    A_ub, b_ub = [], []
    for sigma in simplices:
        sigma_set = set(sigma)
        # Affine system to read off barycentric coords: [P; 1] @ lambda = [p_j; 1].
        M = np.vstack([points[list(sigma)].T, np.ones(len(sigma))])  # (d+1, d+1)
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return False  # degenerate simplex -> not a valid full-dim triangulation
        for j in range(n):
            if j in sigma_set:
                continue
            lam = Minv @ np.append(points[j], 1.0)
            # eps - w_j + sum_i lam_i w_{sigma_i} <= 0
            row = np.zeros(n + 1)
            row[n] = 1.0
            row[j] += -1.0
            for k, idx in enumerate(sigma):
                row[idx] += lam[k]
            A_ub.append(row)
            b_ub.append(0.0)

    if not A_ub:
        return True  # single simplex, nothing to violate

    c = np.zeros(n + 1)
    c[n] = -1.0  # maximize eps
    bounds = [(-1.0, 1.0)] * n + [(None, None)]
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")
    if not res.success:
        return False
    return bool(res.x[n] > tol)


def is_star(n_points, simplices, origin_index=0):
    """Every simplex must contain the origin."""
    return all(origin_index in set(int(i) for i in s) for s in simplices)


def is_fine(n_points, simplices):
    """Every point must be used by at least one simplex."""
    used = set()
    for s in simplices:
        used.update(int(i) for i in s)
    return used == set(range(n_points))
