"""
Local correctness checks for is_regular, using cases with known answers.
(No CYTools needed; the authoritative cross-check vs CYTools runs on the cluster.)

Positive oracle: every Delaunay triangulation is regular (it is literally the
lower convex hull of the points lifted to the paraboloid z = |x|^2). We assert
is_regular == True on Delaunay triangulations in 2D/3D/4D.

Negative case: the classic "pinwheel" non-regular triangulation of a triangle
with three cyclically-offset edge points.
"""
import numpy as np
from scipy.spatial import Delaunay
from regularity import is_regular

rng = np.random.default_rng(0)


def _tri_area(p, a, b, c):
    return 0.5 * abs(np.cross(p[b] - p[a], p[c] - p[a]))


def check_single_simplex():
    pts = np.array([[0, 0], [1, 0], [0, 1]], float)
    assert is_regular(pts, [(0, 1, 2)]) is True
    print("single simplex (2D)            -> regular  OK")


def check_delaunay(dim, npts, trials=3):
    for _ in range(trials):
        pts = rng.random((npts, dim))
        tri = Delaunay(pts)
        assert is_regular(pts, [tuple(s) for s in tri.simplices]) is True, f"Delaunay {dim}D classified non-regular!"
    print(f"Delaunay {dim}D ({npts} pts, {trials}x)    -> regular  OK")


def check_pinwheel():
    # Outer triangle A,B,C plus one point on each edge, offset cyclically (t=0.7).
    A, B, C = np.array([0., 0.]), np.array([6., 0.]), np.array([3., 6.])
    t = 0.7
    P = A + t * (B - A)   # on AB
    Q = B + t * (C - B)   # on BC
    R = C + t * (A - C)   # on CA
    pts = np.array([A, B, C, P, Q, R])
    # pinwheel triangulation: central PQR + three corner triangles
    simplices = [(3, 4, 5), (0, 3, 5), (1, 4, 3), (2, 5, 4)]
    area = sum(_tri_area(pts, *s) for s in simplices)
    outer = _tri_area(pts, 0, 1, 2)
    tiles = np.isclose(area, outer)
    reg = is_regular(pts, simplices)
    print(f"pinwheel: tiles(area)={tiles}  -> is_regular={reg}  (expected non-regular)")
    return tiles, reg


if __name__ == "__main__":
    check_single_simplex()
    check_delaunay(2, 10)
    check_delaunay(3, 10)
    check_delaunay(4, 12)   # <- the dimension that matters for Calabi-Yau polytopes
    check_pinwheel()
    print("\nPositive (Delaunay) checks PASSED.")
