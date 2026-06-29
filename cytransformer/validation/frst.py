"""
A single CYTools-free FRST verdict, assembled from the four independent checks:

    FRST  ==  fine  AND  star  AND  valid  AND  regular

  - star    : every simplex contains the origin                 (is_star, trivial)
  - fine    : every resolved vertex is used                     (is_fine, trivial)
  - valid   : the simplices tile the polytope and meet only on  (check_valid, pycddlib+scipy)
              shared faces
  - regular : a height function exists (lower-hull liftable)     (is_regular, scipy LP)

No CYTools. The polytope volume needed by the validity check is computed from the
convex hull of {origin} u {resolved vertices} via scipy, so nothing external is
required beyond numpy / scipy / pycddlib.

Inputs match the existing pipeline (cf. check_frst.is_triangulation_FRST):
  poly          : (N_vert, 4) resolved-vertex coordinates (padded rows allowed)
  poly_mask     : (N_vert,)   1 for real vertices
  tokens_triang : the model token sequence for one triangulation
  padding_idx   : the encoding's padding index (128 for 9+1, 212 for 10+1, ...)

NOTE: this verifier is being validated against CYTools (see dev/frst_verification);
until that cross-check passes it is opt-in, not the default.
"""
import numpy as np
from scipy.spatial import ConvexHull

from cytransformer.validation.regularity import is_regular, is_star, is_fine
from cytransformer.utilities import tokens_triang_to_vert_indices_wo_triang
from cytransformer.validation.check_valid import check_valid_triangulation


def is_frst(poly, poly_mask, tokens_triang, padding_idx, return_detail=False):
    num_ver = int(np.sum(poly_mask))
    verts = np.asarray(poly, dtype=float)[:num_ver]
    points = np.vstack([np.zeros((1, verts.shape[1])), verts])  # index 0 = origin

    simplices = [tuple(int(i) for i in s)
                 for s in tokens_triang_to_vert_indices_wo_triang(tokens_triang, num_ver, padding_idx)]

    def result(fine, star, valid, regular):
        frst = bool(fine and star and valid and regular)
        if return_detail:
            return {"frst": frst, "fine": fine, "star": star, "valid": valid, "regular": regular}
        return frst

    if not simplices:
        return result(False, False, False, False)

    star = is_star(len(points), simplices, origin_index=0)
    fine = is_fine(len(points), simplices)

    try:
        total_vol = ConvexHull(points).volume
        coord_simplices = [points[list(s)] for s in simplices]
        valid = check_valid_triangulation(coord_simplices, total_vol)
    except Exception:
        valid = False

    regular = is_regular(points, simplices)
    return result(fine, star, valid, regular)


def frst_rate(polys, poly_masks, token_triangs, padding_idx):
    """Count FRSTs among generated candidates.

    polys: (P, N_vert, 4), poly_masks: (P, N_vert), token_triangs: (P, T, L).
    Returns (n_frst, n_total, rate).
    """
    n_frst = n_total = 0
    for i in range(len(polys)):
        for k in range(token_triangs.shape[1]):
            n_total += 1
            if is_frst(polys[i], poly_masks[i], token_triangs[i, k], padding_idx):
                n_frst += 1
    return n_frst, n_total, (n_frst / n_total if n_total else 0.0)
