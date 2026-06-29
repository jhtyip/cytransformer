"""
Cross-check the CYTools-free `is_regular` against CYTools ground truth on real
labeled data. RUN THIS ON THE CLUSTER (where both `cytools` and the installed
`cytransformer` package are available).

For each polytope and each candidate token-triangulation it:
  - builds our (points, simplices) and computes is_regular / is_star / is_fine;
  - asks CYTools (via cytransformer.validation.check_frst.is_triangulation_FRST)
    for the ground-truth category in {FRST, FST, RST, ST, error};
  - compares the REGULAR verdict (CYTools: regular iff category in {FRST, RST}).

The metric that matters most for safety is FALSE POSITIVES: ours says regular but
CYTools says not-regular (and not error). Those must be zero.

Usage:
  python compare_to_cytools.py --poly POLY.npy --mask MASK.npy --ts TS.npy \
      --padding_idx 212 [--limit 50]
(The check_FRST.py __main__ historically used Data/Test_FRST_check/toSave_poly_fast_200.npy,
 toSave_poly_mask_fast_200.npy, toSave_Ts_fast_200.npy with padding_idx=212 for 10+1.)
"""
import argparse
import numpy as np

from regularity import is_regular, is_star, is_fine
from cytransformer.utilities import tokens_triang_to_vert_indices_wo_triang
from cytransformer.validation.check_frst import is_triangulation_FRST

CY_REGULAR = {"FRST", "RST"}
CY_FINE = {"FRST", "FST"}


def squeeze_axis1(a):
    # accept (N, 1, ...) shapes used by some saved files
    return a[:, 0] if a.ndim >= 2 and a.shape[1] == 1 and a.ndim > 2 else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poly", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--ts", required=True)
    ap.add_argument("--padding_idx", type=int, required=True)
    ap.add_argument("--limit", type=int, default=None, help="max polytopes to scan")
    args = ap.parse_args()

    poly = squeeze_axis1(np.load(args.poly)).astype(int)      # (N, N_vert, 4)
    mask = squeeze_axis1(np.load(args.mask)).astype(int)      # (N, N_vert)
    Ts = np.load(args.ts).astype(int)                         # (N, M, L)
    N = poly.shape[0] if args.limit is None else min(args.limit, poly.shape[0])

    agree = disagree = 0
    false_pos = []   # ours regular, CYTools not-regular (the dangerous error)
    false_neg = []   # ours not-regular, CYTools regular
    errors = 0
    total = 0

    for i in range(N):
        numOfVer = int(mask[i].sum())
        points = np.vstack([[0, 0, 0, 0], poly[i][:numOfVer]]).astype(float)  # index 0 = origin
        for k in range(Ts.shape[1]):
            tokens = Ts[i, k]
            cat = is_triangulation_FRST(poly[i], mask[i], tokens, args.padding_idx)
            if cat == "error":
                errors += 1
                continue
            simplices = tokens_triang_to_vert_indices_wo_triang(tokens, numOfVer, args.padding_idx)
            ours_reg = is_regular(points, simplices)
            cy_reg = cat in CY_REGULAR
            total += 1
            if ours_reg == cy_reg:
                agree += 1
            else:
                disagree += 1
                (false_pos if ours_reg and not cy_reg else false_neg).append((i, k, cat))

    print(f"compared: {total}   (CYTools 'error'/invalid skipped: {errors})")
    print(f"agree: {agree}   disagree: {disagree}")
    print(f"FALSE POSITIVES (ours regular, CYTools not): {len(false_pos)}   <-- must be 0")
    print(f"false negatives (ours not-regular, CYTools regular): {len(false_neg)}")
    if false_pos[:10]:
        print("  sample false positives (poly_idx, triang_idx, cytools_cat):", false_pos[:10])
    if false_neg[:10]:
        print("  sample false negatives:", false_neg[:10])


if __name__ == "__main__":
    main()
