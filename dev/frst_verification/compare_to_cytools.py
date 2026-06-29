"""
Cross-check the CYTools-free verifier against CYTools ground truth on real
labeled data. RUN THIS ON THE CLUSTER (where both `cytools` and the installed
`cytransformer` package are available).

For each polytope and each candidate token-triangulation it compares:
  - our full FRST verdict  is_frst(...)            vs  CYTools category == "FRST"
  - our regularity sub-check is_regular(...)        vs  CYTools regular (category in {FRST, RST})

The metric that matters most for safety is FALSE POSITIVES: ours says FRST/regular
but CYTools disagrees. Those must be zero before we trust this as a CYTools
replacement for verification.

Usage:
  python compare_to_cytools.py --poly POLY.npy --mask MASK.npy --ts TS.npy \
      --padding_idx 212 [--limit 50]
"""
import argparse
import numpy as np

from cytransformer.validation.frst import is_frst
from cytransformer.validation.check_frst import is_triangulation_FRST

CY_REGULAR = {"FRST", "RST"}


def squeeze_axis1(a):
    return a[:, 0] if a.ndim > 2 and a.shape[1] == 1 else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poly", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--ts", required=True)
    ap.add_argument("--padding_idx", type=int, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    poly = squeeze_axis1(np.load(args.poly)).astype(int)
    mask = squeeze_axis1(np.load(args.mask)).astype(int)
    Ts = np.load(args.ts).astype(int)
    N = poly.shape[0] if args.limit is None else min(args.limit, poly.shape[0])

    frst_fp = frst_fn = 0      # FRST verdict false positives / negatives
    reg_fp = reg_fn = 0        # regularity false positives / negatives
    total = errors = 0
    fp_examples = []

    for i in range(N):
        for k in range(Ts.shape[1]):
            tokens = Ts[i, k]
            cat = is_triangulation_FRST(poly[i], mask[i], tokens, args.padding_idx)
            if cat == "error":
                errors += 1
                continue
            d = is_frst(poly[i], mask[i], tokens, args.padding_idx, return_detail=True)
            total += 1
            cy_frst = (cat == "FRST")
            cy_reg = cat in CY_REGULAR
            if d["frst"] and not cy_frst:
                frst_fp += 1
                if len(fp_examples) < 10:
                    fp_examples.append((i, k, cat, d))
            if (not d["frst"]) and cy_frst:
                frst_fn += 1
            if d["regular"] and not cy_reg:
                reg_fp += 1
            if (not d["regular"]) and cy_reg:
                reg_fn += 1
        print(f"[{i+1}/{N}] compared={total} skipped={errors} | FRST fp={frst_fp} fn={frst_fn} | reg fp={reg_fp} fn={reg_fn}", flush=True)

    print(f"compared: {total}   (CYTools 'error'/invalid skipped: {errors})")
    print(f"FRST   : false positives {frst_fp}  (must be 0) | false negatives {frst_fn}")
    print(f"regular: false positives {reg_fp}  (must be 0) | false negatives {reg_fn}")
    if fp_examples:
        print("  sample FRST false positives (poly_idx, triang_idx, cytools_cat, detail):")
        for e in fp_examples:
            print("   ", e)


if __name__ == "__main__":
    main()
