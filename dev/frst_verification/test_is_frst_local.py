"""
Positive check of the assembled is_frst() on REAL FRSTs.

The processed training data are genuine FRSTs produced by CYTools, so a correct
CYTools-free verifier must classify every one of them as FRST. Run from the repo
root:  python dev/frst_verification/test_is_frst_local.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frst import is_frst
from cytransformer.dataset import Polys_triangs_dataset
from cytransformer.args import encoding_parameters

polys = json.load(open("Data/9+1/smoke/test_polys.json"))
triangs = json.load(open("Data/9+1/smoke/test_triangs.json"))
ep = encoding_parameters(9)
ds = Polys_triangs_dataset(polys, triangs, ep.max_seq_length_src, ep.max_seq_length_tgt,
                           permutation=False, triang_shuffling=False, seed=0)

n = min(20, len(ds))
bad = 0
for idx in range(n):
    poly_t, tokens_t, mask_t = ds[idx]
    d = is_frst(poly_t.numpy(), mask_t.numpy(), tokens_t.numpy(), ep.padding_idx, return_detail=True)
    if not d["frst"]:
        bad += 1
        print(f"  idx {idx}: NOT classified FRST -> {d}")

print(f"\n{n} genuine FRSTs checked | verified as FRST by the CYTools-free checker: {n - bad}/{n}")
print("RESULT:", "PASS" if bad == 0 else "FAIL")
