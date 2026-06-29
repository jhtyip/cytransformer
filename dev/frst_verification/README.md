# FRST verification without CYTools — prototype

Goal: verify a triangulation is an **FRST** (Fine, Regular, Star) using only
`numpy`/`scipy` (+ `pycddlib` for the validity check), so a user can check
generated outputs **without installing CYTools**. CYTools would then be needed
only for *data generation*, not verification.

An FRST check decomposes into:
- **Star** — every simplex contains the origin (`is_star`, trivial).
- **Fine** — every resolved vertex is used (`is_fine`, trivial).
- **Valid** — the simplices actually tile the polytope, meeting only on shared
  faces. Already CYTools-free in `cytransformer/validation/check_valid.py`
  (`pycddlib` + `scipy`).
- **Regular** — a height function exists whose lower hull projects to exactly
  this triangulation. This is the only non-trivial piece: `regularity.is_regular`
  (an LP — see the module docstring).

## Status
- `is_regular` is correct-by-derivation and **locally validated for no false
  negatives**: across 120 random regular triangulations in 2D/3D/4D (generated as
  lower hulls of random liftings), it accepted every one. Run `test_regularity_local.py`.
- **Not yet validated against CYTools ground truth** (the false-positive
  direction: correctly *rejecting* genuinely non-regular triangulations). That is
  what `compare_to_cytools.py` is for — run it on the cluster.
- `frst.py` assembles the four checks into a single **`is_frst()`** (no CYTools;
  needs `scipy` + `pycddlib`). Locally it classifies **20/20 real FRSTs** as FRST
  and rejects 15/15 random-garbage and a corrupted FRST (`test_is_frst_local.py`).
  Still pending the CYTools ground-truth cross-check for the subtle false-positive
  direction (valid+fine+star but non-regular triangulations).

## How to validate (on the cluster, where CYTools is installed)
```bash
pip install -e .            # the cytransformer package
python dev/frst_verification/compare_to_cytools.py \
    --poly  /path/toSave_poly_fast_200.npy \
    --mask  /path/toSave_poly_mask_fast_200.npy \
    --ts    /path/toSave_Ts_fast_200.npy \
    --padding_idx 212        # 128 for 9+1, 212 for 10+1, ...
```
It prints agree/disagree counts and, crucially, the number of **false positives**
(ours says regular, CYTools says not) — which must be **0** before we trust this
as a CYTools-free verifier.

The verifier now lives in the package: `cytransformer/validation/regularity.py`
and `cytransformer/validation/frst.py` (`is_frst`). It is wired in **behind opt-in
flags** (off by default):
- training: `training.validate_frst_during_training: true` -> logs a live FRST rate
  in the monitoring block;
- inference: `validate: true` -> flags/saves which candidates are FRSTs.

Once `compare_to_cytools.py` confirms zero false positives against CYTools, we can
make it the default verifier and drop CYTools from the verification path entirely.
