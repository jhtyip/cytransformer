# CYTransformer

A transformer that generates **Fine, Regular, Star Triangulations (FRSTs)** of 4‑dimensional
reflexive polytopes — the combinatorial data behind smooth Calabi‑Yau threefolds — and verifies
each generated triangulation is a genuine FRST **in real time, without CYTools**.

Reference implementation for *"Transforming Calabi‑Yau Constructions: Generating New Calabi‑Yau
Manifolds with Transformers"* ([arXiv:2507.03732](https://arxiv.org/abs/2507.03732)), and the first
software component of **AICY** (https://aicy.physics.wisc.edu).

`cyt` is **self‑contained**: it trains on a dataset you give it, or generates FRSTs from polytopes you
give it. It does **not** generate the data — producing datasets / fetching polytopes is an optional
**extra** (the only thing that uses CYTools).

---

## Install

```bash
git clone https://github.com/jhtyip/cytransformer
cd cytransformer
pip install -e .          # core only (PyTorch, scipy, pycddlib, ...). No CYTools needed.
```
This gives you three commands: `cyt-prepare`, `cyt-train`, `cyt-infer`.

## Use case 1 — Generate FRSTs (you have a weights file)

You need: a **weights file** (e.g. `model.pt`) and an **input polytope file**. A small example,
`examples/polytopes.json`, ships with the repo.

```bash
cyt-infer --checkpoint model.pt --polys examples/polytopes.json --num-per-poly 100
```
What happens: the model generates candidate triangulations for each polytope, and **each one is
verified as a real FRST on the spot**, printing e.g.

```
FRST validation: 87/100 candidates are FRSTs (87.0%)
```
plus a saved `*_is_frst.npy` mask flagging which candidates are genuine FRSTs. (Pass `--no-validate`
to skip the check.) The encoding is read from the checkpoint, so you never have to specify it.

## Use case 2 — Train your own model

You need a **dataset**: a polytopes file + a triangulations file (see *Data format* below). A small
example ships in `examples/`.

```bash
# 1. split the dataset into train/val/test
cyt-prepare --polys examples/dataset_polys.json --triangs examples/dataset_triangs.json \
            --n-vertices 9 --n-train 30 --n-val 5 --n-test 5 --out data/

# 2. train (edit configs/train.yaml to point at data/ and pick model size / steps)
cyt-train --config configs/train.yaml
```
During training it prints the **live FRST generation rate** every monitoring step, so you can watch the
model learn to produce valid FRSTs. To continue from existing weights, set `continued_training: true`
in the config with a checkpoint in the run folder. (A paper‑scale run uses `d_model=512`, 16 heads, 16
layers, hundreds of thousands of steps, on a GPU — set `Gpu: true`.)

Both `cyt-infer` and `cyt-prepare` also accept a `--config <file>.yaml` instead of flags; flags
override the config.

## Optional extras (the only part that needs CYTools)

These live in `generation/` and require a separate **CYTools** install
(https://cytools.liammcallister.com). The core above needs none of it.

```bash
# Generate FRSTs for ANY Kreuzer-Skarke polytope, right away:
python generation/fetch_polytopes.py --h11 5 --n 100 --out polys.json
cyt-infer --checkpoint model.pt --polys polys.json

# Build a training dataset from scratch:
python generation/make_dataset.py --n_vertices 9 --upper_bound 2000 \
    --folder data_raw --polys_file data_raw/polys.json --triangs_file data_raw/triangs.json
```

## Data format

Plain JSON the core reads with no CYTools:
- **Polytopes:** a list of `[POLYID, DRESVERTS]`, where `DRESVERTS` is the string
  `"{{x,y,z,w},{...},...}"` of resolved vertices.
- **Triangulations:** a list of `[POLYID, TRIANG]`, where `TRIANG` is `"{{i,j,k,l,m},...}"` of simplex
  vertex indices.

## How FRST verification works (no CYTools)

`cytransformer/validation/` checks **fine + star** (trivial), **valid** tiling (`check_valid`, via
`pycddlib`), and **regular** (`regularity.is_regular`, a small LP — does a height function exist whose
lower hull is this triangulation). Combined in `is_frst`. This was cross‑checked against CYTools on
**6,901 labeled triangulations with 0 false positives and 0 false negatives**; the harness lives in
`dev/frst_verification/`.

## Repository layout

```
cytransformer/        # the self-contained core (no CYTools)
  models  args  dataset  utilities  inference  train  config  data_prep
  cli/         prepare_data  train  infer
  validation/  frst  regularity  check_valid       # CYTools-free FRST verifier
generation/           # OPTIONAL extras (need CYTools): make_dataset, fetch_polytopes
configs/  examples/  tests/  dev/
```

## The frozen model contract

To keep checkpoints loadable across versions, do **not** edit `cytransformer/models.py`, the checkpoint
schema in `train.py`, or the encoding (`args.encoding_parameters`, `dataset.py`, `utilities.py`
translation helpers). `tests/test_checkpoint_contract.py` guards this.

## Citation

```bibtex
@article{Yip:2025hon,
  author  = {Yip, Jacky H. T. and Arnal, Charles and Charton, Fran\c{c}ois and Shiu, Gary},
  title   = {Transforming Calabi-Yau Constructions: Generating New Calabi-Yau Manifolds with Transformers},
  journal = {arXiv e-prints},
  eprint  = {2507.03732},
  year    = {2025},
  doi     = {10.48550/arXiv.2507.03732}
}
```
