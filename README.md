# CYTransformer

A transformer that generates **Fine, Regular, Star Triangulations (FRSTs)** of 4‑dimensional
reflexive polytopes — the combinatorial data behind smooth Calabi-Yau threefolds — and verifies
each generated triangulation is a genuine FRST in real time.

Reference implementation for *"Transforming Calabi-Yau Constructions: Generating New Calabi-Yau
Manifolds with Transformers"* ([arXiv:2507.03732](https://arxiv.org/abs/2507.03732)), and the first
software component of **AICY** (https://aicy.physics.wisc.edu).

---

## Install

```bash
git clone https://github.com/jhtyip/cytransformer
cd cytransformer
pip install -e .
```
This gives you three commands: `cyt-prepare`, `cyt-train`, `cyt-infer`.

## Generate FRSTs (with a trained model)

You need a **weights file** (e.g. `model.pt`) and an **input polytope file**. A small example,
`examples/polytopes.json`, ships with the repo.

```bash
cyt-infer --checkpoint model.pt --polys examples/polytopes.json --num-per-poly 100
```
The model generates candidate triangulations for each polytope and verifies each one as a real FRST
on the spot:
```
FRST validation: 87/100 candidates are FRSTs (87.0%)
```
It also saves a boolean `*_is_frst.npy` mask flagging which candidates are genuine FRSTs. The encoding
is read from the checkpoint, so you never specify it. (Add `--no-validate` to skip the FRST check.)

## Train your own model

The repo ships ready‑to‑train datasets of **favorable** reflexive polytopes, already split into
train/val/test:
- **`datasets/9+1/`** — h11 = 5
- **`datasets/10+1/`** — h11 = 6

**1. Train** (a GPU is recommended):
```bash
cyt-train --config configs/train_9+1.yaml      # h11 = 5   (or configs/train_10+1.yaml for h11 = 6)
```
**2. Watch it learn.** The log shows train/val loss and, every monitoring step, the **live FRST
generation rate** — the fraction of the model's generated triangulations that are genuine FRSTs.
Checkpoints are written to `Checkpoints/<folder>/<exp>/chkpt-<step>`.

**3. Use the trained model:**
```bash
cyt-infer --checkpoint Checkpoints/9+1/run/chkpt-399999 --polys examples/polytopes.json
```

Notes:
- The shipped configs are the **paper‑scale** model (~119M params: `d_model=512`, 16 heads, 16 layers,
  ~400k steps) and want a **GPU**. For a quick CPU sanity run, lower `model.d_model`,
  `model.num_layers` and `training.n_steps` in the config.
- To **continue from a checkpoint**, set `job.continued_training: true` (resumes from the latest
  checkpoint in the run folder).
- To train on **your own** data, make split files with `cyt-prepare` and point the `job.*_file_*`
  paths at them.

`cyt-infer` and `cyt-prepare` also accept plain flags instead of a config (flags override the config):
```bash
cyt-prepare --polys my_polys.json --triangs my_triangs.json --n-vertices 9 \
            --n-train 8000 --n-val 1000 --n-test 1000 --out data/
```

## Data generation (optional)

The scripts in `generation/` produce polytope data, and require a **CYTools** install
(https://cytools.liammcallister.com):

```bash
# Generate FRSTs for any Kreuzer-Skarke polytope: fetch polytopes, then run cyt-infer on them.
python generation/fetch_polytopes.py --h11 5 --n 100 --out polys.json
cyt-infer --checkpoint model.pt --polys polys.json

# Build a fresh training dataset.
python generation/make_dataset.py --n_vertices 9 --upper_bound 2000 \
    --folder data_raw --polys_file data_raw/polys.json --triangs_file data_raw/triangs.json
```

## Data format

Plain JSON:
- **Polytopes:** a list of `[POLYID, DRESVERTS]`, where `DRESVERTS` is `"{{x,y,z,w},{...},...}"` of
  resolved vertices.
- **Triangulations:** a list of `[POLYID, TRIANG]`, where `TRIANG` is `"{{i,j,k,l},...}"` — the
  4-simplices of the triangulation, each given as indices into the polytope's vertex list.

## FRST verification

`cytransformer/validation/` verifies a triangulation is a genuine FRST by checking it is **valid** (a
true triangulation of the polytope, via `check_valid`/`pycddlib`) and satisfies the **Fine**,
**Regular**, and **Star** conditions — Regular being the substantive test (`regularity.is_regular`, a
linear program asking whether a height function exists whose lower hull is exactly this triangulation).
These combine in `is_frst`, whose verdicts match CYTools on a labeled set with full agreement (harness
in `dev/frst_verification/`).

## Repository layout

```
cytransformer/        # the model + training/inference + FRST verifier
  models  args  dataset  utilities  inference  train  config  data_prep
  cli/         prepare_data  train  infer
  validation/  frst  regularity  check_valid
datasets/             # ready-to-train data: 9+1 (h11=5), 10+1 (h11=6)
generation/           # optional tools that make polytope data (need CYTools)
configs/  examples/  tests/  dev/
```

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
