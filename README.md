# CYTransformer

A transformer that generates **Fine, Regular, Star Triangulations (FRSTs)** of 4‑dimensional
reflexive polytopes — the combinatorial data behind smooth Calabi‑Yau threefolds in toric
varieties. CYTransformer learns to sample FRSTs efficiently and *representatively* (unbiasedly)
across polytope sizes, and can self‑improve by retraining on its own validated output.

This is the reference implementation for the paper
**“Transforming Calabi‑Yau Constructions: Generating New Calabi‑Yau Manifolds with Transformers”**
([arXiv:2507.03732](https://arxiv.org/abs/2507.03732)), and the first software component of
**AICY** — *AI‑enabled living Calabi‑Yau repositories* (https://aicy.physics.wisc.edu).

---

## What it does

- **Encoder–decoder transformer**: the encoder reads a polytope (its resolved vertices as 4D
  integer vectors); the decoder autoregressively emits a triangulation as a sequence of simplex
  tokens.
- Trained with cross‑entropy on FRSTs, with vertex‑permutation and simplex‑shuffling augmentation.
- Generates many distinct candidate triangulations per polytope; candidates can optionally be
  verified as genuine FRSTs.

Polytope configurations are labelled by `(h^{1,1}, N_vert)` — e.g. `(5, 9+1)` means a polytope with
9 non‑origin resolved vertices (plus the origin).

## Install

```bash
pip install -e .          # installs the package + the cyt-prepare / cyt-train / cyt-infer commands
# or, without installing:  pip install -r requirements.txt
```

That is enough to **train** and **generate** — no CYTools needed. CYTools (and `pycddlib`) are
**optional**, required only to (a) generate brand‑new triangulation data from the Kreuzer–Skarke
database, or (b) *validate* that generated candidates are genuine FRSTs:

```bash
pip install -e ".[validation]"   # plus a CYTools install: https://cytools.liammcallister.com
```

> Tip: pin `torch` to a build matching your platform/CUDA. CPU works out of the box; for GPU use the
> appropriate CUDA wheel.

## Quickstart

The three steps below run on **CPU, without CYTools**, using the small example configs in `configs/`.

```bash
# 1. Prepare data: split a raw (polytopes, triangulations) pair into aligned train/val/test sets
cyt-prepare --config configs/prepare_9+1.yaml

# 2. Train
cyt-train --config configs/train_9+1.yaml

# 3. Generate FRST candidates from the trained checkpoint
cyt-infer --config configs/infer_9+1.yaml
```

(Didn't `pip install`? Use the module form, e.g. `python -m cytransformer.cli.train --config configs/train_9+1.yaml`.)

The example configs are sized for a quick smoke test. For the **paper‑scale model**, use
`d_model=512, num_heads=16, num_layers=16`, the `exponential` scheduler, `n_steps` in the hundreds of
thousands, a full data split, and a GPU (`job.Gpu: true`).

### Data formats
- **Raw** files (e.g. `9+1_polys_0_4999.json`, `9+1_triangs_0_4999.json`) are lists of records keyed
  by `POLYID` (polytopes carry `DRESVERTS`, triangulations carry `TRIANG`).
- `cyt.prepare_data` pairs each triangulation with its polytope and writes **processed**
  `[POLYID, DRESVERTS]` / `[POLYID, TRIANG]` lists, aligned 1:1, split by polytope group.
- **Training and inference consume the processed format** (not raw). Point `infer`’s `polys_file` at a
  processed polytopes file.

## Using a pretrained checkpoint

Drop a checkpoint at the path your inference config points to and run `cyt.infer`. The encoding
(`n_vertices`, vocabulary, sequence lengths) is read **from the checkpoint**, so it always matches how
the model was trained — do not override it.

## Configuration

Configs are YAML and map directly onto the model/training/job parameters (see `config.py`). A training
config has `model:`, `training:`, and `job:` sections; an inference config is a flat file pointing at a
`checkpoint_path` and a `polys_file`. See the files in `configs/` for annotated examples.

## Running on a GPU / RunPod

Set `job.Gpu: true` in the training config. With one GPU it trains single‑process; with several
visible GPUs it uses PyTorch DistributedDataParallel automatically.

## The frozen model contract

To keep checkpoints interchangeable across versions (and to load original weights), the following are a
**frozen contract — do not edit**:
- `cytransformer/models.py` (architecture / `state_dict` keys),
- the checkpoint dict schema saved in `cytransformer/train.py` and read by
  `cytransformer.args.model_params_from_checkpoint` / `encoding_params_from_checkpoint`,
- the encoding/tokenization (`cytransformer.args.encoding_parameters`, the simplex vocabulary in
  `cytransformer/dataset.py`, and the translation helpers in `cytransformer/utilities.py`).

`tests/test_checkpoint_contract.py` guards this: it proves a checkpoint round‑trips through the loader
bit‑for‑bit (run it after any change). Point it at a real checkpoint to verify that file loads:

```bash
python tests/test_checkpoint_contract.py [path/to/checkpoint]
```

## Repository layout

```
cytransformer/               # the package
├── models.py args.py dataset.py utilities.py   # frozen core (architecture + encoding)
├── inference.py train.py data_generation.py    # core training / generation
├── config.py                                   # YAML -> params loader
├── cli/                                         # one-command entrypoints: prepare_data, train, infer
└── validation/                                 # optional, needs CYTools: check_frst, check_valid,
                                                 #   monitoring, generate_triangs_from_checkpoint, rl
configs/                     # example YAML configs
tests/                       # checkpoint-contract safety net
pyproject.toml  LICENSE  README.md  requirements.txt
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
