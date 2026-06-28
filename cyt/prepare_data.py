"""
One-command data preparation: split a raw (polytopes, triangulations) pair of
JSON files into aligned train / val / test files that the training loader expects.

    python -m cyt.prepare_data --config configs/prepare_9+1.yaml

Pairs each triangulation with its polytope (by POLYID), keeps only polytopes with
the expected number of resolved vertices, then slices polytope GROUPS into
train/val/test by index ranges. CYTools is NOT required (we split existing data;
we do not generate new triangulations here).
"""
import argparse
import os

import cyt  # noqa: F401  (side effect: puts repo root on sys.path)
import yaml

from data_generation import split_data_into_three_files


def main():
    ap = argparse.ArgumentParser(description="Split raw polytope/triangulation JSON into train/val/test.")
    ap.add_argument("--config", required=True, help="Path to a prepare YAML config.")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    j = lambda name: os.path.join(out_dir, name)

    split_data_into_three_files(
        cfg["polys_file"], cfg["triangs_file"],
        j("train_polys.json"), j("train_triangs.json"),
        j("val_polys.json"), j("val_triangs.json"),
        j("test_polys.json"), j("test_triangs.json"),
        cfg["n_vertices"],
        cfg["n_train"], cfg["n_val"], cfg["n_test"],
        randomize=bool(cfg.get("randomize", True)),
        seed=int(cfg.get("seed", 0)),
    )


if __name__ == "__main__":
    main()
