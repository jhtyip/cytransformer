"""
Prepare a dataset: pair a (polytopes, triangulations) JSON pair by POLYID and
split it into aligned train / val / test files the trainer expects.

  Simple:  cyt-prepare --polys polys.json --triangs triangs.json --n-vertices 9 --out data/
  Config:  cyt-prepare --config configs/prepare_9+1.yaml

No CYTools: this only reshapes JSON you already have (produced by the generation
extra, or your own).
"""
import argparse
import os

import yaml

from cytransformer.data_prep import split_data_into_three_files


def main():
    ap = argparse.ArgumentParser(description="Split a polytope/triangulation dataset into train/val/test.")
    ap.add_argument("--config", help="Optional YAML config; the flags below override it.")
    ap.add_argument("--polys", help="Input polytopes JSON.")
    ap.add_argument("--triangs", help="Input triangulations JSON.")
    ap.add_argument("--out", help="Output directory.")
    ap.add_argument("--n-vertices", type=int, help="N_vert-1 (e.g. 9 for the 9+1 family).")
    ap.add_argument("--n-train", type=int, help="# polytopes for training.")
    ap.add_argument("--n-val", type=int, help="# polytopes for validation.")
    ap.add_argument("--n-test", type=int, help="# polytopes for testing.")
    ap.add_argument("--seed", type=int, help="Shuffle seed.")
    args = ap.parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    polys = args.polys or cfg.get("polys_file")
    triangs = args.triangs or cfg.get("triangs_file")
    out_dir = args.out or cfg.get("out_dir", "data")
    n_vertices = args.n_vertices if args.n_vertices is not None else cfg.get("n_vertices")
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    if not polys or not triangs or n_vertices is None:
        ap.error("provide --polys, --triangs and --n-vertices (or a --config that sets them).")

    # Split ranges are polytope-GROUP index ranges [start, end]. From a config we
    # take them directly; from flag counts we lay them out sequentially.
    if args.config and args.n_train is None:
        n_train, n_val, n_test = cfg["n_train"], cfg["n_val"], cfg["n_test"]
    else:
        t = args.n_train if args.n_train is not None else 2000
        v = args.n_val if args.n_val is not None else 200
        w = args.n_test if args.n_test is not None else 200
        n_train, n_val, n_test = [0, t], [t, t + v], [t + v, t + v + w]

    os.makedirs(out_dir, exist_ok=True)
    j = lambda name: os.path.join(out_dir, name)
    split_data_into_three_files(
        polys, triangs,
        j("train_polys.json"), j("train_triangs.json"),
        j("val_polys.json"), j("val_triangs.json"),
        j("test_polys.json"), j("test_triangs.json"),
        n_vertices, n_train, n_val, n_test,
        randomize=bool(cfg.get("randomize", True)), seed=seed,
    )


if __name__ == "__main__":
    main()
