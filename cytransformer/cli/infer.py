"""
Inference: generate candidate triangulations for input polytopes using trained
weights, and verify each is a genuine FRST in real time.

  Simple:  cyt-infer --checkpoint model.pt --polys polytopes.json
  Config:  cyt-infer --config configs/infer_9+1.yaml

No CYTools. FRST validation (CYTools-free) is ON by default; pass --no-validate to
skip it. The encoding (vocab, n_vertices, sequence lengths) is read from the
checkpoint, so it always matches how the model was trained.
"""
import argparse
import os

import torch

from cytransformer.config import load_infer_config
from cytransformer.args import (
    model_params_from_checkpoint,
    encoding_params_from_checkpoint,
    return_transformer,
)
from cytransformer.utilities import get_np_input_polytopes_and_masks_from_file, save_output
from cytransformer.inference import generate_triangulations


def main():
    ap = argparse.ArgumentParser(description="Generate FRSTs from input polytopes (real-time FRST validation by default).")
    ap.add_argument("--config", help="Optional YAML config; the flags below override it.")
    ap.add_argument("--checkpoint", help="Path to the trained weights file.")
    ap.add_argument("--polys", help="Input polytopes JSON ([POLYID, DRESVERTS] format).")
    ap.add_argument("--num-of-polys", type=int, help="How many polytopes to use.")
    ap.add_argument("--num-per-poly", type=int, help="Triangulations to generate per polytope.")
    ap.add_argument("--out", help="Where to save results (path prefix).")
    ap.add_argument("--no-validate", action="store_true", help="Skip real-time FRST validation.")
    ap.add_argument("--gpu", action="store_true", help="Use the GPU if available.")
    args = ap.parse_args()

    # Start from the config (if any), then let explicit flags override it.
    cfg = load_infer_config(args.config) if args.config else {}
    if args.checkpoint is not None:   cfg["checkpoint_path"] = args.checkpoint
    if args.polys is not None:        cfg["polys_file"] = args.polys
    if args.num_of_polys is not None: cfg["num_of_polys"] = args.num_of_polys
    if args.num_per_poly is not None: cfg["num_of_triangs_per_poly"] = args.num_per_poly
    if args.out is not None:          cfg["save_results_file"] = args.out
    if args.no_validate:              cfg["validate"] = False
    if args.gpu:                      cfg["gpu"] = True

    if "checkpoint_path" not in cfg or "polys_file" not in cfg:
        ap.error("provide --checkpoint and --polys (or a --config that sets them).")

    use_gpu = bool(cfg.get("gpu", True)) and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    ckpt = torch.load(cfg["checkpoint_path"], map_location=device, weights_only=False)
    model_params = model_params_from_checkpoint(ckpt)        # FROZEN reader
    encoding_params = encoding_params_from_checkpoint(ckpt)  # FROZEN reader

    model = return_transformer(model_params, encoding_params, device).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    polys, masks = get_np_input_polytopes_and_masks_from_file(
        cfg["polys_file"], ckpt["n_vertices"], int(cfg.get("num_of_polys", 10)),
        unique_sampling=bool(cfg.get("unique_polytope_sampling", True)),
        permutation=bool(cfg.get("permutation", True)),
        verbose=True,
    )

    Ts, num_fails = generate_triangulations(
        model, polys, masks,
        int(cfg.get("num_of_triangs_per_poly", 10)),
        encoding_params.padding_idx, encoding_params.max_seq_length_tgt,
        bool(cfg.get("permutation_for_each_triang", False)),
        "", device, verbose=True,
    )

    out = cfg.get("save_results_file")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        save_output(polys, masks, Ts, out)
        print(f"Saved results to {out}*")

    # On-the-go FRST verification (CYTools-free; default on, needs pycddlib).
    if bool(cfg.get("validate", True)):
        import numpy as np
        from cytransformer.validation.frst import is_frst
        is_frst_mask = np.zeros(Ts.shape[:2], dtype=bool)   # (num_of_polys, num_of_triangs)
        for i in range(len(polys)):
            for k in range(Ts.shape[1]):
                is_frst_mask[i, k] = is_frst(polys[i], masks[i], Ts[i, k], encoding_params.padding_idx)
        n_frst, n_total = int(is_frst_mask.sum()), is_frst_mask.size
        print(f"FRST validation: {n_frst}/{n_total} candidates are FRSTs ({np.round(100*n_frst/n_total, 1)}%)")
        if out:
            np.save(out + "_is_frst.npy", is_frst_mask)
            print(f"Saved FRST mask to {out}_is_frst.npy")

    print(f"Done. polys={polys.shape}, triangulation tokens={Ts.shape}")


if __name__ == "__main__":
    main()
