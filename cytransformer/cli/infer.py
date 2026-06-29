"""
One-command inference entrypoint: generate FRST candidate triangulations for
polytopes using a trained checkpoint.

    python -m cyt.infer --config configs/infer_9+1.yaml

CYTools is NOT required: this only generates candidate token sequences. Checking
whether a candidate is a genuine FRST is a separate, optional step (needs CYTools).
The encoding (vocab, n_vertices, sequence lengths) is read from the checkpoint, so
it always matches how the model was trained.
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
    ap = argparse.ArgumentParser(description="Generate FRST candidates from a checkpoint.")
    ap.add_argument("--config", required=True, help="Path to an inference YAML config.")
    args = ap.parse_args()
    cfg = load_infer_config(args.config)

    use_gpu = bool(cfg.get("gpu", True)) and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    ckpt = torch.load(cfg["checkpoint_path"], map_location=device, weights_only=False)
    model_params = model_params_from_checkpoint(ckpt)      # FROZEN reader
    encoding_params = encoding_params_from_checkpoint(ckpt)  # FROZEN reader
    n_vertices = ckpt["n_vertices"]

    model = return_transformer(model_params, encoding_params, device).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    polys, masks = get_np_input_polytopes_and_masks_from_file(
        cfg["polys_file"], n_vertices, int(cfg.get("num_of_polys", 10)),
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

    # Optional on-the-go FRST verification (CYTools-free; needs scipy + pycddlib).
    if bool(cfg.get("validate", False)):
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
