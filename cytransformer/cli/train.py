"""
One-command training entrypoint.

    python -m cyt.train --config configs/train_9+1.yaml

Single-process by default (CPU or one GPU). If multiple CUDA devices are visible
and the config requests the GPU, it spawns one process per GPU (the project's
existing DistributedDataParallel path). No SLURM / cluster glue involved.
"""
import argparse

import torch

from cytransformer.config import load_train_config
from cytransformer.train import train


def main():
    ap = argparse.ArgumentParser(description="Train CYTransformer from a YAML config.")
    ap.add_argument("--config", required=True, help="Path to a training YAML config.")
    args = ap.parse_args()

    model_params, encoding_params, training_params, job_params, _rl = load_train_config(args.config)

    use_gpu = bool(getattr(job_params, "Gpu", True)) and torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_gpu else 1

    if world_size > 1:
        import torch.multiprocessing as tmp
        tmp.spawn(
            train,
            args=(world_size, 0, world_size, model_params, encoding_params, training_params, job_params),
            nprocs=world_size,
            join=True,
        )
    else:
        # local_rank=0, world_size=1, node_rank=0, gpus_per_node=1  -> no DDP/NCCL
        train(0, 1, 0, 1, model_params, encoding_params, training_params, job_params)


if __name__ == "__main__":
    main()
