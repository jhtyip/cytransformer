"""
Load a trained CYTransformer checkpoint and generate triangulations for a set of
polytopes, saving the polytopes, masks and sampled triangulations to .npy files.

Supports multi-GPU runs: every rank samples the *same* polytopes but draws its
own triangulations (seeds differ per rank), and rank 0 merges the per-rank
outputs at the end.
"""

import torch
import sys
import os
# Generation over many polytopes can be slow; raise the NCCL collective timeout
# so the cross-rank barriers do not abort on long-running ranks.
os.environ["NCCL_TIMEOUT"] = "36000"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import time
from utilities import cleanup, get_np_input_polytopes_and_masks_from_file
from monitoring import parallel_monitor_perf, monitor_perf, evaluate_generation_results_saved_outputs_format, evaluate_generation_results_saved_outputs_to_share_format
from inference import generate_triangulations
import random
from torch.utils.data import DistributedSampler, DataLoader
from Polys_triangs_dataset import Polys_triangs_dataset
from Args import return_transformer, model_params_from_checkpoint, encoding_params_from_checkpoint
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import json

"""
Generate and save triangulations from a checkpointed model

"""

# Unclear whether this is actually needed in this script
def setup(global_rank, world_size):
    """Initialize the distributed environment."""
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=global_rank,
        world_size=world_size
    )

def set_seeds(base_seed, global_rank):
    # Offset by rank so different ranks sample different triangulations.
    seed = base_seed + global_rank
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_triangs_from_checkpoint(local_rank:int,
                                     world_size:int,
                                     node_rank:int,
                                     gpus_per_node:int,
                                     checkpoint_path: str,
                                     num_of_polys: int,
                                     num_of_triangs_per_poly: int,
                                     polys_file: str,
                                     save_results_file_name: str,
                                     seed: int = 42,
                                     verbose = True,
                                     permutation = True,
                                     permutation_for_each_triang = False,
                                     unique_polytope_sampling = True
                                     ):

    """Generate triangulations from a checkpoint and save them to disk.

    When parallelized, all ranks sample the same polytopes, then independently generate guesses

    """
    global_rank = node_rank * gpus_per_node + local_rank
    if global_rank == 0:
        print(f"Starting generation process from checkpoint")
        print(f"Global rank: {global_rank}, Local rank: {local_rank}, Node rank: {node_rank}, World size: {world_size}, GPUs per node: {gpus_per_node}")

    parallelism = world_size > 1
    # sleep 5s to avoid overlapping with the previous job
    time.sleep(5)
    if parallelism:
        # Multi-GPU: join the process group and pin this process to its device.
        setup(global_rank, world_size)
        torch.cuda.set_device(local_rank)  # Set the current device for this process
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        # Single-process: just report the SLURM job id if running under SLURM.
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id is not None:
            print(f"SLURM Job ID: {slurm_job_id}")
        else:
            print("SLURM_JOB_ID is not set.")

    assert num_of_triangs_per_poly % world_size == 0, f"num_of_triangs_per_poly should be divisible by the number of GPUs, but got {num_of_triangs_per_poly} and {world_size}"

    starting_time = time.time()

    device = torch.device(f'cuda:{local_rank}')  # Explicitly assign the device
    # torch.set_default_device(device)  # Set the default device for the current process NOTE not compatible with the dataloader
    torch.set_default_dtype(torch.float32)   # Set the default data type


    if global_rank == 0:
        print(f"Random seed {seed}")
    # Use rank 0's seed on every rank here so all ranks select the same polytopes.
    set_seeds(seed, 0) # same seed to select the polytopes for all ranks


    # Load the checkpoint and rebuild the model/encoding config it was saved with.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_params = model_params_from_checkpoint(checkpoint)
    encoding_params = encoding_params_from_checkpoint(checkpoint)

    # Fetch data (same for all ranks)
    polys, poly_masks = get_np_input_polytopes_and_masks_from_file(polys_file, encoding_params.max_seq_length_src, num_of_polys, unique_sampling=unique_polytope_sampling, permutation=permutation, verbose=verbose and global_rank == 0)

    polys, poly_masks = torch.from_numpy(polys), torch.from_numpy(poly_masks)


    # Load model
    if global_rank == 0:
        print("Loading model")
    transformer = return_transformer(model_params, encoding_params, device).to(device)
    transformer.load_state_dict(checkpoint['model_state_dict'])
    transformer.eval()

    # Now diverge the seeds per rank so each rank samples different triangulations.
    if global_rank == 0:
        print("Changing the random seeds across ranks")
    set_seeds(seed, global_rank)

    if parallelism:
        torch.cuda.synchronize(torch.cuda.current_device())
        dist.barrier()

    with torch.set_grad_enabled(False):
        transformer.eval()
        # Split the requested triangulations-per-poly evenly across ranks.
        num_of_triangs_per_poly_for_this_rank = num_of_triangs_per_poly // world_size
        if global_rank == 0:
            print(f"Rank {global_rank}: generating triangulations ({num_of_polys} polytopes, {num_of_triangs_per_poly_for_this_rank} triangulations per polytope)")
        current_time = time.time()
        triangs = []
        # Process polytopes in blocks of 50 to keep memory bounded.
        for i in range(0, num_of_polys, 50):
            if verbose:
                print(f"Starting processing of polytope {i} - time elapsed: {time.time() - current_time}")
            batch_triangs, _ = generate_triangulations(
                transformer, polys[i:i + 50], poly_masks[i:i + 50],
                num_of_triangs_per_poly_for_this_rank, encoding_params.padding_idx,
                encoding_params.max_seq_length_tgt, permutation_for_each_triang, "", device
            )
            triangs.append(batch_triangs)
        triangs = np.concatenate(triangs, axis=0)

        dir_path = os.path.dirname(save_results_file_name)
        os.makedirs(dir_path, exist_ok=True)
        if parallelism:
            # Each rank writes its own triangulations; the (shared) polys/masks are
            # saved once by rank 0.
            if global_rank==0:
                print(f"Rank {global_rank}: saving results in {save_results_file_name}_rank_{global_rank}")
                np.save(save_results_file_name + "_polys_monitoring.npy", polys)
                np.save(save_results_file_name + "_poly_masks_monitoring.npy", poly_masks)
            np.save(save_results_file_name + f"_rank_{global_rank}_toSave_Ts_new.npy", triangs)
        else:
            print(f"Saving results in {save_results_file_name}")
            np.save(save_results_file_name + "_polys_monitoring.npy", polys)
            np.save(save_results_file_name + "_poly_masks_monitoring.npy", poly_masks)
            np.save(save_results_file_name + "_toSave_Ts_new.npy", triangs)
        print(f"Rank {global_rank}: generation finished, time elapsed: {np.round((time.time()-current_time)/3600, decimals=1)}h")
    if parallelism:
        torch.cuda.synchronize(torch.cuda.current_device())
        dist.barrier()
    # Rank 0 concatenates every rank's triangulations (along the per-poly axis) into
    # a single combined output file.
    if parallelism and global_rank == 0:
        print("Merging the results")
        polys_full = np.load(save_results_file_name+f"_polys_monitoring.npy" )
        poly_masks_full = np.load(save_results_file_name+f"_poly_masks_monitoring.npy" )
        triangs_full = []
        for r in range(world_size):
            triangs = np.load(save_results_file_name+f"_rank_{r}_toSave_Ts_new.npy" )
            triangs_full.append(triangs)
        triangs_full = np.concatenate(triangs_full, axis = 1)
        np.save(save_results_file_name + "_polys_monitoring.npy", polys_full)
        np.save(save_results_file_name + "_poly_masks_monitoring.npy", poly_masks_full)
        np.save(save_results_file_name + "_toSave_Ts_new.npy", triangs_full)
    if parallelism:
        torch.cuda.synchronize(torch.cuda.current_device())
        dist.barrier()
        cleanup()
