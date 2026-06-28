import os
import torch
# os.environ["OMP_NUM_THREADS"] = "10"
# os.environ["MKL_NUM_THREADS"] = "10"
# os.environ["OMP_DYNAMIC"]     = "FALSE"
# os.environ["MKL_DYNAMIC"]     = "FALSE"
# os.environ["KMP_BLOCKTIME"]   = "1"
# os.environ["KMP_AFFINITY"]    = "granularity=fine,compact,1,0"
# torch.set_num_threads(10)
# torch.set_num_interop_threads(10)
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

import time


import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from inference import generate_triangulations
import random
from itertools import product
from utilities import cleanup, save_output, get_np_input_polytopes_and_masks_from_file, get_tensor_input_polytopes_triangs_and_masks_from_file

from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ConstantLR

from Args import ModelParams, EncodingParams, TrainingParams, JobParams, parse_arguments, return_train_data_loader, return_transformer, model_params_from_checkpoint, encoding_params_from_checkpoint


def setup(global_rank, world_size):
    """Initialize the distributed environment."""
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=global_rank,
        world_size=world_size
    )

def set_seeds(base_seed, global_rank):
    seed = base_seed + global_rank
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(local_rank, world_size, node_rank, gpus_per_node, model_params: ModelParams, encoding_params: EncodingParams, training_params: TrainingParams, job_params: JobParams, verbose=True):
    """Main function to train models
    If job_params.continued_training is true, a few hours before the end of the maximum time job_params.max_time, the training is stopped and the model, along with the number of steps already trained, are saved in a checkpoint.
    If job_params.continued_training is true, at launch, the function tries to load a model from checkpoint_from_which_to_resume_dir, along with the number of steps already trained.
    If this number is smaller than training_params.n_steps, the training is continued from this checkpoint.
    """
    global_rank = node_rank * gpus_per_node + local_rank
    print(f"Global rank: {global_rank}, Local rank: {local_rank}, Node rank: {node_rank}, World size: {world_size}, GPUs per node: {gpus_per_node}")
    if global_rank == 0:
        print(f"Process started with global rank: {global_rank}")
    parallelism = world_size > 1
    if parallelism:
        setup(global_rank, world_size)
        torch.cuda.set_device(local_rank)  # Set the current device for this process
        dist.barrier(device_ids=[torch.cuda.current_device()])

    assert training_params.N_polys_monitoring % world_size == 0, "N_polys_monitoring should be divisible by the number of GPUs"


    # Used to stop the training min(5h, job_params.max_time/10) before job_params.max_time and to launch it again from a checkpoint
    starting_time = time.time()

    if global_rank == 0:
        print("\n\n------")
        print("Start of the experiment")
        if verbose:
            print("\n------")
            print("Model parameters:")
            model_params.display()
            print("\n------")
            print("Encoding parameters:")
            encoding_params.display()
            print("\n------")
            print("Training parameters:")
            training_params.display()
            print("\n------")
            print("Job parameters:")
            job_params.display()
            print("\n\n")


    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = f'Checkpoints/{job_params.folder_name}/{job_params.exp_name}'
    saved_output_dir = f'Saved_outputs/{job_params.folder_name}/{job_params.exp_name}'
    if global_rank ==0 and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if global_rank ==0 and not os.path.exists(saved_output_dir):
        os.makedirs(saved_output_dir)

    # Use the GPU when requested and available; otherwise fall back to CPU.
    if getattr(job_params, "Gpu", True) and torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')  # Explicitly assign the device
    else:
        device = torch.device('cpu')
    # torch.set_default_device(device)  # Set the default device for the current process
    torch.set_default_dtype(torch.float32)   # Set the default data type


    # Load the data
    if global_rank == 0:
        print("Create dataloader")
    # Batch size for loader_train and loader_val is training_params.batch_size, but for loader_test it is training_params.N_polys_monitoring//world_size
    # The distributed samplers are already passed to the data loaders
    loader_train = return_train_data_loader(training_params, encoding_params, job_params, world_size, global_rank, verbose=(global_rank == 0))

    set_seeds(training_params.random_seed, global_rank)


    if parallelism:
        torch.cuda.synchronize(torch.cuda.current_device())
        dist.barrier()


    first_loop_after_reloading = False
    # Open the checkpoint directory and see if there is already a model stored there:
    reloading = job_params.continued_training and os.listdir(checkpoint_dir)
    if reloading:
        first_loop_after_reloading = True
        if global_rank == 0:
            print(f"Continued training: loading a model from {checkpoint_dir}")
        checkpoint_files = os.listdir(checkpoint_dir)
        # among the checkpoint_files whose names are chkpt-<step>, we take the one with the maximum step
        checkpoint_file = max(checkpoint_files, key=lambda x: int(x.split('-')[1]))
        checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_file), map_location=device)
        current_step = checkpoint['last_step'] + 1
        # Should normally be the same as those given as arguments to train
        model_params = model_params_from_checkpoint(checkpoint)
        encoding_params = encoding_params_from_checkpoint(checkpoint)
        transformer = return_transformer(model_params, encoding_params, device).to(device)
        transformer.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(
            transformer.parameters(),
            lr=training_params.lr,
            betas=(training_params.beta1, training_params.beta2),
            eps=training_params.eps
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss_train = checkpoint['loss']
        loss_val = checkpoint['validation']
    else:
        if global_rank == 0:
            print("Initializing a new model")
        current_step = 0
        transformer = return_transformer(model_params, encoding_params, device).to(device)
        optimizer = optim.Adam(
            transformer.parameters(),
            lr=training_params.lr,
            betas=(training_params.beta1, training_params.beta2),
            eps=training_params.eps
        )
        loss_train = []
        loss_val = []
    if global_rank == 0:
        print('There are', sum(p.numel() for p in transformer.parameters()), 'parameters.')

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=encoding_params.padding_idx)



    if training_params.scheduler == "None":
        scheduler = ConstantLR(optimizer, factor=1, total_iters=1)
    elif training_params.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=training_params.n_steps)
    elif training_params.scheduler == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=0.8)
    else:
        if global_rank == 0:
            print("Invalid learning rate scheduler")
        scheduler = ConstantLR(optimizer, factor=1, total_iters=1)

    if reloading:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if current_step >= training_params.n_steps:
        if global_rank == 0:
            print(f"Training already completed, last step: {current_step}")
        if parallelism:
            cleanup()
        return

    optimizer.zero_grad()
    training_start = time.time()
    if parallelism:
        transformer = nn.parallel.DistributedDataParallel(transformer, device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device())
        torch.cuda.synchronize(torch.cuda.current_device())
        dist.barrier()
    if global_rank == 0:
        print("Starting training")

    epoch = 0
    step = current_step

    while True:
        loader_train.sampler.set_epoch(epoch)

        sub_step = 0
        for src_data, tgt_data, src_mask_data in loader_train:


            transformer.train()
            src_data = src_data.to(device, non_blocking=True, dtype=torch.float32)
            tgt_data = tgt_data.to(device, non_blocking=True, dtype=torch.int64)
            src_mask_data = src_mask_data.to(device, non_blocking=True, dtype=torch.int64)

            # Forward pass
            output = transformer(src_data, tgt_data, src_mask_data)
            loss = criterion(output[:, :-1, :].contiguous().view(-1, encoding_params.tgt_vocab_size),
                            tgt_data[:, 1:].contiguous().view(-1))
            loss_train.append(loss.item())
            loss.backward()
            sub_step += 1
            if sub_step % training_params.grad_acc_period == 0:
                optimizer.step()
                optimizer.zero_grad()
                if training_params.scheduler in ["cosine"]:
                    scheduler.step()
                elif training_params.scheduler in ["exponential"] and step%100000 == 0 and step >0:
                    scheduler.step()

                # Evaluate model (NOTE: no parallelism here)
                if global_rank ==0 and (step % training_params.loss_eval_frequency == 0 or first_loop_after_reloading or step == training_params.n_steps - 1):
                    with torch.set_grad_enabled(False):
                        transformer_for_eval = transformer.module if isinstance(transformer, nn.parallel.DistributedDataParallel) else transformer
                        transformer_for_eval.eval()
                        # sample and transform polytopes as for the training batches
                        val_src_data, val_tgt_data, val_src_mask_data = get_tensor_input_polytopes_triangs_and_masks_from_file(job_params.polys_file_val, job_params.triangs_file_val,\
                                    training_params.n_vertices, encoding_params.max_seq_length_tgt, training_params.batch_size, permutation=training_params.permute_training_polys, triang_shuffling=training_params.triang_shuffling, verbose=True)
                        val_src_data = val_src_data.to(device, non_blocking=True, dtype=torch.float32)
                        val_tgt_data = val_tgt_data.to(device, non_blocking=True, dtype=torch.int64)
                        val_src_mask_data = val_src_mask_data.to(device, non_blocking=True, dtype=torch.int64)
                        val_output = transformer_for_eval(val_src_data, val_tgt_data, val_src_mask_data)
                        val_loss = criterion(val_output[:, :-1, :].contiguous().view(-1, encoding_params.tgt_vocab_size),
                                            val_tgt_data[:, 1:].contiguous().view(-1))
                        loss_val.append(val_loss.item())
                        print(f"Ep: {step} | Train loss: {np.round(loss_train[-1], decimals=3)} | Val loss: {np.round(loss_val[-1], decimals=3)} | Learning rate scheduler: {np.round(scheduler.get_last_lr()[0], decimals=7)} | Time elapsed: {np.round((time.time()-training_start)/3600, decimals=3)} hr")

                if parallelism:
                    torch.cuda.synchronize(torch.cuda.current_device())
                    dist.barrier()

                if  (step % training_params.frst_gen_eval_frequency == 0 and step>0) or first_loop_after_reloading or step == training_params.n_steps - 1:
                    with torch.set_grad_enabled(False):
                        transformer_for_monitoring = transformer.module if isinstance(transformer, nn.parallel.DistributedDataParallel) else transformer
                        transformer_for_monitoring.eval()
                        monitoring_start_time = time.time()
                        if global_rank == 0 :
                            print("Starting monitoring FRST generation")
                            print(f"Generating triangulations ({training_params.N_polys_monitoring} polytopes, {training_params.numOfTs_per_poly} triangulations per polytope)")
                        sampling_modes = (["uniform"] if training_params.sample_polys_uniformly_for_evaluation else []) + \
                                              (["proportional"] if training_params.sample_polys_wrt_number_of_triangs_for_evaluation else [])
                        permutation_modes = (["perm_once"] if training_params.permute_evaluation_polys_once else []) + \
                                            (["perm_each"] if training_params.permute_evaluation_polys_for_each_triang else [])
                        for sampling_and_permutation_mode in product(sampling_modes, permutation_modes):
                            if global_rank == 0:
                                print(f"Sampling and permutation mode: {sampling_and_permutation_mode}")
                            polys_monitoring, poly_masks_monitoring = get_np_input_polytopes_and_masks_from_file(
                                job_params.polys_file_test, training_params.n_vertices, training_params.N_polys_monitoring//world_size, unique_sampling=(sampling_and_permutation_mode[0] == "uniform"), permutation=True, verbose=False
                            )
                            toSave_Ts_new, toSave_numOfFails = generate_triangulations(
                                transformer_for_monitoring, polys_monitoring, poly_masks_monitoring,
                                training_params.numOfTs_per_poly, encoding_params.padding_idx,
                                encoding_params.max_seq_length_tgt, permutation_for_each_triang=(sampling_and_permutation_mode[1] == "perm_each"), save_dir="", device=device, verbose = (global_rank == 0)
                            )
                            assert toSave_Ts_new.shape == (training_params.N_polys_monitoring//world_size, training_params.numOfTs_per_poly, encoding_params.max_seq_length_tgt), f"toSave_Ts_new shape: {toSave_Ts_new.shape} - Expected shape: {(training_params.N_polys_monitoring//world_size, training_params.numOfTs_per_poly, encoding_params.max_seq_length_tgt)}"
                            if global_rank == 0:
                                print(f"Saving the triangulations")
                            save_output(polys_monitoring, poly_masks_monitoring, toSave_Ts_new, saved_output_dir+f"/{sampling_and_permutation_mode[0]}_{sampling_and_permutation_mode[1]}/global_rank_{global_rank}-{step}")

                        if global_rank == 0:
                            print(f"Monitoring time: {np.round((time.time()-monitoring_start_time)/60, decimals=2)} mn")

                # Save and stop the training if it exceeds the maximum time - 5h
                scheduled_end_of_training =  (job_params.continued_training and (time.time() - starting_time)/3600.0 > job_params.max_time - 5)
                if global_rank ==0 and (scheduled_end_of_training or (step % training_params.checkpoint_frequency == 0 and step>0) or step == training_params.n_steps - 1):
                    if step == training_params.n_steps - 1:
                        print("Training finished")
                    if scheduled_end_of_training:
                        print("The maximum time limit is about to be reached")
                    print(f'Saving the model at Checkpoints/{job_params.folder_name}/{job_params.exp_name}/chkpt-{step}')
                    hyperparams = {
                        'tgt_vocab_size': encoding_params.tgt_vocab_size,
                        'd_model_src': model_params.d_model_src,
                        'd_model_tgt': model_params.d_model_tgt,
                        'd_model': model_params.d_model,
                        'num_heads': model_params.num_heads,
                        'num_layers': model_params.num_layers,
                        'd_ff_enem': model_params.d_ff_enem,
                        'd_ff': model_params.d_ff,
                        'max_seq_length_src': encoding_params.max_seq_length_src,
                        'max_seq_length_tgt': encoding_params.max_seq_length_tgt,
                        'batch_size': training_params.batch_size,
                        'dropout': model_params.dropout,
                        'lr': training_params.lr
                    }
                    transformer_for_saving = transformer.module if isinstance(transformer, nn.parallel.DistributedDataParallel) else transformer
                    # Save the model
                    torch.save({
                        'model_state_dict': transformer_for_saving.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'n_vertices': training_params.n_vertices,
                        'loss': loss_train,
                        'last_step': step,
                        'validation': loss_val,
                        'hyperparams': hyperparams
                    }, f'Checkpoints/{job_params.folder_name}/{job_params.exp_name}/chkpt-{step}')
                if parallelism:
                    torch.cuda.synchronize(torch.cuda.current_device())
                    dist.barrier()
                if scheduled_end_of_training or step == training_params.n_steps - 1:
                    if global_rank ==0:
                        print("Stopping the training")
                    if parallelism:
                        cleanup()
                    return
                if first_loop_after_reloading:
                    first_loop_after_reloading = False
                if sub_step % training_params.grad_acc_period == 0:
                    step += 1

        epoch += 1




if __name__ == "__main__":
    # Direct CLI entry (legacy). Preferred entrypoint: `python -m cyt.train --config ...`
    args = parse_arguments()
    gpus_per_node = torch.cuda.device_count()
    world_size = gpus_per_node if gpus_per_node > 0 else 1
    if world_size > 1:
        mp.spawn(train, args=(world_size, 0, world_size, args[0], args[1], args[2], args[3]), nprocs=gpus_per_node, join=True)
    else:
        train(0, 1, 0, 1, args[0], args[1], args[2], args[3])