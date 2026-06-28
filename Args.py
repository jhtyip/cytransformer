
import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import math
import json

from Models import Transformer
from data_generation import  split_data
from torch.utils.data import DataLoader, DistributedSampler
from Polys_triangs_dataset import Polys_triangs_dataset
from torch import Generator


class Params:
    def display(self):
        for attr_name in dir(self):
            if not attr_name.startswith('__'):
                attr_value = getattr(self, attr_name)
                print(f"{attr_name}: {attr_value}")

class ModelParams(Params):
    d_model_src: int = 4  # Vertex coordinates in 4D; fixed
    d_model_tgt: int = 1  # Class indices; fixed
    d_model: int = 256
    num_heads: int = 16
    num_layers: int = 12
    d_ff_enem: int = 1024
    d_ff: int = 1024  # 4*d_model
    dropout: int = 0.1

class EncodingParams(Params):
    padding_idx: int = math.comb(9, 4) + 1 + 1
    tgt_vocab_size: int = math.comb(9, 4) + 1 + 1 + 1
    max_seq_length_tgt: int = 30
    max_seq_length_src: int = 9

class TrainingParams(Params):
    n_vertices: int = 9

    split_by_polytope: bool = True
    max_number_triangs_per_polytope: int = -1
    n_train: int = 3000
    n_val: int = 200
    n_test: int = 200
    batch_size: int = 128
    grad_acc_period: int = 4

    n_steps: int = 400000
    loss_eval_frequency: int = 500
    frst_gen_eval_frequency: int = 25000
    checkpoint_frequency: int = 2500000
    N_polys_monitoring: int = 10
    numOfTs_per_poly: int = 10
    permute_training_polys: bool = True
    triang_shuffling: bool = True
    # The two options below are NOT mutually exclusive
    permute_evaluation_polys_for_each_triang: bool = True
    permute_evaluation_polys_once: bool = True
    # The two options below are NOT mutually exclusive
    sample_polys_uniformly_for_evaluation: bool = True
    sample_polys_wrt_number_of_triangs_for_evaluation: bool = True

    lr: float = 0.00005
    scheduler: str = "None"
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-9

    random_seed: int = 43
    # Deprecated for now
    resume_from_checkpoint: bool = False

class RLParams(Params):
    rl_data_folder: str = "RL_data/RL_debug"
    n_iterations: int = 5
    n_polys_for_guesses: int = 1000
    n_triangs_per_poly_for_guesses: int = 100
    no_rl: bool = False
    initial_number_of_triangs_per_poly: int = 5
    permute_polys_for_new_data_generation: bool = False
    permute_polys_for_each_triang_for_new_data_generation: bool = False
    sample_polys_uniformly_for_new_data_generation: bool = False



class JobParams(Params):
    folder_name: str = "9+1"
    exp_name: str = "template_run"

    # The maximum number of times training can be relaunched to reach the required number of steps
    # Only compatible with job arrays
    continued_training: bool = True
    max_time: int = 70  # Maximum time in hours
    polys_file_train: str = "Data/9+1/debug_2/9+1_polys_0_4999_train_2000.json"
    triangs_file_train: str = "Data/9+1/debug_2/9+1_triangs_0_4999_train_2000.json"
    polys_file_val: str = "Data/9+1/debug_2/9+1_polys_0_4999_val_500.json"
    triangs_file_val: str = "Data/9+1/debug_2/9+1_triangs_0_4999_val_500.json"
    polys_file_test: str = "Data/9+1/debug_2/9+1_polys_0_4999_test_500.json"
    triangs_file_test: str = "Data/9+1/debug_2/9+1_triangs_0_4999_test_500.json"
    num_cpus: int = 1  # Number of CPU cores to use for data loading

    # Deprecated for now
    checkpoint_from_which_to_resume_dir: None

    Gpu: bool = True

# Legacy
# def return_data_loader(training_params: TrainingParams, encoding_params: EncodingParams, job_params: JobParams, verbose = True):
#     data_loader = Dataloader(max_seq_length_tgt = encoding_params.max_seq_length_tgt, max_seq_length_src = encoding_params.max_seq_length_src, \
#                                 padding_idx = encoding_params.padding_idx, triangs_file = job_params.triangs_file, polys_file = job_params.polys_file, \
#                                     n_train = training_params.n_train, n_val = training_params.n_val, n_test = training_params.n_test, \
#                                         split_by_polytope = training_params.split_by_polytope, max_number_triangs_per_polytope = training_params.max_number_triangs_per_polytope, verbose = verbose)
#     return data_loader



def return_train_data_loader(training_params: TrainingParams, encoding_params: EncodingParams, job_params: JobParams, world_size: int, global_rank: int, verbose = True):
    N_vertices = encoding_params.max_seq_length_src
    max_seq_length_tgt = encoding_params.max_seq_length_tgt
    with open(job_params.polys_file_train, 'r') as f:
        train_polys = json.load(f)
    with open(job_params.triangs_file_train, 'r') as f:
        train_triangs = json.load(f)
    with open(job_params.polys_file_val, 'r') as f:
        val_polys = json.load(f)
    with open(job_params.triangs_file_val, 'r') as f:
        val_triangs = json.load(f)
    with open(job_params.polys_file_test, 'r') as f:
        test_polys = json.load(f)
    with open(job_params.triangs_file_test, 'r') as f:
        test_triangs = json.load(f)


    assert len(train_polys) == len(train_triangs), "Number of polygons and triangulations in training set must match."
    assert len(val_polys) == len(val_triangs), "Number of polygons and triangulations in validation set must match."
    assert len(test_polys) == len(test_triangs), "Number of polygons and triangulations in test set must match."
    if verbose:
        print(f"Max number of points available in train file, val file, test file: {len(train_polys)}, {len(val_polys)}, {len(test_polys)}")

    polys_train, triangs_train, _, _, _, _ = split_data(train_polys, train_triangs, [0,min(len(train_polys), training_params.n_train)], [0,0], [0,0], training_params.max_number_triangs_per_polytope)
    dataset_train = Polys_triangs_dataset(polys_train, triangs_train, N_vertices, max_seq_length_tgt, training_params.permute_training_polys, triang_shuffling=training_params.triang_shuffling, seed = training_params.random_seed+global_rank)

    if verbose:
        print(f"Train dataset: {len(dataset_train)} samples")


    # Create samplers for distributed training

    sampler_train = DistributedSampler(
    dataset_train,
    num_replicas=world_size,
    rank=global_rank
)

    # Create dataloaders
    loader_train = DataLoader(
    dataset_train,
    batch_size=training_params.batch_size,
    sampler=sampler_train,
    num_workers=job_params.num_cpus,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,  # Use persistent workers for faster data loading
)

    return loader_train

# LEGACY
# def return_data_loaders(training_params: TrainingParams, encoding_params: EncodingParams, job_params: JobParams, world_size: int, global_rank: int, verbose = True):
#     N_vertices = encoding_params.max_seq_length_src
#     max_seq_length_tgt = encoding_params.max_seq_length_tgt
#     with open(job_params.polys_file_train, 'r') as f:
#         train_polys = json.load(f)
#     with open(job_params.triangs_file_train, 'r') as f:
#         train_triangs = json.load(f)
#     with open(job_params.polys_file_val, 'r') as f:
#         val_polys = json.load(f)
#     with open(job_params.triangs_file_val, 'r') as f:
#         val_triangs = json.load(f)
#     with open(job_params.polys_file_test, 'r') as f:
#         test_polys = json.load(f)
#     with open(job_params.triangs_file_test, 'r') as f:
#         test_triangs = json.load(f)


#     assert len(train_polys) == len(train_triangs), "Number of polygons and triangulations in training set must match."
#     assert len(val_polys) == len(val_triangs), "Number of polygons and triangulations in validation set must match."
#     assert len(test_polys) == len(test_triangs), "Number of polygons and triangulations in test set must match."
#     if verbose:
#         print(f"Max number of points available in train file, val file, test file: {len(train_polys)}, {len(val_polys)}, {len(test_polys)}")

#     polys_train, triangs_train, _, _, _, _ = split_data(train_polys, train_triangs, [0,min(len(train_polys), training_params.n_train)], [0,0], [0,0], training_params.max_number_triangs_per_polytope)
#     dataset_train = Polys_triangs_dataset(polys_train, triangs_train, N_vertices, max_seq_length_tgt, True, seed = training_params.random_seed+global_rank)
#     _, _, polys_val, triangs_val, _, _ = split_data(val_polys, val_triangs, [0,0], [0,min(len(val_polys), training_params.n_val)], [0,0], training_params.max_number_triangs_per_polytope)
#     dataset_val = Polys_triangs_dataset(polys_val, triangs_val, N_vertices, max_seq_length_tgt, True, seed = training_params.random_seed+global_rank)
#     _, _, _, _, polys_test, triangs_test  = split_data(test_polys, test_triangs, [0,0], [0,0], [0,min(len(test_polys), training_params.n_test)], training_params.max_number_triangs_per_polytope)
#     dataset_test = Polys_triangs_dataset(polys_test, triangs_test, N_vertices, max_seq_length_tgt, True, seed = training_params.random_seed+global_rank)

#     if verbose:
#         print(f"Train dataset: {len(dataset_train)} samples")
#         print(f"Validation dataset: {len(dataset_val)} samples")
#         print(f"Test dataset: {len(dataset_test)} samples")


#     # Create samplers for distributed training

#     sampler_train = DistributedSampler(
#     dataset_train,
#     num_replicas=world_size,
#     rank=global_rank
# )
#     sampler_val = DistributedSampler(
#     dataset_val,
#     num_replicas=world_size,
#     rank=global_rank
# )
#     sampler_test = DistributedSampler(
#     dataset_test,
#     num_replicas=world_size,
#     rank=global_rank
# )

#     # Create dataloaders
#     loader_train = DataLoader(
#     dataset_train,
#     batch_size=training_params.batch_size,
#     sampler=sampler_train,
#     num_workers=job_params.num_cpus,
#     pin_memory=True,
#     drop_last=True,
#     persistent_workers=True,  # Use persistent workers for faster data loading
# )

#     loader_val = DataLoader(
#     dataset_val,
#     batch_size=training_params.batch_size,
#     sampler=sampler_val,
#     num_workers=job_params.num_cpus,
#     pin_memory=True,
#     drop_last=True,
#     persistent_workers=True,  # Use persistent workers for faster data loading
# )

#     loader_test = DataLoader(
#     dataset_test,
#     batch_size=training_params.N_polys_monitoring//world_size, # !!!
#     sampler=sampler_test,
#     num_workers=job_params.num_cpus,
#     pin_memory=True,
#     drop_last=True,
#     persistent_workers=True,  # Use persistent workers for faster data loading
# )

#     # if verbose:
#     #     print(f"Train loader: {len(loader_train)} batches")
#     #     print(f"Validation loader: {len(loader_val)} batches")
#     #     print(f"Test loader: {len(loader_test)} batches")

#     return loader_train, loader_val, loader_test

def return_transformer(model_params: ModelParams, encoding_params: EncodingParams, device):
    transformer = Transformer(
        d_model_src=model_params.d_model_src,
        d_model_tgt=model_params.d_model_tgt,
        d_model=model_params.d_model,
        tgt_vocab_size=encoding_params.tgt_vocab_size,
        num_heads=model_params.num_heads,
        num_layers=model_params.num_layers,
        d_ff_enem=model_params.d_ff_enem,
        d_ff=model_params.d_ff,
        max_seq_length_src=encoding_params.max_seq_length_src,
        max_seq_length_tgt=encoding_params.max_seq_length_tgt,
        dropout=model_params.dropout,
        device=device,
        padding_idx=encoding_params.padding_idx
    )
    return transformer


def encoding_parameters(n_vertices: int):
    assert n_vertices in [9, 10, 11, 12, 13, 14, 15, 20], "Invalid number of vertices."
    encoding_params = EncodingParams()
    encoding_params.tgt_vocab_size = math.comb(n_vertices, 4) + 1 + 1 + 1
    encoding_params.padding_idx = encoding_params.tgt_vocab_size - 1
    if n_vertices == 9:
        encoding_params.max_seq_length_tgt = 30
        encoding_params.max_seq_length_src = 9
    elif n_vertices == 10:
        encoding_params.max_seq_length_tgt = 35
        encoding_params.max_seq_length_src = 10
    elif n_vertices == 11:
        encoding_params.max_seq_length_tgt = 45
        encoding_params.max_seq_length_src = 11
    elif n_vertices == 12:
        encoding_params.max_seq_length_tgt = 55
        encoding_params.max_seq_length_src = 12
    elif n_vertices == 13:
        encoding_params.max_seq_length_tgt = 65
        encoding_params.max_seq_length_src = 13
    elif n_vertices == 14:
        encoding_params.max_seq_length_tgt = 65
        encoding_params.max_seq_length_src = 14
    elif n_vertices == 15:
        encoding_params.max_seq_length_tgt = 65
        encoding_params.max_seq_length_src = 15
    elif n_vertices == 20:
        encoding_params.max_seq_length_tgt = 90
        encoding_params.max_seq_length_src = 20
    return encoding_params


def model_params_from_checkpoint(checkpoint):
    model_params = ModelParams()
    model_params.d_model = checkpoint['hyperparams']['d_model']
    model_params.num_heads = checkpoint['hyperparams']['num_heads']
    model_params.num_layers = checkpoint['hyperparams']['num_layers']
    model_params.d_ff = checkpoint['hyperparams']['d_ff']
    model_params.d_ff_enem = checkpoint['hyperparams']['d_ff_enem']
    model_params.dropout = checkpoint['hyperparams']['dropout']
    return model_params


def encoding_params_from_checkpoint(checkpoint):
    return encoding_parameters(checkpoint['n_vertices'])



def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse training parameters.")

    # ModelParams
    parser.add_argument("--d_model", type=int, default=256, help="Dimension of the model.")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers.")
    parser.add_argument("--d_ff", type=int, default=1024, help="")
    parser.add_argument("--d_ff_enem", type=int, default=1024, help="")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")

    # TrainingParams
    parser.add_argument("--n_vertices", type=int, choices=[9, 10, 11, 12, 13, 14, 15, 20], default=9, help="Number of vertices.")
    parser.add_argument("--split_by_polytope", type=str, default='True', help="Split data by polytope.")
    parser.add_argument("--max_number_triangs_per_polytope", type=int, default=-1, help="Maximum number of triangulations per polytope.")
    parser.add_argument("--n_train", type=int, default=3000, help="Number of training samples.")
    parser.add_argument("--n_val", type=int, default=200, help="Number of validation samples.")
    parser.add_argument("--n_test", type=int, default=200, help="Number of test samples.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--grad_acc_period", type=int, default=4, help="Gradient accumulation period.")
    parser.add_argument("--n_steps", type=int, default=400000, help="Number of steps.")
    parser.add_argument("--loss_eval_frequency", type=int, default=500, help="Loss evaluation frequency.")
    parser.add_argument("--checkpoint_frequency", type=int, default=25000, help="Checkpoint frequency.")
    parser.add_argument("--frst_gen_eval_frequency", type=int, default=25000, help="FRST generation evaluation frequency.")
    parser.add_argument("--N_polys_monitoring", type=int, default=50, help="Number of polytopes for monitoring.")
    parser.add_argument("--numOfTs_per_poly", type=int, default=50, help="Number of triangulations per polytopes for monitoring.")
    parser.add_argument("--permute_training_polys", type=str, default='True', help="Apply permutations to polytopes and FRSTs during training")
    parser.add_argument("--triang_shuffling", type=str, default='True', help="Shuffle triangulations during training")
    parser.add_argument("--permute_evaluation_polys_for_each_triang", type=str, default='False', help="Apply a different permutation to the polytope for each triangulation generated at evaluation time")
    parser.add_argument("--permute_evaluation_polys_once", type=str, default='True', help="Apply a single permutation to each polytope at evaluation time")
    parser.add_argument("--sample_polys_uniformly_for_evaluation", type=str, default='True', help="Sample polytopes uniformly for evaluation.")
    parser.add_argument("--sample_polys_wrt_number_of_triangs_for_evaluation", type=str, default='True', help="Sample polytopes with respect to the number of their triangulations for evaluation.")

    parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate.")
    parser.add_argument("--scheduler", type=str, default="None", help="Scheduler.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1.")
    parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta2.")
    parser.add_argument("--eps", type=float, default=1e-9, help="Adam epsilon")
    parser.add_argument("--random_seed", type=int, default=43, help="Random seed.")
    parser.add_argument("--resume_from_checkpoint", type=str, default='True', help="Resume training from checkpoint.")

    # JobParams
    parser.add_argument("--folder_name", type=str, default="9+1", help="Folder name for checkpoints.")
    parser.add_argument("--exp_name", type=str, default="template_run", help="Experiment name.")
    parser.add_argument("--continued_training", type=str, default='True', help="Continue training from checkpoint.")
    parser.add_argument("--max_time", type=int, default=70, help="Maximum time in hours.")
    parser.add_argument("--polys_file_train", type=str, default="Data/9+1/debug/9+1_polys_0_200_train.json", help="Path to training polytopes file.")
    parser.add_argument("--triangs_file_train", type=str, default="Data/9+1/debug/9+1_triangs_0_200_train.json", help="Path to training triangulations file.")
    parser.add_argument("--polys_file_val", type=str, default="Data/9+1/debug/9+1_polys_0_200_val.json", help="Path to validation polytopes file.")
    parser.add_argument("--triangs_file_val", type=str, default="Data/9+1/debug/9+1_triangs_0_200_val.json", help="Path to validation triangulations file.")
    parser.add_argument("--polys_file_test", type=str, default="Data/9+1/debug/9+1_polys_0_200_test.json", help="Path to test polytopes file.")
    parser.add_argument("--triangs_file_test", type=str, default="Data/9+1/debug/9+1_triangs_0_200_test.json", help="Path to test triangulations file.")
    parser.add_argument("--num_cpus", type=int, default=10, help="Number of CPU cores to use for data loading.")
    parser.add_argument("--Gpu", type=str, default='True', help="Use GPU for training.")

    # RLParams (only used in RL training)
    parser.add_argument("--rl_data_folder", type=str, default="RL_data/RL_debug", help="Folder for RL data.")
    parser.add_argument("--n_iterations", type=int, default=5, help="Number of RL iterations.")
    parser.add_argument("--n_polys_for_guesses", type=int, default=1000, help="Number of polytopes when generating new data.")
    parser.add_argument("--n_triangs_per_poly_for_guesses", type=int, default=100, help="Number of triangulations per polytope when generating new data.")
    parser.add_argument("--no_rl", type=str, default='False', help="Disable RL training.")
    parser.add_argument("--initial_number_of_triangs_per_poly", type=int, default=5, help="Initial number of triangulations per polytope for RL training.")
    parser.add_argument("--permute_polys_for_new_data_generation", type=str, default='False', help="Apply a permutation to the polytope when generating new training data")
    parser.add_argument("--permute_polys_for_each_triang_for_new_data_generation", type=str, default='True', help="Apply a different permutation to the polytope for each triangulation when generating new training data")
    parser.add_argument("--sample_polys_uniformly_for_new_data_generation", type=str, default='True', help="Sample polytopes uniformly when generating new training data.")

    args = parser.parse_args()

    args.split_by_polytope = args.split_by_polytope.lower() == 'true'
    args.resume_from_checkpoint = args.resume_from_checkpoint.lower() == 'true'
    args.continued_training = args.continued_training.lower() == 'true'
    args.Gpu = args.Gpu.lower() == 'true'
    args.permute_training_polys = args.permute_training_polys.lower() == 'true'
    args.triang_shuffling = args.triang_shuffling.lower() == 'true'
    args.permute_evaluation_polys_for_each_triang = args.permute_evaluation_polys_for_each_triang.lower() == 'true'
    args.permute_evaluation_polys_once = args.permute_evaluation_polys_once.lower() == 'true'
    args.sample_polys_uniformly_for_evaluation = args.sample_polys_uniformly_for_evaluation.lower() == 'true'
    args.sample_polys_wrt_number_of_triangs_for_evaluation = args.sample_polys_wrt_number_of_triangs_for_evaluation.lower() == 'true'
    args.no_rl = args.no_rl.lower() == 'true'
    args.permute_polys_for_new_data_generation = args.permute_polys_for_new_data_generation.lower() == 'true'
    args.permute_polys_for_each_triang_for_new_data_generation = args.permute_polys_for_each_triang_for_new_data_generation.lower() == 'true'
    args.sample_polys_uniformly_for_new_data_generation = args.sample_polys_uniformly_for_new_data_generation.lower() == 'true'

    # Populate parameter objects
    model_params = ModelParams()
    model_params.d_model = args.d_model
    model_params.num_heads = args.num_heads
    model_params.num_layers = args.num_layers
    model_params.d_ff = args.d_ff
    model_params.d_ff_enem = args.d_ff_enem
    model_params.dropout = args.dropout

    encoding_params = encoding_parameters(args.n_vertices)

    training_params = TrainingParams()
    training_params.n_vertices = args.n_vertices
    training_params.split_by_polytope = args.split_by_polytope
    training_params.max_number_triangs_per_polytope = args.max_number_triangs_per_polytope
    training_params.n_train = args.n_train
    training_params.n_val = args.n_val
    training_params.n_test = args.n_test
    training_params.batch_size = args.batch_size
    training_params.grad_acc_period = args.grad_acc_period
    training_params.n_steps = args.n_steps
    training_params.loss_eval_frequency = args.loss_eval_frequency
    training_params.checkpoint_frequency = args.checkpoint_frequency
    training_params.frst_gen_eval_frequency = args.frst_gen_eval_frequency
    training_params.N_polys_monitoring = args.N_polys_monitoring
    training_params.numOfTs_per_poly = args.numOfTs_per_poly
    training_params.permute_training_polys = args.permute_training_polys
    training_params.triang_shuffling = args.triang_shuffling
    training_params.permute_evaluation_polys_for_each_triang = args.permute_evaluation_polys_for_each_triang
    training_params.permute_evaluation_polys_once = args.permute_evaluation_polys_once
    training_params.sample_polys_uniformly_for_evaluation = args.sample_polys_uniformly_for_evaluation
    training_params.sample_polys_wrt_number_of_triangs_for_evaluation = args.sample_polys_wrt_number_of_triangs_for_evaluation
    training_params.lr = args.lr
    training_params.scheduler = args.scheduler
    training_params.beta1 = args.beta1
    training_params.beta2 = args.beta2
    training_params.eps = args.eps
    training_params.random_seed = args.random_seed
    training_params.resume_from_checkpoint = args.resume_from_checkpoint


    job_params = JobParams()
    job_params.folder_name = args.folder_name
    job_params.exp_name = args.exp_name
    job_params.continued_training = args.continued_training
    job_params.max_time = args.max_time
    job_params.polys_file_train = args.polys_file_train
    job_params.triangs_file_train = args.triangs_file_train
    job_params.polys_file_val = args.polys_file_val
    job_params.triangs_file_val = args.triangs_file_val
    job_params.polys_file_test = args.polys_file_test
    job_params.triangs_file_test = args.triangs_file_test
    job_params.num_cpus = args.num_cpus
    job_params.Gpu = args.Gpu

    RL_params = RLParams()
    RL_params.rl_data_folder = args.rl_data_folder
    RL_params.n_iterations = args.n_iterations
    RL_params.n_polys_for_guesses = args.n_polys_for_guesses
    RL_params.n_triangs_per_poly_for_guesses = args.n_triangs_per_poly_for_guesses
    RL_params.no_rl = args.no_rl
    RL_params.initial_number_of_triangs_per_poly = args.initial_number_of_triangs_per_poly
    RL_params.permute_polys_for_new_data_generation = args.permute_polys_for_new_data_generation
    RL_params.permute_polys_for_each_triang_for_new_data_generation = args.permute_polys_for_each_triang_for_new_data_generation
    RL_params.sample_polys_uniformly_for_new_data_generation = args.sample_polys_uniformly_for_new_data_generation

    return model_params, encoding_params, training_params, job_params, RL_params




