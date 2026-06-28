"""
Helper utilities for CYTransformer: data loading/sampling, distributed-training
setup, output saving, and the format-translation routines that convert between
the several ways a triangulation can be represented.

The translation helpers (token <-> vertex-index <-> vertex-coordinate, plus the
permutation remapping) implement the frozen tokenisation contract used to train
the checkpoints, so their numeric conventions must not change. The block comment
just below documents the naming convention for each representation.
"""

import numpy as np
import itertools
import math
import os
import torch
import datetime
import torch.distributed as dist
from torch.utils.data import default_collate
from Polys_triangs_dataset import Polys_triangs_dataset
import time

import json

"""
Encodings of triangulations (nuance between lists and arrays not always respected):
tokens_triang: (Polytope +) list of at most L tokens (e.g. output of transformer) (each token = 4-uple of indices of vertices in the polytope) (possibly with padding)  (L)
vert_indices_woo_unshifted_triang: (Polytope +) list of at most L 4-uples of indices (starting at 0) of vertices in the polytope (possibly with padding) (L, 4) (without the origin)
vert_indices_woo_triang: (Polytope +) list of at most L 4-uples of indices (starting at 1) of vertices in the polytope (possibly with padding) (L, 4) (without the origin)
vert_indices_wo_triang: (Polytope +) list of at most L 5-uples of indices (starting at 1) of vertices in the polytope (possibly with padding) (L, 5) (with the origin, corresponding to index 0)
vert_coord_woo_triang: List of numpy arrays of shape (4,4) (4 vertices with 4 coordinates) (without the origin)
vert_coord_wo_triang: List of numpy arrays of shape (5,4) (5 vertices with 4 coordinates) (with the origin)
"""

def get_np_input_polytopes_and_masks_from_file(polys_file, n_vertices, num_of_polys, unique_sampling = True, permutation = True, verbose = False):
    """Load up to ``num_of_polys`` polytopes (no triangulations) as numpy arrays.

    The file may contain repeated polytopes; ``unique_sampling`` deduplicates
    first. The polytopes are shuffled, truncated to ``num_of_polys``, optionally
    vertex-permuted (a data-augmentation no-op the model should be invariant to),
    and returned with an all-ones mask (every vertex is real). The polytopes in
    the file can be with repetition.
    """
    with open(polys_file, 'r') as f:
        unprocessed_polys = json.load(f)

    # Parse each "{{x,y,z,w},{...}}" vertex string into an (n_vertices, 4) int array.
    unprocessed_polys = np.array([np.array([np.array(vertex.split(",")).astype("int") for vertex in poly[1][2:-2].split("},{")]) for poly in unprocessed_polys])
    if unique_sampling:
        # keep only unique polytopes
        unprocessed_polys = np.unique(unprocessed_polys, axis=0)
        if verbose:
            print("Sampling unique polytopes uniformly")
            print(f"Number of unique polytopes: {len(unprocessed_polys)} found in {polys_file}")
    else:
        if verbose:
            print("Sampling non-unique polytopes uniformly with repetition")
            print(f"Number of non-unique polytopes: {len(unprocessed_polys)} found in {polys_file}")

    # shuffle polys
    np.random.shuffle(unprocessed_polys)
    # select the first num_of_polys
    unprocessed_polys = unprocessed_polys[:num_of_polys,:,:]
    polys = []
    if permutation:
        # Apply an independent random vertex permutation to each polytope.
        count = 0
        for poly in unprocessed_polys:
            verPerm_idx = np.random.permutation(n_vertices)
            polys.append( poly[verPerm_idx,:])
        polys = np.array(polys)
    else:
        polys = unprocessed_polys

    poly_masks = np.ones((num_of_polys, n_vertices), dtype=int)
    return polys, poly_masks

def get_tensor_input_polytopes_triangs_and_masks_from_file(polys_file, triangs_file, n_vertices, max_seq_length_tgt, num_of_samples, permutation = True, triang_shuffling = True, verbose = False):
    """Assumes that the files have been processed in the 'split' format, i.e. entries in polys_file and triangs_file correspond to each other
        Samples polytopes wrt to the number of their triangulations
    """

    with open(polys_file, 'r') as f:
        unprocessed_polys = json.load(f)
    with open(triangs_file, 'r') as f:
        unprocessed_triangs = json.load(f)
    assert len(unprocessed_polys) == len(unprocessed_triangs), "Polytopes and triangulations files must have the same number of entries"


    # Shuffle poly/triang pairs together so the correspondence is preserved.
    indices = np.arange(len(unprocessed_polys))
    np.random.shuffle(indices)
    unprocessed_polys = [unprocessed_polys[i] for i in indices]
    unprocessed_triangs = [unprocessed_triangs[i] for i in indices]
    unprocessed_triangs = unprocessed_triangs[:num_of_samples]
    unprocessed_polys = unprocessed_polys[:num_of_samples]


    # Encode through the dataset, then collate the per-sample tensors into batches.
    dataset = Polys_triangs_dataset(unprocessed_polys, unprocessed_triangs, n_vertices, max_seq_length_tgt, permutation, triang_shuffling=triang_shuffling, seed=0)
    data = [sample for sample in dataset]

    polys = default_collate([d[0] for d in data])
    triangs = default_collate([d[1] for d in data])
    poly_masks = default_collate([d[2] for d in data])

    return polys, triangs, poly_masks


def file_barrier(barrier_dir, num_procs, proc_id):
    """Filesystem-based barrier synchronising ``num_procs`` processes/nodes.

    Each process drops a marker file and waits until all markers exist; rank 0
    then cleans them up. Used where a shared filesystem is available but a proper
    distributed group is not (e.g. across nodes before/after a sync point).
    """
    os.makedirs(barrier_dir, exist_ok=True)
    # Each process creates its own file
    open(os.path.join(barrier_dir, f"barrier_{proc_id}"), "w").close()
    # Wait for all files to appear
    while len(os.listdir(barrier_dir)) < num_procs:
        time.sleep(1)
    time.sleep(2)
    # Clean up
    if proc_id == 0:
        for fname in os.listdir(barrier_dir):
            os.remove(os.path.join(barrier_dir, fname))
    time.sleep(1)
    print(f"Barrier {barrier_dir} passed by node {proc_id}")
    time.sleep(1)

def save_output(polys_monitoring, poly_masks_monitoring, toSave_Ts_new, save_path):
    """
    Save the output of the monitoring function
    Args:
        polys_monitoring: numpy array of shape (N, N_vertices - 1, 4) or (NM, N_vertices - 1, 4)
        poly_masks_monitoring: numpy array of shape (N, N_vertices - 1)
        toSave_Ts_new: numpy array of shape (N, M, T) (where T = 35 is an upper bound on the number of simplices) (tokens_triang)
        save_path: path to save the output
    """
    dir_path = os.path.dirname(save_path)
    os.makedirs(dir_path, exist_ok=True)
    np.save(save_path + "_polys_monitoring.npy", polys_monitoring)
    np.save(save_path + "_poly_masks_monitoring.npy", poly_masks_monitoring)
    np.save(save_path + "_toSave_Ts_new.npy", toSave_Ts_new)


def token_encoder():
    # TODO
    return 0

def token_decoder(token, N_vertices):
    """
    token is an integer
    If token belongs to [0, comb(N_vertices, 4)-1], it encodes the corresponding simplex in np.array(list(itertools.combinations(np.arange(N_vertices), 4)))
    (e.g., 1 -> [0 1 2 4]))
    If it is greater or smaller, it is padding and should not be decoded
    """
    simplexList = np.array(list(itertools.combinations(np.arange(N_vertices), 4)))
    return simplexList[token]

def apply_permutation_to_tokens_triang(tokens_triang, perm, N_vertices, padding_idx = None):
    """Relabel a token triangulation under a vertex permutation ``perm``.

    triang is an array of length L of tokens (tokens_triang), e.g. [0, 13, 21, ...].
    Builds a lookup that maps each simplex token to the token of its permuted
    (then re-sorted) vertex set, leaving the three special tokens fixed, and
    applies it to every token in the input. Could maybe speed up?
    """
    if padding_idx == None:
        padding_idx = math.comb(N_vertices, 4) + 1 + 1
    # Special tokens (<sos>, <eos>, <pad>) map to themselves.
    permuted_token_dictionary = {token:token for token in [padding_idx-2, padding_idx-1, padding_idx]}
    simplexList = np.array(list(itertools.combinations(np.arange(N_vertices), 4)))
    # For each simplex token, look up the token of its permuted, re-sorted vertices.
    for token in range( math.comb(N_vertices, 4)):
        token_as_simplex_list = simplexList[token]
        permuted_simplex_list = perm[token_as_simplex_list]
        permuted_simplex_list.sort(0)
        permuted_token = int(np.where((simplexList == permuted_simplex_list).all(1))[0])
        permuted_token_dictionary[token] = permuted_token
    permuted_triang = np.array([permuted_token_dictionary[token.item()] for token in tokens_triang])
    return permuted_triang


def tokens_triang_to_vert_indices_woo_triang(tokens_triang, N_vertices, padding_idx = None):
    """
        triang is an array of length L of tokens (tokens_triang)
        Example: [0, 13, 21, ...]
        output: triangulation as a list of length length_triangulation of numpy arrays of shape (4).
        Each entry corresponds to a simplex (it contains the indices of its 4 vertices (different from the origin) among the numOfVer (+1 not included) vertices of a polytope) (vert_indices_triang)
        The indices start at 0 (which does not correspond to the origin)
        Example: [[0 1 2 3], [0 2 4 5], ...]
     """
    if padding_idx == None:
        padding_idx = math.comb(N_vertices, 4) + 1 + 1
    vert_indices_triang = []
    for simplex_ind in tokens_triang:
        # Skip the special tokens; decode only real simplex tokens.
        if simplex_ind not in [padding_idx-2, padding_idx-1, padding_idx]:
            vert_indices_triang.append(token_decoder(simplex_ind, N_vertices))
    return vert_indices_triang

def vert_indices_woo_triang_to_vert_indices_wo_triang(vert_indices_woo_triang):
    """
    shifts every index by 1 and adds the origin in every simplex
    """
    vert_indices_wo_triang = [np.insert(simplex+1, 0, 0)  for simplex in  vert_indices_woo_triang ]
    return vert_indices_wo_triang


def tokens_triang_to_vert_indices_wo_triang(tokens_triang, N_vertices, padding_idx = None):
    """
        triang is an array of length L of tokens (tokens_triang)
        Example: [0, 13, 21, ...]
        output: triangulation as a list of length L of numpy arrays of shape (5). Each entry corresponds to a simplex (it contains the indices of its 4 vertices (different from the origin) among the numOfVer (+1 not included) vertices of a polytope) (vert_indices_triang)
        The indices start at 1, with 0 being the origin (included)
        Example: [[0 1 2 3 8], [0 2 4 5 7], ...]
     """
    vert_indices_woo_triang = tokens_triang_to_vert_indices_woo_triang(tokens_triang, N_vertices, padding_idx)
    return vert_indices_woo_triang_to_vert_indices_wo_triang(vert_indices_woo_triang)


def vert_indices_triang_to_vert_coord_triang(polytope, triang):
    """
        polytope of shape (N_vertices-1, 4) (the list of its vertices)
        triang of shape (L, 4), where L is an upper bound on the number of simplices.
        Each row corresponds to a simplex (it contains the indices of its 4 or 5 vertices in the list given by polytope), except the rows with only -1 which are padding (vert_indices_triang)
        Output: a list of numpy arrays (4,4), where each array represents a simplex (each row is a vertex given by its 4 coordinates) (vert_coord_woo_triang)
        NOTE: the function is agnostic to whether the origin is included in the polytope and the triangulation (it simply selectes the vertices of the polytope given by the indices)
    """
    vert_coord_woo_triang = []
    triang = np.array(triang)
    for i in range(triang.shape[0]):
        # Skip padding rows (all -1); gather the coordinates of each simplex's vertices.
        if not (-1 in triang[i].tolist()):
            simplex = np.array([polytope[index] for index in triang[i].tolist() ])
            vert_coord_woo_triang.append(simplex)
    return vert_coord_woo_triang

def vert_coord_woo_triang_to_vert_coord_wo_triang(vert_coord_woo_triang, dimension):
    """
        adds [0, 0, 0, 0] to each simplex
    """
    vert_coord_wo_triang = [ np.concatenate( (simplex, np.zeros((1,dimension), dtype = int) ) ) for simplex in vert_coord_woo_triang ]
    return vert_coord_wo_triang


def setup(rank, world_size):
    """Initialise the NCCL distributed process group for this ``rank``.

    Sets the master address/port (port derived from the SLURM job id to avoid
    collisions between concurrent jobs), pins the CUDA device, and joins the
    process group with a long timeout to tolerate slow collective operations.
    """
    os.environ["NCCL_TIMEOUT"] = "36000"
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id is not None:
        slurm_job_id = int(slurm_job_id)
        if rank ==0:
            print(f"SLURM Job ID: {slurm_job_id}")
    else:
        if rank ==0:
            print("SLURM_JOB_ID is not set.")
        slurm_job_id = 0
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(20000 + slurm_job_id%10000)
    torch.cuda.set_device(rank)
    _ = torch.tensor(0).cuda()
    # Initialize the process group
    dist.init_process_group("nccl",
                            init_method='env://',
                            rank=rank,
                            world_size=world_size,
                            timeout=datetime.timedelta(seconds=36000))

def cleanup():
    """Clean up the distributed process group."""
    dist.destroy_process_group()



if __name__ == "__main__":
    # Smoke check: decode the first saved triangulation back to vertex indices.

    # For N_vert=10+1
    padding_idx = 212
    poly = np.load("Data/Test_FRST_check/toSave_poly_fast_200.npy") # (200, 1, 10, 4)
    poly_mask = np.load("Data/Test_FRST_check/toSave_poly_mask_fast_200.npy") # (200, 1, 10)
    Ts = np.load("Data/Test_FRST_check/toSave_Ts_fast_200.npy") #  (200, 200, 35) (200 triangulations per polytope) (tokens_triang)

    print(f"First triangulation as tokens {Ts[0,0,:]}")
    vert_indices_wo_triang = tokens_triang_to_vert_indices_wo_triang(Ts[0,0,:], 10)
    print(f"First triangulation as vertices indices with origin included {vert_indices_wo_triang}")
