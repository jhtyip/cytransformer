"""
Self-improvement ("RL") training loop for the CYTransformer.

Each iteration: train the model on the current data, generate fresh
triangulations from the resulting checkpoint, keep only those that are valid
FRSTs, merge the newly discovered ones (deduplicated) back into the dataset, and
repeat. The growing pool of model-found FRSTs is what drives improvement -- the
model is effectively trained on its own best discoveries.

Multi-node aware: GPU work is spawned per rank, while the CPU-side data
bookkeeping (evaluation, merging, saving) is done only on node 0, with
file-based barriers keeping the nodes in step.
"""

import os

import torch

import sys

import numpy as np

import time

import torch.multiprocessing as mp

from cytransformer.validation.monitoring import monitor_perf, parallel_monitor_perf
from cytransformer.validation.generate_triangs_from_checkpoint import generate_triangs_from_checkpoint
import json
import random
import itertools
from cytransformer.data_generation import split_data
from cytransformer.utilities import tokens_triang_to_vert_indices_woo_triang, file_barrier

from cytransformer.args import ModelParams, EncodingParams, TrainingParams, JobParams, RLParams, parse_arguments
from cytransformer.train import train


def RL_train(world_size, node_rank, num_nodes, gpus_per_node, model_params: ModelParams, encoding_params: EncodingParams, training_params: TrainingParams, job_params: JobParams, RL_params: RLParams):
    """

    start with initial data folder (copy of a preexisting one)
    train model on it (with saved outputs etc)
    checkpoint model
    sample from model
    evaluate samples
    add to the initial data folder (mix everything)
    start training again (same data path) from checkpoint (start from old number of epochs)
    Only the first node will execute the non-gpu operations
    """
    np.random.seed(training_params.random_seed)
    random.seed(training_params.random_seed)
    torch.manual_seed(training_params.random_seed)

    starting_time = time.time()
    if node_rank == 0:
        print("Starting RL training")
        print(f"Number of GPUs: {world_size}")
        print("RL parameters:")
        RL_params.display()

    rl_data_folder = RL_params.rl_data_folder + f"/{job_params.exp_name}"

    # Resume from the latest iteration if this RL run already has data on disk;
    # otherwise bootstrap iteration 0 from the seed dataset.
    if os.path.exists(rl_data_folder) and len(os.listdir(rl_data_folder)) > 0:
        # find current iteration
        iteration = max([int(iteration_folder_name.split("_")[1]) for iteration_folder_name in os.listdir(rl_data_folder)])
        if node_rank == 0:
            print(f"Continuing from a previous RL training run, current iteration: {iteration}")
    else:
        iteration = 0
        if node_rank == 0:
            print("This is a new RL training run")
            ### Copy data from the data folder to the RL data folder, keep only the correct amount of triangs per polytope (to make sure no cheating is possible)
            print(f"Extracting the training data from the data folder to the RL data folder {rl_data_folder}")
            os.makedirs(f"{rl_data_folder}/iteration_0", exist_ok=True)
            with open(job_params.triangs_file_train) as f:
                train_triangs = json.load(f)
            with open(job_params.polys_file_train) as f:
                train_polys = json.load(f)
            polys_train, triangs_train, _, _, _, _ = split_data(train_polys, train_triangs, [0,min(len(train_polys),training_params.n_train)], [0,0], [0,0], RL_params.initial_number_of_triangs_per_poly)
            print(f"Keeping {RL_params.initial_number_of_triangs_per_poly} triangulations per polytope and {training_params.n_train} polytopes. Total number of triangulations saved: {len(triangs_train)}")
            with open(f"{rl_data_folder}/iteration_0/polys.json", 'w') as f:
                json.dump(polys_train, f, indent=2)
            with open(f"{rl_data_folder}/iteration_0/triangs.json", 'w') as f:
                json.dump(triangs_train, f, indent=2)

    time.sleep(3)  # Wait for a few seconds to ensure the files are copied

    # Each RL iteration trains for this many additional steps.
    n_steps_per_iteration = training_params.n_steps

    while iteration < RL_params.n_iterations:
        # File-based barrier so all nodes enter the iteration together.
        file_barrier(rl_data_folder + "barriers/barrier_start_iteration", num_nodes, node_rank)
        if node_rank == 0:
            print(f"\n\n############\nStarting RL iteration {iteration}")
        iteration_starting_time = time.time()
        # Point training at this iteration's data and set the cumulative step target
        # (training resumes from the previous checkpoint, so n_steps keeps growing).
        job_params.polys_file_train = f"{rl_data_folder}/iteration_{iteration}/polys.json"
        job_params.triangs_file_train = f"{rl_data_folder}/iteration_{iteration}/triangs.json"
        training_params.n_steps = (iteration+1)*n_steps_per_iteration


        with open(job_params.polys_file_train) as fp:
            polys = json.load(fp)  # FRSTs
        with open(job_params.triangs_file_train) as fp:
            triangs = json.load(fp)
        if node_rank == 0:
            print(f"Current number of polytopes: {len(set([poly[0] for poly in polys]))}")
            print(f"Current number of triangulations: {len(triangs)}")

        train_time = time.time()
        ### Train the model
        # Spawn one training process per GPU (or run inline on a single GPU). The
        # iteration==0 flag is passed as the verbose argument of train().
        if node_rank == 0:
            print(f"\n##\nTraining the model")
        if world_size > 1:
            mp.spawn(train, args=(world_size, node_rank, gpus_per_node, model_params, encoding_params, training_params, job_params, iteration==0), nprocs=gpus_per_node, join=True)
        else:
            train(0, world_size, node_rank, gpus_per_node, args[0], args[1], args[2], args[3], iteration==0)

        file_barrier(rl_data_folder + "barriers/barrier_end_of_training", num_nodes, node_rank)

        if node_rank == 0:
            print(f"Training time: {np.round((time.time()-train_time)/3600, decimals=2)} h")
        if not RL_params.no_rl:
            ### Generate triangulations
            # Use the freshest (highest-step) checkpoint just produced by training.
            checkpoints_folder = f'Checkpoints/{job_params.folder_name}/{job_params.exp_name}'
            last_epoch = max([int(checkpoint_name.split("-")[1]) for checkpoint_name in os.listdir(checkpoints_folder)])
            checkpoint_path = f'{checkpoints_folder}/chkpt-{last_epoch}'
            if node_rank == 0:
                print("\n##\nGenerating new training data with checkpoint ", checkpoint_path)


            new_data_folder = f"{rl_data_folder}/iteration_{iteration}/new_data"
            generation_args = (checkpoint_path,
                                    RL_params.n_polys_for_guesses,
                                    RL_params.n_triangs_per_poly_for_guesses,
                                    job_params.polys_file_train, # the current polytopes
                                    new_data_folder+"/new_data",
                                    training_params.random_seed,
                                    False, # verbose
                                    RL_params.permute_polys_for_new_data_generation, # permutation
                                    RL_params.permute_polys_for_each_triang_for_new_data_generation,  # permutation_for_each_triang
                                    RL_params.sample_polys_uniformly_for_new_data_generation  # unique_polytope_sampling
            )


            generation_starting_time = time.time()
            if world_size > 1:
                mp.spawn(generate_triangs_from_checkpoint, args=(world_size, node_rank, gpus_per_node,  *generation_args), nprocs=gpus_per_node, join=True)
            else:
                generate_triangs_from_checkpoint(0, world_size, node_rank, gpus_per_node, *generation_args)

            file_barrier(rl_data_folder + "barriers/barrier_end_of_generation", num_nodes, node_rank)
            if node_rank == 0:
                print(f"Generation time: {np.round((time.time()-generation_starting_time)/3600, decimals=2)} h")
            ### Evaluating the new triangulations
            if node_rank == 0: # CPU parallelism, only with the first node for now (this is quite fast anyway)
                # TODO perhaps delete the intermediate data
                print(f"\n##\nEvaluating the new triangulations with node {node_rank}")

                evaluation_starting_time = time.time()

                # Load the just-generated samples and check which are valid FRSTs.
                # NOTE: the polys are not necessarily unique (depending on RL_params.sample_polys_uniformly_for_new_data_generation)
                new_polys = np.load(new_data_folder+ "/new_data_polys_monitoring.npy") # of shape (n_polys, n_vertices, 4)
                new_polys_masks = np.load(new_data_folder+ "/new_data_poly_masks_monitoring.npy")
                new_triangs = np.load(new_data_folder+ "/new_data_toSave_Ts_new.npy")   # of shape (n_polys, n_triangs, target_length)

                if world_size > 1:
                    results = parallel_monitor_perf(new_polys, new_polys_masks, new_triangs)
                else:
                    results = monitor_perf(new_polys, new_polys_masks, new_triangs)

                # Boolean mask (n_polys, n_triangs): which samples are genuine FRSTs.
                FRST_indices = results["FRST_indices"] # the important info
                print(f"Total number of triangulations: {new_triangs.shape[0]*new_triangs.shape[1]}")
                printable_results = {k: int(v) for k, v in results.items() if k != "FRST_indices"}
                print(printable_results)
                print(f"Evaluation time: {np.round((time.time()-evaluation_starting_time)/60, decimals=2)} mn")

                ### Update the data with the new triangulations
                print(f"\n##\nUpdating the data with the new triangulations with node {node_rank}")

                # Load the current triangulations and polys
                print(f"Loading the current triangulations and polys from {rl_data_folder}/iteration_{iteration}")
                with open(f"{rl_data_folder}/iteration_{iteration}/triangs.json") as fp:
                    current_triangs = json.load(fp)
                with open(f"{rl_data_folder}/iteration_{iteration}/polys.json") as fp:
                    current_polys = json.load(fp)
                # Re-encode the existing triangulations (stored as vertex-index strings)
                # into the model's token format, grouped by polytope id, so the new
                # FRSTs can be merged and deduplicated in a common representation.
                current_poly_ids = list(set([poly[0] for poly in current_polys]))
                current_triangs_dict = {id:[] for id in current_poly_ids}
                simplexList = np.array(list(itertools.combinations(np.arange(training_params.n_vertices), 4)))
                for triang in current_triangs:
                    triang_as_tokens = []
                    triang_list_simplices = [np.array(simplex.split(",")).astype("int") for simplex in triang[1][2:-2].split("},{")]
                    for simplex in triang_list_simplices:
                        simplex.sort()
                        triang_as_tokens.append(np.where((simplexList==simplex).all(1))[0][0])
                    triang_as_tokens.insert(0, encoding_params.padding_idx-2)  # <sos>
                    triang_as_tokens.append(encoding_params.padding_idx-1)  # <eos>
                    if len(triang_as_tokens) < encoding_params.max_seq_length_tgt:
                        triang_as_tokens.extend([encoding_params.padding_idx for _ in range(encoding_params.max_seq_length_tgt-len(triang_as_tokens))])  # Padding
                    current_triangs_dict[triang[0]].append(triang_as_tokens)
                # current_triangs_dict is now a dictionary with keys polyids and values lists of triangulations as lists of tokens

                # Match each generated polytope back to its dataset polyid by comparing
                # vertex sets (generation may have permuted/duplicated the polytopes).
                # Find the polyids of the polys in new_polys
                polyids_in_new_polys = []   # a list of length n_polys (not all polyids are necessarily unique)
                # process the current polys and remove duplicates
                current_polys_processed = [] # a list of pairs (poly, polyid)
                for current_poly in current_polys:
                    if current_poly[0] not in [poly[1] for poly in current_polys_processed]: # no duplicates
                        current_poly_processed = set(tuple(np.array(vertex.split(',')).astype('int') ) for vertex in current_poly[1][2:-2].split("},{"))
                        current_polys_processed.append((current_poly_processed, current_poly[0]))
                for new_poly in new_polys:
                    new_poly_processed = set(tuple(row) for row in new_poly)
                    for current_poly in current_polys_processed:
                        if new_poly_processed == current_poly[0]:
                            polyids_in_new_polys.append(current_poly[1])
                assert len(polyids_in_new_polys) == new_triangs.shape[0], f"polyids_in_new_polys not of length new_triangs.shape[0], {len(polyids_in_new_polys)} vs {new_triangs.shape[0]} ({new_triangs.shape = })"

                # Keep only the FRSTs in the new triangulations, grouped by polyid.
                new_triangs_dict = {}
                for index_poly, polyid in enumerate(polyids_in_new_polys):
                    new_FRSTs_for_poly = []
                    for index_triang, new_triang in enumerate(new_triangs[index_poly, :].tolist()):
                        if FRST_indices[index_poly, index_triang]:
                            new_FRSTs_for_poly.append(new_triang)
                    if polyid not in new_triangs_dict: # needed because polyids are not unique
                        new_triangs_dict[polyid] = new_FRSTs_for_poly
                    else:
                        new_triangs_dict[polyid] += new_FRSTs_for_poly

                # Merge the current triangulations with the new triangulations and remove duplicates
                print(f"Merging the current triangulations with the new triangulations with node {node_rank}")
                new_triangulations_found = 0
                for polyid in new_triangs_dict:  # have to use new_triangs_dict, as the polyids in polyids_in_new_polys are not unique
                    merged_triangs_for_poly = current_triangs_dict[polyid] + new_triangs_dict[polyid]
                    merged_triangs_for_poly = np.array(merged_triangs_for_poly)
                    merged_triangs_for_poly.sort(axis=1)  # Sort the simplices in each triangulation
                    # Collapse the three special tokens to a single value so duplicate
                    # triangulations compare equal regardless of sos/eos/pad placement.
                    merged_triangs_for_poly[np.isin(merged_triangs_for_poly, [encoding_params.padding_idx-2, encoding_params.padding_idx-1, encoding_params.padding_idx])] = encoding_params.padding_idx
                    merged_triangs_for_poly, _ = np.unique(merged_triangs_for_poly, return_inverse=True, axis=0)
                    # Net gain in distinct triangulations for this polytope.
                    new_triangulations_found += len(merged_triangs_for_poly) - len(current_triangs_dict[polyid])
                    assert len(merged_triangs_for_poly) - len(current_triangs_dict[polyid]) >= 0, "Some triangulations disappeared"
                    current_triangs_dict[polyid] = merged_triangs_for_poly

                print(f"New FRSTs found: {new_triangulations_found}")

                # Write the merged dataset as iteration+1's data, converting the token
                # triangulations back to the on-disk vertex-index string format.
                print(f"Saving the new data in {rl_data_folder}/iteration_{iteration+1}")
                os.makedirs(f"{rl_data_folder}/iteration_{iteration+1}", exist_ok=True)
                triangs_to_save = []
                polys_to_save = []
                current_polys_as_dictionary = {polyid: current_poly for polyid, current_poly in current_polys}
                for polyid, list_of_triangulations in current_triangs_dict.items():
                    for tokens_triang in list_of_triangulations:
                        vert_indices_wo_triang = tokens_triang_to_vert_indices_woo_triang(tokens_triang, training_params.n_vertices, encoding_params.padding_idx)
                        processed_triang = '{' + ','.join('{' + ','.join(map(str, row)) + '}' for row in vert_indices_wo_triang) + '}'
                        triangs_to_save.append([polyid, processed_triang])
                        polys_to_save.append([polyid, current_polys_as_dictionary[polyid]])

                with open(f"{rl_data_folder}/iteration_{iteration+1}/triangs.json", 'w') as f:
                    json.dump(triangs_to_save, f, indent=2)
                with open(f"{rl_data_folder}/iteration_{iteration+1}/polys.json", 'w') as f:
                    json.dump(polys_to_save, f, indent=2)

        else:
            # no_rl mode (ablation): skip generation entirely and just carry the same
            # data forward, so the model keeps training on the fixed seed dataset.
            print("Not true RL, no triangulations generated")
            os.makedirs(f"{rl_data_folder}/iteration_{iteration+1}", exist_ok=True)
            os.system(f"cp {rl_data_folder}/iteration_{iteration}/polys.json {rl_data_folder}/iteration_{iteration+1}/polys.json")
            os.system(f"cp {rl_data_folder}/iteration_{iteration}/triangs.json {rl_data_folder}/iteration_{iteration+1}/triangs.json")

        if node_rank == 0:
            print(f"\n##\nEnd of RL iteration {iteration}, iteration time elapsed: {np.round((time.time()-iteration_starting_time)/3600, decimals=2)} h")
            print(f"Total time elapsed: {np.round((time.time()-starting_time)/3600, decimals=2)} h")
        iteration += 1
    if node_rank == 0:
        print("End of RL training")






if __name__ == "__main__":

    # Read the multi-node topology from the SLURM environment (defaults to a single
    # node / no SLURM), then launch the RL self-improvement loop.
    args = parse_arguments()
    gpus_per_node = torch.cuda.device_count()
    node_rank = int(os.environ.get("SLURM_NODEID", 0)) if "SLURM_NODEID" in os.environ else 0
    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1)) if "SLURM_JOB_NUM_NODES" in os.environ else 1
    world_size = gpus_per_node * num_nodes
    print(f"Node rank: {node_rank}")
    print(f"Total number of GPUs: {world_size}")
    print(f"Total number of nodes: {num_nodes}")
    print("\n\n")
    time.sleep(3)

    RL_train(world_size, node_rank, num_nodes, gpus_per_node, args[0], args[1], args[2], args[3], args[4])

