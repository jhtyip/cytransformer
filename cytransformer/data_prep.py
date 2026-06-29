"""CYTools-free data preparation.

Pair, shuffle and split a GIVEN dataset of polytopes and triangulations into
aligned train/val/test JSON files that cyt consumes. No CYTools: this only
reshapes JSON that the generation tools (or you) produced.
"""
"""
Raw data generation and dataset splitting for CYTransformer.

Two responsibilities live here:
    database and enumerate/sample their FRST (Fine Regular Star) triangulations,
    serialising both into the same JSON layout as the reference 005.poly.json /
    005.triang.json files (one entry per artefact, keyed by a shared POLYID).
  - Preparing data for training: loading those JSON files, filtering to
    polytopes with the desired vertex count, optionally randomising order while
    keeping each polytope's triangulations grouped, and splitting into
    train/val/test sets on a per-polytope basis (so no polytope leaks across
    splits).

The (polytope, triangulation) string encoding used here is the one the
``Polys_triangs_dataset`` loader parses, so it is part of the frozen data
contract.
"""

import numpy as np
from collections import defaultdict
from itertools import groupby, combinations
import json
import random
import os
import time
import sys


def process_json_files(filepath_polys, filepath_triangs, N, randomize=True, seed=0):
    """
    N as in N_vert=N+1, for removing polytopes with the wrong number of vertices
    """
    with open(filepath_polys, 'r') as f:
        data_polys = json.load(f)
    with open(filepath_triangs, 'r') as f:
        data_triangs = json.load(f)

    # Index polytopes by POLYID so each triangulation can be paired with its polytope.
    toSave_polys = []
    lookup_dict_polys = {}
    for poly in data_polys:
        lookup_dict_polys[poly['POLYID']] = poly['DRESVERTS']
    toSave_triangs = []
    for triang in data_triangs:
        POLYID = triang['POLYID']
        poly = lookup_dict_polys[POLYID]
        # Keep only polytopes with exactly N+1 vertices: the vertex-string contains
        # N-1 "},{" separators (N+1 vertices, minus the implicit origin).
        if poly.count('},{') == N - 1:
            toSave_polys.append([POLYID, poly])
            toSave_triangs.append([POLYID, triang['TRIANG']])


    if randomize:
        random.seed(seed)

        # Shuffle the (poly, triang) pairs, then shuffle again at the polytope-group
        # level. This randomises both the order of polytopes and the order of
        # triangulations within a polytope, while keeping a polytope's triangulations
        # contiguous (so the later per-polytope split stays clean).
        indices = list(range(len(toSave_triangs)))
        random.shuffle(indices)
        toSave_triangs = [toSave_triangs[i] for i in indices]
        toSave_polys = [toSave_polys[i] for i in indices]

        group_dict = defaultdict(list)
        for poly, triang in zip(toSave_polys, toSave_triangs):
            group_dict[poly[0]].append((poly, triang))
        grouped = list(group_dict.values())
        random.shuffle(grouped)
        toSave_polys, toSave_triangs = zip(*[pair for group in grouped for pair in group])
        toSave_polys = list(toSave_polys)
        toSave_triangs = list(toSave_triangs)

        print("The ordering of polytopes and triangulations is randomized.")

    print(f"Total number of favorable polytopes (h11+4+1=N_vert=N+1): {len(group_dict)}")
    print(f"Total number of triangulations: {len(toSave_triangs)}")

    return toSave_polys, toSave_triangs



def split_data(toSave_polys, toSave_triangs, N_train, N_val, N_test, max_triangs_per_poly, verbose = False):
    """
    N_i looks like [start, end], where start/end indicates the (n+1)-th polytope, then total number of polytopes=end-start
    If max_triangs_per_poly=-1, all triangs are taken
    """
    # Group consecutive (poly, triang) pairs by POLYID, optionally capping the
    # number of triangulations kept per polytope.
    groups = []
    for key, group in groupby(zip(toSave_polys, toSave_triangs), key=lambda x: x[0][0]):
        polys_and_triangs = list(group)
        if max_triangs_per_poly == -1:
            groups.append(polys_and_triangs)
        else:
            groups.append(polys_and_triangs[:max_triangs_per_poly])

    if verbose:
        if max_triangs_per_poly == -1:
            print("Using all triangulations for each polytope.")
        else:
            print(f"Using maximum {max_triangs_per_poly} triangulations for each polytope.")

    # Carve out train/val/test by selecting disjoint ranges of polytope groups,
    # then flatten each selected range back into flat poly / triang lists.
    polys_train = []
    triangs_train = []
    for i in range(N_train[0], min(N_train[1], len(groups))):
        polys_train.extend([poly for poly, triang in groups[i]])
        triangs_train.extend([triang for poly, triang in groups[i]])
    if verbose:
        print(f"Polytope {N_train[0]} - {min(N_train[1], len(groups))} for training; {min(N_train[1], len(groups)) - N_train[0]} polytopes & {len(triangs_train)} triangulations in total.")

    polys_val = []
    triangs_val = []
    for i in range(N_val[0], min(N_val[1], len(groups))):
        polys_val.extend([poly for poly, triang in groups[i]])
        triangs_val.extend([triang for poly, triang in groups[i]])
    if verbose:
        print(f"Polytope {N_val[0]} - {min(N_val[1], len(groups))} for validation; {min(N_val[1], len(groups)) - N_val[0]} polytopes & {len(triangs_val)} triangulations in total.")

    polys_test = []
    triangs_test = []
    for i in range(N_test[0], min(N_test[1], len(groups))):
        polys_test.extend([poly for poly, triang in groups[i]])
        triangs_test.extend([triang for poly, triang in groups[i]])
    if verbose:
        print(f"Polytope {N_test[0]} - {min(N_test[1], len(groups))} for testing; {min(N_test[1], len(groups)) - N_test[0]} polytopes & {len(triangs_test)} triangulations in total.")

    return polys_train, triangs_train, polys_val, triangs_val, polys_test, triangs_test


def split_data_into_three_files(filepath_polys, filepath_triangs, filepath_train_polys, filepath_train_triangs, filepath_val_polys, \
                                 filepath_val_triangs, filepath_test_polys, filepath_test_triangs, N_vertices, N_train, N_val, N_test, randomize=True, seed=0):
    """
    End-to-end helper: load the raw poly/triang JSON, filter and (optionally)
    randomise it, split it into train/val/test by polytope range, and write the
    six resulting JSON files (polys + triangs for each split).
    """
    print("Splitting data into three files...")
    print("Source files:")
    print(f"Polytopes: {filepath_polys}")
    print(f"Triangulations: {filepath_triangs}")
    print(f"N_train: {N_train}")
    print(f"N_val: {N_val}")
    print(f"N_test: {N_test}")
    print("Destination files:")
    print(f"Train Polytopes: {filepath_train_polys}")
    print(f"Train Triangulations: {filepath_train_triangs}")
    print(f"Validation Polytopes: {filepath_val_polys}")
    print(f"Validation Triangulations: {filepath_val_triangs}")
    print(f"Test Polytopes: {filepath_test_polys}")
    print(f"Test Triangulations: {filepath_test_triangs}")
    toSave_polys, toSave_triangs = process_json_files(filepath_polys, filepath_triangs, N_vertices, randomize=randomize, seed=seed)
    polys_train, triangs_train, polys_val, triangs_val, polys_test, triangs_test = split_data(toSave_polys, toSave_triangs, N_train, N_val, N_test, max_triangs_per_poly = -1)
    # make sure that the destination folders exist
    os.makedirs(os.path.dirname(filepath_train_polys), exist_ok=True)
    with open(filepath_train_polys, 'w') as f:
        json.dump(polys_train, f, indent=2)
    with open(filepath_train_triangs, 'w') as f:
        json.dump(triangs_train, f, indent=2)
    with open(filepath_val_polys, 'w') as f:
        json.dump(polys_val, f, indent=2)
    with open(filepath_val_triangs, 'w') as f:
        json.dump(triangs_val, f, indent=2)
    with open(filepath_test_polys, 'w') as f:
        json.dump(polys_test, f, indent=2)
    with open(filepath_test_triangs, 'w') as f:
        json.dump(triangs_test, f, indent=2)
    print("Done")



def merge_json_chunks(folder: str, mystring: str):
    """
    Concatenate the per-chunk JSON files written during generation (named
    "{mystring}_<index>.json") back into a single "{mystring}.json" list, in
    ascending chunk-index order.
    """
    # Collect the chunk files and order them by the trailing numeric index.
    chunk_files = sorted(
        [f for f in os.listdir(folder) if f.startswith(f"{mystring}") and f.endswith(".json")],
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    merged_list = []
    for file in chunk_files:
        file_path = os.path.join(folder, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                merged_list.extend(data)
            else:
                print(f"Warning: {file} does not contain a list.")

    print(f"Total number of entries in the merged data {len(merged_list)}")

    output_file = os.path.join(folder, f"{mystring}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_list, f, indent=2)


    print(f"Merged {len(chunk_files)} files into {output_file}")



"""
    The two functions below generate data in the same format as 005.triang.json and 005.poly.json
    polys_file and triangs_file must end in .json
    mode: all, fast, fair. If all, N_triangs_per_poly is unused
    NOTE: the final number of polytopes might be smaller than N_poly if we are very unlucky
    NOTE: as in 005.triang.json and 005.poly.json, each polytope can have the same polyid as several associated triangulations
    NOTE: currently fetching polytopes with lattice = N, without dualizing, and getting triangulations with options include_points_interior_to_facets = False and star_origin = 0 (why these matter is rather unclear)
    NOTE: currently NOT testing whether the polytopes have the correct number of vertices etc. - this is done in the data loader (to facilite comparisons between datasets)
    TODO: understand the randomness of fetch_polytopes
"""




