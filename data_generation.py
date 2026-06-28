"""
Raw data generation and dataset splitting for CYTransformer.

Two responsibilities live here:
  - Generating training data: fetch reflexive polytopes from the CYTools
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
# `cytools` is imported lazily inside the data-generation functions that use it,
# so the rest of the codebase (training, inference) imports without CYTools installed.
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




def generate_triangs(n_vertices, lower_bound, upper_bound, N_triangs_per_poly, folder, polys_file, triangs_file, mode = "all", seed = 42, superset_size = None):
    """
        Generates data in the same format as 005.triang.json and 005.poly.json
        polys_file and triangs_file must end in .json
        mode: all, fast, fair. If all, N_triangs_per_poly is unused
        If superset_size is not None, the function will first fetch the first superset_size polytopes, shuffle them (using the random seed), then take the polytopes in the range [lower_bound, upper_bound] of the shuffled list
        NOTE: the final number of polytopes might be smaller than N_poly if we are very unlucky
        NOTE: as in 005.triang.json and 005.poly.json, each polytope can have the same polyid as several associated triangulations
        NOTE: currently fetching polytopes with lattice = N, without dualizing, and getting triangulations with options include_points_interior_to_facets = False and star_origin = 0 (why these matter is rather unclear)
        NOTE: currently NOT testing whether the polytopes have the correct number of vertices etc. - this is done in the data loader (to facilite comparisons between datasets)
        TODO: understand the randomness of fetch_polytopes
        NOTE: the polytope indices will not be fully contiguous (due to not every polytope between lower_bound and upper_bound being valid)
    """
    print("Triangulations generation process starting")
    assert 0<= lower_bound and lower_bound < upper_bound, "lower_bound must be positive and lower than upper_bound"
    # The CYTools database is queried by Hodge number h11; for these 4D reflexive
    # polytopes h11 = n_vertices - 4 (n_vertices counts the resolved vertices).
    if n_vertices == 9:
        h11 = 5
    elif n_vertices == 10:
        h11 = 6
    elif n_vertices == 11:
        h11 = 7
    elif n_vertices == 12:
        h11 = 8
    elif n_vertices == 13:
        h11 = 9
    elif n_vertices == 14:
        h11 = 10
    elif n_vertices == 15:
        h11 = 11
    elif n_vertices == 20:
        h11 = 16
    else:
        print("Unexpected n_vertices")

    if not os.path.exists(folder):
        os.makedirs(folder)

    np.random.seed(seed)
    random.seed(seed)

    from cytools import fetch_polytopes, Polytope  # lazy import (see note at top of file)
    print("Fetching polytopes")
    # Fetch a pool of candidate polytopes. With superset_size set we draw from a
    # larger pool first (so the shuffle below samples from a wider population)
    # before slicing to the requested [lower_bound, upper_bound) window.
    if not superset_size:
        print(f"Fetching among the first {upper_bound} polytopes")
        gen = fetch_polytopes(h11=h11, lattice="N", limit=upper_bound, as_list=True)
    else:
        print(f"Fetching among the first {superset_size} polytopes")
        gen = fetch_polytopes(h11=h11, lattice="N", limit=superset_size, as_list=True)

    random.shuffle(gen)
    gen = gen[lower_bound:upper_bound]
    assert gen, "Not enough polytopes found in the range"
    print(f"Initial number of candidate polytopes : {len(gen)}")
    print(f"Number of polytopes with the correct number of vertices : {sum([1 for poly in gen if len(poly.points()) == n_vertices + 1])}")
    polys = []  # Polytopes
    triangs = []  # FRSTs
    numTri = []  # Number of triangulations per polytope

    i = 0
    print(f"Generating triangulations using the {mode} method")
    start_time = time.time()
    while i < len(gen):
        poly = gen[i]
        # Obtain triangulations of this polytope. "all" enumerates every FRST
        # exhaustively; "fast"/"fair" draw a random sample (the latter aims for a
        # more uniform sample at higher cost).
        if mode == "all":
            triangs_of_poly = poly.all_triangulations(only_fine=True, only_regular=True, only_star=True, as_list=True, star_origin = 0, include_points_interior_to_facets=False)
        elif mode == "fast":
            triangs_of_poly = poly.random_triangulations_fast(N=N_triangs_per_poly, max_retries=5000, c=20, make_star=True, only_fine=True, as_list=True, progress_bar=False, include_points_interior_to_facets=False)
        elif mode == "fair":
            triangs_of_poly = poly.random_triangulations_fair(N=N_triangs_per_poly, max_retries=500, make_star=True, as_list=True, progress_bar=False, include_points_interior_to_facets=False)

        numTri.append(len(triangs_of_poly))

        for index, triang in enumerate(triangs_of_poly):
            # check that the triangulation is actually FRST
            try:
                p = Polytope(triang.points())
                p.triangulate(simplices=triang.simplices())
            except:
                print("A triangulation was in fact not FRST")
                continue
            if index == 0:
                # NOTE: using triang.points() rather than poly.points()
                # process the polytope (only once)
                # remove the origin
                if not np.array_equal(triang.points()[0], np.array([0, 0, 0, 0])):
                    print("Unexpected behaviour")
                processed_polytope = triang.points()[1:,:]
                # store the polytope as a string
                processed_polytope = '{' + ','.join('{' + ','.join(map(str, row)) + '}' for row in processed_polytope) + '}'
                new_polytope = {"POLYID":i + lower_bound, 'DRESVERTS': processed_polytope }
                polys.append(new_polytope)

            # process the triangulation
            # remove the origin, shift all indices by -1 (1 -> 0, 2 -> 1, ...)
            processed_triang = [[x-1 for x in simplex[1:]] for simplex in triang.simplices()]
            # store the triangulation as a string
            processed_triang = '{' + ','.join('{' + ','.join(map(str, row)) + '}' for row in processed_triang) + '}'
            new_triang = {"POLYID":i + lower_bound, 'TRIANG': processed_triang}
            triangs.append(new_triang)
        i += 1

        if i%int(len(gen)/10) == 0:
            print(f"Number of polytopes successfully processed: {i}, time passed: {np.round((time.time()-start_time)/3600, decimals=2)}h")

    polys_file = polys_file.replace(".json", f"_{lower_bound}_{upper_bound-1}.json")
    triangs_file = triangs_file.replace(".json", f"_{lower_bound}_{upper_bound-1}.json")
    with open(folder + "/"+ polys_file, 'w') as f:
        json.dump(polys, f, indent=2)
    with open(folder + "/"+ triangs_file, 'w') as f:
        json.dump(triangs, f, indent=2)
    print(f"Done, polytopes saved in {folder}/{polys_file}, triangulation saved in {folder}/{triangs_file}")
    print(f"Number of polytopes saved: {len(polys)}")
    print(f"Number of triangulation saved: {len(triangs)}")

    return numTri




def generate_triangs_legacy(n_vertices, N_poly, N_triangs_per_poly, folder, polys_file, triangs_file, mode = "all"):
    """
        Generates data in the same format as 005.triang.json and 005.poly.json
        polys_file and triangs_file must end in .json
        mode: all, fast, fair. If all, N_triangs_per_poly is unused
        NOTE: the final number of polytopes might be smaller than N_poly if we are very unlucky
        NOTE: as in 005.triang.json and 005.poly.json, each polytope can have the same polyid as several associated triangulations
        NOTE: currently fetching polytopes with lattice = N, without dualizing, and getting triangulations with options include_points_interior_to_facets = False and star_origin = 0 (why these matter is rather unclear)
        NOTE: currently NOT testing whether the polytopes have the correct number of vertices etc. - this is done in the data loader (to facilite comparisons between datasets)
        TODO: understand the randomness of fetch_polytopes
    """
    print("Triangulations generation process starting")
    if n_vertices == 9:
        h11 = 5
    elif n_vertices == 10:
        h11 = 6
    elif n_vertices == 11:
        h11 = 7
    elif n_vertices == 12:
        h11 = 8
    else:
        print("Unexpected n_vertices")

    if not os.path.exists(folder):
        os.makedirs(folder)

    from cytools import fetch_polytopes, Polytope  # lazy import (see note at top of file)
    print("Fetching polytopes")
    gen = fetch_polytopes(h11=h11, lattice="N", limit=N_poly*4, as_list=True) # *4 to make sure that we have enough polytopes despite the rejections
    print(f"Initial number of candidate polytopes : {len(gen)}")
    print(f"Number of polytopes with the correct number of vertices : {sum([1 for poly in gen if len(poly.points()) == n_vertices + 1])}")
    random.shuffle(gen)
    polys = []  # Polytopes
    triangs = []  # FRSTs
    numTri = []  # Number of triangulations per polytope

    i = 0
    print("Generating triangulations")
    while i < N_poly and i < len(gen):
        poly = gen[i]
        if mode == "all":
            triangs_of_poly = poly.all_triangulations(only_fine=True, only_regular=True, only_star=True, as_list=True, star_origin = 0, include_points_interior_to_facets=False)
        elif mode == "fast":
            triangs_of_poly = poly.random_triangulations_fast(N=N_triangs_per_poly, max_retries=500, make_star=True, only_fine=True, as_list=True, progress_bar=False, star_origin = 0, include_points_interior_to_facets=False)
        elif mode == "fair":
            triangs_of_poly = poly.random_triangulations_fair(N=N_triangs_per_poly, max_retries=500, make_star=True, only_fine=True, as_list=True, progress_bar=False, star_origin = 0, include_points_interior_to_facets=False)

        numTri.append(len(triangs_of_poly))

        for index, triang in enumerate(triangs_of_poly):
            # check that the triangulation is actually FRST
            try:
                p = Polytope(triang.points())
                p.triangulate(simplices=triang.simplices())
            except:
                print("A triangulation was in fact not FRST")
                continue
            if index == 0:
                # NOTE: using triang.points() rather than poly.points()
                # process the polytope (only once)
                # remove the origin
                if not np.array_equal(triang.points()[0], np.array([0, 0, 0, 0])):
                    print("Unexpected behaviour")
                processed_polytope = triang.points()[1:,:]
                # store the polytope as a string
                processed_polytope = '{' + ','.join('{' + ','.join(map(str, row)) + '}' for row in processed_polytope) + '}'
                new_polytope = {"POLYID":i, 'DRESVERTS': processed_polytope }
                polys.append(new_polytope)

            # process the triangulation
            # remove the origin, shift all indices by -1 (1 -> 0, 2 -> 1, ...)
            processed_triang = [[x-1 for x in simplex[1:]] for simplex in triang.simplices()]
            # store the triangulation as a string
            processed_triang = '{' + ','.join('{' + ','.join(map(str, row)) + '}' for row in processed_triang) + '}'
            new_triang = {"POLYID":i, 'TRIANG': processed_triang}
            triangs.append(new_triang)
        i += 1

        if i%100 == 0:
            print(f"Number of polytopes successfully processed : {i}")

        if i%1000 == 0:
            print("Saving chunks")
            with open(folder + "/"+ polys_file.replace(".json", f".chunk_{int(i/1000)}.json"), 'w') as f:
                json.dump(polys, f, indent=2)
            with open(folder + "/"+ triangs_file.replace(".json", f".chunk_{int(i/1000)}.json"), 'w') as f:
                json.dump(triangs, f, indent=2)
            polys = []
            triangs = []

    print("Merging chunks")

    merge_json_chunks(folder, polys_file.replace(".json",""))
    merge_json_chunks(folder, triangs_file.replace(".json",""))

    return numTri






if __name__ == "__main__":
    print("Data generation process starting")


    n_vertices = int(sys.argv[1])
    folder = f"Data/{str(n_vertices)}+1"
    lower_bound = int(sys.argv[2])
    upper_bound = int(sys.argv[3])
    mode = sys.argv[4]
    assert mode in ["all", "fast", "fair"], "mode must be one of all, fast, fair"
    N_triangs_per_poly = int(sys.argv[5])
    if N_triangs_per_poly == 0:
        N_triangs_per_poly = None
    superset_size = int(sys.argv[6])

    if superset_size == 0:
        superset_size = None
    polys_file = f"{str(n_vertices)}+1_polys.json"
    triangs_file = f"{str(n_vertices)}+1_triangs.json"

    generate_triangs(n_vertices, lower_bound, upper_bound, N_triangs_per_poly, folder, polys_file, triangs_file, mode = mode, seed = 42, superset_size = superset_size)
