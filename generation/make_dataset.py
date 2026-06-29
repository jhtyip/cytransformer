"""EXTRA (CYTools): build a training dataset from the Kreuzer-Skarke database.

This is one of the only parts that depends on CYTools. It fetches reflexive
polytopes and samples triangulations, writing JSON in the format cyt consumes
(then split it with `cyt-prepare`). Needs a CYTools install
(https://cytools.liammcallister.com).
"""
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






if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate a CYTransformer training dataset from Kreuzer-Skarke (needs CYTools).")
    ap.add_argument("--n_vertices", type=int, required=True)
    ap.add_argument("--lower_bound", type=int, default=0)
    ap.add_argument("--upper_bound", type=int, required=True, help="fetch KS polytopes in the index range [lower_bound, upper_bound)")
    ap.add_argument("--n_triangs_per_poly", type=int, default=10)
    ap.add_argument("--folder", required=True, help="output folder")
    ap.add_argument("--polys_file", required=True, help="output polytopes JSON")
    ap.add_argument("--triangs_file", required=True, help="output triangulations JSON")
    ap.add_argument("--mode", default="all", choices=["all", "fast", "fair"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--superset_size", type=int, default=None)
    a = ap.parse_args()
    generate_triangs(a.n_vertices, a.lower_bound, a.upper_bound, a.n_triangs_per_poly,
                     a.folder, a.polys_file, a.triangs_file, mode=a.mode, seed=a.seed,
                     superset_size=a.superset_size)
