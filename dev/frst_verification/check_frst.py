"""
FRST classification of generated triangulations, using CYTools as ground truth.

Given a triangulation encoded as model tokens (the simplex-subset vocabulary
described in Polys_triangs_dataset), these functions rebuild the corresponding
CYTools Polytope/Triangulation and report which of the Star / Fine-Star /
Regular-Star / Fine-Regular-Star (FRST) properties hold. ``FRST_check`` batches
this over many polytopes and their candidate triangulations, deduplicating
identical triangulations before the (expensive) CYTools check.
"""

import sys
import os
import numpy as np
from cytransformer.utilities import tokens_triang_to_vert_indices_wo_triang
import time

def is_triangulation_FRST(poly, poly_mask, tokens_triang, padding_indx):
    """
        poly numpy array of shape (N_vertices -1, 4)
        poly_mask numpy array of shape (N_vertices,)
        tokens_triang is a numpy array of shape (L,) where L is an upper bound on the number of simplices in a triangulation (tokens_triang)
        TODO: improve documentation
        The function returns True if the triangulation is FRST

    """

    from cytools import Polytope  # lazy import: CYTools only needed for FRST validation
    try:
        vertices = poly
        vertices_mask = poly_mask

        numOfVer = np.sum(vertices_mask)

        # of the shape [[0 1 3 4 6], [...], ...] where 0 corresponds to the origin and 1 to the first vertex in vertices
        vert_indices_wo_triang = tokens_triang_to_vert_indices_wo_triang(tokens_triang, numOfVer, padding_indx)

        # points is a permuted version of vertices whose order is imposed to us by the Polytope object
        vertices = vertices[:numOfVer]
        # add the origin to account for the convention above
        vertices_ = [[0,0,0,0]]
        vertices_.extend(vertices)
        vertices = np.array(vertices_).astype(int)
        p = Polytope(vertices)
        points = p.points()
        # find the permutation that goes from vertices to points
        Dict = {}
        for vertex_ind in range(len(vertices)):
            # Dict is such that points[Dict[i]] = vertices[i]
            Dict[vertex_ind] = np.where((points==vertices[vertex_ind]).all(1))[0][0]

        # encode the triangulation with indices that are suited to the ordering of vertices in points
        T_simplices_new = []
        for simplex in vert_indices_wo_triang:
            simplex_new = []
            for pt in simplex:
                simplex_new.append(Dict[pt])
            T_simplices_new.append(simplex_new)
        T_simplices_new = np.array(T_simplices_new)

        try:
            # Hand the relabelled simplices to CYTools and read back which star
            # properties the resulting triangulation actually has.
            tri = p.triangulate(simplices=T_simplices_new)

            if (tri.is_fine() and tri.is_star()) and not (tri._is_regular is not None and tri._is_regular):
                return "FST"
            elif (not tri.is_fine()) and tri.is_star() and tri._is_regular is not None and tri._is_regular:
                return "RST"
            elif not (tri.is_fine() and tri.is_star() and tri._is_regular is not None and tri._is_regular):
                return "ST"
            else:
                return "FRST"
        except:
            return "error"
    except Exception as e:
        print(f"Error in triangulation {tokens_triang} for polytope {poly} with mask {poly_mask}: {e}")
        return "error"



def FRST_check(polys, poly_masks, triangulations, padding_indx, counts_only, verbose = True):
    """
        polys: np.array of shape (N, N_vert, 4)
        poly_masks: np.array of shape (N, N_vert)
        triangulations: np.array of shape (N, M, N_tri) where N_tri is an upper bound on the number of simplices in a triangulation (tokens_triang)
        (M triangulations per polytope)
        padding_indx: the index of the padding token
        counts_only: if True, only return the total number of unique FRST triangulations
        Returns:
        The total number of unique FRST triangulations
        An np.array of booleans of shape (N, M) where the element (i,j) is True if the triangulation j of polytope i is FRST (optional)
    """
    # Add a singleton "triangulation" axis so polys[ind, 0] / poly_masks[ind, 0]
    # index a single polytope inside the per-polytope loop below.
    polys = np.expand_dims(polys, axis = 1)
    poly_masks = np.expand_dims(poly_masks, axis = 1)
    N, M, _ = triangulations.shape
    if not counts_only:
        triangulation_is_FRST = np.zeros((N, triangulations.shape[1]), dtype=bool)

    T_ok = []
    T_failed = []
    non_unique_FRSTs_count = []
    FRST_count = []
    FST_count = []
    RST_count = []
    ST_count = []
    start_time = time.time()
    for ind in range(N):
        FRST_count_this = 0
        FST_count_this = 0
        RST_count_this = 0
        ST_count_this = 0

        # Normalise this polytope's M candidate triangulations so identical ones
        # collapse: sort each simplex's tokens, fold the special <sos>/<eos>/<pad>
        # tokens all onto padding_indx, then deduplicate rows. T_reverse_indices
        # maps each of the M originals back to its representative in T_values.
        T_poly = triangulations[ind]
        T_poly = np.array(T_poly)[:,:]
        T_poly.sort(axis=1)
        T_poly[np.isin(T_poly, [padding_indx-2, padding_indx-1, padding_indx])] = padding_indx
        T_values, T_reverse_indices = np.unique(T_poly, return_inverse=True, axis=0)

        # Run the (expensive) CYTools check once per unique triangulation and
        # tally it into the appropriate Star/FST/RST/FRST bucket.
        for T_i in range(len(T_values)):
            is_FRST = is_triangulation_FRST(polys[ind,0], poly_masks[ind,0], T_values[T_i], padding_indx)
            if is_FRST=="FRST":
                T_ok.append(ind)
                FRST_count_this += 1
                # Propagate the FRST verdict back to every original (pre-dedup) slot.
                if not counts_only:
                    for i in range(M):
                        if T_reverse_indices[i] == T_i:
                            triangulation_is_FRST[ind, i] = True
            elif is_FRST=="FST":
                FST_count_this += 1
            elif is_FRST=="RST":
                RST_count_this += 1
            elif is_FRST=="ST":
                ST_count_this += 1
            else:
                T_failed.append(ind)
        non_unique_FRSTs_count.append(triangulation_is_FRST[ind].sum())
        FRST_count.append(FRST_count_this)
        RST_count.append(RST_count_this)
        FST_count.append(FST_count_this)
        ST_count.append(ST_count_this)
        if ind % max(4,int(N//4)) == 0 and verbose:
            print(f"Processed polytope {ind} - Time elapsed: {(time.time()-start_time)/60} mn")

    if counts_only:
        return non_unique_FRSTs_count, FRST_count, FST_count, RST_count, ST_count
    else:
        return non_unique_FRSTs_count, FRST_count, FST_count, RST_count, ST_count, triangulation_is_FRST



if __name__ == "__main__":
    # Test the monitoring functions


    # For N_vert=10+1
    padding_indx = 212
    poly = np.load("Data/Test_FRST_check/toSave_poly_fast_200.npy").astype(int) # (200, 1, 10, 4)
    poly_mask = np.load("Data/Test_FRST_check/toSave_poly_mask_fast_200.npy", ).astype(int) # (200, 1, 10)
    Ts = np.load("Data/Test_FRST_check/toSave_Ts_fast_200.npy").astype(int) #  (200, 200, 35) (200 triangulations per polytope) (tokens_triang)
    poly = poly[:,0,:,:]
    poly_mask = poly_mask[:,0,:]
    print(poly.shape, poly_mask.shape, Ts.shape)
    count, bool_matrix = FRST_check(poly[:10,:,:], poly_mask[:10,:], Ts[:10,:14,:], 212, False)
    print(count) # [6, 1, 2, 7, 1, 2, 3, 3, 4, 1]
    print(bool_matrix)


    padding_indx = 212
    poly_mask = np.array([1]*10)
    poly = np.array([[-1,  0, -1,  0],
            [-1,  0, -1,  2],
            [-1,  0,  0,  1],
            [-1,  1, -1,  0],
            [-1,  1, -1,  1],
            [ 0,  0, -1,  0],
            [ 1, -1,  1, -1],
            [ 1, -1,  1,  0],
            [ 1,  0,  1, -1],
            [-1,  0, -1,  1]])
    tokens_triang = np.array([210, 209, 211] + [212]*32)
    print(is_triangulation_FRST(poly, poly_mask, tokens_triang, padding_indx))
