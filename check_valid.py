"""
Pure-geometry validity check for triangulations (independent of CYTools).

A triangulation is treated as a list of simplices given by their vertex
coordinates. It is "valid" when (1) the simplices tile the polytope without
gaps or overlaps -- their volumes sum to the polytope's volume -- and (2) every
pair of simplices intersects "cleanly", i.e. their geometric intersection is
exactly the convex hull of their shared vertices (a common face), never a
partial overlap. The pairwise face check is done exactly with rationals via
pycddlib, so it does not depend on the FRST machinery in check_FRST.py and acts
as an independent cross-validation of it.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from fractions import Fraction
from collections import Counter
from scipy.spatial import ConvexHull
from utilities import vert_indices_triang_to_vert_coord_triang, vert_coord_woo_triang_to_vert_coord_wo_triang



def compute_intersection(poly1, poly2):
    """
        Return the intersection of two polyhedra.
        Each polyhedron is given by the set of its vertices (numpy array of floats), and so is the intersection.
    """
    import cdd.gmp as pcdd  # lazy import: pycddlib only needed for triangulation validation
    assert type(poly1) == np.ndarray and type(poly2) == np.ndarray, "poly1 and poly2 must be numpy arrays"
    poly1 = np.array([[Fraction(str(x)) for x in point] for point in poly1.tolist()])
    poly2 = np.array([[Fraction(str(x)) for x in point] for point in poly2.tolist()])
    # Create the V-representation of the first polyhedron, prepending with a column of ones
    v1 = [[Fraction(1)] + list(point) for point in poly1]

    # Convert v1 into a cdd matrix
    mat1 = pcdd.matrix_from_array(np.array(v1, dtype=object), lin_set=set(), rep_type=pcdd.RepType.GENERATOR)

    # Create the polyhedron
    poly1 = pcdd.polyhedron_from_matrix(mat1)

    # Create the V-representation of the second polyhedron, prepending with a column of ones
    v2 = [[Fraction(1)] + list(point) for point in poly2]

    # Convert v2 into a cdd matrix
    mat2 = pcdd.matrix_from_array(np.array(v2, dtype=object), lin_set=set(), rep_type=pcdd.RepType.GENERATOR)
    poly2 = pcdd.polyhedron_from_matrix(mat2)

    # H-representation of the first polyhedron
    h1 = pcdd.copy_inequalities(poly1)

    # H-representation of the second polyhedron
    h2 = pcdd.copy_inequalities(poly2)

    # Join the two sets of linear inequalities to find the intersection
    hintersection = h2
    pcdd.matrix_append_to(hintersection, h1)

    # Create the V-representation of the intersection
    polyintersection =pcdd.polyhedron_from_matrix(hintersection)

    # Get the vertices, removing the column of ones
    vintersection = pcdd.copy_generators(polyintersection)
    vintersection = vintersection.array
    ptsintersection = np.array([[float(x) for x in v[1:]] for v in vintersection])

    return ptsintersection




def check_valid_intersection(simplex1, simplex2, verbose = False):
    """
    Return True iff two simplices meet "cleanly", i.e. their geometric
    intersection coincides with the set of vertices they share (a common face,
    a single shared vertex, or nothing at all). Returns False for any partial
    overlap, containment, or mismatch between the geometric intersection and the
    shared-vertex set -- the situations that make a triangulation invalid.
    """
    # Find the geometric intersection of the two convex hulls
    geom_intersection = compute_intersection(simplex1, simplex2).tolist()

    # The vertices the two simplices literally share (their would-be common face).
    common_vertices = [vertex for vertex in simplex1 if  np.any(np.all(vertex == simplex2, axis=1))]
    # If every vertex of one simplex is shared, one is contained in the other -- invalid.
    if  len(common_vertices) == len(simplex1) or len(common_vertices) == len(simplex2):
        if verbose:
            print("One of the simplices is included in the other")
        return False
    # Case where the intersection and the geometric intersection are empty
    if len(geom_intersection) == 0 and len(common_vertices) == 0:
        if verbose:
            print("Empty intersection")
        return True
    # Case where the set intersection is empty but the geometric intersection is not
    elif len(geom_intersection) == 0 and len(common_vertices) > 0 :
        if verbose: 
            print("Failure case 1")
        return False
    # Case where the geometric intersection is not empty but the set intersection is empty
    elif len(geom_intersection) != 0 and len(common_vertices) == 0:
        if verbose:
            print("Geometric intersection non-empty, set intersection empty")
        return False
    # Case where the intersection is a single point
    elif len(geom_intersection) == 1 and len(common_vertices) == 1 :
        if verbose:
            print("Single point intersection")
        if np.all(np.array(geom_intersection[0])[0] == common_vertices[0]):
            return True
        else:
            if verbose:
                print("Failure case 2")
            return False
    # Generic case
    else:
        if verbose:
            print("Generic case")
            print(f"geom_intersection: {geom_intersection}")
        if Counter(map(tuple, geom_intersection)) == Counter(map(tuple, common_vertices)):
            if verbose:
                print("Same set")
            return True
        else:
            if verbose: 
                print("Different sets")
            return False
   



def compute_volume(simplex):
    """
    Compute the volume of a simplex.
    simplex should be a 2d numpy array of floats
    """
    return ConvexHull(simplex).volume



def check_valid_triangulation(triangulation, total_volume, verbose = False):
    """
    triangulation is a list of numpy arrays (n_vertices, dimension) (floats), where each array represents a simplex (each row is a vertex given by its coordinates)
    Check if the triangulation is valid, i.e. 1) simplices intersect correctly, and 2) the volume of the simplices sums up to the total volume of the triangulated polytope.
    total_volume should be a float
    In particular, triangulation should be a vert_coord_wo_triang (with floats), using our conventions
    """
    N = len(triangulation)
    # Coverage check: the simplices must exactly fill the polytope. A degenerate
    # (flat) simplex makes ConvexHull raise, which itself signals an invalid tiling.
    try:
        sum_of_volumes = np.sum([compute_volume(simplex) for simplex in triangulation])
    except:
        if verbose:
            print("Flat simplex")
        return False
    if not np.isclose(sum_of_volumes, total_volume):
        if verbose:
            print("Invalid volume")
        return False
    # Overlap check: every pair of simplices must meet only along a shared face.
    for i in range(N):
        for j in range(i+1, N):
            if not check_valid_intersection(triangulation[i], triangulation[j], verbose = verbose):
                if verbose:
                    print("Invalid intersection")
                return False
    
    return True




if __name__ == "__main__":
    i = 0

    # Several test cases for the function check_valid_intersection:
    # Test case 1: empty intersection
    simplex1 = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    simplex2 = np.array([[1.5, 1.5], [1, 2], [2, 1], [2, 2]])
    i+=1
    print(f"Test case {i}")
    print(f"Intersection {compute_intersection(simplex1, simplex2)}")
    print("Success" if check_valid_intersection(simplex1, simplex2) else "failure" )  # Expected

    # Test case 2: intersection is a single point
    simplex1 = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    simplex2 = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
    i+=1
    print(f"Test case {i}")
    print(f"Intersection {compute_intersection(simplex1, simplex2)}")
    print("Success" if check_valid_intersection(simplex1, simplex2) else "failure" )  # Expected

    # Test case 3: intersection is a valid line
    simplex1 = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    simplex2 = np.array([[1, 0], [1, 1], [2, 0], [2, 1]])
    i+=1
    print(f"Test case {i}")
    print(f"Intersection {compute_intersection(simplex1, simplex2)}")
    print("Success" if check_valid_intersection(simplex1, simplex2) else "failure" )  # Expected

    # Test case 4: intersection is an invalid line
    simplex1 = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    simplex2 = np.array([[1, 0.5], [1, 1.5], [2, 0.5], [2, 1.5]])
    i+=1
    print(f"Test case {i}")
    print(f"Intersection {compute_intersection(simplex1, simplex2)}")
    print("Success" if not check_valid_intersection(simplex1, simplex2) else "failure" )  # Expected

    # Test case 5: general case, invalid intersection
    simplex1 = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    simplex2 = np.array([[0.5, 0.5], [1.5, 1.5], [1.5, 0.5], [0.5, 1.5]])
    i+=1
    print(f"Test case {i}")
    print(f"Intersection {compute_intersection(simplex1, simplex2)}")
    print("Success" if not check_valid_intersection(simplex1, simplex2) else "failure" )  # Expected

    # Test case 6: general case, 3d, valid intersection
    # define a 3 dimensional simplex with 4 vertices:
    simplex1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # define another 3 dimensional simplex with 4 vertices and one face in common:
    simplex2 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1]])
    i+=1
    print(f"Test case {i}")
    print(f"Intersection {compute_intersection(simplex1, simplex2)}")
    print("Success" if check_valid_intersection(simplex1, simplex2) else "failure" )  # Expected


    # Test case 7: valid triangulation
    i+=1
    print(f"Test case {i}")
    triangulation = [np.array([[0., 0], [1, 1], [1, 0]]), np.array([[0., 0], [1, 1], [0, 1]])]

    print("Success" if check_valid_triangulation(triangulation, 1) else "failure")  # Expected

    triangulation = [np.array([[0., 0], [1, 1], [1, 0]]), np.array([[0., 0], [1, 1], [1, 0]])]

    print("Success" if not check_valid_triangulation(triangulation, 1) else "failure")  # Expected


    # Test case 8: check the triangulations of the database
    i+=1
    print(f"Test case {i}")
    print("Triangulations of the database")
    dresverts = np.load("Data/Various/DRESVERTS_list_test.npy") # (200, 10, 4)
    frsts = np.load("Data/Various/FRST_list_test.npy") # (200, 35, 4) (indices of the vertices of the simplices of the triangulation)


    print(dresverts.shape)
    print(frsts.shape)
    print(dresverts[0])
    print(frsts[0])
    print(vert_coord_woo_triang_to_vert_coord_wo_triang(vert_indices_triang_to_vert_coord_triang(dresverts[0], frsts[0]), dimension = 4))

    for i in range(10): # range(len(dresverts)):
        polytope = dresverts[i]

        triang = frsts[i]
        list_of_simplices = vert_coord_woo_triang_to_vert_coord_wo_triang(vert_indices_triang_to_vert_coord_triang(polytope, triang), dimension = 4)
        volume = ConvexHull(polytope).volume
        if not check_valid_triangulation(list_of_simplices, total_volume=volume):
            print(f"Invalid {i}")
        if i%50 == 0:
            print(f"Tested {i} polytopes")
    print("All tests passed")


    # Test case 9: saved model's outputs
    import json
    i+=1
    print(f"Test case {i}")
    print("Triangulations output by the model")
        
    # Open JSON files


    with open('Data/Various/polytopes_transOut.json', 'r') as file:
        polytopes = json.load(file)

    with open('Data/Various/triangulations_transOut.json', 'r') as file:
        # Note: origin already included
        vert_indices_wo_triang = json.load(file)

    with open('Data/Various/FRSTs_indices.json', 'r') as file:
        frst_indices = json.load(file)

    print(f"first polytope {polytopes[0]}")
    print(f"first triangulation {vert_indices_wo_triang[0]}")
    # no need to add the origin and shift the indices, as it was already included in both the triangulation and the polytope
    print(f"first triangulation adapted {vert_indices_triang_to_vert_coord_triang(polytopes[0],vert_indices_wo_triang[0])}")


    print(len(frst_indices))
    frst_triangs = [triang for i, triang in enumerate(vert_indices_wo_triang) if frst_indices[i] == 1]
    frst_polytopes = [polytope for i, polytope in enumerate(polytopes) if frst_indices[i] == 1]
    print(len(frst_triangs))
    print(len(frst_polytopes))

    n_valid = 0
    n_invalid = 0
    for i in range(len(frst_triangs)):
        if i%50 == 0 and i>0:
            print(f"i: {i}, n_valid: {n_valid}, n_invalid: {n_invalid}")
        polytope = frst_polytopes[i]
        triang = frst_triangs[i]
        # no need to add the origin since it is already included
        list_of_simplices =  vert_indices_triang_to_vert_coord_triang(polytope, triang)
        volume = ConvexHull(polytope).volume
        if not check_valid_triangulation(list_of_simplices, total_volume=volume, verbose = False):
            print(f"Invalid {i}")
            n_invalid += 1
        else:
            n_valid += 1


    print("Done")

