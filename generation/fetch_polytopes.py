"""EXTRA (CYTools): fetch Kreuzer-Skarke reflexive polytopes and write an input
polytope file (processed [POLYID, DRESVERTS]) ready for `cyt-infer`.

    python generation/fetch_polytopes.py --h11 5 --n 100 --out my_polys.json
    cyt-infer --checkpoint model.pt --polys my_polys.json

So a user with trained weights can generate FRSTs for KS polytopes right away.
This is one of the only parts that needs CYTools (https://cytools.liammcallister.com);
the core cyt package needs none of it. The resolved vertices are extracted exactly
as in generation/make_dataset.py (origin dropped, interior-to-facet points excluded)
so the format matches what the model was trained on.
"""
import argparse
import json

import numpy as np

ORIGIN = np.array([0, 0, 0, 0])


def main():
    ap = argparse.ArgumentParser(description="Fetch KS polytopes -> input file for cyt-infer (needs CYTools).")
    ap.add_argument("--h11", type=int, required=True, help="Hodge number h^{1,1}.")
    ap.add_argument("--n", type=int, default=100, help="number of polytopes to write")
    ap.add_argument("--out", required=True, help="output JSON file")
    ap.add_argument("--lattice", default="N", choices=["N", "M"])
    args = ap.parse_args()

    from cytools import fetch_polytopes

    n_vert = args.h11 + 4  # # resolved (favorable) vertices; total points = n_vert + 1 (origin)
    records = []
    pid = 0
    for poly in fetch_polytopes(h11=args.h11, lattice=args.lattice, limit=args.n * 5, as_list=True):
        try:
            pts = poly.triangulate(include_points_interior_to_facets=False).points()
        except Exception:
            continue
        # The model encoding assumes the origin is the first point and is dropped.
        if len(pts) != n_vert + 1 or not np.array_equal(pts[0], ORIGIN):
            continue
        verts = pts[1:]  # resolved vertices, origin excluded
        dresverts = "{" + ",".join("{" + ",".join(str(int(x)) for x in v) + "}" for v in verts) + "}"
        records.append([pid, dresverts])
        pid += 1
        if len(records) >= args.n:
            break

    with open(args.out, "w") as f:
        json.dump(records, f, indent=1)
    print(f"wrote {len(records)} polytopes (h11={args.h11}) to {args.out}")


if __name__ == "__main__":
    main()
