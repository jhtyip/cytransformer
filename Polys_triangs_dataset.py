import numpy as np
import json
import random
from itertools import combinations
from torch.utils.data import Dataset
from torch import from_numpy





class Polys_triangs_dataset(Dataset):
    def __init__(self, polys, triangs, max_seq_length_src, max_seq_length_tgt, permutation, triang_shuffling = True, seed=0):
        self.max_seq_length_src = max_seq_length_src  # This is just N, as in N_vert=N+1
        self.max_seq_length_tgt = max_seq_length_tgt
        self.permutation = permutation
        self.triang_shuffling = triang_shuffling
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.polys = [np.array([np.array(vertex.split(",")).astype("int") for vertex in poly[1][2:-2].split("},{")]) for poly in polys]
        self.triangs = []
        for triang in triangs:
            clean_triang = [np.array(simplex.split(",")).astype("int") for simplex in triang[1][2:-2].split("},{")]
            if len(clean_triang) < self.max_seq_length_tgt:
                clean_triang.extend([[-1,-1,-1,-1] for _ in range(self.max_seq_length_tgt-len(clean_triang))])
            clean_triang = np.array(clean_triang)
            self.triangs.append(clean_triang)
        self.poly_mask = np.ones(self.max_seq_length_src, dtype=int)

    def __len__(self):
        return len(self.triangs)

    def __getitem__(self, idx):
        poly = self.polys[idx].copy()
        triang = self.triangs[idx].copy()
        poly_mask = self.poly_mask

        if self.permutation:
            verPerm_idx = self.rng.permutation(self.max_seq_length_src)
            # verPerm_idx = np.random.permutation(self.max_seq_length_src)
            poly = poly[verPerm_idx,:]

        simplexList = np.array(list(combinations(np.arange(self.max_seq_length_src), 4)))
        padding_idx = len(simplexList) + 2
        triang_enc = []

        for simplex in triang:
            if self.permutation:
                if simplex[0] != -1:
                    for pt_id, pt in enumerate(simplex):
                        simplex[pt_id] = np.where(verPerm_idx==pt)[0][0] # maybe speed up
            if simplex[0] != -1:
                simplex.sort()
                triang_enc.append(np.where((simplexList==simplex).all(1))[0][0])

        if self.triang_shuffling:
            self.rng.shuffle(triang_enc)
        triang_enc.insert(0, padding_idx-2)
        triang_enc.append(padding_idx-1)
        if len(triang_enc) < self.max_seq_length_tgt:
            triang_enc.extend([padding_idx for _ in range(self.max_seq_length_tgt-len(triang_enc))])

        return from_numpy(poly), from_numpy(np.array(triang_enc)), from_numpy(poly_mask)


if __name__ == "__main__":
    from data_generation import  split_data
    from torch.utils.data import DataLoader
    filepath_train_polys = 'Data/9+1/debug/9+1_polys_0_200_train.json'
    filepath_train_triangs = 'Data/9+1/debug/9+1_triangs_0_200_train.json'
    filepath_val_polys = 'Data/9+1/debug/9+1_polys_0_200_val.json'
    filepath_val_triangs = 'Data/9+1/debug/9+1_triangs_0_200_val.json'
    filepath_test_polys = 'Data/9+1/debug/9+1_polys_0_200_test.json'
    filepath_test_triangs = 'Data/9+1/debug/9+1_triangs_0_200_test.json'
    N = 9
    max_seq_length_tgt = 30

    with open(filepath_train_polys, 'r') as f:
        toSave_polys = json.load(f)
    with open(filepath_train_triangs, 'r') as f:
        toSave_triangs = json.load(f)
    polys_train, triangs_train, polys_val, triangs_val, polys_test, triangs_test = split_data(toSave_polys, toSave_triangs, [0,5], [0,0], [0,0], 3)
    loader_train = DataLoader(
    Polys_triangs_dataset(polys_train, triangs_train, N, max_seq_length_tgt, True),
    batch_size=3,
    shuffle=True,
    num_workers=0
)
    print("Polys val:")
    print(polys_val)

    for poly, triang, mask in loader_train:
        print(poly)
        # print(triang)
        # print(mask)