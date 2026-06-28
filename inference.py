import time
import numpy as np
import torch
import torch.nn as nn
import math
import os

from Models import Transformer

from utilities import apply_permutation_to_tokens_triang

# def generate_triangulation(transformer, poly, poly_mask, padding_idx, max_seq_length_tgt, device, verbose=False):
#     """
#         Outdated !!! (now tensors) :
#         Input:
#         poly should be a numpy array of shape (N_vertices, 4)
#         poly_mask should be a numpy array of shape (N_vertices)

#         Output: a triangulation as a numpy array of tokens ending with padding_idx-1 in case of success, an empty numpy array otherwise
#     """
#     if verbose:
#         print("Inside generate_triangulation")

#     poly = torch.as_tensor(poly, dtype=torch.float32, device=device).unsqueeze(0)
#     poly_mask = torch.as_tensor(poly_mask, dtype=torch.int64, device=device).unsqueeze(0)

#     triang = torch.tensor([[padding_idx - 2]], dtype=torch.int64, device=device)
#     last_token_pred = None
#     now_idx = 0
#     max_attempts = max_seq_length_tgt * 2
#     attempts = 0

#     while last_token_pred != padding_idx - 1 and attempts < max_attempts:
#         attempts += 1
#         if verbose:
#             print(f"Step {attempts}: triang = {triang.tolist()}")

#         try:
#             output = transformer(poly, triang, poly_mask)  # (1, seq_len, vocab_size)
#             logits = output[0, now_idx, :]  # (vocab_size,)

#             if torch.isnan(logits).any() or torch.isinf(logits).any():
#                 if verbose:
#                     print("Invalid logits detected.")
#                 return torch.empty(0, dtype=torch.int64)

#             probs = torch.softmax(logits, dim = 0)
#             sampled_token = torch.multinomial(probs, num_samples=1).item()
#             last_token_pred = sampled_token

#             if sampled_token in {padding_idx - 2, padding_idx}:
#                 if verbose:
#                     print("Invalid token encountered — early termination.")
#                 return torch.empty(0, dtype=torch.int64)

#             now_idx += 1

#             new_token = torch.tensor([[sampled_token]], dtype=torch.int64, device=device)
#             triang = torch.cat([triang, new_token], dim=1)
#             if now_idx >= max_seq_length_tgt:
#                 if verbose:
#                     print("Reached max sequence length — early termination.")
#                 return torch.empty(0, dtype=torch.int64)

#         except Exception as e:
#             if verbose:
#                 print(f"Exception during generation: {e}")
#             return torch.empty(0, dtype=torch.int64)

#     if sampled_token == padding_idx - 1:
#         return triang
#     else:
#         if verbose:
#             print("Failed to generate EOS token in time.")
#         return torch.empty(0, dtype=torch.int64)



# def generate_triangulations(transformer, polys, poly_masks, numOfTs_per_poly, padding_idx, max_seq_length_tgt, save_dir, device, verbose=False):
#     """
#         Input:
#         polys should be a numpy array of shape (N, N_vertices, 4)
#         poly_masks should be a numpy array of shape (N, N_vertices)

#         The function generates numOfTs_per_poly triangulations per polytope as a numpy array of shape (N, numOfTs_per_poly, max_seq_length_tgt)
#         It keeps track of the number of failed triangulation attempts for each polytope

#         The function outputs the generated triangulations and the numbers of failed attempts
#         It also stores them (along with other outputs) in Exps/save_dir/
#     """
#     print("Debug - generate triangulations new faster function")
#     transformer.eval()

#     polys_tensor = torch.as_tensor(polys, dtype=torch.float32, device=device)
#     poly_masks_tensor = torch.as_tensor(poly_masks, dtype=torch.int64, device=device)

#     triangs = []
#     list_n_failures = []

#     start_time = time.time()
#     current_time = start_time

#     if verbose:
#         print(f"Number of polytopes {polys.shape[0]}, num of triangulations per polytope {numOfTs_per_poly}, max seq length {max_seq_length_tgt}")

#     for i in range(polys.shape[0]):
#         poly = polys_tensor[i]
#         poly_mask = poly_masks_tensor[i]

#         Ts = []
#         n_failures = 0

#         if time.time() - current_time > 3600 or time.time() - start_time > 15000:
#             print("Too much time passed, verbose mode activated for debugging")
#             verbose = True
#         if time.time() - start_time > 30000:
#             print("More than 8h have passed, stopping the generation")
#             break
#         if verbose:
#             print(f"\nPolytope {i+1}/{polys.shape[0]} - Time since last polytope: {(time.time() - start_time)/60:.2f} mn")
#         current_time = time.time()

#         for _ in range(numOfTs_per_poly):
#             triang = generate_triangulation(transformer, poly, poly_mask, padding_idx, max_seq_length_tgt, device, verbose)
#             if triang.numel() == 0:
#                 n_failures += 1
#                 if verbose:
#                     print("Generation failed.")
#                 Ts.append(torch.tensor([padding_idx - 2, 0, padding_idx - 1], dtype=torch.int64))
#             else:
#                 if verbose:
#                     print("Generation succeeded.")
#                 Ts.append(triang.squeeze(0))

#         # Fill missing if needed
#         while len(Ts) < numOfTs_per_poly:
#             Ts.append(torch.tensor([0, padding_idx - 1], dtype=torch.int64))

#         # Pad to max_seq_length_tgt
#         Ts_padded = []
#         for seq in Ts:
#             padded = torch.full((max_seq_length_tgt,), padding_idx, dtype=torch.int64)
#             length = min(seq.shape[0], max_seq_length_tgt)
#             padded[:length] = seq[:length]
#             Ts_padded.append(padded)

#         triangs.append(torch.stack(Ts_padded))
#         list_n_failures.append(n_failures)

#     triangs = torch.stack(triangs).cpu().numpy()  # Shape: (N, numOfTs_per_poly, max_seq_length_tgt)

#     return triangs, list_n_failures




@torch.inference_mode()
def decode_chunk(transformer, poly_chunk, mask_chunk, padding_idx, max_seq_length_tgt):
    B = poly_chunk.size(0)
    transformer.eval()
    device = poly_chunk.device

    seqs = torch.full((B, 1), padding_idx - 2, dtype=torch.int64, device=device)  # BOS
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    failures = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(max_seq_length_tgt):
        with torch.inference_mode():
            logits = transformer(poly_chunk, seqs, mask_chunk)  # (B, seq_len, vocab_size)
            last_logits = logits[:, -1, :]  # (B, vocab)
            probs = torch.softmax(last_logits, dim=-1)
            sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        invalid = (sampled_tokens == padding_idx - 2) | (sampled_tokens == padding_idx)
        eos = sampled_tokens == (padding_idx - 1)

        failed = invalid & (~finished)
        finished |= eos | failed
        failures |= failed

        sampled_tokens = sampled_tokens.unsqueeze(1)
        seqs = torch.cat([seqs, sampled_tokens], dim=1)

        if finished.all():
            break

    # Pad result to max length
    # TODO check if needed
    padded_seqs = torch.full((B, max_seq_length_tgt), padding_idx, dtype=torch.int64, device=device)
    for i in range(B):
        padded_seqs[i, :seqs[i].size(0)] = seqs[i][:max_seq_length_tgt]

    return padded_seqs, failures



def generate_triangulations_legacy(transformer, polys, poly_masks, numOfTs_per_poly, padding_idx, max_seq_length_tgt, permutation_for_each_triang, save_dir, device, verbose = False, batch_size = 256):
    """
        If permutation_for_each_triang is True, the returned polys and triangulations are of shape (N*numOfTs_per_poly, n_vertices, 4) and (N*numOfTs_per_poly, 1, max_seq_length_tgt) respectively.
        If permutation_for_each_triang is False, the returned polys and triangulations are of shape (N, n_vertices, 4) and (N, numOfTs_per_poly, max_seq_length_tgt) respectively.
    """
    transformer.eval()

    N = polys.shape[0]


    poly_tensor = torch.as_tensor(polys, dtype=torch.float32, device=device)
    mask_tensor = torch.as_tensor(poly_masks, dtype=torch.int64, device=device)

    # Repeat per triangulation
    poly_tensor = poly_tensor.repeat_interleave(numOfTs_per_poly, dim=0)
    mask_tensor = mask_tensor.repeat_interleave(numOfTs_per_poly, dim=0)


    B = poly_tensor.size(0)
    if permutation_for_each_triang:
        n_vertices = poly_tensor.size(1)
        for i in range(B):
            permutation = np.random.permutation(n_vertices)
            poly_tensor[i] = poly_tensor[i][permutation, :]

    all_triangs = torch.empty((B, max_seq_length_tgt), dtype=torch.int64, device=device)
    failures = torch.zeros(N, dtype=torch.int32)

    for i in range(0, B, batch_size):
        poly_chunk = poly_tensor[i:i+batch_size]
        mask_chunk = mask_tensor[i:i+batch_size]

        seqs, failed_chunk = decode_chunk(transformer, poly_chunk, mask_chunk, padding_idx, max_seq_length_tgt)

        all_triangs[i:i+seqs.size(0)] = seqs
        for j, failed in enumerate(failed_chunk):
            if failed:
                failures[(i + j) // numOfTs_per_poly] += 1


    # we need to return the poly_tensor and poly_masks in case permutations were applied
    polys_to_return = poly_tensor.cpu().numpy() if permutation_for_each_triang else polys
    triangs_to_return = all_triangs.view(N*numOfTs_per_poly, 1, max_seq_length_tgt).cpu().numpy() if permutation_for_each_triang else all_triangs.view(N, numOfTs_per_poly, max_seq_length_tgt).cpu().numpy()
    masks_to_return = mask_tensor.cpu().numpy() if permutation_for_each_triang else poly_masks

    return triangs_to_return, failures.tolist(), polys_to_return, masks_to_return




def generate_triangulations(transformer, polys, poly_masks, numOfTs_per_poly, padding_idx, max_seq_length_tgt, permutation_for_each_triang, save_dir, device, verbose = False, batch_size = 256):
    """
        If permutation_for_each_triang is True, the returned polys and triangulations are of shape (N*numOfTs_per_poly, n_vertices, 4) and (N*numOfTs_per_poly, 1, max_seq_length_tgt) respectively.
        If permutation_for_each_triang is False, the returned polys and triangulations are of shape (N, n_vertices, 4) and (N, numOfTs_per_poly, max_seq_length_tgt) respectively.
    """
    transformer.eval()

    N = polys.shape[0]


    poly_tensor = torch.as_tensor(polys, dtype=torch.float32, device=device)
    mask_tensor = torch.as_tensor(poly_masks, dtype=torch.int64, device=device)

    # Repeat per triangulation
    poly_tensor = poly_tensor.repeat_interleave(numOfTs_per_poly, dim=0)
    mask_tensor = mask_tensor.repeat_interleave(numOfTs_per_poly, dim=0)

    B = poly_tensor.size(0)
    if permutation_for_each_triang:
        permutations = []
        n_vertices = poly_tensor.size(1)
        for i in range(B):
            permutation = np.random.permutation(n_vertices)
            poly_tensor[i] = poly_tensor[i][permutation, :]
            permutations.append(permutation)

    all_triangs = torch.empty((B, max_seq_length_tgt), dtype=torch.int64, device=device)
    failures = torch.zeros(N, dtype=torch.int32)

    for i in range(0, B, batch_size):
        poly_chunk = poly_tensor[i:i+batch_size]
        mask_chunk = mask_tensor[i:i+batch_size]

        seqs, failed_chunk = decode_chunk(transformer, poly_chunk, mask_chunk, padding_idx, max_seq_length_tgt)

        all_triangs[i:i+seqs.size(0)] = seqs
        for j, failed in enumerate(failed_chunk):
            if failed:
                failures[(i + j) // numOfTs_per_poly] += 1

    # If permutations were applied, we need to apply the reverse operation to them (to make them compatible with the original polys)
    if permutation_for_each_triang:
        for i in range(B):
            permutation = permutations[i]
            all_triangs[i] = torch.tensor(apply_permutation_to_tokens_triang(all_triangs[i], permutation, poly_tensor.size(1)),
                             dtype=all_triangs.dtype, device=all_triangs.device)

    triangs_to_return = all_triangs.view(N, numOfTs_per_poly, max_seq_length_tgt).cpu().numpy()


    return triangs_to_return, failures.tolist()



if __name__ == "__main__":

    Gpu = False
    n_vertices = 10 # 9 ou 10 (for 9+1 or 10+1)
    if n_vertices == 9:
        padding_idx = 128
        tgt_vocab_size = math.comb(9, 4) + 1 + 1 + 1  # 126 + <sos> + <eos> + padding
        folder_name = "9+1"
        max_seq_length_tgt = 30  # Sufficiently large
        max_seq_length_src = 9  # 9 vertices
        lr = 0.0001
    elif n_vertices == 10:
        padding_idx = 212
        tgt_vocab_size = math.comb(10, 4) + 1 + 1 + 1  # 210 + <sos> + <eos> + padding
        folder_name = "10+1"
        max_seq_length_tgt = 35  # Sufficiently large
        max_seq_length_src = 10  # 10 vertices
        lr = 0.00005

    if Gpu:
        device = 'cuda'
        device_id = 0
        float_dtype = np.float32
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.cuda.set_device(device_id)

    else:
        device = 'cpu'
        float_dtype = np.float32


    epoch = 3
    # Load the model from the checkpoint file
    checkpoint = torch.load(f'Checkpoints/{folder_name}/FRST_transformer-noRotRef-simplexRep-N_vert={n_vertices}-{format(epoch)}', map_location=device)
    # Create a new instance of the Transformer model with the same hyperparameters
    transformer = Transformer(checkpoint['hyperparams']['d_model_src'],
                            checkpoint['hyperparams']['d_model_tgt'],
                            checkpoint['hyperparams']['d_model'],
                            checkpoint['hyperparams']['tgt_vocab_size'],
                            checkpoint['hyperparams']['num_heads'],
                            checkpoint['hyperparams']['num_layers'],
                            checkpoint['hyperparams']['d_ff_enem'],
                            checkpoint['hyperparams']['d_ff'],
                            checkpoint['hyperparams']['max_seq_length_src'],
                            checkpoint['hyperparams']['max_seq_length_tgt'],
                            checkpoint['hyperparams']['dropout'],
                            padding_idx=padding_idx).to(device)


    numOfTs_per_poly = 2
    save_dir = "Debug"
    polys = np.load("Data/10+1/DRESVERTS_list_test.npy") # (200, 10, 4)
    poly_masks = np.load("Data/10+1/DRESVERTS_mask_list_test.npy") # (200, 10)
    polys = polys[:2,:,:]
    poly_masks = poly_masks[:2,:]
    generate_triangulations(transformer, polys, poly_masks, numOfTs_per_poly, padding_idx, max_seq_length_tgt, save_dir, device, verbose = True)
