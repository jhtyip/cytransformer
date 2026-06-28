import numpy as np
import torch

from utilities import apply_permutation_to_tokens_triang


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
