"""
Frozen-model-contract safety net.

Purpose: prove that a CYTransformer checkpoint written in the project's exact
on-disk schema can be re-loaded through the project's own helper functions and
reproduces the model bit-for-bit. This pins the contract that any original
collaborator weights (or our own retrained weights) must satisfy, so that the
upcoming code cleanup cannot silently break weight-loading.

It exercises the REAL frozen code paths:
  - Models.Transformer (architecture / state_dict keys)
  - Args.return_transformer / model_params_from_checkpoint /
    encoding_params_from_checkpoint / encoding_parameters
  - the checkpoint dict schema saved by train.py (lines ~302-328)

Run directly (no pytest needed):
    python tests/test_checkpoint_contract.py
Optionally point Layer A at a real checkpoint:
    python tests/test_checkpoint_contract.py /path/to/chkpt-399999
or set env var CYT_GOLDEN_CKPT.

CYTools is NOT required: it is stubbed below only so that `import Args`
(which transitively imports data_generation -> cytools) succeeds. The functions
under test never call CYTools.
"""

import os
import sys
import types
import tempfile

# --- make the repo root importable (this file lives in tests/) ---
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# --- stub CYTools so `import Args` works without it installed ---
# (Args -> data_generation -> `from cytools import ...`. The contract functions
#  we test do not use CYTools; this is purely to satisfy the import.)
if "cytools" not in sys.modules:
    _stub = types.ModuleType("cytools")
    _stub.fetch_polytopes = lambda *a, **k: None
    _stub.Polytope = object
    sys.modules["cytools"] = _stub

import torch  # noqa: E402
from cytransformer.args import (  # noqa: E402
    ModelParams,
    encoding_parameters,
    return_transformer,
    model_params_from_checkpoint,
    encoding_params_from_checkpoint,
)


def _build_checkpoint_dict(model, model_params, encoding_params,
                           n_vertices, batch_size=128, lr=5e-5):
    """Reproduce the EXACT dict schema that train.py saves (train.py:302-328)."""
    hyperparams = {
        "tgt_vocab_size": encoding_params.tgt_vocab_size,
        "d_model_src": model_params.d_model_src,
        "d_model_tgt": model_params.d_model_tgt,
        "d_model": model_params.d_model,
        "num_heads": model_params.num_heads,
        "num_layers": model_params.num_layers,
        "d_ff_enem": model_params.d_ff_enem,
        "d_ff": model_params.d_ff,
        "max_seq_length_src": encoding_params.max_seq_length_src,
        "max_seq_length_tgt": encoding_params.max_seq_length_tgt,
        "batch_size": batch_size,
        "dropout": model_params.dropout,
        "lr": lr,
    }
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "n_vertices": n_vertices,
        "loss": [],
        "last_step": 0,
        "validation": [],
        "hyperparams": hyperparams,
    }


def test_save_load_roundtrip():
    """Layer B: self-generated checkpoint round-trips through the frozen readers."""
    device = "cpu"
    n_vertices = 9

    # A small but architecturally-faithful model (real layer types, real keys).
    mp = ModelParams()
    mp.d_model = 32
    mp.num_heads = 4
    mp.num_layers = 2
    mp.d_ff = 64
    mp.d_ff_enem = 64
    mp.dropout = 0.0  # eval() also disables dropout; set 0 for clarity
    ep = encoding_parameters(n_vertices)

    torch.manual_seed(0)
    m1 = return_transformer(mp, ep, device).to(device).eval()

    ckpt = _build_checkpoint_dict(m1, mp, ep, n_vertices)
    path = os.path.join(tempfile.gettempdir(), "cyt_contract_ckpt.pt")
    torch.save(ckpt, path)
    loaded = torch.load(path, map_location=device, weights_only=False)

    # Reconstruct purely from the checkpoint via the frozen helpers.
    mp2 = model_params_from_checkpoint(loaded)
    ep2 = encoding_params_from_checkpoint(loaded)
    m2 = return_transformer(mp2, ep2, device).to(device).eval()

    res = m2.load_state_dict(loaded["model_state_dict"], strict=True)
    assert not res.missing_keys, f"missing keys: {res.missing_keys}"
    assert not res.unexpected_keys, f"unexpected keys: {res.unexpected_keys}"

    # Deterministic forward (eval -> dropout off) must match exactly.
    torch.manual_seed(1)
    src = torch.randn(1, ep.max_seq_length_src, mp.d_model_src, device=device)
    sos = ep.padding_idx - 2
    tgt = torch.full((1, 5), sos, dtype=torch.int64, device=device)
    src_mask = torch.ones(1, ep.max_seq_length_src, dtype=torch.int64, device=device)
    with torch.no_grad():
        o1 = m1(src, tgt, src_mask)
        o2 = m2(src, tgt, src_mask)

    assert o1.shape == (1, 5, ep.tgt_vocab_size), f"unexpected output shape {tuple(o1.shape)}"
    max_diff = (o1 - o2).abs().max().item()
    assert torch.allclose(o1, o2, atol=1e-6), f"forward mismatch, max abs diff = {max_diff}"
    os.remove(path)
    return f"vocab={ep.tgt_vocab_size}, padding_idx={ep.padding_idx}, out={tuple(o1.shape)}, max_diff={max_diff:.2e}"


def test_load_golden_checkpoint(golden_path):
    """Layer A: load a REAL checkpoint (collaborator/our own) and run one inference."""
    device = "cpu"
    loaded = torch.load(golden_path, map_location=device, weights_only=False)
    mp = model_params_from_checkpoint(loaded)
    ep = encoding_params_from_checkpoint(loaded)
    model = return_transformer(mp, ep, device).to(device).eval()
    res = model.load_state_dict(loaded["model_state_dict"], strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    src = torch.randn(1, ep.max_seq_length_src, mp.d_model_src, device=device)
    tgt = torch.full((1, 5), ep.padding_idx - 2, dtype=torch.int64, device=device)
    src_mask = torch.ones(1, ep.max_seq_length_src, dtype=torch.int64, device=device)
    with torch.no_grad():
        out = model(src, tgt, src_mask)
    assert out.shape == (1, 5, ep.tgt_vocab_size)
    return f"loaded golden ckpt n_vertices={loaded['n_vertices']}, out={tuple(out.shape)}"


if __name__ == "__main__":
    print("[Layer B] self save/load round-trip ...")
    info = test_save_load_roundtrip()
    print(f"  PASS  ({info})")

    golden = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CYT_GOLDEN_CKPT"))
    if golden:
        print(f"[Layer A] loading golden checkpoint: {golden} ...")
        print(f"  PASS  ({test_load_golden_checkpoint(golden)})")
    else:
        print("[Layer A] skipped (no golden checkpoint given; pass a path or set CYT_GOLDEN_CKPT)")

    print("\nAll contract checks passed.")
