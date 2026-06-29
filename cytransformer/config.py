"""
Thin YAML -> existing-params loader.

This intentionally does NOT introduce any new model/encoding semantics: it only
populates the project's existing ModelParams / TrainingParams / JobParams
objects from a YAML file, and derives the EncodingParams through the
frozen `Args.encoding_parameters(n_vertices)` -- the single source of truth for
tokenization. So configs replace the old wall of CLI flags without touching the
frozen model contract.
"""

import yaml

from cytransformer.args import (
    ModelParams,
    TrainingParams,
    JobParams,
    encoding_parameters,
)


def _apply(obj, section):
    """Set attributes on a params object from a config dict, rejecting unknown keys."""
    for key, value in (section or {}).items():
        if not hasattr(obj, key):
            raise KeyError(
                f"Unknown config key '{key}' for {type(obj).__name__}. "
                f"Known keys include: {[a for a in dir(obj) if not a.startswith('_') and a != 'display']}"
            )
        setattr(obj, key, value)
    return obj


def load_train_config(path):
    """Return (model_params, encoding_params, training_params, job_params)."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    model_params = _apply(ModelParams(), cfg.get("model"))

    if "training" not in cfg or "n_vertices" not in cfg["training"]:
        raise KeyError("config must set training.n_vertices")
    encoding_params = encoding_parameters(cfg["training"]["n_vertices"])  # FROZEN path

    training_params = _apply(TrainingParams(), cfg.get("training"))
    job_params = _apply(JobParams(), cfg.get("job"))

    return model_params, encoding_params, training_params, job_params


def load_infer_config(path):
    """Inference config is a flat dict; encoding is read from the checkpoint, not here."""
    with open(path) as f:
        return yaml.safe_load(f) or {}
