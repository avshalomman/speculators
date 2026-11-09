"""Initialize a DSpark draft's trunk from a trained head and optionally freeze it.

The trunk is everything the draft trains except its sequential heads: ``fc``,
``hidden_norm``, the decoder layers and the final ``norm``. The verifier-shared
tensors (embeddings, ``lm_head``) are frozen and loaded from the verifier anyway.
Loading the trunk alone lets a new head (a corrector in place of the Markov head)
train on top of an already-trained draft.
"""

import json
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file

__all__ = ["HEAD_PREFIXES", "freeze_trunk", "load_trunk"]

logger = logging.getLogger("speculators")

HEAD_PREFIXES = ("markov_head.", "corrector.", "confidence_head.")
_MATCHING_TOP_LEVEL = (
    "block_size",
    "aux_hidden_state_layer_ids",
    "mask_token_id",
    "sliding_window_non_causal",
    "sample_from_anchor",
)
_MATCHING_DECODER = (
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "layer_types",
    "sliding_window",
    "is_causal",
    "rope_parameters",
    "hidden_act",
)


def load_trunk(model: torch.nn.Module, head_dir: str) -> list[str]:
    """Copy the trunk of the head saved in ``head_dir`` into ``model``.

    Args:
        model: Freshly built DSpark draft.
        head_dir: Directory with the trained head's ``config.json`` and
            ``model.safetensors``.

    Returns:
        The names of the trunk tensors that were loaded.

    Raises:
        ValueError: If the head's geometry differs from the model's, a trunk tensor
            is missing or has another shape, or the checkpoint holds a non-head
            tensor the model does not have.
    """
    path = Path(head_dir)
    _check_config_matches(model.config, json.loads((path / "config.json").read_text()))
    weights = load_file(str(path / "model.safetensors"))
    trunk = _trunk_param_names(model)
    state = model.state_dict()
    unknown = sorted(
        n for n in weights if n not in state and not n.startswith(HEAD_PREFIXES)
    )
    missing = sorted(n for n in trunk if n not in weights)
    bad_shapes = sorted(
        n for n in trunk if n in weights and weights[n].shape != state[n].shape
    )
    if unknown or missing or bad_shapes:
        raise ValueError(
            f"{head_dir} does not fit this draft: unknown tensors {unknown[:5]}, "
            f"missing trunk tensors {missing[:5]}, shape mismatches {bad_shapes[:5]}"
        )
    with torch.no_grad():
        for name in trunk:
            state[name].copy_(weights[name].to(state[name].dtype))
    dropped = sorted({n.split(".")[0] for n in weights if n.startswith(HEAD_PREFIXES)})
    logger.info(
        "Loaded %d trunk tensors from %s; the draft's %s keep their own init "
        "(dropped the head's %s)",
        len(trunk),
        head_dir,
        ", ".join(sorted({p.rstrip(".") for p in HEAD_PREFIXES})),
        ", ".join(dropped) or "nothing",
    )
    return trunk


def freeze_trunk(model: torch.nn.Module) -> None:
    """Turn off gradients for every trainable tensor outside the sequential heads.

    Args:
        model: DSpark draft whose trunk was loaded with :func:`load_trunk`.

    Raises:
        ValueError: If no head tensor is left to train.
    """
    trunk = set(_trunk_param_names(model))
    for name, param in model.named_parameters():
        if name in trunk:
            param.requires_grad_(False)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("--freeze-trunk left nothing to train: the draft has no head")
    logger.info(
        "Froze %d trunk tensors; training %d head tensors (%s)",
        len(trunk),
        len(trainable),
        ", ".join(sorted({n.split(".")[0] for n in trainable})),
    )


def _trunk_param_names(model: torch.nn.Module) -> list[str]:
    return [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith(HEAD_PREFIXES)
    ]


def _check_config_matches(config, head_config: dict) -> None:
    ours = config.to_dict()
    decoder, head_decoder = (
        ours["transformer_layer_config"],
        head_config["transformer_layer_config"],
    )
    diffs = [
        f"{key}: draft {ours.get(key)!r} vs head {head_config.get(key)!r}"
        for key in _MATCHING_TOP_LEVEL
        if ours.get(key) != head_config.get(key)
    ] + [
        f"transformer_layer_config.{key}: draft {decoder.get(key)!r} "
        f"vs head {head_decoder.get(key)!r}"
        for key in _MATCHING_DECODER
        if decoder.get(key) != head_decoder.get(key)
    ]
    if diffs:
        raise ValueError(
            "--init-trunk-from head has another geometry: " + "; ".join(diffs)
        )
