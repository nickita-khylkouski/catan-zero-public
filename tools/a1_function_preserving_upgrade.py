#!/usr/bin/env python3
"""Issue and replay immutable, promotion-eligible architecture-upgrade receipts.

The normal A1 learner invariant is exact checkpoint identity: the learner must
start from the declared producer bytes.  A function-preserving architecture
upgrade necessarily changes those bytes by adding parameters, so it needs a
stronger proof than a boolean or an in-checkpoint provenance claim.  This
module provides that proof for a deliberately tiny allowlist of reviewed
zero-output adapters.

Receipts are content addressed, written once, and replay the source/upgraded
checkpoint delta.  Adding another module requires code review here; arbitrary
flags, parameter names, or merely-small forward differences are refused.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = (REPO_ROOT / "src").resolve(strict=True)
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != REPO_SRC]
sys.path.insert(0, str(REPO_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from catan_zero.rl.meaningful_history import (  # noqa: E402
    MEANINGFUL_PUBLIC_HISTORY_LIMIT,
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION,
)
from catan_zero.rl.ordered_history import (  # noqa: E402
    MASKED_MEAN_V1,
    ORDERED_ATTENTION_V2,
)


SCHEMA = "a1-function-preserving-architecture-upgrade-v1"
MODULE_TARGET_GATHER = "entity_graph.action_target_gather.v1"
MODULE_TOPOLOGY_TARGET_GATHER = (
    "entity_graph.topology_residual_adapter+action_target_gather.v1"
)
MODULE_BELIEF_RESOURCE_HEAD = "entity_graph.belief_resource_head.v1"
MODULE_AUX_SUBGOAL_HEADS = "entity_graph.aux_subgoal_heads.v1"
MODULE_AUX_SUBGOAL_POINTER_HEADS = "entity_graph.aux_subgoal_pointer_heads.v1"
MODULE_STATIC_ACTION_RESIDUAL = "entity_graph.static_action_residual.v1"
MODULE_STRUCTURED_ACTION_VALUE = (
    "entity_graph.static_action_residual+legal_action_value_residual.v1"
)
MODULE_ACTION_CROSS_ATTENTION_1 = "entity_graph.action_cross_attention.1.v1"
MODULE_PUBLIC_CARD_COUNT_FEATURES = "entity_graph.public_card_count_features.v1"
MODULE_TARGET_GATHER_PUBLIC_CARD_COUNT = (
    "entity_graph.action_target_gather+public_card_count_features.v1"
)
MODULE_MEANINGFUL_PUBLIC_HISTORY = "entity_graph.meaningful_public_history.v1"
MODULE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY = (
    "entity_graph.public_card_count_features+meaningful_public_history.v1"
)
MODULE_PUBLIC_CARD_COUNT_FEATURES_V2 = (
    "entity_graph.public_card_count_features.v2"
)
MODULE_TARGET_GATHER_PUBLIC_CARD_COUNT_V2 = (
    "entity_graph.action_target_gather+public_card_count_features.v2"
)
MODULE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_V2 = (
    "entity_graph.public_card_count_features+meaningful_public_history.v2"
)
MODULE_ORDERED_MEANINGFUL_PUBLIC_HISTORY = (
    "entity_graph.meaningful_public_history.ordered_attention.v2"
)
MODULE_ORDERED_MEANINGFUL_PUBLIC_HISTORY_FROM_V1 = (
    "entity_graph.meaningful_public_history.ordered_attention.from_v1.v2"
)

# This is intentionally code, not caller-controlled configuration.  A new
# architecture exception must be reviewed and tested before it can initialize
# a promotion-eligible learner.
ALLOWLIST: dict[str, dict[str, Any]] = {
    MODULE_TARGET_GATHER: {
        "flags": {"action_target_gather": True},
        "new_parameter_initialization": {
            "target_gather_proj.0.bias": "zeros",
            "target_gather_proj.0.weight": "ones",
            "target_gather_proj.1.bias": "zeros",
            "target_gather_proj.1.weight": "zeros",
        },
        "config_delta": {"action_target_gather": True},
    },
    MODULE_TOPOLOGY_TARGET_GATHER: {
        "flags": {
            "action_target_gather": True,
            "topology_residual_adapter": True,
        },
        "new_parameter_initialization": {
            "target_gather_proj.0.bias": "zeros",
            "target_gather_proj.0.weight": "ones",
            "target_gather_proj.1.bias": "zeros",
            "target_gather_proj.1.weight": "zeros",
            "topology_residual_adapter.message_norm.bias": "zeros",
            "topology_residual_adapter.message_norm.weight": "ones",
            "topology_residual_adapter.output_projection.bias": "zeros",
            "topology_residual_adapter.output_projection.weight": "zeros",
            "topology_residual_adapter.source_norm.bias": "zeros",
            "topology_residual_adapter.source_norm.weight": "ones",
            "topology_residual_adapter.source_projection.bias": "zeros",
            "topology_residual_adapter.source_projection.weight": "identity",
        },
        "config_delta": {
            "action_target_gather": True,
            "topology_residual_adapter": True,
        },
    },
    MODULE_BELIEF_RESOURCE_HEAD: {
        "flags": {"belief_resource_head": True},
        "new_parameter_initialization": {
            "belief_resource_head.0.bias": "zeros",
            "belief_resource_head.0.weight": "ones",
            "belief_resource_head.1.bias": "seeded_torch_default",
            "belief_resource_head.1.weight": "seeded_torch_default",
            "belief_resource_head.3.bias": "seeded_torch_default",
            "belief_resource_head.3.weight": "seeded_torch_default",
        },
        "config_delta": {"belief_resource_head": True},
    },
    MODULE_AUX_SUBGOAL_HEADS: {
        "flags": {"aux_subgoal_heads": True},
        "new_parameter_initialization": {
            "aux_largest_army_head.0.bias": "seeded_torch_default",
            "aux_largest_army_head.0.weight": "seeded_torch_default",
            "aux_largest_army_head.3.bias": "seeded_torch_default",
            "aux_largest_army_head.3.weight": "seeded_torch_default",
            "aux_longest_road_head.0.bias": "seeded_torch_default",
            "aux_longest_road_head.0.weight": "seeded_torch_default",
            "aux_longest_road_head.3.bias": "seeded_torch_default",
            "aux_longest_road_head.3.weight": "seeded_torch_default",
            "aux_next_settlement_head.0.bias": "seeded_torch_default",
            "aux_next_settlement_head.0.weight": "seeded_torch_default",
            "aux_next_settlement_head.3.bias": "seeded_torch_default",
            "aux_next_settlement_head.3.weight": "seeded_torch_default",
            "aux_robber_target_head.0.bias": "seeded_torch_default",
            "aux_robber_target_head.0.weight": "seeded_torch_default",
            "aux_robber_target_head.3.bias": "seeded_torch_default",
            "aux_robber_target_head.3.weight": "seeded_torch_default",
            "aux_vp_in_n_head.0.bias": "seeded_torch_default",
            "aux_vp_in_n_head.0.weight": "seeded_torch_default",
            "aux_vp_in_n_head.3.bias": "seeded_torch_default",
            "aux_vp_in_n_head.3.weight": "seeded_torch_default",
        },
        "config_delta": {"aux_subgoal_heads": True},
    },
    MODULE_AUX_SUBGOAL_POINTER_HEADS: {
        "flags": {
            "aux_subgoal_heads": True,
            "aux_settlement_pointer_head": True,
        },
        "new_parameter_initialization": {
            "aux_largest_army_head.0.bias": "seeded_torch_default",
            "aux_largest_army_head.0.weight": "seeded_torch_default",
            "aux_largest_army_head.3.bias": "seeded_torch_default",
            "aux_largest_army_head.3.weight": "seeded_torch_default",
            "aux_longest_road_head.0.bias": "seeded_torch_default",
            "aux_longest_road_head.0.weight": "seeded_torch_default",
            "aux_longest_road_head.3.bias": "seeded_torch_default",
            "aux_longest_road_head.3.weight": "seeded_torch_default",
            "aux_next_settlement_pointer_head.0.bias": "zeros",
            "aux_next_settlement_pointer_head.0.weight": "ones",
            "aux_next_settlement_pointer_head.1.bias": "seeded_torch_default",
            "aux_next_settlement_pointer_head.1.weight": "seeded_torch_default",
            "aux_next_settlement_pointer_head.4.bias": "seeded_torch_default",
            "aux_next_settlement_pointer_head.4.weight": "seeded_torch_default",
            "aux_robber_target_head.0.bias": "seeded_torch_default",
            "aux_robber_target_head.0.weight": "seeded_torch_default",
            "aux_robber_target_head.3.bias": "seeded_torch_default",
            "aux_robber_target_head.3.weight": "seeded_torch_default",
            "aux_vp_in_n_head.0.bias": "seeded_torch_default",
            "aux_vp_in_n_head.0.weight": "seeded_torch_default",
            "aux_vp_in_n_head.3.bias": "seeded_torch_default",
            "aux_vp_in_n_head.3.weight": "seeded_torch_default",
        },
        "config_delta": {
            "aux_subgoal_heads": True,
            "aux_settlement_pointer_head": True,
        },
    },
    MODULE_STATIC_ACTION_RESIDUAL: {
        "flags": {"static_action_residual": True},
        "new_parameter_initialization": {
            "static_action_residual_proj.bias": "zeros",
            "static_action_residual_proj.weight": "zeros",
        },
        "config_delta": {"static_action_residual": True},
    },
    MODULE_STRUCTURED_ACTION_VALUE: {
        "flags": {
            "static_action_residual": True,
            "legal_action_value_residual": True,
        },
        "new_parameter_initialization": {
            "legal_action_value_residual_proj.weight": "zeros",
            "static_action_residual_proj.bias": "zeros",
            "static_action_residual_proj.weight": "zeros",
        },
        "config_delta": {
            "static_action_residual": True,
            "legal_action_value_residual": True,
        },
    },
    MODULE_PUBLIC_CARD_COUNT_FEATURES: {
        "flags": {"public_card_count_features": True},
        "new_parameter_initialization": {
            "public_card_count_residual.bias": "zeros",
            "public_card_count_residual.weight": "zeros",
        },
        "config_delta": {"public_card_count_features": True},
    },
    MODULE_TARGET_GATHER_PUBLIC_CARD_COUNT: {
        "flags": {
            "action_target_gather": True,
            "public_card_count_features": True,
        },
        "new_parameter_initialization": {
            "target_gather_proj.0.bias": "zeros",
            "target_gather_proj.0.weight": "ones",
            "target_gather_proj.1.bias": "zeros",
            "target_gather_proj.1.weight": "zeros",
            "public_card_count_residual.bias": "zeros",
            "public_card_count_residual.weight": "zeros",
        },
        "config_delta": {
            "action_target_gather": True,
            "public_card_count_features": True,
        },
    },
    MODULE_MEANINGFUL_PUBLIC_HISTORY: {
        "flags": {
            "meaningful_public_history": True,
            "meaningful_public_history_schema": (
                MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION
            ),
            "event_history_limit": MEANINGFUL_PUBLIC_HISTORY_LIMIT,
        },
        "new_parameter_initialization": {
            "meaningful_history_residual_gate": "zeros",
        },
        "config_delta": {
            "meaningful_public_history": True,
            "meaningful_public_history_schema": (
                MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION
            ),
            "event_history_limit": MEANINGFUL_PUBLIC_HISTORY_LIMIT,
        },
    },
    MODULE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY: {
        "flags": {
            "public_card_count_features": True,
            "meaningful_public_history": True,
            "meaningful_public_history_schema": (
                MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION
            ),
            "event_history_limit": MEANINGFUL_PUBLIC_HISTORY_LIMIT,
        },
        "new_parameter_initialization": {
            "public_card_count_residual.bias": "zeros",
            "public_card_count_residual.weight": "zeros",
            "meaningful_history_residual_gate": "zeros",
        },
        "config_delta": {
            "public_card_count_features": True,
            "meaningful_public_history": True,
            "meaningful_public_history_schema": (
                MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION
            ),
            "event_history_limit": MEANINGFUL_PUBLIC_HISTORY_LIMIT,
        },
    },
    # Bias-free v2 variants deliberately retain the v1 feature tensor and
    # output location, but remove the trainable intercept.  This preserves the
    # stronger invariant that an unknown/all-zero public-card row contributes
    # exactly zero throughout training, not only at initialization.  The v1
    # entries above remain immutable so issued bias-bearing receipts replay.
    MODULE_PUBLIC_CARD_COUNT_FEATURES_V2: {
        "flags": {
            "public_card_count_features": True,
            "public_card_count_residual_bias": False,
        },
        "new_parameter_initialization": {
            "public_card_count_residual.weight": "zeros",
        },
        "config_delta": {
            "public_card_count_features": True,
            "public_card_count_residual_bias": False,
        },
    },
    MODULE_TARGET_GATHER_PUBLIC_CARD_COUNT_V2: {
        "flags": {
            "action_target_gather": True,
            "public_card_count_features": True,
            "public_card_count_residual_bias": False,
        },
        "new_parameter_initialization": {
            "target_gather_proj.0.bias": "zeros",
            "target_gather_proj.0.weight": "ones",
            "target_gather_proj.1.bias": "zeros",
            "target_gather_proj.1.weight": "zeros",
            "public_card_count_residual.weight": "zeros",
        },
        "config_delta": {
            "action_target_gather": True,
            "public_card_count_features": True,
            "public_card_count_residual_bias": False,
        },
    },
    MODULE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_V2: {
        "flags": {
            "public_card_count_features": True,
            "public_card_count_residual_bias": False,
            "meaningful_public_history": True,
            "meaningful_public_history_schema": (
                MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION
            ),
            "event_history_limit": MEANINGFUL_PUBLIC_HISTORY_LIMIT,
        },
        "new_parameter_initialization": {
            "public_card_count_residual.weight": "zeros",
            "meaningful_history_residual_gate": "zeros",
        },
        "config_delta": {
            "public_card_count_features": True,
            "public_card_count_residual_bias": False,
            "meaningful_public_history": True,
            "meaningful_public_history_schema": (
                MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION
            ),
            "event_history_limit": MEANINGFUL_PUBLIC_HISTORY_LIMIT,
        },
    },
    MODULE_ORDERED_MEANINGFUL_PUBLIC_HISTORY: {
        "flags": {
            "meaningful_public_history": True,
            "meaningful_public_history_schema": (
                MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION
            ),
            "event_history_limit": MEANINGFUL_PUBLIC_HISTORY_LIMIT,
            "meaningful_public_history_pooling": ORDERED_ATTENTION_V2,
        },
        "new_parameter_initialization": {
            "meaningful_history_residual_gate": "zeros",
            "meaningful_history_ordered_gate": "zeros",
            "meaningful_history_sequence.norm.bias": "zeros",
            "meaningful_history_sequence.norm.weight": "ones",
            "meaningful_history_sequence.position_embedding": "zeros",
            "meaningful_history_sequence.query": "seeded_torch_default",
        },
        "config_delta": {
            "meaningful_public_history": True,
            "meaningful_public_history_schema": (
                MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION
            ),
            "event_history_limit": MEANINGFUL_PUBLIC_HISTORY_LIMIT,
            "meaningful_public_history_pooling": ORDERED_ATTENTION_V2,
        },
    },
    # Upgrade an already-trained v1 history checkpoint without replacing its
    # learned mean path or gate. The only additions are the ordered branch and
    # its zero gate; all v1 parameters remain byte-identical.
    MODULE_ORDERED_MEANINGFUL_PUBLIC_HISTORY_FROM_V1: {
        "flags": {
            "meaningful_public_history": True,
            "meaningful_public_history_schema": (
                MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION
            ),
            "event_history_limit": MEANINGFUL_PUBLIC_HISTORY_LIMIT,
            "meaningful_public_history_pooling": ORDERED_ATTENTION_V2,
        },
        "new_parameter_initialization": {
            "meaningful_history_ordered_gate": "zeros",
            "meaningful_history_sequence.norm.bias": "zeros",
            "meaningful_history_sequence.norm.weight": "ones",
            "meaningful_history_sequence.position_embedding": "zeros",
            "meaningful_history_sequence.query": "seeded_torch_default",
        },
        "config_delta": {
            "meaningful_public_history_pooling": ORDERED_ATTENTION_V2,
        },
    },
    MODULE_ACTION_CROSS_ATTENTION_1: {
        "flags": {"action_cross_attention_layers": 1},
        "new_parameter_initialization": {
            "action_cross_blocks.0.attn.in_proj_bias": "zeros",
            "action_cross_blocks.0.attn.in_proj_weight": "seeded_torch_default",
            "action_cross_blocks.0.attn.out_proj.bias": "zeros",
            "action_cross_blocks.0.attn.out_proj.weight": "zeros",
            "action_cross_blocks.0.ff.0.bias": "seeded_torch_default",
            "action_cross_blocks.0.ff.0.weight": "seeded_torch_default",
            "action_cross_blocks.0.ff.3.bias": "zeros",
            "action_cross_blocks.0.ff.3.weight": "zeros",
            "action_cross_blocks.0.norm_ff.bias": "zeros",
            "action_cross_blocks.0.norm_ff.weight": "ones",
            "action_cross_blocks.0.norm_kv.bias": "zeros",
            "action_cross_blocks.0.norm_kv.weight": "ones",
            "action_cross_blocks.0.norm_q.bias": "zeros",
            "action_cross_blocks.0.norm_q.weight": "ones",
        },
        "config_delta": {"action_cross_attention_layers": 1},
    },
}


class UpgradeError(ValueError):
    """The architecture delta is not an allowlisted exact function upgrade."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _digest(value: Any) -> str:
    def _json_default(item: Any) -> Any:
        # Checkpoint configs can retain NumPy scalar types (for example an
        # action_size loaded as np.int64).  They are semantically ordinary
        # JSON scalars, but json.dumps rejects them without normalization.
        # Keep this deliberately narrow: arbitrary objects must still fail
        # closed instead of acquiring a surprising string representation.
        import numpy as np

        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(
            f"Object of type {type(item).__name__} is not JSON serializable"
        )

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _effective_config_receipt_view(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep pre-v2 receipt config hashes stable across the appended knob.

    ``public_card_count_residual_bias=True`` is the exact historical topology
    and therefore carries no new semantic information.  Omitting that default
    from the digest preserves replay of receipts issued before the field
    existed.  The v2 value (False) is retained and changes the digest.
    """

    result = dict(value)
    if result.get("public_card_count_residual_bias") is True:
        result.pop("public_card_count_residual_bias")
    if result.get("legal_action_value_residual") is False:
        result.pop("legal_action_value_residual")
    # The historical history adapter was an unordered masked mean. Omitting
    # that appended default preserves receipts issued before this field
    # existed, while the order-aware v2 value remains receipt-significant.
    if result.get("meaningful_public_history_pooling") == MASKED_MEAN_V1:
        result.pop("meaningful_public_history_pooling")
    return result


def _ref(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise UpgradeError(f"checkpoint must be a regular non-symlink file: {path}")
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": _sha(resolved)}


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise UpgradeError(f"cannot load checkpoint {path}: {error}") from error
    if not isinstance(raw, Mapping):
        raise UpgradeError("checkpoint root is not a mapping")
    return raw


def _config(raw: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(raw):
        return {
            field.name: getattr(raw, field.name)
            for field in dataclasses.fields(raw)
            if hasattr(raw, field.name)
        }
    if isinstance(raw, Mapping):
        fields = raw.get("fields", raw)
        if isinstance(fields, Mapping):
            return dict(fields)
    raise UpgradeError("checkpoint config cannot be normalized")


def _equal(left: Any, right: Any) -> bool:
    import numpy as np
    import torch

    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and np.array_equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_equal(a, b) for a, b in zip(left, right))
        )
    return bool(left == right)


def _reconstruct_seeded_parameters(
    source: Path,
    *,
    seed: int,
    config_delta: Mapping[str, Any],
    expected_names: set[str],
) -> dict[str, Any]:
    """Rebuild deterministic PyTorch-default additions from the source bytes.

    Merely recording an initialization seed in checkpoint provenance is not
    evidence: that field could be forged around arbitrary tensors.  Replay the
    exact warm-start construction used by ``f69_upgrade_checkpoint_config`` and
    compare the newly initialized tensors byte-for-byte instead.
    """

    from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphPolicy

    module = sys.modules[EntityGraphPolicy.__module__]
    module_path = Path(str(module.__file__)).resolve(strict=True)
    if REPO_SRC not in module_path.parents:
        raise UpgradeError(
            f"receipt replayer imported catan_zero outside its checkout: {module_path}"
        )

    base = EntityGraphPolicy.load(str(source), device="cpu")
    values = {
        field.name: getattr(base.config, field.name)
        for field in dataclasses.fields(EntityGraphConfig)
        if hasattr(base.config, field.name)
    }
    values.update(config_delta)
    upgraded = EntityGraphPolicy(
        EntityGraphConfig(**values),
        base.static_action_features.detach().cpu().numpy(),
        seed=seed,
        device="cpu",
    )
    missing, unexpected = upgraded.model.load_state_dict(
        base.model.state_dict(), strict=False
    )
    if unexpected or set(missing) != expected_names:
        raise UpgradeError(
            "deterministic upgrade replay parameter delta drift: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    state = upgraded.model.state_dict()
    return {name: state[name].detach().cpu() for name in sorted(expected_names)}


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    metadata = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        "sha256:"
        + hashlib.sha256(metadata + b"\0" + value.numpy().tobytes()).hexdigest()
    )


def _tensor_equal_exact(left: Any, right: Any) -> bool:
    """Require identical tensor type metadata as well as numeric values.

    ``torch.equal`` intentionally considers equal-valued tensors with different
    dtypes equal.  That is insufficient for a receipt which attests that shared
    checkpoint tensors are bit-identical.
    """

    import torch

    return bool(
        torch.is_tensor(left)
        and torch.is_tensor(right)
        and left.dtype == right.dtype
        and left.layout == right.layout
        and tuple(left.shape) == tuple(right.shape)
        and torch.equal(left, right)
    )


def inspect_upgrade(
    source: Path, upgraded: Path, *, module: str = MODULE_TARGET_GATHER
) -> dict[str, Any]:
    """Replay the checkpoint delta and return its typed semantic evidence."""
    import torch

    spec = ALLOWLIST.get(module)
    if spec is None:
        raise UpgradeError(f"architecture module is not allowlisted: {module!r}")
    source_ref, upgraded_ref = _ref(source), _ref(upgraded)
    before, after = (
        _load_checkpoint(Path(source_ref["path"])),
        _load_checkpoint(Path(upgraded_ref["path"])),
    )
    provenance = after.get("upgrade_provenance")
    seed = (
        provenance.get("initialization_seed")
        if isinstance(provenance, Mapping)
        else None
    )
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("schema_version") != "entity-graph-upgrade-v1"
        or provenance.get("source_checkpoint_sha256")
        != source_ref["sha256"].removeprefix("sha256:")
        or provenance.get("flags") != spec["flags"]
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or provenance.get("forward_max_diff") != 0.0
        or provenance.get("forward_identical_at_init") is not True
        or provenance.get("trained_value_readouts_added") != []
    ):
        raise UpgradeError("checkpoint upgrade provenance is not exact and zero-diff")

    before_model, after_model = before.get("model"), after.get("model")
    if not isinstance(before_model, Mapping) or not isinstance(after_model, Mapping):
        raise UpgradeError("checkpoint model state is malformed")
    added = sorted(set(after_model) - set(before_model))
    removed = sorted(set(before_model) - set(after_model))
    expected_added = sorted(spec["new_parameter_initialization"])
    if removed or added != expected_added:
        raise UpgradeError(
            f"parameter key delta is not allowlisted: added={added} removed={removed}"
        )
    changed = [
        name
        for name in before_model
        if not _tensor_equal_exact(before_model[name], after_model[name])
    ]
    if changed:
        raise UpgradeError(f"shared checkpoint parameters changed: {changed[:8]}")
    seeded_names = {
        name
        for name, kind in spec["new_parameter_initialization"].items()
        if kind == "seeded_torch_default"
    }
    seeded_reference = (
        _reconstruct_seeded_parameters(
            Path(source_ref["path"]),
            seed=seed,
            config_delta=spec["config_delta"],
            expected_names=set(expected_added),
        )
        if seeded_names
        else {}
    )
    for name, kind in spec["new_parameter_initialization"].items():
        tensor = after_model[name]
        if kind == "ones":
            expected = torch.ones_like(tensor)
        elif kind == "zeros":
            expected = torch.zeros_like(tensor)
        elif kind == "identity":
            if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
                raise UpgradeError(f"identity parameter is not square: {name}")
            expected = torch.eye(
                tensor.shape[0], dtype=tensor.dtype, device=tensor.device
            )
        elif kind == "seeded_torch_default":
            expected = seeded_reference[name]
        else:
            raise UpgradeError(f"unknown allowlisted initialization {kind!r}: {name}")
        if not _tensor_equal_exact(tensor, expected):
            raise UpgradeError(f"new parameter is not deterministic {kind}: {name}")

    before_config, after_config = (
        _config(before.get("config")),
        _config(after.get("config")),
    )
    expected_config = dict(before_config)
    expected_config.update(spec["config_delta"])
    # Old checkpoints omit default-valued fields while the upgrade utility may
    # serialize a complete config.  Compare effective current configs.
    from catan_zero.rl.entity_token_policy import EntityGraphConfig

    known = {field.name for field in dataclasses.fields(EntityGraphConfig)}
    unknown_before = sorted(set(before_config) - known)
    unknown_after = sorted(set(after_config) - known)
    if unknown_before or unknown_after:
        raise UpgradeError(
            "checkpoint config contains fields unknown to this checkout: "
            f"source={unknown_before} upgraded={unknown_after}"
        )
    effective_before = dataclasses.asdict(EntityGraphConfig(**before_config))
    effective_expected = dataclasses.asdict(EntityGraphConfig(**expected_config))
    effective_after = dataclasses.asdict(EntityGraphConfig(**after_config))
    if effective_after != effective_expected:
        raise UpgradeError("effective checkpoint config delta is not allowlisted")

    ignored = {"model", "config", "upgrade_provenance"}
    unexpected_metadata = sorted(set(after) - set(before) - {"upgrade_provenance"})
    drift = [
        key
        for key in before
        if key not in ignored
        and (key not in after or not _equal(before[key], after[key]))
    ]
    if unexpected_metadata or drift:
        raise UpgradeError(
            "checkpoint metadata/provenance changed: "
            f"unexpected={unexpected_metadata} drift={drift}"
        )
    evidence = {
        "module": module,
        "source": source_ref,
        "upgraded_initializer": upgraded_ref,
        "flags": dict(spec["flags"]),
        "initialization_seed": seed,
        "forward_max_diff": 0.0,
        "forward_identical_at_init": True,
        "shared_parameters_bit_identical": True,
        "shared_parameter_count": len(before_model),
        "new_parameters": added,
        "new_parameter_initialization": dict(spec["new_parameter_initialization"]),
        "effective_source_config_sha256": _digest(
            _effective_config_receipt_view(effective_before)
        ),
        "effective_upgraded_config_sha256": _digest(
            _effective_config_receipt_view(effective_after)
        ),
    }
    if seeded_names:
        evidence["seeded_parameter_sha256"] = {
            name: _tensor_sha256(after_model[name]) for name in sorted(seeded_names)
        }
    return evidence


def issue_receipt(
    source: Path,
    upgraded: Path,
    output: Path,
    *,
    module: str = MODULE_TARGET_GATHER,
) -> dict[str, Any]:
    evidence = inspect_upgrade(source, upgraded, module=module)
    payload = {"schema_version": SCHEMA, **evidence}
    payload["receipt_sha256"] = _digest(payload)
    output = output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    tmp = output.with_name(f".{output.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, output)
    except FileExistsError as error:
        raise UpgradeError(
            f"refusing to overwrite architecture upgrade receipt: {output}"
        ) from error
    finally:
        tmp.unlink(missing_ok=True)
    return payload


def verify_receipt(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise UpgradeError("architecture upgrade receipt must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UpgradeError(
            f"cannot decode architecture upgrade receipt: {error}"
        ) from error
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        raise UpgradeError("architecture upgrade receipt schema drift")
    stated = value.get("receipt_sha256")
    unhashed = dict(value)
    unhashed.pop("receipt_sha256", None)
    if stated != _digest(unhashed):
        raise UpgradeError("architecture upgrade receipt digest drift")
    expected = inspect_upgrade(
        Path(str(value.get("source", {}).get("path", ""))),
        Path(str(value.get("upgraded_initializer", {}).get("path", ""))),
        module=str(value.get("module", "")),
    )
    if unhashed != {"schema_version": SCHEMA, **expected}:
        raise UpgradeError("architecture upgrade receipt does not replay exactly")
    return {
        **value,
        "receipt": {"path": str(path), "sha256": _sha(path)},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--upgraded", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--module", default=MODULE_TARGET_GATHER, choices=tuple(ALLOWLIST)
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = issue_receipt(
            args.source, args.upgraded, args.output, module=args.module
        )
    except UpgradeError as error:
        raise SystemExit(f"REFUSED: {error}") from error
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
