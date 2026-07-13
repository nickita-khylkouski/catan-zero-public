#!/usr/bin/env python3
"""Read-only layer drift audit for two compatible entity_graph checkpoints.

The audit compares checkpoint tensors directly on CPU. It does not construct a
model, mutate either checkpoint, evaluate playing strength, or apply a pass/fail
threshold. Architecture metadata and the complete state-dict key/shape/dtype
contract must match exactly before any drift statistics are emitted.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA = "entity-graph-checkpoint-layer-drift-v1"
_BLOCK_PATTERN = re.compile(r"^blocks\.(\d+)\.")
_INPUT_PREFIXES = (
    "hex_encoder.",
    "vertex_encoder.",
    "edge_encoder.",
    "player_encoder.",
    "global_encoder.",
    "event_encoder.",
    "type_embedding",
    "cls_token",
)
_POLICY_PREFIXES = (
    "action_encoder.",
    "action_bias.",
    "logit_scale",
    "target_gather_proj.",
    "action_cross_blocks.",
    "edge_policy_mlp.",
)
_VALUE_PREFIXES = (
    "value_head.",
    "value_categorical_head.",
    "value_uncertainty_head.",
    "value_probe",
    "value_probe_norm_",
    "value_probe_attn.",
    "value_pool_head.",
)


class DriftAuditError(ValueError):
    """Checkpoint compatibility or input-contract failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    if not path.is_file():
        raise DriftAuditError(f"checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise DriftAuditError(f"checkpoint is not a dictionary: {path}")
    if payload.get("policy_type") != "entity_graph":
        raise DriftAuditError(
            f"checkpoint must declare policy_type='entity_graph': {path}"
        )
    model = payload.get("model")
    if not isinstance(model, Mapping) or not model:
        raise DriftAuditError(f"checkpoint has no non-empty model state_dict: {path}")
    return payload


def _jsonable(value: Any) -> Any:
    import torch

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__config_dataclass__": type(value).__name__,
            "fields": {
                field.name: _jsonable(getattr(value, field.name))
                for field in dataclasses.fields(value)
                if hasattr(value, field.name)
            },
        }
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return {
            "tensor_shape": list(value.shape),
            "tensor_dtype": str(value.dtype),
        }
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return repr(value)


def _architecture(payload: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    from catan_zero.rl.entity_feature_adapter import (
        resolve_checkpoint_entity_feature_adapter,
    )
    from catan_zero.rl.config_serialization import config_from_dict, config_to_dict
    from catan_zero.rl.entity_token_policy import EntityGraphConfig

    static = payload.get("static_action_features")
    static_contract = None
    if torch.is_tensor(static):
        contiguous = static.detach().cpu().contiguous()
        static_contract = {
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "sha256": "sha256:"
            + hashlib.sha256(contiguous.numpy().tobytes()).hexdigest(),
        }
    try:
        effective_config = config_from_dict(
            EntityGraphConfig,
            payload.get("config"),
            warn=lambda _message: None,
        )
    except (TypeError, ValueError) as error:
        raise DriftAuditError(f"invalid entity_graph config: {error}") from error
    adapter_version, _adapter_source = resolve_checkpoint_entity_feature_adapter(
        payload.get("entity_feature_adapter"),
        metadata_present="entity_feature_adapter" in payload,
    )
    return {
        "policy_type": payload.get("policy_type"),
        # Normalize legacy dataclass and durable name-keyed configs through the
        # same current defaults. Representation-only checkpoint age is allowed;
        # every effective architecture field must still compare exactly.
        "config": _jsonable(config_to_dict(effective_config)),
        "action_mask_version": payload.get("action_mask_version"),
        "mask_hidden_info": bool(payload.get("mask_hidden_info", False)),
        "entity_feature_adapter_version": adapter_version,
        "static_action_features_sha256": payload.get("static_action_features_sha256"),
        "static_action_features": static_contract,
    }


def _provenance(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata_keys = (
        "policy_type",
        "config",
        "action_mask_version",
        "mask_hidden_info",
        "entity_feature_adapter",
        "soft_target_source",
        "value_training",
        "trained_value_readouts",
        "grow_from_checkpoint_sha256",
        "a1_curriculum_parent",
    )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "checkpoint_metadata": {
            key: _jsonable(payload[key]) for key in metadata_keys if key in payload
        },
    }


def _assert_compatible(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline_path: Path,
    candidate_path: Path,
) -> None:
    import torch

    baseline_arch = _architecture(baseline)
    candidate_arch = _architecture(candidate)
    if baseline_arch != candidate_arch:
        raise DriftAuditError(
            "entity_graph architecture metadata differs: "
            f"baseline={baseline_arch!r} candidate={candidate_arch!r}"
        )
    baseline_state = baseline["model"]
    candidate_state = candidate["model"]
    baseline_keys = set(baseline_state)
    candidate_keys = set(candidate_state)
    if baseline_keys != candidate_keys:
        raise DriftAuditError(
            f"state_dict keys differ between {baseline_path} and {candidate_path}: "
            f"missing={sorted(baseline_keys - candidate_keys)} "
            f"extra={sorted(candidate_keys - baseline_keys)}"
        )
    for name in sorted(baseline_keys):
        left = baseline_state[name]
        right = candidate_state[name]
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            raise DriftAuditError(f"state_dict entry {name!r} is not a tensor")
        if left.shape != right.shape:
            raise DriftAuditError(
                f"tensor {name!r} shape mismatch: {tuple(left.shape)} != "
                f"{tuple(right.shape)}"
            )
        if left.dtype != right.dtype:
            raise DriftAuditError(
                f"tensor {name!r} dtype mismatch: {left.dtype} != {right.dtype}"
            )


def _group_for_tensor(name: str) -> str:
    block = _BLOCK_PATTERN.match(name)
    if block:
        return f"transformer_block_{int(block.group(1)):03d}"
    if name.startswith(_INPUT_PREFIXES):
        return "input_encoders"
    if name.startswith(_POLICY_PREFIXES):
        return "policy"
    if name.startswith("final_vp_head."):
        return "final_vp"
    if name.startswith("q_head."):
        return "q"
    if name.startswith(_VALUE_PREFIXES):
        return "value"
    return "shared"


def _empty_accumulator() -> dict[str, Any]:
    return {
        "tensor_count": 0,
        "parameter_count": 0,
        "baseline_energy": 0.0,
        "candidate_energy": 0.0,
        "delta_energy": 0.0,
        "dot_product": 0.0,
    }


def _accumulate(target: dict[str, Any], metric: Mapping[str, Any]) -> None:
    target["tensor_count"] += 1
    target["parameter_count"] += int(metric["parameter_count"])
    for key in ("baseline_energy", "candidate_energy", "delta_energy", "dot_product"):
        target[key] += float(metric[key])


def _safe_relative_l2(delta_energy: float, baseline_energy: float) -> float | None:
    if baseline_energy <= 0.0:
        return None
    return math.sqrt(delta_energy / baseline_energy)


def _safe_cosine(
    dot_product: float, left_energy: float, right_energy: float
) -> float | None:
    if left_energy <= 0.0 or right_energy <= 0.0:
        return None
    value = dot_product / math.sqrt(left_energy * right_energy)
    return max(-1.0, min(1.0, value))


def _finalize_metric(
    metric: Mapping[str, Any], total_delta_energy: float
) -> dict[str, Any]:
    delta_energy = float(metric["delta_energy"])
    return {
        "tensor_count": int(metric["tensor_count"]),
        "parameter_count": int(metric["parameter_count"]),
        "baseline_l2": math.sqrt(float(metric["baseline_energy"])),
        "candidate_l2": math.sqrt(float(metric["candidate_energy"])),
        "delta_l2": math.sqrt(delta_energy),
        "delta_energy": delta_energy,
        "delta_energy_share": (
            delta_energy / total_delta_energy if total_delta_energy > 0.0 else 0.0
        ),
        "relative_l2": _safe_relative_l2(
            delta_energy, float(metric["baseline_energy"])
        ),
        "cosine_similarity": _safe_cosine(
            float(metric["dot_product"]),
            float(metric["baseline_energy"]),
            float(metric["candidate_energy"]),
        ),
    }


def audit_checkpoints(
    baseline_path: Path,
    candidate_path: Path,
    *,
    top_tensors: int = 25,
) -> dict[str, Any]:
    """Compare two exact-compatible entity_graph checkpoints without mutation."""
    import torch

    if top_tensors < 0:
        raise DriftAuditError("top_tensors must be non-negative")
    baseline_path = baseline_path.resolve()
    candidate_path = candidate_path.resolve()
    baseline = _load_checkpoint(baseline_path)
    candidate = _load_checkpoint(candidate_path)
    _assert_compatible(baseline, candidate, baseline_path, candidate_path)

    groups: dict[str, dict[str, Any]] = {}
    total = _empty_accumulator()
    tensor_metrics: list[dict[str, Any]] = []
    for name in sorted(baseline["model"]):
        left = baseline["model"][name]
        right = candidate["model"][name]
        if torch.is_complex(left):
            raise DriftAuditError(
                f"complex state tensor {name!r} is unsupported by this real-valued audit"
            )
        if not torch.is_floating_point(left):
            if not torch.equal(left, right):
                raise DriftAuditError(
                    f"non-floating state tensor {name!r} changed; cannot express as "
                    "parameter drift"
                )
            continue
        left64 = left.detach().cpu().to(torch.float64)
        right64 = right.detach().cpu().to(torch.float64)
        if not bool(torch.isfinite(left64).all()) or not bool(
            torch.isfinite(right64).all()
        ):
            raise DriftAuditError(f"non-finite value in state tensor {name!r}")
        delta = right64 - left64
        raw = {
            "parameter_count": left.numel(),
            "baseline_energy": float(torch.sum(left64 * left64).item()),
            "candidate_energy": float(torch.sum(right64 * right64).item()),
            "delta_energy": float(torch.sum(delta * delta).item()),
            "dot_product": float(torch.sum(left64 * right64).item()),
        }
        if not all(math.isfinite(float(value)) for value in raw.values()):
            raise DriftAuditError(f"non-finite drift metric for state tensor {name!r}")
        group = _group_for_tensor(name)
        grouped = groups.setdefault(group, _empty_accumulator())
        _accumulate(grouped, raw)
        _accumulate(total, raw)
        tensor_metrics.append({"name": name, "group": group, **raw})

    total_delta_energy = float(total["delta_energy"])
    finalized_tensors = [
        {
            "name": item["name"],
            "group": item["group"],
            **_finalize_metric({"tensor_count": 1, **item}, total_delta_energy),
        }
        for item in tensor_metrics
    ]
    by_delta = sorted(
        finalized_tensors,
        key=lambda item: (-item["delta_energy"], item["name"]),
    )[:top_tensors]
    by_relative = sorted(
        (item for item in finalized_tensors if item["relative_l2"] is not None),
        key=lambda item: (-item["relative_l2"], item["name"]),
    )[:top_tensors]
    return {
        "schema_version": SCHEMA,
        "audit_kind": "descriptive_read_only",
        "thresholds": None,
        "baseline": _provenance(baseline_path, baseline),
        "candidate": _provenance(candidate_path, candidate),
        "architecture_contract": _architecture(baseline),
        "compatibility": {
            "exact_architecture_metadata": True,
            "exact_state_dict_keys_shapes_dtypes": True,
        },
        "global": _finalize_metric(total, total_delta_energy),
        "groups": {
            name: _finalize_metric(groups[name], total_delta_energy)
            for name in sorted(groups)
        },
        "top_tensor_outliers": {
            "limit": int(top_tensors),
            "by_delta_energy": by_delta,
            "by_relative_l2": by_relative,
        },
        "metric_definitions": {
            "delta_energy": "sum((candidate - baseline)^2)",
            "delta_energy_share": "group_delta_energy / global_delta_energy",
            "relative_l2": "||candidate - baseline||_2 / ||baseline||_2; null when baseline norm is zero",
            "cosine_similarity": "dot(baseline,candidate)/(||baseline||_2*||candidate||_2); null when either norm is zero",
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise DriftAuditError(f"output already exists: {path}; pass --force to replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, required=True, help="Starting entity_graph checkpoint."
    )
    parser.add_argument(
        "--candidate", type=Path, required=True, help="Trained entity_graph checkpoint."
    )
    parser.add_argument(
        "--top-tensors",
        type=int,
        default=25,
        help="Number of tensor outliers to retain in each ranking (default: 25).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON artifact path; omitted prints JSON to stdout.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing --output artifact; checkpoints remain read-only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = audit_checkpoints(
        args.baseline,
        args.candidate,
        top_tensors=args.top_tensors,
    )
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _write_json(args.output.resolve(), report, force=bool(args.force))
        print(
            json.dumps({"output": str(args.output.resolve()), "schema_version": SCHEMA})
        )


if __name__ == "__main__":
    main()
