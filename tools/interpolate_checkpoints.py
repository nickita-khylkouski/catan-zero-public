from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


# Checkpoints contain both learned parameters and immutable inference inputs.
# Interpolating the latter is unsafe even when the endpoints are identical:
# floating-point ``x * (1-a) + x * a`` is not guaranteed to reproduce ``x``
# bit-for-bit.  In particular, changing static action-feature tables while
# retaining their authenticated hash makes an otherwise valid checkpoint
# unloadable.  These are the trainable state roots used by the supported
# policy families; everything else is copied exactly from the base checkpoint.
LEARNED_STATE_ROOTS = frozenset(
    {
        "model",
        "actor",
        "critic",
        "q_head",
        "q_state",
        "q_action_encoder",
        "q_action_bias",
        "action_encoder",
        "action_id_embedding",
        "action_bias",
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate two compatible TorchPPOPolicy checkpoints. "
            "alpha=0 writes the base checkpoint; alpha=1 writes the candidate."
        ),
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--alpha",
        action="append",
        type=float,
        required=True,
        help="Blend weight. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output path. If more than one alpha is supplied, include {alpha} "
            "in the path, e.g. runs/self_play/blend_a{alpha}.pt."
        ),
    )
    parser.add_argument(
        "--receipt",
        help=(
            "Optional new JSON receipt binding both source checkpoints, the "
            "tensor schema, interpolation formula, and every output digest."
        ),
    )
    args = parser.parse_args()

    outputs = interpolate_checkpoints(
        base=Path(args.base),
        candidate=Path(args.candidate),
        alphas=tuple(args.alpha),
        output_template=args.output,
    )
    if args.receipt:
        write_interpolation_receipt(
            base=Path(args.base),
            candidate=Path(args.candidate),
            alphas=tuple(args.alpha),
            outputs=outputs,
            receipt=Path(args.receipt),
        )
    for output in outputs:
        print(output)


def interpolate_checkpoints(
    *,
    base: Path,
    candidate: Path,
    alphas: tuple[float, ...],
    output_template: str,
) -> list[Path]:
    import torch

    if not alphas:
        raise ValueError("at least one alpha is required")
    if len(alphas) > 1 and "{alpha}" not in output_template:
        raise ValueError("multiple alphas require {alpha} in --output")

    base_data = _load_checkpoint(base)
    candidate_data = _load_checkpoint(candidate)
    _assert_compatible(base_data, candidate_data)
    base_ref = _checkpoint_ref(base)
    candidate_ref = _checkpoint_ref(candidate)

    outputs: list[Path] = []
    for alpha in alphas:
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in [0, 1]")
        blended = _blend_checkpoint(base_data, candidate_data, alpha)
        # Never let a diagnostic soup inherit the base checkpoint's provenance
        # and masquerade as an ordinary trained candidate when separated from
        # its sidecar receipt.  Model loaders ignore this metadata, while every
        # promotion/provenance consumer can fail closed on the explicit marker.
        blended["checkpoint_interpolation"] = {
            "schema_version": "checkpoint-interpolation-v1",
            "diagnostic_only": True,
            "promotion_eligible": False,
            "formula": "learned_state=(1-alpha)*base+alpha*candidate",
            "alpha": float(alpha),
            "base": base_ref,
            "candidate": candidate_ref,
        }
        output = Path(output_template.format(alpha=_format_alpha(alpha)))
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blended, output)
        outputs.append(output)
    return outputs


def write_interpolation_receipt(
    *,
    base: Path,
    candidate: Path,
    alphas: tuple[float, ...],
    outputs: list[Path],
    receipt: Path,
) -> dict[str, Any]:
    """Write an authenticated, diagnostic-only interpolation receipt."""
    if len(alphas) != len(outputs):
        raise ValueError("alphas and outputs must have the same length")
    base = base.expanduser().resolve(strict=True)
    candidate = candidate.expanduser().resolve(strict=True)
    resolved_outputs = [path.expanduser().resolve(strict=True) for path in outputs]
    receipt = receipt.expanduser().resolve()
    if receipt.exists():
        raise FileExistsError(f"refusing existing receipt: {receipt}")

    base_data = _load_checkpoint(base)
    candidate_data = _load_checkpoint(candidate)
    _assert_compatible(base_data, candidate_data)
    schema = _tensor_schema(base_data)
    value: dict[str, Any] = {
        "schema_version": "checkpoint-interpolation-receipt-v1",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "formula": "learned_state=(1-alpha)*base+alpha*candidate",
        "learned_state_roots": sorted(LEARNED_STATE_ROOTS),
        "immutable_tensor_source": "base (endpoint equality required)",
        "non_floating_source": "base",
        "output_metadata_source": "base",
        "candidate_only_metadata_ignored": sorted(set(candidate_data) - set(base_data)),
        "base": {"path": str(base), "sha256": _sha256(base)},
        "candidate": {"path": str(candidate), "sha256": _sha256(candidate)},
        "tensor_schema": schema,
        "tensor_schema_sha256": _json_sha256(schema),
        "outputs": [
            {"alpha": alpha, "path": str(path), "sha256": _sha256(path)}
            for alpha, path in zip(alphas, resolved_outputs, strict=True)
        ],
    }
    value["receipt_sha256"] = _json_sha256(value)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a dict checkpoint")
    return data


def _checkpoint_ref(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _assert_compatible(base: dict[str, Any], candidate: dict[str, Any]) -> None:
    from catan_zero.rl.entity_feature_adapter import (
        resolve_checkpoint_entity_feature_adapter,
    )

    structural_keys = {
        "observation_size",
        "action_size",
        "hidden_size",
        "architecture",
        "use_action_id_embedding",
        "context_action_feature_size",
    }
    for key in structural_keys:
        if base.get(key) != candidate.get(key):
            raise ValueError(
                f"incompatible checkpoint metadata for {key}: "
                f"{base.get(key)!r} != {candidate.get(key)!r}"
            )
    base_adapter, _ = resolve_checkpoint_entity_feature_adapter(
        base.get("entity_feature_adapter"),
        metadata_present="entity_feature_adapter" in base,
    )
    candidate_adapter, _ = resolve_checkpoint_entity_feature_adapter(
        candidate.get("entity_feature_adapter"),
        metadata_present="entity_feature_adapter" in candidate,
    )
    if base_adapter != candidate_adapter:
        raise ValueError(
            "incompatible checkpoint entity feature adapters: "
            f"{base_adapter!r} != {candidate_adapter!r}"
        )
    base_schema = _tensor_schema(base)
    candidate_schema = _tensor_schema(candidate)
    if base_schema != candidate_schema:
        raise ValueError("checkpoint tensor key/schema sets differ")
    for key, base_value in base.items():
        if key not in LEARNED_STATE_ROOTS and key != "entity_feature_adapter":
            _assert_tensor_values_equal(
                base_value,
                candidate.get(key),
                path=f"checkpoint.{key}",
            )
    # Newer trainers may append diagnostic metadata (for example the sealed
    # value-training receipt) without changing the deployable model schema.
    # Outputs intentionally retain the base metadata and interpolate only the
    # exactly matching base tensor tree.
    # Non-tensor training metadata may legitimately differ.  The structural
    # inference fields above and the complete tensor path/shape/dtype schema
    # are the deployable compatibility boundary.
def _blend_checkpoint(
    base: dict[str, Any], candidate: dict[str, Any], alpha: float
) -> dict[str, Any]:
    blended = deepcopy(base)
    for key in LEARNED_STATE_ROOTS:
        if key in base:
            blended[key] = _blend_value(base[key], candidate[key], alpha)
    return blended


def _assert_tensor_values_equal(base: Any, candidate: Any, *, path: str) -> None:
    """Require immutable tensor leaves to match exactly at both endpoints."""
    import torch

    if isinstance(base, dict):
        for key, value in base.items():
            other = candidate.get(key) if isinstance(candidate, dict) else None
            _assert_tensor_values_equal(value, other, path=f"{path}.{key}")
    elif isinstance(base, (list, tuple)):
        other_items = candidate if isinstance(candidate, type(base)) else ()
        for index, value in enumerate(base):
            other = other_items[index] if index < len(other_items) else None
            _assert_tensor_values_equal(
                value, other, path=f"{path}[{index}]"
            )
    elif torch.is_tensor(base):
        if not torch.is_tensor(candidate) or not torch.equal(base, candidate):
            raise ValueError(f"immutable checkpoint tensor differs: {path}")


def _blend_value(base: Any, candidate: Any, alpha: float) -> Any:
    import torch

    if isinstance(base, dict):
        return {key: _blend_value(base[key], candidate[key], alpha) for key in base}
    if isinstance(base, list):
        return [_blend_value(left, right, alpha) for left, right in zip(base, candidate)]
    if isinstance(base, tuple):
        return tuple(_blend_value(left, right, alpha) for left, right in zip(base, candidate))
    if torch.is_tensor(base):
        if torch.is_floating_point(base):
            return (base * (1.0 - alpha)) + (candidate * alpha)
        return deepcopy(base)
    return deepcopy(base)


def _tensor_schema(value: Any, *, path: str = "checkpoint") -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key in sorted(value):
            rows.extend(_tensor_schema(value[key], path=f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            rows.extend(_tensor_schema(item, path=f"{path}[{index}]"))
    elif torch.is_tensor(value):
        rows.append(
            {
                "path": path,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "floating": bool(torch.is_floating_point(value)),
            }
        )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _format_alpha(alpha: float) -> str:
    text = f"{alpha:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


if __name__ == "__main__":
    main()
