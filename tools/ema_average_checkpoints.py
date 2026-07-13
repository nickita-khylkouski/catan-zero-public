from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "KataGo-style snapshot-EMA averaging of N compatible entity_graph "
            "checkpoints (state_dict weighted average, newest heaviest). Cheap "
            "alternative to a true training-time EMA: run this once over the last "
            "handful of gate candidates instead of maintaining a shadow model."
        ),
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        help="Checkpoint paths, CHRONOLOGICAL order: oldest first, newest last.",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.75,
        help=(
            "EMA decay in [0.0, 1.0]. Weight for the checkpoint `d` snapshots older "
            "than the newest is proportional to decay**d, so the newest checkpoint "
            "is always at least as heavy as any older one. decay=0.0 degenerates to "
            "'just use the newest checkpoint'; decay=1.0 degenerates to a plain "
            "uniform SWA average (every checkpoint weighted equally); values in "
            "between spread more weight onto older snapshots as decay increases."
        ),
    )
    parser.add_argument("--output", help="Output path for the averaged checkpoint.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the built-in self-test (tiny synthetic checkpoints, no GPU/model "
        "classes required) and exit. Ignores --checkpoints/--output.",
    )
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        print("self-test PASSED")
        return

    if not args.checkpoints:
        raise SystemExit("--checkpoints requires at least one path")
    if not args.output:
        raise SystemExit("--output is required unless --self-test")

    result = ema_average_checkpoints(
        checkpoints=[Path(p) for p in args.checkpoints],
        decay=args.decay,
        output=Path(args.output),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decay": float(args.decay),
                "n_checkpoints": len(args.checkpoints),
                "weights": result["ema_weights"],
            },
            indent=2,
        )
    )


def compute_ema_weights(n: int, decay: float) -> list[float]:
    """Chronological-order (oldest -> newest) EMA weights: weight[i] is proportional to
    decay**(n-1-i), so the newest checkpoint (i == n-1) always gets the largest-or-equal
    single weight. Normalized to sum to 1.0. decay=1.0 is the plain-SWA special case: every
    exponent evaluates to 1, so all checkpoints come out equally weighted."""
    if n < 1:
        raise ValueError("need at least one checkpoint")
    if not (0.0 <= decay <= 1.0):
        raise ValueError(f"decay must be in [0.0, 1.0], got {decay}")
    raw = [decay ** (n - 1 - i) for i in range(n)]
    total = sum(raw)
    return [w / total for w in raw]


def ema_average_checkpoints(
    *,
    checkpoints: list[Path],
    decay: float,
    output: Path | None = None,
) -> dict[str, Any]:
    import torch

    if not checkpoints:
        raise ValueError("at least one checkpoint path is required")

    loaded = [_load_checkpoint(path) for path in checkpoints]
    _assert_compatible_metadata(loaded, checkpoints)
    state_dicts = [ckpt["model"] for ckpt in loaded]
    _assert_same_state_dict_structure(state_dicts, checkpoints)

    weights = compute_ema_weights(len(loaded), decay)
    averaged_state = _ema_blend_value(state_dicts, weights)

    # Carry over the NEWEST checkpoint's metadata (mask_hidden_info, config,
    # policy_type, action_mask_version, static_action_features, ...) verbatim --
    # only the "model" state_dict is the EMA average.
    newest = loaded[-1]
    result: dict[str, Any] = dict(newest)
    result["model"] = averaged_state
    result["ema_decay"] = float(decay)
    result["ema_source_checkpoints"] = [str(p) for p in checkpoints]
    result["ema_weights"] = weights

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, output)
    return result


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a dict checkpoint")
    if "model" not in data:
        raise ValueError(f"{path} has no 'model' state_dict key")
    return data


# Metadata that MUST be identical across every checkpoint being averaged --
# mismatches here mean the checkpoints were trained under different observability
# regimes or architectures, and averaging their weights would silently produce a
# nonsensical model.
_REQUIRED_IDENTICAL_KEYS = ("mask_hidden_info", "config", "policy_type", "action_mask_version")


def _assert_compatible_metadata(checkpoints: list[dict[str, Any]], paths: list[Path]) -> None:
    from catan_zero.rl.entity_feature_adapter import (
        resolve_checkpoint_entity_feature_adapter,
    )

    reference = checkpoints[0]
    ref_path = paths[0]
    for key in _REQUIRED_IDENTICAL_KEYS:
        # mask_hidden_info defaults to False on legacy checkpoints that predate the
        # field (see EntityGraphPolicy.save's identical comment) -- normalize via
        # .get() rather than requiring the key to be literally present.
        ref_value = reference.get(key) if key != "mask_hidden_info" else bool(reference.get(key, False))
        for checkpoint, path in zip(checkpoints[1:], paths[1:]):
            value = checkpoint.get(key) if key != "mask_hidden_info" else bool(checkpoint.get(key, False))
            if not _deep_equal(ref_value, value):
                raise ValueError(
                    f"refusing to EMA-average checkpoints with different {key}: "
                    f"{ref_path} has {ref_value!r}, {path} has {value!r}"
                )
    ref_adapter, _ = resolve_checkpoint_entity_feature_adapter(
        reference.get("entity_feature_adapter"),
        metadata_present="entity_feature_adapter" in reference,
    )
    for checkpoint, path in zip(checkpoints[1:], paths[1:]):
        adapter, _ = resolve_checkpoint_entity_feature_adapter(
            checkpoint.get("entity_feature_adapter"),
            metadata_present="entity_feature_adapter" in checkpoint,
        )
        if adapter != ref_adapter:
            raise ValueError(
                "refusing to EMA-average checkpoints with different entity "
                f"feature adapters: {ref_path} has {ref_adapter!r}, "
                f"{path} has {adapter!r}"
            )


def _assert_same_state_dict_structure(
    state_dicts: list[dict[str, Any]], paths: list[Path]
) -> None:
    reference_keys = set(state_dicts[0])
    for state_dict, path in zip(state_dicts[1:], paths[1:]):
        if set(state_dict) != reference_keys:
            missing = reference_keys - set(state_dict)
            extra = set(state_dict) - reference_keys
            raise ValueError(
                f"{path} state_dict keys differ from {paths[0]}: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
    for key in reference_keys:
        reference_tensor = state_dicts[0][key]
        for state_dict, path in zip(state_dicts[1:], paths[1:]):
            tensor = state_dict[key]
            if tensor.shape != reference_tensor.shape:
                raise ValueError(
                    f"{path} tensor {key!r} shape mismatch: "
                    f"{tensor.shape} != {reference_tensor.shape}"
                )
            if tensor.dtype != reference_tensor.dtype:
                raise ValueError(
                    f"{path} tensor {key!r} dtype mismatch: "
                    f"{tensor.dtype} != {reference_tensor.dtype}"
                )


def _ema_blend_value(values: list[Any], weights: list[float]) -> Any:
    """Recursively weighted-average a list of (nested dict/list/tuple-of-)tensors,
    ``values[i]`` weighted by ``weights[i]``. Non-floating-point tensors (e.g. a
    BatchNorm ``num_batches_tracked`` int64 buffer) are not meaningfully averaged --
    the newest checkpoint's (last in ``values``) value is carried through unchanged,
    matching how such buffers behave under a real training-time EMA."""
    import torch

    reference = values[0]
    if isinstance(reference, dict):
        return {
            key: _ema_blend_value([value[key] for value in values], weights)
            for key in reference
        }
    if isinstance(reference, (list, tuple)):
        blended = [
            _ema_blend_value([value[index] for value in values], weights)
            for index in range(len(reference))
        ]
        return type(reference)(blended) if isinstance(reference, tuple) else blended
    if torch.is_tensor(reference):
        if not torch.is_floating_point(reference):
            return values[-1].clone()
        accumulator = torch.zeros_like(reference, dtype=torch.float64)
        for tensor, weight in zip(values, weights):
            accumulator += tensor.to(torch.float64) * float(weight)
        return accumulator.to(reference.dtype)
    return values[-1]


def _deep_equal(left: Any, right: Any) -> bool:
    import torch

    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            return False
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and bool(torch.equal(left, right))
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_deep_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _deep_equal(item_left, item_right) for item_left, item_right in zip(left, right)
        )
    return left == right


def _self_test() -> None:
    """Tiny end-to-end smoke test with synthetic state dicts -- no real model/GPU
    required. Exercises: weight math, float-tensor blending, integer-buffer pass-
    through, newest-metadata carry-over, disk round-trip, and both refusal paths
    (mask_hidden_info mismatch, config/arch mismatch)."""
    import tempfile

    import torch

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        biases = [0.0, 1.0, 2.0]  # chronological: oldest -> newest
        paths: list[Path] = []
        for index, bias in enumerate(biases):
            path = tmp_path / f"ckpt_{index}.pt"
            torch.save(_fake_checkpoint(bias=bias, step=index, mask_hidden_info=True), path)
            paths.append(path)

        out_path = tmp_path / "ema.pt"
        result = ema_average_checkpoints(checkpoints=paths, decay=0.5, output=out_path)

        weights = compute_ema_weights(3, 0.5)
        assert weights[-1] > weights[0], "newest checkpoint must be heaviest"
        assert abs(sum(weights) - 1.0) < 1e-9

        expected = sum(weight * bias for weight, bias in zip(weights, biases))
        actual = float(result["model"]["trunk.weight"][0, 0])
        assert abs(actual - expected) < 1e-6, f"expected {expected}, got {actual}"

        # Integer buffer must be carried from the newest checkpoint, not averaged.
        assert int(result["model"]["num_batches_tracked"]) == 2

        # Metadata carried from the newest checkpoint.
        assert result["mask_hidden_info"] is True
        assert result["config"] == {"hidden_size": 8, "graph_layers": 2}
        assert result["ema_decay"] == 0.5
        assert len(result["ema_weights"]) == 3
        assert len(result["ema_source_checkpoints"]) == 3

        # Disk round-trip.
        reloaded = torch.load(out_path, map_location="cpu", weights_only=False)
        assert torch.allclose(reloaded["model"]["trunk.weight"], result["model"]["trunk.weight"])

        # Refuse to average checkpoints trained under different observability regimes.
        bad_mask_path = tmp_path / "bad_mask.pt"
        torch.save(_fake_checkpoint(bias=9.0, step=9, mask_hidden_info=False), bad_mask_path)
        try:
            ema_average_checkpoints(checkpoints=[paths[0], bad_mask_path], decay=0.5)
        except ValueError as error:
            assert "mask_hidden_info" in str(error)
        else:
            raise AssertionError("expected ValueError for mask_hidden_info mismatch")

        # Refuse to average checkpoints with different architectures.
        bad_config_path = tmp_path / "bad_config.pt"
        torch.save(
            _fake_checkpoint(bias=9.0, step=9, mask_hidden_info=True, hidden_size=16),
            bad_config_path,
        )
        try:
            ema_average_checkpoints(checkpoints=[paths[0], bad_config_path], decay=0.5)
        except ValueError as error:
            assert "config" in str(error)
        else:
            raise AssertionError("expected ValueError for config mismatch")


def _fake_checkpoint(
    *, bias: float, step: int, mask_hidden_info: bool, hidden_size: int = 8
) -> dict[str, Any]:
    import torch

    return {
        "policy_type": "entity_graph",
        "config": {"hidden_size": hidden_size, "graph_layers": 2},
        "action_mask_version": "v1",
        "mask_hidden_info": mask_hidden_info,
        "static_action_features_sha256": "deadbeef",
        "static_action_features": torch.zeros(3, 3),
        "model": {
            "trunk.weight": torch.full((4, 4), bias),
            "num_batches_tracked": torch.tensor(step, dtype=torch.int64),
        },
    }


if __name__ == "__main__":
    main()
