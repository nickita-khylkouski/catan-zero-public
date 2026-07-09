from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any


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
    args = parser.parse_args()

    outputs = interpolate_checkpoints(
        base=Path(args.base),
        candidate=Path(args.candidate),
        alphas=tuple(args.alpha),
        output_template=args.output,
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

    outputs: list[Path] = []
    for alpha in alphas:
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in [0, 1]")
        blended = _blend_value(base_data, candidate_data, alpha)
        output = Path(output_template.format(alpha=_format_alpha(alpha)))
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blended, output)
        outputs.append(output)
    return outputs


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a dict checkpoint")
    return data


def _assert_compatible(base: dict[str, Any], candidate: dict[str, Any]) -> None:
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
    if set(base) != set(candidate):
        raise ValueError("checkpoint key sets differ")
    _assert_same_tensor_structure(base, candidate, path="checkpoint")


def _assert_same_tensor_structure(left: Any, right: Any, *, path: str) -> None:
    import torch

    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError(f"structure mismatch at {path}")
        for key in left:
            _assert_same_tensor_structure(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise ValueError(f"sequence mismatch at {path}")
        for idx, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_same_tensor_structure(left_item, right_item, path=f"{path}[{idx}]")
        return
    if torch.is_tensor(left):
        if not torch.is_tensor(right):
            raise ValueError(f"tensor/type mismatch at {path}")
        if left.shape != right.shape:
            raise ValueError(f"tensor shape mismatch at {path}")
        if left.dtype != right.dtype:
            raise ValueError(f"tensor dtype mismatch at {path}")
        return
    if left != right:
        raise ValueError(f"non-tensor value mismatch at {path}: {left!r} != {right!r}")


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


def _format_alpha(alpha: float) -> str:
    text = f"{alpha:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


if __name__ == "__main__":
    main()
