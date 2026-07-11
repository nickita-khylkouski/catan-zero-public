#!/usr/bin/env python3
"""Warm-grow a plain Transformer checkpoint with fixed-K latent deliberation.

This R&D-only, versioned transaction preserves every incumbent tensor exactly
and adds only the fixed-K deliberation tensors.  The added residual fusion is
zero-initialised, so the transaction can prove exact policy and value-readout
equivalence before atomically publishing the expanded checkpoint.  An explicit
initialisation seed makes the added state identical for K=1/2/4 (given the same
source and slot count); K controls shared-block recurrence, not parameter shape.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _path in (_ROOT, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE  # noqa: E402
from catan_zero.rl.config_serialization import (  # noqa: E402
    config_from_dict,
    config_to_dict,
)
from catan_zero.rl.entity_token_features import (  # noqa: E402
    EDGE_FEATURE_SIZE,
    EVENT_FEATURE_SIZE,
    GLOBAL_FEATURE_SIZE,
    HEX_FEATURE_SIZE,
    LEGAL_ACTION_FEATURE_SIZE,
    PLAYER_FEATURE_SIZE,
    VERTEX_FEATURE_SIZE,
)
from catan_zero.rl.entity_token_policy import (  # noqa: E402
    EntityGraphConfig,
    EntityGraphPolicy,
)


SCHEMA_VERSION = "catan-zero-transformer-latent-checkpoint-upgrade/v1"
PROVENANCE_KEY = "transformer_latent_deliberation_upgrade"
_DELIBERATION_PREFIXES = (
    "deliberation_slots",
    "deliberation_block.",
    "deliberation_fusion_norm.",
    "deliberation_fusion.",
)
_REQUIRED_EQUIVALENT_OUTPUTS = frozenset(
    {"logits", "value", "final_vp", "q_values"}
)
_SYNTHETIC_BATCH_SEED = 20260710


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("policy_type") != "entity_graph":
        raise ValueError(f"{path} is not an entity_graph checkpoint")
    required = ("model", "config", "static_action_features")
    if any(name not in payload for name in required):
        raise ValueError(f"{path} lacks required EntityGraph checkpoint fields")
    if not isinstance(payload["model"], dict):
        raise ValueError(f"{path} model state is not a mapping")
    return payload


def _public_synthetic_batch(config: EntityGraphConfig) -> dict[str, torch.Tensor]:
    """Return a deterministic public-schema-shaped entity batch on CPU."""
    generator = torch.Generator(device="cpu").manual_seed(_SYNTHETIC_BATCH_SEED)
    batch_size = 2
    action_count = min(4, int(config.action_size))
    if action_count < 1:
        raise ValueError("source action_size must be positive")
    event_count = 3
    batch: dict[str, torch.Tensor] = {}
    for name, count, width in (
        ("hex", 19, HEX_FEATURE_SIZE),
        ("vertex", 54, VERTEX_FEATURE_SIZE),
        ("edge", 72, EDGE_FEATURE_SIZE),
        ("player", 4, PLAYER_FEATURE_SIZE),
        ("global", 1, GLOBAL_FEATURE_SIZE),
        ("event", event_count, EVENT_FEATURE_SIZE),
    ):
        batch[f"{name}_tokens"] = torch.randn(
            batch_size, count, width, generator=generator
        )
        if name != "global":
            batch[f"{name}_mask"] = torch.ones(batch_size, count, dtype=torch.bool)
    batch["legal_action_tokens"] = torch.randn(
        batch_size, action_count, LEGAL_ACTION_FEATURE_SIZE, generator=generator
    )
    batch["legal_action_context"] = torch.randn(
        batch_size, action_count, CONTEXT_ACTION_FEATURE_SIZE, generator=generator
    )
    targets = torch.full((batch_size, action_count, 4), -1, dtype=torch.long)
    targets[:, :, 1] = torch.arange(action_count).remainder(54)
    batch["legal_action_target_ids"] = targets
    batch["hex_vertex_ids"] = torch.full((batch_size, 19, 6), -1, dtype=torch.long)
    batch["hex_edge_ids"] = torch.full((batch_size, 19, 6), -1, dtype=torch.long)
    batch["edge_vertex_ids"] = torch.full((batch_size, 72, 2), -1, dtype=torch.long)
    batch["hex_vertex_ids"][:, 0, :2] = torch.tensor((0, 1))
    batch["hex_edge_ids"][:, 0, :2] = torch.tensor((0, 1))
    batch["edge_vertex_ids"][:, 0, :] = torch.tensor((0, 1))
    batch["event_target_ids"] = torch.full(
        (batch_size, event_count, 4), -1, dtype=torch.long
    )
    batch["event_target_ids"][:, 0, 1] = 0
    return batch


def _publish_no_overwrite(payload: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp.", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(
                f"Transformer latent upgrade refuses to overwrite {output}"
            ) from error
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def upgrade_checkpoint(
    source: str | Path,
    output: str | Path,
    *,
    steps: int,
    slots: int,
    initialization_seed: int,
) -> dict:
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if source_path == output_path:
        raise ValueError("Transformer latent output must differ from its source")
    if output_path.exists():
        raise FileExistsError(
            f"Transformer latent upgrade refuses to overwrite {output_path}"
        )
    if int(steps) < 1:
        raise ValueError("latent deliberation steps must be >= 1")
    if int(slots) < 1:
        raise ValueError("latent deliberation slots must be >= 1")
    if int(initialization_seed) < 0:
        raise ValueError("initialization_seed must be >= 0")

    payload = _load_checkpoint(source_path)
    config = config_from_dict(EntityGraphConfig, payload["config"])
    if str(getattr(config, "state_trunk", "transformer") or "transformer") != (
        "transformer"
    ):
        raise ValueError("Transformer latent source must use state_trunk='transformer'")
    if int(getattr(config, "latent_deliberation_steps", 0) or 0) != 0:
        raise ValueError("Transformer latent source must be a K=0 checkpoint")
    source_deliberation = [
        str(name)
        for name in payload["model"]
        if str(name).startswith("deliberation_")
    ]
    if source_deliberation:
        raise ValueError(
            "K=0 source unexpectedly contains deliberation tensors: "
            f"{source_deliberation[:4]}"
        )

    upgraded_config = replace(
        config,
        latent_deliberation_steps=int(steps),
        latent_deliberation_slots=int(slots),
    )
    static = payload["static_action_features"]
    if hasattr(static, "detach"):
        static = static.detach().cpu().numpy()
    static_array = np.asarray(static, dtype=np.float32)
    source_policy = EntityGraphPolicy(config, static_array, seed=0, device="cpu")
    source_policy.model.load_state_dict(payload["model"], strict=True)
    upgraded_policy = EntityGraphPolicy(
        upgraded_config,
        static_array,
        seed=int(initialization_seed),
        device="cpu",
    )
    missing, unexpected = upgraded_policy.model.load_state_dict(
        payload["model"], strict=False
    )
    invalid_missing = [
        name for name in missing if not name.startswith(_DELIBERATION_PREFIXES)
    ]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "Transformer latent tensor mismatch: "
            f"invalid_missing={invalid_missing[:8]} unexpected={unexpected[:8]}"
        )
    if not missing:
        raise RuntimeError("Transformer latent upgrade added no deliberation tensors")

    upgraded_state = upgraded_policy.model.state_dict()
    for name, tensor in payload["model"].items():
        if not torch.equal(upgraded_state[name].cpu(), tensor.cpu()):
            raise RuntimeError(f"base tensor changed during latent upgrade: {name}")
    added = sorted(set(upgraded_state) - set(payload["model"]))
    if added != sorted(missing):
        raise RuntimeError(
            "Transformer latent additions differ from missing tensors: "
            f"added={added[:8]} missing={sorted(missing)[:8]}"
        )
    if any(not name.startswith(_DELIBERATION_PREFIXES) for name in added):
        raise RuntimeError(
            f"Transformer latent upgrade added a non-deliberation tensor: {added[:8]}"
        )

    source_policy.model.eval()
    upgraded_policy.model.eval()
    synthetic = _public_synthetic_batch(config)
    with torch.no_grad():
        source_outputs = source_policy.model(synthetic, return_q=True)
        upgraded_outputs = upgraded_policy.model(synthetic, return_q=True)
    if source_outputs.keys() != upgraded_outputs.keys():
        raise RuntimeError(
            "function-preserving verification output keys differ: "
            f"source={sorted(source_outputs)} output={sorted(upgraded_outputs)}"
        )
    missing_required = sorted(_REQUIRED_EQUIVALENT_OUTPUTS - source_outputs.keys())
    if missing_required:
        raise RuntimeError(
            "function-preserving verification lacks required outputs: "
            f"{missing_required}"
        )
    verified_outputs = []
    for name, source_tensor in source_outputs.items():
        if not torch.equal(source_tensor, upgraded_outputs[name]):
            raise RuntimeError(
                f"function-preserving verification failed for output {name!r}"
            )
        verified_outputs.append(str(name))

    source_sha = sha256_file(source_path)
    implementation_sha256 = {
        path: hashlib.sha256((_ROOT / path).read_bytes()).hexdigest()
        for path in (
            "src/catan_zero/rl/entity_token_policy.py",
            "tools/rnd_transformer_latent_upgrade_checkpoint.py",
        )
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "source_checkpoint_sha256": source_sha,
        "implementation_sha256": implementation_sha256,
        "source_steps": 0,
        "steps": int(steps),
        "slots": int(slots),
        "initialization_seed": int(initialization_seed),
        "added_deliberation_tensors": added,
        "function_preserving_verification": {
            "exact": True,
            "synthetic_batch_seed": _SYNTHETIC_BATCH_SEED,
            "verified_outputs": sorted(verified_outputs),
        },
    }
    upgraded_payload = dict(payload)
    upgraded_payload["config"] = config_to_dict(upgraded_config)
    upgraded_payload["model"] = upgraded_state
    upgraded_payload[PROVENANCE_KEY] = provenance
    _publish_no_overwrite(upgraded_payload, output_path)

    return {
        **provenance,
        "source_checkpoint": str(source_path),
        "output_checkpoint": str(output_path),
        "output_checkpoint_sha256": sha256_file(output_path),
        "source_parameter_count": sum(
            parameter.numel() for parameter in source_policy.model.parameters()
        ),
        "output_parameter_count": sum(
            parameter.numel() for parameter in upgraded_policy.model.parameters()
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--initialization-seed", required=True, type=int)
    parser.add_argument("--report", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.report and Path(args.report).exists():
        raise FileExistsError(
            f"Transformer latent upgrade refuses to overwrite {args.report}"
        )
    report = upgrade_checkpoint(
        args.source,
        args.output,
        steps=args.steps,
        slots=args.slots,
        initialization_seed=args.initialization_seed,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            report_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
