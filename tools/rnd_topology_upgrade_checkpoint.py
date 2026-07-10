#!/usr/bin/env python3
"""Warm-grow an EntityGraph checkpoint with zero-init topology adapters.

This is an R&D-only, append-only checkpoint transaction. It preserves every
incumbent tensor exactly, adds only adapter tensors, and records source/output
hashes plus the resolved adapter configuration. It does not train or promote a
model.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _path in (_ROOT, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from catan_zero.rl.config_serialization import (  # noqa: E402
    config_from_dict,
    config_to_dict,
)
from catan_zero.rl.entity_token_policy import (  # noqa: E402
    EntityGraphConfig,
    EntityGraphPolicy,
)


SCHEMA_VERSION = "catan-zero-topology-checkpoint-upgrade/v1"


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
    if (
        "model" not in payload
        or "config" not in payload
        or "static_action_features" not in payload
    ):
        raise ValueError(f"{path} lacks required EntityGraph checkpoint fields")
    return payload


def upgrade_checkpoint(
    source: str | Path,
    output: str | Path,
    *,
    layers: str,
    kind: str,
    width: int,
    bases: int,
    heads: int,
    share_weights: bool = False,
    edge_control: str = "true_topology",
) -> dict:
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if source_path == output_path:
        raise ValueError(
            "topology upgrade output must differ from its source checkpoint"
        )
    if output_path.exists():
        raise FileExistsError(f"topology upgrade refuses to overwrite {output_path}")
    payload = _load_checkpoint(source_path)
    config = config_from_dict(EntityGraphConfig, payload["config"])
    if str(getattr(config, "topology_adapter_layers", "") or "").strip():
        raise ValueError("source checkpoint already contains topology adapters")
    upgraded_config = replace(
        config,
        topology_adapter_layers=str(layers),
        topology_adapter_kind=str(kind),
        topology_adapter_width=int(width),
        topology_adapter_bases=int(bases),
        topology_adapter_heads=int(heads),
        topology_adapter_share_weights=bool(share_weights),
        topology_adapter_edge_control=str(edge_control),
    )
    static = payload["static_action_features"]
    if hasattr(static, "detach"):
        static = static.detach().cpu().numpy()
    policy = EntityGraphPolicy(
        upgraded_config,
        np.asarray(static, dtype=np.float32),
        seed=0,
        device="cpu",
    )
    missing, unexpected = policy.model.load_state_dict(payload["model"], strict=False)
    allowed_prefixes = ("topology_adapters.", "topology_adapter_shared.")
    invalid_missing = [
        name for name in missing if not name.startswith(allowed_prefixes)
    ]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "topology upgrade tensor mismatch: "
            f"invalid_missing={invalid_missing[:8]} unexpected={unexpected[:8]}"
        )
    if not missing:
        raise RuntimeError("topology upgrade added no adapter tensors")

    for name, tensor in payload["model"].items():
        if not torch.equal(policy.model.state_dict()[name].cpu(), tensor.cpu()):
            raise RuntimeError(
                f"incumbent tensor changed during topology upgrade: {name}"
            )

    source_sha = sha256_file(source_path)
    implementation_sha256 = {
        path: hashlib.sha256((_ROOT / path).read_bytes()).hexdigest()
        for path in (
            "src/catan_zero/rl/entity_token_policy.py",
            "src/catan_zero/rl/sparse_topology_adapter.py",
            "tools/rnd_topology_upgrade_checkpoint.py",
        )
    }
    upgraded_payload = dict(payload)
    upgraded_payload["config"] = config_to_dict(upgraded_config)
    upgraded_payload["model"] = policy.model.state_dict()
    upgraded_payload["topology_adapter_upgrade"] = {
        "schema_version": SCHEMA_VERSION,
        "source_checkpoint_sha256": source_sha,
        "implementation_sha256": implementation_sha256,
        "layers": str(layers),
        "kind": str(kind),
        "width": int(width),
        "bases": int(bases),
        "heads": int(heads),
        "share_weights": bool(share_weights),
        "edge_control": str(edge_control),
        "missing_adapter_tensors": sorted(missing),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        torch.save(upgraded_payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    report = {
        **upgraded_payload["topology_adapter_upgrade"],
        "source_checkpoint": str(source_path),
        "output_checkpoint": str(output_path),
        "output_checkpoint_sha256": sha256_file(output_path),
        "parameter_count": sum(
            parameter.numel() for parameter in policy.model.parameters()
        ),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="2,4")
    parser.add_argument(
        "--kind",
        choices=("basis_mean_v1", "local_attention_v2"),
        default="local_attention_v2",
    )
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--bases", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--share-weights", action="store_true")
    parser.add_argument(
        "--edge-control",
        choices=("true_topology", "self_message", "type_degree_preserving_rewire"),
        default="true_topology",
    )
    parser.add_argument("--report", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = upgrade_checkpoint(
        args.source,
        args.output,
        layers=args.layers,
        kind=args.kind,
        width=args.width,
        bases=args.bases,
        heads=args.heads,
        share_weights=args.share_weights,
        edge_control=args.edge_control,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
