#!/usr/bin/env python3
"""Audit whether an entity-graph learner can observe the information it names.

This is deliberately a *contract* audit, not a strength proxy.  A larger model
cannot learn topology that never enters its forward pass, and an event encoder
cannot learn public history from a corpus whose event columns are authenticated
constants.  The report makes those two failure modes explicit before an
architecture or learner arm spends GPU time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "entity-graph-information-surface-audit-v1"


class InformationSurfaceError(ValueError):
    """Raised when audit inputs do not satisfy their declared schema."""


def _config_value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def audit_architecture_config(config: Any) -> dict[str, Any]:
    """Return the exact topology/action/history information contract.

    The historical ``transformer`` trunk receives token features only.  Its
    topology arrays are intentionally excluded from the model batch, so its
    state readout is permutation-invariant within each entity type.  A local
    action is spatially bound only when target gather/cross-attention/direct
    edge policy is active (relational trunks enable the binding themselves).
    """

    trunk = str(_config_value(config, "state_trunk", "transformer") or "transformer")
    if trunk not in {"transformer", "rrt", "resrgcn"}:
        raise InformationSurfaceError(f"unknown state_trunk {trunk!r}")
    relational = trunk != "transformer"
    gather = relational or bool(_config_value(config, "action_target_gather", False))
    cross_layers = (
        int(_config_value(config, "relational_action_cross_layers", 1))
        if relational
        else int(_config_value(config, "action_cross_attention_layers", 0))
    )
    edge_head = (
        relational and bool(_config_value(config, "relational_edge_policy_head", True))
    ) or bool(_config_value(config, "edge_policy_head", False))
    value_pool = bool(_config_value(config, "value_attention_pool", False))
    target_bound = gather or cross_layers > 0 or edge_head
    topology_consumed = relational

    limitations: list[str] = []
    if not topology_consumed:
        limitations.append(
            "state trunk is invariant to within-type vertex/edge permutation; "
            "hex_vertex_ids, hex_edge_ids, and edge_vertex_ids do not enter forward"
        )
    if not target_bound:
        limitations.append(
            "policy has no learned legal-action-to-target-entity binding; local "
            "board effects reach it only through CLS and handcrafted action context"
        )
    if not value_pool:
        limitations.append("value reads only the single CLS state token")

    return {
        "state_trunk": trunk,
        "topology_consumed": topology_consumed,
        "action_target_gather": gather,
        "action_cross_attention_layers": cross_layers,
        "edge_policy_head": edge_head,
        "action_target_bound": target_bound,
        "value_attention_pool": value_pool,
        "limitations": limitations,
    }


def audit_memmap_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Audit authenticated memmap metadata for trainable event history."""

    implicit = metadata.get("implicit_zero_columns", ())
    if isinstance(implicit, (str, bytes)) or not isinstance(implicit, (list, tuple)):
        raise InformationSurfaceError("implicit_zero_columns must be a sequence")
    implicit_names = {str(name) for name in implicit}
    event_tokens_zero = "event_tokens" in implicit_names
    event_mask_zero = "event_mask" in implicit_names
    if event_tokens_zero != event_mask_zero:
        raise InformationSurfaceError(
            "event_tokens and event_mask must be implicit-zero together"
        )
    return {
        "event_tokens_implicit_zero": event_tokens_zero,
        "event_mask_implicit_zero": event_mask_zero,
        "event_history_trainable": not (event_tokens_zero and event_mask_zero),
        "limitation": (
            "public event history is absent from every retained row; event_encoder "
            "receives no supervised exposure"
            if event_tokens_zero
            else None
        ),
    }


def build_report(config: Any, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    architecture = audit_architecture_config(config)
    corpus = audit_memmap_metadata(metadata) if metadata is not None else None
    critical: list[str] = []
    if not architecture["topology_consumed"] and not architecture["action_target_bound"]:
        critical.append("spatial_state_action_aliasing")
    if corpus is not None and not corpus["event_history_trainable"]:
        critical.append("public_history_absent")
    return {
        "schema": SCHEMA,
        "architecture": architecture,
        "corpus": corpus,
        "critical_information_bottlenecks": critical,
        "safe_for_scale_only_ablation": not critical,
    }


def _load_checkpoint_config(path: Path) -> Any:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "config" not in payload:
        raise InformationSurfaceError(f"{path}: checkpoint has no config")
    return payload["config"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus-meta", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    metadata = None
    if args.corpus_meta is not None:
        metadata = json.loads(args.corpus_meta.read_text())
    report = build_report(_load_checkpoint_config(args.checkpoint), metadata)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
