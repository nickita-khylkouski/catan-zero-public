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
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


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


def scan_event_payload(
    corpus_dir: Path,
    metadata: Mapping[str, Any],
    *,
    chunk_rows: int = 8192,
) -> dict[str, Any]:
    """Exactly scan physical event columns without materialising the corpus.

    Old v1 corpora predate ``implicit_zero_columns`` and therefore cannot prove
    that a present event file contains useful history.  This sequential scan is
    the migration proof: counts are exact and bound to the already-authenticated
    payload inventory hash plus the physical file sizes.
    """

    root = Path(corpus_dir).resolve(strict=True)
    rows = int(metadata.get("row_count", -1))
    columns = metadata.get("columns")
    if rows < 0 or not isinstance(columns, Mapping):
        raise InformationSurfaceError("memmap metadata lacks row_count/columns")
    if isinstance(chunk_rows, bool) or int(chunk_rows) < 1:
        raise InformationSurfaceError("chunk_rows must be a positive integer")

    specs = {
        "event_tokens": (0, "nonzero_count"),
        "event_mask": (0, "nonzero_count"),
        "event_target_ids": (-1, "nonfill_count"),
    }
    result: dict[str, Any] = {
        "row_count": rows,
        "payload_inventory_sha256": metadata.get("payload_inventory_sha256"),
        "columns": {},
    }
    reclaimable = 0
    for name, (fill, count_name) in specs.items():
        schema = columns.get(name)
        if not isinstance(schema, Mapping):
            result["columns"][name] = {"present": False}
            continue
        if schema.get("kind") == "implicit_constant":
            declared_fill = schema.get("fill")
            if declared_fill != fill:
                raise InformationSurfaceError(
                    f"{name}: implicit fill {declared_fill!r} != expected {fill!r}"
                )
            result["columns"][name] = {
                "present": False,
                "implicit": True,
                count_name: 0,
                "physical_bytes": 0,
            }
            continue
        dtype = np.dtype(str(schema.get("dtype")))
        inner_shape = tuple(int(item) for item in schema.get("inner_shape", ()))
        expected_values = rows * int(np.prod(inner_shape, dtype=np.int64))
        path = root / f"{name}.dat"
        expected_bytes = expected_values * dtype.itemsize
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise InformationSurfaceError(
                f"{path}: size mismatch expected={expected_bytes} "
                f"actual={path.stat().st_size if path.exists() else None}"
            )
        array = np.memmap(path, mode="r", dtype=dtype, shape=(rows, *inner_shape))
        nonfill = 0
        for start in range(0, rows, int(chunk_rows)):
            chunk = np.asarray(array[start : start + int(chunk_rows)])
            nonfill += int(np.count_nonzero(chunk != fill))
        del array
        if nonfill == 0:
            reclaimable += expected_bytes
        result["columns"][name] = {
            "present": True,
            "implicit": False,
            count_name: nonfill,
            "physical_bytes": expected_bytes,
        }
    result["reclaimable_constant_bytes"] = reclaimable
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["scan_sha256"] = "sha256:" + hashlib.sha256(canonical).hexdigest()
    return result


def audit_memmap_metadata(
    metadata: Mapping[str, Any],
    *,
    payload_scan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit authenticated memmap metadata for trainable event history.

    Crucially, physical columns in a v1 corpus are *not* evidence of nonzero
    history.  Without either the v2 implicit-zero declaration or an exact scan,
    the answer is unknown rather than the old fail-open ``True``.
    """

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
    scan_tokens_zero = scan_mask_zero = None
    if payload_scan is not None:
        scanned = payload_scan.get("columns")
        if not isinstance(scanned, Mapping):
            raise InformationSurfaceError("payload scan has no columns mapping")
        token_scan = scanned.get("event_tokens", {})
        mask_scan = scanned.get("event_mask", {})
        scan_tokens_zero = int(token_scan.get("nonzero_count", -1)) == 0
        scan_mask_zero = int(mask_scan.get("nonzero_count", -1)) == 0
        if scan_tokens_zero != scan_mask_zero:
            raise InformationSurfaceError(
                "event token/mask payload scans disagree about zero history"
            )
    proven_zero = (event_tokens_zero and event_mask_zero) or (
        scan_tokens_zero is True and scan_mask_zero is True
    )
    history_trainable: bool | None
    if proven_zero:
        history_trainable = False
    elif payload_scan is not None:
        history_trainable = True
    else:
        history_trainable = None
    return {
        "event_tokens_implicit_zero": event_tokens_zero,
        "event_mask_implicit_zero": event_mask_zero,
        "event_history_trainable": history_trainable,
        "event_payload_verified": payload_scan is not None or proven_zero,
        "payload_scan": payload_scan,
        "limitation": (
            "public event history is absent from every retained row; event_encoder "
            "receives no supervised exposure"
            if proven_zero
            else "event history content is unverified; physical v1 columns may be constant"
            if history_trainable is None
            else None
        ),
    }


def build_report(config: Any, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    architecture = audit_architecture_config(config)
    corpus = audit_memmap_metadata(metadata) if metadata is not None else None
    critical: list[str] = []
    if not architecture["topology_consumed"] and not architecture["action_target_bound"]:
        critical.append("spatial_state_action_aliasing")
    if corpus is not None and corpus["event_history_trainable"] is False:
        critical.append("public_history_absent")
    elif corpus is not None and corpus["event_history_trainable"] is None:
        critical.append("public_history_unverified")
    return {
        "schema": SCHEMA,
        "architecture": architecture,
        "corpus": corpus,
        "critical_information_bottlenecks": critical,
        "safe_for_scale_only_ablation": not critical,
    }


def enforce_graph_history_contract(
    corpus_audit: Mapping[str, Any], *, required: bool
) -> None:
    """Refuse a recipe that declares graph history over absent/unproved data."""

    if not required:
        return
    status = corpus_audit.get("event_history_trainable")
    if status is not True:
        reason = "absent" if status is False else "unverified"
        raise InformationSurfaceError(
            "graph_history_features=true but event history is " + reason
        )


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
    parser.add_argument(
        "--scan-event-payload",
        action="store_true",
        help="exactly scan physical event columns (required to prove old v1 data)",
    )
    parser.add_argument(
        "--require-graph-history",
        action="store_true",
        help="fail unless the corpus proves at least one nonzero history value",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    metadata = None
    payload_scan = None
    if args.corpus_meta is not None:
        metadata = json.loads(args.corpus_meta.read_text())
        if args.scan_event_payload:
            payload_scan = scan_event_payload(args.corpus_meta.parent, metadata)
    elif args.scan_event_payload:
        parser.error("--scan-event-payload requires --corpus-meta")
    report = build_report(_load_checkpoint_config(args.checkpoint), metadata)
    if metadata is not None and payload_scan is not None:
        report["corpus"] = audit_memmap_metadata(
            metadata, payload_scan=payload_scan
        )
        report["critical_information_bottlenecks"] = [
            item
            for item in report["critical_information_bottlenecks"]
            if item != "public_history_unverified"
        ]
        if report["corpus"]["event_history_trainable"] is False:
            report["critical_information_bottlenecks"].append(
                "public_history_absent"
            )
        report["safe_for_scale_only_ablation"] = not report[
            "critical_information_bottlenecks"
        ]
    if args.require_graph_history:
        if report["corpus"] is None:
            parser.error("--require-graph-history requires --corpus-meta")
        enforce_graph_history_contract(report["corpus"], required=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
