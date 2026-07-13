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
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "entity-graph-information-surface-audit-v2"
TRAINING_CONTRACT_SCHEMA = "a1-training-event-history-contract-v1"
NATIVE_INFERENCE_CONTRACT_SCHEMA = "native-entity-event-history-v1"
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")


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
    topology_adapter = bool(
        _config_value(config, "topology_residual_adapter", False)
    )
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
    aux_subgoal_heads = bool(_config_value(config, "aux_subgoal_heads", False))
    aux_settlement_pointer_head = bool(
        _config_value(config, "aux_settlement_pointer_head", False)
    )
    if aux_settlement_pointer_head and not aux_subgoal_heads:
        raise InformationSurfaceError(
            "aux_settlement_pointer_head requires aux_subgoal_heads"
        )
    target_bound = gather or cross_layers > 0 or edge_head
    topology_consumed = relational or topology_adapter

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
    if aux_subgoal_heads and not aux_settlement_pointer_head:
        limitations.append(
            "next-settlement logits classify an absolute vertex id from "
            "permutation-invariant CLS even though vertex tokens carry no "
            "canonical id or coordinate"
        )

    return {
        "state_trunk": trunk,
        "topology_consumed": topology_consumed,
        "topology_residual_adapter": topology_adapter,
        "action_target_gather": gather,
        "action_cross_attention_layers": cross_layers,
        "edge_policy_head": edge_head,
        "action_target_bound": target_bound,
        "value_attention_pool": value_pool,
        "aux_subgoal_heads": aux_subgoal_heads,
        "aux_settlement_pointer_head": aux_settlement_pointer_head,
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
        metadata_rows = metadata.get("row_count")
        scan_rows = payload_scan.get("row_count")
        if (
            isinstance(metadata_rows, bool)
            or not isinstance(metadata_rows, int)
            or isinstance(scan_rows, bool)
            or not isinstance(scan_rows, int)
            or scan_rows != metadata_rows
        ):
            raise InformationSurfaceError(
                "payload scan row count does not match memmap metadata"
            )
        metadata_inventory = metadata.get("payload_inventory_sha256")
        scan_inventory = payload_scan.get("payload_inventory_sha256")
        if (
            not isinstance(metadata_inventory, str)
            or _SHA256_RE.fullmatch(metadata_inventory) is None
            or scan_inventory != metadata_inventory
        ):
            raise InformationSurfaceError(
                "payload scan is not bound to the authenticated payload inventory"
            )
        scanned = payload_scan.get("columns")
        if not isinstance(scanned, Mapping):
            raise InformationSurfaceError("payload scan has no columns mapping")
        token_scan = scanned.get("event_tokens", {})
        mask_scan = scanned.get("event_mask", {})
        token_nonzero = token_scan.get("nonzero_count")
        mask_nonzero = mask_scan.get("nonzero_count")
        if (
            isinstance(token_nonzero, bool)
            or not isinstance(token_nonzero, int)
            or token_nonzero < 0
            or isinstance(mask_nonzero, bool)
            or not isinstance(mask_nonzero, int)
            or mask_nonzero < 0
        ):
            raise InformationSurfaceError(
                "payload scan must contain exact nonnegative event token/mask counts"
            )
        scan_tokens_zero = token_nonzero == 0
        scan_mask_zero = mask_nonzero == 0
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


def build_report(
    config: Any, metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    architecture = audit_architecture_config(config)
    corpus = audit_memmap_metadata(metadata) if metadata is not None else None
    critical: list[str] = []
    if (
        not architecture["topology_consumed"]
        and not architecture["action_target_bound"]
    ):
        critical.append("spatial_state_action_aliasing")
    elif not architecture["topology_consumed"]:
        # Target gather repairs only the policy's action-local lookup. The
        # CLS/value path is still exactly invariant to within-type vertex/edge
        # permutations and cannot represent board connectivity.
        critical.append("spatial_state_topology_aliasing")
    elif not architecture["action_target_bound"]:
        # Conversely, topology-aware state tokens do not give the legacy
        # action head a direct semantic target lookup.
        critical.append("action_target_aliasing")
    if (
        architecture["aux_subgoal_heads"]
        and not architecture["aux_settlement_pointer_head"]
    ):
        critical.append("settlement_aux_target_aliasing")
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


def native_inference_event_history_capability() -> dict[str, Any]:
    """Describe the checked-in native inference event-information surface.

    This is deliberately not a caller-controlled boolean.  Both native paths
    are constant-empty in the current source: the Rust featurizer allocates a
    false ``event_mask`` and the Rust-snapshot adapter supplies ``event_log=[]``.
    The entity schema is recorded so a future implementation must deliberately
    revise this contract when it starts carrying public events end to end.
    """

    from catan_zero.rl.entity_token_features import ENTITY_TOKEN_SCHEMA_VERSION

    return {
        "schema": NATIVE_INFERENCE_CONTRACT_SCHEMA,
        "entity_token_schema": ENTITY_TOKEN_SCHEMA_VERSION,
        "available": False,
        "providers": [
            "catanatron_rs.build_entity_features_flat",
            "catan_zero.search.neural_rust_mcts._entity_payload_from_rust_snapshot",
        ],
        "evidence": [
            "native Rust entity featurizer emits constant-zero event_tokens/event_mask",
            "Rust snapshot adapter emits an empty event_log",
        ],
    }


def build_a1_training_event_history_contract(
    component_metadata: Mapping[str, Mapping[str, Any]],
    *,
    graph_history_features: bool,
    event_history_consumer_enabled: bool,
    empty_payload_inventory_acknowledgements: Sequence[str] = (),
    component_payload_scans: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Bind an A1 learner's history claim to authenticated corpus payloads.

    ``graph_history_features`` names a legacy flat-observation *shape*, not the
    entity model's event consumer.  ``EntityGraphPolicy`` owns an event encoder
    regardless of that flag.  Existing A1 corpora have physical, all-zero
    legacy event columns, so silently calling that encoder trainable is false.

    This contract keeps those two facts separate.  Nonzero history must be
    proven by metadata or an exact payload scan.  Empty/unverified legacy
    components may be used only when the operator explicitly acknowledges the
    exact authenticated ``payload_inventory_sha256`` for every such component.
    The acknowledgement authorizes shape-compatible training; it never turns
    empty history into trainable history.

    The caller must pass metadata only after the normal A1 payload-inventory
    authentication.  Requiring component ids as mapping keys makes a composite
    report stable and prevents positional ambiguity.
    """

    if not isinstance(component_metadata, Mapping) or not component_metadata:
        raise InformationSurfaceError(
            "A1 event-history contract requires at least one named component"
        )
    if isinstance(empty_payload_inventory_acknowledgements, (str, bytes)):
        raise InformationSurfaceError(
            "empty-history payload acknowledgements must be a sequence"
        )
    acknowledgements = [
        str(value) for value in empty_payload_inventory_acknowledgements
    ]
    if any(_SHA256_RE.fullmatch(value) is None for value in acknowledgements):
        raise InformationSurfaceError(
            "empty-history payload acknowledgements must be sha256:<64 lowercase hex>"
        )
    if len(set(acknowledgements)) != len(acknowledgements):
        raise InformationSurfaceError(
            "empty-history payload acknowledgements must not contain duplicates"
        )

    payload_scans = component_payload_scans or {}
    if not isinstance(payload_scans, Mapping) or not set(payload_scans).issubset(
        {str(key) for key in component_metadata}
    ):
        raise InformationSurfaceError(
            "event payload scans must be a mapping for named corpus components"
        )

    components: list[dict[str, Any]] = []
    required_acknowledgements: set[str] = set()
    trainable_components = 0
    for raw_component_id, metadata in component_metadata.items():
        component_id = str(raw_component_id)
        if not component_id or not isinstance(metadata, Mapping):
            raise InformationSurfaceError(
                "A1 event-history components require non-empty ids and metadata objects"
            )
        inventory_sha256 = metadata.get("payload_inventory_sha256")
        if (
            not isinstance(inventory_sha256, str)
            or _SHA256_RE.fullmatch(inventory_sha256) is None
        ):
            raise InformationSurfaceError(
                f"component {component_id!r} lacks an authenticated "
                "payload_inventory_sha256"
            )
        payload_scan = payload_scans.get(component_id)
        if payload_scan is not None and (
            payload_scan.get("payload_inventory_sha256") != inventory_sha256
            or int(payload_scan.get("row_count", -1))
            != int(metadata.get("row_count", -2))
        ):
            raise InformationSurfaceError(
                f"component {component_id!r} event payload scan is not bound to "
                "its authenticated inventory and row count"
            )
        corpus_audit = audit_memmap_metadata(metadata, payload_scan=payload_scan)
        observed = corpus_audit["event_history_trainable"]
        if observed is True:
            trainable_components += 1
            disposition = "verified_nonzero"
        else:
            required_acknowledgements.add(inventory_sha256)
            disposition = (
                "machine_proven_empty"
                if observed is False
                else "legacy_payload_unverified"
            )
        components.append(
            {
                "component_id": component_id,
                "payload_inventory_sha256": inventory_sha256,
                "event_history_trainable": observed,
                "pre_acknowledgement_disposition": disposition,
            }
        )

    provided = set(acknowledgements)
    if not event_history_consumer_enabled:
        if provided:
            raise InformationSurfaceError(
                "empty-history acknowledgements are invalid when no event-history "
                "consumer is enabled"
            )
        for component in components:
            component["status"] = "consumer_disabled"
            del component["pre_acknowledgement_disposition"]
        return {
            "schema": TRAINING_CONTRACT_SCHEMA,
            "graph_history_observation_schema": bool(graph_history_features),
            "event_history_consumer_enabled": False,
            "training_event_history_trainable": False,
            "native_inference": native_inference_event_history_capability(),
            "event_history_end_to_end_usable": False,
            "status": "consumer_disabled",
            "components": components,
            "empty_payload_inventory_acknowledgements": [],
        }

    missing = sorted(required_acknowledgements - provided)
    extra = sorted(provided - required_acknowledgements)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing=" + repr(missing))
        if extra:
            details.append("extra=" + repr(extra))
        raise InformationSurfaceError(
            "graph/history observation schema has absent or unverified event history; "
            "acknowledgements must exactly match authenticated payload inventories "
            + " ".join(details)
        )

    for component in components:
        if component["payload_inventory_sha256"] in required_acknowledgements:
            component["event_history_trainable"] = False
            component["status"] = (
                "empty_acknowledged_machine_proven"
                if component["pre_acknowledgement_disposition"]
                == "machine_proven_empty"
                else "empty_acknowledged_legacy_payload"
            )
        else:
            component["status"] = "verified_nonzero"
        del component["pre_acknowledgement_disposition"]

    any_trainable = trainable_components > 0
    native_capability = native_inference_event_history_capability()
    if any_trainable and native_capability["available"] is not True:
        raise InformationSurfaceError(
            "A1 corpus proves nonzero event history but native inference does not "
            "provide event history; refusing train/deploy information-surface skew"
        )
    if any_trainable and required_acknowledgements:
        status = "partially_trainable_with_empty_components_acknowledged"
    elif any_trainable:
        status = "verified_nonzero"
    else:
        status = "empty_payloads_acknowledged"
    return {
        "schema": TRAINING_CONTRACT_SCHEMA,
        "graph_history_observation_schema": bool(graph_history_features),
        "event_history_consumer_enabled": True,
        "training_event_history_trainable": any_trainable,
        "native_inference": native_capability,
        "event_history_end_to_end_usable": bool(
            any_trainable and native_capability["available"] is True
        ),
        "status": status,
        "components": components,
        "empty_payload_inventory_acknowledgements": sorted(provided),
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
        report["corpus"] = audit_memmap_metadata(metadata, payload_scan=payload_scan)
        report["critical_information_bottlenecks"] = [
            item
            for item in report["critical_information_bottlenecks"]
            if item != "public_history_unverified"
        ]
        if report["corpus"]["event_history_trainable"] is False:
            report["critical_information_bottlenecks"].append("public_history_absent")
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
