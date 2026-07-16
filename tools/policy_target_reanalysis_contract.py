"""Strict learner-side authentication for policy-target reanalysis overlays."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
from pathlib import Path

import numpy as np

from catan_zero.rl.target_reliability import TARGET_RELIABILITY_COLUMNS

SCHEMA = "a1-policy-target-reanalysis-merged-v2"
REWRITTEN = frozenset(
    {
        "teacher_name",
        "target_policy",
        "target_policy_mask",
        "target_scores",
        "target_scores_mask",
        "root_value",
        "root_value_mask",
        "root_prior_value",
        "root_prior_value_mask",
        "prior_policy",
        "simulations_used",
        "used_full_search",
        "search_evidence_version",
        "search_evidence_offsets",
        "search_visit_counts_flat",
        "search_completed_q_flat",
        "search_prior_policy_flat",
        "trajectory_producer_checkpoint_sha256",
        "target_reanalyzer_checkpoint_sha256",
        "target_reanalysis_search_config_sha256",
        "target_reanalysis_plan_sha256",
        *TARGET_RELIABILITY_COLUMNS,
    }
)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _value_sha(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


def _columns_sha(arrays: dict[str, np.ndarray], keys: set[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode() + b"\0")
        buffer = io.BytesIO()
        np.lib.format.write_array(buffer, np.asarray(arrays[key]), allow_pickle=True)
        digest.update(buffer.getvalue())
    return "sha256:" + digest.hexdigest()


def validate_policy_target_reanalysis_manifest(
    manifest_path: Path,
) -> dict[str, object] | None:
    """Replay the complete overlay contract before returning any shard path."""
    try:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"cannot read teacher manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA:
        return None
    unsigned = {
        k: v
        for k, v in payload.items()
        if k not in {"manifest_sha256", "manifest_hmac_sha256"}
    }
    if payload.get("manifest_sha256") != _value_sha(unsigned):
        raise SystemExit("policy-target reanalysis manifest semantic hash mismatch")
    plan_binding = payload.get("plan")
    if not isinstance(plan_binding, dict):
        raise SystemExit("policy-target reanalysis plan binding is missing")
    plan_path = Path(manifest_path).parent / str(plan_binding.get("path", ""))
    if _file_sha(plan_path) != plan_binding.get("file_sha256"):
        raise SystemExit("policy-target reanalysis plan file hash mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_unsigned = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if plan.get("plan_sha256") != _value_sha(plan_unsigned) or plan.get(
        "plan_sha256"
    ) != payload.get("plan_sha256"):
        raise SystemExit("policy-target reanalysis plan semantic hash mismatch")
    key_path = Path(str(plan.get("claim_auth_key_path", "")))
    if _file_sha(key_path) != plan.get("claim_auth_key_sha256"):
        raise SystemExit("policy-target reanalysis authentication key mismatch")
    hmac_payload = {k: v for k, v in payload.items() if k != "manifest_hmac_sha256"}
    expected_hmac = (
        "sha256:"
        + hmac.new(
            key_path.read_bytes(),
            json.dumps(
                hmac_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode(),
            hashlib.sha256,
        ).hexdigest()
    )
    if not hmac.compare_digest(
        str(payload.get("manifest_hmac_sha256", "")), expected_hmac
    ):
        raise SystemExit("policy-target reanalysis manifest authentication failed")
    if set(payload.get("rewritten_columns", [])) != REWRITTEN:
        raise SystemExit("policy-target reanalysis rewritten-column allowlist mismatch")
    if payload.get("search_evidence_invalidated") is not True:
        raise SystemExit(
            "policy-target reanalysis did not invalidate stale search evidence"
        )
    for role in ("trajectory_producer", "target_reanalyzer"):
        binding = payload.get(role)
        if not isinstance(binding, dict) or _file_sha(
            Path(str(binding.get("checkpoint_path", "")))
        ) != binding.get("checkpoint_sha256"):
            raise SystemExit(f"policy-target reanalysis {role} checkpoint mismatch")
    inventory = payload.get("payload_inventory")
    if not isinstance(inventory, list) or payload.get(
        "payload_inventory_sha256"
    ) != _value_sha(inventory):
        raise SystemExit("policy-target reanalysis output inventory mismatch")
    if [row.get("path") for row in inventory] != payload.get("shards"):
        raise SystemExit("policy-target reanalysis shard list/inventory mismatch")
    receipts = payload.get("preservation_receipts")
    if not isinstance(receipts, list) or payload.get(
        "preserved_columns_sha256"
    ) != _value_sha(receipts):
        raise SystemExit("policy-target reanalysis preservation receipt mismatch")
    source_by_index = {int(row["index"]): row for row in plan["source_shards"]}
    receipt_by_index = {int(row["shard_index"]): row for row in receipts}
    eligible: dict[int, list[dict[str, object]]] = {}
    for identity in plan["eligible_rows"]:
        eligible.setdefault(int(identity["shard_index"]), []).append(identity)
    for index, record in enumerate(inventory):
        output_path = Path(manifest_path).parent / str(record["path"])
        if _file_sha(output_path) != record.get(
            "sha256"
        ) or output_path.stat().st_size != int(record.get("bytes", -1)):
            raise SystemExit("policy-target reanalysis output shard hash/size mismatch")
        source = source_by_index[index]
        source_path = Path(str(source["path"]))
        if _file_sha(source_path) != source.get("sha256"):
            raise SystemExit("policy-target reanalysis source shard hash mismatch")
        with np.load(source_path, allow_pickle=True) as raw:
            source_arrays = {key: raw[key] for key in raw.files}
        with np.load(output_path, allow_pickle=True) as raw:
            output_arrays = {key: raw[key] for key in raw.files}
        stale_evidence = {
            "search_evidence_version",
            "search_evidence_offsets",
            "search_visit_counts_flat",
            "search_completed_q_flat",
            "search_prior_policy_flat",
        }.intersection(output_arrays)
        if stale_evidence:
            raise SystemExit(
                "policy-target reanalysis retained stale search evidence: "
                + ", ".join(sorted(stale_evidence))
            )
        preserved = set(source_arrays) - REWRITTEN
        digest = _columns_sha(source_arrays, preserved)
        if (
            _columns_sha(output_arrays, preserved) != digest
            or receipt_by_index[index].get("preserved_columns_sha256") != digest
        ):
            raise SystemExit("policy-target reanalysis changed a preserved column")
        for identity in eligible.get(index, []):
            row = int(identity["row_index"])
            if (
                str(output_arrays["teacher_name"][row]) != "policy_target_reanalysis"
                or not bool(output_arrays["root_value_mask"][row])
                or not bool(output_arrays["root_prior_value_mask"][row])
                or not np.isfinite(float(output_arrays["root_value"][row]))
                or not -1.0 <= float(output_arrays["root_value"][row]) <= 1.0
                or not np.isfinite(float(output_arrays["root_prior_value"][row]))
                or not -1.0 <= float(output_arrays["root_prior_value"][row]) <= 1.0
                or not bool(
                    np.asarray(output_arrays["target_policy_mask"])[row][
                        np.asarray(output_arrays["legal_action_ids"])[row] >= 0
                    ].all()
                )
                or str(output_arrays["trajectory_producer_checkpoint_sha256"][row])
                != payload["trajectory_producer"]["checkpoint_sha256"]
                or str(output_arrays["target_reanalyzer_checkpoint_sha256"][row])
                != payload["target_reanalyzer"]["checkpoint_sha256"]
            ):
                raise SystemExit(
                    "policy-target reanalysis row provenance/masks mismatch"
                )
    return {
        "schema_version": payload["schema_version"],
        "manifest_file_sha256": _file_sha(Path(manifest_path)),
        "manifest_sha256": payload["manifest_sha256"],
        "plan_sha256": payload["plan_sha256"],
        "trajectory_producer": payload["trajectory_producer"],
        "target_reanalyzer": payload["target_reanalyzer"],
        "search_config_sha256": payload["search_config_sha256"],
        "payload_inventory_sha256": payload["payload_inventory_sha256"],
        "row_identity_sha256": payload["row_identity_sha256"],
    }
