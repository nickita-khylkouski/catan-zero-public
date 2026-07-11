#!/usr/bin/env python3
"""Convert npz teacher shards into a flat memmap corpus for streaming training.

``tools/train_bc.py``'s ``load_teacher_data`` materialises the whole corpus in
host RAM: it accumulates per-column lists, ``np.concatenate``s them (a transient
2x spike), and pads every ragged per-decision column (``legal_action_ids`` and
friends) to the global maximum legal width (54). Mean legal width in the
raw-selfplay corpus is ~4.8, so >90% of the ragged storage is padding, and the
whole set has to be resident at once. That ceiling OOM'd a 32.6M-row corpus on a
708GB host.

This tool performs the one-time conversion into a directory of flat files that
``MemmapCorpus`` (see ``train_bc.py``) streams per batch:

* Fixed-width columns (``obs``, board/entity tokens, scalars, VP arrays) are
  written as flat ``<col>.dat`` files, one row after another -- reloaded as an
  ``(N, *inner_shape)`` ``np.memmap``.
* Ragged per-decision columns are stored TRIMMED to each row's true legal count
  (no padding on disk) in flat ``<col>.dat`` value files, sharing a single
  ``row_offsets.dat`` (``int64``, ``N+1``). The batch collate re-pads them to the
  global legal width so batches are byte-identical to the in-RAM loader.
* Unicode columns are factorised into an ``int32`` ``<col>.codes.dat`` plus a
  category list in ``corpus_meta.json``.

The normalisation (dtypes, defaults, string coercion, schema checks) is reused
verbatim from ``train_bc._normalize_teacher_shard`` so the reconstructed batches
match ``load_teacher_data`` exactly. Trimming is lossless only because every
ragged column is exactly its fill value beyond the per-row legal count; the
converter asserts this per shard and aborts if a shard ever violates it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from catan_zero.rl.aux_subgoal_targets import AUX_TARGET_KEYS

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from train_bc import (  # noqa: E402  (sibling module bootstrap above)
    _load_validation_game_seed_manifest_for_training,
    _load_npz,
    _normalize_teacher_shard,
    _teacher_shard_files,
)

MEMMAP_CORPUS_SCHEMA = "memmap_corpus_v1"
MEMMAP_CORPUS_IMPLICIT_SCHEMA = "memmap_corpus_v2"

# Event history is currently disabled in the retained teacher data, but the
# normal entity schema still carries a dense (64, 41) fp16 tensor and a (64,)
# bool mask for every row.  The v2 format can represent those *proven-zero*
# columns without a data file.  Keep this list deliberately narrow: other
# constant-looking columns may have different fill semantics.
IMPLICIT_ZERO_EVENT_COLUMNS = frozenset({"event_tokens", "event_mask"})
MEMMAP_PAYLOAD_INVENTORY_SCHEMA = "memmap-payload-inventory-v1"
A1_SELECTED_GAMES_SCHEMA = "a1-selected-training-games-v1"
DUAL_ARM_SELECTED_GAMES_SCHEMA = "a1-dual-arm-selected-training-games-v1"
DUAL_ARM_AUDIT_SCHEMA = "a1-dual-arm-post-wave-audit-v1"
DUAL_ARM_SUBSET_CATEGORY_COUNTS = {
    ("n128", "full-140k"): {
        "current_producer": 112_000,
        "recent_history": 21_000,
        "hard_negative": 7_000,
    },
    ("n128", "matched-56k"): {
        "current_producer": 44_800,
        "recent_history": 8_400,
        "hard_negative": 2_800,
    },
    ("n128", "compute-112k"): {
        "current_producer": 89_600,
        "recent_history": 16_800,
        "hard_negative": 5_600,
    },
    ("n256", "full-140k"): {
        "current_producer": 112_000,
        "recent_history": 21_000,
        "hard_negative": 7_000,
    },
}
A1_SELECTION_RULE = "lowest_seed_complete_per_job"
A1_SELECTED_GAME_COUNT = 12_000
A1_CATEGORY_GAME_COUNTS = {
    "current_producer": 9_600,
    "recent_history": 1_800,
    "hard_negative": 600,
}
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_A1_SELECTED_RECORD_FIELDS = {
    "game_seed",
    "job_id",
    "worker_id",
    "category",
    "producer_checkpoint_sha256",
    "opponent_checkpoint_sha256",
    "split",
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_direct_a1_source_attestations(
    sources: Sequence[Path | str],
) -> list[dict[str, Any]]:
    """Discover the nearest A1 attestation at or above each source root.

    A rendered A1 job writes its
    immutable attestation to ``<output_dir>/a1_contract.json``. Shards and worker
    manifests may live in descendants such as ``worker_000/``; checking the
    nearest ancestor prevents callers from routing those same bytes through the
    generic conversion path by selecting a nested directory.
    """

    attestations: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for source in sources:
        source_root = Path(source).expanduser().resolve()
        attestation_path = next(
            (
                candidate / "a1_contract.json"
                for candidate in (source_root, *source_root.parents)
                if (candidate / "a1_contract.json").is_file()
            ),
            None,
        )
        if attestation_path is None:
            continue
        try:
            canonical_path = attestation_path.resolve(strict=True)
        except OSError as error:
            raise SystemExit(
                f"cannot resolve A1 source attestation {attestation_path}: {error}"
            ) from error
        if canonical_path in seen_paths:
            continue
        seen_paths.add(canonical_path)
        try:
            payload = json.loads(canonical_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SystemExit(
                f"cannot load A1 source attestation {canonical_path}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise SystemExit(
                f"A1 source attestation {canonical_path} must be a JSON object"
            )
        if payload.get("schema_version") != "a1-generation-job-attestation-v2":
            raise SystemExit(
                f"A1 source attestation {canonical_path} has unsupported schema"
            )
        contract_sha = payload.get("contract_sha256")
        if not isinstance(contract_sha, str) or not _SHA256_RE.fullmatch(contract_sha):
            raise SystemExit(
                f"A1 source attestation {canonical_path} has invalid contract_sha256"
            )
        attestations.append(
            {
                "path": canonical_path,
                "file_sha256": _file_sha256(canonical_path),
                "contract_sha256": contract_sha,
            }
        )
    return attestations


def _memmap_payload_inventory(
    out_dir: Path, schemas: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Hash the exact flat-file payload implied by the corpus column schemas."""

    expected_names = {"row_offsets.dat"}
    expected_names.update(
        f"{name}.codes.dat" if schema["kind"] == "string" else f"{name}.dat"
        for name, schema in schemas.items()
        if schema["kind"] != "implicit_constant"
    )
    actual_names = {
        path.name
        for path in out_dir.iterdir()
        if path.is_file()
        and (path.name.endswith(".dat") or path.name.endswith(".codes.dat"))
    }
    if actual_names != expected_names:
        raise SystemExit(
            "memmap payload filenames differ from the column schema: "
            f"missing={sorted(expected_names - actual_names)} "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    inventory: list[dict[str, Any]] = []
    for filename in sorted(expected_names):
        path = out_dir / filename
        inventory.append(
            {
                "filename": filename,
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return inventory


def _game_seed_set_sha256(seeds: Sequence[int]) -> str:
    canonical = np.asarray(sorted(int(seed) for seed in seeds), dtype="<i8")
    return "sha256:" + hashlib.sha256(canonical.tobytes()).hexdigest()


def _load_a1_selected_game_manifest(path: Path) -> dict[str, Any]:
    """Load and fail-closed validate the immutable A1 game-level selection.

    The post-wave audit selects complete games *before* row expansion.  This
    validator deliberately binds both the canonical record list and the raw
    sidecar bytes so the converter cannot silently train on reserve or
    truncated attempts that happen to share the same shard directory.
    """
    try:
        manifest_path = path.expanduser().resolve(strict=True)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"cannot load selected-game seed manifest {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise SystemExit("selected-game seed manifest must be a JSON object")
    if payload.get("schema_version") == DUAL_ARM_SELECTED_GAMES_SCHEMA:
        return _load_dual_arm_selected_game_manifest(manifest_path, payload)

    expected_fields = {
        "schema_version",
        "a1_contract_sha256",
        "selection_rule",
        "selected_game_count",
        "selected_game_seed_set_sha256",
        "category_game_counts",
        "training_game_count",
        "training_game_seed_set_sha256",
        "validation_game_count",
        "validation_game_seed_set_sha256",
        "records_sha256",
        "records",
    }
    if set(payload) != expected_fields:
        raise SystemExit(
            "selected-game seed manifest fields differ from the exact "
            f"{A1_SELECTED_GAMES_SCHEMA} schema; "
            f"missing={sorted(expected_fields - set(payload))} "
            f"extra={sorted(set(payload) - expected_fields)}"
        )
    if payload["schema_version"] != A1_SELECTED_GAMES_SCHEMA:
        raise SystemExit(
            f"selected-game seed manifest schema must be {A1_SELECTED_GAMES_SCHEMA!r}"
        )
    if payload["selection_rule"] != A1_SELECTION_RULE:
        raise SystemExit(
            f"selected-game seed manifest selection_rule must be {A1_SELECTION_RULE!r}"
        )
    contract_sha = payload["a1_contract_sha256"]
    if not isinstance(contract_sha, str) or not _SHA256_RE.fullmatch(contract_sha):
        raise SystemExit("selected-game seed manifest has invalid a1_contract_sha256")
    if (
        isinstance(payload["selected_game_count"], bool)
        or not isinstance(payload["selected_game_count"], int)
        or payload["selected_game_count"] != A1_SELECTED_GAME_COUNT
    ):
        raise SystemExit(
            f"selected-game seed manifest must declare exactly {A1_SELECTED_GAME_COUNT} games"
        )
    declared_category_counts = payload["category_game_counts"]
    if (
        not isinstance(declared_category_counts, dict)
        or declared_category_counts != A1_CATEGORY_GAME_COUNTS
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in declared_category_counts.values()
        )
    ):
        raise SystemExit(
            "selected-game seed manifest category_game_counts must be exactly "
            f"{A1_CATEGORY_GAME_COUNTS}"
        )

    records = payload["records"]
    if not isinstance(records, list) or len(records) != A1_SELECTED_GAME_COUNT:
        raise SystemExit(
            f"selected-game seed manifest records must contain exactly "
            f"{A1_SELECTED_GAME_COUNT} entries"
        )
    normalized_records: list[dict[str, Any]] = []
    prior_key: tuple[int, str] | None = None
    seen_seeds: set[int] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != _A1_SELECTED_RECORD_FIELDS:
            raise SystemExit(
                f"selected-game record {index} fields differ from the exact schema"
            )
        seed = record["game_seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SystemExit(f"selected-game record {index} game_seed must be an integer")
        if seed < np.iinfo(np.int64).min or seed > np.iinfo(np.int64).max:
            raise SystemExit(f"selected-game record {index} game_seed is outside int64")
        job_id = record["job_id"]
        if not isinstance(job_id, str) or not job_id:
            raise SystemExit(f"selected-game record {index} has invalid job_id")
        key = (seed, job_id)
        if prior_key is not None and key <= prior_key:
            raise SystemExit(
                "selected-game records must be strictly sorted by "
                f"(game_seed, job_id) (drift at index {index})"
            )
        prior_key = key
        if seed in seen_seeds:
            raise SystemExit(f"selected-game record {index} duplicates game_seed {seed}")
        seen_seeds.add(seed)
        if not isinstance(record["worker_id"], str) or not record["worker_id"]:
            raise SystemExit(f"selected-game record {index} has invalid worker_id")
        if record["category"] not in A1_CATEGORY_GAME_COUNTS:
            raise SystemExit(f"selected-game record {index} has invalid category")
        producer_sha = record["producer_checkpoint_sha256"]
        if not isinstance(producer_sha, str) or not _SHA256_RE.fullmatch(producer_sha):
            raise SystemExit(
                f"selected-game record {index} has invalid producer_checkpoint_sha256"
            )
        opponent_shas = record["opponent_checkpoint_sha256"]
        if (
            not isinstance(opponent_shas, list)
            or not opponent_shas
            or any(
                not isinstance(value, str) or not _SHA256_RE.fullmatch(value)
                for value in opponent_shas
            )
            or opponent_shas != sorted(set(opponent_shas))
        ):
            raise SystemExit(
                f"selected-game record {index} has invalid opponent_checkpoint_sha256"
            )
        if record["split"] not in {"train", "validation"}:
            raise SystemExit(f"selected-game record {index} has invalid split")
        normalized_records.append(dict(record))

    actual_counts = dict(Counter(record["category"] for record in normalized_records))
    if actual_counts != A1_CATEGORY_GAME_COUNTS:
        raise SystemExit(
            "selected-game record category counts do not match the declared A1 quotas: "
            f"{actual_counts}"
        )
    all_seeds = [record["game_seed"] for record in normalized_records]
    training_seeds = [
        record["game_seed"]
        for record in normalized_records
        if record["split"] == "train"
    ]
    validation_seeds = [
        record["game_seed"]
        for record in normalized_records
        if record["split"] == "validation"
    ]
    if not training_seeds or not validation_seeds:
        raise SystemExit(
            "selected-game seed manifest must contain non-empty train and validation splits"
        )
    if (
        isinstance(payload["training_game_count"], bool)
        or not isinstance(payload["training_game_count"], int)
        or payload["training_game_count"] != len(training_seeds)
        or isinstance(payload["validation_game_count"], bool)
        or not isinstance(payload["validation_game_count"], int)
        or payload["validation_game_count"] != len(validation_seeds)
        or len(training_seeds) + len(validation_seeds) != A1_SELECTED_GAME_COUNT
    ):
        raise SystemExit("selected-game seed manifest split counts mismatch")

    actual_selected_seed_sha = _game_seed_set_sha256(all_seeds)
    if payload["selected_game_seed_set_sha256"] != actual_selected_seed_sha:
        raise SystemExit(
            "selected-game seed manifest selected_game_seed_set_sha256 mismatch: "
            f"declared={payload['selected_game_seed_set_sha256']!r}, "
            f"actual={actual_selected_seed_sha!r}"
        )
    actual_training_seed_sha = _game_seed_set_sha256(training_seeds)
    if payload["training_game_seed_set_sha256"] != actual_training_seed_sha:
        raise SystemExit(
            "selected-game seed manifest training_game_seed_set_sha256 mismatch: "
            f"declared={payload['training_game_seed_set_sha256']!r}, "
            f"actual={actual_training_seed_sha!r}"
        )
    actual_validation_seed_sha = _game_seed_set_sha256(validation_seeds)
    if payload["validation_game_seed_set_sha256"] != actual_validation_seed_sha:
        raise SystemExit(
            "selected-game seed manifest validation_game_seed_set_sha256 mismatch: "
            f"declared={payload['validation_game_seed_set_sha256']!r}, "
            f"actual={actual_validation_seed_sha!r}"
        )
    actual_records_sha = _value_sha256(normalized_records)
    if payload["records_sha256"] != actual_records_sha:
        raise SystemExit(
            "selected-game seed manifest records_sha256 mismatch: "
            f"declared={payload['records_sha256']!r}, actual={actual_records_sha!r}"
        )
    return {
        "path": manifest_path,
        "file_sha256": _file_sha256(manifest_path),
        "manifest_sha256": _value_sha256(payload),
        "a1_contract_sha256": contract_sha,
        "selected_game_count": A1_SELECTED_GAME_COUNT,
        "selected_game_seed_set_sha256": actual_selected_seed_sha,
        "training_game_count": len(training_seeds),
        "training_game_seed_set_sha256": actual_training_seed_sha,
        "validation_game_count": len(validation_seeds),
        "validation_game_seed_set_sha256": actual_validation_seed_sha,
        "records_sha256": actual_records_sha,
        "selected_game_seeds": np.asarray(all_seeds, dtype=np.int64),
        "training_game_seeds": np.asarray(training_seeds, dtype=np.int64),
        "validation_game_seeds": np.asarray(validation_seeds, dtype=np.int64),
    }


def _load_dual_arm_selected_game_manifest(
    manifest_path: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    expected = {
        "schema_version", "arm_id", "subset_id", "a1_contract_sha256",
        "selection_rule", "selected_game_count", "selected_game_seed_set_sha256",
        "category_game_counts", "training_game_count", "training_game_seed_set_sha256",
        "validation_game_count", "validation_game_seed_set_sha256", "records_sha256",
        "records", "parent_manifest_sha256",
    }
    if set(payload) != expected:
        raise SystemExit("dual-arm selected-game manifest fields drift")
    arm_id = payload["arm_id"]
    subset_id = payload["subset_id"]
    if arm_id not in {"n128", "n256"} or not isinstance(subset_id, str) or not subset_id:
        raise SystemExit("dual-arm selected-game manifest identity is invalid")
    if (
        not isinstance(payload["a1_contract_sha256"], str)
        or not _SHA256_RE.fullmatch(payload["a1_contract_sha256"])
        or not isinstance(payload["parent_manifest_sha256"], str)
        or not _SHA256_RE.fullmatch(payload["parent_manifest_sha256"])
        or not isinstance(payload["selection_rule"], str)
        or not payload["selection_rule"]
    ):
        raise SystemExit("dual-arm selected-game provenance is invalid")
    counts = payload["category_game_counts"]
    expected_counts = DUAL_ARM_SUBSET_CATEGORY_COUNTS.get((arm_id, subset_id))
    if (
        expected_counts is None
        or
        not isinstance(counts, dict)
        or set(counts) != set(A1_CATEGORY_GAME_COUNTS)
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts.values())
        or counts != expected_counts
    ):
        raise SystemExit("dual-arm selected-game arm/subset category quotas are invalid")
    total = payload["selected_game_count"]
    if isinstance(total, bool) or not isinstance(total, int) or total != sum(counts.values()):
        raise SystemExit("dual-arm selected-game total differs from category quotas")
    records = payload["records"]
    record_fields = _A1_SELECTED_RECORD_FIELDS | {"arm_id"}
    if not isinstance(records, list) or len(records) != total:
        raise SystemExit("dual-arm selected-game records have wrong length")
    prior: tuple[int, str] | None = None
    seeds: set[int] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != record_fields or record.get("arm_id") != arm_id:
            raise SystemExit(f"dual-arm selected-game record {index} identity drift")
        key = (record.get("game_seed"), record.get("job_id"))
        if not isinstance(key[0], int) or isinstance(key[0], bool) or not isinstance(key[1], str):
            raise SystemExit(f"dual-arm selected-game record {index} is malformed")
        if prior is not None and key <= prior:
            raise SystemExit("dual-arm selected-game records are not strictly sorted")
        prior = key
        if key[0] in seeds or record.get("category") not in counts or record.get("split") not in {"train", "validation"}:
            raise SystemExit(f"dual-arm selected-game record {index} duplicates or drifts")
        seeds.add(key[0])
        if (
            not isinstance(record.get("worker_id"), str)
            or not record["worker_id"]
            or not isinstance(record.get("producer_checkpoint_sha256"), str)
            or not _SHA256_RE.fullmatch(record["producer_checkpoint_sha256"])
            or not isinstance(record.get("opponent_checkpoint_sha256"), list)
            or record["opponent_checkpoint_sha256"] != sorted(set(record["opponent_checkpoint_sha256"]))
            or any(not isinstance(value, str) or not _SHA256_RE.fullmatch(value) for value in record["opponent_checkpoint_sha256"])
        ):
            raise SystemExit(f"dual-arm selected-game record {index} provenance drift")
    if dict(Counter(record["category"] for record in records)) != counts:
        raise SystemExit("dual-arm selected-game records differ from category quotas")
    training = [record["game_seed"] for record in records if record["split"] == "train"]
    validation = [record["game_seed"] for record in records if record["split"] == "validation"]
    all_seeds = [record["game_seed"] for record in records]
    checks = {
        "selected_game_seed_set_sha256": _game_seed_set_sha256(all_seeds),
        "training_game_seed_set_sha256": _game_seed_set_sha256(training),
        "validation_game_seed_set_sha256": _game_seed_set_sha256(validation),
        "records_sha256": _value_sha256(records),
    }
    if any(payload[key] != value for key, value in checks.items()):
        raise SystemExit("dual-arm selected-game manifest digest drift")
    if payload["training_game_count"] != len(training) or payload["validation_game_count"] != len(validation):
        raise SystemExit("dual-arm selected-game split count drift")
    return {
        "path": manifest_path, "file_sha256": _file_sha256(manifest_path),
        "manifest_sha256": _value_sha256(payload), "a1_contract_sha256": payload["a1_contract_sha256"],
        "arm_id": arm_id, "subset_id": subset_id, "category_game_counts": counts,
        "selected_game_count": total, "selected_game_seed_set_sha256": checks["selected_game_seed_set_sha256"],
        "training_game_count": len(training), "training_game_seed_set_sha256": checks["training_game_seed_set_sha256"],
        "validation_game_count": len(validation), "validation_game_seed_set_sha256": checks["validation_game_seed_set_sha256"],
        "records_sha256": checks["records_sha256"], "selected_game_seeds": np.asarray(all_seeds, dtype=np.int64),
        "training_game_seeds": np.asarray(training, dtype=np.int64), "validation_game_seeds": np.asarray(validation, dtype=np.int64),
    }


def _load_a1_post_wave_audit(
    path: Path, selected_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Validate the passing audit that authorizes the selected-game sidecar."""
    try:
        audit_path = path.expanduser().resolve(strict=True)
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot load A1 post-wave audit {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit("A1 post-wave audit must be a JSON object")
    audit_schema = payload.get("schema_version")
    is_dual = audit_schema == DUAL_ARM_AUDIT_SCHEMA
    if audit_schema not in {"a1-post-wave-audit-v2", "a1-post-wave-audit-v3", DUAL_ARM_AUDIT_SCHEMA}:
        raise SystemExit(
            "A1 post-wave audit schema must be 'a1-post-wave-audit-v2' or "
            "'a1-post-wave-audit-v3'"
        )
    if is_dual and (
        selected_manifest.get("arm_id") not in {"n128", "n256"}
        or payload.get("arm_id") != selected_manifest.get("arm_id")
        or payload.get("category_game_counts") != selected_manifest.get("category_game_counts")
    ):
        raise SystemExit("dual-arm audit/selection arm or quota mismatch")
    if payload.get("passed") is not True or payload.get("errors") != []:
        raise SystemExit("A1 post-wave audit is not a clean passing report")
    declared_audit_sha = payload.get("audit_sha256")
    actual_audit_sha = _value_sha256(
        {key: value for key, value in payload.items() if key != "audit_sha256"}
    )
    if declared_audit_sha != actual_audit_sha:
        raise SystemExit(
            "A1 post-wave audit audit_sha256 mismatch: "
            f"declared={declared_audit_sha!r}, actual={actual_audit_sha!r}"
        )
    contract_sha = payload.get("contract_sha256")
    if contract_sha != selected_manifest["a1_contract_sha256"]:
        raise SystemExit(
            "A1 audit/selected-game manifest contract hash mismatch: "
            f"audit={contract_sha!r}, manifest={selected_manifest['a1_contract_sha256']!r}"
        )

    shards = payload.get("shards")
    if not isinstance(shards, list):
        raise SystemExit("A1 post-wave audit shards must be a list")
    if payload.get("shard_inventory_sha256") != _value_sha256(shards):
        raise SystemExit("A1 post-wave audit shard_inventory_sha256 mismatch")
    harvest_provenance: dict[str, Any] | None = None
    if audit_schema in {"a1-post-wave-audit-v3", DUAL_ARM_AUDIT_SCHEMA}:
        binding = payload.get("harvest_relocation")
        expected_binding_keys = {
            "path",
            "file_sha256",
            "relocation_sha256",
            "render_sha256",
            "job_identities_sha256",
            "file_inventory_sha256",
        }
        if is_dual:
            expected_binding_keys.add("arm_id")
        if not isinstance(binding, dict) or set(binding) != expected_binding_keys:
            raise SystemExit("relocated A1 audit has an invalid harvest binding")
        try:
            relocation_path = Path(str(binding["path"])).expanduser().resolve(strict=True)
            relocation = json.loads(relocation_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SystemExit(f"cannot load A1 harvest relocation map: {error}") from error
        if not isinstance(relocation, dict):
            raise SystemExit("A1 harvest relocation map must be an object")
        unhashed_relocation = dict(relocation)
        relocation_digest = unhashed_relocation.pop("relocation_sha256", None)
        if (
            relocation.get("schema_version") != "a1-fleet-harvest-relocation-v1"
            or _file_sha256(relocation_path) != binding["file_sha256"]
            or relocation_digest != _value_sha256(unhashed_relocation)
            or relocation_digest != binding["relocation_sha256"]
            or relocation.get("contract_sha256") != contract_sha
            or relocation.get("render_sha256") != binding["render_sha256"]
            or relocation.get("job_identities_sha256")
            != binding["job_identities_sha256"]
            or relocation.get("file_inventory_sha256")
            != binding["file_inventory_sha256"]
            or relocation.get("file_inventory_sha256")
            != _value_sha256(relocation.get("files"))
            or (is_dual and binding.get("arm_id") != selected_manifest["arm_id"])
            or (is_dual and relocation.get("arm_id") != selected_manifest["arm_id"])
        ):
            raise SystemExit("A1 harvest relocation binding/digest mismatch")
        relocation_by_local: dict[Path, dict[str, Any]] = {}
        for index, record in enumerate(relocation.get("files", [])):
            if not isinstance(record, dict):
                raise SystemExit(f"A1 harvest file record {index} is malformed")
            relative = Path(str(record.get("relative_path", "")))
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or relative.parts[0] != "jobs"
            ):
                raise SystemExit(f"A1 harvest file record {index} path is unsafe")
            unresolved_local = relocation_path.parent / relative
            try:
                local = unresolved_local.resolve(strict=True)
            except OSError as error:
                raise SystemExit(
                    f"A1 harvest file record {index} is missing: {error}"
                ) from error
            if local != unresolved_local.absolute() or not local.is_file():
                raise SystemExit(
                    f"A1 harvest file record {index} uses a symlink or non-file"
                )
            if local in relocation_by_local:
                raise SystemExit("A1 harvest relocation repeats a local file")
            relocation_by_local[local] = record
        harvest_provenance = {
            "path": relocation_path,
            "file_sha256": binding["file_sha256"],
            "relocation_sha256": relocation_digest,
            "render_sha256": binding["render_sha256"],
            "job_identities_sha256": binding["job_identities_sha256"],
            "file_inventory_sha256": binding["file_inventory_sha256"],
            **({} if not is_dual else {"arm_id": binding["arm_id"]}),
            "by_local": relocation_by_local,
        }
    elif "harvest_relocation" in payload:
        raise SystemExit("legacy A1 audit must not carry a harvest relocation binding")
    data_shards: list[dict[str, Any]] = []
    contract_attestations: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for index, record in enumerate(shards):
        if not isinstance(record, dict):
            raise SystemExit(f"A1 post-wave audit shard record {index} is not an object")
        kind = record.get("kind")
        if kind not in {"data_shard", "contract_attestation"}:
            continue
        try:
            shard_path = Path(str(record["path"])).expanduser().resolve(strict=True)
        except (KeyError, OSError) as error:
            raise SystemExit(
                f"A1 post-wave audit data shard {index} path is invalid: {error}"
            ) from error
        if shard_path in seen_paths:
            raise SystemExit(
                f"A1 post-wave audit repeats canonical data shard path {shard_path}"
            )
        seen_paths.add(shard_path)
        declared_sha = record.get("sha256")
        if not isinstance(declared_sha, str) or not _SHA256_RE.fullmatch(declared_sha):
            raise SystemExit(
                f"A1 post-wave audit {kind} {index} has invalid sha256"
            )
        if kind == "data_shard":
            if harvest_provenance is not None:
                relocation_record = harvest_provenance["by_local"].get(shard_path)
                if (
                    relocation_record is None
                    or relocation_record.get("sha256") != declared_sha
                    or relocation_record.get("size_bytes") != shard_path.stat().st_size
                ):
                    raise SystemExit(
                        "relocated A1 audit data shard is not identically bound by "
                        f"the harvest map: {shard_path}"
                    )
            data_shards.append({**record, "path": str(shard_path)})
            continue
        actual_sha = _file_sha256(shard_path)
        if actual_sha != declared_sha:
            raise SystemExit(
                f"A1 post-wave audit {kind} {index} byte digest mismatch: "
                f"declared={declared_sha!r}, actual={actual_sha!r}"
            )
        try:
            attestation = json.loads(shard_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SystemExit(
                f"cannot load audited A1 contract attestation {shard_path}: {error}"
            ) from error
        if (
            not isinstance(attestation, dict)
            or attestation.get("schema_version")
            != "a1-generation-job-attestation-v2"
            or attestation.get("contract_sha256") != contract_sha
            or (is_dual and attestation.get("arm_id") != selected_manifest["arm_id"])
        ):
            raise SystemExit(
                "audited A1 source attestation does not bind the audit contract: "
                f"{shard_path}"
            )
        contract_attestations.append({**record, "path": str(shard_path)})
    if not data_shards:
        raise SystemExit("A1 post-wave audit contains no data_shard records")

    selected = payload.get("selected_training_games")
    if not isinstance(selected, dict):
        raise SystemExit("A1 post-wave audit has no selected_training_games binding")
    try:
        bound_manifest_path = Path(str(selected["manifest"])).expanduser().resolve(
            strict=True
        )
    except (KeyError, OSError) as error:
        raise SystemExit(
            f"A1 post-wave audit selected manifest path is invalid: {error}"
        ) from error
    expected_selected = {
        "manifest": selected_manifest["path"],
        "manifest_sha256": selected_manifest["manifest_sha256"],
        "manifest_file_sha256": selected_manifest["file_sha256"],
        "selected_game_count": selected_manifest["selected_game_count"],
        "selected_game_seed_set_sha256": selected_manifest[
            "selected_game_seed_set_sha256"
        ],
        "records_sha256": selected_manifest["records_sha256"],
    }
    actual_selected = {
        "manifest": bound_manifest_path,
        "manifest_sha256": selected.get("manifest_sha256"),
        "manifest_file_sha256": selected.get("manifest_file_sha256"),
        "selected_game_count": selected.get("selected_game_count"),
        "selected_game_seed_set_sha256": selected.get(
            "selected_game_seed_set_sha256"
        ),
        "records_sha256": selected.get("records_sha256"),
    }
    if actual_selected != expected_selected:
        raise SystemExit("A1 post-wave audit selected-game manifest binding mismatch")

    validation = payload.get("validation_holdout")
    if not isinstance(validation, dict):
        raise SystemExit("A1 post-wave audit has no validation_holdout binding")
    try:
        bound_validation_path = Path(str(validation["manifest"])).expanduser().resolve(
            strict=True
        )
    except (KeyError, OSError) as error:
        raise SystemExit(
            f"A1 post-wave audit validation manifest path is invalid: {error}"
        ) from error
    validation_manifest = _load_validation_game_seed_manifest_for_training(
        bound_validation_path,
        validation_fraction=0.05,
        validation_seed=17,
        validation_max_samples=0,
        validation_game_seed_ranges=[],
    )
    if (
        validation_manifest["a1_contract_sha256"]
        != selected_manifest["a1_contract_sha256"]
    ):
        raise SystemExit(
            "A1 validation manifest and selected-game manifest contract hash mismatch"
        )
    if (
        validation_manifest["validation_game_seed_set_sha256"]
        != selected_manifest["validation_game_seed_set_sha256"]
        or int(np.asarray(validation_manifest["game_seeds"]).size)
        != int(selected_manifest["validation_game_count"])
        or not np.array_equal(
            np.asarray(validation_manifest["game_seeds"], dtype=np.int64),
            np.asarray(selected_manifest["validation_game_seeds"], dtype=np.int64),
        )
    ):
        raise SystemExit(
            "A1 validation manifest does not match the validation split in the "
            "selected-game manifest"
        )
    expected_validation = {
        "manifest": validation_manifest["path"],
        "manifest_sha256": validation_manifest["manifest_sha256"],
        "manifest_file_sha256": validation_manifest["file_sha256"],
        "validation_game_seed_count": int(
            np.asarray(validation_manifest["game_seeds"]).size
        ),
        "validation_game_seed_set_sha256": validation_manifest[
            "validation_game_seed_set_sha256"
        ],
    }
    actual_validation = {
        "manifest": bound_validation_path,
        "manifest_sha256": validation.get("manifest_sha256"),
        "manifest_file_sha256": validation.get("manifest_file_sha256"),
        "validation_game_seed_count": validation.get("validation_game_seed_count"),
        "validation_game_seed_set_sha256": validation.get(
            "validation_game_seed_set_sha256"
        ),
    }
    if actual_validation != expected_validation:
        raise SystemExit("A1 post-wave audit validation manifest binding mismatch")
    selected_row_count = payload.get("rows")
    if (
        isinstance(selected_row_count, bool)
        or not isinstance(selected_row_count, int)
        or selected_row_count <= 0
    ):
        raise SystemExit("A1 post-wave audit has invalid selected row count")
    validation_row_count = int(validation_manifest["validation_row_count"])
    if validation_row_count <= 0 or validation_row_count >= selected_row_count:
        raise SystemExit(
            "A1 post-wave audit validation row count is incompatible with total rows"
        )

    source_provenance = payload.get("source_provenance")
    if not isinstance(source_provenance, dict) or not source_provenance:
        raise SystemExit("A1 post-wave audit has invalid source_provenance")
    return {
        "path": audit_path,
        **({} if not is_dual else {"arm_id": selected_manifest["arm_id"], "subset_id": selected_manifest["subset_id"]}),
        "file_sha256": _file_sha256(audit_path),
        "audit_sha256": actual_audit_sha,
        "contract_sha256": contract_sha,
        "shard_inventory_sha256": payload["shard_inventory_sha256"],
        "source_provenance": source_provenance,
        "selected_row_count": int(selected_row_count),
        "training_row_count": int(selected_row_count) - validation_row_count,
        "validation_holdout": {
            "path": validation_manifest["path"],
            "file_sha256": validation_manifest["file_sha256"],
            "manifest_sha256": validation_manifest["manifest_sha256"],
            "a1_contract_sha256": validation_manifest["a1_contract_sha256"],
            "validation_game_seed_count": int(
                np.asarray(validation_manifest["game_seeds"]).size
            ),
            "validation_row_count": validation_manifest["validation_row_count"],
            "validation_game_seed_set_sha256": validation_manifest[
                "validation_game_seed_set_sha256"
            ],
        },
        "harvest_relocation": harvest_provenance,
        "data_shards": data_shards,
        "contract_attestations": contract_attestations,
    }

# The exact column set (and order) load_teacher_data keeps in its local ``keys``
# tuple. Anything not present in a normalised shard is simply skipped, matching
# load_teacher_data's ``if key in shard`` guard.
LOADER_KEYS: tuple[str, ...] = (
    "obs",
    "legal_action_ids",
    "legal_action_context",
    "action_taken",
    "target_policy",
    "prior_policy",
    "target_scores",
    "target_policy_mask",
    "target_scores_mask",
    "target_score_source",
    "target_information_regime",
    "root_value",
    "root_value_mask",
    "game_seed",
    "teacher_name",
    "player",
    "seat",
    "phase",
    "decision_index",
    "action_mask_version",
    "winner",
    "terminated",
    "truncated",
    "final_public_vps",
    "has_final_public_vps",
    "final_actual_vps",
    "has_final_actual_vps",
    "policy_weight_multiplier",
    "value_weight_multiplier",
    "is_forced",
    "used_full_search",
    *AUX_TARGET_KEYS,
    "hex_tokens",
    "hex_vertex_ids",
    "hex_edge_ids",
    "vertex_tokens",
    "edge_tokens",
    "edge_vertex_ids",
    "player_tokens",
    "global_tokens",
    "legal_action_tokens",
    "legal_action_target_ids",
    "event_tokens",
    "event_target_ids",
    "hex_mask",
    "vertex_mask",
    "edge_mask",
    "player_mask",
    "legal_action_mask",
    "event_mask",
)

# Columns padded on the legal-action axis by load_teacher_data._concat_padded,
# and the fill value used there. These are stored ragged (trimmed) on disk.
RAGGED_FILLS: dict[str, float] = {
    "legal_action_ids": -1.0,
    "target_policy": 0.0,
    "prior_policy": 0.0,
    "target_scores": float("nan"),
    "target_policy_mask": 0.0,  # False
    "target_scores_mask": 0.0,  # False
    "legal_action_mask": 0.0,  # False
    "legal_action_context": 0.0,
    "legal_action_tokens": 0.0,
    "legal_action_target_ids": -1.0,
}


def _fill_matches(values: np.ndarray, fill: float) -> np.ndarray:
    """Elementwise "is this the pad fill" test, treating NaN fill specially."""
    if np.isnan(fill):
        return np.isnan(values)
    return values == fill


def _classify(name: str, array: np.ndarray) -> dict:
    """Return the on-disk schema record for a normalised column."""
    if array.dtype.kind == "U":
        return {"kind": "string"}
    if name in RAGGED_FILLS:
        if array.ndim == 2:
            return {
                "kind": "ragged2d",
                "dtype": array.dtype.str,
                "fill": RAGGED_FILLS[name],
            }
        if array.ndim == 3:
            return {
                "kind": "ragged3d",
                "dtype": array.dtype.str,
                "feat": int(array.shape[2]),
                "fill": RAGGED_FILLS[name],
            }
        raise SystemExit(f"ragged column {name} has unexpected ndim {array.ndim}")
    return {
        "kind": "fixed",
        "dtype": array.dtype.str,
        "inner_shape": [int(d) for d in array.shape[1:]],
    }


class _GameSeedRunTracker:
    """Tracks maximal contiguous runs of equal ``game_seed`` values across the
    whole corpus (all shards, in order), flagging a value as duplicated the
    moment a genuinely NEW, non-contiguous run starts with a value that has
    already started a run before.

    ``game_seed`` is one value per GAME, repeated across every decision row of
    that game -- not a per-row identity. Seeds are globally disjoint by
    design, so a seed value starting a SECOND, non-contiguous run anywhere in
    the corpus indicates a collision (the class task #77 nearly missed).

    A naive per-shard unique-value set has a false-positive trap:
    GumbelShardWriter flushes shards purely by ROW COUNT
    (``if len(self.rows) >= self.shard_size: self.flush()``), not by game
    boundary, so a game in progress at a shard's end routinely continues with
    the same game_seed at the very start of the next shard -- one game split
    across two files, not a duplicate. This tracker merges a shard's leading
    run into the previous shard's still-open trailing run when they share a
    value, treating the whole corpus as one contiguous stream of runs.

    Earlier revision of this logic (fixed here) deferred registering an
    open/merged run until it was explicitly "closed" by a later, differing
    value. That meant a run continuation-merged across a shard boundary
    (e.g. shard N-1 ends with seed S, shard N opens with seed S) was never
    added to the seen-set while it stayed the open/pending run -- so if S
    reappeared LATER in that same shard N as a second, non-contiguous run
    (a same-shard duplicate), the reappearance became the new pending run
    directly and bypassed the seen-set check entirely, escaping detection.
    This tracker instead registers a value into the seen-set the instant its
    run *starts* (whether merged-open or freshly closed), so any later run
    of an already-registered value is caught regardless of whether the
    earlier run was ever formally closed.
    """

    def __init__(self) -> None:
        self._seen: set[int] = set()
        self._duplicates: set[int] = set()
        self._current: int | None = None

    def _start_run(self, value: int) -> None:
        if value in self._seen:
            self._duplicates.add(value)
        else:
            self._seen.add(value)
        self._current = value

    def observe_shard(self, seed_column: np.ndarray) -> None:
        seed_col = np.asarray(seed_column).reshape(-1)
        if not seed_col.size:
            return
        run_starts = np.concatenate(([0], np.flatnonzero(np.diff(seed_col) != 0) + 1))
        run_values = seed_col[run_starts]
        for value in run_values:
            value = int(value)
            if value == self._current:
                continue  # merges into the still-open run (shard boundary or not)
            self._start_run(value)

    @property
    def duplicate_count(self) -> int:
        return len(self._duplicates)

    @property
    def has_duplicates(self) -> bool:
        return bool(self._duplicates)


class _SelectedGameSeedRunTracker:
    """Detect a selected seed starting more than one raw-source game run.

    Unlike filtering followed by ``_GameSeedRunTracker``, this observes the
    unfiltered stream so an unselected validation/reserve attempt remains a
    boundary.  A legitimate game may still span adjacent shards when the raw
    trailing and leading seed are equal.
    """

    def __init__(self, selected: set[int]) -> None:
        self._selected = selected
        self._seen: set[int] = set()
        self._duplicates: set[int] = set()
        self._current: int | None = None

    def observe_shard(self, seed_column: np.ndarray) -> None:
        seed_col = np.asarray(seed_column, dtype=np.int64).reshape(-1)
        if not seed_col.size:
            return
        run_starts = np.concatenate(([0], np.flatnonzero(np.diff(seed_col) != 0) + 1))
        for raw_value in seed_col[run_starts]:
            value = int(raw_value)
            if value not in self._selected:
                self._current = None
                continue
            if value == self._current:
                continue
            if value in self._seen:
                self._duplicates.add(value)
            else:
                self._seen.add(value)
            self._current = value

    @property
    def duplicate_count(self) -> int:
        return len(self._duplicates)


def build_memmap_corpus(
    source: Path | str | Sequence[Path | str],
    out_dir: Path,
    *,
    max_shards: int | None = None,
    verify_fill: bool = True,
    progress_every: int = 500,
    abort_on_duplicate_seeds: bool = True,
    full_rows_only: bool = False,
    omit_zero_events: bool = False,
    selected_game_seed_manifest: Path | str | None = None,
    a1_post_wave_audit: Path | str | None = None,
) -> dict:
    """Stream one or more sources' npz shards into a flat memmap corpus.

    ``source`` may be a single teacher-shard root or a sequence of them (e.g.
    tranche-1 combined + tranche-2). Shards are concatenated in source order into
    one corpus; every shard across all sources must share the same column schema
    (enforced per shard), and the global legal width, string categories and row
    offsets span the whole set. Returns the written ``corpus_meta.json`` payload.

    When ``selected_game_seed_manifest`` is supplied, it must be the immutable
    ``a1-selected-training-games-v1`` sidecar emitted by the A1 post-wave audit.
    Every shard is filtered by the exact 12,000 selected complete-game seeds
    before row sizing, statistics, duplicate tracking, or writes.  Both split
    labels remain in the memmap: ``train_bc`` consumes the bound validation
    manifest to exclude validation games before optimizer updates while still
    evaluating every held-out row. Conversion fails unless every selected game
    is present.
    """
    sources = [source] if isinstance(source, (str, Path)) else list(source)
    source_attestations = _load_direct_a1_source_attestations(sources)
    selected_manifest = (
        None
        if selected_game_seed_manifest is None
        else _load_a1_selected_game_manifest(Path(selected_game_seed_manifest))
    )
    if source_attestations and (
        selected_manifest is None or a1_post_wave_audit is None
    ):
        raise SystemExit(
            "A1 source attestation detected: both --selected-game-seed-manifest "
            "and --a1-post-wave-audit are mandatory; audited A1 shards cannot be "
            "converted through the generic memmap path"
        )
    if (selected_manifest is None) != (a1_post_wave_audit is None):
        raise SystemExit(
            "--selected-game-seed-manifest and --a1-post-wave-audit must be "
            "provided together; neither artifact authorizes ingest by itself"
        )
    if selected_manifest is not None and full_rows_only:
        raise SystemExit(
            "--full-rows-only is forbidden for audited A1 ingest: all rows from "
            "the exact 12,000 selected complete games must remain available to the "
            "one-dose learner and immutable validation holdout"
        )
    post_wave_audit = (
        None
        if a1_post_wave_audit is None
        else _load_a1_post_wave_audit(Path(a1_post_wave_audit), selected_manifest)
    )
    if source_attestations:
        expected_contract = selected_manifest["a1_contract_sha256"]
        mismatched = [
            str(record["path"])
            for record in source_attestations
            if record["contract_sha256"] != expected_contract
        ]
        if mismatched:
            raise SystemExit(
                "A1 source attestations do not all bind the selected/audited "
                f"contract {expected_contract}: {mismatched}"
            )
    selected_game_seeds = (
        None
        if selected_manifest is None
        else np.asarray(selected_manifest["selected_game_seeds"], dtype=np.int64)
    )
    expected_selected_seed_set = (
        set()
        if selected_game_seeds is None
        else set(map(int, selected_game_seeds.tolist()))
    )
    observed_selected_seed_set: set[int] = set()
    selected_source_tracker = (
        None
        if selected_manifest is None
        else _SelectedGameSeedRunTracker(expected_selected_seed_set)
    )

    files: list[Path] = []
    source_first_files: list[Path] = []
    for src in sources:
        src_files = _teacher_shard_files(Path(src))
        if not src_files:
            raise SystemExit(f"no teacher shards found in {src}")
        source_first_files.append(src_files[0])
        files.extend(src_files)
    if max_shards is not None:
        files = files[:max_shards]
    if post_wave_audit is not None:
        actual_by_path: dict[Path, Path] = {}
        for file in files:
            canonical = file.expanduser().resolve(strict=True)
            if canonical in actual_by_path:
                raise SystemExit(
                    f"input sources repeat canonical data shard path {canonical}"
                )
            actual_by_path[canonical] = file
        audited_paths = [
            Path(record["path"]) for record in post_wave_audit["data_shards"]
        ]
        audited_path_set = set(audited_paths)
        actual_path_set = set(actual_by_path)
        if actual_path_set != audited_path_set:
            raise SystemExit(
                "input data-shard inventory differs from the passing A1 audit: "
                f"missing={len(audited_path_set - actual_path_set)} "
                f"unexpected={len(actual_path_set - audited_path_set)}"
            )
        # The audit's inventory order is authoritative.  This preserves a
        # game that legitimately spans adjacent shards while preventing an
        # alternate same-seed file from being substituted or reordered.
        files = [actual_by_path[path] for path in audited_paths]
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    # Replay windows commonly combine homogeneous per-round sources. If any
    # source carries CAT-100 targets, normalize legacy sources with per-head
    # ignore fills so the memmap keeps a single aligned schema. Inspecting one
    # shard per source avoids decompressing every .npz.zst twice.
    include_aux_targets = False
    for first_file in source_first_files:
        raw = _load_npz(first_file)
        try:
            include_aux_targets = include_aux_targets or any(
                key in raw for key in AUX_TARGET_KEYS
            )
        finally:
            close = getattr(raw, "close", None)
            if callable(close):
                close()
    first = _normalize_teacher_shard(
        _load_npz(files[0]),
        files[0],
        include_aux_defaults=include_aux_targets,
    )
    if selected_game_seeds is not None:
        if "game_seed" not in first:
            raise SystemExit(
                f"{files[0]}: --selected-game-seed-manifest requires a game_seed column"
            )
        first_keep = np.isin(
            np.asarray(first["game_seed"], dtype=np.int64),
            selected_game_seeds,
        )
        first = {
            name: np.asarray(value)[first_keep] for name, value in first.items()
        }
    columns = [key for key in LOADER_KEYS if key in first]
    schemas = {name: _classify(name, first[name]) for name in columns}
    implicit_zero_columns: list[str] = []
    if omit_zero_events:
        missing_event_columns = IMPLICIT_ZERO_EVENT_COLUMNS - set(columns)
        if missing_event_columns:
            raise SystemExit(
                "--omit-zero-events requires both event_tokens and event_mask in "
                f"the normalized source schema; missing={sorted(missing_event_columns)}"
            )
        for name in sorted(IMPLICIT_ZERO_EVENT_COLUMNS & set(columns)):
            array = np.asarray(first[name])
            schemas[name] = {
                "kind": "implicit_constant",
                "dtype": array.dtype.str,
                "inner_shape": [int(d) for d in array.shape[1:]],
                "fill": 0,
            }
            implicit_zero_columns.append(name)
    column_set = set(columns)

    handles = {
        name: open(out_dir / f"{name}.dat", "wb")
        for name in columns
        if schemas[name]["kind"] not in {"string", "implicit_constant"}
    }
    code_handles = {name: open(out_dir / f"{name}.codes.dat", "wb") for name in columns if schemas[name]["kind"] == "string"}
    # Global string factorisation: category -> code, stable in first-seen order.
    categories: dict[str, dict[str, int]] = {name: {} for name in code_handles}
    category_lists: dict[str, list[str]] = {name: [] for name in code_handles}

    row_lengths: list[np.ndarray] = []
    row_count = 0
    flat_count = 0
    legal_width = 0
    stats = {
        "max_legal_action_id": -1,
        "action_taken_min": None,
        "action_taken_max": None,
        "has_duplicate_legal_rows": False,
        "duplicate_game_seed_count": 0,
        "has_duplicate_game_seeds": False,
    }
    # See _GameSeedRunTracker's docstring for the duplicate-detection contract.
    _seed_tracker = _GameSeedRunTracker()

    dropped_fast_rows = 0
    for shard_index, file in enumerate(files):
        raw = _load_npz(file)
        norm = _normalize_teacher_shard(
            raw,
            file,
            include_aux_defaults=include_aux_targets,
        )
        # This check intentionally precedes --full-rows-only filtering.  The
        # opt-in contract is corpus-source-wide: a live event in even a row
        # that a later filter would drop must fail closed rather than allowing
        # an accidental mixed-history source to masquerade as event-free.
        for name in implicit_zero_columns:
            if bool(np.any(norm[name])):
                for handle in (*handles.values(), *code_handles.values()):
                    handle.close()
                raise SystemExit(
                    f"{file}: --omit-zero-events requires every source row's {name} "
                    "to be exactly zero; found live/non-zero event data. Refusing "
                    "lossy conversion. Re-run without --omit-zero-events."
                )
        selected_row_mask: np.ndarray | None = None
        if selected_game_seeds is not None:
            if "game_seed" not in norm:
                raise SystemExit(
                    f"{file}: --selected-game-seed-manifest requires a game_seed column"
                )
            raw_game_seeds = np.asarray(norm["game_seed"], dtype=np.int64)
            selected_source_tracker.observe_shard(raw_game_seeds)
            selected_row_mask = np.isin(raw_game_seeds, selected_game_seeds)
            # This is deliberately the first row-level transform.  Reserve,
            # incomplete, and truncated attempts use unselected game seeds and
            # therefore cannot affect output sizing, statistics, or bytes.
            norm = {
                name: np.asarray(value)[selected_row_mask]
                for name, value in norm.items()
            }
            for status_name, expected in (("terminated", True), ("truncated", False)):
                if status_name not in norm:
                    raise SystemExit(
                        f"{file}: --selected-game-seed-manifest requires a "
                        f"{status_name} column"
                    )
                statuses = np.asarray(norm[status_name], dtype=bool)
                if statuses.size and np.any(statuses != expected):
                    raise SystemExit(
                        f"{file}: selected rows include a non-complete game "
                        f"({status_name} must be {expected})"
                    )
        if full_rows_only:
            # Keep only FULL-search rows (drop fast rows). used_full_search is the
            # ground-truth per-row full/fast marker written by the generator; it is
            # NOT part of LOADER_KEYS (so it never lands in the memmap), so read it
            # from the raw shard here. Forced rows report used_full_search=True (they
            # pay a full enumeration and carry value signal) and are KEPT -- only
            # fast-search rows are dropped. policy_weight_multiplier already zeroes
            # fast+forced rows out of POLICY loss at train time; this filter is the
            # physical-drop variant for building a fast-free corpus (e.g. the
            # pure-teacher ablation arm).
            if "used_full_search" not in raw:
                raise SystemExit(
                    f"{file}: --full-rows-only requires a 'used_full_search' column, "
                    "but this shard has none (pre-marker generation?). Rebuild the "
                    "shards with the current generator or drop --full-rows-only."
                )
            keep = np.asarray(raw["used_full_search"]).astype(bool)
            if selected_row_mask is not None:
                if keep.shape[0] != selected_row_mask.shape[0]:
                    raise SystemExit(
                        f"{file}: used_full_search length {keep.shape[0]} != raw row "
                        f"count {selected_row_mask.shape[0]}"
                    )
                keep = keep[selected_row_mask]
            if keep.shape[0] != int(np.asarray(norm["action_taken"]).shape[0]):
                raise SystemExit(
                    f"{file}: used_full_search length {keep.shape[0]} != row count "
                    f"{int(np.asarray(norm['action_taken']).shape[0])}"
                )
            dropped_fast_rows += int((~keep).sum())
            if not keep.all():
                norm = {name: np.asarray(value)[keep] for name, value in norm.items()}
        present = {key for key in LOADER_KEYS if key in norm}
        if present != column_set:
            raise SystemExit(
                f"{file} column set differs from first shard; refusing to mix schemas. "
                f"missing={sorted(column_set - present)} extra={sorted(present - column_set)}"
            )
        # Re-validate every column's dtype and inner shape against the schema
        # recorded from shard 0. The raw bytes are appended with tofile(), so a
        # shard whose dtype or feature width drifted would not crash here -- it
        # would silently misalign EVERY subsequent row when the flat file is
        # reinterpreted by np.memmap at load time. Fail loudly instead.
        for name in columns:
            schema = schemas[name]
            array = norm[name]
            kind = schema["kind"]
            if kind == "string":
                continue
            if array.dtype.str != schema["dtype"]:
                raise SystemExit(
                    f"{file}: column {name!r} dtype {array.dtype.str} != first shard's "
                    f"{schema['dtype']}; mixed dtypes would corrupt the flat memmap."
                )
            if kind in {"fixed", "implicit_constant"}:
                inner = [int(d) for d in array.shape[1:]]
                if inner != list(schema["inner_shape"]):
                    raise SystemExit(
                        f"{file}: column {name!r} inner shape {inner} != first shard's "
                        f"{list(schema['inner_shape'])}; mixed widths would corrupt the flat memmap."
                    )
            elif kind == "ragged3d":
                if int(array.shape[2]) != int(schema["feat"]):
                    raise SystemExit(
                        f"{file}: column {name!r} feature width {int(array.shape[2])} != "
                        f"first shard's {int(schema['feat'])}; mixed widths would corrupt the flat memmap."
                    )
            elif kind == "ragged2d" and array.ndim != 2:
                raise SystemExit(f"{file}: column {name!r} ndim {array.ndim} != 2")
        legal_ids = norm["legal_action_ids"]
        width = int(legal_ids.shape[1])
        legal_width = max(legal_width, width)
        counts = np.sum(legal_ids >= 0, axis=1).astype(np.int64)
        n = int(legal_ids.shape[0])
        prefix_mask = np.arange(width)[None, :] < counts[:, None]

        # Trimming is lossless only if the valid legal entries are a contiguous
        # prefix (guaranteed by legal_action_mask == legal_action_ids>=0) and
        # everything past the count is exactly the pad fill. Verify per shard so
        # a schema drift aborts the conversion instead of silently dropping data.
        if verify_fill:
            if "legal_action_mask" in norm and not np.array_equal(
                norm["legal_action_mask"], legal_ids >= 0
            ):
                raise SystemExit(f"{file}: legal_action_mask != (legal_action_ids>=0)")
            tail_mask = ~prefix_mask
            for name in columns:
                if name not in RAGGED_FILLS:
                    continue
                tail = norm[name][tail_mask]
                if tail.size and not np.all(_fill_matches(tail, RAGGED_FILLS[name])):
                    raise SystemExit(
                        f"{file}: column {name!r} has non-fill values beyond the legal "
                        "count; per-row trimming would lose data. Regenerate the shard "
                        "or extend build_memmap_corpus to store it at full width."
                    )

        # Cheap corpus-wide validation stats (mirror validate_teacher_data_schema)
        valid_legal = legal_ids[legal_ids >= 0]
        if valid_legal.size:
            stats["max_legal_action_id"] = max(stats["max_legal_action_id"], int(valid_legal.max()))
        actions = norm["action_taken"]
        if actions.size:
            amin, amax = int(actions.min()), int(actions.max())
            stats["action_taken_min"] = amin if stats["action_taken_min"] is None else min(stats["action_taken_min"], amin)
            stats["action_taken_max"] = amax if stats["action_taken_max"] is None else max(stats["action_taken_max"], amax)
        if not stats["has_duplicate_legal_rows"]:
            # A duplicate legal id within a row shows up as adjacent equal values
            # (both >= 0) once each row is sorted; -1 pads sort to the front.
            row_sorted = np.sort(legal_ids, axis=1)
            adjacent_equal = (row_sorted[:, 1:] == row_sorted[:, :-1]) & (row_sorted[:, 1:] >= 0)
            if bool(np.any(adjacent_equal)):
                stats["has_duplicate_legal_rows"] = True
        if "game_seed" in norm:
            _seed_tracker.observe_shard(norm["game_seed"])
            if selected_game_seeds is not None:
                observed_selected_seed_set.update(
                    map(int, np.asarray(norm["game_seed"], dtype=np.int64).tolist())
                )

        for name in columns:
            schema = schemas[name]
            array = norm[name]
            kind = schema["kind"]
            if kind == "string":
                catmap = categories[name]
                catlist = category_lists[name]
                uniq, inverse = np.unique(array, return_inverse=True)
                mapped = np.empty(uniq.shape[0], dtype=np.int32)
                for u_index, value in enumerate(uniq):
                    text = str(value)
                    code = catmap.get(text)
                    if code is None:
                        code = len(catlist)
                        catmap[text] = code
                        catlist.append(text)
                    mapped[u_index] = code
                codes = mapped[inverse].astype(np.int32, copy=False)
                code_handles[name].write(np.ascontiguousarray(codes).tobytes())
            elif kind == "fixed":
                np.ascontiguousarray(array).tofile(handles[name])
            elif kind == "implicit_constant":
                continue
            else:  # ragged2d / ragged3d
                flat = array[prefix_mask]  # row-major prefix concat -> (sum counts, [feat])
                np.ascontiguousarray(flat).tofile(handles[name])

        row_lengths.append(counts)
        row_count += n
        flat_count += int(counts.sum())
        if progress_every and (shard_index + 1) % progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "progress": "memmap_convert",
                        "shards_done": shard_index + 1,
                        "shards_total": len(files),
                        "rows": row_count,
                        "elapsed_s": round(elapsed, 1),
                        "shards_per_s": round((shard_index + 1) / max(elapsed, 1e-9), 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    for handle in handles.values():
        handle.close()
    for handle in code_handles.values():
        handle.close()

    for record in source_attestations:
        actual_sha = _file_sha256(Path(record["path"]))
        if actual_sha != record["file_sha256"]:
            raise SystemExit(
                "A1 source attestation changed while building the corpus: "
                f"{record['path']}"
            )

    if post_wave_audit is not None:
        # Verify bytes after conversion as the acceptance boundary. This both
        # avoids hashing every large shard twice and catches ordinary mutation
        # while a long conversion is reading the audited inventory.
        for record in post_wave_audit["data_shards"]:
            actual_sha = _file_sha256(Path(record["path"]))
            if actual_sha != record["sha256"]:
                raise SystemExit(
                    "input data shard changed from the passing A1 audit while "
                    f"building the corpus: {record['path']} "
                    f"declared={record['sha256']} actual={actual_sha}"
                )
        if int(row_count) != int(post_wave_audit["selected_row_count"]):
            raise SystemExit(
                "A1 memmap row count differs from the passing audit's exact selected "
                f"exposure: corpus={row_count} "
                f"audit={post_wave_audit['selected_row_count']}"
            )

    if selected_manifest is not None:
        if selected_source_tracker.duplicate_count:
            raise SystemExit(
                "selected game_seed starts more than one non-contiguous "
                f"raw-source run for {selected_source_tracker.duplicate_count} seed(s)"
            )
        missing = expected_selected_seed_set - observed_selected_seed_set
        unexpected = observed_selected_seed_set - expected_selected_seed_set
        if missing or unexpected:
            raise SystemExit(
                "selected-game seed set in converted rows differs from the immutable "
                f"manifest: missing={len(missing)} unexpected={len(unexpected)}"
            )

    stats["duplicate_game_seed_count"] = _seed_tracker.duplicate_count
    stats["has_duplicate_game_seeds"] = _seed_tracker.has_duplicates
    if stats["has_duplicate_game_seeds"]:
        message = (
            f"{stats['duplicate_game_seed_count']} game_seed value(s) recur "
            "as a SEPARATE, non-contiguous game elsewhere in this corpus -- games are "
            "supposed to have globally disjoint seeds, so this indicates duplicated "
            "games (the seed-collision class from task #77) silently doubling their "
            "weight in training."
        )
        if abort_on_duplicate_seeds:
            raise SystemExit(
                f"ABORTING: {message} Investigate the source shards (or re-run with "
                "--no-abort-on-duplicate-seeds to only warn and proceed at your own risk)."
            )
        print(
            f"WARNING: {message} NOT aborting the conversion (--no-abort-on-duplicate-seeds "
            "was set); the operator should investigate corpus_meta.json's "
            "stats.duplicate_game_seed_count before training on this corpus.",
            file=sys.stderr,
        )

    lengths = np.concatenate(row_lengths) if row_lengths else np.zeros(0, dtype=np.int64)
    offsets = np.empty(row_count + 1, dtype=np.int64)
    offsets[0] = 0
    if lengths.size:
        np.cumsum(lengths, out=offsets[1:])
    offsets.tofile(out_dir / "row_offsets.dat")

    for name in code_handles:
        schemas[name] = {"kind": "string", "categories": category_lists[name]}

    payload_inventory = _memmap_payload_inventory(out_dir, schemas)

    meta = {
        "schema": MEMMAP_CORPUS_IMPLICIT_SCHEMA if implicit_zero_columns else MEMMAP_CORPUS_SCHEMA,
        "payload_inventory_schema": MEMMAP_PAYLOAD_INVENTORY_SCHEMA,
        "payload_inventory": payload_inventory,
        "payload_inventory_sha256": _value_sha256(payload_inventory),
        "row_count": int(row_count),
        "flat_count": int(flat_count),
        "legal_width": int(legal_width),
        "source": str(sources[0]),
        "sources": [str(src) for src in sources],
        "shard_count": len(files),
        "columns": schemas,
        "game_seed_present": "game_seed" in column_set,
        # --full-rows-only provenance: whether fast rows were physically dropped,
        # and how many. False + 0 for a normal (pooled) build.
        "full_rows_only": bool(full_rows_only),
        "dropped_fast_rows": int(dropped_fast_rows),
        # Whether the lossless-trim guarantee (ragged tails are exactly pad
        # fill) was actually VERIFIED for this corpus. A corpus built with
        # --no-verify-fill is otherwise indistinguishable from a verified one.
        "verify_fill": bool(verify_fill),
        "implicit_zero_columns": implicit_zero_columns,
        "implicit_zero_bytes_saved_per_row": int(
            sum(
                np.prod(schemas[name]["inner_shape"], dtype=np.int64)
                * np.dtype(schemas[name]["dtype"]).itemsize
                for name in implicit_zero_columns
            )
        ),
        "stats": stats,
        "conversion_seconds": round(time.perf_counter() - started, 2),
    }
    if selected_manifest is not None:
        meta["selected_game_seed_manifest"] = {
            "path": str(selected_manifest["path"]),
            "file_sha256": selected_manifest["file_sha256"],
            "a1_contract_sha256": selected_manifest["a1_contract_sha256"],
            "selected_game_count": selected_manifest["selected_game_count"],
            "selected_game_seed_set_sha256": selected_manifest[
                "selected_game_seed_set_sha256"
            ],
            "training_game_count": selected_manifest["training_game_count"],
            "training_game_seed_set_sha256": selected_manifest[
                "training_game_seed_set_sha256"
            ],
            "validation_game_count": selected_manifest["validation_game_count"],
            "validation_game_seed_set_sha256": selected_manifest[
                "validation_game_seed_set_sha256"
            ],
            "records_sha256": selected_manifest["records_sha256"],
        }
        meta["a1_post_wave_audit"] = {
            "path": str(post_wave_audit["path"]),
            "file_sha256": post_wave_audit["file_sha256"],
            "audit_sha256": post_wave_audit["audit_sha256"],
            "contract_sha256": post_wave_audit["contract_sha256"],
            "shard_inventory_sha256": post_wave_audit[
                "shard_inventory_sha256"
            ],
            "source_provenance": post_wave_audit["source_provenance"],
            **(
                {}
                if "arm_id" not in post_wave_audit
                else {
                    "arm_id": post_wave_audit["arm_id"],
                    "subset_id": post_wave_audit["subset_id"],
                }
            ),
            "selected_row_count": post_wave_audit["selected_row_count"],
            "training_row_count": post_wave_audit["training_row_count"],
            "validation_holdout": {
                key: (
                    str(value) if isinstance(value, Path) else value
                )
                for key, value in post_wave_audit["validation_holdout"].items()
            },
            **(
                {}
                if post_wave_audit["harvest_relocation"] is None
                else {
                    "harvest_relocation": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in post_wave_audit[
                            "harvest_relocation"
                        ].items()
                        if key != "by_local"
                    }
                }
            ),
        }
    (out_dir / "corpus_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"progress": "memmap_convert_done", **{k: meta[k] for k in ("row_count", "flat_count", "legal_width", "shard_count", "conversion_seconds")}}, sort_keys=True), flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source",
        type=Path,
        nargs="+",
        help=(
            "One or more teacher shard roots (each a dir with manifest.json); "
            "shards are concatenated in the given order, e.g. "
            "--source runs/raw_selfplay_gen1_combined runs/raw_selfplay_gen2_combined"
        ),
    )
    source_group.add_argument(
        "--source-list",
        type=Path,
        help=(
            "UTF-8 file containing one teacher shard root per line. This avoids "
            "argv limits for harvested corpora with thousands of worker leaves."
        ),
    )
    parser.add_argument("--out", required=True, type=Path, help="output corpus directory")
    parser.add_argument("--max-shards", type=int, default=None, help="convert only the first N shards (slice/estimate)")
    parser.add_argument(
        "--full-rows-only",
        action="store_true",
        help=(
            "Physically DROP fast-search rows (keep rows with used_full_search=True, "
            "including forced-full rows). Builds a fast-free corpus for the "
            "pure-teacher ablation arm. Normal (pooled) builds omit this: fast rows "
            "are kept and already carry policy_weight_multiplier=0, so they train "
            "value only and are excluded from policy loss at train time. Requires the "
            "shards to carry a 'used_full_search' column."
        ),
    )
    parser.add_argument("--no-verify-fill", action="store_true", help="skip the per-shard lossless-trim assertion (faster)")
    parser.add_argument(
        "--omit-zero-events",
        action="store_true",
        help=(
            "Write a memmap_corpus_v2 corpus that omits event_tokens.dat and "
            "event_mask.dat only after proving both columns are exactly zero in "
            "every source row. Training reconstructs exact zero arrays per batch. "
            "Conversion aborts on any live event. Off by default."
        ),
    )
    parser.add_argument(
        "--selected-game-seed-manifest",
        type=Path,
        default=None,
        help=(
            "Immutable a1-selected-training-games-v1 sidecar from the A1 "
            "post-wave audit. When set, only rows belonging to its exact 12,000 "
            "selected complete game seeds enter the corpus. Validation rows remain "
            "available for train_bc's exact game-level holdout; missing/tampered "
            "selection evidence aborts conversion."
        ),
    )
    parser.add_argument(
        "--a1-post-wave-audit",
        type=Path,
        default=None,
        help=(
            "Passing a1-post-wave-audit-v2/v3 report that binds the selected-game "
            "manifest and exact input shard inventory/hashes. Required together "
            "with --selected-game-seed-manifest."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--abort-on-duplicate-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hard-exit if a game_seed value starts a second, non-contiguous run "
        "anywhere in the corpus (the seed-collision class from task #77). Default "
        "on; pass --no-abort-on-duplicate-seeds to only warn (via stats."
        "duplicate_game_seed_count in corpus_meta.json) and proceed at your own risk.",
    )
    args = parser.parse_args()
    sources = args.source
    if args.source_list is not None:
        sources = [
            Path(line.strip())
            for line in args.source_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not sources:
            parser.error(f"--source-list is empty: {args.source_list}")
    build_memmap_corpus(
        sources,
        args.out,
        max_shards=args.max_shards,
        verify_fill=not args.no_verify_fill,
        progress_every=args.progress_every,
        abort_on_duplicate_seeds=args.abort_on_duplicate_seeds,
        full_rows_only=args.full_rows_only,
        omit_zero_events=args.omit_zero_events,
        selected_game_seed_manifest=args.selected_game_seed_manifest,
        a1_post_wave_audit=args.a1_post_wave_audit,
    )


if __name__ == "__main__":
    main()
