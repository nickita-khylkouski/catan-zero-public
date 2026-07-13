#!/usr/bin/env python3
"""Build the exact first-wave fresh/replay training composite.

The post-wave audit authorizes whole games, not whole shard files.  This tool
materializes three source-pure fresh components by filtering every audited NPZ
on the signed ``(job_id, category, game_seed)`` selection before memmap
expansion.  It then attaches an already authenticated historical-replay
component and emits the promotion-eligible .64/.12/.04/.20 descriptor consumed
by ``train_bc``.

This is intentionally a builder only.  It never launches generation or a
learner and it refuses an existing/non-empty output root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from catan_zero.rl.flywheel.composite_contract import (  # noqa: E402
    FRESH_SOURCE_GAME_RATIOS,
    HISTORICAL_REPLAY_CATEGORY,
    build_sampling_receipt,
    canonical_sha256,
    measure_memmap_component,
)
from tools import a1_pre_wave_contract as contract  # noqa: E402
from tools import build_memmap_corpus as memmap_builder  # noqa: E402
from tools import train_bc  # noqa: E402


HISTORICAL_COMPONENT_REF_SCHEMA = "a1-historical-replay-component-ref-v1"
BUILD_RECEIPT_SCHEMA = "a1-post-wave-composite-build-v1"
EFFECTIVE_COMPONENT_RATIOS = {
    "current_producer": 0.64,
    "recent_history": 0.12,
    "hard_negative": 0.04,
    HISTORICAL_REPLAY_CATEGORY: 0.20,
}
LEARNER_RECIPE_OVERRIDES: dict[str, object] = {
    "forced_action_weight": 0.0,
    "forced_row_value_weight": 1.0,
    "loser_sample_weight": 1.0,
    "per_game_policy_weight": True,
    "per_game_policy_weight_mode": "equal",
    "per_game_value_weight": False,
    "per_game_value_weight_mode": "equal",
    "policy_kl_anchor_direction": "forward",
    "policy_kl_anchor_weight": 0.0,
    "policy_loss_weight": 1.0,
    "q_loss_weight": 0.0,
    "soft_target_source": "policy",
    "soft_target_temperature": 0.7,
    "soft_target_weight": 0.9,
    "truncated_vp_margin_value_weight": 0.25,
    "value_target_lambda": 1.0,
}


class CompositeBuildError(RuntimeError):
    pass


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _digest(value: object) -> str:
    return f"sha256:{hashlib.sha256(_canonical_bytes(value)).hexdigest()}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompositeBuildError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise CompositeBuildError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_output_root(path: Path) -> Path:
    root = path.expanduser().absolute()
    if root.exists():
        if root.is_symlink() or not root.is_dir() or any(root.iterdir()):
            raise CompositeBuildError(
                f"output root must be absent or an empty real directory: {root}"
            )
    else:
        root.mkdir(parents=True)
    if root.resolve(strict=True) != root:
        raise CompositeBuildError(f"output root is not canonical: {root}")
    return root


def _validated_wave_inputs(
    lock_path: Path,
    selected_path: Path,
    audit_path: Path,
    *,
    verify_lock_fn: Callable[..., dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        lock = verify_lock_fn(lock_path, require_all_job_claims=True)
        selected = memmap_builder._load_a1_selected_game_manifest(selected_path)  # noqa: SLF001
        audit = memmap_builder._load_a1_post_wave_audit(  # noqa: SLF001
            audit_path, selected
        )
    except (OSError, SystemExit, contract.ContractError) as error:
        raise CompositeBuildError(f"wave input verification failed: {error}") from error
    if selected["a1_contract_sha256"] != lock["contract_sha256"]:
        raise CompositeBuildError("selected-game manifest binds a different lock")
    if audit["contract_sha256"] != lock["contract_sha256"]:
        raise CompositeBuildError("post-wave audit binds a different lock")
    raw_selected = _load_json(Path(selected["path"]))
    if raw_selected.get("records_sha256") != selected["records_sha256"]:
        raise CompositeBuildError("selected-game record digest drift")
    return lock, selected, audit, raw_selected


def _selection_by_job(
    lock: Mapping[str, Any],
    raw_selected: Mapping[str, Any],
    *,
    expected_games: Mapping[str, int],
) -> tuple[dict[str, set[int]], dict[int, tuple[str, str]], list[dict[str, Any]]]:
    jobs = {str(job["job_id"]): job for job in lock["fleet"]["jobs"]}
    producer = contract._producer(dict(lock))  # noqa: SLF001
    selections: dict[str, set[int]] = defaultdict(set)
    owners: dict[int, tuple[str, str]] = {}
    records = raw_selected.get("records")
    if not isinstance(records, list):
        raise CompositeBuildError("selected-game manifest has no record list")
    normalized: list[dict[str, Any]] = []
    for record in records:
        job_id = str(record.get("job_id", ""))
        category = str(record.get("category", ""))
        seed = record.get("game_seed")
        job = jobs.get(job_id)
        if (
            job is None
            or category != job.get("category")
            or record.get("worker_id") != job.get("worker_id")
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not int(job["base_seed"]) <= seed < int(job["seed_end"])
            or record.get("producer_checkpoint_sha256") != producer["sha256"]
            or record.get("opponent_checkpoint_sha256")
            != contract._category_opponent_sha256(dict(lock), category)  # noqa: SLF001
        ):
            raise CompositeBuildError(
                f"selected game does not bind its sealed job/category: {record!r}"
            )
        if seed in owners or seed in selections[job_id]:
            raise CompositeBuildError(f"selected game seed is duplicated: {seed}")
        owners[seed] = (job_id, category)
        selections[job_id].add(seed)
        normalized.append(dict(record))
    counts = Counter(record["category"] for record in normalized)
    if dict(counts) != dict(expected_games):
        raise CompositeBuildError(
            f"selected category quotas differ: actual={dict(counts)} "
            f"expected={dict(expected_games)}"
        )
    return dict(selections), owners, normalized


def _filter_wave_shards(
    *,
    lock: dict[str, Any],
    selected: dict[str, Any],
    audit: dict[str, Any],
    raw_selected: dict[str, Any],
    output_root: Path,
    expected_games: Mapping[str, int],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    selected_by_job, owner_by_seed, _records = _selection_by_job(
        lock, raw_selected, expected_games=expected_games
    )
    jobs = {str(job["job_id"]): job for job in lock["fleet"]["jobs"]}
    checkpoint_by_id = {str(record["id"]): record for record in lock["checkpoints"]}
    category_specs = {
        str(record["name"]): record for record in lock["source_categories"]
    }
    selfplay_colors = tuple(contract._expected_selfplay_config(lock)["colors"])  # noqa: SLF001
    producer = contract._producer(lock)  # noqa: SLF001
    producer_path = Path(str(producer["path"])).expanduser().resolve(strict=True)
    if _file_sha256(producer_path) != producer["sha256"]:
        raise CompositeBuildError("current producer checkpoint bytes drifted")

    audited_by_job: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in audit["data_shards"]:
        job_id = str(record.get("job_id", ""))
        category = str(record.get("category", ""))
        if job_id not in jobs or category != jobs[job_id].get("category"):
            raise CompositeBuildError("audited shard has unknown job/category")
        audited_by_job[job_id].append(record)
    if set(audited_by_job) != set(selected_by_job):
        raise CompositeBuildError(
            "audited shard jobs differ from selected-game jobs: "
            f"missing={sorted(set(selected_by_job) - set(audited_by_job))} "
            f"unexpected={sorted(set(audited_by_job) - set(selected_by_job))}"
        )

    raw_audit = _load_json(Path(str(audit["path"])))
    generation_manifest_by_job: dict[str, dict[str, Any]] = {}
    public_award_by_category: dict[str, dict[str, Any]] = {}
    for record in raw_audit.get("shards", []):
        if not isinstance(record, dict) or record.get("kind") != "generation_manifest":
            continue
        job_id = str(record.get("job_id", ""))
        category = str(record.get("category", ""))
        if job_id not in jobs or category != jobs[job_id].get("category"):
            raise CompositeBuildError(
                "audited generation manifest has unknown job/category"
            )
        manifest_path = Path(str(record.get("path", ""))).resolve(strict=True)
        if _file_sha256(manifest_path) != record.get("sha256"):
            raise CompositeBuildError(
                f"audited generation manifest bytes drifted: {manifest_path}"
            )
        manifest = _load_json(manifest_path)
        public_award = manifest.get("public_award_feature_provenance")
        if not isinstance(public_award, dict):
            raise CompositeBuildError(
                f"audited generation manifest lacks public-award provenance: {manifest_path}"
            )
        prior = public_award_by_category.setdefault(category, dict(public_award))
        if prior != public_award:
            raise CompositeBuildError(
                f"category {category} has multiple public-award feature contracts"
            )
        if job_id in generation_manifest_by_job:
            raise CompositeBuildError(
                f"audit repeats generation manifest for job {job_id}"
            )
        generation_manifest_by_job[job_id] = {
            "path": str(manifest_path),
            "sha256": record["sha256"],
        }
    if set(generation_manifest_by_job) != set(selected_by_job):
        raise CompositeBuildError(
            "audited generation manifests do not cover every selected job"
        )

    filtered_records: dict[str, list[dict[str, Any]]] = {
        category: [] for category in expected_games
    }
    source_bindings: list[dict[str, Any]] = []
    observed_by_job: dict[str, set[int]] = defaultdict(set)
    order_by_category: Counter[str] = Counter()
    for job_id in [str(job["job_id"]) for job in lock["fleet"]["jobs"]]:
        if job_id not in selected_by_job:
            continue
        job = jobs[job_id]
        category = str(job["category"])
        job_selected = selected_by_job[job_id]
        for source_record in audited_by_job[job_id]:
            source = Path(str(source_record["path"])).resolve(strict=True)
            before_sha = _file_sha256(source)
            if before_sha != source_record["sha256"]:
                raise CompositeBuildError(
                    f"audited source shard bytes drifted: {source}"
                )
            try:
                with np.load(source, allow_pickle=False) as payload:
                    if "game_seed" not in payload.files:
                        raise CompositeBuildError(
                            f"source shard lacks game_seed: {source}"
                        )
                    seeds = np.asarray(payload["game_seed"], dtype=np.int64)
                    if seeds.ndim != 1:
                        raise CompositeBuildError(
                            f"game_seed is not one-dimensional: {source}"
                        )
                    selected_mask = np.isin(
                        seeds, np.asarray(sorted(job_selected), dtype=np.int64)
                    )
                    for seed in set(map(int, seeds.tolist())).intersection(
                        owner_by_seed
                    ):
                        if owner_by_seed[seed] != (job_id, category):
                            raise CompositeBuildError(
                                "selected seed appears in the wrong audited job/category: "
                                f"seed={seed} source={job_id}/{category} "
                                f"owner={owner_by_seed[seed]}"
                            )
                    if not np.any(selected_mask):
                        continue
                    for status, expected in (
                        ("terminated", True),
                        ("truncated", False),
                    ):
                        if status not in payload.files or np.any(
                            np.asarray(payload[status], dtype=bool)[selected_mask]
                            != expected
                        ):
                            raise CompositeBuildError(
                                f"selected {job_id} rows are not complete: {status}"
                            )
                    if "policy_weight_multiplier" not in payload.files:
                        raise CompositeBuildError(
                            f"selected source lacks policy_weight_multiplier: {source}"
                        )
                    policy_mass = np.asarray(
                        payload["policy_weight_multiplier"], dtype=np.float64
                    )[selected_mask]
                    if not np.all(np.isfinite(policy_mass)) or np.any(policy_mass < 0):
                        raise CompositeBuildError(
                            f"selected source has invalid policy mass: {source}"
                        )
                    if category != "current_producer":
                        allowed_versions = {
                            int(checkpoint_by_id[checkpoint_id].get("version", -1))
                            for checkpoint_id in category_specs[category][
                                "checkpoint_ids"
                            ]
                        }
                        contract._validate_selected_opponent_rows(  # noqa: SLF001
                            payload,
                            selected_mask=selected_mask,
                            game_seeds=seeds,
                            job=job,
                            allowed_versions=allowed_versions,
                            colors=selfplay_colors,
                        )
                    arrays: dict[str, np.ndarray] = {}
                    for name in payload.files:
                        values = np.asarray(payload[name])
                        if values.ndim < 1 or values.shape[0] != seeds.size:
                            raise CompositeBuildError(
                                f"source column {name!r} is not row-aligned: {source}"
                            )
                        arrays[name] = values[selected_mask]
            except (KeyError, OSError, ValueError, contract.ContractError) as error:
                raise CompositeBuildError(
                    f"cannot filter source shard {source}: {error}"
                ) from error

            observed = set(map(int, np.asarray(arrays["game_seed"]).tolist()))
            observed_by_job[job_id].update(observed)
            filtered_dir = output_root / "filtered_sources" / category
            filtered_path = filtered_dir / (
                f"{order_by_category[category]:05d}-{job_id}.npz"
            )
            _atomic_npz(filtered_path, arrays)
            filtered_path = filtered_path.resolve(strict=True)
            binding = {
                "contract_sha256": lock["contract_sha256"],
                "audit_file_sha256": audit["file_sha256"],
                "audit_sha256": audit["audit_sha256"],
                "selected_manifest_file_sha256": selected["file_sha256"],
                "selected_records_sha256": selected["records_sha256"],
                "job_id": job_id,
                "category": category,
                "source_path": str(source),
                "source_sha256": before_sha,
                "generation_manifest_path": generation_manifest_by_job[job_id]["path"],
                "generation_manifest_sha256": generation_manifest_by_job[job_id][
                    "sha256"
                ],
            }
            source_id = _digest(binding)
            filtered_record = {
                "path": str(filtered_path),
                "rows": int(np.asarray(arrays["game_seed"]).size),
                "order": int(order_by_category[category]),
                "size_bytes": filtered_path.stat().st_size,
                "sha256": _file_sha256(filtered_path),
                "checkpoint_version": int(producer["version"]),
                "producer_checkpoint_path": str(producer_path),
                "producer_checkpoint_sha256": producer["sha256"],
                "source_id": source_id,
                "source_category": category,
            }
            filtered_records[category].append(filtered_record)
            source_bindings.append({"source_id": source_id, **binding})
            order_by_category[category] += 1
            if _file_sha256(source) != before_sha:
                raise CompositeBuildError(
                    f"source shard changed during filtering: {source}"
                )

    for job_id, selected_seeds in selected_by_job.items():
        if observed_by_job[job_id] != selected_seeds:
            raise CompositeBuildError(
                f"filtered rows do not exactly cover selected games for {job_id}: "
                f"missing={len(selected_seeds - observed_by_job[job_id])} "
                f"unexpected={len(observed_by_job[job_id] - selected_seeds)}"
            )
    for category, records in filtered_records.items():
        source_root = output_root / "filtered_sources" / category
        _atomic_json(
            source_root / "manifest.json",
            {
                "shards": [record["path"] for record in records],
                "public_award_feature_provenance": public_award_by_category[category],
            },
        )
    return filtered_records, source_bindings


def _build_fresh_component(
    *,
    category: str,
    records: list[dict[str, Any]],
    producer: Mapping[str, Any],
    output_root: Path,
    expected_games: int,
    build_memmap_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise CompositeBuildError(f"fresh component {category} has no filtered shards")
    source_root = output_root / "filtered_sources" / category
    corpus_dir = output_root / "corpora" / category
    try:
        meta = build_memmap_fn(source_root, corpus_dir, progress_every=0)
        mass = measure_memmap_component(corpus_dir, meta)
    except (OSError, SystemExit, ValueError) as error:
        raise CompositeBuildError(f"cannot build {category} memmap: {error}") from error
    if mass["game_count"] != expected_games or mass["policy_active_row_count"] <= 0:
        raise CompositeBuildError(
            f"fresh {category} mass differs from selected whole-game quota: {mass}"
        )
    version = int(producer["version"])
    provenance = {
        "schema_version": "flywheel-replay-component-v1",
        "component_id": category,
        "source_category": category,
        "role": "fresh",
        "current_checkpoint_version": version,
        "checkpoint_versions": [version],
        "producer_checkpoints": [
            {
                "version": version,
                "path": records[0]["producer_checkpoint_path"],
                "sha256": producer["sha256"],
            }
        ],
        "row_count": sum(int(record["rows"]) for record in records),
        "shards": records,
        "shard_inventory_sha256": canonical_sha256(records),
        "component_mass": mass,
    }
    provenance_path = output_root / "provenance" / f"{category}.json"
    _atomic_json(provenance_path, provenance)
    provenance_path = provenance_path.resolve(strict=True)
    meta_path = corpus_dir / "corpus_meta.json"
    meta = _load_json(meta_path)
    provenance_ref = {
        "path": str(provenance_path),
        "file_sha256": _file_sha256(provenance_path),
    }
    meta["flywheel_component_provenance"] = provenance_ref
    _atomic_json(meta_path, meta)
    return {
        "component_id": category,
        "source_category": category,
        "game_sampling_ratio": EFFECTIVE_COMPONENT_RATIOS[category],
        "corpus_dir": str(corpus_dir.resolve(strict=True)),
        "corpus_meta_sha256": _file_sha256(meta_path),
        "payload_inventory_sha256": meta["payload_inventory_sha256"],
        "provenance_manifest": str(provenance_path),
        "provenance_manifest_sha256": provenance_ref["file_sha256"],
        "component_mass": mass,
    }


def _load_historical_component(path: Path, *, current_version: int) -> dict[str, Any]:
    reference_path = path.expanduser().resolve(strict=True)
    wrapper = _load_json(reference_path)
    if (
        set(wrapper) != {"schema_version", "component"}
        or wrapper.get("schema_version") != HISTORICAL_COMPONENT_REF_SCHEMA
    ):
        raise CompositeBuildError(
            f"historical component reference must use {HISTORICAL_COMPONENT_REF_SCHEMA}"
        )
    component = wrapper.get("component")
    expected = {
        "component_id",
        "source_category",
        "game_sampling_ratio",
        "corpus_dir",
        "corpus_meta_sha256",
        "payload_inventory_sha256",
        "provenance_manifest",
        "provenance_manifest_sha256",
        "component_mass",
    }
    if not isinstance(component, dict) or set(component) != expected:
        raise CompositeBuildError("historical component fields differ from schema")
    if (
        component["component_id"] != HISTORICAL_REPLAY_CATEGORY
        or component["source_category"] != HISTORICAL_REPLAY_CATEGORY
        or float(component["game_sampling_ratio"]) != 0.20
    ):
        raise CompositeBuildError("historical component identity/ratio drift")
    try:
        corpus_dir = Path(str(component["corpus_dir"])).resolve(strict=True)
        provenance_path = Path(str(component["provenance_manifest"])).resolve(
            strict=True
        )
        meta_path = corpus_dir / "corpus_meta.json"
        meta = _load_json(meta_path)
        if (
            str(corpus_dir) != component["corpus_dir"]
            or str(provenance_path) != component["provenance_manifest"]
            or _file_sha256(meta_path) != component["corpus_meta_sha256"]
            or _file_sha256(provenance_path) != component["provenance_manifest_sha256"]
            or train_bc._validate_memmap_payload_inventory(corpus_dir, meta)  # noqa: SLF001
            != component["payload_inventory_sha256"]
        ):
            raise CompositeBuildError("historical component byte binding drift")
        provenance = train_bc._validate_flywheel_component_provenance(  # noqa: SLF001
            provenance_path,
            component_id=HISTORICAL_REPLAY_CATEGORY,
            corpus_dir=corpus_dir,
            corpus_meta=meta,
        )
    except (OSError, SystemExit, ValueError) as error:
        raise CompositeBuildError(
            f"historical replay verification failed: {error}"
        ) from error
    if (
        provenance["role"] != "replay"
        or int(provenance["current_checkpoint_version"]) != current_version
        or component["component_mass"] != provenance["component_mass"]
    ):
        raise CompositeBuildError("historical replay generation/mass drift")
    return dict(component)


def _build_descriptor(
    *,
    components: list[dict[str, Any]],
    producer_path: Path,
    producer_sha256: str,
    current_version: int,
) -> dict[str, Any]:
    component_ids = [str(component["component_id"]) for component in components]
    expected_ids = [*FRESH_SOURCE_GAME_RATIOS, HISTORICAL_REPLAY_CATEGORY]
    if component_ids != expected_ids:
        raise CompositeBuildError(
            f"component order/identity drift: {component_ids} != {expected_ids}"
        )
    effective = {
        str(component["component_id"]): float(component["game_sampling_ratio"])
        for component in components
    }
    if effective != EFFECTIVE_COMPONENT_RATIOS:
        raise CompositeBuildError(
            "effective component ratios differ from .64/.12/.04/.20"
        )
    provenance_payloads = [
        _load_json(Path(str(component["provenance_manifest"])))
        for component in components
    ]
    checkpoint_versions = sorted(
        {
            int(version)
            for provenance in provenance_payloads
            for version in provenance["checkpoint_versions"]
        }
    )
    provenance_binding = [
        {
            "component_id": component["component_id"],
            "provenance_manifest_sha256": component["provenance_manifest_sha256"],
        }
        for component in components
    ]
    sampling_receipt = build_sampling_receipt(components)
    replay_contract = {
        "schema_version": "flywheel-replay-composite-v2",
        "current_checkpoint_version": int(current_version),
        "initializer_checkpoint_path": str(producer_path),
        "initializer_checkpoint_sha256": producer_sha256,
        "fresh_component_ids": list(FRESH_SOURCE_GAME_RATIOS),
        "replay_component_ids": [HISTORICAL_REPLAY_CATEGORY],
        "fresh_source_game_ratios": dict(FRESH_SOURCE_GAME_RATIOS),
        "effective_component_sampling_ratios": effective,
        "minimum_replay_ratio": 0.20,
        "realized_replay_ratio": 0.20,
        "checkpoint_versions": checkpoint_versions,
        "component_provenance_sha256": canonical_sha256(provenance_binding),
        "sampling_receipt": sampling_receipt,
        "sampling_receipt_sha256": canonical_sha256(sampling_receipt),
    }
    recipe = dict(LEARNER_RECIPE_OVERRIDES)
    return {
        "schema_version": "memmap_composite_v2",
        "diagnostic_only": False,
        "promotion_eligible": True,
        "components": components,
        "learner_recipe_overrides": recipe,
        "learner_recipe_overrides_sha256": canonical_sha256(recipe),
        "policy_kl_anchor_component_ids": [HISTORICAL_REPLAY_CATEGORY],
        "policy_distillation_component_ids": component_ids,
        "value_training_component_ids": component_ids,
        "flywheel_replay_contract": replay_contract,
    }


def build_post_wave_composite(
    *,
    lock_path: Path,
    selected_path: Path,
    audit_path: Path,
    historical_component_path: Path,
    output_root: Path,
    verify_lock_fn: Callable[..., dict[str, Any]] = contract.verify_lock,
    build_memmap_fn: Callable[..., dict[str, Any]] = memmap_builder.build_memmap_corpus,
    verify_descriptor_fn: Callable[[Path], dict[str, Any]] = (
        train_bc._preflight_memmap_composite_descriptor  # noqa: SLF001
    ),
    expected_games: Mapping[str, int] = contract.EXPECTED_GAMES,
) -> dict[str, Any]:
    root = _prepare_output_root(output_root)
    lock, selected, audit, raw_selected = _validated_wave_inputs(
        lock_path,
        selected_path,
        audit_path,
        verify_lock_fn=verify_lock_fn,
    )
    producer = contract._producer(lock)  # noqa: SLF001
    if isinstance(producer.get("version"), bool) or not isinstance(
        producer.get("version"), int
    ):
        raise CompositeBuildError("current producer has no authenticated version")
    producer_path = Path(str(producer["path"])).expanduser().resolve(strict=True)
    if _file_sha256(producer_path) != producer["sha256"]:
        raise CompositeBuildError("current producer checkpoint bytes drifted")

    records_by_category, source_bindings = _filter_wave_shards(
        lock=lock,
        selected=selected,
        audit=audit,
        raw_selected=raw_selected,
        output_root=root,
        expected_games=expected_games,
    )
    components = [
        _build_fresh_component(
            category=category,
            records=records_by_category[category],
            producer=producer,
            output_root=root,
            expected_games=int(expected_games[category]),
            build_memmap_fn=build_memmap_fn,
        )
        for category in FRESH_SOURCE_GAME_RATIOS
    ]
    historical = _load_historical_component(
        historical_component_path, current_version=int(producer["version"])
    )
    components.append(historical)
    descriptor = _build_descriptor(
        components=components,
        producer_path=producer_path,
        producer_sha256=str(producer["sha256"]),
        current_version=int(producer["version"]),
    )
    descriptor_path = root / "memmap_composite.json"
    _atomic_json(descriptor_path, descriptor)
    try:
        verified = verify_descriptor_fn(descriptor_path)
    except (OSError, SystemExit, ValueError) as error:
        raise CompositeBuildError(
            f"final composite preflight failed: {error}"
        ) from error
    receipt = {
        "schema_version": BUILD_RECEIPT_SCHEMA,
        "contract": {
            "path": str(lock_path.expanduser().resolve(strict=True)),
            "contract_sha256": lock["contract_sha256"],
        },
        "selected_game_manifest": {
            "path": str(selected["path"]),
            "file_sha256": selected["file_sha256"],
            "records_sha256": selected["records_sha256"],
            "category_game_counts": dict(expected_games),
        },
        "post_wave_audit": {
            "path": str(audit["path"]),
            "file_sha256": audit["file_sha256"],
            "audit_sha256": audit["audit_sha256"],
            "shard_inventory_sha256": audit["shard_inventory_sha256"],
        },
        "historical_component_reference": {
            "path": str(historical_component_path.expanduser().resolve(strict=True)),
            "file_sha256": _file_sha256(
                historical_component_path.expanduser().resolve(strict=True)
            ),
        },
        "source_bindings": source_bindings,
        "source_bindings_sha256": canonical_sha256(source_bindings),
        "descriptor": {
            "path": str(descriptor_path.resolve(strict=True)),
            "file_sha256": _file_sha256(descriptor_path),
            "fingerprint": canonical_sha256(descriptor),
        },
        "sampling_receipt": descriptor["flywheel_replay_contract"]["sampling_receipt"],
        "verified_descriptor_fingerprint": verified.get("descriptor_fingerprint"),
    }
    receipt["receipt_sha256"] = _digest(receipt)
    _atomic_json(root / "build_receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--selected-game-manifest", type=Path, required=True)
    parser.add_argument("--post-wave-audit", type=Path, required=True)
    parser.add_argument("--historical-replay-component", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = build_post_wave_composite(
            lock_path=args.lock,
            selected_path=args.selected_game_manifest,
            audit_path=args.post_wave_audit,
            historical_component_path=args.historical_replay_component,
            output_root=args.out,
        )
    except CompositeBuildError as error:
        parser.error(str(error))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
