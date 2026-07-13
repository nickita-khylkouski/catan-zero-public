#!/usr/bin/env python3
"""Seal one audited prior A1 corpus as portable historical replay evidence.

This is the only supported constructor for the 20% historical component used
by ``a1_build_post_wave_composite.py``.  It replays the prior lock, selection,
post-wave audit, source-shard hashes, and exact selected-row coverage before it
adds replay provenance to the existing memmap corpus.  The resulting reference
is later projected into the new composite's portable authority bundle.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from catan_zero.rl.flywheel.composite_contract import (  # noqa: E402
    HISTORICAL_REPLAY_CATEGORY,
    canonical_sha256,
    measure_memmap_component,
)
from tools import a1_build_post_wave_composite as builder  # noqa: E402
from tools import a1_pre_wave_contract as contract  # noqa: E402
from tools import build_memmap_corpus as memmap_builder  # noqa: E402
from tools import train_bc  # noqa: E402


def seal_historical_replay_component(
    *,
    lock_path: Path,
    selected_path: Path,
    audit_path: Path,
    corpus_dir: Path,
    producer_version: int,
    current_version: int,
    output_path: Path,
) -> dict[str, Any]:
    """Verify and seal the exact prior wave; never infer provenance from a corpus."""

    if isinstance(current_version, bool) or not isinstance(current_version, int):
        raise builder.CompositeBuildError("current version must be an integer")
    try:
        lock_path = lock_path.expanduser().resolve(strict=True)
        selected_path = selected_path.expanduser().resolve(strict=True)
        audit_path = audit_path.expanduser().resolve(strict=True)
        corpus_dir = corpus_dir.expanduser().resolve(strict=True)
    except OSError as error:
        raise builder.CompositeBuildError(
            f"cannot resolve historical replay input: {error}"
        ) from error
    output_path = output_path.expanduser().absolute()
    if output_path.exists():
        raise builder.CompositeBuildError(
            f"historical replay output already exists: {output_path}"
        )
    try:
        lock = contract.verify_lock(lock_path, require_all_job_claims=False)
        selected = memmap_builder._load_a1_selected_game_manifest(selected_path)  # noqa: SLF001
        audit = memmap_builder._load_a1_post_wave_audit(audit_path, selected)  # noqa: SLF001
    except (OSError, SystemExit, contract.ContractError) as error:
        raise builder.CompositeBuildError(
            f"historical wave authority verification failed: {error}"
        ) from error
    if (
        selected["a1_contract_sha256"] != lock["contract_sha256"]
        or audit["contract_sha256"] != lock["contract_sha256"]
    ):
        raise builder.CompositeBuildError(
            "historical lock/selection/audit contract identities differ"
        )
    producer = contract._producer(lock)  # noqa: SLF001
    declared_version = producer.get("version")
    prior_version = producer_version
    if (
        isinstance(prior_version, bool)
        or not isinstance(prior_version, int)
        or (
            declared_version is not None
            and declared_version != prior_version
        )
        or prior_version >= current_version
    ):
        raise builder.CompositeBuildError(
            "historical producer version must be explicit, match the lock when "
            "present, and be strictly older than current"
        )
    producer_path = Path(str(producer["path"])).expanduser().resolve(strict=True)
    if builder._file_sha256(producer_path) != producer["sha256"]:  # noqa: SLF001
        raise builder.CompositeBuildError("historical producer checkpoint drift")

    raw_selected = builder._load_json(selected_path)  # noqa: SLF001
    category_counts = raw_selected.get("category_game_counts")
    if not isinstance(category_counts, dict):
        raise builder.CompositeBuildError(
            "historical selected-game manifest has no category quotas"
        )
    selected_by_job, _owners, _records = builder._selection_by_job(  # noqa: SLF001
        lock, raw_selected, expected_games=category_counts
    )
    jobs = {str(job["job_id"]): job for job in lock["fleet"]["jobs"]}

    raw_audit = builder._load_json(audit_path)  # noqa: SLF001
    generation_manifests: dict[str, dict[str, str]] = {}
    for record in raw_audit.get("shards", []):
        if not isinstance(record, dict) or record.get("kind") != "generation_manifest":
            continue
        job_id = str(record.get("job_id", ""))
        category = str(record.get("category", ""))
        if job_id not in selected_by_job:
            continue
        if job_id not in jobs or category != jobs[job_id].get("category"):
            raise builder.CompositeBuildError(
                "historical generation manifest has unknown job/category"
            )
        path = Path(str(record.get("path", ""))).resolve(strict=True)
        sha256 = str(record.get("sha256", ""))
        if builder._file_sha256(path) != sha256:  # noqa: SLF001
            raise builder.CompositeBuildError(
                f"historical generation manifest bytes drifted: {path}"
            )
        if job_id in generation_manifests:
            raise builder.CompositeBuildError(
                f"historical audit repeats generation manifest for {job_id}"
            )
        generation_manifests[job_id] = {"path": str(path), "sha256": sha256}
    if set(generation_manifests) != set(selected_by_job):
        raise builder.CompositeBuildError(
            "historical audit generation manifests do not cover selected jobs"
        )

    data_by_job: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in audit["data_shards"]:
        job_id = str(record.get("job_id", ""))
        if job_id in selected_by_job:
            data_by_job[job_id].append(record)
    if set(data_by_job) != set(selected_by_job):
        raise builder.CompositeBuildError(
            "historical audited data shards do not cover selected jobs"
        )

    shard_records: list[dict[str, Any]] = []
    source_bindings: list[dict[str, Any]] = []
    observed_by_job: dict[str, set[int]] = defaultdict(set)
    order = 0
    for job_id in [str(job["job_id"]) for job in lock["fleet"]["jobs"]]:
        if job_id not in selected_by_job:
            continue
        job = jobs[job_id]
        category = str(job["category"])
        selected_seeds = selected_by_job[job_id]
        for audited in data_by_job[job_id]:
            source = Path(str(audited["path"])).resolve(strict=True)
            source_sha = builder._file_sha256(source)  # noqa: SLF001
            if source_sha != audited["sha256"]:
                raise builder.CompositeBuildError(
                    f"historical source shard bytes drifted: {source}"
                )
            try:
                with np.load(source, allow_pickle=False) as payload:
                    seeds = np.asarray(payload["game_seed"], dtype=np.int64)
                    mask = np.isin(
                        seeds, np.asarray(sorted(selected_seeds), dtype=np.int64)
                    )
                    rows = int(np.count_nonzero(mask))
                    if rows:
                        if (
                            "terminated" not in payload.files
                            or "truncated" not in payload.files
                            or np.any(~np.asarray(payload["terminated"], dtype=bool)[mask])
                            or np.any(np.asarray(payload["truncated"], dtype=bool)[mask])
                        ):
                            raise builder.CompositeBuildError(
                                f"historical selected rows are incomplete: {source}"
                            )
                        observed_by_job[job_id].update(
                            map(int, np.unique(seeds[mask]).tolist())
                        )
            except (KeyError, OSError, ValueError) as error:
                raise builder.CompositeBuildError(
                    f"cannot inspect historical source shard {source}: {error}"
                ) from error
            if rows == 0:
                continue
            manifest = generation_manifests[job_id]
            binding = {
                "contract_sha256": lock["contract_sha256"],
                "audit_file_sha256": audit["file_sha256"],
                "audit_sha256": audit["audit_sha256"],
                "selected_manifest_file_sha256": selected["file_sha256"],
                "selected_records_sha256": selected["records_sha256"],
                "job_id": job_id,
                "category": category,
                "source_path": str(source),
                "source_sha256": source_sha,
                "generation_manifest_path": manifest["path"],
                "generation_manifest_sha256": manifest["sha256"],
            }
            source_id = builder._binding_source_id(binding)  # noqa: SLF001
            source_bindings.append({"source_id": source_id, **binding})
            shard_records.append(
                {
                    "path": str(source),
                    "rows": rows,
                    "order": order,
                    "size_bytes": source.stat().st_size,
                    "sha256": source_sha,
                    "checkpoint_version": int(prior_version),
                    "producer_checkpoint_path": str(producer_path),
                    "producer_checkpoint_sha256": producer["sha256"],
                    "source_id": source_id,
                    "source_category": category,
                }
            )
            order += 1
    for job_id, expected in selected_by_job.items():
        if observed_by_job[job_id] != expected:
            raise builder.CompositeBuildError(
                f"historical source rows do not cover selected games for {job_id}"
            )

    meta_path = corpus_dir / "corpus_meta.json"
    meta = builder._load_json(meta_path)  # noqa: SLF001
    if (
        meta.get("selected_game_seed_manifest", {}).get("file_sha256")
        != selected["file_sha256"]
        or meta.get("a1_post_wave_audit", {}).get("file_sha256")
        != audit["file_sha256"]
        or train_bc._validate_memmap_payload_inventory(corpus_dir, meta)  # noqa: SLF001
        != meta.get("payload_inventory_sha256")
        or sum(int(record["rows"]) for record in shard_records)
        != int(meta.get("row_count", -1))
    ):
        raise builder.CompositeBuildError(
            "historical corpus differs from its selected/audited source rows"
        )
    mass = measure_memmap_component(corpus_dir, meta)
    provenance = {
        "schema_version": "flywheel-replay-component-v1",
        "component_id": HISTORICAL_REPLAY_CATEGORY,
        "source_category": HISTORICAL_REPLAY_CATEGORY,
        "role": "replay",
        "current_checkpoint_version": current_version,
        "checkpoint_versions": [int(prior_version)],
        "producer_checkpoints": [
            {
                "version": int(prior_version),
                "path": str(producer_path),
                "sha256": producer["sha256"],
            }
        ],
        "row_count": int(meta["row_count"]),
        "shards": shard_records,
        "shard_inventory_sha256": canonical_sha256(shard_records),
        "component_mass": mass,
    }
    provenance_path = output_path.parent / f"{output_path.stem}.provenance.json"
    builder._atomic_json(provenance_path, provenance)  # noqa: SLF001
    provenance_ref = builder._artifact_ref(provenance_path)  # noqa: SLF001

    component = {
        "component_id": HISTORICAL_REPLAY_CATEGORY,
        "source_category": HISTORICAL_REPLAY_CATEGORY,
        "game_sampling_ratio": 0.20,
        "corpus_dir": str(corpus_dir),
        "corpus_meta_sha256": builder._file_sha256(meta_path),  # noqa: SLF001
        "payload_inventory_sha256": meta["payload_inventory_sha256"],
        "provenance_manifest": str(provenance_path.resolve(strict=True)),
        "provenance_manifest_sha256": provenance_ref["file_sha256"],
        "component_mass": mass,
    }
    authority: dict[str, Any] = {
        "schema_version": builder.HISTORICAL_AUTHORITY_SCHEMA,
        "source_contract": {
            **builder._artifact_ref(lock_path),  # noqa: SLF001
            "contract_sha256": lock["contract_sha256"],
        },
        "selected_game_manifest": {
            **builder._artifact_ref(selected_path),  # noqa: SLF001
            "manifest_sha256": selected["manifest_sha256"],
            "records_sha256": selected["records_sha256"],
            "selected_game_seed_set_sha256": selected[
                "selected_game_seed_set_sha256"
            ],
        },
        "post_wave_audit": {
            **builder._artifact_ref(audit_path),  # noqa: SLF001
            "audit_sha256": audit["audit_sha256"],
            "shard_inventory_sha256": audit["shard_inventory_sha256"],
        },
        "source_bindings": source_bindings,
        "source_bindings_sha256": canonical_sha256(source_bindings),
        "component_provenance_sha256": provenance_ref["file_sha256"],
        "component_payload_inventory_sha256": meta["payload_inventory_sha256"],
    }
    authority["authority_sha256"] = builder._digest(authority)  # noqa: SLF001
    wrapper = {
        "schema_version": builder.HISTORICAL_COMPONENT_REF_SCHEMA,
        "component": component,
        "authority": authority,
    }
    builder._atomic_json(output_path, wrapper)  # noqa: SLF001
    return wrapper


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--selected-game-manifest", required=True, type=Path)
    parser.add_argument("--post-wave-audit", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--producer-version", required=True, type=int)
    parser.add_argument("--current-version", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        payload = seal_historical_replay_component(
            lock_path=args.lock,
            selected_path=args.selected_game_manifest,
            audit_path=args.post_wave_audit,
            corpus_dir=args.corpus,
            producer_version=args.producer_version,
            current_version=args.current_version,
            output_path=args.out,
        )
    except builder.CompositeBuildError as error:
        parser.error(str(error))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
