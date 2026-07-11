"""Fail-closed A1 game-selection ingestion for the memmap trainer corpus."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from build_memmap_corpus import (  # type: ignore  # noqa: E402
    A1_CATEGORY_GAME_COUNTS,
    A1_SELECTED_GAME_COUNT,
    _file_sha256,
    _game_seed_set_sha256,
    _value_sha256,
    build_memmap_corpus,
)


_PRODUCER_SHA = "sha256:" + "1" * 64
_OPPONENT_SHA = "sha256:" + "2" * 64
_CONTRACT_SHA = "sha256:" + "3" * 64


def _category(index: int) -> str:
    if index < A1_CATEGORY_GAME_COUNTS["current_producer"]:
        return "current_producer"
    if index < (
        A1_CATEGORY_GAME_COUNTS["current_producer"]
        + A1_CATEGORY_GAME_COUNTS["recent_history"]
    ):
        return "recent_history"
    return "hard_negative"


def _write_selected_manifest(path: Path, *, first_seed: int = 100_000) -> dict:
    records = [
        {
            "game_seed": first_seed + index,
            "job_id": f"gpu{index % 24:02d}-{_category(index)}",
            "worker_id": f"gpu{index % 24:02d}",
            "category": _category(index),
            "producer_checkpoint_sha256": _PRODUCER_SHA,
            "opponent_checkpoint_sha256": [
                _PRODUCER_SHA
                if _category(index) == "current_producer"
                else _OPPONENT_SHA
            ],
            "split": "validation" if index >= 10_800 else "train",
        }
        for index in range(A1_SELECTED_GAME_COUNT)
    ]
    training_seeds = [
        record["game_seed"] for record in records if record["split"] == "train"
    ]
    validation_seeds = [
        record["game_seed"] for record in records if record["split"] == "validation"
    ]
    payload = {
        "schema_version": "a1-selected-training-games-v1",
        "a1_contract_sha256": _CONTRACT_SHA,
        "selection_rule": "lowest_seed_complete_per_job",
        "selected_game_count": len(records),
        "category_game_counts": dict(A1_CATEGORY_GAME_COUNTS),
        "selected_game_seed_set_sha256": _game_seed_set_sha256(
            [record["game_seed"] for record in records]
        ),
        "training_game_count": len(training_seeds),
        "training_game_seed_set_sha256": _game_seed_set_sha256(training_seeds),
        "validation_game_count": len(validation_seeds),
        "validation_game_seed_set_sha256": _game_seed_set_sha256(validation_seeds),
        "records_sha256": _value_sha256(records),
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _write_audit(
    path: Path,
    *,
    manifest_path: Path,
    manifest: dict,
    shards: list[Path],
) -> dict:
    selected_seed_set = {int(record["game_seed"]) for record in manifest["records"]}
    validation_seed_set = {
        int(record["game_seed"])
        for record in manifest["records"]
        if record["split"] == "validation"
    }
    selected_row_count = 0
    validation_row_count = 0
    for shard in shards:
        with np.load(shard, allow_pickle=False) as payload:
            shard_seeds = np.asarray(payload["game_seed"], dtype=np.int64)
            selected_row_count += int(
                np.isin(
                    shard_seeds,
                    np.asarray(sorted(selected_seed_set), dtype=np.int64),
                ).sum()
            )
            validation_row_count += int(
                np.isin(
                    shard_seeds,
                    np.asarray(sorted(validation_seed_set), dtype=np.int64),
                ).sum()
            )
    validation_path = path.with_suffix(".validation_seeds.json")
    validation_seeds = sorted(
        int(record["game_seed"])
        for record in manifest["records"]
        if record["split"] == "validation"
    )
    validation_payload = {
        "schema_version": "train-validation-game-seeds-v1",
        "a1_contract_sha256": _CONTRACT_SHA,
        "validation_fraction": 0.05,
        "validation_seed": 17,
        "validation_max_samples": 0,
        "validation_game_seed_ranges": [],
        "validation_game_seed_count": len(validation_seeds),
        "validation_row_count": validation_row_count,
        "validation_game_seed_set_sha256": _game_seed_set_sha256(
            validation_seeds
        ),
        "game_seeds": validation_seeds,
    }
    validation_path.write_text(
        json.dumps(validation_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    shard_records = [
        {
            "kind": "data_shard",
            "path": str(shard.resolve()),
            "sha256": _file_sha256(shard),
            "job_id": f"job-{index}",
            "category": "current_producer",
            "producer_checkpoint_sha256": _PRODUCER_SHA,
            "opponent_checkpoint_sha256": [_PRODUCER_SHA],
            "search_operator_sha256": "sha256:" + "4" * 64,
            "effective_search_config_sha256": "sha256:" + "5" * 64,
            "evaluator_sha256": "sha256:" + "6" * 64,
        }
        for index, shard in enumerate(shards)
    ]
    payload = {
        "schema_version": "a1-post-wave-audit-v2",
        "passed": True,
        "errors": [],
        "rows": selected_row_count,
        "contract_sha256": _CONTRACT_SHA,
        "shards": shard_records,
        "shard_inventory_sha256": _value_sha256(shard_records),
        "source_provenance": {
            "current_producer": {
                "producer_checkpoint_sha256": _PRODUCER_SHA,
                "opponent_checkpoint_sha256": [_PRODUCER_SHA],
            }
        },
        "selected_training_games": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": _value_sha256(manifest),
            "manifest_file_sha256": _file_sha256(manifest_path),
            "selected_game_count": manifest["selected_game_count"],
            "selected_game_seed_set_sha256": manifest[
                "selected_game_seed_set_sha256"
            ],
            "records_sha256": manifest["records_sha256"],
        },
        "validation_holdout": {
            "manifest": str(validation_path.resolve()),
            "manifest_sha256": _value_sha256(validation_payload),
            "manifest_file_sha256": _file_sha256(validation_path),
            "validation_game_seed_count": len(validation_seeds),
            "validation_game_seed_set_sha256": validation_payload[
                "validation_game_seed_set_sha256"
            ],
        },
    }
    payload["audit_sha256"] = _value_sha256(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _upgrade_audit_to_relocated_v3(audit_path: Path, shard: Path) -> dict:
    """Bind a v2 fixture audit to the same local shard through a typed map."""

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    relative = shard.relative_to(audit_path.parent)
    files = [
        {
            "source_path": "/sealed/a1/job-0/shard0.npz",
            "relative_path": relative.as_posix(),
            "size_bytes": shard.stat().st_size,
            "sha256": _file_sha256(shard),
            "job_id": "job-0",
            "host_alias": "h0",
        }
    ]
    identities = [
        {
            "job_id": "job-0",
            "worker_id": "worker-0",
            "host_alias": "h0",
            "gpu": 0,
            "category": "current_producer",
            "output_dir": "/sealed/a1/job-0",
        }
    ]
    relocation = {
        "schema_version": "a1-fleet-harvest-relocation-v1",
        "contract_path": "/sealed/lock.json",
        "contract_file_sha256": "sha256:" + "7" * 64,
        "contract_sha256": _CONTRACT_SHA,
        "render_path": "/sealed/render.json",
        "render_file_sha256": "sha256:" + "8" * 64,
        "render_sha256": "sha256:" + "9" * 64,
        "host_count": 1,
        "job_count": 1,
        "job_identities": identities,
        "job_identities_sha256": _value_sha256(identities),
        "files": files,
        "file_inventory_sha256": _value_sha256(files),
    }
    relocation["relocation_sha256"] = _value_sha256(relocation)
    relocation_path = audit_path.parent / "relocation_map.json"
    relocation_path.write_text(
        json.dumps(relocation, indent=2, sort_keys=True), encoding="utf-8"
    )
    audit["schema_version"] = "a1-post-wave-audit-v3"
    audit["harvest_relocation"] = {
        "path": str(relocation_path.resolve()),
        "file_sha256": _file_sha256(relocation_path),
        "relocation_sha256": relocation["relocation_sha256"],
        "render_sha256": relocation["render_sha256"],
        "job_identities_sha256": relocation["job_identities_sha256"],
        "file_inventory_sha256": relocation["file_inventory_sha256"],
    }
    audit.pop("audit_sha256")
    audit["audit_sha256"] = _value_sha256(audit)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return relocation


def _write_shard(
    path: Path,
    seeds: np.ndarray | list[int],
    *,
    terminated: np.ndarray | None = None,
    truncated: np.ndarray | None = None,
    obs_fill: float = 0.0,
) -> None:
    game_seed = np.asarray(seeds, dtype=np.int64)
    n = int(game_seed.size)
    if terminated is None:
        terminated = np.ones(n, dtype=bool)
    if truncated is None:
        truncated = np.zeros(n, dtype=bool)
    legal = np.tile(np.asarray([0, -1], dtype=np.int16), (n, 1))
    np.savez(
        path,
        obs=np.full((n, 4), obs_fill, dtype=np.float16),
        legal_action_ids=legal,
        legal_action_context=np.zeros((n, 2, 1), dtype=np.float16),
        action_taken=np.zeros(n, dtype=np.int16),
        game_seed=game_seed,
        terminated=np.asarray(terminated, dtype=bool),
        truncated=np.asarray(truncated, dtype=bool),
    )


def _read_seed_memmap(corpus: Path, rows: int) -> np.ndarray:
    return np.memmap(corpus / "game_seed.dat", dtype="<i8", mode="r", shape=(rows,))


def _write_source_attestation(
    source: Path, *, contract_sha256: str = _CONTRACT_SHA
) -> Path:
    path = source / "a1_contract.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "a1-generation-job-attestation-v2",
                "contract_sha256": contract_sha256,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_direct_a1_source_attestation_closes_generic_ingest_bypass(
    tmp_path: Path,
) -> None:
    source = tmp_path / "teacher"
    source.mkdir()
    _write_source_attestation(source)

    with pytest.raises(SystemExit, match="A1 source attestation detected"):
        build_memmap_corpus(source, tmp_path / "corpus", progress_every=0)


def test_ancestor_a1_attestation_closes_nested_worker_ingest_bypass(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    worker = job / "worker_000"
    worker.mkdir(parents=True)
    _write_source_attestation(job)
    _write_shard(worker / "shard0.npz", [101])

    with pytest.raises(SystemExit, match="A1 source attestation detected"):
        build_memmap_corpus(worker, tmp_path / "corpus", progress_every=0)


def test_audited_ingest_rejects_source_attestation_for_other_contract(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "selected.json"
    manifest = _write_selected_manifest(manifest_path)
    seeds = np.asarray(
        [record["game_seed"] for record in manifest["records"]], dtype=np.int64
    )
    source = tmp_path / "teacher"
    source.mkdir()
    shard = source / "shard0.npz"
    _write_shard(shard, seeds)
    _write_source_attestation(source, contract_sha256="sha256:" + "b" * 64)
    audit_path = tmp_path / "audit.json"
    _write_audit(
        audit_path,
        manifest_path=manifest_path,
        manifest=manifest,
        shards=[shard],
    )

    with pytest.raises(SystemExit, match="do not all bind"):
        build_memmap_corpus(
            source,
            tmp_path / "corpus",
            selected_game_seed_manifest=manifest_path,
            a1_post_wave_audit=audit_path,
            progress_every=0,
        )


def test_selected_games_filter_preserves_game_split_across_shards(tmp_path: Path):
    manifest = tmp_path / "selected.json"
    payload = _write_selected_manifest(manifest)
    seeds = np.asarray(
        [record["game_seed"] for record in payload["records"]], dtype=np.int64
    )
    source = tmp_path / "teacher"
    source.mkdir()
    # Seed 106000 has two decision rows split exactly across the file boundary.
    _write_shard(source / "shard0.npz", seeds[:6_001])
    _write_shard(
        source / "shard1.npz", np.concatenate(([seeds[6_000]], seeds[6_001:]))
    )
    audit = tmp_path / "audit.json"
    _write_audit(
        audit,
        manifest_path=manifest,
        manifest=payload,
        shards=[source / "shard0.npz", source / "shard1.npz"],
    )

    meta = build_memmap_corpus(
        source,
        tmp_path / "corpus",
        selected_game_seed_manifest=manifest,
        a1_post_wave_audit=audit,
        progress_every=0,
    )

    assert meta["row_count"] == A1_SELECTED_GAME_COUNT + 1
    assert meta["stats"]["has_duplicate_game_seeds"] is False
    actual = _read_seed_memmap(tmp_path / "corpus", meta["row_count"])
    expected = {int(record["game_seed"]) for record in payload["records"]}
    assert set(map(int, actual.tolist())) == expected


def test_relocated_v3_audit_is_consumed_without_losing_shard_identity(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "selected.json"
    manifest = _write_selected_manifest(manifest_path)
    source = tmp_path / "jobs/job-0"
    source.mkdir(parents=True)
    shard = source / "shard0.npz"
    _write_shard(shard, [record["game_seed"] for record in manifest["records"]])
    audit_path = tmp_path / "audit.json"
    audit = _write_audit(
        audit_path, manifest_path=manifest_path, manifest=manifest, shards=[shard]
    )
    relocation = _upgrade_audit_to_relocated_v3(audit_path, shard)

    meta = build_memmap_corpus(
        source,
        tmp_path / "corpus",
        selected_game_seed_manifest=manifest_path,
        a1_post_wave_audit=audit_path,
        progress_every=0,
    )

    bound = meta["a1_post_wave_audit"]["harvest_relocation"]
    assert bound["relocation_sha256"] == relocation["relocation_sha256"]
    assert bound["file_inventory_sha256"] == relocation["file_inventory_sha256"]
    assert meta["a1_post_wave_audit"]["shard_inventory_sha256"] == audit[
        "shard_inventory_sha256"
    ]


def test_relocated_v3_audit_rejects_changed_map_bytes(tmp_path: Path) -> None:
    manifest_path = tmp_path / "selected.json"
    manifest = _write_selected_manifest(manifest_path)
    source = tmp_path / "jobs/job-0"
    source.mkdir(parents=True)
    shard = source / "shard0.npz"
    _write_shard(shard, [record["game_seed"] for record in manifest["records"]])
    audit_path = tmp_path / "audit.json"
    _write_audit(
        audit_path, manifest_path=manifest_path, manifest=manifest, shards=[shard]
    )
    _upgrade_audit_to_relocated_v3(audit_path, shard)
    relocation_path = tmp_path / "relocation_map.json"
    relocation = json.loads(relocation_path.read_text(encoding="utf-8"))
    relocation["render_path"] = "/tampered/render.json"
    relocation_path.write_text(json.dumps(relocation), encoding="utf-8")

    with pytest.raises(SystemExit, match="relocation binding/digest mismatch"):
        build_memmap_corpus(
            source,
            tmp_path / "corpus",
            selected_game_seed_manifest=manifest_path,
            a1_post_wave_audit=audit_path,
            progress_every=0,
        )


def test_selected_games_filter_excludes_unselected_complete_and_truncated_reserve(
    tmp_path: Path,
):
    manifest = tmp_path / "selected.json"
    payload = _write_selected_manifest(manifest)
    selected = np.asarray(
        [record["game_seed"] for record in payload["records"]], dtype=np.int64
    )
    reserve = np.asarray([900_000, 900_001, 900_002], dtype=np.int64)
    all_seeds = np.concatenate((selected, reserve))
    source = tmp_path / "teacher"
    source.mkdir()
    _write_shard(
        source / "shard0.npz",
        all_seeds,
        terminated=np.concatenate(
            (np.ones(selected.size, dtype=bool), np.asarray([True, False, False]))
        ),
        truncated=np.concatenate(
            (np.zeros(selected.size, dtype=bool), np.asarray([False, True, False]))
        ),
    )
    audit = tmp_path / "audit.json"
    _write_audit(
        audit,
        manifest_path=manifest,
        manifest=payload,
        shards=[source / "shard0.npz"],
    )

    meta = build_memmap_corpus(
        source,
        tmp_path / "corpus",
        selected_game_seed_manifest=manifest,
        a1_post_wave_audit=audit,
        progress_every=0,
    )

    assert meta["row_count"] == A1_SELECTED_GAME_COUNT
    actual = _read_seed_memmap(tmp_path / "corpus", meta["row_count"])
    np.testing.assert_array_equal(actual, selected)
    assert not set(map(int, reserve)).intersection(map(int, actual.tolist()))


def test_a1_ingest_forbids_full_rows_only_second_filter(tmp_path: Path) -> None:
    manifest_path = tmp_path / "selected.json"
    _write_selected_manifest(manifest_path)
    with pytest.raises(SystemExit, match="full-rows-only is forbidden"):
        build_memmap_corpus(
            tmp_path / "unused-source",
            tmp_path / "corpus",
            selected_game_seed_manifest=manifest_path,
            a1_post_wave_audit=tmp_path / "audit-not-read.json",
            full_rows_only=True,
            progress_every=0,
        )


def test_selected_games_filter_rejects_missing_selected_seed(tmp_path: Path):
    manifest = tmp_path / "selected.json"
    payload = _write_selected_manifest(manifest)
    selected = np.asarray(
        [record["game_seed"] for record in payload["records"]], dtype=np.int64
    )
    source = tmp_path / "teacher"
    source.mkdir()
    _write_shard(source / "shard0.npz", selected[1:])
    audit = tmp_path / "audit.json"
    _write_audit(
        audit,
        manifest_path=manifest,
        manifest=payload,
        shards=[source / "shard0.npz"],
    )

    with pytest.raises(SystemExit, match=r"missing=1 unexpected=0"):
        build_memmap_corpus(
            source,
            tmp_path / "corpus",
            selected_game_seed_manifest=manifest,
            a1_post_wave_audit=audit,
            progress_every=0,
        )


def test_selected_games_filter_rejects_tampered_record_digest(tmp_path: Path):
    manifest = tmp_path / "selected.json"
    payload = _write_selected_manifest(manifest)
    payload["records"][0]["job_id"] = "tampered-job"
    # Keep both declared digests untouched: either must make the sidecar fail.
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(SystemExit, match="records_sha256 mismatch"):
        build_memmap_corpus(
            tmp_path / "does-not-matter",
            tmp_path / "corpus",
            selected_game_seed_manifest=manifest,
            progress_every=0,
        )


def test_selected_games_metadata_binds_absolute_sidecar_and_digests(tmp_path: Path):
    manifest = tmp_path / "selected.json"
    payload = _write_selected_manifest(manifest)
    selected = np.asarray(
        [record["game_seed"] for record in payload["records"]], dtype=np.int64
    )
    source = tmp_path / "teacher"
    source.mkdir()
    _write_shard(source / "shard0.npz", selected)
    audit_path = tmp_path / "audit.json"
    audit = _write_audit(
        audit_path,
        manifest_path=manifest,
        manifest=payload,
        shards=[source / "shard0.npz"],
    )
    corpus = tmp_path / "corpus"

    meta = build_memmap_corpus(
        source,
        corpus,
        selected_game_seed_manifest=manifest,
        a1_post_wave_audit=audit_path,
        progress_every=0,
    )
    persisted = json.loads((corpus / "corpus_meta.json").read_text(encoding="utf-8"))

    expected = {
        "path": str(manifest.resolve()),
        "file_sha256": _file_sha256(manifest),
        "a1_contract_sha256": _CONTRACT_SHA,
        "selected_game_count": A1_SELECTED_GAME_COUNT,
        "selected_game_seed_set_sha256": payload[
            "selected_game_seed_set_sha256"
        ],
        "training_game_count": payload["training_game_count"],
        "training_game_seed_set_sha256": payload[
            "training_game_seed_set_sha256"
        ],
        "validation_game_count": payload["validation_game_count"],
        "validation_game_seed_set_sha256": payload[
            "validation_game_seed_set_sha256"
        ],
        "records_sha256": payload["records_sha256"],
    }
    assert meta["selected_game_seed_manifest"] == expected
    assert persisted["selected_game_seed_manifest"] == expected
    inventory = persisted["payload_inventory"]
    assert persisted["payload_inventory_schema"] == "memmap-payload-inventory-v1"
    assert persisted["payload_inventory_sha256"] == _value_sha256(inventory)
    assert [record["filename"] for record in inventory] == sorted(
        path.name for path in corpus.glob("*.dat")
    )
    for record in inventory:
        payload_path = corpus / record["filename"]
        assert record["size_bytes"] == payload_path.stat().st_size
        assert record["sha256"] == _file_sha256(payload_path)
    raw_digest = "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert expected["file_sha256"] == raw_digest
    validation_path = audit_path.with_suffix(".validation_seeds.json")
    validation_payload = json.loads(validation_path.read_text(encoding="utf-8"))
    assert persisted["a1_post_wave_audit"] == {
        "path": str(audit_path.resolve()),
        "file_sha256": _file_sha256(audit_path),
        "audit_sha256": audit["audit_sha256"],
        "contract_sha256": _CONTRACT_SHA,
        "shard_inventory_sha256": audit["shard_inventory_sha256"],
        "source_provenance": audit["source_provenance"],
        "selected_row_count": audit["rows"],
        "training_row_count": audit["rows"]
        - validation_payload["validation_row_count"],
        "validation_holdout": {
            "path": str(validation_path.resolve()),
            "file_sha256": _file_sha256(validation_path),
            "manifest_sha256": _value_sha256(validation_payload),
            "a1_contract_sha256": _CONTRACT_SHA,
            "validation_game_seed_count": validation_payload[
                "validation_game_seed_count"
            ],
            "validation_row_count": validation_payload["validation_row_count"],
            "validation_game_seed_set_sha256": validation_payload[
                "validation_game_seed_set_sha256"
            ],
        },
    }


def test_selected_games_reject_same_seed_alternate_shard_not_in_audit(tmp_path: Path):
    manifest_path = tmp_path / "selected.json"
    manifest = _write_selected_manifest(manifest_path)
    seeds = np.asarray(
        [record["game_seed"] for record in manifest["records"]], dtype=np.int64
    )
    audited_source = tmp_path / "audited"
    alternate_source = tmp_path / "alternate"
    audited_source.mkdir()
    alternate_source.mkdir()
    _write_shard(audited_source / "shard0.npz", seeds)
    # Identical game seeds and row count are insufficient: this is a different
    # file/path and therefore not the inventory accepted by the deep audit.
    _write_shard(alternate_source / "shard0.npz", seeds)
    audit_path = tmp_path / "audit.json"
    _write_audit(
        audit_path,
        manifest_path=manifest_path,
        manifest=manifest,
        shards=[audited_source / "shard0.npz"],
    )

    with pytest.raises(SystemExit, match=r"inventory differs.*missing=1 unexpected=1"):
        build_memmap_corpus(
            alternate_source,
            tmp_path / "corpus",
            selected_game_seed_manifest=manifest_path,
            a1_post_wave_audit=audit_path,
            progress_every=0,
        )


def test_selected_games_reject_same_path_same_seeds_alternate_bytes(tmp_path: Path):
    manifest_path = tmp_path / "selected.json"
    manifest = _write_selected_manifest(manifest_path)
    seeds = [record["game_seed"] for record in manifest["records"]]
    source = tmp_path / "teacher"
    source.mkdir()
    shard = source / "shard0.npz"
    _write_shard(shard, seeds, obs_fill=0.0)
    audit_path = tmp_path / "audit.json"
    _write_audit(
        audit_path,
        manifest_path=manifest_path,
        manifest=manifest,
        shards=[shard],
    )
    # Preserve every selected seed/status but replace the audited training rows.
    _write_shard(shard, seeds, obs_fill=1.0)

    with pytest.raises(SystemExit, match="changed from the passing A1 audit"):
        build_memmap_corpus(
            source,
            tmp_path / "corpus",
            selected_game_seed_manifest=manifest_path,
            a1_post_wave_audit=audit_path,
            progress_every=0,
        )


def test_selected_games_reject_tampered_validation_sidecar(tmp_path: Path) -> None:
    manifest_path = tmp_path / "selected.json"
    manifest = _write_selected_manifest(manifest_path)
    source = tmp_path / "teacher"
    source.mkdir()
    shard = source / "shard0.npz"
    _write_shard(shard, [record["game_seed"] for record in manifest["records"]])
    audit_path = tmp_path / "audit.json"
    _write_audit(
        audit_path,
        manifest_path=manifest_path,
        manifest=manifest,
        shards=[shard],
    )
    validation_path = audit_path.with_suffix(".validation_seeds.json")
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    # Semantically identical seeds under changed bytes still violate the exact
    # sidecar file identity bound by the immutable audit.
    validation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        SystemExit, match="post-wave audit validation manifest binding mismatch"
    ):
        build_memmap_corpus(
            source,
            tmp_path / "corpus",
            selected_game_seed_manifest=manifest_path,
            a1_post_wave_audit=audit_path,
            progress_every=0,
        )


def test_selected_games_reject_duplicate_run_separated_by_validation_game(tmp_path: Path):
    manifest_path = tmp_path / "selected.json"
    manifest = _write_selected_manifest(manifest_path)
    seeds = np.asarray(
        [record["game_seed"] for record in manifest["records"]], dtype=np.int64
    )
    source = tmp_path / "teacher"
    source.mkdir()
    # A validation seed ends shard 0 between two occurrences of seed[0]. The
    # split label never licenses a second raw-source run of a selected game.
    _write_shard(source / "shard0.npz", [seeds[0], seeds[-1]])
    _write_shard(
        source / "shard1.npz", np.concatenate(([seeds[0]], seeds[1:-1]))
    )
    audit_path = tmp_path / "audit.json"
    _write_audit(
        audit_path,
        manifest_path=manifest_path,
        manifest=manifest,
        shards=[source / "shard0.npz", source / "shard1.npz"],
    )

    with pytest.raises(SystemExit, match="more than one non-contiguous raw-source run"):
        build_memmap_corpus(
            source,
            tmp_path / "corpus",
            selected_game_seed_manifest=manifest_path,
            a1_post_wave_audit=audit_path,
            progress_every=0,
        )


def test_selected_games_reject_cleanly_rehashed_failed_audit(tmp_path: Path):
    manifest_path = tmp_path / "selected.json"
    manifest = _write_selected_manifest(manifest_path)
    source = tmp_path / "teacher"
    source.mkdir()
    seeds = [record["game_seed"] for record in manifest["records"]]
    shard = source / "shard0.npz"
    _write_shard(shard, seeds)
    audit_path = tmp_path / "audit.json"
    audit = _write_audit(
        audit_path,
        manifest_path=manifest_path,
        manifest=manifest,
        shards=[shard],
    )
    audit["passed"] = False
    audit["errors"] = ["synthetic audit failure"]
    audit["audit_sha256"] = _value_sha256(
        {key: value for key, value in audit.items() if key != "audit_sha256"}
    )
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(SystemExit, match="not a clean passing report"):
        build_memmap_corpus(
            source,
            tmp_path / "corpus",
            selected_game_seed_manifest=manifest_path,
            a1_post_wave_audit=audit_path,
            progress_every=0,
        )
