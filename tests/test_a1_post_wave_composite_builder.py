from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import a1_build_post_wave_composite as builder
from tools import a1_one_dose_train
from tools import a1_pre_wave_contract as contract


def _lock(tmp_path: Path) -> dict:
    producer = tmp_path / "producer.pt"
    recent = tmp_path / "recent.pt"
    hard = tmp_path / "hard.pt"
    producer.write_bytes(b"producer")
    recent.write_bytes(b"recent")
    hard.write_bytes(b"hard")
    checkpoints = [
        {
            "id": "producer",
            "role": "producer",
            "path": str(producer),
            "sha256": builder._file_sha256(producer),  # noqa: SLF001
            "version": 7,
        },
        {
            "id": "recent",
            "role": "history",
            "path": str(recent),
            "sha256": builder._file_sha256(recent),  # noqa: SLF001
            "version": 6,
        },
        {
            "id": "hard",
            "role": "hard_negative",
            "path": str(hard),
            "sha256": builder._file_sha256(hard),  # noqa: SLF001
            "version": 5,
        },
    ]
    return {
        "contract_sha256": "sha256:" + "a" * 64,
        "checkpoints": checkpoints,
        "source_categories": [
            {"name": "current_producer", "checkpoint_ids": []},
            {"name": "recent_history", "checkpoint_ids": ["recent"]},
            {"name": "hard_negative", "checkpoint_ids": ["hard"]},
        ],
        "generation": {
            "track": "2p_no_trade",
            "vps_to_win": 10,
            "obs_width": 806,
            "max_decisions": 600,
            "temperature_decisions": 90,
            "temperature_high": 1.0,
            "temperature_low": 0.0,
            "late_temperature_decisions": None,
            "late_temperature": 0.0,
        },
        "science": {"search_operator": {"correct_rust_chance_spectra": True}},
        "fleet": {
            "jobs": [
                {
                    "job_id": "c_gpu0__current_producer",
                    "worker_id": "c_gpu0",
                    "category": "current_producer",
                    "base_seed": 100,
                    "seed_end": 102,
                },
                {
                    "job_id": "c_gpu0__recent_history",
                    "worker_id": "c_gpu0",
                    "category": "recent_history",
                    "base_seed": 200,
                    "seed_end": 202,
                },
                {
                    "job_id": "c_gpu0__hard_negative",
                    "worker_id": "c_gpu0",
                    "category": "hard_negative",
                    "base_seed": 300,
                    "seed_end": 302,
                },
            ]
        },
    }


def _selection(lock: dict) -> dict:
    producer = contract._producer(lock)  # noqa: SLF001
    opponent = {
        category["name"]: contract._category_opponent_sha256(  # noqa: SLF001
            lock, category["name"]
        )
        for category in lock["source_categories"]
    }
    return {
        "records": [
            {
                "game_seed": int(job["base_seed"]),
                "job_id": job["job_id"],
                "worker_id": job["worker_id"],
                "category": job["category"],
                "producer_checkpoint_sha256": producer["sha256"],
                "opponent_checkpoint_sha256": opponent[job["category"]],
                "split": "train",
            }
            for job in lock["fleet"]["jobs"]
        ]
    }


def _write_source(path: Path, *, base_seed: int, version: int | None) -> None:
    seeds = np.asarray([base_seed, base_seed + 1], dtype=np.int64)
    arrays: dict[str, np.ndarray] = {
        "game_seed": seeds,
        "terminated": np.ones(2, dtype=bool),
        "truncated": np.zeros(2, dtype=bool),
        "policy_weight_multiplier": np.asarray([1.0, 0.0], dtype=np.float32),
    }
    if version is not None:
        players = np.asarray(
            [
                "RED"
                if contract._pool_champion_plays_first_seat(index)  # noqa: SLF001
                else "BLUE"
                for index in range(2)
            ],
            dtype="U8",
        )
        arrays.update(
            {
                "is_pool_game": np.ones(2, dtype=bool),
                "opponent_version": np.full(2, version, dtype=np.int32),
                "player": players,
                "seat": np.asarray(
                    [contract.PLAYER_NAMES.index(str(value)) for value in players],
                    dtype=np.int8,
                ),
            }
        )
    np.savez(path, **arrays)


def _write_training_source(
    path: Path,
    *,
    base_seed: int,
    game_count: int,
    version: int | None,
) -> None:
    """Write two decisions per game plus the columns needed by real memmap ingest."""

    seeds = np.repeat(np.arange(base_seed, base_seed + game_count, dtype=np.int64), 2)
    rows = int(seeds.size)
    arrays: dict[str, np.ndarray] = {
        "obs": np.zeros((rows, 3), dtype=np.float16),
        "legal_action_ids": np.tile(np.asarray([[1, 2]], dtype=np.int16), (rows, 1)),
        "legal_action_context": np.zeros((rows, 2, 1), dtype=np.float16),
        "action_taken": np.tile(np.asarray([1, 2], dtype=np.int16), game_count),
        "decision_index": np.tile(np.asarray([0, 1], dtype=np.int32), game_count),
        "game_seed": seeds,
        "terminated": np.ones(rows, dtype=bool),
        "truncated": np.zeros(rows, dtype=bool),
        "policy_weight_multiplier": np.tile(
            np.asarray([1.0, 0.0], dtype=np.float32), game_count
        ),
    }
    if version is not None:
        players = np.asarray(
            [
                (
                    "RED"
                    if contract._pool_champion_plays_first_seat(  # noqa: SLF001
                        int(seed) - base_seed
                    )
                    else "BLUE"
                )
                for seed in seeds
            ],
            dtype="U8",
        )
        arrays.update(
            {
                "is_pool_game": np.ones(rows, dtype=bool),
                "opponent_version": np.full(rows, version, dtype=np.int32),
                "player": players,
                "seat": np.asarray(
                    [contract.PLAYER_NAMES.index(str(value)) for value in players],
                    dtype=np.int8,
                ),
            }
        )
    np.savez(path, **arrays)


def _historical_reference(
    tmp_path: Path,
    *,
    current_version: int,
    base_seed: int,
) -> Path:
    checkpoint = tmp_path / "historical.pt"
    checkpoint.write_bytes(b"historical")
    checkpoint_sha = builder._file_sha256(checkpoint)  # noqa: SLF001
    shard = tmp_path / "historical.npz"
    _write_training_source(
        shard,
        base_seed=base_seed,
        game_count=2,
        version=None,
    )
    source = tmp_path / "historical-source"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps({"shards": [str(shard.resolve())]}), encoding="utf-8"
    )
    corpus_dir = tmp_path / "historical-corpus"
    meta = builder.memmap_builder.build_memmap_corpus(
        source, corpus_dir, progress_every=0
    )
    mass = builder.measure_memmap_component(corpus_dir, meta)
    shard_record = {
        "path": str(shard.resolve()),
        "rows": 4,
        "order": 0,
        "size_bytes": shard.stat().st_size,
        "sha256": builder._file_sha256(shard),  # noqa: SLF001
        "checkpoint_version": current_version - 1,
        "producer_checkpoint_path": str(checkpoint.resolve()),
        "producer_checkpoint_sha256": checkpoint_sha,
        "source_id": "historical-generation",
        "source_category": "current_producer",
    }
    provenance = {
        "schema_version": "flywheel-replay-component-v1",
        "component_id": "historical_replay",
        "source_category": "historical_replay",
        "role": "replay",
        "current_checkpoint_version": current_version,
        "checkpoint_versions": [current_version - 1],
        "producer_checkpoints": [
            {
                "version": current_version - 1,
                "path": str(checkpoint.resolve()),
                "sha256": checkpoint_sha,
            }
        ],
        "row_count": 4,
        "shards": [shard_record],
        "shard_inventory_sha256": builder.canonical_sha256([shard_record]),
        "component_mass": mass,
    }
    provenance_path = tmp_path / "historical.provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    meta_path = corpus_dir / "corpus_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["flywheel_component_provenance"] = {
        "path": str(provenance_path.resolve()),
        "file_sha256": builder._file_sha256(provenance_path),  # noqa: SLF001
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    component = {
        "component_id": "historical_replay",
        "source_category": "historical_replay",
        "game_sampling_ratio": 0.20,
        "corpus_dir": str(corpus_dir.resolve()),
        "corpus_meta_sha256": builder._file_sha256(meta_path),  # noqa: SLF001
        "payload_inventory_sha256": meta["payload_inventory_sha256"],
        "provenance_manifest": str(provenance_path.resolve()),
        "provenance_manifest_sha256": builder._file_sha256(  # noqa: SLF001
            provenance_path
        ),
        "component_mass": mass,
    }
    reference = tmp_path / "historical.component.json"
    reference.write_text(
        json.dumps(
            {
                "schema_version": builder.HISTORICAL_COMPONENT_REF_SCHEMA,
                "component": component,
            }
        ),
        encoding="utf-8",
    )
    return reference


def _audit_fixture(tmp_path: Path, lock: dict, data_shards: list[dict]) -> dict:
    raw_records = []
    for job in lock["fleet"]["jobs"]:
        manifest = tmp_path / f"manifest-{job['category']}.json"
        manifest.write_text(
            json.dumps(
                {
                    "public_award_feature_provenance": {
                        "schema_version": "public-award-feature-provenance-v1",
                        "contract": "authoritative_v1",
                        "feature_producer": "python_snapshot_public_award_v1",
                        "native_capability": None,
                    }
                }
            )
        )
        raw_records.append(
            {
                "kind": "generation_manifest",
                "path": str(manifest),
                "sha256": builder._file_sha256(manifest),  # noqa: SLF001
                "job_id": job["job_id"],
                "category": job["category"],
            }
        )
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps({"shards": raw_records}))
    return {
        "path": audit_path,
        "file_sha256": "sha256:" + "d" * 64,
        "audit_sha256": "sha256:" + "e" * 64,
        "shard_inventory_sha256": "sha256:" + "f" * 64,
        "data_shards": data_shards,
    }


def test_filter_wave_shards_binds_job_category_seed_before_expansion(
    tmp_path: Path,
) -> None:
    lock = _lock(tmp_path)
    selected = _selection(lock)
    data_shards = []
    versions = {"current_producer": None, "recent_history": 6, "hard_negative": 5}
    for job in lock["fleet"]["jobs"]:
        source = tmp_path / f"{job['category']}.npz"
        _write_source(
            source,
            base_seed=int(job["base_seed"]),
            version=versions[job["category"]],
        )
        data_shards.append(
            {
                "path": str(source),
                "sha256": builder._file_sha256(source),  # noqa: SLF001
                "job_id": job["job_id"],
                "category": job["category"],
            }
        )
    output = tmp_path / "out"
    output.mkdir()
    selected_binding = {
        "file_sha256": "sha256:" + "b" * 64,
        "records_sha256": "sha256:" + "c" * 64,
    }
    audit = _audit_fixture(tmp_path, lock, data_shards)

    filtered, bindings = builder._filter_wave_shards(  # noqa: SLF001
        lock=lock,
        selected=selected_binding,
        audit=audit,
        raw_selected=selected,
        output_root=output,
        expected_games={
            "current_producer": 1,
            "recent_history": 1,
            "hard_negative": 1,
        },
    )

    assert {
        category: sum(row["rows"] for row in rows)
        for category, rows in filtered.items()
    } == {
        "current_producer": 1,
        "recent_history": 1,
        "hard_negative": 1,
    }
    assert len(bindings) == 3
    for category, records in filtered.items():
        with np.load(records[0]["path"], allow_pickle=False) as payload:
            assert payload["game_seed"].tolist() == [
                next(
                    job["base_seed"]
                    for job in lock["fleet"]["jobs"]
                    if job["category"] == category
                )
            ]
        manifest = json.loads(
            (output / "filtered_sources" / category / "manifest.json").read_text()
        )
        assert manifest["shards"] == [records[0]["path"]]


def test_filter_wave_shards_rejects_selected_seed_in_wrong_job(tmp_path: Path) -> None:
    lock = _lock(tmp_path)
    selected = _selection(lock)
    current_seed = selected["records"][0]["game_seed"]
    sources = []
    for job in lock["fleet"]["jobs"]:
        source = tmp_path / f"wrong-{job['category']}.npz"
        base = (
            current_seed
            if job["category"] == "recent_history"
            else int(job["base_seed"])
        )
        _write_source(
            source,
            base_seed=base,
            version={"current_producer": None, "recent_history": 6, "hard_negative": 5}[
                job["category"]
            ],
        )
        sources.append(
            {
                "path": str(source),
                "sha256": builder._file_sha256(source),  # noqa: SLF001
                "job_id": job["job_id"],
                "category": job["category"],
            }
        )
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(builder.CompositeBuildError, match="wrong audited job/category"):
        builder._filter_wave_shards(  # noqa: SLF001
            lock=lock,
            selected={
                "file_sha256": "sha256:" + "b" * 64,
                "records_sha256": "sha256:" + "c" * 64,
            },
            audit=_audit_fixture(tmp_path, lock, sources),
            raw_selected=selected,
            output_root=output,
            expected_games={
                "current_producer": 1,
                "recent_history": 1,
                "hard_negative": 1,
            },
        )


def test_descriptor_preserves_nested_fresh_mix_and_historical_replay(
    tmp_path: Path,
) -> None:
    components = []
    for category, ratio in builder.EFFECTIVE_COMPONENT_RATIOS.items():
        provenance = tmp_path / f"{category}.json"
        provenance.write_text(
            json.dumps(
                {"checkpoint_versions": [6] if category == "historical_replay" else [7]}
            )
        )
        components.append(
            {
                "component_id": category,
                "source_category": category,
                "game_sampling_ratio": ratio,
                "provenance_manifest": str(provenance),
                "provenance_manifest_sha256": builder._file_sha256(provenance),  # noqa: SLF001
                "component_mass": {
                    "game_count": 1,
                    "selected_game_count": 1,
                    "training_game_count": 1,
                    "validation_game_count": 0,
                    "row_count": 2,
                    "policy_active_row_count": 1,
                    "policy_weight_multiplier_sum": 1.0,
                    "mean_game_policy_active_fraction": 0.5,
                    "mean_game_policy_weight_multiplier": 0.5,
                },
            }
        )
    producer = tmp_path / "producer.pt"
    producer.write_bytes(b"producer")
    descriptor = builder._build_descriptor(  # noqa: SLF001
        components=components,
        producer_path=producer,
        producer_sha256=builder._file_sha256(producer),  # noqa: SLF001
        current_version=7,
    )
    replay = descriptor["flywheel_replay_contract"]
    assert replay["fresh_source_game_ratios"] == {
        "current_producer": 0.8,
        "recent_history": 0.15,
        "hard_negative": 0.05,
    }
    assert replay["effective_component_sampling_ratios"] == {
        "current_producer": 0.64,
        "recent_history": 0.12,
        "hard_negative": 0.04,
        "historical_replay": 0.20,
    }
    assert replay["checkpoint_versions"] == [6, 7]


def test_real_memmap_composite_is_accepted_by_one_dose_trainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise audited filtering through the exact production trainer preflight."""

    lock = _lock(tmp_path)
    for job in lock["fleet"]["jobs"]:
        job["seed_end"] = int(job["base_seed"]) + 3
    producer = contract._producer(lock)  # noqa: SLF001
    opponent = {
        category["name"]: contract._category_opponent_sha256(  # noqa: SLF001
            lock, category["name"]
        )
        for category in lock["source_categories"]
    }
    raw_selected = {
        "records": [
            {
                "game_seed": int(job["base_seed"]) + offset,
                "job_id": job["job_id"],
                "worker_id": job["worker_id"],
                "category": job["category"],
                "producer_checkpoint_sha256": producer["sha256"],
                "opponent_checkpoint_sha256": opponent[job["category"]],
                "split": "train" if offset == 0 else "validation",
            }
            for job in lock["fleet"]["jobs"]
            for offset in (0, 1)
        ]
    }
    versions = {"current_producer": None, "recent_history": 6, "hard_negative": 5}
    data_shards = []
    for job in lock["fleet"]["jobs"]:
        source = tmp_path / f"training-{job['category']}.npz"
        _write_training_source(
            source,
            base_seed=int(job["base_seed"]),
            game_count=3,
            version=versions[job["category"]],
        )
        data_shards.append(
            {
                "path": str(source.resolve()),
                "sha256": builder._file_sha256(source),  # noqa: SLF001
                "job_id": job["job_id"],
                "category": job["category"],
            }
        )
    expected_games = {
        "current_producer": 2,
        "recent_history": 2,
        "hard_negative": 2,
    }
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps(raw_selected), encoding="utf-8")
    selected = {
        "path": selected_path.resolve(),
        "file_sha256": builder._file_sha256(selected_path),  # noqa: SLF001
        "records_sha256": "sha256:" + "c" * 64,
        "a1_contract_sha256": lock["contract_sha256"],
    }
    audit = _audit_fixture(tmp_path, lock, data_shards)
    historical_ref = _historical_reference(
        tmp_path,
        current_version=int(producer["version"]),
        base_seed=400,
    )
    lock_path = tmp_path / "contract.lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr(
        builder,
        "_validated_wave_inputs",
        lambda *_args, **_kwargs: (lock, selected, audit, raw_selected),
    )
    output = tmp_path / "built"
    receipt = builder.build_post_wave_composite(
        lock_path=lock_path,
        selected_path=selected_path,
        audit_path=Path(str(audit["path"])),
        historical_component_path=historical_ref,
        output_root=output,
        expected_games=expected_games,
    )
    descriptor_path = Path(str(receipt["descriptor"]["path"]))
    meta = builder.train_bc._preflight_memmap_composite_descriptor(  # noqa: SLF001
        descriptor_path
    )
    verified = a1_one_dose_train._verify_production_composite_inputs(  # noqa: SLF001
        lock=lock,
        lock_path=lock_path,
        reviewed_lock_file_sha256=None,
        recipe=dict(contract.CURRENT_LEARNER_TRAINING_RECIPE),
        objective={"type": "mse"},
        producer=producer,
        data_path=descriptor_path,
        meta=meta,
        validation_path=None,
    )

    assert verified["data_kind"] == "production_composite_v2"
    assert (
        verified["production_mix_contract"]["effective_component_sampling_ratios"]
        == builder.EFFECTIVE_COMPONENT_RATIOS
    )
    assert verified["corpus_row_count"] == 16
    assert verified["training_row_count"] + verified["validation_row_count"] == 16
    assert verified["validation_split_receipt"]["aggregate"]["selected_game_count"] == 8
