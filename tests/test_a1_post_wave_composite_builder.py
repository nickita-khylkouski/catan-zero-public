from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import a1_build_post_wave_composite as builder
from tools import a1_one_dose_train
from tools import a1_pre_wave_contract as contract
from tools import a1_seal_historical_replay_component as historical_sealer
from catan_zero.rl.entity_feature_adapter import CURRENT_RUST_ENTITY_ADAPTER_VERSION


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
        "adapter_version": np.full(
            rows, CURRENT_RUST_ENTITY_ADAPTER_VERSION, dtype="U64"
        ),
        "aux_longest_road": np.zeros(rows, dtype=np.float32),
        "aux_largest_army": np.zeros(rows, dtype=np.float32),
        "aux_vp_in_n": np.ones(rows, dtype=np.float32),
        "aux_next_settlement": np.full(rows, 5, dtype=np.int16),
        "aux_robber_target": np.full(rows, 2, dtype=np.int16),
        builder.AUX_SUBGOAL_TARGET_VERSION_KEY: np.full(
            rows, builder.AUX_SUBGOAL_TARGET_VERSION, dtype=np.uint8
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
    monkeypatch: pytest.MonkeyPatch,
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
    meta_path = corpus_dir / "corpus_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    prior_lock = {
        "checkpoints": [
            {
                "id": "producer",
                "role": "producer",
                "path": str(checkpoint.resolve()),
                "sha256": checkpoint_sha,
                "version": current_version - 1,
            }
        ],
        "source_categories": [
            {"name": "current_producer", "checkpoint_ids": []}
        ],
        "generation": {},
        "science": {},
        "fleet": {
            "jobs": [
                {
                    "job_id": "historical_gpu0__current_producer",
                    "worker_id": "historical_gpu0",
                    "category": "current_producer",
                    "base_seed": base_seed,
                    "seed_end": base_seed + 2,
                }
            ]
        },
    }
    prior_lock["contract_sha256"] = builder._digest(prior_lock)  # noqa: SLF001
    prior_lock_path = tmp_path / "historical.contract.lock.json"
    prior_lock_path.write_text(json.dumps(prior_lock), encoding="utf-8")
    selected_payload = {
        "a1_contract_sha256": prior_lock["contract_sha256"],
        "category_game_counts": {"current_producer": 2},
        "selected_game_seed_set_sha256": builder._digest([base_seed, base_seed + 1]),  # noqa: SLF001
        "records": [
            {
                "game_seed": base_seed + offset,
                "job_id": "historical_gpu0__current_producer",
                "worker_id": "historical_gpu0",
                "category": "current_producer",
                "producer_checkpoint_sha256": checkpoint_sha,
                "opponent_checkpoint_sha256": [checkpoint_sha],
                "split": "train" if offset == 0 else "validation",
            }
            for offset in range(2)
        ],
    }
    selected_payload["records_sha256"] = builder._digest(selected_payload["records"])  # noqa: SLF001
    selected_path = tmp_path / "historical.selected.json"
    selected_path.write_text(json.dumps(selected_payload), encoding="utf-8")
    generation_manifest = tmp_path / "historical.generation.json"
    generation_manifest.write_text(json.dumps({"generation": "historical"}))
    audit_shards = [
        {
            "kind": "generation_manifest",
            "path": str(generation_manifest.resolve()),
            "sha256": builder._file_sha256(generation_manifest),  # noqa: SLF001
            "job_id": "historical_gpu0__current_producer",
            "category": "current_producer",
        },
        {
            "kind": "data_shard",
            "path": str(shard.resolve()),
            "sha256": builder._file_sha256(shard),  # noqa: SLF001
            "job_id": "historical_gpu0__current_producer",
            "category": "current_producer",
        },
    ]
    audit_payload = {
        "contract_sha256": prior_lock["contract_sha256"],
        "shard_inventory_sha256": builder._digest(audit_shards),  # noqa: SLF001
        "shards": audit_shards,
    }
    audit_payload["audit_sha256"] = builder._digest(audit_payload)  # noqa: SLF001
    audit_path = tmp_path / "historical.audit.json"
    audit_path.write_text(json.dumps(audit_payload), encoding="utf-8")
    selected = {
        "path": selected_path.resolve(),
        "file_sha256": builder._file_sha256(selected_path),  # noqa: SLF001
        "manifest_sha256": builder._digest(selected_payload),  # noqa: SLF001
        "records_sha256": selected_payload["records_sha256"],
        "selected_game_seed_set_sha256": selected_payload[
            "selected_game_seed_set_sha256"
        ],
        "a1_contract_sha256": prior_lock["contract_sha256"],
    }
    audit = {
        "path": audit_path.resolve(),
        "file_sha256": builder._file_sha256(audit_path),  # noqa: SLF001
        "audit_sha256": audit_payload["audit_sha256"],
        "shard_inventory_sha256": audit_payload["shard_inventory_sha256"],
        "contract_sha256": prior_lock["contract_sha256"],
        "data_shards": [audit_shards[1]],
    }
    meta["selected_game_seed_manifest"] = {
        "file_sha256": selected["file_sha256"]
    }
    meta["a1_post_wave_audit"] = {"file_sha256": audit["file_sha256"]}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    corpus_hash_before = builder._file_sha256(meta_path)  # noqa: SLF001
    monkeypatch.setattr(historical_sealer.contract, "verify_lock", lambda *_a, **_k: prior_lock)
    monkeypatch.setattr(
        historical_sealer.memmap_builder,
        "_load_a1_selected_game_manifest",
        lambda *_a, **_k: selected,
    )
    monkeypatch.setattr(
        historical_sealer.memmap_builder,
        "_load_a1_post_wave_audit",
        lambda *_a, **_k: audit,
    )
    reference = tmp_path / "historical.component.json"
    historical_sealer.seal_historical_replay_component(
        lock_path=prior_lock_path,
        selected_path=selected_path,
        audit_path=audit_path,
        corpus_dir=corpus_dir,
        producer_version=current_version - 1,
        current_version=current_version,
        output_path=reference,
    )
    assert builder._file_sha256(meta_path) == corpus_hash_before  # noqa: SLF001
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
    raw_records.extend(
        {
            "kind": "data_shard",
            "path": str(Path(record["path"]).resolve()),
            "sha256": record["sha256"],
            "job_id": record["job_id"],
            "category": record["category"],
        }
        for record in data_shards
    )
    raw_audit = {
        "contract_sha256": lock["contract_sha256"],
        "shard_inventory_sha256": builder._digest(raw_records),  # noqa: SLF001
        "shards": raw_records,
    }
    raw_audit["audit_sha256"] = builder._digest(raw_audit)  # noqa: SLF001
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(raw_audit))
    return {
        "path": audit_path,
        "file_sha256": builder._file_sha256(audit_path),  # noqa: SLF001
        "audit_sha256": raw_audit["audit_sha256"],
        "shard_inventory_sha256": raw_audit["shard_inventory_sha256"],
        "contract_sha256": lock["contract_sha256"],
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
        corpus_dir = tmp_path / f"{category}.corpus"
        corpus_dir.mkdir()
        corpus_meta = {
            "row_count": 2,
            "schema": "memmap_corpus_v1",
            "legal_width": 2,
            "columns": {
                "adapter_version": {
                    "kind": "string",
                    "categories": [CURRENT_RUST_ENTITY_ADAPTER_VERSION],
                }
            },
        }
        if category != builder.HISTORICAL_REPLAY_CATEGORY:
            corpus_meta["aux_subgoal_target_contract"] = {
                "version_key": "aux_subgoal_target_version",
                "supported_version": 1,
                "semantic": "strict_future_after_current_row_v1",
                "version_zero_means_unversioned_ineligible": True,
                "realized_version_counts": {"1": 2},
                "all_rows_semantically_eligible": True,
            }
        (corpus_dir / "corpus_meta.json").write_text(json.dumps(corpus_meta))
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
                "corpus_dir": str(corpus_dir),
                **(
                    {
                        "entity_feature_adapter_version": (
                            CURRENT_RUST_ENTITY_ADAPTER_VERSION
                        )
                    }
                    if category == builder.HISTORICAL_REPLAY_CATEGORY
                    else {}
                ),
            }
        )
    producer = tmp_path / "producer.pt"
    producer.write_bytes(b"producer")
    authority = tmp_path / "authority.json"
    authority.write_text("{}", encoding="utf-8")
    descriptor = builder._build_descriptor(  # noqa: SLF001
        components=components,
        producer_path=producer,
        producer_sha256=builder._file_sha256(producer),  # noqa: SLF001
        current_version=7,
        source_authority={
            "path": str(authority.resolve()),
            "file_sha256": builder._file_sha256(authority),  # noqa: SLF001
            "authority_sha256": builder._digest({}),  # noqa: SLF001
        },
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
    assert descriptor["aux_subgoal_component_ids"] == [
        "current_producer",
        "recent_history",
        "hard_negative",
    ]
    assert descriptor["stored_policy_component_temperatures"] == {
        "current_producer": 1.0,
        "recent_history": 1.0,
        "hard_negative": 1.0,
        "historical_replay": 0.52,
    }
    assert descriptor["entity_feature_adapter_component_versions"] == {
        component_id: CURRENT_RUST_ENTITY_ADAPTER_VERSION
        for component_id in (
            "current_producer",
            "recent_history",
            "hard_negative",
            "historical_replay",
        )
    }

    stale_meta_path = tmp_path / "recent_history.corpus" / "corpus_meta.json"
    stale_meta = json.loads(stale_meta_path.read_text())
    stale_meta["aux_subgoal_target_contract"]["realized_version_counts"] = {
        "0": 2
    }
    stale_meta["aux_subgoal_target_contract"][
        "all_rows_semantically_eligible"
    ] = False
    stale_meta_path.write_text(json.dumps(stale_meta))
    with pytest.raises(
        builder.CompositeBuildError,
        match="fresh component aux-subgoal target contract is not uniformly",
    ):
        builder._build_descriptor(  # noqa: SLF001
            components=components,
            producer_path=producer,
            producer_sha256=builder._file_sha256(producer),  # noqa: SLF001
            current_version=7,
            source_authority={
                "path": str(authority.resolve()),
                "file_sha256": builder._file_sha256(authority),  # noqa: SLF001
                "authority_sha256": builder._digest({}),  # noqa: SLF001
            },
        )


def test_real_memmap_composite_is_accepted_by_one_dose_trainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise audited filtering through the exact production trainer preflight."""

    lock = _lock(tmp_path)
    for job in lock["fleet"]["jobs"]:
        job["seed_end"] = int(job["base_seed"]) + 3
    expected_games = {
        "current_producer": 2,
        "recent_history": 2,
        "hard_negative": 2,
    }
    lock["game_contract"] = {
        "total_complete_games": sum(expected_games.values()),
        "category_games": dict(expected_games),
    }
    lock.pop("contract_sha256")
    lock["contract_sha256"] = builder._digest(lock)  # noqa: SLF001
    producer = contract._producer(lock)  # noqa: SLF001
    opponent = {
        category["name"]: contract._category_opponent_sha256(  # noqa: SLF001
            lock, category["name"]
        )
        for category in lock["source_categories"]
    }
    raw_selected = {
        "a1_contract_sha256": lock["contract_sha256"],
        "selected_game_seed_set_sha256": "sha256:" + "9" * 64,
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
    raw_selected["records_sha256"] = builder._digest(raw_selected["records"])  # noqa: SLF001
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
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps(raw_selected), encoding="utf-8")
    selected = {
        "path": selected_path.resolve(),
        "file_sha256": builder._file_sha256(selected_path),  # noqa: SLF001
        "manifest_sha256": builder._digest(raw_selected),  # noqa: SLF001
        "records_sha256": raw_selected["records_sha256"],
        "selected_game_seed_set_sha256": raw_selected[
            "selected_game_seed_set_sha256"
        ],
        "a1_contract_sha256": lock["contract_sha256"],
    }
    audit = _audit_fixture(tmp_path, lock, data_shards)
    historical_ref = _historical_reference(
        tmp_path,
        current_version=int(producer["version"]),
        base_seed=400,
        monkeypatch=monkeypatch,
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
    )
    descriptor_path = Path(str(receipt["descriptor"]["path"]))
    # A learner B200 receives the filtered composite and authority bundle, not
    # the multi-gigabyte raw generation tree.  Removing every original data
    # shard/manifest must not weaken or break preflight.
    for record in data_shards:
        Path(record["path"]).unlink()
    for record in json.loads(Path(audit["path"]).read_text())["shards"]:
        if record.get("kind") == "generation_manifest":
            Path(record["path"]).unlink()
    (tmp_path / "historical.npz").unlink()
    (tmp_path / "historical.generation.json").unlink()
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
        build_receipt_path=output / "build_receipt.json",
    )

    assert verified["data_kind"] == "production_composite_v2"
    assert (
        verified["production_mix_contract"]["effective_component_sampling_ratios"]
        == builder.EFFECTIVE_COMPONENT_RATIOS
    )
    assert verified["corpus_row_count"] == 16
    assert verified["training_row_count"] + verified["validation_row_count"] == 16
    assert verified["validation_split_receipt"]["aggregate"]["selected_game_count"] == 8

    authority_path = Path(receipt["source_authority"]["path"])
    tampered = json.loads(authority_path.read_text(encoding="utf-8"))
    tampered["canonical_composite_root"] = str(tmp_path.resolve())
    binding = tampered["fresh_source_bindings"][0]
    binding["source_path"] = str(tmp_path / "never-audited.npz")
    preimage = dict(binding)
    preimage.pop("source_id")
    binding["source_id"] = builder._digest(preimage)  # noqa: SLF001
    tampered["fresh_source_bindings_sha256"] = builder._digest(  # noqa: SLF001
        tampered["fresh_source_bindings"]
    )
    tampered.pop("authority_sha256")
    tampered["authority_sha256"] = builder._digest(tampered)  # noqa: SLF001
    tampered_path = tmp_path / "tampered-source-authority.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(SystemExit, match="evidence drift"):
        builder.train_bc._validate_flywheel_source_authority(tampered_path)  # noqa: SLF001

    staged_selected = Path(
        json.loads(authority_path.read_text(encoding="utf-8"))[
            "selected_game_manifest"
        ]["path"]
    )
    original_selected = staged_selected.read_bytes()
    staged_selected.write_bytes(original_selected + b"\n")
    with pytest.raises(SystemExit, match="byte binding drift"):
        builder.train_bc._validate_flywheel_source_authority(authority_path)  # noqa: SLF001
    staged_selected.write_bytes(original_selected)

    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    fresh_provenance = json.loads(
        Path(descriptor["components"][0]["provenance_manifest"]).read_text()
    )
    Path(fresh_provenance["shards"][0]["path"]).unlink()
    with pytest.raises(SystemExit, match="fresh shard is missing"):
        builder.train_bc._preflight_memmap_composite_descriptor(descriptor_path)  # noqa: SLF001
