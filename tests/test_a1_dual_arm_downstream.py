from __future__ import annotations

import json
import argparse
from pathlib import Path

import pytest
import numpy as np

from tools import a1_dual_arm_subsets as subsets
from tools import build_memmap_corpus as corpus
from tools import train_bc
from tools.fleet import a1_harvest_transaction as harvest

SHA = "sha256:" + "a" * 64


def _record(worker: int, category: str, ordinal: int, seed: int) -> dict:
    return {
        "arm_id": "n128", "game_seed": seed,
        "job_id": f"n128_gpu{worker:02d}__{category}",
        "worker_id": f"n128_gpu{worker:02d}", "category": category,
        "producer_checkpoint_sha256": SHA,
        "opponent_checkpoint_sha256": [SHA],
        "split": "validation" if ordinal == 0 else "train",
    }


def _full_manifest(tmp_path: Path) -> Path:
    per_category = {"current_producer": 5, "recent_history": 2, "hard_negative": 1}
    records = []
    seed = 1_000_000
    for worker in range(28):
        for category, count in per_category.items():
            for ordinal in range(count):
                records.append(_record(worker, category, ordinal, seed))
                seed += 1
    records.sort(key=lambda row: (row["game_seed"], row["job_id"]))
    training = [row["game_seed"] for row in records if row["split"] == "train"]
    validation = [row["game_seed"] for row in records if row["split"] == "validation"]
    value = {
        "schema_version": corpus.DUAL_ARM_SELECTED_GAMES_SCHEMA,
        "arm_id": "n128", "subset_id": "full-140k",
        "a1_contract_sha256": SHA, "selection_rule": "lowest_seed_complete_per_job",
        "selected_game_count": len(records),
        "selected_game_seed_set_sha256": corpus._game_seed_set_sha256([row["game_seed"] for row in records]),  # noqa: SLF001
        "category_game_counts": {key: value * 28 for key, value in per_category.items()},
        "training_game_count": len(training),
        "training_game_seed_set_sha256": corpus._game_seed_set_sha256(training),  # noqa: SLF001
        "validation_game_count": len(validation),
        "validation_game_seed_set_sha256": corpus._game_seed_set_sha256(validation),  # noqa: SLF001
        "records_sha256": corpus._value_sha256(records),  # noqa: SLF001
        "records": records, "parent_manifest_sha256": None,
    }
    path = tmp_path / "full.json"
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def _full_audit(tmp_path: Path, manifest: Path) -> Path:
    selected = json.loads(manifest.read_text())
    jobs = tmp_path / "jobs"
    jobs.mkdir(exist_ok=True)
    shard = jobs / "shard.npz"
    seeds = np.asarray(
        [row["game_seed"] for row in selected["records"]], dtype=np.int64
    )
    n = len(seeds)
    np.savez(
        shard,
        game_seed=seeds,
        obs=np.zeros((n, 1), dtype=np.float16),
        legal_action_ids=np.zeros((n, 1), dtype=np.int16),
        legal_action_context=np.zeros((n, 1, 1), dtype=np.float16),
        action_taken=np.zeros(n, dtype=np.int16),
        target_policy=np.ones((n, 1), dtype=np.float32),
        target_policy_mask=np.ones((n, 1), dtype=bool),
    )
    shard_file_sha = corpus._file_sha256(shard)  # noqa: SLF001
    validation_seeds = sorted(
        row["game_seed"] for row in selected["records"] if row["split"] == "validation"
    )
    validation = {
        "schema_version": "train-validation-game-seeds-v1",
        "a1_contract_sha256": SHA,
        "validation_fraction": 0.05,
        "validation_seed": 17,
        "validation_max_samples": 0,
        "validation_game_seed_ranges": [],
        "validation_game_seed_count": len(validation_seeds),
        "validation_row_count": len(validation_seeds),
        "validation_game_seed_set_sha256": corpus._game_seed_set_sha256(validation_seeds),  # noqa: SLF001
        "game_seeds": validation_seeds,
    }
    validation_path = tmp_path / "full.validation_seeds.json"
    validation_path.write_text(json.dumps(validation, sort_keys=True))
    files = [{
        "relative_path": "jobs/shard.npz",
        "sha256": shard_file_sha,
        "size_bytes": shard.stat().st_size,
    }]
    relocation = {
        "schema_version": "a1-fleet-harvest-relocation-v1",
        "arm_id": "n128",
        "contract_sha256": SHA,
        "render_sha256": SHA,
        "job_identities_sha256": SHA,
        "files": files,
        "file_inventory_sha256": corpus._value_sha256(files),  # noqa: SLF001
    }
    relocation["relocation_sha256"] = corpus._value_sha256(relocation)  # noqa: SLF001
    relocation_path = tmp_path / "relocation.json"
    relocation_path.write_text(json.dumps(relocation, sort_keys=True))
    shards = [{
        "kind": "data_shard",
        "path": str(shard.resolve()),
        "sha256": shard_file_sha,
        "job_id": "fixture",
        "category": "current_producer",
    }]
    audit = {
        "schema_version": corpus.DUAL_ARM_AUDIT_SCHEMA,
        "arm_id": "n128",
        "subset_id": "full-140k",
        "contract_path": str(tmp_path / "lock.json"),
        "contract_sha256": SHA,
        "passed": True,
        "errors": [],
        "category_game_counts": selected["category_game_counts"],
        "rows": len(selected["records"]),
        "shards": shards,
        "shard_inventory_sha256": corpus._value_sha256(shards),  # noqa: SLF001
        "source_provenance": {"current_producer": {"producer_checkpoint_sha256": SHA}},
        "harvest_relocation": {
            "arm_id": "n128",
            "path": str(relocation_path.resolve()),
            "file_sha256": corpus._file_sha256(relocation_path),  # noqa: SLF001
            "relocation_sha256": relocation["relocation_sha256"],
            "render_sha256": SHA,
            "job_identities_sha256": SHA,
            "file_inventory_sha256": relocation["file_inventory_sha256"],
        },
        "selected_training_games": {
            "manifest": str(manifest.resolve()),
            "manifest_sha256": corpus._value_sha256(selected),  # noqa: SLF001
            "manifest_file_sha256": corpus._file_sha256(manifest),  # noqa: SLF001
            "selected_game_count": selected["selected_game_count"],
            "selected_game_seed_set_sha256": selected["selected_game_seed_set_sha256"],
            "records_sha256": selected["records_sha256"],
        },
        "validation_holdout": {
            "manifest": str(validation_path.resolve()),
            "manifest_sha256": corpus._value_sha256(validation),  # noqa: SLF001
            "manifest_file_sha256": corpus._file_sha256(validation_path),  # noqa: SLF001
            "validation_game_seed_count": len(validation_seeds),
            "validation_game_seed_set_sha256": validation["validation_game_seed_set_sha256"],
        },
    }
    audit["audit_sha256"] = corpus._value_sha256(audit)  # noqa: SLF001
    audit_path = tmp_path / "full.audit.json"
    audit_path.write_text(json.dumps(audit, sort_keys=True))
    return audit_path


def test_dual_shape_preserves_historical_and_requires_exact_quotas() -> None:
    assert {"is_forced", "used_full_search"}.issubset(corpus.LOADER_KEYS)
    historical = harvest._contract_shape({})  # noqa: SLF001
    assert historical["job_count"] == 120 and historical["arm_id"] is None
    dual = harvest._contract_shape(  # noqa: SLF001
        {"game_contract": {
            "profile": harvest.DUAL_ARM_PROFILE, "arm_id": "n128",
            "category_games": {"current_producer": 112000, "recent_history": 21000, "hard_negative": 7000},
            "category_attempts": {"current_producer": 114240, "recent_history": 21420, "hard_negative": 7140},
            "total_complete_games": 140000, "total_attempts": 142800,
        }}
    )
    assert dual["job_count"] == 84 and dual["arm_id"] == "n128"
    bad = {"game_contract": {**{key: value for key, value in {
        "profile": harvest.DUAL_ARM_PROFILE, "arm_id": "n128",
        "category_games": dual["category_games"], "category_attempts": dual["category_attempts"],
        "total_complete_games": 1, "total_attempts": 142800}.items()}}}
    with pytest.raises(harvest.HarvestError, match="quotas"):
        harvest._contract_shape(bad)  # noqa: SLF001


def test_n256_full_56k_identity_is_exact_and_old_140k_label_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "current_producer": 44_800,
        "recent_history": 8_400,
        "hard_negative": 2_800,
    }
    assert corpus.DUAL_ARM_SUBSET_CATEGORY_COUNTS[("n256", "full-56k")] == expected
    assert ("n256", "full-140k") not in corpus.DUAL_ARM_SUBSET_CATEGORY_COUNTS

    payload = json.loads(_full_manifest(tmp_path).read_text())
    payload["arm_id"] = "n256"
    payload["subset_id"] = "full-56k"
    for record in payload["records"]:
        record["arm_id"] = "n256"
    payload["records_sha256"] = corpus._value_sha256(payload["records"])  # noqa: SLF001
    miniature_counts = payload["category_game_counts"]
    monkeypatch.setattr(
        corpus,
        "DUAL_ARM_SUBSET_CATEGORY_COUNTS",
        {("n256", "full-56k"): miniature_counts},
    )
    accepted = tmp_path / "n256-full-56k.json"
    accepted.write_text(json.dumps(payload, sort_keys=True))
    loaded = corpus._load_a1_selected_game_manifest(accepted)  # noqa: SLF001
    assert loaded["arm_id"] == "n256"
    assert loaded["subset_id"] == "full-56k"

    payload["subset_id"] = "full-140k"
    rejected = tmp_path / "n256-mislabeled-full-140k.json"
    rejected.write_text(json.dumps(payload, sort_keys=True))
    with pytest.raises(SystemExit, match="arm/subset category quotas"):
        corpus._load_a1_selected_game_manifest(rejected)  # noqa: SLF001


def test_n128_subsets_are_deterministic_stratified_and_arm_pure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _full_manifest(tmp_path)
    monkeypatch.setattr(subsets, "TARGETS", {
        "matched-56k": {"current_producer": 2, "recent_history": 1, "hard_negative": 1},
        "compute-112k": {"current_producer": 4, "recent_history": 2, "hard_negative": 1},
    })
    monkeypatch.setattr(corpus, "DUAL_ARM_SUBSET_CATEGORY_COUNTS", {
        ("n128", "full-140k"): {"current_producer": 140, "recent_history": 56, "hard_negative": 28},
        ("n128", "matched-56k"): {"current_producer": 56, "recent_history": 28, "hard_negative": 28},
        ("n128", "compute-112k"): {"current_producer": 112, "recent_history": 56, "hard_negative": 28},
    })
    parent_audit = _full_audit(tmp_path, source)
    first = subsets.build_subsets(source, parent_audit, tmp_path / "one")
    second = subsets.build_subsets(source, parent_audit, tmp_path / "two")
    for subset_id in first:
        assert first[subset_id].read_bytes() == second[subset_id].read_bytes()
        loaded = corpus._load_a1_selected_game_manifest(first[subset_id])  # noqa: SLF001
        assert loaded["arm_id"] == "n128" and loaded["subset_id"] == subset_id
    forged = json.loads(first["matched-56k"].read_text())
    forged["records"][0]["arm_id"] = "n256"
    forged_path = tmp_path / "forged.json"
    forged_path.write_text(json.dumps(forged))
    with pytest.raises(SystemExit, match="identity drift"):
        corpus._load_a1_selected_game_manifest(forged_path)  # noqa: SLF001
    forged = json.loads(first["matched-56k"].read_text())
    forged["arm_id"] = "n256"
    forged_path.write_text(json.dumps(forged))
    with pytest.raises(SystemExit, match="arm/subset category quotas"):
        corpus._load_a1_selected_game_manifest(forged_path)  # noqa: SLF001


@pytest.mark.parametrize("crash_after", range(1, 7))
def test_subset_publication_resumes_exactly_after_every_crash_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_after: int
) -> None:
    source = _full_manifest(tmp_path)
    monkeypatch.setattr(subsets, "TARGETS", {
        "matched-56k": {"current_producer": 2, "recent_history": 1, "hard_negative": 1},
        "compute-112k": {"current_producer": 4, "recent_history": 2, "hard_negative": 1},
    })
    monkeypatch.setattr(corpus, "DUAL_ARM_SUBSET_CATEGORY_COUNTS", {
        ("n128", "full-140k"): {"current_producer": 140, "recent_history": 56, "hard_negative": 28},
        ("n128", "matched-56k"): {"current_producer": 56, "recent_history": 28, "hard_negative": 28},
        ("n128", "compute-112k"): {"current_producer": 112, "recent_history": 56, "hard_negative": 28},
    })
    parent_audit = _full_audit(tmp_path, source)
    out = tmp_path / f"resumable-{crash_after}"
    original = subsets._write_immutable  # noqa: SLF001
    writes = 0

    def crash_after_publish(path: Path, value: dict) -> None:
        nonlocal writes
        original(path, value)
        writes += 1
        if writes == crash_after:
            raise RuntimeError("simulated subset crash")

    monkeypatch.setattr(subsets, "_write_immutable", crash_after_publish)
    with pytest.raises(RuntimeError, match="subset crash"):
        subsets.build_subsets(source, parent_audit, out)
    assert len(list(out.glob("*.json"))) == crash_after

    monkeypatch.setattr(subsets, "_write_immutable", original)
    outputs = subsets.build_subsets(source, parent_audit, out)
    assert set(outputs) == {"matched-56k", "compute-112k"}
    assert len(list(out.glob("*.json"))) == 6
    assert subsets.build_subsets(source, parent_audit, out) == outputs


def test_real_dual_subset_corpus_and_trainer_preflight_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _full_manifest(tmp_path)
    monkeypatch.setattr(subsets, "TARGETS", {
        "matched-56k": {"current_producer": 2, "recent_history": 1, "hard_negative": 1},
        "compute-112k": {"current_producer": 4, "recent_history": 2, "hard_negative": 1},
    })
    miniature_counts = {
        ("n128", "full-140k"): {"current_producer": 140, "recent_history": 56, "hard_negative": 28},
        ("n128", "matched-56k"): {"current_producer": 56, "recent_history": 28, "hard_negative": 28},
        ("n128", "compute-112k"): {"current_producer": 112, "recent_history": 56, "hard_negative": 28},
    }
    monkeypatch.setattr(corpus, "DUAL_ARM_SUBSET_CATEGORY_COUNTS", miniature_counts)
    parent_audit = _full_audit(tmp_path, source)
    outputs = subsets.build_subsets(source, parent_audit, tmp_path / "derived")
    selected = outputs["matched-56k"]
    audit = selected.with_name("n128-matched-56k.audit.json")
    validation = selected.with_name("n128-matched-56k.validation_seeds.json")

    corpus_dir = tmp_path / "corpus"
    meta = corpus.build_memmap_corpus(
        tmp_path / "jobs",
        corpus_dir,
        selected_game_seed_manifest=selected,
        a1_post_wave_audit=audit,
        progress_every=0,
    )
    monkeypatch.setattr(
        train_bc,
        "DUAL_ARM_SUBSET_COUNTS",
        {("n128", "matched-56k"): sum(miniature_counts[("n128", "matched-56k")].values())},
    )
    monkeypatch.setattr(
        train_bc,
        "DUAL_ARM_SUBSET_CATEGORY_COUNTS",
        {("n128", "matched-56k"): miniature_counts[("n128", "matched-56k")]},
    )
    validation_contract = train_bc._load_validation_game_seed_manifest_for_training(  # noqa: SLF001
        validation,
        validation_fraction=0.05,
        validation_seed=17,
        validation_max_samples=0,
        validation_game_seed_ranges=[],
    )
    train_bc._validate_a1_validation_manifest_corpus_binding(  # noqa: SLF001
        meta, validation_contract
    )
    seeds = np.fromfile(corpus_dir / "game_seed.dat", dtype=np.int64)
    bound = train_bc._validate_a1_corpus_artifacts_and_seeds(  # noqa: SLF001
        meta, validation_contract, seeds
    )
    assert bound["dual_arm"] is True
    assert (bound["arm_id"], bound["subset_id"]) == ("n128", "matched-56k")
    recipe = dict(bound["learner_training_recipe"])
    args = argparse.Namespace(
        **{
            key: value
            for key, value in recipe.items()
            if key not in {"world_size", "global_batch_size"}
        },
        value_head_type="mse",
        value_categorical_bins=0,
        value_hlgauss_sigma_ratio=0.75,
        init_checkpoint_sha256=bound["producer_checkpoint_sha256"],
    )
    train_bc._validate_a1_learner_objective(args, bound)  # noqa: SLF001
    ddp = {"world_size": 8, "rank": 0, "local_rank": 0, "enabled": True}
    assert train_bc._validate_a1_learner_training_recipe(  # noqa: SLF001
        args, ddp, bound
    ) == recipe

    forged_audit = json.loads(audit.read_text())
    forged_audit["rows"] += 1
    audit.chmod(0o644)
    audit.write_text(json.dumps(forged_audit))
    with pytest.raises(SystemExit, match="audit file SHA-256 drift"):
        train_bc._validate_a1_corpus_artifacts_and_seeds(  # noqa: SLF001
            meta, validation_contract, seeds
        )
