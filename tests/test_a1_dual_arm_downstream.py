from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import a1_dual_arm_subsets as subsets
from tools import build_memmap_corpus as corpus
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
        "records": records, "parent_manifest_sha256": SHA,
    }
    path = tmp_path / "full.json"
    path.write_text(json.dumps(value, sort_keys=True))
    return path


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
    first = subsets.build_subsets(source, tmp_path / "one")
    second = subsets.build_subsets(source, tmp_path / "two")
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
