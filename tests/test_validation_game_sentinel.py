from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import train_bc
from tools import derive_validation_game_sentinel as sentinel
from tools.derive_validation_game_sentinel import select_whole_games_near_target


def _sha(byte: str) -> str:
    return "sha256:" + byte * 64


def _fixture(tmp_path: Path):
    full_seeds = np.asarray([10, 11, 20, 21], dtype=np.int64)
    contracts = [
        {
            "file_sha256": _sha("1"),
            "manifest_sha256": _sha("2"),
            "validation_game_seed_set_sha256": _sha("3"),
        },
        {
            "file_sha256": _sha("4"),
            "manifest_sha256": _sha("5"),
            "validation_game_seed_set_sha256": _sha("6"),
        },
    ]
    meta = {
        "descriptor_file_sha256": _sha("a"),
        "descriptor_fingerprint": _sha("b"),
    }
    full = {
        "path": tmp_path / "source.json",
        "file_sha256": _sha("c"),
        "manifest_sha256": _sha("d"),
        "a1_contract_sha256": _sha("e"),
        "validation_row_count": 12,
        "validation_game_seed_set_sha256": train_bc._game_seed_set_sha256(full_seeds),
        "game_seeds": full_seeds,
        "component_contracts": contracts,
        "diagnostic_only": True,
        "promotion_eligible": False,
    }
    selected = np.asarray([10, 20], dtype=np.int64)
    payload = {
        "schema_version": "train-validation-game-sentinel-v1",
        "source_composite_descriptor_file_sha256": meta["descriptor_file_sha256"],
        "source_composite_descriptor_fingerprint": meta["descriptor_fingerprint"],
        "source_validation_bindings": [
            {
                "component_index": index,
                "validation_manifest_file_sha256": contract["file_sha256"],
                "validation_manifest_sha256": contract["manifest_sha256"],
                "validation_game_seed_set_sha256": contract[
                    "validation_game_seed_set_sha256"
                ],
            }
            for index, contract in enumerate(contracts)
        ],
        "selection_seed": 7,
        "target_row_count": 6,
        "selected_row_count": 6,
        "selected_game_seed_count": 2,
        "selected_game_seed_set_sha256": train_bc._game_seed_set_sha256(selected),
        "excluded_game_seed_count": 4,
        "excluded_game_seed_set_sha256": full["validation_game_seed_set_sha256"],
        "game_seeds": selected.tolist(),
    }
    path = tmp_path / "sentinel.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path, payload, meta, full


def test_whole_game_selection_is_deterministic_and_near_target() -> None:
    counts = {10: 101, 11: 97, 12: 103, 13: 99, 14: 96}
    first = select_whole_games_near_target(counts, target_rows=300, selection_seed=19)
    second = select_whole_games_near_target(counts, target_rows=300, selection_seed=19)
    assert first == second
    seeds, rows = first
    assert rows == sum(counts[seed] for seed in seeds)
    assert abs(rows - 300) <= max(counts.values())


def test_derive_can_exclude_consumed_games_and_stratify_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta = {
        "descriptor_file_sha256": _sha("a"),
        "descriptor_fingerprint": _sha("b"),
        "component_ids": ["current", "predecessor_replay"],
    }
    full_seeds = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64)
    full = {
        "game_seeds": full_seeds,
        "validation_game_seed_set_sha256": train_bc._game_seed_set_sha256(full_seeds),
        "component_contracts": [
            {"file_sha256": _sha("1"), "manifest_sha256": _sha("2"),
             "validation_game_seed_set_sha256": _sha("3")},
            {"file_sha256": _sha("4"), "manifest_sha256": _sha("5"),
             "validation_game_seed_set_sha256": _sha("6")},
        ],
    }
    corpus = type("Corpus", (), {"corpora": [
        {"game_seed": np.asarray([1] * 4 + [2] * 4 + [3] * 4)},
        {"game_seed": np.asarray([4] * 4 + [5] * 4 + [6] * 4)},
    ]})()
    monkeypatch.setattr(sentinel.train_bc, "_preflight_memmap_composite_descriptor", lambda _p: meta)
    monkeypatch.setattr(sentinel.train_bc, "_load_composite_validation_contract", lambda *_a, **_k: full)
    monkeypatch.setattr(sentinel.train_bc, "load_teacher_data_memmap", lambda *_a, **_k: corpus)
    value = sentinel.derive(
        tmp_path / "descriptor.json",
        target_rows=8,
        selection_seed=7,
        validation_fraction=0.05,
        validation_seed=17,
        excluded_selection_game_seeds={1, 4},
        component_target_ratios={"current": 0.5, "predecessor_replay": 0.5},
    )
    assert not set(value["game_seeds"]) & {1, 4}
    assert len(set(value["game_seeds"]) & {2, 3}) == 1
    assert len(set(value["game_seeds"]) & {5, 6}) == 1


def test_stratified_derive_refuses_exclusion_that_exhausts_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta = {
        "descriptor_file_sha256": _sha("a"),
        "descriptor_fingerprint": _sha("b"),
        "component_ids": ["current", "predecessor_replay"],
    }
    full_seeds = np.asarray([1, 2], dtype=np.int64)
    full = {
        "game_seeds": full_seeds,
        "validation_game_seed_set_sha256": train_bc._game_seed_set_sha256(full_seeds),
        "component_contracts": [
            {"file_sha256": _sha("1"), "manifest_sha256": _sha("2"),
             "validation_game_seed_set_sha256": _sha("3")},
            {"file_sha256": _sha("4"), "manifest_sha256": _sha("5"),
             "validation_game_seed_set_sha256": _sha("6")},
        ],
    }
    corpus = type("Corpus", (), {"corpora": [
        {"game_seed": np.asarray([1, 1])},
        {"game_seed": np.asarray([2, 2])},
    ]})()
    monkeypatch.setattr(sentinel.train_bc, "_preflight_memmap_composite_descriptor", lambda _p: meta)
    monkeypatch.setattr(sentinel.train_bc, "_load_composite_validation_contract", lambda *_a, **_k: full)
    monkeypatch.setattr(sentinel.train_bc, "load_teacher_data_memmap", lambda *_a, **_k: corpus)
    with pytest.raises(SystemExit, match="predecessor_replay has no fresh validation games"):
        sentinel.derive(
            tmp_path / "descriptor.json",
            target_rows=2,
            selection_seed=7,
            validation_fraction=0.05,
            validation_seed=17,
            excluded_selection_game_seeds={2},
            component_target_ratios={"current": 0.8, "predecessor_replay": 0.2},
        )


def test_sentinel_authentication_retains_full_training_exclusion(tmp_path: Path) -> None:
    path, _payload, meta, full = _fixture(tmp_path)
    contract = train_bc._load_composite_validation_sentinel_manifest(
        path, composite_meta=meta, full_contract=full
    )
    np.testing.assert_array_equal(contract["game_seeds"], [10, 20])
    np.testing.assert_array_equal(contract["excluded_game_seeds"], [10, 11, 20, 21])
    assert contract["validation_row_count"] == 6
    assert contract["file_sha256"] == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_sentinel_refuses_source_binding_drift(tmp_path: Path) -> None:
    path, payload, meta, full = _fixture(tmp_path)
    payload["source_composite_descriptor_file_sha256"] = _sha("f")
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(SystemExit, match="source composite binding drift"):
        train_bc._load_composite_validation_sentinel_manifest(
            path, composite_meta=meta, full_contract=full
        )


def test_split_evaluates_sentinel_but_excludes_complete_holdout() -> None:
    data = {
        "action_taken": np.arange(12),
        "game_seed": np.repeat(np.asarray([1, 2, 3, 4, 5, 6]), 2),
    }
    split = train_bc.split_train_validation_indices(
        data,
        validation_fraction=0.05,
        validation_seed=17,
        validation_max_samples=0,
        validation_game_seeds=np.asarray([2, 4]),
        training_excluded_game_seeds=np.asarray([2, 3, 4, 5]),
    )
    np.testing.assert_array_equal(split["validation"], [2, 3, 6, 7])
    np.testing.assert_array_equal(split["train"], [0, 1, 10, 11])
    assert not set(split["train"]) & set(split["validation"])
