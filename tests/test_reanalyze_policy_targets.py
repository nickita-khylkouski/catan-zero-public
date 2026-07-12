from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from tools import reanalyze_policy_targets as target


def _write_source(
    tmp_path: Path, *, regime: str = target.TARGET_INFORMATION_REGIME_PUBLIC
):
    producer = tmp_path / "producer.pt"
    reanalyzer = tmp_path / "reanalyzer.pt"
    producer.write_bytes(b"producer")
    reanalyzer.write_bytes(b"reanalyzer")
    shard = tmp_path / "source.npz"
    arrays = {
        "action_taken": np.asarray([10, 11], dtype=np.int16),
        "game_seed": np.asarray([7, 7], dtype=np.int64),
        "decision_index": np.asarray([0, 1], dtype=np.int32),
        "phase": np.asarray(["A", "B"]),
        "player": np.asarray(["RED", "BLUE"]),
        "terminated": np.asarray([True, True]),
        "truncated": np.asarray([False, False]),
        "winner": np.asarray(["RED", "RED"]),
        "policy_weight_multiplier": np.asarray([1.0, 1.0], dtype=np.float32),
        "used_full_search": np.asarray([True, True]),
        "is_forced": np.asarray([False, False]),
        "target_information_regime": np.asarray([regime, regime]),
        "legal_action_ids": np.asarray([[10, 12, -1], [11, 13, -1]], dtype=np.int16),
        "target_policy": np.asarray(
            [[0.6, 0.4, 0.0], [0.7, 0.3, 0.0]], dtype=np.float32
        ),
        "target_scores": np.asarray(
            [[0.2, 0.1, np.nan], [0.3, -0.1, np.nan]], dtype=np.float32
        ),
        "target_scores_mask": np.asarray([[True, True, False], [True, True, False]]),
        "root_value": np.asarray([0.2, 0.3], dtype=np.float32),
        "prior_policy": np.asarray(
            [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], dtype=np.float16
        ),
        "aux_vp_in_n": np.asarray([1.0, -1.0], dtype=np.float32),
    }
    np.savez(shard, **arrays)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "producer_checkpoint_sha256": target._sha256(producer),
                "shards": [str(shard)],
            }
        )
    )
    return producer, reanalyzer, shard, manifest, arrays


def _plan(
    tmp_path: Path,
    *,
    chunks: int = 1,
    regime: str = target.TARGET_INFORMATION_REGIME_PUBLIC,
):
    producer, reanalyzer, shard, manifest, arrays = _write_source(
        tmp_path, regime=regime
    )
    plan = target.build_plan(
        source_manifest=manifest,
        trajectory_producer_checkpoint=producer,
        target_checkpoint=reanalyzer,
        chunks=chunks,
        search_config=target.default_search_config(),
    )
    return plan, producer, reanalyzer, shard, arrays


def _patch(_search, _game, _feature):
    return {
        "target_policy": [0.25, 0.75],
        "target_scores": [-0.2, 0.8],
        "target_scores_mask": [True, True],
        "root_value": 0.55,
        "prior_policy": [0.4, 0.6],
    }


def _bypass_reconstruction(**_kwargs):
    return object(), {"legal_policy_ids": (10, 12)}


def test_reconstruction_mismatch_stops_before_search(tmp_path: Path) -> None:
    _plan_value, _producer, _reanalyzer, shard_path, _arrays = _plan(tmp_path)
    shard = target.load_shard(shard_path)
    sequence = target.GameActionSequence(
        7, target.COLORS, [10, 11], [0, 1], ["A", "B"], ["RED", "BLUE"]
    )
    with pytest.raises(
        target.ReanalysisError, match="complete public reconstruction surface"
    ):
        target._verify_reconstruction(shard=shard, row=0, sequence=sequence)


def test_hidden_information_targets_are_not_admitted(tmp_path: Path) -> None:
    producer, reanalyzer, _shard, manifest, _arrays = _write_source(
        tmp_path, regime="authoritative_hidden_state_search_v1"
    )
    with pytest.raises(target.ReanalysisError, match="no authenticated policy-active"):
        target.build_plan(
            source_manifest=manifest,
            trajectory_producer_checkpoint=producer,
            target_checkpoint=reanalyzer,
            chunks=1,
            search_config=target.default_search_config(),
        )


def test_merge_changes_only_search_target_columns(monkeypatch, tmp_path: Path) -> None:
    plan, _producer, _reanalyzer, shard_path, _arrays = _plan(tmp_path)
    monkeypatch.setattr(target, "_verify_reconstruction", _bypass_reconstruction)
    monkeypatch.setattr(target, "_search_patch", _patch)
    claim_path = tmp_path / "claim.json"
    target.run_chunk(plan=plan, chunk_index=0, output=claim_path, search=object())
    output = tmp_path / "merged"
    manifest = target.merge_claims(plan=plan, claim_paths=[claim_path], output=output)
    original = target.load_shard(shard_path)
    rebuilt = target.load_shard(output / manifest["shards"][0])
    assert manifest["rewritten_columns"] == sorted(target.REWRITTEN_COLUMNS)
    for key in original:
        if key not in target.REWRITTEN_COLUMNS:
            assert target._array_equal(original[key], rebuilt[key]), key
    assert np.allclose(rebuilt["target_policy"][:, :2], [[0.25, 0.75], [0.25, 0.75]])
    assert np.allclose(rebuilt["root_value"], 0.55)
    assert manifest["payload_inventory_sha256"] == target._value_sha256(
        manifest["payload_inventory"]
    )


def test_chunk_rerun_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    plan, *_rest = _plan(tmp_path)
    monkeypatch.setattr(target, "_verify_reconstruction", _bypass_reconstruction)
    monkeypatch.setattr(target, "_search_patch", _patch)
    first = target.run_chunk(
        plan=plan, chunk_index=0, output=tmp_path / "first.json", search=object()
    )
    second = target.run_chunk(
        plan=plan, chunk_index=0, output=tmp_path / "second.json", search=object()
    )
    assert first == second
    assert (tmp_path / "first.json").read_bytes() == (
        tmp_path / "second.json"
    ).read_bytes()

    first_out = tmp_path / "merge-first"
    second_out = tmp_path / "merge-second"
    first_merge = target.merge_claims(
        plan=plan, claim_paths=[tmp_path / "first.json"], output=first_out
    )
    second_merge = target.merge_claims(
        plan=plan, claim_paths=[tmp_path / "second.json"], output=second_out
    )
    assert first_merge["payload_inventory"] == second_merge["payload_inventory"]
    assert (first_out / first_merge["shards"][0]).read_bytes() == (
        second_out / second_merge["shards"][0]
    ).read_bytes()


def test_checkpoint_swap_invalidates_plan(tmp_path: Path) -> None:
    plan, _producer, reanalyzer, *_rest = _plan(tmp_path)
    reanalyzer.write_bytes(b"swapped")
    with pytest.raises(
        target.ReanalysisError, match="target_reanalyzer checkpoint hash drift"
    ):
        target._verify_plan(plan)


def test_merge_refuses_incomplete_chunk_set(monkeypatch, tmp_path: Path) -> None:
    plan, *_rest = _plan(tmp_path, chunks=2)
    monkeypatch.setattr(target, "_verify_reconstruction", _bypass_reconstruction)
    monkeypatch.setattr(target, "_search_patch", _patch)
    first = tmp_path / "claim0.json"
    target.run_chunk(plan=plan, chunk_index=0, output=first, search=object())
    with pytest.raises(target.ReanalysisError, match="incomplete claims"):
        target.merge_claims(plan=plan, claim_paths=[first], output=tmp_path / "merged")
