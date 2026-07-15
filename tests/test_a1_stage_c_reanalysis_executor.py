from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import a1_stage_c_reanalysis_executor as executor
from tools import reconstruct_state


def test_sequence_rows_preserves_sparse_absolute_decision_clock() -> None:
    data = {
        "game_seed": np.asarray([7, 7, 7, 9, 9], dtype=np.int64),
        "action_taken": np.asarray([10, 11, 12, 20, 21], dtype=np.int16),
        "decision_index": np.asarray([0, 4, 9, 0, 3], dtype=np.int32),
        "phase": np.asarray(
            ["OPENING", "PLAY_TURN", "PLAY_TURN", "OPENING", "PLAY_TURN"]
        ),
        "player": np.asarray(["RED", "BLUE", "RED", "RED", "BLUE"]),
    }

    sequences = executor._sequence_rows(data, np.asarray([9, 7, 7]))

    sequence, rows = sequences[7]
    assert sequence.actions == [10, 11, 12]
    assert sequence.decision_indices == [0, 4, 9]
    assert sequence.phases == ["OPENING", "PLAY_TURN", "PLAY_TURN"]
    assert rows.tolist() == [0, 1, 2]


def test_sequence_rows_refuses_duplicate_or_out_of_order_decisions() -> None:
    data = {
        "game_seed": np.asarray([7, 7], dtype=np.int64),
        "action_taken": np.asarray([10, 11], dtype=np.int16),
        "decision_index": np.asarray([0, 0], dtype=np.int32),
        "phase": np.asarray(["OPENING", "OPENING"]),
        "player": np.asarray(["RED", "RED"]),
    }

    with pytest.raises(executor.ExecutorError, match="malformed"):
        executor._sequence_rows(data, np.asarray([7]))


def _target_plan() -> dict:
    return {
        "target_policy_target_identity": {
            "search": {"n_full": 128, "c_scale": 0.1},
            "belief": {
                "coherent_public_belief_search": True,
                "information_set_search": False,
            },
            "chance": {"lazy_interior_chance": True},
        }
    }


def test_search_hook_requires_coherent_public_sanitization() -> None:
    calls = []
    safe = SimpleNamespace(
        config=SimpleNamespace(
            n_full=128,
            c_scale=0.1,
            coherent_public_belief_search=True,
            information_set_search=False,
            lazy_interior_chance=True,
        ),
        evaluator=SimpleNamespace(config=SimpleNamespace(public_observation=True)),
        search=lambda game, *, force_full: calls.append((game, force_full)) or "result",
    )
    executor.assert_information_set_safe_search(_target_plan(), safe)
    assert (
        executor.run_information_set_safe_search(_target_plan(), safe, "reconstructed")
        == "result"
    )
    assert calls == [("reconstructed", True)]

    hidden = SimpleNamespace(
        config=safe.config,
        evaluator=SimpleNamespace(config=SimpleNamespace(public_observation=False)),
    )
    with pytest.raises(executor.ExecutorError, match="public-observation"):
        executor.assert_information_set_safe_search(_target_plan(), hidden)

    stale = SimpleNamespace(
        config=SimpleNamespace(
            n_full=256,
            c_scale=0.1,
            coherent_public_belief_search=True,
            information_set_search=False,
            lazy_interior_chance=True,
        ),
        evaluator=safe.evaluator,
    )
    with pytest.raises(executor.ExecutorError, match="differs"):
        executor.assert_information_set_safe_search(_target_plan(), stale)


def test_sparse_failure_classification_is_per_root() -> None:
    error = reconstruct_state.SparseReconstructionError(
        "missing_nonautomatic_decision",
        "two branches",
        game_seed=7,
        decision_index=3,
        legal_action_count=2,
    )

    status, detail = executor._status_for_error(error)

    assert status == executor.STATUS["missing_nonautomatic_decision"]
    assert detail == {
        "classification": "missing_nonautomatic_decision",
        "decision_index": 3,
        "legal_action_count": 2,
        "detail": "two branches",
    }


def test_checkpoint_action_size_uses_model_contract_not_catalog_size(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "parent.pt"
    torch.save(
        {
            "config": {
                "__config_dataclass__": "EntityGraphConfig",
                "fields": {"action_size": np.int64(567)},
            }
        },
        checkpoint,
    )

    assert executor._checkpoint_action_size(checkpoint) == 567


def test_qualification_partition_owns_whole_games_and_covers_once() -> None:
    game_seeds = np.asarray([5, 5, 7, 10, 10, 11], dtype=np.int64)

    first = executor._qualification_partition_ordinals(
        game_seeds, partition_index=0, partitions=2
    )
    second = executor._qualification_partition_ordinals(
        game_seeds, partition_index=1, partitions=2
    )

    assert first.tolist() == [0, 1, 3, 4]
    assert second.tolist() == [2, 5]
    assert sorted(np.concatenate([first, second]).tolist()) == list(range(6))
    for seed in np.unique(game_seeds):
        owners = {
            partition
            for partition, ordinals in enumerate((first, second))
            if np.any(game_seeds[ordinals] == seed)
        }
        assert len(owners) == 1


def test_qualification_merge_restores_global_selected_order(
    monkeypatch, tmp_path
) -> None:
    subset = {
        "row_index": np.asarray([10, 11, 12, 13], dtype=np.int64),
        "game_seed": np.asarray([5, 5, 7, 10], dtype=np.int64),
    }

    def partition(index: int, ordinals: list[int], statuses: list[int]) -> dict:
        arrays = {
            "selected_ordinal": np.asarray(ordinals, dtype=np.int64),
            "status": np.asarray(statuses, dtype=np.uint8),
            "omitted_automatic_transitions": np.asarray(ordinals, dtype=np.uint16),
            "omitted_roll_transitions": np.zeros(len(ordinals), dtype=np.uint16),
            "omitted_end_turn_transitions": np.zeros(len(ordinals), dtype=np.uint16),
            "omitted_other_ui_transitions": np.zeros(len(ordinals), dtype=np.uint16),
        }
        return {
            "path": str(tmp_path / f"part-{index}.json"),
            "file_sha256": f"file-{index}",
            "receipt_sha256": f"receipt-{index}",
            "stage_c_plan": {"plan_sha256": "plan"},
            "target_policy_target_identity_sha256": "target",
            "runtime": {"runtime_sha256": "runtime"},
            "source_checkpoint_action_size": 567,
            "partition": {"partition_index": index, "partitions": 2},
            "failure_examples": [],
            "plan": {"path": str(tmp_path / "plan.json")},
            "subset": subset,
            "arrays": arrays,
        }

    receipts = {
        tmp_path / "part-0.json": partition(
            0,
            [0, 1, 3],
            [
                executor.STATUS["reconstructable_public_roundtrip"],
                executor.STATUS["missing_nonautomatic_decision"],
                executor.STATUS["reconstructable_public_roundtrip"],
            ],
        ),
        tmp_path / "part-1.json": partition(
            1, [2], [executor.STATUS["recorded_action_illegal"]]
        ),
    }
    monkeypatch.setattr(
        executor,
        "_verify_qualification_partition",
        lambda path: receipts[path],
    )
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return {"receipt_sha256": "merged"}

    monkeypatch.setattr(executor, "_write_qualification_artifacts", capture)

    result = executor._merge_qualification_partitions(
        SimpleNamespace(receipt=list(receipts), output_root=tmp_path / "qualification")
    )

    assert result == {"receipt_sha256": "merged"}
    assert captured["status"].tolist() == [
        executor.STATUS["reconstructable_public_roundtrip"],
        executor.STATUS["missing_nonautomatic_decision"],
        executor.STATUS["recorded_action_illegal"],
        executor.STATUS["reconstructable_public_roundtrip"],
    ]
    assert captured["omitted"].tolist() == [0, 1, 2, 3]
    assert [item["partition_index"] for item in captured["partition_receipts"]] == [
        0,
        1,
    ]


def test_effective_search_config_replays_sealed_coherent_native_fields(
    tmp_path,
) -> None:
    typed = tmp_path / "typed.json"
    typed.write_text(
        __import__("json").dumps(
            {
                "schema_version": 13,
                "fields": {
                    "n_full": 128,
                    "n_fast": 16,
                    "p_full": 0.25,
                    "c_visit": 50.0,
                    "c_scale": 0.1,
                    "coherent_public_belief_search": True,
                    "information_set_search": False,
                    "belief_chance_spectra": False,
                    "lazy_interior_chance": True,
                    "symmetry_averaged_eval": True,
                    "symmetry_averaged_eval_threshold": 20,
                    "forced_root_target_mode": "trajectory_only",
                    "native_mcts_hot_loop": True,
                    "public_observation": True,
                    "rust_featurize": True,
                },
            }
        )
    )
    plan = {
        "target_policy_target_identity": {
            "target_information_regime": "public_belief_single_tree_v1",
            "operator_contract_semantics": {
                "native_mcts_hot_loop": True,
                "coherent_public_belief_search": True,
                "information_set_search": False,
            },
            "target_semantics": {"typed_generation_config_schema": 13},
            "authority": {
                "typed_generation_config": {
                    "path": str(typed),
                    "file_sha256": executor.alignment._file_sha256(typed),
                }
            },
        }
    }

    config = executor._effective_search_config(plan, row_seed=73)

    assert config.seed == 73
    assert config.n_full == 128
    assert config.coherent_public_belief_search is True
    assert config.information_set_search is False
    assert config.lazy_interior_chance is True
    assert config.symmetry_averaged_eval_threshold == 20


def test_ragged_target_patch_is_complete_and_uses_neutral_reliability() -> None:
    identity = "sha256:" + "a" * 64
    provenance = {
        "target_policy_target_identity_sha256": "sha256:" + "b" * 64,
        "target_reanalyzer_checkpoint_sha256": "sha256:" + "c" * 64,
        "target_operator_contract_file_sha256": "sha256:" + "d" * 64,
    }
    record = {
        "ready_ordinal": 0,
        "selected_ordinal": 3,
        "row_index": 17,
        "game_seed": 19,
        "decision_index": 23,
        "chunk_index": 1,
        "identity_sha256": identity,
        "search_seed": executor._row_seed(identity),
        "selected_action_policy_id": 11,
        "root_value": 0.2,
        "root_value_mask": True,
        "simulations_used": 128,
        "used_full_search": True,
        "q_values_root_perspective": True,
        **provenance,
        "legal_action_ids": np.asarray([11, 13], dtype=np.int32),
        "target_policy": np.asarray([0.7, 0.3], dtype=np.float32),
        "target_policy_mask": np.asarray([True, True]),
        "target_scores": np.asarray([0.4, np.nan], dtype=np.float32),
        "target_scores_mask": np.asarray([True, False]),
        "completed_q_values": np.asarray([0.4, 0.1], dtype=np.float32),
        "completed_q_mask": np.asarray([True, True]),
        "prior_policy": np.asarray([0.6, 0.4], dtype=np.float32),
    }
    arrays = executor._patch_arrays([record])
    receipt = {
        "patch_columns": sorted(arrays),
        "counts": {"rows": 1, "legal_actions": 2},
        "target_policy_target_identity_sha256": provenance[
            "target_policy_target_identity_sha256"
        ],
        "target_reanalyzer_checkpoint": {
            "sha256": provenance["target_reanalyzer_checkpoint_sha256"]
        },
        "target_operator_contract": {
            "file_sha256": provenance["target_operator_contract_file_sha256"]
        },
    }

    executor._verify_patch_arrays(arrays, receipt=receipt)
    assert arrays["legal_action_offsets"].tolist() == [0, 2]
    assert arrays["target_reliability_audited"].tolist() == [False]
    assert arrays["target_reliability_confidence"].tolist() == [1.0]
