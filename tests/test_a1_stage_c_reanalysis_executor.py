from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import a1_stage_c_reanalysis_executor as executor
from tools import reconstruct_state


def test_stage_c_patch_evidence_schema_is_independent_of_generation_sidecars() -> None:
    assert (
        executor.STAGE_C_PATCH_SEARCH_EVIDENCE_SCHEMA
        == "gumbel_root_search_evidence_v1"
    )
    assert executor.STAGE_C_PATCH_SEARCH_EVIDENCE_VERSION == 1

    search = executor._rebound_search_receipt(  # noqa: SLF001
        {},
        {"target_execution": {"operator": "fixture"}},
    )
    evidence = search["row_search_evidence"]
    assert evidence["schema"] == executor.STAGE_C_PATCH_SEARCH_EVIDENCE_SCHEMA
    assert evidence["version"] == executor.STAGE_C_PATCH_SEARCH_EVIDENCE_VERSION
    assert evidence["visit_counts"] == (
        "not_present_in_v1_patch_not_required_by_overlay"
    )


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


def test_roundtrip_feature_contract_binds_nondefault_history_and_adapter() -> None:
    plan = {
        "target_policy_target_identity": {
            "target_semantics": {
                "meaningful_public_history": True,
                "event_history_limit": 64,
            },
            "operator_contract_semantics": {
                "meaningful_public_history": True,
                "event_history_limit": 64,
                "meaningful_public_history_schema": "history-v2",
                "teacher_entity_feature_adapter_version": "adapter-v6",
                "learner_entity_feature_adapter_version": "adapter-v6",
            },
            "teacher_feature_contract": {
                "entity_feature_adapter_version": "adapter-v6",
            },
        }
    }

    assert executor._roundtrip_feature_contract(plan) == (  # noqa: SLF001
        True,
        64,
        "history-v2",
        "adapter-v6",
    )


def test_roundtrip_feature_contract_refuses_adapter_drift() -> None:
    plan = {
        "target_policy_target_identity": {
            "target_semantics": {
                "meaningful_public_history": True,
                "event_history_limit": 64,
            },
            "operator_contract_semantics": {
                "meaningful_public_history": True,
                "event_history_limit": 64,
                "meaningful_public_history_schema": "history-v1",
                "teacher_entity_feature_adapter_version": "adapter-v6",
                "learner_entity_feature_adapter_version": "adapter-v6",
            },
            "teacher_feature_contract": {
                "entity_feature_adapter_version": "adapter-v5",
            },
        }
    }

    with pytest.raises(executor.ExecutorError, match="feature contracts disagree"):
        executor._roundtrip_feature_contract(plan)  # noqa: SLF001


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


def _generator_patch(
    monkeypatch,
    *,
    root_value: float = 0.2,
    root_prior_value: float = 0.1,
    target: tuple[float, float] = (0.7, 0.3),
    prior: tuple[float, float] = (0.6, 0.4),
) -> dict[str, object]:
    config = SimpleNamespace(colors=("RED", "BLUE"), map_kind=None)
    effective = executor.alignment._complete_effective_search_config(  # noqa: SLF001
        {"n_full": 128}
    )
    plan = {
        "target_policy_target_identity": {
            "target_execution": executor.alignment.STAGE_C_TARGET_EXECUTION,
            "effective_gumbel_config": effective,
        }
    }
    result = SimpleNamespace(
        improved_policy={101: target[0], 102: target[1]},
        priors={101: prior[0], 102: prior[1]},
        q_values={101: 0.4, 102: 0.1},
        completed_q_values={101: 0.4, 102: 0.1},
        used_full_search=True,
        simulations_used=128,
        root_value=root_value,
        root_prior_value=root_prior_value,
        q_values_root_perspective=True,
        selected_action=101,
    )
    game = SimpleNamespace(
        playable_action_indices=lambda _colors, _map_kind: [101, 102]
    )
    monkeypatch.setattr(
        executor, "_effective_search_config", lambda *_args, **_kwargs: config
    )
    monkeypatch.setattr(
        executor, "create_gumbel_search", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        executor, "run_information_set_safe_search", lambda *_args, **_kwargs: result
    )
    monkeypatch.setattr(executor, "_target_checkpoint", lambda _plan: object())
    monkeypatch.setattr(executor, "_checkpoint_action_size", lambda _path: 567)
    monkeypatch.setattr(
        executor,
        "rust_policy_action_ids",
        lambda *_args, **_kwargs: (11, 13),
    )
    return executor._search_patch(
        plan=plan,
        evaluator=object(),
        reconstructed_game=game,
        row_seed=73,
        expected_legal_policy_ids=np.asarray([11, 13], dtype=np.int32),
    )


@pytest.mark.parametrize("root_value", [-1.0, 1.0])
def test_search_patch_accepts_bounded_root_value_endpoints(
    monkeypatch, root_value: float
) -> None:
    patch = _generator_patch(monkeypatch, root_value=root_value)

    assert patch["root_value"] == root_value


@pytest.mark.parametrize("root_prior_value", [-1.0, 1.0])
def test_search_patch_accepts_bounded_root_prior_value_endpoints(
    monkeypatch, root_prior_value: float
) -> None:
    patch = _generator_patch(monkeypatch, root_prior_value=root_prior_value)
    assert patch["root_prior_value"] == root_prior_value


@pytest.mark.parametrize("root_prior_value", [np.nan, np.inf, -1.01, 1.01])
def test_search_patch_rejects_invalid_root_prior_value(
    monkeypatch, root_prior_value: float
) -> None:
    with pytest.raises(executor.ExecutorError, match="incomplete or ambiguous"):
        _generator_patch(monkeypatch, root_prior_value=root_prior_value)


@pytest.mark.parametrize(
    ("root_value", "target", "prior"),
    [
        (np.nan, (0.7, 0.3), (0.6, 0.4)),
        (np.inf, (0.7, 0.3), (0.6, 0.4)),
        (-1.01, (0.7, 0.3), (0.6, 0.4)),
        (1.01, (0.7, 0.3), (0.6, 0.4)),
        (0.2, (-0.25, 1.25), (0.6, 0.4)),
        (0.2, (np.nan, 1.0), (0.6, 0.4)),
        (0.2, (np.inf, 0.0), (0.6, 0.4)),
        (0.2, (0.7, 0.3), (-0.25, 1.25)),
        (0.2, (0.7, 0.3), (np.nan, 1.0)),
        (0.2, (0.7, 0.3), (np.inf, 0.0)),
    ],
)
def test_search_patch_rejects_invalid_replacement_values(
    monkeypatch,
    root_value: float,
    target: tuple[float, float],
    prior: tuple[float, float],
) -> None:
    with pytest.raises(executor.ExecutorError, match="incomplete or ambiguous"):
        _generator_patch(
            monkeypatch,
            root_value=root_value,
            target=target,
            prior=prior,
        )


def _replacement_patch() -> tuple[dict[str, np.ndarray], dict[str, object]]:
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
        "root_prior_value": 0.1,
        "root_prior_value_mask": True,
        "simulations_used": 128,
        "used_full_search": True,
        "q_values_root_perspective": True,
        **provenance,
        "legal_action_ids": np.asarray([11, 13], dtype=np.int32),
        "target_policy": np.asarray([1.0, 0.0], dtype=np.float32),
        "target_policy_mask": np.asarray([True, True]),
        "target_scores": np.asarray([0.4, np.nan], dtype=np.float32),
        "target_scores_mask": np.asarray([True, False]),
        "completed_q_values": np.asarray([0.4, 0.1], dtype=np.float32),
        "completed_q_mask": np.asarray([True, True]),
        "prior_policy": np.asarray([0.6, 0.4], dtype=np.float32),
    }
    arrays = executor._patch_arrays([record])
    receipt = {
        "patch_schema_version": executor.PATCH_SCHEMA,
        "search": {
            "effective_config_without_row_seed": executor.alignment._complete_effective_search_config(  # noqa: SLF001
                {"n_full": 128}
            )
        },
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

    return arrays, receipt


def test_ragged_target_patch_is_complete_and_uses_neutral_reliability() -> None:
    arrays, receipt = _replacement_patch()

    executor._verify_patch_arrays(arrays, receipt=receipt)
    assert arrays["legal_action_offsets"].tolist() == [0, 2]
    assert arrays["target_reliability_audited"].tolist() == [False]
    assert arrays["target_reliability_confidence"].tolist() == [1.0]
    assert arrays["target_policy_mask_flat"].tolist() == [True, True]


@pytest.mark.parametrize("root_value", [-1.0, 1.0])
def test_patch_verifier_accepts_bounded_root_value_endpoints(
    root_value: float,
) -> None:
    arrays, receipt = _replacement_patch()
    arrays["root_value"][...] = root_value

    executor._verify_patch_arrays(arrays, receipt=receipt)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("root_value", np.nan),
        ("root_value", np.inf),
        ("root_value", -1.01),
        ("root_value", 1.01),
        ("root_prior_value", np.nan),
        ("root_prior_value", np.inf),
        ("root_prior_value", -1.01),
        ("root_prior_value", 1.01),
        ("target_policy_flat", np.asarray([-0.25, 1.25], dtype=np.float32)),
        ("target_policy_flat", np.asarray([np.nan, 1.0], dtype=np.float32)),
        ("target_policy_flat", np.asarray([np.inf, 0.0], dtype=np.float32)),
        ("prior_policy_flat", np.asarray([-0.25, 1.25], dtype=np.float32)),
        ("prior_policy_flat", np.asarray([np.nan, 1.0], dtype=np.float32)),
        ("prior_policy_flat", np.asarray([np.inf, 0.0], dtype=np.float32)),
    ],
)
def test_patch_verifier_rejects_invalid_replacement_values(
    column: str, value: object
) -> None:
    arrays, receipt = _replacement_patch()
    arrays[column][...] = value

    with pytest.raises(executor.ExecutorError, match="invalid search evidence"):
        executor._verify_patch_arrays(arrays, receipt=receipt)


def test_v3_patch_requires_paired_root_prior_columns_and_mask() -> None:
    arrays, receipt = _replacement_patch()
    without_prior = dict(arrays)
    without_prior.pop("root_prior_value")
    without_prior.pop("root_prior_value_mask")
    receipt_without = {**receipt, "patch_columns": sorted(without_prior)}
    with pytest.raises(executor.ExecutorError, match="column contract drifted"):
        executor._verify_patch_arrays(without_prior, receipt=receipt_without)

    arrays["root_prior_value_mask"][0] = False
    with pytest.raises(executor.ExecutorError, match="invalid search evidence"):
        executor._verify_patch_arrays(arrays, receipt=receipt)


def test_v2_patch_remains_readable_without_fabricated_root_prior() -> None:
    arrays, receipt = _replacement_patch()
    arrays.pop("root_prior_value")
    arrays.pop("root_prior_value_mask")
    receipt = {
        **receipt,
        "patch_schema_version": executor.PATCH_SCHEMA_V2,
        "patch_columns": sorted(arrays),
    }
    executor._verify_patch_arrays(arrays, receipt=receipt)


def test_stage_c_alignment_imports_before_executor_in_clean_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tools import a1_stage_c_teacher_alignment; "
                "from tools import a1_stage_c_reanalysis_executor"
            ),
        ],
        cwd=executor.REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_forced_full_simulation_accounting_replays_width_dependent_sh() -> None:
    effective = executor.alignment._complete_effective_search_config(  # noqa: SLF001
        {"n_full": 128, "max_root_candidates_wide": 54}
    )

    assert executor._expected_forced_full_simulations(3, effective) == 127
    assert executor._expected_forced_full_simulations(16, effective) == 128
    assert executor._expected_forced_full_simulations(26, effective) == 112
    assert executor._expected_forced_full_simulations(40, effective) == 141
    assert executor._expected_forced_full_simulations(54, effective) == 154


def test_portable_runtime_verifies_historical_git_blobs_not_current_tree(
    monkeypatch, tmp_path
) -> None:
    native = tmp_path / "native.so"
    native.write_bytes(b"sealed native")
    commit = "a" * 40
    sources = [
        {"path": path, "file_sha256": f"sha256:{index:064x}"}
        for index, path in enumerate(sorted(executor.RUNTIME_SOURCE_PATHS), 1)
    ]
    runtime = {
        "schema_version": "a1-stage-c-reconstruction-runtime-v1",
        "repo_commit": commit,
        "sources": sources,
        "native_runtime": {
            "path": str(native),
            "file_sha256": executor.alignment._file_sha256(native),
            "distribution_version": "0.1.10",
            "capabilities": sorted(executor.REQUIRED_COHERENT_CAPABILITIES),
        },
    }
    runtime["runtime_sha256"] = executor._value_sha256(runtime)
    expected = {item["path"]: item["file_sha256"] for item in sources}
    monkeypatch.setattr(
        executor,
        "_git_blob_sha256",
        lambda resolved_commit, path: (
            expected[path] if resolved_commit == commit else "wrong"
        ),
    )
    monkeypatch.setattr(
        executor.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    executor._verify_runtime_attestation(runtime, require_current=False)
