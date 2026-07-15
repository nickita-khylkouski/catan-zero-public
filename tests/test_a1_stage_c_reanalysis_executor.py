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
        evaluator=SimpleNamespace(
            config=SimpleNamespace(public_observation=True)
        ),
        search=lambda game, *, force_full: calls.append((game, force_full)) or "result",
    )
    executor.assert_information_set_safe_search(_target_plan(), safe)
    assert executor.run_information_set_safe_search(
        _target_plan(), safe, "reconstructed"
    ) == "result"
    assert calls == [("reconstructed", True)]

    hidden = SimpleNamespace(
        config=safe.config,
        evaluator=SimpleNamespace(
            config=SimpleNamespace(public_observation=False)
        ),
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
