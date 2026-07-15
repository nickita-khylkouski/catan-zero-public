"""Adversarial tests for authenticated archived-state reconstruction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import reconstruct_state as rs  # noqa: E402


class _FakeGame:
    def playable_action_indices(self, colors, _prompt):
        assert colors == ["RED", "BLUE"]
        return [11, 12]

    def playable_actions_json(self):
        return json.dumps([{"id": 11}, {"id": 12}])

    def json_snapshot(self):
        return json.dumps({"current_prompt": "ROLL"})

    def current_color(self):
        return "RED"

    def winning_color(self):
        return None


def test_featurize_state_always_uses_public_observation(monkeypatch):
    calls: list[tuple[str, bool]] = []

    monkeypatch.setattr(rs, "rust_policy_action_ids", lambda *a, **k: (5, 6))

    def entity(*args, **kwargs):
        calls.append(("entity", kwargs.get("public_observation")))
        return {"hex_tokens": np.zeros((1, 2, 3), dtype=np.float32)}

    def context(*args, **kwargs):
        calls.append(("context", kwargs.get("public_observation")))
        return np.zeros((1, 2, 4), dtype=np.float32)

    monkeypatch.setattr(rs, "rust_game_to_entity_batch", entity)
    monkeypatch.setattr(rs, "rust_action_context_batch", context)

    result = rs.featurize_state(_FakeGame(), action_size=64)

    assert calls == [("entity", True), ("context", True)]
    assert result["legal_policy_ids"] == (5, 6)


def _sequence() -> rs.GameActionSequence:
    return rs.GameActionSequence(
        game_seed=123,
        colors=("RED", "BLUE"),
        actions=[5],
        decision_indices=[0],
        phases=["ROLL"],
        players=["RED"],
    )


def _reconstructed_features(*, legal_ids=(5, 6)):
    return {
        "legal_policy_ids": tuple(legal_ids),
        "phase": "ROLL",
        "acting_color": "RED",
        "features": {
            "hex_tokens": np.array([[1.0, 2.0]], dtype=np.float32),
            "legal_action_tokens": np.array(
                [[1.0, 2.0], [3.0, 4.0]], dtype=np.float32
            ),
            "legal_action_mask": np.array([True, True]),
            "legal_action_target_ids": np.array([[1, 2], [3, 4]], dtype=np.int16),
        },
        "context": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    }


def _stored_features():
    return {
        "hex_tokens": np.array([[1.0, 2.0]], dtype=np.float16),
        "legal_action_tokens": np.array(
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]], dtype=np.float16
        ),
        "legal_action_mask": np.array([True, True, False]),
        "legal_action_target_ids": np.array(
            [[1, 2], [3, 4], [-1, -1]], dtype=np.int16
        ),
        "legal_action_context": np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.0, 0.0]], dtype=np.float16
        ),
    }


def test_round_trip_authenticates_legal_order_action_tokens_and_context(monkeypatch):
    monkeypatch.setattr(rs, "reconstruct_state", lambda *a, **k: _FakeGame())
    monkeypatch.setattr(rs, "featurize_state", lambda *a, **k: _reconstructed_features())

    result = rs.round_trip_row(
        _sequence(),
        0,
        _stored_features(),
        np.array([5, 6, -1], dtype=np.int16),
    )

    assert result.ok
    assert result.legal_ids_match
    assert result.max_abs_diff <= 1e-2


@pytest.mark.parametrize("corruption", ["legal_order", "action_token", "context"])
def test_round_trip_refuses_semantic_mismatch(monkeypatch, corruption):
    features = _stored_features()
    reconstructed = _reconstructed_features(
        legal_ids=(6, 5) if corruption == "legal_order" else (5, 6)
    )
    if corruption == "action_token":
        features["legal_action_tokens"][0, 0] = 99
    if corruption == "context":
        features["legal_action_context"][1, 0] = 99
    monkeypatch.setattr(rs, "reconstruct_state", lambda *a, **k: _FakeGame())
    monkeypatch.setattr(rs, "featurize_state", lambda *a, **k: reconstructed)

    result = rs.round_trip_row(
        _sequence(),
        0,
        features,
        np.array([5, 6, -1], dtype=np.int16),
    )

    assert not result.ok
    if corruption == "legal_order":
        assert not result.legal_ids_match
    else:
        assert result.max_abs_diff == float("inf") or result.max_abs_diff > 1e-2


def _patch_cli(monkeypatch, result):
    seq = _sequence()
    monkeypatch.setattr(rs, "gather_game_action_sequence", lambda *a, **k: seq)
    monkeypatch.setattr(rs, "reconstruct_state", lambda *a, **k: _FakeGame())
    monkeypatch.setattr(rs, "_locate_round_trip_row", lambda *a, **k: (Path("row.npz"), 7))
    calls = []

    def round_trip(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(rs, "round_trip_shard_rows", round_trip)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reconstruct_state.py",
            "--scope",
            "/archive",
            "--game-seed",
            "123",
            "--decision-index",
            "0",
            "--round-trip",
        ],
    )
    return calls


def test_round_trip_cli_executes_exact_archived_row(monkeypatch, capsys):
    summary = {
        "shard": "row.npz",
        "shard_sha256": "sha256:abc",
        "selection": {"kind": "explicit_rows", "row_indices": [7]},
        "rows_checked": 1,
        "rows_passed": 1,
    }
    calls = _patch_cli(monkeypatch, summary)

    rs.main()

    assert calls[0][0] == (Path("row.npz"),)
    assert calls[0][1]["row_indices"] == [7]
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["round_trip"]["shard_sha256"] == "sha256:abc"


def test_round_trip_cli_refuses_mismatch(monkeypatch, capsys):
    calls = _patch_cli(
        monkeypatch,
        {"rows_checked": 1, "rows_passed": 0, "failures": [{"worst_key": "hex_tokens"}]},
    )

    with pytest.raises(SystemExit, match="round-trip verification FAILED"):
        rs.main()

    assert calls
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["round_trip"]["rows_passed"] == 0


class _SparseGame:
    def __init__(self, legal_by_step, step=0):
        self.legal_by_step = legal_by_step
        self.step = step

    def playable_action_indices(self, _colors, _prompt):
        return self.legal_by_step[self.step]

    def playable_actions_json(self):
        action_type = "ROLL" if self.step == 1 else "END_TURN"
        return json.dumps(
            [["RED", action_type, None] for _action in self.legal_by_step[self.step]]
        )

    def winning_color(self):
        return None

    def copy(self):
        return _SparseGame(self.legal_by_step, self.step)


class _SparseGameModule:
    class Game:
        legal_by_step = [[10, 11], [20], [30], [40, 41]]

        @classmethod
        def simple(cls, _colors, seed):
            assert seed == 91
            return _SparseGame(cls.legal_by_step)


def _patch_sparse_replay(monkeypatch):
    _SparseGameModule.Game.legal_by_step = [[10, 11], [20], [30], [40, 41]]
    applied = []
    monkeypatch.setattr(rs, "_require_rust_module", lambda: _SparseGameModule)
    monkeypatch.setattr(
        rs,
        "_policy_id_to_rust_id",
        lambda _game, policy, **_kwargs: int(policy),
    )

    def apply(game, action, **_kwargs):
        applied.append((game.step, action))
        return _SparseGame(game.legal_by_step, game.step + 1)

    monkeypatch.setattr(rs, "_apply_selected_action", apply)
    return applied


def test_sparse_reconstruction_fills_only_unique_automatic_gaps(monkeypatch):
    applied = _patch_sparse_replay(monkeypatch)
    sequence = rs.GameActionSequence(
        game_seed=91,
        colors=("RED", "BLUE"),
        actions=[10, 40],
        decision_indices=[0, 3],
        phases=["BUILD_INITIAL_SETTLEMENT", "PLAY_TURN"],
        players=["RED", "BLUE"],
    )

    game = rs.reconstruct_state_from_sequence(sequence, 3)

    assert game.step == 3
    assert applied == [(0, 10), (1, 20), (2, 30)]

    batch = rs.reconstruct_states_from_sequence(sequence, [3])
    assert batch.failure is None
    assert batch.omitted_automatic_transitions == {3: 2}
    assert batch.omitted_automatic_transition_types == {
        3: {"ROLL": 1, "END_TURN": 1}
    }


def test_sparse_reconstruction_proves_missing_multi_action_is_ambiguous(
    monkeypatch,
):
    _patch_sparse_replay(monkeypatch)
    _SparseGameModule.Game.legal_by_step = [[10], [20, 21], [30], [40, 41]]
    sequence = rs.GameActionSequence(
        game_seed=91,
        colors=("RED", "BLUE"),
        actions=[10, 40],
        decision_indices=[0, 3],
        phases=["BUILD_INITIAL_SETTLEMENT", "PLAY_TURN"],
        players=["RED", "BLUE"],
    )

    with pytest.raises(rs.SparseReconstructionError) as captured:
        rs.reconstruct_state_from_sequence(sequence, 3)

    assert captured.value.code == "missing_nonautomatic_decision"
    assert captured.value.decision_index == 1
    assert captured.value.legal_action_count == 2


def test_sparse_batch_keeps_roots_before_first_ambiguous_gap(monkeypatch):
    _patch_sparse_replay(monkeypatch)
    _SparseGameModule.Game.legal_by_step = [[10], [20, 21], [30], [40, 41]]
    sequence = rs.GameActionSequence(
        game_seed=91,
        colors=("RED", "BLUE"),
        actions=[10, 40],
        decision_indices=[0, 3],
        phases=["BUILD_INITIAL_SETTLEMENT", "PLAY_TURN"],
        players=["RED", "BLUE"],
    )

    result = rs.reconstruct_states_from_sequence(sequence, [0, 3])

    assert set(result.states) == {0}
    assert result.omitted_automatic_transitions == {0: 0}
    assert result.omitted_automatic_transition_types == {0: {}}
    assert result.failure is not None
    assert result.failure.code == "missing_nonautomatic_decision"
