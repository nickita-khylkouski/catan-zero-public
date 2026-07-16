from __future__ import annotations

import numpy as np
import pytest

from tools import train_bc


def _data(**overrides):
    data = {
        "action_taken": np.zeros(4, dtype=np.int16),
        "game_seed": np.asarray([10, 10, 11, 11], dtype=np.int64),
        "player": np.asarray(["BLUE", "RED", "BLUE", "RED"]),
        "seat": np.asarray([0, 1, 0, 1], dtype=np.int8),
        "winner": np.asarray(["RED", "RED", "BLUE", "BLUE"]),
        "terminated": np.ones(4, dtype=np.bool_),
        "truncated": np.zeros(4, dtype=np.bool_),
    }
    data.update(overrides)
    return data


def test_value_outcome_admission_verifies_actor_perspective_and_seated_winner():
    report = train_bc._validate_value_outcome_labels(  # noqa: SLF001
        _data(),
        chunk_rows=1,
    )

    assert report == {
        "schema_version": "value-outcome-label-admission-v1",
        "present": True,
        "rows": 4,
        "games": 2,
        "clean_outcome_rows": 4,
        "clean_outcome_games": 2,
        "bilateral_games": 2,
        "one_sided_games": 0,
        "one_sided_winner_unobserved_games": 0,
        "player_domain": ["BLUE", "RED", "ORANGE", "WHITE"],
        "actor_seat_perspective_verified": True,
        "winner_seated_per_game_verified": True,
        "bilateral_winner_seated_per_game_verified": True,
        "game_identity_namespace": "game_seed",
        "components": {
            "ordinary_corpus": {
                "games": 2,
                "clean_outcome_games": 2,
                "bilateral_games": 2,
                "one_sided_games": 0,
                "one_sided_winner_unobserved_games": 0,
            }
        },
    }


def test_value_outcome_admission_rejects_unknown_winner_before_negative_labels():
    with pytest.raises(SystemExit, match="unknown player/winner"):
        train_bc._validate_value_outcome_labels(  # noqa: SLF001
            _data(winner=np.asarray(["DRAW", "DRAW", "BLUE", "BLUE"]))
        )


def test_value_outcome_admission_rejects_winner_not_seated_in_game():
    with pytest.raises(SystemExit, match="winner is inconsistent or not seated"):
        train_bc._validate_value_outcome_labels(  # noqa: SLF001
            _data(winner=np.asarray(["ORANGE", "ORANGE", "BLUE", "BLUE"]))
        )


def test_value_outcome_admission_accepts_losing_one_sided_game():
    report = train_bc._validate_value_outcome_labels(  # noqa: SLF001
        _data(
            game_seed=np.asarray([10, 10, 11, 11], dtype=np.int64),
            player=np.asarray(["BLUE", "BLUE", "RED", "RED"]),
            seat=np.asarray([0, 0, 1, 1], dtype=np.int8),
            winner=np.asarray(["RED", "RED", "RED", "RED"]),
        )
    )

    assert report["one_sided_games"] == 2
    assert report["one_sided_winner_unobserved_games"] == 1
    assert report["bilateral_games"] == 0


def test_value_outcome_admission_namespaces_reused_seed_by_component():
    class _Composite(dict):
        component_ids = ("one_sided_blue", "one_sided_red")

        @staticmethod
        def component_indices_for_rows(rows):
            mapping = np.asarray([0, 0, 1, 1], dtype=np.int64)
            return mapping[np.asarray(rows, dtype=np.int64)]

    data = _Composite(
        action_taken=np.zeros(4, dtype=np.int16),
        game_seed=np.asarray([7, 7, 7, 7], dtype=np.int64),
        player=np.asarray(["BLUE", "BLUE", "RED", "RED"]),
        seat=np.asarray([0, 0, 1, 1], dtype=np.int8),
        winner=np.asarray(["RED", "RED", "BLUE", "BLUE"]),
        terminated=np.ones(4, dtype=np.bool_),
        truncated=np.zeros(4, dtype=np.bool_),
    )

    report = train_bc._validate_value_outcome_labels(data)  # noqa: SLF001

    assert report["games"] == 2
    assert report["one_sided_winner_unobserved_games"] == 2
    assert report["game_identity_namespace"] == "component_id+game_seed"
    assert set(report["components"]) == {"one_sided_blue", "one_sided_red"}


def test_value_outcome_admission_rejects_actor_seat_perspective_swap():
    with pytest.raises(SystemExit, match="actor/seat perspective mismatch"):
        train_bc._validate_value_outcome_labels(  # noqa: SLF001
            _data(seat=np.asarray([1, 0, 0, 1], dtype=np.int8))
        )


def test_value_outcome_admission_rejects_nonterminal_winner():
    with pytest.raises(SystemExit, match="non-terminal rows"):
        train_bc._validate_value_outcome_labels(  # noqa: SLF001
            _data(terminated=np.asarray([False, False, True, True]))
        )


def test_value_outcome_admission_without_game_seed_checks_domain_not_seating():
    data = _data()
    del data["game_seed"]

    report = train_bc._validate_value_outcome_labels(data)  # noqa: SLF001

    assert report["games"] == 0
    assert report["clean_outcome_rows"] == 4
    assert report["clean_outcome_games"] == 0
    assert report["actor_seat_perspective_verified"] is True
    assert report["winner_seated_per_game_verified"] is False
    assert report["bilateral_winner_seated_per_game_verified"] is False
