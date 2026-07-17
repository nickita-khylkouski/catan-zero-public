from __future__ import annotations

import numpy as np
import pytest

from catan_zero.rl.actor_public_rule_state_admission import (
    ActorPublicRuleStateAdmissionError,
    audit_actor_playable_development_cards,
)
from catan_zero.rl.entity_feature_adapter import RUST_ENTITY_ADAPTER_V5
from catan_zero.rl import gumbel_self_play
from tools import train_bc


def _arrays() -> dict[str, np.ndarray]:
    global_tokens = np.zeros((4, 1, 43), dtype=np.float16)
    # KNIGHT, MONOPOLY, ROAD_BUILDING, YEAR_OF_PLENTY respectively.
    global_tokens[0, 0, 12] = np.float16(0.2)
    global_tokens[1, 0, 14] = np.float16(0.2)
    global_tokens[2, 0, 15] = np.float16(0.2)
    global_tokens[3, 0, 13] = np.float16(0.2)
    action_taken = np.asarray([304, 305, 310, 311], dtype=np.int16)
    legal_action_ids = np.asarray(
        [[304, -1], [307, -1], [310, -1], [329, -1]],
        dtype=np.int16,
    )
    legal_action_mask = legal_action_ids >= 0
    return {
        "global_tokens": global_tokens,
        "action_taken": action_taken,
        "legal_action_ids": legal_action_ids,
        "legal_action_mask": legal_action_mask,
    }


def test_actor_playable_dev_admission_accepts_exact_action_slot_mapping() -> None:
    result = audit_actor_playable_development_cards(
        _arrays(),
        where="valid-v5",
        chunk_rows=2,
    )

    assert result["authenticated"] is True
    assert result["observed_cards"] == [
        "KNIGHT",
        "YEAR_OF_PLENTY",
        "MONOPOLY",
        "ROAD_BUILDING",
    ]
    assert all(
        row["required_rows"] == 1 and row["positive_feature_rows"] == 1
        for row in result["cards"].values()
    )


@pytest.mark.parametrize(
    ("row", "slot", "card"),
    [
        (0, 12, "KNIGHT"),
        (1, 14, "MONOPOLY"),
        (2, 15, "ROAD_BUILDING"),
        (3, 13, "YEAR_OF_PLENTY"),
    ],
)
def test_actor_playable_dev_admission_rejects_selected_action_contradiction(
    row: int,
    slot: int,
    card: str,
) -> None:
    arrays = _arrays()
    arrays["global_tokens"][row, 0, slot] = 0

    with pytest.raises(ActorPublicRuleStateAdmissionError, match=card):
        audit_actor_playable_development_cards(arrays, where="corrupt-v5")


def test_actor_playable_dev_admission_checks_legal_not_only_selected_actions() -> None:
    arrays = _arrays()
    arrays["action_taken"][0] = 1
    arrays["global_tokens"][0, 0, 12] = 0

    with pytest.raises(ActorPublicRuleStateAdmissionError, match="KNIGHT"):
        audit_actor_playable_development_cards(arrays, where="corrupt-legal-row")


def test_actor_playable_dev_admission_rejects_missing_columns_and_bad_shapes() -> None:
    arrays = _arrays()
    del arrays["legal_action_mask"]
    with pytest.raises(ActorPublicRuleStateAdmissionError, match="legal_action_mask"):
        audit_actor_playable_development_cards(arrays, where="missing")

    arrays = _arrays()
    arrays["global_tokens"] = np.zeros((4, 43), dtype=np.float16)
    with pytest.raises(ActorPublicRuleStateAdmissionError, match="global_tokens shape"):
        audit_actor_playable_development_cards(arrays, where="bad-shape")


def test_v5_shard_writer_refuses_corrupt_rows_before_atomic_write(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = _arrays()
    arrays["global_tokens"][0, 0, 12] = 0
    arrays["adapter_version"] = np.full(
        (4,), RUST_ENTITY_ADAPTER_V5, dtype=f"<U{len(RUST_ENTITY_ADAPTER_V5)}"
    )
    monkeypatch.setattr(
        gumbel_self_play,
        "_rows_to_arrays",
        lambda *_args, **_kwargs: arrays,
    )
    writer = gumbel_self_play.GumbelShardWriter(tmp_path, shard_size=99)
    writer.rows = [{}]

    with pytest.raises(ActorPublicRuleStateAdmissionError, match="KNIGHT"):
        writer.flush()

    assert not list(tmp_path.glob("*.npz"))


def test_train_startup_admission_quarantines_preexisting_corrupt_corpus() -> None:
    arrays = _arrays()
    arrays["global_tokens"][:, 0, 12:16] = 0

    with pytest.raises(SystemExit, match="KNIGHT"):
        train_bc._audit_actor_public_rule_state_corpus(  # noqa: SLF001
            arrays,
            ddp={"enabled": False, "rank": 0, "world_size": 1},
        )


def test_train_startup_admission_accepts_valid_component() -> None:
    result = train_bc._audit_actor_public_rule_state_corpus(  # noqa: SLF001
        _arrays(),
        ddp={"enabled": False, "rank": 0, "world_size": 1},
    )

    assert result["authenticated"] is True
    assert result["component_count"] == 1
    assert result["components"][0]["observed_cards"] == [
        "KNIGHT",
        "YEAR_OF_PLENTY",
        "MONOPOLY",
        "ROAD_BUILDING",
    ]


class _FeatureGame:
    def playable_action_indices(self, _colors, _kind):
        return [7]


def _mock_public_learner_feature_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    drift_non_rule_tensor: bool = False,
) -> None:
    python_global = np.zeros((1, 1, 43), dtype=np.float16)
    python_global[0, 0, 8] = np.float16(1.0)
    python_player = np.zeros((1, 4, 31), dtype=np.float16)
    native_global = python_global[0].copy()
    native_global[0, 12] = np.float16(0.2)
    native_player = python_player[0].copy()
    if drift_non_rule_tensor:
        native_player[0, 0] = np.float16(1.0)

    monkeypatch.setattr(
        gumbel_self_play,
        "rust_policy_action_ids",
        lambda *_args, **_kwargs: (304,),
    )
    monkeypatch.setattr(
        gumbel_self_play,
        "_resolve_entity_adapter",
        lambda *_args, **_kwargs: ({}, object(), []),
    )
    monkeypatch.setattr(
        gumbel_self_play,
        "rust_game_to_entity_batch",
        lambda *_args, **_kwargs: {
            "global_tokens": python_global.copy(),
            "player_tokens": python_player.copy(),
        },
    )
    monkeypatch.setattr(
        gumbel_self_play,
        "require_rust_feature_path",
        lambda: None,
    )
    monkeypatch.setattr(
        gumbel_self_play,
        "compute_rust_topology",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        gumbel_self_play,
        "build_entity_features_rust",
        lambda *_args, **_kwargs: {
            "global_tokens": native_global.copy(),
            "player_tokens": native_player.copy(),
        },
    )
    monkeypatch.setattr(
        gumbel_self_play,
        "rust_action_context_batch",
        lambda *_args, **_kwargs: np.zeros((1, 1, 50), dtype=np.float32),
    )


def test_public_learner_v5_replaces_stale_json_zero_with_native_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_public_learner_feature_dependencies(monkeypatch)

    mapped, features, *_ = gumbel_self_play._build_public_learner_features(  # noqa: SLF001
        _FeatureGame(),
        (7,),
        colors=("RED", "BLUE"),
        action_size=332,
        actor="RED",
        snapshot={},
        action_by_id={7: ["RED", "PLAY_KNIGHT_CARD", None]},
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V5,
    )

    assert mapped == (304,)
    assert features["global_tokens"][0, 8] == np.float16(1.0)
    assert features["global_tokens"][0, 12] == np.float16(0.2)
    assert np.count_nonzero(features["global_tokens"][0, 13:16]) == 0


def test_public_learner_v5_refuses_native_drift_outside_rule_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_public_learner_feature_dependencies(
        monkeypatch,
        drift_non_rule_tensor=True,
    )

    with pytest.raises(RuntimeError, match="outside slots 12:16.*player_tokens"):
        gumbel_self_play._build_public_learner_features(  # noqa: SLF001
            _FeatureGame(),
            (7,),
            colors=("RED", "BLUE"),
            action_size=332,
            actor="RED",
            snapshot={},
            action_by_id={7: ["RED", "PLAY_KNIGHT_CARD", None]},
            entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V5,
        )
