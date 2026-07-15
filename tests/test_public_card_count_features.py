from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from catan_zero.deduction_tracker import (
    DEDUCTION_FEATURE_SIZE,
    DEDUCTION_FEATURES_KEY,
    public_card_count_feature_table,
)
from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE
from catan_zero.rl.entity_token_features import (
    EDGE_FEATURE_SIZE,
    EVENT_FEATURE_SIZE,
    GLOBAL_FEATURE_SIZE,
    HEX_FEATURE_SIZE,
    LEGAL_ACTION_FEATURE_SIZE,
    PLAYER_FEATURE_SIZE,
    VERTEX_FEATURE_SIZE,
    public_card_count_features_from_entity_tokens,
)
from catan_zero.rl.entity_token_features_rust import (
    _public_card_count_features_from_rust_game,
)
from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphNet


RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
DEV_CARDS = (
    "KNIGHT",
    "YEAR_OF_PLENTY",
    "MONOPOLY",
    "ROAD_BUILDING",
    "VICTORY_POINT",
)


def _counts(**values: int) -> dict[str, int]:
    return {name: int(values.get(name, 0)) for name in RESOURCES}


def _devs(**values: int) -> dict[str, int]:
    return {name: int(values.get(name, 0)) for name in DEV_CARDS}


def _payload(*, opponent_resources: dict[str, int], opponent_devs: dict[str, int]):
    return {
        "players": {
            "BLUE": {
                "resource_card_count": 5,
                "development_card_count": 1,
                "resources": _counts(WOOD=2, BRICK=1, WHEAT=1, ORE=1),
                "development_cards": _devs(KNIGHT=1),
                "played_development_cards": _devs(KNIGHT=1),
            },
            "RED": {
                "resource_card_count": 6,
                "development_card_count": 3,
                # These authoritative fields may accidentally be present in an
                # omniscient payload. The public feature builder must ignore
                # both identities completely.
                "resources": opponent_resources,
                "development_cards": opponent_devs,
                "played_development_cards": _devs(MONOPOLY=1),
            },
        },
        "bank": {
            "resources": _counts(
                WOOD=16,
                BRICK=15,
                SHEEP=18,
                WHEAT=17,
                ORE=18,
            ),
            "development_cards_remaining": 19,
        },
    }


def test_two_player_resources_are_exactly_recovered_from_public_conservation():
    table = public_card_count_feature_table(
        _payload(
            opponent_resources=_counts(WOOD=1, BRICK=3, SHEEP=1, WHEAT=1),
            opponent_devs=_devs(KNIGHT=2, VICTORY_POINT=1),
        ),
        "BLUE",
    )

    expected = np.asarray([1, 3, 1, 1, 0], dtype=np.float32) / 19.0
    np.testing.assert_allclose(table[1, 0:5], expected)
    assert table.shape == (4, 11)
    assert np.count_nonzero(table[0]) == 0


def test_opponent_hidden_truth_cannot_change_public_card_count_features():
    first = public_card_count_feature_table(
        _payload(
            opponent_resources=_counts(WOOD=1, BRICK=3, SHEEP=1, WHEAT=1),
            opponent_devs=_devs(KNIGHT=2, VICTORY_POINT=1),
        ),
        "BLUE",
    )
    second = public_card_count_feature_table(
        _payload(
            opponent_resources=_counts(ORE=6),
            opponent_devs=_devs(YEAR_OF_PLENTY=2, MONOPOLY=1),
        ),
        "BLUE",
    )
    assert np.array_equal(first, second)


def _masked_entity_card_tensors() -> tuple[np.ndarray, np.ndarray]:
    players = np.zeros((2, 4, PLAYER_FEATURE_SIZE), dtype=np.float32)
    globals_ = np.zeros((2, 1, GLOBAL_FEATURE_SIZE), dtype=np.float32)
    for row in range(2):
        players[row, 0, 0:2] = 1.0  # BLUE present + actor
        players[row, 1, 0] = 1.0  # RED present
        players[row, 0, 6] = 5.0 / 20.0
        players[row, 1, 6] = 6.0 / 20.0
        players[row, 0, 7] = 1.0 / 10.0
        players[row, 1, 7] = 3.0 / 10.0
        players[row, 0, 16:21] = np.asarray([2, 1, 0, 1, 1]) / 10.0
        players[row, 0, 22] = 1.0 / 5.0  # own KNIGHT
        players[row, 0, 27] = 1.0 / 5.0  # publicly played KNIGHT
        players[row, 1, 29] = 1.0 / 5.0  # publicly played MONOPOLY
        globals_[row, 0, 26:31] = np.asarray([16, 15, 18, 17, 18]) / 19.0
        globals_[row, 0, 31] = 19.0 / 25.0
    # Opponent private slots differ wildly between rows. They must be dead to
    # the public backfill path (and are normally already zero in masked data).
    players[0, 1, 4:6] = [1.0, 0.9]
    players[0, 1, 15:27] = np.linspace(0.1, 1.0, 12)
    players[1, 1, 4:6] = [0.0, 0.0]
    players[1, 1, 15:27] = 0.0
    return players, globals_


def test_vectorized_masked_shard_backfill_matches_public_payload_features():
    players, globals_ = _masked_entity_card_tensors()
    backfilled = public_card_count_features_from_entity_tokens(players, globals_)
    expected = public_card_count_feature_table(
        _payload(
            opponent_resources=_counts(ORE=6),
            opponent_devs=_devs(VICTORY_POINT=3),
        ),
        "BLUE",
    )
    np.testing.assert_allclose(backfilled[0], expected, atol=1.0e-6)
    # Strong privacy regression: changing every opponent hidden slot leaves the
    # tensor bit-identical.
    assert np.array_equal(backfilled[0], backfilled[1])


def test_historical_backfill_recovers_played_knights_past_clipped_slot():
    players, globals_ = _masked_entity_card_tensors()
    # Seven KNIGHTs + one MONOPOLY have been played; slot 27 can encode only
    # min(7/5, 1) but the public deck/hand conservation still identifies all 8.
    players[:, :, 27] = 0.0
    players[:, 0, 27] = 1.0
    globals_[:, 0, 31] = 13.0 / 25.0  # 25 - 4 held - 8 played
    features = public_card_count_features_from_entity_tokens(players, globals_)

    unknown_pool_total = 6 + 2 + 1 + 2 + 5
    expected_knights = (3.0 * 6.0 / unknown_pool_total) / 14.0
    assert features[0, 1, 5] == pytest.approx(expected_knights)
    assert np.array_equal(features[0], features[1])


def test_dev_identity_is_a_public_hypergeometric_posterior_not_hidden_truth():
    table = public_card_count_feature_table(
        _payload(
            opponent_resources=_counts(ORE=6),
            opponent_devs=_devs(VICTORY_POINT=3),
        ),
        "BLUE",
    )
    # Public unknown pool after own KNIGHT and played KNIGHT+MONOPOLY:
    # [12, 2, 1, 2, 5], 22 cards total; RED publicly holds three.
    expected_knights = (3.0 * 12.0 / 22.0) / 14.0
    assert table[1, 5] == pytest.approx(expected_knights)
    expected_vp_probability = 1.0 - (17 * 16 * 15) / (22 * 21 * 20)
    assert table[1, 10] == pytest.approx(expected_vp_probability)


class _FakeRustGame:
    def __init__(self, red_state: dict[str, object]) -> None:
        # Deliberately retain arbitrary secret truth only to prove the adapter
        # never calls either omniscient serialization method below.
        self.red_state = red_state

    def current_color(self) -> str:
        return "BLUE"

    def public_card_deductions_json(self, observer: str) -> str:
        assert observer == "BLUE"
        return json.dumps(
            {
                "contract": "public_card_deductions_2p_v1",
                "observer": "BLUE",
                "opponent": "RED",
                "observer_resources": _counts(
                    WOOD=2, BRICK=1, WHEAT=1, ORE=1
                ),
                "opponent_resources": _counts(
                    WOOD=1, BRICK=3, SHEEP=1, WHEAT=1
                ),
                "opponent_resource_card_count": 6,
                "resource_bank": _counts(
                    WOOD=16, BRICK=15, SHEEP=18, WHEAT=17, ORE=18
                ),
                "observer_development_cards": _devs(KNIGHT=1),
                "observer_development_card_count": 1,
                "opponent_face_down_development_card_count": 3,
                "development_deck_count": 19,
                "publicly_played_development_cards": _devs(
                    KNIGHT=1, MONOPOLY=1
                ),
                "unknown_development_pool": _devs(
                    KNIGHT=12,
                    YEAR_OF_PLENTY=2,
                    MONOPOLY=1,
                    ROAD_BUILDING=2,
                    VICTORY_POINT=5,
                ),
            }
        )

    def json_snapshot(self) -> str:
        raise AssertionError("public card adapter read omniscient snapshot")

    def player_state_json(self, color: str) -> str:
        raise AssertionError(f"public card adapter read hidden {color} state")


def test_rust_adapter_drops_opponent_hidden_identities_before_feature_build():
    common = {"played_dev_cards": _devs(MONOPOLY=1)}
    first = _public_card_count_features_from_rust_game(
        _FakeRustGame(
            {
                **common,
                "resources": _counts(WOOD=1, BRICK=3, SHEEP=1, WHEAT=1),
                "dev_cards": _devs(KNIGHT=2, VICTORY_POINT=1),
            }
        ),
        colors=("BLUE", "RED"),
    )
    second = _public_card_count_features_from_rust_game(
        _FakeRustGame(
            {
                **common,
                "resources": _counts(ORE=6),
                "dev_cards": _devs(YEAR_OF_PLENTY=2, MONOPOLY=1),
            }
        ),
        colors=("BLUE", "RED"),
    )
    assert np.array_equal(first, second)


def _config(**overrides) -> EntityGraphConfig:
    values = dict(
        action_size=32,
        static_action_feature_size=45,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=32,
        state_layers=1,
        attention_heads=4,
        dropout=0.0,
    )
    values.update(overrides)
    return EntityGraphConfig(**values)


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260714)
    batch: dict[str, torch.Tensor] = {}
    for name, count, width in (
        ("hex", 19, HEX_FEATURE_SIZE),
        ("vertex", 54, VERTEX_FEATURE_SIZE),
        ("edge", 72, EDGE_FEATURE_SIZE),
        ("player", 4, PLAYER_FEATURE_SIZE),
        ("global", 1, GLOBAL_FEATURE_SIZE),
        ("event", 0, EVENT_FEATURE_SIZE),
    ):
        batch[f"{name}_tokens"] = torch.randn(2, count, width, generator=generator)
        if name != "global":
            batch[f"{name}_mask"] = torch.ones(2, count, dtype=torch.bool)
    batch[DEDUCTION_FEATURES_KEY] = torch.rand(
        2, 4, DEDUCTION_FEATURE_SIZE, generator=generator
    )
    batch["legal_action_tokens"] = torch.randn(
        2, 5, LEGAL_ACTION_FEATURE_SIZE, generator=generator
    )
    batch["legal_action_context"] = torch.randn(
        2, 5, CONTEXT_ACTION_FEATURE_SIZE, generator=generator
    )
    return batch


def test_card_count_residual_is_function_preserving_then_trainable():
    torch.manual_seed(11)
    base = EntityGraphNet(_config()).eval()
    upgraded = EntityGraphNet(_config(public_card_count_features=True)).eval()
    missing, unexpected = upgraded.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert set(missing) == {
        "public_card_count_residual.bias",
        "public_card_count_residual.weight",
    }

    batch = _batch()
    with torch.no_grad():
        control = base(batch, return_q=True)
        initial = upgraded(batch, return_q=True)
    for key, expected in control.items():
        assert torch.equal(initial[key], expected), key

    with torch.no_grad():
        upgraded.public_card_count_residual.weight.normal_(std=0.1)
    changed = {key: value.clone() for key, value in batch.items()}
    changed[DEDUCTION_FEATURES_KEY].zero_()
    with torch.no_grad():
        left = upgraded(batch)["logits"]
        right = upgraded(changed)["logits"]
    assert not torch.equal(left, right)


def test_bias_free_card_count_residual_keeps_zero_input_exactly_zero_after_training():
    torch.manual_seed(19)
    base = EntityGraphNet(_config()).eval()
    upgraded = EntityGraphNet(
        _config(
            public_card_count_features=True,
            public_card_count_residual_bias=False,
        )
    ).eval()
    missing, unexpected = upgraded.load_state_dict(base.state_dict(), strict=False)
    assert unexpected == []
    assert missing == ["public_card_count_residual.weight"]
    assert upgraded.public_card_count_residual.bias is None

    batch = _batch()
    with torch.no_grad():
        control = base(batch, return_q=True)
        initial = upgraded(batch, return_q=True)
    for key, expected in control.items():
        assert torch.equal(initial[key], expected), key

    with torch.no_grad():
        upgraded.public_card_count_residual.weight.normal_(std=0.1)
        zero_features = torch.zeros_like(batch[DEDUCTION_FEATURES_KEY])
        zero_residual = upgraded.public_card_count_residual(zero_features)
    assert torch.count_nonzero(zero_residual).item() == 0

    changed = {key: value.clone() for key, value in batch.items()}
    changed[DEDUCTION_FEATURES_KEY].zero_()
    with torch.no_grad():
        populated_logits = upgraded(batch)["logits"]
        zero_input_logits = upgraded(changed)["logits"]
    assert not torch.equal(populated_logits, zero_input_logits)
