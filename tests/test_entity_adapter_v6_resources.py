from __future__ import annotations

import numpy as np
import pytest

from catan_zero.rl.entity_feature_adapter import (
    CURRENT_RUST_ENTITY_ADAPTER_VERSION,
    ENTITY_FEATURE_ADAPTER_SPECS,
    RUST_ENTITY_ADAPTER_V2,
    RUST_ENTITY_ADAPTER_V3,
    RUST_ENTITY_ADAPTER_V4,
    RUST_ENTITY_ADAPTER_V5,
    RUST_ENTITY_ADAPTER_V6,
)
from catan_zero.rl.entity_token_features import (
    EDGE_FEATURE_SIZE,
    EVENT_FEATURE_SIZE,
    GLOBAL_FEATURE_SIZE,
    HEX_FEATURE_SIZE,
    LEGAL_ACTION_FEATURE_SIZE,
    PLAYER_FEATURE_SIZE,
    VERTEX_FEATURE_SIZE,
    _player_tokens,
    public_card_count_features_from_entity_tokens,
)


RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")


def _counts(*values: int) -> dict[str, int]:
    return dict(zip(RESOURCES, values))


def _payload(resources: dict[str, int]) -> dict[str, object]:
    return {
        "current_player": "BLUE",
        "players": {
            "BLUE": {
                "resource_card_count": sum(resources.values()),
                "resources": resources,
            },
            "RED": {"resource_card_count": 1},
        },
    }


def test_v6_is_fresh_only_and_describes_both_semantic_repairs():
    assert CURRENT_RUST_ENTITY_ADAPTER_VERSION == RUST_ENTITY_ADAPTER_V3
    assert "exact_actor_resources" in RUST_ENTITY_ADAPTER_V6
    spec = ENTITY_FEATURE_ADAPTER_SPECS[RUST_ENTITY_ADAPTER_V6]
    assert spec.player_resource_counts == (
        "exact_physical_deck_scale_total_div95_composition_div19"
    )
    assert "initial_road_max_pips" in spec.context_road_expansion


def test_v2_through_v5_collision_is_frozen_but_v6_is_injective_for_proven_case():
    wood_heavy = _payload(_counts(11, 10, 0, 0, 0))
    brick_heavy = _payload(_counts(10, 11, 0, 0, 0))

    legacy_pairs = []
    for adapter in (
        RUST_ENTITY_ADAPTER_V2,
        RUST_ENTITY_ADAPTER_V3,
        RUST_ENTITY_ADAPTER_V4,
        RUST_ENTITY_ADAPTER_V5,
    ):
        first = _player_tokens(
            wood_heavy,
            "BLUE",
            entity_feature_adapter_version=adapter,
        )
        second = _player_tokens(
            brick_heavy,
            "BLUE",
            entity_feature_adapter_version=adapter,
        )
        legacy_pairs.append((first, second))
        assert np.array_equal(first, second)
    assert all(
        np.array_equal(legacy_pairs[0][index], pair[index])
        for pair in legacy_pairs[1:]
        for index in (0, 1)
    )

    first_v6 = _player_tokens(
        wood_heavy,
        "BLUE",
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6,
    )
    second_v6 = _player_tokens(
        brick_heavy,
        "BLUE",
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6,
    )
    assert not np.array_equal(first_v6, second_v6)
    for row in (first_v6[0], second_v6[0]):
        total = int(np.rint(float(row[6]) * 95.0))
        composition = np.rint(row[16:21].astype(np.float32) * 19.0).astype(int)
        assert total == int(composition.sum()) == 21


def test_v6_public_card_backfill_decodes_physical_scales_without_skew():
    players = _player_tokens(
        _payload(_counts(11, 10, 0, 0, 0)),
        "BLUE",
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6,
    ).astype(np.float32)
    globals_ = np.zeros((1, GLOBAL_FEATURE_SIZE), dtype=np.float32)
    globals_[0, 26:31] = np.asarray([8, 8, 19, 19, 19], dtype=np.float32) / 19.0

    features = public_card_count_features_from_entity_tokens(
        players,
        globals_,
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6,
    )
    np.testing.assert_allclose(
        features[1, 0:5],
        np.asarray([0, 1, 0, 0, 0], dtype=np.float32) / 19.0,
    )


def _minimal_torch_entity_batch(player_tokens):
    torch = pytest.importorskip("torch")
    batch_size = int(player_tokens.shape[0])
    return {
        "hex_tokens": torch.zeros(batch_size, 19, HEX_FEATURE_SIZE),
        "hex_mask": torch.ones(batch_size, 19, dtype=torch.bool),
        "vertex_tokens": torch.zeros(batch_size, 54, VERTEX_FEATURE_SIZE),
        "vertex_mask": torch.ones(batch_size, 54, dtype=torch.bool),
        "edge_tokens": torch.zeros(batch_size, 72, EDGE_FEATURE_SIZE),
        "edge_mask": torch.ones(batch_size, 72, dtype=torch.bool),
        "player_tokens": torch.as_tensor(player_tokens, dtype=torch.float32),
        "player_mask": torch.ones(batch_size, 4, dtype=torch.bool),
        "global_tokens": torch.zeros(batch_size, 1, GLOBAL_FEATURE_SIZE),
        "event_tokens": torch.zeros(batch_size, 0, EVENT_FEATURE_SIZE),
        "event_mask": torch.zeros(batch_size, 0, dtype=torch.bool),
        "legal_action_tokens": torch.zeros(
            batch_size, 1, LEGAL_ACTION_FEATURE_SIZE
        ),
        "legal_action_context": torch.zeros(batch_size, 1, 18),
    }


def test_v7_resource_route_reconstructs_exact_legacy_player_encoder_input():
    torch = pytest.importorskip("torch")
    from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphNet

    class CapturePlayerEncoder(torch.nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.width = width
            self.seen = None

        def forward(self, value):
            self.seen = value.detach().clone()
            return value.new_zeros((*value.shape[:2], self.width))

    physical = _player_tokens(
        _payload(_counts(11, 10, 0, 0, 0)),
        "BLUE",
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6,
    )[None, ...]
    legacy = _player_tokens(
        _payload(_counts(11, 10, 0, 0, 0)),
        "BLUE",
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V5,
    )[None, ...]
    model = EntityGraphNet(
        EntityGraphConfig(
            action_size=1,
            static_action_feature_size=1,
            hidden_size=16,
            state_layers=1,
            attention_heads=4,
            dropout=0.0,
            action_cross_attention_layers=1,
            v6_compatibility_preserving_inputs=True,
        )
    ).eval()
    capture = CapturePlayerEncoder(model.config.hidden_size)
    model.player_encoder = capture

    with torch.no_grad():
        model._state_tokens(_minimal_torch_entity_batch(physical))

    assert capture.seen is not None
    torch.testing.assert_close(
        capture.seen,
        torch.as_tensor(legacy, dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    assert torch.count_nonzero(model.v6_exact_resource_residual.weight) == 0


def test_v7_exact_resource_residual_weight_learns_on_first_backward():
    torch = pytest.importorskip("torch")
    from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphNet

    physical = _player_tokens(
        _payload(_counts(11, 10, 0, 0, 0)),
        "BLUE",
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6,
    )[None, ...]
    model = EntityGraphNet(
        EntityGraphConfig(
            action_size=1,
            static_action_feature_size=1,
            hidden_size=16,
            state_layers=1,
            attention_heads=4,
            dropout=0.0,
            action_cross_attention_layers=1,
            v6_compatibility_preserving_inputs=True,
        )
    )
    tokens, _mask, _history, _event_mask = model._state_tokens(
        _minimal_torch_entity_batch(physical)
    )
    tokens.sum().backward()

    gradient = model.v6_exact_resource_residual.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_v7_resource_route_preserves_inherited_public_card_residual_input():
    torch = pytest.importorskip("torch")
    from catan_zero.deduction_tracker import DEDUCTION_FEATURES_KEY
    from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphNet

    class CaptureResidual(torch.nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.width = width
            self.seen = None

        def forward(self, value):
            self.seen = value.detach().clone()
            return value.new_zeros((*value.shape[:2], self.width))

    payload = _payload(_counts(11, 10, 0, 0, 0))
    physical = _player_tokens(
        payload,
        "BLUE",
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6,
    )[None, ...]
    legacy = _player_tokens(
        payload,
        "BLUE",
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V5,
    )[None, ...]
    globals_ = np.zeros((1, 1, GLOBAL_FEATURE_SIZE), dtype=np.float32)
    globals_[0, 0, 26:31] = (
        np.asarray([8, 8, 19, 19, 19], dtype=np.float32) / 19.0
    )
    physical_features = public_card_count_features_from_entity_tokens(
        physical,
        globals_,
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6,
    )
    legacy_features = public_card_count_features_from_entity_tokens(
        legacy,
        globals_,
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V5,
    )
    assert not np.array_equal(physical_features, legacy_features)

    model = EntityGraphNet(
        EntityGraphConfig(
            action_size=1,
            static_action_feature_size=1,
            hidden_size=16,
            state_layers=1,
            attention_heads=4,
            dropout=0.0,
            action_cross_attention_layers=1,
            v6_compatibility_preserving_inputs=True,
            public_card_count_features=True,
            public_card_count_residual_bias=False,
        )
    ).eval()
    capture = CaptureResidual(model.config.hidden_size)
    model.public_card_count_residual = capture
    batch = _minimal_torch_entity_batch(physical)
    batch["global_tokens"] = torch.as_tensor(globals_)
    batch[DEDUCTION_FEATURES_KEY] = torch.as_tensor(physical_features)

    with torch.no_grad():
        model._state_tokens(batch)

    assert capture.seen is not None
    torch.testing.assert_close(
        capture.seen,
        torch.as_tensor(legacy_features, dtype=torch.float32),
        rtol=0,
        atol=0,
    )


def test_v6_training_batch_binds_decoder_and_admission_identity(monkeypatch):
    import tools.train_bc as train_bc

    observed: dict[str, str] = {}

    def fake_backfill(player_tokens, global_tokens, *, entity_feature_adapter_version):
        del global_tokens
        observed["adapter"] = entity_feature_adapter_version
        return np.zeros((player_tokens.shape[0], 4, 11), dtype=np.float32)

    monkeypatch.setattr(
        train_bc, "public_card_count_features_from_entity_tokens", fake_backfill
    )
    monkeypatch.setattr(train_bc, "_PUBLIC_CARD_COUNT_FEATURES_ENABLED", True)
    monkeypatch.setattr(train_bc, "_MASK_HIDDEN_INFO_PLAYER_TOKENS", False)
    monkeypatch.setattr(
        train_bc, "_TRAINING_ENTITY_FEATURE_ADAPTER_VERSION", RUST_ENTITY_ADAPTER_V6
    )

    data = {
        key: np.zeros((1, 1), dtype=np.float32)
        for key in train_bc.ENTITY_BATCH_KEYS
    }
    v6_players = _player_tokens(
        _payload(_counts(11, 10, 0, 0, 0)),
        "BLUE",
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6,
    )
    data["player_tokens"] = v6_players[None, ...]
    data["global_tokens"] = np.zeros(
        (1, 1, GLOBAL_FEATURE_SIZE), dtype=np.float32
    )
    train_bc._entity_batch(data, np.asarray([0], dtype=np.int64))
    assert observed["adapter"] == RUST_ENTITY_ADAPTER_V6

    authoritative = np.asarray([[11, 10, 0, 0, 0]], dtype=np.int16)
    assert train_bc.v6_actor_resource_identity_violations(
        data["player_tokens"], actor_resource_counts=authoritative
    ) == (0, 1)
    wrong_authoritative = authoritative.copy()
    wrong_authoritative[0, 0] -= 1
    wrong_authoritative[0, 2] += 1
    assert train_bc.v6_actor_resource_identity_violations(
        data["player_tokens"], actor_resource_counts=wrong_authoritative
    ) == (1, 1)
    broken = data["player_tokens"].copy()
    broken[0, 0, 6] = np.float16(20.0 / 95.0)
    assert train_bc.v6_actor_resource_identity_violations(broken) == (1, 1)
    nonfinite = data["player_tokens"].copy()
    nonfinite[0, 1, 2] = np.nan
    assert train_bc.v6_actor_resource_identity_violations(nonfinite) == (1, 1)


def test_v6_resource_decoder_rejects_unknown_adapter():
    with pytest.raises(ValueError, match="unknown entity feature adapter"):
        public_card_count_features_from_entity_tokens(
            np.zeros((4, PLAYER_FEATURE_SIZE), dtype=np.float32),
            np.zeros((1, GLOBAL_FEATURE_SIZE), dtype=np.float32),
            entity_feature_adapter_version="future-v99",
        )
