from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from catan_zero.rl.entity_feature_adapter import (
    RUST_ENTITY_ADAPTER_V2,
    RUST_ENTITY_ADAPTER_V5,
)
from catan_zero.rl.meaningful_history import (
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V1,
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2,
)
from tools import generate_rust_mcts_reanalysis as reanalysis
from tools.generate_rust_mcts_reanalysis import _target_scores_and_mask


def test_unvisited_action_is_excluded_even_though_q_defaults_to_a_finite_zero() -> None:
    """FIX (Q-mask): RustMCTSResult.q_values defaults an unvisited action's Q to 0.0 (finite,
    not NaN), so isfinite alone can't tell an unvisited action from a genuinely-scored one at
    Q==0. The mask must also require visits > 0."""
    legal_rust = (10, 20, 30)
    q_by_rust = {10: 0.4, 20: 0.0, 30: -0.2}  # action 20 has the "looks legal" finite 0.0 Q
    visits_by_rust = {10: 5, 20: 0, 30: 3}  # action 20 was never actually visited

    target_scores, mask = _target_scores_and_mask(q_by_rust, visits_by_rust, legal_rust)

    np.testing.assert_allclose(target_scores, [0.4, 0.0, -0.2])
    assert mask.tolist() == [True, False, True]


def test_missing_q_value_stays_excluded_via_nan() -> None:
    legal_rust = (10, 20)
    q_by_rust = {10: 0.4}  # action 20 never got a q entry at all
    visits_by_rust = {10: 5, 20: 5}

    target_scores, mask = _target_scores_and_mask(q_by_rust, visits_by_rust, legal_rust)

    assert np.isnan(target_scores[1])
    assert mask.tolist() == [True, False]


def test_all_visited_matches_plain_isfinite_behavior() -> None:
    legal_rust = (1, 2, 3)
    q_by_rust = {1: 0.1, 2: 0.2, 3: 0.3}
    visits_by_rust = {1: 1, 2: 4, 3: 2}

    target_scores, mask = _target_scores_and_mask(q_by_rust, visits_by_rust, legal_rust)

    assert mask.tolist() == [True, True, True]


def test_full_coverage_requires_every_legal_action_visited() -> None:
    """soft_score_legal_coverage == 1.0 requires every legal action to be BOTH scored and
    visited; this documents that a single unvisited action breaks full coverage."""
    legal_rust = tuple(range(18))
    q_by_rust = {action: 0.0 for action in legal_rust}
    visits_by_rust = {action: 1 for action in legal_rust}
    visits_by_rust[7] = 0  # one root child never got expanded

    _, mask = _target_scores_and_mask(q_by_rust, visits_by_rust, legal_rust)

    assert mask.sum() == 17
    assert not mask[7]


def _fake_evaluator(
    *,
    adapter_version: str,
    public_observation: bool,
    meaningful_public_history: bool,
    meaningful_public_history_schema: str,
    event_history_limit: int,
    context_fill: float = 0.0,
):
    return SimpleNamespace(
        policy=SimpleNamespace(
            action_size=607,
            entity_feature_adapter_version=adapter_version,
            config=SimpleNamespace(
                meaningful_public_history=meaningful_public_history,
                meaningful_public_history_schema=(
                    meaningful_public_history_schema
                ),
                event_history_limit=event_history_limit,
            ),
        ),
        config=SimpleNamespace(
            entity_feature_adapter_version=adapter_version,
            public_observation=public_observation,
            context_fill=context_fill,
        ),
    )


def test_legacy_reanalysis_rows_use_checkpoint_bound_feature_semantics(
    monkeypatch,
) -> None:
    class Game:
        @staticmethod
        def current_color():
            return "BLUE"

        @staticmethod
        def player_state_json(_color):
            return json.dumps({"victory_points": 0})

        @staticmethod
        def json_snapshot():
            return json.dumps({"current_prompt": "PLAY_TURN"})

    evaluator = _fake_evaluator(
        adapter_version=RUST_ENTITY_ADAPTER_V2,
        public_observation=False,
        meaningful_public_history=False,
        meaningful_public_history_schema=MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V1,
        event_history_limit=17,
        context_fill=-3.5,
    )
    search = SimpleNamespace(
        search=lambda _game: SimpleNamespace(
            policy={11: 1.0},
            q_values={11: 0.25},
            visits={11: 1},
            action=11,
        )
    )
    observed: dict[str, dict] = {}

    monkeypatch.setattr(
        reanalysis,
        "rust_policy_action_ids",
        lambda *_args, **_kwargs: (23,),
    )

    def entity_features(*_args, **kwargs):
        observed["entity"] = kwargs
        return {"global_tokens": np.zeros((1, 1, 1), dtype=np.float32)}

    def action_context(*_args, **kwargs):
        observed["context"] = kwargs
        return np.zeros((1, 1, 1), dtype=np.float32)

    monkeypatch.setattr(reanalysis, "rust_game_to_entity_batch", entity_features)
    monkeypatch.setattr(reanalysis, "rust_action_context_batch", action_context)

    row = reanalysis._mcts_row(
        Game(),
        search=search,
        evaluator=evaluator,
        legal_rust=(11,),
        candidate_color="BLUE",
        game_seed=7,
        decision_index=3,
        obs_width=8,
    )

    assert row["adapter_version"] == RUST_ENTITY_ADAPTER_V2
    assert observed["entity"]["entity_feature_adapter_version"] == (
        RUST_ENTITY_ADAPTER_V2
    )
    assert observed["context"]["entity_feature_adapter_version"] == (
        RUST_ENTITY_ADAPTER_V2
    )
    assert observed["entity"]["public_observation"] is False
    assert observed["context"]["public_observation"] is False
    assert observed["context"]["fill"] == -3.5
    assert observed["entity"]["meaningful_public_history"] is False
    assert observed["entity"]["meaningful_public_history_schema"].endswith("_v1")
    assert observed["entity"]["history_limit"] == 17


def test_commissioned_v5_contract_clamps_history_and_serializes_manifest() -> None:
    evaluator = _fake_evaluator(
        adapter_version=RUST_ENTITY_ADAPTER_V5,
        public_observation=True,
        meaningful_public_history=True,
        meaningful_public_history_schema=MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2,
        event_history_limit=999,
        context_fill=-1.25,
    )

    contract = reanalysis._evaluator_feature_contract(evaluator)

    assert contract == {
        "entity_feature_adapter_version": RUST_ENTITY_ADAPTER_V5,
        "public_observation": True,
        "action_context_fill": -1.25,
        "meaningful_public_history": True,
        "meaningful_public_history_schema": (
            MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
        ),
        "event_history_limit": 64,
    }
    assert reanalysis._feature_contract_manifest_fields(contract) == {
        "adapter_version": RUST_ENTITY_ADAPTER_V5,
        "public_observation": True,
        "action_context_fill": -1.25,
        "meaningful_public_history": True,
        "meaningful_public_history_schema": (
            MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
        ),
        "event_history_limit": 64,
    }


def test_reanalysis_rejects_runtime_checkpoint_adapter_drift() -> None:
    evaluator = _fake_evaluator(
        adapter_version=RUST_ENTITY_ADAPTER_V2,
        public_observation=False,
        meaningful_public_history=False,
        meaningful_public_history_schema=MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V1,
        event_history_limit=0,
    )
    evaluator.config.entity_feature_adapter_version = RUST_ENTITY_ADAPTER_V5

    with pytest.raises(RuntimeError, match="feature adapter drift"):
        reanalysis._evaluator_feature_contract(evaluator)
