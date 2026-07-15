"""Pure contract tests for public-information chance beliefs."""

from __future__ import annotations

from copy import deepcopy
import json
import math

import pytest

from catan_zero.search.public_belief import (
    BASE_DEVELOPMENT_DECK,
    DEVELOPMENT_CARDS,
    OPPONENT_ACTION_SCOPE,
    PublicBelief,
    PublicBeliefSampler,
    RESOURCES,
)
from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    SearchResult,
)
from catan_zero.search.gumbel_chance_mcts import (
    belief_buy_development_card_outcomes,
    belief_move_robber_outcome_weights,
)


def _snapshot() -> dict:
    return {
        "colors": ["BLUE", "RED"],
        # Base-bank conservation with BLUE=[2,1,0,1,0] fixes the sole
        # opponent's hand exactly at RED=[0,4,0,0,0].
        "resource_bank": {
            "WOOD": 17,
            "BRICK": 14,
            "SHEEP": 19,
            "WHEAT": 18,
            "ORE": 19,
        },
        "development_deck_count": 19,
        "development_deck_order": ["KNIGHT", "VICTORY_POINT"],
        "player_state": [
            {
                "resources": {"WOOD": 2, "BRICK": 1, "SHEEP": 0, "WHEAT": 1, "ORE": 0},
                "resource_card_count": 4,
                "dev_cards": {"KNIGHT": 1, "VICTORY_POINT": 0},
                "development_card_count": 1,
                "played_dev_cards": {"KNIGHT": 1},
            },
            {
                "resources": {"WOOD": 0, "BRICK": 4, "SHEEP": 0, "WHEAT": 0, "ORE": 0},
                "resource_card_count": 4,
                "dev_cards": {"KNIGHT": 0, "VICTORY_POINT": 2},
                "development_card_count": 2,
                "played_dev_cards": {"MONOPOLY": 1},
            },
        ],
    }


def _permuted_opponent_truth() -> dict:
    snapshot = deepcopy(_snapshot())
    opponent = snapshot["player_state"][1]
    opponent["resources"] = {"WOOD": 1, "BRICK": 0, "SHEEP": 2, "WHEAT": 0, "ORE": 1}
    opponent["dev_cards"] = {"KNIGHT": 2, "VICTORY_POINT": 0}
    snapshot["development_deck_order"] = ["ROAD_BUILDING", "MONOPOLY"]
    return snapshot


def test_opponent_hidden_truth_does_not_enter_public_belief() -> None:
    first = PublicBelief.from_snapshot(_snapshot(), perspective="BLUE")
    second = PublicBelief.from_snapshot(_permuted_opponent_truth(), perspective="BLUE")

    assert first == second
    assert first.fingerprint() == second.fingerprint()
    assert first.robber_steal_probabilities("RED") == second.robber_steal_probabilities("RED")
    assert first.development_draw_probabilities() == second.development_draw_probabilities()


class _LegacyRobberGame:
    """Minimum engine surface for the fixed-five chance integration boundary."""

    def __init__(self, snapshot: dict) -> None:
        self.snapshot = snapshot

    def json_snapshot(self) -> str:
        return json.dumps(self.snapshot)

    def apply_chance_outcome(self, _action_json: str, outcome_index: int) -> tuple[str, int]:
        return ("materialized", int(outcome_index))


def test_legacy_robber_search_weights_are_hidden_truth_invariant() -> None:
    spectrum = tuple((index, 0.2) for index in range(5))
    action = ["BLUE", "MOVE_ROBBER", [[0, 0, 0], "RED"]]
    first = belief_move_robber_outcome_weights(
        _LegacyRobberGame(_snapshot()),
        action,
        cached_spectrum=spectrum,
        perspective="BLUE",
    )
    second = belief_move_robber_outcome_weights(
        _LegacyRobberGame(_permuted_opponent_truth()),
        action,
        cached_spectrum=spectrum,
        perspective="BLUE",
    )
    assert [(index, weight) for index, weight, _child in first] == [
        (index, weight) for index, weight, _child in second
    ] == [(1, 1.0)]


def test_two_player_opponent_robber_distribution_is_exact_by_conservation() -> None:
    belief = PublicBelief.from_snapshot(_snapshot(), perspective="BLUE")
    probabilities = belief.robber_steal_probabilities("RED")
    assert probabilities == {"BRICK": 1.0}


def test_legacy_snapshot_without_bank_keeps_uniform_fallback() -> None:
    snapshot = _snapshot()
    snapshot.pop("resource_bank")
    belief = PublicBelief.from_snapshot(snapshot, perspective="BLUE")
    probabilities = belief.robber_steal_probabilities("RED")
    assert tuple(probabilities) == RESOURCES
    assert all(math.isclose(value, 0.2) for value in probabilities.values())


def test_known_own_hand_robber_distribution_is_count_weighted() -> None:
    belief = PublicBelief.from_snapshot(_snapshot(), perspective="BLUE")
    assert belief.robber_steal_probabilities("BLUE") == {
        "WOOD": 0.5,
        "BRICK": 0.25,
        "WHEAT": 0.25,
    }


def test_dev_draw_uses_public_played_and_own_known_cards_only() -> None:
    belief = PublicBelief.from_snapshot(_snapshot(), perspective="BLUE")
    probabilities = belief.development_draw_probabilities()
    expected_counts = dict(BASE_DEVELOPMENT_DECK)
    expected_counts["KNIGHT"] -= 2  # own unplayed + publicly played
    expected_counts["MONOPOLY"] -= 1  # opponent publicly played
    denominator = sum(expected_counts.values())
    assert tuple(probabilities) == DEVELOPMENT_CARDS
    for card in DEVELOPMENT_CARDS:
        assert math.isclose(probabilities[card], expected_counts[card] / denominator)


class _PublicDevMaterializerGame:
    def __init__(self, snapshot: dict) -> None:
        self.snapshot = snapshot
        self.calls: list[tuple[str, str, tuple[str, ...], int]] = []

    def json_snapshot(self) -> str:
        return json.dumps(self.snapshot)

    def apply_public_belief_development_draws(
        self,
        action_json: str,
        observer: str,
        card_names: list[str],
        seed: int,
    ) -> list[tuple[str, str]]:
        self.calls.append((action_json, observer, tuple(card_names), seed))
        return [("public-child", card) for card in card_names]


def test_dev_draw_materializes_every_public_supported_card_without_native_deck() -> None:
    game = _PublicDevMaterializerGame(_snapshot())
    action = ["BLUE", "BUY_DEVELOPMENT_CARD", None]
    outcomes = belief_buy_development_card_outcomes(
        game,
        action,
        perspective="BLUE",
        materialization_seed=73,
    )
    assert [index for index, _weight, _child in outcomes] == list(range(5))
    assert [child for _index, _weight, child in outcomes] == [
        ("public-child", card) for card in DEVELOPMENT_CARDS
    ]
    assert game.calls == [
        (json.dumps(action), "BLUE", DEVELOPMENT_CARDS, 73)
    ]


def test_dev_draw_conditions_away_public_card_with_no_nonterminal_allocation() -> None:
    class ConstrainedMaterializer(_PublicDevMaterializerGame):
        def apply_public_belief_development_draws(
            self,
            action_json: str,
            observer: str,
            card_names: list[str],
            seed: int,
        ) -> list[tuple[str, str]]:
            self.calls.append((action_json, observer, tuple(card_names), seed))
            if "VICTORY_POINT" in card_names:
                # The PyO3 binding maps native engine errors through ``py_err``
                # to ValueError.  Exercise the real extension boundary here;
                # pure-Python test doubles may still raise RuntimeError.
                raise ValueError(
                    "no non-terminal hidden allocation can condition on requested dev draw"
                )
            return [("public-child", card) for card in card_names]

    game = ConstrainedMaterializer(_snapshot())
    action = ["BLUE", "BUY_DEVELOPMENT_CARD", None]
    outcomes = belief_buy_development_card_outcomes(
        game,
        action,
        perspective="BLUE",
        materialization_seed=73,
    )

    assert [DEVELOPMENT_CARDS[index] for index, _weight, _child in outcomes] == [
        card for card in DEVELOPMENT_CARDS if card != "VICTORY_POINT"
    ]
    assert math.isclose(sum(weight for _index, weight, _child in outcomes), 1.0)
    assert all(child[1] != "VICTORY_POINT" for _index, _weight, child in outcomes)


def test_keyed_sampling_is_reproducible_order_independent_and_truth_invariant() -> None:
    first = PublicBelief.from_snapshot(_snapshot(), perspective="BLUE")
    second = PublicBelief.from_snapshot(_permuted_opponent_truth(), perspective="BLUE")
    sampler = PublicBeliefSampler(seed=982451653)

    robber_forward = [sampler.sample_robber_steal(first, victim="RED", sample_index=i) for i in range(32)]
    robber_reverse = {
        i: sampler.sample_robber_steal(first, victim="RED", sample_index=i)
        for i in reversed(range(32))
    }
    assert robber_forward == [robber_reverse[i] for i in range(32)]
    assert robber_forward == [
        sampler.sample_robber_steal(second, victim="RED", sample_index=i) for i in range(32)
    ]
    assert [sampler.sample_development_draw(first, sample_index=i) for i in range(32)] == [
        sampler.sample_development_draw(second, sample_index=i) for i in range(32)
    ]


def test_seed_changes_sample_stream_without_flaky_single_draw_assertion() -> None:
    belief = PublicBelief.from_snapshot(_snapshot(), perspective="BLUE")
    stream_a = [PublicBeliefSampler(7).sample_development_draw(belief, sample_index=i) for i in range(64)]
    stream_b = [PublicBeliefSampler(8).sample_development_draw(belief, sample_index=i) for i in range(64)]
    assert stream_a != stream_b


def test_empty_public_source_returns_no_chance_outcome() -> None:
    snapshot = _snapshot()
    snapshot["player_state"][1]["resource_card_count"] = 0
    snapshot["development_deck_count"] = 0
    belief = PublicBelief.from_snapshot(snapshot, perspective="BLUE")
    sampler = PublicBeliefSampler(1)
    assert belief.robber_steal_probabilities("RED") == {}
    assert belief.development_draw_probabilities() == {}
    assert sampler.sample_robber_steal(belief, victim="RED") is None
    assert sampler.sample_development_draw(belief) is None


def test_scope_explicitly_does_not_claim_opponent_legal_action_fix() -> None:
    assert "Opponent legal actions" in OPPONENT_ACTION_SCOPE
    assert "does not claim that fix" in OPPONENT_ACTION_SCOPE


class _DeterminizedGame:
    def __init__(self, public_key: int, hidden_truth: int, particle_value: float = 0.0):
        self.public_key = public_key
        self.hidden_truth = hidden_truth
        self.particle_value = particle_value

    def current_color(self):
        return "BLUE"

    def num_turns(self):
        return 0

    def determinize_for_player(self, perspective: str, seed: int):
        assert perspective == "BLUE"
        # Deliberately exclude hidden_truth: this mimics the native API's public
        # contract and lets the test detect accidental authoritative use.
        value = ((self.public_key * 31 + seed) % 101) / 100.0
        return _DeterminizedGame(self.public_key, hidden_truth=-1, particle_value=value)


class _ParticleSearch(GumbelChanceMCTS):
    def __init__(self, config, evaluator=None):
        self.config = config
        self.evaluator = evaluator or type(
            "_PublicEvaluator",
            (),
            {"config": type("_Config", (), {"public_observation": True})()},
        )()
        import random

        self.rng = random.Random(config.seed)
        self._gumbel_rng = self.rng
        self._chance_rng = self.rng
        self._belief_rng = self.rng
        self._belief_materialization_seed = int(config.seed)

    def _fetch_legal_actions(self, game):
        return (1, 2), {}, {}

    def _search_single_world(
        self,
        game,
        *,
        force_full=None,
        n_simulations_override=None,
        attested_root_phase=None,
    ):
        p = float(game.particle_value)
        simulations = (
            int(n_simulations_override)
            if n_simulations_override is not None
            else 4
        )
        return SearchResult(
            selected_action=1 if p >= 0.5 else 2,
            improved_policy={1: p, 2: 1.0 - p},
            visit_counts={1: 3, 2: 1},
            q_values={1: p, 2: -p},
            priors={1: 0.6, 2: 0.4},
            root_value=p,
            used_full_search=bool(force_full),
            simulations_used=simulations,
        )


def test_information_set_particle_search_is_hidden_truth_invariant() -> None:
    config = GumbelChanceMCTSConfig(
        seed=19,
        information_set_search=True,
        determinization_particles=4,
        p_full=1.0,
        n_full=16,
    )
    first = _ParticleSearch(config).search(_DeterminizedGame(7, hidden_truth=1))
    second = _ParticleSearch(config).search(_DeterminizedGame(7, hidden_truth=999))
    assert first == second
    assert first.simulations_used == 16
    assert sum(first.improved_policy.values()) == pytest.approx(1.0)


def test_information_set_search_fails_without_native_api() -> None:
    search = _ParticleSearch(
        GumbelChanceMCTSConfig(information_set_search=True)
    )
    with pytest.raises(RuntimeError, match="determinize_for_player"):
        search.search(object())
