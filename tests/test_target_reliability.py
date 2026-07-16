from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from catan_zero.rl import gumbel_self_play
from catan_zero.rl.target_reliability import (
    TARGET_RELIABILITY_VERSION,
    duplicate_search_reliability_fields,
    jensen_shannon_divergence,
    target_reliability_root_seed,
    target_reliability_root_selected,
    unaudited_target_reliability_fields,
)
from catan_zero.search.rng_streams import domain_separated_search_seed


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import train_bc  # noqa: E402
from catan_zero.search.gumbel_chance_mcts import (  # noqa: E402
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
)


def test_duplicate_search_reliability_math_is_bounded_and_scale_explicit() -> None:
    identical = jensen_shannon_divergence({1: 0.8, 2: 0.2}, {1: 0.8, 2: 0.2})
    disjoint = jensen_shannon_divergence({1: 1.0}, {2: 1.0})
    assert identical == pytest.approx(0.0, abs=1e-12)
    assert disjoint == pytest.approx(math.log(2.0), abs=1e-9)

    evidence = duplicate_search_reliability_fields(
        primary_policy={3: 0.7, 8: 0.3},
        duplicate_policy={3: 0.6, 8: 0.4},
        primary_completed_q={3: 0.22, 8: 0.20},
        duplicate_completed_q={3: 0.18, 8: 0.17},
    )
    assert int(evidence["target_reliability_version"]) == TARGET_RELIABILITY_VERSION
    assert bool(evidence["target_reliability_audited"])
    assert bool(evidence["target_reliability_policy_top1_agreement"])
    assert bool(evidence["target_reliability_q_top1_agreement"])
    assert float(evidence["target_reliability_q_margin_primary"]) == pytest.approx(
        0.02
    )
    assert float(evidence["target_reliability_q_margin_duplicate"]) == pytest.approx(
        0.01
    )
    assert 0.0 < float(evidence["target_reliability_confidence"]) <= 1.0


def test_audit_selector_and_three_search_streams_are_domain_separated() -> None:
    kwargs = dict(game_seed=701, decision_index=19, audit_seed=44)
    assert target_reliability_root_selected(**kwargs, audit_fraction=1.0)
    assert not target_reliability_root_selected(**kwargs, audit_fraction=0.0)
    assert target_reliability_root_selected(
        **kwargs, audit_fraction=0.37
    ) == target_reliability_root_selected(**kwargs, audit_fraction=0.37)

    root_seed = target_reliability_root_seed(**kwargs)
    streams = {
        domain_separated_search_seed(root_seed, name)
        for name in ("gumbel", "chance", "belief")
    }
    assert len(streams) == 3
    assert target_reliability_root_seed(**kwargs) == root_seed


def test_default_search_reseed_preserves_legacy_materialization_seed() -> None:
    # Avoid constructing a live engine: this regression is only about the RNG
    # reset contract.  Production resets the gameplay stream per game but
    # historically kept public-belief materialization bound to config.seed.
    search = GumbelChanceMCTS.__new__(GumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(seed=17)
    search.rng = random.Random(17)
    search.seed_search_rngs(999)
    assert search._gumbel_rng is search.rng
    assert search._chance_rng is search.rng
    assert search._belief_rng is search.rng
    assert search._belief_materialization_seed == 17

    search.config = GumbelChanceMCTSConfig(seed=17, rng_stream_separation=True)
    search.seed_search_rngs(999)
    assert len({
        search._gumbel_rng.getrandbits(64),
        search._chance_rng.getrandbits(64),
        search._belief_rng.getrandbits(64),
    }) == 3
    assert search._belief_materialization_seed == domain_separated_search_seed(
        999, "belief"
    )


def test_game_search_reseed_removes_worker_seed_from_materialization() -> None:
    searches = []
    for worker_seed in (17, 23):
        search = GumbelChanceMCTS.__new__(GumbelChanceMCTS)
        search.config = GumbelChanceMCTSConfig(seed=worker_seed)
        search.rng = random.Random(worker_seed)
        search.seed_game_search_rngs(999)
        searches.append(search)

    expected = domain_separated_search_seed(999, "belief")
    assert [search._boundary_value_base_seed for search in searches] == [999, 999]
    assert [
        search._belief_materialization_seed for search in searches
    ] == [expected, expected]


def _entity_features(legal_width: int) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    for key in gumbel_self_play.ENTITY_KEYS:
        if key == "legal_action_tokens":
            features[key] = np.zeros((legal_width, 3), dtype=np.float16)
        elif key == "legal_action_target_ids":
            features[key] = np.full((legal_width, 4), -1, dtype=np.int16)
        elif key == "legal_action_mask":
            features[key] = np.ones(legal_width, dtype=np.bool_)
        else:
            features[key] = np.zeros((1,), dtype=np.float16)
    return features


def test_reliability_fields_round_trip_through_shard_array_schema() -> None:
    audited = duplicate_search_reliability_fields(
        primary_policy={2: 0.55, 7: 0.45},
        duplicate_policy={2: 0.52, 7: 0.48},
        primary_completed_q={2: 0.12, 7: 0.10},
        duplicate_completed_q={2: 0.11, 7: 0.105},
    )
    rows = []
    for evidence in (audited, unaudited_target_reliability_fields()):
        row = {
            "legal_action_ids": np.asarray([2, 7], dtype=np.int16),
            "policy_weight_multiplier": np.float32(1.0),
            **_entity_features(2),
            **evidence,
        }
        rows.append(row)
    arrays = gumbel_self_play._rows_to_arrays(rows)
    assert arrays["target_reliability_version"].dtype == np.uint8
    assert arrays["target_reliability_audited"].dtype == np.bool_
    assert arrays["target_reliability_js_divergence"].dtype == np.float32
    assert arrays["target_reliability_audited"].tolist() == [True, False]
    assert np.isnan(arrays["target_reliability_js_divergence"][1])
    assert arrays["target_reliability_confidence"].tolist()[1] == 1.0


def test_trainer_confidence_weighting_is_off_by_default_and_contract_checked() -> None:
    evidence = duplicate_search_reliability_fields(
        primary_policy={1: 0.7, 2: 0.3},
        duplicate_policy={1: 0.6, 2: 0.4},
        primary_completed_q={1: 0.2, 2: 0.1},
        duplicate_completed_q={1: 0.19, 2: 0.11},
    )
    neutral = unaudited_target_reliability_fields()
    data = {
        "action_taken": np.asarray([1, 2, 1], dtype=np.int16),
        "target_information_regime": np.asarray(
            [
                "public_belief_single_tree_v1",
                "public_belief_single_tree_v1",
                "authoritative_hidden_state_search_v1",
            ]
        ),
    }
    for key in evidence:
        data[key] = np.asarray(
            [evidence[key], neutral[key], neutral[key]],
            dtype=np.asarray(evidence[key]).dtype,
        )
    data["target_reliability_version"][2] = 0

    np.testing.assert_array_equal(
        train_bc.target_reliability_policy_factors(
            {"action_taken": np.asarray([], dtype=np.int16)},
            enabled=False,
            confidence_floor=0.25,
        ),
        np.ones(0, dtype=np.float32),
    )
    factors = train_bc.target_reliability_policy_factors(
        data, enabled=True, confidence_floor=0.25
    )
    assert factors[0] == pytest.approx(
        max(0.25, float(evidence["target_reliability_confidence"]))
    )
    assert factors[1:].tolist() == [1.0, 1.0]

    data["target_reliability_confidence"][0] = 0.99
    with pytest.raises(SystemExit, match="differs from its version-1 formula"):
        train_bc.target_reliability_policy_factors(
            data, enabled=True, confidence_floor=0.25
        )
