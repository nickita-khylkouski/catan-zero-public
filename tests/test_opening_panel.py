from __future__ import annotations

import random
import sys
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from opening_panel import (  # type: ignore  # noqa: E402
    _kendall_tau_b,
    _sample_from_priors,
    _shallow_root_trace,
    _validate_information_recipe,
    aggregate,
    build_panel,
    evaluate_root,
    reconstruct_roots,
)
import opening_panel as opening_panel_module  # type: ignore  # noqa: E402

from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    HeuristicRustEvaluator,
)
from catan_zero.search.rust_mcts import _require_rust_module


def _rust():
    try:
        return _require_rust_module()
    except RuntimeError as error:
        pytest.skip(str(error))


def test_kendall_tau_b_perfect_and_reversed():
    assert _kendall_tau_b([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert _kendall_tau_b([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_kendall_tau_b_all_ties_returns_none():
    assert _kendall_tau_b([1.0, 1.0, 1.0], [3.0, 2.0, 1.0]) is None
    assert _kendall_tau_b([1.0], [1.0]) is None


def test_aggregate_means_and_nulls():
    reports = [
        {"flipped": True, "raw_q_spread": 0.1, "spread_over_floor": 1.0, "kendall_tau": 0.5,
         "top1_regret": 0.2, "top3_coverage": 1.0},
        {"flipped": False, "raw_q_spread": 0.3, "spread_over_floor": None, "kendall_tau": None,
         "top1_regret": 0.0, "top3_coverage": 0.666},
    ]
    agg = aggregate(reports)
    assert agg["n_roots"] == 2
    assert agg["flip_rate"] == pytest.approx(0.5)
    assert agg["mean_raw_q_spread"] == pytest.approx(0.2)
    # None values are dropped from the mean.
    assert agg["mean_spread_over_floor"] == pytest.approx(1.0)
    assert agg["mean_kendall_tau"] == pytest.approx(0.5)


def test_sample_from_priors_stays_in_support():
    rng = random.Random(0)
    legal = (3, 7, 9)
    priors = {3: 0.0, 7: 1.0, 9: 0.0}
    for _ in range(20):
        assert _sample_from_priors(priors, legal, rng) == 7
    # Zero total falls back to a legal choice.
    assert _sample_from_priors({}, legal, rng) in legal


def test_shallow_trace_uses_public_search_result_not_private_root_expansion():
    game = SimpleNamespace(current_color=lambda: "RED")
    result = SimpleNamespace(
        selected_action=7,
        improved_policy={3: 0.25, 7: 0.75},
        visit_counts={3: 2, 7: 6},
        q_values={3: -0.1, 7: 0.2},
        priors={3: 0.6, 7: 0.4},
        root_value=0.05,
        simulations_used=8,
    )

    class SearchOnly:
        def __init__(self):
            self.calls = []

        def search(self, received, *, force_full):
            self.calls.append((received, force_full))
            return result

    mcts = SearchOnly()
    trace = _shallow_root_trace(mcts, game)

    assert mcts.calls == [(game, True)]
    assert trace["selected_action"] == 7
    assert trace["simulations_used"] == 8
    assert trace["per_candidate"][7]["ranking_score"] == pytest.approx(0.75)
    assert trace["per_candidate"][3]["raw_q"] == pytest.approx(-0.1)


def test_opening_panel_contains_no_private_gnode_bypass() -> None:
    source = inspect.getsource(opening_panel_module)
    assert "_GNode(" not in source
    assert "._run_root_search(" not in source


def test_public_opening_panel_requires_information_set_search() -> None:
    args = SimpleNamespace(
        public_observation=True,
        information_set_search=False,
        determinization_particles=4,
        determinization_min_simulations=32,
    )
    with pytest.raises(ValueError, match="information-set-search together"):
        _validate_information_recipe(args)

    args.information_set_search = True
    _validate_information_recipe(args)


def test_build_panel_reconstruct_is_deterministic():
    catanatron_rs = _rust()
    panel = build_panel(catanatron_rs, n_roots=3, base_seed=600001, min_settlement_candidates=40)
    assert panel["n_roots"] == 3
    assert len(panel["seeds"]) == 3
    # Panel roots are genuinely wide placement decisions.
    roots = reconstruct_roots(catanatron_rs, panel)
    import json

    for game in roots:
        legal = json.loads(game.playable_actions_json())
        settlements = [a for a in legal if a[1] == "BUILD_SETTLEMENT"]
        assert len(settlements) >= 40
    # Reconstruction is deterministic in the seed.
    again = reconstruct_roots(catanatron_rs, panel)
    assert [g.playable_actions_json() for g in roots] == [g.playable_actions_json() for g in again]


def test_evaluate_root_smoke_deep_search_oracle():
    catanatron_rs = _rust()
    panel = build_panel(catanatron_rs, n_roots=1, base_seed=600001, min_settlement_candidates=40)
    game = reconstruct_roots(catanatron_rs, panel)[0]
    evaluator = HeuristicRustEvaluator(score_actions=False)
    mcts = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=1, n_full=8, n_fast=8, p_full=1.0, temperature=0.0), evaluator
    )
    report = evaluate_root(
        mcts,
        evaluator,
        game.copy(),
        top_k=3,
        oracle="deep_search",
        oracle_sims=8,
        oracle_rollouts=2,
        rollout_max_steps=50,
        seed=600001,
    )
    assert report["n_candidates"] >= 40
    assert isinstance(report["flipped"], bool)
    assert 0.0 <= report["top3_coverage"] <= 1.0
    assert report["kendall_tau"] is None or -1.0 <= report["kendall_tau"] <= 1.0
    assert report["raw_q_spread"] >= 0.0


def test_evaluate_root_without_oracle_keeps_real_spread_and_nulls_ranking():
    catanatron_rs = _rust()
    panel = build_panel(catanatron_rs, n_roots=1, base_seed=600001, min_settlement_candidates=40)
    game = reconstruct_roots(catanatron_rs, panel)[0]
    evaluator = HeuristicRustEvaluator(score_actions=False)
    mcts = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=1, n_full=4, n_fast=4, p_full=1.0, temperature=0.0),
        evaluator,
    )
    report = evaluate_root(
        mcts,
        evaluator,
        game.copy(),
        top_k=1,
        oracle="none",
        oracle_sims=1,
        oracle_rollouts=1,
        rollout_max_steps=1,
        seed=600001,
    )
    assert report["raw_q_spread"] >= 0.0
    assert report["kendall_tau"] is None
    assert report["top1_regret"] is None
    assert report["top3_coverage"] is None
