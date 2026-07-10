from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from sigma_trace_placement_root import (  # type: ignore  # noqa: E402
    _parse_configs,
    find_placement_roots,
    run_sweep,
    trace_one_root,
)

from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTS, GumbelChanceMCTSConfig, HeuristicRustEvaluator
from catan_zero.search.rust_mcts import _require_rust_module


def _rust():
    try:
        return _require_rust_module()
    except RuntimeError as error:
        pytest.skip(str(error))


def test_parse_configs():
    assert _parse_configs("50:0.1,50:0.03,1:0.1") == ((50.0, 0.1), (50.0, 0.03), (1.0, 0.1))
    assert _parse_configs("50:0.1") == ((50.0, 0.1),)


def test_trace_refuses_public_observation_private_tree_bypass():
    mcts = SimpleNamespace(
        evaluator=SimpleNamespace(config=SimpleNamespace(public_observation=True))
    )
    with pytest.raises(RuntimeError, match="refuses public_observation"):
        trace_one_root(mcts, object())


def test_find_placement_roots_returns_real_wide_settlement_decisions():
    catanatron_rs = _rust()
    states = find_placement_roots(catanatron_rs, n_states=3, base_seed=1)
    assert len(states) == 3
    for game in states:
        legal = game.playable_actions_json()
        import json

        settlement_candidates = [a for a in json.loads(legal) if a[1] == "BUILD_SETTLEMENT"]
        assert len(settlement_candidates) >= 40


def test_trace_one_root_reports_well_formed_per_candidate_data():
    catanatron_rs = _rust()
    states = find_placement_roots(catanatron_rs, n_states=1, base_seed=1)
    evaluator = HeuristicRustEvaluator(score_actions=False)
    config = GumbelChanceMCTSConfig(seed=0, n_full=16, n_fast=16, p_full=1.0, temperature=0.0)
    mcts = GumbelChanceMCTS(config, evaluator)

    trace = trace_one_root(mcts, states[0].copy())

    assert trace["n_candidates"] >= 40
    assert trace["n_visited"] <= trace["n_candidates"]
    assert isinstance(trace["flipped"], bool)
    for candidate in trace["per_candidate"]:
        assert 0.0 <= candidate["rescaled_q"] <= 1.0 + 1.0e-6
        assert candidate["visits"] >= 0
    # argmax must actually come from the candidate set.
    action_ids = {c["action_id"] for c in trace["per_candidate"]}
    assert trace["prior_argmax"] in action_ids
    assert trace["search_argmax"] in action_ids


def test_run_sweep_produces_one_entry_per_config_with_correct_keys():
    catanatron_rs = _rust()
    states = find_placement_roots(catanatron_rs, n_states=2, base_seed=1)
    evaluator = HeuristicRustEvaluator(score_actions=False)

    results = run_sweep(
        states,
        evaluator,
        configs=((50.0, 0.1), (1.0, 0.1)),
        n_full=16,
        seed_base=1,
    )

    assert set(results.keys()) == {"cv50.0_cs0.1", "cv1.0_cs0.1"}
    for entry in results.values():
        assert entry["n_states"] == 2
        assert 0 <= entry["flips"] <= 2
        assert len(entry["traces"]) == 2
