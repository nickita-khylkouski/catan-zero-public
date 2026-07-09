from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import mlflow_gate_logging as mgl  # type: ignore  # noqa: E402


_H2H_SUMMARY = {
    "checkpoint": "runs/bc/foo/checkpoint.pt",
    "n_full": 64,
    "max_decisions": 300,
    "pairs_requested": 50,
    "games_played": 100,
    "games_truncated": 20,
    "search_win_rate": 0.42,
    "split_rate": 0.30,
    "decisive_pair_yield": 0.70,
    "pairs_decisive": 35,
    "complete_pairs": 50,
    "pentanomial_sprt": {"llr": -0.99, "decision": "continue", "elo0": 0.0, "elo1": 30.0,
                         "alpha": 0.05, "beta": 0.05},
    "pair_sprt": {"llr": -0.45, "decision": "continue"},
}

_SCOREBOARD_SUMMARY = {
    "candidate": "runs/bc/cand/checkpoint.pt",
    "seed": 7,
    "results": [
        {"opponent": "catanatron_value", "wins": 55, "games": 100, "win_rate": 0.55},
        {"opponent": "catanatron_ab3", "wins": 40, "games": 100, "win_rate": 0.40},
    ],
}


def test_extract_params_includes_observation_mode_and_config() -> None:
    params = mgl.extract_params(_H2H_SUMMARY, observation_mode="omniscient")
    assert params["observation_mode"] == "omniscient"
    assert params["checkpoint"] == "runs/bc/foo/checkpoint.pt"
    assert params["n_full"] == 64
    assert params["sprt_elo1"] == 30.0


def test_extract_params_scoreboard_records_opponents() -> None:
    params = mgl.extract_params(_SCOREBOARD_SUMMARY, observation_mode="public")
    assert params["observation_mode"] == "public"
    assert params["opponents"] == "catanatron_value,catanatron_ab3"


def test_extract_metrics_h2h_includes_split_and_truncation() -> None:
    metrics = mgl.extract_metrics(_H2H_SUMMARY)
    assert metrics["split_rate"] == pytest.approx(0.30)
    assert metrics["decisive_pair_yield"] == pytest.approx(0.70)
    assert metrics["truncation_rate"] == pytest.approx(0.20)  # 20/100
    assert metrics["pentanomial_llr"] == pytest.approx(-0.99)
    assert metrics["concordant_llr"] == pytest.approx(-0.45)


def test_extract_metrics_scoreboard_per_opponent_and_overall() -> None:
    metrics = mgl.extract_metrics(_SCOREBOARD_SUMMARY)
    assert metrics["win_rate__catanatron_value"] == pytest.approx(0.55)
    assert metrics["win_rate__catanatron_ab3"] == pytest.approx(0.40)
    assert metrics["overall_win_rate"] == pytest.approx(95.0 / 200.0)


def test_extract_tags_carries_verdict_and_mode() -> None:
    tags = mgl.extract_tags(_H2H_SUMMARY, gate="search_vs_raw_h2h", observation_mode="omniscient")
    assert tags["gate"] == "search_vs_raw_h2h"
    assert tags["observation_mode"] == "omniscient"
    assert tags["pentanomial_decision"] == "continue"
    assert tags["concordant_decision"] == "continue"


def test_log_gate_run_rejects_bad_observation_mode() -> None:
    with pytest.raises(ValueError):
        mgl.log_gate_run(_H2H_SUMMARY, gate="g", observation_mode="xray")


def test_log_gate_run_is_fail_open_without_mlflow() -> None:
    # When mlflow is not installed the logger must warn and return None, never
    # raise -- capturing history must not be able to break a gate.
    try:
        import mlflow  # noqa: F401
    except Exception:
        run_id = mgl.log_gate_run(_H2H_SUMMARY, gate="g", observation_mode="omniscient")
        assert run_id is None
    else:
        pytest.skip("mlflow is installed; fail-open path not exercised")
