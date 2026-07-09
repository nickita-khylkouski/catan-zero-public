from __future__ import annotations

import json

from tools.update_population_payoffs import (
    append_payoff_entries,
    build_payoff_entries,
    infer_profile_from_summary,
    policy_id_from_path,
)
from tools.summarize_population_payoffs import summarize_payoffs


def test_build_payoff_entries_from_remote_grade_summary() -> None:
    summary = {
        "decisions": [
            {
                "checkpoint": "runs/self_play/s9776_candidate.iter0002.pt",
                "champion": "runs/self_play/champions/current_best_s9752_iter0002.pt",
                "candidate_weighted_win_rate": 0.25,
                "champion_weighted_win_rate": 0.20,
                "decision": "keep_for_training",
                "paired_delta": {
                    "aggregate_delta": 0.05,
                    "aggregate_lower_delta": 0.01,
                    "worst_opponent": "value",
                    "opponents": {
                        "heuristic": {
                            "candidate_games": 4,
                            "candidate_win_rate": 0.5,
                            "candidate_wins": 2,
                            "champion_games": 4,
                            "champion_win_rate": 0.25,
                            "champion_wins": 1,
                            "lower_delta": 0.10,
                            "win_rate_delta": 0.25,
                        },
                        "value": {
                            "candidate_games": 4,
                            "candidate_win_rate": 0.0,
                            "candidate_wins": 0,
                            "champion_games": 4,
                            "champion_win_rate": 0.25,
                            "champion_wins": 1,
                            "lower_delta": -0.05,
                            "win_rate_delta": -0.25,
                        },
                    },
                },
                "reason": "aggregate +0.0500",
                "summary": "summary_s9776_candidate.iter0002_strict_g4_r1_w1_vp4_d300_to1200_current_best_s9752_iter0002_abcd.json",
                "summary_games": 4,
                "worker": "catan-zero-c1",
                "zone": "us-central1-c",
            }
        ],
        "legs": [
            {
                "checkpoint": "s9776_candidate.iter0002",
                "worker": "catan-zero-c1",
                "opponents": {
                    "heuristic": {"wins": 1, "games": 4, "win_rate": 0.25},
                    "jsettlers_lite": {"wins": 2, "games": 4, "win_rate": 0.5},
                },
            }
        ],
    }

    entries = build_payoff_entries(summary, run_label="strict-smoke")

    pair = entries[0]
    assert pair["type"] == "grade_pair"
    assert pair["policy_id"] == "s9776_candidate.iter0002"
    assert pair["opponent_id"] == "current_best_s9752_iter0002"
    assert pair["profile"] == "strict"
    assert pair["delta"] == 0.04999999999999999
    assert pair["paired_aggregate_delta"] == 0.05
    assert pair["paired_aggregate_lower_delta"] == 0.01
    assert pair["paired_worst_opponent"] == "value"
    assert pair["run_label"] == "strict-smoke"

    opponent_pairs = {
        entry["opponent_id"]: entry
        for entry in entries
        if entry["type"] == "grade_opponent_pair"
    }
    assert set(opponent_pairs) == {"heuristic", "value"}
    assert opponent_pairs["heuristic"]["delta"] == 0.25
    assert opponent_pairs["value"]["delta"] == -0.25
    assert opponent_pairs["value"]["champion_wins"] == 1

    leg_opponents = {entry["opponent_id"] for entry in entries if entry["type"] == "grade_leg"}
    assert leg_opponents == {"heuristic", "jsettlers_lite"}


def test_build_payoff_entries_keeps_early_reject_without_champion_score() -> None:
    summary = {
        "decisions": [
            {
                "checkpoint": "runs/self_play/s9777_bad.pt",
                "champion": "runs/self_play/champions/current_best_s9752_iter0002.pt",
                "candidate_weighted_win_rate": 0.0,
                "champion_weighted_win_rate": None,
                "decision": "reject",
                "reason": "candidate weighted win rate 0.0000 below threshold 0.0000",
                "summary": "summary_s9777_bad_strict_g4_r1_w1_vp4_d300_to1200_current_best_s9752_iter0002_abcd.json",
                "summary_games": 4,
                "worker": "catan-zero-c1",
                "zone": "us-central1-c",
            }
        ],
        "legs": [],
    }

    entries = build_payoff_entries(summary, run_label="strict-smoke")

    assert len(entries) == 1
    assert entries[0]["type"] == "grade_pair"
    assert entries[0]["score"] == 0.0
    assert entries[0]["opponent_score"] is None
    assert entries[0]["delta"] == -1.0
    assert entries[0]["decision"] == "reject"


def test_append_payoff_entries_dedupes_by_key(tmp_path) -> None:
    output = tmp_path / "payoffs.jsonl"
    entries = [
        {"key": "a", "policy_id": "p1"},
        {"key": "b", "policy_id": "p2"},
    ]

    assert append_payoff_entries(output, entries, dedupe_existing=True) == 2
    assert append_payoff_entries(output, entries, dedupe_existing=True) == 0

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == entries


def test_profile_and_policy_id_helpers() -> None:
    assert infer_profile_from_summary("summary_x_strict_g4.json") == "strict"
    assert infer_profile_from_summary("summary_x_search_stress_g4.json") == "search_stress"
    assert infer_profile_from_summary("summary_x_jsettlers_triage_g2.json") == "jsettlers_triage"
    assert infer_profile_from_summary("summary_x.json") == "unknown"
    assert policy_id_from_path("runs/self_play/foo.pt") == "foo"


def test_summarize_payoffs_ranks_policy_and_surfaces_weak_opponents() -> None:
    rows = [
        {
            "type": "grade_pair",
            "policy_id": "candidate_a",
            "opponent_id": "current_best_s9752_iter0002",
            "profile": "strict",
            "score": 0.25,
            "opponent_score": 0.20,
            "delta": 0.05,
            "decision": "promote_candidate",
            "summary_games": 4,
            "worker": "catan-zero-c1",
        },
        {
            "type": "grade_pair",
            "policy_id": "candidate_b",
            "opponent_id": "current_best_s9752_iter0002",
            "profile": "strict",
            "score": 0.10,
            "opponent_score": 0.20,
            "delta": -0.10,
            "decision": "reject",
            "summary_games": 4,
            "worker": "catan-zero-c2",
        },
        {
            "type": "grade_leg",
            "policy_id": "candidate_a",
            "opponent_id": "heuristic",
            "wins": 2,
            "games": 4,
            "worker": "catan-zero-c1",
        },
        {
            "type": "grade_leg",
            "policy_id": "candidate_a",
            "opponent_id": "jsettlers_lite",
            "wins": 1,
            "games": 4,
            "worker": "catan-zero-c1",
        },
        {
            "type": "grade_leg",
            "policy_id": "candidate_b",
            "opponent_id": "jsettlers_lite",
            "wins": 0,
            "games": 4,
            "worker": "catan-zero-c2",
        },
        {
            "type": "grade_leg",
            "policy_id": "candidate_a",
            "opponent_id": "value_rollout_search",
            "wins": 2,
            "games": 4,
            "worker": "catan-zero-c1",
        },
    ]

    summary = summarize_payoffs(
        rows,
        profile="strict",
        opponent="current_best_s9752_iter0002",
        top=2,
    )

    assert summary["decision_counts"] == {"promote_candidate": 1, "reject": 1}
    assert summary["top"][0]["policy_id"] == "candidate_a"
    assert summary["top"][0]["score_lcb"] is not None
    assert summary["top"][0]["score_ucb"] is not None
    assert summary["top"][0]["delta_lcb"] is not None
    assert summary["top"][0]["worst_leg"] == "jsettlers_lite"
    assert summary["top"][0]["legs"]["jsettlers_lite"]["score_lcb"] is not None
    assert summary["top"][0]["legs"]["jsettlers_lite"]["score_ucb"] is not None
    assert summary["weak_opponents"][0]["opponent_id"] == "jsettlers_lite"
    assert summary["weak_opponents"][0]["pfsp_priority"] == 0.875
    assert summary["training_recommendation"]["next_action"] == "escalate_best_candidate"
    assert summary["training_recommendation"]["best_candidate"]["policy_id"] == "candidate_a"


def test_recommendation_rejects_zero_win_required_leg_for_escalation() -> None:
    rows = [
        {
            "type": "grade_pair",
            "policy_id": "candidate_brittle",
            "opponent_id": "current_best_s9752_iter0002",
            "profile": "strict",
            "score": 0.25,
            "opponent_score": 0.20,
            "delta": 0.05,
            "decision": "promote_candidate",
            "summary_games": 4,
            "worker": "catan-zero-c1",
        },
        {
            "type": "grade_leg",
            "policy_id": "candidate_brittle",
            "opponent_id": "heuristic",
            "wins": 1,
            "games": 4,
            "worker": "catan-zero-c1",
        },
        {
            "type": "grade_leg",
            "policy_id": "candidate_brittle",
            "opponent_id": "jsettlers_lite",
            "wins": 1,
            "games": 4,
            "worker": "catan-zero-c1",
        },
        {
            "type": "grade_leg",
            "policy_id": "candidate_brittle",
            "opponent_id": "value_rollout_search",
            "wins": 0,
            "games": 4,
            "worker": "catan-zero-c1",
        },
    ]

    summary = summarize_payoffs(
        rows,
        profile="strict",
        opponent="current_best_s9752_iter0002",
        top=2,
    )

    recommendation = summary["training_recommendation"]
    assert recommendation["next_action"] == "train_anti_regression_repair"
    assert recommendation["best_candidate"] is None
    assert recommendation["repair_mix"][0]["opponent_id"] == "value_rollout_search"


def test_recommendation_rejects_zero_win_extra_leg_for_escalation() -> None:
    rows = [
        {
            "type": "grade_pair",
            "policy_id": "candidate_brittle",
            "opponent_id": "current_best_s9752_iter0002",
            "profile": "strict",
            "score": 0.25,
            "opponent_score": 0.20,
            "delta": 0.05,
            "decision": "promote_candidate",
            "summary_games": 4,
            "worker": "catan-zero-c1",
        },
        {
            "type": "grade_leg",
            "policy_id": "candidate_brittle",
            "opponent_id": "heuristic",
            "wins": 1,
            "games": 4,
            "worker": "catan-zero-c1",
        },
        {
            "type": "grade_leg",
            "policy_id": "candidate_brittle",
            "opponent_id": "jsettlers_lite",
            "wins": 1,
            "games": 4,
            "worker": "catan-zero-c1",
        },
        {
            "type": "grade_leg",
            "policy_id": "candidate_brittle",
            "opponent_id": "value_rollout_search",
            "wins": 1,
            "games": 4,
            "worker": "catan-zero-c1",
        },
        {
            "type": "grade_leg",
            "policy_id": "candidate_brittle",
            "opponent_id": "value",
            "wins": 0,
            "games": 12,
            "worker": "catan-zero-c1",
        },
    ]

    summary = summarize_payoffs(
        rows,
        profile="strict",
        opponent="current_best_s9752_iter0002",
        top=2,
    )

    recommendation = summary["training_recommendation"]
    assert recommendation["next_action"] == "train_anti_regression_repair"
    assert recommendation["best_candidate"] is None
    assert recommendation["repair_mix"][0]["opponent_id"] == "value"


def test_summarize_payoffs_dedupes_keyed_ledger_rows() -> None:
    rows = [
        {
            "key": "pair-a",
            "type": "grade_pair",
            "policy_id": "candidate_a",
            "opponent_id": "current_best_s9752_iter0002",
            "profile": "strict",
            "score": 0.25,
            "opponent_score": 0.20,
            "delta": 0.05,
            "decision": "reject",
        },
        {
            "key": "pair-a",
            "type": "grade_pair",
            "policy_id": "candidate_a",
            "opponent_id": "current_best_s9752_iter0002",
            "profile": "strict",
            "score": 0.25,
            "opponent_score": 0.20,
            "delta": 0.05,
            "decision": "reject",
        },
        {
            "key": "leg-a",
            "type": "grade_leg",
            "policy_id": "candidate_a",
            "opponent_id": "jsettlers_lite",
            "wins": 1,
            "games": 4,
            "worker": "catan-zero-c1",
        },
        {
            "key": "leg-a",
            "type": "grade_leg",
            "policy_id": "candidate_a",
            "opponent_id": "jsettlers_lite",
            "wins": 1,
            "games": 4,
            "worker": "catan-zero-c1",
        },
    ]

    summary = summarize_payoffs(
        rows,
        profile="strict",
        opponent="current_best_s9752_iter0002",
        top=2,
    )

    assert summary["pair_rows"] == 1
    assert summary["decision_counts"] == {"reject": 1}
    assert summary["top"][0]["legs"]["jsettlers_lite"]["games"] == 4
    assert summary["weak_opponents"][0]["games"] == 4


def test_summarize_payoffs_recommends_repair_when_no_candidate_survives() -> None:
    rows = [
        {
            "type": "grade_pair",
            "policy_id": "candidate_bad",
            "opponent_id": "current_best_s9752_iter0002",
            "profile": "strict",
            "score": 0.10,
            "opponent_score": 0.20,
            "delta": -0.10,
            "decision": "reject",
        },
        {
            "type": "grade_leg",
            "policy_id": "candidate_bad",
            "opponent_id": "value_rollout_search",
            "wins": 0,
            "games": 4,
            "worker": "catan-zero-c1",
        },
        {
            "type": "grade_leg",
            "policy_id": "candidate_bad",
            "opponent_id": "jsettlers_lite",
            "wins": 1,
            "games": 4,
            "worker": "catan-zero-c1",
        },
    ]

    summary = summarize_payoffs(
        rows,
        profile="strict",
        opponent="current_best_s9752_iter0002",
        top=2,
    )

    recommendation = summary["training_recommendation"]
    assert recommendation["next_action"] == "train_anti_regression_repair"
    assert recommendation["best_candidate"] is None
    assert recommendation["repair_mix"][0]["opponent_id"] == "value_rollout_search"
    assert recommendation["reject_rate"] == 1.0
