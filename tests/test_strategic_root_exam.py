from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.strategic_root_exam import (
    _role_exam,
    build_exam,
    content_sha256,
    describe_action,
    render_markdown,
    run_paired_counterfactuals,
)


def test_describe_settlement_exposes_human_catan_context() -> None:
    description = describe_action(
        91,
        ("RED", "BUILD_SETTLEMENT", 13),
        snapshot={},
        production={13: {"wood": 5, "brick": 4, "wheat": 3}},
        ports={13: "brick"},
    )

    assert description["strategic_context"] == {
        "node": 13,
        "total_pips": 12,
        "resource_pips": {"wood": 5, "brick": 4, "wheat": 3},
        "resource_diversity": 3,
        "port": "brick",
    }


def test_role_exam_flags_unstable_confident_teacher_target() -> None:
    actions = {
        1: {"action_id": 1, "action_type": "END_TURN", "argument": None},
        2: {"action_id": 2, "action_type": "BUY_DEVELOPMENT_CARD", "argument": None},
    }
    role = {
        "runs": [
            {
                "selected_action": 1,
                "prior_policy": {"1": 0.8, "2": 0.2},
                "improved_policy": {"1": 0.99, "2": 0.01},
                "completed_q_top_margin": 0.001,
                "target_prior_js": 0.1,
                "target_top_probability": 0.99,
            },
            {
                "selected_action": 2,
                "prior_policy": {"1": 0.8, "2": 0.2},
                "improved_policy": {"1": 0.01, "2": 0.99},
                "completed_q_top_margin": 0.002,
                "target_prior_js": 0.2,
                "target_top_probability": 0.99,
            },
        ],
        "stability": {
            "cross_seed_js_mean": 0.4,
            "top1_pair_agreement": 0.0,
        },
    }

    exam = _role_exam(
        role,
        action_descriptions=actions,
        top_k=2,
        js_warning=0.1,
        q_margin_warning=0.02,
    )

    assert exam["diagnostic_flags"] == [
        "selected_action_changes_across_search_seeds",
        "high_cross_seed_policy_disagreement",
        "top_completed_q_margin_near_noise_scale",
    ]
    assert exam["raw_prior_top"][0]["action"]["action_type"] == "END_TURN"


def test_markdown_names_actual_moves() -> None:
    exam = {
        "root_count": 1,
        "thresholds": {"js_warning": 0.1, "q_margin_warning": 0.02},
        "roots": [
            {
                "root_index": 0,
                "game_seed": 7,
                "decision_index": 20,
                "phase_raw": "PLAY_TURN",
                "actor": "RED",
                "legal_width": 2,
                "paired_counterfactuals": {
                    "actions": [
                        {
                            "action": {
                                "action_type": "END_TURN",
                                "argument": None,
                            },
                            "summary": {"mean_outcome": -1.0, "wins": 0, "trials": 8},
                        },
                        {
                            "action": {
                                "action_type": "PLAY_ROAD_BUILDING",
                                "argument": None,
                            },
                            "summary": {
                                "mean_outcome": -0.25,
                                "wins": 3,
                                "trials": 8,
                            },
                        },
                    ]
                },
                "roles": {
                    "base_n128": {
                        "raw_prior_top": [
                            {
                                "probability": 0.7,
                                "action": {
                                    "action_type": "END_TURN",
                                    "argument": None,
                                },
                            }
                        ],
                        "selected_actions": [
                            {
                                "action_type": "BUY_DEVELOPMENT_CARD",
                                "argument": None,
                            }
                        ],
                        "cross_seed_js_mean": 0.0,
                        "selected_action_pair_agreement": 1.0,
                        "mean_completed_q_top_margin": 0.2,
                        "diagnostic_flags": [],
                    }
                },
            }
        ],
    }

    markdown = render_markdown(exam)

    assert "END_TURN" in markdown
    assert "BUY_DEVELOPMENT_CARD" in markdown
    assert "PLAY_ROAD_BUILDING" in markdown
    assert "mean outcome -0.250" in markdown


def test_exam_uses_sealed_snapshot_actions_without_runtime_reconstruction() -> None:
    panel = {
        "panel_content_sha256": "sha256:panel",
        "provenance": {
            "checkpoint_sha256": "sha256:checkpoint",
            "evaluator_config_sha256": "sha256:evaluator",
        },
        "roots": [
            {
                "root_index": 0,
                "root_sha256": "sha256:root",
                "game_seed": 7,
                "decision_index": 0,
                "action_prefix": [],
                "phase": "opening_placement",
                "phase_raw": "BUILD_INITIAL_SETTLEMENT",
                "legal_width": 2,
                "current_color": "RED",
                "legal_action_ids": [78, 79],
                "snapshot": {
                    "colors": ["RED", "BLUE"],
                    "player_state": [{}, {}],
                    "current_playable_actions": [
                        ["RED", "BUILD_SETTLEMENT", 0],
                        ["RED", "BUILD_SETTLEMENT", 10],
                    ],
                },
            }
        ],
    }
    run = {
        "selected_action": 78,
        "prior_policy": {"78": 0.6, "79": 0.4},
        "improved_policy": {"78": 0.7, "79": 0.3},
        "completed_q_top_margin": 0.1,
    }
    report = {
        "schema_version": "fixed-root-search-stability-v2",
        "report_content_sha256": "sha256:report",
        "checkpoint": {"sha256": "sha256:checkpoint"},
        "evaluator": {"effective_evaluator_config_sha256": "sha256:evaluator"},
        "roles": {
            "base_n128": {"effective_search_config_sha256": "sha256:search-n128"}
        },
        "root_panel": {"content_sha256": "sha256:panel"},
        "per_root": [
            {
                "root_sha256": "sha256:root",
                "roles": {
                    "base_n128": {
                        "runs": [run, dict(run)],
                        "stability": {
                            "cross_seed_js_mean": 0.0,
                            "top1_pair_agreement": 1.0,
                        },
                    }
                },
            }
        ],
    }

    exam = build_exam(panel, report)

    chosen = exam["roots"][0]["roles"]["base_n128"]["selected_actions"][0]
    assert chosen["action_type"] == "BUILD_SETTLEMENT"
    assert chosen["argument"] == 0


def test_exam_embeds_standalone_replay_and_model_operator_hashes() -> None:
    panel = {
        "panel_content_sha256": "sha256:panel",
        "provenance": {
            "checkpoint_sha256": "sha256:checkpoint",
            "evaluator_config_sha256": "sha256:evaluator",
        },
        "roots": [
            {
                "root_index": 0,
                "root_sha256": "sha256:root",
                "game_seed": 7,
                "decision_index": 2,
                "action_prefix": [91, 92],
                "phase": "play_turn",
                "phase_raw": "PLAY_TURN",
                "legal_width": 2,
                "current_color": "RED",
                "legal_action_ids": [186, 310],
                "snapshot": {
                    "colors": ["RED", "BLUE"],
                    "player_state": [{}, {}],
                    "current_playable_actions": [
                        ["RED", "END_TURN", None],
                        ["RED", "PLAY_ROAD_BUILDING", None],
                    ],
                },
            }
        ],
    }
    run = {
        "selected_action": 186,
        "prior_policy": {"186": 0.784, "310": 0.216},
        "improved_policy": {"186": 0.8, "310": 0.2},
        "completed_q_top_margin": 0.006,
    }
    report = {
        "schema_version": "fixed-root-search-stability-v2",
        "report_content_sha256": "sha256:report",
        "checkpoint": {"sha256": "sha256:checkpoint"},
        "evaluator": {"effective_evaluator_config_sha256": "sha256:evaluator"},
        "roles": {
            "coherent_n128": {"effective_search_config_sha256": "sha256:search-n128"}
        },
        "root_panel": {"content_sha256": "sha256:panel"},
        "per_root": [
            {
                "root_sha256": "sha256:root",
                "roles": {
                    "coherent_n128": {
                        "runs": [run, dict(run)],
                        "stability": {
                            "cross_seed_js_mean": 0.0,
                            "top1_pair_agreement": 1.0,
                        },
                    }
                },
            }
        ],
    }

    counterfactual = run_paired_counterfactuals(
        object(),
        action_ids=[186, 310],
        seeds=[11, 12],
        hook=lambda _state, action, seed: {
            "outcome": 1.0 if action == 310 and seed == 11 else -1.0
        },
        action_descriptions={
            186: {
                "action_id": 186,
                "action_type": "END_TURN",
                "argument": None,
            },
            310: {
                "action_id": 310,
                "action_type": "PLAY_ROAD_BUILDING",
                "argument": None,
            },
        },
    )
    exam = build_exam(panel, report, counterfactuals_by_root={0: counterfactual})

    assert exam["provenance"] == {
        "root_panel_content_sha256": "sha256:panel",
        "source_report_content_sha256": "sha256:report",
        "checkpoint_sha256": "sha256:checkpoint",
        "evaluator_operator_sha256": "sha256:evaluator",
        "search_operator_sha256_by_role": {"coherent_n128": "sha256:search-n128"},
    }
    assert exam["roots"][0]["replay"] == {
        "root_sha256": "sha256:root",
        "game_seed": 7,
        "decision_index": 2,
        "action_prefix": [91, 92],
        "snapshot": panel["roots"][0]["snapshot"],
        "legal_action_ids": [186, 310],
        "current_color": "RED",
    }
    assert exam["roots"][0]["paired_counterfactuals"] == counterfactual


def test_exam_rejects_checkpoint_or_operator_provenance_drift() -> None:
    panel = {
        "panel_content_sha256": "sha256:panel",
        "provenance": {
            "checkpoint_sha256": "sha256:checkpoint-a",
            "evaluator_config_sha256": "sha256:evaluator",
        },
        "roots": [],
    }
    report = {
        "schema_version": "fixed-root-search-stability-v2",
        "checkpoint": {"sha256": "sha256:checkpoint-b"},
        "evaluator": {"effective_evaluator_config_sha256": "sha256:evaluator"},
        "roles": {},
        "root_panel": {"content_sha256": "sha256:panel"},
        "per_root": [],
    }

    with pytest.raises(ValueError, match="checkpoint hashes differ"):
        build_exam(panel, report)


def test_paired_counterfactual_hook_uses_the_same_seeds_for_each_action() -> None:
    calls: list[tuple[int, int]] = []
    actions = {
        186: {"action_id": 186, "action_type": "END_TURN", "argument": None},
        310: {
            "action_id": 310,
            "action_type": "PLAY_ROAD_BUILDING",
            "argument": None,
        },
    }

    def hook(_state: object, action_id: int, seed: int) -> dict:
        calls.append((action_id, seed))
        return {"outcome": 1.0 if action_id == 310 and seed == 11 else -1.0}

    evidence = run_paired_counterfactuals(
        object(),
        action_ids=[186, 310],
        seeds=[11, 12],
        hook=hook,
        action_descriptions=actions,
    )

    assert calls == [(186, 11), (186, 12), (310, 11), (310, 12)]
    assert evidence["protocol"] == "paired-common-seed-v1"
    assert evidence["seed_manifest_sha256"] == content_sha256([11, 12])
    assert evidence["actions"][0]["summary"]["wins"] == 0
    assert evidence["actions"][1]["summary"]["wins"] == 1
    assert evidence["paired_deltas"][0]["mean_outcome_delta"] == pytest.approx(1.0)


def test_checked_in_road_building_failure_shape_needs_no_private_artifact() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "decision_museum"
        / "road_building_vs_end_turn.json"
    )
    payload = json.loads(fixture.read_text(encoding="utf-8"))

    assert payload["private_artifact_dependency"] is False
    assert payload["raw_policy"]["END_TURN"] == pytest.approx(0.784)
    assert payload["search_selected"] == "END_TURN"
    assert payload["search_q"]["PLAY_ROAD_BUILDING"] > payload["search_q"]["END_TURN"]
    end = payload["paired_counterfactual_outcomes"]["END_TURN"]
    road = payload["paired_counterfactual_outcomes"]["PLAY_ROAD_BUILDING"]
    assert len(end) == len(road) == 8
    assert sum(outcome > 0 for outcome in end) == 0
    assert sum(outcome > 0 for outcome in road) == 3
