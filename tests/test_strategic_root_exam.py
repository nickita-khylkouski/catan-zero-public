from __future__ import annotations

from tools.strategic_root_exam import (
    _role_exam,
    describe_action,
    render_markdown,
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


def test_exam_uses_sealed_snapshot_actions_without_runtime_reconstruction() -> None:
    from tools.strategic_root_exam import build_exam

    panel = {
        "panel_content_sha256": "sha256:panel",
        "roots": [
            {
                "root_index": 0,
                "root_sha256": "sha256:root",
                "game_seed": 7,
                "decision_index": 0,
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
