from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.fixed_root_search_stability import content_sha256, root_phase_width_summary
from tools.teacher_operator_campaign import (
    CampaignError,
    aggregate_campaign,
    build_stage_commands,
    load_campaign,
)


_ROOT = Path(__file__).resolve().parents[1]
_CAMPAIGN = _ROOT / "configs/experiments/teacher_operator_coherent_v1/campaign.json"
_CHECKPOINT_HEX = "a" * 64
_CHECKPOINT_SHA = "sha256:" + _CHECKPOINT_HEX


def test_campaign_is_n128_coherent_public_and_commands_change_only_wide_budget(
    tmp_path: Path,
) -> None:
    loaded = load_campaign(_CAMPAIGN)
    commands = build_stage_commands(
        loaded,
        checkpoint=tmp_path / "champion.pt",
        out_dir=tmp_path / "out",
        device="cuda",
        devices="cuda:0,cuda:1",
    )

    assert set(commands) == {"fixed-w20", "fixed-w40", "h2h-w20", "h2h-w40"}
    for command in commands.values():
        joined = " ".join(command)
        assert "--n-full 128" in joined or "base_n128_d6.json" in joined
        assert "--n-full 64" not in joined
    assert "--candidate-n-full-wide-threshold 20" in " ".join(commands["h2h-w20"])
    assert "--candidate-n-full-wide-threshold 40" in " ".join(commands["h2h-w40"])
    assert "--coherent-public-belief-search" in commands["h2h-w20"]
    assert "--no-information-set-search" in commands["h2h-w20"]
    assert "--symmetry-averaged-eval-threshold" in commands["h2h-w20"]
    fixed = commands["fixed-w20"]
    assert fixed[fixed.index("--n-roots") + 1] == "64"
    assert fixed[fixed.index("--max-root-games") + 1] == "512"
    quota_values = [
        fixed[index + 1]
        for index, value in enumerate(fixed)
        if value == "--root-stratum-quota"
    ]
    assert quota_values == [
        "play_turn:2-19=24",
        "play_turn:20-31=16",
        "play_turn:32-39=8",
        "opening_placement:40+=8",
    ]


def _run(seed: int, *, js: float, wall: float, simulations: int) -> dict:
    return {
        "search_seed": seed,
        "selected_action": 1,
        "improved_policy": {"1": 0.6, "2": 0.4},
        "prior_policy": {"1": 0.5, "2": 0.5},
        "target_top_probability": 0.6,
        "target_entropy": 0.67,
        "prior_top_probability": 0.5,
        "prior_entropy": 0.69,
        "target_prior_js": 0.01,
        "completed_q_range": 0.2,
        "completed_q_top_margin": 0.1,
        "simulations_used": simulations,
        "wall_sec": wall,
        "logical_leaf_evaluations": simulations,
        "orientation_evaluation_rows": simulations,
        "evaluator_method_calls": simulations,
        "_test_pair_js": js,
    }


def _role(*, js: float, wall: float, simulations: int, seed_base: int) -> dict:
    runs = [
        _run(seed_base + index, js=js, wall=wall, simulations=simulations)
        for index in range(4)
    ]
    return {
        "runs": runs,
        "stability": {
            "pairwise": [
                {"js_divergence": js, "top1_agreement": True} for _ in range(6)
            ]
        },
    }


def _fixed_report(arm_id: str, threshold: int, *, candidate_js: float) -> dict:
    loaded = load_campaign(_CAMPAIGN)
    roots = []
    strata = (
        *((10, "play_turn") for _ in range(24)),
        *((20, "play_turn") for _ in range(16)),
        *((32, "play_turn") for _ in range(8)),
        *((54, "opening_placement") for _ in range(8)),
        *((2, "opening_placement") for _ in range(8)),
    )
    for index, (width, phase) in enumerate(strata):
        if width <= 10:
            bucket = "5-10"
        elif width <= 20:
            bucket = "11-20"
        elif width <= 40:
            bucket = "21-40"
        else:
            bucket = "41+"
        roots.append(
            {
                "root_index": index,
                "root_sha256": f"sha256:{index:064x}",
                "game_seed": 100 + index,
                "decision_index": index,
                "legal_width": width,
                "legal_width_bucket": bucket,
                "wide_ge_40": width >= 40,
                "phase": phase,
                "phase_raw": (
                    "PLAY_TURN" if phase == "play_turn" else "BUILD_INITIAL_SETTLEMENT"
                ),
                "roles": {
                    "base_n128_d6": _role(
                        js=0.10, wall=1.0, simulations=128, seed_base=1000 + 10 * index
                    ),
                    arm_id: _role(
                        js=candidate_js,
                        wall=1.1,
                        simulations=256,
                        seed_base=2000 + 10 * index,
                    ),
                },
            }
        )
    report = {
        "schema_version": "fixed-root-search-stability-v2",
        "checkpoint": {"sha256": _CHECKPOINT_SHA},
        "root_panel": {
            "content_sha256": "sha256:" + "b" * 64,
            "root_count": 64,
            "root_stratum_quotas": loaded["payload"]["fixed_root_protocol"][
                "root_stratum_quotas"
            ],
            "root_stratum_counts": {
                "play_turn:2-19": 24,
                "play_turn:20-31": 16,
                "play_turn:32-39": 8,
                "opening_placement:40+": 8,
            },
            "root_phase_width_summary": root_phase_width_summary(roots),
        },
        "roles": {
            "base_n128_d6": {
                "effective_search_config_sha256": loaded["base"]["spec"][
                    "effective_search_config_sha256"
                ]
            },
            arm_id: {
                "effective_search_config_sha256": loaded["arms"][arm_id]["spec"][
                    "effective_search_config_sha256"
                ]
            },
        },
        "search_config_differences": {
            "n_full_wide": {},
            "n_full_wide_threshold": {},
            "wide_roots_always_full": {},
        },
        "slices": {"fixture": True},
        "per_root": roots,
    }
    report["report_content_sha256"] = content_sha256(report)
    return report


def _h2h_report(threshold: int) -> dict:
    return {
        "candidate_checkpoint_sha256": _CHECKPOINT_HEX,
        "baseline_checkpoint_sha256": _CHECKPOINT_HEX,
        "candidate_n_full": 128,
        "baseline_n_full": 128,
        "candidate_n_full_wide": 256,
        "baseline_n_full_wide": None,
        "candidate_n_full_wide_threshold": threshold,
        "candidate_wide_roots_always_full": True,
        "baseline_wide_roots_always_full": False,
        "lazy_interior_chance": True,
        "value_squash": "tanh",
        "candidate_value_squash": "tanh",
        "baseline_value_squash": "tanh",
        "value_readout": "scalar",
        "candidate_value_readout": "scalar",
        "baseline_value_readout": "scalar",
        "c_scale": 0.1,
        "candidate_c_scale": 0.1,
        "baseline_c_scale": 0.1,
        "c_visit": 50.0,
        "rescale_noise_floor_c": 0.0,
        "candidate_rescale_noise_floor_c": 0.0,
        "baseline_rescale_noise_floor_c": 0.0,
        "sigma_eval": 0.79,
        "candidate_sigma_eval": 0.79,
        "baseline_sigma_eval": 0.79,
        "max_root_candidates": 16,
        "max_root_candidates_wide": 54,
        "wide_candidates_threshold": 24,
        "correct_rust_chance_spectra": True,
        "public_observation": True,
        "coherent_public_belief_search": True,
        "information_set_search": False,
        "belief_chance_spectra": False,
        "symmetry_averaged_eval": True,
        "symmetry_averaged_eval_threshold": 20,
        "forced_root_target_mode": "trajectory_only",
        "native_mcts_hot_loop": True,
        "determinization_particles": 1,
        "determinization_min_simulations": 32,
        "raw_policy_above_width": None,
        "pairs_requested": 100,
        "games_played": 200,
        "games_truncated": 0,
        "complete_pairs": 100,
        "errors": [],
        "typed_config": {"fields": {"evaluator_rust_featurize": True, "max_depth": 80}},
        "candidate_wins": 100,
        "baseline_wins": 100,
        "candidate_win_rate": 0.5,
        "verdict": "continue",
        "superiority_verdict": "continue",
        "pair_diagnostics": {
            "ww_pairs": 25,
            "ll_pairs": 25,
            "split_pairs": 50,
            "incomplete_pairs": 0,
        },
        "search_telemetry": {
            "candidate_over_baseline_elapsed_ratio": 1.1,
            "candidate_over_baseline_simulations_ratio": 1.2,
            "by_role": {
                "candidate": {
                    "wide_root_calls": 20,
                    "wide_selected_vs_prior_disagreement_rate": 0.3,
                },
                "baseline": {
                    "wide_root_calls": 20,
                    "wide_selected_vs_prior_disagreement_rate": 0.2,
                },
            },
        },
    }


def test_h2h_checkpoint_digest_accepts_canonical_sha256_prefix(tmp_path: Path) -> None:
    loaded = load_campaign(_CAMPAIGN)
    for arm_id, threshold in (
        ("adaptive_n256_w20_d6", 20),
        ("adaptive_n256_w40_d6", 40),
    ):
        (tmp_path / f"fixed.{arm_id}.json").write_text(
            json.dumps(_fixed_report(arm_id, threshold, candidate_js=0.05)),
            encoding="utf-8",
        )
        h2h = _h2h_report(threshold)
        h2h["candidate_checkpoint_sha256"] = _CHECKPOINT_SHA
        h2h["baseline_checkpoint_sha256"] = _CHECKPOINT_SHA
        (tmp_path / f"h2h.{arm_id}.json").write_text(json.dumps(h2h), encoding="utf-8")

    report = aggregate_campaign(loaded, out_dir=tmp_path)
    assert report["checkpoint_sha256"] == _CHECKPOINT_SHA


def test_aggregate_selects_cost_bounded_stability_winner(tmp_path: Path) -> None:
    loaded = load_campaign(_CAMPAIGN)
    for arm_id, threshold, candidate_js in (
        ("adaptive_n256_w20_d6", 20, 0.05),
        ("adaptive_n256_w40_d6", 40, 0.095),
    ):
        (tmp_path / f"fixed.{arm_id}.json").write_text(
            json.dumps(_fixed_report(arm_id, threshold, candidate_js=candidate_js)),
            encoding="utf-8",
        )
        (tmp_path / f"h2h.{arm_id}.json").write_text(
            json.dumps(_h2h_report(threshold)), encoding="utf-8"
        )

    report = aggregate_campaign(loaded, out_dir=tmp_path)

    assert report["checkpoint_sha256"] == _CHECKPOINT_SHA
    assert report["root_distribution"]["width_40_classification"] == "opening_only"
    assert (
        report["root_distribution"]["phase_width_summary"]["play_turn"][
            "max_legal_width"
        ]
        == 32
    )
    assert report["selection"]["selected_operator"] == "adaptive_n256_w20_d6"
    assert report["results"]["adaptive_n256_w20_d6"]["selection_evidence"] == {
        "cost_ok": True,
        "positive_elo_h1": False,
        "stability_proxy_ok": True,
        "eligible": True,
    }
    assert report["causal_contract"]["forbidden_budget"] == 64


def test_aggregate_rejects_old_unstratified_fixed_root_panel(tmp_path: Path) -> None:
    loaded = load_campaign(_CAMPAIGN)
    for arm_id, threshold in (
        ("adaptive_n256_w20_d6", 20),
        ("adaptive_n256_w40_d6", 40),
    ):
        fixed = _fixed_report(arm_id, threshold, candidate_js=0.05)
        if threshold == 20:
            fixed["root_panel"].pop("root_stratum_quotas")
            fixed["report_content_sha256"] = content_sha256(
                {
                    key: value
                    for key, value in fixed.items()
                    if key != "report_content_sha256"
                }
            )
        (tmp_path / f"fixed.{arm_id}.json").write_text(
            json.dumps(fixed), encoding="utf-8"
        )
        (tmp_path / f"h2h.{arm_id}.json").write_text(
            json.dumps(_h2h_report(threshold)), encoding="utf-8"
        )

    with pytest.raises(CampaignError, match="preregistered phase/width panel"):
        aggregate_campaign(loaded, out_dir=tmp_path)
