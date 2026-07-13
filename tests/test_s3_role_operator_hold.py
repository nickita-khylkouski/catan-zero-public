from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import a1_evaluation_pool as pool
from tools import a1_pre_wave_contract as pre_wave
from tools import a1_promotion_transaction as promotion
from tools import s3_role_operator_hold as hold
from tools.sprt_gate import evaluate_pentanomial_sprt, pair_scores_from_h2h_games


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "v5.pt"
    checkpoint.write_bytes(b"same v5 bytes")
    games = []
    for pair in range(200):
        if pair < 40:
            outcomes = (True, True)
        elif pair < 78:
            outcomes = (False, False)
        else:
            outcomes = (True, False)
        for orientation, won in zip(("candidate_red", "candidate_blue"), outcomes):
            games.append(
                {
                    "pair_id": pair,
                    "game_seed": 6_199_952_000 + pair,
                    "orientation": orientation,
                    "candidate_won": won,
                    "search_won": won,
                    "winner": "RED" if won else "BLUE",
                    "terminated": True,
                    "truncated": False,
                    "error": None,
                    "engine_divergence": False,
                    "decisions": 100,
                }
            )
    scores, diagnostics = pair_scores_from_h2h_games(games)
    pent = evaluate_pentanomial_sprt(
        scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    typed = {
        "pipeline": "eval",
        "schema_version": 6,
        "fields": {
            "mode": "cross_net",
            "candidate": str(checkpoint.resolve()),
            "baseline": str(checkpoint.resolve()),
            "base_seed": 6_199_952_000,
            "pairs": 200,
            "candidate_n_full": 128,
            "baseline_n_full": 128,
            "candidate_n_full_wide": 256,
            "baseline_n_full_wide": None,
            "candidate_n_full_wide_threshold": 40,
            "baseline_n_full_wide_threshold": None,
            "candidate_wide_roots_always_full": True,
            "baseline_wide_roots_always_full": False,
        },
    }
    digest = hashlib.sha256(
        json.dumps(typed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    common_role = {
        "c_scale": 0.1,
        "gameplay_policy_aggregation": "mean_improved_policy",
        "rescale_noise_floor_c": 0.0,
        "sigma_eval": 0.98,
        "sigma_reference_visits": None,
        "value_readout": "scalar",
        "value_squash": "tanh",
    }
    telemetry_role = {
        "search_calls": 10_000,
        "non_forced_search_calls": 8_000,
        "search_elapsed_sec": 2_000.0,
        "simulations_used": 1_280_000,
        "wide_root_calls": 800,
        "wide_root_simulations_used": 102_400,
        "selected_vs_prior_disagreement_calls": 3_000,
        "wide_selected_vs_prior_disagreement_calls": 400,
        "search_seconds_per_call": 0.2,
        "simulations_per_call": 128.0,
        "wide_root_simulations_per_call": 128.0,
        "selected_vs_prior_disagreement_rate": 0.375,
        "wide_selected_vs_prior_disagreement_rate": 0.5,
    }
    source = tmp_path / "source.json"
    _write(
        source,
        {
            "candidate_checkpoint": str(checkpoint.resolve()),
            "candidate_checkpoint_sha256": promotion._sha256(checkpoint),
            "baseline_checkpoint": str(checkpoint.resolve()),
            "baseline_checkpoint_sha256": promotion._sha256(checkpoint),
            "gate_config": "flywheel",
            "n_full": 128,
            "config_hash": "sha256:" + digest[:16],
            "full_config_hash": "sha256:" + digest,
            "typed_config": typed,
            "pairs_requested": 200,
            "base_seed": 6_199_952_000,
            "games_played": 400,
            "games_with_winner": 400,
            "games_truncated": 0,
            "candidate_wins": 202,
            "baseline_wins": 198,
            "candidate_win_rate": 0.505,
            "pentanomial_sprt": pent,
            "pair_diagnostics": diagnostics,
            "verdict": pent["decision"],
            "elapsed_sec": 4_000.0,
            "workers": 8,
            "threads_per_worker": 1,
            "public_observation": True,
            "information_set_search": True,
            "determinization_particles": 4,
            "determinization_min_simulations": 32,
            "native_mcts_hot_loop": True,
            "mcts_implementation": "rust_native_hot_loop_v1",
            **{
                f"candidate_{key}": value for key, value in common_role.items()
            },
            **{f"baseline_{key}": value for key, value in common_role.items()},
            "search_budgets_by_role": {
                "candidate": {
                    "n_full": 128,
                    "n_full_wide": 256,
                    "n_full_wide_threshold": 40,
                    "wide_roots_always_full": True,
                },
                "baseline": {
                    "n_full": 128,
                    "n_full_wide": None,
                    "n_full_wide_threshold": None,
                    "wide_roots_always_full": False,
                },
            },
            "search_telemetry": {
                "by_role": {
                    "candidate": {
                        **telemetry_role,
                        "search_elapsed_sec": 2_176.0,
                        "simulations_used": 1_368_000,
                    },
                    "baseline": telemetry_role,
                }
            },
            "errors": [],
            "games": games,
        },
    )
    pooled = pool.pool_internal([source], candidate=checkpoint, champion=checkpoint)
    registry = tmp_path / "registry.json"
    _write(registry, {"generator_champion": str(checkpoint.resolve())})
    checkpoint_ref = {
        "path": str(checkpoint.resolve()),
        "sha256": promotion._sha256(checkpoint),
    }
    pooled["evaluation_binding"] = {
        "schema_version": "a1-evaluation-baseline-binding-v1",
        "comparison_mode": "historical_comparison",
        "promotion_eligible": False,
        "historical_comparison_reason": "same-v5 S3 diagnostic",
        "candidate_parent": checkpoint_ref,
        "baseline": checkpoint_ref,
        "registry": {
            "path": str(registry.resolve()),
            "sha256": promotion._sha256(registry),
        },
        "authoritative_incumbent": {
            **checkpoint_ref,
            "version": 5,
            "agent_identity_sha256": "sha256:" + "1" * 64,
            "search_config": {},
        },
    }
    pooled["planned_engine_identity"] = {
        "schema_version": "a1-neutral-engine-identity-v1",
        "repo_commit": "a" * 40,
        "native_wheel_sha256": "sha256:" + "b" * 64,
        "python_referee_sha256": "sha256:" + "c" * 64,
    }
    pooled_path = tmp_path / "pooled.json"
    _write(pooled_path, pooled)
    return checkpoint, pooled_path


def test_same_checkpoint_adaptive_panel_emits_replayable_hold(tmp_path: Path) -> None:
    checkpoint, pooled = _source(tmp_path)
    s1 = tmp_path / "s1.json"
    s2 = tmp_path / "s2.json"
    _write(s1, {"stage": "s1"})
    _write(s2, {"stage": "s2"})
    artifact = hold.build_hold(
        pooled,
        source_s1=s1,
        source_s2=s2,
        decision_time_utc="2026-07-13T00:00:00Z",
    )

    assert artifact["decision"] == "hold"
    assert artifact["selected_fields"] == hold.SELECTED_FIELDS
    assert artifact["checkpoint"] == {
        "path": str(checkpoint.resolve()),
        "sha256": promotion._sha256(checkpoint),
    }
    assert artifact["observations"]["candidate_win_rate"] == 0.505
    semantic = pre_wave._validate_search_stage_evidence(
        artifact,
        path=tmp_path / "s3.json",
        expected_stage="s3",
        final_search=dict(hold.SELECTED_FIELDS),
        final_evaluator={},
    )
    assert semantic["evidence_class"] == hold.ARTIFACT_KIND
    assert semantic["checkpoint"]["sha256"] == promotion._sha256(checkpoint)


def test_same_checkpoint_hold_rejects_pooled_byte_drift(tmp_path: Path) -> None:
    _checkpoint, pooled = _source(tmp_path)
    s1 = tmp_path / "s1.json"
    s2 = tmp_path / "s2.json"
    _write(s1, {"stage": "s1"})
    _write(s2, {"stage": "s2"})
    payload = json.loads(pooled.read_text(encoding="utf-8"))
    payload["candidate_wins"] += 1
    _write(pooled, payload)

    with pytest.raises(hold.HoldError, match="does not replay"):
        hold.build_hold(pooled, source_s1=s1, source_s2=s2)
