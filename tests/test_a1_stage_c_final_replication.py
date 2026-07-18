from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools import a1_one_dose_train as one_dose
from tools import a1_pre_wave_contract as contract
from tools import a1_stage_c_final_replication as final


def _write_sealed(path: Path, value: dict, field: str) -> dict:
    payload = copy.deepcopy(value)
    payload[field] = final.value_sha256(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _behavioral_competence_gate_contract() -> dict:
    return {
        "schema_version": "a1-maritime-behavioral-competence-gate-v1",
        "required_streams": ["base", "aux"],
        "stream_admission": {
            "mode": "independent",
            "pooling": False,
            "all_required_streams_must_pass": True,
        },
        "required_axis": "teacher_argmax_action_type",
        "required_action_type": "MARITIME_TRADE",
        "minimum_rows": 64,
        "confidence_interval": {
            "method": "wilson_score",
            "sided": "one",
            "confidence_level": 0.95,
            "z": 1.6448536269514722,
        },
        "minimum_teacher_top1_lower_bound": 0.15,
        "maximum_end_turn_confusion_upper_bound": 0.40,
        "maximum_teacher_top1_parent_regression": 0.05,
        "maximum_end_turn_confusion_parent_regression": 0.05,
        "objective_weighted_diagnostic": {
            "required": True,
            "axis": "objective_weighted_teacher_argmax_action_type",
            "selection_authority": False,
        },
        "playing_strength": {
            "candidate_prefilter_only": True,
            "h2h_final_ranking_authority": True,
        },
    }


def _behavioral_metrics(
    *,
    rows: int,
    teacher_top1_successes: int,
    end_turn_confusions: int,
) -> dict:
    return {
        "teacher_top1_accuracy": teacher_top1_successes / rows,
        "end_turn_confusion_rate": end_turn_confusions / rows,
        "teacher_top1_successes": teacher_top1_successes,
        "end_turn_confusions": end_turn_confusions,
        "end_turn_confusion_teacher_probability_regret_per_row": 0.1,
        "end_turn_confusion_teacher_probability_regret_conditional_mean": 0.2,
    }


def _behavioral_metric_deltas(candidate: dict, parent: dict) -> dict:
    return {
        key: float(candidate[key]) - float(parent[key])
        for key in (
            "teacher_top1_accuracy",
            "end_turn_confusion_rate",
            "end_turn_confusion_teacher_probability_regret_per_row",
            "end_turn_confusion_teacher_probability_regret_conditional_mean",
        )
    }


def _behavioral_competence_gate(
    *,
    rows: int,
    teacher_top1_successes: int,
    end_turn_confusions: int,
) -> dict:
    abi_unsigned = {
        "version": "action-catalog-v1",
        "size": 3,
        "ordered_descriptors_sha256": "sha256:" + "1" * 64,
        "action_types_by_id_sha256": "sha256:" + "2" * 64,
    }
    abi = {
        **abi_unsigned,
        "identity_sha256": final.value_sha256(abi_unsigned),
    }
    candidate = _behavioral_metrics(
        rows=rows,
        teacher_top1_successes=teacher_top1_successes,
        end_turn_confusions=end_turn_confusions,
    )
    parent = copy.deepcopy(candidate)
    weighted_candidate = {
        key: candidate[key]
        for key in (
            "teacher_top1_accuracy",
            "end_turn_confusion_rate",
            "end_turn_confusion_teacher_probability_regret_per_row",
            "end_turn_confusion_teacher_probability_regret_conditional_mean",
        )
    }
    weighted_parent = copy.deepcopy(weighted_candidate)
    stream = {
        "axis": "teacher_argmax_action_type",
        "action_type": "MARITIME_TRADE",
        "action_catalog_abi": abi,
        "rows": rows,
        "candidate": candidate,
        "parent": parent,
        "candidate_minus_parent": _behavioral_metric_deltas(candidate, parent),
        "objective_weighted_diagnostic": {
            "axis": "objective_weighted_teacher_argmax_action_type",
            "selection_authority": False,
            "row_probability": float(rows),
            "candidate": weighted_candidate,
            "parent": weighted_parent,
            "candidate_minus_parent": _behavioral_metric_deltas(
                weighted_candidate,
                weighted_parent,
            ),
        },
    }
    return (
        final.stage_c_campaign._maritime_behavioral_competence_gate_from_evidence(  # noqa: SLF001
            {
                "streams": {
                    "base": copy.deepcopy(stream),
                    "aux": copy.deepcopy(stream),
                }
            },
            contract=_behavioral_competence_gate_contract(),
        )
    )


def _admitted_behavioral_competence_gate() -> dict:
    gate = _behavioral_competence_gate(
        rows=100,
        teacher_top1_successes=30,
        end_turn_confusions=20,
    )
    assert gate["selection_admitted"] is True
    return gate


def _failed_behavioral_competence_gate() -> dict:
    gate = _behavioral_competence_gate(
        rows=94,
        teacher_top1_successes=6,
        end_turn_confusions=74,
    )
    assert gate["selection_admitted"] is False
    return gate


def _campaign(tmp_path: Path, *, parent_sha: str) -> tuple[Path, dict]:
    value = {
        "schema_version": final.CAMPAIGN_SCHEMA,
        "arm": final.EXPECTED_ARM,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "lineage": {
            "fresh_adam": True,
            "candidate_chaining": False,
            "learner_parent_sha256": parent_sha,
        },
        "topology": {
            "name": "b200-8gpu-ddp",
            "world_size": 8,
            "local_batch_size": 512,
            "global_batch_size": 4096,
        },
        "recipe": {
            "epochs": 1,
            "max_steps": 32,
            "lr": 6.0e-5,
            "lr_warmup_steps": 16,
            "policy_loss_weight": 1.0,
            "value_loss_weight": 0.25,
            "policy_aux_active_batch_size": 64,
            "policy_aux_loss_weight": 0.25,
            "soft_target_source": "policy",
            "soft_target_weight": 1.0,
            "soft_target_min_legal_coverage": 1.0,
            "public_card_lr_mult": 1.0,
            "value_trunk_grad_scale": 0.1,
            "per_game_policy_surprise_weighting": False,
            "policy_kl_anchor_weight": 0.0,
            "forced_row_value_action_type_weights": "END_TURN=1,ROLL=1",
        },
        "selection_contract": {
            "requires_checkpoint_local_feature_learning_signal": True,
            "earliest_feature_signal_step": (
                final.stage_c_campaign.TRAIN_DIAGNOSTIC_CADENCE_BATCHES
            ),
            "value_quality_gate": final._expected_value_gate_contract(),  # noqa: SLF001
            "behavioral_competence_gate": (
                _behavioral_competence_gate_contract()
            ),
        },
        "feature_learning_signal_contract": (
            final._expected_feature_learning_contract()  # noqa: SLF001
        ),
    }
    path = tmp_path / "campaign.json"
    return path, _write_sealed(path, value, "campaign_sha256")


def _fingerprint(
    tmp_path: Path, *, parent_sha: str, steps: tuple[int, ...]
) -> tuple[Path, dict]:
    campaign_path, campaign = _campaign(tmp_path, parent_sha=parent_sha)
    policy_teacher_gap_objective = (
        final.stage_c_campaign._expected_policy_teacher_gap_objective(  # noqa: SLF001
            campaign["recipe"]
        )
    )
    records = []
    checkpoints = {}
    for step in steps:
        checkpoint = tmp_path / f"candidate_step{step:04d}.pt"
        checkpoint.write_bytes(f"candidate-{step}".encode())
        checkpoints[step] = checkpoint
    terminal = checkpoints.get(final.stage_c_campaign.MAX_STEPS)
    if terminal is None:
        terminal = tmp_path / "candidate.pt"
        terminal.write_bytes(b"terminal-candidate")
    report_path = tmp_path / "train.report.json"
    report = {
        "checkpoint": str(terminal),
        "intermediate_checkpoints": [
            {
                "schema_version": "train-bc-intermediate-checkpoint-v1",
                "optimizer_step": step,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": final.file_sha256(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "same_training_trajectory": True,
                "optimizer_sidecar": None,
            }
            for step, checkpoint in checkpoints.items()
            if step != final.stage_c_campaign.MAX_STEPS
        ],
    }
    dose_rows = []
    for step, checkpoint in checkpoints.items():
        feature_authenticated = True
        observability = {
            "schema_version": "module-optimizer-observability-v1",
            "observed_steps": 1,
            "cadence_batches": (
                final.stage_c_campaign.TRAIN_DIAGNOSTIC_CADENCE_BATCHES
            ),
            "norm_scope": "global_replicated",
            "modules": {
                module: {
                    "mean_pre_clip_grad_norm": 0.4,
                    "max_pre_clip_grad_norm": 0.6,
                    "mean_parameter_delta_norm": 0.02,
                    "mean_parameter_update_rms": 0.001,
                    "parameter_count": 8,
                }
                for module in final.stage_c_campaign.FEATURE_SIGNAL_MODULES
            },
        }
        feature_paths = {
            "public_card": {"enabled": True, "status": "observed"},
            "meaningful_history": {"enabled": True, "status": "observed"},
        }
        feature_signal = {
            "authenticated": True,
            **observability,
            "optimizer_step": step,
            "feature_paths": feature_paths,
        }
        dose_rows.append(
            {
                "schema_version": "train-bc-checkpoint-dose-telemetry-v1",
                "optimizer_step": step,
                "module_optimizer_observability": observability,
                "feature_path_gradients": feature_paths,
            }
        )
        records.append(
            {
                "step": step,
                "policy_teacher_gap_objective": policy_teacher_gap_objective,
                "eligible": step != 16,
                "feature_learning_signal_authenticated": feature_authenticated,
                "feature_learning_signal": feature_signal,
                "value_quality_gate": {"passed": True},
                "behavioral_competence_gate": (
                    _admitted_behavioral_competence_gate()
                ),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": final.file_sha256(checkpoint),
                "checkpoint_report_binding": {
                    "schema_version": "stage-c-checkpoint-report-binding-v1",
                    "optimizer_step": step,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": final.file_sha256(checkpoint),
                    "source": (
                        "receipt_bound_terminal_checkpoint"
                        if step == final.stage_c_campaign.MAX_STEPS
                        else "authenticated_intermediate_checkpoint"
                    ),
                },
            }
        )
    report["checkpoint_dose_trajectory"] = {
        "schema_version": "train-bc-checkpoint-dose-trajectory-v1",
        "checkpoint_steps": list(final.stage_c_campaign.CHECKPOINT_STEPS),
        "checkpoints": dose_rows,
    }
    report_path.write_text(json.dumps(report, sort_keys=True))
    value = {
        "schema_version": final.FRESH_FINGERPRINT_SCHEMA,
        "campaign": {
            **final._artifact(campaign_path),  # noqa: SLF001
            "campaign_sha256": campaign["campaign_sha256"],
        },
        "checkpoints": records,
        "completed_dose": {
            "feature_learning_signal_authenticated": True,
            "report": {
                "path": str(report_path),
                "file_sha256": final.file_sha256(report_path),
            },
            "terminal_checkpoint": {
                "path": str(terminal),
                "file_sha256": final.file_sha256(terminal),
            },
        },
        "stored_generation_prior_used_as_selection_authority": False,
        "optimizer_batch_kl_used_as_trust_authority": False,
        "policy_teacher_gap_objective": policy_teacher_gap_objective,
        "value_quality_gate": final._expected_value_gate_contract(),  # noqa: SLF001
        "behavioral_competence_gate": (
            _behavioral_competence_gate_contract()
        ),
        "separate_exact_parent_evidence": {"selection_authority": True},
    }
    path = tmp_path / "fingerprint.json"
    return path, _write_sealed(path, value, "fingerprint_sha256")


def _panel(
    path: Path,
    *,
    checkpoint: Path,
    baseline_sha: str,
    win_rate: float,
    decision: str,
) -> Path:
    candidate_sha = final.file_sha256(checkpoint)
    candidate_wins = round(win_rate * final.EXPECTED_GAMES)
    report = {
        "fleet_merge": {
            "schema_version": final.EXPECTED_POOL_SCHEMA,
            "candidate": {"sha256": candidate_sha},
            "effective_search_config_sha256": "sha256:" + "e" * 64,
        },
        "engine_identity": {
            "schema_version": final.EXPECTED_ENGINE_SCHEMA,
            "repo_commit": "deadbeef",
        },
        "effective_search_config": {"n_full": 128},
        "candidate_checkpoint_sha256": candidate_sha,
        "baseline_checkpoint_sha256": baseline_sha,
        "comparison_contract": final.EXPECTED_COMPARISON,
        "coherent_public_belief_search": True,
        "public_observation": True,
        "correct_rust_chance_spectra": True,
        "native_mcts_hot_loop": True,
        "forced_root_target_mode": "trajectory_only",
        "candidate_n_full": 128,
        "baseline_n_full": 128,
        "candidate_gameplay_policy_aggregation": "mean_improved_policy",
        "baseline_gameplay_policy_aggregation": "mean_improved_policy",
        "errors": [],
        "games_truncated": 0,
        "games_played": final.EXPECTED_GAMES,
        "games_with_winner": final.EXPECTED_GAMES,
        "complete_pairs": final.EXPECTED_PAIRS,
        "pairs_requested": final.EXPECTED_PAIRS,
        "pairs_truncated_excluded": 0,
        "candidate_wins": candidate_wins,
        "baseline_wins": final.EXPECTED_GAMES - candidate_wins,
        "candidate_win_rate": win_rate,
        "verdict": decision,
        "pentanomial_sprt": {"decision": decision},
        "search_rng_contract": {"stream_key": "role,pair,orientation"},
        "games": [
            {"game_seed": 9_000_000 + pair}
            for pair in range(final.EXPECTED_PAIRS)
            for _orientation in range(2)
        ],
    }
    path.write_text(json.dumps(report))
    return path


def _cohort(seeds: list[int], wins: list[int], *, decision: str) -> dict:
    assert len(seeds) == len(wins)
    pairs = len(seeds)
    candidate_wins = sum(wins)
    rate = candidate_wins / (2 * pairs)
    ll_pairs = sum(value == 0 for value in wins)
    split_pairs = sum(value == 1 for value in wins)
    ww_pairs = sum(value == 2 for value in wins)
    sprt = {
        "decision": decision,
        "pairs": pairs,
        "mean_pair_score": rate,
        "ll_pairs": ll_pairs,
        "split_pairs": split_pairs,
        "ww_pairs": ww_pairs,
    }
    return {
        "pairs": pairs,
        "games": 2 * pairs,
        "candidate_wins": candidate_wins,
        "baseline_wins": 2 * pairs - candidate_wins,
        "candidate_win_rate": rate,
        "pair_outcomes": [
            {"seed": seed, "candidate_wins": won, "pair_score": won / 2.0}
            for seed, won in zip(seeds, wins, strict=True)
        ],
        "pair_diagnostics": {
            "incomplete_pairs": 0,
            "ll_pairs": ll_pairs,
            "split_pairs": split_pairs,
            "ww_pairs": ww_pairs,
        },
        "sprt_minus10_plus15": copy.deepcopy(sprt),
        "superiority_sprt_0_plus15": copy.deepcopy(sprt),
    }


def _tiebreak(
    tmp_path: Path, *, fingerprint: dict, incumbent: Path
) -> tuple[Path, str]:
    incumbent_sha = final.file_sha256(incumbent)
    records = {int(item["step"]): item for item in fingerprint["checkpoints"]}
    engine = {
        "schema_version": final.EXPECTED_ENGINE_SCHEMA,
        "repo_commit": "e" * 40,
        "evaluator_sha256": "sha256:" + "1" * 64,
        "native_runtime_sha256": "sha256:" + "2" * 64,
        "native_wheel_sha256": "sha256:" + "3" * 64,
    }
    role_config = {
        role: {
            "c_scale": 0.1,
            "gameplay_policy_aggregation": "mean_improved_policy",
        }
        for role in ("candidate", "champion")
    }
    science_sha = "sha256:" + "4" * 64
    effective_sha = "sha256:" + "5" * 64
    sources = {}

    def write_source(name: str, value: dict) -> None:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value, sort_keys=True))
        sources[name] = {"path": str(path), "sha256": final.file_sha256(path)}

    for step in (8, 16):
        plan = {
            "schema_version": "a1-h100-eval-fleet-plan-v2",
            "operator_mode": "coherent_public",
            "science_config_hash": science_sha,
            "repo_commit": "e" * 40,
            "internal_engine_identity": engine,
            "role_search_config": role_config,
            "candidate": {"sha256": records[step]["checkpoint_sha256"]},
            "champion": {"sha256": incumbent_sha},
        }
        write_source(f"step{step}_plan", plan)
        write_source(f"step{step}_replacement_plan", plan)
        report = {
            "candidate_checkpoint_sha256": records[step]["checkpoint_sha256"],
            "baseline_checkpoint_sha256": incumbent_sha,
            "comparison_contract": final.EXPECTED_COMPARISON,
            "coherent_public_belief_search": True,
            "public_observation": True,
            "correct_rust_chance_spectra": True,
            "native_mcts_hot_loop": True,
            "candidate_n_full": 128,
            "baseline_n_full": 128,
            "candidate_gameplay_policy_aggregation": "mean_improved_policy",
            "baseline_gameplay_policy_aggregation": "mean_improved_policy",
            "errors": [],
            "games_truncated": 0,
            "pairs_truncated_excluded": 0,
            "games_played": 256,
            "complete_pairs": 128,
            "engine_identity": engine,
            "fleet_merge": {"effective_search_config_sha256": effective_sha},
        }
        write_source(f"step{step}_prior", report)
        replacement = {**report, "games_played": 8, "complete_pairs": 4}
        write_source(f"step{step}_replacement", replacement)
        if step == 8:
            fresh = {**report, "games_played": 512, "complete_pairs": 256}
            write_source("step8_fresh", fresh)

    prior_seeds = list(range(6_198_724_000, 6_198_724_128))
    fresh_seeds = [
        seed
        for seed in range(6_198_726_000, 6_198_726_256)
        if seed != 6_198_726_246
    ] + [6_198_728_000]
    step8_prior = [2] + [1] * 127
    step16_prior = [1] * 128
    step8_fresh = [1] * 222 + [0] * 34
    step16_fresh = [1] * 234 + [0] * 22

    def arm(step: int, prior: list[int], fresh: list[int]) -> dict:
        return {
            "checkpoint": {
                "path": records[step]["checkpoint"],
                "sha256": records[step]["checkpoint_sha256"],
            },
            "prior_128": _cohort(prior_seeds, prior, decision="continue"),
            "tie_break_256": _cohort(fresh_seeds, fresh, decision="H0"),
            "combined_384": _cohort(
                prior_seeds + fresh_seeds, prior + fresh, decision="H0"
            ),
        }

    arms = {
        "strategic_step8": arm(8, step8_prior, step8_fresh),
        "strategic_step16": arm(16, step16_prior, step16_fresh),
    }

    def matched(name: str) -> dict:
        left = {
            item["seed"]: item["pair_score"]
            for item in arms["strategic_step8"][name]["pair_outcomes"]
        }
        right = {
            item["seed"]: item["pair_score"]
            for item in arms["strategic_step16"][name]["pair_outcomes"]
        }
        deltas = [
            {"seed": seed, "delta": right[seed] - left[seed]}
            for seed in sorted(left)
        ]
        return {
            "pairs": len(deltas),
            "per_seed_delta": deltas,
            "step16_better_pairs": sum(item["delta"] > 0 for item in deltas),
            "step8_better_pairs": sum(item["delta"] < 0 for item in deltas),
            "same_pairs": sum(item["delta"] == 0 for item in deltas),
            "step16_minus_step8_mean_pair_score": sum(
                item["delta"] for item in deltas
            )
            / len(deltas),
        }

    lanes = [
        {
            "alias": f"host{index // 4}",
            "gpu": index % 4,
            "pairs_requested": 8,
            "complete_pairs": 7 if index == 31 else 8,
            "games_truncated": 1 if index == 31 else 0,
            "path": f"/remote/lane-{index}.json",
            "sha256": f"sha256:{index + 10:064x}",
        }
        for index in range(32)
    ]
    value = {
        "schema_version": final.TIEBREAK_SCHEMA,
        "diagnostic_non_promotable": True,
        "operator_mode": "coherent_public",
        "repo_commit": "e" * 40,
        "science_config_hash": science_sha,
        "internal_engine_identity": engine,
        "role_search_config": role_config,
        "baseline": {
            "source": str(incumbent),
            "remote": str(incumbent),
            "sha256": incumbent_sha,
        },
        "arms": arms,
        "cohort": {
            "combined_pairs": 384,
            "final_tiebreak_pairs": 256,
            "common_seed_set_sha256": "sha256:" + "6" * 64,
            "prior_interval": [6_198_724_000, 6_198_724_128],
            "fresh_original_interval": [6_198_726_000, 6_198_726_256],
            "deterministic_truncated_seed": 6_198_726_246,
            "truncated_orientation": "candidate_red",
            "truncation_decisions": 600,
            "shared_replacement_block": [6_198_728_000, 6_198_728_004],
            "selected_replacement_seed": 6_198_728_000,
            "replacement_selection_rule": (
                "lowest seed in predeclared shared block; selected without "
                "inspecting outcomes"
            ),
        },
        "matched_comparison": {
            "tie_break_256": matched("tie_break_256"),
            "combined_384": matched("combined_384"),
        },
        "selection_result": {
            "winner": "strategic_step16",
            "reason": "matched recovery only; no strength qualification",
        },
        "sources": sources,
        "step16_fresh_lane_reports": lanes,
    }
    path = tmp_path / "tiebreak.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path, incumbent_sha


def test_external_adjudication_selects_balanced_step16_and_replays(
    tmp_path: Path,
) -> None:
    f7 = "sha256:" + "7" * 64
    fingerprint_path, fingerprint = _fingerprint(
        tmp_path, parent_sha=f7, steps=(8, 16)
    )
    incumbent = tmp_path / "v5.pt"
    incumbent.write_bytes(b"authoritative-v5")
    tiebreak_path, incumbent_sha = _tiebreak(
        tmp_path, fingerprint=fingerprint, incumbent=incumbent
    )
    step16 = next(item for item in fingerprint["checkpoints"] if item["step"] == 16)
    f7_report = _panel(
        tmp_path / "step16-f7.json",
        checkpoint=Path(step16["checkpoint"]),
        baseline_sha=f7,
        win_rate=149 / 256,
        decision="H1",
    )
    adjudication = final.build_adjudication(
        fingerprint_path=fingerprint_path,
        tiebreak_adjudication_path=tiebreak_path,
        selected_f7_report_path=f7_report,
    )
    assert adjudication["selected"]["step"] == 16
    assert adjudication["incumbent_checkpoint_sha256"] == incumbent_sha
    assert adjudication["diagnostic_strength_qualification"] is False
    assert adjudication["all_f7_start_finalists_failed_current_parent_h0"] is True
    assert adjudication["selected"]["combined_matched_lift_over_step8"] == 11 / 768
    assert adjudication["selected_checkpoint_is_initializer"] is False
    path = tmp_path / "adjudication.json"
    final._write_json_immutable(path, adjudication)  # noqa: SLF001
    assert final.verify_adjudication(path) == adjudication


def test_fingerprint_refuses_eligible_checkpoint_without_feature_signal(
    tmp_path: Path,
) -> None:
    fingerprint_path, fingerprint = _fingerprint(
        tmp_path, parent_sha="sha256:" + "7" * 64, steps=(16,)
    )
    fingerprint["checkpoints"][0]["eligible"] = True
    fingerprint["checkpoints"][0][
        "feature_learning_signal_authenticated"
    ] = False
    fingerprint.pop("fingerprint_sha256")
    bad_path = fingerprint_path.with_name("bad-fingerprint.json")
    _write_sealed(bad_path, fingerprint, "fingerprint_sha256")

    with pytest.raises(
        final.FinalReplicationError, match="fingerprint semantics drifted"
    ):
        final._fingerprint(bad_path)  # noqa: SLF001


def test_fingerprint_refuses_base_only_teacher_gap_for_aux_campaign(
    tmp_path: Path,
) -> None:
    fingerprint_path, fingerprint = _fingerprint(
        tmp_path, parent_sha="sha256:" + "7" * 64, steps=(16,)
    )
    base_recipe = copy.deepcopy(
        json.loads(Path(fingerprint["campaign"]["path"]).read_text())["recipe"]
    )
    base_recipe["policy_aux_active_batch_size"] = 0
    base_recipe["policy_aux_loss_weight"] = 0.0
    fingerprint["policy_teacher_gap_objective"] = (
        final.stage_c_campaign._expected_policy_teacher_gap_objective(  # noqa: SLF001
            base_recipe
        )
    )
    fingerprint.pop("fingerprint_sha256")
    _write_sealed(fingerprint_path, fingerprint, "fingerprint_sha256")

    with pytest.raises(
        final.FinalReplicationError, match="fingerprint semantics drifted"
    ):
        final._fingerprint(fingerprint_path)  # noqa: SLF001


@pytest.mark.parametrize(
    "mutation",
    (
        "schema",
        "feature_contract",
        "value_gate",
        "behavioral_gate",
        "policy_aux_objective",
    ),
)
def test_fingerprint_refuses_campaign_contract_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    fingerprint_path, fingerprint = _fingerprint(
        tmp_path, parent_sha="sha256:" + "7" * 64, steps=(16,)
    )
    campaign_path = Path(fingerprint["campaign"]["path"])
    campaign = json.loads(campaign_path.read_text())
    campaign.pop("campaign_sha256")
    if mutation == "schema":
        campaign["schema_version"] = "a1-b200-stage-c-aligned-learner-campaign-v5"
    elif mutation == "feature_contract":
        campaign["feature_learning_signal_contract"].pop("required_modules")
    elif mutation == "policy_aux_objective":
        campaign["recipe"]["policy_aux_loss_weight"] = 0.0
    elif mutation == "behavioral_gate":
        campaign["selection_contract"]["behavioral_competence_gate"][
            "minimum_rows"
        ] = 63
    else:
        campaign["selection_contract"]["value_quality_gate"][
            "max_absolute_regression"
        ] = 0.01
    campaign = _write_sealed(campaign_path, campaign, "campaign_sha256")
    fingerprint["campaign"] = {
        **final._artifact(campaign_path),  # noqa: SLF001
        "campaign_sha256": campaign["campaign_sha256"],
    }
    fingerprint.pop("fingerprint_sha256")
    _write_sealed(fingerprint_path, fingerprint, "fingerprint_sha256")

    with pytest.raises(
        final.FinalReplicationError, match="campaign semantics drifted"
    ):
        final._fingerprint(fingerprint_path)  # noqa: SLF001


def test_behavioral_competence_gate_contract_is_exact() -> None:
    assert (
        final._expected_behavioral_competence_gate_contract()  # noqa: SLF001
        == _behavioral_competence_gate_contract()
    )


def test_fingerprint_refuses_top_level_behavioral_contract_drift(
    tmp_path: Path,
) -> None:
    fingerprint_path, fingerprint = _fingerprint(
        tmp_path, parent_sha="sha256:" + "7" * 64, steps=(16,)
    )
    fingerprint["behavioral_competence_gate"]["required_streams"] = ["base"]
    fingerprint.pop("fingerprint_sha256")
    _write_sealed(fingerprint_path, fingerprint, "fingerprint_sha256")

    with pytest.raises(
        final.FinalReplicationError, match="campaign semantics drifted"
    ):
        final._fingerprint(fingerprint_path)  # noqa: SLF001


def test_fingerprint_refuses_eligible_checkpoint_without_behavioral_admission(
    tmp_path: Path,
) -> None:
    fingerprint_path, fingerprint = _fingerprint(
        tmp_path, parent_sha="sha256:" + "7" * 64, steps=(16,)
    )
    record = fingerprint["checkpoints"][0]
    record["eligible"] = True
    record["behavioral_competence_gate"] = (
        _failed_behavioral_competence_gate()
    )
    fingerprint.pop("fingerprint_sha256")
    _write_sealed(fingerprint_path, fingerprint, "fingerprint_sha256")

    with pytest.raises(
        final.FinalReplicationError, match="fingerprint semantics drifted"
    ):
        final._fingerprint(fingerprint_path)  # noqa: SLF001


def test_checkpoint_record_cannot_bypass_behavioral_admission(
    tmp_path: Path,
) -> None:
    fingerprint_path, fingerprint = _fingerprint(
        tmp_path, parent_sha="sha256:" + "7" * 64, steps=(16,)
    )
    record = fingerprint["checkpoints"][0]
    assert record["eligible"] is False
    record["behavioral_competence_gate"] = (
        _failed_behavioral_competence_gate()
    )
    fingerprint.pop("fingerprint_sha256")
    _write_sealed(fingerprint_path, fingerprint, "fingerprint_sha256")
    _path, verified = final._fingerprint(fingerprint_path)  # noqa: SLF001

    with pytest.raises(
        final.FinalReplicationError, match="behavioral competence"
    ):
        final._checkpoint_record(  # noqa: SLF001
            verified,
            16,
            require_fingerprint_eligible=False,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "stored_count",
        "action_catalog_abi",
        "confidence_bound",
        "paired_delta",
        "objective_weighted_diagnostic",
        "missing_aux_stream",
    ),
)
def test_fingerprint_recomputes_behavioral_competence_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    fingerprint_path, fingerprint = _fingerprint(
        tmp_path, parent_sha="sha256:" + "7" * 64, steps=(16,)
    )
    gate = fingerprint["checkpoints"][0]["behavioral_competence_gate"]
    base = gate["evidence"]["streams"]["base"]
    if mutation == "stored_count":
        base["candidate"]["teacher_top1_successes"] += 1
    elif mutation == "action_catalog_abi":
        base["action_catalog_abi"]["identity_sha256"] = "sha256:" + "f" * 64
    elif mutation == "confidence_bound":
        base["confidence_bounds"]["teacher_top1_lower"] += 0.01
    elif mutation == "paired_delta":
        base["candidate_minus_parent"]["teacher_top1_accuracy"] += 0.01
    elif mutation == "objective_weighted_diagnostic":
        base["objective_weighted_diagnostic"]["candidate"][
            "teacher_top1_accuracy"
        ] += 0.01
    else:
        gate["evidence"]["streams"].pop("aux")
    fingerprint.pop("fingerprint_sha256")
    _write_sealed(fingerprint_path, fingerprint, "fingerprint_sha256")

    with pytest.raises(
        final.FinalReplicationError, match="fingerprint semantics drifted"
    ):
        final._fingerprint(fingerprint_path)  # noqa: SLF001


@pytest.mark.parametrize(
    "mutation",
    ("missing_report_binding", "contradictory_feature_evidence"),
)
def test_checkpoint_record_refuses_unbound_local_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    fingerprint_path, fingerprint = _fingerprint(
        tmp_path, parent_sha="sha256:" + "7" * 64, steps=(16,)
    )
    record = fingerprint["checkpoints"][0]
    if mutation == "missing_report_binding":
        record.pop("checkpoint_report_binding")
    else:
        record["feature_learning_signal"]["modules"][
            next(iter(final.stage_c_campaign.FEATURE_SIGNAL_MODULES))
        ]["mean_parameter_update_rms"] = 0.0
    fingerprint.pop("fingerprint_sha256")
    _write_sealed(fingerprint_path, fingerprint, "fingerprint_sha256")
    _path, verified = final._fingerprint(fingerprint_path)  # noqa: SLF001

    with pytest.raises(
        final.FinalReplicationError,
        match="checkpoint report binding|feature evidence",
    ):
        final._checkpoint_record(  # noqa: SLF001
            verified,
            16,
            require_fingerprint_eligible=False,
        )


@pytest.mark.parametrize("mutation", ("missing_trajectory", "invented_signal"))
def test_checkpoint_record_reconciles_feature_evidence_to_completed_report(
    tmp_path: Path,
    mutation: str,
) -> None:
    fingerprint_path, fingerprint = _fingerprint(
        tmp_path, parent_sha="sha256:" + "7" * 64, steps=(16,)
    )
    report_path = Path(fingerprint["completed_dose"]["report"]["path"])
    report = json.loads(report_path.read_text())
    if mutation == "missing_trajectory":
        report.pop("checkpoint_dose_trajectory")
    else:
        report["checkpoint_dose_trajectory"]["checkpoints"][0][
            "module_optimizer_observability"
        ]["modules"][
            next(iter(final.stage_c_campaign.FEATURE_SIGNAL_MODULES))
        ]["mean_parameter_update_rms"] = 0.123
    report_path.write_text(json.dumps(report, sort_keys=True))
    fingerprint["completed_dose"]["report"]["file_sha256"] = final.file_sha256(
        report_path
    )
    fingerprint.pop("fingerprint_sha256")
    _write_sealed(fingerprint_path, fingerprint, "fingerprint_sha256")
    _path, verified = final._fingerprint(fingerprint_path)  # noqa: SLF001

    with pytest.raises(
        final.FinalReplicationError,
        match="feature trajectory is invalid|feature evidence differs from report",
    ):
        final._checkpoint_record(  # noqa: SLF001
            verified,
            16,
            require_fingerprint_eligible=False,
        )


def test_root_manifest_refuses_diagnostic_or_eval_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production_plan = tmp_path / "production-plan.json"
    production_plan.write_text("{}")
    subset = tmp_path / "production-roots.npz"
    training_games = np.arange(100_000, 107_782, dtype=np.int64)
    validation_games = np.arange(200_000, 200_410, dtype=np.int64)
    selected_games = np.repeat(
        np.concatenate((training_games, validation_games)), 8
    )
    realized_count = int(selected_games.size)
    identities = np.asarray(
        [f"sha256:{index:064x}" for index in range(realized_count)]
    )
    game_seeds = selected_games
    row_indices = np.arange(realized_count, dtype=np.int64)
    decision_cycle = np.asarray([5, 15, 35, 65, 105, 155, 205, 7], dtype=np.int64)
    decision_indices = np.tile(decision_cycle, len(selected_games) // 8)
    phase_cycle = np.asarray(
        [
            "BUILD_INITIAL_ROAD",
            "BUILD_INITIAL_SETTLEMENT",
            "DISCARD",
            "MOVE_ROBBER",
            "PLAY_TURN",
            "PLAY_TURN",
            "PLAY_TURN",
            "PLAY_TURN",
        ]
    )
    phases = np.tile(phase_cycle, len(selected_games) // 8)
    np.savez(
        subset,
        identity_sha256=identities,
        game_seed=game_seeds,
        row_index=row_indices,
        decision_index=decision_indices,
        phase=phases,
    )
    ready = [
        {
            "row_index": int(row_indices[index]),
            "game_seed": int(game_seeds[index]),
            "decision_index": int(decision_indices[index]),
            "identity_sha256": str(identities[index]),
        }
        for index in range(realized_count)
    ]
    validation_manifest_sha = "sha256:" + "a" * 64
    validation_seed_set_sha = "sha256:" + "b" * 64
    validation_split = {
        "validation_row_count": 4_096,
        "validation_game_seed_count": len(validation_games),
        "validation_game_seed_set_sha256": validation_seed_set_sha,
    }
    trainer_exclusion = {
        "schema_version": final.alignment.TRAINER_EXCLUSION_CONTRACT_SCHEMA,
        "input_validation_manifest_file_sha256": validation_manifest_sha,
        "training_excluded_game_seed_count": len(validation_games),
        "training_excluded_game_seed_set_sha256": validation_seed_set_sha,
    }
    learner_validation_scope = {
        "schema_version": final.alignment.LEARNER_VALIDATION_SCOPE_SCHEMA,
        "manifest": {
            "path": str(tmp_path / "validation.json"),
            "file_sha256": validation_manifest_sha,
            "manifest_sha256": "sha256:" + "c" * 64,
            "a1_contract_sha256": "sha256:" + "d" * 64,
        },
        "split_receipt": validation_split,
        "trainer_exclusion_contract": trainer_exclusion,
        "target_coverage_receipt": {},
        "external_final_gate_authority": False,
    }
    prep_value = {
        "schema_version": final.PREP_INVENTORY_SCHEMA,
        "authority": {"is_authority": False, "may_launch_search": False},
        "fully_reconstructable_ready_roots": ready,
        "proof": {
            "satisfied": True,
            "independent_from_all_declared_eval_pair_seeds": True,
            "independent_from_diagnostic_selected_rows": True,
            "learner_validation_manifest_file_sha256": validation_manifest_sha,
            "learner_validation_scope_sha256": "",
            "learner_validation_split_receipt_sha256": final.value_sha256(
                validation_split
            ),
            "learner_trainer_exclusion_contract_sha256": final.value_sha256(
                trainer_exclusion
            ),
            "policy_root_breadth_inventory_sha256": "",
            "required_fully_reconstructable_strategic_roots": realized_count,
            "observed_fully_reconstructable_strategic_roots": realized_count,
            "ready_root_prefix_count": realized_count,
            "ready_root_prefix_sha256": final.value_sha256(ready),
        },
    }
    prep = tmp_path / "prep.json"
    _write_sealed(prep, prep_value, "inventory_sha256")
    root_breadth = final.alignment._stage_c_root_breadth_inventory(  # noqa: SLF001
        corpus_game_seeds=np.concatenate((training_games, validation_games)),
        validation_game_seeds=validation_games,
        selected_game_seeds=game_seeds,
        selected_decision_indices=decision_indices,
        selected_phases=phases,
    )
    learner_validation_scope["target_coverage_receipt"] = {
        "root_breadth_inventory_sha256": root_breadth["inventory_sha256"],
        "selected_validation_root_count": root_breadth["scopes"]["validation"][
            "selected_root_count"
        ],
        "selected_validation_game_count": root_breadth["scopes"]["validation"][
            "selected_game_count"
        ],
    }
    learner_validation_scope["scope_sha256"] = final.value_sha256(
        learner_validation_scope
    )
    prep_value["proof"]["learner_validation_scope_sha256"] = (
        learner_validation_scope["scope_sha256"]
    )
    prep_value["proof"]["policy_root_breadth_inventory_sha256"] = root_breadth[
        "inventory_sha256"
    ]
    _write_sealed(prep, prep_value, "inventory_sha256")
    plan = {
        "plan_sha256": "sha256:" + "p" * 64,
        "learner_validation_scope": learner_validation_scope,
        "subset": {
            "artifact": {"path": str(subset)},
            "selected_rows": realized_count,
            "requested_rows": realized_count,
            "chunks": 7,
            "selection_seed": 42,
            "game_first_selection": {"root_breadth": root_breadth},
        },
        "execution": {"executor_semantics": final.EXPECTED_EXECUTOR},
        "target_policy_target_identity": {
            "target_information_regime": final.EXPECTED_OPERATOR,
            "search_operator": {"n_full": 128},
        },
    }
    monkeypatch.setattr(final.alignment, "_verify_plan", lambda _path: plan)
    forbidden = tmp_path / "diagnostic-roots.npz"
    np.savez(
        forbidden,
        identity_sha256=np.asarray(["sha256:" + "f" * 64]),
        game_seed=np.asarray([400_000], dtype=np.int64),
    )
    eval_report = tmp_path / "eval.json"
    eval_report.write_text(
        json.dumps(
            {
                "games": [
                    {"game_seed": 300_000 + pair}
                    for pair in range(final.EXPECTED_PAIRS)
                    for _orientation in range(2)
                ]
            }
        )
    )
    manifest = final.build_root_manifest(
        production_plan_path=production_plan,
        prep_inventory_path=prep,
        forbidden_subset_paths=[forbidden],
        forbidden_eval_paths=[eval_report],
    )
    assert manifest["root_count"] == realized_count
    assert manifest["requested_root_budget"] == realized_count
    assert manifest["partition_count"] == 7
    assert (
        manifest["learner_validation_scope"]["scope_sha256"]
        == learner_validation_scope["scope_sha256"]
    )
    assert manifest["diagnostic_root_overlap_count"] == 0

    bad_prep_value = copy.deepcopy(prep_value)
    bad_prep_value["proof"]["learner_validation_manifest_file_sha256"] = (
        "sha256:" + "c" * 64
    )
    bad_prep = tmp_path / "prep-bad-validation.json"
    _write_sealed(bad_prep, bad_prep_value, "inventory_sha256")
    with pytest.raises(
        final.FinalReplicationError,
        match="preparation evidence drifted",
    ):
        final.build_root_manifest(
            production_plan_path=production_plan,
            prep_inventory_path=bad_prep,
            forbidden_subset_paths=[forbidden],
            forbidden_eval_paths=[eval_report],
        )

    np.savez(
        forbidden,
        identity_sha256=np.asarray([identities[0]]),
        game_seed=np.asarray([game_seeds[0]], dtype=np.int64),
    )
    with pytest.raises(final.FinalReplicationError, match="overlaps"):
        final.build_root_manifest(
            production_plan_path=production_plan,
            prep_inventory_path=prep,
            forbidden_subset_paths=[forbidden],
            forbidden_eval_paths=[eval_report],
        )


def test_one_dose_final_binder_reloads_current_v5_not_diagnostic_f7(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = {"path": str(tmp_path / "v5.pt"), "sha256": "sha256:" + "5" * 64}
    Path(current["path"]).write_bytes(b"v5")
    current["sha256"] = one_dose._file_sha256(Path(current["path"]))
    initializer = {
        "path": str(tmp_path / "v5-upgraded.pt"),
        "sha256": "sha256:" + "6" * 64,
    }
    Path(initializer["path"]).write_bytes(b"v5 upgraded")
    initializer["sha256"] = one_dose._file_sha256(Path(initializer["path"]))
    admission_path = tmp_path / "final-admission.json"
    admission_path.write_text("{}")
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}")
    authority_path = tmp_path / "authority.json"
    authority_path.write_text("{}")
    base_recipe = dict(contract.EXPECTED_LEARNER_TRAINING_RECIPE)
    selected_recipe = {
        "epochs": 1,
        "max_steps": 12,
        "lr": 6.0e-5,
        "lr_warmup_steps": 16,
        "policy_loss_weight": 1.0,
        "value_loss_weight": 0.25,
        "policy_aux_active_batch_size": 64,
        "soft_target_source": "policy",
        "soft_target_weight": 1.0,
        "soft_target_min_legal_coverage": 1.0,
        "public_card_lr_mult": 1.0,
        "trunk_lr_mult": 1.0,
        "value_lr_mult": 0.3,
        "value_trunk_grad_scale": 0.1,
        "per_game_policy_surprise_weighting": False,
        "policy_kl_anchor_weight": 0.0,
        "forced_row_value_action_type_weights": "END_TURN=1,ROLL=1",
        "sampler_seed": 777,
    }
    authority = {
        "authority_sha256": "sha256:" + "a" * 64,
        "final_corpus_admission": {
            "path": str(admission_path),
            "file_sha256": one_dose._file_sha256(admission_path),
            "admission_sha256": "sha256:" + "c" * 64,
        },
        "initializer": {
            "exact_parent": current,
            "upgraded_initializer": initializer,
            "upgrade_receipt_sha256": "sha256:" + "u" * 64,
            "fresh_adam": True,
            "resume_optimizer": False,
            "candidate_chaining": False,
        },
        "training": {
            "matched_arms": {
                final.FINAL_CONTROL_ARM: {
                    "role": "exact_v5_terminal_value_control",
                    "recipe": selected_recipe,
                    "recipe_sha256": one_dose._value_sha256(selected_recipe),
                    "value_target": "terminal_outcome_only",
                }
            },
            "max_optimizer_steps": 12,
            "checkpoint_steps": [8, 10],
        },
        "reviewed_code": {
            "lock": {
                "path": str(lock_path),
                "file_sha256": one_dose._file_sha256(lock_path),
            },
            "code_tree_sha256": "sha256:" + "d" * 64,
        },
        "diagnostic_selection": {
            "selected_step": 16,
            "selected_diagnostic_checkpoint_sha256": "sha256:" + "f" * 64,
        },
        "external_adjudication": {"adjudication_sha256": "sha256:" + "e" * 64},
    }
    monkeypatch.setattr(
        one_dose.stage_c_final, "verify_final_authority", lambda _path: authority
    )
    monkeypatch.setattr(
        one_dose,
        "_current_ablation_code_binding",
        lambda _lock: {"code_tree_sha256": "sha256:" + "d" * 64},
    )
    verified = {
        "lock": {},
        "lock_path": lock_path,
        "lock_file_sha256": one_dose._file_sha256(lock_path),
        "reviewed_lock_file_sha256": one_dose._file_sha256(lock_path),
        "recipe": base_recipe,
        "producer": current,
        "function_preserving_upgrade": {
            "source": current,
            "upgraded_initializer": initializer,
            "receipt_sha256": "sha256:" + "u" * 64,
        },
        "stage_c_final_corpus_admission": {
            "admission_sha256": "sha256:" + "c" * 64,
            "root_manifest": {"root_manifest_sha256": "sha256:" + "r" * 64},
            "search_value_evidence": {
                "naive_root_blend_authorized": False,
                "terminal_target_remains_authoritative": True,
            },
        },
        "coherent_direct_corpus_binding": {
            "corpus_admission": {"path": str(admission_path)}
        },
    }
    bound = one_dose.bind_stage_c_final_replication(
        verified,
        authority_path=authority_path,
        arm_name=final.FINAL_CONTROL_ARM,
        reviewed_code_tree_sha256="sha256:" + "d" * 64,
    )
    assert bound["recipe"]["max_steps"] == 12
    assert (
        bound["stage_c_final_replication_binding"][
            "initializer_checkpoint_sha256"
        ]
        == initializer["sha256"]
    )
    assert (
        bound["stage_c_final_replication_binding"][
            "selected_diagnostic_checkpoint_loaded"
        ]
        is False
    )
    bad = copy.deepcopy(verified)
    bad["producer"] = {"path": "/f7.pt", "sha256": "sha256:" + "7" * 64}
    with pytest.raises(one_dose.ExecutorError, match="initializer drifted"):
        one_dose.bind_stage_c_final_replication(
            bad,
            authority_path=authority_path,
            arm_name=final.FINAL_CONTROL_ARM,
            reviewed_code_tree_sha256="sha256:" + "d" * 64,
        )
