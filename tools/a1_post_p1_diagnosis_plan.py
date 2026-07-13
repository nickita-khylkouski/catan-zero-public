#!/usr/bin/env python3
"""Emit the Pareto post-P1 learner/architecture probe sequence.

The plan reuses completed evidence instead of scheduling it again.  In
particular, the matched policy-active-dose arm already scored 596-604 over
1,200 games and is retained as a negative result, not another B200 job.  Every
new arm changes one causal axis from the selected TEMP/replay-scope control;
the zero-init target-gather arm therefore does not silently add a policy-active
auxiliary batch.  Full relational warm-starts and Cartesian sweeps are
forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "a1-post-p1-optimization-architecture-plan-v5"
SHORT_SAMPLE_DOSE = 524_288
FULL_SAMPLE_DOSE = 4_194_304
# The matched common-random-number behavior screen selected the short dose.
# The full dose remains evidence below, never the default for a new arm.
SAMPLE_DOSE = SHORT_SAMPLE_DOSE

SHORT_CHECKPOINT = {
    "path": (
        "/home/ubuntu/experimental_nonpromotable/learner-forensics-2ba5ae1/"
        "geometry-probe-128step-r4/ddp8-b512/candidate.pt"
    ),
    "sha256": "sha256:91a4b63ee5b74da0ec7123557d36e084d7ab91963c65bab9848838123e8e86de",
    "report_sha256": (
        "sha256:c28bb2a6992562947db635611150641d124074c5d6c8f08ee399a17ea9bff758"
    ),
    "layer_drift_sha256": (
        "sha256:5fbe47e3746b1c1a17ba07cfeb68e9edd9b605b0167b21b2d2e1817d56d1f936"
    ),
}
FULL_CHECKPOINT = {
    "path": (
        "/home/ubuntu/experimental_nonpromotable/learner-forensics-84c12e9/"
        "p0-temp-f7-dose4194304-r1/ddp8-b512/candidate.pt"
    ),
    "sha256": "sha256:ce29663fe519b88537d54afec3dfa4e0033f79a649f8b04d364baead48c462f4",
    "report_sha256": (
        "sha256:4dbfa0b28156d482eae9f01e3a80bf450e0fb6d71f1e2dc4495293658d8779de"
    ),
    "layer_drift_sha256": (
        "sha256:de680388c594731b9218d7f67b1f53cec54d3758f1cb568b07cc158b2d68f237"
    ),
}


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _integrated_lr_step_equivalents(*, steps: int, warmup_steps: int) -> float:
    """Area under the trainer's linear-warmup learning-rate multiplier."""

    if steps < 0 or warmup_steps < 0:
        raise ValueError("steps and warmup_steps must be non-negative")
    if warmup_steps == 0:
        return float(steps)
    return float(
        sum(min(float(step + 1) / float(warmup_steps), 1.0) for step in range(steps))
    )


def build_plan(
    *, world_size: int = 8, local_batch_size: int = 512, grad_accum_steps: int = 1
) -> dict[str, Any]:
    if world_size != 8:
        raise ValueError("post-P1 diagnosis is bound to all eight B200 GPUs")
    if local_batch_size != 512:
        raise ValueError("post-P1 diagnosis is bound to selected local batch 512")
    if grad_accum_steps != 1:
        raise ValueError(
            "post-P1 diagnosis requires exact DDP accumulation 1; mean-of-means "
            "accumulation is not promotion-safe"
        )
    global_batch = world_size * local_batch_size * grad_accum_steps
    steps_by_dose = {
        str(dose): math.ceil(dose / global_batch)
        for dose in (SHORT_SAMPLE_DOSE, FULL_SAMPLE_DOSE)
    }
    lr_equivalents_by_dose = {
        dose: _integrated_lr_step_equivalents(steps=steps, warmup_steps=100)
        for dose, steps in steps_by_dose.items()
    }
    fixed = {
        "initialization": "authenticated f7 producer bytes, independently loaded per arm",
        "data": "exact P1 globally shuffled current-plus-gen3-replay descriptor",
        "sample_order": "same component/game/row sampler seed as P1",
        # The matched screens are complete: short scored 75-53 and full scored
        # 65-63 against exact f7 on the same 128 (seed, orientation) keys.  The
        # preregistered Pareto rule therefore selects the short dose.  Keep the
        # full checkpoint as dose-response evidence, not a default continuation.
        "sample_dose": SAMPLE_DOSE,
        "max_steps": steps_by_dose[str(SAMPLE_DOSE)],
        "dose_selection": "matched_behavior_pareto_rule",
        "dose_candidates": [SHORT_SAMPLE_DOSE, FULL_SAMPLE_DOSE],
        "max_steps_by_dose": steps_by_dose,
        "world_size": world_size,
        "local_batch_size": local_batch_size,
        "global_batch_size": global_batch,
        "grad_accum_steps": grad_accum_steps,
        "optimizer": "fresh Adam",
        "lr": 3e-5,
        "lr_warmup_steps": 100,
        "lr_schedule": "flat",
        "trunk_lr_mult": 1.0,
        "weight_decay": 0.0,
        # The sealed winning TEMP learner and both dose artifacts are FP32.
        # BF16 here was a stale P1-plan value and would make every purported
        # one-axis learner arm a precision-plus-treatment experiment.
        "amp": "none",
        "policy_kl_anchor_weight": 0.0,
        "policy_kl_anchor_direction": "forward",
        # Exact winning TEMP/P0 scope. Current-only replay scope is plausible,
        # but it is an ablation—not the control—and policy CE versus stale-z are
        # two independent hypotheses.
        "policy_distillation_component_ids": [
            "n128_current",
            "n256_current",
            "gen3_replay",
        ],
        "value_training_component_ids": [
            "n128_current",
            "n256_current",
            "gen3_replay",
        ],
        "gen3_replay_policy_objective": (
            "exact TEMP control: stored search-target CE plus old-policy Monte-Carlo "
            "outcome; no KL anchor"
        ),
        "policy_loss_weight": 1.0,
        "soft_target_source": "policy",
        "soft_target_weight": 0.9,
        "soft_target_temperature": 0.7,
        "soft_target_min_legal_coverage": 0.5,
        "stored_policy_component_temperatures": {
            "n128_current": 1.0,
            "n256_current": 1.11,
            "gen3_replay": 0.52,
        },
        "forced_policy_effective_weight": 0.0,
        "forced_action_weight": 0.0,
        "forced_row_value_weight": 1.0,
        "per_game_policy_weight": False,
        "per_game_policy_weight_mode": "equal",
        "per_game_value_weight": False,
        "per_game_value_weight_mode": "equal",
        "loser_sample_weight": 1.0,
        "winner_sample_weight": 1.0,
        "value_loss_weight": 0.25,
        "final_vp_loss_weight": 0.0,
        "value_lr_mult": 0.3,
        "value_target_lambda": 1.0,
        "truncated_vp_margin_value_weight": 0.0,
        "q_loss_weight": 0.0,
        "aux_subgoal_heads": False,
        "aux_subgoal_loss_weight": 0.0,
        "graph_history_feature_flag": True,
        "event_history_available": False,
        "event_history_semantics": (
            "Rust v1 emits constant-zero event tensors; the 806-wide schema flag "
            "must not be reported as observed history"
        ),
        "mask_hidden_info": True,
        "symmetry_augment": False,
        "validation": "same game-disjoint P1 validation identity",
        "initialization_policy": (
            "independent authenticated f7 producer bytes; checkpoint chaining forbidden"
        ),
        "lineage_dose_schema": "a1-lineage-dose-v1",
    }
    arms = [
        {
            "arm_id": "LEGACY_L03",
            "training": "reuse recovered P1 K0 checkpoint; no new B200 run",
            "recipe_delta": {"loser_sample_weight": 0.3},
            "purpose": (
                "observed asymmetric-forgetting evidence; never use as the "
                "loser=1 control"
            ),
        },
        {
            "arm_id": "TEMP_CONTROL",
            "training": "new matched B200 run",
            "recipe_delta": {
                "checkpoint_upgrade": "none",
                "policy_aux_active_batch_size": 0,
            },
            "purpose": "replicate the exact winning TEMP objective at the selected dose",
        },
        {
            "arm_id": "CURRENT_POLICY_SCOPE",
            "training": "new matched B200 run after TEMP_CONTROL",
            "reference_arm": "TEMP_CONTROL",
            "recipe_delta": {
                "policy_distillation_component_ids": [
                    "n128_current",
                    "n256_current",
                ]
            },
            "purpose": (
                "test only removal of the older replay search-policy teacher; retain "
                "the exact TEMP value scope"
            ),
        },
        {
            "arm_id": "CURRENT_VALUE_SCOPE",
            "training": "new matched B200 run after TEMP_CONTROL",
            "reference_arm": "TEMP_CONTROL",
            "recipe_delta": {
                "value_training_component_ids": [
                    "n128_current",
                    "n256_current",
                ]
            },
            "purpose": (
                "test only removal of old-policy Monte-Carlo outcomes; retain the "
                "exact TEMP policy-distillation scope"
            ),
        },
        {
            "arm_id": "L1_PURE_SEARCH_TARGET",
            "training": "new matched B200 run after replay-scope selection",
            "reference_arm": "selected replay-scope control",
            "recipe_delta": {
                "checkpoint_upgrade": "none",
                "policy_aux_active_batch_size": 0,
                "soft_target_weight": 1.0,
            },
            "purpose": (
                "remove deterministic post-temperature argmax re-sharpening and "
                "distill the completed-Q improved policy exactly"
            ),
        },
        {
            "arm_id": "L1_POLICY_AUX",
            "training": "reuse completed matched result; no new B200 run",
            "reference_arm": "selected replay-scope control",
            "recipe_delta": {
                "checkpoint_upgrade": "none",
                "policy_aux_active_batch_size": 128,
            },
            "purpose": (
                "retain the completed negative policy-active-dose result so it is "
                "not accidentally repeated or bundled into an architecture arm"
            ),
            "completed_evidence": {
                "candidate_wins": 596,
                "control_wins": 604,
                "games": 1200,
                "candidate_score": 0.496667,
                "errors": 0,
                "truncations": 0,
                "ruling": "no demonstrated improvement; do not repeat",
                "audit": "docs/audits/A1_POLICY_AUX_REPLICATION_20260712.md",
            },
        },
        {
            "arm_id": "L1_REPLAY_ANCHOR",
            "training": "conditional matched B200 run after exact-v2 replay telemetry",
            "reference_arm": "selected replay-scope control",
            "recipe_delta": {
                "checkpoint_upgrade": "none",
                "policy_aux_active_batch_size": 0,
                "policy_kl_anchor_scope": "authenticated gen3_replay multi-action rows",
                "policy_kl_anchor_direction": "forward (KL(stored_prior || model))",
                "policy_kl_anchor_weight": "choose from exact eligible-mass telemetry",
            },
            "purpose": (
                "test only preservation of the authenticated incumbent-era replay "
                "population; policy-active dose remains zero so the anchor coefficient "
                "is the sole treatment"
            ),
        },
        {
            "arm_id": "L1_GATHER",
            "training": "new matched B200 run",
            "reference_arm": "selected corrected learner arm",
            "recipe_delta": {
                "checkpoint_upgrade": "f69_upgrade_checkpoint_config.py --flags gather",
                "checkpoint_upgrade_forward_max_diff": 0.0,
                "policy_aux_active_batch_size": 0,
            },
            "purpose": (
                "isolate learned legal-action-to-target binding over the selected "
                "corrected learner without changing policy-active dose"
            ),
        },
    ]
    for arm in arms:
        arm["recipe_sha256"] = _digest({"fixed": fixed, "delta": arm["recipe_delta"]})
    evaluation = {
        "checkpoint_identity": (
            "compare candidate and exact f7 with the same deployed c_scale=0.10 "
            "operator; candidate-specific operator tuning is a separate crossover "
            "experiment and cannot select or relabel a learner arm"
        ),
        "matched_search_operator": {
            "candidate_c_scale": 0.10,
            "baseline_c_scale": 0.10,
            "selection_tuning_allowed": False,
            "reason": (
                "changing checkpoint ancestry and c_scale in one comparison repeats "
                "the historical gen3/.03 adjudication confound"
            ),
        },
        "internal": {
            "pairs": 300,
            "games": 600,
            "opponent": "f7 producer",
            "map_kind": "BASE",
            "seat_swapped": True,
            "common_random_numbers": True,
        },
        "external": {
            "pairs": 250,
            "games_per_candidate": 500,
            "opponent": "catanatron_value",
            "compare_to": "f7 on the identical seeds",
            "common_random_numbers": True,
            "map_kind": "TOURNAMENT",
        },
        "direct_tournament_bridge": {
            "pairs": 300,
            "opponent": "f7 producer",
            "map_kind": "TOURNAMENT",
            "purpose": "separate map specialization from opponent nontransitivity",
        },
        "binding_floor": "external win-rate delta versus f7 >= -2 percentage points",
        "promotion_confirmation": (
            "the selected arm must use a fresh seed cohort disjoint from every "
            "diagnostic/adjudication cohort used to select it"
        ),
        "diagnostics": [
            "active-only teacher-gap and target-to-model KL",
            "value RMSE/calibration by phase and root width",
            "parameter drift split into trunk/action/value groups",
            "model-to-f7 KL on multi-action rows",
            (
                "weighted policy-vs-value shared-trunk gradient norms, cosine, "
                "opposing-coordinate fraction, and per-block concentration; collect "
                "on a single-GPU/DDP diagnostic because FSDP flattens logical blocks"
            ),
            "gradient clipping and non-finite telemetry",
        ],
    }
    dose_adjudication = {
        "schema_version": "a1-dose-adjudication-v1",
        "status": "selected",
        "launch_blocking": False,
        "reason": (
            "the short run reached 0.102290 teacher-gap closure at 0.6913% global "
            "drift; the full run reached only 0.135757 closure at 2.5954% drift"
        ),
        "common_contract": {
            "initializer": "exact authenticated f7 producer bytes",
            "optimizer": "fresh Adam",
            "global_batch_size": 4096,
            "lr": 3e-5,
            "lr_warmup_steps": 100,
            "lr_schedule": "flat",
            "amp": "none",
            "sample_order_prefix": "identical authenticated P1 sampler seed",
        },
        "candidates": {
            str(SHORT_SAMPLE_DOSE): {
                "optimizer_steps": steps_by_dose[str(SHORT_SAMPLE_DOSE)],
                "integrated_lr_step_equivalents": lr_equivalents_by_dose[
                    str(SHORT_SAMPLE_DOSE)
                ],
                "teacher_gap_closure": 0.102290,
                "global_relative_parameter_drift": 0.0069134,
                "artifact": SHORT_CHECKPOINT,
            },
            str(FULL_SAMPLE_DOSE): {
                "optimizer_steps": steps_by_dose[str(FULL_SAMPLE_DOSE)],
                "integrated_lr_step_equivalents": lr_equivalents_by_dose[
                    str(FULL_SAMPLE_DOSE)
                ],
                "teacher_gap_closure": 0.135757,
                "global_relative_parameter_drift": 0.025954,
                "artifact": FULL_CHECKPOINT,
            },
        },
        "evaluation": {
            "primary": (
                "short versus full, seat-swapped with common random numbers under "
                "the exact deployed f7 tanh search operator"
            ),
            "parent_panel": (
                "short and full versus exact f7 on the same seed cohort; never use "
                "old gen3 or c_scale=0.03"
            ),
            "pairs": 300,
            "map_kind": "BASE",
            "c_scale": 0.10,
            "value_squash": "tanh",
            "operator_sensitivity": (
                "repeat short/full/f7 on the identical seed cohort with value_squash=clip; "
                "diagnostic only, never mix tanh and clip inside one match"
            ),
            "heldout_value_calibration": ["raw", "tanh", "clip"],
        },
        "selection_rule": (
            "select the smallest dose whose paired win rate is within 2 percentage "
            "points of the best dose and whose f7 panel is within 2 points of the "
            "best f7 delta; otherwise select the statistically superior dose"
        ),
        "observed_behavior": {
            "common_seed_orientation_keys": 128,
            str(SHORT_SAMPLE_DOSE): {
                "candidate_wins": 75,
                "baseline_wins": 53,
                "win_rate": 0.5859375,
                "errors": 0,
                "truncations": 0,
            },
            str(FULL_SAMPLE_DOSE): {
                "candidate_wins": 65,
                "baseline_wins": 63,
                "win_rate": 0.5078125,
                "errors": 0,
                "truncations": 0,
            },
            "paired_outcomes": {
                "both_win": 41,
                "short_only_win": 34,
                "full_only_win": 24,
                "both_lose": 29,
            },
        },
        "selected_sample_dose": SAMPLE_DOSE,
        "selected_optimizer_steps": steps_by_dose[str(SAMPLE_DOSE)],
        "selection_rationale": (
            "the full dose was not more than two percentage points stronger on "
            "the matched f7 panel; observed strength was 7.8125 points lower, so "
            "the smallest dose satisfies the Pareto rule"
        ),
        "failure_rule": (
            "if neither checkpoint clears the predeclared f7 performance floor, "
            "do not infer that more dose is beneficial from offline loss"
        ),
        "lr_exposure_interpretation": (
            "the full run has about 12.41x the integrated LR exposure of the short "
            "run (974.5 versus 78.5 full-LR-equivalent steps), not merely 8x the "
            "row draws; behavior must adjudicate whether that extra exposure helps"
        ),
    }
    replay_scope_adjudication = {
        "schema_version": "a1-replay-scope-adjudication-v1",
        "baseline": "TEMP_CONTROL",
        "baseline_contract": {
            "policy_distillation_component_ids": [
                "n128_current",
                "n256_current",
                "gen3_replay",
            ],
            "value_training_component_ids": [
                "n128_current",
                "n256_current",
                "gen3_replay",
            ],
        },
        "one_axis_arms": {
            "CURRENT_POLICY_SCOPE": "policy scope only; value scope stays exact TEMP",
            "CURRENT_VALUE_SCOPE": "value scope only; policy scope stays exact TEMP",
        },
        "selection_rule": (
            "compare each arm directly with TEMP_CONTROL at the selected dose and "
            "deployed operator; retain TEMP scope for every unsupported axis"
        ),
        "interaction_rule": (
            "do not create a both-current scope by composition unless both one-axis "
            "arms independently survive and a separately named interaction arm is bound"
        ),
    }
    decision = {
        "loser_targets": (
            "TEMP_CONTROL repairs replay teacher-gap versus LEGACY_L03: suppressing "
            "loser search targets was causal"
        ),
        "replay_policy_scope": (
            "CURRENT_POLICY_SCOPE improves over TEMP_CONTROL: obsolete replay policy "
            "distillation was causal"
        ),
        "replay_value_scope": (
            "CURRENT_VALUE_SCOPE improves over TEMP_CONTROL: old-policy continuation "
            "outcomes were causal"
        ),
        "active_policy_dose": (
            "completed L1_POLICY_AUX scored 596-604/1200 (49.67%): additional "
            "policy-active exposure did not demonstrate improvement and is pruned"
        ),
        "target_semantics": (
            "L1_PURE_SEARCH_TARGET improves over its selected scope control: the 10% played-action "
            "hard label was harmful deterministic re-sharpening"
        ),
        "replay_anchor": (
            "L1_REPLAY_ANCHOR improves exact forward replay KL without erasing "
            "n128/n256 teacher-gap closure: population-behavior drift was causal"
        ),
        "target_binding": (
            "L1_GATHER improves topology-sensitive buckets over its selected "
            "TEMP/replay-scope control: "
            "spatial state-action aliasing was causal"
        ),
        "neither": (
            "TEMP_CONTROL and L1_POLICY_AUX fail: inspect objective-specific trunk "
            "gradients and deployed value squash before architecture escalation"
        ),
        "escalation": (
            "advance at most one Pareto arm to 8,388,608 samples from the original "
            "f7 initialization; never continue an arm's optimizer state"
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "launch_authorized": False,
        "launch_condition": (
            "dose adjudication is complete; run only the next predeclared one-axis "
            "arm at 524288 rows / 128 steps from exact f7"
        ),
        "target_semantics_audit": {
            "path": (
                "/home/ubuntu/experimental_nonpromotable/"
                "a1-combined-80-20-20260711/diagnostics/"
                "policy-target-semantics-6e59f5e/"
                "full-corpus-policy-target-semantics.json"
            ),
            "sha256": (
                "sha256:53e9b92402af598bbe2e2a20ce0fb1d4d36dbb059cad8de772b16c555c58151d"
            ),
            "audit_code_commit": "6e59f5e",
            "conclusion": (
                "gen3 replay is an older, materially softer policy teacher; "
                "test exclusion of its search-target CE independently; policy-conditional "
                "replay outcomes require a separate value-scope arm"
            ),
        },
        "gpu_schedule": [
            "reuse LEGACY_L03 without B200 compute",
            "run exact TEMP_CONTROL on all 8 B200s at the selected dose",
            "run CURRENT_POLICY_SCOPE and CURRENT_VALUE_SCOPE sequentially against TEMP_CONTROL",
            "select one replay scope; never combine two unsupported scope changes",
            "run L1_PURE_SEARCH_TARGET after scope selection releases DDP",
            "reuse completed L1_POLICY_AUX negative evidence; do not launch it again",
            (
                "run L1_REPLAY_ANCHOR only if exact-v2 telemetry shows harmful "
                "authenticated replay forward-KL drift; keep policy auxiliary dose zero"
            ),
            (
                "run L1_GATHER only after selecting the corrected learner objective; "
                "keep policy auxiliary dose zero"
            ),
            "evaluate arms in parallel on the H100 fleet; do not consume B200s",
        ],
        "fixed_recipe": fixed,
        "dose_adjudication": dose_adjudication,
        "replay_scope_adjudication": replay_scope_adjudication,
        "historical_feature_audit": {
            "trunk_lr_multiplier": (
                "implemented as --trunk-lr-mult for the exact canonical entity-graph "
                "trunk; default 1.0 preserves historical grouping and non-unit use "
                "fails closed on unsupported/empty trunks"
            ),
            "action_module_lr_multiplier": (
                "implemented, but f7 has no gather/cross action-local parameters; "
                "non-unit use on the baseline correctly fails closed"
            ),
            "action_target_gather": "implemented, warm-start-safe, f7 default OFF",
            "action_cross_attention": "implemented, warm-start-safe, f7 effective layers=0",
            "auxiliary_heads": "implemented but f7 OFF and P1 loss weight zero",
            "root_value_blend": (
                "implemented and loader-preserved; lambda remains 1 because stored "
                "roots are correlated, stale f7 search estimates and changing the "
                "blend would confound the trunk-localization experiment"
            ),
            "graph_history_features": (
                "flag/schema present but event payload proven absent; never claim "
                "history conditioning until a redacted v2 producer exists"
            ),
            "d6_symmetry": (
                "legal-action identity and valid action-id zero relabeling were fixed; "
                "the earlier 3.3x denoising estimate is contaminated and non-binding "
                "until rerun with exact 567-wide action-catalog semantics"
            ),
        },
        "arms": arms,
        "evaluation": evaluation,
        "early_stop": [
            "abort immediately on non-finite loss/gradient/parameters",
            (
                "do not abort on clipped fraction alone: Adam largely normalizes a "
                "common gradient rescale; require objective-specific trunk conflict "
                "or pathological parameter-update evidence"
            ),
            (
                "evaluate at the selected dose; no arm may silently continue from "
                "the short dose to the full dose on offline loss alone"
            ),
        ],
        "prelaunch_gradient_probe": {
            "purpose": (
                "distinguish destructive multi-task trunk interference from an "
                "ordinary learning-rate problem before spending another full dose"
            ),
            "initialization": "reload f7 independently",
            "data": "exact P1 authenticated composite and sampler",
            "execution": (
                "single GPU, 32 optimizer steps, diagnostics every step; diagnostic-only "
                "checkpoint must never enter candidate lineage"
            ),
            "primary_readouts": [
                "value_to_policy_grad_norm_ratio",
                "trunk_gradient_cosine",
                "blocks.0 policy/value norms and cosine",
                "combined pre-clip norm and actual block parameter delta",
            ],
            "interpretation": {
                "negative_cosine_or_block0_value_dominance": (
                    "value-head LR is not a fix; test value-facing trunk gradient "
                    "scaling/projection or head-only value repair"
                ),
                "aligned_balanced_gradients": (
                    "shared objective is not the proximal cause; prioritize total "
                    "dose/LR schedule and policy-active sample budget"
                ),
            },
        },
        "decision_rules": decision,
        "explicitly_deferred": {
            "auxiliary_heads": "objective/regularization change; conditional after diagnosis",
            "public_event_history": (
                "requires versioned redacted Rust/Python parity; current 261GB "
                "constant event payload is omission/storage debt, not signal"
            ),
            "root_value_blend": (
                "objective change deferred until trunk localization; root operator "
                "identity is valid but the estimates are correlated and stale"
            ),
            "hlgauss": "value-objective change; conditional after scalar stability",
            "relational_trunk": (
                "short warm-start is invalid: only 13.66% transfers and 86.34% is "
                "newly initialized; use gather-only or a function-preserving transplant"
            ),
            "model_scale": "only after corrected 35M underfitting is established",
        },
        "value_readout_probe": {
            "mismatch": "learner fits raw value while deployed search consumes tanh(raw)",
            "ranking_reversal_example": (
                "for z=+1, raw predictions .9 and 1.2 have MSE .01 and .04, but "
                "after tanh their squared errors are .0807 and .0276"
            ),
            "calibration": ["raw", "tanh", "clip"],
            "operator_panel": "matched tanh versus clip, same squash for both agents",
            "default_change_authorized": False,
        },
    }
    payload["plan_sha256"] = _digest(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--local-batch-size", type=int, default=512)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--output", default="-")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(
            world_size=args.world_size,
            local_batch_size=args.local_batch_size,
            grad_accum_steps=args.grad_accum_steps,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    encoded = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output == "-":
        print(encoded, end="")
        return
    path = Path(args.output)
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise SystemExit(f"REFUSED: existing plan differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
