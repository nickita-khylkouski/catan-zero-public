#!/usr/bin/env python3
"""Train one independently initialized f7 learner on Stage-C targets.

This is deliberately one execution path, not another broad sweep.  It consumes
the derived Stage-C memmap admission, reloads the exact function-preserving f7
initializer with fresh Adam on 8xB200, and trains policy only on the coherent
reanalysed roots while retaining the complete base corpus for value learning.

The adaptive KL controller is intentionally disabled: its historical meter was
an optimizer-batch statistic and did not enforce the posthoc functional KL.
Checkpoints are instead selected using a frozen whole-game validation surface
after training.  The 64 active rows/rank dose keeps the 8,192-root overlay from
being replayed almost half an epoch on every optimizer step.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
for root in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from catan_zero.rl.entity_feature_adapter import (  # noqa: E402
    CURRENT_RUST_ENTITY_ADAPTER_VERSION,
)
from catan_zero.rl.meaningful_history import (  # noqa: E402
    MEANINGFUL_PUBLIC_HISTORY_LIMIT,
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION,
)
from catan_zero.rl.ordered_history import MASKED_MEAN_V1  # noqa: E402
from tools import a1_b200_active_policy_campaign as stage_a  # noqa: E402
from tools import a1_function_preserving_upgrade as architecture_upgrade  # noqa: E402
from tools import a1_one_dose_train as one_dose  # noqa: E402
from tools import a1_stage_c_learner_overlay as overlay  # noqa: E402


SCHEMA = "a1-b200-stage-c-aligned-learner-campaign-v7"
EXECUTION_SCHEMA = "a1-b200-stage-c-aligned-learner-execution-v2"
FINGERPRINT_SCHEMA = "a1-b200-stage-c-aligned-learner-fingerprint-v5"
BEHAVIORAL_COMPETENCE_GATE_SCHEMA = "a1-maritime-behavioral-competence-gate-v1"
POSTHOC_BEHAVIORAL_COMPETENCE_SCHEMA = "posthoc-behavioral-competence-v1"
PAIRED_BEHAVIORAL_COMPETENCE_SCHEMA = (
    "posthoc-paired-parent-behavioral-competence-v1"
)
POLICY_BEHAVIOR_METRICS_SCHEMA = "policy-target-behavior-metrics-v1"
POLICY_TEACHER_GAP_OBJECTIVE_SCHEMA = "posthoc-policy-teacher-gap-objective-v1"
PAIRED_PARENT_GAP_SCHEMA = "posthoc-paired-parent-teacher-gap-v2"
TRANSITIONAL_PAIRED_PARENT_GAP_SCHEMA = "posthoc-paired-parent-teacher-gap-v1"
SEPARATE_PARENT_GAP_SCHEMA = "posthoc-separate-parent-teacher-gap-v1"
VALUE_QUALITY_SCHEMA = "posthoc-objective-matched-value-quality-v1"
PAIRED_PARENT_VALUE_SCHEMA = "posthoc-paired-parent-value-quality-v1"
LEARNER_VALIDATION_CONTRACT_SCHEMA = (
    "a1-stage-c-learner-validation-execution-contract-v1"
)
WORLD_SIZE = 8
LOCAL_BATCH_SIZE = 512
GLOBAL_BATCH_SIZE = WORLD_SIZE * LOCAL_BATCH_SIZE
POLICY_AUX_ACTIVE_BATCH_SIZE = 64
POLICY_AUX_LOSS_WEIGHT = 0.25
MAX_STEPS = 32
CHECKPOINT_STEPS = (1, 2, 4, 8, 12, 16, 24, 32)
INTERMEDIATE_CHECKPOINT_STEPS = CHECKPOINT_STEPS[:-1]
LR = 6.0e-5
LR_WARMUP_STEPS = 16
MAX_PARENT_KL = 0.03
MAX_TRUNK_RELATIVE_L2 = 0.03
ARMS = frozenset({"PRODUCTION_WEIGHTED", "STRATEGIC_BALANCED"})
TRAINABLE_ADAPTER_MODULES = frozenset(
    {
        "legal_action_value_residual_proj",
        "legal_action_value_static_proj",
        "meaningful_history_residual_gate",
        "public_card_count_residual",
        "static_action_residual_proj",
    }
)
FEATURE_SIGNAL_MODULES = frozenset({*TRAINABLE_ADAPTER_MODULES, "event_encoder"})
TRAIN_DIAGNOSTIC_CADENCE_BATCHES = 16
MINIMUM_FEATURE_SIGNAL_OBSERVATIONS = MAX_STEPS // TRAIN_DIAGNOSTIC_CADENCE_BATCHES
OBJECTIVE_GRADIENT_CADENCE_BATCHES = MAX_STEPS // 2
MINIMUM_OBJECTIVE_GRADIENT_OBSERVATIONS = 2
POSITIVE_OPTIMIZER_SIGNAL_FIELDS = (
    "mean_pre_clip_grad_norm",
    "max_pre_clip_grad_norm",
    "mean_parameter_delta_norm",
    "mean_parameter_update_rms",
)
EFFECTIVE_FEATURE_CONTRACT = {
    "entity_feature_adapter_version": CURRENT_RUST_ENTITY_ADAPTER_VERSION,
    "static_action_residual": True,
    "legal_action_value_residual": True,
    "public_card_count_features": True,
    "meaningful_public_history": True,
    "meaningful_public_history_schema": (MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION),
    "event_history_limit": MEANINGFUL_PUBLIC_HISTORY_LIMIT,
    "meaningful_public_history_pooling": MASKED_MEAN_V1,
}
VALUE_GATE_POLICY = "require_non_regression"
VALUE_GATE_POLICIES = frozenset(
    {VALUE_GATE_POLICY, "diagnostic_record_only_allow_regression"}
)
MAX_VALUE_MSE_REGRESSION = 0.0
POLICY_AUX_COVERAGE_TAIL_PROBABILITY = 1.0e-9


class CampaignError(RuntimeError):
    """The Stage-C aligned learner campaign is invalid."""


def _uniform_unique_coverage_contract(
    *,
    population: int,
    draws: int,
    tail_probability: float = POLICY_AUX_COVERAGE_TAIL_PROBABILITY,
) -> dict[str, float | int | str]:
    """Return a rigorous lower bound for unique roots under uniform draws.

    The number of distinct roots is 1-Lipschitz in every independent draw.
    McDiarmid's inequality therefore gives a distribution-free lower-tail
    bound around the exact occupancy expectation without relying on a normal
    approximation.
    """

    roots = int(population)
    dose = int(draws)
    alpha = float(tail_probability)
    if roots <= 0 or dose < 0:
        raise CampaignError("policy-AUX coverage population/draws are invalid")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise CampaignError("policy-AUX coverage tail probability is invalid")
    if dose == 0:
        expectation = 0.0
        deviation = 0.0
    else:
        unseen_probability = math.exp(
            float(dose) * math.log1p(-1.0 / float(roots))
        ) if roots > 1 else 0.0
        expectation = float(roots) * (1.0 - unseen_probability)
        deviation = math.sqrt(
            0.5 * float(dose) * math.log(1.0 / alpha)
        )
    lower = max(0, math.floor(expectation - deviation))
    return {
        "schema_version": "uniform-policy-aux-unique-coverage-v1",
        "population": roots,
        "draws": dose,
        "expected_unique_roots": expectation,
        "lower_bound_unique_roots": int(lower),
        "lower_tail_probability_bound": alpha,
        "bound": "mcdiarmid_bounded_differences",
    }


def _verify_strategic_balanced_sampling_surface(
    sampling: Mapping[str, Any], *, selected_training_roots: int
) -> None:
    """Authenticate that STRATEGIC_BALANCED really is uniform over roots."""

    summary = sampling.get("final_training_weights")
    roots = int(selected_training_roots)
    if (
        roots <= 0
        or not isinstance(summary, Mapping)
        or float(summary.get("min", math.nan)) != 1.0
        or float(summary.get("max", math.nan)) != 1.0
        or float(summary.get("mean", math.nan)) != 1.0
        or not math.isclose(
            float(summary.get("effective_sample_size", math.nan)),
            float(roots),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise CampaignError(
            "STRATEGIC_BALANCED sampling is not uniform over selected training roots"
        )


def _verify_realized_policy_aux_coverage(
    *,
    arm: str,
    selected_training_roots: int,
    auxiliary_draws: int,
    unique_source_rows: int,
) -> dict[str, Any]:
    """Refuse a learner whose realized auxiliary sampler collapsed."""

    roots = int(selected_training_roots)
    draws = int(auxiliary_draws)
    unique = int(unique_source_rows)
    if roots <= 0 or draws <= 0 or unique <= 0 or unique > min(roots, draws):
        raise CampaignError("completed learner unique-root exposure drifted")
    if arm != "STRATEGIC_BALANCED":
        return {
            "schema_version": "policy-aux-realized-coverage-v1",
            "arm": str(arm),
            "exact_uniform_lower_bound_available": False,
            "gate_passed": True,
        }
    contract = _uniform_unique_coverage_contract(
        population=roots,
        draws=draws,
    )
    if unique < int(contract["lower_bound_unique_roots"]):
        raise CampaignError(
            "completed learner policy-AUX sampler collapsed: "
            f"unique={unique} lower_bound={contract['lower_bound_unique_roots']} "
            f"draws={draws} population={roots}"
        )
    return {
        **contract,
        "arm": str(arm),
        "exact_uniform_lower_bound_available": True,
        "realized_unique_roots": unique,
        "gate_passed": True,
    }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _value_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _behavioral_competence_gate_contract() -> dict[str, Any]:
    """Return the immutable Stage-C maritime competence admission contract."""

    return {
        "schema_version": BEHAVIORAL_COMPETENCE_GATE_SCHEMA,
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


def _wilson_score_interval(
    successes: int, total: int, *, z: float
) -> tuple[float, float]:
    """Return the Wilson score interval used by the one-sided competence gate."""

    if isinstance(successes, bool) or isinstance(total, bool):
        raise CampaignError("behavioral competence counts must be integers")
    if int(successes) != successes or int(total) != total:
        raise CampaignError("behavioral competence counts must be integers")
    hits = int(successes)
    rows = int(total)
    critical = float(z)
    if (
        rows <= 0
        or hits < 0
        or hits > rows
        or not math.isfinite(critical)
        or critical <= 0.0
    ):
        raise CampaignError("behavioral competence counts/critical value are invalid")
    rate = hits / rows
    z2 = critical * critical
    denominator = 1.0 + z2 / rows
    center = (rate + z2 / (2.0 * rows)) / denominator
    radius = (
        critical
        * math.sqrt((rate * (1.0 - rate) + z2 / (4.0 * rows)) / rows)
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _require_behavior_rate(value: object, *, where: str) -> float:
    try:
        rate = float(value)
    except (TypeError, ValueError) as error:
        raise CampaignError(f"{where} must be a finite probability") from error
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise CampaignError(f"{where} must be a finite probability")
    return rate


def _integer_successes(rate: float, rows: int, *, where: str) -> int:
    scaled = rate * rows
    successes = round(scaled)
    if not math.isclose(scaled, successes, rel_tol=0.0, abs_tol=1.0e-9):
        raise CampaignError(f"{where} is not backed by an integer count")
    return int(successes)


def _require_action_catalog_abi(value: object, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CampaignError(f"{where} action-catalog ABI is missing")
    required = {
        "version",
        "size",
        "ordered_descriptors_sha256",
        "action_types_by_id_sha256",
        "identity_sha256",
    }
    if set(value) != required:
        raise CampaignError(f"{where} action-catalog ABI is malformed")
    digest_keys = (
        "ordered_descriptors_sha256",
        "action_types_by_id_sha256",
        "identity_sha256",
    )
    if (
        isinstance(value["size"], bool)
        or not isinstance(value["size"], int)
        or value["size"] <= 0
        or not isinstance(value["version"], str)
        or not value["version"]
        or any(
            not isinstance(value[key], str)
            or len(value[key]) != 71
            or not value[key].startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in value[key][7:])
            for key in digest_keys
        )
    ):
        raise CampaignError(f"{where} action-catalog ABI is malformed")
    unsigned = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value["identity_sha256"] != _value_sha256(unsigned):
        raise CampaignError(f"{where} action-catalog ABI identity is inconsistent")
    return copy.deepcopy(value)


def _require_behavior_metrics(
    value: object,
    *,
    rows: int,
    where: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CampaignError(f"{where} behavioral metrics are missing")
    top1 = _require_behavior_rate(
        value.get("teacher_top1_accuracy"),
        where=f"{where} teacher_top1_accuracy",
    )
    confusion = _require_behavior_rate(
        value.get("end_turn_confusion_rate"),
        where=f"{where} end_turn_confusion_rate",
    )
    result: dict[str, Any] = {
        "teacher_top1_accuracy": top1,
        "end_turn_confusion_rate": confusion,
        "teacher_top1_successes": _integer_successes(
            top1, rows, where=f"{where} teacher_top1_accuracy"
        ),
        "end_turn_confusions": _integer_successes(
            confusion, rows, where=f"{where} end_turn_confusion_rate"
        ),
    }
    required_metrics = (
        "end_turn_confusion_teacher_probability_regret_per_row",
        "end_turn_confusion_teacher_probability_regret_conditional_mean",
    )
    for key in (
        "teacher_top3_accuracy",
        "teacher_top3_mass",
        *required_metrics,
    ):
        if key not in value:
            if key in required_metrics:
                raise CampaignError(f"{where} {key} is missing")
            continue
        try:
            metric = float(value[key])
        except (TypeError, ValueError) as error:
            raise CampaignError(f"{where} {key} is malformed") from error
        if not math.isfinite(metric) or metric < 0.0:
            raise CampaignError(f"{where} {key} is malformed")
        if key in {"teacher_top3_accuracy", "teacher_top3_mass"} and metric > 1.0:
            raise CampaignError(f"{where} {key} is malformed")
        result[key] = metric
    return result


def _require_weighted_behavior_metrics(
    value: object,
    *,
    where: str,
) -> dict[str, float]:
    """Validate objective-weighted metrics without inventing binomial counts."""

    if not isinstance(value, dict):
        raise CampaignError(f"{where} weighted behavioral metrics are missing")
    result = {
        "teacher_top1_accuracy": _require_behavior_rate(
            value.get("teacher_top1_accuracy"),
            where=f"{where} teacher_top1_accuracy",
        ),
        "end_turn_confusion_rate": _require_behavior_rate(
            value.get("end_turn_confusion_rate"),
            where=f"{where} end_turn_confusion_rate",
        ),
    }
    for key in (
        "end_turn_confusion_teacher_probability_regret_per_row",
        "end_turn_confusion_teacher_probability_regret_conditional_mean",
    ):
        try:
            metric = float(value[key])
        except (KeyError, TypeError, ValueError) as error:
            raise CampaignError(f"{where} {key} is missing or malformed") from error
        if not math.isfinite(metric) or metric < 0.0:
            raise CampaignError(f"{where} {key} is malformed")
        result[key] = metric
    for key in ("teacher_top3_accuracy", "teacher_top3_mass"):
        if key not in value:
            continue
        result[key] = _require_behavior_rate(value[key], where=f"{where} {key}")
    return result


def _require_paired_weighted_behavior_metrics(
    value: object,
    *,
    candidate: Mapping[str, float],
    parent: Mapping[str, float],
    row_probability: float,
    where: str,
) -> dict[str, float]:
    if not isinstance(value, dict):
        raise CampaignError(f"{where} paired weighted metrics are missing")
    try:
        paired_probability = float(value["row_probability"])
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignError(f"{where} paired row probability is missing") from error
    if not math.isfinite(paired_probability) or not math.isclose(
        paired_probability, row_probability, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise CampaignError(f"{where} paired row probability differs")
    paired_parent = _require_weighted_behavior_metrics(
        value.get("parent"), where=f"{where} parent"
    )
    paired_candidate = _require_weighted_behavior_metrics(
        value.get("candidate"), where=f"{where} candidate"
    )
    if paired_parent != dict(parent) or paired_candidate != dict(candidate):
        raise CampaignError(f"{where} paired projections differ from direct metrics")
    deltas = value.get("candidate_minus_parent")
    if not isinstance(deltas, dict):
        raise CampaignError(f"{where} paired deltas are missing")
    projected: dict[str, float] = {}
    for key in candidate:
        if key not in parent:
            raise CampaignError(f"{where} parent metric keys differ")
        expected = float(candidate[key]) - float(parent[key])
        try:
            actual = float(deltas[key])
        except (KeyError, TypeError, ValueError) as error:
            raise CampaignError(f"{where} paired delta {key} is missing") from error
        if not math.isfinite(actual) or not math.isclose(
            actual, expected, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise CampaignError(f"{where} paired delta {key} is inconsistent")
        projected[key] = actual
    return projected


def _require_paired_behavior_metrics(
    value: object,
    *,
    candidate: Mapping[str, Any],
    parent: Mapping[str, Any],
    rows: int,
    where: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CampaignError(f"{where} paired behavioral metrics are missing")
    if int(value.get("rows", -1)) != rows:
        raise CampaignError(f"{where} paired row count differs")
    paired_parent = _require_behavior_metrics(
        value.get("parent"), rows=rows, where=f"{where} parent"
    )
    paired_candidate = _require_behavior_metrics(
        value.get("candidate"), rows=rows, where=f"{where} candidate"
    )
    if paired_parent != dict(parent) or paired_candidate != dict(candidate):
        raise CampaignError(f"{where} paired projections differ from direct metrics")
    deltas = value.get("candidate_minus_parent")
    if not isinstance(deltas, dict):
        raise CampaignError(f"{where} paired deltas are missing")
    metric_keys = set(candidate) - {
        "teacher_top1_successes",
        "end_turn_confusions",
    }
    if metric_keys != set(parent) - {
        "teacher_top1_successes",
        "end_turn_confusions",
    }:
        raise CampaignError(f"{where} parent metric keys differ")
    required_deltas = {
        key: float(candidate[key]) - float(parent[key])
        for key in metric_keys
    }
    projected: dict[str, float] = {}
    for key, expected in required_deltas.items():
        try:
            actual = float(deltas[key])
        except (KeyError, TypeError, ValueError) as error:
            raise CampaignError(f"{where} paired delta {key} is missing") from error
        if not math.isfinite(actual) or not math.isclose(
            actual, expected, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise CampaignError(f"{where} paired delta {key} is inconsistent")
        projected[key] = actual
    return projected


def _behavior_stream(
    projection: Mapping[str, Any],
    *,
    stream: str,
    where: str,
    require_schema: bool = True,
) -> dict[str, Any]:
    streams = projection.get("streams")
    if not isinstance(streams, dict) or not isinstance(streams.get(stream), dict):
        raise CampaignError(f"{where} required {stream} stream is missing")
    result = streams[stream]
    if require_schema and result.get("schema_version") != POLICY_BEHAVIOR_METRICS_SCHEMA:
        raise CampaignError(f"{where} behavioral metric schema drifted")
    return result


def _evaluate_maritime_behavioral_competence_gate(
    functional: Mapping[str, Any],
    *,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate posthoc evidence and evaluate the maritime prefilter."""

    expected_contract = _behavioral_competence_gate_contract()
    supplied_contract = expected_contract if contract is None else dict(contract)
    if supplied_contract != expected_contract:
        raise CampaignError("maritime behavioral competence contract drifted")
    candidate_projection = functional.get("behavioral_competence")
    parent_projection = functional.get("parent_behavioral_competence")
    paired_projection = functional.get("paired_parent_behavioral_competence")
    if (
        not isinstance(candidate_projection, dict)
        or candidate_projection.get("schema_version")
        != POSTHOC_BEHAVIORAL_COMPETENCE_SCHEMA
        or candidate_projection.get("selection_authority") is not False
        or not isinstance(parent_projection, dict)
        or parent_projection.get("schema_version")
        != POSTHOC_BEHAVIORAL_COMPETENCE_SCHEMA
        or parent_projection.get("selection_authority") is not False
        or not isinstance(paired_projection, dict)
        or paired_projection.get("schema_version")
        != PAIRED_BEHAVIORAL_COMPETENCE_SCHEMA
        or paired_projection.get("selection_authority") is not False
    ):
        raise CampaignError("posthoc maritime behavioral competence surface is missing")

    axis_name = str(expected_contract["required_axis"])
    action_type = str(expected_contract["required_action_type"])
    objective_axis = str(
        expected_contract["objective_weighted_diagnostic"]["axis"]
    )
    evidence_streams: dict[str, Any] = {}
    shared_abi: dict[str, Any] | None = None
    for stream_name in expected_contract["required_streams"]:
        candidate_stream = _behavior_stream(
            candidate_projection, stream=stream_name, where="candidate"
        )
        parent_stream = _behavior_stream(
            parent_projection, stream=stream_name, where="parent"
        )
        paired_stream = _behavior_stream(
            paired_projection,
            stream=stream_name,
            where="paired parent",
            require_schema=False,
        )
        candidate_abi = _require_action_catalog_abi(
            candidate_stream.get("action_catalog_abi"),
            where=f"candidate {stream_name}",
        )
        parent_abi = _require_action_catalog_abi(
            parent_stream.get("action_catalog_abi"),
            where=f"parent {stream_name}",
        )
        paired_abi = _require_action_catalog_abi(
            paired_stream.get("action_catalog_abi"),
            where=f"paired parent {stream_name}",
        )
        if candidate_abi != parent_abi or candidate_abi != paired_abi:
            raise CampaignError(
                f"{stream_name} maritime behavioral competence ABI differs"
            )
        if shared_abi is None:
            shared_abi = candidate_abi
        elif candidate_abi != shared_abi:
            raise CampaignError(
                "maritime behavioral competence ABI differs across streams"
            )

        try:
            candidate_row = candidate_stream[axis_name][action_type]
            parent_row = parent_stream[axis_name][action_type]
            paired_row = paired_stream[axis_name][action_type]
        except (KeyError, TypeError) as error:
            raise CampaignError(
                f"{stream_name} uniform maritime behavioral stratum is missing"
            ) from error
        if not isinstance(candidate_row, dict) or not isinstance(parent_row, dict):
            raise CampaignError(
                f"{stream_name} uniform maritime behavioral stratum is malformed"
            )
        candidate_rows = candidate_row.get("rows")
        parent_rows = parent_row.get("rows")
        if (
            isinstance(candidate_rows, bool)
            or not isinstance(candidate_rows, int)
            or isinstance(parent_rows, bool)
            or not isinstance(parent_rows, int)
            or candidate_rows <= 0
            or parent_rows != candidate_rows
        ):
            raise CampaignError(
                f"{stream_name} uniform maritime row count is malformed"
            )
        rows = candidate_rows
        candidate_metrics = _require_behavior_metrics(
            candidate_row, rows=rows, where=f"candidate {stream_name} maritime"
        )
        parent_metrics = _require_behavior_metrics(
            parent_row, rows=rows, where=f"parent {stream_name} maritime"
        )
        paired_deltas = _require_paired_behavior_metrics(
            paired_row,
            candidate=candidate_metrics,
            parent=parent_metrics,
            rows=rows,
            where=f"paired {stream_name} maritime",
        )

        try:
            candidate_objective = candidate_stream[objective_axis][action_type]
            parent_objective = parent_stream[objective_axis][action_type]
            paired_objective = paired_stream[objective_axis][action_type]
        except (KeyError, TypeError) as error:
            raise CampaignError(
                f"{stream_name} objective-weighted maritime diagnostic is missing"
            ) from error
        if not all(
            isinstance(row, dict)
            for row in (candidate_objective, parent_objective, paired_objective)
        ):
            raise CampaignError(
                f"{stream_name} objective-weighted maritime diagnostic is malformed"
            )
        try:
            candidate_probability = float(candidate_objective["row_probability"])
            parent_probability = float(parent_objective["row_probability"])
        except (KeyError, TypeError, ValueError) as error:
            raise CampaignError(
                f"{stream_name} objective-weighted row probability is missing"
            ) from error
        if (
            not math.isfinite(candidate_probability)
            or candidate_probability <= 0.0
            or not math.isclose(
                candidate_probability,
                parent_probability,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise CampaignError(
                f"{stream_name} objective-weighted row probability differs"
            )
        objective_candidate = _require_weighted_behavior_metrics(
            candidate_objective,
            where=f"candidate {stream_name} objective-weighted maritime",
        )
        objective_parent = _require_weighted_behavior_metrics(
            parent_objective,
            where=f"parent {stream_name} objective-weighted maritime",
        )
        objective_deltas = _require_paired_weighted_behavior_metrics(
            paired_objective,
            candidate=objective_candidate,
            parent=objective_parent,
            row_probability=candidate_probability,
            where=f"paired {stream_name} objective-weighted maritime",
        )
        evidence_streams[str(stream_name)] = {
            "axis": axis_name,
            "action_type": action_type,
            "action_catalog_abi": candidate_abi,
            "rows": rows,
            "candidate": candidate_metrics,
            "parent": parent_metrics,
            "candidate_minus_parent": paired_deltas,
            "objective_weighted_diagnostic": {
                "axis": objective_axis,
                "selection_authority": False,
                "row_probability": candidate_probability,
                "candidate": objective_candidate,
                "parent": objective_parent,
                "candidate_minus_parent": objective_deltas,
            },
        }
    evidence = {"streams": evidence_streams}
    return _maritime_behavioral_competence_gate_from_evidence(
        evidence, contract=expected_contract
    )


def _maritime_behavioral_competence_gate_from_evidence(
    evidence: Mapping[str, Any],
    *,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute the canonical gate result from stored, authenticated evidence."""

    expected_contract = _behavioral_competence_gate_contract()
    supplied_contract = expected_contract if contract is None else dict(contract)
    if supplied_contract != expected_contract:
        raise CampaignError("maritime behavioral competence contract drifted")
    streams = evidence.get("streams")
    if not isinstance(streams, dict) or set(streams) != set(
        expected_contract["required_streams"]
    ):
        raise CampaignError("maritime behavioral competence evidence streams drifted")
    canonical_streams: dict[str, Any] = {}
    stream_checks: dict[str, Any] = {}
    shared_abi: dict[str, Any] | None = None
    for stream_name in expected_contract["required_streams"]:
        stream = streams.get(stream_name)
        if not isinstance(stream, dict):
            raise CampaignError(
                f"{stream_name} maritime behavioral evidence is malformed"
            )
        try:
            rows = int(stream["rows"])
            candidate = stream["candidate"]
            parent = stream["parent"]
            candidate_top1 = float(candidate["teacher_top1_accuracy"])
            candidate_confusion = float(candidate["end_turn_confusion_rate"])
            parent_top1 = float(parent["teacher_top1_accuracy"])
            parent_confusion = float(parent["end_turn_confusion_rate"])
            top1_successes = int(candidate["teacher_top1_successes"])
            confusions = int(candidate["end_turn_confusions"])
        except (KeyError, TypeError, ValueError) as error:
            raise CampaignError(
                f"{stream_name} maritime behavioral evidence is malformed"
            ) from error
        if (
            stream.get("axis") != expected_contract["required_axis"]
            or stream.get("action_type") != expected_contract["required_action_type"]
        ):
            raise CampaignError(
                f"{stream_name} maritime behavioral evidence scope drifted"
            )
        abi = _require_action_catalog_abi(
            stream.get("action_catalog_abi"),
            where=f"stored {stream_name} evidence",
        )
        if shared_abi is None:
            shared_abi = abi
        elif abi != shared_abi:
            raise CampaignError(
                "stored maritime behavioral ABI differs across streams"
            )
        candidate_metrics = _require_behavior_metrics(
            candidate, rows=rows, where=f"stored candidate {stream_name}"
        )
        parent_metrics = _require_behavior_metrics(
            parent, rows=rows, where=f"stored parent {stream_name}"
        )
        if (
            candidate_metrics != candidate
            or parent_metrics != parent
            or top1_successes != candidate_metrics["teacher_top1_successes"]
            or confusions != candidate_metrics["end_turn_confusions"]
        ):
            raise CampaignError(
                f"{stream_name} maritime behavioral stored counts differ"
            )
        deltas = _require_paired_behavior_metrics(
            {
                "rows": rows,
                "parent": parent,
                "candidate": candidate,
                "candidate_minus_parent": stream.get("candidate_minus_parent"),
            },
            candidate=candidate_metrics,
            parent=parent_metrics,
            rows=rows,
            where=f"stored paired {stream_name}",
        )
        objective = stream.get("objective_weighted_diagnostic")
        if (
            not isinstance(objective, dict)
            or objective.get("axis")
            != expected_contract["objective_weighted_diagnostic"]["axis"]
            or objective.get("selection_authority") is not False
        ):
            raise CampaignError(
                f"{stream_name} objective-weighted diagnostic evidence is missing"
            )
        try:
            row_probability = float(objective["row_probability"])
        except (KeyError, TypeError, ValueError) as error:
            raise CampaignError(
                f"{stream_name} objective-weighted probability is malformed"
            ) from error
        if not math.isfinite(row_probability) or row_probability <= 0.0:
            raise CampaignError(
                f"{stream_name} objective-weighted probability is malformed"
            )
        objective_candidate = _require_weighted_behavior_metrics(
            objective.get("candidate"),
            where=f"stored candidate {stream_name} objective-weighted",
        )
        objective_parent = _require_weighted_behavior_metrics(
            objective.get("parent"),
            where=f"stored parent {stream_name} objective-weighted",
        )
        objective_deltas = _require_paired_weighted_behavior_metrics(
            {
                "row_probability": row_probability,
                "parent": objective_parent,
                "candidate": objective_candidate,
                "candidate_minus_parent": objective.get("candidate_minus_parent"),
            },
            candidate=objective_candidate,
            parent=objective_parent,
            row_probability=row_probability,
            where=f"stored paired {stream_name} objective-weighted",
        )
        z = float(expected_contract["confidence_interval"]["z"])
        top1_lower, _ = _wilson_score_interval(top1_successes, rows, z=z)
        _, confusion_upper = _wilson_score_interval(confusions, rows, z=z)
        checks = {
            "minimum_rows": rows >= int(expected_contract["minimum_rows"]),
            "teacher_top1_lower_bound": (
                top1_lower
                >= float(expected_contract["minimum_teacher_top1_lower_bound"])
            ),
            "end_turn_confusion_upper_bound": (
                confusion_upper
                <= float(
                    expected_contract["maximum_end_turn_confusion_upper_bound"]
                )
            ),
            "teacher_top1_parent_regression": (
                candidate_top1 - parent_top1
                >= -float(
                    expected_contract["maximum_teacher_top1_parent_regression"]
                )
            ),
            "end_turn_confusion_parent_regression": (
                candidate_confusion - parent_confusion
                <= float(
                    expected_contract[
                        "maximum_end_turn_confusion_parent_regression"
                    ]
                )
            ),
            "objective_weighted_diagnostic_present": True,
        }
        canonical_streams[str(stream_name)] = {
            "axis": expected_contract["required_axis"],
            "action_type": expected_contract["required_action_type"],
            "action_catalog_abi": abi,
            "rows": rows,
            "candidate": candidate_metrics,
            "parent": parent_metrics,
            "candidate_minus_parent": deltas,
            "objective_weighted_diagnostic": {
                "axis": objective["axis"],
                "selection_authority": False,
                "row_probability": row_probability,
                "candidate": objective_candidate,
                "parent": objective_parent,
                "candidate_minus_parent": objective_deltas,
            },
            "confidence_bounds": {
                "teacher_top1_lower": top1_lower,
                "end_turn_confusion_upper": confusion_upper,
            },
        }
        stream_checks[str(stream_name)] = checks
    passed = all(all(check.values()) for check in stream_checks.values())
    canonical_evidence = {"streams": canonical_streams}
    checks = {
        "streams": stream_checks,
        "all_required_streams": passed,
    }
    return {
        "schema_version": BEHAVIORAL_COMPETENCE_GATE_SCHEMA,
        "selection_authority": True,
        "contract": expected_contract,
        "evidence": canonical_evidence,
        "checks": checks,
        "passed": passed,
        "selection_admitted": passed,
    }


def _require_maritime_behavioral_competence_gate_result(
    value: object,
    *,
    expected_contract: Mapping[str, Any] | None = None,
    where: str,
) -> dict[str, Any]:
    """Validate a stored gate by recomputing it from evidence."""

    if not isinstance(value, dict):
        raise CampaignError(f"{where} maritime behavioral competence gate is missing")
    contract = (
        _behavioral_competence_gate_contract()
        if expected_contract is None
        else dict(expected_contract)
    )
    if (
        value.get("schema_version") != BEHAVIORAL_COMPETENCE_GATE_SCHEMA
        or value.get("selection_authority") is not True
        or value.get("contract") != contract
        or not isinstance(value.get("evidence"), dict)
    ):
        raise CampaignError(f"{where} maritime behavioral competence gate is malformed")
    evidence = copy.deepcopy(value["evidence"])
    for stream in evidence.get("streams", {}).values():
        if isinstance(stream, dict):
            stream.pop("confidence_bounds", None)
    recomputed = _maritime_behavioral_competence_gate_from_evidence(
        evidence, contract=contract
    )
    if recomputed != value:
        raise CampaignError(
            f"{where} maritime behavioral competence gate differs from evidence"
        )
    return recomputed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path, *, where: str) -> tuple[Path, dict[str, Any]]:
    lexical = path.expanduser()
    if lexical.is_symlink() or not lexical.is_file():
        raise CampaignError(f"{where} must be a regular file: {lexical}")
    resolved = lexical.resolve(strict=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read {where}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"{where} must contain one JSON object")
    return resolved, value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    destination = path.expanduser().resolve(strict=False)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise CampaignError(f"immutable output is not a file: {destination}")
        if destination.read_text(encoding="utf-8") != rendered:
            raise CampaignError(
                f"immutable output already exists with drift: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")


def _load_plan(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved, plan = _load_json(path, where="Stage-C learner campaign")
    unsigned = dict(plan)
    stated = unsigned.pop("campaign_sha256", None)
    if plan.get("schema_version") != SCHEMA or stated != _value_sha256(unsigned):
        raise CampaignError("Stage-C learner campaign schema/digest drifted")
    return resolved, plan


def _integer(value: object, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _verify_feature_optimizer_observability(
    observability: object,
    *,
    minimum_observations: int,
    where: str,
) -> dict[str, Any]:
    if (
        not isinstance(observability, dict)
        or observability.get("schema_version")
        != "module-optimizer-observability-v1"
        or _integer(observability.get("observed_steps"), default=0)
        < minimum_observations
        or _integer(observability.get("cadence_batches"), default=-1)
        != TRAIN_DIAGNOSTIC_CADENCE_BATCHES
        or observability.get("norm_scope") != "global_replicated"
    ):
        raise CampaignError(
            f"{where} lacks the authenticated feature optimizer observation cadence"
        )
    modules = observability.get("modules")
    if not isinstance(modules, dict):
        raise CampaignError(f"{where} feature optimizer modules are missing")
    failures: dict[str, object] = {}
    selected: dict[str, Any] = {}
    for module_name in sorted(FEATURE_SIGNAL_MODULES):
        row = modules.get(module_name)
        if not isinstance(row, dict):
            failures[module_name] = "missing"
            continue
        failed_fields = []
        for field in POSITIVE_OPTIMIZER_SIGNAL_FIELDS:
            try:
                value = float(row.get(field, math.nan))
            except (TypeError, ValueError):
                value = math.nan
            if not math.isfinite(value) or value <= 0.0:
                failed_fields.append(field)
        parameter_count = _integer(row.get("parameter_count"), default=0)
        if parameter_count <= 0:
            failed_fields.append("parameter_count")
        if failed_fields:
            failures[module_name] = failed_fields
        else:
            selected[module_name] = {
                field: row[field] for field in POSITIVE_OPTIMIZER_SIGNAL_FIELDS
            } | {"parameter_count": parameter_count}
    if failures:
        raise CampaignError(
            f"{where} did not demonstrate positive commissioned feature "
            f"gradients and updates: {failures}"
        )
    return {
        "authenticated": True,
        "schema_version": observability["schema_version"],
        "observed_steps": _integer(
            observability.get("observed_steps"), default=0
        ),
        "cadence_batches": TRAIN_DIAGNOSTIC_CADENCE_BATCHES,
        "norm_scope": "global_replicated",
        "modules": selected,
    }


def _verify_completed_feature_learning_signal(
    report: Mapping[str, Any],
) -> None:
    """Require direct evidence that both commissioned feature paths learned."""

    architecture_drift = {
        field: {"expected": expected, "actual": report.get(field)}
        for field, expected in EFFECTIVE_FEATURE_CONTRACT.items()
        if report.get(field) != expected
    }
    if architecture_drift:
        raise CampaignError(
            "completed learner effective feature contract drifted: "
            f"{architecture_drift}"
        )

    if (
        _integer(report.get("train_diagnostics_every_batches"), default=-1)
        != TRAIN_DIAGNOSTIC_CADENCE_BATCHES
    ):
        raise CampaignError(
            "completed learner lacks the authenticated feature optimizer "
            "observation cadence"
        )
    _verify_feature_optimizer_observability(
        report.get("module_optimizer_observability"),
        minimum_observations=MINIMUM_FEATURE_SIGNAL_OBSERVATIONS,
        where="completed learner",
    )


def _verify_completed_objective_gradient_signal(
    report: Mapping[str, Any],
) -> None:
    """Require policy-base/AUX/value geometry before Stage-C evidence is usable."""

    payload = report.get("objective_gradient_interference")
    observations = payload.get("observations") if isinstance(payload, dict) else None
    if (
        _integer(
            report.get("objective_gradient_interference_every_batches"),
            default=-1,
        )
        != OBJECTIVE_GRADIENT_CADENCE_BATCHES
        or not isinstance(payload, dict)
        or _integer(payload.get("cadence_batches"), default=-1)
        != OBJECTIVE_GRADIENT_CADENCE_BATCHES
        or not isinstance(observations, list)
        or len(observations) < MINIMUM_OBJECTIVE_GRADIENT_OBSERVATIONS
    ):
        raise CampaignError(
            "completed learner lacks the authenticated policy/value gradient "
            "observation cadence"
        )
    steps: list[int] = []
    failures: dict[int, object] = {}
    positive_fields = (
        "policy_trunk_grad_norm",
        "policy_base_trunk_grad_norm",
        "policy_aux_trunk_grad_norm",
        "value_trunk_grad_norm",
    )
    bounded_fields = (
        "trunk_gradient_cosine",
        "policy_base_aux_gradient_cosine",
    )
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict) or observation.get("available") is not True:
            failures[index] = "unavailable"
            continue
        step = _integer(observation.get("optimizer_step"), default=-1)
        steps.append(step)
        bad_fields = []
        if observation.get("scope") != "global_ddp_microbatch":
            bad_fields.append("scope")
        if observation.get("aggregation") != (
            "manual_all_reduce_then_world_average_of_ddp_scaled_gradients"
        ):
            bad_fields.append("aggregation")
        if _integer(observation.get("world_size"), default=-1) <= 1:
            bad_fields.append("world_size")
        for field in positive_fields:
            try:
                value = float(observation.get(field, math.nan))
            except (TypeError, ValueError):
                value = math.nan
            if not math.isfinite(value) or value <= 0.0:
                bad_fields.append(field)
        for field in bounded_fields:
            try:
                value = float(observation.get(field, math.nan))
            except (TypeError, ValueError):
                value = math.nan
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                bad_fields.append(field)
        if bad_fields:
            failures[index] = bad_fields
    if (
        failures
        or steps != sorted(set(steps))
        or not steps
        or steps[-1] < OBJECTIVE_GRADIENT_CADENCE_BATCHES
    ):
        raise CampaignError(
            "completed learner did not demonstrate measurable policy-base/AUX/"
            f"value trunk geometry: failures={failures}, steps={steps}"
        )


def _checkpoint_feature_learning_signal(
    report: Mapping[str, Any], *, step: int
) -> dict[str, Any]:
    trajectory = report.get("checkpoint_dose_trajectory")
    if (
        not isinstance(trajectory, dict)
        or trajectory.get("schema_version")
        != "train-bc-checkpoint-dose-trajectory-v1"
        or trajectory.get("checkpoint_steps") != list(CHECKPOINT_STEPS)
        or not isinstance(trajectory.get("checkpoints"), list)
    ):
        raise CampaignError(
            "completed learner lacks an authenticated checkpoint dose trajectory"
        )
    matches = [
        row
        for row in trajectory["checkpoints"]
        if isinstance(row, dict)
        and _integer(row.get("optimizer_step"), default=-1) == step
    ]
    if len(matches) != 1:
        raise CampaignError(f"checkpoint {step} dose telemetry is missing or duplicated")
    dose = matches[0]
    if dose.get("schema_version") != "train-bc-checkpoint-dose-telemetry-v1":
        raise CampaignError(f"checkpoint {step} dose telemetry schema drifted")
    observability = dose.get("module_optimizer_observability")
    if observability is None:
        return {
            "authenticated": False,
            "reason": "awaiting_feature_optimizer_observation_cadence",
            "optimizer_step": step,
        }
    evidence = _verify_feature_optimizer_observability(
        observability,
        minimum_observations=1,
        where=f"checkpoint {step}",
    )
    feature_paths = dose.get("feature_path_gradients")
    if (
        not isinstance(feature_paths, dict)
        or any(
            not isinstance(feature_paths.get(name), dict)
            or feature_paths[name].get("enabled") is not True
            or feature_paths[name].get("status")
            not in {"observed", "observed_nonzero"}
            for name in ("public_card", "meaningful_history")
        )
    ):
        raise CampaignError(
            f"checkpoint {step} feature-path gradient projection is malformed"
        )
    return {
        **evidence,
        "optimizer_step": step,
        "feature_paths": {
            name: {
                "enabled": True,
                "status": feature_paths[name]["status"],
            }
            for name in ("public_card", "meaningful_history")
        },
    }


def _authenticate_checkpoint_snapshot(
    report: Mapping[str, Any],
    *,
    step: int,
    checkpoint: Path,
    terminal_checkpoint: Path,
) -> dict[str, Any]:
    """Bind checkpoint-local telemetry to the exact saved checkpoint bytes."""

    resolved = checkpoint.resolve(strict=True)
    digest = _file_sha256(resolved)
    if step == MAX_STEPS:
        try:
            reported = Path(str(report["checkpoint"])).resolve(strict=True)
        except (KeyError, OSError) as error:
            raise CampaignError(
                "completed learner report has no terminal checkpoint binding"
            ) from error
        if reported != resolved or resolved != terminal_checkpoint.resolve(strict=True):
            raise CampaignError("terminal checkpoint differs from the completed report")
        return {
            "schema_version": "stage-c-checkpoint-report-binding-v1",
            "optimizer_step": step,
            "checkpoint": str(resolved),
            "checkpoint_sha256": digest,
            "source": "receipt_bound_terminal_checkpoint",
        }

    records = report.get("intermediate_checkpoints")
    if not isinstance(records, list):
        raise CampaignError("completed learner report has no intermediate checkpoints")
    matches = [
        row
        for row in records
        if isinstance(row, dict)
        and _integer(row.get("optimizer_step"), default=-1) == step
    ]
    if len(matches) != 1:
        raise CampaignError(
            f"checkpoint {step} intermediate binding is missing or duplicated"
        )
    record = matches[0]
    try:
        reported = Path(str(record["checkpoint"])).resolve(strict=True)
        size_bytes = int(record["size_bytes"])
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise CampaignError(
            f"checkpoint {step} intermediate binding is malformed"
        ) from error
    if (
        record.get("schema_version") != "train-bc-intermediate-checkpoint-v1"
        or record.get("same_training_trajectory") is not True
        or reported != resolved
        or record.get("checkpoint_sha256") != digest
        or size_bytes != resolved.stat().st_size
    ):
        raise CampaignError(
            f"checkpoint {step} bytes differ from the completed learner report"
        )
    return {
        "schema_version": "stage-c-checkpoint-report-binding-v1",
        "optimizer_step": step,
        "checkpoint": str(resolved),
        "checkpoint_sha256": digest,
        "source": "authenticated_intermediate_checkpoint",
    }


def _recipe() -> dict[str, Any]:
    return {
        "epochs": 1,
        "max_steps": MAX_STEPS,
        "lr": LR,
        "lr_warmup_steps": LR_WARMUP_STEPS,
        "policy_aux_active_batch_size": POLICY_AUX_ACTIVE_BATCH_SIZE,
        "policy_aux_loss_weight": POLICY_AUX_LOSS_WEIGHT,
        "policy_loss_weight": 1.0,
        "soft_target_source": "policy",
        "soft_target_weight": 1.0,
        "soft_target_min_legal_coverage": 1.0,
        "value_loss_weight": 0.25,
        "value_trunk_grad_scale": 0.1,
        "policy_kl_anchor_weight": 0.0,
        "public_card_lr_mult": 1.0,
        "per_game_policy_surprise_weighting": False,
        # The sealed ablation interface accepts a typed map but deliberately
        # refuses an empty override.  Explicit unit weights recover the old
        # all-forced-value-rows behavior without changing any other action
        # type (unspecified types also retain the 1.0 default).
        "forced_row_value_action_type_weights": "END_TURN=1,ROLL=1",
    }


def _expected_policy_teacher_gap_objective(
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    raw_active_batch_size = recipe.get("policy_aux_active_batch_size")
    raw_coefficient = recipe.get("policy_aux_loss_weight")
    try:
        coefficient = float(raw_coefficient)
    except (TypeError, ValueError) as error:
        raise CampaignError("Stage-C policy AUX recipe is malformed") from error
    if (
        isinstance(raw_active_batch_size, bool)
        or not isinstance(raw_active_batch_size, int)
        or raw_active_batch_size < 0
        or isinstance(raw_coefficient, bool)
        or not math.isfinite(coefficient)
        or coefficient < 0.0
    ):
        raise CampaignError("Stage-C policy AUX recipe is malformed")
    active_batch_size = raw_active_batch_size
    enabled = active_batch_size > 0
    return {
        "schema_version": POLICY_TEACHER_GAP_OBJECTIVE_SCHEMA,
        "selection_authority": True,
        "objective_matched": True,
        "formula": (
            "base_plus_coefficient_times_aux_policy_teacher_kl"
            if enabled
            else "base_policy_teacher_kl"
        ),
        "policy_aux_enabled": enabled,
        "policy_aux_active_batch_size": active_batch_size,
        "policy_aux_loss_weight": coefficient,
        "policy_aux_measure": (
            "conditioned_sampling_x_policy_weight" if enabled else "disabled"
        ),
    }


def _require_policy_teacher_gap_objective(
    value: object,
    *,
    expected: Mapping[str, Any],
    where: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or value != dict(expected):
        raise CampaignError(
            f"{where} policy teacher-gap objective differs from Stage-C recipe"
        )
    return copy.deepcopy(value)


def _one_dose_command(plan: Mapping[str, Any]) -> list[str]:
    inputs = plan["inputs"]
    output = Path(str(plan["output_root"])) / "learner"
    return [
        str(inputs["python"]),
        str(inputs["one_dose_trainer"]),
        "--lock",
        str(inputs["lock"]),
        "--data",
        str(inputs["data"]),
        "--validation-manifest",
        str(inputs["validation_manifest"]),
        "--coherent-corpus-admission",
        str(inputs["overlay_admission"]),
        "--architecture-upgrade-receipt",
        str(inputs["architecture_upgrade_receipt"]),
        "--independent-parent-authority",
        str(inputs["independent_parent_authority"]),
        "--checkpoint",
        str(output / "candidate.pt"),
        "--report",
        str(output / "train.report.json"),
        "--receipt",
        str(output / "one-dose.receipt.json"),
        "--python",
        str(inputs["python"]),
        "--gpu",
        "0",
        "--topology",
        "b200-8gpu-ddp",
        "--ddp-canary-receipt",
        str(inputs["ddp_canary_receipt"]),
        "--ablation-id",
        f"stage-c-{str(plan['arm']).lower().replace('_', '-')}",
        "--recipe-overrides-json",
        _canonical_bytes(plan["recipe"]).decode("ascii"),
        "--ablation-code-tree-sha256",
        str(inputs["reviewed_code_tree_sha256"]),
        "--reviewed-lock-file-sha256",
        str(inputs["reviewed_lock_file_sha256"]),
        "--diagnostic-dose-curve",
        "--diagnostic-checkpoint-steps",
        ",".join(map(str, INTERMEDIATE_CHECKPOINT_STEPS)),
    ]


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    try:
        overlay_evidence = overlay.verify_overlay_admission(args.overlay_admission)
        admission_path, admission = stage_a._load_admission(  # noqa: SLF001
            args.overlay_admission
        )
    except (overlay.OverlayError, stage_a.CampaignError) as error:
        raise CampaignError(f"Stage-C overlay admission refused: {error}") from error
    corpus = admission["corpus"]
    arm = str(args.arm)
    sampling = overlay_evidence["receipt"].get("sampling_distribution")
    if (
        arm not in ARMS
        or not isinstance(sampling, dict)
        or sampling.get("schema_version") != overlay.SAMPLING_SCHEMA
        or sampling.get("arm") != arm
    ):
        raise CampaignError("campaign arm differs from overlay sampling distribution")
    data = Path(str(corpus["data_path"])).resolve(strict=True)
    validation = Path(str(corpus["validation_manifest"]["path"])).resolve(strict=True)
    try:
        validation_contract = (
            one_dose.train_bc._load_validation_game_seed_manifest_for_training(  # noqa: SLF001
                validation,
                validation_fraction=0.05,
                validation_seed=17,
                validation_max_samples=0,
                validation_game_seed_ranges=[],
            )
        )
    except SystemExit as error:
        raise CampaignError(f"Stage-C learner validation refused: {error}") from error
    lock = args.lock.expanduser().resolve(strict=True)
    try:
        python = stage_a.base_campaign._python_executable(args.python)  # noqa: SLF001
    except stage_a.base_campaign.CampaignError as error:
        raise CampaignError(f"learner Python refused: {error}") from error
    canary = args.ddp_canary_receipt.expanduser().resolve(strict=True)
    upgrade_path = args.architecture_upgrade_receipt.expanduser().resolve(strict=True)
    if any(
        path.is_symlink() or not path.is_file() for path in (lock, canary, upgrade_path)
    ):
        raise CampaignError("lock/canary/upgrade inputs must be regular files")
    try:
        verified = one_dose.verify_training_inputs(
            lock_path=lock,
            data_path=data,
            validation_path=validation,
            reviewed_lock_file_sha256=_file_sha256(lock),
            coherent_corpus_admission=admission_path,
        )
        upgrade = architecture_upgrade.verify_receipt(upgrade_path)
    except (one_dose.ExecutorError, architecture_upgrade.UpgradeError) as error:
        raise CampaignError(f"Stage-C learner input refused: {error}") from error
    if (
        verified.get("data_kind") != "coherent_direct_memmap_v1"
        or verified.get("recipe", {}).get("soft_target_source") != "policy"
        or float(verified.get("recipe", {}).get("policy_loss_weight", -1.0)) != 1.0
        or float(verified.get("recipe", {}).get("value_loss_weight", -1.0)) != 0.25
        or float(verified.get("recipe", {}).get("policy_kl_anchor_weight", -1.0)) != 0.0
        or verified.get("recipe", {}).get("policy_kl_target") is not None
        or upgrade.get("source", {}).get("sha256") != stage_a.EXPECTED_F7_PARENT_SHA256
        or upgrade.get("module")
        != architecture_upgrade.MODULE_STRUCTURED_ACTION_VALUE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_V3
        or upgrade.get("forward_identical_at_init") is not True
        or float(upgrade.get("forward_max_diff", -1.0)) != 0.0
        or upgrade.get("shared_parameters_bit_identical") is not True
    ):
        raise CampaignError("campaign lost exact f7 function-preserving initialization")

    output_root = args.output_root.expanduser().resolve(strict=False)
    authority_path = output_root / "independent-parent.authority.json"
    authority = stage_a._parent_authority(  # noqa: SLF001
        verified=verified,
        upgrade=upgrade,
        admission_path=admission_path,
        admission=admission,
    )
    _write_json(authority_path, authority)
    code_binding = one_dose._current_ablation_code_binding(verified["lock"])  # noqa: SLF001
    selected_roots_total = int(
        overlay_evidence["receipt"]["projection"]["selected_rows"]
    )
    selected_training_roots = int(
        overlay_evidence["receipt"]["projection"]["selected_training_policy_rows"]
    )
    root_breadth = overlay._verify_stage_c_root_breadth_inventory(  # noqa: SLF001
        overlay_evidence["receipt"].get("root_breadth"),
        selected_rows=selected_roots_total,
    )
    if selected_roots_total <= 0 or selected_training_roots <= 0:
        raise CampaignError("Stage-C overlay has no policy roots")
    if arm == "STRATEGIC_BALANCED":
        _verify_strategic_balanced_sampling_surface(
            sampling,
            selected_training_roots=selected_training_roots,
        )
    trajectory = []
    for step in CHECKPOINT_STEPS:
        aux_draws = POLICY_AUX_ACTIVE_BATCH_SIZE * WORLD_SIZE * step
        trajectory.append(
            {
                "step": step,
                "auxiliary_policy_draws": aux_draws,
                "auxiliary_policy_epochs": aux_draws / selected_training_roots,
                "base_policy_draws_reported_posthoc": True,
            }
        )

    trainer = (REPO_ROOT / "tools" / "a1_one_dose_train.py").resolve(strict=True)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "purpose": "distil_exact_current_coherent_n128_stage_c_targets",
        "arm": arm,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "lineage": {
            "corpus_producer_sha256": verified["producer"]["sha256"],
            "learner_parent_sha256": stage_a.EXPECTED_F7_PARENT_SHA256,
            "initializer_sha256": upgrade["upgraded_initializer"]["sha256"],
            "fresh_adam": True,
            "candidate_chaining": False,
        },
        "policy_target_contract": {
            "target_policy_target_identity_sha256": overlay_evidence["receipt"][
                "target_policy_target_identity_sha256"
            ],
            "selected_unique_roots_total": selected_roots_total,
            "selected_unique_training_roots": selected_training_roots,
            "historical_policy_targets_active": False,
            "nonselected_policy_weight": 0.0,
            "base_value_rows_retained": True,
            "surprise_weighting": False,
            "sampling_distribution": copy.deepcopy(sampling),
            "root_breadth": root_breadth,
        },
        "learner_validation_contract": {
            "schema_version": LEARNER_VALIDATION_CONTRACT_SCHEMA,
            "input_validation_manifest_file_sha256": validation_contract[
                "file_sha256"
            ],
            "input_validation_manifest_sha256": validation_contract[
                "manifest_sha256"
            ],
            "validation_game_seed_count": validation_contract[
                "validation_game_seed_count"
            ],
            "validation_game_seed_set_sha256": validation_contract[
                "validation_game_seed_set_sha256"
            ],
            "training_excluded_game_seed_count": validation_contract[
                "validation_game_seed_count"
            ],
            "training_excluded_game_seed_set_sha256": validation_contract[
                "validation_game_seed_set_sha256"
            ],
        },
        "topology": {
            "name": "b200-8gpu-ddp",
            "world_size": WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
        },
        "recipe": _recipe(),
        "checkpoint_trajectory": trajectory,
        "selection_contract": {
            "optimizer_trust_controller_enabled": False,
            "reason": (
                "historical optimizer-batch KL meter did not enforce posthoc holdout KL"
            ),
            "metric_scope": "frozen_whole_game_validation_policy_active_multi_action_rows",
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "max_parent_kl": MAX_PARENT_KL,
            "max_trunk_relative_l2": MAX_TRUNK_RELATIVE_L2,
            "requires_positive_fresh_parent_teacher_gap_closure": False,
            "requires_checkpoint_local_feature_learning_signal": True,
            "earliest_feature_signal_step": TRAIN_DIAGNOSTIC_CADENCE_BATCHES,
            "stored_generation_prior_selection_authority": False,
            "objective": (
                "maritime_behavior_prefilter_then_minimum_update_within_"
                "posthoc_trust_budgets_then_h2h_final_ranking"
            ),
            "teacher_gap_closure_ranking_authority": False,
            "behavioral_competence_gate": (
                _behavioral_competence_gate_contract()
            ),
            "value_quality_gate": {
                "policy": str(
                    getattr(args, "value_gate_policy", VALUE_GATE_POLICY)
                ),
                "metric": "primary_value_loss",
                "metric_kind": "scalar_mse",
                "parent_baseline": "fresh_exact_report_bound_parent_forward",
                "max_absolute_regression": float(
                    getattr(
                        args,
                        "max_value_mse_regression",
                        MAX_VALUE_MSE_REGRESSION,
                    )
                ),
                "phase_slices_required": [],
                "phase_slice_reason": (
                    "current posthoc evidence has no objective-matched per-phase "
                    "value sufficient statistics"
                ),
            },
            "playing_strength_evaluation_required": True,
        },
        "optimizer_surface_contract": {
            "shared_trunk_trainable": True,
            "value_trunk_grad_scale": 0.1,
            "objective_gradient_interference_cadence_batches": (
                OBJECTIVE_GRADIENT_CADENCE_BATCHES
            ),
            "minimum_objective_gradient_observations": (
                MINIMUM_OBJECTIVE_GRADIENT_OBSERVATIONS
            ),
            "trainable_adapter_modules": sorted(TRAINABLE_ADAPTER_MODULES),
            "implicit_data_driven_freeze": False,
            "explicit_freeze_modules": "",
        },
        "feature_learning_signal_contract": {
            "effective_architecture": dict(EFFECTIVE_FEATURE_CONTRACT),
            "module_optimizer_observability_schema": (
                "module-optimizer-observability-v1"
            ),
            "norm_scope": "global_replicated",
            "diagnostic_cadence_batches": TRAIN_DIAGNOSTIC_CADENCE_BATCHES,
            "minimum_observed_steps": MINIMUM_FEATURE_SIGNAL_OBSERVATIONS,
            "required_modules": sorted(FEATURE_SIGNAL_MODULES),
            "required_positive_fields": list(POSITIVE_OPTIMIZER_SIGNAL_FIELDS),
        },
        "inputs": {
            "python": str(python),
            "one_dose_trainer": str(trainer),
            "one_dose_trainer_sha256": _file_sha256(trainer),
            "lock": str(lock),
            "reviewed_lock_file_sha256": _file_sha256(lock),
            "data": str(data),
            "corpus_meta_file_sha256": _file_sha256(data / "corpus_meta.json"),
            "payload_inventory_sha256": corpus["payload_inventory_sha256"],
            "validation_manifest": str(validation),
            "validation_manifest_file_sha256": _file_sha256(validation),
            "overlay_admission": str(admission_path),
            "overlay_admission_file_sha256": _file_sha256(admission_path),
            "overlay_admission_sha256": admission["admission_sha256"],
            "overlay_materialization_receipt": admission["stage_c_policy_overlay"][
                "materialization_receipt"
            ]["path"],
            "overlay_materialization_receipt_sha256": overlay_evidence["receipt"][
                "receipt_sha256"
            ],
            "architecture_upgrade_receipt": str(upgrade_path),
            "architecture_upgrade_receipt_file_sha256": _file_sha256(upgrade_path),
            "independent_parent_authority": str(authority_path),
            "independent_parent_authority_file_sha256": _file_sha256(authority_path),
            "independent_parent_authority_sha256": authority["authority_sha256"],
            "ddp_canary_receipt": str(canary),
            "ddp_canary_receipt_file_sha256": _file_sha256(canary),
            "reviewed_code_tree_sha256": code_binding["code_tree_sha256"],
        },
        "output_root": str(output_root),
        "expected_artifacts": {
            "terminal_checkpoint": str(output_root / "learner" / "candidate.pt"),
            "intermediate_checkpoints": [
                str(output_root / "learner" / f"candidate_step{step:04d}.pt")
                for step in INTERMEDIATE_CHECKPOINT_STEPS
            ],
            "report": str(output_root / "learner" / "train.report.json"),
            "one_dose_receipt": str(output_root / "learner" / "one-dose.receipt.json"),
            "execution_receipt": str(output_root / "learner.execution.receipt.json"),
            "fingerprint": str(output_root / "fingerprint.fresh-parent.json"),
        },
    }
    payload["command"] = _one_dose_command(payload)
    payload["command_sha256"] = _value_sha256(payload["command"])
    payload["campaign_sha256"] = _value_sha256(payload)
    return payload


def _verify_inputs(plan: Mapping[str, Any]) -> None:
    inputs = plan["inputs"]
    checks = (
        ("one_dose_trainer", "one_dose_trainer_sha256"),
        ("lock", "reviewed_lock_file_sha256"),
        ("validation_manifest", "validation_manifest_file_sha256"),
        ("overlay_admission", "overlay_admission_file_sha256"),
        ("architecture_upgrade_receipt", "architecture_upgrade_receipt_file_sha256"),
        ("independent_parent_authority", "independent_parent_authority_file_sha256"),
        ("ddp_canary_receipt", "ddp_canary_receipt_file_sha256"),
    )
    for path_key, sha_key in checks:
        path = Path(str(inputs[path_key])).resolve(strict=True)
        if path.is_symlink() or _file_sha256(path) != inputs[sha_key]:
            raise CampaignError(f"campaign input bytes changed: {path_key}")
    overlay.verify_overlay_admission(Path(str(inputs["overlay_admission"])))
    recipe = plan.get("recipe", {})
    optimizer_surface = plan.get("optimizer_surface_contract")
    feature_signal = plan.get("feature_learning_signal_contract")
    value_gate = plan.get("selection_contract", {}).get("value_quality_gate")
    target_contract = plan.get("policy_target_contract", {})
    validation_contract = plan.get("learner_validation_contract")
    try:
        overlay._verify_stage_c_root_breadth_inventory(  # noqa: SLF001
            target_contract.get("root_breadth"),
            selected_rows=int(target_contract.get("selected_unique_roots_total", -1)),
        )
    except overlay.OverlayError as error:
        raise CampaignError(f"Stage-C policy-root breadth refused: {error}") from error
    if (
        plan.get("arm") not in ARMS
        or float(recipe.get("value_trunk_grad_scale", -1.0)) != 0.1
        or float(recipe.get("soft_target_min_legal_coverage", -1.0)) != 1.0
        or float(recipe.get("policy_kl_anchor_weight", -1.0)) != 0.0
        or bool(recipe.get("per_game_policy_surprise_weighting", True))
        or optimizer_surface
        != {
            "shared_trunk_trainable": True,
            "value_trunk_grad_scale": 0.1,
            "objective_gradient_interference_cadence_batches": (
                OBJECTIVE_GRADIENT_CADENCE_BATCHES
            ),
            "minimum_objective_gradient_observations": (
                MINIMUM_OBJECTIVE_GRADIENT_OBSERVATIONS
            ),
            "trainable_adapter_modules": sorted(TRAINABLE_ADAPTER_MODULES),
            "implicit_data_driven_freeze": False,
            "explicit_freeze_modules": "",
        }
        or feature_signal
        != {
            "effective_architecture": dict(EFFECTIVE_FEATURE_CONTRACT),
            "module_optimizer_observability_schema": (
                "module-optimizer-observability-v1"
            ),
            "norm_scope": "global_replicated",
            "diagnostic_cadence_batches": (TRAIN_DIAGNOSTIC_CADENCE_BATCHES),
            "minimum_observed_steps": MINIMUM_FEATURE_SIGNAL_OBSERVATIONS,
            "required_modules": sorted(FEATURE_SIGNAL_MODULES),
            "required_positive_fields": list(POSITIVE_OPTIMIZER_SIGNAL_FIELDS),
        }
        or not isinstance(value_gate, dict)
        or not isinstance(validation_contract, dict)
        or validation_contract.get("schema_version")
        != LEARNER_VALIDATION_CONTRACT_SCHEMA
        or validation_contract.get("input_validation_manifest_file_sha256")
        != inputs["validation_manifest_file_sha256"]
        or validation_contract.get("validation_game_seed_count")
        != validation_contract.get("training_excluded_game_seed_count")
        or validation_contract.get("validation_game_seed_set_sha256")
        != validation_contract.get("training_excluded_game_seed_set_sha256")
        or value_gate.get("policy") not in VALUE_GATE_POLICIES
        or value_gate.get("metric") != "primary_value_loss"
        or value_gate.get("metric_kind") != "scalar_mse"
        or value_gate.get("parent_baseline")
        != "fresh_exact_report_bound_parent_forward"
        or not math.isfinite(
            float(value_gate.get("max_absolute_regression", math.nan))
        )
        or float(value_gate.get("max_absolute_regression", math.nan)) < 0.0
        or value_gate.get("phase_slices_required") != []
    ):
        raise CampaignError("Stage-C clean learner semantics drifted")
    if plan.get("command_sha256") != _value_sha256(_one_dose_command(plan)):
        raise CampaignError("Stage-C learner command drifted from campaign")


def _authenticate_completed_stage_c_dose(
    plan: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    receipt_path = Path(str(plan["expected_artifacts"]["one_dose_receipt"]))
    try:
        receipt = one_dose._load_authenticated_completed_ablation_receipt(  # noqa: SLF001
            receipt_path
        )
    except one_dose.ExecutorError as error:
        raise CampaignError(
            f"completed Stage-C learner receipt refused: {error}"
        ) from error
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise CampaignError("completed Stage-C learner receipt has no outputs")
    report_path = Path(str(plan["expected_artifacts"]["report"])).resolve(strict=True)
    terminal_checkpoint = Path(
        str(plan["expected_artifacts"]["terminal_checkpoint"])
    ).resolve(strict=True)
    expected_bindings = (
        ("report", "report_sha256", report_path),
        ("checkpoint", "checkpoint_sha256", terminal_checkpoint),
    )
    for path_field, digest_field, expected_path in expected_bindings:
        try:
            actual_path = Path(str(outputs[path_field])).resolve(strict=True)
        except (KeyError, OSError) as error:
            raise CampaignError(
                f"completed Stage-C receipt lacks {path_field} binding"
            ) from error
        if (
            actual_path != expected_path
            or outputs.get(digest_field) != _file_sha256(expected_path)
        ):
            raise CampaignError(
                f"completed Stage-C receipt {path_field} binding drifted"
            )
    report = _load_json(report_path, where="completed Stage-C learner report")[1]
    information_surface = report.get("training_information_surface")
    freeze = (
        information_surface.get("explicit_module_freeze")
        if isinstance(information_surface, dict)
        else None
    )
    if (
        not isinstance(information_surface, dict)
        or float(report.get("value_trunk_grad_scale", -1.0)) != 0.1
        or str(report.get("freeze_modules", "")) != ""
        or freeze is not None
    ):
        raise CampaignError(
            "completed learner did not keep both feature adapters trainable"
        )
    validation_contract = plan.get("learner_validation_contract")
    if (
        not isinstance(validation_contract, Mapping)
        or report.get("input_validation_game_seed_manifest_sha256")
        != validation_contract.get("input_validation_manifest_file_sha256")
        or report.get("validation_game_seed_count")
        != validation_contract.get("validation_game_seed_count")
        or report.get("validation_game_seed_set_sha256")
        != validation_contract.get("validation_game_seed_set_sha256")
        or report.get("training_excluded_game_seed_count")
        != validation_contract.get("training_excluded_game_seed_count")
        or report.get("training_excluded_game_seed_set_sha256")
        != validation_contract.get("training_excluded_game_seed_set_sha256")
    ):
        raise CampaignError(
            "completed learner did not execute the bound validation exclusion contract"
        )
    _verify_completed_feature_learning_signal(report)
    _verify_completed_objective_gradient_signal(report)
    return receipt_path, receipt, report_path, report


def _run(plan: Mapping[str, Any], *, go: bool) -> dict[str, Any]:
    _verify_inputs(plan)
    command = _one_dose_command(plan)
    if not go:
        return {"mode": "dry-run", "command": command}
    receipt_path = Path(str(plan["expected_artifacts"]["one_dose_receipt"]))
    adopted_completed_receipt = receipt_path.exists() or receipt_path.is_symlink()
    if not adopted_completed_receipt:
        result = subprocess.run([*command, "--go"], check=False)
        if result.returncode != 0:
            raise CampaignError(f"Stage-C aligned learner exited {result.returncode}")
    receipt_path, receipt, report_path, report = (
        _authenticate_completed_stage_c_dose(plan)
    )
    aux_draws = int(report.get("policy_aux_active_rows", -1))
    unique_rows = int(report.get("policy_aux_unique_source_rows", -1))
    expected_aux_draws = POLICY_AUX_ACTIVE_BATCH_SIZE * WORLD_SIZE * MAX_STEPS
    selected_roots = int(
        plan["policy_target_contract"]["selected_unique_training_roots"]
    )
    coverage_evidence = _verify_realized_policy_aux_coverage(
        arm=str(plan["arm"]),
        selected_training_roots=selected_roots,
        auxiliary_draws=aux_draws,
        unique_source_rows=unique_rows,
    )
    if (
        aux_draws != expected_aux_draws
        or not math.isclose(
            float(report.get("policy_aux_reuse_factor", math.nan)),
            aux_draws / unique_rows,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise CampaignError("completed learner unique-root exposure drifted")
    execution: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "campaign_sha256": plan["campaign_sha256"],
        "one_dose_receipt": {
            "path": str(receipt_path),
            "file_sha256": _file_sha256(receipt_path),
            "receipt_sha256": receipt["receipt_sha256"],
        },
        "report": {
            "path": str(report_path),
            "file_sha256": _file_sha256(report_path),
        },
        "checkpoint": {
            "path": receipt["outputs"]["checkpoint"],
            "sha256": receipt["outputs"]["checkpoint_sha256"],
        },
        "policy_exposure": {
            "selected_unique_training_root_population": selected_roots,
            "selected_unique_root_population_total": int(
                plan["policy_target_contract"]["selected_unique_roots_total"]
            ),
            "auxiliary_draws": aux_draws,
            "unique_auxiliary_source_rows": unique_rows,
            "unique_root_coverage_fraction": unique_rows / selected_roots,
            "auxiliary_reuse_factor": aux_draws / unique_rows,
            "realized_coverage_authority": coverage_evidence,
            "base_policy_active_draws": int(report.get("policy_base_active_rows", 0)),
            "root_breadth": copy.deepcopy(
                plan["policy_target_contract"]["root_breadth"]
            ),
        },
        "optimizer_batch_kl_used_as_trust_authority": False,
        "posthoc_frozen_holdout_selection_required": True,
        "existing_completed_dose_adopted": adopted_completed_receipt,
    }
    execution["execution_sha256"] = _value_sha256(execution)
    execution_path = Path(str(plan["expected_artifacts"]["execution_receipt"]))
    _write_json(execution_path, execution)
    return {
        "mode": "finalize-existing" if adopted_completed_receipt else "go",
        "receipt": str(receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "checkpoint": receipt["outputs"]["checkpoint"],
        "checkpoint_sha256": receipt["outputs"]["checkpoint_sha256"],
        "execution_receipt": str(execution_path),
        "execution_sha256": execution["execution_sha256"],
        "policy_exposure": execution["policy_exposure"],
    }


def _checkpoint_path(plan: Mapping[str, Any], step: int) -> Path:
    root = Path(str(plan["output_root"])) / "learner"
    return (
        root / "candidate.pt"
        if step == MAX_STEPS
        else root / f"candidate_step{step:04d}.pt"
    )


def _trunk_relative_l2(report: Mapping[str, Any]) -> float:
    groups = report.get("groups")
    if not isinstance(groups, dict):
        raise CampaignError("layer drift report has no groups")
    selected = [
        row
        for name, row in groups.items()
        if isinstance(row, dict)
        and (
            name in {"input_encoders", "shared", "topology_adapter"}
            or name.startswith("transformer_block_")
        )
    ]
    baseline = sum(float(row["baseline_l2"]) ** 2 for row in selected)
    delta = sum(float(row["delta_energy"]) for row in selected)
    if not selected or baseline <= 0.0 or delta < 0.0:
        raise CampaignError("layer drift cannot define trunk relative L2")
    return math.sqrt(delta / baseline)


def _posthoc_evaluation_surface(report: Mapping[str, Any]) -> dict[str, Any]:
    inputs = report.get("inputs")
    if not isinstance(inputs, dict):
        raise CampaignError("posthoc functional report has no input bindings")
    try:
        return {
            "schema_version": report["schema_version"],
            "arch": report["arch"],
            "batch_size": int(report["batch_size"]),
            "validation_rows": int(report["validation_rows"]),
            "validation_game_seed_set_sha256": report[
                "validation_game_seed_set_sha256"
            ],
            "training_report_sha256": inputs["training_report"]["sha256"],
            "memmap_fingerprint": inputs["memmap"]["fingerprint"],
            "memmap_payload_inventory_sha256": inputs["memmap"][
                "payload_inventory_sha256"
            ],
            "validation_manifest_sha256": inputs["validation_manifest"]["sha256"],
            "validation_manifest_semantic_sha256": inputs["validation_manifest"][
                "manifest_sha256"
            ],
        }
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignError("posthoc evaluation surface is malformed") from error


def _pair_separate_parent_evidence(
    functional: Mapping[str, Any], parent_functional: Mapping[str, Any]
) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Bind separately emitted parent/candidate forwards on one exact surface."""

    if _posthoc_evaluation_surface(functional) != _posthoc_evaluation_surface(
        parent_functional
    ):
        raise CampaignError("separate parent and candidate holdout surfaces differ")
    candidate_inputs = functional["inputs"]
    parent_inputs = parent_functional["inputs"]
    parent_checkpoint = parent_inputs.get("checkpoint")
    if not isinstance(parent_checkpoint, dict):
        raise CampaignError("separate parent report has no checkpoint binding")
    parent_sha = parent_checkpoint.get("sha256")
    if (
        candidate_inputs.get("parent_checkpoint", {}).get("sha256") != parent_sha
        or parent_inputs.get("parent_checkpoint", {}).get("sha256") != parent_sha
    ):
        raise CampaignError("separate functional reports bind different parents")
    candidate_gap = functional.get("teacher_gap")
    parent_gap = parent_functional.get("teacher_gap")
    if not isinstance(candidate_gap, dict) or not isinstance(parent_gap, dict):
        raise CampaignError("separate functional report has no teacher-gap metrics")
    try:
        rows = int(candidate_gap["active_policy_teacher_gap_rows"])
        parent_rows = int(parent_gap["active_policy_teacher_gap_rows"])
        candidate_kl = float(candidate_gap["active_policy_kl_target_model_mean"])
        parent_kl = float(parent_gap["active_policy_kl_target_model_mean"])
        candidate_prior = float(candidate_gap["active_policy_kl_target_prior_mean"])
        parent_prior = float(parent_gap["active_policy_kl_target_prior_mean"])
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignError("separate teacher-gap metrics are malformed") from error
    if (
        rows <= 0
        or rows != parent_rows
        or not math.isclose(candidate_prior, parent_prior, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise CampaignError("separate teacher-gap row/target surfaces differ")
    absolute = parent_kl - candidate_kl
    paired = {
        "schema_version": SEPARATE_PARENT_GAP_SCHEMA,
        "selection_authority": True,
        "authority": "fresh_exact_report_bound_parent_forward",
        "surface": "same_holdout_same_targets_fresh_exact_parent_forward",
        "rows": rows,
        "parent_active_policy_kl_target_model_mean": parent_kl,
        "candidate_active_policy_kl_target_model_mean": candidate_kl,
        "absolute_teacher_gap_closure": absolute,
        "relative_teacher_gap_closure": absolute / parent_kl
        if parent_kl > 1.0e-8
        else 0.0,
        "improved_over_exact_parent": bool(absolute > 0.0),
        "stored_generation_prior": {
            "active_policy_kl_target_prior_mean": candidate_prior,
            "selection_authority": False,
            "semantic_role": "legacy_generation_operator_diagnostic_only",
        },
    }
    return paired, candidate_gap, parent_gap


def _fresh_parent_teacher_gap(
    functional: Mapping[str, Any],
    *,
    parent_functional: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate the only teacher-gap surface allowed for selection.

    Historical rows retain a generation-time ``prior_policy`` distribution.
    That distribution may include a different inference operator (notably D6
    averaging), so its closure is useful telemetry but cannot establish that a
    checkpoint improved over the learner's exact parent bytes.
    """

    paired = functional.get("paired_parent_teacher_gap")
    separate_legacy = separate_parent_projection = None
    if not isinstance(paired, dict):
        if parent_functional is None:
            raise CampaignError(
                "functional report has no authoritative fresh-parent gap"
            )
        paired, separate_legacy, separate_parent_projection = (
            _pair_separate_parent_evidence(functional, parent_functional)
        )
    evidence_schema = paired.get("schema_version")
    if evidence_schema in {PAIRED_PARENT_GAP_SCHEMA, SEPARATE_PARENT_GAP_SCHEMA}:
        if (
            paired.get("selection_authority") is not True
            or paired.get("authority") != "fresh_exact_report_bound_parent_forward"
        ):
            raise CampaignError("fresh-parent evidence authority is malformed")
        stored_prior = paired.get("stored_generation_prior")
        if (
            not isinstance(stored_prior, dict)
            or stored_prior.get("selection_authority") is not False
            or stored_prior.get("semantic_role")
            != "legacy_generation_operator_diagnostic_only"
        ):
            raise CampaignError(
                "stored generation prior was not marked diagnostic-only"
            )
        absolute_key = "absolute_teacher_gap_closure"
        stored_prior_value = stored_prior.get(
            "active_policy_kl_target_prior_mean", math.nan
        )
    elif evidence_schema == TRANSITIONAL_PAIRED_PARENT_GAP_SCHEMA:
        # The first live Stage-C fingerprint run emitted this transition shape
        # before checkpoint selection was corrected. Its exact parent and
        # candidate forwards remain valid, expensive evidence. Authenticate
        # those values while quarantining its stored-prior closure.
        if (
            paired.get("surface")
            != "same_holdout_same_targets_fresh_exact_parent_forward"
            or paired.get("stored_prior_closure_is_legacy_diagnostic_only") is not True
        ):
            raise CampaignError("transitional fresh-parent evidence is malformed")
        absolute_key = "absolute_target_kl_improvement"
        stored_prior_value = paired.get(
            "stored_prior_active_policy_kl_target_mean", math.nan
        )
    else:
        raise CampaignError("functional report fresh-parent schema is unsupported")
    try:
        parent_kl = float(paired["parent_active_policy_kl_target_model_mean"])
        candidate_kl = float(paired["candidate_active_policy_kl_target_model_mean"])
        absolute = float(paired[absolute_key])
        relative = float(paired["relative_teacher_gap_closure"])
        stored_prior_value = float(stored_prior_value)
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignError("fresh-parent teacher-gap fields are malformed") from error
    if (
        int(paired.get("rows", 0)) <= 0
        or not all(
            math.isfinite(value)
            for value in (parent_kl, candidate_kl, absolute, relative)
        )
        or parent_kl < -1.0e-9
        or candidate_kl < -1.0e-9
        or not math.isclose(
            absolute,
            parent_kl - candidate_kl,
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        )
    ):
        raise CampaignError("fresh-parent teacher-gap arithmetic is invalid")
    expected_relative = absolute / parent_kl if parent_kl > 1.0e-8 else 0.0
    if not math.isclose(
        relative, expected_relative, rel_tol=1.0e-9, abs_tol=1.0e-12
    ) or bool(paired.get("improved_over_exact_parent")) != bool(absolute > 0.0):
        raise CampaignError("fresh-parent teacher-gap closure is inconsistent")
    if evidence_schema == PAIRED_PARENT_GAP_SCHEMA:
        legacy = functional.get("legacy_stored_generation_prior_teacher_gap")
        if (
            not isinstance(legacy, dict)
            or legacy.get("selection_authority") is not False
            or legacy.get("semantic_role")
            != "legacy_generation_operator_diagnostic_only"
        ):
            raise CampaignError("functional report did not quarantine legacy closure")
        compatibility_semantics = functional.get("teacher_gap_semantics")
        if (
            not isinstance(compatibility_semantics, dict)
            or compatibility_semantics.get("selection_authority") is not False
            or compatibility_semantics.get("authoritative_replacement")
            != "paired_parent_teacher_gap"
        ):
            raise CampaignError(
                "compatibility teacher gap was not marked non-authoritative"
            )
    elif evidence_schema == TRANSITIONAL_PAIRED_PARENT_GAP_SCHEMA:
        legacy = functional.get("teacher_gap")
        if not isinstance(legacy, dict):
            raise CampaignError("transitional report lacks stored-prior diagnostics")
    else:
        assert separate_legacy is not None and separate_parent_projection is not None
        legacy = separate_legacy
    parent_projection = (
        separate_parent_projection
        if evidence_schema == SEPARATE_PARENT_GAP_SCHEMA
        else functional.get("parent_teacher_gap")
    )
    if (
        not isinstance(parent_projection, dict)
        or int(parent_projection.get("active_policy_teacher_gap_rows", 0))
        != int(paired["rows"])
        or not math.isclose(
            float(
                parent_projection.get("active_policy_kl_target_model_mean", math.nan)
            ),
            parent_kl,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(legacy.get("active_policy_kl_target_model_mean", math.nan)),
            candidate_kl,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(legacy.get("active_policy_kl_target_prior_mean", math.nan)),
            stored_prior_value,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise CampaignError("fresh-parent gap differs from its emitted projections")
    legacy_closure = float(legacy.get("active_policy_teacher_gap_closure", math.nan))
    if not math.isfinite(legacy_closure):
        raise CampaignError("legacy stored-prior closure is malformed")
    return {
        "parent_target_kl_mean": parent_kl,
        "candidate_target_kl_mean": candidate_kl,
        "absolute_closure": absolute,
        "relative_closure": relative,
        "improved": bool(absolute > 0.0),
        "rows": int(paired["rows"]),
        "evidence_schema_version": str(evidence_schema),
        "legacy_stored_prior_closure": legacy_closure,
    }


def _fresh_parent_fingerprint_path(plan: Mapping[str, Any]) -> Path:
    configured = Path(str(plan["expected_artifacts"]["fingerprint"]))
    if configured.name.endswith(".fresh-parent.json"):
        return configured
    if configured.suffix == ".json":
        return configured.with_name(f"{configured.stem}.fresh-parent.json")
    return configured.with_name(f"{configured.name}.fresh-parent.json")


def _legacy_fingerprint_path(plan: Mapping[str, Any]) -> Path:
    configured = Path(str(plan["expected_artifacts"]["fingerprint"]))
    if configured.name.endswith(".fresh-parent.json"):
        stem = configured.name.removesuffix(".fresh-parent.json")
        return configured.with_name(f"{stem}.json")
    return configured


def _parent_functional_artifact(
    output_root: Path,
) -> tuple[Path, dict[str, Any]] | None:
    candidates = (
        output_root / "parent.functional.json",
        output_root.parent / "fingerprints-direct" / "parent.functional.json",
    )
    for candidate in candidates:
        if candidate.is_symlink():
            raise CampaignError(
                f"parent functional artifact must not be a symlink: {candidate}"
            )
        if candidate.is_file():
            return (
                candidate.resolve(strict=True),
                _load_json(candidate, where="separate exact-parent functional")[1],
            )
    return None


def _functional_artifact_path(
    output_root: Path,
    step: int,
    *,
    allow_separate_parent: bool,
    expected_bindings: Mapping[str, Any],
) -> Path:
    """Reuse only the competence-versioned, freshly authenticated evidence."""

    del allow_separate_parent
    versioned = (
        output_root / f"step{step:04d}.functional.maritime-competence.json"
    )
    if versioned.is_symlink():
        raise CampaignError(
            f"functional artifact must not be a symlink: {versioned}"
        )
    if not versioned.is_file():
        return versioned
    payload = _load_json(
        versioned, where=f"step {step} maritime competence functional evidence"
    )[1]
    _authenticate_cached_functional_evidence(
        payload, expected=expected_bindings, step=step
    )
    paired = payload.get("paired_parent_teacher_gap")
    if (
        not isinstance(paired, dict)
        or paired.get("schema_version")
        not in {PAIRED_PARENT_GAP_SCHEMA, TRANSITIONAL_PAIRED_PARENT_GAP_SCHEMA}
    ):
        raise CampaignError(
            f"maritime competence functional artifact is malformed: {versioned}"
        )
    _evaluate_maritime_behavioral_competence_gate(payload)
    return versioned


def _authenticate_cached_functional_evidence(
    payload: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    step: int,
) -> None:
    """Refuse cached evidence unless every live artifact binding still matches."""

    inputs = payload.get("inputs")
    shared = payload.get("shared_holdout")
    if not isinstance(inputs, dict) or not isinstance(shared, dict):
        raise CampaignError(f"step {step} cached functional bindings are missing")
    shared_unsigned = {
        key: value
        for key, value in shared.items()
        if key
        not in {
            "identity_sha256",
            "training_report",
            "memmap",
            "validation_manifest",
        }
    }
    try:
        input_projection = {
            "checkpoint": {
                "path": inputs["checkpoint"]["path"],
                "sha256": inputs["checkpoint"]["sha256"],
            },
            "training_report": {
                "path": inputs["training_report"]["path"],
                "sha256": inputs["training_report"]["sha256"],
            },
            "memmap": {
                "path": inputs["memmap"]["path"],
                "fingerprint": inputs["memmap"]["fingerprint"],
                "payload_inventory_sha256": inputs["memmap"][
                    "payload_inventory_sha256"
                ],
            },
            "validation_manifest": {
                "path": inputs["validation_manifest"]["path"],
                "sha256": inputs["validation_manifest"]["sha256"],
                "manifest_sha256": inputs["validation_manifest"]["manifest_sha256"],
            },
            "validation_game_seed_set_sha256": payload[
                "validation_game_seed_set_sha256"
            ],
        }
        shared_projection = {
            "training_report": shared["training_report"],
            "memmap": shared["memmap"],
            "validation_manifest": shared["validation_manifest"],
            "validation_game_seed_set_sha256": shared[
                "validation_game_seed_set_sha256"
            ],
        }
    except (KeyError, TypeError) as error:
        raise CampaignError(
            f"step {step} cached functional bindings are malformed"
        ) from error
    expected_shared = {
        key: expected[key]
        for key in (
            "training_report",
            "memmap",
            "validation_manifest",
            "validation_game_seed_set_sha256",
        )
    }
    if (
        input_projection != dict(expected)
        or shared_projection != expected_shared
        or shared.get("identity_sha256") != _value_sha256(shared_unsigned)
    ):
        raise CampaignError(
            f"step {step} cached functional evidence is stale or misbound"
        )


def _select_fingerprint_winner(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_objective: Mapping[str, Any] | None = None,
    expected_behavioral_competence_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Nominate the smallest representation- and value-safe update.

    B200 traces showed that larger teacher-gap closure was anti-correlated with
    paired playing strength. Closure is therefore diagnostic only: requiring it
    to be positive lets a noisy teacher decide which checkpoints are allowed to
    reach the paired H2H screen, even though that screen is the actual strength
    authority.
    """

    expected = (
        _expected_policy_teacher_gap_objective(_recipe())
        if expected_objective is None
        else dict(expected_objective)
    )
    behavior_contract = (
        _behavioral_competence_gate_contract()
        if expected_behavioral_competence_gate is None
        else dict(expected_behavioral_competence_gate)
    )
    authenticated_gates: dict[int, dict[str, Any]] = {}
    for row in records:
        _require_policy_teacher_gap_objective(
            row.get("policy_teacher_gap_objective"),
            expected=expected,
            where=f"step {row.get('step', '?')} fingerprint",
        )
        authenticated_gates[id(row)] = (
            _require_maritime_behavioral_competence_gate_result(
                row.get("behavioral_competence_gate"),
                expected_contract=behavior_contract,
                where=f"step {row.get('step', '?')} fingerprint",
            )
        )

    eligible = [
        row
        for row in records
        if row.get("feature_learning_signal_authenticated") is True
        and float(row["parent_kl"]) <= MAX_PARENT_KL
        and float(row["trunk_relative_l2"]) <= MAX_TRUNK_RELATIVE_L2
        and row.get("value_quality_gate", {}).get(
            "selection_admitted",
            row.get("value_quality_gate", {}).get("passed"),
        )
        is True
        and authenticated_gates[id(row)]["selection_admitted"] is True
    ]
    if not eligible:
        return None
    return dict(
        min(
            eligible,
            key=lambda row: (
                int(row["step"]),
                float(row["parent_kl"]),
                float(row["trunk_relative_l2"]),
            ),
        )
    )


def _value_projection_from_metrics(
    functional: Mapping[str, Any], *, where: str
) -> dict[str, Any]:
    metrics = functional.get("metrics")
    if not isinstance(metrics, dict):
        raise CampaignError(f"{where} functional report has no raw value metrics")
    try:
        primary = float(metrics["primary_value_loss"])
        scalar = float(metrics["scalar_value_mse_diagnostic"])
        raw_value = float(metrics["value_loss"])
        kind = str(metrics["primary_value_loss_kind"])
        mass = float(metrics["loss_denominators"]["value_loss"])
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignError(f"{where} raw value metrics are malformed") from error
    if (
        kind != "scalar_mse"
        or not all(math.isfinite(value) for value in (primary, scalar, raw_value, mass))
        or mass <= 0.0
        or not math.isclose(primary, scalar, rel_tol=0.0, abs_tol=1.0e-12)
        or not math.isclose(primary, raw_value, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise CampaignError(f"{where} raw value metrics are inconsistent")
    return {
        "schema_version": VALUE_QUALITY_SCHEMA,
        "selection_authority": True,
        "surface": "same_reconstructed_holdout_and_value_weight_measure",
        "metric": "primary_value_loss",
        "metric_kind": kind,
        "value": primary,
        "scalar_value_mse_diagnostic": scalar,
        "value_weight_mass": mass,
    }


def _reconciled_value_projection(
    functional: Mapping[str, Any],
    *,
    field: str,
    where: str,
    require_emitted: bool,
) -> dict[str, Any]:
    raw = _value_projection_from_metrics(functional, where=where)
    emitted = functional.get(field)
    if emitted is None and not require_emitted:
        return raw
    if not isinstance(emitted, dict):
        raise CampaignError(f"{where} emitted value-quality projection is missing")
    try:
        emitted_value = float(emitted["value"])
        emitted_scalar = float(emitted["scalar_value_mse_diagnostic"])
        emitted_mass = float(emitted["value_weight_mass"])
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignError(
            f"{where} emitted value-quality projection is malformed"
        ) from error
    if (
        emitted.get("schema_version") != VALUE_QUALITY_SCHEMA
        or emitted.get("selection_authority") is not True
        or emitted.get("surface")
        != "same_reconstructed_holdout_and_value_weight_measure"
        or emitted.get("metric") != raw["metric"]
        or emitted.get("metric_kind") != raw["metric_kind"]
        or not math.isclose(
            emitted_value, float(raw["value"]), rel_tol=0.0, abs_tol=1.0e-12
        )
        or not math.isclose(
            emitted_scalar,
            float(raw["scalar_value_mse_diagnostic"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            emitted_mass,
            float(raw["value_weight_mass"]),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise CampaignError(
            f"{where} emitted value-quality projection contradicts raw metrics"
        )
    return dict(emitted)


def _validated_emitted_value_projection(
    projection: Mapping[str, Any], *, where: str
) -> dict[str, Any]:
    try:
        value = float(projection["value"])
        scalar = float(projection["scalar_value_mse_diagnostic"])
        mass = float(projection["value_weight_mass"])
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignError(f"{where} value-quality projection is malformed") from error
    if (
        projection.get("schema_version") != VALUE_QUALITY_SCHEMA
        or projection.get("selection_authority") is not True
        or projection.get("surface")
        != "same_reconstructed_holdout_and_value_weight_measure"
        or projection.get("metric") != "primary_value_loss"
        or projection.get("metric_kind") != "scalar_mse"
        or not all(math.isfinite(item) for item in (value, scalar, mass))
        or mass <= 0.0
        or not math.isclose(value, scalar, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise CampaignError(f"{where} value-quality projection is inconsistent")
    return dict(projection)


def _paired_value_quality(
    functional: Mapping[str, Any],
    *,
    parent_functional: Mapping[str, Any] | None,
    policy: str,
    max_absolute_regression: float,
) -> dict[str, Any]:
    paired = functional.get("paired_parent_value_quality")
    if not isinstance(paired, dict):
        if parent_functional is None:
            raise CampaignError(
                "functional report has no authoritative fresh-parent value baseline"
            )
        if _posthoc_evaluation_surface(functional) != _posthoc_evaluation_surface(
            parent_functional
        ):
            raise CampaignError("parent and candidate value holdout surfaces differ")
        candidate_projection = _reconciled_value_projection(
            functional,
            field="value_quality",
            where="candidate",
            require_emitted=False,
        )
        parent_projection = _reconciled_value_projection(
            parent_functional,
            field="value_quality",
            where="parent",
            require_emitted=False,
        )
        candidate = float(candidate_projection["value"])
        parent = float(parent_projection["value"])
        candidate_kind = str(candidate_projection["metric_kind"])
        parent_kind = str(parent_projection["metric_kind"])
        candidate_mass = float(candidate_projection["value_weight_mass"])
        parent_mass = float(parent_projection["value_weight_mass"])
        paired = {
            "schema_version": PAIRED_PARENT_VALUE_SCHEMA,
            "selection_authority": True,
            "surface": (
                "same_holdout_same_objective_weights_fresh_exact_parent_forward"
            ),
            "metric": "primary_value_loss",
            "metric_kind": candidate_kind,
            "value_weight_mass": candidate_mass,
            "parent_value": parent,
            "candidate_value": candidate,
            "candidate_minus_parent": candidate - parent,
        }
        if (
            candidate_kind != parent_kind
            or not math.isclose(
                candidate_mass, parent_mass, rel_tol=0.0, abs_tol=1.0e-9
            )
        ):
            raise CampaignError("parent and candidate value objectives differ")
    else:
        candidate_projection = _reconciled_value_projection(
            functional,
            field="value_quality",
            where="candidate",
            require_emitted=True,
        )
        parent_projection = functional.get("parent_value_quality")
        if not isinstance(parent_projection, dict):
            raise CampaignError("paired value evidence has no parent projection")
        parent_projection = _validated_emitted_value_projection(
            parent_projection, where="parent"
        )
        if parent_functional is not None:
            exact_parent_projection = _reconciled_value_projection(
                parent_functional,
                field="value_quality",
                where="parent",
                require_emitted=False,
            )
            if parent_projection != exact_parent_projection:
                raise CampaignError(
                    "paired value parent projection contradicts exact parent metrics"
                )
    try:
        parent = float(paired["parent_value"])
        candidate = float(paired["candidate_value"])
        delta = float(paired["candidate_minus_parent"])
        mass = float(paired["value_weight_mass"])
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignError("paired parent value evidence is malformed") from error
    if (
        paired.get("schema_version") != PAIRED_PARENT_VALUE_SCHEMA
        or paired.get("selection_authority") is not True
        or paired.get("surface")
        != "same_holdout_same_objective_weights_fresh_exact_parent_forward"
        or paired.get("metric") != "primary_value_loss"
        or paired.get("metric_kind") != "scalar_mse"
        or not all(math.isfinite(value) for value in (parent, candidate, delta, mass))
        or mass <= 0.0
        or not math.isclose(
            delta, candidate - parent, rel_tol=1.0e-9, abs_tol=1.0e-12
        )
        or not math.isclose(
            candidate,
            float(candidate_projection["value"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            parent,
            float(parent_projection["value"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            mass,
            float(candidate_projection["value_weight_mass"]),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        or not math.isclose(
            mass,
            float(parent_projection["value_weight_mass"]),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise CampaignError("paired parent value evidence is inconsistent")
    if policy not in VALUE_GATE_POLICIES:
        raise CampaignError("value-quality gate policy is unsupported")
    passed = bool(delta <= max_absolute_regression)
    return {
        "policy": policy,
        "metric": "primary_value_loss",
        "metric_kind": "scalar_mse",
        "parent_value": parent,
        "candidate_value": candidate,
        "candidate_minus_parent": delta,
        "max_absolute_regression": max_absolute_regression,
        "passed": passed,
        "selection_admitted": bool(
            passed or policy == "diagnostic_record_only_allow_regression"
        ),
        "promotion_authority": bool(policy == VALUE_GATE_POLICY and passed),
        "evidence_schema_version": PAIRED_PARENT_VALUE_SCHEMA,
        "value_weight_mass": mass,
    }


def _fingerprint(
    plan_path: Path, plan: Mapping[str, Any], *, go: bool, device: str
) -> dict[str, Any]:
    _verify_inputs(plan)
    behavior_contract = plan.get("selection_contract", {}).get(
        "behavioral_competence_gate"
    )
    if behavior_contract != _behavioral_competence_gate_contract():
        raise CampaignError("Stage-C maritime behavioral gate contract drifted")
    receipt_path, receipt, report, report_payload = (
        _authenticate_completed_stage_c_dose(plan)
    )
    output_root = Path(str(plan["output_root"])) / "fingerprints"
    emitted_holdout = report_payload.get("validation_game_seed_manifest")
    if not isinstance(emitted_holdout, str) or not emitted_holdout:
        raise CampaignError(
            "completed Stage-C learner report has no emitted validation holdout"
        )
    validation_manifest = Path(emitted_holdout).expanduser()
    if not validation_manifest.is_absolute():
        validation_manifest = report.parent / validation_manifest
    if validation_manifest.is_symlink() or not validation_manifest.is_file():
        raise CampaignError("emitted validation holdout must be a regular file")
    validation_manifest = validation_manifest.resolve(strict=True)
    validation_payload = _load_json(
        validation_manifest, where="emitted Stage-C validation holdout"
    )[1]
    expected_input_manifest = Path(str(plan["inputs"]["validation_manifest"])).resolve(
        strict=True
    )
    if (
        validation_payload.get("schema_version") != "train-validation-game-seeds-v1"
        or validation_payload.get("a1_contract_sha256")
        != report_payload.get("a1_contract_sha256")
        or validation_payload.get("data") != report_payload.get("data")
        or validation_payload.get("data_fingerprint")
        != report_payload.get("data_fingerprint")
        or validation_payload.get("validation_game_seed_count")
        != report_payload.get("validation_game_seed_count")
        or validation_payload.get("validation_game_seed_set_sha256")
        != report_payload.get("validation_game_seed_set_sha256")
        or validation_payload.get("training_excluded_game_seed_count")
        != report_payload.get("training_excluded_game_seed_count")
        or validation_payload.get("training_excluded_game_seed_set_sha256")
        != report_payload.get("training_excluded_game_seed_set_sha256")
        or validation_payload.get("input_validation_game_seed_manifest")
        != str(expected_input_manifest)
        or validation_payload.get("input_validation_game_seed_manifest_sha256")
        != report_payload.get("input_validation_game_seed_manifest_sha256")
        or validation_payload.get("input_validation_game_seed_manifest_sha256")
        != _file_sha256(expected_input_manifest)
    ):
        raise CampaignError(
            "emitted validation holdout differs from the completed learner report"
        )
    validation_binding = {
        "path": str(validation_manifest),
        "file_sha256": _file_sha256(validation_manifest),
        "validation_game_seed_count": validation_payload["validation_game_seed_count"],
        "validation_game_seed_set_sha256": validation_payload[
            "validation_game_seed_set_sha256"
        ],
        "training_excluded_game_seed_set_sha256": validation_payload[
            "training_excluded_game_seed_set_sha256"
        ],
        "input_validation_game_seed_manifest": str(expected_input_manifest),
        "input_validation_game_seed_manifest_sha256": _file_sha256(
            expected_input_manifest
        ),
    }
    current_functional_binding = {
        "training_report": {
            "path": str(report),
            "sha256": _file_sha256(report),
        },
        "memmap": {
            "path": str(Path(str(plan["inputs"]["data"])).resolve(strict=True)),
            "fingerprint": report_payload["data_fingerprint"],
            "payload_inventory_sha256": plan["inputs"][
                "payload_inventory_sha256"
            ]
            if "payload_inventory_sha256" in plan["inputs"]
            else None,
        },
        "validation_manifest": {
            "path": str(validation_manifest),
            "sha256": _file_sha256(validation_manifest),
            "manifest_sha256": one_dose.train_bc._canonical_json_sha256(  # noqa: SLF001
                validation_payload
            ),
        },
        "validation_game_seed_set_sha256": validation_payload[
            "validation_game_seed_set_sha256"
        ],
    }
    authority = _load_json(
        Path(str(plan["inputs"]["independent_parent_authority"])),
        where="independent parent authority",
    )[1]
    parent = Path(
        str(authority["function_preserving_upgrade"]["upgraded_initializer"]["path"])
    ).resolve(strict=True)
    separate_parent = _parent_functional_artifact(output_root)
    separate_parent_path = None if separate_parent is None else separate_parent[0]
    separate_parent_payload = None if separate_parent is None else separate_parent[1]
    if separate_parent_payload is not None and (
        separate_parent_payload.get("inputs", {}).get("checkpoint", {}).get("sha256")
        != _file_sha256(parent)
    ):
        raise CampaignError("separate parent functional used the wrong checkpoint")
    policy_teacher_gap_objective = _expected_policy_teacher_gap_objective(
        plan["recipe"]
    )
    records = []
    commands = []
    terminal_checkpoint = Path(
        str(plan["expected_artifacts"]["terminal_checkpoint"])
    ).resolve(strict=True)
    for step in CHECKPOINT_STEPS:
        checkpoint = _checkpoint_path(plan, step).resolve(strict=True)
        checkpoint_binding = _authenticate_checkpoint_snapshot(
            report_payload,
            step=step,
            checkpoint=checkpoint,
            terminal_checkpoint=terminal_checkpoint,
        )
        functional_path = _functional_artifact_path(
            output_root,
            step,
            allow_separate_parent=separate_parent_payload is not None,
            expected_bindings={
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": _file_sha256(checkpoint),
                },
                **current_functional_binding,
            },
        )
        drift_path = output_root / f"step{step:04d}.drift.json"
        if drift_path.is_symlink():
            raise CampaignError(f"drift artifact must not be a symlink: {drift_path}")
        functional_command = [
            str(plan["inputs"]["python"]),
            str(REPO_ROOT / "tools" / "posthoc_teacher_gap_probe.py"),
            "--report",
            str(report),
            "--checkpoint",
            str(checkpoint),
            "--parent-checkpoint",
            str(parent),
            "--data",
            str(plan["inputs"]["data"]),
            "--validation-manifest",
            str(validation_manifest),
            "--device",
            device,
            "--output",
            str(functional_path),
        ]
        drift_command = [
            str(plan["inputs"]["python"]),
            str(REPO_ROOT / "tools" / "audit_checkpoint_layer_drift.py"),
            "--baseline",
            str(parent),
            "--candidate",
            str(checkpoint),
            "--output",
            str(drift_path),
        ]
        commands.append(
            {"step": step, "functional": functional_command, "drift": drift_command}
        )
        if not go:
            continue
        output_root.mkdir(parents=True, exist_ok=True)
        for command, artifact in (
            (functional_command, functional_path),
            (drift_command, drift_path),
        ):
            if artifact.is_file() and not artifact.is_symlink():
                continue
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                raise CampaignError(
                    f"checkpoint {step} fingerprint exited {result.returncode}"
                )
        functional = _load_json(functional_path, where=f"step {step} functional")[1]
        drift = _load_json(drift_path, where=f"step {step} drift")[1]
        functional_policy_teacher_gap_objective = (
            _require_policy_teacher_gap_objective(
                functional.get("policy_teacher_gap_objective"),
                expected=policy_teacher_gap_objective,
                where=f"checkpoint {step} functional report",
            )
        )
        functional_parent = functional.get("inputs", {}).get("parent_checkpoint")
        if not isinstance(functional_parent, dict) or functional_parent.get(
            "sha256"
        ) != _file_sha256(parent):
            raise CampaignError(
                f"checkpoint {step} functional report used the wrong parent"
            )
        fingerprint = functional.get("functional_dose_fingerprint")
        if not isinstance(fingerprint, dict):
            raise CampaignError(f"checkpoint {step} has no functional fingerprint")
        parent_kl = float(fingerprint["kl_parent_candidate_mean"])
        fresh_gap = _fresh_parent_teacher_gap(
            functional, parent_functional=separate_parent_payload
        )
        value_gate = _paired_value_quality(
            functional,
            parent_functional=separate_parent_payload,
            policy=str(
                plan["selection_contract"]["value_quality_gate"]["policy"]
            ),
            max_absolute_regression=float(
                plan["selection_contract"]["value_quality_gate"][
                    "max_absolute_regression"
                ]
            ),
        )
        behavior_gate = _evaluate_maritime_behavioral_competence_gate(
            functional, contract=behavior_contract
        )
        legacy_closure = fresh_gap["legacy_stored_prior_closure"]
        trunk = _trunk_relative_l2(drift)
        feature_signal = _checkpoint_feature_learning_signal(
            report_payload, step=step
        )
        eligible = bool(
            feature_signal["authenticated"] is True
            and parent_kl <= MAX_PARENT_KL
            and trunk <= MAX_TRUNK_RELATIVE_L2
            and value_gate["selection_admitted"]
            and behavior_gate["selection_admitted"]
        )
        records.append(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _file_sha256(checkpoint),
                "checkpoint_report_binding": checkpoint_binding,
                "parent_kl": parent_kl,
                "fresh_parent_target_kl_mean": fresh_gap["parent_target_kl_mean"],
                "candidate_target_kl_mean": fresh_gap["candidate_target_kl_mean"],
                "fresh_parent_teacher_gap_absolute_closure": fresh_gap[
                    "absolute_closure"
                ],
                "fresh_parent_teacher_gap_relative_closure": fresh_gap[
                    "relative_closure"
                ],
                "fresh_parent_evidence_schema_version": fresh_gap[
                    "evidence_schema_version"
                ],
                "policy_teacher_gap_objective": (
                    functional_policy_teacher_gap_objective
                ),
                "legacy_stored_generation_prior_teacher_gap_closure": legacy_closure,
                "stored_generation_prior_selection_authority": False,
                "trunk_relative_l2": trunk,
                "feature_learning_signal_authenticated": feature_signal[
                    "authenticated"
                ],
                "feature_learning_signal": feature_signal,
                "value_quality_gate": value_gate,
                "behavioral_competence_gate": behavior_gate,
                "eligible": eligible,
                "functional": {
                    "path": str(functional_path),
                    "file_sha256": _file_sha256(functional_path),
                },
                "drift": {
                    "path": str(drift_path),
                    "file_sha256": _file_sha256(drift_path),
                },
            }
        )
    if not go:
        return {
            "mode": "dry-run",
            "validation_holdout": validation_binding,
            "commands": commands,
        }
    winner = _select_fingerprint_winner(
        records,
        expected_objective=policy_teacher_gap_objective,
        expected_behavioral_competence_gate=behavior_contract,
    )
    payload: dict[str, Any] = {
        "schema_version": FINGERPRINT_SCHEMA,
        "campaign": {
            "path": str(plan_path),
            "file_sha256": _file_sha256(plan_path),
            "campaign_sha256": plan["campaign_sha256"],
        },
        "completed_dose": {
            "receipt": {
                "path": str(receipt_path.resolve(strict=True)),
                "file_sha256": _file_sha256(receipt_path),
                "receipt_sha256": receipt["receipt_sha256"],
            },
            "report": {
                "path": str(report),
                "file_sha256": _file_sha256(report),
            },
            "terminal_checkpoint": {
                "path": receipt["outputs"]["checkpoint"],
                "file_sha256": receipt["outputs"]["checkpoint_sha256"],
            },
            "feature_learning_signal_authenticated": True,
        },
        "metric_scope": "frozen_whole_game_validation_policy_active_multi_action_rows",
        "validation_holdout": validation_binding,
        "optimizer_batch_kl_used_as_trust_authority": False,
        "stored_generation_prior_used_as_selection_authority": False,
        "selection_objective": (
            "maritime_behavior_prefilter_then_minimum_update_within_parent_"
            "kl_and_trunk_drift_budgets_then_h2h_final_ranking"
        ),
        "teacher_gap_closure_ranking_authority": False,
        "policy_teacher_gap_objective": policy_teacher_gap_objective,
        "value_quality_gate": copy.deepcopy(
            plan["selection_contract"]["value_quality_gate"]
        ),
        "behavioral_competence_gate": copy.deepcopy(behavior_contract),
        "output": str(_fresh_parent_fingerprint_path(plan)),
        "checkpoints": records,
        "winner": winner,
        "formal_result": (
            "posthoc_in_budget_candidate_requires_playing_evaluation"
            if winner is not None
            else "no_value_safe_checkpoint_within_posthoc_trust_budget"
        ),
    }
    if separate_parent_path is not None:
        payload["separate_exact_parent_evidence"] = {
            "path": str(separate_parent_path),
            "file_sha256": _file_sha256(separate_parent_path),
            "selection_authority": True,
            "surface": "same_holdout_same_targets_fresh_exact_parent_forward",
        }
    legacy_fingerprint = _legacy_fingerprint_path(plan)
    if legacy_fingerprint.is_symlink():
        raise CampaignError("legacy fingerprint must not be a symlink")
    if legacy_fingerprint.is_file():
        legacy_payload = _load_json(
            legacy_fingerprint, where="superseded legacy Stage-C fingerprint"
        )[1]
        payload["superseded_legacy_fingerprint"] = {
            "path": str(legacy_fingerprint.resolve(strict=True)),
            "file_sha256": _file_sha256(legacy_fingerprint),
            "schema_version": legacy_payload.get("schema_version"),
            "selection_authority": False,
            "reason": "used_generation_time_stored_prior_as_teacher_gap_baseline",
        }
    payload["fingerprint_sha256"] = _value_sha256(payload)
    _write_json(_fresh_parent_fingerprint_path(plan), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--arm", required=True, choices=sorted(ARMS))
    plan.add_argument("--overlay-admission", required=True, type=Path)
    plan.add_argument("--lock", required=True, type=Path)
    plan.add_argument("--architecture-upgrade-receipt", required=True, type=Path)
    plan.add_argument("--ddp-canary-receipt", required=True, type=Path)
    plan.add_argument("--python", required=True, type=Path)
    plan.add_argument("--output-root", required=True, type=Path)
    plan.add_argument("--write", required=True, type=Path)
    plan.add_argument(
        "--value-gate-policy",
        choices=sorted(VALUE_GATE_POLICIES),
        default=VALUE_GATE_POLICY,
    )
    plan.add_argument(
        "--max-value-mse-regression",
        type=float,
        default=MAX_VALUE_MSE_REGRESSION,
    )
    run = commands.add_parser("run")
    run.add_argument("--campaign", required=True, type=Path)
    run.add_argument("--go", action="store_true")
    fingerprint = commands.add_parser("fingerprint")
    fingerprint.add_argument("--campaign", required=True, type=Path)
    fingerprint.add_argument("--device", default="cuda:0")
    fingerprint.add_argument("--go", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            result = _plan(args)
            _write_json(args.write, result)
        else:
            campaign_path, campaign = _load_plan(args.campaign)
            result = (
                _run(campaign, go=args.go)
                if args.command == "run"
                else _fingerprint(
                    campaign_path,
                    campaign,
                    go=args.go,
                    device=args.device,
                )
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (CampaignError, overlay.OverlayError, OSError, ValueError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
