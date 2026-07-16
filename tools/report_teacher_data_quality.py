from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from train_bc import load_teacher_data, teacher_data_quality


SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SOURCE_MANIFEST_BINDING_SCHEMA = "teacher-source-manifest-binding-v1"
TOOL_PROVENANCE_SCHEMA = "teacher-tool-provenance-v1"
REQUIRED_SOURCE_FEATURE_FILES = frozenset(
    {
        "src/catan_zero/rl/self_play.py",
        "src/catan_zero/rl/action_features.py",
        "src/catan_zero/rl/xdim_lite_policy.py",
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report teacher shard quality.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--track",
        default="",
        help="Optional metadata only; accepted so plan/runbook commands stay valid.",
    )
    parser.add_argument(
        "--vps-to-win",
        type=int,
        default=0,
        help="Optional metadata only; accepted so plan/runbook commands stay valid.",
    )
    parser.add_argument(
        "--strict-35m-teacher",
        action="store_true",
        help=(
            "Apply production gates for the 35M pre-PPO teacher corpus: complete "
            "outcomes/VPs, low forced-action rate after curation, strong soft "
            "labels, and true AB-root score provenance for any catanatron_ab* teacher."
        ),
    )
    parser.add_argument(
        "--production-35m-teacher",
        action="store_true",
        help=(
            "Apply high-volume production gates for the next 35M BC run. This is "
            "stricter than --strict-35m-teacher: it requires millions of clean "
            "rows, substantial AB4/AB5/search/JSettlers-style coverage, and "
            "minimum soft-score coverage over legal candidates."
        ),
    )
    parser.add_argument(
        "--min-soft-policy-fraction",
        type=float,
        default=0.0,
        help="Fail if fewer rows have target_policy soft labels.",
    )
    parser.add_argument(
        "--min-soft-score-fraction",
        type=float,
        default=0.0,
        help="Fail if fewer rows have finite target_scores.",
    )
    parser.add_argument(
        "--soft-target-temperature",
        type=float,
        default=0.7,
        help="Temperature used when estimating effective soft-distillation rows.",
    )
    parser.add_argument(
        "--soft-target-source",
        choices=("prefer_policy", "prefer_scores", "policy", "scores"),
        default="prefer_scores",
        help="Soft-label source used when estimating effective soft-distillation rows.",
    )
    parser.add_argument(
        "--soft-target-min-legal-coverage",
        type=float,
        default=0.5,
        help=(
            "Minimum legal-action coverage used to count a row as effective "
            "soft distillation. Must match train_bc.py."
        ),
    )
    parser.add_argument(
        "--min-effective-soft-distillation-fraction",
        type=float,
        default=0.0,
        help=(
            "Fail if too few policy-active rows will actually use soft "
            "distillation after one-hot and low-coverage rows fall back to hard CE."
        ),
    )
    parser.add_argument(
        "--min-q-score-rows-ge2-fraction",
        type=float,
        default=0.0,
        help="Fail if too few rows have at least two finite scored legal actions.",
    )
    parser.add_argument(
        "--min-selected-action-score-fraction",
        type=float,
        default=0.0,
        help="Fail if too few rows have a finite target_score for the teacher-selected action.",
    )
    parser.add_argument(
        "--min-usable-q-score-rows-ge2-fraction",
        type=float,
        default=0.0,
        help="Fail if too few rows have Q-usable scored rows after q-skip prefixes.",
    )
    parser.add_argument(
        "--q-skip-teacher-prefixes",
        default="catanatron_ab",
        help="Comma-separated teacher prefixes ignored for usable Q-score gates.",
    )
    parser.add_argument(
        "--min-outcome-fraction",
        type=float,
        default=0.0,
        help="Fail if too few rows have non-truncated winner labels.",
    )
    parser.add_argument(
        "--min-clean-terminal-outcome-fraction",
        type=float,
        default=0.0,
        help="Fail if too few rows have non-truncated terminal winner labels.",
    )
    parser.add_argument(
        "--min-final-public-vp-fraction",
        type=float,
        default=0.0,
        help="Fail if too few rows have final public VP targets.",
    )
    parser.add_argument(
        "--min-final-actual-vp-fraction",
        type=float,
        default=0.0,
        help="Fail if too few rows have final actual VP targets.",
    )
    parser.add_argument(
        "--max-forced-action-fraction",
        type=float,
        default=1.0,
        help="Fail if one-legal-action rows exceed this fraction.",
    )
    parser.add_argument(
        "--max-truncated-fraction",
        type=float,
        default=1.0,
        help="Fail if truncated/stuck rows exceed this fraction.",
    )
    parser.add_argument(
        "--min-teacher-samples",
        default="",
        help="Comma-separated teacher minimums, e.g. catanatron_ab4=10000,value_rollout_search=10000.",
    )
    parser.add_argument(
        "--min-phase-samples",
        default="",
        help="Comma-separated phase minimums, e.g. robber=5000,initial_build=5000.",
    )
    parser.add_argument(
        "--max-invalid-teacher-actions",
        type=int,
        default=0,
        help="Fail if invalid teacher labels exceed this count.",
    )
    parser.add_argument(
        "--min-soft-policy-by-teacher",
        default="",
        help="Comma-separated teacher minimum soft-policy fractions, e.g. catanatron_ab5=0.95.",
    )
    parser.add_argument(
        "--min-soft-score-by-teacher",
        default="",
        help="Comma-separated teacher minimum soft-score fractions, e.g. value_rollout_search=0.95.",
    )
    parser.add_argument(
        "--min-usable-q-score-by-teacher",
        default="",
        help="Comma-separated teacher minimum usable Q-score fractions after q-skip prefixes.",
    )
    parser.add_argument(
        "--min-ab-root-score-fraction",
        type=float,
        default=0.0,
        help="Fail if too few rows have finite target_scores marked target_score_source=ab_root.",
    )
    parser.add_argument(
        "--min-ab-root-score-by-teacher",
        default="",
        help="Comma-separated teacher minimum ab_root score fractions, e.g. catanatron_ab4=0.95.",
    )
    parser.add_argument(
        "--min-clean-outcome-by-teacher",
        default="",
        help="Comma-separated teacher minimum clean terminal outcome fractions.",
    )
    parser.add_argument(
        "--min-soft-policy-by-phase",
        default="",
        help="Comma-separated phase minimum soft-policy fractions, e.g. robber=0.95.",
    )
    parser.add_argument(
        "--min-usable-q-score-by-phase",
        default="",
        help="Comma-separated phase minimum usable Q-score fractions after q-skip prefixes.",
    )
    parser.add_argument(
        "--max-forced-by-phase",
        default="",
        help="Comma-separated phase maximum forced-action fractions, e.g. robber=0.10.",
    )
    args = parser.parse_args()

    data = load_teacher_data(Path(args.data))
    q_skip_teacher_prefixes = _parse_prefixes(args.q_skip_teacher_prefixes)
    report = teacher_data_quality(
        data,
        q_skip_teacher_prefixes=q_skip_teacher_prefixes,
        soft_target_temperature=float(args.soft_target_temperature),
        soft_target_source=str(args.soft_target_source),
        soft_target_min_legal_coverage=float(args.soft_target_min_legal_coverage),
    )
    input_metadata = _input_metadata(Path(args.data))
    report["input_metadata"] = input_metadata
    if args.track:
        report["track"] = args.track
    if args.vps_to_win:
        report["vps_to_win"] = int(args.vps_to_win)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")

    failures = []
    if args.production_35m_teacher:
        args.strict_35m_teacher = True
    if args.strict_35m_teacher:
        _apply_strict_35m_defaults(args)
        _check_manifest_metadata(failures, args, input_metadata)

    _check_max(
        failures,
        "invalid teacher actions",
        int(report["invalid_teacher_actions"]),
        int(args.max_invalid_teacher_actions),
    )
    _check_min_fraction(
        failures,
        "soft policy fraction",
        report["soft_policy_fraction"],
        args.min_soft_policy_fraction,
    )
    _check_min_fraction(
        failures,
        "soft score fraction",
        report["soft_score_fraction"],
        args.min_soft_score_fraction,
    )
    _check_min_fraction(
        failures,
        "policy-active effective soft distillation fraction",
        _policy_active_effective_soft_fraction(report),
        args.min_effective_soft_distillation_fraction,
    )
    _check_min_fraction(
        failures,
        "policy-active q score rows >=2 fraction",
        _policy_active_q_fraction(report),
        args.min_q_score_rows_ge2_fraction,
    )
    _check_min_fraction(
        failures,
        "policy-active usable q score rows >=2 fraction",
        _policy_active_usable_q_fraction(report),
        args.min_usable_q_score_rows_ge2_fraction,
    )
    _check_min_fraction(
        failures,
        "selected action score fraction",
        report.get("selected_action_score_fraction", 0.0),
        getattr(args, "min_selected_action_score_fraction", 0.0),
    )
    _check_min_fraction(
        failures,
        "ab_root score fraction",
        report.get("ab_root_score_fraction", 0.0),
        args.min_ab_root_score_fraction,
    )
    _check_min_fraction(
        failures,
        "outcome fraction",
        report["outcome_fraction"],
        args.min_outcome_fraction,
    )
    _check_min_fraction(
        failures,
        "clean terminal outcome fraction",
        report["clean_terminal_outcome_fraction"],
        args.min_clean_terminal_outcome_fraction,
    )
    _check_min_fraction(
        failures,
        "final public VP fraction",
        report["final_public_vp_fraction"],
        args.min_final_public_vp_fraction,
    )
    _check_min_fraction(
        failures,
        "final actual VP fraction",
        report["final_actual_vp_fraction"],
        args.min_final_actual_vp_fraction,
    )
    forced_metric = float(
        report.get(
            "policy_effective_forced_action_fraction",
            report["forced_action_fraction"],
        )
    )
    _check_max_fraction(
        failures,
        "policy-effective forced action fraction",
        forced_metric,
        args.max_forced_action_fraction,
    )
    _check_max_fraction(
        failures,
        "truncated fraction",
        report["truncated_fraction"],
        args.max_truncated_fraction,
    )
    for teacher, minimum in _parse_minimums(args.min_teacher_samples).items():
        actual = int(report.get("teacher_counts", {}).get(teacher, 0))
        _check_min_count(failures, f"teacher {teacher} samples", actual, minimum)
    for phase, minimum in _parse_minimums(args.min_phase_samples).items():
        actual = int(report.get("phase_counts", {}).get(phase, 0))
        _check_min_count(failures, f"phase {phase} samples", actual, minimum)
    _check_group_min_fraction(
        failures,
        report,
        group="by_teacher",
        thresholds=_parse_float_thresholds(args.min_soft_policy_by_teacher),
        metric="soft_policy_fraction",
        label="teacher soft policy fraction",
    )
    _check_group_min_fraction(
        failures,
        report,
        group="by_teacher",
        thresholds=_parse_float_thresholds(args.min_soft_score_by_teacher),
        metric="soft_score_fraction",
        label="teacher soft score fraction",
    )
    _check_group_min_fraction(
        failures,
        report,
        group="by_teacher",
        thresholds=_parse_float_thresholds(args.min_usable_q_score_by_teacher),
        metric="usable_q_score_rows_ge2_fraction",
        label="teacher usable q score rows >=2 fraction",
    )
    _check_group_min_fraction(
        failures,
        report,
        group="by_teacher",
        thresholds=_parse_float_thresholds(args.min_ab_root_score_by_teacher),
        metric="ab_root_score_fraction",
        label="teacher ab_root score fraction",
    )
    _check_group_min_fraction(
        failures,
        report,
        group="by_teacher",
        thresholds=_parse_float_thresholds(args.min_clean_outcome_by_teacher),
        metric="clean_terminal_outcome_fraction",
        label="teacher clean terminal outcome fraction",
    )
    _check_group_min_fraction(
        failures,
        report,
        group="by_phase",
        thresholds=_parse_float_thresholds(args.min_soft_policy_by_phase),
        metric="soft_policy_fraction",
        label="phase soft policy fraction",
    )
    _check_group_min_fraction(
        failures,
        report,
        group="by_phase",
        thresholds=_parse_float_thresholds(args.min_usable_q_score_by_phase),
        metric="usable_q_score_rows_ge2_fraction",
        label="phase usable q score rows >=2 fraction",
    )
    _check_group_max_fraction(
        failures,
        report,
        group="by_phase",
        thresholds=_parse_float_thresholds(args.max_forced_by_phase),
        metric="forced_action_fraction",
        label="phase forced action fraction",
    )
    if args.strict_35m_teacher:
        _check_strict_35m_teacher_gates(failures, report)
    if args.production_35m_teacher:
        _check_production_35m_teacher_gates(failures, report)
    if failures:
        raise SystemExit("teacher data quality failed:\n" + "\n".join(f"- {failure}" for failure in failures))


def _check_min_fraction(failures: list[str], label: str, actual: float, minimum: float) -> None:
    if float(actual) < float(minimum):
        failures.append(f"{label} {float(actual):.6f} < {float(minimum):.6f}")


def _apply_strict_35m_defaults(args: argparse.Namespace) -> None:
    args.min_soft_score_fraction = max(float(args.min_soft_score_fraction), 0.99)
    args.min_q_score_rows_ge2_fraction = max(
        float(args.min_q_score_rows_ge2_fraction),
        0.50,
    )
    args.min_clean_terminal_outcome_fraction = max(
        float(args.min_clean_terminal_outcome_fraction),
        0.99,
    )
    # BC value targets prefer final_actual_vps and only fall back to public VPs.
    # Do not reject complete actual-VP corpora just because hidden/private VPs
    # were not also exported as public labels.
    args.min_final_actual_vp_fraction = max(float(args.min_final_actual_vp_fraction), 0.99)
    args.max_forced_action_fraction = min(float(args.max_forced_action_fraction), 0.05)
    args.max_truncated_fraction = min(float(args.max_truncated_fraction), 0.0)
    args.min_effective_soft_distillation_fraction = max(
        float(getattr(args, "min_effective_soft_distillation_fraction", 0.0)),
        0.35,
    )


def _check_max_fraction(failures: list[str], label: str, actual: float, maximum: float) -> None:
    if float(actual) > float(maximum):
        failures.append(f"{label} {float(actual):.6f} > {float(maximum):.6f}")


def _check_max(failures: list[str], label: str, actual: int, maximum: int) -> None:
    if int(actual) > int(maximum):
        failures.append(f"{label} {int(actual)} > {int(maximum)}")


def _check_min_count(failures: list[str], label: str, actual: int, minimum: int) -> None:
    if int(actual) < int(minimum):
        failures.append(f"{label} {int(actual)} < {int(minimum)}")


def _policy_active_q_fraction(metrics: dict[str, Any]) -> float:
    return float(
        metrics.get(
            "q_score_rows_ge2_policy_active_fraction",
            metrics.get("q_score_rows_ge2_fraction", 0.0),
        )
    )


def _policy_active_usable_q_fraction(metrics: dict[str, Any]) -> float:
    return float(
        metrics.get(
            "usable_q_score_rows_ge2_policy_active_fraction",
            metrics.get("usable_q_score_rows_ge2_fraction", 0.0),
        )
    )


def _policy_active_effective_soft_fraction(metrics: dict[str, Any]) -> float:
    return float(
        metrics.get(
            "policy_active_effective_soft_distillation_fraction",
            metrics.get("effective_soft_distillation_fraction", 0.0),
        )
    )


def _check_group_min_fraction(
    failures: list[str],
    report: dict,
    *,
    group: str,
    thresholds: dict[str, float],
    metric: str,
    label: str,
) -> None:
    values = report.get(group, {})
    for key, minimum in thresholds.items():
        if key not in values:
            failures.append(f"{label} {key} missing from {group}")
            continue
        actual = float(values[key].get(metric, 0.0))
        if actual < float(minimum):
            failures.append(f"{label} {key} {actual:.6f} < {float(minimum):.6f}")


def _check_group_max_fraction(
    failures: list[str],
    report: dict,
    *,
    group: str,
    thresholds: dict[str, float],
    metric: str,
    label: str,
) -> None:
    values = report.get(group, {})
    for key, maximum in thresholds.items():
        if key not in values:
            failures.append(f"{label} {key} missing from {group}")
            continue
        actual = float(values[key].get(metric, 0.0))
        if actual > float(maximum):
            failures.append(f"{label} {key} {actual:.6f} > {float(maximum):.6f}")


def _check_strict_35m_teacher_gates(failures: list[str], report: dict) -> None:
    by_teacher = dict(report.get("by_teacher", {}))
    by_phase = dict(report.get("by_phase", {}))
    teacher_counts = dict(report.get("teacher_counts", {}))
    total_samples = int(report.get("samples", 0))
    if total_samples < 250_000:
        failures.append(
            f"strict 35M teacher gate requires at least 250000 samples, got {total_samples}"
        )
    policy_active_rows = int(report.get("policy_active_rows", 0))
    if policy_active_rows < 150_000:
        failures.append(
            "strict 35M teacher gate requires at least 150000 policy-active "
            f"samples, got {policy_active_rows}"
        )
    if int(report.get("unflagged_final_actual_vp_rows", 0)) > 0:
        failures.append(
            "strict 35M teacher gate found final_actual_vps rows without "
            f"has_final_actual_vps=true: {int(report.get('unflagged_final_actual_vp_rows', 0))}"
        )
    if (
        int(report.get("unflagged_final_public_vp_rows", 0)) > 0
        and float(report.get("final_actual_vp_fraction", 0.0)) < 0.99
    ):
        failures.append(
            "strict 35M teacher gate found final_public_vps rows without "
            f"has_final_public_vps=true: {int(report.get('unflagged_final_public_vp_rows', 0))}"
        )

    required_teachers = {
        "catanatron_ab4": 10_000,
        "catanatron_ab5": 10_000,
        "value_rollout_search": 10_000,
        "jsettlers_lite": 10_000,
    }
    for teacher, minimum in required_teachers.items():
        actual = int(teacher_counts.get(teacher, 0))
        if actual < minimum:
            failures.append(
                f"strict 35M teacher gate requires teacher {teacher} >= {minimum} samples, got {actual}"
            )
        metrics = dict(by_teacher.get(teacher, {}))
        policy_active = int(metrics.get("policy_active_rows", 0))
        if policy_active < minimum:
            failures.append(
                "strict 35M teacher gate requires teacher "
                f"{teacher} >= {minimum} policy-active samples, got {policy_active}"
            )

    required_phases = {
        "initial_build": 5_000,
        "main_turn": 50_000,
        "robber": 5_000,
        "discard": 5_000,
    }
    phase_counts = dict(report.get("phase_counts", {}))
    for phase, minimum in required_phases.items():
        actual = int(phase_counts.get(phase, 0))
        if actual < minimum:
            failures.append(
                f"strict 35M teacher gate requires phase {phase} >= {minimum} samples, got {actual}"
            )
        metrics = dict(by_phase.get(phase, {}))
        policy_active = int(metrics.get("policy_active_rows", 0))
        if policy_active < minimum:
            failures.append(
                "strict 35M teacher gate requires phase "
                f"{phase} >= {minimum} policy-active samples, got {policy_active}"
            )

    ab_teachers = sorted(
        teacher
        for teacher in teacher_counts
        if str(teacher).startswith("catanatron_ab")
    )
    if not ab_teachers:
        failures.append("strict 35M teacher gate requires at least one catanatron_ab* teacher")

    for teacher, count in sorted(teacher_counts.items()):
        metrics = dict(by_teacher.get(teacher, {}))
        if not metrics:
            failures.append(f"strict 35M teacher gate missing by_teacher metrics for {teacher}")
            continue
        _check_min_fraction(
            failures,
            f"strict teacher {teacher} policy-active soft score fraction",
            float(
                metrics.get(
                    "policy_active_soft_score_fraction",
                    metrics.get("soft_score_fraction", 0.0),
                )
            ),
            0.95,
        )
        _check_min_fraction(
            failures,
            f"strict teacher {teacher} clean terminal outcome fraction",
            float(metrics.get("clean_terminal_outcome_fraction", 0.0)),
            0.99,
        )
        _check_min_fraction(
            failures,
            f"strict teacher {teacher} final actual VP fraction",
            float(metrics.get("final_actual_vp_fraction", 0.0)),
            0.99,
        )
        _check_min_fraction(
            failures,
            f"strict teacher {teacher} policy-active effective soft distillation fraction",
            _policy_active_effective_soft_fraction(metrics),
            0.35,
        )
        if int(count) <= 0:
            failures.append(f"strict teacher {teacher} has no samples")

    for teacher in ab_teachers:
        metrics = dict(by_teacher.get(teacher, {}))
        _check_min_fraction(
            failures,
            f"strict AB teacher {teacher} policy-active ab_root score fraction",
            float(
                metrics.get(
                    "policy_active_ab_root_score_fraction",
                    metrics.get("ab_root_score_fraction", 0.0),
                )
            ),
            0.95,
        )
        _check_min_fraction(
            failures,
            f"strict AB teacher {teacher} policy-active soft policy fraction",
            float(
                metrics.get(
                    "policy_active_soft_policy_fraction",
                    metrics.get("soft_policy_fraction", 0.0),
                )
            ),
            0.95,
        )

    for teacher in ("catanatron_ab4", "catanatron_ab5", "value_rollout_search"):
        metrics = dict(by_teacher.get(teacher, {}))
        if not metrics:
            continue
        _check_min_fraction(
            failures,
            f"strict teacher {teacher} policy-active q score rows >=2 fraction",
            _policy_active_q_fraction(metrics),
            0.50,
        )

    for phase in ("initial_build", "main_turn", "robber", "discard"):
        metrics = dict(by_phase.get(phase, {}))
        if not metrics:
            continue
        _check_max_fraction(
            failures,
            f"strict phase {phase} policy-effective forced action fraction",
            float(
                metrics.get(
                    "policy_effective_forced_action_fraction",
                    metrics.get("forced_action_fraction", 1.0),
                )
            ),
            0.05,
        )


def _check_production_35m_teacher_gates(failures: list[str], report: dict) -> None:
    by_teacher = dict(report.get("by_teacher", {}))
    by_phase = dict(report.get("by_phase", {}))
    teacher_counts = dict(report.get("teacher_counts", {}))
    phase_counts = dict(report.get("phase_counts", {}))
    total_samples = int(report.get("samples", 0))

    if total_samples < 2_000_000:
        failures.append(
            f"production 35M teacher gate requires at least 2000000 samples, got {total_samples}"
        )
    policy_active_rows = int(report.get("policy_active_rows", 0))
    if policy_active_rows < 1_200_000:
        failures.append(
            "production 35M teacher gate requires at least 1200000 "
            f"policy-active samples, got {policy_active_rows}"
        )

    required_teachers = {
        "catanatron_ab5": 250_000,
        "catanatron_ab4": 250_000,
        "value_rollout_search": 250_000,
        "catanatron_value": 200_000,
        "jsettlers_lite": 150_000,
        "catanatron_ab3": 100_000,
    }
    for teacher, minimum in required_teachers.items():
        actual = int(teacher_counts.get(teacher, 0))
        if actual < minimum:
            failures.append(
                f"production 35M teacher gate requires teacher {teacher} >= {minimum} samples, got {actual}"
            )
        metrics = dict(by_teacher.get(teacher, {}))
        policy_active = int(metrics.get("policy_active_rows", 0))
        if policy_active < minimum:
            failures.append(
                "production 35M teacher gate requires teacher "
                f"{teacher} >= {minimum} policy-active samples, got {policy_active}"
            )

    required_phases = {
        "initial_build": 25_000,
        "main_turn": 500_000,
        "robber": 25_000,
        "discard": 10_000,
    }
    for phase, minimum in required_phases.items():
        actual = int(phase_counts.get(phase, 0))
        if actual < minimum:
            failures.append(
                f"production 35M teacher gate requires phase {phase} >= {minimum} samples, got {actual}"
            )
        metrics = dict(by_phase.get(phase, {}))
        policy_active = int(metrics.get("policy_active_rows", 0))
        if policy_active < minimum:
            failures.append(
                "production 35M teacher gate requires phase "
                f"{phase} >= {minimum} policy-active samples, got {policy_active}"
            )

    _check_min_fraction(
        failures,
        "production soft score legal coverage mean",
        float(report.get("soft_score_legal_coverage_mean", 0.0)),
        0.40,
    )
    _check_min_fraction(
        failures,
        "production soft policy legal coverage mean",
        float(report.get("soft_policy_legal_coverage_mean", 0.0)),
        0.40,
    )
    _check_min_fraction(
        failures,
        "production selected action score fraction",
        float(report.get("selected_action_score_fraction", 0.0)),
        0.90,
    )
    _check_min_fraction(
        failures,
        "production policy-active effective soft distillation fraction",
        _policy_active_effective_soft_fraction(report),
        0.55,
    )

    for teacher in ("catanatron_ab5", "catanatron_ab4", "value_rollout_search"):
        metrics = dict(by_teacher.get(teacher, {}))
        if not metrics:
            continue
        _check_min_fraction(
            failures,
            f"production teacher {teacher} policy-active q score rows >=2 fraction",
            _policy_active_q_fraction(metrics),
            0.75,
        )
        _check_min_fraction(
            failures,
            f"production teacher {teacher} soft score legal coverage mean",
            float(metrics.get("soft_score_legal_coverage_mean", 0.0)),
            0.50,
        )
        _check_min_fraction(
            failures,
            f"production teacher {teacher} selected action score fraction",
            float(metrics.get("selected_action_score_fraction", 0.0)),
            0.95,
        )
        _check_min_fraction(
            failures,
            f"production teacher {teacher} policy-active effective soft distillation fraction",
            _policy_active_effective_soft_fraction(metrics),
            0.55,
        )

    for phase in ("initial_build", "main_turn", "robber", "discard"):
        metrics = dict(by_phase.get(phase, {}))
        if not metrics:
            continue
        _check_min_fraction(
            failures,
            f"production phase {phase} soft score legal coverage mean",
            float(metrics.get("soft_score_legal_coverage_mean", 0.0)),
            0.25,
        )


def _input_metadata(data_path: Path) -> dict[str, Any]:
    manifests: list[dict[str, Any]] = []
    for candidate in (data_path / "manifest.json", data_path / "curation_report.json"):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception as error:
            manifests.append({"path": str(candidate), "error": str(error)})
            continue
        manifests.append({"path": str(candidate), "payload": payload})

    tracks: set[str] = set()
    vps_to_win: set[int] = set()
    mixed_seats: set[bool] = set()
    mixed_seat_modes: set[str] = set()
    graph_history_features: set[bool] = set()
    provenance_hashes: dict[str, set[str]] = {}
    source_provenance_hashes: dict[str, set[str]] = {}
    source_provenance_errors: list[str] = []
    for manifest in manifests:
        payload = manifest.get("payload")
        if payload is None:
            continue
        _collect_source_provenance_values(
            payload,
            source_provenance_hashes,
            source_provenance_errors,
            manifest_path=Path(str(manifest["path"])),
        )
        _collect_manifest_values(
            payload,
            tracks=tracks,
            vps_to_win=vps_to_win,
            mixed_seats=mixed_seats,
            mixed_seat_modes=mixed_seat_modes,
            graph_history_features=graph_history_features,
            provenance_hashes=provenance_hashes,
        )
    return {
        "manifest_count": len(manifests),
        "tracks": sorted(tracks),
        "vps_to_win": sorted(vps_to_win),
        "mixed_seats": sorted(mixed_seats),
        "mixed_seat_modes": sorted(mixed_seat_modes),
        "graph_history_features": sorted(graph_history_features),
        "provenance_hashes": {
            path: sorted(values)
            for path, values in sorted(provenance_hashes.items())
        },
        "source_provenance_hashes": {
            path: sorted(values)
            for path, values in sorted(source_provenance_hashes.items())
        },
        "source_provenance_errors": source_provenance_errors,
        "manifest_paths": [str(item.get("path", "")) for item in manifests],
        "manifest_errors": [
            {"path": str(item.get("path", "")), "error": str(item.get("error", ""))}
            for item in manifests
            if item.get("error")
        ],
    }


def _collect_manifest_values(
    value: Any,
    *,
    tracks: set[str],
    vps_to_win: set[int],
    mixed_seats: set[bool],
    mixed_seat_modes: set[str],
    graph_history_features: set[bool],
    provenance_hashes: dict[str, set[str]],
) -> None:
    if isinstance(value, dict):
        file_sha256 = value.get("file_sha256")
        if isinstance(file_sha256, dict):
            for path, digest in file_sha256.items():
                if isinstance(path, str) and isinstance(digest, str) and digest:
                    provenance_hashes.setdefault(path, set()).add(digest)
        raw_track = value.get("track")
        if isinstance(raw_track, str) and raw_track:
            tracks.add(raw_track)
        raw_vps = value.get("vps_to_win")
        if raw_vps not in (None, ""):
            try:
                vps_to_win.add(int(raw_vps))
            except (TypeError, ValueError):
                pass
        raw_mixed = value.get("mixed_seats")
        if isinstance(raw_mixed, bool):
            mixed_seats.add(bool(raw_mixed))
        raw_mixed_mode = value.get("mixed_seat_mode")
        if isinstance(raw_mixed_mode, str) and raw_mixed_mode:
            mixed_seat_modes.add(raw_mixed_mode)
        raw_graph_history = value.get("graph_history_features")
        if isinstance(raw_graph_history, bool):
            graph_history_features.add(bool(raw_graph_history))
        for child in value.values():
            _collect_manifest_values(
                child,
                tracks=tracks,
                vps_to_win=vps_to_win,
                mixed_seats=mixed_seats,
                mixed_seat_modes=mixed_seat_modes,
                graph_history_features=graph_history_features,
                provenance_hashes=provenance_hashes,
            )
    elif isinstance(value, list):
        for child in value:
            _collect_manifest_values(
                child,
                tracks=tracks,
                vps_to_win=vps_to_win,
                mixed_seats=mixed_seats,
                mixed_seat_modes=mixed_seat_modes,
                graph_history_features=graph_history_features,
                provenance_hashes=provenance_hashes,
            )


def _collect_source_provenance_values(
    payload: Any,
    source_provenance_hashes: dict[str, set[str]],
    errors: list[str],
    *,
    manifest_path: Path,
) -> None:
    if not isinstance(payload, dict):
        errors.append(f"teacher manifest is not a JSON object: {manifest_path}")
        return
    seen: set[Path] = set()
    _collect_authenticated_manifest_chain(
        payload,
        source_provenance_hashes,
        errors,
        manifest_path=manifest_path,
        seen=seen,
    )


def _collect_authenticated_manifest_chain(
    payload: dict[str, Any],
    hashes: dict[str, set[str]],
    errors: list[str],
    *,
    manifest_path: Path,
    seen: set[Path],
) -> None:
    named_files = _collect_named_tool_provenance(
        payload.get("tool_provenance"),
        hashes,
        errors,
        manifest_path=manifest_path,
    )
    follow_source_provenance = not REQUIRED_SOURCE_FEATURE_FILES.issubset(named_files)
    inputs = payload.get("input_manifests")
    if inputs is None:
        return
    if not isinstance(inputs, list):
        errors.append(f"input_manifests must be a list: {manifest_path}")
        return
    for index, item in enumerate(inputs):
        if not isinstance(item, dict):
            errors.append(f"input_manifests[{index}] must be an object: {manifest_path}")
            continue
        binding = item.get("source_manifest")
        modal = item.get("modal_parts_summary")
        if binding is not None:
            _collect_bound_source_manifest(
                binding,
                hashes,
                errors,
                owner=f"{manifest_path}:input_manifests[{index}]",
                seen=seen,
                follow_chain=follow_source_provenance,
            )
        elif not isinstance(modal, dict):
            errors.append(
                "input lineage lacks source_manifest binding: "
                f"{manifest_path}:input_manifests[{index}]"
            )
        if isinstance(modal, dict):
            part_manifests = modal.get("part_manifests")
            if not isinstance(part_manifests, list) or not part_manifests:
                errors.append(
                    "modal source lineage requires non-empty part_manifests: "
                    f"{manifest_path}:input_manifests[{index}]"
                )
                continue
            for part_index, part_binding in enumerate(part_manifests):
                _collect_bound_source_manifest(
                    part_binding,
                    hashes,
                    errors,
                    owner=(
                        f"{manifest_path}:input_manifests[{index}]"
                        f".part_manifests[{part_index}]"
                    ),
                    seen=seen,
                    follow_chain=follow_source_provenance,
                )


def _collect_bound_source_manifest(
    binding: Any,
    hashes: dict[str, set[str]],
    errors: list[str],
    *,
    owner: str,
    seen: set[Path],
    follow_chain: bool,
) -> None:
    expected_fields = {"schema_version", "path", "file_sha256"}
    if not isinstance(binding, dict) or set(binding) != expected_fields:
        errors.append(
            f"source manifest binding has wrong fields at {owner}; "
            f"expected {sorted(expected_fields)}"
        )
        return
    if binding.get("schema_version") != SOURCE_MANIFEST_BINDING_SCHEMA:
        errors.append(f"source manifest binding has wrong schema at {owner}")
        return
    raw_path = binding.get("path")
    expected_sha256 = binding.get("file_sha256")
    if not isinstance(raw_path, str) or not raw_path:
        errors.append(f"source manifest binding path is invalid at {owner}")
        return
    if not _is_sha256(expected_sha256):
        errors.append(f"source manifest binding digest is invalid at {owner}")
        return
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
        payload_bytes = path.read_bytes()
    except OSError as error:
        errors.append(f"source manifest is unreadable at {owner}: {error}")
        return
    if str(path) != raw_path:
        errors.append(f"source manifest binding path is not canonical at {owner}")
        return
    actual_sha256 = "sha256:" + hashlib.sha256(payload_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        errors.append(
            f"source manifest byte hash mismatch at {owner}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
        return
    if path in seen:
        return
    seen.add(path)
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        errors.append(f"source manifest JSON is invalid at {owner}: {error}")
        return
    if not isinstance(payload, dict):
        errors.append(f"source manifest must contain a JSON object at {owner}")
        return
    if follow_chain:
        _collect_authenticated_manifest_chain(
            payload,
            hashes,
            errors,
            manifest_path=path,
            seen=seen,
        )


def _collect_named_tool_provenance(
    provenance: Any,
    hashes: dict[str, set[str]],
    errors: list[str],
    *,
    manifest_path: Path,
) -> set[str]:
    required_fields = {"schema_version", "file_sha256", "feature_semantics_files"}
    if not isinstance(provenance, dict):
        errors.append(f"manifest lacks named tool_provenance object: {manifest_path}")
        return set()
    if not required_fields.issubset(provenance):
        errors.append(
            f"tool_provenance lacks required fields {sorted(required_fields)}: "
            f"{manifest_path}"
        )
        return set()
    if provenance.get("schema_version") != TOOL_PROVENANCE_SCHEMA:
        errors.append(f"tool_provenance has wrong schema: {manifest_path}")
        return set()
    file_sha256 = provenance.get("file_sha256")
    feature_files = provenance.get("feature_semantics_files")
    if not isinstance(file_sha256, dict) or not isinstance(feature_files, list):
        errors.append(f"tool_provenance fields have invalid types: {manifest_path}")
        return set()
    if not feature_files or any(
        not isinstance(path, str) or not path for path in feature_files
    ):
        errors.append(f"tool_provenance feature_semantics_files is invalid: {manifest_path}")
        return set()
    if any(path not in file_sha256 for path in feature_files):
        errors.append(
            f"tool_provenance omits a named feature-semantics digest: {manifest_path}"
        )
        return set()
    invalid = [
        path
        for path, digest in file_sha256.items()
        if not isinstance(path, str) or not path or not _is_sha256(digest)
    ]
    if invalid:
        errors.append(
            f"tool_provenance contains invalid sha256 entries {invalid}: {manifest_path}"
        )
        return set()
    for path, digest in file_sha256.items():
        hashes.setdefault(path, set()).add(str(digest))
    return set(file_sha256)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _check_manifest_metadata(failures: list[str], args: argparse.Namespace, metadata: dict[str, Any]) -> None:
    if metadata.get("manifest_errors"):
        failures.append(f"input manifest metadata has errors: {metadata['manifest_errors']}")
    if int(metadata.get("manifest_count", 0)) <= 0:
        failures.append("strict 35M teacher gate requires manifest.json or curation_report.json")

    tracks = set(str(track) for track in metadata.get("tracks", ()))
    expected_track = str(getattr(args, "track", "") or "")
    if expected_track:
        if not tracks:
            failures.append(f"strict 35M teacher gate requires manifest track={expected_track}, got none")
        elif tracks != {expected_track}:
            failures.append(
                f"strict 35M teacher gate manifest track mismatch: expected {expected_track}, got {sorted(tracks)}"
            )
    elif getattr(args, "production_35m_teacher", False) and len(tracks) != 1:
        failures.append(
            f"production 35M teacher gate requires exactly one manifest track, got {sorted(tracks)}"
        )

    observed_vps = set(int(value) for value in metadata.get("vps_to_win", ()))
    expected_vps = int(getattr(args, "vps_to_win", 0) or 0)
    if expected_vps:
        if not observed_vps:
            failures.append(f"strict 35M teacher gate requires manifest vps_to_win={expected_vps}, got none")
        elif observed_vps != {expected_vps}:
            failures.append(
                f"strict 35M teacher gate manifest vps_to_win mismatch: expected {expected_vps}, got {sorted(observed_vps)}"
            )
    elif getattr(args, "production_35m_teacher", False) and len(observed_vps) != 1:
        failures.append(
            f"production 35M teacher gate requires exactly one manifest vps_to_win, got {sorted(observed_vps)}"
        )

    if getattr(args, "production_35m_teacher", False):
        if metadata.get("source_provenance_errors"):
            failures.append(
                "production 35M teacher gate source provenance authentication "
                f"failed: {metadata['source_provenance_errors']}"
            )
        observed_mixed = set(bool(value) for value in metadata.get("mixed_seats", ()))
        if observed_mixed != {True}:
            failures.append(
                "production 35M teacher gate requires all input manifests to have "
                f"mixed_seats=true, got {sorted(observed_mixed)}"
            )
        observed_modes = set(str(value) for value in metadata.get("mixed_seat_modes", ()))
        if observed_modes != {"random"}:
            failures.append(
                "production 35M teacher gate requires all input manifests to have "
                f"mixed_seat_mode=random, got {sorted(observed_modes)}"
            )
        observed_graph_history = set(bool(value) for value in metadata.get("graph_history_features", ()))
        if observed_graph_history != {True}:
            failures.append(
                "production 35M teacher gate requires all input manifests to have "
                f"graph_history_features=true, got {sorted(observed_graph_history)}"
            )
        _check_feature_provenance(failures, metadata)


def _check_feature_provenance(failures: list[str], metadata: dict[str, Any]) -> None:
    observed = dict(metadata.get("source_provenance_hashes", {}))
    for path in sorted(REQUIRED_SOURCE_FEATURE_FILES):
        hashes = observed.get(path, [])
        if not hashes:
            failures.append(
                "production 35M teacher gate requires source feature provenance hash "
                f"for {path}"
            )
        elif len(hashes) != 1:
            failures.append(
                "production 35M teacher gate requires one source feature provenance hash "
                f"for {path}, got {hashes}"
            )
        elif not _is_sha256(hashes[0]):
            failures.append(
                "production 35M teacher gate requires canonical sha256 provenance "
                f"for {path}, got {hashes[0]!r}"
            )


def _parse_minimums(raw: str) -> dict[str, int]:
    result = {}
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"invalid minimum entry: {item}")
        name, value = item.split("=", 1)
        result[name.strip()] = int(value)
    return result


def _parse_float_thresholds(raw: str) -> dict[str, float]:
    result = {}
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"invalid threshold entry: {item}")
        name, value = item.split("=", 1)
        result[name.strip()] = float(value)
    return result


def _parse_prefixes(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


if __name__ == "__main__":
    main()
