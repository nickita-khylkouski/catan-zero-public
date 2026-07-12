#!/usr/bin/env python3
"""Offline, diagnostic-only adjudication of the matched n256 LR sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import a1_dual_arm_train as training  # noqa: E402
from tools import train_bc  # noqa: E402
from tools.a1_external_panel_compare import (  # noqa: E402
    ExternalPanelComparisonError,
    compare_matched_external_panels,
)
from tools.fleet import a1_n256_lr_eval as trial_tool  # noqa: E402
from tools.fleet import a1_h100_eval_fleet as fleet  # noqa: E402


SCHEMA = "a1-n256-lr-adjudication-v1"
INPUT_SCHEMA = "a1-n256-lr-adjudication-input-v1"
MAX_CLIPPED_FRACTION = 0.50
MAX_PRECLIP_GRAD_NORM = 100.0
MAX_VALIDATION_LOSS = 10.0
EXTERNAL_GENERALIZATION_FLOOR = -0.02
EXTERNAL_POINT_TIE_BAND = 0.02
TEACHER_GAP_TIE_BREAK = 0.02
VALUE_MSE_TIE_BREAK = 0.01


class AdjudicationError(RuntimeError):
    """Inputs cannot be authenticated as one matched diagnostic trial."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _file_ref(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve(strict=True)
    return {"path": str(path), "sha256": fleet._sha256(path)}  # noqa: SLF001


def _load(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdjudicationError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise AdjudicationError(f"{path} is not a JSON object")
    return value


def _wilson(wins: int, games: int, z: float = 1.96) -> list[float]:
    if games <= 0 or not 0 <= wins <= games:
        raise AdjudicationError("invalid win/game count")
    p = wins / games
    denominator = 1.0 + z * z / games
    center = p + z * z / (2.0 * games)
    half = z * math.sqrt(p * (1.0 - p) / games + z * z / (4.0 * games * games))
    return [
        max(0.0, (center - half) / denominator),
        min(1.0, (center + half) / denominator),
    ]


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _pareto_frontier(metrics: dict[str, dict[str, Any]]) -> list[str]:
    keys = ("external_delta", "internal_win_rate", "teacher_gap_closure")
    result = []
    for label, row in metrics.items():
        dominated = False
        for other_label, other in metrics.items():
            if other_label == label:
                continue
            no_worse = all(
                float(other[key]) >= float(row[key]) for key in keys
            ) and float(other["value_mse"]) <= float(row["value_mse"])
            strictly_better = any(
                float(other[key]) > float(row[key]) for key in keys
            ) or float(other["value_mse"]) < float(row["value_mse"])
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            result.append(label)
    return sorted(result)


def adjudicate_metrics(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the predeclared safety and lexicographic evidence policy."""
    safety: dict[str, dict[str, Any]] = {}
    safe: dict[str, dict[str, Any]] = {}
    required_finite = (
        "external_delta",
        "internal_win_rate",
        "teacher_gap_closure",
        "value_mse",
        "validation_loss",
    )
    for label, row in metrics.items():
        reasons = []
        for key in required_finite:
            if not _finite(row.get(key)):
                reasons.append(f"nonfinite_or_missing:{key}")
        clipping = row.get("clipped_fraction")
        max_norm = row.get("max_pre_clip_grad_norm")
        if clipping is not None and (
            not _finite(clipping) or float(clipping) > MAX_CLIPPED_FRACTION
        ):
            reasons.append("clipping_fraction_pathology")
        if max_norm is not None and (
            not _finite(max_norm) or float(max_norm) > MAX_PRECLIP_GRAD_NORM
        ):
            reasons.append("gradient_norm_pathology")
        if (
            _finite(row.get("validation_loss"))
            and float(row["validation_loss"]) > MAX_VALIDATION_LOSS
        ):
            reasons.append("validation_loss_pathology")
        safety[label] = {
            "eligible": not reasons,
            "reasons": reasons,
            "optimizer_telemetry_available": clipping is not None
            and max_norm is not None,
        }
        if not reasons:
            safe[label] = row

    ranking = sorted(
        safe,
        key=lambda label: (
            -float(safe[label]["external_delta"]),
            -float(safe[label]["internal_win_rate"]),
            -float(safe[label]["teacher_gap_closure"]),
            float(safe[label]["value_mse"]),
            label,
        ),
    )
    frontier = _pareto_frontier(safe) if safe else []
    winner = None
    reason = "no_safe_arms" if not safe else "uncertainty_or_tradeoff_unresolved"
    if safe:
        best_external = max(float(row["external_delta"]) for row in safe.values())
        external_contenders = [
            label
            for label, row in safe.items()
            if float(row["external_delta_ci"][1])
            >= max(float(other["external_delta_ci"][0]) for other in safe.values())
            and best_external - float(row["external_delta"]) <= EXTERNAL_POINT_TIE_BAND
        ]
        externally_clear = [
            label
            for label, row in safe.items()
            if all(
                label == other_label
                or float(row["external_delta_ci"][0])
                > float(other["external_delta_ci"][1])
                for other_label, other in safe.items()
            )
        ]
        candidate = externally_clear[0] if len(externally_clear) == 1 else None
        if candidate is None and external_contenders:
            internally_clear = [
                label
                for label in external_contenders
                if all(
                    label == other
                    or float(safe[label]["internal_win_rate_ci"][0])
                    > float(safe[other]["internal_win_rate_ci"][1])
                    for other in external_contenders
                )
            ]
            candidate = internally_clear[0] if len(internally_clear) == 1 else None
        if candidate is None and len(external_contenders) > 1:
            # Final tie-break is deliberately conservative: one arm must improve
            # teacher uptake materially without worse value calibration, or vice versa.
            tie_breakers = []
            for label in external_contenders:
                if all(
                    label == other
                    or (
                        float(safe[label]["teacher_gap_closure"])
                        >= float(safe[other]["teacher_gap_closure"])
                        + TEACHER_GAP_TIE_BREAK
                        and float(safe[label]["value_mse"])
                        <= float(safe[other]["value_mse"])
                    )
                    or (
                        float(safe[label]["value_mse"])
                        <= float(safe[other]["value_mse"]) - VALUE_MSE_TIE_BREAK
                        and float(safe[label]["teacher_gap_closure"])
                        >= float(safe[other]["teacher_gap_closure"])
                    )
                    for other in external_contenders
                ):
                    tie_breakers.append(label)
            candidate = tie_breakers[0] if len(tie_breakers) == 1 else None
        if candidate is not None:
            if float(safe[candidate]["external_delta"]) < EXTERNAL_GENERALIZATION_FLOOR:
                reason = "best_arm_fails_external_generalization_floor"
            else:
                winner = candidate
                reason = "predeclared_evidence_order_resolved"
    return {
        "winner": winner,
        "decision": "diagnostic_winner" if winner else "no_winner",
        "decision_reason": reason,
        "ranking": ranking,
        "pareto_frontier": frontier,
        "safety": safety,
        "thresholds": {
            "max_clipped_fraction": MAX_CLIPPED_FRACTION,
            "max_preclip_grad_norm": MAX_PRECLIP_GRAD_NORM,
            "max_validation_loss": MAX_VALIDATION_LOSS,
            "external_generalization_floor": EXTERNAL_GENERALIZATION_FLOOR,
            "external_point_tie_band": EXTERNAL_POINT_TIE_BAND,
            "teacher_gap_tie_break": TEACHER_GAP_TIE_BREAK,
            "value_mse_tie_break": VALUE_MSE_TIE_BREAK,
        },
    }


def _arm_metrics(
    *,
    label: str,
    paths: dict[str, Path],
    trial: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_path = paths["receipt"].resolve(strict=True)
    receipt = training.verify_receipt(receipt_path)
    expected_lr, expected_ablation = trial_tool.ARM_SPECS[label]
    ablation = receipt.get("inputs", {}).get("learner_ablation", {})
    recipe = ablation.get("effective_recipe", {})
    if (
        (receipt.get("arm_id"), receipt.get("subset_id")) != ("n256", "full-56k")
        or ablation.get("ablation_id") != expected_ablation
        or ablation.get("diagnostic_only") is not True
        or ablation.get("promotion_eligible") is not False
        or recipe.get("lr") != expected_lr
        or recipe.get("loser_sample_weight") != 1.0
        or recipe.get("epochs") != 1
    ):
        raise AdjudicationError(f"{label} receipt has wrong diagnostic provenance")
    report_ref = receipt.get("outputs", {}).get("report")
    if report_ref != _file_ref(paths["report"]):
        raise AdjudicationError(f"{label} report is not bound by its receipt")
    report = _load(paths["report"])
    checkpoint = receipt.get("outputs", {}).get("checkpoint", {})
    internal = _load(paths["internal"])
    external_candidate = _load(paths["external_candidate"])
    external_champion = _load(paths["external_champion"])
    expected_checkpoint_sha = checkpoint.get("sha256")
    if internal.get("candidate_checkpoint_sha256") != expected_checkpoint_sha:
        raise AdjudicationError(f"{label} internal panel binds another checkpoint")
    if external_candidate.get("candidate_checkpoint_sha256") != expected_checkpoint_sha:
        raise AdjudicationError(f"{label} external panel binds another checkpoint")
    champion_sha = trial["champion"]["sha256"]
    if (
        internal.get("baseline_checkpoint_sha256") != champion_sha
        or external_champion.get("candidate_checkpoint_sha256") != champion_sha
    ):
        raise AdjudicationError(f"{label} panels bind another champion")
    plan = fleet.load_plan(Path(trial["arms"][label]["plan_path"]), manifest)
    if plan.get("plan_hash") != trial["arms"][label]["plan_hash"]:
        raise AdjudicationError(f"{label} plan hash drift")
    if int(internal.get("base_seed", -1)) != int(
        plan["pair_claims"]["internal"]["base_seed"]
    ):
        raise AdjudicationError(f"{label} internal seed cohort drift")
    expected_external_seed = int(plan["pair_claims"]["external_matched"]["base_seed"])
    if any(
        int(row.get("base_seed", -1)) != expected_external_seed
        for row in (external_candidate, external_champion)
    ):
        raise AdjudicationError(f"{label} external seed cohort drift")
    if (
        internal.get("errors") != []
        or int(internal.get("games_truncated", -1)) != 0
        or int(internal.get("complete_pairs", -1))
        != int(trial["micro_panel"]["internal_pairs"])
    ):
        raise AdjudicationError(f"{label} internal panel is incomplete or unhealthy")
    for role, panel in (
        ("candidate", external_candidate),
        ("champion", external_champion),
    ):
        if (
            panel.get("errors") != []
            or panel.get("worker_errors") != []
            or int(panel.get("games_truncated", -1)) != 0
            or int(panel.get("games_engine_divergence", -1)) != 0
            or int(panel.get("complete_pairs", -1))
            != int(trial["micro_panel"]["external_pairs"])
        ):
            raise AdjudicationError(
                f"{label} external {role} panel is incomplete or unhealthy"
            )
    metrics = report.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise AdjudicationError(f"{label} report has no epoch metrics")
    validation = train_bc.objective_matched_validation_metrics(metrics[-1])
    try:
        external_comparison = compare_matched_external_panels(
            external_candidate, external_champion
        )
    except ExternalPanelComparisonError as error:
        raise AdjudicationError(
            f"{label} external panels are not one matched cohort: {error}"
        ) from error
    internal_games = int(internal.get("games_played", 0))
    internal_wins = int(internal.get("candidate_wins", -1))
    internal_ci = _wilson(internal_wins, internal_games)
    optimizer = metrics[-1].get("optimizer_observability", {})
    row = {
        "external_delta": float(external_comparison["candidate_minus_champion"]),
        "external_delta_ci": list(external_comparison["paired_seed_cluster_95ci"]),
        "external_paired_comparison": external_comparison,
        "internal_win_rate": float(internal["candidate_win_rate"]),
        "internal_win_rate_ci": internal_ci,
        "teacher_gap_closure": validation.get("active_policy_teacher_gap_closure"),
        "value_mse": validation.get("scalar_value_mse_diagnostic"),
        "validation_loss": validation.get("loss"),
        "clipped_fraction": optimizer.get("clipped_fraction"),
        "max_pre_clip_grad_norm": optimizer.get("max_pre_clip_total_grad_norm"),
    }
    refs = {name: _file_ref(path) for name, path in paths.items()}
    return row, refs


def adjudicate(input_path: Path) -> dict[str, Any]:
    descriptor = _load(input_path)
    if descriptor.get("schema_version") != INPUT_SCHEMA or set(
        descriptor.get("arms", {})
    ) != set(trial_tool.ARM_SPECS):
        raise AdjudicationError("adjudication input must bind all three LR arms")
    trial_path = Path(str(descriptor.get("trial"))).expanduser().resolve(strict=True)
    trial = _load(trial_path)
    if trial.get("schema_version") != trial_tool.TRIAL_SCHEMA:
        raise AdjudicationError("input trial schema drift")
    stated = trial.get("trial_hash")
    if stated != trial_tool._digest(
        {k: v for k, v in trial.items() if k != "trial_hash"}
    ):  # noqa: SLF001
        raise AdjudicationError("input trial hash drift")
    manifest = fleet.load_manifest(Path(trial["manifest"]["path"]))
    if manifest.get("manifest_hash") != trial["manifest"]["hash"]:
        raise AdjudicationError("input trial manifest drift")
    metrics = {}
    refs = {}
    science = set()
    for label, raw in descriptor["arms"].items():
        if not isinstance(raw, dict) or set(raw) != {
            "receipt",
            "report",
            "internal",
            "external_candidate",
            "external_champion",
        }:
            raise AdjudicationError(f"{label} input fields drift")
        paths = {key: Path(str(value)).expanduser() for key, value in raw.items()}
        metrics[label], refs[label] = _arm_metrics(
            label=label,
            paths=paths,
            trial=trial,
            manifest=manifest,
        )
        plan = fleet.load_plan(Path(trial["arms"][label]["plan_path"]), manifest)
        science.add(
            (
                plan["science_config_hash"],
                json.dumps(plan["pair_claims"], sort_keys=True),
            )
        )
    if len(science) != 1:
        raise AdjudicationError("LR panels are not science/seed matched")
    result = adjudicate_metrics(metrics)
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "promotion_action": None,
        "policy": "safety > external delta > internal paired strength > teacher-gap/value calibration; never loss alone",
        "trial": _file_ref(trial_path),
        "input_descriptor": _file_ref(input_path),
        "inputs": refs,
        "metrics": metrics,
        **result,
    }
    payload["verdict_sha256"] = _digest(payload)
    return payload


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = adjudicate(args.inputs)
        _write_new(args.out, value)
        print(json.dumps(value, sort_keys=True))
        return 0
    except (
        AdjudicationError,
        training.DualTrainError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
