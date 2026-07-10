#!/usr/bin/env python3
"""Score the isolated topology-v2 supervised learning gate.

The input is JSONL (or a JSON list) with one row per evaluated decision.  This
is deliberately a scorer, not a trainer: it cannot change model defaults.
Every arm and training seed must be evaluated on the identical, pre-registered
holdout rows.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "catan-zero-topology-learning-gate/v2"
REQUIRED_ROLES = (
    "reference",
    "capacity_compute_control",
    "geometry_control",
    "primary_candidate",
)
_ARM_METADATA_FIELDS = {"arm_id", "role", "expected_parameters", "note"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class GateInputError(ValueError):
    """The evidence cannot support a valid learning-gate decision."""


def _finite_number(value: Any, *, field: str, row: int | None = None) -> float:
    prefix = f"row {row}: " if row is not None else ""
    if isinstance(value, bool):
        raise GateInputError(f"{prefix}{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GateInputError(f"{prefix}{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise GateInputError(f"{prefix}{field} must be finite")
    return result


def _positive_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise GateInputError(f"{field} must be a positive integer")
    return value


def _sha256(value: Any, *, field: str, row: int | None = None) -> str:
    prefix = f"row {row}: " if row is not None else ""
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GateInputError(f"{prefix}{field} must be a lowercase SHA256 hex digest")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ce(record: Mapping[str, Any], row: int) -> float:
    has_ce = record.get("policy_ce") is not None
    has_probability = record.get("target_probability") is not None
    if has_ce == has_probability:
        raise GateInputError(
            f"row {row}: provide exactly one of policy_ce or target_probability"
        )
    if has_ce:
        value = _finite_number(record["policy_ce"], field="policy_ce", row=row)
        if value < 0:
            raise GateInputError(f"row {row}: policy_ce must be non-negative")
        return value
    probability = _finite_number(
        record["target_probability"], field="target_probability", row=row
    )
    if not 0 < probability <= 1:
        raise GateInputError(f"row {row}: target_probability must be in (0, 1]")
    return -math.log(probability)


def _experiment_contract(
    config: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Mapping[str, Any]], Mapping[str, Any]]:
    if not isinstance(config, Mapping):
        raise GateInputError("experiment config must be an object")
    arms = config.get("arms")
    if not isinstance(arms, list):
        raise GateInputError("experiment config arms must be a list")
    common = config.get("common")
    if not isinstance(common, Mapping):
        raise GateInputError("experiment config common must be an object")
    by_role: dict[str, str] = {}
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, arm in enumerate(arms):
        if not isinstance(arm, Mapping):
            raise GateInputError(f"experiment arm {index} must be an object")
        arm_id = arm.get("arm_id")
        if not isinstance(arm_id, str) or not arm_id:
            raise GateInputError(f"experiment arm {index} has no non-empty arm_id")
        if arm_id in by_id:
            raise GateInputError(f"experiment config has duplicate arm_id {arm_id!r}")
        by_id[arm_id] = arm
        role = arm.get("role")
        if role in REQUIRED_ROLES:
            if role in by_role:
                raise GateInputError(f"experiment config has duplicate role {role!r}")
            by_role[str(role)] = arm_id
    missing = sorted(set(REQUIRED_ROLES) - set(by_role))
    if missing:
        raise GateInputError(f"experiment config is missing roles: {missing}")
    if len(set(by_role.values())) != len(by_role):
        raise GateInputError("required roles must refer to distinct arms")
    gate = config.get("learning_gate")
    if not isinstance(gate, Mapping):
        raise GateInputError("experiment config has no learning_gate object")
    return by_role, by_id, gate


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise GateInputError("cannot form a confidence interval from no samples")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise GateInputError("metric mask selected no records")
    return math.fsum(values) / len(values)


def _game_values(
    rows: Sequence[dict[str, Any]], *, arm: str, seed: int, primary: bool
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in rows:
        if record["arm"] != arm or record["training_seed"] != seed or record["forced"]:
            continue
        if primary and not record["topology_sensitive"]:
            continue
        grouped[record["game_id"]].append(record["ce"])
    if not grouped:
        name = "primary" if primary else "overall"
        raise GateInputError(f"arm {arm!r}, seed {seed}: {name} mask selected no games")
    return dict(grouped)


def _aggregate_games(
    values: Mapping[str, Sequence[float]], games: Sequence[str], *, game_macro: bool
) -> float:
    if game_macro:
        return _mean([_mean(values[game]) for game in games])
    return _mean([value for game in games for value in values[game]])


def _paired_crossed_bootstrap(
    candidate: Mapping[int, Mapping[str, Sequence[float]]],
    reference: Mapping[int, Mapping[str, Sequence[float]]],
    *,
    samples: int,
    rng: random.Random,
    game_macro: bool,
) -> dict[str, Any]:
    seeds = sorted(candidate)
    if set(seeds) != set(reference):
        raise GateInputError("candidate/reference seed support differs")
    common_games: list[str] | None = None
    per_seed: dict[str, dict[str, float]] = {}
    for seed in seeds:
        games = sorted(candidate[seed])
        if set(games) != set(reference[seed]):
            raise GateInputError(
                f"candidate/reference game support differs for seed {seed}"
            )
        if common_games is None:
            common_games = games
        elif games != common_games:
            raise GateInputError("holdout game support differs across training seeds")
        # A relative bootstrap is undefined if a resampled reference cluster can be zero.
        for game in games:
            if _mean(reference[seed][game]) <= 0:
                raise GateInputError(
                    f"reference CE must be positive for every game cluster; seed {seed}, game {game!r}"
                )
        cand = _aggregate_games(candidate[seed], games, game_macro=game_macro)
        ref = _aggregate_games(reference[seed], games, game_macro=game_macro)
        per_seed[str(seed)] = {
            "candidate_ce": cand,
            "reference_ce": ref,
            "relative_improvement": (ref - cand) / ref,
        }
    assert common_games is not None
    point_candidate = _mean([value["candidate_ce"] for value in per_seed.values()])
    point_reference = _mean([value["reference_ce"] for value in per_seed.values()])
    differences: list[float] = []
    improvements: list[float] = []
    regressions: list[float] = []
    for _ in range(samples):
        selected_seeds = [rng.choice(seeds) for _ in seeds]
        # Games are a crossed factor shared by every selected model seed.
        selected_games = [rng.choice(common_games) for _ in common_games]
        cand_value = _mean(
            [
                _aggregate_games(candidate[seed], selected_games, game_macro=game_macro)
                for seed in selected_seeds
            ]
        )
        ref_value = _mean(
            [
                _aggregate_games(reference[seed], selected_games, game_macro=game_macro)
                for seed in selected_seeds
            ]
        )
        if (
            ref_value <= 0
        ):  # Defensive; per-game validation above makes this unreachable.
            raise GateInputError("bootstrap reference CE is not positive")
        difference = cand_value - ref_value
        differences.append(difference)
        improvements.append(-difference / ref_value)
        regressions.append(difference / ref_value)
    difference = point_candidate - point_reference
    return {
        "candidate_ce": point_candidate,
        "reference_ce": point_reference,
        "candidate_minus_reference": difference,
        "relative_improvement": -difference / point_reference,
        "relative_regression": difference / point_reference,
        "difference_ci95": [
            _quantile(differences, 0.025),
            _quantile(differences, 0.975),
        ],
        "relative_improvement_ci95": [
            _quantile(improvements, 0.025),
            _quantile(improvements, 0.975),
        ],
        "relative_regression_ci95": [
            _quantile(regressions, 0.025),
            _quantile(regressions, 0.975),
        ],
        "per_seed": per_seed,
    }


def _validate_resolved_config(
    resolved: Mapping[str, Any],
    *,
    common: Mapping[str, Any],
    arm: Mapping[str, Any],
    row: int,
) -> None:
    expected = dict(common)
    expected.update(
        {key: value for key, value in arm.items() if key not in _ARM_METADATA_FIELDS}
    )
    for key, value in expected.items():
        if resolved.get(key) != value:
            raise GateInputError(
                f"row {row}: resolved_config[{key!r}] does not match experiment matrix"
            )


def score_learning_gate(
    records: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260710,
    experiment_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate config-bound evidence and return a deterministic pass/fail report."""

    if type(bootstrap_samples) is not int or bootstrap_samples < 100:
        raise GateInputError("bootstrap_samples must be an integer of at least 100")
    if type(bootstrap_seed) is not int:
        raise GateInputError("bootstrap_seed must be an integer")
    roles, arms_by_id, gate = _experiment_contract(config)
    expected_experiment_sha = (
        _sha256(experiment_config_sha256, field="experiment_config_sha256")
        if experiment_config_sha256 is not None
        else _canonical_sha256(config)
    )
    raw_seeds = gate.get("seeds")
    if (
        not isinstance(raw_seeds, list)
        or len(raw_seeds) < 2
        or any(type(seed) is not int for seed in raw_seeds)
    ):
        raise GateInputError(
            "learning_gate.seeds must contain at least two integer seeds"
        )
    expected_seeds = tuple(raw_seeds)
    if len(set(expected_seeds)) != len(expected_seeds):
        raise GateInputError("learning_gate.seeds must be unique")
    minimum_games = _positive_int(
        gate.get("minimum_holdout_games"), field="learning_gate.minimum_holdout_games"
    )
    minimum_sensitive = _positive_int(
        gate.get("minimum_topology_sensitive_decisions"),
        field="learning_gate.minimum_topology_sensitive_decisions",
    )
    expected_hashes = {
        field: _sha256(gate.get(field), field=f"learning_gate.{field}")
        for field in (
            "topology_mask_registration_artifact_sha256",
            "training_manifest_sha256",
            "holdout_manifest_sha256",
            "training_data_sha256",
        )
    }
    expected_budget = {
        field: _positive_int(gate.get(field), field=f"learning_gate.{field}")
        for field in (
            "optimizer_steps",
            "global_batch_size",
            "sample_presentations_per_arm_seed",
        )
    }
    minimum_incumbent = _finite_number(
        gate.get("minimum_relative_improvement_vs_incumbent"),
        field="learning_gate.minimum_relative_improvement_vs_incumbent",
    )
    minimum_capacity = _finite_number(
        gate.get("minimum_relative_improvement_vs_capacity_control"),
        field="learning_gate.minimum_relative_improvement_vs_capacity_control",
    )
    maximum_regression = _finite_number(
        gate.get("maximum_overall_ce_regression"),
        field="learning_gate.maximum_overall_ce_regression",
    )
    if not 0 <= minimum_incumbent < 1 or not 0 <= minimum_capacity < 1:
        raise GateInputError(
            "minimum relative-improvement thresholds must be in [0, 1)"
        )
    if not 0 <= maximum_regression < 1:
        raise GateInputError("maximum overall CE regression must be in [0, 1)")

    required_arms = set(roles.values())
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str]] = set()
    support_masks: dict[tuple[str, str], tuple[bool, bool]] = {}
    provenance_by_run: dict[tuple[str, int], dict[str, Any]] = {}
    checkpoint_owners: dict[str, tuple[str, int]] = {}
    required_fields = (
        "arm",
        "training_seed",
        "game_id",
        "decision_id",
        "forced",
        "topology_sensitive",
        "evaluation_split",
        "is_training_game",
        "topology_mask_registration_artifact_sha256",
        "training_manifest_sha256",
        "holdout_manifest_sha256",
        "experiment_config_sha256",
        "run_provenance",
    )
    for row_number, raw in enumerate(records, 1):
        if not isinstance(raw, Mapping):
            raise GateInputError(f"row {row_number}: record must be an object")
        arm = raw.get("arm")
        if not isinstance(arm, str):
            raise GateInputError(f"row {row_number}: arm must be a string")
        if arm not in required_arms:
            continue
        missing = [field for field in required_fields if field not in raw]
        if missing:
            raise GateInputError(f"row {row_number}: missing fields {missing}")
        if type(raw["training_seed"]) is not int:
            raise GateInputError(f"row {row_number}: training_seed must be an integer")
        seed = raw["training_seed"]
        if seed not in expected_seeds:
            raise GateInputError(f"row {row_number}: unexpected training_seed {seed!r}")
        if (
            not isinstance(raw["game_id"], str)
            or not raw["game_id"]
            or not isinstance(raw["decision_id"], str)
            or not raw["decision_id"]
        ):
            raise GateInputError(
                f"row {row_number}: game_id and decision_id must be non-empty strings"
            )
        game, decision = raw["game_id"], raw["decision_id"]
        if (
            type(raw["forced"]) is not bool
            or type(raw["topology_sensitive"]) is not bool
        ):
            raise GateInputError(
                f"row {row_number}: forced/topology_sensitive must be booleans"
            )
        if raw["evaluation_split"] != "holdout" or raw["is_training_game"] is not False:
            raise GateInputError(
                f"row {row_number}: training/evaluation game leakage detected"
            )
        for field in (
            "topology_mask_registration_artifact_sha256",
            "training_manifest_sha256",
            "holdout_manifest_sha256",
        ):
            actual = _sha256(raw[field], field=field, row=row_number)
            if actual != expected_hashes[field]:
                raise GateInputError(
                    f"row {row_number}: {field} does not match experiment config"
                )
        actual_experiment_sha = _sha256(
            raw["experiment_config_sha256"],
            field="experiment_config_sha256",
            row=row_number,
        )
        if actual_experiment_sha != expected_experiment_sha:
            raise GateInputError(
                f"row {row_number}: experiment_config_sha256 does not match scorer config"
            )
        key = (arm, seed, game, decision)
        if key in seen:
            raise GateInputError(
                f"row {row_number}: duplicate decision overlap for {key}"
            )
        seen.add(key)
        support_key = (game, decision)
        mask = (raw["forced"], raw["topology_sensitive"])
        if support_key in support_masks and support_masks[support_key] != mask:
            raise GateInputError(
                f"row {row_number}: mask differs across arms/seeds for {support_key}"
            )
        support_masks[support_key] = mask

        provenance = raw["run_provenance"]
        if not isinstance(provenance, Mapping):
            raise GateInputError(f"row {row_number}: run_provenance must be an object")
        provenance_required = (
            "checkpoint_sha256",
            "resolved_config",
            "resolved_config_sha256",
            "parameter_count",
            "training_data_sha256",
            "optimizer_steps",
            "global_batch_size",
            "sample_presentations",
            "training_report_sha256",
            "experiment_config_sha256",
            "optimizer_sidecar_sha256",
            "train_config_hash",
        )
        missing_provenance = [
            field for field in provenance_required if field not in provenance
        ]
        if missing_provenance:
            raise GateInputError(
                f"row {row_number}: run_provenance missing fields {missing_provenance}"
            )
        checkpoint_sha = _sha256(
            provenance["checkpoint_sha256"],
            field="run_provenance.checkpoint_sha256",
            row=row_number,
        )
        for field in (
            "training_report_sha256",
            "experiment_config_sha256",
            "optimizer_sidecar_sha256",
        ):
            value = _sha256(
                provenance[field], field=f"run_provenance.{field}", row=row_number
            )
            if field == "experiment_config_sha256" and value != expected_experiment_sha:
                raise GateInputError(
                    f"row {row_number}: run_provenance experiment config SHA mismatch"
                )
        if not isinstance(provenance["train_config_hash"], str) or not re.fullmatch(
            r"sha256:[0-9a-f]{16}", provenance["train_config_hash"]
        ):
            raise GateInputError(
                f"row {row_number}: run_provenance.train_config_hash is invalid"
            )
        resolved = provenance["resolved_config"]
        if not isinstance(resolved, Mapping):
            raise GateInputError(
                f"row {row_number}: run_provenance.resolved_config must be an object"
            )
        try:
            computed_config_sha = _canonical_sha256(resolved)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GateInputError(
                f"row {row_number}: resolved_config is not canonical JSON"
            ) from exc
        supplied_config_sha = _sha256(
            provenance["resolved_config_sha256"],
            field="run_provenance.resolved_config_sha256",
            row=row_number,
        )
        if supplied_config_sha != computed_config_sha:
            raise GateInputError(
                f"row {row_number}: resolved_config_sha256 does not match resolved_config"
            )
        arm_config = arms_by_id[arm]
        _validate_resolved_config(
            resolved, common=config["common"], arm=arm_config, row=row_number
        )
        parameter_count = _positive_int(
            provenance["parameter_count"],
            field=f"row {row_number}: run_provenance.parameter_count",
        )
        expected_parameters = _positive_int(
            arm_config.get("expected_parameters"),
            field=f"experiment arm {arm}.expected_parameters",
        )
        if parameter_count != expected_parameters:
            raise GateInputError(
                f"row {row_number}: parameter_count does not match experiment matrix"
            )
        training_data_sha = _sha256(
            provenance["training_data_sha256"],
            field="run_provenance.training_data_sha256",
            row=row_number,
        )
        if training_data_sha != expected_hashes["training_data_sha256"]:
            raise GateInputError(
                f"row {row_number}: training_data_sha256 does not match experiment config"
            )
        for record_field, config_field in (
            ("optimizer_steps", "optimizer_steps"),
            ("global_batch_size", "global_batch_size"),
            ("sample_presentations", "sample_presentations_per_arm_seed"),
        ):
            actual = _positive_int(
                provenance[record_field],
                field=f"row {row_number}: run_provenance.{record_field}",
            )
            if actual != expected_budget[config_field]:
                raise GateInputError(
                    f"row {row_number}: {record_field} does not match experiment matrix"
                )
        canonical_provenance = dict(provenance)
        run_key = (arm, seed)
        if (
            run_key in provenance_by_run
            and provenance_by_run[run_key] != canonical_provenance
        ):
            raise GateInputError(
                f"row {row_number}: run provenance differs within arm/seed {run_key}"
            )
        provenance_by_run[run_key] = canonical_provenance
        if (
            checkpoint_sha in checkpoint_owners
            and checkpoint_owners[checkpoint_sha] != run_key
        ):
            raise GateInputError(
                f"row {row_number}: checkpoint reused across arm/seed runs"
            )
        checkpoint_owners[checkpoint_sha] = run_key
        normalized.append(
            {
                "arm": arm,
                "training_seed": seed,
                "game_id": game,
                "decision_id": decision,
                "forced": mask[0],
                "topology_sensitive": mask[1],
                "ce": _ce(raw, row_number),
            }
        )

    expected_support: set[tuple[str, str]] | None = None
    for arm in sorted(required_arms):
        for seed in expected_seeds:
            support = {
                (record["game_id"], record["decision_id"])
                for record in normalized
                if record["arm"] == arm and record["training_seed"] == seed
            }
            if expected_support is None:
                expected_support = support
            elif support != expected_support:
                raise GateInputError(
                    f"evaluation decision support differs for arm {arm!r}, seed {seed}"
                )
    if not expected_support:
        raise GateInputError("no required-arm records were supplied")
    holdout_games = {game for game, _decision in expected_support}
    sensitive_decisions = {
        key
        for key in expected_support
        if support_masks[key][1] and not support_masks[key][0]
    }
    if len(holdout_games) < minimum_games:
        raise GateInputError(
            f"insufficient holdout games: require {minimum_games}, got {len(holdout_games)}"
        )
    if len(sensitive_decisions) < minimum_sensitive:
        raise GateInputError(
            f"insufficient topology-sensitive decisions: require {minimum_sensitive}, got {len(sensitive_decisions)}"
        )

    primary: dict[str, dict[int, dict[str, list[float]]]] = {}
    overall: dict[str, dict[int, dict[str, list[float]]]] = {}
    for arm in required_arms:
        primary[arm] = {
            seed: _game_values(normalized, arm=arm, seed=seed, primary=True)
            for seed in expected_seeds
        }
        overall[arm] = {
            seed: _game_values(normalized, arm=arm, seed=seed, primary=False)
            for seed in expected_seeds
        }
    candidate = roles["primary_candidate"]
    comparisons: dict[str, Any] = {}
    for offset, (name, role) in enumerate(
        (
            ("vs_incumbent", "reference"),
            ("vs_capacity_control", "capacity_compute_control"),
            ("vs_rewired", "geometry_control"),
        )
    ):
        comparisons[name] = _paired_crossed_bootstrap(
            primary[candidate],
            primary[roles[role]],
            samples=bootstrap_samples,
            rng=random.Random(bootstrap_seed + offset),
            game_macro=True,
        )
    incumbent = roles["reference"]
    overall_vs_incumbent = _paired_crossed_bootstrap(
        overall[candidate],
        overall[incumbent],
        samples=bootstrap_samples,
        rng=random.Random(bootstrap_seed + 3),
        game_macro=False,
    )
    checks = {
        "primary_vs_incumbent_point_threshold": comparisons["vs_incumbent"][
            "relative_improvement"
        ]
        >= minimum_incumbent,
        "primary_vs_incumbent_ci_excludes_zero": comparisons["vs_incumbent"][
            "difference_ci95"
        ][1]
        < 0,
        "primary_vs_capacity_point_threshold": comparisons["vs_capacity_control"][
            "relative_improvement"
        ]
        >= minimum_capacity,
        "primary_vs_capacity_ci_excludes_zero": comparisons["vs_capacity_control"][
            "difference_ci95"
        ][1]
        < 0,
        "true_topology_beats_rewired_ci_excludes_zero": comparisons["vs_rewired"][
            "difference_ci95"
        ][1]
        < 0,
        "overall_ce_regression_upper_ci_within_limit": overall_vs_incumbent[
            "relative_regression_ci95"
        ][1]
        <= maximum_regression,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if all(checks.values()) else "fail",
        "metric_contract": {
            "primary": "game-macro policy CE; non-forced and topology-sensitive",
            "overall": "decision-micro policy CE within seed, seed-macro; non-forced",
            "ci": "paired crossed bootstrap over training seeds and common holdout games",
        },
        "roles": roles,
        "provenance_contract": expected_hashes,
        "run_provenance": {
            f"{arm}/seed-{seed}": {
                "checkpoint_sha256": value["checkpoint_sha256"],
                "resolved_config_sha256": value["resolved_config_sha256"],
                "parameter_count": value["parameter_count"],
                "training_data_sha256": value["training_data_sha256"],
                "optimizer_steps": value["optimizer_steps"],
                "global_batch_size": value["global_batch_size"],
                "sample_presentations": value["sample_presentations"],
                "training_report_sha256": value["training_report_sha256"],
                "experiment_config_sha256": value["experiment_config_sha256"],
                "optimizer_sidecar_sha256": value["optimizer_sidecar_sha256"],
                "train_config_hash": value["train_config_hash"],
            }
            for (arm, seed), value in sorted(provenance_by_run.items())
        },
        "evidence": {
            "rows": len(normalized),
            "decisions_per_arm_seed": len(expected_support),
            "holdout_games": len(holdout_games),
            "topology_sensitive_decisions": len(sensitive_decisions),
            "training_seeds": list(expected_seeds),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "thresholds": {
            "minimum_relative_improvement_vs_incumbent": minimum_incumbent,
            "minimum_relative_improvement_vs_capacity_control": minimum_capacity,
            "maximum_overall_ce_regression": maximum_regression,
        },
        "comparisons": comparisons,
        "overall_vs_incumbent": overall_vs_incumbent,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def _load_records(path: Path) -> list[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        value = json.loads(text)
        if not isinstance(value, list):
            raise GateInputError("JSON input must contain a list of records")
        return value
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path, required=True, help="JSONL or JSON list"
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/rnd/topology_v2/experiment_matrix.json"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        raw = args.records.read_bytes()
        config_raw = args.experiment_config.read_bytes()
        records = _load_records(args.records)
        config = json.loads(config_raw)
        report = score_learning_gate(
            records,
            config,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            experiment_config_sha256=hashlib.sha256(config_raw).hexdigest(),
        )
        report["input_sha256"] = hashlib.sha256(raw).hexdigest()
        report["experiment_config_sha256"] = hashlib.sha256(config_raw).hexdigest()
        rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (
        GateInputError,
        json.JSONDecodeError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        AttributeError,
        UnicodeError,
    ) as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "invalid",
            "error": str(exc),
        }
        rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except OSError as exc:
        sys.stderr.write(f"learning-gate output error: {exc}\n")
        return 2
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
