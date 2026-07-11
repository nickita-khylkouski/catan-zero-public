#!/usr/bin/env python3
"""Score the registered E3 fixed-K A1 supervised learning screen.

Input is JSONL (or a JSON list) with one row per holdout decision and run.
Every one of the five arms and three training seeds must cover the identical
holdout decisions. Decision losses are summarized by run and game exactly once;
the crossed bootstrap operates only on those compact sufficient statistics.
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
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "catan-zero-e3-learning-gate/v1"
ARMS = ("rrt-k0", "think-rrt-k1", "think-rrt-k2", "think-rrt-k4", "think-rrt-k8")
PRIMARY = ("think-rrt-k2", "think-rrt-k4")
SECONDARY = ("think-rrt-k8",)
REFERENCE = "think-rrt-k1"
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


class GateInputError(ValueError):
    """Evidence or registration cannot support an E3 learning conclusion."""


def _sha(value: Any, *, field: str, row: int | None = None) -> str:
    prefix = f"row {row}: " if row is not None else ""
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise GateInputError(f"{prefix}{field} must be a lowercase SHA256 digest")
    return value


def _finite(value: Any, *, field: str, row: int | None = None) -> float:
    prefix = f"row {row}: " if row is not None else ""
    if isinstance(value, bool):
        raise GateInputError(f"{prefix}{field} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GateInputError(f"{prefix}{field} must be finite") from exc
    if not math.isfinite(result):
        raise GateInputError(f"{prefix}{field} must be finite")
    return result


def _positive_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise GateInputError(f"{field} must be a positive integer")
    return value


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _validate_config(
    config: Mapping[str, Any], gate_contract: Mapping[str, Any]
) -> dict[str, Any]:
    if config.get("schema_version") != "catan-zero-e3-a1-screen/v1":
        raise GateInputError("unsupported E3 experiment schema")
    semantic = dict(config)
    declared = semantic.pop("config_sha256", None)
    if declared != _canonical_sha(semantic):
        raise GateInputError("experiment config self-hash is invalid")
    if config.get("status") != "registered_ready":
        raise GateInputError("E3 experiment is not registered_ready")
    arms = config.get("arms")
    if not isinstance(arms, list):
        raise GateInputError("experiment arms must be a list")
    by_id = {item.get("arm_id"): item for item in arms if isinstance(item, Mapping)}
    if set(by_id) != set(ARMS) or len(arms) != len(ARMS):
        raise GateInputError("experiment must contain exactly the five E3 arms")
    expected = {
        "rrt-k0": (0, 20_070_932, "compute_control_only", False),
        "think-rrt-k1": (1, 22_146_068, "capacity_matched_reference", False),
        "think-rrt-k2": (2, 22_146_068, "primary_candidate", True),
        "think-rrt-k4": (4, 22_146_068, "primary_candidate", True),
        "think-rrt-k8": (8, 22_146_068, "saturation_secondary", False),
    }
    for arm, (steps, params, role, eligible) in expected.items():
        item = by_id[arm]
        if (
            item.get("latent_deliberation_steps") != steps
            or item.get("expected_parameters") != params
            or item.get("comparison_role") != role
            or item.get("promotion_eligible") is not eligible
        ):
            raise GateInputError(f"registered scientific role drift for {arm}")
    comparison = config.get("comparison_contract")
    gate = gate_contract
    registration = config.get("registration")
    if not isinstance(comparison, Mapping) or not isinstance(gate, Mapping):
        raise GateInputError("comparison/learning-gate registration is missing")
    gate_semantic = dict(gate)
    gate_declared = gate_semantic.pop("config_sha256", None)
    if (
        gate.get("schema_version") != "catan-zero-e3-learning-gate-contract/v1"
        or gate_declared != _canonical_sha(gate_semantic)
    ):
        raise GateInputError("learning-gate contract self-hash is invalid")
    if gate.get("experiment_semantic_sha256") != config.get("config_sha256"):
        raise GateInputError("learning-gate contract binds a different experiment")
    scorer_sha = _sha(
        gate.get("scorer_source_sha256"), field="gate.scorer_source_sha256"
    )
    if scorer_sha != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise GateInputError("learning-gate scorer source differs from preregistration")
    if (
        comparison.get("primary_reference_arm") != REFERENCE
        or comparison.get("primary_candidate_arms") != list(PRIMARY)
        or comparison.get("secondary_candidate_arms") != list(SECONDARY)
        or comparison.get("compute_control_arms") != ["rrt-k0"]
    ):
        raise GateInputError("capacity-aware E3 comparison contract drifted")
    if (
        gate.get("primary_reference_arm") != REFERENCE
        or gate.get("primary_candidate_arms") != list(PRIMARY)
        or gate.get("secondary_candidate_arms") != list(SECONDARY)
        or gate.get("descriptive_compute_control_arms") != ["rrt-k0"]
    ):
        raise GateInputError("learning-gate arm roles differ from experiment")
    if (
        gate.get("primary_metric")
        != "game_macro_soft_target_policy_ce_nonforced"
        or gate.get("primary_metric") != comparison.get("primary_metric")
        or gate.get("uncertainty")
        != "paired_crossed_bootstrap_training_seed_and_common_holdout_game"
        or gate.get("uncertainty") != comparison.get("uncertainty")
        or gate.get("scorer") != "tools/rnd_e3_learning_gate.py"
    ):
        raise GateInputError("learning-gate metric/scorer/uncertainty contract drifted")
    if (
        gate.get("minimum_relative_improvement_vs_k1")
        != comparison.get("minimum_relative_improvement_vs_k1")
        or gate.get("maximum_overall_ce_regression")
        != comparison.get("maximum_overall_ce_regression")
    ):
        raise GateInputError("learning-gate thresholds differ from experiment registration")
    seeds = gate.get("seeds")
    if seeds != [11, 29, 47]:
        raise GateInputError("E3 gate must use exactly training seeds 11/29/47")
    for field, value in {
        "optimizer_steps": 250,
        "global_batch_size": 4096,
        "sample_presentations_per_arm_seed": 1_024_000,
    }.items():
        if gate.get(field) != value:
            raise GateInputError(f"learning_gate.{field} drifted")
    minimum_games = _positive_int(
        gate.get("minimum_holdout_games"), field="learning_gate.minimum_holdout_games"
    )
    minimum_decisions = _positive_int(
        gate.get("minimum_nonforced_decisions"),
        field="learning_gate.minimum_nonforced_decisions",
    )
    improvement = _finite(
        gate.get("minimum_relative_improvement_vs_k1"),
        field="learning_gate.minimum_relative_improvement_vs_k1",
    )
    regression = _finite(
        gate.get("maximum_overall_ce_regression"),
        field="learning_gate.maximum_overall_ce_regression",
    )
    if not 0 <= improvement < 1 or not 0 <= regression < 1:
        raise GateInputError("registered E3 thresholds must be in [0, 1)")
    if gate.get("require_primary_improvement_ci_lower_above_zero") is not True:
        raise GateInputError("primary confidence requirement is not registered")
    if gate.get("require_overall_regression_ci_upper_within_limit") is not True:
        raise GateInputError("overall-regression confidence requirement is not registered")
    if not isinstance(registration, Mapping):
        raise GateInputError("experiment has no frozen registration")
    for field in (
        "corpus_fingerprint",
        "training_manifest_sha256",
        "validation_manifest_sha256",
        "identity_report_sha256",
    ):
        _sha(registration.get(field), field=f"registration.{field}")
    init = registration.get("initial_checkpoint_sha256_by_arm_seed")
    expected_run_keys = {f"{arm}@{seed}" for arm in ARMS for seed in seeds}
    if not isinstance(init, Mapping) or set(init) != expected_run_keys:
        raise GateInputError("registered initialization family is incomplete")
    for key, digest in init.items():
        _sha(digest, field=f"registration.initial_checkpoint[{key}]")
    return {
        "arms": by_id,
        "seeds": tuple(seeds),
        "minimum_games": minimum_games,
        "minimum_decisions": minimum_decisions,
        "improvement": improvement,
        "regression": regression,
        "registration": registration,
        "experiment_file_sha256": _sha(
            gate.get("experiment_file_sha256"), field="gate.experiment_file_sha256"
        ),
        "gate_config_sha256": gate_declared,
        "bootstrap_samples": _positive_int(
            gate.get("bootstrap_samples"), field="gate.bootstrap_samples"
        ),
        "bootstrap_seed": gate.get("bootstrap_seed"),
    }


def _validate_resolved_config(
    resolved: Mapping[str, Any], *, arm: Mapping[str, Any], row: int
) -> None:
    if (
        resolved.get("schema_version") != 6
        or resolved.get("pipeline") != "train"
        or not isinstance(resolved.get("fields"), Mapping)
    ):
        raise GateInputError(f"row {row}: resolved_train_config envelope is invalid")
    fields = resolved["fields"]
    required = {
        "hidden_size": 384,
        "graph_layers": 9,
        "attention_heads": 6,
        "entity_state_trunk": "rrt",
        "relational_block_pattern": "RRTRRTRRT",
        "relational_ff_size": 1024,
        "relational_bases": 4,
        "relational_action_cross_layers": 1,
        "latent_deliberation_slots": 8,
        "latent_deliberation_steps": arm["latent_deliberation_steps"],
        "max_steps": 250,
        "batch_size": 1024,
        "grad_accum_steps": 4,
        "mask_hidden_info": True,
    }
    for field, expected in required.items():
        if fields.get(field) != expected:
            raise GateInputError(
                f"row {row}: resolved_train_config.fields.{field} differs from registration"
            )


def _crossed_bootstrap(
    candidate: Mapping[int, Mapping[str, Any]],
    reference: Mapping[int, Mapping[str, Any]],
    *,
    samples: int,
    rng: random.Random,
    game_macro: bool,
) -> dict[str, Any]:
    seeds = sorted(candidate)
    games = sorted(candidate[seeds[0]])
    for seed in seeds:
        if set(candidate[seed]) != set(games) or set(reference[seed]) != set(games):
            raise GateInputError("bootstrap seed/game summary support differs")

    def aggregate(values: Mapping[int, Mapping[str, Any]], ss, gg) -> float:
        if game_macro:
            return math.fsum(values[seed][game] for seed in ss for game in gg) / (
                len(ss) * len(gg)
            )
        # Decision-micro inside each sampled training seed, then seed-macro.
        per_seed = []
        for seed in ss:
            total = math.fsum(values[seed][game][0] for game in gg)
            count = sum(values[seed][game][1] for game in gg)
            if count <= 0:
                raise GateInputError("nonforced decision-micro sample is empty")
            per_seed.append(total / count)
        return math.fsum(per_seed) / len(per_seed)

    point_candidate = aggregate(candidate, seeds, games)
    point_reference = aggregate(reference, seeds, games)
    if point_reference <= 0:
        raise GateInputError("reference CE must be positive")
    improvements: list[float] = []
    regressions: list[float] = []
    differences: list[float] = []
    for _ in range(samples):
        selected_seeds = [rng.choice(seeds) for _seed in seeds]
        selected_games = [rng.choice(games) for _game in games]
        cand = aggregate(candidate, selected_seeds, selected_games)
        ref = aggregate(reference, selected_seeds, selected_games)
        if ref <= 0:
            raise GateInputError("resampled reference CE is non-positive")
        difference = cand - ref
        differences.append(difference)
        improvements.append(-difference / ref)
        regressions.append(difference / ref)
    difference = point_candidate - point_reference
    return {
        "candidate_ce": point_candidate,
        "reference_ce": point_reference,
        "candidate_minus_reference": difference,
        "relative_improvement": -difference / point_reference,
        "relative_regression": difference / point_reference,
        "difference_ci95": [_quantile(differences, 0.025), _quantile(differences, 0.975)],
        "relative_improvement_ci95": [
            _quantile(improvements, 0.025),
            _quantile(improvements, 0.975),
        ],
        "relative_regression_ci95": [
            _quantile(regressions, 0.025),
            _quantile(regressions, 0.975),
        ],
    }


def score_learning_gate(
    records: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    gate_contract: Mapping[str, Any],
    *,
    experiment_config_sha256: str,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260710,
) -> dict[str, Any]:
    if type(bootstrap_samples) is not int or bootstrap_samples < 100:
        raise GateInputError("bootstrap_samples must be an integer >= 100")
    contract = _validate_config(config, gate_contract)
    if type(contract["bootstrap_seed"]) is not int:
        raise GateInputError("gate.bootstrap_seed must be an integer")
    if (
        bootstrap_samples != contract["bootstrap_samples"]
        or bootstrap_seed != contract["bootstrap_seed"]
    ):
        raise GateInputError("bootstrap settings differ from preregistered gate contract")
    experiment_sha = _sha(experiment_config_sha256, field="experiment_config_sha256")
    if experiment_sha != contract["experiment_file_sha256"]:
        raise GateInputError("experiment file SHA differs from learning-gate binding")
    registration = contract["registration"]
    expected_runs = {(arm, seed) for arm in ARMS for seed in contract["seeds"]}
    support_sequence: list[tuple[str, str]] = []
    masks: dict[tuple[str, str], bool] = {}
    provenance_by_run: dict[tuple[str, int], dict[str, Any]] = {}
    checkpoint_owners: dict[str, tuple[str, int]] = {}
    # [run][game] -> [nonforced total/count, overall total/count].
    summaries: dict[tuple[str, int], dict[str, list[float | int]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0, 0.0, 0])
    )
    first_run: tuple[str, int] | None = None
    current_run: tuple[str, int] | None = None
    current_position = 0
    completed_runs: set[tuple[str, int]] = set()

    def finish_current_run() -> None:
        nonlocal current_run, current_position
        if current_run is None:
            return
        if current_run != first_run and current_position != len(support_sequence):
            raise GateInputError(f"holdout decision support differs for run {current_run}")
        completed_runs.add(current_run)
    for row_number, raw in enumerate(records, 1):
        if not isinstance(raw, Mapping):
            raise GateInputError(f"row {row_number}: record must be an object")
        required = (
            "arm_id", "training_seed", "game_id", "decision_id", "forced",
            "soft_target_policy_ce", "evaluation_split",
            "is_training_game", "experiment_config_sha256",
            "corpus_fingerprint", "training_manifest_sha256",
            "validation_manifest_sha256", "run_provenance",
        )
        missing = [field for field in required if field not in raw]
        if missing:
            raise GateInputError(f"row {row_number}: missing fields {missing}")
        arm, seed = raw["arm_id"], raw["training_seed"]
        if arm not in ARMS or type(seed) is not int or (arm, seed) not in expected_runs:
            raise GateInputError(f"row {row_number}: unregistered arm/seed")
        game, decision = raw["game_id"], raw["decision_id"]
        if not isinstance(game, str) or not game or not isinstance(decision, str) or not decision:
            raise GateInputError(f"row {row_number}: game_id/decision_id must be non-empty strings")
        if type(raw["forced"]) is not bool:
            raise GateInputError(f"row {row_number}: forced must be boolean")
        if raw["evaluation_split"] != "holdout":
            raise GateInputError(f"row {row_number}: row is not soft-policy holdout evidence")
        if raw["is_training_game"] is not False:
            raise GateInputError(f"row {row_number}: training/holdout leakage detected")
        ce = _finite(raw["soft_target_policy_ce"], field="soft_target_policy_ce", row=row_number)
        if ce < 0:
            raise GateInputError(f"row {row_number}: CE must be non-negative")
        for field in (
            "corpus_fingerprint", "training_manifest_sha256", "validation_manifest_sha256"
        ):
            actual = _sha(raw[field], field=field, row=row_number)
            if actual != registration[field]:
                raise GateInputError(f"row {row_number}: {field} differs from registration")
        if _sha(raw["experiment_config_sha256"], field="experiment_config_sha256", row=row_number) != experiment_sha:
            raise GateInputError(f"row {row_number}: experiment config digest mismatch")
        run = (arm, seed)
        if current_run != run:
            finish_current_run()
            if run in completed_runs:
                raise GateInputError(f"row {row_number}: run {run} is not contiguous")
            current_run = run
            current_position = 0
            if first_run is None:
                first_run = run
        decision_key = (game, decision)
        if run == first_run:
            if decision_key in masks:
                raise GateInputError(f"row {row_number}: duplicate decision in first run")
            support_sequence.append(decision_key)
            masks[decision_key] = raw["forced"]
        else:
            if (
                current_position >= len(support_sequence)
                or support_sequence[current_position] != decision_key
            ):
                raise GateInputError(f"row {row_number}: holdout decision support/order differs")
            if masks[decision_key] != raw["forced"]:
                raise GateInputError(f"row {row_number}: forced mask differs across runs")
        current_position += 1

        provenance = raw["run_provenance"]
        if not isinstance(provenance, Mapping):
            raise GateInputError(f"row {row_number}: run_provenance must be an object")
        provenance_required = (
            "checkpoint_sha256", "training_report_sha256", "admission_manifest_sha256",
            "initial_checkpoint_sha256", "resolved_train_config",
            "resolved_train_config_sha256", "graph_history_features",
            "parameter_count", "optimizer_steps", "global_batch_size", "sample_presentations",
        )
        absent = [field for field in provenance_required if field not in provenance]
        if absent:
            raise GateInputError(f"row {row_number}: run_provenance missing {absent}")
        for field in (
            "checkpoint_sha256", "training_report_sha256", "admission_manifest_sha256",
            "initial_checkpoint_sha256", "resolved_train_config_sha256",
        ):
            _sha(provenance[field], field=f"run_provenance.{field}", row=row_number)
        if provenance["initial_checkpoint_sha256"] != registration[
            "initial_checkpoint_sha256_by_arm_seed"
        ][f"{arm}@{seed}"]:
            raise GateInputError(f"row {row_number}: initial checkpoint differs from registration")
        resolved = provenance["resolved_train_config"]
        if not isinstance(resolved, Mapping) or _canonical_sha(resolved) != provenance["resolved_train_config_sha256"]:
            raise GateInputError(f"row {row_number}: resolved config digest mismatch")
        _validate_resolved_config(resolved, arm=contract["arms"][arm], row=row_number)
        if provenance["graph_history_features"] is not True:
            raise GateInputError(f"row {row_number}: graph_history_features must be true")
        numeric_expected = {
            "parameter_count": contract["arms"][arm]["expected_parameters"],
            "optimizer_steps": 250,
            "global_batch_size": 4096,
            "sample_presentations": 1_024_000,
        }
        for field, expected in numeric_expected.items():
            if provenance[field] != expected:
                raise GateInputError(f"row {row_number}: run_provenance.{field} drifted")
        canonical_provenance = dict(provenance)
        if run in provenance_by_run and provenance_by_run[run] != canonical_provenance:
            raise GateInputError(f"row {row_number}: provenance differs within run {run}")
        provenance_by_run[run] = canonical_provenance
        checkpoint = provenance["checkpoint_sha256"]
        if checkpoint in checkpoint_owners and checkpoint_owners[checkpoint] != run:
            raise GateInputError("one trained checkpoint is reused across registered runs")
        checkpoint_owners[checkpoint] = run

        stats = summaries[run][game]
        stats[2] += ce
        stats[3] += 1
        if not raw["forced"]:
            stats[0] += ce
            stats[1] += 1

    finish_current_run()
    if set(provenance_by_run) != expected_runs or completed_runs != expected_runs:
        raise GateInputError("evidence does not contain exactly all 15 registered runs")
    nonforced_count = sum(not masks[key] for key in support_sequence)
    games = {game for game, _decision in support_sequence}
    if len(games) < contract["minimum_games"]:
        raise GateInputError("holdout has fewer games than registered minimum")
    if nonforced_count < contract["minimum_decisions"]:
        raise GateInputError("holdout has fewer nonforced decisions than registered minimum")

    primary_values: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    safety_values: dict[str, dict[int, dict[str, tuple[float, int]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (arm, seed), game_stats in summaries.items():
        if set(game_stats) != games:
            raise GateInputError(f"game support differs for run {(arm, seed)}")
        for game, (nf_total, nf_count, total, count) in game_stats.items():
            if nf_count <= 0 or count <= 0:
                raise GateInputError(f"game {game} has no nonforced support")
            primary_values[arm][seed][game] = float(nf_total) / int(nf_count)
            safety_values[arm][seed][game] = (float(nf_total), int(nf_count))

    rng = random.Random(bootstrap_seed)
    comparisons: dict[str, Any] = {}
    for candidate in (*PRIMARY, *SECONDARY):
        primary_result = _crossed_bootstrap(
            primary_values[candidate], primary_values[REFERENCE],
            samples=bootstrap_samples, rng=rng, game_macro=True,
        )
        overall_result = _crossed_bootstrap(
            safety_values[candidate], safety_values[REFERENCE],
            samples=bootstrap_samples, rng=rng, game_macro=False,
        )
        threshold_pass = primary_result["relative_improvement"] >= contract["improvement"]
        confidence_pass = primary_result["relative_improvement_ci95"][0] > 0.0
        safety_pass = overall_result["relative_regression_ci95"][1] <= contract["regression"]
        comparisons[candidate] = {
            "comparison_role": "primary" if candidate in PRIMARY else "secondary",
            "reference_arm": REFERENCE,
            "primary_nonforced_game_macro": primary_result,
            "nonforced_decision_micro_safety": overall_result,
            "threshold_pass": threshold_pass,
            "confidence_pass": confidence_pass,
            "overall_safety_pass": safety_pass,
            "passed": bool(threshold_pass and confidence_pass and safety_pass),
        }

    descriptive = {
        "arm": "rrt-k0",
        "capacity_matched": False,
        "promotion_eligible": False,
        "nonforced_game_macro_ce": math.fsum(
            value for seed in primary_values["rrt-k0"].values() for value in seed.values()
        ) / (len(contract["seeds"]) * len(games)),
        "warning": "K0 is a smaller-parameter compute control and is excluded from promotion comparisons.",
    }
    primary_passed = [candidate for candidate in PRIMARY if comparisons[candidate]["passed"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if primary_passed else "fail",
        "primary_metric": "game_macro_soft_target_policy_ce_nonforced",
        "experiment_config_sha256": experiment_sha,
        "learning_gate_config_sha256": contract["gate_config_sha256"],
        "support": {
            "arms": len(ARMS),
            "training_seeds": len(contract["seeds"]),
            "runs": len(expected_runs),
            "holdout_games": len(games),
            "decisions_per_run": len(support_sequence),
            "nonforced_decisions_per_run": nonforced_count,
        },
        "registered_thresholds": {
            "minimum_relative_improvement_vs_k1": contract["improvement"],
            "maximum_overall_ce_regression": contract["regression"],
        },
        "comparisons": comparisons,
        "descriptive_compute_control": descriptive,
        "promotion_eligible_passed_arms": primary_passed,
        "bootstrap": {"samples": bootstrap_samples, "seed": bootstrap_seed},
    }


def _load_records(path: Path) -> Iterable[Mapping[str, Any]]:
    """Stream JSONL evidence; retain JSON-list support for compact fixtures."""

    with path.open("r", encoding="utf-8") as stream:
        first = ""
        while True:
            character = stream.read(1)
            if not character:
                return
            if not character.isspace():
                first = character
                break
        stream.seek(0)
        if first == "[":
            value = json.load(stream)
            if not isinstance(value, list):
                raise GateInputError("evidence JSON must be a list")
            yield from value
            return
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateInputError(
                    f"evidence JSONL line {line_number} is malformed: {exc.msg}"
                ) from exc
            yield value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--gate-contract", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        config = json.loads(args.experiment.read_text(encoding="utf-8"))
        gate_contract = json.loads(args.gate_contract.read_text(encoding="utf-8"))
        report = score_learning_gate(
            _load_records(args.evidence),
            config,
            gate_contract,
            experiment_config_sha256=_sha_file(args.experiment),
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
    except (OSError, json.JSONDecodeError, GateInputError) as exc:
        raise SystemExit(str(exc)) from exc
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
