#!/usr/bin/env python3
"""Score the preregistered four-arm Transformer fixed-K learning screen."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.rnd_e3_learning_gate import _crossed_bootstrap, _load_records  # noqa: E402


SCHEMA_VERSION = "catan-zero-transformer-think-learning-gate/v1"
CONTRACT_SCHEMA = "catan-zero-transformer-think-learning-gate-contract/v1"
EVIDENCE_SCHEMA = "catan-zero-transformer-think-holdout-evidence/v1"
ARMS = (
    "transformer-k0",
    "think-transformer-k1",
    "think-transformer-k2",
    "think-transformer-k4",
)
REFERENCE = "think-transformer-k1"
PRIMARY = ("think-transformer-k2", "think-transformer-k4")
DESCRIPTIVE = "transformer-k0"
SEEDS = (101, 103, 107)
EXPECTED_PARAMETERS = {
    "transformer-k0": 35_041_353,
    "think-transformer-k1": 40_793_673,
    "think-transformer-k2": 40_793_673,
    "think-transformer-k4": 40_793_673,
}
EXPECTED_STEPS = dict(zip(ARMS, (0, 1, 2, 4), strict=True))
FROZEN_HOLDOUT_GAMES = 596
FROZEN_DECISIONS_PER_RUN = 146_517
FROZEN_BOOTSTRAP_SAMPLES = 10_000
FROZEN_BOOTSTRAP_SEED = 20260711
FROZEN_POINT_IMPROVEMENT = 0.02
FROZEN_SAFETY_REGRESSION = 0.005
FROZEN_OPTIMIZER_STEPS = 250
FROZEN_GLOBAL_BATCH_SIZE = 4096
FROZEN_SAMPLE_PRESENTATIONS = 1_024_000
FROZEN_INCUMBENT_CHECKPOINT_SHA256 = (
    "89aa133d629e747021bc725f2ad63e0563f3b76e71f0dd563f056c6de8f77ebb"
)


class GateInputError(ValueError):
    """Evidence cannot support the preregistered Transformer conclusion."""


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


def _sha(value: Any, *, field: str, row: int | None = None) -> str:
    prefix = f"row {row}: " if row is not None else ""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GateInputError(f"{prefix}{field} must be a lowercase SHA256 digest")
    return value


def _finite(value: Any, *, field: str, row: int) -> float:
    if isinstance(value, bool):
        raise GateInputError(f"row {row}: {field} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GateInputError(f"row {row}: {field} must be finite") from exc
    if not math.isfinite(result):
        raise GateInputError(f"row {row}: {field} must be finite")
    return result


def _validate_contract(
    experiment: Mapping[str, Any], gate: Mapping[str, Any], *, experiment_file_sha: str
) -> dict[str, Any]:
    experiment_semantic = dict(experiment)
    experiment_declared = experiment_semantic.pop("config_sha256", None)
    if (
        experiment.get("status") != "registered_ready"
        or experiment_declared != _canonical_sha(experiment_semantic)
    ):
        raise GateInputError("Transformer-think registration is invalid")
    gate_semantic = dict(gate)
    gate_declared = gate_semantic.pop("config_sha256", None)
    if (
        gate.get("schema_version") != CONTRACT_SCHEMA
        or gate_declared != _canonical_sha(gate_semantic)
    ):
        raise GateInputError("Transformer-think gate contract is invalid")
    frozen = {
        "experiment_file_sha256": experiment_file_sha,
        "experiment_semantic_sha256": experiment_declared,
        "scorer": "tools/rnd_transformer_think_learning_gate.py",
        "primary_reference_arm": REFERENCE,
        "primary_candidate_arms": list(PRIMARY),
        "descriptive_compute_control_arms": [DESCRIPTIVE],
        "seeds": list(SEEDS),
        "runs": len(ARMS) * len(SEEDS),
        "holdout_games": FROZEN_HOLDOUT_GAMES,
        "decisions_per_run": FROZEN_DECISIONS_PER_RUN,
        "primary_metric": "game_macro_soft_target_policy_ce_nonforced",
        "uncertainty": "paired_crossed_bootstrap_training_seed_and_common_holdout_game",
        "bootstrap_samples": FROZEN_BOOTSTRAP_SAMPLES,
        "bootstrap_seed": FROZEN_BOOTSTRAP_SEED,
        "minimum_relative_improvement_vs_k1": FROZEN_POINT_IMPROVEMENT,
        "require_primary_improvement_ci_lower_above_zero": True,
        "maximum_nonforced_decision_micro_ce_regression": FROZEN_SAFETY_REGRESSION,
        "require_safety_regression_ci_upper_within_limit": True,
        "optimizer_steps": FROZEN_OPTIMIZER_STEPS,
        "global_batch_size": FROZEN_GLOBAL_BATCH_SIZE,
        "sample_presentations_per_arm_seed": FROZEN_SAMPLE_PRESENTATIONS,
    }
    for field, expected in frozen.items():
        if gate.get(field) != expected:
            raise GateInputError(f"gate contract {field} differs from frozen source")
    if gate.get("scorer_source_sha256") != _sha_file(Path(__file__).resolve()):
        raise GateInputError("gate scorer source differs from preregistration")
    root = Path(__file__).resolve().parents[1]
    source_bindings = {
        "exporter_source_sha256": root / "tools/rnd_transformer_think_holdout_export.py",
        "exporter_engine_source_sha256": root / "tools/rnd_e3_holdout_export.py",
        "exporter_helper_source_sha256": root / "tools/rnd_topology_holdout_export.py",
        "scorer_engine_source_sha256": root / "tools/rnd_e3_learning_gate.py",
    }
    for field, path in source_bindings.items():
        if gate.get(field) != _sha_file(path):
            raise GateInputError(f"gate {field} differs from live source")
    evidence_path = root / "configs/rnd/transformer_think_a1_screen_20260711/evidence_export.v1.json"
    if gate.get("evidence_export_contract_file_sha256") != _sha_file(evidence_path):
        raise GateInputError("gate evidence contract file binding differs")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence_semantic = dict(evidence)
    evidence_declared = evidence_semantic.pop("config_sha256", None)
    if (
        evidence_declared != _canonical_sha(evidence_semantic)
        or gate.get("evidence_export_contract_semantic_sha256") != evidence_declared
    ):
        raise GateInputError("gate evidence contract semantic binding differs")
    arms = experiment.get("arms")
    if not isinstance(arms, list) or len(arms) != len(ARMS):
        raise GateInputError("registration must contain exactly four arms")
    by_id = {item.get("arm_id"): item for item in arms if isinstance(item, Mapping)}
    if set(by_id) != set(ARMS):
        raise GateInputError("registration arm identities differ")
    roles = {
        DESCRIPTIVE: ("compute_control_only", False),
        REFERENCE: ("capacity_matched_reference", False),
        PRIMARY[0]: ("primary_candidate", True),
        PRIMARY[1]: ("primary_candidate", True),
    }
    for arm_id in ARMS:
        arm = by_id[arm_id]
        role, eligible = roles[arm_id]
        if (
            arm.get("latent_deliberation_steps") != EXPECTED_STEPS[arm_id]
            or arm.get("expected_parameters") != EXPECTED_PARAMETERS[arm_id]
            or arm.get("comparison_role") != role
            or arm.get("promotion_eligible") is not eligible
        ):
            raise GateInputError(f"registered scientific role drift for {arm_id}")
    common = experiment.get("common")
    frozen_common = {
        "hidden_size": 640,
        "state_layers": 6,
        "attention_heads": 8,
        "state_trunk": "transformer",
        "latent_deliberation_slots": 8,
        "frozen_incumbent_checkpoint_sha256": FROZEN_INCUMBENT_CHECKPOINT_SHA256,
    }
    if not isinstance(common, Mapping) or any(
        common.get(field) != value for field, value in frozen_common.items()
    ):
        raise GateInputError("registered Transformer h640/L6 identity anchor drifted")
    matrix = experiment.get("run_matrix")
    if not isinstance(matrix, Mapping) or matrix.get("seeds") != list(SEEDS):
        raise GateInputError("registration seeds differ")
    comparison = experiment.get("comparison_contract")
    expected_comparison = {
        "primary_reference_arm": REFERENCE,
        "primary_candidate_arms": list(PRIMARY),
        "compute_control_arms": [DESCRIPTIVE],
        "primary_metric": "game_macro_soft_target_policy_ce_nonforced",
        "minimum_relative_improvement_vs_k1": FROZEN_POINT_IMPROVEMENT,
        "maximum_nonforced_decision_micro_ce_regression": FROZEN_SAFETY_REGRESSION,
        "uncertainty": "paired_crossed_bootstrap_training_seed_and_common_holdout_game",
    }
    if not isinstance(comparison, Mapping) or any(
        comparison.get(field) != value
        for field, value in expected_comparison.items()
    ):
        raise GateInputError("registered comparison contract differs from frozen gate")
    registration = experiment.get("registration")
    if not isinstance(registration, Mapping):
        raise GateInputError("registration hashes are missing")
    for field in ("corpus_fingerprint", "training_manifest_sha256", "validation_manifest_sha256"):
        _sha(registration.get(field), field=f"registration.{field}")
    init = registration.get("initial_checkpoint_sha256_by_arm_seed")
    expected_keys = {f"{arm}@{seed}" for arm in ARMS for seed in SEEDS}
    if not isinstance(init, Mapping) or set(init) != expected_keys:
        raise GateInputError("registered initialization family is incomplete")
    return {
        "arms": by_id,
        "registration": registration,
        "gate_sha": gate_declared,
        "evidence_file_sha": gate["evidence_export_contract_file_sha256"],
        "evidence_semantic_sha": evidence_declared,
        **{field: gate[field] for field in source_bindings},
    }


def _validate_resolved_config(resolved: Mapping[str, Any], *, arm: str, row: int) -> None:
    fields = resolved.get("fields") if isinstance(resolved, Mapping) else None
    if resolved.get("pipeline") != "train" or not isinstance(fields, Mapping):
        raise GateInputError(f"row {row}: resolved TrainConfig envelope is invalid")
    required = {
        "hidden_size": 640,
        "graph_layers": 6,
        "attention_heads": 8,
        "entity_state_trunk": "transformer",
        "latent_deliberation_steps": EXPECTED_STEPS[arm],
        "latent_deliberation_slots": 8,
        "max_steps": FROZEN_OPTIMIZER_STEPS,
        "batch_size": 1024,
        "grad_accum_steps": 4,
        "mask_hidden_info": True,
    }
    for field, expected in required.items():
        if fields.get(field) != expected:
            raise GateInputError(f"row {row}: resolved TrainConfig {field} drifted")


def score_learning_gate(
    records: Iterable[Mapping[str, Any]],
    experiment: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    experiment_config_sha256: str,
    bootstrap_samples: int = FROZEN_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = FROZEN_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if bootstrap_samples != FROZEN_BOOTSTRAP_SAMPLES or bootstrap_seed != FROZEN_BOOTSTRAP_SEED:
        raise GateInputError("bootstrap settings differ from frozen source")
    experiment_sha = _sha(experiment_config_sha256, field="experiment_config_sha256")
    contract = _validate_contract(experiment, gate, experiment_file_sha=experiment_sha)
    registration = contract["registration"]
    expected_runs = {(arm, seed) for arm in ARMS for seed in SEEDS}
    support: list[tuple[str, str]] = []
    forced_masks: dict[tuple[str, str], bool] = {}
    completed: set[tuple[str, int]] = set()
    provenance_by_run: dict[tuple[str, int], dict[str, Any]] = {}
    checkpoints: dict[str, tuple[str, int]] = {}
    summaries: dict[tuple[str, int], dict[str, list[float | int]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0, 0.0, 0])
    )
    first_run: tuple[str, int] | None = None
    current_run: tuple[str, int] | None = None
    position = 0

    def finish() -> None:
        nonlocal current_run, position
        if current_run is None:
            return
        if current_run != first_run and position != len(support):
            raise GateInputError(f"holdout support differs for run {current_run}")
        completed.add(current_run)

    for row_number, raw in enumerate(records, 1):
        if not isinstance(raw, Mapping):
            raise GateInputError(f"row {row_number}: evidence must be an object")
        required = {
            "schema_version", "arm_id", "training_seed", "game_id", "decision_id",
            "forced", "soft_target_policy_ce", "public_masked", "evaluation_split",
            "is_training_game", "experiment_config_sha256", "corpus_fingerprint",
            "training_manifest_sha256", "validation_manifest_sha256", "run_provenance",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise GateInputError(f"row {row_number}: missing fields {missing}")
        if raw["schema_version"] != EVIDENCE_SCHEMA or raw["public_masked"] is not True:
            raise GateInputError(f"row {row_number}: evidence schema/public mask is invalid")
        run = (raw["arm_id"], raw["training_seed"])
        if run not in expected_runs:
            raise GateInputError(f"row {row_number}: unregistered arm/seed")
        if raw["evaluation_split"] != "holdout" or raw["is_training_game"] is not False:
            raise GateInputError(f"row {row_number}: training/holdout leakage")
        if type(raw["forced"]) is not bool:
            raise GateInputError(f"row {row_number}: forced must be boolean")
        ce = _finite(raw["soft_target_policy_ce"], field="soft_target_policy_ce", row=row_number)
        if ce < 0:
            raise GateInputError(f"row {row_number}: CE must be nonnegative")
        for field in ("corpus_fingerprint", "training_manifest_sha256", "validation_manifest_sha256"):
            if raw[field] != registration[field]:
                raise GateInputError(f"row {row_number}: {field} differs")
        if raw["experiment_config_sha256"] != experiment_sha:
            raise GateInputError(f"row {row_number}: experiment digest differs")
        if current_run != run:
            finish()
            if run in completed:
                raise GateInputError(f"row {row_number}: run {run} is not contiguous")
            current_run, position = run, 0
            if first_run is None:
                first_run = run
        game, decision = raw["game_id"], raw["decision_id"]
        if not isinstance(game, str) or not game or not isinstance(decision, str) or not decision:
            raise GateInputError(f"row {row_number}: invalid decision identity")
        key = (game, decision)
        if run == first_run:
            if key in forced_masks:
                raise GateInputError(f"row {row_number}: duplicate first-run decision")
            support.append(key)
            forced_masks[key] = raw["forced"]
        elif position >= len(support) or support[position] != key or forced_masks[key] != raw["forced"]:
            raise GateInputError(f"row {row_number}: paired support/order/mask differs")
        position += 1
        provenance = raw["run_provenance"]
        if not isinstance(provenance, Mapping):
            raise GateInputError(f"row {row_number}: run provenance is invalid")
        bindings = {
            "evidence_export_contract_sha256": contract["evidence_file_sha"],
            "evidence_export_contract_semantic_sha256": contract["evidence_semantic_sha"],
            "exporter_source_sha256": contract["exporter_source_sha256"],
            "exporter_engine_source_sha256": contract["exporter_engine_source_sha256"],
            "exporter_helper_source_sha256": contract["exporter_helper_source_sha256"],
        }
        for field, expected in bindings.items():
            if provenance.get(field) != expected:
                raise GateInputError(f"row {row_number}: provenance {field} differs")
        if provenance.get("schema_version") != "catan-zero-transformer-think-run-provenance/v1":
            raise GateInputError(f"row {row_number}: run provenance schema differs")
        if provenance.get("initial_checkpoint_sha256") != registration[
            "initial_checkpoint_sha256_by_arm_seed"
        ][f"{run[0]}@{run[1]}"]:
            raise GateInputError(f"row {row_number}: initialization differs")
        resolved = provenance.get("resolved_train_config")
        if not isinstance(resolved, Mapping) or provenance.get("resolved_train_config_sha256") != _canonical_sha(resolved):
            raise GateInputError(f"row {row_number}: resolved config digest differs")
        _validate_resolved_config(resolved, arm=run[0], row=row_number)
        numeric = {
            "parameter_count": EXPECTED_PARAMETERS[run[0]],
            "optimizer_steps": FROZEN_OPTIMIZER_STEPS,
            "global_batch_size": FROZEN_GLOBAL_BATCH_SIZE,
            "sample_presentations": FROZEN_SAMPLE_PRESENTATIONS,
        }
        for field, expected in numeric.items():
            if provenance.get(field) != expected:
                raise GateInputError(f"row {row_number}: provenance {field} drifted")
        canonical = dict(provenance)
        if run in provenance_by_run and provenance_by_run[run] != canonical:
            raise GateInputError(f"row {row_number}: provenance changes within run")
        provenance_by_run[run] = canonical
        checkpoint = _sha(provenance.get("checkpoint_sha256"), field="checkpoint_sha256", row=row_number)
        if checkpoint in checkpoints and checkpoints[checkpoint] != run:
            raise GateInputError("one checkpoint is reused across runs")
        checkpoints[checkpoint] = run
        stats = summaries[run][game]
        stats[2] += ce
        stats[3] += 1
        if not raw["forced"]:
            stats[0] += ce
            stats[1] += 1
    finish()
    if completed != expected_runs or set(provenance_by_run) != expected_runs:
        raise GateInputError("evidence does not contain exactly all 12 runs")
    if len(support) != FROZEN_DECISIONS_PER_RUN:
        raise GateInputError("decisions per run differ from frozen 146,517")
    games = {game for game, _ in support}
    if len(games) != FROZEN_HOLDOUT_GAMES:
        raise GateInputError("holdout games differ from frozen 596")

    primary_values: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    safety_values: dict[str, dict[int, dict[str, tuple[float, int]]]] = defaultdict(lambda: defaultdict(dict))
    for (arm, seed), game_stats in summaries.items():
        if set(game_stats) != games:
            raise GateInputError(f"game support differs for {(arm, seed)}")
        for game, (nf_total, nf_count, _total, count) in game_stats.items():
            if nf_count <= 0 or count <= 0:
                raise GateInputError(f"game {game} lacks nonforced support")
            primary_values[arm][seed][game] = float(nf_total) / int(nf_count)
            safety_values[arm][seed][game] = (float(nf_total), int(nf_count))

    rng = random.Random(bootstrap_seed)
    comparisons: dict[str, Any] = {}
    for candidate in PRIMARY:
        primary = _crossed_bootstrap(
            primary_values[candidate], primary_values[REFERENCE],
            samples=bootstrap_samples, rng=rng, game_macro=True,
        )
        safety = _crossed_bootstrap(
            safety_values[candidate], safety_values[REFERENCE],
            samples=bootstrap_samples, rng=rng, game_macro=False,
        )
        point_pass = primary["relative_improvement"] >= FROZEN_POINT_IMPROVEMENT
        confidence_pass = primary["relative_improvement_ci95"][0] > 0.0
        safety_pass = safety["relative_regression_ci95"][1] <= FROZEN_SAFETY_REGRESSION
        comparisons[candidate] = {
            "comparison_role": "primary",
            "reference_arm": REFERENCE,
            "primary_nonforced_game_macro": primary,
            "nonforced_decision_micro_safety": safety,
            "point_threshold_pass": point_pass,
            "confidence_pass": confidence_pass,
            "safety_pass": safety_pass,
            "passed": bool(point_pass and confidence_pass and safety_pass),
        }
    passed = [arm for arm in PRIMARY if comparisons[arm]["passed"]]
    descriptive_ce = math.fsum(
        value for seed_values in primary_values[DESCRIPTIVE].values() for value in seed_values.values()
    ) / (len(SEEDS) * len(games))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if passed else "fail",
        "primary_metric": "game_macro_soft_target_policy_ce_nonforced",
        "experiment_config_sha256": experiment_sha,
        "learning_gate_config_sha256": contract["gate_sha"],
        "support": {
            "arms": len(ARMS), "training_seeds": len(SEEDS), "runs": len(expected_runs),
            "holdout_games": len(games), "decisions_per_run": len(support),
            "nonforced_decisions_per_run": sum(not forced_masks[key] for key in support),
        },
        "registered_thresholds": {
            "minimum_relative_improvement_vs_k1": FROZEN_POINT_IMPROVEMENT,
            "maximum_nonforced_decision_micro_ce_regression": FROZEN_SAFETY_REGRESSION,
        },
        "comparisons": comparisons,
        "descriptive_compute_control": {
            "arm": DESCRIPTIVE, "capacity_matched": False, "promotion_eligible": False,
            "nonforced_game_macro_ce": descriptive_ce,
            "warning": "K0 is a smaller-capacity descriptive control and is excluded from promotion.",
        },
        "promotion_eligible_passed_arms": passed,
        "bootstrap": {"samples": bootstrap_samples, "seed": bootstrap_seed},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--gate-contract", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=FROZEN_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=FROZEN_BOOTSTRAP_SEED)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        experiment = json.loads(args.experiment.read_text(encoding="utf-8"))
        gate = json.loads(args.gate_contract.read_text(encoding="utf-8"))
        report = score_learning_gate(
            _load_records(args.evidence), experiment, gate,
            experiment_config_sha256=_sha_file(args.experiment),
            bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed,
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
