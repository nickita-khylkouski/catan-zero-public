#!/usr/bin/env python3
# ruff: noqa: E402 -- executable adds the sibling tools directory before imports.
"""Bind the complete A0 mechanism verdict to sealed training and calibration.

``a0_gen2b_probe.py`` deliberately stops at the matched training-loss gate.
This tool closes the remaining decision contract from the local RL plans.  It
rehashes and recomputes the sealed A0 result, verifies that scalar and HL
calibration used the exact locked validation games and final checkpoints, then
applies the predeclared global/opening/41+ calibration and policy-drift bounds.
If the scalar control reproduces but HL already fails its primary three-epoch
stability gate, those downstream artifacts are scientifically unnecessary: the
tool emits a complete typed ``retain_scalar_for_a1`` decision and exits nonzero
after writing it.  Scalar non-reproduction remains an invalid, blocked A0.

Example::

    python tools/a0_binding_verdict.py \
      --lock runs/rl_program_20260709/a0_gen2b_hlgauss/a0.lock.json \
      --result runs/rl_program_20260709/a0_gen2b_hlgauss/a0.result.json \
      --scalar-calibration runs/rl_program_20260709/a0_gen2b_hlgauss/scalar/calibration.json \
      --hl-calibration runs/rl_program_20260709/a0_gen2b_hlgauss/hlgauss33/calibration.json \
      --policy-drift runs/rl_program_20260709/a0_gen2b_hlgauss/a0.policy_drift.json \
      --repo-root . \
      --out runs/rl_program_20260709/a0_gen2b_hlgauss/a0.binding.json

No threshold is configurable from the CLI: changing one would create a new
experiment rather than adjudicate the sealed A0 protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import a0_gen2b_probe as a0
import a0_policy_drift as policy_probe
from phase_sliced_value_calibration import load_validation_seed_manifest


SCHEMA = "a0-binding-verdict-v1"
CALIBRATION_SCHEMA = "phase-sliced-value-calibration-v2"
GLOBAL_REGRESSION_LIMIT = 0.02
CRITICAL_SLICE_REGRESSION_LIMIT = 0.05
POLICY_DRIFT_LIMIT = 0.02
_CALIBRATION_METRICS = ("brier", "value_rmse")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise a0.ContractError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise a0.ContractError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise a0.ContractError(f"{label} must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return f"sha256:{a0._sha256(path)}"


def _normalized_sha(value: Any) -> str:
    raw = str(value or "")
    return raw if raw.startswith("sha256:") else f"sha256:{raw}"


def _finite_nonnegative(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise a0.ContractError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(number) or number < 0.0:
        raise a0.ContractError(
            f"{label} must be finite and non-negative, got {number!r}"
        )
    return number


def _positive_int(value: Any, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise a0.ContractError(f"{label} must be an integer, got {value!r}") from exc
    if number <= 0:
        raise a0.ContractError(f"{label} must be positive, got {number}")
    return number


def _relative_change(candidate: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else math.inf
    return candidate / baseline - 1.0


def _json_ratio(value: float) -> float | None:
    """Keep verdict JSON standards-compliant when a zero baseline is exceeded."""

    return value if math.isfinite(value) else None


def _resolve_reference(
    reference: Any,
    *,
    repo_root: Path,
    artifact_root: Path,
    owner_path: Path,
    label: str,
) -> Path:
    raw = str(reference or "")
    if not raw:
        raise a0.ContractError(f"{label} has no path reference")
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [
        repo_root / path,
        artifact_root / path,
        owner_path.parent / path,
    ]
    existing: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved not in existing:
            existing.append(resolved)
    if not existing:
        raise a0.ContractError(f"cannot resolve {label} path {raw!r}")
    if len(existing) > 1:
        hashes = {_sha256(path) for path in existing}
        if len(hashes) != 1:
            raise a0.ContractError(
                f"ambiguous {label} path {raw!r} resolves to different files: "
                f"{[str(path) for path in existing]}"
            )
    return existing[0]


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise a0.ContractError(f"{label} must be an object")
    return value


def _require_calibration_slice(
    calibration: Mapping[str, Any],
    *,
    section: str,
    key: str | None,
    label: str,
) -> Mapping[str, Any]:
    container = _require_mapping(calibration.get(section), f"{label}.{section}")
    value: Any = container if key is None else container.get(key)
    slice_label = section if key is None else f"{section}.{key}"
    result = _require_mapping(value, f"{label}.{slice_label}")
    _positive_int(result.get("n"), f"{label}.{slice_label}.n")
    for metric in _CALIBRATION_METRICS:
        _finite_nonnegative(
            result.get(metric), f"{label}.{slice_label}.{metric}"
        )
    _finite_nonnegative(
        result.get("win_probability_ece"),
        f"{label}.{slice_label}.win_probability_ece",
    )
    reliability = result.get("reliability_bins")
    if not isinstance(reliability, list) or not reliability:
        raise a0.ContractError(
            f"{label}.{slice_label}.reliability_bins must be non-empty"
        )
    if "corr_q_z" not in result:
        raise a0.ContractError(f"{label}.{slice_label} lacks corr_q_z")
    return result


def _validate_readout_provenance(
    calibration: Mapping[str, Any], *, expected: str, label: str
) -> None:
    if calibration.get("schema_version") != CALIBRATION_SCHEMA:
        raise a0.ContractError(
            f"{label}: unsupported calibration schema "
            f"{calibration.get('schema_version')!r}"
        )
    if calibration.get("value_readout") != expected:
        raise a0.ContractError(
            f"{label}: expected value_readout={expected!r}, got "
            f"{calibration.get('value_readout')!r}"
        )
    provenance = _require_mapping(
        calibration.get("readout_provenance"), f"{label}.readout_provenance"
    )
    expected_key = "value" if expected == "scalar" else "value_categorical"
    if provenance.get("requested_readout") != expected:
        raise a0.ContractError(f"{label}: requested-readout provenance drift")
    if provenance.get("model_output_key") != expected_key:
        raise a0.ContractError(
            f"{label}: calibration did not consume {expected_key!r}"
        )
    trained = set(str(item) for item in provenance.get("trained_value_readouts") or [])
    if expected not in trained:
        raise a0.ContractError(f"{label}: selected readout lacks trained provenance")
    if provenance.get("value_training_schema_version") != "value-training-v1":
        raise a0.ContractError(f"{label}: wrong value-training provenance schema")
    _positive_int(provenance.get("optimizer_steps"), f"{label}.optimizer_steps")
    if _positive_int(
        provenance.get("completed_epochs"), f"{label}.completed_epochs"
    ) < 3:
        raise a0.ContractError(f"{label}: checkpoint has <3 completed epochs")
    if expected == "scalar":
        if provenance.get("categorical_training_verified") is not False:
            raise a0.ContractError(
                f"{label}: scalar arm unexpectedly attests categorical training"
            )
        if int(provenance.get("categorical_bins", 0) or 0) != 0:
            raise a0.ContractError(f"{label}: scalar A0 arm unexpectedly has cat bins")
    if expected == "categorical":
        if provenance.get("categorical_training_verified") is not True:
            raise a0.ContractError(f"{label}: categorical training is not verified")
        if "scalar" in trained:
            raise a0.ContractError(f"{label}: scalar auxiliary unexpectedly trained")
        if _positive_int(provenance.get("categorical_bins"), f"{label}.bins") != 33:
            raise a0.ContractError(f"{label}: A0 requires exactly 33 categorical bins")
        if provenance.get("categorical_truncation_class") is not True:
            raise a0.ContractError(f"{label}: A0 requires the truncation class")
        _finite_nonnegative(
            provenance.get("categorical_objective_weight"),
            f"{label}.categorical_objective_weight",
        )
        if float(provenance.get("categorical_objective_weight", 0.0)) <= 0.0:
            raise a0.ContractError(f"{label}: categorical objective weight is not positive")
        if _finite_nonnegative(
            provenance.get("categorical_training_weight_sum"),
            f"{label}.categorical_training_weight_sum",
        ) <= 0.0:
            raise a0.ContractError(f"{label}: categorical training mass is not positive")
        sigma = _finite_nonnegative(
            provenance.get("hlgauss_sigma_ratio"), f"{label}.hlgauss_sigma_ratio"
        )
        if not math.isclose(sigma, 0.75, rel_tol=0.0, abs_tol=1.0e-12):
            raise a0.ContractError(
                f"{label}: A0 requires hlgauss_sigma_ratio=0.75, got {sigma}"
            )


def _validation_seed_evidence(
    calibration: Mapping[str, Any],
    *,
    calibration_path: Path,
    repo_root: Path,
    artifact_root: Path,
    expected_seed_sha: str,
    expected_seed_count: int,
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    row_selection = _require_mapping(
        calibration.get("row_selection"), f"{label}.row_selection"
    )
    if row_selection.get("mode") != "validation_seed_manifest":
        raise a0.ContractError(
            f"{label}: binding A0 calibration must consume the trainer seed manifest"
        )
    if row_selection.get("held_out_filter_applied") is not True:
        raise a0.ContractError(f"{label}: held-out filter was not applied")
    configured = _positive_int(
        row_selection.get("configured_game_seed_count"),
        f"{label}.configured_game_seed_count",
    )
    observed = _positive_int(
        row_selection.get("observed_game_seed_count"),
        f"{label}.observed_game_seed_count",
    )
    observed_rows = _positive_int(
        row_selection.get("observed_row_count"), f"{label}.observed_row_count"
    )
    if configured != expected_seed_count or observed > expected_seed_count:
        raise a0.ContractError(
            f"{label}: validation seed count drift: configured={configured}, "
            f"observed={observed}, locked={expected_seed_count}"
        )
    manifest_path = _resolve_reference(
        row_selection.get("seed_manifest_path"),
        repo_root=repo_root,
        artifact_root=artifact_root,
        owner_path=calibration_path,
        label=f"{label} validation-seed manifest",
    )
    try:
        seeds, manifest_file_sha = load_validation_seed_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        raise a0.ContractError(f"{label}: invalid validation-seed manifest: {exc}") from exc
    declared_file_sha = _normalized_sha(row_selection.get("seed_manifest_sha256"))
    actual_file_sha = _normalized_sha(manifest_file_sha)
    if declared_file_sha != actual_file_sha:
        raise a0.ContractError(
            f"{label}: calibration seed-manifest file SHA drift: "
            f"declared={declared_file_sha}, actual={actual_file_sha}"
        )
    actual_seed_sha = a0._int64_set_sha(seeds)
    if actual_seed_sha != expected_seed_sha:
        raise a0.ContractError(
            f"{label}: validation game-seed set drift: "
            f"actual={actual_seed_sha}, locked={expected_seed_sha}"
        )
    if len(seeds) != expected_seed_count:
        raise a0.ContractError(f"{label}: validation manifest seed count drift")
    return seeds, {
        "manifest": str(manifest_path),
        "manifest_sha256": actual_file_sha,
        "validation_game_seed_set_sha256": actual_seed_sha,
        "validation_game_seed_count": int(len(seeds)),
        "observed_game_seed_count": observed,
        "observed_row_count": observed_rows,
    }


def _calibration_comparisons(
    scalar: Mapping[str, Any], hl: Mapping[str, Any]
) -> tuple[dict[str, Any], bool]:
    locations = (
        ("global", "global", None, GLOBAL_REGRESSION_LIMIT),
        ("opening", "by_phase", "opening_placement", CRITICAL_SLICE_REGRESSION_LIMIT),
        ("legal_41_plus", "by_legal_count_bucket", "41+", CRITICAL_SLICE_REGRESSION_LIMIT),
    )
    comparisons: dict[str, Any] = {}
    gate_pass = True
    for output_name, section, key, limit in locations:
        scalar_slice = _require_calibration_slice(
            scalar, section=section, key=key, label="scalar calibration"
        )
        hl_slice = _require_calibration_slice(
            hl, section=section, key=key, label="HL calibration"
        )
        scalar_n = int(scalar_slice["n"])
        hl_n = int(hl_slice["n"])
        if scalar_n != hl_n:
            raise a0.ContractError(
                f"{output_name}: scalar/HL calibration row counts differ "
                f"({scalar_n} != {hl_n})"
            )
        metric_rows: dict[str, Any] = {}
        for metric in _CALIBRATION_METRICS:
            baseline = _finite_nonnegative(
                scalar_slice.get(metric), f"scalar {output_name}.{metric}"
            )
            candidate = _finite_nonnegative(
                hl_slice.get(metric), f"HL {output_name}.{metric}"
            )
            change = _relative_change(candidate, baseline)
            passed = change <= limit + 1.0e-12
            gate_pass = gate_pass and passed
            metric_rows[metric] = {
                "scalar": baseline,
                "hlgauss33": candidate,
                "relative_change": _json_ratio(change),
                "max_regression": limit,
                "pass": passed,
            }
        comparisons[output_name] = {
            "n": scalar_n,
            "metrics": metric_rows,
        }
    global_improves = any(
        comparisons["global"]["metrics"][metric]["hlgauss33"]
        < comparisons["global"]["metrics"][metric]["scalar"]
        for metric in _CALIBRATION_METRICS
    )
    comparisons["global"]["at_least_one_metric_improves"] = global_improves
    gate_pass = gate_pass and global_improves
    return comparisons, gate_pass


def _validate_policy_drift_artifact(
    artifact: Mapping[str, Any],
    *,
    artifact_path: Path,
    lock: Mapping[str, Any],
    lock_path: Path,
    result: Mapping[str, Any],
    calibration_inputs: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], bool]:
    if artifact.get("schema_version") != policy_probe.SCHEMA:
        raise a0.ContractError(
            f"unsupported A0 policy-drift schema {artifact.get('schema_version')!r}"
        )
    required_equal = {
        "lock_sha256": _sha256(lock_path),
        "input_contract_sha256": lock["input_contract_sha256"],
        "recipe_sha256": lock["recipe_sha256"],
        "seed_contract_sha256": lock["seed_contract_sha256"],
        "matched_common_sha256": lock["arm_contracts"]["matched_common_sha256"],
        "corpus_tree_sha256": lock["corpus_tree_sha256"],
    }
    for key, expected in required_equal.items():
        if artifact.get(key) != expected:
            raise a0.ContractError(
                f"policy-drift artifact {key} drift: "
                f"expected {expected!r}, got {artifact.get(key)!r}"
            )
    thresholds = _require_mapping(
        artifact.get("thresholds"), "policy-drift thresholds"
    )
    if float(thresholds.get("max_absolute_relative_policy_drift", -1.0)) != (
        POLICY_DRIFT_LIMIT
    ):
        raise a0.ContractError("policy-drift artifact changed the 2% threshold")
    validation = _require_mapping(
        artifact.get("validation"), "policy-drift validation"
    )
    if validation.get("validation_game_seed_set_sha256") != lock["validation"][
        "validation_game_seed_set_sha256"
    ]:
        raise a0.ContractError("policy-drift validation seed-set digest drift")
    if int(validation.get("validation_game_seed_count", -1)) != int(
        lock["validation"]["validation_game_seed_count_after_row_cap"]
    ):
        raise a0.ContractError("policy-drift validation seed count drift")
    for arm in ("scalar", "hlgauss33"):
        validation_arm = _require_mapping(
            validation.get(arm), f"policy-drift validation.{arm}"
        )
        if _normalized_sha(validation_arm.get("manifest_sha256")) != (
            calibration_inputs[arm]["manifest_sha256"]
        ):
            raise a0.ContractError(
                f"{arm}: policy/calibration trainer seed-manifest hash differs"
            )

    stages = _require_mapping(artifact.get("stages"), "policy-drift stages")
    if set(stages) != set(policy_probe._STAGES):
        raise a0.ContractError(
            f"policy-drift stages must be {list(policy_probe._STAGES)}"
        )
    recomputed: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    for stage in policy_probe._STAGES:
        row = _require_mapping(stages.get(stage), f"policy-drift stage {stage}")
        scalar = _require_mapping(row.get("scalar"), f"{stage}.scalar")
        hl = _require_mapping(row.get("hlgauss33"), f"{stage}.hlgauss33")
        for arm, metrics in (("scalar", scalar), ("hlgauss33", hl)):
            checkpoint_path = _resolve_reference(
                metrics.get("checkpoint"),
                repo_root=repo_root,
                artifact_root=Path(
                    lock.get("artifact_root_at_seal") or repo_root
                ).resolve(),
                owner_path=artifact_path,
                label=f"policy-drift {stage} {arm} checkpoint",
            )
            actual_sha = _sha256(checkpoint_path)
            if actual_sha != _normalized_sha(metrics.get("checkpoint_sha256")):
                raise a0.ContractError(
                    f"policy-drift {stage} {arm} checkpoint hash drift"
                )
            if stage == "final" and actual_sha != _normalized_sha(
                result["artifacts"][arm]["checkpoint_sha256"]
            ):
                raise a0.ContractError(
                    f"policy-drift final {arm} checkpoint is not the sealed result"
                )
        comparison = policy_probe.compare_stage_metrics(scalar, hl)
        if row.get("comparison") != comparison:
            raise a0.ContractError(
                f"policy-drift {stage} stored comparison is not reproducible"
            )
        recomputed[stage] = comparison
        gates[stage] = bool(comparison["pass"])
    if artifact.get("gates") != gates:
        raise a0.ContractError("policy-drift stored gate map is not reproducible")
    if bool(artifact.get("policy_drift_pass")) != all(gates.values()):
        raise a0.ContractError("policy-drift top-level verdict is not reproducible")
    return {
        "artifact": str(artifact_path),
        "artifact_sha256": _sha256(artifact_path),
        "stages": recomputed,
    }, all(gates.values())


def build_binding_verdict(
    *,
    lock_path: Path,
    result_path: Path,
    scalar_calibration_path: Path | None,
    hl_calibration_path: Path | None,
    policy_drift_path: Path | None,
    repo_root: Path,
) -> dict[str, Any]:
    lock = a0._load_and_verify_lock(lock_path, repo_root)
    result = _load_json(result_path, "A0 training result")
    recomputed = a0._postflight(lock, repo_root)
    if result != recomputed:
        raise a0.ContractError(
            "A0 result does not exactly match the sealed, recomputed postflight"
        )
    if result.get("schema_version") != a0.RESULT_SCHEMA:
        raise a0.ContractError("unsupported A0 training-result schema")
    if result.get("a0_interpretable") is not True or result.get(
        "scalar_reproduces_historical_failure"
    ) is not True:
        raise a0.ContractError(
            "A0 scalar control did not reproduce the historical failure; "
            "the experiment is invalid and cannot hand off an A1 objective"
        )

    thresholds = {
        "global_brier_rmse_max_regression": GLOBAL_REGRESSION_LIMIT,
        "critical_slice_brier_rmse_max_regression": (
            CRITICAL_SLICE_REGRESSION_LIMIT
        ),
        "policy_max_absolute_relative_drift": POLICY_DRIFT_LIMIT,
    }
    sealed_inputs = {
        "lock": str(lock_path),
        "lock_sha256": _sha256(lock_path),
        "training_result": str(result_path),
        "training_result_sha256": _sha256(result_path),
        "input_contract_sha256": lock["input_contract_sha256"],
        "recipe_sha256": lock["recipe_sha256"],
        "seed_contract_sha256": lock["seed_contract_sha256"],
        "matched_common_sha256": lock["arm_contracts"]["matched_common_sha256"],
    }
    training_evidence = {
        "scalar_reproduces_historical_failure": True,
        "scalar_primary_validation_trace": result.get(
            "scalar_primary_validation_trace"
        ),
        "historical_scalar_validation_trace": result.get(
            "historical_scalar_validation_trace"
        ),
        "hl_primary_validation_trace": result.get("hl_primary_validation_trace"),
        "hl_training_stable": bool(result.get("hl_training_stable")),
        "a0_training_loss_gate_pass": bool(
            result.get("a0_training_loss_gate_pass")
        ),
    }
    scalar_checkpoint_sha = _normalized_sha(
        result["artifacts"]["scalar"]["checkpoint_sha256"]
    )
    # A reproduced scalar failure plus an unstable HL trace is already a
    # complete, scientifically useful negative result.  Requiring calibration
    # or policy forwards after the primary mechanism failed would spend compute
    # without any path to adoption.  Emit a typed scalar-retention handoff.
    if result.get("hl_training_stable") is not True:
        return {
            "schema_version": SCHEMA,
            "experiment_id": lock.get("experiment_id"),
            "thresholds": thresholds,
            "sealed_inputs": sealed_inputs,
            "training_evidence": training_evidence,
            "calibration_artifacts": None,
            "calibration_comparison": None,
            "policy_drift": None,
            "gates": {
                "scalar_reproduction": True,
                "hl_training_stability": False,
                "exact_validation_seeds": None,
                "categorical_readout_provenance": None,
                "calibration": None,
                "policy_drift": None,
            },
            "a0_interpretable": True,
            "a0_stage_complete": True,
            "a0_binding_pass": True,
            "hlgauss_adoption_pass": False,
            "decision": {
                "status": "retain_scalar_for_a1",
                "learner_objective": "mse",
                "learner_value_readout": "scalar",
                "mechanism_checkpoint_sha256": scalar_checkpoint_sha,
                "mechanism_checkpoint_is_production_candidate": False,
            },
            "interpretation": (
                "A0 validly rejected HL-Gauss at the primary training-stability "
                "gate; retain scalar MSE for A1. No production checkpoint was selected."
            ),
        }

    missing = [
        name
        for name, path in (
            ("--scalar-calibration", scalar_calibration_path),
            ("--hl-calibration", hl_calibration_path),
            ("--policy-drift", policy_drift_path),
        )
        if path is None
    ]
    if missing:
        raise a0.ContractError(
            "HL remains eligible; final A0 adoption requires " + ", ".join(missing)
        )
    assert scalar_calibration_path is not None
    assert hl_calibration_path is not None
    assert policy_drift_path is not None

    scalar_calibration = _load_json(scalar_calibration_path, "scalar calibration")
    hl_calibration = _load_json(hl_calibration_path, "HL calibration")
    _validate_readout_provenance(
        scalar_calibration, expected="scalar", label="scalar calibration"
    )
    _validate_readout_provenance(
        hl_calibration, expected="categorical", label="HL calibration"
    )

    artifact_root = Path(lock.get("artifact_root_at_seal") or repo_root).resolve()
    calibration_inputs: dict[str, Any] = {}
    expected_seed_sha = str(lock["validation"]["validation_game_seed_set_sha256"])
    expected_seed_count = int(
        lock["validation"]["validation_game_seed_count_after_row_cap"]
    )
    seed_arrays: dict[str, np.ndarray] = {}
    for arm, calibration, calibration_path, label in (
        ("scalar", scalar_calibration, scalar_calibration_path, "scalar"),
        ("hlgauss33", hl_calibration, hl_calibration_path, "HL"),
    ):
        contract = lock["arm_contracts"][arm]
        report_path = a0._resolve(repo_root, str(contract["report"]))
        checkpoint_path = a0._resolve(repo_root, str(contract["checkpoint"]))
        expected_artifacts = _require_mapping(
            result.get("artifacts", {}).get(arm), f"result.artifacts.{arm}"
        )
        report_sha = _sha256(report_path)
        checkpoint_sha = _sha256(checkpoint_path)
        if report_sha != _normalized_sha(expected_artifacts.get("report_sha256")):
            raise a0.ContractError(f"{label}: report SHA-256 drift after postflight")
        if checkpoint_sha != _normalized_sha(
            expected_artifacts.get("checkpoint_sha256")
        ):
            raise a0.ContractError(f"{label}: checkpoint SHA-256 drift after postflight")
        calibrated_checkpoint = _resolve_reference(
            calibration.get("checkpoint"),
            repo_root=repo_root,
            artifact_root=artifact_root,
            owner_path=calibration_path,
            label=f"{label} calibrated checkpoint",
        )
        if _sha256(calibrated_checkpoint) != checkpoint_sha:
            raise a0.ContractError(
                f"{label}: calibration checkpoint does not match sealed final checkpoint"
            )
        seeds, seed_evidence = _validation_seed_evidence(
            calibration,
            calibration_path=calibration_path,
            repo_root=repo_root,
            artifact_root=artifact_root,
            expected_seed_sha=expected_seed_sha,
            expected_seed_count=expected_seed_count,
            label=f"{label} calibration",
        )
        seed_arrays[arm] = seeds
        calibration_inputs[arm] = {
            "calibration": str(calibration_path),
            "calibration_sha256": _sha256(calibration_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "report": str(report_path),
            "report_sha256": report_sha,
            **seed_evidence,
        }
    if not np.array_equal(seed_arrays["scalar"], seed_arrays["hlgauss33"]):
        raise a0.ContractError("scalar and HL calibration validation seeds differ")
    if (
        calibration_inputs["scalar"]["observed_game_seed_count"]
        != calibration_inputs["hlgauss33"]["observed_game_seed_count"]
    ):
        raise a0.ContractError(
            "scalar and HL calibration observed game-seed counts differ"
        )
    if (
        scalar_calibration.get("shard_dir") != hl_calibration.get("shard_dir")
    ):
        raise a0.ContractError("scalar and HL calibration shard_dir values differ")
    if not str(scalar_calibration.get("shard_dir") or ""):
        raise a0.ContractError("calibration artifacts have no shard_dir provenance")
    if int(scalar_calibration["global"]["n"]) != int(
        scalar_calibration["row_selection"]["observed_row_count"]
    ) or int(hl_calibration["global"]["n"]) != int(
        hl_calibration["row_selection"]["observed_row_count"]
    ):
        raise a0.ContractError("calibration global row count/provenance mismatch")

    calibration, calibration_pass = _calibration_comparisons(
        scalar_calibration, hl_calibration
    )
    # Prove the categorical distribution itself was scored on every binding
    # slice rather than only converting a scalar/default readout.
    for section, key, label in (
        ("global", None, "global"),
        ("by_phase", "opening_placement", "opening"),
        ("by_legal_count_bucket", "41+", "legal_41_plus"),
    ):
        hl_slice = _require_calibration_slice(
            hl_calibration, section=section, key=key, label="HL calibration"
        )
        if int(hl_slice.get("categorical_score_n", -1)) != int(hl_slice["n"]):
            raise a0.ContractError(
                f"HL {label}: categorical score count does not cover the slice"
            )
    policy_artifact = _load_json(policy_drift_path, "A0 policy-drift artifact")
    policy, policy_pass = _validate_policy_drift_artifact(
        policy_artifact,
        artifact_path=policy_drift_path,
        lock=lock,
        lock_path=lock_path,
        result=result,
        calibration_inputs=calibration_inputs,
        repo_root=repo_root,
    )
    gates = {
        "scalar_reproduction": True,
        "hl_training_stability": True,
        "exact_validation_seeds": True,
        "categorical_readout_provenance": True,
        "calibration": calibration_pass,
        "policy_drift": policy_pass,
    }
    hl_adoption_pass = all(gates.values())
    selected_checkpoint_sha = (
        calibration_inputs["hlgauss33"]["checkpoint_sha256"]
        if hl_adoption_pass
        else scalar_checkpoint_sha
    )
    return {
        "schema_version": SCHEMA,
        "experiment_id": lock.get("experiment_id"),
        "thresholds": thresholds,
        "sealed_inputs": sealed_inputs,
        "training_evidence": training_evidence,
        "calibration_artifacts": calibration_inputs,
        "calibration_comparison": calibration,
        "policy_drift": policy,
        "policy_metric_semantics": {
            "unforced_policy_loss": (
                "posthoc validation policy_loss on locked games after filtering "
                "legal_action_count > 1"
            ),
            "prior_kl_model_prior_mean": (
                "validation KL(model_policy || stored prior_policy) on rows with "
                "recorded priors"
            ),
        },
        "gates": gates,
        "a0_interpretable": True,
        "a0_stage_complete": True,
        "a0_binding_pass": True,
        "hlgauss_adoption_pass": hl_adoption_pass,
        "decision": {
            "status": (
                "adopt_hlgauss_for_a1"
                if hl_adoption_pass
                else "retain_scalar_for_a1"
            ),
            "learner_objective": "hlgauss" if hl_adoption_pass else "mse",
            "learner_value_readout": (
                "categorical" if hl_adoption_pass else "scalar"
            ),
            "mechanism_checkpoint_sha256": selected_checkpoint_sha,
            "mechanism_checkpoint_is_production_candidate": False,
        },
        "interpretation": (
            "A0 mechanism gate only; this does not select a production checkpoint."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--scalar-calibration", default="")
    parser.add_argument("--hl-calibration", default="")
    parser.add_argument("--policy-drift", default="")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing verdict artifact after re-verifying every input",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.out).resolve()
    if output.exists() and not args.force:
        raise a0.ContractError(
            f"binding verdict already exists: {output}; pass --force to replace"
        )
    verdict = build_binding_verdict(
        lock_path=Path(args.lock).resolve(),
        result_path=Path(args.result).resolve(),
        scalar_calibration_path=(
            Path(args.scalar_calibration).resolve() if args.scalar_calibration else None
        ),
        hl_calibration_path=(
            Path(args.hl_calibration).resolve() if args.hl_calibration else None
        ),
        policy_drift_path=(
            Path(args.policy_drift).resolve() if args.policy_drift else None
        ),
        repo_root=Path(args.repo_root).resolve(),
    )
    a0._write_json_atomic(output, verdict)
    print(json.dumps({"out": str(output), **verdict}, indent=2, sort_keys=True))
    if not verdict["hlgauss_adoption_pass"]:
        raise a0.ContractError(
            "A0 complete: HL-Gauss was not adopted; typed scalar-retention "
            "decision was written for the A1 handoff"
        )


if __name__ == "__main__":
    main()
