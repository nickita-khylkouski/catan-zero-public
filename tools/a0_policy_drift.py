#!/usr/bin/env python3
# ruff: noqa: E402 -- executable adds the sibling tools directory before imports.
"""Measure A0 policy drift post hoc on the exact locked validation games.

The sealed A0 jobs predate dedicated unforced-policy telemetry.  Re-running
them would destroy the exact mechanism replication, while treating the
trainer's forced-row-weighted ``policy_loss`` as unforced would be dishonest.
This read-only probe loads the locked gen2 memmap and trainer validation-seed
manifests, selects only roots with more than one legal action, and evaluates
both arms' epoch 1/2/3 and final checkpoints with the trainer's exact policy
target/weight semantics.  It records policy CE and KL(model || stored prior)
and enforces the predeclared 2% absolute relative-drift ceiling.
"""

from __future__ import annotations

import argparse
import gc
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
import train_bc
from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from phase_sliced_value_calibration import load_validation_seed_manifest


SCHEMA = "a0-policy-drift-v1"
POLICY_DRIFT_LIMIT = 0.02
_STAGES = ("epoch1", "epoch2", "epoch3", "final")


def _sha256(path: Path) -> str:
    return f"sha256:{a0._sha256(path)}"


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


def _relative_drift(candidate: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else math.inf
    return abs(candidate - baseline) / baseline


def compare_stage_metrics(
    scalar: Mapping[str, Any], hl: Mapping[str, Any]
) -> dict[str, Any]:
    for key in ("samples", "accuracy_active_count", "prior_kl_rows"):
        scalar_count = _positive_int(scalar.get(key), f"scalar {key}")
        hl_count = _positive_int(hl.get(key), f"HL {key}")
        if scalar_count != hl_count:
            raise a0.ContractError(
                f"scalar/HL posthoc row counts differ for {key}: "
                f"{scalar_count} != {hl_count}"
            )
    comparisons: dict[str, Any] = {}
    passed = True
    for output_name, key in (
        ("unforced_policy_loss", "policy_loss"),
        ("prior_kl_model_prior_mean", "prior_kl_model_prior_mean"),
    ):
        baseline = _finite_nonnegative(scalar.get(key), f"scalar {key}")
        candidate = _finite_nonnegative(hl.get(key), f"HL {key}")
        drift = _relative_drift(candidate, baseline)
        metric_pass = drift <= POLICY_DRIFT_LIMIT + 1.0e-12
        passed = passed and metric_pass
        comparisons[output_name] = {
            "scalar": baseline,
            "hlgauss33": candidate,
            "absolute_relative_drift": drift if math.isfinite(drift) else None,
            "max_absolute_relative_drift": POLICY_DRIFT_LIMIT,
            "pass": metric_pass,
        }
    return {
        "unforced_rows": int(scalar["samples"]),
        "active_policy_rows": int(scalar["accuracy_active_count"]),
        "prior_kl_rows": int(scalar["prior_kl_rows"]),
        "metrics": comparisons,
        "pass": passed,
    }


def _checkpoint_stages(checkpoint: Path) -> dict[str, Path]:
    return {
        "epoch1": train_bc._epoch_checkpoint_path(str(checkpoint), 1).resolve(),
        "epoch2": train_bc._epoch_checkpoint_path(str(checkpoint), 2).resolve(),
        "epoch3": train_bc._epoch_checkpoint_path(str(checkpoint), 3).resolve(),
        "final": checkpoint.resolve(),
    }


def _trainer_seed_manifest(report: Path) -> Path:
    return report.with_suffix(".validation_seeds.json").resolve()


def _load_exact_validation_seeds(
    lock: Mapping[str, Any], repo_root: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    expected_sha = str(lock["validation"]["validation_game_seed_set_sha256"])
    expected_count = int(
        lock["validation"]["validation_game_seed_count_after_row_cap"]
    )
    arrays: dict[str, np.ndarray] = {}
    evidence: dict[str, Any] = {}
    for arm in ("scalar", "hlgauss33"):
        report = a0._resolve(repo_root, str(lock["arm_contracts"][arm]["report"]))
        manifest = _trainer_seed_manifest(report)
        try:
            seeds, file_sha = load_validation_seed_manifest(manifest)
        except (OSError, ValueError) as exc:
            raise a0.ContractError(
                f"{arm}: invalid trainer validation-seed manifest: {exc}"
            ) from exc
        actual_sha = a0._int64_set_sha(seeds)
        if actual_sha != expected_sha or len(seeds) != expected_count:
            raise a0.ContractError(
                f"{arm}: trainer validation seed set/count drift: "
                f"sha={actual_sha}, count={len(seeds)}"
            )
        arrays[arm] = seeds
        evidence[arm] = {
            "manifest": str(manifest),
            "manifest_sha256": f"sha256:{file_sha}",
        }
    if not np.array_equal(arrays["scalar"], arrays["hlgauss33"]):
        raise a0.ContractError("scalar/HL trainer validation seeds differ")
    evidence.update(
        {
            "validation_game_seed_set_sha256": expected_sha,
            "validation_game_seed_count": expected_count,
        }
    )
    return arrays["scalar"], evidence


def _policy_weights(data: Any, recipe: Mapping[str, Any]) -> np.ndarray:
    return train_bc.build_sample_weights(
        data,
        teacher_weights={
            str(key): float(value)
            for key, value in dict(recipe.get("teacher_weights") or {}).items()
        },
        phase_weights={
            str(key): float(value)
            for key, value in dict(recipe.get("phase_weights") or {}).items()
        },
        forced_action_weight=float(recipe["forced_action_weight"]),
        winner_sample_weight=float(recipe["winner_sample_weight"]),
        loser_sample_weight=float(recipe["loser_sample_weight"]),
        vp_margin_weight=float(recipe["vp_margin_weight"]),
        vps_to_win=int(recipe["vps_to_win"]),
        per_game_policy_weight=bool(recipe.get("per_game_policy_weight", False)),
        per_game_policy_weight_mode=str(
            recipe.get("per_game_policy_weight_mode", "equal")
        ),
    )


def _value_weights(data: Any, recipe: Mapping[str, Any]) -> np.ndarray:
    value_phase = recipe.get("value_phase_weights") or recipe.get("phase_weights")
    return train_bc.build_value_sample_weights(
        data,
        phase_weights={
            str(key): float(value)
            for key, value in dict(value_phase or {}).items()
        },
        forced_row_value_weight=float(recipe["forced_row_value_weight"]),
        per_game_value_weight=bool(recipe["per_game_value_weight"]),
        per_game_value_weight_mode=str(recipe["per_game_value_weight_mode"]),
    )


def _evaluate_checkpoint(
    checkpoint: Path,
    *,
    data: Any,
    unforced_indices: np.ndarray,
    policy_weights: np.ndarray,
    value_weights: np.ndarray,
    recipe: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise a0.ContractError(f"missing saved A0 checkpoint: {checkpoint}")
    policy = EntityGraphPolicy.load(checkpoint, device=device)
    try:
        metrics = train_bc.evaluate_bc_batches(
            policy,
            data,
            unforced_indices,
            policy_weights,
            value_weights,
            int(recipe["batch_size"]),
            float(recipe["soft_target_temperature"]),
            float(recipe["soft_target_weight"]),
            str(recipe["soft_target_source"]),
            float(recipe["soft_target_min_legal_coverage"]),
            float(recipe["policy_loss_weight"]),
            0.0,
            0.0,
            0.0,
            tuple(str(value) for value in recipe.get("q_skip_teacher_prefixes") or ()),
            int(recipe["vps_to_win"]),
            str(recipe["advantage_policy_weighting"]),
            float(recipe["advantage_temperature"]),
            float(recipe["advantage_weight_cap"]),
            float(recipe["advantage_weight_floor"]),
            {"enabled": False, "rank": 0, "world_size": 1, "local_rank": 0},
            str(recipe["amp"]),
            truncated_vp_margin_value_weight=float(
                recipe["truncated_vp_margin_value_weight"]
            ),
            value_target_lambda=float(recipe["value_target_lambda"]),
        )
    finally:
        del policy
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    if int(metrics.get("samples", -1)) != len(unforced_indices):
        raise a0.ContractError(
            f"{checkpoint}: evaluated sample count drift "
            f"({metrics.get('samples')} != {len(unforced_indices)})"
        )
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "samples": int(metrics["samples"]),
        "accuracy_active_count": int(metrics["accuracy_active_count"]),
        "prior_kl_rows": int(metrics["prior_kl_rows"]),
        "policy_loss": float(metrics["policy_loss"]),
        "prior_kl_model_prior_mean": float(metrics["prior_kl_model_prior_mean"]),
        "prior_kl_target_prior_mean": float(metrics["prior_kl_target_prior_mean"]),
    }


def run_probe(*, lock_path: Path, repo_root: Path, device: str) -> dict[str, Any]:
    lock = a0._load_and_verify_lock(lock_path, repo_root)
    recipe = lock["resolved_recipe"]
    if str(recipe.get("arch")) != "entity_graph" or str(
        recipe.get("data_format")
    ) != "memmap":
        raise a0.ContractError("A0 policy-drift probe requires entity_graph memmap")
    data_path = Path(lock["arm_contracts"]["matched_common"]["data"]).resolve()
    data = train_bc.load_teacher_data_memmap(data_path)
    train_bc._MASK_HIDDEN_INFO_PLAYER_TOKENS = bool(recipe["mask_hidden_info"])
    seeds, seed_evidence = _load_exact_validation_seeds(lock, repo_root)
    game_seeds = np.asarray(data["game_seed"], dtype=np.int64)
    legal_counts = np.sum(np.asarray(data["legal_action_ids"]) >= 0, axis=1)
    unforced_indices = np.flatnonzero(
        np.isin(game_seeds, seeds) & (legal_counts > 1)
    ).astype(np.int64, copy=False)
    if len(unforced_indices) == 0:
        raise a0.ContractError("locked A0 holdout has no unforced policy rows")
    policy_weights = _policy_weights(data, recipe)
    value_weights = _value_weights(data, recipe)

    arm_paths = {
        arm: _checkpoint_stages(
            a0._resolve(repo_root, str(lock["arm_contracts"][arm]["checkpoint"]))
        )
        for arm in ("scalar", "hlgauss33")
    }
    stage_metrics: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    for stage in _STAGES:
        scalar = _evaluate_checkpoint(
            arm_paths["scalar"][stage],
            data=data,
            unforced_indices=unforced_indices,
            policy_weights=policy_weights,
            value_weights=value_weights,
            recipe=recipe,
            device=device,
        )
        hl = _evaluate_checkpoint(
            arm_paths["hlgauss33"][stage],
            data=data,
            unforced_indices=unforced_indices,
            policy_weights=policy_weights,
            value_weights=value_weights,
            recipe=recipe,
            device=device,
        )
        comparison = compare_stage_metrics(scalar, hl)
        stage_metrics[stage] = {
            "scalar": scalar,
            "hlgauss33": hl,
            "comparison": comparison,
        }
        gates[stage] = bool(comparison["pass"])
    return {
        "schema_version": SCHEMA,
        "experiment_id": lock.get("experiment_id"),
        "lock": str(lock_path),
        "lock_sha256": _sha256(lock_path),
        "input_contract_sha256": lock["input_contract_sha256"],
        "recipe_sha256": lock["recipe_sha256"],
        "seed_contract_sha256": lock["seed_contract_sha256"],
        "matched_common_sha256": lock["arm_contracts"]["matched_common_sha256"],
        "corpus": str(data_path),
        "corpus_tree_sha256": lock["corpus_tree_sha256"],
        "row_filter": "trainer validation game seeds AND legal_action_count > 1",
        "validation": seed_evidence,
        "thresholds": {
            "max_absolute_relative_policy_drift": POLICY_DRIFT_LIMIT,
        },
        "stages": stage_metrics,
        "gates": gates,
        "policy_drift_pass": all(gates.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.out).resolve()
    if output.exists() and not args.force:
        raise a0.ContractError(
            f"policy-drift artifact exists: {output}; pass --force to replace"
        )
    artifact = run_probe(
        lock_path=Path(args.lock).resolve(),
        repo_root=Path(args.repo_root).resolve(),
        device=str(args.device),
    )
    a0._write_json_atomic(output, artifact)
    print(json.dumps({"out": str(output), **artifact}, indent=2, sort_keys=True))
    if not artifact["policy_drift_pass"]:
        raise a0.ContractError(
            "A0 policy drift exceeds 2%; do not adopt HL-Gauss from this probe"
        )


if __name__ == "__main__":
    main()
