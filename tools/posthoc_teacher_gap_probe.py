#!/usr/bin/env python3
"""Re-evaluate a checkpoint's teacher-gap metrics on its locked BC holdout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _load_train_bc():
    path = REPO / "tools" / "train_bc.py"
    spec = importlib.util.spec_from_file_location("posthoc_train_bc", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_policy(arch: str, checkpoint: Path, device: str):
    if arch == "entity_graph":
        from catan_zero.rl.entity_token_policy import EntityGraphPolicy

        return EntityGraphPolicy.load(checkpoint, device=device)
    if arch in {"xdim_lite", "xdim_graph"}:
        from catan_zero.rl.xdim_lite_policy import XDimGraphPolicy, XDimLitePolicy

        policy_class = XDimGraphPolicy if arch == "xdim_graph" else XDimLitePolicy
        return policy_class.load(checkpoint, device=device)
    raise SystemExit(f"posthoc teacher-gap probe does not support arch={arch!r}")


def _required(report: dict[str, Any], key: str) -> Any:
    if key not in report:
        raise SystemExit(
            f"training report lacks {key!r}; exact posthoc recipe reconstruction refused"
        )
    return report[key]


def _weight_map(value: Any, field: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise SystemExit(f"training report {field!r} must be an object")
    return {str(key): float(weight) for key, weight in value.items()}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _prepare_probe(
    *,
    report_path: Path,
    data_path: Path,
    validation_manifest_path: Path,
    device: str,
    batch_size: int | None = None,
) -> dict[str, Any]:
    report_path = report_path.resolve(strict=True)
    data_path = data_path.resolve(strict=True)
    validation_manifest_path = validation_manifest_path.resolve(strict=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise SystemExit("training report must contain a JSON object")
    if report.get("data_format") != "memmap":
        raise SystemExit("posthoc teacher-gap probe requires a memmap training report")

    train_bc = _load_train_bc()
    actual_fingerprint = train_bc._training_data_fingerprint(data_path, "memmap")
    expected_fingerprint = str(_required(report, "data_fingerprint"))
    if actual_fingerprint != expected_fingerprint:
        raise SystemExit(
            "memmap fingerprint differs from training report: "
            f"report={expected_fingerprint!r} actual={actual_fingerprint!r}"
        )
    manifest_sha = _sha256(validation_manifest_path)
    emitted_manifest = report.get("validation_game_seed_manifest")
    if emitted_manifest:
        expected_path = Path(str(emitted_manifest)).expanduser()
        if not expected_path.is_absolute():
            expected_path = report_path.parent / expected_path
        expected_path = expected_path.resolve(strict=True)
        if validation_manifest_path != expected_path:
            raise SystemExit(
                "validation manifest path differs from emitted training holdout: "
                f"report={str(expected_path)!r} actual={str(validation_manifest_path)!r}"
            )
    else:
        # Legacy reports only recorded the manifest supplied to training.
        # Modern reports additionally emit the concrete train-validation seed
        # manifest consumed by this probe; its schema and bytes intentionally
        # differ from the upstream selection sentinel.
        expected_manifest_sha = report.get("input_validation_game_seed_manifest_sha256")
        if expected_manifest_sha and manifest_sha != expected_manifest_sha:
            raise SystemExit(
                "validation manifest bytes differ from training report: "
                f"report={expected_manifest_sha!r} actual={manifest_sha!r}"
            )

    ranges = report.get("validation_game_seed_ranges") or []
    if emitted_manifest:
        emitted = json.loads(validation_manifest_path.read_text(encoding="utf-8"))
        required_emitted = {
            "schema_version",
            "data",
            "data_fingerprint",
            "validation_fraction",
            "validation_seed",
            "validation_max_samples",
            "validation_game_seed_ranges",
            "validation_game_seed_count",
            "validation_game_seed_set_sha256",
            "game_seeds",
        }
        if not isinstance(emitted, dict) or not required_emitted.issubset(emitted):
            raise SystemExit("emitted validation holdout manifest is malformed")
        if emitted["schema_version"] != "train-validation-game-seeds-v1":
            raise SystemExit("emitted validation holdout schema drifted")
        seeds = np.asarray(emitted["game_seeds"], dtype=np.int64)
        if (
            seeds.ndim != 1
            or seeds.size == 0
            or not np.all(seeds[1:] > seeds[:-1])
            or int(emitted["validation_game_seed_count"]) != int(seeds.size)
            or emitted["validation_game_seed_set_sha256"]
            != train_bc._game_seed_set_sha256(seeds)
            or Path(str(emitted["data"])).expanduser().resolve(strict=True) != data_path
            or emitted["data_fingerprint"] != expected_fingerprint
            or float(emitted["validation_fraction"])
            != float(_required(report, "validation_fraction"))
            or int(emitted["validation_seed"])
            != int(_required(report, "validation_seed"))
            or int(emitted["validation_max_samples"])
            != int(_required(report, "validation_max_samples"))
            or emitted["validation_game_seed_ranges"] != ranges
        ):
            raise SystemExit("emitted validation holdout semantics drifted")
        validation_contract = {
            "game_seeds": seeds,
            "validation_row_count": None,
            "validation_game_seed_set_sha256": emitted[
                "validation_game_seed_set_sha256"
            ],
            "manifest_sha256": train_bc._canonical_json_sha256(emitted),
        }
    else:
        validation_contract = train_bc._load_validation_game_seed_manifest_for_training(
            validation_manifest_path,
            validation_fraction=float(_required(report, "validation_fraction")),
            validation_seed=int(_required(report, "validation_seed")),
            validation_max_samples=int(_required(report, "validation_max_samples")),
            validation_game_seed_ranges=[tuple(map(int, item)) for item in ranges],
        )
    # Production one-dose learners may consume an authenticated no-copy
    # memmap_composite descriptor rather than a single corpus directory.  The
    # direct MemmapCorpus constructor treats that JSON file as a directory and
    # makes posthoc evaluation impossible for the very checkpoints this tool is
    # meant to diagnose.  Reuse the trainer's fail-closed loader so component
    # identity, per-component target temperatures, and objective scopes survive
    # exactly as they did during training.
    data = train_bc.load_teacher_data_memmap(data_path)
    split = train_bc.split_train_validation_indices(
        data,
        validation_fraction=float(report["validation_fraction"]),
        validation_seed=int(report["validation_seed"]),
        validation_max_samples=int(report["validation_max_samples"]),
        validation_game_seed_ranges=[tuple(map(int, item)) for item in ranges],
        validation_game_seeds=np.asarray(
            validation_contract["game_seeds"], dtype=np.int64
        ),
        allow_missing_game_seed=bool(
            report.get("allow_missing_game_seed_validation_split", False)
        ),
    )
    validation_indices = np.asarray(split["validation"], dtype=np.int64)
    expected_validation_rows = validation_contract.get("validation_row_count")
    if expected_validation_rows is not None and validation_indices.size != int(
        expected_validation_rows
    ):
        raise SystemExit(
            "locked holdout row count differs from validation manifest: "
            f"split={validation_indices.size} "
            f"manifest={expected_validation_rows}"
        )

    policy_weights = train_bc.build_sample_weights(
        data,
        teacher_weights=_weight_map(
            _required(report, "teacher_weights"), "teacher_weights"
        ),
        phase_weights=_weight_map(_required(report, "phase_weights"), "phase_weights"),
        forced_action_weight=float(_required(report, "forced_action_weight")),
        winner_sample_weight=float(_required(report, "winner_sample_weight")),
        loser_sample_weight=float(_required(report, "loser_sample_weight")),
        vp_margin_weight=float(_required(report, "vp_margin_weight")),
        vps_to_win=int(_required(report, "vps_to_win")),
        per_game_policy_weight=bool(report.get("per_game_policy_weight", False)),
        per_game_policy_weight_mode=str(
            report.get("per_game_policy_weight_mode", "equal")
        ),
    )
    value_weights = train_bc.build_value_sample_weights(
        data,
        phase_weights=_weight_map(
            _required(report, "value_phase_weights"), "value_phase_weights"
        ),
        forced_row_value_weight=float(_required(report, "forced_row_value_weight")),
        per_game_value_weight=bool(_required(report, "per_game_value_weight")),
        per_game_value_weight_mode=str(_required(report, "per_game_value_weight_mode")),
    )
    train_bc._MASK_HIDDEN_INFO_PLAYER_TOKENS = bool(
        _required(report, "mask_hidden_info")
    )
    eval_batch_size = int(batch_size or _required(report, "batch_size"))
    if eval_batch_size < 1:
        raise SystemExit("evaluation batch size must be >= 1")
    scalar_weight = float(
        report.get(
            "resolved_scalar_value_loss_weight", report.get("value_loss_weight", 0.0)
        )
    )
    categorical_weight = float(
        report.get(
            "resolved_categorical_value_loss_weight",
            report.get("value_categorical_loss_weight", 0.0),
        )
    )
    holdout_semantics = {
        "schema_version": "posthoc-shared-holdout-identity/v1",
        "memmap_fingerprint": actual_fingerprint,
        "memmap_payload_inventory_sha256": report.get(
            "a1_memmap_payload_inventory_sha256"
        ),
        "validation_manifest_semantic_sha256": validation_contract.get(
            "manifest_sha256"
        ),
        "validation_game_seed_set_sha256": validation_contract[
            "validation_game_seed_set_sha256"
        ],
        "validation_rows": int(validation_indices.size),
        "validation_fraction": float(report["validation_fraction"]),
        "validation_seed": int(report["validation_seed"]),
        "validation_max_samples": int(report["validation_max_samples"]),
        "validation_game_seed_ranges": ranges,
    }
    shared_holdout = {
        **holdout_semantics,
        "identity_sha256": _canonical_sha256(holdout_semantics),
        "training_report": {
            "path": str(report_path),
            "sha256": _sha256(report_path),
        },
        "memmap": {
            "path": str(data_path),
            "fingerprint": actual_fingerprint,
            "payload_inventory_sha256": report.get(
                "a1_memmap_payload_inventory_sha256"
            ),
        },
        "validation_manifest": {
            "path": str(validation_manifest_path),
            "sha256": manifest_sha,
            "manifest_sha256": validation_contract.get("manifest_sha256"),
        },
    }
    return {
        "train_bc": train_bc,
        "report": report,
        "data": data,
        "validation_indices": validation_indices,
        "policy_weights": policy_weights,
        "value_weights": value_weights,
        "device": str(device),
        "batch_size": eval_batch_size,
        "scalar_weight": scalar_weight,
        "categorical_weight": categorical_weight,
        "shared_holdout": shared_holdout,
    }


def _checkpoint_fingerprint(
    *,
    metrics: Mapping[str, Any],
    anchor_direction: str,
) -> dict[str, Any]:
    loss_denominators = metrics.get("loss_denominators")
    if not isinstance(loss_denominators, Mapping):
        loss_denominators = {}
    return {
        "schema_version": "posthoc-checkpoint-fingerprint/v1",
        "parent_policy_kl": {
            "authenticated_anchor_direction": anchor_direction,
            "authenticated_multi_action_rows": int(
                round(float(loss_denominators.get("policy_kl_anchor_loss", 0.0)))
            ),
            "authenticated_anchor_mean": float(
                metrics["policy_kl_anchor_loss"]
            ),
            "raw_prior_rows": int(metrics["prior_kl_rows"]),
            "forward_prior_to_model_mean": float(
                metrics["prior_kl_prior_model_mean"]
            ),
            "reverse_model_to_prior_mean": float(
                metrics["prior_kl_model_prior_mean"]
            ),
        },
        "policy_top1": {
            "active_rows": int(metrics["accuracy_active_count"]),
            "action_accuracy": float(metrics["accuracy"]),
            "top3_action_accuracy": float(metrics["top3_accuracy"]),
        },
        "policy_entropy": {
            "active_rows": int(metrics["active_policy_model_entropy_rows"]),
            "mean_nats": float(metrics["active_policy_model_entropy_mean"]),
        },
        "value": {
            "primary_kind": str(metrics["primary_value_loss_kind"]),
            "primary_loss": float(metrics["primary_value_loss"]),
            "scalar_mse": float(metrics["scalar_value_mse_diagnostic"]),
            "categorical_loss": float(metrics["value_categorical_loss"]),
            "final_vp_loss": float(metrics["final_vp_loss"]),
            "uncertainty_loss": float(metrics["value_uncertainty_loss"]),
        },
    }


def _evaluate_checkpoint(
    prepared: Mapping[str, Any],
    *,
    label: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve(strict=True)
    report = prepared["report"]
    train_bc = prepared["train_bc"]
    policy = _load_policy(
        str(_required(report, "arch")), checkpoint_path, str(prepared["device"])
    )
    anchor_direction = str(report.get("policy_kl_anchor_direction", "forward"))
    if anchor_direction not in {"forward", "reverse"}:
        raise SystemExit(
            f"training report policy_kl_anchor_direction={anchor_direction!r} is invalid"
        )
    metrics = train_bc.evaluate_bc_batches(
        policy,
        prepared["data"],
        prepared["validation_indices"],
        prepared["policy_weights"],
        prepared["value_weights"],
        int(prepared["batch_size"]),
        float(_required(report, "soft_target_temperature")),
        float(_required(report, "soft_target_weight")),
        str(_required(report, "soft_target_source")),
        float(_required(report, "soft_target_min_legal_coverage")),
        float(_required(report, "policy_loss_weight")),
        float(prepared["scalar_weight"]),
        float(_required(report, "final_vp_loss_weight")),
        float(_required(report, "q_loss_weight")),
        tuple(str(item) for item in _required(report, "q_skip_teacher_prefixes")),
        int(report["vps_to_win"]),
        str(_required(report, "advantage_policy_weighting")),
        float(_required(report, "advantage_temperature")),
        float(_required(report, "advantage_weight_cap")),
        float(_required(report, "advantage_weight_floor")),
        {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0},
        str(_required(report, "amp")),
        truncated_vp_margin_value_weight=float(
            _required(report, "truncated_vp_margin_value_weight")
        ),
        policy_kl_anchor_weight=float(_required(report, "policy_kl_anchor_weight")),
        policy_kl_anchor_direction=anchor_direction,
        value_uncertainty_loss_weight=float(
            _required(report, "value_uncertainty_loss_weight")
        ),
        aux_subgoal_loss_weight=float(_required(report, "aux_subgoal_loss_weight")),
        moe_balance_loss_weight=float(_required(report, "moe_balance_loss_weight")),
        value_categorical_loss_weight=float(prepared["categorical_weight"]),
        value_hlgauss_sigma_ratio=float(_required(report, "value_hlgauss_sigma_ratio")),
        value_target_lambda=float(_required(report, "value_target_lambda")),
    )
    gap_fields = {
        key: metrics[key]
        for key in (
            "active_policy_teacher_gap_rows",
            "active_policy_kl_target_model_mean",
            "active_policy_kl_target_prior_mean",
            "active_policy_teacher_gap_closure",
        )
    }
    legacy_fields = {
        key: metrics[key]
        for key in (
            "prior_kl_rows",
            "prior_kl_model_prior_mean",
            "prior_kl_target_prior_mean",
            "prior_kl_ratio",
        )
    }
    return {
        "label": label,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
        },
        "teacher_gap": gap_fields,
        "legacy_prior_kl": legacy_fields,
        "fingerprint": _checkpoint_fingerprint(
            metrics=metrics, anchor_direction=anchor_direction
        ),
        "metrics": metrics,
    }


def run_probe(
    *,
    report_path: Path,
    checkpoint_path: Path,
    data_path: Path,
    validation_manifest_path: Path,
    device: str,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Single-checkpoint compatibility entry point."""

    prepared = _prepare_probe(
        report_path=report_path,
        data_path=data_path,
        validation_manifest_path=validation_manifest_path,
        device=device,
        batch_size=batch_size,
    )
    evaluated = _evaluate_checkpoint(
        prepared, label="checkpoint", checkpoint_path=checkpoint_path
    )
    shared = prepared["shared_holdout"]
    report = prepared["report"]
    return {
        "schema_version": "posthoc-checkpoint-teacher-gap/v1",
        "inputs": {
            "training_report": shared["training_report"],
            "checkpoint": evaluated["checkpoint"],
            "memmap": shared["memmap"],
            "validation_manifest": shared["validation_manifest"],
        },
        "arch": report["arch"],
        "device": device,
        "batch_size": int(prepared["batch_size"]),
        "validation_rows": int(shared["validation_rows"]),
        "validation_game_seed_set_sha256": shared[
            "validation_game_seed_set_sha256"
        ],
        "shared_holdout_identity_sha256": shared["identity_sha256"],
        "teacher_gap": evaluated["teacher_gap"],
        "legacy_prior_kl": evaluated["legacy_prior_kl"],
        "checkpoint_fingerprint": evaluated["fingerprint"],
        "metrics": evaluated["metrics"],
    }


def _step64_128_comparison(
    checkpoints: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    if "step64" not in checkpoints or "step128" not in checkpoints:
        return None
    left = checkpoints["step64"]["fingerprint"]
    right = checkpoints["step128"]["fingerprint"]
    metrics = {
        "authenticated_parent_kl": (
            left["parent_policy_kl"]["authenticated_anchor_mean"],
            right["parent_policy_kl"]["authenticated_anchor_mean"],
        ),
        "policy_top1_accuracy": (
            left["policy_top1"]["action_accuracy"],
            right["policy_top1"]["action_accuracy"],
        ),
        "policy_entropy_nats": (
            left["policy_entropy"]["mean_nats"],
            right["policy_entropy"]["mean_nats"],
        ),
        "scalar_value_mse": (
            left["value"]["scalar_mse"],
            right["value"]["scalar_mse"],
        ),
        "teacher_gap_closure": (
            checkpoints["step64"]["teacher_gap"][
                "active_policy_teacher_gap_closure"
            ],
            checkpoints["step128"]["teacher_gap"][
                "active_policy_teacher_gap_closure"
            ],
        ),
    }
    return {
        "schema_version": "posthoc-step64-step128-dose-comparison/v1",
        "from": "step64",
        "to": "step128",
        "metrics": {
            name: {
                "step64": float(values[0]),
                "step128": float(values[1]),
                "step128_minus_step64": float(values[1]) - float(values[0]),
            }
            for name, values in metrics.items()
        },
    }


def run_batch_probe(
    *,
    report_path: Path,
    checkpoints: Sequence[tuple[str, Path]],
    data_path: Path,
    validation_manifest_path: Path,
    device: str,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Evaluate labeled checkpoints while materializing the holdout once."""

    if not checkpoints:
        raise SystemExit("batch teacher-gap probe requires at least one checkpoint")
    labels = [str(label).strip() for label, _ in checkpoints]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise SystemExit("batch checkpoint labels must be non-empty and unique")
    prepared = _prepare_probe(
        report_path=report_path,
        data_path=data_path,
        validation_manifest_path=validation_manifest_path,
        device=device,
        batch_size=batch_size,
    )
    results = {
        label: _evaluate_checkpoint(
            prepared, label=label, checkpoint_path=checkpoint_path
        )
        for label, checkpoint_path in checkpoints
    }
    output = {
        "schema_version": "posthoc-checkpoint-teacher-gap-batch/v1",
        "arch": prepared["report"]["arch"],
        "device": str(device),
        "batch_size": int(prepared["batch_size"]),
        "checkpoint_order": labels,
        "shared_holdout": prepared["shared_holdout"],
        "checkpoints": results,
    }
    dose_comparison = _step64_128_comparison(results)
    if dose_comparison is not None:
        output["dose_comparison"] = dose_comparison
    return output


def _parse_labeled_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "labeled checkpoint must use LABEL=PATH (for example step64=/run/step64.pt)"
        )
    label, raw_path = value.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise argparse.ArgumentTypeError(
            "labeled checkpoint must have a non-empty label and path"
        )
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Legacy single-checkpoint mode (output schema remains v1).",
    )
    parser.add_argument(
        "--labeled-checkpoint",
        "--compare-checkpoint",
        dest="labeled_checkpoints",
        action="append",
        type=_parse_labeled_checkpoint,
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Evaluate a labeled checkpoint in shared-corpus mode; repeat the flag. "
            "Labels step64 and step128 additionally emit a direct dose comparison."
        ),
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.checkpoint is not None and args.labeled_checkpoints:
        parser.error(
            "--checkpoint cannot be combined with --labeled-checkpoint"
        )
    if args.checkpoint is not None:
        result = run_probe(
            report_path=args.report,
            checkpoint_path=args.checkpoint,
            data_path=args.data,
            validation_manifest_path=args.validation_manifest,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.labeled_checkpoints:
        result = run_batch_probe(
            report_path=args.report,
            checkpoints=args.labeled_checkpoints,
            data_path=args.data,
            validation_manifest_path=args.validation_manifest,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        parser.error(
            "one --checkpoint or at least one --labeled-checkpoint is required"
        )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
