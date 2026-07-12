#!/usr/bin/env python3
"""Re-evaluate a checkpoint's teacher-gap metrics on its locked BC holdout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

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


def run_probe(
    *,
    report_path: Path,
    checkpoint_path: Path,
    data_path: Path,
    validation_manifest_path: Path,
    device: str,
    batch_size: int | None = None,
) -> dict[str, Any]:
    report_path = report_path.resolve(strict=True)
    checkpoint_path = checkpoint_path.resolve(strict=True)
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
    validation_contract = train_bc._load_validation_game_seed_manifest_for_training(
        validation_manifest_path,
        validation_fraction=float(_required(report, "validation_fraction")),
        validation_seed=int(_required(report, "validation_seed")),
        validation_max_samples=int(_required(report, "validation_max_samples")),
        validation_game_seed_ranges=[tuple(map(int, item)) for item in ranges],
    )
    data = train_bc.MemmapCorpus(data_path)
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
    if validation_indices.size != int(validation_contract["validation_row_count"]):
        raise SystemExit(
            "locked holdout row count differs from validation manifest: "
            f"split={validation_indices.size} "
            f"manifest={validation_contract['validation_row_count']}"
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
    policy = _load_policy(str(_required(report, "arch")), checkpoint_path, device)
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
    metrics = train_bc.evaluate_bc_batches(
        policy,
        data,
        validation_indices,
        policy_weights,
        value_weights,
        eval_batch_size,
        float(_required(report, "soft_target_temperature")),
        float(_required(report, "soft_target_weight")),
        str(_required(report, "soft_target_source")),
        float(_required(report, "soft_target_min_legal_coverage")),
        float(_required(report, "policy_loss_weight")),
        scalar_weight,
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
        value_uncertainty_loss_weight=float(
            _required(report, "value_uncertainty_loss_weight")
        ),
        aux_subgoal_loss_weight=float(_required(report, "aux_subgoal_loss_weight")),
        moe_balance_loss_weight=float(_required(report, "moe_balance_loss_weight")),
        value_categorical_loss_weight=categorical_weight,
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
        "schema_version": "posthoc-checkpoint-teacher-gap/v1",
        "inputs": {
            "training_report": {
                "path": str(report_path),
                "sha256": _sha256(report_path),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _sha256(checkpoint_path),
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
        },
        "arch": report["arch"],
        "device": device,
        "batch_size": eval_batch_size,
        "validation_rows": int(validation_indices.size),
        "validation_game_seed_set_sha256": validation_contract[
            "validation_game_seed_set_sha256"
        ],
        "teacher_gap": gap_fields,
        "legacy_prior_kl": legacy_fields,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(
        report_path=args.report,
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        validation_manifest_path=args.validation_manifest,
        device=args.device,
        batch_size=args.batch_size,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
