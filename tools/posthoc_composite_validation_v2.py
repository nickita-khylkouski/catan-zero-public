#!/usr/bin/env python3
"""Read-only exact-v2 validation for an existing composite-trained checkpoint.

This tool performs no optimizer construction, backward pass, checkpoint save, or
training-report mutation. It replays the completed run's exact learner objective
on its locked whole-game holdout, using ``composite-validation-measure-v2``.
Every input byte identity and the evaluation checkout/runtime identity are bound
into the output so old v1 validation can be replaced without rewriting history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tools import train_bc  # noqa: E402

SCHEMA = "posthoc-composite-validation-v2/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must contain a JSON object")
    return value


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


def _load_policy(arch: str, checkpoint: Path, device: str):
    if arch == "entity_graph":
        from catan_zero.rl.entity_token_policy import EntityGraphPolicy

        return EntityGraphPolicy.load(checkpoint, device=device)
    if arch in {"xdim_lite", "xdim_graph"}:
        from catan_zero.rl.xdim_lite_policy import XDimGraphPolicy, XDimLitePolicy

        cls = XDimGraphPolicy if arch == "xdim_graph" else XDimLitePolicy
        return cls.load(checkpoint, device=device)
    raise SystemExit(f"exact composite validation does not support arch={arch!r}")


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _locked_validation_indices(
    data, manifest_path: Path, report: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = _json_object(manifest_path, "validation manifest")
    raw_seeds = manifest.get("game_seeds")
    if (
        not isinstance(raw_seeds, list)
        or not raw_seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds)
    ):
        raise SystemExit("validation manifest has no valid integer game_seeds")
    seeds = np.asarray(raw_seeds, dtype=np.int64)
    if len(np.unique(seeds)) != len(seeds):
        raise SystemExit("validation manifest game_seeds are not unique")
    seed_set_sha = train_bc._game_seed_set_sha256(seeds)
    expected_seed_sha = str(_required(report, "validation_game_seed_set_sha256"))
    if seed_set_sha != expected_seed_sha:
        raise SystemExit(
            "validation seed identity differs from training report: "
            f"report={expected_seed_sha} manifest={seed_set_sha}"
        )
    all_seeds = np.asarray(data["game_seed"], dtype=np.int64)
    indices = np.flatnonzero(np.isin(all_seeds, seeds)).astype(np.int64, copy=False)
    expected_rows = int(_required(report, "validation_samples"))
    if len(indices) != expected_rows:
        raise SystemExit(
            "validation row count differs from training report: "
            f"report={expected_rows} replay={len(indices)}"
        )
    component_indices = np.asarray(
        data.component_indices_for_rows(indices), dtype=np.int64
    )
    missing = sorted(
        set(range(len(data.corpora))) - set(map(int, np.unique(component_indices)))
    )
    if missing:
        raise SystemExit(f"locked validation omits composite components {missing}")
    return indices, {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
        "game_seed_count": int(len(seeds)),
        "game_seed_set_sha256": seed_set_sha,
        "row_count": int(len(indices)),
    }


def _root_blend_args(report: dict[str, Any]) -> tuple[tuple[str, ...], bool]:
    regime = report.get("value_root_blend_regime")
    if not isinstance(regime, dict):
        return (), False
    mode = str(regime.get("mode", "disabled"))
    phases = regime.get("phases", [])
    if not isinstance(phases, list) or any(not isinstance(item, str) for item in phases):
        raise SystemExit("training report value-root-blend phases are malformed")
    return tuple(phases), mode == "global_compat"


def run_rescore(
    *,
    report_path: Path,
    checkpoint_path: Path,
    descriptor_path: Path,
    validation_manifest_path: Path,
    device: str,
    batch_size: int | None = None,
) -> dict[str, Any]:
    paths = {
        "training_report": report_path.resolve(strict=True),
        "checkpoint": checkpoint_path.resolve(strict=True),
        "descriptor": descriptor_path.resolve(strict=True),
        "validation_manifest": validation_manifest_path.resolve(strict=True),
    }
    if not paths["descriptor"].is_file():
        raise SystemExit("exact v2 posthoc validation requires a composite descriptor file")
    before = {name: _sha256(path) for name, path in paths.items()}
    report = _json_object(paths["training_report"], "training report")
    if report.get("data_format") != "memmap":
        raise SystemExit("exact v2 posthoc validation requires data_format=memmap")
    reported_checkpoint = report.get("checkpoint")
    if isinstance(reported_checkpoint, str) and Path(reported_checkpoint).is_absolute():
        if Path(reported_checkpoint).resolve(strict=False) != paths["checkpoint"]:
            raise SystemExit(
                "checkpoint path differs from the checkpoint bound by the training report"
            )

    authenticated = train_bc._preflight_memmap_composite_descriptor(paths["descriptor"])
    if authenticated.get("schema_version") != "memmap_composite_v2":
        raise SystemExit("exact v2 posthoc validation requires memmap_composite_v2")
    fingerprint = train_bc._training_data_fingerprint(paths["descriptor"], "memmap")
    if fingerprint != str(_required(report, "data_fingerprint")):
        raise SystemExit("composite descriptor fingerprint differs from training report")
    expected_inventory = report.get("a1_memmap_payload_inventory_sha256")
    if expected_inventory and expected_inventory != authenticated.get(
        "payload_inventory_sha256"
    ):
        raise SystemExit("composite payload inventory differs from training report")

    data = train_bc.load_teacher_data_memmap(
        paths["descriptor"], composite_meta=authenticated
    )
    validation_indices, validation_binding = _locked_validation_indices(
        data, paths["validation_manifest"], report
    )
    policy_weights = train_bc.build_sample_weights(
        data,
        teacher_weights=_weight_map(_required(report, "teacher_weights"), "teacher_weights"),
        phase_weights=_weight_map(_required(report, "phase_weights"), "phase_weights"),
        forced_action_weight=float(_required(report, "forced_action_weight")),
        winner_sample_weight=float(_required(report, "winner_sample_weight")),
        loser_sample_weight=float(_required(report, "loser_sample_weight")),
        vp_margin_weight=float(_required(report, "vp_margin_weight")),
        vps_to_win=int(_required(report, "vps_to_win")),
        per_game_policy_weight=bool(_required(report, "per_game_policy_weight")),
        per_game_policy_weight_mode=str(
            _required(report, "per_game_policy_weight_mode")
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
    policy = _load_policy(
        str(_required(report, "arch")), paths["checkpoint"], device
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
    blend_phases, blend_global = _root_blend_args(report)
    ddp = {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}

    def evaluate(indices: np.ndarray) -> dict:
        return train_bc.evaluate_bc_batches(
            policy,
            data,
            indices,
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
            int(_required(report, "vps_to_win")),
            str(_required(report, "advantage_policy_weighting")),
            float(_required(report, "advantage_temperature")),
            float(_required(report, "advantage_weight_cap")),
            float(_required(report, "advantage_weight_floor")),
            ddp,
            str(_required(report, "amp")),
            truncated_vp_margin_value_weight=float(
                _required(report, "truncated_vp_margin_value_weight")
            ),
            policy_kl_anchor_weight=float(_required(report, "policy_kl_anchor_weight")),
            policy_kl_anchor_direction=str(
                _required(report, "policy_kl_anchor_direction")
            ),
            value_uncertainty_loss_weight=float(
                _required(report, "value_uncertainty_loss_weight")
            ),
            aux_subgoal_loss_weight=float(_required(report, "aux_subgoal_loss_weight")),
            moe_balance_loss_weight=float(_required(report, "moe_balance_loss_weight")),
            value_categorical_loss_weight=categorical_weight,
            value_hlgauss_sigma_ratio=float(
                _required(report, "value_hlgauss_sigma_ratio")
            ),
            value_target_lambda=float(_required(report, "value_target_lambda")),
            value_root_blend_phases=blend_phases,
            value_root_blend_global_compat=blend_global,
        )

    exact = train_bc.evaluate_composite_validation_measure(
        data, validation_indices, evaluate
    )
    if exact.get("schema_version") != "composite-validation-measure-v2":
        raise RuntimeError("trainer did not emit exact composite validation v2")
    after = {name: _sha256(path) for name, path in paths.items()}
    if after != before:
        raise RuntimeError("posthoc validation input bytes changed during evaluation")
    runtime_binding = train_bc._assert_checkout_runtime_binding()
    return {
        "schema_version": SCHEMA,
        "read_only": True,
        "optimizer_steps": 0,
        "checkpoint_mutated": False,
        "inputs": {
            name: {"path": str(paths[name]), "sha256": before[name]}
            for name in ("training_report", "checkpoint", "descriptor")
        },
        "validation_manifest": validation_binding,
        "training_runtime_binding": report.get("checkout_runtime_binding"),
        "evaluation_runtime_binding": runtime_binding,
        "evaluation_repo_commit": _git_commit(),
        "evaluation_tool_sha256": _sha256(Path(__file__).resolve()),
        "composite": {
            "descriptor_fingerprint": fingerprint,
            "payload_inventory_sha256": authenticated.get(
                "payload_inventory_sha256"
            ),
            "component_ids": list(authenticated.get("component_ids", [])),
            "component_game_sampling_ratios": list(
                authenticated.get("component_game_sampling_ratios", [])
            ),
        },
        "arch": report["arch"],
        "device": device,
        "batch_size": eval_batch_size,
        "exact_validation": exact,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        args.report.resolve(),
        args.checkpoint.resolve(),
        args.descriptor.resolve(),
        args.validation_manifest.resolve(),
    }
    if args.out.resolve() in inputs:
        raise SystemExit("--out must not overwrite any posthoc validation input")
    result = run_rescore(
        report_path=args.report,
        checkpoint_path=args.checkpoint,
        descriptor_path=args.descriptor,
        validation_manifest_path=args.validation_manifest,
        device=args.device,
        batch_size=args.batch_size,
    )
    train_bc.write_json(args.out, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
