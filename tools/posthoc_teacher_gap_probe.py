#!/usr/bin/env python3
"""Re-evaluate a checkpoint's teacher-gap metrics on its locked BC holdout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
PAIRED_PARENT_GAP_SCHEMA = "posthoc-paired-parent-teacher-gap-v2"
VALUE_QUALITY_SCHEMA = "posthoc-objective-matched-value-quality-v1"
PAIRED_PARENT_VALUE_SCHEMA = "posthoc-paired-parent-value-quality-v1"
POLICY_TEACHER_GAP_OBJECTIVE_SCHEMA = "posthoc-policy-teacher-gap-objective-v1"
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


def _policy_teacher_gap_objective(report: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the policy objective that the teacher-gap probe actually scores."""

    raw_batch_size = report.get("policy_aux_active_batch_size", 0)
    raw_coefficient = report.get("policy_aux_loss_weight", 0.0)
    if (
        isinstance(raw_batch_size, bool)
        or not isinstance(raw_batch_size, int)
        or raw_batch_size < 0
    ):
        raise SystemExit("training report policy_aux_active_batch_size is malformed")
    try:
        coefficient = float(raw_coefficient)
    except (TypeError, ValueError) as error:
        raise SystemExit("training report policy_aux_loss_weight is malformed") from error
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise SystemExit("training report policy_aux_loss_weight is malformed")
    if raw_batch_size > 0:
        raise SystemExit(
            "posthoc teacher-gap probe cannot score an AUX-enabled policy objective: "
            "training used independently normalized base + coefficient*AUX(q*w), "
            "but objective-matched AUX teacher-gap scoring is not implemented"
        )
    return {
        "schema_version": POLICY_TEACHER_GAP_OBJECTIVE_SCHEMA,
        "selection_authority": True,
        "objective_matched": True,
        "formula": "base_policy_teacher_kl",
        "policy_aux_enabled": False,
        "policy_aux_active_batch_size": 0,
        "policy_aux_loss_weight": 0.0,
        "policy_aux_measure": "disabled",
    }


def _functional_drift_batch(
    parent_logits,
    candidate_logits,
    parent_values,
    candidate_values,
    *,
    legal_mask,
    eligible,
) -> dict[str, float]:
    """Return additive parent→candidate functional-distance statistics.

    The caller supplies logits in the shared legal-action row order.  Padded
    actions are removed explicitly before either KL is evaluated, avoiding the
    undefined ``0 * (-inf - -inf)`` arithmetic that otherwise turns a valid
    ragged policy batch into NaNs.  Only multi-action, policy-active anchor rows
    are eligible; value drift is reported over that identical state surface so
    policy and value movement remain directly comparable.
    """

    import torch

    legal = legal_mask.bool()
    active = eligible.bool()
    if legal.ndim != 2 or active.ndim != 1 or legal.shape[0] != active.shape[0]:
        raise ValueError("functional-drift batch shape mismatch")
    active = active & (legal.sum(dim=-1) > 1)
    count = int(active.sum().item())
    if count == 0:
        return {
            "rows": 0.0,
            "parent_candidate_kl_sum": 0.0,
            "candidate_parent_kl_sum": 0.0,
            "top1_flip_sum": 0.0,
            "parent_entropy_sum": 0.0,
            "candidate_entropy_sum": 0.0,
            "value_abs_delta_sum": 0.0,
            "value_squared_delta_sum": 0.0,
        }

    floor = torch.finfo(torch.float32).min
    parent_log = torch.log_softmax(
        parent_logits.float().masked_fill(~legal, floor), dim=-1
    )
    candidate_log = torch.log_softmax(
        candidate_logits.float().masked_fill(~legal, floor), dim=-1
    )
    parent_prob = parent_log.exp().masked_fill(~legal, 0.0)
    candidate_prob = candidate_log.exp().masked_fill(~legal, 0.0)
    parent_log_finite = parent_log.masked_fill(~legal, 0.0)
    candidate_log_finite = candidate_log.masked_fill(~legal, 0.0)
    kl_parent_candidate = (
        parent_prob * (parent_log_finite - candidate_log_finite)
    ).sum(dim=-1)
    kl_candidate_parent = (
        candidate_prob * (candidate_log_finite - parent_log_finite)
    ).sum(dim=-1)
    parent_entropy = -(parent_prob * parent_log_finite).sum(dim=-1)
    candidate_entropy = -(candidate_prob * candidate_log_finite).sum(dim=-1)
    parent_top1 = parent_logits.float().masked_fill(~legal, floor).argmax(dim=-1)
    candidate_top1 = candidate_logits.float().masked_fill(~legal, floor).argmax(dim=-1)
    value_delta = candidate_values.float() - parent_values.float()
    return {
        "rows": float(count),
        "parent_candidate_kl_sum": float(kl_parent_candidate[active].sum().item()),
        "candidate_parent_kl_sum": float(kl_candidate_parent[active].sum().item()),
        "top1_flip_sum": float(
            (parent_top1[active] != candidate_top1[active]).sum().item()
        ),
        "parent_entropy_sum": float(parent_entropy[active].sum().item()),
        "candidate_entropy_sum": float(candidate_entropy[active].sum().item()),
        "value_abs_delta_sum": float(value_delta[active].abs().sum().item()),
        "value_squared_delta_sum": float(value_delta[active].square().sum().item()),
    }


def _functional_drift(
    *,
    train_bc,
    parent_policy,
    candidate_policy,
    data,
    indices: np.ndarray,
    policy_weights: np.ndarray,
    batch_size: int,
) -> dict[str, Any]:
    """Measure functional distance on one immutable validation anchor."""

    import math
    import torch

    totals = {
        "rows": 0.0,
        "parent_candidate_kl_sum": 0.0,
        "candidate_parent_kl_sum": 0.0,
        "top1_flip_sum": 0.0,
        "parent_entropy_sum": 0.0,
        "candidate_entropy_sum": 0.0,
        "value_abs_delta_sum": 0.0,
        "value_squared_delta_sum": 0.0,
    }
    parent_modes = train_bc._set_policy_training(parent_policy, False)
    candidate_modes = train_bc._set_policy_training(candidate_policy, False)
    try:
        with torch.no_grad():
            for start in range(0, len(indices), int(batch_size)):
                batch = np.asarray(
                    indices[start : start + int(batch_size)], dtype=np.int64
                )
                legal_ids = np.asarray(data["legal_action_ids"][batch])
                parent = train_bc._forward_legal_np_for_batch(
                    parent_policy,
                    data,
                    batch,
                    legal_ids,
                    return_q=False,
                    return_final_vp=False,
                )
                candidate = train_bc._forward_legal_np_for_batch(
                    candidate_policy,
                    data,
                    batch,
                    legal_ids,
                    return_q=False,
                    return_final_vp=False,
                )
                if "value" not in parent or "value" not in candidate:
                    raise SystemExit(
                        "functional-dose fingerprint requires scalar value outputs"
                    )
                parts = _functional_drift_batch(
                    parent["logits"],
                    candidate["logits"],
                    parent["value"],
                    candidate["value"],
                    legal_mask=torch.as_tensor(
                        legal_ids >= 0, device=parent_policy.device
                    ),
                    eligible=torch.as_tensor(
                        np.asarray(policy_weights[batch]) > 0.0,
                        device=parent_policy.device,
                    ),
                )
                for key, value in parts.items():
                    totals[key] += float(value)
    finally:
        train_bc._restore_policy_training(parent_policy, parent_modes)
        train_bc._restore_policy_training(candidate_policy, candidate_modes)

    rows = totals["rows"]
    if rows <= 0:
        raise SystemExit("functional-dose fingerprint has no active multi-action rows")
    return {
        "schema_version": "checkpoint-functional-dose-fingerprint-v1",
        "eligible_rows": int(round(rows)),
        "surface": "validation_policy_active_multi_action_rows",
        "kl_parent_candidate_mean": totals["parent_candidate_kl_sum"] / rows,
        "kl_candidate_parent_mean": totals["candidate_parent_kl_sum"] / rows,
        "top1_flip_rate": totals["top1_flip_sum"] / rows,
        "parent_policy_entropy_mean": totals["parent_entropy_sum"] / rows,
        "candidate_policy_entropy_mean": totals["candidate_entropy_sum"] / rows,
        "policy_entropy_delta": (
            totals["candidate_entropy_sum"] - totals["parent_entropy_sum"]
        )
        / rows,
        "value_mean_absolute_delta": totals["value_abs_delta_sum"] / rows,
        "value_root_mean_square_delta": math.sqrt(
            totals["value_squared_delta_sum"] / rows
        ),
    }


def _weight_map(value: Any, field: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise SystemExit(f"training report {field!r} must be an object")
    return {str(key): float(weight) for key, weight in value.items()}


def _root_blend_args(report: Mapping[str, Any]) -> tuple[tuple[str, ...], bool]:
    """Rebuild the checkpoint's authenticated root-value target operator."""

    regime = report.get("value_root_blend_regime")
    if not isinstance(regime, Mapping):
        return (), False
    mode = str(regime.get("mode", "disabled"))
    phases = regime.get("phases", [])
    if not isinstance(phases, list) or any(
        not isinstance(item, str) for item in phases
    ):
        raise SystemExit("training report value-root-blend phases are malformed")
    if mode not in {"disabled", "phase_gated", "global_compat"}:
        raise SystemExit("training report value-root-blend mode is malformed")
    return tuple(phases), mode == "global_compat"


def _scalar_value_loss_args(report: Mapping[str, Any]) -> tuple[str, float]:
    """Return the report-authenticated scalar learner readout."""

    contract = report.get("scalar_value_loss_contract")
    if contract is None:
        return "raw", 1.0
    if not isinstance(contract, Mapping):
        raise SystemExit("scalar_value_loss_contract must be a JSON object")
    if contract.get("schema_version") != "scalar-value-loss-readout-v1":
        raise SystemExit("unsupported scalar_value_loss_contract schema")
    readout = str(contract.get("readout", ""))
    if readout not in {"raw", "deployed_tanh"}:
        raise SystemExit("unsupported scalar value loss readout")
    try:
        scale = float(contract["scale"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit("scalar value loss scale must be numeric") from error
    if not np.isfinite(scale) or scale <= 0.0:
        raise SystemExit("scalar value loss scale must be finite and > 0")
    expected_formula = "raw" if readout == "raw" else "tanh(raw * scale)"
    if contract.get("formula") != expected_formula:
        raise SystemExit("scalar value loss formula differs from its typed contract")
    return readout, scale


def _forced_row_value_recipe(
    train_bc, report: Mapping[str, Any]
) -> tuple[dict[str, float], object | None]:
    """Rebuild the optional action-typed forced-row value weighting."""

    raw = report.get("forced_row_value_action_type_weights")
    if raw is None:
        return {}, None
    weights = _weight_map(raw, "forced_row_value_action_type_weights")
    if not weights:
        return {}, None
    graph_history_features = _required(report, "graph_history_features")
    if type(graph_history_features) is not bool:  # noqa: E721
        raise SystemExit("training report 'graph_history_features' must be a boolean")
    env_config = train_bc.parse_track(
        str(_required(report, "track")),
        vps_to_win=int(_required(report, "vps_to_win")),
        use_graph_history_features=graph_history_features,
    )
    return weights, train_bc._action_catalog_for_env_config(env_config)


def _scope_identity(data, report: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the component eligibility actually reconstructed by the probe."""

    component_ids = tuple(str(value) for value in getattr(data, "component_ids", ()))

    def _one(
        *, authenticated_attr: str, indices_attr: str, report_field: str
    ) -> dict[str, Any]:
        authenticated = bool(getattr(data, authenticated_attr, False))
        indices = tuple(int(value) for value in getattr(data, indices_attr, ()))
        if authenticated and (
            not component_ids
            or not indices
            or any(index < 0 or index >= len(component_ids) for index in indices)
        ):
            raise SystemExit("authenticated objective component scope is malformed")
        report_scope = report.get(report_field)
        if authenticated and not isinstance(report_scope, Mapping):
            raise SystemExit(
                f"training report lacks authenticated {report_field!r} telemetry"
            )
        eligible_ids = (
            [component_ids[index] for index in indices]
            if authenticated
            else list(component_ids)
        )
        if authenticated and report_scope.get("component_ids") != eligible_ids:
            raise SystemExit(
                f"training report {report_field!r} differs from the loaded corpus"
            )
        return {
            "authenticated": authenticated,
            "component_ids": eligible_ids,
        }

    return {
        "policy_distillation": _one(
            authenticated_attr="policy_distillation_scope_authenticated",
            indices_attr="policy_distillation_component_indices",
            report_field="policy_distillation_scope",
        ),
        "value_training": _one(
            authenticated_attr="value_training_scope_authenticated",
            indices_attr="value_training_component_indices",
            report_field="value_training_scope",
        ),
    }


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
    """Authenticate and materialize the shared validation surface once."""

    report_path = report_path.resolve(strict=True)
    data_path = data_path.resolve(strict=True)
    validation_manifest_path = validation_manifest_path.resolve(strict=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise SystemExit("training report must contain a JSON object")
    if report.get("data_format") != "memmap":
        raise SystemExit("posthoc teacher-gap probe requires a memmap training report")
    policy_teacher_gap_objective = _policy_teacher_gap_objective(report)

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
    # memmap_composite descriptor rather than a single corpus directory. Reuse
    # the trainer's fail-closed loader so component identity, target
    # temperatures, and objective scopes remain exactly as trained. Batch mode
    # deliberately executes this once for every compared checkpoint.
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
            f"split={validation_indices.size} manifest={expected_validation_rows}"
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
        target_reliability_confidence_weighting=bool(
            report.get("target_reliability_confidence_weighting", False)
        ),
        target_reliability_confidence_floor=float(
            report.get("target_reliability_confidence_floor", 0.25)
        ),
    )
    policy_weights = train_bc._apply_authenticated_policy_distillation_scope(
        data, policy_weights
    )
    forced_type_weights, forced_action_catalog = _forced_row_value_recipe(
        train_bc, report
    )
    value_weights = train_bc.build_value_sample_weights(
        data,
        phase_weights=_weight_map(
            _required(report, "value_phase_weights"), "value_phase_weights"
        ),
        forced_row_value_weight=float(_required(report, "forced_row_value_weight")),
        forced_row_value_action_type_weights=forced_type_weights,
        action_catalog=forced_action_catalog,
        per_game_value_weight=bool(_required(report, "per_game_value_weight")),
        per_game_value_weight_mode=str(_required(report, "per_game_value_weight_mode")),
    )
    value_weights = train_bc._apply_authenticated_value_training_scope(
        data, value_weights
    )
    eval_batch_size = int(batch_size or _required(report, "batch_size"))
    if eval_batch_size < 1:
        raise SystemExit("evaluation batch size must be >= 1")

    train_bc._MASK_HIDDEN_INFO_PLAYER_TOKENS = bool(
        _required(report, "mask_hidden_info")
    )
    award_contract = str(_required(report, "public_award_feature_contract"))
    train_bc._PUBLIC_AWARD_FEATURE_CONTRACT = award_contract
    blend_phases, blend_global = _root_blend_args(report)
    scalar_readout, scalar_scale = _scalar_value_loss_args(report)
    scope_identity = _scope_identity(data, report)
    objective_reconstruction = {
        "schema_version": "posthoc-objective-reconstruction-v1",
        "policy_teacher_gap_objective": policy_teacher_gap_objective,
        "component_scopes": scope_identity,
        "target_reliability_confidence_weighting": bool(
            report.get("target_reliability_confidence_weighting", False)
        ),
        "target_reliability_confidence_floor": float(
            report.get("target_reliability_confidence_floor", 0.25)
        ),
        "forced_row_value_action_type_weights": forced_type_weights,
        "policy_kl_anchor_direction": str(
            report.get("policy_kl_anchor_direction", "forward")
        ),
        "belief_resource_loss_weight": float(
            report.get("belief_resource_loss_weight", 0.0)
        ),
        "value_root_blend_phases": list(blend_phases),
        "value_root_blend_global_compat": blend_global,
        "value_target_lambda": float(_required(report, "value_target_lambda")),
        "scalar_value_loss_contract": {
            "readout": scalar_readout,
            "scale": scalar_scale,
        },
    }
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
        "objective_reconstruction": objective_reconstruction,
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
        "award_contract": award_contract,
        "objective_reconstruction": objective_reconstruction,
        "policy_teacher_gap_objective": policy_teacher_gap_objective,
        "scalar_value_loss_readout": scalar_readout,
        "scalar_value_loss_scale": scalar_scale,
        "shared_holdout": {
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
        },
    }


def _policy_input_surface(
    prepared: Mapping[str, Any], policy, *, role: str
) -> dict[str, Any]:
    report = prepared["report"]
    award_contract = str(prepared["award_contract"])
    policy_award = str(getattr(policy, "public_award_feature_contract", award_contract))
    if policy_award != award_contract:
        raise SystemExit(
            f"{role} public-award input contract differs from training report"
        )
    card_features = bool(
        getattr(getattr(policy, "config", None), "public_card_count_features", False)
    )
    reported_card_features = report.get("public_card_count_features")
    if reported_card_features is not None:
        if not isinstance(reported_card_features, bool):
            raise SystemExit("training report public_card_count_features is malformed")
        if card_features != reported_card_features:
            raise SystemExit(
                f"{role} public-card input schema differs from training report"
            )
    return {
        "public_award_feature_contract": award_contract,
        "public_card_count_features": card_features,
        "mask_hidden_info": bool(_required(report, "mask_hidden_info")),
    }


def _report_parent_binding(report: Mapping[str, Any]) -> tuple[str, str] | None:
    bindings = [
        (field, str(report[field]))
        for field in ("init_checkpoint_sha256", "grow_from_checkpoint_sha256")
        if report.get(field)
    ]
    if len(bindings) > 1:
        raise SystemExit("training report declares multiple learner parents")
    if not bindings:
        return None
    field, value = bindings[0]
    if not value.startswith("sha256:") or len(value) != 71:
        raise SystemExit(f"training report {field} is malformed")
    return field, value


def _load_parent(
    prepared: Mapping[str, Any],
    parent_checkpoint_path: Path,
    *,
    require_report_binding: bool,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    path = parent_checkpoint_path.resolve(strict=True)
    sha256 = _sha256(path)
    binding = _report_parent_binding(prepared["report"])
    if require_report_binding and binding is None:
        raise SystemExit(
            "teacher-gap probe requires a report-authenticated learner parent"
        )
    if binding is not None and sha256 != binding[1]:
        raise SystemExit(
            f"--parent-checkpoint bytes differ from training report {binding[0]}"
        )
    policy = _load_policy(
        str(_required(prepared["report"], "arch")), path, str(prepared["device"])
    )
    surface = _policy_input_surface(prepared, policy, role="parent checkpoint")
    return (
        policy,
        {
            "path": str(path),
            "sha256": sha256,
            "report_binding_field": None if binding is None else binding[0],
        },
        surface,
    )


def _evaluate_policy_metrics(
    prepared: Mapping[str, Any], policy, *, input_surface: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate one policy on the already-authenticated shared holdout."""

    report = prepared["report"]
    train_bc = prepared["train_bc"]
    train_bc._PUBLIC_CARD_COUNT_FEATURES_ENABLED = bool(
        input_surface["public_card_count_features"]
    )
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
    return train_bc.evaluate_bc_batches(
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
        policy_kl_anchor_direction=str(
            report.get("policy_kl_anchor_direction", "forward")
        ),
        value_uncertainty_loss_weight=float(
            _required(report, "value_uncertainty_loss_weight")
        ),
        aux_subgoal_loss_weight=float(_required(report, "aux_subgoal_loss_weight")),
        belief_resource_loss_weight=float(
            report.get("belief_resource_loss_weight", 0.0)
        ),
        moe_balance_loss_weight=float(_required(report, "moe_balance_loss_weight")),
        value_categorical_loss_weight=categorical_weight,
        value_hlgauss_sigma_ratio=float(_required(report, "value_hlgauss_sigma_ratio")),
        value_target_lambda=float(_required(report, "value_target_lambda")),
        value_root_blend_phases=blend_phases,
        value_root_blend_global_compat=blend_global,
        scalar_value_loss_readout=str(prepared["scalar_value_loss_readout"]),
        scalar_value_loss_scale=float(prepared["scalar_value_loss_scale"]),
    )


def _teacher_gap_projection(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "active_policy_teacher_gap_rows",
            "active_policy_kl_target_model_mean",
            "active_policy_kl_target_prior_mean",
            "active_policy_teacher_gap_closure",
        )
    }


def _value_quality_projection(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Project the objective-matched value statistic used for checkpoint safety."""

    denominators = metrics.get("loss_denominators")
    if not isinstance(denominators, Mapping):
        raise SystemExit("posthoc metrics have no value-loss denominator")
    try:
        primary = float(metrics["primary_value_loss"])
        scalar_mse = float(metrics["scalar_value_mse_diagnostic"])
        raw_value = float(metrics["value_loss"])
        kind = str(metrics["primary_value_loss_kind"])
        weighted_mass = float(denominators["value_loss"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit("posthoc value-quality metrics are malformed") from error
    if (
        kind != "scalar_mse"
        or not math.isfinite(primary)
        or not math.isfinite(scalar_mse)
        or not math.isfinite(raw_value)
        or weighted_mass <= 0.0
        or not math.isclose(primary, scalar_mse, rel_tol=0.0, abs_tol=1.0e-12)
        or not math.isclose(primary, raw_value, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise SystemExit(
            "posthoc value-quality metric does not match the scalar-MSE objective"
        )
    return {
        "schema_version": VALUE_QUALITY_SCHEMA,
        "selection_authority": True,
        "surface": "same_reconstructed_holdout_and_value_weight_measure",
        "metric": "primary_value_loss",
        "metric_kind": kind,
        "value": primary,
        "scalar_value_mse_diagnostic": scalar_mse,
        "value_weight_mass": weighted_mass,
    }


def _paired_parent_value_quality(
    *, candidate: Mapping[str, Any], parent: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        candidate.get("schema_version") != VALUE_QUALITY_SCHEMA
        or parent.get("schema_version") != VALUE_QUALITY_SCHEMA
        or candidate.get("selection_authority") is not True
        or parent.get("selection_authority") is not True
        or candidate.get("metric") != parent.get("metric")
        or candidate.get("metric_kind") != parent.get("metric_kind")
    ):
        raise SystemExit("candidate and parent value-quality objectives differ")
    try:
        candidate_value = float(candidate["value"])
        parent_value = float(parent["value"])
        candidate_mass = float(candidate["value_weight_mass"])
        parent_mass = float(parent["value_weight_mass"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit("paired parent value-quality metrics are malformed") from error
    if (
        not all(
            math.isfinite(value)
            for value in (candidate_value, parent_value, candidate_mass, parent_mass)
        )
        or candidate_mass <= 0.0
        or not math.isclose(candidate_mass, parent_mass, rel_tol=0.0, abs_tol=1.0e-9)
    ):
        raise SystemExit("candidate and parent value-quality measures differ")
    return {
        "schema_version": PAIRED_PARENT_VALUE_SCHEMA,
        "selection_authority": True,
        "surface": "same_holdout_same_objective_weights_fresh_exact_parent_forward",
        "metric": candidate["metric"],
        "metric_kind": candidate["metric_kind"],
        "value_weight_mass": candidate_mass,
        "parent_value": parent_value,
        "candidate_value": candidate_value,
        "candidate_minus_parent": candidate_value - parent_value,
    }


def _paired_parent_teacher_gap(
    *, candidate: Mapping[str, Any], parent: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare candidate and exact parent on one identical target surface.

    The stored ``prior_policy`` can represent a generation-time inference
    operator (for example D6 averaging) rather than the learner parent's raw
    forward pass.  It therefore remains useful telemetry but is not a valid
    checkpoint-selection baseline.
    """

    candidate_rows = int(candidate["active_policy_teacher_gap_rows"])
    parent_rows = int(parent["active_policy_teacher_gap_rows"])
    candidate_prior = float(candidate["active_policy_kl_target_prior_mean"])
    parent_prior = float(parent["active_policy_kl_target_prior_mean"])
    if candidate_rows <= 0 or candidate_rows != parent_rows:
        raise SystemExit("candidate and parent teacher-gap row surfaces differ")
    if not math.isclose(candidate_prior, parent_prior, rel_tol=0.0, abs_tol=1.0e-9):
        raise SystemExit("candidate and parent stored-prior target surfaces differ")
    candidate_kl = float(candidate["active_policy_kl_target_model_mean"])
    parent_kl = float(parent["active_policy_kl_target_model_mean"])
    if (
        not math.isfinite(candidate_kl)
        or not math.isfinite(parent_kl)
        or candidate_kl < -1.0e-9
        or parent_kl < -1.0e-9
    ):
        raise SystemExit("candidate or parent target KL is invalid")
    improvement = parent_kl - candidate_kl
    relative = improvement / parent_kl if parent_kl > 1.0e-8 else 0.0
    return {
        "schema_version": PAIRED_PARENT_GAP_SCHEMA,
        "selection_authority": True,
        "authority": "fresh_exact_report_bound_parent_forward",
        "surface": "same_holdout_same_targets_fresh_exact_parent_forward",
        "rows": candidate_rows,
        "parent_active_policy_kl_target_model_mean": parent_kl,
        "candidate_active_policy_kl_target_model_mean": candidate_kl,
        "absolute_teacher_gap_closure": improvement,
        "relative_teacher_gap_closure": relative,
        "improved_over_exact_parent": bool(improvement > 0.0),
        "stored_generation_prior": {
            "active_policy_kl_target_prior_mean": candidate_prior,
            "selection_authority": False,
            "semantic_role": "legacy_generation_operator_diagnostic_only",
        },
    }


def _evaluate_candidate(
    prepared: Mapping[str, Any],
    *,
    label: str,
    checkpoint_path: Path,
    parent_policy=None,
    parent_ref: Mapping[str, Any] | None = None,
    parent_surface: Mapping[str, Any] | None = None,
    parent_teacher_gap: Mapping[str, Any] | None = None,
    parent_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = checkpoint_path.resolve(strict=True)
    report = prepared["report"]
    train_bc = prepared["train_bc"]
    policy = _load_policy(str(_required(report, "arch")), path, str(prepared["device"]))
    input_surface = _policy_input_surface(prepared, policy, role=f"candidate {label!r}")
    if parent_surface is not None and input_surface != dict(parent_surface):
        raise SystemExit(
            f"candidate {label!r} and parent use different public input schemas"
        )
    metrics = _evaluate_policy_metrics(prepared, policy, input_surface=input_surface)
    teacher_gap = _teacher_gap_projection(metrics)
    value_quality = _value_quality_projection(metrics)
    result = {
        "label": label,
        "checkpoint": {"path": str(path), "sha256": _sha256(path)},
        "input_surface": input_surface,
        "teacher_gap": teacher_gap,
        "value_quality": value_quality,
        "teacher_gap_semantics": {
            "selection_authority": False,
            "semantic_role": "legacy_generation_operator_diagnostic_only",
            "authoritative_replacement": "paired_parent_teacher_gap",
        },
        "legacy_stored_generation_prior_teacher_gap": {
            **teacher_gap,
            "selection_authority": False,
            "semantic_role": "legacy_generation_operator_diagnostic_only",
        },
        "legacy_prior_kl": {
            key: metrics[key]
            for key in (
                "prior_kl_rows",
                "prior_kl_model_prior_mean",
                "prior_kl_target_prior_mean",
                "prior_kl_ratio",
            )
        },
        "metrics": metrics,
        "policy_teacher_gap_objective": dict(
            prepared["policy_teacher_gap_objective"]
        ),
    }
    if parent_policy is not None:
        assert (
            parent_ref is not None
            and parent_surface is not None
            and parent_teacher_gap is not None
            and parent_metrics is not None
        )
        result["parent_checkpoint_sha256"] = str(parent_ref["sha256"])
        result["parent_teacher_gap"] = dict(parent_teacher_gap)
        result["parent_target_kl_mean"] = float(
            parent_teacher_gap["active_policy_kl_target_model_mean"]
        )
        result["paired_parent_teacher_gap"] = _paired_parent_teacher_gap(
            candidate=teacher_gap, parent=parent_teacher_gap
        )
        parent_value_quality = _value_quality_projection(parent_metrics)
        result["parent_value_quality"] = parent_value_quality
        result["paired_parent_value_quality"] = _paired_parent_value_quality(
            candidate=value_quality,
            parent=parent_value_quality,
        )
        result["functional_dose_fingerprint"] = _functional_drift(
            train_bc=train_bc,
            parent_policy=parent_policy,
            candidate_policy=policy,
            data=prepared["data"],
            indices=prepared["validation_indices"],
            policy_weights=prepared["policy_weights"],
            batch_size=int(prepared["batch_size"]),
        )
    return result


def run_probe(
    *,
    report_path: Path,
    checkpoint_path: Path,
    data_path: Path,
    validation_manifest_path: Path,
    device: str,
    batch_size: int | None = None,
    parent_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint and, when supplied, its report-bound parent."""

    prepared = _prepare_probe(
        report_path=report_path,
        data_path=data_path,
        validation_manifest_path=validation_manifest_path,
        device=device,
        batch_size=batch_size,
    )
    parent_policy = parent_ref = parent_surface = parent_teacher_gap = parent_metrics = None
    if parent_checkpoint_path is not None:
        parent_policy, parent_ref, parent_surface = _load_parent(
            prepared,
            parent_checkpoint_path,
            require_report_binding=True,
        )
        parent_metrics = _evaluate_policy_metrics(
            prepared, parent_policy, input_surface=parent_surface
        )
        parent_teacher_gap = _teacher_gap_projection(parent_metrics)
    candidate = _evaluate_candidate(
        prepared,
        label="checkpoint",
        checkpoint_path=checkpoint_path,
        parent_policy=parent_policy,
        parent_ref=parent_ref,
        parent_surface=parent_surface,
        parent_teacher_gap=parent_teacher_gap,
        parent_metrics=parent_metrics,
    )
    shared = prepared["shared_holdout"]
    result = {
        "schema_version": "posthoc-checkpoint-teacher-gap/v1",
        "inputs": {
            "training_report": shared["training_report"],
            "checkpoint": candidate["checkpoint"],
            "memmap": shared["memmap"],
            "validation_manifest": shared["validation_manifest"],
        },
        "arch": prepared["report"]["arch"],
        "device": str(device),
        "batch_size": int(prepared["batch_size"]),
        "validation_rows": int(shared["validation_rows"]),
        "validation_game_seed_set_sha256": shared["validation_game_seed_set_sha256"],
        "shared_holdout": shared,
        "policy_teacher_gap_objective": candidate[
            "policy_teacher_gap_objective"
        ],
        "teacher_gap": candidate["teacher_gap"],
        "value_quality": candidate["value_quality"],
        "teacher_gap_semantics": candidate["teacher_gap_semantics"],
        "legacy_stored_generation_prior_teacher_gap": candidate[
            "legacy_stored_generation_prior_teacher_gap"
        ],
        "legacy_prior_kl": candidate["legacy_prior_kl"],
        "metrics": candidate["metrics"],
    }
    if parent_ref is not None:
        # Keep the established parent input projection small. Batch mode
        # exposes the training-report binding field in its richer shared record.
        result["inputs"]["parent_checkpoint"] = {
            "path": parent_ref["path"],
            "sha256": parent_ref["sha256"],
        }
        result["functional_dose_fingerprint"] = candidate["functional_dose_fingerprint"]
        result["parent_teacher_gap"] = candidate["parent_teacher_gap"]
        result["parent_target_kl_mean"] = candidate["parent_target_kl_mean"]
        result["paired_parent_teacher_gap"] = candidate["paired_parent_teacher_gap"]
        result["parent_value_quality"] = candidate["parent_value_quality"]
        result["paired_parent_value_quality"] = candidate[
            "paired_parent_value_quality"
        ]
    return result


def _step64_128_comparison(
    checkpoints: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    if "step64" not in checkpoints or "step128" not in checkpoints:
        return None
    fields = (
        "kl_parent_candidate_mean",
        "kl_candidate_parent_mean",
        "top1_flip_rate",
        "candidate_policy_entropy_mean",
        "policy_entropy_delta",
        "value_mean_absolute_delta",
        "value_root_mean_square_delta",
    )
    left = checkpoints["step64"]["functional_dose_fingerprint"]
    right = checkpoints["step128"]["functional_dose_fingerprint"]
    return {
        "schema_version": "posthoc-step64-step128-functional-dose-comparison/v1",
        "from": "step64",
        "to": "step128",
        "metrics": {
            field: {
                "step64": float(left[field]),
                "step128": float(right[field]),
                "step128_minus_step64": float(right[field]) - float(left[field]),
            }
            for field in fields
        },
    }


def run_batch_probe(
    *,
    report_path: Path,
    checkpoints: Sequence[tuple[str, Path]],
    parent_checkpoint_path: Path,
    data_path: Path,
    validation_manifest_path: Path,
    device: str,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Evaluate labeled candidates on one corpus against one bound parent."""

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
    parent_policy, parent_ref, parent_surface = _load_parent(
        prepared,
        parent_checkpoint_path,
        require_report_binding=True,
    )
    prepared["train_bc"]._PUBLIC_CARD_COUNT_FEATURES_ENABLED = bool(
        parent_surface["public_card_count_features"]
    )
    parent_metrics = _evaluate_policy_metrics(
        prepared, parent_policy, input_surface=parent_surface
    )
    parent_teacher_gap = _teacher_gap_projection(parent_metrics)
    results = {
        label: _evaluate_candidate(
            prepared,
            label=label,
            checkpoint_path=checkpoint_path,
            parent_policy=parent_policy,
            parent_ref=parent_ref,
            parent_surface=parent_surface,
            parent_teacher_gap=parent_teacher_gap,
            parent_metrics=parent_metrics,
        )
        for label, checkpoint_path in checkpoints
    }
    shared_holdout = dict(prepared["shared_holdout"])
    comparison_semantics = {
        "schema_version": "posthoc-parent-comparison-surface/v1",
        "holdout_identity_sha256": shared_holdout["identity_sha256"],
        "parent_checkpoint_sha256": parent_ref["sha256"],
        "input_surface": parent_surface,
    }
    shared_holdout.update(
        {
            "parent_checkpoint": parent_ref,
            "parent_teacher_gap": parent_teacher_gap,
            "parent_value_quality": _value_quality_projection(parent_metrics),
            "parent_target_kl_mean": float(
                parent_teacher_gap["active_policy_kl_target_model_mean"]
            ),
            "input_surface": parent_surface,
            "comparison_identity_sha256": _canonical_sha256(comparison_semantics),
        }
    )
    output = {
        "schema_version": "posthoc-checkpoint-teacher-gap-batch/v1",
        "arch": prepared["report"]["arch"],
        "device": str(device),
        "batch_size": int(prepared["batch_size"]),
        "checkpoint_order": labels,
        "shared_holdout": shared_holdout,
        "policy_teacher_gap_objective": dict(
            prepared["policy_teacher_gap_objective"]
        ),
        "checkpoints": results,
    }
    comparison = _step64_128_comparison(results)
    if comparison is not None:
        output["dose_comparison"] = comparison
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
            "Evaluate a labeled candidate against the shared parent; repeat the "
            "flag. step64 and step128 labels also emit a direct dose comparison."
        ),
    )
    parser.add_argument(
        "--parent-checkpoint",
        type=Path,
        help=(
            "Authenticated function-preserving learner initializer. When supplied, "
            "emit parent→candidate KL/top-1/entropy/value movement on the exact holdout."
        ),
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.checkpoint is not None and args.labeled_checkpoints:
        parser.error("--checkpoint cannot be combined with --labeled-checkpoint")
    if args.checkpoint is not None:
        result = run_probe(
            report_path=args.report,
            checkpoint_path=args.checkpoint,
            data_path=args.data,
            validation_manifest_path=args.validation_manifest,
            device=args.device,
            batch_size=args.batch_size,
            parent_checkpoint_path=args.parent_checkpoint,
        )
    elif args.labeled_checkpoints:
        if args.parent_checkpoint is None:
            parser.error("batch mode requires --parent-checkpoint")
        result = run_batch_probe(
            report_path=args.report,
            checkpoints=args.labeled_checkpoints,
            parent_checkpoint_path=args.parent_checkpoint,
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
