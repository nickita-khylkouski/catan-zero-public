#!/usr/bin/env python3
"""Run one arm of the compact coherent-n128 joint learner campaign.

The eight arms are independent single-GPU updates from the same exact
function-preserving f7 initializer.  This runner deliberately bypasses the
production one-dose topology: it is a diagnostic experiment that isolates
policy dose, continued value learning, and a parent-policy trust region.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from catan_zero.rl.config_cli import _coerce_config_value  # noqa: E402
from catan_zero.rl.pipeline_configs import config_from_payload  # noqa: E402
from tools import train as canonical_train  # noqa: E402
from tools import train_bc  # noqa: E402


SCHEMA = "a1-coherent-joint-learner-arm-v1"
CHECKPOINT_STEPS = "8,12,16,24,32,48,64,96,128,160"
VALIDATION_RANGES = ",".join(
    f"{96_000_000_000 + lane * 10_000}:{96_000_000_007 + lane * 10_000}"
    for lane in range(8)
)
ARMS: dict[str, dict[str, Any]] = {
    "FULL192": {
        "policy_dose_lr_area": 0.0,
        "post_policy_dose_value_trunk_grad_scale": 1.0,
    },
    "D32_FREEZE": {
        "policy_dose_lr_area": 0.001455502905504519,
        "post_policy_dose_value_trunk_grad_scale": 0.0,
    },
    "D64_FREEZE": {
        "policy_dose_lr_area": 0.0030042999061211844,
        "post_policy_dose_value_trunk_grad_scale": 0.0,
    },
    "D128_FREEZE": {
        "policy_dose_lr_area": 0.0039,
        "post_policy_dose_value_trunk_grad_scale": 0.0,
    },
    "D32_SHARED": {
        "policy_dose_lr_area": 0.001455502905504519,
        "post_policy_dose_value_trunk_grad_scale": 1.0,
    },
    "D32_TRUST012": {
        "policy_dose_lr_area": 0.001455502905504519,
        "post_policy_dose_value_trunk_grad_scale": 1.0,
        "policy_kl_target": 0.012,
    },
    "D64_TRUST012": {
        "policy_dose_lr_area": 0.0030042999061211844,
        "post_policy_dose_value_trunk_grad_scale": 1.0,
        "policy_kl_target": 0.012,
    },
    "D32_TRUST012_BCE": {
        "policy_dose_lr_area": 0.001455502905504519,
        "post_policy_dose_value_trunk_grad_scale": 1.0,
        "policy_kl_target": 0.012,
        "scalar_value_objective": "binary_win_bce",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load_base_config(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return config_from_payload(payload["train_config"])


def _namespace(
    *,
    arm: str,
    base_config: Path,
    data: Path,
    initializer: Path,
    initializer_sha256: str,
    target_identity_sha256: str,
    output: Path,
    device: str,
) -> argparse.Namespace:
    treatment = ARMS[arm]
    config = dataclasses.replace(
        _load_base_config(base_config),
        batch_size=512,
        epochs=999,
        max_steps=192,
        exact_max_steps=True,
        lr=6e-5,
        lr_warmup_steps=16,
        lr_schedule="flat",
        validation_fraction=0.125,
        validation_max_samples=0,
        value_trunk_grad_scale=1.0,
        value_player_outcome_balance_mode="sampler_balanced_v1",
        value_phase_weights="none",
        policy_aux_active_batch_size=0,
        per_game_policy_surprise_weighting=False,
        policy_surprise_weight=0.0,
        target_reliability_confidence_weighting=False,
        public_card_lr_mult=1.0,
        policy_dose_lr_area=float(treatment["policy_dose_lr_area"]),
        policy_dose_reference_global_batch_size=(
            512 if float(treatment["policy_dose_lr_area"]) > 0.0 else 0
        ),
        post_policy_dose_value_trunk_grad_scale=float(
            treatment["post_policy_dose_value_trunk_grad_scale"]
        ),
        policy_kl_anchor_direction="forward",
        policy_kl_anchor_weight=0.0,
        policy_kl_dual_lr=1.0,
        policy_kl_max_weight=1.0,
        policy_kl_target=treatment.get("policy_kl_target"),
    )
    parser = train_bc.build_parser()
    actions = {
        action.dest: action
        for action in parser._actions  # noqa: SLF001
        if action.option_strings
    }
    args = canonical_train._engine_default_namespace(parser)  # noqa: SLF001
    settings = dict(config.field_values())
    settings.update(
        {
            "data": str(data),
            "data_format": "memmap",
            "checkpoint": str(output / "candidate.pt"),
            "report": str(output / "report.json"),
            "init_checkpoint": str(initializer),
            "device": device,
            "host_lock_file": str(output / "train.lock"),
            "allow_concurrent_bc": True,
            # This is one authenticated coherent-n128 corpus, not a composite
            # with component sampling ratios.  The coverage sampler is only
            # defined for such composites and cannot normalize the adaptive
            # parent-KL objective.  With no component weights, the canonical
            # weighted-replacement mode falls back to the ordinary unweighted
            # base traversal used by this compact diagnostic.
            "base_sampler": "weighted_replacement_v1",
            "checkpoint_steps": CHECKPOINT_STEPS,
            "entity_feature_adapter_version": (
                "rust_entity_adapter_v5_meaningful_history_v2"
            ),
            "required_target_information_regime": (
                "public_belief_single_tree_v1"
            ),
            "skip_teacher_quality_gate": True,
            "trust_curated_data_quality": True,
            "require_35m_model": False,
            "require_production_35m_teacher": False,
            "scalar_value_loss_readout": "deployed_tanh",
            "scalar_value_loss_scale": 1.0,
            "scalar_value_objective": str(
                treatment.get("scalar_value_objective", "mse")
            ),
            "value_tower_split_layers": 0,
            "train_diagnostics_every_batches": 8,
            "objective_gradient_interference_every_batches": 8,
            "validation_game_seed_ranges": VALIDATION_RANGES,
            "accepted_policy_target_identity_sha256": [
                target_identity_sha256
            ],
        }
    )
    for name, value in settings.items():
        action = actions.get(name)
        if action is None:
            if name in {
                "data_fingerprint",
                "grow_from_checkpoint_sha256",
                "init_checkpoint_sha256",
                "meaningful_public_history_schema",
                "public_card_count_feature_schema",
                "training_excluded_game_seed_set_sha256",
                "validation_contract_file_sha256",
                "validation_game_seed_set_sha256",
            }:
                continue
            raise SystemExit(f"internal trainer has no setting {name!r}")
        if value is None and action.default is None:
            continue
        setattr(args, name, _coerce_config_value(action, value, parser))
    args._canonical_guard_argv = (
        "--data",
        str(data),
        "--checkpoint",
        str(output / "candidate.pt"),
        "--report",
        str(output / "report.json"),
        "--optimizer",
        str(args.optimizer),
        "--weight-decay",
        str(args.weight_decay),
        "--truncated-vp-margin-value-weight",
        str(args.truncated_vp_margin_value_weight),
        "--lr-schedule",
        str(args.lr_schedule),
        "--mask-hidden-info",
    )
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--initializer-sha256", required=True)
    parser.add_argument("--target-identity-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--go", action="store_true")
    public = parser.parse_args()
    public.output.mkdir(parents=True, exist_ok=True)
    args = _namespace(
        arm=public.arm,
        base_config=public.base_config,
        data=public.data,
        initializer=public.initializer,
        initializer_sha256=public.initializer_sha256,
        target_identity_sha256=public.target_identity_sha256,
        output=public.output,
        device=public.device,
    )
    plan = {
        "schema_version": SCHEMA,
        "arm": public.arm,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "initializer": {
            "path": str(public.initializer.resolve(strict=True)),
            "sha256": _sha256(public.initializer.resolve(strict=True)),
        },
        "data": str(public.data.resolve(strict=True)),
        "target_identity_sha256": public.target_identity_sha256,
        "treatment": copy.deepcopy(ARMS[public.arm]),
        "validation_game_seed_ranges": VALIDATION_RANGES,
        "checkpoint_steps": CHECKPOINT_STEPS,
    }
    plan["plan_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    (public.output / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not public.go:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    train_bc.main(args)


if __name__ == "__main__":
    main()
