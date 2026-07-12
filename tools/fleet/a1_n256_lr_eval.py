#!/usr/bin/env python3
"""Seal matched diagnostic evaluation plans for the three n256 LR points.

This is an experiment controller, not a promotion path.  It refuses to create
any evaluation plan until the low, midpoint, and high LR training receipts all
authenticate their completed checkpoints.  Every arm then reuses the same
validation seeds and the canonical n128 search operator through an explicit
common-random-number cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import a1_dual_arm_train as training  # noqa: E402
from tools.fleet import a1_h100_eval_fleet as fleet  # noqa: E402


TRIAL_SCHEMA = "a1-n256-lr-micro-eval-v1"
ARM_SPECS = {
    "lr60u": (0.00006, "n256-lr-response-lr60u-loser1"),
    "lr120u": (0.00012, "all-196k-corrective-lr120u-loser1"),
    "lr240u": (0.00024, "n256-lr-response-lr240u-loser1"),
}
APPROVED_SHAPES = {
    "c1": 4,
    "c2": 4,
    "c3": 4,
    "c4": 4,
    "c5": 4,
    "c6": 4,
    "h100-8a": 8,
    "h100-8b": 8,
    "h100-8c": 8,
    "h100-8d": 8,
}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class TrialError(RuntimeError):
    """The diagnostic trial cannot be proved matched and complete."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _checkpoint_from_receipt(path: Path, label: str) -> dict[str, str]:
    receipt_path = path.expanduser().resolve(strict=True)
    receipt = training.verify_receipt(receipt_path)
    expected_lr, expected_ablation = ARM_SPECS[label]
    if (receipt.get("arm_id"), receipt.get("subset_id")) != (
        "n256",
        "full-56k",
    ):
        raise TrialError(f"{label} receipt is not the full n256 dose")
    ablation = receipt.get("inputs", {}).get("learner_ablation", {})
    if (
        ablation.get("ablation_id") != expected_ablation
        or ablation.get("diagnostic_only") is not True
        or ablation.get("promotion_eligible") is not False
    ):
        raise TrialError(f"{label} receipt has wrong diagnostic provenance")
    recipe = ablation.get("effective_recipe", {})
    if (
        recipe.get("lr") != expected_lr
        or recipe.get("loser_sample_weight") != 1.0
        or recipe.get("epochs") != 1
    ):
        raise TrialError(f"{label} receipt has wrong effective recipe")
    checkpoint = receipt.get("outputs", {}).get("checkpoint")
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"path", "sha256"}:
        raise TrialError(f"{label} receipt has no sealed checkpoint reference")
    checkpoint_path = Path(str(checkpoint["path"])).expanduser().resolve(strict=True)
    actual = fleet._sha256(checkpoint_path)  # noqa: SLF001
    if checkpoint["sha256"] != actual:
        raise TrialError(f"{label} checkpoint bytes differ from its receipt")
    return {
        "path": str(checkpoint_path),
        "sha256": actual,
        "receipt": str(receipt_path),
        "receipt_sha256": fleet._sha256(receipt_path),  # noqa: SLF001
    }


def _require_approved_fleet(manifest: dict[str, Any]) -> None:
    shape = {row["alias"]: int(row["gpu_count"]) for row in manifest["hosts"]}
    if shape != APPROVED_SHAPES:
        raise TrialError(
            "LR diagnostic launch allowlist requires the approved 56-H100 fleet"
        )


def build_trial(
    *,
    manifest_path: Path,
    champion: Path,
    receipts: dict[str, Path],
    internal_base_seed: int,
    external_base_seed: int,
    trial_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    if not SAFE_ID.fullmatch(trial_id):
        raise TrialError("trial_id must be a safe nonempty identifier")
    if set(receipts) != set(ARM_SPECS):
        raise TrialError("all three LR receipts are required")
    # Establish the three-receipt barrier before parsing any one arm.  This
    # keeps a partially completed LR sweep from producing a misleading plan.
    for label in ARM_SPECS:
        receipts[label].expanduser().resolve(strict=True)
    manifest = fleet.load_manifest(manifest_path)
    _require_approved_fleet(manifest)
    champion = champion.expanduser().resolve(strict=True)
    checkpoints = {
        label: _checkpoint_from_receipt(receipts[label], label)
        for label in ARM_SPECS
    }
    if len({row["sha256"] for row in checkpoints.values()}) != len(ARM_SPECS):
        raise TrialError("LR receipts do not bind three distinct checkpoints")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise TrialError(f"refusing existing trial directory: {output_dir}")
    cohort = f"{trial_id}-common"
    plans: dict[str, dict[str, Any]] = {}
    for label in ARM_SPECS:
        plans[label] = fleet.build_plan(
            manifest,
            candidate=Path(checkpoints[label]["path"]),
            champion=champion,
            internal_pairs=112,
            external_pairs=56,
            internal_base_seed=internal_base_seed,
            external_base_seed=external_base_seed,
            workers_per_gpu=fleet.DEFAULT_WORKERS_PER_GPU,
            iteration_id=f"{trial_id}-{label}",
            seed_cohort_id=cohort,
        )
    science = {plan["science_config_hash"] for plan in plans.values()}
    claims = {json.dumps(plan["pair_claims"], sort_keys=True) for plan in plans.values()}
    if len(science) != 1 or len(claims) != 1:
        raise AssertionError("matched LR plans drifted")
    trial: dict[str, Any] = {
        "schema_version": TRIAL_SCHEMA,
        "trial_id": trial_id,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "manifest": {
            "path": str(manifest_path.expanduser().resolve(strict=True)),
            "hash": manifest["manifest_hash"],
            "physical_gpus": sum(APPROVED_SHAPES.values()),
        },
        "champion": {
            "path": str(champion),
            "sha256": fleet._sha256(champion),  # noqa: SLF001
        },
        "micro_panel": {"internal_pairs": 112, "external_pairs": 56},
        "seed_cohort_id": cohort,
        "science_config_hash": next(iter(science)),
        "arms": {
            label: {
                "lr": ARM_SPECS[label][0],
                "checkpoint": checkpoints[label],
                "plan_path": str(output_dir / f"{label}.plan.json"),
                "plan_hash": plans[label]["plan_hash"],
            }
            for label in ARM_SPECS
        },
        "execution_order": [
            {"arm": label, "phase": phase}
            for label in ARM_SPECS
            for phase in ("internal", "external")
        ],
        "winner_full_panel": {
            "internal_pairs": 600,
            "external_pairs": 500,
            "must_use_fresh_val_seeds": True,
        },
    }
    trial["trial_hash"] = _digest(trial)
    output_dir.mkdir(parents=True)
    for label, plan in plans.items():
        fleet.write_new_readonly(output_dir / f"{label}.plan.json", plan)
    fleet.write_new_readonly(output_dir / "trial.json", trial)
    return trial


def render_commands(trial_path: Path) -> dict[str, Any]:
    trial_path = trial_path.expanduser().resolve(strict=True)
    trial = json.loads(trial_path.read_text(encoding="utf-8"))
    if trial.get("schema_version") != TRIAL_SCHEMA:
        raise TrialError("unsupported LR trial schema")
    stated = trial.get("trial_hash")
    actual = _digest({key: value for key, value in trial.items() if key != "trial_hash"})
    if stated != actual:
        raise TrialError("LR trial hash does not replay")
    manifest = trial["manifest"]["path"]
    commands = []
    for row in trial["execution_order"]:
        plan = trial["arms"][row["arm"]]["plan_path"]
        phase = row["phase"]
        commands.append(
            {
                **row,
                "launch": [
                    sys.executable,
                    "tools/fleet/a1_h100_eval_fleet.py",
                    "--manifest",
                    manifest,
                    "launch",
                    "--plan",
                    plan,
                    "--phase",
                    phase,
                    "--go",
                ],
                "status": [
                    sys.executable,
                    "tools/fleet/a1_h100_eval_fleet.py",
                    "--manifest",
                    manifest,
                    "status",
                    "--plan",
                    plan,
                    "--phase",
                    phase,
                ],
            }
        )
    return {"trial_hash": stated, "diagnostic_only": True, "commands": commands}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--champion", type=Path, required=True)
    plan.add_argument("--low-receipt", type=Path, required=True)
    plan.add_argument("--mid-receipt", type=Path, required=True)
    plan.add_argument("--high-receipt", type=Path, required=True)
    plan.add_argument("--internal-base-seed", type=int, required=True)
    plan.add_argument("--external-base-seed", type=int, required=True)
    plan.add_argument("--trial-id", required=True)
    plan.add_argument("--out-dir", type=Path, required=True)
    render = commands.add_parser("commands")
    render.add_argument("--trial", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            result = build_trial(
                manifest_path=args.manifest,
                champion=args.champion,
                receipts={
                    "lr60u": args.low_receipt,
                    "lr120u": args.mid_receipt,
                    "lr240u": args.high_receipt,
                },
                internal_base_seed=args.internal_base_seed,
                external_base_seed=args.external_base_seed,
                trial_id=args.trial_id,
                output_dir=args.out_dir,
            )
        else:
            result = render_commands(args.trial)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        TrialError,
        training.DualTrainError,
        fleet.FleetError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
