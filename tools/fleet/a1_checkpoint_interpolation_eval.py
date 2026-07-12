#!/usr/bin/env python3
"""Seal matched H100 panels for a diagnostic checkpoint interpolation curve."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.fleet import a1_h100_eval_fleet as fleet  # noqa: E402
from tools.fleet.a1_n256_lr_eval import APPROVED_SHAPES  # noqa: E402
from tools.champion_registry import ChampionRegistry  # noqa: E402


TRIAL_SCHEMA = "a1-checkpoint-interpolation-eval-v1"
RECEIPT_SCHEMA = "checkpoint-interpolation-receipt-v1"
EXPECTED_ALPHAS = (0.1, 0.25, 0.5, 1.0)
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class InterpolationEvalError(RuntimeError):
    """The interpolation panel cannot be authenticated or matched."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _load_receipt(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.expanduser().resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != RECEIPT_SCHEMA:
        raise InterpolationEvalError("unsupported interpolation receipt schema")
    if value.get("diagnostic_only") is not True or value.get("promotion_eligible") is not False:
        raise InterpolationEvalError("interpolation receipt is not diagnostic-only")
    stated = value.get("receipt_sha256")
    actual = _digest({key: item for key, item in value.items() if key != "receipt_sha256"})
    if stated != actual:
        raise InterpolationEvalError("interpolation receipt hash does not replay")
    for source in (value.get("base"), value.get("candidate")):
        if not isinstance(source, dict):
            raise InterpolationEvalError("interpolation receipt has invalid sources")
        source_path = Path(str(source.get("path"))).expanduser().resolve(strict=True)
        if fleet._sha256(source_path) != source.get("sha256"):  # noqa: SLF001
            raise InterpolationEvalError("interpolation source bytes drifted")
    outputs = value.get("outputs")
    if not isinstance(outputs, list) or tuple(row.get("alpha") for row in outputs) != EXPECTED_ALPHAS:
        raise InterpolationEvalError("interpolation receipt must contain the canonical alpha curve")
    for row in outputs:
        output = Path(str(row.get("path"))).expanduser().resolve(strict=True)
        if fleet._sha256(output) != row.get("sha256"):  # noqa: SLF001
            raise InterpolationEvalError("interpolation output bytes drifted")
    return value, path


def _alpha_label(alpha: float) -> str:
    text = f"{alpha:.4f}".rstrip("0").rstrip(".")
    return "a" + text.replace(".", "p")


def build_trial(
    *,
    manifest_path: Path,
    receipt_path: Path,
    internal_base_seed: int,
    external_base_seed: int,
    trial_id: str,
    output_dir: Path,
    registry_path: Path,
    candidate_c_scale: float,
    champion_c_scale: float,
    internal_pairs: int = 112,
    external_pairs: int = 56,
) -> dict[str, Any]:
    if not SAFE_ID.fullmatch(trial_id):
        raise InterpolationEvalError("trial_id must be a safe nonempty identifier")
    receipt, receipt_path = _load_receipt(receipt_path)
    manifest = fleet.load_manifest(manifest_path, expected_shapes=APPROVED_SHAPES)
    shape = {row["alias"]: int(row["gpu_count"]) for row in manifest["hosts"]}
    if shape != APPROVED_SHAPES:
        raise InterpolationEvalError("interpolation panel requires the approved 56-H100 fleet")
    champion = Path(receipt["base"]["path"])
    registry = ChampionRegistry.load(registry_path.expanduser().resolve(strict=True))
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise InterpolationEvalError(f"refusing existing trial directory: {output_dir}")

    cohort = f"{trial_id}-common"
    plans: dict[str, dict[str, Any]] = {}
    outputs: dict[str, dict[str, Any]] = {}
    for row in receipt["outputs"]:
        label = _alpha_label(float(row["alpha"]))
        checkpoint = Path(row["path"])
        outputs[label] = row
        plans[label] = fleet.build_plan(
            manifest,
            candidate=checkpoint,
            champion=champion,
            candidate_parent=champion,
            registry=registry,
            internal_pairs=internal_pairs,
            external_pairs=external_pairs,
            internal_base_seed=internal_base_seed,
            external_base_seed=external_base_seed,
            workers_per_gpu=fleet.DEFAULT_WORKERS_PER_GPU,
            iteration_id=f"{trial_id}-{label}",
            seed_cohort_id=cohort,
            candidate_c_scale=candidate_c_scale,
            champion_c_scale=champion_c_scale,
            comparison_mode="historical_comparison",
            historical_comparison_reason="diagnostic checkpoint interpolation curve",
        )
    if len({p["science_config_hash"] for p in plans.values()}) != 1:
        raise AssertionError("interpolation science configuration drifted")
    if len({json.dumps(p["pair_claims"], sort_keys=True) for p in plans.values()}) != 1:
        raise AssertionError("interpolation common-random-number cohort drifted")

    trial: dict[str, Any] = {
        "schema_version": TRIAL_SCHEMA,
        "trial_id": trial_id,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "manifest": {"path": str(manifest_path.expanduser().resolve(strict=True)), "hash": manifest["manifest_hash"]},
        "interpolation_receipt": {"path": str(receipt_path), "sha256": fleet._sha256(receipt_path)},  # noqa: SLF001
        "champion": receipt["base"],
        "source_candidate": receipt["candidate"],
        "panel": {"internal_pairs": internal_pairs, "external_pairs": external_pairs},
        "seed_cohort_id": cohort,
        "arms": {
            label: {
                "alpha": row["alpha"],
                "checkpoint": {"path": row["path"], "sha256": row["sha256"]},
                "plan_path": str(output_dir / f"{label}.plan.json"),
                "plan_hash": plans[label]["plan_hash"],
            }
            for label, row in outputs.items()
        },
    }
    trial["trial_hash"] = _digest(trial)
    output_dir.mkdir(parents=True)
    for label, plan in plans.items():
        fleet.write_new_readonly(output_dir / f"{label}.plan.json", plan)
    fleet.write_new_readonly(output_dir / "trial.json", trial)
    return trial


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--candidate-c-scale", type=float, required=True)
    parser.add_argument("--champion-c-scale", type=float, required=True)
    parser.add_argument("--internal-base-seed", type=int, required=True)
    parser.add_argument("--external-base-seed", type=int, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--internal-pairs", type=int, default=112)
    parser.add_argument("--external-pairs", type=int, default=56)
    return parser


def main() -> None:
    args = _parser().parse_args()
    value = build_trial(
        manifest_path=args.manifest,
        receipt_path=args.receipt,
        internal_base_seed=args.internal_base_seed,
        external_base_seed=args.external_base_seed,
        trial_id=args.trial_id,
        output_dir=args.out_dir,
        registry_path=args.registry,
        candidate_c_scale=args.candidate_c_scale,
        champion_c_scale=args.champion_c_scale,
        internal_pairs=args.internal_pairs,
        external_pairs=args.external_pairs,
    )
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
