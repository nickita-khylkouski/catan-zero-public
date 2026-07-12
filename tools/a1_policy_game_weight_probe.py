#!/usr/bin/env python3
"""Prepare a sealed equal-vs-sqrt per-game POLICY-weight ablation.

This is diagnostic-only.  Both arms use the same composite corpus, validation
manifests, initialization, optimizer recipe, LR, topology, and step budget; the
only changed field is ``per_game_policy_weight_mode``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from tools.a1_mixed_value_objective_probe import (
    ALLOWED_LRS,
    _canonical_sha,
    _component,
    _descriptor,
    _file_ref,
    _lexical_python_executable,
    _write_once_or_match,
)

SCHEMA = "a1-policy-game-weight-probe-v1"
ARMS = ("equal", "sqrt")


def _recipe(lr: float, mode: str) -> dict[str, Any]:
    if mode not in ARMS:
        raise ValueError(mode)
    return {
        "forced_row_value_weight": 0.1,
        "hlgauss_scalar_aux_loss_weight": 0.0,
        "loser_sample_weight": 1.0,
        "lr": lr,
        "per_game_policy_weight": True,
        "per_game_policy_weight_mode": mode,
        "per_game_value_weight": True,
        "per_game_value_weight_mode": "sqrt",
        "value_categorical_bins": 0,
        "value_categorical_loss_weight": 0.0,
        "value_head_type": "mse",
        "value_hlgauss_sigma_ratio": 0.75,
        "value_loss_weight": 0.25,
    }


def _command(*, python: Path, repo: Path, descriptor: Path, init: Path,
             out: Path, recipe: dict[str, Any], max_steps: int) -> list[str]:
    command = [
        str(python), "-m", "torch.distributed.run", "--standalone",
        "--nproc-per-node=8", str(repo / "tools/train_bc.py"),
        "--data", str(descriptor), "--data-format", "memmap",
        "--init-checkpoint", str(init), "--arch", "entity_graph",
        "--epochs", "1", "--max-steps", str(max_steps),
        "--batch-size", "512", "--grad-accum-steps", "1", "--seed", "1",
        "--validation-max-samples", "0", "--no-resume-optimizer",
        "--skip-teacher-quality-gate", "--trust-curated-data-quality",
        "--allow-concurrent-bc", "--require-35m-model",
        "--checkpoint", str(out / "checkpoint.pt"),
        "--report", str(out / "report.json"),
    ]
    booleans = {"per_game_policy_weight", "per_game_value_weight"}
    for key in sorted(recipe):
        value = recipe[key]
        flag = "--" + key.replace("_", "-")
        if key in booleans:
            if value:
                command.append(flag)
        else:
            command.extend((flag, str(value)))
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr", required=True, choices=tuple(ALLOWED_LRS))
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--n256-corpus", required=True, type=Path)
    parser.add_argument("--n256-validation", required=True, type=Path)
    parser.add_argument("--n128-corpus", required=True, type=Path)
    parser.add_argument("--n128-validation", required=True, type=Path)
    parser.add_argument("--init-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--go", action="store_true")
    return parser


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive for the bounded diagnostic")
    repo = args.repo.expanduser().resolve(strict=True)
    python = _lexical_python_executable(args.python)
    init = args.init_checkpoint.expanduser().resolve(strict=True)
    root = args.output_root.expanduser().resolve()
    components = [
        _component(args.n256_corpus, args.n256_validation),
        _component(args.n128_corpus, args.n128_validation),
    ]
    arms: dict[str, dict[str, Any]] = {}
    for mode in ARMS:
        out = root / mode
        recipe = _recipe(ALLOWED_LRS[args.lr], mode)
        descriptor = out / "memmap_composite.json"
        _write_once_or_match(descriptor, _descriptor(components, recipe))
        command = _command(
            python=python, repo=repo, descriptor=descriptor, init=init,
            out=out, recipe=recipe, max_steps=args.max_steps,
        )
        arms[mode] = {
            "arm": mode, "recipe": recipe,
            "recipe_sha256": _canonical_sha(recipe),
            "descriptor": str(descriptor), "command": command,
            "command_sha256": _canonical_sha(command),
            "checkpoint": str(out / "checkpoint.pt"),
            "report": str(out / "report.json"),
            "receipt": str(out / "training.receipt.json"),
        }
    common = dict(arms["equal"]["recipe"])
    common.pop("per_game_policy_weight_mode")
    comparison = dict(arms["sqrt"]["recipe"])
    comparison.pop("per_game_policy_weight_mode")
    if common != comparison:
        raise SystemExit("REFUSED: arms drift outside policy game-weight mode")
    manifest = {
        "schema_version": SCHEMA, "diagnostic_only": True,
        "promotion_eligible": False, "matched_variable": "per_game_policy_weight_mode",
        "lr_curve_verdict_selection": args.lr, "lr": ALLOWED_LRS[args.lr],
        "max_steps": args.max_steps,
        "topology": {"world_size": 8, "local_batch_size": 512,
                     "global_batch_size": 4096, "global_row_shuffle": True,
                     "data_format": "memmap_composite_v1", "no_copy": True},
        "init_checkpoint": _file_ref(init), "launcher": _file_ref(Path(__file__)),
        "components": components, "arms": arms,
    }
    manifest["manifest_sha256"] = _canonical_sha(manifest)
    path = root / "experiment.manifest.json"
    _write_once_or_match(path, manifest)
    return manifest, path


def _launch(manifest: dict[str, Any], path: Path) -> None:
    for mode in ARMS:
        arm = manifest["arms"][mode]
        receipt = Path(arm["receipt"])
        if receipt.exists() or Path(arm["checkpoint"]).exists() or Path(arm["report"]).exists():
            raise SystemExit(f"REFUSED: partial or completed output exists for {mode}")
        receipt.parent.mkdir(parents=True, exist_ok=True)
        with (receipt.parent / "stdout.log").open("x") as stdout, \
             (receipt.parent / "stderr.log").open("x") as stderr:
            subprocess.run(arm["command"], check=True, stdout=stdout, stderr=stderr)
        payload = {
            "schema_version": "a1-policy-game-weight-training-receipt-v1",
            "diagnostic_only": True, "promotion_eligible": False, "arm": mode,
            "experiment_manifest": _file_ref(path),
            "checkpoint": _file_ref(Path(arm["checkpoint"])),
            "report": _file_ref(Path(arm["report"])),
        }
        payload["receipt_sha256"] = _canonical_sha(payload)
        _write_once_or_match(receipt, payload)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest, path = prepare(args)
    if args.go:
        _launch(manifest, path)
    else:
        print(json.dumps({"prepared": str(path), "arms": list(ARMS),
                          "max_steps": args.max_steps, "launched": False}, sort_keys=True))


if __name__ == "__main__":
    main()
