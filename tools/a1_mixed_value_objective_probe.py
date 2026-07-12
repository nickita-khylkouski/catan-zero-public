#!/usr/bin/env python3
"""Prepare or explicitly launch the corrected mixed MSE/HL-Gauss probe.

Preparation is the default. GPU work requires ``--go`` and every generated
artifact is diagnostic-only and non-promotable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "a1-mixed-value-objective-probe-v1"
ALLOWED_LRS = {
    "6e-5": 0.00006,
    "1.2e-4": 0.00012,
    "2.4e-4": 0.00024,
}
ARMS = ("mse", "hlgauss")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file_ref(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": _file_sha(resolved)}


def _write_once_or_match(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise SystemExit(f"REFUSED: prepared artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(encoded, encoding="utf-8")
    os.chmod(temp, 0o444)
    os.replace(temp, path)


def _component(corpus: Path, validation: Path) -> dict[str, str]:
    corpus = corpus.expanduser().resolve(strict=True)
    validation = validation.expanduser().resolve(strict=True)
    if not corpus.is_dir():
        raise SystemExit(f"REFUSED: corpus is not a directory: {corpus}")
    meta_path = corpus / "corpus_meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"REFUSED: cannot read corpus metadata {meta_path}: {error}") from error
    inventory_sha = meta.get("payload_inventory_sha256") if isinstance(meta, dict) else None
    if not isinstance(inventory_sha, str) or not inventory_sha.startswith("sha256:"):
        raise SystemExit(f"REFUSED: corpus has no payload inventory binding: {corpus}")
    return {
        "corpus_dir": str(corpus),
        "corpus_meta_sha256": _file_sha(meta_path),
        "payload_inventory_sha256": inventory_sha,
        "validation_manifest": str(validation),
        "validation_manifest_sha256": _file_sha(validation),
    }


def _recipe(lr: float, objective: str) -> dict[str, Any]:
    if objective not in ARMS:
        raise ValueError(f"unknown value objective {objective!r}")
    return {
        "forced_row_value_weight": 0.1,
        "hlgauss_scalar_aux_loss_weight": 0.0,
        "loser_sample_weight": 1.0,
        "lr": lr,
        "per_game_policy_weight": True,
        "per_game_policy_weight_mode": "equal",
        "per_game_value_weight": True,
        "per_game_value_weight_mode": "sqrt",
        "value_categorical_bins": 33,
        "value_categorical_loss_weight": 0.0,
        "value_head_type": objective,
        "value_hlgauss_sigma_ratio": 0.75,
        "value_loss_weight": 0.25,
    }


def _descriptor(components: list[dict[str, str]], recipe: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "memmap_composite_v1",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "learner_recipe_overrides": recipe,
        "learner_recipe_overrides_sha256": _canonical_sha(recipe),
        "components": components,
    }


def _arm_command(
    *,
    python: Path,
    repo: Path,
    descriptor: Path,
    init_checkpoint: Path,
    arm_dir: Path,
    recipe: dict[str, Any],
) -> list[str]:
    command = [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        str(repo / "tools/train_bc.py"),
        "--data",
        str(descriptor),
        "--data-format",
        "memmap",
        "--init-checkpoint",
        str(init_checkpoint),
        "--arch",
        "entity_graph",
        "--epochs",
        "1",
        "--batch-size",
        "512",
        "--grad-accum-steps",
        "1",
        "--seed",
        "1",
        "--validation-max-samples",
        "0",
        "--no-resume-optimizer",
        "--save-each-epoch",
        "--skip-teacher-quality-gate",
        "--trust-curated-data-quality",
        "--allow-concurrent-bc",
        "--checkpoint",
        str(arm_dir / "checkpoint.pt"),
        "--report",
        str(arm_dir / "report.json"),
    ]
    boolean_flags = {
        "per_game_policy_weight",
        "per_game_value_weight",
    }
    for key in sorted(recipe):
        value = recipe[key]
        flag = "--" + key.replace("_", "-")
        if key in boolean_flags:
            if value:
                command.append(flag)
        else:
            command.extend((flag, str(value)))
    return command


def _assert_matched(arms: dict[str, dict[str, Any]]) -> None:
    mse = dict(arms["mse"]["recipe"])
    hlgauss = dict(arms["hlgauss"]["recipe"])
    mse_objective = mse.pop("value_head_type")
    hlgauss_objective = hlgauss.pop("value_head_type")
    if mse != hlgauss or (mse_objective, hlgauss_objective) != ARMS:
        raise SystemExit("REFUSED: value arms differ outside the primary value objective")


def _receipt(arm: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    checkpoint = Path(arm["checkpoint"])
    report = Path(arm["report"])
    payload = {
        "schema_version": "a1-mixed-value-objective-training-receipt-v1",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "arm": arm["arm"],
        "objective": arm["recipe"]["value_head_type"],
        "recipe": arm["recipe"],
        "recipe_sha256": _canonical_sha(arm["recipe"]),
        "experiment_manifest": _file_ref(manifest_path),
        "descriptor": _file_ref(Path(arm["descriptor"])),
        "checkpoint": _file_ref(checkpoint),
        "report": _file_ref(report),
    }
    payload["receipt_sha256"] = _canonical_sha(payload)
    return payload


def _launch(manifest: dict[str, Any], manifest_path: Path) -> None:
    for arm_name in ARMS:
        arm = manifest["arms"][arm_name]
        receipt_path = Path(arm["receipt"])
        if receipt_path.exists():
            raise SystemExit(f"REFUSED: completed arm already has a receipt: {arm_name}")
        for output in (arm["checkpoint"], arm["report"]):
            if Path(output).exists():
                raise SystemExit(f"REFUSED: partial output exists without receipt: {output}")
        arm_dir = receipt_path.parent
        arm_dir.mkdir(parents=True, exist_ok=True)
        with (arm_dir / "stdout.log").open("x", encoding="utf-8") as stdout, (
            arm_dir / "stderr.log"
        ).open("x", encoding="utf-8") as stderr:
            subprocess.run(arm["command"], check=True, stdout=stdout, stderr=stderr)
        if not Path(arm["checkpoint"]).is_file() or not Path(arm["report"]).is_file():
            raise SystemExit(f"REFUSED: {arm_name} completed without checkpoint/report")
        _write_once_or_match(receipt_path, _receipt(arm, manifest_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr", required=True, choices=tuple(ALLOWED_LRS))
    parser.add_argument("--n256-corpus", required=True, type=Path)
    parser.add_argument("--n256-validation", required=True, type=Path)
    parser.add_argument("--n128-corpus", required=True, type=Path)
    parser.add_argument("--n128-validation", required=True, type=Path)
    parser.add_argument(
        "--categorical-init-checkpoint",
        required=True,
        type=Path,
        help="Shared behavior-preserving 33-bin-capable initialization for both arms.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--go", action="store_true", help="Launch both arms after preparation.")
    return parser


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    repo = args.repo.expanduser().resolve(strict=True)
    python = args.python.expanduser().resolve(strict=True)
    init_checkpoint = args.categorical_init_checkpoint.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    components = [
        _component(args.n256_corpus, args.n256_validation),
        _component(args.n128_corpus, args.n128_validation),
    ]
    lr = ALLOWED_LRS[args.lr]
    arms: dict[str, dict[str, Any]] = {}
    for arm_name in ARMS:
        arm_dir = output_root / arm_name
        recipe = _recipe(lr, arm_name)
        descriptor_path = arm_dir / "memmap_composite.json"
        descriptor = _descriptor(components, recipe)
        _write_once_or_match(descriptor_path, descriptor)
        command = _arm_command(
            python=python,
            repo=repo,
            descriptor=descriptor_path,
            init_checkpoint=init_checkpoint,
            arm_dir=arm_dir,
            recipe=recipe,
        )
        arms[arm_name] = {
            "arm": arm_name,
            "recipe": recipe,
            "recipe_sha256": _canonical_sha(recipe),
            "descriptor": str(descriptor_path),
            "descriptor_sha256": _file_sha(descriptor_path),
            "checkpoint": str(arm_dir / "checkpoint.pt"),
            "report": str(arm_dir / "report.json"),
            "receipt": str(arm_dir / "training.receipt.json"),
            "command": command,
            "command_sha256": _canonical_sha(command),
        }
    _assert_matched(arms)
    manifest = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "lr_curve_verdict_selection": args.lr,
        "lr": lr,
        "topology": {
            "world_size": 8,
            "local_batch_size": 512,
            "grad_accum_steps": 1,
            "global_batch_size": 4096,
            "data_format": "memmap_composite_v1",
            "global_row_shuffle": True,
            "no_copy": True,
        },
        "categorical_init_checkpoint": _file_ref(init_checkpoint),
        "components": components,
        "arms": arms,
        "matched_variable": "value_head_type",
    }
    manifest["manifest_sha256"] = _canonical_sha(manifest)
    manifest_path = output_root / "experiment.manifest.json"
    _write_once_or_match(manifest_path, manifest)
    return manifest, manifest_path


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest, manifest_path = prepare(args)
    if args.go:
        _launch(manifest, manifest_path)
    else:
        print(
            json.dumps(
                {
                    "prepared": str(manifest_path),
                    "lr": manifest["lr"],
                    "arms": list(ARMS),
                    "world_size": 8,
                    "global_batch_size": 4096,
                    "launched": False,
                    "diagnostic_only": True,
                    "promotion_eligible": False,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
