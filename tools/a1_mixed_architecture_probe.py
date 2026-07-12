#!/usr/bin/env python3
"""Seal or explicitly launch the matched mixed-data relational architecture A/B."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_mixed_value_objective_probe as common  # noqa: E402


SCHEMA = "a1-mixed-relational-architecture-probe-v1"
ARMS = ("baseline", "relational_action")
SOURCE_PATHS = (
    "tools/a1_mixed_architecture_probe.py",
    "tools/a1_mixed_value_objective_probe.py",
    "tools/audit_memmap_architecture_targets.py",
    "tools/train_bc.py",
    "tools/mixed_memmap_corpus.py",
    "src/catan_zero/rl/entity_token_policy.py",
    "src/catan_zero/rl/relational_trunks.py",
)


def _architecture(arm: str) -> dict[str, Any]:
    if arm == "baseline":
        return {
            "entity_state_trunk": "transformer",
            "relational_block_pattern": "",
            "relational_ff_size": 0,
            "relational_bases": 4,
            "relational_action_cross_layers": 1,
            "effective_action_target_gather": False,
            "effective_action_cross_attention_layers": 0,
            "effective_graph_relational_encoding": False,
            "effective_edge_policy_head": False,
        }
    if arm == "relational_action":
        return {
            "entity_state_trunk": "rrt",
            "relational_block_pattern": "RRTRRT",
            "relational_ff_size": 0,
            "relational_bases": 4,
            "relational_action_cross_layers": 1,
            "effective_action_target_gather": True,
            "effective_action_cross_attention_layers": 1,
            "effective_graph_relational_encoding": True,
            "effective_edge_policy_head": False,
        }
    raise ValueError(f"unknown architecture arm {arm!r}")


def _training_recipe(lr: float, max_steps: int) -> dict[str, Any]:
    return {
        "amp": "bf16",
        "attention_heads": 8,
        "batch_size": 512,
        "epochs": 1,
        "forced_row_value_weight": 0.1,
        "global_batch_size": 4096,
        "grad_accum_steps": 1,
        "graph_dropout": 0.05,
        "graph_layers": 6,
        "hidden_size": 640,
        "hlgauss_scalar_aux_loss_weight": 0.0,
        "loser_sample_weight": 1.0,
        "lr": lr,
        "lr_schedule": "flat",
        "lr_warmup_steps": 100,
        "max_steps": max_steps,
        "optimizer": "adam",
        "per_game_policy_weight": True,
        "per_game_policy_weight_mode": "equal",
        "per_game_value_weight": True,
        "per_game_value_weight_mode": "sqrt",
        "seed": 1,
        "value_categorical_loss_weight": 0.0,
        "value_head_type": "mse",
        "value_hlgauss_sigma_ratio": 0.75,
        "value_loss_weight": 0.25,
        "weight_decay": 0.0,
    }


def _descriptor_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "forced_row_value_weight",
        "hlgauss_scalar_aux_loss_weight",
        "loser_sample_weight",
        "lr",
        "per_game_policy_weight",
        "per_game_policy_weight_mode",
        "per_game_value_weight",
        "per_game_value_weight_mode",
        "value_categorical_loss_weight",
        "value_head_type",
        "value_hlgauss_sigma_ratio",
        "value_loss_weight",
    }
    return {key: recipe[key] for key in sorted(fields)}


def _validate_audit(
    path: Path, components: list[dict[str, str]]
) -> tuple[dict[str, Any], dict[str, str]]:
    resolved = path.expanduser().resolve(strict=True)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"REFUSED: cannot parse architecture audit: {error}") from error
    if payload.get("schema_version") != "memmap-architecture-target-audit-bundle-v1":
        raise SystemExit("REFUSED: architecture audit schema drift")
    verdict = payload.get("verdict")
    if not isinstance(verdict, dict) or (
        verdict.get("architecture_action_probe_runnable") is not True
        or verdict.get("requires_generator_changes_for_action_probe") is not False
        or verdict.get("event_relation_probe_runnable") is not False
    ):
        raise SystemExit("REFUSED: audit does not authorize action-only architecture probe")
    audits = payload.get("audits")
    expected_dirs = [row["corpus_dir"] for row in components]
    if not isinstance(audits, list) or [row.get("corpus_dir") for row in audits] != expected_dirs:
        raise SystemExit("REFUSED: architecture audit corpus order differs from experiment")
    for row in audits:
        viability = row.get("viability", {})
        legal = row.get("legal_action_targets", {})
        graph = row.get("graph_incidence", {})
        event = row.get("event_targets", {})
        if not (
            viability.get("action_target_gather") is True
            and viability.get("action_cross_attention") is True
            and viability.get("graph_relational_trunk") is True
            and viability.get("event_target_relations") is False
            and legal.get("out_of_range_target_rows") == 0
            and legal.get("invalid_legal_action_ids") == 0
            and legal.get("search_active_rows_with_any_target", 0) > 0
            and graph.get("out_of_range_ids") == 0
            and event.get("masked_events") == 0
            and event.get("events_with_any_target") == 0
        ):
            raise SystemExit("REFUSED: architecture audit corpus evidence is not viable")
    return payload, common._file_ref(resolved)


def _source_binding(repo: Path) -> dict[str, Any]:
    refs = {
        relative: common._file_ref(repo / relative)
        for relative in SOURCE_PATHS
    }
    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "git_commit": commit,
        "files": refs,
        "files_sha256": common._canonical_sha(refs),
    }


def _command(
    *,
    python: Path,
    repo: Path,
    descriptor: Path,
    initialization: Path,
    arm_dir: Path,
    recipe: dict[str, Any],
    architecture: dict[str, Any],
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
        "--grow-from-checkpoint",
        str(initialization),
        "--arch",
        "entity_graph",
        "--validation-max-samples",
        "0",
        "--no-resume-optimizer",
        "--no-fused-optimizer",
        "--no-relational-edge-policy-head",
        "--save-each-epoch",
        "--skip-teacher-quality-gate",
        "--trust-curated-data-quality",
        "--checkpoint",
        str(arm_dir / "checkpoint.pt"),
        "--report",
        str(arm_dir / "report.json"),
    ]
    boolean_fields = {"per_game_policy_weight", "per_game_value_weight"}
    omitted_fields = {"global_batch_size"}
    for key in sorted(recipe):
        if key in omitted_fields:
            continue
        value = recipe[key]
        flag = "--" + key.replace("_", "-")
        if key in boolean_fields:
            if value:
                command.append(flag)
        else:
            command.extend((flag, str(value)))
    for key in (
        "entity_state_trunk",
        "relational_block_pattern",
        "relational_ff_size",
        "relational_bases",
        "relational_action_cross_layers",
    ):
        value = architecture[key]
        if value != "":
            command.extend(("--" + key.replace("_", "-"), str(value)))
    return command


def _assert_matched(arms: dict[str, dict[str, Any]]) -> None:
    baseline = arms["baseline"]
    treatment = arms["relational_action"]
    for field in (
        "training_recipe",
        "training_recipe_sha256",
        "initialization",
        "descriptor",
        "architecture_audit",
    ):
        if baseline[field] != treatment[field]:
            raise SystemExit(f"REFUSED: matched architecture arms differ in {field}")
    if baseline["architecture"] == treatment["architecture"]:
        raise SystemExit("REFUSED: architecture treatment has no declared delta")


def _receipt(arm: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    payload = {
        "schema_version": "a1-mixed-relational-architecture-receipt-v1",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "arm": arm["arm"],
        "architecture": arm["architecture"],
        "training_recipe": arm["training_recipe"],
        "experiment_manifest": common._file_ref(manifest_path),
        "descriptor": common._file_ref(Path(arm["descriptor"])),
        "architecture_audit": arm["architecture_audit"],
        "checkpoint": common._file_ref(Path(arm["checkpoint"])),
        "report": common._file_ref(Path(arm["report"])),
    }
    payload["receipt_sha256"] = common._canonical_sha(payload)
    return payload


def _launch(manifest: dict[str, Any], manifest_path: Path) -> None:
    for arm_name in ARMS:
        arm = manifest["arms"][arm_name]
        receipt_path = Path(arm["receipt"])
        if receipt_path.exists():
            raise SystemExit(f"REFUSED: completed arm already exists: {arm_name}")
        for output in (arm["checkpoint"], arm["report"]):
            if Path(output).exists():
                raise SystemExit(f"REFUSED: partial architecture output exists: {output}")
        arm_dir = receipt_path.parent
        with (arm_dir / "stdout.log").open("x", encoding="utf-8") as stdout, (
            arm_dir / "stderr.log"
        ).open("x", encoding="utf-8") as stderr:
            subprocess.run(arm["command"], check=True, stdout=stdout, stderr=stderr)
        if not Path(arm["checkpoint"]).is_file() or not Path(arm["report"]).is_file():
            raise SystemExit(f"REFUSED: {arm_name} produced no checkpoint/report")
        common._write_once_or_match(receipt_path, _receipt(arm, manifest_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr", required=True, choices=tuple(common.ALLOWED_LRS))
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--n256-corpus", required=True, type=Path)
    parser.add_argument("--n256-validation", required=True, type=Path)
    parser.add_argument("--n128-corpus", required=True, type=Path)
    parser.add_argument("--n128-validation", required=True, type=Path)
    parser.add_argument("--initialization-checkpoint", required=True, type=Path)
    parser.add_argument("--architecture-audit", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--go", action="store_true")
    return parser


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    if args.max_steps <= 0:
        raise SystemExit("REFUSED: --max-steps must be positive and identical across arms")
    repo = args.repo.expanduser().resolve(strict=True)
    python = args.python.expanduser().resolve(strict=True)
    initialization = args.initialization_checkpoint.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    components = [
        common._component(args.n256_corpus, args.n256_validation),
        common._component(args.n128_corpus, args.n128_validation),
    ]
    _audit, audit_ref = _validate_audit(args.architecture_audit, components)
    recipe = _training_recipe(common.ALLOWED_LRS[args.lr], args.max_steps)
    descriptor = common._descriptor(components, _descriptor_recipe(recipe))
    descriptor_path = output_root / "memmap_composite.json"
    common._write_once_or_match(descriptor_path, descriptor)
    initialization_ref = common._file_ref(initialization)
    arms: dict[str, dict[str, Any]] = {}
    for arm_name in ARMS:
        arm_dir = output_root / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)
        architecture = _architecture(arm_name)
        command = _command(
            python=python,
            repo=repo,
            descriptor=descriptor_path,
            initialization=initialization,
            arm_dir=arm_dir,
            recipe=recipe,
            architecture=architecture,
        )
        arms[arm_name] = {
            "arm": arm_name,
            "architecture": architecture,
            "architecture_sha256": common._canonical_sha(architecture),
            "training_recipe": recipe,
            "training_recipe_sha256": common._canonical_sha(recipe),
            "initialization": initialization_ref,
            "architecture_audit": audit_ref,
            "descriptor": str(descriptor_path),
            "checkpoint": str(arm_dir / "checkpoint.pt"),
            "report": str(arm_dir / "report.json"),
            "receipt": str(arm_dir / "training.receipt.json"),
            "command": command,
            "command_sha256": common._canonical_sha(command),
        }
    _assert_matched(arms)
    manifest = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "event_path": {
            "included": False,
            "reason": "audit proves zero masked events and zero event targets in both corpora",
        },
        "topology": {
            "world_size": 8,
            "local_batch_size": 512,
            "global_batch_size": 4096,
            "global_row_shuffle": True,
            "no_copy": True,
        },
        "components": components,
        "descriptor": common._file_ref(descriptor_path),
        "architecture_audit": audit_ref,
        "initialization": initialization_ref,
        "source_binding": _source_binding(repo),
        "matched_fields": [
            "initialization",
            "data",
            "validation_split",
            "optimizer",
            "max_steps",
            "seed",
        ],
        "only_declared_arm_delta": "architecture",
        "arms": arms,
    }
    manifest["manifest_sha256"] = common._canonical_sha(manifest)
    manifest_path = output_root / "experiment.manifest.json"
    common._write_once_or_match(manifest_path, manifest)
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
                    "arms": list(ARMS),
                    "max_steps": args.max_steps,
                    "launched": False,
                    "diagnostic_only": True,
                    "promotion_eligible": False,
                    "event_path_included": False,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
