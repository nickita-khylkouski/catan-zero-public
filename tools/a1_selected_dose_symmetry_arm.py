#!/usr/bin/env python3
"""Seal and execute the selected-dose D6 training-augmentation arm.

The arm replays the authenticated TEMP midpoint geometry from exact f7:
eight ranks, local batch 512, 128 optimizer steps, and 524,288 total draws.
It changes the learner operator only by enabling training-time D6 augmentation
(including event target relabeling).  Execution uses the current trainer whose
rank-distinct, topology-bound symmetry RNG and all-rank resume state are sealed;
the historical selected-geometry trainer is source evidence only because it
reused one symmetry stream on every rank.  Preparation is non-mutating;
execution is one-shot and diagnostic-only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_corrected_policy_arm_execute as executor_base  # noqa: E402
from tools import a1_topology_gather_arm as bridge  # noqa: E402


SCHEMA = "a1-selected-dose-symmetry-arm-v1"
RECEIPT_SCHEMA = "a1-selected-dose-symmetry-execution-receipt-v1"
STATUS_SCHEMA = "a1-selected-dose-symmetry-execution-status-v1"
CLAIM_SCHEMA = "a1-selected-dose-symmetry-execution-claim-v1"
EXECUTOR_RELATIVE_PATH = "tools/a1_selected_dose_symmetry_arm.py"
SOURCE_FILES = (
    EXECUTOR_RELATIVE_PATH,
    "tools/a1_topology_gather_arm.py",
    "tools/a1_corrected_policy_arm.py",
    "tools/a1_corrected_policy_arm_execute.py",
    "tools/a1_production_temperature_replication.py",
    "tools/a1_production_l1_rerun.py",
    "tools/train_bc.py",
    "tools/mixed_memmap_corpus.py",
    "src/catan_zero/rl/hex_symmetry.py",
    "src/catan_zero/rl/entity_token_policy.py",
    "src/catan_zero/rl/entity_token_features.py",
) + (
    ("tools/a1_learner_dose_contract.py",)
    if hasattr(bridge.production_temp, "LEGACY_MANIFEST_SCHEMA")
    else ()
)


class SymmetryArmError(RuntimeError):
    """The request is not the exact selected-dose D6 arm."""


def _source_binding(repo: Path) -> dict[str, Any]:
    repo = repo.expanduser().resolve(strict=True)
    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=repo, text=True
        ).strip()
        subprocess.run(
            ("git", "diff", "--quiet", "HEAD", "--", *SOURCE_FILES),
            cwd=repo,
            check=True,
        )
        for relative in SOURCE_FILES:
            subprocess.run(
                ("git", "ls-files", "--error-unmatch", relative),
                cwd=repo,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SymmetryArmError(
            "selected-dose symmetry sources must be clean tracked bytes"
        ) from error
    files = {
        relative: bridge.corrected._file_ref(repo / relative)  # noqa: SLF001
        for relative in SOURCE_FILES
    }
    return {
        "repository_root": str(repo),
        "git_commit": commit,
        "files": files,
        "files_sha256": bridge.corrected._digest(files),  # noqa: SLF001
    }


def _set_boolean(command: list[str], flag: str, enabled: bool) -> dict[str, Any]:
    positive = flag
    negative = "--no-" + flag.removeprefix("--")
    positions = [index for index, value in enumerate(command) if value in {positive, negative}]
    if len(positions) > 1:
        raise SymmetryArmError(f"source command repeats boolean option {flag}")
    source = "default"
    if positions:
        index = positions[0]
        source = command.pop(index)
    treatment = positive if enabled else negative
    command.append(treatment)
    return {"source": source, "treatment": treatment}


def _derive_command(
    source: Sequence[str], *, trainer: Path, output_root: Path
) -> tuple[list[str], dict[str, Any]]:
    command = list(source)
    expected = {
        "--max-steps": str(bridge.SELECTED_OPTIMIZER_STEPS),
        "--batch-size": "512",
        "--grad-accum-steps": "1",
        "--lr": "3e-05",
        "--lr-warmup-steps": "100",
        "--soft-target-weight": "0.9",
        "--value-loss-weight": "0.25",
    }
    observed = {
        flag: bridge.corrected._option(command, flag)  # noqa: SLF001
        for flag in expected
    }
    if observed != expected:
        raise SymmetryArmError(
            f"source is not the selected 524,288-row TEMP geometry: {observed}"
        )
    trainer_positions = [
        index for index, value in enumerate(command) if Path(value).name == "train_bc.py"
    ]
    if len(trainer_positions) != 1:
        raise SymmetryArmError("source must name exactly one historical trainer")
    trainer_index = trainer_positions[0]
    changes: dict[str, Any] = {
        "trainer": {
            "source": command[trainer_index],
            "treatment": str(trainer.resolve(strict=True)),
            "reason": "sealed per-rank symmetry RNG/checkpoint-resume contract",
        }
    }
    command[trainer_index] = str(trainer.resolve(strict=True))
    for flag, value in (
        ("--checkpoint", str(output_root / "candidate.pt")),
        ("--report", str(output_root / "train.report.json")),
    ):
        source_value = bridge.corrected._option(command, flag)  # noqa: SLF001
        bridge.corrected._set_option(command, flag, value)  # noqa: SLF001
        changes[flag] = {"source": source_value, "treatment": value}
    changes["--symmetry-augment"] = _set_boolean(
        command, "--symmetry-augment", True
    )
    changes["--symmetry-augment-events"] = _set_boolean(
        command, "--symmetry-augment-events", True
    )
    return command, changes


def _forbidden_outputs(root: Path) -> tuple[Path, ...]:
    checkpoint = root / "candidate.pt"
    return (
        checkpoint,
        Path(str(checkpoint) + ".optimizer.pt"),
        Path(str(checkpoint) + ".training-progress.json"),
        root / "train.report.json",
        root / "diagnostic-execution.claim.json",
        root / "diagnostic-execution.receipt.json",
        root / "diagnostic-execution.status.jsonl",
        root / "stdout.log",
        root / "stderr.log",
    )


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    source, source_ref = bridge._load_source(  # noqa: SLF001
        args.source_manifest,
        args.selected_dose_plan,
        args.selected_dose_report,
    )
    output_root = args.output_root.expanduser().resolve()
    existing = [str(path) for path in _forbidden_outputs(output_root) if path.exists()]
    if existing:
        raise SymmetryArmError(f"selected-dose symmetry output exists: {existing}")
    binding = _source_binding(args.repo)
    executor_ref = binding["files"].get(EXECUTOR_RELATIVE_PATH)
    if not isinstance(executor_ref, dict):
        raise SymmetryArmError("source binding does not authenticate this executor")
    trainer_ref = binding["files"].get("tools/train_bc.py")
    if not isinstance(trainer_ref, Mapping):
        raise SymmetryArmError("source binding does not authenticate current trainer")
    command, changes = _derive_command(
        source["command"],
        trainer=Path(str(trainer_ref["path"])),
        output_root=output_root,
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "launch_authorized": False,
        "diagnostic_execution_authorized": True,
        "launch_interface_present": f"{EXECUTOR_RELATIVE_PATH} execute --go",
        "diagnostic_executor": executor_ref,
        "source_temperature_manifest": source_ref,
        "source_temperature_manifest_sha256": source["manifest_sha256"],
        "selected_geometry_evidence": source["selected_geometry_evidence"],
        "source_recipe": source["recipe"],
        "source_recipe_sha256": source["recipe_sha256"],
        "initialization": source["initialization"],
        "descriptor": source["descriptor"],
        "validation_sentinel": source["validation_sentinel"],
        "source_binding": binding,
        "only_declared_causal_delta": {
            "symmetry_augment": {"source": False, "treatment": True},
            "symmetry_augment_events": {
                "source": False,
                "treatment": True,
                "conditional_on_symmetry": True,
            },
        },
        "runtime_contract_delta": {
            "historical_trainer": changes["trainer"]["source"],
            "current_trainer": trainer_ref,
            "reason": "rank-distinct D6 RNG and all-rank exact resume",
            "distributed_symmetry_contract": (
                "per_rank_seedsequence_checkpoint_resume_v1"
            ),
        },
        "matched_contract": {
            "global_row_dose": bridge.SELECTED_GLOBAL_ROW_DOSE,
            "optimizer_steps": bridge.SELECTED_OPTIMIZER_STEPS,
            "world_size": 8,
            "local_batch_size": 512,
            "global_batch_size": 4096,
            "fresh_f7_initialization": True,
            "fresh_adam": True,
            "candidate_chaining": False,
            "data_component_temperatures_unchanged": True,
            "policy_and_value_objective_unchanged": True,
            "lr_trajectory_unchanged": True,
            "sampler_row_order_unchanged": True,
            "learner_science_recipe_unchanged_except_symmetry": True,
            "distributed_symmetry_contract": (
                "per_rank_seedsequence_checkpoint_resume_v1"
            ),
        },
        "allowlisted_command_changes": changes,
        "command": command,
        "command_sha256": bridge.corrected._digest(command),  # noqa: SLF001
        "output_root": str(output_root),
        "executor_compatibility": {
            "receipt_schema": RECEIPT_SCHEMA,
            "idle_topology": "exactly_8_visible_B200s",
            "one_shot": True,
        },
    }
    manifest["manifest_sha256"] = bridge.corrected._digest(manifest)  # noqa: SLF001
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "selected-dose-symmetry.manifest.json"
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise SymmetryArmError(f"prepared manifest drift: {path}")
    else:
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text(encoded, encoding="utf-8")
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    return manifest, path


def _verify_ref(value: Any, *, label: str) -> Path:
    return executor_base._verify_ref(value, label=label)  # noqa: SLF001


def verify(manifest_path: Path) -> dict[str, Any]:
    payload, manifest_ref = bridge.corrected._load_json(manifest_path)  # noqa: SLF001
    stated = payload.get("manifest_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if stated != bridge.corrected._digest(unhashed):  # noqa: SLF001
        raise SymmetryArmError("selected-dose symmetry manifest digest drift")
    if not (
        payload.get("schema_version") == SCHEMA
        and payload.get("diagnostic_only") is True
        and payload.get("promotion_eligible") is False
        and payload.get("launch_authorized") is False
        and payload.get("diagnostic_execution_authorized") is True
        and payload.get("launch_interface_present")
        == f"{EXECUTOR_RELATIVE_PATH} execute --go"
    ):
        raise SymmetryArmError("selected-dose symmetry authorization drift")
    executor = _verify_ref(payload.get("diagnostic_executor"), label="executor")
    if executor != Path(__file__).resolve():
        raise SymmetryArmError("manifest authorizes a different executor")
    binding = payload.get("source_binding")
    if not isinstance(binding, Mapping):
        raise SymmetryArmError("manifest lacks source binding")
    preparer_repo = Path(str(binding.get("repository_root", ""))).resolve(strict=True)
    if executor_base._git_head(preparer_repo) != binding.get("git_commit"):  # noqa: SLF001
        raise SymmetryArmError("preparer checkout commit drift")
    files = binding.get("files")
    if not isinstance(files, Mapping) or set(files) != set(SOURCE_FILES):
        raise SymmetryArmError("symmetry source binding is incomplete")
    if binding.get("files_sha256") != bridge.corrected._digest(files):  # noqa: SLF001
        raise SymmetryArmError("symmetry source file-set digest drift")
    for relative, ref in files.items():
        if _verify_ref(ref, label=f"source.{relative}") != (
            preparer_repo / relative
        ).resolve(strict=True):
            raise SymmetryArmError(f"bound source escaped checkout: {relative}")
    if files[EXECUTOR_RELATIVE_PATH] != payload["diagnostic_executor"]:
        raise SymmetryArmError("executor identity differs from source binding")

    source_manifest = _verify_ref(
        payload.get("source_temperature_manifest"), label="source_manifest"
    )
    evidence = payload.get("selected_geometry_evidence")
    if not isinstance(evidence, Mapping):
        raise SymmetryArmError("manifest lacks selected geometry evidence")
    plan = _verify_ref(evidence.get("plan"), label="selected_geometry.plan")
    report = _verify_ref(evidence.get("report"), label="selected_geometry.report")
    source, _ = bridge._load_source(source_manifest, plan, report)  # noqa: SLF001
    for key in (
        "selected_geometry_evidence",
        "source_recipe",
        "initialization",
        "descriptor",
        "validation_sentinel",
    ):
        source_key = "recipe" if key == "source_recipe" else key
        if payload.get(key) != source[source_key]:
            raise SymmetryArmError(f"selected-dose source identity drift: {key}")
    if not (
        payload.get("source_temperature_manifest_sha256") == source["manifest_sha256"]
        and payload.get("source_recipe_sha256") == source["recipe_sha256"]
        and payload.get("only_declared_causal_delta")
        == {
            "symmetry_augment": {"source": False, "treatment": True},
            "symmetry_augment_events": {
                "source": False,
                "treatment": True,
                "conditional_on_symmetry": True,
            },
        }
    ):
        raise SymmetryArmError("symmetry causal-source declaration drift")
    expected_matched = {
        "global_row_dose": bridge.SELECTED_GLOBAL_ROW_DOSE,
        "optimizer_steps": bridge.SELECTED_OPTIMIZER_STEPS,
        "world_size": 8,
        "local_batch_size": 512,
        "global_batch_size": 4096,
        "fresh_f7_initialization": True,
        "fresh_adam": True,
        "candidate_chaining": False,
        "data_component_temperatures_unchanged": True,
        "policy_and_value_objective_unchanged": True,
        "lr_trajectory_unchanged": True,
        "sampler_row_order_unchanged": True,
        "learner_science_recipe_unchanged_except_symmetry": True,
        "distributed_symmetry_contract": (
            "per_rank_seedsequence_checkpoint_resume_v1"
        ),
    }
    if payload.get("matched_contract") != expected_matched:
        raise SymmetryArmError("symmetry matched contract drift")
    root = Path(str(payload.get("output_root", ""))).resolve()
    current_trainer = _verify_ref(
        files["tools/train_bc.py"], label="source.tools/train_bc.py"
    )
    expected_command, changes = _derive_command(
        source["command"], trainer=current_trainer, output_root=root
    )
    command = payload.get("command")
    if not (
        command == expected_command
        and payload.get("allowlisted_command_changes") == changes
        and payload.get("command_sha256")
        == bridge.corrected._digest(command)  # noqa: SLF001
    ):
        raise SymmetryArmError("command is not exact selected-dose symmetry derivation")
    trainers = [Path(value).resolve() for value in command if Path(value).name == "train_bc.py"]
    if trainers != [current_trainer]:
        raise SymmetryArmError("symmetry command escaped current sealed trainer")
    runtime_delta = payload.get("runtime_contract_delta")
    if runtime_delta != {
        "historical_trainer": changes["trainer"]["source"],
        "current_trainer": files["tools/train_bc.py"],
        "reason": "rank-distinct D6 RNG and all-rank exact resume",
        "distributed_symmetry_contract": (
            "per_rank_seedsequence_checkpoint_resume_v1"
        ),
    }:
        raise SymmetryArmError("symmetry runtime-contract declaration drift")
    if command.count("--symmetry-augment") != 1 or command.count(
        "--symmetry-augment-events"
    ) != 1 or "--no-symmetry-augment" in command or "--no-symmetry-augment-events" in command:
        raise SymmetryArmError("symmetry command boolean drift")
    existing = [str(path) for path in _forbidden_outputs(root) if path.exists()]
    if existing:
        raise SymmetryArmError(f"selected-dose symmetry output exists: {existing}")
    return {
        "manifest": payload,
        "manifest_ref": manifest_ref,
        "repo": preparer_repo,
        "preparer_repo": preparer_repo,
        "command": command,
        "output_root": root,
    }


def execute(
    manifest_path: Path,
    *,
    unit: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    conflict_probe: Callable[[], list[str]] = executor_base._probe_conflicting_compute,  # noqa: SLF001
) -> dict[str, Any]:
    try:
        verified = verify(manifest_path)
        return executor_base._submit_verified(  # noqa: SLF001
            verified,
            unit=unit,
            runner=runner,
            conflict_probe=conflict_probe,
            claim_schema=CLAIM_SCHEMA,
            receipt_schema=RECEIPT_SCHEMA,
            status_schema=STATUS_SCHEMA,
        )
    except executor_base.ExecutionError as error:
        raise SymmetryArmError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-manifest", required=True, type=Path)
    prep.add_argument("--selected-dose-plan", required=True, type=Path)
    prep.add_argument("--selected-dose-report", required=True, type=Path)
    prep.add_argument("--output-root", required=True, type=Path)
    prep.add_argument("--repo", default=REPO_ROOT, type=Path)
    run = sub.add_parser("execute")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--unit", default="a1-selected-dose-symmetry")
    run.add_argument("--go", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "prepare":
            manifest, path = prepare(args)
            result = {
                "prepared": str(path),
                "launched": False,
                "manifest_sha256": manifest["manifest_sha256"],
            }
        elif not args.go:
            verified = verify(args.manifest)
            result = {"verified": True, "launched": False, "manifest": verified["manifest_ref"]}
        else:
            receipt = execute(args.manifest, unit=args.unit)
            result = {
                "submitted": True,
                "unit": receipt["unit"],
                "receipt_sha256": receipt["receipt_sha256"],
            }
    except (SymmetryArmError, bridge.ArmError) as error:
        raise SystemExit(f"REFUSED: {error}") from error
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
