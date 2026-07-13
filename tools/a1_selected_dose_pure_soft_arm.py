#!/usr/bin/env python3
"""Prepare, verify, or explicitly run the selected-dose pure-search-target arm.

The historical pure-soft launcher is inseparable from the rejected 4.19M-row
dose.  This replacement derives from the authenticated executed 524,288-row
TEMP geometry.  It changes exactly one causal learner field:
``soft_target_weight 0.9 -> 1.0``.  Thus the selected behavior action no longer
contributes a 10% hard CE term, while f7, data/component temperatures, sampler,
value objective, optimizer, LR trajectory, batch geometry, and validation stay
byte/semantics matched to the completed TEMP control.

Preparation never launches.  Execution requires this same file through
``execute --go``, an idle eight-B200 topology, and a fresh one-shot output.
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


SCHEMA = "a1-selected-dose-pure-soft-arm-v1"
RECEIPT_SCHEMA = "a1-selected-dose-pure-soft-execution-receipt-v1"
STATUS_SCHEMA = "a1-selected-dose-pure-soft-execution-status-v1"
CLAIM_SCHEMA = "a1-selected-dose-pure-soft-execution-claim-v1"
EXECUTOR_RELATIVE_PATH = "tools/a1_selected_dose_pure_soft_arm.py"
SOURCE_WEIGHT = "0.9"
TREATMENT_WEIGHT = "1.0"
SOURCE_FILES = (
    EXECUTOR_RELATIVE_PATH,
    "tools/a1_topology_gather_arm.py",
    "tools/a1_corrected_policy_arm.py",
    "tools/a1_corrected_policy_arm_execute.py",
    "tools/a1_production_temperature_replication.py",
    "tools/a1_production_l1_rerun.py",
    "tools/train_bc.py",
    "tools/mixed_memmap_corpus.py",
    "src/catan_zero/rl/entity_token_policy.py",
    "src/catan_zero/rl/entity_token_features.py",
) + (
    ("tools/a1_learner_dose_contract.py",)
    if hasattr(bridge.production_temp, "LEGACY_MANIFEST_SCHEMA")
    else ()
)


class PureSoftError(RuntimeError):
    """The request is not the exact selected-dose pure-soft arm."""


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
        raise PureSoftError(
            "selected-dose pure-soft sources must be clean tracked bytes"
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


def _derive_command(
    source: Sequence[str], *, output_root: Path
) -> tuple[list[str], dict[str, Any]]:
    command = list(source)
    if bridge.corrected._option(command, "--soft-target-weight") != SOURCE_WEIGHT:  # noqa: SLF001
        raise PureSoftError("source is not the selected 0.9-soft TEMP control")
    if not (
        bridge.corrected._option(command, "--max-steps")  # noqa: SLF001
        == str(bridge.SELECTED_OPTIMIZER_STEPS)
        and bridge.corrected._option(command, "--batch-size") == "512"  # noqa: SLF001
        and bridge.corrected._option(command, "--lr") == "3e-05"  # noqa: SLF001
        and bridge.corrected._option(command, "--lr-warmup-steps") == "100"  # noqa: SLF001
    ):
        raise PureSoftError("source is not the selected 524,288-row LR trajectory")
    changes = {
        "--soft-target-weight": {"source": SOURCE_WEIGHT, "treatment": TREATMENT_WEIGHT},
        "--checkpoint": {
            "source": bridge.corrected._option(command, "--checkpoint"),  # noqa: SLF001
            "treatment": str(output_root / "candidate.pt"),
        },
        "--report": {
            "source": bridge.corrected._option(command, "--report"),  # noqa: SLF001
            "treatment": str(output_root / "train.report.json"),
        },
    }
    for flag, row in changes.items():
        bridge.corrected._set_option(command, flag, row["treatment"])  # noqa: SLF001
    return command, changes


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    source, source_ref = bridge._load_source(  # noqa: SLF001
        args.source_manifest,
        args.selected_dose_plan,
        args.selected_dose_report,
    )
    output_root = args.output_root.expanduser().resolve()
    forbidden = (
        output_root / "candidate.pt",
        Path(str(output_root / "candidate.pt") + ".optimizer.pt"),
        Path(str(output_root / "candidate.pt") + ".training-progress.json"),
        output_root / "train.report.json",
        output_root / "diagnostic-execution.claim.json",
        output_root / "diagnostic-execution.receipt.json",
    )
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise PureSoftError(f"selected-dose pure-soft output already exists: {existing}")
    binding = _source_binding(args.repo)
    executor_ref = binding["files"].get(EXECUTOR_RELATIVE_PATH)
    if not isinstance(executor_ref, dict):
        raise PureSoftError("source binding does not authenticate this executor")
    command, changes = _derive_command(source["command"], output_root=output_root)
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
            "soft_target_weight": {"source": 0.9, "treatment": 1.0},
            "played_action_hard_ce_weight": {"source": 0.1, "treatment": 0.0},
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
            "value_objective_unchanged": True,
            "lr_trajectory_unchanged": True,
            "sampler_unchanged": True,
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
    path = output_root / "selected-dose-pure-soft.manifest.json"
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise PureSoftError(f"prepared manifest drift: {path}")
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
        raise PureSoftError("selected-dose pure-soft manifest digest drift")
    if not (
        payload.get("schema_version") == SCHEMA
        and payload.get("diagnostic_only") is True
        and payload.get("promotion_eligible") is False
        and payload.get("launch_authorized") is False
        and payload.get("diagnostic_execution_authorized") is True
        and payload.get("launch_interface_present")
        == f"{EXECUTOR_RELATIVE_PATH} execute --go"
    ):
        raise PureSoftError("selected-dose pure-soft authorization drift")
    executor = _verify_ref(payload.get("diagnostic_executor"), label="diagnostic_executor")
    if executor != Path(__file__).resolve():
        raise PureSoftError("manifest authorizes a different executor")

    binding = payload.get("source_binding")
    if not isinstance(binding, Mapping):
        raise PureSoftError("manifest lacks source checkout binding")
    try:
        preparer_repo = Path(str(binding.get("repository_root", ""))).resolve(strict=True)
    except OSError as error:
        raise PureSoftError(f"cannot resolve preparer checkout: {error}") from error
    if executor_base._git_head(preparer_repo) != binding.get("git_commit"):  # noqa: SLF001
        raise PureSoftError("preparer checkout commit drift")
    files = binding.get("files")
    if not isinstance(files, Mapping) or set(files) != set(SOURCE_FILES):
        raise PureSoftError("selected-dose pure-soft source binding is incomplete")
    if binding.get("files_sha256") != bridge.corrected._digest(files):  # noqa: SLF001
        raise PureSoftError("selected-dose pure-soft file-set digest drift")
    for relative, ref in files.items():
        path = _verify_ref(ref, label=f"source.{relative}")
        if path != (preparer_repo / relative).resolve(strict=True):
            raise PureSoftError(f"bound source escaped checkout: {relative}")
    if files[EXECUTOR_RELATIVE_PATH] != payload["diagnostic_executor"]:
        raise PureSoftError("executor identity differs from source binding")

    source_manifest = _verify_ref(
        payload.get("source_temperature_manifest"), label="source_temperature_manifest"
    )
    evidence = payload.get("selected_geometry_evidence")
    if not isinstance(evidence, Mapping):
        raise PureSoftError("manifest lacks selected geometry evidence")
    plan = _verify_ref(evidence.get("plan"), label="selected_geometry.plan")
    report = _verify_ref(evidence.get("report"), label="selected_geometry.report")
    try:
        source, _ = bridge._load_source(source_manifest, plan, report)  # noqa: SLF001
    except bridge.ArmError as error:
        raise PureSoftError(f"selected-dose provenance bridge failed: {error}") from error
    if not (
        source["selected_geometry_evidence"] == evidence
        and payload.get("source_temperature_manifest_sha256")
        == source["manifest_sha256"]
        and payload.get("source_recipe") == source["recipe"]
        and payload.get("source_recipe_sha256") == source["recipe_sha256"]
        and payload.get("initialization") == source["initialization"]
        and payload.get("descriptor") == source["descriptor"]
        and payload.get("validation_sentinel") == source["validation_sentinel"]
    ):
        raise PureSoftError("selected-dose source identity drift")
    if payload.get("only_declared_causal_delta") != {
        "soft_target_weight": {"source": 0.9, "treatment": 1.0},
        "played_action_hard_ce_weight": {"source": 0.1, "treatment": 0.0},
    }:
        raise PureSoftError("pure-soft causal-delta declaration drift")
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
        "value_objective_unchanged": True,
        "lr_trajectory_unchanged": True,
        "sampler_unchanged": True,
    }
    if payload.get("matched_contract") != expected_matched:
        raise PureSoftError("pure-soft matched contract drift")
    root = Path(str(payload.get("output_root", ""))).resolve()
    expected_command, changes = _derive_command(source["command"], output_root=root)
    command = payload.get("command")
    if not (
        command == expected_command
        and payload.get("allowlisted_command_changes") == changes
        and payload.get("command_sha256")
        == bridge.corrected._digest(command)  # noqa: SLF001
    ):
        raise PureSoftError("pure-soft command is not the exact selected-dose derivation")
    trainers = [Path(value).resolve() for value in command if Path(value).name == "train_bc.py"]
    selected_trainer = Path(source["selected_geometry_trainer"]).resolve(strict=True)
    if trainers != [selected_trainer]:
        raise PureSoftError("pure-soft command escaped selected-geometry trainer")
    trainer_repo = Path(source["selected_geometry_runtime_repo"]).resolve(strict=True)
    forbidden = (
        root / "candidate.pt",
        Path(str(root / "candidate.pt") + ".optimizer.pt"),
        Path(str(root / "candidate.pt") + ".training-progress.json"),
        root / "train.report.json",
        root / "diagnostic-execution.claim.json",
        root / "diagnostic-execution.receipt.json",
        root / "diagnostic-execution.status.jsonl",
        root / "stdout.log",
        root / "stderr.log",
    )
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise PureSoftError(f"selected-dose pure-soft output/claim already exists: {existing}")
    return {
        "manifest": payload,
        "manifest_ref": manifest_ref,
        "repo": trainer_repo,
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
        raise PureSoftError(str(error)) from error


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
    run.add_argument("--unit", default="a1-selected-dose-pure-soft")
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
            result = {
                "verified": True,
                "launched": False,
                "manifest": verified["manifest_ref"],
            }
        else:
            receipt = execute(args.manifest, unit=args.unit)
            result = {
                "submitted": True,
                "unit": receipt["unit"],
                "receipt_sha256": receipt["receipt_sha256"],
            }
    except (PureSoftError, bridge.ArmError) as error:
        raise SystemExit(f"REFUSED: {error}") from error
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
