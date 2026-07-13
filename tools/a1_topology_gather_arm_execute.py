#!/usr/bin/env python3
"""Verify or explicitly submit one immutable topology-gather diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_corrected_policy_arm_execute as base  # noqa: E402
from tools import a1_topology_gather_arm as prepare  # noqa: E402


RECEIPT_SCHEMA = "a1-topology-gather-arm-execution-receipt-v1"
STATUS_SCHEMA = "a1-topology-gather-arm-execution-status-v1"
CLAIM_SCHEMA = "a1-topology-gather-arm-execution-claim-v1"
ExecutionError = base.ExecutionError


def _read_manifest(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    payload, ref = prepare.corrected._load_json(path)  # noqa: SLF001
    stated = payload.get("manifest_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if stated != prepare.corrected._digest(unhashed):  # noqa: SLF001
        raise ExecutionError("topology manifest semantic digest drift")
    if not (
        payload.get("schema_version") == prepare.SCHEMA
        and payload.get("diagnostic_only") is True
        and payload.get("promotion_eligible") is False
        and payload.get("launch_authorized") is False
        and payload.get("diagnostic_execution_authorized") is True
        and payload.get("launch_interface_present")
        == f"{prepare.EXECUTOR_RELATIVE_PATH} --go"
    ):
        raise ExecutionError("manifest does not authorize the topology executor")
    return payload, ref


def _verify_ref(value: Any, *, label: str) -> Path:
    return base._verify_ref(value, label=label)  # noqa: SLF001


def _source_trainer_binding(
    source_manifest: Path,
) -> tuple[dict[str, Any], Path, Path]:
    """Authenticate the trainer/runtime checkout selected by the TEMP control."""

    try:
        source = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutionError(f"cannot decode source selected-TEMP manifest: {error}") from error
    if not isinstance(source, dict):
        raise ExecutionError("source selected-TEMP manifest is not an object")
    stated = source.get("manifest_sha256")
    unhashed = {key: value for key, value in source.items() if key != "manifest_sha256"}
    source_command = source.get("command")
    if not (
        source.get("schema_version") == prepare.SOURCE_SCHEMA
        and stated == prepare.corrected._digest(unhashed)  # noqa: SLF001
        and isinstance(source_command, list)
        and all(isinstance(value, str) for value in source_command)
        and source.get("command_sha256")
        == prepare.corrected._digest(source_command)  # noqa: SLF001
    ):
        raise ExecutionError("source selected-TEMP manifest/command digest drift")
    binding = source.get("source_binding")
    if not isinstance(binding, dict):
        raise ExecutionError("source selected-TEMP manifest has no checkout binding")
    try:
        source_repo = Path(str(binding.get("repository_root", ""))).resolve(strict=True)
    except OSError as error:
        raise ExecutionError(f"cannot resolve source trainer checkout: {error}") from error
    if base._git_head(source_repo) != binding.get("git_commit"):  # noqa: SLF001
        raise ExecutionError("source trainer checkout commit differs from selected TEMP")
    files = binding.get("files")
    if not isinstance(files, dict) or not files:
        raise ExecutionError("source selected-TEMP checkout binding has no files")
    bound_paths: dict[Path, str] = {}
    for relative, ref in files.items():
        path = _verify_ref(ref, label=f"corrected_source.{relative}")
        try:
            expected = (source_repo / relative).resolve(strict=True)
        except OSError as error:
            raise ExecutionError(
                f"cannot resolve corrected source path {relative}: {error}"
            ) from error
        if path != expected:
            raise ExecutionError(f"corrected source path escaped checkout: {relative}")
        bound_paths[path] = relative
    trainers = [
        Path(value).resolve()
        for value in source_command
        if Path(value).name == "train_bc.py"
    ]
    if len(trainers) != 1 or trainers[0] not in bound_paths:
        raise ExecutionError(
            "source selected-TEMP trainer is not authenticated by its checkout binding"
        )
    return source, source_repo, trainers[0]


def verify(manifest_path: Path) -> dict[str, Any]:
    manifest, manifest_ref = _read_manifest(manifest_path)
    executor = _verify_ref(manifest.get("diagnostic_executor"), label="diagnostic_executor")
    if executor != Path(__file__).resolve():
        raise ExecutionError("manifest authorizes a different executor path")

    source_manifest = _verify_ref(
        manifest.get("source_selected_temp_manifest"), label="source_selected_temp_manifest"
    )
    source, trainer_repo, source_trainer = _source_trainer_binding(source_manifest)
    descriptor = _verify_ref(manifest.get("descriptor"), label="descriptor")
    sentinel = _verify_ref(
        manifest.get("validation_sentinel"), label="validation_sentinel"
    )
    source_init = _verify_ref(
        manifest.get("initialization_source"), label="initialization_source"
    )
    treatment_init = _verify_ref(
        manifest.get("initialization_treatment"), label="initialization_treatment"
    )
    coverage = manifest.get("corpus_topology_target_coverage")
    if not isinstance(coverage, dict):
        raise ExecutionError("manifest has no topology coverage binding")
    _verify_ref(coverage.get("artifact"), label="architecture_audit")

    upgrade = manifest.get("function_preserving_upgrade")
    if not isinstance(upgrade, dict) or not (
        upgrade.get("source") == manifest.get("initialization_source")
        and upgrade.get("upgraded") == manifest.get("initialization_treatment")
        and upgrade.get("flags") == {"action_target_gather": True}
        and upgrade.get("forward_max_diff") == 0.0
        and upgrade.get("forward_identical_at_init") is True
        and upgrade.get("shared_parameters_bit_identical") is True
        and upgrade.get("new_parameters") == list(prepare.EXPECTED_NEW_PARAMETERS)
    ):
        raise ExecutionError("function-preserving gather contract drift")

    source_binding = manifest.get("source_binding")
    if not isinstance(source_binding, dict):
        raise ExecutionError("manifest has no source checkout binding")
    try:
        preparer_repo = Path(str(source_binding.get("repository_root", ""))).resolve(
            strict=True
        )
    except OSError as error:
        raise ExecutionError(f"cannot resolve execution checkout: {error}") from error
    if base._git_head(preparer_repo) != source_binding.get("git_commit"):  # noqa: SLF001
        raise ExecutionError("execution checkout commit differs from topology manifest")
    files = source_binding.get("files")
    if not isinstance(files, dict) or prepare.EXECUTOR_RELATIVE_PATH not in files:
        raise ExecutionError("source checkout does not bind the topology executor")
    for relative, ref in files.items():
        path = _verify_ref(ref, label=f"source.{relative}")
        try:
            expected = (preparer_repo / relative).resolve(strict=True)
        except OSError as error:
            raise ExecutionError(f"cannot resolve source path {relative}: {error}") from error
        if path != expected:
            raise ExecutionError(f"source path escaped checkout: {relative}")
    if files[prepare.EXECUTOR_RELATIVE_PATH] != manifest["diagnostic_executor"]:
        raise ExecutionError("executor identity differs from source checkout binding")

    command = manifest.get("command")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ExecutionError("manifest command is malformed")
    if manifest.get("command_sha256") != prepare.corrected._digest(command):  # noqa: SLF001
        raise ExecutionError("manifest command digest drift")
    trainer = [Path(value).resolve() for value in command if Path(value).name == "train_bc.py"]
    if trainer != [source_trainer]:
        raise ExecutionError(
            "topology command trainer differs from bound selected-TEMP trainer"
        )
    exact_inputs = {
        "--data": str(descriptor),
        "--validation-game-sentinel-manifest": str(sentinel),
        "--init-checkpoint": str(treatment_init),
    }
    for flag, expected in exact_inputs.items():
        if base._option(command, flag) != expected:  # noqa: SLF001
            raise ExecutionError(f"command differs from bound {flag}")
    contract = manifest.get("event_history_training_contract")
    descriptor_meta, _ = prepare.corrected._preflight_descriptor(  # noqa: SLF001
        descriptor
    )
    expected_contract = prepare.corrected._event_history_training_contract(  # noqa: SLF001
        descriptor_meta
    )
    if contract != expected_contract:
        raise ExecutionError("topology manifest event-history contract drift")
    expected_acks = [
        row["payload_inventory_sha256"]
        for row in expected_contract["empty_payload_inventory_acknowledgements"]
    ]
    positions = [
        index
        for index, value in enumerate(command)
        if value == prepare.corrected.EVENT_HISTORY_ACK_FLAG
    ]
    observed_acks = [
        command[index + 1]
        for index in positions
        if index + 1 < len(command) and not command[index + 1].startswith("--")
    ]
    if observed_acks != expected_acks or len(positions) != len(expected_acks):
        raise ExecutionError("topology command lacks exact event-history ACK set")
    if command.count(prepare.corrected.EVENT_HISTORY_CROP_FLAG) != 1:
        raise ExecutionError("topology command lacks authenticated event crop flag")
    if "--validation-game-seed-manifest" in command:
        raise ExecutionError("command contains a second validation control")
    if not (
        manifest.get("only_declared_optimization_delta") == "action_target_gather=true"
        and manifest.get("source_recipe_sha256")
        == prepare.corrected._digest(manifest.get("source_recipe"))  # noqa: SLF001
        and manifest.get("source_selected_temp_manifest_sha256")
        == source.get("manifest_sha256")
        and source_init == Path(upgrade["source"]["path"])
    ):
        raise ExecutionError("topology source/recipe identity drift")

    output_root = Path(base._option(command, "--checkpoint")).parent.resolve()  # noqa: SLF001
    checkpoint = output_root / "candidate.pt"
    report = output_root / "train.report.json"
    if Path(base._option(command, "--checkpoint")) != checkpoint or Path(  # noqa: SLF001
        base._option(command, "--report")  # noqa: SLF001
    ) != report:
        raise ExecutionError("command outputs are not canonical topology-arm paths")
    forbidden = (
        checkpoint,
        Path(str(checkpoint) + ".optimizer.pt"),
        Path(str(checkpoint) + ".training-progress.json"),
        report,
        output_root / "diagnostic-execution.claim.json",
        output_root / "diagnostic-execution.receipt.json",
        output_root / "diagnostic-execution.status.jsonl",
        output_root / "stdout.log",
        output_root / "stderr.log",
    )
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise ExecutionError(f"topology-arm output/claim already exists: {existing}")
    return {
        "manifest": manifest,
        "manifest_ref": manifest_ref,
        # Run from the selected-TEMP checkout as well as invoking its exact
        # trainer bytes. Relative imports/config paths therefore retain the
        # source arm's runtime semantics; the current checkout only supplies
        # the independently authenticated preparer/executor.
        "repo": trainer_repo,
        "preparer_repo": preparer_repo,
        "command": command,
        "output_root": output_root,
    }


def execute(
    manifest_path: Path,
    *,
    unit: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    conflict_probe: Callable[[], list[str]] = base._probe_conflicting_compute,  # noqa: SLF001
) -> dict[str, Any]:
    verified = verify(manifest_path)
    return base._submit_verified(  # noqa: SLF001
        verified,
        unit=unit,
        runner=runner,
        conflict_probe=conflict_probe,
        claim_schema=CLAIM_SCHEMA,
        receipt_schema=RECEIPT_SCHEMA,
        status_schema=STATUS_SCHEMA,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--unit", default="a1-selected-temp-gather")
    parser.add_argument("--go", action="store_true")
    args = parser.parse_args(argv)
    if not args.go:
        verified = verify(args.manifest)
        print(
            json.dumps(
                {"verified": True, "launched": False, "manifest": verified["manifest_ref"]},
                sort_keys=True,
            )
        )
        return
    receipt = execute(args.manifest, unit=args.unit)
    print(
        json.dumps(
            {
                "submitted": True,
                "unit": receipt["unit"],
                "receipt_sha256": receipt["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
