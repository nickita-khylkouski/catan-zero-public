#!/usr/bin/env python3
"""Seal and execute the promotion-eligible four-B200 target-gather retrain.

This is a new typed operator, not an edit of the historical eight-rank L1
schema.  It consumes the promoted production-L1 completion plus a replayable
function-preserving upgrade receipt and freezes every mature model surface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_function_preserving_upgrade as upgrade  # noqa: E402
from tools import a1_production_l1_rerun as base  # noqa: E402


MANIFEST_SCHEMA = "a1-production-target-gather-retrain-v1"
CLAIM_SCHEMA = "a1-production-target-gather-retrain-claim-v1"
SUBMISSION_SCHEMA = "a1-production-target-gather-retrain-submission-v1"
COMPLETION_SCHEMA = "a1-production-target-gather-retrain-completion-v1"
FREEZE_MODULES = "trunk,action_encoder,policy_head,value_heads"
TRAINABLE_PREFIX = "target_gather_proj"
WORLD_SIZE = 4
LOCAL_BATCH = 512
OPTIMIZER_STEPS = 2048
GLOBAL_DRAWS = WORLD_SIZE * LOCAL_BATCH * OPTIMIZER_STEPS
BOUND_SOURCE_FILES = (
    "tools/a1_production_gather_retrain.py",
    "tools/a1_production_l1_rerun.py",
    "tools/a1_function_preserving_upgrade.py",
    "tools/f69_upgrade_checkpoint_config.py",
    "tools/train_bc.py",
    "tools/mixed_memmap_corpus.py",
    "src/catan_zero/rl/entity_token_policy.py",
    "src/catan_zero/rl/entity_token_features.py",
)


class GatherRetrainError(RuntimeError):
    """The target-gather production operator cannot be proven or executed."""


def _set(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, item in enumerate(command) if item == flag]
    equals = [index for index, item in enumerate(command) if item.startswith(flag + "=")]
    if len(positions) + len(equals) > 1:
        raise GatherRetrainError(f"source command repeats {flag}")
    if equals:
        command[equals[0]] = f"{flag}={value}"
    elif positions:
        if positions[0] + 1 >= len(command):
            raise GatherRetrainError(f"source command has no value for {flag}")
        command[positions[0] + 1] = value
    else:
        command.extend((flag, value))


def _source_completion(path: Path) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    ref = base._ref(path)  # noqa: SLF001
    value = base._load(Path(ref["path"]))  # noqa: SLF001
    unhashed = dict(value)
    stated = unhashed.pop("receipt_sha256", None)
    if (
        value.get("schema_version") != base.COMPLETION_SCHEMA
        or value.get("diagnostic_only") is not False
        or value.get("production_eligible") is not True
        or value.get("unit_state")
        != {"ActiveState": "inactive", "Result": "success", "ExecMainStatus": "0"}
        or stated != base._digest(unhashed)  # noqa: SLF001
    ):
        raise GatherRetrainError("source completion is not the promoted clean L1 run")
    for field in ("manifest", "submission", "checkpoint", "report"):
        base._verify_ref(value.get(field), f"source_completion.{field}")  # noqa: SLF001
    source_manifest = base._load(Path(value["manifest"]["path"]))  # noqa: SLF001
    manifest_unhashed = dict(source_manifest)
    manifest_stated = manifest_unhashed.pop("manifest_sha256", None)
    if (
        source_manifest.get("schema_version") != base.MANIFEST_SCHEMA
        or manifest_stated != base._digest(manifest_unhashed)  # noqa: SLF001
        or source_manifest.get("production_eligible") is not True
        or source_manifest.get("diagnostic_only") is not False
        or value["checkpoint"]
        != base._ref(Path(source_manifest["output_root"]) / "candidate.pt")  # noqa: SLF001
    ):
        raise GatherRetrainError("source production manifest/completion binding drifted")
    return value, ref, source_manifest


def _validate_geometry(command: list[str]) -> None:
    exact = {
        "--nproc-per-node": str(WORLD_SIZE),
        "--arch": "entity_graph",
        "--hidden-size": "640",
        "--graph-layers": "6",
        "--attention-heads": "8",
        "--epochs": "1",
        "--max-steps": str(OPTIMIZER_STEPS),
        "--batch-size": str(LOCAL_BATCH),
        "--grad-accum-steps": "1",
        "--optimizer": "adam",
        "--lr": "3e-05",
        "--lr-warmup-steps": "100",
        "--soft-target-weight": "0.9",
        "--value-loss-weight": "0.25",
        "--loser-sample-weight": "1.0",
        "--policy-aux-active-batch-size": "0",
        "--action-module-lr-mult": "4.0",
        "--value-lr-mult": "1.0",
        "--freeze-modules": FREEZE_MODULES,
        "--require-only-trainable-prefixes": TRAINABLE_PREFIX,
    }
    for flag, expected in exact.items():
        if base._option(command, flag) != expected:  # noqa: SLF001
            raise GatherRetrainError(f"target-gather geometry drift at {flag}")
    for required in (
        "--no-resume-optimizer",
        "--no-fused-optimizer",
        "--mask-hidden-info",
        "--graph-history-features",
        "--trust-curated-data-quality",
    ):
        if required not in command:
            raise GatherRetrainError(f"target-gather command lacks {required}")
    forbidden = {
        "--ddp-find-unused-parameters",
        "--fsdp",
        "--train-value-only",
    }
    present = sorted(forbidden & set(command))
    if present:
        raise GatherRetrainError(
            f"target-gather command enables an unreviewed distributed mode: {present}"
        )


def prepare(
    *,
    source_completion: Path,
    architecture_upgrade_receipt: Path,
    repo: Path,
    output_root: Path,
    manifest_path: Path,
    python: Path,
) -> dict[str, Any]:
    repo = repo.expanduser().resolve(strict=True)
    commit = base._assert_bound_checkout(repo)  # noqa: SLF001
    completion, completion_ref, source_manifest = _source_completion(source_completion)
    try:
        upgrade_value = upgrade.verify_receipt(architecture_upgrade_receipt)
    except upgrade.UpgradeError as error:
        raise GatherRetrainError(f"architecture upgrade refused: {error}") from error
    if upgrade_value["source"] != completion["checkpoint"]:
        raise GatherRetrainError("architecture upgrade source is not the r3 champion")
    descriptor = source_manifest["source_descriptor"]
    sentinel = source_manifest["validation_sentinel"]
    base._verify_ref(descriptor, "source descriptor")  # noqa: SLF001
    base._verify_ref(sentinel, "validation sentinel")  # noqa: SLF001
    inventories, components = base._descriptor_inventory(  # noqa: SLF001
        Path(descriptor["path"])
    )
    output_root = output_root.expanduser().resolve(strict=False)
    manifest_path = manifest_path.expanduser().resolve(strict=False)
    if manifest_path.exists() or any(
        (output_root / name).exists()
        for name in (
            "candidate.pt",
            "candidate.pt.optimizer.pt",
            "train.report.json",
            "execution.claim.json",
            "submission.receipt.json",
            "completion.receipt.json",
        )
    ):
        raise GatherRetrainError("target-gather output identity is not fresh")

    command = list(source_manifest["command"])
    python_binding = base._python_binding(python)  # noqa: SLF001
    command[0] = python_binding["lexical_path"]
    trainer = next(value for value in command if Path(value).name == "train_bc.py")
    command[command.index(trainer)] = str((repo / "tools/train_bc.py").resolve(strict=True))
    replacements = {
        "--nproc-per-node": str(WORLD_SIZE),
        "--data": descriptor["path"],
        "--validation-game-sentinel-manifest": sentinel["path"],
        "--init-checkpoint": upgrade_value["upgraded_initializer"]["path"],
        "--checkpoint": str(output_root / "candidate.pt"),
        "--report": str(output_root / "train.report.json"),
        "--max-steps": str(OPTIMIZER_STEPS),
        "--batch-size": str(LOCAL_BATCH),
        "--grad-accum-steps": "1",
        "--soft-target-weight": "0.9",
        "--value-loss-weight": "0.25",
        "--loser-sample-weight": "1.0",
        "--policy-aux-active-batch-size": "0",
        "--action-module-lr-mult": "4.0",
        "--value-lr-mult": "1.0",
        "--freeze-modules": FREEZE_MODULES,
        "--require-only-trainable-prefixes": TRAINABLE_PREFIX,
    }
    for flag, value in replacements.items():
        _set(command, flag, value)
    _validate_geometry(command)
    source_files = {relative: base._ref(repo / relative) for relative in BOUND_SOURCE_FILES}  # noqa: SLF001
    operator = {
        "world_size": WORLD_SIZE,
        "per_rank_batch_size": LOCAL_BATCH,
        "optimizer_steps": OPTIMIZER_STEPS,
        "global_base_draws": GLOBAL_DRAWS,
        "current_fraction": 0.8,
        "current_n128_fraction": 5.0 / 7.0,
        "current_n256_fraction": 2.0 / 7.0,
        "exact_predecessor_replay_fraction": 0.2,
        "soft_target_weight": 0.9,
        "value_loss_weight": 0.25,
        "loser_sample_weight": 1.0,
        "action_module_lr_mult": 4.0,
        "freeze_modules": FREEZE_MODULES.split(","),
        "required_trainable_prefixes": [TRAINABLE_PREFIX],
        "fresh_optimizer": True,
    }
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "diagnostic_only": False,
        "production_eligible": True,
        "launch_authorized": True,
        "source_completion": completion_ref,
        "learner_source_incumbent": completion["checkpoint"],
        # The corpus was generated by f7.  r3 is only the learner initializer
        # and promotion baseline; these identities must never be conflated.
        "corpus_producer": source_manifest["f7_parent"],
        "function_preserving_upgrade": upgrade_value,
        "source_descriptor": descriptor,
        "validation_sentinel": sentinel,
        "component_bindings": components,
        "payload_inventory_acknowledgements": inventories,
        "operator": operator,
        "operator_sha256": base._digest(operator),  # noqa: SLF001
        "repo_binding": {
            "repository_root": str(repo),
            "public_main_commit": commit,
            "files": source_files,
        },
        "runtime_python": python_binding,
        "visible_devices": [0, 1, 2, 3],
        "command": command,
        "command_sha256": base._digest(command),  # noqa: SLF001
        "output_root": str(output_root),
    }
    manifest["manifest_sha256"] = base._digest(manifest)  # noqa: SLF001
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    base._write_exclusive(manifest_path, manifest, mode=0o444)  # noqa: SLF001
    return manifest


def verify(manifest_path: Path) -> dict[str, Any]:
    manifest_ref = base._ref(manifest_path)  # noqa: SLF001
    manifest = base._load(Path(manifest_ref["path"]))  # noqa: SLF001
    unhashed = dict(manifest)
    stated = unhashed.pop("manifest_sha256", None)
    if (
        stated != base._digest(unhashed)  # noqa: SLF001
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("diagnostic_only") is not False
        or manifest.get("production_eligible") is not True
        or manifest.get("launch_authorized") is not True
    ):
        raise GatherRetrainError("target-gather manifest schema/digest/authority drifted")
    completion, completion_ref, _ = _source_completion(
        Path(manifest["source_completion"]["path"])
    )
    if completion_ref != manifest["source_completion"]:
        raise GatherRetrainError("source completion bytes drifted")
    try:
        upgrade_value = upgrade.verify_receipt(
            Path(manifest["function_preserving_upgrade"]["receipt"]["path"])
        )
    except upgrade.UpgradeError as error:
        raise GatherRetrainError(f"architecture upgrade replay refused: {error}") from error
    if (
        upgrade_value != manifest["function_preserving_upgrade"]
        or upgrade_value["source"] != completion["checkpoint"]
        or manifest.get("learner_source_incumbent") != completion["checkpoint"]
    ):
        raise GatherRetrainError("architecture upgrade/source champion binding drifted")
    corpus_producer = manifest.get("corpus_producer")
    if (
        corpus_producer != base._load(  # noqa: SLF001
            Path(completion["manifest"]["path"])
        ).get("f7_parent")
        or corpus_producer == manifest.get("learner_source_incumbent")
    ):
        raise GatherRetrainError(
            "corpus producer f7 and learner source incumbent r3 are conflated"
        )
    base._verify_ref(corpus_producer, "corpus producer")  # noqa: SLF001
    operator = manifest.get("operator")
    expected_operator = {
        "world_size": 4,
        "per_rank_batch_size": 512,
        "optimizer_steps": 2048,
        "global_base_draws": 4_194_304,
        "current_fraction": 0.8,
        "current_n128_fraction": 5.0 / 7.0,
        "current_n256_fraction": 2.0 / 7.0,
        "exact_predecessor_replay_fraction": 0.2,
        "soft_target_weight": 0.9,
        "value_loss_weight": 0.25,
        "loser_sample_weight": 1.0,
        "action_module_lr_mult": 4.0,
        "freeze_modules": FREEZE_MODULES.split(","),
        "required_trainable_prefixes": [TRAINABLE_PREFIX],
        "fresh_optimizer": True,
    }
    if operator != expected_operator or manifest.get("operator_sha256") != base._digest(operator):  # noqa: SLF001
        raise GatherRetrainError("target-gather operator geometry drifted")
    descriptor = base._verify_ref(manifest.get("source_descriptor"), "descriptor")  # noqa: SLF001
    inventories, components = base._descriptor_inventory(descriptor)  # noqa: SLF001
    if (
        components != manifest.get("component_bindings")
        or inventories != manifest.get("payload_inventory_acknowledgements")
    ):
        raise GatherRetrainError("target-gather data mixture drifted")
    repo_binding = manifest.get("repo_binding")
    if not isinstance(repo_binding, dict):
        raise GatherRetrainError("repository binding is malformed")
    repo = Path(repo_binding["repository_root"]).resolve(strict=True)
    base._assert_bound_checkout(repo, repo_binding["public_main_commit"])  # noqa: SLF001
    for relative, ref in repo_binding.get("files", {}).items():
        if base._verify_ref(ref, f"source.{relative}") != (repo / relative).resolve(strict=True):  # noqa: SLF001
            raise GatherRetrainError(f"source path escaped checkout: {relative}")
    python = base._verify_python_binding(manifest.get("runtime_python"))  # noqa: SLF001
    command = manifest.get("command")
    if (
        not isinstance(command, list)
        or not all(isinstance(value, str) for value in command)
        or manifest.get("command_sha256") != base._digest(command)  # noqa: SLF001
        or command[0] != python
    ):
        raise GatherRetrainError("target-gather command/runtime binding drifted")
    _validate_geometry(command)
    exact_paths = {
        "--data": manifest["source_descriptor"]["path"],
        "--validation-game-sentinel-manifest": manifest["validation_sentinel"]["path"],
        "--init-checkpoint": upgrade_value["upgraded_initializer"]["path"],
    }
    for flag, expected in exact_paths.items():
        if base._option(command, flag) != expected:  # noqa: SLF001
            raise GatherRetrainError(f"target-gather command path drift at {flag}")
    output_root = Path(manifest["output_root"]).resolve(strict=False)
    if (
        base._option(command, "--checkpoint") != str(output_root / "candidate.pt")  # noqa: SLF001
        or base._option(command, "--report") != str(output_root / "train.report.json")  # noqa: SLF001
    ):
        raise GatherRetrainError("target-gather output paths drifted")
    return {
        "manifest": manifest,
        "manifest_ref": manifest_ref,
        "repo": repo,
        "command": command,
        "output_root": output_root,
    }


def execute(
    manifest_path: Path,
    *,
    unit: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    idle_probe: Callable[[], list[str]] = base._idle_b200s,  # noqa: SLF001
) -> dict[str, Any]:
    if base.SAFE_UNIT.fullmatch(unit) is None:
        raise GatherRetrainError("systemd unit name is invalid")
    verified = verify(manifest_path)
    conflicts = idle_probe()
    if conflicts:
        raise GatherRetrainError(f"B200 compute is not idle: {conflicts}")
    root = verified["output_root"]
    root.mkdir(parents=True, exist_ok=True)
    forbidden = [
        root / name
        for name in (
            "candidate.pt",
            "candidate.pt.optimizer.pt",
            "train.report.json",
            "execution.claim.json",
            "submission.receipt.json",
            "completion.receipt.json",
        )
    ]
    if any(path.exists() for path in forbidden):
        raise GatherRetrainError("target-gather one-shot identity is consumed")
    claim = {
        "schema_version": CLAIM_SCHEMA,
        "created_at_unix_ns": time.time_ns(),
        "manifest": verified["manifest_ref"],
        "unit": unit,
    }
    claim["claim_sha256"] = base._digest(claim)  # noqa: SLF001
    claim_path = root / "execution.claim.json"
    base._write_exclusive(claim_path, claim)  # noqa: SLF001
    stdout, stderr = root / "stdout.log", root / "stderr.log"
    systemd_command = [
        "sudo",
        "-n",
        "systemd-run",
        f"--unit={unit}",
        "--uid=ubuntu",
        "--gid=ubuntu",
        "--service-type=exec",
        "--property=LimitNOFILE=65536",
        f"--property=WorkingDirectory={verified['repo']}",
        f"--property=StandardOutput=append:{stdout}",
        f"--property=StandardError=append:{stderr}",
        "--setenv=HOME=/home/ubuntu",
        "--setenv=PYTHONNOUSERSITE=1",
        "--setenv=CUDA_VISIBLE_DEVICES=0,1,2,3",
        "--",
        *verified["command"],
    ]
    try:
        result = runner(systemd_command, check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise GatherRetrainError(f"systemd submission failed after claim: {error}") from error
    receipt = {
        "schema_version": SUBMISSION_SCHEMA,
        "diagnostic_only": False,
        "production_eligible": True,
        "created_at_unix_ns": time.time_ns(),
        "manifest": verified["manifest_ref"],
        "claim": {"path": str(claim_path), "sha256": base._file_sha(claim_path)},  # noqa: SLF001
        "unit": unit,
        "command_sha256": verified["manifest"]["command_sha256"],
        "systemd_command_sha256": base._digest(systemd_command),  # noqa: SLF001
        "systemd_stdout": result.stdout.strip(),
    }
    receipt["receipt_sha256"] = base._digest(receipt)  # noqa: SLF001
    base._write_exclusive(root / "submission.receipt.json", receipt)  # noqa: SLF001
    return receipt


def finalize(
    manifest_path: Path,
    *,
    unit: str,
    state_reader: Callable[..., str] = subprocess.check_output,
) -> dict[str, Any]:
    verified = verify(manifest_path)
    root = verified["output_root"]
    submission_path = root / "submission.receipt.json"
    submission = base._load(submission_path)  # noqa: SLF001
    if submission.get("schema_version") != SUBMISSION_SCHEMA or submission.get("unit") != unit:
        raise GatherRetrainError("submission receipt/unit does not match")
    state = state_reader(
        ("systemctl", "show", unit, "--property=ActiveState,Result,ExecMainStatus"),
        text=True,
    )
    fields = dict(row.split("=", 1) for row in state.splitlines() if "=" in row)
    if fields != {"ActiveState": "inactive", "Result": "success", "ExecMainStatus": "0"}:
        raise GatherRetrainError(f"target-gather retrain is not complete: {fields}")
    checkpoint = base._ref(root / "candidate.pt")  # noqa: SLF001
    report = base._ref(root / "train.report.json")  # noqa: SLF001
    payload = base._load(Path(report["path"]))  # noqa: SLF001
    expected = {
        "init_checkpoint": verified["manifest"]["function_preserving_upgrade"][
            "upgraded_initializer"
        ]["path"],
        "init_checkpoint_sha256": verified["manifest"]["function_preserving_upgrade"][
            "upgraded_initializer"
        ]["sha256"],
        "world_size": 4,
        "batch_size": 512,
        "effective_global_batch_size": 2048,
        "max_steps": 2048,
        "steps_completed": 2048,
        "training_row_draws": 4_194_304,
        "soft_target_weight": 0.9,
        "value_loss_weight": 0.25,
        "loser_sample_weight": 1.0,
        "action_module_lr_mult": 4.0,
        "freeze_modules": FREEZE_MODULES,
        "require_only_trainable_prefixes": TRAINABLE_PREFIX,
        "action_target_gather": True,
    }
    drift = {key: {"expected": value, "actual": payload.get(key)} for key, value in expected.items() if payload.get(key) != value}
    surface = payload.get("training_information_surface", {}).get(
        "required_trainable_surface"
    )
    if drift or not isinstance(surface, dict) or surface.get("prefixes") != [TRAINABLE_PREFIX]:
        raise GatherRetrainError(
            f"target-gather training report geometry/trainable surface drifted: {drift}"
        )
    progress_path = root / "candidate.pt.training-progress.json"
    optimizer_path = root / "candidate.pt.optimizer.pt"
    progress = base._load(progress_path)  # noqa: SLF001
    progress_unhashed = dict(progress)
    progress_digest = progress_unhashed.pop("progress_sha256", None)
    if (
        progress_digest != base._digest(progress_unhashed)  # noqa: SLF001
        or progress.get("optimizer_step") != OPTIMIZER_STEPS
        or progress.get("completed_epochs") != 1
        or not isinstance(progress.get("rank_torch_rng_states"), list)
        or len(progress["rank_torch_rng_states"]) != WORLD_SIZE
        or base._verify_ref(progress.get("checkpoint"), "progress checkpoint")  # noqa: SLF001
        != Path(checkpoint["path"])
        or base._verify_ref(progress.get("optimizer"), "progress optimizer")  # noqa: SLF001
        != optimizer_path.resolve(strict=True)
    ):
        raise GatherRetrainError("target-gather progress/RNG/optimizer dose drifted")
    completion = {
        "schema_version": COMPLETION_SCHEMA,
        "diagnostic_only": False,
        "production_eligible": True,
        "created_at_unix_ns": time.time_ns(),
        "manifest": verified["manifest_ref"],
        "submission": {"path": str(submission_path), "sha256": base._file_sha(submission_path)},  # noqa: SLF001
        "checkpoint": checkpoint,
        "report": report,
        "operator_sha256": verified["manifest"]["operator_sha256"],
        "progress": base._ref(progress_path),  # noqa: SLF001
        "optimizer": base._ref(optimizer_path),  # noqa: SLF001
        "unit_state": fields,
    }
    completion["receipt_sha256"] = base._digest(completion)  # noqa: SLF001
    base._write_exclusive(root / "completion.receipt.json", completion)  # noqa: SLF001
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-completion", required=True, type=Path)
    prep.add_argument("--architecture-upgrade-receipt", required=True, type=Path)
    prep.add_argument("--repo", required=True, type=Path)
    prep.add_argument("--output-root", required=True, type=Path)
    prep.add_argument("--manifest", required=True, type=Path)
    prep.add_argument("--python", required=True, type=Path)
    run = sub.add_parser("execute")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--unit", default="a1-production-target-gather")
    run.add_argument("--go", action="store_true")
    done = sub.add_parser("finalize")
    done.add_argument("--manifest", required=True, type=Path)
    done.add_argument("--unit", default="a1-production-target-gather")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "prepare":
            value = prepare(
                source_completion=args.source_completion,
                architecture_upgrade_receipt=args.architecture_upgrade_receipt,
                repo=args.repo,
                output_root=args.output_root,
                manifest_path=args.manifest,
                python=args.python,
            )
        elif args.action == "execute" and args.go:
            value = execute(args.manifest, unit=args.unit)
        elif args.action == "execute":
            value = verify(args.manifest)
        else:
            value = finalize(args.manifest, unit=args.unit)
        print(json.dumps(value, indent=2, sort_keys=True, default=str))
        return 0
    except (GatherRetrainError, OSError, KeyError, ValueError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
