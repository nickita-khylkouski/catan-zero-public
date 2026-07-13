#!/usr/bin/env python3
"""Seal and execute the production replication of the winning TEMP learner.

The winning checkpoint and its composite descriptor remain diagnostic-only.
This operator treats them as immutable selection evidence, reloads the exact f7
initializer, and performs a new one-shot training transaction.  Only the new
completion receipt is promotion eligible.
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

from tools import a1_production_l1_rerun as base  # noqa: E402


MANIFEST_SCHEMA = "a1-production-temperature-replication-v1"
CLAIM_SCHEMA = "a1-production-temperature-replication-claim-v1"
SUBMISSION_SCHEMA = "a1-production-temperature-replication-submission-v1"
COMPLETION_SCHEMA = "a1-production-temperature-replication-completion-v1"
DIAGNOSTIC_COMMAND_SCHEMA = "n256-temperature-arm-command-v1"
DIAGNOSTIC_COMPLETION_SCHEMA = "n256-temperature-arm-completion-v1"
F7_SHA256 = "sha256:f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4"
WINNING_DIAGNOSTIC_SHA256 = (
    "sha256:fefba044df58b9508de751d76d09bedeb630a2e832f6db46b70d95b5d4c77394"
)
COMPONENT_IDS = ("n128_current", "n256_current", "gen3_replay")
COMPONENT_RATIOS = (4.0 / 7.0, 1.6 / 7.0, 0.2)
COMPONENT_TEMPERATURES = {
    "n128_current": 1.0,
    "n256_current": 1.11,
    "gen3_replay": 0.52,
}
BOUND_SOURCE_FILES = (
    "tools/a1_production_temperature_replication.py",
    "tools/a1_production_l1_rerun.py",
    "tools/train_bc.py",
    "tools/mixed_memmap_corpus.py",
    "src/catan_zero/rl/entity_token_policy.py",
    "src/catan_zero/rl/entity_token_features.py",
)


class TemperatureReplicationError(RuntimeError):
    """The production temperature replication cannot be proven or executed."""


def _fail(message: str) -> None:
    raise TemperatureReplicationError(message)


def _load_ref(path: Path, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        ref = base._ref(path)  # noqa: SLF001
        return base._load(Path(ref["path"])), ref  # noqa: SLF001
    except (base.L1Error, OSError) as error:
        raise TemperatureReplicationError(f"{label}: {error}") from error


def _option(command: Sequence[str], flag: str) -> str:
    try:
        return base._option(command, flag)  # noqa: SLF001
    except base.L1Error as error:
        raise TemperatureReplicationError(str(error)) from error


def _set(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, item in enumerate(command) if item == flag]
    equals = [
        index for index, item in enumerate(command) if item.startswith(flag + "=")
    ]
    if len(positions) + len(equals) != 1:
        _fail(f"diagnostic command must contain exactly one {flag}")
    if equals:
        command[equals[0]] = f"{flag}={value}"
    else:
        command[positions[0] + 1] = value


def _verify_descriptor(
    path: Path,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    descriptor, _ = _load_ref(path, "descriptor")
    if (
        descriptor.get("schema_version") != "memmap_composite_v2"
        or descriptor.get("diagnostic_only") is not True
        or descriptor.get("promotion_eligible") is not False
    ):
        _fail(
            "winning descriptor must remain diagnostic_only=true and promotion_eligible=false"
        )
    components = descriptor.get("components")
    if (
        not isinstance(components, list)
        or tuple(row.get("component_id") for row in components if isinstance(row, dict))
        != COMPONENT_IDS
    ):
        _fail("temperature descriptor component identity/order drift")
    ratios = tuple(float(row.get("game_sampling_ratio", -1.0)) for row in components)
    if any(
        abs(actual - expected) > 1e-12
        for actual, expected in zip(ratios, COMPONENT_RATIOS)
    ):
        _fail(f"temperature descriptor component ratios drift: {ratios}")
    if descriptor.get("stored_policy_component_temperatures") != COMPONENT_TEMPERATURES:
        _fail("stored-policy temperature map drift")
    expected_scope = list(COMPONENT_IDS)
    if (
        descriptor.get("policy_distillation_component_ids") != expected_scope
        or descriptor.get("value_training_component_ids") != expected_scope
        or descriptor.get("policy_kl_anchor_component_ids") != ["gen3_replay"]
    ):
        _fail("temperature descriptor supervision scope drift")
    try:
        inventories, bindings = base._descriptor_inventory(path)  # noqa: SLF001
    except base.L1Error as error:
        raise TemperatureReplicationError(str(error)) from error
    return descriptor, inventories, bindings


def _verify_diagnostic_selection(
    *,
    completion_path: Path,
    command_path: Path,
    descriptor_path: Path,
    sentinel_path: Path,
    f7_path: Path,
    checkpoint_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    completion, completion_ref = _load_ref(completion_path, "diagnostic completion")
    command_doc, command_ref = _load_ref(command_path, "diagnostic command")
    evidence, evidence_ref = _load_ref(evidence_path, "winning evidence")
    descriptor_ref = base._ref(descriptor_path)  # noqa: SLF001
    sentinel_ref = base._ref(sentinel_path)  # noqa: SLF001
    f7_ref = base._ref(f7_path)  # noqa: SLF001
    checkpoint_ref = base._ref(checkpoint_path)  # noqa: SLF001
    if (
        f7_ref["sha256"] != F7_SHA256
        or checkpoint_ref["sha256"] != WINNING_DIAGNOSTIC_SHA256
    ):
        _fail(
            "f7 or selected diagnostic checkpoint bytes differ from the winning lineage"
        )
    if (
        completion.get("schema_version") != DIAGNOSTIC_COMPLETION_SCHEMA
        or completion.get("state") != "complete"
        or completion.get("checkpoint_sha256") != WINNING_DIAGNOSTIC_SHA256
        or completion.get("parent_checkpoint_sha256") != F7_SHA256
        or completion.get("descriptor_sha256") != descriptor_ref["sha256"]
        or completion.get("global_sample_dose") != 4_194_304
        or completion.get("optimizer_steps") != 1024
        or completion.get("world_size") != 8
        or completion.get("batch_size_per_rank") != 512
    ):
        _fail("diagnostic completion is not the sealed winning TEMP dose")
    command = command_doc.get("argv")
    if (
        command_doc.get("schema_version") != DIAGNOSTIC_COMMAND_SCHEMA
        or not isinstance(command, list)
        or not all(isinstance(item, str) for item in command)
        or command_doc.get("argv_sha256") != base._digest(command)  # noqa: SLF001
    ):
        _fail("diagnostic command receipt is malformed or drifted")
    if completion.get("command_sha256") != command_doc.get("argv_sha256"):
        _fail("diagnostic completion/command binding drift")
    exact_evidence = {
        "candidate_checkpoint_sha256": WINNING_DIAGNOSTIC_SHA256,
        "baseline_checkpoint_sha256": F7_SHA256,
        "candidate_wins": 670,
        "baseline_wins": 530,
        "games_played": 1200,
        "complete_pairs": 600,
        "games_truncated": 0,
    }
    if any(evidence.get(key) != value for key, value in exact_evidence.items()):
        _fail("winning 670-530/1200 evaluation evidence drift")
    errors = evidence.get("errors")
    if errors not in ([], {}, 0, None):
        _fail("winning evidence contains evaluation errors")
    for field in ("sprt", "pentanomial_sprt", "superiority_pentanomial_sprt"):
        row = evidence.get(field)
        if (
            not isinstance(row, dict)
            or row.get("decision") != "H1"
            or float(row.get("llr", float("-inf")))
            < float(row.get("upper_bound", float("inf")))
        ):
            _fail(f"winning evidence lacks crossed-H1 {field}")
    return {
        "completion": completion_ref,
        "command_receipt": command_ref,
        "descriptor": descriptor_ref,
        "sentinel": sentinel_ref,
        "f7": f7_ref,
        "checkpoint": checkpoint_ref,
        "evidence": evidence_ref,
        "command": list(command),
    }


def _validate_recipe(
    command: list[str], *, descriptor: str, sentinel: str, f7: str
) -> None:
    exact = {
        "--nproc-per-node": "8",
        "--data": descriptor,
        "--data-format": "memmap",
        "--init-checkpoint": f7,
        "--arch": "entity_graph",
        "--hidden-size": "640",
        "--graph-layers": "6",
        "--attention-heads": "8",
        "--graph-dropout": "0.05",
        "--entity-state-trunk": "transformer",
        "--track": "2p_no_trade",
        "--vps-to-win": "10",
        "--epochs": "1",
        "--max-steps": "1024",
        "--batch-size": "512",
        "--grad-accum-steps": "1",
        "--seed": "1",
        "--optimizer": "adam",
        "--lr": "3e-05",
        "--lr-warmup-steps": "100",
        "--lr-schedule": "flat",
        "--weight-decay": "0.0",
        "--value-lr-mult": "0.3",
        "--action-module-lr-mult": "1.0",
        "--policy-loss-weight": "1.0",
        "--soft-target-source": "policy",
        "--soft-target-weight": "0.9",
        "--soft-target-min-legal-coverage": "0.5",
        "--value-loss-weight": "0.25",
        "--value-target-lambda": "1.0",
        "--value-head-type": "mse",
        "--truncated-vp-margin-value-weight": "0.0",
        "--final-vp-loss-weight": "0.0",
        "--q-loss-weight": "0.0",
        "--policy-kl-anchor-weight": "0.0",
        "--policy-kl-anchor-direction": "forward",
        "--forced-action-weight": "0.0",
        "--forced-row-value-weight": "1.0",
        "--winner-sample-weight": "1.0",
        "--loser-sample-weight": "1.0",
        "--validation-max-samples": "0",
        "--data-loader-workers": "4",
        "--data-loader-prefetch": "4",
        "--validation-game-sentinel-manifest": sentinel,
    }
    for flag, expected in exact.items():
        if _option(command, flag) != expected:
            _fail(f"winning TEMP recipe drift at {flag}")
    required = {
        "--standalone",
        "--training-rng-rank-offset",
        "--no-resume-optimizer",
        "--no-fused-optimizer",
        "--graph-history-features",
        "--mask-hidden-info",
        "--skip-teacher-quality-gate",
        "--trust-curated-data-quality",
    }
    missing = sorted(required - set(command))
    if missing:
        _fail(f"winning TEMP recipe lacks required flags: {missing}")
    forbidden = {
        "--resume-optimizer",
        "--fsdp",
        "--ddp-shard-data",
        "--train-value-only",
    }
    if forbidden & set(command):
        _fail(
            f"winning TEMP recipe enables forbidden modes: {sorted(forbidden & set(command))}"
        )


def prepare(
    *,
    diagnostic_completion: Path,
    diagnostic_command: Path,
    descriptor: Path,
    validation_sentinel: Path,
    f7_checkpoint: Path,
    diagnostic_checkpoint: Path,
    winning_evidence: Path,
    repo: Path,
    output_root: Path,
    manifest_path: Path,
    python: Path,
) -> dict[str, Any]:
    repo = repo.expanduser().resolve(strict=True)
    try:
        commit = base._assert_bound_checkout(repo)  # noqa: SLF001
    except base.L1Error as error:
        raise TemperatureReplicationError(str(error)) from error
    selected = _verify_diagnostic_selection(
        completion_path=diagnostic_completion,
        command_path=diagnostic_command,
        descriptor_path=descriptor,
        sentinel_path=validation_sentinel,
        f7_path=f7_checkpoint,
        checkpoint_path=diagnostic_checkpoint,
        evidence_path=winning_evidence,
    )
    _, inventories, bindings = _verify_descriptor(descriptor)
    _validate_recipe(
        selected["command"],
        descriptor=selected["descriptor"]["path"],
        sentinel=selected["sentinel"]["path"],
        f7=selected["f7"]["path"],
    )
    ack_positions = [
        index for index, item in enumerate(selected["command"]) if item == base.ACK_FLAG
    ]
    observed_acks = [selected["command"][index + 1] for index in ack_positions]
    if observed_acks != inventories:
        _fail(
            "diagnostic command lacks the descriptor-ordered event-history acknowledgements"
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
        _fail("production temperature output identity is not fresh")
    try:
        python_binding = base._python_binding(python)  # noqa: SLF001
    except base.L1Error as error:
        raise TemperatureReplicationError(str(error)) from error
    command = list(selected["command"])
    command[0] = python_binding["lexical_path"]
    trainers = [item for item in command if Path(item).name == "train_bc.py"]
    if len(trainers) != 1:
        _fail("diagnostic command does not name exactly one trainer")
    command[command.index(trainers[0])] = str(
        (repo / "tools/train_bc.py").resolve(strict=True)
    )
    _set(command, "--checkpoint", str(output_root / "candidate.pt"))
    _set(command, "--report", str(output_root / "train.report.json"))
    source_files = {
        relative: base._ref(repo / relative) for relative in BOUND_SOURCE_FILES
    }  # noqa: SLF001
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "diagnostic_only": False,
        "production_eligible": True,
        "launch_authorized": True,
        "selection_evidence": {
            "diagnostic_only": True,
            "promotion_eligible": False,
            **{
                key: selected[key]
                for key in ("completion", "command_receipt", "checkpoint", "evidence")
            },
        },
        "source_descriptor": selected["descriptor"],
        "validation_sentinel": selected["sentinel"],
        "f7_parent": selected["f7"],
        "component_bindings": bindings,
        "stored_policy_component_temperatures": COMPONENT_TEMPERATURES,
        "event_history_training_contract": {
            "public_observation_masked": True,
            "graph_history_features": True,
            "payload_inventory_acknowledgements": inventories,
        },
        "selected_dose": {
            "optimizer_steps": 1024,
            "world_size": 8,
            "per_rank_batch_size": 512,
            "global_samples": 4_194_304,
            "optimizer": "fresh_adam",
            "lr": 3e-5,
            "training_rng_rank_offset": True,
        },
        "repo_binding": {
            "repository_root": str(repo),
            "public_main_commit": commit,
            "files": source_files,
        },
        "runtime_python": python_binding,
        "execution_preconditions": {
            "visible_gpu_count": 8,
            "gpu_model_substring": "B200",
            "all_compute_idle": True,
            "one_shot_systemd": True,
        },
        "command": command,
        "command_sha256": base._digest(command),  # noqa: SLF001
        "output_root": str(output_root),
    }
    manifest["manifest_sha256"] = base._digest(manifest)  # noqa: SLF001
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    base._write_exclusive(manifest_path, manifest, mode=0o444)  # noqa: SLF001
    return manifest


def verify(manifest_path: Path) -> dict[str, Any]:
    try:
        manifest_ref = base._ref(manifest_path)  # noqa: SLF001
        manifest = base._load(Path(manifest_ref["path"]))  # noqa: SLF001
    except base.L1Error as error:
        raise TemperatureReplicationError(str(error)) from error
    unhashed = dict(manifest)
    stated = unhashed.pop("manifest_sha256", None)
    if stated != base._digest(unhashed):  # noqa: SLF001
        _fail("production temperature manifest semantic digest drift")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("diagnostic_only") is not False
        or manifest.get("production_eligible") is not True
        or manifest.get("launch_authorized") is not True
    ):
        _fail("manifest does not authorize production temperature replication")
    selection = manifest.get("selection_evidence")
    if (
        not isinstance(selection, dict)
        or selection.get("diagnostic_only") is not True
        or selection.get("promotion_eligible") is not False
    ):
        _fail("diagnostic selection evidence was relabelled")
    for key in ("completion", "command_receipt", "checkpoint", "evidence"):
        base._verify_ref(selection.get(key), f"selection.{key}")  # noqa: SLF001
    for key in ("source_descriptor", "validation_sentinel", "f7_parent"):
        base._verify_ref(manifest.get(key), key)  # noqa: SLF001
    if manifest["f7_parent"]["sha256"] != F7_SHA256:
        _fail("production initializer is not exact f7")
    replayed_selection = _verify_diagnostic_selection(
        completion_path=Path(selection["completion"]["path"]),
        command_path=Path(selection["command_receipt"]["path"]),
        descriptor_path=Path(manifest["source_descriptor"]["path"]),
        sentinel_path=Path(manifest["validation_sentinel"]["path"]),
        f7_path=Path(manifest["f7_parent"]["path"]),
        checkpoint_path=Path(selection["checkpoint"]["path"]),
        evidence_path=Path(selection["evidence"]["path"]),
    )
    for key in ("completion", "command_receipt", "checkpoint", "evidence"):
        if replayed_selection[key] != selection[key]:
            _fail(f"selection replay reference drift at {key}")
    _verify_descriptor(Path(manifest["source_descriptor"]["path"]))
    repo_binding = manifest.get("repo_binding")
    if not isinstance(repo_binding, dict):
        _fail("repository binding is malformed")
    repo = Path(str(repo_binding["repository_root"])).resolve(strict=True)
    base._assert_bound_checkout(repo, str(repo_binding["public_main_commit"]))  # noqa: SLF001
    for relative, ref in repo_binding.get("files", {}).items():
        if base._verify_ref(ref, f"source.{relative}") != (repo / relative).resolve(
            strict=True
        ):  # noqa: SLF001
            _fail(f"bound source escaped checkout: {relative}")
    lexical_python = base._verify_python_binding(manifest.get("runtime_python"))  # noqa: SLF001
    command = manifest.get("command")
    if not isinstance(command, list) or not all(
        isinstance(item, str) for item in command
    ):
        _fail("production command is malformed")
    if manifest.get("command_sha256") != base._digest(command):  # noqa: SLF001
        _fail("production command digest drift")
    if command[0] != lexical_python:
        _fail("production command does not use bound venv Python")
    _validate_recipe(
        command,
        descriptor=manifest["source_descriptor"]["path"],
        sentinel=manifest["validation_sentinel"]["path"],
        f7=manifest["f7_parent"]["path"],
    )
    inventories = manifest["event_history_training_contract"][
        "payload_inventory_acknowledgements"
    ]
    positions = [index for index, item in enumerate(command) if item == base.ACK_FLAG]
    if [command[index + 1] for index in positions] != inventories:
        _fail("production command event-history acknowledgement drift")
    trainers = [
        Path(item).resolve() for item in command if Path(item).name == "train_bc.py"
    ]
    if trainers != [(repo / "tools/train_bc.py").resolve(strict=True)]:
        _fail("production command trainer escaped bound checkout")
    root = Path(str(manifest["output_root"])).resolve(strict=False)
    if _option(command, "--checkpoint") != str(root / "candidate.pt") or _option(
        command, "--report"
    ) != str(root / "train.report.json"):
        _fail("production output paths are not canonical")
    return {
        "manifest": manifest,
        "manifest_ref": manifest_ref,
        "repo": repo,
        "command": command,
        "output_root": root,
    }


def execute(
    manifest_path: Path,
    *,
    unit: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    idle_probe: Callable[[], list[str]] = base._idle_b200s,  # noqa: SLF001
) -> dict[str, Any]:
    if base.SAFE_UNIT.fullmatch(unit) is None:
        _fail("systemd unit name is invalid")
    verified = verify(manifest_path)
    conflicts = idle_probe()
    if conflicts:
        _fail(f"B200 compute is not idle: {conflicts}")
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
        _fail("production temperature transaction was already consumed")
    claim = {
        "schema_version": CLAIM_SCHEMA,
        "created_at_unix_ns": time.time_ns(),
        "manifest": verified["manifest_ref"],
        "unit": unit,
    }
    claim["claim_sha256"] = base._digest(claim)  # noqa: SLF001
    claim_path = root / "execution.claim.json"
    base._write_exclusive(claim_path, claim)  # noqa: SLF001
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
        f"--property=StandardOutput=append:{root / 'stdout.log'}",
        f"--property=StandardError=append:{root / 'stderr.log'}",
        "--setenv=HOME=/home/ubuntu",
        "--setenv=PYTHONNOUSERSITE=1",
        "--",
        *verified["command"],
    ]
    try:
        result = runner(systemd_command, check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise TemperatureReplicationError(
            f"systemd submission failed after one-shot claim: {error}"
        ) from error
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


def finalize(manifest_path: Path, *, unit: str) -> dict[str, Any]:
    verified = verify(manifest_path)
    root = verified["output_root"]
    submission_path = root / "submission.receipt.json"
    submission = base._load(submission_path)  # noqa: SLF001
    unhashed = dict(submission)
    submission_hash = unhashed.pop("receipt_sha256", None)
    if (
        submission.get("schema_version") != SUBMISSION_SCHEMA
        or submission.get("unit") != unit
        or submission.get("manifest") != verified["manifest_ref"]
        or submission_hash != base._digest(unhashed)  # noqa: SLF001
    ):
        _fail("submission receipt/unit/manifest binding drift")
    try:
        state = subprocess.check_output(
            ("systemctl", "show", unit, "--property=ActiveState,Result,ExecMainStatus"),
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise TemperatureReplicationError(
            f"cannot read systemd state: {error}"
        ) from error
    fields = dict(row.split("=", 1) for row in state.splitlines() if "=" in row)
    expected_state = {
        "ActiveState": "inactive",
        "Result": "success",
        "ExecMainStatus": "0",
    }
    if fields != expected_state:
        _fail(f"production temperature replication is not complete: {fields}")
    checkpoint = base._ref(root / "candidate.pt")  # noqa: SLF001
    report = base._ref(root / "train.report.json")  # noqa: SLF001
    report_payload = base._load(Path(report["path"]))  # noqa: SLF001
    if (
        report_payload.get("init_checkpoint_sha256") != F7_SHA256
        or report_payload.get("base_training_row_draws") != 4_194_304
        or report_payload.get("max_steps") != 1024
        or report_payload.get("batch_size") != 512
        or report_payload.get("optimizer") != "adam"
        or report_payload.get("resume_optimizer") is not False
        or report_payload.get("lr") != 3e-5
        or report_payload.get("training_rng_rank_offset") is not True
    ):
        _fail("completed report does not prove the exact fresh-f7 TEMP replication")
    # The generic descriptor remains diagnostic by design.  Eligibility comes
    # exclusively from this independently executed, sealed transaction.
    if report_payload.get("diagnostic_only") is not True:
        _fail("generic trainer unexpectedly relabelled the diagnostic descriptor")
    completion = {
        "schema_version": COMPLETION_SCHEMA,
        "diagnostic_only": False,
        "production_eligible": True,
        "created_at_unix_ns": time.time_ns(),
        "manifest": verified["manifest_ref"],
        "submission": {
            "path": str(submission_path),
            "sha256": base._file_sha(submission_path),
        },  # noqa: SLF001
        "checkpoint": checkpoint,
        "report": report,
        "unit_state": fields,
        "replication_contract": {
            "initializer_sha256": F7_SHA256,
            "global_samples": 4_194_304,
            "optimizer": "fresh_adam",
            "stored_policy_component_temperatures": COMPONENT_TEMPERATURES,
            "diagnostic_selection_artifact_relabelled": False,
        },
    }
    completion["receipt_sha256"] = base._digest(completion)  # noqa: SLF001
    base._write_exclusive(root / "completion.receipt.json", completion)  # noqa: SLF001
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    for flag in (
        "diagnostic-completion",
        "diagnostic-command",
        "descriptor",
        "validation-sentinel",
        "f7-checkpoint",
        "diagnostic-checkpoint",
        "winning-evidence",
        "repo",
        "output-root",
        "manifest",
        "python",
    ):
        prep.add_argument("--" + flag, required=True, type=Path)
    run = sub.add_parser("execute")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--unit", default="a1-production-temperature-replication")
    run.add_argument("--go", action="store_true")
    done = sub.add_parser("finalize")
    done.add_argument("--manifest", required=True, type=Path)
    done.add_argument("--unit", default="a1-production-temperature-replication")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "prepare":
            payload = prepare(
                diagnostic_completion=args.diagnostic_completion,
                diagnostic_command=args.diagnostic_command,
                descriptor=args.descriptor,
                validation_sentinel=args.validation_sentinel,
                f7_checkpoint=args.f7_checkpoint,
                diagnostic_checkpoint=args.diagnostic_checkpoint,
                winning_evidence=args.winning_evidence,
                repo=args.repo,
                output_root=args.output_root,
                manifest_path=args.manifest,
                python=args.python,
            )
            print(
                json.dumps(
                    {
                        "prepared": True,
                        "launched": False,
                        "manifest_sha256": payload["manifest_sha256"],
                    },
                    sort_keys=True,
                )
            )
        elif args.action == "execute" and not args.go:
            payload = verify(args.manifest)
            print(
                json.dumps(
                    {
                        "verified": True,
                        "launched": False,
                        "manifest": payload["manifest_ref"],
                    },
                    sort_keys=True,
                )
            )
        elif args.action == "execute":
            payload = execute(args.manifest, unit=args.unit)
            print(
                json.dumps(
                    {"submitted": True, "receipt_sha256": payload["receipt_sha256"]},
                    sort_keys=True,
                )
            )
        else:
            payload = finalize(args.manifest, unit=args.unit)
            print(
                json.dumps(
                    {"completed": True, "receipt_sha256": payload["receipt_sha256"]},
                    sort_keys=True,
                )
            )
        return 0
    except (
        TemperatureReplicationError,
        base.L1Error,
        OSError,
        KeyError,
        ValueError,
    ) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
