#!/usr/bin/env python3
"""Seal, submit, and finalize the selected production L1 learner rerun.

This tool deliberately derives the new run from the completed historical L1
receipt instead of editing that receipt.  Preparation is read-only with
respect to the historical run.  Submission is one-shot and refuses unless the
bound public-main checkout is clean and exactly eight visible B200s are idle.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Sequence


MANIFEST_SCHEMA = "a1-production-l1-rerun-v1"
CLAIM_SCHEMA = "a1-production-l1-rerun-claim-v1"
SUBMISSION_SCHEMA = "a1-production-l1-rerun-submission-v1"
COMPLETION_SCHEMA = "a1-production-l1-rerun-completion-v1"
SOURCE_SCHEMA = "a1-p2-independent-loser1-control-v1"
ACK_FLAG = "--acknowledge-empty-event-history-payload-inventory-sha256"
CROP_FLAG = "--crop-authenticated-empty-event-history"
BOUND_SOURCE_FILES = (
    "tools/a1_production_l1_rerun.py",
    "tools/train_bc.py",
    "tools/mixed_memmap_corpus.py",
    "tools/audit_entity_graph_information_surface.py",
    "src/catan_zero/rl/entity_token_policy.py",
    "src/catan_zero/rl/entity_token_features.py",
)
SAFE_UNIT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,79}")


class L1Error(RuntimeError):
    """The production L1 contract cannot be proven or executed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise L1Error(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise L1Error(f"{path} must contain an object")
    return value


def _ref(path: Path) -> dict[str, str]:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise L1Error(f"cannot resolve input {path}: {error}") from error
    if resolved.is_symlink() or not resolved.is_file():
        raise L1Error(f"input is not a regular file: {resolved}")
    return {"path": str(resolved), "sha256": _file_sha(resolved)}


def _python_binding(path: Path) -> dict[str, str]:
    """Bind venv Python bytes without replacing its lexical activation path."""

    lexical = path.expanduser()
    if not lexical.is_absolute() or not lexical.exists() or not os.access(lexical, os.X_OK):
        raise L1Error(f"Python must be an executable absolute path: {lexical}")
    resolved = lexical.resolve(strict=True)
    return {
        "lexical_path": str(lexical),
        "resolved_path": str(resolved),
        "sha256": _file_sha(resolved),
    }


def _verify_python_binding(value: Any) -> str:
    if not isinstance(value, dict) or set(value) != {
        "lexical_path", "resolved_path", "sha256"
    }:
        raise L1Error("runtime Python binding is malformed")
    lexical = Path(str(value["lexical_path"]))
    if not lexical.is_absolute() or not lexical.exists() or not os.access(lexical, os.X_OK):
        raise L1Error("bound lexical venv Python is unavailable")
    resolved = lexical.resolve(strict=True)
    if str(resolved) != value["resolved_path"] or _file_sha(resolved) != value["sha256"]:
        raise L1Error("bound runtime Python drifted")
    return str(lexical)


def _verify_ref(value: Any, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise L1Error(f"{label} reference is malformed")
    path = Path(str(value["path"])).expanduser().resolve(strict=True)
    if path.is_symlink() or not path.is_file() or _file_sha(path) != value["sha256"]:
        raise L1Error(f"{label} bytes drifted")
    return path


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(("git", *args), cwd=repo, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise L1Error(f"git {' '.join(args)} failed in {repo}: {error}") from error


def _assert_bound_checkout(repo: Path, expected_commit: str | None = None) -> str:
    repo = repo.resolve(strict=True)
    head = _git(repo, "rev-parse", "HEAD")
    if expected_commit is not None and head != expected_commit:
        raise L1Error(f"checkout HEAD drift: expected {expected_commit}, found {head}")
    if _git(repo, "status", "--porcelain"):
        raise L1Error("production checkout is not clean")
    origin_main = _git(repo, "rev-parse", "refs/remotes/origin/main")
    if origin_main != head:
        raise L1Error(f"checkout is not the fetched public origin/main: {origin_main}")
    return head


def _option(command: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(command) if value == flag]
    equals = [
        value[len(flag) + 1 :]
        for value in command
        if value.startswith(flag + "=")
    ]
    if len(positions) + len(equals) != 1:
        raise L1Error(f"command must contain exactly one {flag}")
    if equals:
        return equals[0]
    if positions[0] + 1 >= len(command):
        raise L1Error(f"command has no value for {flag}")
    return command[positions[0] + 1]


def _replace_option(command: list[str], flag: str, value: str) -> None:
    index = command.index(flag)
    command[index + 1] = value


def _validate_historical_recipe(command: list[str]) -> None:
    exact = {
        "--nproc-per-node": "8",
        "--arch": "entity_graph",
        "--hidden-size": "640",
        "--graph-layers": "6",
        "--attention-heads": "8",
        "--epochs": "1",
        "--max-steps": "1024",
        "--batch-size": "512",
        "--grad-accum-steps": "1",
        "--optimizer": "adam",
        "--lr": "3e-05",
        "--lr-warmup-steps": "100",
        "--soft-target-weight": "0.9",
        "--value-loss-weight": "0.25",
        "--forced-action-weight": "0.0",
        "--loser-sample-weight": "1.0",
    }
    for flag, expected in exact.items():
        if _option(command, flag) != expected:
            raise L1Error(f"historical L1 drift at {flag}: expected {expected}")
    required = {
        "--no-resume-optimizer", "--no-fused-optimizer", "--mask-hidden-info",
        "--graph-history-features", "--trust-curated-data-quality",
    }
    missing = sorted(required - set(command))
    if missing:
        raise L1Error(f"historical L1 command lacks required flags: {missing}")
    if ACK_FLAG in command or CROP_FLAG in command or "--max-grad-norm" in command:
        raise L1Error("historical receipt unexpectedly contains latest-main additions")


def _historical_projection(command: list[str], inventories: Sequence[str]) -> list[str]:
    """Remove only the four latest-main additions from a derived command."""

    projected = list(command)
    for flag, expected in (
        ("--max-grad-norm", "1.0"),
        ("--policy-aux-active-batch-size", "0"),
    ):
        if _option(projected, flag) != expected:
            raise L1Error(f"derived command drift at {flag}")
        index = projected.index(flag)
        del projected[index : index + 2]
    for expected in inventories:
        try:
            index = projected.index(ACK_FLAG)
        except ValueError as error:
            raise L1Error("derived command is missing an inventory ACK") from error
        if index + 1 >= len(projected) or projected[index + 1] != expected:
            raise L1Error("derived inventory ACK order drift")
        del projected[index : index + 2]
    if ACK_FLAG in projected:
        raise L1Error("derived command contains an extra inventory ACK")
    if projected.count(CROP_FLAG) != 1:
        raise L1Error("derived command must contain one crop flag")
    projected.remove(CROP_FLAG)
    return projected


def _descriptor_inventory(descriptor: Path) -> tuple[list[str], list[dict[str, Any]]]:
    payload = _load(descriptor)
    components = payload.get("components")
    if payload.get("schema_version") != "memmap_composite_v2" or not isinstance(
        components, list
    ) or len(components) != 3:
        raise L1Error("L1 descriptor must contain exactly three v2 components")
    expected_ids = ["n128_current", "n256_current", "gen3_replay"]
    if [row.get("component_id") for row in components] != expected_ids:
        raise L1Error("L1 descriptor component order/identity drift")
    ratios = [float(row.get("game_sampling_ratio", -1.0)) for row in components]
    expected_ratios = [4.0 / 7.0, 1.6 / 7.0, 0.2]
    if any(abs(left - right) > 1e-12 for left, right in zip(ratios, expected_ratios)):
        raise L1Error(f"L1 descriptor sampling ratios drifted: {ratios}")
    inventories: list[str] = []
    bindings: list[dict[str, Any]] = []
    for row in components:
        inventory = row.get("payload_inventory_sha256")
        if not isinstance(inventory, str) or not inventory.startswith("sha256:"):
            raise L1Error("component payload inventory is malformed")
        meta = _ref(Path(str(row["corpus_dir"])) / "corpus_meta.json")
        validation = _ref(Path(str(row["validation_manifest"])))
        if meta["sha256"] != row.get("corpus_meta_sha256"):
            raise L1Error(f"{row['component_id']} corpus metadata drift")
        if validation["sha256"] != row.get("validation_manifest_sha256"):
            raise L1Error(f"{row['component_id']} validation manifest drift")
        inventories.append(inventory)
        bindings.append(
            {
                "component_id": row["component_id"],
                "corpus_meta": meta,
                "validation_manifest": validation,
                "payload_inventory_sha256": inventory,
            }
        )
    return inventories, bindings


def prepare(
    *, source_receipt: Path, repo: Path, output_root: Path, manifest_path: Path,
    python: Path, failed_attempt_root: Path | None = None,
) -> dict[str, Any]:
    repo = repo.expanduser().resolve(strict=True)
    commit = _assert_bound_checkout(repo)
    receipt_ref = _ref(source_receipt)
    receipt = _load(Path(receipt_ref["path"]))
    if (
        receipt.get("schema_version") != SOURCE_SCHEMA
        or receipt.get("causal_delta") != {"loser_sample_weight": {"from": 0.3, "to": 1.0}}
        or receipt.get("repo_commit") != "48065121a907a4eab3559c83c389b7c116857dd5"
    ):
        raise L1Error("source receipt is not the independently completed L1 control")
    source_command = receipt.get("command")
    if not isinstance(source_command, list) or not all(
        isinstance(value, str) for value in source_command
    ):
        raise L1Error("source L1 command is malformed")
    _validate_historical_recipe(source_command)
    if _digest(source_command) != receipt.get("command_sha256"):
        raise L1Error("source L1 command semantic digest drift")
    descriptor = _ref(Path(str(receipt["descriptor"])))
    sentinel = _ref(Path(str(receipt["sentinel"])))
    parent = _ref(Path(str(receipt["parent_checkpoint"])))
    for ref, key in (
        (descriptor, "descriptor_sha256"), (sentinel, "sentinel_sha256"),
        (parent, "parent_checkpoint_sha256"),
    ):
        if ref["sha256"] != receipt.get(key):
            raise L1Error(f"source receipt {key} no longer matches bytes")
    inventories, component_bindings = _descriptor_inventory(Path(descriptor["path"]))
    output_root = output_root.expanduser().resolve(strict=False)
    manifest_path = manifest_path.expanduser().resolve(strict=False)
    if manifest_path.exists():
        raise L1Error(f"manifest already exists: {manifest_path}")
    if any((output_root / name).exists() for name in (
        "candidate.pt", "candidate.pt.optimizer.pt", "train.report.json",
        "execution.claim.json", "submission.receipt.json", "completion.receipt.json",
    )):
        raise L1Error("production L1 output root is not fresh")
    python_binding = _python_binding(python)
    command = list(source_command)
    command[0] = python_binding["lexical_path"]
    trainer_index = command.index(next(v for v in command if Path(v).name == "train_bc.py"))
    command[trainer_index] = str((repo / "tools/train_bc.py").resolve(strict=True))
    _replace_option(command, "--data", descriptor["path"])
    _replace_option(command, "--validation-game-sentinel-manifest", sentinel["path"])
    _replace_option(command, "--init-checkpoint", parent["path"])
    _replace_option(command, "--checkpoint", str(output_root / "candidate.pt"))
    _replace_option(command, "--report", str(output_root / "train.report.json"))
    command.extend(["--max-grad-norm", "1.0", "--policy-aux-active-batch-size", "0"])
    for inventory in inventories:
        command.extend([ACK_FLAG, inventory])
    command.append(CROP_FLAG)
    source_files = {relative: _ref(repo / relative) for relative in BOUND_SOURCE_FILES}
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "diagnostic_only": False,
        "production_eligible": True,
        "launch_authorized": True,
        "selected_dose": {
            "optimizer_steps": 1024,
            "world_size": 8,
            "per_rank_batch_size": 512,
            "global_samples": 4_194_304,
            "policy_aux_active_batch_size": 0,
            "selection": "exact independently successful L1 dose",
        },
        "source_receipt": receipt_ref,
        "source_descriptor": descriptor,
        "validation_sentinel": sentinel,
        "f7_parent": parent,
        "component_bindings": component_bindings,
        "event_history_training_contract": {
            "authenticated_empty": True,
            "crop_authenticated_empty_event_history": True,
            "payload_inventory_acknowledgements": inventories,
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
            "fresh_adam": True,
        },
        "command": command,
        "command_sha256": _digest(command),
        "output_root": str(output_root),
    }
    if failed_attempt_root is not None:
        failed_root = failed_attempt_root.expanduser().resolve(strict=True)
        absent = [failed_root / "candidate.pt", failed_root / "train.report.json"]
        if any(path.exists() for path in absent):
            raise L1Error("failed retry lineage unexpectedly contains training outputs")
        stderr = _ref(failed_root / "stderr.log")
        stderr_text = Path(stderr["path"]).read_text(encoding="utf-8", errors="replace")
        if "No module named 'torch'" not in stderr_text:
            raise L1Error("failed retry lineage is not the authorized pre-training venv failure")
        manifest["pre_training_retry_lineage"] = {
            "repair": "preserve lexical venv interpreter path",
            "optimizer_steps": 0,
            "outputs": None,
            "claim": _ref(failed_root / "execution.claim.json"),
            "submission": _ref(failed_root / "submission.receipt.json"),
            "stderr": stderr,
        }
    manifest["manifest_sha256"] = _digest(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_fd = os.open(
        manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444
    )
    with os.fdopen(descriptor_fd, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def verify(manifest_path: Path) -> dict[str, Any]:
    manifest_ref = _ref(manifest_path)
    manifest = _load(Path(manifest_ref["path"]))
    stated = manifest.get("manifest_sha256")
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if stated != _digest(unhashed):
        raise L1Error("manifest semantic digest drift")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("diagnostic_only") is not False
        or manifest.get("production_eligible") is not True
        or manifest.get("launch_authorized") is not True
    ):
        raise L1Error("manifest does not authorize the production L1 rerun")
    for field in ("source_receipt", "source_descriptor", "validation_sentinel", "f7_parent"):
        _verify_ref(manifest.get(field), field)
    lexical_python = _verify_python_binding(manifest.get("runtime_python"))
    if manifest.get("pre_training_retry_lineage") is not None:
        lineage = manifest["pre_training_retry_lineage"]
        if not isinstance(lineage, dict) or lineage.get("optimizer_steps") != 0 or lineage.get("outputs") is not None:
            raise L1Error("pre-training retry lineage is malformed")
        for field in ("claim", "submission", "stderr"):
            _verify_ref(lineage.get(field), f"retry.{field}")
    for row in manifest.get("component_bindings", []):
        _verify_ref(row.get("corpus_meta"), f"{row.get('component_id')}.corpus_meta")
        _verify_ref(row.get("validation_manifest"), f"{row.get('component_id')}.validation")
    repo_binding = manifest.get("repo_binding")
    if not isinstance(repo_binding, dict):
        raise L1Error("repo binding is malformed")
    repo = Path(str(repo_binding["repository_root"])).resolve(strict=True)
    _assert_bound_checkout(repo, str(repo_binding["public_main_commit"]))
    for relative, ref in repo_binding.get("files", {}).items():
        if _verify_ref(ref, f"source.{relative}") != (repo / relative).resolve(strict=True):
            raise L1Error(f"source binding escaped checkout: {relative}")
    command = manifest.get("command")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise L1Error("command is malformed")
    if manifest.get("command_sha256") != _digest(command):
        raise L1Error("command digest drift")
    if command[0] != lexical_python:
        raise L1Error("command does not preserve the bound lexical venv interpreter")
    inventories = manifest["event_history_training_contract"][
        "payload_inventory_acknowledgements"
    ]
    _validate_historical_recipe(_historical_projection(command, inventories))
    if _option(command, "--max-grad-norm") != "1.0":
        raise L1Error("max gradient norm is not explicitly 1.0")
    if _option(command, "--policy-aux-active-batch-size") != "0":
        raise L1Error("production L1 unexpectedly enables auxiliary policy rows")
    positions = [i for i, value in enumerate(command) if value == ACK_FLAG]
    observed = [command[i + 1] for i in positions]
    if observed != inventories or len(inventories) != 3 or command.count(CROP_FLAG) != 1:
        raise L1Error("command lacks the exact three inventory ACKs and crop flag")
    exact_paths = {
        "--data": manifest["source_descriptor"]["path"],
        "--validation-game-sentinel-manifest": manifest["validation_sentinel"]["path"],
        "--init-checkpoint": manifest["f7_parent"]["path"],
    }
    for flag, expected in exact_paths.items():
        if _option(command, flag) != expected:
            raise L1Error(f"command path drift at {flag}")
    output_root = Path(str(manifest["output_root"])).resolve(strict=False)
    if _option(command, "--checkpoint") != str(output_root / "candidate.pt") or _option(
        command, "--report"
    ) != str(output_root / "train.report.json"):
        raise L1Error("command output paths are not canonical")
    return {"manifest": manifest, "manifest_ref": manifest_ref, "repo": repo,
            "command": command, "output_root": output_root}


def _idle_b200s(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[str]:
    try:
        topology = runner(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True, text=True, capture_output=True,
        )
        compute = runner(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            check=True, text=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise L1Error(f"cannot prove B200 execution precondition: {error}") from error
    names = [row.strip() for row in topology.stdout.splitlines() if row.strip()]
    if len(names) != 8 or any("B200" not in name for name in names):
        raise L1Error(f"requires exactly eight visible B200s, found {names}")
    return [
        row.strip() for row in compute.stdout.splitlines()
        if row.strip() and "nvidia-cuda-mps" not in row.lower()
    ]


def _write_exclusive(path: Path, payload: dict[str, Any], mode: int = 0o400) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def execute(
    manifest_path: Path, *, unit: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    idle_probe: Callable[[], list[str]] = _idle_b200s,
) -> dict[str, Any]:
    if SAFE_UNIT.fullmatch(unit) is None:
        raise L1Error("systemd unit name is invalid")
    verified = verify(manifest_path)
    conflicts = idle_probe()
    if conflicts:
        raise L1Error(f"B200 compute is not idle: {conflicts}")
    root = verified["output_root"]
    root.mkdir(parents=True, exist_ok=True)
    forbidden = [
        root / "candidate.pt", root / "candidate.pt.optimizer.pt",
        root / "train.report.json", root / "execution.claim.json",
        root / "submission.receipt.json", root / "completion.receipt.json",
    ]
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise L1Error(f"production L1 is already consumed: {existing}")
    claim = {
        "schema_version": CLAIM_SCHEMA,
        "created_at_unix_ns": time.time_ns(),
        "manifest": verified["manifest_ref"],
        "unit": unit,
    }
    claim["claim_sha256"] = _digest(claim)
    claim_path = root / "execution.claim.json"
    _write_exclusive(claim_path, claim)
    stdout, stderr = root / "stdout.log", root / "stderr.log"
    systemd_command = [
        "sudo", "-n", "systemd-run", f"--unit={unit}", "--uid=ubuntu", "--gid=ubuntu",
        "--service-type=exec", "--property=LimitNOFILE=65536",
        f"--property=WorkingDirectory={verified['repo']}",
        f"--property=StandardOutput=append:{stdout}",
        f"--property=StandardError=append:{stderr}",
        "--setenv=HOME=/home/ubuntu", "--setenv=PYTHONNOUSERSITE=1",
        "--", *verified["command"],
    ]
    try:
        result = runner(systemd_command, check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise L1Error(f"systemd submission failed after one-shot claim: {error}") from error
    receipt = {
        "schema_version": SUBMISSION_SCHEMA,
        "diagnostic_only": False,
        "production_eligible": True,
        "created_at_unix_ns": time.time_ns(),
        "manifest": verified["manifest_ref"],
        "claim": {"path": str(claim_path), "sha256": _file_sha(claim_path)},
        "unit": unit,
        "command_sha256": verified["manifest"]["command_sha256"],
        "systemd_command_sha256": _digest(systemd_command),
        "systemd_stdout": result.stdout.strip(),
    }
    receipt["receipt_sha256"] = _digest(receipt)
    _write_exclusive(root / "submission.receipt.json", receipt)
    return receipt


def finalize(manifest_path: Path, *, unit: str) -> dict[str, Any]:
    verified = verify(manifest_path)
    root = verified["output_root"]
    submission_path = root / "submission.receipt.json"
    submission = _load(submission_path)
    if submission.get("schema_version") != SUBMISSION_SCHEMA or submission.get("unit") != unit:
        raise L1Error("submission receipt/unit does not match")
    try:
        state = subprocess.check_output(
            ("systemctl", "show", unit, "--property=ActiveState,Result,ExecMainStatus"),
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise L1Error(f"cannot read completed systemd state: {error}") from error
    fields = dict(row.split("=", 1) for row in state.splitlines() if "=" in row)
    if fields != {"ActiveState": "inactive", "Result": "success", "ExecMainStatus": "0"}:
        raise L1Error(f"production L1 has not completed successfully: {fields}")
    checkpoint = root / "candidate.pt"
    report = root / "train.report.json"
    checkpoint_ref, report_ref = _ref(checkpoint), _ref(report)
    report_payload = _load(report)
    if report_payload.get("init_checkpoint") != verified["manifest"]["f7_parent"]["path"]:
        raise L1Error("training report parent checkpoint drift")
    completion = {
        "schema_version": COMPLETION_SCHEMA,
        "diagnostic_only": False,
        "production_eligible": True,
        "created_at_unix_ns": time.time_ns(),
        "manifest": verified["manifest_ref"],
        "submission": {"path": str(submission_path), "sha256": _file_sha(submission_path)},
        "checkpoint": checkpoint_ref,
        "report": report_ref,
        "unit_state": fields,
    }
    completion["receipt_sha256"] = _digest(completion)
    _write_exclusive(root / "completion.receipt.json", completion)
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-receipt", required=True, type=Path)
    prep.add_argument("--repo", required=True, type=Path)
    prep.add_argument("--output-root", required=True, type=Path)
    prep.add_argument("--manifest", required=True, type=Path)
    prep.add_argument("--python", required=True, type=Path)
    prep.add_argument("--failed-attempt-root", type=Path)
    run = sub.add_parser("execute")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--unit", default="a1-production-l1-rerun")
    run.add_argument("--go", action="store_true")
    done = sub.add_parser("finalize")
    done.add_argument("--manifest", required=True, type=Path)
    done.add_argument("--unit", default="a1-production-l1-rerun")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "prepare":
            payload = prepare(
                source_receipt=args.source_receipt, repo=args.repo,
                output_root=args.output_root, manifest_path=args.manifest,
                python=args.python, failed_attempt_root=args.failed_attempt_root,
            )
            print(json.dumps({"prepared": True, "launched": False,
                              "manifest_sha256": payload["manifest_sha256"]}, sort_keys=True))
        elif args.action == "execute" and not args.go:
            payload = verify(args.manifest)
            print(json.dumps({"verified": True, "launched": False,
                              "manifest": payload["manifest_ref"]}, sort_keys=True))
        elif args.action == "execute":
            payload = execute(args.manifest, unit=args.unit)
            print(json.dumps({"submitted": True, "receipt_sha256": payload["receipt_sha256"]}, sort_keys=True))
        else:
            payload = finalize(args.manifest, unit=args.unit)
            print(json.dumps({"completed": True, "receipt_sha256": payload["receipt_sha256"]}, sort_keys=True))
        return 0
    except (L1Error, OSError, KeyError, ValueError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
