#!/usr/bin/env python3
"""Run the two reviewed experimental A1 learners sequentially on one 8xB200 host.

This tool is intentionally operational only.  It cannot render or approve a
learner spec and it cannot seal a learner lock.  Each arm must already have an
independently reviewed lock and its explicit file SHA-256.  The default is a
read-only plan.  ``--go`` stops the host MPS service, runs n128 and then n256
through the sealed dual-arm executor, and restores MPS in a ``finally`` block.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Sequence


SCHEMA = "a1-experimental-training-sequence-v1"
READY_MARKER = "training_inputs.ready"
ARMS = ("n128", "n256")
MPS_UNIT = "nvidia-mps.service"


class SequenceError(RuntimeError):
    """A fail-closed experimental training-sequence refusal."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file(path: str | Path, *, where: str) -> Path:
    try:
        value = Path(path).expanduser().resolve(strict=True)
    except OSError as error:
        raise SequenceError(f"cannot resolve {where}: {error}") from error
    if not value.is_file():
        raise SequenceError(f"{where} is not a file: {value}")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    path = _file(path, where="sequence config")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SequenceError(f"cannot load sequence config: {error}") from error
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "root", "python", "executor", "mps_unit", "arms"
    }:
        raise SequenceError("sequence config fields drift")
    if (
        value.get("schema_version") != SCHEMA
        or value.get("mps_unit") != MPS_UNIT
        or not isinstance(value.get("arms"), list)
        or [row.get("arm_id") for row in value["arms"] if isinstance(row, dict)]
        != list(ARMS)
    ):
        raise SequenceError("sequence config schema/order drift")
    return value


def build_plan(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    root = Path(config["root"]).expanduser().resolve(strict=True)
    marker = _file(root / READY_MARKER, where="training input readiness marker")
    python = _file(config["python"], where="learner python")
    executor = _file(config["executor"], where="dual-arm executor")
    commands: list[dict[str, Any]] = []
    expected_fields = {
        "arm_id", "data", "validation_manifest", "producer_checkpoint",
        "learner_lock", "reviewed_lock_file_sha256", "checkpoint", "report",
        "receipt",
    }
    for row in config["arms"]:
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise SequenceError("arm config fields drift")
        learner_lock = _file(row["learner_lock"], where=f"{row['arm_id']} learner lock")
        reviewed = row["reviewed_lock_file_sha256"]
        if not isinstance(reviewed, str) or reviewed != _sha256(learner_lock):
            raise SequenceError(
                f"{row['arm_id']} learner lock differs from independently reviewed SHA-256"
            )
        data = Path(row["data"]).expanduser().resolve(strict=True)
        if not data.is_dir():
            raise SequenceError(f"{row['arm_id']} memmap is not a directory: {data}")
        validation = _file(
            row["validation_manifest"], where=f"{row['arm_id']} validation manifest"
        )
        producer = _file(
            row["producer_checkpoint"], where=f"{row['arm_id']} producer checkpoint"
        )
        outputs = {
            key: str(Path(row[key]).expanduser().resolve(strict=False))
            for key in ("checkpoint", "report", "receipt")
        }
        if len(set(outputs.values())) != 3:
            raise SequenceError(f"{row['arm_id']} output paths alias")
        argv = [
            str(python), str(executor), "--data", str(data),
            "--learner-lock", str(learner_lock),
            "--reviewed-lock-file-sha256", reviewed,
            "--validation-manifest", str(validation),
            "--producer-checkpoint", str(producer),
            "--checkpoint", outputs["checkpoint"],
            "--report", outputs["report"], "--receipt", outputs["receipt"],
            "--python", str(python),
        ]
        commands.append(
            {
                "arm_id": row["arm_id"],
                "learner_lock_sha256": reviewed,
                "argv": argv,
                "outputs": outputs,
            }
        )
    all_outputs = [path for command in commands for path in command["outputs"].values()]
    if len(set(all_outputs)) != len(all_outputs):
        raise SequenceError("output paths alias across arms")
    return {
        "schema_version": SCHEMA,
        "config": {"path": str(config_path.resolve()), "sha256": _sha256(config_path)},
        "root": str(root),
        "ready_marker": {"path": str(marker), "sha256": _sha256(marker)},
        "mps_unit": MPS_UNIT,
        "execution_order": list(ARMS),
        "commands": commands,
    }


def _service_state(
    runner: Callable[..., subprocess.CompletedProcess[str]], unit: str
) -> str:
    result = runner(
        ["systemctl", "is-active", unit], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    return result.stdout.strip()


@contextmanager
def _sequence_lock(root: Path):
    """Exclude a second orchestration across the gap between the two arms."""
    path = root / ".training-sequence.lock"
    descriptor = path.open("a+b")
    try:
        try:
            fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SequenceError("another experimental training sequence is active") from error
        yield
    finally:
        fcntl.flock(descriptor.fileno(), fcntl.LOCK_UN)
        descriptor.close()


def execute(
    plan: dict[str, Any],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    with _sequence_lock(Path(plan["root"])):
        if _service_state(runner, plan["mps_unit"]) != "active":
            raise SequenceError("MPS service must be active before sequence ownership")
        stopped = False
        primary: BaseException | None = None
        try:
            result = runner(
                ["sudo", "-n", "systemctl", "stop", plan["mps_unit"]], check=False
            )
            if result.returncode != 0 or _service_state(runner, plan["mps_unit"]) != "inactive":
                raise SequenceError("could not stop MPS service cleanly")
            stopped = True
            for command in plan["commands"]:
                # First replay the complete sealed preflight without touching GPUs.
                dry = runner(command["argv"], check=False)
                if dry.returncode != 0:
                    raise SequenceError(f"{command['arm_id']} sealed dry-run refused")
                go = runner([*command["argv"], "--go"], check=False)
                if go.returncode != 0:
                    raise SequenceError(f"{command['arm_id']} sealed training failed")
                receipt = _file(
                    command["outputs"]["receipt"],
                    where=f"{command['arm_id']} completed receipt",
                )
                if receipt.stat().st_size <= 0:
                    raise SequenceError(f"{command['arm_id']} receipt is empty")
        except BaseException as error:
            primary = error
            raise
        finally:
            if stopped:
                restored = runner(
                    ["sudo", "-n", "systemctl", "start", plan["mps_unit"]],
                    check=False,
                )
                state = _service_state(runner, plan["mps_unit"])
                if restored.returncode != 0 or state != "active":
                    message = "failed to restore MPS service after learner sequence"
                    if primary is None:
                        raise SequenceError(message)
                    primary.add_note(message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--go", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.config)
        print(json.dumps({**plan, "mode": "go" if args.go else "dry-run"}, indent=2))
        if args.go:
            execute(plan)
        return 0
    except (SequenceError, OSError) as error:
        parser.exit(2, f"REFUSED: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
