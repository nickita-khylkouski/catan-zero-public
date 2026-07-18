#!/usr/bin/env python3
"""Launch the authorized parent update on one eight-GPU H100/B200 host.

This is intentionally a thin process-topology adapter.  The checked-in recipe
owns all model, optimizer, sampler, and loss settings; ``tools/train.py`` owns
the canonical training adapter; and ``train_bc.py`` remains the engine.  This
entrypoint only proves the selected production recipe is authorized, verifies
that the host owns eight homogeneous H100s or B200s, and replaces itself with
torchrun.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _root in (_REPO_ROOT, _REPO_ROOT / "src"):
    while str(_root) in sys.path:
        sys.path.remove(str(_root))
    sys.path.insert(0, str(_root))

from tools import a1_current_science_contract as current_science  # noqa: E402


WORLD_SIZE = 8


def _input_file(raw: str, *, label: str) -> Path:
    source = Path(raw).expanduser()
    if source.is_symlink():
        raise SystemExit(f"{label} must not be a symlink: {source}")
    try:
        value = source.resolve(strict=True)
    except OSError as error:
        raise SystemExit(f"{label} cannot be resolved: {error}") from error
    if value.is_symlink() or not value.is_file():
        raise SystemExit(f"{label} must be a regular non-symlink file: {value}")
    return value


def _training_data(raw: str) -> Path:
    """Accept the trainer's two authenticated data surfaces.

    A composite is one regular descriptor file. A direct memmap corpus is the
    directory containing ``corpus_meta.json``; passing that metadata file alone
    loses the payload root and is therefore deliberately not rewritten here.
    """

    source = Path(raw).expanduser()
    if source.is_symlink():
        raise SystemExit(f"training data must not be a symlink: {source}")
    try:
        value = source.resolve(strict=True)
    except OSError as error:
        raise SystemExit(f"training data cannot be resolved: {error}") from error
    if value.is_file():
        return value
    if not value.is_dir():
        raise SystemExit(
            "training data must be a composite descriptor file or memmap directory: "
            f"{value}"
        )
    metadata = value / "corpus_meta.json"
    if metadata.is_symlink() or not metadata.is_file():
        raise SystemExit(
            "memmap training directory requires regular corpus_meta.json: "
            f"{metadata}"
        )
    return value


def _python_executable(raw: str) -> Path:
    try:
        value = Path(raw).expanduser().resolve(strict=True)
    except OSError as error:
        raise SystemExit(f"Python executable cannot be resolved: {error}") from error
    if not value.is_file() or not os.access(value, os.X_OK):
        raise SystemExit(f"Python executable is not executable: {value}")
    return value


def _output_file(raw: str, *, label: str) -> Path:
    value = Path(raw).expanduser().resolve(strict=False)
    value.parent.mkdir(parents=True, exist_ok=True)
    if value.exists():
        raise SystemExit(f"{label} already exists: {value}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_eight_training_gpus() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            "cannot inventory GPUs with nvidia-smi: "
            + (result.stderr.strip() or f"exit {result.returncode}")
        )
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    supported = all(
        "H100" in name.upper() or "B200" in name.upper() for name in names
    )
    if len(names) != WORLD_SIZE or not supported or len(set(names)) != 1:
        raise SystemExit(
            "authorized parent update requires exactly 8 homogeneous H100s or "
            f"B200s; got {names}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--information-contract-migration-receipt", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--master-port", type=int, default=29500)
    return parser


def command_from_args(args: argparse.Namespace) -> list[str]:
    config = current_science.require_selected_parent_update(
        current_science.CANONICAL_PARENT_UPDATE_CONFIG_PATH
    )
    current_science.require_selected_parent_update_go_authorized()
    data = _training_data(args.data)
    parent = _input_file(args.parent_checkpoint, label="parent checkpoint")
    initializer = _input_file(args.init_checkpoint, label="initializer checkpoint")
    migration = (
        _input_file(
            args.information_contract_migration_receipt,
            label="information-contract migration receipt",
        )
        if args.information_contract_migration_receipt
        else None
    )
    same_initializer = _sha256(parent) == _sha256(initializer)
    if same_initializer and migration is not None:
        raise SystemExit(
            "identical parent/initializer bytes must not claim a migration receipt"
        )
    if not same_initializer and migration is None:
        raise SystemExit(
            "different parent/initializer bytes require "
            "--information-contract-migration-receipt"
        )
    checkpoint = _output_file(args.checkpoint, label="candidate checkpoint")
    report = _output_file(args.report, label="training report")
    python = _python_executable(args.python)
    command = [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={WORLD_SIZE}",
        f"--master-port={args.master_port}",
        str((_REPO_ROOT / "tools/train.py").resolve(strict=True)),
        "--config",
        str(config),
        "--data",
        str(data),
        "--parent-checkpoint",
        str(parent),
        "--init-checkpoint",
        str(initializer),
        "--checkpoint",
        str(checkpoint),
        "--report",
        str(report),
        "--device",
        "cuda",
    ]
    if migration is not None:
        command.extend(
            ["--information-contract-migration-receipt", str(migration)]
        )
    return command


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _require_eight_training_gpus()
    command = command_from_args(args)
    print("+ " + shlex.join(command), flush=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in range(8))
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
