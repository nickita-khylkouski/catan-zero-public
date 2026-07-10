#!/usr/bin/env python3
"""Execute the sealed A1 one-dose learner transaction, fail closed.

This is the only production A1 training entry point.  It consumes a sealed
``a1-pre-wave-contract-lock-v2`` plus the audited memmap/validation sidecar,
replays their byte and seed bindings, and then constructs the exact single-B200
``train_bc`` invocation bound by the lock.  The default is a read-only dry run;
``--go`` is required to probe the selected B200 and start training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TOOLS_DIR = _REPO_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from tools import a1_pre_wave_contract as a1_contract  # noqa: E402
from tools import train_bc  # noqa: E402


RECEIPT_SCHEMA = "a1-one-dose-training-receipt-v1"
MIN_NOFILE = 65_536


class ExecutorError(RuntimeError):
    """A fail-closed A1 executor refusal."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _producer(lock: dict[str, Any]) -> dict[str, Any]:
    matches = [
        record
        for record in lock.get("checkpoints", [])
        if isinstance(record, dict) and record.get("role") == "producer"
    ]
    if len(matches) != 1:
        raise ExecutorError("sealed A1 contract must bind exactly one producer")
    return matches[0]


def _require_a1_science(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    science = lock.get("science")
    if not isinstance(science, dict):
        raise ExecutorError("sealed A1 contract has no science section")
    search = science.get("search_operator")
    if not isinstance(search, dict) or int(search.get("n_full", -1)) != 128:
        raise ExecutorError(
            "current A1 operator decision requires global n_full=128; "
            "n64 and global n196/n256 are not authorized"
        )
    recipe = science.get("learner_training_recipe")
    if recipe != a1_contract.EXPECTED_LEARNER_TRAINING_RECIPE:
        raise ExecutorError("sealed A1 learner recipe differs from the exact one-dose recipe")
    objective = science.get("learner_value_objective")
    if objective != {
        "objective": "mse",
        "value_readout": "scalar",
        "value_categorical_bins": None,
        "hlgauss_sigma_ratio": None,
    }:
        raise ExecutorError("current A1 one-dose executor requires scalar MSE/readout")
    if (
        recipe["world_size"] != 1
        or recipe["global_batch_size"] != 4096
        or recipe["optimizer"] != "adam"
        or recipe["resume_optimizer"] is not False
        or recipe["fused_optimizer"] is not False
        or recipe["value_lr_mult"] != 0.3
    ):
        raise ExecutorError("sealed A1 topology/optimizer invariants are not one-B200 fresh Adam")
    return recipe, objective


def verify_training_inputs(
    *, lock_path: Path, data_path: Path, validation_path: Path
) -> dict[str, Any]:
    """Replay the sealed lock and complete audit→memmap→holdout chain."""

    try:
        lock_path = lock_path.expanduser().resolve(strict=True)
        data_path = data_path.expanduser().resolve(strict=True)
        validation_path = validation_path.expanduser().resolve(strict=True)
    except OSError as error:
        raise ExecutorError(f"cannot resolve A1 training input: {error}") from error
    if not data_path.is_dir():
        raise ExecutorError(f"A1 data path is not a directory: {data_path}")

    try:
        lock = a1_contract.verify_lock(lock_path, require_all_job_claims=True)
        recipe, objective = _require_a1_science(lock)
        meta = train_bc._preflight_a1_memmap_metadata(  # noqa: SLF001
            data_path, validation_manifest_path=validation_path
        )
        if meta is None:
            raise ExecutorError("data is not an audited A1 memmap corpus")
        validation = train_bc._load_validation_game_seed_manifest_for_training(  # noqa: SLF001
            validation_path,
            validation_fraction=0.05,
            validation_seed=17,
            validation_max_samples=0,
            validation_game_seed_ranges=[],
        )
        train_bc._validate_a1_validation_manifest_corpus_binding(  # noqa: SLF001
            meta, validation
        )
        corpus = train_bc.load_teacher_data_memmap(data_path)
        bound = train_bc._validate_a1_corpus_artifacts_and_seeds(  # noqa: SLF001
            meta,
            validation,
            np.asarray(corpus["game_seed"], dtype=np.int64),
        )
    except (a1_contract.ContractError, SystemExit, OSError, ValueError) as error:
        raise ExecutorError(f"A1 training-input verification failed: {error}") from error

    contract_sha = str(lock["contract_sha256"])
    if validation["a1_contract_sha256"] != contract_sha:
        raise ExecutorError("validation sidecar binds a different A1 contract")
    if bound["learner_training_recipe"] != recipe:
        raise ExecutorError("memmap audit binds a different learner recipe")
    if bound["learner_value_objective"] != objective:
        raise ExecutorError("memmap audit binds a different learner objective")
    producer = _producer(lock)
    if bound["producer_checkpoint_sha256"] != producer.get("sha256"):
        raise ExecutorError("memmap audit producer differs from the sealed producer")

    meta_path = data_path / "corpus_meta.json"
    return {
        "lock": lock,
        "lock_path": lock_path,
        "lock_file_sha256": _file_sha256(lock_path),
        "contract_sha256": contract_sha,
        "recipe": recipe,
        "objective": objective,
        "producer": producer,
        "data_path": data_path,
        "corpus_meta_file_sha256": _file_sha256(meta_path),
        "payload_inventory_sha256": meta["payload_inventory_sha256"],
        "selected_game_seed_set_sha256": bound["selected_game_seed_set_sha256"],
        "training_game_seed_set_sha256": bound["training_game_seed_set_sha256"],
        "validation_path": validation_path,
        "validation_file_sha256": validation["file_sha256"],
        "validation_game_seed_set_sha256": validation[
            "validation_game_seed_set_sha256"
        ],
    }


def build_train_command(
    verified: dict[str, Any],
    *,
    python: Path,
    checkpoint: Path,
    report: Path,
) -> list[str]:
    """Render every effective learner field bound by the sealed recipe."""

    recipe = verified["recipe"]
    producer = verified["producer"]
    command = [
        str(python),
        str(_REPO_ROOT / "tools" / "train_bc.py"),
        "--arch",
        "entity_graph",
        "--data",
        str(verified["data_path"]),
        "--data-format",
        "memmap",
        "--device",
        "cuda",
        "--track",
        str(recipe["track"]),
        "--vps-to-win",
        str(recipe["vps_to_win"]),
        "--graph-history-features",
        "--seed",
        str(recipe["seed"]),
        "--epochs",
        str(recipe["epochs"]),
        "--max-steps",
        str(recipe["max_steps"]),
        "--batch-size",
        str(recipe["batch_size"]),
        "--grad-accum-steps",
        str(recipe["grad_accum_steps"]),
        "--optimizer",
        str(recipe["optimizer"]),
        "--no-resume-optimizer",
        "--lr",
        str(recipe["lr"]),
        "--lr-warmup-steps",
        str(recipe["lr_warmup_steps"]),
        "--lr-schedule",
        str(recipe["lr_schedule"]),
        "--weight-decay",
        str(recipe["weight_decay"]),
        "--no-fused-optimizer",
        "--value-lr-mult",
        str(recipe["value_lr_mult"]),
        "--action-module-lr-mult",
        str(recipe["action_module_lr_mult"]),
        "--policy-loss-weight",
        str(recipe["policy_loss_weight"]),
        "--soft-target-source",
        str(recipe["soft_target_source"]),
        "--soft-target-weight",
        str(recipe["soft_target_weight"]),
        "--soft-target-temperature",
        str(recipe["soft_target_temperature"]),
        "--soft-target-min-legal-coverage",
        str(recipe["soft_target_min_legal_coverage"]),
        "--value-loss-weight",
        str(recipe["value_loss_weight"]),
        "--value-target-lambda",
        str(recipe["value_target_lambda"]),
        "--value-head-type",
        "mse",
        "--value-categorical-bins",
        "0",
        "--value-categorical-loss-weight",
        str(recipe["value_categorical_loss_weight"]),
        "--hlgauss-scalar-aux-loss-weight",
        str(recipe["hlgauss_scalar_aux_loss_weight"]),
        "--final-vp-loss-weight",
        str(recipe["final_vp_loss_weight"]),
        "--q-loss-weight",
        str(recipe["q_loss_weight"]),
        "--policy-kl-anchor-weight",
        str(recipe["policy_kl_anchor_weight"]),
        "--value-uncertainty-loss-weight",
        str(recipe["value_uncertainty_loss_weight"]),
        "--aux-subgoal-loss-weight",
        str(recipe["aux_subgoal_loss_weight"]),
        "--freeze-modules",
        str(recipe["freeze_modules"]),
        "--policy-surprise-weight",
        str(recipe["policy_surprise_weight"]),
        "--advantage-policy-weighting",
        str(recipe["advantage_policy_weighting"]),
        "--vp-margin-weight",
        str(recipe["vp_margin_weight"]),
        "--truncated-vp-margin-value-weight",
        str(recipe["truncated_vp_margin_value_weight"]),
        "--amp",
        str(recipe["amp"]),
        "--mask-hidden-info",
        "--no-symmetry-augment",
        "--forced-action-weight",
        str(recipe["forced_action_weight"]),
        "--forced-row-value-weight",
        str(recipe["forced_row_value_weight"]),
        "--winner-sample-weight",
        str(recipe["winner_sample_weight"]),
        "--loser-sample-weight",
        str(recipe["loser_sample_weight"]),
        "--teacher-weights",
        str(recipe["teacher_weights"]),
        "--phase-weights",
        str(recipe["phase_weights"]),
        "--value-phase-weights",
        str(recipe["value_phase_weights"]),
        "--validation-fraction",
        "0.05",
        "--validation-seed",
        "17",
        "--validation-max-samples",
        "0",
        "--validation-game-seed-manifest",
        str(verified["validation_path"]),
        "--init-checkpoint",
        str(producer["path"]),
        "--checkpoint",
        str(checkpoint),
        "--report",
        str(report),
        "--require-35m-model",
        "--skip-teacher-quality-gate",
        "--trust-curated-data-quality",
    ]
    return command


def _probe_b200(gpu: int) -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExecutorError(f"cannot verify selected B200 GPU {gpu}: {error}") from error
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(names) != 1 or "B200" not in names[0].upper():
        raise ExecutorError(f"selected GPU {gpu} is not exactly one B200: {names}")
    return names[0]


def _raise_nofile_limit() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(max(soft, MIN_NOFILE), hard)
    if target < MIN_NOFILE:
        raise ExecutorError(
            f"hard RLIMIT_NOFILE={hard} is below required {MIN_NOFILE}"
        )
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))


def _claim_path(receipt: Path) -> Path:
    return receipt.with_name(receipt.name + ".claim")


def _claim_attempt(receipt: Path, payload: dict[str, Any]) -> Path:
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if receipt.exists():
        raise ExecutorError(f"receipt already exists; A1 dose is already consumed: {receipt}")
    claim = _claim_path(receipt)
    try:
        fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise ExecutorError(f"A1 training claim already exists: {claim}") from error
    with os.fdopen(fd, "wb") as handle:
        handle.write(_canonical_bytes(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return claim


def _write_receipt_no_clobber(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["receipt_sha256"] = _value_sha256(payload)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("xb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
    except FileExistsError as error:
        raise ExecutorError(f"refusing to overwrite A1 receipt: {path}") from error
    finally:
        tmp.unlink(missing_ok=True)


def _verify_training_outputs(
    *, checkpoint: Path, report: Path, verified: dict[str, Any]
) -> dict[str, Any]:
    optimizer = Path(str(checkpoint) + ".optimizer.pt")
    for path in (checkpoint, optimizer, report):
        if not path.is_file() or path.stat().st_size <= 0:
            raise ExecutorError(f"A1 training output is missing or empty: {path}")
    try:
        report_payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(f"cannot parse A1 training report: {error}") from error
    expected = {
        "a1_contract_sha256": verified["contract_sha256"],
        "world_size": 1,
        "optimizer": "adam",
        "resume_optimizer": False,
        "optimizer_restored": False,
        "fused_optimizer": False,
        "epochs": 1,
    }
    drift = {
        key: {"expected": value, "actual": report_payload.get(key)}
        for key, value in expected.items()
        if report_payload.get(key) != value
    }
    if drift:
        raise ExecutorError(f"A1 training report invariant drift: {drift}")
    if report_payload.get("a1_bound_learner_training_recipe") != verified["recipe"]:
        raise ExecutorError("A1 training report does not echo the exact sealed recipe")
    if int(report_payload.get("steps_completed", 0)) <= 0:
        raise ExecutorError("A1 training report records no optimizer steps")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "optimizer_sidecar": str(optimizer),
        "optimizer_sidecar_sha256": _file_sha256(optimizer),
        "report": str(report),
        "report_sha256": _file_sha256(report),
        "steps_completed": int(report_payload["steps_completed"]),
    }


def _require_fresh_outputs(checkpoint: Path, report: Path, receipt: Path) -> None:
    paths = (checkpoint, Path(str(checkpoint) + ".optimizer.pt"), report, receipt)
    if len(set(paths)) != len(paths):
        raise ExecutorError(
            "checkpoint, optimizer sidecar, report, and receipt paths must be distinct"
        )
    for path in paths:
        if path.exists():
            raise ExecutorError(f"refusing non-fresh A1 output path: {path}")


def execute(
    *,
    verified: dict[str, Any],
    command: list[str],
    checkpoint: Path,
    report: Path,
    receipt: Path,
    gpu: int,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    probe: Callable[[int], str] = _probe_b200,
) -> dict[str, Any]:
    """Claim, execute, verify, and atomically receipt exactly one A1 dose."""

    _require_fresh_outputs(checkpoint, report, receipt)
    started_ns = time.time_ns()
    claim_payload = {
        "schema_version": "a1-one-dose-training-claim-v1",
        "contract_sha256": verified["contract_sha256"],
        "command_sha256": _value_sha256(command),
        "started_unix_ns": started_ns,
    }
    claim = _claim_attempt(receipt, claim_payload)
    status = "failed"
    returncode: int | None = None
    output_artifacts: dict[str, Any] | None = None
    failure: str | None = None
    gpu_name = ""
    try:
        gpu_name = probe(gpu)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        if env.get("WORLD_SIZE", "") not in {"", "1"}:
            raise ExecutorError(
                f"distributed environment is not world_size=1: "
                f"WORLD_SIZE={env['WORLD_SIZE']}"
            )
        for key in ("RANK", "LOCAL_RANK"):
            if env.get(key, "") not in {"", "0"}:
                raise ExecutorError(
                    f"distributed environment is not rank zero: {key}={env[key]}"
                )
        for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
            env.pop(key, None)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        result = runner(
            command,
            cwd=str(_REPO_ROOT),
            env=env,
            check=False,
            preexec_fn=_raise_nofile_limit,
        )
        returncode = int(result.returncode)
        if returncode != 0:
            raise ExecutorError(f"train_bc exited nonzero: {returncode}")
        output_artifacts = _verify_training_outputs(
            checkpoint=checkpoint, report=report, verified=verified
        )
        status = "complete"
    except Exception as error:  # receipt every claimed attempt, then re-raise.
        failure = f"{type(error).__name__}: {error}"
    finished_ns = time.time_ns()
    receipt_payload = {
        "schema_version": RECEIPT_SCHEMA,
        "status": status,
        "contract_sha256": verified["contract_sha256"],
        "lock": str(verified["lock_path"]),
        "lock_file_sha256": verified["lock_file_sha256"],
        "corpus": str(verified["data_path"]),
        "corpus_meta_file_sha256": verified["corpus_meta_file_sha256"],
        "payload_inventory_sha256": verified["payload_inventory_sha256"],
        "validation_manifest": str(verified["validation_path"]),
        "validation_manifest_file_sha256": verified["validation_file_sha256"],
        "producer_checkpoint_sha256": verified["producer"]["sha256"],
        "learner_training_recipe_sha256": _value_sha256(verified["recipe"]),
        "command": command,
        "command_sha256": _value_sha256(command),
        "world_size": 1,
        "gpu": gpu,
        "gpu_name": gpu_name,
        "started_unix_ns": started_ns,
        "finished_unix_ns": finished_ns,
        "returncode": returncode,
        "outputs": output_artifacts,
        "failure": failure,
    }
    try:
        _write_receipt_no_clobber(receipt, receipt_payload)
    finally:
        claim.unlink(missing_ok=True)
    if status != "complete":
        raise ExecutorError(failure or "A1 training failed")
    return receipt_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu", type=int, default=0, help="one physical B200 index")
    parser.add_argument(
        "--go", action="store_true", help="execute locally; default is verified dry-run"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        python = args.python.expanduser().resolve(strict=True)
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ExecutorError(f"python is not executable: {python}")
        if args.gpu < 0:
            raise ExecutorError("--gpu must be non-negative")
        verified = verify_training_inputs(
            lock_path=args.lock,
            data_path=args.data,
            validation_path=args.validation_manifest,
        )
        checkpoint = args.checkpoint.expanduser().resolve(strict=False)
        report = args.report.expanduser().resolve(strict=False)
        receipt = args.receipt.expanduser().resolve(strict=False)
        _require_fresh_outputs(checkpoint, report, receipt)
        command = build_train_command(
            verified,
            python=python,
            checkpoint=checkpoint,
            report=report,
        )
        plan = {
            "schema_version": "a1-one-dose-training-plan-v1",
            "mode": "go" if args.go else "dry-run",
            "contract_sha256": verified["contract_sha256"],
            "global_n_full": 128,
            "world_size": 1,
            "gpu": args.gpu,
            "command": command,
            "command_sha256": _value_sha256(command),
            "checkpoint": str(checkpoint),
            "report": str(report),
            "receipt": str(receipt),
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        if not args.go:
            return 0
        execute(
            verified=verified,
            command=command,
            checkpoint=checkpoint,
            report=report,
            receipt=receipt,
            gpu=args.gpu,
        )
        return 0
    except (ExecutorError, OSError) as error:
        parser.exit(2, f"REFUSED: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
