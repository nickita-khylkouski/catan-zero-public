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
import math
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


RECEIPT_SCHEMA = "a1-one-dose-training-receipt-v2"
CLAIM_SCHEMA = "a1-one-dose-training-claim-v2"
CLAIM_DIRECTORY = ".a1-one-dose-training-claims"
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


def _fsync_directory(path: Path) -> None:
    """Durably publish directory-entry changes or fail closed."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir_durable(path: Path) -> None:
    """Create ``path`` and sync every newly-created directory entry."""

    path = Path(path)
    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ExecutorError(f"expected directory, found non-directory: {path}")
    for created in reversed(missing):
        _fsync_directory(created)
        _fsync_directory(created.parent)


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _with_digest(payload: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = _value_sha256(result)
    return result


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
    corpus_row_count = int(meta["row_count"])
    validation_row_count = int(validation["validation_row_count"])
    training_row_count = corpus_row_count - validation_row_count
    if training_row_count <= 0:
        raise ExecutorError("audited A1 corpus has no training rows")
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
        "data_fingerprint": train_bc._training_data_fingerprint(  # noqa: SLF001
            str(data_path), "memmap"
        ),
        "corpus_row_count": corpus_row_count,
        "training_row_count": training_row_count,
        "validation_row_count": validation_row_count,
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


def _claim_path(verified: dict[str, Any]) -> Path:
    """Return the one stable claim path for the sealed contract identity.

    The sealed seed-ledger path is the shared, contract-bound anchor. A caller
    cannot obtain a second dose by choosing another receipt or copying the lock.
    """

    contract_sha = str(verified.get("contract_sha256", ""))
    prefix = "sha256:"
    digest = contract_sha.removeprefix(prefix)
    if not contract_sha.startswith(prefix) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ExecutorError(f"invalid sealed contract identity: {contract_sha!r}")
    try:
        ledger_value = verified["lock"]["fleet"]["seed_ledger"]["path"]
        ledger = Path(str(ledger_value)).expanduser().resolve(strict=True)
    except (KeyError, TypeError, OSError) as error:
        raise ExecutorError(
            "sealed contract has no resolvable seed-ledger claim anchor"
        ) from error
    if not ledger.is_file():
        raise ExecutorError(f"sealed seed-ledger anchor is not a file: {ledger}")
    return ledger.parent / CLAIM_DIRECTORY / f"{digest}.json"


def _load_claim_state(claim: Path, *, contract_sha256: str) -> dict[str, Any]:
    try:
        payload = json.loads(claim.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(f"A1 contract claim is unreadable/corrupt: {claim}") from error
    if not isinstance(payload, dict):
        raise ExecutorError(f"A1 contract claim is not an object: {claim}")
    stated_digest = payload.get("state_sha256")
    unhashed = dict(payload)
    unhashed.pop("state_sha256", None)
    if stated_digest != _value_sha256(unhashed):
        raise ExecutorError(f"A1 contract claim digest is invalid: {claim}")
    if payload.get("schema_version") != CLAIM_SCHEMA:
        raise ExecutorError(f"A1 contract claim schema is invalid: {claim}")
    if payload.get("contract_sha256") != contract_sha256:
        raise ExecutorError(f"A1 contract claim identity mismatch: {claim}")
    return payload


def _require_unconsumed_contract(verified: dict[str, Any]) -> None:
    claim = _claim_path(verified)
    if claim.exists():
        state = _load_claim_state(
            claim, contract_sha256=str(verified["contract_sha256"])
        )
        raise ExecutorError(
            "sealed A1 dose already has a durable claim: "
            f"status={state.get('status')!r} path={claim}"
        )


def _claim_attempt(verified: dict[str, Any], payload: dict[str, Any]) -> Path:
    claim = _claim_path(verified)
    _mkdir_durable(claim.parent)
    try:
        fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise ExecutorError(f"A1 training claim already exists: {claim}") from error
    durable_payload = _with_digest(payload, "state_sha256")
    with os.fdopen(fd, "wb") as handle:
        handle.write(_canonical_bytes(durable_payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(claim.parent)
    return claim


def _write_terminal_claim(
    claim: Path, payload: dict[str, Any], *, contract_sha256: str
) -> dict[str, Any]:
    current = _load_claim_state(claim, contract_sha256=contract_sha256)
    if current.get("status") != "claimed":
        raise ExecutorError(
            f"A1 claim is already terminal: status={current.get('status')!r} path={claim}"
        )
    terminal = _with_digest(payload, "state_sha256")
    tmp = claim.with_name(f".{claim.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical_bytes(terminal) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, claim)
        _fsync_directory(claim.parent)
    finally:
        tmp.unlink(missing_ok=True)
    return terminal


def _write_receipt_no_clobber(
    path: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    _mkdir_durable(path.parent)
    payload = _with_digest(payload, "receipt_sha256")
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("xb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
        _fsync_directory(path.parent)
    except FileExistsError as error:
        raise ExecutorError(f"refusing to overwrite A1 receipt: {path}") from error
    finally:
        tmp.unlink(missing_ok=True)
        _fsync_directory(path.parent)
    return payload


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
    recipe = verified["recipe"]
    expected_steps = int(
        math.ceil(
            int(verified["training_row_count"])
            / (int(recipe["batch_size"]) * int(recipe["grad_accum_steps"]))
        )
    )
    if int(recipe["max_steps"]) > 0:
        expected_steps = min(expected_steps, int(recipe["max_steps"]))
    expected = {
        "arch": "entity_graph",
        "a1_contract_sha256": verified["contract_sha256"],
        "world_size": 1,
        "optimizer": "adam",
        "resume_optimizer": False,
        "optimizer_restored": False,
        "fused_optimizer": False,
        "epochs": 1,
        "max_steps": 0,
        "batch_size": int(recipe["batch_size"]),
        "amp": recipe["amp"],
        "lr": float(recipe["lr"]),
        "weight_decay": float(recipe["weight_decay"]),
        "seed": int(recipe["seed"]),
        "mask_hidden_info": True,
        "symmetry_augment": False,
        "data": str(verified["data_path"]),
        "data_format": "memmap",
        "data_fingerprint": verified["data_fingerprint"],
        "samples": int(verified["corpus_row_count"]),
        "global_samples": int(verified["corpus_row_count"]),
        "train_samples": int(verified["training_row_count"]),
        "validation_samples": int(verified["validation_row_count"]),
        "track": recipe["track"],
        "vps_to_win": int(recipe["vps_to_win"]),
        "checkpoint": str(checkpoint),
        "init_checkpoint": str(verified["producer"]["path"]),
        "init_checkpoint_sha256": verified["producer"]["sha256"],
        "input_validation_game_seed_manifest": str(verified["validation_path"]),
        "input_validation_game_seed_manifest_sha256": verified[
            "validation_file_sha256"
        ],
        "validation_game_seed_set_sha256": verified[
            "validation_game_seed_set_sha256"
        ],
        "a1_selected_game_seed_set_sha256": verified[
            "selected_game_seed_set_sha256"
        ],
        "a1_training_game_seed_set_sha256": verified[
            "training_game_seed_set_sha256"
        ],
        "a1_memmap_payload_inventory_sha256": verified[
            "payload_inventory_sha256"
        ],
        "a1_learner_training_recipe_sha256": _value_sha256(recipe),
        "require_35m_model": True,
        "steps_completed": expected_steps,
        "total_training_steps": expected_steps,
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
    if report_payload.get("a1_bound_learner_value_objective") != verified["objective"]:
        raise ExecutorError("A1 training report does not echo the sealed value objective")
    metrics = report_payload.get("metrics")
    if (
        not isinstance(metrics, list)
        or len(metrics) != 1
        or not isinstance(metrics[0], dict)
        or metrics[0].get("epoch") != 1
    ):
        raise ExecutorError("A1 training report does not prove exactly one completed epoch")
    for key in ("loss", "policy_loss", "value_loss"):
        value = metrics[0].get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ExecutorError(f"A1 training report has invalid epoch metric {key!r}")
    validation_metrics = metrics[0].get("validation")
    if not isinstance(validation_metrics, dict):
        raise ExecutorError("A1 training report has no one-epoch validation metrics")
    validation_loss = validation_metrics.get("loss")
    if (
        validation_metrics.get("samples") != int(verified["validation_row_count"])
        or isinstance(validation_loss, bool)
        or not isinstance(validation_loss, (int, float))
        or not math.isfinite(float(validation_loss))
    ):
        raise ExecutorError("A1 training report has invalid validation coverage/metrics")
    parameter_count = report_payload.get("parameter_count")
    if (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or not 30_000_000 <= parameter_count <= 40_000_000
    ):
        raise ExecutorError("A1 training report does not prove the required 35M model")
    value_training = report_payload.get("value_training")
    expected_value_training = {
        "primary_readout": "scalar",
        "optimizer_steps": expected_steps,
        "completed_epochs": 1,
        "a1_contract_sha256": verified["contract_sha256"],
        "a1_selected_game_seed_set_sha256": verified[
            "selected_game_seed_set_sha256"
        ],
        "a1_training_game_seed_set_sha256": verified[
            "training_game_seed_set_sha256"
        ],
        "a1_learner_training_recipe_sha256": _value_sha256(recipe),
        "a1_memmap_payload_inventory_sha256": verified[
            "payload_inventory_sha256"
        ],
    }
    if not isinstance(value_training, dict) or any(
        value_training.get(key) != value
        for key, value in expected_value_training.items()
    ):
        raise ExecutorError("A1 training report value-training provenance drift")
    if "scalar" not in value_training.get("trained_value_readouts", []):
        raise ExecutorError("A1 candidate does not attest a trained scalar value readout")
    for path in (checkpoint, optimizer, report):
        _fsync_file(path)
    for parent in {checkpoint.parent, optimizer.parent, report.parent}:
        _fsync_directory(parent)
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "optimizer_sidecar": str(optimizer),
        "optimizer_sidecar_sha256": _file_sha256(optimizer),
        "report": str(report),
        "report_sha256": _file_sha256(report),
        "steps_completed": expected_steps,
        "corpus_row_count": int(verified["corpus_row_count"]),
        "training_row_count": int(verified["training_row_count"]),
        "validation_row_count": int(verified["validation_row_count"]),
    }


def _require_fresh_outputs(
    checkpoint: Path,
    report: Path,
    receipt: Path,
    *,
    claim: Path | None = None,
) -> None:
    paths = (checkpoint, Path(str(checkpoint) + ".optimizer.pt"), report, receipt)
    if len(set(paths)) != len(paths):
        raise ExecutorError(
            "checkpoint, optimizer sidecar, report, and receipt paths must be distinct"
        )
    if claim is not None and claim in paths:
        raise ExecutorError(
            "checkpoint, optimizer sidecar, report, and receipt must be distinct "
            f"from the sealed-contract claim path: {claim}"
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

    claim = _claim_path(verified)
    _require_fresh_outputs(checkpoint, report, receipt, claim=claim)
    started_ns = time.time_ns()
    claim_payload = {
        "schema_version": CLAIM_SCHEMA,
        "status": "claimed",
        "contract_sha256": verified["contract_sha256"],
        "command_sha256": _value_sha256(command),
        "started_unix_ns": started_ns,
    }
    claim = _claim_attempt(verified, claim_payload)
    status = "failed"
    returncode: int | None = None
    output_artifacts: dict[str, Any] | None = None
    failure: str | None = None
    gpu_name = ""
    try:
        gpu_name = probe(gpu)
        _mkdir_durable(checkpoint.parent)
        _mkdir_durable(report.parent)
        _mkdir_durable(receipt.parent)
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
    evidence_payload = {
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
    terminal_claim_payload = dict(evidence_payload)
    terminal_claim_payload["schema_version"] = CLAIM_SCHEMA
    terminal_claim_payload["receipt_target"] = str(receipt)
    terminal_claim = _write_terminal_claim(
        claim,
        terminal_claim_payload,
        contract_sha256=str(verified["contract_sha256"]),
    )
    evidence_payload["claim"] = str(claim)
    evidence_payload["claim_state_sha256"] = terminal_claim["state_sha256"]
    receipt_payload = _write_receipt_no_clobber(receipt, evidence_payload)
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
        claim = _claim_path(verified)
        _require_fresh_outputs(checkpoint, report, receipt, claim=claim)
        _require_unconsumed_contract(verified)
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
