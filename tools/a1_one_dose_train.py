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
import pwd
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


RECEIPT_SCHEMA = "a1-one-dose-training-receipt-v3"
CLAIM_SCHEMA = "a1-one-dose-training-claim-v3"
RETRY_RECEIPT_SCHEMA = "a1-one-dose-training-receipt-v4"
RETRY_CLAIM_SCHEMA = "a1-one-dose-training-claim-v4"
PLAN_SCHEMA = "a1-one-dose-training-plan-v2"
REPORT_EXECUTION_BINDING_SCHEMA = "a1-one-dose-execution-binding-v1"
REPORT_EXECUTION_BINDING_FIELD = "a1_one_dose_execution_binding"
RETRY_CONTRACT_SCHEMA = "a1-one-dose-learner-retry-contract-v1"
RETRY_IDENTITY_SCHEMA = "a1-one-dose-learner-retry-identity-v1"
RETRY_REPAIR_KIND = "entity_graph_graph_layers_default_4_to_checkpoint_6"
CLAIM_DIRECTORY = ".a1-one-dose-training-claims"
MIN_NOFILE = 65_536
MAX_IDLE_GPU_MEMORY_MIB = 64
DATA_LOADER_WORKERS = 2
DATA_LOADER_PREFETCH = 2
SEALED_A1_MODEL_CLI: dict[str, str] = {
    "--hidden-size": "640",
    "--graph-layers": "6",
    "--attention-heads": "8",
    "--graph-dropout": "0.05",
    "--entity-state-trunk": "transformer",
    "--relational-block-pattern": "",
    "--relational-ff-size": "0",
    "--relational-bases": "4",
    "--relational-action-cross-layers": "1",
    "--latent-deliberation-steps": "0",
    "--latent-deliberation-slots": "8",
    "--moe-routed-experts": "0",
    "--moe-top-k": "2",
    "--moe-expert-ff-size": "0",
}
SEALED_A1_MODEL_REPORT: dict[str, int | float] = {
    "hidden_size": 640,
    "graph_layers": 6,
    "attention_heads": 8,
    "graph_dropout": 0.05,
}
ACCEPTABLE_COMPUTE_MODES = frozenset({"DEFAULT", "EXCLUSIVE_PROCESS"})
MPS_EXECUTABLES = frozenset({"nvidia-cuda-mps-control", "nvidia-cuda-mps-server"})
CHILD_ENVIRONMENT_KEYS = frozenset(
    {
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "PYTHONHASHSEED",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "TMPDIR",
        "TZ",
    }
)


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


def _lexical_python_executable(path: Path) -> Path:
    """Validate an interpreter without resolving away its virtual environment.

    ``venv/bin/python`` is normally a symlink to the base interpreter.  Invoking
    the resolved target bypasses the adjacent ``pyvenv.cfg`` and silently drops
    the venv's Torch/dependencies.  Preserve the absolute lexical path for argv
    while still proving that its target is a real executable file.
    """

    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        target = lexical.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(f"cannot resolve learner python: {error}") from error
    if (
        not lexical.is_file()
        or not target.is_file()
        or not os.access(lexical, os.X_OK)
        or not os.access(target, os.X_OK)
    ):
        raise ExecutorError(f"python is not executable: {lexical}")
    return lexical


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


def _child_environment(gpu: int) -> dict[str, str]:
    """Return the complete, secret-free environment for the learner child.

    Do not start from ``os.environ``: an operator shell may contain distributed,
    CUDA, Python, proxy, credential, or preload variables that silently change
    the one-dose process.  HOME comes from the operating-system account record,
    not the ambient HOME variable, and every other entry is an explicit value.
    """

    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ExecutorError("child environment GPU must be a non-negative integer")
    try:
        account_home = pwd.getpwuid(os.getuid()).pw_dir
    except (KeyError, OSError) as error:
        raise ExecutorError("cannot resolve the learner account home") from error
    environment = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "HOME": str(account_home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": f"{_REPO_ROOT / 'src'}:{_REPO_ROOT}",
        "TMPDIR": "/tmp",
        "TZ": "UTC",
    }
    if set(environment) != CHILD_ENVIRONMENT_KEYS or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ExecutorError("learner child environment allowlist drift")
    return environment


def _execution_binding(
    *, command: list[str], environment: dict[str, str]
) -> dict[str, Any]:
    if set(environment) != CHILD_ENVIRONMENT_KEYS or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ExecutorError("cannot bind a non-allowlisted learner environment")
    return {
        "schema_version": REPORT_EXECUTION_BINDING_SCHEMA,
        "command_sha256": _value_sha256(command),
        "environment": dict(environment),
        "environment_sha256": _value_sha256(environment),
    }


def _validate_execution_binding(binding: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "command_sha256",
        "environment",
        "environment_sha256",
    }
    environment = binding.get("environment")
    if (
        set(binding) != expected_keys
        or binding.get("schema_version") != REPORT_EXECUTION_BINDING_SCHEMA
        or not isinstance(binding.get("command_sha256"), str)
        or not isinstance(environment, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment.items()
        )
        or set(environment) != CHILD_ENVIRONMENT_KEYS
        or binding.get("environment_sha256") != _value_sha256(environment)
    ):
        raise ExecutorError("A1 execution binding is invalid")


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
        raise ExecutorError(
            "sealed A1 learner recipe differs from the exact one-dose recipe"
        )
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
        raise ExecutorError(
            "sealed A1 topology/optimizer invariants are not one-B200 fresh Adam"
        )
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
        raise ExecutorError(
            f"A1 training-input verification failed: {error}"
        ) from error

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
    ]
    for flag, value in SEALED_A1_MODEL_CLI.items():
        command.extend((flag, value))
    command.extend(
        [
            "--data",
            str(verified["data_path"]),
            "--data-format",
            "memmap",
            "--data-loader-workers",
            str(DATA_LOADER_WORKERS),
            "--data-loader-prefetch",
            str(DATA_LOADER_PREFETCH),
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
    )
    return command


def _active_mps_processes(proc_root: Path = Path("/proc")) -> list[str]:
    """Return live CUDA MPS control/server processes without invoking a shell."""

    found: list[str] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError as error:
        raise ExecutorError(
            f"cannot inspect process table for CUDA MPS: {error}"
        ) from error
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            # Processes can exit while /proc is being traversed. A process whose
            # directory disappears during the scan no longer threatens the run.
            continue
        except PermissionError as error:
            raise ExecutorError(
                f"cannot prove process {entry.name} is not CUDA MPS: {error}"
            ) from error
        except OSError as error:
            raise ExecutorError(
                f"cannot inspect process {entry.name} for CUDA MPS: {error}"
            ) from error
        argv = [
            part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part
        ]
        if not argv:
            continue
        executable = Path(argv[0]).name
        if executable in MPS_EXECUTABLES:
            found.append(f"pid={entry.name} executable={executable}")
    return sorted(found)


def _probe_b200(
    gpu: int,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    mps_probe: Callable[[], list[str]] = _active_mps_processes,
) -> str:
    """Fail closed unless ``gpu`` is one idle, directly-owned B200."""

    try:
        result = runner(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=name,compute_mode,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExecutorError(
            f"cannot verify selected B200 GPU {gpu}: {error}"
        ) from error
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ExecutorError(
            f"selected GPU {gpu} query returned {len(rows)} rows: {rows}"
        )
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 3:
        raise ExecutorError(f"selected GPU {gpu} query is malformed: {rows[0]!r}")
    name, compute_mode_raw, memory_used_raw = fields
    if "B200" not in name.upper():
        raise ExecutorError(f"selected GPU {gpu} is not exactly one B200: {name!r}")
    compute_mode = compute_mode_raw.upper().replace("-", "_").replace(" ", "_")
    if compute_mode not in ACCEPTABLE_COMPUTE_MODES:
        raise ExecutorError(
            f"selected B200 GPU {gpu} has unsafe compute mode {compute_mode_raw!r}; "
            f"expected Default or Exclusive Process"
        )
    try:
        memory_used_mib = int(memory_used_raw)
    except ValueError as error:
        raise ExecutorError(
            f"selected B200 GPU {gpu} has unparseable memory usage {memory_used_raw!r}"
        ) from error
    if memory_used_mib < 0 or memory_used_mib > MAX_IDLE_GPU_MEMORY_MIB:
        raise ExecutorError(
            f"selected B200 GPU {gpu} is not idle: memory.used={memory_used_mib} MiB "
            f"(maximum idle allowance {MAX_IDLE_GPU_MEMORY_MIB} MiB)"
        )

    try:
        processes = runner(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        mps_processes = mps_probe()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExecutorError(
            f"cannot prove selected B200 GPU {gpu} is idle: {error}"
        ) from error
    compute_processes = [
        line.strip() for line in processes.stdout.splitlines() if line.strip()
    ]
    if compute_processes:
        raise ExecutorError(
            f"selected B200 GPU {gpu} has active compute process(es): {compute_processes}"
        )
    if mps_processes:
        raise ExecutorError(
            "CUDA MPS is active; the sealed one-B200 learner requires direct exclusive "
            f"ownership: {mps_processes}"
        )
    return name


def _raise_nofile_limit() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(max(soft, MIN_NOFILE), hard)
    if target < MIN_NOFILE:
        raise ExecutorError(f"hard RLIMIT_NOFILE={hard} is below required {MIN_NOFILE}")
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))


def _claim_path(verified: dict[str, Any]) -> Path:
    """Return the one stable claim path for the sealed contract identity.

    The sealed seed-ledger path is the shared, contract-bound anchor. A caller
    cannot obtain a second dose by choosing another receipt or copying the lock.
    """

    contract_sha = str(
        verified.get("claim_identity_sha256", verified.get("contract_sha256", ""))
    )
    prefix = "sha256:"
    digest = contract_sha.removeprefix(prefix)
    if (
        not contract_sha.startswith(prefix)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
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


def _load_claim_state(
    claim: Path,
    *,
    contract_sha256: str,
    claim_identity_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(claim.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(
            f"A1 contract claim is unreadable/corrupt: {claim}"
        ) from error
    if not isinstance(payload, dict):
        raise ExecutorError(f"A1 contract claim is not an object: {claim}")
    stated_digest = payload.get("state_sha256")
    unhashed = dict(payload)
    unhashed.pop("state_sha256", None)
    if stated_digest != _value_sha256(unhashed):
        raise ExecutorError(f"A1 contract claim digest is invalid: {claim}")
    expected_schema = (
        RETRY_CLAIM_SCHEMA
        if claim_identity_sha256 is not None
        and claim_identity_sha256 != contract_sha256
        else CLAIM_SCHEMA
    )
    if payload.get("schema_version") != expected_schema:
        raise ExecutorError(f"A1 contract claim schema is invalid: {claim}")
    if payload.get("contract_sha256") != contract_sha256:
        raise ExecutorError(f"A1 contract claim identity mismatch: {claim}")
    if (
        claim_identity_sha256 is not None
        and payload.get("claim_identity_sha256", payload.get("contract_sha256"))
        != claim_identity_sha256
    ):
        raise ExecutorError(f"A1 derived claim identity mismatch: {claim}")
    return payload


def _require_unconsumed_contract(verified: dict[str, Any]) -> None:
    claim = _claim_path(verified)
    if claim.exists():
        state = _load_claim_state(
            claim,
            contract_sha256=str(verified["contract_sha256"]),
            claim_identity_sha256=str(
                verified.get("claim_identity_sha256", verified["contract_sha256"])
            ),
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
    claim: Path,
    payload: dict[str, Any],
    *,
    contract_sha256: str,
    claim_identity_sha256: str | None = None,
) -> dict[str, Any]:
    current = _load_claim_state(
        claim,
        contract_sha256=contract_sha256,
        claim_identity_sha256=claim_identity_sha256,
    )
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


def _train_command_namespace(command: list[str]) -> argparse.Namespace:
    if len(command) < 3 or Path(command[1]).resolve(strict=False) != (
        _REPO_ROOT / "tools" / "train_bc.py"
    ).resolve(strict=False):
        raise ExecutorError(
            "retry proof command is not the canonical train_bc entry point"
        )
    try:
        args = train_bc.build_parser().parse_args(command[2:])
    except SystemExit as error:
        raise ExecutorError(
            "retry proof command cannot be parsed by train_bc"
        ) from error
    # Replay train_bc's effective architecture, not the parser's sentinel.
    # The failed production argv omitted --hidden-size; train_bc resolves that
    # to 640 for entity_graph before running checkpoint compatibility checks.
    if args.hidden_size is None:
        args.hidden_size = (
            640
            if args.arch == "entity_graph"
            else 768
            if args.arch == "xdim_graph"
            else 512
        )
    return args


def _literal_option_values(command: list[str], flag: str) -> list[str]:
    values: list[str] = []
    for index, item in enumerate(command):
        if item == flag:
            if index + 1 >= len(command):
                raise ExecutorError(f"retry proof command has valueless {flag}")
            values.append(command[index + 1])
        elif item.startswith(flag + "="):
            values.append(item.split("=", 1)[1])
    return values


def _checkpoint_architecture_mismatches(args: argparse.Namespace) -> list[str]:
    if not args.init_checkpoint:
        raise ExecutorError("retry proof command has no --init-checkpoint")
    try:
        import torch

        checkpoint = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise ExecutorError(
            f"cannot replay init-checkpoint architecture preflight: {error}"
        ) from error
    if not isinstance(checkpoint, dict):
        raise ExecutorError("init checkpoint is not a policy-checkpoint object")
    return train_bc._checkpoint_config_mismatches(  # noqa: SLF001
        policy_type=checkpoint.get("policy_type"),
        config=checkpoint.get("config"),
        args=args,
    )


def _load_failed_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(f"parent failure receipt is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise ExecutorError("parent failure receipt is not an object")
    stated = payload.get("receipt_sha256")
    unhashed = dict(payload)
    unhashed.pop("receipt_sha256", None)
    if stated != _value_sha256(unhashed):
        raise ExecutorError("parent failure receipt semantic digest is invalid")
    if payload.get("schema_version") != RECEIPT_SCHEMA:
        raise ExecutorError("parent failure receipt schema is invalid")
    return payload


def _write_retry_contract_no_clobber(path: Path, payload: dict[str, Any]) -> None:
    _mkdir_durable(path.parent)
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
        raise ExecutorError(
            f"refusing to overwrite learner retry contract: {path}"
        ) from error
    finally:
        tmp.unlink(missing_ok=True)


def authorize_failed_before_optimizer_retry(
    *,
    verified: dict[str, Any],
    parent_claim: Path,
    retry_command: list[str],
    checkpoint: Path,
    report: Path,
    receipt: Path,
    retry_contract_path: Path,
    publish: bool,
) -> dict[str, Any]:
    """Derive one immutable learner-only retry from a proven zero-step failure.

    The parent generation/data contract remains the training-data identity.  A
    separate digest identifies this retry authorization and therefore produces a
    fresh durable claim path.  The failed parent claim and receipt are only read
    and hashed; neither is edited, removed, or reused.
    """

    live_bindings = {
        Path(verified["lock_path"]): verified["lock_file_sha256"],
        Path(verified["data_path"]) / "corpus_meta.json": verified[
            "corpus_meta_file_sha256"
        ],
        Path(verified["validation_path"]): verified["validation_file_sha256"],
        Path(verified["producer"]["path"]): verified["producer"]["sha256"],
    }
    for path, expected_sha256 in live_bindings.items():
        try:
            actual_sha256 = _file_sha256(path.resolve(strict=True))
        except OSError as error:
            raise ExecutorError(
                f"cannot replay sealed retry binding {path}: {error}"
            ) from error
        if actual_sha256 != expected_sha256:
            raise ExecutorError(f"sealed retry binding drift: {path}")

    expected_parent_claim = _claim_path(verified).resolve(strict=False)
    parent_claim = parent_claim.expanduser()
    if parent_claim.is_symlink():
        raise ExecutorError("retry parent claim must not be a symlink")
    try:
        parent_claim = parent_claim.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(f"cannot resolve parent failed claim: {error}") from error
    if parent_claim != expected_parent_claim:
        raise ExecutorError(
            "retry parent claim is not the canonical sealed-contract claim"
        )
    parent = _load_claim_state(
        parent_claim, contract_sha256=str(verified["contract_sha256"])
    )
    if parent.get("status") != "failed":
        raise ExecutorError("retry requires a terminal failed parent claim")
    if parent.get("outputs") is not None:
        raise ExecutorError("retry parent claim contains training outputs")
    if not isinstance(parent.get("returncode"), int) or parent["returncode"] == 0:
        raise ExecutorError("retry parent claim does not prove a nonzero child exit")
    if not isinstance(parent.get("failure"), str) or not parent["failure"]:
        raise ExecutorError("retry parent claim has no failure evidence")
    parent_command = parent.get("command")
    if not isinstance(parent_command, list) or not all(
        isinstance(item, str) for item in parent_command
    ):
        raise ExecutorError("retry parent claim has no canonical command")
    if parent.get("command_sha256") != _value_sha256(parent_command):
        raise ExecutorError("retry parent command digest does not replay")
    parent_binding = parent.get("execution_binding")
    if not isinstance(parent_binding, dict):
        raise ExecutorError("retry parent claim has no execution binding")
    _validate_execution_binding(parent_binding)
    if parent_binding["command_sha256"] != parent["command_sha256"]:
        raise ExecutorError("retry parent execution binding disagrees with its command")

    receipt_target = Path(str(parent.get("receipt_target", ""))).expanduser()
    if receipt_target.is_symlink():
        raise ExecutorError("parent failure receipt must not be a symlink")
    try:
        receipt_target = receipt_target.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(
            f"cannot resolve parent failure receipt: {error}"
        ) from error
    parent_receipt = _load_failed_receipt(receipt_target)
    if (
        parent_receipt.get("status") != "failed"
        or parent_receipt.get("outputs") is not None
        or parent_receipt.get("claim") != str(parent_claim)
        or parent_receipt.get("claim_state_sha256") != parent.get("state_sha256")
        or parent_receipt.get("contract_sha256") != verified["contract_sha256"]
        or parent_receipt.get("command_sha256") != parent["command_sha256"]
        or parent_receipt.get("returncode") != parent["returncode"]
        or parent_receipt.get("failure") != parent["failure"]
        or parent_receipt.get("execution_binding") != parent_binding
    ):
        raise ExecutorError("parent failed claim and receipt do not agree")
    preserved_receipt_bindings = {
        "lock": str(verified["lock_path"]),
        "lock_file_sha256": verified["lock_file_sha256"],
        "corpus": str(verified["data_path"]),
        "corpus_meta_file_sha256": verified["corpus_meta_file_sha256"],
        "payload_inventory_sha256": verified["payload_inventory_sha256"],
        "validation_manifest": str(verified["validation_path"]),
        "validation_manifest_file_sha256": verified["validation_file_sha256"],
        "producer_checkpoint_sha256": verified["producer"]["sha256"],
        "learner_training_recipe_sha256": _value_sha256(verified["recipe"]),
    }
    for key, expected in preserved_receipt_bindings.items():
        if parent_receipt.get(key) != expected or parent.get(key) != expected:
            raise ExecutorError(f"parent failure drifted from sealed binding {key}")

    if _literal_option_values(parent_command, "--graph-layers"):
        raise ExecutorError(
            "authorized parent must literally omit --graph-layers and use the "
            "historical train_bc default"
        )
    parent_args = _train_command_namespace(parent_command)
    parent_mismatches = _checkpoint_architecture_mismatches(parent_args)
    if parent_mismatches != ["graph_layers checkpoint=6 cli=4"]:
        raise ExecutorError(
            "parent failure is not the authorized pre-optimizer graph-layer mismatch"
        )
    parent_checkpoint = Path(parent_args.checkpoint).expanduser().resolve(strict=False)
    parent_report = Path(parent_args.report).expanduser().resolve(strict=False)
    parent_optimizer = Path(str(parent_checkpoint) + ".optimizer.pt")
    for path in (parent_checkpoint, parent_optimizer, parent_report):
        if path.exists():
            raise ExecutorError(
                f"cannot prove zero-output/zero-step parent failure; artifact exists: {path}"
            )

    if _literal_option_values(retry_command, "--graph-layers") != ["6"]:
        raise ExecutorError(
            "corrected retry still fails architecture authorization: must contain "
            "exactly one literal --graph-layers 6"
        )
    retry_args = _train_command_namespace(retry_command)
    retry_mismatches = _checkpoint_architecture_mismatches(retry_args)
    if retry_mismatches:
        raise ExecutorError(
            "corrected retry command still fails architecture preflight: "
            + "; ".join(retry_mismatches)
        )
    allowed_drift = {
        "graph_layers",
        "checkpoint",
        "report",
    }
    parent_values = vars(parent_args)
    retry_values = vars(retry_args)
    drift = sorted(
        key
        for key in set(parent_values) | set(retry_values)
        if key not in allowed_drift and parent_values.get(key) != retry_values.get(key)
    )
    if drift:
        raise ExecutorError(
            f"retry changes non-architecture learner semantics: {drift}"
        )
    if Path(retry_args.init_checkpoint).resolve(strict=False) != Path(
        parent_args.init_checkpoint
    ).resolve(strict=False):
        raise ExecutorError("retry changes the sealed producer checkpoint")
    checkpoint = checkpoint.expanduser().resolve(strict=False)
    report = report.expanduser().resolve(strict=False)
    receipt = receipt.expanduser().resolve(strict=False)
    retry_contract_path = retry_contract_path.expanduser()
    if retry_contract_path.is_symlink():
        raise ExecutorError("retry contract path must not be a symlink")
    retry_contract_path = retry_contract_path.resolve(strict=False)
    if (
        Path(retry_args.checkpoint).resolve(strict=False) != checkpoint
        or Path(retry_args.report).resolve(strict=False) != report
    ):
        raise ExecutorError("retry command does not bind the requested fresh outputs")
    _require_fresh_outputs(checkpoint, report, receipt)
    if retry_contract_path in {
        checkpoint,
        Path(str(checkpoint) + ".optimizer.pt"),
        report,
        receipt,
        parent_claim,
        receipt_target,
    }:
        raise ExecutorError("retry contract path aliases a claim, receipt, or output")
    if {checkpoint, Path(str(checkpoint) + ".optimizer.pt"), report} & {
        parent_checkpoint,
        parent_optimizer,
        parent_report,
    }:
        raise ExecutorError("retry must use a completely fresh output set")

    preserved = {
        "parent_contract_sha256": verified["contract_sha256"],
        "parent_lock": str(verified["lock_path"]),
        "parent_lock_file_sha256": verified["lock_file_sha256"],
        "corpus": str(verified["data_path"]),
        "corpus_meta_file_sha256": verified["corpus_meta_file_sha256"],
        "payload_inventory_sha256": verified["payload_inventory_sha256"],
        "data_fingerprint": verified["data_fingerprint"],
        "producer_checkpoint_sha256": verified["producer"]["sha256"],
        "producer_checkpoint": str(verified["producer"]["path"]),
        "learner_training_recipe_sha256": _value_sha256(verified["recipe"]),
        "learner_value_objective_sha256": _value_sha256(verified["objective"]),
        "selected_game_seed_set_sha256": verified["selected_game_seed_set_sha256"],
        "training_game_seed_set_sha256": verified["training_game_seed_set_sha256"],
        "validation_manifest_file_sha256": verified["validation_file_sha256"],
        "validation_manifest": str(verified["validation_path"]),
        "validation_game_seed_set_sha256": verified["validation_game_seed_set_sha256"],
    }
    parent_evidence = {
        "claim": str(parent_claim),
        "claim_file_sha256": _file_sha256(parent_claim),
        "claim_state_sha256": parent["state_sha256"],
        "receipt": str(receipt_target),
        "receipt_file_sha256": _file_sha256(receipt_target),
        "receipt_sha256": parent_receipt["receipt_sha256"],
        "command_sha256": parent["command_sha256"],
        "returncode": parent["returncode"],
        "failure": parent["failure"],
    }
    retry_identity_evidence = {
        "schema_version": RETRY_IDENTITY_SCHEMA,
        "repair_kind": RETRY_REPAIR_KIND,
        "parent_contract_sha256": verified["contract_sha256"],
        "parent": parent_evidence,
    }
    # This is deliberately independent of r2 argv and output paths.  Therefore
    # changing filenames cannot mint another retry claim; O_EXCL on the derived
    # claim path physically caps this repair at one attempt.
    retry_identity = _value_sha256(retry_identity_evidence)
    retry_contract = {
        "schema_version": RETRY_CONTRACT_SCHEMA,
        "retry_identity": retry_identity_evidence,
        "retry_identity_sha256": retry_identity,
        "parent": {
            **parent_evidence,
            "pre_optimizer_proof": {
                "kind": "replayed_init_checkpoint_architecture_preflight",
                "mismatches": parent_mismatches,
                "optimizer_steps": 0,
                "outputs": None,
            },
        },
        "preserved_bindings": preserved,
        "retry": {
            "command_sha256": _value_sha256(retry_command),
            "architecture_correction": {
                "graph_layers_before": int(parent_args.graph_layers),
                "graph_layers_after": int(retry_args.graph_layers),
            },
            "checkpoint": str(checkpoint),
            "optimizer_sidecar": str(Path(str(checkpoint) + ".optimizer.pt")),
            "report": str(report),
            "receipt": str(receipt),
        },
    }
    retry_contract["retry_contract_sha256"] = _value_sha256(retry_contract)
    derived_claim_path = _claim_path(
        {**verified, "claim_identity_sha256": retry_identity}
    ).resolve(strict=False)
    if retry_contract_path == derived_claim_path:
        raise ExecutorError("retry contract path aliases its derived durable claim")
    if publish:
        if retry_contract_path.exists():
            try:
                existing = json.loads(retry_contract_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ExecutorError("existing retry contract is unreadable") from error
            if existing != retry_contract:
                raise ExecutorError(
                    "existing retry contract differs; refusing overwrite/edit"
                )
        else:
            _write_retry_contract_no_clobber(retry_contract_path, retry_contract)
    derived = dict(verified)
    derived.update(
        {
            "claim_identity_sha256": retry_identity,
            "retry_contract": retry_contract,
            "retry_contract_path": retry_contract_path,
            "retry_contract_file_sha256": (
                _file_sha256(retry_contract_path) if publish else None
            ),
        }
    )
    return derived


def _write_receipt_no_clobber(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
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


def _bind_training_report(report: Path, *, execution_binding: dict[str, Any]) -> None:
    """Atomically bind the trainer report to the exact executor environment.

    ``train_bc`` is part of the immutable A1 learner runtime inventory and must
    not be edited after the wave.  The transaction executor is deliberately not
    in that inventory, so it adds this operational binding after the child exits
    and before the report is verified, hashed, or receipted.
    """

    if report.is_symlink() or not report.is_file():
        raise ExecutorError(f"A1 training report is not a regular file: {report}")
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(f"cannot parse A1 training report: {error}") from error
    if not isinstance(payload, dict):
        raise ExecutorError("A1 training report must be a JSON object")
    if REPORT_EXECUTION_BINDING_FIELD in payload:
        raise ExecutorError(
            "A1 training child pre-populated the executor-owned report binding"
        )
    _validate_execution_binding(execution_binding)
    payload[REPORT_EXECUTION_BINDING_FIELD] = execution_binding
    tmp = report.with_name(f".{report.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("xb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, report)
        _fsync_directory(report.parent)
    finally:
        tmp.unlink(missing_ok=True)
        _fsync_directory(report.parent)


def _verify_training_outputs(
    *,
    checkpoint: Path,
    report: Path,
    verified: dict[str, Any],
    execution_binding: dict[str, Any],
) -> dict[str, Any]:
    _validate_execution_binding(execution_binding)
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
        **SEALED_A1_MODEL_REPORT,
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
        "validation_game_seed_set_sha256": verified["validation_game_seed_set_sha256"],
        "a1_selected_game_seed_set_sha256": verified["selected_game_seed_set_sha256"],
        "a1_training_game_seed_set_sha256": verified["training_game_seed_set_sha256"],
        "a1_memmap_payload_inventory_sha256": verified["payload_inventory_sha256"],
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
        raise ExecutorError(
            "A1 training report does not echo the sealed value objective"
        )
    if report_payload.get(REPORT_EXECUTION_BINDING_FIELD) != execution_binding:
        raise ExecutorError(
            "A1 training report does not bind the exact child environment/command"
        )
    metrics = report_payload.get("metrics")
    if (
        not isinstance(metrics, list)
        or len(metrics) != 1
        or not isinstance(metrics[0], dict)
        or metrics[0].get("epoch") != 1
    ):
        raise ExecutorError(
            "A1 training report does not prove exactly one completed epoch"
        )
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
        raise ExecutorError(
            "A1 training report has invalid validation coverage/metrics"
        )
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
        "a1_selected_game_seed_set_sha256": verified["selected_game_seed_set_sha256"],
        "a1_training_game_seed_set_sha256": verified["training_game_seed_set_sha256"],
        "a1_learner_training_recipe_sha256": _value_sha256(recipe),
        "a1_memmap_payload_inventory_sha256": verified["payload_inventory_sha256"],
    }
    if not isinstance(value_training, dict) or any(
        value_training.get(key) != value
        for key, value in expected_value_training.items()
    ):
        raise ExecutorError("A1 training report value-training provenance drift")
    if "scalar" not in value_training.get("trained_value_readouts", []):
        raise ExecutorError(
            "A1 candidate does not attest a trained scalar value readout"
        )
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
        "execution_binding_sha256": _value_sha256(execution_binding),
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
    # Hardware refusal must precede the durable one-dose claim. Occupancy or an
    # MPS daemon is an operational precondition failure, not a consumed dose.
    gpu_name = probe(gpu)
    child_environment = _child_environment(gpu)
    execution_binding = _execution_binding(
        command=command, environment=child_environment
    )
    started_ns = time.time_ns()
    claim_identity = str(
        verified.get("claim_identity_sha256", verified["contract_sha256"])
    )
    is_retry = "retry_contract" in verified
    retry_reference = (
        {
            "path": str(verified["retry_contract_path"]),
            "file_sha256": verified["retry_contract_file_sha256"],
            "retry_contract_sha256": verified["retry_contract"][
                "retry_contract_sha256"
            ],
        }
        if is_retry
        else None
    )
    claim_payload = {
        "schema_version": RETRY_CLAIM_SCHEMA if is_retry else CLAIM_SCHEMA,
        "status": "claimed",
        "contract_sha256": verified["contract_sha256"],
        "command_sha256": _value_sha256(command),
        "execution_binding": execution_binding,
        "started_unix_ns": started_ns,
    }
    if is_retry:
        claim_payload.update(
            {
                "claim_identity_sha256": claim_identity,
                "retry_contract": retry_reference,
            }
        )
    claim = _claim_attempt(verified, claim_payload)
    status = "failed"
    returncode: int | None = None
    output_artifacts: dict[str, Any] | None = None
    failure: str | None = None
    try:
        _mkdir_durable(checkpoint.parent)
        _mkdir_durable(report.parent)
        _mkdir_durable(receipt.parent)
        result = runner(
            command,
            cwd=str(_REPO_ROOT),
            env=child_environment,
            check=False,
            preexec_fn=_raise_nofile_limit,
        )
        returncode = int(result.returncode)
        if returncode != 0:
            raise ExecutorError(f"train_bc exited nonzero: {returncode}")
        _bind_training_report(report, execution_binding=execution_binding)
        output_artifacts = _verify_training_outputs(
            checkpoint=checkpoint,
            report=report,
            verified=verified,
            execution_binding=execution_binding,
        )
        status = "complete"
    except Exception as error:  # receipt every claimed attempt, then re-raise.
        failure = f"{type(error).__name__}: {error}"
    finished_ns = time.time_ns()
    evidence_payload = {
        "schema_version": RETRY_RECEIPT_SCHEMA if is_retry else RECEIPT_SCHEMA,
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
        "execution_binding": execution_binding,
        "world_size": 1,
        "gpu": gpu,
        "gpu_name": gpu_name,
        "started_unix_ns": started_ns,
        "finished_unix_ns": finished_ns,
        "returncode": returncode,
        "outputs": output_artifacts,
        "failure": failure,
    }
    if is_retry:
        evidence_payload.update(
            {
                "claim_identity_sha256": claim_identity,
                "retry_contract": retry_reference,
            }
        )
    terminal_claim_payload = dict(evidence_payload)
    terminal_claim_payload["schema_version"] = (
        RETRY_CLAIM_SCHEMA if is_retry else CLAIM_SCHEMA
    )
    terminal_claim_payload["receipt_target"] = str(receipt)
    terminal_claim = _write_terminal_claim(
        claim,
        terminal_claim_payload,
        contract_sha256=str(verified["contract_sha256"]),
        claim_identity_sha256=claim_identity,
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
        "--retry-parent-claim",
        type=Path,
        default=None,
        help="terminal failed-before-optimizer parent claim authorizing one derived retry",
    )
    parser.add_argument(
        "--retry-contract",
        type=Path,
        default=None,
        help="immutable learner-only retry contract output (required with parent claim)",
    )
    parser.add_argument(
        "--go", action="store_true", help="execute locally; default is verified dry-run"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        python = _lexical_python_executable(args.python)
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
        command = build_train_command(
            verified,
            python=python,
            checkpoint=checkpoint,
            report=report,
        )
        if (args.retry_parent_claim is None) != (args.retry_contract is None):
            raise ExecutorError(
                "--retry-parent-claim and --retry-contract must be supplied together"
            )
        if args.retry_parent_claim is not None:
            verified = authorize_failed_before_optimizer_retry(
                verified=verified,
                parent_claim=args.retry_parent_claim,
                retry_command=command,
                checkpoint=checkpoint,
                report=report,
                receipt=receipt,
                retry_contract_path=args.retry_contract,
                publish=bool(args.go),
            )
        claim = _claim_path(verified)
        _require_fresh_outputs(checkpoint, report, receipt, claim=claim)
        _require_unconsumed_contract(verified)
        child_environment = _child_environment(args.gpu)
        execution_binding = _execution_binding(
            command=command, environment=child_environment
        )
        plan = {
            "schema_version": PLAN_SCHEMA,
            "mode": "go" if args.go else "dry-run",
            "contract_sha256": verified["contract_sha256"],
            "claim_identity_sha256": verified.get(
                "claim_identity_sha256", verified["contract_sha256"]
            ),
            "retry_contract": (
                verified.get("retry_contract") if "retry_contract" in verified else None
            ),
            "global_n_full": 128,
            "world_size": 1,
            "gpu": args.gpu,
            "command": command,
            "command_sha256": _value_sha256(command),
            "execution_binding": execution_binding,
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
