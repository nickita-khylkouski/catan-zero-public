"""One typed production command for generation, training, and evaluation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import socket
import stat
import subprocess
import sys
from time import time
from typing import Any, Iterator, Sequence

from catan_zero.production_contracts import (
    NATIVE_REQUIRED_CAPABILITIES,
    ProductionContractError,
    canonical_json_sha256,
    pipeline_readiness,
    production_status,
    validate_pipeline_contract,
)


JOB_SCHEMA = "catan-zero-production-job-v1"
PLAN_SCHEMA = "catan-zero-production-plan-v1"
RUN_RECEIPT_SCHEMA = "catan-zero-production-run-v1"
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,95}$")
_CUDA_DEVICE = re.compile(r"^cuda:(0|[1-9][0-9]*)$")
_COMMON_KEYS = {"schema_version", "pipeline", "run_id", "run_dir"}
_PIPELINE_KEYS = {
    "generate": {
        "checkpoint",
        "games",
        "base_seed",
        "claim_label",
        "workers",
        "resume",
        "gpu",
    },
    "train": {
        "data",
        "recipe",
        "init_checkpoint",
        "parent_checkpoint",
        "architecture_upgrade_receipt",
        "lock",
        "composite_build_receipt",
        "plan_receipt",
    },
    "evaluate": {
        "candidate",
        "champion",
        "pairs",
        "workers",
        "devices",
        "threads_per_worker",
        "base_seed",
        "held_out_suite",
    },
}


class ProductionCLIError(RuntimeError):
    """A production job is malformed, unsafe, or not currently authorized."""


def repo_root() -> Path:
    candidates = (Path.cwd(), Path(__file__).resolve().parents[2])
    for root in candidates:
        if (root / "tools").is_dir() and (root / "configs").is_dir():
            return root.resolve()
    raise ProductionCLIError(
        "catan-zero production CLI requires a complete repository checkout"
    )


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionCLIError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProductionCLIError(f"{label} must be a JSON object")
    return value


def _positive_integer(value: object, *, field: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProductionCLIError(f"{field} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ProductionCLIError(f"{field} must be >= {minimum}")
    return value


def _cuda_device_indices(value: object, *, field: str = "devices") -> list[int]:
    """Return one unambiguous set of logical CUDA device indices."""

    if not isinstance(value, list) or not value:
        raise ProductionCLIError(f"{field} must be a non-empty list of cuda:N strings")
    indices: list[int] = []
    for device in value:
        if not isinstance(device, str) or _CUDA_DEVICE.fullmatch(device) is None:
            raise ProductionCLIError(
                f"{field} entries must use the exact cuda:N form: {device!r}"
            )
        indices.append(int(device.removeprefix("cuda:")))
    if len(indices) != len(set(indices)):
        raise ProductionCLIError(f"{field} must name unique CUDA devices")
    return indices


def _absolute_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ProductionCLIError(f"{field} must be a non-empty absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ProductionCLIError(f"{field} must be absolute: {path}")
    return Path(os.path.abspath(os.fspath(path)))


def load_job(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve(strict=True)
    job = _load_json_object(source, label="production job")
    if job.get("schema_version") != JOB_SCHEMA:
        raise ProductionCLIError(f"production job schema must be {JOB_SCHEMA!r}")
    pipeline = job.get("pipeline")
    if pipeline == "ppo":
        readiness = pipeline_readiness(repo_root(), "ppo")
        raise ProductionCLIError(
            "PPO is not a commissioned production pipeline: " + readiness["reason"]
        )
    if pipeline not in _PIPELINE_KEYS:
        raise ProductionCLIError(
            f"pipeline must be one of {sorted([*_PIPELINE_KEYS, 'ppo'])}"
        )
    expected = _COMMON_KEYS | _PIPELINE_KEYS[str(pipeline)]
    missing = sorted(_COMMON_KEYS - set(job))
    unknown = sorted(set(job) - expected)
    if missing or unknown:
        raise ProductionCLIError(
            f"production job key drift: missing={missing} unknown={unknown}"
        )
    run_id = job.get("run_id")
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ProductionCLIError(
            "run_id must be 3-96 lowercase letters, digits, dots, dashes, or underscores"
        )
    run_dir = _absolute_path(job.get("run_dir"), field="run_dir")
    if run_dir.name != run_id:
        raise ProductionCLIError("run_dir basename must equal run_id")
    job["run_dir"] = str(run_dir)
    job["_source"] = str(source)

    if pipeline == "generate":
        job["checkpoint"] = str(
            _absolute_path(job.get("checkpoint"), field="checkpoint")
        )
        _positive_integer(job.get("games"), field="games")
        _positive_integer(job.get("base_seed"), field="base_seed", allow_zero=True)
        if not isinstance(job.get("claim_label"), str) or not job["claim_label"]:
            raise ProductionCLIError("claim_label must be a non-empty string")
        if job.get("workers") is not None:
            _positive_integer(job["workers"], field="workers")
        if job.get("gpu") is not None:
            _positive_integer(job["gpu"], field="gpu", allow_zero=True)
        if not isinstance(job.get("resume", False), bool):
            raise ProductionCLIError("resume must be boolean")
    elif pipeline == "train":
        job["data"] = str(_absolute_path(job.get("data"), field="data"))
        recipe = job.get("recipe")
        if recipe == "a1-current-35m-b200":
            required = {"lock", "composite_build_receipt", "plan_receipt"}
            missing = sorted(key for key in required if key not in job)
            forbidden = sorted(
                key
                for key in (
                    "init_checkpoint",
                    "parent_checkpoint",
                    "architecture_upgrade_receipt",
                )
                if key in job
            )
            if missing or forbidden:
                raise ProductionCLIError(
                    "scratch training job field drift: "
                    f"missing={missing} forbidden={forbidden}"
                )
            job["lock"] = str(_absolute_path(job.get("lock"), field="lock"))
            job["composite_build_receipt"] = str(
                _absolute_path(
                    job.get("composite_build_receipt"),
                    field="composite_build_receipt",
                )
            )
            job["plan_receipt"] = str(
                _absolute_path(job.get("plan_receipt"), field="plan_receipt")
            )
        elif recipe == "a1-parent-update-35m-b200":
            forbidden = sorted(
                key
                for key in ("lock", "composite_build_receipt", "plan_receipt")
                if key in job
            )
            required = {
                "init_checkpoint",
                "parent_checkpoint",
            }
            missing = sorted(required - set(job))
            if missing or forbidden:
                raise ProductionCLIError(
                    "parent-update training job field drift: "
                    f"missing={missing} "
                    f"forbidden={forbidden}"
                )
            job["init_checkpoint"] = str(
                _absolute_path(job.get("init_checkpoint"), field="init_checkpoint")
            )
            job["parent_checkpoint"] = str(
                _absolute_path(
                    job.get("parent_checkpoint"), field="parent_checkpoint"
                )
            )
            if "architecture_upgrade_receipt" in job:
                job["architecture_upgrade_receipt"] = str(
                    _absolute_path(
                        job.get("architecture_upgrade_receipt"),
                        field="architecture_upgrade_receipt",
                    )
                )
        else:
            raise ProductionCLIError(
                "training recipe must be 'a1-current-35m-b200' or "
                "'a1-parent-update-35m-b200'"
            )
    else:
        for field in ("candidate", "champion"):
            job[field] = str(_absolute_path(job.get(field), field=field))
        held_out = job.get("held_out_suite")
        if held_out is not None:
            job["held_out_suite"] = str(
                _absolute_path(held_out, field="held_out_suite")
            )
        for field in ("pairs", "workers"):
            _positive_integer(job.get(field), field=field)
        _positive_integer(
            job.get("threads_per_worker", 0),
            field="threads_per_worker",
            allow_zero=True,
        )
        _positive_integer(job.get("base_seed"), field="base_seed", allow_zero=True)
        _cuda_device_indices(job.get("devices"))
    return job


def _file_sha256(path: Path) -> str:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ProductionCLIError(
                f"input must be a regular non-symlink file: {path}"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise ProductionCLIError(f"input changed while opening: {path}")
            digest = hashlib.sha256()
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ProductionCLIError(f"cannot attest input {path}: {error}") from error
    identity_before = (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
        opened.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise ProductionCLIError(f"input changed while hashing: {path}")
    return digest.hexdigest()


def _input_paths(job: dict[str, Any]) -> dict[str, Path]:
    pipeline = str(job["pipeline"])
    if pipeline == "generate":
        return {"checkpoint": Path(job["checkpoint"])}
    if pipeline == "train":
        paths = {"data": Path(job["data"])}
        if job["recipe"] == "a1-parent-update-35m-b200":
            paths["init_checkpoint"] = Path(job["init_checkpoint"])
            paths["parent_checkpoint"] = Path(job["parent_checkpoint"])
            if "architecture_upgrade_receipt" in job:
                paths["architecture_upgrade_receipt"] = Path(
                    job["architecture_upgrade_receipt"]
                )
            return paths
        paths.update(
            lock=Path(job["lock"]),
            composite_build_receipt=Path(job["composite_build_receipt"]),
        )
        plan_receipt = Path(job["plan_receipt"])
        if plan_receipt.is_file():
            paths["plan_receipt"] = plan_receipt
        return paths
    paths = {
        "candidate": Path(job["candidate"]),
        "champion": Path(job["champion"]),
    }
    if job.get("held_out_suite") is not None:
        paths["held_out_suite"] = Path(job["held_out_suite"])
    return paths


def _command(job: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    pipeline = str(job["pipeline"])
    run_dir = Path(job["run_dir"])
    if pipeline == "generate":
        command = [
            sys.executable,
            str(contract["launcher"]),
            "--config",
            str(contract["config"]),
            "--guard",
            str(contract["guard"]),
            "--checkpoint",
            str(job["checkpoint"]),
            "--out-dir",
            str(run_dir),
            "--games",
            str(job["games"]),
            "--base-seed",
            str(job["base_seed"]),
            "--claim-label",
            str(job["claim_label"]),
        ]
        if job.get("workers") is not None:
            command.extend(("--workers", str(job["workers"])))
        if bool(job.get("resume", False)):
            command.append("--resume")
        return command
    if pipeline == "train":
        if job["recipe"] == "a1-parent-update-35m-b200":
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=8",
                str(contract["launcher"]),
                "--config",
                str(contract["config"]),
                "--data",
                str(job["data"]),
                "--init-checkpoint",
                str(job["init_checkpoint"]),
                "--parent-checkpoint",
                str(job["parent_checkpoint"]),
                "--checkpoint",
                str(run_dir / "candidate.pt"),
                "--report",
                str(run_dir / "train.report.json"),
            ]
            if "architecture_upgrade_receipt" in job:
                command.extend(
                    [
                        "--architecture-upgrade-receipt",
                        str(job["architecture_upgrade_receipt"]),
                    ]
                )
            return command
        return [
            sys.executable,
            str(contract["launcher"]),
            "--lock",
            str(job["lock"]),
            "--data",
            str(job["data"]),
            "--composite-build-receipt",
            str(job["composite_build_receipt"]),
            "--checkpoint",
            str(run_dir / "candidate.pt"),
            "--report",
            str(run_dir / "train.report.json"),
            "--receipt",
            str(job["plan_receipt"]),
            "--execution-receipt",
            str(run_dir / "scratch.execution.json"),
            "--python",
            sys.executable,
            "--go",
        ]
    command = [
        sys.executable,
        str(contract["launcher"]),
        "--config",
        str(contract["config"]),
        "--candidate",
        str(job["candidate"]),
        "--champion",
        str(job["champion"]),
        "--out",
        str(run_dir / "evaluation.json"),
        "--pairs",
        str(job["pairs"]),
        "--workers",
        str(job["workers"]),
        "--devices",
        ",".join(job["devices"]),
        "--threads-per-worker",
        str(job.get("threads_per_worker", 0)),
        "--base-seed",
        str(job["base_seed"]),
    ]
    if job.get("held_out_suite") is not None:
        command.extend(("--held-out-suite", str(job["held_out_suite"])))
    return command


def _command_environment(job: dict[str, Any]) -> dict[str, str]:
    if job["pipeline"] == "generate" and job.get("gpu") is not None:
        return {"CUDA_VISIBLE_DEVICES": str(job["gpu"])}
    return {}


def _prepare_command(job: dict[str, Any], contract: dict[str, Any]) -> list[str] | None:
    if job["pipeline"] != "train" or job["recipe"] != "a1-current-35m-b200":
        return None
    return _command(job, contract)[:-1]


def _git_identity(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProductionCLIError(
            f"cannot attest repository identity: {error}"
        ) from error
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ProductionCLIError(
            f"repository HEAD is not a full commit identity: {commit!r}"
        )
    return {"commit": commit, "tracked_changes": status, "clean": not status}


def build_plan(job_path: Path) -> dict[str, Any]:
    job = load_job(job_path)
    root = repo_root()
    pipeline = str(job["pipeline"])
    recipe = str(job["recipe"]) if pipeline == "train" else None
    contract = validate_pipeline_contract(root, pipeline, recipe)
    readiness = pipeline_readiness(root, pipeline, recipe)
    inputs = {
        name: {"path": str(path), "sha256": _file_sha256(path)}
        for name, path in _input_paths(job).items()
    }
    if pipeline == "train" and recipe == "a1-parent-update-35m-b200":
        parent_sha = inputs["parent_checkpoint"]["sha256"]
        initializer_sha = inputs["init_checkpoint"]["sha256"]
        has_receipt = "architecture_upgrade_receipt" in inputs
        if parent_sha != initializer_sha and not has_receipt:
            raise ProductionCLIError(
                "parent-update initializer differs from the exact incumbent; "
                "architecture_upgrade_receipt is required"
            )
        if parent_sha == initializer_sha and has_receipt:
            raise ProductionCLIError(
                "exact-parent update must not claim architecture_upgrade_receipt"
            )
    public_job = {key: value for key, value in job.items() if not key.startswith("_")}
    value: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "job": public_job,
        "job_sha256": canonical_json_sha256(public_job),
        "job_file": {
            "path": job["_source"],
            "sha256": _file_sha256(Path(job["_source"])),
        },
        "contract": contract,
        "readiness": readiness,
        "inputs": inputs,
        "command": _command(job, contract),
        "prepare_command": _prepare_command(job, contract),
        "environment": _command_environment(job),
        "repository": _git_identity(root),
        "run_receipt": str(
            Path(job["run_dir"]).with_name(Path(job["run_dir"]).name + ".run.json")
        ),
    }
    value["plan_sha256"] = canonical_json_sha256(value)
    return value


def _package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _native_runtime_identity() -> dict[str, Any]:
    result: dict[str, Any] = {
        "wheel_sha256": None,
        "extension_path": None,
        "extension_sha256": None,
        "capabilities": [],
    }
    try:
        distribution = metadata.distribution("catanatron-rs")
        raw = distribution.read_text("direct_url.json")
        direct_url = json.loads(raw) if raw is not None else None
        if isinstance(direct_url, dict):
            archive = direct_url.get("archive_info")
            hashes = archive.get("hashes") if isinstance(archive, dict) else None
            if isinstance(hashes, dict) and isinstance(hashes.get("sha256"), str):
                result["wheel_sha256"] = hashes["sha256"]
            elif isinstance(archive, dict) and isinstance(archive.get("hash"), str):
                result["wheel_sha256"] = archive["hash"].removeprefix("sha256=")
        import catanatron_rs  # type: ignore

        native_suffixes = (".so", ".pyd", ".dylib")
        extension_records = [
            record
            for record in (distribution.files or ())
            if str(record).endswith(native_suffixes)
        ]
        if len(extension_records) != 1:
            raise ProductionCLIError(
                "installed catanatron-rs must contain exactly one native extension; "
                f"found={len(extension_records)}"
            )
        relative = Path(str(extension_records[0]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ProductionCLIError(
                f"installed native extension record is unsafe: {relative}"
            )
        extension = Path(
            os.path.abspath(os.fspath(distribution.locate_file(extension_records[0])))
        )
        try:
            extension_stat = extension.lstat()
            resolved_extension = extension.resolve(strict=True)
        except OSError as error:
            raise ProductionCLIError(
                f"cannot resolve installed native extension {extension}: {error}"
            ) from error
        if (
            stat.S_ISLNK(extension_stat.st_mode)
            or not stat.S_ISREG(extension_stat.st_mode)
            or resolved_extension != extension
        ):
            raise ProductionCLIError(
                "installed native extension must be a canonical regular "
                f"non-symlink file: {extension}"
            )
        loaded_module = sys.modules.get("catanatron_rs.catanatron_rs")
        loaded_raw = getattr(loaded_module, "__file__", None)
        if not isinstance(loaded_raw, str) or not loaded_raw:
            raise ProductionCLIError(
                "loaded catanatron-rs native extension has no file identity"
            )
        loaded_extension = Path(os.path.abspath(loaded_raw))
        try:
            resolved_loaded = loaded_extension.resolve(strict=True)
        except OSError as error:
            raise ProductionCLIError(
                f"cannot resolve loaded native extension {loaded_extension}: {error}"
            ) from error
        if resolved_loaded != loaded_extension or loaded_extension != extension:
            raise ProductionCLIError(
                "loaded native extension path drift: "
                f"installed={extension} loaded={loaded_extension}"
            )
        result["extension_path"] = str(extension)
        result["extension_sha256"] = _file_sha256(extension)
        capability_fn = getattr(catanatron_rs, "gumbel_search_capabilities", None)
        if callable(capability_fn):
            result["capabilities"] = sorted(set(map(str, capability_fn())))
    except (
        Exception
    ) as error:  # The doctor reports every unavailable identity together.
        result["error"] = str(error)
    return result


def _nvidia_driver_identity() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        versions = sorted(
            {line.strip() for line in completed.stdout.splitlines() if line.strip()}
        )
        return {"versions": versions, "error": None}
    except (OSError, subprocess.SubprocessError) as error:
        return {"versions": [], "error": str(error)}


def _verify_plan_artifacts(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for name, identity in plan["inputs"].items():
        path = Path(identity["path"])
        try:
            actual = _file_sha256(path)
        except ProductionCLIError as error:
            errors.append(str(error))
            continue
        if actual != identity["sha256"]:
            errors.append(
                f"input {name} drift: expected={identity['sha256']} actual={actual}"
            )
    try:
        job_actual = _file_sha256(Path(plan["job_file"]["path"]))
    except ProductionCLIError as error:
        errors.append(str(error))
    else:
        if job_actual != plan["job_file"]["sha256"]:
            errors.append("production job file changed after planning")
    return errors


def doctor(plan: dict[str, Any]) -> dict[str, Any]:
    root = repo_root()
    runtime = _load_json_object(
        root / "configs/runtime/a1_production_runtime.json",
        label="production runtime contract",
    )
    actual: dict[str, Any] = {
        "python_version": platform.python_version(),
        "catanatron_rs_version": _package_version("catanatron-rs"),
        "numpy_version": _package_version("numpy"),
        "networkx_version": _package_version("networkx"),
        "gymnasium_version": _package_version("gymnasium"),
        "zstandard_version": _package_version("zstandard"),
        "scipy_version": _package_version("scipy"),
        "whr_version": _package_version("whr"),
        "torch_version": _package_version("torch"),
        "hostname": socket.gethostname(),
    }
    errors: list[str] = []
    for key in (
        "python_version",
        "catanatron_rs_version",
        "numpy_version",
        "networkx_version",
        "gymnasium_version",
        "zstandard_version",
        "scipy_version",
        "whr_version",
        "torch_version",
    ):
        if actual.get(key) != runtime.get(key):
            errors.append(
                f"runtime {key} drift: expected={runtime.get(key)} actual={actual.get(key)}"
            )
    try:
        import torch

        actual["torch_cuda_version"] = str(torch.version.cuda or "")
        actual["cuda_available"] = bool(torch.cuda.is_available())
        actual["cuda_device_count"] = int(torch.cuda.device_count())
        try:
            actual["cuda_device_names"] = [
                str(torch.cuda.get_device_name(index))
                for index in range(actual["cuda_device_count"])
            ]
        except Exception as error:  # The doctor must refuse instead of crashing.
            actual["cuda_device_names"] = []
            actual["cuda_device_name_error"] = str(error)
            errors.append(f"cannot identify visible CUDA devices: {error}")
    except ImportError:
        actual.update(
            torch_cuda_version=None,
            cuda_available=False,
            cuda_device_count=0,
            cuda_device_names=[],
        )
    if actual["torch_cuda_version"] != runtime.get("torch_cuda_version"):
        errors.append(
            "runtime torch_cuda_version drift: "
            f"expected={runtime.get('torch_cuda_version')} "
            f"actual={actual['torch_cuda_version']}"
        )
    if not actual["cuda_available"]:
        errors.append("CUDA is unavailable under the executing interpreter")
    driver = _nvidia_driver_identity()
    actual["nvidia_driver_versions"] = driver["versions"]
    if driver["versions"] != [runtime.get("nvidia_driver_version")]:
        errors.append(
            "NVIDIA driver drift: "
            f"expected={[runtime.get('nvidia_driver_version')]} actual={driver['versions']}"
        )
    native = _native_runtime_identity()
    actual["native"] = native
    if native.get("wheel_sha256") != runtime.get("catanatron_rs_wheel_sha256"):
        errors.append(
            "native wheel archive drift: "
            f"expected={runtime.get('catanatron_rs_wheel_sha256')} "
            f"actual={native.get('wheel_sha256')}"
        )
    if native.get("extension_sha256") != runtime.get(
        "catanatron_rs_extension_sha256"
    ):
        errors.append(
            "native extension drift: "
            f"expected={runtime.get('catanatron_rs_extension_sha256')} "
            f"actual={native.get('extension_sha256')}"
        )
    missing_capabilities = sorted(
        NATIVE_REQUIRED_CAPABILITIES - set(native.get("capabilities", []))
    )
    if missing_capabilities:
        errors.append(f"native runtime lacks capabilities: {missing_capabilities}")
    if plan["job"]["pipeline"] == "train" and actual["cuda_device_count"] != 8:
        errors.append("canonical training requires exactly 8 visible CUDA devices")
    requested_device_indices: list[int]
    if plan["job"]["pipeline"] == "evaluate":
        requested_device_indices = _cuda_device_indices(plan["job"].get("devices"))
    elif plan["job"]["pipeline"] == "generate" and plan["job"].get("gpu") is not None:
        requested_device_indices = [int(plan["job"]["gpu"])]
    else:
        requested_device_indices = list(range(actual["cuda_device_count"]))
    out_of_range_devices = [
        index
        for index in requested_device_indices
        if index >= actual["cuda_device_count"]
    ]
    if out_of_range_devices:
        errors.append(
            "requested CUDA devices are outside visible device count "
            f"{actual['cuda_device_count']}: {out_of_range_devices}"
        )
    actual["requested_cuda_device_indices"] = requested_device_indices
    actual["requested_cuda_device_names"] = [
        actual["cuda_device_names"][index]
        for index in requested_device_indices
        if index < len(actual["cuda_device_names"])
    ]
    required_accelerator = plan["contract"].get("required_accelerator_model")
    mismatched_devices = [
        name
        for name in actual["requested_cuda_device_names"]
        if isinstance(required_accelerator, str) and required_accelerator not in name
    ]
    if required_accelerator and (
        not actual["requested_cuda_device_names"] or mismatched_devices
    ):
        errors.append(
            "production placement requires only "
            f"{required_accelerator} devices; "
            f"actual={actual['requested_cuda_device_names']}"
        )
    if (
        plan["job"]["pipeline"] == "train"
        and plan["job"]["recipe"] == "a1-current-35m-b200"
        and "plan_receipt" not in plan["inputs"]
    ):
        errors.append(
            "training requires an authenticated plan receipt; "
            "run catan-zero prepare first"
        )
    readiness = plan["readiness"]
    if readiness.get("authorized") is not True:
        errors.append(f"pipeline is blocked: {readiness.get('reason')}")
    try:
        repository = _git_identity(root)
    except ProductionCLIError as error:
        errors.append(str(error))
        repository = {"commit": None, "tracked_changes": [], "clean": False}
    actual["repository"] = repository
    expected_repository = plan["repository"]
    if repository["commit"] != expected_repository["commit"]:
        errors.append("repository HEAD changed after planning")
    if not repository["clean"]:
        errors.append("production runs require a clean tracked worktree")
    errors.extend(_verify_plan_artifacts(plan))
    run_dir = Path(plan["job"]["run_dir"])
    resume = bool(plan["job"].get("resume", False))
    if resume:
        if not run_dir.is_dir() or not any(run_dir.iterdir()):
            errors.append(f"resume requires a non-empty run_dir: {run_dir}")
    elif run_dir.exists():
        errors.append(f"new run_dir already exists: {run_dir}")
    return {
        "schema_version": "catan-zero-production-doctor-v1",
        "ok": not errors,
        "pipeline": plan["job"]["pipeline"],
        "plan_sha256": plan["plan_sha256"],
        "runtime_expected": runtime,
        "runtime_actual": actual,
        "errors": errors,
    }


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _exclusive_run_claim(receipt_path: Path, plan_sha256: str) -> Iterator[None]:
    """Hold one cross-process claim for a run receipt until execution settles."""

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path = receipt_path.with_name(f".{receipt_path.name}.claim")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(claim_path, flags, 0o600)
    except OSError as error:
        raise ProductionCLIError(
            f"cannot open exclusive run claim {claim_path}: {error}"
        ) from error
    locked = False
    try:
        claim_stat = os.fstat(descriptor)
        if not stat.S_ISREG(claim_stat.st_mode) or claim_stat.st_nlink != 1:
            raise ProductionCLIError(
                f"exclusive run claim must be a singly linked regular file: {claim_path}"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ProductionCLIError(
                f"production run is already claimed: {receipt_path}"
            ) from error
        locked = True
        owner = (
            json.dumps(
                {
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "plan_sha256": plan_sha256,
                    "claimed_unix_seconds": time(),
                },
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        offset = 0
        while offset < len(owner):
            offset += os.write(descriptor, owner[offset:])
        os.fsync(descriptor)
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _stable_output_ref(
    path: Path, *, label: str, json_object: bool = False
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Hash one regular output while proving its path and bytes stay stable."""

    descriptor = -1
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ProductionCLIError(
                f"required {label} must be a regular non-symlink file: {path}"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ProductionCLIError(f"required {label} changed while opening: {path}")
        digest = hashlib.sha256()
        captured = bytearray() if json_object else None
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
                if captured is not None:
                    captured.extend(block)
        after = os.fstat(descriptor)
        path_after = path.lstat()
    except OSError as error:
        raise ProductionCLIError(f"cannot attest required {label} {path}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    identity_path_after = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if identity_before != identity_after or identity_after != identity_path_after:
        raise ProductionCLIError(f"required {label} changed while hashing: {path}")
    if after.st_size <= 0:
        raise ProductionCLIError(f"required {label} is empty: {path}")
    payload: dict[str, Any] | None = None
    if captured is not None:
        try:
            decoded = json.loads(captured.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ProductionCLIError(
                f"required {label} is malformed JSON: {path}: {error}"
            ) from error
        if not isinstance(decoded, dict) or not decoded:
            raise ProductionCLIError(
                f"required {label} must be a non-empty JSON object: {path}"
            )
        payload = decoded
    return (
        {
            "path": str(path),
            "sha256": digest.hexdigest(),
            "size_bytes": int(after.st_size),
        },
        payload,
    )


def _require_exact_integer(
    payload: dict[str, Any], key: str, expected: int, *, label: str
) -> None:
    actual = payload.get(key)
    if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
        raise ProductionCLIError(
            f"required {label} {key} drift: expected={expected} actual={actual!r}"
        )


def _require_checkpoint_ref(
    value: object,
    *,
    expected_path: Path,
    expected_sha256: str,
    label: str,
) -> None:
    if not isinstance(value, dict):
        raise ProductionCLIError(f"required {label} checkpoint reference is malformed")
    raw_path = value.get("path")
    if not isinstance(raw_path, str):
        raise ProductionCLIError(f"required {label} checkpoint path is malformed")
    actual_path = Path(raw_path).expanduser().resolve(strict=False)
    if (
        actual_path != expected_path.expanduser().resolve(strict=False)
        or value.get("sha256") != expected_sha256
    ):
        raise ProductionCLIError(f"required {label} checkpoint binding drift")


def _admit_required_outputs(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate and hash the minimum successful artifact set for a pipeline."""

    job = plan["job"]
    pipeline = str(job["pipeline"])
    run_dir = Path(job["run_dir"])
    outputs: dict[str, dict[str, Any]] = {}
    if pipeline == "generate":
        ref, manifest = _stable_output_ref(
            run_dir / "manifest.json", label="generation manifest", json_object=True
        )
        assert manifest is not None
        if Path(str(manifest.get("out_dir", ""))).resolve(strict=False) != run_dir.resolve(
            strict=False
        ):
            raise ProductionCLIError("required generation manifest out_dir binding drift")
        games = int(job["games"])
        _require_exact_integer(
            manifest, "games_requested", games, label="generation manifest"
        )
        _require_exact_integer(
            manifest, "games_completed", games, label="generation manifest"
        )
        _require_exact_integer(
            manifest, "games_failed", 0, label="generation manifest"
        )
        rows = manifest.get("rows")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ProductionCLIError("required generation manifest has no learner rows")
        errors = manifest.get("errors")
        if not isinstance(errors, list) or errors:
            raise ProductionCLIError(
                f"required generation manifest contains errors: {errors!r}"
            )
        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ProductionCLIError("required generation manifest has no output shards")
        outputs["generation_manifest"] = ref
        run_root = run_dir.resolve(strict=False)
        seen_shards: set[Path] = set()
        for index, raw_shard in enumerate(shards):
            if not isinstance(raw_shard, str):
                raise ProductionCLIError(
                    f"required generation shard {index} path is malformed"
                )
            shard = Path(raw_shard).expanduser()
            if not shard.is_absolute():
                raise ProductionCLIError(
                    f"required generation shard {index} path is not absolute: {shard}"
                )
            resolved_shard = shard.resolve(strict=False)
            try:
                resolved_shard.relative_to(run_root)
            except ValueError as error:
                raise ProductionCLIError(
                    f"required generation shard {index} escapes run_dir: {shard}"
                ) from error
            if resolved_shard in seen_shards:
                raise ProductionCLIError(
                    f"required generation shard inventory contains a duplicate: {shard}"
                )
            seen_shards.add(resolved_shard)
            shard_ref, _ = _stable_output_ref(
                shard, label=f"generation shard {index}"
            )
            outputs[f"generation_shard_{index:06d}"] = shard_ref
        return outputs

    if pipeline == "train":
        checkpoint_path = run_dir / "candidate.pt"
        checkpoint_ref, _ = _stable_output_ref(
            checkpoint_path, label="training candidate"
        )
        report_ref, report = _stable_output_ref(
            run_dir / "train.report.json",
            label="training report",
            json_object=True,
        )
        assert report is not None
        report_checkpoint = report.get("checkpoint")
        if not isinstance(report_checkpoint, str) or (
            Path(report_checkpoint).expanduser().resolve(strict=False)
            != checkpoint_path.resolve(strict=False)
        ):
            raise ProductionCLIError("required training report checkpoint binding drift")
        steps = report.get("steps_completed")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ProductionCLIError(
                "required training report has no completed optimizer steps"
            )
        outputs.update(training_candidate=checkpoint_ref, training_report=report_ref)
        if job["recipe"] == "a1-current-35m-b200":
            execution_ref, execution = _stable_output_ref(
                run_dir / "scratch.execution.json",
                label="scratch execution receipt",
                json_object=True,
            )
            assert execution is not None
            unsigned = dict(execution)
            declared_receipt_sha256 = unsigned.pop("receipt_sha256", None)
            if (
                execution.get("schema_version")
                != "a1-coherent-scratch-training-execution-v2"
                or execution.get("status") != "completed"
                or execution.get("returncode") != 0
                or declared_receipt_sha256
                != "sha256:" + canonical_json_sha256(unsigned)
            ):
                raise ProductionCLIError(
                    "required scratch execution receipt is not an authenticated "
                    "completed execution"
                )
            execution_outputs = execution.get("outputs")
            if not isinstance(execution_outputs, dict):
                raise ProductionCLIError(
                    "required scratch execution receipt has no output bindings"
                )
            terminal = execution_outputs.get("terminal_checkpoint")
            training_report = execution_outputs.get("training_report")
            for value, expected, label in (
                (terminal, checkpoint_ref, "scratch terminal checkpoint"),
                (training_report, report_ref, "scratch training report"),
            ):
                if not isinstance(value, dict):
                    raise ProductionCLIError(f"required {label} binding is malformed")
                bound_path = value.get("path")
                if not isinstance(bound_path, str) or (
                    Path(bound_path).resolve(strict=False)
                    != Path(expected["path"]).resolve(strict=False)
                    or value.get("file_sha256") != "sha256:" + expected["sha256"]
                ):
                    raise ProductionCLIError(f"required {label} binding drift")
            outputs["scratch_execution_receipt"] = execution_ref
        return outputs

    evaluation_ref, evaluation = _stable_output_ref(
        run_dir / "evaluation.json", label="evaluation report", json_object=True
    )
    assert evaluation is not None
    errors = evaluation.get("errors")
    games = evaluation.get("games")
    if not isinstance(errors, list) or errors:
        raise ProductionCLIError(f"required evaluation report contains errors: {errors!r}")
    if not isinstance(games, list) or not games:
        raise ProductionCLIError("required evaluation report has no games")
    if job.get("held_out_suite") is None:
        _require_exact_integer(
            evaluation, "pairs_requested", int(job["pairs"]), label="evaluation report"
        )
        _require_exact_integer(
            evaluation,
            "games_played",
            int(job["pairs"]) * 2,
            label="evaluation report",
        )
        if len(games) != int(job["pairs"]) * 2:
            raise ProductionCLIError("required evaluation report game inventory drift")
        candidate_ref = {
            "path": evaluation.get("candidate_checkpoint"),
            "sha256": evaluation.get("candidate_checkpoint_sha256"),
        }
        champion_ref = {
            "path": evaluation.get("baseline_checkpoint"),
            "sha256": evaluation.get("baseline_checkpoint_sha256"),
        }
    else:
        if (
            evaluation.get("schema_version") != "a1-held-out-high-regret-report-v2"
            or evaluation.get("held_out") is not True
        ):
            raise ProductionCLIError("required held-out evaluation report schema drift")
        candidate_ref = evaluation.get("candidate")
        champion_ref = evaluation.get("champion")
    _require_checkpoint_ref(
        candidate_ref,
        expected_path=Path(job["candidate"]),
        expected_sha256=plan["inputs"]["candidate"]["sha256"],
        label="evaluation candidate",
    )
    _require_checkpoint_ref(
        champion_ref,
        expected_path=Path(job["champion"]),
        expected_sha256=plan["inputs"]["champion"]["sha256"],
        label="evaluation champion",
    )
    outputs["evaluation_report"] = evaluation_ref
    return outputs


def execute(plan: dict[str, Any]) -> int:
    check = doctor(plan)
    if not check["ok"]:
        raise ProductionCLIError(
            "production doctor refused run: " + "; ".join(check["errors"])
        )
    receipt_path = Path(plan["run_receipt"])
    with _exclusive_run_claim(receipt_path, str(plan["plan_sha256"])):
        return _execute_claimed(plan, check, receipt_path)


def _execute_claimed(
    plan: dict[str, Any], check: dict[str, Any], receipt_path: Path
) -> int:
    prior_receipt: dict[str, Any] | None = None
    if receipt_path.exists():
        prior_receipt = _load_json_object(receipt_path, label="prior run receipt")
        if prior_receipt.get("schema_version") != RUN_RECEIPT_SCHEMA:
            raise ProductionCLIError("prior run receipt schema drift")
        if not bool(plan["job"].get("resume", False)):
            raise ProductionCLIError(f"run receipt already exists: {receipt_path}")
        if prior_receipt.get("status") not in {"failed", "failed_to_start"}:
            raise ProductionCLIError(
                "resume requires a prior failed receipt, got "
                f"{prior_receipt.get('status')!r}"
            )
        prior_plan = prior_receipt.get("plan")
        if not isinstance(prior_plan, dict) or not isinstance(
            prior_plan.get("job"), dict
        ):
            raise ProductionCLIError("prior run receipt has no valid plan")
        prior_job = dict(prior_plan["job"])
        current_job = dict(plan["job"])
        prior_job.pop("resume", None)
        current_job.pop("resume", None)
        if (
            prior_job != current_job
            or prior_plan.get("inputs") != plan["inputs"]
            or prior_plan.get("contract") != plan["contract"]
            or prior_plan.get("environment") != plan["environment"]
            or prior_plan.get("repository", {}).get("commit")
            != plan["repository"]["commit"]
        ):
            raise ProductionCLIError(
                "resume job, inputs, contract, environment, or commit differ "
                "from the failed attempt"
            )
    receipt: dict[str, Any] = {
        "schema_version": RUN_RECEIPT_SCHEMA,
        "status": "running",
        "started_unix_seconds": time(),
        "plan": plan,
        "doctor": check,
    }
    if prior_receipt is not None:
        receipt["prior_attempt"] = prior_receipt
    _write_json_atomic(receipt_path, receipt)
    try:
        run_dir = Path(plan["job"]["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=bool(plan["job"].get("resume", False)))
        environment = os.environ.copy()
        environment.update(plan["environment"])
        completed = subprocess.run(
            list(plan["command"]),
            cwd=repo_root(),
            env=environment,
            check=False,
        )
    except OSError as error:
        receipt.update(status="failed_to_start", error=str(error))
        _write_json_atomic(receipt_path, receipt)
        raise ProductionCLIError(f"cannot start production command: {error}") from error
    returncode = int(completed.returncode)
    receipt["command_returncode"] = returncode
    if returncode == 0:
        try:
            receipt["outputs"] = _admit_required_outputs(plan)
        except Exception as error:  # noqa: BLE001 - malformed outputs must fail closed.
            returncode = 1
            receipt["output_admission_error"] = (
                f"{type(error).__name__}: {error}"
            )
    receipt["returncode"] = returncode
    receipt["status"] = "complete" if returncode == 0 else "failed"
    receipt["finished_unix_seconds"] = time()
    _write_json_atomic(receipt_path, receipt)
    return returncode


def prepare_training(plan: dict[str, Any]) -> int:
    if (
        plan["job"]["pipeline"] != "train"
        or plan["job"].get("recipe") != "a1-current-35m-b200"
    ):
        raise ProductionCLIError(
            "prepare is only valid for the authenticated scratch recipe"
        )
    command = plan.get("prepare_command")
    if not isinstance(command, list) or not command:
        raise ProductionCLIError("training plan has no prepare command")
    receipt = Path(plan["job"]["plan_receipt"])
    if receipt.exists():
        raise ProductionCLIError(
            f"authenticated plan receipt already exists: {receipt}"
        )
    repository = _git_identity(repo_root())
    errors = _verify_plan_artifacts(plan)
    if not repository["clean"]:
        errors.append("production preparation requires a clean worktree")
    if repository["commit"] != plan["repository"]["commit"]:
        errors.append("repository HEAD changed after planning")
    if errors:
        raise ProductionCLIError("training preparation refused: " + "; ".join(errors))
    environment = os.environ.copy()
    environment.update(plan["environment"])
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root(),
            env=environment,
            check=False,
        )
    except OSError as error:
        raise ProductionCLIError(
            f"cannot start training preparation: {error}"
        ) from error
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="catan-zero",
        description=(
            "One config-first production interface. Large flag-based tools are "
            "historical replay and research engines, not operator APIs."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Show production authorization state.")
    for command in ("plan", "prepare", "doctor", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("job", type=Path, help="Typed production job JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            result = production_status(repo_root())
            print(json.dumps(result, sort_keys=True, indent=2))
            return 0
        plan = build_plan(args.job)
        if args.command == "plan":
            print(json.dumps(plan, sort_keys=True, indent=2))
            return 0
        if args.command == "prepare":
            return prepare_training(plan)
        if args.command == "doctor":
            result = doctor(plan)
            print(json.dumps(result, sort_keys=True, indent=2))
            return 0 if result["ok"] else 2
        return execute(plan)
    except (ProductionCLIError, ProductionContractError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
