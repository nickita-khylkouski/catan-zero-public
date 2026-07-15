#!/usr/bin/env python3
"""Launch the sealed, self-play-only coherent-target R&D corpus on one host.

This executor is deliberately small and narrow.  It is not a replacement for
the production-wave control plane: it accepts only the authenticated
``a1-coherent-target-rd-contract-v1`` intervention, atomically claims its exact
seed lanes, pins one generator to each declared GPU, and records the launched
PIDs/argv.  The contract verifier forbids opponent mixing and adaptive n256 so
the resulting corpus answers one question: do coherent public-belief n128
targets train differently from the legacy PIMC targets?
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import resource
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT / "tools", REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools import a1_target_eligibility_inventory as identity  # noqa: E402
from tools.prelaunch_guard import parse_seed_ledger  # noqa: E402


class ExecutorError(RuntimeError):
    """The sealed R&D transaction cannot be launched exactly."""


REQUIRED_NOFILE_LIMIT = 65_536


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExecutorError(f"{path} must contain a JSON object")
    return value


def _durable_replace(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_receipt(path: Path, value: Mapping[str, Any]) -> None:
    payload = dict(value)
    payload["receipt_sha256"] = _digest(payload)
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if path.exists():
        if path.read_bytes() != data:
            raise ExecutorError(f"immutable launch receipt drift: {path}")
        return
    _durable_replace(path, data, mode=0o444)
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _claim_rows(contract: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    execution = contract["execution"]
    ledger = Path(str(execution["seed_ledger"])).expanduser().resolve(strict=True)
    contract_sha = str(contract["contract_sha256"])
    rows = [
        (
            int(lane["base_seed"]),
            int(lane["base_seed"]) + int(lane["games"]),
            str(lane["claim_label"]),
            str(lane["lane_id"]),
        )
        for lane in execution["lanes"]
    ]
    rendered = [
        f"[{start} – {end}) | target-identity-rd/{lane_id} "
        f"claim={claim} contract={contract_sha}"
        for start, end, claim, lane_id in rows
    ]
    sidecar = ledger.with_name(ledger.name + ".a1-target-rd.lock")
    descriptor = os.open(sidecar, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        before = ledger.read_bytes()
        if before and not before.endswith(b"\n"):
            raise ExecutorError(f"seed ledger {ledger} does not end with a newline")
        live = parse_seed_ledger(ledger)
        own_counts: list[int] = []
        for start, end, claim, lane_id in rows:
            requested = (start, end)
            token = f"claim={claim}"
            own = [
                row
                for row in live
                if (int(row[0]), int(row[1])) == requested
                and token in str(row[2]).split()
            ]
            if len(own) > 1:
                raise ExecutorError(f"ledger repeats own claim for {lane_id}")
            own_counts.append(len(own))
            collisions = [
                row
                for row in live
                if _overlap(requested, (int(row[0]), int(row[1])))
                and row not in own
            ]
            if collisions:
                raise ExecutorError(
                    f"seed lane {lane_id} {requested} overlaps {collisions[:3]}"
                )
        present = sum(own_counts)
        if present not in (0, len(rows)):
            raise ExecutorError(
                f"refusing partial own seed-claim set: {present}/{len(rows)} present"
            )
        status = "already_claimed" if present == len(rows) else "claimed"
        after = before
        if present == 0:
            after = before + b"".join(line.encode("utf-8") + b"\n" for line in rendered)
            _durable_replace(ledger, after, mode=stat.S_IMODE(ledger.stat().st_mode))
        receipt = {
            "status": status,
            "ledger": str(ledger),
            "ledger_before_sha256": "sha256:" + hashlib.sha256(before).hexdigest(),
            "ledger_after_sha256": "sha256:" + hashlib.sha256(after).hexdigest(),
            "claim_count": len(rows),
            "claims_sha256": _digest(rendered),
        }
        return rendered, receipt
    finally:
        os.close(descriptor)


def _run_text(command: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.run(
            list(command),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExecutorError(f"command failed: {list(command)!r}: {error}") from error


def _python_executable(path: Path) -> Path:
    """Authenticate a venv interpreter without resolving away its prefix.

    Virtualenv Python entry points are commonly symlinks to the base
    interpreter.  Executing the resolved target silently drops the venv's
    site-packages, so retain the lexical absolute path after proving that its
    target exists and the entry point is executable.
    """

    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        target = lexical.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(f"cannot resolve Python executable {lexical}: {error}") from error
    if not target.is_file() or not os.access(lexical, os.X_OK):
        raise ExecutorError(f"python is not executable: {lexical}")
    return lexical


def _ensure_worker_fd_limit() -> tuple[int, int]:
    """Raise the inherited soft fd limit required by multi-worker generation."""

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard < REQUIRED_NOFILE_LIMIT:
        raise ExecutorError(
            "hard RLIMIT_NOFILE is below the generator contract: "
            f"hard={hard} required={REQUIRED_NOFILE_LIMIT}"
        )
    if soft < REQUIRED_NOFILE_LIMIT:
        resource.setrlimit(resource.RLIMIT_NOFILE, (REQUIRED_NOFILE_LIMIT, hard))
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < REQUIRED_NOFILE_LIMIT:
        raise ExecutorError(
            "could not raise soft RLIMIT_NOFILE for generation: "
            f"soft={soft} required={REQUIRED_NOFILE_LIMIT}"
        )
    return int(soft), int(hard)


def _preflight(
    contract_path: Path,
    *,
    repo: Path,
    python: Path,
    host_address: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    verified = identity.inspect_rd_contract(contract_path)
    contract = _load(contract_path)
    execution = contract["execution"]
    if host_address != execution["host"]:
        raise ExecutorError(
            f"--host-address {host_address!r} does not match sealed host {execution['host']!r}"
        )
    repo = repo.expanduser().resolve(strict=True)
    python = _python_executable(python)
    contract_repo = contract_path.resolve(strict=True).parents[3]
    if repo != contract_repo:
        raise ExecutorError(
            f"--repo {repo} differs from the repository authenticated by the "
            f"contract path ({contract_repo})"
        )
    generator = repo / "tools/generate_gumbel_selfplay_data.py"
    if not generator.is_file():
        raise ExecutorError(f"generator is missing: {generator}")
    checkpoint = Path(str(contract["producer_checkpoint"]["path"]))
    if _file_sha256(checkpoint) != contract["producer_checkpoint"]["sha256"]:
        raise ExecutorError(f"producer checkpoint hash drift: {checkpoint}")
    output_root = Path(str(execution["output_root"]))
    if output_root.exists():
        raise ExecutorError(f"fresh output root already exists: {output_root}")

    gpu_indices = {
        int(line.strip())
        for line in _run_text(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()
        if line.strip()
    }
    required_gpus = {int(lane["gpu"]) for lane in execution["lanes"]}
    if not required_gpus <= gpu_indices:
        raise ExecutorError(
            f"sealed GPUs are unavailable: required={sorted(required_gpus)}, "
            f"visible={sorted(gpu_indices)}"
        )
    compute_processes = _run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if compute_processes:
        raise ExecutorError(
            "refusing to stack coherent generation on active GPU work: "
            + compute_processes.replace("\n", "; ")
        )
    git_commit = _run_text(["git", "rev-parse", "HEAD"], cwd=repo)
    tracked_diff = _run_text(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo
    )
    return contract, {
        "verified_contract": verified,
        "repo": str(repo),
        "python": str(python),
        "generator": str(generator),
        "generator_sha256": _file_sha256(generator),
        "git_commit": git_commit,
        "tracked_diff_present": bool(tracked_diff),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "required_gpus": sorted(required_gpus),
    }


def _argv(
    contract: Mapping[str, Any],
    lane: Mapping[str, Any],
    *,
    repo: Path,
    python: Path,
) -> list[str]:
    root = Path(str(contract["execution"]["output_root"]))
    output = root / str(lane["lane_id"])
    return [
        str(python),
        str(repo / "tools/generate_gumbel_selfplay_data.py"),
        "--config",
        str(repo / contract["artifacts"]["typed_generation_config"]["path"]),
        "--prelaunch-guard-config",
        str(repo / contract["artifacts"]["generation_guard"]["path"]),
        "--checkpoint",
        str(contract["producer_checkpoint"]["path"]),
        "--out-dir",
        str(output),
        "--base-seed",
        str(lane["base_seed"]),
        "--games",
        str(lane["games"]),
        "--workers",
        str(contract["execution"]["workers_per_gpu"]),
        "--ledger-claim-label",
        str(lane["claim_label"]),
        "--device",
        "cuda",
        "--preserve-search-evidence",
        "--dump-config",
        str(output / "config.registry.jsonl"),
        "--config-purpose",
        str(contract["contract_id"]),
    ]


def execute(
    contract_path: Path,
    *,
    repo: Path,
    python: Path,
    host_address: str,
    go: bool,
) -> dict[str, Any]:
    contract_path = contract_path.expanduser().resolve(strict=True)
    contract, preflight = _preflight(
        contract_path, repo=repo, python=python, host_address=host_address
    )
    repo = repo.expanduser().resolve(strict=True)
    python = _python_executable(python)
    commands = [
        {
            "lane_id": lane["lane_id"],
            "gpu": int(lane["gpu"]),
            "argv": _argv(contract, lane, repo=repo, python=python),
        }
        for lane in contract["execution"]["lanes"]
    ]
    plan = {
        "schema_version": "a1-coherent-target-rd-launch-receipt-v1",
        "status": "dry_run" if not go else "launching",
        "contract": {
            "path": str(contract_path),
            "file_sha256": _file_sha256(contract_path),
            "contract_sha256": contract["contract_sha256"],
        },
        "preflight": preflight,
        "commands": commands,
    }
    if not go:
        plan["plan_sha256"] = _digest(plan)
        return plan

    execution = contract["execution"]
    service = str(execution["mps_service"])
    active = subprocess.run(
        ["systemctl", "is-active", "--quiet", service], check=False
    ).returncode == 0
    if not active:
        _run_text(["sudo", "-n", "systemctl", "start", service])
    if subprocess.run(
        ["systemctl", "is-active", "--quiet", service], check=False
    ).returncode != 0:
        raise ExecutorError(f"MPS service is not active: {service}")

    nofile_soft, nofile_hard = _ensure_worker_fd_limit()

    _rendered_claims, claim_receipt = _claim_rows(contract)
    output_root = Path(str(execution["output_root"]))
    output_root.mkdir(parents=True, exist_ok=False)
    processes: list[tuple[subprocess.Popen[bytes], Any]] = []
    launched: list[dict[str, Any]] = []
    base_env = os.environ.copy()
    base_env.update({str(key): str(value) for key, value in execution["mps_environment"].items()})
    base_env["CATAN_SEED_LEDGER"] = str(execution["seed_ledger"])
    base_env["PYTHONUNBUFFERED"] = "1"
    # The executor may intentionally use a clean source checkout with the
    # already-provisioned production virtualenv.  Bind imports to that exact
    # checkout instead of whichever older catan-zero wheel happens to be
    # installed in the environment.
    import_roots = [str(repo / "src"), str(repo / "tools")]
    inherited_pythonpath = base_env.get("PYTHONPATH")
    if inherited_pythonpath:
        import_roots.append(inherited_pythonpath)
    base_env["PYTHONPATH"] = os.pathsep.join(import_roots)
    try:
        for command in commands:
            lane_id = str(command["lane_id"])
            log_path = output_root / f"{lane_id}.log"
            log_handle = log_path.open("xb")
            environment = dict(base_env)
            environment["CUDA_VISIBLE_DEVICES"] = str(command["gpu"])
            environment["CATAN_LEDGER_CLAIM_ID"] = str(
                next(
                    lane["claim_label"]
                    for lane in execution["lanes"]
                    if lane["lane_id"] == lane_id
                )
            )
            process = subprocess.Popen(
                command["argv"],
                cwd=repo,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((process, log_handle))
            launched.append(
                {
                    **command,
                    "pid": process.pid,
                    "log": str(log_path),
                    "out_dir": str(output_root / lane_id),
                }
            )
        time.sleep(2.0)
        early = [item for item, (process, _handle) in zip(launched, processes) if process.poll() is not None]
        if early:
            raise ExecutorError(
                f"generator exited during launch preamble: {[item['lane_id'] for item in early]}"
            )
    except BaseException:
        for process, _handle in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for _process, handle in processes:
            handle.close()

    plan.update(
        {
            "status": "launched",
            "launched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "mps_service": service,
            "rlimit_nofile": {"soft": nofile_soft, "hard": nofile_hard},
            "claim_receipt": claim_receipt,
            "commands": launched,
        }
    )
    receipt_path = output_root / "launch.receipt.json"
    _write_receipt(receipt_path, plan)
    plan["receipt"] = str(receipt_path)
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPO_ROOT
        / "configs/operations/a1-target-identity-coherent-n128-rd-v1/contract.json",
    )
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--host-address", required=True)
    parser.add_argument(
        "--go",
        action="store_true",
        help="claim the sealed seeds and detach all declared lanes; omitted is read-only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = execute(
            args.contract,
            repo=args.repo,
            python=args.python,
            host_address=args.host_address,
            go=bool(args.go),
        )
    except (ExecutorError, identity.InventoryError, OSError, ValueError) as error:
        print(f"a1_coherent_target_rd_executor: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
