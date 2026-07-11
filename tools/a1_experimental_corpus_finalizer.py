#!/usr/bin/env python3
"""Finalize the one-off A1 80/20 reconstruction without blessing it for promotion.

This is deliberately not part of the production generation executor.  It
collects the opponent-only recovery tranche, selects the lowest complete seeds
per sealed job, and emits trainer-compatible full-arm selection/audit artifacts.
Every receipt is marked ``experimental_nonpromotable``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import a1_pre_wave_contract as contract  # noqa: E402

LABEL = "experimental_nonpromotable"
PLAN_SCHEMA = "a1-experimental-80-20-corpus-plan-v1"
RECEIPT_SCHEMA = "a1-experimental-80-20-corpus-receipt-v1"
SELECTED_SCHEMA = "a1-dual-arm-selected-training-games-v1"
AUDIT_SCHEMA = "a1-dual-arm-post-wave-audit-v1"
QUOTAS = {
    "n128": {
        "current_producer": 112_000,
        "recent_history": 21_000,
        "hard_negative": 7_000,
    },
    "n256": {
        "current_producer": 44_800,
        "recent_history": 8_400,
        "hard_negative": 2_800,
    },
}
SUBSETS = {"n128": "full-140k", "n256": "full-56k"}
REQUIRED_REGIME = "public_conservation_pimc_v1"


class FinalizerError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _seed_sha(seeds: Iterable[int]) -> str:
    values = np.asarray(sorted(set(map(int, seeds))), dtype="<i8")
    return "sha256:" + hashlib.sha256(values.tobytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FinalizerError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise FinalizerError(f"{path} must contain one JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise FinalizerError(f"immutable output drift: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise FinalizerError(f"immutable output drift: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_plan(
    reconstruction: Path, recovery_plan: Path, arm: str, out: Path
) -> dict[str, Any]:
    existing = _load(reconstruction.resolve(strict=True))
    recovery = _load(recovery_plan.resolve(strict=True))
    if existing.get("classification") != LABEL or recovery.get("label") != LABEL:
        raise FinalizerError("both source tranches must be experimental_nonpromotable")
    if arm not in QUOTAS:
        raise FinalizerError(f"unsupported arm {arm!r}")
    current = [
        row for row in existing.get("job_bindings", []) if row.get("arm_id") == arm
    ]
    opponent = [
        command
        for lane in recovery.get("lanes", [])
        if lane.get("arm_id") == arm
        for command in lane.get("commands", [])
    ]
    if len(current) != 28 or len(opponent) != 56:
        raise FinalizerError(f"{arm}: expected 28 current and 56 recovery jobs")
    current_total = sum(
        int(row.get("arm_selected_game_count", row.get("selected_game_count", 0)))
        for row in current
    )
    # Historical reconstruction manifests use a per-arm aggregate and may only
    # carry the matched per-job count.  The full-current plan is authoritative.
    declared_current = int(
        existing.get("arms", {}).get(arm, {}).get("arm_selected_games", -1)
    )
    if declared_current != QUOTAS[arm]["current_producer"]:
        raise FinalizerError(f"{arm}: current aggregate quota drift")
    by_category = Counter(row.get("category") for row in opponent)
    expected_jobs = {"recent_history": 28, "hard_negative": 28}
    if by_category != expected_jobs:
        raise FinalizerError(
            f"{arm}: recovery category job count drift: {dict(by_category)}"
        )
    contract_ref = recovery["source_artifacts"][arm]["lock"]
    render_ref = recovery["source_artifacts"][arm]["render"]
    value = {
        "schema_version": PLAN_SCHEMA,
        "classification": LABEL,
        "production_eligible": False,
        "arm_id": arm,
        "subset_id": SUBSETS[arm],
        "quotas": QUOTAS[arm],
        "contract": contract_ref,
        "contract_sha256": recovery["source_artifacts"][arm]["lock_sha256"],
        "render": render_ref,
        "render_sha256": recovery["source_artifacts"][arm]["render_sha256"],
        "reconstruction": {
            "path": str(reconstruction.resolve()),
            "sha256": _file_sha(reconstruction),
        },
        "recovery_plan": {
            "path": str(recovery_plan.resolve()),
            "sha256": _file_sha(recovery_plan),
        },
        "source_recovery_plan_sha256": recovery["plan_sha256"],
        "recovery_lanes": [
            {
                "lane_id": lane["lane_id"],
                "host_alias": lane["host_alias"],
                "receipt": lane["receipt"],
                "job_ids": [row["job_id"] for row in lane["commands"]],
            }
            for lane in recovery.get("lanes", [])
            if lane.get("arm_id") == arm
        ],
        "current_jobs": current,
        "recovery_jobs": opponent,
        "unused_current_sum": current_total,
    }
    value["plan_sha256"] = _digest(value)
    _atomic_json(out, value)
    return value


_REMOTE_RECEIPT_PROGRAM = r"""
import json, pathlib, sys
expected_plan = sys.argv[1]
rows = json.loads(sys.argv[2])
result = []
for row in rows:
    path = pathlib.Path(row['receipt'])
    try:
        value = json.loads(path.read_text())
    except Exception as error:
        result.append({'lane_id': row['lane_id'], 'status': 'missing_or_invalid', 'detail': str(error)})
        continue
    jobs = value.get('jobs', {})
    ok = (
        value.get('schema_version') == 'a1-r1-opponent-recovery-lane-receipt-v1'
        and value.get('label') == 'experimental_nonpromotable'
        and value.get('promotable') is False
        and value.get('plan_sha256') == expected_plan
        and value.get('lane_id') == row['lane_id']
        and value.get('status') == 'complete'
        and set(jobs) == set(row['job_ids'])
        and all(item.get('status') == 'complete' and item.get('return_code') == 0 for item in jobs.values())
    )
    result.append({'lane_id': row['lane_id'], 'status': 'complete' if ok else str(value.get('status', 'invalid'))})
print(json.dumps(result, sort_keys=True))
"""


def remote_completion(plan_path: Path, ssh_command: Sequence[str]) -> dict[str, Any]:
    """Read and validate all arm lane receipts without touching output bytes."""
    plan = _verified_plan(plan_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for lane in plan["recovery_lanes"]:
        grouped.setdefault(lane["host_alias"], []).append(lane)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(10, len(grouped))) as executor:
        futures = {
            executor.submit(
                subprocess.run,
                [
                    *ssh_command,
                    host,
                    " ".join(
                        shlex.quote(value)
                        for value in (
                            "python3",
                            "-c",
                            _REMOTE_RECEIPT_PROGRAM,
                            plan["source_recovery_plan_sha256"],
                            json.dumps(sorted(lanes, key=lambda row: row["lane_id"])),
                        )
                    ),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            ): host
            for host, lanes in sorted(grouped.items())
        }
        for future in as_completed(futures):
            host = futures[future]
            response = future.result()
            if response.returncode != 0:
                raise FinalizerError(
                    f"remote receipt preflight failed on {host}: {response.stderr[-2000:]}"
                )
            try:
                rows = json.loads(response.stdout)
            except json.JSONDecodeError as error:
                raise FinalizerError(
                    f"remote receipt response malformed on {host}"
                ) from error
            if not isinstance(rows, list):
                raise FinalizerError(f"remote receipt response malformed on {host}")
            results.extend({**row, "host_alias": host} for row in rows)
    results.sort(key=lambda row: row["lane_id"])
    expected = sorted(row["lane_id"] for row in plan["recovery_lanes"])
    if [row.get("lane_id") for row in results] != expected:
        raise FinalizerError("remote receipt response does not cover exact arm lanes")
    complete = sum(row.get("status") == "complete" for row in results)
    return {
        "classification": LABEL,
        "arm_id": plan["arm_id"],
        "complete": complete,
        "total": len(expected),
        "ready": complete == len(expected),
        "lanes": results,
    }


def wait_ready(
    plan_path: Path,
    ssh_command: Sequence[str],
    *,
    poll_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    if poll_seconds <= 0 or timeout_seconds < 0:
        raise FinalizerError("poll seconds must be positive and timeout nonnegative")
    deadline = time.monotonic() + timeout_seconds
    while True:
        result = remote_completion(plan_path, ssh_command)
        if result["ready"]:
            return result
        if time.monotonic() >= deadline:
            raise FinalizerError(
                f"remote recovery is incomplete: {result['complete']}/{result['total']} lanes"
            )
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def _verified_plan(path: Path) -> dict[str, Any]:
    value = _load(path.resolve(strict=True))
    stated = value.get("plan_sha256")
    unhashed = dict(value)
    unhashed.pop("plan_sha256", None)
    if (
        value.get("schema_version") != PLAN_SCHEMA
        or value.get("classification") != LABEL
        or stated != _digest(unhashed)
    ):
        raise FinalizerError("experimental corpus plan drift")
    if value.get("production_eligible") is not False or value.get(
        "quotas"
    ) != QUOTAS.get(value.get("arm_id")):
        raise FinalizerError("experimental corpus plan authority drift")
    for key in ("reconstruction", "recovery_plan", "contract", "render"):
        ref = value[key]
        path_ref = Path(ref["path"]).resolve(strict=True)
        if _file_sha(path_ref) != ref["sha256"]:
            raise FinalizerError(f"plan-bound input drift: {key}")
    return value


def _harvest_job(
    destination: Path, ssh_command: Sequence[str], job: dict[str, Any]
) -> dict[str, Any]:
    state = destination / ".state"
    state.mkdir(exist_ok=True)
    job_id = job["job_id"]
    final = destination / "jobs" / job_id
    receipt = state / f"{job_id}.json"
    if receipt.exists():
        prior = _load(receipt)
        if prior.get("job_id") != job_id or not final.is_dir():
            raise FinalizerError(f"invalid resumed harvest state for {job_id}")
        actual = []
        for file in sorted(path for path in final.rglob("*") if path.is_file()):
            actual.append(
                {
                    "path": file.relative_to(final).as_posix(),
                    "bytes": file.stat().st_size,
                    "sha256": _file_sha(file),
                }
            )
        if actual != prior.get("files") or _digest(actual) != prior.get("files_sha256"):
            raise FinalizerError(f"resumed harvested bytes drifted for {job_id}")
        return prior
    incoming = destination / ".incoming" / job_id
    if not final.exists():
        incoming.mkdir(parents=True, exist_ok=True)
        remote_shell = " ".join(map(str, ssh_command))
        result = subprocess.run(
            [
                "rsync",
                "-a",
                "--partial",
                "--append-verify",
                "--protect-args",
                "-e",
                remote_shell,
                f"{job['host_alias']}:{job['output_dir'].rstrip('/')}/",
                str(incoming) + "/",
            ],
            check=False,
        )
        if result.returncode != 0:
            raise FinalizerError(f"rsync failed for {job_id}: exit {result.returncode}")
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(incoming, final)
    files = []
    for path in sorted(p for p in final.rglob("*") if p.is_file()):
        files.append(
            {
                "path": path.relative_to(final).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _file_sha(path),
            }
        )
    if not files or not any(row["path"].endswith(".npz") for row in files):
        raise FinalizerError(f"harvested job has no NPZ shards: {job_id}")
    row = {
        "job_id": job_id,
        "host_alias": job["host_alias"],
        "source_dir": job["output_dir"],
        "files": files,
        "files_sha256": _digest(files),
    }
    _atomic_json(receipt, row)
    return row


def _host_round_robin_jobs(jobs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a deterministic schedule that spreads initial work across hosts.

    ``ThreadPoolExecutor`` starts work in submission order.  Sorting solely by
    job id can therefore fill the entire worker pool from the first host when
    job ids are host-contiguous, serializing the available source links.  Keep
    each host's jobs ordered for reproducibility, but interleave the sorted host
    queues so every parallel wave uses as many distinct sources as possible.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        grouped.setdefault(str(job["host_alias"]), []).append(job)
    for host_jobs in grouped.values():
        host_jobs.sort(key=lambda row: row["job_id"])
    hosts = sorted(grouped)
    scheduled: list[dict[str, Any]] = []
    index = 0
    while True:
        wave = [grouped[host][index] for host in hosts if index < len(grouped[host])]
        if not wave:
            return scheduled
        scheduled.extend(wave)
        index += 1


def harvest(
    plan_path: Path,
    destination: Path,
    ssh_command: Sequence[str],
    *,
    parallelism: int = 10,
) -> dict[str, Any]:
    """Crash-resumable bounded-parallel harvest of the complete recovery tranche."""
    if (
        isinstance(parallelism, bool)
        or not isinstance(parallelism, int)
        or not 1 <= parallelism <= 32
    ):
        raise FinalizerError("parallelism must be an integer in [1, 32]")
    plan = _verified_plan(plan_path)
    completion = remote_completion(plan_path, ssh_command)
    if not completion["ready"]:
        raise FinalizerError(
            f"refusing partial harvest: {completion['complete']}/{completion['total']} remote lanes complete"
        )
    destination = destination.absolute()
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / ".harvest.lock"
    lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FinalizerError("another harvest owns this destination") from error
        jobs = sorted(plan["recovery_jobs"], key=lambda row: row["job_id"])
        scheduled_jobs = _host_round_robin_jobs(jobs)
        inventory_by_job: dict[str, dict[str, Any]] = {}
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=min(parallelism, len(jobs))) as executor:
            futures = {
                executor.submit(_harvest_job, destination, ssh_command, job): job
                for job in scheduled_jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    inventory_by_job[job["job_id"]] = future.result()
                except BaseException as error:  # drain every in-flight transfer
                    failures.append(f"{job['job_id']}: {error}")
        if failures:
            raise FinalizerError(
                "parallel harvest failed; completed jobs are resumable: "
                + "; ".join(sorted(failures))
            )
        inventory = [inventory_by_job[job["job_id"]] for job in jobs]
    finally:
        os.close(lock_descriptor)
    receipt_value = {
        "schema_version": RECEIPT_SCHEMA,
        "classification": LABEL,
        "production_eligible": False,
        "arm_id": plan["arm_id"],
        "plan_sha256": plan["plan_sha256"],
        "remote_completion_sha256": _digest(completion),
        "jobs": inventory,
        "jobs_sha256": _digest(inventory),
    }
    receipt_value["receipt_sha256"] = _digest(receipt_value)
    _atomic_json(destination / "harvest.receipt.json", receipt_value)
    return receipt_value


def _npz_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.npz") if path.is_file())


def _link_job_sources(
    source: Path, destination: Path, *, job_id: str, host_alias: str
) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    destination.mkdir(parents=True, exist_ok=True)
    attestation = source / "a1_contract.json"
    if attestation.is_file():
        linked_attestation = destination / "a1_contract.json"
        if not linked_attestation.exists():
            os.link(attestation, linked_attestation, follow_symlinks=False)
        if _file_sha(linked_attestation) != _file_sha(attestation):
            raise FinalizerError(f"hardlinked attestation bytes drift: {attestation}")
    for index, original in enumerate(_npz_files(source)):
        linked = destination / f"shard_{index:06d}.npz"
        if not linked.exists():
            try:
                os.link(original, linked, follow_symlinks=False)
            except OSError as error:
                raise FinalizerError(
                    "experimental relocation requires current/recovery/out on the "
                    f"same filesystem; cannot hardlink {original}: {error}"
                ) from error
        if linked.is_symlink() or not linked.is_file():
            raise FinalizerError(f"unsafe experimental relocation path: {linked}")
        original_sha = _file_sha(original)
        linked_sha = _file_sha(linked)
        if (
            linked_sha != original_sha
            or linked.stat().st_size != original.stat().st_size
        ):
            raise FinalizerError(f"hardlinked source bytes drift: {original}")
        files.append(
            {
                "source_path": str(original.resolve()),
                "relative_path": linked.relative_to(
                    destination.parent.parent
                ).as_posix(),
                "size_bytes": linked.stat().st_size,
                "sha256": linked_sha,
                "job_id": job_id,
                "host_alias": host_alias,
            }
        )
    if not files:
        raise FinalizerError(f"source job contains no NPZ shards: {source}")
    return files


def _verify_job_attestation(
    root: Path,
    job: dict[str, Any],
    *,
    contract_sha256: str,
    selected_quota: int,
) -> None:
    path = root / "a1_contract.json"
    if not path.is_file():
        if job["category"] == "current_producer":
            # The 80% reconstruction intentionally retained NPZ bytes and a
            # separately hashed inventory, not every copied sidecar.
            return
        raise FinalizerError(f"recovery job lacks A1 attestation: {path}")
    value = _load(path)
    required_hashes = (
        "producer_checkpoint_sha256",
        "search_operator_sha256",
        "effective_search_config_sha256",
        "evaluator_sha256",
        "runtime_code_tree_sha256",
    )
    if (
        value.get("schema_version") != "a1-generation-job-attestation-v2"
        or value.get("contract_sha256") != contract_sha256
        or value.get("job_id") != job["job_id"]
        or value.get("worker_id") != job["worker_id"]
        or value.get("category") != job["category"]
        or value.get("arm_id") != job["arm_id"]
        or value.get("games") != selected_quota
        or value.get("producer_checkpoint_sha256") != job["producer_checkpoint_sha256"]
        or value.get("opponent_checkpoint_sha256") != job["opponent_checkpoint_sha256"]
        or any(
            not isinstance(value.get(field), str)
            or not value[field].startswith("sha256:")
            for field in required_hashes
        )
    ):
        raise FinalizerError(f"job attestation/config/checkpoint drift: {path}")
    expected_attempts = job.get("max_attempts")
    if expected_attempts is not None and value.get("attempts") != expected_attempts:
        raise FinalizerError(f"job attempt quota drift: {path}")


def _job_evidence(
    root: Path, job: dict[str, Any], quota: int
) -> tuple[list[int], dict[int, int], list[dict[str, Any]]]:
    statuses: dict[int, tuple[bool, bool]] = {}
    rows_by_seed: Counter[int] = Counter()
    shards: list[dict[str, Any]] = []
    for shard in _npz_files(root):
        with np.load(shard, allow_pickle=False) as payload:
            seeds = np.asarray(payload["game_seed"], dtype=np.int64)
            terminated = np.asarray(payload["terminated"], dtype=bool)
            truncated = np.asarray(payload["truncated"], dtype=bool)
            regime = np.asarray(payload["target_information_regime"]).astype(str)
            if not (
                seeds.ndim == 1
                and seeds.shape == terminated.shape == truncated.shape == regime.shape
            ):
                raise FinalizerError(f"unaligned status arrays in {shard}")
            if np.any(regime != REQUIRED_REGIME):
                raise FinalizerError(f"public-information regime drift in {shard}")
            if job["category"] != "current_producer":
                if "opponent_tag" not in payload:
                    raise FinalizerError(f"opponent rows lack category tag in {shard}")
                tags = np.asarray(payload["opponent_tag"]).astype(str)
                if tags.shape != seeds.shape or np.any(tags != job["category"]):
                    raise FinalizerError(f"opponent category tag drift in {shard}")
            elif "opponent_tag" in payload:
                tags = np.asarray(payload["opponent_tag"]).astype(str)
                if tags.shape != seeds.shape or np.any(tags != ""):
                    raise FinalizerError(
                        f"current-producer opponent tag drift in {shard}"
                    )
            for seed in np.unique(seeds):
                mask = seeds == seed
                values = set(zip(terminated[mask].tolist(), truncated[mask].tolist()))
                if len(values) != 1:
                    raise FinalizerError(
                        f"status drift for seed {int(seed)} in {shard}"
                    )
                value = next(iter(values))
                if int(seed) in statuses and statuses[int(seed)] != value:
                    raise FinalizerError(
                        f"cross-shard status drift for seed {int(seed)}"
                    )
                statuses[int(seed)] = value
            rows_by_seed.update(map(int, seeds.tolist()))
            shards.append(
                {
                    "kind": "data_shard",
                    "path": str(shard.resolve()),
                    "sha256": _file_sha(shard),
                    "job_id": job["job_id"],
                    "category": job["category"],
                    "producer_checkpoint_sha256": job["producer_checkpoint_sha256"],
                    "opponent_checkpoint_sha256": job["opponent_checkpoint_sha256"],
                }
            )
    complete = sorted(
        seed for seed, status in statuses.items() if status == (True, False)
    )
    selected = complete[:quota]
    if len(selected) != quota:
        raise FinalizerError(
            f"{job['job_id']}: only {len(selected)} complete games for quota {quota}"
        )
    return selected, {seed: rows_by_seed[seed] for seed in selected}, shards


def finalize(
    plan_path: Path, current_root: Path, recovery_root: Path, out: Path
) -> dict[str, Any]:
    plan = _verified_plan(plan_path)
    arm = plan["arm_id"]
    quotas = plan["quotas"]
    records: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    rows_by_selected_seed: dict[int, int] = {}
    current_by_id = {row["job_id"]: row for row in plan["current_jobs"]}
    recovery_by_id = {row["job_id"]: row for row in plan["recovery_jobs"]}
    all_jobs: list[tuple[dict[str, Any], Path]] = []
    for job_id, source in current_by_id.items():
        all_jobs.append((source, current_root / arm / job_id))
    for job_id, source in recovery_by_id.items():
        all_jobs.append((source, recovery_root / "jobs" / job_id))
    relocated = out / "relocated"
    relocation_files: list[dict[str, Any]] = []
    relocated_jobs: list[tuple[dict[str, Any], Path]] = []
    for raw, source_root in all_jobs:
        job_root = relocated / "jobs" / raw["job_id"]
        relocation_files.extend(
            _link_job_sources(
                source_root,
                job_root,
                job_id=raw["job_id"],
                host_alias=str(raw.get("host_alias", "reconstructed-current")),
            )
        )
        relocated_jobs.append((raw, job_root))
    all_jobs = relocated_jobs
    per_category_job_quota = {key: value // 28 for key, value in quotas.items()}
    seen: set[int] = set()
    for raw, root in sorted(all_jobs, key=lambda pair: pair[0]["job_id"]):
        category = raw.get("category", "current_producer")
        job = {
            "job_id": raw["job_id"],
            "worker_id": raw.get("worker_id", raw["job_id"].split("__")[0]),
            "category": category,
            "producer_checkpoint_sha256": plan["contract_sha256"]
            if category == "current_producer"
            else raw.get("producer_checkpoint_sha256", plan["contract_sha256"]),
            "opponent_checkpoint_sha256": sorted(
                set(raw.get("opponent_checkpoint_sha256", [plan["contract_sha256"]]))
            ),
            "arm_id": arm,
            "max_attempts": raw.get("max_attempts"),
        }
        # Recovery plan has authoritative checkpoint refs in argv-bound manifests;
        # use the lock's checkpoint identities when compact fields are absent.
        lock = contract.verify_lock(Path(plan["contract"]["path"]))
        producer = next(
            row["sha256"] for row in lock["checkpoints"] if row["role"] == "producer"
        )
        opponent = (
            [producer]
            if category == "current_producer"
            else [
                next(
                    row["sha256"]
                    for row in lock["checkpoints"]
                    if row["role"]
                    == ("history" if category == "recent_history" else "hard_negative")
                )
            ]
        )
        job["producer_checkpoint_sha256"] = producer
        job["opponent_checkpoint_sha256"] = opponent
        _verify_job_attestation(
            root,
            job,
            contract_sha256=plan["contract_sha256"],
            selected_quota=per_category_job_quota[category],
        )
        selected, job_rows_by_seed, job_shards = _job_evidence(
            root, job, per_category_job_quota[category]
        )
        overlap = seen.intersection(selected)
        if overlap:
            raise FinalizerError(
                f"duplicate selected seeds across jobs: {len(overlap)}"
            )
        seen.update(selected)
        rows_by_selected_seed.update(job_rows_by_seed)
        shards.extend(job_shards)
        records.extend(
            {
                "game_seed": seed,
                "job_id": job["job_id"],
                "worker_id": job["worker_id"],
                "category": category,
                "producer_checkpoint_sha256": producer,
                "opponent_checkpoint_sha256": opponent,
                "arm_id": arm,
                "split": "train",
            }
            for seed in selected
        )
    if Counter(row["category"] for row in records) != Counter(quotas):
        raise FinalizerError("combined selected category quotas drift")
    rng = np.random.default_rng(17)
    validation_count = max(1, round(len(records) * 0.05))
    validation = set(
        map(
            int,
            rng.permutation(np.asarray(sorted(seen), dtype=np.int64))[
                :validation_count
            ],
        )
    )
    for row in records:
        row["split"] = "validation" if row["game_seed"] in validation else "train"
    records.sort(key=lambda row: (row["game_seed"], row["job_id"]))
    training = [row["game_seed"] for row in records if row["split"] == "train"]
    validation_seeds = [
        row["game_seed"] for row in records if row["split"] == "validation"
    ]
    selected_rows = sum(rows_by_selected_seed.values())
    validation_rows = sum(rows_by_selected_seed[seed] for seed in validation_seeds)
    out.mkdir(parents=True, exist_ok=True)
    selected = {
        "schema_version": SELECTED_SCHEMA,
        "arm_id": arm,
        "subset_id": SUBSETS[arm],
        "a1_contract_sha256": plan["contract_sha256"],
        "selection_rule": "lowest_seed_complete_per_job",
        "selected_game_count": len(records),
        "selected_game_seed_set_sha256": _seed_sha(seen),
        "category_game_counts": quotas,
        "training_game_count": len(training),
        "training_game_seed_set_sha256": _seed_sha(training),
        "validation_game_count": len(validation_seeds),
        "validation_game_seed_set_sha256": _seed_sha(validation_seeds),
        "records_sha256": _digest(records),
        "records": records,
        "parent_manifest_sha256": None,
    }
    selected_path = out / f"{arm}.selected_games.json"
    _atomic_json(selected_path, selected)
    validation_payload = {
        "schema_version": "train-validation-game-seeds-v1",
        "a1_contract_sha256": plan["contract_sha256"],
        "validation_fraction": 0.05,
        "validation_seed": 17,
        "validation_max_samples": 0,
        "validation_game_seed_ranges": [],
        "validation_game_seed_count": len(validation_seeds),
        "validation_row_count": validation_rows,
        "validation_game_seed_set_sha256": _seed_sha(validation_seeds),
        "game_seeds": sorted(validation_seeds),
    }
    validation_path = out / f"{arm}.validation_seeds.json"
    _atomic_json(validation_path, validation_payload)
    job_identities = [
        {
            "job_id": raw["job_id"],
            "worker_id": raw.get("worker_id", raw["job_id"].split("__")[0]),
            "host_alias": str(raw.get("host_alias", "reconstructed-current")),
            "gpu": int(raw.get("gpu", -1)),
            "category": raw.get("category", "current_producer"),
            "output_dir": str(root),
            "arm_id": arm,
        }
        for raw, root in all_jobs
    ]
    relocation = {
        "schema_version": "a1-fleet-harvest-relocation-v1",
        "arm_id": arm,
        "contract_path": plan["contract"]["path"],
        "contract_file_sha256": plan["contract"]["sha256"],
        "contract_sha256": plan["contract_sha256"],
        "render_path": plan["render"]["path"],
        "render_file_sha256": plan["render"]["sha256"],
        "render_sha256": plan["render_sha256"],
        "host_count": len(set(row["host_alias"] for row in job_identities)),
        "job_count": len(job_identities),
        "job_identities": job_identities,
        "job_identities_sha256": _digest(job_identities),
        "files": relocation_files,
        "file_inventory_sha256": _digest(relocation_files),
        "classification": LABEL,
        "production_eligible": False,
    }
    relocation["relocation_sha256"] = _digest(relocation)
    relocation_path = relocated / "relocation_map.json"
    _atomic_json(relocation_path, relocation)
    audit = {
        "schema_version": AUDIT_SCHEMA,
        "classification": LABEL,
        "production_eligible": False,
        "arm_id": arm,
        "subset_id": SUBSETS[arm],
        "contract_sha256": plan["contract_sha256"],
        "passed": True,
        "errors": [],
        "category_game_counts": quotas,
        "rows": selected_rows,
        "selection_rule": "lowest_seed_complete_per_job",
        "shards": shards,
        "shard_inventory_sha256": _digest(shards),
        "source_provenance": {
            category: {
                "producer_checkpoint_sha256": next(
                    row["producer_checkpoint_sha256"]
                    for row in records
                    if row["category"] == category
                ),
                "opponent_checkpoint_sha256": next(
                    row["opponent_checkpoint_sha256"]
                    for row in records
                    if row["category"] == category
                ),
            }
            for category in quotas
        },
        "selected_training_games": {
            "manifest": str(selected_path.resolve()),
            "manifest_sha256": _digest(selected),
            "manifest_file_sha256": _file_sha(selected_path),
            "selected_game_count": len(records),
            "selected_game_seed_set_sha256": selected["selected_game_seed_set_sha256"],
            "records_sha256": selected["records_sha256"],
        },
        "validation_holdout": {
            "manifest": str(validation_path.resolve()),
            "manifest_sha256": _digest(validation_payload),
            "manifest_file_sha256": _file_sha(validation_path),
            "validation_game_seed_count": len(validation_seeds),
            "validation_game_seed_set_sha256": validation_payload[
                "validation_game_seed_set_sha256"
            ],
        },
        "harvest_relocation": {
            "arm_id": arm,
            "path": str(relocation_path.resolve()),
            "file_sha256": _file_sha(relocation_path),
            "relocation_sha256": relocation["relocation_sha256"],
            "render_sha256": relocation["render_sha256"],
            "job_identities_sha256": relocation["job_identities_sha256"],
            "file_inventory_sha256": relocation["file_inventory_sha256"],
        },
    }
    audit["audit_sha256"] = _digest(audit)
    audit_path = out / f"{arm}.audit.json"
    _atomic_json(audit_path, audit)
    source_list_path = out / f"{arm}.sources.txt"
    source_lines = [str(root.resolve()) for _job, root in all_jobs]
    _atomic_text(source_list_path, "\n".join(source_lines) + "\n")
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "classification": LABEL,
        "production_eligible": False,
        "arm_id": arm,
        "plan_sha256": plan["plan_sha256"],
        "selected_manifest": {
            "path": str(selected_path),
            "sha256": _file_sha(selected_path),
        },
        "audit": {"path": str(audit_path), "sha256": _file_sha(audit_path)},
        "validation": {
            "path": str(validation_path),
            "sha256": _file_sha(validation_path),
        },
        "source_list": {
            "path": str(source_list_path),
            "sha256": _file_sha(source_list_path),
            "source_count": len(source_lines),
        },
        "build_memmap_argv": [
            sys.executable,
            str(ROOT / "tools" / "build_memmap_corpus.py"),
            "--source-list",
            str(source_list_path),
            "--out",
            str(out / f"{arm}.memmap"),
            "--selected-game-seed-manifest",
            str(selected_path),
            "--a1-post-wave-audit",
            str(audit_path),
        ],
    }
    receipt["receipt_sha256"] = _digest(receipt)
    _atomic_json(out / f"{arm}.receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--reconstruction", required=True)
    p.add_argument("--recovery-plan", required=True)
    p.add_argument("--arm", choices=sorted(QUOTAS), required=True)
    p.add_argument("--out", required=True)
    h = sub.add_parser("harvest")
    h.add_argument("--plan", required=True)
    h.add_argument("--destination", required=True)
    h.add_argument("--ssh-command", default="ssh")
    h.add_argument("--parallelism", type=int, default=10)
    w = sub.add_parser("wait-ready")
    w.add_argument("--plan", required=True)
    w.add_argument("--ssh-command", default="ssh")
    w.add_argument("--poll-seconds", type=float, default=30.0)
    w.add_argument("--timeout-seconds", type=float, default=0.0)
    f = sub.add_parser("finalize")
    f.add_argument("--plan", required=True)
    f.add_argument("--current-root", required=True)
    f.add_argument("--recovery-root", required=True)
    f.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = build_plan(
                Path(args.reconstruction),
                Path(args.recovery_plan),
                args.arm,
                Path(args.out),
            )
        elif args.command == "harvest":
            result = harvest(
                Path(args.plan),
                Path(args.destination),
                shlex.split(args.ssh_command),
                parallelism=args.parallelism,
            )
        elif args.command == "wait-ready":
            result = wait_ready(
                Path(args.plan),
                shlex.split(args.ssh_command),
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            result = finalize(
                Path(args.plan),
                Path(args.current_root),
                Path(args.recovery_root),
                Path(args.out),
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (FinalizerError, contract.ContractError) as error:
        parser.exit(2, f"REFUSED: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
