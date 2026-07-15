#!/usr/bin/env python3
"""Repair a tiny number of truncated games in the sealed coherent n128 R&D corpus.

This is intentionally an experimental-only side transaction.  It never edits the
original corpus and it does not weaken the production collector's zero-truncation
contract.  Instead it launches fresh, out-of-range replacement seeds with the
exact source/runtime/configuration recorded by the original launch, then emits a
separate repaired corpus view and a receipt binding every excluded and included
byte.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fleet import a1_coherent_target_rd_executor as executor  # noqa: E402
from tools import a1_target_eligibility_inventory as identity  # noqa: E402


PLAN_SCHEMA = "a1-coherent-target-rd-experimental-repair-plan-v1"
LAUNCH_SCHEMA = "a1-coherent-target-rd-experimental-repair-launch-v1"
RECEIPT_SCHEMA = "a1-coherent-target-rd-experimental-repair-receipt-v1"
SELECTED_SCHEMA = "a1-coherent-target-rd-repaired-selected-games-v1"
CLASSIFICATION = "experimental_nonpromotable"


class RepairError(RuntimeError):
    """The experimental repair cannot be proven equivalent and complete."""


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
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RepairError(f"cannot load {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RepairError(f"{path} must contain one JSON object")
    return payload


def _write_signed(path: Path, payload: Mapping[str, Any], *, field: str) -> None:
    value = dict(payload)
    value[field] = _digest(value)
    rendered = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists():
        if path.is_symlink() or path.read_bytes() != rendered:
            raise RepairError(f"immutable output drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_signed(path: Path, *, schema: str, field: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise RepairError(f"signed artifact is not a regular file: {resolved}")
    payload = _load(resolved)
    unsigned = dict(payload)
    stated = unsigned.pop(field, None)
    if payload.get("schema_version") != schema or stated != _digest(unsigned):
        raise RepairError(f"signed artifact schema/digest drift: {resolved}")
    return payload


def _seed_set_sha256(seeds: Sequence[int]) -> str:
    values = np.asarray(sorted(map(int, seeds)), dtype="<i8")
    return "sha256:" + hashlib.sha256(values.tobytes()).hexdigest()


def _synthetic_contract(
    contract: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    value = json.loads(json.dumps(contract))
    replacement = list(map(int, plan["replacement_seeds"]))
    lanes = [
        {
            "lane_id": f"replacement_{index:03d}",
            "host": str(plan["host_address"]),
            "gpu": int(gpu),
            "base_seed": seed,
            "games": 1,
            "claim_label": (
                f"a1-coherent-n128-experimental-repair:{plan['plan_sha256'][7:19]}:"
                f"{index:03d}"
            ),
        }
        for index, (seed, gpu) in enumerate(
            zip(replacement, plan["replacement_gpus"], strict=True)
        )
    ]
    value["execution"] = {
        **value["execution"],
        "workers_per_gpu": 1,
        "games_per_gpu": 1,
        "total_games": len(lanes),
        "seed_start": min(replacement),
        "seed_end": max(replacement) + 1,
        "output_root": str(plan["repair_root"]),
        "lanes": lanes,
    }
    return value


def build_plan(
    *,
    contract_path: Path,
    host_address: str,
    bad_seeds: Sequence[int],
    replacement_seeds: Sequence[int],
    replacement_gpus: Sequence[int],
    repair_root: Path,
    write: Path,
) -> dict[str, Any]:
    contract_path = contract_path.expanduser().resolve(strict=True)
    verified = identity.inspect_rd_contract(contract_path)
    contract = _load(contract_path)
    _contract, launch, launch_file_sha256, _commands = executor._authenticate_launch(  # noqa: SLF001
        contract_path, host_address=host_address
    )
    bad = sorted(set(map(int, bad_seeds)))
    replacement = sorted(set(map(int, replacement_seeds)))
    gpus = list(map(int, replacement_gpus))
    start = int(contract["execution"]["seed_start"])
    end = int(contract["execution"]["seed_end"])
    expected = set(range(start, end))
    if not bad or any(seed not in expected for seed in bad):
        raise RepairError("every bad seed must belong to the original sealed interval")
    if len(replacement) != len(bad) or len(gpus) != len(replacement):
        raise RepairError("repair requires exactly one replacement seed/GPU per bad seed")
    if replacement != list(range(replacement[0], replacement[0] + len(replacement))):
        raise RepairError("replacement seeds must be one contiguous fresh interval")
    if any(seed in expected for seed in replacement):
        raise RepairError("replacement seeds must be outside the original interval")
    allowed_gpus = {int(lane["gpu"]) for lane in contract["execution"]["lanes"]}
    if len(set(gpus)) != len(gpus) or any(gpu not in allowed_gpus for gpu in gpus):
        raise RepairError("replacement GPUs must be unique members of the sealed host")
    repair_root = repair_root.expanduser().resolve(strict=False)
    if repair_root == Path(str(contract["execution"]["output_root"])).resolve():
        raise RepairError("repair output must never be the live corpus root")
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "classification": CLASSIFICATION,
        "production_eligible": False,
        "contract": {
            "path": str(contract_path),
            "file_sha256": _file_sha256(contract_path),
            "contract_sha256": contract["contract_sha256"],
        },
        "original_launch_receipt": {
            "path": str(Path(contract["execution"]["output_root"]) / "launch.receipt.json"),
            "file_sha256": launch_file_sha256,
            "receipt_sha256": launch["receipt_sha256"],
        },
        "native_runtime": dict(launch["preflight"]["native_runtime"]),
        "host_address": host_address,
        "original_output_root": str(contract["execution"]["output_root"]),
        "repair_root": str(repair_root),
        "bad_seeds": bad,
        "replacement_seeds": replacement,
        "replacement_gpus": gpus,
        "selection_rule": "original_interval_minus_complete_truncated_games_plus_fresh_exact_operator_replacements",
        "verified_contract": verified,
    }
    payload["plan_sha256"] = _digest(payload)
    _write_signed(write, {key: value for key, value in payload.items() if key != "plan_sha256"}, field="plan_sha256")
    return payload


def _authenticate_plan(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = _load_signed(path, schema=PLAN_SCHEMA, field="plan_sha256")
    contract_path = Path(str(plan["contract"]["path"])).resolve(strict=True)
    if _file_sha256(contract_path) != plan["contract"]["file_sha256"]:
        raise RepairError("repair-bound contract bytes drifted")
    contract = _load(contract_path)
    if contract.get("contract_sha256") != plan["contract"]["contract_sha256"]:
        raise RepairError("repair-bound contract identity drifted")
    _contract, launch, launch_file_sha256, commands = executor._authenticate_launch(  # noqa: SLF001
        contract_path, host_address=str(plan["host_address"])
    )
    if (
        launch_file_sha256 != plan["original_launch_receipt"]["file_sha256"]
        or launch.get("receipt_sha256")
        != plan["original_launch_receipt"]["receipt_sha256"]
        or launch.get("preflight", {}).get("native_runtime") != plan["native_runtime"]
    ):
        raise RepairError("original launch/runtime binding drifted")
    return plan, contract, {"launch": launch, "commands": commands}


def launch_replacements(plan_path: Path, *, go: bool) -> dict[str, Any]:
    plan_path = plan_path.expanduser().resolve(strict=True)
    plan, contract, original = _authenticate_plan(plan_path)
    synthetic = _synthetic_contract(contract, plan)
    launch = original["launch"]
    repo = Path(str(launch["preflight"]["repo"])).resolve(strict=True)
    python = Path(str(launch["preflight"]["python"]))
    commands = [
        {
            "lane_id": str(lane["lane_id"]),
            "gpu": int(lane["gpu"]),
            "claim_label": str(lane["claim_label"]),
            "argv": executor._argv(synthetic, lane, repo=repo, python=python),  # noqa: SLF001
        }
        for lane in synthetic["execution"]["lanes"]
    ]
    payload: dict[str, Any] = {
        "schema_version": LAUNCH_SCHEMA,
        "classification": CLASSIFICATION,
        "production_eligible": False,
        "status": "dry_run" if not go else "launching",
        "plan": {
            "path": str(plan_path),
            "file_sha256": _file_sha256(plan_path),
            "plan_sha256": plan["plan_sha256"],
        },
        "native_runtime": plan["native_runtime"],
        "commands": commands,
    }
    if not go:
        payload["launch_sha256"] = _digest(payload)
        return payload
    non_exited = [
        item["lane_id"]
        for item, command in zip(
            contract["execution"]["lanes"], original["commands"], strict=True
        )
        if executor._live_process_status(command)["state"] != "exited"  # noqa: SLF001
    ]
    if non_exited:
        raise RepairError(
            f"original generators are still live; refusing to contend for GPUs: {non_exited}"
        )
    service = str(synthetic["execution"]["mps_service"])
    if subprocess.run(["systemctl", "is-active", "--quiet", service]).returncode != 0:
        raise RepairError(f"MPS service is not active: {service}")
    executor._ensure_worker_fd_limit()  # noqa: SLF001
    _rows, claim = executor._claim_rows(synthetic)  # noqa: SLF001
    repair_root = Path(str(plan["repair_root"]))
    repair_root.mkdir(parents=True, exist_ok=False)
    base_env = os.environ.copy()
    base_env.update(
        {
            str(key): str(value)
            for key, value in synthetic["execution"]["mps_environment"].items()
        }
    )
    base_env["CATAN_SEED_LEDGER"] = str(synthetic["execution"]["seed_ledger"])
    base_env["PYTHONUNBUFFERED"] = "1"
    import_roots = [str(repo / "src"), str(repo / "tools")]
    if base_env.get("PYTHONPATH"):
        import_roots.append(base_env["PYTHONPATH"])
    base_env["PYTHONPATH"] = os.pathsep.join(import_roots)
    processes: list[tuple[subprocess.Popen[bytes], Any]] = []
    launched: list[dict[str, Any]] = []
    try:
        for command in commands:
            environment = dict(base_env)
            environment["CUDA_VISIBLE_DEVICES"] = str(command["gpu"])
            environment["CATAN_LEDGER_CLAIM_ID"] = str(command["claim_label"])
            log = repair_root / f"{command['lane_id']}.log"
            handle = log.open("xb")
            process = subprocess.Popen(
                command["argv"],
                cwd=repo,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((process, handle))
            launched.append(
                {
                    **command,
                    "pid": process.pid,
                    "log": str(log),
                    "out_dir": str(repair_root / command["lane_id"]),
                    "mps_environment": {
                        key: environment[key]
                        for key in (
                            "CUDA_MPS_PIPE_DIRECTORY",
                            "CUDA_MPS_LOG_DIRECTORY",
                        )
                    },
                    "process_identity": executor._process_identity(  # noqa: SLF001
                        process.pid, argv=command["argv"], environment=environment
                    ),
                }
            )
        time.sleep(2.0)
        failed = [
            item["lane_id"]
            for item, (process, _handle) in zip(launched, processes, strict=True)
            if process.poll() is not None
        ]
        if failed:
            raise RepairError(f"replacement generator exited during preamble: {failed}")
    except BaseException:
        for process, _handle in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for _process, handle in processes:
            handle.close()
    payload.update(
        {
            "status": "launched",
            "launched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "claim_receipt": claim,
            "commands": launched,
        }
    )
    receipt = repair_root / "repair.launch.receipt.json"
    _write_signed(receipt, payload, field="launch_sha256")
    return {**payload, "launch_sha256": _digest(payload), "receipt": str(receipt)}


def _load_repair_launch(plan: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    repair_root = Path(str(plan["repair_root"])).resolve(strict=True)
    path = repair_root / "repair.launch.receipt.json"
    launch = _load_signed(path, schema=LAUNCH_SCHEMA, field="launch_sha256")
    bound_plan = launch.get("plan")
    if (
        launch.get("status") != "launched"
        or launch.get("classification") != CLASSIFICATION
        or launch.get("production_eligible") is not False
        or not isinstance(bound_plan, Mapping)
        or Path(str(bound_plan.get("path", ""))).resolve(strict=True)
        != Path(str(plan["plan_path"])).resolve(strict=True)
        or bound_plan.get("file_sha256") != _file_sha256(Path(str(plan["plan_path"])))
        or bound_plan.get("plan_sha256") != plan["plan_sha256"]
        or launch.get("native_runtime") != plan["native_runtime"]
    ):
        raise RepairError("replacement launch lost its signed plan/runtime binding")
    commands = launch.get("commands")
    if not isinstance(commands, list) or len(commands) != len(plan["replacement_seeds"]):
        raise RepairError("replacement launch command count drift")
    return path, launch


def _npz_paths(root: Path) -> list[Path]:
    return sorted(
        path.resolve(strict=True)
        for path in root.rglob("*.npz")
        if path.is_file() and not path.is_symlink()
    )


def _row_seed_inventory(paths: Sequence[Path]) -> tuple[set[int], dict[int, int]]:
    seeds: set[int] = set()
    rows: dict[int, int] = {}
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as shard:
                values = np.asarray(shard["game_seed"], dtype=np.int64).reshape(-1)
                truncated = np.asarray(shard["truncated"], dtype=np.bool_).reshape(-1)
        except (OSError, ValueError, KeyError) as error:
            raise RepairError(f"cannot inspect shard {path}: {error}") from error
        if values.size == 0 or truncated.size != values.size:
            raise RepairError(f"invalid game_seed/truncated surface in {path}")
        for seed in np.unique(values):
            value = int(seed)
            seeds.add(value)
            rows[value] = rows.get(value, 0) + int(np.count_nonzero(values == value))
    return seeds, rows


def _rewrite_without_seeds(source: Path, destination: Path, excluded: set[int]) -> int:
    """Rewrite one shard while preserving row-aligned and ragged search evidence."""

    try:
        with np.load(source, allow_pickle=False) as shard:
            arrays = {name: np.asarray(shard[name]) for name in shard.files}
    except (OSError, ValueError, KeyError) as error:
        raise RepairError(f"cannot load affected shard {source}: {error}") from error
    game_seed = np.asarray(arrays.get("game_seed"), dtype=np.int64).reshape(-1)
    rows = int(game_seed.size)
    keep = ~np.isin(game_seed, np.asarray(sorted(excluded), dtype=np.int64))
    removed = int(rows - np.count_nonzero(keep))
    if removed <= 0:
        raise RepairError(f"affected shard contains no excluded rows: {source}")

    weights = np.asarray(
        arrays.get("policy_weight_multiplier"), dtype=np.float32
    ).reshape(-1)
    offsets = np.asarray(arrays.get("search_evidence_offsets"))
    visits = np.asarray(arrays.get("search_visit_counts_flat"))
    completed_q = np.asarray(arrays.get("search_completed_q_flat"))
    active_rows = np.flatnonzero(weights > 0.0)
    if (
        weights.size != rows
        or offsets.shape != (active_rows.size + 1,)
        or visits.shape != completed_q.shape
        or int(offsets[0]) != 0
        or int(offsets[-1]) != visits.size
    ):
        raise RepairError(f"malformed ragged search evidence in {source}")

    kept_active = keep[active_rows]
    lengths = np.diff(offsets).astype(np.int64, copy=False)
    segments = [
        slice(int(offsets[index]), int(offsets[index + 1]))
        for index in np.flatnonzero(kept_active)
    ]
    kept_lengths = lengths[kept_active]
    new_offsets = np.empty(kept_lengths.size + 1, dtype=offsets.dtype)
    new_offsets[0] = 0
    np.cumsum(kept_lengths, dtype=np.int64, out=new_offsets[1:])
    new_visits = (
        np.concatenate([visits[segment] for segment in segments])
        if segments
        else np.empty(0, dtype=visits.dtype)
    )
    new_completed_q = (
        np.concatenate([completed_q[segment] for segment in segments])
        if segments
        else np.empty(0, dtype=completed_q.dtype)
    )

    rewritten: dict[str, np.ndarray] = {}
    ragged = {
        "search_evidence_offsets": new_offsets,
        "search_visit_counts_flat": new_visits,
        "search_completed_q_flat": new_completed_q,
    }
    for name, value in arrays.items():
        if name in ragged:
            rewritten[name] = ragged[name]
        elif value.ndim > 0 and value.shape[0] == rows:
            rewritten[name] = value[keep]
        else:
            rewritten[name] = value
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **rewritten)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    os.replace(temporary, destination)
    return removed


def _inventory_entry(
    path: Path, *, source: Path, mode: str, relative: Path
) -> dict[str, Any]:
    stat_result = path.stat()
    return {
        "path": str(path.resolve(strict=True)),
        "relative_path": relative.as_posix(),
        "source_path": str(source.resolve(strict=True)),
        "mode": mode,
        "size_bytes": int(stat_result.st_size),
        "sha256": _file_sha256(path),
    }


def finalize_repair(plan_path: Path) -> dict[str, Any]:
    plan_path = plan_path.expanduser().resolve(strict=True)
    plan, contract, original = _authenticate_plan(plan_path)
    plan = {**plan, "plan_path": str(plan_path)}
    launch_path, repair_launch = _load_repair_launch(plan)
    original_root = Path(str(plan["original_output_root"])).resolve(strict=True)
    repair_root = Path(str(plan["repair_root"])).resolve(strict=True)
    final_root = repair_root / "repaired_shards"
    if final_root.exists():
        raise RepairError(f"curated repair view already exists without receipt: {final_root}")

    original_paths = [
        path
        for command in original["commands"]
        for path in _npz_paths(Path(str(command["out_dir"])).resolve(strict=True))
    ]
    replacement_paths = [
        path
        for command in repair_launch["commands"]
        for path in _npz_paths(Path(str(command["out_dir"])).resolve(strict=True))
    ]
    if not original_paths or len(replacement_paths) != len(plan["replacement_seeds"]):
        raise RepairError("original/replacement shard inventory is incomplete")

    original_seeds, original_rows = _row_seed_inventory(original_paths)
    replacement_seeds, replacement_rows = _row_seed_inventory(replacement_paths)
    expected_original = set(
        range(
            int(contract["execution"]["seed_start"]),
            int(contract["execution"]["seed_end"]),
        )
    )
    excluded = set(map(int, plan["bad_seeds"]))
    replacements = set(map(int, plan["replacement_seeds"]))
    if original_seeds != expected_original:
        raise RepairError("original physical shards do not cover the sealed seed interval")
    if replacement_seeds != replacements or any(replacement_rows.get(seed, 0) <= 0 for seed in replacements):
        raise RepairError("replacement physical shards do not cover exactly the planned seeds")
    for path in replacement_paths:
        with np.load(path, allow_pickle=False) as shard:
            if bool(np.any(np.asarray(shard["truncated"], dtype=np.bool_))):
                raise RepairError(f"replacement game truncated: {path}")

    selected_seeds = sorted((expected_original - excluded) | replacements)
    if len(selected_seeds) != int(contract["execution"]["total_games"]):
        raise RepairError("repaired selected-game count drift")

    temporary_root = repair_root / f".repaired_shards.{os.getpid()}.tmp"
    temporary_root.mkdir(parents=True, exist_ok=False)
    inventory: list[dict[str, Any]] = []
    removed_rows = 0
    try:
        for source in original_paths:
            relative = Path("original") / source.relative_to(original_root)
            destination = temporary_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with np.load(source, allow_pickle=False) as shard:
                affected = bool(
                    np.isin(
                        np.asarray(shard["game_seed"], dtype=np.int64),
                        np.asarray(sorted(excluded), dtype=np.int64),
                    ).any()
                )
            if affected:
                removed_rows += _rewrite_without_seeds(source, destination, excluded)
                mode = "rewritten_excluding_truncated_trajectory"
            else:
                os.link(source, destination)
                mode = "hardlink_byte_identical"
            inventory.append(
                _inventory_entry(destination, source=source, mode=mode, relative=relative)
            )
        for source in replacement_paths:
            command_root = next(
                Path(str(command["out_dir"])).resolve(strict=True)
                for command in repair_launch["commands"]
                if source.is_relative_to(Path(str(command["out_dir"])).resolve(strict=True))
            )
            relative = Path("replacement") / command_root.name / source.relative_to(command_root)
            destination = temporary_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
            inventory.append(
                _inventory_entry(
                    destination,
                    source=source,
                    mode="replacement_hardlink_byte_identical",
                    relative=relative,
                )
            )
        os.replace(temporary_root, final_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    # Paths change at the atomic rename; rebind and re-hash the final immutable view.
    final_inventory: list[dict[str, Any]] = []
    for item in inventory:
        path = final_root / str(item["relative_path"])
        source = Path(str(item["source_path"]))
        final_inventory.append(
            _inventory_entry(
                path,
                source=source,
                mode=str(item["mode"]),
                relative=Path(str(item["relative_path"])),
            )
        )
    curated_seeds, curated_rows = _row_seed_inventory(
        [Path(str(item["path"])) for item in final_inventory]
    )
    if curated_seeds != set(selected_seeds):
        raise RepairError("curated repaired view seed set drift")
    payload: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "classification": CLASSIFICATION,
        "production_eligible": False,
        "status": "complete",
        "plan": {
            "path": str(plan_path),
            "file_sha256": _file_sha256(plan_path),
            "plan_sha256": plan["plan_sha256"],
        },
        "repair_launch": {
            "path": str(launch_path),
            "file_sha256": _file_sha256(launch_path),
            "launch_sha256": repair_launch["launch_sha256"],
        },
        "contract": dict(plan["contract"]),
        "original_launch_receipt": dict(plan["original_launch_receipt"]),
        "native_runtime": dict(plan["native_runtime"]),
        "target_information_regime": contract["target_information_regime"],
        "operator_semantic_sha256": _digest(contract["operator"]),
        "original_seed_interval": {
            "start": int(contract["execution"]["seed_start"]),
            "end": int(contract["execution"]["seed_end"]),
        },
        "excluded_truncated_seeds": sorted(excluded),
        "replacement_seeds": sorted(replacements),
        "selected_game_count": len(selected_seeds),
        "selected_game_seed_set_sha256": _seed_set_sha256(selected_seeds),
        "removed_truncated_rows": removed_rows,
        "selected_row_count": int(sum(curated_rows.values())),
        "repaired_shards_root": str(final_root),
        "selected_shards": final_inventory,
        "npz_inventory_sha256": _digest(final_inventory),
        "payload_inventory_sha256": _digest(final_inventory),
    }
    receipt = repair_root / "repair.receipt.json"
    _write_signed(receipt, payload, field="receipt_sha256")
    return {**payload, "receipt_sha256": _digest(payload), "receipt": str(receipt)}


def verify_repair_receipt(path: Path, *, contract_path: Path) -> dict[str, Any]:
    """Replay a repair receipt and its byte-exact curated shard inventory."""

    receipt = _load_signed(path, schema=RECEIPT_SCHEMA, field="receipt_sha256")
    plan_path = Path(str(receipt.get("plan", {}).get("path", ""))).resolve(strict=True)
    plan, contract, _original = _authenticate_plan(plan_path)
    requested_contract_path = contract_path.expanduser().resolve(strict=True)
    requested_contract = _load(requested_contract_path)
    if (
        requested_contract.get("contract_sha256") != contract["contract_sha256"]
        or _file_sha256(requested_contract_path)
        != receipt.get("contract", {}).get("file_sha256")
        or receipt.get("contract", {}).get("contract_sha256")
        != contract["contract_sha256"]
        or receipt.get("plan", {}).get("file_sha256") != _file_sha256(plan_path)
        or receipt.get("plan", {}).get("plan_sha256") != plan["plan_sha256"]
        or receipt.get("status") != "complete"
        or receipt.get("selected_game_count") != contract["execution"]["total_games"]
        or receipt.get("target_information_regime")
        != contract["target_information_regime"]
    ):
        raise RepairError("repair receipt contract/plan/semantic binding drift")
    inventory = receipt.get("selected_shards")
    if not isinstance(inventory, list) or not inventory:
        raise RepairError("repair receipt has no selected shard inventory")
    replayed: list[dict[str, Any]] = []
    for item in inventory:
        if not isinstance(item, Mapping):
            raise RepairError("repair selected shard inventory is malformed")
        shard = Path(str(item.get("path", ""))).resolve(strict=True)
        if (
            shard.is_symlink()
            or shard.stat().st_size != int(item.get("size_bytes", -1))
            or _file_sha256(shard) != item.get("sha256")
        ):
            raise RepairError(f"repair selected shard bytes drifted: {shard}")
        replayed.append(dict(item))
    if (
        _digest(replayed) != receipt.get("npz_inventory_sha256")
        or _digest(replayed) != receipt.get("payload_inventory_sha256")
    ):
        raise RepairError("repair selected shard inventory digest drift")
    seeds, rows = _row_seed_inventory([Path(str(item["path"])) for item in replayed])
    if (
        len(seeds) != receipt["selected_game_count"]
        or _seed_set_sha256(sorted(seeds))
        != receipt.get("selected_game_seed_set_sha256")
        or int(sum(rows.values())) != receipt.get("selected_row_count")
    ):
        raise RepairError("repair selected seed/row inventory drift")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--contract", required=True, type=Path)
    plan.add_argument("--host-address", required=True)
    plan.add_argument("--bad-seed", action="append", required=True, type=int)
    plan.add_argument("--replacement-seed", action="append", required=True, type=int)
    plan.add_argument("--replacement-gpu", action="append", required=True, type=int)
    plan.add_argument("--repair-root", required=True, type=Path)
    plan.add_argument("--write", required=True, type=Path)
    launch = sub.add_parser("launch")
    launch.add_argument("--plan", required=True, type=Path)
    launch.add_argument("--go", action="store_true")
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--plan", required=True, type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--receipt", required=True, type=Path)
    verify.add_argument("--contract", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = build_plan(
                contract_path=args.contract,
                host_address=args.host_address,
                bad_seeds=args.bad_seed,
                replacement_seeds=args.replacement_seed,
                replacement_gpus=args.replacement_gpu,
                repair_root=args.repair_root,
                write=args.write,
            )
        elif args.command == "launch":
            result = launch_replacements(args.plan, go=bool(args.go))
        elif args.command == "finalize":
            result = finalize_repair(args.plan)
        else:
            result = verify_repair_receipt(
                args.receipt, contract_path=args.contract
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (RepairError, executor.ExecutorError, identity.InventoryError) as error:
        parser.exit(2, f"REFUSED: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
