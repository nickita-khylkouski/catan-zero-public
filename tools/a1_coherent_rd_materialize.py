#!/usr/bin/env python3
"""Authenticate and materialize a compact coherent-public n128 R&D corpus.

This tool is intentionally diagnostic-only.  ``attest`` must run in the exact
generation checkout/runtime before it changes.  ``materialize`` replays that
authority, converts the exact authenticated shards to memmap, binds the current
``policy-target-teacher-identity-v2``, and proves learner admission.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from catan_zero.rl.gumbel_self_play import (  # noqa: E402
    TARGET_INFORMATION_REGIME_PUBLIC_COHERENT,
)
from tools import a1_target_eligibility_inventory as identity  # noqa: E402
from tools import build_memmap_corpus  # noqa: E402
from tools import train_bc  # noqa: E402


ATTESTATION_SCHEMA = "a1-direct-coherent-rd-runtime-attestation-v1"
MATERIALIZATION_SCHEMA = "a1-coherent-rd-materialization-receipt-v1"
CODE_PATHS = (
    "tools/generate.py",
    "tools/generate_gumbel_selfplay_data.py",
    "src/catan_zero/rl/gumbel_self_play.py",
    "src/catan_zero/search/neural_rust_mcts.py",
    "src/catan_zero/search/native_gumbel_mcts.py",
)


class MaterializationError(RuntimeError):
    """The direct R&D corpus cannot be authenticated or admitted."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _confirmed_worker_shard_path(
    worker_dir: Path,
    *,
    filename: object,
    index: int,
) -> Path:
    """Resolve one generated shard without permitting path escape or aliases."""

    expected = {
        f"gumbel_self_play_shard_{index:05d}.npz",
        f"gumbel_self_play_shard_{index:05d}.npz.zst",
    }
    if not isinstance(filename, str) or filename not in expected:
        raise MaterializationError(
            f"confirmed shard filename is not canonical for index {index}: "
            f"{filename!r}"
        )
    worker_root = worker_dir.resolve(strict=True)
    shard_path = worker_dir / filename
    try:
        resolved = shard_path.resolve(strict=True)
    except OSError as error:
        raise MaterializationError(
            f"missing confirmed shard {shard_path}: {error}"
        ) from error
    if resolved.parent != worker_root or shard_path.is_symlink():
        raise MaterializationError(
            f"confirmed shard escapes or aliases worker directory: {shard_path}"
        )
    return shard_path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MaterializationError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise MaterializationError(f"{path} must contain a JSON object")
    return value


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise MaterializationError(f"refusing non-fresh receipt path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    value = dict(payload)
    value["receipt_sha256"] = _digest(value)
    data = json.dumps(value, indent=2, sort_keys=True).encode("ascii") + b"\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        os.chmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    finally:
        temporary.unlink(missing_ok=True)


def _authenticated_receipt(path: Path, schema: str) -> dict[str, Any]:
    value = _load_json(path.expanduser().resolve(strict=True))
    unhashed = dict(value)
    declared = unhashed.pop("receipt_sha256", None)
    if value.get("schema_version") != schema or declared != _digest(unhashed):
        raise MaterializationError(f"receipt authentication failed: {path}")
    return value


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, timeout=60
        ).strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise MaterializationError(f"git {' '.join(args)} failed: {error}") from error


def _tracked_inventory(repo: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line in _git(repo, "ls-files", "-s").splitlines():
        metadata, relative = line.split("\t", 1)
        mode, object_sha1, stage = metadata.split()
        if stage != "0":
            raise MaterializationError(f"unmerged tracked file: {relative}")
        path = (repo / relative).resolve(strict=True)
        records.append(
            {
                "path": relative,
                "mode": mode,
                "git_object_sha1": object_sha1,
                "file_sha256": _file_sha256(path),
            }
        )
    return records


def _synthetic_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    cli = manifest.get("cli_args")
    if not isinstance(cli, Mapping):
        raise MaterializationError("lane manifest lacks cli_args")
    return {
        "target_information_regime": manifest["target_information_regime"],
        "producer_checkpoint": {
            "sha256": manifest["producer_checkpoint_sha256"],
        },
        "operator": {
            "record_automatic_transitions": cli["record_automatic_transitions"],
            "event_history_limit": cli["event_history_limit"],
        },
        "execution": {"workers_per_gpu": int(cli["workers"])},
        "acceptance": {
            "require_search_evidence_schema": manifest["search_evidence_schema"],
        },
    }


def _normalized_sha256(value: object) -> str:
    text = str(value)
    return text if text.startswith("sha256:") else "sha256:" + text


def _expected_worker_games(games: int, workers: int, index: int) -> int:
    quotient, remainder = divmod(games, workers)
    return quotient + (1 if index < remainder else 0)


def _verify_shard(
    path: Path,
    *,
    regime: str,
    trace: dict[str, Any],
) -> dict[str, int]:
    try:
        with np.load(path, allow_pickle=False) as shard:
            required = {
                "game_seed",
                "decision_index",
                "seat",
                "terminated",
                "truncated",
                "policy_weight_multiplier",
                "target_information_regime",
                "legal_action_mask",
                "simulations_used",
                "search_evidence_version",
                "search_evidence_offsets",
                "search_visit_counts_flat",
                "search_completed_q_flat",
            }
            missing = required - set(shard.files)
            if missing:
                raise MaterializationError(
                    f"shard lacks closure columns {sorted(missing)}: {path}"
                )
            seeds = np.asarray(shard["game_seed"]).reshape(-1)
            decisions = np.asarray(shard["decision_index"]).reshape(-1)
            seats = np.asarray(shard["seat"]).reshape(-1)
            terminated = np.asarray(shard["terminated"], dtype=np.bool_).reshape(-1)
            truncated = np.asarray(shard["truncated"], dtype=np.bool_).reshape(-1)
            weights = np.asarray(
                shard["policy_weight_multiplier"], dtype=np.float32
            ).reshape(-1)
            regimes = np.asarray(shard["target_information_regime"]).reshape(-1)
            rows = int(seeds.size)
            if not all(
                array.size == rows
                for array in (
                    decisions,
                    seats,
                    terminated,
                    truncated,
                    weights,
                    regimes,
                )
            ):
                raise MaterializationError(f"scalar column length drift: {path}")
            if (
                np.any((seats != 0) & (seats != 1))
                or np.any(regimes != regime)
                or np.any(~np.isfinite(weights))
                or np.any(weights < 0.0)
                or np.any(truncated)
            ):
                raise MaterializationError(f"invalid scalar target payload: {path}")
            active = weights > 0.0
            offsets = np.asarray(shard["search_evidence_offsets"], dtype=np.uint32)
            visits = np.asarray(shard["search_visit_counts_flat"], dtype=np.uint16)
            completed_q = np.asarray(
                shard["search_completed_q_flat"], dtype=np.float32
            )
            if (
                int(np.asarray(shard["search_evidence_version"]).item()) != 1
                or offsets.shape != (int(active.sum()) + 1,)
                or int(offsets[0]) != 0
                or np.any(offsets[1:] < offsets[:-1])
                or visits.shape != completed_q.shape
                or int(offsets[-1]) != visits.size
                or np.any(~np.isfinite(completed_q))
            ):
                raise MaterializationError(f"malformed search evidence: {path}")
            widths = np.asarray(shard["legal_action_mask"], dtype=np.bool_).sum(axis=1)[
                active
            ]
            simulations = np.asarray(shard["simulations_used"]).reshape(-1)[active]
            cumulative = np.concatenate(
                (np.asarray([0], dtype=np.uint64), np.cumsum(visits, dtype=np.uint64))
            )
            evidence_sims = cumulative[offsets[1:]] - cumulative[offsets[:-1]]
            if (
                not np.array_equal(widths.astype(np.uint32), np.diff(offsets))
                or not np.array_equal(evidence_sims, simulations.astype(np.uint64))
            ):
                raise MaterializationError(f"search evidence closure failed: {path}")
            for index in range(rows):
                seed = int(seeds[index])
                decision = int(decisions[index])
                current_seed = trace.get("current_seed")
                if current_seed is None or seed != current_seed:
                    if current_seed is not None and (
                        not trace["current_complete"]
                        or trace["current_seats"] != {0, 1}
                    ):
                        raise MaterializationError(
                            f"incomplete preceding trajectory before seed {seed}"
                        )
                    if seed in trace["seen"] or decision != 0:
                        raise MaterializationError(
                            f"duplicate/nonzero trajectory start {seed}:{decision}"
                        )
                    trace["seen"].add(seed)
                    trace["current_seed"] = seed
                    trace["last_decision"] = decision
                    trace["current_seats"] = {int(seats[index])}
                    trace["current_complete"] = bool(terminated[index])
                else:
                    if decision != int(trace["last_decision"]) + 1:
                        raise MaterializationError(
                            f"decision jump for {seed}: "
                            f"{trace['last_decision']}->{decision}"
                        )
                    trace["last_decision"] = decision
                    trace["current_seats"].add(int(seats[index]))
                    trace["current_complete"] = bool(
                        trace["current_complete"] and terminated[index]
                    )
            return {
                "rows": rows,
                "policy_active_rows": int(np.count_nonzero(active)),
            }
    except MaterializationError:
        raise
    except Exception as error:  # noqa: BLE001
        raise MaterializationError(f"cannot authenticate shard {path}: {error}") from error


def _verify_worker(
    lane: Path,
    *,
    worker_index: int,
    lane_manifest: Mapping[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    cli = lane_manifest["cli_args"]
    workers = int(cli["workers"])
    games = int(lane_manifest["games_requested"])
    worker_id = f"worker_{worker_index:03d}"
    worker_dir = lane / worker_id
    manifest_path = worker_dir / "manifest.json"
    progress_path = worker_dir / "progress.json"
    manifest = _load_json(manifest_path)
    progress = _load_json(progress_path)
    expected_games = _expected_worker_games(games, workers, worker_index)
    expected_start = sum(
        _expected_worker_games(games, workers, prior)
        for prior in range(worker_index)
    )
    expected = {
        "games_requested": expected_games,
        "games_completed": expected_games,
        "games_failed": 0,
        "games_truncated": 0,
        "game_index_start": expected_start,
        "base_seed": int(lane_manifest["base_seed"]),
        "target_information_regime": lane_manifest["target_information_regime"],
        "search_evidence_schema": lane_manifest["search_evidence_schema"],
    }
    if (
        any(manifest.get(key) != value for key, value in expected.items())
        or manifest.get("errors") not in (None, [])
        or int(progress.get("games_succeeded", -1)) != expected_games
        or int(progress.get("games_failed", -1)) != 0
        or int(progress.get("games_truncated", -1)) != 0
        or progress.get("errors") not in (None, [])
    ):
        raise MaterializationError(f"worker completion drift: {worker_dir}")
    selfplay = manifest.get("selfplay_config")
    expected_selfplay = {
        "meaningful_public_history": True,
        "event_history_limit": int(cli["event_history_limit"]),
        "record_automatic_transitions": bool(cli["record_automatic_transitions"]),
    }
    if not isinstance(selfplay, Mapping) or any(
        selfplay.get(key) != value for key, value in expected_selfplay.items()
    ):
        raise MaterializationError(f"worker row-surface drift: {worker_dir}")
    confirmed = progress.get("confirmed_shards")
    manifest_shards = manifest.get("shards")
    if (
        not isinstance(confirmed, list)
        or not isinstance(manifest_shards, list)
        or not confirmed
        or len(confirmed) != len(manifest_shards)
    ):
        raise MaterializationError(f"worker shard inventory drift: {worker_dir}")
    shard_records: list[dict[str, Any]] = []
    rows = 0
    active_rows = 0
    for index, record in enumerate(confirmed):
        path = _confirmed_worker_shard_path(
            worker_dir,
            filename=record.get("filename"),
            index=index,
        )
        if (
            int(record.get("index", -1)) != index
            or str(path) != str(manifest_shards[index])
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(record.get("size_bytes", -1))
            or _file_sha256(path) != _normalized_sha256(record.get("sha256"))
        ):
            raise MaterializationError(f"confirmed shard bytes drift: {path}")
        arrays = _verify_shard(
            path,
            regime=str(lane_manifest["target_information_regime"]),
            trace=trace,
        )
        if arrays["rows"] != int(record.get("rows", -1)):
            raise MaterializationError(f"confirmed shard row drift: {path}")
        rows += arrays["rows"]
        active_rows += arrays["policy_active_rows"]
        shard_records.append(
            {
                "index": index,
                "path": str(path),
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
                "rows": arrays["rows"],
            }
        )
    if (
        rows != int(manifest.get("rows", -1))
        or rows != int(progress.get("rows_confirmed", progress.get("rows", -1)))
        or int(progress.get("shard_count_confirmed", -1)) != len(shard_records)
        or int(progress.get("simulations_used_total", -1))
        != int(manifest.get("simulations_used_total", -2))
    ):
        raise MaterializationError(f"worker payload totals drift: {worker_dir}")
    return {
        "worker_id": worker_id,
        "manifest": {
            "path": str(manifest_path),
            "file_sha256": _file_sha256(manifest_path),
        },
        "progress": {
            "path": str(progress_path),
            "file_sha256": _file_sha256(progress_path),
        },
        "games_completed": expected_games,
        "rows": rows,
        "policy_active_rows": active_rows,
        "simulations_used_total": int(manifest["simulations_used_total"]),
        "shard_count": len(shard_records),
        "shards_sha256": _digest(shard_records),
        "shard_bytes": sum(int(item["size_bytes"]) for item in shard_records),
        "shards": shard_records,
    }


def _resolved_config(
    lane: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    cli = manifest["cli_args"]
    path = Path(str(cli["dump_config"])).expanduser().resolve(strict=True)
    config = _load_json(path)
    full_hash = _digest(config)
    short_hash = "sha256:" + full_hash.removeprefix("sha256:")[:16]
    if (
        config.get("schema_version") != 18
        or manifest.get("full_config_hash") != full_hash
        or manifest.get("config_hash") != short_hash
        or path.parent != lane
    ):
        raise MaterializationError(f"resolved config binding failed: {path}")
    fields = config.get("fields")
    if not isinstance(fields, dict):
        raise MaterializationError(f"resolved config lacks fields: {path}")
    return (
        {
            "path": str(path),
            "file_sha256": _file_sha256(path),
            "full_config_hash": full_hash,
        },
        fields,
    )


def _lane_attestation(lane: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = lane / "manifest.json"
    manifest = _load_json(manifest_path)
    expected = int(manifest.get("games_requested", -1))
    if (
        expected <= 0
        or int(manifest.get("games_completed", -1)) != expected
        or int(manifest.get("games_failed", -1)) != 0
        or int(manifest.get("games_truncated", -1)) != 0
        or manifest.get("errors") not in (None, [])
        or manifest.get("target_information_regime")
        != TARGET_INFORMATION_REGIME_PUBLIC_COHERENT
    ):
        raise MaterializationError(f"lane is not cleanly complete: {lane}")
    cli = manifest.get("cli_args")
    required = {
        "public_observation": True,
        "coherent_public_belief_search": True,
        "information_set_search": False,
        "n_full": 128,
        "n_fast": 16,
        "p_full": 0.25,
        "meaningful_public_history": True,
        "preserve_search_evidence": True,
        "record_automatic_transitions": True,
        "native_mcts_hot_loop": True,
        "rust_featurize": True,
    }
    if not isinstance(cli, Mapping) or any(cli.get(k) != v for k, v in required.items()):
        raise MaterializationError(f"lane operator drift: {lane}")
    workers = int(cli["workers"])
    trace: dict[str, Any] = {
        "seen": set(),
        "current_seed": None,
        "current_complete": False,
        "current_seats": set(),
        "last_decision": -1,
    }
    worker_records = [
        _verify_worker(
            lane,
            worker_index=index,
            lane_manifest=manifest,
            trace=trace,
        )
        for index in range(workers)
    ]
    if (
        trace["current_seed"] is None
        or not trace["current_complete"]
        or trace["current_seats"] != {0, 1}
        or len(trace["seen"]) != expected
    ):
        raise MaterializationError(f"lane trajectory closure failed: {lane}")
    shards = [item for worker in worker_records for item in worker["shards"]]
    if [item["path"] for item in shards] != list(manifest.get("shards", [])):
        raise MaterializationError(f"lane top-level shard inventory drift: {lane}")
    if sum(int(item["rows"]) for item in worker_records) != int(manifest["rows"]):
        raise MaterializationError(f"lane row total drift: {lane}")
    config_record, config_fields = _resolved_config(lane, manifest)
    identity_manifest = dict(manifest)
    identity_manifest["operator"] = config_fields
    return (
        {
            "lane": str(lane),
            "manifest": {
                "path": str(manifest_path),
                "file_sha256": _file_sha256(manifest_path),
            },
            "resolved_config": config_record,
            "base_seed": int(manifest["base_seed"]),
            "games": expected,
            "rows": int(manifest["rows"]),
            "policy_active_rows": sum(
                int(item["policy_active_rows"]) for item in worker_records
            ),
            "workers": worker_records,
            "shards": shards,
            "shards_sha256": _digest(shards),
        },
        identity_manifest,
    )


def attest(
    *,
    repo: Path,
    source_root: Path,
    checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    repo = repo.expanduser().resolve(strict=True)
    source_root = source_root.expanduser().resolve(strict=True)
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    if _git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise MaterializationError("generation checkout has tracked modifications")
    lanes = sorted(
        path for path in source_root.iterdir() if (path / "manifest.json").is_file()
    )
    if not lanes:
        raise MaterializationError(f"no completed lanes under {source_root}")
    lane_records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for lane in lanes:
        record, manifest = _lane_attestation(lane)
        lane_records.append(record)
        manifests.append(manifest)
    checkpoint_sha = _file_sha256(checkpoint)
    if {
        str(manifest.get("producer_checkpoint_sha256")) for manifest in manifests
    } != {checkpoint_sha}:
        raise MaterializationError("checkpoint bytes differ from lane producer identity")

    try:
        import catanatron_rs
        import torch
    except ImportError as error:
        raise MaterializationError("exact generation runtime is unavailable") from error
    native_module = importlib.import_module("catanatron_rs.catanatron_rs")
    native = Path(str(native_module.__file__)).resolve(strict=True)
    package = Path(str(catanatron_rs.__file__)).resolve(strict=True)
    native_distribution = distribution("catanatron-rs")
    inventory = _tracked_inventory(repo)
    authority = {
        "git_commit": _git(repo, "rev-parse", "HEAD"),
        "checkout_tree_sha256": _digest(inventory),
    }
    identity_constructor = getattr(
        identity,
        "canonical_policy_target_identity",
        identity._canonical_policy_target_identity,  # noqa: SLF001
    )
    identity_digest = getattr(
        identity,
        "canonical_value_sha256",
        identity._value_sha256,  # noqa: SLF001
    )
    identities = [
        identity_constructor(
            manifest, authority=authority, strict_current=True
        )
        for manifest in manifests
    ]
    identity_hashes = {identity_digest(value) for value in identities}
    if len(identity_hashes) != 1 or any(value != identities[0] for value in identities):
        raise MaterializationError("lanes do not share one exact policy teacher")
    payload: dict[str, Any] = {
        "schema_version": ATTESTATION_SCHEMA,
        "authority_timing": "posthoc_same_checkout",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "limitations": ["does_not_prove_prelaunch_contemporaneity"],
        "source_root": str(source_root),
        "repo": str(repo),
        "git_commit": authority["git_commit"],
        "git_tree_sha1": _git(repo, "rev-parse", "HEAD^{tree}"),
        "checkout_tree_sha256": authority["checkout_tree_sha256"],
        "tracked_file_inventory": inventory,
        "tracked_file_inventory_sha256": _digest(inventory),
        "runtime": {
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
            "python_version": sys.version,
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "native_module": {
                "path": str(native),
                "file_sha256": _file_sha256(native),
                "package_path": str(package),
                "distribution_name": str(
                    native_distribution.metadata.get("Name") or "catanatron-rs"
                ),
                "distribution_version": str(native_distribution.version),
                "capabilities": sorted(catanatron_rs.gumbel_search_capabilities()),
            },
            "code_files": {
                relative: _file_sha256(repo / relative) for relative in CODE_PATHS
            },
        },
        "checkpoint": {"path": str(checkpoint), "file_sha256": checkpoint_sha},
        "lanes": lane_records,
        "totals": {
            "lanes": len(lane_records),
            "games": sum(int(item["games"]) for item in lane_records),
            "rows": sum(int(item["rows"]) for item in lane_records),
            "policy_active_rows": sum(
                int(item["policy_active_rows"]) for item in lane_records
            ),
            "shards": sum(len(item["shards"]) for item in lane_records),
        },
        "policy_target_identity": identities[0],
        "policy_target_identity_sha256": next(iter(identity_hashes)),
    }
    _write_immutable(output, payload)
    return payload


def _replay_attestation(attestation: Mapping[str, Any]) -> None:
    repo = Path(str(attestation["repo"])).resolve(strict=True)
    if _git(repo, "rev-parse", "HEAD") != attestation["git_commit"]:
        raise MaterializationError("generation checkout commit changed")
    inventory = _tracked_inventory(repo)
    if (
        _digest(inventory) != attestation["tracked_file_inventory_sha256"]
        or _digest(inventory) != attestation["checkout_tree_sha256"]
    ):
        raise MaterializationError("generation checkout bytes changed")
    paths: list[Mapping[str, Any]] = [
        attestation["checkpoint"],
        attestation["runtime"]["native_module"],
    ]
    for lane in attestation["lanes"]:
        paths.extend((lane["manifest"], lane["resolved_config"]))
        for worker in lane["workers"]:
            paths.extend((worker["manifest"], worker["progress"]))
            paths.extend(worker["shards"])
    for record in paths:
        expected = record.get("file_sha256", record.get("sha256"))
        if _file_sha256(Path(str(record["path"])).resolve(strict=True)) != expected:
            raise MaterializationError(f"authenticated artifact changed: {record['path']}")


def materialize(
    *,
    attestation_path: Path,
    output: Path,
    receipt: Path,
    progress_every: int,
) -> dict[str, Any]:
    attestation = _authenticated_receipt(attestation_path, ATTESTATION_SCHEMA)
    _replay_attestation(attestation)
    sources = [Path(str(item["lane"])) for item in attestation["lanes"]]
    if output.exists():
        raise MaterializationError(f"refusing non-fresh corpus output: {output}")
    meta = build_memmap_corpus.build_memmap_corpus(
        sources,
        output,
        progress_every=progress_every,
        abort_on_duplicate_seeds=True,
    )
    expected_inventory = [
        {
            "path": shard["path"],
            "size_bytes": shard["size_bytes"],
            "sha256": shard["sha256"],
        }
        for lane in attestation["lanes"]
        for shard in lane["shards"]
    ]
    if (
        meta["source_shard_inventory"] != expected_inventory
        or int(meta["row_count"]) != int(attestation["totals"]["rows"])
        or int(meta["stats"]["duplicate_game_seed_count"]) != 0
    ):
        raise MaterializationError("memmap does not bind the authenticated shard set")
    meta.update(
        {
            "diagnostic_only": True,
            "promotion_eligible": False,
            "policy_target_identity": attestation["policy_target_identity"],
            "policy_target_identity_sha256": attestation[
                "policy_target_identity_sha256"
            ],
            "coherent_rd_runtime_attestation": {
                "path": str(attestation_path.resolve(strict=True)),
                "file_sha256": _file_sha256(attestation_path.resolve(strict=True)),
                "receipt_sha256": attestation["receipt_sha256"],
            },
        }
    )
    meta_path = output / "corpus_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    corpus = train_bc.MemmapCorpus(output)
    weights = corpus["policy_weight_multiplier"].reshape(-1)
    admission = train_bc._validate_policy_target_identity_scope(  # noqa: SLF001
        corpus,
        weights,
        accepted_identities=[attestation["policy_target_identity_sha256"]],
    )
    inspection = identity.inspect_memmap(
        label="coherent_n128_rd",
        corpus_dir=output,
        required_regime=TARGET_INFORMATION_REGIME_PUBLIC_COHERENT,
    )
    if (
        inspection["policy_targets_eligible_for_requested_learner"] is not True
        or inspection["policy_active_rows"]
        != int(attestation["totals"]["policy_active_rows"])
    ):
        raise MaterializationError("materialized corpus failed policy-target replay")
    payload = {
        "schema_version": MATERIALIZATION_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "runtime_attestation": {
            "path": str(attestation_path.resolve(strict=True)),
            "file_sha256": _file_sha256(attestation_path.resolve(strict=True)),
            "receipt_sha256": attestation["receipt_sha256"],
        },
        "corpus": {
            "path": str(output.resolve(strict=True)),
            "meta_path": str(meta_path),
            "meta_file_sha256": _file_sha256(meta_path),
            "payload_inventory_sha256": meta["payload_inventory_sha256"],
            "source_shard_inventory_sha256": meta[
                "source_shard_inventory_sha256"
            ],
            "rows": int(meta["row_count"]),
        },
        "policy_target_identity_sha256": attestation[
            "policy_target_identity_sha256"
        ],
        "learner_admission": admission,
        "inventory": inspection,
    }
    _write_immutable(receipt, payload)
    return payload


def _paths(values: Sequence[str]) -> list[Path]:
    return [Path(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    attest_parser = subparsers.add_parser("attest")
    attest_parser.add_argument("--repo", type=Path, required=True)
    attest_parser.add_argument("--source-root", type=Path, required=True)
    attest_parser.add_argument("--checkpoint", type=Path, required=True)
    attest_parser.add_argument("--output", type=Path, required=True)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--attestation", type=Path, required=True)
    materialize_parser.add_argument("--output", type=Path, required=True)
    materialize_parser.add_argument("--receipt", type=Path, required=True)
    materialize_parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if args.command == "attest":
        result = attest(
            repo=args.repo,
            source_root=args.source_root,
            checkpoint=args.checkpoint,
            output=args.output,
        )
    else:
        result = materialize(
            attestation_path=args.attestation,
            output=args.output,
            receipt=args.receipt,
            progress_every=args.progress_every,
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "command": args.command,
                "policy_target_identity_sha256": result[
                    "policy_target_identity_sha256"
                ],
                "diagnostic_only": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
