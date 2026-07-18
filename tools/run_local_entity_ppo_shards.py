#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import errno
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any
import uuid

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
for _checkout_path in (_REPO_ROOT, _SRC_DIR, _TOOLS_DIR):
    while str(_checkout_path) in sys.path:
        sys.path.remove(str(_checkout_path))
    sys.path.insert(0, str(_checkout_path))

import numpy as np  # noqa: E402

from catan_zero.rl import ppo_distributed as dist  # noqa: E402
from catan_zero.rl.config_cli import _explicit_cli_dests  # noqa: E402
from catan_zero.rl.ppo_policy_factory import (  # noqa: E402
    CANONICAL_PPO_ARCHITECTURE,
    load_ppo_policy,
    validate_canonical_ppo_actor_contract,
)
from catan_zero.rl.ppo_run_manifest import (  # noqa: E402
    ManifestError,
    PPORunManifest,
    load_manifest,
)
from catan_zero.rl.torch_ppo import collect_ppo_episode  # noqa: E402
try:
    from factory_common import make_named_policy, parse_track
except ModuleNotFoundError as error:  # package import; direct script uses the sibling import
    if error.name != "factory_common":
        raise
    from tools.factory_common import make_named_policy, parse_track


SEATS = ("BLUE", "RED", "ORANGE", "WHITE")

_MANIFEST_RUNTIME_DESTS = {
    "run_manifest",
    "run_base",
    "run_name",
    "checkpoint",
    "devices",
    "games",
    "launch_id",
    "workers",
    "publish",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate local distributed-PPO trajectory shards for the 35M entity policy."
    )
    parser.add_argument(
        "--run-manifest",
        default=None,
        help=(
            "Bound canonical_entity_ppo_run_v2 manifest. In manifest mode it is "
            "the sole actor-science authority."
        ),
    )
    parser.add_argument("--run-base", default="runs/distributed")
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--launch-id",
        default=None,
        help=(
            "Immutable actor-launch identity. Omit for a fresh unique launch; "
            "reuse an emitted identity only to resume that exact launch."
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--architecture",
        choices=(CANONICAL_PPO_ARCHITECTURE,),
        default=CANONICAL_PPO_ARCHITECTURE,
    )
    parser.add_argument("--track", default="2p_no_trade")
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--games-per-shard", type=int, default=2)
    parser.add_argument("--max-decisions", type=int, default=1200)
    parser.add_argument("--opponents", default="catanatron_ab3,catanatron_value,heuristic,random")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--seed", type=int, default=70630001)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--value-shaping-coef", type=float, default=0.0)
    parser.add_argument("--value-shaping-scale", type=float, default=100.0)
    parser.add_argument("--value-shaping-opponent-penalty", type=float, default=0.05)
    parser.add_argument(
        "--action-temperature",
        type=float,
        default=1.0,
        help=(
            "Temperature for stochastic PPO actor sampling. 1.0 preserves the "
            "raw policy; lower values reduce destructive exploration while "
            "recording old_log_probs under the actual behavior distribution."
        ),
    )
    parser.add_argument("--publish", action="store_true")
    return parser


def resolve_config(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, PPORunManifest | None]:
    parser = build_arg_parser()
    effective_argv = list(argv) if argv is not None else None
    args = parser.parse_args(effective_argv)
    explicit_dests = _explicit_cli_dests(
        parser,
        effective_argv if effective_argv is not None else sys.argv[1:],
    )
    manifest = None
    args.run_manifest_sha256 = None
    args.opponent_mode = "fixed"
    args.pfsp_mode = "pfsp"
    if args.run_manifest:
        conflicts = sorted(explicit_dests - _MANIFEST_RUNTIME_DESTS)
        if conflicts:
            parser.error(
                "--run-manifest cannot be combined with legacy actor-science "
                f"flags: {', '.join(conflicts)}"
            )
        try:
            manifest = load_manifest(args.run_manifest)
        except (OSError, ManifestError) as error:
            parser.error(f"invalid --run-manifest: {error}")
        if manifest.status != "bound":
            parser.error("--run-manifest must have status='bound'; templates cannot run")
        expected_sha256 = manifest.spec.identity.initializer_sha256
        try:
            actual_sha256 = f"sha256:{dist.checkpoint_sha256(args.checkpoint)}"
        except OSError as error:
            parser.error(f"cannot hash --checkpoint: {error}")
        if actual_sha256 != expected_sha256:
            parser.error(
                "--checkpoint SHA-256 does not match run manifest identity: "
                f"expected={expected_sha256} actual={actual_sha256}"
            )
        identity = manifest.spec.identity
        actor = manifest.spec.actor
        if actor.opponent_mode != "fixed":
            parser.error(
                "local v2 actor supports only opponent_mode='fixed'; "
                "league/PFSP execution requires the Modal actor"
            )
        args.architecture = identity.architecture
        args.track = identity.track
        args.vps_to_win = identity.vps_to_win
        args.max_decisions = actor.max_decisions
        args.games_per_shard = actor.games_per_shard
        args.gamma = actor.gamma
        args.gae_lambda = actor.gae_lambda
        args.action_temperature = actor.action_temperature
        args.value_shaping_coef = actor.value_shaping_coef
        args.value_shaping_scale = actor.value_shaping_scale
        args.value_shaping_opponent_penalty = actor.value_shaping_opponent_penalty
        args.seed = actor.seed
        args.opponent_mode = actor.opponent_mode
        args.opponents = ",".join(actor.opponents)
        args.pfsp_mode = actor.pfsp_mode
        args.run_manifest_sha256 = manifest.sha256()
    try:
        validate_canonical_ppo_actor_contract(
            architecture=args.architecture,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            action_temperature=args.action_temperature,
        )
    except ValueError as error:
        parser.error(str(error))
    return args, manifest


def _bind_run_root(
    args: argparse.Namespace,
    manifest: PPORunManifest | None,
):
    root = dist.run_root(args.run_base, args.run_name)
    if manifest is not None:
        dist.bind_run_manifest(root, manifest)
        dist.ensure_run_dirs(root)
    else:
        dist.ensure_run_dirs(root)
        dist.bind_run_contract(
            root,
            init_checkpoint=args.checkpoint,
            architecture=args.architecture,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            behavior_temperature=args.action_temperature,
        )
    return root


_LAUNCH_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_LAUNCH_SCHEMA = "local_entity_ppo_actor_launch_v1"


def _launches_dir(root: Path) -> Path:
    return root / "actor_launches"


def _atomic_bind_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Create ``path`` once, or verify that its immutable payload is exact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid immutable actor launch contract at {path}") from error
    if existing != payload:
        raise RuntimeError(
            "actor launch contract mismatch: immutable launch settings changed; "
            f"expected={existing!r} requested={payload!r}"
        )
    return existing


def _snapshot_policy(source: Path, destination: Path) -> str:
    """Create one immutable policy snapshot and return its authenticated digest."""

    if not source.is_file():
        raise RuntimeError(f"cannot snapshot unavailable actor policy checkpoint: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_before = dist.checkpoint_sha256(source)
    temporary: Path | None = None
    try:
        try:
            os.link(source, destination)
        except OSError as error:
            link_fallback_errors = {
                errno.EXDEV,
                errno.EPERM,
                getattr(errno, "EOPNOTSUPP", errno.EPERM),
                getattr(errno, "ENOTSUP", errno.EPERM),
            }
            if error.errno not in link_fallback_errors:
                raise
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
            )
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        snapshot_sha256 = dist.checkpoint_sha256(destination)
        source_after = dist.checkpoint_sha256(source)
        if snapshot_sha256 != source_before or snapshot_sha256 != source_after:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"actor policy checkpoint changed while snapshotting: {source}"
            )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return snapshot_sha256


def _prepare_launch_unlocked(
    args: argparse.Namespace, published: Any, launch_id: str
) -> dict[str, Any]:
    root = dist.run_root(args.run_base, args.run_name)
    launch_dir = _launches_dir(root) / str(launch_id)
    path = launch_dir / "launch.json"
    snapshot = launch_dir / "policy.pt"
    snapshot_binding = launch_dir / "policy_snapshot.json"
    existing: dict[str, Any] | None = None
    stream_nonce: int
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid immutable actor launch contract at {path}") from error
        stream_nonce = existing.get("stream_nonce")
        if not isinstance(stream_nonce, int) or isinstance(stream_nonce, bool) or stream_nonce < 0:
            raise RuntimeError(
                f"invalid immutable actor launch stream_nonce at {path}"
            )
        snapshot_sha256 = existing.get("checkpoint_sha256")
        if not isinstance(snapshot_sha256, str) or len(snapshot_sha256) != 64:
            raise RuntimeError(
                f"invalid immutable actor launch checkpoint_sha256 at {path}"
            )
        if not snapshot.is_file():
            raise RuntimeError(
                f"immutable actor launch policy checkpoint is unavailable: {snapshot}"
            )
        actual_snapshot_sha256 = dist.checkpoint_sha256(snapshot)
        if actual_snapshot_sha256 != snapshot_sha256:
            raise RuntimeError(
                "immutable actor launch policy checkpoint failed SHA-256 "
                f"authentication: expected={snapshot_sha256} "
                f"actual={actual_snapshot_sha256} path={snapshot}"
            )
    else:
        if snapshot_binding.exists():
            try:
                staged = json.loads(snapshot_binding.read_text(encoding="utf-8"))
                snapshot_sha256 = str(staged["checkpoint_sha256"])
                policy_version = int(staged["policy_version"])
                policy_step = int(staged["policy_step"])
                policy_updated_at = float(staged["policy_updated_at"])
                stream_nonce = int(staged["stream_nonce"])
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"invalid staged actor policy snapshot at {snapshot_binding}"
                ) from error
            if not snapshot.is_file() or dist.checkpoint_sha256(snapshot) != snapshot_sha256:
                raise RuntimeError(
                    f"staged actor policy snapshot failed authentication: {snapshot}"
                )
        else:
            # A snapshot without its staged binding is from a crash before any
            # launch identity became authoritative and is safe to reconstruct.
            snapshot.unlink(missing_ok=True)
            snapshot_sha256 = _snapshot_policy(Path(published.path), snapshot)
            policy_version = int(published.version)
            policy_step = int(getattr(published, "step", 0))
            policy_updated_at = float(getattr(published, "updated_at", 0.0))
            stream_nonce = uuid.uuid4().int
            _atomic_bind_json(
                snapshot_binding,
                {
                    "schema": "local_entity_ppo_policy_snapshot_v1",
                    "checkpoint_sha256": snapshot_sha256,
                    "policy_version": policy_version,
                    "policy_step": policy_step,
                    "policy_updated_at": policy_updated_at,
                    "stream_nonce": stream_nonce,
                },
            )
        published = dist.PublishedVersion(
            version=policy_version,
            step=policy_step,
            updated_at=policy_updated_at,
            path=str(snapshot),
        )
    immutable = {
        "schema": _LAUNCH_SCHEMA,
        "launch_id": str(launch_id),
        "policy_version": int(published.version),
        "policy_step": int(getattr(published, "step", 0)),
        "policy_updated_at": float(getattr(published, "updated_at", 0.0)),
        "checkpoint": str(snapshot.resolve()),
        "checkpoint_sha256": snapshot_sha256,
        "architecture": str(args.architecture),
        "track": str(args.track),
        "vps_to_win": int(args.vps_to_win),
        "games": int(args.games),
        "workers": int(args.workers),
        "games_per_shard": int(args.games_per_shard),
        "max_decisions": int(args.max_decisions),
        "opponents": str(args.opponents),
        "opponent_mode": str(args.opponent_mode),
        "pfsp_mode": str(args.pfsp_mode),
        "base_seed": int(args.seed),
        "gamma": float(args.gamma),
        "gae_lambda": float(args.gae_lambda),
        "value_shaping_coef": float(args.value_shaping_coef),
        "value_shaping_scale": float(args.value_shaping_scale),
        "value_shaping_opponent_penalty": float(
            args.value_shaping_opponent_penalty
        ),
        "action_temperature": float(args.action_temperature),
        "run_manifest_sha256": args.run_manifest_sha256,
    }
    if existing is not None:
        return _atomic_bind_json(path, {**immutable, "stream_nonce": stream_nonce})
    # The nonce occupies the high bits of every game/seed namespace. A launch
    # therefore cannot repeat another launch's schedule even when --seed is unchanged.
    return _atomic_bind_json(path, {**immutable, "stream_nonce": stream_nonce})


@contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _prepare_launch(args: argparse.Namespace, published: Any) -> dict[str, Any]:
    """Allocate a fresh stream namespace or verify an explicit deterministic resume."""

    requested_launch_id = getattr(args, "launch_id", None)
    launch_id = str(requested_launch_id or f"local-{uuid.uuid4().hex}")
    if _LAUNCH_ID_PATTERN.fullmatch(launch_id) is None:
        raise RuntimeError(
            "--launch-id must be 1-64 characters using only letters, digits, '.', '_' or '-'"
        )
    root = dist.run_root(args.run_base, args.run_name)
    launch_dir = _launches_dir(root) / launch_id
    with _file_lock(launch_dir / ".creation.lock"):
        return _prepare_launch_unlocked(args, published, launch_id)


def _resolve_published_weights(args: argparse.Namespace, root: Path) -> Any:
    """Read learned weights, publishing the initializer only for an empty run."""

    with _file_lock(dist.policy_dir(root) / ".actor-bootstrap.lock"):
        existing = dist.read_version(root)
        if existing is not None:
            if args.publish:
                raise RuntimeError(
                    "--publish is bootstrap-only and cannot replace existing policy "
                    f"version {existing.version} (learner step {existing.step})"
                )
            return existing
        policy = load_ppo_policy(
            args.checkpoint, architecture=args.architecture, device="cpu"
        )
        return dist.publish_weights(root, policy.save, step=0)


def _published_from_launch(launch: dict[str, Any]) -> dist.PublishedVersion:
    return dist.PublishedVersion(
        version=int(launch["policy_version"]),
        step=int(launch.get("policy_step", 0)),
        updated_at=float(launch.get("policy_updated_at", 0.0)),
        path=str(launch["checkpoint"]),
    )


def _load_launch_completion(
    root: Path, launch: dict[str, Any]
) -> dict[str, Any] | None:
    completion = dist.load_launch_completion(root, str(launch["launch_id"]))
    if completion is None:
        return None
    expected = dist.launch_completion_payload(launch)
    if completion != expected:
        raise RuntimeError(
            "launch completion receipt does not match the requested launch contract"
        )
    return completion


def _finalize_launch_if_complete(
    root: Path, launch: dict[str, Any]
) -> dict[str, Any] | None:
    """Aggregate a fully produced schedule and release per-launch storage."""

    launch_dir = _launches_dir(root) / str(launch["launch_id"])
    with _file_lock(launch_dir / ".creation.lock"):
        completion = _load_launch_completion(root, launch)
        expected_shards = dist.expected_launch_shards(launch)
        if completion is None:
            for relative in expected_shards:
                shard = dist.trajectories_dir(root) / relative
                if not dist.trajectory_is_complete(root, shard):
                    return None
            completion = _atomic_bind_json(
                dist.launch_completion_path(root, str(launch["launch_id"])),
                dist.launch_completion_payload(launch),
            )
            completion = _load_launch_completion(root, launch)
        if completion is None:  # defensive: atomic binding must be immediately visible
            raise RuntimeError("failed to authenticate bound launch completion receipt")
        Path(launch["checkpoint"]).unlink(missing_ok=True)
        (launch_dir / "policy_snapshot.json").unlink(missing_ok=True)
        for relative in expected_shards:
            dist.trajectory_completion_path(
                root, dist.trajectories_dir(root) / relative
            ).unlink(missing_ok=True)
        shutil.rmtree(launch_dir / "shard_claims", ignore_errors=True)
        return completion


def _resolve_launch_and_weights(
    args: argparse.Namespace, root: Path
) -> tuple[dict[str, Any], Any]:
    """Resume bound launch weights, or bind a fresh launch to current weights."""

    if args.launch_id:
        launch_dir = _launches_dir(root) / str(args.launch_id)
        contract_path = launch_dir / "launch.json"
        if contract_path.exists():
            try:
                existing = json.loads(contract_path.read_text(encoding="utf-8"))
                completion = _load_launch_completion(root, existing)
                if completion is not None:
                    return existing, None
                checkpoint = Path(existing["checkpoint"])
                expected_checkpoint_sha256 = str(existing["checkpoint_sha256"])
                policy_version = int(existing["policy_version"])
                policy_step = int(existing.get("policy_step", 0))
                policy_updated_at = float(existing.get("policy_updated_at", 0.0))
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"invalid immutable actor launch contract at {contract_path}"
                ) from error
            if not checkpoint.is_file():
                raise RuntimeError(
                    "cannot resume actor launch because its immutable policy checkpoint "
                    f"is unavailable: {checkpoint}"
                )
            actual_checkpoint_sha256 = dist.checkpoint_sha256(checkpoint)
            if actual_checkpoint_sha256 != expected_checkpoint_sha256:
                raise RuntimeError(
                    "cannot resume actor launch because its immutable policy checkpoint "
                    "failed SHA-256 authentication: "
                    f"expected={expected_checkpoint_sha256} "
                    f"actual={actual_checkpoint_sha256} path={checkpoint}"
                )
            published = dist.PublishedVersion(
                version=policy_version,
                step=policy_step,
                updated_at=policy_updated_at,
                path=str(checkpoint),
            )
            launch = _prepare_launch(args, published)
            return launch, _published_from_launch(launch)
        snapshot_binding = launch_dir / "policy_snapshot.json"
        if snapshot_binding.exists():
            try:
                staged = json.loads(snapshot_binding.read_text(encoding="utf-8"))
                published = dist.PublishedVersion(
                    version=int(staged["policy_version"]),
                    step=int(staged["policy_step"]),
                    updated_at=float(staged["policy_updated_at"]),
                    path=str(launch_dir / "policy.pt"),
                )
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"invalid staged actor policy snapshot at {snapshot_binding}"
                ) from error
            launch = _prepare_launch(args, published)
            return launch, _published_from_launch(launch)
    published = _resolve_published_weights(args, root)
    launch = _prepare_launch(args, published)
    return launch, _published_from_launch(launch)


def _build_worker_payloads(
    args: argparse.Namespace,
    published: Any,
    launch: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], int]:
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        devices = ["cpu"]
    if published is None:
        return [], devices, 0
    workers = max(1, int(args.workers))
    games = max(0, int(args.games))
    base = games // workers
    remainder = games % workers
    payloads: list[dict[str, Any]] = []
    offset = 0
    for worker in range(workers):
        worker_games = base + (1 if worker < remainder else 0)
        if worker_games <= 0:
            continue
        payloads.append(
            {
                "run_base": str(args.run_base),
                "run_name": str(args.run_name),
                "launch_id": str(launch["launch_id"]),
                "worker_id": f"local_{launch['launch_id']}_{worker:03d}",
                "checkpoint": str(launch["checkpoint"]),
                "policy_version": int(launch["policy_version"]),
                "architecture": str(args.architecture),
                "device": devices[worker % len(devices)],
                "track": str(args.track),
                "vps_to_win": int(args.vps_to_win),
                "games": int(worker_games),
                "game_offset": (int(launch["stream_nonce"]) << 64) + int(offset),
                "games_per_shard": int(args.games_per_shard),
                "max_decisions": int(args.max_decisions),
                "opponents": str(args.opponents),
                "opponent_mode": str(args.opponent_mode),
                "pfsp_mode": str(args.pfsp_mode),
                "seed": (
                    (int(launch["stream_nonce"]) << 64)
                    + (int(args.seed) & ((1 << 64) - 1))
                    + worker * 1_000_003
                ),
                "gamma": float(args.gamma),
                "gae_lambda": float(args.gae_lambda),
                "value_shaping_coef": float(args.value_shaping_coef),
                "value_shaping_scale": float(args.value_shaping_scale),
                "value_shaping_opponent_penalty": float(args.value_shaping_opponent_penalty),
                "action_temperature": float(args.action_temperature),
                "run_manifest_sha256": args.run_manifest_sha256,
            }
        )
        offset += worker_games
    return payloads, devices, games


def main(argv: list[str] | None = None) -> None:
    args, manifest = resolve_config(argv)

    root = _bind_run_root(args, manifest)
    launch, published = _resolve_launch_and_weights(args, root)
    payloads, devices, games = _build_worker_payloads(args, published, launch)

    print(
        json.dumps(
            {
                "event": "local_ppo_shards_start",
                "run_root": str(root),
                "policy_version": int(launch["policy_version"]),
                "checkpoint": None if published is None else str(published.path),
                "launch_id": str(launch["launch_id"]),
                "games": games,
                "workers": len(payloads),
                "devices": devices,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    started = time.perf_counter()
    reports: list[dict[str, Any]] = []
    if payloads:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(payloads), mp_context=ctx) as executor:
            futures = [executor.submit(_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                report = future.result()
                reports.append(report)
                print(
                    json.dumps(
                        {"event": "local_ppo_worker_done", **report}, sort_keys=True
                    ),
                    flush=True,
                )
    launch_completion = _finalize_launch_if_complete(root, launch)
    total_games = sum(int(report["games"]) for report in reports)
    total_samples = sum(int(report["samples"]) for report in reports)
    total_shards = sum(int(report["shards"]) for report in reports)
    print(
        json.dumps(
            {
                "event": "local_ppo_shards_done",
                "run_root": str(root),
                "games": total_games,
                "samples": total_samples,
                "shards": total_shards,
                "launch_complete": launch_completion is not None,
                "elapsed_sec": time.perf_counter() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    root = dist.run_root(payload["run_base"], payload["run_name"])
    if str(payload.get("opponent_mode", "fixed")) != "fixed":
        raise RuntimeError(
            "local PPO worker supports only fixed opponents; league/PFSP is unavailable"
        )
    policy = load_ppo_policy(
        payload["checkpoint"],
        architecture=payload["architecture"],
        device=payload["device"],
    )
    model = getattr(policy, "model", None)
    if model is not None:
        model.eval()
    config = parse_track(payload["track"], vps_to_win=int(payload["vps_to_win"]))
    player_count = int(config.players)
    seats = SEATS[:player_count]
    opponent_names = [name.strip() for name in str(payload["opponents"]).split(",") if name.strip()]
    if not opponent_names:
        opponent_names = ["random"]
    samples = 0
    shards = 0
    games_collected = 0
    resumed_games = 0
    resumed_shards = 0
    started = time.perf_counter()
    worker_id = str(payload["worker_id"])
    worker_dir = dist.trajectories_dir(root, worker_id)
    worker_dir.mkdir(parents=True, exist_ok=True)
    games_per_shard = max(1, int(payload["games_per_shard"]))
    total_games = int(payload["games"])
    claims_dir = _launches_dir(root) / str(payload["launch_id"]) / "shard_claims"

    def shard_is_complete(shard_path: Path, consumed_marker: Path) -> bool:
        if dist.trajectory_is_complete(root, shard_path):
            return True
        if shard_path.exists() or consumed_marker.exists():
            # Repair launches interrupted between atomic publication/consumption
            # and their permanent completion receipt.
            dist.mark_trajectory_complete(root, shard_path)
            return True
        return False

    for shard_index, shard_start in enumerate(range(0, total_games, games_per_shard)):
        shard_games = min(games_per_shard, total_games - shard_start)
        shard_path = worker_dir / f"shard_{shard_index:06d}.pkl"
        consumed_marker = (
            dist.consumed_dir(root) / f"{worker_id}__{shard_path.name}"
        )
        claim = {
            "schema": "local_entity_ppo_shard_claim_v1",
            "launch_id": str(payload["launch_id"]),
            "worker_id": worker_id,
            "shard_index": shard_index,
            "game_start": shard_start,
            "games": shard_games,
        }
        _atomic_bind_json(
            claims_dir / worker_id / f"shard_{shard_index:06d}.json", claim
        )
        if shard_is_complete(shard_path, consumed_marker):
            resumed_games += shard_games
            resumed_shards += 1
            continue
        lock_path = claims_dir / worker_id / f"shard_{shard_index:06d}.lock"
        with _file_lock(lock_path):
            # A second resumed process may have finished while this one waited.
            if shard_is_complete(shard_path, consumed_marker):
                resumed_games += shard_games
                resumed_shards += 1
                continue
            buffer = []
            for game_index in range(shard_start, shard_start + shard_games):
                global_game = int(payload["game_offset"]) + game_index
                training_seat = seats[global_game % player_count]
                # Per-game randomness makes retries independent from which earlier
                # shards have already been consumed while preserving exact replay.
                rng = np.random.default_rng(int(payload["seed"]) + game_index)
                opponents = {}
                for seat in seats:
                    if seat == training_seat:
                        continue
                    name = opponent_names[int(rng.integers(0, len(opponent_names)))]
                    opponents[seat] = make_named_policy(name)
                trajectory = collect_ppo_episode(
                    policy,
                    opponents,
                    seed=int(rng.integers(2**31)),
                    config=config,
                    max_decisions=int(payload["max_decisions"]),
                    rng=rng,
                    training_seats={training_seat},
                    gamma=float(payload["gamma"]),
                    gae_lambda=float(payload["gae_lambda"]),
                    value_shaping_coef=float(payload["value_shaping_coef"]),
                    value_shaping_scale=float(payload["value_shaping_scale"]),
                    value_shaping_opponent_penalty=float(
                        payload["value_shaping_opponent_penalty"]
                    ),
                    action_temperature=float(payload["action_temperature"]),
                )
                buffer.append(trajectory)
                samples += len(trajectory.samples)
                games_collected += 1
            # The immutable worker namespace plus the held shard claim means this
            # atomic writer is create-only from the actor's perspective.
            if shard_path.exists() or consumed_marker.exists():
                raise RuntimeError(
                    f"refusing to replace completed PPO shard {shard_path}"
                )
            dist.write_trajectory_shard(
                root,
                worker_id,
                shard_index,
                buffer,
                policy_version=int(payload["policy_version"]),
                run_manifest_sha256=payload.get("run_manifest_sha256"),
            )
            dist.mark_trajectory_complete(root, shard_path)
            shards += 1
    return {
        "launch_id": str(payload["launch_id"]),
        "worker_id": worker_id,
        "device": str(payload["device"]),
        "games": games_collected,
        "samples": int(samples),
        "shards": int(shards),
        "resumed_games": resumed_games,
        "resumed_shards": resumed_shards,
        "elapsed_sec": time.perf_counter() - started,
    }


if __name__ == "__main__":
    main()
