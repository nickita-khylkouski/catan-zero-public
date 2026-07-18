"""Distributed-PPO backbone: the on-disk actor/learner contract.

An actor fleet and the learner
(``tools/ppo_distributed_learner.py``) never talk directly — they coordinate ONLY through a
shared run directory (a Modal volume path, or any shared filesystem):

    {run_root}/
      policy/
        weights_v{N}.pt     # versioned weights for policy version N (self-describing)
        current.pt          # back-compat copy/symlink of the latest weights_v{N}.pt
        version.json        # {"version": int, "step": int, "updated_at": float, "weights": "weights_v{N}.pt"}
      trajectories/
        {worker_id}/
          shard_000123.pkl  # a pickled list[PPOTrajectory] from one actor
      consumed/
        {worker_id}__shard_000123.pkl   # empty marker; learner marks shards it has ingested
      trajectory_completed/
        {worker_id}__shard_000123.pkl.json  # permanent no-regeneration receipt
      checkpoints/          # periodic learner checkpoints
      league/               # League.save() / League.load() (see league.py)
      eval/                 # scoreboard json outputs

This module owns: path layout, atomic weight versioning, and trajectory-shard read/write +
the consumed-marker protocol. It is deliberately agnostic to the policy's checkpoint format —
the learner passes a ``save_fn(path)`` (e.g. ``policy.save``) and the actor loads from the
returned path with its own loader. Pure stdlib + pickle; no torch/env import at module load.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from catan_zero.rl.ppo_run_manifest import PPORunManifest

POLICY_DIRNAME = "policy"
CURRENT_WEIGHTS_NAME = "current.pt"
VERSION_FILENAME = "version.json"
VERSIONED_WEIGHTS_GLOB = "weights_v*.pt"
RUN_CONTRACT_FILENAME = "run_contract.json"
RUN_CONTRACT_SCHEMA = "canonical_entity_ppo_run_v1"
RUN_MANIFEST_FILENAME = "run_manifest_v2.json"
RUN_MANIFEST_BINDING_SCHEMA = "canonical_entity_ppo_run_binding_v2"
# How many old versioned weight files to keep on disk (newest N). The rest are GC'd by
# publish_weights so a multi-day run does not accumulate hundreds of stale checkpoints.
KEEP_VERSIONED_WEIGHTS = 3
TRAJ_DIRNAME = "trajectories"
CONSUMED_DIRNAME = "consumed"
COMPLETED_DIRNAME = "trajectory_completed"
CHECKPOINTS_DIRNAME = "checkpoints"
LEAGUE_DIRNAME = "league"
EVAL_DIRNAME = "eval"


# ----------------------------------------------------------------------------- paths
def run_root(base: str | os.PathLike, run_name: str) -> Path:
    return Path(base) / run_name


def policy_dir(root: str | os.PathLike) -> Path:
    return Path(root) / POLICY_DIRNAME


def current_weights_path(root: str | os.PathLike) -> Path:
    """Back-compat path to the latest weights (a copy of ``weights_v{N}.pt``).

    Prefer ``read_version(root).path`` (which points at the version-stamped file) when you need
    the bytes to match the policy version you stamp. ``current.pt`` is maintained for callers that
    only want "the latest weights" and do not care about the version tie-break.
    """
    return policy_dir(root) / CURRENT_WEIGHTS_NAME


def versioned_weights_path(root: str | os.PathLike, version: int) -> Path:
    return policy_dir(root) / f"weights_v{int(version)}.pt"


def version_path(root: str | os.PathLike) -> Path:
    return policy_dir(root) / VERSION_FILENAME


def run_contract_path(root: str | os.PathLike) -> Path:
    return Path(root) / RUN_CONTRACT_FILENAME


def run_manifest_path(root: str | os.PathLike) -> Path:
    return Path(root) / RUN_MANIFEST_FILENAME


def trajectories_dir(root: str | os.PathLike, worker_id: str | None = None) -> Path:
    base = Path(root) / TRAJ_DIRNAME
    return base / worker_id if worker_id else base


def consumed_dir(root: str | os.PathLike) -> Path:
    return Path(root) / CONSUMED_DIRNAME


def completed_dir(root: str | os.PathLike) -> Path:
    """Permanent actor/learner receipts; unlike consumed markers these are never pruned."""

    return Path(root) / COMPLETED_DIRNAME


def checkpoints_dir(root: str | os.PathLike) -> Path:
    return Path(root) / CHECKPOINTS_DIRNAME


def league_dir(root: str | os.PathLike) -> Path:
    return Path(root) / LEAGUE_DIRNAME


def eval_dir(root: str | os.PathLike) -> Path:
    return Path(root) / EVAL_DIRNAME


def ensure_run_dirs(root: str | os.PathLike) -> None:
    for d in (policy_dir(root), trajectories_dir(root), consumed_dir(root), completed_dir(root),
              checkpoints_dir(root), league_dir(root), eval_dir(root)):
        Path(d).mkdir(parents=True, exist_ok=True)


def checkpoint_sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_run_contract(
    root: str | os.PathLike,
    *,
    init_checkpoint: str | os.PathLike,
    architecture: str,
    gamma: float,
    gae_lambda: float,
    behavior_temperature: float,
) -> dict[str, Any]:
    """Atomically create or verify the immutable PPO initializer/actor contract."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(init_checkpoint).resolve()
    payload = {
        "schema": RUN_CONTRACT_SCHEMA,
        "initializer_sha256": checkpoint_sha256(checkpoint),
        "architecture": str(architecture),
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "behavior_temperature": float(behavior_temperature),
    }
    path = run_contract_path(root_path)

    def verify_existing() -> dict[str, Any]:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid PPO run contract at {path}") from error
        if existing != payload:
            raise RuntimeError(
                "PPO run contract mismatch: immutable initializer/actor binding changed; "
                f"expected={existing!r} requested={payload!r}"
            )
        return existing

    if path.exists():
        return verify_existing()

    # Publish complete bytes with atomic create-if-absent semantics. Multiple
    # actors may race; the loser verifies the winner's exact payload.
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
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return verify_existing()


class RunManifestError(RuntimeError):
    """A production PPO run or trajectory does not match its bound v2 manifest."""


def _require_manifest_sha256(value: Any, *, where: str) -> str:
    if (
        type(value) is not str
        or not value.startswith("sha256:")
        or len(value) != len("sha256:") + 64
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise RunManifestError(f"{where} must be sha256:<64 lowercase hex>")
    return value


def _assert_pristine_v2_root(root: Path) -> None:
    allowed_empty_directories = {
        POLICY_DIRNAME,
        TRAJ_DIRNAME,
        CONSUMED_DIRNAME,
        COMPLETED_DIRNAME,
        CHECKPOINTS_DIRNAME,
        LEAGUE_DIRNAME,
        EVAL_DIRNAME,
    }
    temporary_prefix = f".{RUN_MANIFEST_FILENAME}."
    for entry in root.iterdir():
        if entry.name in allowed_empty_directories:
            if entry.is_symlink() or not entry.is_dir() or any(entry.iterdir()):
                raise RunManifestError(
                    f"refusing to bind v2 manifest over preexisting runtime artifacts: {entry}"
                )
            continue
        if entry.name == RUN_MANIFEST_FILENAME:
            # Another concurrent binder may have published or be publishing the
            # same immutable file after this caller's initial existence check.
            continue
        if entry.name.startswith(temporary_prefix) and entry.name.endswith(".tmp"):
            if entry.is_symlink():
                raise RunManifestError(
                    f"refusing to bind v2 manifest over preexisting runtime artifacts: {entry}"
                )
            if entry.is_file() or not entry.exists():
                # A peer may unlink its temporary between directory iteration
                # and this check after successfully publishing the hard link.
                continue
            raise RunManifestError(
                f"refusing to bind v2 manifest over preexisting runtime artifacts: {entry}"
            )
        raise RunManifestError(
            f"refusing to bind v2 manifest over preexisting runtime artifacts: {entry}"
        )


def bind_run_manifest(
    root: str | os.PathLike, manifest: PPORunManifest
) -> dict[str, Any]:
    """Atomically bind an exact, executable v2 manifest to a new production root.

    The historical v1 ``run_contract.json`` is deliberately neither read nor
    rewritten by this API. A root already carrying that contract must remain a
    v1 root; v2 commissioning starts in a separate directory.
    """

    if not isinstance(manifest, PPORunManifest):
        raise RunManifestError("v2 run binding requires a PPORunManifest")
    if manifest.status != "bound":
        raise RunManifestError("v2 production binding requires manifest status='bound'")

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    if run_contract_path(root_path).exists():
        raise RunManifestError("refusing to bind v2 manifest over a historical v1 root")

    manifest_sha256 = _require_manifest_sha256(
        manifest.sha256(), where="manifest sha256"
    )
    payload = {
        "schema": RUN_MANIFEST_BINDING_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "manifest": json.loads(manifest.canonical_json()),
    }
    path = run_manifest_path(root_path)

    def verify_existing() -> dict[str, Any]:
        if path.is_symlink() or not path.is_file():
            raise RunManifestError(
                f"invalid PPO v2 run manifest binding file at {path}"
            )
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RunManifestError(
                f"invalid PPO v2 run manifest binding at {path}"
            ) from error
        if existing != payload:
            raise RunManifestError(
                "PPO v2 run manifest mismatch: immutable production identity changed; "
                f"expected={existing!r} requested={payload!r}"
            )
        return existing

    if path.exists():
        return verify_existing()

    _assert_pristine_v2_root(root_path)

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
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return verify_existing()


# ------------------------------------------------------------------- weight versioning
@dataclass(frozen=True)
class PublishedVersion:
    version: int
    step: int
    updated_at: float
    path: str  # absolute path to the version-stamped weights (weights_v{version}.pt)


def _gc_versioned_weights(root: str | os.PathLike, *, keep: int = KEEP_VERSIONED_WEIGHTS) -> int:
    """Delete all but the newest ``keep`` ``weights_v{N}.pt`` files. Returns count removed."""
    pdir = policy_dir(root)
    versioned: list[tuple[int, Path]] = []
    for p in pdir.glob(VERSIONED_WEIGHTS_GLOB):
        try:
            n = int(p.stem.split("_v", 1)[1])
        except (IndexError, ValueError):
            continue
        versioned.append((n, p))
    versioned.sort(key=lambda t: t[0], reverse=True)
    removed = 0
    for _, p in versioned[keep:]:
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def publish_weights(root: str | os.PathLike, save_fn: Callable[[str], Any], *, step: int) -> PublishedVersion:
    """Atomically publish new weights so version and bytes can NEVER disagree (FIX H1).

    ``save_fn(tmp_path)`` must write a loadable checkpoint. The protocol:

      1. write ``weights_v{N}.tmp`` then ``os.replace`` -> ``weights_v{N}.pt`` (version-stamped,
         atomic on the same fs);
      2. refresh the back-compat ``current.pt`` copy (best-effort; not the source of truth);
      3. atomically swap ``version.json`` to ``{"version": N, "weights": "weights_v{N}.pt", ...}``.

    Because the weights filename is itself version-stamped, ``read_version`` returns
    ``(version=N, path=weights_v{N}.pt)`` and an actor that stamps ``policy_version=N`` is
    guaranteed to have loaded the matching bytes — even if the learner publishes N+1 in between.
    Old ``weights_v{N}.pt`` files are GC'd keeping the newest ``KEEP_VERSIONED_WEIGHTS``.
    """
    pdir = policy_dir(root)
    pdir.mkdir(parents=True, exist_ok=True)
    prev = read_version(root)
    version = (prev.version + 1) if prev else 1

    final = versioned_weights_path(root, version)
    tmp = final.with_suffix(final.suffix + ".tmp")
    save_fn(str(tmp))
    os.replace(tmp, final)  # atomic rename -> version-stamped weights

    # Back-compat: refresh current.pt as a copy of the just-published weights. Best-effort:
    # version.json (not current.pt) is the source of truth, so a failure here is non-fatal.
    try:
        cur = current_weights_path(root)
        cur_tmp = cur.with_suffix(cur.suffix + ".tmp")
        cur_tmp.write_bytes(final.read_bytes())
        os.replace(cur_tmp, cur)
    except OSError:
        pass

    meta = {
        "version": version,
        "step": int(step),
        "updated_at": time.time(),
        "weights": final.name,
    }
    vtmp = version_path(root).with_suffix(".json.tmp")
    vtmp.write_text(json.dumps(meta))
    os.replace(vtmp, version_path(root))  # atomic swap -> now points at weights_v{N}.pt

    _gc_versioned_weights(root)
    return PublishedVersion(version=version, step=int(step), updated_at=meta["updated_at"], path=str(final))


def read_version(root: str | os.PathLike) -> PublishedVersion | None:
    """Return the published version + the path to the version-stamped weights it describes.

    The returned ``path`` points at ``weights_v{N}.pt`` (the file named in ``version.json``), so the
    bytes always match ``version`` (FIX H1). Falls back to ``current.pt`` only for versions written
    by the legacy publisher (no ``weights`` key); returns ``None`` if neither exists.
    """
    vp = version_path(root)
    if not vp.exists():
        return None
    try:
        meta = json.loads(vp.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    version = int(meta.get("version", 0))
    weights_name = meta.get("weights")
    if weights_name:
        wpath = policy_dir(root) / str(weights_name)
    else:  # legacy version.json without a versioned filename: fall back to current.pt
        wpath = current_weights_path(root)
    if not wpath.exists():
        return None
    return PublishedVersion(
        version=version,
        step=int(meta.get("step", 0)),
        updated_at=float(meta.get("updated_at", 0.0)),
        path=str(wpath),
    )


# --------------------------------------------------------------- trajectory shards
def write_trajectory_shard(
    root: str | os.PathLike,
    worker_id: str,
    shard_index: int,
    trajectories: list[Any],
    *,
    policy_version: int,
    run_manifest_sha256: str | None = None,
) -> Path:
    """Pickle a list[PPOTrajectory] to ``trajectories/{worker_id}/shard_{n:06d}.pkl`` atomically.

    Wrapped in an envelope so the learner can filter by policy_version (staleness) and worker.
    """
    wdir = trajectories_dir(root, worker_id)
    wdir.mkdir(parents=True, exist_ok=True)
    path = wdir / f"shard_{shard_index:06d}.pkl"
    tmp = path.with_suffix(".pkl.tmp")
    envelope = {
        "worker_id": worker_id,
        "shard_index": int(shard_index),
        "policy_version": int(policy_version),
        "created_at": time.time(),
        "trajectories": trajectories,
    }
    if run_manifest_sha256 is not None:
        envelope["run_manifest_sha256"] = _require_manifest_sha256(
            run_manifest_sha256, where="trajectory run_manifest_sha256"
        )
    with open(tmp, "wb") as fh:
        pickle.dump(envelope, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path


def read_trajectory_shard(
    path: str | os.PathLike, *, expected_run_manifest_sha256: str | None = None
) -> dict[str, Any]:
    """Return the envelope dict; ``envelope['trajectories']`` is the list[PPOTrajectory]."""
    if expected_run_manifest_sha256 is not None:
        expected_run_manifest_sha256 = _require_manifest_sha256(
            expected_run_manifest_sha256, where="expected run_manifest_sha256"
        )
    with open(path, "rb") as fh:
        envelope = pickle.load(fh)
    if expected_run_manifest_sha256 is not None:
        if not isinstance(envelope, dict):
            raise RunManifestError("trajectory shard envelope must be an object")
        actual = envelope.get("run_manifest_sha256")
        if actual != expected_run_manifest_sha256:
            raise RunManifestError(
                "trajectory run manifest mismatch: "
                f"expected={expected_run_manifest_sha256!r} actual={actual!r}"
            )
    return envelope


def _consumed_marker(root: str | os.PathLike, shard_path: Path) -> Path:
    # flatten {worker_id}/{shard}.pkl -> {worker_id}__{shard}.pkl marker
    rel = shard_path.resolve().relative_to(trajectories_dir(root).resolve())
    return consumed_dir(root) / str(rel).replace(os.sep, "__")


def trajectory_completion_path(
    root: str | os.PathLike, shard_path: str | os.PathLike
) -> Path:
    """Return the permanent production-completion receipt for one shard path."""

    path = Path(shard_path).resolve()
    try:
        relative = path.relative_to(trajectories_dir(root).resolve())
    except ValueError as error:
        raise ValueError(f"trajectory shard is outside run root: {path}") from error
    return completed_dir(root) / f"{str(relative).replace(os.sep, '__')}.json"


def expected_launch_shards(launch: dict[str, Any]) -> list[str]:
    """Return the exact worker/shard schedule authenticated by a launch contract."""

    games = int(launch["games"])
    workers = max(1, int(launch["workers"]))
    games_per_shard = max(1, int(launch["games_per_shard"]))
    launch_id = str(launch["launch_id"])
    base, remainder = divmod(max(0, games), workers)
    expected: list[str] = []
    for worker in range(workers):
        worker_games = base + (1 if worker < remainder else 0)
        worker_id = f"local_{launch_id}_{worker:03d}"
        shard_count = (worker_games + games_per_shard - 1) // games_per_shard
        expected.extend(
            f"{worker_id}/shard_{shard_index:06d}.pkl"
            for shard_index in range(shard_count)
        )
    return expected


def launch_completion_path(root: str | os.PathLike, launch_id: str) -> Path:
    return Path(root) / "actor_launches" / str(launch_id) / "launch_complete.json"


def launch_completion_payload(launch: dict[str, Any]) -> dict[str, Any]:
    contract_digest = hashlib.sha256(
        json.dumps(launch, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    completed_shards = expected_launch_shards(launch)
    schedule_digest = hashlib.sha256(
        json.dumps(completed_shards, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "local_entity_ppo_launch_complete_v1",
        "launch_id": str(launch["launch_id"]),
        "launch_contract_sha256": contract_digest,
        "completed_shard_count": len(completed_shards),
        "completed_schedule_sha256": schedule_digest,
        "games": int(launch["games"]),
        "policy_version": int(launch["policy_version"]),
    }


def load_launch_completion(
    root: str | os.PathLike, launch_id: str
) -> dict[str, Any] | None:
    """Authenticate and return one aggregate launch completion receipt."""

    completion = launch_completion_path(root, launch_id)
    if not completion.exists():
        return None
    contract_path = completion.parent / "launch.json"
    try:
        launch = json.loads(contract_path.read_text(encoding="utf-8"))
        actual = json.loads(completion.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid launch completion receipt at {completion}") from error
    expected = launch_completion_payload(launch)
    if actual != expected:
        raise RuntimeError(
            "launch completion receipt mismatch: "
            f"expected={expected!r} actual={actual!r}"
        )
    return actual


def _aggregate_completion_for_shard(
    root: str | os.PathLike, shard_path: str | os.PathLike
) -> Path | None:
    path = Path(shard_path).resolve()
    try:
        relative = path.relative_to(trajectories_dir(root).resolve())
    except ValueError as error:
        raise ValueError(f"trajectory shard is outside run root: {path}") from error
    if len(relative.parts) != 2:
        return None
    worker_id = relative.parts[0]
    if not worker_id.startswith("local_") or len(worker_id) < len("local_x_000"):
        return None
    launch_id, separator, worker_suffix = worker_id[len("local_") :].rpartition("_")
    if separator != "_" or len(worker_suffix) != 3 or not worker_suffix.isdigit():
        return None
    completion = load_launch_completion(root, launch_id)
    if completion is None:
        return None
    shard_key = f"{worker_id}/{relative.name}"
    try:
        launch = json.loads(
            (launch_completion_path(root, launch_id).parent / "launch.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid launch contract for completed launch {launch_id}") from error
    if shard_key not in expected_launch_shards(launch):
        return None
    return launch_completion_path(root, launch_id)


def mark_trajectory_complete(
    root: str | os.PathLike, shard_path: str | os.PathLike
) -> Path:
    """Atomically record that a shard was published and must never be produced again.

    Actors call this only after atomic shard publication. The learner repeats the
    same idempotent binding before deleting a consumed shard, closing the actor
    crash/learner-consume race. These receipts intentionally outlive the bounded
    consumed-marker retention window.
    """

    aggregate = _aggregate_completion_for_shard(root, shard_path)
    if aggregate is not None:
        return aggregate
    path = Path(shard_path).resolve()
    try:
        relative = path.relative_to(trajectories_dir(root).resolve())
    except ValueError as error:
        raise ValueError(f"trajectory shard is outside run root: {path}") from error
    if len(relative.parts) != 2 or not relative.name.startswith("shard_"):
        raise ValueError(f"invalid trajectory shard path: {path}")
    receipt = trajectory_completion_path(root, path)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ppo_trajectory_production_completed_v1",
        "worker_id": relative.parts[0],
        "shard": relative.name,
    }
    temporary = receipt.with_name(
        f".{receipt.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    try:
        try:
            os.link(temporary, receipt)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)
    try:
        existing = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid trajectory completion receipt at {receipt}") from error
    if existing != payload:
        raise RuntimeError(
            "trajectory completion receipt mismatch: "
            f"expected={existing!r} requested={payload!r}"
        )
    return receipt


def trajectory_is_complete(
    root: str | os.PathLike, shard_path: str | os.PathLike
) -> bool:
    if _aggregate_completion_for_shard(root, shard_path) is not None:
        return True
    receipt = trajectory_completion_path(root, shard_path)
    if not receipt.is_file():
        return False
    mark_trajectory_complete(root, shard_path)  # authenticate the immutable receipt
    return True


def iter_unconsumed_shards(
    root: str | os.PathLike,
    *,
    max_shards: int | None = None,
    min_policy_version: int = 0,
    stable_secs: float = 0.0,
    max_policy_version: int | None = None,
    with_envelope: bool = False,
    newest_first: bool = False,
    volume_reload_fn: Callable[[], Any] | None = None,
    expected_run_manifest_sha256: str | None = None,
) -> Iterator[Any]:
    """Yield shard paths not yet marked consumed.

    Versions outside the inclusive ``[min_policy_version, max_policy_version]``
    window are consumed without training. ``max_policy_version=None`` leaves the
    upper bound open for backward compatibility.
    ``stable_secs`` skips shards written within the last N seconds (avoid reading mid-write on
    filesystems without atomic rename guarantees; default 0 trusts the atomic rename).

    FIX C2a (freshest-first): with the default ``newest_first=False`` shards are yielded
    OLDEST-first (worker dir name asc, then ``shard_*`` index asc — the original behavior the
    self-test relies on). With ``newest_first=True`` shards are yielded by ``policy_version`` DESC
    then mtime DESC, so when actors outrun the learner it trains on the FRESHEST data instead of
    the stalest. Reading the version requires deserializing the envelope; that deserialization is
    reused (no second read) and is also surfaced when ``with_envelope=True``.

    FIX 5 (efficiency): when ``with_envelope=True`` this yields ``(path, envelope)`` tuples,
    reusing the envelope already deserialized for the staleness/ordering check so the learner does
    NOT deserialize each shard a second time. With the default ``with_envelope=False`` it yields
    bare ``Path`` objects (back-compatible; the ``__main__`` self-test relies on this). When
    ``with_envelope`` (or ``newest_first``) is set, the envelope is deserialized even when
    ``min_policy_version <= 0`` so callers always get a payload / a version to sort on.

    FIX 6 (Modal volume visibility): an optional ``volume_reload_fn`` is invoked once up front so
    a learner running as a Modal function can call ``volume.reload()`` and see actor writes before
    listing shards. It is a no-op when ``None`` (plain-process runs).
    """
    if volume_reload_fn is not None:
        try:
            volume_reload_fn()
        except Exception:  # reload is best-effort; never crash the learner on it
            pass
    if expected_run_manifest_sha256 is not None:
        expected_run_manifest_sha256 = _require_manifest_sha256(
            expected_run_manifest_sha256, where="expected run_manifest_sha256"
        )
    base = trajectories_dir(root)
    if not base.exists():
        return
    cdir = consumed_dir(root)
    cdir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    # When we need the envelope at all (staleness drop, payload, or version-based ordering) we
    # must deserialize. newest_first additionally needs the version+mtime up front to sort.
    need_envelope = (
        min_policy_version > 0
        or max_policy_version is not None
        or with_envelope
        or newest_first
        or expected_run_manifest_sha256 is not None
    )

    def version_is_rejected(envelope: Any) -> bool:
        policy_version = int(envelope.get("policy_version", 0))
        return policy_version < min_policy_version or (
            max_policy_version is not None
            and policy_version > int(max_policy_version)
        )

    if not newest_first:
        # ---- streaming oldest-first (original behavior; cheap when no envelope needed) ----
        count = 0
        for worker in sorted(base.iterdir()):
            if not worker.is_dir():
                continue
            for shard in sorted(worker.glob("shard_*.pkl")):
                if _consumed_marker(root, shard).exists():
                    continue
                if stable_secs > 0.0 and (now - shard.stat().st_mtime) < stable_secs:
                    continue
                envelope = None
                if need_envelope:
                    try:
                        envelope = read_trajectory_shard(
                            shard,
                            expected_run_manifest_sha256=expected_run_manifest_sha256,
                        )
                    except (pickle.UnpicklingError, EOFError, OSError):
                        continue
                    if version_is_rejected(envelope):
                        mark_consumed(root, shard)
                        continue
                yield (shard, envelope) if with_envelope else shard
                count += 1
                if max_shards is not None and count >= max_shards:
                    return
        return

    # ---- newest-first: collect candidates, sort by policy_version DESC then mtime DESC ----
    candidates: list[tuple[int, float, Path, Any]] = []
    for worker in sorted(base.iterdir()):
        if not worker.is_dir():
            continue
        for shard in sorted(worker.glob("shard_*.pkl")):
            if _consumed_marker(root, shard).exists():
                continue
            try:
                mtime = shard.stat().st_mtime
            except OSError:
                continue
            if stable_secs > 0.0 and (now - mtime) < stable_secs:
                continue
            try:
                envelope = read_trajectory_shard(
                    shard,
                    expected_run_manifest_sha256=expected_run_manifest_sha256,
                )
            except (pickle.UnpicklingError, EOFError, OSError):
                continue
            pv = int(envelope.get("policy_version", 0))
            if version_is_rejected(envelope):
                mark_consumed(root, shard)
                continue
            candidates.append((pv, mtime, shard, envelope))
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)  # version DESC, then mtime DESC
    count = 0
    for _, _, shard, envelope in candidates:
        yield (shard, envelope) if with_envelope else shard
        count += 1
        if max_shards is not None and count >= max_shards:
            return


def sweep_drop_stale(root: str | os.PathLike, *, min_policy_version: int) -> int:
    """Drop EVERY unconsumed shard whose ``policy_version < min_policy_version`` (FIX C2b).

    Unlike ``iter_unconsumed_shards`` (which only inspects up to ``max_shards`` per call, so older
    stale shards beyond that window accumulate forever), this scans ALL unconsumed shards and
    ``mark_consumed``s the stale ones. This is the mechanism that actually bounds the trajectory
    backlog when 600 actors outrun 1 learner. Returns the number of shards dropped.
    """
    base = trajectories_dir(root)
    if not base.exists():
        return 0
    consumed_dir(root).mkdir(parents=True, exist_ok=True)
    dropped = 0
    for worker in sorted(base.iterdir()):
        if not worker.is_dir():
            continue
        for shard in sorted(worker.glob("shard_*.pkl")):
            if _consumed_marker(root, shard).exists():
                continue
            try:
                envelope = read_trajectory_shard(shard)
            except (pickle.UnpicklingError, EOFError, OSError):
                continue
            if int(envelope.get("policy_version", 0)) < min_policy_version:
                mark_consumed(root, shard)
                dropped += 1
    return dropped


def sweep_drop_outside_policy_window(
    root: str | os.PathLike,
    *,
    min_policy_version: int,
    max_policy_version: int,
    expected_run_manifest_sha256: str | None = None,
) -> int:
    """Consume every shard outside an inclusive accepted policy-version window."""
    base = trajectories_dir(root)
    if not base.exists():
        return 0
    consumed_dir(root).mkdir(parents=True, exist_ok=True)
    dropped = 0
    for worker in sorted(base.iterdir()):
        if not worker.is_dir():
            continue
        for shard in sorted(worker.glob("shard_*.pkl")):
            if _consumed_marker(root, shard).exists():
                continue
            try:
                envelope = read_trajectory_shard(
                    shard,
                    expected_run_manifest_sha256=expected_run_manifest_sha256,
                )
            except (pickle.UnpicklingError, EOFError, OSError):
                continue
            version = int(envelope.get("policy_version", 0))
            if not int(min_policy_version) <= version <= int(max_policy_version):
                mark_consumed(root, shard)
                dropped += 1
    return dropped


def prune_consumed_markers(root: str | os.PathLike, *, older_than_secs: float) -> int:
    """Delete consumed-marker files older than ``older_than_secs`` (FIX C4-support).

    Each ingested shard leaves one empty marker inode in ``consumed/`` forever; over a multi-day
    run with hundreds of thousands of shards these dominate the inode count. The learner calls this
    periodically to prune markers old enough that their shards can no longer be re-listed. Returns
    the number of markers pruned.
    """
    cdir = consumed_dir(root)
    if not cdir.exists():
        return 0
    cutoff = time.time() - older_than_secs
    pruned = 0
    for marker in cdir.iterdir():
        if not marker.is_file():
            continue
        try:
            if marker.stat().st_mtime < cutoff:
                marker.unlink()
                pruned += 1
        except OSError:
            continue
    return pruned


def mark_consumed(root: str | os.PathLike, shard_path: str | os.PathLike) -> None:
    """Mark a shard ingested: write its consumed-marker, then remove the shard (FIX M1).

    Single-consumer invariant: exactly ONE learner consumes shards, so there is no race between
    two markers. We always write the marker BEFORE removing the shard, so a crash between the two
    leaves the shard discoverable-but-marked (it will be skipped, never double-ingested) rather
    than lost. Removing an already-gone shard is ignored (idempotent / robust to retries).
    """
    # This durable receipt is the actor's authoritative resume guard. It must
    # succeed before either the bounded marker or the queue payload is removed.
    mark_trajectory_complete(root, shard_path)
    marker = _consumed_marker(root, Path(shard_path))
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        marker.touch()  # record ingestion first
    except OSError:
        pass
    try:
        os.remove(shard_path)  # reclaim space; ignore if already gone
    except FileNotFoundError:
        pass
    except OSError:
        pass


if __name__ == "__main__":  # lightweight self-test (no torch/env needed)
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = run_root(d, "selftest")
        ensure_run_dirs(root)
        assert read_version(root) is None
        v1 = publish_weights(root, lambda p: Path(p).write_text("w1"), step=10)
        assert v1.version == 1 and Path(v1.path).read_text() == "w1"
        v2 = publish_weights(root, lambda p: Path(p).write_text("w2"), step=20)
        assert v2.version == 2 and read_version(root).version == 2

        # FIX H1: read_version returns the VERSION-STAMPED weights path, and its bytes match the
        # version (current.pt could be N+1 mid-publish; the versioned file never mis-stamps).
        rv = read_version(root)
        assert rv.path == str(versioned_weights_path(root, 2)), rv.path
        assert Path(rv.path).name == "weights_v2.pt"
        assert Path(rv.path).read_text() == "w2"
        # current_weights_path stays back-compatible (latest weights, copy of weights_v2.pt).
        assert current_weights_path(root).read_text() == "w2"
        # FIX H1 GC: keep only the newest KEEP_VERSIONED_WEIGHTS versioned files.
        for s in range(30, 30 + KEEP_VERSIONED_WEIGHTS + 2):
            publish_weights(root, lambda p, s=s: Path(p).write_text(f"w{s}"), step=s)
        versioned = sorted(policy_dir(root).glob(VERSIONED_WEIGHTS_GLOB))
        assert len(versioned) == KEEP_VERSIONED_WEIGHTS, versioned
        latest = read_version(root)
        assert Path(latest.path).exists() and Path(latest.path).read_text() == f"w{30 + KEEP_VERSIONED_WEIGHTS + 1}"

        p = write_trajectory_shard(root, "worker-A", 0, [{"fake": "traj"}], policy_version=2)
        got = list(iter_unconsumed_shards(root))
        assert got == [p], got
        env = read_trajectory_shard(p)
        assert env["trajectories"] == [{"fake": "traj"}] and env["policy_version"] == 2
        mark_consumed(root, p)
        assert list(iter_unconsumed_shards(root)) == []
        # M1: marking an already-removed shard is a robust no-op (idempotent).
        mark_consumed(root, p)

        # staleness drop (default oldest-first path)
        p2 = write_trajectory_shard(root, "worker-A", 1, [{"x": 1}], policy_version=1)
        assert list(iter_unconsumed_shards(root, min_policy_version=2)) == []

        # FIX C2a: newest_first yields by policy_version DESC then mtime DESC.
        import tempfile as _t  # noqa: F401  (kept local; tempfile already imported)
        a = write_trajectory_shard(root, "worker-A", 2, [{"v": 5}], policy_version=5)
        b = write_trajectory_shard(root, "worker-B", 0, [{"v": 7}], policy_version=7)
        c = write_trajectory_shard(root, "worker-C", 0, [{"v": 6}], policy_version=6)
        oldest_first = list(iter_unconsumed_shards(root))  # default: worker dir asc
        assert oldest_first == [a, b, c], oldest_first
        newest = list(iter_unconsumed_shards(root, newest_first=True))
        assert newest == [b, c, a], newest  # versions 7, 6, 5
        # newest_first honours max_shards (freshest N) and with_envelope.
        fresh1 = list(iter_unconsumed_shards(root, newest_first=True, max_shards=1, with_envelope=True))
        assert len(fresh1) == 1 and fresh1[0][0] == b and fresh1[0][1]["policy_version"] == 7

        # FIX C2b: sweep_drop_stale drops ALL stale shards, not just the first max_shards window.
        for i in range(3, 13):
            write_trajectory_shard(root, "worker-D", i, [{"old": i}], policy_version=4)
        dropped = sweep_drop_stale(root, min_policy_version=6)
        # worker-D (v4, x10) + a (v5) are below v6 -> 11 dropped; b/c (v7/v6) survive.
        assert dropped == 11, dropped
        survivors = sorted(s.name for s in [b, c])
        remaining = sorted(s.name for s in iter_unconsumed_shards(root))
        assert remaining == survivors, remaining

        # FIX C4-support: prune_consumed_markers deletes old markers, keeps fresh ones.
        markers = sorted(consumed_dir(root).iterdir())
        assert markers, "expected consumed markers to exist"
        old_marker = markers[0]
        old_t = time.time() - 10_000
        os.utime(old_marker, (old_t, old_t))
        pruned = prune_consumed_markers(root, older_than_secs=3600)
        assert pruned == 1, pruned
        assert not old_marker.exists()
        assert prune_consumed_markers(root, older_than_secs=3600) == 0  # rest are fresh

        print("ppo_distributed self-test OK")
