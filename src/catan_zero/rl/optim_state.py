"""FSDP-safe optimizer-state persistence for resumable training (CAT-128 / patch #8).

Model checkpoints do NOT carry the optimizer's Adam moment estimates, so a stop or
crash restarts Adam from zero -- losing the in-progress epoch's momentum and risking
catastrophic forgetting on resume (see train_bc's --lr-warmup-steps guard). This module
persists optimizer state as a sidecar file next to the model checkpoint
(``<checkpoint>.optimizer.pt``) so a resume continues cleanly.

Correctness across the whole training stack (this is the bug in the naive rank0-only
patch #8, which is wrong under FSDP):

* single-GPU / DDP -- the optimizer state is REPLICATED across ranks, so rank 0 saves
  and loads a plain ``optimizer.state_dict()``.
* FSDP (FULL_SHARD) -- the optimizer state is SHARDED across ranks, so a rank0-only
  ``state_dict()`` captures only rank 0's shard (wrong) and can deadlock. The correct
  path is the COLLECTIVE ``FSDP.optim_state_dict`` (every rank participates, the full
  unsharded state is gathered to rank 0, CPU-offloaded) to save, and
  ``FSDP.optim_state_dict_to_load`` to re-shard on load. Every rank MUST call
  ``save_optimizer_state`` / ``load_optimizer_state``; only rank 0 touches the file.
  This mirrors the FULL_STATE_DICT model-weight gather in train_bc ``_save_policy``
  (C1); ``is_fsdp`` here is the single FSDP-detection path both call sites share.

The optimizer pickle alone is not a resumable checkpoint: Adam moments without the
matching LR step caused warmup, max-step dose, and reports to restart at zero. A
versioned ``<checkpoint>.training-progress.json`` commit marker therefore binds hashes
of the model and optimizer files to the exact recipe/schedule identity and records the
optimizer step, completed epochs, cumulative value dose, NumPy sampler state, and
per-rank CPU/CUDA torch RNG (dropout) state. ``train_bc``
only restores moments after that marker validates. Any partial/mixed/legacy set fails
closed; an operator can deliberately choose fresh Adam with ``--no-resume-optimizer``.

The low-level ``load_optimizer_state`` remains fail-soft and returns ``False`` on a
pickle/API mismatch. The trainer converts that into a fail-closed resume error after a
progress marker has committed continuity; callers that use this utility independently
retain its historical behavior.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from typing import Any

_OPTIM_SIDECAR_SUFFIX = ".optimizer.pt"
_PROGRESS_SIDECAR_SUFFIX = ".training-progress.json"
TRAINING_PROGRESS_SCHEMA = "train-bc-progress-v1"


class TrainingProgressError(RuntimeError):
    """The checkpoint set cannot be proven to describe one training trajectory."""


def optimizer_sidecar_path(checkpoint_path: str | os.PathLike) -> Path:
    """``<checkpoint>.optimizer.pt`` -- the optimizer sidecar next to a model checkpoint.
    Matches patch #8's convention so any existing sidecars interoperate."""
    p = Path(checkpoint_path)
    return p.with_name(p.name + _OPTIM_SIDECAR_SUFFIX)


def training_progress_sidecar_path(checkpoint_path: str | os.PathLike) -> Path:
    """Commit marker for a model + optimizer checkpoint pair."""
    p = Path(checkpoint_path)
    return p.with_name(p.name + _PROGRESS_SIDECAR_SUFFIX)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _canonical_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_training_progress(
    checkpoint_path: str | os.PathLike,
    *,
    optimizer_step: int,
    completed_epochs: int,
    recipe_identity: dict[str, Any],
    rng_state: dict[str, Any],
    rank_numpy_rng_states: list[dict[str, Any]],
    symmetry_rng_state: dict[str, Any] | None,
    rank_torch_rng_states: list[dict[str, Any]],
    scalar_training_weight_sum: float,
    categorical_training_weight_sum: float,
    policy_kl_controller_state: dict[str, Any] | None = None,
    ddp: dict | None,
) -> Path | None:
    """Atomically commit progress after model and optimizer sidecars are durable.

    The JSON is deliberately written last. Its byte hashes bind the two independently
    atomic torch files into one checkpoint set; a crash or overwrite between any of the
    three writes is detected on resume instead of mixing model weights, Adam moments,
    and an unrelated LR-schedule position.
    """
    if _rank(ddp) != 0:
        return None
    checkpoint = Path(checkpoint_path)
    optimizer_path = optimizer_sidecar_path(checkpoint)
    if not checkpoint.is_file() or not optimizer_path.is_file():
        raise TrainingProgressError(
            "cannot commit training progress without both model and optimizer files"
        )
    payload: dict[str, Any] = {
        "schema_version": TRAINING_PROGRESS_SCHEMA,
        "status": "complete",
        "checkpoint": {
            "path": checkpoint.name,
            "sha256": _file_sha256(checkpoint),
        },
        "optimizer": {
            "path": optimizer_path.name,
            "sha256": _file_sha256(optimizer_path),
        },
        "optimizer_step": int(optimizer_step),
        "completed_epochs": int(completed_epochs),
        "recipe_identity": recipe_identity,
        "recipe_identity_sha256": _canonical_sha256(recipe_identity),
        "rng_state": rng_state,
        "rank_numpy_rng_states": rank_numpy_rng_states,
        "symmetry_rng_state": symmetry_rng_state,
        "rank_torch_rng_states": rank_torch_rng_states,
        "scalar_training_weight_sum": float(scalar_training_weight_sum),
        "categorical_training_weight_sum": float(categorical_training_weight_sum),
    }
    if policy_kl_controller_state is not None:
        if not isinstance(policy_kl_controller_state, dict):
            raise TrainingProgressError(
                "policy-KL controller progress state must be a JSON object"
            )
        payload["policy_kl_controller_state"] = policy_kl_controller_state
    payload["progress_sha256"] = _canonical_sha256(payload)
    output = training_progress_sidecar_path(checkpoint)
    _atomic_json_save(payload, output)
    _log(f"committed training progress -> {output}", ddp)
    return output


def load_training_progress(
    checkpoint_path: str | os.PathLike,
    *,
    expected_recipe_identity: dict[str, Any],
) -> dict[str, Any]:
    """Validate and return a committed checkpoint set, otherwise fail closed."""
    checkpoint = Path(checkpoint_path)
    optimizer_path = optimizer_sidecar_path(checkpoint)
    progress_path = training_progress_sidecar_path(checkpoint)
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingProgressError(
            f"cannot read training progress {progress_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise TrainingProgressError("training progress is not a JSON object")
    stated = payload.get("progress_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "progress_sha256"}
    if (
        payload.get("schema_version") != TRAINING_PROGRESS_SCHEMA
        or payload.get("status") != "complete"
        or stated != _canonical_sha256(unhashed)
    ):
        raise TrainingProgressError("training progress schema/status/digest mismatch")
    if (
        payload.get("recipe_identity") != expected_recipe_identity
        or payload.get("recipe_identity_sha256")
        != _canonical_sha256(expected_recipe_identity)
    ):
        raise TrainingProgressError("training recipe/schedule identity mismatch")
    checkpoint_record = payload.get("checkpoint", {})
    optimizer_record = payload.get("optimizer", {})
    if (
        checkpoint_record.get("path") != checkpoint.name
        or optimizer_record.get("path") != optimizer_path.name
        or not checkpoint.is_file()
        or not optimizer_path.is_file()
        or checkpoint_record.get("sha256") != _file_sha256(checkpoint)
        or optimizer_record.get("sha256") != _file_sha256(optimizer_path)
    ):
        raise TrainingProgressError("model/optimizer checkpoint binding mismatch")
    for field in ("optimizer_step", "completed_epochs"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TrainingProgressError(f"invalid training progress field {field}")
    if not isinstance(payload.get("rng_state"), dict):
        raise TrainingProgressError("training progress lacks numpy RNG state")
    if "policy_kl_controller_state" in payload and not isinstance(
        payload["policy_kl_controller_state"], dict
    ):
        raise TrainingProgressError(
            "training progress has malformed policy-KL controller state"
        )
    expected_world_size = expected_recipe_identity.get("world_size")
    rank_numpy_rng = payload.get("rank_numpy_rng_states")
    if rank_numpy_rng is not None and (
        not isinstance(rank_numpy_rng, list)
        or len(rank_numpy_rng) != expected_world_size
        or any(not isinstance(row, dict) for row in rank_numpy_rng)
    ):
        raise TrainingProgressError(
            "training progress has invalid per-rank numpy RNG state"
        )
    rank_rng = payload.get("rank_torch_rng_states")
    if (
        not isinstance(rank_rng, list)
        or isinstance(expected_world_size, bool)
        or not isinstance(expected_world_size, int)
        or len(rank_rng) != expected_world_size
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("cpu"), list)
            or (row.get("cuda") is not None and not isinstance(row.get("cuda"), list))
            for row in rank_rng
        )
    ):
        raise TrainingProgressError("training progress lacks per-rank torch RNG state")
    return payload


def is_fsdp(model: Any) -> bool:
    """True iff ``model`` is an FSDP-wrapped module. Single FSDP-detection path shared
    by the optimizer-state save/restore here and train_bc ``_save_policy`` (which
    delegates to this). Import-guarded so it is safe on torch builds without FSDP."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except Exception:
        return False
    return isinstance(model, FSDP)


def _rank(ddp: dict | None) -> int:
    return int(ddp["rank"]) if ddp else 0


def _log(message: str, ddp: dict | None) -> None:
    """Rank-0-only structured line, matching train_bc's progress-JSON convention."""
    if _rank(ddp) == 0:
        print(json.dumps({"progress": "optimizer_state", "message": message}, sort_keys=True), flush=True)


def _atomic_torch_save(obj: Any, path: Path) -> None:
    """torch.save to a unique temp then ``os.replace`` -- same atomic temp+rename
    discipline as train_bc ``_save_policy`` (torch pickle, not JSON)."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(obj, tmp)
        if not tmp.exists() or tmp.stat().st_size <= 0:
            raise RuntimeError(f"optimizer sidecar temp file was not written: {tmp}")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_optimizer_state(
    checkpoint_path: str | os.PathLike, model: Any, optimizer: Any, ddp: dict | None
) -> Path | None:
    """Write the optimizer state to ``<checkpoint>.optimizer.pt``.

    MUST be called on EVERY rank -- the FSDP gather is a collective. Returns the sidecar
    path on rank 0, ``None`` on other ranks (which only participate in the collective).
    FSDP: collective full-state gather (rank0-only, CPU-offloaded). DDP/single: rank 0
    writes a plain ``optimizer.state_dict()``. Wrapped fail-soft: a save failure logs and
    returns None rather than crashing the training run (the model checkpoint is already
    written by the caller; a missing sidecar just means the next resume uses a fresh
    optimizer)."""
    sidecar = optimizer_sidecar_path(checkpoint_path)
    is_rank0 = _rank(ddp) == 0
    try:
        if is_fsdp(model):
            from torch.distributed.fsdp import (
                FullOptimStateDictConfig,
                FullStateDictConfig,
                FullyShardedDataParallel as FSDP,
                StateDictType,
            )

            model_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
            # Collective: every rank enters; the full optim state is gathered to rank 0.
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, model_cfg, optim_cfg):
                full_osd = FSDP.optim_state_dict(model, optimizer)
            if not is_rank0:
                return None
            _atomic_torch_save({"optimizer": full_osd, "format": "fsdp_full"}, sidecar)
            _log(f"saved FSDP optimizer state -> {sidecar}", ddp)
            return sidecar
        # DDP / single-GPU: replicated optimizer state; only rank 0 writes.
        if not is_rank0:
            return None
        _atomic_torch_save({"optimizer": optimizer.state_dict(), "format": "plain"}, sidecar)
        _log(f"saved optimizer state -> {sidecar}", ddp)
        return sidecar
    except Exception as error:  # never let a sidecar save crash the training run
        _log(
            f"WARNING: could not save optimizer state to {sidecar} "
            f"({type(error).__name__}: {error}); resume will use a fresh optimizer",
            ddp,
        )
        return None


def load_optimizer_state(
    checkpoint_path: str | os.PathLike, model: Any, optimizer: Any, ddp: dict | None
) -> bool:
    """Restore optimizer state from ``<checkpoint>.optimizer.pt`` in place on ``optimizer``.

    Returns True iff state was restored. FAIL-SAFE: a missing sidecar or ANY error
    (arch/param mismatch from grow-from-checkpoint, torch-version API drift, corrupt
    file) is logged and returns False -- the caller trains with the fresh optimizer it
    already built. NEVER raises. MUST be called on EVERY rank under FSDP
    (``optim_state_dict_to_load`` is collective).

    MULTI-NODE CAVEATS (fine for single-node runs like c1; audit-fixer review): every
    rank gates on ``sidecar.exists()``, so the sidecar must live on storage visible to
    ALL nodes -- on a non-shared filesystem some nodes could disagree on existence and
    the collective would run on a subset and hang. Likewise the per-rank fail-safe means
    a load that raises on only SOME ranks leaves ranks divergent (some fresh, some
    restored); a future multi-node hardening should all-reduce the load-success flag so
    every rank agrees before the collective. Single-node (shared local FS) is unaffected."""
    import torch

    sidecar = optimizer_sidecar_path(checkpoint_path)
    if not sidecar.exists():
        _log(f"no optimizer sidecar at {sidecar}; using fresh optimizer state", ddp)
        return False
    try:
        blob = torch.load(sidecar, map_location="cpu", weights_only=False)
        full_osd = blob["optimizer"] if isinstance(blob, dict) and "optimizer" in blob else blob
        if is_fsdp(model):
            from torch.distributed.fsdp import (
                FullOptimStateDictConfig,
                FullStateDictConfig,
                FullyShardedDataParallel as FSDP,
                StateDictType,
            )

            # Mirror the SAVE context so the reshard reads the FULL osd on every rank
            # (rank0_only=False -- every rank loaded the full sidecar) instead of relying
            # on FSDP's ambient default state-dict type being FULL_STATE_DICT. Scoped
            # `with` auto-reverts; matches _save_policy's model-gather discipline
            # (belt-and-suspenders, audit-fixer FSDP review).
            model_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, model_cfg, optim_cfg):
                sharded = FSDP.optim_state_dict_to_load(
                    model=model, optim=optimizer, optim_state_dict=full_osd
                )
            optimizer.load_state_dict(sharded)
        else:
            optimizer.load_state_dict(full_osd)
        _log(f"restored optimizer state from {sidecar}", ddp)
        return True
    except Exception as error:
        _log(
            f"could not restore optimizer state from {sidecar} "
            f"({type(error).__name__}: {error}); continuing with fresh optimizer state",
            ddp,
        )
        return False
