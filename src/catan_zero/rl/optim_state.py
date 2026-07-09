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

``load_optimizer_state`` is FAIL-SAFE: a missing sidecar or ANY mismatch (arch change
from grow-from-checkpoint, param-group/shape drift, torch-version API drift) is logged
and returns ``False`` -- the caller then trains with a fresh optimizer. It NEVER raises,
so resume degrades gracefully to today's behaviour rather than crashing a run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_OPTIM_SIDECAR_SUFFIX = ".optimizer.pt"


def optimizer_sidecar_path(checkpoint_path: str | os.PathLike) -> Path:
    """``<checkpoint>.optimizer.pt`` -- the optimizer sidecar next to a model checkpoint.
    Matches patch #8's convention so any existing sidecars interoperate."""
    p = Path(checkpoint_path)
    return p.with_name(p.name + _OPTIM_SIDECAR_SUFFIX)


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
