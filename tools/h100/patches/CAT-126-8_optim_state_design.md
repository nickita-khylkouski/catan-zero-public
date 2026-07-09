# #8 Optimizer-State Persistence — Shared Design (CAT-126 + CAT-128)

**Status:** DESIGN LOCKED, no code until `v1.0-freeze` tag. **Date:** 2026-07-09.
**Why shared:** CAT-126 (in-code adoption of finding #8 as a default) and CAT-128 (training resumability) both need this. ONE implementation, not two.

## Problem with the drafted patch (apply_04)
`apply_04_optimizer_state.py` saves `optimizer.state_dict()` on **rank0 only**. That is correct for DDP/single-GPU (optimizer state is replicated) but **wrong under FSDP** — each rank holds only its *shard* of the optimizer state, so a rank0-only `state_dict()` is incomplete, and a naive gather would deadlock. The training target IS the FSDP-capable stack, so the util must branch.

## Placement
New module `src/catan_zero/rl/optim_state.py` (single-responsibility; both CAT-126 adoption and CAT-128 training import it). Not folded into `config_serialization.py` (that is for config dicts).

## Interface
```python
def optimizer_sidecar_path(checkpoint_path: str | Path) -> Path        # "<checkpoint>.optimizer.pt"
def save_optimizer_state(checkpoint_path, model, optimizer, ddp) -> Path | None
def load_optimizer_state(checkpoint_path, model, optimizer, ddp) -> bool
```

## Locked correctness invariants
1. **FSDP save is COLLECTIVE.** `save_optimizer_state` is called on **all ranks**; the FSDP branch runs `FSDP.optim_state_dict(model, optimizer)` on every rank (gather → full dict materialized on rank0), then **rank0 writes**, others return `None`. Never rank0-only-gather (that was the patch bug + a deadlock risk).
2. **Load is symmetric/collective.** FSDP: all ranks read the full sidecar, `FSDP.optim_state_dict_to_load(model, optimizer, full_osd)` → `optimizer.load_state_dict(...)`. DDP/single: `optimizer.load_state_dict(...)`.
3. **Fail-safe load.** Any mismatch (grow-from-checkpoint, edge-head arch, corrupt/absent sidecar) → log + return `False` (fresh Adam). `load_optimizer_state` never raises to the training loop.
4. **Atomic write.** `torch.save` to a temp path + `os.replace` into place (torch pickle, so the temp+rename discipline, not the JSON `atomic_io.write_json_atomic`).
5. **Sidecar convention** `<checkpoint>.optimizer.pt` — matches apply_04 so any pre-existing sidecars interoperate.
6. **One FSDP-detection path.** Reuse the exact `is_fsdp` signal + full-state-dict gather already built for MODEL weights in `_save_policy` (fsdp-builder's C1 tree) so model-weights and optimizer-state save share one detection path. *(Open: awaiting fsdp-builder's file/line pointer.)*

## Ownership
- **audit-fixer:** authors `optim_state.py` (incl. FSDP branch) + the DDP/single-GPU unit test + sidecar-path test. Closes CAT-126 #8 as an in-code default (delete the rank0-only apply_04 path).
- **fsdp-builder:** integrates it into `train_bc.py` (checkpoint-write callsite + `--init-checkpoint`/resume path incl. global_step/epoch) for CAT-128; owns the FSDP multi-rank verification (2-GPU harness).

## Classification
**BEHAVIORAL** — changes the resume trajectory (eliminates the fresh-Adam transient). Strictly an improvement; no change to a from-scratch run.

## Test plan
- Unit (mine, CPU/single): save→load round-trip restores Adam moments/variance exactly; sidecar path helper; fail-safe on arch mismatch returns False + fresh optimizer; atomic write leaves no `.tmp`.
- FSDP (fsdp-builder, 2-GPU): save under FSDP → full osd on rank0 → reload into a fresh FSDP optimizer → step-equivalence vs no-preemption baseline.

## Implementation reference (locked with fsdp-builder)
Mirror the existing model-weight path in `tools/train_bc.py` (branch `c1-multigpu-on-run6` @ 17159e1; identical on `c1-multigpu` @ 646eb5d):
- **`_is_fsdp(model)`** — the ONE detection path to reuse (takes the raw wrapped model = `policy.model`):
  ```python
  try:
      from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
      return isinstance(model, FSDP)
  except Exception:
      return False
  ```
- **Model-weight gather to parallel (`_save_policy`):** `with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)): model_state = model.state_dict()` — collective, ALL ranks call; `if not is_rank0: return` AFTER the collective; rank0 atomic temp+os.replace.
- **Optim-specific API (`optim_state.py` uses):** gather `FSDP.optim_state_dict(model, optimizer)` → full osd on rank0; reshard on load `FSDP.optim_state_dict_to_load(model, optimizer, full_osd)` → `optimizer.load_state_dict(...)`. For rank0_only/offload use the **3-arg** context (extra `FullOptimStateDictConfig` vs the model-only 2-arg form):
  ```python
  with FSDP.state_dict_type(
      model, StateDictType.FULL_STATE_DICT,
      FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
      FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
  ):
      full_osd = FSDP.optim_state_dict(model, optimizer)
  ```
  **All ranks must enter the gather; rank0 writes.** Integration (fsdp-builder) calls `save_optimizer_state` unconditionally on all ranks; the util rank-guards the write (rank0-only call would deadlock the collective).

## Expected behavior (so verification isn't surprised)
champion_v0 and any grow-from arm have **no matching sidecar** ⇒ `load_optimizer_state` returns `False` ⇒ fresh Adam on the FIRST fine-tune. Correct/expected. The resumability win is specifically on **resuming a run started with this code** (its epoch checkpoint carries a sidecar).

## Hold
No code until the `v1.0-freeze` tag; implement on the tagged tree.
