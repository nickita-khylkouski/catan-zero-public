"""2-rank GPU worker that exercises the REAL FSDP collective optimizer-state gather in
optim_state.py (CAT-128). Requires >=2 CUDA devices (our torch 2.11/2.13 both require a
non-CPU accelerator for FSDP -- CPU/gloo FSDP is NOT supported, so this cannot run on
pure CPU). Launched by test_optim_state_cat128.py via
``torchrun --nproc_per_node=2 --tee`` when the GPU verify is opted in. Prints
``FSDP_OPTIM_OK`` on rank 0 and exits 0 iff the collective save+restore round-trips.
"""

import os
import sys
import tempfile

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def _build_fsdp_adam(local_rank: int):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 16), torch.nn.GELU(), torch.nn.Linear(16, 8)
    ).to(f"cuda:{local_rank}")
    # use_orig_params=True matches the C1 wrap so param names round-trip and the optim
    # state maps cleanly through FSDP.optim_state_dict / optim_state_dict_to_load.
    model = FSDP(model, device_id=local_rank, use_orig_params=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(4, 16, device=f"cuda:{local_rank}")).sum().backward()
        opt.step()
    return model, opt


def main() -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    ddp = {"enabled": world > 1, "world_size": world, "rank": rank, "local_rank": local_rank}

    from catan_zero.rl.optim_state import (
        load_optimizer_state,
        optimizer_sidecar_path,
        save_optimizer_state,
    )

    # Shared temp path across ranks (rank 0 creates it, broadcast to all).
    d = tempfile.mkdtemp() if rank == 0 else ""
    obj = [d]
    dist.broadcast_object_list(obj, src=0)
    ckpt = os.path.join(obj[0], "ckpt.pt")

    model, opt = _build_fsdp_adam(local_rank)

    # SAVE -- collective (all ranks enter FSDP.optim_state_dict); rank 0 writes sidecar.
    save_optimizer_state(ckpt, model, opt, ddp)
    dist.barrier()
    if rank == 0 and not optimizer_sidecar_path(ckpt).exists():
        print(f"FSDP_OPTIM_FAIL: sidecar not written at {optimizer_sidecar_path(ckpt)}", flush=True)
        return 1

    # RESTORE into a FRESH FSDP model + zero-state optimizer -- collective load.
    model2, _ = _build_fsdp_adam(local_rank)
    fresh_opt = torch.optim.Adam(model2.parameters(), lr=1e-3)
    restored = load_optimizer_state(ckpt, model2, fresh_opt, ddp)
    dist.barrier()
    if not restored:
        print(f"FSDP_OPTIM_FAIL(rank{rank}): load_optimizer_state returned False", flush=True)
        return 1
    if not any(len(s) for s in fresh_opt.state.values()):
        print(f"FSDP_OPTIM_FAIL(rank{rank}): restored optimizer state is empty", flush=True)
        return 1

    dist.barrier()
    if rank == 0:
        print("FSDP_OPTIM_OK", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
