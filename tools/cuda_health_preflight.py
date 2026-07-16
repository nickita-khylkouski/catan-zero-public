#!/usr/bin/env python3
"""Bounded CUDA/NCCL readiness probe for canonical fleet training launches.

The caller must impose a process-level timeout.  CUDA context creation can hang
inside the driver, where a Python timer or distributed timeout cannot reliably
interrupt it.  ``tools/fleet/fleet_launch.sh`` therefore wraps this program in
GNU ``timeout`` before any training run directory is created.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import os
from typing import Any, Mapping, Sequence


class PreflightError(RuntimeError):
    """The visible CUDA topology or collective health is not launch-safe."""


def check_allocations(
    torch: Any,
    *,
    expected_devices: int,
    device_indices: Sequence[int] | None = None,
) -> None:
    """Create and synchronize a tiny tensor on every requested logical GPU."""
    if expected_devices < 1:
        raise PreflightError("expected_devices must be positive")
    if not torch.cuda.is_available():
        raise PreflightError("torch.cuda.is_available() is false")
    visible_devices = int(torch.cuda.device_count())
    if visible_devices != expected_devices:
        raise PreflightError(
            f"expected {expected_devices} visible CUDA device(s), found {visible_devices}"
        )

    indices = (
        tuple(range(visible_devices))
        if device_indices is None
        else tuple(device_indices)
    )
    for index in indices:
        if index < 0 or index >= visible_devices:
            raise PreflightError(
                f"logical CUDA device {index} is outside visible range 0..{visible_devices - 1}"
            )
        torch.cuda.set_device(index)
        value = torch.ones(1, dtype=torch.float32, device=f"cuda:{index}")
        value.add_(1.0)
        torch.cuda.synchronize(index)
        if float(value.item()) != 2.0:
            raise PreflightError(
                f"CUDA arithmetic check failed on logical device {index}"
            )


def check_nccl_collective(
    torch: Any,
    *,
    expected_devices: int,
    environ: Mapping[str, str] = os.environ,
    process_group_timeout_seconds: float = 20.0,
) -> None:
    """Verify one CUDA allocation per rank and a cross-rank NCCL all-reduce."""
    try:
        local_rank = int(environ["LOCAL_RANK"])
        rank = int(environ["RANK"])
        world_size = int(environ["WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise PreflightError(
            "collective mode requires valid torchrun rank variables"
        ) from error
    if world_size != expected_devices:
        raise PreflightError(
            f"torchrun world size {world_size} does not match {expected_devices} visible devices"
        )

    check_allocations(
        torch,
        expected_devices=expected_devices,
        device_indices=(local_rank,),
    )
    initialized = False
    try:
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=process_group_timeout_seconds),
        )
        initialized = True
        value = torch.tensor(
            [float(rank + 1)], dtype=torch.float32, device=f"cuda:{local_rank}"
        )
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        torch.cuda.synchronize(local_rank)
        expected_sum = float(world_size * (world_size + 1) // 2)
        if float(value.item()) != expected_sum:
            raise PreflightError(
                f"NCCL all-reduce returned {value.item()}, expected {expected_sum}"
            )
    finally:
        if initialized:
            torch.distributed.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-devices", type=int, required=True)
    parser.add_argument("--collective", action="store_true")
    parser.add_argument("--process-group-timeout-seconds", type=float, default=20.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_devices < 1:
        raise SystemExit("--expected-devices must be positive")
    if args.process_group_timeout_seconds <= 0:
        raise SystemExit("--process-group-timeout-seconds must be positive")

    import torch

    if args.collective:
        check_nccl_collective(
            torch,
            expected_devices=args.expected_devices,
            process_group_timeout_seconds=args.process_group_timeout_seconds,
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"ok: CUDA allocation + NCCL all-reduce ({args.expected_devices} ranks)"
            )
    else:
        check_allocations(torch, expected_devices=args.expected_devices)
        print(f"ok: CUDA allocation ({args.expected_devices} device(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
