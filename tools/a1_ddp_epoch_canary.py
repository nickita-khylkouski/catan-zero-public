#!/usr/bin/env python3
"""Eight-rank, non-promotable canary for the sealed B200 learner topology."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import train_bc  # noqa: E402


SCHEMA = "a1-b200-8gpu-ddp-epoch-canary-v1"
WORLD_SIZE = 8
LOCAL_BATCH_SIZE = 512
GLOBAL_BATCH_SIZE = WORLD_SIZE * LOCAL_BATCH_SIZE
SYNTHETIC_ROWS = GLOBAL_BATCH_SIZE * 2 + 97
SEED = 271828


class CanaryError(RuntimeError):
    pass


def _digest(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _atomic_new_json(path: Path, value: dict[str, object]) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value)
    payload["receipt_sha256"] = _digest(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _nvidia_gpu_identities() -> list[dict[str, object]]:
    """Return one stable physical identity per GPU from the driver."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CanaryError(f"cannot bind physical GPU identities: {error}") from error
    records: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            raise CanaryError("nvidia-smi returned malformed GPU identity output")
        try:
            index = int(fields[0])
        except ValueError as error:
            raise CanaryError("nvidia-smi returned a non-integer GPU index") from error
        records.append(
            {
                "physical_index": index,
                "uuid": fields[1],
                "pci_bus_id": fields[2],
                "name": fields[3],
            }
        )
    records.sort(key=lambda record: int(record["physical_index"]))
    if (
        [record["physical_index"] for record in records] != list(range(WORLD_SIZE))
        or len({str(record["uuid"]) for record in records}) != WORLD_SIZE
        or len({str(record["pci_bus_id"]) for record in records}) != WORLD_SIZE
        or any("B200" not in str(record["name"]).upper() for record in records)
    ):
        raise CanaryError("driver inventory is not exactly eight unique B200 GPUs")
    return records


def _rank_slice(rank: int, weights: np.ndarray) -> np.ndarray:
    return train_bc._epoch_order(  # noqa: SLF001
        np.random.default_rng(SEED),
        SYNTHETIC_ROWS,
        LOCAL_BATCH_SIZE,
        {
            "enabled": True,
            "world_size": WORLD_SIZE,
            "rank": rank,
            "local_rank": rank,
        },
        data_sharded=False,
        sample_weights=weights,
    )


def _expected_global_draw(weights: np.ndarray) -> np.ndarray:
    draw = train_bc._epoch_order(  # noqa: SLF001
        np.random.default_rng(SEED),
        SYNTHETIC_ROWS,
        LOCAL_BATCH_SIZE,
        {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0},
        data_sharded=False,
        sample_weights=weights,
    )
    padded_size = int(np.ceil(len(draw) / GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE)
    if padded_size > len(draw):
        draw = np.concatenate((draw, np.resize(draw, padded_size - len(draw))))
    return np.asarray(draw, dtype=np.int64)


def run(out: Path) -> dict[str, object] | None:
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != WORLD_SIZE or rank not in range(WORLD_SIZE):
        raise CanaryError("canary requires torchrun --nproc_per_node=8")
    if local_rank != rank:
        raise CanaryError("canary requires one local rank per physical GPU")
    if torch.cuda.device_count() != WORLD_SIZE:
        raise CanaryError("canary host must expose exactly eight GPUs")
    torch.cuda.set_device(local_rank)
    device_name = torch.cuda.get_device_name(local_rank)
    if "B200" not in device_name.upper():
        raise CanaryError(f"rank {rank} is not mapped to a B200: {device_name!r}")
    identities = _nvidia_gpu_identities()
    identity = identities[local_rank]
    if str(identity["name"]) != device_name:
        raise CanaryError(
            "CUDA logical-rank mapping differs from the driver GPU inventory: "
            f"rank={rank} cuda={device_name!r} driver={identity['name']!r}"
        )
    # Exercise the actual production communication path. A CPU/Gloo-only
    # collective can pass while NCCL, CUDA peer mapping, or the GPU fabric is
    # broken, so it is not sufficient evidence for an eight-rank learner.
    dist.init_process_group("nccl")
    object_group = None
    try:
        collective = torch.tensor(
            [float(rank + 1)], dtype=torch.float64, device=f"cuda:{local_rank}"
        )
        dist.all_reduce(collective, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(local_rank)
        collective_value = float(collective.item())
        expected_collective = float(sum(range(1, WORLD_SIZE + 1)))
        if collective_value != expected_collective:
            raise CanaryError(
                "NCCL CUDA all-reduce returned the wrong value: "
                f"expected={expected_collective} actual={collective_value}"
            )
        rng_contract = train_bc._initialize_training_rng(  # noqa: SLF001
            argparse.Namespace(seed=SEED, training_rng_rank_offset=True),
            {
                "enabled": True,
                "world_size": WORLD_SIZE,
                "rank": rank,
                "local_rank": local_rank,
            },
        )
        dropout_probe = torch.rand(32, device=f"cuda:{local_rank}").cpu().tolist()
        dropout_probe_sha256 = _digest(dropout_probe)

        # Nonuniform weights force the exact production weighted sampler path.
        weights = np.linspace(0.5, 1.5, SYNTHETIC_ROWS, dtype=np.float64)
        local = _rank_slice(rank, weights)
        gathered: list[object] = [None] * WORLD_SIZE
        # Object evidence is small and belongs on a CPU collective. NCCL above
        # independently proves the actual GPU path used by DDP.
        object_group = dist.new_group(backend="gloo")
        dist.all_gather_object(
            gathered,
            {
                "rank": rank,
                "local_rank": local_rank,
                "device_name": device_name,
                "device_identity": identity,
                "nccl_collective_value": collective_value,
                "training_rng_contract": rng_contract,
                "dropout_probe_sha256": dropout_probe_sha256,
                "order": local.tolist(),
            },
            group=object_group,
        )
        if rank != 0:
            return None
        records = sorted(gathered, key=lambda value: int(value["rank"]))
        if [record["rank"] for record in records] != list(range(WORLD_SIZE)):
            raise CanaryError("gathered rank identities are incomplete")
        gathered_identities = [record["device_identity"] for record in records]
        rng_contracts = [record["training_rng_contract"] for record in records]
        dropout_digests = [record["dropout_probe_sha256"] for record in records]
        if (
            gathered_identities != identities
            or len({record["uuid"] for record in gathered_identities}) != WORLD_SIZE
            or len({record["pci_bus_id"] for record in gathered_identities})
            != WORLD_SIZE
            or any(
                float(record["nccl_collective_value"]) != expected_collective
                for record in records
            )
            or [contract["effective_torch_seed"] for contract in rng_contracts]
            != [SEED + rank_value for rank_value in range(WORLD_SIZE)]
            or any(contract["rank_offset_enabled"] is not True for contract in rng_contracts)
            or len(set(dropout_digests)) != WORLD_SIZE
        ):
            raise CanaryError(
                "rank-to-GPU, NCCL collective, or rank-offset RNG evidence drift"
            )
        local_orders = [
            np.asarray(record["order"], dtype=np.int64) for record in records
        ]
        if len({len(order) for order in local_orders}) != 1:
            raise CanaryError("rank epoch slices have different lengths")
        interleaved = np.column_stack(local_orders).reshape(-1)
        expected = _expected_global_draw(weights)
        if not np.array_equal(interleaved, expected):
            raise CanaryError(
                "rank slices do not reconstruct one shared global weighted draw"
            )
        receipt: dict[str, object] = {
            "schema_version": SCHEMA,
            "passed": True,
            "diagnostic_only": True,
            "promotion_eligible": False,
            "hostname": socket.gethostname(),
            "world_size": WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "ddp_shard_data": False,
            "training_rng_rank_offset": True,
            "training_rng_contracts": rng_contracts,
            "dropout_probe_sha256_by_rank": dropout_digests,
            "distributed_backend": "nccl",
            "cuda_collective": {
                "operation": "all_reduce_sum",
                "dtype": "float64",
                "rank_inputs": list(range(1, WORLD_SIZE + 1)),
                "expected": expected_collective,
                "actual_by_rank": [
                    float(record["nccl_collective_value"]) for record in records
                ],
                "passed": True,
            },
            "sampler": "train_bc._epoch_order weighted shared-global draw",
            "synthetic_rows": SYNTHETIC_ROWS,
            "padded_global_draws": int(expected.size),
            "local_draws_per_rank": int(local_orders[0].size),
            "optimizer_steps": int(expected.size // GLOBAL_BATCH_SIZE),
            "seed": SEED,
            "global_draw_sha256": _digest(expected.tolist()),
            "rank_slice_sha256": [
                _digest(order.tolist()) for order in local_orders
            ],
            "gpu_names": [str(record["device_name"]) for record in records],
            "gpu_identities": gathered_identities,
            "tool": {
                "path": str(Path(__file__).resolve()),
                "sha256": _file_sha256(Path(__file__).resolve()),
            },
            "train_bc": {
                "path": str(Path(train_bc.__file__).resolve()),
                "sha256": _file_sha256(Path(train_bc.__file__).resolve()),
            },
            "created_unix_ns": time.time_ns(),
        }
        _atomic_new_json(out, receipt)
        return receipt
    finally:
        if object_group is not None:
            dist.destroy_process_group(object_group)
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args.out)
    except (CanaryError, OSError, ValueError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
