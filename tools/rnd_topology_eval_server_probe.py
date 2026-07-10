#!/usr/bin/env python3
"""Fail-closed EvalServer latency/replay probe for topology checkpoints.

The measured path is the production RemoteEvalClient IPC path. Inputs are
deterministic acting-player views collated from real public Catan states; this
is a systems/replay probe, not evidence of playing strength.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _path in (_ROOT, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transport", choices=("mp_queue", "shared_memory"), default="mp_queue")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--legal-actions", type=int, default=64)
    parser.add_argument("--events", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--client-timeout-ms", type=float, default=120_000.0)
    parser.add_argument("--ready-timeout-sec", type=float, default=180.0)
    parser.add_argument("--output", default="")
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_array_bundle_sha256(arrays: Mapping[str, Any]) -> str:
    """Hash names, dtypes, shapes and C-order bytes without pickle metadata."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = arrays[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(b"\0" + array.tobytes(order="C") + b"\0")
    return digest.hexdigest()


def _percentiles_ms(samples_sec: Sequence[float]) -> dict[str, float]:
    if not samples_sec:
        raise ValueError("latency sample set must not be empty")
    values = np.asarray(samples_sec, dtype=np.float64) * 1000.0
    return {
        label: float(np.percentile(values, quantile))
        for label, quantile in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99))
    }


def _validate_handshake(handshake: Mapping[str, Any], *, cuda_graph: bool) -> None:
    if not bool(handshake.get("needs_relational_topology", False)):
        raise RuntimeError(
            "checkpoint handshake does not require relational topology; refusing "
            "to benchmark a non-topology checkpoint"
        )
    if bool(handshake.get("cuda_graph", False)) != bool(cuda_graph):
        raise RuntimeError("EvalServer CUDA-graph handshake disagrees with requested mode")


def _validate_topology_inputs(entity: Mapping[str, Any]) -> dict[str, str]:
    expected_shapes = {
        "hex_vertex_ids": (19, 6),
        "hex_edge_ids": (19, 6),
        "edge_vertex_ids": (72, 2),
    }
    hashes: dict[str, str] = {}
    for key, tail_shape in expected_shapes.items():
        if key not in entity:
            raise RuntimeError(f"collated input omitted required topology tensor {key!r}")
        array = np.asarray(entity[key])
        if array.ndim != len(tail_shape) + 1 or tuple(array.shape[1:]) != tail_shape:
            raise RuntimeError(
                f"invalid {key} shape {array.shape}; expected [B,{','.join(map(str, tail_shape))}]"
            )
        if array.dtype.kind not in "iu":
            raise RuntimeError(f"{key} must have integer dtype, got {array.dtype}")
        hashes[key] = canonical_array_bundle_sha256({key: array})
    return hashes


def _validate_live_legal_ids(
    legal_ids: np.ndarray, legal_mask: np.ndarray, *, action_size: int
) -> None:
    ids = np.asarray(legal_ids)
    mask = np.asarray(legal_mask, dtype=np.bool_)
    if ids.shape != mask.shape:
        raise RuntimeError(
            f"legal ID/mask shape mismatch: ids={ids.shape} mask={mask.shape}"
        )
    if action_size <= 0:
        raise RuntimeError(f"checkpoint handshake has invalid action_size={action_size}")
    live_ids = ids[mask]
    if live_ids.size == 0:
        raise RuntimeError("collated probe input has no live legal actions")
    invalid = live_ids[(live_ids < 0) | (live_ids >= action_size)]
    if invalid.size:
        examples = sorted({int(value) for value in invalid})[:8]
        raise RuntimeError(
            "collated live legal action IDs exceed the checkpoint action space: "
            f"action_size={action_size} invalid_examples={examples}"
        )


def _measure_requests(
    client: Any,
    *,
    entity: Mapping[str, np.ndarray],
    legal_ids: np.ndarray,
    context: np.ndarray,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    def forward() -> Mapping[str, Any]:
        return client._remote_forward(entity, legal_ids, context, False)

    for _ in range(warmup):
        forward()

    latencies: list[float] = []
    output_hashes: list[str] = []
    started = time.perf_counter()
    for _ in range(iterations):
        request_started = time.perf_counter()
        output = forward()
        latencies.append(time.perf_counter() - request_started)
        output_hashes.append(canonical_array_bundle_sha256(output))
    elapsed = time.perf_counter() - started
    unique_hashes = sorted(set(output_hashes))
    if len(unique_hashes) != 1:
        raise RuntimeError(
            "identical replay requests produced non-identical outputs: "
            f"{len(unique_hashes)} hashes"
        )
    rows = int(legal_ids.shape[0]) * iterations
    return {
        "elapsed_sec": elapsed,
        "rows": rows,
        "rows_per_sec": rows / elapsed,
        "latency_ms": _percentiles_ms(latencies),
        "output_sha256": unique_hashes[0],
        "replay_hashes_identical": True,
    }


def _validate_server_stats(
    stats: Mapping[str, Any], *, cuda_graph: bool, expected_requests: int
) -> None:
    if int(stats.get("requests", -1)) != expected_requests:
        raise RuntimeError(
            f"EvalServer request count mismatch: {stats.get('requests')} != {expected_requests}"
        )
    if int(stats.get("forward_calls", -1)) != expected_requests:
        raise RuntimeError(
            "probe requires one actual server forward per replay request; got "
            f"{stats.get('forward_calls')} for {expected_requests} requests"
        )
    if cuda_graph:
        if int(stats.get("cuda_graph_calls", 0)) != expected_requests:
            raise RuntimeError("not every EvalServer forward entered the CUDA-graph runner")
        if int(stats.get("cuda_graph_graph_count", 0)) < 1:
            raise RuntimeError("CUDA-graph runner did not capture a graph")
        if int(stats.get("cuda_graph_fallbacks", 0)) != 0:
            raise RuntimeError(
                "CUDA-graph runner used eager fallback: "
                f"{stats.get('cuda_graph_fallback_reason_histogram', {})}"
            )


def _source_provenance() -> dict[str, Any]:
    paths = (
        "tools/rnd_topology_eval_server_probe.py",
        "tools/rnd_topology_collated_probe.py",
        "src/catan_zero/search/eval_server.py",
        "src/catan_zero/search/cuda_graph_inference.py",
        "src/catan_zero/rl/entity_token_policy.py",
        "src/catan_zero/rl/entity_token_features.py",
    )
    file_hashes = {relative: _sha256_file(_ROOT / relative) for relative in paths}
    aggregate = hashlib.sha256()
    for relative in paths:
        aggregate.update(relative.encode() + b"\0" + bytes.fromhex(file_hashes[relative]))
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "git_commit": commit,
        "source_bundle_sha256": aggregate.hexdigest(),
        "file_sha256": file_hashes,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from catan_zero.search.eval_server import EvalServer, EvalServerConfig, RemoteEvalClient
    from catan_zero.search.neural_rust_mcts import EntityGraphRustEvaluatorConfig
    from tools.rnd_topology_collated_probe import build_collated_public_batch

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is not a file: {checkpoint}")
    for name in ("batch_size", "legal_actions", "events", "warmup", "iterations"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if int(args.iterations) < 2:
        raise ValueError("iterations must be at least 2 to establish replay stability")
    if float(args.client_timeout_ms) <= 0 or float(args.ready_timeout_sec) <= 0:
        raise ValueError("timeouts must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda is unavailable")
    cuda_graph = device.type == "cuda"
    cpu_batch, batch_provenance = build_collated_public_batch(
        batch_size=int(args.batch_size),
        legal_actions=int(args.legal_actions),
        events=int(args.events),
        seed=int(args.seed),
    )
    numpy_batch = {key: value.numpy() for key, value in cpu_batch.items()}
    legal_ids = numpy_batch.pop("legal_action_ids")
    context = numpy_batch.pop("legal_action_context")
    topology_hashes = _validate_topology_inputs(numpy_batch)
    input_hash = canonical_array_bundle_sha256(
        {**numpy_batch, "__legal_ids__": legal_ids, "__context__": context}
    )

    server = EvalServer(
        str(checkpoint),
        num_clients=1,
        config=EvalServerConfig(
            max_batch_size=1,
            max_wait_ms=0.0,
            device=str(device),
            transport=str(args.transport),
            client_timeout_ms=float(args.client_timeout_ms),
            matmul_precision="highest",
            cuda_graph=cuda_graph,
            cuda_graph_batch_buckets=(int(args.batch_size),),
            cuda_graph_warmup_iterations=3,
        ),
        public_observation=True,
    )
    stats: dict[str, Any] = {}
    try:
        server.start()
        handshake = server.wait_ready(timeout=float(args.ready_timeout_sec))
        _validate_handshake(handshake, cuda_graph=cuda_graph)
        _validate_live_legal_ids(
            legal_ids,
            numpy_batch["legal_action_mask"],
            action_size=int(handshake["action_size"]),
        )
        client = RemoteEvalClient(
            server.request_queue_for_client(0),
            server.response_queues[0],
            0,
            action_size=int(handshake["action_size"]),
            trained_with_masked_hidden_info=bool(
                handshake["trained_with_masked_hidden_info"]
            ),
            needs_action_targets=bool(handshake["needs_action_targets"]),
            needs_relational_topology=bool(handshake["needs_relational_topology"]),
            event_token_limit=handshake["event_token_limit"],
            value_categorical_bins=int(handshake["value_categorical_bins"]),
            value_categorical_head_available=bool(
                handshake["value_categorical_head_available"]
            ),
            config=EntityGraphRustEvaluatorConfig(
                public_observation=True,
                cache_size=0,
            ),
            client_timeout_ms=float(args.client_timeout_ms),
        )
        measurements = _measure_requests(
            client,
            entity=numpy_batch,
            legal_ids=legal_ids,
            context=context,
            warmup=int(args.warmup),
            iterations=int(args.iterations),
        )
    finally:
        stats = server.stop()

    expected_requests = int(args.warmup) + int(args.iterations)
    _validate_server_stats(stats, cuda_graph=cuda_graph, expected_requests=expected_requests)
    return {
        "schema_version": "catan-zero-rnd-topology-eval-server-probe/v1",
        "scope": "EvalServer systems/replay evidence only; no playing-strength claim",
        "checkpoint": {
            "sha256": _sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "input": {
            "sha256": input_hash,
            "topology_tensor_sha256": topology_hashes,
            "batch_provenance": batch_provenance,
        },
        "configuration": {
            "device_type": device.type,
            "transport": str(args.transport),
            "cuda_graph": cuda_graph,
            "cuda_graph_batch_buckets": [int(args.batch_size)] if cuda_graph else [],
            "batch_size": int(args.batch_size),
            "warmup_requests": int(args.warmup),
            "measured_requests": int(args.iterations),
        },
        "handshake": dict(handshake),
        "measurements": measurements,
        "server_stats": stats,
        "runtime": {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
        },
        "source_provenance": _source_provenance(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
