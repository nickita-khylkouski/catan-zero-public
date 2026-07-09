"""Cross-game leaf-batching eval server — PROTOTYPE (CAT-67, Phase D).

BUILD TRIGGER MET (measured under CAT-87, 2026-07-08, B200 gpu1): NN-forward is
~56% of the per-leaf cost (>=25%), generation is forward-bound not featurize-bound
(Rust featurize ~0.065ms), the in-process batcher runs at effective batch size 1
in the one-process-per-worker deployment, and the SM-96%/mem-1% context-thrash
signature is present. See docs/designs/CAT67_eval_server.md sections 4/6 and the
BUILD TRIGGER block.

This is a THROUGHPUT prototype (mp_queue v0 transport), not a production
deployment. It is a THIRD batching layer on top of the within-tree chance
fan-out (evaluate_many) and the within-process micro-batcher
(BatchedEntityGraphRustEvaluator) -- it replaces neither.

Design invariants honored here:
- `RemoteEvalClient` subclasses `EntityGraphRustEvaluator`, so ALL
  featurization (rust_game_to_entity_batch / _entity_batch_via_rust /
  rust_policy_action_ids) and ALL post-processing (softmax@prior_temperature,
  value squash, two-player perspective negation, clip) stay CLIENT-side and
  bit-identical to the local path. Only `policy.forward_legal_np` is centralized
  -- the client hands the server a proxy policy whose forward packs the tensors,
  ships them over IPC, and unpacks the server's raw logits/value back into the
  same dict shape the real policy returns. Outputs match the local path modulo
  GPU nondeterminism inherent to a different batched-matmul composition
  (asserted within tolerance by tools/bench_eval_server.py --parity).
- The client also inherits the CAT-87 warm-topology resolve-skip for free.
- Featurization stays distributed across worker cores; only packed tensors cross
  the IPC boundary (moving featurize into the server would recreate the CPU
  chokepoint as a serial bottleneck).

Reference: docs/designs/CAT67_eval_server.md
"""

from __future__ import annotations

import json
import multiprocessing as mp
import queue as queue_mod
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from catan_zero.search.neural_rust_mcts import (
    EntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)

_DESIGN_DOC = "docs/designs/CAT67_eval_server.md"


@dataclass(frozen=True, slots=True)
class EvalServerConfig:
    """Server + client tuning knobs (mirrors the per-game batcher's two knobs,
    now operating across processes instead of threads).

    Attributes:
        max_batch_size: Max requests packed into one ``forward_legal_np`` window.
        max_wait_ms: Straggler timeout; window flushes at size OR timeout.
        device: Torch device the single server-resident policy runs on.
        transport: "mp_queue" (prototype v0). shared_memory (v1) is not built.
        client_timeout_ms: Per-request client wait before raising (or falling
            back to a local evaluator if one is supplied).
    """

    max_batch_size: int = 64
    max_wait_ms: float = 3.0
    device: str = "cpu"
    transport: str = "mp_queue"
    client_timeout_ms: float = 5000.0


# --- window assembly ---------------------------------------------------------


def _pad_legal_2d(arr: np.ndarray, max_legal: int, fill: Any) -> np.ndarray:
    """(B, L) -> (B, max_legal), padding the legal dim at the tail."""
    rows, width = int(arr.shape[0]), int(arr.shape[1])
    if width == max_legal:
        return arr
    out = np.full((rows, max_legal), fill, dtype=arr.dtype)
    out[:, :width] = arr
    return out


def _pad_legal_3d(arr: np.ndarray, max_legal: int, fill: Any) -> np.ndarray:
    """(B, L, F) -> (B, max_legal, F), padding the legal dim at the tail."""
    rows, width, feat = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    if width == max_legal:
        return arr
    out = np.full((rows, max_legal, feat), fill, dtype=arr.dtype)
    out[:, :width, :] = arr
    return out


def _merge_forward_payloads(
    payloads: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[int]]:
    """Pad every request's legal dim to the window max and concatenate along the
    batch axis. Generalizes `_merge_batched_eval_requests` to variable-B requests
    (each client request may already be a B>1 chance fan-out). Fill values match
    that helper exactly: legal_ids -1, context 0.0, legal_action_tokens 0.0,
    legal_action_target_ids -1, legal_action_mask False; all other (fixed-size,
    non-legal) entity arrays just concatenate.
    """
    max_legal = max(int(p["legal_ids"].shape[1]) for p in payloads)
    row_counts = [int(p["legal_ids"].shape[0]) for p in payloads]

    legal_ids = np.concatenate(
        [_pad_legal_2d(p["legal_ids"], max_legal, -1) for p in payloads], axis=0
    ).astype(np.int64, copy=False)
    context = np.concatenate(
        [_pad_legal_3d(p["context"], max_legal, 0.0) for p in payloads], axis=0
    ).astype(np.float32, copy=False)

    entity: dict[str, np.ndarray] = {}
    for key in payloads[0]["entity"]:
        vals = [p["entity"][key] for p in payloads]
        if key == "legal_action_tokens":
            entity[key] = np.concatenate(
                [_pad_legal_3d(v, max_legal, 0.0) for v in vals], axis=0
            ).astype(vals[0].dtype, copy=False)
        elif key == "legal_action_target_ids":
            entity[key] = np.concatenate(
                [_pad_legal_3d(v, max_legal, -1) for v in vals], axis=0
            ).astype(vals[0].dtype, copy=False)
        elif key == "legal_action_mask":
            entity[key] = np.concatenate(
                [_pad_legal_2d(v, max_legal, False) for v in vals], axis=0
            ).astype(np.bool_, copy=False)
        else:
            entity[key] = np.concatenate(vals, axis=0)
    return entity, legal_ids, context, row_counts


# --- server ------------------------------------------------------------------


_STOP = "__STOP__"


def _server_main(
    checkpoint: str,
    config: EvalServerConfig,
    request_queue: "mp.Queue",
    response_queues: list["mp.Queue"],
    ready_event: "mp.Event",
    handshake: "mp.managers.DictProxy",
    public_observation: bool,
) -> None:
    """Entry point run in the server process (top-level so `spawn` can pickle it)."""
    import torch

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    torch.set_num_threads(1)
    policy = EntityGraphPolicy.load(checkpoint, device=config.device)
    handshake["action_size"] = int(policy.action_size)
    handshake["trained_with_masked_hidden_info"] = bool(
        getattr(policy, "trained_with_masked_hidden_info", False)
    )
    # Warm the CUDA context / kernels once so the first real window isn't slow.
    handshake["ready"] = True
    ready_event.set()

    max_b = max(1, int(config.max_batch_size))
    wait_s = max(0.0, float(config.max_wait_ms) / 1000.0)
    stats = {"windows": 0, "requests": 0, "rows": 0, "sum_batch": 0}
    stopping = False

    while not stopping:
        try:
            first = request_queue.get(timeout=0.25)
        except queue_mod.Empty:
            continue
        if first == _STOP:
            break
        window = [first]
        # Drain whatever is already queued (non-blocking) up to max_b.
        while len(window) < max_b:
            try:
                item = request_queue.get_nowait()
            except queue_mod.Empty:
                break
            if item == _STOP:
                stopping = True
                break
            window.append(item)
        # Straggler wait: give slow producers up to wait_s to fill the window.
        if len(window) < max_b and wait_s > 0.0 and not stopping:
            deadline = time.perf_counter() + wait_s
            while len(window) < max_b:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                try:
                    item = request_queue.get(timeout=remaining)
                except queue_mod.Empty:
                    break
                if item == _STOP:
                    stopping = True
                    break
                window.append(item)

        if not window:
            continue

        ids = [(client_id, req_id) for (client_id, req_id, _payload) in window]
        payloads = [payload for (_c, _r, payload) in window]
        try:
            entity, legal_ids, context, row_counts = _merge_forward_payloads(payloads)
            with torch.no_grad():
                outputs = policy.forward_legal_np(entity, legal_ids, context, return_q=False)
            logits = outputs["logits"].detach().float().cpu().numpy()
            value = outputs["value"].detach().float().cpu().numpy()
            vu = outputs.get("value_uncertainty")
            vu = None if vu is None else vu.detach().float().cpu().numpy()

            offset = 0
            for (client_id, req_id), n_rows in zip(ids, row_counts):
                sl = slice(offset, offset + n_rows)
                offset += n_rows
                result = {"logits": logits[sl].copy(), "value": value[sl].copy()}
                if vu is not None:
                    result["value_uncertainty"] = vu[sl].copy()
                response_queues[client_id].put((req_id, result, None))
            stats["windows"] += 1
            stats["requests"] += len(window)
            stats["rows"] += int(offset)
            stats["sum_batch"] += int(offset)
        except BaseException as error:  # pragma: no cover - surfaced to clients
            for client_id, req_id in ids:
                response_queues[client_id].put((req_id, None, repr(error)))

    handshake["stats"] = dict(stats)


class EvalServer:
    """Single-process inference service holding one GPU/CPU-resident policy and
    servicing packed leaf requests from many game-worker processes over an mp
    Queue. Start it, wait for `ready()`, hand `request_queue` + the per-client
    `response_queues` to `RemoteEvalClient`s, then `stop()` when done.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        num_clients: int,
        config: EvalServerConfig | None = None,
        public_observation: bool = False,
        mp_context: Any | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.config = config or EvalServerConfig()
        self.num_clients = int(num_clients)
        self._ctx = mp_context or mp.get_context("spawn")
        self.request_queue: mp.Queue = self._ctx.Queue()
        self.response_queues: list[mp.Queue] = [self._ctx.Queue() for _ in range(self.num_clients)]
        self._ready = self._ctx.Event()
        self._manager = self._ctx.Manager()
        self._handshake = self._manager.dict()
        self._proc = self._ctx.Process(
            target=_server_main,
            args=(
                checkpoint,
                self.config,
                self.request_queue,
                self.response_queues,
                self._ready,
                self._handshake,
                bool(public_observation),
            ),
            daemon=True,
            name="cat67-eval-server",
        )

    def start(self) -> None:
        self._proc.start()

    def wait_ready(self, timeout: float = 120.0) -> dict[str, Any]:
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError("eval server did not become ready")
        return {
            "action_size": int(self._handshake["action_size"]),
            "trained_with_masked_hidden_info": bool(
                self._handshake["trained_with_masked_hidden_info"]
            ),
        }

    def stop(self) -> dict[str, Any]:
        try:
            self.request_queue.put(_STOP)
        except Exception:
            pass
        self._proc.join(timeout=10.0)
        stats = dict(self._handshake.get("stats", {})) if self._handshake else {}
        if self._proc.is_alive():
            self._proc.terminate()
        return stats


# --- client ------------------------------------------------------------------


class _RemoteForwardProxy:
    """Stands in for `EntityGraphPolicy` inside `RemoteEvalClient`. Exposes only
    what `EntityGraphRustEvaluator` reads off `self.policy`: `action_size`, the
    `trained_with_masked_hidden_info` flag (for the #76 safety-net assert), and
    `forward_legal_np` (routed to the server). Everything else the base
    evaluator needs (featurize, softmax, squash, clip) never touches the policy.
    """

    def __init__(self, client: "RemoteEvalClient", action_size: int, trained_masked: bool) -> None:
        self._client = client
        self.action_size = int(action_size)
        self.trained_with_masked_hidden_info = bool(trained_masked)

    def forward_legal_np(
        self,
        entity_batch: dict[str, np.ndarray],
        legal_action_ids: np.ndarray,
        legal_action_context: np.ndarray,
        *,
        return_q: bool = False,
    ) -> dict[str, Any]:
        return self._client._remote_forward(
            entity_batch, legal_action_ids, legal_action_context, return_q
        )


class RemoteEvalClient(EntityGraphRustEvaluator):
    """Drop-in `RustEvaluator` that featurizes + post-processes locally and
    offloads only the NN forward to a shared `EvalServer`. Because it subclasses
    `EntityGraphRustEvaluator`, `evaluate` / `evaluate_many` are inherited
    verbatim (including the CAT-87 warm-topology resolve-skip); only the policy's
    `forward_legal_np` is redirected across the IPC boundary.
    """

    def __init__(
        self,
        request_queue: "mp.Queue",
        response_queue: "mp.Queue",
        client_id: int,
        *,
        action_size: int,
        trained_with_masked_hidden_info: bool,
        config: EntityGraphRustEvaluatorConfig | None = None,
        client_timeout_ms: float = 5000.0,
        fallback_checkpoint: str | None = None,
        fallback_device: str = "cpu",
    ) -> None:
        proxy = _RemoteForwardProxy(self, action_size, trained_with_masked_hidden_info)
        super().__init__(proxy, config=config)
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._client_id = int(client_id)
        self._req_counter = 0
        self._timeout_s = max(0.001, float(client_timeout_ms) / 1000.0)
        # Failure isolation (design doc risk 5): on server timeout/error, if a
        # fallback checkpoint is configured this client PERMANENTLY degrades to a
        # local in-process policy for the rest of the run rather than hanging the
        # worker. The local policy is loaded lazily on first failure only, so the
        # happy path never pays the per-worker model load (the whole point of the
        # eval server).
        self._fallback_checkpoint = fallback_checkpoint
        self._fallback_device = str(fallback_device)
        self._degraded = False
        self._local_policy: Any = None

    def _ensure_local_policy(self) -> Any:
        if self._local_policy is None:
            from catan_zero.rl.entity_token_policy import EntityGraphPolicy

            self._local_policy = EntityGraphPolicy.load(
                self._fallback_checkpoint, device=self._fallback_device
            )
        return self._local_policy

    def _forward_local(
        self,
        entity_batch: dict[str, np.ndarray],
        legal_action_ids: np.ndarray,
        legal_action_context: np.ndarray,
        return_q: bool,
    ) -> dict[str, Any]:
        return self._ensure_local_policy().forward_legal_np(
            entity_batch, legal_action_ids, legal_action_context, return_q=return_q
        )

    def _remote_forward(
        self,
        entity_batch: dict[str, np.ndarray],
        legal_action_ids: np.ndarray,
        legal_action_context: np.ndarray,
        return_q: bool,
    ) -> dict[str, Any]:
        import torch

        if self._degraded:
            return self._forward_local(
                entity_batch, legal_action_ids, legal_action_context, return_q
            )

        self._req_counter += 1
        req_id = self._req_counter
        payload = {
            "entity": {k: np.asarray(v) for k, v in entity_batch.items()},
            "legal_ids": np.asarray(legal_action_ids),
            "context": np.asarray(legal_action_context),
            "return_q": bool(return_q),
        }
        try:
            self._request_queue.put((self._client_id, req_id, payload))
            got_id, result, error = self._response_queue.get(timeout=self._timeout_s)
            if got_id != req_id:  # pragma: no cover - single-outstanding invariant
                raise RuntimeError(f"response id mismatch: got {got_id} want {req_id}")
            if error is not None:
                raise RuntimeError(f"eval-server forward failed: {error}")
            return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in result.items()}
        except (queue_mod.Empty, RuntimeError, OSError, EOFError, ValueError) as exc:
            if self._fallback_checkpoint is None:
                raise TimeoutError(
                    f"eval-server request failed (client {self._client_id}, req {req_id}); "
                    "no fallback checkpoint configured"
                ) from exc
            print(
                json.dumps(
                    {
                        "progress": "eval_server_client_degraded_to_local",
                        "client_id": self._client_id,
                        "reason": repr(exc),
                    }
                ),
                flush=True,
            )
            self._degraded = True
            return self._forward_local(
                entity_batch, legal_action_ids, legal_action_context, return_q
            )
