"""Cross-game leaf-batching eval server — PROTOTYPE (CAT-67, Phase D).

BUILD TRIGGER MET (measured under CAT-87, 2026-07-08, B200 gpu1): NN-forward is
~56% of the per-leaf cost (>=25%), generation is forward-bound not featurize-bound
(Rust featurize ~0.065ms), the in-process batcher runs at effective batch size 1
in the one-process-per-worker deployment, and the SM-96%/mem-1% context-thrash
signature is present. See docs/designs/CAT67_eval_server.md sections 4/6 and the
BUILD TRIGGER block.

This is a THROUGHPUT prototype with a stable Queue transport and an opt-in
single-slot shared-memory request transport. Responses remain Queue-based. It
is a THIRD batching layer on top of the within-tree chance fan-out
(`evaluate_many`) and the within-process micro-batcher
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

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
import ctypes
import json
import multiprocessing as mp
from multiprocessing.connection import wait as _wait_for_connections
import operator
import queue as queue_mod
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from catan_zero.rl.entity_feature_adapter import (
    CURRENT_RUST_ENTITY_ADAPTER_VERSION,
    policy_entity_feature_adapter_version,
    require_known_entity_feature_adapter,
)
from catan_zero.search.neural_rust_mcts import (
    EntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)

_DESIGN_DOC = "docs/designs/CAT67_eval_server.md"

# These topology/annotation arrays are useful to symmetry transforms and shard
# writers, but the policy forward neither validates nor consumes them.  Keep
# them client-side instead of pickling four immutable arrays on every request.
_NON_FORWARD_ENTITY_KEYS = frozenset(
    {"hex_vertex_ids", "hex_edge_ids", "edge_vertex_ids", "event_target_ids"}
)
_LEGAL_PADDED_ENTITY_KEYS = frozenset(
    {"legal_action_tokens", "legal_action_target_ids", "legal_action_mask"}
)


def _require_implemented_entity_feature_adapter(
    version: object, *, context: str
) -> str:
    resolved = require_known_entity_feature_adapter(version)
    if resolved != CURRENT_RUST_ENTITY_ADAPTER_VERSION:
        raise ValueError(
            f"{context} does not implement entity feature adapter {resolved!r}; "
            f"implemented={CURRENT_RUST_ENTITY_ADAPTER_VERSION!r}"
        )
    return resolved


@dataclass(frozen=True, slots=True)
class EvalServerConfig:
    """Server + client tuning knobs (mirrors the per-game batcher's two knobs,
    now operating across processes instead of threads).

    Attributes:
        max_batch_size: Max requests packed into one ``forward_legal_np`` window.
        max_neural_rows: Optional hard cap on the number of neural rows in any
            single policy forward. Unlike ``max_batch_size``, this counts the
            rows inside batched chance/root-wave requests. Oversized requests
            are row-sliced and reassembled into one response.
        max_wait_ms: Straggler timeout; window flushes at size OR timeout.
        device: Torch device the single server-resident policy runs on.
        transport: ``"mp_queue"`` or opt-in ``"shared_memory"`` request slots.
        client_timeout_ms: Per-request client wait before raising (or falling
            back to a local evaluator if one is supplied).
    """

    max_batch_size: int = 64
    max_neural_rows: int | None = None
    # Immediate queue draining won the H100 sweep at every tested workload;
    # callers may opt into a non-zero aggregation window, but latency is the
    # safer and faster default for the single-outstanding-request clients.
    max_wait_ms: float = 0.0
    device: str = "cpu"
    transport: str = "mp_queue"
    client_timeout_ms: float = 5000.0
    # One request slot per client. Four MiB covers current feature tensors and
    # 11-row dice fan-out with substantial headroom; oversize requests safely
    # fall back to the ordinary Queue payload protocol.
    shared_memory_slot_bytes: int = 4 * 1024 * 1024
    # Optional fail-closed event prefix retained before H2D/model forward.
    # None preserves the checkpoint's historical 64-slot path. A configured
    # limit is accepted only when every omitted event position is masked.
    event_token_limit: int | None = None
    # Optional IPC collector thread. It can overlap Queue deserialization with
    # CUDA work, but remains opt-in because host-memory contention varies by
    # payload/worker count and must be established by an on-box A/B.
    request_collector: bool = False
    # "highest" is strict FP32 matmul. "high" permits CUDA TF32 tensor-core
    # matmuls; kept explicit so throughput/strength gates can compare regimes.
    matmul_precision: str = "highest"
    # Opt-in CUDA Graph capture of the fixed-shape state trunk. The raw policy
    # remains authoritative for checkpoint metadata and the server performs
    # event-tail cropping before this wrapper sees a request.
    cuda_graph: bool = False
    cuda_graph_batch_buckets: tuple[int, ...] = (
        8,
        16,
        24,
        32,
        40,
        48,
        64,
        80,
        96,
        128,
        160,
        192,
    )
    cuda_graph_warmup_iterations: int = 3

    def __post_init__(self) -> None:
        if self.max_neural_rows is not None:
            if isinstance(self.max_neural_rows, bool):
                raise TypeError("max_neural_rows must be an integer, not bool")
            try:
                value = operator.index(self.max_neural_rows)
            except TypeError as error:
                raise TypeError("max_neural_rows must be an integer") from error
            if value <= 0:
                raise ValueError("max_neural_rows must be positive")
            if self.cuda_graph:
                raise ValueError(
                    "max_neural_rows is incompatible with cuda_graph: graph batch "
                    "buckets can execute more physical rows than the logical cap"
                )


# --- window assembly ---------------------------------------------------------


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
    row_counts = [int(p["legal_ids"].shape[0]) for p in payloads]
    total_rows = sum(row_counts)
    max_legal = max(int(p["legal_ids"].shape[1]) for p in payloads)
    context_width = int(payloads[0]["context"].shape[2])

    # Allocate each final array once, then copy every request directly into its
    # slice.  The old pad-then-concatenate path allocated one padded temporary
    # per request and copied it again into np.concatenate's output.
    legal_ids = np.full((total_rows, max_legal), -1, dtype=np.int64)
    context = np.zeros((total_rows, max_legal, context_width), dtype=np.float32)
    entity: dict[str, np.ndarray] = {}
    for key, value in payloads[0]["entity"].items():
        value = np.asarray(value)
        if key == "legal_action_tokens":
            entity[key] = np.zeros(
                (total_rows, max_legal, int(value.shape[2])), dtype=value.dtype
            )
        elif key == "legal_action_target_ids":
            entity[key] = np.full(
                (total_rows, max_legal, int(value.shape[2])), -1, dtype=value.dtype
            )
        elif key == "legal_action_mask":
            entity[key] = np.zeros((total_rows, max_legal), dtype=np.bool_)
        else:
            # Fixed-shape fields need no padding. Let NumPy perform their whole
            # window copy in C rather than issuing requests*fields small Python
            # slice assignments. np.concatenate also preserves the historical
            # mixed-dtype promotion contract.
            entity[key] = np.concatenate(
                [payload["entity"][key] for payload in payloads], axis=0
            )

    offset = 0
    for payload, n_rows in zip(payloads, row_counts):
        end = offset + n_rows
        legal_width = int(payload["legal_ids"].shape[1])
        legal_ids[offset:end, :legal_width] = payload["legal_ids"]
        context[offset:end, :legal_width] = payload["context"]
        for key, destination in entity.items():
            if key not in _LEGAL_PADDED_ENTITY_KEYS:
                continue
            value = payload["entity"][key]
            destination[offset:end, :legal_width] = value
        offset = end
    return entity, legal_ids, context, row_counts


def _slice_forward_payload(
    payload: dict[str, Any], start: int, stop: int
) -> dict[str, Any]:
    """Return a zero-copy row view of one forward request.

    Transport/source telemetry belongs to the original request and is counted
    before slicing, so fragments intentionally carry only forward inputs and
    the request's Q requirement.
    """
    return {
        "entity": {
            key: np.asarray(value)[start:stop]
            for key, value in payload["entity"].items()
        },
        "legal_ids": np.asarray(payload["legal_ids"])[start:stop],
        "context": np.asarray(payload["context"])[start:stop],
        "return_q": bool(payload.get("return_q", False)),
    }


def _forward_groups(
    payloads: list[dict[str, Any]], max_neural_rows: int | None
) -> tuple[list[list[tuple[int, dict[str, Any]]]], int, int]:
    """Plan ordered, row-capped forwards without losing request boundaries.

    Returns ``(groups, oversized_requests, oversized_request_chunks)``. With no
    cap, the single group contains the original payload objects, preserving the
    historical merge/forward path exactly. With a cap, requests remain ordered;
    an oversized request can span groups, and each tuple retains its original
    request index for response reassembly.
    """
    if max_neural_rows is None:
        return [[(index, payload) for index, payload in enumerate(payloads)]], 0, 0

    cap = int(max_neural_rows)
    groups: list[list[tuple[int, dict[str, Any]]]] = []
    group: list[tuple[int, dict[str, Any]]] = []
    group_rows = 0
    oversized_requests = 0
    oversized_request_chunks = 0
    for request_index, payload in enumerate(payloads):
        rows = int(payload["legal_ids"].shape[0])
        if rows <= 0:
            raise ValueError("EvalServer requests must contain at least one row")
        oversized = rows > cap
        if oversized:
            oversized_requests += 1
        start = 0
        while start < rows:
            if group_rows == cap:
                groups.append(group)
                group = []
                group_rows = 0
            take = min(rows - start, cap - group_rows)
            stop = start + take
            fragment = (
                payload
                if start == 0 and stop == rows
                else _slice_forward_payload(payload, start, stop)
            )
            group.append((request_index, fragment))
            group_rows += take
            if oversized:
                oversized_request_chunks += 1
            start = stop
    if group:
        groups.append(group)
    return groups, oversized_requests, oversized_request_chunks


def _legal_cell_counts(payloads: list[dict[str, Any]]) -> tuple[int, int]:
    """Return true legal cells and request-rectangular cells.

    A request may itself contain multiple rows (for example a batched chance
    fan-out), so ``rows * request_width`` is only an upper bound on useful
    actions.  ``legal_action_mask`` is the forward contract's authoritative
    per-row occupancy signal and excludes both that intra-request padding and
    the additional padding introduced while merging a server window.
    """
    request_cells = sum(
        int(payload["legal_ids"].shape[0]) * int(payload["legal_ids"].shape[1])
        for payload in payloads
    )
    # Production entity payloads always carry the authoritative mask. Retain
    # compatibility with narrow custom/test policies that omit action-token
    # fields entirely: their legal_ids rectangle is all real by contract.
    real_cells = sum(
        int(np.count_nonzero(mask))
        if (mask := payload["entity"].get("legal_action_mask")) is not None
        else int(payload["legal_ids"].size)
        for payload in payloads
    )
    return real_cells, request_cells


def _event_tail_info(
    entity: dict[str, np.ndarray], event_token_limit: int | None
) -> tuple[np.ndarray, np.ndarray, int, int | None]:
    """Validate event arrays and return normalized crop information."""
    event_mask = np.asarray(entity["event_mask"], dtype=np.bool_)
    event_tokens = np.asarray(entity["event_tokens"])
    if event_mask.ndim != 2:
        raise ValueError(f"event_mask must be rank 2, got {event_mask.shape}")
    if event_tokens.ndim != 3:
        raise ValueError(f"event_tokens must be rank 3, got {event_tokens.shape}")
    if event_tokens.shape[:2] != event_mask.shape:
        raise ValueError(
            "event token/mask shape mismatch: "
            f"tokens={event_tokens.shape} mask={event_mask.shape}"
        )
    padded_width = int(event_mask.shape[1])
    active_columns = np.flatnonzero(np.any(event_mask, axis=0))
    required_width = int(active_columns[-1] + 1) if active_columns.size else 0
    if event_token_limit is None:
        return event_mask, event_tokens, required_width, None
    if isinstance(event_token_limit, bool):
        raise TypeError("event_token_limit must be an integer, not bool")
    try:
        limit = operator.index(event_token_limit)
    except TypeError as error:
        raise TypeError("event_token_limit must be an integer") from error
    if not 0 <= limit <= padded_width:
        raise ValueError(
            f"event_token_limit {event_token_limit!r} is outside [0, {padded_width}]"
        )
    if required_width > limit:
        raise ValueError(
            "event_token_limit would remove an unmasked event token: "
            f"required={required_width} limit={limit}"
        )
    return event_mask, event_tokens, required_width, limit


def _crop_masked_event_tail(
    entity: dict[str, np.ndarray], event_token_limit: int | None
) -> int:
    """Crop only a batch-wide all-masked event suffix, returning required width."""
    event_mask, event_tokens, required_width, limit = _event_tail_info(
        entity, event_token_limit
    )
    event_targets = entity.get("event_target_ids")
    if event_targets is not None:
        event_targets = np.asarray(event_targets)
        if event_targets.ndim != 3 or event_targets.shape[:2] != event_mask.shape:
            raise ValueError(
                "event target/mask shape mismatch: "
                f"targets={event_targets.shape} mask={event_mask.shape}"
            )
    if limit is None:
        return required_width
    entity["event_mask"] = event_mask[:, :limit]
    entity["event_tokens"] = event_tokens[:, :limit]
    if event_targets is not None:
        entity["event_target_ids"] = event_targets[:, :limit]
    return required_width


def _crop_payload_event_tails_before_merge(
    payloads: list[dict[str, Any]], event_token_limit: int | None
) -> int | None:
    """Validate then view-crop request event arrays before window allocation.

    The historical path allocated and copied every request's 64x41 event tail,
    then discarded it after merging.  For an explicit limit, validate *all*
    requests first and only then replace each server-local payload entry with a
    zero-copy prefix view.  This preserves fail-closed behavior without leaving
    a partially mutated window when a later request is invalid. ``None`` keeps
    the original full-width path and has no per-request overhead.
    """
    if event_token_limit is None:
        return None
    if isinstance(event_token_limit, bool):
        raise TypeError("event_token_limit must be an integer, not bool")
    try:
        limit = operator.index(event_token_limit)
    except TypeError as error:
        raise TypeError("event_token_limit must be an integer") from error
    event_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = []
    for payload in payloads:
        event_mask = np.asarray(payload["entity"]["event_mask"], dtype=np.bool_)
        event_tokens = np.asarray(payload["entity"]["event_tokens"])
        if event_mask.ndim != 2:
            raise ValueError(f"event_mask must be rank 2, got {event_mask.shape}")
        if event_tokens.ndim != 3:
            raise ValueError(f"event_tokens must be rank 3, got {event_tokens.shape}")
        if event_tokens.shape[:2] != event_mask.shape:
            raise ValueError(
                "event token/mask shape mismatch: "
                f"tokens={event_tokens.shape} mask={event_mask.shape}"
            )
        event_targets = payload["entity"].get("event_target_ids")
        if event_targets is not None:
            event_targets = np.asarray(event_targets)
            if event_targets.ndim != 3 or event_targets.shape[:2] != event_mask.shape:
                raise ValueError(
                    "event target/mask shape mismatch: "
                    f"targets={event_targets.shape} mask={event_mask.shape}"
                )
        padded_width = int(event_mask.shape[1])
        if not 0 <= limit <= padded_width:
            raise ValueError(
                f"event_token_limit {event_token_limit!r} is outside "
                f"[0, {padded_width}]"
            )
        event_arrays.append((event_mask, event_tokens, event_targets))
    # One C-level concatenate + reduction is materially cheaper than one NumPy
    # reduction per request at 36-128 requests/window. Widths are normally
    # identical; retain a correct fallback for mixed-width diagnostic clients.
    widths = {
        int(event_mask.shape[1]) for event_mask, _tokens, _targets in event_arrays
    }
    if len(widths) <= 1:
        merged_mask = np.concatenate(
            [event_mask for event_mask, _tokens, _targets in event_arrays], axis=0
        )
        active_columns = np.flatnonzero(np.any(merged_mask, axis=0))
        required_width = int(active_columns[-1] + 1) if active_columns.size else 0
    else:
        required_width = 0
        for event_mask, _tokens, _targets in event_arrays:
            active_columns = np.flatnonzero(np.any(event_mask, axis=0))
            if active_columns.size:
                required_width = max(required_width, int(active_columns[-1] + 1))
    if required_width > limit:
        raise ValueError(
            "event_token_limit would remove an unmasked event token: "
            f"required={required_width} limit={limit}"
        )
    for payload, (event_mask, event_tokens, event_targets) in zip(
        payloads, event_arrays
    ):
        payload["entity"]["event_mask"] = event_mask[:, :limit]
        payload["entity"]["event_tokens"] = event_tokens[:, :limit]
        if event_targets is not None:
            payload["entity"]["event_target_ids"] = event_targets[:, :limit]
    return required_width


def _payload_event_source_counts(payload: dict[str, Any]) -> tuple[int, int]:
    """Recover pre-client-crop event occupancy for server telemetry."""
    if (
        "_event_source_active_tokens" in payload
        and "_event_source_padded_tokens" in payload
    ):
        return (
            int(payload["_event_source_active_tokens"]),
            int(payload["_event_source_padded_tokens"]),
        )
    event_mask = np.asarray(payload["entity"]["event_mask"], dtype=np.bool_)
    return int(np.count_nonzero(event_mask)), int(event_mask.size)


# --- server ------------------------------------------------------------------


_STOP = "__STOP__"
_SHARED_REQUEST = "__SHARED_REQUEST_V1__"


def _aligned_offset(offset: int, alignment: int = 64) -> int:
    """Round a shared-slot byte offset up for cache-line-friendly arrays."""
    return (int(offset) + alignment - 1) // alignment * alignment


def _write_shared_array(
    slot: Any,
    capacity: int,
    value: Any,
    offset: int,
) -> tuple[tuple[str, tuple[int, ...], int, int], int]:
    """Copy one ndarray into a shared request slot and return compact metadata."""
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot use EvalServer shared-memory transport")
    offset = _aligned_offset(offset)
    end = offset + int(array.nbytes)
    if end > int(capacity):
        raise BufferError(
            f"EvalServer shared request needs {end} bytes, slot has {capacity}"
        )
    destination = np.frombuffer(slot, dtype=np.uint8, count=array.nbytes, offset=offset)
    destination[:] = array.view(np.uint8).reshape(-1)
    return (
        array.dtype.str,
        tuple(int(v) for v in array.shape),
        offset,
        array.nbytes,
    ), end


def _read_shared_array(
    slot: Any, descriptor: tuple[str, tuple[int, ...], int, int]
) -> np.ndarray:
    """Return a zero-copy ndarray view described by shared-slot metadata."""
    dtype_string, shape, offset, nbytes = descriptor
    dtype = np.dtype(dtype_string)
    expected = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
    if expected != int(nbytes):
        raise ValueError(
            f"invalid shared request descriptor: shape/dtype need {expected} bytes, "
            f"metadata says {nbytes}"
        )
    return np.ndarray(shape=shape, dtype=dtype, buffer=slot, offset=int(offset))


def _pack_shared_request(
    slot: Any, capacity: int, payload: dict[str, Any]
) -> dict[str, Any]:
    """Pack the forward payload into one single-outstanding client slot."""
    offset = 0
    entity_metadata: dict[str, Any] = {}
    for key, value in payload["entity"].items():
        entity_metadata[key], offset = _write_shared_array(
            slot, capacity, value, offset
        )
    legal_metadata, offset = _write_shared_array(
        slot, capacity, payload["legal_ids"], offset
    )
    context_metadata, offset = _write_shared_array(
        slot, capacity, payload["context"], offset
    )
    metadata = {
        "entity": entity_metadata,
        "legal_ids": legal_metadata,
        "context": context_metadata,
        "return_q": bool(payload.get("return_q", False)),
        "used_bytes": int(offset),
    }
    for key in ("_event_source_active_tokens", "_event_source_padded_tokens"):
        if key in payload:
            metadata[key] = int(payload[key])
    return metadata


def _unpack_shared_request(slot: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    """Recreate forward arrays as views; no request tensor is copied here."""
    payload = {
        "entity": {
            key: _read_shared_array(slot, descriptor)
            for key, descriptor in metadata["entity"].items()
        },
        "legal_ids": _read_shared_array(slot, metadata["legal_ids"]),
        "context": _read_shared_array(slot, metadata["context"]),
        "return_q": bool(metadata.get("return_q", False)),
        "_transport": "shared_memory",
        "_transport_bytes": int(metadata.get("used_bytes", 0)),
    }
    for key in ("_event_source_active_tokens", "_event_source_padded_tokens"):
        if key in metadata:
            payload[key] = int(metadata[key])
    return payload


class _SharedMemoryRequestEndpoint:
    """Client-side single-slot request sender.

    Only the compact descriptor is pickled into the notification queue. The
    tensor bytes live in a spawn-inherited ``RawArray``. A client may issue one
    request at a time, matching ``RemoteEvalClient``'s existing invariant.
    """

    def __init__(
        self,
        notification_queue: Any,
        slot: Any,
        in_flight_request_id: Any,
        slot_bytes: int,
        client_id: int,
    ) -> None:
        self._notification_queue = notification_queue
        self._slot = slot
        self._in_flight_request_id = in_flight_request_id
        self._slot_bytes = int(slot_bytes)
        self._client_id = int(client_id)

    def put(self, item: Any) -> None:
        if item == _STOP:
            self._notification_queue.put(item)
            return
        client_id, req_id, payload = item
        if int(client_id) != self._client_id:
            raise ValueError(
                f"shared request endpoint {self._client_id} received client {client_id}"
            )
        # Enforce the single-outstanding invariant in the transport itself.
        # This state is shared between every facade for a client slot, so even
        # the backward-compatible receiver.put() API cannot overwrite an
        # unread descriptor. A second request safely uses the ordinary Queue
        # payload path until the matching response releases the slot.
        with self._in_flight_request_id.get_lock():
            if int(self._in_flight_request_id.value) >= 0:
                self._notification_queue.put(item)
                return
            try:
                metadata = _pack_shared_request(self._slot, self._slot_bytes, payload)
            except (BufferError, TypeError, ValueError):
                # Correctness-first overflow/unsupported-dtype fallback. The
                # server accepts ordinary Queue payloads in the same stream.
                self._notification_queue.put(item)
                return
            self._in_flight_request_id.value = int(req_id)
        try:
            self._notification_queue.put(
                (_SHARED_REQUEST, self._client_id, int(req_id), metadata)
            )
        except BaseException:
            # No notification means the server cannot be using this slot.
            with self._in_flight_request_id.get_lock():
                if int(self._in_flight_request_id.value) == int(req_id):
                    self._in_flight_request_id.value = -1
            raise

    def request_complete(self, client_id: int, req_id: int) -> None:
        """Release the slot only for the response matching its descriptor."""
        if int(client_id) != self._client_id:
            raise ValueError(
                f"shared request endpoint {self._client_id} completed client {client_id}"
            )
        with self._in_flight_request_id.get_lock():
            if int(self._in_flight_request_id.value) == int(req_id):
                self._in_flight_request_id.value = -1


class _SharedMemoryRequestReceiver:
    """Server-side queue facade that expands shared-slot notifications."""

    def __init__(
        self,
        notification_queue: Any,
        slots: list[Any],
        in_flight_request_ids: list[Any],
    ) -> None:
        self._notification_queue = notification_queue
        self._slots = slots
        self._in_flight_request_ids = in_flight_request_ids

    @property
    def _reader(self) -> Any:
        # _GatedRequestCollector waits on the underlying Queue connection.
        return self._notification_queue._reader

    def _decode_item(self, item: Any) -> Any:
        if not (
            isinstance(item, tuple) and len(item) == 4 and item[0] == _SHARED_REQUEST
        ):
            return item
        _marker, client_id, req_id, metadata = item
        client_id = int(client_id)
        if not 0 <= client_id < len(self._slots):
            raise ValueError(f"invalid shared-memory client id {client_id}")
        payload = _unpack_shared_request(self._slots[client_id], metadata)
        return client_id, int(req_id), payload

    def get(self, *args: Any, **kwargs: Any) -> Any:
        return self._decode_item(self._notification_queue.get(*args, **kwargs))

    def get_nowait(self) -> Any:
        return self._decode_item(self._notification_queue.get_nowait())

    def put(self, item: Any) -> None:
        # This preserves the historical ``server.request_queue`` API. It does
        # transfer all slot handles to that client under spawn, so launchers
        # should prefer ``request_queue_for_client`` once convenient.
        if item == _STOP:
            self._notification_queue.put(item)
            return
        client_id, _req_id, _payload = item
        client_id = int(client_id)
        if not 0 <= client_id < len(self._slots):
            raise ValueError(f"invalid shared-memory client id {client_id}")
        endpoint = _SharedMemoryRequestEndpoint(
            self._notification_queue,
            self._slots[client_id],
            self._in_flight_request_ids[client_id],
            len(self._slots[client_id]),
            client_id,
        )
        endpoint.put(item)

    def request_complete(self, client_id: int, req_id: int) -> None:
        client_id = int(client_id)
        if not 0 <= client_id < len(self._slots):
            raise ValueError(f"invalid shared-memory client id {client_id}")
        with self._in_flight_request_ids[client_id].get_lock():
            if int(self._in_flight_request_ids[client_id].value) == int(req_id):
                self._in_flight_request_ids[client_id].value = -1

    def close(self) -> None:
        self._notification_queue.close()

    def join_thread(self) -> None:
        self._notification_queue.join_thread()


def _make_shared_request_transport(
    ctx: Any, num_clients: int, slot_bytes: int
) -> tuple[_SharedMemoryRequestReceiver, list[_SharedMemoryRequestEndpoint]]:
    if int(slot_bytes) <= 0:
        raise ValueError("shared_memory_slot_bytes must be positive")
    notification_queue = ctx.Queue()
    slots = [ctx.RawArray(ctypes.c_ubyte, int(slot_bytes)) for _ in range(num_clients)]
    in_flight_request_ids = [
        ctx.Value(ctypes.c_longlong, -1, lock=True) for _ in range(num_clients)
    ]
    receiver = _SharedMemoryRequestReceiver(
        notification_queue, slots, in_flight_request_ids
    )
    endpoints = [
        _SharedMemoryRequestEndpoint(
            notification_queue,
            slot,
            in_flight_request_ids[client_id],
            slot_bytes,
            client_id,
        )
        for client_id, slot in enumerate(slots)
    ]
    return receiver, endpoints


class _GatedRequestCollector:
    """Deserialize mp.Queue requests only while the inference loop permits it.

    ``threading.Event`` alone cannot pause a collector already blocked inside
    ``Queue.get()``. This collector instead waits on both the queue's pipe and a
    wakeup socket. The activity lock covers the complete ``get`` (including
    unpickling), so ``paused()`` first wakes an idle collector and then waits out
    any in-flight deserialize before entering the protected merge/scatter phase.
    Both idle waits are kernel-blocking; there is no timeout polling loop.
    """

    def __init__(self, request_queue: "mp.Queue") -> None:
        self._request_queue = request_queue
        self._queue_reader = request_queue._reader
        self.ready_requests: queue_mod.Queue[Any] = queue_mod.Queue()
        self._gate = threading.Event()
        self._gate.set()
        self._shutdown = threading.Event()
        self._activity_lock = threading.Lock()
        self._wake_reader, self._wake_writer = socket.socketpair()
        self._wake_reader.setblocking(False)
        self._wake_writer.setblocking(False)
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="cat67-request-collector",
        )

    def start(self) -> None:
        self._thread.start()

    def _wake(self) -> None:
        try:
            self._wake_writer.send(b"\0")
        except (BlockingIOError, OSError):
            # A full socket already contains a wakeup byte. OSError is only
            # possible during process teardown, when another wake is moot.
            pass

    def _drain_wakeup(self) -> None:
        while True:
            try:
                if not self._wake_reader.recv(4096):
                    return
            except BlockingIOError:
                return

    def _run(self) -> None:
        try:
            while True:
                self._gate.wait()
                if self._shutdown.is_set():
                    return
                ready = _wait_for_connections((self._queue_reader, self._wake_reader))
                if self._wake_reader in ready:
                    self._drain_wakeup()
                if (
                    self._shutdown.is_set()
                    or not self._gate.is_set()
                    or self._queue_reader not in ready
                ):
                    continue
                with self._activity_lock:
                    # paused() may have cleared the gate while this thread was
                    # waiting for a merge/scatter phase to release the lock.
                    if self._shutdown.is_set() or not self._gate.is_set():
                        continue
                    item = self._request_queue.get()
                self.ready_requests.put(item)
                if item == _STOP:
                    return
        except BaseException as error:  # pragma: no cover - process/pipe failure
            if not self._shutdown.is_set():
                self.ready_requests.put(error)

    @contextmanager
    def paused(self) -> Iterator[None]:
        self._gate.clear()
        self._wake()
        try:
            # Acquisition is the acknowledgement: no get/unpickle is active,
            # and the cleared gate prevents a new one from starting.
            with self._activity_lock:
                yield
        finally:
            self._gate.set()

    def close(self, timeout: float = 1.0) -> None:
        self._shutdown.set()
        # The collector can be blocked either on the pause gate or in the OS
        # connection wait. Release both before joining it.
        self._gate.set()
        self._wake()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError("request collector did not stop")
        self._wake_reader.close()
        self._wake_writer.close()


def _policy_needs_action_targets(policy: Any) -> bool:
    """Whether this loaded policy consumes ``legal_action_target_ids``.

    Unknown/custom policy objects fail closed to transporting the field. The
    concrete EntityGraphPolicy always exposes its config, whose two target-aware
    heads are the only forward branches that read these IDs.
    """
    policy_config = getattr(policy, "config", None)
    if policy_config is None:
        return True
    return bool(
        str(getattr(policy_config, "state_trunk", "transformer")) != "transformer"
        or getattr(policy_config, "action_target_gather", False)
        or getattr(policy_config, "edge_policy_head", False)
    )


def _policy_needs_relational_topology(policy: Any) -> bool:
    """Whether this policy consumes immutable board topology tensors."""
    policy_config = getattr(policy, "config", None)
    if policy_config is None:
        return True
    return bool(
        str(getattr(policy_config, "state_trunk", "transformer")) != "transformer"
        or getattr(policy_config, "topology_residual_adapter", False)
    )


def _make_forward_policy(policy: Any, config: EvalServerConfig) -> Any:
    """Return the narrow inference wrapper while retaining ``policy`` itself.

    CUDA Graphs are deliberately restricted to the strict-FP32 production
    regime. Event cropping belongs to EvalServer, so the runner receives
    ``event_token_limit=None`` and cannot crop the already-validated batch a
    second time.
    """
    if not config.cuda_graph:
        return policy
    if str(config.matmul_precision) != "highest":
        raise ValueError(
            "EvalServer CUDA Graph inference requires matmul_precision='highest'"
        )
    from catan_zero.search.cuda_graph_inference import (
        CudaGraphInferenceConfig,
        CudaGraphInferenceRunner,
    )

    return CudaGraphInferenceRunner(
        policy,
        CudaGraphInferenceConfig(
            enabled=True,
            batch_buckets=tuple(config.cuda_graph_batch_buckets),
            event_token_limit=None,
            warmup_iterations=int(config.cuda_graph_warmup_iterations),
        ),
    )


def _record_cuda_graph_call(stats: dict[str, Any], forward_policy: Any) -> None:
    """Accumulate runner-path telemetry after one successful forward call."""
    stats["cuda_graph_calls"] += 1
    stats["cuda_graph_graph_count"] = int(getattr(forward_policy, "graph_count", 0))
    if getattr(forward_policy, "last_path", None) == "cuda_graph":
        return
    stats["cuda_graph_fallbacks"] += 1
    reason = str(getattr(forward_policy, "last_fallback_reason", None) or "unknown")
    stats["cuda_graph_last_fallback_reason"] = reason
    reasons = stats["cuda_graph_fallback_reason_histogram"]
    reasons[reason] = int(reasons.get(reason, 0)) + 1


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
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    precision = str(config.matmul_precision)
    if precision not in {"highest", "high", "medium"}:
        raise ValueError(f"invalid EvalServer matmul_precision={precision!r}")
    torch.set_float32_matmul_precision(precision)
    policy = EntityGraphPolicy.load(checkpoint, device=config.device)
    adapter_contract = _require_implemented_entity_feature_adapter(
        policy_entity_feature_adapter_version(policy),
        context="EvalServer",
    )
    handshake["action_size"] = int(policy.action_size)
    handshake["entity_feature_adapter"] = adapter_contract
    handshake["trained_with_masked_hidden_info"] = bool(
        getattr(policy, "trained_with_masked_hidden_info", False)
    )
    handshake["needs_action_targets"] = _policy_needs_action_targets(policy)
    handshake["needs_relational_topology"] = _policy_needs_relational_topology(policy)
    handshake["matmul_precision"] = precision
    handshake["transport"] = str(config.transport)
    handshake["max_neural_rows"] = config.max_neural_rows
    handshake["event_token_limit"] = config.event_token_limit
    handshake["cuda_graph"] = bool(config.cuda_graph)
    handshake["cuda_graph_batch_buckets"] = tuple(config.cuda_graph_batch_buckets)
    handshake["cuda_graph_warmup_iterations"] = int(config.cuda_graph_warmup_iterations)
    forward_policy = _make_forward_policy(policy, config)
    policy_model = getattr(policy, "model", None)
    categorical_bins = int(
        getattr(policy_model, "value_categorical_bins", 0) or 0
    )
    missing_state_keys = tuple(getattr(policy, "_checkpoint_missing_state_keys", ()))
    trained_readouts = tuple(
        str(readout)
        for readout in getattr(policy, "trained_value_readouts", ("scalar",))
    )
    handshake["value_categorical_bins"] = categorical_bins
    handshake["value_categorical_head_available"] = bool(
        categorical_bins >= 2
        and getattr(policy_model, "value_categorical_head", None) is not None
        and not any(
            str(key).startswith("value_categorical_head.") for key in missing_state_keys
        )
        and "categorical" in trained_readouts
    )
    cuda_memory_device: Any | None = None
    cuda_api = getattr(torch, "cuda", None)
    if cuda_api is not None and cuda_api.is_available():
        configured_device = torch.device(config.device)
        if configured_device.type == "cuda":
            cuda_memory_device = configured_device
            # The model's live allocation remains in the new peak baseline;
            # discard only checkpoint-load transients so row-cap sweeps compare
            # steady-state policy plus forward workspace memory.
            cuda_api.reset_peak_memory_stats(cuda_memory_device)
    # Warm the CUDA context / kernels once so the first real window isn't slow.
    handshake["ready"] = True
    ready_event.set()

    max_b = max(1, int(config.max_batch_size))
    wait_s = max(0.0, float(config.max_wait_ms) / 1000.0)
    stats = {
        "windows": 0,
        "requests": 0,
        "rows": 0,
        "max_window_requests": 0,
        "max_window_rows": 0,
        "max_neural_rows": config.max_neural_rows,
        "forward_calls": 0,
        "max_forward_rows": 0,
        "forward_row_histogram": {},
        "oversized_requests": 0,
        "oversized_request_chunks": 0,
        "real_legal_cells": 0,
        "request_legal_cells": 0,
        "padded_legal_cells": 0,
        "first_request_get_sec": 0.0,
        "queued_request_drain_sec": 0.0,
        "straggler_wait_sec": 0.0,
        "merge_sec": 0.0,
        "forward_and_d2h_sec": 0.0,
        "response_enqueue_sec": 0.0,
        "window_request_histogram": {},
        "window_row_histogram": {},
        "window_legal_width_histogram": {},
        "collector_enabled": bool(config.request_collector),
        "transport": str(config.transport),
        "shared_memory_requests": 0,
        "shared_memory_request_bytes": 0,
        "queue_payload_requests": 0,
        "event_token_limit": config.event_token_limit,
        "event_active_tokens": 0,
        "event_padded_tokens": 0,
        "event_required_width_histogram": {},
        "cuda_graph_enabled": bool(config.cuda_graph),
        "cuda_graph_calls": 0,
        "cuda_graph_fallbacks": 0,
        "cuda_graph_graph_count": 0,
        "cuda_graph_last_fallback_reason": None,
        "cuda_graph_fallback_reason_histogram": {},
        "cuda_memory_stats_enabled": cuda_memory_device is not None,
        "cuda_peak_memory_allocated_bytes": None,
        "cuda_peak_memory_reserved_bytes": None,
    }
    stopping = False

    collector: _GatedRequestCollector | None = None
    request_source: Any = request_queue
    if config.request_collector:
        # Multiprocessing Queue.get() performs pipe reads and NumPy unpickling
        # in the calling thread. A single collector remains the only mp.Queue
        # consumer while the inference loop drains a cheap in-process FIFO.
        # The gated collector overlaps deserialize with CUDA, but acknowledges
        # every pause before merge/scatter so host-memory contention cannot leak
        # through an already-blocked Queue.get().
        collector = _GatedRequestCollector(request_queue)
        collector.start()
        request_source = collector.ready_requests

    while not stopping:
        first_get_started = time.perf_counter()
        try:
            first = request_source.get(timeout=0.25)
        except queue_mod.Empty:
            stats["first_request_get_sec"] += time.perf_counter() - first_get_started
            continue
        stats["first_request_get_sec"] += time.perf_counter() - first_get_started
        if isinstance(first, BaseException):
            raise RuntimeError("eval-server request collector failed") from first
        if first == _STOP:
            break
        window = [first]
        collector_error: BaseException | None = None
        # Drain whatever is already queued (non-blocking) up to max_b.
        drain_started = time.perf_counter()
        while len(window) < max_b:
            try:
                item = request_source.get_nowait()
            except queue_mod.Empty:
                break
            if isinstance(item, BaseException):
                collector_error = item
                stopping = True
                break
            if item == _STOP:
                stopping = True
                break
            window.append(item)
        stats["queued_request_drain_sec"] += time.perf_counter() - drain_started
        # Straggler wait: give slow producers up to wait_s to fill the window.
        if len(window) < max_b and wait_s > 0.0 and not stopping:
            straggler_started = time.perf_counter()
            deadline = time.perf_counter() + wait_s
            while len(window) < max_b:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                try:
                    item = request_source.get(timeout=remaining)
                except queue_mod.Empty:
                    break
                if isinstance(item, BaseException):
                    collector_error = item
                    stopping = True
                    break
                if item == _STOP:
                    stopping = True
                    break
                window.append(item)
            stats["straggler_wait_sec"] += time.perf_counter() - straggler_started

        if not window:
            continue

        ids = [(client_id, req_id) for (client_id, req_id, _payload) in window]
        payloads = [payload for (_c, _r, payload) in window]
        return_q_flags = [bool(payload.get("return_q", False)) for payload in payloads]
        try:
            with collector.paused() if collector is not None else nullcontext():
                if collector_error is not None:
                    raise RuntimeError(
                        "eval-server request collector failed"
                    ) from collector_error
                merge_started = time.perf_counter()
                event_source_counts = [
                    _payload_event_source_counts(payload) for payload in payloads
                ]
                event_active_tokens = sum(counts[0] for counts in event_source_counts)
                event_padded_tokens = sum(counts[1] for counts in event_source_counts)
                premerge_required_event_width: int | None = None
                if config.event_token_limit is not None:
                    client_cropped_events = all(
                        "_event_source_padded_tokens" in payload for payload in payloads
                    )
                    if not client_cropped_events:
                        premerge_required_event_width = (
                            _crop_payload_event_tails_before_merge(
                                payloads, config.event_token_limit
                            )
                        )
                forward_groups, oversized_requests, oversized_chunks = _forward_groups(
                    payloads, config.max_neural_rows
                )
                stats["event_active_tokens"] += int(event_active_tokens)
                stats["event_padded_tokens"] += int(event_padded_tokens)
                stats["merge_sec"] += time.perf_counter() - merge_started

            result_parts: list[dict[str, list[np.ndarray]]] = [
                {} for _payload in payloads
            ]
            required_event_widths: list[int] = []
            padded_legal_cells = 0
            for forward_group in forward_groups:
                group_request_indices = [item[0] for item in forward_group]
                group_payloads = [item[1] for item in forward_group]
                group_return_q = [
                    bool(payload.get("return_q", False)) for payload in group_payloads
                ]
                with collector.paused() if collector is not None else nullcontext():
                    merge_started = time.perf_counter()
                    entity, legal_ids, context, row_counts = _merge_forward_payloads(
                        group_payloads
                    )
                    required_event_widths.append(
                        _crop_masked_event_tail(entity, config.event_token_limit)
                    )
                    stats["merge_sec"] += time.perf_counter() - merge_started

                forward_rows = int(legal_ids.shape[0])
                if config.max_neural_rows is not None and forward_rows > int(
                    config.max_neural_rows
                ):
                    raise AssertionError(
                        "EvalServer internal row-cap violation: "
                        f"forward={forward_rows} cap={config.max_neural_rows}"
                    )
                forward_started = time.perf_counter()
                forward_kwargs = {"return_q": any(group_return_q)}
                if bool(
                    getattr(
                        forward_policy,
                        "supports_final_vp_selection",
                        False,
                    )
                ):
                    forward_kwargs["return_final_vp"] = False
                with torch.inference_mode():
                    outputs = forward_policy.forward_legal_np(
                        entity,
                        legal_ids,
                        context,
                        **forward_kwargs,
                    )
                if config.cuda_graph:
                    _record_cuda_graph_call(stats, forward_policy)
                logits = outputs["logits"].detach().float().cpu().numpy()
                value = outputs["value"].detach().float().cpu().numpy()
                value_categorical = outputs.get("value_categorical")
                value_categorical = (
                    None
                    if value_categorical is None
                    else value_categorical.detach().float().cpu().numpy()
                )
                vu = outputs.get("value_uncertainty")
                vu = None if vu is None else vu.detach().float().cpu().numpy()
                q_values = outputs.get("q_values")
                q_values = (
                    None
                    if q_values is None
                    else q_values.detach().float().cpu().numpy()
                )
                stats["forward_and_d2h_sec"] += time.perf_counter() - forward_started
                stats["forward_calls"] += 1
                stats["max_forward_rows"] = max(
                    int(stats["max_forward_rows"]), forward_rows
                )
                forward_histogram = stats["forward_row_histogram"]
                forward_histogram[forward_rows] = (
                    int(forward_histogram.get(forward_rows, 0)) + 1
                )

                group_max_legal = int(legal_ids.shape[1])
                padded_legal_cells += forward_rows * group_max_legal
                offset = 0
                for request_index, n_rows, wants_q, payload in zip(
                    group_request_indices,
                    row_counts,
                    group_return_q,
                    group_payloads,
                ):
                    sl = slice(offset, offset + n_rows)
                    offset += n_rows
                    legal_width = int(payload["legal_ids"].shape[1])
                    parts = result_parts[request_index]
                    parts.setdefault("logits", []).append(
                        logits[sl, :legal_width].copy()
                    )
                    parts.setdefault("value", []).append(value[sl].copy())
                    if value_categorical is not None:
                        parts.setdefault("value_categorical", []).append(
                            value_categorical[sl].copy()
                        )
                    if vu is not None:
                        parts.setdefault("value_uncertainty", []).append(vu[sl].copy())
                    if wants_q:
                        if q_values is None:
                            raise RuntimeError(
                                "EvalServer policy omitted q_values for return_q request"
                            )
                        parts.setdefault("q_values", []).append(
                            q_values[sl, :legal_width].copy()
                        )

            with collector.paused() if collector is not None else nullcontext():
                response_started = time.perf_counter()
                total_rows = 0
                completed_responses: list[tuple[int, int, dict[str, np.ndarray]]] = []
                for (
                    (client_id, req_id),
                    wants_q,
                    payload,
                    parts,
                ) in zip(ids, return_q_flags, payloads, result_parts):
                    n_rows = int(payload["legal_ids"].shape[0])
                    total_rows += n_rows
                    result = {
                        key: chunks[0]
                        if len(chunks) == 1
                        else np.concatenate(chunks, axis=0)
                        for key, chunks in parts.items()
                    }
                    if int(result["value"].shape[0]) != n_rows:
                        raise RuntimeError(
                            "EvalServer response reassembly row mismatch: "
                            f"got={result['value'].shape[0]} expected={n_rows}"
                        )
                    if wants_q and "q_values" not in result:
                        raise RuntimeError("EvalServer omitted reassembled q_values")
                    completed_responses.append((client_id, req_id, result))
                stats["windows"] += 1
                stats["requests"] += len(window)
                stats["rows"] += int(total_rows)
                stats["oversized_requests"] += int(oversized_requests)
                stats["oversized_request_chunks"] += int(oversized_chunks)
                shared_requests = sum(
                    payload.get("_transport") == "shared_memory" for payload in payloads
                )
                stats["shared_memory_requests"] += int(shared_requests)
                stats["shared_memory_request_bytes"] += sum(
                    int(payload.get("_transport_bytes", 0)) for payload in payloads
                )
                stats["queue_payload_requests"] += len(window) - int(shared_requests)
                stats["max_window_requests"] = max(
                    stats["max_window_requests"], len(window)
                )
                stats["max_window_rows"] = max(
                    stats["max_window_rows"], int(total_rows)
                )
                max_legal = max(
                    int(payload["legal_ids"].shape[1]) for payload in payloads
                )
                real_legal_cells, request_legal_cells = _legal_cell_counts(payloads)
                stats["real_legal_cells"] += real_legal_cells
                stats["request_legal_cells"] += request_legal_cells
                stats["padded_legal_cells"] += int(padded_legal_cells)
                required_event_width = max(
                    required_event_widths or [int(premerge_required_event_width or 0)]
                )
                event_histogram = stats["event_required_width_histogram"]
                event_histogram[required_event_width] = (
                    int(event_histogram.get(required_event_width, 0)) + 1
                )
                for histogram_name, key in (
                    ("window_request_histogram", len(window)),
                    ("window_row_histogram", int(total_rows)),
                    ("window_legal_width_histogram", max_legal),
                ):
                    histogram = stats[histogram_name]
                    histogram[key] = int(histogram.get(key, 0)) + 1
                # Do not publish any partial success until every request has
                # reassembled and all window telemetry inputs have validated.
                for client_id, req_id, result in completed_responses:
                    response_queues[client_id].put((req_id, result, None))
                stats["response_enqueue_sec"] += time.perf_counter() - response_started
        except BaseException as error:  # pragma: no cover - surfaced to clients
            with collector.paused() if collector is not None else nullcontext():
                for client_id, req_id in ids:
                    response_queues[client_id].put((req_id, None, repr(error)))

    if collector is not None:
        collector.close(timeout=1.0)
    if cuda_memory_device is not None:
        stats["cuda_peak_memory_allocated_bytes"] = int(
            cuda_api.max_memory_allocated(cuda_memory_device)
        )
        stats["cuda_peak_memory_reserved_bytes"] = int(
            cuda_api.max_memory_reserved(cuda_memory_device)
        )
    # Compatibility alias retained for older telemetry consumers.
    stats["sum_batch"] = int(stats["rows"])
    handshake["stats"] = dict(stats)


class EvalServer:
    """Single-process inference service holding one GPU/CPU-resident policy and
    servicing packed leaf requests from many game-worker processes. Start it,
    wait for `ready()`, hand `request_queue_for_client(i)` + the corresponding
    `response_queues[i]` to each `RemoteEvalClient`, then `stop()` when done.
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
        transport = str(self.config.transport)
        if transport == "mp_queue":
            self.request_queue: Any = self._ctx.Queue()
            self.request_queues: list[Any] = [self.request_queue] * self.num_clients
        elif transport == "shared_memory":
            self.request_queue, self.request_queues = _make_shared_request_transport(
                self._ctx,
                self.num_clients,
                int(self.config.shared_memory_slot_bytes),
            )
        else:
            raise ValueError(
                f"unsupported EvalServer transport {transport!r}; "
                "expected 'mp_queue' or 'shared_memory'"
            )
        self.response_queues: list[mp.Queue] = [
            self._ctx.Queue() for _ in range(self.num_clients)
        ]
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
        self._stopped = False
        self._last_stats: dict[str, Any] = {}

    def start(self) -> None:
        self._proc.start()

    def request_queue_for_client(self, client_id: int) -> Any:
        """Return the smallest spawn payload for one RemoteEvalClient.

        Callers should prefer this over passing ``request_queue`` directly. For
        ``mp_queue`` both are the same object; shared-memory mode returns only
        that client's slot instead of all client slots.
        """
        client_id = int(client_id)
        if not 0 <= client_id < self.num_clients:
            raise IndexError(f"client id {client_id} outside [0, {self.num_clients})")
        return self.request_queues[client_id]

    def wait_ready(self, timeout: float = 120.0) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.0, float(timeout))
        while not self._ready.wait(
            timeout=min(0.1, max(0.0, deadline - time.monotonic()))
        ):
            if self._proc.exitcode is not None:
                raise RuntimeError(
                    f"eval server exited before ready (exitcode={self._proc.exitcode})"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("eval server did not become ready")
        return {
            "action_size": int(self._handshake["action_size"]),
            "entity_feature_adapter": str(
                self._handshake["entity_feature_adapter"]
            ),
            "trained_with_masked_hidden_info": bool(
                self._handshake["trained_with_masked_hidden_info"]
            ),
            "needs_action_targets": bool(
                self._handshake.get("needs_action_targets", True)
            ),
            "needs_relational_topology": bool(
                self._handshake.get("needs_relational_topology", False)
            ),
            "matmul_precision": str(self._handshake.get("matmul_precision", "highest")),
            "transport": str(self._handshake.get("transport", "mp_queue")),
            "max_neural_rows": self._handshake.get("max_neural_rows"),
            "event_token_limit": self._handshake.get("event_token_limit"),
            "cuda_graph": bool(self._handshake.get("cuda_graph", False)),
            "cuda_graph_batch_buckets": tuple(
                self._handshake.get("cuda_graph_batch_buckets", ())
            ),
            "cuda_graph_warmup_iterations": int(
                self._handshake.get("cuda_graph_warmup_iterations", 0)
            ),
            "value_categorical_bins": int(self._handshake["value_categorical_bins"]),
            "value_categorical_head_available": bool(
                self._handshake["value_categorical_head_available"]
            ),
        }

    @property
    def exitcode(self) -> int | None:
        """Current server-process exit code, or ``None`` while it is running."""
        return self._proc.exitcode

    def stop(self) -> dict[str, Any]:
        if self._stopped:
            return dict(self._last_stats)
        self._stopped = True
        try:
            proc_started = self._proc.pid is not None
            if proc_started and self._proc.is_alive():
                try:
                    self.request_queue.put(_STOP)
                except Exception:
                    pass
                self._proc.join(timeout=10.0)
            try:
                self._last_stats = (
                    dict(self._handshake.get("stats", {})) if self._handshake else {}
                )
            except (EOFError, BrokenPipeError, OSError):
                self._last_stats = {}
            if proc_started and self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=5.0)
            if proc_started and self._proc.is_alive() and hasattr(self._proc, "kill"):
                self._proc.kill()
                self._proc.join(timeout=5.0)
            return dict(self._last_stats)
        finally:
            # The server and every worker are joined before this point in the
            # production launcher, so no process can still legitimately use
            # these handles. Closing them prevents repeated canary arms from
            # retaining Queue pipes/feeder threads for the life of the parent.
            for ipc_queue in (
                self.request_queue,
                *getattr(self, "response_queues", ()),
            ):
                close = getattr(ipc_queue, "close", None)
                if close is not None:
                    try:
                        close()
                    except (EOFError, OSError, ValueError):
                        pass
                join_thread = getattr(ipc_queue, "join_thread", None)
                if join_thread is not None:
                    try:
                        join_thread()
                    except (AssertionError, EOFError, OSError, ValueError):
                        pass
            try:
                self._manager.shutdown()
            except (EOFError, BrokenPipeError, OSError):
                pass


# --- client ------------------------------------------------------------------


class _RemoteForwardProxy:
    """Stands in for `EntityGraphPolicy` inside `RemoteEvalClient`. Exposes only
    what `EntityGraphRustEvaluator` reads off `self.policy`: `action_size`, the
    `trained_with_masked_hidden_info` flag (for the #76 safety-net assert), and
    `forward_legal_np` (routed to the server). Everything else the base
    evaluator needs (featurize, softmax, squash, clip) never touches the policy.
    """

    def __init__(
        self,
        client: "RemoteEvalClient",
        action_size: int,
        trained_masked: bool,
        *,
        entity_feature_adapter: str,
        value_categorical_bins: int = 0,
        value_categorical_head_available: bool = False,
    ) -> None:
        self._client = client
        self.action_size = int(action_size)
        self.trained_with_masked_hidden_info = bool(trained_masked)
        self.entity_feature_adapter_version = (
            _require_implemented_entity_feature_adapter(
                entity_feature_adapter,
                context="RemoteEvalClient handshake",
            )
        )
        self.entity_feature_adapter_binding_source = "eval_server_handshake"
        self.trained_value_readouts = (
            ("scalar", "categorical")
            if value_categorical_head_available
            else ("scalar",)
        )
        self.model = type(
            "_RemoteModelMetadata",
            (),
            {
                "value_categorical_bins": int(value_categorical_bins),
                "value_categorical_head": (
                    object() if value_categorical_head_available else None
                ),
            },
        )()

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
        entity_feature_adapter: str,
        needs_action_targets: bool = True,
        needs_relational_topology: bool = False,
        event_token_limit: int | None = None,
        value_categorical_bins: int = 0,
        value_categorical_head_available: bool = False,
        config: EntityGraphRustEvaluatorConfig | None = None,
        client_timeout_ms: float = 5000.0,
        fallback_checkpoint: str | None = None,
        fallback_device: str = "cpu",
    ) -> None:
        proxy = _RemoteForwardProxy(
            self,
            action_size,
            trained_with_masked_hidden_info,
            entity_feature_adapter=entity_feature_adapter,
            value_categorical_bins=value_categorical_bins,
            value_categorical_head_available=value_categorical_head_available,
        )
        super().__init__(proxy, config=config)
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._client_id = int(client_id)
        self._needs_action_targets = bool(needs_action_targets)
        self._needs_relational_topology = bool(needs_relational_topology)
        self._event_token_limit = event_token_limit
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
        # With no local fallback, a timed-out/failed single-outstanding request
        # leaves this transport unusable (and may leave a stale response queued).
        # Latch the first terminal failure so per-game exception isolation cannot
        # turn one hung server into one full client timeout per remaining game.
        self._terminal_failure: str | None = None

    def _ensure_local_policy(self) -> Any:
        if self._local_policy is None:
            from catan_zero.rl.entity_token_policy import EntityGraphPolicy

            self._local_policy = EntityGraphPolicy.load(
                self._fallback_checkpoint, device=self._fallback_device
            )
            _require_implemented_entity_feature_adapter(
                policy_entity_feature_adapter_version(self._local_policy),
                context="RemoteEvalClient local fallback",
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

        forward_entity_batch = entity_batch
        event_source_stats: dict[str, int] = {}
        if self._event_token_limit is not None:
            # Keep the evaluator-owned feature batch untouched: local post-
            # processing and callers may retain it. Only the shallow mapping is
            # changed; event array prefixes are zero-copy NumPy views.
            forward_entity_batch = dict(entity_batch)
            event_mask = np.asarray(entity_batch["event_mask"], dtype=np.bool_)
            event_source_stats = {
                "_event_source_active_tokens": int(np.count_nonzero(event_mask)),
                "_event_source_padded_tokens": int(event_mask.size),
            }
            _crop_masked_event_tail(forward_entity_batch, self._event_token_limit)

        if self._degraded:
            return self._forward_local(
                forward_entity_batch,
                legal_action_ids,
                legal_action_context,
                return_q,
            )
        if self._terminal_failure is not None:
            raise TimeoutError(self._terminal_failure)

        self._req_counter += 1
        req_id = self._req_counter
        payload = {
            "entity": {
                k: np.asarray(v)
                for k, v in forward_entity_batch.items()
                if (
                    k not in _NON_FORWARD_ENTITY_KEYS
                    or (
                        self._needs_relational_topology
                        and k
                        in {
                            "hex_vertex_ids",
                            "hex_edge_ids",
                            "edge_vertex_ids",
                            "event_target_ids",
                        }
                    )
                )
                and (k != "legal_action_target_ids" or self._needs_action_targets)
            },
            "legal_ids": np.asarray(legal_action_ids),
            "context": np.asarray(legal_action_context),
            "return_q": bool(return_q),
            **event_source_stats,
        }
        try:
            self._request_queue.put((self._client_id, req_id, payload))
            got_id, result, error = self._response_queue.get(timeout=self._timeout_s)
            if got_id != req_id:  # pragma: no cover - single-outstanding invariant
                raise RuntimeError(f"response id mismatch: got {got_id} want {req_id}")
            request_complete = getattr(self._request_queue, "request_complete", None)
            if request_complete is not None:
                request_complete(self._client_id, req_id)
            if error is not None:
                raise RuntimeError(f"eval-server forward failed: {error}")
            return {
                k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in result.items()
            }
        except (queue_mod.Empty, RuntimeError, OSError, EOFError, ValueError) as exc:
            if self._fallback_checkpoint is None:
                self._terminal_failure = (
                    f"eval-server request failed (client {self._client_id}, req {req_id}); "
                    "no fallback checkpoint configured; client permanently failed"
                )
                raise TimeoutError(self._terminal_failure) from exc
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
                forward_entity_batch,
                legal_action_ids,
                legal_action_context,
                return_q,
            )
