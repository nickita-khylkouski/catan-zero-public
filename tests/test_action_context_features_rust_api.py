from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from catan_zero.rl.action_context_features_rust import (
    build_action_context_batch_rust,
    build_action_context_rust,
)
from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE
from catan_zero.rl.entity_feature_adapter import RUST_ENTITY_ADAPTER_V6


def _f64_bytes(shape: tuple[int, ...]) -> bytes:
    return np.arange(np.prod(shape), dtype="<f8").reshape(shape).tobytes()


def test_single_wrapper_omitted_adapter_preserves_legacy_native_call(
    monkeypatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    shape = (2, CONTEXT_ACTION_FEATURE_SIZE)
    native = SimpleNamespace(
        build_action_context_flat=lambda *args: (
            calls.append(args) or (_f64_bytes(shape), shape)
        )
    )
    monkeypatch.setitem(sys.modules, "catanatron_rs", native)
    game = object()
    rust_topology = object()

    result = build_action_context_rust(
        game,
        topology=SimpleNamespace(rust=rust_topology),
    )

    assert calls == [(game, rust_topology)]
    assert result.dtype == np.float32
    assert result.shape == shape


def test_batch_wrapper_omitted_adapter_preserves_legacy_native_call(
    monkeypatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    shape = (2, 3, CONTEXT_ACTION_FEATURE_SIZE)

    def build_batch(*args):
        calls.append(args)
        return {
            "widths": [2, 3],
            "context_tokens": (_f64_bytes(shape), shape),
        }

    native = SimpleNamespace(build_action_context_batch=build_batch)
    monkeypatch.setitem(sys.modules, "catanatron_rs", native)
    games = [object(), object()]
    rust_topology = object()

    result, widths = build_action_context_batch_rust(
        games,
        topology=SimpleNamespace(rust=rust_topology),
        parallel=True,
    )

    assert calls == [(games, rust_topology, True)]
    assert widths == [2, 3]
    assert result.dtype == np.float32
    assert result.shape == shape


def test_explicit_v6_still_threads_adapter_through_single_and_batch(
    monkeypatch,
) -> None:
    flat_calls: list[tuple[object, ...]] = []
    batch_calls: list[tuple[object, ...]] = []
    single_shape = (1, CONTEXT_ACTION_FEATURE_SIZE)
    batch_shape = (2, 1, CONTEXT_ACTION_FEATURE_SIZE)

    def build_flat(*args):
        flat_calls.append(args)
        return _f64_bytes(single_shape), single_shape

    def build_batch(*args):
        batch_calls.append(args)
        return {
            "widths": [1, 1],
            "context_tokens": (_f64_bytes(batch_shape), batch_shape),
        }

    native = SimpleNamespace(
        supported_action_context_adapter_versions=lambda: [RUST_ENTITY_ADAPTER_V6],
        build_action_context_flat=build_flat,
        build_action_context_batch=build_batch,
    )
    monkeypatch.setitem(sys.modules, "catanatron_rs", native)
    game = object()
    games = [game, object()]
    rust_topology = object()
    topology = SimpleNamespace(rust=rust_topology)

    build_action_context_rust(
        game,
        topology=topology,
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6,
    )
    build_action_context_batch_rust(
        games,
        topology=topology,
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6,
        parallel=True,
    )

    assert flat_calls == [(game, rust_topology, RUST_ENTITY_ADAPTER_V6)]
    assert batch_calls == [
        (games, rust_topology, True, RUST_ENTITY_ADAPTER_V6)
    ]
