from __future__ import annotations

import types

import numpy as np
import pytest

from tools.rnd_topology_eval_server_probe import (
    _measure_requests,
    _validate_handshake,
    _validate_live_legal_ids,
    _validate_server_stats,
    _validate_topology_inputs,
    build_parser,
    canonical_array_bundle_sha256,
    run,
)


class _ReplayClient:
    def __init__(self, *, drift: bool = False) -> None:
        self.calls = 0
        self.drift = drift

    def _remote_forward(self, entity, legal_ids, context, return_q):
        assert set(("hex_vertex_ids", "hex_edge_ids", "edge_vertex_ids")) <= set(entity)
        assert legal_ids.shape == (2, 3)
        assert context.shape == (2, 3, 4)
        assert return_q is False
        self.calls += 1
        delta = self.calls if self.drift else 0
        return {
            "logits": np.full((2, 3), delta, dtype=np.float32),
            "value": np.asarray([0.25, -0.5], dtype=np.float32),
        }


def _inputs():
    entity = {
        "hex_vertex_ids": np.zeros((2, 19, 6), dtype=np.int64),
        "hex_edge_ids": np.zeros((2, 19, 6), dtype=np.int64),
        "edge_vertex_ids": np.zeros((2, 72, 2), dtype=np.int64),
    }
    return entity, np.zeros((2, 3), dtype=np.int64), np.zeros((2, 3, 4), dtype=np.float32)


def test_hash_is_stable_and_binds_name_shape_dtype_and_content() -> None:
    base = {"x": np.asarray([[1, 2]], dtype=np.int32)}
    assert canonical_array_bundle_sha256(base) == canonical_array_bundle_sha256(base)
    assert canonical_array_bundle_sha256(base) != canonical_array_bundle_sha256(
        {"x": np.asarray([[1, 2]], dtype=np.int64)}
    )
    assert canonical_array_bundle_sha256(base) != canonical_array_bundle_sha256(
        {"y": np.asarray([[1, 2]], dtype=np.int32)}
    )


def test_measure_requests_reports_replay_and_latency() -> None:
    entity, legal_ids, context = _inputs()
    client = _ReplayClient()
    result = _measure_requests(
        client,
        entity=entity,
        legal_ids=legal_ids,
        context=context,
        warmup=2,
        iterations=3,
    )
    assert client.calls == 5
    assert result["rows"] == 6
    assert result["rows_per_sec"] > 0
    assert set(result["latency_ms"]) == {"p50", "p90", "p95", "p99"}
    assert result["replay_hashes_identical"] is True


def test_measure_requests_fails_closed_on_replay_drift() -> None:
    entity, legal_ids, context = _inputs()
    with pytest.raises(RuntimeError, match="non-identical outputs"):
        _measure_requests(
            _ReplayClient(drift=True),
            entity=entity,
            legal_ids=legal_ids,
            context=context,
            warmup=1,
            iterations=2,
        )


def test_topology_and_handshake_fail_closed() -> None:
    entity, _legal_ids, _context = _inputs()
    hashes = _validate_topology_inputs(entity)
    assert set(hashes) == {"hex_vertex_ids", "hex_edge_ids", "edge_vertex_ids"}
    with pytest.raises(RuntimeError, match="omitted required topology"):
        _validate_topology_inputs({})
    with pytest.raises(RuntimeError, match="non-topology checkpoint"):
        _validate_handshake({"needs_relational_topology": False}, cuda_graph=False)
    with pytest.raises(RuntimeError, match="disagrees"):
        _validate_handshake(
            {"needs_relational_topology": True, "cuda_graph": False}, cuda_graph=True
        )


def test_live_legal_ids_are_bounded_by_checkpoint_action_space() -> None:
    ids = np.asarray([[0, 7, -1], [3, 4, -1]], dtype=np.int64)
    mask = np.asarray([[True, True, False], [True, True, False]])
    _validate_live_legal_ids(ids, mask, action_size=8)

    with pytest.raises(RuntimeError, match=r"action_size=7.*invalid_examples=\[7\]"):
        _validate_live_legal_ids(ids, mask, action_size=7)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        _validate_live_legal_ids(ids, mask[:, :2], action_size=8)
    with pytest.raises(RuntimeError, match="no live legal actions"):
        _validate_live_legal_ids(ids, np.zeros_like(mask), action_size=8)


def test_cuda_graph_stats_require_capture_replay_without_fallback() -> None:
    good = {
        "requests": 4,
        "forward_calls": 4,
        "cuda_graph_calls": 4,
        "cuda_graph_graph_count": 1,
        "cuda_graph_fallbacks": 0,
    }
    _validate_server_stats(good, cuda_graph=True, expected_requests=4)
    for field, value, message in (
        ("cuda_graph_calls", 3, "not every"),
        ("cuda_graph_graph_count", 0, "did not capture"),
        ("cuda_graph_fallbacks", 1, "eager fallback"),
    ):
        broken = dict(good)
        broken[field] = value
        with pytest.raises(RuntimeError, match=message):
            _validate_server_stats(broken, cuda_graph=True, expected_requests=4)


def test_report_helpers_do_not_emit_host_or_ip_identity() -> None:
    module = __import__("tools.rnd_topology_eval_server_probe", fromlist=["run"])
    source = types.ModuleType.__repr__(module) + open(module.__file__, encoding="utf-8").read()
    assert "platform.node" not in source
    assert "socket.gethostname" not in source
    assert '"host"' not in source
    assert '"ip"' not in source


def test_cpu_probe_uses_actual_eval_server_transport(tmp_path, monkeypatch) -> None:
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.search import eval_server

    real_remote_client = eval_server.RemoteEvalClient
    captured_client_configs = []

    class _CapturingRemoteClient:
        def __new__(cls, *args, **kwargs):
            captured_client_configs.append(kwargs.get("config"))
            return real_remote_client(*args, **kwargs)

    monkeypatch.setattr(eval_server, "RemoteEvalClient", _CapturingRemoteClient)

    checkpoint = tmp_path / "tiny_topology.pt"
    policy = EntityGraphPolicy.create(
        hidden_size=16,
        state_layers=2,
        attention_heads=2,
        dropout=0.0,
        device="cpu",
        topology_adapter_layers="1",
        topology_adapter_width=8,
        topology_adapter_kind="local_attention_v2",
        topology_adapter_heads=2,
    )
    policy.save(checkpoint, mask_hidden_info=True)
    args = build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--legal-actions",
            "8",
            "--events",
            "8",
            "--warmup",
            "1",
            "--iterations",
            "2",
        ]
    )

    report = run(args)

    assert report["handshake"]["needs_relational_topology"] is True
    assert report["configuration"]["cuda_graph"] is False
    assert report["measurements"]["replay_hashes_identical"] is True
    assert report["server_stats"]["requests"] == 3
    assert report["server_stats"]["forward_calls"] == 3
    assert report["server_stats"]["queue_payload_requests"] == 3
    assert "path" not in report["checkpoint"]
    assert len(captured_client_configs) == 1
    assert captured_client_configs[0].public_observation is True
    assert captured_client_configs[0].cache_size == 0
