from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = _ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import train_bc  # type: ignore  # noqa: E402


class _Store:
    def __init__(self, *, max_value_bytes: int | None = None) -> None:
        self.values: dict[str, bytes] = {}
        self.max_value_bytes = max_value_bytes

    def set(self, key: str, value: str | bytes) -> None:
        encoded = value.encode("utf-8") if isinstance(value, str) else value
        if self.max_value_bytes is not None and len(encoded) > self.max_value_bytes:
            raise ValueError("store value exceeds transport limit")
        self.values[key] = encoded

    def get(self, key: str) -> bytes:
        return self.values[key]


def _ddp(rank: int, world_size: int = 8) -> dict[str, int | bool]:
    return {
        "enabled": world_size > 1,
        "world_size": world_size,
        "rank": rank,
        "local_rank": rank,
    }


def test_single_node_ddp_rank0_verifies_once_and_peers_reuse_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    calls: list[tuple[object, object]] = []
    metadata = {
        "payload_inventory_sha256": "sha256:" + "a" * 64,
        "selected_game_seed_manifest": {},
        "a1_post_wave_audit": {},
    }

    def verify(data_path: object, *, validation_manifest_path: object) -> dict:
        calls.append((data_path, validation_manifest_path))
        return metadata

    monkeypatch.setattr(train_bc, "_preflight_a1_memmap_metadata", verify)
    store = _Store()
    rank0 = train_bc._coordinated_a1_memmap_preflight(
        "/corpus",
        validation_manifest_path="/validation.json",
        ddp=_ddp(0),
        _store=store,
    )
    rank7 = train_bc._coordinated_a1_memmap_preflight(
        "/corpus",
        validation_manifest_path="/validation.json",
        ddp=_ddp(7),
        _store=store,
    )

    assert calls == [("/corpus", "/validation.json")]
    assert rank0 == metadata
    assert rank7 == metadata


def test_single_node_ddp_rank0_failure_is_published_and_peers_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")

    def reject(*args: object, **kwargs: object) -> None:
        raise SystemExit("game_seed.dat sha256 mismatch")

    monkeypatch.setattr(train_bc, "_preflight_a1_memmap_metadata", reject)
    store = _Store()
    with pytest.raises(SystemExit, match=r"game_seed\.dat sha256 mismatch"):
        train_bc._coordinated_a1_memmap_preflight(
            "/corpus",
            validation_manifest_path="/validation.json",
            ddp=_ddp(0),
            _store=store,
        )
    with pytest.raises(
        SystemExit,
        match=r"rank-0 A1 preflight failed: SystemExit: game_seed\.dat sha256 mismatch",
    ):
        train_bc._coordinated_a1_memmap_preflight(
            "/corpus",
            validation_manifest_path="/validation.json",
            ddp=_ddp(3),
            _store=store,
        )


@pytest.mark.parametrize(
    ("local_world_size", "world_size"),
    [("", 1), ("", 8), ("4", 8)],
)
def test_single_rank_unknown_topology_and_multinode_keep_per_rank_verification(
    monkeypatch: pytest.MonkeyPatch,
    local_world_size: str,
    world_size: int,
) -> None:
    if local_world_size:
        monkeypatch.setenv("LOCAL_WORLD_SIZE", local_world_size)
    else:
        monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    calls = 0

    def verify(*args: object, **kwargs: object) -> dict:
        nonlocal calls
        calls += 1
        return {"rank_call": calls}

    monkeypatch.setattr(train_bc, "_preflight_a1_memmap_metadata", verify)
    first = train_bc._coordinated_a1_memmap_preflight(
        "/corpus",
        validation_manifest_path="/validation.json",
        ddp=_ddp(0, world_size),
    )
    second_rank = 0 if world_size == 1 else world_size - 1
    second = train_bc._coordinated_a1_memmap_preflight(
        "/corpus",
        validation_manifest_path="/validation.json",
        ddp=_ddp(second_rank, world_size),
    )

    assert calls == 2
    assert first != second


def test_peer_rejects_malformed_success_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    store = _Store()
    train_bc._publish_a1_preflight_packet(
        store,
        "result",
        {"schema_version": 1, "ok": True, "metadata": "not-an-object"},
    )
    with pytest.raises(SystemExit, match="malformed metadata"):
        train_bc._coordinated_a1_memmap_preflight(
            "/corpus",
            validation_manifest_path="/validation.json",
            ddp=_ddp(1),
            _store=store,
        )


def test_large_success_metadata_is_chunked_below_tcpstore_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    metadata = {
        "payload_inventory_sha256": "sha256:" + "a" * 64,
        "large_authenticated_metadata": "x" * (9 * 1024 * 1024),
    }
    monkeypatch.setattr(
        train_bc,
        "_preflight_a1_memmap_metadata",
        lambda *args, **kwargs: metadata,
    )
    store = _Store(max_value_bytes=8 * 1024 * 1024)

    rank0 = train_bc._coordinated_a1_memmap_preflight(
        "/corpus",
        validation_manifest_path="/validation.json",
        ddp=_ddp(0),
        _store=store,
    )
    peer = train_bc._coordinated_a1_memmap_preflight(
        "/corpus",
        validation_manifest_path="/validation.json",
        ddp=_ddp(4),
        _store=store,
    )

    assert rank0 == metadata
    assert peer == metadata
    chunk_values = [value for key, value in store.values.items() if "/chunk/" in key]
    assert len(chunk_values) >= 3
    assert max(map(len, chunk_values)) <= 4 * 1024 * 1024


def test_peer_rejects_corrupt_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    store = _Store()
    train_bc._publish_a1_preflight_packet(
        store,
        "result",
        {"schema_version": 1, "ok": True, "metadata": {"bound": True}},
    )
    chunk_key = "result/chunk/00000000"
    store.values[chunk_key] = b"!" + store.values[chunk_key][1:]

    with pytest.raises(SystemExit, match="packet digest drifted"):
        train_bc._coordinated_a1_memmap_preflight(
            "/corpus",
            validation_manifest_path="/validation.json",
            ddp=_ddp(2),
            _store=store,
        )


def test_peer_rejects_malformed_packet_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    store = _Store()
    store.set(
        "result/manifest",
        json.dumps(
            {
                "schema_version": train_bc.A1_PREFLIGHT_STORE_PACKET_SCHEMA,
                "chunk_bytes": train_bc.A1_PREFLIGHT_STORE_CHUNK_BYTES,
                "chunk_count": 0,
                "payload_size_bytes": 1,
                "payload_sha256": "sha256:" + "0" * 64,
            }
        ),
    )

    with pytest.raises(SystemExit, match="packet manifest is malformed"):
        train_bc._coordinated_a1_memmap_preflight(
            "/corpus",
            validation_manifest_path="/validation.json",
            ddp=_ddp(6),
            _store=store,
        )


def test_peer_rejects_oversized_packet_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    store = _Store()
    store.set(
        "result/manifest",
        json.dumps(
            {
                "schema_version": train_bc.A1_PREFLIGHT_STORE_PACKET_SCHEMA,
                "chunk_bytes": train_bc.A1_PREFLIGHT_STORE_CHUNK_BYTES,
                "chunk_count": train_bc.A1_PREFLIGHT_STORE_MAX_CHUNKS + 1,
                "payload_size_bytes": (
                    train_bc.A1_PREFLIGHT_STORE_MAX_PACKET_BYTES + 1
                ),
                "payload_sha256": "sha256:" + "0" * 64,
            }
        ),
    )

    with pytest.raises(SystemExit, match="packet manifest is malformed"):
        train_bc._coordinated_a1_memmap_preflight(
            "/corpus",
            validation_manifest_path="/validation.json",
            ddp=_ddp(5),
            _store=store,
        )
