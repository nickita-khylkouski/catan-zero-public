from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = _ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import train_bc  # type: ignore  # noqa: E402


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _corpus(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "row_offsets.dat").write_bytes(b"offsets-v1")
    (root / "game_seed.dat").write_bytes(b"seeds-v001")
    inventory = [
        {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": _sha(path)}
        for path in sorted(root.glob("*.dat"))
    ]
    meta: dict[str, object] = {
        "columns": {"game_seed": {"kind": "fixed"}},
        "payload_inventory_schema": train_bc.MEMMAP_PAYLOAD_INVENTORY_SCHEMA,
        "payload_inventory": inventory,
        "payload_inventory_sha256": train_bc._canonical_json_sha256(inventory),
    }
    (root / "corpus_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for path in root.glob("*.dat"):
        path.chmod(0o444)
    return root, meta


def _count_payload_hashes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    original = train_bc._sha256_stable_payload

    def counted(path: str | Path, expected_identity: dict[str, int | str]) -> str:
        path = Path(path)
        calls.append(path.name)
        return original(path, expected_identity)

    monkeypatch.setattr(train_bc, "_sha256_stable_payload", counted)
    return calls


def test_second_authentication_hits_identity_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, meta = _corpus(tmp_path)
    monkeypatch.setenv("TRAIN_BC_PAYLOAD_AUTH_CACHE_DIR", str(tmp_path / "cache"))
    calls = _count_payload_hashes(monkeypatch)

    train_bc._validate_memmap_payload_inventory(root, meta)
    assert sorted(calls) == ["game_seed.dat", "row_offsets.dat"]
    calls.clear()
    train_bc._validate_memmap_payload_inventory(root, meta)

    assert calls == []
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["status"] for event in events] == ["miss", "hit"]
    assert events[-1]["bytes_avoided"] == sum(path.stat().st_size for path in root.glob("*.dat"))


def test_same_size_payload_tamper_invalidates_cache_and_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, meta = _corpus(tmp_path)
    monkeypatch.setenv("TRAIN_BC_PAYLOAD_AUTH_CACHE_DIR", str(tmp_path / "cache"))
    train_bc._validate_memmap_payload_inventory(root, meta)

    payload = root / "game_seed.dat"
    payload.chmod(0o644)
    payload.write_bytes(b"TAMPER-v01")
    payload.chmod(0o444)
    with pytest.raises(SystemExit, match=r"game_seed\.dat sha256 mismatch"):
        train_bc._validate_memmap_payload_inventory(root, meta)


def test_mtime_drift_forces_rehash_even_when_bytes_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, meta = _corpus(tmp_path)
    monkeypatch.setenv("TRAIN_BC_PAYLOAD_AUTH_CACHE_DIR", str(tmp_path / "cache"))
    train_bc._validate_memmap_payload_inventory(root, meta)
    calls = _count_payload_hashes(monkeypatch)

    payload = root / "game_seed.dat"
    stat_before = payload.stat()
    os.utime(payload, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns + 1_000_000))
    train_bc._validate_memmap_payload_inventory(root, meta)

    assert sorted(calls) == ["game_seed.dat", "row_offsets.dat"]


def test_inode_replacement_forces_rehash_even_with_identical_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, meta = _corpus(tmp_path)
    monkeypatch.setenv("TRAIN_BC_PAYLOAD_AUTH_CACHE_DIR", str(tmp_path / "cache"))
    train_bc._validate_memmap_payload_inventory(root, meta)
    calls = _count_payload_hashes(monkeypatch)

    payload = root / "game_seed.dat"
    original = payload.read_bytes()
    old_inode = payload.stat().st_ino
    replacement = root / "replacement"
    replacement.write_bytes(original)
    replacement.chmod(0o444)
    os.replace(replacement, payload)
    assert payload.stat().st_ino != old_inode
    train_bc._validate_memmap_payload_inventory(root, meta)

    assert sorted(calls) == ["game_seed.dat", "row_offsets.dat"]


@pytest.mark.parametrize("cache_bytes", [b"", b'{"binding":'])
def test_partial_cache_is_a_miss_and_is_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_bytes: bytes
) -> None:
    root, meta = _corpus(tmp_path)
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("TRAIN_BC_PAYLOAD_AUTH_CACHE_DIR", str(cache_root))
    train_bc._validate_memmap_payload_inventory(root, meta)
    cache_path = next(cache_root.glob("*.json"))
    cache_path.chmod(0o600)
    cache_path.write_bytes(cache_bytes)
    calls = _count_payload_hashes(monkeypatch)

    train_bc._validate_memmap_payload_inventory(root, meta)

    assert sorted(calls) == ["game_seed.dat", "row_offsets.dat"]
    repaired = json.loads(cache_path.read_text(encoding="utf-8"))
    assert repaired["entry_sha256"].startswith("sha256:")


def test_cache_entry_tamper_is_a_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, meta = _corpus(tmp_path)
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("TRAIN_BC_PAYLOAD_AUTH_CACHE_DIR", str(cache_root))
    train_bc._validate_memmap_payload_inventory(root, meta)
    cache_path = next(cache_root.glob("*.json"))
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    cached["authenticated_bytes"] += 1
    cache_path.write_text(json.dumps(cached), encoding="utf-8")
    calls = _count_payload_hashes(monkeypatch)

    train_bc._validate_memmap_payload_inventory(root, meta)

    assert sorted(calls) == ["game_seed.dat", "row_offsets.dat"]


def test_validator_version_change_invalidates_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, meta = _corpus(tmp_path)
    monkeypatch.setenv("TRAIN_BC_PAYLOAD_AUTH_CACHE_DIR", str(tmp_path / "cache"))
    train_bc._validate_memmap_payload_inventory(root, meta)
    calls = _count_payload_hashes(monkeypatch)
    monkeypatch.setattr(
        train_bc,
        "MEMMAP_PAYLOAD_AUTH_VALIDATOR_VERSION",
        train_bc.MEMMAP_PAYLOAD_AUTH_VALIDATOR_VERSION + 1,
    )

    train_bc._validate_memmap_payload_inventory(root, meta)

    assert sorted(calls) == ["game_seed.dat", "row_offsets.dat"]


def test_writable_payloads_never_use_or_publish_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, meta = _corpus(tmp_path)
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("TRAIN_BC_PAYLOAD_AUTH_CACHE_DIR", str(cache_root))
    for path in root.glob("*.dat"):
        path.chmod(0o644)
    calls = _count_payload_hashes(monkeypatch)

    train_bc._validate_memmap_payload_inventory(root, meta)
    train_bc._validate_memmap_payload_inventory(root, meta)

    assert len(calls) == 4
    assert not cache_root.exists() or not list(cache_root.glob("*.json"))
