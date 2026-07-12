from __future__ import annotations

import json

import numpy as np
import pytest

from tools import train_bc


def _identity() -> dict[str, object]:
    _key, identity = train_bc._derived_array_cache_key(
        {
            "data_fingerprint": "sha256:" + "a" * 64,
            "row_count": 6,
            "recipe": {"per_game_value_weight": True},
        }
    )
    return identity


def test_derived_array_cache_round_trip_is_exact_read_only_mmap(tmp_path):
    arrays = {
        "train_indices": np.asarray([5, 1, 4, 2], dtype=np.int64),
        "policy_sample_weights": np.asarray([0.0, 0.5, 2.0], dtype=np.float32),
    }
    identity = _identity()
    directory = train_bc._write_derived_array_cache(tmp_path, identity, arrays)
    loaded = train_bc._load_derived_array_cache(tmp_path, identity)

    assert directory.name == train_bc._canonical_json_sha256(identity).split(":", 1)[1]
    assert set(loaded) == set(arrays)
    for name, expected in arrays.items():
        assert isinstance(loaded[name], np.memmap)
        assert loaded[name].flags.writeable is False
        assert loaded[name].dtype == expected.dtype
        assert np.array_equal(loaded[name], expected)


def test_derived_array_cache_fails_closed_on_payload_tamper(tmp_path):
    identity = _identity()
    directory = train_bc._write_derived_array_cache(
        tmp_path, identity, {"train_indices": np.arange(8, dtype=np.int64)}
    )
    payload = directory / "train_indices.npy"
    payload.chmod(0o644)
    with open(payload, "r+b") as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([byte[0] ^ 0x01]))

    with pytest.raises(SystemExit, match="file digest mismatch"):
        train_bc._load_derived_array_cache(tmp_path, identity)


def test_derived_array_cache_fails_closed_on_inventory_tamper(tmp_path):
    identity = _identity()
    directory = train_bc._write_derived_array_cache(
        tmp_path, identity, {"train_indices": np.arange(3, dtype=np.int64)}
    )
    manifest_path = directory / "manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    manifest["arrays"]["train_indices"]["shape"] = [99]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SystemExit, match="shape/dtype drift"):
        train_bc._load_derived_array_cache(tmp_path, identity)


def test_weighted_epoch_cap_is_exact_historical_prefix():
    n = 10_000
    weights = np.linspace(0.1, 2.0, n, dtype=np.float64)
    ddp = {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
    full = train_bc._epoch_order(
        np.random.default_rng(123), n, 512, ddp, sample_weights=weights
    )
    capped = train_bc._epoch_order(
        np.random.default_rng(123),
        n,
        512,
        ddp,
        sample_weights=weights,
        max_samples=2_048,
    )
    assert np.array_equal(capped, full[:2_048])


def test_uniform_epoch_cap_is_ignored_to_preserve_permutation_semantics():
    ddp = {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
    uncapped = train_bc._epoch_order(np.random.default_rng(7), 100, 8, ddp)
    requested_cap = train_bc._epoch_order(
        np.random.default_rng(7), 100, 8, ddp, max_samples=10
    )
    assert len(requested_cap) == 100
    assert np.array_equal(requested_cap, uncapped)

