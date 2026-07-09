"""Unit tests for `tools/opponent_mix_registry.py` (CAT-54 <-> CAT-9 bridge):
resolving "registry_role"/"registry_pool" mix categories against a real
on-disk `ChampionRegistry`. No rust engine / torch needed."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.champion_registry import ChampionRegistry
from tools.opponent_mix_registry import resolve_opponent_mix_manifest


def _make_registry(tmp_path: Path) -> tuple[Path, Path]:
    """A registry with one checkpoint file, set as public_champion AND
    appended to the pool tagged hard_negative -- exercises both registry_role
    and registry_pool resolution against real (if tiny) files."""
    checkpoint = tmp_path / "champion_v3.pt"
    checkpoint.write_bytes(b"fake-weights-v3")
    registry_path = tmp_path / "registry.json"
    registry = ChampionRegistry(registry_path)
    registry.set_role("public_champion", checkpoint, version=3)
    registry.append_pool(checkpoint, version=3, provenance={"tag": "hard_negative"}, status="active")
    registry.save()
    return registry_path, checkpoint


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    path = tmp_path / "mix.json"
    path.write_text(json.dumps(manifest))
    return path


def test_registry_role_resolves_to_one_checkpoint(tmp_path):
    registry_path, checkpoint = _make_registry(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        {
            "registry": str(registry_path),
            "categories": [
                {"name": "producer_self_play", "weight": 90, "source": "self"},
                {
                    "name": "previous_public_champion",
                    "weight": 10,
                    "source": "registry_role",
                    "role": "public_champion",
                },
            ],
        },
    )
    config = resolve_opponent_mix_manifest(manifest_path)
    by_name = {c.name: c for c in config.categories}
    resolved = by_name["previous_public_champion"]
    assert resolved.source == "checkpoint_list"
    assert not resolved.pending
    assert len(resolved.checkpoints) == 1
    assert resolved.checkpoints[0].path == str(checkpoint)
    assert resolved.checkpoints[0].version == 3
    assert resolved.checkpoints[0].md5  # non-empty, computed from real bytes


def test_registry_pool_filters_by_provenance_tag(tmp_path):
    registry_path, checkpoint = _make_registry(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        {
            "registry": str(registry_path),
            "categories": [
                {"name": "producer_self_play", "weight": 95, "source": "self"},
                {
                    "name": "hard_experimental",
                    "weight": 5,
                    "source": "registry_pool",
                    "filter": {"tag": "hard_negative"},
                },
            ],
        },
    )
    config = resolve_opponent_mix_manifest(manifest_path)
    by_name = {c.name: c for c in config.categories}
    resolved = by_name["hard_experimental"]
    assert len(resolved.checkpoints) == 1
    assert resolved.checkpoints[0].path == str(checkpoint)


def test_registry_pool_filters_by_status(tmp_path):
    checkpoint_a = tmp_path / "a.pt"
    checkpoint_a.write_bytes(b"a")
    checkpoint_b = tmp_path / "b.pt"
    checkpoint_b.write_bytes(b"b")
    registry_path = tmp_path / "registry.json"
    registry = ChampionRegistry(registry_path)
    registry.append_pool(checkpoint_a, version=0, status="active")
    registry.append_pool(checkpoint_b, version=1, status="regressed")
    registry.save()

    manifest_path = _write_manifest(
        tmp_path,
        {
            "registry": str(registry_path),
            "categories": [
                {"name": "producer_self_play", "weight": 95, "source": "self"},
                {
                    "name": "older_champion",
                    "weight": 5,
                    "source": "registry_pool",
                    "filter": {"status": "active"},
                },
            ],
        },
    )
    config = resolve_opponent_mix_manifest(manifest_path)
    resolved = {c.name: c for c in config.categories}["older_champion"]
    assert [ck.path for ck in resolved.checkpoints] == [str(checkpoint_a)]


def test_registry_role_with_no_pointer_set_requires_pending(tmp_path):
    registry_path = tmp_path / "registry.json"
    ChampionRegistry(registry_path).save()  # empty registry, no roles set

    manifest_path = _write_manifest(
        tmp_path,
        {
            "registry": str(registry_path),
            "categories": [
                {"name": "producer_self_play", "weight": 95, "source": "self"},
                {
                    "name": "previous_public_champion",
                    "weight": 5,
                    "source": "registry_role",
                    "role": "public_champion",
                },
            ],
        },
    )
    with pytest.raises(ValueError, match="pending"):
        resolve_opponent_mix_manifest(manifest_path)

    # marking it pending instead resolves cleanly, to zero checkpoints.
    manifest_path_pending = _write_manifest(
        tmp_path,
        {
            "registry": str(registry_path),
            "categories": [
                {"name": "producer_self_play", "weight": 95, "source": "self"},
                {
                    "name": "previous_public_champion",
                    "weight": 5,
                    "source": "registry_role",
                    "role": "public_champion",
                    "pending": True,
                },
            ],
        },
    )
    config = resolve_opponent_mix_manifest(manifest_path_pending)
    resolved = {c.name: c for c in config.categories}["previous_public_champion"]
    assert resolved.pending
    assert resolved.checkpoints == ()
    assert "previous_public_champion" not in config.effective_weights()


def test_registry_pool_with_zero_matches_requires_pending(tmp_path):
    registry_path, _checkpoint = _make_registry(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        {
            "registry": str(registry_path),
            "categories": [
                {"name": "producer_self_play", "weight": 95, "source": "self"},
                {
                    "name": "hard_experimental",
                    "weight": 5,
                    "source": "registry_pool",
                    "filter": {"tag": "no_such_tag"},
                },
            ],
        },
    )
    with pytest.raises(ValueError, match="pending"):
        resolve_opponent_mix_manifest(manifest_path)


def test_registry_source_without_top_level_registry_path_is_an_error(tmp_path):
    manifest_path = _write_manifest(
        tmp_path,
        {
            "categories": [
                {"name": "producer_self_play", "weight": 95, "source": "self"},
                {
                    "name": "previous_public_champion",
                    "weight": 5,
                    "source": "registry_role",
                    "role": "public_champion",
                },
            ]
        },
    )
    with pytest.raises(ValueError, match="registry"):
        resolve_opponent_mix_manifest(manifest_path)


def test_plain_sources_need_no_registry_at_all(tmp_path):
    """A manifest using only self/checkpoint_list/external_engine categories
    must resolve WITHOUT ever touching champion_registry -- this is what
    keeps generate_gumbel_selfplay_data.py's default (no --opponent-mix-
    manifest) and non-registry-mix paths import-safe even in environments
    where `tools.champion_registry`'s own repo-root-qualified import
    (`from tools.sprt_gate import score_to_elo`) would not resolve."""
    manifest_path = _write_manifest(
        tmp_path,
        {
            "categories": [
                {"name": "producer_self_play", "weight": 80, "source": "self"},
                {
                    "name": "hard_experimental",
                    "weight": 20,
                    "source": "checkpoint_list",
                    "checkpoints": [{"path": "/arch/exploiter.pt", "version": -1, "md5": "zzz"}],
                },
            ]
        },
    )
    config = resolve_opponent_mix_manifest(manifest_path)
    assert {c.name for c in config.categories} == {"producer_self_play", "hard_experimental"}
