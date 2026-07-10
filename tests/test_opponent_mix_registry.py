"""Unit tests for `tools/opponent_mix_registry.py` (CAT-54 <-> CAT-9 bridge):
resolving "registry_role"/"registry_pool" mix categories against a real
on-disk `ChampionRegistry`. No rust engine / torch needed."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from tools.champion_registry import ChampionRegistry
from tools.opponent_mix_registry import (
    freeze_opponent_mix_manifest,
    main as opponent_mix_registry_main,
    resolve_opponent_mix_manifest,
)


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
    checkpoint = tmp_path / "hard-negative.pt"
    checkpoint.write_bytes(b"hard-negative")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "categories": [
                {"name": "producer_self_play", "weight": 80, "source": "self"},
                {
                    "name": "hard_experimental",
                    "weight": 20,
                    "source": "checkpoint_list",
                    "checkpoints": [{"path": str(checkpoint), "version": -1}],
                },
            ]
        },
    )
    config = resolve_opponent_mix_manifest(manifest_path)
    assert {c.name for c in config.categories} == {"producer_self_play", "hard_experimental"}
    resolved = next(c for c in config.categories if c.name == "hard_experimental")
    assert resolved.checkpoints[0].path == str(checkpoint.resolve())
    assert resolved.checkpoints[0].md5 == hashlib.md5(b"hard-negative").hexdigest()


def test_checkpoint_list_missing_file_fails_during_resolution(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    manifest_path = _write_manifest(
        tmp_path,
        {
            "categories": [
                {"name": "self", "weight": 90, "source": "self"},
                {
                    "name": "older",
                    "weight": 10,
                    "source": "checkpoint_list",
                    "checkpoints": [{"path": str(missing), "md5": "0" * 32}],
                },
            ]
        },
    )

    with pytest.raises(FileNotFoundError, match="older.*missing"):
        resolve_opponent_mix_manifest(manifest_path)


def test_checkpoint_list_md5_mismatch_fails_during_resolution(tmp_path: Path) -> None:
    checkpoint = tmp_path / "older.pt"
    checkpoint.write_bytes(b"actual")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "categories": [
                {"name": "self", "weight": 90, "source": "self"},
                {
                    "name": "older",
                    "weight": 10,
                    "source": "checkpoint_list",
                    "checkpoints": [{"path": str(checkpoint), "md5": hashlib.md5(b"stale").hexdigest()}],
                },
            ]
        },
    )

    with pytest.raises(ValueError, match="older.*md5 mismatch"):
        resolve_opponent_mix_manifest(manifest_path)


def test_registry_checkpoint_is_reverified_after_registry_write(tmp_path: Path) -> None:
    registry_path, checkpoint = _make_registry(tmp_path)
    checkpoint.write_bytes(b"corrupted-after-registration")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "registry": str(registry_path),
            "categories": [
                {"name": "self", "weight": 90, "source": "self"},
                {
                    "name": "previous",
                    "weight": 10,
                    "source": "registry_role",
                    "role": "public_champion",
                },
            ],
        },
    )

    with pytest.raises(ValueError, match="previous.*md5 mismatch"):
        resolve_opponent_mix_manifest(manifest_path)


def test_duplicate_checkpoint_bytes_across_categories_fail(tmp_path: Path) -> None:
    checkpoint_a = tmp_path / "a.pt"
    checkpoint_b = tmp_path / "b.pt"
    checkpoint_a.write_bytes(b"same-weights")
    checkpoint_b.write_bytes(b"same-weights")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "categories": [
                {"name": "self", "weight": 80, "source": "self"},
                {
                    "name": "older",
                    "weight": 10,
                    "source": "checkpoint_list",
                    "checkpoints": [{"path": str(checkpoint_a)}],
                },
                {
                    "name": "hard",
                    "weight": 10,
                    "source": "checkpoint_list",
                    "checkpoints": [{"path": str(checkpoint_b)}],
                },
            ]
        },
    )

    with pytest.raises(ValueError, match="duplicate checkpoint bytes.*older.*hard"):
        resolve_opponent_mix_manifest(manifest_path)


def test_producer_checkpoint_cannot_reappear_as_opponent(tmp_path: Path) -> None:
    producer = tmp_path / "producer.pt"
    producer.write_bytes(b"current-producer")
    alias = tmp_path / "producer-copy.pt"
    alias.write_bytes(producer.read_bytes())
    manifest_path = _write_manifest(
        tmp_path,
        {
            "categories": [
                {"name": "self", "weight": 90, "source": "self"},
                {
                    "name": "previous",
                    "weight": 10,
                    "source": "checkpoint_list",
                    "checkpoints": [{"path": str(alias)}],
                },
            ]
        },
    )

    with pytest.raises(ValueError, match="previous.*producer checkpoint"):
        resolve_opponent_mix_manifest(manifest_path, producer_checkpoint=producer)


def test_multiple_effective_self_categories_fail_loudly(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        {
            "categories": [
                {"name": "self_a", "weight": 75, "source": "self"},
                {"name": "self_b", "weight": 25, "source": "self"},
            ]
        },
    )

    with pytest.raises(ValueError, match="multiple effective self categories"):
        resolve_opponent_mix_manifest(manifest_path)


def test_freeze_writes_read_only_content_bound_manifest_and_refuses_overwrite(tmp_path: Path) -> None:
    producer = tmp_path / "producer.pt"
    producer.write_bytes(b"producer")
    older = tmp_path / "older.pt"
    older.write_bytes(b"older")
    source = _write_manifest(
        tmp_path,
        {
            "categories": [
                {"name": "producer_self_play", "weight": 87, "source": "self"},
                {
                    "name": "older_champion",
                    "weight": 10,
                    "source": "checkpoint_list",
                    "checkpoints": [{"path": str(older), "version": 1}],
                },
                {
                    "name": "catanatron_value",
                    "weight": 3,
                    "source": "external_engine",
                    "engine": "catanatron_value",
                },
            ]
        },
    )
    output = tmp_path / "run" / "opponent_mix.resolved.json"

    frozen = freeze_opponent_mix_manifest(
        source,
        output,
        producer_checkpoint=producer,
        external_fraction=0.03,
    )

    assert frozen == output
    payload = json.loads(output.read_text())
    assert payload["_frozen"]["producer_checkpoint"]["md5"] == hashlib.md5(b"producer").hexdigest()
    assert payload["_frozen"]["resolved_config_sha256"]
    assert not (os.stat(output).st_mode & 0o222)
    round_tripped = resolve_opponent_mix_manifest(output, producer_checkpoint=producer)
    assert round_tripped.effective_weights()["catanatron_value"] == pytest.approx(0.03)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze_opponent_mix_manifest(source, output, producer_checkpoint=producer)


def test_frozen_manifest_detects_config_tampering(tmp_path: Path) -> None:
    producer = tmp_path / "producer.pt"
    producer.write_bytes(b"producer")
    source = _write_manifest(
        tmp_path,
        {"categories": [{"name": "producer_self_play", "weight": 1, "source": "self"}]},
    )
    output = tmp_path / "resolved.json"
    freeze_opponent_mix_manifest(source, output, producer_checkpoint=producer)
    os.chmod(output, 0o644)
    payload = json.loads(output.read_text())
    payload["categories"][0]["weight"] = 2
    output.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="frozen opponent-mix digest mismatch"):
        resolve_opponent_mix_manifest(output, producer_checkpoint=producer)


def test_cli_freezes_a_producer_bound_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    producer = tmp_path / "producer.pt"
    producer.write_bytes(b"producer")
    source = _write_manifest(
        tmp_path,
        {"categories": [{"name": "producer_self_play", "weight": 1, "source": "self"}]},
    )
    output = tmp_path / "cli-resolved.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "opponent_mix_registry.py",
            "--manifest",
            str(source),
            "--producer-checkpoint",
            str(producer),
            "--freeze-output",
            str(output),
        ],
    )

    opponent_mix_registry_main()

    assert capsys.readouterr().out.strip() == str(output.resolve())
    assert json.loads(output.read_text())["_frozen"]["producer_checkpoint"]["path"] == str(
        producer.resolve()
    )


def test_checked_in_r9_template_resolves_real_75_10_5_5_plus_3_percent_mix(
    tmp_path: Path,
) -> None:
    producer = tmp_path / "producer.pt"
    public = tmp_path / "public.pt"
    older = tmp_path / "older.pt"
    hard = tmp_path / "hard.pt"
    for path, payload in (
        (producer, b"producer"),
        (public, b"public"),
        (older, b"older"),
        (hard, b"hard"),
    ):
        path.write_bytes(payload)

    registry_path = tmp_path / "registry.json"
    registry = ChampionRegistry(registry_path)
    registry.set_role("public_champion", public, version=3)
    registry.append_pool(older, version=2, provenance={"tag": "older_champion"})
    registry.append_pool(hard, version=-1, provenance={"tag": "hard_negative"})
    registry.save()
    template = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "opponent_mix"
        / "opponent_mix_r9_exploiter.json"
    )
    output = tmp_path / "r9.resolved.json"

    freeze_opponent_mix_manifest(
        template,
        output,
        producer_checkpoint=producer,
        registry_path_override=registry_path,
        external_fraction=0.03,
    )

    config = resolve_opponent_mix_manifest(output, producer_checkpoint=producer)
    weights = config.effective_weights()
    assert weights["catanatron_value"] == pytest.approx(0.03)
    assert weights["producer_self_play"] == pytest.approx(0.97 * 75 / 95)
    assert weights["previous_public_champion"] == pytest.approx(0.97 * 10 / 95)
    assert weights["older_champion"] == pytest.approx(0.97 * 5 / 95)
    assert weights["hard_experimental"] == pytest.approx(0.97 * 5 / 95)
    by_name = {category.name: category for category in config.categories}
    assert by_name["previous_public_champion"].checkpoints[0].path == str(public.resolve())
    assert by_name["older_champion"].checkpoints[0].path == str(older.resolve())
    assert by_name["hard_experimental"].checkpoints[0].path == str(hard.resolve())
