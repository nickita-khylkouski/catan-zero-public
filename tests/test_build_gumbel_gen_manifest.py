from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import build_gumbel_gen_manifest as merge  # type: ignore  # noqa: E402


def _write_manifest(root: Path, *, n_shards: int, rows_per_shard: int, prefix: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    shards = []
    for i in range(n_shards):
        shard = root / f"{prefix}_shard_{i:05d}.npz"
        shard.write_bytes(b"fake")
        shards.append(str(shard))
    manifest = {
        "schema": "entity_tokens_v1",
        "converted_rows": n_shards * rows_per_shard,
        "shards": shards,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# _load_manifest_shards_and_rows
# ---------------------------------------------------------------------------


def test_load_manifest_shards_and_rows_resolves_relative_and_absolute_paths(tmp_path):
    manifest_path = _write_manifest(tmp_path / "gen1", n_shards=3, rows_per_shard=100, prefix="gen")

    shards, rows = merge._load_manifest_shards_and_rows(manifest_path)

    assert len(shards) == 3
    assert rows == 300
    assert all(Path(s).exists() for s in shards)


# ---------------------------------------------------------------------------
# _select_teacher_shards_for_budget
# ---------------------------------------------------------------------------


def test_select_teacher_shards_for_budget_stops_near_the_row_budget(tmp_path):
    teacher_path = _write_manifest(
        tmp_path / "teacher", n_shards=20, rows_per_shard=1000, prefix="teacher"
    )
    shards, rows = merge._load_manifest_shards_and_rows(teacher_path)

    selected, selected_rows = merge._select_teacher_shards_for_budget(
        [(shards, rows)], target_rows=5500, seed=0
    )

    # Whole-shard selection can't hit 5500 exactly (1000 rows/shard) -- must
    # stop at or just past the budget, never wildly overshoot.
    assert 5000 <= selected_rows <= 6000
    assert len(selected) == selected_rows // 1000
    # Every selected shard must be a real path from the source list.
    assert set(selected).issubset(set(shards))


def test_select_teacher_shards_for_budget_is_deterministic_given_a_seed(tmp_path):
    teacher_path = _write_manifest(
        tmp_path / "teacher", n_shards=20, rows_per_shard=1000, prefix="teacher"
    )
    shards, rows = merge._load_manifest_shards_and_rows(teacher_path)

    first, _ = merge._select_teacher_shards_for_budget([(shards, rows)], target_rows=5500, seed=42)
    second, _ = merge._select_teacher_shards_for_budget([(shards, rows)], target_rows=5500, seed=42)

    assert first == second


def test_select_teacher_shards_for_budget_caps_at_total_availability(tmp_path):
    teacher_path = _write_manifest(
        tmp_path / "teacher", n_shards=5, rows_per_shard=1000, prefix="teacher"
    )
    shards, rows = merge._load_manifest_shards_and_rows(teacher_path)

    selected, selected_rows = merge._select_teacher_shards_for_budget(
        [(shards, rows)], target_rows=1_000_000, seed=0
    )

    assert selected_rows == 5000
    assert set(selected) == set(shards)


# ---------------------------------------------------------------------------
# build_manifest (end-to-end)
# ---------------------------------------------------------------------------


def test_build_manifest_includes_all_gen_shards_and_a_bounded_teacher_replay_mix(tmp_path):
    gen1 = _write_manifest(tmp_path / "gen1", n_shards=10, rows_per_shard=1000, prefix="gen")
    teacher = _write_manifest(tmp_path / "teacher", n_shards=100, rows_per_shard=1000, prefix="teacher")
    out_dir = tmp_path / "combined"

    result = merge.build_manifest(
        gen_inputs=[gen1],
        teacher_inputs=[teacher],
        replay_fraction=0.15,
        out_dir=out_dir,
        seed=7,
    )

    gen_shard_count = 10
    # All 10k gen rows must be present.
    gen_rows = 10 * 1000
    assert result["gen_rows"] == gen_rows
    # replay_fraction=0.15 of the FINAL total (not of the teacher corpus):
    # total = gen_rows / (1 - 0.15); teacher_rows = total - gen_rows.
    expected_total = gen_rows / (1 - 0.15)
    expected_teacher_rows = expected_total - gen_rows
    assert abs(result["teacher_rows"] - expected_teacher_rows) <= 1000  # within one shard
    assert result["converted_rows"] == result["gen_rows"] + result["teacher_rows"]

    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk["schema"] == "entity_tokens_v1"
    assert len(on_disk["shards"]) == gen_shard_count + result["teacher_rows"] // 1000
    assert on_disk["converted_rows"] == result["converted_rows"]
    assert on_disk["actual_replay_fraction"] == pytest.approx(
        result["teacher_rows"] / result["converted_rows"], abs=1e-6
    )


def test_build_manifest_rejects_a_replay_fraction_outside_valid_range(tmp_path):
    gen1 = _write_manifest(tmp_path / "gen1", n_shards=1, rows_per_shard=1000, prefix="gen")
    teacher = _write_manifest(tmp_path / "teacher", n_shards=1, rows_per_shard=1000, prefix="teacher")

    with pytest.raises(SystemExit):
        merge.build_manifest(
            gen_inputs=[gen1],
            teacher_inputs=[teacher],
            replay_fraction=1.5,
            out_dir=tmp_path / "combined",
            seed=0,
        )


def test_build_manifest_works_with_no_teacher_inputs(tmp_path):
    gen1 = _write_manifest(tmp_path / "gen1", n_shards=4, rows_per_shard=500, prefix="gen")
    out_dir = tmp_path / "combined"

    result = merge.build_manifest(
        gen_inputs=[gen1],
        teacher_inputs=[],
        replay_fraction=0.15,
        out_dir=out_dir,
        seed=0,
    )

    assert result["teacher_rows"] == 0
    assert result["converted_rows"] == result["gen_rows"] == 2000
    on_disk = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(on_disk["shards"]) == 4
    assert on_disk["actual_replay_fraction"] == 0.0
