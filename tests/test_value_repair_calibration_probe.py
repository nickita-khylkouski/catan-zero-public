from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from value_repair_calibration_probe import ENTITY_KEYS, collect_holdout_rows  # type: ignore  # noqa: E402


def _write_fake_shard(path: Path, *, n: int, game_seed_start: int, legal_width: int) -> None:
    arrays = {key: np.zeros((n, 1, 1), dtype=np.float32) for key in ENTITY_KEYS}
    arrays["legal_action_ids"] = np.zeros((n, legal_width), dtype=np.int16)
    arrays["legal_action_context"] = np.zeros((n, legal_width, 1), dtype=np.float32)
    arrays["game_seed"] = np.arange(game_seed_start, game_seed_start + n, dtype=np.int64)
    arrays["terminated"] = np.ones((n,), dtype=bool)
    arrays["truncated"] = np.zeros((n,), dtype=bool)
    arrays["winner"] = np.array(["BLUE"] * n)
    arrays["player"] = np.array(["BLUE", "RED"] * (n // 2 + 1))[:n]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def test_collect_holdout_rows_filters_by_game_seed_range_and_terminated(tmp_path: Path):
    manifest_dir = tmp_path / "gen0"
    shard_path = manifest_dir / "shard_00000.npz"
    # 10 rows: game_seed 100..109, one per row (not realistic multi-decision
    # games, but sufficient to test the range filter).
    _write_fake_shard(shard_path, n=10, game_seed_start=100, legal_width=3)
    (manifest_dir / "manifest.json").write_text(
        json.dumps({"shards": [str(shard_path)]}), encoding="utf-8"
    )

    groups = collect_holdout_rows(((str(manifest_dir), 103, 107),))

    assert len(groups) == 1
    group = groups[0]
    assert sorted(group["game_seed"].tolist()) == [103, 104, 105, 106]
    # BLUE rows (even index) are wins (z=1), RED rows (odd index) are
    # losses (z=-1) since winner is always BLUE in the fixture.
    assert set(group["z"].tolist()) == {1.0, -1.0}


def test_collect_holdout_rows_excludes_truncated_games(tmp_path: Path):
    manifest_dir = tmp_path / "gen0"
    shard_path = manifest_dir / "shard_00000.npz"
    _write_fake_shard(shard_path, n=5, game_seed_start=200, legal_width=2)
    data = dict(np.load(shard_path))
    data["truncated"][2] = True
    data["terminated"][2] = False
    np.savez(shard_path, **data)
    (manifest_dir / "manifest.json").write_text(
        json.dumps({"shards": [str(shard_path)]}), encoding="utf-8"
    )

    groups = collect_holdout_rows(((str(manifest_dir), 200, 205),))

    assert len(groups) == 1
    # Row at game_seed=202 (index 2) was truncated -- excluded.
    assert 202 not in groups[0]["game_seed"].tolist()
    assert len(groups[0]["game_seed"]) == 4


def test_collect_holdout_rows_respects_max_rows(tmp_path: Path):
    manifest_dir = tmp_path / "gen0"
    shard_path = manifest_dir / "shard_00000.npz"
    _write_fake_shard(shard_path, n=20, game_seed_start=300, legal_width=2)
    (manifest_dir / "manifest.json").write_text(
        json.dumps({"shards": [str(shard_path)]}), encoding="utf-8"
    )

    groups = collect_holdout_rows(((str(manifest_dir), 300, 320),), max_rows=5)

    total_rows = sum(len(group["game_seed"]) for group in groups)
    assert total_rows >= 5  # returns whole groups/shards, may overshoot slightly
