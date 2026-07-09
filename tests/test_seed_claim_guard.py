"""Unit tests for generate_gumbel_selfplay_data.py's --seed-claim guard
(FIX 4, task #85 hygiene batch): a filesystem-local check that catches two
same-host launches colliding on --base-seed (the #77 seed-collision class)
without needing the external seed_fleet_planner.py fleet-wide plan.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import generate_gumbel_selfplay_data as cli  # type: ignore  # noqa: E402


def test_claim_seed_range_writes_claim_file(tmp_path):
    out_dir = tmp_path / "run_a"
    out_dir.mkdir()
    cli._claim_seed_range(out_dir, base_seed=1000, games=64)

    claim_path = tmp_path / ".seed_claims" / "run_a.json"
    assert claim_path.exists()
    payload = json.loads(claim_path.read_text())
    assert payload["base_seed"] == 1000
    assert payload["games"] == 64
    assert payload["out_dir"] == str(out_dir.resolve())
    assert "hostname" in payload and "pid" in payload and "timestamp" in payload


def test_claim_seed_range_allows_resuming_same_out_dir(tmp_path):
    out_dir = tmp_path / "run_a"
    out_dir.mkdir()
    cli._claim_seed_range(out_dir, base_seed=1000, games=64)
    # Re-claiming the SAME out-dir with the same (or any) range is a resume,
    # not a collision -- must not raise.
    cli._claim_seed_range(out_dir, base_seed=1000, games=64)


def test_claim_seed_range_rejects_overlap_from_different_out_dir(tmp_path):
    out_dir_a = tmp_path / "run_a"
    out_dir_b = tmp_path / "run_b"
    out_dir_a.mkdir()
    out_dir_b.mkdir()
    cli._claim_seed_range(out_dir_a, base_seed=1000, games=64)  # claims [1000, 1064)

    with pytest.raises(SystemExit, match="seed-claim conflict"):
        cli._claim_seed_range(out_dir_b, base_seed=1032, games=64)  # overlaps


def test_claim_seed_range_allows_disjoint_ranges_from_different_out_dirs(tmp_path):
    out_dir_a = tmp_path / "run_a"
    out_dir_b = tmp_path / "run_b"
    out_dir_a.mkdir()
    out_dir_b.mkdir()
    cli._claim_seed_range(out_dir_a, base_seed=1000, games=64)  # claims [1000, 1064)
    cli._claim_seed_range(out_dir_b, base_seed=1064, games=64)  # claims [1064, 1128), touches, no overlap


def test_claim_seed_range_ignores_malformed_claim_files(tmp_path):
    out_dir = tmp_path / "run_a"
    out_dir.mkdir()
    claims_dir = tmp_path / ".seed_claims"
    claims_dir.mkdir()
    (claims_dir / "stale.json").write_text("{not valid json")
    # Must not raise despite the unreadable neighbor claim file.
    cli._claim_seed_range(out_dir, base_seed=1000, games=64)


def test_claim_seed_range_conflict_message_names_the_no_seed_claim_escape_hatch(tmp_path):
    out_dir_a = tmp_path / "run_a"
    out_dir_b = tmp_path / "run_b"
    out_dir_a.mkdir()
    out_dir_b.mkdir()
    cli._claim_seed_range(out_dir_a, base_seed=1000, games=64)
    with pytest.raises(SystemExit, match="--no-seed-claim"):
        cli._claim_seed_range(out_dir_b, base_seed=1000, games=64)
