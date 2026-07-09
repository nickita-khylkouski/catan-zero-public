from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from h2h_v3conf_aggregate import (  # type: ignore  # noqa: E402
    _arm_label,
    _reduce_by_game_seed,
    aggregate_arm,
    group_files_by_arm,
)


def _g(seed: int, orientation: int, search_won, truncated=False, winner="RED", decisions=50, source="fileA"):
    return {
        "game_seed": seed,
        "orientation": orientation,
        "search_color": "RED" if orientation == 0 else "BLUE",
        "pair_id": seed,
        "search_won": search_won,
        "truncated": truncated,
        "winner": winner,
        "decisions": decisions,
        "_source": source,
    }


def test_arm_label_from_checkpoint_and_cscale():
    assert _arm_label("/runs/bc/entity_graph_35m_v3b_unfreeze_kl_arch_masked/checkpoint.pt", 0.03) == "v3b_cs0.03"
    assert _arm_label("/runs/bc/entity_graph_35m_v3a_unfreeze_kl_masked/checkpoint.pt", 0.1) == "v3a_cs0.1"


def test_reduce_by_game_seed_classifies_pairs():
    # Each seed pairs its two color-swapped orientations (0 and 1).
    games = [
        _g(100, 0, True), _g(100, 1, True),      # WW
        _g(101, 0, False), _g(101, 1, False),    # LL
        _g(102, 0, True), _g(102, 1, False),     # split
        _g(103, 0, True), _g(103, 1, None, truncated=True),  # incomplete
        _g(104, 0, True),                         # missing orientation -> incomplete
    ]
    r = _reduce_by_game_seed(games)
    assert r["pentanomial_counts"] == (1, 1, 1)  # (LL, split, WW)
    assert r["concordant_outcomes"] == [True, False]  # WW=True, LL=False; split excluded
    assert r["diagnostics"]["incomplete_pairs"] == 2


def test_reduce_dedupes_bit_identical_replicas():
    # The seed-collision bug: the SAME (seed, orientation) game run twice
    # bit-for-bit (fleet replicated the block across hosts) must count ONCE,
    # not inflate the pair count / significance.
    games = [
        _g(500, 0, True), _g(500, 0, True),   # orientation 0, replicated
        _g(500, 1, True), _g(500, 1, True),   # orientation 1, replicated
    ]
    r = _reduce_by_game_seed(games)
    assert r["duplicate_games_dropped"] == 2
    assert r["pentanomial_counts"] == (0, 0, 1)  # exactly ONE WW pair
    assert r["concordant_outcomes"] == [True]
    assert r["distinct_seeds"] == 1


def test_reduce_keeps_genuinely_different_same_key_games():
    # Same (seed, orientation) but a DIFFERENT result (e.g. different search
    # seed) is NOT a bit-identical dup -> kept (not silently dropped).
    games = [
        _g(600, 0, True, decisions=40), _g(600, 0, False, decisions=90),
    ]
    r = _reduce_by_game_seed(games)
    assert r["duplicate_games_dropped"] == 0


def test_aggregate_arm_and_grouping(tmp_path):
    ck = "/runs/bc/entity_graph_35m_v3b_x/checkpoint.pt"
    fileA = {"checkpoint": ck, "c_scale": 0.03, "n_full": 64,
             "games": [_g(1000, 0, True), _g(1000, 1, True), _g(1001, 0, True), _g(1001, 1, False)]}
    fileB = {"checkpoint": ck, "c_scale": 0.03, "n_full": 64,
             "games": [_g(2000, 0, False), _g(2000, 1, False)]}
    pa = tmp_path / "arm_a.json"
    pb = tmp_path / "arm_b.json"
    pa.write_text(json.dumps(fileA))
    pb.write_text(json.dumps(fileB))

    arms = group_files_by_arm([pa, pb])
    assert set(arms) == {"v3b_cs0.03"}
    assert len(arms["v3b_cs0.03"]) == 2

    agg = aggregate_arm(arms["v3b_cs0.03"], elo0=0.0, elo1=30.0)
    # seeds: 1000 WW, 1001 split, 2000 LL.
    assert agg["complete_pairs"] == 3
    assert agg["pairs_decisive"] == 2
    assert agg["pairs_split"] == 1
    assert agg["concordant_ww"] == 1 and agg["concordant_ll"] == 1
    assert agg["pentanomial_sprt"]["ww_pairs"] == 1
    assert agg["pentanomial_sprt"]["split_pairs"] == 1
    assert agg["pentanomial_sprt"]["ll_pairs"] == 1
    # per-game win rate: seed1000 (T,T)=2 + seed1001 (T,F)=1 + seed2000 (F,F)=0
    # -> 3 wins / 6 finished games.
    assert agg["search_win_rate_per_game"] == pytest.approx(3 / 6)
    assert agg["search_wins_game"] == 3
    assert agg["raw_wins_game"] == 3
    assert agg["games_truncated"] == 0
    assert agg["truncation_rate_per_game"] == pytest.approx(0.0)
    # gen-1 replication flags: no D1/D2, no n_full_wide/raw_policy -> no gap.
    g = agg["gen1_replication"]
    assert g["d1_noise_floor_on"] is False
    assert g["d2_variance_aware_on"] is False
    assert g["generator_cli_gap"] is False


def test_gen1_replication_flags_detect_used_knobs(tmp_path):
    ck = "/runs/bc/entity_graph_35m_v3b_x/checkpoint.pt"
    # An arm that used n_full_wide (generator lacks a CLI for it) and D2 on.
    f = {"checkpoint": ck, "c_scale": 0.1, "n_full": 64, "n_full_wide": 128,
         "variance_aware_q": True, "public_observation": True,
         "games": [_g(1, 0, True), _g(1, 0, True)]}
    p = tmp_path / "arm.json"
    p.write_text(json.dumps(f))
    agg = aggregate_arm([p], elo0=0.0, elo1=30.0)
    g = agg["gen1_replication"]
    assert g["d2_variance_aware_on"] is True
    assert g["n_full_wide"] == 128
    assert g["public_observation"] is True
    assert g["generator_cli_gap"] is True  # n_full_wide used -> generator needs wiring
