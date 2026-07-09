from __future__ import annotations

import json
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import h2h_postrepair_aggregate as agg  # type: ignore  # noqa: E402


def _game(
    *,
    game_seed: int,
    pair_id: int,
    search_won: bool | None,
    truncated: bool = False,
    orientation: int = 0,
    search_color: str = "RED",
    winner: str = "RED",
    decisions: int = 50,
) -> dict:
    return {
        "game_seed": game_seed,
        "pair_id": pair_id,
        "search_won": search_won,
        "truncated": truncated,
        "orientation": orientation,
        "search_color": search_color,
        "winner": winner,
        "decisions": decisions,
    }


def test_pooled_pair_outcomes_uses_game_seed_not_pair_id():
    # Two "files" both use local pair_id=0, but different game_seed blocks --
    # a pair_id-keyed join would incorrectly merge these into one pair.
    file1_games = [
        _game(game_seed=250000, pair_id=0, search_won=True, orientation=0, search_color="RED"),
        _game(game_seed=250000, pair_id=0, search_won=True, orientation=1, search_color="BLUE"),
    ]
    file2_games = [
        _game(game_seed=251000, pair_id=0, search_won=False, orientation=0, search_color="RED"),
        _game(game_seed=251000, pair_id=0, search_won=False, orientation=1, search_color="BLUE"),
    ]
    outcomes, diagnostics = agg._pooled_pair_outcomes(file1_games + file2_games)
    assert outcomes == [True, False] or outcomes == [False, True]
    assert diagnostics["ww_pairs"] == 1
    assert diagnostics["ll_pairs"] == 1
    assert diagnostics["split_pairs"] == 0
    assert diagnostics["incomplete_pairs"] == 0


def test_pooled_pair_outcomes_excludes_split_and_incomplete():
    games = [
        # split: won as one color, lost as the other
        _game(game_seed=1, pair_id=0, search_won=True),
        _game(game_seed=1, pair_id=0, search_won=False),
        # incomplete: one orientation truncated (search_won=None)
        _game(game_seed=2, pair_id=1, search_won=True),
        _game(game_seed=2, pair_id=1, search_won=None, truncated=True),
    ]
    outcomes, diagnostics = agg._pooled_pair_outcomes(games)
    assert outcomes == []
    assert diagnostics["split_pairs"] == 1
    assert diagnostics["incomplete_pairs"] == 1


def test_aggregate_arm_pools_across_multiple_files(tmp_path: Path):
    file_a = tmp_path / "armA_hostX_gpu0.json"
    file_b = tmp_path / "armA_hostX_gpu1.json"
    file_a.write_text(
        json.dumps(
            {
                "n_full": 64,
                "c_scale": 0.1,
                "c_visit": 50.0,
                "max_root_candidates_wide": 54,
                "max_decisions": 600,
                "games": [
                    _game(game_seed=250000, pair_id=0, search_won=True, orientation=0, search_color="RED"),
                    _game(game_seed=250000, pair_id=0, search_won=True, orientation=1, search_color="BLUE"),
                    _game(game_seed=250001, pair_id=1, search_won=False, orientation=0, search_color="RED"),
                    _game(game_seed=250001, pair_id=1, search_won=False, orientation=1, search_color="BLUE"),
                ],
            }
        )
    )
    file_b.write_text(
        json.dumps(
            {
                "n_full": 64,
                "c_scale": 0.1,
                "c_visit": 50.0,
                "max_root_candidates_wide": 54,
                "max_decisions": 600,
                "games": [
                    _game(game_seed=251000, pair_id=0, search_won=True, orientation=0, search_color="RED"),
                    _game(game_seed=251000, pair_id=0, search_won=True, orientation=1, search_color="BLUE"),
                ],
            }
        )
    )
    result = agg.aggregate_arm([file_a, file_b], elo0=0.0, elo1=30.0)
    assert result["distinct_game_seeds"] == 3
    assert result["games_played"] == 6
    assert result["duplicate_games_dropped"] == 0
    assert result["search_wins"] == 2
    assert result["raw_wins"] == 1
    assert result["pairs_decisive"] == 3
    assert result["pair_win_rate"] == 2 / 3
    assert result["pair_sprt"]["elo0"] == 0.0
    assert result["pair_sprt"]["elo1"] == 30.0


def test_dedupe_games_drops_bit_identical_replicas():
    # The seed-collision bug (task #77 class): the SAME (seed, orientation)
    # game run twice bit-for-bit (fleet replicated a seed block across hosts)
    # must count ONCE, not inflate the pair count / significance.
    games = [
        _game(game_seed=500, pair_id=0, search_won=True, orientation=0, search_color="RED"),
        _game(game_seed=500, pair_id=0, search_won=True, orientation=0, search_color="RED"),  # exact replica
        _game(game_seed=500, pair_id=0, search_won=True, orientation=1, search_color="BLUE"),
    ]
    unique, dropped = agg._dedupe_games(games)
    assert dropped == 1
    assert len(unique) == 2


def test_dedupe_games_keeps_genuinely_different_same_seed_games():
    # Same (seed, orientation) but a DIFFERENT result (e.g. different search
    # seed) is NOT a bit-identical dup -> kept (not silently dropped).
    games = [
        _game(game_seed=600, pair_id=0, search_won=True, orientation=0, decisions=40),
        _game(game_seed=600, pair_id=0, search_won=False, orientation=0, decisions=90),
    ]
    unique, dropped = agg._dedupe_games(games)
    assert dropped == 0
    assert len(unique) == 2


def test_pooled_pair_outcomes_dedupes_before_pairing():
    # A seed-collision replica of an entire pair (both orientations run twice,
    # bit-for-bit, on two hosts) must resolve to exactly ONE pair, not two --
    # the exact failure mode this fix ports from h2h_v3conf_aggregate.py.
    games = [
        _game(game_seed=700, pair_id=0, search_won=True, orientation=0, search_color="RED"),
        _game(game_seed=700, pair_id=0, search_won=True, orientation=1, search_color="BLUE"),
        # replicated block from a second host colliding on the same seed:
        _game(game_seed=700, pair_id=0, search_won=True, orientation=0, search_color="RED"),
        _game(game_seed=700, pair_id=0, search_won=True, orientation=1, search_color="BLUE"),
    ]
    outcomes, diagnostics = agg._pooled_pair_outcomes(games)
    assert diagnostics["duplicate_games_dropped"] == 2
    assert outcomes == [True]
    assert diagnostics["ww_pairs"] == 1
    assert diagnostics["ll_pairs"] == 0
    assert diagnostics["split_pairs"] == 0
    assert diagnostics["incomplete_pairs"] == 0


def test_aggregate_arm_reports_duplicate_games_dropped(tmp_path: Path):
    file_a = tmp_path / "armA_hostX_gpu0.json"
    file_b = tmp_path / "armA_hostY_gpu0.json"
    # Both files carry the SAME seed block (the seed-collision incident).
    duplicated_games = [
        _game(game_seed=800, pair_id=0, search_won=True, orientation=0, search_color="RED"),
        _game(game_seed=800, pair_id=0, search_won=True, orientation=1, search_color="BLUE"),
    ]
    file_a.write_text(json.dumps({"n_full": 64, "games": duplicated_games}))
    file_b.write_text(json.dumps({"n_full": 64, "games": duplicated_games}))
    result = agg.aggregate_arm([file_a, file_b], elo0=0.0, elo1=30.0)
    assert result["duplicate_games_dropped"] == 2
    assert result["pairs_decisive"] == 1
    assert result["search_wins"] == 1
    assert result["games_played"] == 4  # raw count before dedup


def test_aggregate_arm_raises_on_config_mismatch(tmp_path: Path):
    file_a = tmp_path / "armA_hostX_gpu0.json"
    file_b = tmp_path / "armA_hostX_gpu1.json"
    file_a.write_text(json.dumps({"n_full": 64, "games": []}))
    file_b.write_text(json.dumps({"n_full": 32, "games": []}))
    try:
        agg.aggregate_arm([file_a, file_b], elo0=0.0, elo1=30.0)
        raised = False
    except ValueError:
        raised = True
    assert raised
