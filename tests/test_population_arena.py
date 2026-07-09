from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.champion_registry import ChampionRegistry
from tools.population_arena import (
    ArenaPlayer,
    MatchJob,
    arena_panel_result_for,
    build_arena_report,
    build_payoff_matrix,
    build_roster,
    collect_pair_result,
    generate_all_pairs_schedule,
    nash_elo,
    rank_recent_pool_entries,
    solve_nash_equilibrium,
    solve_population_nash,
)


def _write_checkpoint(path: Path, content: bytes = b"weights") -> Path:
    path.write_bytes(content)
    return path


# =============================================================================
# 1. Roster construction
# =============================================================================
def test_build_roster_combines_pool_explicit_and_bots(tmp_path: Path) -> None:
    reg = ChampionRegistry(tmp_path / "registry.json")
    for i in range(3):
        ckpt = _write_checkpoint(tmp_path / f"gen{i}.pt", content=f"gen{i}".encode())
        reg.append_pool(ckpt, version=i)
    v3a = _write_checkpoint(tmp_path / "v3a.pt", content=b"v3a")

    roster = build_roster(registry=reg, explicit_checkpoints=[str(v3a)], bot_kinds=("catanatron_value",))

    kinds = {p.kind for p in roster}
    assert kinds == {"net", "bot"}
    names = {p.name for p in roster}
    assert str(v3a) in names
    assert "catanatron_value" in names
    # 3 pool checkpoints + 1 explicit + 1 bot
    assert len(roster) == 5


def test_build_roster_dedupes_when_explicit_matches_pool_entry(tmp_path: Path) -> None:
    reg = ChampionRegistry(tmp_path / "registry.json")
    ckpt = _write_checkpoint(tmp_path / "gen0.pt")
    reg.append_pool(ckpt, version=0)

    roster = build_roster(registry=reg, explicit_checkpoints=[str(ckpt)], bot_kinds=())
    nets = [p for p in roster if p.kind == "net"]
    assert len(nets) == 1


def test_build_roster_rejects_unknown_bot_kind() -> None:
    with pytest.raises(ValueError, match="unknown bot kind"):
        build_roster(bot_kinds=("not_a_real_bot",))


def test_rank_recent_pool_entries_keeps_most_recent_by_added_at(tmp_path: Path) -> None:
    reg = ChampionRegistry(tmp_path / "registry.json")
    paths = []
    for i in range(5):
        ckpt = _write_checkpoint(tmp_path / f"gen{i}.pt", content=f"gen{i}".encode())
        reg.append_pool(ckpt, version=i)
        paths.append(str(ckpt))

    recent = rank_recent_pool_entries(reg.opponent_pool(), max_nets=2)
    assert [e.checkpoint_path for e in recent] == [paths[4], paths[3]]


def test_build_roster_caps_pool_nets_at_max(tmp_path: Path) -> None:
    reg = ChampionRegistry(tmp_path / "registry.json")
    for i in range(20):
        ckpt = _write_checkpoint(tmp_path / f"gen{i}.pt", content=f"gen{i}".encode())
        reg.append_pool(ckpt, version=i)

    roster = build_roster(registry=reg, bot_kinds=(), max_pool_nets=8)
    assert len([p for p in roster if p.kind == "net"]) == 8


# =============================================================================
# 2. All-pairs schedule generator
# =============================================================================
def _sample_roster() -> list[ArenaPlayer]:
    return [
        ArenaPlayer(name="runs/genA/checkpoint.pt", kind="net", checkpoint_path="runs/genA/checkpoint.pt", source="registry_pool"),
        ArenaPlayer(name="runs/genB/checkpoint.pt", kind="net", checkpoint_path="runs/genB/checkpoint.pt", source="registry_pool"),
        ArenaPlayer(name="runs/genC/checkpoint.pt", kind="net", checkpoint_path="runs/genC/checkpoint.pt", source="explicit_checkpoint"),
        ArenaPlayer(name="catanatron_value", kind="bot", source="bot_roster"),
        ArenaPlayer(name="catanatron_ab3", kind="bot", source="bot_roster"),
    ]


def test_schedule_covers_every_net_net_and_net_bot_pair_exactly_once() -> None:
    roster = _sample_roster()
    jobs = generate_all_pairs_schedule(roster, include_raw_policy_self_pairs=False)

    nets = [p.name for p in roster if p.kind == "net"]
    bots = [p.name for p in roster if p.kind == "bot"]
    expected_pairs = set()
    for i, a in enumerate(nets):
        for b in nets[i + 1 :]:
            expected_pairs.add(frozenset((a, b)))
    for a in nets:
        for b in bots:
            expected_pairs.add(frozenset((a, b)))

    actual_pairs = [frozenset((j.player_a, j.player_b)) for j in jobs]
    assert len(actual_pairs) == len(expected_pairs)  # no duplicates, none missing
    assert set(actual_pairs) == expected_pairs


def test_schedule_has_no_self_pairs_and_no_bot_vs_bot() -> None:
    roster = _sample_roster()
    jobs = generate_all_pairs_schedule(roster, include_raw_policy_self_pairs=False)
    for job in jobs:
        assert job.player_a != job.player_b
    bot_names = {p.name for p in roster if p.kind == "bot"}
    assert not any(j.player_a in bot_names and j.player_b in bot_names for j in jobs)


def test_schedule_raw_policy_self_pairs_one_per_net() -> None:
    roster = _sample_roster()
    jobs = generate_all_pairs_schedule(roster, include_raw_policy_self_pairs=True)
    raw_jobs = [j for j in jobs if j.match_kind == "net_vs_raw_policy"]
    nets = [p.name for p in roster if p.kind == "net"]
    assert len(raw_jobs) == len(nets)
    assert {j.player_a for j in raw_jobs} == set(nets)
    assert all(j.player_b == f"{j.player_a}::raw_policy" for j in raw_jobs)


def test_schedule_command_shape_per_match_kind() -> None:
    roster = _sample_roster()
    jobs = generate_all_pairs_schedule(roster, include_raw_policy_self_pairs=True)

    net_vs_net = next(j for j in jobs if j.match_kind == "net_vs_net")
    assert "gumbel_search_cross_net_h2h.py" in net_vs_net.command[1]
    assert "--candidate" in net_vs_net.command and "--baseline" in net_vs_net.command

    net_vs_bot = next(j for j in jobs if j.match_kind == "net_vs_bot")
    assert "gumbel_search_vs_bot_h2h.py" in net_vs_bot.command[1]
    assert "--baseline-bot" in net_vs_bot.command

    net_vs_raw = next(j for j in jobs if j.match_kind == "net_vs_raw_policy")
    assert "gumbel_search_vs_raw_h2h.py" in net_vs_raw.command[1]
    assert "--checkpoint" in net_vs_raw.command
    assert "--devices" not in net_vs_raw.command  # that tool has no --devices flag


def test_schedule_seeds_are_distinct_per_pair() -> None:
    roster = _sample_roster()
    jobs = generate_all_pairs_schedule(roster, include_raw_policy_self_pairs=True)
    seeds = []
    for job in jobs:
        i = job.command.index("--base-seed")
        seeds.append(job.command[i + 1])
    assert len(seeds) == len(set(seeds))


# =============================================================================
# 3. Payoff-matrix assembly
# =============================================================================
def _write_h2h_result(path: Path, *, candidate_wins: int, opponent_wins: int) -> None:
    """Minimal per-game fixture matching every reused H2H tool's shape:
    one WW pair per candidate win, one LL pair per opponent win (game_seed
    unique per pair, two games per pair -- the paired-seed color-swap
    protocol h2h_postrepair_aggregate.py's game_seed-keyed pairing expects)."""
    games = []
    seed = 0
    for _ in range(candidate_wins):
        games.append({"game_seed": seed, "orientation": "a", "search_won": True, "candidate_won": True})
        games.append({"game_seed": seed, "orientation": "b", "search_won": True, "candidate_won": True})
        seed += 1
    for _ in range(opponent_wins):
        games.append({"game_seed": seed, "orientation": "a", "search_won": False, "candidate_won": False})
        games.append({"game_seed": seed, "orientation": "b", "search_won": False, "candidate_won": False})
        seed += 1
    path.write_text(json.dumps({"games": games}))


def test_collect_pair_result_returns_none_when_no_shards(tmp_path: Path) -> None:
    job = MatchJob("a__vs__b", "a", "b", "net_vs_net", (), str(tmp_path / "a__vs__b_*.json"), str(tmp_path / "a__vs__b_local.json"))
    assert collect_pair_result(job) is None


def test_collect_pair_result_aggregates_via_h2h_postrepair(tmp_path: Path) -> None:
    _write_h2h_result(tmp_path / "a__vs__b_local.json", candidate_wins=7, opponent_wins=3)
    job = MatchJob("a__vs__b", "a", "b", "net_vs_net", (), str(tmp_path / "a__vs__b_*.json"), str(tmp_path / "a__vs__b_local.json"))
    result = collect_pair_result(job)
    assert result is not None
    assert result["pairs_decisive"] == 10
    assert result["pair_win_rate"] == pytest.approx(0.7)


def test_build_payoff_matrix_is_antisymmetric_and_flags_missing(tmp_path: Path) -> None:
    roster = [
        ArenaPlayer(name="net_a", kind="net", checkpoint_path="net_a.pt"),
        ArenaPlayer(name="net_b", kind="net", checkpoint_path="net_b.pt"),
        ArenaPlayer(name="catanatron_value", kind="bot"),
    ]
    _write_h2h_result(tmp_path / "net_a__vs__net_b_local.json", candidate_wins=6, opponent_wins=4)
    jobs = [
        MatchJob("net_a__vs__net_b", "net_a", "net_b", "net_vs_net", (),
                 str(tmp_path / "net_a__vs__net_b_*.json"), str(tmp_path / "net_a__vs__net_b_local.json")),
        # net_a vs bot: no result file written -> missing.
        MatchJob("net_a__vs__catanatron_value", "net_a", "catanatron_value", "net_vs_bot", (),
                 str(tmp_path / "net_a__vs__catanatron_value_*.json"), str(tmp_path / "net_a__vs__catanatron_value_local.json")),
    ]
    matrix = build_payoff_matrix(roster, jobs, include_raw_policy=False)

    i, j = matrix.index("net_a"), matrix.index("net_b")
    assert matrix.payoff[i][j] == pytest.approx(0.2)
    assert matrix.payoff[j][i] == pytest.approx(-0.2)
    assert matrix.games_played[i][j] == 10
    assert ("net_a", "catanatron_value") in matrix.missing_pairs
    assert matrix.payoff[matrix.index("net_a")][matrix.index("catanatron_value")] == 0.0


# =============================================================================
# 4. Nash-averaging solver
# =============================================================================
def test_nash_solver_rock_paper_scissors_is_uniform() -> None:
    rps = [
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
    ]
    result = solve_nash_equilibrium(rps)
    assert result.strategy == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-4)
    assert result.value == pytest.approx(0.0, abs=1e-6)
    for payoff in result.payoff_vs_equilibrium:
        assert payoff == pytest.approx(0.0, abs=1e-4)


def test_nash_solver_dominant_strategy_is_pure() -> None:
    # Row/col 0 beats col/row 1 outright; a pure Nash equilibrium is [1, 0].
    dominant = [
        [0.0, 1.0],
        [-1.0, 0.0],
    ]
    result = solve_nash_equilibrium(dominant)
    assert result.strategy == pytest.approx([1.0, 0.0], abs=1e-4)


def test_nash_solver_three_player_dominant_over_a_decoy_pair() -> None:
    # Player 0 beats both 1 and 2; 1 and 2 are an internal RPS-style wash (0 vs each other).
    matrix = [
        [0.0, 1.0, 1.0],
        [-1.0, 0.0, 0.3],
        [-1.0, -0.3, 0.0],
    ]
    result = solve_nash_equilibrium(matrix)
    # The dominant player must take all the equilibrium mass; 1 and 2 cannot
    # profitably deviate into since they both lose to 0.
    assert result.strategy[0] == pytest.approx(1.0, abs=1e-3)


def test_nash_solver_rejects_non_antisymmetric_matrix() -> None:
    with pytest.raises(ValueError, match="antisymmetric"):
        solve_nash_equilibrium([[0.0, 0.5], [0.5, 0.0]])


def test_nash_solver_rejects_non_square_matrix() -> None:
    with pytest.raises(ValueError, match="square"):
        solve_nash_equilibrium([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0]])


def test_solve_population_nash_attaches_player_names(tmp_path: Path) -> None:
    roster = [
        ArenaPlayer(name="net_a", kind="net", checkpoint_path="a.pt"),
        ArenaPlayer(name="net_b", kind="net", checkpoint_path="b.pt"),
    ]
    _write_h2h_result(tmp_path / "net_a__vs__net_b_local.json", candidate_wins=8, opponent_wins=2)
    jobs = [
        MatchJob("net_a__vs__net_b", "net_a", "net_b", "net_vs_net", (),
                 str(tmp_path / "net_a__vs__net_b_*.json"), str(tmp_path / "net_a__vs__net_b_local.json")),
    ]
    matrix = build_payoff_matrix(roster, jobs, include_raw_policy=False)
    nash = solve_population_nash(matrix)
    assert nash.players == ["net_a", "net_b"]
    assert nash.strategy[0] > nash.strategy[1]  # net_a (80% win rate) dominates


# =============================================================================
# 5. Combined report + tripwire hook
# =============================================================================
def test_build_arena_report_sorted_by_nash_elo_descending(tmp_path: Path) -> None:
    roster = [
        ArenaPlayer(name="net_a", kind="net", checkpoint_path="a.pt"),
        ArenaPlayer(name="net_b", kind="net", checkpoint_path="b.pt"),
    ]
    _write_h2h_result(tmp_path / "net_a__vs__net_b_local.json", candidate_wins=9, opponent_wins=1)
    jobs = [
        MatchJob("net_a__vs__net_b", "net_a", "net_b", "net_vs_net", (),
                 str(tmp_path / "net_a__vs__net_b_*.json"), str(tmp_path / "net_a__vs__net_b_local.json")),
    ]
    matrix = build_payoff_matrix(roster, jobs, include_raw_policy=False)
    nash = solve_population_nash(matrix)
    report = build_arena_report(roster, matrix, nash)

    assert [row["player"] for row in report["ratings"]] == ["net_a", "net_b"]
    assert report["ratings"][0]["nash_elo"] > report["ratings"][1]["nash_elo"]
    assert report["coverage"]["pairs_missing"] == 0
    assert report["coverage"]["pairs_scheduled"] == 1
    assert report["coverage"]["pairs_with_results"] == 1
    assert report["coverage"]["population_pairs_possible_if_fully_meshed"] == 1


def test_nash_elo_is_monotonic_in_payoff() -> None:
    assert nash_elo(0.5) > nash_elo(0.0) > nash_elo(-0.5)
    assert nash_elo(0.0) == pytest.approx(0.0, abs=1e-6)


def test_arena_panel_result_for_pools_wins_and_losses(tmp_path: Path) -> None:
    roster = [
        ArenaPlayer(name="net_a", kind="net", checkpoint_path="a.pt"),
        ArenaPlayer(name="net_b", kind="net", checkpoint_path="b.pt"),
        ArenaPlayer(name="catanatron_value", kind="bot"),
    ]
    _write_h2h_result(tmp_path / "net_a__vs__net_b_local.json", candidate_wins=7, opponent_wins=3)
    _write_h2h_result(tmp_path / "net_a__vs__catanatron_value_local.json", candidate_wins=6, opponent_wins=4)
    jobs = [
        MatchJob("net_a__vs__net_b", "net_a", "net_b", "net_vs_net", (),
                 str(tmp_path / "net_a__vs__net_b_*.json"), str(tmp_path / "net_a__vs__net_b_local.json")),
        MatchJob("net_a__vs__catanatron_value", "net_a", "catanatron_value", "net_vs_bot", (),
                 str(tmp_path / "net_a__vs__catanatron_value_*.json"), str(tmp_path / "net_a__vs__catanatron_value_local.json")),
    ]
    matrix = build_payoff_matrix(roster, jobs, include_raw_policy=False)
    panel = arena_panel_result_for(matrix, subject_name="net_a")
    assert panel.wins == 13
    assert panel.losses == 7
    assert panel.draws == 0


def test_arena_panel_result_for_unknown_subject_raises(tmp_path: Path) -> None:
    roster = [ArenaPlayer(name="net_a", kind="net", checkpoint_path="a.pt")]
    matrix = build_payoff_matrix(roster, [], include_raw_policy=False)
    with pytest.raises(ValueError, match="not in this arena"):
        arena_panel_result_for(matrix, subject_name="nonexistent")
