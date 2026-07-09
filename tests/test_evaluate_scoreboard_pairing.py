from __future__ import annotations

from tools.evaluate_scoreboard import _aggregate_results, _chunked_job_payloads
import argparse


def _base_chunk(*, chunk_index: int, wins: int, games: int, game_outcomes, leg_seed: int) -> dict:
    return {
        "track": "2p_no_trade",
        "opponent": "catanatron_ab3",
        "candidate": "ckpt.pt",
        "wins": wins,
        "games": games,
        "avg_decisions": 200.0,
        "avg_candidate_vp": 7.0,
        "avg_best_opponent_vp": 8.0,
        "avg_vp_margin": -1.0,
        "avg_candidate_win_decisions": 190.0,
        "seat_wins": {"BLUE": wins},
        "illegal_action_count": 0,
        "timeouts_or_stuck_games": 0,
        "chunk_index": chunk_index,
        "chunk_count": 2,
        "elo_vs_opponent": 0.0,
        "leg_seed": leg_seed,
        "game_outcomes": game_outcomes,
    }


def test_aggregate_results_concatenates_game_outcomes_in_chunk_order_even_if_unordered() -> None:
    # Simulate a multi-worker run where chunk 1 finishes (via as_completed)
    # before chunk 0: aggregation must still put chunk 0's games first.
    chunk1 = _base_chunk(chunk_index=1, wins=3, games=5, game_outcomes=[True, True, True, False, False], leg_seed=42)
    chunk0 = _base_chunk(chunk_index=0, wins=2, games=5, game_outcomes=[True, False, True, False, False], leg_seed=42)
    results = _aggregate_results([chunk1, chunk0])  # out of order on purpose
    assert len(results) == 1
    result = results[0]
    assert result["games"] == 10
    assert result["wins"] == 5
    assert result["leg_seed"] == 42
    assert result["game_outcomes"] == [True, False, True, False, False, True, True, True, False, False]


def test_aggregate_results_drops_game_outcomes_on_length_mismatch() -> None:
    # A stale timeout report re-run without game_outcomes-compatible data
    # should not silently hand back a scrambled/short array.
    chunk0 = _base_chunk(chunk_index=0, wins=2, games=5, game_outcomes=[True, False], leg_seed=42)
    results = _aggregate_results([chunk0])
    assert results[0]["game_outcomes"] is None


def test_aggregate_results_preserves_none_for_truncated_games() -> None:
    # Regression test for the adversarial-review truncation-as-loss bias:
    # bool(None) is False, so a naive coercion during aggregation would
    # silently turn a truncated game back into a candidate loss.
    chunk0 = _base_chunk(
        chunk_index=0, wins=2, games=5, game_outcomes=[True, None, False, True, None], leg_seed=42
    )
    chunk0["truncated_games"] = 2
    results = _aggregate_results([chunk0])
    assert results[0]["game_outcomes"] == [True, None, False, True, None]
    assert results[0]["truncated_games"] == 2


def test_aggregate_results_sums_truncated_games_across_chunks() -> None:
    chunk0 = _base_chunk(chunk_index=0, wins=2, games=5, game_outcomes=[True] * 5, leg_seed=42)
    chunk0["truncated_games"] = 1
    chunk1 = _base_chunk(chunk_index=1, wins=2, games=5, game_outcomes=[True] * 5, leg_seed=42)
    chunk1["truncated_games"] = 3
    results = _aggregate_results([chunk0, chunk1])
    assert results[0]["truncated_games"] == 4


def test_aggregate_results_leg_seed_none_when_chunks_disagree() -> None:
    chunk0 = _base_chunk(chunk_index=0, wins=2, games=5, game_outcomes=[True] * 5, leg_seed=42)
    chunk1 = _base_chunk(chunk_index=1, wins=2, games=5, game_outcomes=[True] * 5, leg_seed=999)
    results = _aggregate_results([chunk0, chunk1])
    assert results[0]["leg_seed"] is None


def test_chunked_job_payloads_leg_seed_is_constant_across_chunks() -> None:
    args = argparse.Namespace(
        candidate="ckpt.pt",
        candidate_kind="checkpoint",
        games=10,
        chunk_games=3,
        workers=4,
        seed=100,
        paired_seeds=True,
        vps_to_win=10,
        max_decisions=1000,
        device="cpu",
    )
    job = ("2p_no_trade", {"opponent": "catanatron_ab3", "opponent_kind": "catanatron_ab3", "opponent_checkpoint": None, "opponent_label": "catanatron_ab3"})
    payloads = _chunked_job_payloads(args, job)
    assert len(payloads) > 1
    leg_seeds = {payload["leg_seed"] for payload in payloads}
    assert len(leg_seeds) == 1
    # Each chunk's actual seed still differs (so game streams don't repeat).
    seeds = {payload["seed"] for payload in payloads}
    assert len(seeds) == len(payloads)
