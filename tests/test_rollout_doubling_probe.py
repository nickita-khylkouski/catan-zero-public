from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from rollout_doubling_probe import (  # type: ignore  # noqa: E402
    H2H_TOOL_RELPATH,
    build_h2h_command,
    build_invocation_descriptor,
    extract_rollout_doubling_summary,
)


def test_build_h2h_command_shape_default_pairs():
    command = build_h2h_command(
        champion_checkpoint="ckpt/v3a.pt",
        n_full_a=64,
        n_full_b=128,
        pairs=200,
        h2h_out_path="/tmp/out.json",
    )
    assert command[0] == sys.executable
    assert command[1] == H2H_TOOL_RELPATH
    assert "--candidate" in command
    assert command[command.index("--candidate") + 1] == "ckpt/v3a.pt"
    assert "--baseline" in command
    assert command[command.index("--baseline") + 1] == "ckpt/v3a.pt"
    # candidate = doubled budget (n_full_b), baseline = original (n_full_a).
    assert "--candidate-n-full" in command
    assert command[command.index("--candidate-n-full") + 1] == "128"
    assert "--baseline-n-full" in command
    assert command[command.index("--baseline-n-full") + 1] == "64"
    assert "--pairs" in command
    assert command[command.index("--pairs") + 1] == "200"
    assert "--out" in command
    assert command[command.index("--out") + 1] == "/tmp/out.json"


def test_build_h2h_command_various_combinations():
    for n_full_a, n_full_b, pairs in [(32, 64, 50), (64, 256, 10), (128, 128, 1)]:
        command = build_h2h_command(
            champion_checkpoint="ckpt/foo.pt",
            n_full_a=n_full_a,
            n_full_b=n_full_b,
            pairs=pairs,
            h2h_out_path="/tmp/foo.json",
        )
        assert command[command.index("--candidate-n-full") + 1] == str(n_full_b)
        assert command[command.index("--baseline-n-full") + 1] == str(n_full_a)
        assert command[command.index("--pairs") + 1] == str(pairs)


def test_build_h2h_command_same_checkpoint_both_roles():
    command = build_h2h_command(
        champion_checkpoint="ckpt/self_play_ckpt.pt",
        n_full_a=64,
        n_full_b=128,
        pairs=1,
        h2h_out_path="/tmp/x.json",
    )
    candidate = command[command.index("--candidate") + 1]
    baseline = command[command.index("--baseline") + 1]
    assert candidate == baseline == "ckpt/self_play_ckpt.pt"


def test_build_invocation_descriptor_reports_games_total_as_2x_pairs():
    descriptor = build_invocation_descriptor(
        champion_checkpoint="ckpt/v3a.pt",
        n_full_a=64,
        n_full_b=128,
        pairs=200,
        h2h_out_path="/tmp/out.json",
    )
    assert descriptor["pairs"] == 200
    assert descriptor["games_total"] == 400
    assert descriptor["measurement"] == "rollout_doubling_probe"
    assert descriptor["mechanism"] == "B_exit_fixed_point"
    assert descriptor["command"][0] == sys.executable


def test_extract_rollout_doubling_summary_pulls_expected_fields():
    fake_h2h_json = {
        "candidate_checkpoint": "ckpt/v3a.pt",
        "baseline_checkpoint": "ckpt/v3a.pt",
        "candidate_n_full": 128,
        "baseline_n_full": 64,
        "games_played": 400,
        "games_with_winner": 390,
        "candidate_wins": 210,
        "baseline_wins": 180,
        "candidate_win_rate": 0.538,
        "pentanomial_sprt": {"model": "pentanomial", "decision": "H1", "llr": 4.2},
        "pair_diagnostics": {"ww_pairs": 60, "ll_pairs": 50, "split_pairs": 85, "incomplete_pairs": 5},
        "games": [{"game_seed": 1, "candidate_won": True}],  # should not leak into the compact summary
    }
    summary = extract_rollout_doubling_summary(fake_h2h_json)
    assert summary["candidate_win_rate"] == 0.538
    assert summary["candidate_n_full"] == 128
    assert summary["baseline_n_full"] == 64
    assert summary["pentanomial_sprt"]["decision"] == "H1"
    assert summary["pair_diagnostics"]["ww_pairs"] == 60
    assert "games" not in summary


def test_extract_rollout_doubling_summary_missing_fields_are_none():
    summary = extract_rollout_doubling_summary({})
    assert summary["candidate_win_rate"] is None
    assert summary["pentanomial_sprt"] is None
