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
    # The diagnostic must match the masked production search regime rather
    # than silently inheriting the H2H tool's unsafe/unvalidated defaults.
    assert command[command.index("--c-scale") + 1] == "0.03"
    assert command[command.index("--rescale-noise-floor-c") + 1] == "0.0"
    assert command[command.index("--sigma-eval") + 1] == "0.79"
    assert "--public-observation" in command
    assert "--information-set-search" in command
    assert command[command.index("--determinization-particles") + 1] == "4"
    assert command[command.index("--determinization-min-simulations") + 1] == "32"
    assert "--lazy-interior-chance" in command
    assert "--correct-rust-chance-spectra" in command
    assert "--no-symmetry-averaged-eval" in command


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
    assert descriptor["search_config"]["public_observation"] is True
    assert descriptor["search_config"]["information_set_search"] is True
    assert descriptor["search_config"]["c_scale"] == 0.03


def test_build_command_records_multigpu_and_explicit_denoised_arm():
    command = build_h2h_command(
        champion_checkpoint="ckpt/v3a.pt",
        n_full_a=64,
        n_full_b=128,
        pairs=10,
        h2h_out_path="/tmp/out.json",
        devices="cuda:0,cuda:1",
        symmetry_averaged_eval=True,
    )
    assert command[command.index("--devices") + 1] == "cuda:0,cuda:1"
    assert "--symmetry-averaged-eval" in command
    assert "--no-symmetry-averaged-eval" not in command


def test_build_command_records_fair_adaptive_wide_budget_by_role():
    command = build_h2h_command(
        champion_checkpoint="ckpt/v3a.pt",
        n_full_a=128,
        n_full_b=128,
        n_full_wide_b=256,
        n_full_wide_threshold_b=40,
        symmetry_averaged_eval_threshold=20,
        pairs=50,
        h2h_out_path="/tmp/adaptive.json",
    )
    assert command[command.index("--candidate-n-full-wide") + 1] == "256"
    assert command[command.index("--candidate-n-full-wide-threshold") + 1] == "40"
    assert command[command.index("--symmetry-averaged-eval-threshold") + 1] == "20"
    assert "--baseline-n-full-wide" not in command
    assert "--n-full-wide" not in command


def test_build_command_preserves_shared_wide_fallback_and_role_overrides():
    command = build_h2h_command(
        champion_checkpoint="ckpt/v3a.pt",
        n_full_a=128,
        n_full_b=128,
        n_full_wide=512,
        n_full_wide_a=128,
        n_full_wide_b=256,
        n_full_wide_threshold=40,
        n_full_wide_threshold_a=32,
        n_full_wide_threshold_b=48,
        pairs=1,
        h2h_out_path="/tmp/wide.json",
    )
    assert command[command.index("--n-full-wide") + 1] == "512"
    assert command[command.index("--candidate-n-full-wide") + 1] == "256"
    assert command[command.index("--baseline-n-full-wide") + 1] == "128"
    assert command[command.index("--n-full-wide-threshold") + 1] == "40"
    assert command[command.index("--candidate-n-full-wide-threshold") + 1] == "48"
    assert command[command.index("--baseline-n-full-wide-threshold") + 1] == "32"


def test_descriptor_hashes_resolved_role_budgets_and_is_output_path_independent():
    kwargs = dict(
        champion_checkpoint="ckpt/v3a.pt",
        n_full_a=128,
        n_full_b=128,
        n_full_wide_b=256,
        n_full_wide_threshold_b=40,
        pairs=50,
    )
    first = build_invocation_descriptor(h2h_out_path="/tmp/one.json", **kwargs)
    second = build_invocation_descriptor(h2h_out_path="/tmp/two.json", **kwargs)
    uniform = build_invocation_descriptor(
        h2h_out_path="/tmp/uniform.json",
        champion_checkpoint="ckpt/v3a.pt",
        n_full_a=128,
        n_full_b=128,
        pairs=50,
    )

    assert first["search_budgets_by_role"] == {
        "candidate": {
            "n_full": 128,
            "n_full_wide": 256,
            "n_full_wide_threshold": 40,
        },
        "baseline": {
            "n_full": 128,
            "n_full_wide": None,
            "n_full_wide_threshold": None,
        },
    }
    assert first["config_hash"] == second["config_hash"]
    assert first["full_config_hash"] == second["full_config_hash"]
    assert first["config_hash"] != uniform["config_hash"]
    assert len(first["config_hash"].split(":", 1)[1]) == 16
    assert len(first["full_config_hash"].split(":", 1)[1]) == 64


def test_descriptor_and_command_bind_d1_calibration() -> None:
    descriptor = build_invocation_descriptor(
        champion_checkpoint="ckpt/v3a.pt",
        n_full_a=128,
        n_full_b=128,
        pairs=10,
        h2h_out_path="/tmp/d1.json",
        rescale_noise_floor_c=0.25,
        sigma_eval=0.5,
    )
    command = descriptor["command"]
    assert command[command.index("--rescale-noise-floor-c") + 1] == "0.25"
    assert command[command.index("--sigma-eval") + 1] == "0.5"
    assert descriptor["search_config"]["rescale_noise_floor_c"] == 0.25
    assert descriptor["search_config"]["sigma_eval"] == 0.5

    legacy = build_invocation_descriptor(
        champion_checkpoint="ckpt/v3a.pt",
        n_full_a=128,
        n_full_b=128,
        pairs=10,
        h2h_out_path="/tmp/legacy.json",
    )
    assert descriptor["config_hash"] != legacy["config_hash"]


def test_extract_rollout_doubling_summary_pulls_expected_fields():
    fake_h2h_json = {
        "candidate_checkpoint": "ckpt/v3a.pt",
        "baseline_checkpoint": "ckpt/v3a.pt",
        "candidate_n_full": 128,
        "baseline_n_full": 64,
        "candidate_n_full_wide": 256,
        "baseline_n_full_wide": None,
        "candidate_n_full_wide_threshold": 40,
        "baseline_n_full_wide_threshold": None,
        "symmetry_averaged_eval_threshold": 20,
        "config_hash": "sha256:short",
        "full_config_hash": "sha256:full",
        "search_telemetry": {
            "candidate_over_baseline_seconds_per_call_ratio": 1.5
        },
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
    assert summary["candidate_n_full_wide"] == 256
    assert summary["baseline_n_full_wide"] is None
    assert summary["candidate_n_full_wide_threshold"] == 40
    assert summary["baseline_n_full_wide_threshold"] is None
    assert summary["symmetry_averaged_eval_threshold"] == 20
    assert summary["h2h_config_hash"] == "sha256:short"
    assert summary["h2h_full_config_hash"] == "sha256:full"
    assert summary["search_telemetry"][
        "candidate_over_baseline_seconds_per_call_ratio"
    ] == 1.5
    assert summary["pentanomial_sprt"]["decision"] == "H1"
    assert summary["pair_diagnostics"]["ww_pairs"] == 60
    assert "games" not in summary


def test_extract_rollout_doubling_summary_missing_fields_are_none():
    summary = extract_rollout_doubling_summary({})
    assert summary["candidate_win_rate"] is None
    assert summary["pentanomial_sprt"] is None
