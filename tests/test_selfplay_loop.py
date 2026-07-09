"""BUG-4: the flywheel loop's generation command must pass the full production
search/config recipe EXPLICITLY, never relying on the gen script's (wrong)
defaults. This is the CAT-88 silent-default class -- bare defaults would emit
omniscient (no --public-observation) data at c_scale 0.1 with a half-length
temperature schedule and 65x more leaf evals.
"""
from __future__ import annotations

from types import SimpleNamespace

from tools.selfplay_loop import _build_generation_cmd
from pathlib import Path


def _flag_value(cmd: list[str], flag: str) -> str:
    idx = cmd.index(flag)
    return cmd[idx + 1]


def _make_args() -> SimpleNamespace:
    return SimpleNamespace(
        games_per_gen=1500,
        workers=16,
        device="cuda",
        base_seed=500_000_000_000,
    )


def test_generation_cmd_contains_critical_production_flags() -> None:
    cmd = _build_generation_cmd(Path("/tmp/out"), "/ckpt.pt", 0, _make_args())

    # Valued flags that are WRONG at the gen-script default:
    assert _flag_value(cmd, "--c-scale") == "0.03"           # default 0.1 is worse
    assert _flag_value(cmd, "--temperature-decisions") == "90"  # default 45 = half schedule
    assert _flag_value(cmd, "--max-decisions") == "600"

    # Store-true flags absent at default that MUST be present:
    assert "--public-observation" in cmd    # else omniscient / hidden-info leak
    assert "--lazy-interior-chance" in cmd   # else ~65x more leaf evals
    assert "--correct-rust-chance-spectra" in cmd
    assert "--score-actions" in cmd

    # Recipe completeness (mirrors the live H100 fleet command):
    assert _flag_value(cmd, "--n-full") == "64"
    assert _flag_value(cmd, "--n-fast") == "16"
    assert _flag_value(cmd, "--p-full") == "0.25"
    assert _flag_value(cmd, "--c-visit") == "50.0"
    assert _flag_value(cmd, "--max-depth") == "80"
    assert _flag_value(cmd, "--track") == "2p_no_trade"
    assert _flag_value(cmd, "--vps-to-win") == "10"
    assert _flag_value(cmd, "--shard-size") == "2048"
    assert _flag_value(cmd, "--format") == "npz"


def test_generation_cmd_threads_caller_params() -> None:
    args = _make_args()
    cmd = _build_generation_cmd(Path("/tmp/out"), "/ckpt.pt", 3, args)
    assert _flag_value(cmd, "--games") == "1500"
    assert _flag_value(cmd, "--workers") == "16"
    assert _flag_value(cmd, "--device") == "cuda"
    assert _flag_value(cmd, "--checkpoint") == "/ckpt.pt"
    # gen_index-scaled disjoint seed block
    assert _flag_value(cmd, "--base-seed") == str(500_000_000_000 + 3 * 10_000_019)
