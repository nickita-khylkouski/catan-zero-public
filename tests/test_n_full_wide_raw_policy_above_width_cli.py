"""Task #76 pre-wiring: --n-full-wide / --raw-policy-above-width CLI flags
thread through to GumbelChanceMCTSConfig, matching the H2H tool's identical
flags (tools/gumbel_search_vs_raw_h2h.py) -- needed so gen-1 generation can
replicate whichever confirmation-H2H arm wins without a last-minute CLI gap.
Both default to None (no-op, matches GumbelChanceMCTSConfig's own defaults).
"""
from __future__ import annotations

import sys
from pathlib import Path

from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTSConfig

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import generate_gumbel_selfplay_data as cli  # type: ignore  # noqa: E402


def _base_worker_args(**overrides):
    values = {
        "worker_index": 0,
        "games": 1,
        "game_index_start": 0,
        "out_dir": "/tmp/does-not-matter-n-full-wide-test",
        "checkpoint": None,
        "device": "cpu",
        "n_full": 4,
        "n_fast": 2,
        "p_full": 1.0,
        "c_visit": 50.0,
        "c_scale": 0.1,
        "max_decisions": 4,
        "max_depth": 40,
        "temperature_move_fraction": 0.15,
        "temperature_high": 1.0,
        "temperature_low": 0.0,
        "prior_temperature": 1.0,
        "value_scale": 1.0,
        "track": "2p_no_trade",
        "vps_to_win": 10,
        "obs_width": 806,
        "base_seed": 1,
        "worker_seed": 1,
        "shard_size": 2048,
        "format": "npz",
        "score_actions": False,
        "correct_rust_chance_spectra": True,
        "lazy_interior_chance": False,
        "public_observation": False,
        "belief_chance_spectra": False,
        "n_full_wide": None,
        "raw_policy_above_width": None,
    }
    values.update(overrides)
    return values


def _captured_search_config(monkeypatch, worker_args):
    captured: dict[str, object] = {}

    def _fake_run_worker_games(**kwargs):
        captured["search_config"] = kwargs["search_config"]
        return {
            "out_dir": str(kwargs["out_dir"]),
            "games_completed": 0,
            "games_failed": 0,
            "games_truncated": 0,
            "rows": 0,
            "decisions_total": 0,
            "forced_decisions_total": 0,
            "simulations_used_total": 0,
            "wins_by_color": {},
            "shards": [],
            "errors": [],
        }

    monkeypatch.setattr(cli, "run_worker_games", _fake_run_worker_games)
    cli._run_worker(worker_args)
    return captured["search_config"]


def test_n_full_wide_none_by_default(monkeypatch):
    search_config = _captured_search_config(monkeypatch, _base_worker_args())
    assert search_config.n_full_wide is None


def test_n_full_wide_threads_through_when_set(monkeypatch):
    search_config = _captured_search_config(
        monkeypatch, _base_worker_args(n_full_wide=512)
    )
    assert search_config.n_full_wide == 512


def test_raw_policy_above_width_none_by_default(monkeypatch):
    search_config = _captured_search_config(monkeypatch, _base_worker_args())
    assert search_config.raw_policy_above_width is None


def test_raw_policy_above_width_threads_through_when_set(monkeypatch):
    search_config = _captured_search_config(
        monkeypatch, _base_worker_args(raw_policy_above_width=40)
    )
    assert search_config.raw_policy_above_width == 40


def test_both_flags_together():
    """Direct construction sanity check -- GumbelChanceMCTSConfig itself
    accepts both fields (guards against a stale local copy of the dataclass)."""
    config = GumbelChanceMCTSConfig(n_full_wide=256, raw_policy_above_width=30)
    assert config.n_full_wide == 256
    assert config.raw_policy_above_width == 30


def test_cli_args_include_both_new_flags_by_default(monkeypatch):
    """--n-full-wide/--raw-policy-above-width must exist as real argparse
    flags (not just worker_args keys) so a real launch command can set them."""
    monkeypatch.setattr(sys, "argv", ["generate_gumbel_selfplay_data.py", "--out-dir", "/tmp/x", "--help"])
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            cli.main()
        except SystemExit:
            pass
    help_text = buf.getvalue()
    assert "--n-full-wide" in help_text
    assert "--raw-policy-above-width" in help_text
