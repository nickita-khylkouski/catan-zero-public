from __future__ import annotations

import sys
from pathlib import Path

# `tools/generate_gumbel_selfplay_data.py` does bare sibling imports
# (`from factory_common import ...`), so it only works with `tools/` itself on
# sys.path (matches the bootstrap pattern in tests/test_gumbel_self_play.py and
# tests/test_generate_gumbel_selfplay_data.py).
_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import generate_gumbel_selfplay_data as cli  # type: ignore  # noqa: E402


def _worker_args(**overrides) -> dict:
    """A complete `_run_worker` worker_args dict (checkpoint=None -> the cheap
    HeuristicRustEvaluator path, no rust wheel / neural checkpoint required)."""
    values = {
        "worker_index": 0,
        "games": 1,
        "game_index_start": 0,
        "out_dir": "/tmp/does-not-matter",
        "checkpoint": None,
        "device": "cpu",
        "n_full": 4,
        "n_fast": 2,
        "p_full": 1.0,
        "c_visit": 50.0,
        "c_scale": 0.1,
        "rescale_noise_floor_c": 0.0,
        "sigma_eval": 0.79,
        "n_full_wide": None,
        "n_full_wide_threshold": None,
        "wide_roots_always_full": False,
        "raw_policy_above_width": None,
        "symmetry_averaged_eval": False,
        "symmetry_averaged_eval_threshold": None,
        "wide_candidates_threshold": 24,
        "max_decisions": 600,
        "max_depth": 80,
        "temperature_move_fraction": 0.075,
        "temperature_high": 1.0,
        "temperature_low": 0.0,
        "late_temperature_move_fraction": None,
        "late_temperature": 0.0,
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
        "opponent_pool_manifest": None,
    }
    values.update(overrides)
    return values


def _capture_configs(monkeypatch):
    """Monkeypatch `run_worker_games` to capture the `config`/`search_config`
    dataclasses `_run_worker` builds, without actually playing any games."""
    captured: dict = {}

    def _fake_run_worker_games(**kwargs):
        captured["config"] = kwargs["config"]
        captured["search_config"] = kwargs["search_config"]
        return {
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
    return captured


# --------------------------------------------------------------------------- D1 noise-floor wiring


def test_rescale_noise_floor_c_default_is_the_dataclass_no_op(monkeypatch) -> None:
    captured = _capture_configs(monkeypatch)
    cli._run_worker(_worker_args())

    assert captured["search_config"].rescale_noise_floor_c == 0.0
    assert captured["search_config"].sigma_eval == 0.79


def test_rescale_noise_floor_c_and_sigma_eval_thread_through_from_worker_args(monkeypatch) -> None:
    captured = _capture_configs(monkeypatch)
    cli._run_worker(_worker_args(rescale_noise_floor_c=1.0, sigma_eval=0.5))

    assert captured["search_config"].rescale_noise_floor_c == 1.0
    assert captured["search_config"].sigma_eval == 0.5


# --------------------------------------------------------------------------- D6 root denoising wiring


def test_symmetry_averaging_defaults_to_off_with_canonical_wide_threshold(monkeypatch) -> None:
    captured = _capture_configs(monkeypatch)
    cli._run_worker(_worker_args())

    assert captured["search_config"].symmetry_averaged_eval is False
    assert captured["search_config"].symmetry_averaged_eval_threshold is None
    assert captured["search_config"].wide_candidates_threshold == 24


def test_symmetry_averaging_and_wide_threshold_thread_through(monkeypatch) -> None:
    captured = _capture_configs(monkeypatch)
    cli._run_worker(
        _worker_args(
            symmetry_averaged_eval=True,
            symmetry_averaged_eval_threshold=20,
            wide_candidates_threshold=24,
        )
    )

    assert captured["search_config"].symmetry_averaged_eval is True
    assert captured["search_config"].symmetry_averaged_eval_threshold == 20
    assert captured["search_config"].wide_candidates_threshold == 24


def test_adaptive_wide_budget_threshold_and_always_full_thread_through(monkeypatch) -> None:
    captured = _capture_configs(monkeypatch)
    cli._run_worker(
        _worker_args(
            p_full=0.25,
            n_full_wide=256,
            n_full_wide_threshold=40,
            wide_roots_always_full=True,
        )
    )

    search = captured["search_config"]
    assert search.p_full == 0.25
    assert search.n_full_wide == 256
    assert search.n_full_wide_threshold == 40
    assert search.wide_roots_always_full is True


# --------------------------------------------------------------------------- late-temperature wiring


def test_late_temperature_defaults_are_the_dataclass_no_op(monkeypatch) -> None:
    captured = _capture_configs(monkeypatch)
    cli._run_worker(_worker_args())

    assert captured["config"].late_temperature_move_fraction is None
    assert captured["config"].late_temperature == 0.0


def test_late_temperature_threads_through_from_worker_args(monkeypatch) -> None:
    captured = _capture_configs(monkeypatch)
    cli._run_worker(
        _worker_args(late_temperature_move_fraction=0.25, late_temperature=0.3)
    )

    assert captured["config"].late_temperature_move_fraction == 0.25
    assert captured["config"].late_temperature == 0.3
