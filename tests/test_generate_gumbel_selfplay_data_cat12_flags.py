from __future__ import annotations

import sys
from pathlib import Path

import pytest

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
        "sigma_reference_visits": None,
        "rescale_noise_floor_c": 0.0,
        "rescale_noise_floor_initial_road_only": False,
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
        "temperature_clock": "prompt",
        "temperature_high": 1.0,
        "temperature_low": 0.0,
        "late_temperature_move_fraction": None,
        "late_temperature": 0.0,
        "prior_temperature": 1.0,
        "value_scale": 1.0,
        "track": "2p_no_trade",
        "vps_to_win": 10,
        "obs_width": 806,
        "meaningful_public_history": False,
        "learner_entity_feature_adapter_version": (
            cli.CURRENT_RUST_ENTITY_ADAPTER_VERSION
        ),
        "event_history_limit": 64,
        "record_automatic_transitions": True,
        "base_seed": 1,
        "worker_seed": 1,
        "shard_size": 2048,
        "format": "npz",
        "score_actions": False,
        "correct_rust_chance_spectra": True,
        "lazy_interior_chance": False,
        "public_observation": False,
        "belief_chance_spectra": False,
        "information_set_search": False,
        "coherent_public_belief_search": False,
        "forced_root_target_mode": "full",
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
        captured["resume_semantics_sha256"] = kwargs.get("resume_semantics_sha256")
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


def test_worker_forwards_full_resume_semantics_digest(monkeypatch) -> None:
    captured = _capture_configs(monkeypatch)
    full_digest = "sha256:" + "d" * 64
    cli._run_worker(
        _worker_args(
            run_id="sha256:" + "1" * 16,
            resume_semantics_sha256=full_digest,
        )
    )
    assert captured["resume_semantics_sha256"] == full_digest


def test_public_award_provenance_attests_python_producer() -> None:
    assert cli._public_award_feature_provenance(rust_featurize=False) == {
        "schema_version": "public-award-feature-provenance-v1",
        "contract": "authoritative_v1",
        "feature_producer": "python_snapshot_public_award_v1",
        "native_capability": None,
    }


def test_public_award_provenance_rejects_stale_native_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "catanatron_rs",
        type(
            "StaleWheel", (), {"gumbel_search_capabilities": staticmethod(lambda: [])}
        ),
    )
    with pytest.raises(RuntimeError, match="public_award_feature_parity"):
        cli._public_award_feature_provenance(rust_featurize=True)


# --------------------------------------------------------------------------- D1 noise-floor wiring


def test_rescale_noise_floor_c_default_is_the_dataclass_no_op(monkeypatch) -> None:
    captured = _capture_configs(monkeypatch)
    cli._run_worker(_worker_args())

    assert captured["search_config"].rescale_noise_floor_c == 0.0
    assert captured["search_config"].rescale_noise_floor_initial_road_only is False
    assert captured["search_config"].sigma_eval == 0.79


def test_rescale_noise_floor_c_and_sigma_eval_thread_through_from_worker_args(
    monkeypatch,
) -> None:
    captured = _capture_configs(monkeypatch)
    cli._run_worker(_worker_args(rescale_noise_floor_c=1.0, sigma_eval=0.5))

    assert captured["search_config"].rescale_noise_floor_c == 1.0
    assert captured["search_config"].sigma_eval == 0.5


def test_initial_road_only_noise_floor_threads_through_worker_args(monkeypatch) -> None:
    captured = _capture_configs(monkeypatch)
    cli._run_worker(
        _worker_args(
            rescale_noise_floor_c=8.0,
            rescale_noise_floor_initial_road_only=True,
        )
    )

    assert captured["search_config"].rescale_noise_floor_c == 8.0
    assert captured["search_config"].rescale_noise_floor_initial_road_only is True


def test_sigma_reference_visits_threads_through_worker_args(monkeypatch) -> None:
    captured = _capture_configs(monkeypatch)
    cli._run_worker(_worker_args(sigma_reference_visits=17))

    assert captured["search_config"].sigma_reference_visits == 17


def test_belief_target_aggregation_threads_through_worker_args(monkeypatch) -> None:
    captured = _capture_configs(monkeypatch)
    cli._run_worker(
        _worker_args(
            information_set_search=True,
            information_set_target_aggregation="aggregate_q_then_improve",
            sigma_reference_visits=8,
        )
    )
    config = captured["search_config"]
    assert config.information_set_target_aggregation == "aggregate_q_then_improve"
    assert config.sigma_reference_visits == 8


# --------------------------------------------------------------------------- D6 root denoising wiring


def test_symmetry_averaging_defaults_to_off_with_canonical_wide_threshold(
    monkeypatch,
) -> None:
    captured = _capture_configs(monkeypatch)
    cli._run_worker(_worker_args())

    assert captured["search_config"].symmetry_averaged_eval is False
    assert captured["search_config"].symmetry_averaged_eval_threshold is None
    assert captured["search_config"].wide_candidates_threshold == 24


def test_symmetry_averaging_and_wide_threshold_thread_through(monkeypatch) -> None:
    captured = _capture_configs(monkeypatch)

    class _SymmetryCapableEvaluator:
        def evaluate_symmetry_averaged(self, *args, **kwargs):
            raise AssertionError("configuration wiring test must not evaluate")

    cli._run_worker(
        _worker_args(
            symmetry_averaged_eval=True,
            symmetry_averaged_eval_threshold=20,
            wide_candidates_threshold=24,
        ),
        champion_evaluator=_SymmetryCapableEvaluator(),
    )

    assert captured["search_config"].symmetry_averaged_eval is True
    assert captured["search_config"].symmetry_averaged_eval_threshold == 20
    assert captured["search_config"].wide_candidates_threshold == 24


def test_symmetry_averaging_fails_closed_for_incapable_evaluator(monkeypatch) -> None:
    """A generation manifest may not claim D6 when the evaluator would no-op."""

    _capture_configs(monkeypatch)
    with pytest.raises(ValueError, match="symmetry-capable neural evaluator"):
        cli._run_worker(
            _worker_args(
                symmetry_averaged_eval=True,
                symmetry_averaged_eval_threshold=20,
            )
        )


def test_adaptive_wide_budget_threshold_and_always_full_thread_through(
    monkeypatch,
) -> None:
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


def test_science_validation_accepts_adaptive_n256_by_expanding_particles() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--out-dir",
            "/tmp/adaptive-p8",
            "--n-full",
            "128",
            "--n-full-wide",
            "256",
            "--wide-roots-always-full",
            "--public-observation",
            "--information-set-search",
            "--determinization-particles",
            "8",
            "--determinization-min-simulations",
            "32",
        ]
    )
    cli._validate_science_args(args, parser)


def test_science_validation_rejects_adaptive_n256_that_deepens_each_particle() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--out-dir",
            "/tmp/adaptive-p4",
            "--n-full",
            "128",
            "--n-full-wide",
            "256",
            "--wide-roots-always-full",
            "--public-observation",
            "--information-set-search",
            "--determinization-particles",
            "4",
            "--determinization-min-simulations",
            "32",
        ]
    )
    with pytest.raises(SystemExit):
        cli._validate_science_args(args, parser)


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


def test_strategic_clock_and_forced_trajectory_mode_thread_through(monkeypatch) -> None:
    captured = _capture_configs(monkeypatch)
    cli._run_worker(
        _worker_args(
            temperature_clock="nonforced_choice",
            coherent_public_belief_search=True,
            boundary_value_particles=4,
            forced_root_target_mode="trajectory_only",
            meaningful_public_history=True,
            event_history_limit=32,
            record_automatic_transitions=False,
        )
    )

    assert captured["config"].temperature_clock == "nonforced_choice"
    assert captured["config"].meaningful_public_history is True
    assert captured["config"].event_history_limit == 32
    assert captured["config"].record_automatic_transitions is False
    assert captured["search_config"].coherent_public_belief_search is True
    assert captured["search_config"].information_set_search is False
    assert captured["search_config"].boundary_value_particles == 4
    assert captured["search_config"].forced_root_target_mode == "trajectory_only"


def test_coherent_public_belief_accepts_one_full_budget_tree() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--out-dir",
            "/tmp/coherent-n128-wide256",
            "--n-full",
            "128",
            "--n-full-wide",
            "256",
            "--wide-roots-always-full",
            "--public-observation",
            "--coherent-public-belief-search",
            "--no-information-set-search",
        ]
    )
    cli._validate_science_args(args, parser)


def test_target_reliability_audit_preserves_adopted_legacy_n128_budget() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--out-dir",
            "/tmp/coherent-n128-reliability",
            "--n-full",
            "128",
            "--public-observation",
            "--coherent-public-belief-search",
            "--no-information-set-search",
            "--no-exact-budget-sh",
            "--exact-budget-sh-min-n",
            "0",
            "--target-reliability-audit-fraction",
            "0.05",
            "--target-reliability-audit-seed",
            "20260716",
        ]
    )

    cli._validate_science_args(args, parser)
    assert args.exact_budget_sh is False
    assert args.target_reliability_audit_fraction == pytest.approx(0.05)
