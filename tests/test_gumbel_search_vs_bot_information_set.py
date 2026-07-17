from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import gumbel_search_vs_bot_h2h as h2h  # noqa: E402
from catan_zero.rl.config_cli import load_config  # noqa: E402
from catan_zero.rl.pipeline_configs import EvalConfig  # noqa: E402


_search_config_kwargs = h2h._search_config_kwargs


def test_vs_bot_search_threads_information_set_recipe() -> None:
    config = _search_config_kwargs(
        {
            "n_full": 128,
            "max_depth": 80,
            "correct_rust_chance_spectra": True,
            "information_set_search": True,
            "determinization_particles": 4,
            "determinization_min_simulations": 32,
        }
    )
    assert config["information_set_search"] is True
    assert config["determinization_particles"] == 4
    assert config["determinization_min_simulations"] == 32


def test_vs_bot_search_threads_coherent_boundary_particle_operator() -> None:
    config = _search_config_kwargs(
        {
            "n_full": 128,
            "max_depth": 80,
            "correct_rust_chance_spectra": True,
            "coherent_public_belief_search": True,
            "boundary_value_particles": 4,
        }
    )

    assert config["coherent_public_belief_search"] is True
    assert config["boundary_value_particles"] == 4


@pytest.mark.parametrize("threshold", [None, 20])
def test_vs_bot_search_threads_symmetry_averaging_threshold(
    threshold: int | None,
) -> None:
    config = _search_config_kwargs(
        {
            "n_full": 128,
            "max_depth": 80,
            "correct_rust_chance_spectra": True,
            "symmetry_averaged_eval": True,
            "symmetry_averaged_eval_threshold": threshold,
        }
    )
    assert config["symmetry_averaged_eval"] is True
    assert config["symmetry_averaged_eval_threshold"] == threshold


def test_vs_bot_typed_config_threshold_roundtrip_and_hash_identity(
    tmp_path: Path,
) -> None:
    config = EvalConfig(
        mode="vs_bot",
        candidate="candidate.pt",
        baseline_bot="catanatron_value",
        map_kind="TOURNAMENT",
        symmetry_averaged_eval=True,
        symmetry_averaged_eval_threshold=20,
    )
    payload = config.canonical_payload()
    path = tmp_path / "eval-config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_config(path)
    assert loaded == config
    assert loaded.config_hash() == config.config_hash()
    assert (
        replace(config, symmetry_averaged_eval_threshold=None).config_hash()
        != config.config_hash()
    )
    assert (
        replace(config, native_mcts_hot_loop=True).config_hash() != config.config_hash()
    )


def test_vs_bot_summary_attests_symmetry_averaging_threshold() -> None:
    args = SimpleNamespace(
        candidate="candidate.pt",
        baseline_bot="catanatron_value",
        gate_config="flywheel",
        n_full=128,
        lazy_interior_chance=True,
        value_squash="tanh",
        c_scale=0.03,
        c_visit=50.0,
        max_root_candidates=16,
        max_root_candidates_wide=54,
        correct_rust_chance_spectra=True,
        public_observation=True,
        belief_chance_spectra=False,
        information_set_search=True,
        determinization_particles=4,
        determinization_min_simulations=32,
        symmetry_averaged_eval=True,
        symmetry_averaged_eval_threshold=20,
        native_mcts_hot_loop=True,
        elo0=0.0,
        elo1=30.0,
    )
    summary = h2h._build_summary(
        args,
        all_games=[],
        outcomes=[],
        truncated_count=0,
        divergence_count=0,
        pairs=[],
        elapsed=0.0,
        workers=1,
        threads_per_worker=1,
        errors=[],
    )
    assert summary["symmetry_averaged_eval"] is True
    assert summary["symmetry_averaged_eval_threshold"] == 20
    assert summary["native_mcts_hot_loop"] is True
    assert summary["mcts_implementation"] == "rust_native_hot_loop_v1"


def test_vs_bot_cli_threshold_reaches_dumped_typed_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "report.json"
    dumped = tmp_path / "typed-config.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gumbel_search_vs_bot_h2h.py",
            "--candidate",
            "candidate.pt",
            "--baseline-bot",
            "catanatron_value",
            "--pairs",
            "1",
            "--workers",
            "1",
            "--public-observation",
            "--coherent-public-belief-search",
            "--boundary-value-particles",
            "4",
            "--symmetry-averaged-eval",
            "--symmetry-averaged-eval-threshold",
            "20",
            "--no-evaluator-rust-featurize",
            "--dump-config",
            str(dumped),
            "--out",
            str(out),
        ],
    )
    monkeypatch.setattr(
        h2h,
        "_worker_entry",
        lambda worker_args: {
            "worker_index": worker_args["worker_index"],
            "games": [],
            "error": None,
            "pair_errors": [],
        },
    )

    h2h.main()

    config = load_config(dumped)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert config.symmetry_averaged_eval_threshold == 20
    assert config.coherent_public_belief_search is True
    assert config.boundary_value_particles == 4
    assert report["symmetry_averaged_eval_threshold"] == 20
    assert report["coherent_public_belief_search"] is True
    assert report["boundary_value_particles"] == 4
    assert report["config_hash"] == config.config_hash()
