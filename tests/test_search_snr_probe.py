from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def _subprocess_env() -> dict[str, str]:
    """subprocess.run does not inherit pytest's `pythonpath = ["src"]` ini
    setting (that only mutates sys.path in-process), so any CLI-level
    subprocess test must set PYTHONPATH explicitly or the `catan_zero`
    package import inside factory_common.py fails."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    parts = [str(_SRC_DIR)] + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env

from search_snr_probe import (  # type: ignore  # noqa: E402
    aggregate_per_state_records,
    build_checkpoint_report,
    compute_state_metrics,
    main,
)


def test_compute_state_metrics_identical_policies_agree_and_zero_kl():
    pi = {0: 0.7, 1: 0.2, 2: 0.1}
    prior = {0: 0.5, 1: 0.3, 2: 0.2}
    metrics = compute_state_metrics(pi, prior, dict(pi), dict(prior))
    assert metrics["argmax_agreement"] is True
    assert metrics["kl_pi1_pi2"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["kl_pi2_pi1"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["kl_pi1_pi2_mean"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["priors_match"] is True
    assert metrics["kl_pi_vs_prior_mean"] >= 0.0


def test_compute_state_metrics_disagreeing_policies():
    pi1 = {0: 0.9, 1: 0.1}
    pi2 = {0: 0.1, 1: 0.9}
    prior = {0: 0.5, 1: 0.5}
    metrics = compute_state_metrics(pi1, prior, pi2, dict(prior))
    assert metrics["argmax_agreement"] is False
    assert metrics["kl_pi1_pi2"] > 0.0
    assert metrics["kl_pi2_pi1"] > 0.0


def test_compute_state_metrics_flags_prior_mismatch():
    pi1 = {0: 0.7, 1: 0.3}
    pi2 = {0: 0.7, 1: 0.3}
    prior1 = {0: 0.5, 1: 0.5}
    prior2 = {0: 0.9, 1: 0.1}
    metrics = compute_state_metrics(pi1, prior1, pi2, prior2)
    assert metrics["priors_match"] is False


def test_aggregate_per_state_records_agreement_rate_bounds():
    per_state = [
        {"argmax_agreement": True, "kl_pi1_pi2_mean": 0.1, "kl_pi_vs_prior_mean": 0.2, "priors_match": True},
        {"argmax_agreement": False, "kl_pi1_pi2_mean": 0.3, "kl_pi_vs_prior_mean": 0.4, "priors_match": True},
        {"argmax_agreement": True, "kl_pi1_pi2_mean": 0.05, "kl_pi_vs_prior_mean": 0.15, "priors_match": False},
    ]
    agg = aggregate_per_state_records(per_state)
    assert agg["n_states"] == 3
    assert 0.0 <= agg["mean_argmax_agreement"] <= 1.0
    assert agg["mean_argmax_agreement"] == pytest.approx(2 / 3)
    assert agg["mean_kl_pi1_pi2"] >= 0.0
    assert agg["mean_kl_pi_vs_prior"] >= 0.0
    assert agg["priors_mismatch_count"] == 1


def test_aggregate_per_state_records_empty_is_safe():
    agg = aggregate_per_state_records([])
    assert agg["n_states"] == 0
    assert agg["mean_argmax_agreement"] is None


def test_build_checkpoint_report_groups_correctly():
    per_state = [
        {"argmax_agreement": True, "kl_pi1_pi2_mean": 0.1, "kl_pi_vs_prior_mean": 0.2, "priors_match": True},
    ]
    report = build_checkpoint_report("ckpt-a", per_state)
    assert report["checkpoint"] == "ckpt-a"
    assert report["aggregate"]["n_states"] == 1
    assert report["per_state"] == per_state


def test_cli_dry_run_produces_parseable_json_with_expected_keys(tmp_path):
    out_path = tmp_path / "snr_probe.json"
    argv = [
        "search_snr_probe.py",
        "--dry-run",
        "--n-states",
        "5",
        "--out",
        str(out_path),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = old_argv

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["measurement"] == "search_snr_probe"
    assert data["dry_run"] is True
    assert data["search_config"]["public_observation"] is False
    assert "per_checkpoint" in data
    assert len(data["checkpoints"]) >= 1
    for checkpoint in data["checkpoints"]:
        report = data["per_checkpoint"][checkpoint]
        assert "aggregate" in report
        assert "per_state" in report
        assert report["aggregate"]["n_states"] == 5


def test_cli_dry_run_via_subprocess(tmp_path):
    out_path = tmp_path / "snr_probe_subproc.json"
    script = _TOOLS_DIR / "search_snr_probe.py"
    result = subprocess.run(
        [sys.executable, str(script), "--dry-run", "--n-states", "3", "--out", str(out_path)],
        capture_output=True,
        text=True,
        timeout=60,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out_path.read_text())
    assert data["measurement"] == "search_snr_probe"


def test_cli_help_does_not_crash():
    script = _TOOLS_DIR / "search_snr_probe.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        env=_subprocess_env(),
    )
    assert result.returncode == 0


def test_cli_dry_run_records_explicit_production_regime(tmp_path):
    out_path = tmp_path / "snr_prod_regime.json"
    old_argv = sys.argv
    sys.argv = [
        "search_snr_probe.py",
        "--dry-run",
        "--public-observation",
        "--information-set-search",
        "--determinization-particles",
        "4",
        "--determinization-min-simulations",
        "32",
        "--lazy-interior-chance",
        "--rust-featurize",
        "--c-scale",
        "0.03",
        "--symmetry-averaged-eval",
        "--wide-candidates-threshold",
        "20",
        "--out",
        str(out_path),
    ]
    try:
        main()
    finally:
        sys.argv = old_argv
    cfg = json.loads(out_path.read_text())["search_config"]
    assert cfg["public_observation"] is True
    assert cfg["information_set_search"] is True
    assert cfg["determinization_particles"] == 4
    assert cfg["determinization_min_simulations"] == 32
    assert cfg["lazy_interior_chance"] is True
    assert cfg["rust_featurize"] is True
    assert cfg["c_scale"] == pytest.approx(0.03)
    assert cfg["symmetry_averaged_eval"] is True
    assert cfg["wide_candidates_threshold"] == 20
