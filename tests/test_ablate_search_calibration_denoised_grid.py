from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest


_TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from ablate_search_calibration import (  # type: ignore  # noqa: E402
    _base_search_config_kwargs,
    _load_completed_arm,
    _parse_requested_arms,
    _validate_denoise_grid_invocation,
    denoised_cscale_d1_arm_names,
    resolve_arm,
)


def _args(**overrides):
    values = {
        "arms": "denoise-grid",
        "pairs": 85,
        "n_full": 64,
        "max_depth": 80,
        "max_decisions": 600,
        "masked": True,
        "lazy": True,
        "symmetry_averaged_eval": True,
        "symmetry_averaged_eval_threshold": 20,
        "d1_c": 1.0,
        "d1_sigma_eval": 0.98,
        "d2_k": 1.0,
        "backup_weight_a": 0.25,
        "backup_weight_exp": 1.0,
        "backup_weight_cap": 1.0,
        "elo0": -10.0,
        "elo1": 15.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_denoise_grid_is_exact_cscale_by_d1_cross_without_baseline_duplication():
    assert denoised_cscale_d1_arm_names() == [
        "cv50_cs0.1",
        "cv50_cs0.3",
        "D1",
        "cv50_cs0.1+D1",
        "cv50_cs0.3+D1",
    ]
    assert _parse_requested_arms("denoise-grid") == denoised_cscale_d1_arm_names()


def test_d1_grid_arm_threads_both_cscale_and_noise_floor():
    arm = resolve_arm("cv50_cs0.3+D1", _args(d1_c=0.5, d1_sigma_eval=0.4))
    assert arm.config_overrides == {
        "c_visit": 50.0,
        "c_scale": 0.3,
        "rescale_noise_floor_c": 0.5,
        "sigma_eval": 0.4,
    }


def test_symmetry_is_shared_by_candidate_and_baseline_base_config():
    enabled = _base_search_config_kwargs(_args(symmetry_averaged_eval=True))
    disabled = _base_search_config_kwargs(
        _args(
            symmetry_averaged_eval=False,
            symmetry_averaged_eval_threshold=None,
        )
    )
    assert enabled["symmetry_averaged_eval"] is True
    assert enabled["symmetry_averaged_eval_threshold"] == 20
    assert disabled["symmetry_averaged_eval"] is False
    assert disabled["symmetry_averaged_eval_threshold"] is None


def test_denoise_grid_refuses_implicit_legacy_d6_threshold():
    with pytest.raises(ValueError, match="explicit.*threshold"):
        _validate_denoise_grid_invocation(
            _args(
                arms="denoise-grid",
                symmetry_averaged_eval=True,
                symmetry_averaged_eval_threshold=None,
            )
        )

    with pytest.raises(ValueError, match="requires --symmetry"):
        _validate_denoise_grid_invocation(
            _args(arms="denoise-grid", symmetry_averaged_eval=False)
        )

    _validate_denoise_grid_invocation(_args(arms="denoise-grid"))


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("symmetry_averaged_eval_threshold", 25, "threshold 20"),
        ("pairs", 100, "pairs 85"),
        ("n_full", 128, "n-full 64"),
        ("max_depth", 64, "max-depth 80"),
        ("max_decisions", 300, "max-decisions 600"),
        ("masked", False, "public-observation"),
        ("lazy", False, "requires --lazy"),
        ("d1_c", 0.5, "d1-c 1.0"),
        ("d1_sigma_eval", 0.79, "d1-sigma-eval 0.98"),
        ("elo1", 30.0, "SPRT bounds"),
    ],
)
def test_denoise_grid_refuses_nonbinding_protocol_drift(field, bad, message):
    with pytest.raises(ValueError, match=message):
        _validate_denoise_grid_invocation(_args(**{field: bad}))


def test_resume_accepts_only_a_fully_binding_arm(tmp_path):
    path = tmp_path / "cv50_cs0.1.json"
    valid = {
        "pairs_requested": 85,
        "games_played": 170,
        "complete_pairs": 85,
        "games_truncated": 0,
        "errors": [],
        "pentanomial_sprt": {"pairs": 85},
    }
    path.write_text(json.dumps(valid), encoding="utf-8")
    assert _load_completed_arm(tmp_path, "cv50_cs0.1", pairs=85) == valid

    for drift in (
        {"complete_pairs": 84},
        {"games_truncated": 1},
        {"errors": [{"error": "worker failed"}]},
        {"pentanomial_sprt": {"pairs": 84}},
    ):
        path.write_text(json.dumps({**valid, **drift}), encoding="utf-8")
        assert _load_completed_arm(tmp_path, "cv50_cs0.1", pairs=85) is None
