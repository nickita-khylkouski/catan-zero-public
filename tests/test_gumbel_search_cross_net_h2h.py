"""Tests for the CAT-25 rollout-doubling extension to
tools/gumbel_search_cross_net_h2h.py: `_build_search_config` accepting an
explicit `n_full` override, and worker_args plumbing for
`candidate_n_full` / `baseline_n_full`. Pure argument-plumbing -- no
GPU/checkpoint/rust dependency needed."""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from gumbel_search_cross_net_h2h import _build_search_config  # type: ignore  # noqa: E402


def _base_worker_args(**overrides) -> dict:
    args = {
        "n_full": 64,
        "max_depth": 80,
        "correct_rust_chance_spectra": True,
    }
    args.update(overrides)
    return args


def test_build_search_config_defaults_to_shared_n_full_when_no_override():
    worker_args = _base_worker_args()
    config = _build_search_config(worker_args, seed=1)
    assert config.n_full == 64
    assert config.n_fast == 64


def test_build_search_config_explicit_n_full_overrides_shared_value():
    worker_args = _base_worker_args()
    config = _build_search_config(worker_args, seed=1, n_full=128)
    assert config.n_full == 128
    assert config.n_fast == 128


def test_build_search_config_none_n_full_falls_back_to_shared_value():
    worker_args = _base_worker_args()
    config = _build_search_config(worker_args, seed=1, n_full=None)
    assert config.n_full == 64


def test_worker_args_resolution_uses_candidate_and_baseline_keys_when_present():
    """Mirrors the resolution logic _run_worker applies: worker_args.get(
    'candidate_n_full', worker_args['n_full']) / same for baseline."""
    worker_args = _base_worker_args(candidate_n_full=128, baseline_n_full=64)

    candidate_n_full = int(worker_args.get("candidate_n_full", worker_args["n_full"]))
    baseline_n_full = int(worker_args.get("baseline_n_full", worker_args["n_full"]))

    candidate_config = _build_search_config(worker_args, seed=1, n_full=candidate_n_full)
    baseline_config = _build_search_config(worker_args, seed=1, n_full=baseline_n_full)

    assert candidate_config.n_full == 128
    assert baseline_config.n_full == 64


def test_worker_args_resolution_omits_keys_falls_back_to_shared_n_full():
    """Every existing caller of this tool never sets candidate_n_full/
    baseline_n_full -- both roles must resolve to the shared --n-full,
    byte-identical to pre-extension behavior."""
    worker_args = _base_worker_args(n_full=64)

    candidate_n_full = int(worker_args.get("candidate_n_full", worker_args["n_full"]))
    baseline_n_full = int(worker_args.get("baseline_n_full", worker_args["n_full"]))

    assert candidate_n_full == 64
    assert baseline_n_full == 64

    candidate_config = _build_search_config(worker_args, seed=1, n_full=candidate_n_full)
    baseline_config = _build_search_config(worker_args, seed=1, n_full=baseline_n_full)
    assert candidate_config.n_full == baseline_config.n_full == 64


def test_build_search_config_preserves_other_fields_regardless_of_n_full_override():
    worker_args = _base_worker_args(
        n_full=64,
        c_scale=0.2,
        c_visit=10.0,
        max_root_candidates=8,
    )
    config = _build_search_config(worker_args, seed=7, n_full=128)
    assert config.c_scale == 0.2
    assert config.c_visit == 10.0
    assert config.max_root_candidates == 8
    assert config.seed == 7
