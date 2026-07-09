"""Manifest provenance for `--rust-featurize` (task #81, speed-czar's ask):
`_merge_worker_summaries`'s top-level summary dict must record `rust_featurize`
as its own field -- the same way `exact_budget_sh` already does, and matching
the other regime toggles (`lazy_interior_chance`, `public_observation`) -- not
buried only inside the catch-all `cli_args` dict, so a shard batch is auditable
by regime at a glance.
"""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import generate_gumbel_selfplay_data as cli  # type: ignore  # noqa: E402


def _minimal_args(*, rust_featurize: bool) -> Namespace:
    return Namespace(
        track="2p_no_trade",
        vps_to_win=10,
        games=1,
        n_full=8,
        n_fast=4,
        p_full=0.25,
        correct_rust_chance_spectra=True,
        lazy_interior_chance=True,
        exact_budget_sh=False,
        exact_budget_sh_min_n=0,  # matches the CLI's own default
        rust_featurize=rust_featurize,
        checkpoint=None,
        base_seed=1,
        opponent_pool_manifest=None,
    )


def test_top_level_summary_records_rust_featurize_true(tmp_path):
    summary = cli._merge_worker_summaries(
        [], out_dir=tmp_path, elapsed_sec=1.0, args=_minimal_args(rust_featurize=True)
    )
    assert "rust_featurize" in summary, (
        "rust_featurize must be its own top-level manifest field, like "
        "exact_budget_sh / lazy_interior_chance, not only nested inside cli_args"
    )
    assert summary["rust_featurize"] is True


def test_top_level_summary_records_rust_featurize_false(tmp_path):
    summary = cli._merge_worker_summaries(
        [], out_dir=tmp_path, elapsed_sec=1.0, args=_minimal_args(rust_featurize=False)
    )
    assert summary["rust_featurize"] is False


def test_top_level_summary_rust_featurize_matches_exact_budget_sh_pattern(tmp_path):
    """Both flags describe the same class of thing (a search/featurize-path
    regime toggle recorded for shard-batch auditability) -- they should be
    siblings in the summary dict, both present alongside each other."""
    summary = cli._merge_worker_summaries(
        [], out_dir=tmp_path, elapsed_sec=1.0, args=_minimal_args(rust_featurize=True)
    )
    assert "exact_budget_sh" in summary
    assert "rust_featurize" in summary


def test_top_level_summary_rust_featurize_is_sibling_of_other_regime_toggles(tmp_path):
    """rust_featurize is a featurize-path regime toggle recorded for
    shard-batch auditability -- it should be a top-level sibling of the other
    regime toggles (lazy_interior_chance, public_observation), not buried only
    inside cli_args."""
    summary = cli._merge_worker_summaries(
        [], out_dir=tmp_path, elapsed_sec=1.0, args=_minimal_args(rust_featurize=True)
    )
    assert "lazy_interior_chance" in summary
    assert "rust_featurize" in summary
