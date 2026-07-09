"""Tests for the RGSC ranking-based regret-prioritization sampler (CAT-43).

Covers:
  * ranking math: weights/probabilities are monotonic in regret score, sum to
    1, react to temperature as expected, and degrade gracefully on an
    all-zero-regret corpus.
  * `rgsc_sample_indices` empirically favours high-regret states over many
    trials, returns no duplicates, respects `k`, and is exactly reproducible
    from a seeded `np.random.Generator`.
  * `uniform_sample_indices` is a plain without-replacement draw (no regret
    bias).
  * `split_holdout_indices` / `write_holdout_manifest` (in
    `generate_restart_selfplay`): usable/holdout partitions are disjoint and
    exhaustive, the split is deterministic for a fixed seed, and
    `select_archived_states` never draws from the held-out set.
  * `select_archived_states` in "uniform" mode (the default) reproduces the
    pre-CAT-43 behaviour exactly when no holdout is configured -- the
    regression guard for existing callers/pipelines.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import rgsc_sampler as rs  # noqa: E402
import generate_restart_selfplay as gr  # noqa: E402


# --------------------------------------------------------------------------- #
# Ranking math
# --------------------------------------------------------------------------- #
def test_rgsc_weights_monotonic_in_regret_score():
    scores = np.array([0.0, 0.1, 0.5, 1.0, 2.0], dtype=np.float64)
    weights = rs.rgsc_weights(scores, temperature=0.2)
    assert np.all(np.diff(weights) > 0), weights


def test_rgsc_probabilities_sum_to_one_and_favor_higher_regret():
    scores = np.array([0.05, 0.2, 0.9, 1.8], dtype=np.float64)
    probs = rs.rgsc_probabilities(scores, temperature=0.3)
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)
    # Highest regret gets the highest probability.
    assert np.argmax(probs) == 3
    assert np.all(np.diff(probs) > 0)


def test_rgsc_temperature_controls_concentration():
    scores = np.array([0.1, 0.5, 1.0, 3.0], dtype=np.float64)
    sharp = rs.rgsc_probabilities(scores, temperature=0.02)
    soft = rs.rgsc_probabilities(scores, temperature=1.0)
    # Lower temperature concentrates more mass on the top-scoring state.
    assert sharp[-1] > soft[-1]
    # Lower temperature is a lower-entropy distribution.
    entropy = lambda p: -np.sum(p * np.log(np.clip(p, 1e-300, None)))
    assert entropy(sharp) < entropy(soft)


def test_rgsc_probabilities_degenerate_all_zero_falls_back_to_uniform():
    scores = np.zeros(5, dtype=np.float64)
    probs = rs.rgsc_probabilities(scores, temperature=0.1)
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(probs, 0.2)


def test_rgsc_weights_rejects_nonpositive_temperature():
    with pytest.raises(ValueError):
        rs.rgsc_weights(np.array([0.1, 0.2]), temperature=0.0)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def test_rgsc_sample_indices_no_duplicates_and_respects_k():
    rng = np.random.default_rng(7)
    scores = np.linspace(0.0, 2.0, 50)
    picked = rs.rgsc_sample_indices(scores, 10, temperature=0.3, rng=rng)
    assert picked.shape[0] == 10
    assert len(set(picked.tolist())) == 10
    assert picked.max() < 50


def test_rgsc_sample_indices_k_larger_than_n_returns_all():
    rng = np.random.default_rng(3)
    scores = np.array([0.1, 0.4, 0.9])
    picked = rs.rgsc_sample_indices(scores, 10, temperature=0.2, rng=rng)
    assert sorted(picked.tolist()) == [0, 1, 2]


def test_rgsc_sample_indices_deterministic_given_seeded_rng():
    scores = np.linspace(0.0, 1.0, 30)
    picked_a = rs.rgsc_sample_indices(scores, 8, temperature=0.15, rng=np.random.default_rng(42))
    picked_b = rs.rgsc_sample_indices(scores, 8, temperature=0.15, rng=np.random.default_rng(42))
    assert np.array_equal(picked_a, picked_b)


def test_rgsc_sample_indices_favors_high_regret_over_many_trials():
    # Two-tier corpus: 10 high-regret states, 90 near-zero-regret states.
    scores = np.concatenate([np.full(10, 5.0), np.full(90, 0.01)])
    hits_in_high_tier = 0
    total_picks = 0
    for trial in range(200):
        rng = np.random.default_rng(1000 + trial)
        picked = rs.rgsc_sample_indices(scores, 10, temperature=0.2, rng=rng)
        hits_in_high_tier += int((picked < 10).sum())
        total_picks += picked.shape[0]
    # A uniform draw would put ~10% of picks in the high tier; RGSC sampling
    # should concentrate the overwhelming majority there.
    assert hits_in_high_tier / total_picks > 0.9


def test_uniform_sample_indices_no_regret_bias():
    rng = np.random.default_rng(5)
    scores = np.concatenate([np.full(10, 5.0), np.full(90, 0.01)])  # unused by uniform
    picked = rs.uniform_sample_indices(scores.shape[0], 20, rng=rng)
    assert picked.shape[0] == 20
    assert len(set(picked.tolist())) == 20


def test_mean_regret_by_rank_bucket_decreasing_for_topheavy_sample():
    scores = np.linspace(0.0, 1.0, 100)  # ascending; index 99 has highest score
    # Simulate a "good" RGSC-style sample: mostly the top quartile by index.
    selected = np.arange(75, 100)
    means = rs.mean_regret_by_rank_bucket(scores, selected, n_buckets=4)
    non_nan = [m for m in means if m == m]  # filter NaN
    assert non_nan == sorted(non_nan, reverse=True)


# --------------------------------------------------------------------------- #
# Holdout split
# --------------------------------------------------------------------------- #
def _synthetic_game_seeds_and_decisions(n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    game_seeds = rng.integers(0, 1_000_000, size=n)
    decision_indices = rng.integers(0, 600, size=n)
    return game_seeds, decision_indices


def test_split_holdout_indices_disjoint_and_exhaustive():
    game_seeds, decision_indices = _synthetic_game_seeds_and_decisions(500)
    usable, holdout = gr.split_holdout_indices(
        game_seeds, decision_indices, holdout_fraction=0.2, holdout_seed=123
    )
    assert set(usable.tolist()) & set(holdout.tolist()) == set()
    assert set(usable.tolist()) | set(holdout.tolist()) == set(range(500))
    # Roughly 20% held out (hash-based, not exact, but should be in the ballpark).
    assert 0.1 < holdout.shape[0] / 500 < 0.3


def test_split_holdout_indices_deterministic():
    game_seeds, decision_indices = _synthetic_game_seeds_and_decisions(200)
    u1, h1 = gr.split_holdout_indices(
        game_seeds, decision_indices, holdout_fraction=0.15, holdout_seed=99
    )
    u2, h2 = gr.split_holdout_indices(
        game_seeds, decision_indices, holdout_fraction=0.15, holdout_seed=99
    )
    assert np.array_equal(u1, u2)
    assert np.array_equal(h1, h2)


def test_split_holdout_indices_zero_fraction_holds_out_nothing():
    game_seeds, decision_indices = _synthetic_game_seeds_and_decisions(50)
    usable, holdout = gr.split_holdout_indices(
        game_seeds, decision_indices, holdout_fraction=0.0, holdout_seed=1
    )
    assert holdout.shape[0] == 0
    assert usable.shape[0] == 50


def test_split_holdout_indices_rejects_bad_fraction():
    game_seeds, decision_indices = _synthetic_game_seeds_and_decisions(10)
    with pytest.raises(ValueError):
        gr.split_holdout_indices(game_seeds, decision_indices, holdout_fraction=1.0, holdout_seed=1)


# --------------------------------------------------------------------------- #
# select_archived_states: uniform-mode regression + rgsc-mode + holdout wiring
# --------------------------------------------------------------------------- #
def _write_synthetic_manifest(path: Path, n: int = 60) -> None:
    """Mimics extract_regret_states._write_manifest's schema (score-sorted desc)."""
    rng = np.random.default_rng(11)
    phases = []
    for i in range(n):
        if i % 3 == 0:
            phases.append("BUILD_INITIAL_SETTLEMENT")
        elif i % 3 == 1:
            phases.append("MOVE_ROBBER")
        else:
            phases.append("ROLL")
    scores = np.sort(rng.uniform(0.01, 2.0, size=n))[::-1]  # score-sorted desc
    game_seed = np.arange(1_000_000, 1_000_000 + n, dtype=np.int64)
    decision_index = rng.integers(0, 600, size=n).astype(np.int32)
    cols = {
        "shard_id": np.zeros(n, dtype=np.int32),
        "row_index": np.arange(n, dtype=np.int32),
        "game_seed": game_seed,
        "decision_index": decision_index,
        "regret_score": scores.astype(np.float32),
        "phase": np.asarray(phases),
        "shard_paths": np.asarray(["fake_shard_0.npz"]),
    }
    np.savez(path, **cols)


def test_select_archived_states_uniform_mode_matches_legacy_topslice(tmp_path):
    manifest = tmp_path / "manifest.npz"
    _write_synthetic_manifest(manifest, n=60)
    counts = {"opening": 5, "robber_dev": 4, "random_archived": 6}
    rng = np.random.default_rng(0)
    picked = gr.select_archived_states(manifest, counts, rng=rng)  # default sampling="uniform"

    data = np.load(manifest, allow_pickle=True)
    phases = np.asarray(data["phase"]).astype(str)
    scores = np.asarray(data["regret_score"])

    # Legacy behaviour: highest-scoring rows within each phase bucket, in
    # manifest (score-sorted desc) order -- exact top-slice, no randomness.
    opening_idx = [i for i in range(60) if phases[i] == "BUILD_INITIAL_SETTLEMENT"][:5]
    expected_opening_scores = sorted([float(scores[i]) for i in opening_idx], reverse=True)
    got_opening_scores = sorted([r["regret_score"] for r in picked["opening"]], reverse=True)
    assert got_opening_scores == pytest.approx(expected_opening_scores)
    assert len(picked["robber_dev"]) == 4
    assert len(picked["random_archived"]) == 6


def test_select_archived_states_rgsc_mode_prefers_higher_regret(tmp_path):
    manifest = tmp_path / "manifest.npz"
    _write_synthetic_manifest(manifest, n=60)
    counts = {"opening": 5, "robber_dev": 4, "random_archived": 6}
    rng = np.random.default_rng(0)
    picked = gr.select_archived_states(
        manifest, counts, rng=rng, sampling="rgsc", rgsc_temperature=0.1
    )
    data = np.load(manifest, allow_pickle=True)
    scores = np.asarray(data["regret_score"])
    all_mean = float(scores.mean())
    random_archived_scores = [r["regret_score"] for r in picked["random_archived"]]
    # RGSC-mode random_archived should skew well above the corpus mean.
    assert np.mean(random_archived_scores) > all_mean


def test_select_archived_states_excludes_holdout_rows(tmp_path):
    manifest = tmp_path / "manifest.npz"
    _write_synthetic_manifest(manifest, n=60)
    data = np.load(manifest, allow_pickle=True)
    usable_idx, holdout_idx = gr.split_holdout_indices(
        np.asarray(data["game_seed"]), np.asarray(data["decision_index"]),
        holdout_fraction=0.3, holdout_seed=5,
    )
    holdout_seeds = set(int(s) for s in np.asarray(data["game_seed"])[holdout_idx])
    counts = {"opening": 5, "robber_dev": 4, "random_archived": 10}
    for sampling in ("uniform", "rgsc"):
        rng = np.random.default_rng(0)
        picked = gr.select_archived_states(
            manifest, counts, rng=rng, sampling=sampling, usable_idx=usable_idx
        )
        for bucket_rows in picked.values():
            for row in bucket_rows:
                assert row["game_seed"] not in holdout_seeds


def test_write_holdout_manifest_roundtrip(tmp_path):
    manifest = tmp_path / "manifest.npz"
    _write_synthetic_manifest(manifest, n=40)
    data = np.load(manifest, allow_pickle=True)
    _usable, holdout_idx = gr.split_holdout_indices(
        np.asarray(data["game_seed"]), np.asarray(data["decision_index"]),
        holdout_fraction=0.25, holdout_seed=2,
    )
    out_path = tmp_path / "holdout_manifest.npz"
    gr.write_holdout_manifest(
        out_path, data, holdout_idx, holdout_fraction=0.25, holdout_seed=2
    )
    loaded = np.load(out_path, allow_pickle=True)
    assert loaded["game_seed"].shape[0] == holdout_idx.shape[0]
    assert set(loaded["game_seed"].tolist()) == set(np.asarray(data["game_seed"])[holdout_idx].tolist())
    assert float(loaded["holdout_fraction"]) == pytest.approx(0.25)
    assert int(loaded["holdout_seed"]) == 2
