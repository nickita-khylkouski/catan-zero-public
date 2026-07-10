"""Tests for the independent gen-1 pilot-shard audit tool.

Builds tiny synthetic `.npz` shards by hand (matching the schema documented
in `gumbel_self_play.py`'s `BASE_KEYS`/`EXTRA_KEYS`, but without importing
that module -- the audit tool is deliberately independent of the driver it
audits, and these tests exercise that independence directly) to validate
each check both accepts good data and catches injected violations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from audit_gumbel_pilot_shards import (  # noqa: E402
    PLAYER_NAMES,
    check_afterstate_targets_on_forced_roll,
    check_config_provenance,
    check_game_seed_range,
    check_kl_improved_vs_prior,
    check_label_perspective,
    check_mix,
    check_prior_policy_nondegenerate,
    check_target_policy_sums_to_one,
    check_truncation_rate,
    check_weight_multipliers,
    load_rows,
    run_audit,
)

ROLL_ACTION_ID = 331  # ActionCatalog(("RED","BLUE")).describe(331) == ROLL, confirmed against the live catalog.


def _seat(name: str) -> int:
    return PLAYER_NAMES.index(name)


def _vps(winner: str | None, *, winner_vps: int = 10, loser_vps: int = 6) -> np.ndarray:
    vps = np.zeros(len(PLAYER_NAMES), dtype=np.int16)
    if winner is not None:
        for name in ("RED", "BLUE"):
            vps[_seat(name)] = winner_vps if name == winner else loser_vps
    return vps


def _make_game_rows(
    *,
    game_seed: int,
    num_decisions: int,
    winner: str | None,
    terminated: bool,
    forced_flags: list[bool] | None = None,
    used_full_flags: list[bool] | None = None,
    legal_width: int = 4,
    target_policy_override: np.ndarray | None = None,
    winner_vps: int = 10,
    loser_vps: int = 6,
    afterstate_override: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    forced_roll_at: list[int] | None = None,
) -> list[dict]:
    forced_flags = forced_flags or [False] * num_decisions
    used_full_flags = used_full_flags or [True] * num_decisions
    forced_roll_at = forced_roll_at or []
    vps = _vps(winner, winner_vps=winner_vps, loser_vps=loser_vps) if terminated else np.zeros(
        len(PLAYER_NAMES), dtype=np.int16
    )
    rows = []
    for i in range(num_decisions):
        legal_action_ids = np.full(legal_width, -1, dtype=np.int16)
        n_legal = 1 if forced_flags[i] else min(3, legal_width)
        if i in forced_roll_at:
            legal_action_ids[0] = ROLL_ACTION_ID
        else:
            legal_action_ids[:n_legal] = np.arange(n_legal, dtype=np.int16)

        target_policy = np.zeros(legal_width, dtype=np.float16)
        if target_policy_override is not None:
            target_policy[:n_legal] = target_policy_override[:n_legal]
        else:
            target_policy[:n_legal] = 1.0 / n_legal

        afterstate_target = np.full(legal_width, np.nan, dtype=np.float32)
        afterstate_target_mask = np.zeros(legal_width, dtype=bool)
        if afterstate_override is not None and i in afterstate_override:
            values, mask = afterstate_override[i]
            afterstate_target[: len(values)] = values
            afterstate_target_mask[: len(mask)] = mask
        elif i in forced_roll_at:
            afterstate_target[0] = 0.1
            afterstate_target_mask[0] = True

        rows.append(
            {
                "game_seed": np.int64(game_seed),
                "player": "RED" if i % 2 == 0 else "BLUE",
                "phase": "PLAY_TURN" if i not in forced_roll_at else "PLAY_TURN",
                "winner": winner or "",
                "terminated": bool(terminated),
                "truncated": not bool(terminated),
                "final_actual_vps": vps,
                "has_final_actual_vps": bool(terminated),
                "legal_action_ids": legal_action_ids,
                "target_policy": target_policy,
                "is_forced": bool(forced_flags[i]),
                "used_full_search": bool(used_full_flags[i]),
                "afterstate_target": afterstate_target,
                "afterstate_target_mask": afterstate_target_mask,
                # Matches the real driver's formula: forced rows always get 0
                # (no search signal to imitate); value_weight is always 1.
                "policy_weight_multiplier": np.float32(
                    0.0 if forced_flags[i] else (1.0 if used_full_flags[i] else 0.0)
                ),
                "value_weight_multiplier": np.float32(1.0),
            }
        )
    return rows


def _rows_list_to_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in rows[0]:
        out[key] = np.stack([row[key] for row in rows], axis=0) if isinstance(
            rows[0][key], np.ndarray
        ) else np.array([row[key] for row in rows])
    return out


def _write_shard(path: Path, rows: list[dict]) -> None:
    arrays = _rows_list_to_arrays(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


@pytest.fixture()
def good_dataset(tmp_path: Path) -> Path:
    shards_dir = tmp_path / "shards"
    rows = []
    # Game 1: RED wins cleanly, no forced decisions.
    rows += _make_game_rows(game_seed=1, num_decisions=6, winner="RED", terminated=True)
    # Game 2: BLUE wins, includes a forced ROLL decision with a valid afterstate target.
    rows += _make_game_rows(
        game_seed=2,
        num_decisions=6,
        winner="BLUE",
        terminated=True,
        forced_flags=[False, False, True, False, False, False],
        forced_roll_at=[2],
    )
    # Game 3: truncated, no winner.
    rows += _make_game_rows(game_seed=3, num_decisions=4, winner=None, terminated=False)
    _write_shard(shards_dir / "gumbel_self_play_shard_00000.npz", rows)
    manifest = {"shards": [str(shards_dir / "gumbel_self_play_shard_00000.npz")]}
    (shards_dir / "manifest.json").write_text(json.dumps(manifest))
    return shards_dir


def test_run_audit_passes_on_good_dataset(good_dataset: Path):
    report = run_audit(
        good_dataset, vps_to_win=10, p_full=0.25, colors=("RED", "BLUE"), reference_phase_counts=None
    )
    assert report["games_total"] == 3
    label_check = next(c for c in report["checks"] if c["check"] == "label_perspective_correctness")
    assert label_check["pass"], label_check["failures"]
    policy_check = next(c for c in report["checks"] if c["check"] == "target_policy_sums_to_one")
    assert policy_check["pass"], policy_check["example_bad_rows"]
    afterstate_check = next(
        c for c in report["checks"] if c["check"] == "afterstate_targets_on_forced_roll"
    )
    assert afterstate_check["pass"], afterstate_check["failures"]
    assert afterstate_check["forced_roll_rows_found"] == 1
    kl_check = next(c for c in report["checks"] if c["check"] == "kl_improved_policy_vs_prior")
    assert kl_check["skipped"] is True


def test_label_perspective_catches_winner_under_threshold(tmp_path: Path):
    rows = _make_game_rows(
        game_seed=1, num_decisions=3, winner="RED", terminated=True, winner_vps=8, loser_vps=6
    )
    result = check_label_perspective(_rows_list_to_arrays(rows), vps_to_win=10)
    assert not result["pass"]
    assert result["num_failures"] >= 1


def test_label_perspective_catches_double_qualifier(tmp_path: Path):
    rows = _make_game_rows(
        game_seed=1, num_decisions=3, winner="RED", terminated=True, winner_vps=10, loser_vps=11
    )
    result = check_label_perspective(_rows_list_to_arrays(rows), vps_to_win=10)
    assert not result["pass"]


def test_label_perspective_catches_truncated_with_winner(tmp_path: Path):
    rows = _make_game_rows(game_seed=1, num_decisions=3, winner=None, terminated=False)
    rows[0]["winner"] = "RED"  # inject a violation: truncated game claiming a winner
    result = check_label_perspective(_rows_list_to_arrays(rows), vps_to_win=10)
    assert not result["pass"]


def test_target_policy_sum_catches_bad_normalization():
    rows = _make_game_rows(
        game_seed=1,
        num_decisions=1,
        winner="RED",
        terminated=True,
        target_policy_override=np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float16),
    )
    result = check_target_policy_sums_to_one(_rows_list_to_arrays(rows))
    assert not result["pass"]
    assert result["num_failures"] == 1


def test_target_policy_sum_accepts_normalized_policy():
    rows = _make_game_rows(game_seed=1, num_decisions=5, winner="RED", terminated=True)
    result = check_target_policy_sums_to_one(_rows_list_to_arrays(rows))
    assert result["pass"]


def test_mix_check_flags_forced_fraction_out_of_range():
    # All rows forced -> fraction 1.0, well outside the expected band.
    rows = _make_game_rows(
        game_seed=1, num_decisions=4, winner="RED", terminated=True, forced_flags=[True] * 4
    )
    result = check_mix(_rows_list_to_arrays(rows), p_full=0.25, reference_phase_counts=None)
    assert not result["is_forced_in_range"]
    assert not result["pass"]


def test_mix_check_accepts_higher_forced_fraction_post_f1_calibration():
    """2026-07-04 calibration: post-F1 fix + max_decisions=600, shorter/more
    decisive games legitimately push the forced-ROLL fraction up to ~64% in
    early pilot data (vs the original ~45-50% expectation) -- the default
    range was raised to 0.40-0.70 to match, since the real correctness
    guarantee lives in `check_weight_multipliers`, not this coarse band.
    """
    forced_flags = ([True] * 64) + ([False] * 36)  # 64% forced, matches observed pilot sample
    rows = _make_game_rows(
        game_seed=1,
        num_decisions=100,
        winner="RED",
        terminated=True,
        forced_flags=forced_flags,
        used_full_flags=[False if f else True for f in forced_flags],
    )
    result = check_mix(_rows_list_to_arrays(rows), p_full=1.0, reference_phase_counts=None)
    assert result["is_forced_fraction"] == pytest.approx(0.64)
    assert result["is_forced_in_range"]


def test_mix_check_still_rejects_range_narrower_than_default_when_requested():
    rows = _make_game_rows(
        game_seed=1, num_decisions=4, winner="RED", terminated=True, forced_flags=[True, True, True, False]
    )  # 75% forced
    result = check_mix(
        _rows_list_to_arrays(rows),
        p_full=0.25,
        reference_phase_counts=None,
        is_forced_expected_range=(0.40, 0.70),
    )
    assert not result["is_forced_in_range"]


def test_mix_check_flags_full_search_fraction_mismatch():
    rows = _make_game_rows(
        game_seed=1,
        num_decisions=10,
        winner="RED",
        terminated=True,
        forced_flags=[False] * 10,
        used_full_flags=[True] * 10,  # 100% full search among non-forced, vs configured p_full=0.25
    )
    result = check_mix(_rows_list_to_arrays(rows), p_full=0.25, reference_phase_counts=None)
    assert not result["full_search_matches_p_full"]


def test_afterstate_check_catches_missing_target_on_forced_roll():
    rows = _make_game_rows(
        game_seed=1,
        num_decisions=1,
        winner="RED",
        terminated=True,
        forced_flags=[True],
        forced_roll_at=[0],
        afterstate_override={0: (np.array([np.nan]), np.array([False]))},  # no target present
    )
    result = check_afterstate_targets_on_forced_roll(_rows_list_to_arrays(rows), colors=("RED", "BLUE"))
    assert not result["pass"]
    assert result["forced_roll_rows_found"] == 1


def test_afterstate_check_catches_out_of_range_value():
    rows = _make_game_rows(
        game_seed=1,
        num_decisions=1,
        winner="RED",
        terminated=True,
        forced_flags=[True],
        forced_roll_at=[0],
        afterstate_override={0: (np.array([1.5]), np.array([True]))},  # out of [-1, 1]
    )
    result = check_afterstate_targets_on_forced_roll(_rows_list_to_arrays(rows), colors=("RED", "BLUE"))
    assert not result["pass"]


def test_afterstate_check_accepts_valid_forced_roll_target():
    rows = _make_game_rows(
        game_seed=1,
        num_decisions=1,
        winner="RED",
        terminated=True,
        forced_flags=[True],
        forced_roll_at=[0],
    )
    result = check_afterstate_targets_on_forced_roll(_rows_list_to_arrays(rows), colors=("RED", "BLUE"))
    assert result["pass"]
    assert result["forced_roll_rows_found"] == 1


def test_load_rows_pads_ragged_shards_to_common_width(tmp_path: Path):
    shards_dir = tmp_path / "shards"
    narrow = _make_game_rows(game_seed=1, num_decisions=2, winner="RED", terminated=True, legal_width=3)
    wide = _make_game_rows(game_seed=2, num_decisions=2, winner="BLUE", terminated=True, legal_width=5)
    _write_shard(shards_dir / "shard_a.npz", narrow)
    _write_shard(shards_dir / "shard_b.npz", wide)
    merged = load_rows([shards_dir / "shard_a.npz", shards_dir / "shard_b.npz"])
    assert merged["legal_action_ids"].shape[1] == 5
    assert merged["game_seed"].shape[0] == 4


def test_load_rows_pads_legal_action_context_and_entity_ragged_columns(tmp_path: Path):
    """Regression: real shards (unlike the synthetic fixtures above) also carry
    `legal_action_context` (2D-per-row, i.e. 3D stacked) and the ENTITY_KEYS
    ragged columns (`legal_action_tokens`, `legal_action_target_ids`,
    `legal_action_mask`) -- none of which were in `load_rows`'s width-padding
    set. A real 20-game raw-selfplay smoke run (2026-07-04, task #65) crashed
    here with a `np.concatenate` dimension-mismatch (54 vs 28) the instant two
    shards had genuinely different legal-action widths (e.g. a wide 54-action
    placement decision vs a normal ~10-30-action turn) -- narrow, single-width
    unit fixtures never exercised this path."""
    shards_dir = tmp_path / "shards"
    narrow = _make_game_rows(game_seed=1, num_decisions=2, winner="RED", terminated=True, legal_width=3)
    wide = _make_game_rows(game_seed=2, num_decisions=2, winner="BLUE", terminated=True, legal_width=5)
    feature_size = 7
    for rows, legal_width in ((narrow, 3), (wide, 5)):
        for row in rows:
            row["legal_action_context"] = np.zeros((legal_width, feature_size), dtype=np.float16)
            row["legal_action_tokens"] = np.zeros((legal_width, feature_size), dtype=np.float16)
            row["legal_action_target_ids"] = np.full((legal_width, 4), -1, dtype=np.int16)
            row["legal_action_mask"] = np.ones((legal_width,), dtype=bool)
    _write_shard(shards_dir / "shard_a.npz", narrow)
    _write_shard(shards_dir / "shard_b.npz", wide)

    merged = load_rows([shards_dir / "shard_a.npz", shards_dir / "shard_b.npz"])

    assert merged["legal_action_context"].shape == (4, 5, feature_size)
    assert merged["legal_action_tokens"].shape == (4, 5, feature_size)
    assert merged["legal_action_target_ids"].shape == (4, 5, 4)
    assert merged["legal_action_mask"].shape == (4, 5)
    # Padded slots for the narrow (legal_width=3) rows must not silently claim
    # a real legal action (mask False, target ids -1 -- the same sentinel
    # `legal_action_ids` already uses).
    assert not merged["legal_action_mask"][0, 3:].any()
    assert (merged["legal_action_target_ids"][0, 3:] == -1).all()


# --------------------------------------------------------------------------
# KL(improved_policy || prior) -- forward-compat tests for the pending
# 'prior_policy' schema addition (see the 2026-07-03 schema-gap finding).
# --------------------------------------------------------------------------


def _add_prior_policy(rows: dict[str, np.ndarray], prior: np.ndarray) -> dict[str, np.ndarray]:
    rows = dict(rows)
    rows["prior_policy"] = prior.astype(np.float16)
    return rows


def test_kl_check_skipped_when_prior_policy_absent():
    rows = _make_game_rows(game_seed=1, num_decisions=3, winner="RED", terminated=True)
    result = check_kl_improved_vs_prior(_rows_list_to_arrays(rows))
    assert result["skipped"] is True
    assert result["pass"] is None


def test_kl_check_reports_near_zero_when_prior_equals_target():
    rows = _make_game_rows(game_seed=1, num_decisions=4, winner="RED", terminated=True, legal_width=4)
    arrays = _rows_list_to_arrays(rows)
    # Search added nothing: improved_policy == prior for every row.
    arrays = _add_prior_policy(arrays, arrays["target_policy"].copy())
    result = check_kl_improved_vs_prior(arrays)
    assert result["skipped"] is False
    assert result["kl_distribution"]["mean"] == pytest.approx(0.0, abs=1e-6)
    assert result["kl_distribution"]["near_zero_fraction"] == pytest.approx(1.0)


def test_kl_check_reports_nonzero_when_prior_differs_from_target():
    rows = _make_game_rows(
        game_seed=1,
        num_decisions=1,
        winner="RED",
        terminated=True,
        legal_width=4,
        target_policy_override=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float16),
    )
    arrays = _rows_list_to_arrays(rows)
    # Search moved all mass onto one action the prior considered unlikely.
    prior = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float16)
    arrays = _add_prior_policy(arrays, prior)
    result = check_kl_improved_vs_prior(arrays)
    assert result["skipped"] is False
    assert result["kl_distribution"]["mean"] > 1.0
    assert result["kl_distribution"]["near_zero_fraction"] == pytest.approx(0.0)


def test_kl_check_separates_forced_rows_which_are_trivially_zero():
    """Forced (single-legal-action) rows have exactly one choice, so
    KL(improved||prior) is trivially 0 there regardless of whether the search
    is adding real information -- the 'non_forced' breakdown must isolate the
    actually meaningful signal instead of being diluted by them.
    """
    forced_rows = _make_game_rows(
        game_seed=1, num_decisions=5, winner="RED", terminated=True, forced_flags=[True] * 5
    )
    non_forced_rows = _make_game_rows(
        game_seed=2,
        num_decisions=1,
        winner="BLUE",
        terminated=True,
        legal_width=4,
        forced_flags=[False],
        target_policy_override=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float16),
    )
    all_rows = forced_rows + non_forced_rows
    arrays = _rows_list_to_arrays(all_rows)
    # A properly-normalized prior restricted to LEGAL actions only has no
    # choice but P=1 on the single legal slot when there's exactly one legal
    # action -- that's what makes forced-row KL trivially 0, not an arbitrary
    # prior. The non-forced row's prior is deliberately different from its
    # target to exercise the real (nonzero) signal.
    prior = np.zeros((len(all_rows), 4), dtype=np.float16)
    prior[:5, 0] = 1.0  # 5 forced rows, each with exactly one legal action
    prior[5] = [0.25, 0.25, 0.25, 0.25]  # the one non-forced row
    arrays = _add_prior_policy(arrays, prior)
    result = check_kl_improved_vs_prior(arrays)
    assert result["skipped"] is False
    assert result["kl_distribution_forced"]["near_zero_fraction"] == pytest.approx(1.0)
    assert result["kl_distribution_forced"]["mean"] == pytest.approx(0.0, abs=1e-6)
    assert result["kl_distribution_non_forced"]["rows"] == 1
    assert result["kl_distribution_non_forced"]["mean"] > 1.0


def test_load_rows_pads_prior_policy_with_safe_fill_not_negative_one(tmp_path: Path):
    shards_dir = tmp_path / "shards"
    narrow_rows = _make_game_rows(game_seed=1, num_decisions=1, winner="RED", terminated=True, legal_width=2)
    wide_rows = _make_game_rows(game_seed=2, num_decisions=1, winner="BLUE", terminated=True, legal_width=4)
    narrow = _add_prior_policy(_rows_list_to_arrays(narrow_rows), np.array([[0.5, 0.5]]))
    wide = _add_prior_policy(_rows_list_to_arrays(wide_rows), np.array([[0.25, 0.25, 0.25, 0.25]]))

    def _write(path: Path, arrays: dict[str, np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)

    _write(shards_dir / "shard_a.npz", narrow)
    _write(shards_dir / "shard_b.npz", wide)
    merged = load_rows([shards_dir / "shard_a.npz", shards_dir / "shard_b.npz"])
    assert merged["prior_policy"].shape[1] == 4
    # The padded tail of the narrow-shard row must not be -1 (a probability-
    # breaking fill inherited from legal_action_ids' own convention).
    padded_tail = merged["prior_policy"][0, 2:]
    assert not np.any(padded_tail == -1)


# --------------------------------------------------------------------------
# Truncation rate, weight multipliers, prior-policy non-degeneracy
# --------------------------------------------------------------------------


def _rows_for_games(num_games: int, *, num_truncated: int, decisions_per_game: int = 2) -> list[dict]:
    rows: list[dict] = []
    for seed in range(num_games):
        truncated = seed < num_truncated
        rows += _make_game_rows(
            game_seed=seed,
            num_decisions=decisions_per_game,
            winner=None if truncated else "RED",
            terminated=not truncated,
        )
    return rows


def test_truncation_rate_passes_below_threshold():
    rows = _rows_for_games(10, num_truncated=3)  # 30% < 40%
    result = check_truncation_rate(_rows_list_to_arrays(rows), threshold=0.40)
    assert result["pass"]
    assert result["truncated_fraction"] == pytest.approx(0.3)


def test_truncation_rate_fails_above_threshold():
    rows = _rows_for_games(10, num_truncated=5)  # 50% > 40%
    result = check_truncation_rate(_rows_list_to_arrays(rows), threshold=0.40)
    assert not result["pass"]
    assert result["truncated_fraction"] == pytest.approx(0.5)


def test_game_seed_range_requires_exact_half_open_allocation():
    rows = _make_game_rows(game_seed=100, num_decisions=2, winner="RED", terminated=True)
    rows += _make_game_rows(game_seed=101, num_decisions=2, winner="BLUE", terminated=True)
    arrays = _rows_list_to_arrays(rows)

    assert check_game_seed_range(arrays, expected_range=(100, 102))["pass"]

    missing = check_game_seed_range(arrays, expected_range=(100, 103))
    assert not missing["pass"]
    assert missing["missing_examples"] == [102]

    unexpected = check_game_seed_range(arrays, expected_range=(101, 102))
    assert not unexpected["pass"]
    assert unexpected["unexpected_examples"] == [100]


def test_weight_multipliers_accepts_correct_defaults():
    rows = _make_game_rows(
        game_seed=1,
        num_decisions=4,
        winner="RED",
        terminated=True,
        forced_flags=[True, False, False, True],
        used_full_flags=[True, True, False, True],
    )
    result = check_weight_multipliers(_rows_list_to_arrays(rows))
    assert result["pass"]
    assert result["forced_rows_checked"] == 2


def test_weight_multipliers_catches_nonzero_policy_weight_on_forced_row():
    rows = _make_game_rows(game_seed=1, num_decisions=2, winner="RED", terminated=True, forced_flags=[True, False])
    arrays = _rows_list_to_arrays(rows)
    arrays["policy_weight_multiplier"][0] = 1.0  # violation: forced row must be 0
    result = check_weight_multipliers(arrays)
    assert not result["pass"]
    assert result["num_failures"] == 1


def test_weight_multipliers_catches_value_weight_not_one():
    rows = _make_game_rows(game_seed=1, num_decisions=2, winner="RED", terminated=True)
    arrays = _rows_list_to_arrays(rows)
    arrays["value_weight_multiplier"][1] = 0.0  # violation
    result = check_weight_multipliers(arrays)
    assert not result["pass"]
    assert result["num_failures"] == 1


def test_weight_multipliers_skips_when_columns_absent():
    rows = _make_game_rows(game_seed=1, num_decisions=2, winner="RED", terminated=True)
    arrays = _rows_list_to_arrays(rows)
    del arrays["policy_weight_multiplier"]
    del arrays["value_weight_multiplier"]
    result = check_weight_multipliers(arrays)
    assert result["skipped"] is True
    assert result["pass"] is None


def test_prior_policy_nondegenerate_skipped_when_absent():
    rows = _make_game_rows(game_seed=1, num_decisions=3, winner="RED", terminated=True)
    result = check_prior_policy_nondegenerate(_rows_list_to_arrays(rows))
    assert result["skipped"] is True
    assert result["pass"] is None


def test_prior_policy_nondegenerate_passes_on_spread_out_prior():
    rows = _make_game_rows(
        game_seed=1, num_decisions=3, winner="RED", terminated=True, legal_width=4, forced_flags=[False] * 3
    )
    arrays = _rows_list_to_arrays(rows)
    arrays = _add_prior_policy(arrays, np.full((3, 4), 0.25))  # uniform: max entropy
    result = check_prior_policy_nondegenerate(arrays)
    assert result["pass"]
    assert result["near_one_hot_fraction"] == pytest.approx(0.0)


def test_prior_policy_nondegenerate_fails_when_mostly_one_hot():
    rows = _make_game_rows(
        game_seed=1, num_decisions=4, winner="RED", terminated=True, legal_width=4, forced_flags=[False] * 4
    )
    arrays = _rows_list_to_arrays(rows)
    one_hot_prior = np.zeros((4, 4))
    one_hot_prior[:, 0] = 1.0  # every non-forced row collapses onto a single action
    arrays = _add_prior_policy(arrays, one_hot_prior)
    result = check_prior_policy_nondegenerate(arrays)
    assert not result["pass"]
    assert result["near_one_hot_fraction"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# config_provenance -- reading manifest.json / worker manifests directly
# (post commit 982d344's selfplay_config/search_config/cli_args fields)
# --------------------------------------------------------------------------


def _write_manifest_tree(
    shards_dir: Path,
    *,
    num_workers: int,
    c_scale_per_worker: list[float] | None = None,
    include_cli_args: bool = True,
    legacy_no_config: bool = False,
) -> None:
    worker_summaries = []
    for i in range(num_workers):
        worker_dir = shards_dir / f"worker_{i:03d}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        worker_manifest_path = worker_dir / "manifest.json"
        if legacy_no_config:
            worker_manifest = {"out_dir": str(worker_dir)}
        else:
            c_scale = c_scale_per_worker[i] if c_scale_per_worker else 0.1
            worker_manifest = {
                "out_dir": str(worker_dir),
                "search_config": {"c_scale": c_scale, "n_full": 64, "n_fast": 16},
                "selfplay_config": {"max_decisions": 600},
            }
        worker_manifest_path.write_text(json.dumps(worker_manifest))
        worker_summaries.append(str(worker_manifest_path))

    top_manifest: dict = {"worker_summaries": worker_summaries, "shards": []}
    if include_cli_args and not legacy_no_config:
        top_manifest["cli_args"] = {"c_scale": c_scale_per_worker[0] if c_scale_per_worker else 0.1}
    (shards_dir / "manifest.json").write_text(json.dumps(top_manifest))


def test_config_provenance_skipped_on_legacy_manifest(tmp_path: Path):
    shards_dir = tmp_path / "shards"
    _write_manifest_tree(shards_dir, num_workers=2, legacy_no_config=True)
    result = check_config_provenance(shards_dir)
    assert result["skipped"] is True
    assert result["pass"] is None


def test_config_provenance_passes_when_all_workers_match_expected(tmp_path: Path):
    shards_dir = tmp_path / "shards"
    _write_manifest_tree(shards_dir, num_workers=3, c_scale_per_worker=[0.1, 0.1, 0.1])
    result = check_config_provenance(shards_dir, expected={"c_scale": 0.1})
    assert result["pass"]
    assert result["workers_checked"] == 3
    assert result["observed"]["c_scale"] == 0.1


def test_config_provenance_fails_when_value_does_not_match_expected(tmp_path: Path):
    shards_dir = tmp_path / "shards"
    _write_manifest_tree(shards_dir, num_workers=2, c_scale_per_worker=[1.0, 1.0])
    result = check_config_provenance(shards_dir, expected={"c_scale": 0.1})
    assert not result["pass"]
    assert any("c_scale" in failure for failure in result["failures"])


def test_config_provenance_fails_when_workers_disagree(tmp_path: Path):
    shards_dir = tmp_path / "shards"
    _write_manifest_tree(shards_dir, num_workers=2, c_scale_per_worker=[0.1, 1.0])
    result = check_config_provenance(shards_dir, expected={"c_scale": 0.1})
    assert not result["pass"]
    assert any("disagree" in failure for failure in result["failures"])


def test_config_provenance_fails_when_expected_key_is_unrecorded(tmp_path: Path):
    shards_dir = tmp_path / "shards"
    _write_manifest_tree(shards_dir, num_workers=2)

    result = check_config_provenance(
        shards_dir,
        expected={"c_scale": 0.1, "public_observation": True},
    )

    assert not result["pass"]
    assert "public_observation" not in result["observed"]
    assert any(
        "public_observation" in failure and "no manifest recorded" in failure
        for failure in result["failures"]
    )
