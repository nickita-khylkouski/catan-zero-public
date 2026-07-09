"""Task #79: the H2H runner's output summary must record every knob that
changes the input distribution or search semantics (public_observation,
belief_chance_spectra, n_full_wide, raw_policy_above_width,
symmetry_averaged_eval), not just the search-budget knobs. Without this,
a gate verdict can't be self-certified after the fact -- exactly the gap
that made the h2h_v3conf regime question (was --public-observation used?)
unanswerable from the output JSON alone, task #78's investigation."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from gumbel_search_vs_raw_h2h import _build_summary  # type: ignore  # noqa: E402


def _fake_args(**overrides):
    base = dict(
        checkpoint="ckpt.pt",
        n_full=64,
        lazy_interior_chance=False,
        value_squash="tanh",
        c_scale=0.1,
        c_visit=50.0,
        max_root_candidates=16,
        max_root_candidates_wide=54,
        correct_rust_chance_spectra=True,
        public_observation=True,
        belief_chance_spectra=False,
        n_full_wide=None,
        raw_policy_above_width=None,
        symmetry_averaged_eval=False,
        elo0=0.0,
        elo1=30.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _game(pair_id: int, search_won: bool, terminated=True, truncated=False, game_seed=None):
    return {
        "pair_id": pair_id,
        "search_won": search_won,
        "terminated": terminated,
        "truncated": truncated,
        "game_seed": game_seed if game_seed is not None else pair_id,
    }


def _build(args, games):
    outcomes = [g["search_won"] for g in games if g["search_won"] is not None]
    truncated_count = sum(1 for g in games if g["truncated"])
    return _build_summary(
        args,
        all_games=games,
        outcomes=outcomes,
        truncated_count=truncated_count,
        pairs=list(range(len({g["pair_id"] for g in games}))),
        elapsed=1.23,
        workers=4,
        threads_per_worker=2,
        errors=[],
    )


class TestSummaryRecordsProvenance:
    def test_public_observation_recorded_true(self):
        summary = _build(_fake_args(public_observation=True), [_game(0, True), _game(0, True)])
        assert summary["public_observation"] is True

    def test_public_observation_recorded_false(self):
        summary = _build(_fake_args(public_observation=False), [_game(0, True), _game(0, True)])
        assert summary["public_observation"] is False

    def test_belief_chance_spectra_recorded(self):
        summary = _build(_fake_args(belief_chance_spectra=True), [_game(0, True), _game(0, True)])
        assert summary["belief_chance_spectra"] is True

    def test_n_full_wide_recorded_when_set(self):
        summary = _build(_fake_args(n_full_wide=512), [_game(0, True), _game(0, True)])
        assert summary["n_full_wide"] == 512

    def test_n_full_wide_recorded_as_none_when_unset(self):
        summary = _build(_fake_args(n_full_wide=None), [_game(0, True), _game(0, True)])
        assert summary["n_full_wide"] is None

    def test_raw_policy_above_width_recorded(self):
        summary = _build(_fake_args(raw_policy_above_width=40), [_game(0, True), _game(0, True)])
        assert summary["raw_policy_above_width"] == 40

    def test_symmetry_averaged_eval_recorded(self):
        summary = _build(_fake_args(symmetry_averaged_eval=True), [_game(0, True), _game(0, True)])
        assert summary["symmetry_averaged_eval"] is True

    def test_existing_fields_unaffected(self):
        summary = _build(_fake_args(), [_game(0, True), _game(0, True)])
        assert summary["checkpoint"] == "ckpt.pt"
        assert summary["n_full"] == 64
        assert summary["c_scale"] == 0.1
