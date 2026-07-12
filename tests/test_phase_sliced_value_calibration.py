from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from phase_sliced_value_calibration import (  # type: ignore  # noqa: E402
    ENTITY_KEYS,
    build_calibration_summary,
    build_row_selection_provenance,
    collect_rows,
    derive_validation_game_seeds,
    load_validation_seed_manifest,
    parse_validation_game_seed_ranges,
    _calibration_stats,
    _hlgauss_targets,
    _legal_bucket,
    _slice_by,
    compute_q,
    compute_readout,
    resolve_readout_provenance,
    write_validation_seed_manifest,
)


class _FakeReadoutPolicy:
    def __init__(self, *, trained_categorical: bool) -> None:
        self.model = SimpleNamespace(
            value_categorical_bins=3,
            value_categorical_truncation_class=True,
            value_categorical_head=object(),
        )
        self._checkpoint_missing_state_keys = ()
        self._value_training_provenance_errors = ()
        if trained_categorical:
            self.trained_value_readouts = ("categorical",)
            self.value_training = {
                "schema_version": "value-training-v1",
                "trained_value_readouts": ["categorical"],
                "resolved_categorical_ce_weight": 0.25,
                "categorical_training_weight_sum": 128.0,
                "hlgauss_bins": 3,
                "hlgauss_sigma_ratio": 0.75,
                "optimizer_steps": 10,
                "completed_epochs": 1,
            }
        else:
            # Exactly the config-only upgrade case: a module exists, but no
            # optimizer provenance says its random weights were trained.
            self.trained_value_readouts = ("scalar",)
            self.value_training = None

    def forward_legal_np(self, entity_batch, legal_ids, legal_context):
        del entity_batch, legal_context
        import torch

        rows = len(legal_ids)
        assert rows == 2
        log_three = float(np.log(3.0))
        return {
            "value": torch.tensor([0.9, -0.8]),
            "value_categorical": torch.tensor([0.25, -0.25]),
            # First row is a win and the +1 endpoint has probability 1/2;
            # second is a loss and the -1 endpoint has probability 1/2.
            "value_categorical_logits": torch.tensor(
                [[0.0, 0.0, log_three, 0.0], [log_three, 0.0, 0.0, 0.0]]
            ),
            "value_categorical_truncation_prob": torch.tensor([1 / 6, 1 / 6]),
        }


def _fake_groups() -> list[dict[str, np.ndarray]]:
    group = {key: np.zeros((2, 1), dtype=np.float32) for key in ENTITY_KEYS}
    group.update(
        {
            "legal_action_ids": np.zeros((2, 2), dtype=np.int64),
            "legal_action_context": np.zeros((2, 2, 1), dtype=np.float32),
            "z": np.array([1.0, -1.0], dtype=np.float32),
            "phase_label": np.array(["opening_placement", "play_turn"]),
            "forced": np.array([False, True]),
            "legal_count": np.array([54, 1]),
        }
    )
    return [group]


def _write_calibration_shard(path: Path, game_seeds: list[int]) -> None:
    n = len(game_seeds)
    payload = {key: np.zeros((n, 1), dtype=np.float32) for key in ENTITY_KEYS}
    payload.update(
        {
            "game_seed": np.asarray(game_seeds, dtype=np.int64),
            "terminated": np.ones(n, dtype=bool),
            "truncated": np.zeros(n, dtype=bool),
            "winner": np.asarray(["RED"] * n),
            "player": np.asarray(["RED", "BLUE"] * ((n + 1) // 2))[:n],
            "phase": np.asarray(["PLAY_TURN"] * n),
            "is_forced": np.zeros(n, dtype=bool),
            "legal_action_mask": np.ones((n, 2), dtype=bool),
            "legal_action_ids": np.zeros((n, 2), dtype=np.int64),
            "legal_action_context": np.zeros((n, 2, 1), dtype=np.float32),
        }
    )
    np.savez(path, **payload)


def test_legal_bucket_boundaries():
    assert _legal_bucket(1) == "1"
    assert _legal_bucket(2) == "2-4"
    assert _legal_bucket(4) == "2-4"
    assert _legal_bucket(5) == "5-10"
    assert _legal_bucket(10) == "5-10"
    assert _legal_bucket(11) == "11-20"
    assert _legal_bucket(20) == "11-20"
    assert _legal_bucket(21) == "21-40"
    assert _legal_bucket(40) == "21-40"
    assert _legal_bucket(41) == "41+"
    assert _legal_bucket(54) == "41+"


def test_validation_fraction_derivation_matches_trainer_game_split(
    tmp_path: Path,
) -> None:
    game_seeds = [10] * 3 + [20] * 2 + [30] * 4 + [40] * 3 + [50] * 2
    _write_calibration_shard(tmp_path / "rows.npz", game_seeds)
    selection = derive_validation_game_seeds(
        str(tmp_path), validation_fraction=0.25, validation_seed=17
    )

    from train_bc import split_train_validation_indices

    split = split_train_validation_indices(
        {
            "action_taken": np.zeros(len(game_seeds), dtype=np.int64),
            "game_seed": np.asarray(game_seeds, dtype=np.int64),
        },
        validation_fraction=0.25,
        validation_seed=17,
        validation_max_samples=0,
    )
    expected = np.unique(np.asarray(game_seeds)[split["validation"]])
    assert np.array_equal(selection.game_seeds, expected)
    assert selection.source_row_count == len(game_seeds)
    assert selection.source_game_count == 5


def test_heldout_range_filter_and_manifest_are_explicit(tmp_path: Path) -> None:
    _write_calibration_shard(tmp_path / "rows.npz", [10, 10, 20, 20, 30, 30, 40, 40])
    ranges = parse_validation_game_seed_ranges("20:30")
    groups = collect_rows(str(tmp_path), validation_game_seed_ranges=ranges)
    assert set(np.concatenate([group["game_seed"] for group in groups])) == {20, 30}

    selection = derive_validation_game_seeds(
        str(tmp_path), validation_fraction=0.25, validation_seed=17
    )
    manifest = tmp_path / "validation_seeds.json"
    expected_sha = write_validation_seed_manifest(
        manifest, selection, shard_dir=str(tmp_path)
    )
    loaded, actual_sha = load_validation_seed_manifest(manifest)
    assert np.array_equal(loaded, selection.game_seeds)
    assert actual_sha == expected_sha

    selected_groups = collect_rows(str(tmp_path), validation_game_seeds=loaded)
    provenance = build_row_selection_provenance(
        selected_groups,
        mode="validation_seed_manifest",
        seed_manifest_path=str(manifest),
        seed_manifest_sha256=actual_sha,
        configured_game_seed_count=len(loaded),
    )
    assert provenance["held_out_filter_applied"] is True
    assert provenance["configured_game_seed_count"] == len(loaded)
    assert provenance["observed_game_seed_count"] == len(loaded)
    assert provenance["seed_manifest_sha256"] == actual_sha


def test_loads_trainers_exact_validation_seed_manifest_and_verifies_digest(
    tmp_path: Path,
) -> None:
    seeds = np.asarray([30, 10, 20], dtype=np.int64)
    canonical = np.sort(seeds).astype("<i8", copy=False)
    digest = "sha256:" + hashlib.sha256(canonical.tobytes()).hexdigest()
    manifest = tmp_path / "report.validation_seeds.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "train-validation-game-seeds-v1",
                "validation_game_seed_count": 3,
                "validation_game_seed_set_sha256": digest,
                "game_seeds": seeds.tolist(),
            }
        )
    )

    loaded, file_digest = load_validation_seed_manifest(manifest)

    assert loaded.tolist() == [10, 20, 30]
    assert file_digest == hashlib.sha256(manifest.read_bytes()).hexdigest()

    payload = json.loads(manifest.read_text())
    payload["validation_game_seed_set_sha256"] = "sha256:" + "0" * 64
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_validation_seed_manifest(manifest)


def test_validation_filter_refuses_missing_game_seed(tmp_path: Path) -> None:
    _write_calibration_shard(tmp_path / "rows.npz", [10, 20])
    with np.load(tmp_path / "rows.npz") as original:
        payload = {key: original[key] for key in original.files if key != "game_seed"}
    np.savez(tmp_path / "rows.npz", **payload)
    with pytest.raises(ValueError, match="held-out selection cannot be verified"):
        collect_rows(str(tmp_path), validation_game_seed_ranges=((10, 20),))


def test_calibration_stats_perfectly_calibrated():
    # q equal to z -> corr 1, Brier 0.
    z = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
    q = z.copy()
    stats = _calibration_stats(q, z, min_rows=1)
    assert stats["n"] == 4
    assert stats["corr_q_z"] == pytest.approx(1.0)
    assert stats["brier"] == pytest.approx(0.0)
    assert stats["value_rmse"] == pytest.approx(0.0)
    assert stats["e_q_given_win"] == pytest.approx(1.0)
    assert stats["e_q_given_loss"] == pytest.approx(-1.0)
    assert stats["win_probability_ece"] == pytest.approx(0.0)
    assert sum(row["n"] for row in stats["reliability_bins"]) == 4


def test_value_rmse_measures_residual_std():
    # q constant 0 vs z in {+1,-1} -> residual is +-1 -> RMSE 1.0.
    z = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    q = np.zeros(4, dtype=np.float32)
    assert _calibration_stats(q, z, min_rows=1)["value_rmse"] == pytest.approx(1.0)


def test_calibration_stats_single_class_reports_null_corr():
    # All wins -> corr undefined (guarded to None), but Brier still defined.
    z = np.ones(5, dtype=np.float32)
    q = np.full(5, 0.5, dtype=np.float32)
    stats = _calibration_stats(q, z, min_rows=1)
    assert stats["corr_q_z"] is None
    assert stats["n_loss"] == 0
    # outcome=1, p=(0.5+1)/2=0.75 -> Brier=(0.75-1)^2=0.0625.
    assert stats["brier"] == pytest.approx(0.0625)


def test_calibration_stats_respects_min_rows():
    z = np.array([1.0, -1.0], dtype=np.float32)
    q = np.array([0.9, -0.9], dtype=np.float32)
    assert _calibration_stats(q, z, min_rows=5)["corr_q_z"] is None
    assert _calibration_stats(q, z, min_rows=2)["corr_q_z"] is not None


def test_brier_clips_out_of_range_q():
    # q outside [-1,1] must be clipped so p stays a valid probability.
    z = np.array([1.0, -1.0], dtype=np.float32)
    q = np.array([5.0, -5.0], dtype=np.float32)  # -> p clipped to 1.0 / 0.0
    stats = _calibration_stats(q, z, min_rows=1)
    assert stats["brier"] == pytest.approx(0.0)


def test_slice_by_partitions_rows():
    q = np.array([0.9, 0.8, -0.9, -0.8], dtype=np.float32)
    z = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
    keys = np.array(["opening_placement", "robber", "opening_placement", "robber"])
    sliced = _slice_by(q, z, keys, min_rows=1)
    assert set(sliced.keys()) == {"opening_placement", "robber"}
    assert sliced["opening_placement"]["n"] == 2
    assert sliced["robber"]["n"] == 2


def test_scalar_readout_remains_the_default_without_new_provenance() -> None:
    policy = _FakeReadoutPolicy(trained_categorical=False)
    provenance = resolve_readout_provenance(policy, "scalar")
    assert provenance["requested_readout"] == "scalar"
    assert provenance["model_output_key"] == "value"
    assert provenance["categorical_training_verified"] is False
    assert compute_q(policy, _fake_groups()).tolist() == pytest.approx([0.9, -0.8])


def test_categorical_config_only_upgrade_fails_closed() -> None:
    policy = _FakeReadoutPolicy(trained_categorical=False)
    with pytest.raises(
        ValueError, match="requires a checkpoint with a trained HL-Gauss"
    ):
        compute_readout(policy, _fake_groups(), value_readout="categorical")


def test_calibration_hlgauss_projection_matches_trainer() -> None:
    import torch
    from train_bc import _hl_gauss_value_targets

    targets = torch.tensor([-1.0, -0.35, 0.0, 0.8, 1.0])
    actual = _hlgauss_targets(targets, 9, sigma_ratio=0.75)
    expected = _hl_gauss_value_targets(
        targets,
        9,
        sigma_ratio=0.75,
        add_truncation_class=False,
    )
    assert torch.equal(actual, expected)


def test_trained_categorical_readout_uses_expectation_and_terminal_nll() -> None:
    policy = _FakeReadoutPolicy(trained_categorical=True)
    predictions = compute_readout(policy, _fake_groups(), value_readout="categorical")
    assert predictions.q.tolist() == pytest.approx([0.25, -0.25])
    assert predictions.categorical_hlgauss_ce is not None
    assert np.isfinite(predictions.categorical_hlgauss_ce).all()
    assert predictions.categorical_terminal_nll is not None
    assert predictions.categorical_terminal_nll.tolist() == pytest.approx(
        [np.log(2.0), np.log(2.0)]
    )
    assert predictions.categorical_truncation_probability is not None
    assert predictions.categorical_truncation_probability.tolist() == pytest.approx(
        [1 / 6, 1 / 6]
    )
    assert predictions.provenance["categorical_training_verified"] is True
    assert (
        predictions.provenance["value_training_schema_version"] == "value-training-v1"
    )


def test_categorical_summary_reports_same_slices_plus_proper_score() -> None:
    predictions = compute_readout(
        _FakeReadoutPolicy(trained_categorical=True),
        _fake_groups(),
        value_readout="categorical",
    )
    summary = build_calibration_summary(
        predictions,
        _fake_groups(),
        min_slice_rows=1,
        reliability_bin_count=4,
    )
    assert summary["schema_version"] == "phase-sliced-value-calibration-v2"
    assert summary["value_readout"] == "categorical"
    assert summary["global"]["categorical_hlgauss_ce"] == pytest.approx(
        float(np.mean(predictions.categorical_hlgauss_ce))
    )
    assert summary["global"]["categorical_terminal_nll"] == pytest.approx(np.log(2.0))
    assert summary["global"]["categorical_score_n"] == 2
    assert summary["global"][
        "categorical_truncation_probability_mean"
    ] == pytest.approx(1 / 6)
    assert set(summary["by_phase"]) == {"opening_placement", "play_turn"}
    assert set(summary["by_forced"]) == {"forced", "unforced"}
    assert set(summary["by_legal_count_bucket"]) == {"1", "41+"}
    deployed = summary["deployed_readout_diagnostics"]
    assert deployed["diagnostic_only"] is True
    assert deployed["changes_operator_default"] is False
    assert deployed["categorical_bypasses_scalar_tanh"] is True
    assert deployed["configured_effective_transform"] == "scalar_clip"
    assert summary["by_phase"]["opening_placement"][
        "categorical_terminal_nll"
    ] == pytest.approx(np.log(2.0))


def test_scalar_summary_reports_raw_tanh_and_clip_by_phase() -> None:
    predictions = compute_readout(
        _FakeReadoutPolicy(trained_categorical=False),
        _fake_groups(),
        value_readout="scalar",
    )
    summary = build_calibration_summary(
        predictions,
        _fake_groups(),
        min_slice_rows=1,
        reliability_bin_count=4,
        deployed_value_scale=2.0,
        deployed_value_squash="tanh",
    )
    diagnostic = summary["deployed_readout_diagnostics"]
    assert diagnostic["configured_effective_transform"] == "scalar_tanh"
    assert set(diagnostic["views"]) == {
        "raw_training_readout",
        "scalar_tanh",
        "scalar_clip",
    }
    raw = diagnostic["views"]["raw_training_readout"]
    tanh = diagnostic["views"]["scalar_tanh"]
    clip = diagnostic["views"]["scalar_clip"]
    assert raw["global"]["value_rmse"] == pytest.approx(
        np.sqrt(((0.9 - 1.0) ** 2 + (-0.8 + 1.0) ** 2) / 2)
    )
    expected_tanh = np.tanh(np.array([1.8, -1.6]))
    assert tanh["global"]["value_rmse"] == pytest.approx(
        np.sqrt(np.mean((expected_tanh - np.array([1.0, -1.0])) ** 2))
    )
    assert clip["global"]["value_rmse"] == pytest.approx(0.0)
    assert set(tanh["by_phase"]) == {"opening_placement", "play_turn"}


@pytest.mark.parametrize(
    ("scale", "squash", "message"),
    [(0.0, "tanh", "value_scale"), (1.0, "sigmoid", "value_squash")],
)
def test_deployed_readout_diagnostics_fail_closed(scale, squash, message) -> None:
    predictions = compute_readout(
        _FakeReadoutPolicy(trained_categorical=False),
        _fake_groups(),
        value_readout="scalar",
    )
    with pytest.raises(ValueError, match=message):
        build_calibration_summary(
            predictions,
            _fake_groups(),
            min_slice_rows=1,
            reliability_bin_count=4,
            deployed_value_scale=scale,
            deployed_value_squash=squash,
        )
def test_categorical_nll_partitioning_tracks_each_slice() -> None:
    q = np.array([0.5, -0.5], dtype=np.float32)
    z = np.array([1.0, -1.0], dtype=np.float32)
    nll = np.array([0.1, 1.5], dtype=np.float32)
    keys = np.array(["opening", "play"])
    sliced = _slice_by(
        q,
        z,
        keys,
        min_rows=1,
        categorical_hlgauss_ce=nll,
        categorical_terminal_nll=nll + 0.25,
        reliability_bin_count=2,
    )
    assert sliced["opening"]["categorical_hlgauss_ce"] == pytest.approx(0.1)
    assert sliced["play"]["categorical_hlgauss_ce"] == pytest.approx(1.5)
    assert sliced["opening"]["categorical_terminal_nll"] == pytest.approx(0.35)
    assert sliced["play"]["categorical_terminal_nll"] == pytest.approx(1.75)
