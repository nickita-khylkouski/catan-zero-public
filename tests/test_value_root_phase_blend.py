from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.train_bc import (
    _audit_value_root_blend_corpus,
    _resolve_value_root_blend_regime,
    _value_training_metadata,
    _value_root_blend_mask,
)


def _args(*, lam=0.5, phases="DISCARD,MOVE_ROBBER,PLAY_TURN", compat=False):
    return SimpleNamespace(
        value_target_lambda=lam,
        value_root_blend_phases=phases,
        value_root_blend_global_compat=compat,
    )


def _data() -> dict[str, np.ndarray]:
    return {
        "action_taken": np.arange(6, dtype=np.int16),
        "phase": np.asarray(
            [
                "BUILD_INITIAL_SETTLEMENT",
                "BUILD_INITIAL_ROAD",
                "DISCARD",
                "MOVE_ROBBER",
                "PLAY_TURN",
                "PLAY_TURN",
            ]
        ),
        "root_value": np.asarray([0.9, -0.8, 0.25, -0.5, 0.4, 0.1], dtype=np.float32),
        "root_value_mask": np.asarray([True, True, True, True, True, False]),
        "winner": np.asarray(["RED", "BLUE", "RED", "BLUE", "RED", "RED"]),
        "player": np.asarray(["RED", "RED", "RED", "RED", "RED", "RED"]),
        "truncated": np.asarray([False, False, False, False, False, False]),
        "target_information_regime": np.asarray(["public_conservation_pimc_v1"] * 6),
    }


def test_mature_phase_mask_excludes_opening_and_invalid_roots() -> None:
    data = _data()
    batch = np.arange(6, dtype=np.int64)
    mask = _value_root_blend_mask(
        data,
        batch,
        torch.device("cpu"),
        torch.as_tensor(data["root_value_mask"]),
        torch.ones(6, dtype=torch.bool),
        torch.zeros(6, dtype=torch.bool),
        phases=("DISCARD", "MOVE_ROBBER", "PLAY_TURN"),
    )

    assert mask.tolist() == [False, False, True, True, True, False]


def test_mature_phase_audit_records_realized_operator_and_public_provenance() -> None:
    report = _audit_value_root_blend_corpus(
        _data(),
        np.asarray([1, 1, 2, 3, 4, 5], dtype=np.float32),
        regime=_resolve_value_root_blend_regime(_args()),
    )

    assert report["mode"] == "phase_gated"
    assert report["eligible_rows"] == 3
    assert report["blended_rows"] == 3
    assert report["blended_weighted_mass"] == pytest.approx(9.0)
    assert report["per_phase"]["DISCARD"]["eligible_rows"] == 1
    assert report["per_phase"]["MOVE_ROBBER"]["eligible_rows"] == 1
    assert report["per_phase"]["PLAY_TURN"]["eligible_rows"] == 1
    assert report["mean_abs_root_minus_z"] is not None
    assert report["target_information_regime_counts"] == {
        "public_conservation_pimc_v1": 6
    }


def test_requested_blend_with_zero_realized_rows_fails_closed() -> None:
    data = _data()
    data["root_value_mask"][:] = False
    with pytest.raises(SystemExit, match="zero eligible rows"):
        _audit_value_root_blend_corpus(
            data,
            np.ones(6, dtype=np.float32),
            regime=_resolve_value_root_blend_regime(_args()),
        )


def test_requested_blend_with_zero_training_mass_fails_closed() -> None:
    with pytest.raises(SystemExit, match="zero eligible rows or weighted mass"):
        _audit_value_root_blend_corpus(
            _data(),
            np.zeros(6, dtype=np.float32),
            regime=_resolve_value_root_blend_regime(_args()),
        )


@pytest.mark.parametrize("root", [np.nan, np.inf, 1.001, -1.001])
def test_masked_invalid_root_values_fail_closed(root: float) -> None:
    data = _data()
    data["root_value"][2] = root
    with pytest.raises(SystemExit, match="non-finite or out-of-range"):
        _audit_value_root_blend_corpus(
            data,
            np.ones(6, dtype=np.float32),
            regime=_resolve_value_root_blend_regime(_args()),
        )


def test_nonunit_lambda_requires_explicit_scope_and_unknown_phase_is_rejected() -> None:
    with pytest.raises(SystemExit, match="explicit target-information scope"):
        _resolve_value_root_blend_regime(_args(phases=""))
    with pytest.raises(SystemExit, match="unknown --value-root-blend-phases"):
        _resolve_value_root_blend_regime(_args(phases="MAGIC"))


def test_global_behavior_exists_only_as_explicit_compatibility() -> None:
    regime = _resolve_value_root_blend_regime(_args(phases="", compat=True))
    assert regime["mode"] == "global_compat"
    report = _audit_value_root_blend_corpus(
        _data(), np.ones(6, dtype=np.float32), regime=regime
    )
    assert report["eligible_rows"] == 5


def test_lambda_one_remains_disabled_exact_noop_without_root_column() -> None:
    regime = _resolve_value_root_blend_regime(_args(lam=1.0, phases=""))
    assert regime["mode"] == "disabled"
    report = _audit_value_root_blend_corpus(
        {"action_taken": np.arange(3)}, np.ones(3), regime=regime
    )
    assert report["blended_rows"] == 0


def test_checkpoint_value_metadata_carries_realized_blend_audit() -> None:
    args = _args()
    args.value_head_type = "mse"
    args.hlgauss_scalar_aux_loss_weight = 0.0
    args.value_hlgauss_sigma_ratio = 0.75
    args.truncated_vp_margin_value_weight = 0.0
    args.value_root_blend_audit = {"eligible_rows": 3, "blended_weighted_mass": 9.0}
    metadata = _value_training_metadata(
        args,
        scalar_weight=0.25,
        categorical_weight=0.0,
        categorical_bins=0,
        optimizer_steps=1,
        completed_epochs=1,
        scalar_training_weight_sum=9.0,
        categorical_training_weight_sum=0.0,
    )

    assert metadata["value_root_blend_regime"]["mode"] == "phase_gated"
    assert metadata["value_root_blend_audit"]["eligible_rows"] == 3
