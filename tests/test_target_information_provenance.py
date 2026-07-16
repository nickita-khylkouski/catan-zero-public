from __future__ import annotations

import numpy as np
import pytest

from tools.train_bc import (
    TARGET_INFORMATION_REGIME_PUBLIC,
    TARGET_INFORMATION_REGIME_PUBLIC_COHERENT,
    _validate_target_information_admission,
)


def _data(
    regimes: list[str], *, root_value: bool = False, prior_policy: bool = False
) -> dict:
    n = len(regimes)
    data = {
        "action_taken": np.zeros(n, dtype=np.int16),
        "legal_action_ids": np.tile(np.asarray([[0, 1]], dtype=np.int16), (n, 1)),
        "target_policy": np.tile(np.asarray([[0.6, 0.4]], dtype=np.float32), (n, 1)),
        "target_scores": np.tile(np.asarray([[0.2, -0.1]], dtype=np.float32), (n, 1)),
        "target_information_regime": np.asarray(regimes),
    }
    if root_value:
        data["root_value"] = np.zeros(n, dtype=np.float32)
        data["root_value_mask"] = np.ones(n, dtype=np.bool_)
    if prior_policy:
        data["prior_policy"] = np.tile(
            np.asarray([[0.5, 0.5]], dtype=np.float32), (n, 1)
        )
    return data


def _admit(data: dict, **overrides):
    kwargs = {
        "mask_hidden_info": True,
        "soft_target_weight": 0.7,
        "policy_loss_weight": 1.0,
        "q_loss_weight": 0.0,
        "value_target_lambda": 1.0,
        "policy_kl_anchor_weight": 0.0,
        "policy_surprise_weight": 0.0,
    }
    kwargs.update(overrides)
    return _validate_target_information_admission(data, **kwargs)


def test_public_information_set_targets_are_admitted():
    report = _admit(
        _data([TARGET_INFORMATION_REGIME_PUBLIC] * 3),
        required_target_information_regime=TARGET_INFORMATION_REGIME_PUBLIC,
    )
    assert report["unsafe_or_unknown_rows"] == 0
    assert report["search_target_objectives"] == ["soft_policy"]


def test_coherent_public_belief_targets_are_admitted():
    report = _admit(_data([TARGET_INFORMATION_REGIME_PUBLIC_COHERENT] * 3))
    assert report["unsafe_or_unknown_rows"] == 0
    assert report["required_target_information_regime"] == (
        TARGET_INFORMATION_REGIME_PUBLIC_COHERENT
    )
    assert report["search_target_objectives"] == ["soft_policy"]


def test_public_pimc_is_not_silently_substituted_for_coherent_teacher():
    with pytest.raises(SystemExit, match="different policy-improvement operators"):
        _admit(_data([TARGET_INFORMATION_REGIME_PUBLIC] * 3))


def test_public_teacher_regimes_cannot_be_mixed():
    with pytest.raises(SystemExit, match="mismatched search targets"):
        _admit(
            _data(
                [
                    TARGET_INFORMATION_REGIME_PUBLIC_COHERENT,
                    TARGET_INFORMATION_REGIME_PUBLIC,
                ]
            )
        )


@pytest.mark.parametrize(
    "regime", ["authoritative_hidden_state_search_v1", "unknown", ""]
)
def test_masked_training_rejects_unsafe_or_unknown_soft_targets(regime: str):
    with pytest.raises(SystemExit, match="public-observation training refused"):
        _admit(_data([regime]))


def test_masked_training_rejects_unsafe_q_and_root_value_targets():
    data = _data(["authoritative_hidden_state_search_v1"], root_value=True)
    with pytest.raises(SystemExit, match="q_target.*root_value"):
        _admit(
            data,
            soft_target_weight=0.0,
            q_loss_weight=0.2,
            value_target_lambda=0.5,
        )


def test_unsafe_corpus_can_only_use_hard_actions_and_realised_outcomes():
    report = _admit(
        _data(["authoritative_hidden_state_search_v1"]),
        soft_target_weight=0.0,
        q_loss_weight=0.0,
        value_target_lambda=1.0,
    )
    assert report["unsafe_or_unknown_rows"] == 1
    assert report["search_target_objectives"] == []


@pytest.mark.parametrize(
    ("overrides", "objective"),
    [
        ({"policy_kl_anchor_weight": 0.1}, "policy_kl_anchor"),
        ({"policy_surprise_weight": 0.1}, "policy_surprise_sampling"),
    ],
)
def test_unsafe_corpus_cannot_bypass_admission_via_policy_auxiliary(
    overrides, objective: str
):
    data = _data(["authoritative_hidden_state_search_v1"], prior_policy=True)
    with pytest.raises(SystemExit, match=objective):
        _admit(
            data,
            soft_target_weight=0.0,
            q_loss_weight=0.0,
            value_target_lambda=1.0,
            **overrides,
        )


def test_unmasked_training_keeps_explicit_omniscient_experiment_available():
    report = _admit(
        _data(["authoritative_hidden_state_search_v1"]),
        mask_hidden_info=False,
    )
    assert report["unsafe_or_unknown_rows"] == 1
