from __future__ import annotations

import numpy as np
import pytest

from tools.train_bc import (
    POLICY_TARGET_BLEND_FALLBACK_V2,
    POLICY_TARGET_BLEND_LEGACY_V1,
    TARGET_INFORMATION_REGIME_PUBLIC,
    TARGET_INFORMATION_REGIME_PUBLIC_COHERENT,
    _validate_target_information_admission,
)


class _ScopedPolicyData(dict):
    component_ids = ("current", "historical_replay")
    policy_distillation_component_indices = (0,)
    policy_distillation_scope_authenticated = True

    @staticmethod
    def component_indices_for_rows(rows):
        return np.asarray(rows, dtype=np.int64)


def _data(
    regimes: list[str], *, root_value: bool = False, prior_policy: bool = False
) -> dict:
    n = len(regimes)
    data = {
        "action_taken": np.zeros(n, dtype=np.int16),
        "legal_action_ids": np.tile(np.asarray([[0, 1]], dtype=np.int16), (n, 1)),
        "target_policy": np.tile(np.asarray([[0.6, 0.4]], dtype=np.float32), (n, 1)),
        "target_policy_mask": np.ones((n, 2), dtype=np.bool_),
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
        "soft_target_weight": 1.0,
        "policy_target_blend_semantics": POLICY_TARGET_BLEND_FALLBACK_V2,
        "policy_loss_weight": 1.0,
        "q_loss_weight": 0.0,
        "value_target_lambda": 1.0,
        "policy_kl_anchor_weight": 0.0,
        "policy_surprise_weight": 0.0,
        "soft_target_min_legal_coverage": 1.0,
    }
    kwargs.update(overrides)
    return _validate_target_information_admission(data, **kwargs)


def test_public_information_set_targets_are_admitted():
    report = _admit(
        _data([TARGET_INFORMATION_REGIME_PUBLIC] * 3),
        required_target_information_regime=TARGET_INFORMATION_REGIME_PUBLIC,
        soft_target_weight=0.7,
        policy_target_blend_semantics=POLICY_TARGET_BLEND_LEGACY_V1,
    )
    assert report["unsafe_or_unknown_rows"] == 0
    assert report["search_target_objectives"] == ["soft_policy"]


def test_coherent_public_belief_targets_are_admitted():
    report = _admit(
        _data([TARGET_INFORMATION_REGIME_PUBLIC_COHERENT] * 3),
        soft_target_weight=1.0,
        policy_target_blend_semantics=POLICY_TARGET_BLEND_FALLBACK_V2,
    )
    assert report["unsafe_or_unknown_rows"] == 0
    assert report["required_target_information_regime"] == (
        TARGET_INFORMATION_REGIME_PUBLIC_COHERENT
    )
    assert report["search_target_objectives"] == ["soft_policy"]
    assert report["policy_target_completeness"]["hard_action_fallback_rows"] == 0


def test_aux_only_completed_q_is_an_active_search_target() -> None:
    data = _data([TARGET_INFORMATION_REGIME_PUBLIC_COHERENT] * 3)
    data["completed_q_values"] = np.tile(
        np.asarray([[0.25, -0.25]], dtype=np.float32), (3, 1)
    )
    report = _admit(
        data,
        soft_target_weight=0.0,
        policy_loss_weight=0.0,
        completed_q_loss_weight=0.0,
        policy_aux_completed_q_loss_weight=0.1,
    )
    assert report["search_target_objectives"] == ["completed_q_target"]


def test_coherent_public_belief_rejects_sparse_soft_target_fallback():
    data = _data([TARGET_INFORMATION_REGIME_PUBLIC_COHERENT])
    data["target_policy_mask"][0, 1] = False
    data["target_policy"][0, 1] = 0.0
    data["target_policy"][0, 0] = 1.0

    with pytest.raises(SystemExit, match="every policy-active row"):
        _admit(data)


@pytest.mark.parametrize(
    ("semantics", "weight"),
    [
        (POLICY_TARGET_BLEND_LEGACY_V1, 0.9),
        (POLICY_TARGET_BLEND_FALLBACK_V2, 0.9),
    ],
)
def test_coherent_policy_targets_reject_played_action_blending(
    semantics: str, weight: float
):
    with pytest.raises(SystemExit, match="requires|forbids"):
        _admit(
            _data([TARGET_INFORMATION_REGIME_PUBLIC_COHERENT]),
            soft_target_weight=weight,
            policy_target_blend_semantics=semantics,
        )


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


def test_authenticated_policy_scope_excludes_inactive_legacy_teacher() -> None:
    data = _ScopedPolicyData(
        _data(
            [
                TARGET_INFORMATION_REGIME_PUBLIC_COHERENT,
                TARGET_INFORMATION_REGIME_PUBLIC,
            ]
        )
    )
    data["policy_weight_multiplier"] = np.ones(2, dtype=np.float32)

    report = _admit(data)

    assert report["search_objective_active_rows"] == 1
    assert report["search_objective_target_information_regime_counts"] == {
        TARGET_INFORMATION_REGIME_PUBLIC_COHERENT: 1
    }
    assert report["mismatched_target_information_rows"] == 0
    completeness = report["policy_target_completeness"]
    assert completeness["policy_active_rows"] == 1
    assert completeness["exact_complete_public_soft_target_rows"] == 1


def test_legacy_rows_remain_usable_for_value_only_rehearsal():
    data = _data(
        [
            TARGET_INFORMATION_REGIME_PUBLIC_COHERENT,
            TARGET_INFORMATION_REGIME_PUBLIC,
        ]
    )
    data["policy_weight_multiplier"] = np.asarray([1.0, 0.0], dtype=np.float32)
    report = _admit(data)
    assert report["search_objective_active_rows"] == 1
    assert report["mismatched_target_information_rows"] == 0


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
        required_target_information_regime=TARGET_INFORMATION_REGIME_PUBLIC,
        soft_target_weight=0.7,
        policy_target_blend_semantics=POLICY_TARGET_BLEND_LEGACY_V1,
    )
    assert report["unsafe_or_unknown_rows"] == 1
