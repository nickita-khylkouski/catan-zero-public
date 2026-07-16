"""Fail-closed provenance checks for the default scalar search readout."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from catan_zero.search.neural_rust_mcts import (
    EntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)


def _policy(
    *,
    trained_readouts: tuple[str, ...],
    value_training: dict[str, object] | None,
    provenance_errors: tuple[str, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        trained_with_masked_hidden_info=False,
        trained_value_readouts=trained_readouts,
        value_training=value_training,
        _value_training_provenance_errors=provenance_errors,
        model=SimpleNamespace(
            value_categorical_bins=33,
            value_categorical_head=object(),
        ),
    )


def test_modern_categorical_only_checkpoint_rejects_default_scalar_readout() -> None:
    policy = _policy(
        trained_readouts=("categorical",),
        value_training={
            "schema_version": "value-training-v1",
            "primary_readout": "categorical",
            "trained_value_readouts": ["categorical"],
            "resolved_scalar_mse_weight": 0.0,
            "resolved_categorical_ce_weight": 0.25,
        },
    )

    with pytest.raises(
        ValueError, match="does not attest.*scalar readout was optimized"
    ):
        EntityGraphRustEvaluator(policy, config=EntityGraphRustEvaluatorConfig())

    # The trained head remains usable; the fix does not change readout choice.
    EntityGraphRustEvaluator(
        policy,
        config=EntityGraphRustEvaluatorConfig(value_readout="categorical"),
    )


def test_invalid_modern_scalar_attestation_rejects_scalar_readout() -> None:
    policy = _policy(
        trained_readouts=(),
        value_training={
            "schema_version": "value-training-v1",
            "trained_value_readouts": ["scalar"],
        },
        provenance_errors=("scalar objective mass is non-positive",),
    )

    with pytest.raises(ValueError, match="scalar objective mass is non-positive"):
        EntityGraphRustEvaluator(policy, config=EntityGraphRustEvaluatorConfig())


def test_modern_scalar_attestation_and_legacy_absence_remain_accepted() -> None:
    modern = _policy(
        trained_readouts=("scalar",),
        value_training={
            "schema_version": "value-training-v1",
            "primary_readout": "scalar",
            "trained_value_readouts": ["scalar"],
            "resolved_scalar_mse_weight": 0.25,
        },
    )
    legacy = _policy(trained_readouts=("scalar",), value_training=None)

    EntityGraphRustEvaluator(modern, config=EntityGraphRustEvaluatorConfig())
    EntityGraphRustEvaluator(legacy, config=EntityGraphRustEvaluatorConfig())
