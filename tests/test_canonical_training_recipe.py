from __future__ import annotations

import json
import argparse
from pathlib import Path

import pytest

from tools import train


REPO = Path(__file__).resolve().parents[1]
RECIPE = REPO / "configs/training/a1_current_35m_b200.schema1.json"


def _payload() -> dict[str, object]:
    return json.loads(RECIPE.read_text(encoding="utf-8"))


def _fields() -> dict[str, object]:
    return _payload()["train_config"]["fields"]


def test_canonical_coverage_recipe_can_reach_composite_training() -> None:
    """Keep the public recipe compatible with train_bc's composite admission.

    Authenticated composites require a complete whole-game validation split,
    represented by the zero row-cap sentinel.  A nonzero cap is rejected before
    the first optimizer step.
    """

    fields = _fields()
    assert _payload()["engine_settings"]["base_sampler"] == "coverage_importance_v1"
    assert "minimum_policy_effective_rows_per_global_batch" not in _payload()[
        "engine_settings"
    ]
    assert fields["minimum_policy_effective_rows_per_global_batch"] == 32.0
    assert fields["data_format"] == "memmap"
    assert fields["validation_max_samples"] == 0


def test_canonical_non_moe_recipe_has_no_phantom_moe_objective() -> None:
    """Coverage normalization must not bind an objective with no live head."""

    fields = _fields()
    assert fields["moe_routed_experts"] == 0
    assert fields["moe_expert_ff_size"] == 0
    assert fields["moe_balance_loss_weight"] == 0.0


def test_canonical_forced_value_baseline_preserves_boundary_evidence() -> None:
    """Forced policy is inert while turn-boundary value evidence is retained."""

    fields = _fields()
    assert fields["forced_action_weight"] == 0.0
    assert fields["forced_row_value_weight"] == 1.0
    assert fields["forced_row_value_action_type_weights"] == (
        "END_TURN=1.0,ROLL=1.0"
    )


def test_scratch_horizon_is_not_relabelled_as_the_parent_update_frontier() -> None:
    """The proven 32-step result was a parent update, not scratch evidence."""

    fields = _fields()
    assert fields["init_checkpoint"] == ""
    assert fields["resume_optimizer"] is False
    assert fields["max_steps"] == 0
    assert fields["epochs"] == 3


def test_canonical_scratch_recipe_rejects_candidate_chaining() -> None:
    config, engine = train._load_recipe(RECIPE)
    assert engine["initialization_mode"] == "scratch_fresh_optimizer"
    public = argparse.Namespace(
        data="/tmp/data",
        checkpoint="/tmp/candidate.pt",
        report="/tmp/train.json",
        init_checkpoint="/tmp/previous-candidate.pt",
        device="auto",
        host_lock_file="/tmp/catan-test.lock",
        allow_concurrent_bc=False,
    )
    with pytest.raises(SystemExit, match="forbids parent/grow checkpoints"):
        train._engine_namespace(
            config=config,
            engine_settings=engine,
            public_args=public,
        )
