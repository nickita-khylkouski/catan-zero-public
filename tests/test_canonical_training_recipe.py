from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from tools import train


REPO = Path(__file__).resolve().parents[1]
RECIPE = REPO / "configs/training/a1_current_35m_b200.schema1.json"
PARENT_RECIPE = REPO / "configs/training/a1_parent_update_35m_b200.schema1.json"


def _payload() -> dict[str, object]:
    return json.loads(RECIPE.read_text(encoding="utf-8"))


def _fields() -> dict[str, object]:
    return _payload()["train_config"]["fields"]


def _public_args(*, init_checkpoint: str = "") -> argparse.Namespace:
    return argparse.Namespace(
        data="/tmp/data",
        checkpoint="/tmp/candidate.pt",
        report="/tmp/train.json",
        init_checkpoint=init_checkpoint,
        device="auto",
        host_lock_file="/tmp/catan-test.lock",
        allow_concurrent_bc=False,
    )


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
    with pytest.raises(SystemExit, match="forbids parent/grow checkpoints"):
        train._engine_namespace(
            config=config,
            engine_settings=engine,
            public_args=_public_args(init_checkpoint="/tmp/previous-candidate.pt"),
        )


@pytest.mark.parametrize(
    ("recipe", "role"),
    (
        (RECIPE, "scratch_fresh_optimizer"),
        (PARENT_RECIPE, "parent_fresh_optimizer"),
    ),
)
def test_canonical_recipe_hash_allowlist_is_bound_to_role(
    recipe: Path, role: str
) -> None:
    payload = json.loads(recipe.read_text(encoding="utf-8"))
    digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()

    assert payload["engine_settings"]["initialization_mode"] == role
    assert train.CANONICAL_CONFIG_ROLES_BY_SHA256[digest] == role
    train._load_recipe(recipe)


def test_parent_update_recipe_reproduces_split1_full_step48() -> None:
    payload = json.loads(PARENT_RECIPE.read_text(encoding="utf-8"))
    engine = payload["engine_settings"]
    fields = payload["train_config"]["fields"]

    assert engine["initialization_mode"] == "parent_fresh_optimizer"
    assert engine["base_sampler"] == "weighted_replacement_v1"
    assert engine["checkpoint_steps"] == "8,12,16,24,32,48"
    assert engine["accepted_policy_target_identity_sha256"] == [
        "sha256:d1f6686a2f00012aa54a729f4850e1333d59e57783d323c6a2d2d2a15ab02fed"
    ]
    assert fields["batch_size"] == 64
    assert 8 * fields["batch_size"] == 512
    assert fields["epochs"] == 999
    assert fields["max_steps"] == 48
    assert fields["exact_max_steps"] is True
    assert fields["lr"] == 6e-5
    assert fields["lr_schedule"] == "flat"
    assert fields["lr_warmup_steps"] == 16
    assert fields["validation_fraction"] == 0.125
    assert fields["validation_game_seed_ranges"] == (
        "96000000000:96000000007,96000010000:96000010007,"
        "96000020000:96000020007,96000030000:96000030007,"
        "96000040000:96000040007,96000050000:96000050007,"
        "96000060000:96000060007,96000070000:96000070007"
    )
    assert fields["validation_max_samples"] == 0
    assert fields["minimum_policy_effective_rows_per_global_batch"] == 0.0
    assert fields["value_trunk_grad_scale"] == 1.0
    assert fields["post_policy_dose_value_trunk_grad_scale"] == 1.0
    assert fields["policy_dose_lr_area"] == 0.0
    assert fields["policy_kl_target"] is None
    assert fields["policy_kl_anchor_weight"] == 0.0
    assert fields["freeze_modules"] == ""
    assert fields["forced_action_weight"] == 0.0
    assert fields["forced_row_value_weight"] == 1.0
    assert fields["forced_row_value_action_type_weights"] == (
        "END_TURN=1.0,ROLL=1.0"
    )
    assert fields["resume_optimizer"] is False
    assert fields["init_checkpoint"] == ""
    assert fields["grow_from_checkpoint"] == ""
    assert engine["value_tower_split_layers"] == 1
    assert payload["train_config"]["schema_version"] == 19


def test_parent_update_requires_exact_parent_and_forwards_teacher_identity() -> None:
    config, engine = train._load_recipe(PARENT_RECIPE)
    with pytest.raises(SystemExit, match="requires --init-checkpoint"):
        train._engine_namespace(
            config=config,
            engine_settings=engine,
            public_args=_public_args(),
        )

    resolved = train._engine_namespace(
        config=config,
        engine_settings=engine,
        public_args=_public_args(init_checkpoint="/tmp/f7.pt"),
    )
    assert resolved.init_checkpoint == "/tmp/f7.pt"
    assert resolved.resume_optimizer is False
    assert resolved.grow_from_checkpoint == ""
    assert resolved.accepted_policy_target_identity_sha256 == [
        "sha256:d1f6686a2f00012aa54a729f4850e1333d59e57783d323c6a2d2d2a15ab02fed"
    ]

    from tools import train_bc

    assert resolved.checkpoint_steps == "8,12,16,24,32"
    assert train_bc._parse_checkpoint_steps(
        resolved.checkpoint_steps, max_steps=resolved.max_steps
    ) == (8, 12, 16, 24, 32)
    train_bc._validate_coverage_sampler_configuration(
        resolved, categorical_value_loss_weight=0.0
    )


def test_parent_update_rejects_growth_or_optimizer_resume() -> None:
    base, engine = train._load_recipe(PARENT_RECIPE)
    variants = (
        dataclasses.replace(base, grow_from_checkpoint="/tmp/grow.pt"),
        dataclasses.replace(base, resume_optimizer=True),
    )
    for variant in variants:
        with pytest.raises(SystemExit, match="fresh optimizer"):
            train._engine_namespace(
                config=variant,
                engine_settings=engine,
                public_args=_public_args(init_checkpoint="/tmp/f7.pt"),
            )
