from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools import a1_b200_active_policy_campaign as stage_a
from tools import a1_b200_stage_b_ablation_campaign as campaign
from tools import train_bc


def _selected_fixture(tmp_path: Path) -> tuple[Path, dict, Path, dict]:
    checkpoint = tmp_path / "P50-step0032.pt"
    checkpoint.write_bytes(b"selected-stage-a-dose")
    campaign_path = tmp_path / "stage-a.campaign.json"
    source_arm = {
        **stage_a.ARMS["P50"],
        "optimizer_steps": 128,
        "recipe_overrides": {},
    }
    source = {
        "campaign_sha256": "sha256:" + "1" * 64,
        "arms": {"P50": source_arm},
    }
    campaign_path.write_text(json.dumps(source), encoding="utf-8")
    selection_path = tmp_path / "stage-a.selection.json"
    selection = {
        "selection_sha256": "sha256:" + "2" * 64,
        "campaign": {
            "path": str(campaign_path.resolve()),
            "file_sha256": campaign._file_sha256(campaign_path),
            "campaign_sha256": source["campaign_sha256"],
        },
        "winner": "P50",
        "winner_step": 32,
        "winner_recipe": {
            **source_arm,
            "selected_optimizer_steps": 32,
        },
        "winner_candidate": {
            "arm": "P50",
            "step": 32,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": campaign._file_sha256(checkpoint),
            "eligible": True,
            "parent_kl": 0.012,
            "trunk_relative_l2": 0.006,
            "teacher_gap_closure": 0.05,
        },
        "winner_checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": campaign._file_sha256(checkpoint),
        },
        "winner_is_diagnostic_not_promoted": True,
        "playing_strength_evaluation_still_required": True,
        "candidate_chaining": False,
    }
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    return selection_path.resolve(), selection, campaign_path.resolve(), source


def test_selected_dose_binds_stage_a_arm_multiplier_step_and_checkpoint(
    tmp_path: Path,
) -> None:
    selection_path, selection, campaign_path, source = _selected_fixture(tmp_path)

    dose = campaign._selected_dose(  # noqa: SLF001
        selection_path=selection_path,
        selection=selection,
        campaign_path=campaign_path,
        campaign=source,
    )

    assert dose["selected_arm"] == "P50"
    assert dose["active_policy_branch_multiplier"] == pytest.approx(0.5)
    assert dose["policy_aux_active_batch_size"] == 232
    assert dose["optimizer_steps"] == 32
    assert dose["checkpoint_steps"] == [8, 12, 16, 32]
    assert dose["expected_aux_active_row_draws"] == 232 * 8 * 32
    assert dose["reference_parent_kl"] == pytest.approx(0.012)
    assert dose["reference_trunk_relative_l2"] == pytest.approx(0.006)
    assert dose["stage_a_selected_checkpoint"]["role"] == (
        "dose_evidence_only_never_initializer"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selected_optimizer_steps", 64, "dose binding drifted"),
        ("policy_aux_active_batch_size", 463, "dose binding drifted"),
        ("active_policy_branch_multiplier", 1.0, "dose binding drifted"),
    ],
)
def test_selected_dose_refuses_selection_recipe_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    selection_path, selection, campaign_path, source = _selected_fixture(tmp_path)
    selection["winner_recipe"][field] = value

    with pytest.raises(campaign.CampaignError, match=message):
        campaign._selected_dose(  # noqa: SLF001
            selection_path=selection_path,
            selection=selection,
            campaign_path=campaign_path,
            campaign=source,
        )


def test_stage_b_arms_change_exactly_one_treatment_at_selected_dose() -> None:
    dose = {
        "optimizer_steps": 32,
        "policy_aux_active_batch_size": 232,
        "reference_parent_kl": 0.012,
    }
    source_recipe = {"lr": 6.0e-5, "lr_warmup_steps": 16}
    arms = {
        arm: {
            "recipe_overrides": campaign._arm_overrides(  # noqa: SLF001
                arm, selected_dose=dose, source_recipe=source_recipe
            )
        }
        for arm in campaign.ARM_ORDER
    }

    campaign._assert_treatment_isolation(arms)  # noqa: SLF001

    assert all(row["recipe_overrides"]["max_steps"] == 32 for row in arms.values())
    assert all(
        row["recipe_overrides"]["policy_aux_active_batch_size"] == 232
        for row in arms.values()
    )
    assert "forced_row_value_action_type_weights" not in arms["BASE"][
        "recipe_overrides"
    ]
    assert arms["FORCED"]["recipe_overrides"][
        "forced_row_value_action_type_weights"
    ] == "END_TURN=0.1,ROLL=0.25"
    assert arms["CARD4"]["recipe_overrides"]["public_card_lr_mult"] == 4.0
    assert arms["SURPRISE"]["recipe_overrides"][
        "per_game_policy_surprise_weighting"
    ] is True
    assert arms["TRUNK25"]["recipe_overrides"]["trunk_lr_mult"] == 0.25
    assert arms["TRUNK10"]["recipe_overrides"]["trunk_lr_mult"] == 0.10
    assert arms["VTRUNK25"]["recipe_overrides"][
        "value_trunk_grad_scale"
    ] == 0.25
    assert arms["VTRUNK25"]["recipe_overrides"]["trunk_lr_mult"] == 1.0
    assert arms["TRUST"]["recipe_overrides"]["policy_kl_target"] == pytest.approx(
        0.012
    )
    assert arms["TRUST"]["recipe_overrides"]["policy_kl_anchor_direction"] == (
        "forward"
    )

    broken = copy.deepcopy(arms)
    broken["CARD4"]["recipe_overrides"]["lr"] = 1.2e-4
    with pytest.raises(campaign.CampaignError, match="outside declared treatment"):
        campaign._assert_treatment_isolation(broken)  # noqa: SLF001


class _FakeLegalColumn:
    def __init__(self, counts: list[int]):
        self._counts = np.asarray(counts, dtype=np.int64)

    def row_counts(self) -> np.ndarray:
        return self._counts


class _FakeCorpus:
    def __init__(
        self,
        *,
        legal_counts: list[int],
        action_taken: list[int],
        stored_forced: list[bool],
    ) -> None:
        prior_policy = np.zeros((len(legal_counts), max(legal_counts)), dtype=np.float32)
        for row, count in enumerate(legal_counts):
            if count > 1:
                prior_policy[row, :count] = 1.0 / count
        self._values = {
            "legal_action_ids": _FakeLegalColumn(legal_counts),
            "action_taken": np.asarray(action_taken, dtype=np.int64),
            "is_forced": np.asarray(stored_forced, dtype=np.bool_),
            "prior_policy": prior_policy,
        }

    def __contains__(self, key: str) -> bool:
        return key in self._values

    def __getitem__(self, key: str):
        return self._values[key]

    def __len__(self) -> int:
        return len(self._values["action_taken"])


class _FakeCatalog:
    size = 3

    @staticmethod
    def describe(action_id: int) -> dict[str, str]:
        return {
            "action_type": ("END_TURN", "ROLL", "BUILD_SETTLEMENT")[action_id]
        }


def test_forced_treatment_is_structurally_inactive_without_typed_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "corpus_meta.json").write_text("{}\n", encoding="utf-8")
    fake = _FakeCorpus(
        legal_counts=[3, 8, 2], action_taken=[2, 2, 2], stored_forced=[False] * 3
    )
    monkeypatch.setattr(train_bc, "MemmapCorpus", lambda _path: fake)
    monkeypatch.setattr(train_bc, "parse_track", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        train_bc, "_action_catalog_for_env_config", lambda _config: _FakeCatalog()
    )

    exposure = campaign._treatment_exposure(corpus.resolve())  # noqa: SLF001
    exposure.update(
        campaign._value_trunk_treatment_exposure(  # noqa: SLF001
            {"value_loss_weight": 0.25}, value_attention_pool=False
        )
    )
    exposure.pop("exposure_sha256")
    exposure["exposure_sha256"] = campaign._value_sha256(exposure)  # noqa: SLF001

    assert exposure["stored_is_forced_rows"] == 0
    assert exposure["one_legal_action_rows"] == 0
    assert exposure["typed_forced_rows"] == 0
    assert exposure["forced_treatment_structurally_active"] is False
    assert exposure["policy_kl_anchor_multi_action_rows"] == 3
    assert exposure["trust_treatment_structurally_active"] is True
    assert campaign._active_arms_for_exposure(exposure) == [  # noqa: SLF001
        "BASE",
        "CARD4",
        "SURPRISE",
        "TRUNK25",
        "TRUNK10",
        "VTRUNK25",
        "TRUST",
    ]
    assert exposure["exposure_sha256"] == campaign._value_sha256(  # noqa: SLF001
        {key: value for key, value in exposure.items() if key != "exposure_sha256"}
    )


def test_forced_treatment_activates_only_for_typed_one_legal_action_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "corpus_meta.json").write_text("{}\n", encoding="utf-8")
    fake = _FakeCorpus(
        legal_counts=[1, 1, 1, 4],
        action_taken=[0, 1, 2, 0],
        stored_forced=[True, True, True, False],
    )
    monkeypatch.setattr(train_bc, "MemmapCorpus", lambda _path: fake)
    monkeypatch.setattr(train_bc, "parse_track", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        train_bc, "_action_catalog_for_env_config", lambda _config: _FakeCatalog()
    )

    exposure = campaign._treatment_exposure(corpus.resolve())  # noqa: SLF001
    exposure.update(
        campaign._value_trunk_treatment_exposure(  # noqa: SLF001
            {"value_loss_weight": 0.25}, value_attention_pool=False
        )
    )
    exposure.pop("exposure_sha256")
    exposure["exposure_sha256"] = campaign._value_sha256(exposure)  # noqa: SLF001

    assert exposure["one_legal_action_rows"] == 3
    assert exposure["typed_forced_rows"] == 2
    assert exposure["typed_forced_rows_by_action_type"] == {
        "END_TURN": 1,
        "ROLL": 1,
    }
    assert exposure["forced_treatment_structurally_active"] is True
    assert campaign._active_arms_for_exposure(exposure) == [  # noqa: SLF001
        "BASE",
        "FORCED",
        "CARD4",
        "SURPRISE",
        "TRUNK25",
        "TRUNK10",
        "VTRUNK25",
        "TRUST",
    ]


@pytest.mark.parametrize(
    ("value_loss_weight", "value_attention_pool", "active", "reason"),
    [
        (0.25, False, True, None),
        (0.0, False, False, "zero_scalar_mse_value_objective"),
        (
            0.25,
            True,
            False,
            "value_attention_pool_bypasses_single_shared_state_boundary",
        ),
    ],
)
def test_value_trunk_arm_is_excluded_when_its_boundary_is_inert_or_bypassed(
    value_loss_weight: float,
    value_attention_pool: bool,
    active: bool,
    reason: str | None,
) -> None:
    exposure = {
        "typed_forced_rows": 0,
        "forced_treatment_structurally_active": False,
        "policy_kl_anchor_multi_action_rows": 1,
        "trust_treatment_structurally_active": True,
    }
    exposure.update(
        campaign._value_trunk_treatment_exposure(  # noqa: SLF001
            {"value_loss_weight": value_loss_weight},
            value_attention_pool=value_attention_pool,
        )
    )

    arms = campaign._active_arms_for_exposure(exposure)  # noqa: SLF001

    assert ("VTRUNK25" in arms) is active
    assert exposure["value_trunk_treatment_inactive_reason"] == reason


def test_value_trunk_runtime_receipt_must_prove_exact_boundary() -> None:
    report = {
        "value_trunk_grad_scale": 0.25,
        "value_gradient_routing": {
            "schema_version": "scalar-value-trunk-gradient-routing-v1",
            "scalar_value_trunk_grad_scale": 0.25,
            "active": True,
            "forward_value_identity": True,
            "value_head_parameter_gradient_scale": 1.0,
            "shared_state_upstream_gradient_scale": 0.25,
            "scope": "scalar_value_head_state_input_only",
            "policy_gradient_unchanged": True,
            "optimizer_parameter_groups_unchanged": True,
        },
    }

    routing = campaign._verify_value_trunk_routing(  # noqa: SLF001
        report, arm="VTRUNK25", expected_scale=0.25
    )
    assert routing["active"] is True

    broken = copy.deepcopy(report)
    broken["value_gradient_routing"]["shared_state_upstream_gradient_scale"] = 1.0
    with pytest.raises(campaign.CampaignError, match="did not execute"):
        campaign._verify_value_trunk_routing(  # noqa: SLF001
            broken, arm="VTRUNK25", expected_scale=0.25
        )


def test_effective_recipe_refuses_hidden_second_treatment() -> None:
    dose = {
        "optimizer_steps": 64,
        "policy_aux_active_batch_size": 116,
        "reference_parent_kl": 0.012,
    }
    source_recipe = {"lr": 6.0e-5, "lr_warmup_steps": 16}
    arms = {
        arm: {
            "recipe_overrides": campaign._arm_overrides(  # noqa: SLF001
                arm, selected_dose=dose, source_recipe=source_recipe
            )
        }
        for arm in campaign.ARM_ORDER
    }
    plan = {"arms": arms}
    effective = copy.deepcopy(arms["CARD4"]["recipe_overrides"])
    effective["per_game_policy_surprise_weighting"] = True

    with pytest.raises(campaign.CampaignError, match="effective treatment/dose drift"):
        campaign._effective_treatment_assertion(  # noqa: SLF001
            plan, "CARD4", effective
        )


def test_dose_match_uses_parent_kl_and_trunk_drift_not_teacher_outcome() -> None:
    selected = campaign._select_dose_matched_checkpoint(  # noqa: SLF001
        [
            {
                "step": 16,
                "parent_kl": 0.0105,
                "trunk_relative_l2": 0.0055,
                "teacher_gap_closure": 0.01,
            },
            {
                "step": 32,
                "parent_kl": 0.030,
                "trunk_relative_l2": 0.015,
                "teacher_gap_closure": 0.90,
            },
        ],
        reference_parent_kl=0.010,
        reference_trunk_relative_l2=0.005,
        terminal_step=32,
    )

    assert selected["step"] == 16
    assert selected["teacher_gap_closure"] == pytest.approx(0.01)
    assert selected["parent_kl_ratio_to_stage_a_reference"] == pytest.approx(1.05)
