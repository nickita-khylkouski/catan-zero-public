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
    assert dose["policy_aux_active_batch_size"] == 128
    assert dose["optimizer_steps"] == 32
    assert dose["checkpoint_steps"] == [8, 12, 16, 32]
    assert dose["expected_aux_active_row_draws"] == 128 * 8 * 32
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
    assert (
        "forced_row_value_action_type_weights" not in arms["BASE"]["recipe_overrides"]
    )
    assert (
        arms["FORCED"]["recipe_overrides"]["forced_row_value_action_type_weights"]
        == "END_TURN=0.1,ROLL=0.25"
    )
    assert arms["CARD4"]["recipe_overrides"]["public_card_lr_mult"] == 4.0
    assert (
        arms["SURPRISE"]["recipe_overrides"]["per_game_policy_surprise_weighting"]
        is True
    )
    assert arms["TRUNK25"]["recipe_overrides"]["trunk_lr_mult"] == 0.25
    assert arms["TRUNK10"]["recipe_overrides"]["trunk_lr_mult"] == 0.10
    assert arms["VTRUNK25"]["recipe_overrides"]["value_trunk_grad_scale"] == 0.25
    assert arms["VTRUNK25"]["recipe_overrides"]["trunk_lr_mult"] == 1.0
    assert arms["TRUST"]["recipe_overrides"]["policy_kl_target"] == pytest.approx(0.012)
    assert arms["TRUST"]["recipe_overrides"]["policy_kl_anchor_direction"] == (
        "forward"
    )

    broken = copy.deepcopy(arms)
    broken["CARD4"]["recipe_overrides"]["lr"] = 1.2e-4
    with pytest.raises(campaign.CampaignError, match="outside declared treatment"):
        campaign._assert_treatment_isolation(broken)  # noqa: SLF001


def test_trust_arm_inherits_the_exact_selected_recovery_contract() -> None:
    dose = {
        "optimizer_steps": 32,
        "policy_aux_active_batch_size": 463,
        "reference_parent_kl": 0.021,
        "trust_contract": {
            "policy_kl_anchor_direction": "forward",
            "policy_kl_target": 0.027,
            "policy_kl_dual_lr": 0.75,
            "policy_kl_max_weight": 1.5,
        },
    }
    recipe = campaign._arm_overrides(  # noqa: SLF001
        "TRUST",
        selected_dose=dose,
        source_recipe={"lr": 6.0e-5, "lr_warmup_steps": 16},
    )

    assert recipe["policy_kl_target"] == pytest.approx(0.027)
    assert recipe["policy_kl_dual_lr"] == pytest.approx(0.75)
    assert recipe["policy_kl_max_weight"] == pytest.approx(1.5)


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
        prior_policy = np.zeros(
            (len(legal_counts), max(legal_counts)), dtype=np.float32
        )
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
        return {"action_type": ("END_TURN", "ROLL", "BUILD_SETTLEMENT")[action_id]}


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
        (0.25, True, True, None),
    ],
)
def test_value_trunk_arm_is_excluded_only_when_scalar_objective_is_inert(
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
            "scope": "scalar_value_readout_all_shared_inputs",
            "legacy_scope_alias": "scalar_value_head_state_input_only",
            "shared_input_paths": ["cls_state"],
            "value_attention_pool_enabled": False,
            "all_scalar_value_shared_inputs_scaled": True,
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


def _signed(path: Path, payload: dict, digest_field: str) -> Path:
    value = copy.deepcopy(payload)
    value[digest_field] = campaign._value_sha256(value)  # noqa: SLF001
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path.resolve()


def _recovery_selection_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict]:
    source_root = tmp_path / "source"
    tools = source_root / "tools"
    tools.mkdir(parents=True)
    trainer = tools / "a1_one_dose_train.py"
    train_bc_path = tools / "train_bc.py"
    trainer.write_text("# trainer\n", encoding="utf-8")
    train_bc_path.write_text("# train_bc\n", encoding="utf-8")
    operator = tmp_path / "recovery_operator.py"
    operator.write_text("# operator\n", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    (data / "corpus_meta.json").write_text("{}\n", encoding="utf-8")
    validation = tmp_path / "validation.json"
    admission = tmp_path / "admission.json"
    upgrade = tmp_path / "upgrade.json"
    for artifact in (validation, admission, upgrade):
        artifact.write_text("{}\n", encoding="utf-8")
    stage_a_path = tmp_path / "stage-a.json"
    stage_a_path.write_text("{}\n", encoding="utf-8")
    stage_a_source = {"campaign_sha256": "sha256:" + "9" * 64}
    stage_a_evidence = {
        "campaign": {
            "path": str(stage_a_path.resolve()),
            "file_sha256": campaign._file_sha256(stage_a_path),  # noqa: SLF001
            "campaign_sha256": stage_a_source["campaign_sha256"],
        },
        "fingerprints": {},
        "formal_result": campaign.STAGE_A_FORMAL_REFUSAL,
        "observed_p100_frontier": {},
    }
    monkeypatch.setattr(
        campaign,
        "_replay_stage_a_refusal",
        lambda evidence: (
            (
                stage_a_path.resolve(),
                stage_a_source,
                {},
            )
            if evidence == stage_a_evidence
            else (_ for _ in ()).throw(AssertionError("wrong Stage-A evidence"))
        ),
    )

    frontier = [8, 32]
    arms: dict[str, dict] = {}
    fingerprint_refs: dict[str, dict] = {}
    candidates: list[dict] = []
    receipt_by_arm: dict[str, tuple[Path, dict]] = {}
    for arm, lr, closure in (
        ("TRUST_V25", 6.0e-5, 0.02),
        ("LOWLR_V25", 3.0e-5, 0.06),
    ):
        recipe = {
            "epochs": 1,
            "max_steps": 128,
            "lr": lr,
            "lr_warmup_steps": 16,
            "policy_aux_active_batch_size": 128,
            "policy_aux_loss_weight": 0.25,
            "per_game_policy_surprise_weighting": False,
            "public_card_lr_mult": 1.0,
            "trunk_lr_mult": 1.0,
            "value_trunk_grad_scale": 0.25,
        }
        receipt_path = tmp_path / arm / "one-dose.receipt.json"
        receipt_path.parent.mkdir(parents=True)
        receipt_path.write_text("{}\n", encoding="utf-8")
        receipt = {"receipt_sha256": f"sha256:{arm.lower():0<64}"[:71]}
        receipt_by_arm[arm] = (receipt_path.resolve(), receipt)
        arms[arm] = {
            "recipe_overrides": recipe,
            "command": ["python", "trainer", "--receipt", str(receipt_path.resolve())],
        }
        rows = []
        fingerprint_dir = tmp_path / arm / "fingerprints"
        fingerprint_dir.mkdir()
        for step in frontier:
            checkpoint = (
                tmp_path
                / arm
                / ("candidate.pt" if step == 32 else f"candidate_step{step:04d}.pt")
            )
            checkpoint.write_bytes(f"{arm}-{step}".encode())
            functional = fingerprint_dir / f"step{step:04d}.functional.json"
            drift = fingerprint_dir / f"step{step:04d}.drift.json"
            parent_kl = 0.01 if step == 8 else 0.02
            teacher_gap_closure = -0.01 if step == 8 else closure
            trunk_relative_l2 = 0.004 if step == 8 else 0.008
            functional.write_text(
                json.dumps(
                    {
                        "inputs": {
                            "checkpoint": {
                                "sha256": campaign._file_sha256(checkpoint)  # noqa: SLF001
                            }
                        },
                        "functional_dose_fingerprint": {
                            "kl_parent_candidate_mean": parent_kl
                        },
                        "teacher_gap": {
                            "active_policy_teacher_gap_closure": teacher_gap_closure
                        },
                    }
                ),
                encoding="utf-8",
            )
            drift.write_text(
                json.dumps(
                    {
                        "candidate": {
                            "sha256": campaign._file_sha256(checkpoint)  # noqa: SLF001
                        },
                        "groups": {
                            "shared": {
                                "delta_energy": trunk_relative_l2**2,
                                "baseline_l2": 1.0,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            row = {
                "step": step,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": campaign._file_sha256(checkpoint),  # noqa: SLF001
                "parent_kl": parent_kl,
                "teacher_gap_closure": teacher_gap_closure,
                "trunk_relative_l2": trunk_relative_l2,
                "functional_file_sha256": campaign._file_sha256(functional),  # noqa: SLF001
                "drift_file_sha256": campaign._file_sha256(drift),  # noqa: SLF001
            }
            rows.append(row)
            candidates.append(
                {
                    "arm": arm,
                    **row,
                    "teacher_gap_closure_diagnostic_only": True,
                    "eligible": True,
                }
            )
        fingerprint_payload = {
            "schema_version": campaign.RECOVERY_FINGERPRINT_SCHEMA,
            "campaign_sha256": "placeholder",
            "arm": arm,
            "recipe_overrides": recipe,
            "one_dose_receipt_sha256": receipt["receipt_sha256"],
            "checkpoints": rows,
            "diagnostic_only": True,
            "promotion_eligible": False,
        }
        fingerprint_refs[arm] = {
            "path": str(fingerprint_dir / "fingerprint.json"),
            "payload": fingerprint_payload,
        }

    plan = {
        "schema_version": campaign.RECOVERY_CAMPAIGN_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "operator": {
            "path": str(operator.resolve()),
            "file_sha256": campaign._file_sha256(operator),  # noqa: SLF001
        },
        "source": {
            "path": str(source_root.resolve()),
            "git_sha": "a" * 40,
            "reviewed_code_tree_sha256": "sha256:" + "b" * 64,
            "one_dose_trainer_sha256": campaign._file_sha256(trainer),  # noqa: SLF001
            "train_bc_sha256": campaign._file_sha256(train_bc_path),  # noqa: SLF001
        },
        "stage_a_refusal_evidence": stage_a_evidence,
        "lineage": {
            "learner_parent_sha256": stage_a.EXPECTED_F7_PARENT_SHA256,
            "every_arm_restarts_from_exact_upgraded_f7": True,
            "fresh_adam_every_arm": True,
            "candidate_chaining_forbidden": True,
        },
        "fixed_surface": {
            "data": str(data.resolve()),
            "corpus_meta_file_sha256": campaign._file_sha256(  # noqa: SLF001
                data / "corpus_meta.json"
            ),
            "validation": str(validation.resolve()),
            "validation_file_sha256": campaign._file_sha256(validation),  # noqa: SLF001
            "admission": str(admission.resolve()),
            "admission_file_sha256": campaign._file_sha256(admission),  # noqa: SLF001
            "architecture_upgrade_receipt": str(upgrade.resolve()),
            "architecture_upgrade_receipt_file_sha256": campaign._file_sha256(  # noqa: SLF001
                upgrade
            ),
            "topology": {
                "world_size": 8,
                "local_batch_size": 512,
                "global_batch_size": 4096,
            },
        },
        "trajectory": {"checkpoint_steps": frontier, "terminal_step": 32},
        "arms": arms,
        "selection_contract": {
            "parent_kl_max": 0.03,
            "trunk_relative_l2_max": 0.03,
            "teacher_gap_closure_ranking_authority": False,
            "teacher_gap_closure_admission_authority": False,
            "paired_playing_strength_is_final_authority": True,
            "objective": "minimum_update_within_frozen_trust_and_trunk_budgets",
            "tie_break": [
                "min_parent_kl",
                "min_trunk_relative_l2",
                "min_optimizer_step",
            ],
            "playing_strength_evaluation_required": True,
        },
        "outputs": {"selection": str((tmp_path / "recovery.selection.json").resolve())},
    }
    plan_path = _signed(tmp_path / "recovery.plan.json", plan, "campaign_sha256")
    sealed_plan = json.loads(plan_path.read_text())
    for arm, ref in fingerprint_refs.items():
        ref["payload"]["campaign_sha256"] = sealed_plan["campaign_sha256"]
        fingerprint_path = _signed(
            Path(ref["path"]), ref["payload"], "fingerprint_sha256"
        )
        fingerprint = json.loads(fingerprint_path.read_text())
        fingerprint_refs[arm] = {
            "path": str(fingerprint_path),
            "file_sha256": campaign._file_sha256(fingerprint_path),  # noqa: SLF001
            "fingerprint_sha256": fingerprint["fingerprint_sha256"],
        }
    monkeypatch.setattr(
        campaign,
        "_recovery_arm_receipt",
        lambda _plan, arm, _fingerprint: receipt_by_arm[arm],
    )
    winner = next(
        row for row in candidates if row["arm"] == "LOWLR_V25" and row["step"] == 8
    )
    selection = {
        "schema_version": campaign.RECOVERY_SELECTION_SCHEMA,
        "campaign": {
            "path": str(plan_path),
            "file_sha256": campaign._file_sha256(plan_path),  # noqa: SLF001
            "campaign_sha256": sealed_plan["campaign_sha256"],
        },
        "source": sealed_plan["source"],
        "lineage": sealed_plan["lineage"],
        "stage_a_refusal_evidence": sealed_plan["stage_a_refusal_evidence"],
        "selection_contract": sealed_plan["selection_contract"],
        "fingerprints": fingerprint_refs,
        "checkpoint_candidates": candidates,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "playing_strength_evaluation_required": True,
        "winner": winner,
    }
    selection_path = _signed(
        tmp_path / "recovery.selection.json", selection, "selection_sha256"
    )
    return selection_path, selection


def test_recovery_selection_binds_exact_winner_fingerprint_and_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path, _selection = _recovery_selection_fixture(tmp_path, monkeypatch)

    _path, _payload, _stage_a_path, _stage_a, dose = campaign._load_recovery_selection(
        selection_path
    )  # noqa: SLF001

    assert dose["authority_kind"] == "direction_corrected_recovery_selection"
    assert dose["selected_arm"] == "LOWLR_V25"
    assert dose["optimizer_steps"] == 8
    assert dose["checkpoint_steps"] == [8]
    assert dose["policy_aux_active_batch_size"] == 128
    assert dose["active_policy_branch_multiplier"] == pytest.approx(0.25)
    assert dose["reference_parent_kl"] == pytest.approx(0.01)
    assert dose["reference_trunk_relative_l2"] == pytest.approx(0.004)
    assert dose["reference_teacher_gap_closure"] == pytest.approx(-0.01)
    assert dose["selected_recipe_overrides"]["lr"] == pytest.approx(3.0e-5)
    assert dose["trust_contract"]["policy_kl_target"] == pytest.approx(0.01)
    assert dose["recovery_receipt"]["receipt_sha256"].startswith("sha256:")

    public_dose = campaign.load_recovery_selected_dose(selection_path)
    assert public_dose == dose


def test_recovery_selection_refuses_a_resigned_nonwinning_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path, _selection = _recovery_selection_fixture(tmp_path, monkeypatch)
    payload = json.loads(selection_path.read_text())
    payload.pop("selection_sha256")
    payload["winner"] = next(
        row
        for row in payload["checkpoint_candidates"]
        if row["arm"] == "TRUST_V25" and row["step"] == 32
    )
    payload["selection_sha256"] = campaign._value_sha256(payload)  # noqa: SLF001
    selection_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(campaign.CampaignError, match="winner does not replay"):
        campaign._load_recovery_selection(selection_path)  # noqa: SLF001
