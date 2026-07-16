from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pytest

from tools import a1_b200_active_policy_campaign as campaign
from tools import a1_current_science_contract as current_science
from tools import a1_one_dose_train as one_dose
from tools import train_bc
from tools.fleet import a1_coherent_target_rd_executor as coherent_executor


def _write_signed(path: Path, value: dict, field: str) -> None:
    value[field] = campaign._value_sha256(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _checkpoint_dose_stub(step: int) -> dict:
    return {
        "schema_version": train_bc.CHECKPOINT_DOSE_TELEMETRY_SCHEMA,
        "optimizer_step": step,
    }


def test_completion_fallback_replays_native_runtime_from_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text("{}\n", encoding="utf-8")
    contract_sha256 = "sha256:" + "1" * 64
    contract = {"contract_sha256": contract_sha256}
    native_runtime = {
        "schema_version": coherent_executor.NATIVE_RUNTIME_IDENTITY_SCHEMA,
        "wheel_build_receipt": {"source_commit": "a" * 40},
        "distribution": {"wheel_sha256": "sha256:" + "b" * 64},
        "extension": {
            "sha256": "sha256:" + "c" * 64,
            "wheel_member_sha256": "sha256:" + "c" * 64,
        },
        "capabilities": ["coherent_public_belief_search"],
    }
    native_runtime["identity_sha256"] = coherent_executor._digest(native_runtime)

    launch_path = tmp_path / "launch.receipt.json"
    launch = {
        "schema_version": coherent_executor.LAUNCH_RECEIPT_SCHEMA,
        "status": "launched",
        "contract": {
            "path": str(contract_path),
            "file_sha256": campaign._file_sha256(contract_path),
            "contract_sha256": contract_sha256,
        },
        "preflight": {
            "native_runtime": native_runtime,
            "checkpoint_sha256": campaign.EXPECTED_CORPUS_PRODUCER_SHA256,
        },
        "commands": [],
    }
    _write_signed(launch_path, launch, "receipt_sha256")

    completion_path = tmp_path / "completion.receipt.json"
    completion = {
        "schema_version": campaign.COMPLETION_RECEIPT_SCHEMA,
        "launch_receipt": {
            "path": str(launch_path),
            "file_sha256": campaign._file_sha256(launch_path),
        },
    }
    _write_signed(completion_path, completion, "receipt_sha256")

    observed: dict[str, object] = {}

    def replay_completion(
        path: Path,
        *,
        contract: dict,
        launch_file_sha256: str,
        native_runtime: dict,
    ) -> dict:
        observed.update(
            path=path,
            contract=contract,
            launch_file_sha256=launch_file_sha256,
            native_runtime=native_runtime,
        )
        raise coherent_executor.ExecutorError("replay reached")

    monkeypatch.setattr(
        coherent_executor, "_verify_existing_completion", replay_completion
    )

    with pytest.raises(campaign.CampaignError, match="replay reached"):
        campaign._verify_completion_receipt(
            completion_path,
            contract_path=contract_path.resolve(),
            contract=contract,
        )

    assert observed["native_runtime"] == native_runtime
    assert observed["launch_file_sha256"] == campaign._file_sha256(launch_path)


def test_active_policy_arms_change_only_auxiliary_exposure() -> None:
    science = {
        "public_card_lr_mult": 4.0,
        "per_game_policy_surprise_weighting": True,
        "forced_row_value_action_type_weights": "END_TURN=0.1,ROLL=0.25",
    }
    assert {
        arm: values["policy_aux_active_batch_size"]
        for arm, values in campaign.ARMS.items()
    } == {"P10": 128, "P25": 128, "P50": 128, "P100": 128}
    assert {
        arm: values["policy_aux_loss_weight"]
        for arm, values in campaign.ARMS.items()
    } == {"P10": 0.10, "P25": 0.25, "P50": 0.50, "P100": 1.00}
    recipes = {
        arm: campaign._arm_overrides(arm, science) for arm in campaign.ARMS
    }
    common = {
        key: value
        for key, value in recipes["P10"].items()
        if key != "policy_aux_loss_weight"
    }
    assert all(
        {
            key: value
            for key, value in recipe.items()
            if key != "policy_aux_loss_weight"
        }
        == common
        for recipe in recipes.values()
    )
    assert common["max_steps"] == 128
    assert common["lr"] == 6e-5
    assert common["lr_warmup_steps"] == 16


def test_corpus_admission_requires_complete_forced_value_row_coverage() -> None:
    report = {
        "present": True,
        "missing_columns": [],
        "rows": 1_000_000,
        "forced_rows": 400_000,
        "forced_fraction": 0.4,
        "game_count": campaign.EXPECTED_GAMES,
        "games_with_forced_rows": campaign.EXPECTED_GAMES,
        "forced_game_coverage": 1.0,
        "forced_policy_active_rows": 0,
        "forced_value_inactive_rows": 0,
        "phase_counts": {"PLAY_TURN": 600_000, "ROLL": 300_000, "END_TURN": 100_000},
        "forced_phase_counts": {"ROLL": 300_000, "END_TURN": 100_000},
        "action_taken_counts": {"0": 300_000, "42": 100_000, "9": 600_000},
        "forced_action_taken_counts": {"0": 300_000, "42": 100_000},
        "action_type_counts": {
            "BUILD_CITY": 600_000,
            "ROLL": 300_000,
            "END_TURN": 100_000,
        },
        "forced_action_type_counts": {"ROLL": 300_000, "END_TURN": 100_000},
        "contract_passed": True,
    }

    assert campaign._require_forced_value_rows({"forced_value_rows": report}) == report


def test_corpus_admission_refuses_zero_forced_rows() -> None:
    report = {
        "present": True,
        "missing_columns": [],
        "rows": 600_000,
        "forced_rows": 0,
        "forced_fraction": 0.0,
        "game_count": campaign.EXPECTED_GAMES,
        "games_with_forced_rows": 0,
        "forced_game_coverage": 0.0,
        "forced_policy_active_rows": 0,
        "forced_value_inactive_rows": 0,
        "phase_counts": {"PLAY_TURN": 600_000},
        "forced_phase_counts": {},
        "action_taken_counts": {"9": 600_000},
        "forced_action_taken_counts": {},
        "action_type_counts": {"BUILD_CITY": 600_000},
        "forced_action_type_counts": {},
        "contract_passed": False,
    }

    with pytest.raises(campaign.CampaignError, match="sole-action value signal"):
        campaign._require_forced_value_rows({"forced_value_rows": report})


def test_forced_value_contract_is_new_v2_only() -> None:
    legacy = {
        "schema_version": "a1-coherent-target-rd-contract-v1",
        "operator": {"record_automatic_transitions": False},
    }
    current = {
        "schema_version": campaign.FORCED_VALUE_TARGET_CONTRACT_SCHEMA,
        "operator": {"record_automatic_transitions": True},
    }
    assert campaign._requires_forced_value_rows(
        contract=legacy, repaired_distillation=False
    ) is False
    assert campaign._requires_forced_value_rows(
        contract=current, repaired_distillation=True
    ) is False
    assert campaign._requires_forced_value_rows(
        contract=current, repaired_distillation=False
    ) is True


def test_target_contract_verifier_accepts_authenticated_v1_and_v2() -> None:
    repo = Path(campaign.__file__).resolve().parents[1]
    expected = {
        "a1-target-identity-coherent-n128-rd-v1": (
            campaign.EXPECTED_TARGET_CONTRACT_SHA256,
            False,
        ),
        "a1-target-identity-coherent-n128-rd-v2": (
            campaign.EXPECTED_TARGET_CONTRACT_V2_SHA256,
            True,
        ),
    }
    for operation, (digest, automatic) in expected.items():
        path = repo / "configs/operations" / operation / "contract.json"
        resolved, contract = campaign._verify_target_contract(path)
        assert resolved == path.resolve()
        assert contract["contract_sha256"] == digest
        assert contract["operator"]["record_automatic_transitions"] is automatic


def test_v2_admission_rejects_zero_forced_rows_after_real_contract_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = Path(campaign.__file__).resolve().parents[1]
    contract_path = (
        repo
        / "configs/operations/a1-target-identity-coherent-n128-rd-v2/contract.json"
    )
    corpus_meta = tmp_path / "corpus_meta.json"
    corpus_meta.write_text("{}\n")
    validation = tmp_path / "validation.json"
    validation.write_text("{}\n")
    completion = {
        "schema_version": campaign.COMPLETION_RECEIPT_SCHEMA,
    }
    zero_forced = {
        "present": True,
        "missing_columns": [],
        "rows": 600_000,
        "forced_rows": 0,
        "forced_fraction": 0.0,
        "game_count": campaign.EXPECTED_GAMES,
        "games_with_forced_rows": 0,
        "forced_game_coverage": 0.0,
        "forced_policy_active_rows": 0,
        "forced_value_inactive_rows": 0,
        "phase_counts": {"PLAY_TURN": 600_000},
        "forced_phase_counts": {},
        "action_taken_counts": {"9": 600_000},
        "forced_action_taken_counts": {},
        "action_type_counts": {"BUILD_CITY": 600_000},
        "forced_action_type_counts": {},
        "contract_passed": False,
    }
    inventory_payload = {
        "direct_corpora": [{"forced_value_rows": zero_forced}],
        "aggregate": {},
        "rd_contract": {
            "contract_sha256": campaign.EXPECTED_TARGET_CONTRACT_V2_SHA256
        },
    }
    monkeypatch.setattr(
        campaign,
        "_verify_completion_receipt",
        lambda *_args, **_kwargs: (tmp_path / "completion.json", completion),
    )
    monkeypatch.setattr(
        campaign,
        "_load_signed",
        lambda *_args, **_kwargs: (tmp_path / "inventory.json", inventory_payload),
    )

    with pytest.raises(campaign.CampaignError, match="sole-action value signal"):
        campaign._admit_corpus(
            Namespace(
                contract=contract_path,
                completion_receipt=tmp_path / "completion.json",
                inventory=tmp_path / "inventory.json",
                corpus_meta=corpus_meta,
                validation_manifest=validation,
            )
        )


def test_legacy_admission_without_forced_rows_remains_loadable(
    tmp_path: Path,
) -> None:
    data = tmp_path / "corpus"
    data.mkdir()
    meta = data / "corpus_meta.json"
    meta.write_text("{}\n")
    contract_path = tmp_path / "contract.json"
    completion_path = tmp_path / "completion.json"
    inventory_path = tmp_path / "inventory.json"
    validation_path = tmp_path / "validation.json"
    for path in (contract_path, completion_path, inventory_path, validation_path):
        path.write_text("{}\n")
    corpus = {
        "data_path": str(data),
        "corpus_meta_path": str(meta),
        "corpus_meta_file_sha256": campaign._file_sha256(meta),
        "payload_inventory_sha256": "sha256:" + "1" * 64,
        "validation_manifest": {
            "path": str(validation_path),
            "file_sha256": campaign._file_sha256(validation_path),
        },
        "producer_checkpoint_sha256": campaign.EXPECTED_CORPUS_PRODUCER_SHA256,
        "target_information_regime": campaign.TARGET_INFORMATION_REGIME,
        "search_evidence_schema": campaign.SEARCH_EVIDENCE_SCHEMA,
        "selected_games": campaign.EXPECTED_GAMES,
        "selected_game_seed_set_sha256": "sha256:" + "2" * 64,
        "selection_mode": "explicit_truncation_repair_seed_set",
        "complete_two_seat_trace_games": campaign.EXPECTED_GAMES,
        "stored_policy_target_distillation_eligible": True,
        "state_reanalysis_eligible": False,
        "search_evidence_storage": "receipt_bound_source_npz_only",
        "incompatible_policy_active_rows": 0,
    }
    admission = {
        "schema_version": campaign.ADMISSION_SCHEMA,
        "status": "admitted_for_diagnostic_policy_distillation",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "contract": {
            "path": str(contract_path),
            "file_sha256": campaign._file_sha256(contract_path),
            "contract_sha256": campaign.EXPECTED_TARGET_CONTRACT_SHA256,
        },
        "completion_receipt": {
            "path": str(completion_path),
            "file_sha256": campaign._file_sha256(completion_path),
        },
        "target_eligibility_inventory": {
            "path": str(inventory_path),
            "file_sha256": campaign._file_sha256(inventory_path),
        },
        "corpus": corpus,
        "policy_distillation_contract": {
            "coherent_public_n128_only": True,
            "legacy_pimc_rows_allowed": False,
        },
    }
    admission_path = tmp_path / "admission.json"
    _write_signed(admission_path, admission, "admission_sha256")

    resolved, loaded = campaign._load_admission(admission_path)

    assert resolved == admission_path.resolve()
    assert loaded["corpus"] == corpus
    assert "value_distillation_contract" not in loaded
    assert "forced_value_rows" not in loaded["corpus"]


def test_independent_parent_authority_keeps_producer_and_f7_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "corpus"
    data.mkdir()
    meta = data / "corpus_meta.json"
    meta.write_text("{}\n")
    admission = tmp_path / "admission.json"
    admission.write_text("{}\n")
    upgrade_receipt = tmp_path / "upgrade.json"
    upgrade_receipt.write_text("{}\n")
    initializer = tmp_path / "initializer.pt"
    initializer.write_bytes(b"function-preserving-f7")
    producer = {
        "path": "/models/v5.pt",
        "sha256": campaign.EXPECTED_CORPUS_PRODUCER_SHA256,
    }
    parent = {
        "path": "/models/f7.pt",
        "sha256": campaign.EXPECTED_F7_PARENT_SHA256,
    }
    verified = {
        "data_path": data.resolve(),
        "corpus_meta_file_sha256": campaign._file_sha256(meta),
        "payload_inventory_sha256": "sha256:" + "1" * 64,
        "data_fingerprint": "sha256:" + "2" * 64,
        "producer": producer,
        "contract_sha256": "sha256:" + "5" * 64,
    }
    coherent_binding = {
        "schema_version": one_dose.train_bc.COHERENT_DIRECT_CORPUS_BINDING_SCHEMA,
        "learner_initializer": None,
    }
    coherent_binding["binding_sha256"] = campaign._value_sha256(
        coherent_binding
    )
    verified["coherent_direct_corpus_binding"] = coherent_binding
    upgrade = {
        "module": one_dose.architecture_upgrade.MODULE_STRUCTURED_ACTION_VALUE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_V3,
        "source": parent,
        "receipt": {
            "path": str(upgrade_receipt),
            "sha256": campaign._file_sha256(upgrade_receipt),
        },
        "receipt_sha256": "sha256:" + "3" * 64,
        "upgraded_initializer": {
            "path": str(initializer),
            "sha256": campaign._file_sha256(initializer),
        },
    }
    authority = campaign._parent_authority(
        verified=verified,
        upgrade=upgrade,
        admission_path=admission,
        admission={"admission_sha256": "sha256:" + "4" * 64},
    )
    authority_path = tmp_path / "parent.authority.json"
    authority_path.write_text(json.dumps(authority))
    replayed = one_dose._verify_independent_parent_authority(
        authority_path,
        verified=verified,
        upgrade=upgrade,
    )
    assert replayed["corpus_binding"]["producer_checkpoint"] == producer
    assert replayed["learner_parent"] == parent
    assert producer["sha256"] != parent["sha256"]

    monkeypatch.setattr(
        one_dose.architecture_upgrade,
        "verify_receipt",
        lambda _path: upgrade,
    )
    bound = one_dose.bind_function_preserving_upgrade(
        verified,
        upgrade_receipt,
        independent_parent_authority_path=authority_path,
    )
    assert bound["producer"] == producer
    assert bound["learner_lineage_parent"]["checkpoint"] == parent
    assert one_dose._learner_lineage_parent_sha256(bound) == parent["sha256"]
    rendered_initializer = bound["coherent_direct_corpus_binding"][
        "learner_initializer"
    ]
    assert rendered_initializer["parent_checkpoint_sha256"] == parent["sha256"]
    assert rendered_initializer["initializer_checkpoint_sha256"] == upgrade[
        "upgraded_initializer"
    ]["sha256"]


def test_selection_uses_closure_only_for_admission_and_nominates_smallest_update(
    tmp_path: Path,
) -> None:
    campaign_path = tmp_path / "campaign.json"
    campaign_payload = {
        "schema_version": campaign.SCHEMA,
        "selection_contract": {
            "max_parent_kl": 0.03,
            "max_trunk_relative_l2": 0.03,
            "reference_update_frontier": dict(
                campaign.R2_UPDATE_FRONTIER_REFERENCE
            ),
        },
        "lineage_contract": {
            "upgraded_initializer_sha256": "sha256:" + "6" * 64,
        },
        "arms": {arm: dict(values) for arm, values in campaign.ARMS.items()},
    }
    _write_signed(campaign_path, campaign_payload, "campaign_sha256")

    closures = {"P10": 0.03, "P25": 0.06, "P50": 0.09, "P100": 0.12}
    kls = {"P10": 0.01, "P25": 0.02, "P50": 0.025, "P100": 0.04}
    bindings: dict[str, Path] = {}
    for arm in campaign.ARMS:
        path = tmp_path / f"{arm}.json"
        rows = []
        for step in campaign.CHECKPOINT_STEPS:
            checkpoint = tmp_path / f"{arm}-step{step}.pt"
            checkpoint.write_bytes(f"{arm}:{step}".encode())
            rows.append(
                {
                    "step": step,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": campaign._file_sha256(checkpoint),
                    "functional": {
                        "parent_kl": kls[arm] * step / 128,
                        "teacher_gap_closure": closures[arm] * step / 128,
                    },
                    "layer_drift": {
                        "trunk_relative_l2": min(kls[arm], 0.029) * step / 128,
                    },
                    "dose_telemetry": _checkpoint_dose_stub(step),
                }
            )
        dose_telemetry = {
            "schema_version": campaign.DOSE_TELEMETRY_SCHEMA,
            "active_rows": {"policy_aux": 1},
            "checkpoint_trajectory": [
                row["dose_telemetry"] for row in rows
            ],
        }
        dose_telemetry["dose_telemetry_sha256"] = campaign._value_sha256(
            dose_telemetry
        )
        payload = {
            "schema_version": campaign.FINGERPRINT_SCHEMA,
            "campaign": {
                "path": str(campaign_path),
                "file_sha256": campaign._file_sha256(campaign_path),
                "campaign_sha256": campaign_payload["campaign_sha256"],
            },
            "arm": arm,
            "active_policy_branch_multiplier": campaign.ARMS[arm][
                "active_policy_branch_multiplier"
            ],
            "policy_aux_active_batch_size": campaign.ARMS[arm][
                "policy_aux_active_batch_size"
            ],
            "policy_aux_loss_weight": campaign.ARMS[arm][
                "policy_aux_loss_weight"
            ],
            "parent_checkpoint_sha256": campaign_payload["lineage_contract"][
                "upgraded_initializer_sha256"
            ],
            "dose_telemetry": dose_telemetry,
            "checkpoints": rows,
        }
        _write_signed(path, payload, "fingerprint_sha256")
        bindings[arm] = path

    selected = campaign._select(campaign_path, campaign_payload, bindings)
    assert selected["winner"] == "P10"
    assert selected["winner_step"] == 8
    assert selected["eligible_arms"] == ["P10", "P25", "P50", "P100"]
    assert selected["arm_fingerprints"]["P100"][
        "all_checkpoints_within_drift_budgets"
    ] is False
    assert selected["arm_fingerprints"]["P100"]["eligible_checkpoint_steps"] == [
        8,
        12,
        16,
        32,
        64,
    ]
    assert selected["arm_fingerprints"]["P100"]["selected_checkpoint"][
        "step"
    ] == 8
    assert selected["winner_meets_reference_teacher_gap_closure"] is False


def test_selection_does_not_reward_a_later_teacher_closure_spike(
    tmp_path: Path,
) -> None:
    campaign_path = tmp_path / "campaign.json"
    campaign_payload = {
        "schema_version": campaign.SCHEMA,
        "selection_contract": {
            "max_parent_kl": 0.03,
            "max_trunk_relative_l2": 0.03,
            "reference_update_frontier": dict(
                campaign.R2_UPDATE_FRONTIER_REFERENCE
            ),
        },
        "lineage_contract": {
            "upgraded_initializer_sha256": "sha256:" + "6" * 64,
        },
        "arms": {arm: dict(values) for arm, values in campaign.ARMS.items()},
    }
    _write_signed(campaign_path, campaign_payload, "campaign_sha256")
    bindings: dict[str, Path] = {}
    for arm_index, arm in enumerate(campaign.ARMS):
        path = tmp_path / f"{arm}.json"
        rows = []
        for step in campaign.CHECKPOINT_STEPS:
            checkpoint = tmp_path / f"{arm}-step{step}.pt"
            checkpoint.write_bytes(f"{arm}:{step}".encode())
            peak = 0.20 if arm == "P100" and step == 32 else 0.01 + arm_index * 0.001
            drift = 0.02 if step <= 32 else 0.04
            rows.append(
                {
                    "step": step,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": campaign._file_sha256(checkpoint),
                    "functional": {
                        "parent_kl": drift,
                        "teacher_gap_closure": peak,
                    },
                    "layer_drift": {"trunk_relative_l2": drift},
                    "dose_telemetry": _checkpoint_dose_stub(step),
                }
            )
        dose_telemetry = {
            "schema_version": campaign.DOSE_TELEMETRY_SCHEMA,
            "active_rows": {"policy_aux": 1},
            "checkpoint_trajectory": [
                row["dose_telemetry"] for row in rows
            ],
        }
        dose_telemetry["dose_telemetry_sha256"] = campaign._value_sha256(
            dose_telemetry
        )
        payload = {
            "schema_version": campaign.FINGERPRINT_SCHEMA,
            "campaign": {
                "path": str(campaign_path),
                "file_sha256": campaign._file_sha256(campaign_path),
                "campaign_sha256": campaign_payload["campaign_sha256"],
            },
            "arm": arm,
            "active_policy_branch_multiplier": campaign.ARMS[arm][
                "active_policy_branch_multiplier"
            ],
            "policy_aux_active_batch_size": campaign.ARMS[arm][
                "policy_aux_active_batch_size"
            ],
            "policy_aux_loss_weight": campaign.ARMS[arm][
                "policy_aux_loss_weight"
            ],
            "parent_checkpoint_sha256": campaign_payload["lineage_contract"][
                "upgraded_initializer_sha256"
            ],
            "dose_telemetry": dose_telemetry,
            "checkpoints": rows,
        }
        _write_signed(path, payload, "fingerprint_sha256")
        bindings[arm] = path

    selected = campaign._select(campaign_path, campaign_payload, bindings)

    assert selected["winner"] == "P10"
    assert selected["winner_step"] == 8
    assert selected["winner_checkpoint"]["path"].endswith("P10-step8.pt")
    assert selected["arm_fingerprints"]["P100"]["selected_checkpoint"][
        "step"
    ] == 8


def test_explicit_diagnostic_checkpoint_schedule_excludes_terminal(
    tmp_path: Path,
) -> None:
    repo = Path(one_dose.__file__).resolve().parents[1]
    code_binding = one_dose._current_ablation_code_binding(  # noqa: SLF001
        {
            "provenance": {
                "learner_code": [{"path": str(repo / "tools/train_bc.py")}],
                "runtime_code_tree": [
                    {"path": str(repo / "tools/a1_one_dose_train.py")}
                ],
            }
        }
    )
    assert one_dose.train_bc._parse_checkpoint_steps(
        "8,12,16,32,64", max_steps=128
    ) == (8, 12, 16, 32, 64)
    with pytest.raises(SystemExit):
        one_dose.train_bc._parse_checkpoint_steps(
            "8,12,16,32,64,128", max_steps=128
        )
    verified = {
        "recipe": current_science.learner_training_recipe(),
        "producer": {
            "path": str(tmp_path / "f7.pt"),
            "sha256": campaign.EXPECTED_F7_PARENT_SHA256,
        },
        "architecture_initializer": {
            "path": str(tmp_path / "upgraded-f7.pt"),
            "sha256": "sha256:" + "7" * 64,
        },
        "function_preserving_upgrade": {
            "module": one_dose.architecture_upgrade.MODULE_STRUCTURED_ACTION_VALUE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_V3,
        },
        "data_path": tmp_path / "corpus",
        "validation_path": tmp_path / "validation.json",
        "payload_inventory_sha256": "sha256:" + "8" * 64,
        "learner_ablation": {
            "ablation_id": "coherent-checkpoint-schedule",
            "reporting_contract": {
                "diagnostic_dose_curve": True,
                "checkpoint_steps": [8, 12, 16, 32, 64],
            },
            "code_binding": code_binding,
            "code_tree_sha256": code_binding["code_tree_sha256"],
            "reviewed_lock_file_sha256": "sha256:" + "a" * 64,
        },
    }
    command = one_dose._build_direct_train_command(
        verified,
        python=Path("/usr/bin/python3"),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )
    assert one_dose._literal_option_values(command, "--checkpoint-steps") == [
        "8,12,16,32,64"
    ]
    assert one_dose._literal_option_values(
        command, "--train-diagnostics-every-batches"
    ) == ["16"]
    assert one_dose._literal_option_values(
        command, "--objective-gradient-interference-every-batches"
    ) == ["64"]


def test_arm_dose_telemetry_seals_exposure_grad_clip_and_update_rms() -> None:
    expected_aux = (
        campaign.POLICY_AUX_ACTIVE_BATCH_SIZE
        * campaign.WORLD_SIZE
        * campaign.MAX_STEPS
    )
    report = {
        "policy_base_active_rows": 10_000,
        "policy_aux_active_rows": expected_aux,
        "policy_total_active_rows": 10_000 + expected_aux,
        "policy_base_effective_weight_sum": 20_000.0,
        "policy_aux_effective_weight_sum": 30_000.0,
        "policy_total_effective_weight_sum": 50_000.0,
        "value_active_rows": 500_000,
        "policy_kl_anchor_eligible_rows": 0,
        "metrics": [
            {
                "loss_denominators": {
                    "policy_loss": 50_000.0,
                    "value_loss": 400_000.0,
                    "final_vp_loss": 0.0,
                },
                "optimizer_observability": {
                    "observed_steps": 128,
                    "clipped_steps": 4,
                    "clipped_fraction": 4 / 128,
                    "mean_pre_clip_total_grad_norm": 0.8,
                    "max_pre_clip_total_grad_norm": 1.2,
                },
            }
        ],
        "module_optimizer_observability": {
            "observed_steps": 8,
            "cadence_batches": 16,
            "norm_scope": "global_replicated",
            "modules": {
                "blocks": {
                    "mean_pre_clip_grad_norm": 0.7,
                    "max_pre_clip_grad_norm": 1.1,
                    "mean_parameter_delta_norm": 0.01,
                    "mean_parameter_update_rms": 1.0e-6,
                    "mean_relative_parameter_delta": 2.0e-5,
                    "parameter_count": 1_000_000,
                }
            },
        },
        "objective_gradient_interference": {
            "cadence_batches": 64,
            "observed_steps": 2,
            "observations": [
                {
                    "available": True,
                    "optimizer_step": 64,
                    "scope": "rank_local_microbatch",
                    "policy_trunk_grad_norm": 0.7,
                    "policy_base_trunk_grad_norm": 0.5,
                    "policy_aux_trunk_grad_norm": 0.2,
                    "value_trunk_grad_norm": 0.3,
                    "policy_aux_to_base_grad_norm_ratio": 0.4,
                    "trunk_gradient_cosine": -0.1,
                    "policy_base_aux_gradient_cosine": 0.2,
                    "objective_trunk_grad_l2": {
                        "policy": 0.7,
                        "policy_base": 0.5,
                        "active_policy": 0.2,
                        "value": 0.3,
                    },
                },
                {
                    "available": True,
                    "optimizer_step": 128,
                    "scope": "rank_local_microbatch",
                    "policy_trunk_grad_norm": 0.8,
                    "policy_base_trunk_grad_norm": 0.5,
                    "policy_aux_trunk_grad_norm": 0.3,
                    "value_trunk_grad_norm": 0.25,
                    "policy_aux_to_base_grad_norm_ratio": 0.6,
                    "trunk_gradient_cosine": -0.2,
                    "policy_base_aux_gradient_cosine": 0.1,
                    "objective_trunk_grad_l2": {
                        "policy": 0.8,
                        "policy_base": 0.5,
                        "active_policy": 0.3,
                        "value": 0.25,
                    },
                },
            ],
        },
    }
    gradient_rows = report["objective_gradient_interference"]["observations"]
    checkpoint_doses = []
    for step in campaign.CHECKPOINT_STEPS:
        fraction = step / campaign.MAX_STEPS
        base_rows = int(10_000 * fraction)
        aux_rows = int(expected_aux * fraction)
        clipped = step // 32
        checkpoint_doses.append(
            {
                "schema_version": train_bc.CHECKPOINT_DOSE_TELEMETRY_SCHEMA,
                "optimizer_step": step,
                "training_row_draws": {},
                "active_rows": {
                    "policy_base": base_rows,
                    "policy_aux": aux_rows,
                    "policy_total": base_rows + aux_rows,
                    "value": int(500_000 * fraction),
                    "policy_kl_anchor": 0,
                },
                "policy_effective_weight_sums": {
                    "base": 20_000.0 * fraction,
                    "aux": 30_000.0 * fraction,
                    "total": 50_000.0 * fraction,
                },
                "objective_effective_weight_sums": {
                    "policy_loss": 50_000.0 * fraction,
                    "policy_base_loss": 20_000.0 * fraction,
                    "active_policy_loss": 30_000.0 * fraction,
                    "value_loss": 400_000.0 * fraction,
                    "final_vp_loss": 0.0,
                },
                "optimizer": {
                    "observed_steps": step,
                    "clipped_steps": clipped,
                    "clipped_fraction": clipped / step,
                },
                "shared_trunk_objective_gradients": {
                    "observations": [
                        row for row in gradient_rows if row["optimizer_step"] <= step
                    ]
                },
                "module_optimizer_observability": None,
                "feature_path_gradients": {
                    "public_card": {"status": "observed"},
                    "meaningful_history": {"status": "observed"},
                },
            }
        )
    report["checkpoint_dose_trajectory"] = {
        "schema_version": train_bc.CHECKPOINT_DOSE_TRAJECTORY_SCHEMA,
        "checkpoint_steps": list(campaign.CHECKPOINT_STEPS),
        "checkpoints": checkpoint_doses,
    }

    telemetry = campaign._arm_dose_telemetry(  # noqa: SLF001
        report, expected_aux_rows=expected_aux
    )

    assert telemetry["active_rows"]["policy_aux"] == expected_aux
    assert telemetry["policy_effective_weight_sums"]["aux"] == 30_000.0
    assert telemetry["objective_effective_weight_sums"]["value_loss"] == 400_000.0
    assert telemetry["optimizer"]["clipped_fraction"] == pytest.approx(4 / 128)
    assert telemetry["module_optimizer_observability"]["modules"]["blocks"][
        "mean_parameter_update_rms"
    ] == pytest.approx(1.0e-6)
    assert telemetry["shared_trunk_objective_gradients"]["observed_steps"] == 2
    assert telemetry["dose_telemetry_sha256"].startswith("sha256:")


def test_coherent_binding_authorizes_exact_independent_initializer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    meta = {"payload_inventory_sha256": "sha256:" + "1" * 64}
    meta_path = corpus / "corpus_meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    validation_path = tmp_path / "validation.json"
    validation_path.write_text("{}\n", encoding="utf-8")
    seeds = np.asarray([100, 100, 101, 101], dtype=np.int64)
    validation_seeds = np.asarray([101], dtype=np.int64)
    target_contract_sha = "sha256:" + "2" * 64
    parent_sha = "sha256:" + "3" * 64
    initializer_sha = "sha256:" + "4" * 64
    validation = {
        "path": str(validation_path.resolve()),
        "file_sha256": campaign._file_sha256(validation_path),
        "a1_contract_sha256": target_contract_sha,
        "validation_game_seed_count": 1,
        "validation_game_seed_set_sha256": train_bc._game_seed_set_sha256(
            validation_seeds
        ),
        "validation_row_count": 2,
        "game_seeds": validation_seeds,
    }
    recipe = current_science.learner_training_recipe()
    binding = {
        "schema_version": train_bc.COHERENT_DIRECT_CORPUS_BINDING_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "corpus_admission": {},
        "target_contract_sha256": target_contract_sha,
        "producer_checkpoint_sha256": campaign.EXPECTED_CORPUS_PRODUCER_SHA256,
        "learner_initializer": {
            "role": "diagnostic_independent_parent",
            "parent_checkpoint_sha256": parent_sha,
            "initializer_checkpoint_sha256": initializer_sha,
            "upgrade_module": train_bc.COHERENT_DIRECT_UPGRADE_MODULE,
            "upgrade_receipt_file_sha256": "sha256:" + "5" * 64,
            "upgrade_receipt_sha256": "sha256:" + "6" * 64,
            "independent_parent_authority_sha256": "sha256:" + "7" * 64,
        },
        "corpus": {
            "path": str(corpus.resolve()),
            "corpus_meta_file_sha256": campaign._file_sha256(meta_path),
            "payload_inventory_sha256": meta["payload_inventory_sha256"],
            "selected_game_count": 2,
            "seed_start": 100,
            "seed_end": 102,
            "selected_game_seed_set_sha256": train_bc._game_seed_set_sha256(
                np.asarray([100, 101], dtype=np.int64)
            ),
            "training_game_count": 1,
            "training_game_seed_set_sha256": train_bc._game_seed_set_sha256(
                np.asarray([100], dtype=np.int64)
            ),
        },
        "validation": {
            "path": validation["path"],
            "file_sha256": validation["file_sha256"],
            "game_count": 1,
            "game_seed_set_sha256": validation[
                "validation_game_seed_set_sha256"
            ],
            "row_count": 2,
        },
        "learner": {
            "training_recipe": recipe,
            "training_recipe_sha256": campaign._value_sha256(recipe),
            "value_objective": {
                "objective": "mse",
                "value_readout": "scalar",
                "value_categorical_bins": None,
                "hlgauss_sigma_ratio": None,
            },
            "topology": {
                "name": "b200-8gpu-ddp",
                "world_size": 8,
                "local_batch_size": 512,
                "grad_accum_steps": 1,
                "global_batch_size": 4096,
            },
        },
    }
    binding["binding_sha256"] = campaign._value_sha256(binding)
    monkeypatch.setattr(
        train_bc, "_validate_memmap_payload_inventory", lambda *_args: None
    )
    result = train_bc._validate_coherent_direct_corpus_binding(
        json.dumps(binding),
        data_path=corpus,
        corpus_meta=meta,
        validation_seed_contract=validation,
        game_seed_column=seeds,
    )
    assert result["producer_checkpoint_sha256"] != parent_sha
    assert result["learner_parent_checkpoint_sha256"] == parent_sha
    assert result["learner_initializer_sha256"] == initializer_sha
