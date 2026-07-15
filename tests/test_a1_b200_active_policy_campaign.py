from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import a1_b200_active_policy_campaign as campaign
from tools import a1_current_science_contract as current_science
from tools import a1_one_dose_train as one_dose
from tools import train_bc


def _write_signed(path: Path, value: dict, field: str) -> None:
    value[field] = campaign._value_sha256(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_active_policy_arms_change_only_auxiliary_exposure() -> None:
    science = {
        "public_card_lr_mult": 4.0,
        "per_game_policy_surprise_weighting": True,
        "forced_row_value_action_type_weights": "END_TURN=0.1,ROLL=0.25",
    }
    assert {
        arm: values["policy_aux_active_batch_size"]
        for arm, values in campaign.ARMS.items()
    } == {"P10": 46, "P25": 116, "P50": 232, "P100": 463}
    recipes = {
        arm: campaign._arm_overrides(arm, science) for arm in campaign.ARMS
    }
    common = {
        key: value
        for key, value in recipes["P10"].items()
        if key != "policy_aux_active_batch_size"
    }
    assert all(
        {
            key: value
            for key, value in recipe.items()
            if key != "policy_aux_active_batch_size"
        }
        == common
        for recipe in recipes.values()
    )
    assert common["max_steps"] == 128


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
        "module": one_dose.architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_V2,
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


def test_selection_maximizes_teacher_gap_inside_explicit_drift_budgets(
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
            rows.append(
                {
                    "step": step,
                    "functional": {
                        "parent_kl": kls[arm] * step / 128,
                        "teacher_gap_closure": closures[arm] * step / 128,
                    },
                    "layer_drift": {
                        "trunk_relative_l2": min(kls[arm], 0.029) * step / 128,
                    },
                }
            )
        dose_telemetry = {
            "schema_version": "a1-active-policy-dose-telemetry-v1",
            "active_rows": {"policy_aux": 1},
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
            "parent_checkpoint_sha256": campaign_payload["lineage_contract"][
                "upgraded_initializer_sha256"
            ],
            "dose_telemetry": dose_telemetry,
            "checkpoints": rows,
        }
        _write_signed(path, payload, "fingerprint_sha256")
        bindings[arm] = path

    selected = campaign._select(campaign_path, campaign_payload, bindings)
    assert selected["winner"] == "P50"
    assert selected["eligible_arms"] == ["P10", "P25", "P50"]
    assert selected["arm_fingerprints"]["P100"]["within_drift_budgets"] is False
    assert selected["winner_meets_reference_teacher_gap_closure"] is True


def test_explicit_diagnostic_checkpoint_schedule_excludes_terminal(
    tmp_path: Path,
) -> None:
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
            "module": one_dose.architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_V2,
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
            "code_binding": {},
            "code_tree_sha256": "sha256:" + "9" * 64,
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
    expected_aux = 46 * campaign.WORLD_SIZE * campaign.MAX_STEPS
    report = {
        "policy_base_active_rows": 10_000,
        "policy_aux_active_rows": expected_aux,
        "policy_total_active_rows": 10_000 + expected_aux,
        "policy_base_effective_weight_sum": 20_000.0,
        "policy_aux_effective_weight_sum": 30_000.0,
        "policy_total_effective_weight_sum": 50_000.0,
        "value_active_rows": 500_000,
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
                    "scope": "rank_local_microbatch",
                    "policy_trunk_grad_norm": 0.7,
                    "policy_base_trunk_grad_norm": 0.5,
                    "policy_aux_trunk_grad_norm": 0.2,
                    "value_trunk_grad_norm": 0.3,
                    "policy_aux_to_base_grad_norm_ratio": 0.4,
                    "trunk_gradient_cosine": -0.1,
                    "policy_base_aux_gradient_cosine": 0.2,
                },
                {
                    "available": True,
                    "scope": "rank_local_microbatch",
                    "policy_trunk_grad_norm": 0.8,
                    "policy_base_trunk_grad_norm": 0.5,
                    "policy_aux_trunk_grad_norm": 0.3,
                    "value_trunk_grad_norm": 0.25,
                    "policy_aux_to_base_grad_norm_ratio": 0.6,
                    "trunk_gradient_cosine": -0.2,
                    "policy_base_aux_gradient_cosine": 0.1,
                },
            ],
        },
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
