from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from tools import a1_corrected_policy_arm as arm


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _source_receipt(
    path: Path,
    command: list[str],
    *,
    parent_checkpoint_sha256: str = "sha256:parent",
    descriptor_sha256: str = "sha256:descriptor",
    sentinel: str = "/validation/sentinel.json",
    sentinel_sha256: str = "sha256:sentinel",
) -> Path:
    payload = {
        "schema_version": "existing-sealed-training-receipt-v4",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "command": command,
        "command_sha256": arm._digest(command),
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "descriptor_sha256": descriptor_sha256,
        "sentinel": sentinel,
        "sentinel_sha256": sentinel_sha256,
    }
    payload["receipt_sha256"] = arm._digest(payload)
    return _write_json(path, payload)


def _base_command(tmp_path: Path) -> list[str]:
    return [
        "/venv/bin/python", "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node=8", "/old/tools/train_bc.py",
        "--data", "/old/descriptor.json", "--data-format", "memmap",
        "--validation-game-sentinel-manifest", "/old/validation.sentinel.json",
        "--init-checkpoint", "/old/f7.pt", "--checkpoint", "/old/candidate.pt",
        "--report", "/old/report.json", "--batch-size", "512",
        "--grad-accum-steps", "1", "--max-steps", "1024", "--epochs", "1",
        "--loser-sample-weight", "0.3", "--winner-sample-weight", "1.0",
        "--forced-action-weight", "0.0", "--forced-row-value-weight", "1.0",
        "--policy-loss-weight", "1.0", "--soft-target-source", "policy",
        "--soft-target-weight", "0.9", "--soft-target-temperature", "0.7",
        "--soft-target-min-legal-coverage", "0.5",
        "--policy-kl-anchor-direction", "forward", "--policy-kl-anchor-weight", "0.0",
        "--value-loss-weight", "0.25", "--value-lr-mult", "0.3",
        "--value-target-lambda", "1.0", "--lr", "3e-5",
        "--lr-warmup-steps", "100", "--lr-schedule", "flat",
        "--no-resume-optimizer", "--mask-hidden-info",
    ]


def _args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    future_two_component: bool = False,
) -> argparse.Namespace:
    descriptor = (tmp_path / "descriptor.json")
    descriptor.write_text("{}", encoding="utf-8")
    sentinel = _write_json(
        tmp_path / "validation.sentinel.json",
        {
            "schema_version": "train-validation-game-sentinel-v1",
            "selected_game_seed_set_sha256": "sha256:source-selection",
            "selected_row_count": 3,
            "game_seeds": [1],
        },
    )
    f7 = (tmp_path / "f7.pt")
    f7.write_bytes(b"f7")
    source = _source_receipt(
        tmp_path / "source.json",
        _base_command(tmp_path),
        parent_checkpoint_sha256=arm._file_sha(f7),
        descriptor_sha256=arm._file_sha(descriptor),
        sentinel=str(sentinel),
        sentinel_sha256=arm._file_sha(sentinel),
    )
    lineage = []
    for role in arm.LINEAGE_ROLES:
        payload = {"schema_version": f"{role}-v1"}
        payload[arm.LINEAGE_DIGEST_FIELDS[role]] = arm._digest(payload)
        artifact = _write_json(tmp_path / f"{role}.json", payload)
        lineage.append(f"{role}={artifact}")
    source_ids = (
        ("n128_current", "predecessor_replay")
        if future_two_component
        else ("n128_current", "n256_current", "gen3_replay")
    )
    source_ratios = [0.8, 0.2] if future_two_component else [4 / 7, 8 / 35, 1 / 5]
    source_components = [
        {
            "corpus_dir": f"/corpus/{component_id}",
            "corpus_meta_sha256": f"sha256:meta-{component_id}",
            "payload_inventory_sha256": "sha256:" + str(index) * 64,
            "validation_manifest": f"/validation/{component_id}.json",
            "validation_manifest_sha256": f"sha256:validation-{component_id}",
            "component_id": component_id,
            "game_sampling_ratio": source_ratios[index - 1],
            "corpus_meta": {
                "a1_post_wave_audit": {
                    "source_provenance": {
                        "current_producer": {
                            "producer_checkpoint_sha256": (
                                arm._file_sha(f7)
                                if component_id in arm.CURRENT_TEACHER_COMPONENT_IDS
                                else "sha256:" + "f" * 64
                            )
                        }
                    }
                }
            },
        }
        for index, component_id in enumerate(source_ids, start=1)
    ]

    def fake_preflight(path: Path):
        if Path(path).resolve() == descriptor.resolve():
            return (
                {
                    "schema_version": "memmap_composite_v2",
                    "component_ids": list(source_ids),
                    "component_game_sampling_ratios": source_ratios,
                    "components": source_components,
                    "learner_recipe_overrides": {
                        "per_game_policy_weight": False,
                        "per_game_policy_weight_mode": "equal",
                        "policy_kl_anchor_weight": 0.0,
                        "policy_kl_anchor_direction": "forward",
                        "loser_sample_weight": 0.3,
                    },
                },
                arm._file_ref(descriptor),
            )
        payload = json.loads(Path(path).read_text())
        component_ids = [row["component_id"] for row in payload["components"]]
        source_by_id = {row["component_id"]: row for row in source_components}
        verified_components = [
            row | {"corpus_meta": source_by_id[row["component_id"]]["corpus_meta"]}
            for row in payload["components"]
        ]
        return (
            {
                "schema_version": "memmap_composite_v2",
                "descriptor_fingerprint": arm._digest(payload),
                "policy_distillation_scope_explicit": True,
                "value_training_scope_explicit": True,
                "policy_distillation_component_ids": component_ids,
                "value_training_component_ids": component_ids,
                "component_ids": component_ids,
                "components": verified_components,
                "component_game_sampling_ratios": [
                    row["game_sampling_ratio"] for row in payload["components"]
                ],
                "policy_kl_anchor_component_ids": component_ids,
                "learner_recipe_overrides": payload["learner_recipe_overrides"],
            },
            arm._file_ref(path),
        )

    monkeypatch.setattr(arm, "_preflight_descriptor", fake_preflight)
    monkeypatch.setattr(
        arm,
        "_source_binding",
        lambda repo: {"repository_root": str(repo), "git_commit": "abc", "files": {}},
    )
    monkeypatch.setattr(
        arm,
        "_rebind_a1_metadata",
        lambda command, repo: {"effective_recipe": {}, "code_binding": {}},
    )
    def fake_sentinel(**_kwargs):
        fresh = _write_json(
            tmp_path / "fresh.validation.sentinel.json",
            {
                "schema_version": "train-validation-game-sentinel-v1",
                "selected_game_seed_set_sha256": "sha256:selection",
                "selected_row_count": len(source_ids),
                "game_seeds": list(range(2, 2 + len(source_ids))),
            },
        )
        component_rows = [
            {
                "component_id": component_id,
                "target_row_ratio": float(ratio),
                "target_row_count": 1,
                "selected_game_count": 1,
                "selected_row_count": 1,
                "max_whole_game_row_count": 1,
            }
            for component_id, ratio in zip(source_ids, source_ratios, strict=True)
        ]
        independence = {
            "schema_version": "a1-validation-independence-contract-v1",
            "source_selected_game_seed_set_sha256": "sha256:source-selection",
            "fresh_selected_game_seed_set_sha256": "sha256:selection",
            "selection_overlap_game_count": 0,
            "selection_scope": "fresh_whole_games_stratified_to_winning_operator",
            "component_rows": component_rows,
            "predecessor_component_id": source_ids[-1],
            "predecessor_target_row_ratio": 0.2,
            "complete_component_holdouts_remain_training_excluded": True,
        }
        independence["contract_sha256"] = arm._digest(independence)
        return (
            json.loads(fresh.read_text()),
            arm._file_ref(fresh),
            arm._file_ref(sentinel),
            independence,
        )

    monkeypatch.setattr(arm, "_build_corrected_sentinel", fake_sentinel)
    active_rows = {
        "n128_current": 121_132,
        "n256_current": 121_128,
        "gen3_replay": 130_532,
        "predecessor_replay": 130_532,
    }
    game_uniform_active_fractions = {
        "n128_current": 0.12139372557799329,
        "n256_current": 0.12125098774975895,
        "gen3_replay": 0.1291620365495644,
        "predecessor_replay": 0.1291620365495644,
    }
    monkeypatch.setattr(
        arm,
        "_component_training_policy_activity",
        lambda component: {
            "component_id": component["component_id"],
            "training_rows": 5_000_000,
            "training_policy_active_rows": active_rows[component["component_id"]],
            "training_game_count": 10_000,
            "raw_row_policy_active_fraction": (
                active_rows[component["component_id"]] / 1_000_000
            ),
            "game_uniform_policy_active_fraction": (
                game_uniform_active_fractions[component["component_id"]]
            ),
            "validation_manifest_sha256": component["validation_manifest_sha256"],
            "payload_inventory_sha256": component["payload_inventory_sha256"],
        },
    )
    return argparse.Namespace(
        source_receipt=source,
        source_descriptor=descriptor,
        f7_checkpoint=f7,
        expected_f7_sha256=arm._file_sha(f7),
        failed_lineage_artifact=lineage,
        output_root=tmp_path / "out",
        repo=tmp_path,
    )


def test_prepares_exact_one_dose_winning_operator_control_without_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, path = arm.prepare(_args(tmp_path, monkeypatch))
    assert path.is_file()
    assert manifest["launch_authorized"] is False
    assert manifest["diagnostic_execution_authorized"] is True
    assert manifest["recipe"]["base_value_row_dose"] == 524_288
    assert manifest["recipe"]["policy_aux_active_row_dose"] == 0
    assert manifest["recipe"]["expected_policy_base_active_rows"] == 64_443
    assert manifest["recipe"]["policy_base_active_row_tolerance"] == 4_100
    assert manifest["recipe"]["expected_policy_aux_active_rows"] == 0
    assert manifest["recipe"]["policy_distillation_component_ids"] == [
        "n128_current", "n256_current", "gen3_replay"
    ]
    assert manifest["recipe"]["component_game_sampling_ratios"] == pytest.approx(
        [4 / 7, 8 / 35, 1 / 5]
    )
    assert manifest["recipe"]["replay_supervised_policy"] is True
    assert manifest["recipe"]["replay_supervised_value"] is True
    assert manifest["recipe"]["replay_forward_kl_weight"] == 0.0
    assert manifest["causal_interpretation"]["exact_winning_operator_control"] is True
    assert manifest["causal_interpretation"]["bundled_optimization_not_parent_replication"] is False
    assert manifest["recipe"]["independent_parent_initialization"] is True
    assert manifest["evaluation_baseline"] == manifest["initialization"]
    assert manifest["causal_interpretation"][
        "teacher_gap_closure_is_search_improvement_over_initializer"
    ] is True
    assert manifest["parent_lineage"]["mode"] == "historical_f7_cli_compatibility"
    assert manifest["validation_independence_contract"][
        "selection_overlap_game_count"
    ] == 0
    assert manifest["teacher_lineage"]["learner_parent_checkpoint_sha256"] == (
        manifest["initialization"]["sha256"]
    )
    assert manifest["teacher_lineage"]["predecessor_checkpoint_sha256"] != (
        manifest["initialization"]["sha256"]
    )
    descriptor = json.loads(Path(manifest["descriptor"]["path"]).read_text())
    assert [row["component_id"] for row in descriptor["components"]] == [
        "n128_current", "n256_current", "gen3_replay"
    ]
    assert descriptor["policy_kl_anchor_component_ids"] == [
        "n128_current", "n256_current", "gen3_replay"
    ]
    assert descriptor["policy_distillation_component_ids"] == [
        "n128_current", "n256_current", "gen3_replay"
    ]
    assert descriptor["value_training_component_ids"] == [
        "n128_current", "n256_current", "gen3_replay"
    ]
    assert descriptor["learner_recipe_overrides"]["policy_kl_anchor_weight"] == 0.0
    command = manifest["command"]
    assert arm._option(command, "--soft-target-weight") == "0.9"
    assert arm._option(command, "--policy-aux-active-batch-size") == "0"
    assert arm._option(command, "--policy-kl-anchor-weight") == "0.0"
    assert arm._option(command, "--policy-kl-anchor-direction") == "forward"
    assert arm._option(command, "--loser-sample-weight") == "1.0"
    assert arm._option(command, "--max-steps") == "128"
    assert command.count("--validation-game-sentinel-manifest") == 1
    assert command.count(arm.EVENT_HISTORY_ACK_FLAG) == 3
    assert command.count(arm.EVENT_HISTORY_CROP_FLAG) == 1
    assert manifest["event_history_training_contract"][
        "crop_authenticated_empty_event_history"
    ] is True
    assert manifest["supervision_contract"]["soft_target_weight"] == 0.9
    assert manifest["supervision_contract"]["replay_objective"] == (
        "supervised_policy_and_value_exact_winning_operator"
    )
    assert manifest["supervision_contract"]["policy_aux_active_batch_size_per_rank"] == 0
    active_dose = manifest["supervision_contract"]["policy_active_row_dose"]
    assert active_dose == {
        "derivation": "authenticated_game_uniform_activity_weighted_by_component_sampling_ratio",
        "component_statistics": active_dose["component_statistics"],
        "component_sampling_ratios": pytest.approx([4 / 7, 8 / 35, 1 / 5]),
        "global_row_dose": 524_288,
        "available_training_rows": 15_000_000,
        "expected_active_fraction": pytest.approx(0.1229147619829968),
        "reference_base_active_rows": 64_443,
        "base_active_rows_tolerance": 4_100,
        "min_base_active_rows": 60_343,
        "max_base_active_rows": 68_543,
        "expected_aux_active_rows": 0,
        "accounting": "realized_policy_active_rows_not_global_samples",
    }
    assert "--validation-game-seed-manifest" not in command
    assert command[command.index("torch.distributed.run") + 1] == "--standalone"
    assert [row["role"] for row in manifest["failed_retry_lineage"]["artifacts"]] == list(
        arm.LINEAGE_ROLES
    )


def test_event_history_binding_preserves_winning_replay_ack() -> None:
    current = ["sha256:" + "1" * 64, "sha256:" + "2" * 64]
    replay = "sha256:" + "3" * 64
    command = ["python", "train_bc.py"]
    for inventory in [*current, replay]:
        command.extend((arm.EVENT_HISTORY_ACK_FLAG, inventory))
    descriptor = {
        "components": [
            {
                "component_id": component,
                "payload_inventory_sha256": inventory,
            }
            for component, inventory in zip(
                (*arm.CURRENT_TEACHER_COMPONENT_IDS, arm.REPLAY_COMPONENT_ID),
                (*current, replay), strict=True
            )
        ]
    }

    contract, change = arm._bind_event_history_training_command(command, descriptor)

    positions = [
        index for index, value in enumerate(command) if value == arm.EVENT_HISTORY_ACK_FLAG
    ]
    assert [command[index + 1] for index in positions] == [*current, replay]
    assert change["event_history_acknowledgements"]["source"] == [*current, replay]
    assert contract["empty_payload_inventory_acknowledgements"][0]["component_id"] == (
        "n128_current"
    )


def test_refuses_incomplete_failed_retry_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, monkeypatch)
    args.failed_lineage_artifact.pop()
    with pytest.raises(arm.ArmError, match="lineage is incomplete"):
        arm.prepare(args)


def test_refuses_source_receipt_command_digest_drift(tmp_path: Path) -> None:
    path = _source_receipt(tmp_path / "source.json", _base_command(tmp_path))
    payload = json.loads(path.read_text())
    payload["command"].extend(("--lr", "999"))
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(arm.ArmError, match="semantic digest"):
        arm._load_source_receipt(path)


def test_active_dose_is_derived_from_training_only_component_rows(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    game_seeds = np.asarray([1, 1, 2, 2, 3, 3], dtype="<i8")
    policy_weights = np.asarray([1, 0, 1, 1, 1, 0], dtype="<f4")
    game_seeds.tofile(root / "game_seed.dat")
    policy_weights.tofile(root / "policy_weight_multiplier.dat")
    validation = _write_json(tmp_path / "validation.json", {"game_seeds": [2]})
    component = {
        "component_id": "tiny",
        "corpus_dir": str(root),
        "corpus_meta": {"row_count": 6},
        "validation_manifest": str(validation),
        "validation_manifest_sha256": arm._file_sha(validation),
        "payload_inventory_sha256": "sha256:" + "1" * 64,
    }
    measured = arm._component_training_policy_activity(component)
    assert measured["training_rows"] == 4
    assert measured["training_policy_active_rows"] == 2
    assert measured["training_game_count"] == 2
    assert measured["raw_row_policy_active_fraction"] == 0.5
    assert measured["game_uniform_policy_active_fraction"] == 0.5


def test_active_dose_refuses_corpus_that_cannot_reach_one_dose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = {
        "component_ids": ["n128_current", "predecessor_replay"],
        "component_game_sampling_ratios": [0.8, 0.2],
        "components": [
            {"component_id": "n128_current"},
            {"component_id": "predecessor_replay"},
        ],
    }
    monkeypatch.setattr(
        arm,
        "_component_training_policy_activity",
        lambda component: {
            "component_id": component["component_id"],
                "training_rows": 200_000,
            "game_uniform_policy_active_fraction": 0.1,
        },
    )
    with pytest.raises(arm.ArmError, match="smaller than the sealed one-dose draw"):
        arm._derive_policy_active_dose(descriptor)


def test_teacher_lineage_refuses_stale_current_data_and_parent_replay() -> None:
    parent = "sha256:" + "a" * 64
    predecessor = "sha256:" + "b" * 64

    def component(component_id: str, producer: str) -> dict:
        return {
            "component_id": component_id,
            "corpus_meta": {
                "a1_post_wave_audit": {
                    "source_provenance": {
                        "current_producer": {
                            "producer_checkpoint_sha256": producer
                        }
                    }
                }
            },
        }

    valid = {
        "components": [
            component("n128_current", parent),
            component("predecessor_replay", predecessor),
        ]
    }
    assert arm._bind_teacher_lineage(
        valid,
        parent_checkpoint_sha256=parent,
        expected_predecessor_sha256=predecessor,
    )["predecessor_checkpoint_sha256"] == predecessor
    stale = {
        "components": [
            component("n128_current", predecessor),
            component("predecessor_replay", "sha256:" + "c" * 64),
        ]
    }
    with pytest.raises(arm.ArmError, match="current teacher data was not generated"):
        arm._bind_teacher_lineage(stale, parent_checkpoint_sha256=parent)
    self_replay = {
        "components": [
            component("n128_current", parent),
            component("predecessor_replay", parent),
        ]
    }
    with pytest.raises(arm.ArmError, match="parent's predecessor"):
        arm._bind_teacher_lineage(self_replay, parent_checkpoint_sha256=parent)
    with pytest.raises(arm.ArmError, match="exact dethroned champion"):
        arm._bind_teacher_lineage(
            valid,
            parent_checkpoint_sha256=parent,
            expected_predecessor_sha256="sha256:" + "c" * 64,
        )


def test_rederives_sentinel_after_descriptor_scope_is_made_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_sentinel = _write_json(
        tmp_path / "source.sentinel.json",
        {
            "schema_version": "train-validation-game-sentinel-v1",
            "source_composite_descriptor_file_sha256": "sha256:old-file",
            "source_composite_descriptor_fingerprint": "sha256:old-fingerprint",
            "selection_seed": 20260711,
            "target_row_count": 4,
            "selected_game_seed_set_sha256": "sha256:same-games",
            "game_seeds": [1, 2],
        },
    )
    receipt = {
        "sentinel": str(source_sentinel),
        "sentinel_sha256": arm._file_sha(source_sentinel),
    }
    output = tmp_path / "new.sentinel.json"

    def run(command, **_kwargs):
        assert "--validation-game-seed-manifest" not in command
        _write_json(
            output,
            {
                "schema_version": "train-validation-game-sentinel-v1",
                "source_composite_descriptor_file_sha256": "sha256:new-file",
                "source_composite_descriptor_fingerprint": "sha256:new-fingerprint",
                "selection_seed": 20260711,
                "target_row_count": 4,
                "selected_game_seed_set_sha256": "sha256:new-current-games",
                "excluded_game_seed_set_sha256": "sha256:current-holdout",
                "selected_game_seed_count": 2,
                "selected_row_count": 4,
                "game_seeds": [3, 4],
            },
        )
        return None

    monkeypatch.setattr(arm.subprocess, "run", run)
    current_dir = tmp_path / "current"
    predecessor_dir = tmp_path / "predecessor"
    current_dir.mkdir()
    predecessor_dir.mkdir()
    np.asarray([1, 3, 3, 3], dtype="<i8").tofile(current_dir / "game_seed.dat")
    np.asarray([2, 4], dtype="<i8").tofile(predecessor_dir / "game_seed.dat")
    current_validation = _write_json(
        tmp_path / "current.validation.json",
        {"schema_version": "train-validation-game-seeds-v1", "game_seeds": [1, 3]},
    )
    predecessor_validation = _write_json(
        tmp_path / "predecessor.validation.json",
        {"schema_version": "train-validation-game-seeds-v1", "game_seeds": [2, 4]},
    )
    corrected, _ref, source_ref, independence = arm._build_corrected_sentinel(
        source_receipt=receipt,
        source_descriptor={
            "descriptor_file_sha256": "sha256:old-file",
            "descriptor_fingerprint": "sha256:old-fingerprint",
        },
        descriptor=tmp_path / "descriptor.json",
        descriptor_meta={
            "descriptor_file_sha256": "sha256:new-file",
            "descriptor_fingerprint": "sha256:new-fingerprint",
            "component_ids": ["n128_current", "predecessor_replay"],
            "component_game_sampling_ratios": [0.8, 0.2],
            "components": [
                {
                    "component_id": "n128_current",
                    "validation_manifest": str(current_validation),
                    "corpus_dir": str(current_dir),
                    "corpus_meta": {"row_count": 4},
                },
                {
                    "component_id": "predecessor_replay",
                    "validation_manifest": str(predecessor_validation),
                    "corpus_dir": str(predecessor_dir),
                    "corpus_meta": {"row_count": 2},
                },
            ],
        },
        output_path=output,
        python="python",
        repo=tmp_path,
    )
    assert corrected["selected_game_seed_set_sha256"] == "sha256:new-current-games"
    assert source_ref["sha256"] == receipt["sentinel_sha256"]
    assert independence["selection_overlap_game_count"] == 0
    assert independence["predecessor_component_id"] == "predecessor_replay"


def test_descriptor_builder_refuses_noncanonical_source_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "descriptor.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        arm,
        "_preflight_descriptor",
        lambda _path: ({
            "schema_version": "memmap_composite_v2",
            "component_ids": ["n128_current", "gen3_replay", "n256_current"],
        }, arm._file_ref(path)),
    )
    with pytest.raises(arm.ArmError, match="80% current teachers then 20% predecessor replay"):
        arm._build_corrected_descriptor(path, tmp_path / "out.json")


def test_command_requires_hidden_information_masking(tmp_path: Path) -> None:
    command = _base_command(tmp_path)
    command.remove("--mask-hidden-info")
    with pytest.raises(arm.ArmError, match="required safety flag"):
        arm._derive_command(
            command,
            repo=tmp_path,
            descriptor=tmp_path / "d",
            sentinel=tmp_path / "v",
            parent=tmp_path / "f",
            output_root=tmp_path / "out",
        )


def test_rebinds_full_tracked_runtime_closure_and_effective_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = tmp_path / "tools" / "train_bc.py"
    trainer.parent.mkdir()
    trainer.write_text("# trainer\n", encoding="utf-8")
    effective = {
        "batch_size": 512, "grad_accum_steps": 1, "global_batch_size": 4096,
        "world_size": 8, "max_steps": 1024, "epochs": 1,
        "loser_sample_weight": 0.3, "winner_sample_weight": 1.0,
        "forced_action_weight": 0.0, "forced_row_value_weight": 1.0,
        "policy_loss_weight": 1.0, "soft_target_source": "policy",
        "soft_target_weight": 0.9, "soft_target_temperature": 0.7,
        "soft_target_min_legal_coverage": 0.5, "policy_kl_anchor_weight": 0.0,
        "value_loss_weight": 0.25, "value_lr_mult": 0.3,
        "value_target_lambda": 1.0, "lr": 3e-5,
        "lr_warmup_steps": 100, "lr_schedule": "flat",
    }
    binding = {
        "schema_version": "a1-learner-ablation-code-binding-v1",
        "repository_root": "/old",
        "records": [{"kind": "learner_code", "relative_path": "tools/train_bc.py",
                     "path": "/old/train_bc.py", "sha256": "sha256:old"}],
        "code_tree_sha256": "sha256:old",
    }
    command = [
        "python", "tools/train_bc.py",
        "--a1-learner-ablation-id", "old",
        "--a1-effective-learner-recipe-json", arm._canonical(effective).decode(),
        "--a1-effective-learner-recipe-sha256", arm._digest(effective),
        "--a1-ablation-code-binding-json", arm._canonical(binding).decode(),
        "--a1-ablation-code-tree-sha256", "sha256:old",
        "--a1-reviewed-lock-file-sha256", "sha256:lock",
    ]
    monkeypatch.setattr(arm.subprocess, "run", lambda *args, **kwargs: None)
    result = arm._rebind_a1_metadata(command, tmp_path)
    assert result["effective_recipe"]["policy_aux_active_batch_size"] == 0
    assert result["effective_recipe"]["soft_target_weight"] == 0.9
    assert result["effective_recipe"]["max_steps"] == 128
    assert arm._option(command, "--a1-learner-ablation-id") == (
        "next-winning-operator-control"
    )
    assert json.loads(arm._option(command, "--a1-effective-learner-recipe-json")) == (
        result["effective_recipe"]
    )
    rebound = json.loads(arm._option(command, "--a1-ablation-code-binding-json"))
    assert rebound["records"][0]["path"] == str(trainer)
    assert rebound["records"][0]["sha256"] == arm._file_sha(trainer)


def test_rebind_refuses_gradient_probe_in_training_runtime(tmp_path: Path) -> None:
    probe = tmp_path / "tools" / "a1_shared_trunk_gradient_probe.py"
    probe.parent.mkdir()
    probe.write_text("# diagnostic only\n", encoding="utf-8")
    effective = {
        "batch_size": 512, "grad_accum_steps": 1, "global_batch_size": 4096,
        "world_size": 8, "max_steps": 1024, "epochs": 1,
        "loser_sample_weight": 1.0, "winner_sample_weight": 1.0,
        "forced_action_weight": 0.0, "forced_row_value_weight": 1.0,
        "policy_loss_weight": 1.0, "soft_target_source": "policy",
        "soft_target_weight": 0.9, "soft_target_temperature": 0.7,
        "soft_target_min_legal_coverage": 0.5, "policy_kl_anchor_weight": 0.0,
        "value_loss_weight": 0.25, "value_lr_mult": 0.3,
        "value_target_lambda": 1.0, "lr": 3e-5,
        "lr_warmup_steps": 100, "lr_schedule": "flat",
    }
    binding = {"records": [{"kind": "learner_code",
        "relative_path": "tools/a1_shared_trunk_gradient_probe.py"}]}
    command = ["python", "tools/train_bc.py"]
    for flag, value in (
        ("--a1-learner-ablation-id", "old"),
        ("--a1-effective-learner-recipe-json", arm._canonical(effective).decode()),
        ("--a1-effective-learner-recipe-sha256", arm._digest(effective)),
        ("--a1-ablation-code-binding-json", arm._canonical(binding).decode()),
        ("--a1-ablation-code-tree-sha256", "sha256:old"),
        ("--a1-reviewed-lock-file-sha256", "sha256:lock"),
    ):
        command.extend((flag, value))
    with pytest.raises(arm.ArmError, match="untracked gradient probe"):
        arm._rebind_a1_metadata(command, tmp_path)
