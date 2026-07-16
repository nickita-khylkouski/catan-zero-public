from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from tools import a1_current_science_contract as current_science
from tools import a1_scratch_train as scratch
from tools import train_bc


def _science_binding() -> dict:
    values = {
        "science_schema_version": "a1-pre-wave-science-v2",
        "search_operator": current_science.search(),
        "evaluator": current_science.evaluator(),
        "learner_value_objective": {
            "objective": "scalar_mse",
            "value_readout": "deployed_tanh",
            "value_categorical_bins": 0,
            "hlgauss_sigma_ratio": 0.75,
        },
        "learner_training_recipe": current_science.learner_training_recipe(),
        "learner_initialization": current_science.learner_initialization(),
        "learner_model_construction": current_science.learner_model_construction(),
        "learner_execution_topology": current_science.learner_execution_topology(),
    }
    for key in tuple(values):
        if key != "science_schema_version":
            values[f"{key}_sha256"] = scratch._value_sha256(values[key])  # noqa: SLF001
    return values


def _write_semantic_json(path: Path, payload: dict, digest_field: str) -> dict:
    value = copy.deepcopy(payload)
    value[digest_field] = scratch._value_sha256(value)  # noqa: SLF001
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return value


def _authority_fixture(tmp_path: Path) -> tuple[dict, dict, dict]:
    descriptor_path = tmp_path / "composite.json"
    descriptor_path.write_text('{"schema_version":"memmap_composite_v2"}\n')
    descriptor_path = descriptor_path.resolve()
    descriptor_sha = scratch._file_sha256(descriptor_path)  # noqa: SLF001
    descriptor_fingerprint = "sha256:" + "1" * 64
    payload_inventory_sha256 = "sha256:" + "2" * 64

    staged_path = tmp_path / "staged.lock.json"
    staged_lock = _write_semantic_json(
        staged_path,
        {"science": _science_binding()},
        "contract_sha256",
    )
    staged_path = staged_path.resolve()
    staged_ref = {
        "path": str(staged_path),
        "file_sha256": scratch._file_sha256(staged_path),  # noqa: SLF001
        "contract_sha256": staged_lock["contract_sha256"],
    }
    source_ref = {
        "path": str((tmp_path / "source-authority.json").resolve()),
        "file_sha256": "sha256:" + "3" * 64,
        "authority_sha256": "sha256:" + "4" * 64,
    }
    source_authority = {
        "current_contract": staged_ref,
        "fresh_source_bindings": [],
    }
    source_semantic = scratch._value_sha256(source_authority)  # noqa: SLF001

    build_path = tmp_path / "build.receipt.json"
    build = _write_semantic_json(
        build_path,
        {
            "schema_version": "a1-post-wave-composite-build-v2",
            "descriptor": {
                "path": str(descriptor_path),
                "file_sha256": descriptor_sha,
                "fingerprint": descriptor_fingerprint,
            },
            "source_authority": source_ref,
            "contract": staged_ref,
        },
        "receipt_sha256",
    )
    build_path = build_path.resolve()
    build_ref = {
        "path": str(build_path),
        "file_sha256": scratch._file_sha256(build_path),  # noqa: SLF001
        "receipt_sha256": build["receipt_sha256"],
    }
    meta = {
        "schema_version": "memmap_composite_v2",
        "diagnostic_only": False,
        "promotion_eligible": True,
        "descriptor_file_sha256": descriptor_sha,
        "descriptor_fingerprint": descriptor_fingerprint,
        "payload_inventory_sha256": payload_inventory_sha256,
        "source_authority": source_authority,
        "source_authority_ref": source_ref,
        "source_authority_semantic_sha256": source_semantic,
    }
    authority = {
        "schema_version": scratch.CHILD_AUTHORITY_SCHEMA,
        "staged_contract": staged_ref,
        "science": _science_binding(),
        "descriptor": {
            "path": str(descriptor_path),
            "file_sha256": descriptor_sha,
            "fingerprint": descriptor_fingerprint,
            "payload_inventory_sha256": payload_inventory_sha256,
        },
        "source_authority": source_ref,
        "source_authority_semantic_sha256": source_semantic,
        "build_receipt": build_ref,
    }
    verified = _verified(tmp_path)
    verified.update(
        data_path=descriptor_path,
        corpus_meta_file_sha256=descriptor_sha,
        descriptor_fingerprint=descriptor_fingerprint,
        payload_inventory_sha256=payload_inventory_sha256,
        source_authority=source_authority,
        source_authority_ref=source_ref,
        source_authority_semantic_sha256=source_semantic,
        composite_build_receipt=build_ref,
    )
    return verified, meta, authority


def _verified(tmp_path: Path) -> dict:
    recipe = current_science.learner_training_recipe()
    topology = current_science.learner_execution_topology()
    recipe.update(
        world_size=topology["world_size"],
        batch_size=topology["local_batch_size"],
        grad_accum_steps=topology["grad_accum_steps"],
        global_batch_size=topology["global_batch_size"],
    )
    return {
        "recipe": recipe,
        "initialization": current_science.learner_initialization(),
        "model_construction": current_science.learner_model_construction(),
        "execution_topology": topology,
        "trainer_authority": scratch.one_dose._current_production_trainer_authority(),  # noqa: SLF001
        "data_path": tmp_path / "composite.json",
        "event_history_training_contract": {
            "empty_payload_inventory_acknowledgements": [],
            "training_event_history_trainable": True,
        },
    }


def test_scratch_command_is_native_bias_free_8gpu_and_fresh(tmp_path: Path) -> None:
    verified, _, authority = _authority_fixture(tmp_path)
    command = scratch.build_train_command(
        verified,
        python=Path("/usr/bin/python3"),
        checkpoint=tmp_path / "model.pt",
        report=tmp_path / "report.json",
    )

    assert command.count("torch.distributed.run") == 1
    assert command.count("--nproc_per_node=8") == 1
    assert command[command.index("--batch-size") + 1] == "64"
    assert command[command.index("--base-sampler") + 1] == (
        "coverage_importance_v1"
    )
    assert "--init-checkpoint" not in command
    assert "--grow-from-checkpoint" not in command
    assert "--resume-optimizer" not in command
    assert command.count("--no-resume-optimizer") == 1
    assert command.count("--no-public-card-count-residual-bias") == 1
    assert command.count("--public-rule-state-features") == 1
    assert command[command.index("--value-tower-split-layers") + 1] == "1"
    assert command.count("--meaningful-public-history-target-gather") == 1
    assert command.count("--entity-feature-adapter-version") == 1
    assert command.count("--fused-optimizer") == 1
    assert command.count("--symmetry-augment") == 1
    assert command.count("--symmetry-augment-events") == 1
    assert command.count("--required-target-information-regime") == 1
    assert command.count("--save-each-epoch") == 1
    assert command[command.index("--train-diagnostics-every-batches") + 1] == "16"
    assert (
        command[
            command.index(
                "--objective-gradient-interference-every-batches"
            )
            + 1
        ]
        == "16"
    )
    assert (
        command[
            command.index("--minimum-feature-learning-signal-observations")
            + 1
        ]
        == "2"
    )
    required_modules = command[
        command.index("--require-feature-learning-signal-modules") + 1
    ].split(",")
    assert "event_encoder" in required_modules
    assert "public_rule_state_residual" in required_modules
    assert "--ddp-shard-data" not in command
    assert "--target-reliability-confidence-weighting" not in command
    marker = json.loads(command[command.index("--a1-scratch-authority-json") + 1])
    assert marker == authority


def test_train_bc_fresh_create_boundary_builds_card_count_v2() -> None:
    model = current_science.learner_model_construction()
    args = SimpleNamespace(
        static_action_residual=True,
        legal_action_value_residual=True,
        public_card_count_residual_bias=False,
    )
    policy = EntityGraphPolicy.create(
        hidden_size=64,
        state_layers=1,
        attention_heads=8,
        dropout=0.0,
        device="cpu",
        public_card_count_features=True,
        public_rule_state_features=model["public_rule_state_features"],
        public_rule_state_feature_schema=model[
            "public_rule_state_feature_schema"
        ],
        entity_feature_adapter_version=model["entity_feature_adapter_version"],
        meaningful_public_history=model["meaningful_public_history"],
        meaningful_public_history_schema=model[
            "meaningful_public_history_schema"
        ],
        meaningful_public_history_pooling=model[
            "meaningful_public_history_pooling"
        ],
        meaningful_public_history_target_gather=model[
            "meaningful_public_history_target_gather"
        ],
        event_history_limit=model["event_history_limit"],
        **train_bc._structured_action_create_kwargs(args),  # noqa: SLF001
        **train_bc._public_card_count_create_kwargs(args),  # noqa: SLF001
    )

    assert policy.config.public_card_count_features is True
    assert policy.config.public_card_count_residual_bias is False
    assert policy.config.public_rule_state_features is True
    assert (
        policy.entity_feature_adapter_version
        == model["entity_feature_adapter_version"]
    )
    assert policy.config.static_action_residual is True
    assert policy.config.legal_action_value_residual is True


def test_planned_receipt_is_semantically_authenticated(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    scratch._write_receipt(  # noqa: SLF001
        path,
        {
            "schema_version": scratch.PLAN_SCHEMA,
            "status": "planned",
            "command": ["python", "train_bc.py"],
        },
    )
    payload = json.loads(path.read_text())
    unsigned = dict(payload)
    stated = unsigned.pop("receipt_sha256")
    assert stated == scratch._value_sha256(unsigned)  # noqa: SLF001


def test_completed_outputs_bind_terminal_report_and_epoch_frontier(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    report = tmp_path / "report.json"
    checkpoint.write_bytes(b"terminal")
    report.write_text('{"epochs":3}\n', encoding="utf-8")
    for epoch, epoch_path in enumerate(
        scratch._epoch_outputs(checkpoint, 3), start=1  # noqa: SLF001
    ):
        epoch_path.write_bytes(f"epoch-{epoch}".encode())
        Path(str(epoch_path) + ".optimizer.pt").write_bytes(b"optimizer")
        Path(str(epoch_path) + ".training-progress.json").write_text(
            "{}\n", encoding="utf-8"
        )

    outputs = scratch._completed_outputs(  # noqa: SLF001
        checkpoint=checkpoint,
        report=report,
        epochs=3,
    )

    assert outputs["terminal_checkpoint"]["path"] == str(checkpoint.resolve())
    assert [row["epoch"] for row in outputs["epoch_frontier"]] == [1, 2, 3]
    assert outputs["epoch_frontier_sha256"] == scratch._value_sha256(  # noqa: SLF001
        outputs["epoch_frontier"]
    )


def _runtime_args() -> SimpleNamespace:
    model = current_science.learner_model_construction()
    topology = current_science.learner_execution_topology()
    return SimpleNamespace(
        init_checkpoint="",
        grow_from_checkpoint="",
        resume_optimizer=False,
        arch=model["arch"],
        hidden_size=model["hidden_size"],
        graph_layers=model["graph_layers"],
        attention_heads=model["attention_heads"],
        graph_dropout=model["graph_dropout"],
        entity_state_trunk=model["entity_state_trunk"],
        static_action_residual=model["static_action_residual"],
        legal_action_value_residual=model["legal_action_value_residual"],
        value_tower_split_layers=model["value_tower_split_layers"],
        public_card_count_features=model["public_card_count_features"],
        public_card_count_residual_bias=model["public_card_count_residual_bias"],
        public_rule_state_features=model["public_rule_state_features"],
        entity_feature_adapter_version=model["entity_feature_adapter_version"],
        meaningful_public_history=model["meaningful_public_history"],
        meaningful_public_history_pooling=model[
            "meaningful_public_history_pooling"
        ],
        meaningful_public_history_target_gather=model[
            "meaningful_public_history_target_gather"
        ],
        event_history_limit=model["event_history_limit"],
        mask_hidden_info=model["mask_hidden_info"],
        require_35m_model=model["require_35m_model"],
        batch_size=topology["local_batch_size"],
        grad_accum_steps=topology["grad_accum_steps"],
        ddp_shard_data=topology["ddp_shard_data"],
        training_rng_rank_offset=topology["training_rng_rank_offset"],
    )


def test_scratch_runtime_projection_accepts_every_current_field() -> None:
    train_bc._validate_a1_scratch_runtime_projection(  # noqa: SLF001
        _runtime_args(),
        {"world_size": 8},
        current_science.learner_model_construction(),
        current_science.learner_execution_topology(),
    )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("static_action_residual", False),
        ("public_card_count_residual_bias", True),
        ("public_rule_state_features", False),
        ("meaningful_public_history", False),
        ("entity_state_trunk", "rrt"),
        ("mask_hidden_info", False),
        ("require_35m_model", False),
        ("training_rng_rank_offset", False),
        ("batch_size", 256),
    ),
)
def test_scratch_runtime_projection_rejects_grouped_tamper(
    field: str, bad_value
) -> None:
    args = _runtime_args()
    setattr(args, field, bad_value)
    with pytest.raises(SystemExit, match="scratch runtime projection drift"):
        train_bc._validate_a1_scratch_runtime_projection(  # noqa: SLF001
            args,
            {"world_size": 8},
            current_science.learner_model_construction(),
            current_science.learner_execution_topology(),
        )


def test_scratch_plan_has_explicit_execution_switch(tmp_path: Path) -> None:
    argv = [
        "--lock",
        str(tmp_path / "lock.json"),
        "--data",
        str(tmp_path / "descriptor.json"),
        "--composite-build-receipt",
        str(tmp_path / "build.json"),
        "--checkpoint",
        str(tmp_path / "model.pt"),
        "--report",
        str(tmp_path / "report.json"),
        "--receipt",
        str(tmp_path / "receipt.json"),
    ]
    parsed = scratch.parse_args(argv)
    assert parsed.go is False
    assert scratch.parse_args([*argv, "--go"]).go is True


def test_train_bc_refuses_current_unresolved_optimizer_authority() -> None:
    with pytest.raises(SystemExit, match="schedule is unresolved"):
        train_bc._require_a1_scratch_execution_schedule(  # noqa: SLF001
            current_science.learner_execution_topology()
        )


def test_train_bc_rejects_unresolved_scratch_authority_in_cheap_preflight(
    tmp_path: Path,
) -> None:
    _, _, authority = _authority_fixture(tmp_path)
    with pytest.raises(SystemExit, match="schedule is unresolved"):
        train_bc._preflight_a1_scratch_execution_authority(  # noqa: SLF001
            json.dumps(authority)
        )


def test_fresh_production_composite_cannot_delete_scratch_marker(
    tmp_path: Path,
) -> None:
    _, meta, _ = _authority_fixture(tmp_path)
    args = _runtime_args()
    args.a1_scratch_authority_json = ""
    with pytest.raises(SystemExit, match="requires the sealed A1 scratch plan marker"):
        train_bc._require_scratch_marker_for_fresh_production_composite(  # noqa: SLF001
            args, meta
        )


def test_exact_scratch_plan_binding_accepts_authenticated_inputs(
    tmp_path: Path,
) -> None:
    verified, meta, authority = _authority_fixture(tmp_path)
    binding = train_bc._validate_a1_scratch_plan_binding(  # noqa: SLF001
        authority,
        data_path=str(verified["data_path"]),
        composite_meta=meta,
    )
    assert binding["staged_contract"] == authority["staged_contract"]
    assert binding["descriptor"] == authority["descriptor"]


def test_scratch_plan_binding_rejects_swapped_descriptor(tmp_path: Path) -> None:
    _, meta, authority = _authority_fixture(tmp_path)
    swapped = tmp_path / "swapped.json"
    swapped.write_text("{}\n")
    with pytest.raises(SystemExit, match="descriptor/source authority binding drift"):
        train_bc._validate_a1_scratch_plan_binding(  # noqa: SLF001
            authority,
            data_path=str(swapped),
            composite_meta=meta,
        )


def test_scratch_plan_binding_rejects_swapped_staged_lock(tmp_path: Path) -> None:
    verified, meta, authority = _authority_fixture(tmp_path)
    staged_path = Path(authority["staged_contract"]["path"])
    staged_path.write_text(staged_path.read_text() + "\n")
    with pytest.raises(SystemExit, match="staged contract identity drift"):
        train_bc._validate_a1_scratch_plan_binding(  # noqa: SLF001
            authority,
            data_path=str(verified["data_path"]),
            composite_meta=meta,
        )


def test_scratch_plan_binding_rejects_current_science_self_assertion(
    tmp_path: Path,
) -> None:
    verified, meta, authority = _authority_fixture(tmp_path)
    tampered = copy.deepcopy(authority)
    tampered["science"]["learner_training_recipe"]["lr"] = 0.5
    tampered["science"]["learner_training_recipe_sha256"] = scratch._value_sha256(  # noqa: SLF001
        tampered["science"]["learner_training_recipe"]
    )
    with pytest.raises(SystemExit, match="self-asserts different science"):
        train_bc._validate_a1_scratch_plan_binding(  # noqa: SLF001
            tampered,
            data_path=str(verified["data_path"]),
            composite_meta=meta,
        )


def test_scratch_plan_binding_rejects_swapped_build_receipt(tmp_path: Path) -> None:
    verified, meta, authority = _authority_fixture(tmp_path)
    build_path = Path(authority["build_receipt"]["path"])
    build_path.write_text(build_path.read_text() + "\n")
    with pytest.raises(SystemExit, match="build-receipt binding drift"):
        train_bc._validate_a1_scratch_plan_binding(  # noqa: SLF001
            authority,
            data_path=str(verified["data_path"]),
            composite_meta=meta,
        )


def test_production_composite_dispatches_exact_scratch_authority_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, meta, authority = _authority_fixture(tmp_path)
    args = _runtime_args()
    args.data = str(verified["data_path"])
    args.a1_scratch_authority_json = json.dumps(authority)
    monkeypatch.setattr(
        train_bc, "_require_a1_scratch_execution_schedule", lambda _topology: None
    )
    binding = train_bc._validate_production_composite_scratch_binding(  # noqa: SLF001
        args,
        {"world_size": 8},
        meta,
    )
    assert binding["science"] == authority["science"]


def test_production_composite_dispatch_retains_late_schedule_refusal(
    tmp_path: Path,
) -> None:
    verified, meta, authority = _authority_fixture(tmp_path)
    args = _runtime_args()
    args.data = str(verified["data_path"])
    args.a1_scratch_authority_json = json.dumps(authority)
    with pytest.raises(SystemExit, match="schedule is unresolved"):
        train_bc._validate_production_composite_scratch_binding(  # noqa: SLF001
            args,
            {"world_size": 8},
            meta,
        )


def test_scratch_code_surface_binds_feature_signal_admission() -> None:
    assert "tools/a1_feature_signal_admission.py" in scratch.CODE_SURFACE
