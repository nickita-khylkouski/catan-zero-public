from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import a1_production_temperature_replication as temp
from tools import a1_promotion_transaction as promotion


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _command(tmp_path: Path) -> list[str]:
    descriptor = str(tmp_path / "descriptor.json")
    sentinel = str(tmp_path / "sentinel.json")
    f7 = str(tmp_path / "f7.pt")
    return [
        "/venv/python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        "/repo/tools/train_bc.py",
        "--data",
        descriptor,
        "--data-format",
        "memmap",
        "--init-checkpoint",
        f7,
        "--arch",
        "entity_graph",
        "--hidden-size",
        "640",
        "--graph-layers",
        "6",
        "--attention-heads",
        "8",
        "--graph-dropout",
        "0.05",
        "--entity-state-trunk",
        "transformer",
        "--track",
        "2p_no_trade",
        "--vps-to-win",
        "10",
        "--graph-history-features",
        "--mask-hidden-info",
        "--epochs",
        "1",
        "--max-steps",
        "1024",
        "--batch-size",
        "512",
        "--grad-accum-steps",
        "1",
        "--seed",
        "1",
        "--training-rng-rank-offset",
        "--optimizer",
        "adam",
        "--no-resume-optimizer",
        "--no-fused-optimizer",
        "--lr",
        "3e-05",
        "--lr-warmup-steps",
        "100",
        "--lr-schedule",
        "flat",
        "--weight-decay",
        "0.0",
        "--value-lr-mult",
        "0.3",
        "--action-module-lr-mult",
        "1.0",
        "--policy-loss-weight",
        "1.0",
        "--soft-target-source",
        "policy",
        "--soft-target-weight",
        "0.9",
        "--soft-target-min-legal-coverage",
        "0.5",
        "--value-loss-weight",
        "0.25",
        "--value-target-lambda",
        "1.0",
        "--value-head-type",
        "mse",
        "--truncated-vp-margin-value-weight",
        "0.0",
        "--final-vp-loss-weight",
        "0.0",
        "--q-loss-weight",
        "0.0",
        "--policy-kl-anchor-weight",
        "0.0",
        "--policy-kl-anchor-direction",
        "forward",
        "--forced-action-weight",
        "0.0",
        "--forced-row-value-weight",
        "1.0",
        "--winner-sample-weight",
        "1.0",
        "--loser-sample-weight",
        "1.0",
        "--validation-max-samples",
        "0",
        "--skip-teacher-quality-gate",
        "--trust-curated-data-quality",
        "--data-loader-workers",
        "4",
        "--data-loader-prefetch",
        "4",
        "--validation-game-sentinel-manifest",
        sentinel,
        "--checkpoint",
        str(tmp_path / "diagnostic.pt"),
        "--report",
        str(tmp_path / "report.json"),
    ]


def _descriptor(tmp_path: Path) -> Path:
    components = []
    for index, (component_id, ratio) in enumerate(
        zip(temp.COMPONENT_IDS, temp.COMPONENT_RATIOS, strict=True)
    ):
        corpus = tmp_path / f"corpus-{index}"
        corpus.mkdir()
        meta = corpus / "corpus_meta.json"
        meta.write_text("{}", encoding="utf-8")
        validation = tmp_path / f"validation-{index}.json"
        validation.write_text("{}", encoding="utf-8")
        components.append(
            {
                "component_id": component_id,
                "corpus_dir": str(corpus),
                "corpus_meta_sha256": temp.base._file_sha(meta),
                "game_sampling_ratio": ratio,
                "payload_inventory_sha256": "sha256:" + str(index) * 64,
                "validation_manifest": str(validation),
                "validation_manifest_sha256": temp.base._file_sha(validation),
            }
        )
    return _write(
        tmp_path / "descriptor.json",
        {
            "schema_version": "memmap_composite_v2",
            "diagnostic_only": True,
            "promotion_eligible": False,
            "components": components,
            "stored_policy_component_temperatures": temp.COMPONENT_TEMPERATURES,
            "policy_distillation_component_ids": list(temp.COMPONENT_IDS),
            "value_training_component_ids": list(temp.COMPONENT_IDS),
            "policy_kl_anchor_component_ids": ["gen3_replay"],
        },
    )


def test_descriptor_preserves_diagnostic_boundary_and_exact_temperature_map(
    tmp_path: Path,
) -> None:
    path = _descriptor(tmp_path)
    payload, inventories, bindings = temp._verify_descriptor(path)

    assert payload["diagnostic_only"] is True
    assert payload["promotion_eligible"] is False
    assert payload["stored_policy_component_temperatures"] == {
        "n128_current": 1.0,
        "n256_current": 1.11,
        "gen3_replay": 0.52,
    }
    assert len(inventories) == len(bindings) == 3


def test_component_ratios_match_authenticated_descriptor_encoding() -> None:
    assert list(temp.COMPONENT_RATIOS) == [
        0.5714285714285715,
        0.22857142857142856,
        0.2,
    ]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("promotion_eligible", True, "must remain diagnostic"),
        ("diagnostic_only", False, "must remain diagnostic"),
        (
            "stored_policy_component_temperatures",
            {"n128_current": 1.0, "n256_current": 1.0, "gen3_replay": 0.52},
            "temperature map drift",
        ),
    ],
)
def test_descriptor_refuses_relabel_or_temperature_drift(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    path = _descriptor(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    _write(path, payload)

    with pytest.raises(temp.TemperatureReplicationError, match=match):
        temp._verify_descriptor(path)


def test_winning_recipe_accepts_only_exact_fresh_f7_dose(tmp_path: Path) -> None:
    command = _command(tmp_path)
    temp._validate_recipe(
        command,
        descriptor=str(tmp_path / "descriptor.json"),
        sentinel=str(tmp_path / "sentinel.json"),
        f7=str(tmp_path / "f7.pt"),
    )


def test_production_command_adds_only_outputs_runtime_and_proven_empty_crop(
    tmp_path: Path,
) -> None:
    trainer = tmp_path / "checkout" / "tools" / "train_bc.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# bound trainer\n", encoding="utf-8")
    selected = _command(tmp_path)

    production = temp._production_command(
        selected,
        python="/bound/venv/python",
        trainer=trainer,
        checkpoint=tmp_path / "production" / "candidate.pt",
        report=tmp_path / "production" / "report.json",
    )

    expected = list(selected)
    expected[0] = "/bound/venv/python"
    expected[expected.index("/repo/tools/train_bc.py")] = str(trainer.resolve())
    expected[expected.index("--checkpoint") + 1] = str(
        tmp_path / "production" / "candidate.pt"
    )
    expected[expected.index("--report") + 1] = str(
        tmp_path / "production" / "report.json"
    )
    expected[expected.index("--max-steps") + 1] = "128"
    expected.append(temp.base.CROP_FLAG)
    assert production == expected
    assert production.count(temp.base.CROP_FLAG) == 1


def test_new_production_dose_is_typed_short_while_diagnostic_stays_full() -> None:
    selected = {
        **temp.base.learner_dose.PARETO_SELECTED_DOSE.payload(),
        "optimizer": "fresh_adam",
        "lr": 3e-5,
        "training_rng_rank_offset": True,
    }
    manifest = {"schema_version": temp.MANIFEST_SCHEMA, "selected_dose": selected}

    assert temp._manifest_dose(manifest) == temp.base.learner_dose.PARETO_SELECTED_DOSE
    assert temp.SEALED_REPORT_RECIPE["max_steps"] == 128
    assert temp.SEALED_REPORT_RECIPE["base_training_row_draws"] == 524_288


def test_production_command_refuses_ambiguous_preexisting_crop(
    tmp_path: Path,
) -> None:
    trainer = tmp_path / "checkout" / "tools" / "train_bc.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# bound trainer\n", encoding="utf-8")
    selected = [*_command(tmp_path), temp.base.CROP_FLAG]

    with pytest.raises(temp.TemperatureReplicationError, match="already contains"):
        temp._production_command(
            selected,
            python="/bound/venv/python",
            trainer=trainer,
            checkpoint=tmp_path / "production" / "candidate.pt",
            report=tmp_path / "production" / "report.json",
        )


def test_production_recipe_validator_alone_does_not_authorize_additive_flags(
    tmp_path: Path,
) -> None:
    """Document why verify must compare the complete derived command."""

    command = _command(tmp_path)
    command.append("--symmetry-augment")
    # Pointwise flag validation intentionally permits future unrelated flags;
    # the sealed transaction's complete-command equality is the fail-closed
    # layer that rejects this causal addition.
    temp._validate_recipe(
        command,
        descriptor=str(tmp_path / "descriptor.json"),
        sentinel=str(tmp_path / "sentinel.json"),
        f7=str(tmp_path / "f7.pt"),
    )
    assert command != _command(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (("--lr", "0.0001"), "recipe drift"),
        (("--max-steps", "2048"), "recipe drift"),
        (("--nproc-per-node=8", "--nproc-per-node=4"), "nproc-per-node"),
        (("--training-rng-rank-offset", None), "lacks required flags"),
        (("--no-resume-optimizer", "--resume-optimizer"), "lacks required flags"),
    ],
)
def test_winning_recipe_refuses_causal_drift(
    tmp_path: Path, mutation: tuple[str, str | None], match: str
) -> None:
    command = _command(tmp_path)
    old, new = mutation
    if old.startswith("--") and old in command and new is not None:
        index = command.index(old)
        if old in {"--lr", "--max-steps"}:
            command[index + 1] = new
        else:
            command[index] = new
    elif old in command:
        command.remove(old)

    with pytest.raises(temp.TemperatureReplicationError, match=match):
        temp._validate_recipe(
            command,
            descriptor=str(tmp_path / "descriptor.json"),
            sentinel=str(tmp_path / "sentinel.json"),
            f7=str(tmp_path / "f7.pt"),
        )


@pytest.mark.parametrize(
    ("field", "drifted"),
    [
        ("graph_layers", 5),
        ("attention_heads", 4),
        ("graph_dropout", 0.0),
        ("amp", "bf16"),
        ("max_grad_norm", 0.0),
        ("soft_target_temperature", 1.0),
        ("per_game_policy_weight", True),
        ("phase_weights", {"setup": 2.0}),
        ("policy_surprise_weight", 1.0),
    ],
)
def test_completed_recipe_binding_covers_silent_strength_drift(
    field: str, drifted: object
) -> None:
    report = dict(temp.SEALED_REPORT_RECIPE)
    report[field] = drifted
    drift = temp._completed_recipe_drift(report)
    assert drift == {
        field: {"expected": temp.SEALED_REPORT_RECIPE[field], "actual": drifted}
    }


def test_objective_validation_binding_rejects_raw_compatibility_measure() -> None:
    objective = {
        "schema_version": "composite-validation-measure-v2",
        "objective_matched": True,
        "measure": (
            "authenticated_component_then_uniform_game_then_uniform_row_with_"
            "objective_weight_density"
        ),
        "component_sampling_ratios": dict(
            zip(temp.COMPONENT_IDS, temp.COMPONENT_RATIOS, strict=True)
        ),
        "policy_distillation_component_ids": list(temp.COMPONENT_IDS),
        "games": 1153,
        "samples": 262132,
    }
    report = {"metrics": [{"validation_objective_matched": objective}]}
    assert temp._authenticated_objective_validation(report) is True

    report["metrics"][-1] = {
        "validation": {"accuracy": 0.59, "value_loss": 0.62}
    }
    assert temp._authenticated_objective_validation(report) is False


def test_selection_requires_exact_h1_evidence_and_diagnostic_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = _command(tmp_path)
    descriptor = _descriptor(tmp_path)
    sentinel = _write(tmp_path / "sentinel.json", {})
    f7 = tmp_path / "f7.pt"
    f7.write_bytes(b"f7")
    checkpoint = tmp_path / "diagnostic.pt"
    checkpoint.write_bytes(b"winner")
    command.extend(
        [
            temp.base.ACK_FLAG,
            "sha256:" + "0" * 64,
            temp.base.ACK_FLAG,
            "sha256:" + "1" * 64,
            temp.base.ACK_FLAG,
            "sha256:" + "2" * 64,
        ]
    )
    command_doc = {
        "schema_version": temp.DIAGNOSTIC_COMMAND_SCHEMA,
        "argv": command,
        "argv_sha256": temp.base._digest(command),
    }
    command_path = _write(tmp_path / "command.json", command_doc)
    completion_path = _write(
        tmp_path / "completion.json",
        {
            "schema_version": temp.DIAGNOSTIC_COMPLETION_SCHEMA,
            "state": "complete",
            "checkpoint_sha256": temp.WINNING_DIAGNOSTIC_SHA256,
            "parent_checkpoint_sha256": temp.F7_SHA256,
            "descriptor_sha256": temp.base._file_sha(descriptor),
            "global_sample_dose": 4_194_304,
            "optimizer_steps": 1024,
            "world_size": 8,
            "batch_size_per_rank": 512,
            "command_sha256": temp.base._file_sha(command_path),
        },
    )
    evidence_path = _write(
        tmp_path / "evidence.json",
        {
            "candidate_checkpoint_sha256": temp.WINNING_DIAGNOSTIC_SHA256,
            "baseline_checkpoint_sha256": temp.F7_SHA256,
            "candidate_wins": 670,
            "baseline_wins": 530,
            "games_played": 1200,
            "complete_pairs": 600,
            "games_truncated": 0,
            "errors": [],
            "sprt": {"decision": "H1", "llr": 9.4, "upper_bound": 2.94},
            "pentanomial_sprt": {"decision": "H1", "llr": 11.1, "upper_bound": 2.94},
            "superiority_pentanomial_sprt": {
                "decision": "H1",
                "llr": 5.7,
                "upper_bound": 2.94,
            },
        },
    )
    original_ref = temp.base._ref

    def ref(path: Path) -> dict[str, str]:
        result = original_ref(path)
        if Path(path) == f7:
            result["sha256"] = temp.F7_SHA256
        elif Path(path) == checkpoint:
            result["sha256"] = temp.WINNING_DIAGNOSTIC_SHA256
        return result

    monkeypatch.setattr(temp.base, "_ref", ref)

    selected = temp._verify_diagnostic_selection(
        completion_path=completion_path,
        command_path=command_path,
        descriptor_path=descriptor,
        sentinel_path=sentinel,
        f7_path=f7,
        checkpoint_path=checkpoint,
        evidence_path=evidence_path,
    )
    assert selected["checkpoint"]["sha256"] == temp.WINNING_DIAGNOSTIC_SHA256

    # Rewriting the command receipt and updating its internal argv digest is
    # still forbidden: the completed launch binds the original receipt bytes.
    rewritten = json.loads(command_path.read_text(encoding="utf-8"))
    rewritten["argv"][rewritten["argv"].index("--lr") + 1] = "0.0001"
    rewritten["argv_sha256"] = temp.base._digest(rewritten["argv"])
    _write(command_path, rewritten)
    with pytest.raises(
        temp.TemperatureReplicationError,
        match="command-receipt file binding drift",
    ):
        temp._verify_diagnostic_selection(
            completion_path=completion_path,
            command_path=command_path,
            descriptor_path=descriptor,
            sentinel_path=sentinel,
            f7_path=f7,
            checkpoint_path=checkpoint,
            evidence_path=evidence_path,
        )
    _write(command_path, command_doc)

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["superiority_pentanomial_sprt"]["decision"] = "continue"
    _write(evidence_path, evidence)
    with pytest.raises(temp.TemperatureReplicationError, match="crossed-H1"):
        temp._verify_diagnostic_selection(
            completion_path=completion_path,
            command_path=command_path,
            descriptor_path=descriptor,
            sentinel_path=sentinel,
            f7_path=f7,
            checkpoint_path=checkpoint,
            evidence_path=evidence_path,
        )


def test_promotion_report_path_accepts_exact_replica_and_rejects_rng_drift(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"new production replica")
    candidate_sha = promotion._sha256(candidate)
    report_path = tmp_path / "report.json"
    report = {
        "arch": "entity_graph",
        "mask_hidden_info": True,
        "track": "2p_no_trade",
        "vps_to_win": 10,
        "world_size": 8,
        "batch_size": 512,
        "effective_global_batch_size": 4096,
        "epochs": 1,
        "max_steps": 1024,
        "steps_completed": 1024,
        "base_training_row_draws": 4_194_304,
        "optimizer": "adam",
        "resume_optimizer": False,
        "optimizer_restored": False,
        "lr": 3e-5,
        "lr_schedule": "flat",
        "lr_warmup_steps": 100,
        "weight_decay": 0.0,
        "value_lr_mult": 0.3,
        "action_module_lr_mult": 1.0,
        "soft_target_weight": 0.9,
        "value_loss_weight": 0.25,
        "value_target_lambda": 1.0,
        "forced_action_weight": 0.0,
        "forced_row_value_weight": 1.0,
        "winner_sample_weight": 1.0,
        "loser_sample_weight": 1.0,
        "training_rng_rank_offset": True,
        "ddp_shard_data": False,
        "checkpoint": str(candidate),
    }
    _write(report_path, report)

    verified = promotion._verify_training_report(
        report_path,
        contract={"science": {"learner_training_recipe": {"symmetry_augment": False}}},
        contract_sha256="unused-for-production-replica",
        candidate_path=candidate,
        candidate_sha256=candidate_sha,
        production_temperature_completion=True,
    )
    assert verified["base_training_row_draws"] == 4_194_304

    report["training_rng_rank_offset"] = False
    _write(report_path, report)
    with pytest.raises(promotion.PromotionError, match="training_rng_rank_offset"):
        promotion._verify_training_report(
            report_path,
            contract={
                "science": {"learner_training_recipe": {"symmetry_augment": False}}
            },
            contract_sha256="unused-for-production-replica",
            candidate_path=candidate,
            candidate_sha256=candidate_sha,
            production_temperature_completion=True,
        )
