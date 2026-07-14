from __future__ import annotations

import copy
from contextlib import nullcontext
import json
import subprocess
from pathlib import Path

import pytest

from tools import a1_aux_stage_executor as executor


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _geometry_rng_transaction(cuda_device: int = 0) -> dict:
    state = {
        "schema_version": "a1-aux-geometry-rng-state-v1",
        "python_random_sha256": _sha("1"),
        "numpy_generator_sha256": {"sampler": _sha("2")},
        "torch_cpu_sha256": _sha("3"),
        "cuda_device": cuda_device,
        "torch_cuda_sha256": _sha("4"),
    }
    state["state_sha256"] = executor.coordinator._digest(state)
    after_probe = copy.deepcopy(state)
    after_probe["torch_cuda_sha256"] = _sha("5")
    after_probe["state_sha256"] = executor.coordinator._digest(
        {
            key: value
            for key, value in after_probe.items()
            if key != "state_sha256"
        }
    )
    return {
        "schema_version": "a1-aux-geometry-rng-transaction-v1",
        "scope": "one_complete_ordered_five_batch_probe",
        "restore_frequency": "once_after_all_five_batches",
        "before": copy.deepcopy(state),
        "after_probe": after_probe,
        "after_restore": copy.deepcopy(state),
        "restored_exactly": True,
    }


def _published(tmp_path: Path, *, stage: str) -> dict:
    authority = {
        "schema_version": f"a1-aux-{stage.lower()}-executor-authority-v1",
        "stage": stage,
        "state_sha256": _sha("1"),
        "portable_science_identity": {
            "composite": {
                "descriptor_sha256": _sha("2"),
                "payload_inventory_sha256": _sha("3"),
            },
            "pointer_upgrade_authority": {
                "upgraded_initializer_sha256": _sha("4")
            },
        },
        "allocation": {"test": "allocation"},
    }
    if stage == "GEOMETRY":
        authority["warmup_terminal"] = {
            "result": {"warmed_checkpoint_sha256": _sha("5")}
        }
    name = (
        "15-warmup-executor-authority.json"
        if stage == "WARMUP"
        else "35-geometry-executor-authority.json"
    )
    path = (tmp_path / name).resolve()
    path.write_text(json.dumps(authority, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return {
        "schema_version": "a1-published-executor-authority-v1",
        "path": str(path),
        "file_sha256": executor._stable_read(path)[2],
        "authority": authority,
    }


def _stage_commitment(
    published: dict,
    binding: dict,
    checkpoint: Path,
    report: Path,
) -> dict:
    published.setdefault("file_sha256", _sha("e"))
    published["authority"].setdefault("state_sha256", _sha("f"))
    outputs = {
        "checkpoint": str(checkpoint.resolve(strict=False)),
        "report": str(report.resolve(strict=False)),
        "optimizer_sidecar": str(
            Path(str(checkpoint.resolve(strict=False)) + ".optimizer.pt")
        ),
    }
    return executor.coordinator._sealed(
        {
            "schema_version": "a1-stage-execution-commitment-v1",
            "stage": published["authority"]["stage"],
            "executor_authority_file_sha256": published["file_sha256"],
            "executor_authority_state_sha256": published["authority"][
                "state_sha256"
            ],
            "training_binding_sha256": executor._canonical_sha256(binding),
            "output_namespace": outputs,
            "output_namespace_sha256": executor._canonical_sha256(outputs),
        }
    )


@pytest.mark.parametrize("stage", ("WARMUP", "GEOMETRY"))
def test_stage_authority_is_path_sha_and_dag_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    published = _published(tmp_path, stage=stage)
    monkeypatch.setattr(
        executor.coordinator,
        "verify_published_executor_authority",
        lambda _path: published,
    )
    replay = executor.verify_stage_executor_authority(
        Path(published["path"]),
        expected_file_sha256=published["file_sha256"],
        expected_stage=stage,
    )
    assert replay == published


def test_stage_authority_refuses_wrong_digest_or_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = _published(tmp_path, stage="WARMUP")
    monkeypatch.setattr(
        executor.coordinator,
        "verify_published_executor_authority",
        lambda _path: published,
    )
    with pytest.raises(executor.StageExecutorError, match="digest drift"):
        executor.verify_stage_executor_authority(
            Path(published["path"]),
            expected_file_sha256=_sha("9"),
            expected_stage="WARMUP",
        )
    with pytest.raises(executor.StageExecutorError, match="replay/projection"):
        executor.verify_stage_executor_authority(
            Path(published["path"]),
            expected_file_sha256=published["file_sha256"],
            expected_stage="GEOMETRY",
        )


@pytest.mark.parametrize(
    ("stage", "initializer"),
    (("WARMUP", _sha("4")), ("GEOMETRY", _sha("5"))),
)
def test_stage_input_binding_selects_exact_initializer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    initializer: str,
) -> None:
    published = _published(tmp_path, stage=stage)
    monkeypatch.setattr(
        executor.coordinator, "verify_allocation", lambda value: value
    )
    binding = executor.bind_stage_inputs(
        published,
        descriptor_sha256=_sha("2"),
        payload_inventory_sha256=_sha("3"),
        initializer_sha256=initializer,
    )
    assert binding["stage"] == stage
    assert binding["optimizer_construction_authorized"] is (stage == "WARMUP")
    assert binding["gradient_probe_authorized"] is (stage == "GEOMETRY")

    with pytest.raises(executor.StageExecutorError, match="byte drift"):
        executor.bind_stage_inputs(
            published,
            descriptor_sha256=_sha("2"),
            payload_inventory_sha256=_sha("3"),
            initializer_sha256=_sha("8"),
        )


def _warmup_execution_fixture(tmp_path: Path):
    torch = pytest.importorskip("torch")
    prefixes = executor.coordinator.POINTER_TRAINABLE_PREFIXES
    names = [f"{prefix}weight" for prefix in prefixes]
    initializer = tmp_path / "initializer.pt"
    checkpoint = tmp_path / "warmed.pt"
    before_model = {name: torch.zeros(2) for name in names}
    before_model["trunk.weight"] = torch.arange(4, dtype=torch.float32)
    after_model = {name: torch.ones(2) for name in names}
    after_model["trunk.weight"] = before_model["trunk.weight"].clone()
    torch.save({"model": before_model}, initializer)
    torch.save({"model": after_model}, checkpoint)
    optimizer = Path(str(checkpoint) + ".optimizer.pt")
    torch.save({"state": "discard-me"}, optimizer)
    recipe = {
        "sample_dose": 524_288,
        "optimizer_steps": 128,
        "sample_order_sha256": _sha("6"),
    }
    authority = {
        "stage": "WARMUP",
        "experiment_id": _sha("7"),
        "portable_science_identity": {
            "warmup_recipe": recipe,
            "pointer_upgrade_authority": {
                "new_parameter_set_sha256": executor._canonical_sha256(sorted(names))
            },
        },
    }
    published = {"authority": authority}
    binding = {
        "schema_version": "a1-aux-stage-training-binding-v1",
        "stage": "WARMUP",
    }
    report = tmp_path / "warmup.json"
    report.write_text(
        json.dumps(
            {
                "a1_aux_stage_binding": binding,
                "a1_realized_aux_stage_sample_order": {
                    "sample_order_sha256": recipe["sample_order_sha256"],
                    "sample_dose": recipe["sample_dose"],
                },
                "steps_completed": 128,
                "base_training_row_draws": 524_288,
                "optimizer_restored": False,
                "public_award_feature_contract": "authoritative_v1",
                "require_only_trainable_prefixes": ",".join(prefixes),
            }
        ),
        encoding="utf-8",
    )
    return published, binding, initializer, checkpoint, report, optimizer


def test_warmup_completion_is_derived_from_checkpoint_report_and_discards_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published, binding, initializer, checkpoint, report, optimizer = (
        _warmup_execution_fixture(tmp_path)
    )
    captured = {}
    monkeypatch.setattr(executor, "_warmup_main_output_max_diff", lambda *_: 0.0)
    def complete(_root, _experiment, *, result):
        captured["result"] = result
        return {"status": "complete"}

    monkeypatch.setattr(executor.coordinator, "complete_warmup", complete)
    terminal = executor.complete_stage_from_outputs(
        coordinator_root=tmp_path,
        published=published,
        commitment=_stage_commitment(
            published, binding, checkpoint, report
        ),
        binding=binding,
        initializer=initializer,
        checkpoint=checkpoint,
        report=report,
    )
    assert terminal["status"] == "complete"
    assert captured["result"]["changed_parameter_set_sha256"] == published[
        "authority"
    ]["portable_science_identity"]["pointer_upgrade_authority"][
        "new_parameter_set_sha256"
    ]
    assert captured["result"]["inherited_parameters_bit_identical"] is True
    assert not optimizer.exists()
    assert Path(str(optimizer) + ".discarded").is_file()


def test_warmup_completion_refuses_inherited_tensor_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    published, binding, initializer, checkpoint, report, _optimizer = (
        _warmup_execution_fixture(tmp_path)
    )
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw["model"]["trunk.weight"][0] += 1
    torch.save(raw, checkpoint)
    monkeypatch.setattr(executor, "_warmup_main_output_max_diff", lambda *_: 0.0)
    with pytest.raises(executor.StageExecutorError, match="changed-parameter set"):
        executor.complete_stage_from_outputs(
            coordinator_root=tmp_path,
            published=published,
            commitment=_stage_commitment(
                published, binding, checkpoint, report
            ),
            binding=binding,
            initializer=initializer,
            checkpoint=checkpoint,
            report=report,
        )


def test_geometry_completion_derives_exact_nonmutating_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parameter_set = _sha("8")
    manifest = {
        "sampler_seed": 424_244,
        "probe_row_order_sha256": _sha("9"),
        "probe_batches": 5,
        "local_batch_size": 512,
    }
    batches = [
        {
            "batch_index": index,
            "shared_parameter_set_sha256": parameter_set,
            "main_squared_norm_decimal": "1",
            "unit_aux_squared_norm_decimal": "2",
            "gradient_dot_decimal": "0",
        }
        for index in range(5)
    ]
    binding = {
        "schema_version": "a1-aux-stage-training-binding-v1",
        "stage": "GEOMETRY",
        "initializer_sha256": _sha("5"),
        "probe_manifest": {"path": "/probe.json", "file_sha256": _sha("6")},
    }
    authority = {
        "stage": "GEOMETRY",
        "experiment_id": _sha("7"),
        "portable_science_identity": {
            "selector_rule": {
                "probe_batches": 5,
                "shared_parameter_set_sha256": parameter_set,
            }
        },
    }
    report = tmp_path / "geometry.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "a1-aux-gradient-geometry-child-report-v1",
                "stage_binding": binding,
                "probe_manifest": manifest,
                "model_state_before_sha256": _sha("a"),
                "model_state_after_sha256": _sha("a"),
                "per_batch_geometry": batches,
                "rng_transactions_by_rank": [
                    _geometry_rng_transaction(rank) for rank in range(8)
                ],
                "optimizer_constructed": False,
                "optimizer_steps": 0,
                "persistent_state_mutated": False,
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def complete(_root, _experiment, *, evidence):
        captured["evidence"] = evidence
        return {"status": "complete"}

    monkeypatch.setattr(executor.coordinator, "complete_geometry", complete)
    published = {"authority": authority}
    terminal = executor.complete_stage_from_outputs(
        coordinator_root=tmp_path,
        published=published,
        commitment=_stage_commitment(
            published, binding, tmp_path / "unused-output.pt", report
        ),
        binding=binding,
        initializer=tmp_path / "unused.pt",
        checkpoint=tmp_path / "unused-output.pt",
        report=report,
    )
    assert terminal == {"status": "complete"}
    assert captured["evidence"]["optimizer_steps"] == 0
    assert captured["evidence"]["persistent_state_mutated"] is False
    assert captured["evidence"]["batch_shared_parameter_set_sha256"] == [
        parameter_set
    ] * 5


def test_execute_stage_invokes_exact_command_then_terminalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = {
        "path": str(tmp_path / "authority.json"),
        "file_sha256": _sha("1"),
        "authority": {
            "stage": "GEOMETRY",
            "experiment_id": _sha("2"),
            "state_sha256": _sha("3"),
            "allocation": {"physical_gpu_indices": list(range(8))},
        }
    }
    binding = {"stage": "GEOMETRY"}
    monkeypatch.setattr(
        executor,
        "build_stage_train_command",
        lambda *_args, **_kwargs: (["python", "train_bc.py"], binding, {}),
    )
    monkeypatch.setattr(
        executor.coordinator,
        "verify_allocation",
        lambda value: value,
    )
    monkeypatch.setattr(
        executor,
        "complete_stage_from_outputs",
        lambda **_kwargs: {"status": "complete"},
    )
    monkeypatch.setattr(
        executor,
        "_fresh_output_namespace",
        lambda **_kwargs: {
            "checkpoint": str(tmp_path / "checkpoint.pt"),
            "report": str(tmp_path / "report.json"),
            "optimizer_sidecar": str(tmp_path / "checkpoint.pt.optimizer.pt"),
        },
    )
    monkeypatch.setattr(executor, "_verify_live_allocation", lambda *_a, **_k: {})
    monkeypatch.setattr(executor.one_dose, "_raise_nofile_limit", lambda: None)
    monkeypatch.setattr(
        executor.coordinator,
        "commit_stage_execution",
        lambda *_a, **_k: {"status": "committed"},
    )
    monkeypatch.setattr(executor.coordinator, "_artifact", lambda *_a, **_k: None)
    observed = {}

    def runner(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0)

    terminal = executor.execute_stage(
        coordinator_root=tmp_path,
        published=published,
        python=Path("/usr/bin/python3"),
        descriptor=tmp_path / "descriptor.json",
        initializer=tmp_path / "initializer.pt",
        checkpoint=tmp_path / "checkpoint.pt",
        report=tmp_path / "report.json",
        runner=runner,
        gpu_probe=lambda _gpu: "NVIDIA B200",
        gpu_lock=lambda _gpu: nullcontext(),
    )
    assert terminal == {"status": "complete"}
    assert observed["command"] == ["python", "train_bc.py"]
    assert observed["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert observed["check"] is True
