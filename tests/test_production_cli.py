from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest

from catan_zero import production_cli as cli
from catan_zero import production_contracts as contracts
from catan_zero.production_contracts import (
    NATIVE_REQUIRED_CAPABILITIES,
    canonical_json_sha256,
    production_status,
    validate_pipeline_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_job(tmp_path: Path, pipeline: str = "generate", **updates: object) -> Path:
    run_id = "production-canary-001"
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    payload: dict[str, object] = {
        "schema_version": cli.JOB_SCHEMA,
        "pipeline": pipeline,
        "run_id": run_id,
        "run_dir": str(tmp_path / run_id),
    }
    if pipeline == "generate":
        payload.update(
            checkpoint=str(checkpoint),
            games=8,
            base_seed=2026071600,
            claim_label="production_canary",
            workers=2,
            resume=False,
            gpu=3,
        )
    elif pipeline == "train":
        recipe = str(updates.get("recipe", "a1-current-35m-b200"))
        data = tmp_path / "composite.json"
        data.write_text("{}", encoding="utf-8")
        payload.update(data=str(data), recipe=recipe)
        if recipe == "a1-parent-update-35m-b200":
            parent = tmp_path / "parent.pt"
            parent.write_bytes(b"parent-v1")
            upgrade = tmp_path / "architecture-upgrade.receipt.json"
            upgrade.write_text("{}", encoding="utf-8")
            payload.update(
                init_checkpoint=str(checkpoint),
                parent_checkpoint=str(parent),
                architecture_upgrade_receipt=str(upgrade),
            )
        else:
            lock = tmp_path / "reviewed-lock.json"
            lock.write_text("{}", encoding="utf-8")
            build_receipt = tmp_path / "composite-build-receipt.json"
            build_receipt.write_text("{}", encoding="utf-8")
            payload.update(
                lock=str(lock),
                composite_build_receipt=str(build_receipt),
                plan_receipt=str(tmp_path / "authenticated-plan.json"),
            )
    elif pipeline == "evaluate":
        champion = tmp_path / "champion.pt"
        champion.write_bytes(b"champion-v1")
        payload.update(
            candidate=str(checkpoint),
            champion=str(champion),
            pairs=200,
            workers=8,
            devices=["cuda:0", "cuda:1"],
            threads_per_worker=6,
            base_seed=2026071600,
        )
    payload.update(updates)
    path = tmp_path / f"{pipeline}.job.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _mock_exact_runtime(
    plan: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    *,
    device_names: list[str],
) -> None:
    runtime = json.loads(
        (ROOT / "configs/runtime/a1_production_runtime.json").read_text(
            encoding="utf-8"
        )
    )
    monkeypatch.setattr(
        cli.platform, "python_version", lambda: runtime["python_version"]
    )
    monkeypatch.setattr(
        cli,
        "_package_version",
        lambda distribution: {
            "catanatron-rs": runtime["catanatron_rs_version"],
            "numpy": runtime["numpy_version"],
            "networkx": runtime["networkx_version"],
            "gymnasium": runtime["gymnasium_version"],
            "zstandard": runtime["zstandard_version"],
            "scipy": runtime["scipy_version"],
            "whr": runtime["whr_version"],
            "torch": runtime["torch_version"],
        }[distribution],
    )
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda=runtime["torch_cuda_version"]),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: len(device_names),
            get_device_name=lambda index: device_names[index],
        ),
    )
    monkeypatch.setitem(cli.sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        cli,
        "_nvidia_driver_identity",
        lambda: {"versions": [runtime["nvidia_driver_version"]], "error": None},
    )
    monkeypatch.setattr(
        cli,
        "_native_runtime_identity",
        lambda: {
            "wheel_sha256": runtime["catanatron_rs_wheel_sha256"],
            "extension_sha256": runtime["catanatron_rs_extension_sha256"],
            "capabilities": sorted(NATIVE_REQUIRED_CAPABILITIES),
        },
    )
    repository = plan["repository"]
    assert isinstance(repository, dict)
    clean_repository = {
        "commit": repository["commit"],
        "tracked_changes": [],
        "clean": True,
    }
    plan["repository"] = clean_repository
    monkeypatch.setattr(cli, "_git_identity", lambda _root: clean_repository)


def test_status_exposes_v5_parent_commissioning_as_invalidated() -> None:
    status = production_status(ROOT)

    assert status["supported_operator_interface"] == "catan-zero"
    assert status["pipelines"]["generate"]["authorized"] is True
    assert status["pipelines"]["evaluate"]["authorized"] is True
    train = status["pipelines"]["train"]
    assert train["authorized"] is False
    assert train["reason"] == "recipe_specific_authorization"
    assert train["recipes"]["a1-current-35m-b200"]["authorized"] is False
    parent = train["recipes"]["a1-parent-update-35m-b200"]
    assert parent["authorized"] is False
    assert parent["status"] == "blocked"
    assert parent["reason"] == (
        "commissioned_b12_invalidated_by_adapter_v5_information_aliasing"
    )
    assert parent["unresolved_requirements"] == [
        "materialize and authenticate a fresh "
        "rust_entity_adapter_v6_exact_actor_resources_initial_road_two_hop "
        "training composite",
        "recommission B12 topology, opening-policy, and value no-regression "
        "gates on adapter v6",
    ]
    assert status["pipelines"]["ppo"]["authorized"] is False


def test_all_canonical_config_and_guard_identities_are_exact() -> None:
    identities = [
        validate_pipeline_contract(ROOT, "generate"),
        validate_pipeline_contract(ROOT, "evaluate"),
        validate_pipeline_contract(ROOT, "train", "a1-current-35m-b200"),
        validate_pipeline_contract(ROOT, "train", "a1-parent-update-35m-b200"),
    ]
    for identity in identities:
        payload = json.loads(Path(identity["config"]).read_text(encoding="utf-8"))
        assert identity["config_sha256"] == canonical_json_sha256(payload)
    generation_entry = contracts.production_recipes("generate")[0]
    assert identities[0]["guard"] == generation_entry["guard"]
    assert identities[0]["guard_sha256"] == generation_entry["guard_sha256"]
    assert identities[0]["required_accelerator_model"] == "NVIDIA H100"
    assert identities[1]["required_accelerator_model"] == "NVIDIA H100"
    assert identities[2]["required_accelerator_model"] == "NVIDIA B200"
    assert identities[3]["required_accelerator_model"] == "NVIDIA B200"


def test_generate_plan_has_one_canonical_command_and_bound_inputs(
    tmp_path: Path,
) -> None:
    job_path = _write_job(tmp_path)
    plan = cli.build_plan(job_path)

    assert plan["schema_version"] == cli.PLAN_SCHEMA
    assert plan["readiness"]["authorized"] is True
    assert plan["environment"] == {"CUDA_VISIBLE_DEVICES": "3"}
    assert plan["command"][0] == cli.sys.executable
    assert plan["command"][1] == str((ROOT / "tools/generate.py").resolve())
    assert "--config" in plan["command"]
    assert "--guard" in plan["command"]
    assert plan["inputs"]["checkpoint"]["sha256"] == cli._file_sha256(  # noqa: SLF001
        Path(plan["inputs"]["checkpoint"]["path"])
    )
    unhashed = dict(plan)
    plan_sha256 = unhashed.pop("plan_sha256")
    assert plan_sha256 == canonical_json_sha256(unhashed)


@pytest.mark.parametrize("pipeline", ("train", "evaluate"))
def test_other_pipelines_resolve_through_compact_launchers(
    pipeline: str, tmp_path: Path
) -> None:
    plan = cli.build_plan(_write_job(tmp_path, pipeline))

    launcher = "train.py" if pipeline == "train" else "evaluate.py"
    if pipeline == "train":
        launcher = "a1_scratch_train.py"
    assert str((ROOT / "tools" / launcher).resolve()) in plan["command"]
    assert "train_bc.py" not in plan["command"]
    assert "gumbel_search_cross_net_h2h.py" not in plan["command"]
    if pipeline == "train":
        assert plan["readiness"]["authorized"] is False
        assert plan["command"][-1] == "--go"
        assert plan["prepare_command"] == plan["command"][:-1]
        assert "--lock" in plan["command"]


def test_cataloged_parent_update_uses_exact_recipe_and_parent_but_is_quarantined(
    tmp_path: Path,
) -> None:
    plan = cli.build_plan(
        _write_job(tmp_path, "train", recipe="a1-parent-update-35m-b200")
    )

    assert plan["readiness"]["authorized"] is False
    assert plan["readiness"]["reason"] == (
        "commissioned_b12_invalidated_by_adapter_v5_information_aliasing"
    )
    assert plan["contract"]["recipe"] == "a1-parent-update-35m-b200"
    assert plan["contract"]["config_sha256"] == (
        "757b1376bf1903841f6f26f60f361f81eb59a38b682a5468e14cafcce2d38131"
    )
    assert str((ROOT / "tools/train.py").resolve()) in plan["command"]
    assert "--init-checkpoint" in plan["command"]
    assert "--parent-checkpoint" in plan["command"]
    assert "--architecture-upgrade-receipt" in plan["command"]
    assert "--nproc-per-node=8" in plan["command"]
    assert plan["prepare_command"] is None


def test_training_science_admission_cannot_authorize_recipe_digest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = ROOT / contracts.TRAINING_SCIENCE_ADMISSION
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["recipes"]["a1-parent-update-35m-b200"]["recipe_canonical_sha256"] = (
        "0" * 64
    )
    drifted = tmp_path / "training-science-admission.json"
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(contracts, "TRAINING_SCIENCE_ADMISSION", drifted)

    with pytest.raises(
        contracts.ProductionContractError,
        match="does not bind the exact recipe",
    ):
        contracts.pipeline_readiness(ROOT, "train", "a1-parent-update-35m-b200")


def test_training_science_admission_quarantines_v5_and_requires_fresh_v6() -> None:
    payload = json.loads(
        (ROOT / contracts.TRAINING_SCIENCE_ADMISSION).read_text(encoding="utf-8")
    )
    expected_adapter = (
        "rust_entity_adapter_v6_exact_actor_resources_initial_road_two_hop"
    )

    for admission in payload["recipes"].values():
        assert admission["authorized"] is False
        assert any(
            expected_adapter in requirement
            for requirement in admission["unresolved_requirements"]
        )
        observations = admission["observations"]
        assert observations["required_fresh_learner_adapter"] == expected_adapter
        assert observations["adapter_v5_resource_quarantine"] == {
            "status": "quarantined_actor_resource_clipping_contradiction",
            "contradictory_rows": 4061,
            "contradictory_games": 1271,
            "full_search_rows": 1868,
            "saturation_risk_rows": 7440,
            "training_admission": False,
        }


def test_parent_update_requires_receipt_only_for_changed_initializer(
    tmp_path: Path,
) -> None:
    changed = _write_job(tmp_path, "train", recipe="a1-parent-update-35m-b200")
    payload = json.loads(changed.read_text(encoding="utf-8"))
    payload.pop("architecture_upgrade_receipt")
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(cli.ProductionCLIError, match="receipt is required"):
        cli.build_plan(changed)

    payload["init_checkpoint"] = payload["parent_checkpoint"]
    changed.write_text(json.dumps(payload), encoding="utf-8")
    plan = cli.build_plan(changed)
    assert "--architecture-upgrade-receipt" not in plan["command"]


def test_production_job_rejects_unknown_keys_and_relative_paths(tmp_path: Path) -> None:
    unknown = _write_job(tmp_path, unexpected_knob=True)
    with pytest.raises(cli.ProductionCLIError, match="unknown=.*unexpected_knob"):
        cli.load_job(unknown)

    relative = _write_job(tmp_path, checkpoint="candidate.pt")
    with pytest.raises(cli.ProductionCLIError, match="checkpoint must be absolute"):
        cli.load_job(relative)


@pytest.mark.parametrize(
    ("devices", "message"),
    (
        (["cpu"], "exact cuda:N form"),
        ([""], "exact cuda:N form"),
        (["cuda:0", "cuda:0"], "unique CUDA devices"),
    ),
)
def test_evaluation_job_rejects_ambiguous_cuda_placement(
    tmp_path: Path, devices: list[str], message: str
) -> None:
    job = _write_job(tmp_path, "evaluate", devices=devices)

    with pytest.raises(cli.ProductionCLIError, match=message):
        cli.load_job(job)


def test_doctor_refuses_out_of_range_device_before_stage_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = cli.build_plan(
        _write_job(tmp_path, "evaluate", devices=["cuda:0", "cuda:8"])
    )
    _mock_exact_runtime(
        plan,
        monkeypatch,
        device_names=["NVIDIA H100 80GB HBM3"] * 8,
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("stage subprocess must not start"),
    )

    with pytest.raises(cli.ProductionCLIError, match="outside visible device count"):
        cli.execute(plan)
    assert not Path(plan["run_receipt"]).exists()


def test_doctor_attests_requested_evaluation_devices_are_h100(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = cli.build_plan(
        _write_job(tmp_path, "evaluate", devices=["cuda:0", "cuda:1"])
    )
    _mock_exact_runtime(
        plan,
        monkeypatch,
        device_names=["NVIDIA H100 80GB HBM3", "NVIDIA B200"],
    )

    result = cli.doctor(plan)

    assert result["ok"] is False
    assert result["runtime_actual"]["requested_cuda_device_indices"] == [0, 1]
    assert any(
        "production placement requires only NVIDIA H100" in error
        for error in result["errors"]
    )


def test_ppo_gets_a_typed_refusal(tmp_path: Path) -> None:
    job = _write_job(tmp_path, "ppo")

    with pytest.raises(cli.ProductionCLIError, match="PPO is not a commissioned"):
        cli.build_plan(job)


def test_plan_artifact_drift_is_detected(tmp_path: Path) -> None:
    plan = cli.build_plan(_write_job(tmp_path))
    Path(plan["inputs"]["checkpoint"]["path"]).write_bytes(b"replaced")

    assert any(
        "input checkpoint drift" in error for error in cli._verify_plan_artifacts(plan)
    )  # noqa: SLF001


class _FakeNativeDistribution:
    def __init__(self, root: Path, files: list[Path]) -> None:
        self.root = root
        self.files = files

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return json.dumps(
            {
                "archive_info": {
                    "hashes": {"sha256": "a" * 64},
                }
            }
        )

    def locate_file(self, record: Path) -> Path:
        return self.root / record


def _fake_native_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    records: list[Path] | None = None,
    loaded_path: Path | None = None,
) -> Path:
    relative = Path("catanatron_rs/catanatron_rs.cpython-311-x86_64-linux-gnu.so")
    extension = tmp_path / relative
    extension.parent.mkdir(parents=True, exist_ok=True)
    extension.write_bytes(b"sealed-native-extension")
    distribution = _FakeNativeDistribution(tmp_path, records or [relative])
    monkeypatch.setattr(cli.metadata, "distribution", lambda _name: distribution)
    package = SimpleNamespace(
        gumbel_search_capabilities=lambda: sorted(NATIVE_REQUIRED_CAPABILITIES)
    )
    native_module = SimpleNamespace(__file__=str(loaded_path or extension))
    monkeypatch.setitem(cli.sys.modules, "catanatron_rs", package)
    monkeypatch.setitem(
        cli.sys.modules, "catanatron_rs.catanatron_rs", native_module
    )
    return extension


def test_native_runtime_hashes_the_exact_loaded_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extension = _fake_native_runtime(tmp_path, monkeypatch)

    exact = cli._native_runtime_identity()  # noqa: SLF001
    assert exact["wheel_sha256"] == "a" * 64
    assert exact["extension_path"] == str(extension)
    assert exact["extension_sha256"] == cli._file_sha256(extension)  # noqa: SLF001
    assert "error" not in exact

    extension.write_bytes(b"tampered-native-extension")
    drifted = cli._native_runtime_identity()  # noqa: SLF001
    assert drifted["wheel_sha256"] == exact["wheel_sha256"]
    assert drifted["capabilities"] == exact["capabilities"]
    assert drifted["extension_sha256"] != exact["extension_sha256"]


def test_native_runtime_refuses_multiple_or_symlinked_extensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    second = Path("catanatron_rs/other.so")
    _fake_native_runtime(
        tmp_path,
        monkeypatch,
        records=[
            Path("catanatron_rs/catanatron_rs.cpython-311-x86_64-linux-gnu.so"),
            second,
        ],
    )
    (tmp_path / second).write_bytes(b"other")
    multiple = cli._native_runtime_identity()  # noqa: SLF001
    assert "exactly one native extension; found=2" in multiple["error"]
    assert multiple["extension_sha256"] is None

    target = tmp_path / "native-target.so"
    target.write_bytes(b"target")
    relative = Path("catanatron_rs/catanatron_rs.cpython-311-x86_64-linux-gnu.so")
    extension = tmp_path / relative
    extension.unlink()
    extension.symlink_to(target)
    distribution = _FakeNativeDistribution(tmp_path, [relative])
    monkeypatch.setattr(cli.metadata, "distribution", lambda _name: distribution)
    cli.sys.modules["catanatron_rs.catanatron_rs"].__file__ = str(extension)

    symlinked = cli._native_runtime_identity()  # noqa: SLF001
    assert "canonical regular non-symlink file" in symlinked["error"]
    assert symlinked["extension_sha256"] is None


def test_native_runtime_refuses_loaded_extension_path_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "other-loaded-extension.so"
    other.write_bytes(b"other")
    _fake_native_runtime(tmp_path, monkeypatch, loaded_path=other)

    identity = cli._native_runtime_identity()  # noqa: SLF001

    assert "loaded native extension path drift" in identity["error"]
    assert identity["extension_sha256"] is None


def test_doctor_accepts_only_exact_runtime_and_clean_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = cli.build_plan(_write_job(tmp_path))
    runtime = json.loads(
        (ROOT / "configs/runtime/a1_production_runtime.json").read_text(
            encoding="utf-8"
        )
    )
    monkeypatch.setattr(
        cli.platform, "python_version", lambda: runtime["python_version"]
    )
    monkeypatch.setattr(
        cli,
        "_package_version",
        lambda distribution: {
            "catanatron-rs": runtime["catanatron_rs_version"],
            "numpy": runtime["numpy_version"],
            "networkx": runtime["networkx_version"],
            "gymnasium": runtime["gymnasium_version"],
            "zstandard": runtime["zstandard_version"],
            "scipy": runtime["scipy_version"],
            "whr": runtime["whr_version"],
            "torch": runtime["torch_version"],
        }[distribution],
    )
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda=runtime["torch_cuda_version"]),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 8,
            get_device_name=lambda _index: "NVIDIA H100 80GB HBM3",
        ),
    )
    monkeypatch.setitem(cli.sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        cli,
        "_nvidia_driver_identity",
        lambda: {"versions": [runtime["nvidia_driver_version"]], "error": None},
    )
    monkeypatch.setattr(
        cli,
        "_native_runtime_identity",
        lambda: {
            "wheel_sha256": runtime["catanatron_rs_wheel_sha256"],
            "extension_sha256": runtime["catanatron_rs_extension_sha256"],
            "capabilities": sorted(NATIVE_REQUIRED_CAPABILITIES),
        },
    )
    clean_repository = {
        "commit": plan["repository"]["commit"],
        "tracked_changes": [],
        "clean": True,
    }
    plan["repository"] = clean_repository
    monkeypatch.setattr(cli, "_git_identity", lambda _root: clean_repository)

    result = cli.doctor(plan)

    assert result["ok"] is True
    assert result["errors"] == []

    monkeypatch.setattr(
        cli,
        "_native_runtime_identity",
        lambda: {
            "wheel_sha256": runtime["catanatron_rs_wheel_sha256"],
            "extension_sha256": "0" * 64,
            "capabilities": sorted(NATIVE_REQUIRED_CAPABILITIES),
        },
    )
    drifted = cli.doctor(plan)
    assert drifted["ok"] is False
    assert any("native extension drift" in error for error in drifted["errors"])


def test_doctor_refuses_b200_training_recipe_on_h100(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = cli.build_plan(
        _write_job(tmp_path, "train", recipe="a1-parent-update-35m-b200")
    )
    runtime = json.loads(
        (ROOT / "configs/runtime/a1_production_runtime.json").read_text(
            encoding="utf-8"
        )
    )
    monkeypatch.setattr(
        cli.platform, "python_version", lambda: runtime["python_version"]
    )
    monkeypatch.setattr(
        cli,
        "_package_version",
        lambda distribution: {
            "catanatron-rs": runtime["catanatron_rs_version"],
            "numpy": runtime["numpy_version"],
            "networkx": runtime["networkx_version"],
            "gymnasium": runtime["gymnasium_version"],
            "zstandard": runtime["zstandard_version"],
            "scipy": runtime["scipy_version"],
            "whr": runtime["whr_version"],
            "torch": runtime["torch_version"],
        }[distribution],
    )
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda=runtime["torch_cuda_version"]),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 8,
            get_device_name=lambda _index: "NVIDIA H100 80GB HBM3",
        ),
    )
    monkeypatch.setitem(cli.sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        cli,
        "_nvidia_driver_identity",
        lambda: {"versions": [runtime["nvidia_driver_version"]], "error": None},
    )
    monkeypatch.setattr(
        cli,
        "_native_runtime_identity",
        lambda: {
            "wheel_sha256": runtime["catanatron_rs_wheel_sha256"],
            "extension_sha256": runtime["catanatron_rs_extension_sha256"],
            "capabilities": sorted(NATIVE_REQUIRED_CAPABILITIES),
        },
    )
    clean_repository = {
        "commit": plan["repository"]["commit"],
        "tracked_changes": [],
        "clean": True,
    }
    plan["repository"] = clean_repository
    monkeypatch.setattr(cli, "_git_identity", lambda _root: clean_repository)

    result = cli.doctor(plan)

    assert result["ok"] is False
    assert (
        result["runtime_actual"]["cuda_device_names"] == ["NVIDIA H100 80GB HBM3"] * 8
    )
    assert any(
        "production placement requires only NVIDIA B200" in error
        for error in result["errors"]
    )


def test_doctor_refuses_blocked_training_even_with_exact_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = cli.build_plan(_write_job(tmp_path, "train"))
    monkeypatch.setattr(cli, "_package_version", lambda _name: None)
    monkeypatch.setattr(
        cli, "_nvidia_driver_identity", lambda: {"versions": [], "error": "test"}
    )
    monkeypatch.setattr(
        cli,
        "_native_runtime_identity",
        lambda: {"wheel_sha256": None, "capabilities": []},
    )

    result = cli.doctor(plan)

    assert result["ok"] is False
    assert (
        "pipeline is blocked: scratch_training_signal_contract_unresolved"
        in result["errors"]
    )
    assert any("authenticated plan receipt" in error for error in result["errors"])


def test_execute_refuses_before_receipt_or_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = cli.build_plan(_write_job(tmp_path))
    monkeypatch.setattr(
        cli,
        "doctor",
        lambda _plan: {"ok": False, "errors": ["deliberate refusal"]},
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not start"),
    )

    with pytest.raises(cli.ProductionCLIError, match="deliberate refusal"):
        cli.execute(plan)
    assert not Path(plan["run_receipt"]).exists()


def _write_admissible_outputs(plan: dict[str, object]) -> set[str]:
    job = plan["job"]
    inputs = plan["inputs"]
    assert isinstance(job, dict)
    assert isinstance(inputs, dict)
    run_dir = Path(str(job["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    if job["pipeline"] == "generate":
        shard = run_dir / "worker_000" / "shard_000000.npz"
        shard.parent.mkdir()
        shard.write_bytes(b"rows")
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "out_dir": str(run_dir),
                    "games_requested": job["games"],
                    "games_completed": job["games"],
                    "games_failed": 0,
                    "rows": 8,
                    "errors": [],
                    "shards": [str(shard)],
                }
            ),
            encoding="utf-8",
        )
        return {"generation_manifest", "generation_shard_000000"}
    if job["pipeline"] == "train":
        candidate = run_dir / "candidate.pt"
        report = run_dir / "train.report.json"
        candidate.write_bytes(b"trained-candidate")
        report.write_text(
            json.dumps(
                {
                    "checkpoint": str(candidate),
                    "steps_completed": 3,
                    "epochs": 1,
                }
            ),
            encoding="utf-8",
        )
        expected = {"training_candidate", "training_report"}
        if job["recipe"] == "a1-current-35m-b200":
            execution = {
                "schema_version": "a1-coherent-scratch-training-execution-v2",
                "status": "completed",
                "returncode": 0,
                "outputs": {
                    "terminal_checkpoint": {
                        "path": str(candidate),
                        "file_sha256": "sha256:" + cli._file_sha256(candidate),  # noqa: SLF001
                    },
                    "training_report": {
                        "path": str(report),
                        "file_sha256": "sha256:" + cli._file_sha256(report),  # noqa: SLF001
                    },
                },
            }
            execution["receipt_sha256"] = (
                "sha256:" + canonical_json_sha256(execution)
            )
            (run_dir / "scratch.execution.json").write_text(
                json.dumps(execution), encoding="utf-8"
            )
            expected.add("scratch_execution_receipt")
        return expected
    candidate = inputs["candidate"]
    champion = inputs["champion"]
    assert isinstance(candidate, dict)
    assert isinstance(champion, dict)
    games = [{"pair_id": index // 2} for index in range(int(job["pairs"]) * 2)]
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "errors": [],
                "games": games,
                "pairs_requested": job["pairs"],
                "games_played": len(games),
                "candidate_checkpoint": job["candidate"],
                "candidate_checkpoint_sha256": candidate["sha256"],
                "baseline_checkpoint": job["champion"],
                "baseline_checkpoint_sha256": champion["sha256"],
            }
        ),
        encoding="utf-8",
    )
    return {"evaluation_report"}


@pytest.mark.parametrize(
    ("pipeline", "recipe", "expected_error"),
    (
        ("generate", None, "generation manifest"),
        ("train", "a1-parent-update-35m-b200", "training report"),
        ("train", "a1-current-35m-b200", "scratch execution receipt"),
        ("evaluate", None, "evaluation report must be a non-empty JSON object"),
    ),
)
def test_zero_exit_without_required_outputs_is_failed_and_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline: str,
    recipe: str | None,
    expected_error: str,
) -> None:
    updates = {} if recipe is None else {"recipe": recipe}
    plan = cli.build_plan(_write_job(tmp_path, pipeline, **updates))
    monkeypatch.setattr(cli, "doctor", lambda _plan: {"ok": True, "errors": []})

    def zero_exit(_command, **_kwargs):
        run_dir = Path(plan["job"]["run_dir"])
        if pipeline == "train":
            candidate = run_dir / "candidate.pt"
            candidate.write_bytes(b"candidate")
            if recipe == "a1-current-35m-b200":
                (run_dir / "train.report.json").write_text(
                    json.dumps(
                        {
                            "checkpoint": str(candidate),
                            "steps_completed": 1,
                        }
                    ),
                    encoding="utf-8",
                )
        elif pipeline == "evaluate":
            (run_dir / "evaluation.json").write_text("[]", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", zero_exit)

    assert cli.execute(plan) == 1
    receipt = json.loads(Path(plan["run_receipt"]).read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["command_returncode"] == 0
    assert receipt["returncode"] == 1
    assert expected_error in receipt["output_admission_error"]
    assert "outputs" not in receipt


@pytest.mark.parametrize(
    ("pipeline", "recipe"),
    (
        ("generate", None),
        ("train", "a1-parent-update-35m-b200"),
        ("train", "a1-current-35m-b200"),
        ("evaluate", None),
    ),
)
def test_success_requires_and_hashes_pipeline_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline: str,
    recipe: str | None,
) -> None:
    updates = {} if recipe is None else {"recipe": recipe}
    plan = cli.build_plan(_write_job(tmp_path, pipeline, **updates))
    monkeypatch.setattr(cli, "doctor", lambda _plan: {"ok": True, "errors": []})
    expected_outputs: set[str] = set()

    def zero_exit(_command, **_kwargs):
        expected_outputs.update(_write_admissible_outputs(plan))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", zero_exit)

    assert cli.execute(plan) == 0
    receipt = json.loads(Path(plan["run_receipt"]).read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["command_returncode"] == 0
    assert receipt["returncode"] == 0
    assert set(receipt["outputs"]) == expected_outputs
    for output in receipt["outputs"].values():
        path = Path(output["path"])
        assert output["sha256"] == cli._file_sha256(path)  # noqa: SLF001
        assert output["size_bytes"] == path.stat().st_size


def test_zero_exit_with_input_mutation_is_failed_before_output_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = cli.build_plan(_write_job(tmp_path, "generate"))
    monkeypatch.setattr(cli, "doctor", lambda _plan: {"ok": True, "errors": []})

    def mutate_input_then_exit(_command, **_kwargs):
        Path(plan["inputs"]["checkpoint"]["path"]).write_bytes(b"mutated")
        _write_admissible_outputs(plan)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", mutate_input_then_exit)

    assert cli.execute(plan) == 1
    receipt = json.loads(Path(plan["run_receipt"]).read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["command_returncode"] == 0
    assert receipt["returncode"] == 1
    assert "production inputs changed during execution" in receipt[
        "output_admission_error"
    ]
    assert "input checkpoint drift" in receipt["output_admission_error"]
    assert "outputs" not in receipt


def test_zero_exit_generation_refuses_a_missing_manifest_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = cli.build_plan(_write_job(tmp_path))
    monkeypatch.setattr(cli, "doctor", lambda _plan: {"ok": True, "errors": []})

    def zero_exit(_command, **_kwargs):
        run_dir = Path(plan["job"]["run_dir"])
        missing = run_dir / "worker_000" / "missing.npz"
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "out_dir": str(run_dir),
                    "games_requested": plan["job"]["games"],
                    "games_completed": plan["job"]["games"],
                    "games_failed": 0,
                    "rows": 8,
                    "errors": [],
                    "shards": [str(missing)],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", zero_exit)

    assert cli.execute(plan) == 1
    receipt = json.loads(Path(plan["run_receipt"]).read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert "generation shard 0" in receipt["output_admission_error"]
    assert "outputs" not in receipt


def test_run_claim_is_exclusive_across_processes(tmp_path: Path) -> None:
    plan = cli.build_plan(_write_job(tmp_path))
    context = multiprocessing.get_context("fork")
    claimed = context.Event()
    release = context.Event()

    def hold_claim() -> None:
        with cli._exclusive_run_claim(  # noqa: SLF001
            Path(plan["run_receipt"]), str(plan["plan_sha256"])
        ):
            claimed.set()
            release.wait(timeout=5)

    worker = context.Process(target=hold_claim)
    worker.start()
    assert claimed.wait(timeout=5)
    try:
        with pytest.raises(cli.ProductionCLIError, match="already claimed"):
            with cli._exclusive_run_claim(  # noqa: SLF001
                Path(plan["run_receipt"]), str(plan["plan_sha256"])
            ):
                pytest.fail("a second process acquired the same run claim")
    finally:
        release.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert worker.exitcode == 0
    with cli._exclusive_run_claim(  # noqa: SLF001
        Path(plan["run_receipt"]), str(plan["plan_sha256"])
    ):
        pass


def test_resume_refuses_changed_attempt_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = cli.build_plan(_write_job(tmp_path))
    receipt_path = Path(first["run_receipt"])
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": cli.RUN_RECEIPT_SCHEMA,
                "status": "failed",
                "plan": first,
            }
        ),
        encoding="utf-8",
    )
    run_dir = Path(first["job"]["run_dir"])
    run_dir.mkdir()
    (run_dir / "partial").write_text("partial", encoding="utf-8")
    resumed_job = _write_job(tmp_path, resume=True, games=16)
    resumed = cli.build_plan(resumed_job)
    monkeypatch.setattr(cli, "doctor", lambda _plan: {"ok": True, "errors": []})

    with pytest.raises(cli.ProductionCLIError, match="differ from the failed attempt"):
        cli.execute(resumed)


def test_prepare_runs_only_authenticated_scratch_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = cli.build_plan(_write_job(tmp_path, "train"))
    clean_repository = {
        "commit": plan["repository"]["commit"],
        "tracked_changes": [],
        "clean": True,
    }
    plan["repository"] = clean_repository
    monkeypatch.setattr(cli, "_git_identity", lambda _root: clean_repository)
    captured: dict[str, object] = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", run)

    assert cli.prepare_training(plan) == 0
    assert captured["command"] == plan["prepare_command"]
    assert "--go" not in captured["command"]


def test_cli_surface_is_five_commands_with_one_job_argument() -> None:
    parser = cli.build_parser()
    subparser_action = next(
        action
        for action in parser._actions
        if action.dest == "command"  # noqa: SLF001
    )

    assert set(subparser_action.choices) == {
        "status",
        "plan",
        "prepare",
        "doctor",
        "run",
    }
    for name in ("plan", "prepare", "doctor", "run"):
        public = [
            action.dest
            for action in subparser_action.choices[name]._actions  # noqa: SLF001
            if action.dest != "help"
        ]
        assert public == ["job"]
