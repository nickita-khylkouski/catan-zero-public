from __future__ import annotations

import copy
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
        if recipe != "a1-current-35m-b200":
            parent = tmp_path / "parent.pt"
            parent.write_bytes(b"parent-v1")
            migration = tmp_path / "information-contract-migration.receipt.json"
            migration.write_text("{}", encoding="utf-8")
            payload.update(
                init_checkpoint=str(checkpoint),
                parent_checkpoint=str(parent),
                information_contract_migration_receipt=str(migration),
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


def _commissioning_fixture(tmp_path: Path) -> tuple[Path, dict[str, object], Path]:
    repo = tmp_path / "repo"
    config = repo / "configs/training/recipe.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")
    evidence_dir = repo / "docs/evidence"
    evidence_dir.mkdir(parents=True)
    identity: dict[str, object] = {
        "config": str(config),
        "config_sha256": canonical_json_sha256({}),
    }
    return repo, identity, evidence_dir


def _primary_commissioning(
    identity: dict[str, object], repo: Path
) -> dict[str, object]:
    return {
        "schema_version": "a1-coherent-v6-b12-commissioning-evidence-v1",
        "code": {
            "recipe": Path(str(identity["config"])).relative_to(repo).as_posix(),
            "recipe_canonical_sha256": identity["config_sha256"],
        },
        "commissioning_gates": {"passed": True},
        "decision": {"authorize_sealed_parent_update": True},
    }


def test_authorized_training_evidence_binds_exact_primary_and_support(
    tmp_path: Path,
) -> None:
    repo, identity, evidence_dir = _commissioning_fixture(tmp_path)
    primary = evidence_dir / "primary.json"
    primary.write_text(
        json.dumps(_primary_commissioning(identity, repo)), encoding="utf-8"
    )
    support = evidence_dir / "support.json"
    support.write_text(
        json.dumps({"schema_version": "a1-effective-policy-signal-audit-v1"}),
        encoding="utf-8",
    )

    validated = contracts.validate_training_commissioning_evidence(
        repo,
        identity=identity,
        evidence=["docs/evidence/primary.json", "docs/evidence/support.json"],
    )

    assert [item["schema_version"] for item in validated] == [
        "a1-coherent-v6-b12-commissioning-evidence-v1",
        "a1-effective-policy-signal-audit-v1",
    ]


@pytest.mark.parametrize("drift", ("recipe", "recipe_canonical_sha256"))
def test_authorized_training_evidence_rejects_primary_identity_drift(
    tmp_path: Path, drift: str
) -> None:
    repo, identity, evidence_dir = _commissioning_fixture(tmp_path)
    payload = _primary_commissioning(identity, repo)
    code = payload["code"]
    assert isinstance(code, dict)
    code[drift] = "wrong"
    (evidence_dir / "primary.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        contracts.ProductionContractError,
        match="does not bind the exact recipe",
    ):
        contracts.validate_training_commissioning_evidence(
            repo,
            identity=identity,
            evidence=["docs/evidence/primary.json"],
        )


@pytest.mark.parametrize("field", ("gates", "decision"))
def test_authorized_training_evidence_rejects_failed_primary(
    tmp_path: Path, field: str
) -> None:
    repo, identity, evidence_dir = _commissioning_fixture(tmp_path)
    payload = _primary_commissioning(identity, repo)
    if field == "gates":
        payload["commissioning_gates"] = {"passed": False}
    else:
        payload["decision"] = {"authorize_sealed_parent_update": False}
    (evidence_dir / "primary.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        contracts.ProductionContractError,
        match="does not authorize training",
    ):
        contracts.validate_training_commissioning_evidence(
            repo,
            identity=identity,
            evidence=["docs/evidence/primary.json"],
        )


def test_authorized_training_evidence_rejects_support_only_and_unsafe_paths(
    tmp_path: Path,
) -> None:
    repo, identity, evidence_dir = _commissioning_fixture(tmp_path)
    support = evidence_dir / "support.json"
    support.write_text(
        json.dumps({"schema_version": "a1-effective-policy-signal-audit-v1"}),
        encoding="utf-8",
    )
    with pytest.raises(
        contracts.ProductionContractError,
        match="matching primary commissioning evidence",
    ):
        contracts.validate_training_commissioning_evidence(
            repo,
            identity=identity,
            evidence=["docs/evidence/support.json"],
        )
    with pytest.raises(
        contracts.ProductionContractError,
        match="checked-in JSON under docs/evidence",
    ):
        contracts.validate_training_commissioning_evidence(
            repo,
            identity=identity,
            evidence=["../outside.json"],
        )


def test_status_exposes_v7_parent_and_scratch_as_fail_closed() -> None:
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
    assert parent["reason"] == ("v8_parent_update_requires_fresh_commissioning")
    assert len(parent["unresolved_requirements"]) == 1
    shared_action = train["recipes"]["a1-parent-update-shared-action25-35m-b200"]
    assert shared_action["authorized"] is False
    assert shared_action["status"] == "blocked"
    assert shared_action["reason"] == (
        "shared_action_trust_region_requires_fresh_commissioning"
    )
    assert len(shared_action["unresolved_requirements"]) == 1
    action_local = train["recipes"][
        "a1-parent-update-action25-shared-action25-35m-b200"
    ]
    assert action_local["authorized"] is False
    assert action_local["status"] == "blocked"
    assert action_local["reason"] == (
        "action_local_trust_region_requires_fresh_commissioning"
    )
    assert len(action_local["unresolved_requirements"]) == 1
    combined = train["recipes"][
        "a1-parent-update-shared-action25-value25-35m-b200"
    ]
    assert combined["authorized"] is False
    assert combined["status"] == "blocked"
    assert combined["reason"] == (
        "combined_trust_region_requires_fresh_commissioning"
    )
    assert len(combined["unresolved_requirements"]) == 1
    value_head = train["recipes"]["a1-parent-update-value25-35m-b200"]
    assert value_head["authorized"] is False
    assert value_head["status"] == "blocked"
    assert value_head["reason"] == (
        "value_head_trust_region_requires_fresh_commissioning"
    )
    assert len(value_head["unresolved_requirements"]) == 1
    p10 = train["recipes"]["a1-parent-update-active-p10-35m-b200"]
    assert p10["authorized"] is False
    assert p10["status"] == "blocked"
    assert p10["reason"] == (
        "active_policy_parent_treatment_requires_fresh_commissioning"
    )
    assert len(p10["unresolved_requirements"]) == 1
    assert set(train["recipes"]) == {
        entry["name"] for entry in contracts.production_recipes("train")
    }
    assert all(
        readiness["authorized"] is False
        for readiness in train["recipes"].values()
    )
    assert status["pipelines"]["ppo"]["authorized"] is False


def test_training_launcher_resolution_fails_closed_on_unknown_mode(
    tmp_path: Path,
) -> None:
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "name": "experimental",
                "engine_settings": {"initialization_mode": "unreviewed_mode"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        contracts.ProductionContractError,
        match="unsupported initialization mode",
    ):
        contracts._train_launcher(recipe, recipe="experimental")  # noqa: SLF001


def test_all_canonical_config_and_guard_identities_are_exact() -> None:
    identities = [
        validate_pipeline_contract(ROOT, "generate"),
        validate_pipeline_contract(ROOT, "evaluate"),
        *[
            validate_pipeline_contract(ROOT, "train", entry["name"])
            for entry in contracts.production_recipes("train")
        ],
    ]
    for identity in identities:
        payload = json.loads(Path(identity["config"]).read_text(encoding="utf-8"))
        assert identity["config_sha256"] == canonical_json_sha256(payload)
    generation_entry = contracts.production_recipes("generate")[0]
    assert identities[0]["guard"] == generation_entry["guard"]
    assert identities[0]["guard_sha256"] == generation_entry["guard_sha256"]
    assert identities[0]["required_accelerator_model"] == "NVIDIA H100"
    assert identities[1]["required_accelerator_model"] == "NVIDIA H100"
    assert all(
        identity["required_accelerator_model"] == "NVIDIA B200"
        for identity in identities[2:]
    )


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


@pytest.mark.parametrize(
    "recipe",
    [
        entry["name"]
        for entry in contracts.production_recipes("train")
        if entry["name"] != "a1-current-35m-b200"
    ],
)
def test_cataloged_parent_update_uses_exact_blocked_recipe_and_parent(
    tmp_path: Path, recipe: str
) -> None:
    plan = cli.build_plan(
        _write_job(tmp_path, "train", recipe=recipe)
    )

    assert plan["readiness"]["authorized"] is False
    assert plan["contract"]["recipe"] == recipe
    assert str((ROOT / "tools/train.py").resolve()) in plan["command"]
    assert "--init-checkpoint" in plan["command"]
    assert "--parent-checkpoint" in plan["command"]
    assert "--information-contract-migration-receipt" in plan["command"]
    assert "--architecture-upgrade-receipt" not in plan["command"]
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


def test_training_science_admission_keeps_v5_quarantine_after_v6_commissioning() -> (
    None
):
    payload = json.loads(
        (ROOT / contracts.TRAINING_SCIENCE_ADMISSION).read_text(encoding="utf-8")
    )
    expected_adapter = (
        "rust_entity_adapter_v6_exact_actor_resources_initial_road_two_hop"
    )

    scratch = payload["recipes"]["a1-current-35m-b200"]
    parent = payload["recipes"]["a1-parent-update-35m-b200"]
    assert scratch["authorized"] is False
    assert any(
        expected_adapter in requirement
        for requirement in scratch["unresolved_requirements"]
    )
    assert parent["authorized"] is False
    assert len(parent["unresolved_requirements"]) == 1

    for admission in (scratch, parent):
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
    assert parent["observations"]["retained_composite_adapter_status"] == (
        "fresh_adapter_v6_composite_authenticated"
    )


def test_parent_update_requires_receipt_only_for_changed_initializer(
    tmp_path: Path,
) -> None:
    changed = _write_job(tmp_path, "train", recipe="a1-parent-update-35m-b200")
    payload = json.loads(changed.read_text(encoding="utf-8"))
    payload.pop("information_contract_migration_receipt")
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(cli.ProductionCLIError, match="receipt is required"):
        cli.build_plan(changed)

    payload["init_checkpoint"] = payload["parent_checkpoint"]
    changed.write_text(json.dumps(payload), encoding="utf-8")
    plan = cli.build_plan(changed)
    assert "--information-contract-migration-receipt" not in plan["command"]


def test_parent_update_rejects_obsolete_architecture_upgrade_receipt(
    tmp_path: Path,
) -> None:
    job = _write_job(tmp_path, "train", recipe="a1-parent-update-35m-b200")
    payload = json.loads(job.read_text(encoding="utf-8"))
    payload["architecture_upgrade_receipt"] = payload.pop(
        "information_contract_migration_receipt"
    )
    job.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(cli.ProductionCLIError, match="architecture_upgrade_receipt"):
        cli.load_job(job)


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
    monkeypatch.setitem(cli.sys.modules, "catanatron_rs.catanatron_rs", native_module)
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


def _parent_update_admissible_report(
    plan: dict[str, object],
    *,
    checkpoint: Path,
) -> tuple[dict[str, object], set[str]]:
    contract = plan["contract"]
    assert isinstance(contract, dict)
    config_path = Path(str(contract["config"]))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fields = config["train_config"]["fields"]
    engine = config["engine_settings"]
    max_steps = int(fields["max_steps"])
    world_size = 8
    global_batch_size = (
        world_size * int(fields["batch_size"]) * int(fields["grad_accum_steps"])
    )
    checkpoint_steps = [
        int(token) for token in str(engine["checkpoint_steps"]).split(",")
    ]
    terminal_steps = [*checkpoint_steps, max_steps]
    required_modules = sorted(
        name.strip()
        for name in str(engine["require_feature_learning_signal_modules"]).split(",")
        if name.strip()
    )
    module_row = {
        "mean_pre_clip_grad_norm": 0.25,
        "max_pre_clip_grad_norm": 0.5,
        "mean_parameter_delta_norm": 0.01,
        "mean_parameter_update_rms": 0.001,
        "mean_relative_parameter_delta": 0.0001,
        "parameter_count": 64,
    }
    modules = {name: dict(module_row) for name in required_modules}
    admitted_modules = {
        name: {
            key: value
            for key, value in module_row.items()
            if key
            in {
                "mean_pre_clip_grad_norm",
                "max_pre_clip_grad_norm",
                "mean_parameter_delta_norm",
                "mean_parameter_update_rms",
                "parameter_count",
            }
        }
        for name in required_modules
    }
    module_observability = {
        "schema_version": "module-optimizer-observability-v1",
        "observed_steps": 2,
        "cadence_batches": int(engine["train_diagnostics_every_batches"]),
        "norm_scope": "global_replicated",
        "modules": modules,
    }
    feature_admission = {
        "schema_version": "a1-feature-learning-signal-admission-v1",
        "authenticated": True,
        "observed_steps": 2,
        "cadence_batches": int(engine["train_diagnostics_every_batches"]),
        "norm_scope": "global_replicated",
        "required_modules": required_modules,
        "positive_signal_fields": [
            "mean_pre_clip_grad_norm",
            "max_pre_clip_grad_norm",
            "mean_parameter_delta_norm",
            "mean_parameter_update_rms",
        ],
        "modules": admitted_modules,
    }
    objective_observations = [
        {
            "optimizer_step": step,
            "available": True,
            "policy_trunk_grad_norm": 1.0,
            "value_trunk_grad_norm": 0.2,
            "combined_trunk_grad_norm": 1.1,
            "value_to_policy_grad_norm_ratio": 0.2,
            "trunk_gradient_cosine": 0.25,
            "opposing_coordinate_fraction": 0.1,
        }
        for step in (1, int(engine["objective_gradient_interference_every_batches"]))
    ]
    objective_admission_rows = [
        {
            key: value
            for key, value in row.items()
            if key
            in {
                "optimizer_step",
                "policy_trunk_grad_norm",
                "value_trunk_grad_norm",
                "combined_trunk_grad_norm",
                "value_to_policy_grad_norm_ratio",
                "trunk_gradient_cosine",
                "opposing_coordinate_fraction",
            }
        }
        for row in objective_observations
    ]

    intermediate_records: list[dict[str, object]] = []
    checkpoint_ref_by_step: dict[int, dict[str, str]] = {}
    expected_outputs: set[str] = set()
    for step in checkpoint_steps:
        path = checkpoint.with_name(
            f"{checkpoint.stem}_step{step:04d}{checkpoint.suffix}"
        )
        path.write_bytes(f"checkpoint-step-{step}".encode("ascii"))
        digest = cli._file_sha256(path)  # noqa: SLF001
        intermediate_records.append(
            {
                "schema_version": "train-bc-intermediate-checkpoint-v1",
                "optimizer_step": step,
                "checkpoint": str(path),
                "checkpoint_sha256": f"sha256:{digest}",
                "size_bytes": path.stat().st_size,
                "same_training_trajectory": True,
                "optimizer_sidecar": None,
            }
        )
        checkpoint_ref_by_step[step] = {
            "path": str(path),
            "sha256": f"sha256:{digest}",
        }
        expected_outputs.add(f"training_checkpoint_step_{step:06d}")
    checkpoint_ref_by_step[max_steps] = {
        "path": str(checkpoint),
        "sha256": f"sha256:{cli._file_sha256(checkpoint)}",  # noqa: SLF001
    }

    aux_batch_size = int(fields["policy_aux_active_batch_size"])
    aux_coefficient = float(fields["policy_aux_loss_weight"])

    def dose_row(step: int) -> dict[str, object]:
        base_draws = step * global_batch_size
        aux_draws = step * world_size * aux_batch_size
        base_mass = float(base_draws)
        aux_mass = float(aux_draws)
        total_mass = base_mass + aux_mass
        weighted_mass = base_mass + (
            aux_coefficient * aux_mass if aux_draws > 0 else 0.0
        )
        optimizer = {
            "observed_steps": step,
            "clipped_steps": step // 4,
            "clipped_fraction": (step // 4) / step,
            "zero_objective_steps_skipped": 0,
            "mean_pre_clip_total_grad_norm": 0.75,
            "max_pre_clip_total_grad_norm": 1.25,
        }
        return {
            "schema_version": "train-bc-checkpoint-dose-telemetry-v1",
            "optimizer_step": step,
            "training_row_draws": {
                "base_training_row_draws": base_draws,
                "policy_aux_training_row_draws": aux_draws,
                "policy_active_training_row_draws": base_draws + aux_draws,
                "value_active_training_row_draws": base_draws,
                "total_training_row_draws": base_draws + aux_draws,
            },
            "active_rows": {
                "policy_base": base_draws,
                "policy_aux": aux_draws,
                "policy_total": base_draws + aux_draws,
                "value": base_draws,
                "policy_kl_anchor": 0,
            },
            "policy_objective_dose": {
                "active_rows": base_draws + aux_draws,
                "equivalent_active_rows": float(base_draws + aux_draws),
                "coefficient_weighted_effective_weight_sum": weighted_mass,
                "equivalent_effective_weight_sum": weighted_mass,
                "optimizer_updates": step,
                "equivalent_optimizer_updates": float(step),
            },
            "policy_effective_weight_sums": {
                "base": base_mass,
                "aux": aux_mass,
                "total": total_mass,
            },
            "policy_stream_objective": {
                "schema_version": "train-policy-stream-objective-v1",
                "formula": "base_mean + aux_coefficient * aux_mean",
                "normalization": "independent_weighted_means",
                "base_coefficient": 1.0,
                "aux_enabled": aux_draws > 0,
                "aux_coefficient": aux_coefficient if aux_draws > 0 else 0.0,
                "base_denominator": base_mass,
                "aux_denominator": aux_mass,
            },
            "objective_effective_weight_sums": {
                "policy_base_loss": base_mass,
                "policy_aux_loss": aux_mass,
                "active_policy_loss": aux_mass,
            },
            "optimizer": optimizer,
            "shared_trunk_objective_gradients": {
                "schema_version": "objective-gradient-dose-observations-v2",
                "cadence_batches": int(
                    engine["objective_gradient_interference_every_batches"]
                ),
                "observed_steps": sum(
                    row["optimizer_step"] <= step for row in objective_observations
                ),
                "observations": [
                    row
                    for row in objective_observations
                    if int(row["optimizer_step"]) <= step
                ],
            },
            "module_optimizer_observability": {
                **module_observability,
                "observed_steps": step
                // int(engine["train_diagnostics_every_batches"]),
            },
        }

    trajectory_rows = [dose_row(step) for step in terminal_steps]
    terminal_dose = trajectory_rows[-1]
    multiplier_area = sum(
        (step + 1) / int(fields["lr_warmup_steps"]) for step in range(max_steps)
    )
    base_lr = float(fields["lr"])
    semantic_multipliers = {"base": 1.0}
    for name, field in (
        ("value", "value_lr_mult"),
        ("action_local", "action_module_lr_mult"),
        ("shared_action", "shared_action_lr_mult"),
        ("public_card", "public_card_lr_mult"),
        ("trunk", "trunk_lr_mult"),
    ):
        multiplier = float(fields[field])
        if multiplier != 1.0:
            semantic_multipliers[name] = multiplier
    optimizer_lr_groups = [
        {
            "semantic_group_name": name,
            "optimizer_group_indices": [index * 2, index * 2 + 1],
            "optimizer_group_count": 2,
            "parameter_tensors": 4,
            "parameters": 256,
            "base_lr": base_lr * multiplier,
            "integrated_lr_area": group_lr * multiplier_area,
            "mean_applied_lr": group_lr * multiplier_area / max_steps,
        }
        for index, (name, multiplier) in enumerate(semantic_multipliers.items())
        for group_lr in (base_lr * multiplier,)
    ]
    terminal_optimizer = terminal_dose["optimizer"]
    assert isinstance(terminal_optimizer, dict)
    report: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "max_steps": max_steps,
        "exact_max_steps": True,
        "steps_completed": max_steps,
        "total_training_steps": max_steps,
        "world_size": world_size,
        "batch_size": int(fields["batch_size"]),
        "grad_accum_steps": int(fields["grad_accum_steps"]),
        "effective_global_batch_size": global_batch_size,
        "optimizer": fields["optimizer"],
        "fused_optimizer": bool(fields["fused_optimizer"]),
        "fused_optimizer_requested": bool(fields["fused_optimizer"]),
        "fused_optimizer_runtime": {
            "requested": bool(fields["fused_optimizer"]),
            "attempted": bool(fields["fused_optimizer"]),
            "effective": bool(fields["fused_optimizer"]),
            "fallback_after_type_error": False,
        },
        "lr": float(fields["lr"]),
        "lr_warmup_steps": int(fields["lr_warmup_steps"]),
        "lr_schedule": fields["lr_schedule"],
        "max_grad_norm": float(fields["max_grad_norm"]),
        "weight_decay": float(fields["weight_decay"]),
        "policy_loss_weight": float(fields["policy_loss_weight"]),
        "policy_aux_active_batch_size": aux_batch_size,
        "policy_aux_loss_weight": aux_coefficient,
        "policy_aux_sampling_mode": fields["policy_aux_sampling_mode"],
        "value_lr_mult": float(fields["value_lr_mult"]),
        "action_module_lr_mult": float(fields["action_module_lr_mult"]),
        "shared_action_lr_mult": float(fields["shared_action_lr_mult"]),
        "public_card_lr_mult": float(fields["public_card_lr_mult"]),
        "trunk_lr_mult": float(fields["trunk_lr_mult"]),
        "value_trunk_grad_scale": float(fields["value_trunk_grad_scale"]),
        "train_diagnostics_every_batches": int(
            engine["train_diagnostics_every_batches"]
        ),
        "objective_gradient_interference_every_batches": int(
            engine["objective_gradient_interference_every_batches"]
        ),
        "a1_canonical_parent_update_authority": {
            "schema_version": "a1-canonical-parent-update-runtime-authority-v1",
            "config": str(config_path.resolve()),
            "config_file_sha256": f"sha256:{cli._file_sha256(config_path)}",  # noqa: SLF001
            "diagnostic_only": True,
            "promotion_eligible": False,
        },
        "optimizer_lr_dose": {
            "schema_version": "optimizer-lr-dose-v2",
            "scope": "updates_applied_in_this_process_invocation",
            "applied_updates": max_steps,
            "integrated_schedule_multiplier_area": multiplier_area,
            "mean_schedule_multiplier": multiplier_area / max_steps,
            "parameter_groups": optimizer_lr_groups,
        },
        "checkpoint_steps_requested": checkpoint_steps,
        "intermediate_checkpoints": intermediate_records,
        "checkpoint_dose_trajectory": {
            "schema_version": "train-bc-checkpoint-dose-trajectory-v1",
            "checkpoint_steps": terminal_steps,
            "checkpoints": trajectory_rows,
        },
        "checkpoint_holdout_frontier": {
            "schema_version": "train-bc-checkpoint-holdout-frontier-v1",
            "measure": "report_bound_raw_validation_rows",
            "validation_game_seed_set_sha256": "sha256:" + "1" * 64,
            "checkpoints": [
                {
                    "schema_version": "train-bc-checkpoint-holdout-v1",
                    "optimizer_step": step,
                    "checkpoint": checkpoint_ref_by_step[step]["path"],
                    "checkpoint_sha256": checkpoint_ref_by_step[step]["sha256"],
                    "measure": "report_bound_raw_validation_rows",
                    "validation_game_seed_set_sha256": "sha256:" + "1" * 64,
                    "metrics": {
                        "loss": 0.75,
                        "policy_loss": 0.5,
                        "value_loss": 0.25,
                        "samples": 512,
                    },
                }
                for step in terminal_steps
            ],
        },
        "module_optimizer_observability": module_observability,
        "feature_learning_signal_admission": feature_admission,
        "objective_gradient_interference": {
            "schema_version": "objective-gradient-dose-observations-v1",
            "cadence_batches": int(
                engine["objective_gradient_interference_every_batches"]
            ),
            "observed_steps": len(objective_observations),
            "observations": objective_observations,
        },
        "objective_gradient_signal_admission": {
            "schema_version": "a1-objective-gradient-signal-admission-v1",
            "authenticated": True,
            "cadence_batches": int(
                engine["objective_gradient_interference_every_batches"]
            ),
            "observed_steps": len(objective_admission_rows),
            "world_size": world_size,
            "scalar_value_trunk_grad_scale": float(fields["value_trunk_grad_scale"]),
            "observations": objective_admission_rows,
        },
        "base_training_row_draws": terminal_dose["training_row_draws"][
            "base_training_row_draws"
        ],
        "policy_aux_training_row_draws": terminal_dose["training_row_draws"][
            "policy_aux_training_row_draws"
        ],
        "total_training_row_draws": terminal_dose["training_row_draws"][
            "total_training_row_draws"
        ],
        "policy_base_active_rows": terminal_dose["active_rows"]["policy_base"],
        "policy_aux_active_rows": terminal_dose["active_rows"]["policy_aux"],
        "policy_total_active_rows": terminal_dose["active_rows"]["policy_total"],
        "value_active_rows": terminal_dose["active_rows"]["value"],
        "policy_base_effective_weight_sum": terminal_dose[
            "policy_effective_weight_sums"
        ]["base"],
        "policy_aux_effective_weight_sum": terminal_dose[
            "policy_effective_weight_sums"
        ]["aux"],
        "policy_total_effective_weight_sum": terminal_dose[
            "policy_effective_weight_sums"
        ]["total"],
        "policy_objective_effective_weight_sum": terminal_dose["policy_objective_dose"][
            "coefficient_weighted_effective_weight_sum"
        ],
        "policy_objective_optimizer_updates": max_steps,
        "metrics": [
            {
                "epoch": 1,
                "loss": 1.0,
                "policy_loss": 0.75,
                "value_loss": 0.25,
                "optimizer_observability": terminal_optimizer,
            }
        ],
    }
    return report, expected_outputs


def _parent_validation_fixture(
    tmp_path: Path,
    *,
    recipe: str = "a1-parent-update-35m-b200",
) -> tuple[dict[str, object], dict[str, object], Path, dict[str, object]]:
    with patch.object(
        cli,
        "pipeline_readiness",
        return_value={"authorized": False, "reason": "validator_fixture"},
    ):
        plan = cli.build_plan(_write_job(tmp_path, "train", recipe=recipe))
    run_dir = Path(str(plan["job"]["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "candidate.pt"
    checkpoint.write_bytes(b"trained-candidate")
    report, _ = _parent_update_admissible_report(plan, checkpoint=checkpoint)
    checkpoint_ref, _ = cli._stable_output_ref(  # noqa: SLF001
        checkpoint, label="training checkpoint"
    )
    return plan, report, checkpoint, checkpoint_ref


@pytest.mark.parametrize(
    "recipe",
    (
        "a1-parent-update-35m-b200",
        "a1-parent-update-shared-action25-35m-b200",
        "a1-parent-update-shared-action25-value25-35m-b200",
        "a1-parent-update-value25-35m-b200",
        "a1-parent-update-active-p10-35m-b200",
        "a1-parent-update-active-p25-35m-b200",
    ),
)
def test_parent_update_validator_accepts_every_catalog_recipe(
    tmp_path: Path, recipe: str
) -> None:
    plan, report, checkpoint, checkpoint_ref = _parent_validation_fixture(
        tmp_path, recipe=recipe
    )

    admitted = cli._verify_parent_update_outputs(  # noqa: SLF001
        plan,
        report=report,
        checkpoint_path=checkpoint,
        checkpoint_ref=checkpoint_ref,
    )

    assert set(admitted) == {
        "training_checkpoint_step_000008",
        "training_checkpoint_step_000010",
    }
    if recipe in {
        "a1-parent-update-35m-b200",
        "a1-parent-update-active-p10-35m-b200",
    }:
        dose = report["optimizer_lr_dose"]
        assert isinstance(dose, dict)
        assert [row["semantic_group_name"] for row in dose["parameter_groups"]] == [
            "base",
            "trunk",
        ]


@pytest.mark.parametrize(
    "tamper",
    (
        "short_run",
        "lr_area",
        "semantic_group",
        "trajectory",
        "nonfinite_clipping",
        "aux_dose",
        "feature_admission",
        "empty_feature_evidence",
        "objective_admission",
        "empty_objective_evidence",
        "aux_coefficient",
        "weighted_objective_mass",
        "fused_receipt_omitted",
        "fused_receipt_contradiction",
        "holdout_binding",
        "holdout_nan",
        "holdout_zero_samples",
    ),
)
def test_parent_update_validator_rejects_authenticated_report_drift(
    tmp_path: Path, tamper: str
) -> None:
    plan, original, checkpoint, checkpoint_ref = _parent_validation_fixture(
        tmp_path, recipe="a1-parent-update-active-p10-35m-b200"
    )
    report = copy.deepcopy(original)
    trajectory = report["checkpoint_dose_trajectory"]
    assert isinstance(trajectory, dict)
    rows = trajectory["checkpoints"]
    assert isinstance(rows, list)

    if tamper == "short_run":
        report["steps_completed"] = 3
    elif tamper == "lr_area":
        dose = report["optimizer_lr_dose"]
        assert isinstance(dose, dict)
        dose["integrated_schedule_multiplier_area"] = 0.0
    elif tamper == "semantic_group":
        dose = report["optimizer_lr_dose"]
        assert isinstance(dose, dict)
        groups = dose["parameter_groups"]
        assert isinstance(groups, list)
        groups[0]["semantic_group_name"] = "anonymous"
    elif tamper == "trajectory":
        trajectory["checkpoint_steps"] = [10, 8, 12]
    elif tamper == "nonfinite_clipping":
        rows[0]["optimizer"]["clipped_fraction"] = float("nan")
    elif tamper == "aux_dose":
        rows[-1]["active_rows"]["policy_aux"] = 0
    elif tamper == "feature_admission":
        report["feature_learning_signal_admission"]["authenticated"] = False
    elif tamper == "empty_feature_evidence":
        report["feature_learning_signal_admission"]["modules"] = {}
    elif tamper == "objective_admission":
        report["objective_gradient_signal_admission"]["observed_steps"] = "2"
    elif tamper == "empty_objective_evidence":
        report["objective_gradient_signal_admission"]["observations"] = []
    elif tamper == "aux_coefficient":
        rows[-1]["policy_stream_objective"]["aux_coefficient"] = 0.9
    elif tamper == "weighted_objective_mass":
        rows[-1]["policy_objective_dose"][
            "coefficient_weighted_effective_weight_sum"
        ] += 1.0
    elif tamper == "fused_receipt_omitted":
        report.pop("fused_optimizer_runtime")
    elif tamper == "fused_receipt_contradiction":
        report["fused_optimizer_runtime"]["effective"] = False
    elif tamper == "holdout_binding":
        report["checkpoint_holdout_frontier"]["checkpoints"][-1][
            "checkpoint_sha256"
        ] = "sha256:" + "0" * 64
    elif tamper == "holdout_nan":
        report["checkpoint_holdout_frontier"]["checkpoints"][-1]["metrics"][
            "loss"
        ] = float("nan")
    elif tamper == "holdout_zero_samples":
        report["checkpoint_holdout_frontier"]["checkpoints"][-1]["metrics"][
            "samples"
        ] = 0
    else:
        raise AssertionError(f"unknown tamper: {tamper}")

    with pytest.raises(cli.ProductionCLIError):
        cli._verify_parent_update_outputs(  # noqa: SLF001
            plan,
            report=report,
            checkpoint_path=checkpoint,
            checkpoint_ref=checkpoint_ref,
        )


def test_parent_update_validator_hashes_and_requires_intermediate_checkpoints(
    tmp_path: Path,
) -> None:
    plan, report, checkpoint, checkpoint_ref = _parent_validation_fixture(tmp_path)
    records = report["intermediate_checkpoints"]
    assert isinstance(records, list)
    intermediate = Path(str(records[0]["checkpoint"]))
    intermediate.write_bytes(b"tampered-after-report")

    with pytest.raises(cli.ProductionCLIError, match="binding drifted"):
        cli._verify_parent_update_outputs(  # noqa: SLF001
            plan,
            report=report,
            checkpoint_path=checkpoint,
            checkpoint_ref=checkpoint_ref,
        )


def test_parent_update_validator_allows_explicit_fused_backend_fallback(
    tmp_path: Path,
) -> None:
    plan, report, checkpoint, checkpoint_ref = _parent_validation_fixture(tmp_path)
    report["fused_optimizer"] = False
    runtime = report["fused_optimizer_runtime"]
    assert isinstance(runtime, dict)
    runtime["effective"] = False
    runtime["fallback_after_type_error"] = True

    cli._verify_parent_update_outputs(  # noqa: SLF001
        plan,
        report=report,
        checkpoint_path=checkpoint,
        checkpoint_ref=checkpoint_ref,
    )


def test_parent_update_validator_rejects_legacy_minimal_success_report(
    tmp_path: Path,
) -> None:
    plan, _report, checkpoint, checkpoint_ref = _parent_validation_fixture(tmp_path)

    with pytest.raises(cli.ProductionCLIError, match="recipe echo drift"):
        cli._verify_parent_update_outputs(  # noqa: SLF001
            plan,
            report={
                "checkpoint": str(checkpoint),
                "steps_completed": 3,
                "epochs": 1,
            },
            checkpoint_path=checkpoint,
            checkpoint_ref=checkpoint_ref,
        )


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
        expected = {"training_candidate", "training_report"}
        if cli._is_parent_update_recipe(job["recipe"]):  # noqa: SLF001
            report_payload, parent_outputs = _parent_update_admissible_report(
                plan, checkpoint=candidate
            )
            expected.update(parent_outputs)
            report.write_text(json.dumps(report_payload), encoding="utf-8")
        else:
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
            execution["receipt_sha256"] = "sha256:" + canonical_json_sha256(execution)
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
        ("train", "a1-parent-update-active-p10-35m-b200"),
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
    assert (
        "production inputs changed during execution"
        in receipt["output_admission_error"]
    )
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
