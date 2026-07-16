from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from catan_zero import production_cli as cli
from catan_zero import production_contracts as contracts
from catan_zero.production_contracts import (
    NATIVE_REQUIRED_CAPABILITIES,
    ProductionContractError,
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
            payload.update(init_checkpoint=str(checkpoint))
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


def test_status_exposes_only_commissioned_production_state() -> None:
    status = production_status(ROOT)

    assert status["supported_operator_interface"] == "catan-zero"
    assert status["pipelines"]["generate"]["authorized"] is True
    assert status["pipelines"]["evaluate"]["authorized"] is True
    train = status["pipelines"]["train"]
    assert train["authorized"] is True
    assert train["reason"] == "recipe_specific_authorization"
    assert train["recipes"]["a1-current-35m-b200"]["authorized"] is False
    assert train["recipes"]["a1-parent-update-35m-b200"] == {
        "pipeline": "train",
        "recipe": "a1-parent-update-35m-b200",
        "status": "ready",
        "authorized": True,
        "reason": "commissioned_parent_update_recipe",
        "authority": str(
            (ROOT / "configs/training/a1_parent_update_35m_b200.schema1.json").resolve()
        ),
        "authority_sha256": "ed804b9180e6ee773cd85590d69bdd79160a2813ba4ead012eb7b7d1f4e43cd7",
    }
    assert status["pipelines"]["ppo"]["authorized"] is False


def test_all_canonical_config_and_guard_identities_are_exact() -> None:
    identities = [
        validate_pipeline_contract(ROOT, "generate"),
        validate_pipeline_contract(ROOT, "evaluate"),
        validate_pipeline_contract(ROOT, "train", "a1-current-35m-b200"),
        validate_pipeline_contract(
            ROOT, "train", "a1-parent-update-35m-b200"
        ),
    ]
    for identity in identities:
        payload = json.loads(Path(identity["config"]).read_text(encoding="utf-8"))
        assert identity["config_sha256"] == canonical_json_sha256(payload)
    assert identities[0]["guard_sha256"] == contracts.GENERATION_GUARD_SHA256


def test_guard_identity_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = ROOT / contracts.GENERATION_GUARD
    payload = json.loads(original.read_text(encoding="utf-8"))
    payload["schema_version"] = "drifted"
    drifted = tmp_path / "guard.json"
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(contracts, "GENERATION_GUARD", str(drifted))
    with pytest.raises(ProductionContractError, match="guard identity drift"):
        validate_pipeline_contract(ROOT, "generate")


def test_generate_plan_has_one_canonical_command_and_bound_inputs(tmp_path: Path) -> None:
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


def test_commissioned_parent_update_uses_exact_recipe_and_parent(
    tmp_path: Path,
) -> None:
    plan = cli.build_plan(
        _write_job(tmp_path, "train", recipe="a1-parent-update-35m-b200")
    )

    assert plan["readiness"]["authorized"] is True
    assert plan["contract"]["recipe"] == "a1-parent-update-35m-b200"
    assert plan["contract"]["config_sha256"] == (
        "ed804b9180e6ee773cd85590d69bdd79160a2813ba4ead012eb7b7d1f4e43cd7"
    )
    assert str((ROOT / "tools/train.py").resolve()) in plan["command"]
    assert "--init-checkpoint" in plan["command"]
    assert "--nproc-per-node=8" in plan["command"]
    assert plan["prepare_command"] is None


def test_production_job_rejects_unknown_keys_and_relative_paths(tmp_path: Path) -> None:
    unknown = _write_job(tmp_path, unexpected_knob=True)
    with pytest.raises(cli.ProductionCLIError, match="unknown=.*unexpected_knob"):
        cli.load_job(unknown)

    relative = _write_job(tmp_path, checkpoint="candidate.pt")
    with pytest.raises(cli.ProductionCLIError, match="checkpoint must be absolute"):
        cli.load_job(relative)


def test_ppo_gets_a_typed_refusal(tmp_path: Path) -> None:
    job = _write_job(tmp_path, "ppo")

    with pytest.raises(cli.ProductionCLIError, match="PPO is not a commissioned"):
        cli.build_plan(job)


def test_plan_artifact_drift_is_detected(tmp_path: Path) -> None:
    plan = cli.build_plan(_write_job(tmp_path))
    Path(plan["inputs"]["checkpoint"]["path"]).write_bytes(b"replaced")

    assert any("input checkpoint drift" in error for error in cli._verify_plan_artifacts(plan))  # noqa: SLF001


def test_doctor_accepts_only_exact_runtime_and_clean_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = cli.build_plan(_write_job(tmp_path))
    runtime = json.loads(
        (ROOT / "configs/runtime/a1_production_runtime.json").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(cli.platform, "python_version", lambda: runtime["python_version"])
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
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 8),
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


def test_doctor_refuses_blocked_training_even_with_exact_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = cli.build_plan(_write_job(tmp_path, "train"))
    monkeypatch.setattr(cli, "_package_version", lambda _name: None)
    monkeypatch.setattr(cli, "_nvidia_driver_identity", lambda: {"versions": [], "error": "test"})
    monkeypatch.setattr(cli, "_native_runtime_identity", lambda: {"wheel_sha256": None, "capabilities": []})

    result = cli.doctor(plan)

    assert result["ok"] is False
    assert "pipeline is blocked: scratch_optimizer_schedule_unresolved" in result["errors"]
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
        action for action in parser._actions if action.dest == "command"  # noqa: SLF001
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
