from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess

import pytest

from tools import a1_selected_dose_value_axis_arm as arm
from test_a1_topology_gather_arm import _args as topology_args


def _args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    axis: str,
):
    source = topology_args(tmp_path, monkeypatch)
    executor = tmp_path / arm.EXECUTOR_RELATIVE_PATH
    executor.parent.mkdir(exist_ok=True)
    executor.write_text("# value-axis executor\n", encoding="utf-8")
    executor_ref = arm.bridge.corrected._file_ref(executor)  # noqa: SLF001
    monkeypatch.setattr(
        arm,
        "_source_binding",
        lambda repo: {
            "repository_root": str(repo),
            "git_commit": "value-axis-test-head",
            "files": {arm.EXECUTOR_RELATIVE_PATH: executor_ref},
            "files_sha256": arm.bridge.corrected._digest(  # noqa: SLF001
                {arm.EXECUTOR_RELATIVE_PATH: executor_ref}
            ),
        },
    )
    monkeypatch.setattr(arm, "_assert_runtime_support", lambda *_args, **_kwargs: None)

    base_preflight = arm.bridge.corrected._preflight_descriptor  # noqa: SLF001

    def preflight(path: Path):
        meta, _ = base_preflight(path)
        meta = copy.deepcopy(meta)
        meta["component_ids"] = list(arm.EXPECTED_COMPONENT_IDS)
        meta["component_game_sampling_ratios"] = [0.5714286, 0.2285714, 0.2]
        meta["policy_distillation_component_ids"] = list(
            arm.EXPECTED_COMPONENT_IDS
        )
        meta["value_training_component_ids"] = (
            list(arm.CURRENT_COMPONENT_IDS)
            if Path(path).name == "current-value-scope.memmap-composite.json"
            else list(arm.EXPECTED_COMPONENT_IDS)
        )
        meta["policy_kl_anchor_component_ids"] = [arm.REPLAY_COMPONENT_ID]
        meta["stored_policy_component_temperatures"] = (
            arm.bridge.production_temp.COMPONENT_TEMPERATURES
        )
        meta["learner_recipe_overrides"] = {
            "per_game_policy_weight": False,
            "per_game_policy_weight_mode": "equal",
        }
        meta["learner_recipe_overrides_sha256"] = "sha256:" + "a" * 64
        return meta, arm.bridge.corrected._file_ref(Path(path))  # noqa: SLF001

    monkeypatch.setattr(arm, "_preflight_descriptor", preflight)
    return type(
        "Args",
        (),
        {
            "axis": axis,
            "source_manifest": source.source_manifest,
            "selected_dose_plan": source.selected_dose_plan,
            "selected_dose_report": source.selected_dose_report,
            "output_root": tmp_path / axis.lower(),
            "repo": tmp_path,
        },
    )()


def test_current_value_scope_is_one_axis_and_retains_replay_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, path = arm.prepare(
        _args(tmp_path, monkeypatch, axis=arm.CURRENT_VALUE_SCOPE)
    )
    assert path.is_file()
    assert manifest["completion_interface_present"].endswith("finalize --manifest")
    assert manifest["only_declared_causal_delta"] == {
        "value_training_component_ids": {
            "source": list(arm.EXPECTED_COMPONENT_IDS),
            "treatment": list(arm.CURRENT_COMPONENT_IDS),
        },
        "replay_value_training_enabled": {"source": True, "treatment": False},
    }
    semantics = manifest["treatment_descriptor_semantics"]
    assert semantics["policy_distillation_component_ids"] == list(
        arm.EXPECTED_COMPONENT_IDS
    )
    assert semantics["value_training_component_ids"] == list(
        arm.CURRENT_COMPONENT_IDS
    )
    assert semantics["policy_kl_anchor_component_ids"] == [
        arm.REPLAY_COMPONENT_ID
    ]
    assert semantics["stored_policy_component_temperatures"][
        arm.REPLAY_COMPONENT_ID
    ] == pytest.approx(0.52)
    command = manifest["command"]
    assert arm.bridge.corrected._option(command, "--value-loss-weight") == "0.25"  # noqa: SLF001
    assert arm.bridge.corrected._option(command, "--soft-target-weight") == "0.9"  # noqa: SLF001
    assert arm.bridge.corrected._option(command, "--max-steps") == "128"  # noqa: SLF001
    assert set(manifest["allowlisted_command_changes"]) == {
        "--data",
        "--checkpoint",
        "--report",
    }
    descriptor = json.loads(
        Path(manifest["treatment_descriptor"]["path"]).read_text(encoding="utf-8")
    )
    assert descriptor["value_training_component_ids"] == list(
        arm.CURRENT_COMPONENT_IDS
    )


def test_value_loss_off_is_separate_one_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _ = arm.prepare(_args(tmp_path, monkeypatch, axis=arm.VALUE_LOSS_OFF))
    assert manifest["only_declared_causal_delta"] == {
        "value_loss_weight": {"source": 0.25, "treatment": 0.0}
    }
    assert manifest["source_descriptor"] == manifest["treatment_descriptor"]
    assert manifest["treatment_descriptor_semantics"][
        "value_training_component_ids"
    ] == list(arm.EXPECTED_COMPONENT_IDS)
    command = manifest["command"]
    assert arm.bridge.corrected._option(command, "--value-loss-weight") == "0.0"  # noqa: SLF001
    assert arm.bridge.corrected._option(command, "--data") == manifest[  # noqa: SLF001
        "source_descriptor"
    ]["path"]
    assert set(manifest["allowlisted_command_changes"]) == {
        "--value-loss-weight",
        "--checkpoint",
        "--report",
    }


def test_derivation_refuses_nonselected_value_or_dose(tmp_path: Path) -> None:
    source_descriptor = (tmp_path / "source.json").resolve()
    source = [
        "python",
        "train_bc.py",
        "--data",
        str(source_descriptor),
        "--max-steps",
        "128",
        "--batch-size",
        "512",
        "--grad-accum-steps",
        "1",
        "--lr",
        "3e-05",
        "--lr-warmup-steps",
        "100",
        "--policy-loss-weight",
        "1.0",
        "--soft-target-weight",
        "0.9",
        "--value-loss-weight",
        "0.25",
        "--value-target-lambda",
        "1.0",
        "--checkpoint",
        "old.pt",
        "--report",
        "old.json",
    ]
    treatment = tmp_path / "treatment.json"
    command, _ = arm._derive_command(
        source,
        axis=arm.VALUE_LOSS_OFF,
        source_descriptor=source_descriptor,
        treatment_descriptor=treatment,
        output_root=tmp_path / "out",
    )
    assert arm.bridge.corrected._option(command, "--value-loss-weight") == "0.0"  # noqa: SLF001
    for flag, bad_value in (("--max-steps", "1024"), ("--value-loss-weight", "1.0")):
        bad = list(source)
        bad[bad.index(flag) + 1] = bad_value
        with pytest.raises(arm.ValueAxisError, match="exact selected-dose TEMP"):
            arm._derive_command(
                bad,
                axis=arm.VALUE_LOSS_OFF,
                source_descriptor=source_descriptor,
                treatment_descriptor=treatment,
                output_root=tmp_path / "bad",
            )


def test_verify_replays_bridge_and_complete_source_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, path = arm.prepare(
        _args(tmp_path, monkeypatch, axis=arm.CURRENT_VALUE_SCOPE)
    )
    repo = Path(arm.__file__).resolve().parents[1]
    files = {
        relative: arm.bridge.corrected._file_ref(repo / relative)  # noqa: SLF001
        for relative in arm.SOURCE_FILES
    }
    manifest["source_binding"] = {
        "repository_root": str(repo),
        "git_commit": "value-axis-test-head",
        "files": files,
        "files_sha256": arm.bridge.corrected._digest(files),  # noqa: SLF001
    }
    manifest["diagnostic_executor"] = files[arm.EXECUTOR_RELATIVE_PATH]
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = arm.bridge.corrected._digest(manifest)  # noqa: SLF001
    path.chmod(0o644)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        arm.executor_base, "_git_head", lambda _repo: "value-axis-test-head"
    )
    verified = arm.verify(path)
    assert verified["command"] == manifest["command"]
    assert verified["repo"] == (tmp_path / "geometry-checkout").resolve()


def test_verify_refuses_treatment_descriptor_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, path = arm.prepare(
        _args(tmp_path, monkeypatch, axis=arm.CURRENT_VALUE_SCOPE)
    )
    descriptor = Path(manifest["treatment_descriptor"]["path"])
    descriptor.chmod(0o644)
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    payload["policy_distillation_component_ids"] = list(arm.CURRENT_COMPONENT_IDS)
    descriptor.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((arm.ValueAxisError, arm.executor_base.ExecutionError)):
        arm.verify(path)


def test_execute_submits_only_verified_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "out"
    verified = {
        "manifest": {"command_sha256": "sha256:" + "1" * 64},
        "manifest_ref": {"path": "/manifest", "sha256": "sha256:" + "2" * 64},
        "repo": tmp_path,
        "command": ["python", "train_bc.py"],
        "output_root": root,
    }
    monkeypatch.setattr(arm, "verify", lambda _path: verified)
    calls: list[list[str]] = []

    def runner(command, **_kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(
            command, 0, stdout="Running as unit.", stderr=""
        )

    receipt = arm.execute(
        tmp_path / "manifest.json",
        unit="a1-value-axis-test",
        runner=runner,
        conflict_probe=lambda: [],
    )
    assert receipt["schema_version"] == arm.RECEIPT_SCHEMA
    assert len(calls) == 1
    assert calls[0][calls[0].index("--") + 1 :] == verified["command"]
