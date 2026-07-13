from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tools import a1_selected_dose_value_axis_arm as arm
from tools import train_bc
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
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
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
        meta["learner_recipe_overrides"] = payload["learner_recipe_overrides"]
        meta["learner_recipe_overrides_sha256"] = payload[
            "learner_recipe_overrides_sha256"
        ]
        return meta, arm.bridge.corrected._file_ref(Path(path))  # noqa: SLF001

    monkeypatch.setattr(arm, "_preflight_descriptor", preflight)

    def scope_sentinel(
        _source_reference,
        *,
        source_descriptor_meta,
        treatment_descriptor_meta,
        destination,
    ):
        del source_descriptor_meta, treatment_descriptor_meta
        payload = {
            "schema_version": "train-validation-game-sentinel-v1",
            "selected_game_seed_set_sha256": "sha256:" + "1" * 64,
            "excluded_game_seed_set_sha256": "sha256:" + "2" * 64,
            "game_seeds": [1, 2],
        }
        destination.write_text(json.dumps(payload), encoding="utf-8")
        return payload, arm.bridge.corrected._file_ref(destination)  # noqa: SLF001

    monkeypatch.setattr(arm, "_write_scope_validation_sentinel", scope_sentinel)
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
        "--validation-game-sentinel-manifest",
        "--checkpoint",
        "--report",
    }
    assert manifest["source_validation_sentinel"] != manifest[
        "treatment_validation_sentinel"
    ]
    assert arm.bridge.corrected._option(  # noqa: SLF001
        command, "--validation-game-sentinel-manifest"
    ) == manifest["treatment_validation_sentinel"]["path"]
    descriptor = json.loads(
        Path(manifest["treatment_descriptor"]["path"]).read_text(encoding="utf-8")
    )
    assert descriptor["value_training_component_ids"] == list(
        arm.CURRENT_COMPONENT_IDS
    )


def test_scope_sentinel_rebinds_only_descriptor_identity(tmp_path: Path) -> None:
    source_meta = {
        "descriptor_file_sha256": "sha256:" + "a" * 64,
        "descriptor_fingerprint": "sha256:" + "b" * 64,
    }
    treatment_meta = {
        "descriptor_file_sha256": "sha256:" + "c" * 64,
        "descriptor_fingerprint": "sha256:" + "d" * 64,
    }
    contracts = [
        {
            "file_sha256": "sha256:" + "1" * 64,
            "manifest_sha256": "sha256:" + "2" * 64,
            "validation_game_seed_set_sha256": "sha256:" + "3" * 64,
        }
    ]
    selected = np.asarray([10, 20], dtype=np.int64)
    excluded = np.asarray([10, 11, 20, 21], dtype=np.int64)
    source = {
        "schema_version": "train-validation-game-sentinel-v1",
        "source_composite_descriptor_file_sha256": source_meta[
            "descriptor_file_sha256"
        ],
        "source_composite_descriptor_fingerprint": source_meta[
            "descriptor_fingerprint"
        ],
        "source_validation_bindings": [
            {
                "component_index": 0,
                "validation_manifest_file_sha256": contracts[0]["file_sha256"],
                "validation_manifest_sha256": contracts[0]["manifest_sha256"],
                "validation_game_seed_set_sha256": contracts[0][
                    "validation_game_seed_set_sha256"
                ],
            }
        ],
        "selection_seed": 7,
        "target_row_count": 100,
        "selected_row_count": 99,
        "selected_game_seed_count": 2,
        "selected_game_seed_set_sha256": train_bc._game_seed_set_sha256(selected),  # noqa: SLF001
        "excluded_game_seed_count": 4,
        "excluded_game_seed_set_sha256": train_bc._game_seed_set_sha256(excluded),  # noqa: SLF001
        "game_seeds": [10, 20],
    }
    source_path = tmp_path / "source-sentinel.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    payload, reference = arm._write_scope_validation_sentinel(  # noqa: SLF001
        arm.bridge.corrected._file_ref(source_path),  # noqa: SLF001
        source_descriptor_meta=source_meta,
        treatment_descriptor_meta=treatment_meta,
        destination=tmp_path / "treatment-sentinel.json",
    )
    assert payload["source_composite_descriptor_file_sha256"] == treatment_meta[
        "descriptor_file_sha256"
    ]
    assert payload["source_composite_descriptor_fingerprint"] == treatment_meta[
        "descriptor_fingerprint"
    ]
    for key in set(source) - {
        "source_composite_descriptor_file_sha256",
        "source_composite_descriptor_fingerprint",
    }:
        assert payload[key] == source[key]
    assert reference["path"].endswith("treatment-sentinel.json")
    full_contract = {
        "validation_row_count": 4,
        "validation_game_seed_set_sha256": source[
            "excluded_game_seed_set_sha256"
        ],
        "game_seeds": excluded,
        "component_contracts": contracts,
    }
    with pytest.raises(SystemExit, match="source composite binding drift"):
        train_bc._load_composite_validation_sentinel_manifest(  # noqa: SLF001
            source_path,
            composite_meta=treatment_meta,
            full_contract=full_contract,
        )
    accepted = train_bc._load_composite_validation_sentinel_manifest(  # noqa: SLF001
        Path(reference["path"]),
        composite_meta=treatment_meta,
        full_contract=full_contract,
    )
    np.testing.assert_array_equal(accepted["game_seeds"], selected)


def test_value_loss_off_is_separate_one_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _ = arm.prepare(_args(tmp_path, monkeypatch, axis=arm.VALUE_LOSS_OFF))
    assert manifest["only_declared_causal_delta"] == {
        "value_loss_weight": {"source": 0.25, "treatment": 0.0}
    }
    assert manifest["source_descriptor"] != manifest["treatment_descriptor"]
    assert manifest["treatment_descriptor_semantics"][
        "value_training_component_ids"
    ] == list(arm.EXPECTED_COMPONENT_IDS)
    command = manifest["command"]
    assert arm.bridge.corrected._option(command, "--value-loss-weight") == "0.0"  # noqa: SLF001
    assert arm.bridge.corrected._option(command, "--data") == manifest[  # noqa: SLF001
        "treatment_descriptor"
    ]["path"]
    assert set(manifest["allowlisted_command_changes"]) == {
        "--data",
        "--validation-game-sentinel-manifest",
        "--value-loss-weight",
        "--checkpoint",
        "--report",
    }
    descriptor = json.loads(
        Path(manifest["treatment_descriptor"]["path"]).read_text(encoding="utf-8")
    )
    assert descriptor["learner_recipe_overrides"]["value_loss_weight"] == 0.0
    runtime_args = type(
        "RuntimeArgs", (), descriptor["learner_recipe_overrides"]
    )()
    train_bc._validate_composite_learner_recipe_authorization(  # noqa: SLF001
        runtime_args,
        {"learner_recipe_overrides": descriptor["learner_recipe_overrides"]},
    )
    source_descriptor = json.loads(
        Path(manifest["source_descriptor"]["path"]).read_text(encoding="utf-8")
    )
    with pytest.raises(SystemExit, match="command differs"):
        train_bc._validate_composite_learner_recipe_authorization(  # noqa: SLF001
            runtime_args,
            {
                "learner_recipe_overrides": source_descriptor[
                    "learner_recipe_overrides"
                ]
            },
        )
    assert manifest["source_validation_sentinel"] != manifest[
        "treatment_validation_sentinel"
    ]


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
        "--validation-game-sentinel-manifest",
        str(tmp_path / "source-sentinel.json"),
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
        source_validation_sentinel=tmp_path / "source-sentinel.json",
        treatment_validation_sentinel=tmp_path / "treatment-sentinel.json",
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
                source_validation_sentinel=tmp_path / "source-sentinel.json",
                treatment_validation_sentinel=tmp_path / "treatment-sentinel.json",
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
