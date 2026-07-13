#!/usr/bin/env python3
"""Seal and execute one selected-dose HL-Gauss value-objective treatment.

The treatment replays the authenticated TEMP geometry from exact f7: eight
ranks, local batch 512, 128 optimizer steps, and 524,288 total row draws.  Its
single causal axis is the primary value representation/loss:

* scalar tanh + MSE -> 33-bin HL-Gauss + categorical cross entropy;
* the scalar value budget (0.25) is reallocated to categorical CE;
* scalar MSE is off, sigma is the trainer-bound default 0.75 bin widths;
* policy targets, rows/order, temperatures, LR trajectory, and value labels are
  unchanged.

The categorical branch is commissioned with the existing checkpoint upgrader.
That initializer is allowed to add only ``value_categorical_head.*`` tensors
and must prove the deployed scalar/policy/Q outputs are bit-identical to f7.
Execution is diagnostic-only, one-shot, and requires an idle eight-B200 host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_corrected_policy_arm_execute as executor_base  # noqa: E402
from tools import a1_selected_dose_value_axis_arm as value_axis  # noqa: E402
from tools import a1_topology_gather_arm as bridge  # noqa: E402


SCHEMA = "a1-selected-dose-hlgauss-arm-v1"
RECEIPT_SCHEMA = "a1-selected-dose-hlgauss-execution-receipt-v1"
STATUS_SCHEMA = "a1-selected-dose-hlgauss-execution-status-v1"
CLAIM_SCHEMA = "a1-selected-dose-hlgauss-execution-claim-v1"
EXECUTOR_RELATIVE_PATH = "tools/a1_selected_dose_hlgauss_arm.py"
RUNTIME_FILES = (
    "tools/train_bc.py",
    "tools/f69_upgrade_checkpoint_config.py",
    "src/catan_zero/rl/entity_token_policy.py",
)
TREATMENT_RUNTIME_DIFF_FILES = (
    "tests/test_categorical_value_readout.py",
    "tools/train_bc.py",
)
TREATMENT_RUNTIME_DIFF_SHA256 = (
    "sha256:151adf4e237e5c2e8b6985551fb55acd5df4d397fd8273ac1b3c672a43785fe1"
)
SOURCE_FILES = tuple(dict.fromkeys(
    (EXECUTOR_RELATIVE_PATH,)
    + value_axis.SOURCE_FILES
    + ("tools/f69_upgrade_checkpoint_config.py",)
))


class HLGaussArmError(RuntimeError):
    """The request is not the exact selected-dose HL-Gauss treatment."""


def _causal_delta() -> dict[str, Any]:
    return {
        "primary_value_representation": {
            "source": "scalar_tanh",
            "treatment": "hlgauss33",
        },
        "primary_value_loss": {
            "source": "mse",
            "treatment": "categorical_cross_entropy",
        },
        "value_budget": {
            "source_scalar_mse": 0.25,
            "treatment_categorical_ce": 0.25,
        },
        "categorical_initialization": {
            "bins": 33,
            "seed": 1,
            "additive_only": True,
            "deployed_outputs_max_abs_diff": 0.0,
        },
    }


def _resolved_contract() -> dict[str, Any]:
    return {
        "bins": 33,
        "truncation_class": True,
        "sigma_ratio_bin_widths": 0.75,
        "categorical_ce_weight": 0.25,
        "scalar_mse_weight": 0.0,
        "categorical_search_readout_requires_trained_provenance": True,
    }


def _matched_contract() -> dict[str, Any]:
    return {
        "global_row_dose": bridge.SELECTED_GLOBAL_ROW_DOSE,
        "optimizer_steps": bridge.SELECTED_OPTIMIZER_STEPS,
        "world_size": 8,
        "local_batch_size": 512,
        "global_batch_size": 4096,
        "fresh_f7_initialization": True,
        "fresh_adam": True,
        "candidate_chaining": False,
        "sample_order_unchanged": True,
        "lr_trajectory_unchanged": True,
        "policy_objective_unchanged": True,
        "value_targets_and_component_scope_unchanged": True,
        "value_lr_mult": 0.3,
        "shared_trunk_uses_base_lr": True,
    }


def _source_binding(repo: Path) -> dict[str, Any]:
    repo = repo.expanduser().resolve(strict=True)
    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=repo, text=True
        ).strip()
        subprocess.run(
            ("git", "diff", "--quiet", "HEAD", "--", *SOURCE_FILES),
            cwd=repo,
            check=True,
        )
        for relative in SOURCE_FILES:
            subprocess.run(
                ("git", "ls-files", "--error-unmatch", relative),
                cwd=repo,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except (OSError, subprocess.CalledProcessError) as error:
        raise HLGaussArmError(
            "selected-dose HL-Gauss sources must be clean tracked bytes"
        ) from error
    files = {
        relative: bridge.corrected._file_ref(repo / relative)  # noqa: SLF001
        for relative in SOURCE_FILES
    }
    return {
        "repository_root": str(repo),
        "git_commit": commit,
        "files": files,
        "files_sha256": bridge.corrected._digest(files),  # noqa: SLF001
    }


def _runtime_binding(source: Mapping[str, Any]) -> dict[str, Any]:
    repo = Path(str(source["selected_geometry_runtime_repo"])).resolve(strict=True)
    evidence = source.get("selected_geometry_evidence")
    runtime = evidence.get("runtime") if isinstance(evidence, Mapping) else None
    if not isinstance(runtime, Mapping):
        raise HLGaussArmError("selected geometry lacks runtime binding")
    commit = executor_base._git_head(repo)  # noqa: SLF001
    if commit != runtime.get("repository_commit"):
        raise HLGaussArmError("selected runtime commit drift")
    try:
        subprocess.run(
            ("git", "diff", "--quiet", "HEAD", "--", *RUNTIME_FILES),
            cwd=repo,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise HLGaussArmError("selected runtime HL-Gauss files are dirty") from error
    files = {
        relative: bridge.corrected._file_ref(repo / relative)  # noqa: SLF001
        for relative in RUNTIME_FILES
    }
    trainer = Path(str(source["selected_geometry_trainer"])).resolve(strict=True)
    if trainer != Path(files["tools/train_bc.py"]["path"]):
        raise HLGaussArmError("selected trainer escaped its bound runtime")
    return {
        "repository_root": str(repo),
        "git_commit": commit,
        "files": files,
        "files_sha256": bridge.corrected._digest(files),  # noqa: SLF001
    }


def _treatment_runtime_binding(
    source: Mapping[str, Any], treatment_repo: Path
) -> dict[str, Any]:
    """Bind the one-commit DDP categorical repair atop selected runtime."""

    repo = treatment_repo.expanduser().resolve(strict=True)
    source_commit = str(
        source["selected_geometry_evidence"]["runtime"]["repository_commit"]
    )
    commit = executor_base._git_head(repo)  # noqa: SLF001
    try:
        parent = subprocess.check_output(
            ("git", "rev-parse", "HEAD^"), cwd=repo, text=True
        ).strip()
        changed = tuple(subprocess.check_output(
            ("git", "diff", "--name-only", f"{parent}..{commit}"),
            cwd=repo,
            text=True,
        ).splitlines())
        patch = subprocess.check_output(
            ("git", "diff", "--binary", f"{parent}..{commit}"), cwd=repo
        )
        subprocess.run(
            ("git", "diff", "--quiet", "HEAD", "--", *RUNTIME_FILES),
            cwd=repo,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise HLGaussArmError("cannot authenticate treatment runtime") from error
    patch_sha = "sha256:" + hashlib.sha256(patch).hexdigest()
    if not (
        parent == source_commit
        and changed == TREATMENT_RUNTIME_DIFF_FILES
        and patch_sha == TREATMENT_RUNTIME_DIFF_SHA256
    ):
        raise HLGaussArmError(
            "treatment runtime is not the reviewed one-commit DDP categorical repair"
        )
    files = {
        relative: bridge.corrected._file_ref(repo / relative)  # noqa: SLF001
        for relative in RUNTIME_FILES
    }
    return {
        "repository_root": str(repo),
        "git_commit": commit,
        "parent_commit": parent,
        "changed_files": list(changed),
        "diff_sha256": patch_sha,
        "files": files,
        "files_sha256": bridge.corrected._digest(files),  # noqa: SLF001
    }


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise HLGaussArmError(f"prepared artifact drift: {path}")
    else:
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text(encoded, encoding="utf-8")
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    return bridge.corrected._file_ref(path)  # noqa: SLF001


def _write_hlgauss_descriptor(
    source_payload: Mapping[str, Any],
    source_meta: Mapping[str, Any],
    destination: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    source_overrides = source_payload.get("learner_recipe_overrides")
    if not isinstance(source_overrides, Mapping):
        raise HLGaussArmError("selected TEMP descriptor lacks learner overrides")
    if source_overrides.get("value_head_type") != "mse":
        raise HLGaussArmError("selected TEMP primary value objective is not MSE")
    if float(source_overrides.get("value_loss_weight", -1.0)) != 0.25:
        raise HLGaussArmError("selected TEMP value budget is not 0.25")
    overrides = dict(source_overrides)
    overrides["value_head_type"] = "hlgauss"
    payload = dict(source_payload)
    payload["learner_recipe_overrides"] = overrides
    payload["learner_recipe_overrides_sha256"] = bridge.corrected._digest(  # noqa: SLF001
        overrides
    )
    _write_json_once(destination, payload)
    try:
        derived_meta, derived_ref = value_axis._preflight_descriptor(destination)  # noqa: SLF001
    except bridge.corrected.ArmError as error:
        raise HLGaussArmError(f"HL-Gauss descriptor preflight failed: {error}") from error
    stable_fields = (
        "component_ids",
        "component_game_sampling_ratios",
        "policy_kl_anchor_component_ids",
        "policy_distillation_component_ids",
        "value_training_component_ids",
        "stored_policy_component_temperatures",
    )
    drift = {
        field: {"source": source_meta.get(field), "treatment": derived_meta.get(field)}
        for field in stable_fields
        if source_meta.get(field) != derived_meta.get(field)
    }
    if drift:
        raise HLGaussArmError(f"HL-Gauss descriptor changed data semantics: {drift}")
    actual_overrides = derived_meta.get("learner_recipe_overrides")
    changed = {
        key: {"source": source_overrides.get(key), "treatment": actual_overrides.get(key)}
        for key in set(source_overrides) | set(actual_overrides or {})
        if source_overrides.get(key) != (actual_overrides or {}).get(key)
    }
    if changed != {"value_head_type": {"source": "mse", "treatment": "hlgauss"}}:
        raise HLGaussArmError(f"HL-Gauss descriptor has extra recipe drift: {changed}")
    return derived_meta, derived_ref


def _verify_categorical_initializer(
    source: Path, upgraded: Path
) -> dict[str, Any]:
    import torch

    source_ref = bridge.corrected._file_ref(source)  # noqa: SLF001
    upgraded_ref = bridge.corrected._file_ref(upgraded)  # noqa: SLF001
    source_raw = torch.load(source, map_location="cpu", weights_only=False)
    upgraded_raw = torch.load(upgraded, map_location="cpu", weights_only=False)
    if not isinstance(source_raw, Mapping) or not isinstance(upgraded_raw, Mapping):
        raise HLGaussArmError("initializer checkpoint is not a mapping")
    source_model = source_raw.get("model")
    upgraded_model = upgraded_raw.get("model")
    if not isinstance(source_model, Mapping) or not isinstance(upgraded_model, Mapping):
        raise HLGaussArmError("initializer lacks entity-graph model state")
    if not set(source_model).issubset(upgraded_model):
        raise HLGaussArmError("categorical commissioning dropped inherited weights")
    changed_inherited = [
        key for key in source_model
        if not torch.equal(source_model[key], upgraded_model[key])
    ]
    added = sorted(set(upgraded_model) - set(source_model))
    if changed_inherited or not added or not all(
        key.startswith("value_categorical_head.") for key in added
    ):
        raise HLGaussArmError(
            "categorical commissioning changed non-additive weights: "
            f"changed={changed_inherited[:8]} added={added[:8]}"
        )
    source_config = bridge._effective_config(source_raw.get("config"))  # noqa: SLF001
    upgraded_config = bridge._effective_config(upgraded_raw.get("config"))  # noqa: SLF001
    config_drift = {
        key: {"source": source_config.get(key), "treatment": upgraded_config.get(key)}
        for key in set(source_config) | set(upgraded_config)
        if source_config.get(key) != upgraded_config.get(key)
    }
    if config_drift != {
        "value_categorical_bins": {"source": 0, "treatment": 33}
    }:
        raise HLGaussArmError(f"categorical config has extra drift: {config_drift}")
    provenance = upgraded_raw.get("upgrade_provenance")
    if not isinstance(provenance, Mapping) or not (
        provenance.get("schema_version") == "entity-graph-upgrade-v1"
        and provenance.get("source_checkpoint_sha256")
        == source_ref["sha256"].removeprefix("sha256:")
        and provenance.get("flags") == {"value_categorical_bins": 33}
        and provenance.get("initialization_seed") == 1
        and provenance.get("trained_value_readouts_added") == []
        and provenance.get("forward_max_diff") == 0.0
        and provenance.get("forward_identical_at_init") is True
    ):
        raise HLGaussArmError("categorical initializer provenance is incomplete")
    if "categorical" in set(map(str, upgraded_raw.get("trained_value_readouts", []))):
        raise HLGaussArmError("fresh categorical head is falsely marked trained")
    return {
        "source": source_ref,
        "initializer": upgraded_ref,
        "config_delta": config_drift,
        "added_parameter_keys": added,
        "added_parameter_keys_sha256": bridge.corrected._digest(added),  # noqa: SLF001
        "upgrade_provenance": dict(provenance),
    }


def _commission_initializer(
    *, source: Path, output: Path, python: Path, upgrader: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    receipt_path = output.with_suffix(".commissioning.json")
    if not output.exists():
        result = subprocess.run(
            (
                str(python),
                str(upgrader),
                "--in-checkpoint", str(source),
                "--out-checkpoint", str(output),
                "--flags", "catbins:33",
                "--seed", "1",
                "--device", "cpu",
            ),
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise HLGaussArmError(
                "categorical commissioning failed: " + result.stderr[-2000:]
            )
        contract = _verify_categorical_initializer(source, output)
        receipt = {
            "schema_version": "a1-hlgauss-commissioning-v1",
            "command": [
                str(python), str(upgrader), "--in-checkpoint", str(source),
                "--out-checkpoint", str(output), "--flags", "catbins:33",
                "--seed", "1", "--device", "cpu",
            ],
            "stdout": result.stdout,
            "contract": contract,
        }
        receipt["receipt_sha256"] = bridge.corrected._digest(receipt)  # noqa: SLF001
        _write_json_once(receipt_path, receipt)
        os.chmod(output, 0o444)
    contract = _verify_categorical_initializer(source, output)
    receipt, receipt_ref = bridge.corrected._load_json(receipt_path)  # noqa: SLF001
    stated = receipt.get("receipt_sha256")
    unhashed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if stated != bridge.corrected._digest(unhashed) or receipt.get("contract") != contract:  # noqa: SLF001
        raise HLGaussArmError("categorical commissioning receipt drift")
    return contract, receipt_ref


def _derive_command(
    source: Sequence[str],
    *,
    source_descriptor: Path,
    treatment_descriptor: Path,
    source_sentinel: Path,
    treatment_sentinel: Path,
    source_init: Path,
    treatment_init: Path,
    treatment_trainer: Path,
    output_root: Path,
) -> tuple[list[str], dict[str, Any]]:
    command = list(source)
    value_axis._assert_selected_source_command(  # noqa: SLF001
        command, source_descriptor=source_descriptor
    )
    option = bridge.corrected._option  # noqa: SLF001
    if option(command, "--validation-game-sentinel-manifest") != str(source_sentinel):
        raise HLGaussArmError("source command is not bound to selected sentinel")
    if option(command, "--init-checkpoint") != str(source_init):
        raise HLGaussArmError("source command is not bound to exact f7")
    if option(command, "--value-head-type") != "mse":
        raise HLGaussArmError("source command primary value objective is not MSE")
    trainer_positions = [
        index for index, value in enumerate(command) if Path(value).name == "train_bc.py"
    ]
    if len(trainer_positions) != 1:
        raise HLGaussArmError("source command does not name exactly one trainer")
    trainer_index = trainer_positions[0]
    changes = {
        "trainer": {
            "source": command[trainer_index],
            "treatment": str(treatment_trainer.resolve(strict=True)),
        },
        "--data": {"source": str(source_descriptor), "treatment": str(treatment_descriptor)},
        "--validation-game-sentinel-manifest": {
            "source": str(source_sentinel), "treatment": str(treatment_sentinel)
        },
        "--init-checkpoint": {"source": str(source_init), "treatment": str(treatment_init)},
        "--value-head-type": {"source": "mse", "treatment": "hlgauss"},
        "--checkpoint": {"source": option(command, "--checkpoint"), "treatment": str(output_root / "candidate.pt")},
        "--report": {"source": option(command, "--report"), "treatment": str(output_root / "train.report.json")},
    }
    command[trainer_index] = changes["trainer"]["treatment"]
    for flag, row in changes.items():
        if flag == "trainer":
            continue
        bridge.corrected._set_option(command, flag, row["treatment"])  # noqa: SLF001
    return command, changes


def _forbidden_outputs(root: Path) -> tuple[Path, ...]:
    checkpoint = root / "candidate.pt"
    return (
        checkpoint,
        Path(str(checkpoint) + ".optimizer.pt"),
        Path(str(checkpoint) + ".training-progress.json"),
        root / "train.report.json",
        root / "diagnostic-execution.claim.json",
        root / "diagnostic-execution.receipt.json",
        root / "diagnostic-execution.status.jsonl",
        root / "stdout.log",
        root / "stderr.log",
    )


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    source, source_ref = bridge._load_source(  # noqa: SLF001
        args.source_manifest, args.selected_dose_plan, args.selected_dose_report
    )
    source_payload, source_meta, source_descriptor_ref = (
        value_axis._source_descriptor_contract(source)  # noqa: SLF001
    )
    output_root = args.output_root.expanduser().resolve()
    existing = [str(path) for path in _forbidden_outputs(output_root) if path.exists()]
    if existing:
        raise HLGaussArmError(f"HL-Gauss output/claim already exists: {existing}")
    output_root.mkdir(parents=True, exist_ok=True)
    runtime = _runtime_binding(source)
    treatment_runtime = _treatment_runtime_binding(
        source, args.treatment_runtime_repo
    )
    source_init = Path(source["initialization"]["path"])
    # Keep the lexical venv entry point. Resolving the symlink to /usr/bin/python
    # silently drops the venv's site-packages and is a real remote-launch footgun.
    python = Path(os.path.abspath(os.fspath(Path(source["command"][0]).expanduser())))
    if not python.is_file() or not os.access(python, os.X_OK):
        raise HLGaussArmError(f"selected training Python is not executable: {python}")
    upgrader = Path(
        treatment_runtime["files"]["tools/f69_upgrade_checkpoint_config.py"]["path"]
    )
    init_contract, commissioning_ref = _commission_initializer(
        source=source_init,
        output=output_root / "f7-catbins33-init.pt",
        python=python,
        upgrader=upgrader,
    )
    descriptor_meta, descriptor_ref = _write_hlgauss_descriptor(
        source_payload,
        source_meta,
        output_root / "hlgauss33.memmap-composite.json",
    )
    sentinel_payload, sentinel_ref = value_axis._write_scope_validation_sentinel(  # noqa: SLF001
        source["validation_sentinel"],
        source_descriptor_meta=source_meta,
        treatment_descriptor_meta=descriptor_meta,
        destination=output_root / "hlgauss33.validation-game-sentinel.json",
    )
    source_descriptor = Path(source_descriptor_ref["path"])
    source_sentinel = Path(source["validation_sentinel"]["path"])
    command, changes = _derive_command(
        source["command"],
        source_descriptor=source_descriptor,
        treatment_descriptor=Path(descriptor_ref["path"]),
        source_sentinel=source_sentinel,
        treatment_sentinel=Path(sentinel_ref["path"]),
        source_init=source_init,
        treatment_init=Path(init_contract["initializer"]["path"]),
        treatment_trainer=Path(
            treatment_runtime["files"]["tools/train_bc.py"]["path"]
        ),
        output_root=output_root,
    )
    binding = _source_binding(args.repo)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "launch_authorized": False,
        "diagnostic_execution_authorized": True,
        "launch_interface_present": f"{EXECUTOR_RELATIVE_PATH} execute --go",
        "diagnostic_executor": binding["files"][EXECUTOR_RELATIVE_PATH],
        "source_temperature_manifest": source_ref,
        "source_temperature_manifest_sha256": source["manifest_sha256"],
        "selected_geometry_evidence": source["selected_geometry_evidence"],
        "source_recipe": source["recipe"],
        "source_recipe_sha256": source["recipe_sha256"],
        "initialization": source["initialization"],
        "categorical_initializer": init_contract,
        "commissioning_receipt": commissioning_ref,
        "source_descriptor": source_descriptor_ref,
        "treatment_descriptor": descriptor_ref,
        "source_validation_sentinel": source["validation_sentinel"],
        "treatment_validation_sentinel": sentinel_ref,
        "treatment_validation_selection_sha256": bridge.corrected._digest({  # noqa: SLF001
            key: value for key, value in sentinel_payload.items()
            if key not in {"source_composite_descriptor_file_sha256", "source_composite_descriptor_fingerprint"}
        }),
        "source_binding": binding,
        "selected_runtime_binding": runtime,
        "treatment_runtime_binding": treatment_runtime,
        "runtime_contract_delta": {
            "source_commit": runtime["git_commit"],
            "treatment_commit": treatment_runtime["git_commit"],
            "diff_sha256": TREATMENT_RUNTIME_DIFF_SHA256,
            "only_runtime_fix": "unwrap DDP model for categorical support width",
        },
        "only_declared_causal_delta": _causal_delta(),
        "resolved_hlgauss_contract": _resolved_contract(),
        "matched_contract": _matched_contract(),
        "allowlisted_command_changes": changes,
        "command": command,
        "command_sha256": bridge.corrected._digest(command),  # noqa: SLF001
        "output_root": str(output_root),
        "executor_compatibility": {
            "receipt_schema": RECEIPT_SCHEMA,
            "idle_topology": "exactly_8_visible_B200s",
            "one_shot": True,
        },
    }
    manifest["manifest_sha256"] = bridge.corrected._digest(manifest)  # noqa: SLF001
    path = output_root / "selected-dose-hlgauss33.manifest.json"
    _write_json_once(path, manifest)
    return manifest, path


def _verify_ref(value: Any, *, label: str) -> Path:
    return executor_base._verify_ref(value, label=label)  # noqa: SLF001


def verify(manifest_path: Path) -> dict[str, Any]:
    payload, manifest_ref = bridge.corrected._load_json(manifest_path)  # noqa: SLF001
    stated = payload.get("manifest_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if stated != bridge.corrected._digest(unhashed):  # noqa: SLF001
        raise HLGaussArmError("HL-Gauss manifest digest drift")
    if not (
        payload.get("schema_version") == SCHEMA
        and payload.get("diagnostic_only") is True
        and payload.get("promotion_eligible") is False
        and payload.get("launch_authorized") is False
        and payload.get("diagnostic_execution_authorized") is True
        and payload.get("launch_interface_present") == f"{EXECUTOR_RELATIVE_PATH} execute --go"
    ):
        raise HLGaussArmError("HL-Gauss authorization drift")
    if not (
        payload.get("only_declared_causal_delta") == _causal_delta()
        and payload.get("resolved_hlgauss_contract") == _resolved_contract()
        and payload.get("matched_contract") == _matched_contract()
        and payload.get("executor_compatibility")
        == {
            "receipt_schema": RECEIPT_SCHEMA,
            "idle_topology": "exactly_8_visible_B200s",
            "one_shot": True,
        }
    ):
        raise HLGaussArmError("HL-Gauss causal/matched contract drift")
    executor = _verify_ref(payload.get("diagnostic_executor"), label="executor")
    if executor != Path(__file__).resolve():
        raise HLGaussArmError("manifest authorizes a different executor")
    binding = payload.get("source_binding")
    if not isinstance(binding, Mapping):
        raise HLGaussArmError("manifest lacks source binding")
    repo = Path(str(binding.get("repository_root", ""))).resolve(strict=True)
    if executor_base._git_head(repo) != binding.get("git_commit"):  # noqa: SLF001
        raise HLGaussArmError("preparer checkout commit drift")
    files = binding.get("files")
    if not isinstance(files, Mapping) or set(files) != set(SOURCE_FILES):
        raise HLGaussArmError("source file-set binding is incomplete")
    if binding.get("files_sha256") != bridge.corrected._digest(files):  # noqa: SLF001
        raise HLGaussArmError("source file-set digest drift")
    for relative, ref in files.items():
        if _verify_ref(ref, label=f"source.{relative}") != (repo / relative).resolve(strict=True):
            raise HLGaussArmError(f"bound source escaped checkout: {relative}")
    evidence = payload.get("selected_geometry_evidence")
    if not isinstance(evidence, Mapping):
        raise HLGaussArmError("manifest lacks selected geometry evidence")
    source, source_ref = bridge._load_source(  # noqa: SLF001
        _verify_ref(payload.get("source_temperature_manifest"), label="source_manifest"),
        _verify_ref(evidence.get("plan"), label="geometry.plan"),
        _verify_ref(evidence.get("report"), label="geometry.report"),
    )
    if not (
        payload.get("source_temperature_manifest") == source_ref
        and payload.get("source_temperature_manifest_sha256") == source["manifest_sha256"]
        and payload.get("source_recipe") == source["recipe"]
        and payload.get("source_recipe_sha256") == source["recipe_sha256"]
        and payload.get("initialization") == source["initialization"]
    ):
        raise HLGaussArmError("selected-dose source identity drift")
    runtime = _runtime_binding(source)
    if payload.get("selected_runtime_binding") != runtime:
        raise HLGaussArmError("selected runtime binding drift")
    treatment_binding_raw = payload.get("treatment_runtime_binding")
    if not isinstance(treatment_binding_raw, Mapping):
        raise HLGaussArmError("manifest lacks treatment runtime binding")
    treatment_runtime = _treatment_runtime_binding(
        source, Path(str(treatment_binding_raw.get("repository_root", "")))
    )
    if not (
        treatment_binding_raw == treatment_runtime
        and payload.get("runtime_contract_delta")
        == {
            "source_commit": runtime["git_commit"],
            "treatment_commit": treatment_runtime["git_commit"],
            "diff_sha256": TREATMENT_RUNTIME_DIFF_SHA256,
            "only_runtime_fix": "unwrap DDP model for categorical support width",
        }
    ):
        raise HLGaussArmError("treatment runtime delta drift")
    source_payload, source_meta, source_descriptor_ref = value_axis._source_descriptor_contract(source)  # noqa: SLF001
    root = Path(str(payload.get("output_root", ""))).resolve()
    descriptor_meta, descriptor_ref = _write_hlgauss_descriptor(
        source_payload, source_meta, root / "hlgauss33.memmap-composite.json"
    )
    sentinel_payload, sentinel_ref = value_axis._write_scope_validation_sentinel(  # noqa: SLF001
        source["validation_sentinel"],
        source_descriptor_meta=source_meta,
        treatment_descriptor_meta=descriptor_meta,
        destination=root / "hlgauss33.validation-game-sentinel.json",
    )
    initializer = _verify_categorical_initializer(
        Path(source["initialization"]["path"]), root / "f7-catbins33-init.pt"
    )
    receipt, receipt_ref = bridge.corrected._load_json(  # noqa: SLF001
        root / "f7-catbins33-init.commissioning.json"
    )
    receipt_unhashed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if not (
        receipt.get("receipt_sha256") == bridge.corrected._digest(receipt_unhashed)  # noqa: SLF001
        and receipt.get("contract") == initializer
        and payload.get("categorical_initializer") == initializer
        and payload.get("commissioning_receipt") == receipt_ref
        and payload.get("source_descriptor") == source_descriptor_ref
        and payload.get("treatment_descriptor") == descriptor_ref
        and payload.get("source_validation_sentinel") == source["validation_sentinel"]
        and payload.get("treatment_validation_sentinel") == sentinel_ref
        and payload.get("treatment_validation_selection_sha256") == bridge.corrected._digest({  # noqa: SLF001
            key: value for key, value in sentinel_payload.items()
            if key not in {"source_composite_descriptor_file_sha256", "source_composite_descriptor_fingerprint"}
        })
    ):
        raise HLGaussArmError("HL-Gauss derived artifact drift")
    expected_command, changes = _derive_command(
        source["command"],
        source_descriptor=Path(source_descriptor_ref["path"]),
        treatment_descriptor=Path(descriptor_ref["path"]),
        source_sentinel=Path(source["validation_sentinel"]["path"]),
        treatment_sentinel=Path(sentinel_ref["path"]),
        source_init=Path(source["initialization"]["path"]),
        treatment_init=Path(initializer["initializer"]["path"]),
        treatment_trainer=Path(
            treatment_runtime["files"]["tools/train_bc.py"]["path"]
        ),
        output_root=root,
    )
    command = payload.get("command")
    if not (
        command == expected_command
        and payload.get("allowlisted_command_changes") == changes
        and payload.get("command_sha256") == bridge.corrected._digest(command)  # noqa: SLF001
    ):
        raise HLGaussArmError("command is not the exact HL-Gauss derivation")
    trainers = [Path(value).resolve() for value in command if Path(value).name == "train_bc.py"]
    trainer = Path(
        treatment_runtime["files"]["tools/train_bc.py"]["path"]
    ).resolve(strict=True)
    if trainers != [trainer]:
        raise HLGaussArmError("HL-Gauss command escaped selected trainer")
    existing = [str(path) for path in _forbidden_outputs(root) if path.exists()]
    if existing:
        raise HLGaussArmError(f"HL-Gauss output/claim already exists: {existing}")
    return {
        "manifest": payload,
        "manifest_ref": manifest_ref,
        "repo": Path(treatment_runtime["repository_root"]).resolve(strict=True),
        "preparer_repo": repo,
        "command": command,
        "output_root": root,
    }


def execute(
    manifest_path: Path,
    *,
    unit: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    conflict_probe: Callable[[], list[str]] = executor_base._probe_conflicting_compute,  # noqa: SLF001
) -> dict[str, Any]:
    try:
        verified = verify(manifest_path)
        return executor_base._submit_verified(  # noqa: SLF001
            verified,
            unit=unit,
            runner=runner,
            conflict_probe=conflict_probe,
            claim_schema=CLAIM_SCHEMA,
            receipt_schema=RECEIPT_SCHEMA,
            status_schema=STATUS_SCHEMA,
        )
    except executor_base.ExecutionError as error:
        raise HLGaussArmError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-manifest", required=True, type=Path)
    prep.add_argument("--selected-dose-plan", required=True, type=Path)
    prep.add_argument("--selected-dose-report", required=True, type=Path)
    prep.add_argument("--output-root", required=True, type=Path)
    prep.add_argument("--repo", default=REPO_ROOT, type=Path)
    prep.add_argument("--treatment-runtime-repo", required=True, type=Path)
    run = sub.add_parser("execute")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--unit", default="a1-selected-dose-hlgauss33")
    run.add_argument("--go", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "prepare":
            manifest, path = prepare(args)
            result = {"prepared": str(path), "launched": False, "manifest_sha256": manifest["manifest_sha256"]}
        elif not args.go:
            verified = verify(args.manifest)
            result = {"verified": True, "launched": False, "manifest": verified["manifest_ref"]}
        else:
            receipt = execute(args.manifest, unit=args.unit)
            result = {"submitted": True, "unit": receipt["unit"], "receipt_sha256": receipt["receipt_sha256"]}
    except (HLGaussArmError, bridge.ArmError) as error:
        raise SystemExit(f"REFUSED: {error}") from error
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
