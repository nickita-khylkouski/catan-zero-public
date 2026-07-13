#!/usr/bin/env python3
"""Seal selected-dose value-objective diagnostics against exact TEMP.

Two independent arms are supported.  Both replay the authenticated 524,288-row
/ 128-step TEMP geometry from exact f7 with the same row order, optimizer, LR
trajectory, policy targets, component temperatures, and validation sentinel.

``CURRENT_VALUE_SCOPE`` changes only the authenticated value component scope:
gen3 replay remains a policy-distillation component (at its bound temperature)
but its old-policy Monte-Carlo outcomes receive zero value weight.

``VALUE_LOSS_OFF`` changes only ``value_loss_weight: 0.25 -> 0.0``.  It is a
localization diagnostic for continuing shared-trunk value gradients, not a
candidate recipe and not an architectural claim.

Preparation never launches.  Execution is one-shot and requires ``--go`` plus
the same idle-eight-B200 checks used by the selected-dose pure-target arm.
"""

from __future__ import annotations

import argparse
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
from tools import a1_topology_gather_arm as bridge  # noqa: E402


SCHEMA = "a1-selected-dose-value-axis-arm-v1"
RECEIPT_SCHEMA = "a1-selected-dose-value-axis-execution-receipt-v1"
STATUS_SCHEMA = "a1-selected-dose-value-axis-execution-status-v1"
CLAIM_SCHEMA = "a1-selected-dose-value-axis-execution-claim-v1"
EXECUTOR_RELATIVE_PATH = "tools/a1_selected_dose_value_axis_arm.py"
CURRENT_VALUE_SCOPE = "CURRENT_VALUE_SCOPE"
VALUE_LOSS_OFF = "VALUE_LOSS_OFF"
AXES = frozenset({CURRENT_VALUE_SCOPE, VALUE_LOSS_OFF})
EXPECTED_COMPONENT_IDS = tuple(bridge.production_temp.COMPONENT_IDS)
CURRENT_COMPONENT_IDS = EXPECTED_COMPONENT_IDS[:-1]
REPLAY_COMPONENT_ID = EXPECTED_COMPONENT_IDS[-1]
SOURCE_FILES = (
    EXECUTOR_RELATIVE_PATH,
    "tools/a1_topology_gather_arm.py",
    "tools/a1_corrected_policy_arm.py",
    "tools/a1_corrected_policy_arm_execute.py",
    "tools/a1_production_temperature_replication.py",
    "tools/a1_production_l1_rerun.py",
    "tools/train_bc.py",
    "tools/mixed_memmap_corpus.py",
    "src/catan_zero/rl/entity_token_policy.py",
    "src/catan_zero/rl/entity_token_features.py",
) + (
    ("tools/a1_learner_dose_contract.py",)
    if hasattr(bridge.production_temp, "LEGACY_MANIFEST_SCHEMA")
    else ()
)


class ValueAxisError(RuntimeError):
    """The request is not an exact selected-dose value-axis diagnostic."""


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
        raise ValueAxisError(
            "selected-dose value-axis sources must be clean tracked bytes"
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


def _preflight_descriptor(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    return bridge.corrected._preflight_descriptor(path)  # noqa: SLF001


def _source_descriptor_contract(
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    reference = source.get("descriptor")
    if not isinstance(reference, Mapping):
        raise ValueAxisError("selected TEMP source descriptor reference is malformed")
    path = Path(str(reference.get("path", ""))).expanduser().resolve(strict=True)
    payload, observed_ref = bridge.corrected._load_json(path)  # noqa: SLF001
    if observed_ref != dict(reference):
        raise ValueAxisError("selected TEMP source descriptor bytes drifted")
    try:
        meta, preflight_ref = _preflight_descriptor(path)
    except bridge.corrected.ArmError as error:
        raise ValueAxisError(f"selected TEMP descriptor preflight failed: {error}") from error
    expected = list(EXPECTED_COMPONENT_IDS)
    expected_temperatures = bridge.production_temp.COMPONENT_TEMPERATURES
    component_ids = list(meta.get("component_ids", ()))
    if not (
        preflight_ref == observed_ref
        and component_ids == expected
        and meta.get("policy_distillation_component_ids") == expected
        and meta.get("value_training_component_ids") == expected
        and meta.get("policy_kl_anchor_component_ids") == [REPLAY_COMPONENT_ID]
        and meta.get("stored_policy_component_temperatures")
        == expected_temperatures
    ):
        raise ValueAxisError("source is not the exact all-component TEMP objective")
    replay_temperature = float(expected_temperatures[REPLAY_COMPONENT_ID])
    if replay_temperature != 0.52:
        raise ValueAxisError("TEMP replay policy temperature is not the sealed 0.52")
    return payload, meta, observed_ref


def _write_scope_descriptor(
    source_payload: Mapping[str, Any],
    source_meta: Mapping[str, Any],
    destination: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    payload = dict(source_payload)
    payload["value_training_component_ids"] = list(CURRENT_COMPONENT_IDS)
    expected_payload = dict(source_payload)
    expected_payload["value_training_component_ids"] = list(CURRENT_COMPONENT_IDS)
    if payload != expected_payload:
        raise ValueAxisError("current-value descriptor contains an undeclared mutation")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != encoded:
            raise ValueAxisError(f"current-value descriptor drift: {destination}")
    else:
        temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
        temporary.write_text(encoded, encoding="utf-8")
        os.chmod(temporary, 0o444)
        os.replace(temporary, destination)
    try:
        derived_meta, derived_ref = _preflight_descriptor(destination)
    except bridge.corrected.ArmError as error:
        raise ValueAxisError(f"current-value descriptor preflight failed: {error}") from error
    stable_fields = (
        "component_ids",
        "component_game_sampling_ratios",
        "policy_kl_anchor_component_ids",
        "policy_distillation_component_ids",
        "stored_policy_component_temperatures",
        "learner_recipe_overrides",
        "learner_recipe_overrides_sha256",
    )
    drift = {
        field: {"source": source_meta.get(field), "treatment": derived_meta.get(field)}
        for field in stable_fields
        if source_meta.get(field) != derived_meta.get(field)
    }
    if drift:
        raise ValueAxisError(f"current-value descriptor changed a matched field: {drift}")
    if not (
        derived_meta.get("policy_distillation_component_ids")
        == list(EXPECTED_COMPONENT_IDS)
        and derived_meta.get("value_training_component_ids")
        == list(CURRENT_COMPONENT_IDS)
        and derived_meta.get("policy_kl_anchor_component_ids")
        == [REPLAY_COMPONENT_ID]
        and derived_meta.get("stored_policy_component_temperatures")
        == bridge.production_temp.COMPONENT_TEMPERATURES
    ):
        raise ValueAxisError("derived descriptor does not isolate replay value outcomes")
    return derived_meta, derived_ref


def _assert_selected_source_command(
    command: Sequence[str], *, source_descriptor: Path
) -> None:
    option = bridge.corrected._option  # noqa: SLF001
    expected = {
        "--data": str(source_descriptor),
        "--max-steps": str(bridge.SELECTED_OPTIMIZER_STEPS),
        "--batch-size": "512",
        "--grad-accum-steps": "1",
        "--lr": "3e-05",
        "--lr-warmup-steps": "100",
        "--policy-loss-weight": "1.0",
        "--soft-target-weight": "0.9",
        "--value-loss-weight": "0.25",
        "--value-target-lambda": "1.0",
    }
    observed = {flag: option(command, flag) for flag in expected}
    if observed != expected:
        raise ValueAxisError(
            f"source is not the exact selected-dose TEMP command: {observed}"
        )


def _derive_command(
    source: Sequence[str],
    *,
    axis: str,
    source_descriptor: Path,
    treatment_descriptor: Path,
    output_root: Path,
) -> tuple[list[str], dict[str, Any]]:
    if axis not in AXES:
        raise ValueAxisError(f"unsupported value axis: {axis!r}")
    command = list(source)
    _assert_selected_source_command(command, source_descriptor=source_descriptor)
    changes: dict[str, dict[str, str]] = {
        "--checkpoint": {
            "source": bridge.corrected._option(command, "--checkpoint"),  # noqa: SLF001
            "treatment": str(output_root / "candidate.pt"),
        },
        "--report": {
            "source": bridge.corrected._option(command, "--report"),  # noqa: SLF001
            "treatment": str(output_root / "train.report.json"),
        },
    }
    if axis == CURRENT_VALUE_SCOPE:
        changes["--data"] = {
            "source": str(source_descriptor),
            "treatment": str(treatment_descriptor),
        }
    else:
        changes["--value-loss-weight"] = {
            "source": "0.25",
            "treatment": "0.0",
        }
    for flag, row in changes.items():
        bridge.corrected._set_option(command, flag, row["treatment"])  # noqa: SLF001
    return command, changes


def _axis_delta(axis: str) -> dict[str, Any]:
    if axis == CURRENT_VALUE_SCOPE:
        return {
            "value_training_component_ids": {
                "source": list(EXPECTED_COMPONENT_IDS),
                "treatment": list(CURRENT_COMPONENT_IDS),
            },
            "replay_value_training_enabled": {"source": True, "treatment": False},
        }
    if axis == VALUE_LOSS_OFF:
        return {"value_loss_weight": {"source": 0.25, "treatment": 0.0}}
    raise ValueAxisError(f"unsupported value axis: {axis!r}")


def _matched_contract(axis: str) -> dict[str, Any]:
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
        "validation_unchanged": True,
        "policy_objective_unchanged": True,
        "replay_policy_distillation_enabled": True,
        "replay_policy_temperature": 0.52,
        "value_component_scope_unchanged": axis != CURRENT_VALUE_SCOPE,
        "value_loss_weight_unchanged": axis != VALUE_LOSS_OFF,
    }


def _assert_runtime_support(source: Mapping[str, Any], *, axis: str) -> None:
    if axis != CURRENT_VALUE_SCOPE:
        return
    trainer = Path(str(source["selected_geometry_trainer"])).resolve(strict=True)
    try:
        text = trainer.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueAxisError(f"cannot inspect selected trainer value scope: {error}") from error
    required = (
        "value_training_component_ids",
        "_apply_authenticated_value_training_scope",
    )
    if any(token not in text for token in required):
        raise ValueAxisError(
            "selected geometry trainer predates authenticated value-component scope"
        )


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    axis = str(args.axis)
    if axis not in AXES:
        raise ValueAxisError(f"unsupported value axis: {axis!r}")
    source, source_ref = bridge._load_source(  # noqa: SLF001
        args.source_manifest,
        args.selected_dose_plan,
        args.selected_dose_report,
    )
    _assert_runtime_support(source, axis=axis)
    source_payload, source_meta, source_descriptor_ref = _source_descriptor_contract(
        source
    )
    source_descriptor = Path(source_descriptor_ref["path"])
    output_root = args.output_root.expanduser().resolve()
    forbidden = (
        output_root / "candidate.pt",
        Path(str(output_root / "candidate.pt") + ".optimizer.pt"),
        Path(str(output_root / "candidate.pt") + ".training-progress.json"),
        output_root / "train.report.json",
        output_root / "diagnostic-execution.claim.json",
        output_root / "diagnostic-execution.receipt.json",
    )
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise ValueAxisError(f"selected-dose value-axis output already exists: {existing}")
    output_root.mkdir(parents=True, exist_ok=True)
    descriptor_meta = source_meta
    descriptor_ref = source_descriptor_ref
    if axis == CURRENT_VALUE_SCOPE:
        descriptor_meta, descriptor_ref = _write_scope_descriptor(
            source_payload,
            source_meta,
            output_root / "current-value-scope.memmap-composite.json",
        )
    treatment_descriptor = Path(descriptor_ref["path"])
    binding = _source_binding(args.repo)
    executor_ref = binding["files"].get(EXECUTOR_RELATIVE_PATH)
    if not isinstance(executor_ref, dict):
        raise ValueAxisError("source binding does not authenticate this executor")
    command, changes = _derive_command(
        source["command"],
        axis=axis,
        source_descriptor=source_descriptor,
        treatment_descriptor=treatment_descriptor,
        output_root=output_root,
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "arm_id": axis,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "launch_authorized": False,
        "diagnostic_execution_authorized": True,
        "launch_interface_present": f"{EXECUTOR_RELATIVE_PATH} execute --go",
        "completion_interface_present": (
            "tools/a1_selected_dose_diagnostic_completion.py finalize --manifest"
        ),
        "diagnostic_executor": executor_ref,
        "source_temperature_manifest": source_ref,
        "source_temperature_manifest_sha256": source["manifest_sha256"],
        "selected_geometry_evidence": source["selected_geometry_evidence"],
        "source_recipe": source["recipe"],
        "source_recipe_sha256": source["recipe_sha256"],
        "initialization": source["initialization"],
        "source_descriptor": source_descriptor_ref,
        "treatment_descriptor": descriptor_ref,
        "treatment_descriptor_semantics": {
            "policy_distillation_component_ids": descriptor_meta.get(
                "policy_distillation_component_ids"
            ),
            "value_training_component_ids": descriptor_meta.get(
                "value_training_component_ids"
            ),
            "policy_kl_anchor_component_ids": descriptor_meta.get(
                "policy_kl_anchor_component_ids"
            ),
            "stored_policy_component_temperatures": descriptor_meta.get(
                "stored_policy_component_temperatures"
            ),
        },
        "validation_sentinel": source["validation_sentinel"],
        "source_binding": binding,
        "only_declared_causal_delta": _axis_delta(axis),
        "matched_contract": _matched_contract(axis),
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
    path = output_root / f"selected-dose-{axis.lower().replace('_', '-')}.manifest.json"
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueAxisError(f"prepared value-axis manifest drift: {path}")
    else:
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text(encoded, encoding="utf-8")
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    return manifest, path


def _verify_ref(value: Any, *, label: str) -> Path:
    return executor_base._verify_ref(value, label=label)  # noqa: SLF001


def verify(manifest_path: Path) -> dict[str, Any]:
    payload, manifest_ref = bridge.corrected._load_json(manifest_path)  # noqa: SLF001
    stated = payload.get("manifest_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if stated != bridge.corrected._digest(unhashed):  # noqa: SLF001
        raise ValueAxisError("selected-dose value-axis manifest digest drift")
    axis = payload.get("arm_id")
    if not (
        payload.get("schema_version") == SCHEMA
        and axis in AXES
        and payload.get("diagnostic_only") is True
        and payload.get("promotion_eligible") is False
        and payload.get("launch_authorized") is False
        and payload.get("diagnostic_execution_authorized") is True
        and payload.get("launch_interface_present")
        == f"{EXECUTOR_RELATIVE_PATH} execute --go"
        and payload.get("completion_interface_present")
        == "tools/a1_selected_dose_diagnostic_completion.py finalize --manifest"
    ):
        raise ValueAxisError("selected-dose value-axis authorization drift")
    executor = _verify_ref(payload.get("diagnostic_executor"), label="diagnostic_executor")
    if executor != Path(__file__).resolve():
        raise ValueAxisError("manifest authorizes a different value-axis executor")

    binding = payload.get("source_binding")
    if not isinstance(binding, Mapping):
        raise ValueAxisError("manifest lacks source checkout binding")
    try:
        preparer_repo = Path(str(binding.get("repository_root", ""))).resolve(strict=True)
    except OSError as error:
        raise ValueAxisError(f"cannot resolve preparer checkout: {error}") from error
    if executor_base._git_head(preparer_repo) != binding.get("git_commit"):  # noqa: SLF001
        raise ValueAxisError("preparer checkout commit drift")
    files = binding.get("files")
    if not isinstance(files, Mapping) or set(files) != set(SOURCE_FILES):
        raise ValueAxisError("selected-dose value-axis source binding is incomplete")
    if binding.get("files_sha256") != bridge.corrected._digest(files):  # noqa: SLF001
        raise ValueAxisError("selected-dose value-axis file-set digest drift")
    for relative, ref in files.items():
        path = _verify_ref(ref, label=f"source.{relative}")
        if path != (preparer_repo / relative).resolve(strict=True):
            raise ValueAxisError(f"bound source escaped checkout: {relative}")
    if files[EXECUTOR_RELATIVE_PATH] != payload["diagnostic_executor"]:
        raise ValueAxisError("executor identity differs from source binding")

    source_manifest = _verify_ref(
        payload.get("source_temperature_manifest"), label="source_temperature_manifest"
    )
    evidence = payload.get("selected_geometry_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueAxisError("manifest lacks selected geometry evidence")
    plan = _verify_ref(evidence.get("plan"), label="selected_geometry.plan")
    report = _verify_ref(evidence.get("report"), label="selected_geometry.report")
    try:
        source, _ = bridge._load_source(source_manifest, plan, report)  # noqa: SLF001
    except bridge.ArmError as error:
        raise ValueAxisError(f"selected-dose provenance bridge failed: {error}") from error
    _assert_runtime_support(source, axis=str(axis))
    source_payload, source_meta, source_descriptor_ref = _source_descriptor_contract(
        source
    )
    if not (
        payload.get("source_temperature_manifest_sha256")
        == source["manifest_sha256"]
        and payload.get("selected_geometry_evidence")
        == source["selected_geometry_evidence"]
        and payload.get("source_recipe") == source["recipe"]
        and payload.get("source_recipe_sha256") == source["recipe_sha256"]
        and payload.get("initialization") == source["initialization"]
        and payload.get("source_descriptor") == source_descriptor_ref
        and payload.get("validation_sentinel") == source["validation_sentinel"]
    ):
        raise ValueAxisError("selected-dose source identity drift")
    root = Path(str(payload.get("output_root", ""))).resolve()
    treatment_ref = payload.get("treatment_descriptor")
    if axis == CURRENT_VALUE_SCOPE:
        treatment_path = _verify_ref(treatment_ref, label="treatment_descriptor")
        expected_path = (root / "current-value-scope.memmap-composite.json").resolve()
        if treatment_path != expected_path:
            raise ValueAxisError("current-value descriptor escaped the output root")
        expected_meta, expected_ref = _write_scope_descriptor(
            source_payload, source_meta, expected_path
        )
    else:
        expected_meta, expected_ref = source_meta, source_descriptor_ref
        if treatment_ref != source_descriptor_ref:
            raise ValueAxisError("value-off arm changed the source descriptor")
    semantics = {
        "policy_distillation_component_ids": expected_meta.get(
            "policy_distillation_component_ids"
        ),
        "value_training_component_ids": expected_meta.get(
            "value_training_component_ids"
        ),
        "policy_kl_anchor_component_ids": expected_meta.get(
            "policy_kl_anchor_component_ids"
        ),
        "stored_policy_component_temperatures": expected_meta.get(
            "stored_policy_component_temperatures"
        ),
    }
    if not (
        treatment_ref == expected_ref
        and payload.get("treatment_descriptor_semantics") == semantics
        and payload.get("only_declared_causal_delta") == _axis_delta(str(axis))
        and payload.get("matched_contract") == _matched_contract(str(axis))
    ):
        raise ValueAxisError("value-axis descriptor/delta contract drift")
    source_descriptor = Path(source_descriptor_ref["path"])
    treatment_descriptor = Path(expected_ref["path"])
    expected_command, changes = _derive_command(
        source["command"],
        axis=str(axis),
        source_descriptor=source_descriptor,
        treatment_descriptor=treatment_descriptor,
        output_root=root,
    )
    command = payload.get("command")
    if not (
        command == expected_command
        and payload.get("allowlisted_command_changes") == changes
        and payload.get("command_sha256")
        == bridge.corrected._digest(command)  # noqa: SLF001
    ):
        raise ValueAxisError("value-axis command is not the exact selected-dose derivation")
    trainers = [Path(value).resolve() for value in command if Path(value).name == "train_bc.py"]
    selected_trainer = Path(source["selected_geometry_trainer"]).resolve(strict=True)
    if trainers != [selected_trainer]:
        raise ValueAxisError("value-axis command escaped selected-geometry trainer")
    trainer_repo = Path(source["selected_geometry_runtime_repo"]).resolve(strict=True)
    forbidden = (
        root / "candidate.pt",
        Path(str(root / "candidate.pt") + ".optimizer.pt"),
        Path(str(root / "candidate.pt") + ".training-progress.json"),
        root / "train.report.json",
        root / "diagnostic-execution.claim.json",
        root / "diagnostic-execution.receipt.json",
        root / "diagnostic-execution.status.jsonl",
        root / "stdout.log",
        root / "stderr.log",
    )
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise ValueAxisError(f"selected-dose value-axis output/claim already exists: {existing}")
    return {
        "manifest": payload,
        "manifest_ref": manifest_ref,
        "repo": trainer_repo,
        "preparer_repo": preparer_repo,
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
        raise ValueAxisError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--axis", required=True, choices=sorted(AXES))
    prep.add_argument("--source-manifest", required=True, type=Path)
    prep.add_argument("--selected-dose-plan", required=True, type=Path)
    prep.add_argument("--selected-dose-report", required=True, type=Path)
    prep.add_argument("--output-root", required=True, type=Path)
    prep.add_argument("--repo", default=REPO_ROOT, type=Path)
    run = sub.add_parser("execute")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--unit", default="a1-selected-dose-value-axis")
    run.add_argument("--go", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "prepare":
            manifest, path = prepare(args)
            result = {
                "prepared": str(path),
                "launched": False,
                "manifest_sha256": manifest["manifest_sha256"],
            }
        elif not args.go:
            verified = verify(args.manifest)
            result = {
                "verified": True,
                "launched": False,
                "manifest": verified["manifest_ref"],
            }
        else:
            receipt = execute(args.manifest, unit=args.unit)
            result = {
                "submitted": True,
                "unit": receipt["unit"],
                "receipt_sha256": receipt["receipt_sha256"],
            }
    except (ValueAxisError, bridge.ArmError) as error:
        raise SystemExit(f"REFUSED: {error}") from error
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
