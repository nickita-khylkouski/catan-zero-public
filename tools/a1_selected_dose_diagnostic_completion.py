#!/usr/bin/env python3
"""Finalize or replay selected-dose learner diagnostic artifacts.

Selected-dose launchers intentionally create an authorization claim and a
submission receipt before systemd starts the learner.  Those files do not prove
that 128 optimizer steps completed or bind the eventual checkpoint/report.
This post-hoc finalizer leaves the original source-bound manifest untouched and
adds a separate, replayable, non-promotable completion receipt.

Supported manifests:

* selected-dose pure-search-target (soft target 0.9 -> 1.0);
* selected-dose current-value-scope;
* selected-dose value-loss-off.

Every completion replays the source TEMP + selected geometry bridge, the exact
derived command, claim, submission systemd command, report objective/dose,
component scopes/temperatures, and hashes all required output artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from tools import a1_corrected_policy_arm_execute as executor_base
from tools import a1_selected_dose_pure_soft_arm as pure_soft
from tools import a1_selected_dose_value_axis_arm as value_axis


SCHEMA = "a1-selected-dose-diagnostic-completion-v1"
STATUS = "complete_nonpromotable"
COMPLETION_NAME = "diagnostic-completion.receipt.json"


class CompletionError(RuntimeError):
    """A selected-dose diagnostic cannot be authenticated as complete."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _file_ref(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise CompletionError(f"artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": "sha256:" + digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _compact_ref(path: Path) -> dict[str, str]:
    ref = _file_ref(path)
    return {"path": str(ref["path"]), "sha256": str(ref["sha256"])}


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompletionError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise CompletionError(f"{label} is not a JSON object")
    return value


def _verify_bound_ref(value: Any, *, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise CompletionError(f"{label} reference is malformed")
    try:
        path = executor_base._verify_ref(value, label=label)  # noqa: SLF001
    except executor_base.ExecutionError as error:
        raise CompletionError(str(error)) from error
    return path


def _verify_source_binding(
    manifest: Mapping[str, Any], *, expected_files: Sequence[str]
) -> None:
    binding = manifest.get("source_binding")
    if not isinstance(binding, Mapping):
        raise CompletionError("manifest lacks source binding")
    files = binding.get("files")
    if not isinstance(files, Mapping) or set(files) != set(expected_files):
        raise CompletionError("manifest source binding file set drift")
    if binding.get("files_sha256") != value_axis.bridge.corrected._digest(files):  # noqa: SLF001
        raise CompletionError("manifest source binding digest drift")
    try:
        repo = Path(str(binding.get("repository_root", ""))).resolve(strict=True)
    except OSError as error:
        raise CompletionError(f"cannot resolve manifest source checkout: {error}") from error
    for relative, reference in files.items():
        path = _verify_bound_ref(reference, label=f"source.{relative}")
        if path != (repo / relative).resolve(strict=True):
            raise CompletionError(f"bound source escaped checkout: {relative}")


def _load_selected_source(manifest: Mapping[str, Any]) -> dict[str, Any]:
    source_manifest = _verify_bound_ref(
        manifest.get("source_temperature_manifest"),
        label="source_temperature_manifest",
    )
    evidence = manifest.get("selected_geometry_evidence")
    if not isinstance(evidence, Mapping):
        raise CompletionError("manifest lacks selected geometry evidence")
    plan = _verify_bound_ref(evidence.get("plan"), label="selected_geometry.plan")
    report = _verify_bound_ref(
        evidence.get("report"), label="selected_geometry.report"
    )
    try:
        source, _ = value_axis.bridge._load_source(  # noqa: SLF001
            source_manifest, plan, report
        )
    except value_axis.bridge.ArmError as error:
        raise CompletionError(f"selected-dose source bridge failed: {error}") from error
    if not (
        manifest.get("source_temperature_manifest_sha256")
        == source["manifest_sha256"]
        and manifest.get("selected_geometry_evidence")
        == source["selected_geometry_evidence"]
        and manifest.get("source_recipe") == source["recipe"]
        and manifest.get("source_recipe_sha256") == source["recipe_sha256"]
        and manifest.get("initialization") == source["initialization"]
        and manifest.get("validation_sentinel") == source["validation_sentinel"]
    ):
        raise CompletionError("manifest selected-dose source identity drift")
    return source


def _verify_value_axis_descriptor(
    manifest: Mapping[str, Any], source: Mapping[str, Any], *, arm_id: str
) -> tuple[dict[str, Any], dict[str, str]]:
    source_payload, source_meta, source_ref = value_axis._source_descriptor_contract(  # noqa: SLF001
        source
    )
    treatment = manifest.get("treatment_descriptor")
    if arm_id == value_axis.CURRENT_VALUE_SCOPE:
        treatment_path = _verify_bound_ref(treatment, label="treatment_descriptor")
        root = Path(str(manifest["output_root"])).resolve()
        if treatment_path != (root / "current-value-scope.memmap-composite.json").resolve():
            raise CompletionError("current-value descriptor escaped output root")
        try:
            treatment_payload, treatment_ref = value_axis.bridge.corrected._load_json(  # noqa: SLF001
                treatment_path
            )
            treatment_meta, preflight_ref = value_axis._preflight_descriptor(  # noqa: SLF001
                treatment_path
            )
        except value_axis.bridge.corrected.ArmError as error:
            raise CompletionError(f"current-value descriptor preflight failed: {error}") from error
        expected_payload = dict(source_payload)
        expected_payload["value_training_component_ids"] = list(
            value_axis.CURRENT_COMPONENT_IDS
        )
        if treatment_payload != expected_payload or treatment_ref != preflight_ref:
            raise CompletionError("current-value descriptor has hidden drift")
    else:
        if treatment != source_ref:
            raise CompletionError("value-off diagnostic changed the TEMP descriptor")
        treatment_meta, treatment_ref = source_meta, source_ref
    expected_policy = list(value_axis.EXPECTED_COMPONENT_IDS)
    expected_value = (
        list(value_axis.CURRENT_COMPONENT_IDS)
        if arm_id == value_axis.CURRENT_VALUE_SCOPE
        else expected_policy
    )
    if not (
        treatment_meta.get("policy_distillation_component_ids") == expected_policy
        and treatment_meta.get("value_training_component_ids") == expected_value
        and treatment_meta.get("policy_kl_anchor_component_ids")
        == [value_axis.REPLAY_COMPONENT_ID]
        and treatment_meta.get("stored_policy_component_temperatures")
        == value_axis.bridge.production_temp.COMPONENT_TEMPERATURES
    ):
        raise CompletionError("value-axis treatment supervision contract drift")
    return treatment_meta, treatment_ref


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    manifest = _load_json(manifest_path, label="selected-dose manifest")
    stated = manifest.get("manifest_sha256")
    actual = value_axis.bridge.corrected._digest(  # noqa: SLF001
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if stated != actual:
        raise CompletionError("selected-dose manifest semantic digest drift")
    schema = manifest.get("schema_version")
    if schema == pure_soft.SCHEMA:
        kind = "PURE_SEARCH_TARGET"
        arm_id = kind
        expected_files = pure_soft.SOURCE_FILES
        claim_schema = pure_soft.CLAIM_SCHEMA
        submission_schema = pure_soft.RECEIPT_SCHEMA
    elif schema == value_axis.SCHEMA and manifest.get("arm_id") in value_axis.AXES:
        kind = "VALUE_AXIS"
        arm_id = str(manifest["arm_id"])
        expected_files = value_axis.SOURCE_FILES
        claim_schema = value_axis.CLAIM_SCHEMA
        submission_schema = value_axis.RECEIPT_SCHEMA
    else:
        raise CompletionError("unsupported selected-dose diagnostic manifest schema")
    if not (
        manifest.get("diagnostic_only") is True
        and manifest.get("promotion_eligible") is False
        and manifest.get("launch_authorized") is False
        and manifest.get("diagnostic_execution_authorized") is True
    ):
        raise CompletionError("manifest diagnostic authorization drift")
    _verify_source_binding(manifest, expected_files=expected_files)
    executor = _verify_bound_ref(manifest.get("diagnostic_executor"), label="executor")
    if executor != Path(str(manifest["source_binding"]["files"][
        pure_soft.EXECUTOR_RELATIVE_PATH
        if kind == "PURE_SEARCH_TARGET"
        else value_axis.EXECUTOR_RELATIVE_PATH
    ]["path"])).resolve(strict=True):
        raise CompletionError("diagnostic executor differs from source binding")
    source = _load_selected_source(manifest)
    root = Path(str(manifest.get("output_root", ""))).resolve(strict=True)
    source_descriptor = Path(str(source["descriptor"]["path"])).resolve(strict=True)
    if kind == "PURE_SEARCH_TARGET":
        if not (
            manifest.get("descriptor") == source["descriptor"]
            and manifest.get("only_declared_causal_delta")
            == {
                "soft_target_weight": {"source": 0.9, "treatment": 1.0},
                "played_action_hard_ce_weight": {"source": 0.1, "treatment": 0.0},
            }
        ):
            raise CompletionError("pure-target manifest causal contract drift")
        expected_command, changes = pure_soft._derive_command(  # noqa: SLF001
            source["command"], output_root=root
        )
        treatment_meta, treatment_ref = value_axis._source_descriptor_contract(  # noqa: SLF001
            source
        )[1:]
        data_path = source_descriptor
    else:
        if manifest.get("source_descriptor") != source["descriptor"]:
            raise CompletionError("value-axis source descriptor drift")
        treatment_meta, treatment_ref = _verify_value_axis_descriptor(
            manifest, source, arm_id=arm_id
        )
        data_path = Path(treatment_ref["path"])
        expected_command, changes = value_axis._derive_command(  # noqa: SLF001
            source["command"],
            axis=arm_id,
            source_descriptor=source_descriptor,
            treatment_descriptor=data_path,
            output_root=root,
        )
        if not (
            manifest.get("only_declared_causal_delta")
            == value_axis._axis_delta(arm_id)  # noqa: SLF001
            and manifest.get("matched_contract")
            == value_axis._matched_contract(arm_id)  # noqa: SLF001
        ):
            raise CompletionError("value-axis causal contract drift")
    command = manifest.get("command")
    if not (
        command == expected_command
        and manifest.get("allowlisted_command_changes") == changes
        and manifest.get("command_sha256")
        == value_axis.bridge.corrected._digest(command)  # noqa: SLF001
    ):
        raise CompletionError("manifest command is not the exact selected-dose derivation")
    trainers = [Path(value).resolve() for value in command if Path(value).name == "train_bc.py"]
    selected_trainer = Path(source["selected_geometry_trainer"]).resolve(strict=True)
    if trainers != [selected_trainer]:
        raise CompletionError("manifest command escaped selected geometry trainer")
    return {
        "manifest": manifest,
        "manifest_ref": _compact_ref(manifest_path),
        "kind": kind,
        "arm_id": arm_id,
        "claim_schema": claim_schema,
        "submission_schema": submission_schema,
        "source": source,
        "output_root": root,
        "data_path": data_path,
        "data_ref": _compact_ref(data_path),
        "treatment_meta": treatment_meta,
        "command": command,
        "selected_trainer": selected_trainer,
    }


def _systemd_command(verified: Mapping[str, Any], *, unit: str) -> list[str]:
    root = verified["output_root"]
    return [
        "sudo",
        "-n",
        "systemd-run",
        f"--unit={unit}",
        "--uid=ubuntu",
        "--gid=ubuntu",
        "--service-type=exec",
        "--collect",
        "--property=LimitNOFILE=65536",
        f"--property=WorkingDirectory={verified['source']['selected_geometry_runtime_repo']}",
        f"--property=StandardOutput=append:{root / 'stdout.log'}",
        f"--property=StandardError=append:{root / 'stderr.log'}",
        "--setenv=HOME=/home/ubuntu",
        "--setenv=PYTHONNOUSERSITE=1",
        "--",
        *verified["command"],
    ]


def _verify_submission(verified: Mapping[str, Any]) -> dict[str, Any]:
    root = verified["output_root"]
    receipt_path = root / "diagnostic-execution.receipt.json"
    receipt = _load_json(receipt_path, label="submission receipt")
    unhashed = dict(receipt)
    digest = unhashed.pop("receipt_sha256", None)
    expected_keys = {
        "schema_version",
        "diagnostic_only",
        "promotion_eligible",
        "created_at_unix_ns",
        "manifest",
        "claim",
        "unit",
        "command_sha256",
        "systemd_command_sha256",
        "systemd_stdout",
        "receipt_sha256",
    }
    if not (
        set(receipt) == expected_keys
        and receipt.get("schema_version") == verified["submission_schema"]
        and receipt.get("diagnostic_only") is True
        and receipt.get("promotion_eligible") is False
        and receipt.get("manifest") == verified["manifest_ref"]
        and receipt.get("command_sha256")
        == verified["manifest"]["command_sha256"]
        and digest == value_axis.bridge.corrected._digest(unhashed)  # noqa: SLF001
    ):
        raise CompletionError("selected-dose submission receipt drift")
    unit = receipt.get("unit")
    if not isinstance(unit, str) or not unit:
        raise CompletionError("submission unit is missing")
    claim_path = _verify_bound_ref(receipt.get("claim"), label="execution claim")
    claim = _load_json(claim_path, label="execution claim")
    claim_unhashed = dict(claim)
    claim_digest = claim_unhashed.pop("claim_sha256", None)
    if not (
        claim_path == root / "diagnostic-execution.claim.json"
        and set(claim)
        == {
            "schema_version",
            "created_at_unix_ns",
            "manifest",
            "unit",
            "claim_sha256",
        }
        and claim.get("schema_version") == verified["claim_schema"]
        and claim.get("manifest") == verified["manifest_ref"]
        and claim.get("unit") == unit
        and claim_digest
        == value_axis.bridge.corrected._digest(claim_unhashed)  # noqa: SLF001
    ):
        raise CompletionError("selected-dose execution claim drift")
    expected_systemd = _systemd_command(verified, unit=unit)
    if receipt.get("systemd_command_sha256") != value_axis.bridge.corrected._digest(  # noqa: SLF001
        expected_systemd
    ):
        raise CompletionError("submission does not bind the selected-dose command")
    return {
        "unit": unit,
        "claim": _file_ref(claim_path),
        "submission": _file_ref(receipt_path),
    }


def _report_expected(verified: Mapping[str, Any]) -> dict[str, Any]:
    arm_id = verified["arm_id"]
    return {
        "diagnostic_only": True,
        "promotion_eligible": False,
        "world_size": 8,
        "batch_size": 512,
        "effective_global_batch_size": 4096,
        "grad_accum_steps": 1,
        "max_steps": 128,
        "steps_completed": 128,
        "base_training_row_draws": 524_288,
        "total_training_row_draws": 524_288,
        "optimizer": "adam",
        "resume_optimizer": False,
        "optimizer_restored": False,
        "lr": 3e-5,
        "lr_warmup_steps": 100,
        "lr_schedule": "flat",
        "policy_loss_weight": 1.0,
        "soft_target_source": "policy",
        "soft_target_weight": 1.0 if verified["kind"] == "PURE_SEARCH_TARGET" else 0.9,
        "soft_target_temperature": 0.7,
        "value_loss_weight": 0.0 if arm_id == value_axis.VALUE_LOSS_OFF else 0.25,
        "resolved_scalar_value_loss_weight": (
            0.0 if arm_id == value_axis.VALUE_LOSS_OFF else 0.25
        ),
        "value_lr_mult": 0.3,
        "value_target_lambda": 1.0,
        "forced_action_weight": 0.0,
        "forced_row_value_weight": 1.0,
        "winner_sample_weight": 1.0,
        "loser_sample_weight": 1.0,
        "policy_kl_anchor_weight": 0.0,
        "q_loss_weight": 0.0,
        "final_vp_loss_weight": 0.0,
        "mask_hidden_info": True,
        "graph_history_features": True,
        "training_rng_rank_offset": True,
        "freeze_modules": "",
        "require_only_trainable_prefixes": "",
    }


def _verify_report(verified: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    root = verified["output_root"]
    checkpoint_path = root / "candidate.pt"
    report_path = root / "train.report.json"
    checkpoint = _file_ref(checkpoint_path)
    report_ref = _file_ref(report_path)
    report = _load_json(report_path, label="training report")
    expected = _report_expected(verified)
    drift = {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in expected.items()
        if report.get(key) != value
    }
    if drift:
        raise CompletionError(f"selected-dose report recipe/dose drift: {drift}")
    source = verified["source"]
    try:
        report_checkpoint = Path(str(report.get("checkpoint", ""))).resolve(strict=True)
        init = Path(str(report.get("init_checkpoint", ""))).resolve(strict=True)
        data = Path(str(report.get("data", ""))).resolve(strict=True)
        sentinel = Path(
            str(report.get("input_validation_game_sentinel_manifest", ""))
        ).resolve(strict=True)
    except OSError as error:
        raise CompletionError(f"training report path binding is invalid: {error}") from error
    if not (
        report_checkpoint == checkpoint_path
        and init == Path(source["initialization"]["path"])
        and report.get("init_checkpoint_sha256")
        == source["initialization"]["sha256"]
        and data == verified["data_path"]
        and sentinel == Path(source["validation_sentinel"]["path"])
        and checkpoint["sha256"] != source["initialization"]["sha256"]
    ):
        raise CompletionError("training report input/output lineage drift")
    runtime = report.get("checkout_runtime_binding")
    if not (
        isinstance(runtime, Mapping)
        and Path(str(runtime.get("trainer", ""))).resolve()
        == verified["selected_trainer"]
        and runtime.get("trainer_sha256")
        == value_axis.bridge.corrected._file_sha(verified["selected_trainer"])  # noqa: SLF001
    ):
        raise CompletionError("training report runtime trainer binding drift")
    expected_policy = list(value_axis.EXPECTED_COMPONENT_IDS)
    expected_value = (
        list(value_axis.CURRENT_COMPONENT_IDS)
        if verified["arm_id"] == value_axis.CURRENT_VALUE_SCOPE
        else expected_policy
    )
    composite = report.get("memmap_composite")
    if not (
        isinstance(composite, Mapping)
        and Path(str(composite.get("descriptor_path", ""))).resolve()
        == verified["data_path"]
        and composite.get("descriptor_file_sha256")
        == verified["data_ref"]["sha256"]
        and composite.get("component_ids") == expected_policy
        and composite.get("policy_distillation_component_ids") == expected_policy
        and composite.get("value_training_component_ids") == expected_value
        and composite.get("policy_kl_anchor_component_ids")
        == [value_axis.REPLAY_COMPONENT_ID]
    ):
        raise CompletionError("training report composite supervision drift")
    if report.get("stored_policy_component_temperatures") != (
        value_axis.bridge.production_temp.COMPONENT_TEMPERATURES
    ):
        raise CompletionError("training report component-temperature drift")
    policy_scope = report.get("policy_distillation_scope")
    value_scope = report.get("value_training_scope")
    if not (
        isinstance(policy_scope, Mapping)
        and policy_scope.get("component_ids") == expected_policy
        and isinstance(value_scope, Mapping)
        and value_scope.get("component_ids") == expected_value
    ):
        raise CompletionError("training report realized objective scope drift")
    metrics = report.get("metrics")
    matched = (
        metrics[0].get("validation_objective_matched")
        if isinstance(metrics, list)
        and len(metrics) == 1
        and isinstance(metrics[0], Mapping)
        else None
    )
    if not (
        isinstance(matched, Mapping)
        and matched.get("schema_version") == "composite-validation-measure-v2"
        and matched.get("objective_matched") is True
        and set(matched.get("components", {})) == set(expected_policy)
    ):
        raise CompletionError("training report lacks objective-matched validation")
    return checkpoint, report_ref


def _artifact_inventory(root: Path) -> dict[str, dict[str, Any]]:
    required = (
        "candidate.pt",
        "train.report.json",
        "diagnostic-execution.claim.json",
        "diagnostic-execution.receipt.json",
        "diagnostic-execution.status.jsonl",
        "stdout.log",
        "stderr.log",
    )
    optional = (
        "candidate.pt.optimizer.pt",
        "candidate.pt.training-progress.json",
        "train.report.validation_seeds.json",
        "layer_drift.audit.json",
    )
    artifacts = {name: _file_ref(root / name) for name in required}
    for name in optional:
        path = root / name
        if path.is_file():
            artifacts[name] = _file_ref(path)
    return artifacts


def build_completion(
    manifest_path: Path,
    *,
    expected_checkpoint_sha256: str = "",
    created_at_unix_ns: int | None = None,
) -> dict[str, Any]:
    verified = verify_manifest(manifest_path)
    submission = _verify_submission(verified)
    checkpoint, report = _verify_report(verified)
    if expected_checkpoint_sha256 and checkpoint["sha256"] != expected_checkpoint_sha256:
        raise CompletionError(
            "completed checkpoint SHA-256 differs from the expected artifact"
        )
    artifacts = _artifact_inventory(verified["output_root"])
    completion: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "created_at_unix_ns": (
            time.time_ns() if created_at_unix_ns is None else int(created_at_unix_ns)
        ),
        "kind": verified["kind"],
        "arm_id": verified["arm_id"],
        "manifest": verified["manifest_ref"],
        "command": verified["command"],
        "command_sha256": verified["manifest"]["command_sha256"],
        "source_temperature_manifest": verified["manifest"][
            "source_temperature_manifest"
        ],
        "selected_geometry_evidence": verified["manifest"][
            "selected_geometry_evidence"
        ],
        "initialization": verified["source"]["initialization"],
        "data": verified["data_ref"],
        "submission": submission,
        "checkpoint": checkpoint,
        "report": report,
        "artifacts": artifacts,
        "verified_recipe": _report_expected(verified),
        "only_declared_causal_delta": verified["manifest"][
            "only_declared_causal_delta"
        ],
        "expected_checkpoint_sha256": expected_checkpoint_sha256 or None,
        "completion_finalizer": _file_ref(Path(__file__)),
    }
    completion["receipt_sha256"] = _digest(completion)
    return completion


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def verify_completion(path: Path) -> dict[str, Any]:
    receipt_path = path.expanduser().resolve(strict=True)
    receipt = _load_json(receipt_path, label="completion receipt")
    unhashed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if not (
        receipt.get("schema_version") == SCHEMA
        and receipt.get("status") == STATUS
        and receipt.get("diagnostic_only") is True
        and receipt.get("promotion_eligible") is False
        and receipt.get("receipt_sha256") == _digest(unhashed)
    ):
        raise CompletionError("completion receipt schema/status/digest drift")
    finalizer = _file_ref(Path(__file__))
    if receipt.get("completion_finalizer") != finalizer:
        raise CompletionError("completion finalizer bytes drifted")
    replay = build_completion(
        Path(receipt["manifest"]["path"]),
        expected_checkpoint_sha256=str(
            receipt.get("expected_checkpoint_sha256") or ""
        ),
        created_at_unix_ns=int(receipt["created_at_unix_ns"]),
    )
    if replay != receipt:
        raise CompletionError("completion no longer replays from bound artifacts")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--manifest", type=Path, required=True)
    finalize.add_argument("--receipt", type=Path)
    finalize.add_argument("--expected-checkpoint-sha256", default="")
    verify = sub.add_parser("verify")
    verify.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "finalize":
            manifest = args.manifest.expanduser().resolve(strict=True)
            payload = build_completion(
                manifest,
                expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            )
            destination = (
                args.receipt
                if args.receipt is not None
                else Path(payload["checkpoint"]["path"]).parent / COMPLETION_NAME
            )
            _write_exclusive(destination, payload)
            result = verify_completion(destination)
        else:
            result = verify_completion(args.receipt)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        CompletionError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        pure_soft.PureSoftError,
        value_axis.ValueAxisError,
    ) as error:
        print(f"REFUSED: {error}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
