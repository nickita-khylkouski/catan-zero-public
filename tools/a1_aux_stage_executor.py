#!/usr/bin/env python3
"""Fail-closed executor boundary for A1 pointer WARMUP and GEOMETRY.

The central coordinator owns issuance and append-only stage state.  This module
is the only code allowed to turn a published WARMUP/GEOMETRY authority into
learner work.  In particular, an inline recipe or an operator-provided stage
label is never authority: the immutable coordinator artifact is stable-read,
hashed, and canonically replayed before an optimizer or autograd probe may be
constructed.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import a1_aux_pair_coordinator as coordinator  # noqa: E402
from tools import a1_one_dose_train as one_dose  # noqa: E402
from tools import train_bc  # noqa: E402


class StageExecutorError(RuntimeError):
    """Refusal at the WARMUP/GEOMETRY execution trust boundary."""


def _file_identity(value) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_read(path: Path) -> tuple[Path, bytes, str]:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        resolved = lexical.resolve(strict=True)
        before = resolved.stat()
    except OSError as error:
        raise StageExecutorError(f"cannot inspect executor authority: {error}") from error
    if (
        lexical != resolved
        or path.expanduser().is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_mode & 0o222
    ):
        raise StageExecutorError(
            "stage executor authority must be one canonical immutable file"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        opened_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = resolved.stat()
    if not (
        _file_identity(before)
        == _file_identity(opened_before)
        == _file_identity(opened_after)
        == _file_identity(after)
    ):
        raise StageExecutorError("stage executor authority changed during stable read")
    payload = b"".join(chunks)
    return resolved, payload, "sha256:" + hashlib.sha256(payload).hexdigest()


def verify_stage_executor_authority(
    path: Path, *, expected_file_sha256: str, expected_stage: str
) -> dict[str, Any]:
    """Authenticate and replay one centrally issued stage authority."""

    if expected_stage not in {"WARMUP", "GEOMETRY"}:
        raise StageExecutorError("stage must be exactly WARMUP or GEOMETRY")
    if not isinstance(expected_file_sha256, str) or len(expected_file_sha256) != 71:
        raise StageExecutorError("stage executor authority SHA is malformed")
    resolved, raw, file_sha = _stable_read(path)
    if file_sha != expected_file_sha256:
        raise StageExecutorError("stage executor authority file digest drift")
    try:
        stable_authority = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise StageExecutorError(f"stage executor authority is not JSON: {error}") from error
    try:
        published = coordinator.verify_published_executor_authority(resolved)
    except coordinator.CoordinatorError as error:
        raise StageExecutorError(f"stage coordinator replay refused: {error}") from error
    authority = published.get("authority")
    expected_schema = f"a1-aux-{expected_stage.lower()}-executor-authority-v1"
    if (
        published.get("path") != str(resolved)
        or published.get("file_sha256") != file_sha
        or authority != stable_authority
        or not isinstance(authority, dict)
        or authority.get("schema_version") != expected_schema
        or authority.get("stage") != expected_stage
    ):
        raise StageExecutorError("stage executor authority replay/projection drift")
    return published


def bind_stage_inputs(
    published: Mapping[str, Any],
    *,
    descriptor_sha256: str,
    payload_inventory_sha256: str,
    initializer_sha256: str,
) -> dict[str, Any]:
    """Project exact data/model bytes before optimizer or gradient construction."""

    authority = published.get("authority")
    if not isinstance(authority, dict):
        raise StageExecutorError("published stage authority has no authority object")
    science = authority.get("portable_science_identity")
    if not isinstance(science, dict):
        raise StageExecutorError("stage authority has no portable science identity")
    stage = authority.get("stage")
    composite = science.get("composite")
    pointer = science.get("pointer_upgrade_authority")
    if not isinstance(composite, dict) or not isinstance(pointer, dict):
        raise StageExecutorError("stage authority lacks composite/pointer identity")
    expected_initializer = (
        pointer.get("upgraded_initializer_sha256")
        if stage == "WARMUP"
        else authority.get("warmup_terminal", {})
        .get("result", {})
        .get("warmed_checkpoint_sha256")
    )
    if (
        descriptor_sha256 != composite.get("descriptor_sha256")
        or payload_inventory_sha256 != composite.get("payload_inventory_sha256")
        or initializer_sha256 != expected_initializer
    ):
        raise StageExecutorError("stage descriptor/payload/initializer byte drift")
    allocation = coordinator.verify_allocation(dict(authority.get("allocation", {})))
    return {
        "schema_version": "a1-aux-stage-execution-binding-v1",
        "stage": stage,
        "executor_authority_path": published["path"],
        "executor_authority_file_sha256": published["file_sha256"],
        "executor_authority_state_sha256": authority["state_sha256"],
        "descriptor_sha256": descriptor_sha256,
        "payload_inventory_sha256": payload_inventory_sha256,
        "initializer_sha256": initializer_sha256,
        "allocation": allocation,
        "optimizer_construction_authorized": stage == "WARMUP",
        "gradient_probe_authorized": stage == "GEOMETRY",
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _replace_option(command: list[str], flag: str, value: object) -> None:
    matches = [index for index, token in enumerate(command) if token == flag]
    if len(matches) != 1 or matches[0] + 1 >= len(command):
        raise StageExecutorError(f"stage command has no unique {flag}")
    command[matches[0] + 1] = str(value)


def _stage_recipe(authority: Mapping[str, Any]) -> dict[str, Any]:
    science = authority["portable_science_identity"]
    recipe = copy.deepcopy(science["effective_recipe"])
    if authority["stage"] == "WARMUP":
        warmup = science["warmup_recipe"]
        recipe.update(
            {
                "epochs": 1,
                "max_steps": warmup["max_steps"],
                "batch_size": warmup["local_batch_size"],
                "global_batch_size": warmup["global_batch_size"],
                "grad_accum_steps": warmup["grad_accum_steps"],
                "optimizer": "adam",
                "lr": warmup["lr"],
                "lr_warmup_steps": warmup["lr_warmup_steps"],
                "lr_schedule": warmup["lr_schedule"],
                "weight_decay": warmup["weight_decay"],
                "amp": warmup["amp"],
                "seed": warmup["seed"],
                "sampler_seed": warmup["sampler_seed"],
                "training_rng_rank_offset": warmup[
                    "training_rng_rank_offset"
                ],
                "aux_subgoal_loss_weight": warmup[
                    "aux_subgoal_loss_weight"
                ],
                **warmup["main_objective_coefficients"],
            }
        )
    else:
        recipe["aux_subgoal_loss_weight"] = 1.0
    return recipe


def _stage_training_binding(
    published: Mapping[str, Any],
    *,
    descriptor: Path,
    descriptor_sha256: str,
    payload_inventory_sha256: str,
    initializer: Path,
    initializer_sha256: str,
    checkpoint: Path,
    report: Path,
    probe_manifest: Path | None,
) -> dict[str, Any]:
    authority = published["authority"]
    stage = authority["stage"]
    manifest_ref = None
    if stage == "GEOMETRY":
        if probe_manifest is None:
            raise StageExecutorError("GEOMETRY requires its preregistered probe manifest")
        manifest = probe_manifest.expanduser().resolve(strict=True)
        if manifest.is_symlink() or not manifest.is_file():
            raise StageExecutorError("GEOMETRY probe manifest must be a regular file")
        manifest_ref = {"path": str(manifest), "file_sha256": _file_sha256(manifest)}
        rule = authority["portable_science_identity"]["selector_rule"]
        if manifest_ref["file_sha256"] != rule["probe_manifest_sha256"]:
            raise StageExecutorError("GEOMETRY probe manifest differs from selector rule")
    elif probe_manifest is not None:
        raise StageExecutorError("WARMUP may not bind a GEOMETRY probe manifest")
    return {
        "schema_version": "a1-aux-stage-training-binding-v1",
        "stage": stage,
        "experiment_id": authority["experiment_id"],
        "executor_authority_path": published["path"],
        "executor_authority_file_sha256": published["file_sha256"],
        "executor_authority_state_sha256": authority["state_sha256"],
        "descriptor_path": str(descriptor),
        "descriptor_sha256": descriptor_sha256,
        "payload_inventory_sha256": payload_inventory_sha256,
        "initializer_path": str(initializer),
        "initializer_sha256": initializer_sha256,
        "output_report": str(report.expanduser().resolve(strict=False)),
        "output_checkpoint": str(checkpoint.expanduser().resolve(strict=False)),
        "probe_manifest": manifest_ref,
    }


def build_stage_train_command(
    published: Mapping[str, Any],
    *,
    python: Path,
    descriptor: Path,
    initializer: Path,
    checkpoint: Path,
    report: Path,
    probe_manifest: Path | None,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    """Render the only train command accepted by the authenticated stage."""

    descriptor = descriptor.expanduser().resolve(strict=True)
    initializer = initializer.expanduser().resolve(strict=True)
    descriptor_meta = train_bc._preflight_memmap_composite_descriptor(  # noqa: SLF001
        descriptor
    )
    descriptor_sha = str(descriptor_meta["descriptor_file_sha256"])
    payload_sha = str(descriptor_meta["payload_inventory_sha256"])
    initializer_sha = _file_sha256(initializer)
    bind_stage_inputs(
        published,
        descriptor_sha256=descriptor_sha,
        payload_inventory_sha256=payload_sha,
        initializer_sha256=initializer_sha,
    )
    authority = published["authority"]
    recipe = _stage_recipe(authority)
    verified = {
        "recipe": recipe,
        "producer": {"path": str(initializer), "sha256": initializer_sha},
        "architecture_initializer": {
            "path": str(initializer),
            "sha256": initializer_sha,
        },
        "data_path": descriptor,
        "data_kind": "production_composite_v2",
        "payload_inventory_sha256": payload_sha,
        "function_preserving_upgrade": {
            "module": one_dose.AUX_REGULARIZATION_MODULE
        },
    }
    command = one_dose._build_direct_train_command(  # noqa: SLF001
        verified,
        python=python.expanduser().resolve(strict=True),
        checkpoint=checkpoint.expanduser().resolve(strict=False),
        report=report.expanduser().resolve(strict=False),
    )
    binding = _stage_training_binding(
        published,
        descriptor=descriptor,
        descriptor_sha256=descriptor_sha,
        payload_inventory_sha256=payload_sha,
        initializer=initializer,
        initializer_sha256=initializer_sha,
        checkpoint=checkpoint,
        report=report,
        probe_manifest=probe_manifest,
    )
    command.extend(
        [
            one_dose.EVENT_HISTORY_ACK_FLAG,
            payload_sha,
            one_dose.EVENT_HISTORY_CROP_FLAG,
            "--a1-aux-stage-binding-json",
            json.dumps(binding, sort_keys=True, separators=(",", ":")),
            "--a1-aux-stage-executor-authority",
            published["path"],
            "--a1-aux-stage-executor-authority-sha256",
            published["file_sha256"],
        ]
    )
    if authority["stage"] == "WARMUP":
        command.extend(
            [
                "--require-only-trainable-prefixes",
                ",".join(coordinator.POINTER_TRAINABLE_PREFIXES),
            ]
        )
    command = one_dose._topologize_train_command(command, world_size=8)  # noqa: SLF001
    return command, binding, descriptor_meta


def _load_json_file(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StageExecutorError(f"cannot load {where}: {error}") from error
    if not isinstance(value, dict):
        raise StageExecutorError(f"{where} is not an object")
    return value


def _tensor_sha256(name: str, tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(
        header + b"\0" + value.numpy().tobytes()
    ).hexdigest()


def _warmup_tensor_evidence(
    initializer: Path, checkpoint: Path
) -> tuple[list[str], str]:
    import torch

    before = torch.load(initializer, map_location="cpu", weights_only=False)
    after = torch.load(checkpoint, map_location="cpu", weights_only=False)
    before_model = before.get("model") if isinstance(before, Mapping) else None
    after_model = after.get("model") if isinstance(after, Mapping) else None
    if not isinstance(before_model, Mapping) or not isinstance(after_model, Mapping):
        raise StageExecutorError("WARMUP checkpoint model state is malformed")
    if set(before_model) != set(after_model):
        raise StageExecutorError("WARMUP changed checkpoint parameter keys")
    changed: list[str] = []
    inherited: list[dict[str, str]] = []
    for name in sorted(before_model):
        left, right = before_model[name], after_model[name]
        if (
            not torch.is_tensor(left)
            or not torch.is_tensor(right)
            or left.dtype != right.dtype
            or left.layout != right.layout
            or tuple(left.shape) != tuple(right.shape)
        ):
            raise StageExecutorError(f"WARMUP parameter metadata drift: {name}")
        if torch.equal(left, right):
            inherited.append({"name": name, "tensor_sha256": _tensor_sha256(name, left)})
        else:
            changed.append(name)
    expected = sorted(
        name
        for name in before_model
        if any(name.startswith(prefix) for prefix in coordinator.POINTER_TRAINABLE_PREFIXES)
    )
    if changed != expected or not expected:
        raise StageExecutorError(
            "WARMUP changed-parameter set differs from the exact pointer heads: "
            f"expected={expected} changed={changed}"
        )
    return changed, _canonical_sha256(inherited)


def _warmup_main_output_max_diff(initializer: Path, checkpoint: Path) -> float:
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from tools.f69_upgrade_checkpoint_config import _verify_forward_identical

    before = EntityGraphPolicy.load(str(initializer), device="cpu")
    after = EntityGraphPolicy.load(str(checkpoint), device="cpu")
    value = float(_verify_forward_identical(before, after, "cpu"))
    if value != 0.0:
        raise StageExecutorError(
            f"WARMUP changed a main model output: max_diff={value}"
        )
    return value


def _discard_optimizer_sidecar(path: Path) -> str:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    discarded = lexical.with_name(f"{lexical.name}.discarded")
    if lexical.is_symlink() or discarded.is_symlink():
        raise StageExecutorError("WARMUP optimizer evidence may not be a symlink")
    if lexical.exists() and discarded.exists():
        raise StageExecutorError("WARMUP has both live and discarded optimizer bytes")
    if lexical.exists():
        if not lexical.is_file():
            raise StageExecutorError("WARMUP optimizer sidecar is not regular")
        digest = _file_sha256(lexical)
        os.replace(lexical, discarded)
        os.chmod(discarded, 0o444)
    elif discarded.exists():
        info = discarded.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222:
            raise StageExecutorError(
                "WARMUP discarded optimizer evidence is not immutable"
            )
        digest = _file_sha256(discarded)
    else:
        raise StageExecutorError("WARMUP optimizer sidecar evidence is missing")
    directory = os.open(discarded.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if lexical.exists() or not discarded.is_file():
        raise StageExecutorError("WARMUP optimizer sidecar was not discarded")
    return digest


def _canonical_output_namespace(checkpoint: Path, report: Path) -> dict[str, str]:
    def _canonical(path: Path, where: str) -> Path:
        lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
        if lexical.is_symlink():
            raise StageExecutorError(f"{where} output may not be a symlink")
        try:
            parent = lexical.parent.resolve(strict=True)
        except OSError as error:
            raise StageExecutorError(
                f"{where} output parent is unavailable: {error}"
            ) from error
        if lexical.parent != parent:
            raise StageExecutorError(f"{where} output parent may not be a symlink")
        return parent / lexical.name

    checkpoint_path = _canonical(checkpoint, "checkpoint")
    report_path = _canonical(report, "report")
    optimizer_path = _canonical(
        Path(str(checkpoint_path) + ".optimizer.pt"), "optimizer sidecar"
    )
    outputs = [checkpoint_path, report_path, optimizer_path]
    if len(set(outputs)) != len(outputs):
        raise StageExecutorError("stage output namespace contains aliases")
    return {
        "checkpoint": str(checkpoint_path),
        "report": str(report_path),
        "optimizer_sidecar": str(optimizer_path),
    }


def _fresh_output_namespace(
    *,
    coordinator_root: Path,
    published: Mapping[str, Any],
    descriptor: Path,
    initializer: Path,
    checkpoint: Path,
    report: Path,
    probe_manifest: Path | None,
) -> dict[str, str]:
    """Resolve three fresh outputs and prove they cannot clobber authority."""

    root = coordinator_root.expanduser().resolve(strict=True)
    inputs = [
        Path(str(published["path"])).expanduser().resolve(strict=True),
        descriptor.expanduser().resolve(strict=True),
        initializer.expanduser().resolve(strict=True),
    ]
    if probe_manifest is not None:
        inputs.append(probe_manifest.expanduser().resolve(strict=True))
    namespace = _canonical_output_namespace(checkpoint, report)
    for where, raw in namespace.items():
        path = Path(raw)
        if path.exists() or path.is_symlink():
            raise StageExecutorError(
                f"{where} output must be a fresh canonical non-symlink path"
            )
        try:
            path.relative_to(root)
        except ValueError:
            pass
        else:
            raise StageExecutorError(
                f"{where} output may not enter the coordinator authority tree"
            )
        if path in inputs:
            raise StageExecutorError(f"{where} output aliases an immutable input")
    return namespace


def _verify_live_allocation(
    allocation: Mapping[str, Any],
    *,
    runtime_probe=None,
) -> dict[str, Any]:
    """Replay exact host/machine/GPU UUID/PCI identity immediately prelaunch."""

    probe = runtime_probe or coordinator.scientific_evidence._local_runtime_report  # noqa: SLF001
    try:
        report = probe(_REPO_ROOT)
        coordinator._verify_allocation_matches_native_report(  # noqa: SLF001
            allocation, report
        )
    except (
        OSError,
        ValueError,
        coordinator.CoordinatorError,
        coordinator.scientific_evidence.EvidenceError,
    ) as error:
        raise StageExecutorError(f"live B200 allocation replay failed: {error}") from error
    return report


def _verify_execution_commitment(
    commitment: Mapping[str, Any],
    *,
    published: Mapping[str, Any],
    binding: Mapping[str, Any],
    checkpoint: Path,
    report: Path,
) -> dict[str, Any]:
    try:
        value = coordinator._verify_sealed(  # noqa: SLF001
            commitment, "stage execution commitment"
        )
    except coordinator.CoordinatorError as error:
        raise StageExecutorError(f"stage execution commitment refused: {error}") from error
    outputs = value.get("output_namespace")
    expected_outputs = {
        "checkpoint": str(checkpoint.expanduser().resolve(strict=False)),
        "report": str(report.expanduser().resolve(strict=False)),
        "optimizer_sidecar": str(
            Path(str(checkpoint.expanduser().resolve(strict=False)) + ".optimizer.pt")
        ),
    }
    if (
        value.get("schema_version") != "a1-stage-execution-commitment-v1"
        or value.get("stage") != published["authority"]["stage"]
        or value.get("executor_authority_file_sha256") != published["file_sha256"]
        or value.get("executor_authority_state_sha256")
        != published["authority"]["state_sha256"]
        or value.get("training_binding_sha256") != _canonical_sha256(binding)
        or outputs != expected_outputs
        or value.get("output_namespace_sha256") != _canonical_sha256(outputs)
    ):
        raise StageExecutorError("stage execution commitment/output binding drift")
    return value


def complete_stage_from_outputs(
    *,
    coordinator_root: Path,
    published: Mapping[str, Any],
    commitment: Mapping[str, Any],
    binding: Mapping[str, Any],
    initializer: Path,
    checkpoint: Path,
    report: Path,
) -> dict[str, Any]:
    """Derive a coordinator terminal only from measured child artifacts."""

    authority = published["authority"]
    stage = authority["stage"]
    experiment_id = authority["experiment_id"]
    report_lexical = Path(os.path.abspath(os.fspath(report.expanduser())))
    if report_lexical.is_symlink():
        raise StageExecutorError("stage child report may not be a symlink")
    report = report_lexical.resolve(strict=True)
    if report != report_lexical:
        raise StageExecutorError("stage child report path is not canonical")
    _verify_execution_commitment(
        commitment,
        published=published,
        binding=binding,
        checkpoint=checkpoint,
        report=report,
    )
    if report.is_symlink() or not report.is_file():
        raise StageExecutorError("stage child report is not a regular file")
    report_value = _load_json_file(report, where=f"{stage} child report")
    report_sha = _file_sha256(report)
    origin_sha = _file_sha256(Path(__file__).resolve(strict=True))
    if stage == "WARMUP" and report_value.get("a1_aux_stage_binding") != binding:
        raise StageExecutorError("WARMUP child report lost its executor authority")
    if stage == "WARMUP":
        warmup = authority["portable_science_identity"]["warmup_recipe"]
        realized = report_value.get("a1_realized_aux_stage_sample_order")
        checkpoint_lexical = Path(
            os.path.abspath(os.fspath(checkpoint.expanduser()))
        )
        initializer_lexical = Path(
            os.path.abspath(os.fspath(initializer.expanduser()))
        )
        if checkpoint_lexical.is_symlink() or initializer_lexical.is_symlink():
            raise StageExecutorError("WARMUP checkpoint inputs/outputs may not be symlinks")
        checkpoint = checkpoint_lexical.resolve(strict=True)
        initializer = initializer_lexical.resolve(strict=True)
        if checkpoint != checkpoint_lexical or initializer != initializer_lexical:
            raise StageExecutorError("WARMUP checkpoint path is not canonical")
        if (
            not isinstance(realized, dict)
            or realized.get("sample_order_sha256") != warmup["sample_order_sha256"]
            or realized.get("sample_dose") != warmup["sample_dose"]
            or report_value.get("steps_completed") != warmup["optimizer_steps"]
            or report_value.get("base_training_row_draws") != warmup["sample_dose"]
            or report_value.get("optimizer_restored") is not False
            or report_value.get("public_award_feature_contract")
            != "authoritative_v1"
            or report_value.get("require_only_trainable_prefixes")
            != ",".join(coordinator.POINTER_TRAINABLE_PREFIXES)
        ):
            raise StageExecutorError("WARMUP child dose/order/optimizer report drift")
        changed, inherited_sha = _warmup_tensor_evidence(initializer, checkpoint)
        changed_sha = _canonical_sha256(changed)
        pointer = authority["portable_science_identity"][
            "pointer_upgrade_authority"
        ]
        if changed_sha != pointer["new_parameter_set_sha256"]:
            raise StageExecutorError("WARMUP changed-parameter identity drift")
        main_diff = _warmup_main_output_max_diff(initializer, checkpoint)
        optimizer_sha = _discard_optimizer_sidecar(
            Path(str(checkpoint) + ".optimizer.pt")
        )
        result = {
            "schema_version": "a1-aux-pointer-warmup-result-v1",
            "status": "complete",
            "sampled_rows": warmup["sample_dose"],
            "optimizer_steps": warmup["optimizer_steps"],
            "input_initializer_sha256": _file_sha256(initializer),
            "warmed_checkpoint_sha256": _file_sha256(checkpoint),
            "optimizer_sidecar_sha256": optimizer_sha,
            "optimizer_sidecar_discarded_for_joint": True,
            "changed_parameter_prefixes": list(
                coordinator.POINTER_TRAINABLE_PREFIXES
            ),
            "changed_parameter_set_sha256": changed_sha,
            "inherited_parameter_identity_sha256": inherited_sha,
            "inherited_parameters_bit_identical": True,
            "main_output_max_diff": main_diff,
            "report_sha256": report_sha,
            "origin_tool_sha256": origin_sha,
        }
        try:
            terminal = coordinator.complete_warmup(
                coordinator_root, experiment_id, result=result
            )
        except coordinator.CoordinatorError as error:
            raise StageExecutorError(
                f"coordinator refused measured WARMUP terminal: {error}"
            ) from error
        os.chmod(checkpoint, 0o444)
        os.chmod(report, 0o444)
        return terminal

    if report_value.get("schema_version") != "a1-aux-gradient-geometry-child-report-v1":
        raise StageExecutorError("GEOMETRY child report schema drift")
    if report_value.get("stage_binding") != binding:
        raise StageExecutorError("GEOMETRY child report lost its executor authority")
    manifest = report_value.get("probe_manifest")
    batches = report_value.get("per_batch_geometry")
    rng_transactions = report_value.get("rng_transactions_by_rank")
    rule = authority["portable_science_identity"]["selector_rule"]
    if checkpoint.exists() or Path(str(checkpoint) + ".optimizer.pt").exists():
        raise StageExecutorError("GEOMETRY wrote forbidden model/optimizer output")
    if (
        not isinstance(manifest, dict)
        or not isinstance(batches, list)
        or report_value.get("model_state_before_sha256")
        != report_value.get("model_state_after_sha256")
        or report_value.get("optimizer_constructed") is not False
        or report_value.get("optimizer_steps") != 0
        or report_value.get("persistent_state_mutated") is not False
        or len(batches) != rule["probe_batches"]
    ):
        raise StageExecutorError("GEOMETRY child mutation/dose evidence drift")
    if not isinstance(rng_transactions, list) or len(rng_transactions) != 8:
        raise StageExecutorError(
            "GEOMETRY child must report one RNG transaction per DDP rank"
        )
    try:
        for rank, transaction in enumerate(rng_transactions):
            verified_transaction = coordinator._verify_geometry_rng_transaction(
                transaction
            )
            if (
                verified_transaction["before"]["cuda_device"] != rank
                or verified_transaction["after_probe"]["cuda_device"] != rank
                or verified_transaction["after_restore"]["cuda_device"] != rank
            ):
                raise coordinator.CoordinatorError(
                    "RNG transaction rank/device binding drift"
                )
    except coordinator.CoordinatorError as error:
        raise StageExecutorError(
            f"GEOMETRY child RNG isolation evidence drift: {error}"
        ) from error
    parameter_set = rule["shared_parameter_set_sha256"]
    if any(
        not isinstance(batch, dict)
        or batch.get("batch_index") != index
        or batch.get("shared_parameter_set_sha256") != parameter_set
        for index, batch in enumerate(batches)
    ):
        raise StageExecutorError("GEOMETRY per-batch parameter surface drift")
    evidence = {
        "schema_version": "a1-aux-gradient-geometry-evidence-v1",
        "status": "complete",
        "warmed_checkpoint_sha256": binding["initializer_sha256"],
        "probe_manifest_sha256": binding["probe_manifest"]["file_sha256"],
        "probe_sampler_seed": manifest["sampler_seed"],
        "probe_row_order_sha256": manifest["probe_row_order_sha256"],
        "probe_batches": manifest["probe_batches"],
        "probe_batch_size": manifest["local_batch_size"],
        "shared_parameter_set_sha256": parameter_set,
        "batch_shared_parameter_set_sha256": [
            batch["shared_parameter_set_sha256"] for batch in batches
        ],
        "per_batch_geometry": copy.deepcopy(batches),
        "same_forward_graph": True,
        "global_ddp_aggregation": True,
        "optimizer_steps": 0,
        "persistent_state_mutated": False,
        "rng_transactions_by_rank": copy.deepcopy(rng_transactions),
        "report_sha256": report_sha,
        "origin_tool_sha256": origin_sha,
    }
    try:
        terminal = coordinator.complete_geometry(
            coordinator_root, experiment_id, evidence=evidence
        )
    except coordinator.CoordinatorError as error:
        raise StageExecutorError(
            f"coordinator refused measured GEOMETRY terminal: {error}"
        ) from error
    os.chmod(report, 0o444)
    return terminal


def execute_stage(
    *,
    coordinator_root: Path,
    published: Mapping[str, Any],
    python: Path,
    descriptor: Path,
    initializer: Path,
    checkpoint: Path,
    report: Path,
    probe_manifest: Path | None = None,
    runner=subprocess.run,
    runtime_probe=None,
    gpu_probe=one_dose._probe_b200,
    gpu_lock=one_dose._physical_gpu_lock,
) -> dict[str, Any]:
    command, binding, _descriptor_meta = build_stage_train_command(
        published,
        python=python,
        descriptor=descriptor,
        initializer=initializer,
        checkpoint=checkpoint,
        report=report,
        probe_manifest=probe_manifest,
    )
    allocation = coordinator.verify_allocation(
        dict(published["authority"]["allocation"])
    )
    gpus = list(allocation["physical_gpu_indices"])
    try:
        environment = one_dose._child_environment(gpus)  # noqa: SLF001
    except one_dose.ExecutorError as error:
        raise StageExecutorError(f"cannot construct sealed child environment: {error}") from error
    with ExitStack() as locks:
        try:
            for gpu in gpus:
                locks.enter_context(gpu_lock(gpu))
            _verify_live_allocation(allocation, runtime_probe=runtime_probe)
            for gpu in gpus:
                gpu_probe(gpu)
            one_dose._raise_nofile_limit()  # noqa: SLF001
        except one_dose.ExecutorError as error:
            raise StageExecutorError(f"B200 ownership/probe refused: {error}") from error
        outputs = _canonical_output_namespace(checkpoint, report)
        commitment_filename = (
            "17-warmup-execution-commitment.json"
            if published["authority"]["stage"] == "WARMUP"
            else "37-geometry-execution-commitment.json"
        )
        try:
            existing_commitment = coordinator._artifact(  # noqa: SLF001
                coordinator_root,
                published["authority"]["experiment_id"],
                commitment_filename,
                required=False,
            )
        except coordinator.CoordinatorError as error:
            raise StageExecutorError(
                f"cannot inspect prior stage execution commitment: {error}"
            ) from error
        output_paths = [Path(path) for path in outputs.values()]
        output_paths.append(
            Path(outputs["optimizer_sidecar"] + ".discarded")
        )
        resume_outputs = any(path.exists() or path.is_symlink() for path in output_paths)
        if resume_outputs and existing_commitment is None:
            raise StageExecutorError(
                "stage outputs exist without a prelaunch execution commitment"
            )
        if not resume_outputs:
            outputs = _fresh_output_namespace(
                coordinator_root=coordinator_root,
                published=published,
                descriptor=descriptor,
                initializer=initializer,
                checkpoint=checkpoint,
                report=report,
                probe_manifest=probe_manifest,
            )
        try:
            commitment = coordinator.commit_stage_execution(
                coordinator_root.expanduser().resolve(strict=True),
                published["authority"]["experiment_id"],
                stage=published["authority"]["stage"],
                command=command,
                environment=environment,
                output_namespace=outputs,
                training_binding=binding,
            )
        except coordinator.CoordinatorError as error:
            raise StageExecutorError(
                f"stage execution commitment refused: {error}"
            ) from error
        if resume_outputs:
            return complete_stage_from_outputs(
                coordinator_root=coordinator_root.expanduser().resolve(strict=True),
                published=published,
                commitment=commitment,
                binding=binding,
                initializer=initializer,
                checkpoint=checkpoint,
                report=report,
            )
        try:
            runner(command, cwd=str(_REPO_ROOT), env=environment, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise StageExecutorError(
                f"{published['authority']['stage']} learner execution failed: {error}"
            ) from error
        return complete_stage_from_outputs(
            coordinator_root=coordinator_root.expanduser().resolve(strict=True),
            published=published,
            commitment=commitment,
            binding=binding,
            initializer=initializer,
            checkpoint=checkpoint,
            report=report,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("WARMUP", "GEOMETRY"), required=True)
    parser.add_argument("--executor-authority", type=Path, required=True)
    parser.add_argument("--executor-authority-sha256", required=True)
    parser.add_argument("--coordinator-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, default=None)
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the authenticated command without executing it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        published = verify_stage_executor_authority(
            args.executor_authority,
            expected_file_sha256=args.executor_authority_sha256,
            expected_stage=args.stage,
        )
        if args.print_command:
            command, binding, _meta = build_stage_train_command(
                published,
                python=args.python,
                descriptor=args.descriptor,
                initializer=args.initializer,
                checkpoint=args.checkpoint,
                report=args.report,
                probe_manifest=args.probe_manifest,
            )
            print(json.dumps({"binding": binding, "command": command}, sort_keys=True))
            return 0
        terminal = execute_stage(
            coordinator_root=args.coordinator_root,
            published=published,
            python=args.python,
            descriptor=args.descriptor,
            initializer=args.initializer,
            checkpoint=args.checkpoint,
            report=args.report,
            probe_manifest=args.probe_manifest,
        )
    except (StageExecutorError, coordinator.CoordinatorError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(terminal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
