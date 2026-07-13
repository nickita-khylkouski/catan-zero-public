#!/usr/bin/env python3
"""Seal the selected-dose D6 -> target-gather composition diagnostic.

This is deliberately not another gather commissioning recipe.  It starts from
the authenticated mature D6 checkpoint, proves that the supplied treatment is
an output-identical gather-only upgrade of those exact bytes, and then exposes
only ``target_gather_proj`` to one already-selected TEMP row dose.  The mature
D6 network remains frozen.  Training-time D6 augmentation (including event
action-id relabeling) stays enabled, so the experiment answers one question:
does the action-local gather residual add signal on top of the known-positive
D6 parent?

Preparation and verification are non-mutating.  Execution is explicit,
diagnostic-only, one-shot, and uses the current sealed trainer checkout.
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
from tools import a1_topology_gather_arm as gather  # noqa: E402


SCHEMA = "a1-d6-gather-composition-arm-v1"
RECEIPT_SCHEMA = "a1-d6-gather-composition-execution-receipt-v1"
STATUS_SCHEMA = "a1-d6-gather-composition-execution-status-v1"
CLAIM_SCHEMA = "a1-d6-gather-composition-execution-claim-v1"
EXECUTOR_RELATIVE_PATH = "tools/a1_d6_gather_composition_arm.py"
COMPLETION_RELATIVE_PATH = "tools/a1_d6_gather_composition_completion.py"

D6_PARENT_SHA256 = (
    "sha256:761135ead3e9ec2d3b2816e2bc0b4fcd1fda1b2f897115e46295ed9198a1d28b"
)
D6_REPORT_SHA256 = (
    "sha256:dc360a97c1d6659684483deeb47295b9d48f4042799d64ae3cded3ad4818383b"
)
D6_PROGRESS_SHA256 = (
    "sha256:f56ce788dbc31d51cd250a55843fde36a20fcb9021c07f63355a5cf7ee881f62"
)
D6_GATHER_INIT_SHA256 = (
    "sha256:015be3463b424d5694fd459c819d677fb1f7a2b1aaf590101bdc403e2411858d"
)
D6_SHORT_PARENT_SHA256 = (
    "sha256:9dd1d261a39d7b04713505a301097faf18e84e8a3508b4abb92a8b964f7ab921"
)
D6_SHORT_REPORT_SHA256 = (
    "sha256:42b8f620b2d22edffd4e0d223052f0e5873c48de4b3cf8f037c53af0b08cdae5"
)
D6_SHORT_PROGRESS_SHA256 = (
    "sha256:9e2019557268281144bc7b06cece2831fe3e3abe5fdf9aea3ab6d0ee32b72492"
)
D6_SHORT_GATHER_INIT_SHA256 = (
    "sha256:14f0a8634d61afccea8eade03f4bb40304ed5e68729d1fda85bb28d2ab1708ef"
)
D6_F7_PARENT_SHA256 = (
    "sha256:f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4"
)

WORLD_SIZE = 8
SELECTED_LOCAL_BATCH_SIZE = 512
SELECTED_GLOBAL_BATCH_SIZE = WORLD_SIZE * SELECTED_LOCAL_BATCH_SIZE
SELECTED_OPTIMIZER_STEPS = 128
LOCAL_BATCH_SIZE = 64
GLOBAL_BATCH_SIZE = WORLD_SIZE * LOCAL_BATCH_SIZE
OPTIMIZER_STEPS = 1024
GLOBAL_ROW_DOSE = GLOBAL_BATCH_SIZE * OPTIMIZER_STEPS
ACTION_MODULE_LR_MULT = 4.0
VALUE_LR_MULT = 1.0
FREEZE_MODULES = gather.FREEZE_MODULES
TRAINABLE_PREFIX = gather.TRAINABLE_PREFIX

SOURCE_FILES = (
    EXECUTOR_RELATIVE_PATH,
    COMPLETION_RELATIVE_PATH,
    "tools/a1_topology_gather_arm.py",
    "tools/a1_corrected_policy_arm.py",
    "tools/a1_corrected_policy_arm_execute.py",
    "tools/a1_production_temperature_replication.py",
    "tools/a1_production_l1_rerun.py",
    "tools/f69_upgrade_checkpoint_config.py",
    "tools/audit_memmap_architecture_targets.py",
    "tools/train_bc.py",
    "tools/mixed_memmap_corpus.py",
    "src/catan_zero/rl/hex_symmetry.py",
    "src/catan_zero/rl/entity_token_policy.py",
    "src/catan_zero/rl/entity_token_features.py",
) + (
    ("tools/a1_learner_dose_contract.py",)
    if hasattr(gather.production_temp, "LEGACY_MANIFEST_SCHEMA")
    else ()
)


class CompositionArmError(RuntimeError):
    """The request is not the exact selected-dose D6+gather diagnostic."""


def _d6_parent_profiles() -> dict[str, dict[str, Any]]:
    """Return the immutable, evidence-bearing D6 parents allowed by this arm.

    The full historical D6 model was the first composition parent.  The later
    matched panel showed that the independently initialized short-dose D6 model
    beat it 73-55, so inheriting the full-dose parent would preserve a known
    over-training confound.  Both profiles remain replayable, but an input is
    selected only by its exact checkpoint/report/progress bytes; callers cannot
    describe an arbitrary model as "D6" with a flag.
    """

    return {
        D6_PARENT_SHA256: {
            "parent_profile": "historical_full_d6",
            "report_sha256": D6_REPORT_SHA256,
            "progress_sha256": D6_PROGRESS_SHA256,
            "gather_init_sha256": D6_GATHER_INIT_SHA256,
            "optimizer_steps": 1024,
            "training_row_draws": 4_194_304,
            "symmetry_rng_provenance": (
                "historical_single_stream_receipt; treatment must use current "
                "rank-distinct SeedSequence streams"
            ),
            "rank_distinct_symmetry_progress": False,
        },
        D6_SHORT_PARENT_SHA256: {
            "parent_profile": "selected_short_d6",
            "report_sha256": D6_SHORT_REPORT_SHA256,
            "progress_sha256": D6_SHORT_PROGRESS_SHA256,
            "gather_init_sha256": D6_SHORT_GATHER_INIT_SHA256,
            "optimizer_steps": 128,
            "training_row_draws": 524_288,
            "symmetry_rng_provenance": ("per_rank_seedsequence_checkpoint_resume_v1"),
            "rank_distinct_symmetry_progress": True,
        },
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
        raise CompositionArmError(
            "D6+gather sources must be clean tracked canonical bytes"
        ) from error
    files = {
        relative: gather.corrected._file_ref(repo / relative)  # noqa: SLF001
        for relative in SOURCE_FILES
    }
    return {
        "repository_root": str(repo),
        "git_commit": commit,
        "files": files,
        "files_sha256": gather.corrected._digest(files),  # noqa: SLF001
    }


def _load_json_ref(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        ref = gather.corrected._file_ref(path)  # noqa: SLF001
        payload = json.loads(Path(ref["path"]).read_text(encoding="utf-8"))
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        gather.corrected.ArmError,
    ) as error:
        raise CompositionArmError(f"cannot authenticate {label}: {error}") from error
    if not isinstance(payload, dict):
        raise CompositionArmError(f"{label} is not a JSON object")
    return payload, ref


def _require_ref_sha(ref: Mapping[str, str], expected: str, *, label: str) -> None:
    if ref.get("sha256") != expected:
        raise CompositionArmError(
            f"{label} bytes are not the authenticated D6 artifact: {ref.get('sha256')}"
        )


def _load_d6_parent(
    checkpoint_path: Path,
    report_path: Path,
    progress_path: Path,
) -> dict[str, Any]:
    checkpoint = gather.corrected._file_ref(checkpoint_path)  # noqa: SLF001
    report, report_ref = _load_json_ref(report_path, label="D6 report")
    progress, progress_ref = _load_json_ref(progress_path, label="D6 progress")
    profile = _d6_parent_profiles().get(checkpoint["sha256"])
    if profile is None:
        raise CompositionArmError(
            "D6 checkpoint is not an authenticated completed parent: "
            f"{checkpoint['sha256']}"
        )
    _require_ref_sha(report_ref, profile["report_sha256"], label="D6 report")
    _require_ref_sha(progress_ref, profile["progress_sha256"], label="D6 progress")

    expected_report = {
        "init_checkpoint_sha256": D6_F7_PARENT_SHA256,
        "world_size": WORLD_SIZE,
        "batch_size": SELECTED_LOCAL_BATCH_SIZE,
        "effective_global_batch_size": SELECTED_GLOBAL_BATCH_SIZE,
        "max_steps": profile["optimizer_steps"],
        "steps_completed": profile["optimizer_steps"],
        "training_row_draws": profile["training_row_draws"],
        "optimizer": "adam",
        "resume_optimizer": False,
        "optimizer_restored": False,
        "lr": 3e-5,
        "lr_schedule": "flat",
        "lr_warmup_steps": 100,
        "action_module_lr_mult": 1.0,
        "soft_target_weight": 0.9,
        "soft_target_temperature": 0.7,
        "value_loss_weight": 0.25,
        "forced_action_weight": 0.0,
        "forced_row_value_weight": 1.0,
        "mask_hidden_info": True,
        "graph_history_features": True,
        "symmetry_augment": True,
        "diagnostic_only": True,
        "promotion_eligible": False,
    }
    drift = {
        key: {"expected": expected, "actual": report.get(key)}
        for key, expected in expected_report.items()
        if report.get(key) != expected
    }
    if drift:
        raise CompositionArmError(f"D6 parent training provenance drift: {drift}")

    stated_progress_digest = progress.get("progress_sha256")
    progress_unhashed = {
        key: value for key, value in progress.items() if key != "progress_sha256"
    }
    if stated_progress_digest != gather.corrected._digest(progress_unhashed):  # noqa: SLF001
        raise CompositionArmError("D6 progress semantic digest drift")
    checkpoint_binding = progress.get("checkpoint")
    recipe = progress.get("recipe_identity")
    rank_states = progress.get("rank_torch_rng_states")
    symmetry = progress.get("symmetry_rng_state")
    rank_distinct_symmetry = bool(profile["rank_distinct_symmetry_progress"])
    if rank_distinct_symmetry:
        symmetry_states = (
            symmetry.get("rank_states") if isinstance(symmetry, Mapping) else None
        )
        symmetry_ok = bool(
            isinstance(symmetry, Mapping)
            and symmetry.get("schema_version") == "train-bc-rank-symmetry-rng-v1"
            and symmetry.get("world_size") == WORLD_SIZE
            and isinstance(symmetry_states, list)
            and len(symmetry_states) == WORLD_SIZE
            and all(isinstance(state, Mapping) for state in symmetry_states)
            and len(
                {
                    gather.corrected._digest(state)  # noqa: SLF001
                    for state in symmetry_states
                }
            )
            == WORLD_SIZE
        )
    else:
        symmetry_ok = bool(
            isinstance(symmetry, Mapping)
            and symmetry.get("schema_version") is None
            and symmetry.get("bit_generator") == "PCG64"
        )
    if not (
        progress.get("schema_version") == "train-bc-progress-v1"
        and progress.get("status") == "complete"
        and progress.get("optimizer_step") == profile["optimizer_steps"]
        and isinstance(checkpoint_binding, Mapping)
        and checkpoint_binding.get("sha256") == checkpoint["sha256"]
        and isinstance(recipe, Mapping)
        and recipe.get("schema_version") == "train-bc-resume-recipe-v1"
        and recipe.get("world_size") == WORLD_SIZE
        and recipe.get("grad_accum_steps") == 1
        and recipe.get("ddp_shard_data") is False
        and recipe.get("fsdp") is False
        and isinstance(rank_states, list)
        and len(rank_states) == WORLD_SIZE
        and symmetry_ok
    ):
        raise CompositionArmError(
            "D6 progress does not prove complete 8-rank D6 training"
        )
    if sorted(
        row.get("rank") for row in rank_states if isinstance(row, Mapping)
    ) != list(range(WORLD_SIZE)):
        raise CompositionArmError("D6 progress lacks one RNG state per rank")

    evidence = {
        "parent_profile": profile["parent_profile"],
        "checkpoint": checkpoint,
        "report": report_ref,
        "progress": progress_ref,
        "gather_init_sha256": profile["gather_init_sha256"],
        "training_contract": expected_report,
        "progress_semantic_sha256": stated_progress_digest,
        "symmetry_rng_provenance": profile["symmetry_rng_provenance"],
        "evaluation_reference": "exact_D6_parent",
    }
    evidence["evidence_sha256"] = gather.corrected._digest(evidence)  # noqa: SLF001
    return evidence


def _set_boolean(command: list[str], flag: str, enabled: bool) -> dict[str, str]:
    positive = flag
    negative = "--no-" + flag.removeprefix("--")
    positions = [
        index for index, value in enumerate(command) if value in {positive, negative}
    ]
    if len(positions) > 1:
        raise CompositionArmError(f"source command repeats boolean option {flag}")
    source = "default"
    if positions:
        source = command.pop(positions[0])
    treatment = positive if enabled else negative
    command.append(treatment)
    return {"source": source, "treatment": treatment}


def _derive_command(
    source: Sequence[str],
    *,
    trainer: Path,
    gather_checkpoint: Path,
    output_root: Path,
) -> tuple[list[str], dict[str, Any]]:
    command = list(source)
    expected = {
        "--max-steps": str(SELECTED_OPTIMIZER_STEPS),
        "--batch-size": str(SELECTED_LOCAL_BATCH_SIZE),
        "--grad-accum-steps": "1",
        "--lr": "3e-05",
        "--lr-warmup-steps": "100",
        "--soft-target-weight": "0.9",
        "--value-loss-weight": "0.25",
        "--action-module-lr-mult": "1.0",
        "--value-lr-mult": "0.3",
    }
    observed = {
        flag: gather.corrected._option(command, flag)  # noqa: SLF001
        for flag in expected
    }
    if observed != expected:
        raise CompositionArmError(
            f"source is not exact selected-dose TEMP geometry: {observed}"
        )
    if command.count("--no-resume-optimizer") != 1 or "--resume-optimizer" in command:
        raise CompositionArmError("source does not require fresh Adam")
    trainer_positions = [
        index
        for index, value in enumerate(command)
        if Path(value).name == "train_bc.py"
    ]
    if len(trainer_positions) != 1:
        raise CompositionArmError("source must name exactly one historical trainer")
    index = trainer_positions[0]
    current_trainer = trainer.expanduser().resolve(strict=True)
    changes: dict[str, Any] = {
        "trainer": {
            "source": command[index],
            "treatment": str(current_trainer),
            "reason": "current rank-distinct symmetry RNG and exact resume contract",
        }
    }
    command[index] = str(current_trainer)
    updates = {
        "--init-checkpoint": str(gather_checkpoint.resolve(strict=True)),
        "--checkpoint": str(output_root / "candidate.pt"),
        "--report": str(output_root / "train.report.json"),
        "--batch-size": str(LOCAL_BATCH_SIZE),
        "--max-steps": str(OPTIMIZER_STEPS),
        "--action-module-lr-mult": str(ACTION_MODULE_LR_MULT),
        "--value-lr-mult": str(VALUE_LR_MULT),
        "--freeze-modules": FREEZE_MODULES,
        "--require-only-trainable-prefixes": TRAINABLE_PREFIX,
    }
    appendable = {"--freeze-modules", "--require-only-trainable-prefixes"}
    for flag, value in updates.items():
        source_value: Any = (
            "absent"
            if flag in appendable and flag not in command
            else gather.corrected._option(command, flag)  # noqa: SLF001
        )
        gather.corrected._set_option(command, flag, value)  # noqa: SLF001
        changes[flag] = {"source": source_value, "treatment": value}
    changes["--symmetry-augment"] = _set_boolean(command, "--symmetry-augment", True)
    changes["--symmetry-augment-events"] = _set_boolean(
        command, "--symmetry-augment-events", True
    )
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
    source, source_ref = gather._load_source(  # noqa: SLF001
        args.source_manifest,
        args.selected_dose_plan,
        args.selected_dose_report,
    )
    output_root = args.output_root.expanduser().resolve()
    existing = [str(path) for path in _forbidden_outputs(output_root) if path.exists()]
    if existing:
        raise CompositionArmError(f"D6+gather output already exists: {existing}")
    d6_parent = _load_d6_parent(args.d6_checkpoint, args.d6_report, args.d6_progress)
    gather_checkpoint = args.gather_checkpoint.expanduser().resolve(strict=True)
    try:
        upgrade = gather._validate_upgrade(  # noqa: SLF001
            Path(d6_parent["checkpoint"]["path"]), gather_checkpoint
        )
        coverage = gather._validate_coverage(  # noqa: SLF001
            args.architecture_audit, Path(source["descriptor"]["path"])
        )
    except gather.ArmError as error:
        raise CompositionArmError(str(error)) from error
    _require_ref_sha(
        upgrade["upgraded"],
        d6_parent["gather_init_sha256"],
        label="D6 gather initialization",
    )

    binding = _source_binding(args.repo)
    executor_ref = binding["files"].get(EXECUTOR_RELATIVE_PATH)
    completion_ref = binding["files"].get(COMPLETION_RELATIVE_PATH)
    trainer_ref = binding["files"].get("tools/train_bc.py")
    if not (
        isinstance(executor_ref, Mapping)
        and isinstance(completion_ref, Mapping)
        and isinstance(trainer_ref, Mapping)
    ):
        raise CompositionArmError(
            "source binding lacks executor/finalizer/current trainer"
        )
    command, changes = _derive_command(
        source["command"],
        trainer=Path(str(trainer_ref["path"])),
        gather_checkpoint=gather_checkpoint,
        output_root=output_root,
    )
    descriptor_meta, _ = gather.corrected._preflight_descriptor(  # noqa: SLF001
        Path(source["descriptor"]["path"])
    )
    event_contract, event_changes = (
        gather.corrected._bind_event_history_training_command(  # noqa: SLF001
            command, descriptor_meta
        )
    )
    changes.update(event_changes)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "launch_authorized": False,
        "diagnostic_execution_authorized": True,
        "launch_interface_present": f"{EXECUTOR_RELATIVE_PATH} execute --go",
        "diagnostic_executor": dict(executor_ref),
        "completion_finalizer": dict(completion_ref),
        "completion_interface_present": (
            f"{COMPLETION_RELATIVE_PATH} finalize --expected-checkpoint-sha256 SHA256"
        ),
        "source_temperature_manifest": source_ref,
        "source_temperature_manifest_sha256": source["manifest_sha256"],
        "selected_geometry_evidence": source["selected_geometry_evidence"],
        "source_recipe": source["recipe"],
        "source_recipe_sha256": source["recipe_sha256"],
        "descriptor": source["descriptor"],
        "validation_sentinel": source["validation_sentinel"],
        "d6_parent": d6_parent,
        "initialization_treatment": upgrade["upgraded"],
        "function_preserving_upgrade": upgrade,
        "corpus_topology_target_coverage": coverage,
        "event_history_training_contract": event_contract,
        "source_binding": binding,
        "only_declared_model_delta": (
            "train function-preserving target_gather_proj on frozen exact D6 parent"
        ),
        "matched_contract": {
            "reference_checkpoint": d6_parent["checkpoint"],
            "evaluation_reference": "exact_D6_parent",
            "candidate_chaining": False,
            "world_size": WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "optimizer_steps": OPTIMIZER_STEPS,
            "global_row_dose": GLOBAL_ROW_DOSE,
            "fresh_adam": True,
            "action_module_lr_mult": ACTION_MODULE_LR_MULT,
            "value_lr_mult": VALUE_LR_MULT,
            "freeze_modules": FREEZE_MODULES.split(","),
            "required_trainable_prefixes": [TRAINABLE_PREFIX],
            "new_trainable_parameter_names": list(gather.EXPECTED_NEW_PARAMETERS),
            "mature_parameters_trainable": False,
            "symmetry_augment": True,
            "symmetry_augment_events": True,
            "selected_TEMP_data_descriptor_and_seed_unchanged": True,
            "sampler_batch_partition_unchanged": False,
            "selected_TEMP_policy_value_loss_coefficients_and_forward_loss_unchanged": True,
            "treatment_distributed_symmetry_contract": (
                "per_rank_seedsequence_checkpoint_resume_v1"
            ),
        },
        "effective_trainable_objective": {
            "policy_only": True,
            "policy_loss_reaches_target_gather_proj": True,
            "value_loss_forward_computed": True,
            "value_loss_reaches_target_gather_proj": False,
            "reason": (
                "target_gather_proj affects policy logits only; trunk and value heads "
                "are frozen, and the completion receipt must bind policy-active dose"
            ),
        },
        "optimizer_geometry_contract": {
            "source_selected_TEMP": {
                "world_size": WORLD_SIZE,
                "local_batch_size": SELECTED_LOCAL_BATCH_SIZE,
                "global_batch_size": SELECTED_GLOBAL_BATCH_SIZE,
                "optimizer_steps": SELECTED_OPTIMIZER_STEPS,
                "global_row_dose": GLOBAL_ROW_DOSE,
            },
            "treatment_adapter_commissioning": {
                "world_size": WORLD_SIZE,
                "local_batch_size": LOCAL_BATCH_SIZE,
                "global_batch_size": GLOBAL_BATCH_SIZE,
                "optimizer_steps": OPTIMIZER_STEPS,
                "global_row_dose": GLOBAL_ROW_DOSE,
                "lr_warmup_steps": 100,
                "integrated_lr_step_equivalents": 974.5,
                "action_integrated_lr_step_equivalents": 3898.0,
            },
            "optimizer_update_count_multiplier": 8.0,
            "row_dose_unchanged": True,
            "reason": (
                "the zero-output gather residual needs the proven 1024-update "
                "commissioning geometry; this is not the 128-update D6-short arm"
            ),
        },
        "allowlisted_command_changes": changes,
        "command": command,
        "command_sha256": gather.corrected._digest(command),  # noqa: SLF001
        "output_root": str(output_root),
        "evaluation_contract": {
            "primary_opponent": d6_parent["checkpoint"],
            "comparison": "same-key behavior screen after sealed completion",
            "promotion_from_this_diagnostic": False,
        },
        "executor_compatibility": {
            "receipt_schema": RECEIPT_SCHEMA,
            "idle_topology": "exactly_8_visible_B200s",
            "one_shot": True,
        },
    }
    manifest["manifest_sha256"] = gather.corrected._digest(manifest)  # noqa: SLF001
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "d6-gather-composition.manifest.json"
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise CompositionArmError(f"prepared manifest drift: {path}")
    else:
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text(encoded, encoding="utf-8")
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    return manifest, path


def _verify_ref(value: Any, *, label: str) -> Path:
    try:
        return executor_base._verify_ref(value, label=label)  # noqa: SLF001
    except executor_base.ExecutionError as error:
        raise CompositionArmError(str(error)) from error


def verify(
    manifest_path: Path,
    *,
    expected_executor: Path | None = None,
    require_fresh_outputs: bool = True,
) -> dict[str, Any]:
    payload, manifest_ref = _load_json_ref(manifest_path, label="composition manifest")
    stated = payload.get("manifest_sha256")
    unhashed = {
        key: value for key, value in payload.items() if key != "manifest_sha256"
    }
    if stated != gather.corrected._digest(unhashed):  # noqa: SLF001
        raise CompositionArmError("D6+gather manifest semantic digest drift")
    if not (
        payload.get("schema_version") == SCHEMA
        and payload.get("diagnostic_only") is True
        and payload.get("promotion_eligible") is False
        and payload.get("launch_authorized") is False
        and payload.get("diagnostic_execution_authorized") is True
        and payload.get("launch_interface_present")
        == f"{EXECUTOR_RELATIVE_PATH} execute --go"
    ):
        raise CompositionArmError("manifest does not authorize this diagnostic")

    executor = _verify_ref(payload.get("diagnostic_executor"), label="executor")
    bound_executor = (
        Path(__file__).resolve()
        if expected_executor is None
        else expected_executor.expanduser().resolve(strict=True)
    )
    if executor != bound_executor:
        raise CompositionArmError("manifest authorizes a different executor")
    binding = payload.get("source_binding")
    if not isinstance(binding, Mapping):
        raise CompositionArmError("manifest lacks source binding")
    repo = Path(str(binding.get("repository_root", ""))).resolve(strict=True)
    if executor_base._git_head(repo) != binding.get("git_commit"):  # noqa: SLF001
        raise CompositionArmError("composition checkout commit drift")
    files = binding.get("files")
    if not isinstance(files, Mapping) or set(files) != set(SOURCE_FILES):
        raise CompositionArmError("composition source binding is incomplete")
    if binding.get("files_sha256") != gather.corrected._digest(files):  # noqa: SLF001
        raise CompositionArmError("composition source file-set digest drift")
    for relative, ref in files.items():
        if _verify_ref(ref, label=f"source.{relative}") != (repo / relative).resolve(
            strict=True
        ):
            raise CompositionArmError(f"bound source escaped checkout: {relative}")
    if files[EXECUTOR_RELATIVE_PATH] != payload["diagnostic_executor"]:
        raise CompositionArmError("executor differs from bound source bytes")
    completion = _verify_ref(
        payload.get("completion_finalizer"), label="completion finalizer"
    )
    if not (
        completion == (repo / COMPLETION_RELATIVE_PATH).resolve(strict=True)
        and files[COMPLETION_RELATIVE_PATH] == payload["completion_finalizer"]
        and payload.get("completion_interface_present")
        == f"{COMPLETION_RELATIVE_PATH} finalize --expected-checkpoint-sha256 SHA256"
    ):
        raise CompositionArmError(
            "completion finalizer differs from bound source bytes"
        )

    source_manifest = _verify_ref(
        payload.get("source_temperature_manifest"), label="source manifest"
    )
    selected = payload.get("selected_geometry_evidence")
    if not isinstance(selected, Mapping):
        raise CompositionArmError("manifest lacks selected TEMP evidence")
    plan = _verify_ref(selected.get("plan"), label="selected geometry plan")
    report = _verify_ref(selected.get("report"), label="selected geometry report")
    try:
        source, _ = gather._load_source(source_manifest, plan, report)  # noqa: SLF001
    except gather.ArmError as error:
        raise CompositionArmError(f"selected TEMP bridge failed: {error}") from error
    for manifest_key, source_key in (
        ("selected_geometry_evidence", "selected_geometry_evidence"),
        ("source_recipe", "recipe"),
        ("descriptor", "descriptor"),
        ("validation_sentinel", "validation_sentinel"),
    ):
        if payload.get(manifest_key) != source[source_key]:
            raise CompositionArmError(f"selected TEMP identity drift: {manifest_key}")
    if not (
        payload.get("source_temperature_manifest_sha256") == source["manifest_sha256"]
        and payload.get("source_recipe_sha256") == source["recipe_sha256"]
    ):
        raise CompositionArmError("selected TEMP recipe digest drift")

    parent_payload = payload.get("d6_parent")
    if not isinstance(parent_payload, Mapping):
        raise CompositionArmError("manifest lacks D6 parent provenance")
    parent = _load_d6_parent(
        _verify_ref(parent_payload.get("checkpoint"), label="D6 checkpoint"),
        _verify_ref(parent_payload.get("report"), label="D6 report"),
        _verify_ref(parent_payload.get("progress"), label="D6 progress"),
    )
    if parent != parent_payload:
        raise CompositionArmError("D6 parent provenance replay drift")
    treatment = _verify_ref(
        payload.get("initialization_treatment"), label="gather initialization"
    )
    try:
        upgrade = gather._validate_upgrade(  # noqa: SLF001
            Path(parent["checkpoint"]["path"]), treatment
        )
    except gather.ArmError as error:
        raise CompositionArmError(str(error)) from error
    if upgrade != payload.get("function_preserving_upgrade"):
        raise CompositionArmError("function-preserving gather upgrade replay drift")
    _require_ref_sha(
        upgrade["upgraded"],
        parent["gather_init_sha256"],
        label="D6 gather initialization",
    )
    coverage = payload.get("corpus_topology_target_coverage")
    if not isinstance(coverage, Mapping):
        raise CompositionArmError("manifest lacks topology coverage")
    audit = _verify_ref(coverage.get("artifact"), label="architecture audit")
    try:
        expected_coverage = gather._validate_coverage(  # noqa: SLF001
            audit, Path(source["descriptor"]["path"])
        )
    except gather.ArmError as error:
        raise CompositionArmError(str(error)) from error
    if expected_coverage != coverage:
        raise CompositionArmError("topology coverage replay drift")

    root = Path(str(payload.get("output_root", ""))).resolve()
    trainer = _verify_ref(files["tools/train_bc.py"], label="current trainer")
    expected_command, changes = _derive_command(
        source["command"],
        trainer=trainer,
        gather_checkpoint=treatment,
        output_root=root,
    )
    descriptor_meta, _ = gather.corrected._preflight_descriptor(  # noqa: SLF001
        Path(source["descriptor"]["path"])
    )
    expected_event_contract, event_changes = (
        gather.corrected._bind_event_history_training_command(  # noqa: SLF001
            expected_command, descriptor_meta
        )
    )
    changes.update(event_changes)
    command = payload.get("command")
    if not (
        command == expected_command
        and payload.get("allowlisted_command_changes") == changes
        and payload.get("command_sha256") == gather.corrected._digest(command)  # noqa: SLF001
        and payload.get("event_history_training_contract") == expected_event_contract
    ):
        raise CompositionArmError("command is not exact D6+gather derivation")
    if (
        command.count("--symmetry-augment") != 1
        or command.count("--symmetry-augment-events") != 1
        or "--no-symmetry-augment" in command
        or "--no-symmetry-augment-events" in command
    ):
        raise CompositionArmError("D6+gather symmetry boolean drift")
    expected_matched = {
        "reference_checkpoint": parent["checkpoint"],
        "evaluation_reference": "exact_D6_parent",
        "candidate_chaining": False,
        "world_size": WORLD_SIZE,
        "local_batch_size": LOCAL_BATCH_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "optimizer_steps": OPTIMIZER_STEPS,
        "global_row_dose": GLOBAL_ROW_DOSE,
        "fresh_adam": True,
        "action_module_lr_mult": ACTION_MODULE_LR_MULT,
        "value_lr_mult": VALUE_LR_MULT,
        "freeze_modules": FREEZE_MODULES.split(","),
        "required_trainable_prefixes": [TRAINABLE_PREFIX],
        "new_trainable_parameter_names": list(gather.EXPECTED_NEW_PARAMETERS),
        "mature_parameters_trainable": False,
        "symmetry_augment": True,
        "symmetry_augment_events": True,
        "selected_TEMP_data_descriptor_and_seed_unchanged": True,
        "sampler_batch_partition_unchanged": False,
        "selected_TEMP_policy_value_loss_coefficients_and_forward_loss_unchanged": True,
        "treatment_distributed_symmetry_contract": (
            "per_rank_seedsequence_checkpoint_resume_v1"
        ),
    }
    expected_effective_objective = {
        "policy_only": True,
        "policy_loss_reaches_target_gather_proj": True,
        "value_loss_forward_computed": True,
        "value_loss_reaches_target_gather_proj": False,
        "reason": (
            "target_gather_proj affects policy logits only; trunk and value heads "
            "are frozen, and the completion receipt must bind policy-active dose"
        ),
    }
    expected_optimizer_geometry = {
        "source_selected_TEMP": {
            "world_size": WORLD_SIZE,
            "local_batch_size": SELECTED_LOCAL_BATCH_SIZE,
            "global_batch_size": SELECTED_GLOBAL_BATCH_SIZE,
            "optimizer_steps": SELECTED_OPTIMIZER_STEPS,
            "global_row_dose": GLOBAL_ROW_DOSE,
        },
        "treatment_adapter_commissioning": {
            "world_size": WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "optimizer_steps": OPTIMIZER_STEPS,
            "global_row_dose": GLOBAL_ROW_DOSE,
            "lr_warmup_steps": 100,
            "integrated_lr_step_equivalents": 974.5,
            "action_integrated_lr_step_equivalents": 3898.0,
        },
        "optimizer_update_count_multiplier": 8.0,
        "row_dose_unchanged": True,
        "reason": (
            "the zero-output gather residual needs the proven 1024-update "
            "commissioning geometry; this is not the 128-update D6-short arm"
        ),
    }
    if not (
        payload.get("matched_contract") == expected_matched
        and payload.get("effective_trainable_objective") == expected_effective_objective
        and payload.get("optimizer_geometry_contract") == expected_optimizer_geometry
        and payload.get("only_declared_model_delta")
        == "train function-preserving target_gather_proj on frozen exact D6 parent"
        and payload.get("evaluation_contract")
        == {
            "primary_opponent": parent["checkpoint"],
            "comparison": "same-key behavior screen after sealed completion",
            "promotion_from_this_diagnostic": False,
        }
    ):
        raise CompositionArmError("D6+gather causal/evaluation contract drift")
    if require_fresh_outputs:
        existing = [str(path) for path in _forbidden_outputs(root) if path.exists()]
        if existing:
            raise CompositionArmError(f"D6+gather output already exists: {existing}")
    return {
        "manifest": payload,
        "manifest_ref": manifest_ref,
        "repo": repo,
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
    verified = verify(manifest_path)
    try:
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
        raise CompositionArmError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-manifest", required=True, type=Path)
    prep.add_argument("--selected-dose-plan", required=True, type=Path)
    prep.add_argument("--selected-dose-report", required=True, type=Path)
    prep.add_argument("--d6-checkpoint", required=True, type=Path)
    prep.add_argument("--d6-report", required=True, type=Path)
    prep.add_argument("--d6-progress", required=True, type=Path)
    prep.add_argument("--gather-checkpoint", required=True, type=Path)
    prep.add_argument("--architecture-audit", required=True, type=Path)
    prep.add_argument("--output-root", required=True, type=Path)
    prep.add_argument("--repo", default=REPO_ROOT, type=Path)
    check = sub.add_parser("verify")
    check.add_argument("--manifest", required=True, type=Path)
    run = sub.add_parser("execute")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--unit", default="a1-d6-gather-composition")
    run.add_argument("--go", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "prepare":
        manifest, path = prepare(args)
        print(
            json.dumps(
                {
                    "prepared": str(path),
                    "launched": False,
                    "manifest_sha256": manifest["manifest_sha256"],
                },
                sort_keys=True,
            )
        )
        return
    if args.action == "verify" or (args.action == "execute" and not args.go):
        verified = verify(args.manifest)
        print(
            json.dumps(
                {
                    "verified": True,
                    "launched": False,
                    "manifest": verified["manifest_ref"],
                },
                sort_keys=True,
            )
        )
        return
    receipt = execute(args.manifest, unit=args.unit)
    print(
        json.dumps(
            {
                "submitted": True,
                "unit": receipt["unit"],
                "receipt_sha256": receipt["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
