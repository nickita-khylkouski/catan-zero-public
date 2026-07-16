#!/usr/bin/env python3
"""Sealed native-scratch executor for the current coherent-public A1 learner.

The historical one-dose executor is checkpoint-initialized by design.  This
entrypoint is the separate execution path for the current science contract's
native v5 model and fresh optimizer.  Without ``--go`` it emits the exact
authenticated plan; with ``--go`` it executes only a commissioned topology and
retains every epoch checkpoint for playing-strength selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_current_science_contract as current_science  # noqa: E402
from tools import a1_one_dose_train as one_dose  # noqa: E402
from tools import a1_pre_wave_contract as contract  # noqa: E402
from tools import train_bc  # noqa: E402


PLAN_SCHEMA = "a1-coherent-scratch-training-receipt-v1"
EXECUTION_SCHEMA = "a1-coherent-scratch-training-execution-v2"
CHILD_AUTHORITY_SCHEMA = "a1-coherent-scratch-plan-authority-v2"
CODE_SURFACE = (
    "tools/a1_scratch_train.py",
    "tools/a1_current_science_contract.py",
    "tools/a1_pre_wave_contract.py",
    "tools/a1_build_post_wave_composite.py",
    "tools/a1_feature_signal_admission.py",
    "tools/train_bc.py",
    "src/catan_zero/rl/entity_feature_adapter.py",
    "src/catan_zero/rl/entity_token_policy.py",
)


class ScratchTrainError(RuntimeError):
    """The coherent scratch launch cannot be authenticated."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _value_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _regular_file(path: Path, *, where: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ScratchTrainError(f"{where} may not be a symlink: {expanded}")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise ScratchTrainError(f"{where} must be a regular file: {resolved}")
    return resolved


def _ref(path: Path, *, where: str) -> dict[str, str]:
    resolved = _regular_file(path, where=where)
    return {"path": str(resolved), "file_sha256": _file_sha256(resolved)}


def _executable_ref(path: Path, *, where: str) -> dict[str, str]:
    """Bind a venv-safe lexical executable and the bytes it resolves to."""

    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        target = lexical.resolve(strict=True)
    except OSError as error:
        raise ScratchTrainError(f"cannot resolve {where}: {error}") from error
    if (
        not lexical.is_file()
        or not target.is_file()
        or not os.access(lexical, os.X_OK)
        or not os.access(target, os.X_OK)
    ):
        raise ScratchTrainError(f"{where} must be executable: {lexical}")
    return {
        "path": str(lexical),
        "target_path": str(target),
        "file_sha256": _file_sha256(target),
    }


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ScratchTrainError(f"refusing non-fresh receipt path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    unsigned = dict(payload)
    unsigned["receipt_sha256"] = _value_sha256(unsigned)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="ascii") as handle:
            handle.write(json.dumps(unsigned, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _execution_receipt_path(plan_receipt: Path) -> Path:
    suffix = plan_receipt.suffix
    if suffix:
        name = f"{plan_receipt.stem}.execution{suffix}"
    else:
        name = f"{plan_receipt.name}.execution.json"
    return plan_receipt.with_name(name)


def _load_matching_plan_receipt(
    path: Path, *, expected: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    plan_path = _regular_file(path, where="scratch plan receipt")
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScratchTrainError(f"scratch plan receipt is unreadable: {error}") from error
    if not isinstance(payload, dict):
        raise ScratchTrainError("scratch plan receipt must be a JSON object")
    unsigned = dict(payload)
    declared = unsigned.pop("receipt_sha256", None)
    if declared != _value_sha256(unsigned):
        raise ScratchTrainError("scratch plan receipt semantic digest mismatch")
    if unsigned.get("schema_version") != PLAN_SCHEMA or unsigned.get("status") != "planned":
        raise ScratchTrainError("scratch execution requires a completed plan-only receipt")
    if set(unsigned) != set(expected):
        raise ScratchTrainError("scratch plan receipt fields differ from current plan")
    for key, value in expected.items():
        if key == "created_unix_ns":
            continue
        if unsigned[key] != value:
            raise ScratchTrainError(
                f"scratch plan receipt differs from current plan at {key!r}"
            )
    return payload, {
        "path": str(plan_path),
        "file_sha256": _file_sha256(plan_path),
        "receipt_sha256": str(declared),
    }


def _science_authority(lock: Mapping[str, Any]) -> dict[str, Any]:
    science = lock.get("science")
    if not isinstance(science, dict):
        raise ScratchTrainError("sealed A1 lock has no science section")
    search = science.get("search_operator")
    if not isinstance(search, dict) or not current_science.is_coherent_search(search):
        raise ScratchTrainError("scratch launcher accepts only coherent-public locks")
    try:
        current_science.require_current_operator(
            search_value=search,
            evaluator_value=science.get("evaluator"),
            generation_value=lock.get("generation"),
            learner_recipe_value=science.get("learner_training_recipe"),
            target_regime=lock.get("post_wave_acceptance", {}).get(
                "require_target_information_regime"
            ),
            require_adopted=True,
        )
    except current_science.ScienceContractError as error:
        raise ScratchTrainError(str(error)) from error
    initialization = current_science.learner_initialization()
    if (
        science.get("learner_initialization") != initialization
        or science.get("learner_initialization_sha256")
        != _value_sha256(initialization)
    ):
        raise ScratchTrainError(
            "coherent lock does not bind the current native scratch initialization"
        )
    model = current_science.learner_model_construction()
    topology = current_science.learner_execution_topology()
    if (
        science.get("learner_model_construction") != model
        or science.get("learner_model_construction_sha256") != _value_sha256(model)
        or science.get("learner_execution_topology") != topology
        or science.get("learner_execution_topology_sha256")
        != _value_sha256(topology)
    ):
        raise ScratchTrainError(
            "coherent lock does not bind current scratch model/topology authority"
        )
    return {
        "initialization": initialization,
        "model_construction": model,
        "execution_topology": topology,
        "logical_recipe": current_science.learner_training_recipe(),
    }


_SCIENCE_BINDING_FIELDS = (
    "science_schema_version",
    "search_operator",
    "search_operator_sha256",
    "evaluator",
    "evaluator_sha256",
    "learner_value_objective",
    "learner_value_objective_sha256",
    "learner_training_recipe",
    "learner_training_recipe_sha256",
    "learner_initialization",
    "learner_initialization_sha256",
    "learner_model_construction",
    "learner_model_construction_sha256",
    "learner_execution_topology",
    "learner_execution_topology_sha256",
)


def _scratch_plan_authority(verified: Mapping[str, Any]) -> dict[str, Any]:
    source_authority = verified.get("source_authority")
    if not isinstance(source_authority, Mapping):
        raise ScratchTrainError("verified composite has no source authority")
    current_contract = source_authority.get("current_contract")
    if (
        not isinstance(current_contract, Mapping)
        or set(current_contract) != {"path", "file_sha256", "contract_sha256"}
    ):
        raise ScratchTrainError("source authority has no exact staged current contract")
    staged_path = _regular_file(
        Path(str(current_contract["path"])), where="staged current A1 lock"
    )
    if _file_sha256(staged_path) != current_contract["file_sha256"]:
        raise ScratchTrainError("staged current A1 lock bytes drifted")
    try:
        staged_lock = json.loads(staged_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScratchTrainError(f"cannot read staged current A1 lock: {error}") from error
    unhashed = dict(staged_lock)
    stated = unhashed.pop("contract_sha256", None)
    if (
        stated != _value_sha256(unhashed)
        or stated != current_contract["contract_sha256"]
    ):
        raise ScratchTrainError("staged current A1 lock semantic digest drift")
    science = staged_lock.get("science")
    if not isinstance(science, dict) or any(
        field not in science for field in _SCIENCE_BINDING_FIELDS
    ):
        raise ScratchTrainError("staged current A1 lock science authority is incomplete")
    science_binding = {field: science[field] for field in _SCIENCE_BINDING_FIELDS}
    descriptor = {
        "path": str(verified["data_path"]),
        "file_sha256": verified["corpus_meta_file_sha256"],
        "fingerprint": verified["descriptor_fingerprint"],
        "payload_inventory_sha256": verified["payload_inventory_sha256"],
    }
    build_receipt = verified.get("composite_build_receipt")
    source_ref = verified.get("source_authority_ref")
    if (
        not isinstance(build_receipt, dict)
        or set(build_receipt) != {"path", "file_sha256", "receipt_sha256"}
        or not isinstance(source_ref, dict)
    ):
        raise ScratchTrainError("composite receipt/source authority binding is incomplete")
    return {
        "schema_version": CHILD_AUTHORITY_SCHEMA,
        "staged_contract": dict(current_contract),
        "science": science_binding,
        "descriptor": descriptor,
        "source_authority": dict(source_ref),
        "source_authority_semantic_sha256": verified[
            "source_authority_semantic_sha256"
        ],
        "build_receipt": dict(build_receipt),
    }


def _require_plan_authority_matches_verified(
    authority: Mapping[str, Any], verified: Mapping[str, Any]
) -> None:
    if dict(authority) != _scratch_plan_authority(verified):
        raise ScratchTrainError("scratch plan authority differs from verified inputs")


def _accepted_policy_target_identity(meta: Mapping[str, Any]) -> str:
    """Resolve the one exact teacher operator admitted to scratch policy CE."""

    components = meta.get("components")
    distillation_ids = meta.get("policy_distillation_component_ids")
    if not isinstance(components, list) or not isinstance(distillation_ids, list):
        raise ScratchTrainError(
            "scratch composite lacks an explicit policy-distillation target scope"
        )
    by_id = {
        str(component.get("component_id")): component
        for component in components
        if isinstance(component, Mapping)
    }
    identities: set[str] = set()
    missing: list[str] = []
    for component_id in distillation_ids:
        component = by_id.get(str(component_id))
        corpus_meta = (
            component.get("corpus_meta")
            if isinstance(component, Mapping)
            else None
        )
        identity = (
            corpus_meta.get("policy_target_identity_sha256")
            if isinstance(corpus_meta, Mapping)
            else None
        )
        if identity is None:
            missing.append(str(component_id))
        elif not train_bc._is_sha256(str(identity)):  # noqa: SLF001
            raise ScratchTrainError(
                f"scratch policy component {component_id!r} has malformed "
                "target identity"
            )
        else:
            identities.add(str(identity))
    if missing:
        raise ScratchTrainError(
            "scratch policy components lack exact target identity: "
            + ", ".join(missing)
        )
    if len(identities) != 1:
        raise ScratchTrainError(
            "scratch policy components do not share one exact target operator"
        )
    return next(iter(identities))


def verify_inputs(
    *,
    lock_path: Path,
    data_path: Path,
    composite_build_receipt: Path,
) -> dict[str, Any]:
    try:
        lock_path = _regular_file(lock_path, where="A1 lock")
        data_path = _regular_file(data_path, where="composite descriptor")
        composite_build_receipt = _regular_file(
            composite_build_receipt, where="composite build receipt"
        )
        lock = contract.verify_lock(lock_path, require_all_job_claims=True)
        science = _science_authority(lock)
        meta = train_bc._preflight_memmap_composite_descriptor(data_path)  # noqa: SLF001
        verified = one_dose._verify_production_composite_inputs(  # noqa: SLF001
            lock=lock,
            lock_path=lock_path,
            reviewed_lock_file_sha256=_file_sha256(lock_path),
            recipe=science["logical_recipe"],
            objective=dict(lock["science"]["learner_value_objective"]),
            producer=one_dose._producer(lock),  # noqa: SLF001
            data_path=data_path,
            meta=meta,
            validation_path=None,
            build_receipt_path=composite_build_receipt,
        )
        verified = _bind_scratch_training_topology(
            verified,
            logical_recipe=science["logical_recipe"],
            topology=science["execution_topology"],
        )
        verified["accepted_policy_target_identity_sha256"] = (
            _accepted_policy_target_identity(meta)
        )
    except (
        contract.ContractError,
        one_dose.ExecutorError,
        OSError,
        SystemExit,
        ValueError,
    ) as error:
        raise ScratchTrainError(f"scratch input verification failed: {error}") from error
    adapters = set(verified["entity_feature_adapter_component_versions"].values())
    if adapters != {science["initialization"]["entity_feature_adapter_version"]}:
        raise ScratchTrainError("composite adapter semantics differ from scratch model")
    return {**verified, **science}


def _bind_scratch_training_topology(
    verified: Mapping[str, Any],
    *,
    logical_recipe: Mapping[str, Any],
    topology: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the logical scratch batch onto its dedicated 8-rank topology.

    ``one_dose.bind_training_topology`` is intentionally sealed to the
    historical 4096-global checkpoint-initialized dose.  Reusing it here made
    the native scratch executor reject its own 512-global science contract
    before it could even emit a plan.  Scratch owns a separate topology
    contract, so bind it directly and prove that the projection preserves the
    logical global batch.
    """

    bound = dict(verified.get("bound_recipe", verified.get("recipe", logical_recipe)))
    expected = dict(logical_recipe)
    if bound != expected:
        raise ScratchTrainError(
            "verified composite learner recipe differs from current scratch science"
        )
    required_topology = current_science.learner_execution_topology()
    if dict(topology) != required_topology:
        raise ScratchTrainError("scratch execution topology differs from current science")
    if (
        str(topology.get("name")) != "b200-8gpu-ddp"
        or str(topology.get("launcher")) != "torch.distributed.run"
        or list(topology.get("physical_gpus", ())) != list(range(8))
    ):
        raise ScratchTrainError("scratch execution topology is not exact 8-GPU B200 DDP")
    effective = dict(verified.get("recipe", bound))
    topology_fields = (
        "world_size",
        "batch_size",
        "grad_accum_steps",
        "global_batch_size",
    )
    topology_drift = {
        field: {"logical": bound.get(field), "effective": effective.get(field)}
        for field in topology_fields
        if effective.get(field) != bound.get(field)
    }
    if topology_drift:
        raise ScratchTrainError(
            "authenticated composite override changes scratch topology/dose fields: "
            f"{topology_drift}"
        )
    effective.update(
        {
            "world_size": int(topology["world_size"]),
            "batch_size": int(topology["local_batch_size"]),
            "grad_accum_steps": int(topology["grad_accum_steps"]),
            "global_batch_size": int(topology["global_batch_size"]),
        }
    )
    realized_global_batch = (
        int(effective["world_size"])
        * int(effective["batch_size"])
        * int(effective["grad_accum_steps"])
    )
    if (
        realized_global_batch != int(bound["global_batch_size"])
        or realized_global_batch != int(topology["global_batch_size"])
    ):
        raise ScratchTrainError("scratch topology changes the logical global batch")
    result = dict(verified)
    result.update(
        {
            "bound_recipe": bound,
            "recipe": effective,
            "training_topology": {
                "schema_version": str(topology["schema_version"]),
                "name": str(topology["name"]),
                "world_size": int(topology["world_size"]),
                "physical_gpus": list(topology["physical_gpus"]),
                "local_batch_size": int(topology["local_batch_size"]),
                "grad_accum_steps": int(topology["grad_accum_steps"]),
                "global_batch_size": realized_global_batch,
                "dose_preserving": True,
            },
        }
    )
    return result


def _add(command: list[str], flag: str, value: object) -> None:
    command.extend((flag, str(value)))


def build_train_command(
    verified: Mapping[str, Any],
    *,
    python: Path,
    checkpoint: Path,
    report: Path,
) -> list[str]:
    recipe = dict(verified["recipe"])
    accepted_policy_target_identity = str(
        verified.get("accepted_policy_target_identity_sha256", "")
    )
    if not train_bc._is_sha256(accepted_policy_target_identity):  # noqa: SLF001
        raise ScratchTrainError(
            "scratch training requires one verified policy-target identity"
        )
    model = dict(verified["model_construction"])
    topology = dict(verified["execution_topology"])
    plan_authority = _scratch_plan_authority(verified)
    trainer = Path(str(verified["trainer_authority"]["path"]))
    command = [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={topology['world_size']}",
        str(trainer),
    ]
    for flag, key in (
        ("--arch", "arch"),
        ("--hidden-size", "hidden_size"),
        ("--graph-layers", "graph_layers"),
        ("--attention-heads", "attention_heads"),
        ("--graph-dropout", "graph_dropout"),
        ("--entity-state-trunk", "entity_state_trunk"),
        ("--value-tower-split-layers", "value_tower_split_layers"),
        ("--entity-feature-adapter-version", "entity_feature_adapter_version"),
        ("--meaningful-public-history-pooling", "meaningful_public_history_pooling"),
        ("--event-history-limit", "event_history_limit"),
    ):
        _add(command, flag, model[key])
    # Keep train_bc's backward-compatible CLI name while the sealed science
    # contract uses a model-class-neutral parameter ceiling.
    _add(command, "--max-35m-params", model["max_parameter_count"])
    command.extend(
        [
            "--action-target-gather",
            "--static-action-residual",
            "--legal-action-value-residual",
            "--legal-action-value-set-statistics",
            "--public-card-count-features",
            "--no-public-card-count-residual-bias",
            "--meaningful-public-history",
            "--data",
            str(verified["data_path"]),
            "--data-format",
            "memmap",
            "--data-loader-workers",
            "2",
            "--data-loader-prefetch",
            "2",
            "--device",
            "cuda",
            "--graph-history-features",
        ]
    )
    command.append(
        "--meaningful-public-history-target-gather"
        if model["meaningful_public_history_target_gather"]
        else "--no-meaningful-public-history-target-gather"
    )
    command.append(
        "--public-rule-state-features"
        if model["public_rule_state_features"]
        else "--no-public-rule-state-features"
    )
    scalar_fields = (
        ("track", "--track"),
        ("vps_to_win", "--vps-to-win"),
        ("seed", "--seed"),
        ("epochs", "--epochs"),
        ("max_steps", "--max-steps"),
        ("checkpoint_steps", "--checkpoint-steps"),
        ("base_sampler", "--base-sampler"),
        ("batch_size", "--batch-size"),
        ("grad_accum_steps", "--grad-accum-steps"),
        ("optimizer", "--optimizer"),
        ("lr", "--lr"),
        ("lr_warmup_steps", "--lr-warmup-steps"),
        ("lr_schedule", "--lr-schedule"),
        ("weight_decay", "--weight-decay"),
        ("max_grad_norm", "--max-grad-norm"),
        ("value_lr_mult", "--value-lr-mult"),
        ("action_module_lr_mult", "--action-module-lr-mult"),
        ("public_card_lr_mult", "--public-card-lr-mult"),
        ("trunk_lr_mult", "--trunk-lr-mult"),
        ("value_trunk_grad_scale", "--value-trunk-grad-scale"),
        ("train_diagnostics_every_batches", "--train-diagnostics-every-batches"),
        (
            "objective_gradient_interference_every_batches",
            "--objective-gradient-interference-every-batches",
        ),
        (
            "require_feature_learning_signal_modules",
            "--require-feature-learning-signal-modules",
        ),
        (
            "minimum_feature_learning_signal_observations",
            "--minimum-feature-learning-signal-observations",
        ),
        ("policy_loss_weight", "--policy-loss-weight"),
        ("policy_dose_lr_area", "--policy-dose-lr-area"),
        (
            "policy_dose_reference_global_batch_size",
            "--policy-dose-reference-global-batch-size",
        ),
        (
            "post_policy_dose_value_trunk_grad_scale",
            "--post-policy-dose-value-trunk-grad-scale",
        ),
        ("soft_target_source", "--soft-target-source"),
        ("soft_target_weight", "--soft-target-weight"),
        ("policy_target_blend_semantics", "--policy-target-blend-semantics"),
        ("soft_target_temperature", "--soft-target-temperature"),
        ("soft_target_min_legal_coverage", "--soft-target-min-legal-coverage"),
        ("value_loss_weight", "--value-loss-weight"),
        ("scalar_value_loss_readout", "--scalar-value-loss-readout"),
        ("scalar_value_loss_scale", "--scalar-value-loss-scale"),
        ("value_target_lambda", "--value-target-lambda"),
        ("value_categorical_loss_weight", "--value-categorical-loss-weight"),
        ("hlgauss_scalar_aux_loss_weight", "--hlgauss-scalar-aux-loss-weight"),
        ("final_vp_loss_weight", "--final-vp-loss-weight"),
        ("q_loss_weight", "--q-loss-weight"),
        ("policy_kl_anchor_weight", "--policy-kl-anchor-weight"),
        ("value_uncertainty_loss_weight", "--value-uncertainty-loss-weight"),
        ("aux_subgoal_loss_weight", "--aux-subgoal-loss-weight"),
        ("freeze_modules", "--freeze-modules"),
        ("policy_surprise_weight", "--policy-surprise-weight"),
        ("target_reliability_confidence_floor", "--target-reliability-confidence-floor"),
        ("advantage_policy_weighting", "--advantage-policy-weighting"),
        ("vp_margin_weight", "--vp-margin-weight"),
        ("truncated_vp_margin_value_weight", "--truncated-vp-margin-value-weight"),
        ("amp", "--amp"),
        ("forced_action_weight", "--forced-action-weight"),
        ("forced_row_value_weight", "--forced-row-value-weight"),
        ("forced_row_value_action_type_weights", "--forced-row-value-action-type-weights"),
        ("winner_sample_weight", "--winner-sample-weight"),
        ("loser_sample_weight", "--loser-sample-weight"),
        ("teacher_weights", "--teacher-weights"),
        ("phase_weights", "--phase-weights"),
        ("value_phase_weights", "--value-phase-weights"),
    )
    optional_scalar_defaults = {
        "public_card_lr_mult": 1.0,
        "target_reliability_confidence_floor": 0.25,
        "policy_dose_lr_area": 0.0,
        "policy_dose_reference_global_batch_size": 0,
        "post_policy_dose_value_trunk_grad_scale": 1.0,
    }
    for key, flag in scalar_fields:
        if key in optional_scalar_defaults:
            _add(command, flag, recipe.get(key, optional_scalar_defaults[key]))
        else:
            _add(command, flag, recipe[key])
    command.extend(
        [
            "--no-resume-optimizer",
            (
                "--fused-optimizer"
                if recipe["fused_optimizer"]
                else "--no-fused-optimizer"
            ),
            "--value-head-type",
            "mse",
            "--value-categorical-bins",
            "0",
            "--policy-kl-anchor-direction",
            str(recipe.get("policy_kl_anchor_direction", "forward")),
            "--mask-hidden-info",
            (
                "--symmetry-augment"
                if recipe["symmetry_augment"]
                else "--no-symmetry-augment"
            ),
            (
                "--symmetry-augment-events"
                if recipe["symmetry_augment_events"]
                else "--no-symmetry-augment-events"
            ),
            "--validation-fraction",
            "0.05",
            "--validation-seed",
            "17",
            "--validation-max-samples",
            "0",
            "--required-target-information-regime",
            current_science.target_information_regime(),
            "--accepted-policy-target-identity-sha256",
            accepted_policy_target_identity,
            "--public-award-feature-contract",
            "authoritative_v1",
            "--allow-mixed-public-award-feature-contracts",
            "--training-rng-rank-offset",
            "--a1-scratch-authority-json",
            _canonical_bytes(plan_authority).decode("ascii"),
            "--per-game-value-weight",
            "--per-game-value-weight-mode",
            str(recipe.get("per_game_value_weight_mode", "equal")),
            "--value-player-outcome-balance-mode",
            str(recipe["value_player_outcome_balance_mode"]),
            "--per-game-policy-weight",
            "--per-game-policy-weight-mode",
            str(recipe.get("per_game_policy_weight_mode", "equal")),
            "--no-per-game-policy-surprise-weighting",
            "--checkpoint",
            str(checkpoint),
            "--report",
            str(report),
            "--save-each-epoch",
            "--require-35m-model",
            "--skip-teacher-quality-gate",
            "--trust-curated-data-quality",
        ]
    )
    history = verified["event_history_training_contract"]
    for digest in history["empty_payload_inventory_acknowledgements"]:
        command.extend((one_dose.EVENT_HISTORY_ACK_FLAG, str(digest)))
    if history["training_event_history_trainable"] is False:
        command.append(one_dose.EVENT_HISTORY_CROP_FLAG)
    forbidden = {"--init-checkpoint", "--grow-from-checkpoint", "--resume-optimizer"}
    if forbidden.intersection(command) or "--ddp-shard-data" in command:
        raise ScratchTrainError("scratch command inherited checkpoint/sharded semantics")
    if command.count("--no-public-card-count-residual-bias") != 1:
        raise ScratchTrainError("scratch command lost bias-free card-count v2")
    one_dose._require_current_production_trainer_authority(  # noqa: SLF001
        verified, command=command
    )
    return command


def _code_authority() -> dict[str, Any]:
    records = [
        {"relative_path": value, **_ref(REPO_ROOT / value, where=value)}
        for value in CODE_SURFACE
    ]
    return {"records": records, "records_sha256": _value_sha256(records)}


def _fresh_outputs(checkpoint: Path, report: Path, receipt: Path) -> None:
    one_dose._require_fresh_outputs(checkpoint, report, receipt)  # noqa: SLF001


def _epoch_outputs(checkpoint: Path, epochs: int) -> list[Path]:
    return [
        train_bc._epoch_checkpoint_path(str(checkpoint), epoch)  # noqa: SLF001
        for epoch in range(1, int(epochs) + 1)
    ]


def _checkpoint_steps(recipe: Mapping[str, Any]) -> tuple[int, ...]:
    raw = str(recipe.get("checkpoint_steps", "") or "").strip()
    if not raw:
        return ()
    try:
        values = tuple(int(token.strip()) for token in raw.split(","))
    except ValueError as error:
        raise ScratchTrainError("scratch checkpoint_steps is malformed") from error
    if (
        any(step <= 0 for step in values)
        or tuple(sorted(set(values))) != values
    ):
        raise ScratchTrainError(
            "scratch checkpoint_steps must be unique, positive, and increasing"
        )
    return values


def _step_outputs(checkpoint: Path, steps: Sequence[int]) -> list[Path]:
    return [
        train_bc._step_checkpoint_path(str(checkpoint), int(step))  # noqa: SLF001
        for step in steps
    ]


def _require_fresh_epoch_outputs(checkpoint: Path, epochs: int) -> None:
    collisions: list[str] = []
    for epoch_path in _epoch_outputs(checkpoint, epochs):
        for path in (
            epoch_path,
            Path(str(epoch_path) + ".optimizer.pt"),
            Path(str(epoch_path) + ".training-progress.json"),
        ):
            if path.exists() or path.is_symlink():
                collisions.append(str(path))
    if collisions:
        raise ScratchTrainError(
            "scratch epoch frontier output already exists: " + ", ".join(collisions)
        )


def _require_fresh_step_outputs(checkpoint: Path, steps: Sequence[int]) -> None:
    collisions = [
        str(path)
        for path in _step_outputs(checkpoint, steps)
        if path.exists() or path.is_symlink()
    ]
    if collisions:
        raise ScratchTrainError(
            "scratch optimizer-step frontier output already exists: "
            + ", ".join(collisions)
        )


def _completed_outputs(
    *,
    checkpoint: Path,
    report: Path,
    epochs: int,
    checkpoint_steps: Sequence[int],
) -> dict[str, Any]:
    terminal = _ref(checkpoint, where="terminal scratch checkpoint")
    report_ref = _ref(report, where="scratch training report")
    epoch_records: list[dict[str, Any]] = []
    for epoch, epoch_path in enumerate(
        _epoch_outputs(checkpoint, epochs), start=1
    ):
        epoch_records.append(
            {
                "epoch": epoch,
                "checkpoint": _ref(
                    epoch_path, where=f"scratch epoch-{epoch} checkpoint"
                ),
                "optimizer": _ref(
                    Path(str(epoch_path) + ".optimizer.pt"),
                    where=f"scratch epoch-{epoch} optimizer",
                ),
                "training_progress": _ref(
                    Path(str(epoch_path) + ".training-progress.json"),
                    where=f"scratch epoch-{epoch} training progress",
                ),
            }
        )
    try:
        report_payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScratchTrainError(f"scratch training report is unreadable: {error}") from error
    if int(report_payload.get("epochs", -1)) != int(epochs):
        raise ScratchTrainError("scratch report completed epoch count drift")
    if report_payload.get("checkpoint_steps_requested") != list(checkpoint_steps):
        raise ScratchTrainError("scratch report optimizer-step request drift")
    raw_intermediate = report_payload.get("intermediate_checkpoints")
    if not isinstance(raw_intermediate, list):
        raise ScratchTrainError("scratch report has no intermediate checkpoint frontier")
    by_step: dict[int, Mapping[str, Any]] = {}
    for raw_record in raw_intermediate:
        if not isinstance(raw_record, dict):
            raise ScratchTrainError("scratch intermediate checkpoint record is malformed")
        step = raw_record.get("optimizer_step")
        if isinstance(step, bool) or not isinstance(step, int) or step in by_step:
            raise ScratchTrainError("scratch intermediate checkpoint step is malformed")
        by_step[step] = raw_record
    optimizer_step_records: list[dict[str, Any]] = []
    for step, step_path in zip(
        checkpoint_steps,
        _step_outputs(checkpoint, checkpoint_steps),
        strict=True,
    ):
        raw_record = by_step.get(int(step))
        if raw_record is None:
            raise ScratchTrainError(
                f"scratch report omitted requested optimizer-step checkpoint {step}"
            )
        checkpoint_ref = _ref(
            step_path, where=f"scratch optimizer-step-{step} checkpoint"
        )
        if (
            Path(str(raw_record.get("checkpoint", ""))).expanduser().resolve(
                strict=False
            )
            != step_path.expanduser().resolve(strict=False)
            or raw_record.get("checkpoint_sha256")
            != checkpoint_ref["file_sha256"]
            or raw_record.get("same_training_trajectory") is not True
            or raw_record.get("optimizer_sidecar") is not None
        ):
            raise ScratchTrainError(
                f"scratch optimizer-step-{step} checkpoint report binding drift"
            )
        optimizer_step_records.append(
            {
                "optimizer_step": int(step),
                "checkpoint": checkpoint_ref,
                "same_training_trajectory": True,
                "optimizer_sidecar_intentionally_omitted": True,
            }
        )
    return {
        "terminal_checkpoint": terminal,
        "training_report": report_ref,
        "epoch_frontier": epoch_records,
        "epoch_frontier_sha256": _value_sha256(epoch_records),
        "optimizer_step_frontier": optimizer_step_records,
        "optimizer_step_frontier_sha256": _value_sha256(optimizer_step_records),
    }


def run(
    args: argparse.Namespace,
    *,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    python_authority = _executable_ref(args.python, where="learner Python")
    python = Path(python_authority["path"])
    checkpoint = args.checkpoint.expanduser().absolute()
    report = args.report.expanduser().absolute()
    plan_receipt = args.receipt.expanduser().absolute()
    execution_receipt_arg = getattr(args, "execution_receipt", None)
    execution_receipt = (
        _execution_receipt_path(plan_receipt)
        if execution_receipt_arg is None
        else execution_receipt_arg.expanduser().absolute()
    )
    output_receipt = execution_receipt if bool(args.go) else plan_receipt
    _fresh_outputs(checkpoint, report, output_receipt)
    verified = verify_inputs(
        lock_path=args.lock,
        data_path=args.data,
        composite_build_receipt=args.composite_build_receipt,
    )
    command = build_train_command(
        verified, python=python, checkpoint=checkpoint, report=report
    )
    plan_authority = _scratch_plan_authority(verified)
    execution_topology = verified["execution_topology"]
    optimization_schedule_authorized = bool(
        execution_topology.get("go_authorized") is True
        and execution_topology.get("optimization_schedule_status")
        == "commissioned_scratch_update_horizon_v1"
    )
    epochs = int(verified["recipe"]["epochs"])
    if "--save-each-epoch" not in command:
        raise ScratchTrainError("scratch command lost its checkpoint frontier")
    checkpoint_steps = _checkpoint_steps(verified["recipe"])
    _require_fresh_epoch_outputs(checkpoint, epochs)
    _require_fresh_step_outputs(checkpoint, checkpoint_steps)
    base = {
        "schema_version": PLAN_SCHEMA,
        "created_unix_ns": time.time_ns(),
        "status": "planned",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "go_authorized": optimization_schedule_authorized,
        "optimization_schedule_authorized": optimization_schedule_authorized,
        "maximum_result": "none_plan_only",
        "contract": _ref(Path(verified["lock_path"]), where="A1 lock"),
        "science_contract": {
            **_ref(current_science.CONTRACT_PATH, where="current science contract"),
            "semantic_sha256": _value_sha256(current_science.load()),
        },
        "initialization": verified["initialization"],
        "model_construction": verified["model_construction"],
        "execution_topology": verified["execution_topology"],
        "logical_recipe": verified["logical_recipe"],
        "effective_recipe": verified["recipe"],
        "composite": {
            "descriptor": _ref(Path(verified["data_path"]), where="descriptor"),
            "descriptor_fingerprint": verified["descriptor_fingerprint"],
            "build_receipt": verified["composite_build_receipt"],
            "source_authority_semantic_sha256": verified[
                "source_authority_semantic_sha256"
            ],
            "validation_split_receipt_sha256": verified[
                "validation_split_receipt_sha256"
            ],
        },
        "plan_authority": plan_authority,
        "python": python_authority,
        "trainer_authority": verified["trainer_authority"],
        "launcher_authority": _code_authority(),
        "command": command,
        "command_sha256": _value_sha256(command),
    }
    if not bool(args.go):
        _write_receipt(plan_receipt, base)
        return base
    plan_payload, plan_ref = _load_matching_plan_receipt(
        plan_receipt, expected=base
    )
    base = {
        key: value
        for key, value in plan_payload.items()
        if key != "receipt_sha256"
    }
    if not optimization_schedule_authorized:
        raise ScratchTrainError(
            "--go requires a commissioned scratch optimizer schedule"
        )
    started = time.time_ns()
    try:
        result = runner(command, cwd=REPO_ROOT, check=False)
    except OSError as error:
        raise ScratchTrainError(f"cannot execute scratch learner: {error}") from error
    returncode = int(result.returncode)
    if returncode != 0:
        failed = {
            **base,
            "schema_version": EXECUTION_SCHEMA,
            "status": "failed",
            "go": True,
            "diagnostic_only": True,
            "promotion_eligible": False,
            "maximum_result": "failed_training_attempt",
            "started_unix_ns": started,
            "finished_unix_ns": time.time_ns(),
            "returncode": returncode,
            "plan_receipt": plan_ref,
        }
        _write_receipt(execution_receipt, failed)
        raise ScratchTrainError(
            f"scratch learner exited with return code {returncode}; "
            f"failure receipt={execution_receipt}"
        )
    outputs = _completed_outputs(
        checkpoint=checkpoint,
        report=report,
        epochs=epochs,
        checkpoint_steps=checkpoint_steps,
    )
    completed = {
        **base,
        "schema_version": EXECUTION_SCHEMA,
        "status": "completed",
        "go": True,
        "diagnostic_only": False,
        "promotion_eligible": False,
        "maximum_result": "training_complete_requires_checkpoint_selection_and_gate",
        "started_unix_ns": started,
        "finished_unix_ns": time.time_ns(),
        "returncode": 0,
        "plan_receipt": plan_ref,
        "outputs": outputs,
    }
    _write_receipt(execution_receipt, completed)
    return completed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--composite-build-receipt", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--execution-receipt",
        type=Path,
        help=(
            "Fresh execution receipt path used only with --go. Defaults to an "
            "immutable sibling of --receipt; --receipt itself is always the "
            "pre-existing authenticated plan."
        ),
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--go",
        action="store_true",
        help="Execute the commissioned 8-GPU scratch learner instead of plan-only.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        payload = run(parse_args(argv))
    except ScratchTrainError as error:
        print(f"a1_scratch_train: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
