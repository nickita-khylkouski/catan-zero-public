#!/usr/bin/env python3
"""Sealed native-scratch plan for the current coherent-public A1 learner.

The historical one-dose executor is checkpoint-initialized by design.  This
entrypoint is the separate planning projection for the current science
contract's native v3 model and fresh optimizer.  It cannot execute training
until a production scratch optimizer schedule is reviewed and sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
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
CHILD_AUTHORITY_SCHEMA = "a1-coherent-scratch-plan-authority-v2"
CODE_SURFACE = (
    "tools/a1_scratch_train.py",
    "tools/a1_current_science_contract.py",
    "tools/a1_pre_wave_contract.py",
    "tools/a1_build_post_wave_composite.py",
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
        verified = one_dose.bind_training_topology(
            verified,
            topology=str(science["execution_topology"]["name"]),
            gpu=0,
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
        ("--entity-feature-adapter-version", "entity_feature_adapter_version"),
        ("--meaningful-public-history-pooling", "meaningful_public_history_pooling"),
        ("--event-history-limit", "event_history_limit"),
    ):
        _add(command, flag, model[key])
    command.extend(
        [
            "--static-action-residual",
            "--legal-action-value-residual",
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
        ("batch_size", "--batch-size"),
        ("grad_accum_steps", "--grad-accum-steps"),
        ("optimizer", "--optimizer"),
        ("lr", "--lr"),
        ("lr_warmup_steps", "--lr-warmup-steps"),
        ("lr_schedule", "--lr-schedule"),
        ("weight_decay", "--weight-decay"),
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
            "--target-reliability-confidence-weighting",
            "--no-per-game-policy-surprise-weighting",
            "--checkpoint",
            str(checkpoint),
            "--report",
            str(report),
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    python = _regular_file(args.python, where="learner Python")
    checkpoint = args.checkpoint.expanduser().absolute()
    report = args.report.expanduser().absolute()
    receipt = args.receipt.expanduser().absolute()
    _fresh_outputs(checkpoint, report, receipt)
    verified = verify_inputs(
        lock_path=args.lock,
        data_path=args.data,
        composite_build_receipt=args.composite_build_receipt,
    )
    command = build_train_command(
        verified, python=python, checkpoint=checkpoint, report=report
    )
    plan_authority = _scratch_plan_authority(verified)
    base = {
        "schema_version": PLAN_SCHEMA,
        "created_unix_ns": time.time_ns(),
        "status": "planned",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "go_authorized": False,
        "optimization_schedule_authorized": False,
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
        "python": _ref(python, where="learner Python"),
        "trainer_authority": verified["trainer_authority"],
        "launcher_authority": _code_authority(),
        "command": command,
        "command_sha256": _value_sha256(command),
    }
    _write_receipt(receipt, base)
    return base


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--composite-build-receipt", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
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
