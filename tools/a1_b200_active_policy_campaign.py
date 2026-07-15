#!/usr/bin/env python3
"""Seal and run the coherent-n128 active-policy exposure campaign.

This is a thin campaign layer over ``a1_one_dose_train``.  It does not invent
another trainer.  Four 8xB200 arms independently reload the same exact f7
function-preserving initializer with fresh Adam and differ only in the number
of auxiliary policy-active rows sampled per rank and optimizer step.

The tool deliberately has two gates before optimizer launch:

* ``admit-corpus`` turns the canonical completion receipt plus the independent
  target eligibility inventory into one immutable coherent-corpus admission
  receipt.
* ``plan`` refuses to exist without that admission receipt and the exact f7
  architecture-upgrade receipt.

Selection is similarly explicit.  Existing read-only tools measure functional
parent KL and layer drift at steps 8/12/16/32/64/128.  The selected exposure is
the eligible arm with the largest terminal active-policy teacher-gap closure
under operator-declared parent-KL and trunk-drift budgets.  Candidate chaining
is never an allowed transition.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from tools import a1_b200_lr_dose_campaign as base_campaign  # noqa: E402
from tools import a1_current_science_contract as current_science  # noqa: E402
from tools import a1_function_preserving_upgrade as architecture_upgrade  # noqa: E402
from tools import a1_one_dose_train as one_dose  # noqa: E402
from tools.fleet import a1_coherent_target_rd_executor as coherent_executor  # noqa: E402


SCHEMA = "a1-b200-active-policy-exposure-campaign-v1"
ADMISSION_SCHEMA = "a1-coherent-n128-corpus-admission-v1"
FINGERPRINT_SCHEMA = "a1-b200-active-policy-fingerprint-v1"
SELECTION_SCHEMA = "a1-b200-active-policy-selection-v1"
INVENTORY_SCHEMA = "a1-target-eligibility-inventory-v1"
COMPLETION_RECEIPT_SCHEMA = "a1-coherent-target-rd-completion-receipt-v1"

TARGET_CONTRACT_RELATIVE = Path(
    "configs/operations/a1-target-identity-coherent-n128-rd-v1/contract.json"
)
EXPECTED_TARGET_CONTRACT_SHA256 = (
    "sha256:a37d3a707d4cdb05dd2174f4a8b1125d535876971e8efee6db12061c0713dd4f"
)
EXPECTED_CORPUS_PRODUCER_SHA256 = (
    "sha256:6817ab054506f962a758ebf48addce5cc7eb801bf451cf2d02b62fb91f5da39c"
)
EXPECTED_F7_PARENT_SHA256 = (
    "sha256:f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4"
)
TARGET_INFORMATION_REGIME = "public_belief_single_tree_v1"
SEARCH_EVIDENCE_SCHEMA = "gumbel_root_search_evidence_v1"
EXPECTED_GAMES = 8_192
WORLD_SIZE = 8
LOCAL_BATCH_SIZE = 512
GLOBAL_BATCH_SIZE = WORLD_SIZE * LOCAL_BATCH_SIZE
MAX_STEPS = 128
CHECKPOINT_STEPS = (8, 12, 16, 32, 64, 128)
INTERMEDIATE_STEPS = CHECKPOINT_STEPS[:-1]
TRAIN_DIAGNOSTIC_CADENCE = 16
OBJECTIVE_GRADIENT_CADENCE = 64
BASE_AUX_ACTIVE_BATCH_SIZE = 463
ARM_MULTIPLIERS = {
    "P10": 0.10,
    "P25": 0.25,
    "P50": 0.50,
    "P100": 1.00,
}
ARMS = {
    arm: {
        "active_policy_branch_multiplier": multiplier,
        "policy_aux_active_batch_size": int(
            math.floor(BASE_AUX_ACTIVE_BATCH_SIZE * multiplier + 0.5)
        ),
    }
    for arm, multiplier in ARM_MULTIPLIERS.items()
}
EXPECTED_SEARCH_EVIDENCE_COLUMNS = {
    "search_evidence_version",
    "search_evidence_offsets",
    "search_visit_counts_flat",
    "search_completed_q_flat",
}
R2_UPDATE_FRONTIER_REFERENCE = {
    "source": "a1-r2-early-62-of-64-aggregate",
    "reference_arm": "B",
    "games_per_matchup": 256,
    "win_rate_vs_f7": 0.570,
    "win_rate_vs_v5": 0.504,
    "active_policy_teacher_gap_closure": 0.0495,
    "global_parameter_relative_l2": 0.017204,
    "role": "diagnostic_reference_not_promotion_authority",
}


class CampaignError(RuntimeError):
    """An immutable campaign input or result is semantically invalid."""


def _file_sha256(path: Path) -> str:
    return base_campaign._file_sha256(path)  # noqa: SLF001


def _value_sha256(value: object) -> str:
    return base_campaign._value_sha256(value)  # noqa: SLF001


def _canonical_json(value: object) -> str:
    return base_campaign._canonical_bytes(value).decode("ascii")  # noqa: SLF001


def _regular_file(path: Path, *, where: str) -> Path:
    try:
        return base_campaign._regular_file(path, where=where)  # noqa: SLF001
    except base_campaign.CampaignError as error:
        raise CampaignError(str(error)) from error


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    destination = path.expanduser().resolve(strict=False)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise CampaignError(f"immutable output is not a regular file: {destination}")
        if destination.read_text(encoding="utf-8") != rendered:
            raise CampaignError(f"immutable output already exists with drift: {destination}")
        return
    base_campaign._write_json(destination, payload)  # noqa: SLF001


def _load_json(path: Path, *, where: str) -> tuple[Path, dict[str, Any]]:
    resolved = _regular_file(path, where=where)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot load {where}: {error}") from error
    if not isinstance(payload, dict):
        raise CampaignError(f"{where} must contain one JSON object")
    return resolved, payload


def _load_signed(
    path: Path,
    *,
    where: str,
    schema: str,
    digest_field: str,
) -> tuple[Path, dict[str, Any]]:
    resolved, payload = _load_json(path, where=where)
    unsigned = dict(payload)
    stated = unsigned.pop(digest_field, None)
    if payload.get("schema_version") != schema or stated != _value_sha256(unsigned):
        raise CampaignError(f"{where} schema or semantic digest drift")
    return resolved, payload


def _parse_bindings(
    values: Sequence[str], *, allowed: set[str], label: str
) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for raw in values:
        key, separator, value = raw.partition("=")
        if separator != "=" or key not in allowed or key in parsed or not value:
            raise CampaignError(
                f"{label} must contain each unique NAME=/absolute/path binding"
            )
        parsed[key] = Path(value)
    if set(parsed) != allowed:
        raise CampaignError(
            f"{label} requires exactly {sorted(allowed)}, got {sorted(parsed)}"
        )
    return parsed


def _verify_target_contract(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved, contract = _load_json(path, where="coherent target contract")
    unsigned = dict(contract)
    stated = unsigned.pop("contract_sha256", None)
    if (
        contract.get("schema_version") != "a1-coherent-target-rd-contract-v1"
        or stated != _value_sha256(unsigned)
        or stated != EXPECTED_TARGET_CONTRACT_SHA256
        or contract.get("target_information_regime") != TARGET_INFORMATION_REGIME
        or contract.get("producer_checkpoint", {}).get("sha256")
        != EXPECTED_CORPUS_PRODUCER_SHA256
        or contract.get("execution", {}).get("total_games") != EXPECTED_GAMES
        or contract.get("acceptance", {}).get("games_completed") != EXPECTED_GAMES
    ):
        raise CampaignError("coherent target contract identity/acceptance drift")
    return resolved, contract


def _verify_completion_receipt(
    path: Path, *, contract_path: Path, contract: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Replay the canonical collector instead of duplicating fleet closure."""

    resolved, receipt = _load_signed(
        path,
        where="coherent completion receipt",
        schema=COMPLETION_RECEIPT_SCHEMA,
        digest_field="receipt_sha256",
    )
    launch = receipt.get("launch_receipt")
    if not isinstance(launch, dict):
        raise CampaignError("coherent completion receipt lost its launch binding")
    launch_path = _regular_file(
        Path(str(launch.get("path", ""))), where="completion-bound launch receipt"
    )
    if _file_sha256(launch_path) != launch.get("file_sha256"):
        raise CampaignError("completion-bound launch receipt bytes drifted")
    try:
        replayed = coherent_executor._verify_existing_completion(  # noqa: SLF001
            resolved,
            contract=contract,
            launch_file_sha256=_file_sha256(launch_path),
        )
    except (coherent_executor.ExecutorError, OSError, ValueError) as error:
        raise CampaignError(f"coherent completion receipt refused: {error}") from error
    seeds = replayed.get("seed_inventory")
    totals = replayed.get("totals")
    operator = replayed.get("coherent_operator")
    if (
        Path(str(replayed.get("contract", {}).get("path", ""))).resolve(
            strict=True
        )
        != contract_path
        or replayed.get("contract", {}).get("contract_sha256")
        != contract["contract_sha256"]
        or replayed.get("producer_checkpoint", {}).get("sha256")
        != EXPECTED_CORPUS_PRODUCER_SHA256
        or replayed.get("target_information_regime")
        != TARGET_INFORMATION_REGIME
        or replayed.get("search_evidence_schema") != SEARCH_EVIDENCE_SCHEMA
        or not isinstance(operator, dict)
        or operator.get("semantic_sha256") != _value_sha256(contract["operator"])
        or not isinstance(seeds, dict)
        or seeds.get("count") != EXPECTED_GAMES
        or seeds.get("unique_count") != EXPECTED_GAMES
        or seeds.get("minimum") != contract["execution"]["seed_start"]
        or seeds.get("maximum_exclusive") != contract["execution"]["seed_end"]
        or seeds.get("contiguous") is not True
        or not isinstance(totals, dict)
        or totals.get("games_completed") != EXPECTED_GAMES
        or totals.get("games_failed") != 0
        or totals.get("games_truncated") != 0
        or totals.get("complete_trace_games") != EXPECTED_GAMES
        or totals.get("incomplete_trace_games") != 0
        or int(totals.get("policy_active_rows", 0)) <= 0
        or len(replayed.get("lanes", ())) != len(contract["execution"]["lanes"])
    ):
        raise CampaignError("coherent completion receipt semantic closure drift")
    return resolved, replayed


def _admit_corpus(args: argparse.Namespace) -> dict[str, Any]:
    contract_path, contract = _verify_target_contract(args.contract)
    completion_path, completion = _verify_completion_receipt(
        args.completion_receipt, contract_path=contract_path, contract=contract
    )

    inventory_path, inventory = _load_signed(
        args.inventory,
        where="target eligibility inventory",
        schema=INVENTORY_SCHEMA,
        digest_field="inventory_sha256",
    )
    corpus_meta, corpus_meta_payload = _load_json(
        args.corpus_meta, where="coherent corpus_meta"
    )
    validation = _regular_file(
        args.validation_manifest, where="whole-game validation manifest"
    )
    direct = inventory.get("direct_corpora")
    aggregate = inventory.get("aggregate")
    rd_contract = inventory.get("rd_contract")
    if not isinstance(direct, list) or len(direct) != 1:
        raise CampaignError("target inventory must bind exactly one coherent corpus")
    item = direct[0]
    meta_ref = item.get("corpus_meta") if isinstance(item, dict) else None
    trace = item.get("exact_root_reanalysis") if isinstance(item, dict) else None
    active_regimes = (
        item.get("policy_active_target_regime_rows")
        if isinstance(item, dict)
        else None
    )
    if (
        inventory.get("required_target_information_regime")
        != TARGET_INFORMATION_REGIME
        or not isinstance(rd_contract, dict)
        or rd_contract.get("contract_sha256") != EXPECTED_TARGET_CONTRACT_SHA256
        or not isinstance(meta_ref, dict)
        or Path(str(meta_ref.get("path", ""))).resolve(strict=True) != corpus_meta
        or meta_ref.get("sha256") != _file_sha256(corpus_meta)
        or not isinstance(meta_ref.get("payload_inventory_sha256"), str)
        or item.get("selected_games") != EXPECTED_GAMES
        or not isinstance(active_regimes, dict)
        or set(active_regimes) != {TARGET_INFORMATION_REGIME}
        or int(active_regimes.get(TARGET_INFORMATION_REGIME, 0)) <= 0
        or item.get("incompatible_policy_active_rows") != 0
        or item.get("policy_targets_eligible_for_requested_learner") is not True
        or not EXPECTED_SEARCH_EVIDENCE_COLUMNS.issubset(
            set(item.get("search_evidence_columns", ()))
        )
        or not isinstance(trace, dict)
        or trace.get("complete_action_trace_game_count") != EXPECTED_GAMES
        or trace.get("incomplete_action_trace_game_count") != 0
        or trace.get("complete_action_trace_fraction") != 1.0
        or trace.get("full_corpus_replayable") is not True
        or not isinstance(aggregate, dict)
        or aggregate.get("incompatible_policy_active_rows") != 0
        or aggregate.get("policy_targets_eligible_for_requested_learner") is not True
        or aggregate.get("old_targets_remain_policy_active") is not False
    ):
        raise CampaignError(
            "coherent inventory does not prove complete two-seat, PIMC-free n128 targets"
        )

    completed_shards = [
        {
            "path": str(Path(str(shard["path"])).resolve(strict=True)),
            "size_bytes": int(shard["size_bytes"]),
            "sha256": str(shard["sha256"]),
        }
        for lane in completion["lanes"]
        for worker in lane["workers"]
        for shard in worker["shards"]
    ]
    memmap_shards = corpus_meta_payload.get("source_shard_inventory")
    if (
        not isinstance(memmap_shards, list)
        or memmap_shards != completed_shards
        or corpus_meta_payload.get("source_shard_inventory_sha256")
        != _value_sha256(completed_shards)
    ):
        raise CampaignError(
            "coherent memmap input inventory is not exactly the completion-bound "
            "NPZ sequence"
        )

    payload: dict[str, Any] = {
        "schema_version": ADMISSION_SCHEMA,
        "status": "admitted_for_diagnostic_policy_distillation",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "contract": {
            "path": str(contract_path),
            "file_sha256": _file_sha256(contract_path),
            "contract_sha256": contract["contract_sha256"],
        },
        "completion_receipt": {
            "path": str(completion_path),
            "file_sha256": _file_sha256(completion_path),
            "receipt_sha256": completion["receipt_sha256"],
            "payload_inventory_sha256": completion[
                "payload_inventory_sha256"
            ],
        },
        "target_eligibility_inventory": {
            "path": str(inventory_path),
            "file_sha256": _file_sha256(inventory_path),
            "inventory_sha256": inventory["inventory_sha256"],
        },
        "corpus": {
            "data_path": str(corpus_meta.parent),
            "corpus_meta_path": str(corpus_meta),
            "corpus_meta_file_sha256": _file_sha256(corpus_meta),
            "payload_inventory_sha256": meta_ref["payload_inventory_sha256"],
            "validation_manifest": {
                "path": str(validation),
                "file_sha256": _file_sha256(validation),
            },
            "producer_checkpoint_sha256": EXPECTED_CORPUS_PRODUCER_SHA256,
            "target_information_regime": TARGET_INFORMATION_REGIME,
            "search_evidence_schema": SEARCH_EVIDENCE_SCHEMA,
            "selected_games": EXPECTED_GAMES,
            "seed_start": int(contract["execution"]["seed_start"]),
            "seed_end": int(contract["execution"]["seed_end"]),
            "source_npz_count": len(completed_shards),
            "source_npz_inventory_sha256": completion[
                "npz_inventory_sha256"
            ],
            "memmap_source_shard_inventory_sha256": corpus_meta_payload[
                "source_shard_inventory_sha256"
            ],
            "complete_two_seat_trace_games": EXPECTED_GAMES,
            "incompatible_policy_active_rows": 0,
        },
        "policy_distillation_contract": {
            "coherent_public_n128_only": True,
            "legacy_pimc_rows_allowed": False,
            "policy_active_rows": int(
                active_regimes[TARGET_INFORMATION_REGIME]
            ),
        },
    }
    payload["admission_sha256"] = _value_sha256(payload)
    return payload


def _load_admission(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved, admission = _load_signed(
        path,
        where="coherent corpus admission",
        schema=ADMISSION_SCHEMA,
        digest_field="admission_sha256",
    )
    corpus = admission.get("corpus")
    policy = admission.get("policy_distillation_contract")
    if (
        admission.get("status") != "admitted_for_diagnostic_policy_distillation"
        or admission.get("diagnostic_only") is not True
        or admission.get("promotion_eligible") is not False
        or admission.get("contract", {}).get("contract_sha256")
        != EXPECTED_TARGET_CONTRACT_SHA256
        or not isinstance(corpus, dict)
        or corpus.get("producer_checkpoint_sha256")
        != EXPECTED_CORPUS_PRODUCER_SHA256
        or corpus.get("target_information_regime") != TARGET_INFORMATION_REGIME
        or corpus.get("search_evidence_schema") != SEARCH_EVIDENCE_SCHEMA
        or corpus.get("selected_games") != EXPECTED_GAMES
        or corpus.get("seed_end", -1) - corpus.get("seed_start", -1)
        != EXPECTED_GAMES
        or corpus.get("complete_two_seat_trace_games") != EXPECTED_GAMES
        or corpus.get("incompatible_policy_active_rows") != 0
        or not isinstance(policy, dict)
        or policy.get("coherent_public_n128_only") is not True
        or policy.get("legacy_pimc_rows_allowed") is not False
    ):
        raise CampaignError("coherent corpus admission semantics drifted")
    for record, where in (
        (admission["contract"], "target contract"),
        (admission["completion_receipt"], "completion receipt"),
        (admission["target_eligibility_inventory"], "target inventory"),
        (corpus["validation_manifest"], "validation manifest"),
    ):
        artifact = _regular_file(Path(str(record["path"])), where=where)
        if _file_sha256(artifact) != record["file_sha256"]:
            raise CampaignError(f"coherent admission {where} bytes drifted")
    data = Path(str(corpus["data_path"])).expanduser().resolve(strict=True)
    meta = _regular_file(
        Path(str(corpus["corpus_meta_path"])), where="coherent corpus_meta"
    )
    if not data.is_dir() or meta.parent != data:
        raise CampaignError("coherent corpus admission data root drifted")
    if _file_sha256(meta) != corpus["corpus_meta_file_sha256"]:
        raise CampaignError("coherent corpus_meta bytes drifted after admission")
    return resolved, admission


def _arm_overrides(arm: str, science_recipe: Mapping[str, Any]) -> dict[str, Any]:
    if arm not in ARMS:
        raise CampaignError(f"unknown active-policy arm {arm!r}")
    return {
        "epochs": 1,
        "max_steps": MAX_STEPS,
        "public_card_lr_mult": float(science_recipe["public_card_lr_mult"]),
        "per_game_policy_surprise_weighting": bool(
            science_recipe["per_game_policy_surprise_weighting"]
        ),
        "forced_row_value_action_type_weights": str(
            science_recipe["forced_row_value_action_type_weights"]
        ),
        "policy_aux_active_batch_size": int(
            ARMS[arm]["policy_aux_active_batch_size"]
        ),
    }


def _parent_authority(
    *,
    verified: Mapping[str, Any],
    upgrade: Mapping[str, Any],
    admission_path: Path,
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": one_dose.INDEPENDENT_PARENT_AUTHORITY_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "corpus_binding": {
            "data_path": str(Path(str(verified["data_path"])).resolve(strict=True)),
            "corpus_meta_file_sha256": verified.get("corpus_meta_file_sha256"),
            "payload_inventory_sha256": verified.get("payload_inventory_sha256"),
            "data_fingerprint": verified.get("data_fingerprint"),
            "producer_checkpoint": copy.deepcopy(verified.get("producer")),
            "coherent_corpus_admission": {
                "path": str(admission_path),
                "file_sha256": _file_sha256(admission_path),
                "admission_sha256": admission["admission_sha256"],
            },
        },
        "learner_parent": copy.deepcopy(upgrade["source"]),
        "function_preserving_upgrade": {
            "module": upgrade["module"],
            "receipt_file_sha256": upgrade["receipt"]["sha256"],
            "receipt_sha256": upgrade["receipt_sha256"],
            "upgraded_initializer": copy.deepcopy(upgrade["upgraded_initializer"]),
        },
    }
    payload["authority_sha256"] = _value_sha256(payload)
    return payload


def _one_dose_invocation(campaign: Mapping[str, Any], arm: str) -> list[str]:
    inputs = campaign["inputs"]
    arm_root = Path(campaign["output_root"]) / "arms" / arm
    return [
        str(inputs["python"]),
        str(inputs["one_dose_trainer"]),
        "--lock",
        str(inputs["lock"]),
        "--data",
        str(inputs["data"]),
        "--validation-manifest",
        str(inputs["validation_manifest"]),
        "--coherent-corpus-admission",
        str(inputs["coherent_corpus_receipt"]),
        "--architecture-upgrade-receipt",
        str(inputs["architecture_upgrade_receipt"]),
        "--independent-parent-authority",
        str(inputs["independent_parent_authority"]),
        "--checkpoint",
        str(arm_root / "candidate.pt"),
        "--report",
        str(arm_root / "train.report.json"),
        "--receipt",
        str(arm_root / "one-dose.receipt.json"),
        "--python",
        str(inputs["python"]),
        "--gpu",
        "0",
        "--topology",
        "b200-8gpu-ddp",
        "--ddp-canary-receipt",
        str(inputs["ddp_canary_receipt"]),
        "--ablation-id",
        f"coherent-n128-active-policy-{arm.lower()}",
        "--recipe-overrides-json",
        _canonical_json(campaign["arms"][arm]["recipe_overrides"]),
        "--ablation-code-tree-sha256",
        str(inputs["reviewed_code_tree_sha256"]),
        "--reviewed-lock-file-sha256",
        str(inputs["reviewed_lock_file_sha256"]),
        "--diagnostic-dose-curve",
        "--diagnostic-checkpoint-steps",
        ",".join(map(str, INTERMEDIATE_STEPS)),
    ]


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    admission_path, admission = _load_admission(args.coherent_corpus_receipt)
    corpus = admission["corpus"]
    data = Path(str(corpus["data_path"])).resolve(strict=True)
    validation = Path(str(corpus["validation_manifest"]["path"])).resolve(strict=True)
    lock = _regular_file(args.lock, where="sealed coherent learner lock")
    python = base_campaign._python_executable(args.python)  # noqa: SLF001
    trainer = _regular_file(
        REPO_ROOT / "tools" / "a1_one_dose_train.py", where="one-dose trainer"
    )
    canary = _regular_file(args.ddp_canary_receipt, where="8xB200 DDP canary")
    upgrade_path = _regular_file(
        args.architecture_upgrade_receipt, where="f7 architecture upgrade receipt"
    )
    try:
        upgrade = architecture_upgrade.verify_receipt(upgrade_path)
    except architecture_upgrade.UpgradeError as error:
        raise CampaignError(f"f7 architecture upgrade receipt refused: {error}") from error
    if (
        upgrade.get("source", {}).get("sha256") != EXPECTED_F7_PARENT_SHA256
        or upgrade.get("module")
        != architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_V2
        or upgrade.get("forward_identical_at_init") is not True
        or float(upgrade.get("forward_max_diff", -1.0)) != 0.0
        or upgrade.get("shared_parameters_bit_identical") is not True
    ):
        raise CampaignError(
            "campaign requires exact f7 plus the combined public-card/history-v2 "
            "function-preserving initializer"
        )
    reviewed_lock = (
        base_campaign._normalize_sha256(  # noqa: SLF001
            args.reviewed_lock_file_sha256, where="reviewed lock"
        )
        if args.reviewed_lock_file_sha256
        else _file_sha256(lock)
    )
    if reviewed_lock != _file_sha256(lock):
        raise CampaignError("reviewed lock file digest mismatch")
    try:
        verified = one_dose.verify_training_inputs(
            lock_path=lock,
            data_path=data,
            validation_path=validation,
            composite_build_receipt=None,
            reviewed_lock_file_sha256=reviewed_lock,
            coherent_corpus_admission=admission_path,
        )
    except one_dose.ExecutorError as error:
        raise CampaignError(f"coherent learner input binding refused: {error}") from error
    if (
        verified.get("data_kind") != "coherent_direct_memmap_v1"
        or verified.get("coherent_direct_corpus_binding", {}).get(
            "target_contract_sha256"
        )
        != EXPECTED_TARGET_CONTRACT_SHA256
        or verified.get("producer", {}).get("sha256")
        != EXPECTED_CORPUS_PRODUCER_SHA256
        or verified.get("payload_inventory_sha256")
        != corpus["payload_inventory_sha256"]
        or verified.get("corpus_meta_file_sha256")
        != corpus["corpus_meta_file_sha256"]
    ):
        raise CampaignError("sealed learner lock differs from coherent corpus admission")

    science = current_science.load()
    learner = science["learner"]
    science_recipe = learner["training_recipe"]
    if (
        science["target_information_regime"] != TARGET_INFORMATION_REGIME
        or learner["topology"] != "b200-8gpu-ddp"
        or learner["architecture_upgrade_module"]
        != architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_V2
        or science_recipe["policy_loss_weight"] != 1.0
        or science_recipe["value_loss_weight"] != 0.25
    ):
        raise CampaignError("canonical coherent learner contract drifted")

    output_root = args.output_root.expanduser().resolve(strict=False)
    authority_path = output_root / "independent-parent.authority.json"
    authority = _parent_authority(
        verified=verified,
        upgrade=upgrade,
        admission_path=admission_path,
        admission=admission,
    )
    _write_immutable(authority_path, authority)
    reviewed_code = base_campaign._normalize_sha256(  # noqa: SLF001
        args.reviewed_code_tree_sha256, where="reviewed code tree"
    )
    for name, value in (
        ("max parent KL", float(args.max_parent_kl)),
        ("max trunk relative L2", float(args.max_trunk_relative_l2)),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise CampaignError(f"{name} must be finite and positive")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "purpose": "coherent_n128_active_policy_exposure_curve",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "lineage_contract": {
            "corpus_producer_sha256": EXPECTED_CORPUS_PRODUCER_SHA256,
            "learner_parent_sha256": EXPECTED_F7_PARENT_SHA256,
            "upgraded_initializer_sha256": upgrade["upgraded_initializer"]["sha256"],
            "every_arm_restarts_from_upgraded_initializer": True,
            "fresh_adam_every_arm": True,
            "candidate_chaining_forbidden": True,
        },
        "topology": {
            "name": "b200-8gpu-ddp",
            "world_size": WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
        },
        "trajectory": {
            "optimizer_steps": MAX_STEPS,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "terminal_step": MAX_STEPS,
            "base_row_draws": GLOBAL_BATCH_SIZE * MAX_STEPS,
        },
        "treatment": {
            "field": "policy_aux_active_batch_size",
            "interpretation": (
                "auxiliary draws conditioned on authenticated positive-policy rows; "
                "global policy/value losses remain unchanged"
            ),
            "base_aux_active_batch_size": BASE_AUX_ACTIVE_BATCH_SIZE,
        },
        "canonical_learner": {
            "science_contract": str(current_science.CONTRACT_PATH),
            "science_contract_file_sha256": _file_sha256(
                current_science.CONTRACT_PATH
            ),
            "training_recipe": copy.deepcopy(science_recipe),
            "training_recipe_sha256": _value_sha256(science_recipe),
        },
        "arms": {
            arm: {
                **values,
                "optimizer_steps": MAX_STEPS,
                "recipe_overrides": _arm_overrides(arm, science_recipe),
                "expected_aux_active_row_draws": (
                    int(values["policy_aux_active_batch_size"])
                    * WORLD_SIZE
                    * MAX_STEPS
                ),
                "output_subdir": f"arms/{arm}",
            }
            for arm, values in ARMS.items()
        },
        "selection_contract": {
            "functional_surface": "validation_policy_active_multi_action_rows",
            "max_parent_kl": float(args.max_parent_kl),
            "max_trunk_relative_l2": float(args.max_trunk_relative_l2),
            "all_checkpoint_fingerprints_required": True,
            "eligibility": "every checkpoint stays within both drift budgets",
            "objective": "max_terminal_active_policy_teacher_gap_closure",
            "reference_update_frontier": copy.deepcopy(
                R2_UPDATE_FRONTIER_REFERENCE
            ),
            "reference_role": (
                "target/rationale only; coherent arms still require explicit "
                "playing-strength evaluation"
            ),
            "tie_break": [
                "min_terminal_parent_kl",
                "min_terminal_trunk_relative_l2",
                "min_aux_exposure",
            ],
            "playing_strength_evaluation_required_before_promotion": True,
        },
        "inputs": {
            "python": str(python),
            "one_dose_trainer": str(trainer),
            "one_dose_trainer_sha256": _file_sha256(trainer),
            "lock": str(lock),
            "lock_file_sha256": _file_sha256(lock),
            "data": str(data),
            "corpus_meta_file_sha256": corpus["corpus_meta_file_sha256"],
            "payload_inventory_sha256": corpus["payload_inventory_sha256"],
            "validation_manifest": str(validation),
            "validation_manifest_file_sha256": _file_sha256(validation),
            "coherent_corpus_receipt": str(admission_path),
            "coherent_corpus_receipt_file_sha256": _file_sha256(admission_path),
            "coherent_corpus_admission_sha256": admission["admission_sha256"],
            "architecture_upgrade_receipt": str(upgrade_path),
            "architecture_upgrade_receipt_file_sha256": _file_sha256(upgrade_path),
            "independent_parent_authority": str(authority_path),
            "independent_parent_authority_file_sha256": _file_sha256(authority_path),
            "independent_parent_authority_sha256": authority["authority_sha256"],
            "ddp_canary_receipt": str(canary),
            "ddp_canary_receipt_file_sha256": _file_sha256(canary),
            "reviewed_code_tree_sha256": reviewed_code,
            "reviewed_lock_file_sha256": reviewed_lock,
        },
        "output_root": str(output_root),
    }
    payload["commands"] = {
        arm: _one_dose_invocation(payload, arm) for arm in ARMS
    }
    payload["campaign_sha256"] = _value_sha256(payload)
    return payload


def _load_campaign(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved, campaign = _load_signed(
        path,
        where="active-policy campaign",
        schema=SCHEMA,
        digest_field="campaign_sha256",
    )
    if set(campaign.get("arms", {})) != set(ARMS):
        raise CampaignError("active-policy campaign arm set drifted")
    return resolved, campaign


def _verify_campaign_inputs(campaign: Mapping[str, Any]) -> None:
    inputs = campaign["inputs"]
    checks = (
        ("one_dose_trainer", "one_dose_trainer_sha256"),
        ("lock", "lock_file_sha256"),
        ("validation_manifest", "validation_manifest_file_sha256"),
        ("coherent_corpus_receipt", "coherent_corpus_receipt_file_sha256"),
        ("architecture_upgrade_receipt", "architecture_upgrade_receipt_file_sha256"),
        ("independent_parent_authority", "independent_parent_authority_file_sha256"),
        ("ddp_canary_receipt", "ddp_canary_receipt_file_sha256"),
    )
    for path_key, digest_key in checks:
        path = _regular_file(Path(str(inputs[path_key])), where=path_key)
        if _file_sha256(path) != inputs[digest_key]:
            raise CampaignError(f"campaign input bytes changed: {path_key}")
    data = Path(str(inputs["data"])).expanduser().resolve(strict=True)
    meta = _regular_file(data / "corpus_meta.json", where="coherent corpus_meta")
    if not data.is_dir() or _file_sha256(meta) != inputs["corpus_meta_file_sha256"]:
        raise CampaignError("coherent corpus_meta changed after planning")
    _load_admission(Path(str(inputs["coherent_corpus_receipt"])))


def _option(command: Sequence[str], flag: str) -> str:
    try:
        return base_campaign._option(command, flag)  # noqa: SLF001
    except base_campaign.CampaignError as error:
        raise CampaignError(str(error)) from error


def _dry_run_arm(campaign: Mapping[str, Any], arm: str) -> dict[str, Any]:
    invocation = _one_dose_invocation(campaign, arm)
    try:
        plan = base_campaign._one_dose_dry_run(invocation)  # noqa: SLF001
    except base_campaign.CampaignError as error:
        raise CampaignError(str(error)) from error
    command = [str(value) for value in plan["command"]]
    initializer = _regular_file(
        Path(_option(command, "--init-checkpoint")), where="rendered f7 initializer"
    )
    expected_initializer = campaign["lineage_contract"][
        "upgraded_initializer_sha256"
    ]
    learner_parent = plan.get("learner_lineage_parent")
    try:
        coherent_binding = json.loads(
            _option(command, "--a1-coherent-corpus-binding-json")
        )
    except json.JSONDecodeError as error:
        raise CampaignError(f"arm {arm} coherent binding is malformed") from error
    if not isinstance(coherent_binding, dict):
        raise CampaignError(f"arm {arm} coherent binding is not an object")
    if (
        _file_sha256(initializer) != expected_initializer
        or "--no-resume-optimizer" not in command
        or not any(token == "--nproc_per_node=8" for token in command)
        or int(_option(command, "--max-steps")) != MAX_STEPS
        or _option(command, "--checkpoint-steps")
        != ",".join(map(str, INTERMEDIATE_STEPS))
        or int(_option(command, "--policy-aux-active-batch-size"))
        != int(ARMS[arm]["policy_aux_active_batch_size"])
        or int(_option(command, "--train-diagnostics-every-batches"))
        != TRAIN_DIAGNOSTIC_CADENCE
        or int(
            _option(command, "--objective-gradient-interference-every-batches")
        )
        != OBJECTIVE_GRADIENT_CADENCE
        # Parent KL is an observed selection fingerprint in this campaign, not
        # an optimizer constraint.  Enabling the adaptive controller would
        # cause each exposure arm to follow a different effective objective.
        or "--policy-kl-target" in command
        or coherent_binding.get("schema_version")
        != one_dose.train_bc.COHERENT_DIRECT_CORPUS_BINDING_SCHEMA
        or coherent_binding.get("diagnostic_only") is not True
        or coherent_binding.get("promotion_eligible") is not False
        or not isinstance(learner_parent, dict)
        or learner_parent.get("role") != "diagnostic_independent_parent"
        or learner_parent.get("checkpoint", {}).get("sha256")
        != EXPECTED_F7_PARENT_SHA256
    ):
        raise CampaignError(f"rendered arm {arm} lost independent f7/fresh-Adam dose")
    effective = plan.get("learner_ablation", {}).get("effective_recipe")
    if not isinstance(effective, dict):
        raise CampaignError(f"rendered arm {arm} lost its effective learner recipe")
    canonical = campaign["canonical_learner"]["training_recipe"]
    invariants = {
        "lr": canonical["lr"],
        "lr_warmup_steps": canonical["lr_warmup_steps"],
        "lr_schedule": canonical["lr_schedule"],
        "policy_loss_weight": canonical["policy_loss_weight"],
        "value_loss_weight": canonical["value_loss_weight"],
        "value_lr_mult": canonical["value_lr_mult"],
        "public_card_lr_mult": canonical["public_card_lr_mult"],
        "per_game_policy_surprise_weighting": canonical[
            "per_game_policy_surprise_weighting"
        ],
        "forced_row_value_action_type_weights": canonical[
            "forced_row_value_action_type_weights"
        ],
        "policy_aux_active_batch_size": ARMS[arm][
            "policy_aux_active_batch_size"
        ],
        "max_steps": MAX_STEPS,
    }
    drift = {
        key: {"expected": value, "actual": effective.get(key)}
        for key, value in invariants.items()
        if effective.get(key) != value
    }
    if drift:
        raise CampaignError(
            f"arm {arm} changed more than auxiliary active-policy exposure: "
            + json.dumps(drift, sort_keys=True)
        )
    if effective.get("policy_kl_target") is not None:
        raise CampaignError(
            f"arm {arm} unexpectedly enabled adaptive parent-policy KL"
        )
    return plan


def _arm_dose_telemetry(
    report: Mapping[str, Any], *, expected_aux_rows: int
) -> dict[str, Any]:
    """Project already-collected learner evidence into one comparable dose row."""

    metrics = report.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise CampaignError("active-policy arm report has no epoch metrics")
    denominators: dict[str, float] = {}
    for metric in metrics:
        if not isinstance(metric, dict):
            raise CampaignError("active-policy arm epoch metric is malformed")
        rows = metric.get("loss_denominators")
        if not isinstance(rows, dict):
            raise CampaignError("active-policy arm lacks objective denominators")
        for name, value in rows.items():
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise CampaignError("active-policy objective exposure is invalid")
            denominators[str(name)] = denominators.get(str(name), 0.0) + numeric

    optimizer = metrics[-1].get("optimizer_observability")
    modules = report.get("module_optimizer_observability")
    gradients = report.get("objective_gradient_interference")
    if (
        not isinstance(optimizer, dict)
        or int(optimizer.get("observed_steps", -1)) != MAX_STEPS
        or not isinstance(modules, dict)
        or int(modules.get("observed_steps", -1))
        != MAX_STEPS // TRAIN_DIAGNOSTIC_CADENCE
        or int(modules.get("cadence_batches", -1)) != TRAIN_DIAGNOSTIC_CADENCE
        or not isinstance(gradients, dict)
        or int(gradients.get("cadence_batches", -1))
        != OBJECTIVE_GRADIENT_CADENCE
        or int(gradients.get("observed_steps", -1))
        != MAX_STEPS // OBJECTIVE_GRADIENT_CADENCE
    ):
        raise CampaignError("active-policy optimizer telemetry cadence drifted")
    clipped = int(optimizer.get("clipped_steps", -1))
    clipped_fraction = float(optimizer.get("clipped_fraction", math.nan))
    if (
        clipped < 0
        or clipped > MAX_STEPS
        or not math.isfinite(clipped_fraction)
        or not 0.0 <= clipped_fraction <= 1.0
        or abs(clipped_fraction - clipped / MAX_STEPS) > 1.0e-12
    ):
        raise CampaignError("active-policy clipping telemetry is invalid")

    module_rows = modules.get("modules")
    if not isinstance(module_rows, dict) or not module_rows:
        raise CampaignError("active-policy module update telemetry is empty")
    normalized_modules: dict[str, Any] = {}
    for name, row in sorted(module_rows.items()):
        if not isinstance(row, dict):
            raise CampaignError("active-policy module update row is malformed")
        numeric = {
            key: float(row[key])
            for key in (
                "mean_pre_clip_grad_norm",
                "max_pre_clip_grad_norm",
                "mean_parameter_delta_norm",
                "mean_parameter_update_rms",
                "mean_relative_parameter_delta",
            )
        }
        parameter_count = int(row.get("parameter_count", 0))
        if (
            parameter_count <= 0
            or not all(math.isfinite(value) and value >= 0.0 for value in numeric.values())
        ):
            raise CampaignError("active-policy module update metric is invalid")
        normalized_modules[str(name)] = {**numeric, "parameter_count": parameter_count}

    gradient_rows = gradients.get("observations")
    if not isinstance(gradient_rows, list) or len(gradient_rows) != 2:
        raise CampaignError("active-policy objective gradient observations are incomplete")
    normalized_gradients = []
    required_gradient_fields = (
        "policy_trunk_grad_norm",
        "policy_base_trunk_grad_norm",
        "policy_aux_trunk_grad_norm",
        "value_trunk_grad_norm",
        "policy_aux_to_base_grad_norm_ratio",
    )
    for row in gradient_rows:
        if not isinstance(row, dict) or row.get("available") is not True:
            raise CampaignError("active-policy objective gradient probe unavailable")
        values = {key: float(row[key]) for key in required_gradient_fields}
        if not all(math.isfinite(value) and value >= 0.0 for value in values.values()):
            raise CampaignError("active-policy objective gradient metric is invalid")
        normalized_gradients.append(
            {
                **values,
                "scope": str(row.get("scope")),
                "trunk_gradient_cosine": row.get("trunk_gradient_cosine"),
                "policy_base_aux_gradient_cosine": row.get(
                    "policy_base_aux_gradient_cosine"
                ),
            }
        )

    active_rows = {
        "policy_base": int(report.get("policy_base_active_rows", -1)),
        "policy_aux": int(report.get("policy_aux_active_rows", -1)),
        "policy_total": int(report.get("policy_total_active_rows", -1)),
        "value": int(report.get("value_active_rows", -1)),
    }
    if (
        active_rows["policy_aux"] != expected_aux_rows
        or active_rows["policy_base"] < 0
        or active_rows["value"] < 0
        or active_rows["policy_total"]
        != active_rows["policy_base"] + active_rows["policy_aux"]
    ):
        raise CampaignError("active-policy objective row dose is invalid")
    effective_policy_weights = {
        "base": float(report.get("policy_base_effective_weight_sum", math.nan)),
        "aux": float(report.get("policy_aux_effective_weight_sum", math.nan)),
        "total": float(report.get("policy_total_effective_weight_sum", math.nan)),
    }
    if (
        not all(
            math.isfinite(value) and value >= 0.0
            for value in effective_policy_weights.values()
        )
        or abs(
            effective_policy_weights["total"]
            - effective_policy_weights["base"]
            - effective_policy_weights["aux"]
        )
        > 1.0e-6 * max(1.0, effective_policy_weights["total"])
        or abs(
            effective_policy_weights["total"]
            - denominators.get("policy_loss", math.nan)
        )
        > 1.0e-6 * max(1.0, effective_policy_weights["total"])
    ):
        raise CampaignError("active-policy effective policy mass is invalid")
    payload = {
        "schema_version": "a1-active-policy-dose-telemetry-v1",
        "active_rows": active_rows,
        "policy_effective_weight_sums": effective_policy_weights,
        "objective_effective_weight_sums": denominators,
        "optimizer": {
            "observed_steps": MAX_STEPS,
            "clipped_steps": clipped,
            "clipped_fraction": clipped_fraction,
            "mean_pre_clip_total_grad_norm": float(
                optimizer["mean_pre_clip_total_grad_norm"]
            ),
            "max_pre_clip_total_grad_norm": float(
                optimizer["max_pre_clip_total_grad_norm"]
            ),
        },
        "module_optimizer_observability": {
            "observed_steps": int(modules["observed_steps"]),
            "norm_scope": str(modules.get("norm_scope")),
            "modules": normalized_modules,
        },
        "shared_trunk_objective_gradients": {
            "observed_steps": len(normalized_gradients),
            "observations": normalized_gradients,
        },
    }
    payload["dose_telemetry_sha256"] = _value_sha256(payload)
    return payload


def _verify_completed_arm(campaign: Mapping[str, Any], arm: str) -> dict[str, Any]:
    arm_root = Path(campaign["output_root"]) / "arms" / arm
    receipt_path, receipt = _load_json(
        arm_root / "one-dose.receipt.json", where=f"arm {arm} receipt"
    )
    report_path, report = _load_json(
        arm_root / "train.report.json", where=f"arm {arm} training report"
    )
    checkpoint = _regular_file(arm_root / "candidate.pt", where=f"arm {arm} checkpoint")
    outputs = receipt.get("outputs")
    expected_aux = int(ARMS[arm]["policy_aux_active_batch_size"])
    expected_aux_rows = expected_aux * WORLD_SIZE * MAX_STEPS
    requested = report.get("checkpoint_steps_requested")
    intermediate = report.get("intermediate_checkpoints")
    learner_parent = receipt.get("learner_lineage_parent")
    if (
        receipt.get("status") != "complete"
        or receipt.get("returncode") != 0
        or not isinstance(outputs, dict)
        or outputs.get("checkpoint_sha256") != _file_sha256(checkpoint)
        or outputs.get("report_sha256") != _file_sha256(report_path)
        or report.get("steps_completed") != MAX_STEPS
        or report.get("optimizer_restored") is not False
        or report.get("policy_aux_active_batch_size") != expected_aux
        or report.get("policy_aux_active_rows") != expected_aux_rows
        or requested != list(INTERMEDIATE_STEPS)
        or not isinstance(intermediate, list)
        or [record.get("optimizer_step") for record in intermediate]
        != list(INTERMEDIATE_STEPS)
        or not isinstance(learner_parent, dict)
        or learner_parent.get("role") != "diagnostic_independent_parent"
        or learner_parent.get("checkpoint", {}).get("sha256")
        != EXPECTED_F7_PARENT_SHA256
    ):
        raise CampaignError(f"arm {arm} did not complete its exact independent dose")
    for step, record in zip(INTERMEDIATE_STEPS, intermediate, strict=True):
        path = _regular_file(Path(str(record["checkpoint"])), where=f"arm {arm} step {step}")
        if (
            record.get("checkpoint_sha256") != _file_sha256(path)
            or record.get("same_training_trajectory") is not True
        ):
            raise CampaignError(f"arm {arm} step {step} checkpoint drifted")
    dose_telemetry = _arm_dose_telemetry(
        report, expected_aux_rows=expected_aux_rows
    )
    return {
        "arm": arm,
        "receipt": str(receipt_path),
        "receipt_file_sha256": _file_sha256(receipt_path),
        "report": str(report_path),
        "report_file_sha256": _file_sha256(report_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "policy_aux_active_rows": expected_aux_rows,
        "dose_telemetry": dose_telemetry,
    }


def _run_arm(campaign: Mapping[str, Any], arm: str, *, go: bool) -> dict[str, Any]:
    _verify_campaign_inputs(campaign)
    plan = _dry_run_arm(campaign, arm)
    if not go:
        return {"mode": "dry-run", "arm": arm, "one_dose_plan": plan}
    invocation = _one_dose_invocation(campaign, arm)
    result = subprocess.run([*invocation, "--go"], check=False)
    if result.returncode != 0:
        raise CampaignError(f"active-policy arm {arm} exited {result.returncode}")
    return {"mode": "go", **_verify_completed_arm(campaign, arm)}


def _step_checkpoint(arm_root: Path, step: int) -> Path:
    if step == MAX_STEPS:
        return arm_root / "candidate.pt"
    return arm_root / f"candidate_step{step:04d}.pt"


def _trunk_relative_l2(drift: Mapping[str, Any]) -> float:
    groups = drift.get("groups")
    if not isinstance(groups, dict):
        raise CampaignError("layer-drift report has no parameter groups")
    selected = [
        value
        for name, value in groups.items()
        if name in {"input_encoders", "shared", "topology_adapter"}
        or name.startswith("transformer_block_")
    ]
    baseline_energy = sum(float(value["baseline_l2"]) ** 2 for value in selected)
    delta_energy = sum(float(value["delta_energy"]) for value in selected)
    if not selected or baseline_energy <= 0.0 or delta_energy < 0.0:
        raise CampaignError("layer-drift report cannot define shared-trunk drift")
    return math.sqrt(delta_energy / baseline_energy)


def _fingerprint_arm(
    campaign_path: Path,
    campaign: Mapping[str, Any],
    arm: str,
    *,
    go: bool,
    device: str,
) -> dict[str, Any]:
    completed = _verify_completed_arm(campaign, arm)
    arm_root = Path(campaign["output_root"]) / "arms" / arm
    report_path = Path(completed["report"])
    _report_path, report = _load_json(report_path, where=f"arm {arm} report")
    validation_value = report.get("validation_game_seed_manifest")
    if not validation_value:
        validation = Path(campaign["inputs"]["validation_manifest"])
    else:
        validation = Path(str(validation_value)).expanduser()
        if not validation.is_absolute():
            validation = report_path.parent / validation
    validation = _regular_file(validation, where=f"arm {arm} emitted validation manifest")
    parent = _regular_file(
        Path(
            str(
                _load_json(
                    Path(campaign["inputs"]["independent_parent_authority"]),
                    where="independent parent authority",
                )[1]["function_preserving_upgrade"]["upgraded_initializer"]["path"]
            )
        ),
        where="upgraded f7 initializer",
    )
    output_root = arm_root / "fingerprints"
    commands: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for step in CHECKPOINT_STEPS:
        checkpoint = _regular_file(
            _step_checkpoint(arm_root, step), where=f"arm {arm} step {step} checkpoint"
        )
        functional_output = output_root / f"step{step:04d}.functional.json"
        drift_output = output_root / f"step{step:04d}.drift.json"
        functional_command = [
            str(campaign["inputs"]["python"]),
            str(REPO_ROOT / "tools" / "posthoc_teacher_gap_probe.py"),
            "--report",
            str(report_path),
            "--checkpoint",
            str(checkpoint),
            "--parent-checkpoint",
            str(parent),
            "--data",
            str(campaign["inputs"]["data"]),
            "--validation-manifest",
            str(validation),
            "--device",
            device,
            "--output",
            str(functional_output),
        ]
        drift_command = [
            str(campaign["inputs"]["python"]),
            str(REPO_ROOT / "tools" / "audit_checkpoint_layer_drift.py"),
            "--baseline",
            str(parent),
            "--candidate",
            str(checkpoint),
            "--output",
            str(drift_output),
        ]
        commands.append(
            {"step": step, "functional": functional_command, "drift": drift_command}
        )
        if not go:
            continue
        output_root.mkdir(parents=True, exist_ok=True)
        for command in (functional_command, drift_command):
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                raise CampaignError(
                    f"arm {arm} step {step} fingerprint command exited {result.returncode}"
                )
        functional_path, functional = _load_json(
            functional_output, where=f"arm {arm} step {step} functional fingerprint"
        )
        drift_path, drift = _load_json(
            drift_output, where=f"arm {arm} step {step} layer drift"
        )
        fingerprint = functional.get("functional_dose_fingerprint")
        parent_kl = (
            fingerprint.get("kl_parent_candidate_mean")
            if isinstance(fingerprint, dict)
            else None
        )
        closure = functional.get("teacher_gap", {}).get(
            "active_policy_teacher_gap_closure"
        )
        trunk = _trunk_relative_l2(drift)
        if (
            functional.get("schema_version") != "posthoc-checkpoint-teacher-gap/v1"
            or not isinstance(fingerprint, dict)
            or fingerprint.get("surface")
            != "validation_policy_active_multi_action_rows"
            or functional.get("inputs", {}).get("checkpoint", {}).get("sha256")
            != _file_sha256(checkpoint)
            or functional.get("inputs", {}).get("parent_checkpoint", {}).get("sha256")
            != _file_sha256(parent)
            or drift.get("schema_version")
            != "entity-graph-checkpoint-layer-drift-v1"
            or drift.get("baseline", {}).get("sha256") != _file_sha256(parent)
            or drift.get("candidate", {}).get("sha256") != _file_sha256(checkpoint)
            or not all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in (parent_kl, closure, trunk)
            )
            or float(parent_kl) < 0.0
            or trunk < 0.0
        ):
            raise CampaignError(f"arm {arm} step {step} fingerprint semantics drifted")
        records.append(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _file_sha256(checkpoint),
                "functional": {
                    "path": str(functional_path),
                    "file_sha256": _file_sha256(functional_path),
                    "parent_kl": float(parent_kl),
                    "teacher_gap_closure": float(closure),
                    "eligible_rows": int(fingerprint["eligible_rows"]),
                },
                "layer_drift": {
                    "path": str(drift_path),
                    "file_sha256": _file_sha256(drift_path),
                    "trunk_relative_l2": trunk,
                    "global_relative_l2": float(drift["global"]["relative_l2"]),
                },
            }
        )
    if not go:
        return {"mode": "dry-run", "arm": arm, "commands": commands}
    payload: dict[str, Any] = {
        "schema_version": FINGERPRINT_SCHEMA,
        "campaign": {
            "path": str(campaign_path),
            "file_sha256": _file_sha256(campaign_path),
            "campaign_sha256": campaign["campaign_sha256"],
        },
        "arm": arm,
        "active_policy_branch_multiplier": ARMS[arm][
            "active_policy_branch_multiplier"
        ],
        "policy_aux_active_batch_size": ARMS[arm][
            "policy_aux_active_batch_size"
        ],
        "dose_telemetry": completed["dose_telemetry"],
        "parent_checkpoint_sha256": _file_sha256(parent),
        "checkpoints": records,
    }
    payload["fingerprint_sha256"] = _value_sha256(payload)
    output = output_root / "fingerprint.json"
    _write_immutable(output, payload)
    return {
        "mode": "go",
        "arm": arm,
        "fingerprint": str(output),
        "fingerprint_file_sha256": _file_sha256(output),
        "fingerprint_sha256": payload["fingerprint_sha256"],
    }


def _select(
    campaign_path: Path,
    campaign: Mapping[str, Any],
    bindings: Mapping[str, Path],
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    eligible: list[str] = []
    cap_kl = float(campaign["selection_contract"]["max_parent_kl"])
    cap_trunk = float(campaign["selection_contract"]["max_trunk_relative_l2"])
    for arm in ARMS:
        path, fingerprint = _load_signed(
            bindings[arm],
            where=f"arm {arm} fingerprint",
            schema=FINGERPRINT_SCHEMA,
            digest_field="fingerprint_sha256",
        )
        campaign_ref = fingerprint.get("campaign")
        checkpoints = fingerprint.get("checkpoints")
        if (
            fingerprint.get("arm") != arm
            or fingerprint.get("active_policy_branch_multiplier")
            != ARMS[arm]["active_policy_branch_multiplier"]
            or fingerprint.get("policy_aux_active_batch_size")
            != ARMS[arm]["policy_aux_active_batch_size"]
            or fingerprint.get("parent_checkpoint_sha256")
            != campaign["lineage_contract"]["upgraded_initializer_sha256"]
            or not isinstance(fingerprint.get("dose_telemetry"), dict)
            or fingerprint["dose_telemetry"].get("schema_version")
            != "a1-active-policy-dose-telemetry-v1"
            or fingerprint["dose_telemetry"].get("dose_telemetry_sha256")
            != _value_sha256(
                {
                    key: value
                    for key, value in fingerprint["dose_telemetry"].items()
                    if key != "dose_telemetry_sha256"
                }
            )
            or not isinstance(campaign_ref, dict)
            or campaign_ref.get("campaign_sha256") != campaign["campaign_sha256"]
            or campaign_ref.get("file_sha256") != _file_sha256(campaign_path)
            or not isinstance(checkpoints, list)
            or [item.get("step") for item in checkpoints] != list(CHECKPOINT_STEPS)
        ):
            raise CampaignError(f"arm {arm} fingerprint is not bound to this campaign")
        try:
            parent_kls = [
                float(item["functional"]["parent_kl"]) for item in checkpoints
            ]
            trunk_drifts = [
                float(item["layer_drift"]["trunk_relative_l2"])
                for item in checkpoints
            ]
            closures = [
                float(item["functional"]["teacher_gap_closure"])
                for item in checkpoints
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise CampaignError(f"arm {arm} fingerprint metrics are malformed") from error
        if (
            not all(math.isfinite(value) and value >= 0.0 for value in parent_kls)
            or not all(math.isfinite(value) and value >= 0.0 for value in trunk_drifts)
            or not all(math.isfinite(value) for value in closures)
        ):
            raise CampaignError(f"arm {arm} fingerprint metrics are non-finite")
        closure = closures[-1]
        within = max(parent_kls) <= cap_kl and max(trunk_drifts) <= cap_trunk
        positive = math.isfinite(closure) and closure > 0.0
        if within and positive:
            eligible.append(arm)
        records[arm] = {
            "path": str(path),
            "file_sha256": _file_sha256(path),
            "fingerprint_sha256": fingerprint["fingerprint_sha256"],
            "max_parent_kl": max(parent_kls),
            "max_trunk_relative_l2": max(trunk_drifts),
            "terminal_parent_kl": parent_kls[-1],
            "terminal_trunk_relative_l2": trunk_drifts[-1],
            "terminal_teacher_gap_closure": closure,
            "within_drift_budgets": within,
            "positive_terminal_teacher_gap_closure": positive,
            "dose_telemetry_sha256": fingerprint["dose_telemetry"][
                "dose_telemetry_sha256"
            ],
        }
    if not eligible:
        raise CampaignError("no active-policy exposure arm remained inside both drift budgets")
    winner = sorted(
        eligible,
        key=lambda arm: (
            -records[arm]["terminal_teacher_gap_closure"],
            records[arm]["terminal_parent_kl"],
            records[arm]["terminal_trunk_relative_l2"],
            ARMS[arm]["policy_aux_active_batch_size"],
            arm,
        ),
    )[0]
    payload: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA,
        "campaign": {
            "path": str(campaign_path),
            "file_sha256": _file_sha256(campaign_path),
            "campaign_sha256": campaign["campaign_sha256"],
        },
        "selection_policy": copy.deepcopy(campaign["selection_contract"]),
        "arm_fingerprints": records,
        "eligible_arms": eligible,
        "winner": winner,
        "winner_recipe": copy.deepcopy(campaign["arms"][winner]),
        "winner_is_diagnostic_not_promoted": True,
        "playing_strength_evaluation_still_required": True,
        "winner_meets_reference_teacher_gap_closure": (
            records[winner]["terminal_teacher_gap_closure"]
            >= float(
                campaign["selection_contract"]["reference_update_frontier"][
                    "active_policy_teacher_gap_closure"
                ]
            )
        ),
        "candidate_chaining": False,
    }
    payload["selection_sha256"] = _value_sha256(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    admit = sub.add_parser("admit-corpus", help="seal completed coherent n128 data")
    admit.add_argument("--contract", required=True, type=Path)
    admit.add_argument("--completion-receipt", required=True, type=Path)
    admit.add_argument("--inventory", required=True, type=Path)
    admit.add_argument("--corpus-meta", required=True, type=Path)
    admit.add_argument("--validation-manifest", required=True, type=Path)
    admit.add_argument("--write", required=True, type=Path)

    plan = sub.add_parser("plan", help="seal the four independent 8xB200 arms")
    plan.add_argument("--coherent-corpus-receipt", required=True, type=Path)
    plan.add_argument("--lock", required=True, type=Path)
    plan.add_argument("--architecture-upgrade-receipt", required=True, type=Path)
    plan.add_argument("--ddp-canary-receipt", required=True, type=Path)
    plan.add_argument("--python", required=True, type=Path)
    plan.add_argument("--reviewed-code-tree-sha256", required=True)
    plan.add_argument("--reviewed-lock-file-sha256", default="")
    plan.add_argument("--max-parent-kl", required=True, type=float)
    plan.add_argument("--max-trunk-relative-l2", required=True, type=float)
    plan.add_argument("--output-root", required=True, type=Path)
    plan.add_argument("--write", required=True, type=Path)

    run = sub.add_parser("run-arm", help="dry-run or execute one independent arm")
    run.add_argument("--campaign", required=True, type=Path)
    run.add_argument("--arm", required=True, choices=tuple(ARMS))
    run.add_argument("--go", action="store_true")

    sequence = sub.add_parser("run-sequence", help="run all independent arms serially")
    sequence.add_argument("--campaign", required=True, type=Path)
    sequence.add_argument("--arms", default=",".join(ARMS))
    sequence.add_argument("--go", action="store_true")

    fingerprint = sub.add_parser(
        "fingerprint-arm", help="measure parent KL and trunk drift at every checkpoint"
    )
    fingerprint.add_argument("--campaign", required=True, type=Path)
    fingerprint.add_argument("--arm", required=True, choices=tuple(ARMS))
    fingerprint.add_argument("--device", default="cuda:0")
    fingerprint.add_argument("--go", action="store_true")

    select = sub.add_parser("select", help="select the best in-budget exposure")
    select.add_argument("--campaign", required=True, type=Path)
    select.add_argument("--fingerprint", action="append", default=[])
    select.add_argument("--write", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "admit-corpus":
            payload = _admit_corpus(args)
            _write_immutable(args.write, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "plan":
            campaign = _plan(args)
            _write_immutable(args.write, campaign)
            print(json.dumps(campaign, indent=2, sort_keys=True))
            return 0
        campaign_path, campaign = _load_campaign(args.campaign)
        if args.command == "run-arm":
            result = _run_arm(campaign, args.arm, go=bool(args.go))
        elif args.command == "run-sequence":
            arms = [value.strip() for value in args.arms.split(",") if value.strip()]
            if not arms or len(arms) != len(set(arms)) or any(arm not in ARMS for arm in arms):
                raise CampaignError("--arms must contain unique active-policy arm names")
            result = {
                "mode": "go" if args.go else "dry-run",
                "arms": [
                    _run_arm(campaign, arm, go=bool(args.go)) for arm in arms
                ],
            }
        elif args.command == "fingerprint-arm":
            result = _fingerprint_arm(
                campaign_path,
                campaign,
                args.arm,
                go=bool(args.go),
                device=args.device,
            )
        elif args.command == "select":
            bindings = _parse_bindings(
                args.fingerprint, allowed=set(ARMS), label="arm fingerprint"
            )
            result = _select(campaign_path, campaign, bindings)
            _write_immutable(args.write, result)
        else:  # pragma: no cover
            raise CampaignError(f"unknown command {args.command!r}")
    except (
        CampaignError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"active-policy campaign refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
