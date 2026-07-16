#!/usr/bin/env python3
"""One machine-readable authority for the current A1 improvement operator.

Issued PIMC locks remain replayable by their historical verifiers.  This
module governs only the new coherent-public operator and supplies projections
for generation sealing, one-dose learning, fleet evaluation, and promotion.
"""

from __future__ import annotations

import copy
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    REPO_ROOT
    / "configs/operations/a1-next-wave-coherent-public-v3/science.contract.json"
)
TEMPLATE_PATH = REPO_ROOT / "configs/experiments/a1_pre_wave_contract.template.json"
GENERATOR_CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/next_wave/coherent_public_n128_adaptive256_forced_value_v3.schema14.json"
)
GENERATOR_GUARD_PATH = (
    REPO_ROOT
    / "configs/guards/a1_generation_coherent_public_n128_adaptive256_forced_value_v3.json"
)
SCHEMA_VERSION = "a1-current-science-contract-v1"
TEACHER_REPORT_SCHEMA = "teacher-operator-causal-report-v1"
ADOPTION_RECEIPT_SCHEMA = "a1-teacher-operator-adoption-v1"
ADAPTIVE_FIELDS = (
    "n_full_wide",
    "n_full_wide_threshold",
    "wide_roots_always_full",
)
POLICY_TARGET_BLEND_FALLBACK_V2 = "policy_target_fallback_v2"
PRODUCTION_LEARNER_SIGNAL_CONTRACT = {
    # The science recipe is sealed in its legacy single-process representation;
    # a1_one_dose_train overlays 8x512 while preserving the 4096-row global dose.
    "world_size": 1,
    "batch_size": 4096,
    "global_batch_size": 4096,
    "grad_accum_steps": 1,
    "max_steps": 128,
    "resume_optimizer": False,
    "lr": 6e-5,
    "lr_warmup_steps": 16,
    "lr_schedule": "flat",
    "value_lr_mult": 1.0,
    "value_trunk_grad_scale": 1.0,
}
DIAGNOSTIC_POLICY_AUX_FIELDS = frozenset(
    {"policy_aux_active_batch_size", "policy_aux_loss_weight"}
)


class ScienceContractError(ValueError):
    """The current production science contract is malformed or drifted."""


def _load() -> dict[str, Any]:
    try:
        value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScienceContractError(
            f"cannot load current science contract {CONTRACT_PATH}: {error}"
        ) from error
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ScienceContractError("current science contract schema drift")
    required = {
        "schema_version",
        "contract_id",
        "operator_selection",
        "target_information_regime",
        "operator",
        "generation",
        "learner",
        "evaluation",
        "promotion",
    }
    if set(value) != required:
        raise ScienceContractError(
            "current science contract top-level fields drifted: "
            f"missing={sorted(required - set(value))}, "
            f"extra={sorted(set(value) - required)}"
        )
    operator = value.get("operator")
    if not isinstance(operator, dict) or set(operator) != {"search", "evaluator"}:
        raise ScienceContractError("current operator must bind search and evaluator")
    for key in ("search", "evaluator"):
        if not isinstance(operator[key], dict) or not operator[key]:
            raise ScienceContractError(f"current operator.{key} is empty")
    for key in ("generation", "learner", "evaluation", "promotion"):
        if not isinstance(value[key], dict) or not value[key]:
            raise ScienceContractError(f"current {key} contract is empty")
    selection = value["operator_selection"]
    if not isinstance(selection, dict) or selection.get("status") not in {
        "provisional_pending_teacher_campaign",
        "adopted_teacher_campaign",
    }:
        raise ScienceContractError("current operator selection status is invalid")
    if (
        selection.get("report_schema") != TEACHER_REPORT_SCHEMA
        or selection.get("mutable_fields") != list(ADAPTIVE_FIELDS)
    ):
        raise ScienceContractError("current operator selection authority drifted")
    if selection["status"] == "adopted_teacher_campaign":
        selected_fields = _selected_adaptive_fields(
            str(selection.get("selected_operator"))
        )
        actual_fields = {
            key: operator["search"].get(key) for key in ADAPTIVE_FIELDS
        }
        if actual_fields != selected_fields or not isinstance(
            selection.get("report"), dict
        ):
            raise ScienceContractError("adopted teacher operator evidence drifted")
    learner_value = value["learner"]
    if set(learner_value) != {
        "architecture_upgrade_flags",
        "architecture_upgrade_module",
        "topology",
        "training_recipe",
    } or not isinstance(learner_value["training_recipe"], dict):
        raise ScienceContractError("current learner contract shape drifted")
    recipe = learner_value["training_recipe"]
    if (
        recipe.get("policy_target_blend_semantics")
        != POLICY_TARGET_BLEND_FALLBACK_V2
        or recipe.get("soft_target_weight") != 1.0
    ):
        raise ScienceContractError(
            "current coherent learner must bind pure authenticated policy CE "
            "with hard-action fallback only"
        )
    learner_signal_drift = {
        key: {
            "expected": expected,
            "actual": recipe.get(key),
        }
        for key, expected in PRODUCTION_LEARNER_SIGNAL_CONTRACT.items()
        if recipe.get(key) != expected
    }
    if learner_signal_drift:
        raise ScienceContractError(
            "current coherent learner inherited a diagnostic/approximate training "
            f"setting: {learner_signal_drift}"
        )
    leaked_aux_fields = sorted(DIAGNOSTIC_POLICY_AUX_FIELDS & set(recipe))
    if leaked_aux_fields:
        raise ScienceContractError(
            "current coherent base learner must not bind diagnostic active-policy "
            f"AUX fields: {leaked_aux_fields}"
        )
    evaluator_value = operator["evaluator"]
    if (
        evaluator_value.get("value_readout") == "scalar"
        and evaluator_value.get("value_squash") == "tanh"
        and (
            recipe.get("scalar_value_loss_readout") != "deployed_tanh"
            or recipe.get("scalar_value_loss_scale")
            != evaluator_value.get("value_scale")
        )
    ):
        raise ScienceContractError(
            "current scalar learner must optimize the exact deployed tanh "
            "search readout and scale"
        )
    return value


def load() -> dict[str, Any]:
    return copy.deepcopy(_load())


def search() -> dict[str, Any]:
    return copy.deepcopy(_load()["operator"]["search"])


def evaluator() -> dict[str, Any]:
    return copy.deepcopy(_load()["operator"]["evaluator"])


def generation() -> dict[str, Any]:
    return copy.deepcopy(_load()["generation"])


def learner() -> dict[str, Any]:
    return copy.deepcopy(_load()["learner"])


def learner_training_recipe() -> dict[str, Any]:
    return copy.deepcopy(_load()["learner"]["training_recipe"])


def target_information_regime() -> str:
    return str(_load()["target_information_regime"])


def operator_selection_status() -> str:
    return str(_load()["operator_selection"]["status"])


def is_coherent_search(value: Mapping[str, Any]) -> bool:
    return value.get("coherent_public_belief_search") is True


def require_current_operator(
    *,
    search_value: Mapping[str, Any],
    evaluator_value: Mapping[str, Any] | None = None,
    generation_value: Mapping[str, Any] | None = None,
    learner_recipe_value: Mapping[str, Any] | None = None,
    target_regime: str | None = None,
    require_adopted: bool = False,
) -> None:
    """Fail closed when a current coherent-public authority drifts.

    Callers invoke this only for coherent-public locks/drafts.  Historical
    information-set/PIMC objects therefore retain their original semantics.
    """

    expected_search = search()
    if require_adopted and operator_selection_status() != "adopted_teacher_campaign":
        raise ScienceContractError(
            "coherent-public teacher operator is provisional; aggregate and adopt "
            "the causal teacher campaign before sealing a production wave"
        )
    if dict(search_value) != expected_search:
        differing = sorted(
            key
            for key in set(search_value) | set(expected_search)
            if search_value.get(key) != expected_search.get(key)
        )
        raise ScienceContractError(
            f"coherent-public search differs from current science contract: {differing}"
        )
    if evaluator_value is not None:
        expected_evaluator = evaluator()
        actual_evaluator = {
            key: evaluator_value.get(key) for key in expected_evaluator
        }
        if actual_evaluator != expected_evaluator:
            differing = sorted(
                key
                for key in expected_evaluator
                if actual_evaluator.get(key) != expected_evaluator.get(key)
            )
            raise ScienceContractError(
                "coherent-public evaluator differs from current science contract: "
                f"{differing}"
            )
    if generation_value is not None:
        expected_generation = generation()
        actual_generation = {
            key: generation_value.get(key) for key in expected_generation
        }
        if actual_generation != expected_generation:
            differing = sorted(
                key
                for key in expected_generation
                if actual_generation.get(key) != expected_generation.get(key)
            )
            raise ScienceContractError(
                "coherent-public generation differs from current science contract: "
                f"{differing}"
            )
    if learner_recipe_value is not None:
        expected_recipe = learner_training_recipe()
        actual_recipe = dict(learner_recipe_value)
        if actual_recipe != expected_recipe:
            differing = sorted(
                key
                for key in set(actual_recipe) | set(expected_recipe)
                if actual_recipe.get(key) != expected_recipe.get(key)
            )
            raise ScienceContractError(
                "coherent-public learner recipe differs from current science "
                f"contract: {differing}"
            )
    if target_regime is not None and target_regime != target_information_regime():
        raise ScienceContractError(
            "coherent-public target-information regime differs from current "
            f"science contract: {target_regime!r}"
        )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScienceContractError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ScienceContractError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _selected_adaptive_fields(selected: str) -> dict[str, Any]:
    if selected == "base_n128_d6":
        return {
            "n_full_wide": None,
            "n_full_wide_threshold": None,
            "wide_roots_always_full": False,
        }
    if selected in {"adaptive_n256_w20_d6", "adaptive_n256_w40_d6"}:
        return {
            "n_full_wide": 256,
            "n_full_wide_threshold": 20 if "w20" in selected else 40,
            "wide_roots_always_full": True,
        }
    raise ScienceContractError(f"teacher campaign selected unknown operator {selected!r}")


def adopt_teacher_campaign(
    report_path: str | Path,
    *,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Adopt one completed causal campaign into every production authority.

    The current w20 value is intentionally provisional.  This transaction
    changes only the three preregistered adaptive-dose fields in the contract,
    draft template, typed generator config, and guard.  The canonical contract
    is replaced last so an interrupted transaction can be safely rerun.
    """

    report_path = Path(report_path).expanduser().resolve(strict=True)
    receipt_path = Path(receipt_path).expanduser().resolve(strict=False)
    report = _read_object(report_path)
    if report.get("schema_version") != TEACHER_REPORT_SCHEMA:
        raise ScienceContractError("teacher campaign report schema drift")
    report_content = dict(report)
    reported_content_sha = report_content.pop("report_content_sha256", None)
    if reported_content_sha != _content_sha256(report_content):
        raise ScienceContractError("teacher campaign report content digest mismatch")

    contract = _load()
    selection = contract["operator_selection"]
    if selection.get("status") != "provisional_pending_teacher_campaign":
        raise ScienceContractError("teacher operator has already been adopted")
    before_contract_sha = _file_sha256(CONTRACT_PATH)
    report_authority = report.get("science_contract")
    if not isinstance(report_authority, dict) or (
        report_authority.get("sha256") != before_contract_sha
        or report_authority.get("contract_id") != contract["contract_id"]
        or report_authority.get("experimental_dose_fields")
        != list(ADAPTIVE_FIELDS)
    ):
        raise ScienceContractError(
            "teacher campaign was not run against these provisional contract bytes"
        )
    selected = report.get("selection", {}).get("selected_operator")
    adaptive = _selected_adaptive_fields(str(selected))

    template = _read_object(TEMPLATE_PATH)
    generator = _read_object(GENERATOR_CONFIG_PATH)
    guard = _read_object(GENERATOR_GUARD_PATH)
    search_targets = (
        contract["operator"]["search"],
        template["science"]["search"],
        generator["fields"],
    )
    provisional = {key: contract["operator"]["search"].get(key) for key in ADAPTIVE_FIELDS}
    for target in search_targets:
        observed = {key: target.get(key) for key in ADAPTIVE_FIELDS}
        if observed != provisional:
            raise ScienceContractError(
                f"adaptive authority drift before adoption: {observed} != {provisional}"
            )
        target.update(adaptive)

    try:
        lint_args = next(
            item["args"] for item in guard["guards"] if item.get("name") == "cli_flag_lint"
        )
        critical = lint_args["critical_flags"]
        expected = lint_args["expected_values"]
    except (KeyError, StopIteration, TypeError) as error:
        raise ScienceContractError("coherent generator guard shape drifted") from error
    old_guard = {
        "n_full_wide": expected.get("--n-full-wide"),
        "n_full_wide_threshold": expected.get("--n-full-wide-threshold"),
        "wide_roots_always_full": expected.get("--wide-roots-always-full"),
    }
    if old_guard != provisional:
        raise ScienceContractError(
            f"adaptive guard drift before adoption: {old_guard} != {provisional}"
        )
    for flag in ("--n-full-wide", "--n-full-wide-threshold"):
        while flag in critical:
            critical.remove(flag)
        expected.pop(flag, None)
    if adaptive["n_full_wide"] is not None:
        insertion = critical.index("--wide-roots-always-full")
        critical[insertion:insertion] = ["--n-full-wide", "--n-full-wide-threshold"]
        expected["--n-full-wide"] = adaptive["n_full_wide"]
        expected["--n-full-wide-threshold"] = adaptive["n_full_wide_threshold"]
    expected["--wide-roots-always-full"] = adaptive["wide_roots_always_full"]

    contract["operator_selection"] = {
        "status": "adopted_teacher_campaign",
        "report_schema": TEACHER_REPORT_SCHEMA,
        "mutable_fields": list(ADAPTIVE_FIELDS),
        "selected_operator": selected,
        "report": {
            "path": str(report_path),
            "file_sha256": _file_sha256(report_path),
            "content_sha256": reported_content_sha,
            "checkpoint_sha256": report.get("checkpoint_sha256"),
        },
    }
    receipt = {
        "schema_version": ADOPTION_RECEIPT_SCHEMA,
        "selected_operator": selected,
        "adaptive_fields": adaptive,
        "teacher_report": contract["operator_selection"]["report"],
        "contract_before_sha256": before_contract_sha,
        "artifacts": {
            "template": str(TEMPLATE_PATH),
            "generator_config": str(GENERATOR_CONFIG_PATH),
            "generator_guard": str(GENERATOR_GUARD_PATH),
            "science_contract": str(CONTRACT_PATH),
        },
    }
    _atomic_json(GENERATOR_CONFIG_PATH, generator)
    _atomic_json(GENERATOR_GUARD_PATH, guard)
    _atomic_json(TEMPLATE_PATH, template)
    _atomic_json(CONTRACT_PATH, contract)
    receipt["contract_after_sha256"] = _file_sha256(CONTRACT_PATH)
    receipt["receipt_content_sha256"] = _content_sha256(receipt)
    _atomic_json(receipt_path, receipt)
    return copy.deepcopy(receipt)


def fleet_evaluation_science_config() -> dict[str, Any]:
    """Project the current operator into the H100 evaluator's plan schema."""

    contract = _load()
    search_value = contract["operator"]["search"]
    evaluator_value = contract["operator"]["evaluator"]
    evaluation_value = contract["evaluation"]
    return {
        "internal_map_kind": evaluation_value["internal_map_kind"],
        "external_map_kind": evaluation_value["external_map_kind"],
        "n_full": search_value["n_full"],
        "c_scale": search_value["c_scale"],
        "c_visit": search_value["c_visit"],
        "sigma_eval": search_value["sigma_eval"],
        "rescale_noise_floor_c": search_value["rescale_noise_floor_c"],
        "lazy_interior_chance": search_value["lazy_interior_chance"],
        "correct_rust_chance_spectra": search_value["correct_rust_chance_spectra"],
        "public_observation": evaluator_value["public_observation"],
        "information_set_search": search_value["information_set_search"],
        "belief_chance_spectra": search_value["belief_chance_spectra"],
        "coherent_public_belief_search": search_value[
            "coherent_public_belief_search"
        ],
        "determinization_particles": search_value["determinization_particles"],
        "determinization_min_simulations": search_value[
            "determinization_min_simulations"
        ],
        "forced_root_target_mode": search_value["forced_root_target_mode"],
        "n_full_wide": search_value["n_full_wide"],
        "n_full_wide_threshold": search_value["n_full_wide_threshold"],
        "wide_roots_always_full": search_value["wide_roots_always_full"],
        "symmetry_averaged_eval": search_value["symmetry_averaged_eval"],
        "symmetry_averaged_eval_threshold": search_value[
            "symmetry_averaged_eval_threshold"
        ],
        "evaluator_rust_featurize": evaluator_value["rust_featurize"],
        "native_mcts_hot_loop": contract["generation"]["native_mcts_hot_loop"],
        "value_readout": evaluator_value["value_readout"],
        "value_squash": evaluator_value["value_squash"],
        "max_depth": search_value["max_depth"],
        "max_decisions": contract["generation"]["max_decisions"],
        "max_root_candidates": evaluation_value["max_root_candidates"],
        "max_root_candidates_wide": evaluation_value[
            "max_root_candidates_wide"
        ],
        "wide_candidates_threshold": search_value["wide_candidates_threshold"],
        "gate_config": evaluation_value["gate_config"],
        "external_vps_to_win": evaluation_value["external_vps_to_win"],
        "external_max_player_trade_offers_per_turn": evaluation_value[
            "external_max_player_trade_offers_per_turn"
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    adopt = subparsers.add_parser(
        "adopt-teacher",
        help="adopt a completed causal teacher report into production authorities",
    )
    adopt.add_argument("--report", required=True)
    adopt.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    if args.command == "adopt-teacher":
        receipt = adopt_teacher_campaign(args.report, receipt_path=args.receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
