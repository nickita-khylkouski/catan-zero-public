"""Single source of truth for supported production pipeline identities.

The repository retains large flag-based executors for sealed replay and R&D.
New operator workflows must resolve through these exact config identities.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from catan_zero.rl.production_recipe_catalog import (
    CATALOG_RELATIVE_PATH,
    ProductionRecipeError,
    production_recipes,
)


PRODUCTION_CONTRACT_SCHEMA = "catan-zero-production-contracts-v1"
NATIVE_REQUIRED_CAPABILITIES = frozenset(
    {
        "belief_target_evidence",
        "coherent_public_belief_search",
        "forced_root_trajectory_only",
        "initial_road_d1_scope",
        "policy_temperature_semantics",
        "public_award_feature_parity",
        "sigma_reference_visits",
    }
)


DEFAULT_RECIPES = {
    "generate": "coherent-public-n128",
    "train": "a1-current-35m-b200",
    "evaluate": "coherent-public-n128",
}
PIPELINE_LAUNCHERS = {
    "generate": "tools/generate.py",
    "evaluate": "tools/evaluate.py",
}
TRAIN_LAUNCHERS = {
    "a1-current-35m-b200": "tools/a1_scratch_train.py",
    "a1-parent-update-35m-b200": "tools/train.py",
}
PIPELINE_ACCELERATOR_MODELS = {
    "generate": "NVIDIA H100",
    "evaluate": "NVIDIA H100",
}
TRAIN_ACCELERATOR_MODELS = {
    "a1-current-35m-b200": "NVIDIA B200",
    "a1-parent-update-35m-b200": "NVIDIA B200",
}
TRAINING_SCIENCE_ADMISSION = Path("configs/production/training_science_admission.json")
TRAINING_SCIENCE_ADMISSION_SCHEMA = "catan-zero-training-science-admission-v1"


class ProductionContractError(RuntimeError):
    """A checked-in production identity or readiness assertion drifted."""


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionContractError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProductionContractError(f"{label} must contain a JSON object: {path}")
    return value


def _recipe_entry(pipeline: str, recipe: str | None) -> dict[str, str]:
    selected = recipe or DEFAULT_RECIPES.get(pipeline)
    if selected is None:
        raise ProductionContractError(f"pipeline {pipeline!r} has no recipe")
    try:
        matches = [
            entry for entry in production_recipes(pipeline) if entry["name"] == selected
        ]
    except ProductionRecipeError as error:
        raise ProductionContractError(str(error)) from error
    if len(matches) != 1:
        raise ProductionContractError(
            f"unknown production {pipeline} recipe {selected!r}"
        )
    return matches[0]


def validate_pipeline_contract(
    repo: Path, pipeline: str, recipe: str | None = None
) -> dict[str, Any]:
    if pipeline == "ppo":
        return {
            "schema_version": PRODUCTION_CONTRACT_SCHEMA,
            "pipeline": pipeline,
            "recipe": None,
            "config": None,
            "config_sha256": None,
            "launcher": None,
            "guard": None,
            "guard_sha256": None,
            "required_accelerator_model": None,
        }
    if pipeline not in DEFAULT_RECIPES:
        raise ProductionContractError(f"unknown production pipeline {pipeline!r}")
    entry = _recipe_entry(pipeline, recipe)
    config_path = Path(entry["path"])
    try:
        config_path.relative_to(repo.resolve())
    except ValueError as error:
        raise ProductionContractError(
            f"cataloged {pipeline} config escapes repository: {config_path}"
        ) from error
    launcher = (
        TRAIN_LAUNCHERS.get(entry["name"])
        if pipeline == "train"
        else PIPELINE_LAUNCHERS[pipeline]
    )
    if launcher is None:
        raise ProductionContractError(
            f"production recipe {entry['name']!r} has no launcher"
        )
    launcher_path = (repo / launcher).resolve()
    if not launcher_path.is_file():
        raise ProductionContractError(
            f"{pipeline} launcher is missing: {launcher_path}"
        )
    guard_path: Path | None = None
    guard_sha256: str | None = None
    if pipeline == "generate":
        guard_path = Path(entry["guard"])
        guard_sha256 = entry["guard_sha256"]
    return {
        "schema_version": PRODUCTION_CONTRACT_SCHEMA,
        "pipeline": pipeline,
        "recipe": entry["name"],
        "launcher": str(launcher_path),
        "config": str(config_path),
        "config_sha256": entry["canonical_sha256"],
        "guard": None if guard_path is None else str(guard_path),
        "guard_sha256": guard_sha256,
        "required_accelerator_model": (
            TRAIN_ACCELERATOR_MODELS[entry["name"]]
            if pipeline == "train"
            else PIPELINE_ACCELERATOR_MODELS[pipeline]
        ),
    }


def pipeline_readiness(
    repo: Path, pipeline: str, recipe: str | None = None
) -> dict[str, Any]:
    """Return the current checked-in authorization state for one pipeline."""

    identity = validate_pipeline_contract(repo, pipeline, recipe)
    if pipeline == "train":
        science_path = (repo / TRAINING_SCIENCE_ADMISSION).resolve()
        science = _read_json_object(science_path, label="training science admission")
        if (
            set(science) != {"schema_version", "recipes"}
            or science.get("schema_version") != TRAINING_SCIENCE_ADMISSION_SCHEMA
        ):
            raise ProductionContractError("training science admission schema drift")
        recipes = science.get("recipes")
        if not isinstance(recipes, dict) or set(recipes) != set(
            TRAIN_ACCELERATOR_MODELS
        ):
            raise ProductionContractError("training science admission recipe drift")
        admission = recipes.get(identity["recipe"])
        expected_fields = {
            "recipe_canonical_sha256",
            "authorized",
            "reason",
            "unresolved_requirements",
            "observations",
            "commissioning_evidence",
        }
        if not isinstance(admission, dict) or set(admission) != expected_fields:
            raise ProductionContractError("training science admission fields drift")
        if admission.get("recipe_canonical_sha256") != identity["config_sha256"]:
            raise ProductionContractError(
                "training science admission does not bind the exact recipe"
            )
        authorized = admission.get("authorized")
        reason = admission.get("reason")
        unresolved = admission.get("unresolved_requirements")
        observations = admission.get("observations")
        evidence = admission.get("commissioning_evidence")
        if (
            not isinstance(authorized, bool)
            or not isinstance(reason, str)
            or not reason
            or not isinstance(unresolved, list)
            or any(not isinstance(item, str) or not item for item in unresolved)
            or not isinstance(observations, dict)
            or not isinstance(evidence, list)
        ):
            raise ProductionContractError("training science admission value drift")
        if authorized and (unresolved or not evidence):
            raise ProductionContractError(
                "authorized training requires resolved blockers and commissioning evidence"
            )
        if not authorized and not unresolved:
            raise ProductionContractError(
                "blocked training science admission must name unresolved requirements"
            )
        return {
            "pipeline": pipeline,
            "recipe": identity["recipe"],
            "status": "ready" if authorized else "blocked",
            "authorized": authorized,
            "reason": reason,
            "authority": str(science_path),
            "authority_sha256": canonical_json_sha256(science),
            "unresolved_requirements": unresolved,
            "observations": observations,
        }
    if pipeline == "ppo":
        return {
            "pipeline": pipeline,
            "recipe": None,
            "status": "blocked",
            "authorized": False,
            "reason": "negative_exact_initializer_canary_and_no_canonical_ppo_recipe",
            "authority": str(
                (
                    repo / "docs/reviews/CATAN_ZERO_DIFFERENTIAL_REVIEW_2026-07-16.md"
                ).resolve()
            ),
            "authority_sha256": hashlib.sha256(
                (
                    repo / "docs/reviews/CATAN_ZERO_DIFFERENTIAL_REVIEW_2026-07-16.md"
                ).read_bytes()
            ).hexdigest(),
        }
    return {
        "pipeline": pipeline,
        "recipe": identity["recipe"],
        "status": "ready",
        "authorized": True,
        "reason": "canonical_contract_valid",
        "authority": identity["config"],
        "authority_sha256": identity["config_sha256"],
    }


def production_status(repo: Path) -> dict[str, Any]:
    pipelines = {
        name: pipeline_readiness(repo, name) for name in ("generate", "evaluate", "ppo")
    }
    train_recipes = {
        entry["name"]: pipeline_readiness(repo, "train", entry["name"])
        for entry in production_recipes("train")
    }
    pipelines["train"] = {
        "pipeline": "train",
        "status": "ready"
        if any(value["authorized"] for value in train_recipes.values())
        else "blocked",
        "authorized": any(value["authorized"] for value in train_recipes.values()),
        "reason": "recipe_specific_authorization",
        "authority": str((repo / CATALOG_RELATIVE_PATH).resolve()),
        "authority_sha256": canonical_json_sha256(
            _read_json_object(
                (repo / CATALOG_RELATIVE_PATH).resolve(), label="production catalog"
            )
        ),
        "recipes": train_recipes,
    }
    return {
        "schema_version": PRODUCTION_CONTRACT_SCHEMA,
        "pipelines": pipelines,
        "supported_operator_interface": "catan-zero",
        "historical_executor_policy": "replay_and_research_only",
    }
