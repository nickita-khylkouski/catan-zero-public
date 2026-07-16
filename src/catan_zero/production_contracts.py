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
GENERATION_GUARD = (
    "configs/guards/"
    "a1_generation_coherent_public_n128_adaptive256_forced_value_v3.json"
)
GENERATION_GUARD_SHA256 = (
    "9d86aba856305cb98fef3d8a318d1e5fc82abfe011d7f93bb4bc1cd7be3fc4c1"
)


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
            entry
            for entry in production_recipes(pipeline)
            if entry["name"] == selected
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
        guard_path = (repo / GENERATION_GUARD).resolve()
        guard_payload = _read_json_object(guard_path, label=f"{pipeline} guard")
        guard_sha256 = canonical_json_sha256(guard_payload)
        if guard_sha256 != GENERATION_GUARD_SHA256:
            raise ProductionContractError(
                f"{pipeline} guard identity drift: "
                f"expected={GENERATION_GUARD_SHA256} actual={guard_sha256} "
                f"path={guard_path}"
            )
    return {
        "schema_version": PRODUCTION_CONTRACT_SCHEMA,
        "pipeline": pipeline,
        "recipe": entry["name"],
        "launcher": str(launcher_path),
        "config": str(config_path),
        "config_sha256": entry["canonical_sha256"],
        "guard": None if guard_path is None else str(guard_path),
        "guard_sha256": guard_sha256,
    }


def pipeline_readiness(
    repo: Path, pipeline: str, recipe: str | None = None
) -> dict[str, Any]:
    """Return the current checked-in authorization state for one pipeline."""

    identity = validate_pipeline_contract(repo, pipeline, recipe)
    if pipeline == "train":
        if identity["recipe"] == "a1-parent-update-35m-b200":
            return {
                "pipeline": pipeline,
                "recipe": identity["recipe"],
                "status": "ready",
                "authorized": True,
                "reason": "commissioned_parent_update_recipe",
                "authority": identity["config"],
                "authority_sha256": identity["config_sha256"],
            }
        science_path = (
            repo
            / "configs/operations/a1-next-wave-coherent-public-v3/"
            "science.contract.json"
        ).resolve()
        science = _read_json_object(science_path, label="current science contract")
        execution = science.get("learner", {}).get("execution_topology", {})
        ready = bool(
            isinstance(execution, dict)
            and execution.get("go_authorized") is True
            and execution.get("optimization_schedule_status")
            == "commissioned_scratch_update_horizon_v1"
        )
        return {
            "pipeline": pipeline,
            "recipe": identity["recipe"],
            "status": "ready" if ready else "blocked",
            "authorized": ready,
            "reason": (
                "commissioned_scratch_schedule"
                if ready
                else "scratch_optimizer_schedule_unresolved"
            ),
            "authority": str(science_path),
            "authority_sha256": canonical_json_sha256(science),
        }
    if pipeline == "ppo":
        return {
            "pipeline": pipeline,
            "recipe": None,
            "status": "blocked",
            "authorized": False,
            "reason": "negative_exact_initializer_canary_and_no_canonical_ppo_recipe",
            "authority": str(
                (repo / "docs/reviews/CATAN_ZERO_DIFFERENTIAL_REVIEW_2026-07-16.md").resolve()
            ),
            "authority_sha256": hashlib.sha256(
                (
                    repo
                    / "docs/reviews/CATAN_ZERO_DIFFERENTIAL_REVIEW_2026-07-16.md"
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
        name: pipeline_readiness(repo, name)
        for name in ("generate", "evaluate", "ppo")
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
        "authorized": any(
            value["authorized"] for value in train_recipes.values()
        ),
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
