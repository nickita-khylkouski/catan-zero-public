"""Typed initialization and cumulative-dose contracts for A1 learner runs."""

from __future__ import annotations

from typing import Any, Mapping

LINEAGE_DOSE_SCHEMA = "a1-lineage-dose-v1"
CURRICULUM_DECLARATION_SCHEMA = "a1-curriculum-declaration-v1"


class LineageDoseError(ValueError):
    """Invalid initialization or cumulative learner-dose provenance."""


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LineageDoseError(f"{field} must be a positive integer")
    return value


def validate_lineage_dose(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema_version") != LINEAGE_DOSE_SCHEMA:
        raise LineageDoseError("lineage dose schema drift")
    mode = value.get("mode")
    if mode not in {"direct_from_declared_producer", "typed_curriculum"}:
        raise LineageDoseError("lineage dose mode drift")
    if value.get("optimizer_state_continuity") != "fresh_optimizer_per_dose":
        raise LineageDoseError("lineage optimizer-state continuity drift")
    objective = value.get("objective_exposure")
    if not isinstance(objective, Mapping):
        raise LineageDoseError("lineage objective-specific exposure schema drift")
    measurement_status = objective.get("measurement_status")
    if measurement_status == "not_yet_bound_exactly":
        if (
            set(objective)
            != {
                "measurement_status",
                "policy_active_sampled_rows",
                "value_active_sampled_rows",
                "anchor_eligible_sampled_rows",
            }
            or any(
                objective[field] is not None
                for field in (
                    "policy_active_sampled_rows",
                    "value_active_sampled_rows",
                    "anchor_eligible_sampled_rows",
                )
            )
        ):
            raise LineageDoseError("lineage objective-specific exposure schema drift")
    elif measurement_status == "bound_exactly":
        exact_fields = {
            "measurement_status",
            "measurement_scope",
            "base_sampled_rows",
            "policy_base_active_sampled_rows",
            "policy_aux_active_sampled_rows",
            "policy_active_sampled_rows",
            "value_active_sampled_rows",
            "anchor_eligible_sampled_rows",
        }
        if set(objective) != exact_fields or objective.get("measurement_scope") != "current_dose":
            raise LineageDoseError("lineage exact objective exposure schema drift")
        numeric_fields = exact_fields - {"measurement_status", "measurement_scope"}
        if any(
            isinstance(objective[field], bool)
            or not isinstance(objective[field], int)
            or objective[field] < 0
            for field in numeric_fields
        ):
            raise LineageDoseError("lineage exact objective exposure must be non-negative integers")
        if (
            objective["base_sampled_rows"] <= 0
            or objective["value_active_sampled_rows"] > objective["base_sampled_rows"]
            or objective["anchor_eligible_sampled_rows"]
            > objective["base_sampled_rows"]
            or objective["policy_base_active_sampled_rows"] > objective["base_sampled_rows"]
            or objective["policy_active_sampled_rows"]
            != objective["policy_base_active_sampled_rows"]
            + objective["policy_aux_active_sampled_rows"]
        ):
            raise LineageDoseError("lineage exact objective exposure arithmetic drift")
    else:
        raise LineageDoseError("lineage objective-specific exposure schema drift")
    current_rows = _positive_int(value.get("current_sampled_rows"), "current_sampled_rows")
    if (
        measurement_status == "bound_exactly"
        and objective["base_sampled_rows"] != current_rows
    ):
        raise LineageDoseError(
            "lineage exact base exposure does not match current sampled rows"
        )
    current_steps = _positive_int(value.get("current_optimizer_steps"), "current_optimizer_steps")
    cumulative_rows = _positive_int(value.get("cumulative_sampled_rows"), "cumulative_sampled_rows")
    cumulative_steps = _positive_int(value.get("cumulative_optimizer_steps"), "cumulative_optimizer_steps")
    prior_rows = value.get("prior_sampled_rows")
    prior_steps = value.get("prior_optimizer_steps")
    if (
        isinstance(prior_rows, bool)
        or not isinstance(prior_rows, int)
        or prior_rows < 0
        or isinstance(prior_steps, bool)
        or not isinstance(prior_steps, int)
        or prior_steps < 0
        or cumulative_rows != prior_rows + current_rows
        or cumulative_steps != prior_steps + current_steps
    ):
        raise LineageDoseError("lineage cumulative dose arithmetic drift")
    if mode == "direct_from_declared_producer" and (prior_rows or prior_steps):
        raise LineageDoseError("direct lineage cannot carry a prior dose")
    if mode == "typed_curriculum" and (prior_rows <= 0 or prior_steps <= 0):
        raise LineageDoseError("curriculum lineage must carry a positive prior dose")
    for field in ("declared_producer_sha256", "init_checkpoint_sha256"):
        raw = value.get(field)
        if not isinstance(raw, str) or not raw.startswith("sha256:") or len(raw) != 71:
            raise LineageDoseError(f"{field} is not a typed SHA-256")
    upgrade = value.get("function_preserving_upgrade")
    if mode == "typed_curriculum":
        if upgrade is not None:
            raise LineageDoseError(
                "curriculum lineage cannot also claim an architecture upgrade"
            )
    elif value["init_checkpoint_sha256"] == value["declared_producer_sha256"]:
        if upgrade is not None:
            raise LineageDoseError(
                "function-preserving upgrade is forbidden for an exact-parent init"
            )
    else:
        if (
            not isinstance(upgrade, Mapping)
            or set(upgrade) != {
                "schema_version",
                "module",
                "receipt",
                "receipt_sha256",
                "source_checkpoint_sha256",
                "upgraded_initializer_sha256",
            }
            or upgrade.get("schema_version")
            != "a1-lineage-function-preserving-upgrade-v1"
            or upgrade.get("source_checkpoint_sha256")
            != value["declared_producer_sha256"]
            or upgrade.get("upgraded_initializer_sha256")
            != value["init_checkpoint_sha256"]
            or not isinstance(upgrade.get("module"), str)
            or not upgrade["module"]
        ):
            raise LineageDoseError(
                "non-parent init requires an exact typed function-preserving upgrade"
            )
        for field in ("receipt_sha256",):
            raw = upgrade.get(field)
            if not isinstance(raw, str) or not raw.startswith("sha256:") or len(raw) != 71:
                raise LineageDoseError(f"upgrade {field} is not a typed SHA-256")
        if not isinstance(upgrade.get("receipt"), str) or not upgrade["receipt"]:
            raise LineageDoseError("upgrade receipt path is missing")
    return dict(value)


def direct_lineage_dose(
    *, declared_producer_sha256: str, init_checkpoint_sha256: str,
    current_sampled_rows: int, current_optimizer_steps: int,
    function_preserving_upgrade: Mapping[str, Any] | None = None,
    objective_exposure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if init_checkpoint_sha256 != declared_producer_sha256 and function_preserving_upgrade is None:
        raise LineageDoseError(
            "untyped checkpoint chaining: init SHA differs from declared producer SHA"
        )
    return validate_lineage_dose({
        "schema_version": LINEAGE_DOSE_SCHEMA,
        "mode": "direct_from_declared_producer",
        "declared_producer_sha256": declared_producer_sha256,
        "init_checkpoint_sha256": init_checkpoint_sha256,
        "function_preserving_upgrade": (
            None if function_preserving_upgrade is None else dict(function_preserving_upgrade)
        ),
        "parent_receipt_sha256": None,
        "optimizer_state_continuity": "fresh_optimizer_per_dose",
        "objective_exposure": (
            {
                "measurement_status": "not_yet_bound_exactly",
                "policy_active_sampled_rows": None,
                "value_active_sampled_rows": None,
                "anchor_eligible_sampled_rows": None,
            }
            if objective_exposure is None
            else dict(objective_exposure)
        ),
        "prior_sampled_rows": 0,
        "prior_optimizer_steps": 0,
        "current_sampled_rows": current_sampled_rows,
        "current_optimizer_steps": current_optimizer_steps,
        "cumulative_sampled_rows": current_sampled_rows,
        "cumulative_optimizer_steps": current_optimizer_steps,
    })


def curriculum_lineage_dose(
    *, declared_producer_sha256: str, init_checkpoint_sha256: str,
    parent_receipt_sha256: str, parent_lineage_dose: Mapping[str, Any],
    current_sampled_rows: int, current_optimizer_steps: int,
) -> dict[str, Any]:
    parent = validate_lineage_dose(parent_lineage_dose)
    if parent["declared_producer_sha256"] != declared_producer_sha256:
        raise LineageDoseError("curriculum producer lineage drift")
    return validate_lineage_dose({
        "schema_version": LINEAGE_DOSE_SCHEMA,
        "mode": "typed_curriculum",
        "declared_producer_sha256": declared_producer_sha256,
        "init_checkpoint_sha256": init_checkpoint_sha256,
        "parent_receipt_sha256": parent_receipt_sha256,
        "optimizer_state_continuity": "fresh_optimizer_per_dose",
        "objective_exposure": {
            "measurement_status": "not_yet_bound_exactly",
            "policy_active_sampled_rows": None,
            "value_active_sampled_rows": None,
            "anchor_eligible_sampled_rows": None,
        },
        "prior_sampled_rows": parent["cumulative_sampled_rows"],
        "prior_optimizer_steps": parent["cumulative_optimizer_steps"],
        "current_sampled_rows": current_sampled_rows,
        "current_optimizer_steps": current_optimizer_steps,
        "cumulative_sampled_rows": parent["cumulative_sampled_rows"] + current_sampled_rows,
        "cumulative_optimizer_steps": parent["cumulative_optimizer_steps"] + current_optimizer_steps,
    })
