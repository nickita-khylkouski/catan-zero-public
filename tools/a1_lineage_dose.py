"""Typed initialization and cumulative-dose contracts for A1 learner runs."""

from __future__ import annotations

import math
from typing import Any, Mapping

LINEAGE_DOSE_SCHEMA = "a1-lineage-dose-v1"
CURRICULUM_DECLARATION_SCHEMA = "a1-curriculum-declaration-v1"
INITIALIZER_TRANSITION_SCHEMA = "a1-initializer-transition-v1"
INITIALIZER_TRANSITION_KINDS = (
    "public_award_zero_initialization",
    "function_preserving_pointer_upgrade",
    "head_only_auxiliary_warmup",
)


class LineageDoseError(ValueError):
    """Invalid initialization or cumulative learner-dose provenance."""


def exact_objective_exposure_from_training_report(
    report_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the trainer's exact current-dose objective counters into lineage."""

    integer_fields = (
        "training_row_draws",
        "base_training_row_draws",
        "policy_aux_training_row_draws",
        "policy_base_active_training_row_draws",
        "policy_active_training_row_draws",
        "value_active_training_row_draws",
        "policy_base_active_rows",
        "policy_aux_active_rows",
        "policy_total_active_rows",
        "value_active_rows",
        "policy_kl_anchor_eligible_rows",
    )
    values: dict[str, int] = {}
    for field in integer_fields:
        value = report_payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LineageDoseError(
                f"training report has invalid exact-dose counter {field!r}"
            )
        values[field] = value
    total_draws = report_payload.get("total_training_row_draws")
    if (
        isinstance(total_draws, bool)
        or not isinstance(total_draws, int)
        or total_draws < 0
        or values["policy_aux_training_row_draws"]
        != values["policy_aux_active_rows"]
        or values["training_row_draws"]
        != values["base_training_row_draws"]
        or values["policy_base_active_training_row_draws"]
        != values["policy_base_active_rows"]
        or values["policy_active_training_row_draws"]
        != values["policy_total_active_rows"]
        or values["value_active_training_row_draws"]
        != values["value_active_rows"]
        or values["policy_total_active_rows"]
        != values["policy_base_active_rows"] + values["policy_aux_active_rows"]
        or total_draws
        != values["base_training_row_draws"]
        + values["policy_aux_training_row_draws"]
    ):
        raise LineageDoseError("training report objective-dose arithmetic drift")
    if report_payload.get("training_row_draws_semantics") != (
        "base_sampler_draw_events; may repeat rows; excludes_policy_aux"
    ):
        raise LineageDoseError("training report row-draw semantics drift")
    if values["value_active_rows"] > values["base_training_row_draws"]:
        raise LineageDoseError(
            "exact value-active dose exceeds the base draw dose"
        )
    if (
        values["policy_kl_anchor_eligible_rows"]
        > values["base_training_row_draws"]
    ):
        raise LineageDoseError(
            "exact anchor-eligible dose exceeds the base draw dose"
        )
    metrics = report_payload.get("metrics")
    effective_recipe = report_payload.get("a1_effective_learner_training_recipe")
    if not isinstance(effective_recipe, Mapping):
        effective_recipe = {}
    completed_q_base_enabled = float(
        report_payload.get(
            "completed_q_loss_weight",
            effective_recipe.get("completed_q_loss_weight", 0.0),
        )
    ) != 0.0
    completed_q_aux_enabled = float(
        report_payload.get(
            "policy_aux_completed_q_loss_weight",
            effective_recipe.get("policy_aux_completed_q_loss_weight", 0.0),
        )
    ) != 0.0
    if (
        completed_q_base_enabled or completed_q_aux_enabled
    ) and not (isinstance(metrics, list) and metrics):
        raise LineageDoseError(
            "enabled completed-Q objective lacks exact per-epoch exposure metrics"
        )
    completed_q_base_exposure = 0.0
    completed_q_aux_exposure = 0.0
    completed_q_base_active_rows = 0
    completed_q_aux_active_rows = 0
    for metric in metrics if isinstance(metrics, list) else ():
        denominators = (
            metric.get("loss_denominators")
            if isinstance(metric, Mapping)
            else None
        )
        if not isinstance(denominators, Mapping):
            raise LineageDoseError(
                "training report lacks completed-Q exposure denominators"
            )
        for key, destination in (
            ("completed_q_loss", "base"),
            ("policy_aux_completed_q_loss", "aux"),
        ):
            enabled = (
                completed_q_base_enabled
                if destination == "base"
                else completed_q_aux_enabled
            )
            if enabled and key not in denominators:
                raise LineageDoseError(
                    f"enabled completed-Q objective lacks {key} exposure"
                )
            raw = denominators.get(key, 0.0)
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or float(raw) < 0.0
            ):
                raise LineageDoseError(
                    f"training report has invalid {key} exposure"
                )
            if destination == "base":
                completed_q_base_exposure += float(raw)
            else:
                completed_q_aux_exposure += float(raw)
        for key, destination in (
            ("completed_q_active_rows", "base"),
            ("policy_aux_completed_q_active_rows", "aux"),
        ):
            enabled = (
                completed_q_base_enabled
                if destination == "base"
                else completed_q_aux_enabled
            )
            if enabled and key not in metric:
                raise LineageDoseError(
                    f"enabled completed-Q objective lacks {key} count"
                )
            raw = metric.get(key, 0)
            if (
                isinstance(raw, bool)
                or not isinstance(raw, int)
                or raw < 0
            ):
                raise LineageDoseError(
                    f"training report has invalid {key} count"
                )
            if destination == "base":
                completed_q_base_active_rows += raw
            else:
                completed_q_aux_active_rows += raw
    result = {
        "measurement_status": "bound_exactly",
        "measurement_scope": "current_dose",
        "base_sampled_rows": values["base_training_row_draws"],
        "policy_base_active_sampled_rows": values["policy_base_active_rows"],
        "policy_aux_active_sampled_rows": values["policy_aux_active_rows"],
        "policy_active_sampled_rows": values["policy_total_active_rows"],
        "value_active_sampled_rows": values["value_active_rows"],
        "anchor_eligible_sampled_rows": values[
            "policy_kl_anchor_eligible_rows"
        ],
    }
    result.update(
        {
            "completed_q_base_effective_weight_exposure": (
                completed_q_base_exposure
            ),
            "completed_q_aux_effective_weight_exposure": (
                completed_q_aux_exposure
            ),
            "completed_q_base_active_rows": completed_q_base_active_rows,
            "completed_q_aux_active_rows": completed_q_aux_active_rows,
            "completed_q_exposure_measurement_status": "bound_exactly",
        }
    )
    return result


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LineageDoseError(f"{field} must be a positive integer")
    return value


def _typed_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
    ):
        raise LineageDoseError(f"{field} is not a typed SHA-256")
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError as error:
        raise LineageDoseError(f"{field} is not a typed SHA-256") from error
    return value


def _validate_initializer_transition_chain(
    value: Any,
    *,
    declared_producer_sha256: str,
    init_checkpoint_sha256: str,
) -> tuple[list[dict[str, Any]], int, int]:
    """Validate one exact, contiguous initializer-preparation lineage.

    These transitions are not candidate chaining.  The first two are
    zero-optimizer, function-preserving schema/architecture transforms; the
    optional third is the measured head-only commissioning dose whose inherited
    policy/value/trunk bytes remain unchanged.  Keeping that warmup dose in the
    cumulative lineage prevents it from becoming invisible training.
    """

    if not isinstance(value, list) or not value:
        raise LineageDoseError("initializer transition chain must be non-empty")
    expected_kinds = list(INITIALIZER_TRANSITION_KINDS[: len(value)])
    observed_kinds = [row.get("kind") if isinstance(row, Mapping) else None for row in value]
    if len(value) > len(INITIALIZER_TRANSITION_KINDS) or observed_kinds != expected_kinds:
        raise LineageDoseError(
            "initializer transition order must be public-award, pointer, then warmup"
        )
    required = {
        "schema_version",
        "kind",
        "role",
        "source_checkpoint_sha256",
        "output_checkpoint_sha256",
        "sampled_rows",
        "optimizer_steps",
        "optimizer_state_terminal",
        "receipt_path",
        "receipt_file_sha256",
        "receipt_state_sha256",
        "inherited_parameters_bit_identical",
        "main_output_max_abs_diff_decimal",
    }
    normalized: list[dict[str, Any]] = []
    previous = declared_producer_sha256
    prior_rows = 0
    prior_steps = 0
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise LineageDoseError(
                f"initializer transition {index} field set drift"
            )
        row = dict(raw)
        kind = row["kind"]
        source = _typed_sha256(
            row["source_checkpoint_sha256"],
            f"initializer transition {index} source",
        )
        output = _typed_sha256(
            row["output_checkpoint_sha256"],
            f"initializer transition {index} output",
        )
        _typed_sha256(
            row["receipt_file_sha256"],
            f"initializer transition {index} receipt file",
        )
        _typed_sha256(
            row["receipt_state_sha256"],
            f"initializer transition {index} receipt state",
        )
        rows = row["sampled_rows"]
        steps = row["optimizer_steps"]
        if (
            row["schema_version"] != INITIALIZER_TRANSITION_SCHEMA
            or source != previous
            or output == source
            or not isinstance(row["receipt_path"], str)
            or not row["receipt_path"]
            or row["inherited_parameters_bit_identical"] is not True
            or row["main_output_max_abs_diff_decimal"] != "0"
            or isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows < 0
            or isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps < 0
        ):
            raise LineageDoseError(
                f"initializer transition {index} semantic drift"
            )
        if kind == "public_award_zero_initialization":
            expected_role = "feature_schema_zero_initialization"
            expected_terminal = "not_constructed"
            expected_dose = (0, 0)
        elif kind == "function_preserving_pointer_upgrade":
            expected_role = "architecture_zero_diff_upgrade"
            expected_terminal = "not_constructed"
            expected_dose = (0, 0)
        else:
            expected_role = "head_only_auxiliary_commissioning"
            expected_terminal = "discarded_before_joint_training"
            expected_dose = (524_288, 128)
        if (
            row["role"] != expected_role
            or row["optimizer_state_terminal"] != expected_terminal
            or (rows, steps) != expected_dose
        ):
            raise LineageDoseError(
                f"initializer transition {index} role/dose drift"
            )
        previous = output
        prior_rows += rows
        prior_steps += steps
        normalized.append(row)
    if previous != init_checkpoint_sha256:
        raise LineageDoseError(
            "initializer transition chain does not terminate at the actual initializer"
        )
    return normalized, prior_rows, prior_steps


def validate_lineage_dose(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema_version") != LINEAGE_DOSE_SCHEMA:
        raise LineageDoseError("lineage dose schema drift")
    mode = value.get("mode")
    if mode not in {
        "direct_from_declared_producer",
        "direct_with_information_contract_migration",
        "direct_with_typed_initializer_chain",
        "typed_curriculum",
    }:
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
        completed_q_fields = {
            "completed_q_base_effective_weight_exposure",
            "completed_q_aux_effective_weight_exposure",
            "completed_q_base_active_rows",
            "completed_q_aux_active_rows",
            "completed_q_exposure_measurement_status",
        }
        if (
            set(objective) != exact_fields | completed_q_fields
            or objective.get("measurement_scope") != "current_dose"
        ):
            raise LineageDoseError("lineage exact objective exposure schema drift")
        if completed_q_fields.issubset(objective):
            if objective["completed_q_exposure_measurement_status"] != "bound_exactly":
                raise LineageDoseError("lineage completed-Q exposure status drift")
            if any(
                isinstance(objective[field], bool)
                or not isinstance(objective[field], (int, float))
                or not math.isfinite(float(objective[field]))
                or float(objective[field]) < 0.0
                for field in (
                    "completed_q_base_effective_weight_exposure",
                    "completed_q_aux_effective_weight_exposure",
                )
            ):
                raise LineageDoseError(
                    "lineage completed-Q exposure must be finite and non-negative"
                )
            if any(
                isinstance(objective[field], bool)
                or not isinstance(objective[field], int)
                or objective[field] < 0
                for field in (
                    "completed_q_base_active_rows",
                    "completed_q_aux_active_rows",
                )
            ):
                raise LineageDoseError(
                    "lineage completed-Q active-row counts must be non-negative integers"
                )
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
    if mode in {
        "direct_from_declared_producer",
        "direct_with_information_contract_migration",
        "direct_with_typed_initializer_chain",
    } and (prior_rows or prior_steps):
        raise LineageDoseError("direct lineage cannot carry a prior dose")
    if mode == "typed_curriculum" and (prior_rows <= 0 or prior_steps <= 0):
        raise LineageDoseError("curriculum lineage must carry a positive prior dose")
    for field in ("declared_producer_sha256", "init_checkpoint_sha256"):
        _typed_sha256(value.get(field), field)
    upgrade = value.get("function_preserving_upgrade")
    migration = value.get("information_contract_migration")
    transition_chain = value.get("initializer_transition_chain")
    preparation = value.get("initializer_preparation_exposure")
    if mode == "typed_curriculum":
        if (
            upgrade is not None
            or migration is not None
            or transition_chain is not None
            or preparation is not None
        ):
            raise LineageDoseError(
                "curriculum lineage cannot also claim an initializer transform"
            )
    elif mode == "direct_with_typed_initializer_chain":
        if upgrade is not None or migration is not None:
            raise LineageDoseError(
                "typed initializer chain cannot also claim a legacy single upgrade"
            )
        chain, chain_rows, chain_steps = _validate_initializer_transition_chain(
            transition_chain,
            declared_producer_sha256=value["declared_producer_sha256"],
            init_checkpoint_sha256=value["init_checkpoint_sha256"],
        )
        expected_preparation = {
            "schema_version": "a1-initializer-preparation-exposure-v1",
            "measurement_scope": "initializer_preparation_only",
            "sampled_rows": chain_rows,
            "optimizer_steps": chain_steps,
            "active_parameter_surface": (
                "new_auxiliary_heads_only" if chain_steps else "no_optimizer_surface"
            ),
            "policy_active_sampled_rows": 0,
            "value_active_sampled_rows": 0,
            "shared_trunk_active_sampled_rows": 0,
            "auxiliary_head_active_sampled_rows": chain_rows,
        }
        if chain != transition_chain or preparation != expected_preparation:
            raise LineageDoseError("initializer preparation exposure drift")
    elif mode == "direct_with_information_contract_migration":
        if upgrade is not None or transition_chain is not None or preparation is not None:
            raise LineageDoseError(
                "information migration cannot also claim an initializer transform"
            )
        migration_kind = (
            migration.get("migration") if isinstance(migration, Mapping) else None
        )
        expected_forward_identity = {
            (
                "entity_graph.current_v2_to_v6_information_contract"
                "+topology+split1.v1"
            ): False,
            (
                "entity_graph.current_v2_to_v6_information_contract+topology+"
                "split1+public_resource_v8.v1"
            ): False,
            "entity_graph.v5_to_v7_input_compatibility.v1": True,
            "entity_graph.v5_to_v8_public_resource_compatibility.v1": True,
        }.get(migration_kind)
        if (
            value["init_checkpoint_sha256"] == value["declared_producer_sha256"]
            or not isinstance(migration, Mapping)
            or set(migration)
            != {
                "schema_version",
                "migration",
                "receipt",
                "receipt_sha256",
                "source_checkpoint_sha256",
                "migrated_initializer_sha256",
                "forward_identical",
                "promotion_eligible",
            }
            or migration.get("schema_version")
            != "a1-lineage-information-contract-migration-v1"
            or migration.get("source_checkpoint_sha256")
            != value["declared_producer_sha256"]
            or migration.get("migrated_initializer_sha256")
            != value["init_checkpoint_sha256"]
            or expected_forward_identity is None
            or migration.get("forward_identical") is not expected_forward_identity
            or migration.get("promotion_eligible") is not False
            or not isinstance(migration.get("receipt"), str)
            or not migration["receipt"]
        ):
            raise LineageDoseError("information-contract migration lineage drift")
        _typed_sha256(migration.get("receipt_sha256"), "migration receipt_sha256")
    elif value["init_checkpoint_sha256"] == value["declared_producer_sha256"]:
        if (
            upgrade is not None
            or migration is not None
            or transition_chain is not None
            or preparation is not None
        ):
            raise LineageDoseError(
                "initializer transform is forbidden for an exact-parent init"
            )
    else:
        if migration is not None or transition_chain is not None or preparation is not None:
            raise LineageDoseError(
                "legacy single-upgrade lineage cannot carry a transition chain"
            )
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
            _typed_sha256(upgrade.get(field), f"upgrade {field}")
        if not isinstance(upgrade.get("receipt"), str) or not upgrade["receipt"]:
            raise LineageDoseError("upgrade receipt path is missing")
    return dict(value)


def direct_lineage_dose(
    *, declared_producer_sha256: str, init_checkpoint_sha256: str,
    current_sampled_rows: int, current_optimizer_steps: int,
    function_preserving_upgrade: Mapping[str, Any] | None = None,
    information_contract_migration: Mapping[str, Any] | None = None,
    initializer_transition_chain: list[Mapping[str, Any]] | None = None,
    objective_exposure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        init_checkpoint_sha256 != declared_producer_sha256
        and function_preserving_upgrade is None
        and information_contract_migration is None
        and initializer_transition_chain is None
    ):
        raise LineageDoseError(
            "untyped checkpoint chaining: init SHA differs from declared producer SHA"
        )
    if sum(
        value is not None
        for value in (
            function_preserving_upgrade,
            information_contract_migration,
            initializer_transition_chain,
        )
    ) > 1:
        raise LineageDoseError(
            "initializer can use only one typed transition mechanism"
        )
    normalized_chain = None
    preparation_exposure = None
    if initializer_transition_chain is not None:
        normalized_chain, preparation_rows, preparation_steps = _validate_initializer_transition_chain(
            list(initializer_transition_chain),
            declared_producer_sha256=declared_producer_sha256,
            init_checkpoint_sha256=init_checkpoint_sha256,
        )
        preparation_exposure = {
            "schema_version": "a1-initializer-preparation-exposure-v1",
            "measurement_scope": "initializer_preparation_only",
            "sampled_rows": preparation_rows,
            "optimizer_steps": preparation_steps,
            "active_parameter_surface": (
                "new_auxiliary_heads_only"
                if preparation_steps
                else "no_optimizer_surface"
            ),
            "policy_active_sampled_rows": 0,
            "value_active_sampled_rows": 0,
            "shared_trunk_active_sampled_rows": 0,
            "auxiliary_head_active_sampled_rows": preparation_rows,
        }
    return validate_lineage_dose({
        "schema_version": LINEAGE_DOSE_SCHEMA,
        "mode": (
            "direct_with_typed_initializer_chain"
            if normalized_chain is not None
            else "direct_with_information_contract_migration"
            if information_contract_migration is not None
            else "direct_from_declared_producer"
        ),
        "declared_producer_sha256": declared_producer_sha256,
        "init_checkpoint_sha256": init_checkpoint_sha256,
        "function_preserving_upgrade": (
            None if function_preserving_upgrade is None else dict(function_preserving_upgrade)
        ),
        **(
            {
                "information_contract_migration": dict(
                    information_contract_migration
                )
            }
            if information_contract_migration is not None
            else {}
        ),
        "initializer_transition_chain": normalized_chain,
        "initializer_preparation_exposure": preparation_exposure,
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
        "function_preserving_upgrade": None,
        "initializer_transition_chain": None,
        "initializer_preparation_exposure": None,
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
