from __future__ import annotations

import copy

import pytest

from tools import a1_lineage_dose as lineage

PRODUCER = "sha256:" + "1" * 64
PARENT = "sha256:" + "2" * 64
RECEIPT = "sha256:" + "3" * 64
TRANSITIONED = "sha256:" + "4" * 64
POINTER = "sha256:" + "5" * 64
WARMED = "sha256:" + "6" * 64


def _transition(
    *,
    kind: str,
    source: str,
    output: str,
    receipt_digit: str,
) -> dict[str, object]:
    roles = {
        "public_award_zero_initialization": (
            "feature_schema_zero_initialization",
            0,
            0,
            "not_constructed",
        ),
        "function_preserving_pointer_upgrade": (
            "architecture_zero_diff_upgrade",
            0,
            0,
            "not_constructed",
        ),
        "head_only_auxiliary_warmup": (
            "head_only_auxiliary_commissioning",
            524_288,
            128,
            "discarded_before_joint_training",
        ),
    }
    role, rows, steps, optimizer_terminal = roles[kind]
    return {
        "schema_version": lineage.INITIALIZER_TRANSITION_SCHEMA,
        "kind": kind,
        "role": role,
        "source_checkpoint_sha256": source,
        "output_checkpoint_sha256": output,
        "sampled_rows": rows,
        "optimizer_steps": steps,
        "optimizer_state_terminal": optimizer_terminal,
        "receipt_path": f"/immutable/{kind}.json",
        "receipt_file_sha256": "sha256:" + receipt_digit * 64,
        "receipt_state_sha256": "sha256:" + "a" * 64,
        "inherited_parameters_bit_identical": True,
        "main_output_max_abs_diff_decimal": "0",
    }


def _transition_only_chain() -> list[dict[str, object]]:
    return [
        _transition(
            kind="public_award_zero_initialization",
            source=PRODUCER,
            output=TRANSITIONED,
            receipt_digit="7",
        )
    ]


def _full_initializer_chain() -> list[dict[str, object]]:
    return [
        *_transition_only_chain(),
        _transition(
            kind="function_preserving_pointer_upgrade",
            source=TRANSITIONED,
            output=POINTER,
            receipt_digit="8",
        ),
        _transition(
            kind="head_only_auxiliary_warmup",
            source=POINTER,
            output=WARMED,
            receipt_digit="9",
        ),
    ]


def test_direct_dose_requires_init_to_equal_declared_producer() -> None:
    dose = lineage.direct_lineage_dose(
        declared_producer_sha256=PRODUCER,
        init_checkpoint_sha256=PRODUCER,
        current_sampled_rows=100,
        current_optimizer_steps=5,
    )
    assert dose["cumulative_sampled_rows"] == 100
    assert dose["cumulative_optimizer_steps"] == 5
    assert dose["optimizer_state_continuity"] == "fresh_optimizer_per_dose"
    with pytest.raises(lineage.LineageDoseError, match="untyped checkpoint chaining"):
        lineage.direct_lineage_dose(
            declared_producer_sha256=PRODUCER,
            init_checkpoint_sha256=PARENT,
            current_sampled_rows=100,
            current_optimizer_steps=5,
        )


def test_typed_curriculum_adds_parent_and_current_dose() -> None:
    parent = lineage.direct_lineage_dose(
        declared_producer_sha256=PRODUCER,
        init_checkpoint_sha256=PRODUCER,
        current_sampled_rows=56_000,
        current_optimizer_steps=14,
    )
    child = lineage.curriculum_lineage_dose(
        declared_producer_sha256=PRODUCER,
        init_checkpoint_sha256=PARENT,
        parent_receipt_sha256=RECEIPT,
        parent_lineage_dose=parent,
        current_sampled_rows=140_000,
        current_optimizer_steps=35,
    )
    assert child["mode"] == "typed_curriculum"
    assert child["prior_sampled_rows"] == 56_000
    assert child["cumulative_sampled_rows"] == 196_000
    assert child["cumulative_optimizer_steps"] == 49


def test_validator_rejects_forged_cumulative_arithmetic() -> None:
    dose = lineage.direct_lineage_dose(
        declared_producer_sha256=PRODUCER,
        init_checkpoint_sha256=PRODUCER,
        current_sampled_rows=100,
        current_optimizer_steps=5,
    )
    dose["cumulative_sampled_rows"] = 101
    with pytest.raises(lineage.LineageDoseError, match="arithmetic drift"):
        lineage.validate_lineage_dose(dose)


def test_direct_dose_accepts_authenticated_v5_to_v7_compatibility_migration() -> None:
    initializer = "sha256:" + "4" * 64
    migration = {
        "schema_version": "a1-lineage-information-contract-migration-v1",
        "migration": "entity_graph.v5_to_v7_input_compatibility.v1",
        "receipt": "/evidence/v5-to-v7.json",
        "receipt_sha256": RECEIPT,
        "source_checkpoint_sha256": PRODUCER,
        "migrated_initializer_sha256": initializer,
        "forward_identical": True,
        "promotion_eligible": False,
    }

    dose = lineage.direct_lineage_dose(
        declared_producer_sha256=PRODUCER,
        init_checkpoint_sha256=initializer,
        information_contract_migration=migration,
        current_sampled_rows=6_144,
        current_optimizer_steps=12,
    )

    assert dose["mode"] == "direct_with_information_contract_migration"
    assert dose["information_contract_migration"] == migration


def test_v5_to_v7_compatibility_migration_rejects_false_forward_identity() -> None:
    initializer = "sha256:" + "4" * 64
    migration = {
        "schema_version": "a1-lineage-information-contract-migration-v1",
        "migration": "entity_graph.v5_to_v7_input_compatibility.v1",
        "receipt": "/evidence/v5-to-v7.json",
        "receipt_sha256": RECEIPT,
        "source_checkpoint_sha256": PRODUCER,
        "migrated_initializer_sha256": initializer,
        "forward_identical": False,
        "promotion_eligible": False,
    }

    with pytest.raises(lineage.LineageDoseError, match="lineage drift"):
        lineage.direct_lineage_dose(
            declared_producer_sha256=PRODUCER,
            init_checkpoint_sha256=initializer,
            information_contract_migration=migration,
            current_sampled_rows=6_144,
            current_optimizer_steps=12,
        )


def test_direct_dose_binds_exact_objective_exposure() -> None:
    exposure = {
        "measurement_status": "bound_exactly",
        "measurement_scope": "current_dose",
        "base_sampled_rows": 4_194_304,
        "policy_base_active_sampled_rows": 515_337,
        "policy_aux_active_sampled_rows": 1_048_576,
        "policy_active_sampled_rows": 1_563_913,
        "value_active_sampled_rows": 4_194_304,
        "anchor_eligible_sampled_rows": 0,
    }
    dose = lineage.direct_lineage_dose(
        declared_producer_sha256=PRODUCER,
        init_checkpoint_sha256=PRODUCER,
        current_sampled_rows=4_194_304,
        current_optimizer_steps=1_024,
        objective_exposure=exposure,
    )

    assert dose["objective_exposure"] == exposure


def test_training_report_reconstructs_exact_objective_exposure() -> None:
    exposure = lineage.exact_objective_exposure_from_training_report(
        {
            "training_row_draws": 6_144,
            "training_row_draws_semantics": (
                "base_sampler_draw_events; may repeat rows; excludes_policy_aux"
            ),
            "base_training_row_draws": 6_144,
            "policy_aux_training_row_draws": 128,
            "policy_base_active_training_row_draws": 1_399,
            "policy_active_training_row_draws": 1_527,
            "value_active_training_row_draws": 6_144,
            "total_training_row_draws": 6_272,
            "policy_base_active_rows": 1_399,
            "policy_aux_active_rows": 128,
            "policy_total_active_rows": 1_527,
            "value_active_rows": 6_144,
            "policy_kl_anchor_eligible_rows": 0,
        }
    )

    assert exposure == {
        "measurement_status": "bound_exactly",
        "measurement_scope": "current_dose",
        "base_sampled_rows": 6_144,
        "policy_base_active_sampled_rows": 1_399,
        "policy_aux_active_sampled_rows": 128,
        "policy_active_sampled_rows": 1_527,
        "value_active_sampled_rows": 6_144,
        "anchor_eligible_sampled_rows": 0,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("training_row_draws", 6_143),
        ("total_training_row_draws", 6_143),
        ("policy_total_active_rows", 1_398),
        ("policy_aux_training_row_draws", True),
    ),
)
def test_training_report_rejects_invalid_objective_counters(
    field: str,
    value: object,
) -> None:
    report = {
        "training_row_draws": 6_144,
        "training_row_draws_semantics": (
            "base_sampler_draw_events; may repeat rows; excludes_policy_aux"
        ),
        "base_training_row_draws": 6_144,
        "policy_aux_training_row_draws": 0,
        "policy_base_active_training_row_draws": 1_399,
        "policy_active_training_row_draws": 1_399,
        "value_active_training_row_draws": 6_144,
        "total_training_row_draws": 6_144,
        "policy_base_active_rows": 1_399,
        "policy_aux_active_rows": 0,
        "policy_total_active_rows": 1_399,
        "value_active_rows": 6_144,
        "policy_kl_anchor_eligible_rows": 0,
    }
    report[field] = value

    with pytest.raises(lineage.LineageDoseError):
        lineage.exact_objective_exposure_from_training_report(report)


def test_exact_objective_exposure_rejects_policy_arithmetic_drift() -> None:
    exposure = {
        "measurement_status": "bound_exactly",
        "measurement_scope": "current_dose",
        "base_sampled_rows": 100,
        "policy_base_active_sampled_rows": 20,
        "policy_aux_active_sampled_rows": 10,
        "policy_active_sampled_rows": 29,
        "value_active_sampled_rows": 100,
        "anchor_eligible_sampled_rows": 0,
    }
    with pytest.raises(lineage.LineageDoseError, match="exposure arithmetic drift"):
        lineage.direct_lineage_dose(
            declared_producer_sha256=PRODUCER,
            init_checkpoint_sha256=PRODUCER,
            current_sampled_rows=100,
            current_optimizer_steps=1,
            objective_exposure=exposure,
        )


def test_exact_objective_exposure_must_match_current_dose() -> None:
    exposure = {
        "measurement_status": "bound_exactly",
        "measurement_scope": "current_dose",
        "base_sampled_rows": 99,
        "policy_base_active_sampled_rows": 20,
        "policy_aux_active_sampled_rows": 10,
        "policy_active_sampled_rows": 30,
        "value_active_sampled_rows": 99,
        "anchor_eligible_sampled_rows": 0,
    }
    with pytest.raises(
        lineage.LineageDoseError, match="does not match current sampled rows"
    ):
        lineage.direct_lineage_dose(
            declared_producer_sha256=PRODUCER,
            init_checkpoint_sha256=PRODUCER,
            current_sampled_rows=100,
            current_optimizer_steps=1,
            objective_exposure=exposure,
        )


def test_transition_only_initializer_has_zero_preparation_exposure() -> None:
    dose = lineage.direct_lineage_dose(
        declared_producer_sha256=PRODUCER,
        init_checkpoint_sha256=TRANSITIONED,
        initializer_transition_chain=_transition_only_chain(),
        current_sampled_rows=524_288,
        current_optimizer_steps=128,
    )

    assert dose["mode"] == "direct_with_typed_initializer_chain"
    assert dose["initializer_transition_chain"] == _transition_only_chain()
    assert dose["initializer_preparation_exposure"] == {
        "schema_version": "a1-initializer-preparation-exposure-v1",
        "measurement_scope": "initializer_preparation_only",
        "sampled_rows": 0,
        "optimizer_steps": 0,
        "active_parameter_surface": "no_optimizer_surface",
        "policy_active_sampled_rows": 0,
        "value_active_sampled_rows": 0,
        "shared_trunk_active_sampled_rows": 0,
        "auxiliary_head_active_sampled_rows": 0,
    }
    assert dose["prior_sampled_rows"] == 0
    assert dose["prior_optimizer_steps"] == 0
    assert dose["current_sampled_rows"] == 524_288
    assert dose["current_optimizer_steps"] == 128
    assert dose["cumulative_sampled_rows"] == 524_288
    assert dose["cumulative_optimizer_steps"] == 128


def test_pointer_warmup_is_aux_only_preparation_not_prior_learner_dose() -> None:
    dose = lineage.direct_lineage_dose(
        declared_producer_sha256=PRODUCER,
        init_checkpoint_sha256=WARMED,
        initializer_transition_chain=_full_initializer_chain(),
        current_sampled_rows=524_288,
        current_optimizer_steps=128,
    )

    assert dose["initializer_transition_chain"] == _full_initializer_chain()
    assert dose["initializer_preparation_exposure"] == {
        "schema_version": "a1-initializer-preparation-exposure-v1",
        "measurement_scope": "initializer_preparation_only",
        "sampled_rows": 524_288,
        "optimizer_steps": 128,
        "active_parameter_surface": "new_auxiliary_heads_only",
        "policy_active_sampled_rows": 0,
        "value_active_sampled_rows": 0,
        "shared_trunk_active_sampled_rows": 0,
        "auxiliary_head_active_sampled_rows": 524_288,
    }
    # Head preparation is visible but never masquerades as a previous policy,
    # value, or trunk learner dose.
    assert dose["prior_sampled_rows"] == 0
    assert dose["prior_optimizer_steps"] == 0
    assert dose["current_sampled_rows"] == 524_288
    assert dose["current_optimizer_steps"] == 128
    assert dose["cumulative_sampled_rows"] == 524_288
    assert dose["cumulative_optimizer_steps"] == 128


@pytest.mark.parametrize(
    "chain",
    [
        lambda: list(reversed(_full_initializer_chain())),
        lambda: _full_initializer_chain()[1:],
        lambda: [
            _full_initializer_chain()[0],
            _full_initializer_chain()[2],
        ],
    ],
)
def test_initializer_transition_chain_rejects_wrong_order(chain) -> None:
    with pytest.raises(lineage.LineageDoseError, match="transition order"):
        lineage.direct_lineage_dose(
            declared_producer_sha256=PRODUCER,
            init_checkpoint_sha256=WARMED,
            initializer_transition_chain=chain(),
            current_sampled_rows=524_288,
            current_optimizer_steps=128,
        )


def test_initializer_transition_chain_rejects_broken_edge() -> None:
    chain = _full_initializer_chain()
    chain[1]["source_checkpoint_sha256"] = PRODUCER
    with pytest.raises(lineage.LineageDoseError, match="semantic drift"):
        lineage.direct_lineage_dose(
            declared_producer_sha256=PRODUCER,
            init_checkpoint_sha256=WARMED,
            initializer_transition_chain=chain,
            current_sampled_rows=524_288,
            current_optimizer_steps=128,
        )


def test_initializer_transition_chain_rejects_wrong_final_initializer() -> None:
    with pytest.raises(lineage.LineageDoseError, match="actual initializer"):
        lineage.direct_lineage_dose(
            declared_producer_sha256=PRODUCER,
            init_checkpoint_sha256=POINTER,
            initializer_transition_chain=_full_initializer_chain(),
            current_sampled_rows=524_288,
            current_optimizer_steps=128,
        )


@pytest.mark.parametrize(
    "field",
    [
        "policy_active_sampled_rows",
        "value_active_sampled_rows",
        "shared_trunk_active_sampled_rows",
    ],
)
def test_initializer_preparation_rejects_hidden_main_model_exposure(field: str) -> None:
    dose = lineage.direct_lineage_dose(
        declared_producer_sha256=PRODUCER,
        init_checkpoint_sha256=WARMED,
        initializer_transition_chain=_full_initializer_chain(),
        current_sampled_rows=524_288,
        current_optimizer_steps=128,
    )
    dose["initializer_preparation_exposure"][field] = 1
    with pytest.raises(lineage.LineageDoseError, match="preparation exposure drift"):
        lineage.validate_lineage_dose(dose)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sampled_rows", 524_287),
        ("sampled_rows", 524_289),
        ("optimizer_steps", 127),
        ("optimizer_steps", 129),
        ("optimizer_state_terminal", "retained"),
    ],
)
def test_initializer_transition_chain_rejects_wrong_warmup_dose(
    field: str, value: object
) -> None:
    chain = _full_initializer_chain()
    chain[2][field] = value
    with pytest.raises(lineage.LineageDoseError, match="role/dose drift"):
        lineage.direct_lineage_dose(
            declared_producer_sha256=PRODUCER,
            init_checkpoint_sha256=WARMED,
            initializer_transition_chain=chain,
            current_sampled_rows=524_288,
            current_optimizer_steps=128,
        )


@pytest.mark.parametrize("transition_index", [0, 1])
def test_zero_optimizer_transforms_reject_optimizer_steps(
    transition_index: int,
) -> None:
    chain = _full_initializer_chain()
    chain[transition_index]["optimizer_steps"] = 1
    with pytest.raises(lineage.LineageDoseError, match="role/dose drift"):
        lineage.direct_lineage_dose(
            declared_producer_sha256=PRODUCER,
            init_checkpoint_sha256=WARMED,
            initializer_transition_chain=chain,
            current_sampled_rows=524_288,
            current_optimizer_steps=128,
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda row: row.pop("receipt_path"), "field set drift"),
        (lambda row: row.__setitem__("receipt_path", ""), "semantic drift"),
        (
            lambda row: row.__setitem__("receipt_file_sha256", "forged"),
            "typed SHA-256",
        ),
        (
            lambda row: row.__setitem__("receipt_state_sha256", "sha256:not-hex"),
            "typed SHA-256",
        ),
        (
            lambda row: row.__setitem__("schema_version", "forged-transition-v0"),
            "semantic drift",
        ),
    ],
)
def test_initializer_transition_chain_rejects_missing_or_forged_receipts(
    mutation, match: str
) -> None:
    chain = copy.deepcopy(_full_initializer_chain())
    mutation(chain[0])
    with pytest.raises(lineage.LineageDoseError, match=match):
        lineage.direct_lineage_dose(
            declared_producer_sha256=PRODUCER,
            init_checkpoint_sha256=WARMED,
            initializer_transition_chain=chain,
            current_sampled_rows=524_288,
            current_optimizer_steps=128,
        )
