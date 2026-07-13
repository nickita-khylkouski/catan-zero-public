#!/usr/bin/env python3
"""Typed learner-dose contracts shared by launch, completion, and promotion.

The historical A1 evidence used an eight-rank, 1,024-step dose.  The learner
forensics Pareto comparison selected the independently initialized 128-step
dose instead.  Keeping both identities explicit prevents a historical receipt
from silently becoming the recipe for a new candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


SCHEMA_VERSION = "a1-learner-dose-contract-v1"


class LearnerDoseError(ValueError):
    """A learner dose is malformed or differs from its selected identity."""


@dataclass(frozen=True)
class LearnerDose:
    dose_id: str
    optimizer_steps: int
    world_size: int
    per_rank_batch_size: int
    grad_accum_steps: int
    policy_aux_active_batch_size: int
    selection: str

    def __post_init__(self) -> None:
        integer_fields = {
            "optimizer_steps": self.optimizer_steps,
            "world_size": self.world_size,
            "per_rank_batch_size": self.per_rank_batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "policy_aux_active_batch_size": self.policy_aux_active_batch_size,
        }
        invalid = {
            key: value
            for key, value in integer_fields.items()
            if isinstance(value, bool)
            or not isinstance(value, int)
            or (value < 0 if key == "policy_aux_active_batch_size" else value <= 0)
        }
        if invalid:
            raise LearnerDoseError(f"invalid learner-dose integers: {invalid}")
        if not self.dose_id or not self.selection:
            raise LearnerDoseError("learner dose requires a stable id and selection reason")

    @property
    def effective_global_batch_size(self) -> int:
        return self.world_size * self.per_rank_batch_size * self.grad_accum_steps

    @property
    def global_samples(self) -> int:
        return self.optimizer_steps * self.effective_global_batch_size

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "dose_id": self.dose_id,
            "optimizer_steps": self.optimizer_steps,
            "world_size": self.world_size,
            "per_rank_batch_size": self.per_rank_batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "effective_global_batch_size": self.effective_global_batch_size,
            "global_samples": self.global_samples,
            "policy_aux_active_batch_size": self.policy_aux_active_batch_size,
            "selection": self.selection,
        }


HISTORICAL_FULL_DOSE = LearnerDose(
    dose_id="historical-full-8x512-1024",
    optimizer_steps=1024,
    world_size=8,
    per_rank_batch_size=512,
    grad_accum_steps=1,
    policy_aux_active_batch_size=0,
    selection="historical completed diagnostic and production evidence",
)

PARETO_SELECTED_DOSE = LearnerDose(
    dose_id="pareto-short-8x512-128",
    optimizer_steps=128,
    world_size=8,
    per_rank_batch_size=512,
    grad_accum_steps=1,
    policy_aux_active_batch_size=0,
    selection=(
        "independent fresh-f7 Pareto winner: 524288 row draws / 128 optimizer steps"
    ),
)


def assert_payload(
    value: Any,
    expected: LearnerDose,
    *,
    allow_extra: bool = False,
) -> dict[str, Any]:
    """Return a typed payload only when every dose-defining field matches."""

    if not isinstance(value, Mapping):
        raise LearnerDoseError("learner dose must be an object")
    payload = dict(value)
    expected_payload = expected.payload()
    if not allow_extra and set(payload) != set(expected_payload):
        raise LearnerDoseError(
            "learner-dose fields drifted: "
            f"expected {sorted(expected_payload)}, found {sorted(payload)}"
        )
    drift = {
        key: {"expected": wanted, "actual": payload.get(key)}
        for key, wanted in expected_payload.items()
        if payload.get(key) != wanted
    }
    if drift:
        raise LearnerDoseError(f"learner dose drifted: {drift}")
    return payload


def assert_legacy_payload(value: Any, expected: LearnerDose) -> dict[str, Any]:
    """Validate the unversioned projection emitted by historical tools.

    This compatibility path is intentionally unable to authorize a *new* run;
    new manifests must carry :data:`SCHEMA_VERSION` through completion.
    """

    if not isinstance(value, Mapping):
        raise LearnerDoseError("historical learner dose must be an object")
    payload = dict(value)
    projection = {
        "optimizer_steps": expected.optimizer_steps,
        "world_size": expected.world_size,
        "per_rank_batch_size": expected.per_rank_batch_size,
        "global_samples": expected.global_samples,
        "policy_aux_active_batch_size": expected.policy_aux_active_batch_size,
    }
    drift = {
        key: {"expected": wanted, "actual": payload.get(key)}
        for key, wanted in projection.items()
        if payload.get(key) != wanted
    }
    if drift:
        raise LearnerDoseError(f"historical learner dose drifted: {drift}")
    return payload


def report_drift(report: Mapping[str, Any], expected: LearnerDose) -> dict[str, Any]:
    """Return launch/report dose mismatches, including actual sampler draws."""

    required = {
        "max_steps": expected.optimizer_steps,
        "steps_completed": expected.optimizer_steps,
        "world_size": expected.world_size,
        "batch_size": expected.per_rank_batch_size,
        "grad_accum_steps": expected.grad_accum_steps,
        "effective_global_batch_size": expected.effective_global_batch_size,
        "training_row_draws": expected.global_samples,
        "base_training_row_draws": expected.global_samples,
        "policy_aux_training_row_draws": 0,
        "total_training_row_draws": expected.global_samples,
    }
    return {
        key: {"expected": wanted, "actual": report.get(key)}
        for key, wanted in required.items()
        if report.get(key) != wanted
    }


def assert_report(report: Mapping[str, Any], expected: LearnerDose) -> None:
    drift = report_drift(report, expected)
    if drift:
        raise LearnerDoseError(f"completed learner report dose drifted: {drift}")
