"""Shared prior-temperature semantics for evaluators and search."""

from __future__ import annotations

import math


MIN_EFFECTIVE_PRIOR_TEMPERATURE = 1.0e-6


def positive_prior_temperature(value: float, *, name: str) -> float:
    """Validate a temperature that an evaluator reports it already applied."""

    temperature = float(value)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return temperature


def effective_prior_temperature(value: float, *, name: str) -> float:
    """Validate and return the configured temperature actually executed.

    The neural evaluator historically floors positive configured temperatures
    at ``1e-6`` for numerical safety. Search and provenance must use that same
    effective value; otherwise a sub-floor request executes one operator while
    its resume and target identities attest another. An evaluator's explicit
    ``applied_prior_temperature`` marker is already an effective measurement
    and is validated, not clamped, by :func:`positive_prior_temperature`.
    """

    temperature = positive_prior_temperature(value, name=name)
    return max(temperature, MIN_EFFECTIVE_PRIOR_TEMPERATURE)
