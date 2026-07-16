"""Fail-closed launch binding for the legacy Modal L4 Gumbel factory.

This module is deliberately stdlib-only so the guard can be tested without the
Modal SDK, CUDA, or the native Catan wheel installed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "modal-gumbel-factory-gpu-launch-binding-v1"
ACKNOWLEDGEMENT_PREFIX = "acknowledge-legacy-modal-l4-factory:"
CRITICAL_CURRENT_SCIENCE_FIELDS = frozenset(
    {
        "coherent_public_belief_search",
        "event_history_limit",
        "learner_entity_feature_adapter_version",
        "meaningful_public_history",
        "native_mcts_hot_loop",
        "preserve_search_evidence",
        "public_card_count_feature_schema",
        "symmetry_averaged_eval",
        "symmetry_averaged_eval_threshold",
        "target_reliability_audit_fraction",
        "temperature_clock",
    }
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def build_launch_binding(
    *,
    factory_source: Path,
    wheel_path: Path,
    accelerator: str,
    canonical_generation_config: Path,
    production_runtime_config: Path,
    launch_science: Mapping[str, Any],
    containers: int,
    games_per_container: int,
) -> dict[str, Any]:
    """Bind the exact legacy runtime, science recipe, and requested wave size."""

    if not factory_source.is_file():
        raise ValueError(f"Modal GPU factory source is missing: {factory_source}")
    if not wheel_path.is_file():
        raise ValueError(
            "legacy Modal GPU factory wheel is missing: "
            f"{wheel_path}; do not substitute a same-named or newer wheel silently"
        )
    if int(containers) < 1 or int(games_per_container) < 1:
        raise ValueError("containers and games_per_container must both be positive")

    canonical_generation = _load_json_object(
        canonical_generation_config, label="canonical generation config"
    )
    production_runtime = _load_json_object(
        production_runtime_config, label="production runtime config"
    )
    canonical_fields = canonical_generation.get("fields")
    if not isinstance(canonical_fields, dict):
        raise ValueError("canonical generation config fields must be a JSON object")

    comparable_science = {
        key: value for key, value in launch_science.items() if key in canonical_fields
    }
    science_drift = {
        key: {"factory": value, "canonical": canonical_fields[key]}
        for key, value in sorted(comparable_science.items())
        if canonical_fields[key] != value
    }
    missing_critical_science = sorted(
        CRITICAL_CURRENT_SCIENCE_FIELDS - launch_science.keys()
    )
    binding: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "factory_source_sha256": _file_sha256(factory_source),
        "legacy_runtime": {
            "accelerator": str(accelerator),
            "native_wheel_filename": wheel_path.name,
            "native_wheel_sha256": _file_sha256(wheel_path),
        },
        "current_authority": {
            "canonical_generation_config_sha256": _canonical_json_sha256(
                canonical_generation
            ),
            "production_runtime_config_sha256": _canonical_json_sha256(
                production_runtime
            ),
            "native_wheel_filename": production_runtime.get(
                "catanatron_rs_wheel_filename"
            ),
            "native_wheel_sha256": (
                "sha256:"
                + str(production_runtime.get("catanatron_rs_wheel_sha256", ""))
            ),
        },
        "launch_science": dict(launch_science),
        "science_drift_from_current_canonical": science_drift,
        "missing_critical_current_science_fields": missing_critical_science,
        "launch_size": {
            "containers": int(containers),
            "games_per_container": int(games_per_container),
            "games_target": int(containers) * int(games_per_container),
        },
        "modal_concurrency_scope": (
            "max_containers is per app/function pool; this acknowledgement does "
            "not prevent a second concurrent Modal app from allocating another pool"
        ),
    }
    binding["binding_sha256"] = _canonical_json_sha256(binding)
    return binding


def required_acknowledgement(binding: Mapping[str, Any]) -> str:
    digest = str(binding.get("binding_sha256", ""))
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise ValueError("factory launch binding has no valid binding_sha256")
    return ACKNOWLEDGEMENT_PREFIX + digest


def require_acknowledgement(
    binding: Mapping[str, Any], acknowledgement: str
) -> str:
    """Return the accepted token or fail with the exact token required."""

    required = required_acknowledgement(binding)
    if str(acknowledgement) != required:
        drift = binding.get("science_drift_from_current_canonical", {})
        missing = binding.get("missing_critical_current_science_fields", ())
        launch_size = binding.get("launch_size", {})
        runtime = binding.get("legacy_runtime", {})
        raise ValueError(
            "refusing legacy Modal L4 Gumbel factory launch without an exact "
            "runtime/science/size acknowledgement. This factory is not the "
            "current canonical generation path. "
            f"legacy_runtime={runtime!r} launch_size={launch_size!r} "
            f"canonical_science_drift={drift!r} "
            f"missing_critical_current_science_fields={missing!r}. "
            "Re-run only after review with "
            f"--acknowledge-factory-binding {required}"
        )
    return required
