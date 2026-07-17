#!/usr/bin/env python3
"""Issue and replay non-promotable neural information-contract migrations.

Unlike ``a1_function_preserving_upgrade``, this module explicitly permits a
reviewed observation-surface change.  It proves the parameter-topology part is
still the historical deterministic zero-output/suffix-clone construction, then
binds adapter-specific step-zero evidence showing that the v2->v6 information
surface changes both features and model outputs.  Such a receipt establishes
lineage and reproducibility only; it never establishes strength or promotion
eligibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_function_preserving_upgrade as function_upgrade  # noqa: E402
from catan_zero.rl.entity_feature_adapter import (  # noqa: E402
    RUST_ENTITY_ADAPTER_V2,
    RUST_ENTITY_ADAPTER_V5,
    RUST_ENTITY_ADAPTER_V6,
    resolve_checkpoint_entity_feature_adapter,
)


SCHEMA = "a1-information-contract-migration-v1"
MIGRATION_CURRENT_V2_TO_V6_TOPOLOGY_SPLIT1 = (
    "entity_graph.current_v2_to_v6_information_contract+topology+split1.v1"
)
MIGRATION_V5_TO_V7_INPUT_COMPATIBILITY = "entity_graph.v5_to_v7_input_compatibility.v1"
MIGRATION_V5_TO_V8_PUBLIC_RESOURCE_COMPATIBILITY = (
    "entity_graph.v5_to_v8_public_resource_compatibility.v1"
)
CHECKPOINT_PROVENANCE_SCHEMA = "entity-graph-information-contract-migration-v1"
ANCHOR_SCHEMA = "adapter-v6-step0-anchor-evidence-v1"
V7_INPUT_ANCHOR_SCHEMA = "adapter-v7-compatibility-step0-anchor-evidence-v1"
V8_INPUT_ANCHOR_SCHEMA = (
    "adapter-v8-public-resource-compatibility-step0-anchor-evidence-v1"
)


class MigrationError(RuntimeError):
    pass


_ANCHOR_REPLAY_FLOAT_ABS_TOL = 1e-6
_ANCHOR_REPLAY_FLOAT_REL_TOL = 1e-6


def _anchor_replay_matches(expected: object, actual: object) -> bool:
    """Compare replay evidence without making CPU reduction order semantic.

    ``torchrun`` sets ``OMP_NUM_THREADS=1`` by default.  The same CPU forward
    pass therefore differs from a normally issued receipt by a few float32
    ulps even though its discrete identities, features, and decisions are
    unchanged.  Receipt bytes and all non-floating fields remain exact; only
    measured floating diagnostics receive a tight numerical tolerance.
    """

    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        return set(expected) == set(actual) and all(
            _anchor_replay_matches(expected[key], actual[key]) for key in expected
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            _anchor_replay_matches(left, right)
            for left, right in zip(expected, actual, strict=True)
        )
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, int) and isinstance(actual, int):
        return expected == actual
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        return (
            math.isfinite(expected)
            and math.isfinite(float(actual))
            and math.isclose(
                expected,
                float(actual),
                rel_tol=_ANCHOR_REPLAY_FLOAT_REL_TOL,
                abs_tol=_ANCHOR_REPLAY_FLOAT_ABS_TOL,
            )
        )
    return type(expected) is type(actual) and expected == actual


def _verify_anchor_replay(
    expected: Mapping[str, Any],
    actual: object,
    *,
    migration: str,
) -> None:
    """Validate replay invariants before applying diagnostic float tolerance.

    The historical comparator tolerates CPU reduction noise in measured
    diagnostics.  V7's exact-forward claim is not a diagnostic: a nonzero
    replayed output delta must fail even when it is below that tolerance.
    """

    verified_actual = _verify_anchor_evidence(actual, migration=migration)
    if not _anchor_replay_matches(expected, verified_actual):
        raise MigrationError("step-zero adapter-specific anchor replay drift")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _ref(path: Path) -> dict[str, str]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise MigrationError(f"migration artifact must not be a symlink: {expanded}")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise MigrationError(f"migration artifact must be a regular file: {resolved}")
    return {"path": str(resolved), "sha256": _sha(resolved)}


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _finite_nonnegative(value: object) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _verify_anchor_evidence(
    value: object,
    *,
    migration: str = MIGRATION_CURRENT_V2_TO_V6_TOPOLOGY_SPLIT1,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MigrationError("migration lacks step-zero anchor evidence")
    evidence = dict(value)
    v7_input_routing = migration == MIGRATION_V5_TO_V7_INPUT_COMPATIBILITY
    v8_input_routing = (
        migration == MIGRATION_V5_TO_V8_PUBLIC_RESOURCE_COMPATIBILITY
    )
    v5_compatibility = v7_input_routing or v8_input_routing
    expected_schema = (
        V8_INPUT_ANCHOR_SCHEMA
        if v8_input_routing
        else V7_INPUT_ANCHOR_SCHEMA
        if v7_input_routing
        else ANCHOR_SCHEMA
    )
    expected_source_adapter = (
        RUST_ENTITY_ADAPTER_V5 if v5_compatibility else RUST_ENTITY_ADAPTER_V2
    )
    if (
        evidence.get("schema_version") != expected_schema
        or evidence.get("device") != "cpu"
        or evidence.get("source_adapter") != expected_source_adapter
        or evidence.get("target_adapter") != RUST_ENTITY_ADAPTER_V6
        or evidence.get("public_observation") is not True
        or evidence.get("separate_adapter_specific_entity_features") is not True
        or evidence.get("separate_adapter_specific_action_contexts") is not True
        or evidence.get("promotion_eligible") is not False
    ):
        raise MigrationError("step-zero anchor contract is malformed")
    if evidence.get("forward_identical") is not v5_compatibility:
        raise MigrationError(
            "step-zero forward-identity claim mismatches migration type"
        )
    if v5_compatibility and evidence.get("adapter_features_identical") is not False:
        raise MigrationError("V7/V8 evidence must bind the real V5->V6 feature change")
    numeric = (
        "migration_output_max_abs_diff",
        "feature_max_abs_diff",
        "legal_policy_forward_kl_mean",
        "legal_policy_forward_kl_max",
        "legal_policy_reverse_kl_mean",
        "legal_policy_reverse_kl_max",
        "legal_policy_top1_flip_rate",
        "scalar_value_rmse",
        "scalar_value_max_abs_error",
    )
    if not all(_finite_nonnegative(evidence.get(name)) for name in numeric):
        raise MigrationError("step-zero anchor evidence is non-finite")
    if (
        evidence.get("topology_construction_proof")
        != "deterministic_parameter_replay_in_receipt"
        or isinstance(evidence.get("feature_changed_value_count"), bool)
        or not isinstance(evidence.get("feature_changed_value_count"), int)
    ):
        raise MigrationError(
            "step-zero evidence does not prove a bounded real migration"
        )
    feature_max = float(evidence["feature_max_abs_diff"])
    feature_changes = int(evidence["feature_changed_value_count"])
    output_max = float(evidence["migration_output_max_abs_diff"])
    if feature_max <= 0.0 or feature_changes <= 0:
        raise MigrationError("information migration must prove a real feature change")
    if v5_compatibility and output_max != 0.0:
        raise MigrationError(
            "V7 compatibility routing must preserve exact forward output"
        )
    if not v7_input_routing and output_max <= 0.0:
        raise MigrationError("V2->V6 evidence must prove real output drift")
    anchors = evidence.get("anchors")
    if not isinstance(anchors, list) or not anchors:
        raise MigrationError("migration anchor suite is empty")
    phases = {row.get("phase") for row in anchors if isinstance(row, Mapping)}
    resource_rows = [
        row
        for row in anchors
        if isinstance(row, Mapping)
        and isinstance(row.get("actor_resource_total"), int)
        and int(row["actor_resource_total"]) > 0
    ]
    if "BUILD_INITIAL_ROAD" not in phases or not resource_rows:
        raise MigrationError(
            "migration anchors must include BUILD_INITIAL_ROAD and resource states"
        )
    for row in anchors:
        if not isinstance(row, Mapping):
            raise MigrationError("migration anchor row is malformed")
        for name in (
            "migration_output_max_abs_diff",
            "feature_max_abs_diff",
            "legal_policy_forward_kl",
            "legal_policy_reverse_kl",
            "scalar_value_abs_error",
        ):
            if not _finite_nonnegative(row.get(name)):
                raise MigrationError(f"migration anchor {name} is non-finite")
        identity = row.get("anchor_identity_sha256")
        if (
            not isinstance(identity, str)
            or not identity.startswith("sha256:")
            or len(identity) != 71
            or not isinstance(row.get("legal_policy_top1_flip"), bool)
        ):
            raise MigrationError("migration anchor identity/flip evidence is malformed")
        if isinstance(row.get("feature_changed_value_count"), bool) or not isinstance(
            row.get("feature_changed_value_count"), int
        ):
            raise MigrationError("migration anchor feature-change count is malformed")
    count = len(anchors)
    if (
        evidence.get("anchor_count") != count
        or isinstance(evidence.get("legal_policy_top1_flip_count"), bool)
        or not isinstance(evidence.get("legal_policy_top1_flip_count"), int)
    ):
        raise MigrationError("migration anchor aggregate counts are malformed")
    flip_count = sum(int(row["legal_policy_top1_flip"]) for row in anchors)
    forward = [float(row["legal_policy_forward_kl"]) for row in anchors]
    reverse = [float(row["legal_policy_reverse_kl"]) for row in anchors]
    values = [float(row["scalar_value_abs_error"]) for row in anchors]
    migration_diffs = [float(row["migration_output_max_abs_diff"]) for row in anchors]
    feature_diffs = [float(row["feature_max_abs_diff"]) for row in anchors]
    feature_change_count = sum(
        int(row["feature_changed_value_count"]) for row in anchors
    )
    import numpy as np

    expected_aggregates = {
        "legal_policy_forward_kl_mean": sum(forward) / count,
        "legal_policy_forward_kl_max": max(forward),
        "legal_policy_reverse_kl_mean": sum(reverse) / count,
        "legal_policy_reverse_kl_max": max(reverse),
        "legal_policy_top1_flip_rate": flip_count / count,
        "scalar_value_rmse": float(np.sqrt(np.mean(np.square(values)))),
        "scalar_value_max_abs_error": max(values),
        "migration_output_max_abs_diff": max(migration_diffs),
        "feature_max_abs_diff": max(feature_diffs),
    }
    if (
        evidence["legal_policy_top1_flip_count"] != flip_count
        or evidence["feature_changed_value_count"] != feature_change_count
        or any(
            float(evidence[name]) != expected
            for name, expected in expected_aggregates.items()
        )
    ):
        raise MigrationError("migration anchor aggregate metrics do not replay")
    expected_set = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                [row["anchor_identity_sha256"] for row in anchors],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    if evidence.get("anchor_set_sha256") != expected_set:
        raise MigrationError("migration anchor-set identity drift")
    return evidence


def _verify_topology_delta(
    source: Path,
    migrated: Path,
    *,
    provenance: Mapping[str, Any],
    anchor: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay parameter construction through the reviewed historical v5 spec.

    The temporary v5 view is only a deterministic parameter/config-delta
    verifier. It is not an intermediate checkpoint and never represents the
    actual incumbent's v2 information surface.
    """

    import torch

    raw = torch.load(migrated, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise MigrationError("migrated checkpoint is malformed")
    normalized = dict(raw)
    normalized.pop("information_contract_migration_provenance", None)
    normalized["entity_feature_adapter"] = {
        "schema_version": "entity-feature-adapter-v1",
        "version": RUST_ENTITY_ADAPTER_V5,
    }
    normalized["upgrade_provenance"] = {
        "schema_version": "entity-graph-upgrade-v1",
        "source_checkpoint_sha256": _sha(source).removeprefix("sha256:"),
        "flags": dict(provenance["flags"]),
        "initialization_seed": int(provenance["initialization_seed"]),
        "trained_value_readouts_added": [],
        # These values authenticate only deterministic parameter construction
        # in the historical verifier. They do not claim V2/V6 forward parity.
        "forward_max_diff": 0.0,
        "forward_tolerance": function_upgrade._module_forward_tolerance(  # noqa: SLF001
            function_upgrade.MODULE_CURRENT_V5_TOPOLOGY_VALUE_TOWER_SPLIT_1
        ),
        "forward_identical_at_init": True,
        "value_tower_initialization": provenance.get("value_tower_initialization"),
    }
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="a1-migration-v5-view-", suffix=".pt", delete=False
        ) as handle:
            temporary_name = handle.name
        torch.save(normalized, temporary_name)
        replay = function_upgrade.inspect_upgrade(
            source,
            Path(temporary_name),
            module=function_upgrade.MODULE_CURRENT_V5_TOPOLOGY_VALUE_TOWER_SPLIT_1,
        )
    except (OSError, function_upgrade.UpgradeError) as error:
        raise MigrationError(f"migration topology replay failed: {error}") from error
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return {
        "module": function_upgrade.MODULE_CURRENT_V5_TOPOLOGY_VALUE_TOWER_SPLIT_1,
        "shared_parameters_bit_identical": replay["shared_parameters_bit_identical"],
        "new_parameters": replay["new_parameters"],
        "new_parameter_initialization": replay["new_parameter_initialization"],
        "effective_source_config_sha256": replay["effective_source_config_sha256"],
        "effective_migrated_config_sha256": replay["effective_upgraded_config_sha256"],
    }


def _verify_v7_input_routing_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    initialization_seed: int,
    require_exact_public_resource_residual: bool = False,
) -> dict[str, Any]:
    """Replay the complete budgeted V7 construction from the V5 bytes.

    The inherited tensors and all non-migration checkpoint metadata must remain
    byte-identical. V7 adds the topology residual, the low-rank action decoder,
    and two compatibility-preserving input residuals in one deterministic
    construction; checking only the two zero residuals would leave more than a
    million unauthenticated parameters outside the receipt.
    """

    import dataclasses
    import torch

    before_model = before.get("model")
    after_model = after.get("model")
    if not isinstance(before_model, Mapping) or not isinstance(after_model, Mapping):
        raise MigrationError("V7 migration checkpoint model state is malformed")
    added = set(after_model) - set(before_model)
    removed = set(before_model) - set(after_model)
    required_parameters = {
        "v6_exact_resource_residual.weight",
        "v6_initial_road_residual.weight",
    }
    if require_exact_public_resource_residual:
        required_parameters.add("public_card_exact_resource_residual.weight")
    allowed_prefixes = (
        "topology_residual_adapter.",
        "action_cross_blocks.0.",
        "v6_exact_resource_residual.",
        "v6_initial_road_residual.",
        *(
            ("public_card_exact_resource_residual.",)
            if require_exact_public_resource_residual
            else ()
        ),
    )
    if (
        removed
        or not required_parameters <= added
        or any(not name.startswith(allowed_prefixes) for name in added)
    ):
        raise MigrationError(
            "V7 input migration parameter delta drift: "
            f"added={sorted(added)} removed={sorted(removed)}"
        )
    changed = [
        name
        for name in before_model
        if not function_upgrade._tensor_equal_exact(  # noqa: SLF001
            before_model[name], after_model[name]
        )
    ]
    if changed:
        raise MigrationError(
            f"V7 input migration changed inherited parameters: {changed[:8]}"
        )
    from catan_zero.rl.entity_token_policy import EntityGraphConfig

    before_config = function_upgrade._config(before.get("config"))  # noqa: SLF001
    after_config = function_upgrade._config(after.get("config"))  # noqa: SLF001
    known = {field.name for field in dataclasses.fields(EntityGraphConfig)}
    if set(before_config) - known or set(after_config) - known:
        raise MigrationError("V7 migration checkpoint config has unknown fields")
    effective_before = dataclasses.asdict(EntityGraphConfig(**before_config))
    effective_after = dataclasses.asdict(EntityGraphConfig(**after_config))
    if effective_before.get("v6_compatibility_preserving_inputs") is not False:
        raise MigrationError("V7 migration source already uses compatibility routing")
    expected_config = {
        **effective_before,
        "topology_residual_adapter": True,
        "v6_compatibility_preserving_inputs": True,
        "action_cross_attention_layers": 1,
        "action_cross_attention_bottleneck": 80,
        **(
            {"public_card_exact_resource_residual": True}
            if require_exact_public_resource_residual
            else {}
        ),
    }
    if effective_after != expected_config:
        raise MigrationError("V7 migration changed config outside input routing")

    # Rebuild the complete target module from the source tensors and the
    # provenance-bound initialization seed. This simultaneously proves that
    # the set of additions is complete and that every random/zero/identity
    # initialization matches the actual reviewed constructor.
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    static = before.get("static_action_features")
    if static is None:
        raise MigrationError("V7 migration source lacks static action features")
    if hasattr(static, "detach"):
        static = static.detach().cpu().numpy()
    replay = EntityGraphPolicy(
        EntityGraphConfig(**expected_config),
        static,
        seed=int(initialization_seed),
        device="cpu",
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6,
    )
    missing, unexpected = replay.model.load_state_dict(before_model, strict=False)
    if unexpected or set(missing) != added:
        raise MigrationError(
            "V7 deterministic constructor parameter delta drift: "
            f"missing={sorted(missing)} added={sorted(added)} "
            f"unexpected={sorted(unexpected)}"
        )
    replay_state = replay.model.state_dict()
    initialization: dict[str, str] = {}
    for parameter in sorted(added):
        observed = after_model[parameter]
        expected = replay_state[parameter]
        if not function_upgrade._tensor_equal_exact(expected, observed):  # noqa: SLF001
            raise MigrationError(
                f"V7 deterministic initialization replay drift: {parameter}"
            )
        if torch.equal(expected, torch.zeros_like(expected)):
            initialization[parameter] = "zeros"
        elif torch.equal(expected, torch.ones_like(expected)):
            initialization[parameter] = "ones"
        else:
            initialization[parameter] = "seeded_torch_default"

    ignored = {
        "model",
        "config",
        "entity_feature_adapter",
        "information_contract_migration_provenance",
        # Information migrations deliberately remove the source checkpoint's
        # function-preserving upgrade claim. inspect_migration independently
        # requires this key to be absent from the migrated artifact, so its
        # removal is mandatory rather than unauthenticated metadata drift.
        "upgrade_provenance",
    }
    unexpected_added = sorted(
        set(after) - set(before) - {"information_contract_migration_provenance"}
    )
    metadata_drift = [
        key
        for key in before
        if key not in ignored
        and (
            key not in after or not function_upgrade._equal(before[key], after[key])  # noqa: SLF001
        )
    ]
    if unexpected_added or metadata_drift:
        raise MigrationError(
            "V7 migration changed checkpoint metadata: "
            f"added={unexpected_added} drift={metadata_drift}"
        )
    source_adapter, _ = resolve_checkpoint_entity_feature_adapter(
        before.get("entity_feature_adapter"),
        metadata_present="entity_feature_adapter" in before,
    )
    target_adapter, _ = resolve_checkpoint_entity_feature_adapter(
        after.get("entity_feature_adapter"),
        metadata_present="entity_feature_adapter" in after,
    )
    if (
        source_adapter != RUST_ENTITY_ADAPTER_V5
        or target_adapter != RUST_ENTITY_ADAPTER_V6
    ):
        raise MigrationError(
            "V7 compatibility migration must be constructed directly from "
            f"V5/f7 bytes, got {source_adapter} -> {target_adapter}"
        )
    return {
        "shared_parameters_bit_identical": True,
        "shared_parameter_count": len(before_model),
        "new_parameters": sorted(added),
        "new_parameter_initialization": initialization,
        "source_input_routing": "v5_legacy_resource_and_initial_road_inputs",
        "target_input_routing": (
            "v6_raw_inputs_with_legacy_encoder_views_plus_zero_output_residuals"
        ),
        "effective_source_config_sha256": function_upgrade._digest(  # noqa: SLF001
            function_upgrade._effective_config_receipt_view(  # noqa: SLF001
                effective_before
            )
        ),
        "effective_migrated_config_sha256": function_upgrade._digest(  # noqa: SLF001
            function_upgrade._effective_config_receipt_view(  # noqa: SLF001
                effective_after
            )
        ),
    }


def inspect_migration(
    source: Path,
    migrated: Path,
    *,
    migration: str = MIGRATION_CURRENT_V2_TO_V6_TOPOLOGY_SPLIT1,
) -> dict[str, Any]:
    if migration not in {
        MIGRATION_CURRENT_V2_TO_V6_TOPOLOGY_SPLIT1,
        MIGRATION_V5_TO_V7_INPUT_COMPATIBILITY,
        MIGRATION_V5_TO_V8_PUBLIC_RESOURCE_COMPATIBILITY,
    }:
        raise MigrationError(f"information migration is not allowlisted: {migration!r}")
    import torch

    source_ref, migrated_ref = _ref(source), _ref(migrated)
    before = torch.load(source_ref["path"], map_location="cpu", weights_only=False)
    after = torch.load(migrated_ref["path"], map_location="cpu", weights_only=False)
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise MigrationError("migration checkpoints are malformed")
    provenance = after.get("information_contract_migration_provenance")
    v7_input_routing = migration == MIGRATION_V5_TO_V7_INPUT_COMPATIBILITY
    v8_input_routing = (
        migration == MIGRATION_V5_TO_V8_PUBLIC_RESOURCE_COMPATIBILITY
    )
    v5_compatibility = v7_input_routing or v8_input_routing
    expected_checkpoint_migration = (
        "v5_to_v8_public_resource_compatibility"
        if v8_input_routing
        else "v5_to_v7_input_compatibility"
        if v7_input_routing
        else "current_v2_to_v6_topology_split1"
    )
    expected_source_adapter = (
        RUST_ENTITY_ADAPTER_V5 if v5_compatibility else RUST_ENTITY_ADAPTER_V2
    )
    expected_source_routing = (
        "v5_legacy_resource_and_initial_road_inputs"
        if v5_compatibility
        else "adapter_v2_legacy_information_surface"
    )
    expected_target_routing = (
        "v6_raw_inputs_with_legacy_encoder_views_plus_zero_output_residuals"
        if v5_compatibility
        else "adapter_v6_exact_resource_and_initial_road_surface"
    )
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("schema_version") != CHECKPOINT_PROVENANCE_SCHEMA
        or provenance.get("migration") != expected_checkpoint_migration
        or provenance.get("source_checkpoint_sha256")
        != source_ref["sha256"].removeprefix("sha256:")
        or provenance.get("source_adapter") != expected_source_adapter
        or provenance.get("target_adapter") != RUST_ENTITY_ADAPTER_V6
        or (
            v5_compatibility
            and provenance.get("source_input_routing") != expected_source_routing
        )
        or (
            v5_compatibility
            and provenance.get("target_input_routing") != expected_target_routing
        )
        or provenance.get("forward_identical") is not v5_compatibility
        or provenance.get("promotion_eligible") is not False
        or provenance.get("commissioning_status")
        != "non_promotable_architecture_treatment"
        or "upgrade_provenance" in after
    ):
        raise MigrationError("checkpoint migration provenance is malformed")
    try:
        initialization_seed = int(provenance["initialization_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise MigrationError(
            "checkpoint migration provenance lacks a valid initialization seed"
        ) from error
    source_adapter, _ = resolve_checkpoint_entity_feature_adapter(
        before.get("entity_feature_adapter"),
        metadata_present="entity_feature_adapter" in before,
    )
    target_adapter, _ = resolve_checkpoint_entity_feature_adapter(
        after.get("entity_feature_adapter"),
        metadata_present="entity_feature_adapter" in after,
    )
    if (
        source_adapter != expected_source_adapter
        or target_adapter != RUST_ENTITY_ADAPTER_V6
    ):
        raise MigrationError(
            "checkpoint adapters do not realize the selected information migration: "
            f"{source_adapter} -> {target_adapter}"
        )
    anchor = _verify_anchor_evidence(
        provenance.get("step0_anchor_evidence"),
        migration=migration,
    )
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from tools import f69_upgrade_checkpoint_config as checkpoint_migration

    base_policy = EntityGraphPolicy.load(source_ref["path"], device="cpu")
    migrated_policy = EntityGraphPolicy.load(migrated_ref["path"], device="cpu")
    base_policy.model.eval()
    migrated_policy.model.eval()
    recomputed_anchor = checkpoint_migration._migration_anchor_evidence(  # noqa: SLF001
        base_policy,
        migrated_policy,
        "cpu",
        migration=expected_checkpoint_migration,
    )
    _verify_anchor_replay(anchor, recomputed_anchor, migration=migration)
    topology = (
        _verify_v7_input_routing_delta(
            before,
            after,
            initialization_seed=initialization_seed,
            require_exact_public_resource_residual=v8_input_routing,
        )
        if v5_compatibility
        else _verify_topology_delta(
            Path(source_ref["path"]),
            Path(migrated_ref["path"]),
            provenance=provenance,
            anchor=anchor,
        )
    )
    return {
        "migration": migration,
        "source": source_ref,
        "migrated_initializer": migrated_ref,
        "source_adapter": source_adapter,
        "target_adapter": target_adapter,
        "forward_identical": bool(v5_compatibility),
        "promotion_eligible": False,
        "commissioning_status": "non_promotable_architecture_treatment",
        "step0_anchor_evidence": anchor,
        "topology_replay": topology,
    }


def issue_receipt(
    source: Path,
    migrated: Path,
    output: Path,
    *,
    migration: str = MIGRATION_CURRENT_V2_TO_V6_TOPOLOGY_SPLIT1,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA,
        **inspect_migration(source, migrated, migration=migration),
    }
    payload["receipt_sha256"] = _digest(payload)
    destination = output.expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        os.chmod(destination, 0o444)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise MigrationError(
            f"refusing to overwrite information migration receipt: {destination}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def verify_receipt(path: Path) -> dict[str, Any]:
    receipt_ref = _ref(path)
    try:
        payload = json.loads(Path(receipt_ref["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MigrationError(f"cannot read migration receipt: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA:
        raise MigrationError("migration receipt schema is invalid")
    declared = payload.get("receipt_sha256")
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    if declared != _digest(unsigned):
        raise MigrationError("migration receipt digest mismatch")
    replayed = inspect_migration(
        Path(payload["source"]["path"]),
        Path(payload["migrated_initializer"]["path"]),
        migration=str(payload.get("migration", "")),
    )
    expected = {"schema_version": SCHEMA, **replayed}
    expected["receipt_sha256"] = _digest(expected)
    if payload != expected:
        raise MigrationError("migration receipt replay drift")
    return {
        **replayed,
        "receipt": receipt_ref,
        "receipt_sha256": declared,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--migrated", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--migration",
        default=MIGRATION_CURRENT_V2_TO_V6_TOPOLOGY_SPLIT1,
        choices=(
            MIGRATION_CURRENT_V2_TO_V6_TOPOLOGY_SPLIT1,
            MIGRATION_V5_TO_V7_INPUT_COMPATIBILITY,
            MIGRATION_V5_TO_V8_PUBLIC_RESOURCE_COMPATIBILITY,
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = issue_receipt(
            args.source,
            args.migrated,
            args.output,
            migration=args.migration,
        )
    except MigrationError as error:
        raise SystemExit(f"REFUSED: {error}") from error
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
