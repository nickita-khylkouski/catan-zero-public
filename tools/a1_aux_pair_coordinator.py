#!/usr/bin/env python3
"""Central authority for the corrected two-stage A1 auxiliary experiment.

This module deliberately does not train a model.  It is the small central
transaction layer between host-local canonical verifiers and the learner
executor.  It prevents four scientifically expensive mistakes:

* treating the superseded recovery plan as current recipe authority;
* chaining from an unpromoted P1 candidate instead of the current handoff parent;
* commissioning a separate random auxiliary initializer for each arm; and
* reissuing an arm under another label after seeing an inconvenient result.

The transaction is an append-only DAG of O_EXCL-created JSON artifacts:

    experiment -> warmup claim -> warmup terminal
               -> geometry claim -> geometry terminal -> pair issued
               -> AUX0/AUXT claims -> terminals -> pair terminal

Every artifact binds the digest of its predecessor.  An exact repeated call is
an idempotent resume; different bytes at an already-issued stage are refused.
Host-local paths and timestamps are intentionally absent from scientific
identity.  Exact host, SSH host key, checkout, and eight physical GPU identities
remain in the execution identity and therefore cannot drift at launch.

The geometry/dose authority and the learner recipe/data authority are separate
on purpose.  ``a1_post_p1_diagnosis_plan:v5`` selected generic FP32 8x512 and a
524,288-row dose.  The current authenticated P1 receipt plus the typed
64/12/4/20 composite selects the learner objective and data.  Neither authority
selects an initializer: every arm starts from the separately authenticated
exact recovered generator reference, after one shared pointer-head commissioning
transaction.  Recovery does not recreate historical promotion proof.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Iterable, Mapping

from tools import a1_post_p1_diagnosis_plan as post_p1
from tools import a1_pre_wave_contract as pre_wave
from tools import a1_scientific_evidence as scientific_evidence
from tools import a1_v5_disaster_recovery as v5_recovery
from tools import a1_v5_recovery_gate as v5_recovery_gate
from tools.fleet import a1_production_executor as production_executor


POST_P1_PLAN_SCHEMA = "a1-post-p1-optimization-architecture-plan-v5"
STALE_RECOVERY_PLAN_SCHEMA = "a1-learner-recovery-plan-v1"
GEOMETRY_DOSE_AUTHORITY_SCHEMA = "a1-current-geometry-dose-authority-v2"
LEGACY_GEOMETRY_DOSE_AUTHORITY_SCHEMA = "a1-post-p1-geometry-dose-authority-v1"
P1_RECIPE_DATA_AUTHORITY_SCHEMA = "a1-p1-recipe-data-authority-v1"
P1_FINAL_LOCK_AUTHORITY_SCHEMA = "a1-p1-final-v3-lock-authority-v1"
P1_SWEEP_SCHEMA = "a1-p1-central-kl-sweep-v1"
P1_EVALUATION_PLAN_SCHEMA = "a1-p1-fixed-evaluation-plan-v1"
AUX_EVALUATION_PLAN_SCHEMA = "a1-aux-fixed-evaluation-plan-v1"
HANDOFF_PARENT_AUTHORITY_SCHEMA = "a1-current-recovered-parent-authority-v1"
POINTER_UPGRADE_AUTHORITY_SCHEMA = "a1-aux-pointer-upgrade-authority-v1"
EXPERIMENT_SCHEMA = "a1-aux-two-stage-experiment-v1"
ALLOCATION_SCHEMA = "a1-exact-b200-allocation-v1"
WARMUP_RECIPE_SCHEMA = "a1-aux-pointer-warmup-recipe-v1"
SELECTOR_RULE_SCHEMA = "a1-aux-gradient-selector-rule-v1"
NATIVE_RUNTIME_AUTHORITY_SCHEMA = "a1-native-runtime-authority-v1"
PAIR_SCHEMA = "a1-aux-pair-contract-v1"
EXECUTOR_AUTHORITY_SCHEMA = "a1-aux-pair-executor-authority-v1"
EXECUTION_SCHEMA = "a1-central-stage-execution-v1"
FINAL_REPLICATION_SCHEMA = "a1-final-replication-authority-v1"
FINAL_EXECUTOR_AUTHORITY_SCHEMA = "a1-final-replication-executor-authority-v1"
FINAL_GATE_PLAN_SCHEMA = "a1-final-full-gate-plan-v1"
FINAL_GATE_ENTRY_SCHEMA = "a1-final-full-gate-entry-authority-v1"
PUBLISHED_EXECUTOR_AUTHORITY_SCHEMA = "a1-published-executor-authority-v1"

SHORT_SAMPLE_DOSE = 524_288
SHORT_OPTIMIZER_STEPS = 128
WORLD_SIZE = 8
LOCAL_BATCH_SIZE = 512
GLOBAL_BATCH_SIZE = WORLD_SIZE * LOCAL_BATCH_SIZE
P1_SAMPLER_SEED = 424_242
B200_LEARNER_HOST_ID = "b200-learner"
B200_LEARNER_HOSTNAME = "149-118-65-110"
B200_LEARNER_MACHINE_ID = "e71d46177526e026a826ec4afcd39d70"
B200_LEARNER_GPU_UUIDS = (
    "GPU-c444a2e6-e5e4-0974-4144-9807e6f7d68a",
    "GPU-a6f349ce-a4fb-3291-d268-d4950107751f",
    "GPU-7998a0f4-43e2-c0f7-bf7c-b2999fdd63c8",
    "GPU-80857176-d21e-4b06-ae6c-9821f4395c52",
    "GPU-1971a2ec-2a2d-1120-2f4a-72a154703794",
    "GPU-82c0cf1c-c4e2-67ff-c0fa-dcd89cc3967f",
    "GPU-7ea809bd-52f9-9422-d69c-5bb446186cbe",
    "GPU-7c0a25f9-9c0a-fa76-99ec-1a1fca4d8bea",
)
_NATIVE_RELEASE = production_executor._native_wheel_release_identity()
# Backward-compatible exported aliases, derived from the single runtime
# contract rather than maintained as an independent coordinator authority.
NATIVE_WHEEL_SHA256 = _NATIVE_RELEASE["sha256"]
NATIVE_CAPABILITIES = tuple(_NATIVE_RELEASE["required_capabilities"])
ARM_CONTROL = "AUX0"
ARM_TREATMENT = "AUXT"
ARMS = (ARM_CONTROL, ARM_TREATMENT)
STAGES = ("WARMUP", "GEOMETRY", *ARMS)
P1_ARMS = ("K0", "K3", "K10")
P1_SCHEDULE_SLOTS = {"K0": 0, "K3": 1, "K10": 2}
P1_TARGET_GLOBAL_DECIMALS = {"K0": "0", "K3": "0.03", "K10": "0.1"}
P1_COEFFICIENT_QUANTUM = Decimal("0.000000000001")

# Evaluation is an experiment input, not an operator knob.  These disjoint
# cohorts live in the reserved validation-only seed band and are deliberately
# fixed here so neither a launcher nor a disappointed experimenter can shrink
# a panel or replace its seeds after seeing an arm result.
P1_INTERNAL_BASE_SEED = 6_199_600_000
P1_EXTERNAL_BASE_SEED = 6_199_610_000
AUX_INTERNAL_BASE_SEED = 6_199_620_000
AUX_EXTERNAL_BASE_SEED = 6_199_630_000
FINAL_PRIMARY_BASE_SEED = 6_199_640_000
FINAL_SAFETY_BASE_SEED = 6_199_650_000
FINAL_EXTERNAL_BASE_SEED = 6_199_660_000
FINAL_SAMPLER_SEED = 424_243

COMPONENT_IDS = (
    "current_producer",
    "recent_history",
    "hard_negative",
    "historical_replay",
)
COMPONENT_RATIOS = (0.64, 0.12, 0.04, 0.20)

POINTER_MODULE = "entity_graph.aux_subgoal_pointer_heads.v1"
POINTER_FLAGS = {
    "aux_subgoal_heads": True,
    "aux_settlement_pointer_head": True,
}
POINTER_TRAINABLE_PREFIXES = (
    "aux_longest_road_head.",
    "aux_largest_army_head.",
    "aux_vp_in_n_head.",
    "aux_next_settlement_pointer_head.",
    "aux_robber_target_head.",
)
WARMUP_MAIN_OBJECTIVE_ZERO = {
    "policy_loss_weight": 0.0,
    "value_loss_weight": 0.0,
    "final_vp_loss_weight": 0.0,
    "q_loss_weight": 0.0,
    "policy_kl_anchor_weight": 0.0,
    "policy_surprise_weight": 0.0,
    "truncated_vp_margin_value_weight": 0.0,
}

_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}")


class CoordinatorError(RuntimeError):
    """Fail-closed central transaction error."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CoordinatorError(f"value is not canonical JSON: {error}") from error


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise CoordinatorError(f"{where} must be a canonical sha256 digest")
    return value


def _repo_tool_sha256(relative_path: str) -> str:
    """Hash one repo-relative producer whose exact bytes are authority."""

    path = (pre_wave.REPO_ROOT / relative_path).resolve(strict=True)
    try:
        path.relative_to(pre_wave.REPO_ROOT.resolve(strict=True))
    except ValueError as error:
        raise CoordinatorError("producer tool escaped the repository") from error
    if not path.is_file():
        raise CoordinatorError(f"producer tool is not a file: {relative_path}")
    return production_executor._sha256(path)


def _require_exact_keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise CoordinatorError(
            f"{where} shape drift: expected={sorted(expected)} actual={actual}"
        )
    return value


def _sealed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    if "state_sha256" in result:
        raise CoordinatorError("caller may not pre-populate state_sha256")
    result["state_sha256"] = _digest(result)
    return result


def _verify_sealed(payload: Any, where: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CoordinatorError(f"{where} is not an object")
    unsigned = dict(payload)
    stated = unsigned.pop("state_sha256", None)
    if stated != _digest(unsigned):
        raise CoordinatorError(f"{where} state digest drift")
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_json(path: Path, *, where: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CoordinatorError(f"{where} must be a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CoordinatorError(f"cannot load {where}: {error}") from error
    return _verify_sealed(payload, where)


def _stable_read_immutable_json(
    path: Path, *, where: str
) -> tuple[dict[str, Any], str, tuple[int, int, int, int, int]]:
    """Read one immutable artifact through a pinned inode and reject replacement."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CoordinatorError(f"cannot open {where}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_mode & 0o222:
            raise CoordinatorError(f"{where} must be an immutable regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise CoordinatorError(f"{where} changed while read")
    live = path.stat(follow_symlinks=False)
    if identity != (
        live.st_dev,
        live.st_ino,
        live.st_size,
        live.st_mtime_ns,
        live.st_ctime_ns,
    ):
        raise CoordinatorError(f"{where} was replaced while read")
    raw = b"".join(chunks)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CoordinatorError(f"cannot parse {where}: {error}") from error
    return (
        _verify_sealed(payload, where),
        "sha256:" + hashlib.sha256(raw).hexdigest(),
        identity,
    )


def _write_once(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create an immutable artifact, or idempotently replay exact bytes."""

    sealed = _sealed(payload)
    encoded = json.dumps(sealed, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise CoordinatorError(
            f"artifact directory may not be a symlink: {path.parent}"
        )
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing, _sha256, _identity = _stable_read_immutable_json(
            path, where=path.name
        )
        if existing != sealed:
            raise CoordinatorError(
                f"stage already issued with different authority: {path.name}"
            )
        return existing
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o444)
        _fsync_directory(path.parent)
    except BaseException:
        # A short/failed creator never becomes resumable authority.
        try:
            path.unlink(missing_ok=True)
            _fsync_directory(path.parent)
        finally:
            raise
    return sealed


def _artifact_dir(root: Path, experiment_id: str, *, create: bool) -> Path:
    _require_sha(experiment_id, "experiment_id")
    root = root.expanduser()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise CoordinatorError("coordinator root must be a real directory")
    resolved_root = root.resolve(strict=True)
    directory = resolved_root / experiment_id.removeprefix("sha256:")
    if create:
        directory.mkdir(mode=0o700, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise CoordinatorError("experiment directory must be a real directory")
    return directory


def _artifact(
    root: Path, experiment_id: str, filename: str, *, required: bool = True
) -> dict[str, Any] | None:
    path = _artifact_dir(root, experiment_id, create=False) / filename
    if not path.exists() and not required:
        return None
    payload, _sha256, _identity = _stable_read_immutable_json(path, where=filename)
    return payload


def _post_p1_plan() -> dict[str, Any]:
    plan = post_p1.build_plan()
    if plan.get("schema_version") != POST_P1_PLAN_SCHEMA:
        raise CoordinatorError("local post-P1 v5 implementation drift")
    return plan


def canonical_geometry_dose_authority(
    *, dose_receipt_sha256: str, dose_replay_sha256: str
) -> dict[str, Any]:
    """Describe a quarantined legacy claim; never issue current work from it."""

    plan = _post_p1_plan()
    adjudication = plan["dose_adjudication"]
    return {
        "schema_version": LEGACY_GEOMETRY_DOSE_AUTHORITY_SCHEMA,
        "source_plan_schema": POST_P1_PLAN_SCHEMA,
        "source_plan_sha256": _digest(plan),
        "dose_adjudication_sha256": _digest(adjudication),
        "dose_receipt_sha256": _require_sha(dose_receipt_sha256, "dose_receipt_sha256"),
        "dose_replay_sha256": _require_sha(dose_replay_sha256, "dose_replay_sha256"),
        "replay_verified": False,
        "historical_only": True,
        "current_central_authority": False,
        "sample_dose": SHORT_SAMPLE_DOSE,
        "optimizer_steps": SHORT_OPTIMIZER_STEPS,
        "world_size": WORLD_SIZE,
        "local_batch_size": LOCAL_BATCH_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "grad_accum_steps": 1,
        "amp": "none",
    }


def _current_geometry_dose_authority(
    p1_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive current dose authority from the centrally selected measured sample."""

    sample = p1_authority["p1_sample_evidence_receipt"]
    return {
        "schema_version": GEOMETRY_DOSE_AUTHORITY_SCHEMA,
        "source_p1_recipe_data_authority_sha256": _digest(p1_authority),
        "source_p1_sample_state_sha256": sample["state_sha256"],
        "source_p1_sample_order_sha256": sample["sample_order_sha256"],
        "source_p1_rows_file_sha256": sample["rows_file_sha256"],
        "source_effective_recipe_sha256": p1_authority["effective_recipe_sha256"],
        "sample_dose": SHORT_SAMPLE_DOSE,
        "optimizer_steps": SHORT_OPTIMIZER_STEPS,
        "world_size": WORLD_SIZE,
        "local_batch_size": LOCAL_BATCH_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "grad_accum_steps": 1,
        "amp": "none",
        "measured_sample_replay_verified": True,
        "current_central_authority": True,
    }


def _verify_geometry_dose_authority(
    value: Any, *, p1_authority: Mapping[str, Any]
) -> dict[str, Any]:
    expected = _current_geometry_dose_authority(p1_authority)
    authority = _require_exact_keys(value, set(expected), "geometry/dose authority")
    if authority != expected:
        raise CoordinatorError(
            "geometry/dose authority is not derived from central P1 sample evidence"
        )
    return copy.deepcopy(authority)


def _verify_composite(value: Any) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "component_ids",
        "component_sampling_ratios",
        "descriptor_sha256",
        "data_fingerprint",
        "payload_inventory_sha256",
        "production_sampling_receipt_sha256",
        "validation_split_receipt_sha256",
        "sampler_identity_sha256",
        "sample_order_sha256",
        "training_game_seed_set_sha256",
        "validation_game_seed_set_sha256",
        "truncation_surface_sha256",
        "truncated_rows",
        "complete_game_inputs",
    }
    composite = _require_exact_keys(value, expected_keys, "typed composite authority")
    if (
        composite["schema_version"] != "a1-typed-64-12-4-20-composite-v1"
        or composite["component_ids"] != list(COMPONENT_IDS)
        or composite["component_sampling_ratios"] != list(COMPONENT_RATIOS)
    ):
        raise CoordinatorError("P1 data is not the exact typed 64/12/4/20 composite")
    for key in expected_keys - {
        "schema_version",
        "component_ids",
        "component_sampling_ratios",
        "truncated_rows",
        "complete_game_inputs",
    }:
        _require_sha(composite[key], f"typed composite {key}")
    if (
        isinstance(composite["truncated_rows"], bool)
        or not isinstance(composite["truncated_rows"], int)
        or composite["truncated_rows"] < 0
        or composite["complete_game_inputs"] is not True
    ):
        raise CoordinatorError("typed composite truncation surface is unproven")
    return copy.deepcopy(composite)


def _verify_sample_receipt_projection(
    value: Any,
    *,
    composite: Mapping[str, Any],
    sampler_seed: int,
    prior_required: bool,
) -> dict[str, Any]:
    """Verify a retained receipt already replayed by scientific evidence.

    This is deliberately only a cross-binding projection.  Payload scanning and
    exact sampler replay belong to :mod:`a1_scientific_evidence` and are consumed
    at issuance; the coordinator does not implement a second sampler.
    """

    receipt = _verify_sealed(value, "retained sample evidence receipt")
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "status",
            "sample_dose",
            "sampler_seed",
            "sampler_algorithm",
            "sampler_identity_sha256",
            "sample_order_sha256",
            "row_set_sha256",
            "unique_row_count",
            "prior_rows_file_sha256",
            "prior_row_set_sha256",
            "prior_unique_row_count",
            "observed_unique_overlap_count",
            "analytic_expected_unique_overlap_decimal",
            "overlap_excess_bound_decimal",
            "overlap_alpha_decimal",
            "overlap_within_independent_bound",
            "component_overlap",
            "kl_eligible_rows",
            "kl_eligible_mass_decimal",
            "kl_ordered_evidence_sha256",
            "kl_eligible_evidence_sha256",
            "descriptor_sha256",
            "payload_inventory_sha256",
            "rows_file_sha256",
            "origin_tool_sha256",
            "replay_verified",
            "state_sha256",
        },
        "retained sample evidence receipt",
    )
    eligible = receipt["kl_eligible_rows"]
    expected_mass = format(eligible / SHORT_SAMPLE_DOSE, ".12f").rstrip("0").rstrip(
        "."
    )
    if (
        receipt["schema_version"] != "a1-authenticated-sample-evidence-v1"
        or receipt["status"] != "complete"
        or receipt["sample_dose"] != SHORT_SAMPLE_DOSE
        or receipt["sampler_seed"] != sampler_seed
        or receipt["sampler_algorithm"]
        != scientific_evidence.SAMPLER_ALGORITHM
        or type(eligible) is not int
        or not 0 < eligible <= SHORT_SAMPLE_DOSE
        or receipt["kl_eligible_mass_decimal"] != expected_mass
        or receipt["descriptor_sha256"] != composite["descriptor_sha256"]
        or receipt["payload_inventory_sha256"]
        != composite["payload_inventory_sha256"]
        or receipt["origin_tool_sha256"]
        != _repo_tool_sha256("tools/a1_scientific_evidence.py")
        or receipt["replay_verified"] is not True
        or receipt["overlap_within_independent_bound"] is not True
        or not isinstance(receipt["component_overlap"], dict)
        or set(receipt["component_overlap"]) != set(COMPONENT_IDS)
    ):
        raise CoordinatorError("sample evidence receipt semantic/corpus drift")
    prior_fields = (
        receipt["prior_rows_file_sha256"],
        receipt["prior_row_set_sha256"],
    )
    if prior_required:
        if (
            any(value is None for value in prior_fields)
            or receipt["prior_unique_row_count"] <= 0
        ):
            raise CoordinatorError("FINAL sample lacks its retained P1 prior")
    elif (
        prior_fields != (None, None)
        or receipt["prior_unique_row_count"] != 0
        or receipt["observed_unique_overlap_count"] != 0
    ):
        raise CoordinatorError("P1 sample unexpectedly names a prior sample")
    for field in (
        "sampler_identity_sha256",
        "sample_order_sha256",
        "row_set_sha256",
        "kl_ordered_evidence_sha256",
        "kl_eligible_evidence_sha256",
        "descriptor_sha256",
        "payload_inventory_sha256",
        "rows_file_sha256",
        "origin_tool_sha256",
        "state_sha256",
    ):
        _require_sha(receipt[field], f"sample evidence {field}")
    for index, value in enumerate(prior_fields):
        if value is not None:
            _require_sha(value, f"sample evidence prior field {index}")
    return copy.deepcopy(receipt)


def _kl_authority_from_verified_sample(
    sample: Mapping[str, Any], *, composite: Mapping[str, Any]
) -> dict[str, Any]:
    """Project trainer KL-mask facts from one exact replayed draw order."""

    sampled = int(sample["sample_dose"])
    eligible = int(sample["kl_eligible_rows"])
    core = {
        "schema_version": "a1-p1-kl-eligibility-authority-v1",
        "sampled_rows": sampled,
        "eligible_rows": eligible,
        "eligible_mass_decimal": _canonical_decimal(
            Decimal(eligible) / Decimal(sampled)
        ),
        "descriptor_sha256": composite["descriptor_sha256"],
        "payload_inventory_sha256": composite["payload_inventory_sha256"],
        "sampler_identity_sha256": sample["sampler_identity_sha256"],
        "sample_order_sha256": sample["sample_order_sha256"],
        "ordered_evidence_sha256": sample["kl_ordered_evidence_sha256"],
        "eligible_evidence_sha256": sample["kl_eligible_evidence_sha256"],
        "scope": "authenticated_historical_replay",
        "prior_policy_required": True,
        "multi_action_required": True,
    }
    receipt_sha = _digest(core)
    return {
        **core,
        "receipt_sha256": receipt_sha,
        "replay_sha256": _digest(
            {
                "receipt_sha256": receipt_sha,
                "source_sample_state_sha256": sample["state_sha256"],
                "replay": "scientific_evidence_exact_order_v1",
            }
        ),
        "replay_verified": True,
    }


def verify_p1_recipe_data_authority(value: Any) -> dict[str, Any]:
    """Verify current P1 recipe/data authority, independent of post-P1 v5."""

    authority = _require_exact_keys(
        value,
        {
            "schema_version",
            "sweep_id",
            "selected_arm",
            "central_authority",
            "selection_receipt_sha256",
            "selection_replay_sha256",
            "replay_verified",
            "scientific_role",
            "diagnostic_only",
            "promotion_eligible",
            "requires_independent_final_replication",
            "effective_recipe",
            "effective_recipe_sha256",
            "composite",
            "recovery_authority",
            "current_parent_authority",
            "native_runtime_authority",
            "native_learner_admission_receipt",
            "p1_sample_evidence_receipt",
            "recovery_component_semantics",
        },
        "P1 recipe/data authority",
    )
    if authority["schema_version"] != P1_RECIPE_DATA_AUTHORITY_SCHEMA:
        if authority["schema_version"] == STALE_RECOVERY_PLAN_SCHEMA:
            raise CoordinatorError(
                "stale learner recovery plan cannot select current P1 recipe/data"
            )
        raise CoordinatorError("P1 recipe/data authority schema drift")
    _require_sha(authority["sweep_id"], "P1 central sweep id")
    if (
        authority["selected_arm"] not in P1_ARMS
        or authority["central_authority"] is not True
        or authority["scientific_role"] != "diagnostic_recipe_selection"
        or authority["diagnostic_only"] is not True
        or authority["promotion_eligible"] is not False
        or authority["requires_independent_final_replication"] is not True
    ):
        raise CoordinatorError("P1 recipe/data authority was not centrally selected")
    _require_sha(authority["selection_receipt_sha256"], "P1 selection receipt")
    _require_sha(authority["selection_replay_sha256"], "P1 selection replay")
    if authority["replay_verified"] is not True:
        raise CoordinatorError("P1 selection was not canonically replayed")
    recipe = authority["effective_recipe"]
    if not isinstance(recipe, dict) or not recipe:
        raise CoordinatorError("P1 selected recipe is empty")
    if authority["effective_recipe_sha256"] != _digest(recipe):
        raise CoordinatorError("P1 selected recipe digest drift")
    if recipe.get("truncated_vp_margin_value_weight") != 0.25:
        raise CoordinatorError("P1 recipe lost the production truncated-VP weight")
    composite = _verify_composite(authority["composite"])
    recovery = authority["recovery_authority"]
    parent = verify_current_parent_authority(
        authority["current_parent_authority"], recovery_authority=recovery
    )
    native_receipt = authority["native_learner_admission_receipt"]
    native = authority["native_runtime_authority"]
    if native != _native_runtime_authority_from_verified_receipt(native_receipt):
        raise CoordinatorError("retained native runtime projection drift")
    sample = _verify_sample_receipt_projection(
        authority["p1_sample_evidence_receipt"],
        composite=composite,
        sampler_seed=P1_SAMPLER_SEED,
        prior_required=False,
    )
    if (
        recovery.get("authority_sha256") != parent["recovery_authority_sha256"]
        or authority["recovery_component_semantics"]
        != recovery_component_semantics(recovery)
        or native["learner_admission_state_sha256"] != native_receipt.get("state_sha256")
        or sample["sampler_identity_sha256"] != composite["sampler_identity_sha256"]
        or sample["sample_order_sha256"] != composite["sample_order_sha256"]
    ):
        raise CoordinatorError("P1 retained recovery/runtime/sample authority drift")
    return copy.deepcopy(authority)


def verify_v5_recovery_receipt(path: Path) -> dict[str, Any]:
    """Consume the canonical disaster-recovery verifier, never old claims."""

    try:
        verified = v5_recovery.verify_committed_receipt(path)
    except (v5_recovery.RecoveryError, OSError, ValueError) as error:
        raise CoordinatorError(f"v5 recovery receipt refused: {error}") from error
    authority = verified.get("authority") if isinstance(verified, dict) else None
    if not isinstance(authority, dict):
        raise CoordinatorError("v5 recovery verifier returned no authority")
    if (
        authority.get("schema_version")
        != "a1-v5-disaster-recovery-authority-v1"
        or authority.get("promotion_proof_recreated") is not False
        or authority.get("dual_baseline_fresh_gate_required") is not True
        or authority.get("promotion_eligible") is not False
        or authority.get("training_proof") is not False
        or authority.get("wave_lineage_mode") != "recovery_reference"
    ):
        raise CoordinatorError("v5 recovery authority weakened quarantine policy")
    _require_sha(authority.get("authority_sha256"), "v5 recovery authority")
    return copy.deepcopy(authority)


def current_parent_authority_from_recovery(
    recovery_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the exact recovered generator without inventing promotion proof."""

    recovered = recovery_authority["recovered_generator"]
    receipt = recovery_authority["recovery_receipt"]
    return {
        "schema_version": HANDOFF_PARENT_AUTHORITY_SCHEMA,
        "role": "exact_recovered_generator_parent_not_p1_candidate",
        "checkpoint_path": recovered["path"],
        "checkpoint_sha256": recovered["sha256"],
        "checkpoint_md5": recovered["md5"],
        "historical_generation_version_claim": recovered[
            "historical_generation_version_claim"
        ],
        "recovery_authority_sha256": recovery_authority["authority_sha256"],
        "recovery_lineage_id": recovery_authority["recovery_lineage_id"],
        "recovery_receipt_file_sha256": receipt["sha256"],
        "recovery_receipt_semantic_sha256": receipt[
            "recovery_receipt_sha256"
        ],
        "promotion_proof_recreated": False,
        "independent_reload_per_arm": True,
        "unpromoted_candidate_forbidden": True,
    }


def recovery_component_semantics(
    recovery_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Disambiguate storage component IDs from unproven recovery lineage."""

    recovered = recovery_authority["recovered_generator"]
    safety = recovery_authority["safety_reference_unproven_predecessor"]
    return {
        "schema_version": "a1-recovery-component-semantics-v1",
        "storage_id_to_semantic_role": {
            "current_producer": "recovery_reference",
            "recent_history": "safety_reference_unproven_predecessor",
            "hard_negative": "hard_negative",
            "historical_replay": "historical_replay",
        },
        "current_producer_checkpoint_sha256": recovered["sha256"],
        "recent_history_checkpoint_sha256": safety["sha256"],
        "recent_history_causal_parent_proven": False,
        "recovery_authority_sha256": recovery_authority["authority_sha256"],
    }


def verify_current_parent_authority(
    value: Any, *, recovery_authority: Mapping[str, Any]
) -> dict[str, Any]:
    expected = current_parent_authority_from_recovery(recovery_authority)
    authority = _require_exact_keys(
        value, set(expected), "current handoff parent authority"
    )
    if authority != expected:
        raise CoordinatorError(
            "initializer authority is not the exact recovered generator reference"
        )
    return copy.deepcopy(authority)


def verify_pointer_upgrade_authority(
    value: Any, *, expected_parent_sha256: str
) -> dict[str, Any]:
    authority = _require_exact_keys(
        value,
        {
            "schema_version",
            "module",
            "source_checkpoint_sha256",
            "upgraded_initializer_sha256",
            "receipt_sha256",
            "receipt_replay_sha256",
            "flags",
            "new_parameter_set_sha256",
            "main_output_max_diff",
            "shared_parameters_bit_identical",
        },
        "pointer upgrade authority",
    )
    if (
        authority["schema_version"] != POINTER_UPGRADE_AUTHORITY_SCHEMA
        or authority["module"] != POINTER_MODULE
        or authority["source_checkpoint_sha256"]
        != expected_parent_sha256
        or authority["flags"] != POINTER_FLAGS
        or authority["main_output_max_diff"] != 0.0
        or authority["shared_parameters_bit_identical"] is not True
    ):
        raise CoordinatorError("legacy/aliased auxiliary upgrade is not admissible")
    for field in (
        "upgraded_initializer_sha256",
        "receipt_sha256",
        "receipt_replay_sha256",
        "new_parameter_set_sha256",
    ):
        _require_sha(authority[field], f"pointer upgrade {field}")
    return copy.deepcopy(authority)


def verify_warmup_recipe(value: Any) -> dict[str, Any]:
    recipe = _require_exact_keys(
        value,
        {
            "schema_version",
            "sample_dose",
            "optimizer_steps",
            "max_steps",
            "world_size",
            "local_batch_size",
            "global_batch_size",
            "grad_accum_steps",
            "amp",
            "optimizer",
            "resume_optimizer",
            "optimizer_betas",
            "optimizer_eps",
            "weight_decay",
            "fused_optimizer",
            "lr",
            "lr_warmup_steps",
            "lr_schedule",
            "seed",
            "training_rng_rank_offset",
            "effective_torch_seeds",
            "max_grad_norm",
            "gradient_clipping",
            "head_only",
            "inherited_parameters_frozen",
            "trainable_prefixes",
            "aux_subgoal_loss_weight",
            "main_objective_coefficients",
            "descriptor_sha256",
            "data_fingerprint",
            "payload_inventory_sha256",
            "validation_split_receipt_sha256",
            "sampler_identity_sha256",
            "sample_order_sha256",
            "sampler_seed",
            "target_scope",
            "target_scope_sha256",
            "target_version",
            "checkpoint_selection",
        },
        "pointer warmup recipe",
    )
    steps = recipe["optimizer_steps"]
    dose = recipe["sample_dose"]
    selection = recipe["checkpoint_selection"]
    if (
        recipe["schema_version"] != WARMUP_RECIPE_SCHEMA
        or isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps != SHORT_OPTIMIZER_STEPS
        or isinstance(dose, bool)
        or not isinstance(dose, int)
        or dose != SHORT_SAMPLE_DOSE
        or recipe["max_steps"] != SHORT_OPTIMIZER_STEPS
        or recipe["world_size"] != WORLD_SIZE
        or recipe["local_batch_size"] != LOCAL_BATCH_SIZE
        or recipe["global_batch_size"] != GLOBAL_BATCH_SIZE
        or recipe["grad_accum_steps"] != 1
        or recipe["amp"] != "none"
        or recipe["optimizer"] != "fresh_adam"
        or recipe["resume_optimizer"] is not False
        or recipe["optimizer_betas"] != [0.9, 0.999]
        or recipe["optimizer_eps"] != 1.0e-8
        or recipe["weight_decay"] != 0.0
        or recipe["fused_optimizer"] is not False
        or recipe["lr"] != 3.0e-5
        or recipe["lr_warmup_steps"] != 100
        or recipe["lr_schedule"] != "flat"
        or recipe["seed"] != 1
        or recipe["training_rng_rank_offset"] is not True
        or recipe["effective_torch_seeds"]
        != [recipe["seed"] + rank for rank in range(WORLD_SIZE)]
        or recipe["max_grad_norm"] != 1.0
        or recipe["gradient_clipping"] is not True
        or recipe["head_only"] is not True
        or recipe["inherited_parameters_frozen"] is not True
        or recipe["trainable_prefixes"] != list(POINTER_TRAINABLE_PREFIXES)
        or recipe["aux_subgoal_loss_weight"] != 1.0
        or recipe["main_objective_coefficients"] != WARMUP_MAIN_OBJECTIVE_ZERO
        or isinstance(recipe["sampler_seed"], bool)
        or not isinstance(recipe["sampler_seed"], int)
        or recipe["target_scope"] != "authenticated_aux_component_scope"
        or recipe["target_scope_sha256"]
        != _digest(
            {
                "scope": "authenticated_aux_component_scope",
                "module": POINTER_MODULE,
                "target_version": 1,
            }
        )
        or recipe["target_version"] != 1
        or selection
        != {
            "rule": "fixed_terminal_step",
            "optimizer_step": steps,
            "adaptive_best_checkpoint": False,
        }
    ):
        raise CoordinatorError("pointer warmup is not the preregistered head-only dose")
    for field in (
        "descriptor_sha256",
        "data_fingerprint",
        "payload_inventory_sha256",
        "validation_split_receipt_sha256",
        "sampler_identity_sha256",
        "sample_order_sha256",
        "target_scope_sha256",
    ):
        _require_sha(recipe[field], f"pointer warmup {field}")
    return copy.deepcopy(recipe)


def _decimal(value: Any, where: str) -> Decimal:
    if not isinstance(value, str):
        raise CoordinatorError(f"{where} must be a canonical decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise CoordinatorError(f"{where} is not a decimal") from error
    if not result.is_finite():
        raise CoordinatorError(f"{where} must be finite")
    return result


def _canonical_decimal(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def verify_selector_rule(value: Any) -> dict[str, Any]:
    rule = _require_exact_keys(
        value,
        {
            "schema_version",
            "formula",
            "maximum_aux_to_main_ratio_decimal",
            "maximum_opposing_projection_decimal",
            "maximum_coefficient_decimal",
            "minimum_coefficient_decimal",
            "quantum_decimal",
            "rounding",
            "out_of_range",
            "probe_manifest_sha256",
            "probe_sampler_seed",
            "probe_row_order_sha256",
            "probe_batches",
            "probe_batch_size",
            "shared_parameter_surface",
            "shared_parameter_set_sha256",
            "same_forward_graph",
            "global_ddp_aggregation",
            "ddp_reduction",
            "cross_batch_aggregation",
            "no_optimizer_step",
            "no_persistent_mutation",
        },
        "gradient selector rule",
    )
    ratio_cap = _decimal(
        rule["maximum_aux_to_main_ratio_decimal"], "selector ratio cap"
    )
    opposing_cap = _decimal(
        rule["maximum_opposing_projection_decimal"],
        "selector opposing projection cap",
    )
    minimum = _decimal(rule["minimum_coefficient_decimal"], "selector minimum")
    maximum = _decimal(rule["maximum_coefficient_decimal"], "selector maximum")
    quantum = _decimal(rule["quantum_decimal"], "selector quantum")
    if (
        rule["schema_version"] != SELECTOR_RULE_SCHEMA
        or rule["formula"]
        != (
            "min(max_aux_to_main_ratio/r,"
            "max_opposing_projection/max(-r*cos,epsilon)_if_cos_negative,"
            "maximum_coefficient)"
        )
        or ratio_cap != Decimal("0.05")
        or opposing_cap != Decimal("0.01")
        or not Decimal("0") < minimum <= maximum
        or minimum != Decimal("0.001")
        or maximum != Decimal("0.05")
        or quantum != Decimal("0.001")
        or rule["rounding"] != "ROUND_DOWN"
        or rule["out_of_range"] != "refuse"
        or isinstance(rule["probe_sampler_seed"], bool)
        or not isinstance(rule["probe_sampler_seed"], int)
        or rule["probe_batches"] != 5
        or rule["probe_batch_size"] != 512
        or rule["shared_parameter_surface"] != "inherited_trunk_only"
        or rule["same_forward_graph"] is not True
        or rule["global_ddp_aggregation"] is not True
        or rule["ddp_reduction"] != "global_numerator_and_denominator_sum"
        or rule["cross_batch_aggregation"] != "concatenated_batch_gradient_geometry"
        or rule["no_optimizer_step"] is not True
        or rule["no_persistent_mutation"] is not True
    ):
        raise CoordinatorError("gradient selector is not preregistered/fail-closed")
    _require_sha(rule["probe_manifest_sha256"], "selector probe manifest")
    _require_sha(rule["probe_row_order_sha256"], "selector probe row order")
    _require_sha(rule["shared_parameter_set_sha256"], "selector parameter set")
    return copy.deepcopy(rule)


def verify_allocation(value: Any) -> dict[str, Any]:
    allocation = _require_exact_keys(
        value,
        {
            "schema_version",
            "host_id",
            "hostname",
            "machine_id",
            "ssh_host_key_sha256",
            "checkout_tree_sha256",
            "tool_sha256",
            "physical_gpu_indices",
            "gpu_names",
            "gpu_uuids",
            "pci_bus_ids",
        },
        "B200 allocation",
    )
    if (
        allocation["schema_version"] != ALLOCATION_SCHEMA
        or not isinstance(allocation["host_id"], str)
        or not allocation["host_id"]
        or not isinstance(allocation["hostname"], str)
        or not allocation["hostname"]
        or allocation["host_id"] != B200_LEARNER_HOST_ID
        or allocation["hostname"] != B200_LEARNER_HOSTNAME
        or allocation["machine_id"] != B200_LEARNER_MACHINE_ID
        or allocation["physical_gpu_indices"] != list(range(WORLD_SIZE))
        or not isinstance(allocation["gpu_names"], list)
        or len(allocation["gpu_names"]) != WORLD_SIZE
        or any("B200" not in str(name).upper() for name in allocation["gpu_names"])
        or not isinstance(allocation["gpu_uuids"], list)
        or len(allocation["gpu_uuids"]) != WORLD_SIZE
        or len(set(allocation["gpu_uuids"])) != WORLD_SIZE
        or allocation["gpu_uuids"] != list(B200_LEARNER_GPU_UUIDS)
        or not all(
            isinstance(value, str) and value.startswith("GPU-")
            for value in allocation["gpu_uuids"]
        )
        or not isinstance(allocation["pci_bus_ids"], list)
        or len(allocation["pci_bus_ids"]) != WORLD_SIZE
        or len(set(allocation["pci_bus_ids"])) != WORLD_SIZE
    ):
        raise CoordinatorError("allocation is not one exact 8xB200 host")
    for field in ("ssh_host_key_sha256", "checkout_tree_sha256", "tool_sha256"):
        _require_sha(allocation[field], f"allocation {field}")
    return copy.deepcopy(allocation)


def _verify_p1_scheduled_allocations(
    allocations: Mapping[str, Mapping[str, Any]],
) -> None:
    if not (allocations["K0"] == allocations["K3"] == allocations["K10"]):
        raise CoordinatorError(
            "P1 K0/K3/K10 must strictly sequentially reuse the sole 8xB200 learner"
        )


def _verify_aux_scheduled_allocations(
    allocations: Mapping[str, Mapping[str, Any]],
) -> None:
    """Strictly serialize all AUX stages on the sole 8xB200 learner."""

    if not all(allocations[stage] == allocations["WARMUP"] for stage in STAGES):
        raise CoordinatorError(
            "all AUX stages must strictly sequentially reuse the sole 8xB200 learner"
        )


def _verify_allocation_matches_native_report(
    allocation: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    """Require launch allocation bytes to match the admitted physical host."""

    expected = {
        "host_id": report.get("host_id"),
        "hostname": report.get("hostname"),
        "machine_id": report.get("machine_id"),
        "ssh_host_key_sha256": report.get("ssh_host_key_sha256"),
        "checkout_tree_sha256": report.get("checkout_tree_sha256"),
        "tool_sha256": report.get("tool_sha256"),
        "physical_gpu_indices": report.get("gpu_indices"),
        "gpu_names": report.get("gpu_names"),
        "gpu_uuids": report.get("gpu_uuids"),
        "pci_bus_ids": report.get("pci_bus_ids"),
    }
    observed = {field: allocation.get(field) for field in expected}
    if observed != expected:
        raise CoordinatorError(
            f"allocation does not match native admission for {allocation.get('host_id')}"
        )


def _native_runtime_authority_from_verified_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Project runtime authority only after the scientific verifier succeeded."""

    reports = receipt["hosts"]
    release = production_executor._native_wheel_release_identity()
    return {
        "schema_version": NATIVE_RUNTIME_AUTHORITY_SCHEMA,
        "distribution": "catanatron-rs",
        "version": release["version"],
        "wheel_filename": release["filename"],
        "wheel_sha256": release["sha256"],
        "artifact_inventory_sha256": production_executor._sha256(
            production_executor.NATIVE_WHEEL_INVENTORY
        ),
        "release_identity_sha256": _digest(release),
        "learner_admission_state_sha256": receipt["state_sha256"],
        "host_runtime_identity_sha256": _digest(reports),
        "admitted_hosts": sorted(reports),
        "capabilities": release["required_capabilities"],
        "replay_verified": True,
    }


def _consume_runtime_admission_receipt(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay the sole measured B200 admission and return its projection."""

    origin = _repo_tool_sha256("tools/a1_scientific_evidence.py")
    try:
        receipt = scientific_evidence.verify_runtime_admission_receipt(
            path.expanduser().resolve(strict=True),
            expected_origin_tool_sha256=origin,
        )
    except (scientific_evidence.EvidenceError, OSError, ValueError) as error:
        raise CoordinatorError(f"B200 runtime admission refused: {error}") from error
    return copy.deepcopy(receipt), _native_runtime_authority_from_verified_receipt(
        receipt
    )


def build_native_runtime_authority(
    *, learner_admission_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive learner-native admission from a sealed read-only B200 probe."""

    receipt = _verify_sealed(dict(learner_admission_receipt), "B200 learner admission")
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "status",
            "hosts",
            "origin_tool_sha256",
            "state_sha256",
        },
        "B200 learner admission",
    )
    if (
        receipt["schema_version"] != "a1-b200-learner-runtime-admission-v1"
        or receipt["status"] != "complete"
        or receipt["origin_tool_sha256"]
        != _repo_tool_sha256("tools/a1_scientific_evidence.py")
    ):
        raise CoordinatorError(
            "native learner admission receipt schema/status/producer drift"
        )
    _require_sha(receipt["origin_tool_sha256"], "native learner admission tool")
    reports = receipt["hosts"]
    if not isinstance(reports, dict) or len(reports) != 1:
        raise CoordinatorError(
            "native learner admission must contain the sole 8xB200 learner"
        )
    release = production_executor._native_wheel_release_identity()
    runtime = production_executor.PRODUCTION_RUNTIME
    if (
        release.get("version") != runtime["catanatron_rs_version"]
        or release.get("filename") != runtime["catanatron_rs_wheel_filename"]
        or release.get("sha256") != "sha256:" + runtime["catanatron_rs_wheel_sha256"]
        or release.get("required_capabilities")
        != sorted(production_executor.NATIVE_REQUIRED_CAPABILITIES)
    ):
        raise CoordinatorError("local native wheel release inventory drift")
    expected_report_keys = {
        "host_id",
        "hostname",
        "machine_id",
        "ssh_host_key_sha256",
        "checkout_tree_sha256",
        "tool_sha256",
        "gpu_indices",
        "gpu_names",
        "gpu_uuids",
        "pci_bus_ids",
        "python",
        "torch_version",
        "torch_cuda_version",
        "catanatron_rs_version",
        "native_wheel_sha256",
        "native_mcts_capabilities",
        "nofile_soft",
        "nofile_hard",
    }
    for alias, raw in sorted(reports.items()):
        if not isinstance(alias, str) or not alias:
            raise CoordinatorError("native preflight host alias drift")
        report = _require_exact_keys(
            raw, expected_report_keys, f"native preflight {alias}"
        )
        capabilities = report["native_mcts_capabilities"]
        if (
            report["host_id"] != alias
            or alias != B200_LEARNER_HOST_ID
            or not isinstance(report["hostname"], str)
            or report["hostname"] != B200_LEARNER_HOSTNAME
            or report["machine_id"] != B200_LEARNER_MACHINE_ID
            or report["tool_sha256"] != receipt["origin_tool_sha256"]
            or report["gpu_indices"] != list(range(WORLD_SIZE))
            or not isinstance(report["gpu_names"], list)
            or len(report["gpu_names"]) != WORLD_SIZE
            or any("B200" not in str(name).upper() for name in report["gpu_names"])
            or not isinstance(report["gpu_uuids"], list)
            or len(report["gpu_uuids"]) != WORLD_SIZE
            or len(set(report["gpu_uuids"])) != WORLD_SIZE
            or report["gpu_uuids"] != list(B200_LEARNER_GPU_UUIDS)
            or not isinstance(report["pci_bus_ids"], list)
            or len(report["pci_bus_ids"]) != WORLD_SIZE
            or len(set(report["pci_bus_ids"])) != WORLD_SIZE
            or report["catanatron_rs_version"] != release["version"]
            or report["native_wheel_sha256"] != release["sha256"]
            or not isinstance(capabilities, list)
            or any(not isinstance(item, str) for item in capabilities)
            or not set(NATIVE_CAPABILITIES) <= set(capabilities)
            or any(
                type(report[field]) is not int
                for field in ("nofile_soft", "nofile_hard")
            )
            or report["nofile_soft"] < 65_536
            or report["nofile_hard"] < report["nofile_soft"]
            or report["python"] != runtime["python_version"]
            or report["torch_version"] != runtime["torch_version"]
            or report["torch_cuda_version"] != runtime["torch_cuda_version"]
        ):
            raise CoordinatorError(f"native preflight evidence drift on {alias}")
        for field in ("ssh_host_key_sha256", "checkout_tree_sha256", "tool_sha256"):
            _require_sha(report[field], f"native preflight {alias} {field}")
    inventory_sha = production_executor._sha256(
        production_executor.NATIVE_WHEEL_INVENTORY
    )
    preflight_sha = _digest(reports)
    return {
        "schema_version": NATIVE_RUNTIME_AUTHORITY_SCHEMA,
        "distribution": "catanatron-rs",
        "version": release["version"],
        "wheel_filename": release["filename"],
        "wheel_sha256": release["sha256"],
        "artifact_inventory_sha256": inventory_sha,
        "release_identity_sha256": _digest(release),
        "learner_admission_state_sha256": receipt["state_sha256"],
        "host_runtime_identity_sha256": preflight_sha,
        "admitted_hosts": sorted(reports),
        "capabilities": release["required_capabilities"],
        "replay_verified": True,
    }


def verify_native_runtime_authority(
    value: Any, *, learner_admission_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    authority = _require_exact_keys(
        value,
        {
            "schema_version",
            "distribution",
            "version",
            "wheel_filename",
            "wheel_sha256",
            "artifact_inventory_sha256",
            "release_identity_sha256",
            "learner_admission_state_sha256",
            "host_runtime_identity_sha256",
            "admitted_hosts",
            "capabilities",
            "replay_verified",
        },
        "native runtime authority",
    )
    release = production_executor._native_wheel_release_identity()
    if (
        authority["schema_version"] != NATIVE_RUNTIME_AUTHORITY_SCHEMA
        or authority["distribution"] != "catanatron-rs"
        or authority["version"] != release["version"]
        or authority["wheel_filename"] != release["filename"]
        or authority["wheel_sha256"] != release["sha256"]
        or authority["capabilities"] != release["required_capabilities"]
        or authority["replay_verified"] is not True
    ):
        raise CoordinatorError("native runtime is not canonical catanatron-rs 0.1.8")
    for field in (
        "artifact_inventory_sha256",
        "release_identity_sha256",
        "learner_admission_state_sha256",
        "host_runtime_identity_sha256",
    ):
        _require_sha(authority[field], f"native runtime {field}")
    replayed = build_native_runtime_authority(
        learner_admission_receipt=learner_admission_receipt
    )
    if replayed != authority:
        raise CoordinatorError("native runtime authority failed executor replay")
    return copy.deepcopy(authority)


def _verify_execution(value: Any) -> dict[str, Any]:
    execution = _require_exact_keys(
        value,
        {
            "schema_version",
            "command_sha256",
            "environment_sha256",
            "output_namespace_sha256",
        },
        "stage execution binding",
    )
    if execution["schema_version"] != EXECUTION_SCHEMA:
        raise CoordinatorError("stage execution schema drift")
    for field in ("command_sha256", "environment_sha256", "output_namespace_sha256"):
        _require_sha(execution[field], f"stage execution {field}")
    return copy.deepcopy(execution)


def canonical_p1_final_lock_authority() -> dict[str, Any]:
    """Build the only P1 base recipe from the repo's current v3 source."""

    draft_path = (
        pre_wave.REPO_ROOT / "configs/experiments/a1_pre_wave_contract.template.json"
    )
    recipe = copy.deepcopy(pre_wave.CURRENT_LEARNER_TRAINING_RECIPE)
    # The prior recipe identified the sampler only by a digest, which is not
    # executable.  Freeze the seed that produced the authenticated P1 order.
    recipe["sampler_seed"] = P1_SAMPLER_SEED
    projection = {
        "source_draft_schema": "a1-pre-wave-contract-draft-v3",
        "lock_schema": "a1-pre-wave-contract-lock-v3",
        "source_draft_file_sha256": production_executor._sha256(draft_path),
        "canonical_recipe_source_sha256": _digest(
            pre_wave.CURRENT_LEARNER_TRAINING_RECIPE
        ),
        "base_recipe": recipe,
        "base_recipe_sha256": _digest(recipe),
    }
    return {
        "schema_version": P1_FINAL_LOCK_AUTHORITY_SCHEMA,
        **projection,
        "lock_projection_sha256": _digest(projection),
        "replay_sha256": _digest(
            {"lock_projection_sha256": _digest(projection), "replay": "v3"}
        ),
        "replay_verified": True,
    }


def verify_p1_final_lock_authority(value: Any) -> dict[str, Any]:
    """Verify the final v3 source-draft/lock replay selected-dose control.

    The coordinator never patches an old BF16/max-steps-zero recipe.  The final
    source draft itself must seal these values so K0/K3/K10 differ only in KL
    weight.
    """

    expected = canonical_p1_final_lock_authority()
    authority = _require_exact_keys(value, set(expected), "P1 final v3 lock authority")
    if (
        authority["schema_version"] != P1_FINAL_LOCK_AUTHORITY_SCHEMA
        or authority["source_draft_schema"] != "a1-pre-wave-contract-draft-v3"
        or authority["lock_schema"] != "a1-pre-wave-contract-lock-v3"
        or authority["replay_verified"] is not True
    ):
        raise CoordinatorError("P1 source draft/lock is not exact replayed final v3")
    for field in (
        "source_draft_file_sha256",
        "canonical_recipe_source_sha256",
        "lock_projection_sha256",
        "replay_sha256",
    ):
        _require_sha(authority[field], f"P1 final lock {field}")
    recipe = authority["base_recipe"]
    if not isinstance(recipe, dict) or authority["base_recipe_sha256"] != _digest(
        recipe
    ):
        raise CoordinatorError("P1 final lock base recipe digest drift")
    exact = {
        "amp": "none",
        "epochs": 1,
        "max_steps": SHORT_OPTIMIZER_STEPS,
        "world_size": 1,
        "batch_size": GLOBAL_BATCH_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "grad_accum_steps": 1,
        "resume_optimizer": False,
        "policy_kl_anchor_weight": 0.0,
        "truncated_vp_margin_value_weight": 0.25,
        "sampler_seed": P1_SAMPLER_SEED,
    }
    drift = {
        field: {"expected": expected, "actual": recipe.get(field)}
        for field, expected in exact.items()
        if recipe.get(field) != expected
    }
    if drift:
        raise CoordinatorError(
            f"P1 final lock did not seal selected FP32/128-step control: {drift}"
        )
    if authority != expected:
        raise CoordinatorError("P1 final v3 source/recipe replay drift")
    return copy.deepcopy(authority)


def verify_p1_kl_eligibility_authority(
    value: Any, *, composite: Mapping[str, Any]
) -> dict[str, Any]:
    authority = _require_exact_keys(
        value,
        {
            "schema_version",
            "sampled_rows",
            "eligible_rows",
            "eligible_mass_decimal",
            "descriptor_sha256",
            "payload_inventory_sha256",
            "sampler_identity_sha256",
            "sample_order_sha256",
            "ordered_evidence_sha256",
            "eligible_evidence_sha256",
            "scope",
            "prior_policy_required",
            "multi_action_required",
            "receipt_sha256",
            "replay_sha256",
            "replay_verified",
        },
        "P1 KL eligibility authority",
    )
    sampled = authority["sampled_rows"]
    eligible = authority["eligible_rows"]
    if (
        authority["schema_version"] != "a1-p1-kl-eligibility-authority-v1"
        or sampled != SHORT_SAMPLE_DOSE
        or isinstance(eligible, bool)
        or not isinstance(eligible, int)
        or not 0 < eligible <= sampled
        or authority["descriptor_sha256"] != composite["descriptor_sha256"]
        or authority["payload_inventory_sha256"]
        != composite["payload_inventory_sha256"]
        or authority["sampler_identity_sha256"] != composite["sampler_identity_sha256"]
        or authority["sample_order_sha256"] != composite["sample_order_sha256"]
        or authority["scope"] != "authenticated_historical_replay"
        or authority["prior_policy_required"] is not True
        or authority["multi_action_required"] is not True
        or authority["replay_verified"] is not True
    ):
        raise CoordinatorError("P1 KL eligibility is not the exact sampled-row surface")
    for field in (
        "sample_order_sha256",
        "ordered_evidence_sha256",
        "eligible_evidence_sha256",
        "receipt_sha256",
        "replay_sha256",
    ):
        _require_sha(authority[field], f"P1 KL eligibility {field}")
    expected_mass = Decimal(eligible) / Decimal(sampled)
    observed_mass = _decimal(authority["eligible_mass_decimal"], "eligible mass")
    if observed_mass != expected_mass:
        raise CoordinatorError("P1 KL eligible mass/count arithmetic drift")
    return copy.deepcopy(authority)


def _ordered_identity_update(
    digest: Any, *, index: int, row_identity_sha256: str
) -> None:
    digest.update(str(index).encode("ascii"))
    digest.update(b"\0")
    digest.update(row_identity_sha256.encode("ascii"))
    digest.update(b"\n")


def canonical_p1_row_identity(
    *,
    payload_member_sha256: str,
    row_offset: int,
    component_id: str,
    prior_policy_present: bool,
    legal_action_count: int,
) -> str:
    """Bind every KL-mask fact to the sampled row identity."""

    _require_sha(payload_member_sha256, "P1 payload member")
    if type(row_offset) is not int or row_offset < 0:
        raise CoordinatorError("P1 row offset drift")
    if component_id not in COMPONENT_IDS:
        raise CoordinatorError("P1 row component drift")
    if type(prior_policy_present) is not bool:
        raise CoordinatorError("P1 row prior flag drift")
    if type(legal_action_count) is not int or legal_action_count < 1:
        raise CoordinatorError("P1 row legal-action count drift")
    return _digest(
        {
            "schema_version": "a1-p1-kl-row-identity-v1",
            "payload_member_sha256": payload_member_sha256,
            "row_offset": row_offset,
            "component_id": component_id,
            "prior_policy_present": prior_policy_present,
            "legal_action_count": legal_action_count,
        }
    )


def canonical_p1_sample_order_sha256(
    row_identity_sha256s: Iterable[str],
) -> str:
    """Digest an exact 524,288-draw row order without materializing it."""

    digest = hashlib.sha256()
    count = 0
    for count, row_sha in enumerate(row_identity_sha256s, start=1):
        _require_sha(row_sha, f"P1 sampled row identity {count - 1}")
        _ordered_identity_update(digest, index=count - 1, row_identity_sha256=row_sha)
    if count != SHORT_SAMPLE_DOSE:
        raise CoordinatorError(
            f"P1 sampled order has {count} rows, expected {SHORT_SAMPLE_DOSE}"
        )
    return "sha256:" + digest.hexdigest()


def build_p1_kl_eligibility_authority(
    *,
    composite: Mapping[str, Any],
    sampled_row_evidence: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay the exact sampler draws and compute conditional KL eligibility.

    Each ordered row carries only the facts needed by the trainer's KL mask.
    The authority therefore cannot be manufactured from a count plus a
    ``replay_verified`` boolean: sweep issuance independently replays these
    rows and requires byte-identical derived authority.
    """

    data = _verify_composite(dict(composite))
    order_digest = hashlib.sha256()
    evidence_digest = hashlib.sha256()
    eligible_digest = hashlib.sha256()
    sampled = 0
    eligible = 0
    for sampled, raw in enumerate(sampled_row_evidence, start=1):
        index = sampled - 1
        row = _require_exact_keys(
            dict(raw),
            {
                "row_identity_sha256",
                "payload_member_sha256",
                "row_offset",
                "component_id",
                "prior_policy_present",
                "legal_action_count",
            },
            f"P1 KL sampled row {index}",
        )
        row_sha = _require_sha(
            row["row_identity_sha256"], f"P1 KL sampled row {index} identity"
        )
        recomputed_row_sha = canonical_p1_row_identity(
            payload_member_sha256=row["payload_member_sha256"],
            row_offset=row["row_offset"],
            component_id=row["component_id"],
            prior_policy_present=row["prior_policy_present"],
            legal_action_count=row["legal_action_count"],
        )
        if row_sha != recomputed_row_sha:
            raise CoordinatorError(
                f"P1 KL sampled row {index} identity does not bind mask facts"
            )
        if row["component_id"] not in COMPONENT_IDS:
            raise CoordinatorError(f"P1 KL sampled row {index} component drift")
        if type(row["prior_policy_present"]) is not bool:
            raise CoordinatorError(f"P1 KL sampled row {index} prior flag drift")
        if type(row["legal_action_count"]) is not int or row["legal_action_count"] < 1:
            raise CoordinatorError(f"P1 KL sampled row {index} legal count drift")
        _ordered_identity_update(order_digest, index=index, row_identity_sha256=row_sha)
        encoded = _canonical_bytes({"draw_index": index, **row}) + b"\n"
        evidence_digest.update(encoded)
        is_eligible = (
            row["component_id"] == "historical_replay"
            and row["prior_policy_present"]
            and row["legal_action_count"] > 1
        )
        if is_eligible:
            eligible += 1
            eligible_digest.update(encoded)
    if sampled != SHORT_SAMPLE_DOSE:
        raise CoordinatorError(
            f"P1 KL evidence has {sampled} rows, expected {SHORT_SAMPLE_DOSE}"
        )
    order_sha = "sha256:" + order_digest.hexdigest()
    if order_sha != data["sample_order_sha256"]:
        raise CoordinatorError("P1 KL evidence does not replay canonical sample order")
    if eligible == 0:
        raise CoordinatorError("P1 KL sampled order contains zero eligible rows")
    core = {
        "schema_version": "a1-p1-kl-eligibility-authority-v1",
        "sampled_rows": sampled,
        "eligible_rows": eligible,
        "eligible_mass_decimal": _canonical_decimal(
            Decimal(eligible) / Decimal(sampled)
        ),
        "descriptor_sha256": data["descriptor_sha256"],
        "payload_inventory_sha256": data["payload_inventory_sha256"],
        "sampler_identity_sha256": data["sampler_identity_sha256"],
        "sample_order_sha256": order_sha,
        "ordered_evidence_sha256": "sha256:" + evidence_digest.hexdigest(),
        "eligible_evidence_sha256": "sha256:" + eligible_digest.hexdigest(),
        "scope": "authenticated_historical_replay",
        "prior_policy_required": True,
        "multi_action_required": True,
    }
    receipt_sha = _digest(core)
    return {
        **core,
        "receipt_sha256": receipt_sha,
        "replay_sha256": _digest(
            {"receipt_sha256": receipt_sha, "replay": "exact_order_v1"}
        ),
        "replay_verified": True,
    }


def _canonical_evaluation_design_authority() -> dict[str, Any]:
    """Bind panel design to the committed diagnosis and evaluator sources."""

    repo = pre_wave.REPO_ROOT
    diagnosis_path = repo / "tools/a1_post_p1_diagnosis_plan.py"
    evaluator_path = repo / "tools/fleet/a1_h100_eval_fleet.py"
    operator_reference_path = (
        repo / "configs/operations/"
        "a1-r3-gather-aux64-reproduction-eval600-20260712-r1/README.md"
    )
    diagnosis_evaluation = post_p1.build_plan()["evaluation"]
    if (
        diagnosis_evaluation["internal"]
        != {
            "pairs": 300,
            "games": 600,
            "opponent": "f7 producer",
            "map_kind": "BASE",
            "seat_swapped": True,
            "common_random_numbers": True,
        }
        or diagnosis_evaluation["external"]["pairs"] != 250
        or diagnosis_evaluation["external"]["games_per_candidate"] != 500
        or diagnosis_evaluation["external"]["opponent"] != "catanatron_value"
        or diagnosis_evaluation["external"]["map_kind"] != "TOURNAMENT"
        or diagnosis_evaluation["external"]["common_random_numbers"] is not True
    ):
        raise CoordinatorError("committed post-P1 evaluation authority drift")
    authority = {
        "schema_version": "a1-fixed-evaluation-design-authority-v1",
        "diagnosis_plan_path": "tools/a1_post_p1_diagnosis_plan.py",
        "diagnosis_plan_file_sha256": production_executor._sha256(diagnosis_path),
        "diagnosis_evaluation_sha256": _digest(diagnosis_evaluation),
        "evaluator_path": "tools/fleet/a1_h100_eval_fleet.py",
        "evaluator_file_sha256": production_executor._sha256(evaluator_path),
        "operator_reference_path": (
            "configs/operations/"
            "a1-r3-gather-aux64-reproduction-eval600-20260712-r1/README.md"
        ),
        "operator_reference_file_sha256": production_executor._sha256(
            operator_reference_path
        ),
    }
    authority["authority_sha256"] = _digest(authority)
    return authority


def _canonical_search_operator() -> dict[str, Any]:
    """The exact matched search instrument documented by the repo authority."""

    return {
        "schema_version": "a1-fixed-matched-search-operator-v1",
        "engine": "native_rust_information_set_search",
        "n_full": 128,
        "particle_count": 4,
        "minimum_simulations_per_particle": 32,
        "d6_root_averaging": True,
        "d6_minimum_legal_width": 20,
        "candidate_c_scale": 0.10,
        "baseline_c_scale": 0.10,
        "c_visit": 50,
        "sigma_eval": 0.98,
        "value_readout": "scalar_tanh",
        "root_candidate_cap_narrow": 16,
        "root_candidate_cap_wide": 54,
        "root_candidate_cap_width_threshold": 24,
        "selection_tuning_allowed": False,
    }


def _canonical_cohort(
    *,
    family: str,
    panel_kind: str,
    base_seed: int,
    pairs: int,
    map_kind: str,
    opponent: str,
) -> dict[str, Any]:
    return {
        "schema_version": "a1-fixed-evaluation-cohort-v1",
        "family": family,
        "panel_kind": panel_kind,
        "base_seed": base_seed,
        "pairs": pairs,
        "games": pairs * 2,
        "seed_schedule": {
            "algorithm": "contiguous_pair_index_v1",
            "pair_seed_expression": "base_seed + pair_index",
            "pair_index_start": 0,
            "pair_index_stop_exclusive": pairs,
        },
        "orientation_schedule": ["candidate_first", "candidate_second"],
        "map_kind": map_kind,
        "opponent": opponent,
        "common_random_numbers": True,
        "seat_swapped": True,
    }


def canonical_p1_evaluation_plan(
    *, baseline_checkpoint_sha256: str
) -> dict[str, Any]:
    design = _canonical_evaluation_design_authority()
    search = _canonical_search_operator()
    internal = _canonical_cohort(
        family="P1",
        panel_kind="internal",
        base_seed=P1_INTERNAL_BASE_SEED,
        pairs=300,
        map_kind="BASE",
        opponent="recovered_generator_reference",
    )
    external = _canonical_cohort(
        family="P1",
        panel_kind="external",
        base_seed=P1_EXTERNAL_BASE_SEED,
        pairs=250,
        map_kind="TOURNAMENT",
        opponent="catanatron_value",
    )
    rule = {
        "schema_version": "a1-p1-deterministic-selection-rule-v1",
        "primary": "maximum_internal_pair_points",
        "external_gate": "candidate_mean_not_below_control_minus_tolerance",
        "external_non_regression_tolerance_milli": 25,
        "tie_break_order": list(P1_ARMS),
        "fallback": "K0",
    }
    return {
        "schema_version": P1_EVALUATION_PLAN_SCHEMA,
        "design_authority": design,
        "design_authority_sha256": design["authority_sha256"],
        "baseline_checkpoint_sha256": _require_sha(
            baseline_checkpoint_sha256, "P1 evaluation baseline"
        ),
        "internal_cohort": internal,
        "internal_cohort_sha256": _digest(internal),
        "external_cohort": external,
        "external_cohort_sha256": _digest(external),
        "search_operator": search,
        "search_operator_sha256": _digest(search),
        "decision_rule_sha256": _digest(rule),
        "decision_rule": rule,
        "internal_pairs_per_arm": internal["pairs"],
        "external_games_per_arm": external["games"],
        "panel_origin_tool_sha256": design["evaluator_file_sha256"],
        "common_random_numbers": True,
        "seat_swapped": True,
        "all_fixed_arms_required": True,
        "selection_source": "authenticated_fixed_evaluation_terminal",
    }


def verify_p1_evaluation_plan(
    value: Any, *, baseline_checkpoint_sha256: str
) -> dict[str, Any]:
    """Accept only the repo-derived fixed evaluation design."""

    expected = canonical_p1_evaluation_plan(
        baseline_checkpoint_sha256=baseline_checkpoint_sha256
    )
    plan = _require_exact_keys(value, set(expected), "P1 fixed evaluation plan")
    if plan != expected:
        raise CoordinatorError(
            "P1 evaluation is not the canonical current-parent-relative design"
        )
    return copy.deepcopy(plan)


def _p1_selected_arm_from_outcomes(
    plan: Mapping[str, Any],
    *,
    internal_pair_points_milli: Any,
    external_game_points_milli: Any,
) -> str:
    internal = _require_exact_keys(
        internal_pair_points_milli, set(P1_ARMS), "P1 internal outcomes"
    )
    external = _require_exact_keys(
        external_game_points_milli, set(P1_ARMS), "P1 external outcomes"
    )
    expected_internal = plan["internal_pairs_per_arm"]
    expected_external = plan["external_games_per_arm"]
    for label, values, expected, ceiling in (
        ("internal", internal, expected_internal, 2000),
        ("external", external, expected_external, 1000),
    ):
        for arm in P1_ARMS:
            points = values[arm]
            if (
                not isinstance(points, list)
                or len(points) != expected
                or any(
                    type(point) is not int or not 0 <= point <= ceiling
                    for point in points
                )
            ):
                raise CoordinatorError(f"P1 {label} outcomes drift for {arm}")
    tolerance = plan["decision_rule"]["external_non_regression_tolerance_milli"]
    control_external_sum = sum(external["K0"])
    eligible = [
        arm
        for arm in P1_ARMS
        if arm == "K0"
        or sum(external[arm]) >= control_external_sum - tolerance * expected_external
    ]
    return max(
        eligible,
        key=lambda arm: (sum(internal[arm]), -P1_ARMS.index(arm)),
    )


def canonical_aux_evaluation_plan(
    *, baseline_checkpoint_sha256: str
) -> dict[str, Any]:
    design = _canonical_evaluation_design_authority()
    search = _canonical_search_operator()
    internal = _canonical_cohort(
        family="AUX",
        panel_kind="internal",
        base_seed=AUX_INTERNAL_BASE_SEED,
        pairs=300,
        map_kind="BASE",
        opponent="recovered_generator_reference",
    )
    external = _canonical_cohort(
        family="AUX",
        panel_kind="external",
        base_seed=AUX_EXTERNAL_BASE_SEED,
        pairs=250,
        map_kind="TOURNAMENT",
        opponent="catanatron_value",
    )
    rule = {
        "schema_version": "a1-aux-deterministic-selection-rule-v1",
        "internal": "treatment_pair_points_strictly_greater_than_control",
        "external": "treatment_mean_not_below_control_minus_tolerance",
        "external_non_regression_tolerance_milli": 25,
    }
    return {
        "schema_version": AUX_EVALUATION_PLAN_SCHEMA,
        "design_authority": design,
        "design_authority_sha256": design["authority_sha256"],
        "baseline_checkpoint_sha256": _require_sha(
            baseline_checkpoint_sha256, "AUX evaluation baseline"
        ),
        "internal_cohort": internal,
        "internal_cohort_sha256": _digest(internal),
        "external_cohort": external,
        "external_cohort_sha256": _digest(external),
        "search_operator": search,
        "search_operator_sha256": _digest(search),
        "decision_rule": rule,
        "decision_rule_sha256": _digest(rule),
        "internal_pairs_per_arm": internal["pairs"],
        "external_games_per_arm": external["games"],
        "panel_origin_tool_sha256": design["evaluator_file_sha256"],
        "common_random_numbers": True,
        "seat_swapped": True,
    }


def verify_aux_evaluation_plan(
    value: Any, *, baseline_checkpoint_sha256: str
) -> dict[str, Any]:
    expected = canonical_aux_evaluation_plan(
        baseline_checkpoint_sha256=baseline_checkpoint_sha256
    )
    plan = _require_exact_keys(value, set(expected), "AUX fixed evaluation plan")
    if plan != expected:
        raise CoordinatorError("AUX evaluation is not the canonical matched design")
    return copy.deepcopy(plan)


def _aux_passed_from_outcomes(
    plan: Mapping[str, Any],
    *,
    internal_pair_points_milli: Any,
    external_game_points_milli: Any,
) -> bool:
    internal = _require_exact_keys(
        internal_pair_points_milli, set(ARMS), "AUX internal outcomes"
    )
    external = _require_exact_keys(
        external_game_points_milli, set(ARMS), "AUX external outcomes"
    )
    for label, values, expected, ceiling in (
        ("internal", internal, plan["internal_pairs_per_arm"], 2000),
        ("external", external, plan["external_games_per_arm"], 1000),
    ):
        for arm in ARMS:
            points = values[arm]
            if (
                not isinstance(points, list)
                or len(points) != expected
                or any(
                    type(point) is not int or not 0 <= point <= ceiling
                    for point in points
                )
            ):
                raise CoordinatorError(f"AUX {label} outcomes drift for {arm}")
    internal_pass = sum(internal[ARM_TREATMENT]) > sum(internal[ARM_CONTROL])
    external_n = plan["external_games_per_arm"]
    tolerance = plan["decision_rule"]["external_non_regression_tolerance_milli"]
    external_pass = sum(external[ARM_TREATMENT]) >= (
        sum(external[ARM_CONTROL]) - tolerance * external_n
    )
    return internal_pass and external_pass


def _load_panel_receipt(
    path: Path,
    *,
    family: str,
    panel_kind: str,
    authority_id: str,
    arms: tuple[str, ...],
    arm_checkpoint_sha256: Mapping[str, str],
    cohort_sha256: str,
    search_operator_sha256: str,
    origin_tool_sha256: str,
) -> dict[str, Any]:
    receipt = _load_json(
        path.expanduser().resolve(strict=True),
        where=f"{family} {panel_kind} panel receipt",
    )
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "family",
            "panel_kind",
            "authority_id",
            "arms",
            "arm_checkpoint_sha256",
            "cohort_sha256",
            "search_operator_sha256",
            "common_random_numbers",
            "seat_swapped",
            "points_milli",
            "origin_tool_sha256",
            "state_sha256",
        },
        f"{family} {panel_kind} panel receipt",
    )
    if (
        receipt["schema_version"] != "a1-fixed-panel-receipt-v1"
        or receipt["family"] != family
        or receipt["panel_kind"] != panel_kind
        or receipt["authority_id"] != authority_id
        or receipt["arms"] != list(arms)
        or receipt["arm_checkpoint_sha256"] != dict(arm_checkpoint_sha256)
        or receipt["cohort_sha256"] != cohort_sha256
        or receipt["search_operator_sha256"] != search_operator_sha256
        or receipt["common_random_numbers"] is not True
        or receipt["seat_swapped"] is not True
        or receipt["origin_tool_sha256"] != origin_tool_sha256
    ):
        raise CoordinatorError(f"{family} {panel_kind} panel receipt drift")
    _require_sha(receipt["origin_tool_sha256"], "panel origin tool")
    return receipt


def _p1_coefficient_decimal(arm_id: str, eligible_mass: Decimal) -> str:
    if arm_id not in P1_ARMS:
        raise CoordinatorError("P1 sweep permits only K0, K3, and K10")
    target = Decimal(P1_TARGET_GLOBAL_DECIMALS[arm_id])
    coefficient = (target * eligible_mass).quantize(
        P1_COEFFICIENT_QUANTUM, rounding=ROUND_HALF_EVEN
    )
    return _canonical_decimal(coefficient)


def _p1_arm_recipe(
    base_recipe: Mapping[str, Any], arm_id: str, *, coefficient_decimal: str
) -> dict[str, Any]:
    if arm_id not in P1_ARMS:
        raise CoordinatorError("P1 sweep permits only K0, K3, and K10")
    recipe = copy.deepcopy(dict(base_recipe))
    recipe["world_size"] = WORLD_SIZE
    recipe["batch_size"] = LOCAL_BATCH_SIZE
    recipe["global_batch_size"] = GLOBAL_BATCH_SIZE
    recipe["grad_accum_steps"] = 1
    recipe["policy_kl_anchor_weight"] = float(Decimal(coefficient_decimal))
    return recipe


def prepare_p1_sweep(
    root: Path,
    *,
    final_lock_authority: Mapping[str, Any],
    composite: Mapping[str, Any],
    composite_descriptor_path: Path,
    p1_sample_receipt_path: Path,
    p1_sample_rows_path: Path,
    v5_recovery_receipt_path: Path,
    native_learner_admission_receipt_path: Path,
    portable_code_identity_sha256: str,
    allocations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Issue one central, non-repeatable K0/K3/K10 P1 sweep."""

    lock = verify_p1_final_lock_authority(dict(final_lock_authority))
    data = _verify_composite(dict(composite))
    evidence_origin = _repo_tool_sha256("tools/a1_scientific_evidence.py")
    try:
        sample = scientific_evidence.verify_sample_evidence(
            p1_sample_receipt_path.expanduser().resolve(strict=True),
            descriptor=composite_descriptor_path.expanduser().resolve(strict=True),
            rows_path=p1_sample_rows_path.expanduser().resolve(strict=True),
            prior_rows_path=None,
            expected_origin_tool_sha256=evidence_origin,
        )
    except (scientific_evidence.EvidenceError, OSError, ValueError) as error:
        raise CoordinatorError(f"P1 sample evidence refused: {error}") from error
    sample = _verify_sample_receipt_projection(
        sample,
        composite=data,
        sampler_seed=P1_SAMPLER_SEED,
        prior_required=False,
    )
    if (
        sample["sampler_identity_sha256"] != data["sampler_identity_sha256"]
        or sample["sample_order_sha256"] != data["sample_order_sha256"]
    ):
        raise CoordinatorError("P1 composite does not bind the replayed physical order")
    eligibility = _kl_authority_from_verified_sample(sample, composite=data)
    recovery = verify_v5_recovery_receipt(v5_recovery_receipt_path)
    parent = current_parent_authority_from_recovery(recovery)
    native_receipt, native_runtime = _consume_runtime_admission_receipt(
        native_learner_admission_receipt_path
    )
    code_sha = _require_sha(portable_code_identity_sha256, "P1 portable code")
    runtime_sha = _digest(native_runtime)
    eval_plan = canonical_p1_evaluation_plan(
        baseline_checkpoint_sha256=parent["checkpoint_sha256"]
    )
    rule_sha = eval_plan["decision_rule_sha256"]
    if set(allocations) != set(P1_ARMS):
        raise CoordinatorError("P1 allocations must bind exactly K0/K3/K10")
    fixed_allocations = {
        arm: verify_allocation(dict(allocations[arm])) for arm in P1_ARMS
    }
    _verify_p1_scheduled_allocations(fixed_allocations)
    unadmitted_hosts = {
        allocation["host_id"] for allocation in fixed_allocations.values()
    } - set(native_runtime["admitted_hosts"])
    if unadmitted_hosts:
        raise CoordinatorError(
            f"P1 allocation host lacks native admission: {sorted(unadmitted_hosts)}"
        )
    native_reports = native_receipt["hosts"]
    for allocation in fixed_allocations.values():
        _verify_allocation_matches_native_report(
            allocation, native_reports[allocation["host_id"]]
        )
    eligible_mass = _decimal(eligibility["eligible_mass_decimal"], "P1 eligible mass")
    arms = {}
    for arm in P1_ARMS:
        coefficient_decimal = _p1_coefficient_decimal(arm, eligible_mass)
        recipe = _p1_arm_recipe(
            lock["base_recipe"], arm, coefficient_decimal=coefficient_decimal
        )
        arms[arm] = {
            "arm_id": arm,
            "target_global_equivalent_decimal": P1_TARGET_GLOBAL_DECIMALS[arm],
            "eligible_mass_decimal": eligibility["eligible_mass_decimal"],
            "policy_kl_anchor_weight_decimal": coefficient_decimal,
            "policy_kl_anchor_weight": float(Decimal(coefficient_decimal)),
            "effective_recipe": recipe,
            "effective_recipe_sha256": _digest(recipe),
        }
    identity = {
        "schema_version": P1_SWEEP_SCHEMA,
        "final_lock_authority": lock,
        "composite": data,
        "kl_eligibility_authority": eligibility,
        "p1_sample_evidence_receipt": sample,
        "recovery_authority": recovery,
        "recovery_component_semantics": recovery_component_semantics(recovery),
        "current_parent_authority": parent,
        "native_runtime_authority": native_runtime,
        "native_learner_admission_receipt": native_receipt,
        "portable_code_identity_sha256": code_sha,
        "portable_runtime_identity_sha256": runtime_sha,
        "sample_dose": SHORT_SAMPLE_DOSE,
        "optimizer_steps": SHORT_OPTIMIZER_STEPS,
        "scientific_role": "diagnostic_kl_selection",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "topology": {
            "world_size": WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "grad_accum_steps": 1,
            "amp": "none",
        },
        "arms": arms,
        "selection_rule_sha256": rule_sha,
        "evaluation_plan": eval_plan,
        "schedule_slots": dict(P1_SCHEDULE_SLOTS),
        "allocations": fixed_allocations,
    }
    sweep_id = _digest(identity)
    payload = {**identity, "sweep_id": sweep_id}
    directory = _artifact_dir(root, sweep_id, create=True)
    _write_once(
        directory.parent / "a1-p1-central-kl-v1-issuance.json",
        {
            "schema_version": "a1-global-experiment-issuance-v1",
            "authority_key": "a1-p1-central-kl-v1",
            "experiment_id": sweep_id,
        },
    )
    return _write_once(directory / "p1-00-sweep.json", payload)


def load_p1_sweep(root: Path, sweep_id: str) -> dict[str, Any]:
    payload = _artifact(root, sweep_id, "p1-00-sweep.json")
    assert payload is not None
    identity = dict(payload)
    identity.pop("state_sha256", None)
    stated = identity.pop("sweep_id", None)
    if (
        payload.get("schema_version") != P1_SWEEP_SCHEMA
        or stated != sweep_id
        or _digest(identity) != sweep_id
    ):
        raise CoordinatorError("P1 central sweep identity drift")
    return payload


def claim_p1_arm(
    root: Path,
    sweep_id: str,
    *,
    arm_id: str,
    observed_allocation: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    if arm_id not in P1_ARMS:
        raise CoordinatorError("P1 sweep permits only canonical K0/K3/K10 labels")
    sweep = load_p1_sweep(root, sweep_id)
    allocation = verify_allocation(dict(observed_allocation))
    if allocation != sweep["allocations"][arm_id]:
        raise CoordinatorError(f"P1 {arm_id} host/GPU allocation drift")
    slot = sweep["schedule_slots"][arm_id]
    unfinished_prior = [
        prior
        for prior in P1_ARMS
        if sweep["schedule_slots"][prior] < slot
        and _artifact(
            root,
            sweep_id,
            f"p1-20-{prior.lower()}-terminal.json",
            required=False,
        )
        is None
    ]
    if unfinished_prior:
        raise CoordinatorError(
            f"P1 {arm_id} schedule slot cannot start before {unfinished_prior} terminate"
        )
    payload = {
        "schema_version": "a1-p1-central-arm-claim-v1",
        "sweep_id": sweep_id,
        "arm_id": arm_id,
        "prior_authority_sha256": sweep["state_sha256"],
        "arm": copy.deepcopy(sweep["arms"][arm_id]),
        "allocation": allocation,
        "execution": _verify_execution(dict(execution)),
    }
    directory = _artifact_dir(root, sweep_id, create=False)
    return _write_once(directory / f"p1-10-{arm_id.lower()}-claim.json", payload)


def load_p1_arm_executor_authority(
    root: Path,
    sweep_id: str,
    *,
    arm_id: str,
    observed_allocation: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate one claimed fixed arm into complete sealed learner inputs."""

    if arm_id not in P1_ARMS:
        raise CoordinatorError("P1 executor permits only K0/K3/K10")
    sweep = load_p1_sweep(root, sweep_id)
    observed = verify_allocation(dict(observed_allocation))
    if observed != sweep["allocations"][arm_id]:
        raise CoordinatorError(f"P1 {arm_id} executor allocation drift")
    claim = _artifact(root, sweep_id, f"p1-10-{arm_id.lower()}-claim.json")
    assert claim is not None
    authority = {
        "schema_version": "a1-p1-arm-executor-authority-v1",
        "sweep_id": sweep_id,
        "arm_id": arm_id,
        "sweep_state_sha256": sweep["state_sha256"],
        "arm_claim": copy.deepcopy(claim),
        "arm": copy.deepcopy(sweep["arms"][arm_id]),
        "current_parent_authority": copy.deepcopy(sweep["current_parent_authority"]),
        "composite": copy.deepcopy(sweep["composite"]),
        "kl_eligibility_authority": copy.deepcopy(sweep["kl_eligibility_authority"]),
        "p1_sample_evidence_receipt": copy.deepcopy(
            sweep["p1_sample_evidence_receipt"]
        ),
        "recovery_authority": copy.deepcopy(sweep["recovery_authority"]),
        "recovery_component_semantics": copy.deepcopy(
            sweep["recovery_component_semantics"]
        ),
        "native_runtime_authority": copy.deepcopy(sweep["native_runtime_authority"]),
        "native_learner_admission_receipt": copy.deepcopy(
            sweep["native_learner_admission_receipt"]
        ),
        "portable_code_identity_sha256": sweep["portable_code_identity_sha256"],
        "portable_runtime_identity_sha256": sweep["portable_runtime_identity_sha256"],
        "allocation": observed,
    }
    authority["authority_sha256"] = _digest(authority)
    directory = _artifact_dir(root, sweep_id, create=False)
    return _write_once(
        directory / f"p1-15-{arm_id.lower()}-executor-authority.json",
        authority,
    )


def complete_p1_arm(
    root: Path,
    sweep_id: str,
    *,
    arm_id: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    if arm_id not in P1_ARMS:
        raise CoordinatorError("P1 sweep permits only K0/K3/K10")
    sweep = load_p1_sweep(root, sweep_id)
    claim = _artifact(root, sweep_id, f"p1-10-{arm_id.lower()}-claim.json")
    assert claim is not None
    arm = sweep["arms"][arm_id]
    result = _require_exact_keys(
        dict(result),
        {
            "schema_version",
            "status",
            "sweep_id",
            "arm_id",
            "policy_kl_anchor_weight_decimal",
            "initializer_sha256",
            "sampled_rows",
            "optimizer_steps",
            "world_size",
            "local_batch_size",
            "global_batch_size",
            "amp",
            "fresh_adam",
            "optimizer_restored",
            "effective_recipe_sha256",
            "payload_inventory_sha256",
            "validation_split_receipt_sha256",
            "sampler_identity_sha256",
            "sample_order_sha256",
            "checkpoint_sha256",
            "optimizer_sidecar_sha256",
            "report_sha256",
            "origin_tool_sha256",
        },
        f"P1 {arm_id} result",
    )
    composite = sweep["composite"]
    if (
        result["schema_version"] != "a1-p1-central-arm-result-v1"
        or result["status"] != "complete"
        or result["sweep_id"] != sweep_id
        or result["arm_id"] != arm_id
        or result["policy_kl_anchor_weight_decimal"]
        != arm["policy_kl_anchor_weight_decimal"]
        or result["initializer_sha256"]
        != sweep["current_parent_authority"]["checkpoint_sha256"]
        or result["sampled_rows"] != SHORT_SAMPLE_DOSE
        or result["optimizer_steps"] != SHORT_OPTIMIZER_STEPS
        or result["world_size"] != WORLD_SIZE
        or result["local_batch_size"] != LOCAL_BATCH_SIZE
        or result["global_batch_size"] != GLOBAL_BATCH_SIZE
        or result["amp"] != "none"
        or result["fresh_adam"] is not True
        or result["optimizer_restored"] is not False
        or result["effective_recipe_sha256"] != arm["effective_recipe_sha256"]
        or result["payload_inventory_sha256"] != composite["payload_inventory_sha256"]
        or result["validation_split_receipt_sha256"]
        != composite["validation_split_receipt_sha256"]
        or result["sampler_identity_sha256"] != composite["sampler_identity_sha256"]
        or result["sample_order_sha256"] != composite["sample_order_sha256"]
        or result["origin_tool_sha256"]
        != _repo_tool_sha256("tools/a1_one_dose_train.py")
    ):
        raise CoordinatorError(f"P1 {arm_id} terminal drifted from central authority")
    for field in (
        "checkpoint_sha256",
        "optimizer_sidecar_sha256",
        "report_sha256",
        "origin_tool_sha256",
    ):
        _require_sha(result[field], f"P1 {arm_id} {field}")
    payload = {
        "schema_version": "a1-p1-central-arm-terminal-v1",
        "sweep_id": sweep_id,
        "arm_id": arm_id,
        "prior_authority_sha256": claim["state_sha256"],
        "result": copy.deepcopy(result),
    }
    directory = _artifact_dir(root, sweep_id, create=False)
    return _write_once(directory / f"p1-20-{arm_id.lower()}-terminal.json", payload)


def claim_p1_evaluation(
    root: Path,
    sweep_id: str,
    *,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    """Claim the one fixed evaluation only after all three arms terminate."""

    sweep = load_p1_sweep(root, sweep_id)
    terminals = {
        arm: _artifact(
            root,
            sweep_id,
            f"p1-20-{arm.lower()}-terminal.json",
            required=False,
        )
        for arm in P1_ARMS
    }
    if any(value is None for value in terminals.values()):
        raise CoordinatorError("P1 evaluation requires all K0/K3/K10 terminals")
    payload = {
        "schema_version": "a1-p1-fixed-evaluation-claim-v1",
        "sweep_id": sweep_id,
        "prior_authority_sha256": sweep["state_sha256"],
        "arm_terminal_sha256": {arm: terminals[arm]["state_sha256"] for arm in P1_ARMS},
        "evaluation_plan": copy.deepcopy(sweep["evaluation_plan"]),
        "execution": _verify_execution(dict(execution)),
    }
    directory = _artifact_dir(root, sweep_id, create=False)
    return _write_once(directory / "p1-25-evaluation-claim.json", payload)


def complete_p1_evaluation(
    root: Path,
    sweep_id: str,
    *,
    internal_panel_receipt_path: Path,
    external_panel_receipt_path: Path,
) -> dict[str, Any]:
    """Seal fixed internal+external evidence and its rule-selected winner."""

    sweep = load_p1_sweep(root, sweep_id)
    claim = _artifact(root, sweep_id, "p1-25-evaluation-claim.json")
    assert claim is not None
    terminals = {
        arm: _artifact(
            root,
            sweep_id,
            f"p1-20-{arm.lower()}-terminal.json",
            required=False,
        )
        for arm in P1_ARMS
    }
    plan = sweep["evaluation_plan"]
    expected_checkpoints = {
        arm: terminals[arm]["result"]["checkpoint_sha256"] for arm in P1_ARMS
    }
    internal = _load_panel_receipt(
        internal_panel_receipt_path,
        family="P1",
        panel_kind="internal",
        authority_id=sweep_id,
        arms=P1_ARMS,
        arm_checkpoint_sha256=expected_checkpoints,
        cohort_sha256=plan["internal_cohort_sha256"],
        search_operator_sha256=plan["search_operator_sha256"],
        origin_tool_sha256=plan["panel_origin_tool_sha256"],
    )
    external = _load_panel_receipt(
        external_panel_receipt_path,
        family="P1",
        panel_kind="external",
        authority_id=sweep_id,
        arms=P1_ARMS,
        arm_checkpoint_sha256=expected_checkpoints,
        cohort_sha256=plan["external_cohort_sha256"],
        search_operator_sha256=plan["search_operator_sha256"],
        origin_tool_sha256=plan["panel_origin_tool_sha256"],
    )
    selected_arm = _p1_selected_arm_from_outcomes(
        plan,
        internal_pair_points_milli=internal["points_milli"],
        external_game_points_milli=external["points_milli"],
    )
    selection_receipt_sha = _digest(
        {
            "decision_rule_sha256": plan["decision_rule_sha256"],
            "internal_panel_state_sha256": internal["state_sha256"],
            "external_panel_state_sha256": external["state_sha256"],
            "selected_arm": selected_arm,
        }
    )
    selection_replay_sha = _digest(
        {"selection_receipt_sha256": selection_receipt_sha, "replay": "v1"}
    )
    payload = {
        "schema_version": "a1-p1-fixed-evaluation-terminal-v1",
        "sweep_id": sweep_id,
        "prior_authority_sha256": claim["state_sha256"],
        "result": {
            "internal_panel_receipt": copy.deepcopy(internal),
            "external_panel_receipt": copy.deepcopy(external),
            "decision_rule_sha256": plan["decision_rule_sha256"],
            "selected_arm": selected_arm,
            "selection_receipt_sha256": selection_receipt_sha,
            "selection_replay_sha256": selection_replay_sha,
        },
    }
    directory = _artifact_dir(root, sweep_id, create=False)
    return _write_once(directory / "p1-27-evaluation-terminal.json", payload)


def adjudicate_p1_sweep(
    root: Path,
    sweep_id: str,
    *,
    selected_arm: str,
    applied_selection_rule_sha256: str,
) -> dict[str, Any]:
    """Seal one selected P1 recipe after all fixed arms terminate."""

    if selected_arm not in P1_ARMS:
        raise CoordinatorError("P1 adjudication selected an unknown arm")
    sweep = load_p1_sweep(root, sweep_id)
    terminals = {
        arm: _artifact(
            root,
            sweep_id,
            f"p1-20-{arm.lower()}-terminal.json",
            required=False,
        )
        for arm in P1_ARMS
    }
    if any(value is None for value in terminals.values()):
        raise CoordinatorError("P1 adjudication requires all K0/K3/K10 terminals")
    evaluation = _artifact(
        root, sweep_id, "p1-27-evaluation-terminal.json", required=False
    )
    if evaluation is None:
        raise CoordinatorError("P1 adjudication requires fixed evaluation terminal")
    evaluated = evaluation["result"]
    if selected_arm != evaluated["selected_arm"]:
        raise CoordinatorError("P1 selected arm differs from fixed evaluation")
    if applied_selection_rule_sha256 != sweep["selection_rule_sha256"]:
        raise CoordinatorError("P1 adjudication selection-rule drift")
    receipt_sha = evaluated["selection_receipt_sha256"]
    replay_sha = evaluated["selection_replay_sha256"]
    arm = sweep["arms"][selected_arm]
    authority = {
        "schema_version": P1_RECIPE_DATA_AUTHORITY_SCHEMA,
        "sweep_id": sweep_id,
        "selected_arm": selected_arm,
        "central_authority": True,
        "selection_receipt_sha256": receipt_sha,
        "selection_replay_sha256": replay_sha,
        "replay_verified": True,
        "scientific_role": "diagnostic_recipe_selection",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "requires_independent_final_replication": True,
        "effective_recipe": copy.deepcopy(arm["effective_recipe"]),
        "effective_recipe_sha256": arm["effective_recipe_sha256"],
        "composite": copy.deepcopy(sweep["composite"]),
        "recovery_authority": copy.deepcopy(sweep["recovery_authority"]),
        "recovery_component_semantics": copy.deepcopy(
            sweep["recovery_component_semantics"]
        ),
        "current_parent_authority": copy.deepcopy(
            sweep["current_parent_authority"]
        ),
        "native_runtime_authority": copy.deepcopy(
            sweep["native_runtime_authority"]
        ),
        "native_learner_admission_receipt": copy.deepcopy(
            sweep["native_learner_admission_receipt"]
        ),
        "p1_sample_evidence_receipt": copy.deepcopy(
            sweep["p1_sample_evidence_receipt"]
        ),
    }
    payload = {
        "schema_version": "a1-p1-central-selection-v1",
        "sweep_id": sweep_id,
        "prior_authority_sha256": sweep["state_sha256"],
        "arm_terminal_sha256": {
            arm_id: terminals[arm_id]["state_sha256"] for arm_id in P1_ARMS
        },
        "evaluation_terminal_sha256": evaluation["state_sha256"],
        "selection_rule_sha256": sweep["selection_rule_sha256"],
        "selected_recipe_data_authority": authority,
    }
    directory = _artifact_dir(root, sweep_id, create=False)
    return _write_once(directory / "p1-30-selection.json", payload)


def load_selected_p1_recipe_data_authority(root: Path, sweep_id: str) -> dict[str, Any]:
    selection = _artifact(root, sweep_id, "p1-30-selection.json")
    assert selection is not None
    authority = verify_p1_recipe_data_authority(
        selection["selected_recipe_data_authority"]
    )
    if authority["sweep_id"] != sweep_id:
        raise CoordinatorError("selected P1 authority sweep drift")
    return authority


def _verify_central_p1_authority(
    root: Path, authority: Mapping[str, Any]
) -> dict[str, Any]:
    selected = load_selected_p1_recipe_data_authority(root, authority["sweep_id"])
    if selected != authority:
        raise CoordinatorError("AUX did not consume exact central P1 selection")
    return selected


def prepare_experiment(
    root: Path,
    *,
    p1_recipe_data_authority: Mapping[str, Any],
    pointer_upgrade_authority: Mapping[str, Any],
    warmup_recipe: Mapping[str, Any],
    selector_rule: Mapping[str, Any],
    portable_code_identity_sha256: str,
    allocations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    p1 = verify_p1_recipe_data_authority(dict(p1_recipe_data_authority))
    p1 = _verify_central_p1_authority(root, p1)
    geometry = _current_geometry_dose_authority(p1)
    recovery = p1["recovery_authority"]
    parent = verify_current_parent_authority(
        p1["current_parent_authority"], recovery_authority=recovery
    )
    native_receipt = p1["native_learner_admission_receipt"]
    native_runtime = p1["native_runtime_authority"]
    if native_runtime != _native_runtime_authority_from_verified_receipt(
        native_receipt
    ):
        raise CoordinatorError("AUX retained native runtime projection drift")
    pointer = verify_pointer_upgrade_authority(
        dict(pointer_upgrade_authority),
        expected_parent_sha256=parent["checkpoint_sha256"],
    )
    warmup = verify_warmup_recipe(dict(warmup_recipe))
    selector = verify_selector_rule(dict(selector_rule))
    aux_evaluation = canonical_aux_evaluation_plan(
        baseline_checkpoint_sha256=parent["checkpoint_sha256"]
    )
    code_sha = _require_sha(portable_code_identity_sha256, "portable code identity")
    runtime_sha = _digest(native_runtime)
    composite = p1["composite"]
    warmup_data_drift = {
        warmup_field: {
            "expected": composite[composite_field],
            "actual": warmup[warmup_field],
        }
        for warmup_field, composite_field in (
            ("descriptor_sha256", "descriptor_sha256"),
            ("data_fingerprint", "data_fingerprint"),
            ("payload_inventory_sha256", "payload_inventory_sha256"),
            (
                "validation_split_receipt_sha256",
                "validation_split_receipt_sha256",
            ),
            ("sampler_identity_sha256", "sampler_identity_sha256"),
            ("sample_order_sha256", "sample_order_sha256"),
        )
        if warmup[warmup_field] != composite[composite_field]
    }
    if warmup_data_drift:
        raise CoordinatorError(
            f"pointer warmup data/split/sampler drift: {warmup_data_drift}"
        )
    if set(allocations) != set(STAGES):
        raise CoordinatorError(f"allocations must bind exactly {list(STAGES)}")
    verified_allocations = {
        stage: verify_allocation(dict(allocations[stage])) for stage in STAGES
    }
    _verify_aux_scheduled_allocations(verified_allocations)
    unadmitted_hosts = {
        allocation["host_id"] for allocation in verified_allocations.values()
    } - set(native_runtime["admitted_hosts"])
    if unadmitted_hosts:
        raise CoordinatorError(
            f"AUX allocation host lacks native admission: {sorted(unadmitted_hosts)}"
        )
    native_reports = native_receipt["hosts"]
    for allocation in verified_allocations.values():
        _verify_allocation_matches_native_report(
            allocation, native_reports[allocation["host_id"]]
        )
    portable_science = {
        "schema_version": "a1-aux-portable-science-identity-v1",
        "geometry_dose_authority": geometry,
        "p1_recipe_data_authority": p1,
        "effective_recipe": copy.deepcopy(p1["effective_recipe"]),
        "effective_recipe_sha256": p1["effective_recipe_sha256"],
        "composite": copy.deepcopy(p1["composite"]),
        "current_parent_authority": parent,
        "recovery_authority": copy.deepcopy(recovery),
        "recovery_component_semantics": copy.deepcopy(
            p1["recovery_component_semantics"]
        ),
        "native_runtime_authority": native_runtime,
        "native_learner_admission_receipt": copy.deepcopy(native_receipt),
        "p1_sample_evidence_receipt": copy.deepcopy(
            p1["p1_sample_evidence_receipt"]
        ),
        "pointer_upgrade_authority": pointer,
        "warmup_recipe": warmup,
        "selector_rule": selector,
        "evaluation_plan": aux_evaluation,
        "portable_code_identity_sha256": code_sha,
        "portable_runtime_identity_sha256": runtime_sha,
    }
    portable_science["portable_science_identity_sha256"] = _digest(portable_science)
    experiment_identity = {
        "schema_version": EXPERIMENT_SCHEMA,
        "portable_science_identity_sha256": portable_science[
            "portable_science_identity_sha256"
        ],
        "allocations": verified_allocations,
    }
    experiment_id = _digest(experiment_identity)
    payload = {
        "schema_version": EXPERIMENT_SCHEMA,
        "experiment_id": experiment_id,
        "portable_science_identity": portable_science,
        "portable_science_identity_sha256": portable_science[
            "portable_science_identity_sha256"
        ],
        "allocations": verified_allocations,
    }
    directory = _artifact_dir(root, experiment_id, create=True)
    _write_once(
        directory.parent / "a1-pointer-aux-v1-issuance.json",
        {
            "schema_version": "a1-global-experiment-issuance-v1",
            "authority_key": "a1-pointer-aux-v1",
            "experiment_id": experiment_id,
        },
    )
    return _write_once(directory / "00-experiment.json", payload)


def load_experiment(root: Path, experiment_id: str) -> dict[str, Any]:
    payload = _artifact(root, experiment_id, "00-experiment.json")
    assert payload is not None
    if (
        payload.get("schema_version") != EXPERIMENT_SCHEMA
        or payload.get("experiment_id") != experiment_id
        or payload.get("portable_science_identity_sha256")
        != _digest(
            {
                key: value
                for key, value in payload["portable_science_identity"].items()
                if key != "portable_science_identity_sha256"
            }
        )
        or payload["portable_science_identity"].get("portable_science_identity_sha256")
        != payload.get("portable_science_identity_sha256")
    ):
        raise CoordinatorError("experiment authority semantic drift")
    return payload


def _stage_claim(
    root: Path,
    experiment_id: str,
    *,
    stage: str,
    prior_filename: str,
    claim_filename: str,
    observed_allocation: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    experiment = load_experiment(root, experiment_id)
    if stage not in STAGES:
        raise CoordinatorError(f"unsupported central stage: {stage}")
    observed = verify_allocation(dict(observed_allocation))
    if observed != experiment["allocations"][stage]:
        raise CoordinatorError(f"{stage} host/GPU allocation drift")
    prior = _artifact(root, experiment_id, prior_filename)
    assert prior is not None
    payload = {
        "schema_version": "a1-aux-stage-claim-v1",
        "experiment_id": experiment_id,
        "stage": stage,
        "prior_authority_sha256": prior["state_sha256"],
        "allocation": observed,
        "execution": _verify_execution(dict(execution)),
    }
    directory = _artifact_dir(root, experiment_id, create=False)
    return _write_once(directory / claim_filename, payload)


def claim_warmup(
    root: Path,
    experiment_id: str,
    *,
    observed_allocation: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    return _stage_claim(
        root,
        experiment_id,
        stage="WARMUP",
        prior_filename="00-experiment.json",
        claim_filename="10-warmup-claim.json",
        observed_allocation=observed_allocation,
        execution=execution,
    )


def _load_stage_executor_authority(
    root: Path,
    experiment_id: str,
    *,
    stage: str,
    claim_filename: str,
    observed_allocation: Mapping[str, Any],
) -> dict[str, Any]:
    experiment = load_experiment(root, experiment_id)
    observed = verify_allocation(dict(observed_allocation))
    if observed != experiment["allocations"][stage]:
        raise CoordinatorError(f"{stage} executor allocation drift")
    claim = _artifact(root, experiment_id, claim_filename)
    assert claim is not None
    payload = {
        "schema_version": f"a1-aux-{stage.lower()}-executor-authority-v1",
        "experiment_id": experiment_id,
        "stage": stage,
        "experiment_state_sha256": experiment["state_sha256"],
        "portable_science_identity": copy.deepcopy(
            experiment["portable_science_identity"]
        ),
        "stage_claim": copy.deepcopy(claim),
        "allocation": observed,
    }
    if stage == "GEOMETRY":
        warmup = _artifact(root, experiment_id, "20-warmup-terminal.json")
        assert warmup is not None
        payload["warmup_terminal"] = copy.deepcopy(warmup)
    payload["authority_sha256"] = _digest(payload)
    filename = (
        "15-warmup-executor-authority.json"
        if stage == "WARMUP"
        else "35-geometry-executor-authority.json"
    )
    directory = _artifact_dir(root, experiment_id, create=False)
    return _write_once(directory / filename, payload)


def load_warmup_executor_authority(
    root: Path,
    experiment_id: str,
    *,
    observed_allocation: Mapping[str, Any],
) -> dict[str, Any]:
    return _load_stage_executor_authority(
        root,
        experiment_id,
        stage="WARMUP",
        claim_filename="10-warmup-claim.json",
        observed_allocation=observed_allocation,
    )


def load_geometry_executor_authority(
    root: Path,
    experiment_id: str,
    *,
    observed_allocation: Mapping[str, Any],
) -> dict[str, Any]:
    return _load_stage_executor_authority(
        root,
        experiment_id,
        stage="GEOMETRY",
        claim_filename="30-geometry-claim.json",
        observed_allocation=observed_allocation,
    )


def complete_warmup(
    root: Path, experiment_id: str, *, result: Mapping[str, Any]
) -> dict[str, Any]:
    experiment = load_experiment(root, experiment_id)
    claim = _artifact(root, experiment_id, "10-warmup-claim.json")
    assert claim is not None
    warmup = experiment["portable_science_identity"]["warmup_recipe"]
    pointer = experiment["portable_science_identity"]["pointer_upgrade_authority"]
    result = _require_exact_keys(
        dict(result),
        {
            "schema_version",
            "status",
            "sampled_rows",
            "optimizer_steps",
            "input_initializer_sha256",
            "warmed_checkpoint_sha256",
            "optimizer_sidecar_sha256",
            "optimizer_sidecar_discarded_for_joint",
            "changed_parameter_prefixes",
            "changed_parameter_set_sha256",
            "inherited_parameter_identity_sha256",
            "inherited_parameters_bit_identical",
            "main_output_max_diff",
            "report_sha256",
            "origin_tool_sha256",
        },
        "warmup terminal result",
    )
    if (
        result["schema_version"] != "a1-aux-pointer-warmup-result-v1"
        or result["status"] != "complete"
        or result["sampled_rows"] != warmup["sample_dose"]
        or result["optimizer_steps"] != warmup["optimizer_steps"]
        or result["input_initializer_sha256"] != pointer["upgraded_initializer_sha256"]
        or result["optimizer_sidecar_discarded_for_joint"] is not True
        or result["changed_parameter_prefixes"] != list(POINTER_TRAINABLE_PREFIXES)
        or result["changed_parameter_set_sha256"] != pointer["new_parameter_set_sha256"]
        or result["inherited_parameters_bit_identical"] is not True
        or result["main_output_max_diff"] != 0.0
    ):
        raise CoordinatorError("warmup terminal did not prove head-only commissioning")
    for field in (
        "warmed_checkpoint_sha256",
        "optimizer_sidecar_sha256",
        "inherited_parameter_identity_sha256",
        "report_sha256",
        "origin_tool_sha256",
    ):
        _require_sha(result[field], f"warmup result {field}")
    payload = {
        "schema_version": "a1-aux-pointer-warmup-terminal-v1",
        "experiment_id": experiment_id,
        "prior_authority_sha256": claim["state_sha256"],
        "result": copy.deepcopy(result),
    }
    directory = _artifact_dir(root, experiment_id, create=False)
    return _write_once(directory / "20-warmup-terminal.json", payload)


def claim_geometry(
    root: Path,
    experiment_id: str,
    *,
    observed_allocation: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    return _stage_claim(
        root,
        experiment_id,
        stage="GEOMETRY",
        prior_filename="20-warmup-terminal.json",
        claim_filename="30-geometry-claim.json",
        observed_allocation=observed_allocation,
        execution=execution,
    )


def _select_coefficient(
    rule: Mapping[str, Any],
    *,
    main_norm: Decimal,
    unit_aux_norm: Decimal,
    cosine: Decimal,
) -> tuple[str, str]:
    if main_norm <= 0 or unit_aux_norm <= 0:
        raise CoordinatorError("gradient selector norms must be positive")
    ratio_cap = _decimal(
        rule["maximum_aux_to_main_ratio_decimal"], "selector ratio cap"
    )
    opposing_cap = _decimal(
        rule["maximum_opposing_projection_decimal"],
        "selector opposing projection cap",
    )
    minimum = _decimal(rule["minimum_coefficient_decimal"], "selector minimum")
    maximum = _decimal(rule["maximum_coefficient_decimal"], "selector maximum")
    quantum = _decimal(rule["quantum_decimal"], "selector quantum")
    ratio = unit_aux_norm / main_norm
    candidates = [ratio_cap / ratio, maximum]
    if cosine < 0:
        opposing = -(ratio * cosine)
        if opposing <= 0:
            raise CoordinatorError("negative-cosine opposing projection is invalid")
        candidates.append(opposing_cap / opposing)
    raw = min(candidates)
    # Decimal ``//`` implements an exact floor for positive values.  Do not use
    # ROUND_HALF_EVEN here: rounding upward could exceed either safety cap.
    selected = (raw // quantum) * quantum
    if selected < minimum or selected > maximum:
        raise CoordinatorError("gradient-selected coefficient is outside safe range")
    return _canonical_decimal(raw), _canonical_decimal(selected)


def complete_geometry(
    root: Path, experiment_id: str, *, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    experiment = load_experiment(root, experiment_id)
    warmup = _artifact(root, experiment_id, "20-warmup-terminal.json")
    claim = _artifact(root, experiment_id, "30-geometry-claim.json")
    assert warmup is not None and claim is not None
    rule = experiment["portable_science_identity"]["selector_rule"]
    evidence = _require_exact_keys(
        dict(evidence),
        {
            "schema_version",
            "status",
            "warmed_checkpoint_sha256",
            "probe_manifest_sha256",
            "probe_sampler_seed",
            "probe_row_order_sha256",
            "probe_batches",
            "probe_batch_size",
            "shared_parameter_set_sha256",
            "batch_shared_parameter_set_sha256",
            "per_batch_geometry",
            "same_forward_graph",
            "global_ddp_aggregation",
            "optimizer_steps",
            "persistent_state_mutated",
            "report_sha256",
            "origin_tool_sha256",
        },
        "gradient geometry evidence",
    )
    batches = evidence["per_batch_geometry"]
    if not isinstance(batches, list) or len(batches) != rule["probe_batches"]:
        raise CoordinatorError("gradient geometry batch count drift")
    main_squared = Decimal(0)
    aux_squared = Decimal(0)
    dot = Decimal(0)
    for index, raw_batch in enumerate(batches):
        batch = _require_exact_keys(
            raw_batch,
            {
                "batch_index",
                "shared_parameter_set_sha256",
                "main_squared_norm_decimal",
                "unit_aux_squared_norm_decimal",
                "gradient_dot_decimal",
            },
            f"gradient geometry batch {index}",
        )
        batch_main_squared = _decimal(
            batch["main_squared_norm_decimal"], f"batch {index} main squared norm"
        )
        batch_aux_squared = _decimal(
            batch["unit_aux_squared_norm_decimal"],
            f"batch {index} aux squared norm",
        )
        batch_dot = _decimal(
            batch["gradient_dot_decimal"], f"batch {index} gradient dot"
        )
        if (
            batch["batch_index"] != index
            or batch["shared_parameter_set_sha256"]
            != rule["shared_parameter_set_sha256"]
            or batch_main_squared <= 0
            or batch_aux_squared <= 0
        ):
            raise CoordinatorError("gradient geometry per-batch surface drift")
        main_squared += batch_main_squared
        aux_squared += batch_aux_squared
        dot += batch_dot
    main_norm = main_squared.sqrt()
    aux_norm = aux_squared.sqrt()
    cosine = dot / (main_norm * aux_norm)
    if (
        evidence["schema_version"] != "a1-aux-gradient-geometry-evidence-v1"
        or evidence["status"] != "complete"
        or evidence["warmed_checkpoint_sha256"]
        != warmup["result"]["warmed_checkpoint_sha256"]
        or evidence["probe_manifest_sha256"] != rule["probe_manifest_sha256"]
        or evidence["probe_sampler_seed"] != rule["probe_sampler_seed"]
        or evidence["probe_row_order_sha256"] != rule["probe_row_order_sha256"]
        or evidence["probe_batches"] != rule["probe_batches"]
        or evidence["probe_batch_size"] != rule["probe_batch_size"]
        or evidence["shared_parameter_set_sha256"]
        != rule["shared_parameter_set_sha256"]
        or evidence["batch_shared_parameter_set_sha256"]
        != [rule["shared_parameter_set_sha256"]] * rule["probe_batches"]
        or evidence["same_forward_graph"] is not True
        or evidence["global_ddp_aggregation"] is not True
        or evidence["optimizer_steps"] != 0
        or evidence["persistent_state_mutated"] is not False
        or not Decimal("-1") <= cosine <= Decimal("1")
    ):
        raise CoordinatorError("gradient geometry did not satisfy preregistered probe")
    for field in ("report_sha256", "origin_tool_sha256"):
        _require_sha(evidence[field], f"gradient geometry {field}")
    raw, selected = _select_coefficient(
        rule, main_norm=main_norm, unit_aux_norm=aux_norm, cosine=cosine
    )
    payload = {
        "schema_version": "a1-aux-gradient-selector-terminal-v1",
        "experiment_id": experiment_id,
        "prior_authority_sha256": claim["state_sha256"],
        "selector_rule_sha256": _digest(rule),
        "evidence": copy.deepcopy(evidence),
        "evidence_sha256": _digest(evidence),
        "derived_geometry": {
            "aggregation": "concatenated_batch_gradient_geometry",
            "main_gradient_norm_decimal": _canonical_decimal(main_norm),
            "unit_aux_gradient_norm_decimal": _canonical_decimal(aux_norm),
            "gradient_dot_decimal": _canonical_decimal(dot),
            "gradient_cosine_decimal": _canonical_decimal(cosine),
        },
        "raw_coefficient_decimal": raw,
        "selected_coefficient_decimal": selected,
    }
    directory = _artifact_dir(root, experiment_id, create=False)
    return _write_once(directory / "40-geometry-terminal.json", payload)


def issue_pair(root: Path, experiment_id: str) -> dict[str, Any]:
    experiment = load_experiment(root, experiment_id)
    warmup = _artifact(root, experiment_id, "20-warmup-terminal.json")
    geometry = _artifact(root, experiment_id, "40-geometry-terminal.json")
    assert warmup is not None and geometry is not None
    selected_decimal = geometry["selected_coefficient_decimal"]
    selected = float(_decimal(selected_decimal, "selected coefficient"))
    science = copy.deepcopy(experiment["portable_science_identity"])
    science.update(
        {
            "warmup_terminal_sha256": warmup["state_sha256"],
            "warmed_checkpoint_sha256": warmup["result"]["warmed_checkpoint_sha256"],
            "inherited_parameter_identity_sha256": warmup["result"][
                "inherited_parameter_identity_sha256"
            ],
            "gradient_geometry_terminal_sha256": geometry["state_sha256"],
            "gradient_selector_evidence_sha256": geometry["evidence_sha256"],
            "selected_aux_coefficient_decimal": selected_decimal,
        }
    )
    science["portable_science_identity_sha256"] = _digest(
        {
            key: value
            for key, value in science.items()
            if key != "portable_science_identity_sha256"
        }
    )
    joint = {
        "sample_dose": SHORT_SAMPLE_DOSE,
        "optimizer_steps": SHORT_OPTIMIZER_STEPS,
        "world_size": WORLD_SIZE,
        "local_batch_size": LOCAL_BATCH_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "grad_accum_steps": 1,
        "amp": "none",
        "fresh_adam": True,
        "resume_optimizer": False,
        "initializer_sha256": warmup["result"]["warmed_checkpoint_sha256"],
        "warmup_optimizer_sidecar_sha256": warmup["result"]["optimizer_sidecar_sha256"],
        "warmup_optimizer_sidecar_discarded": True,
        "effective_recipe": copy.deepcopy(science["effective_recipe"]),
        "effective_recipe_sha256": science["effective_recipe_sha256"],
        "composite": copy.deepcopy(science["composite"]),
    }
    arms = {
        ARM_CONTROL: {
            "arm_id": ARM_CONTROL,
            "aux_subgoal_loss_weight_decimal": "0",
            "aux_subgoal_loss_weight": 0.0,
            "aux_heads_frozen_and_skipped": True,
        },
        ARM_TREATMENT: {
            "arm_id": ARM_TREATMENT,
            "aux_subgoal_loss_weight_decimal": selected_decimal,
            "aux_subgoal_loss_weight": selected,
            "aux_heads_frozen_and_skipped": False,
        },
    }
    pair_identity = {
        "schema_version": PAIR_SCHEMA,
        "scientific_role": "diagnostic_aux_selection",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "requires_independent_final_replication": True,
        "portable_science_identity_sha256": science["portable_science_identity_sha256"],
        "selected_aux_coefficient_decimal": selected_decimal,
        "joint": joint,
        "arms": arms,
        "allocations": {arm: experiment["allocations"][arm] for arm in ARMS},
    }
    pair_id = _digest(pair_identity)
    payload = {
        "schema_version": PAIR_SCHEMA,
        "experiment_id": experiment_id,
        "pair_id": pair_id,
        "prior_authority_sha256": geometry["state_sha256"],
        "scientific_role": "diagnostic_aux_selection",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "requires_independent_final_replication": True,
        "portable_science_identity": science,
        "portable_science_identity_sha256": science["portable_science_identity_sha256"],
        "p1_selection_authority_sha256": _digest(science["p1_recipe_data_authority"]),
        "geometry_dose_authority_sha256": _digest(science["geometry_dose_authority"]),
        "exact_current_parent_sha256": science["current_parent_authority"][
            "checkpoint_sha256"
        ],
        "pointer_upgrade_identity_sha256": _digest(
            science["pointer_upgrade_authority"]
        ),
        "warmup_terminal_sha256": warmup["state_sha256"],
        "gradient_geometry_terminal_sha256": geometry["state_sha256"],
        "selector_rule_sha256": geometry["selector_rule_sha256"],
        "selected_aux_coefficient_decimal": selected_decimal,
        "selected_aux_coefficient": selected,
        "joint": joint,
        "arms": arms,
        "allocations": pair_identity["allocations"],
    }
    directory = _artifact_dir(root, experiment_id, create=False)
    return _write_once(directory / "50-pair-issued.json", payload)


def claim_arm(
    root: Path,
    experiment_id: str,
    *,
    arm_id: str,
    observed_allocation: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    if arm_id not in ARMS:
        raise CoordinatorError("AUX pair permits only canonical AUX0 and AUXT labels")
    if (
        arm_id == ARM_TREATMENT
        and _artifact(
            root,
            experiment_id,
            "70-aux0-terminal.json",
            required=False,
        )
        is None
    ):
        raise CoordinatorError(
            "AUXT cannot start before AUX0 terminates on the sole 8xB200 learner"
        )
    return _stage_claim(
        root,
        experiment_id,
        stage=arm_id,
        prior_filename="50-pair-issued.json",
        claim_filename=f"60-{arm_id.lower()}-claim.json",
        observed_allocation=observed_allocation,
        execution=execution,
    )


def load_aux_pair_executor_authority(
    root: Path,
    experiment_id: str,
    *,
    arm_id: str,
    observed_allocation: Mapping[str, Any],
) -> dict[str, Any]:
    if arm_id not in ARMS:
        raise CoordinatorError("executor authority permits only AUX0 or AUXT")
    pair = _artifact(root, experiment_id, "50-pair-issued.json")
    claim = _artifact(root, experiment_id, f"60-{arm_id.lower()}-claim.json")
    assert pair is not None and claim is not None
    observed = verify_allocation(dict(observed_allocation))
    if (
        observed != pair["allocations"][arm_id]
        or claim["allocation"] != observed
        or claim["prior_authority_sha256"] != pair["state_sha256"]
    ):
        raise CoordinatorError("executor host/GPU allocation or pair claim drift")
    payload = {
        "schema_version": EXECUTOR_AUTHORITY_SCHEMA,
        "aux_pair_contract": copy.deepcopy(pair),
        "arm": copy.deepcopy(pair["arms"][arm_id]),
        "arm_claim": copy.deepcopy(claim),
        "selected_aux_coefficient_decimal": pair["selected_aux_coefficient_decimal"],
        "selected_aux_coefficient": pair["selected_aux_coefficient"],
    }
    payload["authority_sha256"] = _digest(payload)
    directory = _artifact_dir(root, experiment_id, create=False)
    return _write_once(
        directory / f"65-{arm_id.lower()}-executor-authority.json", payload
    )


def complete_arm(
    root: Path,
    experiment_id: str,
    *,
    arm_id: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    if arm_id not in ARMS:
        raise CoordinatorError("AUX pair permits only AUX0 and AUXT")
    pair = _artifact(root, experiment_id, "50-pair-issued.json")
    claim = _artifact(root, experiment_id, f"60-{arm_id.lower()}-claim.json")
    assert pair is not None and claim is not None
    arm = pair["arms"][arm_id]
    result = _require_exact_keys(
        dict(result),
        {
            "schema_version",
            "status",
            "pair_id",
            "arm_id",
            "aux_subgoal_loss_weight_decimal",
            "initializer_sha256",
            "sampled_rows",
            "optimizer_steps",
            "fresh_adam",
            "optimizer_restored",
            "effective_recipe_sha256",
            "payload_inventory_sha256",
            "validation_split_receipt_sha256",
            "sampler_identity_sha256",
            "sample_order_sha256",
            "checkpoint_sha256",
            "optimizer_sidecar_sha256",
            "report_sha256",
            "origin_tool_sha256",
        },
        f"{arm_id} terminal result",
    )
    composite = pair["joint"]["composite"]
    if (
        result["schema_version"] != "a1-aux-joint-arm-result-v1"
        or result["status"] != "complete"
        or result["pair_id"] != pair["pair_id"]
        or result["arm_id"] != arm_id
        or result["aux_subgoal_loss_weight_decimal"]
        != arm["aux_subgoal_loss_weight_decimal"]
        or result["initializer_sha256"] != pair["joint"]["initializer_sha256"]
        or result["sampled_rows"] != SHORT_SAMPLE_DOSE
        or result["optimizer_steps"] != SHORT_OPTIMIZER_STEPS
        or result["fresh_adam"] is not True
        or result["optimizer_restored"] is not False
        or result["effective_recipe_sha256"] != pair["joint"]["effective_recipe_sha256"]
        or result["payload_inventory_sha256"] != composite["payload_inventory_sha256"]
        or result["validation_split_receipt_sha256"]
        != composite["validation_split_receipt_sha256"]
        or result["sampler_identity_sha256"] != composite["sampler_identity_sha256"]
        or result["sample_order_sha256"] != composite["sample_order_sha256"]
        or result["origin_tool_sha256"]
        != _repo_tool_sha256("tools/a1_one_dose_train.py")
    ):
        raise CoordinatorError(f"{arm_id} terminal drifted from issued pair")
    for field in (
        "checkpoint_sha256",
        "optimizer_sidecar_sha256",
        "report_sha256",
        "origin_tool_sha256",
    ):
        _require_sha(result[field], f"{arm_id} result {field}")
    payload = {
        "schema_version": "a1-aux-joint-arm-terminal-v1",
        "experiment_id": experiment_id,
        "pair_id": pair["pair_id"],
        "arm_id": arm_id,
        "prior_authority_sha256": claim["state_sha256"],
        "result": copy.deepcopy(result),
    }
    directory = _artifact_dir(root, experiment_id, create=False)
    return _write_once(directory / f"70-{arm_id.lower()}-terminal.json", payload)


def claim_pair_evaluation(
    root: Path,
    experiment_id: str,
    *,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    pair = _artifact(root, experiment_id, "50-pair-issued.json")
    control = _artifact(root, experiment_id, "70-aux0-terminal.json")
    treatment = _artifact(root, experiment_id, "70-auxt-terminal.json")
    assert pair is not None and control is not None and treatment is not None
    payload = {
        "schema_version": "a1-aux-fixed-evaluation-claim-v1",
        "experiment_id": experiment_id,
        "pair_id": pair["pair_id"],
        "prior_authority_sha256": pair["state_sha256"],
        "arm_terminal_sha256": {
            ARM_CONTROL: control["state_sha256"],
            ARM_TREATMENT: treatment["state_sha256"],
        },
        "evaluation_plan": copy.deepcopy(
            pair["portable_science_identity"]["evaluation_plan"]
        ),
        "execution": _verify_execution(dict(execution)),
    }
    directory = _artifact_dir(root, experiment_id, create=False)
    return _write_once(directory / "75-evaluation-claim.json", payload)


def complete_pair_evaluation(
    root: Path,
    experiment_id: str,
    *,
    internal_panel_receipt_path: Path,
    external_panel_receipt_path: Path,
) -> dict[str, Any]:
    pair = _artifact(root, experiment_id, "50-pair-issued.json")
    claim = _artifact(root, experiment_id, "75-evaluation-claim.json")
    control = _artifact(root, experiment_id, "70-aux0-terminal.json")
    treatment = _artifact(root, experiment_id, "70-auxt-terminal.json")
    assert pair is not None and claim is not None
    assert control is not None and treatment is not None
    plan = pair["portable_science_identity"]["evaluation_plan"]
    expected_checkpoints = {
        ARM_CONTROL: control["result"]["checkpoint_sha256"],
        ARM_TREATMENT: treatment["result"]["checkpoint_sha256"],
    }
    internal = _load_panel_receipt(
        internal_panel_receipt_path,
        family="AUX",
        panel_kind="internal",
        authority_id=pair["pair_id"],
        arms=ARMS,
        arm_checkpoint_sha256=expected_checkpoints,
        cohort_sha256=plan["internal_cohort_sha256"],
        search_operator_sha256=plan["search_operator_sha256"],
        origin_tool_sha256=plan["panel_origin_tool_sha256"],
    )
    external = _load_panel_receipt(
        external_panel_receipt_path,
        family="AUX",
        panel_kind="external",
        authority_id=pair["pair_id"],
        arms=ARMS,
        arm_checkpoint_sha256=expected_checkpoints,
        cohort_sha256=plan["external_cohort_sha256"],
        search_operator_sha256=plan["search_operator_sha256"],
        origin_tool_sha256=plan["panel_origin_tool_sha256"],
    )
    passed = _aux_passed_from_outcomes(
        plan,
        internal_pair_points_milli=internal["points_milli"],
        external_game_points_milli=external["points_milli"],
    )
    payload = {
        "schema_version": "a1-aux-fixed-evaluation-terminal-v1",
        "experiment_id": experiment_id,
        "pair_id": pair["pair_id"],
        "prior_authority_sha256": claim["state_sha256"],
        "result": {
            "internal_panel_receipt": copy.deepcopy(internal),
            "external_panel_receipt": copy.deepcopy(external),
            "decision_rule_sha256": plan["decision_rule_sha256"],
            "passed": passed,
        },
    }
    directory = _artifact_dir(root, experiment_id, create=False)
    return _write_once(directory / "77-evaluation-terminal.json", payload)


def finalize_pair(root: Path, experiment_id: str) -> dict[str, Any]:
    pair = _artifact(root, experiment_id, "50-pair-issued.json")
    control = _artifact(root, experiment_id, "70-aux0-terminal.json")
    treatment = _artifact(root, experiment_id, "70-auxt-terminal.json")
    evaluation = _artifact(root, experiment_id, "77-evaluation-terminal.json")
    assert pair is not None and control is not None and treatment is not None
    assert evaluation is not None
    payload = {
        "schema_version": "a1-aux-pair-terminal-v1",
        "experiment_id": experiment_id,
        "pair_id": pair["pair_id"],
        "prior_authority_sha256": evaluation["state_sha256"],
        "arm_terminal_sha256": {
            ARM_CONTROL: control["state_sha256"],
            ARM_TREATMENT: treatment["state_sha256"],
        },
        "evaluation_terminal_sha256": evaluation["state_sha256"],
        "passed": evaluation["result"]["passed"],
        "selected_aux_decision": (
            ARM_TREATMENT if evaluation["result"]["passed"] else ARM_CONTROL
        ),
        "diagnostic_only": True,
        "promotion_eligible": False,
        "requires_independent_final_replication": True,
    }
    directory = _artifact_dir(root, experiment_id, create=False)
    return _write_once(directory / "80-pair-terminal.json", payload)


def issue_final_replication(
    root: Path,
    experiment_id: str,
    *,
    composite_descriptor_path: Path,
    component_routing_receipt_path: Path,
    sampling_receipt_path: Path,
    sampling_rows_path: Path,
    p1_sample_rows_path: Path,
) -> dict[str, Any]:
    """Issue the sole promotion-eligible replication after diagnostic selection."""

    experiment = load_experiment(root, experiment_id)
    pair = _artifact(root, experiment_id, "50-pair-issued.json")
    warmup = _artifact(root, experiment_id, "20-warmup-terminal.json")
    pair_terminal = _artifact(root, experiment_id, "80-pair-terminal.json")
    assert pair is not None and warmup is not None and pair_terminal is not None
    science = experiment["portable_science_identity"]
    p1 = verify_p1_recipe_data_authority(science["p1_recipe_data_authority"])
    composite = p1["composite"]
    evidence_origin = _repo_tool_sha256("tools/a1_scientific_evidence.py")
    descriptor = composite_descriptor_path.expanduser().resolve(strict=True)
    try:
        routing = scientific_evidence.verify_mixed_routing_receipt(
            component_routing_receipt_path.expanduser().resolve(strict=True),
            descriptor=descriptor,
            expected_origin_tool_sha256=evidence_origin,
        )
        sampling = scientific_evidence.verify_sample_evidence(
            sampling_receipt_path.expanduser().resolve(strict=True),
            descriptor=descriptor,
            rows_path=sampling_rows_path.expanduser().resolve(strict=True),
            prior_rows_path=p1_sample_rows_path.expanduser().resolve(strict=True),
            expected_origin_tool_sha256=evidence_origin,
        )
    except (scientific_evidence.EvidenceError, OSError, ValueError) as error:
        raise CoordinatorError(f"FINAL scientific evidence refused: {error}") from error
    sampling = _verify_sample_receipt_projection(
        sampling,
        composite=composite,
        sampler_seed=FINAL_SAMPLER_SEED,
        prior_required=True,
    )
    prior = p1["p1_sample_evidence_receipt"]
    if (
        routing["descriptor_sha256"] != composite["descriptor_sha256"]
        or routing["payload_inventory_sha256"]
        != composite["payload_inventory_sha256"]
        or routing["origin_tool_sha256"] != evidence_origin
        or sampling["sampler_identity_sha256"]
        == prior["sampler_identity_sha256"]
        or sampling["sample_order_sha256"] == prior["sample_order_sha256"]
        or sampling["row_set_sha256"] == prior["row_set_sha256"]
        or sampling["prior_rows_file_sha256"] != prior["rows_file_sha256"]
        or sampling["prior_row_set_sha256"] != prior["row_set_sha256"]
        or sampling["prior_unique_row_count"] != prior["unique_row_count"]
    ):
        raise CoordinatorError(
            "FINAL sample/routing is not an independent replay bound to P1"
        )
    selected_aux = pair_terminal["selected_aux_decision"]
    use_treatment = selected_aux == ARM_TREATMENT
    selected_coefficient_decimal = (
        pair["selected_aux_coefficient_decimal"] if use_treatment else "0"
    )
    selected_coefficient = float(Decimal(selected_coefficient_decimal))
    final_eligible_mass = Decimal(sampling["kl_eligible_rows"]) / Decimal(
        sampling["sample_dose"]
    )
    final_kl_coefficient_decimal = _p1_coefficient_decimal(
        p1["selected_arm"], final_eligible_mass
    )
    effective_recipe = copy.deepcopy(p1["effective_recipe"])
    effective_recipe.update(
        {
            "sampler_seed": FINAL_SAMPLER_SEED,
            "policy_kl_anchor_weight": float(
                Decimal(final_kl_coefficient_decimal)
            ),
            "aux_subgoal_heads": use_treatment,
            "aux_settlement_pointer_head": use_treatment,
            "aux_subgoal_loss_weight": selected_coefficient,
        }
    )
    initializer = {
        "exact_current_parent_authority": copy.deepcopy(
            science["current_parent_authority"]
        ),
        "diagnostic_arm_checkpoint_forbidden": True,
        "base_parent_lineage_reloaded": True,
        "selected_aux_decision": selected_aux,
        "pointer_upgrade_authority": (
            copy.deepcopy(science["pointer_upgrade_authority"])
            if use_treatment
            else None
        ),
        "warmup_recipe": (
            copy.deepcopy(science["warmup_recipe"]) if use_treatment else None
        ),
        "reference_warmup_terminal": (copy.deepcopy(warmup) if use_treatment else None),
        "warmup_initializer_role": (
            "shared_immutable_architecture_initializer" if use_treatment else None
        ),
        "exact_reference_warmup_bytes_reused": use_treatment,
    }
    authority = {
        "schema_version": FINAL_REPLICATION_SCHEMA,
        "experiment_id": experiment_id,
        "pair_id": pair["pair_id"],
        "prior_authority_sha256": pair_terminal["state_sha256"],
        "diagnostic_p1_selection_authority": copy.deepcopy(p1),
        "diagnostic_aux_pair_terminal": copy.deepcopy(pair_terminal),
        "selected_aux_decision": selected_aux,
        "selected_aux_coefficient_decimal": selected_coefficient_decimal,
        "selected_aux_coefficient": selected_coefficient,
        "selected_p1_target_global_equivalent_decimal": P1_TARGET_GLOBAL_DECIMALS[
            p1["selected_arm"]
        ],
        "final_kl_eligible_mass_decimal": sampling["kl_eligible_mass_decimal"],
        "final_policy_kl_anchor_weight_decimal": final_kl_coefficient_decimal,
        "initializer_authority": initializer,
        "component_routing_receipt": copy.deepcopy(routing),
        "component_routing_state_sha256": routing["state_sha256"],
        "sampling_receipt": copy.deepcopy(sampling),
        "sampling_state_sha256": sampling["state_sha256"],
        "effective_recipe": effective_recipe,
        "effective_recipe_sha256": _digest(effective_recipe),
        "training": {
            "sample_dose": SHORT_SAMPLE_DOSE,
            "optimizer_steps": SHORT_OPTIMIZER_STEPS,
            "world_size": WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "grad_accum_steps": 1,
            "amp": "none",
            "optimizer": "fresh_adam",
            "resume_optimizer": False,
            "sampler_seed": FINAL_SAMPLER_SEED,
            "sample_order_sha256": sampling["sample_order_sha256"],
        },
        "allocation": copy.deepcopy(experiment["allocations"][ARM_CONTROL]),
        "diagnostic_only": False,
        "promotion_eligible_after_full_gate": True,
        "auto_promotion": False,
        "full_gate_required": True,
    }
    final_replication_id = _digest(authority)
    payload = {**authority, "final_replication_id": final_replication_id}
    directory = _artifact_dir(root, experiment_id, create=False)
    _write_once(
        directory.parent / "a1-final-replication-v1-issuance.json",
        {
            "schema_version": "a1-global-experiment-issuance-v1",
            "authority_key": "a1-final-replication-v1",
            "experiment_id": final_replication_id,
        },
    )
    return _write_once(directory / "90-final-issued.json", payload)


def claim_final_replication(
    root: Path,
    experiment_id: str,
    *,
    observed_allocation: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    final = _artifact(root, experiment_id, "90-final-issued.json")
    assert final is not None
    observed = verify_allocation(dict(observed_allocation))
    if observed != final["allocation"]:
        raise CoordinatorError("FINAL host/GPU allocation drift")
    payload = {
        "schema_version": "a1-final-replication-claim-v1",
        "experiment_id": experiment_id,
        "final_replication_id": final["final_replication_id"],
        "prior_authority_sha256": final["state_sha256"],
        "allocation": observed,
        "execution": _verify_execution(dict(execution)),
    }
    directory = _artifact_dir(root, experiment_id, create=False)
    return _write_once(directory / "92-final-claim.json", payload)


def load_final_replication_executor_authority(
    root: Path,
    experiment_id: str,
    *,
    observed_allocation: Mapping[str, Any],
) -> dict[str, Any]:
    final = _artifact(root, experiment_id, "90-final-issued.json")
    claim = _artifact(root, experiment_id, "92-final-claim.json")
    assert final is not None and claim is not None
    observed = verify_allocation(dict(observed_allocation))
    if (
        observed != final["allocation"]
        or claim["allocation"] != observed
        or claim["prior_authority_sha256"] != final["state_sha256"]
    ):
        raise CoordinatorError("FINAL executor allocation/claim drift")
    payload = {
        "schema_version": FINAL_EXECUTOR_AUTHORITY_SCHEMA,
        "final_replication_authority": copy.deepcopy(final),
        "final_claim": copy.deepcopy(claim),
        "allocation": observed,
    }
    payload["authority_sha256"] = _digest(payload)
    directory = _artifact_dir(root, experiment_id, create=False)
    return _write_once(directory / "93-final-executor-authority.json", payload)


def _immutable_file_snapshot(
    paths: Iterable[Path],
) -> dict[str, tuple[str, tuple[int, int, int, int, int]]]:
    snapshot = {}
    for path in paths:
        _payload, sha256, identity = _stable_read_immutable_json(
            path, where=f"executor predecessor {path.name}"
        )
        snapshot[str(path)] = (sha256, identity)
    return snapshot


def _verify_global_issuance(
    path: Path, *, authority_key: str, experiment_id: str
) -> None:
    payload, _sha256, _identity = _stable_read_immutable_json(
        path, where=f"global issuance {authority_key}"
    )
    expected = {
        "schema_version": "a1-global-experiment-issuance-v1",
        "authority_key": authority_key,
        "experiment_id": experiment_id,
        "state_sha256": payload.get("state_sha256"),
    }
    if payload != expected:
        raise CoordinatorError(f"global issuance drift for {authority_key}")


def verify_published_executor_authority(path: Path) -> dict[str, Any]:
    """Replay one immutable executor authority from its central transaction DAG.

    The learner consumes this path+file digest, never caller-authored inline JSON.
    Reconstructing the expected artifact through the coordinator also proves the
    claim, allocation, and predecessor artifacts still match before optimization.
    """

    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise CoordinatorError(f"cannot resolve executor authority: {error}") from error
    if (
        resolved != lexical
        or resolved.is_symlink()
        or not resolved.is_file()
        or resolved.stat().st_mode & 0o222
    ):
        raise CoordinatorError("executor authority must be canonical immutable file")
    authority, before_sha, before_identity = _stable_read_immutable_json(
        resolved, where="published executor authority"
    )
    schema = authority.get("schema_version")
    root = resolved.parent.parent
    predecessor_paths: list[Path]
    if schema == "a1-p1-arm-executor-authority-v1":
        sweep_id = authority.get("sweep_id")
        arm_id = authority.get("arm_id")
        expected_name = f"p1-15-{str(arm_id).lower()}-executor-authority.json"
        if resolved.name != expected_name:
            raise CoordinatorError("P1 executor authority filename drift")
        predecessor_paths = [
            root / "a1-p1-central-kl-v1-issuance.json",
            resolved.parent / "p1-00-sweep.json",
            resolved.parent / f"p1-10-{str(arm_id).lower()}-claim.json",
            resolved,
        ]
        predecessor_snapshot = _immutable_file_snapshot(predecessor_paths)
        _verify_global_issuance(
            predecessor_paths[0],
            authority_key="a1-p1-central-kl-v1",
            experiment_id=str(sweep_id),
        )
        replay = load_p1_arm_executor_authority(
            root,
            str(sweep_id),
            arm_id=str(arm_id),
            observed_allocation=authority.get("allocation", {}),
        )
    elif schema in {
        "a1-aux-warmup-executor-authority-v1",
        "a1-aux-geometry-executor-authority-v1",
    }:
        experiment_id = str(authority.get("experiment_id"))
        stage = str(authority.get("stage"))
        expected_name = (
            "15-warmup-executor-authority.json"
            if stage == "WARMUP"
            else "35-geometry-executor-authority.json"
        )
        if (
            stage not in {"WARMUP", "GEOMETRY"}
            or resolved.name != expected_name
        ):
            raise CoordinatorError("stage executor authority filename/stage drift")
        predecessor_paths = [
            root / "a1-pointer-aux-v1-issuance.json",
            resolved.parent / "00-experiment.json",
            resolved.parent
            / ("10-warmup-claim.json" if stage == "WARMUP" else "20-warmup-terminal.json"),
            resolved.parent
            / ("15-warmup-executor-authority.json" if stage == "WARMUP" else "30-geometry-claim.json"),
            resolved,
        ]
        predecessor_snapshot = _immutable_file_snapshot(predecessor_paths)
        _verify_global_issuance(
            predecessor_paths[0],
            authority_key="a1-pointer-aux-v1",
            experiment_id=experiment_id,
        )
        loader = (
            load_warmup_executor_authority
            if stage == "WARMUP"
            else load_geometry_executor_authority
        )
        replay = loader(
            root,
            experiment_id,
            observed_allocation=authority.get("allocation", {}),
        )
    elif schema == EXECUTOR_AUTHORITY_SCHEMA:
        pair = authority.get("aux_pair_contract")
        arm = authority.get("arm")
        if not isinstance(pair, dict) or not isinstance(arm, dict):
            raise CoordinatorError("AUX executor authority shape drift")
        experiment_id = pair.get("experiment_id")
        arm_id = arm.get("arm_id")
        expected_name = f"65-{str(arm_id).lower()}-executor-authority.json"
        if resolved.name != expected_name:
            raise CoordinatorError("AUX executor authority filename drift")
        predecessor_paths = [
            root / "a1-pointer-aux-v1-issuance.json",
            resolved.parent / "00-experiment.json",
            resolved.parent / "20-warmup-terminal.json",
            resolved.parent / "40-geometry-terminal.json",
            resolved.parent / "50-pair-issued.json",
            resolved.parent / f"60-{str(arm_id).lower()}-claim.json",
            resolved,
        ]
        predecessor_snapshot = _immutable_file_snapshot(predecessor_paths)
        _verify_global_issuance(
            predecessor_paths[0],
            authority_key="a1-pointer-aux-v1",
            experiment_id=str(experiment_id),
        )
        replay = load_aux_pair_executor_authority(
            root,
            str(experiment_id),
            arm_id=str(arm_id),
            observed_allocation=authority.get("arm_claim", {}).get("allocation", {}),
        )
    elif schema == FINAL_EXECUTOR_AUTHORITY_SCHEMA:
        final = authority.get("final_replication_authority")
        if not isinstance(final, dict) or resolved.name != (
            "93-final-executor-authority.json"
        ):
            raise CoordinatorError("FINAL executor authority shape/filename drift")
        predecessor_paths = [
            root / "a1-pointer-aux-v1-issuance.json",
            root / "a1-final-replication-v1-issuance.json",
            resolved.parent / "00-experiment.json",
            resolved.parent / "90-final-issued.json",
            resolved.parent / "92-final-claim.json",
            resolved,
        ]
        predecessor_snapshot = _immutable_file_snapshot(predecessor_paths)
        _verify_global_issuance(
            predecessor_paths[0],
            authority_key="a1-pointer-aux-v1",
            experiment_id=str(final.get("experiment_id")),
        )
        _verify_global_issuance(
            predecessor_paths[1],
            authority_key="a1-final-replication-v1",
            experiment_id=str(final.get("final_replication_id")),
        )
        replay = load_final_replication_executor_authority(
            root,
            str(final.get("experiment_id")),
            observed_allocation=authority.get("allocation", {}),
        )
    else:
        raise CoordinatorError("unsupported published executor authority schema")
    after_authority, after_sha, after_identity = _stable_read_immutable_json(
        resolved, where="published executor authority replay"
    )
    after_predecessors = _immutable_file_snapshot(predecessor_paths)
    if (
        replay != authority
        or after_authority != authority
        or after_sha != before_sha
        or after_identity != before_identity
        or after_predecessors != predecessor_snapshot
    ):
        raise CoordinatorError("published executor authority changed or failed replay")
    return {
        "schema_version": PUBLISHED_EXECUTOR_AUTHORITY_SCHEMA,
        "path": str(resolved),
        "file_sha256": before_sha,
        "authority": copy.deepcopy(authority),
    }


def complete_final_replication(
    root: Path,
    experiment_id: str,
    *,
    result: Mapping[str, Any],
    initializer_checkpoint_path: Path,
    candidate_checkpoint_path: Path,
    initializer_slot12_zero_receipt_path: Path,
    trained_slot12_delta_receipt_path: Path,
) -> dict[str, Any]:
    final = _artifact(root, experiment_id, "90-final-issued.json")
    claim = _artifact(root, experiment_id, "92-final-claim.json")
    assert final is not None and claim is not None
    result = _require_exact_keys(
        dict(result),
        {
            "schema_version",
            "status",
            "final_replication_id",
            "initializer_parent_checkpoint_sha256",
            "diagnostic_checkpoint_loaded",
            "selected_aux_decision",
            "selected_aux_coefficient_decimal",
            "pointer_upgrade_replayed",
            "pointer_upgrade_initializer_sha256",
            "warmed_checkpoint_sha256",
            "shared_warmup_initializer_consumed",
            "initializer_slot12_zero_receipt_state_sha256",
            "trained_slot12_delta_receipt_state_sha256",
            "candidate_slot12_finite",
            "candidate_slot12_nonzero_count",
            "learned_signal_observed",
            "component_routing_state_sha256",
            "sampling_state_sha256",
            "sampled_rows",
            "optimizer_steps",
            "world_size",
            "local_batch_size",
            "global_batch_size",
            "amp",
            "fresh_adam",
            "optimizer_restored",
            "effective_recipe_sha256",
            "sampler_seed",
            "sampler_identity_sha256",
            "sample_order_sha256",
            "row_set_sha256",
            "checkpoint_sha256",
            "optimizer_sidecar_sha256",
            "report_sha256",
            "origin_tool_sha256",
            "full_gate_entry_eligible",
        },
        "FINAL replication result",
    )
    initializer = final["initializer_authority"]
    treatment = final["selected_aux_decision"] == ARM_TREATMENT
    expected_warmup = (
        initializer["reference_warmup_terminal"]["result"]["warmed_checkpoint_sha256"]
        if treatment
        else None
    )
    sampling = final["sampling_receipt"]
    parent_sha = initializer["exact_current_parent_authority"]["checkpoint_sha256"]
    evidence_origin = _repo_tool_sha256("tools/a1_scientific_evidence.py")
    try:
        zero_receipt = scientific_evidence.verify_initializer_slot12_zero_receipt(
            initializer_slot12_zero_receipt_path.expanduser().resolve(strict=True),
            checkpoint=initializer_checkpoint_path.expanduser().resolve(strict=True),
            expected_origin_tool_sha256=evidence_origin,
        )
        delta_receipt = scientific_evidence.verify_trained_slot12_delta_receipt(
            trained_slot12_delta_receipt_path.expanduser().resolve(strict=True),
            initializer_checkpoint=initializer_checkpoint_path.expanduser().resolve(
                strict=True
            ),
            candidate_checkpoint=candidate_checkpoint_path.expanduser().resolve(
                strict=True
            ),
            expected_origin_tool_sha256=evidence_origin,
        )
    except (scientific_evidence.EvidenceError, OSError, ValueError) as error:
        raise CoordinatorError(f"FINAL model slot12 evidence refused: {error}") from error
    expected_initializer_sha = expected_warmup if treatment else parent_sha
    if (
        result["schema_version"] != "a1-final-replication-result-v1"
        or result["status"] != "complete"
        or result["final_replication_id"] != final["final_replication_id"]
        or result["initializer_parent_checkpoint_sha256"] != parent_sha
        or result["diagnostic_checkpoint_loaded"] is not False
        or result["selected_aux_decision"] != final["selected_aux_decision"]
        or result["selected_aux_coefficient_decimal"]
        != final["selected_aux_coefficient_decimal"]
        or result["pointer_upgrade_replayed"] is not False
        or result["pointer_upgrade_initializer_sha256"] is not None
        or result["warmed_checkpoint_sha256"] != expected_warmup
        or result["shared_warmup_initializer_consumed"] is not treatment
        or initializer["exact_reference_warmup_bytes_reused"] is not treatment
        or initializer["warmup_initializer_role"]
        != (
            "shared_immutable_architecture_initializer" if treatment else None
        )
        or zero_receipt["initializer_checkpoint_sha256"]
        != expected_initializer_sha
        or delta_receipt["initializer_checkpoint_sha256"]
        != expected_initializer_sha
        or result["checkpoint_sha256"]
        != delta_receipt["candidate_checkpoint_sha256"]
        or delta_receipt["model_slot12_parameter_set_sha256"]
        != zero_receipt["model_slot12_parameter_set_sha256"]
        or delta_receipt["model_slot12_parameter_count"]
        != zero_receipt["model_slot12_parameter_count"]
        or result["initializer_slot12_zero_receipt_state_sha256"]
        != zero_receipt["state_sha256"]
        or result["trained_slot12_delta_receipt_state_sha256"]
        != delta_receipt["state_sha256"]
        or result["candidate_slot12_finite"] is not True
        or result["candidate_slot12_finite"]
        != delta_receipt["candidate_slot12_finite"]
        or type(result["candidate_slot12_nonzero_count"]) is not int
        or result["candidate_slot12_nonzero_count"] <= 0
        or result["candidate_slot12_nonzero_count"]
        != delta_receipt["candidate_slot12_nonzero_count"]
        or result["learned_signal_observed"] is not True
        or delta_receipt["learned_signal_observed"] is not True
        or result["learned_signal_observed"]
        != delta_receipt["learned_signal_observed"]
        or result["component_routing_state_sha256"]
        != final["component_routing_state_sha256"]
        or result["sampling_state_sha256"] != final["sampling_state_sha256"]
        or result["sampled_rows"] != SHORT_SAMPLE_DOSE
        or result["optimizer_steps"] != SHORT_OPTIMIZER_STEPS
        or result["world_size"] != WORLD_SIZE
        or result["local_batch_size"] != LOCAL_BATCH_SIZE
        or result["global_batch_size"] != GLOBAL_BATCH_SIZE
        or result["amp"] != "none"
        or result["fresh_adam"] is not True
        or result["optimizer_restored"] is not False
        or result["effective_recipe_sha256"] != final["effective_recipe_sha256"]
        or result["sampler_seed"] != FINAL_SAMPLER_SEED
        or result["sampler_identity_sha256"] != sampling["sampler_identity_sha256"]
        or result["sample_order_sha256"] != sampling["sample_order_sha256"]
        or result["row_set_sha256"] != sampling["row_set_sha256"]
        or result["origin_tool_sha256"]
        != _repo_tool_sha256("tools/a1_one_dose_train.py")
        or result["full_gate_entry_eligible"] is not True
    ):
        raise CoordinatorError("FINAL result drifted from promotion-safe replication")
    for field in (
        "checkpoint_sha256",
        "optimizer_sidecar_sha256",
        "report_sha256",
        "origin_tool_sha256",
        "initializer_slot12_zero_receipt_state_sha256",
        "trained_slot12_delta_receipt_state_sha256",
    ):
        _require_sha(result[field], f"FINAL result {field}")
    forbidden_checkpoints = {parent_sha}
    pair = _artifact(root, experiment_id, "50-pair-issued.json")
    assert pair is not None
    for arm in ARMS:
        terminal = _artifact(root, experiment_id, f"70-{arm.lower()}-terminal.json")
        assert terminal is not None
        forbidden_checkpoints.add(terminal["result"]["checkpoint_sha256"])
    p1 = final["diagnostic_p1_selection_authority"]
    for arm in P1_ARMS:
        terminal = _artifact(root, p1["sweep_id"], f"p1-20-{arm.lower()}-terminal.json")
        assert terminal is not None
        forbidden_checkpoints.add(terminal["result"]["checkpoint_sha256"])
    if result["checkpoint_sha256"] in forbidden_checkpoints:
        raise CoordinatorError("FINAL output reused a diagnostic/parent checkpoint")
    payload = {
        "schema_version": "a1-final-replication-terminal-v1",
        "experiment_id": experiment_id,
        "final_replication_id": final["final_replication_id"],
        "prior_authority_sha256": claim["state_sha256"],
        "result": copy.deepcopy(result),
        "initializer_slot12_zero_receipt": copy.deepcopy(zero_receipt),
        "trained_slot12_delta_receipt": copy.deepcopy(delta_receipt),
        "diagnostic_only": False,
        "promotion_eligible": False,
        "eligible_for_full_gate": True,
        "full_gate_required": True,
        "auto_promotion": False,
    }
    directory = _artifact_dir(root, experiment_id, create=False)
    return _write_once(directory / "95-final-terminal.json", payload)


def load_final_gate_entry_authority(
    root: Path,
    experiment_id: str,
    *,
    recovery_gate_authority_path: Path,
) -> dict[str, Any]:
    """Admit FINAL only after the canonical fresh dual-baseline gate replays."""

    final = _artifact(root, experiment_id, "90-final-issued.json")
    terminal = _artifact(root, experiment_id, "95-final-terminal.json")
    assert final is not None and terminal is not None
    try:
        gate = v5_recovery_gate.verify_recovery_gate_authority(
            recovery_gate_authority_path
        )
    except (v5_recovery_gate.RecoveryGateError, OSError, ValueError) as error:
        raise CoordinatorError(f"FINAL recovery gate refused: {error}") from error
    p1 = final["diagnostic_p1_selection_authority"]
    recovery = p1["recovery_authority"]
    recovered = recovery["recovered_generator"]
    safety = recovery["safety_reference_unproven_predecessor"]
    candidate = gate.get("candidate")
    strict = gate.get("strict_h1_parent_gate")
    veto = gate.get("f7_non_regression_veto")
    policy = gate.get("policy")
    expected_policy = {
        "dual_baseline_conjunctive": True,
        "strict_h1_over_recovered_parent": True,
        "f7_h0_veto": True,
        "fresh_cohorts_required": True,
        "promotion_eligible": True,
        "auto_promotion": False,
    }
    if (
        gate.get("schema_version") != "a1-v5-recovery-full-gate-authority-v1"
        or gate.get("recovery_authority") != recovery
        or not isinstance(candidate, dict)
        or candidate.get("sha256") != terminal["result"]["checkpoint_sha256"]
        or not isinstance(strict, dict)
        or strict.get("passed") is not True
        or strict.get("baseline") != recovered
        or not isinstance(veto, dict)
        or veto.get("passed") is not True
        or veto.get("baseline") != safety
        or veto.get("verdict") not in {"H1", "continue"}
        or policy != expected_policy
    ):
        raise CoordinatorError("FINAL gate is not the exact recovery dual-baseline gate")
    _require_sha(gate.get("authority_sha256"), "FINAL recovery gate authority")
    payload = {
        "schema_version": FINAL_GATE_ENTRY_SCHEMA,
        "experiment_id": experiment_id,
        "final_replication_id": final["final_replication_id"],
        "final_authority_state_sha256": final["state_sha256"],
        "final_terminal_state_sha256": terminal["state_sha256"],
        "candidate_checkpoint_sha256": terminal["result"]["checkpoint_sha256"],
        "recovered_parent_checkpoint_sha256": recovered["sha256"],
        "f7_safety_checkpoint_sha256": safety["sha256"],
        "recovery_gate_authority": copy.deepcopy(gate),
        "recovery_gate_authority_sha256": gate["authority_sha256"],
        "promotion_eligible": True,
        "auto_promotion": False,
        "full_gate_satisfied": True,
    }
    payload["authority_sha256"] = _digest(payload)
    return payload


def inspect_state(root: Path, experiment_id: str) -> dict[str, Any]:
    """Return a read-only resumability snapshot; absence is never failure."""

    load_experiment(root, experiment_id)
    filenames = (
        "10-warmup-claim.json",
        "20-warmup-terminal.json",
        "30-geometry-claim.json",
        "40-geometry-terminal.json",
        "50-pair-issued.json",
        "60-aux0-claim.json",
        "60-auxt-claim.json",
        "70-aux0-terminal.json",
        "70-auxt-terminal.json",
        "75-evaluation-claim.json",
        "77-evaluation-terminal.json",
        "80-pair-terminal.json",
        "90-final-issued.json",
        "92-final-claim.json",
        "95-final-terminal.json",
    )
    artifacts = {
        filename: _artifact(root, experiment_id, filename, required=False)
        for filename in filenames
    }
    return {
        "schema_version": "a1-aux-coordinator-state-v1",
        "experiment_id": experiment_id,
        "present": [name for name, value in artifacts.items() if value is not None],
        "warmup_claimed": artifacts["10-warmup-claim.json"] is not None,
        "warmup_terminal": artifacts["20-warmup-terminal.json"] is not None,
        "geometry_claimed": artifacts["30-geometry-claim.json"] is not None,
        "geometry_terminal": artifacts["40-geometry-terminal.json"] is not None,
        "pair_issued": artifacts["50-pair-issued.json"] is not None,
        "arms_claimed": [
            arm for arm in ARMS if artifacts[f"60-{arm.lower()}-claim.json"] is not None
        ],
        "arms_terminal": [
            arm
            for arm in ARMS
            if artifacts[f"70-{arm.lower()}-terminal.json"] is not None
        ],
        "pair_terminal": artifacts["80-pair-terminal.json"] is not None,
        "evaluation_claimed": artifacts["75-evaluation-claim.json"] is not None,
        "evaluation_terminal": artifacts["77-evaluation-terminal.json"] is not None,
        "final_issued": artifacts["90-final-issued.json"] is not None,
        "final_claimed": artifacts["92-final-claim.json"] is not None,
        "final_terminal": artifacts["95-final-terminal.json"] is not None,
    }


__all__ = [
    "ALLOCATION_SCHEMA",
    "ARM_CONTROL",
    "ARM_TREATMENT",
    "ARMS",
    "AUX_EVALUATION_PLAN_SCHEMA",
    "B200_LEARNER_GPU_UUIDS",
    "B200_LEARNER_HOSTNAME",
    "B200_LEARNER_MACHINE_ID",
    "B200_LEARNER_HOST_ID",
    "CoordinatorError",
    "EXECUTION_SCHEMA",
    "FINAL_EXECUTOR_AUTHORITY_SCHEMA",
    "FINAL_REPLICATION_SCHEMA",
    "FINAL_SAMPLER_SEED",
    "GEOMETRY_DOSE_AUTHORITY_SCHEMA",
    "HANDOFF_PARENT_AUTHORITY_SCHEMA",
    "NATIVE_CAPABILITIES",
    "NATIVE_RUNTIME_AUTHORITY_SCHEMA",
    "NATIVE_WHEEL_SHA256",
    "P1_ARMS",
    "P1_EVALUATION_PLAN_SCHEMA",
    "P1_FINAL_LOCK_AUTHORITY_SCHEMA",
    "P1_RECIPE_DATA_AUTHORITY_SCHEMA",
    "P1_SWEEP_SCHEMA",
    "PUBLISHED_EXECUTOR_AUTHORITY_SCHEMA",
    "POINTER_FLAGS",
    "POINTER_MODULE",
    "POINTER_TRAINABLE_PREFIXES",
    "POINTER_UPGRADE_AUTHORITY_SCHEMA",
    "SELECTOR_RULE_SCHEMA",
    "STALE_RECOVERY_PLAN_SCHEMA",
    "SHORT_OPTIMIZER_STEPS",
    "SHORT_SAMPLE_DOSE",
    "WARMUP_MAIN_OBJECTIVE_ZERO",
    "WARMUP_RECIPE_SCHEMA",
    "adjudicate_p1_sweep",
    "build_native_runtime_authority",
    "build_p1_kl_eligibility_authority",
    "canonical_aux_evaluation_plan",
    "current_parent_authority_from_recovery",
    "recovery_component_semantics",
    "canonical_geometry_dose_authority",
    "canonical_p1_final_lock_authority",
    "canonical_p1_evaluation_plan",
    "canonical_p1_row_identity",
    "canonical_p1_sample_order_sha256",
    "claim_arm",
    "claim_geometry",
    "claim_p1_arm",
    "claim_p1_evaluation",
    "claim_pair_evaluation",
    "claim_final_replication",
    "claim_warmup",
    "complete_arm",
    "complete_geometry",
    "complete_p1_arm",
    "complete_p1_evaluation",
    "complete_pair_evaluation",
    "complete_final_replication",
    "complete_warmup",
    "finalize_pair",
    "inspect_state",
    "issue_final_replication",
    "issue_pair",
    "load_aux_pair_executor_authority",
    "load_experiment",
    "load_final_gate_entry_authority",
    "load_final_replication_executor_authority",
    "load_geometry_executor_authority",
    "load_p1_arm_executor_authority",
    "load_p1_sweep",
    "load_selected_p1_recipe_data_authority",
    "load_warmup_executor_authority",
    "prepare_experiment",
    "prepare_p1_sweep",
    "verify_aux_evaluation_plan",
    "verify_allocation",
    "verify_current_parent_authority",
    "verify_native_runtime_authority",
    "verify_p1_evaluation_plan",
    "verify_p1_final_lock_authority",
    "verify_p1_kl_eligibility_authority",
    "verify_p1_recipe_data_authority",
    "verify_pointer_upgrade_authority",
    "verify_published_executor_authority",
    "verify_selector_rule",
    "verify_warmup_recipe",
    "verify_v5_recovery_receipt",
]
