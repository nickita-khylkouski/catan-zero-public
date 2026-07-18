#!/usr/bin/env python3
"""Seal policy-target quarantine and coherent-n128 reanalysis evidence.

Policy targets are valid only for the exact checkpoint/search operator that
produced them.  Public states and realised outcomes have a longer lifetime.
This tool makes that distinction executable without mutating the source
corpus:

* authenticate the source corpus admission and source/target operator identity;
* write immutable row sidecars for stored-policy eligibility, quarantine,
  value retention, and reanalysis candidacy;
* inventory duplicate-search reliability without treating neutral unaudited
  sentinels as measured confidence;
* choose a deterministic, phase/width/surprise/reliability-stratified subset;
* seal a current coherent-n128 reanalysis plan and its exact blockers.

The current repaired corpus intentionally omitted automatic UI transitions,
so its admission may be valid for stored target distillation while still being
unready for exact state reconstruction.  A plan remains useful in that state:
it seals the target identity and selected roots, but advertises
``execution_ready=false`` rather than silently invoking the legacy PIMC
reanalyzer or conditioning on hidden truth.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zipfile
import io
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tools import a1_target_eligibility_inventory as target_inventory  # noqa: E402
from tools import a1_current_science_contract as current_science  # noqa: E402
from tools import train_bc  # noqa: E402
from catan_zero.rl.target_reliability import (  # noqa: E402
    TARGET_RELIABILITY_COLUMNS,
    TARGET_RELIABILITY_CONFIDENCE_FORMULA,
    TARGET_RELIABILITY_SCHEMA,
    TARGET_RELIABILITY_VERSION,
    target_reliability_confidence,
)
from catan_zero.search.rng_streams import SEARCH_RNG_STREAM_SCHEMA  # noqa: E402
from catan_zero.search.gumbel_chance_mcts import (  # noqa: E402
    GumbelChanceMCTSConfig,
)
from catan_zero.search.neural_rust_mcts import (  # noqa: E402
    EntityGraphRustEvaluatorConfig,
)
from catan_zero.rl.gumbel_self_play import ACTION_MASK_VERSION  # noqa: E402
from catan_zero.rl.entity_feature_adapter import (  # noqa: E402
    ENTITY_FEATURE_ADAPTER_CHECKPOINT_SCHEMA,
    RUST_ENTITY_ADAPTER_V6,
)


PLAN_SCHEMA = "a1-stage-c-teacher-alignment-plan-v3"
OVERLAY_SCHEMA = "a1-stage-c-target-eligibility-overlay-v1"
SUBSET_SCHEMA = "a1-stage-c-reanalysis-subset-v2"
COHERENT_REGIME = "public_belief_single_tree_v1"
OPERATOR_IDENTITY_SCHEMA_V1 = "a1-operator-bound-policy-target-identity-v1"
OPERATOR_IDENTITY_SCHEMA_V2 = "a1-operator-bound-policy-target-identity-v2"
OPERATOR_IDENTITY_SCHEMA_V3 = "a1-operator-bound-policy-target-identity-v3"
OPERATOR_IDENTITY_SCHEMA_V4 = "a1-operator-bound-policy-target-identity-v4"
RD_TEACHER_TRANSITION_BINDING_SCHEMA = "a1-rd-teacher-transition-binding-v1"
POST_WAVE_SOURCE_BINDING_SCHEMA = "a1-post-wave-source-operator-binding-v1"
PAIRED_ROOT_VALUE_OUTPUT_SCHEMA = "a1-paired-root-value-output-contract-v1"
STAGE_C_ROW_SEED_SCHEMA = "a1-stage-c-coherent-reanalysis-root-seed-v1"
STAGE_C_TARGET_EXECUTION = {
    "schema_version": "a1-stage-c-target-execution-v1",
    "mode": "forced_full_root_reanalysis",
    "force_full_override": True,
    "nominal_n_full": 128,
    "actual_simulations": "authoritative_per_row_deterministic_schedule_result",
    "simulation_accounting_schema": (
        "gumbel_root_candidate_count_plus_sequential_halving_v1"
    ),
    "budget_semantics": (
        "force_full selects n_full; legacy Sequential-Halving schedule accounting "
        "can realize a legal-width-dependent count different from nominal_n_full"
    ),
    "row_seed_schema": STAGE_C_ROW_SEED_SCHEMA,
}
POLICY_STATUS = {
    "inactive_no_stored_policy": 0,
    "eligible_exact_operator": 1,
    "quarantined_stale_operator": 2,
}
RELIABILITY_CLASS = {
    "not_collected": 0,
    "unaudited_neutral_sentinel": 1,
    "duplicate_search_audited": 2,
}
ROOT_BREADTH_SCHEMA = "a1-stage-c-policy-root-breadth-v1"
GAME_TRACE_QUALIFICATION_SCHEMA = "a1-stage-c-game-trace-qualification-v1"
PRODUCTION_ROOT_COUNT = 65_536
LEGACY_CORPUS_ADMISSION_SCHEMA = "a1-coherent-n128-corpus-admission-v1"
POST_WAVE_CORPUS_ADMISSION_SCHEMA = "a1-post-wave-stage-c-corpus-admission-v1"
LEARNER_VALIDATION_SCOPE_SCHEMA = "a1-stage-c-learner-validation-scope-v2"
TRAINER_EXCLUSION_CONTRACT_SCHEMA = "a1-stage-c-trainer-exclusion-contract-v1"
ROOT_BREADTH_REQUIRED_PHASES = (
    "BUILD_INITIAL_ROAD",
    "BUILD_INITIAL_SETTLEMENT",
    "DISCARD",
    "MOVE_ROBBER",
    "PLAY_TURN",
)
ROOT_BREADTH_DECISION_BINS = (
    ("d000_009", 0, 10),
    ("d010_029", 10, 30),
    ("d030_059", 30, 60),
    ("d060_099", 60, 100),
    ("d100_149", 100, 150),
    ("d150_199", 150, 200),
    ("d200_plus", 200, None),
)
ROOT_BREADTH_CONTRACT = {
    "minimum_unique_game_fraction": 0.95,
    "minimum_roots_per_represented_game": 8,
    "minimum_phase_fraction": 0.01,
    "minimum_decision_bin_fraction": 0.01,
    "required_phases": list(ROOT_BREADTH_REQUIRED_PHASES),
    "decision_index_bins": [
        {
            "name": name,
            "start_inclusive": start,
            "stop_exclusive": stop,
        }
        for name, start, stop in ROOT_BREADTH_DECISION_BINS
    ],
    "required_scopes": ["training", "validation"],
}


def _minimum_stage_c_root_budget(
    *,
    population_game_seeds: np.ndarray,
    validation_game_seeds: np.ndarray,
) -> int:
    """Return the deterministic breadth floor for this admitted game population."""
    population = np.unique(np.asarray(population_game_seeds, dtype=np.int64))
    validation = np.unique(np.asarray(validation_game_seeds, dtype=np.int64))
    if not len(population) or not len(validation):
        raise AlignmentError(
            "Stage-C training and validation populations must be nonempty"
        )
    if np.setdiff1d(validation, population).size:
        raise AlignmentError("Stage-C validation game is outside the corpus")
    training = np.setdiff1d(population, validation)
    if not len(training):
        raise AlignmentError("Stage-C training game population is empty")
    minimum_fraction = float(ROOT_BREADTH_CONTRACT["minimum_unique_game_fraction"])
    minimum_roots = int(ROOT_BREADTH_CONTRACT["minimum_roots_per_represented_game"])
    required_games = math.ceil(minimum_fraction * len(training)) + math.ceil(
        minimum_fraction * len(validation)
    )
    return minimum_roots * required_games


def _resolve_stage_c_root_budget(
    *,
    requested_rows: int | None,
    admission_schema: str,
    population_game_seeds: np.ndarray,
    validation_game_seeds: np.ndarray,
) -> int:
    """Resolve the exact root count without changing the legacy production contract."""
    if admission_schema == LEGACY_CORPUS_ADMISSION_SCHEMA:
        resolved = (
            PRODUCTION_ROOT_COUNT
            if requested_rows is None
            else int(requested_rows)
        )
        if resolved != PRODUCTION_ROOT_COUNT:
            raise AlignmentError(
                "legacy production Stage-C plans require exactly "
                f"{PRODUCTION_ROOT_COUNT:,} requested roots"
            )
        return resolved
    if admission_schema != POST_WAVE_CORPUS_ADMISSION_SCHEMA:
        raise AlignmentError(
            f"unsupported coherent corpus admission schema {admission_schema!r}"
        )

    minimum = _minimum_stage_c_root_budget(
        population_game_seeds=population_game_seeds,
        validation_game_seeds=validation_game_seeds,
    )
    resolved = minimum if requested_rows is None else int(requested_rows)
    if resolved < minimum:
        raise AlignmentError(
            "post-wave production Stage-C root budget cannot satisfy admitted "
            f"breadth: requested={resolved:,} required_at_least={minimum:,}"
        )
    return resolved


def _qualify_stage_c_game_traces(
    *,
    game_seeds: np.ndarray,
    decision_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Classify traces that can replay from the canonical seeded initial state."""

    games = np.asarray(game_seeds, dtype=np.int64)
    decisions = np.asarray(decision_indices, dtype=np.int64)
    if games.ndim != 1 or decisions.shape != games.shape or not len(games):
        raise AlignmentError("Stage-C game-trace qualification inputs are malformed")
    if np.any(games[1:] < games[:-1]):
        raise AlignmentError("Stage-C corpus game_seed rows are not monotonically grouped")

    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(games[1:] != games[:-1]).astype(np.int64) + 1,
        )
    )
    stops = np.concatenate((starts[1:], np.asarray([len(games)], dtype=np.int64)))
    qualified: list[int] = []
    excluded: list[int] = []
    reason_counts = {
        "missing_initial_decision_prefix": 0,
        "negative_decision_index": 0,
        "non_increasing_decision_index": 0,
    }
    examples: list[dict[str, Any]] = []
    for start, stop in zip(starts.tolist(), stops.tolist(), strict=True):
        seed = int(games[start])
        game_decisions = decisions[start:stop]
        if np.any(game_decisions < 0):
            reason = "negative_decision_index"
        elif int(game_decisions[0]) != 0:
            reason = "missing_initial_decision_prefix"
        elif np.any(game_decisions[1:] <= game_decisions[:-1]):
            reason = "non_increasing_decision_index"
        else:
            qualified.append(seed)
            continue
        excluded.append(seed)
        reason_counts[reason] += 1
        if len(examples) < 100:
            examples.append(
                {
                    "game_seed": seed,
                    "reason": reason,
                    "first_decision_index": int(game_decisions[0]),
                    "recorded_row_count": int(stop - start),
                }
            )

    qualified_array = np.asarray(qualified, dtype=np.int64)
    excluded_array = np.asarray(excluded, dtype=np.int64)
    receipt: dict[str, Any] = {
        "schema_version": GAME_TRACE_QUALIFICATION_SCHEMA,
        "contract": {
            "canonical_replay_start": "seeded_initial_state_at_decision_0",
            "decision_indices": "unique_nonnegative_strictly_increasing",
            "missing_initial_prefix_semantics": (
                "unreconstructable: 2p-no-trade decision 0 is multi-action "
                "BUILD_INITIAL_SETTLEMENT, never an automatic transition"
            ),
            "later_sparse_gaps": (
                "retained; executor must prove each omitted transition has exactly "
                "one legal action"
            ),
        },
        "total_games": int(len(starts)),
        "qualified_games": int(len(qualified_array)),
        "excluded_games": int(len(excluded_array)),
        "exclusion_counts": reason_counts,
        "qualified_game_seed_set_sha256": _value_sha256(
            qualified_array.astype(int).tolist()
        ),
        "excluded_game_seed_set_sha256": _value_sha256(
            excluded_array.astype(int).tolist()
        ),
        "exclusion_examples": examples,
    }
    receipt["qualification_sha256"] = _value_sha256(receipt)
    return qualified_array, receipt
SEARCH_FIELDS = (
    "n_full",
    "n_fast",
    "p_full",
    "n_full_wide",
    "n_full_wide_threshold",
    "wide_roots_always_full",
    "c_visit",
    "c_scale",
    "sigma_eval",
    "max_depth",
    "prior_temperature",
    "exact_budget_sh",
    "exact_budget_sh_min_n",
)
BELIEF_FIELDS = (
    "coherent_public_belief_search",
    # K changes opponent/new-turn continuation backups and therefore the
    # stored policy target. It is not a GumbelChanceMCTSConfig field, so it
    # must be bound here rather than relying on effective_gumbel_config.
    "boundary_value_particles",
    "information_set_search",
    "determinization_particles",
    "determinization_min_simulations",
    "belief_chance_spectra",
    "information_set_target_aggregation",
)
CHANCE_FIELDS = (
    "correct_rust_chance_spectra",
    "lazy_interior_chance",
)
SYMMETRY_FIELDS = (
    "symmetry_averaged_eval",
    "symmetry_averaged_eval_threshold",
)
TARGET_SEMANTIC_FIELDS = (
    "public_observation",
    "forced_root_target_mode",
    "record_automatic_transitions",
    "meaningful_public_history",
    "event_history_limit",
    "public_card_count_feature_schema",
    "temperature_clock",
    "preserve_search_evidence",
)


class AlignmentError(RuntimeError):
    """A target identity, corpus, reliability field, or plan is invalid."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _value_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _regular_file(path: Path, *, where: str) -> Path:
    lexical = path.expanduser()
    if lexical.is_symlink() or not lexical.is_file():
        raise AlignmentError(f"{where} must be a regular file: {lexical}")
    try:
        return lexical.resolve(strict=True)
    except OSError as error:
        raise AlignmentError(f"cannot resolve {where}: {error}") from error


def _load_json(path: Path, *, where: str) -> tuple[Path, dict[str, Any]]:
    resolved = _regular_file(path, where=where)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AlignmentError(f"cannot load {where}: {error}") from error
    if not isinstance(payload, dict):
        raise AlignmentError(f"{where} must contain one JSON object")
    return resolved, payload


def _write_immutable(path: Path, payload: bytes) -> None:
    destination = path.expanduser().resolve(strict=False)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise AlignmentError(
                f"immutable output is not a regular file: {destination}"
            )
        if destination.read_bytes() != payload:
            raise AlignmentError(
                f"immutable output already exists with drift: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)


def _write_json_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    _write_immutable(
        path, json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"
    )


def _write_array_immutable(path: Path, array: np.ndarray) -> None:
    value = np.ascontiguousarray(array)
    _write_immutable(path, value.tobytes(order="C"))


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for key in sorted(arrays):
            if "/" in key or "\\" in key:
                raise AlignmentError(f"unsafe subset column name: {key!r}")
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.asarray(arrays[key]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    return output.getvalue()


def _resolve_artifact(contract_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        # Operational contracts live three directories below the repository.
        path = contract_path.parents[3] / path
    return _regular_file(path, where="operator artifact")


def _field_bundle(fields: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    missing = [name for name in names if name not in fields]
    if missing:
        raise AlignmentError(f"typed operator config is missing {missing}")
    return {name: fields[name] for name in names}


def _semantic_field_bundle(
    fields: Mapping[str, Any],
    operator: Mapping[str, Any],
    names: Sequence[str],
) -> dict[str, Any]:
    """Bind generation semantics from either of the two sealed authorities.

    Schema-13 generation configs predate ``preserve_search_evidence`` as a
    typed field, while the coherent R&D contract already binds that value in
    its hash-authenticated ``operator`` block.  Refusing that exact contract
    makes its Stage-C target identity impossible to construct; inventing a
    default would be worse.  A semantic is therefore admitted only when it is
    explicit in the typed config or, for a missing typed field, explicit in
    the operator block.  Existing overlap-drift checking above still rejects
    any disagreement between the two authorities.
    """

    missing = [name for name in names if name not in fields and name not in operator]
    if missing:
        raise AlignmentError(
            f"typed config and sealed operator are both missing {missing}"
        )
    return {name: fields[name] if name in fields else operator[name] for name in names}


def _json_normalized(value: object) -> Any:
    """Return the exact JSON-domain representation used by sealed receipts."""

    return json.loads(_canonical_bytes(value))


def _complete_effective_search_config(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve every Gumbel field that can affect a root target.

    The typed generation schema intentionally omits defaults.  Hashing only its
    present keys allowed a newly added/defaulted search knob to change targets
    without changing target identity.  Resolve the live dataclass here and bind
    every field except ``seed``.  The seed is the sole exclusion: it identifies
    a stochastic replicate, not the scientific operator, and Stage C binds its
    deterministic per-row derivation separately in ``target_execution``.

    Execution/fleet layout is outside this dataclass.  Performance toggles that
    can alter evaluation order (including root-wave batching and batch API use)
    are deliberately retained rather than assumed numerically invisible.
    """

    allowed = {field.name for field in dataclasses.fields(GumbelChanceMCTSConfig)}
    kwargs = {name: value for name, value in fields.items() if name in allowed}
    if "colors" in kwargs:
        kwargs["colors"] = tuple(str(value) for value in kwargs["colors"])
    try:
        resolved = dataclasses.asdict(GumbelChanceMCTSConfig(**kwargs))
    except (TypeError, ValueError) as error:
        raise AlignmentError(
            f"cannot resolve effective Gumbel config: {error}"
        ) from error
    resolved.pop("seed")
    return _json_normalized(resolved)


def _complete_effective_evaluator_config(
    fields: Mapping[str, Any],
    *,
    entity_feature_adapter_version: str | None = None,
) -> dict[str, Any]:
    """Resolve target-changing evaluator fields and document the cache exclusion."""

    kwargs = {
        "value_scale": float(fields.get("value_scale", 1.0)),
        "prior_temperature": float(fields.get("prior_temperature", 1.0)),
        "value_squash": str(fields.get("value_squash", "tanh")),
        "value_readout": str(fields.get("value_readout", "scalar")),
        "public_observation": bool(fields.get("public_observation", False)),
        "rust_featurize": bool(fields.get("rust_featurize", True)),
        "entity_feature_adapter_version": entity_feature_adapter_version,
    }
    resolved = dataclasses.asdict(EntityGraphRustEvaluatorConfig(**kwargs))
    # Cache capacity changes storage/layout only.  Every other resolved field
    # affects features, values, priors, or emitted uncertainty and stays bound.
    resolved.pop("cache_size")
    return _json_normalized(resolved)


def _paired_root_value_output_contract(
    fields: Mapping[str, Any],
    operator: Mapping[str, Any],
) -> dict[str, Any]:
    science = current_science.load()
    generation = science.get("generation")
    if not isinstance(generation, Mapping):
        raise AlignmentError("current science lacks a generation contract")
    preservation = generation.get("preserve_root_prior_value")
    if not isinstance(preservation, bool):
        raise AlignmentError(
            "current science does not explicitly bind preserve_root_prior_value"
        )
    target_bindings: list[bool] = []
    for label, source in (
        ("typed generation config", fields),
        ("sealed operator", operator),
    ):
        if "preserve_root_prior_value" not in source:
            continue
        value = source["preserve_root_prior_value"]
        if type(value) is not bool:
            raise AlignmentError(
                f"{label} preserve_root_prior_value must be boolean"
            )
        target_bindings.append(value)
    if target_bindings and any(value is not preservation for value in target_bindings):
        raise AlignmentError(
            "target authority disagrees with current paired-root output contract"
        )
    science_path = current_science.CONTRACT_PATH.resolve(strict=True)
    return {
        "schema_version": PAIRED_ROOT_VALUE_OUTPUT_SCHEMA,
        "root_value_semantics": "post_search_root_value",
        "root_prior_value_semantics": "pre_search_root_evaluator_value",
        "preserve_root_prior_value": preservation,
        "atomic_pair_required": True,
        "authority": {
            "schema_version": science["schema_version"],
            "contract_id": science["contract_id"],
            "path": str(science_path),
            "file_sha256": _file_sha256(science_path),
        },
    }


def _rd_teacher_transition_authority(
    binding_path: Path,
    binding: Mapping[str, Any],
    checkpoint: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Resolve one non-promotable V6 checkpoint into a Stage-C authority.

    The production coherent-n128 contract remains the authority for search
    semantics.  This binding replaces only the producer network and its
    explicitly checked feature adapter for root reanalysis.  It cannot be used
    by the production generation executor because it has a distinct schema and
    carries no seed/fleet schedule.
    """

    unsigned = dict(binding)
    stated = unsigned.pop("binding_sha256", None)
    if (
        binding.get("schema_version") != RD_TEACHER_TRANSITION_BINDING_SCHEMA
        or stated != _value_sha256(unsigned)
        or binding.get("diagnostic_only") is not True
        or binding.get("promotion_eligible") is not False
        or binding.get("production_authority") is not False
        or binding.get("status") != "ready_nonpromotable_reanalysis_teacher"
    ):
        raise AlignmentError("R&D teacher-transition binding is not fail-closed")

    base_ref = binding.get("base_operator_contract")
    typed_ref = binding.get("typed_generation_config")
    producer = binding.get("producer_checkpoint")
    feature_contract = binding.get("teacher_feature_contract")
    if not all(
        isinstance(value, Mapping)
        for value in (base_ref, typed_ref, producer, feature_contract)
    ):
        raise AlignmentError("R&D teacher-transition binding is incomplete")

    base_path = _regular_file(
        Path(str(base_ref.get("path", ""))),
        where="R&D teacher base operator contract",
    )
    if _file_sha256(base_path) != base_ref.get("file_sha256"):
        raise AlignmentError("R&D teacher base operator contract bytes drifted")
    try:
        inspected = target_inventory.inspect_rd_contract(base_path)
    except (target_inventory.InventoryError, OSError, ValueError) as error:
        raise AlignmentError(f"R&D teacher base operator refused: {error}") from error
    if (
        inspected.get("contract_sha256") != base_ref.get("contract_sha256")
        or inspected.get("target_information_regime") != COHERENT_REGIME
    ):
        raise AlignmentError("R&D teacher base operator identity drifted")
    _base_path, base = _load_json(base_path, where="R&D teacher base operator")

    config_path = _regular_file(
        Path(str(typed_ref.get("path", ""))),
        where="R&D teacher typed generation config",
    )
    if _file_sha256(config_path) != typed_ref.get("file_sha256"):
        raise AlignmentError("R&D teacher typed generation config bytes drifted")
    _config_path, config = _load_json(
        config_path, where="R&D teacher typed generation config"
    )
    fields = config.get("fields")
    if (
        config.get("pipeline") != "generate"
        or config.get("schema_version") != 13
        or not isinstance(fields, Mapping)
    ):
        raise AlignmentError("R&D teacher typed generation config is malformed")

    checkpoint = _regular_file(checkpoint, where="R&D teacher checkpoint")
    producer_path = _regular_file(
        Path(str(producer.get("path", ""))),
        where="bound R&D teacher checkpoint",
    )
    checkpoint_sha = _file_sha256(checkpoint)
    if (
        producer_path != checkpoint
        or producer.get("sha256") != checkpoint_sha
    ):
        raise AlignmentError("R&D teacher checkpoint bytes/path drifted")
    try:
        checkpoint_adapter = train_bc._checkpoint_entity_feature_adapter_version(  # noqa: SLF001
            str(checkpoint)
        )
    except (OSError, RuntimeError, SystemExit, ValueError) as error:
        raise AlignmentError(
            f"cannot authenticate R&D teacher checkpoint adapter: {error}"
        ) from error
    declared_adapter = feature_contract.get("entity_feature_adapter_version")
    if (
        feature_contract.get("schema_version")
        != ENTITY_FEATURE_ADAPTER_CHECKPOINT_SCHEMA
        or declared_adapter != RUST_ENTITY_ADAPTER_V6
        or checkpoint_adapter != declared_adapter
        or fields.get("teacher_entity_feature_adapter_version") != declared_adapter
        or fields.get("learner_entity_feature_adapter_version") != declared_adapter
    ):
        raise AlignmentError(
            "R&D teacher requires one exact V6 checkpoint/config adapter binding"
        )

    operator = base.get("operator")
    if not isinstance(operator, Mapping):
        raise AlignmentError("R&D teacher base operator has no semantic block")
    overlap_drift = {
        key: {"base_contract": value, "typed_config": fields.get(key)}
        for key, value in operator.items()
        if key in fields and fields.get(key) != value
    }
    if overlap_drift:
        raise AlignmentError(
            "R&D teacher typed config changes the base search operator: "
            + json.dumps(overlap_drift, sort_keys=True)
        )

    synthetic = {
        "schema_version": binding["schema_version"],
        "contract_sha256": binding["binding_sha256"],
        "target_information_regime": base["target_information_regime"],
        "producer_checkpoint": dict(producer),
        "operator": dict(operator),
        "acceptance": dict(base.get("acceptance", {})),
        "artifacts": {
            "typed_generation_config": {
                "path": str(config_path),
                "sha256": _file_sha256(config_path),
            }
        },
    }
    authority = {
        "kind": "nonpromotable_rd_teacher_transition",
        "binding": {
            "path": str(binding_path),
            "file_sha256": _file_sha256(binding_path),
            "binding_sha256": binding["binding_sha256"],
        },
        "base_operator_contract": {
            "path": str(base_path),
            "file_sha256": _file_sha256(base_path),
            "contract_sha256": inspected["contract_sha256"],
        },
        "teacher_feature_contract": dict(feature_contract),
    }
    return synthetic, config_path, authority


def _operator_identity(
    contract_path: Path,
    checkpoint: Path,
    *,
    require_current_target: bool = False,
    identity_schema: str = OPERATOR_IDENTITY_SCHEMA_V2,
    target_execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the scientific identity of policy targets, excluding fleet layout."""

    contract_path, contract = _load_json(
        contract_path, where="coherent target operator contract"
    )
    transition_authority: dict[str, Any] | None = None
    preloaded_config: dict[str, Any] | None = None
    if contract.get("schema_version") == RD_TEACHER_TRANSITION_BINDING_SCHEMA:
        contract, config_path, transition_authority = (
            _rd_teacher_transition_authority(contract_path, contract, checkpoint)
        )
    elif contract.get("schema_version") == POST_WAVE_SOURCE_BINDING_SCHEMA:
        unsigned = dict(contract)
        stated = unsigned.pop("binding_sha256", None)
        typed = contract.get("typed_generation_config")
        producer = contract.get("producer_checkpoint")
        evidence = contract.get("evidence")
        if (
            stated != _value_sha256(unsigned)
            or contract.get("diagnostic_only") is not True
            or contract.get("promotion_eligible") is not False
            or not isinstance(typed, Mapping)
            or not isinstance(producer, Mapping)
            or not isinstance(evidence, Mapping)
        ):
            raise AlignmentError("post-wave source operator binding is not fail-closed")
        source_ref = typed.get("source_manifest")
        if not isinstance(source_ref, Mapping):
            raise AlignmentError("post-wave source operator lost its manifest")
        config_path = _regular_file(
            Path(str(source_ref.get("path", ""))),
            where="post-wave representative generation manifest",
        )
        if _file_sha256(config_path) != source_ref.get("file_sha256"):
            raise AlignmentError("post-wave generation manifest bytes drifted")
        preloaded_config = {
            "schema_version": typed.get("schema_version"),
            "pipeline": typed.get("pipeline"),
            "fields": typed.get("fields"),
        }
        contract = {
            "schema_version": POST_WAVE_SOURCE_BINDING_SCHEMA,
            "contract_sha256": str(stated),
            "target_information_regime": contract.get(
                "target_information_regime"
            ),
            "producer_checkpoint": dict(producer),
            "operator": dict(contract.get("operator", {})),
            "acceptance": dict(contract.get("acceptance", {})),
        }
        transition_authority = {
            "kind": "historical_post_wave_source_operator",
            "source_binding": {
                "path": str(contract_path),
                "file_sha256": _file_sha256(contract_path),
                "binding_sha256": stated,
            },
            "post_wave_evidence": dict(evidence),
        }
    else:
        if require_current_target:
            try:
                target_inventory.inspect_rd_contract(contract_path)
            except (target_inventory.InventoryError, OSError, ValueError) as error:
                raise AlignmentError(
                    f"current coherent operator contract refused: {error}"
                ) from error
        config_ref = contract.get("artifacts", {}).get("typed_generation_config")
        if not isinstance(config_ref, Mapping):
            raise AlignmentError("operator contract has no typed generation config")
        config_path = _resolve_artifact(
            contract_path, str(config_ref.get("path", ""))
        )
        if _file_sha256(config_path) != config_ref.get("sha256"):
            raise AlignmentError("typed generation config bytes drifted")
    if preloaded_config is None:
        _config_path, config = _load_json(
            config_path, where="typed generation config"
        )
    else:
        config = preloaded_config
    fields = config.get("fields")
    operator = contract.get("operator")
    producer = contract.get("producer_checkpoint")
    checkpoint = _regular_file(checkpoint, where="operator checkpoint")
    contract_unsigned = dict(contract)
    contract_digest = contract_unsigned.pop("contract_sha256", None)
    if (
        (
            transition_authority is None
            and contract.get("schema_version")
            not in target_inventory.RD_CONTRACT_SCHEMAS
        )
        or (
            transition_authority is None
            and contract_digest != _value_sha256(contract_unsigned)
        )
        or config.get("schema_version") != 13
        or config.get("pipeline") != "generate"
        or not isinstance(fields, Mapping)
        or not isinstance(operator, Mapping)
        or not isinstance(producer, Mapping)
        or producer.get("sha256") != _file_sha256(checkpoint)
    ):
        raise AlignmentError("operator config/checkpoint/regime identity drifted")
    overlap_drift = {
        key: {"contract": value, "typed_config": fields.get(key)}
        for key, value in operator.items()
        if key in fields and fields.get(key) != value
    }
    if overlap_drift:
        raise AlignmentError(
            "contract and typed operator config disagree: "
            + json.dumps(overlap_drift, sort_keys=True)
        )
    search = _field_bundle(fields, SEARCH_FIELDS)
    belief = _field_bundle(fields, BELIEF_FIELDS)
    chance = _field_bundle(fields, CHANCE_FIELDS)
    symmetry = _field_bundle(fields, SYMMETRY_FIELDS)
    semantics = _semantic_field_bundle(fields, operator, TARGET_SEMANTIC_FIELDS)
    # Stored policy columns are probabilities over the legal rows produced by
    # this catalog. The action catalog is code-owned rather than a tunable
    # GenerateConfig field, so bind its reviewed version explicitly instead of
    # allowing a Stage-C reanalysis plan to compare targets with different
    # action-id semantics.
    semantics["action_mask_version"] = ACTION_MASK_VERSION
    regime = contract.get("target_information_regime")
    if not isinstance(regime, str) or not regime:
        raise AlignmentError("operator target information regime is missing")
    if require_current_target and (
        regime != COHERENT_REGIME
        or search["n_full"] != 128
        or search["c_scale"] is None
        or belief["coherent_public_belief_search"] is not True
        or belief["boundary_value_particles"] != 1
        or belief["information_set_search"] is not False
        or belief["determinization_particles"] != 1
        or chance["correct_rust_chance_spectra"] is not True
        or chance["lazy_interior_chance"] is not True
        or symmetry["symmetry_averaged_eval"] is not True
        or semantics["public_observation"] is not True
    ):
        raise AlignmentError("target operator is not current coherent-public n128")
    if identity_schema not in {
        OPERATOR_IDENTITY_SCHEMA_V1,
        OPERATOR_IDENTITY_SCHEMA_V2,
        OPERATOR_IDENTITY_SCHEMA_V3,
        OPERATOR_IDENTITY_SCHEMA_V4,
    }:
        raise AlignmentError(
            f"unsupported policy-target identity schema {identity_schema!r}"
        )
    value: dict[str, Any] = {
        "schema_version": identity_schema,
        "producer_checkpoint": {
            "path": str(checkpoint),
            "sha256": _file_sha256(checkpoint),
        },
        "target_information_regime": regime,
        "operator_contract_semantics": dict(operator),
        "search": search,
        "belief": {
            **belief,
            "semantic_identity": regime,
        },
        "chance": {
            **chance,
            "rng_stream_schema": SEARCH_RNG_STREAM_SCHEMA,
            "separate_rng_domains": ["gumbel", "chance", "belief"],
        },
        "symmetry": symmetry,
        "target_semantics": {
            **semantics,
            "search_evidence_schema": contract.get("acceptance", {}).get(
                "require_search_evidence_schema"
            ),
            "typed_generation_config_schema": config["schema_version"],
        },
        "authority": {
            "contract": {
                "path": str(contract_path),
                "file_sha256": _file_sha256(contract_path),
                "contract_sha256": contract["contract_sha256"],
            },
            "typed_generation_config": {
                "path": str(config_path),
                "file_sha256": _file_sha256(config_path),
            },
            "operator_semantic_sha256": _value_sha256(operator),
        },
    }
    if identity_schema == OPERATOR_IDENTITY_SCHEMA_V4:
        try:
            teacher_adapter = train_bc._checkpoint_entity_feature_adapter_version(  # noqa: SLF001
                str(checkpoint)
            )
        except (OSError, RuntimeError, SystemExit, ValueError) as error:
            raise AlignmentError(
                f"cannot authenticate target checkpoint adapter: {error}"
            ) from error
        if (
            teacher_adapter != RUST_ENTITY_ADAPTER_V6
            or fields.get("teacher_entity_feature_adapter_version")
            != teacher_adapter
        ):
            raise AlignmentError(
                "v4 target identity requires explicit checkpoint-matched V6 adapter"
            )
        value["teacher_feature_contract"] = {
            "schema_version": ENTITY_FEATURE_ADAPTER_CHECKPOINT_SCHEMA,
            "entity_feature_adapter_version": teacher_adapter,
        }
        if transition_authority is not None:
            value["authority"].update(transition_authority)
    if identity_schema in {
        OPERATOR_IDENTITY_SCHEMA_V2,
        OPERATOR_IDENTITY_SCHEMA_V3,
        OPERATOR_IDENTITY_SCHEMA_V4,
    }:
        execution = (
            dict(target_execution)
            if target_execution is not None
            else {
                "schema_version": "a1-generation-target-execution-v1",
                "mode": "sealed_generation_schedule",
                "force_full_override": None,
                "effective_simulations": None,
                "budget_source": "per_row_playout_cap_randomization",
                "row_seed_schema": None,
            }
        )
        if target_execution is not None and execution != STAGE_C_TARGET_EXECUTION:
            raise AlignmentError("Stage-C target execution override drifted")
        value["effective_gumbel_config"] = _complete_effective_search_config(fields)
        value["effective_evaluator_config"] = _complete_effective_evaluator_config(
            fields,
            entity_feature_adapter_version=(
                value["teacher_feature_contract"][
                    "entity_feature_adapter_version"
                ]
                if identity_schema == OPERATOR_IDENTITY_SCHEMA_V4
                else None
            ),
        )
        value["target_execution"] = _json_normalized(execution)
        value["identity_exclusions"] = {
            "gumbel.seed": (
                "per-row stochastic replicate; Stage-C binds the deterministic "
                "row-seed schema in target_execution"
            ),
            "evaluator.cache_size": "execution layout only; no model output semantics",
            "fleet_and_worker_layout": (
                "authenticated as provenance by contracts/receipts, not a root-target "
                "operator input"
            ),
        }
    if identity_schema in {
        OPERATOR_IDENTITY_SCHEMA_V3,
        OPERATOR_IDENTITY_SCHEMA_V4,
    }:
        value["root_value_output_contract"] = _paired_root_value_output_contract(
            fields,
            operator,
        )
        if (
            require_current_target
            and value["root_value_output_contract"]["preserve_root_prior_value"]
            is not True
        ):
            raise AlignmentError(
                "current paired-root target must preserve root_prior_value"
            )
    # The scientific identity deliberately excludes contract path, seed lanes,
    # and fleet placement. Those authenticate provenance but do not change a
    # root target. It includes the producer network and every search semantic.
    scientific_keys = [
        "producer_checkpoint",
        "target_information_regime",
        "operator_contract_semantics",
        "search",
        "belief",
        "chance",
        "symmetry",
        "target_semantics",
    ]
    if identity_schema in {
        OPERATOR_IDENTITY_SCHEMA_V2,
        OPERATOR_IDENTITY_SCHEMA_V3,
        OPERATOR_IDENTITY_SCHEMA_V4,
    }:
        scientific_keys.extend(
            (
                "effective_gumbel_config",
                "effective_evaluator_config",
                "target_execution",
                "identity_exclusions",
            )
        )
    if identity_schema in {
        OPERATOR_IDENTITY_SCHEMA_V3,
        OPERATOR_IDENTITY_SCHEMA_V4,
    }:
        scientific_keys.append("root_value_output_contract")
    if identity_schema == OPERATOR_IDENTITY_SCHEMA_V4:
        scientific_keys.append("teacher_feature_contract")
    scientific = {key: value[key] for key in scientific_keys}
    scientific["producer_checkpoint"] = {
        "sha256": value["producer_checkpoint"]["sha256"]
    }
    if identity_schema in {
        OPERATOR_IDENTITY_SCHEMA_V3,
        OPERATOR_IDENTITY_SCHEMA_V4,
    }:
        scientific_output = dict(scientific["root_value_output_contract"])
        scientific_authority = dict(scientific_output["authority"])
        scientific_authority.pop("path")
        scientific_output["authority"] = scientific_authority
        scientific["root_value_output_contract"] = scientific_output
    value["identity_sha256"] = _value_sha256(scientific)
    return value


def _classify_policy_rows(
    policy_active: np.ndarray,
    *,
    source_identity_sha256: str,
    target_identity_sha256: str,
) -> tuple[np.ndarray, np.ndarray]:
    active = np.asarray(policy_active, dtype=np.bool_)
    exact = source_identity_sha256 == target_identity_sha256
    eligible = active.copy() if exact else np.zeros(active.shape, dtype=np.bool_)
    status = np.full(active.shape, POLICY_STATUS["inactive_no_stored_policy"], np.uint8)
    status[active & eligible] = POLICY_STATUS["eligible_exact_operator"]
    status[active & ~eligible] = POLICY_STATUS["quarantined_stale_operator"]
    return eligible, status


def _reliability_inventory(
    data: Mapping[str, Any], *, row_count: int
) -> tuple[np.ndarray, dict[str, Any]]:
    present = set(TARGET_RELIABILITY_COLUMNS) & set(data.keys())
    if not present:
        classes = np.full(row_count, RELIABILITY_CLASS["not_collected"], dtype=np.uint8)
        value = {
            "schema_version": "a1-stage-c-reliability-inventory-v1",
            "storage": "not_collected",
            "rows": row_count,
            "audited_rows": 0,
            "unaudited_rows": 0,
            "not_collected_rows": row_count,
            "confidence_weighting_authorized": False,
            "reason": "duplicate-search reliability columns are absent",
        }
        value["inventory_sha256"] = _value_sha256(value)
        return classes, value
    if present != set(TARGET_RELIABILITY_COLUMNS):
        raise AlignmentError(
            "partial target reliability schema; missing "
            f"{sorted(set(TARGET_RELIABILITY_COLUMNS) - present)}"
        )
    columns = {
        name: np.asarray(data[name]).reshape(-1) for name in TARGET_RELIABILITY_COLUMNS
    }
    if any(values.shape != (row_count,) for values in columns.values()):
        raise AlignmentError("target reliability columns are not row-aligned")
    version = columns["target_reliability_version"].astype(np.int64)
    audited = columns["target_reliability_audited"].astype(np.bool_)
    js = columns["target_reliability_js_divergence"].astype(np.float64)
    policy_agree = columns["target_reliability_policy_top1_agreement"].astype(np.bool_)
    q_agree = columns["target_reliability_q_top1_agreement"].astype(np.bool_)
    margin_primary = columns["target_reliability_q_margin_primary"].astype(np.float64)
    margin_duplicate = columns["target_reliability_q_margin_duplicate"].astype(
        np.float64
    )
    confidence = columns["target_reliability_confidence"].astype(np.float64)
    if np.any((version != 0) & (version != TARGET_RELIABILITY_VERSION)):
        raise AlignmentError("target reliability version drifted")
    not_collected = version == 0
    versioned = version == TARGET_RELIABILITY_VERSION
    unaudited = versioned & ~audited
    if np.any(
        not_collected
        & (
            audited
            | ~np.isnan(js)
            | policy_agree
            | q_agree
            | ~np.isnan(margin_primary)
            | ~np.isnan(margin_duplicate)
            | (confidence != 1.0)
        )
    ):
        raise AlignmentError(
            "version-zero reliability rows are not exact not-collected sentinels"
        )
    if np.any(audited & ~versioned):
        raise AlignmentError("audited reliability row has no versioned evidence")
    if np.any(
        unaudited
        & (
            ~np.isnan(js)
            | policy_agree
            | q_agree
            | ~np.isnan(margin_primary)
            | ~np.isnan(margin_duplicate)
            | (confidence != 1.0)
        )
    ):
        raise AlignmentError(
            "unaudited reliability rows do not carry the neutral typed sentinel"
        )
    if np.any(
        audited
        & (
            ~np.isfinite(js)
            | (js < 0.0)
            | (js > math.log(2.0) + 1.0e-6)
            | ~np.isfinite(margin_primary)
            | (margin_primary < 0.0)
            | ~np.isfinite(margin_duplicate)
            | (margin_duplicate < 0.0)
            | ~np.isfinite(confidence)
            | (confidence < 0.0)
            | (confidence > 1.0)
        )
    ):
        raise AlignmentError("audited reliability values are malformed")
    audited_indices = np.flatnonzero(audited)
    for index in audited_indices.tolist():
        expected = target_reliability_confidence(
            float(js[index]), policy_top1_agreement=bool(policy_agree[index])
        )
        if not math.isclose(float(confidence[index]), expected, abs_tol=2.0e-6):
            raise AlignmentError("audited reliability confidence formula drifted")
    classes = np.full(row_count, RELIABILITY_CLASS["not_collected"], dtype=np.uint8)
    classes[unaudited] = RELIABILITY_CLASS["unaudited_neutral_sentinel"]
    classes[audited] = RELIABILITY_CLASS["duplicate_search_audited"]
    collected = int(np.count_nonzero(versioned))
    value = {
        "schema_version": "a1-stage-c-reliability-inventory-v1",
        "storage": (
            "typed_duplicate_search_fields"
            if collected
            else "schema_columns_present_but_not_collected"
        ),
        "reliability_schema": TARGET_RELIABILITY_SCHEMA,
        "reliability_version": TARGET_RELIABILITY_VERSION,
        "rows": row_count,
        "audited_rows": int(np.count_nonzero(audited)),
        "unaudited_rows": int(np.count_nonzero(unaudited)),
        "not_collected_rows": int(np.count_nonzero(not_collected)),
        "confidence_formula": TARGET_RELIABILITY_CONFIDENCE_FORMULA,
        "confidence_weighting_authorized": bool(row_count and np.all(audited)),
        "unaudited_confidence_semantics": (
            "neutral sentinel for learner compatibility; never audited evidence"
        ),
    }
    value["inventory_sha256"] = _value_sha256(value)
    return classes, value


def _policy_surprise(
    data: Mapping[str, Any], rows: np.ndarray, *, chunk_rows: int = 8192
) -> np.ndarray:
    result = np.zeros(rows.shape, dtype=np.float32)
    target_column = data["target_policy"]
    prior_column = data["prior_policy"]
    legal_column = data["legal_action_ids"]
    for start in range(0, len(rows), chunk_rows):
        stop = min(start + chunk_rows, len(rows))
        index = rows[start:stop]
        target = np.asarray(target_column[index], dtype=np.float64)
        prior = np.asarray(prior_column[index], dtype=np.float64)
        legal = np.asarray(legal_column[index]) >= 0
        target = np.where(legal, np.maximum(target, 0.0), 0.0)
        prior = np.where(legal, np.maximum(prior, 0.0), 0.0)
        target /= np.maximum(target.sum(axis=1, keepdims=True), 1.0e-12)
        prior /= np.maximum(prior.sum(axis=1, keepdims=True), 1.0e-12)
        terms = np.where(
            target > 0.0,
            target * np.log((target + 1.0e-12) / (prior + 1.0e-12)),
            0.0,
        )
        values = terms.sum(axis=1)
        if np.any(~np.isfinite(values)) or np.any(values < -1.0e-7):
            raise AlignmentError("stored policy surprise is malformed")
        result[start:stop] = np.maximum(values, 0.0).astype(np.float32)
    return result


def _width_bucket(width: int) -> str:
    if width <= 3:
        return "w02_03"
    if width <= 7:
        return "w04_07"
    if width <= 15:
        return "w08_15"
    if width <= 31:
        return "w16_31"
    return "w32_plus"


def _stable_u64(seed: int, *values: int) -> int:
    payload = ":".join(
        ("a1-stage-c-game-first-subset-v2", str(seed), *(str(v) for v in values))
    )
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big")


def _decision_bucket(decision_index: int) -> str:
    for name, start, stop in ROOT_BREADTH_DECISION_BINS:
        if decision_index >= start and (stop is None or decision_index < stop):
            return name
    raise AlignmentError("Stage-C decision index is negative")


def _root_breadth_scope(
    *,
    population_game_seeds: np.ndarray,
    selected_game_seeds: np.ndarray,
    selected_decision_indices: np.ndarray,
    selected_phases: np.ndarray,
) -> dict[str, Any]:
    population = np.unique(np.asarray(population_game_seeds, dtype=np.int64))
    games = np.asarray(selected_game_seeds, dtype=np.int64)
    decisions = np.asarray(selected_decision_indices, dtype=np.int64)
    phases = np.asarray(selected_phases).astype(str, copy=False)
    if (
        population.size == 0
        or games.ndim != 1
        or decisions.shape != games.shape
        or phases.shape != games.shape
        or np.any(decisions < 0)
    ):
        raise AlignmentError("Stage-C root-breadth inputs are malformed")
    selected_unique, roots_per_game = np.unique(games, return_counts=True)
    if np.setdiff1d(selected_unique, population).size:
        raise AlignmentError("Stage-C selected root references a game outside its split")
    phase_counts = {
        phase: int(np.count_nonzero(phases == phase))
        for phase in ROOT_BREADTH_REQUIRED_PHASES
    }
    unknown_phases = sorted(set(phases.tolist()) - set(ROOT_BREADTH_REQUIRED_PHASES))
    decision_counts = {
        name: int(
            np.count_nonzero(
                (decisions >= start)
                & (True if stop is None else (decisions < stop))
            )
        )
        for name, start, stop in ROOT_BREADTH_DECISION_BINS
    }
    selected_count = int(games.size)
    population_count = int(population.size)
    selected_game_count = int(selected_unique.size)
    denominator = max(selected_count, 1)
    return {
        "population_game_count": population_count,
        "selected_root_count": selected_count,
        "selected_game_count": selected_game_count,
        "unique_game_fraction": selected_game_count / population_count,
        "missing_game_count": population_count - selected_game_count,
        "roots_per_represented_game": {
            "minimum": int(roots_per_game.min()) if roots_per_game.size else 0,
            "maximum": int(roots_per_game.max()) if roots_per_game.size else 0,
            "mean": (
                float(selected_count / selected_game_count)
                if selected_game_count
                else 0.0
            ),
        },
        "phase_counts": phase_counts,
        "phase_fractions": {
            phase: count / denominator for phase, count in phase_counts.items()
        },
        "unknown_phases": unknown_phases,
        "decision_index_bin_counts": decision_counts,
        "decision_index_bin_fractions": {
            name: count / denominator for name, count in decision_counts.items()
        },
    }


def _root_breadth_failures(
    scopes: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    failures: list[str] = []
    for scope_name, scope in scopes.items():
        if (
            float(scope["unique_game_fraction"])
            < float(ROOT_BREADTH_CONTRACT["minimum_unique_game_fraction"])
        ):
            failures.append(f"{scope_name}:unique_game_fraction")
        if (
            int(scope["roots_per_represented_game"]["minimum"])
            < int(ROOT_BREADTH_CONTRACT["minimum_roots_per_represented_game"])
        ):
            failures.append(f"{scope_name}:minimum_roots_per_represented_game")
        if scope["unknown_phases"]:
            failures.append(f"{scope_name}:unknown_phases")
        for phase, fraction in scope["phase_fractions"].items():
            if float(fraction) < float(ROOT_BREADTH_CONTRACT["minimum_phase_fraction"]):
                failures.append(f"{scope_name}:phase:{phase}")
        for name, fraction in scope["decision_index_bin_fractions"].items():
            if float(fraction) < float(
                ROOT_BREADTH_CONTRACT["minimum_decision_bin_fraction"]
            ):
                failures.append(f"{scope_name}:decision_bin:{name}")
    return failures


def _stage_c_root_breadth_inventory(
    *,
    corpus_game_seeds: np.ndarray,
    validation_game_seeds: np.ndarray,
    selected_game_seeds: np.ndarray,
    selected_decision_indices: np.ndarray,
    selected_phases: np.ndarray,
) -> dict[str, Any]:
    all_games = np.unique(np.asarray(corpus_game_seeds, dtype=np.int64))
    validation_games = np.unique(
        np.asarray(validation_game_seeds, dtype=np.int64)
    )
    if np.setdiff1d(validation_games, all_games).size:
        raise AlignmentError("Stage-C validation split names a game outside the corpus")
    selected_games = np.asarray(selected_game_seeds, dtype=np.int64)
    decisions = np.asarray(selected_decision_indices, dtype=np.int64)
    phases = np.asarray(selected_phases).astype(str, copy=False)
    if (
        selected_games.ndim != 1
        or decisions.shape != selected_games.shape
        or phases.shape != selected_games.shape
    ):
        raise AlignmentError("Stage-C selected root-breadth arrays are misaligned")
    selected_validation = np.isin(selected_games, validation_games)
    scopes = {
        "training": _root_breadth_scope(
            population_game_seeds=np.setdiff1d(all_games, validation_games),
            selected_game_seeds=selected_games[~selected_validation],
            selected_decision_indices=decisions[~selected_validation],
            selected_phases=phases[~selected_validation],
        ),
        "validation": _root_breadth_scope(
            population_game_seeds=validation_games,
            selected_game_seeds=selected_games[selected_validation],
            selected_decision_indices=decisions[selected_validation],
            selected_phases=phases[selected_validation],
        ),
    }
    failures = _root_breadth_failures(scopes)
    value: dict[str, Any] = {
        "schema_version": ROOT_BREADTH_SCHEMA,
        "contract": json.loads(json.dumps(ROOT_BREADTH_CONTRACT)),
        "scopes": scopes,
        "passed": not failures,
        "failures": failures,
    }
    value["inventory_sha256"] = _value_sha256(value)
    return value


def _verify_stage_c_root_breadth_inventory(
    value: object, *, selected_rows: int
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AlignmentError("Stage-C root-breadth inventory is missing")
    unsigned = dict(value)
    stated = unsigned.pop("inventory_sha256", None)
    scopes = value.get("scopes")
    if (
        value.get("schema_version") != ROOT_BREADTH_SCHEMA
        or stated != _value_sha256(unsigned)
        or value.get("contract") != ROOT_BREADTH_CONTRACT
        or not isinstance(scopes, dict)
        or set(scopes) != set(ROOT_BREADTH_CONTRACT["required_scopes"])
        or value.get("passed") is not True
        or value.get("failures") != []
        or _root_breadth_failures(scopes) != []
        or sum(
            int(scope.get("selected_root_count", -1))
            for scope in scopes.values()
            if isinstance(scope, Mapping)
        )
        != int(selected_rows)
    ):
        raise AlignmentError("Stage-C root-breadth inventory failed or drifted")
    return json.loads(json.dumps(value))


def _select_game_first(
    *,
    rows: np.ndarray,
    game_seeds: np.ndarray,
    decision_indices: np.ndarray,
    phases: np.ndarray,
    legal_widths: np.ndarray,
    surprise: np.ndarray,
    reliability_class: np.ndarray,
    policy_status: np.ndarray,
    population_game_seeds: np.ndarray,
    validation_game_seeds: np.ndarray,
    limit: int,
    selection_seed: int,
    max_rows_per_game: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, Any]]:
    minimum_roots = int(ROOT_BREADTH_CONTRACT["minimum_roots_per_represented_game"])
    minimum_fraction = float(ROOT_BREADTH_CONTRACT["minimum_unique_game_fraction"])
    if limit <= 0 or max_rows_per_game < minimum_roots:
        raise AlignmentError(
            "Stage-C subset budget must be positive and max rows per game must "
            f"be at least {minimum_roots}"
        )
    arrays = (
        game_seeds,
        decision_indices,
        phases,
        legal_widths,
        surprise,
        reliability_class,
        policy_status,
    )
    if any(np.asarray(value).shape != rows.shape for value in arrays):
        raise AlignmentError("game-first subset inputs are not row-aligned")
    if not len(rows):
        raise AlignmentError("no multi-action policy roots are available for reanalysis")

    games = np.asarray(game_seeds, dtype=np.int64)
    decisions = np.asarray(decision_indices, dtype=np.int64)
    phase_values = np.asarray(phases).astype(str, copy=False)
    population = np.unique(np.asarray(population_game_seeds, dtype=np.int64))
    validation = np.unique(np.asarray(validation_game_seeds, dtype=np.int64))
    if (
        np.setdiff1d(validation, population).size
        or np.setdiff1d(np.unique(games), population).size
    ):
        raise AlignmentError("Stage-C candidate or validation game is outside the corpus")
    training = np.setdiff1d(population, validation)
    candidate_positions_by_game: dict[int, list[int]] = {}
    for position, game in enumerate(games.tolist()):
        candidate_positions_by_game.setdefault(int(game), []).append(position)

    scopes = {"training": training, "validation": validation}
    qualified_by_scope: dict[str, list[int]] = {}
    required_games_by_scope: dict[str, int] = {}
    coverage: dict[str, Any] = {}
    for scope_name, scope_games in scopes.items():
        if len(scope_games) == 0:
            raise AlignmentError(f"Stage-C {scope_name} game population is empty")
        qualified = [
            int(game)
            for game in scope_games.tolist()
            if len(candidate_positions_by_game.get(int(game), ())) >= minimum_roots
        ]
        required = int(math.ceil(minimum_fraction * len(scope_games)))
        if len(qualified) < required:
            raise AlignmentError(
                "Stage-C candidate coverage cannot satisfy root breadth: "
                f"{scope_name} has {len(qualified)}/{len(scope_games)} games with "
                f"at least {minimum_roots} multi-action roots; requires {required}"
            )
        qualified_by_scope[scope_name] = qualified
        required_games_by_scope[scope_name] = required
        coverage[scope_name] = {
            "population_game_count": int(len(scope_games)),
            "games_with_minimum_candidate_roots": int(len(qualified)),
            "required_selected_game_count": required,
        }
    minimum_breadth_roots = minimum_roots * sum(required_games_by_scope.values())
    if limit < minimum_breadth_roots:
        raise AlignmentError(
            "Stage-C subset budget cannot satisfy root breadth: "
            f"requested={limit} required_at_least={minimum_breadth_roots}"
        )

    selected_games: dict[str, list[int]] = {}
    for scope_index, scope_name in enumerate(("training", "validation")):
        ordered = sorted(
            qualified_by_scope[scope_name],
            key=lambda game: (_stable_u64(selection_seed, scope_index, game), game),
        )
        selected_games[scope_name] = ordered[: required_games_by_scope[scope_name]]
    remaining_game_capacity = limit // minimum_roots - sum(
        len(values) for values in selected_games.values()
    )
    extras = sorted(
        (
            _stable_u64(selection_seed, 2, game),
            scope_name,
            game,
        )
        for scope_name in ("training", "validation")
        for game in qualified_by_scope[scope_name]
        if game not in set(selected_games[scope_name])
    )
    for _key, scope_name, game in extras[: max(remaining_game_capacity, 0)]:
        selected_games[scope_name].append(game)
    selected_game_set = {
        game for values in selected_games.values() for game in values
    }

    quantiles = np.quantile(np.asarray(surprise, dtype=np.float64), [0.25, 0.5, 0.75])
    surprise_bins = np.searchsorted(quantiles, surprise, side="right")
    full_strata = np.asarray(
        [
            "|".join(
                (
                    str(phase_values[position]),
                    _width_bucket(int(legal_widths[position])),
                    f"surprise_q{int(surprise_bins[position])}",
                    f"reliability_{int(reliability_class[position])}",
                    f"policy_status_{int(policy_status[position])}",
                )
            )
            for position in range(len(rows))
        ],
        dtype=str,
    )
    selected_positions: list[int] = []
    selected_position_set: set[int] = set()
    per_game_selected: dict[int, int] = {}
    for game in sorted(
        selected_game_set,
        key=lambda value: (_stable_u64(selection_seed, 3, value), value),
    ):
        candidates: list[tuple[int, int, str, str]] = []
        for position in candidate_positions_by_game[game]:
            candidates.append(
                (
                    _stable_u64(
                        selection_seed,
                        game,
                        int(decisions[position]),
                        int(rows[position]),
                    ),
                    position,
                    str(phase_values[position]),
                    _decision_bucket(int(decisions[position])),
                )
            )
        candidates.sort()
        covered_phases: set[str] = set()
        covered_decision_bins: set[str] = set()
        while per_game_selected.get(game, 0) < minimum_roots:
            remaining = [
                item for item in candidates if item[1] not in selected_position_set
            ]
            if not remaining:
                raise AlignmentError(
                    f"Stage-C qualified game {game} lost its breadth roots"
                )
            _key, position, phase, decision_bin = min(
                remaining,
                key=lambda item: (
                    -int(item[2] not in covered_phases)
                    - int(item[3] not in covered_decision_bins),
                    item[0],
                    item[1],
                ),
            )
            selected_positions.append(position)
            selected_position_set.add(position)
            per_game_selected[game] = per_game_selected.get(game, 0) + 1
            covered_phases.add(phase)
            covered_decision_bins.add(decision_bin)

    extra_groups: dict[str, list[tuple[int, int, int]]] = {}
    for game in selected_game_set:
        for position in candidate_positions_by_game[game]:
            if position in selected_position_set:
                continue
            extra_groups.setdefault(str(full_strata[position]), []).append(
                (
                    _stable_u64(
                        selection_seed,
                        game,
                        int(decisions[position]),
                        int(rows[position]),
                    ),
                    game,
                    position,
                )
            )
    for values in extra_groups.values():
        values.sort()
    extra_cursors = {stratum: 0 for stratum in extra_groups}
    while len(selected_positions) < limit:
        progressed = False
        for stratum in sorted(extra_groups):
            values = extra_groups[stratum]
            cursor = extra_cursors[stratum]
            while cursor < len(values):
                _key, game, position = values[cursor]
                cursor += 1
                if per_game_selected[game] < max_rows_per_game:
                    selected_positions.append(position)
                    selected_position_set.add(position)
                    per_game_selected[game] += 1
                    progressed = True
                    break
            extra_cursors[stratum] = cursor
            if len(selected_positions) == limit:
                break
        if not progressed:
            break

    positions = np.asarray(selected_positions, dtype=np.int64)
    selected_strata = full_strata[positions]
    stratum_counts = {
        stratum: int(np.count_nonzero(selected_strata == stratum))
        for stratum in np.unique(selected_strata)
    }
    selected_game_candidate_mask = np.isin(games, list(selected_game_set))
    candidate_counts_by_stratum = {
        stratum: int(
            np.count_nonzero(selected_game_candidate_mask & (full_strata == stratum))
        )
        for stratum in np.unique(full_strata[selected_game_candidate_mask])
    }
    inventory = _stage_c_root_breadth_inventory(
        corpus_game_seeds=population,
        validation_game_seeds=validation,
        selected_game_seeds=games[positions],
        selected_decision_indices=decisions[positions],
        selected_phases=phase_values[positions],
    )
    if inventory["passed"] is not True:
        raise AlignmentError(
            "Stage-C selected roots cannot satisfy root breadth: "
            + ",".join(inventory["failures"])
        )
    selection = {
        "schema_version": "a1-stage-c-game-first-selection-v1",
        "requested_root_budget": int(limit),
        "minimum_breadth_root_count": int(minimum_breadth_roots),
        "breadth_root_count": int(minimum_roots * len(selected_game_set)),
        "extra_root_count": int(len(positions) - minimum_roots * len(selected_game_set)),
        "max_rows_per_game": int(max_rows_per_game),
        "candidate_coverage": coverage,
        "selected_game_counts": {
            name: int(len(values)) for name, values in selected_games.items()
        },
        "candidate_counts_by_stratum": candidate_counts_by_stratum,
        "root_breadth": inventory,
    }
    selection["selection_sha256"] = _value_sha256(selection)
    return positions, selected_strata, stratum_counts, selection


def _artifact_ref(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve(strict=True)),
        "file_sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _build_plan(args: argparse.Namespace) -> dict[str, Any]:
    # Import lazily: the campaign module imports the one-dose/final-replication
    # stack, which reaches the learner overlay and reanalysis executor.  The
    # executor imports this module for the sealed identity helpers, so importing
    # the campaign at module load time forms an alignment -> ... -> executor ->
    # alignment cycle and makes the selector CLI depend on import order.
    from tools import a1_b200_active_policy_campaign as active_campaign
    from tools import a1_post_wave_stage_c_admission as post_wave_admission

    if args.subset_rows is not None and int(args.subset_rows) <= 0:
        raise AlignmentError(
            "production Stage-C requested root count must be positive"
        )

    try:
        _candidate_path, candidate = _load_json(
            args.coherent_corpus_admission,
            where="coherent corpus admission",
        )
        admission_schema = str(candidate.get("schema_version", ""))
        if admission_schema == POST_WAVE_CORPUS_ADMISSION_SCHEMA:
            admission_path, admission = post_wave_admission.verify_admission(
                _candidate_path
            )
        else:
            admission_path, admission = active_campaign._load_admission(  # noqa: SLF001
                _candidate_path
            )
    except (
        active_campaign.CampaignError,
        post_wave_admission.AdmissionError,
        OSError,
        ValueError,
    ) as error:
        raise AlignmentError(f"coherent corpus admission refused: {error}") from error
    corpus_record = admission["corpus"]
    corpus_root = Path(str(corpus_record["data_path"])).resolve(strict=True)
    validation_ref = corpus_record.get("validation_manifest")
    if not isinstance(validation_ref, Mapping):
        raise AlignmentError("coherent corpus admission lost its validation manifest")
    try:
        validation = train_bc._load_validation_game_seed_manifest_for_training(  # noqa: SLF001
            Path(str(validation_ref["path"])),
            validation_fraction=0.05,
            validation_seed=17,
            validation_max_samples=0,
            validation_game_seed_ranges=[],
        )
    except SystemExit as error:
        raise AlignmentError(
            f"coherent corpus validation manifest refused: {error}"
        ) from error
    validation_game_seeds = np.asarray(validation["game_seeds"], dtype=np.int64)
    source_contract_path = Path(str(admission["contract"]["path"]))
    source_checkpoint = Path(
        str(
            _load_json(source_contract_path, where="source operator contract")[1][
                "producer_checkpoint"
            ]["path"]
        )
    )
    source_identity = _operator_identity(source_contract_path, source_checkpoint)
    target_identity = _operator_identity(
        args.target_operator_contract,
        args.target_checkpoint,
        require_current_target=True,
        identity_schema=OPERATOR_IDENTITY_SCHEMA_V4,
        target_execution=STAGE_C_TARGET_EXECUTION,
    )
    if (
        source_identity["producer_checkpoint"]["sha256"]
        != corpus_record["producer_checkpoint_sha256"]
        or source_identity["target_information_regime"]
        != corpus_record["target_information_regime"]
    ):
        raise AlignmentError(
            "source operator identity is not bound to the admitted corpus"
        )

    data = train_bc.MemmapCorpus(corpus_root)
    rows = len(data)
    population_game_seeds = np.asarray(data["game_seed"], dtype=np.int64)
    population_decision_indices = np.asarray(data["decision_index"], dtype=np.int64)
    game_trace_qualification: dict[str, Any] | None = None
    if admission_schema == POST_WAVE_CORPUS_ADMISSION_SCHEMA:
        (
            reconstructable_game_seeds,
            game_trace_qualification,
        ) = _qualify_stage_c_game_traces(
            game_seeds=population_game_seeds,
            decision_indices=population_decision_indices,
        )
    else:
        reconstructable_game_seeds = np.unique(population_game_seeds)
    reconstructable_validation_game_seeds = np.intersect1d(
        validation_game_seeds,
        reconstructable_game_seeds,
        assume_unique=True,
    )
    requested_root_count = _resolve_stage_c_root_budget(
        requested_rows=args.subset_rows,
        admission_schema=admission_schema,
        population_game_seeds=population_game_seeds,
        validation_game_seeds=validation_game_seeds,
    )
    policy_weight = np.asarray(data["policy_weight_multiplier"], dtype=np.float32)
    policy_active = policy_weight > 0.0
    legal_widths_all = np.asarray(data["legal_action_ids"].row_counts(), dtype=np.int64)
    if policy_weight.shape != (rows,) or legal_widths_all.shape != (rows,):
        raise AlignmentError("corpus policy/legal columns are not row-aligned")
    stored_policy_eligible, policy_status = _classify_policy_rows(
        policy_active,
        source_identity_sha256=source_identity["identity_sha256"],
        target_identity_sha256=target_identity["identity_sha256"],
    )
    value_eligible = (
        np.asarray(data["value_weight_multiplier"], dtype=np.float32) > 0.0
        if "value_weight_multiplier" in data
        else np.ones(rows, dtype=np.bool_)
    )
    # Reanalysis creates a fresh operator-bound target.  It is therefore useful
    # for stale active rows *and* for multi-action roots whose original fast
    # search did not store policy supervision.  Policy inactivity must not be
    # confused with state ineligibility.
    reconstructable_game_rows = np.isin(
        population_game_seeds, reconstructable_game_seeds
    )
    reanalysis_candidate = (legal_widths_all > 1) & reconstructable_game_rows
    reliability_classes, reliability = _reliability_inventory(data, row_count=rows)

    candidate_rows = np.flatnonzero(reanalysis_candidate).astype(np.int64)
    game_seeds = np.asarray(data["game_seed"][candidate_rows], dtype=np.int64)
    decision_indices = np.asarray(
        data["decision_index"][candidate_rows], dtype=np.int64
    )
    phases = np.asarray(data["phase"][candidate_rows]).astype(str)
    legal_widths = legal_widths_all[candidate_rows]
    surprise = _policy_surprise(data, candidate_rows)
    candidate_reliability = reliability_classes[candidate_rows]
    candidate_status = policy_status[candidate_rows]
    (
        selected_positions,
        selected_strata,
        stratum_counts,
        game_first_selection,
    ) = _select_game_first(
        rows=candidate_rows,
        game_seeds=game_seeds,
        decision_indices=decision_indices,
        phases=phases,
        legal_widths=legal_widths,
        surprise=surprise,
        reliability_class=candidate_reliability,
        policy_status=candidate_status,
        population_game_seeds=reconstructable_game_seeds,
        validation_game_seeds=reconstructable_validation_game_seeds,
        limit=requested_root_count,
        selection_seed=int(args.selection_seed),
        max_rows_per_game=int(args.max_rows_per_game),
    )
    selected_rows = candidate_rows[selected_positions]
    if len(selected_rows) != requested_root_count:
        raise AlignmentError(
            "production Stage-C selection did not realize exactly "
            f"{requested_root_count:,} roots: realized={len(selected_rows)}"
        )
    chunks = int(args.chunks)
    if chunks <= 0:
        raise AlignmentError("chunks must be positive")
    chunk_index = np.arange(len(selected_rows), dtype=np.int32) % chunks
    identity_sha = np.asarray(
        [
            _value_sha256(
                {
                    "corpus_meta_file_sha256": corpus_record["corpus_meta_file_sha256"],
                    "row_index": int(row),
                    "game_seed": int(game_seeds[position]),
                    "decision_index": int(decision_indices[position]),
                }
            )
            for row, position in zip(
                selected_rows.tolist(), selected_positions.tolist(), strict=True
            )
        ],
        dtype="<U71",
    )

    output_root = args.output_root.expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    policy_status_path = output_root / "policy_status.u8.dat"
    policy_eligible_path = output_root / "stored_policy_eligible.bool.dat"
    value_eligible_path = output_root / "value_eligible.bool.dat"
    candidate_path = output_root / "reanalysis_candidate.bool.dat"
    subset_path = output_root / "selected_reanalysis_rows.npz"
    _write_array_immutable(policy_status_path, policy_status)
    _write_array_immutable(policy_eligible_path, stored_policy_eligible)
    _write_array_immutable(value_eligible_path, value_eligible)
    _write_array_immutable(candidate_path, reanalysis_candidate)
    subset_arrays = {
        "row_index": selected_rows.astype(np.int64),
        "game_seed": game_seeds[selected_positions].astype(np.int64),
        "decision_index": decision_indices[selected_positions].astype(np.int64),
        "phase": phases[selected_positions].astype(str),
        "legal_width": legal_widths[selected_positions].astype(np.int16),
        "policy_surprise_kl": surprise[selected_positions].astype(np.float32),
        "reliability_class": candidate_reliability[selected_positions].astype(np.uint8),
        "source_policy_status": candidate_status[selected_positions].astype(np.uint8),
        "stratum": selected_strata.astype(str),
        "chunk_index": chunk_index,
        "identity_sha256": identity_sha,
    }
    _write_immutable(subset_path, _npz_bytes(subset_arrays))

    overlay: dict[str, Any] = {
        "schema_version": OVERLAY_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "corpus": {
            "path": str(corpus_root),
            "corpus_meta_file_sha256": corpus_record["corpus_meta_file_sha256"],
            "payload_inventory_sha256": corpus_record["payload_inventory_sha256"],
            "row_count": rows,
        },
        "source_policy_target_identity_sha256": source_identity["identity_sha256"],
        "required_policy_target_identity_sha256": target_identity["identity_sha256"],
        "status_codes": POLICY_STATUS,
        "artifacts": {
            "policy_status": _artifact_ref(policy_status_path),
            "stored_policy_eligible": _artifact_ref(policy_eligible_path),
            "value_eligible": _artifact_ref(value_eligible_path),
            "reanalysis_candidate": _artifact_ref(candidate_path),
        },
        "counts": {
            "rows": rows,
            "stored_policy_active_rows": int(np.count_nonzero(policy_active)),
            "stored_policy_eligible_rows": int(
                np.count_nonzero(stored_policy_eligible)
            ),
            "stored_policy_quarantined_rows": int(
                np.count_nonzero(
                    policy_status == POLICY_STATUS["quarantined_stale_operator"]
                )
            ),
            "value_retained_rows": int(np.count_nonzero(value_eligible)),
            "reanalysis_candidate_rows": int(np.count_nonzero(reanalysis_candidate)),
        },
        "policy_quarantine_changes_value_eligibility": False,
        "policy_quarantine_changes_state_reanalysis_eligibility": False,
    }
    if game_trace_qualification is not None:
        overlay["counts"].update(
            {
                "trace_qualified_games": int(len(reconstructable_game_seeds)),
                "trace_excluded_games": int(
                    game_trace_qualification["excluded_games"]
                ),
            }
        )
    overlay["overlay_sha256"] = _value_sha256(overlay)
    overlay_path = output_root / "target_eligibility.overlay.json"
    _write_json_immutable(overlay_path, overlay)

    state_ready = bool(corpus_record.get("state_reanalysis_eligible"))
    blockers: list[str] = []
    if not state_ready:
        blockers.extend(
            [
                "per_row_sparse_reconstruction_qualification_required",
            ]
        )
    validation_scope: dict[str, Any] = {
        "schema_version": LEARNER_VALIDATION_SCOPE_SCHEMA,
        "manifest": {
            "path": str(Path(str(validation_ref["path"])).resolve(strict=True)),
            "file_sha256": str(validation["file_sha256"]),
            "manifest_sha256": str(validation["manifest_sha256"]),
            "a1_contract_sha256": str(validation["a1_contract_sha256"]),
        },
        "split_receipt": {
            "validation_row_count": int(validation["validation_row_count"]),
            "validation_game_seed_count": int(
                validation["validation_game_seed_count"]
            ),
            "validation_game_seed_set_sha256": str(
                validation["validation_game_seed_set_sha256"]
            ),
        },
        "trainer_exclusion_contract": {
            "schema_version": TRAINER_EXCLUSION_CONTRACT_SCHEMA,
            "input_validation_manifest_file_sha256": str(validation["file_sha256"]),
            "training_excluded_game_seed_count": int(
                validation["validation_game_seed_count"]
            ),
            "training_excluded_game_seed_set_sha256": str(
                validation["validation_game_seed_set_sha256"]
            ),
        },
        "target_coverage_receipt": {
            "root_breadth_inventory_sha256": game_first_selection["root_breadth"][
                "inventory_sha256"
            ],
            "selected_validation_root_count": int(
                game_first_selection["root_breadth"]["scopes"]["validation"][
                    "selected_root_count"
                ]
            ),
            "selected_validation_game_count": int(
                game_first_selection["root_breadth"]["scopes"]["validation"][
                    "selected_game_count"
                ]
            ),
        },
        "external_final_gate_authority": False,
    }
    validation_scope["scope_sha256"] = _value_sha256(validation_scope)
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "purpose": "current_coherent_n128_operator_aligned_reanalysis_subset",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "source_corpus_admission": {
            "path": str(admission_path),
            "file_sha256": _file_sha256(admission_path),
            "admission_sha256": admission["admission_sha256"],
        },
        "learner_validation_scope": validation_scope,
        "evidence_lifetimes": {
            "state_evidence": "reusable_if_information_surface_and_reconstruction_hold",
            "terminal_value_evidence": "reusable_independent_of_search_operator",
            "stored_policy_targets": "eligible_only_for_exact_policy_target_identity_sha256",
        },
        "source_policy_target_identity": source_identity,
        "target_policy_target_identity": target_identity,
        "target_identity_matches_stored_policy": (
            source_identity["identity_sha256"] == target_identity["identity_sha256"]
        ),
        "eligibility_overlay": {
            "path": str(overlay_path),
            "file_sha256": _file_sha256(overlay_path),
            "overlay_sha256": overlay["overlay_sha256"],
        },
        "source_reliability_evidence": reliability,
        "output_reliability_contract": {
            "mode": "primary_search_only_unaudited_v1",
            "duplicate_search_audit_scheduled": False,
            "unaudited_rows_must_use_typed_neutral_sentinel": True,
            "unaudited_confidence_is_measured_evidence": False,
            "future_duplicate_search_input_schema": TARGET_RELIABILITY_SCHEMA,
            "future_duplicate_search_rng_stream_schema": SEARCH_RNG_STREAM_SCHEMA,
        },
        "subset": {
            "schema_version": SUBSET_SCHEMA,
            "artifact": _artifact_ref(subset_path),
            "selected_rows": len(selected_rows),
            "candidate_rows": len(candidate_rows),
            "requested_rows": requested_root_count,
            "selection_seed": int(args.selection_seed),
            "max_rows_per_game": int(args.max_rows_per_game),
            "game_first_selection": game_first_selection,
            "stratification": [
                "training_validation_scope",
                "game_first_minimum_root_quota",
                "phase",
                "decision_index_bin",
                "legal_width_bucket",
                "stored_policy_surprise_quartile",
                "reliability_evidence_class",
                "stored_policy_eligibility_status",
            ],
            "stratum_counts": stratum_counts,
            "chunks": chunks,
            "assignment": "selected_ordinal_mod_chunks",
        },
        "execution": {
            "executor_semantics": "coherent_public_belief_n128_reanalysis_v1",
            "reconstruction_qualifier": (
                "tools/a1_stage_c_reanalysis_executor.py qualify"
            ),
            "sparse_gap_rule": (
                "an omitted decision is executable iff replay proves exactly "
                "one legal action at that index"
            ),
            "readiness_scope": "per_selected_row_not_whole_corpus",
            "legacy_public_conservation_pimc_executor_allowed": False,
            "authoritative_hidden_state_search_allowed": False,
            "execution_ready": state_ready,
            "blockers": blockers,
        },
    }
    if game_trace_qualification is not None:
        plan["source_game_trace_qualification"] = game_trace_qualification
    plan["plan_sha256"] = _value_sha256(plan)
    return plan


def _verify_plan(path: Path) -> dict[str, Any]:
    plan_path, plan = _load_json(path, where="Stage-C alignment plan")
    unsigned = dict(plan)
    stated = unsigned.pop("plan_sha256", None)
    if plan.get("schema_version") != PLAN_SCHEMA or stated != _value_sha256(unsigned):
        raise AlignmentError("Stage-C plan schema or semantic digest drifted")
    validation_scope = plan.get("learner_validation_scope")
    if not isinstance(validation_scope, Mapping):
        raise AlignmentError("Stage-C plan lacks learner validation scope")
    validation_scope_unsigned = dict(validation_scope)
    validation_scope_stated = validation_scope_unsigned.pop("scope_sha256", None)
    validation_ref = validation_scope.get("manifest")
    split_receipt = validation_scope.get("split_receipt")
    exclusion_contract = validation_scope.get("trainer_exclusion_contract")
    if not isinstance(validation_ref, Mapping):
        raise AlignmentError("Stage-C learner validation manifest binding is malformed")
    if not isinstance(split_receipt, Mapping) or not isinstance(
        exclusion_contract, Mapping
    ):
        raise AlignmentError("Stage-C learner validation split binding is malformed")
    validation_path = _regular_file(
        Path(str(validation_ref["path"])), where="Stage-C learner validation manifest"
    )
    try:
        validation = train_bc._load_validation_game_seed_manifest_for_training(  # noqa: SLF001
            validation_path,
            validation_fraction=0.05,
            validation_seed=17,
            validation_max_samples=0,
            validation_game_seed_ranges=[],
        )
    except SystemExit as error:
        raise AlignmentError(
            f"Stage-C learner validation manifest refused: {error}"
        ) from error
    if (
        validation_scope.get("schema_version") != LEARNER_VALIDATION_SCOPE_SCHEMA
        or validation_scope_stated != _value_sha256(validation_scope_unsigned)
        or validation_ref.get("file_sha256") != _file_sha256(validation_path)
        or validation_ref.get("file_sha256") != validation["file_sha256"]
        or validation_ref.get("manifest_sha256") != validation["manifest_sha256"]
        or validation_ref.get("a1_contract_sha256")
        != validation["a1_contract_sha256"]
        or split_receipt.get("validation_row_count")
        != validation["validation_row_count"]
        or split_receipt.get("validation_game_seed_count")
        != validation["validation_game_seed_count"]
        or split_receipt.get("validation_game_seed_set_sha256")
        != validation["validation_game_seed_set_sha256"]
        or exclusion_contract.get("schema_version")
        != TRAINER_EXCLUSION_CONTRACT_SCHEMA
        or exclusion_contract.get("input_validation_manifest_file_sha256")
        != validation["file_sha256"]
        or exclusion_contract.get("training_excluded_game_seed_count")
        != validation["validation_game_seed_count"]
        or exclusion_contract.get("training_excluded_game_seed_set_sha256")
        != validation["validation_game_seed_set_sha256"]
        or validation_scope.get("external_final_gate_authority") is not False
    ):
        raise AlignmentError("Stage-C learner validation scope drifted")
    admission_ref = plan.get("source_corpus_admission")
    if not isinstance(admission_ref, Mapping):
        raise AlignmentError("Stage-C plan lacks source corpus admission")
    admission_path, source_admission = _load_json(
        Path(str(admission_ref["path"])), where="Stage-C source corpus admission"
    )
    admission_unsigned = dict(source_admission)
    admission_stated = admission_unsigned.pop("admission_sha256", None)
    admission_schema = str(source_admission.get("schema_version", ""))
    if (
        admission_schema
        not in {
            LEGACY_CORPUS_ADMISSION_SCHEMA,
            POST_WAVE_CORPUS_ADMISSION_SCHEMA,
        }
        or admission_ref.get("file_sha256") != _file_sha256(admission_path)
        or admission_ref.get("admission_sha256") != admission_stated
        or admission_stated != _value_sha256(admission_unsigned)
    ):
        raise AlignmentError("Stage-C source corpus admission drifted")
    for identity_key in (
        "source_policy_target_identity",
        "target_policy_target_identity",
    ):
        identity = plan.get(identity_key)
        if not isinstance(identity, Mapping):
            raise AlignmentError(f"Stage-C plan lacks {identity_key}")
        authority = identity.get("authority", {}).get("contract")
        checkpoint = identity.get("producer_checkpoint")
        if not isinstance(authority, Mapping) or not isinstance(checkpoint, Mapping):
            raise AlignmentError(f"Stage-C {identity_key} authority is malformed")
        schema = identity.get("schema_version")
        replayed = _operator_identity(
            Path(str(authority["path"])),
            Path(str(checkpoint["path"])),
            require_current_target=(identity_key == "target_policy_target_identity"),
            identity_schema=str(schema),
            target_execution=(
                identity.get("target_execution")
                if schema
                in {
                    OPERATOR_IDENTITY_SCHEMA_V2,
                    OPERATOR_IDENTITY_SCHEMA_V3,
                    OPERATOR_IDENTITY_SCHEMA_V4,
                }
                and identity_key == "target_policy_target_identity"
                else None
            ),
        )
        if replayed != identity:
            raise AlignmentError(f"Stage-C {identity_key} no longer replays")
    overlay_ref = plan.get("eligibility_overlay")
    if not isinstance(overlay_ref, Mapping):
        raise AlignmentError("Stage-C plan lacks eligibility overlay")
    overlay_path, overlay = _load_json(
        Path(str(overlay_ref["path"])), where="Stage-C eligibility overlay"
    )
    overlay_unsigned = dict(overlay)
    overlay_stated = overlay_unsigned.pop("overlay_sha256", None)
    if (
        overlay.get("schema_version") != OVERLAY_SCHEMA
        or overlay_stated != _value_sha256(overlay_unsigned)
        or overlay_ref.get("file_sha256") != _file_sha256(overlay_path)
        or overlay_ref.get("overlay_sha256") != overlay_stated
    ):
        raise AlignmentError("Stage-C eligibility overlay drifted")
    for artifact in overlay["artifacts"].values():
        artifact_path = _regular_file(
            Path(str(artifact["path"])), where="Stage-C overlay artifact"
        )
        if (
            artifact.get("file_sha256") != _file_sha256(artifact_path)
            or artifact.get("size_bytes") != artifact_path.stat().st_size
        ):
            raise AlignmentError("Stage-C overlay artifact bytes drifted")
    subset = plan.get("subset", {}).get("artifact")
    if not isinstance(subset, Mapping):
        raise AlignmentError("Stage-C plan lacks selected subset")
    subset_path = _regular_file(Path(str(subset["path"])), where="Stage-C subset")
    if (
        subset.get("file_sha256") != _file_sha256(subset_path)
        or subset.get("size_bytes") != subset_path.stat().st_size
    ):
        raise AlignmentError("Stage-C selected subset bytes drifted")
    with np.load(subset_path, allow_pickle=False) as arrays:
        count = len(arrays["row_index"])
        requested_rows = int(plan["subset"].get("requested_rows", -1))
        selection = plan["subset"].get("game_first_selection")
        if not isinstance(selection, Mapping):
            raise AlignmentError("Stage-C game-first selection receipt is missing")
        selection_unsigned = dict(selection)
        selection_stated = selection_unsigned.pop("selection_sha256", None)
        if (
            count != int(plan["subset"]["selected_rows"])
            or count != requested_rows
            or any(len(arrays[name]) != count for name in arrays.files)
            or np.unique(arrays["identity_sha256"]).size != count
            or selection.get("schema_version")
            != "a1-stage-c-game-first-selection-v1"
            or selection_stated != _value_sha256(selection_unsigned)
            or int(selection.get("breadth_root_count", -1))
            + int(selection.get("extra_root_count", -1))
            != count
            or int(selection.get("requested_root_budget", -1))
            != int(plan["subset"].get("requested_rows", -1))
            or int(selection.get("max_rows_per_game", -1))
            != int(plan["subset"].get("max_rows_per_game", -1))
            or not {"game_seed", "decision_index", "phase"} <= set(arrays.files)
        ):
            raise AlignmentError("Stage-C selected subset row identity drifted")
        sealed_breadth = _verify_stage_c_root_breadth_inventory(
            selection.get("root_breadth"),
            selected_rows=count,
        )
        target_coverage = validation_scope.get("target_coverage_receipt")
        validation_breadth = sealed_breadth["scopes"]["validation"]
        if (
            not isinstance(target_coverage, Mapping)
            or target_coverage.get("root_breadth_inventory_sha256")
            != sealed_breadth["inventory_sha256"]
            or target_coverage.get("selected_validation_root_count")
            != validation_breadth["selected_root_count"]
            or target_coverage.get("selected_validation_game_count")
            != validation_breadth["selected_game_count"]
        ):
            raise AlignmentError("Stage-C learner validation target coverage drifted")
        source_data = train_bc.MemmapCorpus(
            Path(str(overlay["corpus"]["path"])).resolve(strict=True)
        )
        population_game_seeds = np.asarray(
            source_data["game_seed"], dtype=np.int64
        )
        population_decision_indices = np.asarray(
            source_data["decision_index"], dtype=np.int64
        )
        if admission_schema == POST_WAVE_CORPUS_ADMISSION_SCHEMA:
            (
                reconstructable_game_seeds,
                replayed_trace_qualification,
            ) = _qualify_stage_c_game_traces(
                game_seeds=population_game_seeds,
                decision_indices=population_decision_indices,
            )
            if plan.get("source_game_trace_qualification") != (
                replayed_trace_qualification
            ):
                raise AlignmentError(
                    "Stage-C source game-trace qualification drifted"
                )
        else:
            reconstructable_game_seeds = np.unique(population_game_seeds)
            if "source_game_trace_qualification" in plan:
                raise AlignmentError(
                    "legacy Stage-C plan unexpectedly changed its trace receipt"
                )
        selected_game_seeds = np.asarray(arrays["game_seed"], dtype=np.int64)
        if np.setdiff1d(selected_game_seeds, reconstructable_game_seeds).size:
            raise AlignmentError(
                "Stage-C selected subset contains an unreconstructable game trace"
            )
        validation_game_seeds = np.asarray(
            validation["game_seeds"], dtype=np.int64
        )
        reconstructable_validation_game_seeds = np.intersect1d(
            validation_game_seeds,
            reconstructable_game_seeds,
            assume_unique=True,
        )
        resolved_root_count = _resolve_stage_c_root_budget(
            requested_rows=requested_rows,
            admission_schema=admission_schema,
            population_game_seeds=population_game_seeds,
            validation_game_seeds=validation_game_seeds,
        )
        minimum_root_count = _minimum_stage_c_root_budget(
            population_game_seeds=reconstructable_game_seeds,
            validation_game_seeds=reconstructable_validation_game_seeds,
        )
        if (
            resolved_root_count != count
            or int(selection.get("minimum_breadth_root_count", -1))
            != minimum_root_count
        ):
            raise AlignmentError("Stage-C selected subset root budget drifted")
        replayed_breadth = _stage_c_root_breadth_inventory(
            corpus_game_seeds=reconstructable_game_seeds,
            validation_game_seeds=reconstructable_validation_game_seeds,
            selected_game_seeds=selected_game_seeds,
            selected_decision_indices=np.asarray(
                arrays["decision_index"], dtype=np.int64
            ),
            selected_phases=np.asarray(arrays["phase"]).astype(str),
        )
        if replayed_breadth != sealed_breadth:
            raise AlignmentError("Stage-C selected subset root breadth does not replay")
    return {"path": str(plan_path), "file_sha256": _file_sha256(plan_path), **plan}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="write quarantine overlay and sealed subset")
    plan.add_argument("--coherent-corpus-admission", required=True, type=Path)
    plan.add_argument("--target-operator-contract", required=True, type=Path)
    plan.add_argument("--target-checkpoint", required=True, type=Path)
    plan.add_argument(
        "--subset-rows",
        type=int,
        default=None,
        help=(
            "exact root count; defaults to 65,536 for the legacy admission and "
            "to the deterministic breadth minimum for a post-wave admission"
        ),
    )
    plan.add_argument("--selection-seed", type=int, default=20260715)
    plan.add_argument("--max-rows-per-game", type=int, default=16)
    plan.add_argument("--chunks", type=int, default=64)
    plan.add_argument("--output-root", required=True, type=Path)
    plan.add_argument("--write", required=True, type=Path)
    verify = sub.add_parser("verify", help="replay a sealed Stage-C plan")
    verify.add_argument("--plan", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            result = _build_plan(args)
            _write_json_immutable(args.write, result)
        else:
            result = _verify_plan(args.plan)
    except (
        AlignmentError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Stage-C teacher alignment refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
