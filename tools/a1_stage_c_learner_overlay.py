#!/usr/bin/env python3
"""Export and materialize a Stage-C coherent-policy learner overlay.

Stage C emits a sparse, authenticated patch keyed by absolute corpus row.  The
learner consumes a normal ``MemmapCorpus``.  This tool joins those two ABIs
without copying the large observation/entity payloads and, critically, without
letting historical policy targets remain active:

* every base row remains available to the terminal-outcome/value objective;
* policy weight and policy tensors are zero for every non-reanalysed row;
* qualified rows receive the coherent-n128 target, prior and score evidence;
* qualified rows receive distinct all-legal completed-Q values/masks, bound to
  their row identities, coherent operator and duplicate-search reliability;
* all other memmap payloads are hard-linked byte-for-byte from the base corpus.

``export`` runs beside the completed Stage-C merge, where the full receipt DAG
is still replayable.  It creates a portable content-addressed bundle.  After
that bundle is copied host-to-host, ``materialize`` binds it to the exact base
corpus and emits a normal authenticated memmap plus a derived diagnostic
admission that the existing one-dose learner can consume unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for root in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools import a1_b200_active_policy_campaign as active_campaign  # noqa: E402
from tools import a1_post_wave_stage_c_admission as post_wave_admission  # noqa: E402
from tools import a1_stage_c_reanalysis_executor as stage_c  # noqa: E402
from tools import a1_stage_c_teacher_alignment as alignment  # noqa: E402
from tools import train_bc  # noqa: E402
from catan_zero.rl.memmap_corpus import COMPLETED_Q_COLUMN_SCHEMAS  # noqa: E402
from catan_zero.rl.target_reliability import (  # noqa: E402
    TARGET_RELIABILITY_COLUMNS,
    TARGET_RELIABILITY_SCHEMA,
    TARGET_RELIABILITY_VERSION,
)


EXPORT_SCHEMA = "a1-stage-c-learner-overlay-export-v4"
MATERIALIZATION_SCHEMA = "a1-stage-c-policy-overlay-materialization-v4"
ADMISSION_OVERLAY_SCHEMA = "a1-stage-c-policy-overlay-admission-binding-v4"
COMPLETED_Q_BINDING_SCHEMA = "a1-stage-c-completed-q-binding-v1"
SAMPLING_SCHEMA = "a1-stage-c-policy-sampling-distribution-v2"
ROOT_BREADTH_SCHEMA = alignment.ROOT_BREADTH_SCHEMA
POLICY_TEACHER = "stage_c_coherent_n128_reanalysis"
SAMPLING_COLUMN = "stage_c_policy_sampling_weight"
COMPLETED_Q_VALUE_COLUMN = "completed_q_values"
COMPLETED_Q_MASK_COLUMN = "completed_q_mask"
SAMPLING_ARMS = frozenset({"PRODUCTION_WEIGHTED", "STRATEGIC_BALANCED"})
DEFAULT_PRODUCTION_WEIGHT_CAP = 4.0
SUPPORTED_BASE_ADMISSION_SCHEMAS = frozenset(
    {
        active_campaign.ADMISSION_SCHEMA,
        post_wave_admission.ADMISSION_SCHEMA,
    }
)
ROOT_BREADTH_REQUIRED_PHASES = alignment.ROOT_BREADTH_REQUIRED_PHASES
ROOT_BREADTH_DECISION_BINS = alignment.ROOT_BREADTH_DECISION_BINS
ROOT_BREADTH_CONTRACT = alignment.ROOT_BREADTH_CONTRACT
REWRITTEN_COLUMNS = frozenset(
    {
        "policy_weight_multiplier",
        "prior_policy",
        "target_policy",
        "target_policy_mask",
        "target_scores",
        "target_scores_mask",
        COMPLETED_Q_VALUE_COLUMN,
        COMPLETED_Q_MASK_COLUMN,
        "teacher_name",
    }
)
POLICY_RAGGED_COLUMNS = {
    "prior_policy": ("prior_policy_flat", 0.0),
    "target_policy": ("target_policy_flat", 0.0),
    "target_policy_mask": ("target_policy_mask_flat", False),
    "target_scores": ("target_scores_flat", np.nan),
    "target_scores_mask": ("target_scores_mask_flat", False),
}
COMPLETED_Q_RAGGED_COLUMNS = {
    COMPLETED_Q_VALUE_COLUMN: ("completed_q_values_flat", np.nan),
    COMPLETED_Q_MASK_COLUMN: ("completed_q_mask_flat", False),
}
RAGGED_PATCH_COLUMNS = {
    **POLICY_RAGGED_COLUMNS,
    **COMPLETED_Q_RAGGED_COLUMNS,
}
OPTIONAL_FIXED_PATCH_COLUMNS = {
    "simulations_used": "simulations_used",
    "used_full_search": "used_full_search",
    "root_value": "root_value",
    "root_value_mask": "root_value_mask",
    "root_prior_value": "root_prior_value",
    "root_prior_value_mask": "root_prior_value_mask",
    **{name: name for name in TARGET_RELIABILITY_COLUMNS},
}


class OverlayError(RuntimeError):
    """A Stage-C export, base corpus, or derived overlay is invalid."""


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
        games.ndim != 1
        or decisions.shape != games.shape
        or phases.shape != games.shape
        or population.size == 0
    ):
        raise OverlayError("Stage-C root-breadth inputs are malformed")
    if np.any(decisions < 0):
        raise OverlayError("Stage-C selected decision index is negative")
    selected_unique, roots_per_game = np.unique(games, return_counts=True)
    if np.setdiff1d(selected_unique, population).size:
        raise OverlayError("Stage-C selected root references a game outside the corpus")

    phase_counts = {
        phase: int(np.count_nonzero(phases == phase))
        for phase in ROOT_BREADTH_REQUIRED_PHASES
    }
    unknown_phases = sorted(set(phases.tolist()) - set(ROOT_BREADTH_REQUIRED_PHASES))
    decision_counts = {}
    for name, start, stop in ROOT_BREADTH_DECISION_BINS:
        mask = decisions >= int(start)
        if stop is not None:
            mask &= decisions < int(stop)
        decision_counts[name] = int(np.count_nonzero(mask))

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


def _stage_c_root_breadth_inventory(
    *,
    corpus_game_seeds: np.ndarray,
    validation_game_seeds: np.ndarray,
    selected_game_seeds: np.ndarray,
    selected_decision_indices: np.ndarray,
    selected_phases: np.ndarray,
) -> dict[str, Any]:
    """Bind realized policy-root breadth to the full train/validation populations."""

    corpus_games = np.asarray(corpus_game_seeds, dtype=np.int64)
    validation_games = np.unique(np.asarray(validation_game_seeds, dtype=np.int64))
    selected_games = np.asarray(selected_game_seeds, dtype=np.int64)
    selected_decisions = np.asarray(selected_decision_indices, dtype=np.int64)
    selected_phase_values = np.asarray(selected_phases).astype(str, copy=False)
    if (
        corpus_games.ndim != 1
        or selected_games.ndim != 1
        or selected_decisions.shape != selected_games.shape
        or selected_phase_values.shape != selected_games.shape
    ):
        raise OverlayError("Stage-C root-breadth arrays are not row-aligned")
    all_games = np.unique(corpus_games)
    if np.setdiff1d(validation_games, all_games).size:
        raise OverlayError(
            "Stage-C validation manifest names a game outside the corpus"
        )
    training_games = np.setdiff1d(all_games, validation_games)
    selected_validation = np.isin(selected_games, validation_games)
    scopes = {
        "training": _root_breadth_scope(
            population_game_seeds=training_games,
            selected_game_seeds=selected_games[~selected_validation],
            selected_decision_indices=selected_decisions[~selected_validation],
            selected_phases=selected_phase_values[~selected_validation],
        ),
        "validation": _root_breadth_scope(
            population_game_seeds=validation_games,
            selected_game_seeds=selected_games[selected_validation],
            selected_decision_indices=selected_decisions[selected_validation],
            selected_phases=selected_phase_values[selected_validation],
        ),
    }
    failures = _root_breadth_failures(scopes)
    inventory: dict[str, Any] = {
        "schema_version": ROOT_BREADTH_SCHEMA,
        "contract": copy.deepcopy(ROOT_BREADTH_CONTRACT),
        "scopes": scopes,
        "passed": not failures,
        "failures": failures,
    }
    inventory["inventory_sha256"] = _value_sha256(inventory)
    return inventory


def _root_breadth_failures(scopes: Mapping[str, Mapping[str, Any]]) -> list[str]:
    failures: list[str] = []
    for scope_name, scope in scopes.items():
        if float(scope["unique_game_fraction"]) < float(
            ROOT_BREADTH_CONTRACT["minimum_unique_game_fraction"]
        ):
            failures.append(f"{scope_name}:unique_game_fraction")
        if int(scope["roots_per_represented_game"]["minimum"]) < int(
            ROOT_BREADTH_CONTRACT["minimum_roots_per_represented_game"]
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


def _verify_stage_c_root_breadth_inventory(
    value: object, *, selected_rows: int
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OverlayError("Stage-C root-breadth inventory is missing")
    unsigned = dict(value)
    stated = unsigned.pop("inventory_sha256", None)
    scopes = value.get("scopes")
    malformed = (
        value.get("schema_version") != ROOT_BREADTH_SCHEMA
        or stated != _value_sha256(unsigned)
        or value.get("contract") != ROOT_BREADTH_CONTRACT
        or not isinstance(scopes, dict)
        or set(scopes) != set(ROOT_BREADTH_CONTRACT["required_scopes"])
        or isinstance(selected_rows, bool)
        or int(selected_rows) < 0
    )
    if malformed:
        raise OverlayError("Stage-C root-breadth inventory failed or drifted")
    assert isinstance(scopes, dict)
    required_phases = set(ROOT_BREADTH_REQUIRED_PHASES)
    required_bins = {name for name, _start, _stop in ROOT_BREADTH_DECISION_BINS}
    for scope in scopes.values():
        if not isinstance(scope, dict):
            raise OverlayError("Stage-C root-breadth inventory failed or drifted")
        population_count = int(scope.get("population_game_count", -1))
        root_count = int(scope.get("selected_root_count", -1))
        game_count = int(scope.get("selected_game_count", -1))
        roots_per_game = scope.get("roots_per_represented_game")
        phase_counts = scope.get("phase_counts")
        phase_fractions = scope.get("phase_fractions")
        decision_counts = scope.get("decision_index_bin_counts")
        decision_fractions = scope.get("decision_index_bin_fractions")
        if (
            population_count <= 0
            or root_count < 0
            or game_count < 0
            or game_count > population_count
            or int(scope.get("missing_game_count", -1)) != population_count - game_count
            or not math.isclose(
                float(scope.get("unique_game_fraction", math.nan)),
                game_count / population_count,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not isinstance(roots_per_game, dict)
            or not isinstance(phase_counts, dict)
            or set(phase_counts) != required_phases
            or not isinstance(phase_fractions, dict)
            or set(phase_fractions) != required_phases
            or not isinstance(decision_counts, dict)
            or set(decision_counts) != required_bins
            or not isinstance(decision_fractions, dict)
            or set(decision_fractions) != required_bins
            or not isinstance(scope.get("unknown_phases"), list)
        ):
            raise OverlayError("Stage-C root-breadth inventory failed or drifted")
        minimum = int(roots_per_game.get("minimum", -1))
        maximum = int(roots_per_game.get("maximum", -1))
        mean = float(roots_per_game.get("mean", math.nan))
        expected_mean = root_count / game_count if game_count else 0.0
        denominator = max(root_count, 1)
        if (
            minimum < 0
            or maximum < minimum
            or (game_count == 0 and (minimum != 0 or maximum != 0))
            or (game_count > 0 and (minimum == 0 or root_count < game_count * minimum))
            or not math.isclose(mean, expected_mean, rel_tol=0.0, abs_tol=1.0e-12)
            or sum(int(count) for count in phase_counts.values())
            + len(scope["unknown_phases"])
            != root_count
            or sum(int(count) for count in decision_counts.values()) != root_count
            or any(
                int(count) < 0
                or not math.isclose(
                    float(phase_fractions[name]),
                    int(count) / denominator,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                for name, count in phase_counts.items()
            )
            or any(
                int(count) < 0
                or not math.isclose(
                    float(decision_fractions[name]),
                    int(count) / denominator,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                for name, count in decision_counts.items()
            )
        ):
            raise OverlayError("Stage-C root-breadth inventory failed or drifted")
    if (
        sum(int(scope["selected_root_count"]) for scope in scopes.values())
        != int(selected_rows)
        or value.get("passed") is not True
        or value.get("failures") != []
        or _root_breadth_failures(scopes) != []
    ):
        raise OverlayError("Stage-C root-breadth inventory failed or drifted")
    return copy.deepcopy(value)


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


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "file_sha256": _file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _load_json(path: Path, *, where: str) -> tuple[Path, dict[str, Any]]:
    lexical = path.expanduser()
    if lexical.is_symlink() or not lexical.is_file():
        raise OverlayError(f"{where} must be a regular file: {lexical}")
    resolved = lexical.resolve(strict=True)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OverlayError(f"cannot read {where}: {error}") from error
    if not isinstance(payload, dict):
        raise OverlayError(f"{where} must contain one JSON object")
    return resolved, payload


def _source_policy_semantics(admission: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize immutable source-policy authority across admission generations.

    The historical 8,192-game admission authorizes its stored coherent targets.
    The production post-wave admission deliberately does not: its states and
    outcomes are reusable, but every learner-visible policy target must come
    from the Stage-C patch.  Keeping these cases explicit prevents a schema
    adapter from accidentally reactivating the quarantined historical rows.
    """

    schema = admission.get("schema_version")
    corpus = admission.get("corpus")
    if not isinstance(corpus, Mapping):
        raise OverlayError("Stage-C base admission has no corpus semantics")
    if schema == active_campaign.ADMISSION_SCHEMA:
        policy = admission.get("policy_distillation_contract")
        if (
            not isinstance(policy, Mapping)
            or corpus.get("stored_policy_target_distillation_eligible") is not True
            or policy.get("coherent_public_n128_only") is not True
            or policy.get("legacy_pimc_rows_allowed") is not False
        ):
            raise OverlayError("legacy Stage-C base policy authority drifted")
        result = {
            "source_admission_schema": str(schema),
            "stored_policy_target_distillation_eligible": True,
            "stored_policy_targets_are_historical_operator_only": False,
            "current_teacher_requires_reanalysis": False,
            "state_reanalysis_eligible": bool(
                corpus.get("state_reanalysis_eligible")
            ),
            "legacy_pimc_rows_allowed": False,
        }
    elif schema == post_wave_admission.ADMISSION_SCHEMA:
        policy = admission.get("policy_target_policy")
        if (
            not isinstance(policy, Mapping)
            or corpus.get("stored_policy_target_distillation_eligible") is not False
            or corpus.get("state_reanalysis_eligible") is not True
            or policy.get("stored_targets_are_historical_operator_only") is not True
            or policy.get("current_teacher_requires_reanalysis") is not True
            or policy.get("legacy_pimc_rows_allowed") is not False
        ):
            raise OverlayError("post-wave Stage-C policy quarantine drifted")
        result = {
            "source_admission_schema": str(schema),
            "stored_policy_target_distillation_eligible": False,
            "stored_policy_targets_are_historical_operator_only": True,
            "current_teacher_requires_reanalysis": True,
            "state_reanalysis_eligible": True,
            "legacy_pimc_rows_allowed": False,
        }
    else:
        raise OverlayError(f"unsupported Stage-C base admission schema: {schema!r}")
    result["semantics_sha256"] = _value_sha256(result)
    return result


def _load_base_admission(
    path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Dispatch to the schema's original verifier; never relax legacy checks."""

    candidate_path, candidate = _load_json(path, where="Stage-C base admission")
    schema = candidate.get("schema_version")
    try:
        if schema == active_campaign.ADMISSION_SCHEMA:
            resolved, admission = active_campaign._load_admission(  # noqa: SLF001
                candidate_path
            )
        elif schema == post_wave_admission.ADMISSION_SCHEMA:
            resolved, admission = post_wave_admission.verify_admission(candidate_path)
        else:
            raise OverlayError(
                f"unsupported Stage-C base admission schema: {schema!r}"
            )
    except (
        active_campaign.CampaignError,
        post_wave_admission.AdmissionError,
    ) as error:
        raise OverlayError(f"Stage-C base admission refused: {error}") from error
    return resolved, admission, _source_policy_semantics(admission)


def _load_plan_source_admission(
    plan: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Replay the plan-bound admission and its row-level quarantine decision."""

    reference = plan.get("source_corpus_admission")
    if not isinstance(reference, Mapping):
        raise OverlayError("Stage-C plan lost its source admission")
    path, admission, semantics = _load_base_admission(
        Path(str(reference.get("path", "")))
    )
    if (
        _file_sha256(path) != reference.get("file_sha256")
        or admission.get("admission_sha256") != reference.get("admission_sha256")
    ):
        raise OverlayError("Stage-C plan source admission bytes drifted")

    overlay_ref = plan.get("eligibility_overlay")
    if not isinstance(overlay_ref, Mapping):
        raise OverlayError("Stage-C plan lost its eligibility overlay")
    overlay_path, eligibility = _load_json(
        Path(str(overlay_ref.get("path", ""))),
        where="Stage-C eligibility overlay",
    )
    counts = eligibility.get("counts")
    target_matches = plan.get("target_identity_matches_stored_policy")
    if (
        _file_sha256(overlay_path) != overlay_ref.get("file_sha256")
        or eligibility.get("overlay_sha256") != overlay_ref.get("overlay_sha256")
        or not isinstance(counts, Mapping)
        or eligibility.get("policy_quarantine_changes_value_eligibility") is not False
        or eligibility.get(
            "policy_quarantine_changes_state_reanalysis_eligibility"
        )
        is not False
        or not isinstance(target_matches, bool)
    ):
        raise OverlayError("Stage-C plan policy quarantine evidence drifted")
    active = int(counts.get("stored_policy_active_rows", -1))
    eligible = int(counts.get("stored_policy_eligible_rows", -1))
    quarantined = int(counts.get("stored_policy_quarantined_rows", -1))
    if (
        active < 0
        or eligible < 0
        or quarantined < 0
        or eligible + quarantined != active
        or (target_matches and quarantined != 0)
        or (not target_matches and (eligible != 0 or quarantined != active))
    ):
        raise OverlayError("Stage-C row-level policy quarantine counts drifted")
    if (
        semantics["current_teacher_requires_reanalysis"] is True
        and target_matches is not False
    ):
        raise OverlayError(
            "post-wave admission requires a different reanalysed target operator"
        )
    semantics = {
        **semantics,
        "target_identity_matches_stored_policy": target_matches,
        "stored_policy_active_rows": active,
        "stored_policy_eligible_rows": eligible,
        "stored_policy_quarantined_rows": quarantined,
        "derived_overlay_historical_policy_targets_active": False,
    }
    semantics.pop("semantics_sha256", None)
    semantics["semantics_sha256"] = _value_sha256(semantics)
    return path, admission, semantics


def _completed_q_binding(
    *,
    merge: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    row_identity_sha256: str,
) -> dict[str, Any]:
    """Bind learner-visible completed-Q bytes to rows and one sealed operator.

    ``target_scores`` remains the sparse raw visited-Q column. This contract
    gives the all-legal-action completed-Q vector a distinct name and authority
    so a future learner objective cannot accidentally consume legacy/unbound Q.
    """

    required = {
        "row_index",
        "game_seed",
        "decision_index",
        "identity_sha256",
        "legal_action_offsets",
        "legal_action_ids_flat",
        "completed_q_values_flat",
        "completed_q_mask_flat",
        "target_policy_target_identity_sha256",
        *TARGET_RELIABILITY_COLUMNS,
    }
    missing = required - set(arrays)
    row_count = int(np.asarray(arrays.get("row_index", ())).size)
    offsets = np.asarray(arrays.get("legal_action_offsets", ()), dtype=np.int64)
    values = np.asarray(
        arrays.get("completed_q_values_flat", ()), dtype=np.float32
    )
    mask = np.asarray(arrays.get("completed_q_mask_flat", ()), dtype=np.bool_)
    target_identity = str(merge.get("target_policy_target_identity_sha256", ""))
    reliability = merge.get("reliability")
    operator_contract = merge.get("target_operator_contract")
    row_identities = np.asarray(arrays.get("identity_sha256", ())).astype(
        str, copy=False
    )
    per_row_operator = np.asarray(
        arrays.get("target_policy_target_identity_sha256", ())
    ).astype(str, copy=False)
    reliability_versions = np.asarray(
        arrays.get("target_reliability_version", ()), dtype=np.int64
    )
    row_indices = np.asarray(arrays.get("row_index", ()), dtype=np.int64)
    game_seeds = np.asarray(arrays.get("game_seed", ()), dtype=np.int64)
    decision_indices = np.asarray(
        arrays.get("decision_index", ()), dtype=np.int64
    )
    identity_shapes_valid = (
        row_indices.shape
        == game_seeds.shape
        == decision_indices.shape
        == row_identities.shape
        == (row_count,)
    )
    ordered_row_identities = (
        [
            {
                "row_index": int(row),
                "game_seed": int(seed),
                "decision_index": int(decision),
                "identity_sha256": str(identity),
            }
            for row, seed, decision, identity in zip(
                row_indices,
                game_seeds,
                decision_indices,
                row_identities,
                strict=True,
            )
        ]
        if identity_shapes_valid
        else []
    )
    computed_row_identity_sha256 = (
        _value_sha256(ordered_row_identities) if identity_shapes_valid else ""
    )
    if (
        missing
        or merge.get("patch_schema_version") != stage_c.PATCH_SCHEMA
        or row_count <= 0
        or offsets.shape != (row_count + 1,)
        or int(offsets[0]) != 0
        or bool(np.any(offsets[1:] < offsets[:-1]))
        or int(offsets[-1]) != int(values.size)
        or mask.shape != values.shape
        or not bool(np.all(mask))
        or not bool(np.all(np.isfinite(values)))
        or bool(np.any(values < -1.000001))
        or bool(np.any(values > 1.000001))
        or row_identities.shape != (row_count,)
        or not identity_shapes_valid
        or np.unique(row_identities).size != row_count
        or any(
            not identity.startswith("sha256:") or len(identity) != 71
            for identity in row_identities.tolist()
        )
        or per_row_operator.shape != (row_count,)
        or not target_identity.startswith("sha256:")
        or len(target_identity) != 71
        or not bool(np.all(per_row_operator == target_identity))
        or not isinstance(operator_contract, Mapping)
        or not isinstance(reliability, Mapping)
        or reliability.get("schema_version") != TARGET_RELIABILITY_SCHEMA
        or int(reliability.get("audited_rows", -1))
        + int(reliability.get("unaudited_rows", -1))
        != row_count
        or reliability.get("duplicate_selected_action_applied") is not False
        or reliability_versions.shape != (row_count,)
        or not bool(np.all(reliability_versions == TARGET_RELIABILITY_VERSION))
        or not str(row_identity_sha256).startswith("sha256:")
        or len(str(row_identity_sha256)) != 71
        or str(row_identity_sha256) != computed_row_identity_sha256
    ):
        detail = f"; missing={sorted(missing)}" if missing else ""
        raise OverlayError(
            "Stage-C completed-Q lacks current row/operator/reliability "
            f"authority{detail}"
        )
    return {
        "schema_version": COMPLETED_Q_BINDING_SCHEMA,
        "columns": {
            "values": COMPLETED_Q_VALUE_COLUMN,
            "mask": COMPLETED_Q_MASK_COLUMN,
        },
        "source_patch_columns": {
            "values": "completed_q_values_flat",
            "mask": "completed_q_mask_flat",
            "legal_action_ids": "legal_action_ids_flat",
            "legal_action_offsets": "legal_action_offsets",
        },
        "semantics": {
            "value": (
                "root_actor_perspective_completed_q_after_configured_completion_"
                "and_shrinkage_before_minmax_rescale_and_policy_sigma"
            ),
            "range": [-1.0, 1.0],
            "support": "every_legal_action_on_selected_stage_c_rows",
            "row_alignment": (
                "corpus_row_offsets_and_legal_action_ids_exact_set_reordered"
            ),
            "nonselected_rows": "nan_values_and_false_mask",
            "target_scores_relation": (
                "separate_raw_visited_q_column_never_overwritten"
            ),
            "default_learner_objective": "none_evidence_only",
        },
        "row_identity": {
            "ordered_row_identity_sha256": computed_row_identity_sha256,
            "selected_rows": row_count,
            "identity_column_in_immutable_patch": "identity_sha256",
        },
        "operator_identity": {
            "target_policy_target_identity_sha256": target_identity,
            "target_operator_contract": copy.deepcopy(dict(operator_contract)),
            "q_values_root_perspective": True,
            "legacy_or_unbound_q_allowed": False,
        },
        "reliability_identity": {
            "schema_version": TARGET_RELIABILITY_SCHEMA,
            "version": TARGET_RELIABILITY_VERSION,
            "columns": list(TARGET_RELIABILITY_COLUMNS),
            "receipt_sha256": _value_sha256(reliability),
            "audited_rows": int(reliability["audited_rows"]),
            "unaudited_rows": int(reliability["unaudited_rows"]),
        },
    }


def _ensure_completed_q_columns(meta: dict[str, Any]) -> None:
    columns = meta.get("columns")
    if not isinstance(columns, dict):
        raise OverlayError("base corpus column schema is malformed")
    for name, expected in COMPLETED_Q_COLUMN_SCHEMAS.items():
        current = columns.get(name)
        if current is not None:
            expected_fill = expected["fill"]
            current_fill = current.get("fill") if isinstance(current, Mapping) else None
            try:
                dtype_matches = (
                    isinstance(current, Mapping)
                    and np.dtype(current.get("dtype")) == np.dtype(expected["dtype"])
                )
            except TypeError:
                dtype_matches = False
            fill_matches = (
                isinstance(current_fill, (int, float))
                and not isinstance(current_fill, bool)
                and math.isnan(float(current_fill))
                if isinstance(expected_fill, float) and math.isnan(expected_fill)
                else current_fill == expected_fill
            )
            compatible = (
                isinstance(current, Mapping)
                and set(current) == set(expected)
                and current.get("kind") == expected["kind"]
                and dtype_matches
                and fill_matches
            )
        else:
            compatible = True
        if not compatible:
            raise OverlayError(
                f"existing {name} column does not match Stage-C completed-Q ABI"
            )
        columns[name] = copy.deepcopy(expected)


def _write_json_immutable(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    destination = path.expanduser().resolve(strict=False)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise OverlayError(f"immutable output is not a file: {destination}")
        if destination.read_text(encoding="utf-8") != rendered:
            raise OverlayError(
                f"immutable output already exists with drift: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _copy_immutable(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise OverlayError(f"immutable bundle path is not a file: {destination}")
        if _file_sha256(destination) != _file_sha256(source):
            raise OverlayError(f"immutable bundle path already differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _export_sampling_population(
    *,
    plan: Mapping[str, Any],
    eligibility: Mapping[str, Any],
    source_admission: Mapping[str, Any],
    base_root: Path,
    patch: Mapping[str, np.ndarray],
    output: Path,
) -> dict[str, Any]:
    """Replay the sealed selector and bind admitted/selected stratum counts."""

    subset_ref = plan.get("subset", {}).get("artifact")
    if not isinstance(subset_ref, dict):
        raise OverlayError("Stage-C plan lost its selected-subset artifact")
    subset_source = Path(str(subset_ref.get("path", ""))).resolve(strict=True)
    if _file_sha256(subset_source) != subset_ref.get("file_sha256"):
        raise OverlayError("Stage-C selected-subset artifact drifted")
    subset_path = output / "source_selected_reanalysis_rows.npz"
    _copy_immutable(subset_source, subset_path)
    with np.load(subset_path, allow_pickle=False) as source:
        subset = {name: np.asarray(source[name]) for name in source.files}
    required = {"row_index", "stratum", "phase", "legal_width"}
    if not required <= set(subset):
        raise OverlayError("Stage-C subset lacks sampling strata")
    selected_rows = np.asarray(subset["row_index"], dtype=np.int64)
    patch_rows = np.asarray(patch["row_index"], dtype=np.int64)
    if not set(patch_rows.tolist()) <= set(selected_rows.tolist()):
        raise OverlayError("Stage-C patch contains a row outside the sealed subset")

    data = train_bc.MemmapCorpus(base_root)
    rows = len(data)
    artifacts = eligibility.get("artifacts")
    if not isinstance(artifacts, dict):
        raise OverlayError("Stage-C eligibility artifacts are missing")

    def _bound_array(name: str, dtype) -> np.ndarray:
        ref = artifacts.get(name)
        if not isinstance(ref, dict):
            raise OverlayError(f"Stage-C eligibility lost {name}")
        path = Path(str(ref.get("path", ""))).resolve(strict=True)
        if _file_sha256(path) != ref.get("file_sha256"):
            raise OverlayError(f"Stage-C eligibility {name} drifted")
        value = np.fromfile(path, dtype=dtype)
        if value.shape != (rows,):
            raise OverlayError(f"Stage-C eligibility {name} row count drifted")
        return value

    candidate_mask = _bound_array("reanalysis_candidate", np.bool_)
    policy_status = _bound_array("policy_status", np.uint8)
    candidate_rows = np.flatnonzero(candidate_mask).astype(np.int64)
    legal_widths_all = np.asarray(data["legal_action_ids"].row_counts(), dtype=np.int64)
    reliability_classes, _report = alignment._reliability_inventory(  # noqa: SLF001
        data, row_count=rows
    )
    game_seeds = np.asarray(data["game_seed"][candidate_rows], dtype=np.int64)
    decision_indices = np.asarray(
        data["decision_index"][candidate_rows], dtype=np.int64
    )
    phases = np.asarray(data["phase"][candidate_rows]).astype(str)
    legal_widths = legal_widths_all[candidate_rows]
    surprise = alignment._policy_surprise(data, candidate_rows)  # noqa: SLF001
    try:
        validation_ref = source_admission["corpus"]["validation_manifest"]
        validation = train_bc._load_validation_game_seed_manifest_for_training(  # noqa: SLF001
            Path(str(validation_ref["path"])),
            validation_fraction=0.05,
            validation_seed=17,
            validation_max_samples=0,
            validation_game_seed_ranges=[],
        )
    except (KeyError, SystemExit) as error:
        raise OverlayError(
            f"Stage-C selector validation split refused: {error}"
        ) from error
    replay_kwargs = {
        "rows": candidate_rows,
        "game_seeds": game_seeds,
        "decision_indices": decision_indices,
        "phases": phases,
        "legal_widths": legal_widths,
        "surprise": surprise,
        "reliability_class": reliability_classes[candidate_rows],
        "policy_status": policy_status[candidate_rows],
        "population_game_seeds": np.asarray(data["game_seed"], dtype=np.int64),
        "validation_game_seeds": np.asarray(validation["game_seeds"], dtype=np.int64),
        "selection_seed": int(plan["subset"]["selection_seed"]),
        "max_rows_per_game": int(plan["subset"]["max_rows_per_game"]),
    }
    (
        replay_positions,
        replay_strata,
        replay_selected_counts,
        replay_selection,
    ) = alignment._select_game_first(  # noqa: SLF001
        **replay_kwargs,
        limit=int(plan["subset"]["requested_rows"]),
    )
    if not np.array_equal(
        candidate_rows[replay_positions], selected_rows
    ) or not np.array_equal(
        replay_strata.astype(str), np.asarray(subset["stratum"]).astype(str)
    ):
        raise OverlayError("current code cannot replay the sealed Stage-C selection")
    population_counts = replay_selection["candidate_counts_by_stratum"]
    declared_selected = {
        str(key): int(value) for key, value in plan["subset"]["stratum_counts"].items()
    }
    if replay_selected_counts != declared_selected or replay_selection != plan[
        "subset"
    ].get("game_first_selection"):
        raise OverlayError("replayed selected stratum counts differ from the plan")
    subset_strata = {
        int(row): str(stratum)
        for row, stratum in zip(
            selected_rows.tolist(),
            np.asarray(subset["stratum"]).astype(str).tolist(),
            strict=True,
        )
    }
    materialized_selected: dict[str, int] = {}
    for row in patch_rows.tolist():
        stratum = subset_strata[int(row)]
        materialized_selected[stratum] = materialized_selected.get(stratum, 0) + 1
    if any(
        int(population_counts.get(key, 0)) < count
        for key, count in materialized_selected.items()
    ):
        raise OverlayError("Stage-C selected count exceeds admitted population")
    return {
        "schema_version": SAMPLING_SCHEMA,
        "selected_subset": _artifact(subset_path),
        "population_definition": "selected_game_candidate_population",
        "candidate_rows_before_game_selection": int(candidate_rows.size),
        "selected_game_candidate_rows": int(sum(population_counts.values())),
        "planned_selected_rows": int(selected_rows.size),
        "selected_rows": int(patch_rows.size),
        "selection_seed": int(plan["subset"]["selection_seed"]),
        "max_rows_per_game": int(plan["subset"]["max_rows_per_game"]),
        "candidate_counts_by_stratum": dict(sorted(population_counts.items())),
        "planned_selected_counts_by_stratum": dict(sorted(declared_selected.items())),
        "selected_counts_by_stratum": dict(sorted(materialized_selected.items())),
        "inverse_inclusion_formula": "candidate_count / selected_count",
    }


def _export(args: argparse.Namespace) -> dict[str, Any]:
    try:
        merge = stage_c._verify_merge_receipt(args.merge_receipt)  # noqa: SLF001
        plan = alignment._verify_plan(  # noqa: SLF001
            Path(str(merge["stage_c_plan"]["path"]))
        )
        _overlay_path, eligibility = alignment._load_json(  # noqa: SLF001
            Path(str(plan["eligibility_overlay"]["path"])),
            where="Stage-C eligibility overlay",
        )
    except (stage_c.ExecutorError, alignment.AlignmentError, OSError) as error:
        raise OverlayError(f"Stage-C merge export refused: {error}") from error

    base_root = Path(str(eligibility["corpus"]["path"])).resolve(strict=True)
    source_admission_path, source_admission, source_policy_semantics = (
        _load_plan_source_admission(plan)
    )
    meta_path, meta = _load_json(
        base_root / "corpus_meta.json", where="base corpus metadata"
    )
    if (
        _file_sha256(meta_path) != eligibility["corpus"]["corpus_meta_file_sha256"]
        or meta.get("payload_inventory_sha256")
        != eligibility["corpus"]["payload_inventory_sha256"]
    ):
        raise OverlayError("Stage-C source corpus metadata drifted before export")

    output = args.output_root.expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    patch_source = Path(str(merge["artifact"]["path"])).resolve(strict=True)
    merge_source = args.merge_receipt.expanduser().resolve(strict=True)
    patch_path = output / "stage_c_target_patch.npz"
    merge_path = output / "source_merge_receipt.json"
    _copy_immutable(patch_source, patch_path)
    _copy_immutable(merge_source, merge_path)

    with np.load(patch_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    stage_c._verify_patch_arrays(arrays, receipt=merge)  # noqa: SLF001
    paired_root_columns = {
        "root_value",
        "root_value_mask",
        "root_prior_value",
        "root_prior_value_mask",
    }
    if merge.get(
        "patch_schema_version"
    ) != stage_c.PATCH_SCHEMA or not paired_root_columns <= set(arrays):
        raise OverlayError(
            "Stage-C learner export requires a v3 search patch with paired "
            "root_value/root_prior_value evidence"
        )
    sampling_population = _export_sampling_population(
        plan=plan,
        eligibility=eligibility,
        source_admission=source_admission,
        base_root=base_root,
        patch=arrays,
        output=output,
    )
    identities = [
        {
            "row_index": int(row),
            "game_seed": int(seed),
            "decision_index": int(decision),
            "identity_sha256": str(identity),
        }
        for row, seed, decision, identity in zip(
            arrays["row_index"],
            arrays["game_seed"],
            arrays["decision_index"],
            arrays["identity_sha256"],
            strict=True,
        )
    ]
    row_identity_sha256 = _value_sha256(identities)
    completed_q_binding = _completed_q_binding(
        merge=merge,
        arrays=arrays,
        row_identity_sha256=row_identity_sha256,
    )
    manifest: dict[str, Any] = {
        "schema_version": EXPORT_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "source_merge_receipt": {
            **_artifact(merge_path),
            "receipt_sha256": merge["receipt_sha256"],
            "schema_version": merge["schema_version"],
        },
        "source_stage_c_plan": copy.deepcopy(merge["stage_c_plan"]),
        "source_corpus": {
            "corpus_meta_file_sha256": _file_sha256(meta_path),
            "payload_inventory_sha256": meta["payload_inventory_sha256"],
            "row_count": int(meta["row_count"]),
            "flat_count": int(meta["flat_count"]),
            "legal_width": int(meta["legal_width"]),
            "base_admission": {
                "schema_version": source_admission["schema_version"],
                "file_sha256": _file_sha256(source_admission_path),
                "admission_sha256": source_admission["admission_sha256"],
            },
            "source_policy_semantics": source_policy_semantics,
        },
        "target_policy_target_identity_sha256": merge[
            "target_policy_target_identity_sha256"
        ],
        "target_reanalyzer_checkpoint": copy.deepcopy(
            merge["target_reanalyzer_checkpoint"]
        ),
        "target_operator_contract": copy.deepcopy(merge["target_operator_contract"]),
        "patch": _artifact(patch_path),
        "counts": copy.deepcopy(merge["counts"]),
        "row_identity_sha256": row_identity_sha256,
        "completed_q_binding": completed_q_binding,
        "sampling_population": sampling_population,
        "learner_projection": {
            "policy_rows": "exact_stage_c_reanalysed_rows_only",
            "nonselected_policy_weight": 0.0,
            "selected_policy_weight": 1.0,
            "base_value_rows_retained": True,
            "root_value_patch_consumed": {
                "root_value",
                "root_value_mask",
            }
            <= set(OPTIONAL_FIXED_PATCH_COLUMNS) & set(arrays),
            "root_prior_value_patch_consumed": {
                "root_prior_value",
                "root_prior_value_mask",
            }
            <= set(OPTIONAL_FIXED_PATCH_COLUMNS) & set(arrays),
            "paired_root_value_patch_consumed": paired_root_columns
            <= set(OPTIONAL_FIXED_PATCH_COLUMNS) & set(arrays),
            "completed_q_patch_consumed": True,
            "completed_q_evidence_sidecar_preserved": {
                "completed_q_values_flat",
                "completed_q_mask_flat",
            }
            <= set(arrays),
            "target_reliability_patch_consumed": set(TARGET_RELIABILITY_COLUMNS)
            <= set(OPTIONAL_FIXED_PATCH_COLUMNS) & set(arrays),
            "authoritative_search_fixed_columns": sorted(
                set(OPTIONAL_FIXED_PATCH_COLUMNS) & set(arrays)
            ),
            "rewritten_columns": sorted(REWRITTEN_COLUMNS),
        },
    }
    manifest["export_sha256"] = _value_sha256(manifest)
    _write_json_immutable(output / "manifest.json", manifest)
    return manifest


def _load_export(
    path: Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    manifest_path, manifest = _load_json(path, where="Stage-C learner export")
    unsigned = dict(manifest)
    stated = unsigned.pop("export_sha256", None)
    if (
        manifest.get("schema_version") != EXPORT_SCHEMA
        or manifest.get("diagnostic_only") is not True
        or manifest.get("promotion_eligible") is not False
        or stated != _value_sha256(unsigned)
        or manifest.get("learner_projection", {}).get("rewritten_columns")
        != sorted(REWRITTEN_COLUMNS)
        or manifest.get("learner_projection", {}).get(
            "paired_root_value_patch_consumed"
        )
        is not True
        or manifest.get("learner_projection", {}).get(
            "completed_q_patch_consumed"
        )
        is not True
    ):
        raise OverlayError("Stage-C learner export schema/digest/semantics drifted")
    source_corpus = manifest.get("source_corpus")
    source_admission = (
        source_corpus.get("base_admission")
        if isinstance(source_corpus, Mapping)
        else None
    )
    source_semantics = (
        source_corpus.get("source_policy_semantics")
        if isinstance(source_corpus, Mapping)
        else None
    )
    semantics_unsigned = (
        dict(source_semantics) if isinstance(source_semantics, Mapping) else {}
    )
    semantics_stated = semantics_unsigned.pop("semantics_sha256", None)
    if (
        not isinstance(source_admission, Mapping)
        or source_admission.get("schema_version")
        not in SUPPORTED_BASE_ADMISSION_SCHEMAS
        or not isinstance(source_admission.get("file_sha256"), str)
        or not isinstance(source_admission.get("admission_sha256"), str)
        or not isinstance(source_semantics, Mapping)
        or source_semantics.get("source_admission_schema")
        != source_admission.get("schema_version")
        or source_semantics.get("legacy_pimc_rows_allowed") is not False
        or source_semantics.get(
            "derived_overlay_historical_policy_targets_active"
        )
        is not False
        or semantics_stated != _value_sha256(semantics_unsigned)
    ):
        raise OverlayError("Stage-C exported source-policy authority drifted")

    patch_ref = manifest.get("patch")
    merge_ref = manifest.get("source_merge_receipt")
    subset_ref = manifest.get("sampling_population", {}).get("selected_subset")
    if (
        not isinstance(patch_ref, dict)
        or not isinstance(merge_ref, dict)
        or not isinstance(subset_ref, dict)
    ):
        raise OverlayError("Stage-C learner export artifact bindings are malformed")
    patch = manifest_path.parent / Path(str(patch_ref.get("path", ""))).name
    merge_path = manifest_path.parent / Path(str(merge_ref.get("path", ""))).name
    subset_path = manifest_path.parent / Path(str(subset_ref.get("path", ""))).name
    for artifact_path, reference, where in (
        (patch, patch_ref, "target patch"),
        (merge_path, merge_ref, "merge receipt"),
        (subset_path, subset_ref, "selected subset"),
    ):
        if (
            artifact_path.is_symlink()
            or not artifact_path.is_file()
            or _file_sha256(artifact_path) != reference.get("file_sha256")
            or artifact_path.stat().st_size != int(reference.get("size_bytes", -1))
        ):
            raise OverlayError(f"Stage-C exported {where} bytes drifted")
    _merge_path, merge = _load_json(merge_path, where="exported Stage-C merge receipt")
    if (
        merge.get("schema_version")
        not in {
            stage_c.MERGE_RECEIPT_SCHEMA,
            stage_c.REBOUND_MERGE_RECEIPT_SCHEMA,
        }
        or merge.get("receipt_sha256") != merge_ref.get("receipt_sha256")
        or merge.get("artifact", {}).get("file_sha256") != patch_ref.get("file_sha256")
    ):
        raise OverlayError("exported Stage-C merge binding drifted")
    with np.load(patch, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    stage_c._verify_patch_arrays(arrays, receipt=merge)  # noqa: SLF001
    expected_completed_q_binding = _completed_q_binding(
        merge=merge,
        arrays=arrays,
        row_identity_sha256=str(manifest.get("row_identity_sha256", "")),
    )
    if manifest.get("completed_q_binding") != expected_completed_q_binding:
        raise OverlayError("exported Stage-C completed-Q binding drifted")
    with np.load(subset_path, allow_pickle=False) as source:
        subset = {name: np.asarray(source[name]) for name in source.files}
    return manifest_path, manifest, arrays, subset


def _column_payload_filename(name: str, schema: Mapping[str, Any]) -> str | None:
    kind = schema.get("kind")
    if kind == "implicit_constant":
        return None
    return f"{name}.codes.dat" if kind == "string" else f"{name}.dat"


def _sha_record(path: Path) -> dict[str, Any]:
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _hardlink_payloads(
    base_root: Path,
    output_root: Path,
    columns: Mapping[str, Mapping[str, Any]],
    *,
    rewritten_columns: frozenset[str] | set[str] = REWRITTEN_COLUMNS,
) -> set[str]:
    linked: set[str] = {"row_offsets.dat"}
    for name, schema in columns.items():
        filename = _column_payload_filename(name, schema)
        if filename is not None and name not in rewritten_columns:
            linked.add(filename)
    for filename in sorted(linked):
        source = base_root / filename
        destination = output_root / filename
        try:
            os.link(source, destination)
        except OSError as error:
            raise OverlayError(
                f"cannot hard-link immutable base payload {source} -> {destination}: {error}"
            ) from error
    return linked


def _fixed_memmap(
    root: Path, name: str, schema: Mapping[str, Any], rows: int, *, mode: str
) -> np.memmap:
    if schema.get("kind") != "fixed":
        raise OverlayError(f"required overlay column {name!r} is not fixed")
    inner = tuple(int(value) for value in schema.get("inner_shape", ()))
    return np.memmap(
        root / f"{name}.dat",
        dtype=np.dtype(str(schema["dtype"])),
        mode=mode,
        shape=(rows, *inner),
    )


def _ragged_flat_memmap(
    root: Path, name: str, schema: Mapping[str, Any], flat_count: int, *, mode: str
) -> np.memmap:
    if schema.get("kind") != "ragged2d":
        raise OverlayError(f"required overlay column {name!r} is not ragged2d")
    return np.memmap(
        root / f"{name}.dat",
        dtype=np.dtype(str(schema["dtype"])),
        mode=mode,
        shape=(flat_count,),
    )


def _project_policy_patch(
    *,
    base_root: Path,
    output_root: Path,
    meta: dict[str, Any],
    patch: Mapping[str, np.ndarray],
    selected_sampling_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Write policy plus completed-Q payloads and return projection evidence."""

    _ensure_completed_q_columns(meta)
    columns = meta.get("columns")
    if not isinstance(columns, dict) or not REWRITTEN_COLUMNS <= set(columns):
        raise OverlayError(
            "base corpus lacks required Stage-C policy projection columns: "
            f"{sorted(REWRITTEN_COLUMNS - set(columns or {}))}"
        )
    rows = int(meta["row_count"])
    flat_count = int(meta["flat_count"])
    offsets = np.memmap(
        base_root / "row_offsets.dat",
        dtype=np.int64,
        mode="r",
        shape=(rows + 1,),
    )
    if int(offsets[0]) != 0 or int(offsets[-1]) != flat_count:
        raise OverlayError("base corpus row offsets drifted")
    selected_rows = np.asarray(patch["row_index"], dtype=np.int64)
    if (
        selected_rows.ndim != 1
        or selected_rows.size == 0
        or np.unique(selected_rows).size != selected_rows.size
        or np.any(selected_rows < 0)
        or np.any(selected_rows >= rows)
    ):
        raise OverlayError("Stage-C patch row indices are invalid for base corpus")

    base_seed = _fixed_memmap(
        base_root, "game_seed", columns["game_seed"], rows, mode="r"
    )
    base_decision = _fixed_memmap(
        base_root, "decision_index", columns["decision_index"], rows, mode="r"
    )
    if not np.array_equal(
        np.asarray(base_seed[selected_rows], dtype=np.int64).reshape(-1),
        np.asarray(patch["game_seed"], dtype=np.int64),
    ) or not np.array_equal(
        np.asarray(base_decision[selected_rows], dtype=np.int64).reshape(-1),
        np.asarray(patch["decision_index"], dtype=np.int64),
    ):
        raise OverlayError("Stage-C patch row seed/decision identity differs from base")

    legal_schema = columns.get("legal_action_ids")
    if not isinstance(legal_schema, dict) or legal_schema.get("kind") != "ragged2d":
        raise OverlayError("base legal_action_ids is not a ragged2d column")
    legal_flat = np.memmap(
        base_root / "legal_action_ids.dat",
        dtype=np.dtype(str(legal_schema["dtype"])),
        mode="r",
        shape=(flat_count,),
    )

    policy_weight = _fixed_memmap(
        output_root,
        "policy_weight_multiplier",
        columns["policy_weight_multiplier"],
        rows,
        mode="w+",
    )
    policy_weight[...] = 0
    policy_weight[selected_rows] = 1
    policy_weight.flush()

    if SAMPLING_COLUMN in columns:
        sampling_weight = _fixed_memmap(
            output_root,
            SAMPLING_COLUMN,
            columns[SAMPLING_COLUMN],
            rows,
            mode="w+",
        )
        sampling_weight[...] = 0
        selected_weight = (
            np.ones(selected_rows.size, dtype=np.float32)
            if selected_sampling_weights is None
            else np.asarray(selected_sampling_weights, dtype=np.float32)
        )
        if (
            selected_weight.shape != selected_rows.shape
            or not np.isfinite(selected_weight).all()
            or np.any(selected_weight <= 0.0)
        ):
            raise OverlayError("Stage-C selected sampling weights are invalid")
        sampling_weight[selected_rows] = selected_weight
        sampling_weight.flush()

    outputs: dict[str, np.memmap] = {}
    for name, (_patch_name, fill) in RAGGED_PATCH_COLUMNS.items():
        output = _ragged_flat_memmap(
            output_root, name, columns[name], flat_count, mode="w+"
        )
        output[...] = fill
        outputs[name] = output

    patch_offsets = np.asarray(patch["legal_action_offsets"], dtype=np.int64)
    patch_legal = np.asarray(patch["legal_action_ids_flat"], dtype=np.int64)
    for ordinal, row in enumerate(selected_rows.tolist()):
        base_start, base_stop = int(offsets[row]), int(offsets[row + 1])
        patch_start = int(patch_offsets[ordinal])
        patch_stop = int(patch_offsets[ordinal + 1])
        base_ids = np.asarray(legal_flat[base_start:base_stop], dtype=np.int64)
        patch_ids = patch_legal[patch_start:patch_stop]
        if (
            base_ids.size != patch_ids.size
            or np.unique(base_ids).size != base_ids.size
            or set(base_ids.tolist()) != set(patch_ids.tolist())
        ):
            raise OverlayError(f"Stage-C legal action set differs at corpus row {row}")
        patch_position = {int(action): index for index, action in enumerate(patch_ids)}
        gather = np.asarray(
            [patch_position[int(action)] for action in base_ids], dtype=np.int64
        )
        for name, (patch_name, _fill) in RAGGED_PATCH_COLUMNS.items():
            source = np.asarray(patch[patch_name])[patch_start:patch_stop]
            outputs[name][base_start:base_stop] = source[gather]
    for output in outputs.values():
        output.flush()

    projected_search_columns: list[str] = []
    for column_name, patch_name in OPTIONAL_FIXED_PATCH_COLUMNS.items():
        if column_name not in columns or patch_name not in patch:
            continue
        source = _fixed_memmap(
            base_root, column_name, columns[column_name], rows, mode="r"
        )
        target = _fixed_memmap(
            output_root, column_name, columns[column_name], rows, mode="w+"
        )
        target[...] = source
        values = np.asarray(patch[patch_name])
        if values.shape[0] != selected_rows.size:
            raise OverlayError(f"Stage-C {patch_name} is not aligned to selected rows")
        target[selected_rows] = values
        target.flush()
        projected_search_columns.append(column_name)

    teacher_schema = columns["teacher_name"]
    if teacher_schema.get("kind") != "string":
        raise OverlayError("base teacher_name is not dictionary encoded")
    categories = [str(value) for value in teacher_schema.get("categories", ())]
    if POLICY_TEACHER not in categories:
        categories.append(POLICY_TEACHER)
    teacher_code = categories.index(POLICY_TEACHER)
    source_codes = np.memmap(
        base_root / "teacher_name.codes.dat",
        dtype=np.int32,
        mode="r",
        shape=(rows,),
    )
    target_codes = np.memmap(
        output_root / "teacher_name.codes.dat",
        dtype=np.int32,
        mode="w+",
        shape=(rows,),
    )
    target_codes[...] = source_codes
    target_codes[selected_rows] = teacher_code
    target_codes.flush()
    teacher_schema["categories"] = categories

    target_policy = outputs["target_policy"]
    target_mask = outputs["target_policy_mask"]
    completed_q = outputs[COMPLETED_Q_VALUE_COLUMN]
    completed_q_mask = outputs[COMPLETED_Q_MASK_COLUMN]
    selected_mass = np.asarray(
        [
            float(
                np.asarray(
                    target_policy[int(offsets[row]) : int(offsets[row + 1])]
                ).sum()
            )
            for row in selected_rows
        ]
    )
    if not np.allclose(selected_mass, 1.0, rtol=0.0, atol=1.0e-5):
        raise OverlayError("materialized Stage-C target policies are not normalized")
    if int(np.count_nonzero(np.asarray(target_mask))) != int(
        np.count_nonzero(np.asarray(patch["target_policy_mask_flat"]))
    ):
        raise OverlayError("materialized Stage-C target mask lost support")
    selected_completed_q = np.concatenate(
        [
            np.asarray(
                completed_q[int(offsets[row]) : int(offsets[row + 1])],
                dtype=np.float32,
            )
            for row in selected_rows
        ]
    )
    selected_completed_q_mask = np.concatenate(
        [
            np.asarray(
                completed_q_mask[int(offsets[row]) : int(offsets[row + 1])],
                dtype=np.bool_,
            )
            for row in selected_rows
        ]
    )
    if (
        not bool(np.all(selected_completed_q_mask))
        or not bool(np.all(np.isfinite(selected_completed_q)))
        or int(np.count_nonzero(np.asarray(completed_q_mask)))
        != int(np.count_nonzero(np.asarray(patch["completed_q_mask_flat"])))
    ):
        raise OverlayError("materialized Stage-C completed-Q lost exact support")

    return {
        "selected_rows": int(selected_rows.size),
        "nonselected_policy_disabled_rows": rows - int(selected_rows.size),
        "selected_row_index_sha256": _value_sha256(selected_rows.tolist()),
        "selected_policy_mass_min": float(selected_mass.min()),
        "selected_policy_mass_max": float(selected_mass.max()),
        "base_value_rows_retained": rows,
        "completed_q_rows": int(selected_rows.size),
        "completed_q_legal_actions": int(selected_completed_q.size),
        "completed_q_value_min": float(selected_completed_q.min()),
        "completed_q_value_max": float(selected_completed_q.max()),
        "completed_q_target_scores_separate": True,
        "authoritative_search_fixed_columns": projected_search_columns,
    }


def _updated_inventory(
    *,
    base_meta: Mapping[str, Any],
    output_meta: Mapping[str, Any],
    output_root: Path,
    rewritten_filenames: set[str],
) -> list[dict[str, Any]]:
    base_records = {
        str(record["filename"]): dict(record)
        for record in base_meta.get("payload_inventory", ())
    }
    base_expected = train_bc._expected_memmap_payload_filenames(base_meta)  # noqa: SLF001
    expected = train_bc._expected_memmap_payload_filenames(output_meta)  # noqa: SLF001
    if set(base_records) != base_expected:
        raise OverlayError("base payload inventory differs from its column schema")
    result = []
    for filename in sorted(expected):
        path = output_root / filename
        if filename in rewritten_filenames:
            result.append(_sha_record(path))
        else:
            record = base_records.get(filename)
            if record is None:
                raise OverlayError(
                    f"new payload was not declared rewritten: {filename}"
                )
            if path.stat().st_size != int(record["size_bytes"]):
                raise OverlayError(f"hard-linked payload size drifted: {filename}")
            result.append(record)
    return result


def _unit_mean_capped_weights(raw: np.ndarray, *, cap: float) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or np.any(values <= 0.0)
        or not math.isfinite(float(cap))
        or float(cap) < 1.0
    ):
        raise OverlayError("production sampling weights/cap are invalid")
    # Find the unique scale with mean(min(cap, scale * raw)) == 1.  This keeps
    # both the declared cap and unit mean exact; cap-then-renormalize can exceed
    # its own advertised ceiling.
    low, high = 0.0, 1.0 / float(np.mean(values))
    while float(np.mean(np.minimum(float(cap), high * values))) < 1.0:
        high *= 2.0
    for _ in range(96):
        middle = (low + high) / 2.0
        if float(np.mean(np.minimum(float(cap), middle * values))) < 1.0:
            low = middle
        else:
            high = middle
    result = np.minimum(float(cap), high * values)
    if not math.isclose(float(np.mean(result)), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise OverlayError("capped production weights failed unit normalization")
    return result


def _effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64)
    return float(values.sum() ** 2 / np.square(values).sum())


def _selected_sampling_weights(
    *,
    export: Mapping[str, Any],
    subset: Mapping[str, np.ndarray],
    patch: Mapping[str, np.ndarray],
    selected_validation: np.ndarray,
    arm: str,
    production_weight_cap: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if arm not in SAMPLING_ARMS:
        raise OverlayError(f"unknown Stage-C sampling arm {arm!r}")
    patch_rows = np.asarray(patch["row_index"], dtype=np.int64)
    subset_rows = np.asarray(subset["row_index"], dtype=np.int64)
    subset_strata = np.asarray(subset["stratum"]).astype(str)
    subset_phases = np.asarray(subset["phase"]).astype(str)
    subset_widths = np.asarray(subset["legal_width"], dtype=np.int64)
    lookup = {
        int(row): (str(stratum), str(phase), int(width))
        for row, stratum, phase, width in zip(
            subset_rows.tolist(),
            subset_strata.tolist(),
            subset_phases.tolist(),
            subset_widths.tolist(),
            strict=True,
        )
    }
    try:
        descriptors = [lookup[int(row)] for row in patch_rows.tolist()]
    except KeyError as error:
        raise OverlayError("Stage-C patch row lost its sealed stratum") from error
    population = export.get("sampling_population")
    if not isinstance(population, dict):
        raise OverlayError("Stage-C export has no sampling population")
    candidate_counts = {
        str(key): int(value)
        for key, value in population["candidate_counts_by_stratum"].items()
    }
    selected_counts = {
        str(key): int(value)
        for key, value in population["selected_counts_by_stratum"].items()
    }
    raw = np.asarray(
        [
            candidate_counts[stratum] / selected_counts[stratum]
            for stratum, _, _ in descriptors
        ],
        dtype=np.float64,
    )
    validation = np.asarray(selected_validation, dtype=np.bool_)
    if validation.shape != raw.shape or np.all(validation):
        raise OverlayError("Stage-C selected training split is empty or misaligned")
    train = ~validation
    weights = np.ones(raw.size, dtype=np.float64)
    if arm == "PRODUCTION_WEIGHTED":
        weights[train] = _unit_mean_capped_weights(
            raw[train], cap=float(production_weight_cap)
        )
        # Normalize held-out roots independently. Reusing the training scale or
        # forcing validation to uniform changes the policy measure being
        # validated and made the production-weighted learner look as though it
        # trained on the strategic-balanced objective.
        if np.any(validation):
            weights[validation] = _unit_mean_capped_weights(
                raw[validation], cap=float(production_weight_cap)
            )

    def _mass_by(values: Sequence[str]) -> dict[str, float]:
        labels = np.asarray(values).astype(str)
        return {
            str(label): float(weights[train & (labels == label)].sum())
            for label in sorted(set(labels[train].tolist()))
        }

    phases = np.asarray([phase for _, phase, _ in descriptors]).astype(str)
    width_buckets = np.asarray(
        [alignment._width_bucket(width) for _, _, width in descriptors]  # noqa: SLF001
    ).astype(str)
    train_weights = weights[train]
    report = {
        "schema_version": SAMPLING_SCHEMA,
        "arm": arm,
        "column": SAMPLING_COLUMN,
        "selected_rows": int(raw.size),
        "selected_training_rows": int(np.count_nonzero(train)),
        "selected_validation_rows": int(np.count_nonzero(validation)),
        "inverse_inclusion_formula": "candidate_count / selected_count",
        "normalization_scope": "training_and_validation_roots_independently",
        "production_weight_cap": (
            float(production_weight_cap) if arm == "PRODUCTION_WEIGHTED" else None
        ),
        "raw_inverse_inclusion": {
            "min": float(raw[train].min()),
            "max": float(raw[train].max()),
            "mean": float(raw[train].mean()),
            "effective_sample_size": _effective_sample_size(raw[train]),
        },
        "final_training_weights": {
            "min": float(train_weights.min()),
            "max": float(train_weights.max()),
            "mean": float(train_weights.mean()),
            "effective_sample_size": _effective_sample_size(train_weights),
        },
        "final_validation_weights": (
            {
                "min": float(weights[validation].min()),
                "max": float(weights[validation].max()),
                "mean": float(weights[validation].mean()),
                "effective_sample_size": _effective_sample_size(weights[validation]),
            }
            if np.any(validation)
            else None
        ),
        "training_mass_by_phase": _mass_by(phases),
        "training_mass_by_legal_width_bucket": _mass_by(width_buckets),
    }
    report["sampling_sha256"] = _value_sha256(report)
    return weights.astype(np.float32), report


def _derived_policy_distillation_contract(
    *,
    base_admission: Mapping[str, Any],
    source_policy_semantics: Mapping[str, Any],
    selected_rows: int,
    root_breadth_inventory_sha256: str,
    target_policy_target_identity_sha256: str,
) -> dict[str, Any]:
    """Authorize only the newly materialized Stage-C policy rows."""

    existing = base_admission.get("policy_distillation_contract")
    contract = copy.deepcopy(dict(existing)) if isinstance(existing, Mapping) else {}
    contract.update(
        {
            "coherent_public_n128_only": True,
            "legacy_pimc_rows_allowed": False,
            "policy_active_rows": int(selected_rows),
            "stage_c_reanalysis_only": True,
            "historical_policy_targets_active": False,
            "source_admission_schema": source_policy_semantics[
                "source_admission_schema"
            ],
            "source_stored_policy_target_distillation_eligible": bool(
                source_policy_semantics[
                    "stored_policy_target_distillation_eligible"
                ]
            ),
            "source_stored_policy_quarantined_rows": int(
                source_policy_semantics["stored_policy_quarantined_rows"]
            ),
            "root_breadth_inventory_sha256": root_breadth_inventory_sha256,
            "target_policy_target_identity_sha256": (
                target_policy_target_identity_sha256
            ),
        }
    )
    return contract


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    export_path, export, patch, subset = _load_export(args.export_manifest)
    try:
        (
            base_admission_path,
            base_admission,
            base_source_policy_semantics,
        ) = _load_base_admission(
            args.base_admission
        )
    except OverlayError as error:
        raise OverlayError(f"base coherent admission refused: {error}") from error
    base_root = args.base_corpus.expanduser().resolve(strict=True)
    base_meta_path, base_meta = _load_json(
        base_root / "corpus_meta.json", where="base corpus metadata"
    )
    source_binding = export["source_corpus"]
    exported_admission = source_binding.get("base_admission")
    exported_semantics = source_binding.get("source_policy_semantics")
    base_semantics_unsigned = dict(base_source_policy_semantics)
    base_semantics_unsigned.pop("semantics_sha256", None)
    if (
        not isinstance(exported_admission, Mapping)
        or exported_admission.get("schema_version")
        != base_admission.get("schema_version")
        or exported_admission.get("file_sha256")
        != _file_sha256(base_admission_path)
        or exported_admission.get("admission_sha256")
        != base_admission.get("admission_sha256")
        or not isinstance(exported_semantics, Mapping)
        or any(
            exported_semantics.get(key) != value
            for key, value in base_semantics_unsigned.items()
        )
        or Path(str(base_admission["corpus"]["data_path"])).resolve(strict=True)
        != base_root
        or _file_sha256(base_meta_path) != source_binding["corpus_meta_file_sha256"]
        or base_meta.get("payload_inventory_sha256")
        != source_binding["payload_inventory_sha256"]
        or int(base_meta.get("row_count", -1)) != int(source_binding["row_count"])
        or int(base_meta.get("flat_count", -1)) != int(source_binding["flat_count"])
    ):
        raise OverlayError("portable Stage-C export binds a different base corpus")
    source_policy_semantics = dict(exported_semantics)

    output = args.output_root.expanduser().resolve(strict=False)
    if output.exists():
        raise OverlayError(f"overlay output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        meta = copy.deepcopy(base_meta)
        columns = meta.get("columns")
        if not isinstance(columns, dict):
            raise OverlayError("base corpus column schema is malformed")
        _ensure_completed_q_columns(meta)
        columns[SAMPLING_COLUMN] = {
            "kind": "fixed",
            "dtype": "float32",
            "inner_shape": [],
        }
        validation_ref = base_admission["corpus"]["validation_manifest"]
        try:
            validation = train_bc._load_validation_game_seed_manifest_for_training(  # noqa: SLF001
                Path(str(validation_ref["path"])),
                validation_fraction=0.05,
                validation_seed=17,
                validation_max_samples=0,
                validation_game_seed_ranges=[],
            )
        except SystemExit as error:
            raise OverlayError(
                f"base coherent validation manifest refused: {error}"
            ) from error
        selected_validation = np.isin(
            np.asarray(patch["game_seed"], dtype=np.int64),
            np.asarray(validation["game_seeds"], dtype=np.int64),
        )
        subset_rows = np.asarray(subset["row_index"], dtype=np.int64)
        subset_phases = np.asarray(subset["phase"]).astype(str, copy=False)
        if (
            subset_rows.ndim != 1
            or subset_phases.shape != subset_rows.shape
            or np.unique(subset_rows).size != subset_rows.size
        ):
            raise OverlayError("Stage-C selected subset phase evidence is malformed")
        phase_by_row = {
            int(row): str(phase)
            for row, phase in zip(
                subset_rows.tolist(), subset_phases.tolist(), strict=True
            )
        }
        try:
            selected_phases = np.asarray(
                [phase_by_row[int(row)] for row in patch["row_index"]]
            ).astype(str)
        except KeyError as error:
            raise OverlayError(
                "Stage-C materialized patch row is absent from the selected subset"
            ) from error
        base_data = train_bc.MemmapCorpus(base_root)
        root_breadth = _stage_c_root_breadth_inventory(
            corpus_game_seeds=np.asarray(base_data["game_seed"], dtype=np.int64),
            validation_game_seeds=np.asarray(validation["game_seeds"], dtype=np.int64),
            selected_game_seeds=np.asarray(patch["game_seed"], dtype=np.int64),
            selected_decision_indices=np.asarray(
                patch["decision_index"], dtype=np.int64
            ),
            selected_phases=selected_phases,
        )
        _verify_stage_c_root_breadth_inventory(
            root_breadth, selected_rows=len(patch["row_index"])
        )
        selected_sampling_weights, sampling_report = _selected_sampling_weights(
            export=export,
            subset=subset,
            patch=patch,
            selected_validation=selected_validation,
            arm=str(args.sampling_arm),
            production_weight_cap=float(args.production_weight_cap),
        )
        completed_q_binding = export.get("completed_q_binding")
        if (
            not isinstance(completed_q_binding, dict)
            or completed_q_binding.get("schema_version")
            != COMPLETED_Q_BINDING_SCHEMA
            or completed_q_binding.get("row_identity", {}).get("selected_rows")
            != len(patch["row_index"])
            or completed_q_binding.get("operator_identity", {}).get(
                "target_policy_target_identity_sha256"
            )
            != export["target_policy_target_identity_sha256"]
        ):
            raise OverlayError("Stage-C completed-Q export binding is missing")
        paired_root_columns = {
            "root_value",
            "root_value_mask",
            "root_prior_value",
            "root_prior_value_mask",
        }
        if not paired_root_columns <= set(columns) or not paired_root_columns <= set(
            patch
        ):
            raise OverlayError(
                "Stage-C v3 materialization requires paired root value columns "
                "in both the base corpus and target patch"
            )
        optional_fixed = set(OPTIONAL_FIXED_PATCH_COLUMNS) & set(columns) & set(patch)
        rewritten_columns = set(REWRITTEN_COLUMNS) | optional_fixed | {SAMPLING_COLUMN}
        _hardlink_payloads(
            base_root,
            temporary,
            columns,
            rewritten_columns=rewritten_columns,
        )
        projection = _project_policy_patch(
            base_root=base_root,
            output_root=temporary,
            meta=meta,
            patch=patch,
            selected_sampling_weights=selected_sampling_weights,
        )
        projection["selected_validation_policy_rows"] = int(
            np.count_nonzero(selected_validation)
        )
        projection["selected_training_policy_rows"] = int(
            len(selected_validation) - np.count_nonzero(selected_validation)
        )
        if projection["selected_training_policy_rows"] <= 0:
            raise OverlayError(
                "Stage-C overlay has no policy roots in the training split"
            )
        rewritten_filenames = {
            _column_payload_filename(name, columns[name]) for name in rewritten_columns
        }
        if None in rewritten_filenames:
            raise OverlayError("Stage-C rewritten column unexpectedly has no payload")
        inventory = _updated_inventory(
            base_meta=base_meta,
            output_meta=meta,
            output_root=temporary,
            rewritten_filenames={str(value) for value in rewritten_filenames},
        )
        inventory_sha = _value_sha256(inventory)
        meta["payload_inventory"] = inventory
        meta["payload_inventory_sha256"] = inventory_sha
        stats = meta.setdefault("stats", {})
        if isinstance(stats, dict):
            stats["policy_weight_zero_rows"] = int(meta["row_count"]) - int(
                projection["selected_rows"]
            )
            stats["stage_c_reanalysed_policy_rows"] = int(projection["selected_rows"])
        scan = meta.get("event_history_payload_scan")
        if isinstance(scan, dict):
            scan["payload_inventory_sha256"] = inventory_sha
            scan.pop("scan_sha256", None)
            scan["scan_sha256"] = _value_sha256(scan)
        meta["stage_c_policy_overlay"] = {
            "schema_version": ADMISSION_OVERLAY_SCHEMA,
            "export_sha256": export["export_sha256"],
            "target_policy_target_identity_sha256": export[
                "target_policy_target_identity_sha256"
            ],
            "selected_policy_rows": int(projection["selected_rows"]),
            "nonselected_policy_weight": 0.0,
            "selected_policy_weight": 1.0,
            "base_value_rows_retained": True,
            "paired_root_value_patch_consumed": True,
            "completed_q_patch_consumed": True,
            "completed_q_binding": copy.deepcopy(completed_q_binding),
            "rewritten_columns": sorted(rewritten_columns),
            "sampling_distribution": sampling_report,
            "root_breadth": root_breadth,
        }
        meta_path = temporary / "corpus_meta.json"
        meta_path.write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        final_meta = output / "corpus_meta.json"
        final_receipt = output / "stage_c_policy_overlay.receipt.json"
        receipt: dict[str, Any] = {
            "schema_version": MATERIALIZATION_SCHEMA,
            "diagnostic_only": True,
            "promotion_eligible": False,
            "export": {
                "path": str(export_path),
                "file_sha256": _file_sha256(export_path),
                "export_sha256": export["export_sha256"],
            },
            "base_admission": {
                "path": str(base_admission_path),
                "file_sha256": _file_sha256(base_admission_path),
                "admission_sha256": base_admission["admission_sha256"],
            },
            "base_corpus": {
                "path": str(base_root),
                "corpus_meta_file_sha256": _file_sha256(base_meta_path),
                "payload_inventory_sha256": base_meta["payload_inventory_sha256"],
            },
            "overlay_corpus": {
                "path": str(output),
                "corpus_meta_path": str(final_meta),
                "corpus_meta_file_sha256": _file_sha256(meta_path),
                "payload_inventory_sha256": inventory_sha,
                "row_count": int(meta["row_count"]),
                "flat_count": int(meta["flat_count"]),
            },
            "target_policy_target_identity_sha256": export[
                "target_policy_target_identity_sha256"
            ],
            "target_reanalyzer_checkpoint": copy.deepcopy(
                export["target_reanalyzer_checkpoint"]
            ),
            "target_operator_contract": copy.deepcopy(
                export["target_operator_contract"]
            ),
            "projection": projection,
            "sampling_distribution": sampling_report,
            "root_breadth": root_breadth,
            "rewritten_columns": sorted(rewritten_columns),
            "preserved_columns": sorted(set(columns) - rewritten_columns),
            "non_target_source_columns_mutated": False,
            "base_value_and_outcome_columns_retained": True,
            "paired_root_value_patch_consumed": True,
            "completed_q_patch_consumed": True,
            "completed_q_binding": copy.deepcopy(completed_q_binding),
        }
        receipt["receipt_sha256"] = _value_sha256(receipt)
        _write_json_immutable(temporary / final_receipt.name, receipt)

        admission = copy.deepcopy(base_admission)
        admission.pop("admission_sha256", None)
        admission["status"] = "admitted_for_diagnostic_policy_distillation"
        corpus = admission["corpus"]
        corpus.update(
            {
                "data_path": str(output),
                "corpus_meta_path": str(final_meta),
                "corpus_meta_file_sha256": _file_sha256(meta_path),
                "payload_inventory_sha256": inventory_sha,
                "stored_policy_target_distillation_eligible": True,
                "incompatible_policy_active_rows": 0,
            }
        )
        source_target_policy = admission.pop("policy_target_policy", None)
        if source_target_policy is not None:
            admission["source_policy_target_policy"] = source_target_policy
            admission["policy_target_policy"] = {
                "stored_targets_are_current_stage_c_operator_only": True,
                "historical_policy_targets_active": False,
                "legacy_pimc_rows_allowed": False,
                "target_policy_target_identity_sha256": export[
                    "target_policy_target_identity_sha256"
                ],
            }
        admission["policy_distillation_contract"] = (
            _derived_policy_distillation_contract(
                base_admission=base_admission,
                source_policy_semantics=source_policy_semantics,
                selected_rows=int(projection["selected_rows"]),
                root_breadth_inventory_sha256=root_breadth["inventory_sha256"],
                target_policy_target_identity_sha256=export[
                    "target_policy_target_identity_sha256"
                ],
            )
        )
        admission["stage_c_policy_overlay"] = {
            "schema_version": ADMISSION_OVERLAY_SCHEMA,
            "paired_root_value_patch_consumed": True,
            "completed_q_patch_consumed": True,
            "completed_q_binding": copy.deepcopy(completed_q_binding),
            "materialization_receipt": {
                "path": str(final_receipt),
                "file_sha256": _file_sha256(temporary / final_receipt.name),
                "receipt_sha256": receipt["receipt_sha256"],
            },
            "export": receipt["export"],
            "target_policy_target_identity_sha256": export[
                "target_policy_target_identity_sha256"
            ],
            "selected_policy_rows": int(projection["selected_rows"]),
            "selected_training_policy_rows": int(
                projection["selected_training_policy_rows"]
            ),
            "selected_validation_policy_rows": int(
                projection["selected_validation_policy_rows"]
            ),
            "base_value_rows_retained": True,
            "historical_policy_targets_active": False,
            "source_admission": {
                "path": str(base_admission_path),
                "file_sha256": _file_sha256(base_admission_path),
                "admission_sha256": base_admission["admission_sha256"],
                "schema_version": base_admission["schema_version"],
            },
            "source_policy_semantics": source_policy_semantics,
            "sampling_distribution": sampling_report,
            "root_breadth": root_breadth,
        }
        admission["admission_sha256"] = _value_sha256(admission)
        _write_json_immutable(temporary / "overlay.admission.json", admission)
        os.replace(temporary, output)
        return {
            "receipt": str(final_receipt),
            "receipt_sha256": receipt["receipt_sha256"],
            "admission": str(output / "overlay.admission.json"),
            "admission_sha256": admission["admission_sha256"],
            "corpus": str(output),
            "selected_policy_rows": int(projection["selected_rows"]),
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _verify_overlay_source_authority(
    admission: Mapping[str, Any], overlay: Mapping[str, Any]
) -> None:
    """Replay the immutable base admission behind a post-wave overlay."""

    schema = admission.get("schema_version")
    source_ref = overlay.get("source_admission")
    source_semantics = overlay.get("source_policy_semantics")
    # Preserve verification compatibility for already materialized legacy
    # overlays. New exports of either schema always carry the stronger source
    # binding below.
    if schema == active_campaign.ADMISSION_SCHEMA and source_ref is None:
        return
    if not isinstance(source_ref, Mapping) or not isinstance(
        source_semantics, Mapping
    ):
        raise OverlayError("Stage-C overlay lost its source-policy authority")
    source_path, source_admission, base_semantics = _load_base_admission(
        Path(str(source_ref.get("path", "")))
    )
    base_semantics_unsigned = dict(base_semantics)
    base_semantics_unsigned.pop("semantics_sha256", None)
    source_unsigned = dict(source_semantics)
    source_stated = source_unsigned.pop("semantics_sha256", None)
    if (
        source_ref.get("schema_version") != source_admission.get("schema_version")
        or source_ref.get("schema_version") != schema
        or source_ref.get("file_sha256") != _file_sha256(source_path)
        or source_ref.get("admission_sha256")
        != source_admission.get("admission_sha256")
        or source_stated != _value_sha256(source_unsigned)
        or any(
            source_semantics.get(key) != value
            for key, value in base_semantics_unsigned.items()
        )
        or source_semantics.get(
            "derived_overlay_historical_policy_targets_active"
        )
        is not False
    ):
        raise OverlayError("Stage-C overlay source-policy authority drifted")


def verify_overlay_admission(path: Path) -> dict[str, Any]:
    """Verify the portable Stage-C binding on a derived coherent admission."""

    admission_path, admission = _load_json(path, where="Stage-C overlay admission")
    unsigned = dict(admission)
    stated = unsigned.pop("admission_sha256", None)
    overlay = admission.get("stage_c_policy_overlay")
    completed_q_binding = (
        overlay.get("completed_q_binding") if isinstance(overlay, dict) else None
    )
    corpus = admission.get("corpus")
    policy_contract = admission.get("policy_distillation_contract")
    if (
        admission.get("schema_version") not in SUPPORTED_BASE_ADMISSION_SCHEMAS
        or admission.get("status") != "admitted_for_diagnostic_policy_distillation"
        or admission.get("diagnostic_only") is not True
        or admission.get("promotion_eligible") is not False
        or stated != _value_sha256(unsigned)
        or not isinstance(corpus, Mapping)
        or corpus.get("stored_policy_target_distillation_eligible") is not True
        or corpus.get("incompatible_policy_active_rows") != 0
        or not isinstance(policy_contract, Mapping)
        or not isinstance(overlay, dict)
        or overlay.get("schema_version") != ADMISSION_OVERLAY_SCHEMA
        or overlay.get("historical_policy_targets_active") is not False
        or overlay.get("base_value_rows_retained") is not True
        or overlay.get("paired_root_value_patch_consumed") is not True
        or overlay.get("completed_q_patch_consumed") is not True
        or not isinstance(completed_q_binding, dict)
        or completed_q_binding.get("schema_version") != COMPLETED_Q_BINDING_SCHEMA
        or completed_q_binding.get("semantics", {}).get(
            "default_learner_objective"
        )
        != "none_evidence_only"
        or completed_q_binding.get("operator_identity", {}).get(
            "legacy_or_unbound_q_allowed"
        )
        is not False
        or int(overlay.get("selected_policy_rows", 0)) <= 0
        or int(overlay.get("selected_training_policy_rows", 0)) <= 0
        or int(overlay.get("selected_validation_policy_rows", -1)) < 0
        or int(overlay.get("selected_training_policy_rows", 0))
        + int(overlay.get("selected_validation_policy_rows", 0))
        != int(overlay.get("selected_policy_rows", 0))
        or policy_contract.get("stage_c_reanalysis_only") is not True
        or policy_contract.get("coherent_public_n128_only") is not True
        or policy_contract.get("legacy_pimc_rows_allowed") is not False
        or int(policy_contract.get("policy_active_rows", -1))
        != int(overlay.get("selected_policy_rows", 0))
        or policy_contract.get("target_policy_target_identity_sha256")
        != overlay.get("target_policy_target_identity_sha256")
        or overlay.get("sampling_distribution", {}).get("schema_version")
        != SAMPLING_SCHEMA
        or overlay.get("sampling_distribution", {}).get("arm") not in SAMPLING_ARMS
    ):
        raise OverlayError("Stage-C overlay admission digest/semantics drifted")
    _verify_overlay_source_authority(admission, overlay)
    if overlay.get("source_admission") is not None and (
        policy_contract.get("historical_policy_targets_active") is not False
        or policy_contract.get("source_admission_schema")
        != admission.get("schema_version")
    ):
        raise OverlayError("Stage-C derived policy contract drifted")
    if admission.get("schema_version") == post_wave_admission.ADMISSION_SCHEMA:
        policy_target = admission.get("policy_target_policy")
        source_policy_target = admission.get("source_policy_target_policy")
        if (
            not isinstance(policy_target, Mapping)
            or not isinstance(source_policy_target, Mapping)
            or policy_target.get(
                "stored_targets_are_current_stage_c_operator_only"
            )
            is not True
            or policy_target.get("historical_policy_targets_active") is not False
            or policy_target.get("legacy_pimc_rows_allowed") is not False
            or policy_target.get("target_policy_target_identity_sha256")
            != overlay.get("target_policy_target_identity_sha256")
            or source_policy_target.get(
                "stored_targets_are_historical_operator_only"
            )
            is not True
            or source_policy_target.get("current_teacher_requires_reanalysis")
            is not True
        ):
            raise OverlayError("post-wave Stage-C policy authority drifted")
    receipt_ref = overlay.get("materialization_receipt")
    if not isinstance(receipt_ref, dict):
        raise OverlayError("Stage-C overlay admission lost materialization receipt")
    receipt_path = Path(str(receipt_ref.get("path", ""))).resolve(strict=True)
    _receipt_path, receipt = _load_json(
        receipt_path, where="Stage-C overlay materialization receipt"
    )
    receipt_unsigned = dict(receipt)
    receipt_stated = receipt_unsigned.pop("receipt_sha256", None)
    selected_rows = int(overlay.get("selected_policy_rows", 0))
    root_breadth = _verify_stage_c_root_breadth_inventory(
        overlay.get("root_breadth"), selected_rows=selected_rows
    )
    if (
        receipt.get("schema_version") != MATERIALIZATION_SCHEMA
        or receipt_stated != _value_sha256(receipt_unsigned)
        or _file_sha256(receipt_path) != receipt_ref.get("file_sha256")
        or receipt_stated != receipt_ref.get("receipt_sha256")
        or receipt.get("target_policy_target_identity_sha256")
        != overlay.get("target_policy_target_identity_sha256")
        or receipt.get("root_breadth") != root_breadth
        or receipt.get("paired_root_value_patch_consumed") is not True
        or receipt.get("completed_q_patch_consumed") is not True
        or receipt.get("completed_q_binding") != completed_q_binding
        or admission.get("policy_distillation_contract", {}).get(
            "root_breadth_inventory_sha256"
        )
        != root_breadth["inventory_sha256"]
    ):
        raise OverlayError("Stage-C overlay materialization binding drifted")
    corpus_root = Path(str(admission["corpus"]["data_path"])).resolve(strict=True)
    meta_path = corpus_root / "corpus_meta.json"
    _meta_path, meta = _load_json(meta_path, where="Stage-C overlay corpus metadata")
    if (
        _file_sha256(meta_path) != admission["corpus"]["corpus_meta_file_sha256"]
        or corpus_root != receipt_path.parent
        or receipt["overlay_corpus"]["payload_inventory_sha256"]
        != admission["corpus"]["payload_inventory_sha256"]
        or meta.get("stage_c_policy_overlay", {}).get("root_breadth") != root_breadth
        or meta.get("stage_c_policy_overlay", {}).get(
            "paired_root_value_patch_consumed"
        )
        is not True
        or meta.get("stage_c_policy_overlay", {}).get(
            "completed_q_patch_consumed"
        )
        is not True
        or meta.get("stage_c_policy_overlay", {}).get("completed_q_binding")
        != completed_q_binding
        or any(
            meta.get("columns", {}).get(name) != schema
            for name, schema in COMPLETED_Q_COLUMN_SCHEMAS.items()
        )
    ):
        raise OverlayError("Stage-C overlay admission differs from corpus bytes")
    return {
        "path": str(admission_path),
        "admission": admission,
        "receipt": receipt,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--merge-receipt", required=True, type=Path)
    export.add_argument("--output-root", required=True, type=Path)
    materialize = commands.add_parser("materialize")
    materialize.add_argument("--export-manifest", required=True, type=Path)
    materialize.add_argument("--base-corpus", required=True, type=Path)
    materialize.add_argument("--base-admission", required=True, type=Path)
    materialize.add_argument("--output-root", required=True, type=Path)
    materialize.add_argument(
        "--sampling-arm", required=True, choices=sorted(SAMPLING_ARMS)
    )
    materialize.add_argument(
        "--production-weight-cap",
        type=float,
        default=DEFAULT_PRODUCTION_WEIGHT_CAP,
    )
    verify = commands.add_parser("verify")
    verify.add_argument("--admission", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "export":
            result = _export(args)
        elif args.command == "materialize":
            result = _materialize(args)
        else:
            result = verify_overlay_admission(args.admission)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OverlayError, stage_c.ExecutorError, OSError, ValueError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
