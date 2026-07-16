#!/usr/bin/env python3
"""Qualify sparse historical roots for coherent-n128 Stage-C reanalysis.

Modern self-play intentionally omits one-action UI/chance transitions from the
learner corpus while retaining their absolute ``decision_index`` positions.
Those gaps are not inherently ambiguous: replay can fill a gap exactly when
the live engine proves that precisely one action was legal.  This tool applies
that rule to each selected Stage-C root, round-trips its complete public
feature/history surface, and writes a content-addressed per-row readiness
receipt.  One bad root never blocks unrelated reconstructable roots.

The reconstructed Rust game contains the realised world needed to reproduce
the actor's legitimate private hand and public history.  It is never an
authority for opponent hidden information: the only allowed search hook below
requires ``coherent_public_belief_search=True`` and a public-observation
evaluator, which sanitizes the root before any neural evaluation or expansion.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for root in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools import a1_stage_c_teacher_alignment as alignment  # noqa: E402
from tools import reconstruct_state  # noqa: E402
from tools import train_bc  # noqa: E402
from catan_zero.rl.target_reliability import (  # noqa: E402
    TARGET_RELIABILITY_COLUMNS,
    TARGET_RELIABILITY_SCHEMA,
    unaudited_target_reliability_fields,
)
from catan_zero.rl.gumbel_self_play import (  # noqa: E402
    SEARCH_EVIDENCE_SCHEMA,
    SEARCH_EVIDENCE_VERSION,
)
from catan_zero.search.gumbel_chance_mcts import (  # noqa: E402
    GumbelChanceMCTSConfig,
    _root_candidate_count,
    exact_budget_sh_phases,
    sequential_halving_schedule,
)
from catan_zero.search.native_gumbel_mcts import (  # noqa: E402
    create_gumbel_search,
)
from catan_zero.search.neural_rust_mcts import (  # noqa: E402
    EntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
    rust_policy_action_ids,
)


RECEIPT_SCHEMA = "a1-stage-c-sparse-reconstruction-receipt-v1"
READY_SUBSET_SCHEMA = "a1-stage-c-reconstructable-subset-v1"
QUALIFICATION_PARTITION_RECEIPT_SCHEMA = (
    "a1-stage-c-sparse-reconstruction-partition-receipt-v1"
)
QUALIFICATION_PARTITION_SCHEMA = "a1-stage-c-sparse-reconstruction-partition-v1"
EXECUTION_RECEIPT_SCHEMA = "a1-stage-c-coherent-reanalysis-chunk-receipt-v1"
PATCH_SCHEMA_V1 = "a1-stage-c-coherent-reanalysis-target-patch-v1"
PATCH_SCHEMA = "a1-stage-c-coherent-reanalysis-target-patch-v2"
REBOUND_MERGE_RECEIPT_SCHEMA = "a1-stage-c-coherent-reanalysis-rebound-merge-receipt-v2"
MERGE_RECEIPT_SCHEMA = "a1-stage-c-coherent-reanalysis-merge-receipt-v1"
ROW_SEED_SCHEMA = alignment.STAGE_C_ROW_SEED_SCHEMA
STATUS = {
    "unclassified": 0,
    "reconstructable_public_roundtrip": 1,
    "missing_nonautomatic_decision": 2,
    "recorded_action_illegal": 3,
    "terminal_before_target": 4,
    "public_surface_mismatch": 5,
    "malformed_sequence": 6,
    "runtime_error": 7,
}
QUALIFICATION_PARTITION_COLUMNS = (
    "selected_ordinal",
    "status",
    "omitted_automatic_transitions",
    "omitted_roll_transitions",
    "omitted_end_turn_transitions",
    "omitted_other_ui_transitions",
)
REQUIRED_COHERENT_CAPABILITIES = frozenset(
    {
        "coherent_public_belief_search",
        "forced_root_trajectory_only",
    }
)
PATCH_ROW_COLUMNS = (
    "ready_ordinal",
    "selected_ordinal",
    "row_index",
    "game_seed",
    "decision_index",
    "chunk_index",
    "identity_sha256",
    "search_seed",
    "selected_action_policy_id",
    "root_value",
    "root_value_mask",
    "simulations_used",
    "used_full_search",
    "q_values_root_perspective",
    "target_policy_target_identity_sha256",
    "target_reanalyzer_checkpoint_sha256",
    "target_operator_contract_file_sha256",
    *TARGET_RELIABILITY_COLUMNS,
)
PATCH_RAGGED_COLUMNS = (
    "legal_action_ids_flat",
    "target_policy_flat",
    "target_policy_mask_flat",
    "target_scores_flat",
    "target_scores_mask_flat",
    "completed_q_values_flat",
    "completed_q_mask_flat",
    "prior_policy_flat",
)
RUNTIME_SOURCE_PATHS = frozenset(
    {
        "tools/a1_stage_c_reanalysis_executor.py",
        "tools/reconstruct_state.py",
        "tools/a1_stage_c_teacher_alignment.py",
        "src/catan_zero/rl/gumbel_self_play.py",
        "src/catan_zero/rl/target_reliability.py",
        "src/catan_zero/search/gumbel_chance_mcts.py",
        "src/catan_zero/search/native_gumbel_mcts.py",
        "src/catan_zero/search/neural_rust_mcts.py",
    }
)


class ExecutorError(RuntimeError):
    """Stage-C sparse reconstruction or coherent execution is invalid."""


def _value_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(alignment._canonical_bytes(value)).hexdigest()  # noqa: SLF001


def _runtime_attestation() -> dict[str, Any]:
    try:
        import catanatron_rs  # type: ignore

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (ImportError, OSError, subprocess.CalledProcessError) as error:
        raise ExecutorError(
            f"cannot attest sparse reconstruction runtime: {error}"
        ) from error
    native_module = getattr(catanatron_rs, "catanatron_rs", catanatron_rs)
    extension = Path(native_module.__file__).resolve(strict=True)
    capability_fn = getattr(catanatron_rs, "gumbel_search_capabilities", None)
    capabilities = sorted(capability_fn()) if callable(capability_fn) else []
    missing = REQUIRED_COHERENT_CAPABILITIES - set(capabilities)
    if missing:
        raise ExecutorError(
            "native runtime cannot execute current coherent n128: missing "
            f"{sorted(missing)}"
        )
    try:
        version = importlib.metadata.version("catanatron-rs")
    except importlib.metadata.PackageNotFoundError:
        version = str(getattr(catanatron_rs, "__version__", "unknown"))
    sources = []
    for path in (
        Path(__file__).resolve(),
        REPO_ROOT / "tools/reconstruct_state.py",
        REPO_ROOT / "tools/a1_stage_c_teacher_alignment.py",
        REPO_ROOT / "src/catan_zero/rl/gumbel_self_play.py",
        REPO_ROOT / "src/catan_zero/rl/target_reliability.py",
        REPO_ROOT / "src/catan_zero/search/gumbel_chance_mcts.py",
        REPO_ROOT / "src/catan_zero/search/native_gumbel_mcts.py",
        REPO_ROOT / "src/catan_zero/search/neural_rust_mcts.py",
    ):
        sources.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "file_sha256": alignment._file_sha256(path),  # noqa: SLF001
            }
        )
    value: dict[str, Any] = {
        "schema_version": "a1-stage-c-reconstruction-runtime-v1",
        "repo_commit": commit,
        "sources": sources,
        "native_runtime": {
            "path": str(extension),
            "file_sha256": alignment._file_sha256(extension),  # noqa: SLF001
            "distribution_version": version,
            "capabilities": capabilities,
        },
    }
    value["runtime_sha256"] = _value_sha256(value)
    return value


def _git_blob_sha256(commit: str, path: str) -> str:
    """Hash one historical source blob without consulting the worktree."""

    if Path(path).is_absolute() or ".." in Path(path).parts:
        raise ExecutorError(f"unsafe historical runtime source path: {path!r}")
    try:
        blob = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExecutorError(
            f"cannot resolve sealed historical source {commit}:{path}"
        ) from error
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _verify_runtime_attestation(
    runtime: Mapping[str, Any], *, require_current: bool
) -> None:
    """Verify a current producer or a portable authenticated historical one.

    Historical verification is read-only: source bytes are read from the
    recorded git commit rather than the current checkout, while the recorded
    native extension must still exist byte-for-byte.  This lets a newer export
    tool consume a sealed old DAG without pretending that old code is current.
    New search execution calls this with ``require_current=True``.
    """

    unsigned = dict(runtime)
    stated = unsigned.pop("runtime_sha256", None)
    if runtime.get(
        "schema_version"
    ) != "a1-stage-c-reconstruction-runtime-v1" or stated != _value_sha256(unsigned):
        raise ExecutorError("Stage-C reconstruction runtime digest drifted")
    if require_current:
        if dict(runtime) != _runtime_attestation():
            raise ExecutorError("Stage-C runtime is not the current executable runtime")
        return

    commit = str(runtime.get("repo_commit", ""))
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ExecutorError("Stage-C historical runtime commit is malformed")
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExecutorError(
            "Stage-C historical runtime commit is unavailable or not an ancestor"
        ) from error
    sources = runtime.get("sources")
    if not isinstance(sources, list):
        raise ExecutorError("Stage-C historical runtime sources are malformed")
    by_path = {
        str(source.get("path")): source
        for source in sources
        if isinstance(source, Mapping)
    }
    if set(by_path) != RUNTIME_SOURCE_PATHS or len(by_path) != len(sources):
        raise ExecutorError("Stage-C historical runtime source inventory drifted")
    for path, source in by_path.items():
        if source.get("file_sha256") != _git_blob_sha256(commit, path):
            raise ExecutorError(f"Stage-C historical source bytes drifted: {path}")
    native = runtime.get("native_runtime")
    if not isinstance(native, Mapping):
        raise ExecutorError("Stage-C reconstruction native runtime is malformed")
    native_path = Path(str(native.get("path", ""))).resolve(strict=True)
    if native.get("file_sha256") != alignment._file_sha256(native_path):  # noqa: SLF001
        raise ExecutorError("Stage-C reconstruction native extension drifted")
    capabilities = native.get("capabilities")
    if (
        not isinstance(capabilities, list)
        or not REQUIRED_COHERENT_CAPABILITIES.issubset(map(str, capabilities))
        or not isinstance(native.get("distribution_version"), str)
    ):
        raise ExecutorError("Stage-C historical native runtime identity is incomplete")


def _sequence_rows(
    data: Mapping[str, Any], selected_game_seeds: np.ndarray
) -> dict[int, tuple[reconstruct_state.GameActionSequence, np.ndarray]]:
    all_seeds = np.asarray(data["game_seed"], dtype=np.int64)
    if all_seeds.ndim != 1 or np.any(all_seeds[1:] < all_seeds[:-1]):
        raise ExecutorError("corpus game_seed rows are not monotonically grouped")
    requested = np.unique(np.asarray(selected_game_seeds, dtype=np.int64))
    actions = np.asarray(data["action_taken"], dtype=np.int64)
    decisions = np.asarray(data["decision_index"], dtype=np.int64)
    phases = data["phase"]
    players = data["player"]
    result: dict[int, tuple[reconstruct_state.GameActionSequence, np.ndarray]] = {}
    for game_seed in requested.tolist():
        start = int(np.searchsorted(all_seeds, game_seed, side="left"))
        stop = int(np.searchsorted(all_seeds, game_seed, side="right"))
        if start == stop:
            raise ExecutorError(f"selected game_seed={game_seed} is absent from corpus")
        row_indices = np.arange(start, stop, dtype=np.int64)
        game_decisions = decisions[row_indices]
        if game_decisions[0] != 0 or np.any(game_decisions[1:] <= game_decisions[:-1]):
            raise ExecutorError(
                f"game_seed={game_seed} has malformed recorded decision indices"
            )
        sequence = reconstruct_state.GameActionSequence(
            game_seed=int(game_seed),
            colors=reconstruct_state.DEFAULT_COLORS,
            actions=actions[row_indices].astype(np.int64).tolist(),
            decision_indices=game_decisions.astype(np.int64).tolist(),
            phases=np.asarray(phases[row_indices]).astype(str).tolist(),
            players=np.asarray(players[row_indices]).astype(str).tolist(),
        )
        result[int(game_seed)] = (sequence, row_indices)
    return result


def _stored_roundtrip_features(
    data: Mapping[str, Any], row: int
) -> dict[str, np.ndarray]:
    names = (
        *reconstruct_state._ROUNDTRIP_ENTITY_KEYS,  # noqa: SLF001
        *reconstruct_state._ROUNDTRIP_ACTION_KEYS,  # noqa: SLF001
        "legal_action_context",
    )
    return {
        name: np.asarray(data[name][np.asarray([int(row)], dtype=np.int64)])[0]
        for name in names
        if name in data
    }


def _one_row(column: Any, row: int) -> np.ndarray:
    """Read one fixed/ragged memmap row without scalar-index ambiguity."""

    return np.asarray(column[np.asarray([int(row)], dtype=np.int64)])[0]


def _status_for_error(error: BaseException) -> tuple[int, dict[str, Any]]:
    if isinstance(error, reconstruct_state.SparseReconstructionError):
        code = error.code if error.code in STATUS else "runtime_error"
        return STATUS[code], {
            "classification": code,
            "decision_index": error.decision_index,
            "legal_action_count": error.legal_action_count,
            "detail": str(error),
        }
    return STATUS["runtime_error"], {
        "classification": "runtime_error",
        "detail": f"{type(error).__name__}: {error}",
    }


def _checkpoint_action_size(path: Path) -> int:
    """Read the authenticated producer's action normalization dimension."""

    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise ExecutorError(f"cannot load source checkpoint config: {error}") from error
    config = payload.get("config") if isinstance(payload, Mapping) else None
    if isinstance(config, Mapping) and isinstance(config.get("fields"), Mapping):
        value = config["fields"].get("action_size")
    else:
        value = getattr(config, "action_size", None)
    try:
        action_size = int(value)
    except (TypeError, ValueError) as error:
        raise ExecutorError("source checkpoint has no typed action_size") from error
    if action_size <= 0:
        raise ExecutorError("source checkpoint action_size must be positive")
    return action_size


def _qualification_partition_ordinals(
    game_seeds: np.ndarray, *, partition_index: int, partitions: int
) -> np.ndarray:
    """Assign whole games to deterministic qualification partitions.

    Selected roots from one game must never be replayed by multiple workers.
    Ownership therefore follows the ordinal of the sorted unique game seed,
    not the selected-row ordinal and not Python's process-randomized hash.
    """

    values = np.asarray(game_seeds, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ExecutorError("qualification requires a non-empty game_seed column")
    if partitions <= 0 or not 0 <= partition_index < partitions:
        raise ExecutorError("invalid qualification partition index/count")
    unique_games = np.unique(values)
    if partitions > len(unique_games):
        raise ExecutorError(
            "qualification partition count exceeds selected unique games"
        )
    owned_games = unique_games[
        np.arange(len(unique_games), dtype=np.int64) % partitions == partition_index
    ]
    return np.flatnonzero(np.isin(values, owned_games)).astype(np.int64)


def _qualify_ordinals(
    *,
    data: Mapping[str, Any],
    subset: Mapping[str, np.ndarray],
    selected_ordinals: np.ndarray,
    action_size: int,
    meaningful_public_history: bool,
    history_limit: int,
    correct_chance: bool,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Replay and public-roundtrip one complete set of game-owned roots."""

    ordinals = np.asarray(selected_ordinals, dtype=np.int64)
    count = len(subset["row_index"])
    if (
        ordinals.ndim != 1
        or ordinals.size == 0
        or np.any(ordinals < 0)
        or np.any(ordinals >= count)
        or np.unique(ordinals).size != ordinals.size
        or np.any(ordinals[1:] <= ordinals[:-1])
    ):
        raise ExecutorError("qualification selected ordinals are malformed")
    selected_game_seeds = np.asarray(subset["game_seed"], dtype=np.int64)[ordinals]
    sequences = _sequence_rows(data, selected_game_seeds)
    status = np.full(len(ordinals), STATUS["unclassified"], dtype=np.uint8)
    omitted = np.zeros(len(ordinals), dtype=np.uint16)
    omitted_roll = np.zeros(len(ordinals), dtype=np.uint16)
    omitted_end_turn = np.zeros(len(ordinals), dtype=np.uint16)
    omitted_other_ui = np.zeros(len(ordinals), dtype=np.uint16)
    failures: list[dict[str, Any]] = []
    local_by_global = {
        int(selected_ordinal): local
        for local, selected_ordinal in enumerate(ordinals.tolist())
    }
    ordinals_by_game: dict[int, list[int]] = {}
    for selected_ordinal in ordinals.tolist():
        game_seed = int(subset["game_seed"][selected_ordinal])
        ordinals_by_game.setdefault(game_seed, []).append(int(selected_ordinal))

    for game_seed, game_ordinals in sorted(ordinals_by_game.items()):
        sequence, game_rows = sequences[game_seed]
        target_decisions = [
            int(subset["decision_index"][ordinal]) for ordinal in game_ordinals
        ]
        batch = reconstruct_state.reconstruct_states_from_sequence(
            sequence,
            target_decisions,
            correct_rust_chance_spectra=correct_chance,
            action_size=action_size,
        )
        for selected_ordinal in game_ordinals:
            local_ordinal = local_by_global[selected_ordinal]
            row = int(subset["row_index"][selected_ordinal])
            decision = int(subset["decision_index"][selected_ordinal])
            local = int(np.searchsorted(game_rows, row))
            if local >= len(game_rows) or int(game_rows[local]) != row:
                raise ExecutorError(
                    f"selected row={row} is not in its claimed game_seed={game_seed}"
                )
            if int(sequence.decision_indices[local]) != decision:
                raise ExecutorError("selected row decision identity drifted")
            reconstructed_game = batch.states.get(decision)
            if reconstructed_game is None:
                error = batch.failure or ExecutorError(
                    "sparse reconstruction stopped without a classified failure"
                )
                status[local_ordinal], detail = _status_for_error(error)
                failures.append(
                    {
                        "ordinal": selected_ordinal,
                        "row_index": row,
                        "game_seed": game_seed,
                        "decision_index": decision,
                        **detail,
                    }
                )
                continue
            omitted_count = batch.omitted_automatic_transitions[decision]
            if omitted_count > np.iinfo(np.uint16).max:
                raise ExecutorError("omitted automatic-transition count overflow")
            omitted[local_ordinal] = np.uint16(omitted_count)
            omitted_types = batch.omitted_automatic_transition_types[decision]
            roll_count = int(omitted_types.get("ROLL", 0))
            end_turn_count = int(omitted_types.get("END_TURN", 0))
            other_count = omitted_count - roll_count - end_turn_count
            if min(roll_count, end_turn_count, other_count) < 0:
                raise ExecutorError("omitted automatic-transition type counts drifted")
            omitted_roll[local_ordinal] = np.uint16(roll_count)
            omitted_end_turn[local_ordinal] = np.uint16(end_turn_count)
            omitted_other_ui[local_ordinal] = np.uint16(other_count)
            try:
                result = reconstruct_state.round_trip_row(
                    sequence,
                    decision,
                    _stored_roundtrip_features(data, row),
                    _one_row(data["legal_action_ids"], row),
                    correct_rust_chance_spectra=correct_chance,
                    action_size=action_size,
                    meaningful_public_history=meaningful_public_history,
                    history_limit=history_limit,
                    reconstructed_game=reconstructed_game,
                )
            except Exception as error:  # noqa: BLE001 - classify row, continue.
                status[local_ordinal], detail = _status_for_error(error)
                failures.append(
                    {
                        "ordinal": selected_ordinal,
                        "row_index": row,
                        "game_seed": game_seed,
                        "decision_index": decision,
                        **detail,
                    }
                )
                continue
            if not result.ok:
                status[local_ordinal] = STATUS["public_surface_mismatch"]
                failures.append(
                    {
                        "ordinal": selected_ordinal,
                        "row_index": row,
                        "game_seed": game_seed,
                        "decision_index": decision,
                        "classification": "public_surface_mismatch",
                        "legal_ids_match": result.legal_ids_match,
                        "phase_match": result.phase_match,
                        "player_match": result.player_match,
                        "worst_key": result.worst_key,
                        "max_abs_diff": (
                            result.max_abs_diff
                            if math.isfinite(result.max_abs_diff)
                            else None
                        ),
                        "detail": result.detail,
                    }
                )
                continue
            status[local_ordinal] = STATUS["reconstructable_public_roundtrip"]

    return (
        {
            "selected_ordinal": ordinals,
            "status": status,
            "omitted_automatic_transitions": omitted,
            "omitted_roll_transitions": omitted_roll,
            "omitted_end_turn_transitions": omitted_end_turn,
            "omitted_other_ui_transitions": omitted_other_ui,
        },
        failures,
    )


def _write_qualification_artifacts(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    subset: Mapping[str, np.ndarray],
    status: np.ndarray,
    omitted: np.ndarray,
    omitted_roll: np.ndarray,
    omitted_end_turn: np.ndarray,
    omitted_other_ui: np.ndarray,
    failures: Sequence[Mapping[str, Any]],
    action_size: int,
    runtime: Mapping[str, Any],
    output_root: Path,
    partition_receipts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Write the canonical qualification surface from ordered global arrays."""

    count = len(subset["row_index"])
    arrays = (status, omitted, omitted_roll, omitted_end_turn, omitted_other_ui)
    if any(np.asarray(value).shape != (count,) for value in arrays):
        raise ExecutorError("merged qualification arrays are not globally aligned")
    if np.any(status == STATUS["unclassified"]):
        raise ExecutorError("merged qualification left unclassified roots")
    output_root = output_root.expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "selected_reconstruction_status.u8.dat"
    omitted_path = output_root / "selected_omitted_automatic_count.u16.dat"
    roll_path = output_root / "selected_omitted_roll_count.u16.dat"
    end_turn_path = output_root / "selected_omitted_end_turn_count.u16.dat"
    other_ui_path = output_root / "selected_omitted_other_ui_count.u16.dat"
    ready_path = output_root / "reconstructable_reanalysis_rows.npz"
    alignment._write_array_immutable(status_path, status)  # noqa: SLF001
    alignment._write_array_immutable(omitted_path, omitted)  # noqa: SLF001
    alignment._write_array_immutable(roll_path, omitted_roll)  # noqa: SLF001
    alignment._write_array_immutable(end_turn_path, omitted_end_turn)  # noqa: SLF001
    alignment._write_array_immutable(other_ui_path, omitted_other_ui)  # noqa: SLF001
    ready = status == STATUS["reconstructable_public_roundtrip"]
    ready_arrays = {name: values[ready] for name, values in subset.items()}
    ready_arrays["ready_ordinal"] = np.arange(
        int(np.count_nonzero(ready)), dtype=np.int64
    )
    ready_arrays["selected_ordinal"] = np.flatnonzero(ready).astype(np.int64)
    ready_arrays["omitted_automatic_transitions_before_root"] = omitted[ready]
    ready_arrays["omitted_roll_transitions_before_root"] = omitted_roll[ready]
    ready_arrays["omitted_end_turn_transitions_before_root"] = omitted_end_turn[ready]
    ready_arrays["omitted_other_ui_transitions_before_root"] = omitted_other_ui[ready]
    alignment._write_immutable(  # noqa: SLF001
        ready_path,
        alignment._npz_bytes(ready_arrays),  # noqa: SLF001
    )
    counts = {
        name: int(np.count_nonzero(status == code)) for name, code in STATUS.items()
    }
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "stage_c_plan": {
            "path": str(plan_path),
            "file_sha256": alignment._file_sha256(plan_path),  # noqa: SLF001
            "plan_sha256": plan["plan_sha256"],
        },
        "source_subset": plan["subset"]["artifact"],
        "target_policy_target_identity_sha256": plan["target_policy_target_identity"][
            "identity_sha256"
        ],
        "runtime": dict(runtime),
        "source_checkpoint_action_size": action_size,
        "status_codes": STATUS,
        "counts": {"selected_rows": count, **counts},
        "artifacts": {
            "status": alignment._artifact_ref(status_path),  # noqa: SLF001
            "omitted_automatic_transition_count": alignment._artifact_ref(  # noqa: SLF001
                omitted_path
            ),
            "omitted_roll_count": alignment._artifact_ref(roll_path),  # noqa: SLF001
            "omitted_end_turn_count": alignment._artifact_ref(  # noqa: SLF001
                end_turn_path
            ),
            "omitted_other_ui_count": alignment._artifact_ref(  # noqa: SLF001
                other_ui_path
            ),
            "reconstructable_subset": {
                "schema_version": READY_SUBSET_SCHEMA,
                **alignment._artifact_ref(ready_path),  # noqa: SLF001
            },
        },
        "execution": {
            "execution_ready": bool(np.any(ready)),
            "fully_reconstructable": bool(np.all(ready)),
            "scope": "only_rows_with_reconstructable_public_roundtrip_status",
            "omitted_gap_rule": "exactly_one_live_legal_action",
            "omitted_transition_types": ["ROLL", "END_TURN", "OTHER_UI"],
            "legacy_pimc_allowed": False,
            "authoritative_hidden_state_search_allowed": False,
            "required_search": "coherent_public_belief_search",
        },
        "information_set_safety": {
            "roundtrip_surface": "public_observation_plus_actor_private_information",
            "opponent_hidden_truth_used_by_search": False,
            "coherent_root_sanitization_required_before_evaluation": True,
            "event_history_compared": True,
        },
        "failure_examples": [dict(item) for item in failures[:100]],
    }
    if partition_receipts:
        receipt["qualification_partitions"] = list(partition_receipts)
    receipt["receipt_sha256"] = _value_sha256(receipt)
    return receipt


def _qualify(args: argparse.Namespace) -> dict[str, Any]:
    try:
        plan = alignment._verify_plan(args.plan)  # noqa: SLF001
    except alignment.AlignmentError as error:
        raise ExecutorError(f"Stage-C plan refused: {error}") from error
    plan_path = Path(plan["path"])
    subset_path = Path(str(plan["subset"]["artifact"]["path"]))
    overlay_path = Path(str(plan["eligibility_overlay"]["path"]))
    _overlay_path, overlay = alignment._load_json(  # noqa: SLF001
        overlay_path, where="Stage-C eligibility overlay"
    )
    corpus_root = Path(str(overlay["corpus"]["path"])).resolve(strict=True)
    if (
        alignment._file_sha256(corpus_root / "corpus_meta.json")
        != overlay[  # noqa: SLF001
            "corpus"
        ]["corpus_meta_file_sha256"]
    ):
        raise ExecutorError("Stage-C corpus metadata drifted")
    data = train_bc.MemmapCorpus(corpus_root)
    with np.load(subset_path, allow_pickle=False) as subset_file:
        subset = {name: np.asarray(subset_file[name]) for name in subset_file.files}
    count = len(subset["row_index"])
    if count != int(plan["subset"]["selected_rows"]):
        raise ExecutorError("Stage-C selected subset row count drifted")
    source_checkpoint = Path(
        str(plan["source_policy_target_identity"]["producer_checkpoint"]["path"])
    )
    action_size = _checkpoint_action_size(source_checkpoint)
    history = plan["target_policy_target_identity"]["target_semantics"]
    correct_chance = bool(
        plan["target_policy_target_identity"]["chance"]["correct_rust_chance_spectra"]
    )
    qualification, failures = _qualify_ordinals(
        data=data,
        subset=subset,
        selected_ordinals=np.arange(count, dtype=np.int64),
        action_size=action_size,
        meaningful_public_history=bool(history["meaningful_public_history"]),
        history_limit=int(history["event_history_limit"]),
        correct_chance=correct_chance,
    )
    status = qualification["status"]
    omitted = qualification["omitted_automatic_transitions"]
    omitted_roll = qualification["omitted_roll_transitions"]
    omitted_end_turn = qualification["omitted_end_turn_transitions"]
    omitted_other_ui = qualification["omitted_other_ui_transitions"]

    return _write_qualification_artifacts(
        plan=plan,
        plan_path=plan_path,
        subset=subset,
        status=status,
        omitted=omitted,
        omitted_roll=omitted_roll,
        omitted_end_turn=omitted_end_turn,
        omitted_other_ui=omitted_other_ui,
        failures=failures,
        action_size=action_size,
        runtime=_runtime_attestation(),
        output_root=args.output_root,
    )


def _load_qualification_inputs(
    plan_path: Path,
) -> tuple[dict[str, Any], Path, dict[str, np.ndarray], Any, int]:
    try:
        plan = alignment._verify_plan(plan_path)  # noqa: SLF001
    except alignment.AlignmentError as error:
        raise ExecutorError(f"Stage-C plan refused: {error}") from error
    resolved_plan = Path(str(plan["path"]))
    subset_path = Path(str(plan["subset"]["artifact"]["path"]))
    overlay_path = Path(str(plan["eligibility_overlay"]["path"]))
    _overlay_path, overlay = alignment._load_json(  # noqa: SLF001
        overlay_path, where="Stage-C eligibility overlay"
    )
    corpus_root = Path(str(overlay["corpus"]["path"])).resolve(strict=True)
    if (
        alignment._file_sha256(corpus_root / "corpus_meta.json")
        != overlay["corpus"]["corpus_meta_file_sha256"]  # noqa: SLF001
    ):
        raise ExecutorError("Stage-C corpus metadata drifted")
    data = train_bc.MemmapCorpus(corpus_root)
    with np.load(subset_path, allow_pickle=False) as subset_file:
        subset = {name: np.asarray(subset_file[name]) for name in subset_file.files}
    if len(subset["row_index"]) != int(plan["subset"]["selected_rows"]):
        raise ExecutorError("Stage-C selected subset row count drifted")
    source_checkpoint = Path(
        str(plan["source_policy_target_identity"]["producer_checkpoint"]["path"])
    )
    return plan, resolved_plan, subset, data, _checkpoint_action_size(source_checkpoint)


def _qualify_partition(args: argparse.Namespace) -> dict[str, Any]:
    """Qualify all roots from one deterministic set of whole games."""

    plan, plan_path, subset, data, action_size = _load_qualification_inputs(args.plan)
    selected_ordinals = _qualification_partition_ordinals(
        subset["game_seed"],
        partition_index=int(args.partition_index),
        partitions=int(args.partitions),
    )
    history = plan["target_policy_target_identity"]["target_semantics"]
    arrays, failures = _qualify_ordinals(
        data=data,
        subset=subset,
        selected_ordinals=selected_ordinals,
        action_size=action_size,
        meaningful_public_history=bool(history["meaningful_public_history"]),
        history_limit=int(history["event_history_limit"]),
        correct_chance=bool(
            plan["target_policy_target_identity"]["chance"][
                "correct_rust_chance_spectra"
            ]
        ),
    )
    artifact_path = args.artifact.expanduser().resolve(strict=False)
    alignment._write_immutable(  # noqa: SLF001
        artifact_path,
        alignment._npz_bytes(arrays),  # noqa: SLF001
    )
    status = arrays["status"]
    counts = {
        name: int(np.count_nonzero(status == code)) for name, code in STATUS.items()
    }
    receipt: dict[str, Any] = {
        "schema_version": QUALIFICATION_PARTITION_RECEIPT_SCHEMA,
        "artifact_schema_version": QUALIFICATION_PARTITION_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "stage_c_plan": {
            "path": str(plan_path),
            "file_sha256": alignment._file_sha256(plan_path),  # noqa: SLF001
            "plan_sha256": plan["plan_sha256"],
        },
        "source_subset": plan["subset"]["artifact"],
        "target_policy_target_identity_sha256": plan["target_policy_target_identity"][
            "identity_sha256"
        ],
        "runtime": _runtime_attestation(),
        "source_checkpoint_action_size": action_size,
        "status_codes": STATUS,
        "partition": {
            "partition_index": int(args.partition_index),
            "partitions": int(args.partitions),
            "assignment": "sorted_unique_game_seed_ordinal_mod_partitions",
            "selected_games": int(
                np.unique(subset["game_seed"][selected_ordinals]).size
            ),
        },
        "counts": {"selected_rows": len(selected_ordinals), **counts},
        "columns": sorted(arrays),
        "artifact": alignment._artifact_ref(artifact_path),  # noqa: SLF001
        "failure_examples": failures[:100],
    }
    receipt["receipt_sha256"] = _value_sha256(receipt)
    return receipt


def _verify_qualification_partition(path: Path) -> dict[str, Any]:
    receipt_path, receipt = alignment._load_json(  # noqa: SLF001
        path, where="Stage-C qualification partition receipt"
    )
    unsigned = dict(receipt)
    stated = unsigned.pop("receipt_sha256", None)
    if (
        receipt.get("schema_version") != QUALIFICATION_PARTITION_RECEIPT_SCHEMA
        or receipt.get("artifact_schema_version") != QUALIFICATION_PARTITION_SCHEMA
        or stated != _value_sha256(unsigned)
    ):
        raise ExecutorError("Stage-C qualification partition digest drifted")
    plan_ref = receipt.get("stage_c_plan")
    if not isinstance(plan_ref, Mapping):
        raise ExecutorError("qualification partition lost its plan")
    plan = alignment._verify_plan(Path(str(plan_ref["path"])))  # noqa: SLF001
    plan_path = Path(str(plan["path"]))
    if (
        plan_ref.get("file_sha256") != alignment._file_sha256(plan_path)  # noqa: SLF001
        or plan_ref.get("plan_sha256") != plan["plan_sha256"]
        or receipt.get("target_policy_target_identity_sha256")
        != plan["target_policy_target_identity"]["identity_sha256"]
    ):
        raise ExecutorError("qualification partition plan binding drifted")
    runtime = receipt.get("runtime")
    if not isinstance(runtime, Mapping) or runtime != _runtime_attestation():
        raise ExecutorError("qualification partition runtime drifted")
    artifact = receipt.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ExecutorError("qualification partition lost its artifact")
    artifact_path = Path(str(artifact["path"])).resolve(strict=True)
    if (
        artifact.get("file_sha256") != alignment._file_sha256(artifact_path)  # noqa: SLF001
        or artifact.get("size_bytes") != artifact_path.stat().st_size
    ):
        raise ExecutorError("qualification partition artifact bytes drifted")
    with np.load(artifact_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    if set(arrays) != set(QUALIFICATION_PARTITION_COLUMNS) or sorted(
        arrays
    ) != receipt.get("columns"):
        raise ExecutorError("qualification partition column contract drifted")
    with np.load(
        Path(str(plan["subset"]["artifact"]["path"])), allow_pickle=False
    ) as source:
        subset = {name: np.asarray(source[name]) for name in source.files}
    partition = receipt.get("partition", {})
    index = int(partition.get("partition_index", -1))
    partitions = int(partition.get("partitions", 0))
    if partition.get("assignment") != "sorted_unique_game_seed_ordinal_mod_partitions":
        raise ExecutorError("qualification partition assignment contract drifted")
    expected = _qualification_partition_ordinals(
        subset["game_seed"], partition_index=index, partitions=partitions
    )
    actual = np.asarray(arrays["selected_ordinal"], dtype=np.int64)
    if not np.array_equal(actual, expected):
        raise ExecutorError("qualification partition root ownership drifted")
    count = len(expected)
    if any(np.asarray(value).shape != (count,) for value in arrays.values()):
        raise ExecutorError("qualification partition columns are misaligned")
    status = np.asarray(arrays["status"], dtype=np.uint8)
    if np.any(status == STATUS["unclassified"]) or np.any(
        ~np.isin(status, np.fromiter(STATUS.values(), dtype=np.uint8))
    ):
        raise ExecutorError("qualification partition has invalid status values")
    counts = {
        name: int(np.count_nonzero(status == code)) for name, code in STATUS.items()
    }
    if receipt.get("counts") != {"selected_rows": count, **counts}:
        raise ExecutorError("qualification partition counts drifted")
    expected_games = int(np.unique(subset["game_seed"][expected]).size)
    if int(partition.get("selected_games", -1)) != expected_games:
        raise ExecutorError("qualification partition game count drifted")
    return {
        "path": str(receipt_path),
        "file_sha256": alignment._file_sha256(receipt_path),  # noqa: SLF001
        "plan": plan,
        "subset": subset,
        "arrays": arrays,
        **receipt,
    }


def _merge_qualification_partitions(args: argparse.Namespace) -> dict[str, Any]:
    partitions = [_verify_qualification_partition(path) for path in args.receipt]
    if not partitions:
        raise ExecutorError("qualification merge requires partition receipts")
    first = partitions[0]
    expected_partitions = int(first["partition"]["partitions"])
    indices = [int(item["partition"]["partition_index"]) for item in partitions]
    if sorted(indices) != list(range(expected_partitions)):
        raise ExecutorError(
            "qualification merge requires exactly one receipt for every partition"
        )
    if any(
        int(item["partition"]["partitions"]) != expected_partitions
        or item["stage_c_plan"]["plan_sha256"] != first["stage_c_plan"]["plan_sha256"]
        or item["target_policy_target_identity_sha256"]
        != first["target_policy_target_identity_sha256"]
        or item["runtime"] != first["runtime"]
        or int(item["source_checkpoint_action_size"])
        != int(first["source_checkpoint_action_size"])
        for item in partitions[1:]
    ):
        raise ExecutorError("qualification merge received foreign partitions")
    subset = first["subset"]
    count = len(subset["row_index"])
    status = np.full(count, STATUS["unclassified"], dtype=np.uint8)
    omitted = np.zeros(count, dtype=np.uint16)
    omitted_roll = np.zeros(count, dtype=np.uint16)
    omitted_end_turn = np.zeros(count, dtype=np.uint16)
    omitted_other_ui = np.zeros(count, dtype=np.uint16)
    claimed = np.zeros(count, dtype=np.uint8)
    failures: list[dict[str, Any]] = []
    for item in partitions:
        arrays = item["arrays"]
        ordinals = np.asarray(arrays["selected_ordinal"], dtype=np.int64)
        if np.any(claimed[ordinals]):
            raise ExecutorError("qualification merge has duplicate selected roots")
        claimed[ordinals] = 1
        status[ordinals] = arrays["status"]
        omitted[ordinals] = arrays["omitted_automatic_transitions"]
        omitted_roll[ordinals] = arrays["omitted_roll_transitions"]
        omitted_end_turn[ordinals] = arrays["omitted_end_turn_transitions"]
        omitted_other_ui[ordinals] = arrays["omitted_other_ui_transitions"]
        failures.extend(item.get("failure_examples", ()))
    if not np.all(claimed):
        raise ExecutorError("qualification merge has incomplete selected-root coverage")
    references = [
        {
            "path": item["path"],
            "file_sha256": item["file_sha256"],
            "receipt_sha256": item["receipt_sha256"],
            "partition_index": item["partition"]["partition_index"],
        }
        for item in sorted(
            partitions, key=lambda value: int(value["partition"]["partition_index"])
        )
    ]
    failures.sort(key=lambda item: int(item.get("ordinal", -1)))
    return _write_qualification_artifacts(
        plan=first["plan"],
        plan_path=Path(str(first["plan"]["path"])),
        subset=subset,
        status=status,
        omitted=omitted,
        omitted_roll=omitted_roll,
        omitted_end_turn=omitted_end_turn,
        omitted_other_ui=omitted_other_ui,
        failures=failures,
        action_size=int(first["source_checkpoint_action_size"]),
        runtime=first["runtime"],
        output_root=args.output_root,
        partition_receipts=references,
    )


def _verify_receipt(
    path: Path, *, require_current_runtime: bool = False
) -> dict[str, Any]:
    receipt_path, receipt = alignment._load_json(  # noqa: SLF001
        path, where="Stage-C reconstruction receipt"
    )
    unsigned = dict(receipt)
    stated = unsigned.pop("receipt_sha256", None)
    if receipt.get("schema_version") != RECEIPT_SCHEMA or stated != _value_sha256(
        unsigned
    ):
        raise ExecutorError("Stage-C reconstruction receipt digest drifted")
    plan_ref = receipt.get("stage_c_plan")
    if not isinstance(plan_ref, Mapping):
        raise ExecutorError("Stage-C reconstruction receipt lost its plan")
    plan = alignment._verify_plan(Path(str(plan_ref["path"])))  # noqa: SLF001
    if (
        plan_ref.get("file_sha256")
        != alignment._file_sha256(Path(str(plan_ref["path"])))  # noqa: SLF001
        or plan_ref.get("plan_sha256") != plan["plan_sha256"]
    ):
        raise ExecutorError("Stage-C reconstruction plan bytes drifted")
    runtime = receipt.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ExecutorError("Stage-C reconstruction receipt lost its runtime")
    _verify_runtime_attestation(runtime, require_current=require_current_runtime)
    for artifact in receipt["artifacts"].values():
        artifact_path = Path(str(artifact["path"])).resolve(strict=True)
        if (
            artifact.get("file_sha256") != alignment._file_sha256(artifact_path)  # noqa: SLF001
            or artifact.get("size_bytes") != artifact_path.stat().st_size
        ):
            raise ExecutorError("Stage-C reconstruction artifact bytes drifted")
    return {"path": str(receipt_path), **receipt}


def assert_information_set_safe_search(plan: Mapping[str, Any], search: Any) -> None:
    """Gate the only supported consumer of reconstructed authoritative roots."""

    config = getattr(search, "config", None)
    evaluator_config = getattr(getattr(search, "evaluator", None), "config", None)
    target = plan["target_policy_target_identity"]
    expected = {**target["search"], **target["belief"], **target["chance"]}
    if (
        config is None
        or evaluator_config is None
        or not bool(getattr(config, "coherent_public_belief_search", False))
        or bool(getattr(config, "information_set_search", False))
        or not bool(getattr(evaluator_config, "public_observation", False))
    ):
        raise ExecutorError(
            "reconstructed roots require coherent public-belief search and a "
            "public-observation evaluator"
        )
    drift = {
        name: {"required": value, "actual": getattr(config, name, None)}
        for name, value in expected.items()
        if hasattr(config, name) and getattr(config, name) != value
    }
    if drift:
        raise ExecutorError(
            "coherent search differs from sealed Stage-C target: "
            + json.dumps(drift, sort_keys=True)
        )


def run_information_set_safe_search(
    plan: Mapping[str, Any], search: Any, reconstructed_game: Any
) -> Any:
    """The sole Stage-C search hook: attest first, then force full n128."""

    assert_information_set_safe_search(plan, search)
    return search.search(reconstructed_game, force_full=True)


def _load_ready_subset(receipt: Mapping[str, Any]) -> dict[str, np.ndarray]:
    artifact = receipt["artifacts"]["reconstructable_subset"]
    path = Path(str(artifact["path"])).resolve(strict=True)
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    if not arrays or "identity_sha256" not in arrays:
        raise ExecutorError("reconstructable subset has no row identities")
    count = len(arrays["identity_sha256"])
    if any(len(value) != count for value in arrays.values()):
        raise ExecutorError("reconstructable subset columns are not row-aligned")
    if np.unique(arrays["identity_sha256"]).size != count:
        raise ExecutorError("reconstructable subset row identities are not unique")
    # v1 qualification receipts written before the execute path did not carry
    # explicit ordinals.  Their immutable row order is nevertheless sealed by
    # the subset file hash, so deriving these columns is unambiguous.
    arrays.setdefault("ready_ordinal", np.arange(count, dtype=np.int64))
    arrays.setdefault("selected_ordinal", np.arange(count, dtype=np.int64))
    return arrays


def _sealed_typed_fields(plan: Mapping[str, Any]) -> dict[str, Any]:
    target = plan["target_policy_target_identity"]
    typed = target["authority"]["typed_generation_config"]
    path, payload = alignment._load_json(  # noqa: SLF001
        Path(str(typed["path"])), where="Stage-C typed target config"
    )
    if typed.get("file_sha256") != alignment._file_sha256(path):  # noqa: SLF001
        raise ExecutorError("Stage-C typed target config bytes drifted")
    fields = payload.get("fields")
    if payload.get("schema_version") != target["target_semantics"][
        "typed_generation_config_schema"
    ] or not isinstance(fields, Mapping):
        raise ExecutorError("Stage-C typed target config is malformed")
    return dict(fields)


def _target_checkpoint(plan: Mapping[str, Any]) -> Path:
    checkpoint = plan["target_policy_target_identity"]["producer_checkpoint"]
    path = Path(str(checkpoint["path"])).resolve(strict=True)
    if checkpoint.get("sha256") != alignment._file_sha256(path):  # noqa: SLF001
        raise ExecutorError("Stage-C target checkpoint bytes drifted")
    return path


def _effective_search_config(
    plan: Mapping[str, Any], *, row_seed: int
) -> GumbelChanceMCTSConfig:
    target = plan["target_policy_target_identity"]
    fields = _sealed_typed_fields(plan)
    allowed = {field.name for field in dataclasses.fields(GumbelChanceMCTSConfig)}
    kwargs = {name: value for name, value in fields.items() if name in allowed}
    kwargs["seed"] = int(row_seed)
    if "colors" in kwargs:
        kwargs["colors"] = tuple(str(value) for value in kwargs["colors"])
    config = GumbelChanceMCTSConfig(**kwargs)
    effective = alignment._complete_effective_search_config(fields)  # noqa: SLF001
    if (
        target.get("schema_version") == alignment.OPERATOR_IDENTITY_SCHEMA_V2
        and target.get("effective_gumbel_config") != effective
    ):
        raise ExecutorError("Stage-C effective Gumbel identity drifted")
    if (
        int(config.n_full) != 128
        or not bool(config.coherent_public_belief_search)
        or bool(config.information_set_search)
        or bool(config.belief_chance_spectra)
        or target.get("target_information_regime") != alignment.COHERENT_REGIME
    ):
        raise ExecutorError(
            "Stage-C execute requires exact coherent-public n128, never PIMC"
        )
    operator = target.get("operator_contract_semantics", {})
    if (
        operator.get("native_mcts_hot_loop") is not True
        or operator.get("coherent_public_belief_search") is not True
        or operator.get("information_set_search") is not False
    ):
        raise ExecutorError(
            "sealed target operator does not require native coherent search"
        )
    return config


def _expected_forced_full_simulations(
    legal_width: int, effective_config: Mapping[str, Any]
) -> int:
    """Replay the exact root-budget accounting sealed by Stage C."""

    kwargs = dict(effective_config)
    kwargs["seed"] = 0
    if "colors" in kwargs:
        kwargs["colors"] = tuple(str(value) for value in kwargs["colors"])
    try:
        config = GumbelChanceMCTSConfig(**kwargs)
    except (TypeError, ValueError) as error:
        raise ExecutorError(f"invalid effective search config: {error}") from error
    nominal = int(config.n_full)
    candidates = _root_candidate_count(int(legal_width), config)
    exact = bool(config.exact_budget_sh) and (
        int(config.exact_budget_sh_min_n) <= 0
        or nominal >= int(config.exact_budget_sh_min_n)
    )
    schedule = (
        exact_budget_sh_phases(candidates, nominal)
        if exact
        else sequential_halving_schedule(candidates, nominal)
    )
    return sum(int(count) * int(per_candidate) for count, per_candidate in schedule)


def _evaluator_from_plan(plan: Mapping[str, Any], *, device: str) -> Any:
    fields = _sealed_typed_fields(plan)
    target = plan["target_policy_target_identity"]
    if (
        target.get("schema_version") == alignment.OPERATOR_IDENTITY_SCHEMA_V2
        and target.get("effective_evaluator_config")
        != alignment._complete_effective_evaluator_config(fields)  # noqa: SLF001
    ):
        raise ExecutorError("Stage-C effective evaluator identity drifted")
    if fields.get("public_observation") is not True:
        raise ExecutorError("sealed target evaluator is not public-observation safe")
    if fields.get("rust_featurize") is not True:
        raise ExecutorError("sealed target evaluator did not bind native featurization")
    return EntityGraphRustEvaluator.from_checkpoint(
        str(_target_checkpoint(plan)),
        device=str(device),
        config=EntityGraphRustEvaluatorConfig(
            value_scale=float(fields.get("value_scale", 1.0)),
            prior_temperature=float(fields.get("prior_temperature", 1.0)),
            cache_size=int(fields.get("eval_cache_size", 0)),
            value_readout=str(fields.get("value_readout", "scalar")),
            public_observation=True,
            rust_featurize=True,
        ),
    )


def _row_seed(identity_sha256: str) -> int:
    identity = str(identity_sha256)
    if not identity.startswith("sha256:") or len(identity) != 71:
        raise ExecutorError("malformed Stage-C row identity digest")
    return int.from_bytes(
        hashlib.sha256(
            ROW_SEED_SCHEMA.encode("ascii") + b"\0" + identity.encode("ascii")
        ).digest()[:8],
        "big",
    )


def _search_patch(
    *,
    plan: Mapping[str, Any],
    evaluator: Any,
    reconstructed_game: Any,
    row_seed: int,
    expected_legal_policy_ids: np.ndarray,
) -> dict[str, Any]:
    config = _effective_search_config(plan, row_seed=row_seed)
    search = create_gumbel_search(
        config,
        evaluator,
        native_hot_loop=True,
        allow_python_fallback=False,
    )
    result = run_information_set_safe_search(plan, search, reconstructed_game)
    legal_rust = tuple(
        int(action)
        for action in reconstructed_game.playable_action_indices(
            list(config.colors), config.map_kind
        )
    )
    action_size = _checkpoint_action_size(_target_checkpoint(plan))
    legal_policy_ids = np.asarray(
        rust_policy_action_ids(
            reconstructed_game,
            legal_rust,
            colors=tuple(config.colors),
            action_size=action_size,
        ),
        dtype=np.int32,
    )
    stored = np.asarray(expected_legal_policy_ids, dtype=np.int64).reshape(-1)
    stored = stored[stored >= 0]
    if not np.array_equal(legal_policy_ids.astype(np.int64), stored):
        raise ExecutorError(
            "legal action order changed between qualification and search"
        )
    support = set(legal_rust)
    if (
        set(map(int, result.improved_policy)) != support
        or set(map(int, result.priors)) != support
    ):
        raise ExecutorError("coherent search did not cover the exact legal root")
    target = np.asarray(
        [float(result.improved_policy[action]) for action in legal_rust],
        dtype=np.float32,
    )
    priors = np.asarray(
        [float(result.priors[action]) for action in legal_rust], dtype=np.float32
    )
    raw_q = np.asarray(
        [float(result.q_values.get(action, np.nan)) for action in legal_rust],
        dtype=np.float32,
    )
    completed = np.asarray(
        [float(result.completed_q_values.get(action, np.nan)) for action in legal_rust],
        dtype=np.float32,
    )
    execution = plan["target_policy_target_identity"].get("target_execution")
    if execution != alignment.STAGE_C_TARGET_EXECUTION:
        raise ExecutorError("Stage-C forced-full target execution identity drifted")
    effective_config = plan["target_policy_target_identity"]["effective_gumbel_config"]
    if (
        not bool(result.used_full_search)
        or int(result.simulations_used)
        != _expected_forced_full_simulations(len(legal_rust), effective_config)
        or not math.isfinite(float(result.root_value))
        or not bool(result.q_values_root_perspective)
        or not np.all(np.isfinite(completed))
        or not np.isclose(float(target.sum()), 1.0, atol=1.0e-5)
        or not np.isclose(float(priors.sum()), 1.0, atol=1.0e-5)
    ):
        raise ExecutorError(
            "coherent n128 search returned incomplete or ambiguous target evidence"
        )
    try:
        selected_position = legal_rust.index(int(result.selected_action))
    except ValueError as error:
        raise ExecutorError("coherent search selected a non-legal action") from error
    return {
        "legal_action_ids": legal_policy_ids,
        "target_policy": target,
        # Coverage means the teacher supplied a value for this legal action;
        # an exact zero is still a valid soft-target label, never missing data.
        "target_policy_mask": np.ones(target.shape, dtype=np.bool_),
        "target_scores": raw_q,
        "target_scores_mask": np.isfinite(raw_q),
        "completed_q_values": completed,
        "completed_q_mask": np.ones(completed.shape, dtype=np.bool_),
        "prior_policy": priors,
        "selected_action_policy_id": int(legal_policy_ids[selected_position]),
        "root_value": float(result.root_value),
        "root_value_mask": True,
        "simulations_used": int(result.simulations_used),
        "used_full_search": bool(result.used_full_search),
        "q_values_root_perspective": bool(result.q_values_root_perspective),
    }


def _partition_positions(
    subset: Mapping[str, np.ndarray], *, partition_index: int, partitions: int
) -> np.ndarray:
    if partitions < 1 or not 0 <= partition_index < partitions:
        raise ExecutorError("partition_index must be in [0, partitions)")
    chunks = np.asarray(subset["chunk_index"], dtype=np.int64)
    return np.flatnonzero((chunks % int(partitions)) == int(partition_index))


def _patch_arrays(records: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    offsets = [0]
    for record in records:
        offsets.append(offsets[-1] + len(record["legal_action_ids"]))
    count = len(records)
    arrays: dict[str, np.ndarray] = {
        "ready_ordinal": np.asarray(
            [record["ready_ordinal"] for record in records], dtype=np.int64
        ),
        "selected_ordinal": np.asarray(
            [record["selected_ordinal"] for record in records], dtype=np.int64
        ),
        "row_index": np.asarray(
            [record["row_index"] for record in records], dtype=np.int64
        ),
        "game_seed": np.asarray(
            [record["game_seed"] for record in records], dtype=np.int64
        ),
        "decision_index": np.asarray(
            [record["decision_index"] for record in records], dtype=np.int64
        ),
        "chunk_index": np.asarray(
            [record["chunk_index"] for record in records], dtype=np.int32
        ),
        "identity_sha256": np.asarray(
            [record["identity_sha256"] for record in records], dtype="<U71"
        ),
        "search_seed": np.asarray(
            [record["search_seed"] for record in records], dtype=np.uint64
        ),
        "selected_action_policy_id": np.asarray(
            [record["selected_action_policy_id"] for record in records],
            dtype=np.int32,
        ),
        "root_value": np.asarray(
            [record["root_value"] for record in records], dtype=np.float32
        ),
        "root_value_mask": np.ones(count, dtype=np.bool_),
        "simulations_used": np.asarray(
            [record["simulations_used"] for record in records], dtype=np.int32
        ),
        "used_full_search": np.ones(count, dtype=np.bool_),
        "q_values_root_perspective": np.ones(count, dtype=np.bool_),
        "legal_action_offsets": np.asarray(offsets, dtype=np.int64),
    }
    string_fields = (
        "target_policy_target_identity_sha256",
        "target_reanalyzer_checkpoint_sha256",
        "target_operator_contract_file_sha256",
    )
    for name in string_fields:
        arrays[name] = np.asarray([record[name] for record in records], dtype="<U71")
    ragged = {
        "legal_action_ids_flat": ("legal_action_ids", np.int32),
        "target_policy_flat": ("target_policy", np.float32),
        "target_policy_mask_flat": ("target_policy_mask", np.bool_),
        "target_scores_flat": ("target_scores", np.float32),
        "target_scores_mask_flat": ("target_scores_mask", np.bool_),
        "completed_q_values_flat": ("completed_q_values", np.float32),
        "completed_q_mask_flat": ("completed_q_mask", np.bool_),
        "prior_policy_flat": ("prior_policy", np.float32),
    }
    for output_name, (record_name, dtype) in ragged.items():
        values = [np.asarray(record[record_name], dtype=dtype) for record in records]
        arrays[output_name] = (
            np.concatenate(values) if values else np.asarray([], dtype=dtype)
        )
    neutral = unaudited_target_reliability_fields()
    for name in TARGET_RELIABILITY_COLUMNS:
        scalar = np.asarray(neutral[name])
        arrays[name] = np.full(count, scalar.item(), dtype=scalar.dtype)
    return arrays


def _execute_partition(args: argparse.Namespace) -> dict[str, Any]:
    qualification = _verify_receipt(args.receipt, require_current_runtime=True)
    plan = alignment._verify_plan(  # noqa: SLF001
        Path(str(qualification["stage_c_plan"]["path"]))
    )
    subset = _load_ready_subset(qualification)
    positions = _partition_positions(
        subset,
        partition_index=int(args.partition_index),
        partitions=int(args.partitions),
    )
    if positions.size == 0:
        raise ExecutorError("requested Stage-C execution partition is empty")
    overlay_path = Path(str(plan["eligibility_overlay"]["path"]))
    _overlay_path, overlay = alignment._load_json(  # noqa: SLF001
        overlay_path, where="Stage-C eligibility overlay"
    )
    corpus_root = Path(str(overlay["corpus"]["path"])).resolve(strict=True)
    if (
        alignment._file_sha256(corpus_root / "corpus_meta.json")
        != overlay[  # noqa: SLF001
            "corpus"
        ]["corpus_meta_file_sha256"]
    ):
        raise ExecutorError("Stage-C corpus metadata drifted before execution")
    data = train_bc.MemmapCorpus(corpus_root)
    selected_seeds = np.asarray(subset["game_seed"][positions], dtype=np.int64)
    sequences = _sequence_rows(data, selected_seeds)
    action_size = _checkpoint_action_size(_target_checkpoint(plan))
    correct_chance = bool(
        plan["target_policy_target_identity"]["chance"]["correct_rust_chance_spectra"]
    )
    history = plan["target_policy_target_identity"]["target_semantics"]
    evaluator = _evaluator_from_plan(plan, device=str(args.device))
    ordinals_by_game: dict[int, list[int]] = {}
    for position in positions.tolist():
        ordinals_by_game.setdefault(int(subset["game_seed"][position]), []).append(
            int(position)
        )
    target = plan["target_policy_target_identity"]
    if (
        target.get("schema_version") != alignment.OPERATOR_IDENTITY_SCHEMA_V2
        or target.get("target_execution") != alignment.STAGE_C_TARGET_EXECUTION
    ):
        raise ExecutorError(
            "new Stage-C execution requires the complete forced-full v2 target identity"
        )
    checkpoint_sha = str(target["producer_checkpoint"]["sha256"])
    operator_contract_sha = str(target["authority"]["contract"]["file_sha256"])
    records: list[dict[str, Any]] = []
    for game_seed, game_positions in sorted(ordinals_by_game.items()):
        sequence, game_rows = sequences[game_seed]
        decisions = [
            int(subset["decision_index"][position]) for position in game_positions
        ]
        reconstructed = reconstruct_state.reconstruct_states_from_sequence(
            sequence,
            decisions,
            correct_rust_chance_spectra=correct_chance,
            action_size=action_size,
        )
        for position in sorted(
            game_positions,
            key=lambda value: int(subset["decision_index"][value]),
        ):
            row = int(subset["row_index"][position])
            decision = int(subset["decision_index"][position])
            game = reconstructed.states.get(decision)
            if game is None:
                detail = reconstructed.failure or "unclassified sparse replay failure"
                raise ExecutorError(
                    f"qualified root became unreconstructable row={row}: {detail}"
                )
            roundtrip = reconstruct_state.round_trip_row(
                sequence,
                decision,
                _stored_roundtrip_features(data, row),
                _one_row(data["legal_action_ids"], row),
                correct_rust_chance_spectra=correct_chance,
                action_size=action_size,
                meaningful_public_history=bool(history["meaningful_public_history"]),
                history_limit=int(history["event_history_limit"]),
                reconstructed_game=game,
            )
            if not roundtrip.ok:
                raise ExecutorError(
                    "qualified root public surface drifted before search: "
                    f"row={row} detail={roundtrip.detail}"
                )
            identity = str(subset["identity_sha256"][position])
            seed = _row_seed(identity)
            patch = _search_patch(
                plan=plan,
                evaluator=evaluator,
                reconstructed_game=game,
                row_seed=seed,
                expected_legal_policy_ids=_one_row(data["legal_action_ids"], row),
            )
            records.append(
                {
                    "ready_ordinal": int(subset["ready_ordinal"][position]),
                    "selected_ordinal": int(subset["selected_ordinal"][position]),
                    "row_index": row,
                    "game_seed": game_seed,
                    "decision_index": decision,
                    "chunk_index": int(subset["chunk_index"][position]),
                    "identity_sha256": identity,
                    "search_seed": seed,
                    "target_policy_target_identity_sha256": target["identity_sha256"],
                    "target_reanalyzer_checkpoint_sha256": checkpoint_sha,
                    "target_operator_contract_file_sha256": operator_contract_sha,
                    **patch,
                }
            )
    records.sort(key=lambda value: int(value["ready_ordinal"]))
    arrays = _patch_arrays(records)
    patch_path = args.patch.expanduser().resolve(strict=False)
    alignment._write_immutable(  # noqa: SLF001
        patch_path,
        alignment._npz_bytes(arrays),  # noqa: SLF001
    )
    qualification_path = Path(str(qualification["path"]))
    effective = target["effective_gumbel_config"]
    receipt: dict[str, Any] = {
        "schema_version": EXECUTION_RECEIPT_SCHEMA,
        "patch_schema_version": PATCH_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "stage_c_plan": qualification["stage_c_plan"],
        "qualification_receipt": {
            "path": str(qualification_path),
            "file_sha256": alignment._file_sha256(qualification_path),  # noqa: SLF001
            "receipt_sha256": qualification["receipt_sha256"],
        },
        "target_policy_target_identity_sha256": target["identity_sha256"],
        "target_reanalyzer_checkpoint": target["producer_checkpoint"],
        "target_operator_contract": target["authority"]["contract"],
        "search": {
            "required_operator": "coherent_public_belief_search",
            "legacy_pimc_allowed": False,
            "authoritative_hidden_state_search_allowed": False,
            "native_hot_loop_required": True,
            "force_full": True,
            "nominal_n_full": 128,
            "row_seed_schema": ROW_SEED_SCHEMA,
            "effective_config_without_row_seed": effective,
            "effective_config_sha256": _value_sha256(effective),
            "target_execution": target["target_execution"],
            "target_execution_sha256": _value_sha256(target["target_execution"]),
            "row_search_evidence": {
                "schema": SEARCH_EVIDENCE_SCHEMA,
                "version": SEARCH_EVIDENCE_VERSION,
                "simulations": "simulations_used",
                "full_search": "used_full_search",
                "completed_q": "completed_q_values_flat",
                "completed_q_coverage": "completed_q_mask_flat",
                "raw_visited_q": "target_scores_flat_with_target_scores_mask_flat",
                "target_coverage": "target_policy_mask_flat_all_legal_actions",
            },
        },
        "evaluator": {
            "type": "EntityGraphRustEvaluator",
            "public_observation": True,
            "rust_featurize": True,
        },
        "runtime": _runtime_attestation(),
        "partition": {
            "partition_index": int(args.partition_index),
            "partitions": int(args.partitions),
            "assignment": "sealed_chunk_index_mod_partitions",
            "chunk_indices": sorted(
                set(int(value) for value in arrays["chunk_index"].tolist())
            ),
        },
        "counts": {
            "rows": len(records),
            "legal_actions": int(arrays["legal_action_offsets"][-1]),
        },
        "reliability": {
            "schema_version": TARGET_RELIABILITY_SCHEMA,
            "mode": "primary_search_only_unaudited_v1",
            "typed_neutral_sentinel": True,
        },
        "patch_columns": sorted(arrays),
        "artifact": alignment._artifact_ref(patch_path),  # noqa: SLF001
    }
    receipt["receipt_sha256"] = _value_sha256(receipt)
    return receipt


def _verify_patch_arrays(
    arrays: Mapping[str, np.ndarray], *, receipt: Mapping[str, Any]
) -> None:
    patch_schema = str(receipt.get("patch_schema_version", ""))
    if patch_schema not in {PATCH_SCHEMA_V1, PATCH_SCHEMA}:
        raise ExecutorError("unsupported Stage-C patch schema")
    expected = {
        *PATCH_ROW_COLUMNS,
        *PATCH_RAGGED_COLUMNS,
        "legal_action_offsets",
    }
    if set(arrays) != expected or sorted(arrays) != receipt.get("patch_columns"):
        raise ExecutorError("Stage-C patch column contract drifted")
    count = int(receipt["counts"]["rows"])
    for name in PATCH_ROW_COLUMNS:
        if np.asarray(arrays[name]).shape != (count,):
            raise ExecutorError(f"Stage-C patch row column {name} is misaligned")
    offsets = np.asarray(arrays["legal_action_offsets"], dtype=np.int64)
    if (
        offsets.shape != (count + 1,)
        or offsets[0] != 0
        or np.any(offsets[1:] <= offsets[:-1])
    ):
        raise ExecutorError("Stage-C patch legal offsets are malformed")
    flat_count = int(offsets[-1])
    if flat_count != int(receipt["counts"]["legal_actions"]):
        raise ExecutorError("Stage-C patch legal-action count drifted")
    for name in PATCH_RAGGED_COLUMNS:
        if np.asarray(arrays[name]).shape != (flat_count,):
            raise ExecutorError(f"Stage-C patch ragged column {name} is misaligned")
    if (
        np.unique(arrays["ready_ordinal"]).size != count
        or np.unique(arrays["identity_sha256"]).size != count
    ):
        raise ExecutorError("Stage-C patch contains duplicate rows")
    target_identity = str(receipt["target_policy_target_identity_sha256"])
    checkpoint_sha = str(receipt["target_reanalyzer_checkpoint"]["sha256"])
    operator_sha = str(receipt["target_operator_contract"]["file_sha256"])
    for name, expected_value in (
        ("target_policy_target_identity_sha256", target_identity),
        ("target_reanalyzer_checkpoint_sha256", checkpoint_sha),
        ("target_operator_contract_file_sha256", operator_sha),
    ):
        if not np.all(np.asarray(arrays[name]).astype(str) == expected_value):
            raise ExecutorError(f"Stage-C patch {name} provenance drifted")
    neutral = unaudited_target_reliability_fields()
    for name in TARGET_RELIABILITY_COLUMNS:
        values = np.asarray(arrays[name])
        expected_value = np.asarray(neutral[name]).item()
        if isinstance(expected_value, float) and math.isnan(expected_value):
            valid = np.all(np.isnan(values))
        else:
            valid = np.all(values == expected_value)
        if not valid:
            raise ExecutorError("Stage-C patch reliability sentinel drifted")
    effective_config = None
    if patch_schema == PATCH_SCHEMA:
        search = receipt.get("search")
        if not isinstance(search, Mapping) or not isinstance(
            search.get("effective_config_without_row_seed"), Mapping
        ):
            raise ExecutorError("Stage-C v2 patch lost its effective search config")
        effective_config = search["effective_config_without_row_seed"]
    for row in range(count):
        start, stop = int(offsets[row]), int(offsets[row + 1])
        legal = np.asarray(arrays["legal_action_ids_flat"])[start:stop]
        target = np.asarray(arrays["target_policy_flat"])[start:stop]
        prior = np.asarray(arrays["prior_policy_flat"])[start:stop]
        scores = np.asarray(arrays["target_scores_flat"])[start:stop]
        score_mask = np.asarray(arrays["target_scores_mask_flat"])[start:stop]
        completed = np.asarray(arrays["completed_q_values_flat"])[start:stop]
        target_coverage = np.asarray(arrays["target_policy_mask_flat"])[start:stop]
        coverage_valid = (
            np.array_equal(target_coverage, target > 0.0)
            if patch_schema == PATCH_SCHEMA_V1
            else bool(np.all(target_coverage))
        )
        if (
            np.any(legal < 0)
            or np.unique(legal).size != legal.size
            or int(arrays["selected_action_policy_id"][row]) not in set(legal.tolist())
            or not np.all(np.isfinite(target))
            or np.any(target < 0.0)
            or not coverage_valid
            or not np.array_equal(score_mask, np.isfinite(scores))
            or not np.all(np.asarray(arrays["completed_q_mask_flat"])[start:stop])
            or not np.isclose(float(target.sum()), 1.0, atol=1.0e-5)
            or not np.isclose(float(prior.sum()), 1.0, atol=1.0e-5)
            or not np.all(np.isfinite(completed))
            or int(arrays["search_seed"][row])
            != _row_seed(str(arrays["identity_sha256"][row]))
            or not bool(arrays["root_value_mask"][row])
            or not bool(arrays["used_full_search"][row])
            or not bool(arrays["q_values_root_perspective"][row])
            or (
                int(arrays["simulations_used"][row]) <= 0
                if effective_config is None
                else int(arrays["simulations_used"][row])
                != _expected_forced_full_simulations(stop - start, effective_config)
            )
        ):
            raise ExecutorError("Stage-C patch contains invalid search evidence")


def _verify_execution_receipt(path: Path) -> dict[str, Any]:
    receipt_path, receipt = alignment._load_json(  # noqa: SLF001
        path, where="Stage-C execution receipt"
    )
    unsigned = dict(receipt)
    stated = unsigned.pop("receipt_sha256", None)
    patch_schema = receipt.get("patch_schema_version")
    if (
        receipt.get("schema_version") != EXECUTION_RECEIPT_SCHEMA
        or patch_schema not in {PATCH_SCHEMA_V1, PATCH_SCHEMA}
        or stated != _value_sha256(unsigned)
    ):
        raise ExecutorError("Stage-C execution receipt digest drifted")
    qualification_ref = receipt["qualification_receipt"]
    qualification_path = Path(str(qualification_ref["path"]))
    qualification = _verify_receipt(qualification_path)
    if (
        qualification_ref.get("file_sha256")
        != alignment._file_sha256(qualification_path)  # noqa: SLF001
        or qualification_ref.get("receipt_sha256") != qualification["receipt_sha256"]
    ):
        raise ExecutorError("Stage-C qualification binding drifted")
    plan = alignment._verify_plan(  # noqa: SLF001
        Path(str(receipt["stage_c_plan"]["path"]))
    )
    target = plan["target_policy_target_identity"]
    if (
        receipt.get("target_policy_target_identity_sha256") != target["identity_sha256"]
        or receipt.get("target_reanalyzer_checkpoint") != target["producer_checkpoint"]
        or receipt.get("target_operator_contract") != target["authority"]["contract"]
    ):
        raise ExecutorError("Stage-C execution target identity drifted")
    if (
        alignment._file_sha256(_target_checkpoint(plan))
        != target[  # noqa: SLF001
            "producer_checkpoint"
        ]["sha256"]
    ):
        raise ExecutorError("Stage-C target checkpoint drifted")
    runtime = receipt.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ExecutorError("Stage-C execution runtime is malformed")
    _verify_runtime_attestation(runtime, require_current=False)
    search = receipt.get("search")
    if not isinstance(search, Mapping):
        raise ExecutorError("Stage-C execution search evidence is malformed")
    effective = dataclasses.asdict(_effective_search_config(plan, row_seed=0))
    effective.pop("seed", None)
    effective = alignment._json_normalized(effective)  # noqa: SLF001
    if (
        search.get("force_full") is not True
        or search.get("nominal_n_full") != 128
        or search.get("effective_config_without_row_seed") != effective
        or search.get("effective_config_sha256") != _value_sha256(effective)
    ):
        raise ExecutorError("Stage-C execution effective search config drifted")
    if patch_schema == PATCH_SCHEMA and (
        search.get("target_execution") != alignment.STAGE_C_TARGET_EXECUTION
        or search.get("target_execution_sha256")
        != _value_sha256(alignment.STAGE_C_TARGET_EXECUTION)
        or search.get("row_search_evidence", {}).get("schema") != SEARCH_EVIDENCE_SCHEMA
    ):
        raise ExecutorError("Stage-C v2 execution target evidence drifted")
    artifact = receipt["artifact"]
    patch_path = Path(str(artifact["path"])).resolve(strict=True)
    if (
        artifact.get("file_sha256") != alignment._file_sha256(patch_path)  # noqa: SLF001
        or artifact.get("size_bytes") != patch_path.stat().st_size
    ):
        raise ExecutorError("Stage-C execution patch bytes drifted")
    with np.load(patch_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    _verify_patch_arrays(arrays, receipt=receipt)
    return {
        "path": str(receipt_path),
        "file_sha256": alignment._file_sha256(receipt_path),  # noqa: SLF001
        "qualification": qualification,
        "plan": plan,
        "arrays": arrays,
        **receipt,
    }


def _record_from_patch(arrays: Mapping[str, np.ndarray], row: int) -> dict[str, Any]:
    offsets = np.asarray(arrays["legal_action_offsets"], dtype=np.int64)
    start, stop = int(offsets[row]), int(offsets[row + 1])
    record = {name: np.asarray(arrays[name])[row].item() for name in PATCH_ROW_COLUMNS}
    for output_name, record_name in (
        ("legal_action_ids_flat", "legal_action_ids"),
        ("target_policy_flat", "target_policy"),
        ("target_policy_mask_flat", "target_policy_mask"),
        ("target_scores_flat", "target_scores"),
        ("target_scores_mask_flat", "target_scores_mask"),
        ("completed_q_values_flat", "completed_q_values"),
        ("completed_q_mask_flat", "completed_q_mask"),
        ("prior_policy_flat", "prior_policy"),
    ):
        record[record_name] = np.asarray(arrays[output_name])[start:stop]
    return record


def _merge_executions(args: argparse.Namespace) -> dict[str, Any]:
    executions = [_verify_execution_receipt(path) for path in args.receipt]
    if not executions:
        raise ExecutorError("Stage-C merge requires execution receipts")
    first = executions[0]
    plan_sha = first["stage_c_plan"]["plan_sha256"]
    qualification_sha = first["qualification_receipt"]["receipt_sha256"]
    target_sha = first["target_policy_target_identity_sha256"]
    if any(
        execution["stage_c_plan"]["plan_sha256"] != plan_sha
        or execution["qualification_receipt"]["receipt_sha256"] != qualification_sha
        or execution["target_policy_target_identity_sha256"] != target_sha
        for execution in executions[1:]
    ):
        raise ExecutorError("Stage-C merge received foreign execution receipts")
    partition_counts = {int(item["partition"]["partitions"]) for item in executions}
    if len(partition_counts) != 1:
        raise ExecutorError("Stage-C execution receipts disagree on partition count")
    partitions = partition_counts.pop()
    partition_indices = [
        int(item["partition"]["partition_index"]) for item in executions
    ]
    if sorted(partition_indices) != list(range(partitions)):
        raise ExecutorError(
            "Stage-C merge requires exactly one receipt for every partition"
        )
    ready_subset = _load_ready_subset(first["qualification"])
    expected_count = len(ready_subset["identity_sha256"])
    records_by_ordinal: dict[int, dict[str, Any]] = {}
    for execution in executions:
        arrays = execution["arrays"]
        for row in range(int(execution["counts"]["rows"])):
            record = _record_from_patch(arrays, row)
            ordinal = int(record["ready_ordinal"])
            if ordinal in records_by_ordinal:
                raise ExecutorError(f"duplicate Stage-C row claim ordinal={ordinal}")
            if not 0 <= ordinal < expected_count:
                raise ExecutorError("Stage-C row claim has foreign ready ordinal")
            if (
                str(record["identity_sha256"])
                != str(ready_subset["identity_sha256"][ordinal])
                or int(record["row_index"]) != int(ready_subset["row_index"][ordinal])
                or int(record["game_seed"]) != int(ready_subset["game_seed"][ordinal])
                or int(record["decision_index"])
                != int(ready_subset["decision_index"][ordinal])
                or int(record["chunk_index"])
                != int(ready_subset["chunk_index"][ordinal])
            ):
                raise ExecutorError("Stage-C row claim identity drifted")
            records_by_ordinal[ordinal] = record
    if set(records_by_ordinal) != set(range(expected_count)):
        raise ExecutorError(
            "Stage-C merge has incomplete reconstructable-root coverage: "
            f"got={len(records_by_ordinal)} expected={expected_count}"
        )
    merged_arrays = _patch_arrays(
        [records_by_ordinal[index] for index in range(expected_count)]
    )
    output_path = args.output.expanduser().resolve(strict=False)
    alignment._write_immutable(  # noqa: SLF001
        output_path,
        alignment._npz_bytes(merged_arrays),  # noqa: SLF001
    )
    receipt_refs = [
        {
            "path": execution["path"],
            "file_sha256": execution["file_sha256"],
            "receipt_sha256": execution["receipt_sha256"],
            "partition_index": execution["partition"]["partition_index"],
        }
        for execution in sorted(
            executions, key=lambda item: int(item["partition"]["partition_index"])
        )
    ]
    receipt: dict[str, Any] = {
        "schema_version": MERGE_RECEIPT_SCHEMA,
        "patch_schema_version": PATCH_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "stage_c_plan": first["stage_c_plan"],
        "qualification_receipt": first["qualification_receipt"],
        "target_policy_target_identity_sha256": target_sha,
        "target_reanalyzer_checkpoint": first["target_reanalyzer_checkpoint"],
        "target_operator_contract": first["target_operator_contract"],
        "search": first["search"],
        "evaluator": first["evaluator"],
        "runtime": first["runtime"],
        "execution_receipts": receipt_refs,
        "counts": {
            "partitions": partitions,
            "rows": expected_count,
            "legal_actions": int(merged_arrays["legal_action_offsets"][-1]),
        },
        "coverage": {
            "scope": "all_qualified_reconstructable_roots",
            "missing_rows": 0,
            "duplicate_rows": 0,
            "ordered_by": "qualification_ready_ordinal",
        },
        "non_target_source_columns_mutated": False,
        "source_corpus_rewritten": False,
        "patch_columns": sorted(merged_arrays),
        "artifact": alignment._artifact_ref(output_path),  # noqa: SLF001
    }
    receipt["receipt_sha256"] = _value_sha256(receipt)
    return receipt


def _new_target_identity_from_legacy_merge(
    legacy: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the forced-full v2 identity from one fully verified v1 DAG."""

    plan = alignment._verify_plan(  # noqa: SLF001
        Path(str(legacy["stage_c_plan"]["path"]))
    )
    old_target = plan["target_policy_target_identity"]
    if (
        legacy.get("schema_version") != MERGE_RECEIPT_SCHEMA
        or legacy.get("patch_schema_version") != PATCH_SCHEMA_V1
        or old_target.get("schema_version") != alignment.OPERATOR_IDENTITY_SCHEMA_V1
        or legacy.get("target_policy_target_identity_sha256")
        != old_target.get("identity_sha256")
    ):
        raise ExecutorError("rebind accepts only a complete legacy Stage-C v1 merge")
    contract = old_target["authority"]["contract"]
    checkpoint = old_target["producer_checkpoint"]
    rebound = alignment._operator_identity(  # noqa: SLF001
        Path(str(contract["path"])),
        Path(str(checkpoint["path"])),
        require_current_target=True,
        identity_schema=alignment.OPERATOR_IDENTITY_SCHEMA_V2,
        target_execution=alignment.STAGE_C_TARGET_EXECUTION,
    )
    if rebound["identity_sha256"] == old_target["identity_sha256"]:
        raise ExecutorError("forced-full v2 target identity did not separate from v1")
    effective = rebound["effective_gumbel_config"]
    search = legacy.get("search")
    if (
        not isinstance(search, Mapping)
        or search.get("force_full") is not True
        or search.get("nominal_n_full") != 128
        or search.get("row_seed_schema") != ROW_SEED_SCHEMA
        or search.get("effective_config_without_row_seed") != effective
        or search.get("effective_config_sha256") != _value_sha256(effective)
        or effective.get("policy_target_min_visits") != 0
    ):
        raise ExecutorError("legacy merge lacks an unambiguous forced-full config")
    return rebound


def _rebound_search_receipt(
    legacy_search: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, Any]:
    search = dict(legacy_search)
    search.update(
        {
            "target_execution": target["target_execution"],
            "target_execution_sha256": _value_sha256(target["target_execution"]),
            "row_search_evidence": {
                "schema": SEARCH_EVIDENCE_SCHEMA,
                "version": SEARCH_EVIDENCE_VERSION,
                "simulations": "simulations_used",
                "full_search": "used_full_search",
                "completed_q": "completed_q_values_flat",
                "completed_q_coverage": "completed_q_mask_flat",
                "raw_visited_q": "target_scores_flat_with_target_scores_mask_flat",
                "target_coverage": "target_policy_mask_flat_all_legal_actions",
                "visit_counts": "not_present_in_v1_patch_not_required_by_overlay",
            },
        }
    )
    return search


def _rebind_legacy_merge(args: argparse.Namespace) -> dict[str, Any]:
    legacy = _verify_merge_receipt(args.receipt)
    target = _new_target_identity_from_legacy_merge(legacy)
    source_patch = Path(str(legacy["artifact"]["path"])).resolve(strict=True)
    with np.load(source_patch, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]).copy() for name in source.files}
    if (
        not np.all(arrays["used_full_search"])
        or np.any(arrays["simulations_used"] <= 0)
        or not np.all(arrays["completed_q_mask_flat"])
    ):
        raise ExecutorError("legacy patch lacks exact forced-full per-row evidence")
    arrays["target_policy_mask_flat"] = np.ones(
        arrays["target_policy_flat"].shape, dtype=np.bool_
    )
    arrays["target_policy_target_identity_sha256"] = np.full(
        arrays["row_index"].shape,
        target["identity_sha256"],
        dtype="<U71",
    )
    output = args.output.expanduser().resolve(strict=False)
    alignment._write_immutable(  # noqa: SLF001
        output,
        alignment._npz_bytes(arrays),  # noqa: SLF001
    )
    source_receipt = args.receipt.expanduser().resolve(strict=True)
    receipt: dict[str, Any] = {
        "schema_version": REBOUND_MERGE_RECEIPT_SCHEMA,
        "patch_schema_version": PATCH_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "stage_c_plan": legacy["stage_c_plan"],
        "qualification_receipt": legacy["qualification_receipt"],
        "source_legacy_merge": {
            "path": str(source_receipt),
            "file_sha256": alignment._file_sha256(source_receipt),  # noqa: SLF001
            "receipt_sha256": legacy["receipt_sha256"],
            "patch_file_sha256": legacy["artifact"]["file_sha256"],
            "old_target_policy_target_identity_sha256": legacy[
                "target_policy_target_identity_sha256"
            ],
        },
        "target_policy_target_identity": target,
        "target_policy_target_identity_sha256": target["identity_sha256"],
        "target_reanalyzer_checkpoint": target["producer_checkpoint"],
        "target_operator_contract": target["authority"]["contract"],
        "search": _rebound_search_receipt(legacy["search"], target),
        "evaluator": legacy["evaluator"],
        "runtime": legacy["runtime"],
        "counts": dict(legacy["counts"]),
        "coverage": {
            **dict(legacy["coverage"]),
            "target_policy_mask_semantics": "all_legal_actions_have_teacher_labels",
        },
        "migration": {
            "mode": "authenticated_semantic_rebind_without_search_rerun",
            "allowed_mutations": [
                "target_policy_mask_flat",
                "target_policy_target_identity_sha256",
            ],
            "search_outputs_recomputed": False,
            "reason": (
                "v1 omitted force_full and resolved default search fields from "
                "target identity; the sealed execution DAG proves both"
            ),
        },
        "non_target_source_columns_mutated": False,
        "source_corpus_rewritten": False,
        "patch_columns": sorted(arrays),
        "artifact": alignment._artifact_ref(output),  # noqa: SLF001
    }
    receipt["receipt_sha256"] = _value_sha256(receipt)
    _verify_patch_arrays(arrays, receipt=receipt)
    return receipt


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype.kind in "fc" or right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _verify_rebound_merge_receipt(
    receipt_path: Path, receipt: Mapping[str, Any]
) -> dict[str, Any]:
    unsigned = dict(receipt)
    stated = unsigned.pop("receipt_sha256", None)
    if receipt.get("patch_schema_version") != PATCH_SCHEMA or stated != _value_sha256(
        unsigned
    ):
        raise ExecutorError("Stage-C rebound merge digest drifted")
    source_ref = receipt.get("source_legacy_merge")
    if not isinstance(source_ref, Mapping):
        raise ExecutorError("Stage-C rebound merge lost its legacy authority")
    source_path = Path(str(source_ref["path"])).resolve(strict=True)
    legacy = _verify_merge_receipt(source_path)
    if (
        legacy.get("schema_version") != MERGE_RECEIPT_SCHEMA
        or legacy.get("patch_schema_version") != PATCH_SCHEMA_V1
        or source_ref.get("file_sha256") != alignment._file_sha256(source_path)  # noqa: SLF001
        or source_ref.get("receipt_sha256") != legacy["receipt_sha256"]
        or source_ref.get("patch_file_sha256") != legacy["artifact"]["file_sha256"]
        or source_ref.get("old_target_policy_target_identity_sha256")
        != legacy["target_policy_target_identity_sha256"]
    ):
        raise ExecutorError("Stage-C rebound legacy merge binding drifted")
    target = _new_target_identity_from_legacy_merge(legacy)
    if (
        receipt.get("target_policy_target_identity") != target
        or receipt.get("target_policy_target_identity_sha256")
        != target["identity_sha256"]
        or receipt.get("target_reanalyzer_checkpoint") != target["producer_checkpoint"]
        or receipt.get("target_operator_contract") != target["authority"]["contract"]
        or receipt.get("search") != _rebound_search_receipt(legacy["search"], target)
        or receipt.get("runtime") != legacy["runtime"]
        or receipt.get("counts") != legacy["counts"]
    ):
        raise ExecutorError("Stage-C rebound semantic identity drifted")
    artifact = receipt.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ExecutorError("Stage-C rebound patch artifact is malformed")
    output = Path(str(artifact["path"])).resolve(strict=True)
    if (
        artifact.get("file_sha256") != alignment._file_sha256(output)  # noqa: SLF001
        or artifact.get("size_bytes") != output.stat().st_size
    ):
        raise ExecutorError("Stage-C rebound patch bytes drifted")
    with np.load(output, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    _verify_patch_arrays(arrays, receipt=receipt)
    legacy_path = Path(str(legacy["artifact"]["path"])).resolve(strict=True)
    with np.load(legacy_path, allow_pickle=False) as source:
        old_arrays = {name: np.asarray(source[name]) for name in source.files}
    mutable = {
        "target_policy_mask_flat",
        "target_policy_target_identity_sha256",
    }
    if set(arrays) != set(old_arrays) or any(
        not _arrays_equal(arrays[name], old_arrays[name])
        for name in arrays
        if name not in mutable
    ):
        raise ExecutorError("Stage-C rebound modified search outputs or row identity")
    if not np.all(arrays["target_policy_mask_flat"]) or not np.all(
        arrays["target_policy_target_identity_sha256"].astype(str)
        == target["identity_sha256"]
    ):
        raise ExecutorError("Stage-C rebound mask or target identity is incomplete")
    return {"path": str(receipt_path), **dict(receipt)}


def _verify_merge_receipt(path: Path) -> dict[str, Any]:
    receipt_path, receipt = alignment._load_json(  # noqa: SLF001
        path, where="Stage-C merge receipt"
    )
    if receipt.get("schema_version") == REBOUND_MERGE_RECEIPT_SCHEMA:
        return _verify_rebound_merge_receipt(receipt_path, receipt)
    unsigned = dict(receipt)
    stated = unsigned.pop("receipt_sha256", None)
    if (
        receipt.get("schema_version") != MERGE_RECEIPT_SCHEMA
        or receipt.get("patch_schema_version") not in {PATCH_SCHEMA_V1, PATCH_SCHEMA}
        or stated != _value_sha256(unsigned)
    ):
        raise ExecutorError("Stage-C merge receipt digest drifted")
    executions = [
        _verify_execution_receipt(Path(str(reference["path"])))
        for reference in receipt["execution_receipts"]
    ]
    for reference, execution in zip(
        receipt["execution_receipts"], executions, strict=True
    ):
        if (
            reference.get("file_sha256") != execution["file_sha256"]
            or reference.get("receipt_sha256") != execution["receipt_sha256"]
        ):
            raise ExecutorError("Stage-C merge execution receipt binding drifted")
    artifact = receipt["artifact"]
    output = Path(str(artifact["path"])).resolve(strict=True)
    if (
        artifact.get("file_sha256") != alignment._file_sha256(output)  # noqa: SLF001
        or artifact.get("size_bytes") != output.stat().st_size
    ):
        raise ExecutorError("Stage-C merged patch bytes drifted")
    with np.load(output, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    _verify_patch_arrays(arrays, receipt=receipt)
    if not executions:
        raise ExecutorError("Stage-C merge contains no execution receipts")
    first = executions[0]
    partitions = int(first["partition"]["partitions"])
    if (
        sorted(int(item["partition"]["partition_index"]) for item in executions)
        != list(range(partitions))
        or any(
            int(item["partition"]["partitions"]) != partitions for item in executions
        )
        or receipt.get("stage_c_plan") != first["stage_c_plan"]
        or receipt.get("qualification_receipt") != first["qualification_receipt"]
        or receipt.get("target_policy_target_identity_sha256")
        != first["target_policy_target_identity_sha256"]
        or receipt.get("target_reanalyzer_checkpoint")
        != first["target_reanalyzer_checkpoint"]
        or receipt.get("target_operator_contract") != first["target_operator_contract"]
        or receipt.get("search") != first["search"]
        or receipt.get("evaluator") != first["evaluator"]
        or receipt.get("runtime") != first["runtime"]
    ):
        raise ExecutorError("Stage-C merge execution authority drifted")
    ready = _load_ready_subset(first["qualification"])
    records: dict[int, dict[str, Any]] = {}
    for execution in executions:
        for row in range(int(execution["counts"]["rows"])):
            record = _record_from_patch(execution["arrays"], row)
            ordinal = int(record["ready_ordinal"])
            if ordinal in records or not 0 <= ordinal < len(ready["row_index"]):
                raise ExecutorError("Stage-C merge has duplicate/foreign ready ordinal")
            records[ordinal] = record
    if set(records) != set(range(len(ready["row_index"]))):
        raise ExecutorError("Stage-C merge execution DAG has incomplete row coverage")
    rebuilt = _patch_arrays([records[index] for index in range(len(records))])
    if set(rebuilt) != set(arrays) or any(
        not _arrays_equal(rebuilt[name], arrays[name]) for name in arrays
    ):
        raise ExecutorError("Stage-C merged patch differs from its execution DAG")
    if (
        int(receipt["counts"]["rows"]) != len(arrays["row_index"])
        or int(receipt["counts"]["partitions"]) != partitions
        or int(receipt["counts"]["legal_actions"])
        != int(arrays["legal_action_offsets"][-1])
    ):
        raise ExecutorError("Stage-C merged row count drifted")
    return {"path": str(receipt_path), **receipt}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    qualify = commands.add_parser("qualify")
    qualify.add_argument("--plan", required=True, type=Path)
    qualify.add_argument("--output-root", required=True, type=Path)
    qualify.add_argument("--write", required=True, type=Path)
    qualify_partition = commands.add_parser(
        "qualify-partition",
        help="qualify one deterministic whole-game CPU partition",
    )
    qualify_partition.add_argument("--plan", required=True, type=Path)
    qualify_partition.add_argument("--partition-index", required=True, type=int)
    qualify_partition.add_argument("--partitions", required=True, type=int)
    qualify_partition.add_argument("--artifact", required=True, type=Path)
    qualify_partition.add_argument("--write", required=True, type=Path)
    merge_qualification = commands.add_parser(
        "merge-qualification",
        help="merge complete whole-game qualification partitions",
    )
    merge_qualification.add_argument(
        "--receipt", action="append", required=True, type=Path
    )
    merge_qualification.add_argument("--output-root", required=True, type=Path)
    merge_qualification.add_argument("--write", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--receipt", required=True, type=Path)
    execute = commands.add_parser(
        "execute", help="run one deterministic GPU partition of coherent n128 roots"
    )
    execute.add_argument("--receipt", required=True, type=Path)
    execute.add_argument("--partition-index", required=True, type=int)
    execute.add_argument("--partitions", required=True, type=int)
    execute.add_argument("--device", default="cuda")
    execute.add_argument("--patch", required=True, type=Path)
    execute.add_argument("--write", required=True, type=Path)
    verify_execute = commands.add_parser("verify-execution")
    verify_execute.add_argument("--receipt", required=True, type=Path)
    merge = commands.add_parser(
        "merge", help="merge complete coherent n128 execution partitions"
    )
    merge.add_argument("--receipt", action="append", required=True, type=Path)
    merge.add_argument("--output", required=True, type=Path)
    merge.add_argument("--write", required=True, type=Path)
    verify_merge = commands.add_parser("verify-merge")
    verify_merge.add_argument("--receipt", required=True, type=Path)
    rebind = commands.add_parser(
        "rebind-legacy-merge",
        help="rebind one authenticated v1 full-search DAG to the complete v2 identity",
    )
    rebind.add_argument("--receipt", required=True, type=Path)
    rebind.add_argument("--output", required=True, type=Path)
    rebind.add_argument("--write", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "qualify":
            result = _qualify(args)
            alignment._write_json_immutable(args.write, result)  # noqa: SLF001
        elif args.command == "qualify-partition":
            result = _qualify_partition(args)
            alignment._write_json_immutable(args.write, result)  # noqa: SLF001
        elif args.command == "merge-qualification":
            result = _merge_qualification_partitions(args)
            alignment._write_json_immutable(args.write, result)  # noqa: SLF001
        elif args.command == "verify":
            result = _verify_receipt(args.receipt)
        elif args.command == "execute":
            result = _execute_partition(args)
            alignment._write_json_immutable(args.write, result)  # noqa: SLF001
        elif args.command == "verify-execution":
            result = _verify_execution_receipt(args.receipt)
            result = {
                key: value
                for key, value in result.items()
                if key not in {"arrays", "qualification", "plan"}
            }
        elif args.command == "merge":
            result = _merge_executions(args)
            alignment._write_json_immutable(args.write, result)  # noqa: SLF001
        elif args.command == "rebind-legacy-merge":
            result = _rebind_legacy_merge(args)
            alignment._write_json_immutable(args.write, result)  # noqa: SLF001
        else:
            result = _verify_merge_receipt(args.receipt)
    except (
        ExecutorError,
        alignment.AlignmentError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Stage-C reconstruction refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
