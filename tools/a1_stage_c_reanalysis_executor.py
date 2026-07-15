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


RECEIPT_SCHEMA = "a1-stage-c-sparse-reconstruction-receipt-v1"
READY_SUBSET_SCHEMA = "a1-stage-c-reconstructable-subset-v1"
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
REQUIRED_COHERENT_CAPABILITIES = frozenset(
    {
        "coherent_public_belief_search",
        "forced_root_trajectory_only",
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
        raise ExecutorError(f"cannot attest sparse reconstruction runtime: {error}") from error
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
        REPO_ROOT / "src/catan_zero/search/gumbel_chance_mcts.py",
        REPO_ROOT / "src/catan_zero/search/native_gumbel_mcts.py",
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
        if (
            game_decisions[0] != 0
            or np.any(game_decisions[1:] <= game_decisions[:-1])
        ):
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


def _stored_roundtrip_features(data: Mapping[str, Any], row: int) -> dict[str, np.ndarray]:
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
    if alignment._file_sha256(corpus_root / "corpus_meta.json") != overlay[  # noqa: SLF001
        "corpus"
    ]["corpus_meta_file_sha256"]:
        raise ExecutorError("Stage-C corpus metadata drifted")
    data = train_bc.MemmapCorpus(corpus_root)
    with np.load(subset_path, allow_pickle=False) as subset_file:
        subset = {name: np.asarray(subset_file[name]) for name in subset_file.files}
    count = len(subset["row_index"])
    if count != int(plan["subset"]["selected_rows"]):
        raise ExecutorError("Stage-C selected subset row count drifted")
    sequences = _sequence_rows(data, subset["game_seed"])

    status = np.full(count, STATUS["unclassified"], dtype=np.uint8)
    omitted = np.zeros(count, dtype=np.uint16)
    omitted_roll = np.zeros(count, dtype=np.uint16)
    omitted_end_turn = np.zeros(count, dtype=np.uint16)
    omitted_other_ui = np.zeros(count, dtype=np.uint16)
    failures: list[dict[str, Any]] = []
    source_checkpoint = Path(
        str(plan["source_policy_target_identity"]["producer_checkpoint"]["path"])
    )
    action_size = _checkpoint_action_size(source_checkpoint)
    history = plan["target_policy_target_identity"]["target_semantics"]
    correct_chance = bool(
        plan["target_policy_target_identity"]["chance"][
            "correct_rust_chance_spectra"
        ]
    )
    ordinals_by_game: dict[int, list[int]] = {}
    for ordinal, game_seed in enumerate(subset["game_seed"].astype(np.int64)):
        ordinals_by_game.setdefault(int(game_seed), []).append(ordinal)
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
        for ordinal in game_ordinals:
            row = int(subset["row_index"][ordinal])
            decision = int(subset["decision_index"][ordinal])
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
                status[ordinal], detail = _status_for_error(error)
                failures.append(
                    {
                        "ordinal": ordinal,
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
            omitted[ordinal] = np.uint16(omitted_count)
            omitted_types = batch.omitted_automatic_transition_types[decision]
            roll_count = int(omitted_types.get("ROLL", 0))
            end_turn_count = int(omitted_types.get("END_TURN", 0))
            other_count = omitted_count - roll_count - end_turn_count
            if min(roll_count, end_turn_count, other_count) < 0:
                raise ExecutorError("omitted automatic-transition type counts drifted")
            omitted_roll[ordinal] = np.uint16(roll_count)
            omitted_end_turn[ordinal] = np.uint16(end_turn_count)
            omitted_other_ui[ordinal] = np.uint16(other_count)
            try:
                result = reconstruct_state.round_trip_row(
                    sequence,
                    decision,
                    _stored_roundtrip_features(data, row),
                    _one_row(data["legal_action_ids"], row),
                    correct_rust_chance_spectra=correct_chance,
                    action_size=action_size,
                    meaningful_public_history=bool(
                        history["meaningful_public_history"]
                    ),
                    history_limit=int(history["event_history_limit"]),
                    reconstructed_game=reconstructed_game,
                )
            except Exception as error:  # noqa: BLE001 - classify row, continue.
                status[ordinal], detail = _status_for_error(error)
                failures.append(
                    {
                        "ordinal": ordinal,
                        "row_index": row,
                        "game_seed": game_seed,
                        "decision_index": decision,
                        **detail,
                    }
                )
                continue
            if not result.ok:
                status[ordinal] = STATUS["public_surface_mismatch"]
                failures.append(
                    {
                        "ordinal": ordinal,
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
            status[ordinal] = STATUS["reconstructable_public_roundtrip"]

    output_root = args.output_root.expanduser().resolve(strict=False)
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
    ready_arrays["omitted_automatic_transitions_before_root"] = omitted[ready]
    ready_arrays["omitted_roll_transitions_before_root"] = omitted_roll[ready]
    ready_arrays["omitted_end_turn_transitions_before_root"] = omitted_end_turn[
        ready
    ]
    ready_arrays["omitted_other_ui_transitions_before_root"] = omitted_other_ui[
        ready
    ]
    alignment._write_immutable(  # noqa: SLF001
        ready_path, alignment._npz_bytes(ready_arrays)  # noqa: SLF001
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
        "target_policy_target_identity_sha256": plan[
            "target_policy_target_identity"
        ]["identity_sha256"],
        "runtime": _runtime_attestation(),
        "source_checkpoint_action_size": action_size,
        "status_codes": STATUS,
        "counts": {
            "selected_rows": count,
            **counts,
        },
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
        "failure_examples": failures[:100],
    }
    receipt["receipt_sha256"] = _value_sha256(receipt)
    return receipt


def _verify_receipt(path: Path) -> dict[str, Any]:
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
    runtime_unsigned = dict(runtime)
    runtime_stated = runtime_unsigned.pop("runtime_sha256", None)
    if runtime_stated != _value_sha256(runtime_unsigned):
        raise ExecutorError("Stage-C reconstruction runtime digest drifted")
    for source in runtime.get("sources", ()):
        source_path = REPO_ROOT / str(source["path"])
        if source.get("file_sha256") != alignment._file_sha256(source_path):  # noqa: SLF001
            raise ExecutorError("Stage-C reconstruction source bytes drifted")
    native = runtime.get("native_runtime")
    if not isinstance(native, Mapping):
        raise ExecutorError("Stage-C reconstruction native runtime is malformed")
    native_path = Path(str(native["path"])).resolve(strict=True)
    if native.get("file_sha256") != alignment._file_sha256(native_path):  # noqa: SLF001
        raise ExecutorError("Stage-C reconstruction native extension drifted")
    for artifact in receipt["artifacts"].values():
        artifact_path = Path(str(artifact["path"])).resolve(strict=True)
        if (
            artifact.get("file_sha256")
            != alignment._file_sha256(artifact_path)  # noqa: SLF001
            or artifact.get("size_bytes") != artifact_path.stat().st_size
        ):
            raise ExecutorError("Stage-C reconstruction artifact bytes drifted")
    return {"path": str(receipt_path), **receipt}


def assert_information_set_safe_search(
    plan: Mapping[str, Any], search: Any
) -> None:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    qualify = commands.add_parser("qualify")
    qualify.add_argument("--plan", required=True, type=Path)
    qualify.add_argument("--output-root", required=True, type=Path)
    qualify.add_argument("--write", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--receipt", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "qualify":
            result = _qualify(args)
            alignment._write_json_immutable(args.write, result)  # noqa: SLF001
        else:
            result = _verify_receipt(args.receipt)
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
