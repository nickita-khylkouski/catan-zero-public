#!/usr/bin/env python3
"""Build the immutable evidence graph consumed by A1 promotion.

The promotion transaction deliberately only *consumes* typed artifacts.  This
module is its producer-side counterpart: it derives high-regret and bucket
results from raw reports, wraps verified sources in promotion-evidence
envelopes, and constructs the final adjudication.  Every output is created
with ``O_EXCL`` and made read-only; an existing artifact is never overwritten.

This tool does not run evaluation and cannot turn a failing result into a
passing one.  Evidence/adjudication commands replay the same validators used by
``a1_promotion_transaction.py`` before publishing their output.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import a1_promotion_transaction as promotion  # noqa: E402
from tools.champion_registry import ChampionRegistry  # noqa: E402
from tools.high_regret_suite_contract import (  # noqa: E402
    REPLAY_CONTRACT,
    SUITE_SCHEMA,
    load_source_validation_binding,
    scope_inventory_sha256,
)


HIGH_REGRET_REPORT_SCHEMA = "a1-held-out-high-regret-report-v1"
HIGH_REGRET_SUITE_SCHEMA = SUITE_SCHEMA
BUCKET_GAME_REPORT_SCHEMA = "a1-bucket-game-report-v1"


class ArtifactBuildError(RuntimeError):
    """Raised when source evidence cannot produce a promotion artifact."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactBuildError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactBuildError(f"{path} must contain a JSON object")
    return value


def _exact(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactBuildError(f"{where} must be an object")
    actual = set(value)
    if actual != keys:
        raise ArtifactBuildError(
            f"{where} keys differ: missing={sorted(keys - actual)} "
            f"unexpected={sorted(actual - keys)}"
        )
    return dict(value)


def _positive_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArtifactBuildError(f"{where} must be a positive integer")
    return value


def _checkpoint_ref(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ArtifactBuildError(f"checkpoint is missing: {path}")
    return {"path": str(path), "sha256": promotion._sha256(path)}  # noqa: SLF001


def _verify_checkpoint_ref(raw: Any, *, expected: Path, where: str) -> dict[str, str]:
    value = _exact(raw, {"path", "sha256"}, where=where)
    expected_ref = _checkpoint_ref(expected)
    actual_path = Path(str(value["path"])).expanduser().resolve()
    if (
        actual_path != Path(expected_ref["path"])
        or value["sha256"] != expected_ref["sha256"]
    ):
        raise ArtifactBuildError(f"{where} does not bind the expected checkpoint bytes")
    return expected_ref


def _file_ref(path: Path, *, where: str) -> dict[str, str]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ArtifactBuildError(f"{where} is missing or is a symlink: {path}")
    return {"path": str(path), "sha256": promotion._sha256(path)}  # noqa: SLF001


def _bound_file_ref(raw: Any, *, base: Path, where: str) -> tuple[Path, dict[str, str]]:
    value = _exact(raw, {"path", "sha256"}, where=where)
    path = Path(str(value["path"])).expanduser()
    if not path.is_absolute():
        path = base / path
    expected = _file_ref(path, where=where)
    if value != expected:
        raise ArtifactBuildError(f"{where} does not bind the referenced file bytes")
    return path.resolve(), expected


def _paired_game_identity(game: Any, *, index: int, where: str) -> tuple[int, str]:
    if not isinstance(game, dict):
        raise ArtifactBuildError(f"{where}[{index}] must be an object")
    pair_id = game.get("pair_id")
    orientation = game.get("orientation")
    if isinstance(pair_id, bool) or not isinstance(pair_id, int) or pair_id < 0:
        raise ArtifactBuildError(f"{where}[{index}].pair_id must be non-negative")
    if not isinstance(orientation, str) or not orientation:
        raise ArtifactBuildError(f"{where}[{index}].orientation must be non-empty")
    return pair_id, orientation


_PAIR_ORIENTATIONS = {
    "legacy": {"candidate_first", "candidate_second"},
    "color": {"candidate_red", "candidate_blue"},
}
_COLOR_ORIENTATION = {
    "candidate_red": ("RED", "BLUE"),
    "candidate_blue": ("BLUE", "RED"),
}


def _paired_orientation_encoding(game: dict[str, Any], *, index: int, where: str) -> str:
    orientation = game["orientation"]
    if orientation in _COLOR_ORIENTATION:
        expected_candidate, expected_baseline = _COLOR_ORIENTATION[orientation]
        if (
            game.get("candidate_color") != expected_candidate
            or game.get("baseline_color") != expected_baseline
        ):
            raise ArtifactBuildError(
                f"{where}[{index}] orientation does not bind candidate/baseline colors"
            )
        return "color"
    if orientation in _PAIR_ORIENTATIONS["legacy"]:
        candidate_color = game.get("candidate_color")
        baseline_color = game.get("baseline_color")
        if (candidate_color is None) != (baseline_color is None):
            raise ArtifactBuildError(
                f"{where}[{index}] has incomplete legacy color fields"
            )
        if candidate_color is not None:
            expected = (
                ("RED", "BLUE")
                if orientation == "candidate_first"
                else ("BLUE", "RED")
            )
            if (candidate_color, baseline_color) != expected:
                raise ArtifactBuildError(
                    f"{where}[{index}] legacy orientation has inconsistent colors"
                )
        return "legacy"
    raise ArtifactBuildError(f"{where}[{index}] has invalid orientation")


def _validated_high_regret_games(
    games: Any, *, where: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Validate raw outcomes and return complete games plus replay input."""

    if not isinstance(games, list) or not games:
        raise ArtifactBuildError(f"{where} must be a non-empty list")
    identities: set[tuple[int, str]] = set()
    by_pair: dict[int, dict[str, dict[str, Any]]] = {}
    encoding: str | None = None
    for index, game in enumerate(games):
        identity = _paired_game_identity(game, index=index, where=where)
        pair_id, orientation = identity
        if identity in identities:
            raise ArtifactBuildError(f"{where} contains duplicate games")
        game_encoding = _paired_orientation_encoding(game, index=index, where=where)
        if encoding is not None and game_encoding != encoding:
            raise ArtifactBuildError(f"{where} mixes orientation encodings")
        encoding = game_encoding
        truncated = game.get("truncated")
        outcome = game.get("candidate_won")
        if not isinstance(truncated, bool):
            raise ArtifactBuildError(f"{where}[{index}].truncated must be boolean")
        if truncated:
            if outcome is not None:
                raise ArtifactBuildError(
                    f"{where}[{index}] truncated game must have candidate_won=null"
                )
        elif not isinstance(outcome, bool):
            raise ArtifactBuildError(
                f"{where}[{index}] nontruncated game must have boolean candidate_won"
            )
        identities.add(identity)
        by_pair.setdefault(pair_id, {})[orientation] = game

    incomplete_pairs: set[int] = set()
    for pair_id, pair_games in by_pair.items():
        if encoding is None or set(pair_games) != _PAIR_ORIENTATIONS[encoding]:
            raise ArtifactBuildError(
                f"{where} pair {pair_id} must contain both orientations"
            )
        if any(game["truncated"] for game in pair_games.values()):
            incomplete_pairs.add(pair_id)
    complete_games = [game for game in games if game["pair_id"] not in incomplete_pairs]
    normalized_games = [{**game, "search_won": game["candidate_won"]} for game in games]
    return complete_games, normalized_games, len(incomplete_pairs)


def _validate_high_regret_evaluation_config(raw: Any, *, where: str) -> None:
    if not isinstance(raw, dict):
        raise ArtifactBuildError(f"{where} must be an object")
    expected = {
        "candidate_n_full": 128,
        "baseline_n_full": 128,
        "p_full": 1.0,
        "force_full_every_decision": True,
    }
    for key, value in expected.items():
        if raw.get(key) != value or type(raw.get(key)) is not type(value):
            raise ArtifactBuildError(
                f"{where} has inconsistent {key}={raw.get(key)!r}, expected {value!r}"
            )
    for key in ("c_scale", "candidate_c_scale", "baseline_c_scale"):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ArtifactBuildError(f"{where} has invalid {key}={value!r}")
    if float(raw["c_scale"]) != float(raw["candidate_c_scale"]):
        raise ArtifactBuildError(f"{where} c_scale must echo candidate_c_scale")


def _write_new_readonly(path: Path, value: dict[str, Any]) -> None:
    path = Path(os.path.abspath(os.fspath(path.expanduser())))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ArtifactBuildError(f"output must be a fresh non-symlink path: {path}")
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def build_high_regret_source(
    *, report_path: Path, candidate: Path, champion: Path
) -> dict[str, Any]:
    """Derive the transaction's high-regret source from a canonical raw report."""
    report_path = report_path.expanduser().resolve()
    raw = _exact(
        _load_json(report_path),
        {
            "schema_version",
            "suite",
            "held_out",
            "candidate",
            "champion",
            "errors",
            "games",
            "suite_manifest",
            "pentanomial_sprt",
            "pair_diagnostics",
            "evaluation_config",
            "planned_engine_identity",
            "engine_identity",
            "archived_state_reconstruction",
        },
        where="high-regret report",
    )
    if raw["schema_version"] != HIGH_REGRET_REPORT_SCHEMA:
        raise ArtifactBuildError(
            f"high-regret report schema must be {HIGH_REGRET_REPORT_SCHEMA!r}"
        )
    if raw["suite"] != "held_out_high_regret" or raw["held_out"] is not True:
        raise ArtifactBuildError("high-regret report is not the held-out suite")
    candidate_ref = _verify_checkpoint_ref(
        raw["candidate"], expected=candidate, where="high-regret report.candidate"
    )
    champion_ref = _verify_checkpoint_ref(
        raw["champion"], expected=champion, where="high-regret report.champion"
    )
    if raw["errors"] != []:
        raise ArtifactBuildError("high-regret report contains evaluation errors")
    _validate_high_regret_evaluation_config(
        raw["evaluation_config"], where="high-regret report.evaluation_config"
    )
    _suite_path, suite_ref = _bound_file_ref(
        raw["suite_manifest"],
        base=report_path.parent,
        where="high-regret report.suite_manifest",
    )
    _complete_games, normalized_games, truncated_pairs = _validated_high_regret_games(
        raw["games"], where="high-regret report.games"
    )
    pair_scores, diagnostics = promotion.pair_scores_from_h2h_games(normalized_games)
    pentanomial = promotion.evaluate_pentanomial_sprt(
        pair_scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    complete_pairs = sum(
        diagnostics[key] for key in ("ww_pairs", "split_pairs", "ll_pairs")
    )
    _positive_int(complete_pairs, where="high-regret report complete pairs")
    if diagnostics["incomplete_pairs"] != truncated_pairs:
        raise ArtifactBuildError("high-regret report truncation diagnostics do not replay")
    if raw["pair_diagnostics"] != diagnostics or raw["pentanomial_sprt"] != pentanomial:
        raise ArtifactBuildError("high-regret report paired statistics do not replay")
    if pentanomial["decision"] != "H1":
        raise ArtifactBuildError("high-regret report has no passing paired verdict")
    return {
        "schema_version": promotion.HIGH_REGRET_SCHEMA,
        "suite": "held_out_high_regret",
        "held_out": True,
        "candidate": candidate_ref,
        "champion": champion_ref,
        "passed": True,
        "verdict": "H1",
        "complete_pairs": complete_pairs,
        "errors": [],
        "report": _file_ref(report_path, where="high-regret report"),
        "suite_manifest": suite_ref,
        "pentanomial_sprt": pentanomial,
        "pair_diagnostics": diagnostics,
    }


def build_held_out_high_regret_suite(
    *,
    manifest_path: Path,
    holdout_fraction: float,
    holdout_seed: int,
    pairs: int,
) -> dict[str, Any]:
    """Materialize a deterministic frozen suite from the regret archive.

    Selection deliberately reuses the restart generator's stable
    ``(game_seed, decision_index, holdout_seed)`` partition.  Within the held
    out partition, states are ordered by descending regret then stable identity;
    no RNG or operator-curated row list is accepted.
    """

    import numpy as np

    manifest_path = manifest_path.expanduser().resolve()
    try:
        held_out_seeds, validation_binding = load_source_validation_binding(
            manifest_path
        )
    except ValueError as error:
        raise ArtifactBuildError(str(error)) from error
    if holdout_fraction != 1.0:
        raise ArtifactBuildError(
            "promotion suite must use the full authenticated trainer validation set"
        )
    pairs = _positive_int(pairs, where="held-out suite pairs")
    if pairs < 20:
        raise ArtifactBuildError("held-out suite requires at least 20 pairs")
    try:
        with np.load(manifest_path, allow_pickle=False) as data:
            required = {
                "shard_id",
                "row_index",
                "game_seed",
                "decision_index",
                "regret_score",
                "phase",
                "shard_paths",
            }
            if not required.issubset(data.files):
                raise ArtifactBuildError(
                    f"regret manifest lacks fields {sorted(required - set(data.files))}"
                )
            game_seeds = np.asarray(data["game_seed"]).reshape(-1)
            decisions = np.asarray(data["decision_index"]).reshape(-1)
            scores = np.asarray(data["regret_score"], dtype=np.float64).reshape(-1)
            phases = np.asarray(data["phase"]).astype(str).reshape(-1)
            shard_ids = np.asarray(data["shard_id"]).reshape(-1)
            row_indices = np.asarray(data["row_index"]).reshape(-1)
            legal_counts = (
                np.asarray(data["legal_count"]).reshape(-1)
                if "legal_count" in data.files
                else np.zeros(game_seeds.shape, dtype=np.int64)
            )
            shard_paths = [str(item) for item in np.asarray(data["shard_paths"])]
    except ArtifactBuildError:
        raise
    except (OSError, ValueError) as error:
        raise ArtifactBuildError(f"cannot load regret manifest: {error}") from error
    lengths = {
        len(game_seeds),
        len(decisions),
        len(scores),
        len(phases),
        len(shard_ids),
        len(row_indices),
        len(legal_counts),
    }
    if len(lengths) != 1 or not game_seeds.size:
        raise ArtifactBuildError("regret manifest columns are empty or misaligned")
    leaked = set(map(int, game_seeds)) - held_out_seeds
    if leaked:
        raise ArtifactBuildError(
            f"regret manifest contains {len(leaked)} non-validation game seeds"
        )

    # The source manifest is already restricted to the trainer-authenticated
    # game-level holdout. A second per-state hash split both wastes power and
    # can cluster the retained states into too few source games.
    eligible = list(range(len(game_seeds)))
    eligible.sort(
        key=lambda index: (
            -float(scores[index]),
            int(game_seeds[index]),
            int(decisions[index]),
            int(shard_ids[index]),
            int(row_indices[index]),
        )
    )
    eligible_unique_states = len(
        {(int(game_seeds[index]), int(decisions[index])) for index in eligible}
    )
    eligible_unique_games = len({int(game_seeds[index]) for index in eligible})

    replay_complete, replay_stats, scope_inventories = _replay_complete_manifest_rows(
        manifest_path=manifest_path,
        candidate_indices=eligible,
        shard_paths=shard_paths,
        shard_ids=shard_ids,
        row_indices=row_indices,
        game_seeds=game_seeds,
        decisions=decisions,
    )
    eligible = [index for index in eligible if index in replay_complete]
    replay_complete_unique_games = len(
        {int(game_seeds[index]) for index in eligible}
    )

    def phase_stratum(phase: str) -> str:
        upper = str(phase).upper()
        if "BUILD_INITIAL_SETTLEMENT" in upper or "BUILD_INITIAL_ROAD" in upper:
            return "opening"
        if "ROBBER" in upper or "KNIGHT" in upper or "DEVELOPMENT_CARD" in upper:
            return "robber_dev"
        if "DISCARD" in upper or "ROLL" in upper:
            return "chance"
        return "build_trade"

    selected: list[int] = []
    seen_game_seeds: set[int] = set()
    selected_by_stratum: dict[str, int] = {}

    def select_from(indices: Sequence[int], want: int, *, label: str) -> None:
        before = len(selected)
        for index in indices:
            game_seed = int(game_seeds[index])
            if game_seed in seen_game_seeds:
                continue
            seen_game_seeds.add(game_seed)
            selected.append(index)
            if len(selected) - before == want:
                break
        selected_by_stratum[label] = len(selected) - before
        if selected_by_stratum[label] != want:
            raise ArtifactBuildError(
                f"held-out partition cannot fill required {label!r} stratum: "
                f"{selected_by_stratum[label]} < {want} after replay-completeness "
                f"preflight ({replay_stats})"
            )

    stratum_min_pairs = max(4, pairs // 10)
    for stratum in ("opening", "robber_dev", "chance", "build_trade"):
        select_from(
            [index for index in eligible if phase_stratum(phases[index]) == stratum],
            stratum_min_pairs,
            label=f"phase:{stratum}",
        )
    select_from(
        [index for index in eligible if int(legal_counts[index]) >= 41],
        stratum_min_pairs,
        label="41+",
    )
    if len(selected) < pairs:
        for index in eligible:
            game_seed = int(game_seeds[index])
            if game_seed in seen_game_seeds:
                continue
            seen_game_seeds.add(game_seed)
            selected.append(index)
            if len(selected) == pairs:
                break
    if len(selected) != pairs:
        raise ArtifactBuildError(
            f"held-out partition has only {len(selected)} replay-complete unique "
            f"independent source games, need {pairs} ({replay_stats})"
        )
    states: list[dict[str, Any]] = []
    for pair_id, index in enumerate(selected):
        shard_id = int(shard_ids[index])
        if shard_id < 0 or shard_id >= len(shard_paths):
            raise ArtifactBuildError(
                f"regret manifest shard_id out of range: {shard_id}"
            )
        shard_path = Path(shard_paths[shard_id]).expanduser()
        if not shard_path.is_absolute():
            shard_path = manifest_path.parent / shard_path
        states.append(
            {
                "pair_id": pair_id,
                "shard_path": str(shard_path.resolve()),
                "shard_id": shard_id,
                "row_index": int(row_indices[index]),
                "game_seed": int(game_seeds[index]),
                "decision_index": int(decisions[index]),
                "phase": str(phases[index]),
                "legal_count": int(legal_counts[index]),
                "regret_score": float(scores[index]),
                "replay_source": {
                    "contract": REPLAY_CONTRACT,
                    "scope": str(shard_path.resolve().parent),
                    "scope_inventory_sha256": scope_inventories[
                        shard_path.resolve().parent
                    ][0],
                    "scope_shard_count": scope_inventories[
                        shard_path.resolve().parent
                    ][1],
                },
            }
        )
    value = {
        "schema_version": HIGH_REGRET_SUITE_SCHEMA,
        "suite": "held_out_high_regret",
        "held_out": True,
        "source_manifest": _file_ref(manifest_path, where="regret manifest"),
        "validation_seed_manifest": validation_binding,
        "selection": {
            "algorithm": "trainer-validation-stratified-regret-unique-game-v3",
            "selection_scope": "full_authenticated_training_validation_manifest",
            "holdout_fraction": float(holdout_fraction),
            "holdout_seed": int(holdout_seed),
            "eligible_unique_states": eligible_unique_states,
            "eligible_unique_games": eligible_unique_games,
            "replay_complete_unique_games": replay_complete_unique_games,
            "selected_unique_games": len({state["game_seed"] for state in states}),
            "selected_pairs": pairs,
            "stratum_min_pairs": stratum_min_pairs,
            "selected_by_stratum": selected_by_stratum,
            "replay_preflight": replay_stats,
        },
        "states": states,
    }
    value["suite_sha256"] = promotion._digest_value(value)  # noqa: SLF001
    return value


def _replay_complete_manifest_rows(
    *,
    manifest_path: Path,
    candidate_indices: Sequence[int],
    shard_paths: Sequence[str],
    shard_ids: Any,
    row_indices: Any,
    game_seeds: Any,
    decisions: Any,
) -> tuple[set[int], dict[str, Any], dict[Path, tuple[str, int]]]:
    """Return candidates with an authoritative, replay-complete prefix.

    The evaluator reconstructs a state by scanning every shard below the
    selected shard's parent directory.  Mirror that contract before sealing:
    the manifest row must bind to the stated source row and its source scope
    must contain each decision exactly once from zero through its final row,
    including the target. Missing, negative, duplicate, or role-filtered
    partial trajectories fail closed exactly as evaluator replay would.
    """

    import numpy as np

    from regret_common import discover_shards, load_shard

    requested_by_scope: dict[Path, dict[int, int]] = {}
    source_paths: dict[int, Path] = {}
    rejected_bad_source = 0
    for index in candidate_indices:
        shard_id = int(shard_ids[index])
        if shard_id < 0 or shard_id >= len(shard_paths):
            rejected_bad_source += 1
            continue
        source = Path(shard_paths[shard_id]).expanduser()
        if not source.is_absolute():
            source = manifest_path.parent / source
        source = source.resolve()
        source_paths[index] = source
        seed = int(game_seeds[index])
        target = int(decisions[index])
        if target < 0:
            rejected_bad_source += 1
            source_paths.pop(index, None)
            continue
        scope_targets = requested_by_scope.setdefault(source.parent, {})
        scope_targets[seed] = max(scope_targets.get(seed, -1), target)

    # Count decision occurrences only for requested games. Checking the whole
    # recorded trajectory mirrors gather_game_action_sequence exactly while
    # remaining streaming over shard files.
    counts_by_scope: dict[Path, dict[int, dict[int, int]]] = {}
    malformed_seeds_by_scope: dict[Path, set[int]] = {}
    source_arrays: dict[Path, dict[str, Any] | None] = {}
    selected_source_paths = set(source_paths.values())
    scope_inventories: dict[Path, tuple[str, int]] = {}
    for scope, targets in requested_by_scope.items():
        try:
            inventory_before = scope_inventory_sha256(scope)
        except ValueError as error:
            raise ArtifactBuildError(str(error)) from error
        seed_counts: dict[int, dict[int, int]] = {seed: {} for seed in targets}
        malformed: set[int] = set()
        for shard_path in discover_shards([scope]):
            try:
                shard = load_shard(shard_path)
            except (OSError, ValueError):
                # An unreadable shard makes this authoritative scope unsafe.
                malformed.update(targets)
                continue
            if shard_path.resolve() in selected_source_paths:
                source_arrays[shard_path.resolve()] = shard
            if "game_seed" not in shard:
                # gather_game_action_sequence indexes game_seed in every shard
                # discovered below the scope, even when that shard would not
                # have contained the requested game.
                malformed.update(targets)
                continue
            seeds = np.asarray(shard["game_seed"]).reshape(-1)
            if "decision_index" not in shard or "action_taken" not in shard:
                for seed in set(int(value) for value in seeds if int(value) in targets):
                    malformed.add(seed)
                continue
            didx = np.asarray(shard["decision_index"]).reshape(-1)
            actions = np.asarray(shard["action_taken"]).reshape(-1)
            phase = shard.get("phase")
            player = shard.get("player")
            if (
                len(seeds) != len(didx)
                or len(actions) != len(seeds)
                or (phase is not None and len(np.asarray(phase).reshape(-1)) < len(seeds))
                or (player is not None and len(np.asarray(player).reshape(-1)) < len(seeds))
            ):
                for seed in set(int(value) for value in seeds if int(value) in targets):
                    malformed.add(seed)
                continue
            for row in range(len(seeds)):
                seed = int(seeds[row])
                if seed not in targets:
                    continue
                decision = int(didx[row])
                if decision < 0:
                    malformed.add(seed)
                    continue
                per_seed = seed_counts[seed]
                per_seed[decision] = per_seed.get(decision, 0) + 1
        counts_by_scope[scope] = seed_counts
        malformed_seeds_by_scope[scope] = malformed
        try:
            inventory_after = scope_inventory_sha256(scope)
        except ValueError as error:
            raise ArtifactBuildError(str(error)) from error
        if inventory_before != inventory_after:
            raise ArtifactBuildError(
                f"held-out replay scope changed during preflight: {scope}"
            )
        scope_inventories[scope] = inventory_after

    complete: set[int] = set()
    rejected_noncontiguous = 0
    for index, source in source_paths.items():
        seed = int(game_seeds[index])
        target = int(decisions[index])
        shard = source_arrays.get(source)
        row = int(row_indices[index])
        source_bound = False
        if shard is not None and row >= 0:
            try:
                source_seeds = np.asarray(shard["game_seed"]).reshape(-1)
                source_decisions = np.asarray(shard["decision_index"]).reshape(-1)
                source_actions = np.asarray(shard["action_taken"]).reshape(-1)
                source_bound = (
                    row < len(source_seeds)
                    and row < len(source_decisions)
                    and row < len(source_actions)
                    and int(source_seeds[row]) == seed
                    and int(source_decisions[row]) == target
                )
            except (KeyError, TypeError, ValueError):
                source_bound = False
        scope = source.parent
        counts = counts_by_scope.get(scope, {}).get(seed, {})
        max_recorded = max(counts, default=-1)
        contiguous = (
            seed not in malformed_seeds_by_scope.get(scope, set())
            and max_recorded >= target
            and all(counts.get(decision) == 1 for decision in range(max_recorded + 1))
        )
        if source_bound and contiguous:
            complete.add(index)
        elif not source_bound:
            rejected_bad_source += 1
        else:
            rejected_noncontiguous += 1

    return (
        complete,
        {
            "contract": REPLAY_CONTRACT,
            "candidate_states": len(candidate_indices),
            "replay_complete_states": len(complete),
            "rejected_bad_source": rejected_bad_source,
            "rejected_noncontiguous": rejected_noncontiguous,
        },
        scope_inventories,
    )


def build_bucket_game_report(
    *, report_path: Path, candidate: Path, champion: Path
) -> dict[str, Any]:
    """Extract bucket-labelled outcomes from retained evaluator games."""

    report_path = report_path.expanduser().resolve()
    raw = _exact(
        _load_json(report_path),
        {
            "schema_version",
            "suite",
            "held_out",
            "suite_manifest",
            "candidate",
            "champion",
            "evaluation_config",
            "errors",
            "games",
            "pentanomial_sprt",
            "pair_diagnostics",
            "planned_engine_identity",
            "engine_identity",
            "archived_state_reconstruction",
        },
        where="high-regret evaluation report",
    )
    if raw["schema_version"] != HIGH_REGRET_REPORT_SCHEMA or raw["errors"] != []:
        raise ArtifactBuildError(
            "bucket extraction requires a clean high-regret report"
        )
    _validate_high_regret_evaluation_config(
        raw["evaluation_config"], where="bucket report.evaluation_config"
    )
    candidate_ref = _verify_checkpoint_ref(
        raw["candidate"], expected=candidate, where="bucket report.candidate"
    )
    champion_ref = _verify_checkpoint_ref(
        raw["champion"], expected=champion, where="bucket report.champion"
    )
    games, normalized_games, truncated_pairs = _validated_high_regret_games(
        raw["games"], where="report.games"
    )
    pair_scores, diagnostics = promotion.pair_scores_from_h2h_games(normalized_games)
    pentanomial = promotion.evaluate_pentanomial_sprt(
        pair_scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    if (
        diagnostics["incomplete_pairs"] != truncated_pairs
        or raw["pair_diagnostics"] != diagnostics
        or raw["pentanomial_sprt"] != pentanomial
        or pentanomial["decision"] != "H1"
    ):
        raise ArtifactBuildError("bucket extraction paired statistics do not replay")
    projected: list[dict[str, Any]] = []
    for index, game in enumerate(games):
        identity = _paired_game_identity(game, index=index, where="report.games")
        outcome = game.get("candidate_won")
        labels = game.get("buckets")
        if (
            not isinstance(labels, list)
            or not labels
            or not all(isinstance(label, str) and label for label in labels)
            or len(labels) != len(set(labels))
        ):
            raise ArtifactBuildError(f"report.games[{index}] has invalid bucket labels")
        projected.append(
            {
                "pair_id": identity[0],
                "orientation": identity[1],
                "candidate_won": outcome,
                "buckets": sorted(labels),
                **(
                    {
                        "candidate_color": game["candidate_color"],
                        "baseline_color": game["baseline_color"],
                    }
                    if identity[1] in _COLOR_ORIENTATION
                    else {}
                ),
            }
        )
    return {
        "schema_version": BUCKET_GAME_REPORT_SCHEMA,
        "candidate": candidate_ref,
        "champion": champion_ref,
        "errors": [],
        "games": projected,
    }


def build_bucket_veto_source(
    *, report_path: Path, candidate: Path, champion: Path
) -> dict[str, Any]:
    """Replay per-bucket outcomes from immutable, game-level evaluation data."""
    report_path = report_path.expanduser().resolve()
    raw = _exact(
        _load_json(report_path),
        {"schema_version", "candidate", "champion", "errors", "games"},
        where="bucket game report",
    )
    if raw["schema_version"] != BUCKET_GAME_REPORT_SCHEMA:
        raise ArtifactBuildError(
            f"bucket game report schema must be {BUCKET_GAME_REPORT_SCHEMA!r}"
        )
    candidate_ref = _verify_checkpoint_ref(
        raw["candidate"], expected=candidate, where="bucket game report.candidate"
    )
    champion_ref = _verify_checkpoint_ref(
        raw["champion"], expected=champion, where="bucket game report.champion"
    )
    if raw["errors"] != []:
        raise ArtifactBuildError("bucket game report contains evaluation errors")
    games = raw["games"]
    if not isinstance(games, list) or not games:
        raise ArtifactBuildError("bucket game report.games must be a non-empty list")
    identities: set[tuple[int, str]] = set()
    counts: dict[str, list[int]] = {}
    for index, game in enumerate(games):
        identity = _paired_game_identity(
            game, index=index, where="bucket game report.games"
        )
        outcome = game.get("candidate_won")
        labels = game.get("buckets")
        if identity in identities:
            raise ArtifactBuildError("bucket game report contains duplicate games")
        if not isinstance(outcome, bool):
            raise ArtifactBuildError(
                f"bucket game report.games[{index}].candidate_won must be boolean"
            )
        if (
            not isinstance(labels, list)
            or not labels
            or not all(isinstance(label, str) and label for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ArtifactBuildError(
                f"bucket game report.games[{index}].buckets is invalid"
            )
        identities.add(identity)
        for label in labels:
            bucket_counts = counts.setdefault(label, [0, 0])
            bucket_counts[0 if outcome else 1] += 1
    per_bucket: dict[str, dict[str, Any]] = {}
    veto_buckets: list[str] = []
    for label, (wins, losses) in sorted(counts.items()):
        count = wins + losses
        winrate = wins / count
        status = (
            "insufficient_data"
            if count < promotion.MIN_BUCKET_GAMES
            else "pass"
            if winrate >= promotion.MIN_BUCKET_WIN_RATE
            else "fail"
        )
        per_bucket[label] = {"status": status, "n": count, "winrate": winrate}
        if status == "fail":
            veto_buckets.append(label)
    return {
        "schema_version": promotion.BUCKET_VETO_SCHEMA,
        "candidate": candidate_ref,
        "champion": champion_ref,
        "veto": bool(veto_buckets),
        "veto_buckets": veto_buckets,
        "per_bucket": per_bucket,
        "report": _file_ref(report_path, where="bucket game report"),
    }


def build_legacy_incumbent_calibration_source(
    *,
    calibration_path: Path,
    historical_training_report: Path,
    contract: dict[str, Any],
    champion: Path,
) -> dict[str, Any]:
    """Attach the only permitted provenance bridge to a legacy incumbent.

    The numerical calibration payload remains untouched.  The bridge only
    supplies missing training provenance and is valid exclusively when both
    the sealed contract and the immutable historical report bind the same
    incumbent bytes.
    """

    calibration_path = calibration_path.expanduser().resolve()
    champion = champion.expanduser().resolve()
    value = _load_json(calibration_path)
    if value.get("schema_version") != "phase-sliced-value-calibration-v2":
        raise ArtifactBuildError("calibration is not phase-sliced-value-calibration-v2")
    if Path(str(value.get("checkpoint"))).expanduser().resolve() != champion:
        raise ArtifactBuildError("calibration does not bind the incumbent checkpoint")
    if value.get("value_readout") != "scalar":
        raise ArtifactBuildError("legacy incumbent bridge is scalar-only")
    provenance = value.get("readout_provenance")
    if not isinstance(provenance, dict):
        raise ArtifactBuildError("calibration has no readout_provenance")
    if (
        provenance.get("optimizer_steps") is not None
        or provenance.get("completed_epochs") is not None
    ):
        raise ArtifactBuildError("calibration already has native training provenance")
    if "legacy_incumbent_provenance" in value:
        raise ArtifactBuildError("calibration already has a legacy incumbent bridge")
    champion_ref = _checkpoint_ref(champion)
    producers = [
        item
        for item in contract.get("checkpoints", [])
        if isinstance(item, dict) and item.get("role") == "producer"
    ]
    if len(producers) != 1:
        raise ArtifactBuildError("contract has no unique producer checkpoint")
    producer = producers[0]
    if (
        Path(str(producer.get("path"))).expanduser().resolve() != champion
        or producer.get("sha256") != champion_ref["sha256"]
    ):
        raise ArtifactBuildError("champion is not the contract-bound producer")
    report_path = historical_training_report.expanduser().resolve()
    historical = _load_json(report_path)
    try:
        promotion._historical_checkpoint_path(  # noqa: SLF001
            historical.get("checkpoint"),
            report_path=report_path,
            checkpoint=champion,
            where="historical report checkpoint",
        )
    except promotion.PromotionError as error:
        raise ArtifactBuildError(str(error)) from error
    _positive_int(
        historical.get("steps_completed"), where="historical report.steps_completed"
    )
    _positive_int(historical.get("epochs"), where="historical report.epochs")
    if (
        historical.get("checkpoint_sha256") is not None
        and historical["checkpoint_sha256"] != champion_ref["sha256"]
    ):
        raise ArtifactBuildError("historical report checkpoint hash mismatch")
    return {
        **value,
        "legacy_incumbent_provenance": {
            "schema_version": promotion.LEGACY_INCUMBENT_PROVENANCE_SCHEMA,
            "contract_sha256": contract["contract_sha256"],
            "checkpoint_sha256": champion_ref["sha256"],
            "historical_training_report": _file_ref(
                report_path, where="historical training report"
            ),
        },
    }


def _source_ref(role: str, path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ArtifactBuildError(f"source {role!r} is missing: {path}")
    if not role:
        raise ArtifactBuildError("source role must be non-empty")
    return {"role": role, "path": str(path), "sha256": promotion._sha256(path)}  # noqa: SLF001


def build_evidence_envelope(
    *,
    kind: str,
    contract: dict[str, Any],
    candidate: Path,
    champion: Path,
    sources: Sequence[tuple[str, Path]],
    promotion_mode: str = "promotion_parent",
) -> dict[str, Any]:
    if kind not in promotion.REQUIRED_EVIDENCE_KINDS:
        raise ArtifactBuildError(f"unsupported evidence kind {kind!r}")
    roles = [role for role, _path in sources]
    if len(set(roles)) != len(roles):
        raise ArtifactBuildError("evidence source roles must be unique")
    if promotion_mode not in {"promotion_parent", "branch_challenge"}:
        raise ArtifactBuildError(f"unsupported promotion mode {promotion_mode!r}")
    verdict = "H1" if kind == "internal_h2h" else "pass"
    result: dict[str, Any]
    if kind == "mechanism_calibration":
        result = {
            "value_readout": promotion._contract_value_readout(contract),  # noqa: SLF001
            "max_rmse_regression": promotion.MAX_CALIBRATION_RMSE_REGRESSION,
        }
    elif kind == "internal_h2h":
        result = dict(promotion.INTERNAL_STRENGTH_RESULT)
    elif kind == "external_panel":
        result = {"max_win_rate_regression": promotion.MAX_EXTERNAL_WIN_RATE_REGRESSION}
    elif kind == "internal_h2h" and promotion_mode == "branch_challenge":
        if set(roles) != {"internal_h2h_cohort_1", "internal_h2h_cohort_2"}:
            raise ArtifactBuildError(
                "branch challenge requires two fresh internal H2H cohort roles"
            )
        result = {"required_fresh_cohorts": 2, "strict_superiority": True}
    else:
        result = {}
    value = {
        "schema_version": promotion.EVIDENCE_SCHEMA,
        "kind": kind,
        "passed": True,
        "verdict": verdict,
        "contract_sha256": contract["contract_sha256"],
        "candidate": _checkpoint_ref(candidate),
        "champion": _checkpoint_ref(champion),
        "sources": [_source_ref(role, path) for role, path in sources],
        "result": result,
    }
    value["evidence_sha256"] = promotion._digest_value(value)  # noqa: SLF001
    return value


def _validate_envelope_before_write(
    path: Path,
    *,
    value: dict[str, Any],
    kind: str,
    contract: dict[str, Any],
    candidate: Path,
    champion: Path,
    registry: ChampionRegistry,
    promotion_mode: str = "promotion_parent",
    candidate_parent: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".verify", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        candidate_search_config = promotion._candidate_search_config(contract)  # noqa: SLF001
        champion_ref = _checkpoint_ref(champion)
        champion_search_config = promotion._incumbent_search_config(  # noqa: SLF001
            contract,
            registry=registry,
            champion_path=champion.expanduser().resolve(),
            champion_sha256=champion_ref["sha256"],
        )
        promotion._verify_promotion_evidence(  # noqa: SLF001
            temporary,
            kind=kind,
            contract=contract,
            expected_readout=promotion._contract_value_readout(contract),  # noqa: SLF001
            candidate={
                **_checkpoint_ref(candidate),
                "md5": promotion._md5(candidate),  # noqa: SLF001
                "search_config": candidate_search_config,
            },
            champion={
                **_checkpoint_ref(champion),
                "md5": promotion._md5(champion),  # noqa: SLF001
                "search_config": champion_search_config,
            },
            promotion_mode=promotion_mode,
            candidate_parent=(
                None
                if candidate_parent is None
                else _checkpoint_ref(candidate_parent)
            ),
        )
    except promotion.PromotionError as error:
        raise ArtifactBuildError(
            f"evidence does not pass transaction replay: {error}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def build_adjudication(
    *,
    contract: dict[str, Any],
    contract_lock: Path,
    training_receipt: Path,
    registry: ChampionRegistry,
    current_pointer: Path,
    candidate: Path,
    candidate_version: int,
    training_report: Path,
    champion: Path,
    champion_version: int,
    evidence: Sequence[tuple[str, Path]],
    nth_confirmation: Path | None,
    promotion_mode: str = "promotion_parent",
) -> dict[str, Any]:
    if promotion_mode not in {"promotion_parent", "branch_challenge"}:
        raise ArtifactBuildError(f"unsupported promotion mode {promotion_mode!r}")
    by_kind = dict(evidence)
    if (
        len(by_kind) != len(evidence)
        or set(by_kind) != promotion.REQUIRED_EVIDENCE_KINDS
    ):
        raise ArtifactBuildError(
            "adjudication requires each promotion evidence kind exactly once"
        )
    next_count = registry.promotion_count("generator_champion") + 1
    nth_required = next_count % 3 == 0
    if nth_required and nth_confirmation is None:
        raise ArtifactBuildError(
            "this promotion requires an immutable n64 confirmation artifact"
        )
    if not nth_required and nth_confirmation is not None:
        raise ArtifactBuildError(
            "n64 confirmation was supplied for a non-third promotion"
        )
    candidate_search_config = promotion._candidate_search_config(contract)  # noqa: SLF001
    champion_search_config = promotion._incumbent_search_config(  # noqa: SLF001
        contract,
        registry=registry,
        champion_path=champion.expanduser().resolve(),
        champion_sha256=_checkpoint_ref(champion)["sha256"],
    )
    candidate_ref = _checkpoint_ref(candidate)
    champion_ref = _checkpoint_ref(champion)
    value = {
        "schema_version": (
            promotion.BRANCH_CHALLENGE_ADJUDICATION_SCHEMA
            if promotion_mode == "branch_challenge"
            else promotion.ADJUDICATION_SCHEMA
        ),
        "passed": True,
        "decision": "promote",
        "contract_sha256": contract["contract_sha256"],
        "candidate": {
            **candidate_ref,
            "version": candidate_version,
            "agent_identity": promotion._agent_identity(  # noqa: SLF001
                candidate_ref,
                candidate_search_config,
            ),
            "training_report": {
                "path": str(training_report.expanduser().resolve()),
                "sha256": promotion._sha256(training_report),  # noqa: SLF001
            },
        },
        "champion": {
            **champion_ref,
            "version": champion_version,
            "agent_identity": promotion._agent_identity(  # noqa: SLF001
                champion_ref,
                champion_search_config,
            ),
        },
        "checks": {name: True for name in promotion.REQUIRED_CHECKS},
        "nth_confirmation_required": nth_required,
        "nth_confirmation": (
            None
            if nth_confirmation is None
            else {
                "path": str(nth_confirmation.expanduser().resolve()),
                "sha256": promotion._sha256(nth_confirmation),  # noqa: SLF001
            }
        ),
        "evidence": [
            {
                "kind": kind,
                "path": str(by_kind[kind].expanduser().resolve()),
                "sha256": promotion._sha256(by_kind[kind]),  # noqa: SLF001
            }
            for kind in sorted(by_kind)
        ],
    }
    if promotion_mode == "branch_challenge":
        report_path = training_report.expanduser().resolve()
        report = _load_json(report_path)
        init_path_raw = report.get("init_checkpoint")
        init_sha = report.get("init_checkpoint_sha256")
        if not isinstance(init_path_raw, str) or not init_path_raw:
            raise ArtifactBuildError(
                "branch challenge training report has no initializer path"
            )
        init_path = Path(init_path_raw).expanduser()
        if not init_path.is_absolute():
            init_path = report_path.parent / init_path
        init_ref = _checkpoint_ref(init_path)
        if init_ref["sha256"] != init_sha:
            raise ArtifactBuildError(
                "branch challenge training report initializer hash drift"
            )
        if init_ref["sha256"] == champion_ref["sha256"]:
            raise ArtifactBuildError(
                "branch challenge initializer must differ from displaced incumbent"
            )
        value["promotion_mode"] = "branch_challenge"
        value["candidate_lineage"] = {
            "schema_version": promotion.BRANCH_CHALLENGE_LINEAGE_SCHEMA,
            "initializer": init_ref,
            "displaced_incumbent": {
                **champion_ref,
                "version": champion_version,
                "agent_identity_sha256": value["champion"]["agent_identity"][
                    "agent_identity_sha256"
                ],
            },
        }
    value["adjudication_sha256"] = promotion._digest_value(value)  # noqa: SLF001
    return value


def _parse_role_paths(values: Sequence[str], *, option: str) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for raw in values:
        role, separator, path = raw.partition("=")
        if not separator or not role or not path:
            raise ArtifactBuildError(f"{option} entries must be ROLE=PATH")
        parsed.append((role, Path(path)))
    return parsed


def build_cohort_exclusions(
    *,
    contract: dict[str, Any],
    candidate: Path,
    cohorts: Sequence[tuple[str, str, Path]],
) -> dict[str, Any]:
    """Build exclusions from explicit seed identities in prior reports.

    Callers do not provide ranges: they provide immutable source reports and
    this producer derives the minimal contiguous intervals from retained game
    seeds.  That prevents a typo (or a conveniently narrow hand-authored
    range) from weakening the freshness check.
    """

    candidate_ref = _checkpoint_ref(candidate)
    if not cohorts:
        raise ArtifactBuildError("cohort exclusions require at least one prior source")
    labels: set[str] = set()
    records: list[dict[str, Any]] = []
    for label, kind, source in cohorts:
        if not label.strip() or label in labels or not kind.strip():
            raise ArtifactBuildError("cohort label/kind is empty or duplicated")
        labels.add(label)
        source = source.expanduser().resolve()
        payload = _load_json(source)
        bound_hashes = {
            payload.get("candidate_checkpoint_sha256"),
            (
                payload.get("candidate", {}).get("sha256")
                if isinstance(payload.get("candidate"), dict)
                else None
            ),
        }
        explicit_hashes = [value for value in bound_hashes if value is not None]
        if not explicit_hashes:
            raise ArtifactBuildError(
                f"prior cohort {label!r} has no explicit candidate checkpoint binding"
            )
        for value in explicit_hashes:
            try:
                promotion._validate_sha256(  # noqa: SLF001
                    value, where=f"prior cohort {label!r} candidate checkpoint"
                )
            except promotion.PromotionError as error:
                raise ArtifactBuildError(str(error)) from error
        try:
            seeds = promotion._explicit_game_seeds(  # noqa: SLF001
                payload, where=f"prior cohort {label!r}"
            )
            intervals = promotion._contiguous_seed_intervals(  # noqa: SLF001
                seeds, kind=kind, where=f"prior cohort {label!r}"
            )
        except promotion.PromotionError as error:
            raise ArtifactBuildError(str(error)) from error
        records.append(
            {
                "label": label,
                "kind": kind,
                "source": _file_ref(source, where=f"prior cohort {label!r}"),
                "seed_intervals": [
                    {
                        "base_seed": interval["base_seed"],
                        "end_seed": interval["end_seed"],
                    }
                    for interval in intervals
                ],
            }
        )
    value = {
        "schema_version": promotion.COHORT_EXCLUSIONS_SCHEMA,
        "contract_sha256": contract["contract_sha256"],
        "candidate_sha256": candidate_ref["sha256"],
        "cohorts": records,
    }
    value["manifest_sha256"] = promotion._digest_value(value)  # noqa: SLF001
    return value


def _parse_cohort_paths(values: Sequence[str]) -> list[tuple[str, str, Path]]:
    parsed: list[tuple[str, str, Path]] = []
    for raw in values:
        identity, separator, path = raw.partition("=")
        label, kind_separator, kind = identity.partition(":")
        if not separator or not kind_separator or not label or not kind or not path:
            raise ArtifactBuildError(
                "--cohort entries must be LABEL:KIND=PATH"
            )
        parsed.append((label, kind, Path(path)))
    return parsed


def _contract(
    path: Path, legacy_contract_attestation: Path | None = None
) -> dict[str, Any]:
    try:
        return promotion._verify_contract(  # noqa: SLF001
            path.expanduser().resolve(),
            legacy_contract_attestation=legacy_contract_attestation,
        )
    except promotion.PromotionError as error:
        raise ArtifactBuildError(str(error)) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    legacy_contract = subparsers.add_parser(
        "legacy-contract-attestation",
        help="bind the allowlisted markerless v2 contract to its exact completed dose",
    )
    legacy_contract.add_argument("--contract-lock", type=Path, required=True)
    legacy_contract.add_argument("--training-receipt", type=Path, required=True)
    legacy_contract.add_argument("--out", type=Path, required=True)

    high = subparsers.add_parser("high-regret", help="derive a high-regret source")
    high.add_argument("--report", type=Path, required=True)
    high.add_argument("--candidate", type=Path, required=True)
    high.add_argument("--champion", type=Path, required=True)
    high.add_argument("--out", type=Path, required=True)

    suite = subparsers.add_parser(
        "held-out-suite", help="materialize a deterministic high-regret suite"
    )
    suite.add_argument("--manifest", type=Path, required=True)
    suite.add_argument("--holdout-fraction", type=float, default=1.0)
    suite.add_argument("--holdout-seed", type=int, required=True)
    suite.add_argument("--pairs", type=int, required=True)
    suite.add_argument("--out", type=Path, required=True)

    bucket_report = subparsers.add_parser(
        "bucket-report", help="extract bucket-labelled retained game outcomes"
    )
    bucket_report.add_argument("--report", type=Path, required=True)
    bucket_report.add_argument("--candidate", type=Path, required=True)
    bucket_report.add_argument("--champion", type=Path, required=True)
    bucket_report.add_argument("--out", type=Path, required=True)

    buckets = subparsers.add_parser("bucket-veto", help="derive a bucket veto")
    buckets.add_argument("--report", type=Path, required=True)
    buckets.add_argument("--candidate", type=Path, required=True)
    buckets.add_argument("--champion", type=Path, required=True)
    buckets.add_argument("--out", type=Path, required=True)

    legacy = subparsers.add_parser(
        "legacy-incumbent-calibration",
        help="bind a legacy incumbent calibration to its historical training report",
    )
    legacy.add_argument("--calibration", type=Path, required=True)
    legacy.add_argument("--historical-training-report", type=Path, required=True)
    legacy.add_argument("--contract-lock", type=Path, required=True)
    legacy.add_argument("--legacy-contract-attestation", type=Path)
    legacy.add_argument("--champion", type=Path, required=True)
    legacy.add_argument("--out", type=Path, required=True)

    evidence = subparsers.add_parser("evidence", help="build one evidence envelope")
    evidence.add_argument(
        "--kind", choices=sorted(promotion.REQUIRED_EVIDENCE_KINDS), required=True
    )
    evidence.add_argument("--contract-lock", type=Path, required=True)
    evidence.add_argument("--legacy-contract-attestation", type=Path)
    evidence.add_argument("--candidate", type=Path, required=True)
    evidence.add_argument("--champion", type=Path, required=True)
    evidence.add_argument(
        "--promotion-mode",
        choices=("promotion_parent", "branch_challenge"),
        default="promotion_parent",
    )
    evidence.add_argument(
        "--candidate-parent",
        type=Path,
        help="authenticated older initializer for branch_challenge evidence",
    )
    evidence.add_argument("--registry", type=Path, required=True)
    evidence.add_argument("--source", action="append", default=[], metavar="ROLE=PATH")
    evidence.add_argument("--out", type=Path, required=True)

    exclusions = subparsers.add_parser(
        "cohort-exclusions",
        help="derive candidate-bound prior-cohort seed exclusions from raw reports",
    )
    exclusions.add_argument("--contract-lock", type=Path, required=True)
    exclusions.add_argument("--legacy-contract-attestation", type=Path)
    exclusions.add_argument("--candidate", type=Path, required=True)
    exclusions.add_argument(
        "--cohort", action="append", default=[], metavar="LABEL:KIND=PATH"
    )
    exclusions.add_argument("--out", type=Path, required=True)

    adjudicate = subparsers.add_parser(
        "adjudicate", help="build final promotion adjudication"
    )
    adjudicate.add_argument("--contract-lock", type=Path, required=True)
    adjudicate.add_argument("--legacy-contract-attestation", type=Path)
    adjudicate.add_argument("--training-receipt", type=Path, required=True)
    adjudicate.add_argument("--registry", type=Path, required=True)
    adjudicate.add_argument("--current-pointer", type=Path, required=True)
    adjudicate.add_argument("--candidate", type=Path, required=True)
    adjudicate.add_argument("--candidate-version", type=int, required=True)
    adjudicate.add_argument("--training-report", type=Path, required=True)
    adjudicate.add_argument("--champion", type=Path, required=True)
    adjudicate.add_argument("--champion-version", type=int, required=True)
    adjudicate.add_argument(
        "--promotion-mode",
        choices=("promotion_parent", "branch_challenge"),
        default="promotion_parent",
    )
    adjudicate.add_argument(
        "--evidence", action="append", default=[], metavar="KIND=PATH"
    )
    adjudicate.add_argument("--nth-confirmation", type=Path)
    adjudicate.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "legacy-contract-attestation":
            value = promotion.build_legacy_contract_attestation(
                args.contract_lock, args.training_receipt
            )
        elif args.command == "high-regret":
            value = build_high_regret_source(
                report_path=args.report,
                candidate=args.candidate,
                champion=args.champion,
            )
        elif args.command == "held-out-suite":
            value = build_held_out_high_regret_suite(
                manifest_path=args.manifest,
                holdout_fraction=args.holdout_fraction,
                holdout_seed=args.holdout_seed,
                pairs=args.pairs,
            )
        elif args.command == "bucket-report":
            value = build_bucket_game_report(
                report_path=args.report,
                candidate=args.candidate,
                champion=args.champion,
            )
        elif args.command == "bucket-veto":
            value = build_bucket_veto_source(
                report_path=args.report,
                candidate=args.candidate,
                champion=args.champion,
            )
        elif args.command == "legacy-incumbent-calibration":
            value = build_legacy_incumbent_calibration_source(
                calibration_path=args.calibration,
                historical_training_report=args.historical_training_report,
                contract=_contract(
                    args.contract_lock, args.legacy_contract_attestation
                ),
                champion=args.champion,
            )
        elif args.command == "evidence":
            contract = _contract(
                args.contract_lock, args.legacy_contract_attestation
            )
            sources = _parse_role_paths(args.source, option="--source")
            value = build_evidence_envelope(
                kind=args.kind,
                contract=contract,
                candidate=args.candidate,
                champion=args.champion,
                sources=sources,
                promotion_mode=args.promotion_mode,
            )
            if (
                args.promotion_mode == "branch_challenge"
                and args.candidate_parent is None
            ):
                raise ArtifactBuildError(
                    "branch_challenge evidence requires --candidate-parent"
                )
            if (
                args.promotion_mode == "promotion_parent"
                and args.candidate_parent is not None
            ):
                raise ArtifactBuildError(
                    "--candidate-parent is only valid for branch_challenge"
                )
            _validate_envelope_before_write(
                args.out,
                value=value,
                kind=args.kind,
                contract=contract,
                candidate=args.candidate,
                champion=args.champion,
                registry=ChampionRegistry.load(args.registry),
                promotion_mode=args.promotion_mode,
                candidate_parent=args.candidate_parent,
            )
        elif args.command == "cohort-exclusions":
            value = build_cohort_exclusions(
                contract=_contract(
                    args.contract_lock, args.legacy_contract_attestation
                ),
                candidate=args.candidate,
                cohorts=_parse_cohort_paths(args.cohort),
            )
        else:
            contract = _contract(
                args.contract_lock, args.legacy_contract_attestation
            )
            registry = ChampionRegistry.load(args.registry)
            value = build_adjudication(
                contract=contract,
                contract_lock=args.contract_lock.expanduser().resolve(),
                training_receipt=args.training_receipt,
                registry=registry,
                current_pointer=args.current_pointer,
                candidate=args.candidate,
                candidate_version=args.candidate_version,
                training_report=args.training_report,
                champion=args.champion,
                champion_version=args.champion_version,
                evidence=_parse_role_paths(args.evidence, option="--evidence"),
                nth_confirmation=args.nth_confirmation,
                promotion_mode=args.promotion_mode,
            )
            # Transaction replay is the final authority.  Use a temporary file
            # because the verifier deliberately consumes a path.
            args.out.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(
                prefix=f".{args.out.name}.", suffix=".verify", dir=args.out.parent
            )
            temporary = Path(name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(value, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                promotion._verify_adjudication(  # noqa: SLF001
                    temporary,
                    contract=contract,
                    contract_lock=args.contract_lock.expanduser().resolve(),
                    training_receipt=args.training_receipt.expanduser().resolve(),
                    registry=registry,
                    current_pointer=args.current_pointer.expanduser().resolve(),
                )
            finally:
                temporary.unlink(missing_ok=True)
        _write_new_readonly(args.out, value)
        print(
            json.dumps(
                {
                    "path": str(args.out.expanduser().resolve()),
                    "sha256": promotion._sha256(args.out),
                },
                sort_keys=True,
            )
        )  # noqa: SLF001
        return 0
    except (ArtifactBuildError, promotion.PromotionError, OSError, KeyError) as error:
        print(f"a1 promotion artifact build refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
