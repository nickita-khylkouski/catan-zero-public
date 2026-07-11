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


HIGH_REGRET_REPORT_SCHEMA = "a1-held-out-high-regret-report-v1"
HIGH_REGRET_SUITE_SCHEMA = "a1-held-out-high-regret-suite-v1"
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
    _suite_path, suite_ref = _bound_file_ref(
        raw["suite_manifest"],
        base=report_path.parent,
        where="high-regret report.suite_manifest",
    )
    games = raw["games"]
    if not isinstance(games, list) or not games:
        raise ArtifactBuildError("high-regret report.games must be a non-empty list")
    identities: set[tuple[int, str]] = set()
    for index, game in enumerate(games):
        identity = _paired_game_identity(
            game, index=index, where="high-regret report.games"
        )
        if identity in identities:
            raise ArtifactBuildError("high-regret report contains duplicate games")
        if not isinstance(game.get("candidate_won"), bool):
            raise ArtifactBuildError(
                f"high-regret report.games[{index}].candidate_won must be boolean"
            )
        identities.add(identity)
    normalized_games = [{**game, "search_won": game["candidate_won"]} for game in games]
    pair_scores, diagnostics = promotion.pair_scores_from_h2h_games(normalized_games)
    pentanomial = promotion.evaluate_pentanomial_sprt(
        pair_scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    complete_pairs = sum(
        diagnostics[key] for key in ("ww_pairs", "split_pairs", "ll_pairs")
    )
    _positive_int(complete_pairs, where="high-regret report complete pairs")
    if diagnostics["incomplete_pairs"] != 0:
        raise ArtifactBuildError("high-regret report contains incomplete pairs")
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
    if not (0.0 < holdout_fraction < 1.0):
        raise ArtifactBuildError("holdout_fraction must be in (0, 1)")
    pairs = _positive_int(pairs, where="held-out suite pairs")
    if pairs < 20:
        raise ArtifactBuildError("held-out suite requires at least 20 pairs")
    try:
        with np.load(manifest_path, allow_pickle=True) as data:
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

    def stable_unit_hash(game_seed: int, decision_index: int) -> float:
        return (
            hash((int(game_seed), int(decision_index), int(holdout_seed))) & 0xFFFFFFFF
        ) / 0xFFFFFFFF

    eligible = [
        index
        for index in range(len(game_seeds))
        if stable_unit_hash(int(game_seeds[index]), int(decisions[index]))
        < holdout_fraction
    ]
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

    replay_complete, replay_stats = _replay_complete_manifest_rows(
        manifest_path=manifest_path,
        candidate_indices=eligible,
        shard_paths=shard_paths,
        shard_ids=shard_ids,
        row_indices=row_indices,
        game_seeds=game_seeds,
        decisions=decisions,
    )
    eligible = [index for index in eligible if index in replay_complete]

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
    seen: set[tuple[int, int]] = set()
    selected_by_stratum: dict[str, int] = {}

    def select_from(indices: Sequence[int], want: int, *, label: str) -> None:
        before = len(selected)
        for index in indices:
            identity = (int(game_seeds[index]), int(decisions[index]))
            if identity in seen:
                continue
            seen.add(identity)
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
            identity = (int(game_seeds[index]), int(decisions[index]))
            if identity in seen:
                continue
            seen.add(identity)
            selected.append(index)
            if len(selected) == pairs:
                break
    if len(selected) != pairs:
        raise ArtifactBuildError(
            f"held-out partition has only {len(selected)} replay-complete unique "
            f"states, need {pairs} ({replay_stats})"
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
                    "contract": "authoritative-shard-parent-unique-contiguous-prefix-v1",
                    "scope": str(shard_path.resolve().parent),
                },
            }
        )
    value = {
        "schema_version": HIGH_REGRET_SUITE_SCHEMA,
        "suite": "held_out_high_regret",
        "held_out": True,
        "source_manifest": _file_ref(manifest_path, where="regret manifest"),
        "selection": {
            "algorithm": "stable-hash-holdout-stratified-regret-v1",
            "holdout_fraction": float(holdout_fraction),
            "holdout_seed": int(holdout_seed),
            "eligible_unique_states": eligible_unique_states,
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
) -> tuple[set[int], dict[str, Any]]:
    """Return candidates with an authoritative, replay-complete prefix.

    The evaluator reconstructs a state by scanning every shard below the
    selected shard's parent directory.  Mirror that contract before sealing:
    the manifest row must bind to the stated source row and its source scope
    must contain each decision exactly once from zero through the target.
    Missing, duplicate, or role-filtered partial trajectories fail closed.
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
    for scope, targets in requested_by_scope.items():
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
            if "game_seed" not in shard or "decision_index" not in shard:
                continue
            seeds = np.asarray(shard["game_seed"]).reshape(-1)
            didx = np.asarray(shard["decision_index"]).reshape(-1)
            actions = shard.get("action_taken")
            if (
                len(seeds) != len(didx)
                or actions is None
                or len(np.asarray(actions).reshape(-1)) != len(seeds)
            ):
                for seed in set(int(value) for value in seeds if int(value) in targets):
                    malformed.add(seed)
                continue
            for row in range(len(seeds)):
                seed = int(seeds[row])
                if seed not in targets:
                    continue
                decision = int(didx[row])
                if decision >= 0:
                    per_seed = seed_counts[seed]
                    per_seed[decision] = per_seed.get(decision, 0) + 1
        counts_by_scope[scope] = seed_counts
        malformed_seeds_by_scope[scope] = malformed

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

    return complete, {
        "contract": "authoritative-shard-parent-unique-contiguous-prefix-v1",
        "candidate_states": len(candidate_indices),
        "replay_complete_states": len(complete),
        "rejected_bad_source": rejected_bad_source,
        "rejected_noncontiguous": rejected_noncontiguous,
    }


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
        },
        where="high-regret evaluation report",
    )
    if raw["schema_version"] != HIGH_REGRET_REPORT_SCHEMA or raw["errors"] != []:
        raise ArtifactBuildError(
            "bucket extraction requires a clean high-regret report"
        )
    candidate_ref = _verify_checkpoint_ref(
        raw["candidate"], expected=candidate, where="bucket report.candidate"
    )
    champion_ref = _verify_checkpoint_ref(
        raw["champion"], expected=champion, where="bucket report.champion"
    )
    games = raw["games"]
    if not isinstance(games, list) or not games:
        raise ArtifactBuildError("bucket extraction report has no games")
    projected: list[dict[str, Any]] = []
    identities: set[tuple[int, str]] = set()
    for index, game in enumerate(games):
        identity = _paired_game_identity(game, index=index, where="report.games")
        outcome = game.get("candidate_won")
        labels = game.get("buckets")
        if identity in identities or not isinstance(outcome, bool):
            raise ArtifactBuildError("bucket extraction has duplicate/incomplete games")
        if (
            not isinstance(labels, list)
            or not labels
            or not all(isinstance(label, str) and label for label in labels)
            or len(labels) != len(set(labels))
        ):
            raise ArtifactBuildError(f"report.games[{index}] has invalid bucket labels")
        identities.add(identity)
        projected.append(
            {
                "pair_id": identity[0],
                "orientation": identity[1],
                "candidate_won": outcome,
                "buckets": sorted(labels),
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
    if Path(str(historical.get("checkpoint"))).expanduser().resolve() != champion:
        raise ArtifactBuildError("historical report does not bind the incumbent")
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
) -> dict[str, Any]:
    if kind not in promotion.REQUIRED_EVIDENCE_KINDS:
        raise ArtifactBuildError(f"unsupported evidence kind {kind!r}")
    roles = [role for role, _path in sources]
    if len(set(roles)) != len(roles):
        raise ArtifactBuildError("evidence source roles must be unique")
    verdict = "H1" if kind == "internal_h2h" else "pass"
    result: dict[str, Any]
    if kind == "mechanism_calibration":
        result = {
            "value_readout": str(
                contract["science"]["learner_value_objective"]["value_readout"]
            ),
            "max_rmse_regression": promotion.MAX_CALIBRATION_RMSE_REGRESSION,
        }
    elif kind == "external_panel":
        result = {"max_win_rate_regression": promotion.MAX_EXTERNAL_WIN_RATE_REGRESSION}
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
        sealed_semantics = promotion._sealed_evaluation_semantics(contract)  # noqa: SLF001
        promotion._verify_promotion_evidence(  # noqa: SLF001
            temporary,
            kind=kind,
            contract=contract,
            expected_readout=str(
                contract["science"]["learner_value_objective"]["value_readout"]
            ),
            candidate={
                **_checkpoint_ref(candidate),
                "md5": promotion._md5(candidate),  # noqa: SLF001
                "search_config": promotion._role_search_config(  # noqa: SLF001
                    sealed_semantics, role="candidate"
                ),
            },
            champion={
                **_checkpoint_ref(champion),
                "md5": promotion._md5(champion),  # noqa: SLF001
                "search_config": promotion._role_search_config(  # noqa: SLF001
                    sealed_semantics, role="champion"
                ),
            },
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
    nth_confirmation_passed: bool,
) -> dict[str, Any]:
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
    if nth_required and not nth_confirmation_passed:
        raise ArtifactBuildError("this promotion requires a passing n64 confirmation")
    if not nth_required and nth_confirmation_passed:
        raise ArtifactBuildError(
            "n64 confirmation was asserted for a non-third promotion"
        )
    sealed_semantics = promotion._sealed_evaluation_semantics(contract)  # noqa: SLF001
    candidate_ref = _checkpoint_ref(candidate)
    champion_ref = _checkpoint_ref(champion)
    value = {
        "schema_version": promotion.ADJUDICATION_SCHEMA,
        "passed": True,
        "decision": "promote",
        "contract_sha256": contract["contract_sha256"],
        "candidate": {
            **candidate_ref,
            "version": candidate_version,
            "agent_identity": promotion._agent_identity(  # noqa: SLF001
                candidate_ref,
                promotion._role_search_config(  # noqa: SLF001
                    sealed_semantics, role="candidate"
                ),
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
                promotion._role_search_config(  # noqa: SLF001
                    sealed_semantics, role="champion"
                ),
            ),
        },
        "checks": {name: True for name in promotion.REQUIRED_CHECKS},
        "nth_confirmation_required": nth_required,
        "nth_confirmation_passed": nth_confirmation_passed,
        "evidence": [
            {
                "kind": kind,
                "path": str(by_kind[kind].expanduser().resolve()),
                "sha256": promotion._sha256(by_kind[kind]),  # noqa: SLF001
            }
            for kind in sorted(by_kind)
        ],
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


def _contract(path: Path) -> dict[str, Any]:
    try:
        return promotion._verify_contract(path.expanduser().resolve())  # noqa: SLF001
    except promotion.PromotionError as error:
        raise ArtifactBuildError(str(error)) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    high = subparsers.add_parser("high-regret", help="derive a high-regret source")
    high.add_argument("--report", type=Path, required=True)
    high.add_argument("--candidate", type=Path, required=True)
    high.add_argument("--champion", type=Path, required=True)
    high.add_argument("--out", type=Path, required=True)

    suite = subparsers.add_parser(
        "held-out-suite", help="materialize a deterministic high-regret suite"
    )
    suite.add_argument("--manifest", type=Path, required=True)
    suite.add_argument("--holdout-fraction", type=float, default=0.10)
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
    legacy.add_argument("--champion", type=Path, required=True)
    legacy.add_argument("--out", type=Path, required=True)

    evidence = subparsers.add_parser("evidence", help="build one evidence envelope")
    evidence.add_argument(
        "--kind", choices=sorted(promotion.REQUIRED_EVIDENCE_KINDS), required=True
    )
    evidence.add_argument("--contract-lock", type=Path, required=True)
    evidence.add_argument("--candidate", type=Path, required=True)
    evidence.add_argument("--champion", type=Path, required=True)
    evidence.add_argument("--source", action="append", default=[], metavar="ROLE=PATH")
    evidence.add_argument("--out", type=Path, required=True)

    adjudicate = subparsers.add_parser(
        "adjudicate", help="build final promotion adjudication"
    )
    adjudicate.add_argument("--contract-lock", type=Path, required=True)
    adjudicate.add_argument("--training-receipt", type=Path, required=True)
    adjudicate.add_argument("--registry", type=Path, required=True)
    adjudicate.add_argument("--current-pointer", type=Path, required=True)
    adjudicate.add_argument("--candidate", type=Path, required=True)
    adjudicate.add_argument("--candidate-version", type=int, required=True)
    adjudicate.add_argument("--training-report", type=Path, required=True)
    adjudicate.add_argument("--champion", type=Path, required=True)
    adjudicate.add_argument("--champion-version", type=int, required=True)
    adjudicate.add_argument(
        "--evidence", action="append", default=[], metavar="KIND=PATH"
    )
    adjudicate.add_argument("--nth-confirmation-passed", action="store_true")
    adjudicate.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "high-regret":
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
                contract=_contract(args.contract_lock),
                champion=args.champion,
            )
        elif args.command == "evidence":
            contract = _contract(args.contract_lock)
            sources = _parse_role_paths(args.source, option="--source")
            value = build_evidence_envelope(
                kind=args.kind,
                contract=contract,
                candidate=args.candidate,
                champion=args.champion,
                sources=sources,
            )
            _validate_envelope_before_write(
                args.out,
                value=value,
                kind=args.kind,
                contract=contract,
                candidate=args.candidate,
                champion=args.champion,
            )
        else:
            contract = _contract(args.contract_lock)
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
                nth_confirmation_passed=args.nth_confirmation_passed,
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
