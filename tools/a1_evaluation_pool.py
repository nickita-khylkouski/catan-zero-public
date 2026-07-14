#!/usr/bin/env python3
"""Fail-closed pooling for distributed A1 promotion evaluations.

Fleet evaluators reset ``pair_id`` in every process.  Concatenating their JSON
therefore corrupts pairing and can double-count a seed.  This tool pools either
cross-net H2H or neutral-harness reports by globally unique ``game_seed``,
requires identical science/config/checkpoint bytes, rejects duplicate or
incomplete pairs, renumbers pairs deterministically, and recomputes all gate
statistics from the retained raw games.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import a1_promotion_artifacts as artifacts  # noqa: E402
from tools import a1_promotion_transaction as promotion  # noqa: E402
from tools.gumbel_search_cross_net_h2h import (  # noqa: E402
    _add_search_telemetry,
    _finalize_search_telemetry,
    _new_search_telemetry,
)
from tools.sprt_gate import (  # noqa: E402
    evaluate_pentanomial_sprt,
    evaluate_sprt,
    pair_scores_from_h2h_games,
)


POOL_SCHEMA = "a1-fleet-evaluation-pool-v1"
ORIENTATIONS = {
    "internal": {"candidate_red", "candidate_blue"},
    "neutral": {"candidate_first", "candidate_second"},
}


class PoolError(RuntimeError):
    """Raised when fleet reports cannot form one promotion-grade cohort."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PoolError(f"cannot load report {path}: {error}") from error
    if not isinstance(value, dict):
        raise PoolError(f"report {path} must contain a JSON object")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _validate_checkpoint_sha256(
    report: dict[str, Any], *, key: str, expected: Path, where: str
) -> str:
    actual = promotion._sha256(expected.expanduser().resolve(strict=True))  # noqa: SLF001
    declared = report.get(key)
    if declared != actual:
        raise PoolError(f"{where} checkpoint SHA-256 drift: {declared!r} != {actual}")
    return actual


def _validate_config_hash(report: dict[str, Any], *, where: str) -> None:
    typed = report.get("typed_config")
    if not isinstance(typed, dict):
        raise PoolError(f"{where} has no typed_config")
    digest = hashlib.sha256(_canonical(typed)).hexdigest()
    if report.get("full_config_hash") != "sha256:" + digest:
        raise PoolError(f"{where} full_config_hash does not replay")
    if report.get("config_hash") != "sha256:" + digest[:16]:
        raise PoolError(f"{where} config_hash does not replay")


def _internal_effective_search_config(
    report: dict[str, Any], *, where: str
) -> dict[str, Any]:
    """Remove only per-shard identity/seed fields from a replayed typed config."""
    _validate_config_hash(report, where=where)
    typed = report["typed_config"]
    if typed.get("pipeline") != "eval" or not isinstance(typed.get("fields"), dict):
        raise PoolError(f"{where} typed_config is not an evaluation config")
    fields = copy.deepcopy(typed["fields"])
    for name in ("candidate", "baseline", "base_seed", "pairs"):
        fields.pop(name, None)
    return fields


def _neutral_effective_search_config(
    report: dict[str, Any], *, where: str
) -> dict[str, Any]:
    search = report.get("search_config")
    if not isinstance(search, dict) or not search:
        raise PoolError(f"{where} has no effective search_config")
    identity = {
        "stratum": report.get("stratum"),
        "harness": report.get("harness"),
        "referee_engine": report.get("referee_engine"),
        "engine_identity": report.get("engine_identity"),
        "baseline_bot": report.get("baseline_bot"),
        "mode": report.get("mode"),
        "map_kind": report.get("map_kind"),
        "gate_config": report.get("gate_config"),
        "vps_to_win": report.get("vps_to_win"),
        "max_player_trade_offers_per_turn": report.get(
            "max_player_trade_offers_per_turn"
        ),
        "trained_value_readouts": report.get("trained_value_readouts"),
        "search_config": search,
    }
    return identity


def _seed_interval(report: dict[str, Any], *, where: str) -> tuple[int, int]:
    try:
        base = int(report["base_seed"])
        pairs = int(report["pairs_requested"])
    except (KeyError, TypeError, ValueError) as error:
        raise PoolError(f"{where} has no valid seed interval") from error
    if pairs <= 0:
        raise PoolError(f"{where} pairs_requested must be positive")
    games = report.get("games")
    if not isinstance(games, list):
        raise PoolError(f"{where} has no retained games")
    counts: dict[int, int] = {}
    for game in games:
        try:
            seed = int(game["game_seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise PoolError(f"{where} game has no valid game_seed") from error
        counts[seed] = counts.get(seed, 0) + 1
    expected = set(range(base, base + pairs))
    if set(counts) != expected or any(count != 2 for count in counts.values()):
        raise PoolError(
            f"{where} raw games do not exactly cover [{base}, {base + pairs}) twice"
        )
    return base, base + pairs


def validate_complete_report(
    report: dict[str, Any],
    *,
    kind: str,
    expected_pairs: int | None = None,
    where: str = "report",
) -> dict[str, Any]:
    """Prove that one evaluator lane produced its entire clean assignment.

    Evaluators intentionally retain worker failures in a JSON report and can
    exit zero so that diagnostics are not lost.  That behavior is useful, but
    it means process success and a nonempty file are not completion signals.
    This validator is the shared semantic boundary used by launch/status and
    pooling: every requested seed must have both orientations, all headline
    counts must replay from the raw games, and every error counter/list must be
    empty.
    """

    if kind not in ORIENTATIONS:
        raise PoolError(f"{where} has unsupported report kind {kind!r}")
    if not isinstance(report, dict):
        raise PoolError(f"{where} is not a JSON object")
    games = report.get("games")
    if not isinstance(games, list):
        raise PoolError(f"{where} has no retained games")

    # Preserve the most informative error when the two legacy outcome aliases
    # disagree, before validating their aggregate counters.
    for index, game in enumerate(games):
        if not isinstance(game, dict):
            raise PoolError(f"{where}.games[{index}] is not an object")
        if type(game.get("candidate_won")) is not bool or type(  # noqa: E721
            game.get("search_won")
        ) is not bool:
            raise PoolError(f"{where}.games[{index}] has no boolean outcome")
        if game["candidate_won"] != game["search_won"]:
            raise PoolError(
                f"{where}.games[{index}] candidate_won/search_won alias drift"
            )

    base, end = _seed_interval(report, where=where)
    pairs = report.get("pairs_requested")
    if type(pairs) is not int or pairs <= 0:  # noqa: E721
        raise PoolError(f"{where} pairs_requested must be a positive integer")
    if expected_pairs is not None and pairs != int(expected_pairs):
        raise PoolError(
            f"{where} pairs_requested={pairs} differs from planned {expected_pairs}"
        )
    expected_games = 2 * pairs
    if len(games) != expected_games:
        raise PoolError(
            f"{where} retained {len(games)} games, expected {expected_games}"
        )
    if kind == "internal":
        try:
            promotion._verify_internal_h2h_rng_contract(  # noqa: SLF001
                report, where=where
            )
        except promotion.PromotionError as error:
            raise PoolError(str(error)) from error

    def require_count(name: str, expected: int) -> None:
        value = report.get(name)
        if type(value) is not int or value != expected:  # noqa: E721
            raise PoolError(
                f"{where} {name}={value!r} does not reconcile to {expected}"
            )

    require_count("games_played", expected_games)
    require_count("games_with_winner", expected_games)
    require_count("complete_pairs", pairs)
    if kind == "neutral":
        require_count("games_requested", expected_games)

    for name in ("errors", "worker_errors", "pair_errors"):
        if name == "errors" or name in report:
            value = report.get(name)
            if not isinstance(value, list) or value:
                raise PoolError(f"{where} {name} must be an empty list")
    for name in (
        "games_truncated",
        "games_errored",
        "games_engine_divergence",
        "total_illegal_policy_picks",
    ):
        if name == "games_truncated" or name in report:
            require_count(name, 0)

    expected_orientations = ORIENTATIONS[kind]
    orientations_by_seed: dict[int, set[str]] = {}
    wins = 0
    for index, game in enumerate(games):
        seed = int(game["game_seed"])
        orientation = str(game.get("orientation"))
        orientations_by_seed.setdefault(seed, set()).add(orientation)
        wins += int(game["candidate_won"])
        if (
            orientation not in expected_orientations
            or game.get("terminated") is not True
            or game.get("truncated") is not False
            or game.get("error") not in {None, ""}
            or bool(game.get("engine_divergence", False))
        ):
            raise PoolError(f"{where}.games[{index}] is not a complete clean game")
    for seed in range(base, end):
        if orientations_by_seed.get(seed) != expected_orientations:
            raise PoolError(
                f"{where} seed {seed} does not contain both required orientations"
            )

    require_count("candidate_wins", wins)
    require_count("baseline_wins", expected_games - wins)
    diagnostics = report.get("pair_diagnostics")
    if not isinstance(diagnostics, dict):
        raise PoolError(f"{where} has no pair_diagnostics")
    diagnostic_counts: dict[str, int] = {}
    for name in ("ww_pairs", "split_pairs", "ll_pairs", "incomplete_pairs"):
        value = diagnostics.get(name)
        if type(value) is not int or value < 0:  # noqa: E721
            raise PoolError(f"{where} pair_diagnostics.{name} is invalid")
        diagnostic_counts[name] = value
    if diagnostic_counts["incomplete_pairs"] != 0 or sum(
        diagnostic_counts.values()
    ) != pairs:
        raise PoolError(f"{where} pair_diagnostics do not cover all planned pairs")
    return report


def validate_complete_report_path(
    path: Path, *, kind: str, expected_pairs: int | None = None
) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    return validate_complete_report(
        _load(path), kind=kind, expected_pairs=expected_pairs, where=str(path)
    )


def _contiguous_intervals(
    reports: Sequence[tuple[Path, dict[str, Any]]],
    *,
    allow_gaps: bool = False,
) -> list[dict[str, Any]]:
    intervals = sorted(
        ((*_seed_interval(report, where=str(path)), path) for path, report in reports),
        key=lambda row: (row[0], row[1], str(row[2])),
    )
    for previous, current in zip(intervals, intervals[1:]):
        if previous[1] > current[0]:
            raise PoolError(
                "fleet seed intervals have an overlap: "
                f"[{previous[0]}, {previous[1]}) then [{current[0]}, {current[1]})"
            )
        if previous[1] < current[0] and not allow_gaps:
            raise PoolError(
                "fleet seed intervals have a gap: "
                f"[{previous[0]}, {previous[1]}) then [{current[0]}, {current[1]})"
            )
    return [
        {"base_seed": lo, "end_seed": hi, "path": str(path)}
        for lo, hi, path in intervals
    ]


def _pair_diagnostics(
    games: list[dict[str, Any]],
) -> tuple[list[float], dict[str, int]]:
    scores, diagnostics = pair_scores_from_h2h_games(games)
    return scores, diagnostics


def _validate_and_normalize_games(
    reports: Sequence[tuple[Path, dict[str, Any]]], *, kind: str
) -> list[dict[str, Any]]:
    seen: dict[tuple[int, str], Path] = {}
    by_seed: dict[int, list[dict[str, Any]]] = {}
    expected_orientations = ORIENTATIONS[kind]
    for path, report in reports:
        games = report.get("games")
        if not isinstance(games, list) or not games:
            raise PoolError(f"{path} has no retained raw games")
        if report.get("games_played") != len(games):
            raise PoolError(f"{path} games_played differs from retained games")
        source_sha = promotion._sha256(path)  # noqa: SLF001
        for index, raw in enumerate(games):
            if not isinstance(raw, dict):
                raise PoolError(f"{path}.games[{index}] is not an object")
            game = copy.deepcopy(raw)
            try:
                seed = int(game["game_seed"])
                orientation = str(game["orientation"])
                source_pair_id = int(game["pair_id"])
            except (KeyError, TypeError, ValueError) as error:
                raise PoolError(f"{path}.games[{index}] lacks pair identity") from error
            if orientation not in expected_orientations:
                raise PoolError(
                    f"{path}.games[{index}] has invalid orientation {orientation!r}"
                )
            identity = (seed, orientation)
            prior = seen.get(identity)
            if prior is not None:
                raise PoolError(
                    f"duplicate fleet game seed/orientation {identity}: {prior} and {path}"
                )
            seen[identity] = path
            if (
                game.get("candidate_won") is None
                or game.get("search_won") is None
                or game.get("truncated") is True
                or game.get("terminated") is not True
                or game.get("error") not in {None, ""}
                or bool(game.get("engine_divergence", False))
            ):
                raise PoolError(f"{path}.games[{index}] is not a complete clean game")
            # ``search_won`` is the legacy field consumed by the pentanomial
            # helper while every headline win-rate field is computed from
            # ``candidate_won``.  They are aliases for these candidate-vs-
            # baseline reports, not two independent observations.  Refuse a
            # corrupt/mixed producer rather than letting the gate and the
            # displayed result score different winners.
            if (
                not isinstance(game.get("candidate_won"), bool)
                or not isinstance(game.get("search_won"), bool)
                or game["candidate_won"] is not game["search_won"]
            ):
                raise PoolError(
                    f"{path}.games[{index}] candidate_won/search_won alias drift"
                )
            game["source_pair_id"] = source_pair_id
            game["source_report_sha256"] = source_sha
            by_seed.setdefault(seed, []).append(game)
    for seed, games in by_seed.items():
        orientations = {str(game["orientation"]) for game in games}
        if len(games) != 2 or orientations != expected_orientations:
            raise PoolError(
                f"game seed {seed} is not one complete seat-swapped pair: {sorted(orientations)}"
            )
    normalized: list[dict[str, Any]] = []
    for pair_id, seed in enumerate(sorted(by_seed)):
        for game in sorted(by_seed[seed], key=lambda item: str(item["orientation"])):
            game["pair_id"] = pair_id
            normalized.append(game)
    return normalized


def _source_refs(paths: Sequence[Path]) -> list[dict[str, str]]:
    return [
        {"path": str(path.resolve()), "sha256": promotion._sha256(path)}  # noqa: SLF001
        for path in paths
    ]


def _pool_internal_search_telemetry(
    reports: Sequence[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    """Reconstruct additive H2H search telemetry across every fleet shard.

    Individual H2H reports contain finalized rates as well as the additive
    counters from which those rates were derived.  Copying the first report's
    finalized object into the pool makes a many-shard cohort look like one
    shard.  Retain only the exact additive fields, sum them, and derive rates
    once from the fleet totals.
    """

    totals = _new_search_telemetry()
    additive_fields = tuple(next(iter(totals.values())).keys())
    for path, report in reports:
        telemetry = report.get("search_telemetry")
        by_role = telemetry.get("by_role") if isinstance(telemetry, dict) else None
        if not isinstance(by_role, dict):
            raise PoolError(f"{path} has no finalized search_telemetry.by_role")
        raw: dict[str, dict[str, float | int]] = {}
        for role in ("candidate", "baseline"):
            values = by_role.get(role)
            if not isinstance(values, dict):
                raise PoolError(f"{path} has no search telemetry for role {role!r}")
            raw[role] = {}
            for field in additive_fields:
                value = values.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise PoolError(
                        f"{path} search telemetry {role}.{field} is not numeric"
                    )
                numeric = float(value)
                if not math.isfinite(numeric) or numeric < 0.0:
                    raise PoolError(
                        f"{path} search telemetry {role}.{field} is invalid"
                    )
                raw[role][field] = value
            if int(raw[role]["non_forced_search_calls"]) > int(
                raw[role]["search_calls"]
            ):
                raise PoolError(
                    f"{path} search telemetry {role} has more non-forced than "
                    "total calls"
                )
        _add_search_telemetry(totals, raw)
    return _finalize_search_telemetry(totals)


def pool_internal(
    paths: Sequence[Path],
    *,
    candidate: Path,
    champion: Path,
    allow_disjoint_cohorts: bool = False,
) -> dict[str, Any]:
    if not paths:
        raise PoolError("at least one internal H2H report is required")
    loaded = [(path.resolve(), _load(path.resolve())) for path in paths]
    candidate = candidate.expanduser().resolve(strict=True)
    champion = champion.expanduser().resolve(strict=True)
    candidate_sha256 = promotion._sha256(candidate)  # noqa: SLF001
    champion_sha256 = promotion._sha256(champion)  # noqa: SLF001
    effective_config: dict[str, Any] | None = None
    for path, report in loaded:
        validate_complete_report(report, kind="internal", where=str(path))
        _validate_checkpoint_sha256(
            report,
            key="candidate_checkpoint_sha256",
            expected=candidate,
            where=f"{path} candidate",
        )
        _validate_checkpoint_sha256(
            report,
            key="baseline_checkpoint_sha256",
            expected=champion,
            where=f"{path} champion",
        )
        shard_effective = _internal_effective_search_config(report, where=str(path))
        if effective_config is None:
            effective_config = shard_effective
        elif _canonical(shard_effective) != _canonical(effective_config):
            raise PoolError(f"fleet report effective science/config drift in {path}")
        if report.get("gate_config") != "flywheel":
            raise PoolError(f"{path} is not a flywheel gate report")
        if report.get("errors") != [] or int(report.get("games_truncated", -1)) != 0:
            raise PoolError(f"{path} contains errors or truncations")
        scores, diagnostics = _pair_diagnostics(report.get("games", []))
        replay = evaluate_pentanomial_sprt(
            scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
        )
        if replay != report.get("pentanomial_sprt") or diagnostics != report.get(
            "pair_diagnostics"
        ):
            raise PoolError(f"{path} gate statistics do not replay")
    intervals = _contiguous_intervals(loaded, allow_gaps=allow_disjoint_cohorts)
    games = _validate_and_normalize_games(loaded, kind="internal")
    search_telemetry = _pool_internal_search_telemetry(loaded)
    outcomes = [bool(game["candidate_won"]) for game in games]
    pair_scores, diagnostics = _pair_diagnostics(games)
    concordant = []
    for pair_id in range(len(games) // 2):
        results = {
            bool(game["candidate_won"]) for game in games if game["pair_id"] == pair_id
        }
        if len(results) == 1:
            concordant.append(next(iter(results)))
    # The mean/variance GSPRT is a terminal statistic with a pooled nuisance
    # variance and one shared regularizing prior.  Per-cohort terminal LLRs are
    # deliberately NOT summed: they are non-additive.  Recompute once from the
    # union of retained raw games, which is the authoritative continuation
    # verdict (see tools/sprt_gate.py).
    pentanomial = evaluate_pentanomial_sprt(
        pair_scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    superiority = evaluate_pentanomial_sprt(
        pair_scores, elo0=0.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    complete_pairs = len(games) // 2
    result = copy.deepcopy(loaded[0][1])
    result.pop("config_hash", None)
    result.pop("full_config_hash", None)
    result.pop("typed_config", None)
    result.update(
        {
            "candidate_checkpoint": str(candidate),
            "candidate_checkpoint_sha256": candidate_sha256,
            "baseline_checkpoint": str(champion),
            "baseline_checkpoint_sha256": champion_sha256,
            "base_seed": intervals[0]["base_seed"],
            "effective_search_config": effective_config,
            "pairs_requested": complete_pairs,
            "games_played": len(games),
            "games_with_winner": len(games),
            "games_truncated": 0,
            "candidate_wins": sum(outcomes),
            "baseline_wins": len(outcomes) - sum(outcomes),
            "candidate_win_rate": sum(outcomes) / len(outcomes),
            "sprt": evaluate_sprt(outcomes=outcomes, elo0=-10.0, elo1=15.0),
            "pair_sprt": evaluate_sprt(outcomes=concordant, elo0=-10.0, elo1=15.0),
            "pentanomial_sprt": pentanomial,
            "verdict": pentanomial["decision"],
            # The flywheel gate is a regression-protection indifference band.
            # Its H1 is not a confidence claim that true Elo is positive.
            "gate_interpretation": {
                "schema_version": "a1-gate-interpretation-v1",
                "promotion_gate_semantics": "regression_protection",
                "promotion_elo0": -10.0,
                "promotion_elo1": 15.0,
                "h1_proves_positive_elo": False,
                "superiority_elo0": 0.0,
                "superiority_elo1": 15.0,
            },
            "superiority_pentanomial_sprt": superiority,
            "superiority_verdict": superiority["decision"],
            "pair_diagnostics": diagnostics,
            "pairs_decisive": diagnostics["ww_pairs"] + diagnostics["ll_pairs"],
            "pairs_split_excluded": diagnostics["split_pairs"],
            "pairs_truncated_excluded": diagnostics["incomplete_pairs"],
            "complete_pairs": complete_pairs,
            "split_rate": diagnostics["split_pairs"] / complete_pairs,
            "decisive_pair_yield": (diagnostics["ww_pairs"] + diagnostics["ll_pairs"])
            / complete_pairs,
            # Fleet shards execute concurrently.  Summing their durations reports
            # aggregate lane-seconds, not wall time, and inflated prior throughput
            # denominators by the number of lanes.  The slowest shard is the best
            # wall-clock approximation available in the shard schema; preserve the
            # useful summed quantity under an explicit name.
            "elapsed_sec": max(
                (float(report.get("elapsed_sec", 0.0)) for _, report in loaded),
                default=0.0,
            ),
            "aggregate_compute_sec": sum(
                float(report.get("elapsed_sec", 0.0)) for _, report in loaded
            ),
            "workers": sum(int(report.get("workers", 0)) for _, report in loaded),
            "search_telemetry": search_telemetry,
            "errors": [],
            "games": games,
            "fleet_merge": {
                "schema_version": POOL_SCHEMA,
                "kind": "internal_h2h",
                "candidate": artifacts._checkpoint_ref(candidate),  # noqa: SLF001
                "champion": artifacts._checkpoint_ref(champion),  # noqa: SLF001
                "sources": _source_refs([path for path, _ in loaded]),
                "seed_intervals": intervals,
                "disjoint_cohorts": bool(allow_disjoint_cohorts),
                "shard_config_hashes": [
                    {
                        "path": str(path),
                        "config_hash": report["config_hash"],
                        "full_config_hash": report["full_config_hash"],
                    }
                    for path, report in loaded
                ],
                "effective_search_config_sha256": promotion._digest_value(  # noqa: SLF001
                    effective_config
                ),
            },
        }
    )
    return result


def _wilson(wins: int, games: int, z: float = 1.96) -> list[float]:
    p = wins / games
    denominator = 1 + z * z / games
    center = p + z * z / (2 * games)
    half = z * ((p * (1 - p) / games + z * z / (4 * games * games)) ** 0.5)
    return [
        max(0.0, (center - half) / denominator),
        min(1.0, (center + half) / denominator),
    ]


def pool_neutral(
    paths: Sequence[Path],
    *,
    checkpoint: Path,
    allow_disjoint_cohorts: bool = False,
) -> dict[str, Any]:
    if not paths:
        raise PoolError("at least one neutral-harness report is required")
    loaded = [(path.resolve(), _load(path.resolve())) for path in paths]
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    checkpoint_md5 = promotion._md5(checkpoint)  # noqa: SLF001
    checkpoint_sha256 = promotion._sha256(checkpoint)  # noqa: SLF001
    effective_config: dict[str, Any] | None = None
    for path, report in loaded:
        validate_complete_report(report, kind="neutral", where=str(path))
        if report.get("candidate_checkpoint_md5") != checkpoint_md5:
            raise PoolError(f"{path} checkpoint MD5 drift")
        _validate_checkpoint_sha256(
            report,
            key="candidate_checkpoint_sha256",
            expected=checkpoint,
            where=f"{path} candidate",
        )
        shard_effective = _neutral_effective_search_config(report, where=str(path))
        if effective_config is None:
            effective_config = shard_effective
        elif _canonical(shard_effective) != _canonical(effective_config):
            raise PoolError(f"fleet report effective science/config drift in {path}")
        if (
            report.get("stratum") != "neutral-harness"
            or report.get("harness") != "catanatron_native_engine"
            or report.get("mode") != "search"
            or report.get("gate_config") != "flywheel"
        ):
            raise PoolError(f"{path} is not a promotion neutral-search report")
        if (
            report.get("errors") != []
            or report.get("worker_errors") != []
            or int(report.get("games_truncated", -1)) != 0
            or int(report.get("games_engine_divergence", -1)) != 0
        ):
            raise PoolError(f"{path} contains errors, truncations, or divergence")
        scores, diagnostics = _pair_diagnostics(report.get("games", []))
        replay = evaluate_pentanomial_sprt(
            scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
        )
        if replay != report.get("pentanomial_sprt") or diagnostics != report.get(
            "pair_diagnostics"
        ):
            raise PoolError(f"{path} gate statistics do not replay")
    intervals = _contiguous_intervals(loaded, allow_gaps=allow_disjoint_cohorts)
    games = _validate_and_normalize_games(loaded, kind="neutral")
    outcomes = [bool(game["candidate_won"]) for game in games]
    wins = sum(outcomes)
    scores, diagnostics = _pair_diagnostics(games)
    pentanomial = evaluate_pentanomial_sprt(
        scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    superiority = evaluate_pentanomial_sprt(
        scores, elo0=0.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    pairs = len(games) // 2
    result = copy.deepcopy(loaded[0][1])
    result.update(
        {
            "candidate_checkpoint": str(checkpoint),
            "candidate_checkpoint_md5": checkpoint_md5,
            "candidate_checkpoint_sha256": checkpoint_sha256,
            "base_seed": intervals[0]["base_seed"],
            "effective_search_config": effective_config["search_config"],
            "pairs_requested": pairs,
            "complete_pairs": pairs,
            "games_requested": len(games),
            "games_played": len(games),
            "games_with_winner": len(games),
            "games_truncated": 0,
            "games_errored": 0,
            "games_engine_divergence": 0,
            "candidate_wins": wins,
            "baseline_wins": len(outcomes) - wins,
            "candidate_win_rate": wins / len(outcomes),
            "candidate_win_rate_wilson_95ci": _wilson(wins, len(outcomes)),
            "total_illegal_policy_picks": sum(
                int(game.get("illegal_policy_picks", 0)) for game in games
            ),
            "total_search_decisions": sum(
                int(game.get("search_decisions", 0)) for game in games
            ),
            "total_simulations_used": sum(
                int(game.get("simulations_used", 0)) for game in games
            ),
            "sprt": evaluate_sprt(outcomes=outcomes, elo0=-10.0, elo1=15.0),
            "pentanomial_sprt": pentanomial,
            "verdict": pentanomial["decision"],
            "gate_interpretation": {
                "schema_version": "a1-gate-interpretation-v1",
                "promotion_gate_semantics": "regression_protection",
                "promotion_elo0": -10.0,
                "promotion_elo1": 15.0,
                "h1_proves_positive_elo": False,
                "superiority_elo0": 0.0,
                "superiority_elo1": 15.0,
            },
            "superiority_pentanomial_sprt": superiority,
            "superiority_verdict": superiority["decision"],
            "pair_diagnostics": diagnostics,
            "workers": sum(int(report.get("workers", 0)) for _, report in loaded),
            "run_fingerprint": promotion._digest_value(
                _source_refs([path for path, _ in loaded])
            ),  # noqa: SLF001
            "artifact_dir": "fleet-pooled; see fleet_merge.sources",
            "resume": {
                "enabled": False,
                "games_resumed": 0,
                "games_run_this_invocation": len(games),
            },
            "elapsed_sec": max(
                (float(report.get("elapsed_sec", 0.0)) for _, report in loaded),
                default=0.0,
            ),
            "aggregate_compute_sec": sum(
                float(report.get("elapsed_sec", 0.0)) for _, report in loaded
            ),
            "worker_errors": [],
            "errors": [],
            "games": games,
            "fleet_merge": {
                "schema_version": POOL_SCHEMA,
                "kind": "external_panel",
                "checkpoint": artifacts._checkpoint_ref(checkpoint),  # noqa: SLF001
                "sources": _source_refs([path for path, _ in loaded]),
                "seed_intervals": intervals,
                "disjoint_cohorts": bool(allow_disjoint_cohorts),
                "effective_search_config_sha256": promotion._digest_value(  # noqa: SLF001
                    effective_config["search_config"]
                ),
            },
        }
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    internal = subparsers.add_parser("internal", help="pool cross-net H2H reports")
    internal.add_argument("--report", action="append", type=Path, required=True)
    internal.add_argument("--candidate", type=Path, required=True)
    internal.add_argument("--champion", type=Path, required=True)
    internal.add_argument(
        "--allow-disjoint-cohorts",
        action="store_true",
        help=(
            "Pool non-overlapping fresh seed intervals from sequential continuations. "
            "Default remains one contiguous fleet cohort; overlaps are always refused."
        ),
    )
    internal.add_argument("--out", type=Path, required=True)
    neutral = subparsers.add_parser("neutral", help="pool neutral-harness reports")
    neutral.add_argument("--report", action="append", type=Path, required=True)
    neutral.add_argument("--checkpoint", type=Path, required=True)
    neutral.add_argument(
        "--allow-disjoint-cohorts",
        action="store_true",
        help=(
            "Pool non-overlapping replicated external cohorts. Default remains one "
            "contiguous cohort; overlaps are always refused."
        ),
    )
    neutral.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        value = (
            pool_internal(
                args.report,
                candidate=args.candidate,
                champion=args.champion,
                allow_disjoint_cohorts=bool(args.allow_disjoint_cohorts),
            )
            if args.command == "internal"
            else pool_neutral(
                args.report,
                checkpoint=args.checkpoint,
                allow_disjoint_cohorts=bool(args.allow_disjoint_cohorts),
            )
        )
        artifacts._write_new_readonly(args.out, value)  # noqa: SLF001
        print(
            json.dumps(
                {
                    "path": str(args.out.expanduser().resolve()),
                    "sha256": promotion._sha256(args.out),  # noqa: SLF001
                    "pairs": value["complete_pairs"],
                    "verdict": value["verdict"],
                },
                sort_keys=True,
            )
        )
        return 0
    except (
        PoolError,
        artifacts.ArtifactBuildError,
        OSError,
        KeyError,
        ValueError,
    ) as error:
        print(f"A1 fleet evaluation pool refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
