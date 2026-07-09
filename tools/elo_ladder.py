from __future__ import annotations

"""Maximum-likelihood Elo ladder (Bradley-Terry model) over a directory of
tools/evaluate_scoreboard.py (and single-leg tools/grade_agent.py /
evaluate_self_play.py) scoreboard JSON files.

Fits strengths pi_i via the classic Zermelo/Newman iterative
Minorization-Maximization (MM) algorithm -- no external deps (pure stdlib):

    pi_i <- (total wins by i) / sum_j [ games(i, j) / (pi_i + pi_j) ]

repeated to convergence, then rescaled so the anchor's pi == 1 (Bradley-Terry
strengths are only defined up to an overall scale). Elo is then
elo_i = 400 * log10(pi_i), so the anchor sits at elo 0 by construction.

Nodes not connected to the anchor by any nonzero-game path have no defined
relative strength and are reported separately as "unranked".

CLI: python tools/elo_ladder.py --scoreboards-dir runs/scoreboards --anchor catanatron_value

ERA TAGGING: two commits changed how the AB bots (and the vendored rules
engine underneath every bot AND every checkpoint) behave --
  - 75451aa (A2: AB teacher chance-node enumeration fix, 2026-07-02T23:26:16Z,
    immediately followed by the A4 robber-scoring fix) made the AB bots
    genuinely stronger, not just less noisy.
  - 0586f12 (A15-A17: vendored Longest Road tie/parity/winning_color fixes,
    2026-07-03T00:32:15Z) changed game-ending/scoring edge cases for BOTH
    engines, i.e. for every matchup, not just AB-bot ones.
0586f12 is strictly later, so a single cutoff at 0586f12 is sufficient --
anything at/after it is also at/after 75451aa. A checkpoint's win rate
measured before that cutoff is not comparable to one measured after it, even
though names are unchanged. Every scoreboard JSON is tagged by file mtime
relative to the cutoff ("era": "prefix-bots" or "postfix-bots") and
--era-filter lets the ladder be refit on postfix-bots-only data -- that is
now the official ladder for gating. "prefix-bots" data (including the A2/A4-
only window between 75451aa and 0586f12) is kept only for historical
comparison, never for gating.
"""

import argparse
import datetime as dt
import json
import math
import os
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

# Last rules-affecting commit (A15-A17, see module docstring). Any scoreboard
# file written at or after this instant reflects the fully-fixed engine +
# bots on every host.
POSTFIX_BOTS_ERA_CUTOFF_ISO = "2026-07-03T00:32:15+00:00"
POSTFIX_BOTS_ERA_CUTOFF_EPOCH = dt.datetime.fromisoformat(POSTFIX_BOTS_ERA_CUTOFF_ISO).timestamp()
ERA_CHOICES = ("all", "prefix-bots", "postfix-bots")


def era_for_mtime(mtime: float) -> str:
    return "postfix-bots" if mtime >= POSTFIX_BOTS_ERA_CUTOFF_EPOCH else "prefix-bots"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoreboards-dir", default="runs/scoreboards")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Root used to resolve part_files references and deduplicate double-counted parts.",
    )
    parser.add_argument("--anchor", default="catanatron_value")
    parser.add_argument(
        "--track",
        default="2p_no_trade",
        help="Only combine matchups recorded under this track name.",
    )
    parser.add_argument(
        "--era-filter",
        choices=ERA_CHOICES,
        default="all",
        help=(
            "Restrict to scoreboard files from before/after the last rules- "
            "affecting fix (commit 0586f12, 2026-07-03T00:32:15Z; supersedes "
            "the earlier A2 AB-bot fix cutoff), tagged by file mtime. "
            "'postfix-bots' is the official ladder for gating going forward; "
            "'all' (default) mixes both eras and should not be used to gate."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20000,
        help=(
            "MM iterations cap. A ~60-node ladder needs several thousand "
            "iterations to converge under --tolerance; a report that silently "
            "hits this cap without converging should be treated as unreliable."
        ),
    )
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.5,
        help="Additive smoothing per edge (avoids divide-by-zero for 0%/100% win rate legs).",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--out", help="Optional JSON output path for the full ladder report.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    matchups, skipped_tracks, era_counts = _load_matchups(
        Path(args.scoreboards_dir), repo_root=repo_root, track=args.track, era_filter=args.era_filter
    )
    report = build_ladder_report(
        matchups,
        anchor=args.anchor,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        smoothing=args.smoothing,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    report["skipped_other_tracks"] = sorted(skipped_tracks)
    report["era_filter"] = args.era_filter
    report["era_file_counts"] = era_counts
    text = json.dumps(report, indent=2, sort_keys=True)
    print(_render_table(report))
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


def _load_matchups(
    scoreboards_dir: Path, *, repo_root: Path, track: str, era_filter: str = "all"
) -> tuple[list[dict[str, Any]], set[str], dict[str, int]]:
    """Parse every *.json under scoreboards_dir into flat matchup records:
    {candidate, opponent, wins, games, game_outcomes (optional), era}.

    Handles both the multi-opponent {"results": [...]} schema written by
    evaluate_scoreboard.py and the flat single-leg schema written by
    evaluate_self_play.py / grade_agent.py's normalized reports. Files listed
    in another file's "part_files" are skipped so their games are not
    double-counted against the summary that already aggregates them.

    Each file is tagged with an "era" (see era_for_mtime) from its mtime;
    era_filter restricts the returned matchups to one era ("all" keeps both,
    but mixing eras should never be used for gating -- see module docstring).
    Returns (matchups, skipped_tracks, era_file_counts) where era_file_counts
    is a count of *scoreboard files* seen per era (before filtering).
    """
    if era_filter not in ERA_CHOICES:
        raise ValueError(f"era_filter must be one of {ERA_CHOICES}, got {era_filter!r}")
    paths = sorted(scoreboards_dir.glob("**/*.json"))
    parsed: dict[Path, Any] = {}
    for path in paths:
        try:
            parsed[path] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

    referenced_parts: set[Path] = set()
    for data in parsed.values():
        if not isinstance(data, dict):
            continue
        for part in data.get("part_files") or []:
            referenced_parts.add((repo_root / part).resolve())

    matchups: list[dict[str, Any]] = []
    skipped_tracks: set[str] = set()
    era_counts: dict[str, int] = defaultdict(int)
    for path, data in parsed.items():
        if path.resolve() in referenced_parts:
            continue
        if not isinstance(data, dict):
            continue
        try:
            era = era_for_mtime(path.stat().st_mtime)
        except OSError:
            era = "prefix-bots"  # unknown mtime: don't silently claim postfix
        era_counts[era] += 1
        if era_filter != "all" and era != era_filter:
            continue
        if isinstance(data.get("results"), list):
            top_candidate = data.get("candidate")
            for entry in data["results"]:
                if not isinstance(entry, dict) or "wins" not in entry or "games" not in entry:
                    continue
                entry_track = str(entry.get("track") or "2p_no_trade")
                if entry_track != track:
                    skipped_tracks.add(entry_track)
                    continue
                matchups.append(_matchup_record(entry, top_candidate, era=era))
        elif "wins" in data and "games" in data and "opponent" in data:
            entry_track = str(data.get("track") or "2p_no_trade")
            if entry_track != track:
                skipped_tracks.add(entry_track)
                continue
            matchups.append(_matchup_record(data, data.get("candidate"), era=era))
    return matchups, skipped_tracks, dict(era_counts)


def _canonicalize_node_id(raw: Any) -> str:
    """Normalize a candidate/opponent identifier to one canonical node id.

    FIX (node-identity split): evaluate_scoreboard.py's opponent spec parser
    prefixes checkpoint opponents with "checkpoint:" (see
    _parse_opponent_spec), but the report-level "candidate" field for that
    same checkpoint, when it plays as the candidate in a different scoreboard
    file, is the bare path with no prefix. Left unstripped, one physical
    checkpoint silently becomes two disconnected-ish Bradley-Terry nodes: a
    "candidate-side" node (which sees its losses to bots) and a
    "checkpoint:"-prefixed "opponent-side" node (which only sees the H2H
    games it was invited into) -- exactly the kind of split that produced a
    wildly inflated +280 Elo for one checkpoint that actually loses to
    catanatron_value in its candidate-side matchups. Stripping the prefix and
    running the result through os.path.normpath (harmless no-op for
    non-path bot names like "catanatron_ab3") merges both references into one
    node. This intentionally does NOT collapse genuinely different files
    (e.g. checkpoint.pt vs step_1.pt) -- only the exact same path string.
    """
    text = str(raw) if raw is not None else ""
    if text.startswith("checkpoint:"):
        text = text[len("checkpoint:") :]
    if not text:
        return "unknown_candidate"
    return os.path.normpath(text)


def _matchup_record(entry: dict[str, Any], top_candidate: Any, *, era: str = "all") -> dict[str, Any]:
    # NOTE (schema surprise): evaluate_scoreboard.py's per-result "candidate"
    # field is the short architecture label (e.g. "xdim_graph"), NOT the
    # specific checkpoint -- using it would collapse every checkpoint of one
    # architecture into a single ladder node. The report-level "candidate" is
    # the actual checkpoint path, so it is strongly preferred as node identity.
    candidate = _canonicalize_node_id(top_candidate or entry.get("candidate") or "unknown_candidate")
    outcomes = entry.get("game_outcomes")
    # NOTE (adversarial-review truncation-as-loss bias): an entry may be None
    # (truncated game, no winner) -- bool(None) is False, which would
    # silently fold a truncated game into "loss" for bootstrap resampling.
    # Preserve None here; _build_win_tables filters it out before it ever
    # reaches the resampler.
    game_outcomes = (
        [o if o is None else bool(o) for o in outcomes] if isinstance(outcomes, list) else None
    )
    return {
        "candidate": candidate,
        "opponent": _canonicalize_node_id(entry["opponent"]),
        "wins": int(entry["wins"]),
        "games": int(entry["games"]),
        "game_outcomes": game_outcomes,
        "era": era,
    }


def _build_win_tables(
    matchups: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, list[bool]]]]:
    """Aggregate matchup records into wins[i][j] (i beat j count) and, where
    available, the concatenated ordered per-game outcomes for (i, j).

    FIX (adversarial-review truncation-as-loss bias): individual
    game_outcomes entries may be None (truncated game, no winner). These
    carry no win/loss information for bootstrap resampling and are dropped
    here rather than passed through, so _resample_wins never has to treat
    a falsy None as a loss.
    """
    wins: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    outcomes: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for record in matchups:
        i, j = record["candidate"], record["opponent"]
        if i == j:
            continue
        wins[i][j] += record["wins"]
        wins[j][i] += record["games"] - record["wins"]
        if record["game_outcomes"] is not None:
            outcomes[i][j].extend(o for o in record["game_outcomes"] if o is not None)
    return wins, outcomes


def _connected_component(
    nodes: list[str], wins: dict[str, dict[str, float]], anchor: str
) -> set[str]:
    if anchor not in nodes:
        return set()
    adjacency: dict[str, set[str]] = defaultdict(set)
    for i in nodes:
        for j, wij in wins.get(i, {}).items():
            wji = wins.get(j, {}).get(i, 0.0)
            if wij + wji > 0:
                adjacency[i].add(j)
                adjacency[j].add(i)
    seen = {anchor}
    queue = deque([anchor])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, ()):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def fit_bradley_terry(
    wins: dict[str, dict[str, float]],
    nodes: list[str],
    *,
    anchor: str,
    max_iterations: int = 20000,
    tolerance: float = 1e-10,
    smoothing: float = 0.5,
) -> tuple[dict[str, float], int]:
    """Zermelo/Newman MM fit of Bradley-Terry strengths pi_i, rescaled so
    pi[anchor] == 1. Returns (pi, iterations_used). Nodes with no path of
    nonzero-game edges to any other node keep pi == 1 (undefined strength)."""
    games: dict[str, dict[str, float]] = {i: {} for i in nodes}
    smoothed_wins: dict[str, dict[str, float]] = {i: {} for i in nodes}
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            wij = wins.get(i, {}).get(j, 0.0)
            wji = wins.get(j, {}).get(i, 0.0)
            n = wij + wji
            if n <= 0:
                continue
            smoothed_wins[i][j] = wij + smoothing
            games[i][j] = n + 2.0 * smoothing

    pi = {node: 1.0 for node in nodes}
    iterations_used = 0
    for iteration in range(max_iterations):
        iterations_used = iteration + 1
        new_pi = {}
        for i in nodes:
            if not games[i]:
                new_pi[i] = pi[i]
                continue
            numerator = sum(smoothed_wins[i].values())
            denominator = sum(games[i][j] / (pi[i] + pi[j]) for j in games[i])
            new_pi[i] = numerator / denominator if denominator > 0 else pi[i]
        scale = new_pi.get(anchor, 1.0)
        if scale <= 0:
            scale = 1.0
        new_pi = {k: v / scale for k, v in new_pi.items()}
        delta = max(
            abs(math.log(new_pi[k]) - math.log(pi[k]))
            for k in nodes
            if new_pi[k] > 0 and pi[k] > 0
        )
        pi = new_pi
        if delta < tolerance:
            break
    return pi, iterations_used


def _resample_wins(
    wins: dict[str, dict[str, float]],
    outcomes: dict[str, dict[str, list[bool]]],
    nodes: list[str],
    rng: random.Random,
) -> dict[str, dict[str, float]]:
    """Bootstrap-resample win counts over games for one replicate: for edges
    with recorded per-game outcomes, resample games with replacement
    (non-parametric); otherwise resample a binomial count at the observed
    win rate (parametric), since historical scoreboards only have aggregates.
    """
    resampled: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    seen_pairs: set[tuple[str, str]] = set()
    for i in nodes:
        for j, wij in wins.get(i, {}).items():
            pair = tuple(sorted((i, j)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            wji = wins.get(j, {}).get(i, 0.0)
            n = int(round(wij + wji))
            if n <= 0:
                continue
            leg_outcomes = outcomes.get(i, {}).get(j)
            if leg_outcomes:
                sample = [rng.choice(leg_outcomes) for _ in range(len(leg_outcomes))]
                i_wins = sum(1 for won in sample if won)
                total = len(sample)
            else:
                p = wij / n
                i_wins = sum(1 for _ in range(n) if rng.random() < p)
                total = n
            resampled[i][j] = i_wins
            resampled[j][i] = total - i_wins
    return resampled


def build_ladder_report(
    matchups: list[dict[str, Any]],
    *,
    anchor: str,
    max_iterations: int = 20000,
    tolerance: float = 1e-10,
    smoothing: float = 0.5,
    bootstrap_samples: int = 100,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    wins, outcomes = _build_win_tables(matchups)
    nodes = sorted(set(wins) | {opponent for row in wins.values() for opponent in row})
    if anchor not in nodes:
        nodes.append(anchor)
    connected = _connected_component(nodes, wins, anchor)

    pi, iterations_used = fit_bradley_terry(
        wins, nodes, anchor=anchor, max_iterations=max_iterations, tolerance=tolerance, smoothing=smoothing
    )

    bootstrap_elos: dict[str, list[float]] = defaultdict(list)
    if bootstrap_samples > 0:
        rng = random.Random(bootstrap_seed)
        for _ in range(bootstrap_samples):
            resampled_wins = _resample_wins(wins, outcomes, nodes, rng)
            boot_pi, _ = fit_bradley_terry(
                resampled_wins,
                nodes,
                anchor=anchor,
                max_iterations=max_iterations,
                tolerance=tolerance,
                smoothing=smoothing,
            )
            for node in connected:
                if boot_pi.get(node, 0.0) > 0:
                    bootstrap_elos[node].append(400.0 * math.log10(boot_pi[node]))

    total_games: dict[str, int] = defaultdict(int)
    for i in nodes:
        for j, wij in wins.get(i, {}).items():
            total_games[i] += int(wij + wins.get(j, {}).get(i, 0.0))

    ladder = []
    unranked = []
    for node in nodes:
        games_played = total_games.get(node, 0)
        if node in connected and pi.get(node, 0.0) > 0:
            elo = 400.0 * math.log10(pi[node])
            samples = sorted(bootstrap_elos.get(node, []))
            if samples:
                lower = samples[max(0, int(0.025 * len(samples)))]
                upper = samples[min(len(samples) - 1, int(0.975 * len(samples)))]
            else:
                lower = upper = None
            ladder.append(
                {
                    "node": node,
                    "elo": elo,
                    "elo_ci95_lower": lower,
                    "elo_ci95_upper": upper,
                    "games": games_played,
                    "is_anchor": node == anchor,
                }
            )
        else:
            unranked.append({"node": node, "games": games_played, "reason": "disconnected from anchor"})

    ladder.sort(key=lambda row: row["elo"], reverse=True)
    return {
        "anchor": anchor,
        "iterations_used": iterations_used,
        "bootstrap_samples": bootstrap_samples,
        "smoothing": smoothing,
        "nodes_ranked": len(ladder),
        "nodes_unranked": len(unranked),
        "ladder": ladder,
        "unranked": unranked,
    }


def _render_table(report: dict[str, Any]) -> str:
    lines = [
        f"Elo ladder (anchor={report['anchor']}=0, {report['bootstrap_samples']} bootstrap "
        f"samples, {report['iterations_used']} MM iterations)",
        f"{'node':<70} {'elo':>8} {'ci95_lower':>11} {'ci95_upper':>11} {'games':>8}",
    ]
    for row in report["ladder"]:
        lower = f"{row['elo_ci95_lower']:.1f}" if row["elo_ci95_lower"] is not None else "n/a"
        upper = f"{row['elo_ci95_upper']:.1f}" if row["elo_ci95_upper"] is not None else "n/a"
        lines.append(
            f"{row['node']:<70} {row['elo']:>8.1f} {lower:>11} {upper:>11} {row['games']:>8}"
        )
    if report["unranked"]:
        lines.append("")
        lines.append("Unranked (disconnected from anchor):")
        for row in report["unranked"]:
            lines.append(f"  {row['node']} (games={row['games']}, {row['reason']})")
    if report.get("skipped_other_tracks"):
        lines.append("")
        lines.append(f"Skipped tracks (not matching --track): {report['skipped_other_tracks']}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
