from __future__ import annotations

r"""Whole-History Rating (Coulom 2008) ladder fit over ALL gate/panel/H2H
result JSONs on disk, stratified (Linear CAT-21).

WHY THIS EXISTS. Every promotion gate, external panel, and H2H ablation arm
is evaluated in ISOLATION (its own SPRT, its own Elo delta vs one baseline).
The reported per-generation Elo-gain sequence (+49 -> +49 -> +33 -> +20) may
be a real compounding-diminishing-returns trend, or it may just be an
artifact of overlapping confidence intervals across gates that were never
pooled into one joint model. This tool ingests every such result file and
fits ONE Whole-History Rating model per stratum (see `pip install whr`,
Remi Coulom's algorithm: a Bayesian dynamic Bradley-Terry model where each
player's strength is a smoothly-evolving curve over integer "day" time
steps, regularized by `w2`, the variance of day-to-day rating drift), then
reports the per-checkpoint trajectory with real WHR uncertainty (posterior
stddev of Elo) so a reviewer can directly check whether consecutive
generations' 95% CIs overlap -- i.e. whether the "compression trend" is
real or noise.

===========================================================================
DESIGN DECISION 1 -- the champion lineage is ONE evolving WHR player.
===========================================================================
WHR models a single named player's rating evolving over calendar time via
games against various (possibly also-evolving, possibly static) opponents.
In this codebase "the champion" is a sequence of DISTINCT, immutable
checkpoint files (checkpoint.pt never changes after training; a new
generation is a new file). To get the "does Elo-gain-per-generation shrink"
trajectory the ticket asks for, every game a lineage checkpoint plays is
recorded under ONE shared WHR player name (`--champion-name`, default
"champion_lineage"), at a time_step equal to that checkpoint's parsed
generation ordinal (see Decision 2). Static opponents (catanatron_* bots,
external engines, a frozen BC hard-target anchor checkpoint, etc.) are kept
as their OWN distinct WHR players -- exactly the "static player observed at
a handful of days" case WHR already supports natively (Coulom's own worked
example plays a fixed handicap-less opponent across widely spaced days).

A checkpoint identity is judged "part of the lineage" by the exact same
heuristic used to resolve its time_step (Decision 2): if a generation/step
ordinal is parseable from its path, it is a lineage node and collapses to
`champion_name`; if not, it is treated as a static, non-collapsing node
(canonicalized via `tools.elo_ladder._canonicalize_node_id`, reused
verbatim -- see Decision 3).

DEGENERATE CASE (flagged, not silently mishandled): promotion_gate_runner.py
and gumbel_search_cross_net_h2h.py play ONE lineage checkpoint (candidate,
e.g. gen3) directly against ANOTHER lineage checkpoint (baseline, e.g.
gen2). Collapsing BOTH sides to `champion_name` would create a degenerate
WHR self-game (WHR has no "player faces themself" concept -- exactly like
`tools.elo_ladder._connected_component`'s `if i == j: continue`). Per
`resolve_pair_identities` below: the CANDIDATE side (always the newer/higher
checkpoint by convention in every H2H tool in this repo) collapses to
`champion_name`; the OPPONENT/baseline side is kept as its own distinct,
un-collapsed checkpoint node. This preserves the game as real signal without
self-play, at the documented cost that a prior-generation checkpoint's own
node does not itself accumulate into the lineage curve FROM THIS GAME (it
still gets its own games elsewhere if it was itself once a "candidate").
This is a judgment call, not a full fix -- flagged here per the ticket's
explicit ask, not chased to perfection.

The search-vs-raw-policy family (Family B, `gumbel_search_vs_raw_h2h.py`)
compares ONE checkpoint's searched play against THAT SAME checkpoint's raw
(no-search) policy. To keep these two roles from colliding into the same
WHR node (which would either self-collapse if lineage-parseable, or overlap
if not), the raw-policy side is always given a `"::raw_policy"` suffix
before identity resolution, forcing it to remain a distinct, per-generation,
non-collapsing node even when the checkpoint path IS lineage-parseable. The
searched side still collapses to `champion_name` when lineage-parseable, so
"how good is search on top of the current champion" is trended over
generations by champion_name's trajectory, while "which specific
generation's raw policy" stays checkpoint-specific (each raw_policy
opponent is its own snapshot, which is the more correct framing anyway --
raw-policy strength is a property of one frozen net, not an evolving one).

===========================================================================
DESIGN DECISION 2 -- time_step ("day") heuristic. FLAGGED LOUDLY: this
directly controls how wide the reported uncertainty bands are.
===========================================================================
WHR's `time_step` is an arbitrary monotonic integer "day" index -- it need
not be a real calendar day, but LARGE spurious gaps between two time_steps
that are actually close in reality make WHR treat them as having had more
opportunity to drift (inflating uncertainty and dampening apparent
continuity), while collapsing two genuinely different days onto the same
time_step overstates confidence (WHR then treats them as literally
simultaneous observations). The parser tries, IN ORDER, to pull a
generation/step/epoch ordinal out of the checkpoint/candidate path:
  1. r"gen(\d+)([a-zA-Z]?)"   e.g. "gen2A" -> 2*100 + 1 = 201, "gen3" -> 300
  2. r"step[_-]?(\d+)"        e.g. "step_20000" -> 20000
  3. r"epoch[_-]?(\d+)"       e.g. "bc_epoch3" -> 3
If NONE match, time_step falls back to the JSON file's mtime bucketed to a
day (`int(mtime // 86400)`, `time_step_source="mtime_fallback"`). This is a
real, LOUD caveat: ordinal-parsed generations are small integers (1, 2, 3,
...) while mtime-fallback days are ~epoch-day integers in the tens of
thousands, so if the SAME lineage identity ever mixes both sources within
one stratum, the resulting "trajectory" has one cluster of tiny time_steps
and one huge outlier day far in the future of WHR's internal day axis --
this does not corrupt individual ratings (each is still fit from its own
games) but DOES weaken the drift-smoothing prior's ability to relate them,
which is exactly the kind of thing that affects the reported error bars.
The ingest report surfaces `time_step_sources` per stratum (counts of
parsed_ordinal vs mtime_fallback records) specifically so this is auditable
rather than silently baked into the numbers.

===========================================================================
DESIGN DECISION 3 -- node canonicalization is REUSED, not reinvented.
===========================================================================
`_canonicalize_node_id` is imported directly from `tools.elo_ladder` (same
function, not a copy) -- it strips a leading "checkpoint:" prefix (added by
evaluate_scoreboard.py's opponent-spec parser but absent from the same
checkpoint's report-level "candidate" field) and normalizes the path, fixing
the exact node-identity split bug documented there (one physical checkpoint
silently becoming two disconnected Bradley-Terry nodes, inflating one of
them by +280 Elo). Reusing it here means this tool cannot reintroduce that
bug by drifting out of sync with a hand-copied version.

===========================================================================
DESIGN DECISION 4 -- pentanomial-aggregate-only records need an
approximation. FLAGGED LOUDLY: this is a variance-narrowing approximation.
===========================================================================
Most Family B tools (`gumbel_search_vs_raw_h2h.py`,
`gumbel_search_vs_bot_h2h.py`, `gumbel_search_cross_net_h2h.py`) write their
FULL per-game list (`"games": [...]`) to disk, so real per-game B/W outcomes
are fed to WHR directly for those files -- no approximation needed, and each
individual game (including "split" color-swapped pairs) is real signal.

However, the FLEET AGGREGATORS (`tools/h2h_v3conf_aggregate.py`,
`tools/h2h_postrepair_aggregate.py`) pool many such files and, when run with
`--out`, persist ONLY the reduced pentanomial pair counts (`ll_pairs`,
`split_pairs`, `ww_pairs` from `sprt_gate.evaluate_pentanomial_sprt`'s
output shape) -- the per-game list is not re-emitted. If ONLY this reduced
form is on disk (the per-game source files having been cleaned up, or only
the aggregate having been archived), `pentanomial_pairs_to_synthetic_games`
below expands each pair count into TWO synthetic per-game outcomes that
exactly preserve the pair's mean score:
    WW pair (score 1.0)   -> two synthetic wins   (True, True)
    split pair (score 0.5)-> one win, one loss     (True, False)
    LL pair (score 0.0)   -> two synthetic losses  (False, False)
This is an APPROXIMATION, not real data, and it has a known, one-directional
effect on the reported uncertainty: treating the two synthetic games of a
pair as independent Bernoulli trials UNDERSTATES their true correlation (a
color-swapped pair's two legs share the same seed and are the same net vs
the same opponent, minus color -- they are not independent draws), so the
effective sample size this approximation feeds WHR is inflated relative to
the pair count, and the resulting CI on the affected node/time_step is
NARROWER than it should be. Concretely: N complete pairs become 2N
"games" of correlated-in-reality but modeled-as-independent outcomes: the
true effective sample size is closer to N (one independent draw per pair)
than 2N. Callers who need conservative bars should treat any time_step
built substantially from this path with that in mind; the ingest report
tags exactly which stratum/time_step combinations used this path
(`used_pentanomial_approximation` count) so it is auditable per the
ticket's explicit ask.

===========================================================================
STRATIFICATION (R9) -- 5 required strata, tagging heuristic + known gaps.
===========================================================================
  low_n_internal          checkpoint-vs-checkpoint comparison, < LOW_N_GAMES_THRESHOLD
                           (200) games represented in the source record.
  production_n_internal    checkpoint-vs-checkpoint comparison, >= 200 games.
                           200 is taken directly from sprt_gate.py's own
                           docstring reference point ("a 60%-over-200-games
                           sample will usually land in 'continue'") -- i.e.
                           200 games is the project's own documented rough
                           floor for a gate result to carry real information
                           at typical promotion effect sizes.
  external_catanatron      a genuinely-external Catanatron engine harness,
                           as distinct from this repo's INTERNAL
                           catanatron_value / catanatron_ab3/4/5 bot
                           baselines. GAP, FLAGGED PER TICKET: no file in
                           this codebase (evaluate_scoreboard.py,
                           gumbel_search_vs_bot_h2h.py, sprt_gate.py,
                           promotion_gate_runner.py) records any metadata
                           field distinguishing an "external" catanatron
                           harness invocation from these internal bot
                           baselines -- BOT_KINDS in
                           gumbel_search_vs_bot_h2h.py and
                           KNOWN_BOT_KINDS in promotion_gate_runner.py list
                           only the internal bots. `_EXTERNAL_MARKER_HINTS`
                           below is a placeholder pattern (opponent name
                           containing "external") that would route a
                           correctly-tagged future record here IF such a
                           naming convention is ever added upstream; until
                           then this stratum is expected to be EMPTY on any
                           real run, and that emptiness is itself the
                           correct, honest signal that the distinguishing
                           metadata does not exist yet -- not a bug in this
                           tool.
  neutral_harness          the internal catanatron_* bot baselines
                           (catanatron_value, catanatron_ab3/4/5, and the
                           G2-roster value-function specialists
                           catanatron_value_ore_city/road_race/robber) and
                           any other recognized non-neural bot kind (mirrors
                           `tools.promotion_gate_runner.KNOWN_BOT_KINDS`).
  raw_policy               the `gumbel_search_vs_raw_h2h.py` family: search
                           vs the SAME checkpoint's raw (no-search) policy.
                           Forced at record-construction time (this family
                           is unambiguous from its own JSON shape), not
                           inferred from the opponent name.

opening_panel.py's output (`runs/panels/opening_200.json` builds and
per-root eval reports) is INTENTIONALLY NOT ingested here: it records
per-root ranking-quality diagnostics (Kendall tau, top-1 regret, flip rate)
against a fixed frozen panel, not a win/loss game against a named opponent
-- there is no second "player" for WHR to rate against. Skimmed per the
ticket's own allowance ("you likely don't need to replicate the era-cutoff
logic itself... your call, but note why").

era-tagging (the prefix-bots/postfix-bots rules-engine-fix cutoff in
tools/elo_ladder.py) is NOT replicated here either, for the same reason the
ticket flags as acceptable to skip: it is an engine-correctness workaround
for one specific comparability bug window, not something WHR's time-drift
model needs to represent -- WHR already treats every observation as
attached to its own time_step, so a rules change mid-history shows up (if
anything) as a legitimate strength discontinuity at that point in time
rather than a silent double-count. A follow-up could feed `era` through as
an additional stratum axis if the rules-change window turns out to
materially confound a specific report.

===========================================================================
KNOWN GAP -- NO REAL DATA IN THIS CLONE.
===========================================================================
This repository clone has ZERO real gate/panel/H2H JSON result files on
disk (`logs/` is empty; no fixtures under `runs/`). The actual outputs live
on the GPU training hosts referenced in project history (B200, A100A,
A100B), not here. Every test in `tests/test_whr_ladder.py` is therefore
built on SYNTHETIC fixtures that mirror the exact on-disk schemas read
above. Real-data validation -- confirming the fitted per-checkpoint Elo
ordering is sane against known head-to-head results, and actually answering
whether the +49/+49/+33/+20 compression is real -- still needs someone to
run `tools/whr_ladder.py --runs-dir <path>` against the hosts' real
`runs/scoreboards`, `runs/h2h_*`, `runs/promotion_gate_*` style directories.

CLI:
  python tools/whr_ladder.py --runs-dir runs/scoreboards --runs-dir runs/h2h_v3conf \\
      --champion-name champion_lineage --out runs/whr/report.json \\
      --out-markdown runs/whr/report.md
"""

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reused verbatim (see Decision 3 in the module docstring) -- NOT copied.
from tools.elo_ladder import _canonicalize_node_id  # noqa: E402

# Mirrors the recognized non-neural bot-kind set used for promotion gates
# (see the neutral_harness entry in the STRATIFICATION docstring section).
from tools.promotion_gate_runner import KNOWN_BOT_KINDS  # noqa: E402

STRATA: tuple[str, ...] = (
    "low_n_internal",
    "production_n_internal",
    "external_catanatron",
    "neutral_harness",
    "raw_policy",
)

# See STRATIFICATION docstring section: 200 is sprt_gate.py's own documented
# rough floor for a gate result to carry real information at typical
# promotion effect sizes ("a 60%-over-200-games sample will usually land in
# 'continue'").
LOW_N_GAMES_THRESHOLD = 200

# Placeholder pattern for a genuinely-external Catanatron harness -- GAP,
# see the external_catanatron docstring entry. No file in this codebase
# currently emits a name matching this; it exists so a future upstream
# naming convention routes correctly without code changes here.
_EXTERNAL_MARKER_HINTS: tuple[str, ...] = ("external", "engine_external")

_ORDINAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"gen(\d+)([a-zA-Z]?)", re.IGNORECASE),
    re.compile(r"step[_-]?(\d+)", re.IGNORECASE),
    re.compile(r"epoch[_-]?(\d+)", re.IGNORECASE),
)


# ---------------------------------------------------------------------------
# internal record type
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GameRecord:
    """One atomic WHR game: exactly one `whr.Base.create_game(...)` call.

    `n_games_represented` is always 1 on an emitted record -- each record IS
    one WHR game by construction. It is kept in the schema (per the ticket's
    required internal record shape) to make explicit that a single *source*
    entry (an aggregate {"wins": W, "games": G} scoreboard row, or a
    pentanomial pair-count triple) may fan out into MANY of these atomic
    records at ingest time; see `expand_aggregate_wins_losses` and
    `pentanomial_pairs_to_synthetic_games`.
    """

    player_a: str
    player_b: str
    winner_of_player_a: bool | None  # None = draw/truncated/discard, never fed to WHR
    time_step_source: str  # "parsed_ordinal" | "mtime_fallback"
    time_step: int
    stratum: str
    source_file: str
    n_games_represented: int = 1


@dataclass
class IngestStats:
    files_seen: int = 0
    files_malformed: int = 0
    files_skipped_part_file: int = 0
    files_family_a: int = 0
    files_family_b_per_game: int = 0
    files_family_b_pentanomial_only: int = 0
    files_unrecognized: int = 0
    files_opening_panel_skipped: int = 0
    games_emitted: int = 0
    games_discarded_no_winner: int = 0
    used_pentanomial_approximation: int = 0
    identity_from_arm_label_gap: int = 0
    per_stratum_game_counts: dict[str, int] = field(default_factory=dict)
    # Mirrors tools.elo_ladder.py's "skipped_other_tracks" report field: Family
    # A entries whose "track" doesn't match --track are dropped in
    # _family_a_records, and without this they'd vanish with no record
    # anywhere in the ingest report (unlike every other discard path above,
    # which all have a counter).
    skipped_other_tracks: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_seen": self.files_seen,
            "files_malformed": self.files_malformed,
            "files_skipped_part_file": self.files_skipped_part_file,
            "files_family_a": self.files_family_a,
            "files_family_b_per_game": self.files_family_b_per_game,
            "files_family_b_pentanomial_only": self.files_family_b_pentanomial_only,
            "files_unrecognized": self.files_unrecognized,
            "files_opening_panel_skipped": self.files_opening_panel_skipped,
            "games_emitted": self.games_emitted,
            "games_discarded_no_winner": self.games_discarded_no_winner,
            "used_pentanomial_approximation": self.used_pentanomial_approximation,
            "identity_from_arm_label_gap": self.identity_from_arm_label_gap,
            "per_stratum_game_counts": dict(self.per_stratum_game_counts),
            "skipped_other_tracks": sorted(self.skipped_other_tracks),
        }


# ---------------------------------------------------------------------------
# file discovery
# ---------------------------------------------------------------------------
def walk_result_files(root: Path) -> list[Path]:
    """Every *.json under `root` (recursive), sorted for determinism."""
    if not root.exists():
        return []
    return sorted(p for p in root.glob("**/*.json") if p.is_file())


# ---------------------------------------------------------------------------
# time_step / lineage-identity heuristic (Decision 2)
# ---------------------------------------------------------------------------
def parse_ordinal_time_step(path_like: str) -> int | None:
    """Pull a generation/step/epoch ordinal out of a path-like string, per
    the DESIGN DECISION 2 precedence: gen(N)(letter) > step_N > epoch_N.
    Returns None if none match (caller falls back to file mtime)."""
    text = str(path_like)
    for pattern in _ORDINAL_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        if pattern is _ORDINAL_PATTERNS[0]:
            base = int(match.group(1))
            letter = match.group(2)
            suffix = (ord(letter.lower()) - ord("a") + 1) if letter else 0
            return base * 100 + suffix
        return int(match.group(1))
    return None


def is_lineage_identity(path_like: str) -> bool:
    """Same heuristic used for time_step parsing decides lineage-collapse
    eligibility (Decision 1): an identity is "part of the champion lineage"
    iff a generation/step/epoch ordinal is parseable from it."""
    return parse_ordinal_time_step(path_like) is not None


def resolve_time_step(*, identity_hint: str, file_path: Path) -> tuple[int, str]:
    """Resolve one game's/file's time_step: parsed ordinal from
    `identity_hint` (normally the candidate/checkpoint identity), else the
    JSON file's mtime bucketed to a day (see Decision 2 for the caveats)."""
    ordinal = parse_ordinal_time_step(identity_hint)
    if ordinal is not None:
        return ordinal, "parsed_ordinal"
    try:
        mtime = file_path.stat().st_mtime
    except OSError:
        mtime = time.time()
    return int(mtime // 86400), "mtime_fallback"


def resolve_pair_identities(
    candidate_raw: str, opponent_raw: str, *, champion_name: str
) -> tuple[str, str]:
    """Map a (candidate, opponent) identity pair to final WHR player names,
    collapsing lineage-parseable identities into `champion_name` (Decision
    1), with the degenerate lineage-vs-lineage case handled per the module
    docstring: when BOTH sides are lineage-parseable, only the candidate
    side collapses (the opponent/baseline is kept as its own distinct node)
    to avoid a self-game."""
    candidate_is_lineage = is_lineage_identity(candidate_raw)
    opponent_is_lineage = is_lineage_identity(opponent_raw)
    if candidate_is_lineage and opponent_is_lineage:
        return champion_name, _canonicalize_node_id(opponent_raw)
    candidate_name = champion_name if candidate_is_lineage else _canonicalize_node_id(candidate_raw)
    opponent_name = champion_name if opponent_is_lineage else _canonicalize_node_id(opponent_raw)
    return candidate_name, opponent_name


# ---------------------------------------------------------------------------
# stratification (R9)
# ---------------------------------------------------------------------------
def classify_stratum(opponent_raw: str, *, n_games: int, forced: str | None = None) -> str:
    """Tag a matchup into exactly one of STRATA. `forced` lets a caller that
    already knows the matchup family (e.g. search-vs-raw-policy) skip the
    name-based heuristic entirely."""
    if forced is not None:
        if forced not in STRATA:
            raise ValueError(f"forced stratum {forced!r} not in {STRATA}")
        return forced
    name = str(opponent_raw).lower()
    if any(marker in name for marker in _EXTERNAL_MARKER_HINTS):
        return "external_catanatron"
    if "catanatron" in name or name in KNOWN_BOT_KINDS:
        return "neutral_harness"
    return "low_n_internal" if n_games < LOW_N_GAMES_THRESHOLD else "production_n_internal"


# ---------------------------------------------------------------------------
# pentanomial approximation (Decision 4) -- standalone + testable
# ---------------------------------------------------------------------------
def pentanomial_pairs_to_synthetic_games(n_ll: int, n_split: int, n_ww: int) -> list[bool]:
    """Expand aggregate (LL, split, WW) pair counts into synthetic per-game
    win/loss booleans (True = the "candidate"/search side won that
    synthetic game), preserving the pair's exact mean score:

        WW pair (score 1.0)    -> [True, True]
        split pair (score 0.5) -> [True, False]
        LL pair (score 0.0)    -> [False, False]

    Example: pentanomial_pairs_to_synthetic_games(n_ll=1, n_split=2, n_ww=3)
    -> 3*[T,T] + 1*[F,F] + 2*[T,F], i.e. 12 synthetic games with mean score
    (3*1.0 + 2*0.5 + 1*0.0) / 6 == 4.0/6 == the original mean pair score.

    APPROXIMATION CAVEAT (see Decision 4): the two synthetic games of one
    pair are modeled as independent Bernoulli draws, but a real color-swapped
    pair's two legs share a seed and are not independent -- this UNDERSTATES
    correlation and so OVERSTATES effective sample size (2N games fed to WHR
    for N real pairs, vs. the ~N independent-draws-equivalent a conservative
    model would use), making any CI built substantially from this path
    narrower than it should be. Prefer real per-game records when available
    (see `used_pentanomial_approximation` in the ingest report).
    """
    games: list[bool] = []
    games.extend([True, True] * int(n_ww))
    games.extend([False, False] * int(n_ll))
    games.extend([True, False] * int(n_split))
    return games


# ---------------------------------------------------------------------------
# Family A (tools/evaluate_scoreboard.py / grade_agent.py / evaluate_self_play.py)
# ---------------------------------------------------------------------------
def _collect_referenced_parts(parsed: dict[Path, Any], *, repo_root: Path) -> set[Path]:
    referenced: set[Path] = set()
    for data in parsed.values():
        if not isinstance(data, dict):
            continue
        for part in data.get("part_files") or []:
            referenced.add((repo_root / part).resolve())
    return referenced


def expand_aggregate_wins_losses(wins: int, games: int) -> list[bool]:
    """A Family A entry with only aggregate {"wins": W, "games": G} (no
    per-game `game_outcomes` breakdown) is expanded into exactly W
    True + (G - W) False synthetic per-game records -- this is an EXACT
    expansion (no information lost or invented; WHR needs a day + outcome
    per game, not game order within a day), unlike the pentanomial
    approximation above."""
    wins = int(wins)
    losses = int(games) - wins
    return [True] * max(0, wins) + [False] * max(0, losses)


def _family_a_records(
    data: dict[str, Any],
    *,
    path: Path,
    repo_root: Path,
    champion_name: str,
    track_filter: str | None,
    stats: IngestStats,
) -> list[GameRecord]:
    records: list[GameRecord] = []
    top_candidate = data.get("candidate")

    def _one_entry(entry: dict[str, Any], candidate_raw: Any) -> None:
        if not isinstance(entry, dict) or "wins" not in entry or "games" not in entry or "opponent" not in entry:
            return
        entry_track = str(entry.get("track") or "2p_no_trade")
        if track_filter is not None and entry_track != track_filter:
            stats.skipped_other_tracks.add(entry_track)
            return
        candidate_str = str(candidate_raw if candidate_raw is not None else entry.get("candidate") or "unknown_candidate")
        opponent_str = str(entry["opponent"])
        player_a, player_b = resolve_pair_identities(candidate_str, opponent_str, champion_name=champion_name)
        time_step, time_step_source = resolve_time_step(identity_hint=candidate_str, file_path=path)
        games_n = int(entry["games"])
        stratum = classify_stratum(opponent_str, n_games=games_n)

        outcomes = entry.get("game_outcomes")
        if isinstance(outcomes, list):
            per_game = [o if o is None else bool(o) for o in outcomes]
        else:
            per_game = expand_aggregate_wins_losses(int(entry["wins"]), games_n)
        for outcome in per_game:
            records.append(
                GameRecord(
                    player_a=player_a,
                    player_b=player_b,
                    winner_of_player_a=outcome,
                    time_step_source=time_step_source,
                    time_step=time_step,
                    stratum=stratum,
                    source_file=str(path),
                )
            )

    if isinstance(data.get("results"), list):
        for entry in data["results"]:
            _one_entry(entry, top_candidate)
    elif "wins" in data and "games" in data and "opponent" in data:
        _one_entry(data, top_candidate)
    return records


# ---------------------------------------------------------------------------
# Family B -- raw per-game H2H outputs (real per-game records, no approximation)
# ---------------------------------------------------------------------------
def _is_family_b_per_game(data: dict[str, Any]) -> bool:
    games = data.get("games")
    if not isinstance(games, list) or not games:
        return False
    first = games[0]
    return isinstance(first, dict) and ("search_won" in first or "candidate_won" in first)


def _family_b_per_game_records(
    data: dict[str, Any], *, path: Path, champion_name: str
) -> list[GameRecord]:
    records: list[GameRecord] = []
    games = data["games"]

    if "checkpoint" in data and "candidate_checkpoint" not in data and "baseline_checkpoint" not in data:
        # gumbel_search_vs_raw_h2h.py: search vs the SAME checkpoint's raw policy.
        checkpoint = str(data["checkpoint"])
        candidate_raw = checkpoint
        opponent_raw = checkpoint + "::raw_policy"
        player_a, player_b = resolve_pair_identities(candidate_raw, opponent_raw, champion_name=champion_name)
        time_step, time_step_source = resolve_time_step(identity_hint=candidate_raw, file_path=path)
        stratum = "raw_policy"
        for game in games:
            won = game.get("search_won")
            records.append(
                GameRecord(
                    player_a=player_a,
                    player_b=player_b,
                    winner_of_player_a=None if won is None else bool(won),
                    time_step_source=time_step_source,
                    time_step=time_step,
                    stratum=stratum,
                    source_file=str(path),
                )
            )
        return records

    if "candidate_checkpoint" in data and "baseline_bot" in data:
        # gumbel_search_vs_bot_h2h.py: candidate checkpoint vs a hardcoded Catanatron bot.
        candidate_raw = str(data["candidate_checkpoint"])
        opponent_raw = str(data["baseline_bot"])
        player_a, player_b = resolve_pair_identities(candidate_raw, opponent_raw, champion_name=champion_name)
        time_step, time_step_source = resolve_time_step(identity_hint=candidate_raw, file_path=path)
        n_games = len(games)
        stratum = classify_stratum(opponent_raw, n_games=n_games)
        for game in games:
            won = game.get("candidate_won")
            records.append(
                GameRecord(
                    player_a=player_a,
                    player_b=player_b,
                    winner_of_player_a=None if won is None else bool(won),
                    time_step_source=time_step_source,
                    time_step=time_step,
                    stratum=stratum,
                    source_file=str(path),
                )
            )
        return records

    if "candidate_checkpoint" in data and "baseline_checkpoint" in data:
        # gumbel_search_cross_net_h2h.py: candidate checkpoint vs baseline checkpoint, both searched.
        candidate_raw = str(data["candidate_checkpoint"])
        opponent_raw = str(data["baseline_checkpoint"])
        player_a, player_b = resolve_pair_identities(candidate_raw, opponent_raw, champion_name=champion_name)
        time_step, time_step_source = resolve_time_step(identity_hint=candidate_raw, file_path=path)
        n_games = len(games)
        stratum = classify_stratum(opponent_raw, n_games=n_games)
        for game in games:
            won = game.get("candidate_won")
            records.append(
                GameRecord(
                    player_a=player_a,
                    player_b=player_b,
                    winner_of_player_a=None if won is None else bool(won),
                    time_step_source=time_step_source,
                    time_step=time_step,
                    stratum=stratum,
                    source_file=str(path),
                )
            )
        return records

    return records  # unrecognized Family B per-game shape


# ---------------------------------------------------------------------------
# Family B -- pentanomial-aggregate-only reports (fleet aggregators' --out)
# ---------------------------------------------------------------------------
def _pentanomial_counts_from(node: dict[str, Any]) -> tuple[int, int, int] | None:
    pentanomial = node.get("pentanomial_sprt")
    if not isinstance(pentanomial, dict):
        return None
    if not all(key in pentanomial for key in ("ll_pairs", "split_pairs", "ww_pairs")):
        return None
    return int(pentanomial["ll_pairs"]), int(pentanomial["split_pairs"]), int(pentanomial["ww_pairs"])


def _is_family_b_pentanomial_only(data: dict[str, Any]) -> bool:
    if "games" in data:
        return False  # real per-game data takes precedence -- see _is_family_b_per_game
    if _pentanomial_counts_from(data) is not None:
        return True
    arms = data.get("arms")
    if isinstance(arms, dict):
        return any(
            isinstance(arm, dict) and _pentanomial_counts_from(arm) is not None for arm in arms.values()
        )
    return False


def _family_b_pentanomial_arm_records(
    arm: dict[str, Any], *, arm_name: str, path: Path, champion_name: str, stats: IngestStats
) -> list[GameRecord]:
    counts = _pentanomial_counts_from(arm)
    if counts is None:
        return []
    n_ll, n_split, n_ww = counts
    config = arm.get("config") if isinstance(arm.get("config"), dict) else {}
    checkpoint = config.get("checkpoint")
    if checkpoint:
        candidate_raw = str(checkpoint)
        identity_gap = False
    else:
        # KNOWN GAP (module docstring): h2h_postrepair_aggregate.py's pooled
        # per-arm report does not carry the checkpoint identity at all
        # (its load_games only collects n_full/c_scale/c_visit/
        # max_root_candidates_wide/max_decisions into "config" -- no
        # "checkpoint" key). Fall back to the arm label itself as a visibly
        # low-confidence proxy identity.
        candidate_raw = f"arm:{arm_name}"
        identity_gap = True
    opponent_raw = candidate_raw + "::raw_policy"
    player_a, player_b = resolve_pair_identities(candidate_raw, opponent_raw, champion_name=champion_name)
    time_step, time_step_source = resolve_time_step(identity_hint=candidate_raw, file_path=path)
    stats.used_pentanomial_approximation += 1
    if identity_gap:
        stats.identity_from_arm_label_gap += 1

    synthetic = pentanomial_pairs_to_synthetic_games(n_ll, n_split, n_ww)
    return [
        GameRecord(
            player_a=player_a,
            player_b=player_b,
            winner_of_player_a=outcome,
            time_step_source=time_step_source,
            time_step=time_step,
            stratum="raw_policy",
            source_file=str(path),
        )
        for outcome in synthetic
    ]


def _family_b_pentanomial_records(
    data: dict[str, Any], *, path: Path, champion_name: str, stats: IngestStats
) -> list[GameRecord]:
    arms = data.get("arms")
    if isinstance(arms, dict):
        records: list[GameRecord] = []
        for arm_name, arm in arms.items():
            if isinstance(arm, dict):
                records.extend(
                    _family_b_pentanomial_arm_records(
                        arm, arm_name=str(arm_name), path=path, champion_name=champion_name, stats=stats
                    )
                )
        return records
    return _family_b_pentanomial_arm_records(
        data, arm_name=path.stem, path=path, champion_name=champion_name, stats=stats
    )


# ---------------------------------------------------------------------------
# top-level ingest dispatch
# ---------------------------------------------------------------------------
def _is_opening_panel_report(data: dict[str, Any]) -> bool:
    return "per_root" in data and "aggregate" in data and "panel" in data


def ingest_files(
    paths: Sequence[Path], *, repo_root: Path, champion_name: str, track: str | None = "2p_no_trade"
) -> tuple[list[GameRecord], IngestStats]:
    """Parse every path, dispatching Family A / Family B (per-game or
    pentanomial-only), skipping opening_panel.py reports and malformed/
    unrecognized files, and returning (records, ingest_stats)."""
    stats = IngestStats()
    parsed: dict[Path, Any] = {}
    for path in paths:
        stats.files_seen += 1
        try:
            parsed[path] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            stats.files_malformed += 1

    referenced_parts = _collect_referenced_parts(parsed, repo_root=repo_root)

    records: list[GameRecord] = []
    for path, data in parsed.items():
        if path.resolve() in referenced_parts:
            stats.files_skipped_part_file += 1
            continue
        if not isinstance(data, dict):
            stats.files_unrecognized += 1
            continue

        if _is_opening_panel_report(data):
            stats.files_opening_panel_skipped += 1
            continue

        if isinstance(data.get("results"), list) or ("wins" in data and "games" in data and "opponent" in data):
            stats.files_family_a += 1
            file_records = _family_a_records(
                data, path=path, repo_root=repo_root, champion_name=champion_name, track_filter=track, stats=stats
            )
        elif _is_family_b_per_game(data):
            stats.files_family_b_per_game += 1
            file_records = _family_b_per_game_records(data, path=path, champion_name=champion_name)
        elif _is_family_b_pentanomial_only(data):
            stats.files_family_b_pentanomial_only += 1
            file_records = _family_b_pentanomial_records(
                data, path=path, champion_name=champion_name, stats=stats
            )
        else:
            stats.files_unrecognized += 1
            continue

        for record in file_records:
            if record.player_a == record.player_b:
                continue  # degenerate self-game, never informative -- drop (mirrors elo_ladder's i==j guard)
            if record.winner_of_player_a is None:
                stats.games_discarded_no_winner += 1
                continue
            records.append(record)
            stats.games_emitted += 1
            stats.per_stratum_game_counts[record.stratum] = (
                stats.per_stratum_game_counts.get(record.stratum, 0) + 1
            )

    return records, stats


# ---------------------------------------------------------------------------
# WHR fit
# ---------------------------------------------------------------------------
def fit_whr_stratum(
    records: Sequence[GameRecord],
    *,
    w2: float = 300.0,
    virtual_games: int = 2,
    iterations: int | None = None,
) -> Any:
    """Build and converge one `whr.Base` for a single stratum's records.
    `player_a` is WHR's "black", `player_b` is "white" -- a pure naming
    convention with no first-move-advantage meaning here (paired-seed H2H
    protocols already cancel color bias upstream, before this tool sees the
    data). Lazily imports `whr` so `--help` and any caller that never fits
    a real stratum works without the dependency installed."""
    try:
        import whr
    except ImportError as exc:  # pragma: no cover - exercised only when whr is absent
        raise RuntimeError(
            "tools/whr_ladder.py requires the `whr` package to fit a WHR "
            "model: pip install whr (also available via `pip install "
            "-e .[whr]`)."
        ) from exc

    base = whr.Base(config={"w2": w2, "virtual_games": virtual_games})
    games = [
        [record.player_a, record.player_b, "B" if record.winner_of_player_a else "W", int(record.time_step)]
        for record in records
        if record.winner_of_player_a is not None
    ]
    if games:
        base.create_games(games)
    if iterations is not None:
        base.iterate(int(iterations))
    else:
        base.iterate_until_converge(verbose=False)
    return base


# ---------------------------------------------------------------------------
# report building
# ---------------------------------------------------------------------------
def _ci95(elo: float, uncertainty: float) -> tuple[float, float]:
    return elo - 1.96 * uncertainty, elo + 1.96 * uncertainty


def _cis_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def trajectory_report(base: Any, player_name: str) -> list[dict[str, Any]]:
    """Per-consecutive-time_step Elo delta + CI-overlap flag for one WHR
    player -- the exact "resolves whether the Elo-gain-per-generation
    sequence is real or CI-overlap noise" analysis the ticket asks for."""
    raw = base.ratings_for_player(player_name)
    rows: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for time_step, elo, uncertainty in raw:
        lower, upper = _ci95(float(elo), float(uncertainty))
        row: dict[str, Any] = {
            "time_step": int(time_step),
            "elo": float(elo),
            "uncertainty": float(uncertainty),
            "ci95_lower": lower,
            "ci95_upper": upper,
            "delta_from_previous": None,
            "cis_overlap_with_previous": None,
        }
        if previous is not None:
            row["delta_from_previous"] = row["elo"] - previous["elo"]
            row["cis_overlap_with_previous"] = _cis_overlap(
                (previous["ci95_lower"], previous["ci95_upper"]), (lower, upper)
            )
        rows.append(row)
        previous = row
    return rows


def build_report(
    records: Sequence[GameRecord],
    *,
    stats: IngestStats,
    champion_name: str,
    w2: float = 300.0,
    virtual_games: int = 2,
    iterations: int | None = None,
) -> dict[str, Any]:
    by_stratum: dict[str, list[GameRecord]] = defaultdict(list)
    for record in records:
        by_stratum[record.stratum].append(record)

    strata_report: dict[str, Any] = {}
    for stratum in STRATA:
        stratum_records = by_stratum.get(stratum, [])
        time_step_sources = {"parsed_ordinal": 0, "mtime_fallback": 0}
        for record in stratum_records:
            time_step_sources[record.time_step_source] += 1
        if not stratum_records:
            strata_report[stratum] = {
                "games": 0,
                "time_step_sources": time_step_sources,
                "trajectory": [],
            }
            continue
        base = fit_whr_stratum(
            stratum_records, w2=w2, virtual_games=virtual_games, iterations=iterations
        )
        trajectory = trajectory_report(base, champion_name)
        strata_report[stratum] = {
            "games": len(stratum_records),
            "log_likelihood": base.log_likelihood(),
            "time_step_sources": time_step_sources,
            "trajectory": trajectory,
        }

    return {
        "champion_name": champion_name,
        "w2": w2,
        "virtual_games": virtual_games,
        "ingest_stats": stats.to_dict(),
        "strata": strata_report,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# WHR ladder report (champion_name={report['champion_name']})", ""]
    stats = report["ingest_stats"]
    lines.append(
        f"Ingest: {stats['files_seen']} files seen, {stats['files_family_a']} family-A, "
        f"{stats['files_family_b_per_game']} family-B-per-game, "
        f"{stats['files_family_b_pentanomial_only']} family-B-pentanomial-only "
        f"({stats['used_pentanomial_approximation']} used the approximation), "
        f"{stats['files_unrecognized']} unrecognized, {stats['files_malformed']} malformed, "
        f"{stats['files_opening_panel_skipped']} opening-panel-skipped."
    )
    if stats["skipped_other_tracks"]:
        lines.append(f"Skipped entries from other tracks: {', '.join(stats['skipped_other_tracks'])}.")
    lines.append("")
    for stratum, entry in report["strata"].items():
        lines.append(f"## {stratum} ({entry['games']} games)")
        if not entry["trajectory"]:
            lines.append("_no data_")
            lines.append("")
            continue
        lines.append("| time_step | elo | +/-CI | delta_from_previous | cis_overlap_with_previous |")
        lines.append("|---|---|---|---|---|")
        for row in entry["trajectory"]:
            ci = f"[{row['ci95_lower']:.1f}, {row['ci95_upper']:.1f}]"
            delta = "n/a" if row["delta_from_previous"] is None else f"{row['delta_from_previous']:+.1f}"
            overlap = "n/a" if row["cis_overlap_with_previous"] is None else str(row["cis_overlap_with_previous"])
            lines.append(f"| {row['time_step']} | {row['elo']:.1f} | {ci} | {delta} | {overlap} |")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--runs-dir",
        action="append",
        required=True,
        help="Directory to recursively glob **/*.json under; may be passed multiple times "
        "(gate/panel/H2H/ablation outputs live under different subtrees). No default guess "
        "is hardcoded -- point this at the real runs/ style directories on the GPU hosts.",
    )
    parser.add_argument("--repo-root", default=".", help="Root used to resolve part_files references.")
    parser.add_argument("--champion-name", default="champion_lineage")
    parser.add_argument("--track", default="2p_no_trade", help="Family A track filter (None to disable).")
    parser.add_argument("--w2", type=float, default=300.0, help="WHR day-to-day rating-drift variance.")
    parser.add_argument("--virtual-games", type=int, default=2, help="WHR first-day prior pseudo-games.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Fixed WHR iteration count; default iterates to convergence.",
    )
    parser.add_argument("--out", help="JSON report output path.")
    parser.add_argument("--out-markdown", help="Markdown report output path.")
    args = parser.parse_args()

    all_paths: list[Path] = []
    for runs_dir in args.runs_dir:
        all_paths.extend(walk_result_files(Path(runs_dir)))
    all_paths = sorted(set(all_paths))

    records, stats = ingest_files(
        all_paths,
        repo_root=Path(args.repo_root),
        champion_name=args.champion_name,
        track=args.track,
    )
    report = build_report(
        records,
        stats=stats,
        champion_name=args.champion_name,
        w2=args.w2,
        virtual_games=args.virtual_games,
        iterations=args.iterations,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if args.out_markdown:
        md_output = Path(args.out_markdown)
        md_output.parent.mkdir(parents=True, exist_ok=True)
        md_output.write_text(render_markdown(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
