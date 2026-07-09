from __future__ import annotations

"""Population arena orchestrator + Nash-averaging solver (Linear CAT-58).

WHY THIS EXISTS. Every promotion gate in this codebase is latest-vs-latest
(candidate vs one baseline). That protocol is blind to non-transitivity: if
net C beats B, B beats A, but A beats C (a rock-paper-scissors cycle), a
chain of latest-vs-latest gates can promote a lineage that is not actually
strictly improving in any population sense -- this is documented as the
"Leela Zero failure mode" in the master plan (R6). This module builds the
missing all-pairs cross-play harness: schedule every pair in a population
(last 8-12 champion nets + named catanatron bots, per the roadmap), collect
results, assemble a full payoff matrix, and solve for the population's
Nash equilibrium (Balduzzi et al., "Re-evaluating Evaluation", 1806.02643)
so cycling is made visible instead of silently averaged away by a linear
Elo/win-rate aggregate.

===========================================================================
DESIGN DECISION 1 -- reuse, don't reimplement, every existing H2H tool.
===========================================================================
This module is a SCHEDULER + AGGREGATOR + SOLVER. It does not itself play a
single game. Three existing tools already play exactly the three matchup
shapes this population needs, and are invoked here as subprocesses:

  net vs net   -> tools/gumbel_search_cross_net_h2h.py  (--candidate/--baseline)
  net vs bot   -> tools/gumbel_search_vs_bot_h2h.py      (--candidate/--baseline-bot)
  net vs raw   -> tools/gumbel_search_vs_raw_h2h.py      (--checkpoint; SAME net vs
                  its own no-search policy -- this is inherently a self-pair, not
                  an all-pairs matchup; see Decision 3)

Every one of these tools already writes a per-game "games": [...] list with a
"search_won"/"candidate_won" alias, which is exactly the shape
`tools/h2h_postrepair_aggregate.py` (commit 276f33b) already dedupes and pairs
by game_seed. This module calls `h2h_postrepair_aggregate.aggregate_arm`
directly for every pair's result reduction -- per the ticket's explicit
instruction, pairing/dedup logic is reused verbatim, not rewritten.

===========================================================================
DESIGN DECISION 2 -- seeds are hash-derived, never drawn from a shared range.
===========================================================================
This repo has an authoritative seed ledger (claimed base-seed ranges) because
a previous fleet run collided two independently-launched jobs on the same
50M-seed block and silently double-counted games into a false-significant
verdict (see tools/h2h_postrepair_aggregate.py's own `_dedupe_games` docstring
for the incident this hardened against). Rather than claim yet another slice
of that shared integer range, every pair's --base-seed here is derived via
`tools.promotion_gate_runner.derive_seed` (sha256 of the two player names +
a run label), the SAME derandomization fix (F9) already used for promotion
gates -- collisions across two different pairs would require a sha256
collision, not an accounting error.

===========================================================================
DESIGN DECISION 3 -- raw-policy population members are self-paired only.
===========================================================================
The roadmap asks for "raw policies" as population members. The only existing
tool that plays a raw (no-search) policy is `gumbel_search_vs_raw_h2h.py`,
which is hardcoded to compare ONE checkpoint's searched play against THAT
SAME checkpoint's raw policy -- there is no existing tool that plays net A's
raw policy against net B, or against a bot, with real search-based games.
Rather than invent a new match-runner (explicitly out of scope: "reuse...
do not reimplement matches"), each net's raw-policy variant is scheduled
as exactly one self-pair (`{net}::raw_policy` vs `{net}`), mirroring the
identical scoping decision `tools/whr_ladder.py` already made for the same
data (see its "raw_policy" stratum docstring). All-pairs cross-play is run
over {nets} union {bots} only.

===========================================================================
DESIGN DECISION 4 -- the Nash solver is a maximin LP, refined for max-entropy.
===========================================================================
For a SYMMETRIC ZERO-SUM matrix game (our payoff matrix is antisymmetric by
construction: catan-zero games always have exactly one winner, so
win_rate(B beats A) == 1 - win_rate(A beats B), see tools/sprt_gate.py's own
docstring), von Neumann's minimax theorem says the maximin mixed strategy of
either player IS a Nash equilibrium of the game, and it can be found with one
linear program (`scipy.optimize.linprog`) -- this is the "small scipy job"
the roadmap explicitly calls for, not a from-scratch game-theory solver.
Stage 1 solves that LP for the guaranteed game value v* and *a* maximin
strategy. Stage 2 (max-entropy refinement, `scipy.optimize.minimize`,
SLSQP) picks the most-spread-out strategy among the (possibly many) optima
achieving v* -- this is what makes the result a canonical "maximum entropy
Nash equilibrium" per Balduzzi et al., not an arbitrary vertex of the LP's
optimal face. Verified against two known-closed-form cases (see
tests/test_population_arena.py): rock-paper-scissors resolves to uniform
1/3 each (the antisymmetric RPS matrix forces this even from stage 1 alone,
see the module's derivation in the test file), and a strictly dominant pure
strategy resolves to a one-hot distribution on that strategy.
"""

import argparse
import glob
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

# Reused verbatim -- see Decision 1 / Decision 2 above. NOT copied.
from tools.champion_registry import ChampionRegistry, PanelResult, PoolEntry  # noqa: E402
from tools.elo_ladder import _canonicalize_node_id  # noqa: E402
from tools.gumbel_search_vs_bot_h2h import BOT_KINDS  # noqa: E402
from tools.h2h_postrepair_aggregate import aggregate_arm  # noqa: E402
from tools.promotion_gate_runner import derive_seed  # noqa: E402
from tools.sprt_gate import score_to_elo  # noqa: E402

# "value function, AB3/AB4" per the ticket -- AB5 is a valid BOT_KINDS member too
# (available via --bot-kinds) but not defaulted on, matching the ticket's wording.
DEFAULT_BOT_KINDS: tuple[str, ...] = ("catanatron_value", "catanatron_ab3", "catanatron_ab4")


# =============================================================================
# 1. Population roster
# =============================================================================
@dataclass(frozen=True)
class ArenaPlayer:
    """One population member. `kind == "net"` implies `checkpoint_path` is set;
    `kind == "bot"` is one of `tools.gumbel_search_vs_bot_h2h.BOT_KINDS`."""

    name: str
    kind: str  # "net" | "bot"
    checkpoint_path: str | None = None
    source: str = "explicit"  # "registry_pool" | "explicit_checkpoint" | "bot_roster"

    def __post_init__(self) -> None:
        if self.kind not in ("net", "bot"):
            raise ValueError(f"unknown ArenaPlayer.kind {self.kind!r}; expected 'net' or 'bot'")
        if self.kind == "net" and not self.checkpoint_path:
            raise ValueError(f"net player {self.name!r} requires a checkpoint_path")


def rank_recent_pool_entries(entries: Sequence[PoolEntry], *, max_nets: int) -> list[PoolEntry]:
    """Most-recent-first `max_nets` entries from an append-only opponent pool.
    Sorted by `added_at` (always present, monotonic with append order) rather
    than `version` (frequently None for older entries) -- since the pool is
    append-only in insertion order, `added_at` recency already recovers
    "the last N champion nets by lineage" without needing a parseable
    generation ordinal in every checkpoint path."""
    return sorted(entries, key=lambda e: e.added_at, reverse=True)[: max(0, max_nets)]


def build_roster(
    *,
    registry: ChampionRegistry | None = None,
    explicit_checkpoints: Sequence[str] = (),
    bot_kinds: Sequence[str] = DEFAULT_BOT_KINDS,
    max_pool_nets: int = 12,
) -> list[ArenaPlayer]:
    """Population set (roadmap Step B7 / ticket Step 1): the last
    `max_pool_nets` registry opponent-pool checkpoints, plus any
    explicitly-named checkpoints (e.g. v3a, a frozen hard-target anchor),
    plus the named catanatron bot roster. Deduplicated by canonical
    checkpoint identity (`tools.elo_ladder._canonicalize_node_id`, reused
    so a player's name here agrees with its WHR-ladder identity for the
    combined report in Step 5)."""
    for bot in bot_kinds:
        if bot not in BOT_KINDS:
            raise ValueError(f"unknown bot kind {bot!r}; expected one of {BOT_KINDS}")

    players: dict[str, ArenaPlayer] = {}
    if registry is not None:
        for entry in rank_recent_pool_entries(registry.opponent_pool(), max_nets=max_pool_nets):
            name = _canonicalize_node_id(entry.checkpoint_path)
            players[name] = ArenaPlayer(name=name, kind="net", checkpoint_path=entry.checkpoint_path, source="registry_pool")
    for checkpoint in explicit_checkpoints:
        name = _canonicalize_node_id(checkpoint)
        players[name] = ArenaPlayer(name=name, kind="net", checkpoint_path=checkpoint, source="explicit_checkpoint")
    for bot in bot_kinds:
        players[bot] = ArenaPlayer(name=bot, kind="bot", checkpoint_path=None, source="bot_roster")

    return sorted(players.values(), key=lambda p: (p.kind, p.name))


# =============================================================================
# 2. All-pairs schedule generator
# =============================================================================
def _slugify(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


@dataclass(frozen=True)
class MatchJob:
    """One scheduled H2H invocation. `out_glob` is what a caller aggregates
    over (supports the fleet case: many shard files from many hosts, one
    pattern per pair); `out_path` is the single-shard path a LOCAL smoke-test
    run should write its one file to."""

    pair_id: str
    player_a: str
    player_b: str
    match_kind: str  # "net_vs_net" | "net_vs_bot" | "net_vs_raw_policy"
    command: tuple[str, ...]
    out_glob: str
    out_path: str


def _base_command(
    tool: str,
    *,
    n_full: int,
    pairs: int,
    base_seed: int,
    workers: int,
    elo0: float,
    elo1: float,
    out_path: str,
    devices: str | None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(_TOOLS_DIR / tool),
        "--pairs", str(pairs),
        "--n-full", str(n_full),
        "--workers", str(workers),
        "--base-seed", str(base_seed),
        "--elo0", str(elo0),
        "--elo1", str(elo1),
        "--out", out_path,
    ]
    if devices:
        cmd += ["--devices", devices]
    return cmd


def generate_all_pairs_schedule(
    roster: Sequence[ArenaPlayer],
    *,
    n_full: int = 8,
    games_per_pair: int = 200,
    workers: int = 8,
    elo0: float = 0.0,
    elo1: float = 30.0,
    out_dir: str = "runs/population_arena",
    run_label: str = "arena",
    devices: str | None = None,
    include_raw_policy_self_pairs: bool = True,
) -> list[MatchJob]:
    """All-pairs schedule (ticket Step 2): every {net, net} and {net, bot}
    unordered pair exactly once, plus (if enabled) one net-vs-its-own-raw-
    policy self-pair per net (Decision 3). bot-vs-bot pairs are never
    scheduled -- no existing tool plays two hardcoded Catanatron bots
    against each other, and a fixed-strategy-vs-fixed-strategy matchup has
    no search/network component for this arena to measure anyway.

    `games_per_pair` is "few hundred games/pair" per the roadmap; each
    scheduled --pairs count is halved (`games_per_pair // 2`) because every
    one of the three reused H2H tools plays each seed TWICE, once per color
    (its own paired-seed color-swap protocol) -- i.e. `--pairs P` already
    produces `2P` games.
    """
    nets = [p for p in roster if p.kind == "net"]
    bots = [p for p in roster if p.kind == "bot"]
    pairs_arg = max(1, games_per_pair // 2)
    jobs: list[MatchJob] = []

    def _out(pair_id: str) -> tuple[str, str]:
        out_glob = f"{out_dir}/{pair_id}_*.json"
        out_path = f"{out_dir}/{pair_id}_local.json"
        return out_glob, out_path

    for i, a in enumerate(nets):
        for b in nets[i + 1 :]:
            pair_id = f"{_slugify(a.name)}__vs__{_slugify(b.name)}"
            out_glob, out_path = _out(pair_id)
            seed = derive_seed(a.name, b.name, run_label)
            cmd = _base_command(
                "gumbel_search_cross_net_h2h.py", n_full=n_full, pairs=pairs_arg, base_seed=seed,
                workers=workers, elo0=elo0, elo1=elo1, out_path=out_path, devices=devices,
            )
            cmd += ["--candidate", a.checkpoint_path, "--baseline", b.checkpoint_path]
            jobs.append(MatchJob(pair_id, a.name, b.name, "net_vs_net", tuple(cmd), out_glob, out_path))

    for a in nets:
        for bot in bots:
            pair_id = f"{_slugify(a.name)}__vs__{_slugify(bot.name)}"
            out_glob, out_path = _out(pair_id)
            seed = derive_seed(a.name, bot.name, run_label)
            cmd = _base_command(
                "gumbel_search_vs_bot_h2h.py", n_full=n_full, pairs=pairs_arg, base_seed=seed,
                workers=workers, elo0=elo0, elo1=elo1, out_path=out_path, devices=devices,
            )
            cmd += ["--candidate", a.checkpoint_path, "--baseline-bot", bot.name]
            jobs.append(MatchJob(pair_id, a.name, bot.name, "net_vs_bot", tuple(cmd), out_glob, out_path))

    if include_raw_policy_self_pairs:
        for a in nets:
            raw_name = f"{a.name}::raw_policy"
            pair_id = f"{_slugify(a.name)}__vs__raw_policy"
            out_glob, out_path = _out(pair_id)
            seed = derive_seed(a.name, raw_name, run_label)
            # No --devices on gumbel_search_vs_raw_h2h.py (single --device only).
            cmd = [
                sys.executable, str(_TOOLS_DIR / "gumbel_search_vs_raw_h2h.py"),
                "--checkpoint", a.checkpoint_path,
                "--pairs", str(pairs_arg), "--n-full", str(n_full), "--workers", str(workers),
                "--base-seed", str(seed), "--elo0", str(elo0), "--elo1", str(elo1),
                "--out", out_path,
            ]
            jobs.append(MatchJob(pair_id, a.name, raw_name, "net_vs_raw_policy", tuple(cmd), out_glob, out_path))

    return jobs


# =============================================================================
# 3. Payoff-matrix assembly (reuses h2h_postrepair_aggregate.aggregate_arm)
# =============================================================================
def collect_pair_result(job: MatchJob, *, elo0: float = 0.0, elo1: float = 30.0) -> dict[str, Any] | None:
    """Aggregate every shard file matching `job.out_glob` via
    `tools.h2h_postrepair_aggregate.aggregate_arm` (dedup + game_seed
    pairing, commit 276f33b -- reused, not reimplemented). Returns None if no
    shard files exist yet (pair not run / still running)."""
    paths = sorted(Path(p) for p in glob.glob(job.out_glob))
    if not paths:
        return None
    return aggregate_arm(paths, elo0=elo0, elo1=elo1)


@dataclass
class PayoffMatrix:
    players: list[str]
    payoff: list[list[float]]  # antisymmetric, in [-1, 1]; 0.0 for unplayed/no-decisive-pairs
    games_played: list[list[int]]  # decisive-pair count backing each entry (0 for unplayed)
    missing_pairs: list[tuple[str, str]] = field(default_factory=list)

    def index(self, name: str) -> int:
        return self.players.index(name)


def build_payoff_matrix(
    roster: Sequence[ArenaPlayer],
    jobs: Sequence[MatchJob],
    *,
    elo0: float = 0.0,
    elo1: float = 30.0,
    include_raw_policy: bool = True,
) -> PayoffMatrix:
    """Assemble the full payoff matrix (ticket Step 3) from every job's
    aggregated result. Antisymmetric by construction: catan-zero games always
    resolve to exactly one winner (`tools/sprt_gate.py`'s own documented
    no-draw assumption), so `win_rate(B > A) == 1 - win_rate(A > B)` exactly
    -- payoff[i][j] = 2*win_rate(i beats j) - 1, payoff[j][i] = -payoff[i][j].
    Unplayed pairs (no shard files yet, or zero decisive concordant pairs)
    are recorded as 0.0 (neutral) AND listed in `missing_pairs` so a caller
    can tell "measured as a coin flip" apart from "not measured yet" --
    Nash-averaging assumes a fully-played population, so missing coverage
    should be treated as a loud caveat on the resulting rating, not silently
    absorbed."""
    names = [p.name for p in roster if p.kind in ("net", "bot")]
    if include_raw_policy:
        for p in roster:
            if p.kind == "net":
                raw_name = f"{p.name}::raw_policy"
                if any(j.player_b == raw_name for j in jobs):
                    names.append(raw_name)
    n = len(names)
    index = {name: i for i, name in enumerate(names)}
    payoff = [[0.0] * n for _ in range(n)]
    games_played = [[0] * n for _ in range(n)]
    missing_pairs: list[tuple[str, str]] = []

    for job in jobs:
        if job.player_a not in index or job.player_b not in index:
            continue
        i, j = index[job.player_a], index[job.player_b]
        result = collect_pair_result(job, elo0=elo0, elo1=elo1)
        win_rate = result.get("pair_win_rate") if result else None
        decisive = result.get("pairs_decisive", 0) if result else 0
        if win_rate is None:
            missing_pairs.append((job.player_a, job.player_b))
            continue
        value = 2.0 * float(win_rate) - 1.0
        payoff[i][j] = value
        payoff[j][i] = -value
        games_played[i][j] = int(decisive)
        games_played[j][i] = int(decisive)

    return PayoffMatrix(players=names, payoff=payoff, games_played=games_played, missing_pairs=missing_pairs)


# =============================================================================
# 4. Nash-averaging solver (Balduzzi et al. 1806.02643) -- see Decision 4.
# =============================================================================
@dataclass
class NashResult:
    players: list[str]
    strategy: list[float]  # equilibrium mixture weight per player, sums to 1
    value: float  # guaranteed floor payoff at equilibrium (== 0.0 for an exact antisymmetric matrix)
    payoff_vs_equilibrium: list[float]  # each pure strategy's expected payoff facing the mixture
    maxent_refined: bool


def solve_nash_equilibrium(matrix: Sequence[Sequence[float]], *, maxent: bool = True) -> NashResult:
    """Maximum-entropy Nash equilibrium of a symmetric zero-sum matrix game
    via `scipy.optimize.linprog` + `scipy.optimize.minimize` (Decision 4).
    `matrix` must be square and antisymmetric (payoff[i][j] == -payoff[j][i],
    diagonal 0) -- exactly the shape `build_payoff_matrix` produces.
    """
    import numpy as np
    try:
        from scipy.optimize import linprog, minimize
    except ImportError as exc:  # pragma: no cover - exercised only when scipy is absent
        raise RuntimeError(
            "tools/population_arena.py requires the `scipy` package to solve the "
            "Nash equilibrium LP: pip install scipy (also available via `pip install "
            "-e .[dev]`)."
        ) from exc

    a = np.asarray(matrix, dtype=float)
    n = a.shape[0]
    if a.shape != (n, n):
        raise ValueError(f"payoff matrix must be square, got shape {a.shape}")
    if n == 0:
        raise ValueError("cannot solve a Nash equilibrium of an empty population")
    if not np.allclose(a, -a.T, atol=1e-6):
        raise ValueError("payoff matrix must be antisymmetric (payoff[i][j] == -payoff[j][i])")

    # Stage 1: maximin LP. Variables x = [p_0..p_{n-1}, v]; maximize v (minimize -v)
    # s.t. for every column j: sum_i p_i * A[i,j] >= v  <=>  -A[:,j]^T p + v <= 0.
    c = np.zeros(n + 1)
    c[-1] = -1.0
    a_ub = np.zeros((n, n + 1))
    a_ub[:, :n] = -a.T
    a_ub[:, n] = 1.0
    b_ub = np.zeros(n)
    a_eq = np.zeros((1, n + 1))
    a_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * n + [(None, None)]

    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Nash LP failed to converge: {res.message}")
    p_stage1 = np.clip(res.x[:n], 0.0, None)
    p_stage1 = p_stage1 / p_stage1.sum()
    v_star = float(res.x[n])

    maxent_refined = False
    p_final = p_stage1
    if maxent and n > 1:
        eps = 1e-9

        def neg_entropy(p: np.ndarray) -> float:
            q = np.clip(p, eps, None)
            return float(np.sum(q * np.log(q)))

        constraints = [
            {"type": "ineq", "fun": lambda p: a.T @ p - v_star + 1e-7},
            {"type": "eq", "fun": lambda p: np.sum(p) - 1.0},
        ]
        x0 = np.full(n, 1.0 / n)
        refined = minimize(
            neg_entropy, x0, method="SLSQP", bounds=[(eps, 1.0)] * n, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if refined.success:
            candidate = np.clip(refined.x, 0.0, None)
            candidate = candidate / candidate.sum()
            # Only accept the refinement if it still clears the stage-1 floor value
            # (SLSQP can land just outside tolerance on badly-conditioned matrices).
            if np.min(a.T @ candidate) >= v_star - 1e-4:
                p_final = candidate
                maxent_refined = True

    payoff_vs_equilibrium = (a @ p_final).tolist()
    return NashResult(
        players=list(range(n)),  # placeholder; caller fills in names via with_names below
        strategy=p_final.tolist(),
        value=v_star,
        payoff_vs_equilibrium=payoff_vs_equilibrium,
        maxent_refined=maxent_refined,
    )


def solve_population_nash(matrix: PayoffMatrix, *, maxent: bool = True) -> NashResult:
    result = solve_nash_equilibrium(matrix.payoff, maxent=maxent)
    result.players = list(matrix.players)
    return result


# =============================================================================
# 5. Combined report: Nash rating + WHR-stratified table + tripwire hook
# =============================================================================
def nash_elo(payoff_vs_equilibrium: float) -> float:
    """Elo-scale rendering of a player's expected payoff against the Nash
    mixture, reusing `tools.sprt_gate.score_to_elo` so the Elo<->score
    mapping stays defined in exactly one place across the codebase.
    `payoff_vs_equilibrium` is in [-1, 1]; rescaled to a [0, 1] score first."""
    score = (payoff_vs_equilibrium + 1.0) / 2.0
    score = min(max(score, 1e-6), 1.0 - 1e-6)
    return score_to_elo(score)


def build_arena_report(
    roster: Sequence[ArenaPlayer],
    matrix: PayoffMatrix,
    nash: NashResult,
    *,
    whr_report: dict[str, Any] | None = None,
    champion_name: str = "champion_lineage",
) -> dict[str, Any]:
    """Per-player: Nash equilibrium weight + payoff-vs-equilibrium (Elo-scale)
    -- the "world ranking" -- alongside, if supplied, the WHR ladder's own
    trajectory tail for that identity -- the "self-play ladder" -- kept as
    two clearly-labeled scores side by side per the master plan's explicit
    requirement (Sec 2.6/Sec 3 queue #3), never collapsed into one number."""
    kind_by_name = {p.name: p.kind for p in roster}
    whr_strata = (whr_report or {}).get("strata", {})

    def _whr_tail(name: str) -> dict[str, Any] | None:
        # tools/whr_ladder.py's build_report only extracts a trajectory for the
        # ONE named lineage player it was fit with (`--champion-name`), never
        # per-opponent trajectories -- so a WHR tail is only ever attachable to
        # that same identity here. Every other arena player (a specific pool
        # checkpoint, a bot, a raw-policy variant) has no WHR trajectory to
        # attach; this is a real coverage gap, not a lookup bug -- the caller
        # would need to re-run whr_ladder once per non-lineage identity of
        # interest to get a comparable trajectory for it.
        if name != champion_name:
            return None
        best: dict[str, Any] | None = None
        for stratum_name, stratum in whr_strata.items():
            trajectory = stratum.get("trajectory") or []
            if trajectory:
                best = {"stratum": stratum_name, **trajectory[-1]}
        return best

    rows = []
    for i, name in enumerate(nash.players):
        rows.append(
            {
                "player": name,
                "kind": kind_by_name.get(name, "raw_policy" if name.endswith("::raw_policy") else "unknown"),
                "nash_weight": nash.strategy[i],
                "nash_payoff_vs_equilibrium": nash.payoff_vs_equilibrium[i],
                "nash_elo": nash_elo(nash.payoff_vs_equilibrium[i]),
                "whr_latest": _whr_tail(name),
            }
        )
    rows.sort(key=lambda r: r["nash_elo"], reverse=True)

    n = len(matrix.players)
    pairs_with_results = sum(
        1 for i in range(n) for j in range(i + 1, n) if matrix.games_played[i][j] > 0
    )
    pairs_scheduled = pairs_with_results + len(matrix.missing_pairs)
    return {
        "players": nash.players,
        "nash_value": nash.value,
        "nash_maxent_refined": nash.maxent_refined,
        "coverage": {
            # Pairs this run actually attempted (has a result, or a shard glob that
            # produced no decisive result yet) -- NOT every n*(n-1)/2 combination
            # among `players`, since bot-vs-bot / raw-vs-raw / raw-vs-bot pairs are
            # never scheduled by design (Decision 3) and would otherwise masquerade
            # as "missing" coverage.
            "pairs_scheduled": pairs_scheduled,
            "pairs_with_results": pairs_with_results,
            "pairs_missing": len(matrix.missing_pairs),
            "missing_pairs": matrix.missing_pairs,
            "population_pairs_possible_if_fully_meshed": n * (n - 1) // 2,
        },
        "ratings": rows,
    }


def arena_panel_result_for(matrix: PayoffMatrix, *, subject_name: str, label: str = "population_arena") -> PanelResult:
    """Pool this arena run's decisive games for `subject_name` into a
    `tools.champion_registry.PanelResult` -- the "trend-vs-level tripwire
    hook (feed champion_registry.tripwire)" the ticket asks for. This reuses
    `champion_registry.auto_revert_tripwire` AS-IS (the caller passes this
    PanelResult, optionally alongside a previous run's PanelResult, straight
    into that function) rather than inventing a second tripwire -- the
    population arena's own head-to-head results become one more valid
    PanelResult source, on equal footing with an external panel run."""
    if subject_name not in matrix.players:
        raise ValueError(f"{subject_name!r} not in this arena's player list: {matrix.players}")
    i = matrix.index(subject_name)
    wins = 0
    losses = 0
    for j, opponent in enumerate(matrix.players):
        if j == i:
            continue
        n_games = matrix.games_played[i][j]
        if n_games <= 0:
            continue
        # payoff[i][j] in [-1, 1] == 2*win_rate - 1 over n_games decisive pairs.
        win_rate = (matrix.payoff[i][j] + 1.0) / 2.0
        wins += round(win_rate * n_games)
        losses += n_games - round(win_rate * n_games)
    return PanelResult(wins=wins, losses=losses, draws=0, label=label)


# =============================================================================
# CLI
# =============================================================================
def _cmd_schedule(args: argparse.Namespace) -> None:
    registry = ChampionRegistry.load(args.registry) if args.registry else None
    roster = build_roster(
        registry=registry,
        explicit_checkpoints=args.checkpoint or (),
        bot_kinds=tuple(args.bot_kinds.split(",")) if args.bot_kinds else DEFAULT_BOT_KINDS,
        max_pool_nets=args.max_pool_nets,
    )
    jobs = generate_all_pairs_schedule(
        roster,
        n_full=args.n_full,
        games_per_pair=args.games_per_pair,
        workers=args.workers,
        elo0=args.elo0,
        elo1=args.elo1,
        out_dir=args.out_dir,
        run_label=args.run_label,
        devices=args.devices,
        include_raw_policy_self_pairs=not args.no_raw_policy,
    )
    out = {
        "roster": [
            {"name": p.name, "kind": p.kind, "checkpoint_path": p.checkpoint_path, "source": p.source}
            for p in roster
        ],
        "jobs": [
            {
                "pair_id": j.pair_id, "player_a": j.player_a, "player_b": j.player_b,
                "match_kind": j.match_kind, "command": list(j.command), "out_glob": j.out_glob,
            }
            for j in jobs
        ],
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")


def _cmd_report(args: argparse.Namespace) -> None:
    schedule = json.loads(Path(args.schedule).read_text(encoding="utf-8"))
    roster = [
        ArenaPlayer(name=r["name"], kind=r["kind"], checkpoint_path=r.get("checkpoint_path"), source=r.get("source", "explicit"))
        for r in schedule["roster"]
    ]
    jobs = [
        MatchJob(
            pair_id=j["pair_id"], player_a=j["player_a"], player_b=j["player_b"],
            match_kind=j["match_kind"], command=tuple(j["command"]), out_glob=j["out_glob"],
            out_path=j["out_glob"].replace("_*.json", "_local.json"),
        )
        for j in schedule["jobs"]
    ]
    matrix = build_payoff_matrix(roster, jobs, elo0=args.elo0, elo1=args.elo1)
    nash = solve_population_nash(matrix, maxent=not args.no_maxent)
    whr_report = json.loads(Path(args.whr_report).read_text(encoding="utf-8")) if args.whr_report else None
    report = build_arena_report(roster, matrix, nash, whr_report=whr_report, champion_name=args.champion_name)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sched = sub.add_parser("schedule", help="Build the roster + all-pairs match schedule.")
    p_sched.add_argument("--registry", help="Path to a CAT-9 ChampionRegistry JSON file.")
    p_sched.add_argument("--checkpoint", action="append", help="Explicit checkpoint to include (repeatable).")
    p_sched.add_argument("--bot-kinds", help=f"Comma-separated bot roster (default {','.join(DEFAULT_BOT_KINDS)}).")
    p_sched.add_argument("--max-pool-nets", type=int, default=12)
    p_sched.add_argument("--n-full", type=int, default=8, help="Search sims/side (roadmap default n=8).")
    p_sched.add_argument("--games-per-pair", type=int, default=200)
    p_sched.add_argument("--workers", type=int, default=8)
    p_sched.add_argument("--elo0", type=float, default=0.0)
    p_sched.add_argument("--elo1", type=float, default=30.0)
    p_sched.add_argument("--out-dir", default="runs/population_arena")
    p_sched.add_argument("--run-label", default="arena")
    p_sched.add_argument("--devices", default=None)
    p_sched.add_argument("--no-raw-policy", action="store_true")
    p_sched.add_argument("--out", help="Write the schedule JSON here (also printed to stdout).")
    p_sched.set_defaults(func=_cmd_schedule)

    p_report = sub.add_parser("report", help="Aggregate results for a schedule and solve the Nash rating.")
    p_report.add_argument("--schedule", required=True, help="Schedule JSON produced by the 'schedule' subcommand.")
    p_report.add_argument("--elo0", type=float, default=0.0)
    p_report.add_argument("--elo1", type=float, default=30.0)
    p_report.add_argument("--no-maxent", action="store_true")
    p_report.add_argument("--whr-report", help="Optional tools/whr_ladder.py --out JSON to merge in.")
    p_report.add_argument("--champion-name", default="champion_lineage")
    p_report.add_argument("--out", help="Write the report JSON here (also printed to stdout).")
    p_report.set_defaults(func=_cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
