"""Opponent-pool sampling: play a fraction of self-play games against ARCHIVED champions instead
of latest-vs-latest.

Research basis (memory ``catan-discrete-vs-continuous-verdict``):
  - Pure latest-vs-latest self-play is provably non-convergent in some games (can cycle). Two
    independent results say mixing in older opponents fixes it: the Tablut reproduction (asymmetric
    2-player, arXiv:2604.05476) got clear Elo gains playing 25% of games vs a pool of past
    checkpoints; OpenAI Five sampled historical opponents ~20% of the time, weighted toward
    opponents it beat at a favorable-but-not-trivial rate.
  - This matters MORE for Catan than for Go: Catan is asymmetric (first-player advantage + hidden
    dev cards), the exact setting where the Tablut paper reported catastrophic role-forgetting.
  - Connects to task #64 (regret-restart / Go-Exploit archived states): same "don't only train
    against your newest self" family.

Design choices:
  - Selection is DETERMINISTIC given the global ``game_index`` (a splittable hash), NOT a global
    RNG. This makes the whole flywheel resume-safe and reproducible: replaying game N always picks
    the same opponent, so a crashed-and-resumed generation cannot silently change its opponent mix.
  - Only the CHAMPION's decisions are training targets. The archived opponent just diversifies the
    STATES/opponents the champion faces; we never distill a frozen old net's policy. The caller is
    responsible for recording only champion-seat decisions on pool games (see ``PairedGame.record_seats``).
  - Weighting: if per-version win-rates are supplied, weight peaks at a target win-rate (OpenAI
    Five's "beatable but not trivial"). Without win-rates, weight by recency (newer archived nets
    are stronger, closer to current-policy distribution) while keeping a floor so an old anchor is
    still occasionally sampled (prevents drift away from fundamentals).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

from .checkpoint_registry import ChampionRef


@dataclass(frozen=True)
class OpponentPolicy:
    pool_fraction: float = 0.20          # fraction of games vs an archived opponent (Tablut 0.25 / OF ~0.20)
    target_winrate: float = 0.60         # champion's desired win-rate vs sampled opponent (OpenAI Five band)
    winrate_width: float = 0.20          # Gaussian width around target when win-rates are known
    recency_bias: float = 2.0            # exponent on rank when NO win-rates: >1 favors recent, keeps a floor
    min_archive_version_gap: int = 1     # never sample the champion itself out of the archive

    def __post_init__(self) -> None:
        if not (0.0 <= self.pool_fraction <= 1.0):
            raise ValueError(f"pool_fraction must be in [0,1], got {self.pool_fraction}")
        if self.recency_bias < 0:
            raise ValueError("recency_bias must be >= 0")


@dataclass(frozen=True)
class OpponentChoice:
    """Who the champion plays this game. ``is_pool`` False => mirror self-play (both seats champion).
    ``path``/``version`` describe the opponent net when ``is_pool`` is True."""
    is_pool: bool
    path: str
    version: int

    @property
    def kind(self) -> str:
        return "pool" if self.is_pool else "mirror"


def _u01(game_index: int, salt: str) -> float:
    """Deterministic uniform(0,1) from (game_index, salt) via a stable hash. Not for crypto — just a
    reproducible, well-distributed per-game draw that survives resume (unlike a global RNG)."""
    h = hashlib.sha256(f"{salt}:{int(game_index)}".encode()).digest()
    # take 8 bytes -> [0, 2^64) -> [0,1)
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def _weights(archive: Sequence[ChampionRef], policy: OpponentPolicy,
             winrates: dict[int, float] | None) -> list[float]:
    """Per-archived-champion sampling weight (unnormalized, all >= 0)."""
    n = len(archive)
    if n == 0:
        return []
    w: list[float] = []
    for rank, ref in enumerate(archive):  # archive is oldest-first
        if winrates and ref.version in winrates:
            wr = winrates[ref.version]
            # Gaussian peak at target_winrate — "beatable but not trivial".
            z = (wr - policy.target_winrate) / max(1e-6, policy.winrate_width)
            weight = math.exp(-0.5 * z * z)
        else:
            # recency: newest (highest rank) gets most weight; +1 floor keeps old anchors alive.
            frac = (rank + 1) / n  # in (0,1], 1.0 for the newest
            weight = frac ** policy.recency_bias + (1.0 / n)
        w.append(max(0.0, float(weight)))
    total = sum(w)
    if total <= 0:  # degenerate -> uniform
        return [1.0 / n] * n
    return w


def choose_opponent(game_index: int, champion: ChampionRef,
                    archive: Sequence[ChampionRef], policy: OpponentPolicy,
                    *, winrates: dict[int, float] | None = None) -> OpponentChoice:
    """Deterministically pick this game's opponent.

    ``archive`` is oldest-first (as returned by ``checkpoint_registry.list_archive``). Archived
    entries at or newer than the champion (and the champion itself) are excluded so we only ever
    sample STRICTLY older nets. If the pool is effectively empty (early run, only the champion in
    the archive) we fall back to mirror self-play regardless of ``pool_fraction``."""
    eligible = [a for a in archive
                if a.version <= champion.version - policy.min_archive_version_gap]
    if not eligible or policy.pool_fraction <= 0.0:
        return OpponentChoice(is_pool=False, path=champion.path, version=champion.version)

    if _u01(game_index, "pool_gate") >= policy.pool_fraction:
        return OpponentChoice(is_pool=False, path=champion.path, version=champion.version)

    # sample an eligible archived champion by weight, deterministically
    w = _weights(eligible, policy, winrates)
    total = sum(w)
    r = _u01(game_index, "pool_pick") * total
    acc = 0.0
    chosen = eligible[-1]
    for ref, weight in zip(eligible, w):
        acc += weight
        if r <= acc:
            chosen = ref
            break
    return OpponentChoice(is_pool=True, path=chosen.path, version=chosen.version)


def realized_pool_fraction(n_games: int, champion: ChampionRef,
                           archive: Sequence[ChampionRef], policy: OpponentPolicy,
                           *, winrates: dict[int, float] | None = None) -> float:
    """Diagnostic: the actual pool fraction over ``n_games`` (deterministic draws don't hit the
    nominal fraction exactly at small N). Useful for a startup assertion/log."""
    if n_games <= 0:
        return 0.0
    pool = sum(1 for i in range(n_games)
               if choose_opponent(i, champion, archive, policy, winrates=winrates).is_pool)
    return pool / n_games


if __name__ == "__main__":  # self-test (pure stdlib)
    def champ(v: int) -> ChampionRef:
        return ChampionRef(version=v, path=f"/arch/champion_v{v}.pt", promoted_at=0.0)

    pol = OpponentPolicy(pool_fraction=0.25)
    current = champ(10)
    archive = [champ(v) for v in range(0, 11)]  # 0..10, oldest-first (10 == champion)

    # determinism: same index -> same choice
    for i in (0, 1, 7, 42, 1000):
        assert choose_opponent(i, current, archive, pol) == choose_opponent(i, current, archive, pol)

    # never samples the champion itself or a newer/equal version
    for i in range(500):
        c = choose_opponent(i, current, archive, pol)
        if c.is_pool:
            assert c.version <= current.version - pol.min_archive_version_gap, c

    # realized pool fraction is close to nominal over many games
    frac = realized_pool_fraction(4000, current, archive, pol)
    assert 0.22 < frac < 0.28, f"realized pool fraction {frac:.3f} far from 0.25"

    # empty/degenerate archive -> always mirror
    only_self = [champ(10)]
    assert not choose_opponent(0, current, only_self, pol).is_pool
    assert realized_pool_fraction(100, current, [], pol) == 0.0

    # pool_fraction 0 -> always mirror
    assert realized_pool_fraction(100, current, archive, OpponentPolicy(pool_fraction=0.0)) == 0.0

    # recency bias: newer archived nets sampled more than the oldest, over many draws
    counts: dict[int, int] = {}
    for i in range(20000):
        c = choose_opponent(i, current, archive, pol)
        if c.is_pool:
            counts[c.version] = counts.get(c.version, 0) + 1
    assert counts.get(9, 0) > counts.get(0, 0), f"expected recency bias; got {counts}"

    # win-rate weighting peaks near target: opponents at ~0.6 wr sampled more than a crushed 0.95
    wr = {v: (0.6 if v == 5 else 0.95) for v in range(11)}
    counts_wr: dict[int, int] = {}
    for i in range(20000):
        c = choose_opponent(i, current, archive, pol, winrates=wr)
        if c.is_pool:
            counts_wr[c.version] = counts_wr.get(c.version, 0) + 1
    assert counts_wr.get(5, 0) > counts_wr.get(9, 0), f"expected target-winrate peak; got {counts_wr}"

    print("opponent_pool self-test OK")
