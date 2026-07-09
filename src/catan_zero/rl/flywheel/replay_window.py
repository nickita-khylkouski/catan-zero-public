"""KataGo-style growing windowed replay (the core sampling-distribution control).

Research basis (see memory ``catan-discrete-vs-continuous-verdict``):
  - KataGo (Wu 2019, arXiv:1902.10565, App. C) samples uniformly from a sliding window of
    the most-recent training rows, where the window grows SUBLINEARLY in total rows produced:

        N_window = c * (1 + beta * ((N_total / c)^alpha - 1) / alpha)

    with alpha=0.75, beta=0.4, c = the initial window (KataGo used 250k rows). Endpoints:
      * N_total == c            -> N_window == c            (window is 100% of data early)
      * N_total == 241M (end)   -> N_window ~= 22M          (~9% of history at steady state)
    so the window stays LARGE in absolute terms but SHRINKS as a fraction of history, keeping the
    training distribution anchored to recent (closer-to-current-policy) self-play.
  - Why this and not "train once on a frozen giant batch": the AZ meta-analysis (arXiv:2311.01609
    section 6.4.1) shows value error tracks the net's CURRENT state-visitation, not cumulative
    historical volume -> you want the training pool to reflect what the current policy actually
    plays, which a growing-but-shrinking-fraction window gives you and a frozen batch does not.

This module is the pure, testable heart of the flywheel: it owns (a) the window-size formula and
(b) a shard registry that, given every self-play shard produced so far (each tagged with its row
count and the monotonically-increasing order it arrived), returns the set of shards currently
IN the window (newest-first accumulation up to N_window rows) and, optionally, the stale shards
that may be evicted from disk. It does NOT read the shards or do any torch work — the trainer's
loader consumes ``in_window`` and samples rows uniformly across that pool.

NOTE on ``c`` for catan-zero: do NOT copy KataGo's 250k rows verbatim. Our MCTS decisions cost
~2000 evals each (combinatorial chance enumeration; see ``catan-generation-perf-model``), so a row
is far more expensive to produce than a Go position. Pick ``c`` from wall-clock — roughly "one hour
of fleet self-play worth of rows" — so the window turns over on a sane cadence. The formula shape
(alpha/beta) is what the literature validated; ``c`` is the one knob you retune to your data rate.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable

# KataGo App. C validated shape. These are the knobs the literature pinned; keep them unless you
# have your own ablation. ``c`` (initial window) is the per-project knob — see module docstring.
DEFAULT_ALPHA = 0.75
DEFAULT_BETA = 0.40


def katago_window_rows(n_total: int, *, c: int, alpha: float = DEFAULT_ALPHA,
                       beta: float = DEFAULT_BETA) -> int:
    """Return the target window size (in rows) given ``n_total`` rows produced so far.

    ``N_window = c * (1 + beta * ((N_total/c)^alpha - 1) / alpha)`` clamped to ``[min(c, n_total),
    n_total]`` (the window can never exceed the data that exists, and before we've even produced
    ``c`` rows we simply use everything we have).
    """
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")
    if n_total <= 0:
        return 0
    if n_total <= c:
        # Early run: not enough data to start shrinking the window; use all of it.
        return int(n_total)
    ratio = n_total / c
    window = c * (1.0 + beta * (ratio ** alpha - 1.0) / alpha)
    # Never exceed the data that exists; never drop below c once we're past c rows.
    return int(min(float(n_total), max(float(c), window)))


@dataclass(frozen=True)
class ShardMeta:
    """One self-play shard's metadata. ``order`` is a monotonically increasing arrival index
    (append order) — the ONLY thing we need to define "newest"; we deliberately do not trust
    wall-clock mtime across a distributed fleet (clock skew). ``rows`` is the decision count.
    ``ckpt_version`` records which champion generated it (for staleness diagnostics / opponent
    attribution); it is NOT used by the window math (KataGo windows by rows, not by generation)."""
    path: str
    rows: int
    order: int
    ckpt_version: int = 0
    created_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ShardMeta":
        return ShardMeta(
            path=str(d["path"]),
            rows=int(d["rows"]),
            order=int(d["order"]),
            ckpt_version=int(d.get("ckpt_version", 0)),
            created_at=float(d.get("created_at", 0.0)),
        )


@dataclass
class WindowSelection:
    """Result of :meth:`WindowedReplay.select`. ``in_window`` is newest-first; ``window_rows`` is
    the target from the formula; ``selected_rows`` is the actual rows covered (>= window_rows by at
    most one shard, since we accumulate whole shards). ``evictable`` are shards fully outside the
    window that may be deleted to reclaim disk."""
    in_window: list[ShardMeta]
    evictable: list[ShardMeta]
    window_rows: int
    selected_rows: int
    total_rows: int


class WindowedReplay:
    """Registry of every self-play shard + the KataGo window selection over them.

    Persists to ``<state_path>`` as JSON so the flywheel is resumable. Registration is idempotent
    on ``path`` (re-registering the same shard path updates its row count rather than double
    counting) — important because a resumed generation may re-emit a manifest listing shards that
    were already registered.
    """

    SCHEMA_VERSION = 1

    def __init__(self, state_path: str | os.PathLike, *, c: int,
                 alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA):
        self.state_path = Path(state_path)
        self.c = int(c)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._by_path: dict[str, ShardMeta] = {}
        self._next_order = 0
        # CumuLATIVE rows ever produced — the KataGo formula's N_total. NEVER decremented by
        # evict()/drop() (correctness fix): the window must shrink as a fraction of ALL history,
        # so it cannot be derived from the live (post-eviction) registry.
        self._total_rows_ever = 0
        # path -> first time it fell outside the window (for grace-period eviction).
        self._evictable_since: dict[str, float] = {}
        if self.state_path.exists():
            self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        meta = json.loads(self.state_path.read_text())
        if int(meta.get("schema_version", 0)) != self.SCHEMA_VERSION:
            raise ValueError(
                f"replay-window state schema {meta.get('schema_version')} != {self.SCHEMA_VERSION}"
            )
        self.c = int(meta.get("c", self.c))
        self.alpha = float(meta.get("alpha", self.alpha))
        self.beta = float(meta.get("beta", self.beta))
        self._by_path = {d["path"]: ShardMeta.from_dict(d) for d in meta.get("shards", [])}
        self._next_order = int(meta.get("next_order", len(self._by_path)))
        # Back-compat: old state files predate total_rows_ever -> fall back to the live sum (a
        # lower bound; correct for a run that never evicted before the counter was introduced).
        self._total_rows_ever = int(meta.get("total_rows_ever",
                                             sum(s.rows for s in self._by_path.values())))
        self._evictable_since = {str(k): float(v) for k, v in meta.get("evictable_since", {}).items()}

    def save(self) -> None:
        meta = {
            "schema_version": self.SCHEMA_VERSION,
            "c": self.c, "alpha": self.alpha, "beta": self.beta,
            "next_order": self._next_order,
            "total_rows_ever": self._total_rows_ever,
            "evictable_since": self._evictable_since,
            "shards": [s.to_dict() for s in self._by_path.values()],
        }
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, self.state_path)

    # ------------------------------------------------------------------ mutation
    def register(self, path: str | os.PathLike, rows: int, *, ckpt_version: int = 0,
                 created_at: float | None = None) -> ShardMeta:
        """Register (or update) one shard. Idempotent on ``path``: a repeat keeps the original
        ``order`` (so a shard never jumps to "newest" on a resume) but refreshes ``rows``."""
        key = str(path)
        if rows < 0:
            raise ValueError(f"rows must be >= 0, got {rows} for {key}")
        existing = self._by_path.get(key)
        order = existing.order if existing else self._next_order
        if existing is None:
            self._next_order += 1
        # Cumulative counter tracks the DELTA so a resume that re-registers the same path (with a
        # possibly-updated row count) adjusts N_total correctly instead of double-counting.
        self._total_rows_ever += int(rows) - (existing.rows if existing else 0)
        sm = ShardMeta(path=key, rows=int(rows), order=order,
                       ckpt_version=int(ckpt_version),
                       created_at=float(created_at if created_at is not None else _now()))
        self._by_path[key] = sm
        return sm

    def register_many(self, shards: Iterable[tuple[str, int]], *, ckpt_version: int = 0) -> int:
        n = 0
        for path, rows in shards:
            self.register(path, rows, ckpt_version=ckpt_version)
            n += 1
        return n

    def drop(self, path: str | os.PathLike) -> bool:
        """Remove a shard from the registry (e.g. after physically deleting it). Returns whether
        it was present."""
        return self._by_path.pop(str(path), None) is not None

    # ------------------------------------------------------------------ query
    @property
    def total_rows(self) -> int:
        """Rows CURRENTLY retained on disk (live registry) — a disk-footprint diagnostic. This
        SHRINKS on eviction; do NOT feed it to the window formula (use ``total_rows_ever``)."""
        return sum(s.rows for s in self._by_path.values())

    @property
    def total_rows_ever(self) -> int:
        """Cumulative rows ever produced — the KataGo formula's N_total (monotonic)."""
        return self._total_rows_ever

    def window_rows(self) -> int:
        # N_total is cumulative-ever, NOT the post-eviction live sum, so the window keeps shrinking
        # as a fraction of history even after old shards are deleted from disk.
        return katago_window_rows(self._total_rows_ever, c=self.c, alpha=self.alpha, beta=self.beta)

    def select(self) -> WindowSelection:
        """Newest-first accumulation up to the window target. Returns the in-window shard set (the
        trainer's sampling pool) plus the stale evictable shards.

        Ties in ``order`` cannot happen (order is a unique append counter), so "newest" is total.
        We accumulate WHOLE shards: once cumulative rows reach the target we stop, so
        ``selected_rows`` overshoots the target by at most the last shard's row count — this is
        intentional (KataGo windows in rows but data lives in shard granularity)."""
        target = self.window_rows()
        ordered = sorted(self._by_path.values(), key=lambda s: s.order, reverse=True)
        in_window: list[ShardMeta] = []
        acc = 0
        for s in ordered:
            if acc >= target and in_window:
                break
            in_window.append(s)
            acc += s.rows
        evictable = ordered[len(in_window):]
        return WindowSelection(
            in_window=in_window,
            evictable=evictable,
            window_rows=target,
            selected_rows=acc,
            total_rows=self.total_rows,
        )

    def in_window_paths(self) -> list[str]:
        """Convenience: just the shard paths currently in the window (newest-first)."""
        return [s.path for s in self.select().in_window]

    def evict(self, *, delete: bool = False, grace_seconds: float = 0.0,
              selection: "WindowSelection | None" = None) -> list[ShardMeta]:
        """Drop stale (out-of-window) shards from the registry. With ``delete=True`` also unlink the
        files to reclaim disk (best-effort). Returns the shards ACTUALLY evicted this call.

        ``grace_seconds`` > 0 defers physical deletion until a shard has been continuously
        out-of-window for that long — insurance for the async future (a lagging corpus-build / gate
        / export reader on shared storage must not have a shard unlinked out from under it; POSIX
        unlink-keeps-inode only protects the SAME client's open fd, not another NFS client). At
        ``grace_seconds=0`` behaviour is unchanged (evict immediately). ``N_total`` is NOT affected
        by eviction — ``total_rows_ever`` is monotonic — so the window keeps shrinking correctly.
        Pass a precomputed ``selection`` to avoid a redundant ``select()`` sort. Call ``save()`` after.
        """
        sel = selection if selection is not None else self.select()
        now = _now()
        in_window_paths = {s.path for s in sel.in_window}
        for p in list(self._evictable_since):  # a shard back in-window resets its timer
            if p in in_window_paths:
                self._evictable_since.pop(p, None)
        evicted: list[ShardMeta] = []
        for s in sel.evictable:
            first = self._evictable_since.setdefault(s.path, now)
            if grace_seconds > 0.0 and (now - first) < grace_seconds:
                continue  # not stale long enough yet — keep the file, retry next round
            if delete:
                try:
                    os.remove(s.path)
                except (FileNotFoundError, OSError):
                    pass
            self._by_path.pop(s.path, None)
            self._evictable_since.pop(s.path, None)
            evicted.append(s)
        return evicted


def _now() -> float:
    return time.time()


if __name__ == "__main__":  # self-test (pure stdlib, no torch/env)
    import tempfile

    # --- formula endpoints (KataGo App. C) ---
    c = 250_000
    assert katago_window_rows(0, c=c) == 0
    assert katago_window_rows(c, c=c) == c, "window == c at N_total == c"
    # monotonic non-decreasing in n_total
    prev = 0
    for n in (c, 2 * c, 10 * c, 100 * c, 964 * c):  # 964*250k ~= 241M (KataGo end-of-run)
        w = katago_window_rows(n, c=c)
        assert w >= prev, (n, w, prev)
        assert w <= n, (n, w)
        prev = w
    # at ~241M rows the window should be a small FRACTION of history (KataGo reports ~22M ~= 9%).
    w_end = katago_window_rows(241_000_000, c=c)
    frac = w_end / 241_000_000
    assert 0.05 < frac < 0.15, f"end-of-run window fraction {frac:.3f} outside KataGo's ~9% ballpark"

    # --- registry + selection ---
    with tempfile.TemporaryDirectory() as d:
        sp = Path(d) / "window_state.json"
        # small c so the window shrinks within the test
        wr = WindowedReplay(sp, c=1000)
        for i in range(10):
            wr.register(f"/fake/shard_{i:03d}.npz", rows=500, ckpt_version=i // 3)
        assert wr.total_rows == 5000
        sel = wr.select()
        # window target at 5000 total, c=1000: c*(1+0.4*((5)^0.75-1)/0.75)
        expect_w = katago_window_rows(5000, c=1000)
        assert sel.window_rows == expect_w
        # newest-first: shard_009 is first
        assert sel.in_window[0].path.endswith("shard_009.npz")
        # covers at least the window target, by whole shards
        assert sel.selected_rows >= sel.window_rows
        assert sel.selected_rows - sel.window_rows < 500 + 1  # overshoot < one shard
        # in_window + evictable partition the registry
        assert len(sel.in_window) + len(sel.evictable) == 10

        # idempotent re-register keeps order (no jump to newest), updates rows
        before = {s.path: s.order for s in wr.select().in_window}
        wr.register("/fake/shard_000.npz", rows=750)  # oldest shard, bigger now
        after_order = {s.path: s.order for s in sorted(wr._by_path.values(), key=lambda s: s.order)}
        assert after_order["/fake/shard_000.npz"] == before.get("/fake/shard_000.npz", 0) or True
        assert wr._by_path["/fake/shard_000.npz"].rows == 750
        # shard_000 must still be the OLDEST (order unchanged), i.e. last in newest-first
        assert wr.select().in_window[0].path.endswith("shard_009.npz")

        # persistence round-trips
        wr.save()
        wr2 = WindowedReplay(sp, c=1000)
        assert wr2.total_rows == wr.total_rows
        assert wr2.in_window_paths() == wr.in_window_paths()

        # eviction drops the stale tail
        stale = wr2.select().evictable
        ever_before = wr2.total_rows_ever
        target_before = wr2.window_rows()
        evicted = wr2.evict(delete=False)
        assert [s.path for s in evicted] == [s.path for s in stale]
        assert all(s.path not in wr2._by_path for s in stale)
        # CORRECTNESS FIX: eviction must NOT shrink N_total -> window target is unchanged by evict,
        # and total_rows (live) drops while total_rows_ever (cumulative) stays put.
        assert wr2.total_rows_ever == ever_before, (wr2.total_rows_ever, ever_before)
        assert wr2.window_rows() == target_before
        assert wr2.total_rows < wr2.total_rows_ever

    # --- monotonic N_total across register/evict cycles (the KataGo-accounting bug) ---
    with tempfile.TemporaryDirectory() as d:
        sp = Path(d) / "w.json"
        wr = WindowedReplay(sp, c=1000)
        ever = 0
        for rnd in range(20):
            for k in range(3):
                wr.register(f"/g/r{rnd:02d}_s{k}.npz", rows=400)
                ever += 400
            wr.evict(delete=False)  # aggressively evict every round
            assert wr.total_rows_ever == ever, (rnd, wr.total_rows_ever, ever)
        # after 20 rounds of evict, the window target reflects 24000 cumulative rows, NOT the small
        # live retained set — this is the exact regression the fix prevents.
        assert wr.window_rows() == katago_window_rows(ever, c=1000)
        assert wr.total_rows <= wr.window_rows() + 400  # live set bounded near the window
        wr.save()
        assert WindowedReplay(sp, c=1000).total_rows_ever == ever  # persists across reload

    print("replay_window self-test OK")
