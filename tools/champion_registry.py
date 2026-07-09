from __future__ import annotations

"""Registry roles (CAT-9): ``generator_champion``, ``public_champion`` (pinned to gen-3
for now), ``tournament_bot`` and ``opponent_pool[]``, plus the promotion tripwires as
executable code instead of manual judgment.

This converts the manual "registry + runs/CURRENT_CHAMPION pointer" runbook (Roadmap
Step A1; Master Plan Sec 2.2/2.3) into disk-backed state. It sits alongside, not on top
of, ``src/catan_zero/rl/flywheel/checkpoint_registry.py`` (candidate/champion/archive
file management for the continuous trainer) -- this module is the higher-level "which
role points at which checkpoint, and why" ledger the ticket asks for; it tracks pointers
by path + md5 + provenance rather than owning the weight files themselves.

Three independent pieces, importable and unit-testable without a GPU:

  1. ``ChampionRegistry`` -- disk-backed JSON store for the four roles. Singleton role
     pointers (``generator_champion``/``public_champion``/``tournament_bot``) can be
     reassigned; ``opponent_pool`` is APPEND-ONLY (Master Plan Sec 2.2, R1: "kept, not
     deleted" -- regressed nets and prior champions stay in the pool as hard negatives).
     Every mutation is recorded in an append-only ``transitions`` journal.

  2. Auto-revert tripwire (Master Plan Sec 2.3 / Roadmap Sec 2): a pure function that
     turns "P(dElo_ext < -25) > 0.9 OR two consecutive external-panel declines" into
     code. Posteriors are a SIMPLE NORMAL APPROXIMATION on Elo from raw win/loss/draw
     counts -- see ``elo_posterior_normal`` for the documented assumptions.

  3. Every-3rd-promotion confirmation flag + bucket-veto hook (Master Plan Sec 4.1 /
     R8): a promotion counter and a pure ``bucket_veto`` function so any single
     phase/opening/blowout bucket can veto a promotion even if the pooled aggregate
     would pass.

Pure stdlib; reuses ``tools.sprt_gate``'s elo_to_score/score_to_elo so the Elo<->score
mapping is defined in exactly one place.
"""

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tools.sprt_gate import score_to_elo

ROLE_NAMES: tuple[str, ...] = ("generator_champion", "public_champion", "tournament_bot")


# --------------------------------------------------------------------------- helpers
def _md5_of_file(path: str | os.PathLike, *, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


# =============================================================================
# 1. Disk-backed registry: roles + append-only opponent pool + append-only journal
# =============================================================================
@dataclass(frozen=True)
class RolePointer:
    role: str
    checkpoint_path: str
    md5: str
    version: int | None
    updated_at: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PoolEntry:
    checkpoint_path: str
    md5: str
    version: int | None
    added_at: float
    status: str  # descriptive only ("active" / "regressed" / ...) -- never removed
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Transition:
    ts: float
    kind: str  # "set_role" | "auto_revert" | "pool_append" | "promotion_recorded"
    role: str | None
    reason: str
    from_pointer: dict[str, Any] | None
    to_pointer: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChampionRegistry:
    """Disk-backed registry of the four CAT-9 roles.

    ``generator_champion`` / ``public_champion`` / ``tournament_bot`` are singleton
    pointers that can be reassigned via ``set_role``. ``opponent_pool`` is append-only:
    there is deliberately no delete/remove API -- regressed nets and dethroned
    champions stay available as hard negatives (Master Plan Sec 2.2, R1). Every
    mutation is appended to ``transitions``; nothing already written there is ever
    edited or removed, so the journal is a durable audit trail of every role change
    and every promotion.
    """

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self._roles: dict[str, RolePointer] = {}
        self._pool: list[PoolEntry] = []
        self._transitions: list[Transition] = []
        self._promotion_counts: dict[str, int] = {}
        if self.path.exists():
            self._load()

    # ---------------------------------------------------------------- persistence
    def _load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._roles = {name: RolePointer(**entry) for name, entry in raw.get("roles", {}).items()}
        self._pool = [PoolEntry(**entry) for entry in raw.get("opponent_pool", [])]
        self._transitions = [Transition(**entry) for entry in raw.get("transitions", [])]
        self._promotion_counts = {str(k): int(v) for k, v in raw.get("promotion_counts", {}).items()}

    def save(self) -> None:
        state = {
            "roles": {name: ptr.to_dict() for name, ptr in self._roles.items()},
            "opponent_pool": [e.to_dict() for e in self._pool],
            "transitions": [t.to_dict() for t in self._transitions],
            "promotion_counts": dict(self._promotion_counts),
        }
        _atomic_write_json(self.path, state)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ChampionRegistry":
        return cls(path)

    # ---------------------------------------------------------------- role CRUD
    def get_role(self, role: str) -> RolePointer | None:
        return self._roles.get(role)

    def roles(self) -> dict[str, RolePointer]:
        return dict(self._roles)

    def set_role(
        self,
        role: str,
        checkpoint_path: str | os.PathLike,
        *,
        expected_md5: str | None = None,
        version: int | None = None,
        provenance: dict[str, Any] | None = None,
        reason: str = "",
        kind: str = "set_role",
    ) -> RolePointer:
        """Point ``role`` at ``checkpoint_path``. Rejects the write if the file's
        computed md5 doesn't match a caller-supplied ``expected_md5`` -- the same
        "version and bytes can never disagree" discipline as
        ``checkpoint_registry.py`` -- so a stale or corrupted provenance claim can
        never silently repoint a role."""
        if role not in ROLE_NAMES:
            raise ValueError(f"unknown role {role!r}; expected one of {ROLE_NAMES}")
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        actual_md5 = _md5_of_file(path)
        if expected_md5 is not None and expected_md5 != actual_md5:
            raise ValueError(
                f"md5 mismatch for {checkpoint_path}: expected {expected_md5}, "
                f"computed {actual_md5} -- refusing to point {role!r} at a "
                "checkpoint whose bytes don't match the claimed provenance"
            )
        previous = self._roles.get(role)
        pointer = RolePointer(
            role=role,
            checkpoint_path=str(path),
            md5=actual_md5,
            version=version,
            updated_at=time.time(),
            provenance=dict(provenance or {}),
        )
        self._roles[role] = pointer
        self._transitions.append(
            Transition(
                ts=pointer.updated_at,
                kind=kind,
                role=role,
                reason=reason,
                from_pointer=previous.to_dict() if previous else None,
                to_pointer=pointer.to_dict(),
            )
        )
        return pointer

    # ---------------------------------------------------------------- opponent pool
    def opponent_pool(self) -> tuple[PoolEntry, ...]:
        return tuple(self._pool)

    def append_pool(
        self,
        checkpoint_path: str | os.PathLike,
        *,
        expected_md5: str | None = None,
        version: int | None = None,
        provenance: dict[str, Any] | None = None,
        status: str = "active",
        reason: str = "",
    ) -> PoolEntry:
        """Append a checkpoint to the opponent pool. Idempotent on an identical
        (path, md5) pair already present (re-running a promotion step doesn't
        duplicate the entry) but there is no remove path: entries are kept forever
        (R1) as future hard negatives / diversity anchors."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        actual_md5 = _md5_of_file(path)
        if expected_md5 is not None and expected_md5 != actual_md5:
            raise ValueError(
                f"md5 mismatch for {checkpoint_path}: expected {expected_md5}, computed {actual_md5}"
            )
        for existing in self._pool:
            if existing.checkpoint_path == str(path) and existing.md5 == actual_md5:
                return existing
        entry = PoolEntry(
            checkpoint_path=str(path),
            md5=actual_md5,
            version=version,
            added_at=time.time(),
            status=status,
            provenance=dict(provenance or {}),
        )
        self._pool.append(entry)
        self._transitions.append(
            Transition(
                ts=entry.added_at,
                kind="pool_append",
                role="opponent_pool",
                reason=reason,
                from_pointer=None,
                to_pointer=entry.to_dict(),
            )
        )
        return entry

    # ---------------------------------------------------------------- promotion counter
    def record_promotion(self, role: str = "generator_champion") -> int:
        """Increment and return the promotion count for ``role``. Combine with
        ``requires_nth_confirmation`` to route every 3rd promotion through the
        heavier 200-game/n=64 bucketed confirmation (Master Plan Sec 4.1, R8)."""
        count = self._promotion_counts.get(role, 0) + 1
        self._promotion_counts[role] = count
        self._transitions.append(
            Transition(
                ts=time.time(),
                kind="promotion_recorded",
                role=role,
                reason=f"promotion #{count}",
                from_pointer=None,
                to_pointer=None,
            )
        )
        return count

    def promotion_count(self, role: str = "generator_champion") -> int:
        return self._promotion_counts.get(role, 0)

    def promotion_counts(self) -> dict[str, int]:
        return dict(self._promotion_counts)

    # ---------------------------------------------------------------- auto-revert
    def auto_revert(
        self,
        decision: "TripwireDecision",
        *,
        revert_to_checkpoint: str | os.PathLike,
        role: str = "generator_champion",
        expected_md5: str | None = None,
        version: int | None = None,
    ) -> RolePointer:
        """Write ``role`` back to ``revert_to_checkpoint`` (the last externally
        stable pointer) and log the revert with the tripwire's justification.
        Raises if ``decision.should_revert`` is False -- this is meant to be called
        only after ``auto_revert_tripwire`` has actually tripped, never
        speculatively."""
        if not decision.should_revert:
            raise ValueError("auto_revert called but the tripwire decision did not trip")
        return self.set_role(
            role,
            revert_to_checkpoint,
            expected_md5=expected_md5,
            version=version,
            provenance={"auto_revert": True, "trigger_reason": decision.reason},
            reason=f"AUTO-REVERT: {decision.reason}",
            kind="auto_revert",
        )

    # ---------------------------------------------------------------- read-only view
    def transitions(self) -> tuple[Transition, ...]:
        return tuple(self._transitions)


# =============================================================================
# 2. Auto-revert tripwire: normal-approximation Elo posteriors from panel counts
# =============================================================================
def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def elo_posterior_normal(
    wins: int, losses: int, draws: int = 0, *, prior_pseudocount: float = 0.5
) -> tuple[float, float]:
    """Return ``(mean_elo, se_elo)``: a normal approximation to the posterior over
    the true external Elo gap, from raw win/loss/draw panel counts.

    ASSUMPTIONS (documented, not hidden -- this is deliberately the "simple normal
    approximation on Elo" the ticket asks for, not a full Bayesian model):

    1. Each game is treated as an independent Bernoulli/trinomial trial with a
       fixed per-game win probability ``p`` (a draw counts as half a win, the same
       convention ``tools.sprt_gate``'s pentanomial GSPRT uses for color-swapped
       pairs). This ignores the negative within-pair correlation color-swapping
       introduces (fishtest #348 measures roughly -0.15), which makes the
       resulting standard error a slight OVER-estimate of the true uncertainty --
       i.e. this tripwire is conservative (less trigger-happy) relative to an
       exact pentanomial treatment, biasing toward NOT auto-reverting on
       borderline evidence.
    2. The sampling distribution of the observed score is approximated as Normal
       (central limit theorem) and then mapped to Elo via the logistic
       ``score_to_elo`` and its local derivative (delta method), rather than
       solving a fully Bayesian posterior. A ``prior_pseudocount`` of half a win
       plus half a loss is added before computing the observed rate, both to keep
       it away from the {0, 1} boundary (where the delta method's Jacobian
       diverges) and as a mild uninformative-prior regularizer at very small n.
    3. The prior on the true win probability is otherwise flat; "posterior" here
       means the frequentist sampling distribution of the point estimate,
       reinterpreted as an approximate posterior under that flat prior.

    Raises ``ValueError`` if given zero games.
    """
    n = wins + losses + draws
    if n <= 0:
        raise ValueError("elo_posterior_normal requires at least one game")
    score = wins + 0.5 * draws
    total = n + 2.0 * prior_pseudocount
    p_hat = (score + prior_pseudocount) / total
    p_hat = min(max(p_hat, 1e-6), 1.0 - 1e-6)
    mean_elo = score_to_elo(p_hat)
    se_p = math.sqrt(p_hat * (1.0 - p_hat) / total)
    # delta method: d(elo)/dp = 400 / (ln(10) * p * (1 - p)), since
    # elo(p) = 400 * log10(p / (1 - p)).
    d_elo_dp = 400.0 / (math.log(10.0) * p_hat * (1.0 - p_hat))
    se_elo = se_p * d_elo_dp
    return mean_elo, se_elo


def prob_elo_below(mean_elo: float, se_elo: float, threshold: float) -> float:
    """P(true Elo gap < ``threshold``) under the normal approximation."""
    if se_elo <= 0.0:
        return 1.0 if mean_elo < threshold else 0.0
    z = (threshold - mean_elo) / se_elo
    return normal_cdf(z)


@dataclass(frozen=True)
class PanelResult:
    """Raw win/loss/draw counts from one external panel run, for the CURRENT
    generator_champion lineage. ``label`` is free-form (panel id / date) purely
    for logging -- it doesn't affect the math."""

    wins: int
    losses: int
    draws: int = 0
    label: str = ""

    def posterior(self, *, prior_pseudocount: float = 0.5) -> "PanelPosterior":
        mean_elo, se_elo = elo_posterior_normal(
            self.wins, self.losses, self.draws, prior_pseudocount=prior_pseudocount
        )
        return PanelPosterior(
            mean_elo=mean_elo, se_elo=se_elo, n=self.wins + self.losses + self.draws, label=self.label
        )


@dataclass(frozen=True)
class PanelPosterior:
    mean_elo: float
    se_elo: float
    n: int
    label: str = ""

    @property
    def median_elo(self) -> float:
        # Normal approximation: median == mean.
        return self.mean_elo

    def prob_below(self, threshold: float) -> float:
        return prob_elo_below(self.mean_elo, self.se_elo, threshold)


def combine_panels(*panels: PanelResult) -> PanelResult:
    """Pool raw counts across panels (used for the "combined P(dElo_ext<0)>0.9"
    leg of the two-consecutive-decline condition)."""
    if not panels:
        raise ValueError("combine_panels requires at least one panel")
    return PanelResult(
        wins=sum(p.wins for p in panels),
        losses=sum(p.losses for p in panels),
        draws=sum(p.draws for p in panels),
        label="combined(" + ",".join(p.label for p in panels if p.label) + ")",
    )


@dataclass(frozen=True)
class TripwireDecision:
    should_revert: bool
    condition_a: bool
    condition_b: bool
    reason: str
    current_posterior: PanelPosterior
    previous_posterior: PanelPosterior | None
    combined_posterior: PanelPosterior | None


def auto_revert_tripwire(
    current_panel: PanelResult,
    previous_panel: PanelResult | None = None,
    *,
    catastrophic_threshold: float = -25.0,
    catastrophic_prob: float = 0.9,
    decline_median_threshold: float = -10.0,
    combined_decline_threshold: float = 0.0,
    combined_decline_prob: float = 0.9,
) -> TripwireDecision:
    """Auto-revert rule (Master Plan Sec 2.3 / Roadmap Sec 2), as pure code:

        P(dElo_ext < -25) > 0.9
        OR
        (two consecutive external panels each with posterior median < -10
         AND combined P(dElo_ext < 0) > 0.9)

    ``current_panel`` is the most recent external-panel result for the current
    generator_champion lineage; ``previous_panel`` (if given) is the one before
    it. Condition (a) looks only at ``current_panel``, so a single good panel
    after a bad one clears it immediately. Condition (b) requires BOTH panels to
    individually show a decline (median < ``decline_median_threshold``) as well
    as the pooled evidence -- a flapping bad-then-good (or good-then-bad)
    sequence trips neither condition, by construction.
    """
    current_posterior = current_panel.posterior()
    condition_a = current_posterior.prob_below(catastrophic_threshold) > catastrophic_prob

    condition_b = False
    previous_posterior: PanelPosterior | None = None
    combined_posterior: PanelPosterior | None = None
    if previous_panel is not None:
        previous_posterior = previous_panel.posterior()
        combined_posterior = combine_panels(previous_panel, current_panel).posterior()
        both_declining = (
            current_posterior.median_elo < decline_median_threshold
            and previous_posterior.median_elo < decline_median_threshold
        )
        combined_bad = combined_posterior.prob_below(combined_decline_threshold) > combined_decline_prob
        condition_b = both_declining and combined_bad

    should_revert = condition_a or condition_b
    if condition_a and condition_b:
        reason = (
            f"P(dElo_ext<{catastrophic_threshold})>{catastrophic_prob} on the latest panel AND "
            f"two consecutive declining panels with combined P(dElo_ext<{combined_decline_threshold})>{combined_decline_prob}"
        )
    elif condition_a:
        reason = (
            f"P(dElo_ext<{catastrophic_threshold})={current_posterior.prob_below(catastrophic_threshold):.3f} "
            f">{catastrophic_prob} on the latest panel ({current_posterior.label or 'unlabeled'})"
        )
    elif condition_b:
        reason = (
            "two consecutive external panels below the decline threshold "
            f"(medians {previous_posterior.median_elo:.1f}, {current_posterior.median_elo:.1f} < {decline_median_threshold}) "
            f"with combined P(dElo_ext<{combined_decline_threshold})="
            f"{combined_posterior.prob_below(combined_decline_threshold):.3f}>{combined_decline_prob}"
        )
    else:
        reason = "no tripwire condition met"

    return TripwireDecision(
        should_revert=should_revert,
        condition_a=condition_a,
        condition_b=condition_b,
        reason=reason,
        current_posterior=current_posterior,
        previous_posterior=previous_posterior,
        combined_posterior=combined_posterior,
    )


# =============================================================================
# 3. Every-3rd-promotion confirmation flag + bucket-veto hook
# =============================================================================
def requires_nth_confirmation(promotion_count: int, *, every: int = 3) -> bool:
    """True when ``promotion_count`` (1-indexed: the count AFTER recording this
    promotion) is a multiple of ``every`` -- i.e. every 3rd promotion routes
    through the heavier 200-game/n=64 non-regression confirmation instead of the
    cheap low-sim gate alone (Master Plan Sec 4.1, R8)."""
    if promotion_count <= 0:
        raise ValueError("promotion_count must be >= 1")
    if every <= 0:
        raise ValueError("every must be >= 1")
    return promotion_count % every == 0


@dataclass(frozen=True)
class BucketResult:
    """Raw win/loss/draw counts for one bucket (e.g. phase / opening / blowout)
    of the every-3rd-promotion confirmation battery."""

    bucket: str
    wins: int
    losses: int
    draws: int = 0

    @property
    def n(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def winrate(self) -> float | None:
        if self.n == 0:
            return None
        return (self.wins + 0.5 * self.draws) / self.n


@dataclass(frozen=True)
class BucketVetoResult:
    veto: bool
    veto_buckets: tuple[str, ...]
    per_bucket: dict[str, dict[str, Any]]


def bucket_veto(
    buckets: Sequence[BucketResult],
    *,
    min_winrate: float = 0.50,
    min_n: int = 8,
) -> BucketVetoResult:
    """Per-bucket pass/fail hook for the every-3rd-promotion confirmation check.

    Any single bucket at or above ``min_n`` games whose win-rate is below
    ``min_winrate`` VETOES the promotion, even if the pooled/aggregate win-rate
    across all buckets would pass (Master Plan Sec 4.1, R8: "an explicit
    per-bucket pass/fail so any single bucket can veto even if the aggregate
    passes"). Buckets below ``min_n`` are reported as ``insufficient_data`` and
    do not themselves veto -- there isn't enough signal to convict, but callers
    should treat that as "collect more games in this bucket", not an implicit
    pass.
    """
    per_bucket: dict[str, dict[str, Any]] = {}
    veto_buckets: list[str] = []
    for b in buckets:
        if b.n < min_n:
            per_bucket[b.bucket] = {"status": "insufficient_data", "n": b.n, "winrate": b.winrate}
            continue
        passed = b.winrate is not None and b.winrate >= min_winrate
        per_bucket[b.bucket] = {"status": "pass" if passed else "fail", "n": b.n, "winrate": b.winrate}
        if not passed:
            veto_buckets.append(b.bucket)
    return BucketVetoResult(veto=bool(veto_buckets), veto_buckets=tuple(veto_buckets), per_bucket=per_bucket)


# =============================================================================
# CLI
# =============================================================================
def _cmd_show(args: argparse.Namespace) -> None:
    reg = ChampionRegistry.load(args.registry)
    out = {
        "roles": {name: (ptr.to_dict() if ptr else None) for name, ptr in reg.roles().items()},
        "opponent_pool": [e.to_dict() for e in reg.opponent_pool()],
        "promotion_counts": reg.promotion_counts(),
        "n_transitions": len(reg.transitions()),
    }
    print(json.dumps(out, indent=2, sort_keys=True))


def _cmd_set_role(args: argparse.Namespace) -> None:
    reg = ChampionRegistry.load(args.registry)
    provenance = json.loads(args.provenance) if args.provenance else {}
    pointer = reg.set_role(
        args.role,
        args.checkpoint,
        expected_md5=args.expected_md5,
        version=args.version,
        provenance=provenance,
        reason=args.reason or "",
    )
    reg.save()
    print(json.dumps(pointer.to_dict(), indent=2, sort_keys=True))


def _cmd_append_pool(args: argparse.Namespace) -> None:
    reg = ChampionRegistry.load(args.registry)
    provenance = json.loads(args.provenance) if args.provenance else {}
    entry = reg.append_pool(
        args.checkpoint,
        expected_md5=args.expected_md5,
        version=args.version,
        provenance=provenance,
        status=args.status,
        reason=args.reason or "",
    )
    reg.save()
    print(json.dumps(entry.to_dict(), indent=2, sort_keys=True))


def _cmd_record_promotion(args: argparse.Namespace) -> None:
    reg = ChampionRegistry.load(args.registry)
    count = reg.record_promotion(args.role)
    reg.save()
    print(
        json.dumps(
            {
                "role": args.role,
                "promotion_count": count,
                "requires_nth_confirmation": requires_nth_confirmation(count, every=args.every),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _cmd_tripwire_check(args: argparse.Namespace) -> None:
    current = PanelResult(**json.loads(Path(args.current_panel).read_text(encoding="utf-8")))
    previous = None
    if args.previous_panel:
        previous = PanelResult(**json.loads(Path(args.previous_panel).read_text(encoding="utf-8")))
    decision = auto_revert_tripwire(current, previous)
    print(
        json.dumps(
            {
                "should_revert": decision.should_revert,
                "condition_a": decision.condition_a,
                "condition_b": decision.condition_b,
                "reason": decision.reason,
                "current_mean_elo": decision.current_posterior.mean_elo,
                "current_se_elo": decision.current_posterior.se_elo,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CAT-9 champion registry: roles, opponent pool, tripwires.")
    parser.add_argument("--registry", required=True, help="Path to the registry JSON file.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_show = sub.add_parser("show", help="Print the current registry state.")
    p_show.set_defaults(func=_cmd_show)

    p_set = sub.add_parser("set-role", help="Point a role at a checkpoint.")
    p_set.add_argument("--role", required=True, choices=ROLE_NAMES)
    p_set.add_argument("--checkpoint", required=True)
    p_set.add_argument("--expected-md5", dest="expected_md5")
    p_set.add_argument("--version", type=int)
    p_set.add_argument("--provenance", help="JSON object string.")
    p_set.add_argument("--reason", default="")
    p_set.set_defaults(func=_cmd_set_role)

    p_pool = sub.add_parser("append-pool", help="Append a checkpoint to the opponent pool.")
    p_pool.add_argument("--checkpoint", required=True)
    p_pool.add_argument("--expected-md5", dest="expected_md5")
    p_pool.add_argument("--version", type=int)
    p_pool.add_argument("--provenance", help="JSON object string.")
    p_pool.add_argument("--status", default="active")
    p_pool.add_argument("--reason", default="")
    p_pool.set_defaults(func=_cmd_append_pool)

    p_promo = sub.add_parser("record-promotion", help="Increment the promotion counter and check the every-Nth flag.")
    p_promo.add_argument("--role", default="generator_champion")
    p_promo.add_argument("--every", type=int, default=3)
    p_promo.set_defaults(func=_cmd_record_promotion)

    p_trip = sub.add_parser("tripwire-check", help="Evaluate the auto-revert tripwire against panel JSON files.")
    p_trip.add_argument("--current-panel", required=True, help='JSON file with {"wins":.., "losses":.., "draws":..}')
    p_trip.add_argument("--previous-panel", help="Optional JSON file for the prior panel.")
    p_trip.set_defaults(func=_cmd_tripwire_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
