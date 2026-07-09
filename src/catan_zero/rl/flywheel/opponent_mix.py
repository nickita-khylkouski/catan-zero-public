"""Generalized N-way opponent-mix sampling (CAT-54).

``opponent_pool.py`` (H2) is a binary mirror-self-play-vs-one-archived-opponent
mechanism (a single ``pool_fraction``). CAT-5's R9 ruling adopted an EXACT
five-way starting mix -- 75% latest producer self-play / 10% previous+public
champion / 5% older champions / 5% hard-experimental nets + exploiters / 5%
catanatron_value -- and explicitly asks the generator to support *arbitrary*
mix configs (weights and categories are data, not hardcoded constants), not
just this one five-way split. This module is that generalization: it stays
independent of ``opponent_pool.py`` (which keeps its own binary callers, e.g.
``tools/continuous_flywheel.py``, working unchanged) and adds a categorical
sampler over any number of named ``MixCategory`` entries.

Design choices (mirrors ``opponent_pool.py``'s, see its docstring for the
full research rationale):
  - Selection is DETERMINISTIC given the global ``game_index`` (a splittable
    hash), NOT a global RNG -- resume-safe: replaying game N always draws the
    same category and, within a checkpoint-backed category, the same
    checkpoint.
  - Only the PRODUCER's (our own side's) decisions are ever training targets.
    This module only decides WHO the opponent is; the own-side-row filter
    itself lives in ``gumbel_self_play.play_one_game`` (unchanged by this
    module -- it already guards on ``PoolGameAssignment.is_pool`` /
    ``opponent_color``, which this module's callers populate identically).
  - A category's ``source`` is one of:
      "self"            -- producer self-play (mirror), no checkpoint needed.
      "checkpoint_list" -- sample one of an explicit list of checkpoints.
      "external_engine" -- a non-neural named engine (e.g. catanatron_value).
        CAT-54 does NOT wire any such engine (that is CAT-56's "Exploiter
        lane" ticket, which this ticket explicitly BLOCKS/enables); a category
        with this source must be marked ``pending=True`` (documents the
        intended weight without ever being sampled) or generation fails fast
        at manifest-parse time in the main process -- never silently drops
        the category or silently reweights around it without saying so.
  - Registry-awareness (CAT-9's ``tools/champion_registry.py``) is
    deliberately NOT imported here: this module stays pure-stdlib and
    registry-free, exactly like ``opponent_pool.py``. Resolving
    "registry_role"/"registry_pool" manifest sources into concrete
    ``checkpoint_list`` categories is ``tools/opponent_mix_registry.py``'s job
    (that module may import ``tools.champion_registry``; this one must not,
    matching the existing "src/catan_zero/rl does not depend on tools/"
    boundary already documented in gumbel_self_play.py).
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .opponent_pool import _u01

VALID_SOURCES: tuple[str, ...] = ("self", "checkpoint_list", "external_engine")

# External (non-neural) engines CAT-56 ("Exploiter lane") actually wires an
# in-generator opponent for. CAT-54 wired none (every "external_engine" category
# had to be marked ``pending``); CAT-56 flips that for exactly these names, which
# resolve to real Catanatron bots in the cross-engine lockstep
# (``catan_zero.rl.exploiter_lockstep.make_external_bot``): ``catanatron_value``
# -> ``ValueFunctionPlayer``, ``catanatron_ab{3,4,5}`` -> ``AlphaBetaPlayer``
# depth 3/4/5. Kept here (not in the lockstep module) so this pure-stdlib module
# can reject an UNWIRED engine name at manifest-parse time in the main process --
# a typo or an unimplemented engine must fail fast, never silently get sampled
# and then blow up per-worker. This is just a string allowlist: it introduces no
# torch/catanatron import into this module.
WIRED_EXTERNAL_ENGINES: tuple[str, ...] = (
    "catanatron_value",
    "catanatron_ab3",
    "catanatron_ab4",
    "catanatron_ab5",
)

# R9 ceiling on the exploiter lane's share of generation: "start 2-3% until
# neutral-harness parity proven, then 5%". This is the hard cap the generator
# refuses to exceed (see tools/generate_gumbel_selfplay_data.py's
# --exploiter-fraction and ``validate_external_engine_fraction`` below); the
# adopted DEFAULT (opponent_mix_r9_exploiter.json) is 3%, well under it.
EXTERNAL_ENGINE_FRACTION_CAP: float = 0.05


@dataclass(frozen=True)
class MixCheckpointRef:
    """One concrete opponent checkpoint available within a category."""

    path: str
    version: int = -1
    md5: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "version": self.version, "md5": self.md5}


@dataclass(frozen=True)
class MixCategory:
    """One named slice of the opponent mix (e.g. "producer_self_play",
    "hard_experimental"). ``weight`` is a raw (not pre-normalized) share --
    normalization happens over the EFFECTIVE (non-pending, weight>0, resolved)
    categories at sampling time, so the exact ratios in a manifest (e.g.
    75/10/5/5/5) are preserved verbatim in the config and only collapse to a
    smaller effective set when a category is explicitly ``pending``.
    """

    name: str
    weight: float
    source: str
    checkpoints: tuple[MixCheckpointRef, ...] = ()
    engine: str | None = None
    # Documents a category that is configured but not yet sample-able (e.g.
    # catanatron_value before CAT-56 wires a real evaluator for it). Pending
    # categories are excluded from sampling and from weight normalization,
    # but stay in the manifest/summary so the intended target mix is legible.
    pending: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MixCategory.name must be non-empty")
        if self.weight < 0.0:
            raise ValueError(f"MixCategory {self.name!r} weight must be >= 0, got {self.weight}")
        if self.source not in VALID_SOURCES:
            raise ValueError(
                f"MixCategory {self.name!r} has unknown source {self.source!r}; "
                f"expected one of {VALID_SOURCES}"
            )
        if self.source == "checkpoint_list" and not self.checkpoints and not self.pending:
            raise ValueError(
                f"MixCategory {self.name!r} has source='checkpoint_list' but no checkpoints "
                "(and is not marked pending) -- there is nothing to sample"
            )
        if self.source == "external_engine" and not self.engine:
            raise ValueError(f"MixCategory {self.name!r} has source='external_engine' but no engine name")
        if (
            self.source == "external_engine"
            and not self.pending
            and self.engine not in WIRED_EXTERNAL_ENGINES
        ):
            # CAT-56 wires the engines in ``WIRED_EXTERNAL_ENGINES`` (catanatron_value,
            # catanatron_ab3/4/5); any OTHER external-engine name is still unimplemented and
            # must fail loudly at construction, exactly as every external_engine did under
            # CAT-54 -- never silently get sampled and then crash a worker mid-generation.
            raise NotImplementedError(
                f"MixCategory {self.name!r} uses source='external_engine' (engine={self.engine!r}) "
                f"with weight>0 and pending=False, but no in-generator opponent is wired for "
                f"engine {self.engine!r}. CAT-56 ('Exploiter lane') wires only "
                f"{WIRED_EXTERNAL_ENGINES}. Use one of those engine names, mark this category "
                '`"pending": true` until it is wired, or drop it.'
            )

    @property
    def is_effective(self) -> bool:
        return (not self.pending) and self.weight > 0.0

    @property
    def is_external_engine(self) -> bool:
        return self.source == "external_engine"


@dataclass(frozen=True)
class OpponentMixConfig:
    categories: tuple[MixCategory, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for category in self.categories:
            if category.name in seen:
                raise ValueError(f"duplicate MixCategory name {category.name!r}")
            seen.add(category.name)
        if not self.effective_categories:
            raise ValueError(
                "OpponentMixConfig has no effective categories (all are pending or weight=0) -- "
                "nothing would ever be sampled"
            )

    @property
    def effective_categories(self) -> tuple[MixCategory, ...]:
        return tuple(c for c in self.categories if c.is_effective)

    def effective_weights(self) -> dict[str, float]:
        """Normalized (sum-to-1) weight per EFFECTIVE category name -- the ratios
        actually realized once pending categories are excluded, for logging/
        provenance so a manifest with a pending catanatron_value slice
        transparently reports its 75/10/5/5 (not 75/10/5/5/5) effective split."""
        effective = self.effective_categories
        total = sum(c.weight for c in effective)
        return {c.name: (c.weight / total if total > 0 else 0.0) for c in effective}


@dataclass(frozen=True)
class MixChoice:
    """The resolved opponent for one game.

    ``engine`` (CAT-56) is the external-engine name (e.g. ``"catanatron_value"``)
    when this game's opponent is an external Catanatron bot rather than a neural
    checkpoint or mirror self-play; it is ``""`` for every "self"/"checkpoint_list"
    choice, so the CAT-54 tests' ``choice == choice`` equality and the neural-mix
    row schema are unaffected. ``is_external`` routes ``run_worker_games`` to the
    cross-engine lockstep (``play_one_exploiter_game``) instead of loading a
    checkpoint evaluator; in that case ``path``/``version``/``md5`` stay empty/-1
    (there is no checkpoint), and ``is_pool`` is True because the opponent still
    occupies the non-producer seat (it is a non-mirror game)."""

    tag: str
    is_pool: bool
    path: str
    version: int
    md5: str
    engine: str = ""

    @property
    def kind(self) -> str:
        if self.is_external:
            return "external"
        return "pool" if self.is_pool else "self"

    @property
    def is_external(self) -> bool:
        return bool(self.engine)


def choose_mix_category(game_index: int, categories: Sequence[MixCategory]) -> MixCategory:
    """Deterministically pick this game's category from the EFFECTIVE
    (non-pending, weight>0) subset of ``categories``, weighted by ``weight``."""
    effective = [c for c in categories if c.is_effective]
    if not effective:
        raise ValueError("choose_mix_category requires at least one effective category")
    total = sum(c.weight for c in effective)
    draw = _u01(game_index, "mix_category") * total
    cumulative = 0.0
    for category in effective:
        cumulative += category.weight
        if draw <= cumulative:
            return category
    return effective[-1]  # floating-point fallback: last category


def choose_checkpoint_in_category(game_index: int, category: MixCategory) -> MixCheckpointRef:
    """Deterministically pick one checkpoint from a "checkpoint_list" category.
    Uniform over the category's own checkpoints -- the ticket's exact-mix ask
    is about CATEGORY-level ratios (75/10/5/5/5); within a category (e.g.
    "older_champion" spanning several archived versions) no particular
    weighting scheme was specified, so uniform is the simplest correct choice
    and keeps this fully deterministic/resume-safe like the rest of the
    module."""
    if category.source != "checkpoint_list":
        raise ValueError(f"category {category.name!r} has source={category.source!r}, not checkpoint_list")
    if not category.checkpoints:
        raise ValueError(f"category {category.name!r} has no checkpoints to choose from")
    draw = _u01(game_index, f"mix_checkpoint:{category.name}")
    index = min(int(draw * len(category.checkpoints)), len(category.checkpoints) - 1)
    return category.checkpoints[index]


def choose_mix_opponent(game_index: int, categories: Sequence[MixCategory]) -> MixChoice:
    """Deterministically resolve this game's full opponent choice: which
    category, and (if it needs one) which concrete checkpoint."""
    category = choose_mix_category(game_index, categories)
    if category.source == "self":
        return MixChoice(tag=category.name, is_pool=False, path="", version=-1, md5="")
    if category.source == "external_engine":
        # CAT-56 exploiter lane: the opponent is an external Catanatron bot, not
        # a checkpoint. Only reachable for a category that passed
        # MixCategory.__post_init__'s WIRED_EXTERNAL_ENGINES / pending gate, so
        # ``category.engine`` is a known-wired engine name here. There is no
        # checkpoint to pick (path/version/md5 stay empty); ``run_worker_games``
        # sees ``is_external`` and routes to the cross-engine lockstep.
        return MixChoice(
            tag=category.name, is_pool=True, path="", version=-1, md5="", engine=str(category.engine)
        )
    ref = choose_checkpoint_in_category(game_index, category)
    return MixChoice(tag=category.name, is_pool=True, path=ref.path, version=ref.version, md5=ref.md5)


def realized_mix_fractions(n_games: int, categories: Sequence[MixCategory]) -> dict[str, float]:
    """Diagnostic: the actual per-category fraction over ``n_games`` (small-N
    deterministic draws don't hit the nominal fraction exactly)."""
    if n_games <= 0:
        return {}
    counts: dict[str, int] = {c.name: 0 for c in categories if c.is_effective}
    for i in range(n_games):
        choice = choose_mix_opponent(i, categories)
        counts[choice.tag] = counts.get(choice.tag, 0) + 1
    return {name: count / n_games for name, count in counts.items()}


# --------------------------------------------------------------------------- exploiter-fraction (CAT-56)
def external_engine_effective_fraction(config: OpponentMixConfig) -> float:
    """Combined normalized (sum-to-1 over effective categories) weight of the
    external-engine (exploiter-lane) categories -- i.e. the share of generation
    that will play against a Catanatron bot in cross-engine lockstep. 0.0 when
    no external engine is effective (a pure neural mix / self-play), which is
    what the ``--exploiter-fraction``-off default and the cap check below key
    off of."""
    weights = config.effective_weights()
    return sum(
        weights.get(category.name, 0.0)
        for category in config.effective_categories
        if category.is_external_engine
    )


def validate_external_engine_fraction(
    config: OpponentMixConfig, *, cap: float = EXTERNAL_ENGINE_FRACTION_CAP
) -> float:
    """Fail fast (in the main process, before any worker spawns) if the exploiter
    lane's effective share exceeds ``cap`` (R9's 5% ceiling). Returns the realized
    external fraction for provenance/logging. A pure neural mix returns 0.0 and
    always passes -- this is a no-op for every non-exploiter run."""
    fraction = external_engine_effective_fraction(config)
    if fraction > cap + 1e-9:
        raise ValueError(
            f"exploiter-lane (external_engine) categories take {fraction:.4f} of the effective "
            f"mix, over the {cap:.4f} cap (R9: start 2-3%, ramp to 5% only once neutral-harness "
            "parity is proven). Lower the external categories' weights, pass a smaller "
            "--exploiter-fraction, or raise the cap deliberately."
        )
    return fraction


def scale_external_engine_fraction(config: OpponentMixConfig, fraction: float) -> OpponentMixConfig:
    """Return a copy of ``config`` whose external-engine categories together take
    exactly ``fraction`` of the effective mix, preserving (a) the relative ratios
    AMONG the external categories and (b) the raw weights of every non-external
    category. This is what ``--exploiter-fraction`` uses to let an operator dial
    the exploiter share with one number (e.g. ramp 0.02 -> 0.03 -> 0.05) without
    hand-editing per-category weights in the manifest.

    ``fraction`` must be in [0, 1). ``fraction == 0`` marks every external
    category ``pending`` (documented in the manifest, never sampled). Requires at
    least one effective external category and at least one effective non-external
    category (there is nothing to trade off otherwise)."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"exploiter fraction must be in [0, 1), got {fraction}")
    external = [c for c in config.effective_categories if c.is_external_engine]
    internal = [c for c in config.effective_categories if not c.is_external_engine]
    if not external:
        raise ValueError("scale_external_engine_fraction: config has no effective external-engine category")
    if not internal:
        raise ValueError(
            "scale_external_engine_fraction: config has no effective non-external category to trade "
            "the exploiter fraction against"
        )

    external_names = {c.name for c in external}
    if fraction == 0.0:
        return OpponentMixConfig(
            categories=tuple(
                dataclasses.replace(c, pending=True) if c.name in external_names else c
                for c in config.categories
            )
        )

    internal_raw = sum(c.weight for c in internal)
    external_raw = sum(c.weight for c in external)
    # Want external_raw' / (internal_raw + external_raw') = fraction.
    target_external_raw = fraction * internal_raw / (1.0 - fraction)
    scale = target_external_raw / external_raw
    return OpponentMixConfig(
        categories=tuple(
            dataclasses.replace(c, weight=c.weight * scale) if c.name in external_names else c
            for c in config.categories
        )
    )


# --------------------------------------------------------------------------- manifest I/O
def _category_from_dict(entry: dict[str, Any]) -> MixCategory:
    checkpoints = tuple(
        MixCheckpointRef(
            path=str(ck["path"]),
            version=int(ck.get("version", -1)) if ck.get("version") is not None else -1,
            md5=str(ck.get("md5", "")),
        )
        for ck in entry.get("checkpoints", [])
    )
    return MixCategory(
        name=str(entry["name"]),
        weight=float(entry["weight"]),
        source=str(entry["source"]),
        checkpoints=checkpoints,
        engine=entry.get("engine"),
        pending=bool(entry.get("pending", False)),
    )


def read_opponent_mix_manifest(path: str | Path) -> OpponentMixConfig:
    """Parse a pure-JSON opponent-mix manifest (pure stdlib -- safe to call in
    the main process before workers spawn, same fail-fast-early pattern as
    ``gumbel_self_play.read_opponent_pool_manifest``):

        {"categories": [
            {"name": "producer_self_play", "weight": 75, "source": "self"},
            {"name": "previous_public_champion", "weight": 10,
             "source": "checkpoint_list",
             "checkpoints": [{"path": "...", "version": 3, "md5": "..."}]},
            {"name": "catanatron_value", "weight": 5, "source": "external_engine",
             "engine": "catanatron_value", "pending": true}
         ]}

    "registry_role"/"registry_pool" sources (referencing CAT-9's
    ``ChampionRegistry``) are NOT understood here -- resolve those first via
    ``tools/opponent_mix_registry.py``, which expands them into concrete
    "checkpoint_list" categories before this parser ever sees them (this
    module never imports ``tools.champion_registry``).
    """
    data = json.loads(Path(path).read_text())
    raw_categories = list(data.get("categories", []))
    if not raw_categories:
        raise ValueError(f"opponent-mix manifest {path} has no 'categories' entries")
    categories = tuple(_category_from_dict(entry) for entry in raw_categories)
    return OpponentMixConfig(categories=categories)


def config_to_dict(config: OpponentMixConfig) -> dict[str, Any]:
    """Full round-trippable provenance dump (for the generation manifest.json)."""
    return {
        "categories": [
            {
                "name": c.name,
                "weight": c.weight,
                "source": c.source,
                "engine": c.engine,
                "pending": c.pending,
                "checkpoints": [ck.to_dict() for ck in c.checkpoints],
            }
            for c in config.categories
        ],
        "effective_weights": config.effective_weights(),
    }


if __name__ == "__main__":  # self-test (pure stdlib)
    self_play = MixCategory(name="producer_self_play", weight=75.0, source="self")
    prev_public = MixCategory(
        name="previous_public_champion",
        weight=10.0,
        source="checkpoint_list",
        checkpoints=(MixCheckpointRef(path="/arch/champion_v3.pt", version=3, md5="aaa"),),
    )
    older = MixCategory(
        name="older_champion",
        weight=5.0,
        source="checkpoint_list",
        checkpoints=(
            MixCheckpointRef(path="/arch/champion_v0.pt", version=0, md5="bbb"),
            MixCheckpointRef(path="/arch/champion_v1.pt", version=1, md5="ccc"),
        ),
    )
    hard_negative = MixCategory(
        name="hard_experimental",
        weight=5.0,
        source="checkpoint_list",
        checkpoints=(MixCheckpointRef(path="/arch/exploiter_v0.pt", version=-1, md5="ddd"),),
    )
    catanatron_value = MixCategory(
        name="catanatron_value", weight=5.0, source="external_engine", engine="catanatron_value", pending=True
    )
    categories = (self_play, prev_public, older, hard_negative, catanatron_value)
    config = OpponentMixConfig(categories=categories)

    # determinism: same index -> same choice
    for i in (0, 1, 7, 42, 1000):
        assert choose_mix_opponent(i, categories) == choose_mix_opponent(i, categories)

    # pending category is never sampled
    for i in range(2000):
        assert choose_mix_opponent(i, categories).tag != "catanatron_value"

    # effective weights renormalize over the 4 non-pending categories (95 total)
    weights = config.effective_weights()
    assert abs(weights["producer_self_play"] - 75.0 / 95.0) < 1e-9, weights
    assert "catanatron_value" not in weights

    # realized fractions land close to the effective nominal ratios
    fractions = realized_mix_fractions(20000, categories)
    assert 0.74 < fractions["producer_self_play"] < 0.81, fractions
    assert 0.06 < fractions["previous_public_champion"] < 0.14, fractions

    # a non-pending external_engine category with an UNWIRED engine refuses to construct
    try:
        MixCategory(name="bad", weight=1.0, source="external_engine", engine="not_a_real_bot", pending=False)
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass

    # CAT-56: a WIRED engine (catanatron_value) now constructs and samples as external
    wired = OpponentMixConfig(
        categories=(
            MixCategory(name="producer_self_play", weight=97.0, source="self"),
            MixCategory(name="catanatron_value", weight=3.0, source="external_engine", engine="catanatron_value"),
        )
    )
    assert abs(external_engine_effective_fraction(wired) - 0.03) < 1e-9
    saw_external = any(choose_mix_opponent(i, wired.categories).is_external for i in range(2000))
    assert saw_external, "expected the wired catanatron_value category to be sampled"
    # scaling the exploiter fraction preserves internal weights and hits the target
    scaled = scale_external_engine_fraction(wired, 0.02)
    assert abs(external_engine_effective_fraction(scaled) - 0.02) < 1e-9
    # the cap check refuses an over-cap exploiter share
    over = OpponentMixConfig(
        categories=(
            MixCategory(name="producer_self_play", weight=80.0, source="self"),
            MixCategory(name="catanatron_value", weight=20.0, source="external_engine", engine="catanatron_value"),
        )
    )
    try:
        validate_external_engine_fraction(over)
        raise AssertionError("expected ValueError for over-cap exploiter fraction")
    except ValueError:
        pass

    # an all-pending/empty config refuses to construct
    try:
        OpponentMixConfig(categories=(catanatron_value,))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    print("opponent_mix self-test OK")
