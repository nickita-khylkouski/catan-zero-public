"""Typed pipeline configs + a stable config-hash (task CAT-66).

WHY THIS EXISTS: the four science-critical entry points (train / generate /
gate / eval) each declare their reproducibility-relevant knobs as argparse
flags. Every flag carries its own default, and those defaults have silently
diverged from the dataclass/config defaults they feed at least 7 documented
times -- the "CLI-default-override trap" (e.g. the c_scale 1.0-vs-0.1 near-miss
that produced near-one-hot training targets, and the coupled
max_decisions/temperature_move_fraction pair). Grepping a run's CLI string to
prove two runs used the same regime is error-prone and, for masking, outright
impossible (train_bc records --mask-hidden-info only obliquely).

THE FIX (built on task #74's name-keyed serialization in
``config_serialization``): capture each pipeline's science-critical flags in a
frozen, schema-versioned dataclass; serialize it canonically (sorted-key JSON
over the fully-resolved field values); and hash that to a short, stable
``config_hash``. Every run threads its hash into its output artifact, so any
two runs can be compared for exact config equality by string comparison, and
the registry (see ``config_registry``) maps a hash back to the full config.

DESIGN NOTES
    * These configs are ADDITIVE. They are built from the already-parsed
      argparse namespace via ``from_namespace`` and never change how the CLIs
      parse or behave when the new flags are unused (see ``config_cli``).
    * A field's default here mirrors the corresponding CLI/dataclass default so
      that a config built from an argv-only invocation equals one built from a
      ``--config`` file carrying the same values. The pre-existing
      ``tests/test_cli_config_drift.py`` guard keeps the CLI defaults equal to
      the underlying search/self-play dataclass defaults; this module's tests
      keep the typed-config defaults equal to the CLI defaults, closing the
      loop.
    * The hash covers exactly the fields listed on each dataclass. Adding a
      field is a schema change (bump ``SCHEMA_VERSION``) and changes every
      hash; that is intended -- a new science-relevant knob is a new regime.

CAT-66 AUDIT FINDINGS (step 1: explicit divergence list, 2026-07-08)
    The four entry points were audited flag-by-flag against the dataclasses
    they feed (train_bc.py, generate_gumbel_selfplay_data.py, sprt_gate.py, and
    the three gumbel_search_*_h2h.py tools). Results:

    * NO currently-live CLI-vs-dataclass default divergence exists. Every
      generate/eval CLI default equals its underlying GumbelSelfPlayConfig /
      GumbelChanceMCTSConfig / EntityGraphRustEvaluatorConfig default
      (c_visit=50, c_scale=0.1, n_full=64, n_fast=16, p_full=0.25,
      max_decisions=600, temperature_high/low, prior_temperature=1.0,
      value_scale=1.0, correct_rust_chance_spectra=True, lazy_interior_chance,
      belief_chance_spectra, public_observation). The historical incidents --
      the c_scale 1.0-vs-0.1 near-miss and the coupled
      max_decisions/temperature_move_fraction pair -- are already structurally
      guarded at test time by ``tests/test_cli_config_drift.py``, which reads
      each CLI's argparse defaults via AST and fails on any drift.
    * This ticket ADDS the complementary guard: ``tests/test_pipeline_configs``
      asserts the typed-config defaults here equal those same CLI defaults, so
      a drift is caught from both directions, and threads a positive record
      (``config_hash``) into every output artifact so a run's regime is
      recorded, not merely prevented from drifting.
    * Latent (not-yet-fired) risks documented rather than a live bug:
      (a) the three h2h tools rebuild search config via
      ``worker_args.get(key, hardcoded_default)``, a SECOND copy of each
      argparse default that must be hand-synced;
      (b) several dataclass fields have NO CLI flag at all and silently take
      their dataclass default (GumbelChanceMCTSConfig.max_root_candidates /
      temperature / play_sh_winner; EntityGraphRustEvaluatorConfig.value_squash
      in generate, context_fill, cache_size);
      (c) gumbel_search_vs_bot_h2h.py accepts --n-full-wide /
      --raw-policy-above-width / --symmetry-averaged-eval but omits them from
      its output summary (files 1 & 3 record them).
      The config_hash closes (c) regardless of the summary gap: EvalConfig
      captures all three, so two vs-bot runs differing only in those flags now
      hash differently.
    * sprt_gate.py had no way to record the masking regime of the games it
      judged; GateConfig.generation_public_observation now echoes it from the
      consumed h2h summary. ``check_masking_consistency`` is provided for a
      caller holding two or more of the generate/eval/gate configs (e.g. an
      offline audit script loading their ``--dump-config`` artifacts) to
      cross-check their ``public_observation`` regimes; NOTE it is not yet
      called from sprt_gate.py or any other CLI entry point in this ticket --
      wiring it into a live gate run needs the generate config to be
      addressable from gate time (it currently is not), which is future work.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, ClassVar, TypeVar

_T = TypeVar("_T", bound="PipelineConfig")

# Bump when the *set* of fields on any pipeline config changes so that hashes
# from before/after the change are never mistaken for equal regimes.
CONFIG_SCHEMA_VERSION = 1

# Length (hex chars) of the short hash embedded in artifacts. 16 hex chars =
# 64 bits; collision probability is negligible for the run counts here and the
# full digest is always recoverable from the registry.
SHORT_HASH_LEN = 16


def _jsonable(value: Any) -> Any:
    """Normalize a field value to a deterministic, JSON-serializable form.

    Tuples/sets become sorted-agnostic lists (order preserved for tuples, which
    are ordered; sets are sorted for determinism), nested dataclasses recurse,
    and everything else is passed through to ``json.dumps`` which raises on a
    genuinely unserializable value rather than hashing an opaque ``repr``.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return [_jsonable(v) for v in sorted(value, key=repr)]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Base for the four typed pipeline configs.

    Subclasses set ``PIPELINE`` and list their science-critical fields. The
    hash and canonical payload are defined once here over ``dataclasses.fields``
    so a new field is picked up automatically.
    """

    PIPELINE: ClassVar[str] = "base"

    def field_values(self) -> dict[str, Any]:
        """Field name -> JSON-normalized value, in declaration order."""
        return {f.name: _jsonable(getattr(self, f.name)) for f in dataclasses.fields(self)}

    def canonical_payload(self) -> dict[str, Any]:
        """The full, self-describing record that gets hashed and registered."""
        return {
            "pipeline": self.PIPELINE,
            "schema_version": CONFIG_SCHEMA_VERSION,
            "fields": self.field_values(),
        }

    def canonical_json(self) -> str:
        """Sorted-key, whitespace-free JSON -- the exact bytes that are hashed.

        ``sort_keys`` makes field/declaration order irrelevant; the compact
        separators make the string stable across Python versions.
        """
        return json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        """Short, stable content hash of the fully-resolved config.

        Format ``sha256:<16 hex>``. Identical field values -> identical hash;
        any single differing field -> different hash (that is the whole point).
        """
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return f"sha256:{digest[:SHORT_HASH_LEN]}"

    def full_config_hash(self) -> str:
        """The complete ``sha256:<64 hex>`` digest (registry-side, no truncation)."""
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    @classmethod
    def _from_args(cls: type[_T], args: Any, **overrides: Any) -> _T:
        """Build ``cls`` from an argparse namespace by field name.

        For every dataclass field, take ``getattr(args, field.name)`` when the
        namespace has it, else the field's own default; ``overrides`` win last
        (for derived fields whose value is not a bare CLI flag, e.g. a test's
        mode discriminator). Field names are chosen to equal the argparse
        ``dest`` (de-dashed flag), so no per-field mapping table is needed.
        """
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name in overrides:
                kwargs[f.name] = overrides[f.name]
            elif hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        return cls(**kwargs)  # type: ignore[arg-type]


@dataclasses.dataclass(frozen=True, slots=True)
class TrainConfig(PipelineConfig):
    """Science-critical training knobs from ``tools/train_bc.py``.

    Defaults mirror the argparse defaults in ``build_arg_parser``/``main`` as of
    CAT-66. ``hidden_size`` defaults to ``None`` here because train_bc resolves
    it from ``--arch`` after parsing; ``from_namespace`` is called after that
    resolution so the recorded value is the effective width.
    """

    PIPELINE: ClassVar[str] = "train"

    arch: str = "candidate"
    data_format: str = "npz"
    track: str = "2p_no_trade"
    vps_to_win: int = 10
    # Masking / regime -- the field that is otherwise impossible to grep.
    mask_hidden_info: bool = False
    seed: int = 1
    validation_seed: int = 17
    epochs: int = 2
    max_steps: int = 0
    batch_size: int = 65536
    optimizer: str = "adam"
    weight_decay: float = 0.0
    lr: float = 2e-4
    lr_warmup_steps: int = 0
    lr_schedule: str = "flat"
    hidden_size: int | None = None
    graph_tokens: int = 32
    graph_layers: int = 4
    attention_heads: int = 8
    graph_dropout: float = 0.05
    symmetry_augment: bool = False
    symmetry_augment_events: bool = True
    soft_target_temperature: float = 0.7
    soft_target_weight: float = 0.7
    soft_target_source: str = "policy"
    soft_target_min_legal_coverage: float = 0.5
    # Value-head weights -- the group the "flag-flips" work verifies.
    policy_loss_weight: float = 1.0
    # RUN-6/EXP3: default aligned with train_bc --value-loss-weight (0.10, was 0.25).
    value_loss_weight: float = 0.10
    final_vp_loss_weight: float = 0.05
    q_loss_weight: float = 0.0
    policy_kl_anchor_weight: float = 0.0
    value_uncertainty_loss_weight: float = 0.0
    truncated_vp_margin_value_weight: float = 0.25
    freeze_modules: str = ""
    train_value_only: bool = False
    winner_sample_weight: float = 1.0
    loser_sample_weight: float = 0.3
    vp_margin_weight: float = 0.0
    forced_action_weight: float = 0.1
    advantage_policy_weighting: str = "none"
    advantage_temperature: float = 1.0
    advantage_weight_cap: float = 5.0
    advantage_weight_floor: float = 0.05

    @classmethod
    def from_namespace(cls, args: Any) -> "TrainConfig":
        return cls._from_args(args)


@dataclasses.dataclass(frozen=True, slots=True)
class GenerateConfig(PipelineConfig):
    """Science-critical self-play generation knobs from
    ``tools/generate_gumbel_selfplay_data.py``.

    ``checkpoint`` is included because the generating net's identity is part of
    the regime; ``games``/``workers``/``shard_size`` are recorded but callers
    that want a topology-invariant hash can note that a differing worker count
    still changes the hash (it changes the shard layout, so treat it as part of
    the regime). ``temperature_decisions`` is the absolute, cap-invariant flag;
    the derived ``temperature_move_fraction`` is intentionally not stored (it is
    a function of temperature_decisions/max_decisions and would double-count).
    """

    PIPELINE: ClassVar[str] = "generate"

    checkpoint: str | None = None
    track: str = "2p_no_trade"
    vps_to_win: int = 10
    obs_width: int = 806
    # Masking / regime.
    public_observation: bool = False
    belief_chance_spectra: bool = False
    # Seeds.
    base_seed: int = 1
    seed_claim: bool = True
    # Gumbel search.
    n_full: int = 64
    n_fast: int = 16
    p_full: float = 0.25
    n_full_wide: int | None = None
    raw_policy_above_width: int | None = None
    c_visit: float = 50.0
    c_scale: float = 0.1
    max_decisions: int = 600
    max_depth: int = 80
    temperature_decisions: int = 45
    temperature_high: float = 1.0
    temperature_low: float = 0.0
    prior_temperature: float = 1.0
    value_scale: float = 1.0
    correct_rust_chance_spectra: bool = True
    lazy_interior_chance: bool = False
    # Run topology (part of the regime: changes shard layout / seed mapping).
    games: int = 8
    workers: int = 1
    shard_size: int = 2048
    fmt: str = "npz"
    score_actions: bool = True
    opponent_pool_manifest: str | None = None

    @classmethod
    def from_namespace(cls, args: Any) -> "GenerateConfig":
        # ``--format`` parses to dest ``format`` (a builtin name); map it to the
        # ``fmt`` field explicitly so the rest can flow by name.
        overrides: dict[str, Any] = {}
        if hasattr(args, "format"):
            overrides["fmt"] = getattr(args, "format")
        return cls._from_args(args, **overrides)


@dataclasses.dataclass(frozen=True, slots=True)
class EvalConfig(PipelineConfig):
    """Science-critical head-to-head evaluation knobs shared by the three
    ``gumbel_search_*_h2h.py`` tools.

    A single ``mode`` discriminates the three topologies:
      * ``search_vs_raw``  -- one ``checkpoint`` plays search vs its own raw policy.
      * ``cross_net``      -- ``candidate`` vs ``baseline`` (both checkpoints).
      * ``vs_bot``         -- ``candidate`` (checkpoint) vs ``baseline_bot`` (a
                              non-neural Catanatron bot); ``map_kind`` is pinned
                              to TOURNAMENT for cross-engine parity.
    Identity fields not used by a given mode stay ``None`` so the hash cleanly
    reflects the actual matchup.
    """

    PIPELINE: ClassVar[str] = "eval"

    mode: str = "search_vs_raw"
    # Identity.
    checkpoint: str | None = None
    candidate: str | None = None
    baseline: str | None = None
    baseline_bot: str | None = None
    map_kind: str | None = None
    # Masking / regime.
    public_observation: bool = False
    belief_chance_spectra: bool = False
    # Seeds + games.
    base_seed: int = 1
    pairs: int = 50
    # Gumbel search.
    n_full: int = 64
    n_full_wide: int | None = None
    raw_policy_above_width: int | None = None
    max_depth: int = 80
    max_decisions: int = 300
    c_visit: float = 50.0
    c_scale: float = 0.1
    max_root_candidates: int = 16
    max_root_candidates_wide: int = 54
    symmetry_averaged_eval: bool = False
    correct_rust_chance_spectra: bool = True
    lazy_interior_chance: bool = False
    prior_temperature: float = 1.0
    value_scale: float = 1.0
    value_squash: str = "tanh"
    # SPRT thresholds echoed by the eval tool (the gate re-derives its own).
    elo0: float = 0.0
    elo1: float = 30.0

    @classmethod
    def from_namespace(cls, args: Any, *, mode: str, **overrides: Any) -> "EvalConfig":
        # ``overrides`` carries derived fields that are not bare CLI flags, e.g.
        # ``map_kind`` (pinned to a module constant in the vs-bot tool).
        return cls._from_args(args, mode=mode, **overrides)


@dataclasses.dataclass(frozen=True, slots=True)
class GateConfig(PipelineConfig):
    """Science-critical SPRT gate knobs from ``tools/sprt_gate.py``.

    ``test_kind`` records which branch ran (the gate selects it implicitly from
    which input flag is present); ``generation_public_observation`` is the
    ECHO of the masking regime the compared games were generated/evaluated
    under, so the gate result can be checked for consistency against the
    generate/eval config that produced its inputs (see
    ``check_masking_consistency``). It is ``None`` when unknown.
    """

    PIPELINE: ClassVar[str] = "gate"

    test_kind: str = "bernoulli"
    elo0: float = 0.0
    elo1: float = 5.0
    alpha: float = 0.05
    beta: float = 0.05
    generation_public_observation: bool | None = None

    @classmethod
    def from_namespace(cls, args: Any, *, test_kind: str, generation_public_observation: bool | None = None) -> "GateConfig":
        return cls._from_args(
            args,
            test_kind=test_kind,
            generation_public_observation=generation_public_observation,
        )


# --------------------------------------------------------------------------- #
# Cross-pipeline consistency checks.
# --------------------------------------------------------------------------- #

def check_masking_consistency(*configs: PipelineConfig) -> list[str]:
    """Return human-readable discrepancies in the public-observation regime.

    The masking regime must be identical across the generate config, the eval
    config, and the gate's echoed ``generation_public_observation``; a mismatch
    means a run is comparing games produced under different observability, which
    silently invalidates the comparison (the exact hazard behind the
    catan-checkpoint-regime-verification incidents). An empty list means
    consistent.
    """
    observed: list[tuple[str, bool]] = []
    for cfg in configs:
        value: bool | None
        if isinstance(cfg, GateConfig):
            value = cfg.generation_public_observation
        else:
            value = getattr(cfg, "public_observation", None)
        if value is not None:
            observed.append((cfg.PIPELINE, bool(value)))
    problems: list[str] = []
    if observed:
        regimes = {v for _, v in observed}
        if len(regimes) > 1:
            detail = ", ".join(f"{name}.public_observation={val}" for name, val in observed)
            problems.append(
                "masking regime mismatch across pipelines "
                f"(all compared configs must share public_observation): {detail}"
            )
    return problems


CONFIG_CLASSES: dict[str, type[PipelineConfig]] = {
    TrainConfig.PIPELINE: TrainConfig,
    GenerateConfig.PIPELINE: GenerateConfig,
    EvalConfig.PIPELINE: EvalConfig,
    GateConfig.PIPELINE: GateConfig,
}


def config_from_payload(payload: dict[str, Any]) -> PipelineConfig:
    """Rebuild a pipeline config from a ``canonical_payload`` dict.

    Reconstruction is by field NAME (task #74 discipline): unknown fields are
    dropped, missing fields take the current default. Raises for an unknown
    pipeline name.
    """
    pipeline = str(payload.get("pipeline", ""))
    cls = CONFIG_CLASSES.get(pipeline)
    if cls is None:
        raise ValueError(f"unknown pipeline {pipeline!r}; expected one of {sorted(CONFIG_CLASSES)}")
    stored = dict(payload.get("fields", {}))
    known = {f.name for f in dataclasses.fields(cls)}
    kept = {name: value for name, value in stored.items() if name in known}
    # Normalize the identity/collection fields JSON turned into lists back to
    # the dataclass's declared types where it matters (none currently need it;
    # all list-valued fields are absent from these configs), so a plain splat
    # is correct.
    return cls(**kept)  # type: ignore[arg-type]
