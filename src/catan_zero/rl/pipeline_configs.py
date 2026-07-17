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
# Schema 16 was already issued before ``boundary_value_particles`` became part
# of both generation and evaluation identity.  Reusing 16 would allow an
# evaluator payload without that field and one with it to claim the same
# schema. Schema 17 first bound the boundary-particle count across every
# search-bearing pipeline. Schema 18 additionally binds the value-label
# outcome-balancing control used by the canonical learner. Schema 19 binds the
# coverage sampler's minimum effective policy-signal admission floor; a
# fail-closed run must not share an identity with the historical no-floor run.
# Schema 20 binds the optional nominal forced-row scalar-value mass ceiling.
# A commissioned objective with a ceiling must never hash like the historical
# diagnostic-only forced-row weighting regime.
CONFIG_SCHEMA_VERSION = 20

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
        return {
            f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
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
        return {
            f.name: _jsonable(getattr(self, f.name)) for f in dataclasses.fields(self)
        }

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
        return json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        )

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

    Defaults mirror the internal engine defaults. The supported learner starts
    from the canonical entity-token architecture and streamed memmap corpus;
    legacy architectures and the whole-corpus NPZ loader require an explicit
    non-production recipe. ``hidden_size`` defaults to ``None`` here because
    train_bc resolves it from ``--arch`` after parsing; ``from_namespace`` is
    called after that resolution so the recorded value is the effective width.
    """

    PIPELINE: ClassVar[str] = "train"

    arch: str = "entity_graph"
    # Input identities belong in the science hash: changing the corpus or the
    # warm-start source changes the experiment even when every optimizer flag is
    # identical. Output checkpoint/report paths are intentionally excluded.
    data: str = ""
    data_fingerprint: str = ""
    init_checkpoint: str = ""
    init_checkpoint_sha256: str = ""
    grow_from_checkpoint: str = ""
    grow_from_checkpoint_sha256: str = ""
    resume_optimizer: bool = False
    data_format: str = "memmap"
    # Distributed execution can change the optimizer trajectory.  Keep these in
    # the typed identity in addition to train_bc's resume-only topology fields so
    # standalone config hashes never collapse distinct learner geometries.
    grad_accum_steps: int = 1
    ddp_shard_data: bool = False
    fsdp: bool = False
    track: str = "2p_no_trade"
    vps_to_win: int = 10
    # Masking / regime -- the field that is otherwise impossible to grep.
    mask_hidden_info: bool = False
    # Opt-in public card-count residual. This is an input/architecture change,
    # not a cosmetic loader flag, so it belongs in the typed science identity.
    # ``None`` means inherit the checkpoint-owned architecture. Main resolves
    # it to a concrete bool before the immutable TrainConfig is recorded.
    public_card_count_features: bool | None = None
    public_card_count_feature_schema: str = "public_card_state_v2"
    public_card_count_residual_bias: bool | None = None
    # Structured action/value residuals are checkpoint-owned architecture.
    # Main resolves the inherit sentinel to concrete bools before hashing.
    static_action_residual: bool | None = None
    legal_action_value_residual: bool | None = None
    legal_action_value_set_statistics: bool | None = None
    meaningful_public_history: bool | None = None
    meaningful_public_history_schema: str = "meaningful_public_history_2p_no_trade_v1"
    event_history_limit: int | None = None
    meaningful_public_history_pooling: str | None = None
    meaningful_public_history_target_gather: bool | None = None
    # Checkpoint-owned interpretation of player-token longest-road slot 12.
    # This changes the learner's actual input tensor and therefore must be in
    # both the experiment hash and optimizer-resume identity.
    public_award_feature_contract: str = "legacy_zero_v0"
    # This is an explicit diagnostic authorization.  It does not change the
    # legacy bridge numerics, but it changes promotion eligibility and must not
    # disappear from a sealed run identity.
    allow_mixed_public_award_feature_contracts: bool = False
    graph_history_features: bool = False
    acknowledge_empty_event_history_payload_inventory_sha256: list[str] = (
        dataclasses.field(default_factory=list)
    )
    crop_authenticated_empty_event_history: bool = False
    seed: int = 1
    # Keep data-order randomness independent from model initialization.  None
    # preserves the historical shared seed; explicit experiments bind a
    # separate sampler stream so architecture/LR arms see identical rows.
    sampler_seed: int | None = None
    # Historical DDP used an identical PyTorch RNG stream on every rank. Keep
    # that trajectory as the generic default; sealed future flywheel recipes
    # opt in so different-rank samples see independent dropout masks.
    training_rng_rank_offset: bool = False
    validation_seed: int = 17
    validation_fraction: float = 0.05
    validation_max_samples: int = 200_000
    validation_game_seed_ranges: str = ""
    # Effective immutable holdout identity.  These are populated by train_bc
    # after authenticating an A1/composite validation contract and before the
    # typed config is sealed.  They bind both the evaluated sentinel and the
    # complete game set excluded from optimizer updates.
    validation_contract_file_sha256: str = ""
    validation_game_seed_set_sha256: str = ""
    training_excluded_game_seed_set_sha256: str = ""
    allow_missing_game_seed_validation_split: bool = False
    epochs: int = 2
    max_steps: int = 0
    # Whether max_steps is an exact applied-update dose or only an upper bound.
    # This changes the optimizer trajectory when the configured epoch ceiling
    # would otherwise end first, so it must be sealed in the science identity.
    exact_max_steps: bool = False
    batch_size: int = 65536
    optimizer: str = "adam"
    weight_decay: float = 0.0
    fused_optimizer: bool = False
    amp: str = "none"
    lr: float = 2e-4
    # Global gradient-norm clipping. 0 is the explicit no-clip sentinel; the
    # historical/default trajectory remains exactly 1.0.
    max_grad_norm: float = 1.0
    lr_warmup_steps: int = 0
    lr_schedule: str = "flat"
    hidden_size: int | None = None
    graph_tokens: int = 32
    graph_layers: int = 4
    attention_heads: int = 8
    graph_dropout: float = 0.05
    entity_state_trunk: str = "transformer"
    relational_block_pattern: str = ""
    relational_ff_size: int = 0
    relational_bases: int = 4
    relational_action_cross_layers: int = 1
    latent_deliberation_steps: int = 0
    latent_deliberation_slots: int = 8
    moe_routed_experts: int = 0
    moe_top_k: int = 2
    moe_expert_ff_size: int = 0
    moe_balance_loss_weight: float = 0.01
    relational_edge_policy_head: bool = True
    symmetry_augment: bool = False
    symmetry_augment_events: bool = True
    soft_target_temperature: float = 0.7
    soft_target_weight: float = 1.0
    soft_target_source: str = "policy"
    policy_target_blend_semantics: str = "policy_target_fallback_v2"
    soft_target_min_legal_coverage: float = 0.5
    # Value-head weights -- the group the "flag-flips" work verifies.
    policy_loss_weight: float = 1.0
    # Changing this from zero to a positive floor can turn an otherwise
    # identical corpus/optimizer recipe into a refusal before its first update.
    # It is therefore part of the authoritative typed training identity.
    minimum_policy_effective_rows_per_global_batch: float = 0.0
    policy_dose_lr_area: float = 0.0
    policy_dose_reference_global_batch_size: int = 0
    post_policy_dose_value_trunk_grad_scale: float = 1.0
    # Additional policy-active rows drawn per local optimizer step.  This is a
    # science-critical sampling knob: it changes the effective policy batch
    # without changing the base/value row dose, so it must participate in the
    # typed config hash even when its backward-compatible default is zero.
    policy_aux_active_batch_size: int = 0
    # Independently normalized AUX-policy objective coefficient. Batch size is
    # only a sampling/throughput knob and must not redefine objective strength.
    policy_aux_loss_weight: float = 1.0
    policy_aux_sampling_mode: str = "weighted_with_replacement_legacy_v1"
    # RUN-6/EXP3: default aligned with train_bc --value-loss-weight (0.10, was 0.25).
    value_loss_weight: float = 0.10
    final_vp_loss_weight: float = 0.05
    q_loss_weight: float = 0.0
    q_skip_teacher_prefixes: str = "catanatron_ab"
    policy_kl_anchor_weight: float = 0.0
    policy_kl_anchor_direction: str = "forward"
    # Optional projected-dual controller over the existing authenticated
    # parent-prior anchor. ``None`` preserves the historical fixed coefficient.
    policy_kl_target: float | None = None
    policy_kl_dual_lr: float = 0.01
    policy_kl_max_weight: float = 1.0
    value_uncertainty_loss_weight: float = 0.0
    value_uncertainty_head: bool = False
    value_lr_mult: float = 1.0
    action_module_lr_mult: float = 1.0
    # Mature legal-action representation is shared by policy and the
    # legal-action-aware value path.  It needs an optimizer group independent
    # from newly initialized action-local adapters.
    shared_action_lr_mult: float = 1.0
    # Function-preserving public-card residual can be commissioned separately
    # from the mature state trunk.
    public_card_lr_mult: float = 1.0
    trunk_lr_mult: float = 1.0
    # Training-only causal intervention: scale the scalar value objective's
    # gradient at the shared-state boundary without changing its forward value
    # or the value-head parameter gradient.
    value_trunk_grad_scale: float = 1.0
    value_head_type: str = "mse"
    # ``None`` is the raw argparse/default sentinel. train_bc resolves it to the
    # effective fresh/checkpoint/grow integer before constructing a live config;
    # keeping the sentinel here also preserves typed-default == CLI-default.
    value_categorical_bins: int | None = None
    value_categorical_loss_weight: float = 0.0
    hlgauss_scalar_aux_loss_weight: float = 0.0
    value_hlgauss_sigma_ratio: float = 0.75
    value_target_lambda: float = 1.0
    value_root_blend_phases: str = ""
    value_root_blend_global_compat: bool = False
    aux_subgoal_heads: bool = False
    aux_settlement_pointer_head: bool = False
    aux_subgoal_loss_weight: float = 0.0
    belief_resource_head: bool = False
    belief_resource_loss_weight: float = 0.0
    edge_policy_head: bool = False
    truncated_vp_margin_value_weight: float = 0.25
    freeze_modules: str = ""
    train_value_only: bool = False
    teacher_weights: str = ""
    phase_weights: str = ""
    value_phase_weights: str = ""
    # Training-only value-label coverage control.  This must be part of the
    # typed identity because equal per-game mass does not by itself equalize
    # winner/loser actor mass inside a game.
    value_player_outcome_balance_mode: str = "none"
    winner_sample_weight: float = 1.0
    # Search targets remain supervised signal even when later stochastic play
    # loses the game. Outcome-conditioned downweighting is diagnostic-only.
    loser_sample_weight: float = 1.0
    vp_margin_weight: float = 0.0
    forced_action_weight: float = 0.0
    per_game_policy_weight: bool = False
    per_game_policy_weight_mode: str = "equal"
    # Explicit, versioned policy-target reliability objective.  The learner
    # validates coherent v1 row evidence before this can affect a weight.
    target_reliability_confidence_weighting: bool = False
    target_reliability_confidence_floor: float = 0.25
    forced_row_value_weight: float = 1.0
    # Optional canonical ACTION_TYPE=multiplier map applied only to forced
    # rows, after ``forced_row_value_weight``.  The empty string preserves the
    # historical objective exactly while making an enabled typed objective
    # part of the immutable learner identity.
    forced_row_value_action_type_weights: str = ""
    # Optional fail-closed admission ceiling on forced-row mass under the exact
    # nominal scalar-value objective measure. ``None`` preserves the historical
    # diagnostic-only behavior.
    maximum_nominal_forced_scalar_value_mass_fraction: float | None = None
    per_game_value_weight: bool = False
    per_game_value_weight_mode: str = "equal"
    policy_surprise_weight: float = 0.0
    policy_surprise_cap: float = 4.0
    # Exact KataGo-style within-game sampling redistribution.  This is separate
    # from the older global ``1 + scale * KL`` sampler so legacy configs and
    # seeded epoch orders remain unchanged unless explicitly opted in.
    per_game_policy_surprise_weighting: bool = False
    advantage_policy_weighting: str = "none"
    advantage_temperature: float = 1.0
    advantage_weight_cap: float = 5.0
    advantage_weight_floor: float = 0.05
    # Post-trunk action-to-target entity join. Keep this at the end of the
    # typed config so older positional dataclass payloads retain their field
    # order. ``None`` is the CLI inherit sentinel; train_bc resolves it to the
    # checkpoint-owned concrete architecture before hashing.
    action_target_gather: bool | None = None
    # Function-preserving incidence path for the legacy Transformer trunk. This
    # changes the value information surface, so it is inherited and hashed as
    # checkpoint-owned architecture. Keep new fields appended for compatibility.
    topology_residual_adapter: bool | None = None
    # Warm-start-safe action-to-board decoder for the incumbent Transformer.
    # This is distinct from ``relational_action_cross_layers``, which is consumed
    # only by RRT/ResRGCN. ``None`` inherits checkpoint topology; fresh models
    # resolve it to zero unless explicitly commissioned.
    action_cross_attention_layers: int | None = None
    action_cross_attention_bottleneck: int | None = None

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
    # A path is not an immutable model identity: the bytes at that path can be
    # replaced between runs. Bind generation provenance to the actual producer.
    producer_checkpoint_sha256: str = ""
    # Multiple independent EvalServer/generator processes sharing a device can
    # alter request-window composition and therefore numerical search paths.
    # Bind that topology in the science hash; per-pipeline identity/index stays
    # operational provenance so sibling shards remain merge-compatible.
    fleet_pipelines_per_gpu: int = 1
    device: str = "cpu"
    track: str = "2p_no_trade"
    vps_to_win: int = 10
    obs_width: int = 806
    # Masking / regime.
    public_observation: bool = False
    # Every new Gumbel shard carries this additive public-only tensor, even
    # when the producer checkpoint predates/does not consume the adapter.
    public_card_count_feature_schema: str = "public_card_state_v2"
    # Stored learner features can intentionally advance beyond the
    # checkpoint-bound teacher adapter. None preserves the historical tied
    # evaluator/row contract.
    teacher_entity_feature_adapter_version: str | None = None
    learner_entity_feature_adapter_version: str | None = None
    meaningful_public_history: bool = False
    event_history_limit: int = 64
    record_automatic_transitions: bool = True
    belief_chance_spectra: bool = False
    # Masking neural features is not sufficient for hidden-information games:
    # the search tree itself must be rooted in public-belief determinizations.
    # These fields are science-bearing provenance because changing either the
    # number of particles or their minimum budget changes the teacher target.
    information_set_search: bool = False
    # One sanitized two-player public-belief tree using the full search budget.
    # This is deliberately separate from legacy multi-particle PIMC so a
    # coherent-tree wave cannot share a config identity with four fragmented
    # determinizations.
    coherent_public_belief_search: bool = False
    determinization_particles: int = 1
    determinization_min_simulations: int = 32
    information_set_target_aggregation: str = "mean_improved_policy"
    # Forced single-action prompts do not need a neural search when their
    # policy target is masked and the learner uses terminal outcome for value.
    # Keep the historical full path as the default; the trajectory-only mode
    # is an explicit producer identity because it intentionally omits root-Q.
    forced_root_target_mode: str = "full"
    # Seeds.
    base_seed: int = 1
    seed_claim: bool = True
    # Gumbel search.
    n_full: int = 64
    n_fast: int = 16
    p_full: float = 0.25
    # Diagnostic duplicate-search dose. Zero is a strict producer no-op; when
    # enabled the generator admits only coherent exact-n128 roots.
    target_reliability_audit_fraction: float = 0.0
    target_reliability_audit_seed: int = 0
    # Preserve completed-Q and visit-count tensors needed for post-hoc target
    # stability calibration and exact policy-target reconstruction.
    preserve_search_evidence: bool = False
    # The generator has always materialized the pre-search evaluator value on
    # eligible full-search roots.  Bind that invariant into typed provenance
    # so the canonical schema20 file and the resolved GenerateConfig cannot
    # silently claim different science identities.
    preserve_root_prior_value: bool = True
    n_full_wide: int | None = None
    n_full_wide_threshold: int | None = None
    wide_roots_always_full: bool = False
    raw_policy_above_width: int | None = None
    symmetry_averaged_eval: bool = False
    symmetry_averaged_eval_threshold: int | None = None
    wide_candidates_threshold: int = 24
    c_visit: float = 50.0
    c_scale: float = 0.1
    sigma_reference_visits: int | None = None
    rescale_noise_floor_c: float = 0.0
    # Science-bearing scope for D1. Without this field an all-node D1 run and
    # an opening-road-only D1 run can share the same generation config hash,
    # allowing semantically different targets to be merged under one run id.
    rescale_noise_floor_initial_road_only: bool = False
    sigma_eval: float = 0.79
    max_decisions: int = 600
    max_depth: int = 80
    temperature_decisions: int = 45
    # ``prompt`` is the historical engine-prompt clock. ``nonforced_choice``
    # advances only when the acting player has more than one legal action, so
    # mandatory ROLL/END_TURN plumbing cannot consume the exploration window.
    temperature_clock: str = "prompt"
    temperature_high: float = 1.0
    temperature_low: float = 0.0
    late_temperature_decisions: int | None = None
    late_temperature: float = 0.0
    prior_temperature: float = 1.0
    value_scale: float = 1.0
    value_readout: str = "scalar"
    correct_rust_chance_spectra: bool = True
    lazy_interior_chance: bool = False
    exact_budget_sh: bool = False
    exact_budget_sh_min_n: int = 0
    root_wave_batching: bool = False
    # Implementation choice is science-bearing: the native loop is required
    # to be parity-gated, but recording it still prevents native/reference
    # rows from being merged under one opaque config identity.
    native_mcts_hot_loop: bool = False
    # Evaluator/transport choices can change batching composition and numeric
    # results, so they are part of provenance rather than "mere performance"
    # flags. In particular TF32 was measured to diverge self-play trajectories.
    # Native featurization is the canonical production evaluator path. Legacy
    # sealed replay manifests bind False explicitly; an omitted setting on a
    # new run must not fall back to the Python/JSON hot path.
    rust_featurize: bool = True
    eval_server: bool = False
    eval_server_max_batch: int = 64
    eval_server_max_neural_rows: int | None = None
    eval_server_max_wait_ms: float = 0.0
    eval_server_timeout_ms: float = 20_000.0
    eval_server_batch_timeout_sec: float = 0.0
    eval_server_local_fallback: bool = False
    eval_server_matmul_precision: str = "highest"
    eval_server_request_collector: bool = False
    # Request payload transport remains provenance-bearing until its parity and
    # H100 throughput arms are certified.  Shared memory changes only IPC, but
    # can change cross-game batch composition and therefore floating ordering.
    eval_server_transport: str = "mp_queue"
    eval_server_shared_memory_slot_bytes: int = 4 * 1024 * 1024
    eval_server_event_token_limit: int | None = None
    eval_server_cuda_graph: bool = False
    eval_server_cuda_graph_batch_buckets: tuple[int, ...] = (
        8,
        16,
        24,
        32,
        40,
        48,
        64,
        80,
        96,
        128,
        160,
        192,
    )
    eval_server_cuda_graph_warmup_iterations: int = 3
    eval_cache_size: int = 100_000
    # Run topology (part of the regime: changes shard layout / seed mapping).
    games: int = 8
    workers: int = 1
    shard_size: int = 2048
    fmt: str = "npz"
    score_actions: bool = True
    opponent_pool_manifest: str | None = None
    opponent_mix_manifest: str | None = None
    exploiter_fraction: float | None = None
    # Append-only producer field: number of observer-information worlds
    # averaged only at the first opponent/new-turn continuation-value
    # boundary. K=1 is the exact historical coherent-public operator; larger
    # values therefore require a distinct producer identity and config hash.
    boundary_value_particles: int = 1

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
    information_set_search: bool = False
    coherent_public_belief_search: bool = False
    forced_root_target_mode: str = "full"
    # Number of observer-information worlds averaged at the first
    # opponent/new-turn continuation-value boundary. K=1 preserves the
    # historical coherent-public evaluator exactly.
    boundary_value_particles: int = 1
    # Explicit implementation arm. False preserves the reference Python tree
    # loop; True requires the matching catanatron_rs native-search binding and
    # fails closed rather than silently changing the evaluation operator.
    native_mcts_hot_loop: bool = False
    determinization_particles: int = 1
    determinization_min_simulations: int = 32
    # Seeds + games.
    base_seed: int = 1
    pairs: int = 50
    # Gumbel search.
    n_full: int = 64
    # Cross-net H2H resolves these role-specific fields from their explicit
    # flags or the shared n_full fallback before hashing. Other eval modes
    # leave them None.
    candidate_n_full: int | None = None
    baseline_n_full: int | None = None
    n_full_wide: int | None = None
    # Same resolution contract for adaptive wide-root budgets. Recording both
    # effective sides is required to distinguish a fair adaptive-vs-uniform
    # gate from the old shared n_full_wide arm in the config hash.
    candidate_n_full_wide: int | None = None
    baseline_n_full_wide: int | None = None
    n_full_wide_threshold: int | None = None
    candidate_n_full_wide_threshold: int | None = None
    baseline_n_full_wide_threshold: int | None = None
    candidate_wide_roots_always_full: bool | None = None
    baseline_wide_roots_always_full: bool | None = None
    raw_policy_above_width: int | None = None
    # Cross-net evaluation can independently replace either role's search
    # operator with raw-prior argmax (0 => every multi-action root).  This is
    # used by the three-panel neural/search decomposition contract:
    # raw-vs-raw, searched-vs-searched, and searched-candidate-vs-own-raw.
    candidate_raw_policy_above_width: int | None = None
    baseline_raw_policy_above_width: int | None = None
    max_depth: int = 80
    max_decisions: int = 300
    c_visit: float = 50.0
    c_scale: float = 0.1
    sigma_reference_visits: int | None = None
    # Cross-net H2H resolves these from role-specific flags or the shared
    # c_scale fallback.  Persisting the effective values keeps an
    # independently tuned search-operator comparison distinct from a
    # checkpoint-only gate in both config hashes and report provenance.
    candidate_c_scale: float | None = None
    baseline_c_scale: float | None = None
    gameplay_policy_aggregation: str = "mean_improved_policy"
    candidate_gameplay_policy_aggregation: str | None = None
    baseline_gameplay_policy_aggregation: str | None = None
    rescale_noise_floor_c: float = 0.0
    candidate_rescale_noise_floor_c: float | None = None
    baseline_rescale_noise_floor_c: float | None = None
    sigma_eval: float = 0.79
    candidate_sigma_eval: float | None = None
    baseline_sigma_eval: float | None = None
    candidate_sigma_reference_visits: int | None = None
    baseline_sigma_reference_visits: int | None = None
    max_root_candidates: int = 16
    max_root_candidates_wide: int = 54
    wide_candidates_threshold: int = 24
    symmetry_averaged_eval: bool = False
    symmetry_averaged_eval_threshold: int | None = None
    correct_rust_chance_spectra: bool = True
    lazy_interior_chance: bool = False
    prior_temperature: float = 1.0
    value_scale: float = 1.0
    value_squash: str = "tanh"
    # Cross-net H2H resolves these from role-specific flags or the shared
    # value_squash fallback. This makes clip-vs-tanh comparisons part of the
    # typed evaluation identity instead of an untracked evaluator detail.
    candidate_value_squash: str | None = None
    baseline_value_squash: str | None = None
    # ``value_readout`` is the backwards-compatible shared CLI fallback.
    # Cross-net gates record the effective role-specific values as well so a
    # categorical candidate vs scalar incumbent is a distinct, auditable
    # evaluation regime and therefore receives a distinct config hash.
    value_readout: str = "scalar"
    candidate_value_readout: str | None = None
    baseline_value_readout: str | None = None
    # SPRT thresholds echoed by the eval tool (the gate re-derives its own).
    elo0: float = 0.0
    elo1: float = 30.0

    # Complete search/evaluator semantics.  These fields are intentionally
    # persisted even when the corresponding experiment is disabled: promotion
    # evidence must prove the evaluator used the sealed no-op value rather than
    # silently inheriting whatever default a newer binary happens to have.
    n_fast: int = 64
    p_full: float = 1.0
    force_full_every_decision: bool = True
    temperature: float = 0.0
    play_sh_winner: bool = False
    wide_roots_always_full: bool = False
    exact_budget_sh: bool = False
    exact_budget_sh_min_n: int = 0
    root_wave_batching: bool = False
    use_batch_api: bool = True
    policy_target_min_visits: int = 0
    uncertainty_backup_weighting: bool = False
    uncertainty_backup_a: float = 0.25
    uncertainty_backup_exp: float = 1.0
    uncertainty_backup_cap: float = 1.0
    variance_aware_q: bool = False
    variance_aware_k: float = 1.0
    variance_aware_closed_form_js: bool = False
    evaluator_context_fill: float = 0.0
    evaluator_cache_size: int = 0
    # Match generation: new evaluations use the native feature surface unless
    # a historical replay contract explicitly binds the legacy path.
    evaluator_rust_featurize: bool = True
    evaluator_emit_uncertainty: bool = False

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
    def from_namespace(
        cls,
        args: Any,
        *,
        test_kind: str,
        generation_public_observation: bool | None = None,
    ) -> "GateConfig":
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
            detail = ", ".join(
                f"{name}.public_observation={val}" for name, val in observed
            )
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
        raise ValueError(
            f"unknown pipeline {pipeline!r}; expected one of {sorted(CONFIG_CLASSES)}"
        )
    stored = dict(payload.get("fields", {}))
    known = {f.name for f in dataclasses.fields(cls)}
    kept = {name: value for name, value in stored.items() if name in known}
    # Canonical JSON normalizes tuples to lists. Restore tuple-valued defaults
    # so payload round trips retain dataclass identity, not only hash parity.
    defaults = cls()
    for name, value in tuple(kept.items()):
        if isinstance(getattr(defaults, name), tuple) and isinstance(value, list):
            kept[name] = tuple(value)
    return cls(**kept)  # type: ignore[arg-type]
