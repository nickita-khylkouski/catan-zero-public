"""Durable, name-keyed configuration for the continuous flywheel.

Serialization is name-keyed dict + schema version (NOT positional pickle) — the exact discipline
from task #74 that killed the positional-pickle SHIFT bug class. Adding a field is backward
compatible: ``from_dict`` fills missing keys from defaults; an unknown schema version is a hard
error, not a silent mis-load.

Every knob here traces to the discrete-vs-continuous research verdict (memory
``catan-discrete-vs-continuous-verdict``). The ``regime`` switch is deliberately first-class so the
SAME code path can run either the discrete-generational baseline or the continuous flywheel — that
is what makes the (never-published) clean discrete-vs-continuous ablation possible on one codebase.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field, fields

from .replay_window import DEFAULT_ALPHA, DEFAULT_BETA
from .opponent_pool import OpponentPolicy

SCHEMA_VERSION = 2


@dataclass
class FlywheelConfig:
    # --- regime (the ablation switch) ---
    regime: str = "continuous"           # "continuous" | "discrete" — same code, two data-gen policies

    # --- replay window (KataGo App. C; see replay_window.py) ---
    # c = initial window in ROWS. Retune to ~1hr of fleet self-play — do NOT copy KataGo's 250k
    # (our decisions cost ~2000 evals each). Placeholder default sized for a pilot; override in prod.
    window_c_rows: int = 300_000
    window_alpha: float = DEFAULT_ALPHA  # 0.75
    window_beta: float = DEFAULT_BETA    # 0.40
    evict_stale_shards: bool = False     # physically delete out-of-window shards to reclaim disk

    # --- sample reuse (KataGo ran ~4x; small-compute can push 4-8x; >8x needs resets) ---
    target_reuse: float = 6.0            # (steps*batch)/new-rows-this-round; incremental target
    max_reuse: float = 8.0               # hard ceiling on CUMULATIVE reuse; trainer clamps past it
    train_batch_size: int = 65536        # MUST equal train_bc.py --batch-size; drives the reuse math
                                         # (was silently mismatched 16x when hardcoded to 4096)

    # --- production-next learner objective (schema v2) ---
    # Forced actions have zero policy information: after legal masking their
    # probability is identically one, so including them can only dilute policy
    # accounting.  They still carry state/value information, but at low weight
    # because rows from one game are strongly correlated.  The per-game sqrt
    # correction retains more effective sample size than exact equalization
    # while preventing long games from dominating the scalar value objective.
    learner_forced_action_weight: float = 0.0
    # Keep forced-state value mass unchanged until the matched {1.0, 0.25}
    # experiment adjudicates it.  Per-game normalization below already removes
    # the much larger long-game replication bias.
    learner_forced_row_value_weight: float = 1.0
    learner_per_game_policy_weight: bool = True
    learner_per_game_policy_weight_mode: str = "sqrt"
    learner_per_game_value_weight: bool = True
    learner_per_game_value_weight_mode: str = "sqrt"
    learner_loser_sample_weight: float = 1.0
    learner_global_shuffle: bool = True
    # Catastrophic-forgetting control is mandatory.  Replay is the default;
    # a positive KL anchor remains an explicit alternative/ablation rather than
    # an unmeasured global constant.
    learner_replay_required: bool = True
    learner_policy_kl_anchor_weight: float = 0.0
    learner_policy_kl_anchor_direction: str = "forward"
    # Do not silently consume stale/bootstrap targets.  Refreshed-root lambda
    # and return-scale Q are separate experiments with their own provenance.
    learner_value_target_lambda: float = 1.0
    learner_q_loss_weight: float = 0.0

    # --- checkpoint refresh cadence (KataGo snapshot+EMA; wall-clock, not per-step) ---
    checkpoint_every_rows: int = 500_000  # publish a candidate every N new rows trained-through
    ema_decay: float = 0.75               # EMA over recent snapshots -> candidate (KataGo SWA idea)
    ema_snapshots: int = 4                # average the last N snapshots into a candidate

    # --- cheap gate (KataGo kept it, ~7% of fleet; trainer NEVER blocks on it) ---
    gate_enabled: bool = True
    gate_games: int = 150                 # legacy scoreboard-only override; masking-safe H2H uses
                                          # named flywheel config's 300 -> 600 total-game tiers
    gate_min_winrate: float = 0.50        # candidate must not REGRESS vs current champion
    gate_sims: int = 16                   # compatibility field; named H2H policy owns n_sims (=16)
    gate_noise: bool = False              # Dirichlet/forced-playout noise OFF during gating
    # "h2h" (default): tools/gumbel_search_cross_net_h2h.py, supports hidden-info masking end to end.
    # "scoreboard": tools/promotion_gate_runner.py -> evaluate_scoreboard.py, which has NO
    # public-observation masking anywhere in its policy-loading chain -- a masked-trained candidate
    # gated on this path is evaluated with omniscient features (train/eval mismatch). Kept only as an
    # historical replay option; Runner.gate rejects it for promotion because a
    # warning cannot make its omniscient evidence valid for masked checkpoints.
    gate_style: str = "h2h"               # "h2h" | "scoreboard"
    masked: bool = True                   # candidate/champion were trained with --mask-hidden-info;
                                           # threads to the h2h gate's --public-observation flag
    # Gate search-operator knobs. The h2h tool's own CLI defaults (c_scale=0.1, lazy OFF) do NOT
    # match the established gate methodology (G1 gate + n64 confirm both ran c_scale=0.03 + lazy);
    # relying on tool defaults here is the exact CLI-default-drift trap from memory
    # ``catan-cli-default-override-trap`` — so both are explicit config, always passed.
    gate_c_scale: float = 0.03            # threads to the h2h gate's --c-scale
    gate_lazy_interior_chance: bool = True  # threads to --lazy-interior-chance (cheap gate, G1 parity)
    # Role-specific gate readouts. Scalar/scalar preserves existing flywheel
    # behavior; an HL-Gauss run sets categorical/scalar to compare the trained
    # candidate readout against the legacy incumbent without requiring the
    # incumbent to have a categorical head.
    gate_candidate_value_readout: str = "scalar"
    gate_baseline_value_readout: str = "scalar"
    # A masked evaluator alone does not make search public-information safe:
    # authoritative game clones still contain opponent hands/dev cards.  The
    # production H2H gate is therefore pinned to actor-turn information-set
    # search and records the particle recipe in its artifact/config hash.
    gate_information_set_search: bool = True
    gate_determinization_particles: int = 4
    gate_determinization_min_simulations: int = 32

    # --- generation search-operator config (CAT-88: pass EXPLICITLY, never inherit
    #     generate_gumbel_selfplay_data.py's own CLI defaults). The flywheel's generate()
    #     previously omitted the search config entirely, so every subprocess silently
    #     resolved the tool defaults -- "a whole unvalidated preset incl. D1" (the tool
    #     defaults DIFFER from the canonical recipe: c_scale 0.1 vs 0.03,
    #     temperature-decisions 45 vs 90, lazy-interior-chance OFF vs ON). POSTURE =
    #     LOUD-FAIL-IF-UNSET with NO hardcoded defaults: gen config is RUN-DEPENDENT
    #     (volume n64/p0.25 vs teacher n128/p1.0), so the operator MUST specify every
    #     field; resolve_gen_search_argv() RAISES if any is None rather than silently
    #     picking a preset. Serialized into flywheel_config.json (auditable). ---
    gen_n_full: int | None = None
    gen_n_fast: int | None = None
    gen_p_full: float | None = None
    gen_n_full_wide: int | None = None
    gen_n_full_wide_threshold: int | None = None
    gen_wide_roots_always_full: bool = False
    gen_symmetry_averaged_eval: bool | None = None
    gen_symmetry_averaged_eval_threshold: int | None = None
    gen_wide_candidates_threshold: int | None = None
    gen_c_visit: float | None = None
    gen_c_scale: float | None = None
    gen_max_decisions: int | None = None
    gen_max_depth: int | None = None
    gen_temperature_decisions: int | None = None
    gen_lazy_interior_chance: bool | None = None
    gen_correct_rust_chance_spectra: bool | None = None
    gen_information_set_search: bool | None = None
    gen_determinization_particles: int | None = None
    gen_determinization_min_simulations: int | None = None

    # --- opponent pool (anti-forgetting; asymmetric-Catan critical) ---
    opponent_pool_fraction: float = 0.20
    opponent_target_winrate: float = 0.60
    opponent_recency_bias: float = 2.0

    # --- staleness handling (MuZero target-net, NOT V-trace for MCTS distillation) ---
    # NOTE: these are the SPEC for the staleness mechanism; the target-network value bootstrap is
    # NOT yet wired into train_bc's distillation loss (follow-up, tracked in the module docstring).
    # They are honest config placeholders, not live behaviour — do not assume staleness handling is
    # active until the trainer reads them.
    value_target_network: bool = True     # lag a copy of the net for the value bootstrap
    value_target_lag_rows: int = 1_000_000  # refresh the target net every N rows
    max_staleness_versions: int = 3        # drop self-play whose champion is > N versions behind

    # --- anchor telemetry (CAT-30/CAT-26; tools/build_anchor_corpus.py builds these) ---
    # DECISION RULE (Roadmap Sec 1 standing rule, R8/gen-4 lesson): anchor telemetry is a DRIFT
    # TRIPWIRE ONLY, NEVER a promotion signal. gen-4 showed "the historical promotion signature"
    # and still gated flat -- a flat anchor cannot distinguish "distillation complete" from
    # "anchor gone stale / off-distribution", so it must never gate a promote/hold decision in
    # EITHER direction. Runner.gate()/g.get("pass") in continuous_flywheel.py is the ONLY thing
    # that may decide a promotion, and it reads nothing from anchor telemetry -- enforced by
    # tests/test_continuous_flywheel_anchor_tripwire.py, not just this comment.
    anchor_corpora: list[str] = field(default_factory=list)  # ordered names, e.g. ["anchor_r7",
                                            # "anchor_gen4"]; each resolves to <loop_dir>/anchors/
                                            # <name> (built by tools/build_anchor_corpus.py, tracked
                                            # in its own anchor_manifest.json for the build history).
                                            # A plain list (not tuple): survives a JSON round-trip
                                            # (save_config/from_dict) byte-identically.
    anchor_eval_every_rounds: int = 1       # cadence for the anchor tripwire probe; 0 disables
    anchor_drift_alert_threshold: float = 0.10  # relative value-loss increase vs this anchor's
                                            # first-recorded baseline that triggers a WARNING log
                                            # line -- alert/log only, never consumed by gate()
    anchor_holdout_ranges: str = ""         # INFORMATIONAL provenance only (not consumed by the
                                            # probe): comma-separated start:end .valonly game_seed
                                            # ranges the configured anchor_corpora were built from,
                                            # so flywheel_config.json documents which reserved range
                                            # each run's anchors trace back to.

    def opponent_policy(self) -> OpponentPolicy:
        return OpponentPolicy(
            pool_fraction=self.opponent_pool_fraction,
            target_winrate=self.opponent_target_winrate,
            recency_bias=self.opponent_recency_bias,
        )

    # CAT-88: the generation search-config fields the flywheel must pass EXPLICITLY to
    # every generate_gumbel_selfplay_data.py subprocess (no silent tool-default inherit).
    _REQUIRED_GEN_FIELDS = (
        "gen_n_full", "gen_n_fast", "gen_p_full", "gen_symmetry_averaged_eval",
        "gen_wide_candidates_threshold", "gen_c_visit", "gen_c_scale",
        "gen_max_decisions", "gen_max_depth", "gen_temperature_decisions",
        "gen_lazy_interior_chance", "gen_correct_rust_chance_spectra",
        "gen_information_set_search", "gen_determinization_particles",
        "gen_determinization_min_simulations",
    )

    _REQUIRED_LEARNER_FIELDS = (
        "learner_forced_action_weight",
        "learner_forced_row_value_weight",
        "learner_per_game_policy_weight",
        "learner_per_game_policy_weight_mode",
        "learner_per_game_value_weight",
        "learner_per_game_value_weight_mode",
        "learner_loser_sample_weight",
        "learner_global_shuffle",
        "learner_replay_required",
        "learner_policy_kl_anchor_weight",
        "learner_policy_kl_anchor_direction",
        "learner_value_target_lambda",
        "learner_q_loss_weight",
    )

    def resolve_learner_argv(self) -> list[str]:
        """Return the explicit production-next learner objective.

        Keeping this at the flywheel boundary preserves historical ``train_bc``
        replay while making every new iteration use one fail-closed recipe.
        """
        self.validate()
        argv = [
            "--forced-action-weight", str(self.learner_forced_action_weight),
            "--forced-row-value-weight", str(self.learner_forced_row_value_weight),
            "--per-game-policy-weight-mode", self.learner_per_game_policy_weight_mode,
            "--per-game-value-weight-mode", self.learner_per_game_value_weight_mode,
            "--loser-sample-weight", str(self.learner_loser_sample_weight),
            "--policy-kl-anchor-weight", str(self.learner_policy_kl_anchor_weight),
            "--policy-kl-anchor-direction", self.learner_policy_kl_anchor_direction,
            "--value-target-lambda", str(self.learner_value_target_lambda),
            "--q-loss-weight", str(self.learner_q_loss_weight),
        ]
        argv.append(
            "--per-game-policy-weight"
            if self.learner_per_game_policy_weight
            else "--no-per-game-policy-weight"
        )
        argv.append(
            "--per-game-value-weight"
            if self.learner_per_game_value_weight
            else "--no-per-game-value-weight"
        )
        return argv

    def resolve_gen_search_argv(self) -> list[str]:
        """CAT-88: return the EXPLICIT generation search-config CLI args, or RAISE if any
        field is unset. gen config is RUN-DEPENDENT (volume n64/p0.25 vs teacher n128/p1.0),
        so the flywheel REFUSES to silently inherit generate_gumbel_selfplay_data.py's tool
        defaults (which differ from canonical: c_scale 0.1 vs 0.03, temperature-decisions 45
        vs 90, lazy-interior-chance OFF vs ON) -- the operator MUST set every field."""
        missing = [f for f in self._REQUIRED_GEN_FIELDS if getattr(self, f) is None]
        if missing:
            raise ValueError(
                "CAT-88: continuous_flywheel refuses to generate with unset gen search "
                f"config (run-dependent, no safe default): set {', '.join(missing)}. "
                "e.g. volume: --gen-n-full 64 --gen-n-fast 16 --gen-p-full 0.25 "
                "--no-gen-symmetry-averaged-eval --gen-wide-candidates-threshold 24 "
                "--gen-c-visit 50 --gen-c-scale 0.03 --gen-max-decisions 600 "
                "--gen-max-depth 80 --gen-temperature-decisions 90 --gen-lazy-interior-chance "
                "--gen-correct-rust-chance-spectra --gen-information-set-search "
                "--gen-determinization-particles 4 "
                "--gen-determinization-min-simulations 32; teacher: "
                "--gen-n-full 128 --gen-p-full 1.0."
            )
        argv = [
            "--n-full", str(self.gen_n_full),
            "--n-fast", str(self.gen_n_fast),
            "--p-full", str(self.gen_p_full),
            ("--wide-roots-always-full" if self.gen_wide_roots_always_full
             else "--no-wide-roots-always-full"),
            ("--symmetry-averaged-eval" if self.gen_symmetry_averaged_eval
             else "--no-symmetry-averaged-eval"),
            "--wide-candidates-threshold", str(self.gen_wide_candidates_threshold),
            "--c-visit", str(self.gen_c_visit),
            "--c-scale", str(self.gen_c_scale),
            "--max-decisions", str(self.gen_max_decisions),
            "--max-depth", str(self.gen_max_depth),
            "--temperature-decisions", str(self.gen_temperature_decisions),
            ("--lazy-interior-chance" if self.gen_lazy_interior_chance
             else "--no-lazy-interior-chance"),
            ("--correct-rust-chance-spectra" if self.gen_correct_rust_chance_spectra
             else "--no-correct-rust-chance-spectra"),
            ("--information-set-search" if self.gen_information_set_search
             else "--no-information-set-search"),
            "--determinization-particles", str(self.gen_determinization_particles),
            "--determinization-min-simulations",
            str(self.gen_determinization_min_simulations),
        ]
        if self.gen_n_full_wide is not None:
            argv.extend(["--n-full-wide", str(self.gen_n_full_wide)])
        if self.gen_n_full_wide_threshold is not None:
            argv.extend(
                ["--n-full-wide-threshold", str(self.gen_n_full_wide_threshold)]
            )
        if self.gen_symmetry_averaged_eval_threshold is not None:
            argv.extend(
                [
                    "--symmetry-averaged-eval-threshold",
                    str(self.gen_symmetry_averaged_eval_threshold),
                ]
            )
        return argv

    # ------------------------------------------------------------------ (de)serialize
    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        return d

    @staticmethod
    def from_dict(d: dict) -> "FlywheelConfig":
        sv = int(d.get("schema_version", SCHEMA_VERSION))
        if sv != SCHEMA_VERSION:
            raise ValueError(f"FlywheelConfig schema {sv} != {SCHEMA_VERSION}; migrate explicitly")
        known = {f.name for f in fields(FlywheelConfig)}
        missing_learner = set(FlywheelConfig._REQUIRED_LEARNER_FIELDS) - set(d)
        if missing_learner:
            raise ValueError(
                "FlywheelConfig v2 is missing production learner recipe fields: "
                + ", ".join(sorted(missing_learner))
            )
        return FlywheelConfig(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> "FlywheelConfig":
        from catan_zero.search.gumbel_chance_mcts import (
            information_set_particle_budgets,
        )

        if self.regime not in ("continuous", "discrete"):
            raise ValueError(f"regime must be continuous|discrete, got {self.regime!r}")
        if self.window_c_rows <= 0:
            raise ValueError("window_c_rows must be positive")
        if not (0.0 <= self.opponent_pool_fraction <= 1.0):
            raise ValueError("opponent_pool_fraction must be in [0,1]")
        if self.max_reuse < self.target_reuse:
            raise ValueError("max_reuse must be >= target_reuse")
        if self.learner_forced_action_weight != 0.0:
            raise ValueError("production learner requires forced policy weight 0")
        if not 0.0 <= self.learner_forced_row_value_weight <= 1.0:
            raise ValueError("learner_forced_row_value_weight must be in [0,1]")
        if not self.learner_per_game_policy_weight:
            raise ValueError("production learner requires per-game policy weighting")
        if not self.learner_per_game_value_weight:
            raise ValueError("production learner requires per-game value weighting")
        if self.learner_per_game_policy_weight_mode != "sqrt":
            raise ValueError("production learner requires sqrt per-game policy weighting")
        if self.learner_per_game_value_weight_mode != "sqrt":
            raise ValueError("production learner requires sqrt per-game value weighting")
        if self.learner_loser_sample_weight != 1.0:
            raise ValueError("production learner requires loser_sample_weight=1")
        if not self.learner_global_shuffle:
            raise ValueError("production learner requires a global mixed-corpus shuffle")
        if not self.learner_replay_required and self.learner_policy_kl_anchor_weight <= 0.0:
            raise ValueError("production learner requires replay or a positive anti-forgetting KL anchor")
        if self.learner_policy_kl_anchor_direction != "forward":
            raise ValueError("production learner requires forward policy-KL distillation")
        if self.learner_value_target_lambda != 1.0:
            raise ValueError("production learner requires outcome-only value targets until refreshed-root graduation")
        if self.learner_q_loss_weight != 0.0:
            raise ValueError("production learner requires q_loss_weight=0 until return-scale Q graduation")
        if self.gate_enabled and not (0.0 <= self.gate_min_winrate <= 1.0):
            raise ValueError("gate_min_winrate must be in [0,1]")
        if self.gate_style not in ("h2h", "scoreboard"):
            raise ValueError(f"gate_style must be h2h|scoreboard, got {self.gate_style!r}")
        if self.gate_style == "h2h" and self.gate_enabled:
            if not self.gate_information_set_search:
                raise ValueError(
                    "masked H2H gate requires gate_information_set_search=true"
                )
            if self.gate_determinization_particles < 1:
                raise ValueError("gate_determinization_particles must be >= 1")
            if self.gate_determinization_min_simulations < 1:
                raise ValueError(
                    "gate_determinization_min_simulations must be >= 1"
                )
        if self.gen_information_set_search is False:
            raise ValueError(
                "continuous flywheel generation is public-observation and requires "
                "gen_information_set_search=true"
            )
        if self.gen_determinization_particles is not None and self.gen_determinization_particles < 1:
            raise ValueError("gen_determinization_particles must be >= 1")
        if (
            self.gen_determinization_min_simulations is not None
            and self.gen_determinization_min_simulations < 1
        ):
            raise ValueError("gen_determinization_min_simulations must be >= 1")
        for field_name in ("gate_candidate_value_readout", "gate_baseline_value_readout"):
            value = getattr(self, field_name)
            if value not in ("scalar", "categorical"):
                raise ValueError(
                    f"{field_name} must be scalar|categorical, got {value!r}"
                )
        if self.anchor_eval_every_rounds < 0:
            raise ValueError("anchor_eval_every_rounds must be >= 0 (0 disables)")
        if self.anchor_drift_alert_threshold < 0.0:
            raise ValueError("anchor_drift_alert_threshold must be >= 0")
        if self.gen_wide_roots_always_full and self.gen_n_full_wide is None:
            raise ValueError(
                "gen_wide_roots_always_full requires gen_n_full_wide"
            )
        if (
            self.gen_information_set_search
            and self.gen_n_full_wide is not None
            and self.gen_n_full is not None
            and self.gen_determinization_particles is not None
            and self.gen_determinization_min_simulations is not None
        ):
            base_budgets = information_set_particle_budgets(
                self.gen_n_full,
                self.gen_determinization_particles,
                self.gen_determinization_min_simulations,
            )
            wide_budgets = information_set_particle_budgets(
                self.gen_n_full_wide,
                self.gen_determinization_particles,
                self.gen_determinization_min_simulations,
            )
            if len(set(base_budgets + wide_budgets)) != 1:
                raise ValueError(
                    "adaptive generation must preserve per-particle simulation "
                    f"dose: base={base_budgets}, wide={wide_budgets}; increase "
                    "gen_determinization_particles so adaptive compute expands "
                    "hidden-world coverage"
                )
        if len(self.anchor_corpora) != len(set(self.anchor_corpora)):
            raise ValueError(f"anchor_corpora must not contain duplicate names: {self.anchor_corpora}")
        # sanity: opponent policy constructs
        self.opponent_policy()
        return self


if __name__ == "__main__":  # self-test
    import json

    cfg = FlywheelConfig().validate()
    # round-trips by NAME (order-independent)
    d = cfg.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    cfg2 = FlywheelConfig.from_dict(d)
    assert cfg2 == cfg

    # forward-compat: an extra unknown key is ignored, missing keys fill from defaults
    d2 = dict(d); d2["some_future_knob"] = 123; del d2["gate_games"]
    cfg3 = FlywheelConfig.from_dict(d2)
    assert cfg3.gate_games == FlywheelConfig().gate_games

    # wrong schema is a hard error
    bad = dict(d); bad["schema_version"] = 999
    try:
        FlywheelConfig.from_dict(bad)
        raise AssertionError("expected schema mismatch to raise")
    except ValueError:
        pass

    # discrete regime validates too
    FlywheelConfig(regime="discrete").validate()
    try:
        FlywheelConfig(regime="bogus").validate()
        raise AssertionError("expected bad regime to raise")
    except ValueError:
        pass

    # opponent policy derives
    op = FlywheelConfig(opponent_pool_fraction=0.25).opponent_policy()
    assert op.pool_fraction == 0.25

    # gate_style defaults to the masking-safe h2h path; scoreboard is a valid opt-out; bogus rejected
    assert FlywheelConfig().gate_style == "h2h"
    assert FlywheelConfig().masked is True
    FlywheelConfig(gate_style="scoreboard").validate()

    # anchor telemetry config: defaults are inert (no anchors configured), a real JSON
    # round-trip (not just the in-memory dict above) preserves the list, and duplicate
    # names / negative knobs are rejected.
    assert FlywheelConfig().anchor_corpora == []
    cfg_anchors = FlywheelConfig(anchor_corpora=["anchor_r7", "anchor_gen4"]).validate()
    json_round_tripped = FlywheelConfig.from_dict(json.loads(json.dumps(cfg_anchors.to_dict())))
    assert json_round_tripped == cfg_anchors
    assert json_round_tripped.anchor_corpora == ["anchor_r7", "anchor_gen4"]
    try:
        FlywheelConfig(anchor_corpora=["anchor_r7", "anchor_r7"]).validate()
        raise AssertionError("expected duplicate anchor name to raise")
    except ValueError:
        pass
    try:
        FlywheelConfig(anchor_eval_every_rounds=-1).validate()
        raise AssertionError("expected negative anchor_eval_every_rounds to raise")
    except ValueError:
        pass

    try:
        FlywheelConfig(gate_style="bogus").validate()
        raise AssertionError("expected bad gate_style to raise")
    except ValueError:
        pass

    print("config self-test OK")
