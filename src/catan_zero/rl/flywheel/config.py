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

SCHEMA_VERSION = 1


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

    # --- checkpoint refresh cadence (KataGo snapshot+EMA; wall-clock, not per-step) ---
    checkpoint_every_rows: int = 500_000  # publish a candidate every N new rows trained-through
    ema_decay: float = 0.75               # EMA over recent snapshots -> candidate (KataGo SWA idea)
    ema_snapshots: int = 4                # average the last N snapshots into a candidate

    # --- cheap gate (KataGo kept it, ~7% of fleet; trainer NEVER blocks on it) ---
    gate_enabled: bool = True
    gate_games: int = 150                 # ~50-200 games; total games, split into gate_games/2 pairs
    gate_min_winrate: float = 0.50        # candidate must not REGRESS vs current champion
    gate_sims: int = 16                   # reduced sim budget for cheap, low-variance gating (n_full)
    gate_noise: bool = False              # Dirichlet/forced-playout noise OFF during gating
    # "h2h" (default): tools/gumbel_search_cross_net_h2h.py, supports hidden-info masking end to end.
    # "scoreboard": tools/promotion_gate_runner.py -> evaluate_scoreboard.py, which has NO
    # public-observation masking anywhere in its policy-loading chain -- a masked-trained candidate
    # gated on this path is evaluated with omniscient features (train/eval mismatch). Kept only as an
    # opt-out; loudly warns when selected. Do not use for masked checkpoints.
    gate_style: str = "h2h"               # "h2h" | "scoreboard"
    masked: bool = True                   # candidate/champion were trained with --mask-hidden-info;
                                           # threads to the h2h gate's --public-observation flag
    # Gate search-operator knobs. The h2h tool's own CLI defaults (c_scale=0.1, lazy OFF) do NOT
    # match the established gate methodology (G1 gate + n64 confirm both ran c_scale=0.03 + lazy);
    # relying on tool defaults here is the exact CLI-default-drift trap from memory
    # ``catan-cli-default-override-trap`` — so both are explicit config, always passed.
    gate_c_scale: float = 0.03            # threads to the h2h gate's --c-scale
    gate_lazy_interior_chance: bool = True  # threads to --lazy-interior-chance (cheap gate, G1 parity)

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
    gen_c_visit: float | None = None
    gen_c_scale: float | None = None
    gen_max_decisions: int | None = None
    gen_max_depth: int | None = None
    gen_temperature_decisions: int | None = None
    gen_lazy_interior_chance: bool | None = None
    gen_correct_rust_chance_spectra: bool | None = None

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
        "gen_n_full", "gen_n_fast", "gen_p_full", "gen_c_visit", "gen_c_scale",
        "gen_max_decisions", "gen_max_depth", "gen_temperature_decisions",
        "gen_lazy_interior_chance", "gen_correct_rust_chance_spectra",
    )

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
                "--gen-c-visit 50 --gen-c-scale 0.03 --gen-max-decisions 600 "
                "--gen-max-depth 80 --gen-temperature-decisions 90 --gen-lazy-interior-chance "
                "--gen-correct-rust-chance-spectra; teacher: --gen-n-full 128 --gen-p-full 1.0."
            )
        return [
            "--n-full", str(self.gen_n_full),
            "--n-fast", str(self.gen_n_fast),
            "--p-full", str(self.gen_p_full),
            "--c-visit", str(self.gen_c_visit),
            "--c-scale", str(self.gen_c_scale),
            "--max-decisions", str(self.gen_max_decisions),
            "--max-depth", str(self.gen_max_depth),
            "--temperature-decisions", str(self.gen_temperature_decisions),
            ("--lazy-interior-chance" if self.gen_lazy_interior_chance
             else "--no-lazy-interior-chance"),
            ("--correct-rust-chance-spectra" if self.gen_correct_rust_chance_spectra
             else "--no-correct-rust-chance-spectra"),
        ]

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
        return FlywheelConfig(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> "FlywheelConfig":
        if self.regime not in ("continuous", "discrete"):
            raise ValueError(f"regime must be continuous|discrete, got {self.regime!r}")
        if self.window_c_rows <= 0:
            raise ValueError("window_c_rows must be positive")
        if not (0.0 <= self.opponent_pool_fraction <= 1.0):
            raise ValueError("opponent_pool_fraction must be in [0,1]")
        if self.max_reuse < self.target_reuse:
            raise ValueError("max_reuse must be >= target_reuse")
        if self.gate_enabled and not (0.0 <= self.gate_min_winrate <= 1.0):
            raise ValueError("gate_min_winrate must be in [0,1]")
        if self.gate_style not in ("h2h", "scoreboard"):
            raise ValueError(f"gate_style must be h2h|scoreboard, got {self.gate_style!r}")
        if self.anchor_eval_every_rounds < 0:
            raise ValueError("anchor_eval_every_rounds must be >= 0 (0 disables)")
        if self.anchor_drift_alert_threshold < 0.0:
            raise ValueError("anchor_drift_alert_threshold must be >= 0")
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
