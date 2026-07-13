"""Gumbel-AlphaZero search with true chance nodes over the Rust engine.

Implements the root/non-root selection rules and completed-Q formalism from
Danihelka et al., "Policy improvement by planning with Gumbel" (ICLR 2022),
mirroring the semantics of `mctx.gumbel_muzero_policy`:

- Root: Gumbel-Top-k candidate sampling followed by Sequential Halving, using
  completed Q-values (unvisited actions get the mixed value estimate, not 0)
  to re-score surviving candidates each round.
- Non-root: deterministic selection of the action maximizing
  ``pi'(a) - visits(a) / (1 + total_visits)`` where ``pi'`` is the improved
  policy ``softmax(logits + sigma(completed_q))``.
- Chance: ROLL actions are true chance nodes -- all 11 dice outcomes are
  enumerated with exact `spectrum_json` probabilities and the node value backs
  up the probability-weighted average of the (possibly still-refining) child
  values. MOVE_ROBBER-steal and BUY_DEVELOPMENT_CARD chance outcomes are
  single-sampled per traversal, same as `RustMCTS`.

This module is the reusable search core; it is agnostic to the evaluator
implementation as long as it exposes the same `.evaluate(...)` interface as
`HeuristicRustEvaluator` / `EntityGraphRustEvaluator` /
`BatchedEntityGraphRustEvaluator`.

Known Rust engine chance-spectrum bugs (equivalence harness findings A19/A20,
see `catan_zero.adapters.engine_equivalence`'s module docstring and
`apply_chance_step`): `spectrum_json` for MOVE_ROBBER-with-victim always
returns uniform 0.2 over the 5 resources regardless of the victim's actual
hand (applying an outcome for a resource the victim doesn't hold silently
no-ops -- no card is stolen), and for BUY_DEVELOPMENT_CARD it can include a
large-probability "phantom" outcome that draws no card at all. ROLL's
spectrum is verified bit-identical and is untouched. When
`GumbelChanceMCTSConfig.correct_rust_chance_spectra` is True (the default),
`_traverse_single_sample` recomputes MOVE_ROBBER-with-victim weights from the
victim's real hand and filters BUY_DEVELOPMENT_CARD's phantom outcomes before
sampling, exactly mirroring the harness's proven-correct workaround. Set it
False to A/B against a future Rust wheel fix.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from catan_zero.search.rust_mcts import (
    HeuristicRustEvaluator,
    RustEvaluator,
    _legal_action_indices,
    _normalize_policy,
    _playable_action_json_by_index,
    _require_rust_module,
    _spectrum,
    _terminal_or_zero,
)
from catan_zero.search.public_belief import PublicBelief

__all__ = [
    "GumbelChanceMCTSConfig",
    "SearchResult",
    "GumbelChanceMCTS",
    "HeuristicRustEvaluator",
    "RustEvaluator",
    "sequential_halving_schedule",
    "exact_budget_sh_phases",
    "information_set_particle_budgets",
    "_root_candidate_count",
    "_prune_policy_target",
    "RESOURCES",
    "DEVELOPMENT_CARDS",
    "is_move_robber_with_victim",
    "move_robber_victim_outcome_weights",
    "buy_development_card_real_outcomes",
    "belief_move_robber_outcome_weights",
    "belief_buy_development_card_outcomes",
    "BASE_DEVELOPMENT_DECK",
    "batch_api_available",
]

# Rust engine's fixed resource/dev-card serialization order (verified against
# `catan_zero.adapters.engine_equivalence`'s equivalence harness).
RESOURCES: tuple[str, ...] = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
DEVELOPMENT_CARDS: tuple[str, ...] = (
    "KNIGHT",
    "YEAR_OF_PLENTY",
    "MONOPOLY",
    "ROAD_BUILDING",
    "VICTORY_POINT",
)

_BATCH_API_AVAILABLE: bool | None = None


def batch_api_available() -> bool:
    """Whether the installed catanatron_rs wheel exposes the 0.1.2 batch API.

    Cached after the first check (module import doesn't fail without the
    Rust extension, but `GumbelChanceMCTS.__init__` already requires it via
    `_require_rust_module`, so by the time this matters the module is
    guaranteed importable).
    """
    global _BATCH_API_AVAILABLE
    if _BATCH_API_AVAILABLE is None:
        try:
            import catanatron_rs  # type: ignore

            _BATCH_API_AVAILABLE = hasattr(catanatron_rs.Game, "decision_context_json") and hasattr(
                catanatron_rs.Game, "apply_chance_outcomes_batch"
            )
        except ImportError:
            _BATCH_API_AVAILABLE = False
    return _BATCH_API_AVAILABLE


def _decision_context(
    game: Any, *, colors: tuple[str, ...], map_kind: str | None
) -> tuple[tuple[int, ...], dict[int, Any], dict[int, tuple[tuple[int, float], ...]]]:
    """One-round-trip node-expansion context via `Game.decision_context_json`.

    Returns (legal_actions, action_json_by_id, spectrum_by_id) -- `spectrum_by_id`
    only has entries for chance-typed actions (matches `_spectrum`'s
    (outcome_index, probability) pair format, already normalized by the
    engine). Replaces `_legal_action_indices` + `_playable_action_json_by_index`
    + per-action `spectrum_json` calls with a single call.
    """
    raw = json.loads(game.decision_context_json(list(colors), map_kind, True))
    legal_actions: list[int] = []
    action_json_by_id: dict[int, Any] = {}
    spectrum_by_id: dict[int, tuple[tuple[int, float], ...]] = {}
    for entry in raw["actions"]:
        action_id = int(entry["index"])
        legal_actions.append(action_id)
        action_json_by_id[action_id] = entry["action"]
        spectrum = entry.get("spectrum")
        if spectrum is not None:
            spectrum_by_id[action_id] = tuple(
                (index, float(probability)) for index, probability in enumerate(spectrum)
            )
    return tuple(legal_actions), action_json_by_id, spectrum_by_id


def sequential_halving_schedule(m: int, n_simulations: int) -> list[tuple[int, int]]:
    """Return the (candidate_count, sims_per_candidate) schedule for Sequential Halving.

    Standard fixed-round-count Sequential Halving (Karnin et al. 2013), as used by
    `mctx.gumbel_muzero_policy`: with `m` starting candidates, run
    `ceil(log2(m))` rounds, each round giving every surviving candidate
    `floor(n_simulations / (num_rounds * count))` (at least 1) simulations, then
    keeping the top half (at least 1) for the next round. When
    `n_simulations < m * num_rounds` the `at least 1` floor means total usage can
    exceed `n_simulations` -- this is expected/standard behavior, not a bug; callers
    should provision `n_simulations >= m * ceil(log2(m))` to avoid overrun.
    """
    m = max(int(m), 1)
    num_rounds = max(1, math.ceil(math.log2(m))) if m > 1 else 1
    schedule: list[tuple[int, int]] = []
    count = m
    for _round_index in range(num_rounds):
        budget = max(1, int(n_simulations) // (num_rounds * count))
        schedule.append((count, budget))
        count = max(1, count // 2)
    return schedule


def exact_budget_sh_phases(m: int, n_simulations: int) -> list[tuple[int, int]]:
    """Exact-budget Sequential Halving phases (task #61), mctx-conformant.

    Returns [(candidates_to_visit, sims_per_candidate), ...] where each phase
    visits the TOP-`candidates_to_visit` of the caller's current re-ranked
    survivor list, and the total over all phases is EXACTLY `n_simulations` --
    replicating `mctx.policies._sequence_of_considered_visits` semantics:

    - per-pass extra visits = max(1, floor(n / (ceil(log2(m)) * considered)))
      (same floor as `sequential_halving_schedule`), BUT the sequence is
      truncated at exactly `n` simulations instead of always finishing every
      round -- the truncation mctx applies via `sequence[:num_simulations]`
      and the legacy schedule above omits. Without it the >=1-sim floor
      overruns the budget whenever n < m * ceil(log2(m)): the production
      configs' real costs are n_full=64 at a 54-wide placement root -> 104
      sims (1.63x) and n_fast=16 at m=16 -> 30 sims (1.88x).
    - a truncated final pass visits the top-`leftover` survivors once each
      (mctx's sequence order within a pass is ranked-candidate order, so
      truncation lands on the current best-ranked prefix).
    - halving keeps a floor of 2 considered candidates (mctx's
      `max(2, considered // 2)`; the legacy schedule floors at 1) and keeps
      spending passes at 2 until the budget is gone, so large budgets are
      still fully spent.

    The legacy `sequential_halving_schedule` is left untouched for the
    default (`GumbelChanceMCTSConfig.exact_budget_sh=False`) path.
    """
    m = max(int(m), 1)
    n = max(int(n_simulations), 1)
    if m == 1:
        return [(1, n)]
    log2max = math.ceil(math.log2(m))
    phases: list[tuple[int, int]] = []
    total = 0
    considered = m
    while total < n:
        extra = max(1, int(n / (log2max * considered)))
        full_passes = min(extra, (n - total) // considered)
        if full_passes > 0:
            phases.append((considered, full_passes))
            total += considered * full_passes
        if full_passes < extra:
            leftover = n - total
            if leftover > 0:
                phases.append((leftover, 1))
                total = n
            break
        considered = max(2, considered // 2)
    return phases


def information_set_particle_budgets(
    total_budget: int,
    requested_particles: int,
    min_simulations_per_particle: int,
) -> tuple[int, ...]:
    """Return the exact PIMC sub-budget assigned to each hidden-world particle.

    Keeping this calculation public and pure lets launchers validate adaptive
    search recipes against the same arithmetic used at runtime.  In particular,
    an n128/P4/min32 root resolves to ``(32, 32, 32, 32)`` while an
    n256/P8/min32 root resolves to eight 32-simulation particles.  That spends
    extra compute on belief-world coverage without silently doubling each
    particle's visit-dependent sigma sharpening.
    """
    total = max(int(total_budget), 1)
    requested = int(requested_particles)
    minimum = int(min_simulations_per_particle)
    if requested < 1:
        raise ValueError("requested_particles must be >= 1")
    if minimum < 1:
        raise ValueError("min_simulations_per_particle must be >= 1")
    count = min(requested, max(1, total // minimum))
    base, remainder = divmod(total, count)
    return tuple(base + int(index < remainder) for index in range(count))


def _root_candidate_count(num_legal: int, config: "GumbelChanceMCTSConfig") -> int:
    """CAT-62: how many of the root's `num_legal` legal actions the Gumbel-Top-k
    considered set includes, given `config`. Pulled out of `_run_root_search` as a
    pure function (no `_GNode`/game/evaluator dependency) so the cap-binding math is
    directly unit-testable. See `GumbelChanceMCTSConfig.root_candidate_cap`'s
    docstring for why the override is a separate knob rather than a change to
    `max_root_candidates`/`max_root_candidates_wide`'s defaults. Always clamped to
    `num_legal` (a cap never pads UP to itself -- "K only binds when legal > K") and
    to at least 1 (an empty considered set would break Sequential Halving)."""
    num_legal = int(num_legal)
    if config.root_candidate_cap is not None:
        m = min(num_legal, int(config.root_candidate_cap))
    elif num_legal > int(config.wide_candidates_threshold):
        m = min(num_legal, int(config.max_root_candidates_wide))
    else:
        m = min(num_legal, int(config.max_root_candidates))
    return max(m, 1)


def _matches_explicit_or_legacy_width_gate(
    num_legal: int,
    *,
    min_legal_actions: int | None,
    legacy_exclusive_threshold: int,
) -> bool:
    """Evaluate a decoupled root-width gate without changing legacy defaults.

    New explicit thresholds are inclusive (``num_legal >= min_legal_actions``),
    matching recipe language such as "every >=40-action opening root".  When
    the new field is unset, preserve the historical
    ``num_legal > wide_candidates_threshold`` contract exactly.
    """
    if min_legal_actions is None:
        return int(num_legal) > int(legacy_exclusive_threshold)
    return int(num_legal) >= int(min_legal_actions)


def _wide_budget_applies(num_legal: int, config: "GumbelChanceMCTSConfig") -> bool:
    return config.n_full_wide is not None and _matches_explicit_or_legacy_width_gate(
        num_legal,
        min_legal_actions=config.n_full_wide_threshold,
        legacy_exclusive_threshold=config.wide_candidates_threshold,
    )


def _choose_full_search(
    config: "GumbelChanceMCTSConfig",
    *,
    force_full: bool | None,
    wide_budget_root: bool,
    random_draw: Callable[[], float],
) -> bool:
    """Resolve playout-cap randomization with an opt-in wide-root override."""
    if force_full is not None:
        return bool(force_full)
    if bool(config.wide_roots_always_full) and wide_budget_root:
        return True
    return float(random_draw()) < float(config.p_full)


def _prune_policy_target(
    policy: dict[int, float], visits: dict[int, int], *, min_visits: int
) -> dict[int, float]:
    """CAT-62 low-evidence pruning: zero out (never remove -- see
    `GumbelChanceMCTSConfig.policy_target_min_visits`'s docstring on the key-set
    contract) probability mass on actions with fewer than `min_visits` visits, then
    renormalize the remaining mass to sum to 1. A pure function over plain dicts (no
    `_GNode` dependency) so prune+renormalize correctness is directly unit-testable.

    `min_visits <= 0` or an empty `policy` is an exact no-op (returns `policy`
    unchanged). If every action would be pruned (e.g. an unusually small sim budget
    left every candidate below `min_visits`), also returns `policy` unchanged rather
    than emit a degenerate all-zero distribution -- pruning is a strict restriction
    of an already-valid target, never a replacement for one when it would otherwise
    empty out.
    """
    if min_visits <= 0 or not policy:
        return policy
    kept_mass = sum(
        prob for action_id, prob in policy.items() if visits.get(action_id, 0) >= min_visits
    )
    if kept_mass <= 0.0:
        return policy
    return {
        action_id: (prob / kept_mass if visits.get(action_id, 0) >= min_visits else 0.0)
        for action_id, prob in policy.items()
    }


@dataclass(frozen=True, slots=True)
class GumbelChanceMCTSConfig:
    colors: tuple[str, ...] = ("RED", "BLUE")
    map_kind: str | None = None
    max_depth: int = 80
    seed: int = 0
    # Root Gumbel-Top-k + Sequential Halving. max_root_candidates_wide=54 is
    # full-width at placement decisions (54 legal BUILD_SETTLEMENT/ROAD
    # candidates -- see entity_token_features.py's 54 vertex tokens): a
    # narrower cap left some legal placements never explored regardless of
    # Gumbel noise, and full width costs only ~0.5% more compute (F8).
    max_root_candidates: int = 16
    max_root_candidates_wide: int = 54
    wide_candidates_threshold: int = 24
    # Exact-budget Sequential Halving (task #61): when True, `_run_root_search`
    # follows `exact_budget_sh_phases` so total root simulations equal
    # n_simulations exactly (the legacy `sequential_halving_schedule` overruns
    # the budget by the >=1-sim floor). Default False keeps the legacy path
    # bit-for-bit. `exact_budget_sh_min_n` gates the exact schedule to budgets
    # at or above the threshold (0 = always apply when exact_budget_sh=True).
    exact_budget_sh: bool = False
    exact_budget_sh_min_n: int = 0
    # Batch one ready neural leaf from each independent forced root candidate
    # during a Sequential-Halving visit wave. Every candidate keeps its own
    # deterministic RNG stream and receives exactly the schedule's visit budget.
    # The independent streams intentionally change which random chance samples
    # are assigned to each candidate (and the later high-temperature action draw),
    # so this is a statistically equivalent scheduling candidate rather than a
    # bit-identical rewrite. Default OFF retains the legacy action-major traversal
    # and RNG ordering while the batched path is strength-gated on production
    # checkpoints.
    root_wave_batching: bool = False
    # Completed-Q sigma transform (ported verbatim from mctx's
    # qtransform_completed_by_mix_value, see `_rescale_completed_q` +
    # `_improved_policy`): sigma(q) = (c_visit + max_visits) * c_scale *
    # rescale_to_unit_interval(q) -- note the rescale, and note c_scale here
    # is mctx's `value_scale` (0.1 default), NOT an arbitrary multiplier on
    # raw Q. Kept the `c_scale` field name (existing CLI plumbing in
    # generate_gumbel_selfplay_data.py already exposes --c-scale, and
    # `EntityGraphRustEvaluatorConfig` has an unrelated `value_scale` field
    # for the network's value head, so reusing that name here would collide
    # in scripts wiring up both configs) but ported the correct DEFAULT and
    # semantics (F1a/F1b): without the rescale, c_scale=1.0 applied to raw Q
    # produced unbounded sharpening (vs mctx's bounded <=6.5 nats) -- the
    # verified root cause of near-one-hot targets, the 50-65% self-agreement
    # collapse, and (since the PLAYED move also follows
    # argmax(logits+scale*completed_q)) an unreliable H2H/Gate A signal.
    c_visit: float = 50.0
    c_scale: float = 0.1
    # Final action selection: argmax(improved_policy) when temperature<=0,
    # otherwise sample from improved_policy for self-play diversity. Set
    # play_sh_winner=True to instead play the raw Sequential Halving winner
    # (what the paper's reference algorithm plays) -- kept as an A/B knob;
    # the policy-improvement guarantee this module relies on is for
    # improved_policy as a *training target*, not for the SH winner as a move.
    temperature: float = 0.0
    play_sh_winner: bool = False
    # Softens (>1) or sharpens (<1) the evaluator's prior before it enters the
    # search as logits: logits = log(prior) / prior_temperature.
    prior_temperature: float = 1.0
    # Playout-cap randomization (Wu, "Accelerating Self-Play Learning in Go"):
    # only full-search moves should emit policy targets downstream.
    n_full: int = 64
    n_fast: int = 16
    p_full: float = 0.25
    # ARM (placement budget asymmetry, default disabled): when set and a FULL
    # search hits a wide root, spend `n_full_wide` simulations there instead of
    # `n_full`. `n_full_wide_threshold` decouples this budget gate from the
    # candidate-cap/D6 threshold: an explicit value is an inclusive minimum
    # legal-action count (40 => every >=40-action root); None preserves the old
    # `len(legal) > wide_candidates_threshold` rule exactly. Motivation: Gate-A
    # losses concentrate at wide placement roots where n_full=64 over ~54
    # candidates leaves ~1.2 sims/candidate. None (default) => use `n_full`
    # everywhere, a pure no-op.
    n_full_wide: int | None = None
    # ARM (phase-gated search, default disabled): when set, at any root wider than
    # `raw_policy_above_width` SKIP search entirely and play argmax(prior) (the raw
    # policy). The returned SearchResult carries used_full_search=False so the
    # self-play row builder emits policy_weight=0 for it (raw argmax is not a
    # policy-improvement target worth imitating). Motivation: if the value net is
    # per-position biased at wide placement roots (SNR arms showed more search
    # converges confidently to the WRONG placement), deferring to the raw prior at
    # exactly those roots may beat searching them. None (default) => always search,
    # a pure no-op.
    raw_policy_above_width: int | None = None
    # Lazy interior chance evaluation (#52): at interior nodes (depth > 0),
    # traverse ROLL actions through the single-sample path (materialize +
    # evaluate ONLY the sampled dice outcome) instead of enumerating and
    # evaluating all ~11 children up front. The ROOT is unaffected in both
    # modes: root ROLL actions keep the full enumeration (exact
    # afterstate_values), and the forced-single-action fast path is a
    # separate code path entirely. Interior backups become single-sample
    # Monte Carlo estimates of the same expectation (unbiased, higher
    # variance). Measured on the F1-corrected search (mid-game states):
    # full n_full=64 searches drop from ~5,400 to ~83 leaf evals; the cost
    # is target noise (self argmax-stability 51% vs 56% at n=128 vs full-64)
    # and a genuinely different improvement operator under prior-weighted
    # v_mix (lazy-128-vs-full-64 pick agreement 37-53%). Default OFF;
    # enabling it for generation is gated on a strength-based H2H A/B
    # (optimizer's #52 report, 2026-07-04).
    lazy_interior_chance: bool = False
    # Work around verified Rust engine chance-spectrum bugs (A19/A20) for
    # MOVE_ROBBER-with-victim and BUY_DEVELOPMENT_CARD by recomputing correct
    # weights ourselves before sampling (see module docstring). Set False to
    # A/B against a future corrected Rust wheel.
    correct_rust_chance_spectra: bool = True
    # Use the 0.1.2 wheel's `decision_context_json`/`apply_chance_outcomes_batch`
    # batch API (one round-trip per node expansion instead of
    # playable_action_indices + playable_actions_json + per-action
    # spectrum_json; one call to materialize all ROLL children instead of 11
    # sequential apply_chance_outcome calls). Auto-gated at runtime by
    # `hasattr(catanatron_rs.Game, "decision_context_json")` regardless of
    # this flag's value, so it transparently falls back to the legacy path on
    # older wheels (e.g. the local dev mirror's 0.1.0) even if left True.
    use_batch_api: bool = True
    # --- Search-reliability experiments (f70, default OFF = exact no-op) ---
    # D1: noise-floor attenuation of the completed-Q rescale. The min-max
    # `_rescale_completed_q` stretches WHATEVER raw-Q spread it sees to fill
    # [0, 1], so a spread that is pure 1-2-sample sampling noise is treated
    # identically to a genuinely separated true-value spread -- the verified
    # Gate-A mechanism (manufactured confidence swamping a near-flat prior at
    # 54-wide placement roots). When `rescale_noise_floor_c > 0`, each node's
    # rescaled completed-Q is blended toward the neutral 0.5 by
    #   alpha = raw_spread / (raw_spread + c * sigma_eval / sqrt(mean_visits))
    #   out   = 0.5 + alpha * (rescaled - 0.5)
    # where `raw_spread = max(raw_q) - min(raw_q)` over the node's candidates
    # and `mean_visits` is the mean visit count across those candidates. The
    # denominator's second term is the expected sampling noise of a
    # per-candidate Q estimate (`sigma_eval / sqrt(mean_visits)`), so when the
    # observed spread is at or below that floor the rescaled values collapse
    # toward neutral (prior order preserved) and when the spread dwarfs the
    # floor alpha -> 1 and the rescale is untouched. `c` scales the floor's
    # aggressiveness; 0.0 disables the blend entirely (exact current
    # behavior, guaranteed short-circuit before any division).
    rescale_noise_floor_c: float = 0.0
    # Per-eval value-estimate noise standard deviation, the sigma in D1's
    # noise floor. 0.79 is a ROUGH estimate from corr(q, z) on the BC corpus
    # (a placeholder -- it MUST be re-calibrated empirically per checkpoint
    # from the phase-sliced value-calibration tool / sigma trace before any
    # noise-floor arm is trusted; it only matters when
    # rescale_noise_floor_c > 0).
    sigma_eval: float = 0.79
    # D2: variance-aware completed-Q. When True, each VISITED candidate's
    # completed-Q is shrunk toward the mixed value v_mix by its own standard
    # error before rescaling: q' = v_mix + shrink_a * (q_a - v_mix), with
    #   shrink_a = signal_var / (signal_var + variance_aware_k * SE_a^2)
    # where SE_a^2 = Var[per-visit backups] / visits (from `value_sq_sum`)
    # and `signal_var` is the between-candidate variance of the raw
    # completed-Q. A precise estimate (small SE) keeps its Q; a noisy one
    # (few visits / high per-visit variance) is pulled back to the
    # prior-weighted v_mix -- a James-Stein / empirical-Bayes shrinkage.
    # Unvisited candidates are unaffected (already v_mix). See
    # `_completed_q`'s docstring for the deviation from arXiv 2512.21648
    # (which puts the same per-action variance sigma-hat in a UCB-V
    # exploration bonus; we have no UCB selector, so we reuse the variance
    # signal to gate the completed-Q operator instead).
    variance_aware_q: bool = False
    variance_aware_k: float = 1.0
    # --- APPEND-ONLY BOUNDARY -----------------------------------------------
    # New fields go BELOW this line, never above it. This config is not
    # persisted today (constructed kwargs-only from CLI at every site), but the
    # frozen+slots dataclasses in this repo pickle POSITIONALLY, and a mid-list
    # insert is exactly the class of silent-shift bug that corrupted config
    # loading on 2026-07-05 (see docs + task #74). belief_chance_spectra (f72)
    # was originally inserted mid-list next to the other chance-spectra fields;
    # it lives here now to restore the discipline. Field ORDER does not affect
    # behavior anywhere (all construction is by keyword).
    # Hidden-information leak fix (f72), PLANNER-ONLY. When True, the search's
    # internal simulation resolves the two hidden-info chance nodes from a
    # public BELIEF instead of the true hidden state:
    #   - MOVE_ROBBER steal: for an opponent victim, uniform over all five public
    #     resource identities on the legacy fixed-five wheel (rather than the
    #     victim's true held-type set/composition); for the perspective player's
    #     own hand, use the known count-weighted distribution. Residual on newer
    #     hand-filtered wheels: the engine may expose only materializable types.
    #   - BUY_DEVELOPMENT_CARD: reweight drawable outcomes by the perspective's
    #     posterior predictive deck (full 25-card composition minus their own
    #     cards minus ALL players' PLAYED cards -- i.e. opponents' face-down cards
    #     are exchangeable with the deck). Residual: a card
    #     type held 100% face-down by opponents has no drawable engine outcome to
    #     materialize, so it cannot appear as a simulated draw.
    # This is a CHANCE-node correction, not full information-set search:
    # opponent legal actions still come from the authoritative hidden state.
    # It ONLY changes the planner's expectation backups; the live-game/env
    # transitions in `gumbel_self_play._apply_selected_action` keep sampling from
    # the TRUE hidden state (correct -- the world runs on truth). Default OFF; a
    # search-semantics change, so shipping is gated on a strength-based H2H A/B
    # (standing rule: fidelity metrics do not bind).
    belief_chance_spectra: bool = False

    # f74b: at roots wider than `wide_candidates_threshold`, denoise the leaf
    # value+prior by averaging the evaluator over all 12 D6 board orientations
    # (see catan_zero.rl.hex_symmetry). Default OFF and a pure no-op when off or
    # when the evaluator lacks `evaluate_symmetry_averaged`. Search-semantics
    # change -- only a strength H2H binds.
    symmetry_averaged_eval: bool = False

    # CAT-62: root considered-set cap, applied in `_root_candidate_count`. When
    # set, OVERRIDES both `max_root_candidates` and `max_root_candidates_wide`
    # with a single top-K (by prior+Gumbel noise, same Gumbel-Top-k ranking
    # `_run_root_search` already used) cap at EVERY root, narrow or wide.
    # This is deliberately a separate knob from those two fields rather than
    # just changing their defaults: `max_root_candidates_wide` defaults to
    # full width (54) specifically because F8 found a narrower cap left some
    # legal placements never explored regardless of Gumbel noise (see that
    # field's docstring) -- so re-capping wide roots needs to be an explicit,
    # A/B-able opt-in, not a silent change to the existing defaults. Only
    # binds when the root has more legal actions than the cap (`min(num_legal,
    # cap)`); a cap wider than the root's legal-action count is a no-op there.
    # TODO(CAT-62 follow-up, deferred): the top-k selection this feeds is
    # still plain (gumbel + logit) ranking with no symmetry awareness, so a
    # tight cap at a wide root can end up full of near-duplicate symmetric
    # candidates while excluding meaningfully different placements (the same
    # regression class F8 found). A symmetry-diverse selection pass --
    # deduplicating within `catan_zero.rl.hex_symmetry`'s D6 orbit groups
    # before taking top-K -- is the natural fix and is intentionally left for
    # a follow-up; any A/B of `root_candidate_cap` at wide roots should keep
    # this in mind. Default None: an exact no-op, current behavior unchanged.
    root_candidate_cap: int | None = None

    # CAT-62: low-evidence policy-target pruning (KataGo's "policy-target
    # pruning", the companion to forced playouts, aimed at near-tied wide
    # roots). After search, `_prune_policy_target` zeros out (not removes --
    # `SearchResult.improved_policy`'s key set is a contract covering ALL
    # legal root actions, see `gumbel_self_play._build_decision_row`)
    # probability mass on actions with fewer than `policy_target_min_visits`
    # visits -- i.e. whose completed-Q rested entirely on the unvisited
    # mixed-value completion `v_mix`, not a real backup -- then renormalizes
    # the remaining mass to sum to 1. This ONLY affects the emitted training
    # target; the actual move played by `search()` still uses the unpruned
    # `improved_policy` (this ticket is scoped to policy-target hygiene, not
    # to changing which move gets played -- see SW-0 scope note on this
    # ticket).
    # A/B CAVEAT (train_bc interaction): the pruned rows are persisted with
    # `target_policy == 0` on the zeroed actions, which `_build_decision_row`
    # turns into `target_policy_mask == False` there. train_bc's soft-target
    # coverage gate (`--soft-target-min-legal-coverage`) counts covered legal
    # actions off exactly that mask, so a large `policy_target_min_visits` can
    # drop a near-tied wide root below the coverage threshold and fall that row
    # back to one-hot hard CE for the POLICY head -- i.e. the opposite of
    # distilling the (correctly) pruned soft target. Any A/B enabling this knob
    # should check the coverage-gate telemetry / lower `--soft-target-min-legal-
    # coverage` accordingly. Default 0: an exact no-op, current behavior
    # unchanged.
    policy_target_min_visits: int = 0

    # --- CAT-61: V4 uncertainty-driven capped backup weighting (default OFF) ---
    # When True, each per-visit node backup is weighted inversely by the leaf
    # value-error head's prediction, following KataGo's uncertainty weighting:
    #     sigma = sqrt(predicted_squared_error)
    #     weight = a / (sigma ** exp + a / max_weight)
    # with a = uncertainty_backup_a, exp = uncertainty_backup_exp, and
    # max_weight = uncertainty_backup_cap. The square root is required because
    # our auxiliary head predicts squared error while KataGo's operator consumes
    # an error scale. Low-error leaves approach max_weight; uncertain leaves get
    # less influence. (The previous ``min(cap, a * err**exp)`` implementation
    # accidentally did the opposite and upweighted the least reliable leaves.)
    # The prediction is surfaced
    # by EntityGraphRustEvaluatorConfig.emit_uncertainty and carried on each
    # node's `prior_uncertainty`). A visited action's completed-Q then uses the
    # weight-weighted mean of its backups (`_GAction.weighted_q`) instead of the
    # plain visit mean. The CAP is the load-bearing R8 lesson: KataGo's
    # uncertainty-weighted playouts "required a weight cap to work", and our
    # earlier UNCAPPED D2 experiment came back neutral -- this is the recap.
    #
    # Default OFF is bit-identical: no weight is accumulated, completed-Q reads
    # the plain `stats.q`, and the tree traversal/visit budget is unchanged. With
    # a non-emitting evaluator every prediction is 0.0, so every backup receives
    # the same max weight and weighted-Q equals plain Q; the feature is therefore
    # inert unless paired with an uncertainty-emitting evaluator AND a trained
    # error head. This is a search-semantics change: shipping is gated on a
    # strength-based H2H, never blind-shipped.
    uncertainty_backup_weighting: bool = False
    uncertainty_backup_a: float = 0.25
    uncertainty_backup_exp: float = 1.0
    uncertainty_backup_cap: float = 1.0

    # --- CAT-61: D2 closed-form James-Stein shrinkage coefficient (default OFF) --
    # Only consulted when `variance_aware_q` is True. When True, the D2
    # completed-Q shrinkage stops using the per-candidate, hand-tuned-k form
    #     shrink_a = signal_var / (signal_var + variance_aware_k * SE_a^2)
    # and instead applies ONE closed-form James-Stein / empirical-Bayes
    # coefficient shared by every visited candidate:
    #     lambda* = v2 / (v2 + s2)
    # where v2 is the across-candidate (between-arm) variance of the visited Q's
    # and s2 is the mean within-candidate sampling variance (mean of SE_a^2 over
    # visited candidates). This is the shrinkage that minimizes total risk under
    # the normal-means model with NO free k to tune (R8: "the James-Stein
    # shrinkage coefficient has a closed form ... no hand-tuning"). Default OFF
    # keeps the existing per-candidate k-tuned shrinkage exactly.
    variance_aware_closed_form_js: bool = False

    # CAT-25/B6 decoupled wide-root gates. APPENDED here (never inserted near
    # their older related fields) because this frozen+slots dataclass pickles
    # positionally; see the append-only boundary above.
    # Inclusive minimum legal-action count for D6 averaging. None preserves
    # the legacy `len(legal) > wide_candidates_threshold` gate exactly.
    symmetry_averaged_eval_threshold: int | None = None
    # Inclusive minimum legal-action count for n_full_wide. None preserves the
    # legacy `len(legal) > wide_candidates_threshold` budget gate exactly.
    n_full_wide_threshold: int | None = None
    # When enabled, every root selected by the n_full_wide gate uses FULL
    # search independently of p_full. Explicit search(force_full=...) wins.
    wide_roots_always_full: bool = False

    # Public-conservation PIMC boundary. When enabled, search NEVER expands the
    # authoritative game. The Rust engine samples rules-valid worlds from bank/
    # deck conservation plus public hand sizes (not a history-conditioned
    # Bayesian posterior), search stops when control leaves the root actor's
    # turn, and root evidence is aggregated. `n_full`/`n_fast` remain TOTAL
    # nominal budgets divided deterministically across particles.
    information_set_search: bool = False
    determinization_particles: int = 1
    determinization_min_simulations: int = 32

    # Optional budget-invariant sigma calibration. The legacy Gumbel operator
    # scales completed-Q logits by ``(c_visit + max_child_visits) * c_scale``.
    # Consequently, increasing only the simulation budget sharpens both
    # Sequential-Halving re-ranking and the emitted policy target. When set,
    # use this fixed reference visit count in place of the realized maximum.
    # None is the exact historical behavior.
    sigma_reference_visits: int | None = None

    # How public-belief particles are combined into the emitted POLICY target.
    # ``mean_improved_policy`` is the historical operator:
    # E_world[softmax(log(prior_world) + sigma(minmax(completed_q_world)))].
    # ``aggregate_q_then_improve`` is an experimental target-only operator:
    # uniformly aggregate action-aligned completed-Q evidence first, then apply
    # one min-max/sigma improvement.  Gameplay selection is
    # deliberately left on the historical mean policy so this arm isolates the
    # learner target.  The experimental mode fails closed unless every particle
    # attests root-actor Q perspective and exposes completed-Q for every legal
    # action.
    information_set_target_aggregation: str = "mean_improved_policy"
    # Which public-belief policy SELECTS the live gameplay action.  This is
    # intentionally separate from the learner-target field above: changing a
    # target-only experiment must never silently change playing strength.
    # The legacy default selects from the mean of per-world improved policies.
    # The opt-in corrected mode selects from one improvement applied after
    # uniformly aggregating completed-Q evidence across hidden worlds.
    gameplay_policy_aggregation: str = "mean_improved_policy"


@dataclass(frozen=True, slots=True)
class SearchResult:
    selected_action: int
    improved_policy: dict[int, float]
    visit_counts: dict[int, int]
    q_values: dict[int, float]
    priors: dict[int, float]
    root_value: float
    used_full_search: bool
    simulations_used: int
    afterstate_values: dict[int, float] = field(default_factory=dict)
    # Evidence required to construct a belief-level completed-Q target.  These
    # append-only defaults preserve every existing SearchResult constructor.
    # Every root action is present: unvisited actions already carry mctx's
    # per-world v_mix completion.  Aggregating this evidence uniformly avoids
    # conditioning an action's value on only the worlds where SH visited it.
    completed_q_values: dict[int, float] = field(default_factory=dict)
    # q_values/completed_q_values are only safe to combine across
    # determinizations when both are expressed from the root actor's
    # perspective. Real Python/native searches attest that shared contract;
    # synthetic/old-wheel results default false and experimental aggregation
    # refuses them.
    q_values_root_perspective: bool = False


@dataclass(slots=True)
class _GAction:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    # Running sum of squared per-visit backup values, for the variance-aware
    # completed-Q shrinkage (config.variance_aware_q). Always accumulated
    # alongside `value_sum` (a single float multiply-add per backup) so the
    # per-candidate empirical variance is available without a second pass;
    # it is a pure no-op when `variance_aware_q` is False (nothing reads it).
    value_sq_sum: float = 0.0
    # CAT-61 uncertainty backup weighting accumulators. Only written when
    # `config.uncertainty_backup_weighting` is True (a pure no-op otherwise:
    # both stay 0.0 and nothing reads them, so `q`/completed-Q are unchanged).
    # `weighted_value_sum` = sum of weight*value; `weight_sum` = sum of weights.
    weighted_value_sum: float = 0.0
    weight_sum: float = 0.0
    children: dict[int, "_GNode"] = field(default_factory=dict)
    probabilities: dict[int, float] = field(default_factory=dict)
    afterstate_value: float | None = None

    @property
    def q(self) -> float:
        if self.visits <= 0:
            return 0.0
        return self.value_sum / float(self.visits)

    @property
    def weighted_q(self) -> float:
        """Backup-weight-weighted mean of this action's per-visit backups
        (CAT-61). Falls back to the plain visit mean `q` when no weighted
        backups were recorded (weight_sum <= 0), so it is always safe to read
        -- on the default (unweighted) path it returns exactly `q`."""
        if self.weight_sum <= 0.0:
            return self.q
        return self.weighted_value_sum / self.weight_sum

    @property
    def q_variance(self) -> float:
        """Population variance of this action's per-visit backup values,
        Var[v] = E[v^2] - E[v]^2, clamped at 0 for float round-off. Zero
        when fewer than 2 visits (no spread is estimable)."""
        if self.visits < 2:
            return 0.0
        mean = self.value_sum / float(self.visits)
        mean_sq = self.value_sq_sum / float(self.visits)
        return max(0.0, mean_sq - mean * mean)


@dataclass(slots=True)
class _GNode:
    game: Any
    root_color: str
    prior_value: float = 0.0
    # CAT-61: leaf value-error head prediction for this node, surfaced by an
    # uncertainty-emitting evaluator (0.0 when the evaluator does not emit it).
    # Read only by the capped backup weighting (config.uncertainty_backup_weighting).
    prior_uncertainty: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    actions: dict[int, _GAction] = field(default_factory=dict)
    action_json: dict[int, Any] = field(default_factory=dict)
    action_logits: dict[int, float] = field(default_factory=dict)
    # Per-action chance spectrum, cached from `decision_context_json` at
    # expansion time (batch-API path only) so ROLL traversal doesn't need a
    # second `spectrum_json` round-trip. Empty/missing when the legacy path
    # was used or an action has no chance component.
    action_spectrum: dict[int, tuple[tuple[int, float], ...]] = field(default_factory=dict)
    expanded: bool = False

    @property
    def value(self) -> float:
        if self.visits <= 0:
            return self.prior_value
        return self.value_sum / float(self.visits)


@dataclass(slots=True)
class _PendingSimulation:
    """A selected leaf whose neural evaluation has not run yet.

    ``finish`` expands the leaf and performs every deferred backup on its path.
    Selection and game mutation stay serial and deterministic; only evaluator
    calls from independent root candidates are coalesced.
    """

    node: _GNode
    legal_actions: tuple[int, ...]
    finish: Callable[[Any], float]


def _split_evaluation(result: Any) -> tuple[Any, float, float]:
    """Unpack an evaluator result into (priors, value, uncertainty).

    Evaluators may return either a 2-tuple `(priors, value)` (every evaluator
    today, and the default of the neural one) or a 3-tuple
    `(priors, value, uncertainty)` when opted into
    EntityGraphRustEvaluatorConfig.emit_uncertainty. A 2-tuple yields
    uncertainty 0.0, so with backup weighting off the search is bit-identical;
    the third element only becomes load-bearing under
    `config.uncertainty_backup_weighting` (CAT-61)."""
    priors = result[0]
    value = result[1]
    uncertainty = float(result[2]) if len(result) > 2 else 0.0
    return priors, value, uncertainty


def _evaluate_many_checked(
    evaluator: Any,
    requests: list[tuple[Any, tuple[int, ...]]],
    *,
    root_color: str,
    colors: tuple[str, ...],
) -> list[Any]:
    """Materialize and cardinality-check one evaluator batch atomically.

    Callers must validate the complete response before expanding any child or
    applying any backup. Otherwise a short iterator silently leaves children
    at fabricated default values while a long iterator silently drops results.
    """
    results = list(
        evaluator.evaluate_many(
            requests,
            root_color=root_color,
            colors=colors,
        )
    )
    if len(results) != len(requests):
        raise RuntimeError(
            "evaluate_many returned "
            f"{len(results)} results for {len(requests)} requests"
        )
    return results


class GumbelChanceMCTS:
    def __init__(
        self,
        config: GumbelChanceMCTSConfig | None = None,
        evaluator: RustEvaluator | None = None,
    ) -> None:
        self.config = config or GumbelChanceMCTSConfig()
        if (
            self.config.sigma_reference_visits is not None
            and int(self.config.sigma_reference_visits) < 0
        ):
            raise ValueError("sigma_reference_visits must be non-negative")
        if self.config.information_set_target_aggregation not in {
            "mean_improved_policy",
            "aggregate_q_then_improve",
        }:
            raise ValueError(
                "information_set_target_aggregation must be "
                "'mean_improved_policy' or 'aggregate_q_then_improve'"
            )
        if self.config.gameplay_policy_aggregation not in {
            "mean_improved_policy",
            "aggregate_q_then_improve",
        }:
            raise ValueError(
                "gameplay_policy_aggregation must be "
                "'mean_improved_policy' or 'aggregate_q_then_improve'"
            )
        if (
            self.config.information_set_target_aggregation
            != "mean_improved_policy"
            and not bool(self.config.information_set_search)
        ):
            raise ValueError(
                "aggregate_q_then_improve requires information_set_search=True"
            )
        if (
            self.config.gameplay_policy_aggregation
            != "mean_improved_policy"
            and not bool(self.config.information_set_search)
        ):
            raise ValueError(
                "aggregate_q_then_improve gameplay requires "
                "information_set_search=True"
            )
        if (
            self.config.information_set_target_aggregation
            == "aggregate_q_then_improve"
            and self.config.sigma_reference_visits is None
        ):
            raise ValueError(
                "aggregate_q_then_improve requires sigma_reference_visits so "
                "particle-count/budget changes cannot silently sharpen targets"
            )
        if (
            self.config.gameplay_policy_aggregation
            == "aggregate_q_then_improve"
            and self.config.sigma_reference_visits is None
        ):
            raise ValueError(
                "aggregate_q_then_improve gameplay requires "
                "sigma_reference_visits so particle-count/budget changes cannot "
                "silently sharpen action selection"
            )
        if bool(self.config.information_set_search) and bool(
            self.config.belief_chance_spectra
        ):
            raise ValueError(
                "information_set_search cannot be combined with belief_chance_spectra"
            )
        self.evaluator = evaluator or HeuristicRustEvaluator()
        self.rng = random.Random(self.config.seed)
        _require_rust_module()

    def new_game(self, *, seed: int | None = None) -> Any:
        catanatron_rs = _require_rust_module()
        return catanatron_rs.Game.simple(list(self.config.colors), seed=seed)

    # ------------------------------------------------------------------
    # Public search entry point.
    # ------------------------------------------------------------------
    def search(self, game: Any, *, force_full: bool | None = None) -> SearchResult:
        if bool(getattr(self.config, "information_set_search", False)):
            return self._search_information_set(game, force_full=force_full)
        return self._search_authoritative(game, force_full=force_full)

    def _search_authoritative(
        self, game: Any, *, force_full: bool | None = None
    ) -> SearchResult:
        return self._search_single_world(game, force_full=force_full)

    def _search_information_set(
        self, game: Any, *, force_full: bool | None = None
    ) -> SearchResult:
        """Actor-turn PIMC over public-conservation determinizations.

        The authoritative game is used only to read the root actor and that
        actor's legal actions.  It is never evaluated or expanded.  Each search
        particle comes from the engine's atomic ``determinize_for_player``
        primitive, and traversal stops as soon as control leaves the root
        actor's turn.  Root policies/Q/value are then averaged across particles.
        This closes both hidden chance-support leakage and opponent legal-action
        leakage for the sampled horizon while leaving live action resolution on
        authoritative truth. Samples are not conditioned on full public-history
        deductions; provenance deliberately calls this conservation PIMC.
        """
        if not hasattr(game, "determinize_for_player"):
            raise RuntimeError(
                "information_set_search requires a catanatron_rs wheel exposing "
                "Game.determinize_for_player"
            )
        evaluator_config = getattr(self.evaluator, "config", None)
        if evaluator_config is None or not bool(
            getattr(evaluator_config, "public_observation", False)
        ):
            raise RuntimeError(
                "information_set_search requires evaluator public_observation=True"
            )
        if bool(self.config.play_sh_winner):
            raise RuntimeError(
                "play_sh_winner is undefined across information-set particles; "
                "select from the aggregated improved policy"
            )

        root_color = str(game.current_color())
        authoritative_legal, _actions, _spectra = self._fetch_legal_actions(game)
        if not authoritative_legal:
            raise RuntimeError("no legal actions at information-set MCTS root")

        requested_particles = int(self.config.determinization_particles)
        if requested_particles < 1:
            raise ValueError("determinization_particles must be >= 1")
        min_per_particle = int(self.config.determinization_min_simulations)
        if min_per_particle < 1:
            raise ValueError("determinization_min_simulations must be >= 1")

        # Choose PCR/full-vs-fast ONCE for the information set, then divide the
        # exact total budget across particles.  Forced roots do not spend a
        # Sequential-Halving budget but still aggregate their chance/value result.
        if len(authoritative_legal) > 1:
            wide_budget_root = _wide_budget_applies(
                len(authoritative_legal), self.config
            )
            use_full = _choose_full_search(
                self.config,
                force_full=force_full,
                wide_budget_root=wide_budget_root,
                random_draw=self.rng.random,
            )
            n_full_effective = int(self.config.n_full)
            if wide_budget_root:
                n_full_effective = int(self.config.n_full_wide)
            total_budget = max(
                int(n_full_effective if use_full else self.config.n_fast), 1
            )
            # Spend belief averaging where it improves distilled policy, but do
            # not fragment n16 fast searches into four nearly-empty trees.  With
            # the A1 settings this realizes p4 for n128 full rows and p1 for n16
            # fast rows.  Forced rows below are also p1 because they carry no
            # policy target.
            budgets = list(
                information_set_particle_budgets(
                    total_budget,
                    requested_particles,
                    min_per_particle,
                )
            )
            particle_count = len(budgets)
        else:
            use_full = bool(force_full) if force_full is not None else False
            particle_count = 1
            budgets = [None]

        # Pre-draw every determinization seed before any particle search.  This
        # makes the particle set independent of how many RNG draws a particular
        # sampled tree consumes.
        particle_seeds = [self.rng.getrandbits(64) for _ in range(particle_count)]
        results: list[SearchResult] = []
        for particle_index, particle_seed in enumerate(particle_seeds):
            sampled = game.determinize_for_player(root_color, int(particle_seed))
            sampled_legal, _sampled_actions, _sampled_spectra = self._fetch_legal_actions(
                sampled
            )
            if tuple(sampled_legal) != tuple(authoritative_legal):
                raise RuntimeError(
                    "public-belief determinization changed root legal actions: "
                    f"authoritative={tuple(authoritative_legal)} sampled={tuple(sampled_legal)}"
                )
            self._information_set_root_turn = int(sampled.num_turns())
            results.append(
                self._search_single_world(
                    sampled,
                    force_full=use_full,
                    n_simulations_override=budgets[particle_index],
                )
            )
        return self._aggregate_information_set_results(
            results,
            legal_actions=tuple(authoritative_legal),
            used_full_search=use_full,
        )

    def _aggregate_information_set_results(
        self,
        results: list[SearchResult],
        *,
        legal_actions: tuple[int, ...],
        used_full_search: bool,
    ) -> SearchResult:
        if not results:
            raise RuntimeError("information-set search produced no particles")
        count = float(len(results))
        priors = _normalize_policy(
            {
                action: sum(result.priors.get(action, 0.0) for result in results)
                / count
                for action in legal_actions
            }
        )
        improved = _normalize_policy(
            {
                action: sum(
                    result.improved_policy.get(action, 0.0) for result in results
                )
                / count
                for action in legal_actions
            }
        )
        visit_counts = {
            action: sum(result.visit_counts.get(action, 0) for result in results)
            for action in legal_actions
        }
        q_values: dict[int, float] = {}
        for action in legal_actions:
            weighted = [
                (result.q_values[action], result.visit_counts.get(action, 0))
                for result in results
                if action in result.q_values and result.visit_counts.get(action, 0) > 0
            ]
            total_visits = sum(visits for _value, visits in weighted)
            if total_visits > 0:
                q_values[action] = sum(
                    value * visits for value, visits in weighted
                ) / float(total_visits)
        afterstate_values = {
            action: sum(values) / len(values)
            for action in legal_actions
            if (
                values := [
                    result.afterstate_values[action]
                    for result in results
                    if action in result.afterstate_values
                ]
            )
        }

        belief_improved = improved
        if (
            len(legal_actions) > 1
            and (
                self.config.information_set_target_aggregation
                == "aggregate_q_then_improve"
                or self.config.gameplay_policy_aggregation
                == "aggregate_q_then_improve"
            )
        ):
            belief_improved = self._belief_level_improved_policy(
                results,
                legal_actions=legal_actions,
                aggregate_priors=priors,
            )
        gameplay_policy = (
            belief_improved
            if self.config.gameplay_policy_aggregation
            == "aggregate_q_then_improve"
            else improved
        )
        if float(self.config.temperature) > 0.0:
            selected = self._sample_categorical(gameplay_policy)
        else:
            selected = max(
                legal_actions,
                key=lambda action: (
                    gameplay_policy.get(action, 0.0),
                    visit_counts.get(action, 0),
                    priors.get(action, 0.0),
                    -int(action),
                ),
            )
        target_policy = improved
        if (
            self.config.information_set_target_aggregation
            == "aggregate_q_then_improve"
            and len(legal_actions) > 1
        ):
            target_policy = belief_improved
        training_policy = _prune_policy_target(
            target_policy,
            visit_counts,
            min_visits=int(self.config.policy_target_min_visits),
        )
        return SearchResult(
            selected_action=int(selected),
            improved_policy=training_policy,
            visit_counts=visit_counts,
            q_values=q_values,
            priors=priors,
            root_value=sum(result.root_value for result in results) / count,
            used_full_search=bool(used_full_search),
            simulations_used=sum(result.simulations_used for result in results),
            afterstate_values=afterstate_values,
            completed_q_values={
                action: sum(
                    float(result.completed_q_values[action])
                    for result in results
                )
                / count
                for action in legal_actions
                if all(action in result.completed_q_values for result in results)
            },
            q_values_root_perspective=all(
                result.q_values_root_perspective for result in results
            ),
        )

    def _belief_level_improved_policy(
        self,
        results: list[SearchResult],
        *,
        legal_actions: tuple[int, ...],
        aggregate_priors: dict[int, float],
    ) -> dict[int, float]:
        """Improve once after uniform hidden-world completed-Q aggregation.

        Each particle first applies the existing mctx completion in its own
        hidden world: visited actions retain root-actor Q and unvisited actions
        receive that world's prior-weighted ``v_mix``.  Uniformly averaging the
        full action vectors avoids conditioning an action's value on only worlds
        where Sequential Halving happened to visit it.  One minmax/noise-floor/
        fixed-sigma transform is applied after averaging.  Mean per-particle
        visits are carried only for the optional D1 noise-floor calculation.
        """
        if not results:
            raise RuntimeError("belief target aggregation requires particles")

        legal = set(legal_actions)
        particle_count = len(results)
        for index, result in enumerate(results):
            if not result.q_values_root_perspective:
                raise RuntimeError(
                    f"particle {index} does not attest root-actor Q perspective"
                )
            if set(result.priors) != legal:
                raise RuntimeError(
                    f"particle {index} prior support does not match root actions"
                )
            if not set(result.q_values).issubset(legal):
                raise RuntimeError(f"particle {index} Q support is not root-aligned")
            if set(result.completed_q_values) != legal:
                raise RuntimeError(
                    f"particle {index} completed-Q support does not match root actions"
                )
            if not all(
                math.isfinite(float(value))
                for value in result.completed_q_values.values()
            ):
                raise RuntimeError(
                    f"particle {index} contains non-finite completed-Q evidence"
                )
            for action in legal_actions:
                visits = int(result.visit_counts.get(action, 0))
                has_q = action in result.q_values
                if has_q != (visits > 0):
                    raise RuntimeError(
                        f"particle {index} action {action} has incompatible "
                        "Q/visit coverage"
                    )
                if has_q and not math.isfinite(float(result.q_values[action])):
                    raise RuntimeError(
                        f"particle {index} action {action} has non-finite Q"
                    )

        completed_q = {
            action: sum(
                float(result.completed_q_values[action]) for result in results
            )
            / float(particle_count)
            for action in legal_actions
        }
        # Reuse the exact existing minmax/noise-floor/sigma implementation on a
        # synthetic belief root. D1 needs the *fractional* mean number of visits
        # per action and particle. Rounding each action's particle mean first is
        # not equivalent: at sparse wide roots every positive mean can be below
        # 0.5, which would make the synthetic root look entirely unvisited and
        # incorrectly collapse all belief Q evidence to the neutral policy.
        mean_visits = sum(
            int(result.visit_counts.get(action, 0))
            for result in results
            for action in legal_actions
        ) / float(particle_count * len(legal_actions))
        belief_root = _GNode(
            game=None,
            root_color="__belief_root__",
            prior_value=0.0,
            actions={
                action: _GAction(
                    prior=float(aggregate_priors[action]),
                )
                for action in legal_actions
            },
            action_logits={
                action: math.log(max(float(aggregate_priors[action]), 1.0e-8))
                / max(float(self.config.prior_temperature), 1.0e-6)
                for action in legal_actions
            },
            expanded=True,
        )
        return self._improved_policy(
            belief_root,
            completed_q,
            mean_visits_override=mean_visits,
        )

    def _search_single_world(
        self,
        game: Any,
        *,
        force_full: bool | None = None,
        n_simulations_override: int | None = None,
    ) -> SearchResult:
        root_color = str(game.current_color())
        legal_actions, action_json_by_id, spectrum_by_id = self._fetch_legal_actions(game)
        if not legal_actions:
            raise RuntimeError("no legal actions at MCTS root")
        if len(legal_actions) == 1:
            return self._forced_single_action_result(
                game,
                legal_actions,
                root_color=root_color,
                action_json_by_id=action_json_by_id,
                spectrum_by_id=spectrum_by_id,
            )

        # ARM (phase-gated search): at roots wider than raw_policy_above_width,
        # skip search entirely and defer to the raw prior (argmax). Placed before
        # any simulation so it costs a single expansion (the prior forward pass)
        # and nothing more. Disabled (None) => fall through to normal search.
        raw_above = self.config.raw_policy_above_width
        if raw_above is not None and len(legal_actions) > int(raw_above):
            return self._raw_policy_root_result(game, root_color)

        wide_budget_root = _wide_budget_applies(len(legal_actions), self.config)
        use_full = _choose_full_search(
            self.config,
            force_full=force_full,
            wide_budget_root=wide_budget_root,
            random_draw=self.rng.random,
        )
        # ARM (placement budget asymmetry): a full search at a wide root spends
        # n_full_wide sims instead of n_full. Disabled (None) => n_full everywhere.
        n_full_effective = int(self.config.n_full)
        if wide_budget_root:
            n_full_effective = int(self.config.n_full_wide)
        n_simulations = (
            max(int(n_simulations_override), 1)
            if n_simulations_override is not None
            else max(int(n_full_effective if use_full else self.config.n_fast), 1)
        )

        root = _GNode(game=game.copy(), root_color=root_color)
        self._expand(root, at_root=True)
        priors = {action_id: stats.prior for action_id, stats in root.actions.items()}

        sh_winner_action, used = self._run_root_search(
            root,
            n_simulations,
            # Information-set search divides one locked TOTAL budget across
            # particles. Legacy phase rounding inside each small particle tree
            # can otherwise turn n128/p4 into 4*108=432 simulations at a
            # 54-action opening. An explicit particle override is therefore an
            # exact sub-budget; ordinary non-PIMC searches retain the configured
            # legacy/exact operator choice.
            exact_budget_override=n_simulations_override is not None,
        )

        completed_q = self._completed_q(root)
        improved_policy = self._improved_policy(root, completed_q)
        visit_counts = {action_id: stats.visits for action_id, stats in root.actions.items()}
        q_values = {
            action_id: stats.q for action_id, stats in root.actions.items() if stats.visits > 0
        }
        afterstate_values = {
            action_id: stats.afterstate_value
            for action_id, stats in root.actions.items()
            if stats.afterstate_value is not None
        }

        # NOTE: this deliberately deviates from the paper's reference
        # algorithm, which plays the raw Sequential Halving winner
        # (`sh_winner_action`, including its Gumbel noise). We instead pick
        # from `improved_policy` (argmax at T=0, sample at T>0). This is
        # intentional: the policy-improvement guarantee we rely on is that
        # `improved_policy` -- the training TARGET -- is an improvement over
        # the raw prior at any sim budget; it says nothing about which single
        # action the SH schedule happened to spend the last round's budget on.
        # Set `config.play_sh_winner=True` to play the SH winner instead (A/B).
        if bool(self.config.play_sh_winner):
            selected = int(sh_winner_action)
        elif float(self.config.temperature) > 0.0:
            selected = self._sample_categorical(improved_policy)
        else:
            selected = max(
                improved_policy,
                key=lambda action_id: (
                    improved_policy[action_id],
                    visit_counts.get(action_id, 0),
                    root.actions[action_id].prior,
                ),
            )

        # CAT-62: the TRAINING TARGET is pruned (when policy_target_min_visits > 0),
        # not `improved_policy` itself -- `selected` above already picked its move
        # from the unpruned distribution, so this ticket's scope (policy-target
        # hygiene) cannot change which move gets played.
        training_policy = _prune_policy_target(
            improved_policy, visit_counts, min_visits=int(self.config.policy_target_min_visits)
        )

        return SearchResult(
            selected_action=int(selected),
            improved_policy=training_policy,
            visit_counts=visit_counts,
            q_values=q_values,
            priors=priors,
            root_value=root.value,
            used_full_search=use_full,
            simulations_used=used,
            afterstate_values=afterstate_values,
            completed_q_values={
                int(action): float(value) for action, value in completed_q.items()
            },
            q_values_root_perspective=True,
        )

    def _raw_policy_root_result(self, game: Any, root_color: str) -> SearchResult:
        """Phase-gated-search arm: play argmax(prior) at a too-wide root with no
        tree search. Expands the root once (the prior forward pass) and selects the
        highest-prior action, ties broken toward the lower action id to match
        raw_selfplay.py's raw-argmax convention. used_full_search=False so the
        self-play row builder emits policy_weight=0 -- a raw argmax is not a
        policy-improvement target worth imitating. improved_policy is reported as
        the (unimproved) prior itself."""
        root = _GNode(game=game.copy(), root_color=root_color)
        self._expand(root, at_root=True)
        priors = {action_id: stats.prior for action_id, stats in root.actions.items()}
        if not priors:
            raise RuntimeError("no legal actions at raw-policy MCTS root")
        selected = max(
            priors,
            key=lambda action_id: (float(priors[action_id]), -int(action_id)),
        )
        return SearchResult(
            selected_action=int(selected),
            improved_policy=dict(priors),
            visit_counts={},
            q_values={},
            priors=priors,
            root_value=root.value,
            used_full_search=False,
            simulations_used=0,
            completed_q_values={
                action: float(root.prior_value) for action in priors
            },
            q_values_root_perspective=True,
        )

    def _forced_single_action_result(
        self,
        game: Any,
        legal_actions: tuple[int, ...],
        *,
        root_color: str,
        action_json_by_id: dict[int, Any] | None = None,
        spectrum_by_id: dict[int, tuple[tuple[int, float], ...]] | None = None,
    ) -> SearchResult:
        """Handle the (very common, e.g. every real ROLL) single-legal-action case.

        ROLL is almost always the sole legal action at its own decision point, so
        without this path every real dice roll would skip chance enumeration
        entirely and report a fake root_value/afterstate_value. Still enumerate
        the 11 outcomes here (no Gumbel/Sequential Halving needed -- there is
        only one candidate) so downstream training signal stays real.
        """
        action = int(legal_actions[0])
        if action_json_by_id is None:
            action_json_by_id = _playable_action_json_by_index(
                game, legal_actions, self.config.colors, self.config.map_kind
            )
        action_json = action_json_by_id[action]
        if _action_type(action_json) != "ROLL":
            # A single legal action (e.g. a forced discard) is not necessarily
            # terminal -- call the evaluator once for a real value instead of
            # reporting a fabricated 0.0, which would poison any downstream
            # value/reanalyze target (the exact "fabricated label" bug class
            # that broke the earlier plain-PUCT MCTS run).
            _priors, value, _uncertainty = _split_evaluation(
                self.evaluator.evaluate(
                    game, legal_actions, root_color=root_color, colors=self.config.colors
                )
            )
            return SearchResult(
                selected_action=action,
                improved_policy={action: 1.0},
                visit_counts={action: 1},
                q_values={},
                priors={action: 1.0},
                root_value=float(max(min(value, 1.0), -1.0)),
                used_full_search=True,
                simulations_used=0,
                completed_q_values={
                    action: float(max(min(value, 1.0), -1.0))
                },
                q_values_root_perspective=True,
            )

        cached_spectrum = spectrum_by_id.get(action) if spectrum_by_id else None
        children, probabilities, afterstate_value = self._enumerate_roll_outcomes(
            game, action_json, root_color=root_color, cached_spectrum=cached_spectrum
        )
        total_probability = sum(probabilities.values())
        root_value = (
            sum(probabilities[index] * children[index].value for index in children)
            / total_probability
            if total_probability > 0.0
            else afterstate_value
        )
        return SearchResult(
            selected_action=action,
            improved_policy={action: 1.0},
            visit_counts={action: 1},
            q_values={},
            priors={action: 1.0},
            root_value=root_value,
            used_full_search=True,
            simulations_used=0,
            afterstate_values={action: afterstate_value},
            completed_q_values={action: float(root_value)},
            q_values_root_perspective=True,
        )

    # ------------------------------------------------------------------
    # Root: Gumbel-Top-k + Sequential Halving.
    # ------------------------------------------------------------------
    def _run_root_search(
        self,
        root: _GNode,
        n_simulations: int,
        *,
        exact_budget_override: bool = False,
    ) -> tuple[int, int]:
        legal = tuple(root.actions.keys())
        num_legal = len(legal)
        m = _root_candidate_count(num_legal, self.config)

        gumbel = {action_id: self._sample_gumbel() for action_id in legal}
        logits = root.action_logits
        top_k = sorted(
            legal, key=lambda action_id: gumbel[action_id] + logits.get(action_id, 0.0), reverse=True
        )[:m]
        remaining = list(top_k)
        candidate_rngs: dict[int, random.Random] | None = None
        if bool(self.config.root_wave_batching):
            # Draw the per-candidate seeds in the stable top-k order. Separate
            # streams make the result reproducible even though root-wave
            # scheduling interleaves candidates instead of exhausting one
            # candidate before starting the next.
            candidate_rngs = {
                action_id: random.Random(self.rng.getrandbits(64)) for action_id in top_k
            }

        if exact_budget_override or (
            bool(self.config.exact_budget_sh)
            and n_simulations >= int(self.config.exact_budget_sh_min_n)
        ):
            # Task #61: exact-budget phases (see `exact_budget_sh_phases`).
            # Each phase visits the top-`count` of the CURRENT ranking, then
            # re-ranks by gumbel + logits + sigma(completed_q) -- the same
            # scoring the legacy loop uses; only the visit counts differ
            # (total == n_simulations exactly, including a possibly-partial
            # final pass over the best-ranked prefix).
            used = 0
            for count, budget in exact_budget_sh_phases(m, n_simulations):
                visit = remaining[:count]
                if candidate_rngs is None:
                    for action_id in visit:
                        for _ in range(budget):
                            self._simulate(root, depth=0, forced_action=action_id)
                            used += 1
                else:
                    for _ in range(budget):
                        used += self._run_root_wave(root, visit, candidate_rngs)
                completed_q = self._completed_q(root)
                rescaled_q = self._rescaled_completed_q(root, completed_q)
                scale = self._sigma_scale(root)
                remaining = sorted(
                    visit,
                    key=lambda action_id: (
                        gumbel[action_id]
                        + logits.get(action_id, 0.0)
                        + scale * rescaled_q.get(action_id, 0.0)
                    ),
                    reverse=True,
                )
            final_action = remaining[0] if remaining else top_k[0]
            return int(final_action), used

        schedule = sequential_halving_schedule(m, n_simulations)

        used = 0
        for count, budget in schedule:
            if candidate_rngs is None:
                for action_id in remaining:
                    for _ in range(budget):
                        self._simulate(root, depth=0, forced_action=action_id)
                        used += 1
            else:
                for _ in range(budget):
                    used += self._run_root_wave(root, remaining, candidate_rngs)
            completed_q = self._completed_q(root)
            rescaled_q = self._rescaled_completed_q(root, completed_q)
            scale = self._sigma_scale(root)
            remaining = sorted(
                remaining,
                key=lambda action_id: (
                    gumbel[action_id]
                    + logits.get(action_id, 0.0)
                    + scale * rescaled_q.get(action_id, 0.0)
                ),
                reverse=True,
            )
            keep = max(1, count // 2)
            remaining = remaining[:keep]

        final_action = remaining[0] if remaining else top_k[0]
        return int(final_action), used

    def _run_root_wave(
        self,
        root: _GNode,
        action_ids: list[int],
        candidate_rngs: dict[int, random.Random],
    ) -> int:
        """Run one visit for every independent root candidate.

        Selection is performed in candidate order with one persistent RNG per
        candidate. Ready leaves are evaluated together, then completed in the
        same order. Exact chance nodes retain their existing enumeration and
        expectation-backup behavior; if encountered during selection, that
        candidate completes synchronously while other ready leaves still batch.
        """
        pending: list[_PendingSimulation] = []
        outer_rng = self.rng
        try:
            for action_id in action_ids:
                self.rng = candidate_rngs[action_id]
                selected = self._prepare_simulation(
                    root, depth=0, forced_action=action_id
                )
                if isinstance(selected, _PendingSimulation):
                    pending.append(selected)
        finally:
            self.rng = outer_rng

        if pending:
            requests = [(item.node.game, item.legal_actions) for item in pending]
            if len(pending) > 1 and hasattr(self.evaluator, "evaluate_many"):
                results = _evaluate_many_checked(
                    self.evaluator,
                    requests,
                    root_color=root.root_color,
                    colors=self.config.colors,
                )
            else:
                results = [
                    self.evaluator.evaluate(
                        game,
                        legal_actions,
                        root_color=root.root_color,
                        colors=self.config.colors,
                    )
                    for game, legal_actions in requests
                ]
            for item, result in zip(pending, results):
                item.finish(result)
        return len(action_ids)

    def _sample_gumbel(self) -> float:
        uniform = min(max(self.rng.random(), 1.0e-12), 1.0 - 1.0e-12)
        return -math.log(-math.log(uniform))

    # ------------------------------------------------------------------
    # Completed-Q / improved policy (shared by root output and non-root rule).
    # ------------------------------------------------------------------
    def _sigma_scale(self, node: _GNode) -> float:
        max_visits = self.config.sigma_reference_visits
        if max_visits is None:
            max_visits = max((stats.visits for stats in node.actions.values()), default=0)
        return (float(self.config.c_visit) + float(max_visits)) * float(self.config.c_scale)

    def _completed_q(self, node: _GNode) -> dict[int, float]:
        """Completed Q-values: visited actions keep their real Q; unvisited
        actions are "completed" with the mixed value estimate v_mix (mctx's
        `_compute_mixed_value`) instead of 0.0 or the raw node value alone.

        F1c: v_mix's inner average over visited actions is PRIOR-weighted
        (`stats.prior`), not visit-weighted. Visit-weighting let an action
        Gumbel-Top-k happened to visit many times (not necessarily the
        highest-prior one) dominate v_mix, systematically over-completing
        unvisited actions toward whatever got sampled early -- prior
        weighting is what mctx actually does and is invariant to the
        (somewhat arbitrary, budget-dependent) visit distribution. The OUTER
        mix against the raw node value still uses total visit count as its
        weight (unchanged from mctx and from this function's prior
        behavior) -- only the inner per-visited-action weighting changes.
        Returns RAW (unrescaled) completed Q; see `_rescale_completed_q` for
        the min-max-to-[0,1] step mctx applies before the sigma transform.

        D2 (`config.variance_aware_q`, default OFF -> exact behavior above):
        each VISITED candidate's Q is additionally shrunk toward v_mix in
        proportion to how precisely it is estimated:
            q'_a = v_mix + shrink_a * (sign * q_a - v_mix)
            shrink_a = signal_var / (signal_var + k * SE_a^2)
        where SE_a^2 = Var[a's per-visit backups] / visits_a (from
        `value_sq_sum`) is the sampling variance of the Q estimate,
        `signal_var` is the between-candidate variance of the raw visited
        Q's (a plug-in for the true value spread we are trying to detect),
        and k = `config.variance_aware_k`. A precisely estimated candidate
        (SE_a -> 0) keeps its Q (shrink -> 1); a noisy one (few visits / high
        per-visit variance) collapses to the prior-weighted v_mix
        (shrink -> 0). When there is no discernible between-candidate signal
        (signal_var <= 0) nothing is shrunk. This is a James-Stein /
        empirical-Bayes shrinkage of the completed-Q operator.

        DEVIATION FROM arXiv 2512.21648 ("Variance-Aware Prior-Based Tree
        Policies for MCTS", Inverse-RPO / UCT-V-P): that work injects the
        same per-action empirical standard deviation sigma-hat into a UCB-V
        *exploration bonus* (S_a = q_a + c1*sigma-hat*sqrt(prior*logN/(1+n))
        + c2*prior*logN/(1+n)) for a UCB-style selector, and stores (n, mu,
        sigma^2) per node via Welford's update. Our search is Gumbel-Top-k +
        Sequential Halving over completed-Q with a min-max rescale -- there
        is no UCB selection term to add sigma-hat to. We therefore reuse the
        paper's core signal (per-action value variance, tracked the same way
        via a running sum of squares) but apply it where our operator is
        actually vulnerable: gating how far each completed-Q is allowed to
        depart from the mixed value before the noise-amplifying rescale.
        """
        root_to_act = str(node.game.current_color()) == str(node.root_color)
        sign = 1.0 if root_to_act else -1.0

        # CAT-61: with uncertainty backup weighting on, a visited action's Q is
        # its backup-weight-weighted mean (`weighted_q`) rather than the plain
        # visit mean (`q`). `weighted_q` falls back to exactly `q` when no
        # weighted backups were recorded, so the default (flag-off) path uses
        # `stats.q` and is bit-identical.
        use_weighted = bool(self.config.uncertainty_backup_weighting)

        def q_of(stats: _GAction) -> float:
            return stats.weighted_q if use_weighted else stats.q

        total_child_visits = sum(stats.visits for stats in node.actions.values())
        visited_prior_sum = 0.0
        visited_q_sum = 0.0
        for stats in node.actions.values():
            if stats.visits > 0:
                visited_prior_sum += stats.prior
                visited_q_sum += stats.prior * (sign * q_of(stats))
        weighted_q = (visited_q_sum / visited_prior_sum) if visited_prior_sum > 0 else 0.0

        node_value = sign * node.prior_value
        v_mix = (node_value + total_child_visits * weighted_q) / (1.0 + total_child_visits)

        completed = {
            action_id: (sign * q_of(stats)) if stats.visits > 0 else v_mix
            for action_id, stats in node.actions.items()
        }

        if bool(self.config.variance_aware_q):
            self._shrink_completed_q_by_variance(node, completed, v_mix)
        return completed

    def _shrink_completed_q_by_variance(
        self, node: _GNode, completed: dict[int, float], v_mix: float
    ) -> None:
        """Mutate `completed` in place, shrinking each visited candidate's Q
        toward `v_mix` by its standard error (D2 -- see `_completed_q`)."""
        visited = [
            (action_id, stats)
            for action_id, stats in node.actions.items()
            if stats.visits > 0
        ]
        if len(visited) < 2:
            return  # no between-candidate spread to preserve
        visited_qs = [completed[action_id] for action_id, _stats in visited]
        mean_q = sum(visited_qs) / len(visited_qs)
        signal_var = sum((q - mean_q) ** 2 for q in visited_qs) / len(visited_qs)
        if signal_var <= 0.0:
            return
        if bool(self.config.variance_aware_closed_form_js):
            # CAT-61: closed-form James-Stein. ONE shrinkage coefficient shared
            # by every visited candidate -- lambda* = v2 / (v2 + s2) -- where v2
            # is the across-candidate (between-arm) variance `signal_var` and s2
            # is the mean within-candidate sampling variance (mean of SE_a^2).
            # This is the empirical-Bayes optimum with no free k. lambda* is in
            # [0, 1] by construction (both variances are >= 0 and v2 > 0 here).
            se_sqs = [stats.q_variance / float(stats.visits) for _id, stats in visited]
            mean_se_sq = sum(se_sqs) / len(se_sqs)
            lam = signal_var / (signal_var + mean_se_sq)
            for action_id, _stats in visited:
                completed[action_id] = v_mix + lam * (completed[action_id] - v_mix)
            return
        k = float(self.config.variance_aware_k)
        for action_id, stats in visited:
            se_sq = stats.q_variance / float(stats.visits)
            shrink = signal_var / (signal_var + k * se_sq)
            completed[action_id] = v_mix + shrink * (completed[action_id] - v_mix)

    @staticmethod
    def _rescale_completed_q(completed_q: dict[int, float], *, epsilon: float = 1.0e-8) -> dict[int, float]:
        """Min-max rescale completed Q to [0, 1] over the node's own action
        range (mctx's `_rescale_values`), BEFORE the sigma transform.

        F1a: without this, sigma(q) = scale * q is applied to whatever raw
        magnitude Q happens to have -- unbounded sharpening of the improved
        policy (and thus of the played move, which follows
        argmax(logits + sigma(completed_q))). mctx bounds sigma's output to
        [0, scale] specifically by rescaling first; this is not optional
        polish, it's the mechanism that keeps completed-Q from overriding
        the prior with 15-64-visit noise.
        """
        if not completed_q:
            return {}
        values = completed_q.values()
        min_q = min(values)
        max_q = max(values)
        denom = (max_q - min_q) + epsilon
        return {action_id: (value - min_q) / denom for action_id, value in completed_q.items()}

    def _rescaled_completed_q(
        self,
        node: _GNode,
        completed_q: dict[int, float],
        *,
        mean_visits_override: float | None = None,
    ) -> dict[int, float]:
        """Min-max rescale (`_rescale_completed_q`) followed by the D1
        noise-floor attenuation (`_apply_noise_floor`). This is the single
        entry point the improved policy and the Sequential-Halving re-ranking
        both use, so the attenuation applies uniformly wherever completed-Q
        enters selection. Exactly equals `_rescale_completed_q` when
        `config.rescale_noise_floor_c == 0` (attenuation short-circuits)."""
        rescaled = self._rescale_completed_q(completed_q)
        return self._apply_noise_floor(
            node,
            completed_q,
            rescaled,
            mean_visits_override=mean_visits_override,
        )

    def _apply_noise_floor(
        self,
        node: _GNode,
        completed_q: dict[int, float],
        rescaled: dict[int, float],
        *,
        mean_visits_override: float | None = None,
    ) -> dict[int, float]:
        """D1: blend rescaled completed-Q toward the neutral 0.5 when the raw
        completed-Q spread is small relative to the per-candidate evaluation
        noise floor. See `GumbelChanceMCTSConfig.rescale_noise_floor_c`.

        alpha = raw_spread / (raw_spread + c * sigma_eval / sqrt(mean_visits))
        out   = 0.5 + alpha * (rescaled - 0.5)

        Properties: c == 0 -> returns `rescaled` unchanged (exact no-op);
        an exact tie (raw_spread == 0) -> alpha == 0 -> every value 0.5
        (constant, so the prior's order is preserved); mean_visits -> inf
        (noise floor -> 0) -> alpha -> 1 -> converges exactly to `rescaled`.
        """
        c = float(self.config.rescale_noise_floor_c)
        if c <= 0.0 or not rescaled:
            return rescaled
        values = completed_q.values()
        raw_spread = max(values) - min(values)
        if mean_visits_override is None:
            visits = [stats.visits for stats in node.actions.values()]
            mean_visits = (sum(visits) / len(visits)) if visits else 0.0
        else:
            mean_visits = float(mean_visits_override)
            if not math.isfinite(mean_visits) or mean_visits < 0.0:
                raise ValueError("mean_visits_override must be finite and >= 0")
        if mean_visits <= 0.0:
            noise_floor = float("inf")
        else:
            noise_floor = c * float(self.config.sigma_eval) / math.sqrt(mean_visits)
        denom = raw_spread + noise_floor
        if denom <= 0.0 or math.isinf(noise_floor):
            alpha = 0.0
        else:
            alpha = raw_spread / denom
        return {action_id: 0.5 + alpha * (value - 0.5) for action_id, value in rescaled.items()}

    def _improved_policy(
        self,
        node: _GNode,
        completed_q: dict[int, float],
        *,
        mean_visits_override: float | None = None,
    ) -> dict[int, float]:
        scale = self._sigma_scale(node)
        rescaled_q = self._rescaled_completed_q(
            node,
            completed_q,
            mean_visits_override=mean_visits_override,
        )
        logits = node.action_logits
        scores = {
            action_id: logits.get(action_id, 0.0) + scale * rescaled_q.get(action_id, 0.0)
            for action_id in node.actions
        }
        return _softmax_from_scores(scores)

    def _select_nonroot_action(self, node: _GNode) -> tuple[int, _GAction]:
        completed_q = self._completed_q(node)
        improved = self._improved_policy(node, completed_q)
        total_visits = sum(stats.visits for stats in node.actions.values())
        best: tuple[int, _GAction] | None = None
        best_score = float("-inf")
        for action_id, stats in node.actions.items():
            score = improved.get(action_id, 0.0) - stats.visits / (1.0 + total_visits)
            if score > best_score:
                best_score = score
                best = (action_id, stats)
        if best is None:  # pragma: no cover - guarded by caller.
            raise RuntimeError("cannot select action from empty MCTS node")
        return best

    # ------------------------------------------------------------------
    # Simulation / backup.
    # ------------------------------------------------------------------
    def _backup_weight(self, uncertainty: float) -> float:
        """Return KataGo's inverse-uncertainty backup weight (CAT-61).

        ``uncertainty`` is our head's predicted squared value error. Convert it
        to an error scale before applying KataGo's exponent, then use the
        coefficient/max-weight floor from ``searchupdatehelpers.cpp``. A zero
        (terminal or uniformly non-emitting evaluator) receives ``cap``; larger
        predicted error monotonically receives less weight.
        """
        predicted_squared_error = max(0.0, float(uncertainty))
        a = float(self.config.uncertainty_backup_a)
        exp = float(self.config.uncertainty_backup_exp)
        cap = float(self.config.uncertainty_backup_cap)
        if (
            not math.isfinite(a)
            or not math.isfinite(exp)
            or not math.isfinite(cap)
            or a <= 0.0
            or exp <= 0.0
            or cap <= 0.0
        ):
            raise ValueError(
                "uncertainty backup coefficient, exponent, and cap must be "
                "finite and > 0"
            )
        sigma = math.sqrt(predicted_squared_error)
        return a / ((sigma**exp) + a / cap)

    def _accumulate_backup_weight(
        self, stats: _GAction, value: float, uncertainty: float
    ) -> None:
        """Record a weighted backup on `stats` (CAT-61). Called ONLY from inside
        an `if config.uncertainty_backup_weighting:` guard at each backup site,
        so the plain `value_sum`/`visits`/`value_sq_sum` accumulation stays
        byte-for-byte identical on the default path. Optionally records the
        realized weight into `self._last_backup_weights` for smoke-test
        telemetry (verifying the cap engages)."""
        weight = self._backup_weight(uncertainty)
        stats.weight_sum += weight
        stats.weighted_value_sum += weight * value
        recorder = getattr(self, "_last_backup_weights", None)
        if recorder is not None:
            recorder.append(weight)

    @staticmethod
    def _expected_outcome_uncertainty(stats: _GAction) -> float:
        """Probability-weighted mean of the child leaves' `prior_uncertainty`
        for an expectation-backed-up chance action (ROLL / robber / dev-card),
        matching how that action's backed-up value is itself a probability-
        weighted mean over the same children (CAT-61)."""
        return sum(
            stats.probabilities.get(index, 0.0) * child.prior_uncertainty
            for index, child in stats.children.items()
        )

    def _prepare_simulation(
        self, node: _GNode, *, depth: int, forced_action: int | None = None
    ) -> float | _PendingSimulation:
        """Select one simulation and stop at the next neural leaf.

        This is the split-phase counterpart of ``_simulate`` used only by the
        flag-gated root-wave scheduler. Full-enumeration chance actions keep the
        established synchronous traversal because one logical chance expansion
        may itself require several neural rows and its expectation backup must
        observe all children together.
        """
        winner = node.game.winning_color()
        if winner is not None:
            return 1.0 if str(winner) == node.root_color else -1.0

        # Actor-turn PIMC cutoff: opponent policy/value may use that opponent's
        # own hand in the SAMPLED world, but we never search beyond that first
        # boundary (which would let the root actor condition later choices on a
        # single determinization).  The leaf is safe because the sampled world
        # was constructed independently of authoritative hidden truth.
        if self._is_information_set_turn_boundary(node, depth=depth):
            if not node.expanded:
                return self._pending_expansion(node, record_leaf_visit=True)
            value = node.prior_value
            node.visits += 1
            node.value_sum += value
            return value

        if depth >= int(self.config.max_depth):
            if not node.expanded:
                return self._pending_expansion(node, record_leaf_visit=False)
            return node.prior_value

        if not node.expanded:
            if self._expand_forced(node):
                return self._prepare_simulation(
                    node, depth=depth, forced_action=forced_action
                )
            return self._pending_expansion(node, record_leaf_visit=True)

        if not node.actions:
            value = node.prior_value
            node.visits += 1
            node.value_sum += value
            return value

        if forced_action is not None:
            action_id = forced_action
            stats = node.actions[action_id]
        else:
            action_id, stats = self._select_nonroot_action(node)

        action_json = node.action_json[action_id]
        if _action_type(action_json) == "ROLL" and not (
            bool(self.config.lazy_interior_chance) and depth > 0
        ):
            value = self._traverse_roll(node, action_id, stats, depth)
        elif is_move_robber_with_victim(action_json) or _action_type(
            action_json
        ) == "BUY_DEVELOPMENT_CARD":
            value = self._traverse_robber_or_dev(node, action_id, stats, depth)
        else:
            return self._prepare_single_sample(node, action_id, stats, depth)

        node.visits += 1
        node.value_sum += value
        return value

    def _pending_expansion(
        self, node: _GNode, *, record_leaf_visit: bool
    ) -> _PendingSimulation:
        legal_actions, action_json_by_id, spectrum_by_id = self._fetch_legal_actions(
            node.game
        )

        def finish(result: Any) -> float:
            priors, value, uncertainty = _split_evaluation(result)
            value = self._finish_expand(
                node,
                legal_actions,
                action_json_by_id,
                spectrum_by_id,
                priors,
                value,
                uncertainty,
            )
            if record_leaf_visit:
                node.visits += 1
                node.value_sum += value
            return value

        return _PendingSimulation(
            node=node,
            legal_actions=legal_actions,
            finish=finish,
        )

    def _prepare_single_sample(
        self, node: _GNode, action_id: int, stats: _GAction, depth: int
    ) -> float | _PendingSimulation:
        action_json = node.action_json[action_id]
        if not stats.probabilities:
            cached_spectrum = node.action_spectrum.get(action_id)
            outcomes = (
                cached_spectrum
                if cached_spectrum is not None
                else _spectrum(node.game, action_json)
            )
            node.action_spectrum.setdefault(action_id, outcomes)
            stats.probabilities = dict(outcomes)

        outcome_index = self._sample_outcome(tuple(stats.probabilities.items()))
        child = stats.children.get(outcome_index)
        if child is None:
            child_game = node.game.apply_chance_outcome(
                json.dumps(action_json), outcome_index
            )
            child = _GNode(game=child_game, root_color=node.root_color)
            stats.children[outcome_index] = child

        selected = self._prepare_simulation(child, depth=depth + 1)

        def backup(value: float) -> float:
            stats.visits += 1
            stats.value_sum += value
            stats.value_sq_sum += value * value
            if self.config.uncertainty_backup_weighting:
                self._accumulate_backup_weight(
                    stats, value, child.prior_uncertainty
                )
            node.visits += 1
            node.value_sum += value
            return value

        if isinstance(selected, _PendingSimulation):
            finish_child = selected.finish

            def finish(result: Any) -> float:
                return backup(finish_child(result))

            selected.finish = finish
            return selected
        return backup(selected)

    def _simulate(self, node: _GNode, *, depth: int, forced_action: int | None = None) -> float:
        winner = node.game.winning_color()
        if winner is not None:
            return 1.0 if str(winner) == node.root_color else -1.0
        if self._is_information_set_turn_boundary(node, depth=depth):
            if not node.expanded:
                value = self._expand(node)
            else:
                value = node.prior_value
            node.visits += 1
            node.value_sum += value
            return value
        if depth >= int(self.config.max_depth):
            if not node.expanded:
                self._expand(node)
            return node.prior_value
        if not node.expanded:
            if self._expand_forced(node):
                return self._simulate(node, depth=depth, forced_action=forced_action)
            value = self._expand(node)
            node.visits += 1
            node.value_sum += value
            return value
        if not node.actions:
            value = node.prior_value
            node.visits += 1
            node.value_sum += value
            return value

        if forced_action is not None:
            action_id = forced_action
            stats = node.actions[action_id]
        else:
            action_id, stats = self._select_nonroot_action(node)

        action_json = node.action_json[action_id]
        # `lazy_interior_chance` routes INTERIOR (depth > 0) ROLL actions
        # through the generic single-sample path below (ROLL spectra were
        # never subject to the A19/A20 corrections, and the expansion-cached
        # spectrum is reused, so single-sampling them is a drop-in lazy
        # traversal). Root ROLLs (depth == 0) keep full enumeration in both
        # modes. The F7 robber/dev-card branch is deliberately NOT gated:
        # those candidates are already materialized for real-vs-phantom
        # classification, so enumerating them is free either way.
        if _action_type(action_json) == "ROLL" and not (
            bool(self.config.lazy_interior_chance) and depth > 0
        ):
            value = self._traverse_roll(node, action_id, stats, depth)
        elif is_move_robber_with_victim(action_json) or _action_type(action_json) == "BUY_DEVELOPMENT_CARD":
            # F7: enumerate + expectation-backup like ROLL, instead of
            # single-sampling -- these outcome spaces are small (<=5) and
            # `move_robber_victim_outcome_weights`/`buy_development_card_real_outcomes`
            # already materialize every real candidate to classify real vs.
            # phantom, so full enumeration costs nothing extra it wasn't
            # already paying, and kills a variance term (a single-sampled
            # child's value was a 1-sample estimate of what should be an
            # exact expectation) that fed into F1's instability.
            value = self._traverse_robber_or_dev(node, action_id, stats, depth)
        else:
            value = self._traverse_single_sample(node, action_id, stats, depth)

        node.visits += 1
        node.value_sum += value
        return value

    def _is_information_set_turn_boundary(self, node: _GNode, *, depth: int) -> bool:
        if not bool(self.config.information_set_search) or depth <= 0:
            return False
        if str(node.game.current_color()) != str(node.root_color):
            return True
        root_turn = getattr(self, "_information_set_root_turn", None)
        return root_turn is not None and int(node.game.num_turns()) != int(root_turn)

    def _enumerate_roll_outcomes(
        self,
        game: Any,
        action_json: Any,
        *,
        root_color: str,
        cached_spectrum: tuple[tuple[int, float], ...] | None = None,
    ) -> tuple[dict[int, "_GNode"], dict[int, float], float]:
        """Enumerate all (up to 11) dice outcomes with exact spectrum probabilities.

        Each child is expanded (one evaluator leaf call) immediately so the
        returned afterstate value is a real probability-weighted average, not a
        placeholder -- this is what makes the ROLL chance node "true" rather
        than single-sampled.

        Uses `Game.apply_chance_outcomes_batch` (one call materializes every
        child) instead of 11 sequential `apply_chance_outcome` calls, and
        `self.evaluator.evaluate_many` (one batched forward pass, if the
        evaluator supports it) instead of 11 sequential leaf evaluations,
        when `use_batch_api` is enabled and the wheel/evaluator support it.
        `cached_spectrum` avoids a second `spectrum_json` round-trip when the
        caller already has it from `decision_context_json` at expansion time.
        """
        outcomes = cached_spectrum if cached_spectrum is not None else _spectrum(game, action_json)
        positive_outcomes = [(index, probability) for index, probability in outcomes if probability > 0.0]
        if not positive_outcomes:
            return {}, {}, 0.0

        use_batch = self.config.use_batch_api and batch_api_available()
        # OPT-7: serialize action_json ONCE. The else-branch comprehension below
        # otherwise re-dumps the same list for every outcome index; the batch
        # branch already dumped once. json.dumps is deterministic for a fixed
        # object, so the string handed to the Rust API is identical either way.
        raw_action_json = json.dumps(action_json)
        if use_batch:
            positive_indices = [index for index, _probability in positive_outcomes]
            child_games_list = game.apply_chance_outcomes_batch(
                raw_action_json, positive_indices
            )
            child_games = dict(zip(positive_indices, child_games_list))
        else:
            child_games = {
                index: game.apply_chance_outcome(raw_action_json, index)
                for index, _probability in positive_outcomes
            }

        children: dict[int, _GNode] = {
            index: _GNode(game=child_games[index], root_color=root_color)
            for index, _probability in positive_outcomes
        }
        probabilities: dict[int, float] = dict(positive_outcomes)

        can_batch_evaluate = hasattr(self.evaluator, "evaluate_many")
        if can_batch_evaluate:
            contexts = {index: self._fetch_legal_actions(child.game) for index, child in children.items()}
            requests = [(child.game, contexts[index][0]) for index, child in children.items()]
            batch_results = _evaluate_many_checked(
                self.evaluator,
                requests,
                root_color=root_color,
                colors=self.config.colors,
            )
            for (index, child), result in zip(children.items(), batch_results):
                priors, value, uncertainty = _split_evaluation(result)
                legal_actions, action_json_by_id, spectrum_by_id = contexts[index]
                self._finish_expand(
                    child, legal_actions, action_json_by_id, spectrum_by_id, priors, value, uncertainty
                )
        else:
            for child in children.values():
                self._expand(child)

        weighted_prior_sum = sum(
            probabilities[index] * children[index].prior_value for index in children
        )
        total_probability = sum(probabilities.values())
        afterstate_value = (
            weighted_prior_sum / total_probability if total_probability > 0.0 else 0.0
        )
        return children, probabilities, afterstate_value

    def _traverse_roll(self, node: _GNode, action_id: int, stats: _GAction, depth: int) -> float:
        if not stats.children:
            stats.children, stats.probabilities, stats.afterstate_value = (
                self._enumerate_roll_outcomes(
                    node.game,
                    node.action_json[action_id],
                    root_color=node.root_color,
                    cached_spectrum=node.action_spectrum.get(action_id),
                )
            )
            if not stats.children:
                # Pathological: spectrum_json returned no positive-probability
                # outcomes. Fall back to the node's own leaf estimate instead
                # of sampling from an empty outcome set (which would IndexError).
                stats.afterstate_value = node.prior_value
                stats.visits += 1
                stats.value_sum += node.prior_value
                stats.value_sq_sum += node.prior_value * node.prior_value
                if self.config.uncertainty_backup_weighting:
                    self._accumulate_backup_weight(
                        stats, node.prior_value, node.prior_uncertainty
                    )
                return node.prior_value

        outcome_index = self._sample_outcome(tuple(stats.probabilities.items()))
        self._simulate(stats.children[outcome_index], depth=depth + 1)

        value = sum(
            stats.probabilities[index] * stats.children[index].value for index in stats.children
        )
        stats.visits += 1
        stats.value_sum += value
        stats.value_sq_sum += value * value
        if self.config.uncertainty_backup_weighting:
            self._accumulate_backup_weight(
                stats, value, self._expected_outcome_uncertainty(stats)
            )
        return value

    def _traverse_single_sample(
        self, node: _GNode, action_id: int, stats: _GAction, depth: int
    ) -> float:
        """Generic non-ROLL, non-robber/dev traversal: single-sample from
        the action's spectrum (a deterministic action's spectrum degenerates
        to one guaranteed outcome via `_spectrum`'s all-zero-probability
        fallback, so this also correctly handles ordinary deterministic
        actions, not just chance ones)."""
        action_json = node.action_json[action_id]
        if not stats.probabilities:
            cached_spectrum = node.action_spectrum.get(action_id)
            outcomes = (
                cached_spectrum if cached_spectrum is not None else _spectrum(node.game, action_json)
            )
            node.action_spectrum.setdefault(action_id, outcomes)
            stats.probabilities = dict(outcomes)

        outcome_index = self._sample_outcome(tuple(stats.probabilities.items()))
        child = stats.children.get(outcome_index)
        if child is None:
            child_game = node.game.apply_chance_outcome(json.dumps(action_json), outcome_index)
            child = _GNode(game=child_game, root_color=node.root_color)
            stats.children[outcome_index] = child

        value = self._simulate(child, depth=depth + 1)
        stats.visits += 1
        stats.value_sum += value
        stats.value_sq_sum += value * value
        if self.config.uncertainty_backup_weighting:
            self._accumulate_backup_weight(stats, value, child.prior_uncertainty)
        return value

    def _enumerate_materialized_outcomes(
        self,
        candidates: list[tuple[int, float, Any]],
        *,
        root_color: str,
    ) -> tuple[dict[int, "_GNode"], dict[int, float], float]:
        """F7: like `_enumerate_roll_outcomes`, but starting from an
        already-materialized (index, weight, child_game) candidates list
        (from `move_robber_victim_outcome_weights`/
        `buy_development_card_real_outcomes`'s phantom-filtering pass)
        instead of re-deriving children from a raw spectrum -- that pass
        already paid the cost of applying+diffing every real outcome, so
        this is a zero-extra-materialization expansion."""
        total_weight = sum(weight for _index, weight, _game in candidates)
        if not candidates or total_weight <= 0.0:
            return {}, {}, 0.0
        probabilities = {index: weight / total_weight for index, weight, _game in candidates}
        children: dict[int, _GNode] = {
            index: _GNode(game=child_game, root_color=root_color)
            for index, _weight, child_game in candidates
        }

        can_batch_evaluate = hasattr(self.evaluator, "evaluate_many")
        if can_batch_evaluate:
            contexts = {index: self._fetch_legal_actions(child.game) for index, child in children.items()}
            requests = [(child.game, contexts[index][0]) for index, child in children.items()]
            batch_results = _evaluate_many_checked(
                self.evaluator,
                requests,
                root_color=root_color,
                colors=self.config.colors,
            )
            for (index, child), result in zip(children.items(), batch_results):
                priors, value, uncertainty = _split_evaluation(result)
                legal_actions, action_json_by_id, spectrum_by_id = contexts[index]
                self._finish_expand(
                    child, legal_actions, action_json_by_id, spectrum_by_id, priors, value, uncertainty
                )
        else:
            for child in children.values():
                self._expand(child)

        afterstate_value = sum(probabilities[index] * children[index].prior_value for index in children)
        return children, probabilities, afterstate_value

    def _traverse_robber_or_dev(
        self, node: _GNode, action_id: int, stats: _GAction, depth: int
    ) -> float:
        """F7: MOVE_ROBBER-with-victim and BUY_DEVELOPMENT_CARD chance
        outcomes, enumerated + expectation-backed-up exactly like ROLL
        (small outcome spaces, <=5, so this is cheap) instead of
        single-sampled. When `correct_rust_chance_spectra` is False, or
        shape-detection finds nothing to correct (fixed wheel,
        MOVE_ROBBER-with-victim only), falls through to enumerating the
        native spectrum directly via the same machinery `_traverse_roll`
        uses (`_enumerate_roll_outcomes` is generic over any action_json
        with a spectrum, not ROLL-specific)."""
        if not stats.children:
            action_json = node.action_json[action_id]
            cached_spectrum = node.action_spectrum.get(action_id)
            candidates: list[tuple[int, float, Any]] | None = None
            if self.config.belief_chance_spectra:
                # Planner-only public-belief de-leak (config docstring). Takes
                # precedence over correct_rust_chance_spectra: belief IS a
                # correction and never passes the native (true-state) spectrum
                # through. Always returns a list (never None), so it cannot fall
                # into the native-enumeration branch below.
                if is_move_robber_with_victim(action_json):
                    candidates = belief_move_robber_outcome_weights(
                        node.game,
                        action_json,
                        cached_spectrum=cached_spectrum,
                        perspective=node.root_color,
                    )
                else:
                    candidates = belief_buy_development_card_outcomes(
                        node.game,
                        action_json,
                        cached_spectrum=cached_spectrum,
                        perspective=node.root_color,
                    )
            elif self.config.correct_rust_chance_spectra:
                if is_move_robber_with_victim(action_json):
                    candidates = move_robber_victim_outcome_weights(
                        node.game, action_json, cached_spectrum=cached_spectrum
                    )
                else:
                    candidates = buy_development_card_real_outcomes(
                        node.game, action_json, cached_spectrum=cached_spectrum
                    )
            if candidates is None:
                stats.children, stats.probabilities, stats.afterstate_value = (
                    self._enumerate_roll_outcomes(
                        node.game, action_json, root_color=node.root_color, cached_spectrum=cached_spectrum
                    )
                )
            else:
                stats.children, stats.probabilities, stats.afterstate_value = (
                    self._enumerate_materialized_outcomes(candidates, root_color=node.root_color)
                )
            if not stats.children:
                # Defensive: no real outcome at all (should not happen) --
                # fall back to the node's own leaf estimate rather than
                # sampling from an empty outcome set.
                stats.afterstate_value = node.prior_value
                stats.visits += 1
                stats.value_sum += node.prior_value
                stats.value_sq_sum += node.prior_value * node.prior_value
                if self.config.uncertainty_backup_weighting:
                    self._accumulate_backup_weight(
                        stats, node.prior_value, node.prior_uncertainty
                    )
                return node.prior_value

        outcome_index = self._sample_outcome(tuple(stats.probabilities.items()))
        self._simulate(stats.children[outcome_index], depth=depth + 1)

        value = sum(
            stats.probabilities[index] * stats.children[index].value for index in stats.children
        )
        stats.visits += 1
        stats.value_sum += value
        stats.value_sq_sum += value * value
        if self.config.uncertainty_backup_weighting:
            self._accumulate_backup_weight(
                stats, value, self._expected_outcome_uncertainty(stats)
            )
        return value

    def _corrected_move_robber_outcome(self, game: Any, action_json: Any) -> tuple[int, Any]:
        """Sample a stolen resource weighted by the victim's REAL hand.

        Single-shot, non-cached convenience wrapper (kept for direct
        unit-testing and one-off callers) -- see `_traverse_single_sample`
        for the cached path search actually uses, and
        `move_robber_victim_outcome_weights` (module-level, shared with
        `catan_zero.rl.gumbel_self_play`'s live-game chance resolution) for
        the shape-aware, materialize-and-observe weight computation itself.
        """
        candidates = move_robber_victim_outcome_weights(game, action_json)
        if candidates is None:
            # Shape-detection found the native spectrum already correctly
            # hand-weighted -- pass through natively, zero extra work.
            outcomes = _spectrum(game, action_json)
            outcome_index = self._sample_outcome(outcomes)
            return outcome_index, game.apply_chance_outcome(json.dumps(action_json), outcome_index)
        if not candidates:
            # Defensive: no real single-resource-steal outcome at all (should
            # not happen for a victim with >=1 card) -- fall back to the raw
            # highest-probability outcome rather than crashing the search.
            outcomes = _spectrum(game, action_json)
            outcome_index, _probability = max(outcomes, key=lambda item: item[1])
            return outcome_index, game.apply_chance_outcome(json.dumps(action_json), outcome_index)

        total = sum(weight for _index, weight, _game in candidates)
        normalized = tuple((index, weight / total) for index, weight, _game in candidates)
        chosen_index = self._sample_outcome(normalized)
        chosen_game = next(
            candidate_game
            for index, _weight, candidate_game in candidates
            if index == chosen_index
        )
        return chosen_index, chosen_game

    def _corrected_buy_dev_card_outcome(self, game: Any, action_json: Any) -> tuple[int, Any]:
        """Sample among only the REAL (card-drawing) BUY_DEVELOPMENT_CARD outcomes.

        Single-shot, non-cached convenience wrapper (kept for direct
        unit-testing and one-off callers) -- see `_traverse_single_sample`
        for the cached path search actually uses, and
        `buy_development_card_real_outcomes` (module-level, shared with
        `catan_zero.rl.gumbel_self_play`'s live-game chance resolution) for
        the phantom-outcome filtering itself.
        """
        real_candidates = buy_development_card_real_outcomes(game, action_json)
        if not real_candidates:
            # Defensive: no real card-drawing outcome at all (should not
            # happen with a non-empty deck) -- fall back to the raw
            # highest-probability outcome rather than crashing the search.
            outcomes = _spectrum(game, action_json)
            outcome_index, _probability = max(outcomes, key=lambda item: item[1])
            return outcome_index, game.apply_chance_outcome(json.dumps(action_json), outcome_index)

        total = sum(probability for _index, probability, _game in real_candidates)
        normalized = tuple(
            (index, probability / total) for index, probability, _game in real_candidates
        )
        chosen_index = self._sample_outcome(normalized)
        chosen_game = next(
            candidate_game
            for index, _probability, candidate_game in real_candidates
            if index == chosen_index
        )
        return chosen_index, chosen_game

    def _sample_outcome(self, outcomes: tuple[tuple[int, float], ...]) -> int:
        if len(outcomes) == 1:
            return outcomes[0][0]
        draw = self.rng.random()
        cumulative = 0.0
        for outcome_index, probability in outcomes:
            cumulative += probability
            if draw <= cumulative:
                return outcome_index
        return outcomes[-1][0]

    def _sample_categorical(self, policy: dict[int, float]) -> int:
        if not policy:
            raise RuntimeError("cannot sample from an empty policy")
        draw = self.rng.random()
        cumulative = 0.0
        items = list(policy.items())
        for action_id, probability in items:
            cumulative += probability
            if draw <= cumulative:
                return int(action_id)
        return int(items[-1][0])

    def _fetch_legal_actions(
        self, game: Any
    ) -> tuple[tuple[int, ...], dict[int, Any], dict[int, tuple[tuple[int, float], ...]]]:
        """Legal actions + action JSON (+ cached chance spectra), batch-API-aware.

        Uses `Game.decision_context_json` (one round trip) when
        `use_batch_api` is set and the installed wheel supports it, else
        falls back to `_legal_action_indices` + `_playable_action_json_by_index`
        (two round trips, no cached spectra) for compatibility with older
        wheels (e.g. wheel 0.1.0 on the local dev mirror).
        """
        if self.config.use_batch_api and batch_api_available():
            return _decision_context(game, colors=self.config.colors, map_kind=self.config.map_kind)
        legal_actions = _legal_action_indices(
            game, colors=self.config.colors, map_kind=self.config.map_kind
        )
        action_json_by_id = (
            _playable_action_json_by_index(game, legal_actions, self.config.colors, self.config.map_kind)
            if legal_actions
            else {}
        )
        return legal_actions, action_json_by_id, {}

    def _expand_forced(self, node: _GNode) -> bool:
        legal_actions, action_json_by_id, spectrum_by_id = self._fetch_legal_actions(node.game)
        if len(legal_actions) != 1:
            return False
        action_id = int(legal_actions[0])
        node.action_json = action_json_by_id
        node.action_spectrum = spectrum_by_id
        node.actions = {action_id: _GAction(prior=1.0)}
        node.action_logits = {action_id: 0.0}
        node.prior_value = _terminal_or_zero(node.game, node.root_color)
        node.expanded = True
        return True

    def _expand(self, node: _GNode, *, at_root: bool = False) -> float:
        legal_actions, action_json_by_id, spectrum_by_id = self._fetch_legal_actions(node.game)
        if (
            at_root
            and bool(self.config.symmetry_averaged_eval)
            and _matches_explicit_or_legacy_width_gate(
                len(legal_actions),
                min_legal_actions=self.config.symmetry_averaged_eval_threshold,
                legacy_exclusive_threshold=self.config.wide_candidates_threshold,
            )
            and hasattr(self.evaluator, "evaluate_symmetry_averaged")
        ):
            # f74b: denoise the wide-root value+prior by averaging the net over
            # all 12 D6 board orientations. Gated to wide roots (the ~4 placement
            # decisions/game) so we pay the 12x eval only where the ranking
            # failure lives; a no-op when the evaluator lacks the method.
            priors, value, uncertainty = _split_evaluation(
                self.evaluator.evaluate_symmetry_averaged(
                    node.game,
                    legal_actions,
                    root_color=node.root_color,
                    colors=self.config.colors,
                )
            )
        else:
            priors, value, uncertainty = _split_evaluation(
                self.evaluator.evaluate(
                    node.game, legal_actions, root_color=node.root_color, colors=self.config.colors
                )
            )
        return self._finish_expand(
            node, legal_actions, action_json_by_id, spectrum_by_id, priors, value, uncertainty
        )

    def _finish_expand(
        self,
        node: _GNode,
        legal_actions: tuple[int, ...],
        action_json_by_id: dict[int, Any],
        spectrum_by_id: dict[int, tuple[tuple[int, float], ...]],
        priors: dict[int, float],
        value: float,
        uncertainty: float = 0.0,
    ) -> float:
        """Pure bookkeeping half of node expansion, given an already-computed
        (priors, value[, uncertainty]) tuple. Split out from `_expand` so
        batch-evaluated children (see `_enumerate_roll_outcomes`) can skip a
        second, redundant per-child `evaluator.evaluate()` call. `uncertainty`
        (CAT-61) defaults to 0.0 so evaluators that do not emit it leave the
        capped backup weighting inert."""
        if legal_actions:
            missing = [action for action in legal_actions if action not in priors]
            if missing:
                floor = min((p for p in priors.values() if p > 0.0), default=1.0)
                for action in missing:
                    priors[int(action)] = floor * 0.01
            priors = _normalize_policy(
                {int(action): float(priors.get(int(action), 0.0)) for action in legal_actions}
            )
            node.action_json = action_json_by_id
            node.action_spectrum = spectrum_by_id
            node.actions = {
                int(action): _GAction(prior=float(priors[int(action)])) for action in legal_actions
            }
            prior_temperature = max(float(self.config.prior_temperature), 1.0e-6)
            node.action_logits = {
                int(action): math.log(max(float(priors[int(action)]), 1.0e-8)) / prior_temperature
                for action in legal_actions
            }
        node.prior_value = float(max(min(value, 1.0), -1.0))
        node.prior_uncertainty = max(0.0, float(uncertainty))
        node.expanded = True
        return node.prior_value


def _action_type(action_json: Any) -> str:
    if isinstance(action_json, (list, tuple)) and len(action_json) > 1:
        return str(action_json[1])
    return ""


def is_move_robber_with_victim(action_json: Any) -> bool:
    if _action_type(action_json) != "MOVE_ROBBER":
        return False
    value = action_json[2] if len(action_json) > 2 else None
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    return value[1] is not None


def move_robber_victim_outcome_weights(
    game: Any, action_json: Any, *, cached_spectrum: tuple[tuple[int, float], ...] | None = None
) -> list[tuple[int, float, Any]] | None:
    """Return (outcome_index, weight, materialized_child_game) for each native
    MOVE_ROBBER-with-victim outcome, weighted by the victim's REAL hand -- or
    `None` if the native spectrum is already correctly hand-weighted and
    there's nothing to correct.

    Rust's `spectrum_json` for MOVE_ROBBER-with-victim historically (wheel
    <0.1.1) always returned uniform 0.2 over a fixed 5-entry, RESOURCES-order
    index space regardless of what the victim actually held (verified bug
    A19); applying an outcome for a resource the victim didn't hold was a
    silent no-op. From 0.1.1 the native spectrum is itself hand-weighted, and
    its outcome-index space is no longer guaranteed to be the fixed
    RESOURCES order -- it can be a shorter, hand-filtered list instead (e.g.
    2 entries for a 2-distinct-resource hand). This function is SHAPE-AWARE:
    `_is_legacy_move_robber_spectrum` detects the pre-fix shape (exactly 5
    entries, uniform probability, independent of the hand); only then does it
    do the expensive materialize-and-observe correction (applying each native
    outcome and diffing the victim's hand before/after to see which resource
    it actually steals, exactly mirroring `buy_development_card_real_outcomes`'s
    pattern -- correct and outcome-index-safe regardless of wheel version).
    If the spectrum doesn't match the legacy shape, it's already correct and
    this returns `None` so the caller passes it through natively at zero
    extra cost -- this makes `correct_rust_chance_spectra=True` and `False`
    behave identically (and equally cheap) once the wheel is fixed, with no
    version sniffing. Weights are un-renormalized (callers should renormalize
    among whatever subset they keep). Shared by `GumbelChanceMCTS`'s internal
    search simulation and `catan_zero.rl.gumbel_self_play`'s live-game chance
    resolution.
    """
    outcomes = cached_spectrum if cached_spectrum is not None else _spectrum(game, action_json)
    if not _is_legacy_move_robber_spectrum(outcomes):
        return None

    victim_name = action_json[2][1]
    raw_json = json.dumps(action_json)
    snapshot = json.loads(game.json_snapshot())
    colors = [str(color) for color in snapshot["colors"]]
    victim_index = colors.index(str(victim_name))
    victim_hand = snapshot["player_state"][victim_index]["resources"]

    candidates: list[tuple[int, float, Any]] = []
    for outcome_index, _probability in outcomes:
        candidate_game = game.apply_chance_outcome(raw_json, outcome_index)
        candidate_hand = json.loads(candidate_game.json_snapshot())["player_state"][victim_index][
            "resources"
        ]
        stolen = [
            resource
            for resource in RESOURCES
            if int(victim_hand.get(resource, 0)) - int(candidate_hand.get(resource, 0)) == 1
        ]
        if len(stolen) != 1:
            continue  # defensive: not a real single-resource steal outcome
        weight = float(victim_hand.get(stolen[0], 0))
        if weight > 0.0:
            candidates.append((outcome_index, weight, candidate_game))
    return candidates


def _is_legacy_move_robber_spectrum(outcomes: tuple[tuple[int, float], ...]) -> bool:
    """Detect the pre-A19-fix MOVE_ROBBER-with-victim spectrum shape: exactly
    5 entries, all with the same (uniform) probability, independent of the
    victim's hand. A hand that happens to hold all 5 resources in equal
    counts would also produce this shape on a FIXED wheel, but correcting it
    anyway is harmless there (the correction is idempotent: it reproduces the
    same uniform distribution)."""
    if len(outcomes) != 5:
        return False
    probabilities = [probability for _index, probability in outcomes]
    return max(probabilities) - min(probabilities) < 1.0e-9


def buy_development_card_real_outcomes(
    game: Any,
    action_json: Any,
    *,
    cached_spectrum: tuple[tuple[int, float], ...] | None = None,
) -> list[tuple[int, float, Any]]:
    """Return (outcome_index, probability, materialized_child_game) for REAL outcomes.

    Rust's `spectrum_json` for BUY_DEVELOPMENT_CARD can include a
    large-probability "phantom" outcome that draws no card at all (verified
    bug A20). This materializes every outcome and discards phantoms (actor's
    dev card counts unchanged or deck count not decremented), exactly
    mirroring `catan_zero.adapters.engine_equivalence.apply_chance_step`'s
    proven-correct workaround. Probabilities are returned un-renormalized
    (callers should renormalize among whatever subset they keep). Shared by
    `GumbelChanceMCTS`'s internal search simulation and
    `catan_zero.rl.gumbel_self_play`'s live-game chance resolution. Unlike
    MOVE_ROBBER-with-victim, there's no cheap shape signature that predicts
    "no phantom outcomes" without materializing and diffing each one, so this
    always does the full pass -- it's naturally a no-op when there ARE no
    phantoms (every outcome is classified real, weights are the original
    native probabilities), and the spectrum here is typically short (2-5
    entries) so the cost is modest either way.
    """
    actor_color = str(action_json[0])
    raw_json = json.dumps(action_json)
    outcomes = cached_spectrum if cached_spectrum is not None else _spectrum(game, action_json)
    snapshot = json.loads(game.json_snapshot())
    colors = [str(color) for color in snapshot["colors"]]
    actor_index = colors.index(actor_color)
    before_cards = snapshot["player_state"][actor_index]["dev_cards"]
    before_deck_count = int(snapshot.get("development_deck_count", 0))

    real_candidates: list[tuple[int, float, Any]] = []
    for outcome_index, probability in outcomes:
        candidate_game = game.apply_chance_outcome(raw_json, outcome_index)
        candidate_snapshot = json.loads(candidate_game.json_snapshot())
        candidate_cards = candidate_snapshot["player_state"][actor_index]["dev_cards"]
        gained = [
            card
            for card in DEVELOPMENT_CARDS
            if int(candidate_cards.get(card, 0)) - int(before_cards.get(card, 0)) == 1
        ]
        deck_decreased = (
            int(candidate_snapshot.get("development_deck_count", 0)) == before_deck_count - 1
        )
        if len(gained) == 1 and deck_decreased:
            real_candidates.append((outcome_index, probability, candidate_game))
    return real_candidates


# Standard base Catan development-card deck composition (25 cards). Used ONLY by
# the planner-belief dev-draw spectrum to reconstruct the deck an actor could
# believe remains. Asserted against the engine's initial development_deck_count
# in tests/test_public_observation_masking.py.
BASE_DEVELOPMENT_DECK: dict[str, int] = {
    "KNIGHT": 14,
    "VICTORY_POINT": 5,
    "YEAR_OF_PLENTY": 2,
    "MONOPOLY": 2,
    "ROAD_BUILDING": 2,
}


def belief_move_robber_outcome_weights(
    game: Any,
    action_json: Any,
    *,
    cached_spectrum: tuple[tuple[int, float], ...] | None = None,
    perspective: str | None = None,
) -> list[tuple[int, float, Any]]:
    """PLANNER-belief variant of `move_robber_victim_outcome_weights`.

    Weights come from :class:`PublicBelief`, so an opponent victim is uniform
    over all five resource identities and the perspective player's own hand is
    count-weighted exactly.  On the legacy five-entry Rust spectrum the fixed
    resource/index mapping lets us retain all five children, including a no-op
    child when the authoritative hidden hand lacks the sampled type.  This is
    hidden-composition-invariant at the *belief distribution* boundary.

    A fixed Rust wheel may expose only true-hand-filtered outcomes with no
    stable resource/index mapping.  In that case we can only reweight the
    materializable outcomes, so the child set still leaks held resource types.
    Full removal needs determinizing the engine state before expansion; this
    function deliberately does not claim to solve that or opponent legal-action
    leakage (see ``public_belief.OPPONENT_ACTION_SCOPE``).
    """
    outcomes = cached_spectrum if cached_spectrum is not None else _spectrum(game, action_json)
    victim_name = str(action_json[2][1])
    actor_name = str(action_json[0])
    raw_json = json.dumps(action_json)
    snapshot = json.loads(game.json_snapshot())
    colors = [str(color) for color in snapshot["colors"]]
    victim_index = colors.index(victim_name)
    victim_hand = snapshot["player_state"][victim_index]["resources"]
    belief = PublicBelief.from_snapshot(
        snapshot,
        perspective=str(perspective) if perspective is not None else actor_name,
    )
    public_probabilities = belief.robber_steal_probabilities(victim_name)

    # Legacy wheels use an explicit RESOURCES-order five-entry chance space.
    # Keeping every public-belief outcome avoids leaking the held-type set.
    if _is_legacy_move_robber_spectrum(outcomes):
        candidates: list[tuple[int, float, Any]] = []
        for outcome_index, _probability in outcomes:
            if not 0 <= int(outcome_index) < len(RESOURCES):
                candidates = []
                break
            resource = RESOURCES[int(outcome_index)]
            weight = float(public_probabilities.get(resource, 0.0))
            if weight > 0.0:
                candidates.append(
                    (outcome_index, weight, game.apply_chance_outcome(raw_json, outcome_index))
                )
        if candidates:
            return candidates

    candidates = []
    for outcome_index, _probability in outcomes:
        candidate_game = game.apply_chance_outcome(raw_json, outcome_index)
        candidate_hand = json.loads(candidate_game.json_snapshot())["player_state"][victim_index][
            "resources"
        ]
        stolen = [
            resource
            for resource in RESOURCES
            if int(victim_hand.get(resource, 0)) - int(candidate_hand.get(resource, 0)) == 1
        ]
        if len(stolen) != 1:
            continue
        weight = float(public_probabilities.get(stolen[0], 0.0))
        if weight > 0.0:
            candidates.append((outcome_index, weight, candidate_game))
    return candidates


def belief_buy_development_card_outcomes(
    game: Any,
    action_json: Any,
    *,
    cached_spectrum: tuple[tuple[int, float], ...] | None = None,
    perspective: str | None = None,
) -> list[tuple[int, float, Any]]:
    """PLANNER-belief variant of `buy_development_card_real_outcomes`.

    Keeps the same real (card-drawing) outcomes, but reweights each by the
    posterior predictive distribution from :class:`PublicBelief`: base deck
    minus the perspective's own unplayed cards and every publicly played card.
    Opponents' face-down cards and the deck are exchangeable allocations of
    that pool. Planner-only; the live env keeps true-deck resolution.

    Residual leak (documented, accepted for v1): a card type held 100% face-down
    by opponents has no drawable engine outcome to materialize, so it cannot
    appear as a simulated draw even though the belief deck says it could.
    """
    actor_color = str(action_json[0])
    raw_json = json.dumps(action_json)
    outcomes = cached_spectrum if cached_spectrum is not None else _spectrum(game, action_json)
    snapshot = json.loads(game.json_snapshot())
    colors = [str(color) for color in snapshot["colors"]]
    actor_index = colors.index(actor_color)
    before_cards = snapshot["player_state"][actor_index]["dev_cards"]
    before_deck_count = int(snapshot.get("development_deck_count", 0))

    belief = PublicBelief.from_snapshot(
        snapshot,
        perspective=str(perspective) if perspective is not None else actor_color,
    )
    public_probabilities = belief.development_draw_probabilities()

    candidates: list[tuple[int, float, Any]] = []
    for outcome_index, _probability in outcomes:
        candidate_game = game.apply_chance_outcome(raw_json, outcome_index)
        candidate_snapshot = json.loads(candidate_game.json_snapshot())
        candidate_cards = candidate_snapshot["player_state"][actor_index]["dev_cards"]
        gained = [
            card
            for card in DEVELOPMENT_CARDS
            if int(candidate_cards.get(card, 0)) - int(before_cards.get(card, 0)) == 1
        ]
        deck_decreased = (
            int(candidate_snapshot.get("development_deck_count", 0)) == before_deck_count - 1
        )
        if len(gained) == 1 and deck_decreased:
            weight = float(public_probabilities.get(gained[0], 0.0))
            if weight > 0.0:
                candidates.append((outcome_index, weight, candidate_game))
    return candidates


def _softmax_from_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    weights = {
        action_id: math.exp(max(min(score - max_score, 40.0), -40.0))
        for action_id, score in scores.items()
    }
    return _normalize_policy(weights)
