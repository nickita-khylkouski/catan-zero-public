# Corrected Belief Gameplay Operator Differential Review

## Executive Summary

| Severity | Count |
|---|---:|
| Critical | 0 |
| High | 2 |
| Medium | 1 |
| Low | 0 |

**Overall risk before fixes:** high
**Recommendation after fixes and focused regression tests:** approve for bounded
diagnostic evaluation; promotion still requires the normal statistical gates.

The review covered commit `1a27613..dbda2b1` and the corrective changes made on
top of it. The root-perspective completed-Q contract, uniform hidden-world
aggregation, target/gameplay separation, native evidence fail-closed behavior,
per-role internal wiring, external receipt wiring, and legacy no-op path were
traced end to end.

## What Changed

The reviewed commit added an opt-in public-belief gameplay operator that averages
root-perspective completed-Q evidence across determinizations before applying one
policy improvement, plus role-specific evaluator controls in the H100 fleet
planner. Ten files changed in the original commit. This review added corrections
to the belief-level D1 calculation, the external neutral harness, evaluator
provenance schema, and focused tests.

## Findings and Corrections

### High: sparse belief roots lost all D1 Q evidence

`src/catan_zero/search/gumbel_chance_mcts.py` rounded each action's
particle-mean visits before D1 computed its noise floor. At wide n128/P4 roots,
positive means below 0.5 could all round to zero. D1 then treated the belief root
as unvisited and set `alpha=0`, silently reducing the corrected operator to the
prior regardless of completed-Q evidence.

The corrected path now supplies the exact fractional mean visits across actions
and particles to the shared D1 implementation. Ordinary search calls do not pass
the override, preserving the legacy path exactly. A sparse-root regression test
reproduces the former failure.

### High: external evaluation silently used the legacy gameplay operator

The fleet plan sealed role-specific corrected gameplay for internal
checkpoint-vs-checkpoint jobs, but external Catanatron jobs passed only the
role-specific `c_scale`. Consequently, an external receipt could claim the
corrected role contract while evaluating the checkpoint with legacy
`mean_improved_policy` gameplay.

`tools/catanatron_neutral_harness_match.py` now accepts, validates, executes, and
fingerprints `gameplay_policy_aggregation` and `sigma_reference_visits`.
`tools/fleet/a1_h100_eval_fleet.py` passes each role's complete sealed operator to
the external cohort and replays those arguments during plan validation.

### Medium: evaluator config schema was not advanced

Nine science-bearing fields were added to `EvalConfig` without advancing
`CONFIG_SCHEMA_VERSION`, contrary to the config format's own compatibility
contract. The schema is now version 10, so stale typed configs fail closed rather
than being interpreted under a changed field set.

## Invariant Review

- **Root perspective:** every experimental particle must attest
  `q_values_root_perspective`, expose finite completed-Q for every legal root
  action, and have Q support exactly consistent with positive visits.
- **Hidden information:** the authoritative game supplies only root actor and
  actor-legal actions. Search expansion occurs exclusively in
  `determinize_for_player` samples with public-observation evaluation, and stops
  when control or turn leaves the root actor.
- **Uniform belief weighting:** completed-Q vectors are averaged once per sampled
  world, never weighted by SH visits.
- **Target/gameplay separation:** corrected gameplay changes selected actions only
  when explicitly enabled. The learner target remains the configured target
  operator.
- **Native/Python boundary:** native owns per-particle traversal only; Python owns
  belief aggregation. Corrected aggregation refuses a native wheel that does not
  advertise completed-Q/root-perspective evidence.
- **Role isolation:** candidate and baseline separately resolve aggregation, D1,
  sigma estimate, and fixed sigma reference. Plan, command, science, config, and
  artifact fingerprints bind the resolved values.
- **Legacy no-op:** omitted role calibration still emits the historical fleet
  command; runtime defaults remain `mean_improved_policy`, D1 disabled, and
  realized-visit sigma.

## Test Coverage

Focused regression result: **160 passed, 18 skipped**. The wider search/config/
fleet suite was also run after the fixes. Added coverage includes sparse
fractional D1 visits, gameplay-vs-target separation, native capability refusal,
per-role internal construction, neutral-harness runtime/fingerprint binding,
external role argv binding, and plan reload/hash replay.

## Blast Radius

The modified aggregation is reachable only when information-set search is enabled
and either the target or gameplay aggregation explicitly selects
`aggregate_q_then_improve`. The D1 override is private to that aggregation call.
Neutral-harness and fleet additions are opt-in; legacy argv construction remains
unchanged. The config schema bump intentionally invalidates older executable
typed-config payloads.

## Methodology and Limitations

Strategy: focused differential review of all ten changed files, one-hop callers,
historical commits introducing belief targets, native binding evidence, fleet
plan/replay validation, and the neutral external evaluator. Static tracing was
combined with targeted pytest, Ruff, compile, and diff checks. No live fleet job
was launched, per review scope. Confidence is high for the reviewed Python/native
boundary and fleet wiring; statistical playing-strength benefit remains an
experimental question, not a code-review claim.
