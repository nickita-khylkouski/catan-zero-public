# A1 flywheel invariant audit — 2026-07-13

Scope: the canonical post-promotion turn, one-dose learner, typed evaluation,
promotion transaction, and the retired experimental continuous controller. This
is a code-path audit only; it did not deploy code, mutate a registry, or launch a
training/generation workload.

## Outcome

The canonical loop already has the right causal boundaries. It binds the live
promoted producer to the next campaign/corpus, consumes a corpus once, executes
one durably claimed learner dose, compares the candidate to its exact evaluation
parent under typed operator identities, and requires a positive replayed
adjudication before an atomic registry/pointer mutation.

Three integration defects remained and are fixed in this change:

1. The durable iteration orchestrator did not accept or forward the required
   promotion cohort-exclusions artifact. The real promotion verifier therefore
   could not be called from the canonical state machine. The artifact is now
   required, hash-bound in iteration state, forwarded in both dry-run and commit,
   and its derived disjointness result must match between preflight and commit
   (`tools/a1_iteration_orchestrator.py:1448-1664`).
2. `initialize-next` could permanently strand a fresh corpus if the process died
   after publishing the immutable turn/consumption claim but before publishing
   iteration state. It now adopts only an exact replay of the same turn and the
   same state-path-bound claim; any changed turn or consumer remains refused
   (`tools/a1_iteration_orchestrator.py:396-645`).
3. The retired experimental controller treated `gate_enabled=false` as a passing
   promotion and trusted a generic truthy `pass` field. Disabled evidence now
   holds, and promotion requires `ok is True`, `pass is True`, and an exact
   `promote` or `canary_promote` verdict
   (`tools/continuous_flywheel.py:430-447,793-805,1345`).

## End-to-end invariant map

| Historical failure | Existing canonical boundary | Audit result |
|---|---|---|
| Candidate chaining | A next turn requires the committed post-promotion handoff; learner and evaluation parents must equal its producer. The initializer must be that exact parent or a replayed function-preserving upgrade (`tools/a1_flywheel_turn.py:69-105,245-343`). Promotion independently requires the training/evaluation parent hash to equal the adjudicated incumbent and the registry/current pointer (`tools/a1_promotion_transaction.py:4880-4985`). | Guard exists and is fail closed. |
| Oversized/repeated dose | The corpus gets one immutable consumption claim bound to state path and turn (`tools/a1_iteration_orchestrator.py:396-471`). The learner owns a durable contract/lineage claim and receipts the exact command, corpus inventory, producer, outputs, and completed dose (`tools/a1_one_dose_train.py:2250-2345`). | Guard exists; crash-resume gap fixed. |
| Wrong comparison parent/operator | `a1_flywheel_turn` binds `evaluation_parent` to the promoted producer. The promotion transaction replays candidate/champion agent identities and role-specific search configurations, and requires candidate version `incumbent+1`. | Guard exists and is independently replayed. |
| Stale producer after promotion | The handoff replays the committed receipt plus live registry and `CURRENT_CHAMPION`; the next campaign must bind that exact handoff transaction. `initialize-next` refuses bootstrap semantics for a current turn (`tools/a1_flywheel_turn.py:69-105,182-206,245-287`). | Guard exists and is fail closed. |
| Mixed feature contracts | The next turn binds one exact contract, corpus metadata hash, payload inventory, selected/train/validation seed-set hashes, and completed generation audit (`tools/a1_flywheel_turn.py:288-316`). The feature-semantic contract/recompute validation is owned by corpus finalization and the separately integrated feature-semantics lane; the flywheel preserves those hashes and cannot substitute another corpus. | No duplicate flywheel implementation added. Re-run this boundary suite after the semantic lane lands on main. |
| Promotion on non-positive evidence | The canonical promotion verifier requires `passed is True` and `decision == "promote"`, then replays every evidence source. Cohort freshness is now part of the orchestrator's exact dry-run/commit equality. The legacy fail-open is fixed. | Fixed across both canonical integration and legacy fallback. |

## Function-level context

### `a1_flywheel_turn.build_turn`

- Inputs/assumptions: committed handoff, ready post-promotion campaign, passing
  generation audit, verified corpus contract, explicit learner/evaluation parent,
  and optional architecture-upgrade receipt.
- Effects: none; it returns a digest-sealed turn.
- Critical dependencies: handoff replay, campaign verifier, audit inventory and
  producer provenance, corpus metadata/payload identity, upgrade receipt replay.
- Failure posture: any producer, campaign, audit, corpus, parent, or initializer
  mismatch raises before learner state exists.

### `a1_iteration_orchestrator.initialize_next`

- Inputs/assumptions: one verified unconsumed contract and fresh learner outputs.
- Effects: publishes the immutable turn, one corpus-consumption claim, then the
  digest-sealed state.
- Recovery: exact pre-state publications are adoptable; different bytes/state
  destinations are not. This preserves one-consumer semantics without deleting
  or rewriting evidence.

### `a1_iteration_orchestrator.verify_evaluation` / `promote`

- Inputs/assumptions: completed one-dose outputs, exact registry/pointer,
  adjudication, cohort-exclusions manifest, and fresh receipt path.
- Effects: preflight only at `verify_evaluation`; registry/pointer mutation is
  delegated to the locked promotion transaction at `promote`.
- Recovery: a committed receipt is adopted only when its science/mutation fields,
  including cohort disjointness, equal the stored preflight.

### `continuous_flywheel.Runner.gate`

- Status: explicitly retired/noncanonical; production requires a separate
  acknowledgement.
- Remaining safety rule: absence/disablement of a gate is never positive
  evidence. The exact verdict predicate prevents compatibility fields from
  authorizing mutation.

## Verification

- Focused invariant tests cover exact next-turn recovery, cross-turn refusal,
  cohort artifact forwarding and drift, preflight/commit cohort equality, and
  typed positive legacy verdicts.
- No production host, registry, pointer, checkpoint, or GPU workload was changed.
