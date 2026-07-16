# A1 Current Codebase Execution Audit — 2026-07-16

## Executive conclusion

The repository now contains most of the components needed for a strong
two-player expert-iteration loop. The current coherent-public v3 path can now
authenticate and render its 8×B200 plan, but execution is intentionally
uncommissioned and the durable iteration orchestrator still cannot launch the
from-scratch learner.

The search/data implementation is substantially healthier than the learner
and orchestration integration. The next useful work is not another large data
wave. It is to commission one coherent training/evaluation path whose teacher,
target identity, value objective, optimizer dose, checkpoint selection, and
promotion transaction agree end to end.

## Integration update

The repair wave described by this audit is now on canonical `main` through
commit `73ad400`:

- width 640 is sealed at 41,700,000–42,000,000 parameters; the realized model
  has 41,708,233 parameters and measured 28.6% higher H100 training throughput
  than width 624;
- action-target gather now includes parameter-free local target identity,
  repairing opening-road target aliasing without changing checkpoint bytes or
  zero-step behavior;
- Python and Rust production/resource features use the same dice-pip semantics
  and always emit the deduction feature surface;
- Year of Plenty now preserves all 15 resource bundles in v3/v4/v5 instead of
  collapsing v5 to one all-zero resource row;
- scratch execution consumes an authenticated plan and selects across
  optimizer-step checkpoints;
- a bounded, non-promotable V0/V25/V100 value-routing campaign is available;
- binary-win BCE is implemented as an opt-in scalar value objective while
  deployed search still reads `tanh(scale * raw)`;
- validation ranges and value-evidence game identities now fail closed on
  partial ranges and cross-component seed collisions.

The integrated source completed a real two-step, eight-rank H100 DDP
forward/backward/AdamW run with BF16, width 640, the v5 learner surface,
binary-win BCE, and value-trunk gradient scale 0.25. This is an execution
smoke, not strength evidence.

## Verified current state

- Canonical GitHub base: `main` at
  `3e92b097fc7796a53687000df1d912cac2ac914f` at the start of this audit.
- The integration worktree also contains concurrent uncommitted feature/eval
  changes. They were preserved and audited in place rather than overwritten.
- The clean B200 audit checkout used that exact source and the installed CUDA
  12.8 / PyTorch 2.11 / `catanatron_rs` runtime.
- Search/data/action-focused tests: 140 passed.
- Learner/current-science focused tests: 187 passed.
- A broad current-tree run reached 1,057 passing tests and exposed two stale
  fixtures; both are corrected in this worktree.
- The actual archived production composite replay exposed three portability
  regressions in sequence: a new MCTS dataclass default reinterpreted an issued
  effective config, an implicit historical loss semantic was rejected before
  normalization, and verification required the verifier checkout's fleet
  manifest path. All three now preserve the issued bytes and semantics.
- After those repairs, the replay reaches and correctly refuses the true
  science mismatch: the only archived composite is PIMC/information-set data,
  not a coherent-public teacher composite.
- The current 8×B200 node was idle before this audit. Its older runtime checkout
  was detached and dirty, so it was not used as source authority.

## Software repaired in this audit

### 1. Issued contracts are replayable across additive code evolution

The verifier now preserves:

- PIMC locks issued before `rng_stream_separation`;
- recipes issued before explicit `policy_target_blend_semantics`;
- staged/archive locks whose authenticated fleet manifest is not at the
  verifier checkout's lexical path;
- guard-sync receipts against the synchronizer SHA bound in the sealed runtime
  tree, rather than against today's edited verifier.

This does not grant old locks new behavior. They replay only their original
effective operator and legacy target interpolation.

### 2. Plan and execution receipts are now separate

Plan-only writes an immutable plan receipt. `--go` consumes that plan and writes
a fresh execution receipt bound to it.

### 3. Promotion now consumes optimizer-step checkpoints

Training requests checkpoints at optimizer steps
`8,16,32,64,128,256,512,1024`. The training receipt and scratch promotion
transaction now authenticate and evaluate that optimizer-step frontier rather
than selecting only epoch terminals.

### 4. Other immediate regressions repaired

- Canonical P1 now binds its legacy target-loss operator explicitly.
- Issued P1 authorities still replay their historical implicit legacy operator.
- `weighted_cycle_mode` is initialized for ordinary non-AUX training.
- The scratch launcher accepts a normal venv Python symlink while hashing the
  resolved executable bytes.
- Mandatory-sequence target accounting uses the current v2 taxonomy value.
- Pipeline config schema is now 17; schema 16 remains immutable.
- `boundary_value_particles=1` is explicit in generation and evaluation
  identity.
- Resource production features now convert Catanatron probabilities to pip
  counts before action encoding.

## Remaining execution blockers

### 1. Optimizer schedule is still uncommissioned

The science contract declares `optimization_schedule_status=unresolved` and
`go_authorized=false`. This is correct fail-closed behavior: the 8×B200 plan
may be rendered, but `--go` must continue to refuse until the dose/value-routing
matrix selects a schedule.

### 2. Scratch learner and durable loop remain disconnected

The durable iteration orchestrator explicitly rejects
`initialization.mode=from_scratch`. Connect the scratch executor's authenticated
plan/execution/checkpoint-selection receipts to the same durable state machine
used by subsequent parent-initialized turns.

### 3. Current fleet authority is stale

Checked-in accepted fleet inventory still includes `192.222.55.216`, which is
not project-owned according to the current provider inventory. The previous
H100 fleet has also been archived.

Historical sealed operation records must remain immutable, but no checked-in
historical manifest should be treated as current launch authority. A new
private owner-attested manifest is required before fan-out.

### 4. Teacher adoption is not one recoverable multi-file transaction

The adoption command replaces several authority files and writes its receipt
last. Per-file replacement is atomic, but the complete authority change is not.
Add a journaled prepare/commit/recover protocol or one immutable authority
directory plus an atomic pointer.

### 5. The existing composite is not the current coherent teacher corpus

The July 14 composite's authenticated current contract is the earlier
information-set/PIMC operator. The current learner/search contract is a
single coherent public-belief tree. Historical states and terminal outcomes
remain useful, but their policy targets must not silently masquerade as the
current teacher.

Use exact coherent n128 reanalysis for policy-bearing rows, or admit stale rows
only to value/rehearsal scopes.

At audit time the B200 archive contains one production composite, and its
authenticated operator is `information_set_search=true`,
`coherent_public_belief_search` absent, `n_full=128`. There is no current
coherent-public composite available to launch the scratch learner.
Its component metadata also binds the v2 entity adapter, while the current
scratch model requires the v5 meaningful-history adapter. Therefore the old
composite is evidence/state inventory, not a drop-in current learner input.

## Highest-impact research defects

### 1. Boundary belief averaging exists, and the working tree binds K=1

`boundary_value_particles` is implemented, including native averaging. The
working tree now binds K=1 in the science contract, generator config, guard,
and evaluation path, which closes the previous silent-parser-default problem.
K=1 still means one sampled hidden world at actor handoff; K>1 remains
uncommissioned.

Commission K=1/K=2/K=4 at matched evaluator-call cost, then replace the
explicit K=1 authority with the selected value everywhere.

### 2. Learner history is richer than teacher belief

The learner receives redacted meaningful public history, but search
determinizations remain conservation-based and do not condition their
opponent-card belief on that history. This is not hidden-information leakage;
it is a teacher ceiling. The target policy cannot teach history-specific
belief behavior that the teacher does not perform.

Build an actor-centric public-history posterior, use it for boundary
determinizations and development-card materialization, and bind its version
and implementation hash into the target identity.

### 3. The production scalar value objective can stop correcting confident errors

The canonical objective applies MSE after `tanh`. Its gradient includes
`1 - tanh(raw)^2`, which approaches zero for confidently wrong predictions.

An opt-in causal binary-win BCE objective is now implemented using logit
`2 * scale * raw` and target `(z + 1) / 2`; search continues to consume
`tanh(scale * raw)`. It remains uncommissioned until the matched value-objective
campaign measures calibration and playing strength.

### 4. Value validation does not match the search query distribution

Current reporting is dominated by recorded decision-row loss. Search is
load-bearing at actor handoff, pre/post roll, end-turn, and opening roots.

Add a whole-game evaluator-query holdout with:

- bias, RMSE, Pearson and Spearman correlation;
- calibration/ECE;
- phase and turn-boundary strata;
- game-bootstrap confidence intervals;
- fixed-root search uplift versus the parent.

### 5. Shared-representation controls omit the mature action encoder

The action encoder feeds both policy and value, but it is neither in the trunk
LR group nor the action-local LR group. It therefore receives the full base LR
while the contract claims a conservative shared-trunk update.

Create explicit optimizer groups for state/history trunk, shared action
representation, policy-private modules, and value-private modules. Track
functional and parameter drift for every shared group.

### 6. One global gradient clip couples all objectives

A large new-head or auxiliary gradient can globally shrink mature policy and
trunk gradients. Add pre/post-clip group norms and commission fresh heads
head-only or with scoped caps before joint optimization.

### 7. The current from-scratch path bundles too many changes

The canonical scratch arm simultaneously changes initialization, history,
rule/card features, action gather, legal-set value features, and value tower.
It also has no parent-policy trust region.

The near-term main arm should be an independently initialized,
function-preserving upgrade of the exact current parent with fresh Adam,
coherent targets, lower shared-representation LR, and explicit parent KL.
Keep full scratch as a separate architecture arm.

### 8. END_TURN value weight remains an unresolved risky default

`ROLL` is restored to value weight 1.0 but `END_TURN` remains 0.1. Search
delegates future play to value at turn boundaries, so this default may damage
exactly the states where value is most important.

Use 1.0 as the safe baseline until a matched, gradient-mass-controlled ablation
proves 0.1 is stronger.

### 9. Reliability evidence is measured but unused

The generator duplicates a small target-reliability audit slice, but canonical
training does not enable confidence weighting and the current confidence
formula ignores stored Q margins.

Calibrate reliability by phase/operator first. High-surprise,
low-repeatability rows should be reanalysed, downweighted, or value-only.

## Execution plan

### Phase 0 — commission and connect one loop

1. Use the 8×B200 matrix to select the optimizer schedule and authorize it.
2. Connect the current learner mode to the durable iteration state machine.
3. Make teacher-operator adoption recoverable as one transaction.
4. Replace stale live fleet authority with a new owner-attested private
   inventory.

Exit criterion: one non-promotable run completes
`train → checkpoint selection → paired internal eval → promotion dry-run`.

### Phase 1 — commission value and optimizer semantics

Run a compact 8×B200 matched matrix:

- value objective: deployed-tanh MSE vs binary-win BCE;
- END_TURN value weight: 1.0 vs 0.1;
- shared representation: current grouping vs explicit shared-action group;
- initialization: function-preserving parent upgrade vs scratch control.

Select by evaluator-query calibration, parent KL/drift, and paired playing
strength—not training loss alone.

### Phase 2 — commission the teacher

1. K=1/K=2/K=4 boundary-belief probe at matched evaluator calls.
2. Reliability calibration on duplicate-search roots.
3. Reanalyse existing states with the selected exact teacher before requesting
   a large new wave.

### Phase 3 — improve imperfect-information reasoning

Implement the public-history-conditioned actor belief, privileged belief
auxiliary, invariance tests, and target-identity binding. Then compare the
history-aware teacher/student against the Phase-2 baseline.

### Phase 4 — resume the flywheel

Only after Phases 0–2:

1. generate fresh coherent n128 data;
2. audit and build the composite;
3. train the selected parent-preserving arm and scratch control;
4. select across optimizer-step checkpoints;
5. evaluate checkpoint and full-agent improvements separately;
6. promote only on paired, disjoint evidence.
