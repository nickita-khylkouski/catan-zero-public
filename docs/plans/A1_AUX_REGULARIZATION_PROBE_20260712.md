# A1 corrected pointer-auxiliary commissioning experiment

Status: non-launching preregistration. The AUX0/AUXT selection experiment is
diagnostic-only and cannot authorize promotion. A separately issued FINAL
replication may become promotion-eligible only after the fresh conjunctive
dual-baseline full gate replays; neither this document nor the plan emitter
authorizes launch, deployment, or automatic promotion.

## Why the historical AUX2 result is invalid

The historical settlement auxiliary predicted one of 54 absolute vertex IDs
from the pooled CLS state. The entity trunk is permutation invariant over
vertices that do not carry an absolute identity feature, so two relabelings can
produce the same CLS readout while requiring different settlement labels. That
is target aliasing, not a weak hyperparameter choice. The fixed `0.02` weight,
4,194,304-row dose, and candidate-lineage initialization therefore have no
standing as evidence.

The corrected settlement target is a pointer score over the 54 **post-trunk
vertex tokens**. A vertex relabeling relabels the logits and target together.
The other four heads remain training-only public-input readouts. None feeds the
policy or value outputs.

## Immutable parent and P1 boundary

The immutable causal parent remains the exact recovered current promoted v5
handoff parent:

`sha256:6817ab054506f962a758ebf48addce5cc7eb801bf451cf2d02b62fb91f5da39c`

The handoff/recovery receipt, not this prose or a caller-supplied hash, is the
authority. Old f7 bytes are a safety baseline, not a legal rollback
initializer. A completed K0/K3/K10 checkpoint is an unpromoted diagnostic
candidate and may not initialize another arm. P1 contributes only the
centrally selected learner recipe and typed 64/12/4/20 data authority.

That legacy parent has public-award feature contract `legacy_zero_v0`. Before
pointer construction, an immutable zero-optimizer transition writes an exact
zero into input column 12 and declares `authoritative_v1`. Its receipt must
prove: source bytes are the raw parent, target bytes differ only in the
authorized column-12 initializer/metadata, optimizer steps are zero, unrelated
tensors are bit-identical, and legacy-zero inputs preserve the old function.
The transitioned checkpoint is an initializer transform, not a candidate or a
new promotion parent.

The exact causal initializer chain is therefore:

`raw promoted parent → public-award zero transition → pointer upgrade → head-only warmup`

The pointer upgrade must source the transitioned checkpoint, never the raw
parent and never a diagnostic candidate. Its immutable receipt must prove all
inherited tensors are bit-identical and all main outputs have maximum absolute
difference exactly `0.0`.

## Commissioning sequence

1. Replay the immutable raw-parent-to-public-award transition described above,
   then make one deterministic function-preserving pointer-head upgrade from
   those exact transitioned bytes.
2. Run exactly 128 FP32 optimizer steps of **head-only** warmup on 8 B200 ranks,
   local batch 512, accumulation 1. Only the five pointer-auxiliary parameter
   prefixes may change. Every inherited tensor must remain bit-identical. The
   warmup optimizer sidecar is discarded.
3. From the one shared warmed checkpoint, run five ordered 512-row FP32
   same-forward probes. For each batch, obtain the main-objective and unit-AUX
   gradients over the exact inherited trainable trunk with `autograd.grad`.
   Manually aggregate DDP gradients because `autograd.grad` does not invoke the
   DDP reducer. Apply no backward call, optimizer step, checkpoint write, or
   persistent state mutation.
4. Seal, for every ordered batch, the shared-parameter surface digest and the
   additive sufficient statistics `||g_main||^2`, `||g_aux||^2`, and
   `dot(g_main,g_aux)`. The coordinator sums them across batches and derives
   both norms and cosine. It never accepts a caller-selected cosine.
5. Select the treatment coefficient mechanically: cap the AUX/main norm ratio
   at `0.05`, cap opposing projection at `0.01`, cap the coefficient at `0.05`,
   floor to a `0.001` quantum, and refuse zero/non-finite/inconsistent evidence.
6. Execute one matched diagnostic pair from the same warmed bytes: `AUX0` at `0.0` and
   `AUXT` at the selected coefficient. Each arm receives exactly 524,288 sampler
   draws (128 optimizer steps at global batch 4096), fresh Adam, FP32, and the
   selected P1 recipe. Auxiliary weight is the only scientific delta.

Every P1, WARMUP, GEOMETRY, AUX, and FINAL execution is centrally claimed once under the exact sole 8-B200
allocation. WARMUP and GEOMETRY then publish immutable executor authorities and
seal a post-authority commitment to the exact rendered argv, complete
allowlisted environment, fresh output namespace, and executor-authority
file/semantic digests before the child starts. AUX0, AUXT, and FINAL likewise
require their centrally published executor authorities and a post-authority
one-dose commitment to the exact command, allowlisted environment, output
namespace, and authority bytes before execution. At execution time the live hostname, machine
identity, physical-index-to-UUID mapping, and physical-index-to-PCI mapping
must match authority, and the executor must hold all eight physical GPU locks.
Pre-existing, aliased, or symlink outputs fail closed. Crash recovery may
terminalize already-authenticated outputs but may never silently rerun a dose.

The admitted native runtime, code/runtime identities, data inventory, sampler
order, current parent, public-award transition, pointer upgrade, warmup
terminal, geometry terminal, execution commitments, and arm output bytes are
content-bound. Operator-chosen `AUX2`, `0.02`, retry labels, candidate chaining,
and generic ablation overrides fail closed.

## Correct objective semantics

The auxiliary objective is the sum of five conditional head means. Missing
future-event labels are masked per head and means use global DDP denominators;
rank-local masked means are forbidden. The settlement pointer is topology
equivariant. Auxiliary readouts contain no head-local dropout, so merely
evaluating AUXT cannot advance the process-global dropout stream relative to
AUX0; stochasticity still comes from the shared training trunk.

The geometry main objective is the exact selected P1 training objective with
the auxiliary term excluded. The unit-AUX objective is the unscaled sum above,
formed from the same forward graph. Geometry is measured over inherited trunk
parameters only; policy/value/action/auxiliary readouts are excluded.

## Fixed diagnostic evaluation

After both arms terminalize, the coordinator must claim and complete the one
fixed evaluation; an operator may not pick a favorable panel after seeing
results. Both panels use common random numbers, seat swaps, the exact raw
recovered parent baseline, and the fixed native Rust information-set search
operator (`n_full=128`, four particles, D6 root averaging at legal width at
least 20, no selection tuning).

- Internal: 300 pairs/600 games per arm on `BASE` against the recovered
  generator reference. AUXT must score strictly more pair points than AUX0.
- External: 250 pairs/500 games per arm on `TOURNAMENT` against
  `catanatron_value`. AUXT's mean may not fall more than 0.025 points/game below
  AUX0.

The coordinator replays raw-game receipts for both panels. Validation loss or
auxiliary predictability is not decision evidence. If both conditions pass,
the diagnostic decision is AUXT; otherwise it is AUX0. Neither arm checkpoint
is promotion-eligible.

## Independent FINAL replication and full gate

Only after the fixed evaluation and pair terminal may the coordinator issue
FINAL. FINAL independently reloads the raw causal lineage and uses a fresh
component-routing receipt plus a sampler seed, order, and physical row set that
differ from P1. It never initializes from AUX0, AUXT, or any P1 diagnostic
checkpoint:

- If AUX0 won, FINAL starts from the exact public-award transitioned checkpoint
  and performs no pointer upgrade or reference warmup.
- If AUXT won, FINAL starts from the exact immutable reference warmed pointer
  checkpoint. The warmed bytes are architecture commissioning evidence, not an
  AUXT candidate checkpoint.

FINAL applies only the selected recipe for one fresh 524,288-row, 128-step FP32
8×512 dose with fresh Adam. Its initializer must have exact-zero public-award
slot-12 parameters; its candidate must contain finite, nonzero learned slot-12
signal. Completing FINAL makes it eligible to enter a gate, not eligible for
automatic promotion.

The gate is fresh and conjunctive: the ordinary promotion adjudication must
prove strict H1 superiority over the recovered current parent, and an
independent fixed 300-pair n128 cohort must not hit H0 against the authenticated
f7 safety reference. The ordinary training-provenance, calibration, external,
high-regret, and bucket gates still apply. Only replay of that immutable
dual-baseline gate authority can mark the FINAL candidate promotion-eligible;
promotion remains an explicit later transaction.

## Authoritative lifecycle

1. Prepare the P1 K0/K3/K10 sweep. Sequentially claim → publish executor
   authority → post-authority commit → execute → complete each arm, then claim
   and complete its fixed internal/external evaluation and mechanically
   adjudicate the selected P1 recipe/data authority.
2. `prepare_experiment` from that selected P1 authority
3. claim → publish executor authority → post-authority commit → execute →
   complete WARMUP
4. claim → publish executor authority → post-authority commit → execute →
   complete GEOMETRY
5. issue pair; claim → publish → commit → execute → complete AUX0, then AUXT
6. claim and complete the fixed internal/external pair evaluation; finalize the
   diagnostic decision
7. issue → claim → publish → commit → execute → complete independent FINAL
8. verify the fresh recovery dual-baseline full gate and load the FINAL gate
   entry authority
