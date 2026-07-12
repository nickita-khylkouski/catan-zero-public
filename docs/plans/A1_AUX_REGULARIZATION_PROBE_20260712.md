# A1 post-P1 auxiliary regularization probe

Status: sealed diagnostic design. It does not authorize launch or promotion.

## What already exists

The current 35M `entity_graph` implementation already has five optional,
behaviorally disconnected heads:

- horizon longest-road ownership;
- horizon largest-army ownership;
- realized VP change over eight plies;
- the actor's next settlement node;
- the actor's next robber hex.

They read the pooled public-state trunk. They never feed the policy or value
readouts. `f69_upgrade_checkpoint_config.py --flags aux` adds them to an old
checkpoint and asserts that its policy/value/Q outputs remain identical. The
learner masks missing labels per head and fails loudly if a nonzero auxiliary
weight is requested from a corpus with no auxiliary columns.

The feature is not merely implemented: the present n128 and n256 memmaps
already contain all five label columns. A direct B200 inventory measured:

| corpus | rows | road/army/VP finite | settlement valid | robber valid |
|---|---:|---:|---:|---:|
| n128 | 31,919,276 | 100% | 70.095% | 89.878% |
| n256 | 12,773,247 | 100% | 70.096% | 89.874% |

Road ownership is positive on 19.78%/19.86% of rows and army ownership on
15.46%/15.68%, so the binary heads are not degenerate. VP-in-eight is nonzero
on about 21.7% of rows. The negative VP changes are legal: an actor can lose
Longest Road and/or Largest Army inside the horizon.

No completed B200 report was found with a nonzero
`aux_subgoal_loss_weight`. This is an untested built-and-banked lever, not a
request for new generation or a broad architecture rewrite.

One observability limitation remains: `train_bc` reports the summed auxiliary
loss and active-head count, not five separate validation losses. The sentinel
can still be adjudicated from strength, main-task metrics, combined auxiliary
loss, and the measured per-head coverage above. Do not modify the active P1
learner merely to add optional telemetry; add a post-hoc per-head scorer before
the two-seed confirmation if AUX2 survives.

## The minimal experiment

Run only after P1 has selected the anti-forgetting recipe. Both arms inherit
that exact authenticated P1 parent, mixed n128+n256 data recipe, game-disjoint
validation set, sample dose, and evaluation cohorts.

| arm | change | samples |
|---|---|---:|
| AUX0 | no auxiliary heads/loss | 4,194,304 |
| AUX2 | add all five heads; summed auxiliary loss weight 0.02 | 4,194,304 |

`0.02` is deliberately the first dose. `_aux_subgoal_loss` sums five separately
normalized head losses; it does not average them. At random initialization the
two categorical cross-entropies alone can be near `log(54)+log(19)`, so a
nominal `0.05` is not a mild first intervention in this implementation.

The checkpoint upgrade adds dropout-bearing auxiliary readouts, so this is a
small architecture-plus-objective ablation, not a claim of bitwise-identical
training trajectories. The required invariant is narrower and testable: main
outputs are identical immediately after upgrade and the heads never feed them.
The matched sample dose and common evaluation cohorts supply the causal test.

## Admission and decision

Before execution, bind the P1 winner receipt and recipe hash, verify all five
columns against the coverage floors emitted by
`tools/a1_aux_regularization_plan.py`, and require zero main-output difference
from the checkpoint upgrader. Do not bundle root-value blending, Q loss,
forced-row changes, replay-ratio changes, or a trunk architecture change.

External population non-regression is binding. AUX2 advances only if it also
improves internal playing strength or a predeclared calibration/active-teacher
metric. Lower validation loss alone is insufficient. If it passes, repeat the
two-arm comparison with two independent auxiliary-module seeds before making
the heads a production default.
