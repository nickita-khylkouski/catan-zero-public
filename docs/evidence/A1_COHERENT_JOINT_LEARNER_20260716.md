# Coherent-public joint learner commissioning

Date: 2026-07-16

## Question

Can the exact f7 incumbent learn a stronger policy from a small corpus produced
by the deployed coherent-public n128 operator, and does a private final value
tower make continued policy/value learning safer?

This is an R&D commissioning run, not a production promotion.

## Authenticated corpus

The corpus at `/tmp/a1-coherent-n128-rd-512-feaf8bc` on the assigned
8×H100 host contains:

- 512 completed games and 123,811 decision rows;
- 27,298 policy-active rows;
- 63,748 forced or automatic rows with zero policy weight and full value
  weight;
- 4,070,857 MCTS simulations;
- 99.59% meaningful public event history;
- zero failures, truncations, invalid actions, or invalid active targets.

The producer is exact f7
`sha256:f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4`.
The accepted coherent-public teacher identity is
`sha256:d1f6686a2f00012aa54a729f4850e1333d59e57783d323c6a2d2d2a15ab02fed`.
All learner arms use the same 16,407-row, seed-selected holdout.

## Binding defects repaired before interpreting results

The commissioning work repaired several root-level failures:

1. Direct policy learning crashed at the first intermediate checkpoint because
   base-policy denominator sufficient statistics were recorded only when the
   optional policy-AUX sampler was enabled.
2. Single-corpus diagnostics incorrectly selected a composite-only coverage
   sampler.
3. Teacher-gap probes admitted two competing holdout authorities.
4. Parent comparison did not reconstruct the report-authenticated
   legacy-to-authoritative public-award transition.
5. The first campaign silently instantiated `value_tower_split_layers=0`.
   A function-preserving split initializer now source-clones the final shared
   block and normalization into a private value tower with exact step-zero
   output parity.
6. Canonical evaluation lacked a host CPU-placement control, preventing safe
   concurrent panels.
7. Neural prior temperature could be applied in both the evaluator and search;
   ownership is now explicit and the executed effective temperature is bound
   into resume identity.
8. The minimum effective policy rows per global batch affected admission but
   was absent from typed training identity. It is now a hashed `TrainConfig`
   field.

## Campaigns

Both campaigns contain eight independent arms. Every arm reloads the same
initializer bytes with fresh Adam; no candidate chaining occurs. Each arm
trains for 192 optimizer updates with checkpoints at
8, 12, 16, 24, 32, 48, 64, 96, 128, and 160.

The first campaign used the function-preserving current-feature f7 initializer
with a shared value path (36,783,433 parameters). The corrected campaign used:

- initializer:
  `/tmp/a1-coherent-joint-8arm-20260716/f7.current-v5-split1.pt`;
- initializer SHA:
  `sha256:bad737bb9299a1e4f61a6fef03bd189bea1f09da44acbb35505f48f9fdb18486`;
- `value_tower_split_layers=1`;
- 41,708,233 parameters;
- exact step-zero output parity with its source.

The corrected split campaign is at
`/tmp/a1-coherent-joint-split1-8arm-20260716`.

## Offline result

Step 48 reproduced as the useful balanced frontier in both architectures.

| Architecture / checkpoint | teacher-gap closure | scalar value MSE delta | parent-policy KL |
|---|---:|---:|---:|
| shared FULL step 48 | +4.5192% | -0.01050 | 0.07004 |
| split FULL step 48 | +4.8202% | -0.00392 | 0.07431 |
| split D64/TRUST step 48 | +4.4290% | -0.00444 | 0.04297 |
| split D128/FREEZE step 128 | +3.0868% | -0.02505 | 0.08944 |
| split FULL step 160 | +5.4736% | +0.02728 | 0.09001 |

The step-160 result is an over-dose warning: teacher imitation continues to
improve while value quality regresses. Offline value-only improvement is also
not a playing-strength proxy; the shared D128/FREEZE step-96 checkpoint lost
its gameplay screen despite its strong value MSE.

## Matched coherent-n128 gameplay screens

All games use paired seeds, seat swaps, exact f7 as baseline, public
observation, the Rust native hot loop, n128 for both roles, `c_scale=0.1`,
D6 averaging at width 20, and zero truncations.

| Candidate | games | candidate-baseline | win rate |
|---|---:|---:|---:|
| shared FULL step 12 | 32 | 24-8 | 75.00% |
| shared FULL step 48 | 64 | 34-30 | 53.13% |
| shared FULL step 160 | 32 | 21-11 | 65.63% |
| shared D64/TRUST step 48 | 32 | 14-18 | 43.75% |
| shared D128/FREEZE step 96 | 32 | 10-22 | 31.25% |
| split FULL step 12 | 32 | 18-14 | 56.25% |
| split FULL step 48 | 32 | 19-13 | 59.38% |
| split D128/FREEZE step 128 | 32 | 19-13 | 59.38% |
| **split FULL step 48 confirmation** | **128** | **82-46** | **64.06%** |
| split FULL step 48 replication | 128 | 69-59 | 53.91% |
| **strictly pooled split FULL step 48** | **256** | **151-105** | **58.98%** |

The screens establish a real learning signal and reject the claim that the
coherent n128 corpus is intrinsically unlearnable. They are not by themselves
promotion evidence.

The independent 64-pair confirmation is stored at
`/tmp/a1-coherent-joint-split1-8arm-20260716/h2h/split1_full48_vs_f7_64pairs.json`.
It completed all 64 pairs with zero truncations. Its production
`[-10,+15]` pentanomial SPRT crossed the H1 boundary:
`LLR=3.0708 > 2.9444`. The stricter `[0,+15]` superiority test remains
`continue`; this result commissions the learner recipe but does not itself
perform a champion promotion.

The second disjoint 64-pair cohort also favored the candidate. The repository's
strict evaluation pool authenticated identical checkpoint bytes, exact f7,
matching effective search, non-overlapping seed intervals, and all 128 complete
pairs. The pooled artifact is
`/tmp/a1-coherent-joint-split1-8arm-20260716/h2h/split1_full48_vs_f7_pooled128pairs.json`
with SHA
`sha256:6fc12764eb3ab45e7c4f88caefd4db58d014ed63788bc294952eaacb766ecd9b`.
Its production pentanomial verdict is H1 (`LLR=3.3860`); the stricter
superiority test remains `continue` (`LLR=1.8591`).

## Current interpretation

- Fresh coherent n128 policy targets produce playable improvement from f7.
- The useful warm-start update window begins much earlier than the old
  128-step convention; step count alone is not a portable dose identity.
- A private value tower makes post-policy value continuation viable, but does
  not remove the need to cap total update size.
- Parent trust based on stored-prior KL did not control authoritative
  functional drift and is not commissioned.
- Long unrestricted training can improve teacher KL while damaging value
  quality. Selection must combine target closure, value quality, functional
  drift, and matched gameplay.
- The commissioned parent-update treatment is the split-value FULL arm at
  exactly 48 updates, with fresh optimizer state and no candidate chaining.
  The separate fresh-scratch recipe is not modified by this warm-start result.

The selected checkpoint and evidence bundle are durably staged at
`/home/ubuntu/catan-zero-artifacts/rl-system-repair-20260716/split1-full48`.
The checkpoint SHA is
`sha256:bd8adcba9bcb9f6d9d25ce4b05e545d6b1134354758fefd30bbf49cab4ab68be`.
