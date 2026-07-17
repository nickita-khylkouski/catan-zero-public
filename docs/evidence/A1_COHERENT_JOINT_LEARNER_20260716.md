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
9. DDP symmetry augmentation gave every rank the same 64-draw stream. An
   eight-rank update therefore replayed each sampled transform eight times
   instead of consuming one globally partitioned stream. Commit `33bf402`
   makes ranks consume `rank::world_size` positions from one deterministic
   global symmetry stream and preserves that stream across resume.

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

The first offline pass identified step 48 as a useful balanced frontier in
both architectures. The later matched-runtime dose comparison below supersedes
48 as the selected dose.

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
| split FULL step 48 replication 1 | 128 | 69-59 | 53.91% |
| split FULL step 48 replication 2 | 128 | 70-58 | 54.69% |
| **strictly pooled split FULL step 48** | **384** | **221-163** | **57.55%** |

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

Two disjoint replication cohorts also favored the candidate. The repository's
strict evaluation pool authenticated identical checkpoint bytes, exact f7,
matching effective search, non-overlapping seed intervals, and all 192 complete
pairs. The pooled artifact is
`/tmp/a1-coherent-joint-split1-8arm-20260716/h2h/split1_full48_vs_f7_pooled192pairs.json`
with SHA
`sha256:67d8c2a3296a0fea78c30f01f21b41226369b838d01b7f650e11c1c28b104a81`.
Its production pentanomial verdict is H1 (`LLR=4.5494`); the stricter
superiority pentanomial test remains `continue` (`LLR=2.4533`).

## Canonical eight-rank commissioning correction

The first canonical 8-GPU replay produced candidate
`sha256:1fcda56cfcc7a194fb14155af1dc4b33db2a797832e2ae69444f1772765d163c`
and scored 163-93 over 128 paired seeds (63.67%). Its pooled report has SHA
`sha256:60279fe8527ab5be15e82f60ba8f237358da2ae07da51698c1d975f7643c8e2d`;
both the production pentanomial test (`LLR=6.1754`) and strict superiority
test (`LLR=3.5013`) crossed H1.

That run is **not valid commissioning evidence**. It predates `33bf402`, so
all eight ranks replayed the same symmetry augmentation draws. The gameplay
result is an authentic measurement of that trained checkpoint, but it cannot
commission the intended globally partitioned DDP training recipe.

The corrected DDP replay produced candidate
`sha256:65f3c1e7a7604633e5bc0adab615d2bfdeb4dfdac876e7a87795cbbc60e75deb`.
Its first disjoint 64-pair cohort finished 69-59 (53.91%) with all tests still
`continue`; the report SHA is
`sha256:b9103a008a6fbf19b5a4f867833815313b8dd0cd662a4384e06d2bc461d4fa68`.
This is positive but inconclusive. It supersedes the pre-fix result for recipe
commissioning and does not establish a promotion.

## Direct parent-upgrade lineage commissioning

The canonical architecture transition is now one explicit, allowlisted
f7-to-current-v5-plus-split1 upgrade rather than two implicit initializer
stages. With seed 1, the direct initializer is tensor-identical to the old
two-stage initializer for all 175 model tensors. The direct initializer SHA is
`sha256:9dd934c90f02e6460bc71e491fd5efab1cb19c5ca56c358f05ac98f309e4b5d3`
and its bound upgrade receipt SHA is
`sha256:1462e19057d7a9593d534ccb09df1fa1aeb3991ee20b7c43e7ccd78abb9ba456`.

An eight-GPU replay started from those exact bytes, used fresh Adam, and
completed 48 optimizer steps / 24,576 global row draws. It produced candidate
`sha256:b896811535f41d75f78e89c97f845e750722be00633b647ac545a529680d1ddc`
under entity-model source SHA
`sha256:b4e2618bc36296470f13ce3dee228b34fd7d117c0211380c46393450793ce975`.
The training report records the parent SHA, initializer SHA, and upgrade
receipt as one lineage-bound initialization contract.

A partial H2H launched from the older evaluator checkout was discarded before
interpretation: that runtime carried entity-model source SHA
`sha256:70be73d688372fa857799e1ca61a3c3e00a8b70a263b17f0dba03d1626f90906`,
not the candidate's current source. No result from that partial panel is
promotion or commissioning evidence.

The replacement panel used a clean checkout at
`4c4322ba1ae1c525bcec3c9c17f8e6455871a173`, exact matching entity-model SHA
`sha256:b4e2618bc36296470f13ce3dee228b34fd7d117c0211380c46393450793ce975`,
and native runtime SHA
`sha256:461b9d1637e7027ff59b3ec781fec9bd4b1ad51331693c75951e8f44f4f9014c`.
It completed two disjoint cohorts with zero truncations:

| Cohort | games | candidate-baseline | win rate | report SHA |
|---|---:|---:|---:|---|
| seed 102 | 128 | 76-52 | 59.375% | `sha256:3722597d22abb344c3eaba014f8ae5c48bdd258efe0220500d1551646e51b84d` |
| seed 103 | 128 | 65-63 | 50.781% | `sha256:7318448f9209f90b445cc9b687909897689daa75a96108815d8695a4399ae94c` |
| **strict pool** | **256** | **141-115** | **55.078%** | `sha256:fbbba4598b4451cb230035da2b63a6f32bf37adbffbbece12b2fb9e0e2194ccc` |

The seed-102 production pentanomial test remained `continue`
(`LLR=1.719`), as did seed 103. The strict 128-pair pool also remained
`continue`: production pentanomial `LLR=2.025` and superiority pentanomial
`LLR=1.027`. This is positive current-runtime commissioning evidence for the
lineage-bound treatment, not a promotion result.

The `b896...` standalone training report records `promotion_eligible=false`
because it predates the sealed one-dose transaction. The next canonical
candidate must be emitted by the `15d548f` (or later) sealed path, which binds
the parent update, execution receipt, evaluation, and promotion transaction;
this R&D checkpoint must not be promoted retroactively.

## Superseding minimal-dose selection

The exact current-runtime step-12 checkpoint is
`sha256:92a7df02d2f99a5cad993d667a78776256d7cb215ad7809a4c0739dfbf95d392`.
It completed two new disjoint, paired cohorts with zero truncations and zero
errors:

| Cohort | games | candidate-baseline | win rate | report SHA |
|---|---:|---:|---:|---|
| seed 104 | 128 | 72-56 | 56.250% | `sha256:990b823d8ee99e8cb55b4c47a7a6cc65320b579915eee3f2ca6002b75e47a824` |
| seed 105 | 128 | 69-59 | 53.906% | `sha256:32b00ab8b6557df05ede3cedeb55668e0951499951507f32e43a247c8d7fdf12` |
| **strict pool** | **256** | **141-115** | **55.078125%** | `sha256:bd9ada544583e8a4506693ded01fb5dbc230e806d80ce631b1f14421b2e4d404` |

The step-12 pooled score exactly matches the step-48 pool, while making a
materially smaller update from the same parent:

| Dose | parent forward KL | global relative L2 drift |
|---|---:|---:|
| **step 12** | **0.0332787** | **0.00882354** |
| step 48 | 0.0692853 | 0.0133801 |

The teacher-gap artifact covering the comparison has SHA
`sha256:6952d251b7bd49e492294379c62a4e32a08373244e79005ea74874c6396cfd45`.
At equal observed pooled playing score, step 12 uses about 48% of the
parent-policy KL and 66% of the global parameter drift. Step 12 is therefore
the selected **minimal effective dose**; step 48 is retained as overdose-side
evidence, not the canonical horizon.

These checkpoints remain diagnostics, not promotion artifacts. The step-12
selection must be reproduced by the sealed `15d548f` (or later) one-dose path
before any promotion transaction.

## Runtime performance repairs

Two implementation changes reduce work without changing the learner recipe:

- `c006f24` uses a CLS-query-only final private value block during evaluation
  when `value_tower_split_layers=1` and value attention pooling is disabled.
  It keeps every key/value token and the exact padding mask; training, deeper
  private towers, and attention-pool readouts retain the full-token path.
- `11dcdc0` scans Rust public history backward only until the bounded retained
  suffix is filled, then restores chronological order. This avoids rebuilding
  and scanning an entire long-game history at every MCTS leaf while preserving
  the existing 32/64-event feature contract.

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
- The selected research treatment is the split-value FULL arm at exactly 12
  updates, with fresh optimizer state and no candidate chaining. It matches
  the step-48 pooled score with substantially lower parent KL and parameter
  drift. Its valid current-runtime pool is positive but statistically
  inconclusive, so it commissions the minimal dose for a sealed replay rather
  than promoting the R&D checkpoint. The separate fresh-scratch recipe is not
  modified by this warm-start result.

The earlier step-48 checkpoint and evidence bundle remain staged at
`/home/ubuntu/catan-zero-artifacts/rl-system-repair-20260716/split1-full48`.
That historical checkpoint SHA is
`sha256:bd8adcba9bcb9f6d9d25ce4b05e545d6b1134354758fefd30bbf49cab4ab68be`.
The superseding diagnostic checkpoint is
`/tmp/a1-parent-update-canonical-direct-bootstrap-8gpu-20260716/candidate_step0012.pt`
with SHA
`sha256:92a7df02d2f99a5cad993d667a78776256d7cb215ad7809a4c0739dfbf95d392`.
