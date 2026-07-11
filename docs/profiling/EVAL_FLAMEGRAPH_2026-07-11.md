# Evaluation flamegraph baseline — 2026-07-11

This is the optimization baseline for the production A1 n128 evaluator.  It
profiles the two canonical entry points without changing their search recipe:

- internal searched cross-network H2H:
  `tools/gumbel_search_cross_net_h2h.py`;
- external neutral panel, with Python Catanatron as authoritative referee:
  `tools/catanatron_neutral_harness_match.py --mode search`.

The captures use commit `c807874940fd5b3e4c51775f33a64279786504da` and the
real A1 and gen-3 checkpoints.  They preserve n128, D6 at width 20,
information-set search P4/min32, public observations, corrected Rust chance
spectra, lazy interior chance, scalar value readout, and the flywheel gate.
They are one-pair profiling canaries, not strength evidence.

## Results

There are two complementary views:

1. The SVG flamegraphs sample active CPU stacks.  Their percentages are
   **inclusive**: a parent such as `_simulate` includes its feature, FFI, and
   neural children and must not be added to those children.
2. The raw `--idle --threads` capture is classified by the deepest relevant
   stack into mutually exclusive wall-sample buckets.  The deliberately idle
   evaluator queue thread is removed from the denominator.  These percentages
   can be added.

### Mutually exclusive main-thread wall samples

| Bucket | Internal cross-net | External neutral |
|---|---:|---:|
| Python MCTS traversal/bookkeeping | 34.31% | 31.72% |
| Feature encoding + leaf FFI | 34.57% | 30.00% |
| Waiting for async neural evaluator | 20.39% | 19.40% |
| Active neural forward / CUDA dispatch | 7.96% | 10.69% |
| Python↔Rust referee synchronization | — | 5.17% |
| External `catanatron_value` policy | — | 1.29% |
| Python Catanatron referee outside bot/search | — | 0.09% |
| Other Rust FFI/runtime | 2.77% | 1.64% |

Internal has 2,349 main-thread samples.  External has 1,160.  The async queue
threads contributed 4,551 internal and 1,096 external idle samples; those are
thread lifetime, not latency on the game thread, and are excluded above.

The earlier informal “77% Python traversal” estimate is not supported by this
exact production-recipe capture.  Python is still the dominant opportunity,
but it is split almost evenly between traversal and feature construction.
Active forward work is only a few percent of the active-CPU flame; the game
thread nevertheless spends about 20% of wall samples waiting for the async
evaluator.  Native sampling sees CUDA dispatch/waiting, not GPU kernel
occupancy, so those two statements are not contradictory.

### Inclusive active-CPU hotspots

| Function | Internal | External | Meaning |
|---|---:|---:|---|
| `_simulate` | 29.54% | 30.01% | recursive Python tree traversal |
| `_traverse_single_sample` | 24.57% | 24.53% | action/chance child traversal |
| `_traverse_robber_or_dev` | 7.01% | 7.31% | stochastic materialization path |
| `_spectrum` | 4.90% | 5.35% | chance-spectrum FFI/bookkeeping |
| `_evaluate_many_checked` | 3.92% | 4.63% | batched leaf boundary |
| `_enumerate_roll_outcomes` | 3.92% | 4.63% | roll children and evaluation |
| `_expand` | 3.20% | 5.28% | Python node expansion |
| `EntityGraphRustEvaluator.evaluate*` | 2.15–2.37% | 2.22–3.39% | active evaluator CPU |
| `forward_legal_np` | 1.96% | 2.02% | active NN dispatch/forward CPU |
| `rust_game_to_entity_batch` | 1.70% | 2.94% | Python feature path |
| `sync_from_native` | — | 2.80% | neutral-harness shadow sync |
| `apply_native_action_record_to_rust` | — | 2.41% | authoritative-action replay |

These rows overlap by design.  For example, `_simulate` contains
`_traverse_single_sample`, which can contain evaluator and feature work.

### Rust-featurizer A/B on a real checkpoint

`tools/perf_snapshot.py leaf` evaluated the same 200 deterministic leaf states
on one otherwise-idle B200 GPU.

| p50 stage | Python feature path | Existing Rust path | Change |
|---|---:|---:|---:|
| total leaf evaluation | 6.354 ms | 4.500 ms | 1.41x faster |
| entity feature construction | 0.785 ms | 0.0969 ms | 8.11x faster |
| snapshot fetch | 0.993 ms | 0.300 ms | 3.31x faster |
| neural forward | 3.779 ms | 3.726 ms | unchanged |

At p95, total leaf time fell from 6.779 ms to 4.711 ms (1.44x).  Means include
CUDA/checkpoint warm-up outliers and are not used for the steady-state claim.

### Neutral-harness wall counters

The wall capture completed both orientations in 57.72 seconds on one B200:
5,376 simulations, 256 authoritative decisions, 42 searched decisions, zero
divergences, and zero errors.  This is about 93 simulations/second for this
short cohort.  A second exact cohort took 96.55 seconds for 7,936 simulations.
The large variance is game-length/search-position variance, not a stall.

The external bridge is therefore not the principal bottleneck: sync, bot, and
referee-only samples total about 6.55%.  The same MCTS/feature/evaluator split
appears in both harnesses.  Replacing Python Catanatron alone cannot produce a
multi-x speedup.

## Native hot-loop follow-up

The feature-compatible Rust hot loop was subsequently wired behind
`--native-mcts-hot-loop`, with the existing Python implementation retained as
the default/fallback-free control.  The production recipe above was preserved,
including P4/min32 information-set search, public observations, D6-at-20,
corrected/lazy chance handling, scalar/tanh value semantics, and the native
featurizer.  A fresh cp311 wheel passed 8 Rust semantic tests, strict Clippy,
and 18 non-skipped Python binding/information-set tests.

On the matched 12-decision neutral cohort, both implementations completed the
same 1,152 simulations, 9 searched decisions, and 24 authoritative decisions
with zero errors or engine divergences.  Elapsed time fell from 5.905s to
4.676s (1.263x); game time fell from 5.386s to 4.134s (1.303x).  Internal H2H
trajectories differ, so that comparison is normalized by reported work:
201.86 simulations/s for Python versus 279.06 simulations/s after native clone
reduction (1.383x).  These are performance canaries, not Elo evidence.

The corrected post-native wall classifier no longer labels the surviving
`_search_information_set` wrapper as Python traversal.  In the long internal
capture, the non-idle buckets are now led by feature/leaf FFI (44.15%) and
native traversal/allocation (26.62%); Python search orchestration is 3.73% and
PIMC orchestration is 0.75%.  Recursive Python `_simulate`/`_traverse` frames
are absent.  The next optimization ceiling is therefore the evaluator/feature
boundary and neural batching, not another rewrite of the tree policy.

The first native active profile exposed `malloc` at 29.64% inclusive and
`Vec::clone` at 11.08%.  Moving chance-outcome `Game` values directly into
arena nodes and borrowing evaluator requests removed `Vec::clone` from the
top-40 post-change profile; `malloc` measured 15.0% in the shorter confirmation
capture.  Because capture lengths differ, the defensible clone-change claim is
the matched +1.4% internal normalized throughput and +0.8% neutral elapsed
improvement.

Tracked visual evidence:

- [native internal active flamegraph](assets/2026-07-11/native-internal-n128.svg)
- [native external active flamegraph](assets/2026-07-11/native-external-n128.svg)
- [post-clone internal active flamegraph](assets/2026-07-11/native-internal-postclone.svg)
- [native internal wall analysis](assets/2026-07-11/native-internal-wall-analysis.json)
- [native external wall analysis](assets/2026-07-11/native-external-wall-analysis.json)
- [post-clone internal wall analysis](assets/2026-07-11/native-internal-postclone-wall-analysis.json)
- [complete B200 evidence receipt](../evidence/NATIVE_MCTS_B200_20260711.json)

## Optimization implications

- Moving only traversal to Rust has an Amdahl ceiling near 1.52x internally.
- Moving only the remaining feature/leaf boundary has a similar 1.53x ceiling.
- Removing both measured Python buckets has a 3.22x internal ceiling before
  neural evaluation becomes dominant.  The corresponding external ceiling is
  2.61x; removing bridge sync as well raises it to about 3.0x.
- The existing Rust featurizer already realizes a measured 1.41–1.44x leaf
  gain.  Production evaluation should expose it only after exact output/parity
  tests prove it preserves every feature and action-row invariant.
- After the traversal/feature port, optimize neural batching rather than raw
  GPU count.  A one-worker canary uses only a small fraction of a B200; worker
  packing is what turns that headroom into throughput.

## Reproduction

Both harnesses used this common science suffix:

```bash
COMMON=(
  --n-full 128 --c-scale 0.03 --c-visit 50.0
  --sigma-eval 0.98 --rescale-noise-floor-c 0.0
  --lazy-interior-chance --correct-rust-chance-spectra
  --public-observation --information-set-search
  --no-belief-chance-spectra
  --determinization-particles 4 --determinization-min-simulations 32
  --symmetry-averaged-eval --symmetry-averaged-eval-threshold 20
  --value-readout scalar --value-squash tanh
  --max-depth 80 --max-decisions 600
  --max-root-candidates 16 --max-root-candidates-wide 54
  --wide-candidates-threshold 24 --gate-config flywheel
)
```

Active-CPU flamegraph command, with `HARNESS` and its checkpoint arguments
replaced by one of the two canonical invocations below:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/src:$PWD/tools" \
  py-spy record --duration 180 --rate 20 --subprocesses --native \
  --full-filenames -o profile.svg -- \
  .venv/bin/python "$HARNESS" ... --pairs 1 --workers 1 \
  --threads-per-worker 1 --device cuda "${COMMON[@]}"
```

The mutually exclusive wall capture adds
`--idle --threads --format raw`.  Analyze it with:

```bash
python tools/profiling/analyze_pyspy_raw.py profile.raw --out analysis.json
```

Canonical internal argv prefix:

```bash
.venv/bin/python tools/gumbel_search_cross_net_h2h.py \
  --candidate "$A1" --baseline "$GEN3" --pairs 1 --base-seed 6199000201 \
  --workers 1 --threads-per-worker 1 --device cuda "${COMMON[@]}" \
  --out internal.json
```

Canonical external argv prefix:

```bash
.venv/bin/python tools/catanatron_neutral_harness_match.py \
  --checkpoint "$A1" --opponent catanatron_value --mode search \
  --pairs 1 --base-seed 6199000101 --workers 1 \
  --threads-per-worker 1 --device cuda "${COMMON[@]}" \
  --artifact-dir external-games --no-resume --out external.json
```

## Isolation and artifacts

The accepted internal capture began with both B200s at 0% utilization and only
the persistent MPS server present.  It was pinned with
`CUDA_VISIBLE_DEVICES=0`; the external capture was pinned to GPU 1.  Start/end
snapshots are stored beside the flamegraphs.

An earlier internal profile is deliberately rejected: a separate worker-
packing experiment started on GPU 0 after sampling began.  It is retained only
at `/home/ubuntu/catan-eval-profiles-c807874/internal.svg` on
`132.145.197.81` and is not used anywhere in this report.

Tracked artifacts:

| Artifact | SHA-256 |
|---|---|
| `assets/2026-07-11/internal-cross-net-n128.svg` | `a93c8024975e82361eeefb17bea1bfb2e3a41f761dff6d7e7ecf19eed233f5d3` |
| `assets/2026-07-11/external-neutral-n128.svg` | `74c120daac17e4e2c6d2d5bc85066a60303cb4c9516d33ea245b45c445497a52` |
| `assets/2026-07-11/internal-wall-analysis.json` | `1fa18627a3ec2d0afd34b53f5b2ec42647fd65119bb699f5014b5bd74d39b208` |
| `assets/2026-07-11/external-wall-analysis.json` | `f48982ffc07233560b7e01955851d84a7c2791802d45da2b76d516fa3e6fd683` |

Untracked raw captures are preserved for reclassification without committing
multi-megabyte stack text:

- internal: `140.238.192.66:/home/ubuntu/catan-eval-profiles-c807874/internal-wall.raw`,
  SHA-256 `82c8e9dbfb4f302cb589c69199d50e7fc5d428e39d2a9c639862a4ba14970457`;
- external: `132.145.197.81:/home/ubuntu/catan-eval-profiles-c807874/external-wall.raw`,
  SHA-256 `fa837f01f329c1cedf64a8e6364f039ff8ca9851ab83f0df6b39502338eebc28`.
