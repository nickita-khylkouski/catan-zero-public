# Catan-Zero 100+ experiment R&D report

Date: 2026-07-10/11
Frozen public source: `c807874940fd5b3e4c51775f33a64279786504da`
Status: local, promotion-ineligible research; no push or merge

## Experiment accounting

| Family | Registered | Successful | Preserved failures |
|---|---:|---:|---:|
| Architecture synthetic screen | 63 | 63 | 0 |
| Learner/objective screen | 48 | 48 | 0 |
| Learner 200-step confirmation | 15 | 15 | 0 |
| Public-masked architecture confirmation | 15 | 15 | 0 |
| GH200 primary systems/search matrix | 45 | 42 | 3 |
| GH200 compile workarounds | 8 | 8 | 0 |
| GH200 evaluator-cache addendum | 2 | 2 | 0 |
| A100 cross-hardware systems confirmation | 15 | 15 | 0 |
| **Accepted total** | **211** | **208** | **3** |

Ten contended search first passes are retained but excluded; all ten were rerun
cleanly. Three architecture launch attempts are also retained but excluded: an
unsafe soft-target corpus was correctly refused, a critical-flag omission was
blocked by the CLI guard, and an exploratory run did not bind the complete
command digest. No optimizer result from those attempts informs a conclusion.

## Main decision

The production incumbent remains the 35,041,353-parameter h640/L6 Transformer.
This sweep did not run a production searched H2H, and the supplied H100 pair was
unusable because GPU fabric remained `In Progress` and CUDA returned error 802.

The strongest next architecture hypothesis is **direct action-to-board binding**,
not more depth, recurrence, MoE, or a larger generic Transformer.

### Next architecture tournament

| Candidate | Parameters | A100 held-out policy CE | Versus incumbent | A100 train throughput | Role |
|---|---:|---:|---:|---:|---|
| Incumbent h640/L6 | 35,041,353 | 1.44643 | control | 866.9 rows/s | current production reference |
| h640/L6 + direct edge policy head | 35,453,514 | 1.30703 | 9.64% lower | 810.3 rows/s | primary causal/warm-growth test |
| h512/L4 + direct edge policy head | 16,421,962 | 1.31118 | 9.35% lower | 1,151.8 rows/s | Pareto efficiency challenger |
| RRT256/L6 | 5,797,938 | 1.19467 | 17.41% lower | 200.2 rows/s | quality clue; rejected by systems gate |

Every row used public-masked inputs, hard teacher actions, terminal outcomes,
three seeds, 200 optimizer steps, and the same 12,862-row held-out set. Soft/Q/
root-value targets were disabled because the proxy corpus could not attest a
safe search-target information regime.

The direct edge head reads each legal move from its post-trunk target entity
(road edge, settlement/city vertex, robber hex) and adds a zero-initialized
per-action logit. The h640 candidate can therefore warm-grow the incumbent while
isolating the mechanism. The h512 candidate is faster and smaller but requires a
fresh/matched initialization and may expose a later capacity ceiling.

Value RMSE stayed approximately 1.0 with near-zero outcome correlation for all
five arms. This result supports the policy readout mechanism only; it supplies
no evidence for a value-head or playing-strength improvement. The 51-bin
HL-Gauss substitute did not improve value learning.

## Learner result

The initial 40-step screen produced unstable rankings. A separate 200-step,
three-seed confirmation retained only one balanced candidate:

- Search soft-target weight `0.60`, temperature `0.50`: policy loss 1.44516,
  value RMSE 0.93348, ECE 0.04574.
- Control: 1.45077 / 0.93639 / 0.04968.

The improvement is small and comes from a 384-game, 6-VP proxy rather than the
production searched corpus. It is an ablation candidate, not a new default.
Half value weight and Adam without warmup did not survive confirmation. A
lambda-0.75 root-value blend remains uninformative because the proxy root label
was constructed from terminal information; it must be retested only with
genuine time-local search root values.

## Systems result

Ranked by expected value and semantic risk:

1. **Native Rust featurization.** 10.3x public microbenchmark speedup, 13.3%
   lower public leaf latency, and 18.8% more n24 logical leaves/s with identical
   selected actions.
2. **Tune the existing shared evaluation-server path.** Twelve GH200 actors at
   batch cap 36 reached 75.02 rows/s, 3.84x the repeated four-local-process
   control, with less than half the memory. A100 independently preserved worker
   scaling: 37.996 rows/s at 12 workers versus 18.944 at four.
3. **Root-wave batching.** Equal 75-leaf work improved from 65.99 to 110.58
   leaves/s (1.68x) with identical actions on three roots. It requires full-game
   parity and searched H2H before adoption.
4. **Large-batch mixed precision.** On A100, BF16 beat FP16 at all tested shapes
   and was 4.58x FP32 at batch 72. On GH200, FP16/BF16 were both about 3.7x FP32
   at batch 72, while FP32 won at batch 12. Inference dtype must remain tied to
   runtime shape and real-checkpoint parity.
5. **Trunk compile, opt-in.** Normal-MHA `fullgraph=False` improved GH200
   forward throughput 7.6-10.6% with 100% argmax agreement and approximately
   1e-7 max logit drift. A100 independently measured 5.6-7.4%. Dynamic fullgraph
   was runtime-dependent, succeeding on A100 and failing on GH200.

Rejected or deferred:

- CUDA Graph eval-server mode regressed 29.05 to 24.54 rows/s and used more
  memory/power.
- Disabling MHA fastpath to force compilation gave 0.97-1.03x and is not useful.
- Evaluator cache reduced repeated-state latency 7.75 to 1.20 ms, but equal-work
  search correctly forbids it; real independent self-play needs measured reuse
  before reconsideration.
- RRT remains too slow in its current implementation despite its policy-quality
  signal.
- Transient MPS improved local multi-process evaluation but remained behind the
  shared server and used substantially more memory.

## Recommended next controlled experiment

Do not replace the incumbent yet. After the dual-arm corpus handoff is fixed:

1. Add R&D-only learner plumbing for the already-implemented edge policy head;
   keep production defaults unchanged.
2. Train three matched seeds each for incumbent, h640/L6+edge, and h512/L4+edge
   on the same audited public-search corpus. Use the confirmed soft-target
   recipe only as a separate factorial arm, not bundled into the architecture
   comparison.
3. Keep terminal/value objectives identical and report phase/legal-width slices,
   policy CE, value calibration, and real H100 throughput.
4. Advance only candidates clearing a predeclared real-corpus learning gate,
   then run paired searched H2H against the incumbent and population/external
   panels.
5. Separately validate Rust featurization, 8/12/16 shared-server actors, root-wave
   parity, and BF16/FP16 on a working H100. Do not combine these systems changes
   until each passes parity.

## Evidence map

- Architecture synthetic ledger and raw results: `architecture/results/`
- Architecture public-hard confirmation and chart:
  `architecture/real_fixture_validation/results/decision_grade/aggregate/`
- Learner screen, confirmation, and chart: `learner/sweep_v2/` and
  `learner/confirmation_v1/`
- GH200 systems report, aggregate, charts, and raw envelopes: `systems/`
- A100 cross-hardware confirmation: `systems/a100_confirmation_20260711/`

All conclusions above are R&D hypotheses except the negative/mechanical facts
directly measured by their bounded harnesses. Production architecture and
training remain unchanged.
