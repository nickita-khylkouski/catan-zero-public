# Future RL architecture R&D — executable hypotheses, not conclusions

Date: 2026-07-10. Hardware boundary: eight H100 80GB GPUs on the permitted
host; the separately supplied B200 host was not used. Production defaults
remain unchanged.

## What the current system is actually doing

The incumbent is expert iteration, not plain PPO: a public-observation entity
Transformer supplies policy/value estimates; Gumbel search constructs improved
policy, Q, and value targets; BC/distillation learns those targets; later
self-play/PPO and league machinery can refine the checkpoint. Search was still
able to see authoritative hidden state even when the network input was masked.
This branch adds a fail-closed, opt-in public-conservation PIMC path that samples
complete worlds from bank/deck conservation and public hand sizes, searches only
during the root actor's turn, and averages root evidence. It is not yet a
history-conditioned Bayesian posterior; the provenance label says so. Chance-only
public beliefs are useful compatibility logic but are not mislabeled as full
information-set search.

This matters more than swapping attention blocks: a sophisticated trunk trained
against hidden-state-leaking targets learns a teacher artifact, not better Catan.

## Five falsifiable experiments

| ID | One changed mechanism | Exact first screen | Why it might improve decisions | Primary falsifier |
|---|---|---|---|---|
| E1 | search operator | legacy Gumbel, exact-budget Gumbel, PUCT, reverse-KL regularized MCTS, raw control | tests whether Gumbel is the intelligence bottleneck rather than assuming it | paired equal-work and equal-time loss to incumbent |
| E2 | state representation | 35,041,353-param incumbent; 20,070,932-param RRT-384-L9; 20,936,010-param ResRGCN-384-L14 | explicit typed board incidence may improve sample efficiency and transfer | H100 cost, then held-out loss and paired strength |
| E3 | latent deliberation | shared block with K={0,2,4,8}; compare extra-search and untied-depth controls | DeepSeek-R1 suggests extra test-time computation can elicit better reasoning, but Catan needs latent state updates rather than text CoT | no gain over equal-FLOP search or untied depth |
| E4 | conditional capacity | dense control versus top-2, 8-expert MoE; match active FLOPs and separately total parameters | DeepSeek-V3 shows sparse capacity can scale knowledge without proportional active compute | routing collapse, poor utilization, or no gain over both dense controls |
| E5 | self-improvement rule | search distillation; KLENT direct RL; distill then KLENT | tests AlphaZero-style policy improvement against a modern stable direct-RL update | instability, lower paired strength, or higher compute for equal strength |

E1/E2 have implementations and a fail-closed measurement stack. E3/E4 source is
also implemented and correctness-tested: shared-weight Think-RRT K={1,2,4,8}
has 22,146,453 scalar-contract parameters for every K; top-2-of-8 MoE RRT has
28,508,948 total and 20,525,588 nominal active parameters. E5's KLENT loss,
actor, two-player rollout collector, and updater are implemented. None has a
strength result yet.

## Exact E2 candidates and the first H100 result

The primary E2 contract holds the incumbent scalar value/readout fixed. The
optional 51-bin contract is a separate objective experiment, not a silent extra
change.

| trunk | width / depth | details | scalar params | optional 51-bin params |
|---|---:|---|---:|---:|
| Transformer incumbent | 640 / 6 | 8 heads | 35,041,353 | 35,484,925 |
| RRT | 384 / 9 | 6 heads, `RRTRRTRRT`, FF 1024, one action cross layer | 20,070,932 | 20,238,792 |
| ResRGCN | 384 / 14 | 4 bases, FF 512, no action cross layer | 20,936,010 | 21,103,870 |

A bounded synthetic BF16 forward/backward probe used batch 32, 64 legal actions,
64 events, three warmups, and twelve measured steps on otherwise idle H100s.
The probe now refuses non-H100 devices and checks every materialized parameter
gradient for finiteness. Raw per-run JSON retains the measured GPU name, device,
resolved architecture, elapsed time, and runtime versions. It remains a
kernel/feasibility measurement, not a strength result.

| trunk | rows/s | peak allocated | relative to incumbent | current ruling |
|---|---:|---:|---:|---|
| incumbent | 1,500.4 | 1.50 GiB | 1.00x | reference |
| RRT | 203.1 | 1.51 GiB | 0.135x | correctness candidate; optimize before training screen |
| ResRGCN | 6.68 | 2.04 GiB | 0.0045x | stop this dense implementation before training |

RRT at batch 64 reached 181.7 rows/s and 2.79 GiB, so the batch-32 gap is not a
small-batch artifact. The likely mechanism is full `[B,L,L]` relation materialization
and attention-bias handling; the probe does not establish whether a faster
implementation would be stronger. ResRGCN is ruled out only in its current dense
implementation, not as an architectural idea.

Two additional bounded H100 probes used the same shape. Think-RRT K=4 reached
197.9 rows/s with 1.61 GiB peak; top-2-of-8 MoE RRT reached 133.6 rows/s with
1.76 GiB peak. Both are finite and mechanically viable, but both inherit the
RRT systems bottleneck. The K=4 result means recurrence itself is not the main
tax; it does not show that recurrent computation improves play.

The next E2 implementation should therefore be a cheap hybrid: retain the
incumbent six global Transformer blocks and add at most two topology message
passing adapters, then require at least 0.5x incumbent H100 throughput before
any learning screen. Do not spend a 40-GPU run to discover a 10–300x kernel tax.

## Executed scaled ladder and exploratory learning smoke

The cheap hybrid was implemented as sparse typed incidence adapters after
Transformer blocks 2 and 4. It never materializes a dense `[B,S,S]` relation
tensor, and its output projection is zero initialized so growing from the 35M
checkpoint initially preserves the incumbent function exactly. The matched
scaled ladder is:

| arm | exact scalar parameters | H100 rows/s | relative systems rate | peak probe allocation |
|---|---:|---:|---:|---:|
| incumbent, h640/L6 | 35,041,353 | 1,222.99 | 1.000x | 1.498 GiB |
| sparse hybrid, h640/L6, adapters 2/4, b448 | 38,602,057 | 657.50 | 0.538x | 1.947 GiB |
| dense h832/L6 | 59,131,977 | 1,090.68 | 0.892x | 1.972 GiB |
| dense h832/L10 | 92,401,993 | 660.68 | 0.540x | 3.083 GiB |

All four passed finite forward/backward and the predeclared 0.5x throughput
gate. These concurrent synthetic probes are a relative systems screen, not an
isolated absolute benchmark. Raw results are in
`configs/rnd/scaled_probe_raw_20260710/`.

The 59M and 92M models then received the same 200-step, 204,800-presentation
hard-teacher/outcome bootstrap as the incumbent. The held-out results were
nearly identical: policy CE was 1.47262 / 1.47031 / 1.47348 for 35M / 59M /
92M, while top-1 accuracy was 70.44% / 70.17% / 70.43%. More parameters did not
buy measurable bootstrap sample efficiency at this budget.

A fresh public-information Gumbel/PIMC corpus used 64 unique games and 37,504
rows. All eight independent shard audits passed: zero invalid actions, exactly
one information regime (`public_conservation_pimc_v1`), 100% soft-label legal
coverage, and a mean non-forced KL from prior of roughly 0.18. Search therefore
changed the policy on real decisions. However, 58/64 games truncated at the
600-decision cap. Terminal/value/Q/VP objectives were consequently disabled;
the architecture comparison trained only on full-coverage soft policy targets,
with forced rows carrying zero policy weight.

The equal-exposure exploratory smoke presented 25,600 raw rows to every arm. The hybrid
was warm-grown from the incumbent with 90.7759% of its parameters copied; the
wide arms used their matched teacher bootstraps.

| arm | validation policy CE | change versus incumbent | top-1 | top-3 | train time |
|---|---:|---:|---:|---:|---:|
| incumbent 35M | 1.384050 | reference | 67.20% | 93.58% | 11.25 s |
| sparse hybrid 38.6M | 1.384022 | 0.0020% lower | 66.49% | 93.76% | 19.96 s |
| dense 59.1M | 1.384003 | 0.0034% lower | 66.67% | 93.23% | 14.25 s |
| dense 92.4M | 1.383902 | 0.0107% lower | 66.49% | 93.67% | 20.47 s |

Every difference is below 0.011%, so this smoke found no resolvable separation.
It did **not** execute or clear the ladder's predeclared fixed-corpus gate of
250 steps, global batch 4,096, and 1,024,000 presentations. It only rules out a
dramatic immediate sample-efficiency gain at this tiny budget; it does not rank
playing strength or rule out a gain after substantially more fresh data.
Machine-readable reports are in
`configs/rnd/architecture_screen_raw_20260710/`, with bootstraps and corpus
audits in the adjacent `bootstrap_raw_20260710/` and
`public_pimc_raw_20260710/` directories.

The direct KLENT signal gate was also executed from the masked 35M checkpoint.
Both games truncated (1,200 decisions total), so the new fail-closed runner
refused the update before publishing a checkpoint. With a zero-initialized Q
head and no terminal reward, proceeding would mostly optimize self-generated
regularization targets rather than intelligence. The refusal artifact is in
`configs/rnd/klent_raw_20260710/report.json`.

### Current architecture decision

Keep the 35,041,353-parameter Transformer as the production reference. The
38,602,057-parameter sparse hybrid is the only new trunk that merits a larger
replicated learning test because it preserves the incumbent at initialization
and passed the systems gate. Do not allocate a 40-GPU campaign to the 92M model:
depth scaling produced no meaningful small-data gain. If the hybrid clears the
2% replicated learning gate and paired-search strength gate, the next scale
candidate is h832/L6 at exactly 59,131,977 parameters; h832/L10 is a later
capacity control, not the default future architecture.

The next intelligence bottleneck is the loop, not another attention variant:
first drive complete public-information self-play above 90%, then compare
search-distillation plus complete outcome learning against the incumbent over
at least 1,024 games and three training seeds. Keep Gumbel search until an
equal-work/equal-time search tournament beats it. KLENT remains blocked until
direct rollouts reliably terminate and supply external outcome signal.

## Why DeepSeek and AlphaGo do not imply the same change

DeepSeek-R1's reported intelligence gain comes from large-scale RL on verifiable
reasoning tasks and long generated token trajectories. A Catan action does not
have a naturally supervised textual chain of thought. The transferable
hypothesis is additional, outcome-trained computation (E3/E5), not emitting text
or copying GRPO mechanically. DeepSeek-V3's MoE and latent-attention choices
target enormous language-model capacity and KV-cache cost; Catan's roughly 150–215
entity tokens make MLA a weak first bet, while sparse experts remain testable
only after routing/utilization instrumentation exists.

AlphaGo Zero/AlphaZero improved through a coupled loop: self-play distribution,
search-based policy improvement, outcome value targets, training, and a gate.
It was not “PB0 alone” and it was not architecture alone. Catan adds stochastic
chance, hidden information, and multiple prompts by the same player, so this
branch keeps search, target provenance, and Catan-aware return signs explicit.

## Stage gates before a 40-GPU campaign

1. Contract gate: public-input invariance, legal-action support, finite gradients,
   checkpoint round-trip, exact parameter count.
2. Systems gate: two H100 repeats; at least 0.5x incumbent rows/s, no OOM at the
   chosen batch, and less than 1.5x incumbent peak memory.
3. Learning screen: same 100k–500k public rows, same seeds/order/optimizer, report
   equal-step and equal-time held-out policy loss, value error, calibration, and
   phase/legal-width slices.
4. Search screen: frozen network, 50 paired games per arm under both equal logical
   leaves and equal wall time. Raw policy stays a control, never padded with fake work.
5. Confirmation: 200 pairs, two independent seed blocks, frozen opponent/reference,
   complete games only. Promote on a predeclared confidence gate, not point estimate.
6. Scale: only the surviving mechanism receives the 40-GPU training campaign.
   Run at least three independent training seeds and reserve GPUs for continuous
   frozen-reference evaluation; do not launch five unfiltered ideas at full scale.

## Leaderboard if a candidate fails

Rank evidence in this order, separately for equal-work and equal-time regimes:

1. paired win/draw/loss score and confidence interval;
2. high-regret/opening and public-belief invariance vetoes;
3. held-out policy/value/calibration slices;
4. logical leaves and orientation rows purchased per second;
5. H100 memory and throughput;
6. parameter count and total training GPU-hours.

Failure routes are explicit: E2 systems failure returns to the cheap hybrid;
E3 failure retains ordinary search; E4 failure retains dense capacity; E5
failure retains search distillation. A failed hypothesis does not alter the
incumbent default or invalidate the other experiment axes.

## Primary research anchors

- Danihelka et al., *Policy improvement by planning with Gumbel* (ICLR 2022): https://openreview.net/forum?id=bERaNdoegnO
- Silver et al., *Mastering the game of Go without human knowledge* (2017): https://doi.org/10.1038/nature24270
- Silver et al., *A general reinforcement learning algorithm that masters chess, shogi, and Go* (2018): https://doi.org/10.1126/science.aar6404
- DeepSeek-AI, *DeepSeek-R1* (2025): https://arxiv.org/abs/2501.12948
- DeepSeek-AI, *DeepSeek-V3* (2024): https://arxiv.org/abs/2412.19437
- Ohta et al., *KLENT* (2026): https://arxiv.org/abs/2602.10894
- Schlichtkrull et al., *Relational Graph Convolutional Networks* (2017): https://arxiv.org/abs/1703.06103
- Ying et al., *Graphormer* (2021): https://arxiv.org/abs/2106.05234
- Danihelka et al., *Student of Games* (2023): https://arxiv.org/abs/2305.13740
- Brown et al., *ReBeL* (2020): https://arxiv.org/abs/2007.13544

These papers motivate hypotheses. The local paired ledger, not citation count,
decides what advances.
