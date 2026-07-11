# Architecture and training-loop decision, 2026-07-11

## Decision

Keep the 35,041,353-parameter h640/L6/8-head Transformer as the control model
for the next research cycle. Stop fixed-depth latent recurrence and sparse
topology-adapter scaling on the current behavioral-cloning target.

The next campaign will test searched, regret-controlled Expert Iteration
(SC-ExIt). It changes the improvement signal by generating new search-policy
and terminal-outcome targets. The first campaign must demonstrate two
consecutive improvement generations before we call it a self-improvement loop.

## Evidence

### Sparse topology adapters

The preregistered topology screen failed. The 38,602,057-parameter h640/L6
Transformer with adapters after layers 2 and 4 did not beat the dense
incumbent. Its paired primary relative improvement was -0.0000331, with a
confidence interval crossing zero.

Result: configs/rnd/topology_real_train_20260710/result_20260710.json.

### RRT fixed latent compute

The RRT screen trained K0/K1/K2/K4/K8 across seeds 11/29/47. K1 through K8
used 22,146,068 parameters; K0 used 20,070,932. Each run consumed 250
optimizer steps and 1,024,000 presentations.

K2 regressed by 0.02289% against capacity-matched K1. K4 regressed by
0.03034%. K8 improved by 0.01713%, but its confidence interval crossed zero
and K8 had no promotion eligibility. The gate rejected every arm.

Result: configs/rnd/e3_a1_screen_20260710/result_20260711.json.

### Transformer fixed latent compute

We added a function-preserving shared cross-attention block with eight latent
slots to the incumbent Transformer. K1/K2/K4 each used 40,793,673 parameters.
K0 retained 35,041,353 parameters as a descriptive control.

The systems gate passed on 16 H100 probes:

| Arm | Median rows/s | Peak allocated bytes |
|---|---:|---:|
| K0 | 1,588.52 | 1,553,824,768 |
| K1 | 1,352.43 | 1,604,660,736 |
| K2 | 1,285.50 | 1,650,892,800 |
| K4 | 1,189.54 | 1,736,968,192 |

K4 retained 74.88% of K0 throughput and used 1.118 times its memory. The
system could run the model.

The learning screen trained 12 runs across seeds 101/103/107. Every run used
the byte-identical frozen teacher 89aa133d...77ebb, a fresh optimizer, 250
steps, and 1,024,000 presentations. The scorer authenticated 1,758,204
evidence rows with identical ordered support.

| Candidate vs K1 | Game-macro CE improvement | 95% CI | Gate |
|---|---:|---:|---|
| K2 | +0.01246% | [-0.00043%, +0.02602%] | fail |
| K4 | +0.00711% | [-0.02645%, +0.03405%] | fail |

The preregistered threshold required at least 2% improvement and a confidence
interval lower bound above zero. Both candidates passed the 0.5% nonforced
decision-micro safety limit and failed the intelligence gate. An independent
full-evidence replay reproduced result SHA
0362251d96d32ac061a2a0a56aff256681c774395d88e07f47648fb168598d85.

Result:
configs/rnd/transformer_think_a1_screen_20260711/result_20260711.json.

## Interpretation

The two fixed-K screens changed compute and representation while preserving
the same supervised target. Neither screen produced a useful gain. K2 and K4
landed within a few hundredths of a percent of K1 on both backbones. More K,
more seeds, or a longer run would spend compute on an effect at least two
orders of magnitude below the registered threshold.

The 35M Transformer remains the control because no tested architecture beat
it and it runs faster than the recurrent candidates. This does not establish
35M as the final scale. We need a stronger target before measuring whether
more capacity can distill it.

## Next campaign: searched regret-controlled Expert Iteration

Expert Iteration separates a planner from the network that learns its
improved policy. AlphaZero repeats the same policy-improvement and
distillation cycle. DeepSeek-R1 supplies a related lesson from another domain:
outcome-verifiable optimization can produce behaviors absent from supervised
imitation. Catan gives us an exact simulator, terminal outcomes, and search.

Primary references:

- [Expert Iteration](https://arxiv.org/abs/1705.08439)
- [AlphaZero](https://arxiv.org/abs/1712.01815)
- [Gumbel AlphaZero](https://openreview.net/forum?id=bERaNdoegnO)
- [KataGo](https://arxiv.org/abs/1902.10565)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Regret-guided search control](https://arxiv.org/abs/2602.20809)

### Stage 0: search signal-to-noise gate

Use 4,096 frozen public states:

- 1,024 setup states with legal width at least 41
- 1,024 robber or development-card states
- 1,024 midgame states
- 1,024 endgame states

Compare current Gumbel n128, completed-Q denoising D1 and D2, diagnostic PUCT,
and diagnostic regularized MCTS at matched logical leaves and orientation
rows. Use independent n512 Gumbel repeats as the reference measurement.

Advance one search semantic only if it meets these checks:

1. Top-1 agreement across repeated seeds rises by at least 5 percentage points
   overall and 10 points for width at least 41.
2. Mean Jensen-Shannon divergence between repeats falls by at least 20%.
3. The n512 action-value delta has a 95% confidence lower bound of at least
   -0.005.
4. Public-observation and determinization provenance match across operators.

Keep current Gumbel if no challenger passes. The test calibrates Gumbel instead
of assuming sequential halving causes the current ceiling.

### Generation 1: four-arm restart experiment

Use one frozen search operator and the 35M Transformer for all arms:

| Arm | Restart sampling | Restart probability |
|---|---|---:|
| C0 | normal start | 0 |
| C1 | uniform archived state | 0.20 |
| C2 | regret-weighted state, temperature 0.10 | 0.20 |
| C3 | regret-weighted state, temperature 0.10 | 0.50 |

Stop each arm after the first completed trajectory that crosses 25,000,000
logical leaves. Require at least 2,400 clean terminal outcomes per arm or fail
the campaign as underpowered. Match orientation rows within 1%. Record leaf
overshoot and keep the between-arm difference below one maximum complete
trajectory.

Each restart continuation must run search and emit the improved root policy,
terminal Monte Carlo outcomes, public-information provenance, and an exact
work ledger.

The existing generate_restart_selfplay.py path cannot run this experiment. It
uses raw-policy continuations, emits value-only rows, and runs in one process.
Reuse its restart-state parsing and replace its continuation and target path.

Train seeds 11/29/47 for every arm at 250 steps and global batch 4096. This
yields 12 learner runs and 12,288,000 presentations.

The learning gate requires:

- at least 1% relative nonforced game-macro CE improvement over C0
- a crossed-bootstrap 95% confidence interval above zero
- nonforced decision-micro CE regression no worse than 0.5%
- held-out terminal-value Brier or MSE regression no worse than 1%
- no critical phase or width-at-least-41 bucket worse by more than 2%
- a positive point estimate for all three learner seeds

The strength screen uses 100 paired seat-swapped n16 games per learner seed.
Advance the best registered arm only when P(mean Elo > 0) is at least 0.95,
P(mean Elo < -10) is at most 0.05, and no seed point estimate falls below
-20 Elo.

Run a 150-to-300-pair pentanomial SPRT for the selected arm with elo0=-10,
elo1=+15, and alpha=beta=0.05. Add 500-pair n128 panels against the
incumbent and catanatron_value, plus a 200-pair held-out high-regret suite.
An external drop below -10 Elo or a critical-bucket regression above 5%
vetoes promotion.

### Generation 2: prove iteration

Run C0 and the winning restart arm from their own generation-1 checkpoints.
Give each arm 4,800 terminal trajectories under matched work.

Call the loop successful only if the restart lineage gains at least 10 Elo
over its generation-1 checkpoint and stays at least 15 Elo ahead of
generation-2 C0. One generation can demonstrate useful data selection. Two
gains demonstrate iterative improvement.

## Population training comes after the loop test

The repository contains league snapshots, PFSP, a payoff matrix, population
arena scheduling, and a max-entropy Nash solver. It lacks a transactional
Nash-mixture oracle loop and held-out best-response exploitability evaluation.
KLENT should wait: its H100 smoke produced zero terminal games and truncated
both games at 600 decisions.

If SC-ExIt gains internal Elo and then cycles or loses against the external
panel, run one-step Nash-PSRO versus PFSP. Use four frozen population members,
three seeds per arm, and independent best-response probes.

Primary references:

- [PSRO](https://papers.nips.cc/paper_files/paper/2017/hash/3323fe11e9595c09af38fe67567a9394-Abstract.html)
- [NFSP](https://arxiv.org/abs/1603.01121)
- [AlphaStar league training](https://www.nature.com/articles/s41586-019-1724-z)
- [OpenSpiel](https://arxiv.org/abs/1908.09453)

## Architecture work that remains justified

Public-history belief features can change the information available to the
policy. Add explicit resource and development-card posterior targets before
another attention or recurrence sweep. Train the auxiliary heads from private
simulator labels while feeding the policy public observations. Measure
calibration and downstream high-regret strength.

Return to width, action cross-attention, MoE, or adaptive halting after search
creates a target that the 35M model cannot distill. Any adaptive-compute
experiment needs a matched random-halting control and intermediate
search-improvement supervision.

## Software work before the next GPU campaign

1. Add a searched restart continuation worker with policy and terminal-outcome
   targets.
2. Add exact terminal-game, logical-leaf, and orientation-row ledgers.
3. Build the Stage 0 public-information operator parity adapter.
4. Add phase, legal-width, and regret-bucket holdouts.
5. Add generation-to-generation registration and no-overwrite promotion
   transactions.
6. Preserve paired seat/game seeds through internal and external strength
   evaluation.

No production defaults or checkpoints changed during this research.
