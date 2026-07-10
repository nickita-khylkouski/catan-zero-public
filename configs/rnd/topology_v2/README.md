# Topology adapter v2 R&D sandbox

This directory is an isolated, non-production experiment contract for improving
the 35,041,353-parameter h640/L6 EntityGraph Transformer. Nothing here changes
the default model or authorizes promotion.

## Architectural judgment

Catan has a small, fixed physical graph and already pays for six global
Transformer blocks. Sparse-global attention and FlashAttention variants solve
the wrong bottleneck. The useful inductive bias is state-dependent selection
over typed local neighbors, followed by an action readout that can use the
selected board entity.

The current `basis_mean_v1` adapter is a valid control, but it uniformly
averages incident messages and spends four dense bottleneck transforms on every
token. The v2 candidate uses destination-conditioned sparse local attention:

```text
q_i = Wq Down(Norm(h_i))
k_j = Wk Down(Norm(h_j))
v_j = Wv Down(Norm(h_j))
s_ij = <q_i, k_j> / sqrt(d_head) + relation_bias[r_ij]
a_ij = softmax over incoming edges to destination i
m_i = sum_j a_ij v_j
h_i' = h_i + live_destination_i * Up(SwiGLU(m_i))
```

`Up` is zero initialized, so a warm-grown checkpoint is exactly the incumbent
function at step zero. Deltas are masked off for padded and degree-zero tokens.

## Required controls

- `C0`: unchanged 35M incumbent.
- `C1`: true typed `basis_mean_v1` incidence adapter.
- `C2-v1` / `C2-v2`: identical adapter and edge kernels, but sources are
  replaced by their destination token. These isolate parameter/compute
  capacity from neighbor communication within each architecture.
- `C3-v1` / `C3-v2`: fixed type-cyclic endpoint rotation. This preserves
  source types, relation counts, destination degrees, and edge work, but not
  each source node's degree. It is a first geometry control, not a fully random
  configuration-model rewire.
- `C4`: a near-parameter-matched v1 mechanism arm (3,600 versus 3,553 toy
  parameters) to separate toy capacity from the attention mechanism.
- `V2`: receiver-conditioned `local_attention_v2`. The systems-screen lead is
  now a shared bottleneck-128 adapter at layers `2,4`: 35,325,833 parameters.

Placement remains layers `2,4` until true topology beats both controls. Layers
`1,3` and `4,6` are conditional follow-ups, not part of the first sweep.

## Fail-closed sequence

1. Deployment: eval-server and CUDA-graph paths retain every topology tensor;
   malformed per-type IDs are rejected.
2. Identity: every warm-grown arm matches incumbent logits/value bit-for-bit at
   step zero.
3. Mechanism: correct topology beats self-message and rewired controls on
   counterfactual road, settlement-distance, robber-production, port-access,
   and D6-consistency probes.
4. Systems: two isolated H100 repeats, a real collated batch, and stored direct,
   eval-server, and CUDA-graph evidence; at least 0.5x incumbent throughput and
   less than 1.5x peak allocation.
5. Learning: exactly 1,024,000 presentations per arm/seed, three seeds, at
   least 256 shared holdout games, and at least 1,024 pre-registered sensitive
   decisions. Training, holdout, mask, data, config, and checkpoint hashes must
   be registered before scoring. The primary metric is game-macro soft-target
   CE on non-forced topology-sensitive decisions.
6. Strength: paired equal-leaf and equal-wall-time search matches, followed by
   200-pair confirmation before any promotion.

The prior 37,504-row corpus is smoke data only: 58/64 games truncated. Value/Q
claims remain blocked until a fixed-seed completion pilot reaches at least 90%
terminal games. Truncated rows may train policy but cannot support primary
outcome claims.

## Research anchors

- [GraphGPS](https://arxiv.org/abs/2205.12454): combine local message passing
  and global attention rather than choosing one exclusively.
- [Graphormer](https://arxiv.org/abs/2106.05234): inject shortest-path and edge
  structure into attention logits.
- [GRIT](https://proceedings.mlr.press/v202/ma23c.html): relative random-walk
  encodings and learned pair-state attention; retained as a later upper bound.
- [GNN+ reassessment](https://proceedings.mlr.press/v267/luo25h.html): strong
  gated local message passing can rival more elaborate graph Transformers.

These papers motivate falsifiable mechanisms. They are not evidence that a
particular Catan model is stronger.

## First implementation and H100 systems result

The drop-in implementation now includes:

- topology transport through direct, eval-server, and conditional CUDA-graph
  inference paths;
- fail-closed topology shape, integer, and per-entity range validation;
- `basis_mean_v1` at configurable bottleneck width;
- receiver-conditioned `local_attention_v2` with relation-aware K/V/bias,
  sparse destination softmax, receiver gate, and live-destination masking;
- shared-weight, self-message, and type-preserving rewired controls;
- an atomic checkpoint upgrader that preserves all incumbent tensors and
  records source/output hashes; normal loading rejects missing adapter tensors;
- an executable learning-gate scorer that requires paired holdout support,
  seeds 1/2/3, immutable training/holdout/mask/checkpoint provenance, and a
  crossed paired bootstrap over model seed and common holdout game.

Two GPU-only bugs were found and fixed before recording evidence: AMP relation
values initially promoted BF16 messages to FP32 before scatter, and CUDA graph
capture initially attempted a forbidden host-to-device constant construction.
Regression tests now cover BF16 scatter and actual CUDA graph capture/replay.

Two isolated H100 training-step repeats at B32/A64/E64 used random token and
action features plus a real Catan board's full incidence tensors. They measured:

| arm | parameters | mean rows/s | incumbent ratio | peak GiB |
|---|---:|---:|---:|---:|
| incumbent | 35,041,353 | 1,070.46 | 1.0000x | 1.498 |
| basis mean b192 | 35,979,081 | 951.32 | 0.8887x | 1.739 |
| local attention v2 b192 | 36,065,481 | 562.20 | 0.5252x | 1.798 |
| shared local attention v2 b192 | 35,553,417 | 563.24 | 0.5262x | 1.795 |

All pass the provisional 0.5x throughput and 1.5x memory thresholds in this
short synthetic-feature smoke. A real-collated screen then used 32 deterministic
public env states, actual legal-action features/contexts/targets, 21.39% legal
mask utilization, and 25.78% event utilization:

| real-collated arm | parameters | mean rows/s | incumbent ratio | peak GiB |
|---|---:|---:|---:|---:|
| incumbent | 35,041,353 | 1,135.71 | 1.0000x | 1.497 |
| basis mean b192 | 35,979,081 | 948.22 | 0.8349x | 1.739 |
| local attention b128 | 35,610,313 | 583.10 | 0.5134x | 1.727 |
| shared local attention b128 | 35,325,833 | 591.73 | 0.5210x | 1.726 |
| local attention b160 | 35,823,561 | 580.93 | 0.5115x | 1.764 |
| shared local attention b160 | 35,432,457 | 551.43 | 0.4855x | 1.762 |
| local attention b192 | 36,065,481 | 554.14 | 0.4879x | 1.798 |
| shared local attention b192 | 35,553,417 | 560.65 | 0.4937x | 1.795 |

The exact next learning-screen candidate is therefore shared-b128 at layers
`2,4`, not unshared-b192. It has the highest observed v2 throughput, the lowest
v2 parameter count, and a 2.10-point margin over the throughput threshold.
These are two short repeats, so this selects a candidate; it is not a stable
fleet-capacity estimate. A stored eval-server latency artifact remains open.

The first mechanism probe assigns random values to hex tokens and asks each
vertex to predict the mean value of its physically incident hexes. Across three
seeds and 200 updates, tail-mean MSE was:

| mechanism arm | mean MSE |
|---|---:|
| basis mean, true topology | 0.034308 |
| basis mean, near-parameter-matched true topology | 0.033031 |
| basis mean, self-message control | 0.592250 |
| basis mean, fixed type-cyclic rewiring | 0.572829 |
| local attention v2, true topology | 0.011063 |
| local attention v2, self-message control | 0.592683 |
| local attention v2, fixed type-cyclic rewiring | 0.571790 |

This validates neighbor communication and both architectures' control wiring.
At the same 200-update budget, v2 also beats the near-parameter-matched v1 toy
arm. Because uniform averaging does not test attention itself, a second probe
uses paired scenes with identical incident-hex keys/values and opposite receiver
queries. Its fixed readout prevents the residual from leaking the query:

| receiver-conditioned arm | mean tail MSE | normalized improvement |
|---|---:|---:|
| basis mean, true topology | 1.217511 | 74.45% |
| local attention v2, true topology | 0.015463 | 99.68% |
| local attention v2, self-message | 4.764995 | -0.00% |
| local attention v2, fixed type-cyclic rewiring | 4.577317 | 3.94% |

This supports receiver-conditioned neighbor selection as a real architectural
capability, while remaining only a mechanism result. The next required systems
test is a stored eval-server latency/replay smoke; only after that passes is the
1,024,000-presentation-per-arm/per-seed game-level policy screen admissible.
`tools/rnd_topology_learning_gate.py` is its fail-closed scorer. Registration
hashes are deliberately null until the data split and mask artifact are frozen,
so the scorer refuses to run prematurely. The actual learning run has not been
performed.
