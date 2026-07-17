# Coherent-public boundary-value particle experiment

This is an explicit prospective teacher-operator experiment. It compares the
current coherent-public n128 operator with one sampled hidden world at an
opponent/new-turn boundary (`K=1`) against the same operator with four
observer-preserving determinizations averaged at that boundary (`K=4`).

It is deliberately **not** part of the canonical generation recipe yet.
Changing `boundary_value_particles` changes the policy teacher, its cost, and
the data contract; it must be measured with the native Rust hot loop used for
production generation before it can be adopted.

The causal difference is exactly `boundary_value_particles`.

```bash
CHECKPOINT=/absolute/path/to/current-champion.pt
OUT=/absolute/path/to/boundary-k1-vs-k4

python tools/fixed_root_search_stability.py \
  --checkpoint "$CHECKPOINT" \
  --evaluator-config configs/experiments/teacher_operator_coherent_v1/evaluator_public_scalar.json \
  --config-a configs/experiments/boundary_particles_k4_v1/base_coherent_n128_d6_k1.json \
  --config-b configs/experiments/boundary_particles_k4_v1/coherent_n128_d6_k4.json \
  --allowed-search-config-differences boundary_value_particles \
  --root-panel "$OUT/real-roots.json" \
  --create-root-panel \
  --n-roots 64 \
  --repeats 4 \
  --device cuda \
  --native-mcts-hot-loop \
  --out "$OUT/fixed-root.json"
```

The installed native wheel must advertise both
`coherent_public_belief_search` and `boundary_value_particles`; the probe
fails closed otherwise. A candidate is worth a paired native H2H panel only
if it reduces cross-seed target disagreement without unacceptable wall-time
cost. It is not a substitute for a future policy-conditioned history/belief
model: it reduces one-sample boundary variance while preserving the current
public-information contract.
