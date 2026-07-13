# Initial-road D1 fixed-root probe

This is a target-reliability probe, not a promotion run. It replays the same
opening settlement and road roots with disjoint search RNG seeds. The only
operator differences allowed are the D1 coefficient and its initial-road-only
scope. The report includes raw completed-Q range/top margin, target confidence,
target entropy, target-to-prior JS, and cross-seed JS/top-1 stability by exact
raw phase.

```bash
python tools/fixed_root_search_stability.py \
  --checkpoint "$CHECKPOINT" \
  --evaluator-config configs/experiments/s3_fixed_root_r3/evaluator_public_scalar.json \
  --config-a configs/experiments/opening_road_d1/control_n128_p4.json \
  --config-b configs/experiments/opening_road_d1/road_d1_c8_n128_p4.json \
  --allowed-search-config-differences rescale_noise_floor_c,rescale_noise_floor_initial_road_only \
  --root-panel "$OUT/opening-roots.json" \
  --create-root-panel \
  --n-roots 32 \
  --decisions-per-game 0,1,2,3 \
  --repeats 4 \
  --device cuda \
  --out "$OUT/road-d1-c8.json"
```

The arm is promising only if `BUILD_INITIAL_ROAD` target confidence and
target-to-prior JS fall while cross-seed stability improves. Every
`BUILD_INITIAL_SETTLEMENT` run must remain exactly equal between roles for the
same search seed in a same-seed follow-up; this disjoint-seed probe first checks
that its aggregate settlement distributions and costs do not drift.
