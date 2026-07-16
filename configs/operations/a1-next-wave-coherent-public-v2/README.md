# A1 coherent-public next wave v2

This operation supersedes the issued v1 recipe for future generation only.
The search, evaluator, learner, and evaluation contracts are unchanged; v2
adds complete forced-transition retention so every game contributes value-only
rows for sole-action states. Historical v1 artifacts remain immutable and
diagnostically replayable.

Generate through the sealed pre-wave control plane. For a direct lane command,
the versioned config and guard are authoritative:

```bash
"$PY" tools/generate_gumbel_selfplay_data.py \
  --config configs/experiments/next_wave/coherent_public_n128_adaptive256_forced_value_v2.schema13.json \
  --prelaunch-guard-config configs/guards/a1_generation_coherent_public_n128_adaptive256_forced_value_v2.json \
  --checkpoint "$CHECKPOINT" --out-dir "$OUT" \
  --base-seed "$BASE_SEED" --games "$GAMES" --workers "$WORKERS" \
  --ledger-claim-label "$CLAIM_ID" --device cuda \
  --track 2p_no_trade --vps-to-win 10 --format npz --score-actions \
  --n-full 128 --n-fast 16 --p-full 0.25 \
  --c-visit 50.0 --c-scale 0.1 --sigma-eval 0.79 \
  --max-decisions 600 --max-depth 80 \
  --temperature-clock nonforced_choice --temperature-decisions 40 \
  --temperature-high 1.0 --temperature-low 0.0 \
  --late-temperature-decisions 100 --late-temperature 0.1 \
  --public-observation --coherent-public-belief-search \
  --no-information-set-search --no-belief-chance-spectra \
  --determinization-particles 1 --determinization-min-simulations 32 \
  --correct-rust-chance-spectra --lazy-interior-chance \
  --symmetry-averaged-eval --symmetry-averaged-eval-threshold 20 \
  --native-mcts-hot-loop --forced-root-target-mode trajectory_only \
  --record-automatic-transitions \
  --meaningful-public-history --event-history-limit 32 \
  --rust-featurize --eval-cache-size 0 \
  --dump-config "$OUT/config.json" --config-purpose a1-next-wave-coherent-public-v2
```

Before training, the post-wave contract must report forced rows in every
completed game, with `policy_weight_multiplier=0`,
`value_weight_multiplier=1`, and non-empty phase/action coverage. Training
continues to use equal per-game value mass and the typed forced-value recipe
`END_TURN=1.0,ROLL=1.0`.
