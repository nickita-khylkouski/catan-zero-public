# A1 coherent-public next wave v3

This operation supersedes the issued v2 recipe for future generation only.
It retains complete forced-transition retention as value-only rows and
explicitly separates the pinned
legacy teacher feature contract from the fresh learner-row contract:

- teacher/search evaluator: `rust_entity_adapter_v2_land_topology_ports_maritime`
- stored learner rows: `rust_entity_adapter_v3_structured_action_resources`

Search priors, values, and selected actions remain checkpoint-bound. Only the
post-search learner tensors advance, restoring Year of Plenty and Monopoly
resource identity for the from-scratch v3 learner. Historical v1/v2 artifacts
remain immutable and diagnostically replayable.

Generate through the sealed pre-wave control plane. For a direct lane command:

```bash
"$PY" tools/generate_gumbel_selfplay_data.py \
  --config configs/experiments/next_wave/coherent_public_n128_adaptive256_forced_value_v3.schema15.json \
  --prelaunch-guard-config configs/guards/a1_generation_coherent_public_n128_adaptive256_forced_value_v3.json \
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
  --no-exact-budget-sh --exact-budget-sh-min-n 0 \
  --native-mcts-hot-loop --forced-root-target-mode trajectory_only \
  --target-reliability-audit-fraction 0.05 \
  --target-reliability-audit-seed 20260716 \
  --record-automatic-transitions \
  --meaningful-public-history --event-history-limit 32 \
  --learner-entity-feature-adapter-version \
    rust_entity_adapter_v3_structured_action_resources \
  --rust-featurize --eval-cache-size 0 \
  --dump-config "$OUT/config.json" --config-purpose a1-next-wave-coherent-public-v3
```

Post-wave admission must prove every worker used teacher v2 and emitted learner
rows v3, with the legacy `adapter_version` row column equal to the learner
identity. Forced rows retain `policy_weight_multiplier=0` and
`value_weight_multiplier=1`; training continues to use equal per-game value mass.

The five-percent reliability slice is selected by a stable hash of audit seed,
game seed, and decision index. Its duplicate search uses independent
Gumbel/chance/belief streams and never selects the live move. The learner binds
`--target-reliability-confidence-weighting` with a `0.25` floor and leaves both
global and per-game surprise sampling disabled. This prevents raw
search-vs-parent disagreement from amplifying unstable labels before duplicate
search has measured their reliability.

The learner also binds `--phase-weights PLAY_TURN=4.0`. On the admitted
959,142-row coherent corpus, equal-per-game policy weighting otherwise assigns
only 34.16% of policy objective mass to ordinary `PLAY_TURN` decisions and
65.84% to opening, discard, and robber prompts. The 4x multiplier restores
66.49% `PLAY_TURN` mass, matching the 66.08% strategic-turn share of the
historically successful selected-dose corpus while retaining supervision for
every mandatory multi-action prompt.

The coherent learner is capped at 32 optimizer steps. This is not the
historical corpus's 128-step dose copied forward: on the matched coherent
frontier, step 32 scored 56.25% against f7 and 51.17% against v5, while the
step-128 checkpoint fell to 51.95% and 45.31%. The denser current sampler
changes what one optimizer step means, so continuing to 128 erased the useful
early update.
