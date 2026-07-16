# A1 coherent-public next wave v3

This operation supersedes the issued v2 recipe for future generation only.
It retains complete forced-transition retention as value-only rows and
explicitly separates the pinned
legacy teacher feature contract from the fresh learner-row contract:

- teacher/search evaluator: `rust_entity_adapter_v2_land_topology_ports_maritime`
- stored learner rows: `rust_entity_adapter_v4_actor_public_rule_state`

Search priors, values, and selected actions remain checkpoint-bound. Only the
post-search learner tensors advance, restoring Year of Plenty and Monopoly
resource identity plus the current actor's development-card playability, Road
Building continuation, remaining free roads, and discard remainder. Historical
v1/v2 artifacts remain immutable and diagnostically replayable.

The production learner starts natively from scratch with adapter v4,
`public_rule_state` enabled, no learner checkpoint, and fresh optimizer state.
The search teacher remains the deployed adapter-v2 checkpoint; only stored
learner features advance. The function-preserving v4 checkpoint upgrader
remains available for isolated compatibility experiments, but it is not the
production initialization path.

The checkpoint-initialized legacy one-dose/iteration executor must refuse this
production learner. The full retrain must construct a native v4 model rather
than relabeling or resuming a v2/v3 checkpoint.

The same authority now binds the fresh model construction and physical
execution separately from the logical 4096-row dose. The model enables both
structured-action residual paths, public-card counts, and meaningful public
history at construction time. Execution is exactly 8 B200 ranks × 512 rows ×
one accumulation step; no launcher may reinterpret the logical `1 × 4096`
recipe as an arbitrary topology.

`tools/a1_scratch_train.py` authenticates the admitted post-wave composite and
renders the exact native-v4, bias-free 35M command plus a digest-bound planning
receipt. It is intentionally plan-only today: the observed 32-step optimum came
from checkpoint-initialized dose evidence and covers less than one percent of
the full scratch corpus. Its LR, warmup, and flat schedule were likewise
reviewed only for that warm start. Until a complete scratch-optimizer schedule
authority is reviewed, the planner exposes no execution switch and `train_bc`
rejects its child marker before data loading; every planning receipt is
diagnostic-only and non-promotion-eligible.

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
    rust_entity_adapter_v4_actor_public_rule_state \
  --rust-featurize --eval-cache-size 0 \
  --dump-config "$OUT/config.json" --config-purpose a1-next-wave-coherent-public-v3
```

Post-wave admission must prove every worker used teacher v2 and emitted learner
rows v4, with the legacy `adapter_version` row column equal to the learner
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

The native scratch launcher must also render the four admission fields from
`learner.training_recipe` exactly: diagnostics and policy/value interference
every 16 batches, at least two observations, and the complete commissioned
module list. `train_bc.py` verifies those observations before writing the
terminal checkpoint. A run is refused if any required module has a missing,
zero, non-finite, or malformed gradient/update signal, or if the global
policy/value trunk geometry probe is absent or malformed.
