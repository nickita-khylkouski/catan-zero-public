# A1 coherent-public next wave v3

This operation supersedes the issued v2 recipe for future generation only.
It retains complete forced-transition retention as value-only rows and
explicitly separates the pinned
legacy teacher feature contract from the fresh learner-row contract:

- teacher/search evaluator: `rust_entity_adapter_v5_meaningful_history_v2`
- stored learner rows: `rust_entity_adapter_v5_meaningful_history_v2`

Search priors, values, and selected actions remain checkpoint-bound. Only the
post-search learner tensors advance, restoring Year of Plenty and Monopoly
resource identity plus the current actor's development-card playability, Road
Building continuation, remaining free roads, and discard remainder. Historical
v1/v2 artifacts remain immutable and diagnostically replayable.
The typed generator resolves the internal historical engine settings
`--forced-root-target-mode trajectory_only`,
`--coherent-public-belief-search`, `--record-automatic-transitions`,
`--meaningful-public-history`, `--event-history-limit 64`, and
`--preserve-search-evidence`; callers do not override them. The learner
separately binds `--value-phase-weights none` so the policy's phase sampler
cannot silently redefine the value-state distribution.
The current typed generator artifact is schema 19, which binds
`boundary_value_particles` and the retained H100 EvalServer runtime into
generation and evaluation identity. Current-producer jobs run one generator
per physical GPU with 24 cross-game workers, strict-FP32 EvalServer inference,
an immediate request collector, and a 96-request batch cap. Opponent-mix jobs
use the generator's supported 16-worker local/MPS evaluator path because the
EvalServer cannot yet route multiple checkpoint evaluators. The
schema-16 artifact remains byte-identical historical evidence and must not be
used as the current launch config.

The production learner starts natively from scratch with adapter v5,
`public_rule_state` and meaningful-history target gather enabled, no learner
checkpoint, and fresh optimizer state.
Its final value block is private, while terminal-value gradients enter the
shared encoders and first five blocks at 0.25 strength. This retains the
measured interference reduction without making the roughly 87.6% value-only
rows unable to teach the new representation.
The search teacher remains the deployed adapter-v2 checkpoint; only stored
learner features advance. The function-preserving v4 checkpoint upgrader
remains available for isolated compatibility experiments, but it is not the
production initialization path.

The checkpoint-initialized legacy one-dose/iteration executor must refuse this
production learner. The full retrain must construct a native v5 model rather
than relabeling or resuming a v2/v3 checkpoint.

The same authority now binds the fresh model construction and physical
execution separately from the logical 512-row global batch. The model enables both
structured-action residual paths, public-card counts, and ordered meaningful
public history with target-entity gather at construction time. History is
retained at the adapter-v5 cap of 64 events. Execution is exactly 8 B200 ranks
× 64 rows × one accumulation step; no launcher may reinterpret the logical
global batch of 512 as an arbitrary topology.

`tools/a1_scratch_train.py` authenticates the admitted post-wave composite and
renders the exact native-v5, bias-free 35M command plus a digest-bound planning
receipt. It is intentionally plan-only today. The current candidate recipe is
epoch-bound (`epochs=3`, `max_steps=0`) with fresh AdamW, 250-step warmup,
cosine decay, BF16, and the sealed symmetry/history relabeling flags. The
candidate global batch of 512 provides roughly eight times as many optimizer
updates as the rejected fine-tune-sized global batch of 4096, but the execution
topology still records `optimization_schedule_status=unresolved` and
`go_authorized=false`. Until a complete scratch-optimizer schedule authority is
reviewed—including the 0.25 shared-value routing and AdamW/cosine dose—the planner
refuses its explicit `--go` switch and `train_bc` rejects its child marker
before data loading; every planning receipt is diagnostic-only and
non-promotion-eligible. Once commissioned, the same path executes the 8-rank
learner and retains a digest-bound checkpoint for every epoch.

Generate through the sealed pre-wave control plane. For a direct lane command:

```bash
"$PY" tools/generate.py \
  --config configs/generation/coherent_public_n128.schema21.json \
  --guard configs/guards/a1_generation_coherent_public_n128_adaptive256_forced_value_v3.json \
  --checkpoint "$CHECKPOINT" --out-dir "$OUT" \
  --base-seed "$BASE_SEED" --games "$GAMES" --workers "$WORKERS" \
  --claim-label "$CLAIM_ID"
```

`tools/generate_gumbel_selfplay_data.py` remains the internal historical replay
executor. New launches must not address its experiment-by-flag interface
directly; the schema-20 config above is the complete science contract.

Post-wave admission must prove every worker used teacher v2 and emitted learner
rows v5, with the legacy `adapter_version` row column equal to the learner
identity. Forced rows retain `policy_weight_multiplier=0` and
`value_weight_multiplier=1` in the immutable shard. Training continues to use
equal per-game value mass and retains both forced `ROLL` and `END_TURN` states
at value weight 1.0. Their actions remain policy-inactive, but both boundaries
are valid value evidence; reducing either weight requires a causal ablation.

The five-percent reliability slice is selected by a stable hash of audit seed,
game seed, and decision index. Its duplicate search uses independent
Gumbel/chance/belief streams and never selects the live move. This slice and
the preserved completed-Q/visit fields are calibration evidence only:
production confidence weighting and both global and per-game surprise
sampling remain disabled until selector coverage and a weighting rule are
separately authenticated.

The learner also binds `--phase-weights PLAY_TURN=4.0`. On the admitted
959,142-row coherent corpus, equal-per-game policy weighting otherwise assigns
only 34.16% of policy objective mass to ordinary `PLAY_TURN` decisions and
65.84% to opening, discard, and robber prompts. The 4x multiplier restores
66.49% `PLAY_TURN` mass, matching the 66.08% strategic-turn share of the
historically successful selected-dose corpus while retaining supervision for
every mandatory multi-action prompt.

Policy phase repair does not silently alter the value objective:
`--value-phase-weights none` leaves value supervision unweighted across phases.
The old checkpoint-initialized 32-step frontier is retained as diagnostic
history, not copied into the native scratch schedule. The scratch planner is
epoch-bound, and remains non-executable and non-promotion-eligible while its
optimizer schedule authority is unresolved.

The native scratch launcher must also render the four admission fields from
`learner.training_recipe` exactly: diagnostics and policy/value interference
every 16 batches, at least two observations, and the complete commissioned
module list. `train_bc.py` verifies those observations before writing the
terminal checkpoint. A run is refused if any required module has a missing,
zero, non-finite, or malformed gradient/update signal, or if the global
policy/value trunk geometry probe is absent or malformed.
