# Next wave: coherent public-belief n128/adaptive-n256

> **2026-07-15 diagnostic status:** this directory remains the immutable record
> of the executable coherent-public experiment, but its learner treatments are
> not current production selections. `ROLL=0.25`, `END_TURN=0.1`, the 4x
> public-card residual LR, and surprise weighting were bundled without causal
> isolation. Large execution is paused under
> `docs/audits/A1_RL_SOFTWARE_DIAGNOSIS_20260715.md` and
> `docs/plans/A1_REPRESENTATION_VALUE_RECOVERY_PLAN_20260715.md`. Preserve this
> contract's issued artifacts as historical; do not reinterpret them using the
> repaired learner recipe below. New retraining locks must bind the current
> `science.contract.json` bytes.

This is the versioned next-wave recipe. It does not reinterpret or modify any
issued A1 generation guard, contract, seed claim, or corpus. The generation
config has `games=0` and `checkpoint=null` deliberately: checkpoint identity,
output path, seed range, lane quota, and worker count are operational inputs
that every real lane must pass explicitly.

`science.contract.json` is the single machine-readable authority consumed by
generation sealing, one-dose learning, fleet evaluation, and promotion. Its
adaptive width-20 dose is provisional until the causal teacher campaign is
aggregated and adopted. No coherent production lock can be sealed while the
status is provisional. Adopt the campaign winner (base n128, width 20, or
width 40) with:

```bash
python3 tools/a1_current_science_contract.py adopt-teacher \
  --report /path/to/teacher-operator-report.json \
  --receipt /path/to/teacher-operator-adoption.receipt.json
```

That transaction verifies the report against the exact provisional contract
bytes and updates only `n_full_wide`, `n_full_wide_threshold`, and
`wide_roots_always_full` in the science contract, draft template, typed
generator config, and generator guard. The receipt records the before/after
contract identities.

## Generation operator

The teacher uses one sanitized two-player public-belief tree. It does not split
the budget across PIMC particles and it never clone-searches authoritative
hidden truth. Normal roots use n128/n16 playout-cap randomization; roots with at
least 20 legal actions always receive n256. D6 averaging begins at width 20.
One-action engine transitions are applied directly and emit no learner row.
Public mandatory prompts with multiple choices (initial placement, discard,
robber movement, and Road Building placement) always receive full search. The
exploration clock advances only on real choices: temperature 1.0 for the first
40 choices, 0.1 through choice 99, then argmax.

The following width-20 lane is illustrative while the contract is provisional.
After adoption, use the sealed fleet renderer described below; it emits the
selected base/width-20/width-40 argv once per pinned GPU:

```bash
set -euo pipefail
REPO=/path/to/catan-zero-public
PY=/path/to/python
CHECKPOINT=/path/to/current-champion.pt
OUT=/path/to/new/output/lane-id
BASE_SEED=__DISJOINT_LEDGERED_SEED__
GAMES=__LANE_GAME_QUOTA__
WORKERS=__MEASURED_WORKERS_PER_GPU__
CLAIM_ID=__UNIQUE_LEDGER_CLAIM_ID__

cd "$REPO"
ulimit -n 65536
"$PY" tools/generate_gumbel_selfplay_data.py \
  --config configs/experiments/next_wave/coherent_public_n128_adaptive256.schema13.json \
  --prelaunch-guard-config configs/guards/a1_generation_coherent_public_n128_adaptive256_v1.json \
  --checkpoint "$CHECKPOINT" --out-dir "$OUT" \
  --base-seed "$BASE_SEED" --games "$GAMES" --workers "$WORKERS" \
  --ledger-claim-label "$CLAIM_ID" --device cuda \
  --track 2p_no_trade --vps-to-win 10 --format npz --score-actions \
  --n-full 128 --n-fast 16 --p-full 0.25 \
  --n-full-wide 256 --n-full-wide-threshold 20 --wide-roots-always-full \
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
  --no-record-automatic-transitions \
  --meaningful-public-history --event-history-limit 32 \
  --rust-featurize --eval-cache-size 0 \
  --dump-config "$OUT/config.json" --config-purpose a1-next-wave-coherent-public-v1
```

The new guard intentionally requires the science flags in argv even though the
typed config supplies the same values. This prevents a stale fleet wrapper from
silently inheriting parser defaults. A lane is merge-compatible only when its
resolved `config.json` differs from its siblings solely in declared operational
identity fields.

For a real fleet wave, do not hand-fan the lane command above. Resolve the
checked-in `configs/experiments/a1_pre_wave_contract.template.json`, then use
the existing sealed control plane end to end:

```bash
DRAFT=/path/to/resolved.coherent-public.draft.json
LOCK=/fresh/path/contract.lock.json
RENDER_DIR=/fresh/path/render
CLAIM_RECEIPT=/fresh/path/seed-claim.receipt.json
EXEC_RECEIPT=/fresh/path/executor.receipt.json
HOSTS=/path/to/private/fleet-hosts.json

"$PY" tools/a1_pre_wave_contract.py seal --draft "$DRAFT" --out "$LOCK"
"$PY" tools/a1_pre_wave_contract.py render --lock "$LOCK" --out-dir "$RENDER_DIR"
"$PY" tools/a1_pre_wave_contract.py claim --lock "$LOCK" \
  --render "$RENDER_DIR/commands.json" --receipt "$CLAIM_RECEIPT"
"$PY" tools/fleet/a1_production_executor.py run --lock "$LOCK" \
  --render "$RENDER_DIR/commands.json" --hosts "$HOSTS" \
  --receipt "$EXEC_RECEIPT" --go
```

The sealed renderer now emits the same automatic-transition and bounded
meaningful-history flags as the typed recipe, and its post-wave audit accepts
non-empty authenticated event tensors. Adaptive-n256 rows are excluded from
the p=0.25 randomized-root rate test rather than causing the whole wave to be
rejected.

## Learner operator

Use the combined public-card-count + zero-gated meaningful-history
function-preserving upgrade and canonical one-dose executor for fresh
next-wave shards. The action-target gather experiment was neutral and is not
bundled into this learner. The card residual receives 2p opponent resource
counts derived from public conservation and a public dev-card posterior; it
never receives engine-secret identities. The existing-data-compatible schema
uses the identical public entity-token transform in training and serving;
rare legacy counter saturation fails the resource slice closed to zero rather
than creating train/serve skew. Historical/replay rows
are recomputed from their existing public entity columns at batch load, so no
corpus regeneration is required. Every candidate starts independently from the current
champion; never use a candidate from this wave as another candidate's parent.
The one-dose executor enforces fresh Adam (`resume_optimizer=false`) and the
fixed 8-rank batch (`512/rank`, global `4096`) and exactly 128 optimizer steps
(`524,288` sampled rows). The selected optimizer/loss recipe is sealed in
`science.contract.json` (including LR `6e-5` and 16 warmup steps): policy
mass remains zero on forced rows, value loss is 0.25, END_TURN forced values
receive 0.1x, ROLL receives 0.25x, every unlisted forced type (including
DISCARD_RESOURCE) retains the global 1.0x value weight, and only the new
zero-initialized card residual uses the 4x LR group; the 640-parameter history
gate remains in the ordinary trunk group. Value rows are normalized to equal
total mass per game, so long games cannot dominate the scalar outcome target.
Policy-active roots use capped per-game surprise weighting; the redistribution
preserves each game's total sample mass and therefore does not let one long or
pathological game dominate the dose.

Create an exactly function-preserving initializer and receipt:

```bash
CHAMPION=/path/to/current-champion.pt
UPGRADED=/fresh/path/champion.public-cards.pt
UPGRADE_RECEIPT=/fresh/path/public-cards.receipt.json

"$PY" tools/f69_upgrade_checkpoint_config.py \
  --in-checkpoint "$CHAMPION" --out-checkpoint "$UPGRADED" \
  --flags structured_action_value,card_count_v2,meaningful_history \
  --device cuda:0 --seed 1
"$PY" tools/a1_function_preserving_upgrade.py \
  --source "$CHAMPION" --upgraded "$UPGRADED" \
  --module entity_graph.static_action_residual+legal_action_value_residual+public_card_count_features+meaningful_public_history.v3 \
  --output "$UPGRADE_RECEIPT"
```

The v2 card residual is bias-free. This is not cosmetic: when public-card
evidence is zero, its residual is structurally zero even after training. The
legacy v1 module remains replayable for already-issued checkpoints but is not
the fresh-wave default.

The existing-corpus diagnostic is intentionally card-only because its event
payloads are authenticated empty. For that one diagnostic use `--flags
card_count`, receipt module `entity_graph.public_card_count_features.v1`, and
`--crop-authenticated-empty-event-history`. Never pretend those old rows train
the history gate. Fresh coherent-wave shards use the combined module above.

Then run the exact one-dose transaction on the 8-B200 topology. The lock and
composite must bind this same champion as producer/parent, and the corpus must
carry the coherent public-belief target-information regime:

```bash
LOCK=/path/to/new-wave/contract.lock.json
COMPOSITE=/path/to/new-wave/composite/descriptor.json
COMPOSITE_RECEIPT=/path/to/new-wave/composite/build.receipt.json
DDP_CANARY=/path/to/current/b200-8gpu-ddp.canary.json
CHECKPOINT_OUT=/fresh/path/candidate.pt
REPORT_OUT=/fresh/path/train.report.json
TRAIN_RECEIPT=/fresh/path/train.receipt.json

STATE=/fresh/path/iteration.state.json
TURN=/fresh/path/flywheel.turn.json
HANDOFF=/path/to/post-promotion-handoff.json
CAMPAIGN=/path/to/post-promotion-generation-campaign.json
AUDIT=/path/to/new-wave/post-wave.audit.json

"$PY" tools/a1_iteration_orchestrator.py initialize-next \
  --state "$STATE" --turn "$TURN" \
  --post-promotion-handoff "$HANDOFF" \
  --generation-campaign "$CAMPAIGN" --generation-audit "$AUDIT" \
  --lock "$LOCK" --data "$COMPOSITE" \
  --composite-build-receipt "$COMPOSITE_RECEIPT" \
  --learner-parent "$CHAMPION" --evaluation-parent "$CHAMPION" \
  --initializer "$UPGRADED" \
  --architecture-upgrade-receipt "$UPGRADE_RECEIPT" \
  --topology b200-8gpu-ddp --gpu 0 --ddp-canary-receipt "$DDP_CANARY" \
  --checkpoint "$CHECKPOINT_OUT" --report "$REPORT_OUT" \
  --training-receipt "$TRAIN_RECEIPT" --python "$PY"

"$PY" tools/a1_iteration_orchestrator.py dose-dry --state "$STATE"
"$PY" tools/a1_iteration_orchestrator.py dose-go --state "$STATE"
```

`initialize-next` now carries the complete one-dose topology/recipe into the
existing immutable turn state; `dose-dry` and `dose-go` render and execute that
same binding. This is not candidate chaining: the architecture receipt always
names the champion bytes, the lock names the same parent, and optimizer state
is not inherited.

The repaired short-dose LR/warmup, equal per-game value mass, 4x public-card
LR, exact per-game policy surprise weighting, and typed forced-row value
weights are fields of the sealed coherent-production learner recipe. They are
deliberately **not** passed through the generic ablation interface: that
interface is diagnostic only and its receipts are promotion-ineligible.

## Evaluation and loop closure

Evaluate candidate and incumbent with the same public operator used to create
the data. Single-action prompts are direct transitions here too: they change no
move and require neither model. Use paired seeds and seat swaps; distribute
disjoint pair shards across H100 hosts and feed their outputs to the existing
aggregate/SPRT path.

```bash
CANDIDATE=/path/to/candidate.pt
INCUMBENT=/path/to/current-champion.pt
EVAL_OUT=/fresh/path/candidate-vs-incumbent.json
DEVICES=cuda:0,cuda:1,cuda:2,cuda:3

"$PY" tools/gumbel_search_cross_net_h2h.py \
  --candidate "$CANDIDATE" --baseline "$INCUMBENT" --out "$EVAL_OUT" \
  --pairs 200 --workers 32 --devices "$DEVICES" --base-seed __EVAL_SEED__ \
  --max-decisions 600 --max-depth 80 \
  --n-full 128 --n-full-wide 256 --n-full-wide-threshold 20 \
  --wide-roots-always-full \
  --c-visit 50.0 --c-scale 0.1 --sigma-eval 0.79 \
  --max-root-candidates 16 --max-root-candidates-wide 54 \
  --wide-candidates-threshold 24 --correct-rust-chance-spectra \
  --lazy-interior-chance --map-kind BASE \
  --symmetry-averaged-eval --symmetry-averaged-eval-threshold 20 \
  --public-observation --coherent-public-belief-search \
  --no-information-set-search --no-belief-chance-spectra \
  --determinization-particles 1 --determinization-min-simulations 32 \
  --gameplay-policy-aggregation mean_improved_policy \
  --native-mcts-hot-loop --evaluator-rust-featurize \
  --forced-root-target-mode trajectory_only
```

The command above is the exact single-host role operator. Production
evaluation is sharded by the existing fleet controller, whose CLI default is
already `coherent_public` but is stated explicitly here so the plan records it:

```bash
EVAL_MANIFEST=/path/to/private/eval-fleet.json
REGISTRY=/path/to/champion-registry.json
EVAL_PLAN=/fresh/path/eval.plan.json

"$PY" tools/fleet/a1_h100_eval_fleet.py --manifest "$EVAL_MANIFEST" plan \
  --candidate "$CANDIDATE" --champion "$INCUMBENT" \
  --candidate-parent "$INCUMBENT" --registry "$REGISTRY" \
  --operator-mode coherent_public \
  --candidate-c-scale 0.1 --champion-c-scale 0.1 \
  --internal-base-seed __VAL_ONLY_INTERNAL_SEED__ \
  --external-base-seed __VAL_ONLY_EXTERNAL_SEED__ \
  --iteration-id __ITERATION_ID__ --out "$EVAL_PLAN"

"$PY" tools/fleet/a1_h100_eval_fleet.py --manifest "$EVAL_MANIFEST" launch \
  --plan "$EVAL_PLAN" --phase internal --go
```

Promote only the checkpoint bytes that clear the paired candidate-versus-
incumbent decision and the existing phase/regression panels. A rejection starts
a new independent learner dose from the same incumbent, not from the rejected
candidate. A promotion atomically changes the champion identity; the next data
wave then repeats this exact generation -> one-dose training -> paired
evaluation sequence with fresh seed ranges.
