# Next wave: coherent public-belief n128/adaptive-n256

This is the versioned next-wave recipe. It does not reinterpret or modify any
issued A1 generation guard, contract, seed claim, or corpus. The generation
config has `games=0` and `checkpoint=null` deliberately: checkpoint identity,
output path, seed range, lane quota, and worker count are operational inputs
that every real lane must pass explicitly.

## Generation operator

The teacher uses one sanitized two-player public-belief tree. It does not split
the budget across PIMC particles and it never clone-searches authoritative
hidden truth. Normal roots use n128/n16 playout-cap randomization; roots with at
least 40 legal actions always receive n256. D6 averaging begins at width 20.
Mandatory one-action prompts are recorded for trajectory and terminal-value
learning but skip neural MCTS target work. The exploration clock advances only
on non-forced choices: temperature 1.0 for the first 40 choices, 0.1 through
choice 99, then argmax.

Run one lane with explicit operational values (the fleet renderer may emit the
same argv once per pinned GPU):

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
  --n-full-wide 256 --n-full-wide-threshold 40 --wide-roots-always-full \
  --c-visit 50.0 --c-scale 0.1 --max-decisions 600 --max-depth 80 \
  --temperature-clock nonforced_choice --temperature-decisions 40 \
  --temperature-high 1.0 --temperature-low 0.0 \
  --late-temperature-decisions 100 --late-temperature 0.1 \
  --public-observation --coherent-public-belief-search \
  --no-information-set-search --no-belief-chance-spectra \
  --determinization-particles 1 --determinization-min-simulations 32 \
  --correct-rust-chance-spectra --lazy-interior-chance \
  --symmetry-averaged-eval --symmetry-averaged-eval-threshold 20 \
  --native-mcts-hot-loop --forced-root-target-mode trajectory_only \
  --rust-featurize --eval-cache-size 0 \
  --dump-config "$OUT/config.json" --config-purpose a1-next-wave-coherent-public-v1
```

The new guard intentionally requires the science flags in argv even though the
typed config supplies the same values. This prevents a stale fleet wrapper from
silently inheriting parser defaults. A lane is merge-compatible only when its
resolved `config.json` differs from its siblings solely in declared operational
identity fields.

## Learner operator

Use the existing function-preserving target-gather upgrade and canonical
one-dose executor. Every candidate starts independently from the current
champion; never use a candidate from this wave as another candidate's parent.
The one-dose executor enforces fresh Adam (`resume_optimizer=false`) and the
fixed global batch/dose. The selected loss delta is in
`one_dose_action_gather_overrides.json`: policy mass remains zero on forced
rows, value loss is 0.25, END_TURN forced values receive 0.1x, ROLL receives
0.25x, and every unlisted forced type (including DISCARD_RESOURCE) retains the
global 1.0x value weight.

Create an exactly function-preserving initializer and receipt:

```bash
CHAMPION=/path/to/current-champion.pt
UPGRADED=/fresh/path/champion.action-target-gather.pt
UPGRADE_RECEIPT=/fresh/path/action-target-gather.receipt.json

"$PY" tools/f69_upgrade_checkpoint_config.py \
  --in-checkpoint "$CHAMPION" --out-checkpoint "$UPGRADED" \
  --flags gather --device cuda:0 --seed 1
"$PY" tools/a1_function_preserving_upgrade.py \
  --source "$CHAMPION" --upgraded "$UPGRADED" \
  --module entity_graph.action_target_gather.v1 \
  --output "$UPGRADE_RECEIPT"
```

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
OVERRIDES="$(tr -d '\n' < configs/experiments/next_wave/one_dose_action_gather_overrides.json)"
LOCK_SHA="sha256:$(sha256sum "$LOCK" | awk '{print $1}')"
CODE_SHA="$("$PY" - "$LOCK" <<'PY'
import json
from pathlib import Path
import sys

from tools.a1_one_dose_train import _current_ablation_code_binding

lock = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(_current_ablation_code_binding(lock)["code_tree_sha256"])
PY
)"

"$PY" tools/a1_one_dose_train.py \
  --lock "$LOCK" --data "$COMPOSITE" \
  --composite-build-receipt "$COMPOSITE_RECEIPT" \
  --architecture-upgrade-receipt "$UPGRADE_RECEIPT" \
  --topology b200-8gpu-ddp --gpu 0 --ddp-canary-receipt "$DDP_CANARY" \
  --ablation-id coherent-public-action-gather-v1 \
  --recipe-overrides-json "$OVERRIDES" \
  --ablation-code-tree-sha256 "$CODE_SHA" \
  --reviewed-lock-file-sha256 "$LOCK_SHA" \
  --checkpoint "$CHECKPOINT_OUT" --report "$REPORT_OUT" \
  --receipt "$TRAIN_RECEIPT" --python "$PY"
```

That invocation is the executor's plan rendering mode. Review it, then repeat
the identical command with `--go` appended. This is not candidate chaining:
the architecture receipt always names the champion bytes, the lock names the
same parent, and optimizer state is not inherited.

## Evaluation and loop closure

Evaluate candidate and incumbent with the same public operator used to create
the data. Single-action prompts use `trajectory_only` here too: it changes no
move, while removing roughly half of the prompt-level neural work. Use paired
seeds and seat swaps; distribute disjoint pair shards across H100 hosts and feed
their outputs to the existing aggregate/SPRT path.

```bash
CANDIDATE=/path/to/candidate.pt
INCUMBENT=/path/to/current-champion.pt
EVAL_OUT=/fresh/path/candidate-vs-incumbent.json
DEVICES=cuda:0,cuda:1,cuda:2,cuda:3

"$PY" tools/gumbel_search_cross_net_h2h.py \
  --candidate "$CANDIDATE" --baseline "$INCUMBENT" --out "$EVAL_OUT" \
  --pairs 200 --workers 32 --devices "$DEVICES" --base-seed __EVAL_SEED__ \
  --max-decisions 600 --max-depth 80 \
  --n-full 128 --n-full-wide 256 --n-full-wide-threshold 40 \
  --wide-roots-always-full \
  --c-visit 50.0 --c-scale 0.1 --correct-rust-chance-spectra \
  --lazy-interior-chance --map-kind BASE \
  --symmetry-averaged-eval --symmetry-averaged-eval-threshold 20 \
  --public-observation --coherent-public-belief-search \
  --no-information-set-search --no-belief-chance-spectra \
  --native-mcts-hot-loop --evaluator-rust-featurize \
  --forced-root-target-mode trajectory_only
```

Promote only the checkpoint bytes that clear the paired candidate-versus-
incumbent decision and the existing phase/regression panels. A rejection starts
a new independent learner dose from the same incumbent, not from the rejected
candidate. A promotion atomically changes the champion identity; the next data
wave then repeats this exact generation -> one-dose training -> paired
evaluation sequence with fresh seed ranges.
