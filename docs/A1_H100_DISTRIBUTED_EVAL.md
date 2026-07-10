# A1 distributed H100 evaluation

Training stays on the selected B200. Evaluation runs on the 40-H100 fleet and
is controlled directly from that B200; checkpoint and report traffic never
passes through an operator laptop.

`tools/fleet/a1_h100_eval_fleet.py` provides a fail-closed SSH backend today and
can render a Ray cluster specification without installing or starting Ray.
Capacity is allocated per physical GPU: the two 8-GPU hosts receive twice the
work of each 4-GPU host. The sealed evaluator recipe is n128, c-scale 0.03,
sigma 0.98, public-observation information-set search with four
determinizations, and D6 averaging from width 20.

Copy `configs/a1_h100_eval_fleet.example.json` to the gitignored
`configs/a1_h100_eval_fleet.json` on the B200 and fill private addresses. Then:

```bash
PY=.venv/bin/python
CTL=tools/fleet/a1_h100_eval_fleet.py
M=configs/a1_h100_eval_fleet.json

$PY "$CTL" --manifest "$M" plan \
  --candidate /immutable/a1/candidate.pt \
  --champion /immutable/champion/champion.pt \
  --internal-base-seed <VAL_ONLY_INTERNAL_BASE> \
  --external-base-seed <VAL_ONLY_EXTERNAL_BASE> \
  --iteration-id a1-infoset-n128-v133 \
  --internal-pairs 600 --external-pairs 500 \
  --out /immutable/a1/eval.plan.json

$PY "$CTL" --manifest "$M" launch --plan /immutable/a1/eval.plan.json \
  --phase internal --dry-run
$PY "$CTL" --manifest "$M" launch --plan /immutable/a1/eval.plan.json \
  --phase internal --go
$PY "$CTL" --manifest "$M" status --plan /immutable/a1/eval.plan.json \
  --phase internal
$PY "$CTL" --manifest "$M" collect --plan /immutable/a1/eval.plan.json \
  --phase internal --output-dir /immutable/a1/evaluation
```

Run the external phase the same way after internal H2H. Adjacent GPU lanes run
candidate and incumbent against `catanatron_value` on identical seed cohorts;
collection emits separate pooled candidate and incumbent reports. `resume`
only restarts missing, failed, or stale shards. Internal shards replay their
same interval; external shards resume atomic per-game artifacts.

Every `--go` operation proves it is running on a B200-only origin, stages
hash-qualified checkpoints directly to each H100 host, verifies the remote Git
commit, evaluator/launcher source hashes, checkpoint hashes, H100 model, and
declared GPU count, then launches one detached evaluator per GPU. Collection
rechecks transfer hashes and pools raw games only after semantic config,
checkpoint, pairing, and per-shard statistics replay.

To inspect the optional Ray topology without changing the fleet:

```bash
$PY "$CTL" --manifest "$M" ray-config --plan /immutable/a1/eval.plan.json \
  --out /immutable/a1/ray-cluster-spec.json
```

The generated head advertises zero B200 GPUs, each worker advertises its real
4- or 8-H100 capacity, and each future evaluator actor requires one H100. SSH
remains the production backend until Ray is installed and its lifecycle is
managed explicitly.
