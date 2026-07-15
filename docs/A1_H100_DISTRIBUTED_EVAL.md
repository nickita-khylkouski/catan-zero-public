# A1 distributed H100 evaluation

Training stays on the selected eight-GPU B200 host. Evaluation can run on the
full 64-H100 fleet and is controlled directly from that B200; checkpoint and report traffic never
passes through an operator laptop.

`tools/fleet/a1_h100_eval_fleet.py` provides a fail-closed SSH backend today and
can render a Ray cluster specification without installing or starting Ray.
Capacity is allocated per physical GPU: each of the four 8-GPU hosts receives
twice the work of each 4-GPU host. New CLI plans use the coherent-public
operator: one public-belief tree (no information-set/PIMC search), n128 with
adaptive n256 from legal width 20, D6 averaging from width 20, one public
belief state, `trajectory_only` forced roots, native MCTS, and native Rust
featurization. `c_scale` is role-bound: the current v5
candidate and incumbent identities both use 0.10. The 0.03 default remains
only for replaying an older sealed identity and must not silently override a
post-promotion role.

Future plans default to 16 evaluator workers per GPU.  A counterbalanced,
full-game n128 B200 packing run completed 128/128 games with zero truncations
or errors and found 16 workers improved combined work-normalized throughput by
19.4% over 8 workers; both independent seed cohorts agreed.  This changes only
newly rendered plans.  Historical sealed plans retain their recorded worker
count.  See
[`EVAL_PACKING_B200_FULL_GAME_20260711.json`](evidence/EVAL_PACKING_B200_FULL_GAME_20260711.json).

Copy `configs/a1_h100_eval_fleet.example.json` to the gitignored
`configs/a1_h100_eval_fleet.json` on the B200 and fill private addresses. Then:

Use a separate immutable evaluation clone/worktree on the B200 and every H100;
do not edit the sealed deployment at `/home/ubuntu/catan-zero-v1`. Point
`remote_repo` at that evaluation tree and `remote_python` at the sealed venv.
The controller pins `PYTHONPATH` to the evaluation tree and proves the imported
`catan_zero` package comes from it, so the venv supplies dependencies without
silently importing the older sealed source tree.

```bash
PY=.venv/bin/python
CTL=tools/fleet/a1_h100_eval_fleet.py
M=configs/a1_h100_eval_fleet.json

$PY "$CTL" --manifest "$M" plan \
  --candidate /immutable/a1/candidate.pt \
  --champion /immutable/champion/champion.pt \
  --candidate-parent /immutable/champion/champion.pt \
  --registry /immutable/champion/champion_registry.json \
  --operator-mode coherent_public \
  --candidate-c-scale 0.10 --champion-c-scale 0.10 \
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

The mode is recorded in the immutable plan and command hashes. Use
`--operator-mode legacy_pimc` only to replay or deliberately compare against
the historical four-determinization information-set operator; old sealed plans
without an operator field are interpreted as that legacy mode.

The default `promotion_parent` mode requires the authenticated candidate
parent/init checkpoint, the internal baseline, and the registry's
`generator_champion` agent identity to name the same bytes. A diagnostic panel
against an older checkpoint must explicitly use
`--comparison-mode historical_comparison` and
`--historical-comparison-reason ...`; such a plan is sealed
`promotion_eligible=false` and promotion artifact construction rejects it.

Before the full plan, use `--scope canary` with 24 internal pairs and 12
external pairs. That exercises every GPU on `c1` (4 GPUs) and `h100-8a` (8
GPUs), two pairs per internal lane and two matched pairs per external cohort.
It is a separate immutable plan with separate VAL-only ranges. After both
canary phases collect cleanly, create the independent `--scope full` 600/500
plan shown above; never extend or reinterpret the canary plan as the full gate.

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
