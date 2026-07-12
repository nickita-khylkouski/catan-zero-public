# 35M Teacher Training Plan

> **Historical plan, not a current production recipe.** Its retained
> `loser_sample_weight=0.3` commands document old runs. That setting was found to
> suppress valid MCTS targets on losing trajectories; current production uses
> `1.0`, and the trainer refuses an unacknowledged value below one.

Date: 2026-06-28

Formal goal-tool status:

```text
The thread goal is active and references this plan. Keep the goal active until
the teacher-phase gates are actually proven by scoreboard evidence.
```

Active goal:

```text
Execute the 35M teacher-training plan before PPO and push the 35M xdim_graph
model toward teacher-dominant strength.

Primary target:
  >80% win rate against catanatron_ab3 on full 10-VP 2p_no_trade games.

Required strong-opponent targets:
  strongly positive vs jsettlers_lite
  strongly positive vs catanatron_value
  materially improved vs catanatron_search / value_rollout_search
  measurable progress vs catanatron_ab4 and catanatron_ab5

This is an aspirational target, not a guarantee from one run. The operating
goal is to keep climbing by repeatedly improving data quality, teacher strength,
training correctness, and scoreboard selection until the 35M model is genuinely
stronger than the teachers it is learning from.
```

This is still the teacher-training phase. PPO does not start until this plan's
promotion gates pass.

## Operating Doctrine

```text
1. PPO stays off.
2. B200 GPUs should always be doing useful work:
   training the 35M model,
   evaluating checkpoints,
   or running strong-opponent scoreboards.
3. Modal 600 CPU is the primary bulk data factory after a passing smoke test.
4. A100 and GH200 CPU boxes continue generating and curating teacher data.
5. Training data must be high quality, not just large:
   soft targets,
   mixed-seat games,
   outcome/value targets,
   low forced-action policy weight,
   no invalid teacher actions,
   no silent schema mismatch.
6. Teacher strength should increase when needed:
   catanatron_ab3 for bulk pressure,
   catanatron_ab4 / catanatron_ab5 for higher-quality labels,
   catanatron_search / value_rollout_search for soft search targets,
   jsettlers_lite for a different planning style.
7. Every few cycles, spawn/read subagent audits for:
   architecture bugs,
   data-pipeline bugs,
   orchestration/resource bugs,
   scoreboard/eval bugs.
   Fix high-impact findings before spending large B200 runs.
8. Promote by scoreboard strength, not BC accuracy.
```

## Execution Loop

```text
repeat:
  1. Generate teacher data:
     Modal 600 CPU:
       mixed-seat soft 2p10 runs as primary bulk source
     A100/GH200:
       parallel 2p10 softmix, AB3-only, and targeted AB4/AB5/search runs

  2. Validate data:
     invalid_teacher_actions must be 0
     soft_score_fraction should be high for search/value/AB data
     catanatron_ab* rows must have target_score_source=ab_root
     target_policy coverage should be tracked
     final_public_vp_fraction and final_actual_vp_fraction must be near 1.0
     truncated/outcome rows must be separated
     forced moves must not dominate policy loss

	  3. Curate:
	     drop/downweight roll and forced-action noise
	     for strict 35M BC corpora, use:
	       --forced-keep-prob 0.0
	       --roll-keep-prob 0.0
	       --drop-forced-in-important-phases
	     preserve important phases:
	       initial_build
	       main_turn
       robber
       discard
     if phase diagnostics stall, upweight weak phases in BC:
       robber ~= 3.0
       initial_build ~= 2.0
       discard ~= 1.5
     keep enough AB4/AB5/search labels to raise the ceiling

  4. Train 35M on B200:
     init from current best 35M checkpoint
     xdim_graph only
     policy loss = soft distillation + hard CE blend
     value loss = terminal win/loss
     final-VP auxiliary loss only when true VP targets exist
     prefer target_scores when available for temperature retuning

  5. Evaluate every epoch:
     AB3
     search/value_rollout_search
     jsettlers_lite
     catanatron_value
     AB4/AB5 probes
     random/heuristic sanity checks

  6. Promote only if scoreboard improves:
     50% vs AB3 = internal best_bc
     65% vs AB3 = strong BC
     80% vs AB3 plus strong search/JSettlers = teacher-dominant target

  7. If scoreboard stalls:
     add DAgger-style student-state labeling
     increase AB4/AB5/search label share
     audit action mapping and architecture
     fix bugs before scaling more compute
```

Production 35M data gate:

```bash
PYTHONPATH=tools:src python tools/report_teacher_data_quality.py \
  --data runs/teacher/CURATED_DATASET \
  --track 2p_no_trade \
  --vps-to-win 10 \
  --production-35m-teacher
```

`--strict-35m-teacher` is the smoke/provenance gate. It fails closed on
old/no-provenance AB data and requires complete terminal outcomes, actual VP
targets, strong `target_scores` coverage, very low forced-action rate after
curation, and `ab_root` score provenance for every `catanatron_ab*` teacher
present in the dataset. `--production-35m-teacher` includes the strict gate and
adds production-scale row counts: millions of samples, heavy AB4/AB5/search
coverage, JSettlers-style coverage, required phase coverage, and minimum
soft-label coverage over legal candidates. Use the production gate for the next
real 35M BC run; use strict only for small smoke datasets.

## Current State

Model:

```text
Architecture: xdim_graph
Parameters:   33,469,570
Width:        768
Graph tokens: 32
Graph layers: 4
Track now:    2p_no_trade, 10 VP
```

Best measured checkpoint so far:

```text
Base 35M vs catanatron_ab3:
  34.47% win rate over 1,024 games

AB3-finetuned epoch 1 vs catanatron_ab3:
  435 / 1,024 wins
  42.48% win rate
  0 illegal actions

Modal AB4/search soft-mix DDP run, epoch 1:
  data samples: 2,221,874
  soft_score_fraction: 100%
  invalid teacher actions: 0
  training top-1: 77.38%
  training top-3: 90.58%
  loss: 1.6797
  weak phases:
    robber top-1: 18.95%
    initial_build top-1: 42.21%

Modal AB4/search soft-mix DDP run, epoch 2:
  training top-1: 78.53%
  training top-3: 91.10%
  loss: 1.6281
  weak phases still stalled:
    robber top-1: 18.97%
    initial_build top-1: 41.97%
```

This is a real improvement, but not enough. The near-term target is:

```text
>=50% vs catanatron_ab3
competitive vs catanatron_search and jsettlers_lite
no regression vs value/heuristic/random
```

Immediate corrective action:

```text
Do not rely on more epochs alone to fix robber/opening.
Next 35M round should sync the newer train_bc.py and use phase weights, e.g.
  --phase-weights robber=3.0,initial_build=2.0,discard=1.5
Then compare scoreboard, not only BC accuracy.
```

## Hard Rule

Do not start PPO until the teacher model has passed the promotion gate below.

Teacher phase is complete only when:

```text
1. The 35M model beats random and heuristic cleanly.
2. The 35M model is >=80% vs catanatron_ab3 over a meaningful sample.
3. The 35M model is strongly positive vs catanatron_search and jsettlers_lite.
4. It has no illegal actions.
5. Seat-conditioned win rates are not broken.
6. BC diagnostics do not show catastrophic phase failure.
7. The value head has been pretrained, not left random.
```

Tiered checkpoint labels:

```text
>=50% vs catanatron_ab3:
  internal best_bc candidate only

>=65% vs catanatron_ab3:
  strong teacher-cloned checkpoint

>=80% vs catanatron_ab3 plus strong search/JSettlers results:
  teacher-dominant checkpoint; PPO can be considered after final gates
```

## Machine Roles

### B200 Box

SSH:

```bash
ssh -i $HOME/.ssh/gpu_access_ed25519 \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile=/tmp/catanatron-b200-known_hosts \
  ubuntu@B200
```

Use for:

```text
Primary: 35M model training / fine-tuning
Primary: checkpoint scoreboard evaluation
Secondary: curation if GPUs are idle and CPU has room
```

Plan:

```text
Default:
  one 2-GPU DDP train_bc.py job across GPU0+GPU1

When not training:
  run GPU scoreboards intentionally, or immediately start the next queued train
  job once a curated dataset is ready.

Only split GPU0/GPU1 into separate jobs when:
  CUDA_VISIBLE_DEVICES pins each job to disjoint devices
  --allow-concurrent-bc is passed deliberately
  CPU/data-loader pressure is known to be safe
```

Do not leave a B200 GPU idle for long. If no training job is ready, run
scoreboard eval for the latest checkpoint.

### A100 Box

SSH:

```bash
ssh -i $HOME/.ssh/gpu_access_ed25519 \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=accept-new \
  ubuntu@a100-legacy
```

Use for:

```text
Primary: CPU teacher data generation
Primary: curation/reporting
Secondary: scoreboard eval if B200 is saturated
```

Current shape:

```text
~194 generation processes
softmix 2p10 generation
soft AB3 generation
```

A100 GPUs are not the bottleneck for teacher rollout. Their CPUs are useful.

### GH200

SSH:

```bash
ssh -i $HOME/.ssh/gpu_access_ed25519 \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile=/tmp/catanatron-gh200-new-known_hosts \
  ubuntu@gh200
```

Use for:

```text
Primary: CPU teacher data generation
Secondary: curation
```

The GH200 GPU is not needed for current teacher rollout generation unless we
move neural inference or training there.

### Modal 600 CPU

Use as the primary bulk teacher-data source after smoke tests pass.

Current status:

```text
Patched smoke test must show:
  completed parts > 0
  samples > 0
  invalid_teacher_actions = 0
  soft_score_fraction > 0
  mixed_seats = true
```

When fixed:

```text
75 containers
8 CPU/container
600 CPU max
short parts only: 128-256 games/container
commit partial shards every few completed chunks
fresh run_name by default; use resume only deliberately
```

Main 2p launch:

```bash
python3 -m modal run tools/modal_teacher_factory.py::launch_600 \
  --run-name teacher_2p10_mixed_soft_600cpu_$(date -u +%Y%m%d_%H%M) \
  --containers 75 \
  --games-per-container 256 \
  --cpu-workers 8 \
  --seed 606280746 \
  --fmt npz_zst \
  --commit-every-chunks 8 \
  --mixed-seats
```

High-quality AB4/AB5/search launch:

```bash
python3 -m modal run tools/modal_teacher_factory.py::launch_600_ab45 \
  --run-name teacher_2p10_ab45_search_600cpu_$(date -u +%Y%m%d_%H%M) \
  --containers 75 \
  --games-per-container 128 \
  --cpu-workers 8 \
  --teachers catanatron_ab4,catanatron_ab5,value_rollout_search,catanatron_ab3,catanatron_value,jsettlers_lite \
  --seed 60629700 \
  --fmt npz_zst \
  --commit-every-chunks 8 \
  --mixed-seats
```

Status:

```bash
python3 -m modal run tools/modal_teacher_factory.py::status \
  --run-name teacher_2p10_mixed_soft_600cpu_YYYYMMDD_HHMM
```

Tail / attach a locally supervised Modal launch:

```bash
MODAL_RUN=teacher_2p10_ab45_search_600cpu_YYYYMMDD_HHMM

# Only works for locally supervised launches that write this log.
tail -f runs/ops/"$MODAL_RUN".log

# If launched under tmux:
tmux ls | grep "$MODAL_RUN"
tmux attach -t catan_modal_"$MODAL_RUN"

# Robust fallback for any Modal run recorded in current_modal_teacher_runs.txt:
python3 tools/cluster_teacher_status.py --modal-run "$MODAL_RUN" --modal-timeout 90
```

Record every active or recently completed Modal teacher run so ordinary status
and watchdog commands include it automatically:

```bash
mkdir -p runs/ops
echo teacher_2p10_ab45_search_600cpu_YYYYMMDD_HHMM \
  >> runs/ops/current_modal_teacher_runs.txt
```

Status must report both completed and partial work. During long 600-CPU runs,
`parts_complete` may stay flat while `parts_partial`, `observed_games`, and
`observed_samples` climb. Treat a run as stuck only if both completed and
partial counters stop moving and no Modal task logs are updating.

Do not run this Modal map entrypoint with `--detach` unless a durable Modal queue
or remote supervisor owns the whole run. Detached local entrypoints can leave the
map incomplete while the status pointer still looks current.

## Phase 1 - Finish Current Scoreboards

Purpose:

```text
Pick the best checkpoint among the current 35M AB3 fine-tune epochs.
Do not assume epoch 3 is best just because imitation accuracy is higher.
```

Current B200 evals:

```text
runs/scoreboards/xdim_graph_ab3_more_ft_epoch2_20260628_0706
runs/scoreboards/xdim_graph_ab3_more_ft_epoch3_20260628_0707
```

Poll:

```bash
cd /home/ubuntu/catan-zero

nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits

for d in \
  runs/scoreboards/xdim_graph_ab3_more_ft_epoch2_20260628_0706 \
  runs/scoreboards/xdim_graph_ab3_more_ft_epoch3_20260628_0707
do
  echo "==== $d ===="
  for f in "$d"/*.json; do [ -f "$f" ] && echo "---- $f ----" && cat "$f"; done
  for log in "$d"/*.log; do [ -f "$log" ] && echo "$log $(wc -l < "$log")" && tail -n 2 "$log"; done
done
```

Decision:

```text
If epoch 2 or 3 beats epoch 1 vs AB3:
  use that checkpoint as the next init checkpoint.

If epoch 2/3 regress vs epoch 1:
  keep epoch 1 as best teacher checkpoint.

If all are close:
  prefer the checkpoint with better AB3 + no regression vs jsettlers/value.
```

## Live Monitoring Commands

Purpose:

```text
Give a fast terminal view of whether teacher generation, training, and eval are
alive without opening every log manually.
```

Local one-shot / watch command:

```bash
cd <verified-local-checkout>

python3 tools/cluster_teacher_status.py

watch -n 30 'python3 tools/cluster_teacher_status.py'

# Active Modal runs are read from runs/ops/current_modal_teacher_runs.txt.
# You can also include or override explicit Modal runs in the same terminal view:
CATAN_ZERO_MODAL_RUNS=teacher_2p10_ab45_search_600cpu_YYYYMMDD_HHMM \
  watch -n 60 'python3 tools/cluster_teacher_status.py --modal-timeout 90'

# Portable fallback for macOS shells without watch:
while true; do
  clear
  CATAN_ZERO_MODAL_RUNS=teacher_2p10_ab45_search_600cpu_YYYYMMDD_HHMM \
    python3 tools/cluster_teacher_status.py --modal-timeout 90
  sleep 60
done

# Or pass one or more runs explicitly:
python3 tools/cluster_teacher_status.py \
  --modal-run teacher_2p10_ab45_search_600cpu_YYYYMMDD_HHMM
```

Watchdog command:

```bash
cd <verified-local-checkout>

CATAN_ZERO_MODAL_RUNS=teacher_2p10_ab45_search_600cpu_YYYYMMDD_HHMM \
  python3 tools/teacher_phase_watchdog.py \
  --modal-run teacher_2p10_ab45_search_600cpu_YYYYMMDD_HHMM \
  --modal-timeout 90 \
  --status-timeout 25

# Continuous local watchdog:
while true; do
  clear
  CATAN_ZERO_MODAL_RUNS=teacher_2p10_ab45_search_600cpu_YYYYMMDD_HHMM \
    python3 tools/teacher_phase_watchdog.py \
    --modal-run teacher_2p10_ab45_search_600cpu_YYYYMMDD_HHMM \
    --modal-timeout 90 \
    --status-timeout 25
  sleep 120
done
```

The watchdog exits nonzero if PPO is detected, if the status command itself
fails, or if B200 has useful BC/eval/curation processes but all GPUs stay under
10% utilization for the configured stale window. It warns when B200 is not doing
BC/eval/curation, when A100/GH200 have no teacher generation process, when one
B200 GPU is idle while useful work exists, when stale empty logs are present, or
when Modal `observed_samples` stops advancing across checks.

B200 status:

```bash
ssh -i $HOME/.ssh/gpu_access_ed25519 \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile=/tmp/catanatron-b200-known_hosts \
  ubuntu@B200 '
cd /home/ubuntu/catan-zero
echo "== GPU =="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits
echo "== process counts =="
printf "bc=%s eval=%s ppo=%s gen=%s\n" \
  "$(pgrep -af "tools/train_bc.py|torchrun.*train_bc.py" | grep -v pgrep | wc -l)" \
  "$(pgrep -af "tools/evaluate_scoreboard.py" | grep -v pgrep | wc -l)" \
  "$(pgrep -af "train_selfplay_gpu.py|train_ppo.py" | grep -v pgrep | wc -l)" \
  "$(pgrep -af "tools/generate_teacher_data.py" | grep -v pgrep | wc -l)"
echo "== latest scoreboards =="
find runs/scoreboards -mindepth 1 -maxdepth 1 -type d -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -5 | cut -d" " -f2-
'
```

Do not run two multi-GPU BC jobs on the same B200 box. `tools/train_bc.py`
uses `/tmp/catan_zero_train_bc.lock` by default to reject accidental concurrent
launches; pass `--allow-concurrent-bc` only when devices are explicitly
partitioned and the launch command pins each job to disjoint GPUs.

Current B200 train tail:

```bash
cd <verified-local-checkout>
python3 tools/teacher_phase_tail.py b200-train --follow
```

Manual equivalent:

```bash
ssh -i $HOME/.ssh/gpu_access_ed25519 \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile=/tmp/catanatron-b200-known_hosts \
  ubuntu@B200 '
cd /home/ubuntu/catan-zero
RUN=$(cat runs/teacher/current_b200_35m_teacher_run.txt)
tail -f "$RUN/train.log"
'
```

Current B200 scoreboard tail:

```bash
cd <verified-local-checkout>
python3 tools/teacher_phase_tail.py b200-scoreboard
```

Manual equivalent:

```bash
ssh -i $HOME/.ssh/gpu_access_ed25519 \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile=/tmp/catanatron-b200-known_hosts \
  ubuntu@B200 '
cd /home/ubuntu/catan-zero
OUT=$(cat runs/scoreboards/current_b200_scoreboard_run.txt 2>/dev/null || true)
test -n "$OUT" || OUT=$(find runs/scoreboards -mindepth 1 -maxdepth 1 -type d -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -1 | cut -d" " -f2-)
echo "scoreboard=$OUT"
for f in "$OUT"/*.log; do [ -f "$f" ] && echo "---- $f" && tail -n 20 "$f"; done
for f in "$OUT"/*.json; do [ -f "$f" ] && echo "---- $f" && cat "$f"; done
'
```

A100 data-generation status:

```bash
cd <verified-local-checkout>
python3 tools/teacher_phase_tail.py a100-generate --follow
```

Manual status:

```bash
ssh -i $HOME/.ssh/gpu_access_ed25519 \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=accept-new \
  ubuntu@a100-legacy '
cd /home/ubuntu/catan-zero
echo "== load/processes =="
uptime
printf "gen=%s curate=%s eval=%s ppo=%s\n" \
  "$(pgrep -af "tools/generate_teacher_data.py" | grep -v pgrep | wc -l)" \
  "$(pgrep -af "tools/curate_teacher_data.py" | grep -v pgrep | wc -l)" \
  "$(pgrep -af "tools/evaluate_scoreboard.py" | grep -v pgrep | wc -l)" \
  "$(pgrep -af "train_selfplay_gpu.py|train_ppo.py" | grep -v pgrep | wc -l)"
echo "== latest teacher runs =="
find runs/teacher -mindepth 1 -maxdepth 1 -type d -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -8 | cut -d" " -f2-
'
```

GH200 data-generation status:

```bash
cd <verified-local-checkout>
python3 tools/teacher_phase_tail.py gh200-generate --follow
```

Manual status:

```bash
ssh -i $HOME/.ssh/gpu_access_ed25519 \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile=/tmp/catanatron-gh200-new-known_hosts \
  ubuntu@gh200 '
cd /home/ubuntu/catan-zero
uptime
printf "gen=%s curate=%s ppo=%s\n" \
  "$(pgrep -af "tools/generate_teacher_data.py" | grep -v pgrep | wc -l)" \
  "$(pgrep -af "tools/curate_teacher_data.py" | grep -v pgrep | wc -l)" \
  "$(pgrep -af "train_selfplay_gpu.py|train_ppo.py" | grep -v pgrep | wc -l)"
find runs/teacher -mindepth 1 -maxdepth 1 -type d -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -8 | cut -d" " -f2-
'
```

Hard expectation during this plan:

```text
ppo=0 on every machine.
If ppo is nonzero, inspect before continuing.
```

## Phase 2 - Generate High-Quality Soft Teacher Data

Purpose:

```text
Build a high-quality supervised dataset for the 35M model.
The data must contain soft labels, outcome fields, and final VP fields.
```

Dataset requirements:

```text
target_policy present
target_scores present where teacher supports it
winner present
final_public_vps present
phase present
legal_action_ids valid
action_taken always in legal_action_ids
```

Active 2p soft data jobs:

```text
A100:
  runs/teacher/softmix_2p10_a100_20260628_0709
  target: 30,000 games
  workers: 144

A100:
  runs/teacher/soft_ab3_2p10_a100_20260628_0709
  target: 15,000 games
  workers: 48

GH200:
  runs/teacher/softmix_2p10_gh200_20260628_0709
  target: 50,000 games
  workers: 56
```

Target before next 35M training run:

```text
Minimum:
  20,000 total new soft games
  ~5M new decision samples

Preferred:
  80,000-100,000 total new soft games
  ~20M+ new decision samples
```

Teacher mix:

```text
target softmix for all new runs:
  catanatron_ab3
  catanatron_ab3
  catanatron_value
  jsettlers_lite
  value_rollout_search
  value_rollout_search

AB3-only:
  catanatron_ab3

target high-quality depth teachers:
  catanatron_ab4
  catanatron_ab5
```

Note:

```text
Some already-running 20260628 jobs were launched more AB3-heavy. Keep those
shards, but future runs use the target softmix above plus a separate AB3-only
stream. This gives AB3 pressure without letting one style dominate everything.
```

Why:

```text
AB3-only helps target the current weakness.
Softmix prevents overfitting to one style.
Search/value teachers provide soft target distributions.
JSettlers adds a different planning style.
AB4/AB5 are used selectively for high-quality labels because they are slower.
```

Poll A100:

```bash
cd /home/ubuntu/catan-zero

echo gen=$(pgrep -af "tools/generate_teacher_data.py" | grep -v pgrep | wc -l)

for r in \
  runs/teacher/softmix_2p10_a100_20260628_0709 \
  runs/teacher/soft_ab3_2p10_a100_20260628_0709
do
  echo "==== $r ===="
  tail -n 5 "$r/generate.log"
  find "$r/teacher_data" -name "teacher_shard_*.npz*" | wc -l
  du -sh "$r"
done
```

Poll GH200:

```bash
cd /home/ubuntu/catan-zero

echo gen=$(pgrep -af "tools/generate_teacher_data.py" | grep -v pgrep | wc -l)

r=runs/teacher/softmix_2p10_gh200_20260628_0709
tail -n 5 "$r/generate.log"
find "$r/teacher_data" -name "teacher_shard_*.npz*" | wc -l
du -sh "$r"
```

Error checks:

```text
If progress log stops for >10 minutes:
  inspect process list and stderr/log tail

If shards are not increasing:
  check disk space with df -h
  check Python traceback in generate.log

If invalid teacher actions appear:
  stop using that run
  inspect action mapping before training

If generated shards lack target_policy/target_scores:
  discard from soft training set or treat as hard-only data
```

## Phase 3 - Curate The Data

Purpose:

```text
Remove obvious noise before training the 35M model.
Keep hard decisions; down-weight/drop forced and roll-only samples.
Preserve soft targets.
```

Run per machine first:

```bash
cd /home/ubuntu/catan-zero

RUN=runs/teacher/curated_soft_2p10_v1_$(date -u +%Y%m%d_%H%M)
mkdir -p "$RUN"

nohup env PYTHONPATH=/home/ubuntu/catan-zero/src:/home/ubuntu/catan-zero/tools \
  python3 tools/curate_teacher_data.py \
  --data runs/teacher/softmix_2p10_a100_20260628_0709/teacher_data \
  --data runs/teacher/soft_ab3_2p10_a100_20260628_0709/teacher_data \
  --out "$RUN/teacher_data" \
  --format npz_zst \
  --shard-size 100000 \
  --seed 60628800 \
  --forced-keep-prob 0.0 \
  --drop-forced-in-important-phases \
  --roll-keep-prob 0.0 \
  > "$RUN/curate.log" 2>&1 &
```

Expected curation behavior:

```text
forced action rows: mostly dropped or downweighted
roll phase rows: heavily downsampled
initial_build/main_turn/robber/discard: preserved
target_policy and target_scores: preserved
```

Quality report:

```bash
cd /home/ubuntu/catan-zero

env PYTHONPATH=/home/ubuntu/catan-zero/src:/home/ubuntu/catan-zero/tools \
  python3 tools/report_teacher_data_quality.py \
  --data "$RUN/teacher_data" \
  --track 2p_no_trade \
  --vps-to-win 10 \
  --max-invalid-teacher-actions 0 \
  --min-soft-policy-fraction 0.50 \
  --min-soft-score-fraction 0.50 \
  --min-outcome-fraction 0.95 \
  --min-final-public-vp-fraction 0.95 \
  --min-final-actual-vp-fraction 0.95 \
  --max-forced-action-fraction 0.30 \
  --max-truncated-fraction 0.05 \
  --min-teacher-samples catanatron_ab4=10000,catanatron_ab5=10000,value_rollout_search=10000 \
  --min-soft-score-by-teacher catanatron_ab4=0.50,catanatron_ab5=0.50,value_rollout_search=0.90 \
  --min-clean-outcome-by-teacher catanatron_ab4=0.90,catanatron_ab5=0.90,value_rollout_search=0.90 \
  --min-phase-samples main_turn=10000,initial_build=1000,robber=1000,discard=1000 \
  --out "$RUN/quality.json"

	cat "$RUN/teacher_data/curation_report.json"
	cat "$RUN/quality.json"
```

Do not reference GH200 paths from the A100 filesystem unless that raw data has
already been transferred. The safe flow is:

```text
1. Curate A100-local data on A100.
2. Curate GH200-local data on GH200.
3. Export Modal data from the Modal volume.
4. Transfer curated outputs to B200.
5. Merge/train on B200.
```

Hard failure conditions:

```text
invalid teacher actions > 0:
  do not train

soft target rows unexpectedly zero:
  do not call it soft training

forced-action fraction remains >30%:
  curate again with lower forced keep probability

robber/discard/trade phases tiny:
  generate targeted data or adjust curation
```

## Phase 4 - Transfer Curated Data To B200

Purpose:

```text
B200 is the training box. It should train from curated data only.
```

Preferred transfer:

```bash
SRC_RUN=curated_soft_2p10_v1_YYYYMMDD_HHMM

rsync -av --progress \
  -e 'ssh -i $HOME/.ssh/gpu_access_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new' \
  ubuntu@a100-legacy:/home/ubuntu/catan-zero/runs/teacher/"$SRC_RUN"/ \
  /tmp/"$SRC_RUN"/
```

Then push to B200:

```bash
rsync -av --progress \
  -e 'ssh -i $HOME/.ssh/gpu_access_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/catanatron-b200-known_hosts' \
  /tmp/"$SRC_RUN"/ \
  ubuntu@B200:/home/ubuntu/catan-zero/runs/teacher/"$SRC_RUN"/
```

Modal volume export:

```bash
MODAL_RUN=teacher_2p10_ab45_search_600cpu_YYYYMMDD_HHMM

SCRATCH=/tmp/catan_modal_exports/$MODAL_RUN
rm -rf "$SCRATCH"
mkdir -p "$SCRATCH"
cd "$SCRATCH"

python3 -m modal volume get \
  catan-zero-teacher-data \
  "$MODAL_RUN/parts"

test -d "$SCRATCH/parts" || { echo "missing Modal parts export"; exit 1; }
find "$SCRATCH/parts" -name manifest.json | grep -q . || {
  echo "missing Modal part manifests"
  exit 1
}

rsync -av --progress \
  -e 'ssh -i $HOME/.ssh/gpu_access_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/catanatron-b200-known_hosts' \
  "$SCRATCH/parts" \
  ubuntu@B200:/home/ubuntu/catan-zero/runs/teacher/"$MODAL_RUN"/
```

If `/tmp` is small, use a larger local scratch path instead. Large Modal exports
can exceed laptop `/tmp` capacity.

Then curate the Modal export on B200 before using it for training:

```bash
ssh -i $HOME/.ssh/gpu_access_ed25519 \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile=/tmp/catanatron-b200-known_hosts \
  ubuntu@B200 '
cd /home/ubuntu/catan-zero
set -euo pipefail
RAW=runs/teacher/teacher_2p10_ab45_search_600cpu_YYYYMMDD_HHMM
CURATED=runs/teacher/curated_teacher_2p10_ab45_search_600cpu_YYYYMMDD_HHMM
mkdir -p "$CURATED"
env PYTHONPATH=/home/ubuntu/catan-zero/src:/home/ubuntu/catan-zero/tools \
  .venv/bin/python3.11 tools/curate_teacher_data.py \
  --data "$RAW" \
  --out "$CURATED/teacher_data" \
  --format npz_zst \
  --shard-size 100000 \
  --seed 60628810 \
  --forced-keep-prob 0.0 \
  --drop-forced-in-important-phases \
  --roll-keep-prob 0.0 \
  > "$CURATED/curate.log" 2>&1
env PYTHONPATH=/home/ubuntu/catan-zero/src:/home/ubuntu/catan-zero/tools \
  .venv/bin/python3.11 tools/report_teacher_data_quality.py \
	  --data "$CURATED/teacher_data" \
	  --track 2p_no_trade \
	  --vps-to-win 10 \
	  --production-35m-teacher \
	  --out "$CURATED/quality.json"
	'
```

`tools/curate_teacher_data.py` accepts a Modal run root only when the nested
manifests match the safe Modal layout `parts/part_*/manifest.json`. It still
rejects arbitrary directories with several unrelated nested manifests.

If direct box-to-box SSH keys are configured, transfer directly:

```bash
ssh ubuntu@a100-legacy '
rsync -av --progress \
  /home/ubuntu/catan-zero/runs/teacher/curated_soft_2p10_v1/ \
  ubuntu@B200:/home/ubuntu/catan-zero/runs/teacher/curated_soft_2p10_v1/
'
```

Do not use a wildcard source with a trailing slash for curated run transfer; it
can collapse several runs into one destination and overwrite manifests. Do not
use a local command with two remote paths unless it has already been tested. The
safe default is local relay through `/tmp` or an rsync launched from the source
box to the destination box.

Verify on B200:

```bash
cd /home/ubuntu/catan-zero
find runs/teacher/"$SRC_RUN"/teacher_data -name "teacher_shard_*.npz*" | wc -l
du -sh runs/teacher/"$SRC_RUN"
cat runs/teacher/"$SRC_RUN"/teacher_data/manifest.json
```

Before B200 training, create or choose one final training dataset directory.
`tools/train_bc.py` accepts one `--data` path, so Modal, A100, and GH200 curated
outputs must be explicitly merged/curated into a single `teacher_data` directory
or trained in separate scheduled rounds. Do not assume several transferred run
directories will be consumed automatically.

Concrete merged-production dataset build on B200:

```bash
ssh -i $HOME/.ssh/gpu_access_ed25519 \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile=/tmp/catanatron-b200-known_hosts \
  ubuntu@B200 '
cd /home/ubuntu/catan-zero
set -euo pipefail

MERGED_RUN=runs/teacher/merged_35m_teacher_2p10_$(date -u +%Y%m%d_%H%M)
mkdir -p "$MERGED_RUN"

MODAL_CURATED=runs/teacher/curated_teacher_2p10_ab45_search_600cpu_YYYYMMDD_HHMM/teacher_data
A100_CURATED=runs/teacher/curated_actualvp_ab4_search_a100_YYYYMMDD_HHMM/teacher_data
GH200_CURATED=runs/teacher/curated_actualvp_ab4_search_gh200_YYYYMMDD_HHMM/teacher_data

for d in "$MODAL_CURATED" "$A100_CURATED" "$GH200_CURATED"; do
  test -d "$d" || { echo "missing curated input: $d"; exit 1; }
done

env PYTHONPATH=/home/ubuntu/catan-zero/src:/home/ubuntu/catan-zero/tools \
  .venv/bin/python3.11 tools/curate_teacher_data.py \
  --data "$MODAL_CURATED" \
  --data "$A100_CURATED" \
  --data "$GH200_CURATED" \
  --out "$MERGED_RUN/teacher_data" \
  --format npz_zst \
  --shard-size 100000 \
  --seed 60635000 \
  --forced-keep-prob 0.0 \
  --drop-forced-in-important-phases \
  --roll-keep-prob 0.0 \
  > "$MERGED_RUN/merge_curate.log" 2>&1

env PYTHONPATH=/home/ubuntu/catan-zero/src:/home/ubuntu/catan-zero/tools \
  .venv/bin/python3.11 tools/report_teacher_data_quality.py \
  --data "$MERGED_RUN/teacher_data" \
  --track 2p_no_trade \
  --vps-to-win 10 \
	--production-35m-teacher \
	--out "$MERGED_RUN/quality.json"

echo "$MERGED_RUN" > runs/teacher/current_35m_training_dataset.txt
cat "$MERGED_RUN/quality.json"
'
```

Only launch the next serious 35M B200 run from:

```bash
DATA=$(cat runs/teacher/current_35m_training_dataset.txt)/teacher_data
test -d "$DATA" || { echo "missing DATA=$DATA"; exit 1; }
```

If the production gate fails, do not bypass it by pointing `train_bc.py` at an
older strict or smoke dataset. Either add more AB4/AB5/search/JSettlers data,
fix bad provenance, or train a deliberately labeled small diagnostic run.

## Phase 5 - Train The 35M Teacher Model

Purpose:

```text
Train policy + value head together so the checkpoint is a real PPO warm-start.
```

Training loss:

```text
policy loss:
  0.7 * KL/CE against soft teacher distribution
  0.3 * hard CE on teacher action

value loss:
  V(s) -> final outcome for acting player
  win = +1
  loss = -1

final VP loss:
  final_vp_head(s) -> final public VP / vps_to_win

sample weights:
  forced legal action: 0.1
  winner trajectory: 1.0
  loser trajectory: 0.3
```

Initial checkpoint:

```text
Use the best of:
  epoch1 AB3 fine-tune
  epoch2 AB3 fine-tune
  epoch3 AB3 fine-tune

Pick based on scoreboard, not imitation accuracy.
```

Single-GPU command if data is not huge:

```bash
cd /home/ubuntu/catan-zero
set -euo pipefail

DATA=$(cat runs/teacher/current_35m_training_dataset.txt)/teacher_data
BEST=$(cat runs/teacher/current_best_35m_checkpoint.txt)
test -d "$DATA" || { echo "missing DATA=$DATA"; exit 1; }
test -f "$BEST" || { echo "missing BEST=$BEST"; exit 1; }

RUN=runs/teacher/xdim_graph_soft_outcome_2p10_$(date -u +%Y%m%d_%H%M)
mkdir -p "$RUN"

CUDA_VISIBLE_DEVICES=0 nohup env PYTHONPATH=/home/ubuntu/catan-zero/src:/home/ubuntu/catan-zero/tools \
  .venv/bin/python3.11 tools/train_bc.py \
  --arch xdim_graph \
  --data "$DATA" \
  --track 2p_no_trade \
  --vps-to-win 10 \
  --epochs 5 \
  --batch-size 8192 \
  --hidden-size 768 \
  --graph-tokens 32 \
  --graph-layers 4 \
  --lr 0.00005 \
  --soft-target-temperature 0.7 \
  --soft-target-weight 0.7 \
  --forced-action-weight 0.1 \
  --phase-weights robber=3.0,initial_build=2.0,discard=1.5 \
  --winner-sample-weight 1.0 \
  --loser-sample-weight 0.3 \
  --value-loss-weight 0.25 \
  --final-vp-loss-weight 0.05 \
  --teacher-weights catanatron_ab5=1.8,catanatron_ab4=1.6,value_rollout_search=1.5,catanatron_value=1.1,jsettlers_lite=0.8,catanatron_ab3=1.0 \
  --init-checkpoint "$BEST" \
  --require-production-35m-teacher \
  --save-each-epoch \
  --no-ddp-find-unused-parameters \
	--checkpoint "$RUN/xdim_graph_soft_outcome_2p10.pt" \
	--report "$RUN/xdim_graph_soft_outcome_2p10.json" \
	--device auto \
	> "$RUN/train.log" 2>&1 &
PID=$!
echo "$PID" > "$RUN/train.pid"
sleep 10
kill -0 "$PID" || { echo "train exited immediately"; tail -n 80 "$RUN/train.log"; exit 1; }
grep -Ei "error|traceback|unrecognized arguments" "$RUN/train.log" && {
  echo "train log shows startup failure"
  exit 1
}
echo "$RUN" > runs/teacher/current_b200_35m_teacher_run.txt
```

Two-GPU DDP command if data is large:

```bash
cd /home/ubuntu/catan-zero
set -euo pipefail

DATA=$(cat runs/teacher/current_35m_training_dataset.txt)/teacher_data
BEST=$(cat runs/teacher/current_best_35m_checkpoint.txt)
test -d "$DATA" || { echo "missing DATA=$DATA"; exit 1; }
test -f "$BEST" || { echo "missing BEST=$BEST"; exit 1; }

RUN=runs/teacher/xdim_graph_soft_outcome_2p10_ddp_$(date -u +%Y%m%d_%H%M)
mkdir -p "$RUN"

nohup env PYTHONPATH=/home/ubuntu/catan-zero/src:/home/ubuntu/catan-zero/tools \
  .venv/bin/python3.11 -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/train_bc.py \
  --arch xdim_graph \
  --data "$DATA" \
  --track 2p_no_trade \
  --vps-to-win 10 \
  --epochs 5 \
  --batch-size 8192 \
  --hidden-size 768 \
  --graph-tokens 32 \
  --graph-layers 4 \
  --lr 0.00005 \
  --soft-target-temperature 0.7 \
  --soft-target-weight 0.7 \
  --forced-action-weight 0.1 \
  --phase-weights robber=3.0,initial_build=2.0,discard=1.5 \
  --winner-sample-weight 1.0 \
  --loser-sample-weight 0.3 \
  --value-loss-weight 0.25 \
  --final-vp-loss-weight 0.05 \
  --teacher-weights catanatron_ab5=1.8,catanatron_ab4=1.6,value_rollout_search=1.5,catanatron_value=1.1,jsettlers_lite=0.8,catanatron_ab3=1.0 \
  --init-checkpoint "$BEST" \
  --require-production-35m-teacher \
  --save-each-epoch \
  --no-ddp-find-unused-parameters \
	--checkpoint "$RUN/xdim_graph_soft_outcome_2p10.pt" \
	--report "$RUN/xdim_graph_soft_outcome_2p10.json" \
	--device auto \
	> "$RUN/train.log" 2>&1 &
PID=$!
echo "$PID" > "$RUN/train.pid"
sleep 10
kill -0 "$PID" || { echo "train exited immediately"; tail -n 80 "$RUN/train.log"; exit 1; }
grep -Ei "error|traceback|unrecognized arguments" "$RUN/train.log" && {
  echo "train log shows startup failure"
  exit 1
}
echo "$RUN" > runs/teacher/current_b200_35m_teacher_run.txt
```

Training monitors:

```bash
tail -f "$RUN/train.log"
nvidia-smi dmon -s pucm
```

Standalone B200 tail command:

```bash
cd /home/ubuntu/catan-zero
RUN=$(cat runs/teacher/current_b200_35m_teacher_run.txt)
tail -f "$RUN/train.log"
```

Expected metrics:

```text
loss decreases
top1 accuracy stable or improves
top3 accuracy improves
value loss decreases
phase diagnostics not catastrophic
GPU memory stable
no NaN
no OOM
```

Stop and debug if:

```text
loss becomes NaN
value loss explodes
accuracy collapses by >10 points
GPU memory rises every batch
teacher action not present in legal candidates
phase diagnostics show 0% on important phases for multiple epochs
```

## Phase 6 - Evaluate Every Epoch

Purpose:

```text
The best epoch is selected by game strength, not training accuracy.
```

For each epoch checkpoint:

```bash
cd /home/ubuntu/catan-zero

EPOCH=runs/teacher/xdim_graph_soft_outcome_2p10/xdim_graph_soft_outcome_2p10_epoch0001.pt
test -f "$EPOCH" || { echo "missing EPOCH=$EPOCH"; exit 1; }
OUT=runs/scoreboards/soft_outcome_epoch0001_$(date -u +%Y%m%d_%H%M)
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES="" nohup env \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH=/home/ubuntu/catan-zero/src:/home/ubuntu/catan-zero/tools \
  python3 tools/evaluate_scoreboard.py \
  --candidate "$EPOCH" \
  --candidate-kind checkpoint \
  --games 2000 \
  --tracks 2p_no_trade \
  --opponents catanatron_ab3 \
  --workers 24 \
  --chunk-games 25 \
  --seed 60628901 \
  --vps-to-win 10 \
  --max-decisions 1200 \
  --device cpu \
  --out "$OUT/vs_ab3_2k.json" \
  > "$OUT/ab3.log" 2>&1 &

CUDA_VISIBLE_DEVICES="" nohup env \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH=/home/ubuntu/catan-zero/src:/home/ubuntu/catan-zero/tools \
  python3 tools/evaluate_scoreboard.py \
  --candidate "$EPOCH" \
  --candidate-kind checkpoint \
  --games 1000 \
  --tracks 2p_no_trade \
  --opponents random,heuristic,value,jsettlers_lite,catanatron_search,value_rollout_search,catanatron_ab4,catanatron_ab5 \
  --workers 24 \
  --chunk-games 25 \
  --seed 60628902 \
  --vps-to-win 10 \
  --max-decisions 1200 \
	--device cpu \
	--out "$OUT/vs_core_1k.json" \
	> "$OUT/core.log" 2>&1 &
echo "$OUT" > runs/scoreboards/current_b200_scoreboard_run.txt
```

Default scoreboard mode is CPU-only. CPU scoreboard launches must cap
OpenMP/BLAS/Torch thread pools to one thread per worker; otherwise even
`--workers 1` can consume dozens of CPU cores and stall the box. Run GPU
scoreboards only when B200
training is intentionally idle, and pass both an explicit CUDA device and
`--allow-gpu-workers` if using more than one worker. This prevents evaluation
workers from silently stealing the training GPUs.

When B200 is idle and a training dataset is not ready, split GPU scoreboards
explicitly so both GPUs do useful work:

```bash
cd /home/ubuntu/catan-zero

EPOCH=runs/teacher/xdim_graph_soft_outcome_2p10/xdim_graph_soft_outcome_2p10_epoch0001.pt
test -f "$EPOCH" || { echo "missing EPOCH=$EPOCH"; exit 1; }
OUT=runs/scoreboards/gpu_split_epoch0001_$(date -u +%Y%m%d_%H%M)
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 nohup env PYTHONPATH=/home/ubuntu/catan-zero/src:/home/ubuntu/catan-zero/tools \
  .venv/bin/python3.11 tools/evaluate_scoreboard.py \
  --candidate "$EPOCH" \
  --candidate-kind checkpoint \
  --games 2000 \
  --tracks 2p_no_trade \
  --opponents catanatron_ab3 \
  --workers 1 \
  --chunk-games 25 \
  --seed 60628901 \
  --vps-to-win 10 \
  --max-decisions 1200 \
  --device cuda:0 \
  --out "$OUT/vs_ab3_2k.json" \
  > "$OUT/ab3_gpu0.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup env PYTHONPATH=/home/ubuntu/catan-zero/src:/home/ubuntu/catan-zero/tools \
  .venv/bin/python3.11 tools/evaluate_scoreboard.py \
  --candidate "$EPOCH" \
  --candidate-kind checkpoint \
  --games 1000 \
  --tracks 2p_no_trade \
  --opponents random,heuristic,value,jsettlers_lite,catanatron_search,value_rollout_search,catanatron_ab4,catanatron_ab5 \
  --workers 1 \
  --chunk-games 25 \
  --seed 60628902 \
  --vps-to-win 10 \
  --max-decisions 1200 \
	--device cuda:0 \
	--out "$OUT/vs_core_1k.json" \
	> "$OUT/core_gpu1.log" 2>&1 &
echo "$OUT" > runs/scoreboards/current_b200_scoreboard_run.txt
```

In the second command, `CUDA_VISIBLE_DEVICES=1` remaps physical GPU1 to
process-local `cuda:0`; therefore `--device cuda:0` is intentional.

Minimum dev eval:

```text
2,000 games vs AB3
1,000 games each vs search/jsettlers/value/heuristic/random
```

Promotion eval:

```text
10,000 games vs AB3
5,000 games each vs search/jsettlers/value
```

Final teacher claim:

```text
20,000+ games vs AB3
10,000+ games vs search/jsettlers/value
```

Metrics to compare:

```text
win_rate
confidence_interval_95
moves_to_win
avg_vp_margin
avg_candidate_vp
seat_wins
illegal_action_count
timeouts_or_stuck_games
```

## Phase 7 - Promotion Gate

Promote a checkpoint as the teacher-phase champion if:

```text
illegal_action_count = 0
timeouts_or_stuck_games = 0
win_rate vs random >= 95%
win_rate vs heuristic >= 65%
win_rate vs value >= 50%
win_rate vs jsettlers_lite >= 65%
win_rate vs catanatron_ab3 >= 80%
no major seat collapse
avg_vp_margin improves vs previous champion
moves_to_win does not regress badly
```

Intermediate labels:

```text
internal best_bc:
  win_rate vs catanatron_ab3 >= 50%

strong BC:
  win_rate vs catanatron_ab3 >= 65%

teacher-dominant target:
  win_rate vs catanatron_ab3 >= 80%
  win_rate vs catanatron_search >= 65%
  win_rate vs jsettlers_lite >= 80%
```

If the checkpoint is better than old 35M but still below teacher strength:

```text
promote internally as best_bc
generate DAgger data from it
continue teacher training
do not start PPO
```

## Phase 8 - DAgger Loop

Purpose:

```text
Fix BC distribution shift.
The student must learn on states it reaches itself, not only teacher states.
```

DAgger plan:

```text
1. Roll out current 35M student.
2. At each student-visited state, ask teacher/search for label.
3. Save those labels with target_policy/target_scores.
4. Curate and mix into teacher dataset.
5. Fine-tune 35M again.
```

Initial DAgger target:

```text
10,000 games labeled by catanatron_search / catanatron_value / AB3
```

Training mix after DAgger:

```text
70% curated teacher data
30% DAgger student-state data
```

Do DAgger if:

```text
BC accuracy rises but scoreboard stalls
model loses to AB3 in weird midgame states
phase diagnostics show robber/discard/build timing weakness
```

## Phase 9 - 4p And Trade Data

Purpose:

```text
Avoid making a 2p-only specialist that fails on real Catan.
```

Do not block the current 2p target on this, but start collecting:

```text
4p_no_trade
4p_bank_trade
```

Modal command when Modal is stable:

```bash
/usr/local/bin/python3 -m modal run tools/modal_teacher_factory.py::launch_600_4p \
  --run-name teacher_4p_bank_trade_softmix_600cpu_v1 \
  --containers 75 \
  --games-per-container 128 \
  --cpu-workers 8 \
  --seed 60630700 \
  --fmt npz_zst \
  --track 4p_bank_trade
```

CPU-box command:

```bash
cd /home/ubuntu/catan-zero

RUN=runs/teacher/softmix_4p_bank_trade_$(date -u +%Y%m%d_%H%M)
mkdir -p "$RUN"

nohup env PYTHONPATH=/home/ubuntu/catan-zero/src:/home/ubuntu/catan-zero/tools \
  python3 tools/generate_teacher_data.py \
  --track 4p_bank_trade \
	--teachers catanatron_ab3,catanatron_value,jsettlers_lite,value_rollout_search,value_rollout_search \
	--mixed-seats \
	--mixed-seat-mode random \
	--games 20000 \
  --workers 56 \
  --chunk-games 1 \
  --seed 60630701 \
  --vps-to-win 10 \
  --max-decisions 1800 \
  --format npz_zst \
  --out "$RUN/teacher_data" \
  > "$RUN/generate.log" 2>&1 &
```

Do not mix 4p data into the 2p champion training blindly. Either:

```text
train a separate 4p checkpoint
or mix with a track embedding / track-specific eval
```

## Phase 10 - Error Catching Checklist

### Data Errors

Check after every generation run:

```text
invalid teacher actions = 0
teacher action in legal mask = true
legal mask nonempty
target_policy rows > 0 for soft teachers
target_scores rows > 0 for search/value teachers
winner field populated
final_public_vps shape valid
phase distribution sane
forced-action fraction not dominating curated data
```

Commands:

```bash
env PYTHONPATH=/home/ubuntu/catan-zero/src:/home/ubuntu/catan-zero/tools \
  python3 tools/report_teacher_data_quality.py \
  --data "$RUN/teacher_data" \
  --track 2p_no_trade \
  --vps-to-win 10 \
  --out "$RUN/quality.json"
```

### Training Errors

Stop and inspect if:

```text
loss NaN
value loss NaN
top1 accuracy collapses
top3 accuracy collapses
phase accuracy all zeros for important phase
CUDA OOM
memory leak across batches
```

### Eval Errors

Reject eval if:

```text
illegal_action_count > 0
timeouts_or_stuck_games > 0
seat_wins all from one seat
games_per_matchup too small
vps_to_win != 10 for serious gate
opponents do not include AB3/search/jsettlers
```

### Process Errors

No PPO during teacher phase:

```bash
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | \
  grep -E "train_selfplay_gpu.py|train_ppo.py" | \
  grep -v grep
```

Expected result during teacher phase:

```text
no matching PPO process
```

## Final Teacher-Phase Exit Criteria

We can move to PPO only after:

```text
1. A 35M checkpoint is selected by scoreboard, not BC accuracy.
2. It has pretrained policy and value heads.
3. It beats or matches the major teachers:
   catanatron_ab3
   jsettlers_lite
   catanatron_value
   ideally catanatron_search
4. It is evaluated on full 10-VP games.
5. It has no illegal actions or stuck games.
6. It has no severe seat bias.
7. Per-phase diagnostics do not show an obvious untrained phase.
```

If it still cannot cross AB3/search:

```text
do not start PPO
run DAgger
add AB3 soft targets
add targeted robber/discard/opening data
continue BC
```

## Summary

The training factory should run in this order:

```text
1. Finish current B200 scoreboards.
2. Continue A100/GH200 soft teacher generation.
3. Curate soft/outcome data.
4. Transfer curated data to B200.
5. Fine-tune 35M with:
   soft distillation
   hard CE blend
   outcome weighting
   value-head pretraining
   final-VP auxiliary loss
6. Score every epoch against strong opponents.
7. Promote only if it is honestly teacher-competitive.
8. If not teacher-competitive, run DAgger and continue.
9. Start PPO only after this succeeds.
```
