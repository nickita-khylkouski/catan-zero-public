# Active RL Experiments - 2026-06-27

Goal: improve the permissioned Colonist-style four-player Catan agent by promoting only checkpoints that beat the current internal champion under strict multi-opponent gates.

Current promoted internal champion:

```text
runs/self_play/champions/current_best_s9752_iter0002.pt
```

Promotion evidence:

```text
s9752_plain_lowkl_control.iter0002
strict g12 vs current_best_s4806_iter0002
candidate weighted win rate: 0.2115
old champion weighted win rate: 0.1731
decision: promote_candidate
```

This is still an internal prototype checkpoint, not evidence of Colonist #1 strength.

## 2026-06-27 13:46 PDT GPU Expansion

- 14:09 PDT update:
  `tools/ssh_gpu_fleet_controller.py` now has conservative SSH-GPU
  `refill-plan`/`refill` support. It plans from live polls, separates CPU and
  CUDA capacity, uses GPU memory/active-label evidence before launching, filters
  shell wrappers out of process counts, and uses a remote Python
  `subprocess.Popen(start_new_session=True)` starter for future launches.
- Safe refill launched two more V100-only remote trainers, both guarded
  shared-policy VRPO self-play from `s9752` with Q-advantage warmup/ramp gates,
  Expected-SARSA Q targets, EMA/old-policy KL, value clipping, and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:
  `s20400_selfplay_vrpo_guard_v100g3` and
  `s20401_selfplay_vrpo_guard_v100g6`.
- Fresh poll after refill: 28 remote trainers, 0 final checkpoints. V100 has
  21 active trainers (9 CUDA, 12 CPU) with all 8 GPU indices occupied; H100 has
  7 active trainers (3 CUDA, 4 CPU) with about 64.7 GiB used, so no H100 refill
  is safe. Interims visible: 3 on V100, 4 on H100.
- The latest refill plan is a no-op: CPU targets are full, V100 has no unused
  GPU index, and H100 is memory-blocked. No local training, local self-play,
  local grading, Claude-mini, or subagent work was used.

- Added two plain SSH GPU workers for remote-only CatanZero training:
  - `gpu-v100`: `ubuntu@gpu-v100`, 8x Tesla V100-SXM2-16GB, 88 CPU cores.
  - `gpu-h100`: `ubuntu@gpu-h100`, 1x NVIDIA H100 80GB HBM3, 26 CPU cores.
- Created a neutral SSH key for these hosts at
  `/Users/nickita/.ssh/gpu_access_ed25519`; only the `.pub` key was added to
  the hosts. Do not print or copy the private key.
- Synced the lean CatanZero repo plus
  `runs/self_play/champions/current_best_s9752_iter0002.pt` to
  `~/catan-zero-gpu` on both hosts. Local machine did not run Catan training,
  self-play, or game grading.
- Added CUDA device plumbing:
  `TorchPPOPolicy` now supports explicit `device`, `TorchPPOPolicy.load(...,
  device=...)` preserves CUDA placement, `create_ppo_policy(..., device=...)`
  forwards placement, and `tools/train_ppo.py --device cuda` can run policy
  updates/inference on GPU. Default remains CPU for existing GCP workflows.
- Remote environment status:
  - H100: Python 3.11, PyTorch `2.12.1+cu130`, CUDA available, H100 detected.
  - V100: Python 3.11, PyTorch replaced with `2.5.1+cu121` because the newest
    CUDA 13 wheel no longer supports V100 `sm70`; CUDA matmul and policy
    save/load smoke tests passed.
- Launched 44 detached remote GPU trainers, all from champion
  `current_best_s9752_iter0002.pt`, all with `--device cuda`:
  - V100 host: 32 trainers across 8 GPUs.
  - H100 host: 12 trainers on the H100.
  - Wave 1: 12 exploratory repair/anti-regression runs.
  - Wave 2: 32 larger runs, mostly actual shared-policy self-play
    (`learner-seats all`, `opponents self`) with EMA/KL/VRPO guardrails plus
    longer JSettlers/value-repair and strict-gate repair branches.
- Planned wave-2 training volume from remote manifests:
  `20,496` remote training games (`15,360` V100 + `5,136` H100). Wave 1 adds
  roughly `2,764` more, so the active GPU campaign is about `23k` remote
  training games before checkpoint diagnostic evals.
- Follow-up load correction: the initial 44 CUDA-trainer launch intentionally
  tested the host ceilings and overcommitted memory on the 16 GB V100 cards
  and the single H100. Failed lanes show CUDA OOM in their own logs and are no
  longer active; do not restart them at the same density.
- Stable active GPU-host campaign after pruning and replacement:
  40 remote trainers, about 21.8k planned remote training games:
  - V100 host: 30 active trainers, consisting of 18 CUDA survivors plus
    12 remote CPU self-play workers using otherwise idle CPU cores.
  - H100 host: 10 active trainers, consisting of 6 CUDA survivors plus
    4 remote CPU self-play workers.
  - The CPU lanes are remote-only (`--device cpu`) and exist to increase game
    generation without causing more CUDA OOM. Local machine still does not run
    Catan training or grading.
- Stable active labels include CUDA repair/anti-regression lanes
  (`s20000`-`s20006`, `s20050`, `s20053`, `s20116`-`s20121`, `s20150`,
  `s20151`, `s20154`, `s20156`, `s20200`-`s20203`) and CPU self-play lanes
  (`s20300`-`s20311`, `s20350`-`s20353`).
- Added `tools/ssh_gpu_fleet_controller.py` to poll these plain SSH GPU hosts
  and pull finished artifacts into `runs/self_play/gpu_imports` without using
  local training.
- Next rule: do not stop or overwrite these jobs. Poll with
  `tools/ssh_gpu_fleet_controller.py poll --output
  runs/self_play/ssh_gpu_poll_latest.json`. When final `.pt/.json` artifacts
  appear, pull them, then run strict grading remotely on GPU/VM workers before
  promoting anything.

## 2026-06-27 10:20 PDT Status

- No local self-play training, local game grading, Claude-mini, or subagent
  work was launched. Local work was controller review, syntax/unit tests,
  VM polling/status, primary-source research, and no-busy planning only.
- Safety fix landed in `tools/remote_fleet_autopilot.py`: busy-worker
  scheduling is now genuinely opt-in at the CLI parser level. Default
  autopilot cycles no longer plan gates on training-busy VMs or training on
  grade-busy VMs unless `--allow-training-busy-gates` or
  `--allow-grade-busy-training` is explicitly passed.
- VM-opening diagnostic tooling landed in `tools/gcp_fleet_controller.py`:
  `remote-opening-eval` and `plan-remote-opening-evals` can run
  `tools/evaluate_openings.py` on VMs and report `running_opening_eval_processes`.
  The planner requires the remote `opening_evaluator` feature and a worker
  that is train-idle, grade-clean, opening-eval-idle, and locally unclaimed.
- Verification passed without local RL games:
  `py_compile tools/gcp_fleet_controller.py tools/remote_fleet_autopilot.py`
  and `pytest tests/test_remote_fleet_autopilot.py tests/test_agent_grader.py -q`
  (`115 passed`).
- Fresh VM poll from `runs/self_play/gcp_poll_latest_s.json`: 10 active VM
  trainers, 0 active reanalysis generators, 0 active opening evaluators, and
  811 visible candidate checkpoints. Active training is split between the
  original VRPO/Expected-SARSA anti-regression wave and the newer JSettlers
  value-repair response:
  `s10080_vrpo_esarsa_antireg_c1`,
  `s10091_vrpo_esarsa_antireg_c2`,
  `s10072_vrpo_esarsa_antireg_c3`,
  `s10103_vrpo_esarsa_antireg_c4`,
  `s10119_vrpo_esarsa_antireg_w4d`,
  `s10124_vrpo_jsettlers_value_repair_w1a`,
  `s10135_vrpo_jsettlers_value_repair_w1b`,
  `s10146_vrpo_jsettlers_value_repair_w4a`,
  `s10157_vrpo_jsettlers_value_repair_w4b`, and
  `s10168_vrpo_jsettlers_value_repair_w4c`.
- Fresh remote grade summary: 2 active VM-side strict grades, 252 completed
  decisions, 5 historical promote signals, and 247 rejects. Active grades are
  `s9861_topadv_repair_w1a.iter0004` on `catan-zero-c1` and
  `s10072_vrpo_esarsa_antireg_c3.iter0002` on `catan-zero-c3`.
- `remote_opening_eval_plan_latest.json` planned 0, as intended. The planner
  found no clean slot; candidates were skipped for busy workers, active
  family, already-decided checkpoints, older snapshots, or the run-number
  filter. Do not override this with busy-worker flags.
- Current critique: the `s100xx/s101xx` VRPO interims are still mostly
  rejected on JSettlers/value-rollout regression, so the west-worker
  `vrpo_jsettlers_value_repair` branch is reasonable. The next useful VM
  diagnostic is opening quality once a worker is clean; the next training
  branch should remain either JSettlers+value repair or DAGS/midgame
  reanalysis, not blind PPO fanout.

## 2026-06-27 10:05 PDT Status

- No local self-play training or local game grading was launched. Local work
  was controller/tooling, syntax/unit verification, VM status consumption, and
  no-busy planning only. No Claude-mini or subagent work was used.
- Fresh VM poll from `runs/self_play/gcp_poll_latest_s.json`: 10 active VM
  trainers, 0 active reanalysis generators, and 786 visible candidate
  checkpoints. The active training wave has moved to VRPO/Expected-SARSA
  anti-regression:
  `s10080_vrpo_esarsa_antireg_c1`,
  `s10091_vrpo_esarsa_antireg_c2`,
  `s10072_vrpo_esarsa_antireg_c3`,
  `s10103_vrpo_esarsa_antireg_c4`,
  `s10024_vrpo_esarsa_antireg_w1a`,
  `s10035_vrpo_esarsa_antireg_w1b`,
  `s10056_vrpo_esarsa_antireg_w4a`,
  `s10067_vrpo_esarsa_antireg_w4b`,
  `s10048_vrpo_esarsa_antireg_w4c`, and
  `s10119_vrpo_esarsa_antireg_w4d`.
- Fresh remote grade summary from `remote_grade_summary_latest.json`: 5 active
  VM-side grades, 240 completed decisions, 5 promote signals, and 235 rejects.
  Active grades are the `s9861_topadv_repair_w1a.iter0004` strict escalation
  plus VRPO triage/strict gates for `s10072.iter0002`, `s10056.iter0002`,
  `s10067.iter0002`, and `s10048.iter0002`.
- Added GCP fleet support for the next VM-only DAGS/reanalysis lane:
  - `remote-reanalysis-train` launches a detached VM pipeline that runs
    `tools/generate_reanalysis.py --teacher value_rollout` with midgame record
    windows, then runs `tools/train_ppo.py --reanalysis-input` to produce a
    reanalysis-only candidate checkpoint.
  - `plan-remote-reanalysis-train` schedules that pipeline only on workers
    that are train-idle, reanalysis-idle, grade-clean, locally unclaimed, and
    feature-complete.
  - Broad polls now report `running_reanalysis_processes` and remote feature
    flags for `reanalysis_training` and `reanalysis_decision_windows`.
  - Default code syncs now include `tools/generate_reanalysis.py` and
    `src/catan_zero/rl/reanalysis.py`.
  - Verification passed: `py_compile tools/gcp_fleet_controller.py` and
    `pytest tests/test_agent_grader.py -q` (`102 passed`). No local RL games,
    PPO training, or grading were run.
- Safe no-busy planners after the fresh poll:
  - `remote_train_plan_dags_midgame_reanalysis_latest.json`: `planned_count: 0`,
    `next_seed: 10120`; every worker skipped for active training.
  - `remote_code_sync_plan_dags_midgame_reanalysis_latest.json`:
    `planned_count: 0`; every worker skipped for active training.
  - `remote_train_plan_ema_mixed_after_dags_tool_latest.json`:
    `planned_count: 0`, `next_seed: 10120`; every worker skipped for active
    training.
  - `remote_transfer_gate_plan_after_dags_tool_latest.json`:
    `planned_count: 0`, `eligible_targets: []`; all target workers are
    training or remote-grade busy. The planner also blocks rejected regression
    families including `s9981`, `s9993`, `s10009`, and `s10012`.
- Next action remains VM-only monitoring. When one worker is both train-idle
  and grade-clean, first rerun local status, broad `s` poll, grade status, and
  grade summary. Then either gate a fresh VRPO checkpoint or launch exactly one
  `plan-remote-reanalysis-train` command if no fresher gate target has priority.

## 2026-06-27 09:49 PDT Status

- No local self-play training, local game grading, Claude-mini, or subagent
  work was launched. Local process checks show no local trainer/grader or
  mini-agent process.
- Consumed the other controller's fresh 09:44 artifacts instead of starting a
  duplicate full poll. Current VM state from `gcp_poll_latest_s.json`:
  10 active VM trainers and 750 visible candidate checkpoints. Active trainers
  remain the mixed anti-regression wave:
  `s9970_ema_mixed_antireg_c1`, `s9981_ema_mixed_antireg_c2`,
  `s10012_ema_mixed_antireg_c3`, `s9993_ema_mixed_antireg_c4`,
  `s9924_ema_mixed_antireg_w1a`, `s9935_ema_mixed_antireg_w1b`,
  `s9946_ema_mixed_antireg_w4a`, `s9957_ema_mixed_antireg_w4b`,
  `s9968_ema_mixed_antireg_w4c`, and `s10009_ema_mixed_antireg_w4d`.
- Fresh remote grade status reports 1 active VM-side grade, 924 completed
  legs, and 223 completed summaries. The remaining active grade is
  `s9861_topadv_repair_w1a.iter0004` strict g12 on `catan-zero-c1`.
- Fresh no-busy plan checks still made no launch:
  - `remote_train_plan_ema_mixed_post_dags_latest.json`: `planned_count: 0`,
    `next_seed: 10068`, all 10 workers skipped for active training.
  - `remote_transfer_gate_plan_post_dags_latest.json`: `planned_count: 0`;
    newest mixed checkpoints are active/decided/older-family blocked or have
    no eligible clean target. Do not use training-busy target overrides.
- Code improvement for the next VM-only method branch:
  - `src/catan_zero/rl/self_play.py`: `collect_imitation_game` now accepts
    `record_after_decisions` and `record_until_decision`, and `StepSample`
    carries optional `decision_index`.
  - `src/catan_zero/rl/reanalysis.py`: sparse reanalysis JSONL now preserves
    `decision_index` while remaining backward-compatible with older records.
  - `tools/generate_reanalysis.py`: added `--record-after-decisions` and
    `--record-window-decisions` so a future VM run can generate DAGS-style
    mid/late-game search targets without local training.
  - Verification run: `py_compile` for the touched modules plus
    `tests/test_self_play.py::test_reanalysis_records_preserve_decision_index`,
    `test_should_record_imitation_sample_supports_midgame_windows`, and
    `test_collect_imitation_game_can_record_later_decision_window` passed
    (`3 passed`). No local PPO training or local grading was run.
- Next safe VM experiment when a train-idle and grade-clean slot opens:
  generate a small `value_rollout` reanalysis set with something like
  `--record-after-decisions 40 --record-window-decisions 120`, then train a
  reanalysis-warm-start or reanalysis-only candidate from that JSONL and gate
  it strictly against `current_best_s9752_iter0002`. Do this VM-side only.

## 2026-06-27 09:39 PDT Status

- No local self-play training, local game grading, Claude-mini, or subagent
  work is running. Local controller status remains clean:
  `active_count: 0`, no claimed workers.
- Fresh broad VM poll from `runs/self_play/gcp_poll_latest_s.json`:
  10 active VM trainers and 743 visible candidate checkpoints. The active
  training wave is still fully VM-side mixed anti-regression:
  `s9970_ema_mixed_antireg_c1`, `s9981_ema_mixed_antireg_c2`,
  `s10012_ema_mixed_antireg_c3`, `s9993_ema_mixed_antireg_c4`,
  `s9924_ema_mixed_antireg_w1a`, `s9935_ema_mixed_antireg_w1b`,
  `s9946_ema_mixed_antireg_w4a`, `s9957_ema_mixed_antireg_w4b`,
  `s9968_ema_mixed_antireg_w4c`, and `s10009_ema_mixed_antireg_w4d`.
- Fresh VM-side grade status sees all 10 workers running grade processes,
  915 completed grade legs, and 214 completed summaries. Active grade targets
  are:
  `s9861_topadv_repair_w1a.iter0004` strict g12,
  `s9901_ema_jsettlers_antireg_c2.pt` strict g4,
  `s9902_ema_jsettlers_antireg_c4.pt` strict g4, and triage gates for
  `s10012.iter0002`, `s9924.iter0004`, `s9935.iter0004`,
  `s9946.iter0004`, `s9957.iter0004`, `s9968.iter0004`, and
  `s10009.iter0002`.
- Fresh no-busy planners returned no launch:
  - `remote_transfer_gate_plan_fresh_s_latest.json`: `planned_count: 0`;
    all target workers are remote-grade busy and the newest mixed families are
    already active or have no clean target.
  - `remote_train_plan_ema_mixed_fresh_latest.json`: `planned_count: 0`,
    `next_seed: 10013`; every worker is training.
  - `remote_escalation_plan_fresh_latest.json`: `planned_count: 0`; no clean
    escalation worker without a busy override.
- Research critique, refreshed from primary arXiv metadata/API:
  - Keep the production lane conservative: regularized PPO with old-policy KL,
    EMA-policy KL, entropy retention, DAgger anchors, top-advantage filtering,
    mixed anti-regression opponents, and strict external gates.
  - The current code already supports the right near-term knobs: EMA/old-policy
    KL, gated Q-advantage/Expected-SARSA diagnostics, reanalysis input,
    value-rollout teachers, `JSettlersLitePolicy`, and VM-side gate planning.
  - JSettlersLite is useful as a tactical resource/trade/robber regression
    guard, but recent strict evidence says JSettlers-only fanout overfits and
    regresses value-rollout. Do not revive JSettlers-only training without a
    contradictory strict gate.
  - DAGS-style intermediate-start sampling
    (`https://arxiv.org/abs/2605.14379`) is the next plausible algorithmic
    upgrade after the current gates finish: start Catan self-play from
    realistic mid/late-game states or public replay states to improve sparse
    late-game credit assignment. This should be VM-side and tied to strict
    gates, not a local experiment.
  - Student of Games, BRExIt/GenBR-style opponent modeling, and Global PSRO
    (`https://arxiv.org/abs/2112.03178`,
    `https://arxiv.org/abs/2206.00113`,
    `https://arxiv.org/abs/2302.00797`,
    `https://arxiv.org/abs/2605.28273`) remain later-stage population/search
    work after at least one current-generation parent survives stronger gates.
- Next safe action remains monitoring. Before any new VM launch, re-poll
  local controller status, broad `s`, current-family `s99`/`s100`, remote
  grade status, and remote grade summary. Launch only on a worker that is both
  train-idle and grade-clean.

## 2026-06-27 09:31 PDT Status

- No local self-play training, local game grading, Claude-mini, or subagent
  work was launched. Final local controller status is clean:
  `active_count: 0`, no claimed workers.
- Coordination/tooling improvement:
  - `tools/gcp_fleet_controller.py` now ignores shell wrapper lines in
    `local-controller-status`, preventing fake claims such as `$spec` when
    another agent runs a shell loop around controller commands.
  - `tools/remote_fleet_autopilot.py` now requires explicit opt-in flags for
    busy-worker scheduling: `--allow-training-busy-gates` and
    `--allow-grade-busy-training`. The default autopilot path no longer plans
    gates on training-busy workers or training on grade-busy workers.
  - Verification passed: `py_compile` for both controller tools and
    `pytest tests/test_agent_grader.py tests/test_remote_fleet_autopilot.py -q`
    (`103 passed`).
- Fresh VM state:
  - `s99`: 10 active trainers and 48 visible `s99` candidates.
  - `s100`: 10 active trainers and 2 visible `s100` candidates:
    `s10012_ema_mixed_antireg_c3.iter0002.pt` and
    `s10009_ema_mixed_antireg_w4d.iter0002.pt`.
  - broad `s9`: 10 active trainers and 738 visible candidate checkpoints.
- Active training remains fully VM-side mixed anti-regression:
  `s9970_ema_mixed_antireg_c1`, `s9981_ema_mixed_antireg_c2`,
  `s10012_ema_mixed_antireg_c3`, `s9993_ema_mixed_antireg_c4`,
  `s9924_ema_mixed_antireg_w1a`, `s9935_ema_mixed_antireg_w1b`,
  `s9946_ema_mixed_antireg_w4a`, `s9957_ema_mixed_antireg_w4b`,
  `s9968_ema_mixed_antireg_w4c`, and `s10009_ema_mixed_antireg_w4d`.
- Fresh VM-side grade state: 8 active grades, 909 completed grade legs, 214
  completed summaries, 5 promote signals, and 209 rejects. Active gates are:
  `s9861_topadv_repair_w1a.iter0004` g12 escalation plus mixed interims
  `s10012.iter0002`, `s9924.iter0004`, `s9935.iter0004`,
  `s9946.iter0004`, `s9957.iter0004`, `s9968.iter0004`, and
  `s10009.iter0002`.
- Safe planners without busy overrides all returned no launch:
  - `remote_transfer_gate_plan_latest_s99.json`: `planned_count: 0`.
  - `remote_transfer_gate_plan_latest_s100.json`: `planned_count: 0`.
  - `remote_train_plan_ema_mixed_latest.json`: `planned_count: 0`,
    `next_seed: 10013`, every worker skipped for active training.
- Research update from primary arXiv sources remains conservative:
  regularized PPO plus EMA/KL/entropy and strict mixed-opponent gates is still
  the right production lane. VRPO/Q-boosting should stay critic-side until
  diagnostics justify actor mixing. Search distillation, Student-of-Games
  style reanalysis, BRExIt opponent modeling, and JBR/PSRO are next-stage
  methods after a non-regressing parent survives stronger gates.

## 2026-06-27 09:22 PDT Status

- No local self-play training, local game grading, Claude-mini, or subagent
  work is running. Local work in this pass was VM orchestration/status,
  transfer planning, doc coordination, and primary-source research only.
- Fresh VM polls:
  - `s99`: 10 active VM trainers and 40 visible `s99` candidate checkpoints.
  - `s100`: 10 active VM trainers, with `s10012_ema_mixed_antireg_c3` and
    `s10009_ema_mixed_antireg_w4d` visible as live out-of-`s99` mixed jobs,
    but no `s100` candidate checkpoints yet.
  - broad `s9`: 10 active VM trainers and 730 visible candidate checkpoints.
- Active training is fully VM-side mixed anti-regression:
  `s9970_ema_mixed_antireg_c1`, `s9981_ema_mixed_antireg_c2`,
  `s10012_ema_mixed_antireg_c3`, `s9993_ema_mixed_antireg_c4`,
  `s9924_ema_mixed_antireg_w1a`, `s9935_ema_mixed_antireg_w1b`,
  `s9946_ema_mixed_antireg_w4a`, `s9957_ema_mixed_antireg_w4b`,
  `s9968_ema_mixed_antireg_w4c`, and `s10009_ema_mixed_antireg_w4d`.
- Fresh VM-side grade summary: 9 active remote grades and 206 completed
  summaries. Decisions now total 5 promote signals and 201 rejects. The only
  current-champion promote signals remain
  `s9829_qadv_rollout_c4.iter0003` and
  `s9861_topadv_repair_w1a.iter0004`; `s9861` is still under stricter g12
  escalation.
- Recent JSettlers/EMA evidence is negative or unstable. Strict/triage rejects
  now include `s9896_jsettlers_dagger_antireg_c3`,
  `s9897_ema_jsettlers_antireg_w1a`, `s9898_ema_jsettlers_antireg_w1b`,
  `s9899_ema_jsettlers_antireg_w4a`,
  `s9900_ema_jsettlers_antireg_w4b`,
  `s9901_ema_jsettlers_antireg_c2`,
  `s9901_ema_jsettlers_antireg_w4c`,
  `s9902_ema_jsettlers_antireg_c4`,
  `s9903_ema_jsettlers_antireg_w4d`, and
  `s9912_ema_jsettlers_antireg_c3`. Treat timeout-heavy triage as a reject
  unless a later strict gate contradicts it.
- Fresh transfer planning is still a no-op:
  `remote_transfer_gate_plan_latest_s99.json` and
  `remote_transfer_gate_plan_latest_s100.json` both report
  `planned_count: 0`. The one grade-free worker, `catan-zero-c3`, is training
  `s10012_ema_mixed_antireg_c3`; every other worker is remote-grade busy. Do
  not use busy-worker overrides.
- Research critique after primary-source refresh:
  - Catan remains a real multi-player, imperfect-information, stochastic
    benchmark, and the published JSettlers result is the minimum floor to beat
    (`https://arxiv.org/abs/2008.07079`).
  - PPO remains defensible for imperfect-information games, including recent
    broad exploitability comparisons and four-player Big 2 self-play
    (`https://arxiv.org/abs/2502.08938`,
    `https://arxiv.org/abs/2605.28863`).
  - EMA/KL/entropy regularization remains the right near-term stabilizer
    (`https://arxiv.org/abs/2606.23995`,
    `https://arxiv.org/abs/2602.10894`).
  - VRPO/Q-boosting is promising but should stay behind critic sign/correlation
    gates before actor mixing is increased (`https://arxiv.org/abs/2605.19235`).
  - Competitive PPO failure work supports the current emphasis on
    implementation hygiene, opponent mixing, terminal/time-limit correctness,
    and strict external gates (`https://arxiv.org/abs/2604.04983`).
  - Search distillation and population methods should wait for a
    non-regressing parent, then draw from Student of Games, BRExIt, and JBR/PSRO
    (`https://arxiv.org/abs/2112.03178`,
    `https://arxiv.org/abs/2206.00113`,
    `https://arxiv.org/abs/2602.06599`).

## 2026-06-27 09:10 PDT Status

- No local self-play training or local game grading was launched by this pass.
  A separate controller briefly appeared and started additional VM-side
  training; it was not killed.
- Fresh `s99` poll now shows 8 active VM trainers and 29 visible `s99`
  candidate checkpoints. Finished EMA/JSettlers finals are visible for
  `s9900_ema_jsettlers_antireg_c1.pt`,
  `s9901_ema_jsettlers_antireg_c2.pt`,
  `s9900_ema_jsettlers_antireg_w4b.pt`, and
  `s9901_ema_jsettlers_antireg_w4c.pt`.
- Active trainers are `s9912_ema_jsettlers_antireg_c3`,
  `s9902_ema_jsettlers_antireg_c4`,
  `s9903_ema_jsettlers_antireg_w4d`, plus the externally-started mixed
  anti-regression wave: `s9924_ema_mixed_antireg_w1a`,
  `s9935_ema_mixed_antireg_w1b`, `s9946_ema_mixed_antireg_w4a`,
  `s9957_ema_mixed_antireg_w4b`, and `s9968_ema_mixed_antireg_w4c`.
- Fresh VM-side transfer planner with explicit `--poll`, `--summary`,
  `--local-status`, `--run-prefix s99`, and `--min-run-number 0` is still a
  no-op: `planned_count: 0`, `eligible_targets: []`. Every worker is skipped
  for active `remote_grade`. Do not stack more VM work until at least one
  remote grade finishes.
- The strategic read is unchanged: the fleet is now doing the right broad
  category, mixed anti-regression rather than more JSettlers-only fanout, but
  nothing should be promoted or extended until the VM gates finish.

## 2026-06-27 08:52 PDT Status

- No local self-play training, local game grading, Claude-mini, or subagent
  work is running. Local work in this pass was controller status, VM polling,
  remote summary compaction, transfer planning, docs, and primary-source
  research.
- Fresh `s98` poll: 10 active VM trainers, 383 visible `s98` candidate
  checkpoints, and all workers report the current trainer features including
  `q_advantage_gate`, EMA policy KL, old-policy KL, top-advantage filtering,
  and mixed anti-regression support.
- Fresh `s99` poll: the EMA JSettlers anti-regression wave is alive with 15
  visible interim checkpoints. Current active trainers are
  `s9900_ema_jsettlers_antireg_c1`,
  `s9901_ema_jsettlers_antireg_c2`,
  `s9912_ema_jsettlers_antireg_c3`,
  `s9902_ema_jsettlers_antireg_c4`,
  `s9897_ema_jsettlers_antireg_w1a`,
  `s9898_ema_jsettlers_antireg_w1b`,
  `s9899_ema_jsettlers_antireg_w4a`,
  `s9900_ema_jsettlers_antireg_w4b`,
  `s9901_ema_jsettlers_antireg_w4c`, and
  `s9903_ema_jsettlers_antireg_w4d`.
- Fresh remote grade status: 14 active VM-side gates, 867 completed grade
  legs, and 180 completed summaries. The four newest transfer gates are active
  rather than completed: `s9890_jsettlers_dagger_antireg_w1a`,
  `s9894_jsettlers_dagger_antireg_w4c`,
  `s9896_jsettlers_dagger_antireg_c3`, and
  `s9897_ema_jsettlers_antireg_w1a.iter0004`.
- Fresh transfer planner is a no-op: `planned_count: 0`,
  `eligible_targets: []`. Every VM target is skipped for `remote_grade`.
  Do not stack more gates or launch from stale plans.
- Refreshed population summary against current `s9752`: 133 strict pair rows,
  2 promote decisions, 131 rejects, reject rate `0.9850`. Current keepers are
  unchanged: `s9861_topadv_repair_w1a.iter0004` and
  `s9829_qadv_rollout_c4.iter0003`. Best candidate remains
  `s9861_topadv_repair_w1a.iter0004`, and its g12 escalation is still active.
- Recent JSettlers triage timeouts (`s9886.iter0006`, `s9888.iter0002`,
  `s9895.iter0002`) are rejects, not promotion evidence. Treat timeouts as
  stability failures unless later strict gates contradict them.
- Research critique remains conservative: PPO is still justified for
  imperfect-information self-play (`https://arxiv.org/abs/2502.08938`) and
  four-player imperfect-information card play (`https://arxiv.org/abs/2605.28863`).
  VRPO/Q-boosting (`https://arxiv.org/abs/2605.19235`) should stay behind
  critic-quality gates. EMAgnet (`https://arxiv.org/abs/2606.23995`) and
  regularized board-game policy optimization
  (`https://arxiv.org/abs/2602.10894`) support the active EMA/KL/entropy
  anti-regression direction. Student-of-Games/BRExIt and PSRO/JBR ideas are
  long-term search/population work after at least one non-regressing parent
  survives higher-confidence gates (`https://arxiv.org/abs/2112.03178`,
  `https://arxiv.org/abs/2206.00113`,
  `https://arxiv.org/abs/2602.06599`).

## 2026-06-27 09:06 PDT Status

- No local self-play training or local game grading was launched. Local
  controller status is clean with `active_count: 0`. No Claude-mini/subagent
  process was used.
- Fresh VM state: `s99` poll sees 5 active trainers and 24 visible `s99`
  candidate checkpoints. Active trainers are `s9900_ema_jsettlers_antireg_c1`,
  `s9901_ema_jsettlers_antireg_c2`, `s9912_ema_jsettlers_antireg_c3`,
  `s9902_ema_jsettlers_antireg_c4`, and
  `s9903_ema_jsettlers_antireg_w4d`. Final `s99` checkpoints are visible for
  `s9900_ema_jsettlers_antireg_w4b.pt` and
  `s9901_ema_jsettlers_antireg_w4c.pt`.
- Fresh remote grade status: all 10 workers are running VM-side gates, with
  891 completed grade legs and 197 completed summaries. Active gates include
  `s9861_topadv_repair_w1a.iter0004` g12 escalation plus EMA/JSettlers gates
  for `s9897`, `s9898`, `s9899`, `s9901_c2.iter0006`,
  `s9912_c3.iter0004`, `s9901_w4c`, and `s9903_w4d.iter0006`.
- Recent strict rejects now include
  `s9890_jsettlers_dagger_antireg_w1a.pt`,
  `s9894_jsettlers_dagger_antireg_w4c.pt`,
  `s9896_jsettlers_dagger_antireg_c3.pt`, and
  `s9897_ema_jsettlers_antireg_w1a.iter0004.pt`; all regressed on
  `jsettlers_lite` and `value_rollout`. Treat the current JSettlers-only
  direction as unproven until a strict multi-opponent gate says otherwise.
- Fresh explicit `s99` transfer planner is a no-op:
  `planned_count: 0`, `eligible_targets: []`; every worker is skipped for
  active `remote_grade`. No launch is safe now.
- Planning footgun found: when using `--run-prefix s99`, do not pass
  `--min-run-number 9897`; the parser interprets `s9900` as run number `0`
  after the `s99` prefix. Use `--min-run-number 0` plus `--prefer-prefix s991`
  and `--prefer-prefix s990`, or use `--run-prefix s9` with an absolute
  `--min-run-number`.
- Research critique update: Catan-specific evidence keeps JSettlers as a real
  floor (`https://arxiv.org/abs/2008.07079`), but the recent results still
  favor regularized PPO, entropy, EMA/KL anchoring, and external gates
  (`https://arxiv.org/abs/2502.08938`,
  `https://arxiv.org/abs/2605.28863`,
  `https://arxiv.org/abs/2606.23995`). VRPO/Q-boosting remains a critic-side
  variance-reduction idea until diagnostics prove actor mixing is safe
  (`https://arxiv.org/abs/2605.19235`). Competitive PPO failures reinforce
  opponent mixing and external non-self-play gates, not self-play-only
  promotion (`https://arxiv.org/abs/2604.04983`).

## 2026-06-27 08:36 PDT Status

- No local self-play training, local game grading, Claude-mini, or subagent
  work was launched. Local controller status remains clean with
  `active_count: 0`.
- Critical orchestration safety fix landed in `tools/gcp_fleet_controller.py`:
  `poll --run-prefix s98` now still counts all active remote `train_ppo.py`
  processes as busy, including `s99` jobs. Prefix filtering still applies to
  checkpoint/log artifact listing, but no planner should see a worker as idle
  just because another agent launched a newer run prefix.
- Second planner safety fix: automatic VM training seeds are now slotted by
  worker and skip already consumed seeds. This reduces duplicate RNG seeds when
  separate controllers independently plan from the same stale fleet snapshot.
- Verification: `py_compile tools/gcp_fleet_controller.py`; focused
  controller tests passed (`16 passed, 73 deselected`); full
  `tests/test_agent_grader.py` passed (`89 passed`).
- Fresh post-fix poll: both `s98` and broad `s9` polls report 10 active
  trainers and no train-idle workers. Active training is the VM-only EMA
  JSettlers anti-regression wave (`s9897`-`s9903`) plus
  `s9896_jsettlers_dagger_antireg_c3`, all against `jsettlers_lite`. Remote
  grade status remains saturated with 10 active grades, 840 completed legs, and
  175 completed summaries.
- Fresh planners are no-op: `remote_train_plan_ema_jsettlers_latest.json`
  reports every worker skipped for `training`; `remote_transfer_gate_plan_latest_s98.json`
  reports every target skipped for `remote_grade`. No VM launch is safe or
  useful right now.
- Payoff summary unchanged: 131 strict pair rows, 2 promote decisions, 129
  rejects, reject rate `0.9847`. Best current candidate remains
  `s9861_topadv_repair_w1a.iter0004`; dominant failure modes remain
  JSettlers regression, value-rollout regression, and heuristic regression.
- Research update: keep the active EMA/regularized PPO direction. New
  primary-source notes reinforce the current ladder: asymmetric board games may
  need role-aware heads later (`https://arxiv.org/abs/2604.05476`), strong
  imperfect-information agents are often throughput/evaluation constrained
  (`https://arxiv.org/abs/2606.23348`), and competitive PPO failures are often
  implementation/evaluation hygiene issues before algorithm novelty
  (`https://arxiv.org/abs/2604.04983`). Do not add architecture churn until
  the current EMA JSettlers wave clears strict gates.

## 2026-06-27 08:24 PDT Status

- No local self-play training or local game grading was launched. No
  Claude-mini/subagent processes are running. Local controller state is clean:
  `active_count: 0`, with no claimed workers.
- Fresh VM status changed during this pass. A later poll now shows all remote
  trainers have `q_advantage_gate: true`, 9 workers are training, and only
  `w4d` is train-idle. Active trainers are `s9896_jsettlers_dagger_antireg_c3`
  plus `s9897_ema_jsettlers_antireg_w1a`,
  `s9898_ema_jsettlers_antireg_w1b`,
  `s9899_ema_jsettlers_antireg_w4a`,
  `s9900_ema_jsettlers_antireg_c1`,
  `s9900_ema_jsettlers_antireg_w4b`,
  `s9901_ema_jsettlers_antireg_c2`,
  `s9901_ema_jsettlers_antireg_w4c`, and
  `s9902_ema_jsettlers_antireg_c4`. This launch appears to have happened via
  another controller while this pass was updating docs; do not duplicate it.
  Note that seeds `9900` and `9901` are reused across worker-specific labels,
  so future planners should prefer globally unique seeds for independence.
- Remote grade refresh: 10 active remote grades, 839 completed grade legs, and
  175 completed summaries. Active grade targets are
  `s9861_topadv_repair_w1a.iter0004`, `s9888_jsettlers_dagger_antireg_c2.iter0002`,
  `s9886_weighted_dagger_antireg_c3.iter0006`,
  `s9889_jsettlers_dagger_antireg_c4.iter0004`,
  `s9869_value_rollout_repair_w4b`, `s9891_jsettlers_dagger_antireg_w1b.iter0006`,
  `s9892_jsettlers_dagger_antireg_w4a.iter0006`,
  `s9893_jsettlers_dagger_antireg_w4b.iter0006`,
  `s9874_weighted_dagger_antireg_w1a`, and
  `s9895_jsettlers_dagger_antireg_w4d.iter0002`.
- Population summary refreshed from remote results: 131 strict pair rows, 2
  promote decisions, 129 rejects, reject rate `0.9847`. The current best
  candidate remains `s9861_topadv_repair_w1a.iter0004`; the other
  current-champion keeper remains `s9829_qadv_rollout_c4.iter0003`.
  Dominant failure modes remain `opponent_regression:jsettlers_lite` (85),
  `opponent_regression:value_rollout` (66), and
  `opponent_regression:heuristic` (43).
- Earlier planned `jsettlers_dagger_antireg` syncs to `w1a`, `w1b`, and `w4a`
  skipped safely because active remote grades were detected in preflight.
  A subsequent poll shows the remote code is now synced across the fleet and
  the EMA JSettlers anti-regression wave is running. Do not force additional
  `remote-sync-code --allow-busy` or launch from older sync/train plans.
- Research/action map update from primary sources: PPO remains the production
  learner for imperfect-information self-play, but Q/Expected-SARSA should stay
  critic-side unless sign/correlation gates clear. EMA/old-policy KL and
  entropy anchors remain mandatory. DAGS/intermediate starts, ExIt-style
  search distillation, Student-of-Games style reanalysis, and PSRO/PFSP
  population scheduling are long-term upgrades after a non-regressing parent is
  established, not replacements for the current strict anti-regression gate.

## 2026-06-27 08:14 PDT Status

- No local self-play training or local game grading was launched. No
  Claude-mini/subagent processes are running. Local controller status is clean
  after refresh with `active_count: 0`.
- Fresh VM poll: all 10 workers are training and all 10 grade targets are
  busy. The poll reports 338 visible candidate checkpoints. Active trainers
  are `s9887`, `s9888`, `s9889`, `s9890`-`s9895`, and the newly observed
  `s9896_jsettlers_dagger_antireg_c3`, all against `jsettlers_lite`.
  `s9886_weighted_dagger_antireg_c3` has finished/freed its train slot and
  remains under remote `jsettlers_triage` at `iter0006`.
- JSettlers-DAGGER iter-4 checkpoints are now visible for `s9887`, `s9888`,
  `s9889`, `s9890`, `s9891`, `s9892`, `s9893`, `s9894`, and `s9895`.
  `s9896` is the newest central-worker JSettlers continuation.
- Remote grade summary is still 10 active gates, 171 completed summaries, and
  823 completed legs. Current-champion keepers are unchanged:
  `s9861_topadv_repair_w1a.iter0004` and
  `s9829_qadv_rollout_c4.iter0003`; `s9861.iter0004` remains in g12
  escalation.
- Refreshed transfer, tactical-train, tactical-sync, JSettlers-Q-gate train,
  and JSettlers-Q-gate sync planners are all no-op because all VM targets are
  busy. The new JSettlers sync planner requires `q_advantage_gate`, so any
  future guarded JSettlers-DAGGER branch must first sync clean VMs.
- Local algorithm improvement added for the next VM branch: `tools/train_ppo.py`
  now supports `--q-advantage-min-sign-agreement` and
  `--q-advantage-min-return-corr`. These keep VRPO-style Q-advantage actor
  mixing disabled until prior-iteration critic diagnostics clear thresholds,
  while still training/logging the Q critic. `weighted_dagger_antireg` and
  `jsettlers_dagger_antireg` now request this guard in the controller recipe.
  This is local-only until a future code-sync runs on clean VMs.
- Verification run, without local games: `py_compile` on
  `tools/train_ppo.py` and `tools/gcp_fleet_controller.py`; focused Q-gate
  tests in `tests/test_self_play.py` passed (`3 passed, 91 deselected`);
  focused controller feature/planner tests in `tests/test_agent_grader.py`
  passed (`5 passed, 80 deselected`).
- Research critique update: VRPO/Q-boosting remains useful only when critic
  diagnostics agree with GAE/returns (`https://arxiv.org/abs/2605.19235`).
  The new guard operationalizes that critique. Keep PPO anchored by KL/EMA
  and entropy per the imperfect-information PPO and regularized policy
  optimization evidence (`https://arxiv.org/abs/2502.08938`,
  `https://arxiv.org/abs/2602.10894`, `https://arxiv.org/abs/2606.23995`).

## 2026-06-27 08:04 PDT Status

- No local self-play training, local game grading, Claude-mini, or subagent
  work was launched. Local work was VM polling, remote-summary bookkeeping,
  transfer planning, docs, and primary-source research. The local controller
  status file is clean with `active_count: 0`.
- Fresh VM poll: all 10 workers are training, with 331 visible candidate
  checkpoints. Active trainers are still the VM-only JSettlers repair wave:
  `s9887`, `s9888`, `s9889`, and `s9890`-`s9895`
  `jsettlers_dagger_antireg`, plus `s9886_weighted_dagger_antireg_c3`.
  JSettlers iter-4 snapshots are now visible for `s9890`, `s9891`,
  `s9892`, `s9893`, and `s9894`; central-worker and `w4d` iter-2 snapshots
  are visible.
- Fresh remote grade status: 10 active VM-side gates, 823 completed grade
  legs, and 171 completed summaries. Active gates now include early triage
  for `s9888_jsettlers_dagger_antireg_c2.iter0002.pt`,
  `s9886_weighted_dagger_antireg_c3.iter0006.pt`, and
  `s9895_jsettlers_dagger_antireg_w4d.iter0002.pt`, alongside the still
  active `s9861` g12, value-rollout repair gates, `s9871`, and `s9874`.
- Refreshed population summary against current `s9752`: 127 strict pair rows,
  2 current-champion keepers, 125 strict rejects, reject rate `0.9843`.
  The only current-champion keepers remain
  `s9861_topadv_repair_w1a.iter0004` and
  `s9829_qadv_rollout_c4.iter0003`; `s9861.iter0004` is already in a g12
  escalation. Primary failure modes remain
  `opponent_regression:jsettlers_lite` (85), `value_rollout` (66), and
  `heuristic` (43).
- Refreshed transfer planner is no-op: `planned_count: 0`,
  `eligible_targets: []`. Every target is busy with a remote grade. Do not
  execute stale commands from older plans.
- Next grade queue once a clean grade VM appears: prefer the newest
  JSettlers-DAGGER snapshots, led by `s9890`-`s9894` iter-4, then finals from
  weighted-DAGGER (`s9881`, `s9878`, `s9876`-`s9885`) unless a currently
  active triage gate resolves first and changes the priority.
- Algorithm critique: keep PPO as the production learner, but keep it
  conservative. PPO is still defensible for imperfect-information self-play
  (`https://arxiv.org/abs/2502.08938`) and recent card-game self-play
  (`https://arxiv.org/abs/2605.28863`). VRPO/Q-boosting
  (`https://arxiv.org/abs/2605.19235`) maps here to critic-side variance
  reduction, not unchecked actor steering. EMAgnet
  (`https://arxiv.org/abs/2606.23995`) and reverse-KL/entropy regularized
  policy optimization (`https://arxiv.org/abs/2602.10894`) support the
  existing high old-policy/EMA KL anchor strategy. DAGS/ExIt/Student of Games
  stay later-stage search/reanalysis work after a non-regressing parent is
  established.

## 2026-06-27 07:56 PDT Status

- No local self-play training or local game grading was launched. Local work
  was VM polling, compact summary refresh, no-op planner refresh, source
  inspection, documentation, and primary-source research. Final local
  controller state is clean: `active_count: 0`.
- Fresh VM poll: all 10 workers are training, with 322 visible candidate
  checkpoints. The active wave is still useful: `s9887`, `s9888`, `s9889`,
  and `s9890`-`s9895` are `jsettlers_dagger_antireg`; `s9886` on `c3` is
  `weighted_dagger_antireg`. The JSettlers wave has begun producing iter-2
  snapshots on `w1a`, `w1b`, `w4a`, `w4b`, and `w4c`.
- Fresh remote grade status: 10 active strict gates, 817 completed legs, and
  168 completed summaries. Active grade targets are unchanged:
  `s9861_topadv_repair_w1a.iter0004.pt` g12, `s9861` iter-8/final g4,
  `s9865`, `s9866`, `s9867`, `s9868`, `s9869`,
  `s9871_topadv_keeper_continue_w1a.iter0004.pt`, and
  `s9874_weighted_dagger_antireg_w1a.pt`.
- Refreshed launch plans are no-op. `remote_train_plan_tactical_rollout_guard_latest.json`
  and `remote_code_sync_plan_tactical_latest.json` both skip every worker for
  active training. `remote_transfer_gate_plan_latest_s98.json` skips every
  target for active remote grading.
- The current first transfer-gate queue once any grade VM frees is now:
  `s9890_jsettlers_dagger_antireg_w1a.iter0002.pt`,
  `s9881_weighted_dagger_antireg_w1a.pt`,
  `s9878_weighted_dagger_antireg_c3.pt`,
  then final weighted-DAGGER checkpoints and the other JSettlers iter-2
  snapshots. Rerun status, summary, poll, and transfer planner immediately
  before launching any gate.
- Population summary remains strongly diagnostic: 122 strict rejects against
  current `s9752`, 2 g4 keepers, reject rate `0.9839`, and primary failure
  mode `opponent_regression:jsettlers_lite` with value-rollout second. This
  supports the active JSettlers-DAGGER repair wave, but promotion still needs
  all strict legs to be non-regressing.
- Research update: regularized PPO is still the right production path. A new
  relevant source, `Revisiting Regularized Policy Optimization...`
  (`https://arxiv.org/abs/2602.10894`), supports the same conservative
  direction as EMAgnet: stronger KL/entropy control before adding more search
  complexity. PyTAG (`https://arxiv.org/abs/2307.09905`) is useful evidence
  that tabletop RL is brittle and evaluation must stay multi-opponent. Do not
  add another broad PPO fanout; choose the next branch by the failing leg:
  JSettlers-only failure gets JSettlers-DAGGER continuation, value-rollout
  failure gets `tactical_rollout_guard_repair`, and broad KL drift gets a
  low-learning-rate high-KL/entropy repair recipe using existing PPO knobs.

## 2026-06-27 07:40 PDT Status

- No local self-play training or local game grading was launched. Local work
  was controller/status polling, no-op VM planning, controller unit tests,
  code review, and primary-source research.
- Local controller state is clean: `active_count: 0`, with no local
  `train_ppo.py`, `grade_agent.py`, Claude-mini, or subagent worker
  processes.
- Fresh fleet state remains saturated: 10 active VM trainers, 297 visible
  candidate checkpoints, 10 active strict remote gates, 811 completed grade
  legs, and 168 completed summaries. All current trainers are still the
  weighted-DAGGER anti-regression wave.
- Weighted-DAGGER has advanced while running: `s9881` is visible at
  `iter0006`, `s9878` has a final `.pt` visible, and the other active
  weighted-DAGGER workers have fresh iter-6 or iter-4 snapshots. The logs
  show top-advantage filtering plus sample-weighted DAgger anchor rows, not
  idle jobs.
- Active strict gates still fill every grade target:
  `s9861_topadv_repair_w1a.iter0004.pt` g12 on `c1`,
  `s9861_topadv_repair_w1a.iter0008.pt` g4 on `c2`,
  `s9861_topadv_repair_w1a.pt` g4 on `c3`,
  `s9865_value_rollout_repair_c1.iter0002.pt` g4 on `c4`,
  `s9869_value_rollout_repair_w4b.pt` g4 on `w1a`,
  `s9866_value_rollout_repair_c2.iter0002.pt` g4 on `w1b`,
  `s9868_value_rollout_repair_c4.iter0002.pt` g4 on `w4a`,
  `s9871_topadv_keeper_continue_w1a.iter0004.pt` g4 on `w4b`,
  `s9874_weighted_dagger_antireg_w1a.pt` g4 on `w4c`, and
  `s9867_value_rollout_repair_c3.iter0002.pt` g4 on `w4d`.
- Latest planners remain correctly no-op:
  `remote_transfer_gate_plan_latest_s98.json` has `planned_count: 0` with
  `eligible_targets: []`; `remote_train_plan_tactical_rollout_guard_latest.json`
  has `planned_count: 0`; and the new
  `remote_code_sync_plan_tactical_latest.json` has `planned_count: 0` because
  every worker is still training.
- Added controller support for `plan-remote-code-sync` and
  `remote-sync-code`. This is VM-only launch support, not training: it plans
  code syncs only for clean workers missing recipe-required features, backs up
  remote files before copying, and refuses active train/grade workers unless
  `--allow-busy` is explicitly passed. Verification passed with
  `.venv/bin/python -m py_compile tools/gcp_fleet_controller.py` and
  `.venv/bin/python -m pytest tests/test_agent_grader.py -q` (`83 passed`).
- Next grade slot rule: rerun status/summary/poll first, then transfer-gate
  the highest priority fresh weighted-DAGGER snapshot. The current planner
  queue starts with `s9881_weighted_dagger_antireg_w1a.iter0006.pt`, then
  `s9878_weighted_dagger_antireg_c3.pt`, then
  `s9876_weighted_dagger_antireg_c1.iter0006.pt` and
  `s9880_weighted_dagger_antireg_w4d.iter0006.pt`.
- Next clean train slot rule: use `tactical_rollout_guard_repair` for one
  seed only. If the slot is not `w1a`, first run the new code-sync planner,
  sync with backups, repoll for `tactical_rollout_mixed: true`, then plan the
  train launch. Do not start another blind PPO fanout.
- Research critique update: PPO remains the right base learner for now;
  EMA/old-policy anchors remain mandatory; Expected-SARSA/Q should stay a
  critic variance-reduction lane rather than a policy-control lane; ExIt,
  Student-of-Games, DAGS, and PSRO are still later-stage tools after one
  tactical/weighted-DAGGER parent clears strict anti-regression gates.

## 2026-06-27 07:51 PDT Status

- No local self-play training or local game grading was launched. Local work
  was process cleanup checks, VM-status/planner refreshes, source inspection,
  and primary-source research. There are no Claude-mini/subagent sidecars and
  `local_controller_status_latest.json` is clean with `active_count: 0`.
- Fresh VM state from `gcp_poll_latest_s98.json`: all 10 workers are training
  and there are 316 visible candidate checkpoints. Active trainers are
  `s9887_jsettlers_dagger_antireg_c1`, `s9888_jsettlers_dagger_antireg_c2`,
  `s9886_weighted_dagger_antireg_c3`, `s9889_jsettlers_dagger_antireg_c4`,
  `s9890_jsettlers_dagger_antireg_w1a`,
  `s9891_jsettlers_dagger_antireg_w1b`,
  `s9892_jsettlers_dagger_antireg_w4a`,
  `s9893_jsettlers_dagger_antireg_w4b`,
  `s9894_jsettlers_dagger_antireg_w4c`, and
  `s9895_jsettlers_dagger_antireg_w4d`.
- Fresh strict-grade state still has 10 active remote gates and 168 completed
  summaries. Active gates are unchanged: the `s9861` g12/g4 escalations,
  value-rollout repair gates `s9865`, `s9866`, `s9867`, `s9868`, `s9869`,
  the `s9871_topadv_keeper_continue_w1a.iter0004` gate, and
  `s9874_weighted_dagger_antireg_w1a.pt`.
- Refreshed plans are intentionally no-op:
  `remote_train_plan_tactical_rollout_guard_latest.json` has
  `planned_count: 0` because every worker is training;
  `remote_code_sync_plan_tactical_latest.json` has `planned_count: 0` for the
  same reason; and `remote_transfer_gate_plan_latest_s98.json` has
  `planned_count: 0` because every target worker is running a remote grade.
  Treat any older `planned_count: 1` tactical plan as expired.
- The top queued transfer-gate targets once any grade VM frees are final
  weighted-DAGGER checkpoints, led by
  `s9881_weighted_dagger_antireg_w1a.pt`,
  `s9878_weighted_dagger_antireg_c3.pt`,
  `s9876_weighted_dagger_antireg_c1.pt`, and the remaining `s9877`-`s9885`
  finals. Launch only after rerunning status, summary, poll, and the transfer
  planner.
- The active JSettlers-DAGGER wave is a reasonable tactical-regression repair:
  it narrows the opponent to `jsettlers_lite`, lowers learning rate/clip,
  strengthens old-policy and EMA KL anchors, and raises DAgger weighting. The
  risk is overfitting to JSettlers, so promotion still requires strict legs
  against heuristic, `catanatron_value`, `jsettlers_lite`, and
  `value_rollout_search`.
- Research critique update from primary arXiv sources:
  policy-gradient reevaluation in imperfect-information games
  (`https://arxiv.org/abs/2502.08938`) and Big 2 self-play
  (`https://arxiv.org/abs/2605.28863`) support keeping PPO as the production
  learner; VRPO/Q-boosting (`https://arxiv.org/abs/2605.19235`) supports
  critic-side variance reduction but not unchecked actor steering; EMAgnet
  (`https://arxiv.org/abs/2606.23995`) supports adaptive EMA anchors; the
  Catan cross-dimensional paper (`https://arxiv.org/abs/2008.07079`) confirms
  JSettlers is a meaningful baseline; DAGS, ExIt, and Student of Games
  (`https://arxiv.org/abs/2605.14379`, `https://arxiv.org/abs/1705.08439`,
  `https://arxiv.org/abs/2112.03178`) remain next-stage search/reanalysis
  work after a non-regressing parent exists.

## 2026-06-27 07:27 PDT Status

- No local self-play training or local game grading was launched. Local work
  was fleet polling, no-op planning, source/code inspection, and
  primary-source research.
- Fresh fleet state: all 10 VMs are training and all 10 VMs are running
  remote strict gates. The poll reports 10 active VM trainers and 273 visible
  candidate checkpoints. Local controller status is clean:
  `active_count: 0`.
- Active trainers are the weighted-DAGGER anti-regression wave:
  `s9876_weighted_dagger_antireg_c1`,
  `s9877_weighted_dagger_antireg_c2`,
  `s9878_weighted_dagger_antireg_c3`,
  `s9879_weighted_dagger_antireg_c4`,
  `s9881_weighted_dagger_antireg_w1a`,
  `s9884_weighted_dagger_antireg_w1b`,
  `s9885_weighted_dagger_antireg_w4a`,
  `s9882_weighted_dagger_antireg_w4b`,
  `s9883_weighted_dagger_antireg_w4c`, and
  `s9880_weighted_dagger_antireg_w4d`. Their logs show live PPO rows with
  top-advantage filtering, DAgger anchors, and sample-weighted imitation.
- Active strict gates are still full:
  `s9861_topadv_repair_w1a.iter0004.pt` g12 on `c1`,
  `s9861_topadv_repair_w1a.iter0008.pt` g4 on `c2`,
  `s9861_topadv_repair_w1a.pt` g4 on `c3`,
  `s9865_value_rollout_repair_c1.iter0002.pt` g4 on `c4`,
  `s9869_value_rollout_repair_w4b.pt` g4 on `w1a`,
  `s9866_value_rollout_repair_c2.iter0002.pt` g4 on `w1b`,
  `s9868_value_rollout_repair_c4.iter0002.pt` g4 on `w4a`,
  `s9871_topadv_keeper_continue_w1a.iter0004.pt` g4 on `w4b`,
  `s9874_weighted_dagger_antireg_w1a.pt` g4 on `w4c`, and
  `s9867_value_rollout_repair_c3.iter0002.pt` g4 on `w4d`.
- Latest plans were intentionally no-op. `remote_gate_plan_latest_s98.json`
  and `remote_transfer_gate_plan_latest_s98.json` have `planned_count: 0`;
  the transfer planner has `eligible_targets: []` because every VM is already
  a strict-grade target. `remote_train_plan_tactical_rollout_guard_latest.json`
  also has `planned_count: 0` because every VM is training.
- The next gate candidate once any target frees is
  `s9881_weighted_dagger_antireg_w1a.iter0002.pt`; the transfer planner
  skipped it only for `no_target_worker`. Do not grade it locally.
- The next train candidate remains `tactical_rollout_guard_repair`, but only
  `w1a` currently advertises `tactical_rollout_mixed: true`. When a clean
  train slot opens elsewhere, sync the tactical teacher code with backups,
  repoll feature flags, then launch one seed only.
- Research critique update: the local `JSettlersLitePolicy` is a deterministic
  resource-plan/trade/robber scorer, so it is a meaningful tactical regression
  guard rather than just a weak random baseline. The Catan cross-dimensional
  network paper shows Catan-specific representation can beat JSettlers, but
  the current blocker is tactical regression, not raw model capacity. ExIt and
  Student of Games support search/reanalysis distillation after a stable
  parent exists. The 2026 policy-gradient reevaluation and Big 2 paper support
  keeping PPO as the main learner; VRPO should stay a critic-variance lane;
  EMAgnet supports keeping EMA/old-policy anchors in all serious branches.

## 2026-06-27 07:17 PDT Status

- No local self-play training or local game grading was launched. Local work
  was limited to VM/status polling, controller planner code, unit tests, and
  primary-source research.
- Fresh fleet state: 8 active VM trainers, 246 visible candidate checkpoints,
  and 8 active strict gates. Local controller status is clean:
  `active_count: 0`.
- Active strict gates remain:
  `s9861_topadv_repair_w1a.iter0004.pt` g12 on `c1`,
  `s9861_topadv_repair_w1a.iter0008.pt` g4 on `c2`,
  `s9861_topadv_repair_w1a.pt` g4 on `c3`,
  `s9865_value_rollout_repair_c1.iter0002.pt` g4 on `c4`,
  `s9866_value_rollout_repair_c2.iter0002.pt` g4 on `w1b`,
  `s9868_value_rollout_repair_c4.iter0002.pt` g4 on `w4a`,
  `s9871_topadv_keeper_continue_w1a.iter0004.pt` g4 on `w4b`, and
  `s9867_value_rollout_repair_c3.iter0002.pt` g4 on `w4d`.
- Added controller support for `plan-remote-transfer-gates`. This plans
  `remote-grade-from-worker` commands for checkpoints that live on busy
  training VMs, while requiring the target grade VM to be idle by default.
  The source can be busy; the target cannot be training, grading, or locally
  claimed unless explicitly overridden.
- Verification: `.venv/bin/python -m py_compile tools/gcp_fleet_controller.py`
  and `.venv/bin/python -m pytest tests/test_agent_grader.py -q` passed
  with `78 passed`. These are controller/unit checks only, not local games.
- Latest plans are no-ops:
  `remote_gate_plan_latest_s98.json` has `planned_count: 0`;
  `remote_transfer_gate_plan_latest_s98.json` has `planned_count: 0` and
  `eligible_targets: []`;
  `remote_train_plan_tactical_rollout_guard_latest.json` has
  `planned_count: 0`.
- The transfer planner's top next target is now
  `s9874_weighted_dagger_antireg_w1a.iter0006.pt`, not the older iter-2
  snapshot. It is on busy training VM `w1a`; when any clean grade VM opens,
  run `plan-remote-transfer-gates` again and launch the first generated
  `remote-grade-from-worker` command.
- Research critique update: keep PPO as the main learner, use Q/Expected-SARSA
  as critic variance reduction only, keep EMA/old-policy anchors in all
  serious recipes, and prioritize tactical/search teachers over raw Q-policy
  mixing. The next real training launch remains `tactical_rollout_guard_repair`
  once a train slot is clean.

## 2026-06-27 07:03 PDT Status

- No local self-play training or local game grading was launched. Local work
  was limited to process checks, VM polling, remote-grade planning, direct
  arXiv research, and documentation.
- Fresh fleet state: 9 active VM trainers, 226 visible candidate checkpoints,
  and 8 active strict gates. `w4d` is train-idle but grading, so there is no
  clean VM slot. The tactical recipe plan still has `planned_count: 0`.
- Targeted `w1a` poll confirms `s9874_weighted_dagger_antireg_w1a.pt` is live
  and has produced `s9874_weighted_dagger_antireg_w1a.iter0002.pt`. It should
  be strict-gated when a grade worker frees, but no active grade should be
  interrupted for it.
- Active strict gates:
  `s9861_topadv_repair_w1a.iter0004.pt` g12 on `c1`,
  `s9861_topadv_repair_w1a.iter0008.pt` g4 on `c2`,
  `s9861_topadv_repair_w1a.pt` g4 on `c3`,
  `s9865_value_rollout_repair_c1.iter0002.pt` g4 on `c4`,
  `s9866_value_rollout_repair_c2.iter0002.pt` g4 on `w1b`,
  `s9868_value_rollout_repair_c4.iter0002.pt` g4 on `w4a`,
  `s9871_topadv_keeper_continue_w1a.iter0004.pt` g4 on `w4b`, and
  `s9867_value_rollout_repair_c3.iter0002.pt` g4 on `w4d`.
- The only fresh positive remains
  `s9861_topadv_repair_w1a.iter0004.pt`: small strict g4
  `promote_candidate`, candidate weighted win rate `0.2500` versus champion
  `0.1923`. Its g12 escalation is already active; do not promote from g4.
- Recent rejections still fail the same legs: heuristic,
  `jsettlers_lite`, and especially `value_rollout`. This reinforces the next
  branch choice: tactical rollout guard first, then DAGS/reanalysis starts
  after a non-regressing parent exists.
- Current training-stack critique:
  `JSettlersLitePolicy` is mostly a resource-deficit/trade-plan heuristic;
  `ValueRolloutSearchPolicy` is good within a tactical class but can override
  that class. `TacticalRolloutMixedTeacherPolicy` is the right next recipe
  because it keeps the JSettlers/baseline action type and uses rollout scores
  for within-type ranking.
- Research additions checked from primary arXiv metadata:
  DAGS (`https://arxiv.org/abs/2605.14379`) supports intermediate-start
  self-play, Big 2 (`https://arxiv.org/abs/2605.28863`) supports PPO as the
  main imperfect-information learner over raw Q variants, and the 2025 policy
  gradient reevaluation (`https://arxiv.org/abs/2502.08938`) supports
  regularized PPO as a credible baseline. Keep Q/VRPO as critic calibration,
  not actor steering, until strict gates prove it safe.

## 2026-06-27 06:51 PDT Status

- No local self-play training or local game grading was launched. Local work
  was VM polling, remote-grade status, source changes, targeted unit tests,
  no-op planning, and research.
- Another controller synced the rollout-mixed files to all VMs. Fresh poll now
  reports `baseline_rollout_mixed: true` on every worker. The newly added
  `tactical_rollout_mixed` branch is local only and has not been synced.
- Current VM load is saturated: 9 active trainers, 217 visible candidate
  checkpoints, and 8 active strict gates. Active gates are:
  `s9861_topadv_repair_w1a.iter0004.pt` g12 on `c1`,
  `s9861_topadv_repair_w1a.iter0008.pt` g4 on `c2`,
  `s9861_topadv_repair_w1a.pt` g4 on `c3`,
  `s9865_value_rollout_repair_c1.iter0002.pt` g4 on `c4`,
  `s9866_value_rollout_repair_c2.iter0002.pt` g4 on `w1b`,
  `s9868_value_rollout_repair_c4.iter0002.pt` g4 on `w4a`,
  `s9871_topadv_keeper_continue_w1a.iter0004.pt` g4 on `w4b`, and
  `s9867_value_rollout_repair_c3.iter0002.pt` g4 on `w4d`.
- Added local next-method code: `TacticalRolloutMixedTeacherPolicy` and
  controller recipe `tactical_rollout_guard_repair`. This teacher preserves
  the baseline/JSettlers tactical action class and uses value-rollout search
  mainly to rank actions inside that class, directly targeting the repeated
  `jsettlers_lite` and `value_rollout` non-regression failures.
- Verification:
  `.venv/bin/python -m py_compile tools/train_ppo.py tools/gcp_fleet_controller.py src/catan_zero/rl/self_play.py src/catan_zero/rl/torch_ppo.py`
  plus targeted planner/teacher tests passed with `3 passed`.
- The tactical rollout plan was intentionally a no-op:
  `runs/self_play/remote_train_plan_tactical_rollout_guard_latest.json` has
  `planned_count: 0` because every worker is either training or grading. Do
  not kill or steal the active jobs. When a clean slot opens, sync
  `tools/train_ppo.py` and `tools/gcp_fleet_controller.py` with backups to
  one idle VM, confirm `tactical_rollout_mixed: true`, then launch a single
  `tactical_rollout_guard_repair` seed.

## 2026-06-27 06:42 PDT Status

- No local self-play training or local game grading was launched. Local work
  stayed to controller polling, planning, research, and documentation.
- Fresh VM poll shows 9 active trainers and 206 visible candidate checkpoints:
  `s9865_value_rollout_repair_c1`, `s9866_value_rollout_repair_c2`,
  `s9867_value_rollout_repair_c3`, `s9868_value_rollout_repair_c4`,
  `s9871_topadv_keeper_continue_w1a`, `s9872_value_rollout_repair_w1b`,
  `s9873_value_rollout_repair_w4a`, `s9869_value_rollout_repair_w4b`, and
  `s9870_value_rollout_repair_w4c`.
- Fresh strict gate summary has 168 completed decisions, 5 keepers, 163
  rejections, and 4 active gates:
  `s9861_topadv_repair_w1a.iter0004.pt` g12 on `c1`,
  `s9861_topadv_repair_w1a.iter0008.pt` g4 on `c2`,
  `s9861_topadv_repair_w1a.pt` g4 on `c3`, and
  `s9867_value_rollout_repair_c3.iter0002.pt` g4 on `w4d`.
- `s9861_topadv_repair_w1a.iter0004.pt` is the only new positive signal:
  small strict g4 `promote_candidate`, candidate weighted win rate `0.2500`
  versus champion `0.1923`, no recorded per-opponent regression in that small
  gate. Do not promote it from g4; the g12 escalation is already active.
- Most new resource-plan and top-adv candidates still fail on heuristic,
  `jsettlers_lite`, or `value_rollout`. That means plain top-advantage
  filtering and score-target PPO are not enough by themselves.
- `rollout_guard_score_repair` remains the next queued branch, but it was not
  launched: the no-op planner wrote
  `runs/self_play/remote_train_plan_rollout_guard_latest.json` with zero
  planned launches because every VM is training or grading. Remote feature
  flags still show `baseline_rollout_mixed: false`, so sync with backups is
  required before the first rollout-guard launch on a clean idle VM.
- Research-to-branch critique:
  ExIt and Student of Games support search/reanalysis distillation rather than
  blind self-play; VRPO supports Q/Expected-SARSA critic calibration for
  stochastic-policy variance, not unchecked Q-policy mixing; EMAgnet supports
  keeping EMA/old-policy regularization in every serious PPO branch; the
  Catan cross-dimensional network paper supports graph/history representation
  work, but only after strict gates prove no tactical regression.

## 2026-06-27 05:35 PDT Status

- No local self-play training or local grading was launched. Local checks found
  no local `tools/train_ppo.py`, no local `tools/grade_agent.py`, and no
  Claude-mini sidecars.
- `s9837_strict_repair_kl_w1a` is verified live on `catan-zero-w1a`, seed
  `9837`, checkpoint `runs/self_play/s9837_strict_repair_kl_w1a.pt`, with
  `anti_regression_mixed` opponents and `baseline_mixed` warmup rows.
- Current strict-repair VM sweep:
  `s9837_strict_repair_kl_w1a`,
  `s9840_strict_repair_kl_c4`,
  `s9841_strict_repair_kl_w1b`,
  `s9842_strict_repair_kl_w4a`,
  `s9843_strict_repair_kl_c1`,
  `s9844_strict_repair_kl_c2`,
  `s9845_strict_repair_kl_c3`,
  `s9846_strict_repair_kl_w4b`, and
  `s9847_strict_repair_kl_w4c`.
- Targeted post-launch poll verified `s9845`, `s9846`, and `s9847` are alive
  and writing warmup logs. The earlier full poll verified `s9840`-`s9844`.
- `s9837_strict_repair_kl_w1a.iter0002.pt` finished strict g4 and was rejected
  for `jsettlers_lite` regression. Current summary has 145 decisions, 4
  historical keepers, 141 rejections, and 5 active repair gates:
  `s9843_strict_repair_kl_c1.iter0002.pt`,
  `s9844_strict_repair_kl_c2.iter0002.pt`,
  `s9840_strict_repair_kl_c4.iter0002.pt`,
  `s9841_strict_repair_kl_w1b.iter0004.pt`, and
  `s9842_strict_repair_kl_w4a.iter0002.pt`. The newest rejects still point at
  tactical/resource-plan regression, especially `jsettlers_lite` and
  `value_rollout`; `s9836.iter0002` also regressed against heuristic.
- Local code now gives the next synced branch real score-margin targets:
  `HeuristicPolicy` and `JSettlersLitePolicy` expose deterministic
  `target_scores`, and `BaselineMixedTeacherPolicy` blends their normalized
  rankings. This is intended for the resource/trade-plan auxiliary lane and
  should be synced only when a safe VM slot is available.
- Next gate action: keep remote strict gates filled from fresh poll and
  grade-status snapshots as repair checkpoints arrive. No promotion from
  aggregate noise; require no regression on heuristic, `jsettlers_lite`,
  `catanatron_value`, or `value_rollout`.
- Research readout:
  moving-reference regularization is supported by EMAgnet/GARIP and is already
  in the repair recipe; VRPO remains a critic-calibration lane only; Global
  PSRO is the long-term population scheduler direction; the next local code
  work should add resource/trade-plan auxiliary signals or tactical module
  gates because generic PPO branches keep failing the same JSettlers/rollout
  legs.

## 2026-06-27 06:18 PDT Status

- No local self-play training or local grading was launched.
- Remote strict summary now has 158 decisions and 6 active gates. Resource-plan
  iter-2 checkpoints were rejected:
  `s9857.iter0002` failed `jsettlers_lite` and `value_rollout`,
  `s9858.iter0002` failed `value_rollout`,
  `s9859.iter0002` improved aggregate (`0.2308` candidate versus `0.1923`
  champion) but failed `value_rollout`, and `s9860.iter0002` failed heuristic,
  `jsettlers_lite`, and `value_rollout`.
- Active gates now include resource-plan iter-4 checks for `s9857` and `s9859`,
  plus top-adv repair iter-2 checks for `s9861`-`s9864`.
- Added the next queued branch:
  `BaselineRolloutMixedTeacherPolicy` in `tools/train_ppo.py` and
  `rollout_guard_score_repair` in `tools/gcp_fleet_controller.py`. It blends
  baseline resource/JSettlers scores with value-rollout search scores and is
  intended to repair the exact rollout regression seen in the resource-plan
  lane. Do not sync or launch it until a fresh VM poll shows free capacity.

## 2026-06-27 06:00 PDT Status

- No local self-play training or local grading was launched. Local process
  checks found no local `tools/train_ppo.py`, no local `tools/grade_agent.py`,
  and no Claude-mini sidecars.
- The final strict-repair gates completed and were rejected:
  `s9845_strict_repair_kl_c3.pt`, `s9846_strict_repair_kl_w4b.pt`, and
  `s9847_strict_repair_kl_w4c.pt` each scored `0.0` weighted win rate against
  `current_best_s9752_iter0002` and regressed on heuristic,
  `jsettlers_lite`, and `value_rollout`. Do not continue the old
  strict-repair family as-is.
- Added controller recipe `resource_plan_score_repair`. It keeps Q-policy
  mixing off, uses `baseline_mixed` warmup, `anti_regression_mixed` PPO
  opponents, stronger old-policy/EMA KL anchors, and raises
  `--imitation-score-coef` to `0.08` so the new heuristic/JSettlers-lite
  `target_scores` shape resource/trade-plan rankings.
- Synced `tools/train_ppo.py` and `src/catan_zero/rl/self_play.py` to the
  idle VM repos with remote backups. A fresh poll reports
  `baseline_score_targets: true` on all workers.
- Launched four VM-only resource-plan repair trainers from
  `current_best_s9752_iter0002`:
  `s9857_resource_plan_score_repair_c1` on `c1`,
  `s9858_resource_plan_score_repair_c2` on `c2`,
  `s9859_resource_plan_score_repair_c3` on `c3`, and
  `s9860_resource_plan_score_repair_c4` on `c4`. Post-launch poll shows all
  four processes live, writing `baseline_mixed` warmup rows with nonzero
  `score_loss`.
- First early checkpoint gate is active:
  `s9859_resource_plan_score_repair_c3.iter0002.pt` was copied from training
  VM `c3` to idle VM `w1a` and launched under strict g4 against
  `current_best_s9752_iter0002`. Continue gating iter-2 snapshots from the
  other resource-plan runs as they appear.
- Follow-up iter-2 gates launched:
  `s9857_resource_plan_score_repair_c1.iter0002.pt` is active on `w4b`, and
  `s9858_resource_plan_score_repair_c2.iter0002.pt` is active on `w4c`.
  `s9860_resource_plan_score_repair_c4.iter0002.pt` exists but should wait for
  the next free grade slot because the fleet is now busy with four
  resource-plan trainers, four top-adv repair trainers from another
  controller, and active gates.
- Local verification after the controller change:
  `.venv/bin/python -m pytest tests/test_agent_grader.py tests/test_self_play.py -q`
  passed with `159 passed`.
- Long-term algorithm ladder:
  1. First repair the tactical/resource-plan regressions with explicit
     score-margin targets and hard non-regression gates.
  2. Use Q/Expected-SARSA only as critic calibration until strict gates stop
     showing JSettlers/value-rollout regressions.
  3. Add guided search/reanalysis only for checkpoints that clear the repair
     gate.
  4. Move to population scheduling/PSRO once there are multiple
     non-regressing policies instead of many rejected PPO snapshots.

Research anchors for this direction: Catan cross-dimensional network
representation work (`https://arxiv.org/abs/2008.07079`), Expert Iteration
search distillation (`https://arxiv.org/abs/1705.08439`), Student of Games
guided self-play/search (`https://arxiv.org/abs/2112.03178`), DeepNash-style
regularized self-play (`https://arxiv.org/abs/2206.15378`), VRPO critic
variance reduction (`https://arxiv.org/abs/2605.19235`), Generals.io
high-throughput filtering/EMA (`https://arxiv.org/abs/2606.23348`), and Global
PSRO population scheduling (`https://arxiv.org/abs/2605.28273`).

## 2026-06-27 05:17 PDT Status

- No local self-play training or local grading was launched.
- Current VM trainers:
  `s9832_pfsp_rollout_teacher_c1` on `c1`,
  `s9836_pfsp_klent_control_c3` on `c3`,
  `s9835_s9829_antireg_repair_w4b` on `w4b`, and
  `s9834_s9829_search_continue_w4c` on `w4c`.
- New VM-side branch launched:
  `s9836_pfsp_klent_control_c3`, seed `9836`, remote PID `1109926`.
  It starts from `current_best_s9752_iter0002`, uses `pfsp_mixed`, Q-policy
  mixing off, stronger old-policy/EMA KL, higher entropy, and
  `--gae-lambda 0.90`. Post-launch poll confirms the process is alive and
  warmup rows are writing.
- Active remote gates now include:
  `s9831_search_qcal_champ_c3.iter0004.pt` on `c2`,
  `s9832_pfsp_rollout_teacher_c1.iter0002.pt` on `c4`,
  `s9829_qadv_rollout_c4.iter0003.pt` on `w4a`, and same-minute preflight
  found `s9833_s9829_strict_continue_w1b.iter0003.pt` active on `w1a`.
- Added `strict_repair_kl` to the controller as the next anti-regression
  branch: `baseline_mixed` short warmup, `anti_regression_mixed` opponents,
  Q-policy mixing disabled, old-policy/EMA KL anchors, and mild imitation
  margin pressure. Targeted planner tests pass.
- `s9837_strict_repair_kl_w1a` was planned but not launched because `w1a`
  was actively grading. With four trainers and four active gates, treat the
  fleet as saturated at the intended 8-workload level.
- Research/readout:
  regularized reverse-KL plus entropy is the stability lane; GAE variance in
  imperfect-information self-play argues for critic-only Q calibration unless
  gates prove otherwise; the local `jsettlers_lite` policy is resource-deficit
  and trade-plan based, so promotion gates must keep JSettlers/value-rollout
  non-regression as hard constraints.

## 2026-06-27 05:02 PDT Status

- No local training or local grading was launched.
- `s9829_qadv_rollout_c4.iter0003` produced the first fresh small-gate
  `promote_candidate` signal against `current_best_s9752_iter0002`
  (`0.2308` weighted win rate vs champion `0.1923`). Do not promote from this:
  g12 escalation is active on `catan-zero-w4a`.
- Active strict remote gates now include:
  `s9829_qadv_rollout_c4.iter0003.pt` g12 on `w4a`,
  `s9830_strict_qcal_champ_w1b.iter0002.pt` on `w4b`,
  `s9826_graph_history_value_w4c.pt` on `w4c`,
  `s9831_search_qcal_champ_c3.iter0002.pt` on `c2`, and
  `s9830_strict_qcal_champ_w1b.iter0004.pt` on `w1a`.
- Added `pfsp_klent_control` to `tools/gcp_fleet_controller.py`: a
  PFSP/moving-reference PPO recipe with Q-policy mixing off, stronger KL/EMA
  anchors, higher entropy, and `--gae-lambda 0.90`. Targeted planner tests pass.
- `s9833_pfsp_klent_control_c2` was planned but not launched because preflight
  found `c2` already running a strict gate. `w1a` was also checked and was
  already gating. No duplicate compute was started.

## 2026-06-27 04:50 PDT Status

- No local self-play training or local grading was launched.
- No `claude-mini-spawn` sidecars were found; broad Claude daemon processes
  were not killed because they may belong to the user's interactive tooling.
- Full VM poll refreshed in `runs/self_play/gcp_poll_latest_s98.json`: 6 active
  VM trainers, 61 visible candidate checkpoints.
- Current active VM training now includes:
  `s9828_qadv_antireg_c2`,
  `s9831_search_qcal_champ_c3`,
  `s9829_qadv_rollout_c4`,
  `s9830_strict_qcal_champ_w1b`,
  `s9826_graph_history_value_w4c`, and
  `s9827_graph_history_rollout_w4d`.
- Current active remote strict gates are:
  `s9823_qcal_antireg_w1a.pt`,
  `s9824_qcal_antireg_w4a.pt`, and
  `s9825_qcal_antireg_w4b.pt`.
- A stale gate plan for `s9822_qcal_antireg_c3.pt` was blocked by same-minute
  preflight because `c3` had already been filled by another controller. No
  duplicate gate was launched.
- New VM-side training launch:
  `s9832_pfsp_rollout_teacher_c1`, seed `9832`, remote PID `991010`.
  It is a champion-initialized PFSP branch with rollout-teacher warmup,
  old-policy KL, EMA-policy KL, and Q-policy mixing disabled. Post-launch poll
  confirms `train_ppo.py` is running on `c1`; follow-up poll shows warmup rows
  with `teacher: value_rollout_search`.
- Research guidance added for the next wave:
  favor moving-reference PPO regularization plus payoff-sampled opponent
  pressure; use Q/Expected-SARSA only as critic calibration until strict gates
  disprove regressions; gate graph-history branches before using them as
  parents.

## 2026-06-27 04:39 PDT Status

- Added a default remote-side busy guard to `tools/gcp_fleet_controller.py`
  `remote-train`: if any `tools/train_ppo.py` process is already active on the
  target VM at SSH execution time, the launcher now returns a JSON
  `worker_busy` skip instead of starting a second trainer. Use `--force` only
  for a deliberate override.
- Verified the controller change with syntax compile and targeted tests:
  remote train default guard, `--force` override, and busy-worker train
  planner behavior all pass.
- Current live VM training from `runs/self_play/gcp_poll_latest_s98.json`:
  `s9820_pfsp_value_jsettlers_w1b`,
  `s9822_qcal_antireg_c3`,
  `s9823_qcal_antireg_w1a`,
  `s9824_qcal_antireg_w4a`,
  `s9825_qcal_antireg_w4b`,
  `s9826_graph_history_value_w4c`,
  `s9827_graph_history_rollout_w4d`,
  `s9828_qadv_antireg_c2`, and
  `s9829_qadv_rollout_c4`.
- Current active remote strict gate:
  `s9820_pfsp_value_jsettlers_w1b.iter0004.pt` on `catan-zero-c1`.
- Updated payoff summary has 75 strict rows against
  `current_best_s9752_iter0002`, all rejects. Weak-opponent priority remains
  `catanatron_value`, `jsettlers_lite`, `value_rollout_search`, then
  heuristic.
- No new training launch was made after the guard change because the fleet is
  effectively saturated. A dry strict-gate planner pass also produced
  `planned_count: 0`, so `c1` was left idle rather than duplicating stale or
  already-decided gates.

## 2026-06-27 04:30 PDT Status

- Local training and local grading remain disabled. The only local work was
  VM orchestration, trainer sync, source research, unit tests, and docs.
- Updated `tools/train_ppo.py` PFSP weights to match the deduped strict payoff
  table: value and `jsettlers_lite` remain the dominant pressure, but
  `value_rollout_search` now gets a larger share and heuristic is a smaller
  guardrail. Verified with `py_compile` and two opponent-factory/unit tests.
- Synced the updated trainer to the nine idle/stale workers with remote
  backups. Post-sync poll showed `pfsp_mixed: true` and trainer SHA
  `0cc1d0cb9d087b1cf128bb9b9eaf7b4d202c6b10` on the synced workers.
- Active VM training after collision cleanup:
  `s9820_pfsp_value_jsettlers_w1b`,
  `s9822_qcal_antireg_c3`,
  `s9823_qcal_antireg_w1a`,
  `s9824_qcal_antireg_w4a`,
  `s9825_qcal_antireg_w4b`,
  `s9826_graph_history_value_w4c`, and
  `s9827_graph_history_rollout_w4d`.
- The attempted PFSP launches on `w4c`/`w4d` were superseded by another
  controller's graph-history lanes. `s9827_pfsp_value_jsettlers_w4d` was
  stopped by exact checkpoint match to avoid same-VM contention; direct process
  checks show the graph-history job remains alive. `s9826_pfsp_value_jsettlers_w4c`
  logged warmup rows and then terminated after the graph-history lane appeared.
  Treat `s9820_pfsp_value_jsettlers_w1b` as the active PFSP lane.
- `c1`, `c2`, and `c4` are not training slots right now; they are still marked
  as active remote-grade workers for blend checkpoints
  `s9821_blend_s9806_a0p05`, `s9821_blend_s9806_a0p1`, and
  `s9821_blend_s9806_a0p2`.
- Next action: let the active VM jobs produce iter-2 checkpoints, then launch
  strict remote gates only for fresh snapshots. Do not promote any checkpoint
  unless it beats `current_best_s9752_iter0002` without `jsettlers_lite`,
  value, or value-rollout regression.

## 2026-06-27 04:22 PDT Status

- No local self-play training or local grading was launched. Local work stayed
  limited to VM orchestration, VM polling, source research, and documentation.
- No Claude-mini/subagent sidecars are running.
- `catan-zero-w1b` was preflighted twice before launch:
  `runs/self_play/gcp_poll_w1b_recheck.json` reported 0 train processes,
  `runs/self_play/remote_grade_status_w1b_recheck.json` reported 0 active
  remote grades, and the remote trainer exposed the required `pfsp_mixed`,
  `old_policy_kl`, and `ema_policy_kl` feature flags.
- Launched VM-side training:
  `s9820_pfsp_value_jsettlers_w1b`, seed `9820`, remote PID `916768`,
  log `runs/self_play/logs/s9820_pfsp_value_jsettlers_w1b.log`.
  Post-launch poll `runs/self_play/gcp_poll_w1b_postlaunch.json` confirms 1
  train process on `w1b` with checkpoint
  `runs/self_play/s9820_pfsp_value_jsettlers_w1b.pt`.
- The `s9820` run is champion-initialized from
  `runs/self_play/champions/current_best_s9752_iter0002.pt`, uses
  `--opponents pfsp_mixed`, disables Q policy mixing, and keeps old-policy
  plus EMA policy KL regularization. Its early log is in value-teacher warmup
  and has emitted clean warmup rows.
- Latest compact strict summary has 0 active remote grades, 115 completed
  decisions, 3 historical keepers, and 112 rejects. Fresh reanalysis rejects:
  `s9811_reanalysis_rollout_c1.reanalysis.pt` tied aggregate below threshold;
  `s9812`, `s9813`, `s9814`, `s9815`, `s9816`, `s9817`, and `s9819`
  reanalysis variants regressed on `jsettlers_lite`, heuristic, value, or
  value-rollout legs. No promotion candidate.
- Current payoff evidence remains conservative: the deduped population payoff
  summary has 72 strict pair rows versus `current_best_s9752_iter0002`, all
  rejects. PFSP priority is now `catanatron_value`, `jsettlers_lite`,
  `value_rollout_search`, then heuristic. The top rejected branch still fails
  on `jsettlers_lite`; do not promote ties.

## 2026-06-27 04:10 PDT Status

- Full VM poll is refreshed in `runs/self_play/gcp_poll_latest.json`:
  10 VM trainers are active and 303 candidate checkpoints are visible.
- Active training:
  `s9805_warmup_jsettlers_agree_c1`,
  `s9807_warmup_baseline_agree_c2`,
  `s9806_warmup_rollout_valueblend_c3`,
  `s9798_warmup_only_baseline_c4`,
  `s9799_graph_history_teacher_w1a`,
  `s9809_warmup_rollout_w1b`,
  `s9803_warmup_only_baseline_w4a`,
  `s9801_warmup_only_jsettlers_w4b`,
  `s9802_warmup_only_rollout_w4c`, and
  `s9810_warmup_baseline_w4d`.
- Remote strict grading is active on 2 warmup checkpoints:
  `s9809_warmup_rollout_w1b.warmup0008` and
  `s9810_warmup_baseline_w4d.warmup0008`.
- New strict rejects include `s9798_warmup_only_baseline_c4.warmup0032`,
  `s9799_graph_history_teacher_w1a.warmup0032`,
  `s9801_warmup_only_jsettlers_w4b.warmup0032`,
  `s9802_warmup_only_rollout_w4c.warmup0040`,
  `s9803_warmup_only_baseline_w4a.warmup0032`,
  `s9804_rollout_anchor_lowdrift_w4d.iter0001`,
  `s9805_warmup_jsettlers_agree_c1.warmup0008`,
  `s9806_warmup_rollout_valueblend_c3.warmup0008`, and
  `s9807_warmup_baseline_agree_c2.warmup0008`; no promotion candidate.
- Population payoff summary now has 166 strict rows against
  `current_best_s9752_iter0002`; all are rejects. Weakness priority remains
  `catanatron_value` first (`0.9688`), `jsettlers_lite` second (`0.9292`),
  then heuristic (`0.8238`) and `value_rollout_search` (`0.7972`).
- Dry VM planners are conservative: `remote_train_plan_latest.json` is
  `planned_count: 0`, `next_seed: 9811`; `remote_gate_plan_latest.json` is
  also `planned_count: 0`.
- The trainer planner now blocks automatic seed selection from partial polls.
  Use a full-fleet poll or an explicit `--seed` before any launch.

## 2026-06-27 03:56 PDT Status

- Full fleet poll is refreshed in `runs/self_play/gcp_poll_latest.json`:
  10 VM trainers are active and 303 candidate checkpoints are visible.
- Remote strict grading is currently idle. Latest new rejects:
  `s9796_jsettlers_repair_c2.iter0002` regressed on `jsettlers_lite` and
  `value_rollout`; `s9804_rollout_anchor_lowdrift_w4d.iter0001` regressed on
  `jsettlers_lite`.
- Population payoff summary now has 50 strict rows against
  `current_best_s9752_iter0002`; all are rejects. Weakness priority remains
  `catanatron_value` first (`0.9800`), `jsettlers_lite` second (`0.9250`),
  then heuristic (`0.8250`) and `value_rollout_search` (`0.7850`).
- A concurrent controller filled the newly free `w1b` slot after the PFSP
  relaunch. Current `w1b` process is
  `s9809_warmup_rollout_w1b.pt`, seed `9809`, with rollout-teacher warmup rows.
- The short-lived PFSP launches `s9805_pfsp_value_jsettlers_w1b` and
  `s9808_pfsp_value_jsettlers_w1b` are both terminated. `s9808` produced one
  clean warmup row with `teacher: catanatron_value`, but it is not live and is
  not a promotion candidate.
- `w1b` still has the synced trainer with `pfsp_mixed: true`, so the next PFSP
  attempt should use the next free VM slot and seed `9811` or later. Current
  dry train plan is `planned_count: 0` because every worker is busy.
- Do not repeatedly stop/relaunch into another controller's active sweep. Let
  the current VM jobs run, then strict-gate only fresh non-warmup checkpoints.

## 2026-06-27 03:51 PDT Status

- No local self-play training or local grading was launched. Local work was
  limited to orchestration, VM polling, one remote trainer sync, payoff summary
  refresh, and documentation.
- No Claude-mini/subagent sidecars are running.
- Fresh full-fleet poll showed 9 live VM trainers and 300 visible candidate
  checkpoints before the new launch. `catan-zero-w1b` was the only idle,
  non-grading worker.
- Remote strict grading has 1 active gate:
  `s9796_jsettlers_repair_c2.iter0002.pt` on `catan-zero-c2`.
- New strict rejects against `current_best_s9752_iter0002`:
  `s9795_teacher_anchor_baseline_c1.iter0002` regressed on heuristic and
  `jsettlers_lite`; `s9797_rollout_teacher_anchor_c3.iter0002` regressed on
  `jsettlers_lite`. No promotion candidate.
- Population payoff summary now has 48 strict rows against
  `current_best_s9752_iter0002`; all are rejects. Weakness priority remains
  `catanatron_value`, `jsettlers_lite`, heuristic, then
  `value_rollout_search`.
- `catan-zero-w1b` was synced with the local `tools/train_ppo.py`, then
  re-polled feature-aware. The remote trainer now reports `pfsp_mixed: true`
  and hash `58a73df670eb981c0c9ce95bf52e2d1abd72384d`.
- A short-lived `s9805_pfsp_value_jsettlers_w1b` VM launch was stopped after
  the full-fleet refresh showed another controller had already occupied seeds
  `9805`-`9807` with warmup branches. The stopped run left only a terminated
  startup log and is not a promotion candidate.
- A relaunch as `s9808_pfsp_value_jsettlers_w1b` also started cleanly with
  champion init, value warmup, `--opponents pfsp_mixed`, old-policy KL, EMA
  policy KL, and Q policy mixing off. It was later terminated when another
  controller occupied `w1b` with `s9809_warmup_rollout_w1b`.

## 2026-06-27 03:40 PDT Status

- Controller polling now defaults to `s9` so `s980x` trainers are visible.
  The earlier `s97` poll missed `s9803` and briefly produced a stale PFSP
  launch plan; `runs/self_play/remote_train_plan_latest.json` has been
  refreshed back to `planned_count: 0`.
- Full fleet is saturated with 10 live VM trainers:
  `s9795_teacher_anchor_baseline_c1`,
  `s9796_jsettlers_repair_c2`,
  `s9797_rollout_teacher_anchor_c3`,
  `s9798_warmup_only_baseline_c4`,
  `s9799_graph_history_teacher_w1a`,
  `s9792_s9752_antireg_jsettlers_teacher_w1b`,
  `s9803_warmup_only_baseline_w4a`,
  `s9801_warmup_only_jsettlers_w4b`,
  `s9802_warmup_only_rollout_w4c`, and
  `s9804_rollout_anchor_lowdrift_w4d`.
- Remote strict grading is idle. New strict rejects:
  `s9791_s9752_antireg_baseline_teacher_w4d.iter0006` regressed against
  `jsettlers_lite`; `s9792_s9752_antireg_jsettlers_teacher_w1b.iter0004`
  regressed against heuristic and `jsettlers_lite`. No promotion candidate.
- Population payoff summary now has 46 strict rows against
  `current_best_s9752_iter0002`; all are rejects. Weakness order remains
  `catanatron_value`, `jsettlers_lite`, heuristic, then
  `value_rollout_search`.
- Next free-slot plan is PFSP, but only after a fresh `s9` one-worker preflight
  and remote code sync. At least W4A's active `/home/nickita/catan-zero`
  trainer copy lacked `pfsp_mixed` during the preflight.

## 2026-06-27 03:16 PDT Status

- Targeted preflight on `w1b`, `w4b`, and `w4c` found all previously free
  workers have been filled by another controller. Active additions:
  `s9792_s9752_antireg_jsettlers_teacher_w1b.pt`,
  `s9793_s9752_antireg_heuristic_teacher_w4b.pt`, and
  `s9794_s9752_antireg_baseline_q_w4c.pt`.
- No local training or local grading was launched. No Claude-mini/subagent
  sidecars are running or needed for this workflow.
- Remote strict grading is idle, but the safe gate planner has no eligible
  launches: current anti-regression snapshots are active/decided, or blocked
  by prior strict-regression failures.
- `runs/self_play/population_payoff_summary_latest.json` now has 34 strict
  rows versus `current_best_s9752_iter0002`; all are rejects. The live weakness
  order is `catanatron_value`, `jsettlers_lite`, heuristic, then
  `value_rollout_search`.
- Code support for the next branch now includes `--opponents pfsp_mixed`, a
  payoff-ledger-weighted opponent sampler. It should be used only on the next
  genuinely free VM slot after polling, not by stopping active training.

## 2026-06-27 02:49 PDT Status

- 2026-06-27 03:02 PDT refresh: full fleet poll shows 10 live VM training
  processes and 268 visible candidate checkpoints. The fleet is saturated, so
  do not launch additional training until a worker frees.
- Current live training is now an `anti_regression_mixed` sweep from
  `current_best_s9752_iter0002`: `s9782`, `s9783`, `s9784`, `s9785`,
  `s9786`, `s9787`, `s9788`, `s9789`, `s9790`, and `s9791`.
- Latest strict VM gates have no active remote grades. New results:
  `s9780_s9752_strictmix_champ_anchor_w1a.iter0002` rejected for heuristic
  regression, `s9781_s9752_strictmix_ema_noq_w1b.iter0002` rejected for
  `jsettlers_lite` regression, and
  `s9782_s9752_antireg_mix_w4a.iter0002` rejected because it only tied
  aggregate.
- Population payoff summary now has 26 strict pair rows against
  `current_best_s9752_iter0002`; all are rejects. Weakness priority remains
  `catanatron_value`, `jsettlers_lite`, heuristic, then
  `value_rollout_search`.
- `catan-zero-w4d` had a stale nested trainer bundle. Its remote
  `tools/train_ppo.py` was backed up and synced so `anti_regression_mixed`
  is supported there; the active `w4d` run is
  `s9791_s9752_antireg_baseline_teacher_w4d.pt`.

- Fresh full-fleet poll: 9 live VM training processes, 237 visible candidate
  checkpoints.
- A later one-worker preflight on `catan-zero-w1a` found that another
  controller had already filled the previously free slot with
  `s9780_s9752_strictmix_champ_anchor_w1a.pt`, opponents `strict_mixed`, seed
  `9780`, with `iter0002` already present. No new training launch was started
  from this controller.
- No Claude-mini/subagent sidecars are running locally; do not start them for
  this workflow.
- Active remote strict gates:
  - `s9776_s9752_jsettlers_direct_c1.iter0002.pt` on `catan-zero-c1`.
  - `s9772_s9752_long_lowkl_c2.iter0004.pt` on `catan-zero-c2`.
  - `s9777_s9752_jsettlers_ema_c3.iter0002.pt` on `catan-zero-c3`.
  - `s9774_s9752_strictmix_lowkl_c4.iter0002.pt` on `catan-zero-c4`.
  - `s9775_s9752_strictmix_ema_w4a.iter0002.pt` on `catan-zero-w4a`.
  - `s9778_s9752_strictmix_rollout_teacher_w4b.iter0002.pt` on
    `catan-zero-w4b`.
  - `s9779_s9752_strictmix_mild_q_w4c.iter0002.pt` on `catan-zero-w4c`.
- `s9770_s9752_rollout_teacher_w1a.pt` is rejected by strict g4 versus
  `s9752`: candidate weighted win rate `0.0000` versus champion `0.1923`, with
  heuristic, `jsettlers_lite`, and `value_rollout` regressions.
- `s9763_graph_reanalysis_score_w1a.pt` is rejected by strict g4 versus
  `s9752`: candidate weighted win rate `0.1154` versus champion `0.1923`, with
  `value_rollout` regression. The iter0002 gate also regressed on heuristic.
- `s9773_s9752_rollout_ema_w1b.iter0004.pt` tied the champion exactly under
  strict g4 (`0.1923` versus `0.1923`) and is not an escalation candidate.
- The stricter dry planner found no clean current-family gate to launch. The
  free `catan-zero-w1a` slot was only surfacing stale or already-rejected
  families, so no VM grade was launched. The slot was then occupied by
  `s9780_s9752_strictmix_champ_anchor_w1a`.
- The latest population-payoff summary is
  `runs/self_play/population_payoff_summary_latest.json`. Against
  `current_best_s9752_iter0002`, all 23 strict pair rows are rejects; the
  weakest aggregate opponent legs are `catanatron_value` and `jsettlers_lite`.
- The gate planner should be rerun before any new launch. The current latest
  files are `runs/self_play/gcp_fleet_poll_latest.json`,
  `runs/self_play/remote_grade_status_latest.json`,
  `runs/self_play/remote_grade_summary_latest.json`, and
  `runs/self_play/remote_gate_plan_latest.json`.

## Gate Rules

A branch is kept only if it passes the strict grader against the current champion.

First pass:

```text
strict profile
4 games per opponent
opponents: heuristic, jsettlers_lite, value, value_rollout
vps_to_win: 4
max_decisions: 300
```

Escalation:

```text
strict profile
12 games per opponent
same opponent set
same game limits
```

Promote only after the larger gate has a positive aggregate delta and no unacceptable opponent regression.

## Active Branches

| Branch | Idea | Keep If | Kill If |
| --- | --- | --- | --- |
| `s9758_plain_lowkl_ema` | PPO continuation with EMA policy KL regularization. Tests whether EMA anchoring improves stability over plain old-policy KL. | Beats `s9752` on strict smoke, then passes g12. | Regresses against `jsettlers_lite` or `value_rollout`, or only ties aggregate. |
| `s9762_adaptive_league_lowkl_c2` | Adaptive opponent mix. Tests whether tougher opponent sampling improves robustness. | Improves aggregate without collapsing against heuristic/value opponents. | Beats only one narrow opponent or loses broad aggregate. |
| `s9763_graph_reanalysis_score_w1a` | Graph/history candidate plus reanalysis score supervision. Tests representation and search-distillation direction. | Improves despite extra architecture complexity. | Rejected at iter0002: aggregate regression and heuristic leg regression. |
| `s9760_graph_reanalysis_score_w4b` | Independent graph/reanalysis branch for replication. | Confirms `s9763` direction or finds a stronger seed. | Rejected at iter0002: aggregate regression and `jsettlers_lite`/`value_rollout` regressions. |
| `s9761_flat_search_dagger_w4c` | Flat candidate with search-mixed/DAGGER pressure. Tests whether search teacher improves current architecture. | Beats `s9752` while preserving value-rollout performance. | Rejected at iter0002: aggregate regression and losses against `jsettlers_lite`/`value_rollout`. |
| `s9764_adaptive_ema_lowkl_c4` | Adaptive league plus EMA regularization. Tests if EMA helps harder opponent mixtures. | Improves over both plain adaptive and plain EMA branches. | Same regressions as either ingredient branch. |
| `s9765_search_ema_dagger_w4a` | Search-mixed opponents plus EMA/DAGGER. Tests a conservative search-distillation path. | Improves aggregate and value-rollout leg. | Fails value-rollout or produces unstable policy. |
| `s9767_s9752_lowkl_continuation_c1` | Continue the recipe that produced `s9752`, initialized from `s9752`. Tests whether the gain compounds. | Later checkpoint beats `s9752` under the same gate. | Iter0002/0004/0006 show regression, meaning the recipe exhausted its gain. |
| `s9768_s9752_ema_continuation_c3` | Continue from `s9752` with EMA policy KL. Tests whether EMA should become the default continuation recipe. | Beats `s9767` or passes the strict gate with better stability. | Underperforms the plain continuation or regresses against `value_rollout`. |
| `s9769_s9752_mild_qcritic_w4c` | Mild Expected-SARSA Q critic from `s9752`, with Q as auxiliary learning signal only. Tests whether VRPO-style variance reduction helps without dominating policy updates. | Beats `s9752` or improves value/value_rollout legs without broad regression. | Any repeat of earlier Q-heavy collapse against `jsettlers_lite` or `value_rollout`. |
| `s9770_s9752_rollout_teacher_w1a` | Conservative `value_rollout` teacher from `s9752`. Tests whether shallow search imitation helps without changing opponent mix. | No longer active unless changed materially. | Rejected final strict g4: weighted 0.0000 vs champion 0.1923, with heuristic, `jsettlers_lite`, and `value_rollout` regressions. |
| `s9771_s9752_value_rollout_opponents_w4b` | Train from `s9752` against `value_rollout` opponents. Tests robustness against planning-style opponents. | Raises value_rollout leg while preserving aggregate. | Becomes too specialized and loses broad strict gate. |
| `s9774_s9752_strictmix_lowkl_c4` | Current champion continuation against strict mixed opponents. | Beats `s9752` without `jsettlers_lite` or `value_rollout` regression. | Repeats broad low-KL continuation regression. |
| `s9775_s9752_strictmix_ema_w4a` | Strict mixed continuation with EMA regularization. | Improves over `s9774` or preserves tactical legs better. | EMA fails to fix strict-mix instability. |
| `s9776_s9752_jsettlers_direct_c1` | Direct pressure against `jsettlers_lite`. | Improves the `jsettlers_lite` leg without losing aggregate/value-rollout. | Overfits JSettlers and loses other strict opponents. |
| `s9777_s9752_jsettlers_ema_c3` | JSettlers pressure plus EMA regularization. | Matches/improves `s9776` with better aggregate stability. | Same overfit pattern or no aggregate gain. |
| `s9778_s9752_strictmix_rollout_teacher_w4b` | Strict mixed opponents with rollout-teacher pressure. | Only keep if it avoids the `s9770` rollout-teacher collapse. | Any heuristic/JSettlers/value-rollout regression. |
| `s9779_s9752_strictmix_mild_q_w4c` | Strict mixed opponents with mild auxiliary Q. | Improves value/value-rollout without JSettlers regression. | Repeats the Q/VRPO collapse pattern. |
| `s9780_s9752_strictmix_champ_anchor_w1a` | Current champion strict-mix anchor branch launched by another controller. | Beats `s9752` with no tactical/value regression. | Repeats current strict-mix weak legs against `catanatron_value` or `jsettlers_lite`. |
| `s9782`-`s9791` anti-regression sweep | Current saturated VM sweep from `s9752`, using `anti_regression_mixed` to attack `catanatron_value`/`jsettlers_lite` regressions. | Any iter checkpoint beats `s9752` under strict g4, then g12, with no tactical/value-regression leg. | Ties aggregate, improves one leg while losing another, or repeats Q/rollout-teacher collapse. |
| `s9801`-`s9804` warmup/anchor sweep | VM-side warmup-heavy baseline/JSettlers/rollout/low-drift anchor recipes launched by another controller. | Produces a non-warmup checkpoint that beats `s9752` under strict gates. | Warmup-only checkpoints fail strict gates or do not produce a real trained candidate. |
| next free slot: PFSP sampler branch | Ready to launch after a VM frees because `w1b` has verified `pfsp_mixed` support and local planning works. Use seed `9811` or later, `--opponents pfsp_mixed`, champion init, EMA/old-policy KL, and Q policy mixing off. | Produces non-warmup checkpoints that beat `s9752` under strict g4, then g12, with no tactical/value regression. | Launches into another controller's active slot, repeats the stopped `s9805`/`s9808` collision, or improves one weak leg while regressing heuristic/JSettlers/value-rollout. |

## Research Threads Being Tested

The active branches correspond to current game-RL ideas:

```text
regularized PPO / policy damping -> s9752, s9758, s9767
EMA policy regularization -> s9758, s9764, s9765
league/exploiter-style training -> s9762, s9764
search reanalysis / distillation -> s9761, s9763, s9765
graph/history representation -> s9760, s9763
```

References checked during this run:

```text
VRPO / Expected SARSA for imperfect-information self-play:
https://arxiv.org/html/2605.19235v1

PPO-EMAg / parameter-space EMA regularization:
https://arxiv.org/html/2606.23995v1

Data-Augmented Game Starts for imperfect-information exploration:
https://arxiv.org/abs/2605.14379

OpenSpiel AlphaRank support for nontransitive multi-agent evaluation:
https://openspiel.readthedocs.io/en/latest/alpha_rank.html

Catan graph/CNN representation baseline:
https://arxiv.org/abs/2008.07079

Catanatron external Catan benchmark:
https://github.com/bcollazo/catanatron

JSettlers2 historical rule-based benchmark:
https://github.com/jdmonin/JSettlers2

Expert Iteration / search distillation:
https://arxiv.org/abs/1705.08439

ReBeL public-belief search:
https://arxiv.org/abs/2007.13544

Global PSRO:
https://arxiv.org/abs/2605.28273
```

Next grading-system improvement:

```text
population payoff table
candidate-vs-champion gate summaries
nontransitive matchup tracking
AlphaRank-style ranking once enough pairwise data exists
PFSP-weighted VM training branch via --opponents pfsp_mixed
```

The grader is currently the main keeper: if a checkpoint does not beat the champion under the strict opponent suite, it is rejected even if training loss improves.

Research critique: PSRO, ReBeL, DAGS, and much of the regularized self-play
literature is primarily two-player zero-sum. Use their mechanisms as tools,
not their objective assumptions. For four-player general-sum Catan, the near
term plan is payoff-table/PFSP opponent sampling, tactical non-regression
gates, belief/search reanalysis as data generation, and AlphaRank only after
enough cross-play data exists. Recent Big 2 results are more directly relevant
because they study four-player imperfect-information self-play; the actionable
lesson is staged curriculum plus stronger opponent diversity, not a new
promotion metric. Recent policy-gradient re-evaluation and EMAgnet results also
support keeping regularized PPO as the baseline while improving population
selection and regularization.
