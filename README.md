# CatanZero

This repository consolidates the Catan-Zero source, development history,
planning documents, and external reviews. The original import came from one
B200 and two now-retired A100 hosts. The production data fleet is 64 H100s:
eight four-GPU nodes plus four eight-GPU nodes. Two separate eight-B200 nodes
provide evaluation, orchestration, and independent learner R&D lanes; FLEET.md
is the live inventory. Project #1 goal:
build the strongest Catan agent under the benchmark below.

## Hard rule

**All software must be complete and reviewed before any large training run
consumes GPU-hours.** Training is expensive and self-play generation data
compounds — a bug shipped into a multi-day generation run poisons everything
downstream of it. Land the fix, land the test, then generate.

## Pointer map

- `RL_AGENT_HANDOFF.md` — production RL operator runbook: release and artifact
  acceptance, fleet launch, seed allocation, corpus QA, DDP training, searched
  gates, promotion, rollback, and current integration gaps.
- `src/`, `tools/`, `tests/` — the Python package described below (search,
  training, self-play generation, promotion gate, H2H evaluation). Use the
  verified current worktree and its next immutable H100 release, not an old
  host branch or the obsolete `v1.0-deploy` tag.
- `docs/plans/` — the live planning documents: `CATAN_ZERO_MASTER_PLAN.md`
  (plan of record with a status table and per-recommendation verdicts),
  `CATAN_ZERO_ROADMAP.md`, and `CATAN_ZERO_RESEARCH_CHRONICLE.md` (the
  claims ledger reviewers read).
- `docs/reviews/` — eight external expert reviews (R1-R8) plus the internal
  2026-07-06 critique report; see `docs/reviews/README.md` for an index
  mapping each file back to its original filename and what it added.
- `tools/modal_gumbel_factory_gpu.py` — the GPU self-play generation
  factory (Modal/L4 fleet), recovered from a local-only mirror that was
  never committed anywhere.
- `rescue/` — untracked working-tree files rescued from each GPU host before
  they could be lost to reprovisioning, kept on separate `rescue/untracked-*`
  branches rather than merged into `master`; see `rescue/README.md`.
- Task tracking: Linear workspace, team **Catan**.
- Production H100 aliases are `c1` through `c8` and `h100-8a` through
  `h100-8d`.
  B200 control-plane aliases are separate. Host IPs live only in the
  uncommitted `$FLEET_CONF`; see FLEET.md. The retired A100 names below
  describe repository history, not active compute.

## Branches

The historical import happened simultaneously across three hosts with
independent local commits, so the repository preserves those branches:
`f60-value-squash` through `f80b-hygiene-harden`, `gen3-wheel-sync`,
`savefix`, `v3-combined-staging`, `integ-v3`, `opt-arch`, `opt-inference`,
`opt-rust-work` (B200 feature branches, each a `git worktree` sibling of
`~/catan-zero` sharing full history with `master`); `host-a100a/master` and
`host-a100a/integrated_master` (a100a's own `master`, which diverged from
B200's with a host-local sync commit, plus its `integrated_master` branch);
`host-a100b/master` and `host-a100b/integrated_master` (same, for a100b);
`rust-engine/master` and `rust-engine/gen3-wheel-sync` (`catanatron-rs`, the
standalone Rust engine — native featurizer + MCTS — pulled from its own
separate repo on B200, not a worktree of the main repo).

Those branches are provenance, not current deployment inputs. For operations,
follow RL_AGENT_HANDOFF.md and FLEET.md and require one immutable release.

## Not imported (and why)

- `~/gen3_gate_test_catan_zero` on B200 (5.6 GB, not a git repo — includes a
  full `.venv` and appears to be an old non-version-controlled snapshot of
  the working tree). Flagged for manual review rather than bulk-copied.
- `~/catanatron-upstream-git-backup` on B200: a local mirror of the public
  third-party repo `github.com/bcollazo/catanatron` (upstream dependency,
  not project code).
- `~/gh200_backup_20260628/catan-zero` on B200: a plain-file backup
  directory (not a git repo) from a decommissioned GH200 host generation;
  not inspected for unique content.
- Uncommitted local edits to already-tracked files on any host (e.g.
  `catanatron-rs`'s modified `pyproject.toml`/`Cargo.lock` on B200) — these
  are in-progress working-tree diffs, not untracked artifacts, and weren't
  captured by this import.

---

CatanZero is a research stack for building a top full-game Catan agent.

The target benchmark is `CatanBench-4P-Full-v1`:

- Standard four-player base Catan.
- Ten victory points.
- Full legal player trading through structured offers.
- Player-chosen discards.
- Robber placement, victim selection, and hidden resources/development cards.
- No free-form natural-language agreements in the primary benchmark.

The project is intentionally split into contracts before algorithms:

1. Certified simulator and replay contract.
2. Per-player observation and legal-action schemas.
3. Human-game importer and supervised foundation model.
4. Population self-play and exploiters.
5. Belief-aware search and search distillation.

The final public claim should be "strongest evaluated agent under
CatanBench-4P-Full-v1" until a permissioned Colonist/human evaluation exists.
Do not build stealth automation against live services.

## Current runnable environments

The main Colonist/self-play target is `catan_zero.rl.ColonistMultiAgentEnv`.
It uses Catanatron's rule engine but surfaces every current-player decision to
the caller instead of auto-playing opponent seats. It exposes:

- all four per-seat observations,
- serializable per-seat API payloads through `observation_payload(player)`,
- current-player legal actions, structured legal-action descriptions, and masks,
- `structured_legal_actions` plus `step_structured_action(action)` for
  Colonist-like action objects on top of stable integer indices,
- a simple playable loop: `reset()`, `valid_actions()`, `step(action)`,
- concrete domestic trades,
- strategic chat,
- open/wildcard/counteroffer negotiation state,
- a Colonist-like `trade_panel()` snapshot with waiting, accepted, rejected,
  countered, confirmable, and cancellable offer state,
- proposal-to-board-trade resolution,
- virtual Colonist-style timers and timeout fallbacks,
- a lightweight public `event_log` in `info` with hidden discard, robber-steal,
  and development-card-buy results redacted,
- `replay_trace()` frames that pair each redacted public event with safe
  per-seat observation payloads, rewards, and terminal flags for replay and
  offline training,
- `write_replay_jsonl(path)` and `catan_zero.rl.dump_replay_jsonl()` /
  `load_replay_jsonl()` for local imitation/RL datasets.

For turn-based self-play trainer integration, use
`catan_zero.rl.ColonistAECEnv`. It is a lightweight PettingZoo-style AEC adapter
around `ColonistMultiAgentEnv` with `possible_agents`, `agent_selection`,
`observe(agent)`, `last()`, `step(action)`, and `agent_iter()`.

The older one-seat wrapper is `catan_zero.rl.CatanZeroGymEnv`. It wraps
Catanatron's Gymnasium environment for one learning player against bot enemies
and extends its action space with structured player-to-player trade actions.
It exposes:

- `reset(seed=...)`
- `step(action)`
- `valid_actions()`
- `action_mask()` / `action_masks()`

The default config targets a Colonist-like base-game surface:

- Four players.
- Ten victory points.
- Player-chosen discard, robber, steal, and build actions from Catanatron.
- Structured domestic trade offers, accept/reject, confirm, and cancel.
- A per-turn offer cap to prevent RL agents from learning zero-cost trade spam.
- A public strategic chat side channel exposed through `post_chat()`,
  `post_chat_template()`, and Gym `info`.

Chat is intentionally separate from `step(action)`: it is a negotiation/event
stream for trade, robber, leader-blocking, and future LLM adapters, not a board
rule action. This keeps PPO-style action masks tied to legal game moves while
still preserving Colonist-like table talk as observable public context.

Important limitation: the multi-agent env is a local simulator for self-play
and benchmark development, not stealth live-site automation.

Smoke test:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python tools/smoke_multiagent_random.py --games 1 --players 4
.venv/bin/python tools/export_random_replay.py --output /tmp/catan_replay.jsonl --seed 42
.venv/bin/python tools/smoke_random_policy.py --games 5 --players 4
```

The lower-level `catan_zero.rl.CatanatronRLEnv` keeps a smaller flat action
catalog that mirrors Catanatron's default non-domestic-trade Gym surface. Keep
it for algorithm plumbing and fast baseline tests.

## Self-play

The first trainable loop is in `catan_zero.rl.self_play` with commands:

```bash
.venv/bin/python tools/train_self_play.py --bootstrap-episodes 8 --episodes 32
.venv/bin/python tools/evaluate_self_play.py --candidate heuristic --opponent random
```

See `docs/SELF_PLAY.md` for the current smoke result and next PPO upgrade.
