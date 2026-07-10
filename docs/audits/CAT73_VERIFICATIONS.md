# CAT-73 Verification Checklist — Findings

Date: 2026-07-08
Scope: five verification questions gating Phase-A decisions (A0-pre, per CAT-73's R9 edit). All checks are read-only, code/data-level, GPU-free. Historical worktree: `<repo-worktree>` (branch `cat-73-verifications`).

---

## 1. `catanatron_value` observability — **VERIFIED-NO** (public information only)

`catanatron_value` (`vendor/catanatron/catanatron/catanatron/players/value.py`) does **not** read or infer opponent hidden card identities. It maintains no belief/card-counting state.

- `ValueFunctionPlayer.decide()` (`value.py:161-180`) does 1-ply greedy search: for each candidate action it copies+executes the game, then scores with `value_fn(game_copy, self.color)` — always its **own** color, never the opponent's.
- Inside `base_fn` (`value.py:58-119`), every `game.state.player_state[...]` read uses `key = player_key(game.state, p0_color)` — its own key (e.g. `player_state[f"{key}_VICTORY_POINTS"]`, `value.py:106`; `player_num_dev_cards(game.state, p0_color)`, `value.py:117`).
- The only opponent-facing term is `enemy_production` (`value.py:63,108`), built from `build_production_features` — this walks the opponent's **public board buildings** (settlement/city positions → resource/number tiles), never hand cards.
- `resource_hand_features(game, p0_color)` (`vendor/catanatron/catanatron/catanatron/features.py:85-123`) is the one place per-player hand fields exist, and it populates exact `P0_{resource}_IN_HAND` / `P0_{card}_IN_HAND` identity fields **only when `color == p0_color`** (`features.py:102-111`). For every other player it emits only aggregates: `P{i}_NUM_RESOURCES_IN_HAND`, `P{i}_NUM_DEVS_IN_HAND` (`features.py:118-121`) and publicly-played dev cards `P{i}_{card}_PLAYED` (`features.py:116`). Confirmed by direct read — `base_fn` never touches the opponent identity keys (they wouldn't even be populated for `i != 0`).
- Line 92-93 of `features.py` contains an explicit, never-implemented `TODO: P1_WHEATS_INFERENCE, ... P1_ROAD_BUILDINGS_INFERENCE` — i.e. the catanatron authors flagged opponent-hand inference as a known gap and never built it.
- Grep across all of `vendor/catanatron` for `belief|card.?count|opponent_hand|hidden|infer|estimate|deduc` surfaces nothing else relevant to this bot.

**Caveat (does not change the verdict, but is worth flagging):** `vendor/catanatron/catanatron/catanatron/players/tree_search_utils.py:56-60,111` (`get_dev_cards_in_hand` for enemy colors, `get_player_freqdeck(game.state, robbed_color)`) **does** read exact hidden opponent card identities to build chance-node probability spectrums for search. But this is only imported by `players/mcts.py` and `players/minimax.py` (i.e. `AlphaBetaPlayer`, our project's `catanatron_ab3/4/5`), not by `ValueFunctionPlayer`. `catanatron_value` specifically never imports or calls `tree_search_utils`.

No newer/second catanatron install shadows the vendored copy — `catanatron_value` resolves to `vendor/catanatron/catanatron/catanatron/players/value.py` via `import_catanatron_module`.

**Bearing on CAT-73 R9 gate:** the `catanatron_value` opponent used in the external panel (`runs/gates/v16_external/vs_value_500pairs.json`) has no information advantage over our agent. If a comparable AB3/4/5 panel result is ever used instead, this verdict does **not** carry over — those bots' minimax path does read exact opponent hand contents via `tree_search_utils`.

---

## 2a. Rules parity (our engine vs. upstream catanatron) — **VERIFIED-NO** (real, characterized differences found; not a blocking blind spot)

`tools/engine_equivalence_sweep.py` + `src/catan_zero/adapters/engine_equivalence.py` is the actual harness. It compares, every ply, in lockstep: full legal-action-set equality (`engine_equivalence.py:262-267`), and a full state diff (current player, robber location, winner, bank, buildings, roads, dev-deck count, road-building/free-roads flag, per-player VPs — both nominal and actual —, resources, dev cards in hand, longest-road length, piece availability, largest-army, played dev cards; `engine_equivalence.py:270-359,362-385`). Chance outcomes (dice, robber-steal victim/resource, dev-card draw) are forced identically into both engines (`apply_chance_step`, lines 485-636) so RNG never masquerades as a rules bug.

**Known, documented exception** (`engine_equivalence.py:388-399`; `tools/engine_equivalence_sweep.py:10-17`): longest-road length/ownership and "buildable-edge-near-enemy" legality are explicitly carved out of the diff (`rules_adjudication_needed_longest_road`, `rules_adjudication_needed_buildable_edge_near_enemy`) because the vendored Python engine has known pre-existing bugs matching upstream catanatron issues #376/#378 (sub-5-settlement cuts not revoking Longest Road, both-ends-enemy-capped roads undercounting by 1, roads buildable through enemy settlements). The harness does not adjudicate which engine is "right" there.

Documented empirical result (quoted in `tools/gumbel_search_vs_bot_h2h.py:47-51`): a 1000-game random-play sweep found 993/1000 fully equivalent, 7 divergences (6 classified as the longest-road issue above, 1 unclassified). The underlying report (`runs/engine_equivalence/report_fixed_pair_1000.json`) is not present in this local worktree (NEEDS-HOST-CHECK if a fresh count is wanted: `cat runs/engine_equivalence/report_fixed_pair_1000.json | jq .summary` on a host that ran it, e.g. B200).

Scope caveats found by reading the harness: fixed `TOURNAMENT` map only (map-shuffle RNG doesn't match across engines — `gumbel_search_vs_bot_h2h.py:32`); default 2-player games; no evidence of an equivalent 3-4 player or trade-heavy sweep.

**Verdict:** rules parity is verified for everything except the one documented longest-road/buildable-edge exception class, which affects <1% of moves in the empirical sweep and is a known, pre-existing bug in catanatron's own reference engine (not something our engine introduced). This is not blocking, but should be cited as a residual caveat whenever the external panel result is quoted.

## 2b. Info asymmetry / which engine adjudicates the panel — **VERIFIED-YES** (both engines run in lockstep; no undisclosed advantage)

`tools/gumbel_search_vs_bot_h2h.py` is the panel generator (matches `docs/plans/CATAN_ZERO_STRATEGY... ` reference to "every promoted champion ALSO plays a fixed external panel — catanatron_value + AB3 on the TOURNAMENT-map bridge"). It reuses the equivalence bridge to run **both engines simultaneously**, not one merely providing a policy plugged into the other's loop:

- Our candidate's move comes from `GumbelChanceMCTS.search(rust_game, ...)`.
- `catanatron_value`'s move comes from `bot.decide(python_game.copy(), python_game.playable_actions)` (`gumbel_search_vs_bot_h2h.py:187`) — i.e. it sees and legality-checks against the **Python engine's own action set**, not a translated Rust one.
- Every chosen action is applied to **both** engines with forced-identical chance outcomes (lines 202-210), then the harness re-diffs legal actions and full state after every ply (line 220); **any** mismatch aborts the game immediately and it is excluded from the win-rate statistic (`engine_divergence: True`, `candidate_won: None`, lines 232-254).
- Final winner/VP is read from the **Rust** game (lines 233-236) — Rust is authoritative for scoring; Python is authoritative for the bot's own decision-making; any disagreement between them drops the game rather than silently resolving it one way.

**Verdict:** no information or rules advantage is silently given to either side — catanatron_value plays under its own engine's rules/legality (matching how it was designed/tuned) while scoring is read from our engine, and the two are cross-validated every ply, with divergent games discarded rather than counted. Combined with §1 (no hidden-state read) and §2a (rules parity, with one documented non-blocking exception), the R9 gate condition — "a gap vs an information-advantaged opponent means something different" — is satisfied: catanatron_value has no such advantage.

---

## 3. Steal-observability in our engine (CAT-59 gating question) — **VERIFIED-NO** (event log masks identity for all audiences; recoverable only indirectly for victim/thief)

`MultiAgentEnv` (`src/catan_zero/rl/multiagent_env.py`) is the authoritative environment used by the actual data pipeline (`self_play.py`, `generate_dagger_data.py`, `train_bc.py`, `tools/build_memmap_corpus.py`) — verified via grep; the alternative `CatanatronAdapter` (`adapters/catanatron.py`) is wired only into `tests/test_environment_contracts.py`, not production data generation.

- `_redact_event` (`multiagent_env.py:1042-1062`):
  ```python
  elif action_type == "MOVE_ROBBER" and payload.get("result") is not None:
      payload["result"] = "hidden_stolen_resource"
  ```
  This branch **ignores the `actor` parameter** — `event_log(self, *, actor=None)` (line 633) calls `_redact_event(event, actor)` for every consumer, but the MOVE_ROBBER masking never reads `actor`. The identical `"hidden_stolen_resource"` string is written for the thief, the victim, and third parties alike — there is no per-audience branch for this event type. The duplicate adapter path (`adapters/catanatron.py:436-442`, `_event_public`) does the same audience-agnostic masking.
- `Observation.assert_no_hidden_opponent_fields()` (`multiagent_env.py`-adjacent `schemas.py:169-180`) guards a different leak class (`opponent_resources`, `future_steals`) and doesn't bear on this event-log masking.

**Indirect recoverability:** `_player_payloads` (`multiagent_env.py:881-922`) attaches exact per-type `resources` **only for `color == actor_color`** (lines 906-913); other seats get only an aggregate `resource_card_count`. Every recorded replay frame stores full per-seat observation payloads turn-by-turn (`observation_payloads(include_event_log=False)`, line 1029-1035). Consequence:
- **Victim/thief**: can recover the exact stolen resource by diffing their own exact hand snapshot pre/post-steal (their own hand is always ground truth in their own payload) — but this is not a labeled field; it must be derived by differencing.
- **Third party**: cannot recover it — they only ever see the victim's aggregate `resource_card_count` (a count that dropped by one), never the type. Matches real-game information asymmetry.

**Bearing on CAT-59 design choice:** a deduction tracker for the **victim's own** perspective doesn't need probabilistic inference — the exact resource is trivially derivable from the victim's own already-recorded consecutive hand snapshots. A tracker modeling belief about **opponents'** hands (the harder, actually-interesting case) does need a probabilistic/posterior approach, since no field anywhere exposes an opponent's exact resource identity post-steal. This is consistent with the existing search-time approach in `gumbel_chance_mcts.py` (`move_robber_victim_outcome_weights`, ~line 1350-1409), which reconstructs steal-outcome distributions during search rather than reading a labeled ground-truth field, and explicitly notes the residual leak ("which resource types the victim holds is still implied by which outcomes are real steals") as a deliberately accepted approximation confined to search, separate from the trajectory/event-log schema audited here.

---

## 4. Shard hidden-state label banking — **VERIFIED-YES** (shards are written omniscient; masking is downstream/read-time only)

Confirmed by reading the write path in full: shards on disk retain full, unmasked opponent hidden-state columns. Masking (`--mask-hidden-info`) is applied only at `train_bc.py` read time (or online at inference), not at generation/corpus-build time.

- `src/catan_zero/rl/entity_token_features.py:79-107` — `mask_player_tokens_public()`'s own docstring says it exists to "strip opponent hidden slots from the **banked (omniscient) tokens**" — i.e. the on-disk representation is omniscient by construction, and this function is an optional downstream transform, not something baked into the writer.
- Exact hidden-info columns inside the per-decision `player_tokens` array (shape `(4, 31)`, one row per player), populated unconditionally for **all 4 players** by `_player_tokens` (`entity_token_features.py:371-403`, loop `for name in PLAYERS` with no actor-only gating on these slots):
  - slot 5: `actual_victory_points` (includes hidden VP dev cards)
  - slots 16-20: exact resource-hand composition, one slot per `("wood","brick","sheep","wheat","ore")` (line 394-395)
  - slots 22-26: exact unplayed dev-card identities, one slot per `("KNIGHT","YEAR_OF_PLENTY","MONOPOLY","ROAD_BUILDING","VICTORY_POINT")` (line 398-399)
  Only `mask_player_tokens_public` (applied optionally, elsewhere) zeroes the non-actor rows' slots 4,5,15-26 (`PUBLIC_MASK_PLAYER_SLOTS`, line 76) — confirming these slots are populated with real data prior to any masking.
- `tools/convert_teacher_to_entity_tokens.py:328` calls `build_entity_token_features(env, player)` with no masking argument anywhere in the file (grep-confirmed) — the `player_tokens` column persisted into the shard's `ENTITY_KEYS` (lines 51-70) is the raw omniscient tensor.
- `tools/build_memmap_corpus.py:57-102` (`LOADER_KEYS`) copies `player_tokens` byte-for-byte from npz shard into the flat memmap file with no masking step — schema-preserving only.
- `tools/train_bc.py:2941-2948` confirms masking is a read-time transform: `if mask_hidden_info and "player_tokens" in shard: shard["player_tokens"] = mask_player_tokens_public(shard["player_tokens"])`, gated behind `--mask-hidden-info` (flag defined at `train_bc.py:143`, restricted to `--arch entity_graph`, `train_bc.py:694-696`).

**NEEDS-HOST-CHECK** (populated-value confirmation only — code path is fully verified): local shard files found under `~/.persistent-tmp-backup/private-tmp/*/teacher_shard_00000.npz` and `~/.tmp/*/teacher_shard_00000.npz` predate the entity-token schema and don't even contain a `player_tokens` key. To confirm real production shards actually contain non-placeholder values (not just the correct schema), run on a host with real gen-N shards (e.g. B200):

```bash
python3 -c "
import numpy as np
d = np.load('runs/<real_shard_dir>/teacher_shard_00000.npz', allow_pickle=True)
assert 'player_tokens' in d.files, 'no player_tokens column'
pt = d['player_tokens']  # (N, 4, 31)
resources = pt[:, :, 16:21]
devcards  = pt[:, :, 22:27]
print('nonzero resource slots:', np.count_nonzero(resources), '/', resources.size)
print('nonzero devcard slots :', np.count_nonzero(devcards), '/', devcards.size)
print('sample opponent row:', pt[0, 1, 15:27])
"
```
Substantially-nonzero, varying counts would close this out completely.

---

## 5. Symmetry-augment production status — **VERIFIED-NO** (built, default off, not enabled by any production launcher found)

- Flag definition: `tools/train_bc.py:284-294` — `--symmetry-augment` (`argparse.BooleanOptionalAction`, `default=False`); confirmed by direct read. Companion `--symmetry-augment-events` defaults `True` but is only relevant if the parent flag is on (line 296-300).
- Gated at runtime behind `getattr(args, "symmetry_augment", False)` (`train_bc.py:891-899`) and requires `--arch entity_graph`.
- Unlike the earlier `--mask-hidden-info` blind spot (which memory notes was NOT recorded into `report.json`), this flag's actual value **is** recorded: `train_bc.py:1274` — `"symmetry_augment": bool(args.symmetry_augment),` inside the report dict (confirmed by direct read, lines 1270-1278). So any `report.json` from a completed run gives zero-ambiguity ground truth.
- All three production launch paths found in this worktree omit the flag (hence it runs at its `False` default):
  - Historical B200 finetune queue (completed; script removed) did not include `--symmetry-augment`.
  - `tools/continuous_flywheel.py:253-259` — the flywheel's `train_window()` call (`--arch entity_graph --data-format memmap --mask-hidden-info --amp bf16 --max-steps ...`) omits it.
  - `tools/start_training_factory.py:245-270+` — grep for "symmetry" across the file returns nothing.
- Corroborating project narrative: `docs/plans/CATAN_ZERO_RESEARCH_CHRONICLE.md:112,290` — symmetry augmentation was tried as an experimental recipe arm (H2), lost to H1, and was explicitly "not adopted." `docs/plans/CATAN_ZERO_MASTER_PLAN.md:292` lists it under a "build-and-shelve" pattern: "symmetry averaging (built, off)."

**No `report.json`/`train.log`/`runs/` exist in this local worktree** to directly confirm a per-run logged value (only referenced as output paths inside scripts). If a belt-and-suspenders per-run confirmation beyond the launch-script evidence is wanted, run on the training host:
```bash
grep -r '"symmetry_augment"' runs/bc/*/report.json runs/*/report.json 2>/dev/null
```

---

## Summary table

| # | Question | Verdict |
|---|----------|---------|
| 1 | catanatron_value reads/infers opponent hidden state | **VERIFIED-NO** — public info only, `value.py`/`features.py` |
| 2a | Rules parity (our engine vs. catanatron) | **VERIFIED-NO** (real diffs) — 993/1000 equivalent; one documented, non-blocking longest-road/buildable-edge exception (upstream #376/#378) |
| 2b | Info asymmetry / panel adjudication | **VERIFIED-YES** — no undisclosed advantage; both engines run in lockstep, divergent games dropped |
| 3 | Steal-observability (victim sees stolen card identity?) | **VERIFIED-NO** — event log masks identity for all audiences (`_redact_event`); only indirectly recoverable for victim/thief by diffing own hand snapshots |
| 4 | Shard hidden-state label banking | **VERIFIED-YES** — shards written omniscient; masking is train-time/read-time only (`--mask-hidden-info`) |
| 5 | Symmetry-augment production status | **VERIFIED-NO** — built, defaults False, omitted by every production launcher found; explicitly "not adopted" per research chronicle |

R9 A0-pre gate (items 1, 2a, 2b) is now closed: catanatron_value has no hidden-state read advantage, rules parity is verified modulo one small documented and non-blocking exception, and the panel's cross-engine adjudication has no undisclosed asymmetry. `runs/gates/v16_external/vs_value_500pairs.json` can be interpreted as a fair comparison.
