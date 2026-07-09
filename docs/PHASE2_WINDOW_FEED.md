# Phase-2 Window Feed — design notes (task #94)

2026-07-07, ml-czar v2. Companion to the implementation in tools/flywheel_feed_daemon.py,
tools/continuous_flywheel.py (ingest_feed_batches / build_round_corpus / --max-ckpt-lag),
src/catan_zero/rl/flywheel/replay_window.py (select-time staleness), tools/train_bc.py
(ConcatMemmapCorpus). Binding amendments honored: manual-first champion push; lineage
hold-warning valve (N=5) unchanged; feed batches carry the generating checkpoint md5
(registry-derived, verified against the fleet's live file every cycle).

## Data flow

    A100 fleet (16 GPUs, champion-v0 self-play, host-owned seed blocks 6.1B/6.2B)
        └── flywheel_feed_daemon.py  (B200): scan completed npz → md5 both sides →
            immutable memmap corpus per batch + feed_manifest.json {row_count,
            ckpt_version, checkpoint_md5, wave_roots, game_seed_range} + .ready
                └── continuous_flywheel.py round top: ingest_feed_batches() registers
                    corpus dirs into WindowedReplay (ckpt_version from manifest)
                        └── select(current_version, max_ckpt_lag=2) → in-window corpora
                            └── train_bc --data <comma-list> (ConcatMemmapCorpus)

Own-round generation (24 games) is built into window_corpus/round_NNN and registered the
same way — the window is corpus-granular; per-round build cost scales with NEW rows (T4).

## Throughput model (measured 2026-07-07)

- Fleet: ~4-9 games/hr/worker × 16 workers × 16 GPUs ≈ 1,000-2,300 games/hr ≈ 250-580k
  rows/hr ≈ 100-240k rows per 25-min round at steady state. Backfill: throttled to
  max_batch_shards=64/cycle/host ≈ 260k rows/cycle-pair ≈ absorbed over ~2h.
- Loop own-gen: 24 games ≈ 6k rows/round (~3-6% of ingest).
- Reuse math: steps = new_rows × target_reuse(6.0) / batch(4096). 165k fed rows/round →
  ~240 steps; backfill peak ~500k → ~730 steps (~15-20 min at ~1s/step for 35M bf16).

## Pool dilution (lead-flagged design consequence — NOT built yet)

Anti-forgetting opponent-pool games come only from the loop's own generation:
pool_fraction 0.20 × 24 games ≈ 4.8 games ≈ 1.2k rows/round. With the feed, the trained
mix is ~170k rows/round → pool share collapses 20% → ~0.7%.

Mitigation options, with math:
(a) Raise loop opponent_pool_fraction to 1.0: 6k pool rows/round → ~3.5%. Cheap but capped.
(b) RECOMMENDED: weight pool-flagged rows in the trainer. The loop's shards are
    identifiable (fleet shards carry no pool flag — their absence is the discriminator);
    the trainer already consumes per-row policy/value weight multipliers. Effective pool
    share s_eff ≈ k·p / (T + (k−1)·p) with p=pool rows, T=total; for target s_eff=5-10%
    at p=1.2k, T=170k → k ≈ 7-15 (cap at 10; recompute per round from the realized mix).
    No fleet changes, no new data path — a sampler/weight patch in train_window.
(c) At scale (after 2-3 promotions): dedicate 1-2 fleet GPUs to pool-opponent generation
    via the generator's existing --opponent-pool-manifest: 2/16 GPUs ≈ 12.5% of fleet
    rows ≈ true ~10% pool share, and pool games get fleet-quality search budgets.
Decision: (b) first (build trigger = first promotion), (c) when promotions are routine.
At champion v0 the dilution is tolerable: archive opponents (v-1/-2/-3) are pre-seed
nets, so the forgetting risk the pool guards against hasn't materialized yet.

## When does 24 games/round become the bottleneck? (lead design question)

Own-gen matters for (i) pool rows (see above) and (ii) the ONLY champion-v(k+1) data
between a promotion and the manual fleet rotation. Thresholds:
- Row share: own 6k vs fed 100-240k → own-gen is already <6% of ingest; as a DATA source
  it is never the bottleneck while the fleet runs — no reason to raise games-per-round
  for volume.
- Post-promotion freshness: with staleness lag 2, fed v(k) data stays trainable through
  two promotions, but reuse-math steps collapse to ~8-9/round if fed ingest were ever
  excluded — the correct lever is FAST fleet rotation (manual trigger + watchdog
  champion-pointer), not more loop games.
- Pool floor: if mitigation (b) is rejected, keeping pool share ≥5% by volume alone would
  need own-gen ≈ 0.05×T/0.2 ≈ 40-60 games/round (rounds lengthen ~2x; rejected — use (b)).
Conclusion: hold games-per-round at 24; revisit only if (b) and (c) both fail.

## 2-tier reuse (deferred by lead decision)

Uniform ConcatMemmapCorpus sampling is correct TODAY because every window row is
champion-v0 self-play — tiers would be a no-op. The tier boundary that will eventually
matter is loop-fresh v(k+1) vs fleet v(k) after a promotion; if post-promotion KL/val
trends show the stale mass drowning the fresh signal, add per-tier sampling budgets
(fresh-tier floor, e.g. ≥25% of each batch) in train_window's --data assembly. Watch,
don't build.

## Operational notes

- Backfill vs steady state is journal-separable: every ingest record carries wave_roots.
- Quarantine (.quarantined) is terminal until a human clears it; quarantined shards stay
  in the dedup registry so they are not re-pulled every cycle.
- Kill switches: loop-dir/STOP (orchestrator), feed/STOP (daemon), feed_daemon.lock
  (single instance).
- On promotion (manual runbook): update watchdog champion pointer on both hosts, update
  feed_config ckpt_version (the md5 contract then verifies the rotation actually
  happened before any new data is ingested).
