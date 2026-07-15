# Coherent-public teacher operator campaign

This is the executable Phase-2 decision for the next data wave. It compares
three search teachers while holding the checkpoint and every other search
field fixed:

1. `base_n128_d6`: n128 everywhere, D6 at legal width >=20.
2. `adaptive_n256_w20_d6`: n128 plus always-full n256 at width >=20.
3. `adaptive_n256_w40_d6`: n128 plus always-full n256 at width >=40.

n64 is not an arm. All three use the coherent public-belief single-tree
operator, public observations, one determinization, `c_scale=.1`,
`sigma_eval=.79`, lazy Rust chance spectra, and the same model bytes. The
adaptive budget is therefore the only causal dose.

The checkpoint is intentionally not hardcoded. Use the current production
champion path and SHA-256 from the sealed current-champion handoff. At the time
this campaign was added, production metadata named v5 (`6817ab...`) while f7
was an experimental baseline; silently defaulting to f7 would invalidate the
experiment.

## Single-host execution

```bash
cd /path/to/catan-zero-public
CHECKPOINT=/absolute/path/from/sealed-current-champion-handoff.pt
CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
OUT=/absolute/path/teacher-operator-coherent-v1

python tools/teacher_operator_campaign.py \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --out-dir "$OUT" \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --stage all
```

This creates one immutable panel of 64 real roots, runs four independent
search seeds per role/root, then plays 100 exact seed/color-swapped pairs per
adaptive threshold (400 games total). The final artifact is
`$OUT/teacher-operator-report.json`.

The panel is not the first 64 eligible states. It deterministically walks real
Rust-engine games and reserves 24 `play_turn` roots at legal width 2-19, 16 at
width 20-31, 8 at width 32-39, and 8 width-40+ opening placements. The final 8
roots are unconstrained. A live census of thousands of real champion states
found a maximum `play_turn` width of 39, so claiming a `play_turn` width-40
quota would be impossible. Width-40 activation is therefore explicitly sealed
and reported as opening-only for this distribution. Collection is bounded at
512 real game trajectories and fails rather than relaxing a quota.

## Parallel fleet execution

The stages are independently runnable. Start both H2H stages immediately on
separate clean hosts. Run `fixed-w20` on a B200 to create the root panel, copy
`real-roots.json` and its first report directly to the second B200, then run
`fixed-w40`. Do not route artifacts through a laptop.

```bash
# Host A (B200): creates the shared real-root panel.
python tools/teacher_operator_campaign.py --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" --out-dir "$OUT" \
  --stage fixed-w20 --device cuda

# Host B (B200), after host-to-host copying $OUT/real-roots.json.
python tools/teacher_operator_campaign.py --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" --out-dir "$OUT" \
  --stage fixed-w40 --device cuda

# Host C (H100s), independent of the root probe.
python tools/teacher_operator_campaign.py --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" --out-dir "$OUT" \
  --stage h2h-w20 --devices cuda:0,cuda:1,cuda:2,cuda:3

# Host D (H100s), in parallel with Host C.
python tools/teacher_operator_campaign.py --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" --out-dir "$OUT" \
  --stage h2h-w40 --devices cuda:0,cuda:1,cuda:2,cuda:3
```

Collect the four JSON reports and `real-roots.json` into one `$OUT` on a clean
host, then aggregate:

```bash
python tools/teacher_operator_campaign.py --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" --out-dir "$OUT" --stage aggregate
```

The aggregator rejects mixed checkpoints, mixed root panels, incomplete pairs,
truncations, hidden-state/PIMC search, missing D6, n64, or operator drift beyond
`n_full_wide`, its threshold, and `wide_roots_always_full`. It reports:

- cross-seed JS and emitted-action agreement globally and by phase/width;
- target/prior JS, target entropy/confidence, and completed-Q margins;
- exact simulations, D6-expanded evaluator rows, and wall cost;
- paired playing strength and whole-game adaptive-search overhead;
- a selected operator only when the gain is cost-bounded and supported by
  positive-Elo H1 or the preregistered stability proxy.

If paired H2H completed before the stratified panel contract was added, those
two H2H reports remain usable: they do not consume the fixed-root panel. Copy
only `h2h.adaptive_n256_w20_d6.json` and
`h2h.adaptive_n256_w40_d6.json` into a fresh output directory, rerun the two
fixed-root stages there, and aggregate. The old fixed-root reports and old
`real-roots.json` are intentionally rejected by the new quota contract.
