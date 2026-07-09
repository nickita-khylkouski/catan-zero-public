# f70 Search-Reliability Arms

Makes the SEARCH side of the Gate-A fix uncertainty-aware. Context: full-64
search lost to its own raw policy because the mctx min-max completed-Q rescale
(`GumbelChanceMCTS._rescale_completed_q`) stretches whatever raw-Q spread it
observes to fill `[0, 1]`, so 1-2-sample sampling noise at 54-wide placement
roots is treated identically to real signal and manufactures false confidence
that swamps a near-flat prior. All arms are flag-gated in
`GumbelChanceMCTSConfig`, default OFF, and are an exact no-op when off.

## D1 — noise-floor rescale attenuation

Config: `rescale_noise_floor_c: float = 0.0` (disabled), `sigma_eval: float = 0.79`.

After the min-max rescale, each node's rescaled completed-Q is blended toward
the neutral `0.5`:

```
alpha = raw_spread / (raw_spread + c * sigma_eval / sqrt(mean_visits))
out   = 0.5 + alpha * (rescaled - 0.5)
```

`raw_spread = max(raw_q) - min(raw_q)` over the node's candidates;
`mean_visits` is the mean visit count across them. The denominator's second
term is the expected sampling noise of a per-candidate Q estimate
(`sigma_eval / sqrt(mean_visits)`). When the observed spread is at/below that
floor the rescaled values collapse toward neutral (prior order preserved);
when it dwarfs the floor `alpha -> 1` and the rescale is untouched.

- exact tie (`raw_spread == 0`) -> every value `0.5`
- `mean_visits -> inf` -> converges exactly to the current rescale
- `c == 0` -> short-circuits before any division (exact no-op)

## D2 — variance-aware completed-Q

Config: `variance_aware_q: bool = False`, `variance_aware_k: float = 1.0`.

`_GAction` accumulates `value_sq_sum` (a single multiply-add per backup,
alongside `value_sum`). In `_completed_q`, each visited candidate's Q is shrunk
toward the mixed value `v_mix` by its standard error:

```
q' = v_mix + shrink_a * (q_a - v_mix)
shrink_a = signal_var / (signal_var + k * SE_a^2)
SE_a^2 = Var[a's per-visit backups] / visits_a
signal_var = between-candidate variance of the raw visited Q's
```

A precise estimate (`SE_a -> 0`) keeps its Q; a noisy one (few visits / high
per-visit variance) collapses to the prior-weighted `v_mix`. James-Stein /
empirical-Bayes shrinkage of the completed-Q operator.

**Deviation from arXiv 2512.21648 (Inverse-RPO / UCT-V-P).** That work injects
the same per-action empirical standard deviation into a UCB-V *exploration
bonus* for a UCB-style selector, storing `(n, mu, sigma^2)` via Welford. Our
search is Gumbel-Top-k + Sequential Halving over completed-Q with a min-max
rescale — there is no UCB selection term to add sigma-hat to. We reuse the
paper's core signal (per-action value variance, tracked the same way via a
running sum of squares) but apply it where our operator is actually
vulnerable: gating how far each completed-Q may depart from the mixed value
before the noise-amplifying rescale.

## D3 — 200-root frozen opening panel (`tools/opening_panel.py`)

Standing pre-H2H diagnostic. `build` persists reconstruction seeds (fresh
base-seed block 600001) for 200 near-full-width placement roots to
`runs/panels/opening_200.json` (regenerable; `runs/` is gitignored). `eval`
scores any `(checkpoint, search-config)` pair — all search knobs incl. the
D1/D2 flags are exposed — reporting per root and in aggregate: harmful-flip
proxy (search argmax vs prior argmax), raw-Q spread vs eval noise floor, and
action-ranking quality vs a deeper-eval oracle over the top-K prior candidates
(Kendall tau-b, top-1 regret, top-3 coverage).

```
python tools/opening_panel.py build --out runs/panels/opening_200.json
python tools/opening_panel.py eval --panel runs/panels/opening_200.json \
    --checkpoint <ckpt> --device cuda:0 --oracle deep_search \
    --oracle-sims 64 --top-k 8 --out runs/panels/readout.json
```

**Cost.** The `deep_search` oracle (apply each top-K candidate, run an
`oracle-sims` search from the afterstate) is the entire cost driver:
~47 s/root under CPU contention at `oracle-sims=32/top-k=4`, ~16x at 256/8.
The shallow flip-rate / spread metrics are cheap. Run the full 200 as a
background job, subsample with `--max-roots`, or cap `oracle-sims`. Batching
the oracle's per-sim evaluator calls is the obvious follow-up (approved as
later work).

## D4 — phase-sliced value calibration (`tools/phase_sliced_value_calibration.py`)

Forward-passes the value head over any self-play shard dir
(naturally-terminated rows) and reports `corr(q, z)`, Brier, and
`value_rmse` globally and sliced by phase, by the `is_forced` flag, and by
legal-action-count bucket. `phase` is coarse (ROLL/dev/build/trade all ==
`PLAY_TURN`), so dev-vs-build is not separable without decoding
`action_taken`.

### Readout (gen1_pilot, 37,594 rows): value-repair-v2 vs old baseline

| slice | OLD corr | NEW corr | NEW Brier |
|---|---|---|---|
| global | 0.420 | 0.546 | 0.176 |
| opening_placement | -0.064 | **-0.009** | 0.260 |
| robber | 0.408 | 0.521 | 0.182 |
| discard | 0.536 | 0.630 | 0.142 |
| play_turn | 0.424 | 0.551 | 0.175 |
| legal-bucket 31-53 | -0.121 | **-0.068** | 0.266 |
| legal-bucket 54 | -0.060 | **+0.074** | 0.253 |

Value-repair-v2 improved global and most-phase calibration substantially, but
value calibration at **opening placement / the widest legal-count buckets
stays ~0 in both checkpoints** — precisely where search lost to raw policy.
The repair did not close the placement gap, so the D1/D2 arms remain necessary
there. Only a strength H2H binds ship (standing rule).

## sigma_eval calibration (wiring note)

`sigma_eval=0.79` is a placeholder. When D3/D4 are merged, set `sigma_eval`
per checkpoint from the D4 tool's value residual RMSE (`value_rmse`),
specifically the **`opening_placement` slice** (that is the phase whose noise
floor D1 governs at wide roots). `value_rmse` is an upper bound on the pure
estimator noise (it also absorbs the irreducible outcome variance given a
state), but it is the standard directly-usable proxy. A future step can wire
`opening_panel.py eval` to read `sigma_eval` from a D4 report path instead of
the CLI default.
