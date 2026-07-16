# Production flag retirement

New training, self-play, and evaluation runs are config-first. Science is no
longer assembled from dozens of command-line switches.

## Canonical command surfaces

- `tools/generate.py`: 9 options
- `tools/evaluate.py`: 9 options
- `tools/train.py`: config-first learner entrypoint

The older executors remain importable for authenticated historical replay and
specialized R&D tools. They are not production launch interfaces.

## Retired generation experiments

The production generation entrypoint rejects configs that revive these paths:

- legacy PIMC `information_set_search`;
- `belief_chance_spectra`;
- `aggregate_q_then_improve`;
- adaptive-wide n256 and wide-root forced-full overrides;
- raw-policy-above-width;
- fixed-sigma/D1 noise-floor rescaling;
- categorical value readout;
- exact-budget Sequential Halving;
- root-wave batching;
- the superseded binary opponent-pool interface.

The deprecated temperature-fraction override, guard bypass, heuristic scoring,
fleet pipeline metadata, transport tuning, and config-registry bookkeeping are
also absent from the canonical CLI. Current science and implementation choices
live in `configs/generation/coherent_public_n128.schema18.json`.

`configs/RECOMMENDED_FLAGS.md` was removed because it recommended uncommissioned
D1 and uncertainty arms through copy-pasted CLI switches. Its suggested values
contradicted the adopted science contract.

## Retired evaluation experiments

The production candidate-versus-champion entrypoint fixes both roles to the
same coherent-public n128 operator. It rejects:

- role-specific search budgets and calibration;
- legacy PIMC and chance-spectrum modes;
- adaptive-wide and raw-policy decomposition arms;
- categorical/clip value readouts;
- D1/fixed-sigma calibration;
- exact-budget and root-wave variants;
- uncertainty-weighted and variance-aware backups.

Those experiments remain available only through the historical H2H executor.
The checked-in production evaluator recipe is
`configs/eval/coherent_public_n128.schema18.json`.

```bash
python tools/evaluate.py \
  --config configs/eval/coherent_public_n128.schema18.json \
  --candidate candidate.pt --champion champion.pt \
  --out evaluation.json --pairs 400 --workers 32 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --base-seed 2026071600
```

## Fixed config-order defect

`gumbel_search_cross_net_h2h.py` previously validated parser defaults before
loading `--config`. A config could enable the native or coherent operator after
the corresponding capability/information-set checks had already been skipped.
The evaluator now applies and validates config before deriving or checking any
search behavior.

## Historical replay

No archived config, receipt, command, or executor was deleted. Historical
commands continue to use `generate_gumbel_selfplay_data.py` and
`gumbel_search_cross_net_h2h.py` directly. The slim entrypoints deliberately
accept only the current schema and current production semantics.
