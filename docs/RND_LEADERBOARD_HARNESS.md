# R&D experiment and leaderboard harness

`tools/rnd_leaderboard.py` is a fail-closed aggregation boundary for the five
architecture/search/training experiments. Search decisions run through
`src/catan_zero/search/operator_runner.py`. The complete two-player paired game
boundary lives in `tools/rnd_paired_operator_runner.py`: it applies selected
actions to authoritative live games, repeats each seed with exact seat swaps,
and emits the documented result bundle only after every game terminates.

The separation is intentional. It prevents a nominal `n=16` Gumbel run, a
16-leaf PUCT run, and a 16-millisecond raw-policy run from being described as
equal compute merely because their CLI flags happen to share a number.

## Required evidence

Every arm records:

- architecture ID, actual instantiated parameter count, architecture-config
  SHA-256, checkpoint path and SHA-256;
- search ID and search-config SHA-256;
- a frozen reference ID with its architecture, search, parameter count,
  checkpoint, and config hashes; the reference must match exactly across arms;
- exact Git commit and, for a dirty run, a non-zero patch SHA-256;
- immutable paired-seed manifest path, locally remeasured SHA-256, schema,
  track, count, and an exact match between its ordered seed array and games;
- immutable training-data/recipe manifest path, locally remeasured SHA-256,
  and schema;
- the loaded native-engine version, binary path, and locally remeasured SHA-256;
- device, accelerator model/UUID/memory/capability, and hashed host identity;
- paired seed, exact 0/1 candidate/reference seats, terminal winner, completion,
  and candidate score re-derived from winner plus seat;
- candidate and frozen-reference information regimes for every game;
- nominal visits, scheduled visits, logical neural leaves, orientation rows,
  evaluator method calls, and measured wall time for every game.

All six counters are mandatory, including for raw-policy controls. A genuine
zero is recorded as zero. A missing counter is not inferred and invalidates the
bundle. Truncated games are also rejected rather than silently treated as
losses.
Equal-time arms must also match device type, accelerator model, memory, and
compute capability. A campaign may set `required_accelerator_model` to reject
results from the wrong GPU class.

Each pair contains exactly two games with the same seed and exact seat swaps.
All arms must cover the same pair IDs and seeds. The campaign's
`required_arm_ids` must match the submitted arm set exactly.

An arm may declare `comparison_role: control`. It must still satisfy all
pairing, reference, provenance, and counter checks, but it is shown separately
and excluded from compute-matching and rank order. This is the correct role for
a raw-policy lower-bound: padding it with useless evaluations would create the
appearance of equal work without changing its decision rule.

## Budget regimes

Run two separate campaigns over the same disjoint paired-seed ledger:

1. `equal_work`: by default, pair totals must have identical logical leaves and
   orientation rows across arms. Evaluator calls are still recorded because
   batching efficiency is an outcome, not a fair-work unit.
2. `equal_time`: pair wall times must match within the explicitly declared
   absolute/relative tolerance. Logical leaves, orientation rows, and evaluator
   calls show what each method purchased in that time.

The contract is checked per pair rather than only in aggregate. An arm cannot
overspend early games and compensate with underspending later games. A mismatch
causes a non-zero exit and no leaderboard files are written.

## Implementation status is binding

Campaign templates explicitly separate:

- `source_status`: whether the actual architecture/search implementation exists;
- `measurement_adapter_status`: whether a runner emits this complete contract;
- `runnable`: true only when both statuses equal `implemented`.

E1/E2 source and a fake-engine end-to-end bundle path exist, but their templates
remain `runnable: false` with `measurement_adapter_status` set to
`provisional_fake_e2e`. Promote one arm to runnable only after the actual native
wheel, checkpoint evaluator, public-information search, live chance resolver,
and complete result-bundle validator pass together. Recurrent-deliberation,
MoE, and direct-RL arms remain separate hypotheses.

The exact prototypes and hypotheses are cataloged in
`configs/rnd/five_experiment_catalog.json`. Catalog membership is not an
implementation claim.

## Files and usage

- `configs/rnd/e1_search_operator.template.json`: legacy-modal Gumbel,
  exact-budget Gumbel, PUCT, regularized-policy MCTS, and raw-policy campaign.
- `configs/rnd/e2_architecture.template.json`: incumbent, RRT-384, and
  ResRGCN-384 campaign.
- `configs/rnd/result_bundle.template.json`: per-arm runner output.

Validate a campaign while wiring its arms:

```bash
python3 tools/rnd_leaderboard.py validate-config \
  --campaign configs/rnd/e1_search_operator.template.json
```

Run one candidate arm against its frozen reference. The seed manifest must use
`catan-zero-rnd-paired-seeds/v1`, declare track `2p_no_trade`, and contain a
unique non-empty `seeds` array. Public-information operation is the default;
`--allow-authoritative-hidden-state` is an explicit diagnostic opt-in.

```bash
python3 tools/rnd_paired_operator_runner.py \
  --campaign runs/rnd/e1-public-search.json \
  --run-id e1-gumbel-exact-seed1 \
  --arm-id incumbent-gumbel-exact-budget \
  --budget-regime equal_work \
  --seed-manifest runs/rnd/paired-seeds.json \
  --training-manifest runs/rnd/training-manifest.json \
  --candidate-kind gumbel \
  --candidate-architecture-id entity-graph-net-35m \
  --candidate-search-id gumbel-exact-budget \
  --candidate-checkpoint checkpoints/candidate.pt \
  --candidate-search-config runs/rnd/gumbel-exact.json \
  --reference-id frozen-reference-v1 \
  --reference-kind gumbel \
  --reference-architecture-id entity-graph-net-35m \
  --reference-search-id gumbel-reference \
  --reference-checkpoint checkpoints/reference.pt \
  --reference-search-config runs/rnd/reference-search.json \
  --device cuda:0 \
  --out runs/rnd/e1/gumbel-exact.json
```

Aggregate one budget regime after all required adapters are implemented:

```bash
python3 tools/rnd_leaderboard.py aggregate \
  --campaign runs/rnd/e1-equal-work.json \
  --result runs/rnd/e1/gumbel-legacy.json \
  --result runs/rnd/e1/gumbel-exact.json \
  --result runs/rnd/e1/puct.json \
  --result runs/rnd/e1/regularized.json \
  --result runs/rnd/e1/raw.json \
  --verify-local-checkpoints \
  --out-json runs/rnd/e1/leaderboard.json \
  --out-md runs/rnd/e1/leaderboard.md
```

The JSON is the machine record. Markdown is a concise rendering of that same
validated object. It labels the ordering descriptive because promotion and
strength claims require a separately predeclared statistical gate.

## Runner integration checklist

For each architecture/search adapter:

1. Reset counters at game start and read them after the terminal state.
2. Count scheduled tree visits separately from logical neural leaves.
3. Count every orientation passed to the model, including symmetry averaging.
4. Count evaluator method calls separately from rows so batching is visible.
5. Measure wall time around the whole policy/search workload, using one clock
   boundary for every arm.
6. Hash resolved configs and checkpoints after staging, not before copying.
7. Emit two games per seed with candidate/reference seats swapped.
8. Add an adapter test with non-equal synthetic counters to prove the harness
   rejects an unfair comparison.

`MeasuredSearchOperator.run(..., require_public_information=True)` fails closed
for any authoritative-state operator. Public determinized search is recorded as
`public_conservation_pimc`; a masked raw policy is recorded separately as
`public_observation_policy`. The campaign records `public_only`, every game
retains both exact role regimes, and aggregation rejects an authoritative role
in a public-only campaign. The complete runner disables evaluator caching so
orientation rows remain actual model rows rather than cache-hit requests.

Do not repurpose `nominal_visits` as a measured-work counter. It exists to make
nominal/scheduled/logical inflation visible in the report.
