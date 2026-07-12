# A1 checkpoint × operator crossover

`tools/a1_checkpoint_operator_crossover.py` answers one narrow causal question:
did the neural checkpoint improve, did the `c_scale` search operator improve,
or does their interaction explain a role-native H2H result?

The diagnostic seals four paired, color-swapped BASE-map panels on the exact
same seeds:

| Panel | Candidate role | Baseline role | Estimand |
|---|---|---|---|
| `checkpoint_at_cscale_003` | candidate, `.03` | f7, `.03` | checkpoint effect at `.03` |
| `checkpoint_at_cscale_010` | candidate, `.10` | f7, `.10` | checkpoint effect at `.10` |
| `operator_on_candidate` | candidate, `.10` | candidate, `.03` | operator effect on candidate |
| `operator_on_f7` | f7, `.10` | f7, `.03` | operator effect on f7 |

All four panels use the current n128 information-set/D6/Rust evaluator recipe.
The plan binds checkpoint SHA-256s, repo/tool identities, the BASE map, the
common seed cohort, both role operators, and every command. The collector
rejects reports whose checkpoint, operator, map, seed, or pair identity differs.

The crossover is always `diagnostic_only=true` and
`promotion_eligible=false`. Candidate-native `.10` versus f7-native `.03`
changes two variables simultaneously, so it is recorded but not emitted as a
crossover job. Plan that separate binding panel with
`tools/fleet/a1_h100_eval_fleet.py --comparison-mode promotion_parent`.

Example (planning only):

```bash
python3 tools/a1_checkpoint_operator_crossover.py plan \
  --candidate /abs/candidate.pt --f7 /abs/f7.pt \
  --pairs 200 --base-seed 6195000000 \
  --output-dir /abs/crossover/reports \
  --out /abs/crossover/plan.json
```

Run the four emitted `argv` arrays through the normal evaluator executor, then
collect only after all four reports are complete:

```bash
python3 tools/a1_checkpoint_operator_crossover.py collect \
  --plan /abs/crossover/plan.json \
  --out /abs/crossover/summary.json
```

Interpretation is intentionally descriptive. A checkpoint win at both matched
operators is robust neural evidence. An operator win on both frozen checkpoints
is operator evidence. Different signs across checkpoints indicate interaction.
None of those outcomes promotes a model; the separate role-native binding panel
does that.
