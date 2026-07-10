# Reanalysis q-head safety

`tools/reanalyze_lite.py` and `tools/reanalyze_banked_corpus.py` default to
`--v-component root_value`. This is the controlled value-reanalysis probe: it
forwards the configured trained value readout and materializes the per-state
column consumed by `train_bc --value-target-lambda`. The default matches scalar
search (`--value-readout scalar --value-scale 1 --value-squash tanh`). A
categorical-search corpus must explicitly use `--value-readout categorical`.
Categorical expectations bypass scalar tanh exactly as search does; both paths
receive search's final `[-1,1]` clip.

Every `root_value` output records a
`catan_zero_root_value_materialization_v1` object in its reanalysis manifest and
in `corpus_meta.json` next to the column schema. It binds the source output key,
readout, scale, configured/applied squash, final clip, range, and root-to-move
semantics. Banked jobs treat it as immutable plan shape and validate it again at
run and merge. Legacy raw-forward jobs and mixed-provenance chunks fail closed and
must be replanned.

Do not use `target_scores` merely because that column already exists in an older
corpus. Those modes forward the model's per-action `q_values`. Normal `train_bc`
runs use `q_loss_weight=0` and freeze that branch, so its presence in a checkpoint
does not mean it was trained. Rewriting `target_scores` from it would silently
replace searched targets with untrained outputs.

## Explicit q-values mode

`target_scores` and `afterstate_target` are fail-closed. They require
`--q-head-provenance PATH`, where `PATH` is a JSON object of this form:

```json
{
  "schema": "catan_zero_q_head_provenance_v1",
  "checkpoint_md5": "0123456789abcdef0123456789abcdef",
  "q_head": {
    "trained": true,
    "target_semantics": "root_to_move_search_action_value_v1",
    "value_range": [-1, 1]
  },
  "validation": {
    "passed": true,
    "evidence": "runs/q-head-calibration-20260709/report.json"
  }
}
```

The checkpoint md5 must match the exact reanalyzer checkpoint. The target
semantics deliberately exclude the normalized teacher-preference q objective:
those scores are not return-scale search-action values. `validation.evidence`
must identify the calibration or held-out validation that established the q head
is usable; the tool records both the provenance contents and the provenance file's
SHA-256 in its output manifest.

Example, only after producing that evidence:

```bash
python tools/reanalyze_lite.py \
  --corpus runs/memmap_corpus_window \
  --checkpoint runs/q_trained_champion.pt \
  --v-component target_scores \
  --q-head-provenance runs/q_trained_champion.q-head.json \
  --device cuda
```

Banked jobs validate the record at planning, again before processing chunks, and
again before merge. Pre-hardening job manifests that selected a q-values component
without this record therefore cannot continue or be merged silently.
