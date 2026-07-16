# Reanalysis q-head safety

`tools/reanalyze_lite.py` and `tools/reanalyze_banked_corpus.py` do not support
either root value column. They require an explicit q-value component.

`root_value` is the post-search backed-up root value.
`root_prior_value` is the pre-search evaluator baseline used by the same sealed
search operator. A direct stored-feature forward cannot reproduce that operator:
wide roots may use symmetry averaging, and information-set modes may aggregate
determinizations. Relabeling such a forward as either root field corrupts the
training or quality signal.

Only true search reruns may refresh these fields. The policy-target reanalyzer
updates both atomically but currently invalidates compact completed-Q/visit
evidence, so its output intentionally cannot pass the empirical policy-quality
gate. Stage-C v3 updates the paired root fields together with the full search
patch and is the gate-capable route. Stage-C v1/v2 artifacts remain readable as
diagnostics but cannot be exported into a new learner overlay.

Legacy overlays carrying
`catan_zero_root_value_materialization_v1` mislabeled a direct forward as a
post-search backup. `train_bc` now rejects every `--value-target-lambda < 1`,
including metadata-free overlays, until authenticated target authority binds
the producer, operator, payload, and exact eligible-row mask. A true search
rerun is necessary but is not by itself sufficient for learner admission.

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
