# Entity-graph checkpoint layer-drift audit

`tools/audit_checkpoint_layer_drift.py` is a read-only diagnostic for locating
where an `entity_graph` learner changed relative to its initialization. It does
not evaluate strength and deliberately has no pass/fail threshold.

## Usage

```bash
python3 tools/audit_checkpoint_layer_drift.py \
  --baseline /path/to/initial.pt \
  --candidate /path/to/trained.pt \
  --top-tensors 25 \
  --output /path/to/layer_drift.json
```

Omit `--output` to print the report. Existing output files are refused unless
`--force` is explicit. Both checkpoint files are only opened for reading.

Before measuring drift, the tool requires exact equality of the entity-graph
configuration, action-mask/public-observation/static-feature contract, and the
complete model key/shape/dtype structure. A mismatch is an error rather than a
partially comparable report.

The JSON includes checkpoint paths, byte SHA-256 hashes, durable checkpoint
metadata, and metrics for:

- every `blocks.N` transformer/state block independently;
- all typed input encoders and learned type/CLS tokens;
- policy/action modules;
- scalar/categorical/uncertainty value modules;
- final-VP and Q heads;
- remaining shared modules such as the post-trunk state norm.

Each group reports parameter and tensor counts, update-energy share, relative
L2 change, and baseline/candidate cosine similarity. Tensor outliers are ranked
both by absolute delta energy and relative L2. Relative L2 is `null` when the
baseline norm is zero; cosine is `null` when either vector has zero norm. These
measurements are descriptive evidence for diagnosing forgetting. They are not
promotion criteria and should be interpreted alongside held-out losses and
playing evaluations.
