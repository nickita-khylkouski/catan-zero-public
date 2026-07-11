# Experimental 80/20 corpus finalization

This procedure is intentionally outside the production generation executor.
It combines the reconstructed current-producer 80% with the opponent-only
recovery 20%, while keeping every output `experimental_nonpromotable`.

Run on the 8×B200 host so H100 data moves directly over the provider network.
The current and recovery roots and the output root must share a filesystem;
the finalizer uses hardlinks to create an audited relocation without copying
the roughly 900 GB current tranche again.

For each arm:

```bash
python3 tools/a1_experimental_corpus_finalizer.py plan \
  --reconstruction "$CURRENT_ROOT/experimental_nonpromotable.reconstruction_manifest.json" \
  --recovery-plan "$RECOVERY_PLAN" \
  --arm n128 \
  --out "$FINAL_ROOT/n128.plan.json"

python3 tools/a1_experimental_corpus_finalizer.py harvest \
  --plan "$FINAL_ROOT/n128.plan.json" \
  --destination "$FINAL_ROOT/n128.recovery-harvest" \
  --ssh-command "$HOME/a1_h100_ssh"

python3 tools/a1_experimental_corpus_finalizer.py finalize \
  --plan "$FINAL_ROOT/n128.plan.json" \
  --current-root "$CURRENT_RAW_ROOT" \
  --recovery-root "$FINAL_ROOT/n128.recovery-harvest" \
  --out "$FINAL_ROOT/n128"
```

Repeat with `n256`. The final receipt contains an exact
`build_memmap_argv`. Finalization selects the lowest complete seeds per job,
enforces the 112k/21k/7k or 44.8k/8.4k/2.8k category quotas, rejects duplicate
seeds and non-public-information rows, hashes every shard, and writes an exact
game-level validation lock. It never promotes, changes the champion, or alters
the canonical production campaign.
