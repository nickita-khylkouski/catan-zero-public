# Coherent-target identity R&D corpus

The legacy 196k composite cannot be root-reanalyzed as one corpus. Its policy
targets are all `public_conservation_pimc_v1`, and the archived-opponent 20%
retains only the producer seat: opponent decisions are absent from the action
trace. A game seed cannot reconstruct actions chosen by a different network.

This transaction therefore generates one small causal corpus instead of
quietly relabeling old targets. All 8,192 games are producer mirror self-play,
use coherent public-belief single-tree n128, retain both seats' complete action
trace, and preserve completed-Q/visit evidence. Adaptive n256 and opponent
mixing are forbidden so target identity is the only intervention.

The contract is non-promotable R&D. Its rows become eligible for policy
distillation only after the declared post-wave acceptance checks pass; legacy
PIMC rows may not be mixed back into the policy objective.

## Exact B200 launch

Deploy the commit containing this directory to
`/home/ubuntu/catan-zero-v1` on `149.118.65.110`, then run:

```bash
ssh ubuntu@149.118.65.110 '
  set -euo pipefail
  cd /home/ubuntu/catan-zero-v1
  /home/ubuntu/catan-zero-v1/.venv/bin/python \
    tools/a1_target_eligibility_inventory.py \
    --rd-contract configs/operations/a1-target-identity-coherent-n128-rd-v1/contract.json \
    --out /tmp/a1-coherent-target-rd.contract-inventory.json >/dev/null
  /home/ubuntu/catan-zero-v1/.venv/bin/python \
    tools/fleet/a1_coherent_target_rd_executor.py \
    --contract configs/operations/a1-target-identity-coherent-n128-rd-v1/contract.json \
    --repo /home/ubuntu/catan-zero-v1 \
    --python /home/ubuntu/catan-zero-v1/.venv/bin/python \
    --host-address 149.118.65.110 \
    --go
'
```

The executor starts the already-installed `nvidia-mps.service` if needed,
atomically claims `[570000000000, 570000008192)` as eight disjoint 1,024-game
lanes, pins one 16-worker generator to each B200, and writes the immutable
launch receipt at:

`/home/ubuntu/experimental_nonpromotable/coherent-target-rd-n128-8192-20260715-r1/launch.receipt.json`

Omit `--go` for a read-only render. The executor refuses a pre-existing output
root, a checkpoint hash mismatch, a partial seed claim, an overlapping seed
range, or any contract/config/guard digest drift.
