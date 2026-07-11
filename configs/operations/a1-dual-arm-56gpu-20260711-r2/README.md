# A1 dual-arm 56-GPU generation contract, revision 2

This directory is the checked-in output of the fail-closed generation-revision
machinery. It supersedes the irrecoverable r1 execution attempt without
reusing any output, receipt namespace, or seed claim. It is a blueprint only:
`contract.json` remains
`blocked_pending_post_promotion_handoff`, and neither sealing nor launch is
authorized by the checked-in file.

The science and allocation remain the reviewed dual-arm comparison:

- `n256`: 28 GPU lanes, 2,000 selected games per lane;
- `n128`: 28 GPU lanes, 5,000 selected games per lane;
- 80/15/5 current-producer, recent-history, and hard-negative selection;
- 16 MPS workers per GPU, public information-set search, D6 threshold 20;
- native MCTS hot-loop and Rust featurization enabled;
- no adaptive/wide search overrides.

## Fresh identity boundary

r1 consumed its claims even though its parent manifests failed. r2 therefore
uses only the next unconsumed interval:

- n256: `[300000626944, 300000856320)`;
- n128: `[300000856320, 300001085696)`;
- every later campaign starts at or after `300001085696`.

Output roots are under `a1-dual-arm-20260711-r2`. The deterministic revision
retains the reviewed logical-to-physical placement, so isolation comes from
fresh output, local receipt, and remote executor roots. Never point r2 at an r1
receipt.

The private per-arm host manifests must use a campaign-specific executor root,
for example:

```json
{"remote_root": "/home/ubuntu/a1-production-dual-arm-20260711-r2"}
```

Using the old remote root is forbidden: its O_EXCL job receipts, lane locks,
quarantine records, and logs belong to r1. Host endpoints and the physical
placement may remain the same; `placement.assignments.json` restates those
assignments and must be sealed against the r2 campaign digest.

## Lineage gate

The producer checkpoint bytes are known and recorded, but checkpoint bytes are
not promotion authority. Materialization requires a newly issued immutable
`a1-post-promotion-producer-handoff-v1` from the forthcoming lineage
re-attestation. No handoff, registry receipt, or pointer state is fabricated in
this directory.

Read-only verification is available now:

```bash
python3 tools/a1_pre_wave_contract.py verify-generation-campaign \
  --contract configs/operations/a1-dual-arm-56gpu-20260711-r2/contract.json
```

`--require-ready` must refuse until the lineage handoff exists. After that
external artifact has been issued, use fresh paths throughout:

```bash
CAMPAIGN=configs/operations/a1-dual-arm-56gpu-20260711-r2/contract.json
ASSIGNMENTS=configs/operations/a1-dual-arm-56gpu-20260711-r2/placement.assignments.json
HANDOFF=/home/ubuntu/catan-zero-production/a1-post-promotion-handoff-r2.json
PLACEMENT=/home/ubuntu/catan-zero-production/a1-dual-arm-placement-r2.json
RUN=/home/ubuntu/catan-zero-production/contracts/a1-dual-arm-20260711-r2

python3 tools/a1_pre_wave_contract.py seal-generation-placement \
  --contract "$CAMPAIGN" --assignments "$ASSIGNMENTS" --out "$PLACEMENT"
python3 tools/a1_pre_wave_contract.py materialize-generation-campaign \
  --contract "$CAMPAIGN" --promotion-handoff "$HANDOFF" \
  --placement "$PLACEMENT" --out-dir "$RUN/locks"
```

Then render and claim each arm into new files. Run the production executor as
a dry run first with private host manifests whose `remote_root` is the r2 root;
only an operator may repeat it with `--go`. No checked-in command launches or
claims work.
