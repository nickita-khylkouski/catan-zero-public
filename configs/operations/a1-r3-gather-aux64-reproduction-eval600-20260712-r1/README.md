# AUX64 reproduction: prepared 600-pair internal evaluation

`plan.request.json` is a reproducible request, not an executable evaluation
plan.  The candidate subsequently finalized, and the controller created and
launched the actual read-only sealed plan at:

```
/home/ubuntu/experimental_nonpromotable/r3-gather-aux64-reproduction-eval600-20260712-r1/plan.json
```

Its file SHA-256 is `30f88233d4e3df69d65d110c09ddae03b34b0cac877801ed894e16829be6f815`,
plan hash is `sha256:453449fbba198c79e7ceee6f64af644a349b739e852d432cd34672727b806653`,
and run ID is `a1-eval-8106b08c1e81af75`.  The launch atomically claimed
`[6198340000,6198340600)` for 600 internal pairs and reserved
`[6198350000,6198350020)` for the controller-required external phase.  It
launched exactly 15 pairs on each of the established 40 H100s; launch status
was 40 active with zero failed, missing, stale, or unsafe jobs.

The internal phase subsequently completed and collected all 600 pairs / 1,200
games with no failed, missing, stale, or unsafe jobs and zero truncations.  The
tensor-identical reproduction scored 591-609 games (49.25%), pentanomial LLR
`-2.263827184220417` from counts `(LL, split, WW) = (131, 347, 122)`,
verdict `continue`.  Its pooled report is:

```
/home/ubuntu/experimental_nonpromotable/r3-gather-aux64-reproduction-eval600-20260712-r1/collected/a1-eval-8106b08c1e81af75/pooled/internal.json
```

with SHA-256 `b264841eea1dc82ce53f1e93b2ddd742f8024a61b7d5382941c9e8175f156e26`.
This fresh result does not reproduce the original AUX64 cohort's 52.5% raw
600-pair aggregate.  Keep the cohorts separate: checkpoint container hashes
differ, and the pooling tool correctly refuses to erase that provenance even
though the authenticated model tensors are bit-identical.

The configured `elo0=-10`, `elo1=15` GSPRT is a regression-protection
indifference band.  Its `H1` decision must not be reported as proof that true
Elo is positive.  Replaying this fresh cohort with a superiority null
(`elo0=0`, `elo1=15`) gives LLR `-2.240792498749374`, still `continue`.
Replaying all 1,200 tensor-identical pairs together gives 50.875% and
superiority LLR `-0.5255305237692693`, also `continue`.  The earlier AUX64 H1
was therefore a permissible non-regression decision that was over-described
as a demonstrated strength gain.

`fleet48.manifest.json` is separately approved by the controller and resolves
exactly 48 GPUs by adding only c7/c8.  It was **not** used by this immutable
40-GPU plan and must not be substituted into it.  It is available only for a
future, separately sealed, disjoint-seed extension after the 48-topology code
is reviewed, committed, and deployed.

`fleet64.manifest.json` is the separately approved full topology.  It adds
the onboarded 8-GPU `h100-8c` and `h100-8d` hosts to that exact 48-GPU fleet.
The controller accepts only the canonical 40-, 48-, or 64-GPU
alias/address/GPU mappings; partial 56/60-GPU mixtures and address
substitutions fail closed.  This manifest also was **not** used by the sealed
40-GPU result above and cannot be substituted into its plan because the
manifest hash is part of the run identity.

The comparison reproduces the earlier AUX64 evaluation operator exactly:
paired same-seed/color-swapped BASE games; native Rust information-set search;
`n_full=128`; P4 with minimum 32 simulations per particle; D6 at width 20;
`c_scale=.1`; `c_visit=50`; `sigma_eval=.98`; scalar+tanh value readout; root
candidate caps 16/54 at width 24.  Both roles use the same operator.  The
baseline and causal learner parent are exact production r3 (`6817ab...`).

## Sealing replay

First replay the production completion finalizer.  Then require its checkpoint
reference to name `candidate.pt`, require its SHA-256 to match the candidate
bytes, and require the receipt to remain production-eligible with a successful
unit state.  The target-gather finalizer already authenticates the manifest,
submission, report, progress/RNG state, optimizer, exact training dose, and
adapter-only model delta.

From the pinned controller checkout on the B200:

```bash
set -euo pipefail
CTRL=/home/ubuntu/catan-eval-ab3618c
PY=/home/ubuntu/catan-zero-v1/.venv/bin/python
MAN=/home/ubuntu/a1-eval-40-ad9b894.manifest.json
ROOT=/home/ubuntu/catan-zero-production/runs/learner/a1-production-r3-target-gather-aux64-reproduction-20260712-r1
OUT=/home/ubuntu/experimental_nonpromotable/r3-gather-aux64-reproduction-eval600-20260712-r1
R3=/home/ubuntu/catan-zero-production/runs/learner/a1-production-l1-one-dose-20260712-r3/candidate.pt
REG=/home/ubuntu/catan-zero-production/private/champion_registry.json

test "$(git -C "$CTRL" rev-parse HEAD)" = ab3618c5f911b50d5b7e047750038f79478686fd
test "$(sha256sum "$R3" | cut -d' ' -f1)" = 6817ab054506f962a758ebf48addce5cc7eb801bf451cf2d02b62fb91f5da39c
test -s "$ROOT/completion.receipt.json"
test -s "$ROOT/candidate.pt"
"$PY" - "$ROOT/completion.receipt.json" "$ROOT/candidate.pt" <<'PY'
import hashlib
import json
import pathlib
import sys

receipt_path = pathlib.Path(sys.argv[1]).resolve(strict=True)
candidate = pathlib.Path(sys.argv[2]).resolve(strict=True)
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
unhashed = dict(receipt)
stated = unhashed.pop("receipt_sha256")
actual_receipt = "sha256:" + hashlib.sha256(
    json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
candidate_sha = "sha256:" + hashlib.sha256(candidate.read_bytes()).hexdigest()
assert stated == actual_receipt
assert receipt["schema_version"] == "a1-production-target-gather-retrain-completion-v1"
assert receipt["diagnostic_only"] is False
assert receipt["production_eligible"] is True
assert receipt["unit_state"] == {
    "ActiveState": "inactive",
    "Result": "success",
    "ExecMainStatus": "0",
}
assert pathlib.Path(receipt["checkpoint"]["path"]).resolve() == candidate
assert receipt["checkpoint"]["sha256"] == candidate_sha
assert receipt["manifest"] == {
    "path": "/home/ubuntu/catan-zero-production/private/manifests/a1-production-r3-target-gather-aux64-reproduction-20260712-r1.manifest.json",
    "sha256": "sha256:e3e6dc2225ca02e32fd5983bb0d5f709c860e059739e2992b81053b03a89cb6",
}
print(candidate_sha)
PY
mkdir -p "$OUT"

cd "$CTRL"
"$PY" tools/fleet/a1_h100_eval_fleet.py \
  --manifest "$MAN" plan \
  --candidate "$ROOT/candidate.pt" \
  --champion "$R3" \
  --candidate-parent "$R3" \
  --registry "$REG" \
  --comparison-mode historical_comparison \
  --historical-comparison-reason sealed_aux64_adapter_reproduction \
  --internal-pairs 600 --internal-base-seed 6198340000 \
  --external-pairs 20 --external-base-seed 6198350000 \
  --workers-per-gpu 16 \
  --iteration-id r3-gather-aux64-reproduction-eval600 \
  --seed-cohort-id r3-gather-aux64-reproduction-eval600 \
  --scope full \
  --candidate-c-scale 0.1 --champion-c-scale 0.1 \
  --candidate-value-squash tanh --champion-value-squash tanh \
  --out "$OUT/plan.json"
```

Before replaying any launch, load the resulting plan with the same controller and verify:
600 internal pairs, 40 internal jobs, the candidate hash from the completion
receipt, r3 hash `6817ab...`, science hash `7c3d8f...`, and fresh intervals
`[6198340000,6198340600)` / `[6198350000,6198350020)`.  The controller rechecks
the live ledger and claims both ranges atomically only when `launch --go` is
eventually authorized.

## Collection and pooling

After all 40 internal jobs are done:

```bash
"$PY" tools/fleet/a1_h100_eval_fleet.py \
  --manifest "$MAN" collect --plan "$OUT/plan.json" --phase internal \
  --output-dir "$OUT/collected"
```

`collect` authenticates and pools the 40 shards into
`$OUT/collected/<run_id>/pooled/internal.json`; that raw-union replay is the
authoritative 600-pair result.  Do not add terminal LLRs or pool reports from
different checkpoint hashes.

If and only if the reproduction candidate hash is exactly the original AUX64
hash (`4d687727...`), the new pooled report may additionally be raw-union pooled
with the original disjoint 200+400 cohorts using
`tools/a1_evaluation_pool.py internal --allow-disjoint-cohorts`.  Otherwise the
reproduction result remains a separate 600-pair experiment.
