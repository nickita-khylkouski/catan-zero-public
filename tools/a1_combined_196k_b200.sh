#!/usr/bin/env bash
# One-off experimental curriculum: sealed n256 dose followed by all n128 data.
# This script is prepared but must only be started after n128.training_input.ready.
set -euo pipefail

root=${A1_COMBINED_ROOT:-/home/ubuntu/experimental_nonpromotable/a1-combined-80-20-20260711}
repo=${A1_REPO:-/home/ubuntu/catan-rl-finalizer-2ccaa64}
python=${A1_PYTHON:-$repo/.venv/bin/python}
producer=${A1_PRODUCER:-/home/ubuntu/catan-zero-production/runs/learner/a1-infoset-n128-20260710-r2/candidate.pt}
arm_lock=${A1_N128_ARM_LOCK:-/home/ubuntu/catan-zero-production/contracts/a1-dual-arm-20260710-r1/locks/n128.lock.json}
parent_receipt=$root/training/n256-early/training.b307c09.receipt.json
data=$root/n128/n128.memmap
validation=$root/n128/n128.validation_seeds.json
ready=$root/n128.training_input.ready
out=$root/training/combined-196k
spec=$out/learner.spec.json
lock=$out/learner.lock.json
checkpoint=$out/candidate.pt
report=$out/report.json
receipt=$out/training.receipt.json

[[ -f "$ready" ]] || { echo "REFUSED: n128 input is not ready: $ready" >&2; exit 2; }
[[ -s "$parent_receipt" && -s "$data/corpus_meta.json" && -s "$validation" ]] || {
  echo "REFUSED: combined curriculum inputs are incomplete" >&2; exit 2;
}
mkdir -p "$out"

tmp_raw=$out/.learner.spec.$$.raw
tmp_spec=$out/.learner.spec.$$.tmp
"$python" "$repo/tools/a1_dual_learner_contract.py" inspect-spec \
  --data "$data" --validation "$validation" \
  --producer-checkpoint "$producer" --world-size 8 >"$tmp_raw"
# Old reviewed runtimes emitted zero or more progress JSON objects before the
# final spec on stdout.  Normalize that stream while new runtimes keep stdout
# machine-readable directly.
"$python" - "$tmp_raw" "$tmp_spec" <<'PY'
import json, pathlib, sys
source, target = map(pathlib.Path, sys.argv[1:])
text = source.read_text()
decoder = json.JSONDecoder()
values, offset = [], 0
while offset < len(text):
    while offset < len(text) and text[offset].isspace():
        offset += 1
    if offset == len(text):
        break
    value, offset = decoder.raw_decode(text, offset)
    values.append(value)
if not values or any(not isinstance(row, dict) or "progress" not in row for row in values[:-1]):
    raise SystemExit("REFUSED: unexpected inspect-spec stdout stream")
v = values[-1]
assert v["arm_id"] == "n128" and v["subset_id"] == "full-140k"
assert v["topology"]["world_size"] == 8 and v["topology"]["global_batch_size"] == 4096
target.write_text(json.dumps(v, indent=2, sort_keys=True) + "\n")
PY
rm -f "$tmp_raw"
if [[ -e "$spec" ]]; then
  cmp -s "$tmp_spec" "$spec" || { echo "REFUSED: learner spec drift" >&2; exit 2; }
  rm -f "$tmp_spec"
else
  chmod 0444 "$tmp_spec"
  mv "$tmp_spec" "$spec"
fi

[[ ! -e "$lock" ]] || { echo "REFUSED: learner lock output already exists" >&2; exit 2; }
"$python" "$repo/tools/a1_dual_learner_contract.py" seal \
  --arm-lock "$arm_lock" --learner-spec "$spec" --data "$data" \
  --validation "$validation" --producer-checkpoint "$producer" --out "$lock"
lock_sha=$(sha256sum "$lock" | awk '{print "sha256:"$1}')

command=(
  "$python" "$repo/tools/a1_dual_arm_train.py"
  --data "$data" --learner-lock "$lock"
  --reviewed-lock-file-sha256 "$lock_sha"
  --validation-manifest "$validation" --producer-checkpoint "$producer"
  --curriculum-parent-receipt "$parent_receipt"
  --checkpoint "$checkpoint" --report "$report" --receipt "$receipt"
  --python "$python"
)

# The dry run replays both arm authority and the completed n256 parent before
# this script takes GPU/MPS ownership.
"${command[@]}" >"$out/dry-run.json"
[[ $(systemctl is-active nvidia-mps.service) == active ]] || {
  echo "REFUSED: MPS must be active at ownership handoff" >&2; exit 2;
}
restore_mps() { sudo -n systemctl start nvidia-mps.service || true; }
trap restore_mps EXIT
sudo -n systemctl stop nvidia-mps.service
"${command[@]}" --go >"$out/go.json" 2>"$out/go.stderr.log"
test -s "$receipt"
