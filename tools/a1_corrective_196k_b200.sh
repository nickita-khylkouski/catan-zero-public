#!/usr/bin/env bash
# Diagnostic-only corrective all-196k curriculum.  This is intentionally inert
# unless the canonical combined candidate has completed and failed its fixed
# internal/external handoff gates, and an operator supplies --go.
set -euo pipefail

mode=${1:-}
[[ -z "$mode" || "$mode" == --go ]] || { echo "usage: $0 [--go]" >&2; exit 2; }

root=${A1_COMBINED_ROOT:-/home/ubuntu/experimental_nonpromotable/a1-combined-80-20-20260711}
script_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
repo=${A1_REPO:-$script_repo}
handoff_repo=${A1_HANDOFF_REPO:-$repo}
python=${A1_PYTHON:-$repo/.venv/bin/python}
producer=${A1_PRODUCER:-/home/ubuntu/catan-zero-production/runs/learner/a1-infoset-n128-20260710-r2/candidate.pt}
contracts=${A1_CONTRACTS:-/home/ubuntu/catan-zero-production/contracts/a1-dual-arm-20260710-r1/locks}
handoff=${A1_COMBINED_HANDOFF:-$root/evaluation/combined-196k-0f53e97/handoff.result.json}
out=$root/training/corrective-196k-lr120u-loser1
overrides='{"loser_sample_weight":1.0,"lr":0.00012}'

if [[ "$mode" == --go ]]; then
  [[ -s "$handoff" ]] || { echo "REFUSED: canonical combined handoff is not complete" >&2; exit 2; }
  "$python" - "$handoff_repo" "$handoff" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from tools import a1_combined_candidate_handoff as combined
result=combined.verify_result(Path(sys.argv[2]))
if result["passed"]:
    raise SystemExit("REFUSED: combined candidate passed; corrective run is unauthorized")
if result["decision"] != "reject_candidate":
    raise SystemExit("REFUSED: canonical combined handoff has no rejection decision")
print("canonical combined handoff rejects candidate; corrective diagnostic is eligible")
PY
  [[ -s "$root/n128.training_input.ready" ]] || { echo "REFUSED: n128 input is not ready" >&2; exit 2; }
fi
mkdir -p "$out/n256" "$out/n128"

lock_sha() { sha256sum "$1" | awk '{print "sha256:"$1}'; }
code_sha() {
  local lock=$1
  "$python" - "$repo" "$lock" "$(lock_sha "$lock")" <<'PY'
import sys
from pathlib import Path
repo, lock, digest = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0,str(repo))
from tools import a1_dual_learner_contract as contract
from tools import a1_one_dose_train as one
authority=contract.verify_lock(lock,reviewed_file_sha256=digest)
shape={"provenance":{
  "learner_code":[{"path":str((repo/"tools/train_bc.py").resolve())}],
  "runtime_code_tree":[{"path":str((repo/r["path"]).resolve())} for r in authority["runtime"]],
}}
print(one._current_ablation_code_binding(shape)["code_tree_sha256"])
PY
}

run_dose() {
  local arm=$1 subset=$2 data=$3 validation=$4 parent=${5:-}
  local dose=$out/$arm spec=$out/$arm/learner.spec.json
  local lock=$out/$arm/learner.lock.json lock_digest code_digest tmp
  tmp=$spec.tmp.$$
  "$python" "$repo/tools/a1_dual_learner_contract.py" inspect-spec \
    --data "$data" --validation "$validation" \
    --producer-checkpoint "$producer" --world-size 8 >"$tmp"
  if [[ -e "$spec" ]]; then
    cmp -s "$tmp" "$spec" || { echo "REFUSED: $arm corrective spec drift" >&2; exit 2; }
    rm -f "$tmp"
  else
    chmod 0444 "$tmp"; mv "$tmp" "$spec"
  fi
  if [[ ! -e "$lock" ]]; then
    "$python" "$repo/tools/a1_dual_learner_contract.py" seal \
      --arm-lock "$contracts/$arm.lock.json" --learner-spec "$spec" \
      --data "$data" --validation "$validation" \
      --producer-checkpoint "$producer" --out "$lock"
  fi
  lock_digest=$(lock_sha "$lock")
  code_digest=$(code_sha "$lock")
  local command=(
    "$python" "$repo/tools/a1_dual_arm_train.py"
    --data "$data" --learner-lock "$lock"
    --reviewed-lock-file-sha256 "$lock_digest"
    --validation-manifest "$validation" --producer-checkpoint "$producer"
    --ablation-id all-196k-corrective-lr120u-loser1
    --recipe-overrides-json "$overrides"
    --ablation-code-tree-sha256 "$code_digest"
    --checkpoint "$dose/candidate.pt" --report "$dose/report.json"
    --receipt "$dose/training.receipt.json" --python "$python"
  )
  [[ -z "$parent" ]] || command+=(--curriculum-parent-receipt "$parent")
  "${command[@]}" >"$dose/dry-run.json"
  "$python" - "$dose/dry-run.json" "$arm" "$subset" <<'PY'
import json, math, sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
assert (p["arm_id"],p["subset_id"])==(sys.argv[2],sys.argv[3])
assert p["world_size"]==8 and p["global_batch_size"]==4096
assert p["inputs"]["learner_ablation"]["effective_recipe"]["lr"]==0.00012
assert p["inputs"]["learner_ablation"]["effective_recipe"]["loser_sample_weight"]==1.0
meta=json.load(open(p["inputs"]["corpus_meta"]["path"],encoding="utf-8"))
rows=int(meta["a1_post_wave_audit"]["training_row_count"])
print(json.dumps({"arm":sys.argv[2],"training_rows":rows,"optimizer_steps":math.ceil(rows/4096)},sort_keys=True))
PY
  [[ "$mode" == --go ]] || return 0
  "${command[@]}" --go >"$dose/go.json" 2>"$dose/go.stderr.log"
  test -s "$dose/training.receipt.json"
}

if [[ "$mode" != --go ]]; then
  run_dose n256 full-56k "$root/n256-early/n256.memmap" \
    "$root/n256-early/n256.validation_seeds.json"
  cat <<EOF
prepared corrective recipe: lr=1.2e-4, loser_sample_weight=1.0, one epoch per dose,
world8/global-batch4096. The n128 dose is identically bound after the corrective
n256 parent receipt exists. Re-run with --go only after reviewing this dry run.
EOF
  exit 0
fi

[[ $(systemctl is-active nvidia-mps.service) == active ]] || {
  echo "REFUSED: MPS must be active at ownership handoff" >&2; exit 2;
}
restore_mps() { sudo -n systemctl start nvidia-mps.service || true; }
trap restore_mps EXIT
sudo -n systemctl stop nvidia-mps.service

run_dose n256 full-56k "$root/n256-early/n256.memmap" \
  "$root/n256-early/n256.validation_seeds.json"
run_dose n128 full-140k "$root/n128/n128.memmap" \
  "$root/n128/n128.validation_seeds.json" \
  "$out/n256/training.receipt.json"

"$python" - "$out/n128/report.json" "$out/final.telemetry.json" <<'PY'
import json, os, sys
r=json.load(open(sys.argv[1],encoding="utf-8")); v=r["metrics"][-1]["validation"]
value={"prior_kl_ratio":v["prior_kl_ratio"],"target_range":[0.6,0.8],
       "in_target_range":0.6 <= v["prior_kl_ratio"] <= 0.8,
       "diagnostic_only":True,"promotion_eligible":False}
fd=os.open(sys.argv[2],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,"w",encoding="utf-8") as f:
 json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
print(json.dumps(value,sort_keys=True))
PY
