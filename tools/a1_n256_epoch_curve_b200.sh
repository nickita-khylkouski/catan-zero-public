#!/usr/bin/env bash
# Clean producer-started 3-epoch n256 diagnostic selected by a completed LR point.
set -euo pipefail

usage() { echo "usage: $0 --anchor-receipt PATH [--go]" >&2; exit 2; }
anchor_receipt=
mode=dry-run
while (($#)); do
  case "$1" in
    --anchor-receipt)
      (($# >= 2)) || usage; [[ -z "$anchor_receipt" ]] || usage
      anchor_receipt=$2; shift 2 ;;
    --go) [[ "$mode" == dry-run ]] || usage; mode=go; shift ;;
    *) usage ;;
  esac
done
[[ -n "$anchor_receipt" ]] || usage

root=${A1_COMBINED_ROOT:-/home/ubuntu/experimental_nonpromotable/a1-combined-80-20-20260711}
script_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
repo=${A1_REPO:-$script_repo}
python=${A1_PYTHON:-$repo/.venv/bin/python}
producer=${A1_PRODUCER:-/home/ubuntu/catan-zero-production/runs/learner/a1-infoset-n128-20260710-r2/candidate.pt}
data=$root/n256-early/n256.memmap
validation=$root/n256-early/n256.validation_seeds.json
cd "$repo"
[[ -z $(git status --porcelain --untracked-files=no) ]] || {
  echo "REFUSED: epoch-curve runtime repository has tracked modifications" >&2; exit 2;
}

anchor_json=$("$python" - "$repo" "$anchor_receipt" "$producer" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tools import a1_dual_arm_train as train
from tools import a1_dual_learner_contract as contract
r = train.verify_receipt(Path(sys.argv[2]))
if (r.get("arm_id"), r.get("subset_id")) != ("n256", "full-56k"):
    raise SystemExit("REFUSED: curve anchor is not a complete n256 point")
a = r.get("inputs", {}).get("learner_ablation", {})
if a.get("diagnostic_only") is not True or a.get("promotion_eligible") is not False:
    raise SystemExit("REFUSED: curve anchor is not diagnostic-only")
effective = a.get("effective_recipe", {})
lr = effective.get("lr")
if lr not in {0.00006, 0.00012, 0.00024} or effective.get("loser_sample_weight") != 1.0:
    raise SystemExit("REFUSED: curve anchor is not a declared LR-response recipe")
if effective.get("epochs") != 1:
    raise SystemExit("REFUSED: curve anchor must be the one-epoch response point")
lock_ref = r.get("inputs", {}).get("learner_lock")
if not isinstance(lock_ref, dict):
    raise SystemExit("REFUSED: curve anchor has no learner lock")
authority = contract.verify_lock(
    Path(lock_ref["path"]), reviewed_file_sha256=lock_ref["sha256"]
)
if authority.get("topology", {}).get("world_size") != 8:
    raise SystemExit("REFUSED: curve anchor is not world8")
producer = r.get("inputs", {}).get("producer")
expected_producer = Path(sys.argv[3]).expanduser().resolve(strict=True)
if producer != train._file_ref(expected_producer, where="curve producer"):
    raise SystemExit("REFUSED: curve anchor producer binding drift")
print(json.dumps({
    "lr": lr, "lock_path": lock_ref["path"], "lock_sha256": lock_ref["sha256"],
    "producer": producer, "anchor_receipt": str(Path(sys.argv[2]).resolve(strict=True)),
    "anchor_receipt_sha256": train._sha256(Path(sys.argv[2]).resolve(strict=True)),
}, sort_keys=True))
PY
)
lr=$("$python" -c 'import json,sys; print(json.loads(sys.argv[1])["lr"])' "$anchor_json")
lock=$("$python" -c 'import json,sys; print(json.loads(sys.argv[1])["lock_path"])' "$anchor_json")
lock_digest=$("$python" -c 'import json,sys; print(json.loads(sys.argv[1])["lock_sha256"])' "$anchor_json")
case "$lr" in
  6e-05|0.00006) lr_label=lr60u ;;
  0.00012) lr_label=lr120u ;;
  0.00024) lr_label=lr240u ;;
  *) echo "REFUSED: normalized curve LR drift" >&2; exit 2 ;;
esac
out=$root/training/n256-epoch-curve-${lr_label}-clean3
dose=$out/n256
receipt=$dose/training.receipt.json
ablation_id=n256-epoch-curve-${lr_label}-clean3
overrides=$(printf '{"epochs":3,"loser_sample_weight":1.0,"lr":%s}' "$lr")
mkdir -p "$dose"

code_digest=$("$python" - "$repo" "$lock" "$lock_digest" <<'PY'
import sys
from pathlib import Path
repo, lock, digest = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(repo))
from tools import a1_dual_learner_contract as contract
from tools import a1_one_dose_train as one
authority = contract.verify_lock(lock, reviewed_file_sha256=digest)
shape = {"provenance": {
    "learner_code": [{"path": str((repo / "tools/train_bc.py").resolve())}],
    "runtime_code_tree": [
        {"path": str((repo / row["path"]).resolve())} for row in authority["runtime"]
    ],
}}
print(one._current_ablation_code_binding(shape)["code_tree_sha256"])
PY
)

command=(
  "$python" "$repo/tools/a1_dual_arm_train.py"
  --data "$data" --learner-lock "$lock"
  --reviewed-lock-file-sha256 "$lock_digest"
  --validation-manifest "$validation" --producer-checkpoint "$producer"
  --ablation-id "$ablation_id" --recipe-overrides-json "$overrides"
  --ablation-code-tree-sha256 "$code_digest"
  --checkpoint "$dose/candidate.pt" --report "$dose/report.json"
  --receipt "$receipt" --python "$python"
)

validate_plan() {
  "$python" - "$1" "$lr" "$ablation_id" "$producer" <<'PY'
import json, pathlib, sys
text=pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
decoder=json.JSONDecoder(); values=[]; offset=0
while offset < len(text):
    while offset < len(text) and text[offset].isspace(): offset += 1
    if offset == len(text): break
    value,offset=decoder.raw_decode(text,offset); values.append(value)
p=values[-1]
if not values or any(not isinstance(v,dict) or "progress" not in v for v in values[:-1]):
    raise SystemExit("REFUSED: unexpected epoch-curve dry-run stream")
a=p["inputs"]["learner_ablation"]; cmd=p["command"]
assert (p["arm_id"],p["subset_id"],p["world_size"],p["global_batch_size"]) == ("n256","full-56k",8,4096)
assert (a["ablation_id"],a["diagnostic_only"],a["promotion_eligible"]) == (sys.argv[3],True,False)
assert set(a["recipe_drift"]) == {"epochs","lr","loser_sample_weight"}
assert a["effective_recipe"]["epochs"] == 3 and a["effective_recipe"]["lr"] == float(sys.argv[2])
assert a["effective_recipe"]["loser_sample_weight"] == 1.0
assert cmd[cmd.index("--init-checkpoint")+1] == sys.argv[4]
assert "--no-resume-optimizer" in cmd and "--save-each-epoch" in cmd
assert cmd[cmd.index("--train-diagnostics-every-batches")+1] == "100"
print(json.dumps({"epochs":[1,2,3],"half_epoch_checkpoint":None,
 "half_epoch_omission":"only validated epoch-boundary checkpoints are comparable",
 "clean_producer_start":True,"resume_optimizer":False,"diagnostic_only":True},sort_keys=True))
PY
}

verify_completed() {
  "$python" - "$repo" "$receipt" "$ablation_id" "$lr" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tools import a1_dual_arm_train as train
r=train.verify_receipt(Path(sys.argv[2]))
a=r.get("inputs",{}).get("learner_ablation",{})
if (r.get("arm_id"),r.get("subset_id")) != ("n256","full-56k"):
    raise SystemExit("REFUSED: completed curve receipt belongs to another dose")
if (a.get("ablation_id"),a.get("diagnostic_only"),a.get("promotion_eligible")) != (sys.argv[3],True,False):
    raise SystemExit("REFUSED: completed curve ablation binding drift")
effective=a.get("effective_recipe",{})
if (effective.get("epochs"),effective.get("lr"),effective.get("loser_sample_weight")) != (3,float(sys.argv[4]),1.0):
    raise SystemExit("REFUSED: completed curve recipe drift")
epochs=r.get("outputs",{}).get("epoch_checkpoints",{})
if set(epochs) != {"1","2","3"}:
    raise SystemExit("REFUSED: completed curve receipt lacks epoch checkpoints")
for key,row in epochs.items():
    if row.get("exposures") != float(key):
        raise SystemExit("REFUSED: curve exposure label drift")
    for field in ("checkpoint","optimizer"):
        ref=row.get(field,{}); path=Path(str(ref.get("path","")))
        if ref != train._file_ref(path,where=f"curve epoch {key} {field}"):
            raise SystemExit(f"REFUSED: curve epoch {key} {field} bytes drift")
print("authenticated epoch-curve receipt; no retraining")
PY
}

publish_manifest() {
  local manifest=$out/curve.manifest.json
  [[ ! -e "$manifest" ]] || return 0
  "$python" - "$receipt" "$anchor_json" "$manifest" <<'PY'
import json, os, sys
from pathlib import Path
r=json.load(open(sys.argv[1],encoding="utf-8")); anchor=json.loads(sys.argv[2])
epochs=r.get("outputs",{}).get("epoch_checkpoints",{})
if set(epochs) != {"1","2","3"}:
    raise SystemExit("REFUSED: completed curve receipt lacks epoch checkpoints")
value={
 "schema_version":"a1-n256-epoch-curve-v1","diagnostic_only":True,
 "promotion_eligible":False,"clean_producer_start":True,"optimizer_resumed":False,
 "anchor":anchor,"receipt":str(Path(sys.argv[1]).resolve(strict=True)),
 "checkpoints":epochs,"exposure_points":[1.0,2.0,3.0],"half_epoch_checkpoint":None,
 "half_epoch_omission":"only validated epoch-boundary checkpoints are comparable",
 "external_micro_panel":{"baseline":"catanatron_value","pairs_per_checkpoint":64,
   "seat_swapped":True,"common_random_numbers":True,"proposed_base_seed":6199300000,
   "launch_authorized":False},
}
fd=os.open(sys.argv[3],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,"w",encoding="utf-8") as f:
    json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
PY
}

if [[ "$mode" == dry-run ]]; then
  tmp=$dose/dry-run.json.tmp.$$
  "${command[@]}" >"$tmp"
  validate_plan "$tmp"
  if [[ -e "$dose/dry-run.json" ]]; then
    cmp -s "$tmp" "$dose/dry-run.json" || {
      echo "REFUSED: reviewed epoch-curve dry run drift" >&2; exit 2;
    }
    rm -f "$tmp"
  else
    chmod 0444 "$tmp"; mv "$tmp" "$dose/dry-run.json"
  fi
  exit 0
fi

if [[ -s "$receipt" ]]; then
  verify_completed
  publish_manifest
  exit 0
fi
for partial in "$dose/candidate.pt" "$dose/candidate.pt.optimizer.pt" \
  "$dose/report.json" "$dose/go.json" "$dose/go.stderr.log"; do
  [[ ! -e "$partial" ]] || {
    echo "REFUSED: partial epoch-curve outputs exist without receipt: $partial" >&2
    exit 2
  }
done
[[ $(systemctl is-active nvidia-mps.service) == active ]] || {
  echo "REFUSED: MPS must be active at ownership handoff" >&2; exit 2;
}
restore_mps() { sudo -n systemctl start nvidia-mps.service || true; }
trap restore_mps EXIT
sudo -n systemctl stop nvidia-mps.service
go_tmp=$dose/go.json.tmp.$$
err_tmp=$dose/go.stderr.log.tmp.$$
"${command[@]}" --go >"$go_tmp" 2>"$err_tmp"
test -s "$receipt"
chmod 0444 "$go_tmp" "$err_tmp"
mv "$go_tmp" "$dose/go.json"; mv "$err_tmp" "$dose/go.stderr.log"
verify_completed
publish_manifest
