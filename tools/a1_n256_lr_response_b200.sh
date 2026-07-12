#!/usr/bin/env bash
# Diagnostic-only n256 LR-response points around the completed 1.2e-4 midpoint.
# No invocation is promotion eligible; execution is inert unless --go is explicit.
set -euo pipefail

usage() { echo "usage: $0 --lr {6e-5|2.4e-4} [--go]" >&2; exit 2; }

lr_input=
mode=dry-run
while (($#)); do
  case "$1" in
    --lr) (($# >= 2)) || usage; [[ -z "$lr_input" ]] || usage; lr_input=$2; shift 2 ;;
    --go) [[ "$mode" == dry-run ]] || usage; mode=go; shift ;;
    *) usage ;;
  esac
done
[[ -n "$lr_input" ]] || usage
case "$lr_input" in
  6e-5|0.00006) lr=0.00006; lr_label=lr60u ;;
  2.4e-4|0.00024) lr=0.00024; lr_label=lr240u ;;
  *) echo "REFUSED: --lr must be exactly 6e-5 or 2.4e-4" >&2; exit 2 ;;
esac

root=${A1_COMBINED_ROOT:-/home/ubuntu/experimental_nonpromotable/a1-combined-80-20-20260711}
script_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
repo=${A1_REPO:-$script_repo}
python=${A1_PYTHON:-$repo/.venv/bin/python}
producer=${A1_PRODUCER:-/home/ubuntu/catan-zero-production/runs/learner/a1-infoset-n128-20260710-r2/candidate.pt}
generation_contracts=${A1_CONTRACTS:-/home/ubuntu/catan-zero-production/contracts/a1-dual-arm-20260710-r1/locks}
midpoint_dir=${A1_CORRECTIVE_MIDPOINT_DIR:-$root/training/corrective-196k-lr120u-loser1/n256}
midpoint_receipt=$midpoint_dir/training.receipt.json
midpoint_spec=$midpoint_dir/learner.spec.json
midpoint_lock=$midpoint_dir/learner.lock.json
data=$root/n256-early/n256.memmap
validation=$root/n256-early/n256.validation_seeds.json
out=$root/training/n256-lr-response-${lr_label}-loser1
dose=$out/n256
repo_commit=$(git -C "$repo" rev-parse HEAD)
[[ -z $(git -C "$repo" status --porcelain --untracked-files=no) ]] || {
  echo "REFUSED: LR-response runtime repository has tracked modifications" >&2; exit 2;
}
shared_contract_dir=$root/training/n256-lr-response-contracts/$repo_commit
spec=$shared_contract_dir/learner.spec.json
lock=$shared_contract_dir/learner.lock.json
receipt=$dose/training.receipt.json
ablation_id=n256-lr-response-${lr_label}-loser1
overrides=$(printf '{"loser_sample_weight":1.0,"lr":%s}' "$lr")

cd "$repo"
mkdir -p "$dose"

lock_sha() { sha256sum "$1" | awk '{print "sha256:"$1}'; }
code_sha() {
  local learner_lock=$1
  "$python" - "$repo" "$learner_lock" "$(lock_sha "$learner_lock")" <<'PY'
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
}

verify_midpoint() {
  [[ -s "$midpoint_receipt" ]] || {
    echo "REFUSED: completed 1.2e-4 n256 midpoint receipt is required" >&2; exit 2;
  }
  "$python" - "$repo" "$midpoint_receipt" "$midpoint_spec" "$midpoint_lock" <<'PY'
import hashlib, json, stat, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tools import a1_dual_arm_train as train

def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()

r = train.verify_receipt(Path(sys.argv[2]))
if (r.get("arm_id"), r.get("subset_id")) != ("n256", "full-56k"):
    raise SystemExit("REFUSED: midpoint receipt is not the full n256 dose")
a = r.get("inputs", {}).get("learner_ablation", {})
if (a.get("ablation_id"), a.get("diagnostic_only"), a.get("promotion_eligible")) != (
    "all-196k-corrective-lr120u-loser1", True, False
):
    raise SystemExit("REFUSED: midpoint receipt is not the corrective diagnostic")
effective = a.get("effective_recipe", {})
if effective.get("lr") != 0.00012 or effective.get("loser_sample_weight") != 1.0:
    raise SystemExit("REFUSED: midpoint receipt recipe drift")
spec = Path(sys.argv[3]).expanduser().resolve(strict=True)
lock = Path(sys.argv[4]).expanduser().resolve(strict=True)
for path in (spec, lock):
    if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise SystemExit(f"REFUSED: midpoint contract is writable: {path}")
lock_ref = r.get("inputs", {}).get("learner_lock")
if lock_ref != train._file_ref(lock, where="midpoint learner lock"):
    raise SystemExit("REFUSED: midpoint receipt binds another learner lock")
try:
    authority = json.loads(lock.read_text(encoding="utf-8"))
    reviewed_spec = json.loads(spec.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"REFUSED: cannot parse midpoint learner contract: {error}") from error
expected_lock_fields = {
    "schema_version", "arm_id", "subset_id", "generation_arm_lock",
    "generation_contract_sha256", "learner_spec", "objective", "recipe",
    "topology", "inputs", "payload_inventory_sha256", "data_fingerprint",
    "row_counts", "selected_game_seed_set_sha256",
    "training_game_seed_set_sha256", "validation_game_seed_set_sha256",
    "runtime", "runtime_sha256", "trainer_report_bindings", "lock_sha256",
}
if not isinstance(authority, dict) or set(authority) != expected_lock_fields:
    raise SystemExit("REFUSED: midpoint learner lock fields drift")
unhashed = dict(authority); stated_lock_sha = unhashed.pop("lock_sha256", None)
if authority.get("schema_version") != "a1-dual-arm-learner-lock-v1" or stated_lock_sha != digest(unhashed):
    raise SystemExit("REFUSED: midpoint learner lock schema/digest drift")
runtime = authority.get("runtime")
if not isinstance(runtime, list) or authority.get("runtime_sha256") != digest(runtime):
    raise SystemExit("REFUSED: midpoint historical runtime binding drift")
if authority.get("learner_spec") != train._file_ref(spec, where="midpoint learner spec"):
    raise SystemExit("REFUSED: midpoint learner lock binds another spec")
if (authority.get("arm_id"), authority.get("subset_id")) != ("n256", "full-56k"):
    raise SystemExit("REFUSED: midpoint learner contract is not full n256")
expected_topology = {
    "world_size": 8, "local_batch_size": 512, "grad_accum_steps": 1,
    "global_batch_size": 4096, "data_format": "memmap",
    "ddp_shard_data": False, "fsdp": False,
}
if authority.get("topology") != expected_topology:
    raise SystemExit("REFUSED: midpoint learner contract is not world8")
if authority.get("recipe", {}).get("epochs") != 1:
    raise SystemExit("REFUSED: midpoint learner contract is not exactly one epoch")
expected_spec_fields = {"schema_version", "arm_id", "subset_id", "objective", "recipe", "topology"}
if (
    not isinstance(reviewed_spec, dict)
    or set(reviewed_spec) != expected_spec_fields
    or reviewed_spec.get("schema_version") != "a1-dual-arm-learner-spec-v1"
    or reviewed_spec.get("arm_id") != authority.get("arm_id")
    or reviewed_spec.get("subset_id") != authority.get("subset_id")
    or reviewed_spec.get("objective") != authority.get("objective")
    or reviewed_spec.get("recipe") != authority.get("recipe")
    or reviewed_spec.get("topology") != authority.get("topology")
):
    raise SystemExit("REFUSED: midpoint learner spec/lock semantics drift")
lock_inputs = authority.get("inputs")
if not isinstance(lock_inputs, dict) or set(lock_inputs) != {
    "corpus_meta", "selected_manifest", "audit", "validation", "producer"
}:
    raise SystemExit("REFUSED: midpoint learner input bindings drift")
receipt_inputs = r.get("inputs", {})
for name, ref in lock_inputs.items():
    if ref != train._file_ref(Path(ref.get("path", "")), where=f"midpoint {name}"):
        raise SystemExit(f"REFUSED: midpoint learner input bytes drift: {name}")
    if receipt_inputs.get(name) != ref:
        raise SystemExit(f"REFUSED: midpoint receipt/lock input mismatch: {name}")
print("authenticated completed n256 midpoint at lr=1.2e-4")
PY
}

prepare_shared_contract() {
  mkdir -p "$shared_contract_dir"
  exec {contract_fd}>"$shared_contract_dir/build.lock"
  flock "$contract_fd"
  if [[ -s "$spec" && -s "$lock" ]]; then
    :
  elif [[ -e "$lock" ]]; then
    echo "REFUSED: shared current-runtime lock exists without its spec" >&2
    exit 2
  else
    if [[ ! -s "$spec" ]]; then
      [[ ! -e "$spec" ]] || {
        echo "REFUSED: shared current-runtime spec is incomplete" >&2; exit 2;
      }
      local spec_tmp=$spec.tmp.$$
      cp "$midpoint_spec" "$spec_tmp"
      chmod 0444 "$spec_tmp"
      mv "$spec_tmp" "$spec"
    fi
    # One authoritative payload scan creates a lock whose runtime records match
    # this exact clean commit. Every LR arm then reuses these immutable bytes.
    "$python" "$repo/tools/a1_dual_learner_contract.py" seal \
      --arm-lock "$generation_contracts/n256.lock.json" --learner-spec "$spec" \
      --data "$data" --validation "$validation" \
      --producer-checkpoint "$producer" --out "$lock"
  fi
  "$python" "$repo/tools/a1_dual_learner_contract.py" verify \
    --lock "$lock" --reviewed-lock-file-sha256 "$(lock_sha "$lock")" >/dev/null
  flock -u "$contract_fd"
  exec {contract_fd}>&-
}

verify_completed() {
  "$python" - "$repo" "$receipt" "$ablation_id" "$lr" "$dose/candidate.pt" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tools import a1_dual_arm_train as train
r = train.verify_receipt(Path(sys.argv[2]))
if (r.get("arm_id"), r.get("subset_id")) != ("n256", "full-56k"):
    raise SystemExit("REFUSED: LR-response receipt belongs to another dose")
a = r.get("inputs", {}).get("learner_ablation", {})
if (a.get("ablation_id"), a.get("diagnostic_only"), a.get("promotion_eligible")) != (
    sys.argv[3], True, False
):
    raise SystemExit("REFUSED: LR-response ablation provenance drift")
effective = a.get("effective_recipe", {})
if effective.get("lr") != float(sys.argv[4]) or effective.get("loser_sample_weight") != 1.0:
    raise SystemExit("REFUSED: LR-response effective recipe drift")
checkpoint = r.get("outputs", {}).get("checkpoint", {})
if checkpoint.get("path") != str(Path(sys.argv[5]).resolve(strict=True)):
    raise SystemExit("REFUSED: LR-response receipt binds another checkpoint")
print("authenticated completed diagnostic LR-response dose; no retraining")
PY
}

if [[ -s "$receipt" ]]; then
  verify_completed
  exit 0
fi
[[ "$mode" != go ]] || verify_midpoint
for partial in "$dose/candidate.pt" "$dose/candidate.pt.optimizer.pt" \
  "$dose/report.json" "$dose/go.json" "$dose/go.stderr.log"; do
  [[ ! -e "$partial" ]] || {
    echo "REFUSED: partial LR-response outputs exist without a completed receipt: $partial" >&2
    exit 2
  }
done
[[ "$mode" == go ]] || verify_midpoint
prepare_shared_contract

lock_digest=$(lock_sha "$lock")
code_digest=$(code_sha "$lock")
command=(
  "$python" "$repo/tools/a1_dual_arm_train.py"
  --data "$data" --learner-lock "$lock"
  --reviewed-lock-file-sha256 "$lock_digest"
  --validation-manifest "$validation" --producer-checkpoint "$producer"
  --ablation-id "$ablation_id"
  --recipe-overrides-json "$overrides"
  --ablation-code-tree-sha256 "$code_digest"
  --checkpoint "$dose/candidate.pt" --report "$dose/report.json"
  --receipt "$receipt" --python "$python"
)

validate_plan() {
  local plan_file=$1
  "$python" - "$plan_file" "$lr" "$ablation_id" <<'PY'
import json, pathlib, sys
text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
decoder = json.JSONDecoder(); values = []; offset = 0
while offset < len(text):
    while offset < len(text) and text[offset].isspace(): offset += 1
    if offset == len(text): break
    value, offset = decoder.raw_decode(text, offset); values.append(value)
if not values or any(not isinstance(value, dict) or "progress" not in value for value in values[:-1]):
    raise SystemExit("REFUSED: unexpected LR-response dry-run stdout stream")
p = values[-1]
assert (p["arm_id"], p["subset_id"]) == ("n256", "full-56k")
assert p["world_size"] == 8 and p["global_batch_size"] == 4096
a = p["inputs"]["learner_ablation"]
assert (a["ablation_id"], a["diagnostic_only"], a["promotion_eligible"]) == (sys.argv[3], True, False)
assert set(a["recipe_drift"]) == {"lr", "loser_sample_weight"}
assert a["effective_recipe"]["epochs"] == 1
assert a["effective_recipe"]["lr"] == float(sys.argv[2])
assert a["effective_recipe"]["loser_sample_weight"] == 1.0
print(json.dumps({"ablation_id": sys.argv[3], "lr": float(sys.argv[2]), "world_size": 8,
                  "global_batch_size": 4096, "epochs": 1, "diagnostic_only": True}, sort_keys=True))
PY
}

if [[ "$mode" == dry-run ]]; then
  tmp=$dose/dry-run.json.tmp.$$
  "${command[@]}" >"$tmp"
  validate_plan "$tmp"
  if [[ -e "$dose/dry-run.json" ]]; then
    cmp -s "$tmp" "$dose/dry-run.json" || {
      echo "REFUSED: reviewed LR-response dry run drift" >&2; exit 2;
    }
    rm -f "$tmp"
  else
    chmod 0444 "$tmp"; mv "$tmp" "$dose/dry-run.json"
  fi
  echo "prepared diagnostic n256 LR-response ${lr_label}: lr=$lr, loser_sample_weight=1.0, one epoch, world8/global-batch4096"
  exit 0
fi

[[ $(systemctl is-active nvidia-mps.service) == active ]] || {
  echo "REFUSED: MPS must be active at ownership handoff" >&2; exit 2;
}
restore_mps() { sudo -n systemctl start nvidia-mps.service || true; }
trap restore_mps EXIT
sudo -n systemctl stop nvidia-mps.service

go_tmp=$dose/go.json.tmp.$$
err_tmp=$dose/go.stderr.log.tmp.$$
if "${command[@]}" --go >"$go_tmp" 2>"$err_tmp"; then
  test -s "$receipt"
  chmod 0444 "$go_tmp" "$err_tmp"
  mv "$go_tmp" "$dose/go.json"
  mv "$err_tmp" "$dose/go.stderr.log"
else
  echo "LR-response executor failed; preserving $go_tmp and $err_tmp" >&2
  exit 1
fi
verify_completed
