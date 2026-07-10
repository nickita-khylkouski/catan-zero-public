#!/usr/bin/env bash
# CAT-129 canonical test/CI gate — the ONE mandatory pre-deploy + post-deploy
# (CAT-130) check, and consolidator's Wave-2 re-gate. Exit 0 IFF ALL stages pass:
#   suite  — full pytest suite green (GPU tests self-skip; CAT-94 concat + modal
#            collection-quarantined via conftest.py). This is the comprehensive
#            check; parity + CLI goldens are part of it.
#   parity — featurizer parity 19/19 (rust_featurize + action_context + symmetry),
#            re-run explicitly for a clear standalone PASS line.
#   goldens— CAT-75 CLI option-string goldens + CLI-drift/default guards.
#   noop   — champion_v0 no-op forward is BIT-IDENTICAL (flags OFF): 16-row
#            max_diff 0.0 + 64-row value_sha1 aba0012a / logits_sha1 a6bba3f3.
#            Byte-exact is same-CPU-vendor only (float32 reduction order
#            differs Intel vs AMD etc); cross-vendor fleet-box acceptance
#            sets NOOP_ATOL=1e-4 to accept a benign ~1e-6 delta instead.
#
# CPU-first: runs green off-GPU; use the current command output as evidence
# instead of preserving a stale test-count snapshot in this launcher.
# Usage:  bash scripts/gate.sh            # full gate
#         bash scripts/gate.sh --only suite|parity|goldens|noop
# Env:    PY=<python>  (default: .venv/bin/python)   REPO=<repo root>
#         NOOP_CHAMPION=<checkpoint> (default: ~/bundle/champion_v0.pt)
#         NOOP_ATOL=<float>  (default: unset = strict --atol 0.0 byte-exact;
#                    fleet-box acceptance across CPU vendors uses 1e-4)
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-$REPO/.venv/bin/python}"
cd "$REPO"
ulimit -n 65536 2>/dev/null || true
export PYTHONPATH="$REPO/src"
# CPU-first: run the gate off-GPU by default so it's box-independent. Override
# GATE_ALLOW_GPU=1 to let GPU tests run (they self-skip when CUDA is absent).
[ "${GATE_ALLOW_GPU:-0}" = "1" ] || export CUDA_VISIBLE_DEVICES=""

ONLY="${2:-all}"; [ "${1:-}" = "--only" ] && ONLY="${2:?--only needs a stage}" || ONLY="all"

QUARANTINE=(--ignore=tests/test_concat_memmap_corpus.py
            --ignore=tests/test_modal_gumbel_factory_legacy_guard.py)
PARITY=(tests/test_rust_featurize_parity.py
        tests/test_rust_action_context_parity.py
        tests/test_rust_symmetry_averaging_parity.py)
GOLDENS=(tests/test_cli_config_drift.py
         tests/test_train_bc_cli_defaults.py
         tests/test_n_full_wide_raw_policy_above_width_cli.py
         tests/test_launcher_guard_wiring.py)

rc=0
run() { echo -e "\n=== GATE STAGE: $1 ==="; shift; "$@"; local e=$?; [ $e -eq 0 ] || rc=1; return $e; }

stage_suite()  { run suite "$PY" -m pytest tests/ -q -p no:cacheprovider "${QUARANTINE[@]}"; }
stage_parity() { run "parity 19/19" "$PY" -m pytest "${PARITY[@]}" -q -p no:cacheprovider; }
stage_goldens(){ run "CAT-75 CLI goldens" "$PY" -m pytest "${GOLDENS[@]}" -q -p no:cacheprovider; }
stage_noop()   {
  local atol_args=()
  local champion_args=()
  [ -n "${NOOP_ATOL:-}" ] && atol_args=(--atol "$NOOP_ATOL")
  [ -n "${NOOP_CHAMPION:-}" ] && champion_args=(--champion "$NOOP_CHAMPION")
  run "champion no-op BIT-IDENTICAL" "$PY" scripts/check_champion_noop.py \
    "${champion_args[@]}" "${atol_args[@]}"
}

case "$ONLY" in
  suite)   stage_suite ;;
  parity)  stage_parity ;;
  goldens) stage_goldens ;;
  noop)    stage_noop ;;
  all)     stage_noop; stage_parity; stage_goldens; stage_suite ;;   # cheap checks first, full suite last
  *) echo "unknown --only '$ONLY'"; exit 2 ;;
esac

echo -e "\n=================== CAT-129 GATE $( [ $rc -eq 0 ] && echo PASS || echo FAIL ) ==================="
exit $rc
