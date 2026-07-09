from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

# `tools/perf_snapshot.py` does bare sibling imports (`import perf_common`,
# `from bench_leaf_eval_batching import ...`), so it only works with `tools/`
# itself on sys.path -- matches the bootstrap pattern in
# tests/test_generate_gumbel_selfplay_data.py.
_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import perf_common  # type: ignore  # noqa: E402
import perf_snapshot  # type: ignore  # noqa: E402


def _ns(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def test_run_genlog_mode(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"rows": 3600, "elapsed_sec": 3600.0, "workers": 8, "out_dir": "runs/x", "games_completed": 10}
        ),
        encoding="utf-8",
    )
    row = perf_snapshot._run_genlog_mode(_ns(manifest=str(manifest), hostname="gpu0"))
    assert row["kind"] == "generation"
    assert row["hostname"] == "gpu0"
    assert row["rows_per_hr"] == 3600.0


def test_run_gate_mode_from_existing_summary(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"games": 200, "elapsed_sec": 5.0, "decision": "H1"}), encoding="utf-8")
    row = perf_snapshot._run_gate_mode(_ns(cmd=None, summary=str(summary), out_capture=None, gate_name="g1", hostname=None))
    assert row["games"] == 200
    assert row["decision"] == "H1"


def test_run_gate_mode_wraps_command_and_measures_wall_clock(tmp_path) -> None:
    out_capture = tmp_path / "out.json"
    cmd = (
        f"{sys.executable} -c \"import json; json.dump({{'games': 50, 'elapsed_sec': 999.0}}, "
        f"open(r'{out_capture}', 'w'))\""
    )
    row = perf_snapshot._run_gate_mode(
        _ns(cmd=cmd, summary=None, out_capture=str(out_capture), gate_name="g2", hostname=None)
    )
    assert row["games"] == 50
    # Our own wall-clock measurement must win over the (bogus) 999.0 the fake
    # gate command wrote into its own summary.
    assert row["elapsed_sec"] < 5.0


def test_run_gate_mode_requires_out_capture_with_cmd() -> None:
    with pytest.raises(SystemExit):
        perf_snapshot._run_gate_mode(_ns(cmd="true", summary=None, out_capture=None, gate_name="g", hostname=None))


def test_run_gate_mode_requires_cmd_or_summary() -> None:
    with pytest.raises(SystemExit):
        perf_snapshot._run_gate_mode(_ns(cmd=None, summary=None, out_capture=None, gate_name="g", hostname=None))


def test_run_gpu_util_mode(tmp_path) -> None:
    dmon = tmp_path / "dmon.txt"
    dmon.write_text(
        "# gpu   pwr gtemp mtemp    sm   mem\n"
        "# Idx     W     C     C     %     %\n"
        "    0   250    45    60    92     1\n",
        encoding="utf-8",
    )
    rows = perf_snapshot._run_gpu_util_mode(_ns(live=False, input=str(dmon), hostname=None))
    assert len(rows) == 1
    assert rows[0]["context_thrash_flagged"] is True
    assert rows[0]["sm_util_pct"] == 92.0


def test_run_gpu_util_mode_requires_live_or_input() -> None:
    with pytest.raises(SystemExit):
        perf_snapshot._run_gpu_util_mode(_ns(live=False, input=None, hostname=None))


def _rust_available() -> bool:
    try:
        from catan_zero.search.rust_mcts import _require_rust_module

        _require_rust_module()
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _rust_available(), reason="catanatron_rs Rust binding not installed in this environment"
)
def test_profile_leaf_eval_produces_all_stages_and_honest_baseline_check() -> None:
    row = perf_snapshot.profile_leaf_eval(
        num_evals=5,
        seed=11,
        device="cpu",
        checkpoint=None,
        public_observation=False,
    )
    assert row["kind"] == "leaf"
    assert row["num_evals"] == 5
    for stage in ("snapshot_fetch", "ffi_resolve", "featurize", "ffi_context", "nn_forward", "total"):
        assert row["stages"][stage]["n"] == 5.0
    # No --checkpoint given: the tiny fast policy can't reproduce the
    # documented CPU/GPU split (see bench_leaf_eval_batching.py's docstring),
    # so the baseline check must honestly report "not evaluated" rather than
    # silently claim a pass.
    assert row["baseline_check"]["matches_baseline"] is None


@pytest.mark.skipif(
    not _rust_available(), reason="catanatron_rs Rust binding not installed in this environment"
)
def test_profile_leaf_eval_times_real_ffi_cost_not_postprocess() -> None:
    """CAT-71 review finding 1 regression test.

    `EntityGraphRustEvaluator.evaluate()` resolves the entity adapter ITSELF
    (`_fetch_leaf_decision_inputs` + `_resolve_entity_adapter`) and passes
    the result as `resolved=` into `rust_game_to_entity_batch`/
    `rust_action_context_batch`, so on the real `evaluate()` path those two
    functions take their short-circuit branch and never repeat the
    resolution work. The old profiler only monkeypatched those two
    functions, so the real Rust FFI/snapshot-resolution cost silently fell
    into the residual "postprocess" bucket instead -- this is the exact
    check the original test omitted (it only asserted the stage percentages
    summed to 100%, not where the real cost landed).

    This must hold on the real evaluator path with no --checkpoint (a
    faithful stand-in: even the tiny fast policy pays the real per-leaf Rust
    FFI cost, which is what's under test here, not the forward pass).
    """
    row = perf_snapshot.profile_leaf_eval(
        num_evals=40,
        seed=17,
        device="cpu",
        checkpoint=None,
        public_observation=False,
    )
    pct = row["stage_pct_of_total"]
    ffi_and_featurize_pct = (
        pct["snapshot_fetch"] + pct["ffi_resolve"] + pct["featurize"] + pct["ffi_context"]
    )
    # Real measured split on this path is ~30% FFI/featurize vs ~5-7%
    # postprocess (vs. ~6%/~17% before this fix, with the missing ~19%
    # silently absorbed into postprocess) -- thresholds below leave generous
    # margin for host/CI variance while still failing if the FFI cost
    # regresses back into being untimed.
    assert ffi_and_featurize_pct > 15.0, row["stage_pct_of_total"]
    assert pct["postprocess"] < 20.0, row["stage_pct_of_total"]
    # The two newly-timed resolution stages must each have actually measured
    # something -- i.e. the monkeypatch actually intercepted real calls, not
    # zero-length lists silently no-op-ing.
    assert row["stages"]["snapshot_fetch"]["n"] == 40.0
    assert row["stages"]["ffi_resolve"]["n"] == 40.0
    pct = row["stage_pct_of_total"]
    assert abs(sum(pct.values()) - 100.0) < 1.0e-6


def test_cli_gen_log_end_to_end_appends_to_ledger(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rows": 1000, "elapsed_sec": 100.0, "out_dir": "x"}), encoding="utf-8")
    ledger = tmp_path / "ledger.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(_TOOLS_DIR / "perf_snapshot.py"),
            "--ledger",
            str(ledger),
            "gen-log",
            "--manifest",
            str(manifest),
            "--hostname",
            "hostA",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["written"] == 1
    ledger_rows = perf_common.load_ledger(ledger)
    assert ledger_rows[0]["hostname"] == "hostA"


def test_cli_no_ledger_flag_skips_append(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rows": 1000, "elapsed_sec": 100.0, "out_dir": "x"}), encoding="utf-8")
    ledger = tmp_path / "ledger.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(_TOOLS_DIR / "perf_snapshot.py"),
            "--ledger",
            str(ledger),
            "--no-ledger",
            "gen-log",
            "--manifest",
            str(manifest),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["written"] == 0
    assert not ledger.exists()
