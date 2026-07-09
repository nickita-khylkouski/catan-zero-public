from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Same bootstrap pattern as tests/test_perf_snapshot.py -- tools/opt_sweep.py
# does bare sibling imports (`import perf_common`, `import perf_snapshot`).
_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import opt_sweep  # type: ignore  # noqa: E402
import perf_snapshot  # type: ignore  # noqa: E402

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def _subprocess_env() -> dict[str, str]:
    # Unlike the in-process tests above, a subprocess doesn't inherit
    # pytest's `pythonpath = ["src"]` ini setting -- opt_sweep.py's leaf-mode
    # code path does `from catan_zero...` imports, so `src/` must be on
    # PYTHONPATH explicitly for the child process.
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_SRC_DIR) + (os.pathsep + existing if existing else "")
    return env


def _rust_available() -> bool:
    try:
        from catan_zero.search.rust_mcts import _require_rust_module

        _require_rust_module()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _rust_available(), reason="catanatron_rs Rust binding not installed in this environment"
)


def test_run_ab_sweep_produces_before_and_after_rows() -> None:
    report = opt_sweep.run_ab_sweep(
        num_evals=5, seed=13, device="cpu", checkpoint=None, public_observation=False
    )
    assert report["before"]["rust_featurize"] is False
    assert report["after"]["rust_featurize"] is True
    # Both runs must have profiled the same number of (deterministically
    # collected) leaf states -- otherwise this isn't an apples-to-apples A/B.
    assert report["before"]["num_evals"] == report["after"]["num_evals"] == 5


def test_run_ab_sweep_rust_featurize_stage_is_faster_than_legacy() -> None:
    """The whole point of task #81/CAT-65: the native featurize/context path
    must not be SLOWER than the Python path it replaces, on the same
    deterministic leaf-state sequence."""
    report = opt_sweep.run_ab_sweep(
        num_evals=20, seed=13, device="cpu", checkpoint=None, public_observation=False
    )
    for stage in ("featurize", "ffi_context"):
        d = report["stage_deltas"][stage]
        assert d["after_total_ms"] < d["before_total_ms"], (
            f"rust_featurize path regressed on stage {stage!r}: "
            f"before={d['before_total_ms']}ms after={d['after_total_ms']}ms"
        )


def test_ranking_excludes_nn_forward_without_real_checkpoint() -> None:
    report = opt_sweep.run_ab_sweep(
        num_evals=5, seed=13, device="cpu", checkpoint=None, public_observation=False
    )
    ranked_names = {name for name, _pct in report["ranked_stages_after"]}
    assert "nn_forward" not in ranked_names
    assert "total" not in ranked_names
    assert "postprocess" not in ranked_names
    # every other named stage from perf_snapshot._NAMED_STAGES should be present
    expected = set(perf_snapshot._NAMED_STAGES) - {"nn_forward"}
    assert ranked_names == expected


def test_ranked_stages_after_sorted_descending() -> None:
    report = opt_sweep.run_ab_sweep(
        num_evals=20, seed=13, device="cpu", checkpoint=None, public_observation=False
    )
    pcts = [pct for _name, pct in report["ranked_stages_after"]]
    assert pcts == sorted(pcts, reverse=True)


def test_top_cost_post_rust_is_a_named_stage() -> None:
    report = opt_sweep.run_ab_sweep(
        num_evals=20, seed=13, device="cpu", checkpoint=None, public_observation=False
    )
    assert report["top_cost_post_rust"] in perf_snapshot._NAMED_STAGES


def test_cli_writes_report_and_ledger(tmp_path) -> None:
    import subprocess

    out_path = tmp_path / "report.json"
    ledger_path = tmp_path / "ledger.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(_TOOLS_DIR / "opt_sweep.py"),
            "--num-evals",
            "5",
            "--seed",
            "13",
            "--device",
            "cpu",
            "--out",
            str(out_path),
            "--ledger",
            str(ledger_path),
        ],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    import json

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["kind"] == "opt_sweep_ab"
    assert ledger_path.exists()
    ledger_lines = [line for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # both the before (legacy) and after (rust_featurize) leaf rows get appended
    assert len(ledger_lines) == 2


def test_cli_no_ledger_flag_skips_append(tmp_path) -> None:
    import subprocess

    ledger_path = tmp_path / "ledger.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(_TOOLS_DIR / "opt_sweep.py"),
            "--num-evals",
            "5",
            "--seed",
            "13",
            "--device",
            "cpu",
            "--no-ledger",
            "--ledger",
            str(ledger_path),
        ],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stderr
    assert not ledger_path.exists()
