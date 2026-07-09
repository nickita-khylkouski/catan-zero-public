"""CAT-30 regression guard: anchor telemetry must never gate a promotion.

Roadmap Sec 1 standing rule (R8/gen-4 lesson): anchor telemetry is a drift
TRIPWIRE ONLY, never a promotion signal -- gen-4 showed "the historical
promotion signature" and still gated flat, so a flat/healthy anchor cannot be
trusted to predict a flat/healthy gate either. These tests audit the actual
code (static + a live dry-run), not just the docstrings/comments that say so.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_TOOLS_DIR = _REPO / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

FLYWHEEL_PATH = _TOOLS_DIR / "continuous_flywheel.py"
_SOURCE = FLYWHEEL_PATH.read_text()
_TREE = ast.parse(_SOURCE, filename=str(FLYWHEEL_PATH))


def _find_function(name: str) -> ast.FunctionDef:
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no top-level function {name!r} found in {FLYWHEEL_PATH}")


def _find_method(cls_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(_TREE):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"no method {cls_name}.{method_name} found in {FLYWHEEL_PATH}")


def _references_anchor(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and "anchor" in child.id.lower():
            return True
        if isinstance(child, ast.Attribute) and "anchor" in child.attr.lower():
            return True
        if isinstance(child, ast.Constant) and isinstance(child.value, str) and "anchor" in child.value.lower():
            return True
    return False


def test_gate_methods_never_reference_anchor():
    """Runner.gate/_gate_h2h/_gate_scoreboard -- the ONLY code that decides
    promote-vs-hold -- must not reference anything anchor-related anywhere in
    their body. If this ever fails, someone wired anchor telemetry into the
    promotion decision, which is exactly the mistake CAT-30 exists to
    prevent."""
    for method in ("gate", "_gate_h2h", "_gate_scoreboard"):
        node = _find_method("Runner", method)
        assert not _references_anchor(node), (
            f"Runner.{method} references 'anchor' -- the promotion gate must never "
            "read anchor telemetry (Roadmap Sec 1 standing rule, R8/gen-4 lesson)."
        )


def test_anchor_probe_call_happens_after_decision_is_finalized_in_main():
    """Structural ordering check in main(): every assignment to
    rec["decision"] (abort_generation / abort_train / skip_zero_rows /
    promoting / promote(vN) / hold(...)) must occur, in source order, BEFORE
    the anchor_probe() call -- i.e. the promotion decision can never be
    computed using (or after conditionally depending on) anchor results."""
    main_fn = _find_function("main")
    decision_linenos = []
    anchor_probe_linenos = []
    for node in ast.walk(main_fn):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            if isinstance(node.slice, ast.Constant) and node.slice.value == "decision":
                decision_linenos.append(node.lineno)
        if isinstance(node, ast.Call):
            func = node.func
            attr = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if attr == "anchor_probe":
                anchor_probe_linenos.append(node.lineno)
    assert decision_linenos, "expected at least one rec['decision'] = ... assignment in main()"
    assert anchor_probe_linenos, "expected an anchor_probe(...) call in main()"
    assert max(decision_linenos) < min(anchor_probe_linenos), (
        f"a rec['decision'] assignment at line {max(decision_linenos)} occurs AFTER "
        f"(or the anchor_probe call at line {min(anchor_probe_linenos)} occurs BEFORE) "
        "the anchor probe -- the promotion decision must be fully finalized before "
        "anchor telemetry is even computed."
    )


def test_anchor_telemetry_stored_under_a_separate_journal_key_not_decision():
    """rec["anchor_telemetry"] must be a distinct dict key from rec["decision"]
    in the journal -- i.e. anchor results are stored as parallel telemetry,
    not merged into (or replacing) the promotion verdict."""
    main_fn = _find_function("main")
    keys_assigned = set()
    for node in ast.walk(main_fn):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            if isinstance(node.value, ast.Name) and node.value.id == "rec":
                if isinstance(node.slice, ast.Constant):
                    keys_assigned.add(node.slice.value)
    assert "anchor_telemetry" in keys_assigned
    assert "decision" in keys_assigned
    assert "anchor_telemetry" != "decision"


def test_dry_run_end_to_end_anchor_telemetry_does_not_affect_promotion(tmp_path, monkeypatch):
    """Live dry-run smoke test (the ticket's prescribed verification): run the
    real control flow with anchor corpora configured but never built on disk,
    and confirm every round still promotes based purely on the (stubbed) h2h
    gate winrate -- unaffected by anchor telemetry being present/absent/failed."""
    import importlib

    cf = importlib.import_module("continuous_flywheel")
    seed_ckpt = tmp_path / "seed.pt"
    seed_ckpt.write_text("stub")
    loop_dir = tmp_path / "loop"

    argv = [
        "continuous_flywheel.py",
        "--loop-dir", str(loop_dir),
        "--seed-checkpoint", str(seed_ckpt),
        "--dry-run",
        "--max-rounds", "2",
        "--anchor-corpus", "anchor_r7",
        "--anchor-corpus", "anchor_gen4",
        "--anchor-eval-every-rounds", "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = cf.main()
    assert rc == 0

    journal = json.loads((loop_dir / "flywheel_state.json").read_text())
    telemetry = json.loads((loop_dir / "anchor_telemetry.json").read_text())
    assert set(telemetry.keys()) == {"anchor_r7", "anchor_gen4"}
    for rec in journal["rounds"]:
        # dry-run's stubbed h2h gate always promotes (winrate >= min_winrate at
        # these round indices) -- decision must be the promote(...) form
        # regardless of anchor_telemetry's presence in the same record.
        assert rec["decision"].startswith("promote(")
        assert "anchor_telemetry" in rec
