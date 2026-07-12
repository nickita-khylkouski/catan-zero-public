"""Structural guard on train_bc.py's `report` dict provenance fields (FIX 3,
task #85 hygiene batch): `--data`, the resolved `validation_game_seed_ranges`,
and `truncated_vp_margin_value_weight` must all be recorded in report.json so
a run can be reproduced/audited without re-deriving them from train.log.

Parsed statically via `ast` (matching tests/test_cli_config_drift.py's
approach) so this doesn't need to execute a real training run to check the
report dict's literal key set.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_BC_PATH = REPO_ROOT / "tools" / "train_bc.py"

REQUIRED_REPORT_KEYS = (
    "data",
    "validation_game_seed_ranges",
    "truncated_vp_margin_value_weight",
    "value_lr_mult",
)


def _find_report_dict_keys() -> set[str]:
    """Locate the `report = {...}` assignment inside train_bc.main() and
    return its literal string keys."""
    tree = ast.parse(TRAIN_BC_PATH.read_text(), filename=str(TRAIN_BC_PATH))
    main_func = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "main"),
        None,
    )
    assert main_func is not None, "expected a top-level def main() in train_bc.py"

    for node in ast.walk(main_func):
        if not isinstance(node, ast.Assign):
            continue
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "report"):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        keys = set()
        for key_node in node.value.keys:
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                keys.add(key_node.value)
        return keys
    raise AssertionError("could not find `report = {...}` dict literal in train_bc.main()")


def test_report_dict_records_data_and_validation_seed_ranges_and_vp_margin_weight() -> None:
    keys = _find_report_dict_keys()
    missing = [key for key in REQUIRED_REPORT_KEYS if key not in keys]
    assert not missing, f"train_bc.py report dict is missing provenance keys: {missing}"
