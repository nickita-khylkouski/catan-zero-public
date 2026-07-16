from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import evaluate  # noqa: E402
import generate  # noqa: E402
import train  # noqa: E402


@pytest.mark.parametrize(
    ("validator", "relative_path"),
    (
        (
            generate._validate_config,  # noqa: SLF001
            "configs/generation/coherent_public_n128.schema18.json",
        ),
        (
            evaluate._validate_config,  # noqa: SLF001
            "configs/eval/coherent_public_n128.schema18.json",
        ),
    ),
)
def test_canonical_generation_and_evaluation_accept_only_exact_payload(
    validator, relative_path: str, tmp_path: Path
) -> None:
    source = ROOT / relative_path
    validator(source)

    payload = json.loads(source.read_text(encoding="utf-8"))
    fields = payload["fields"]
    if payload["pipeline"] == "generate":
        fields["score_actions"] = not fields["score_actions"]
    else:
        fields["play_sh_winner"] = not fields["play_sh_winner"]
    drifted = tmp_path / source.name
    drifted.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exact commissioned canonical payload"):
        validator(drifted)


def test_canonical_training_accepts_only_exact_payload(tmp_path: Path) -> None:
    source = ROOT / "configs/training/a1_current_35m_b200.schema1.json"
    train._load_recipe(source)  # noqa: SLF001

    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["train_config"]["fields"]["action_target_gather"] = False
    drifted = tmp_path / source.name
    drifted.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="exact commissioned payload"):
        train._load_recipe(drifted)  # noqa: SLF001
