from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
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
            "configs/generation/coherent_public_n128.schema19.json",
        ),
        (
            evaluate._validate_config,  # noqa: SLF001
            "configs/eval/coherent_public_n128.schema19.json",
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


@pytest.mark.parametrize("launcher", ("generate.py", "evaluate.py", "train.py"))
def test_canonical_launchers_ignore_ambient_stale_pythonpath(
    launcher: str, tmp_path: Path
) -> None:
    stale = tmp_path / "stale"
    package = stale / "catan_zero"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise RuntimeError('ambient stale package imported')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(stale)

    completed = subprocess.run(
        [sys.executable, str(TOOLS / launcher), "--help"],
        cwd=tmp_path,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
    assert "ambient stale package imported" not in completed.stderr


def test_canonical_evaluation_uses_the_production_promotion_gate() -> None:
    payload = json.loads(
        (
            ROOT / "configs/eval/coherent_public_n128.schema19.json"
        ).read_text(encoding="utf-8")
    )

    assert payload["fields"]["elo0"] == -10.0
    assert payload["fields"]["elo1"] == 15.0


@pytest.mark.parametrize("threads_per_worker", (0, 6))
def test_canonical_evaluation_forwards_cpu_placement(
    threads_per_worker: int,
) -> None:
    parser = evaluate.build_parser()
    args = parser.parse_args(
        [
            "--config",
            "eval.json",
            "--candidate",
            "candidate.pt",
            "--champion",
            "champion.pt",
            "--out",
            "report.json",
            "--pairs",
            "16",
            "--workers",
            "8",
            "--devices",
            "cuda:0,cuda:1",
            "--threads-per-worker",
            str(threads_per_worker),
            "--base-seed",
            "1",
        ]
    )

    forwarded = evaluate._executor_argv(args)  # noqa: SLF001
    index = forwarded.index("--threads-per-worker")
    assert forwarded[index + 1] == str(threads_per_worker)
