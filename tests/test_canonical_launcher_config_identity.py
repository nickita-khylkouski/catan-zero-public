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


def test_canonical_generation_default_requires_adapter_v6() -> None:
    assert generate.REQUIRED_SCIENCE_FIELDS[
        "learner_entity_feature_adapter_version"
    ] == "rust_entity_adapter_v6_exact_actor_resources_initial_road_two_hop"


def test_canonical_generation_guard_is_bound_to_recipe() -> None:
    config = ROOT / "configs/generation/coherent_public_n128.schema21.json"
    expected = (
        ROOT
        / "configs/guards/"
        "a1_generation_coherent_public_n128_v4.json"
    )
    generate._validate_guard(config=config, guard=expected)  # noqa: SLF001

    with pytest.raises(ValueError, match="canonical generation guard mismatch"):
        generate._validate_guard(  # noqa: SLF001
            config=config,
            guard=ROOT / "configs/guards/generate_gumbel_selfplay_data.json",
        )


@pytest.mark.parametrize(
    ("validator", "relative_path"),
    (
        (
            generate._validate_config,  # noqa: SLF001
            "configs/generation/coherent_public_n128.schema21.json",
        ),
        (
            evaluate._validate_config,  # noqa: SLF001
            "configs/eval/coherent_public_n128.schema21.json",
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

    with pytest.raises(ValueError, match="checked-in regular file"):
        validator(drifted)


@pytest.mark.parametrize(
    "relative_path",
    (
        "configs/training/a1_current_35m_b200.schema1.json",
        "configs/training/a1_parent_update_35m_b200.schema1.json",
    ),
)
def test_canonical_training_accepts_only_exact_payload(
    relative_path: str,
) -> None:
    source = ROOT / relative_path
    train._load_recipe(source)  # noqa: SLF001

    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["train_config"]["fields"]["action_target_gather"] = False
    with pytest.raises(train.ProductionRecipeError, match="recipe bytes drifted"):
        train.require_production_recipe(
            entrypoint="train", path=source, payload=payload
        )


def test_canonical_training_routes_fresh_scratch_through_authenticated_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_engine_launch(**_kwargs) -> None:
        pytest.fail("direct scratch launch reached the internal training engine")

    monkeypatch.setattr(train, "_engine_namespace", unexpected_engine_launch)

    with pytest.raises(
        SystemExit,
        match=(
            r"not launch authority.*tools/a1_scratch_train\.py.*"
            r"authenticated plan.*--go"
        ),
    ):
        train.main(
            [
                "--config",
                str(ROOT / "configs/training/a1_current_35m_b200.schema1.json"),
                "--data",
                "/authenticated/composite.json",
                "--checkpoint",
                "/outputs/candidate.pt",
                "--report",
                "/outputs/report.json",
            ]
        )


def test_canonical_training_rejects_role_substitution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = ROOT / "configs/training/a1_parent_update_35m_b200.schema1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["engine_settings"]["initialization_mode"] = "scratch_fresh_optimizer"
    drifted = tmp_path / source.name
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        train,
        "require_production_recipe",
        lambda **_kwargs: "a1-parent-update-35m-b200",
    )

    with pytest.raises(SystemExit, match="role does not match"):
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
            ROOT / "configs/eval/coherent_public_n128.schema21.json"
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


def test_canonical_evaluation_attests_native_runtime_before_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(evaluate, "_validate_config", lambda _path: None)
    monkeypatch.setattr(
        evaluate.production_runtime_contract,
        "assert_native_runtime_contract",
        lambda: events.append("attest"),
    )
    monkeypatch.setattr(
        evaluate.os,
        "execv",
        lambda _executable, _argv: events.append("exec"),
    )

    evaluate.main(
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
            "1",
            "--workers",
            "1",
            "--devices",
            "cuda:0",
            "--base-seed",
            "1",
        ]
    )

    assert events == ["attest", "exec"]
