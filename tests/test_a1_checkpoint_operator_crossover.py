from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import a1_checkpoint_operator_crossover as crossover


def _checkpoint(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _plan(tmp_path: Path) -> dict:
    candidate = _checkpoint(tmp_path / "candidate.pt", b"candidate")
    f7 = _checkpoint(tmp_path / "f7.pt", b"f7")
    return crossover.build_plan(
        candidate=candidate,
        f7=f7,
        pairs=32,
        base_seed=6_195_000_000,
        output_dir=tmp_path / "reports",
        workers=4,
        repo_commit="a" * 40,
        tool_hashes={"evaluator": "sha256:" + "1" * 64},
    )


def test_plan_is_exact_four_panel_common_seed_causal_crossover(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    crossover.verify_plan(plan)

    assert plan["diagnostic_only"] is True
    assert plan["promotion_eligible"] is False
    assert len(plan["panels"]) == 4
    assert {panel["estimand"] for panel in plan["panels"]} == {
        "checkpoint_effect_at_c_scale_0.03",
        "checkpoint_effect_at_c_scale_0.10",
        "operator_effect_0.10_minus_0.03_on_candidate",
        "operator_effect_0.10_minus_0.03_on_f7",
    }
    assert {
        (panel["base_seed"], panel["pairs"], panel["map_kind"])
        for panel in plan["panels"]
    } == {(6_195_000_000, 32, "BASE")}
    assert {
        (panel["candidate_c_scale"], panel["baseline_c_scale"])
        for panel in plan["panels"]
    } == {(0.03, 0.03), (0.1, 0.1), (0.1, 0.03)}
    assert plan["role_native_promotion_panel"] == {
        "included_in_crossover": False,
        "reason": (
            "candidate-native vs f7-native changes checkpoint and operator "
            "simultaneously; it is a separate parent/registry-bound promotion panel"
        ),
        "candidate": {**plan["checkpoints"]["candidate"], "c_scale": 0.1},
        "f7": {**plan["checkpoints"]["f7"], "c_scale": 0.03},
        "required_planner": "tools/fleet/a1_h100_eval_fleet.py",
        "required_comparison_mode": "promotion_parent",
    }


def test_every_panel_uses_exact_current_eval_recipe_and_role_scales(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    for panel in plan["panels"]:
        argv = panel["argv"]
        assert argv[argv.index("--n-full") + 1] == "128"
        assert argv[argv.index("--map-kind") + 1] == "BASE"
        assert argv[argv.index("--candidate-c-scale") + 1] == str(
            panel["candidate_c_scale"]
        )
        assert argv[argv.index("--baseline-c-scale") + 1] == str(
            panel["baseline_c_scale"]
        )
        assert argv.count("--c-scale") == 0
        for flag in (
            "--information-set-search",
            "--symmetry-averaged-eval",
            "--evaluator-rust-featurize",
            "--native-mcts-hot-loop",
        ):
            assert argv.count(flag) == 1


def test_plan_rejects_checkpoint_drift_and_promotion_relabel(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    Path(plan["checkpoints"]["candidate"]["path"]).write_bytes(b"drift")
    with pytest.raises(crossover.CrossoverError, match="checkpoint bytes drifted"):
        crossover.verify_plan(plan)

    plan = _plan(tmp_path)
    plan["promotion_eligible"] = True
    plan["plan_hash"] = crossover._digest(  # noqa: SLF001
        {key: value for key, value in plan.items() if key != "plan_hash"}
    )
    with pytest.raises(crossover.CrossoverError, match="diagnostic-only"):
        crossover.verify_plan(plan)


def test_collect_refuses_report_from_wrong_operator_or_seed(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    for panel in plan["panels"]:
        report = {
            "candidate_checkpoint_sha256": panel["candidate"]["sha256"],
            "baseline_checkpoint_sha256": panel["baseline"]["sha256"],
            "candidate_c_scale": panel["candidate_c_scale"],
            "baseline_c_scale": panel["baseline_c_scale"],
            "map_kind": "BASE",
            "base_seed": panel["base_seed"],
            "pairs_requested": panel["pairs"],
            "complete_pairs": panel["pairs"],
            "candidate_win_rate": 0.5,
            "pair_diagnostics": {
                "ww_pairs": 8,
                "ll_pairs": 8,
                "split_pairs": 16,
                "incomplete_pairs": 0,
            },
            "errors": [],
        }
        path = Path(panel["output"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report), encoding="utf-8")

    result = crossover.collect(plan)
    assert result["diagnostic_only"] is True
    assert len(result["results"]) == 4

    bad = Path(plan["panels"][0]["output"])
    report = json.loads(bad.read_text(encoding="utf-8"))
    report["base_seed"] += 1
    bad.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(crossover.CrossoverError, match="report identity mismatch"):
        crossover.collect(plan)
