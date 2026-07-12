#!/usr/bin/env python3
"""Plan and collect a diagnostic checkpoint x search-operator crossover.

This is deliberately *not* a promotion gate.  It uses four paired, color-swapped
H2H panels on one common BASE-map seed cohort:

* candidate vs f7 with c_scale=.03 on both roles;
* candidate vs f7 with c_scale=.10 on both roles;
* candidate(.10) vs candidate(.03); and
* f7(.10) vs f7(.03).

The first pair measures the checkpoint contrast under each matched operator.
The second pair measures the operator contrast on each frozen checkpoint.  A
candidate-native-vs-f7-native panel changes both variables at once, so its
identity is recorded separately and it is intentionally not emitted as a job.
Promotion must continue to use the role-native, parent-bound fleet evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.fleet import a1_h100_eval_fleet as fleet  # noqa: E402
from tools.prelaunch_guard import VAL_ONLY_SEED_RANGE  # noqa: E402


SCHEMA = "a1-checkpoint-operator-crossover-v1"
REPORT_SCHEMA = "a1-checkpoint-operator-crossover-report-v1"
SCALES = (0.03, 0.10)


class CrossoverError(RuntimeError):
    """The diagnostic identity or one of its reports is invalid."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _checkpoint_ref(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _git_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _panel_specs(candidate: dict[str, str], f7: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "panel_id": "checkpoint_at_cscale_003",
            "estimand": "checkpoint_effect_at_c_scale_0.03",
            "candidate": candidate,
            "baseline": f7,
            "candidate_c_scale": 0.03,
            "baseline_c_scale": 0.03,
        },
        {
            "panel_id": "checkpoint_at_cscale_010",
            "estimand": "checkpoint_effect_at_c_scale_0.10",
            "candidate": candidate,
            "baseline": f7,
            "candidate_c_scale": 0.10,
            "baseline_c_scale": 0.10,
        },
        {
            "panel_id": "operator_on_candidate",
            "estimand": "operator_effect_0.10_minus_0.03_on_candidate",
            "candidate": candidate,
            "baseline": candidate,
            "candidate_c_scale": 0.10,
            "baseline_c_scale": 0.03,
        },
        {
            "panel_id": "operator_on_f7",
            "estimand": "operator_effect_0.10_minus_0.03_on_f7",
            "candidate": f7,
            "baseline": f7,
            "candidate_c_scale": 0.10,
            "baseline_c_scale": 0.03,
        },
    ]


def _argv(
    panel: dict[str, Any],
    *,
    python: str,
    pairs: int,
    base_seed: int,
    workers: int,
    output: Path,
) -> list[str]:
    return [
        python,
        "tools/gumbel_search_cross_net_h2h.py",
        "--candidate",
        panel["candidate"]["path"],
        "--baseline",
        panel["baseline"]["path"],
        "--pairs",
        str(pairs),
        "--base-seed",
        str(base_seed),
        "--workers",
        str(workers),
        "--device",
        "cuda",
        "--map-kind",
        "BASE",
        *fleet._science_args(c_scale=None),  # noqa: SLF001 - exact production evaluator recipe.
        "--candidate-c-scale",
        str(panel["candidate_c_scale"]),
        "--baseline-c-scale",
        str(panel["baseline_c_scale"]),
        "--out",
        str(output),
    ]


def build_plan(
    *,
    candidate: Path,
    f7: Path,
    pairs: int,
    base_seed: int,
    output_dir: Path,
    workers: int = 8,
    python: str = "python3",
    candidate_native_c_scale: float = 0.10,
    f7_native_c_scale: float = 0.03,
    repo_root: Path = _REPO_ROOT,
    repo_commit: str | None = None,
    tool_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    if pairs <= 0 or workers <= 0:
        raise CrossoverError("pairs and workers must be positive")
    lo, hi = VAL_ONLY_SEED_RANGE
    if not (lo <= base_seed < base_seed + pairs <= hi):
        raise CrossoverError("seed cohort must be inside the VAL-only range")
    native_values = (candidate_native_c_scale, f7_native_c_scale)
    if any(not math.isfinite(value) or value <= 0.0 for value in native_values):
        raise CrossoverError("native c_scale values must be finite and positive")

    candidate_ref = _checkpoint_ref(candidate)
    f7_ref = _checkpoint_ref(f7)
    if candidate_ref["sha256"] == f7_ref["sha256"]:
        raise CrossoverError("candidate and f7 checkpoint bytes must differ")
    resolved_output = output_dir.expanduser().resolve()
    resolved_commit = repo_commit or _git_commit(repo_root)
    if tool_hashes is None:
        tool_hashes = {
            "tools/a1_checkpoint_operator_crossover.py": _sha256(Path(__file__)),
            "tools/gumbel_search_cross_net_h2h.py": _sha256(
                repo_root / "tools/gumbel_search_cross_net_h2h.py"
            ),
        }
    panels = []
    for spec in _panel_specs(candidate_ref, f7_ref):
        output = resolved_output / spec["panel_id"] / "report.json"
        argv = _argv(
            spec,
            python=python,
            pairs=pairs,
            base_seed=base_seed,
            workers=workers,
            output=output,
        )
        panels.append(
            {
                **spec,
                "diagnostic_only": True,
                "promotion_eligible": False,
                "map_kind": "BASE",
                "base_seed": base_seed,
                "pairs": pairs,
                "output": str(output),
                "argv": argv,
                "command_hash": _digest(argv),
            }
        )

    seed_cohort = {
        "map_kind": "BASE",
        "base_seed": base_seed,
        "end_seed": base_seed + pairs,
        "pairs": pairs,
    }
    seed_cohort["identity_sha256"] = _digest(seed_cohort)
    plan: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "causal_question": (
            "separate frozen-checkpoint strength from c_scale operator strength "
            "and their interaction"
        ),
        "repo_commit": resolved_commit,
        "tool_hashes": tool_hashes,
        "checkpoints": {"candidate": candidate_ref, "f7": f7_ref},
        "operators": {
            "cscale_003": {"c_scale": 0.03},
            "cscale_010": {"c_scale": 0.10},
        },
        "common_science_config": fleet.SCIENCE_CONFIG,
        "seed_cohort": seed_cohort,
        "panels": panels,
        "role_native_promotion_panel": {
            "included_in_crossover": False,
            "reason": (
                "candidate-native vs f7-native changes checkpoint and operator "
                "simultaneously; it is a separate parent/registry-bound promotion panel"
            ),
            "candidate": {
                **candidate_ref,
                "c_scale": float(candidate_native_c_scale),
            },
            "f7": {**f7_ref, "c_scale": float(f7_native_c_scale)},
            "required_planner": "tools/fleet/a1_h100_eval_fleet.py",
            "required_comparison_mode": "promotion_parent",
        },
    }
    plan["plan_hash"] = _digest(plan)
    return plan


def verify_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != SCHEMA:
        raise CrossoverError("unsupported crossover plan schema")
    declared = plan.get("plan_hash")
    actual = _digest({key: value for key, value in plan.items() if key != "plan_hash"})
    if declared != actual:
        raise CrossoverError("crossover plan hash does not replay")
    if (
        plan.get("diagnostic_only") is not True
        or plan.get("promotion_eligible") is not False
    ):
        raise CrossoverError("crossover must remain diagnostic-only")
    for checkpoint in plan["checkpoints"].values():
        path = Path(checkpoint["path"]).resolve(strict=True)
        if _sha256(path) != checkpoint["sha256"]:
            raise CrossoverError("checkpoint bytes drifted after crossover planning")
    cohort = plan["seed_cohort"]
    cohort_without_identity = {
        key: value for key, value in cohort.items() if key != "identity_sha256"
    }
    if cohort.get("identity_sha256") != _digest(cohort_without_identity):
        raise CrossoverError("seed cohort identity does not replay")
    if len(plan.get("panels", ())) != 4:
        raise CrossoverError("crossover must contain exactly four causal panels")
    for panel in plan["panels"]:
        if (
            panel.get("diagnostic_only") is not True
            or panel.get("promotion_eligible") is not False
            or panel.get("map_kind") != "BASE"
            or panel.get("base_seed") != cohort["base_seed"]
            or panel.get("pairs") != cohort["pairs"]
            or panel.get("command_hash") != _digest(panel.get("argv"))
        ):
            raise CrossoverError(f"panel identity drift: {panel.get('panel_id')}")
    if (
        plan.get("role_native_promotion_panel", {}).get("included_in_crossover")
        is not False
    ):
        raise CrossoverError("role-native promotion panel was mixed into the crossover")


def collect(plan: dict[str, Any]) -> dict[str, Any]:
    verify_plan(plan)
    results = []
    for panel in plan["panels"]:
        report_path = Path(panel["output"])
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise CrossoverError(
                f"cannot read {panel['panel_id']} report: {error}"
            ) from error
        expected = {
            "candidate_checkpoint_sha256": panel["candidate"]["sha256"],
            "baseline_checkpoint_sha256": panel["baseline"]["sha256"],
            "candidate_c_scale": panel["candidate_c_scale"],
            "baseline_c_scale": panel["baseline_c_scale"],
            "map_kind": "BASE",
            "base_seed": panel["base_seed"],
            "pairs_requested": panel["pairs"],
        }
        if any(report.get(key) != value for key, value in expected.items()):
            raise CrossoverError(f"{panel['panel_id']} report identity mismatch")
        if report.get("errors") or report.get("complete_pairs") != panel["pairs"]:
            raise CrossoverError(f"{panel['panel_id']} report is incomplete")
        results.append(
            {
                "panel_id": panel["panel_id"],
                "estimand": panel["estimand"],
                "candidate_win_rate": report.get("candidate_win_rate"),
                "complete_pairs": report["complete_pairs"],
                "pair_diagnostics": report["pair_diagnostics"],
                "report": {"path": str(report_path), "sha256": _sha256(report_path)},
            }
        )
    return {
        "schema_version": REPORT_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "plan_hash": plan["plan_hash"],
        "seed_cohort": plan["seed_cohort"],
        "results": results,
        "interpretation_guard": (
            "No crossover result promotes a checkpoint. Use the separate role-native "
            "parent/registry-bound panel for promotion."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--candidate", type=Path, required=True)
    plan_parser.add_argument("--f7", type=Path, required=True)
    plan_parser.add_argument("--pairs", type=int, required=True)
    plan_parser.add_argument("--base-seed", type=int, required=True)
    plan_parser.add_argument("--output-dir", type=Path, required=True)
    plan_parser.add_argument("--workers", type=int, default=8)
    plan_parser.add_argument("--python", default="python3")
    plan_parser.add_argument("--candidate-native-c-scale", type=float, default=0.10)
    plan_parser.add_argument("--f7-native-c-scale", type=float, default=0.03)
    plan_parser.add_argument("--out", type=Path, required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--plan", type=Path, required=True)
    collect_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "plan":
        value = build_plan(
            candidate=args.candidate,
            f7=args.f7,
            pairs=args.pairs,
            base_seed=args.base_seed,
            output_dir=args.output_dir,
            workers=args.workers,
            python=args.python,
            candidate_native_c_scale=args.candidate_native_c_scale,
            f7_native_c_scale=args.f7_native_c_scale,
        )
    else:
        value = json.loads(args.plan.read_text(encoding="utf-8"))
        value = collect(value)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
