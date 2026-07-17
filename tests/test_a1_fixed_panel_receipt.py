from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import a1_aux_pair_coordinator as coordinator
from tools.fleet import a1_h100_eval_fleet as fleet


def _write_report(
    root: Path,
    *,
    arm: str,
    checkpoint_bytes: bytes,
    base_seed: int,
    pairs: int,
    panel_kind: str,
    baseline_path: Path,
    baseline_sha256: str,
    evaluation_binding: dict,
    planned_engine_identity: dict,
    invert: bool = False,
) -> tuple[Path, str]:
    checkpoint = root / f"{arm}.pt"
    checkpoint.write_bytes(checkpoint_bytes)
    checkpoint_sha = fleet._sha256(checkpoint)  # noqa: SLF001
    orientations = (
        ("candidate_red", "candidate_blue")
        if panel_kind == "internal"
        else ("candidate_first", "candidate_second")
    )
    games = []
    for pair_id in range(pairs):
        for orientation_index, orientation in enumerate(orientations):
            won = bool((pair_id + orientation_index + int(invert)) % 2)
            games.append(
                {
                    "game_seed": base_seed + pair_id,
                    "pair_id": pair_id,
                    "orientation": orientation,
                    "candidate_won": won,
                    "search_won": won,
                    "terminated": True,
                    "truncated": False,
                    "error": None,
                    "engine_divergence": False,
                }
            )
    raw_source = root / f"{arm}-{panel_kind}-raw.json"
    raw_source.write_text(json.dumps({"arm": arm, "games": games}, sort_keys=True))
    report = {
        "candidate_checkpoint": str(checkpoint.resolve(strict=True)),
        "candidate_checkpoint_sha256": checkpoint_sha,
        "baseline_checkpoint": str(baseline_path.resolve(strict=True)),
        "baseline_checkpoint_sha256": baseline_sha256,
        "base_seed": base_seed,
        "pairs_requested": pairs,
        "games_played": len(games),
        "games_truncated": 0,
        "errors": [],
        "map_kind": "BASE" if panel_kind == "internal" else "TOURNAMENT",
        "gate_config": "flywheel",
        "effective_search_config": fleet._fixed_panel_expected_search_fields(  # noqa: SLF001
            panel_kind=panel_kind,
            canonical_operator=coordinator._canonical_search_operator(),  # noqa: SLF001
        ),
        "evaluation_binding": evaluation_binding,
        "planned_engine_identity": planned_engine_identity,
        **(
            {"engine_identity": dict(planned_engine_identity)}
            if panel_kind == "internal"
            else {}
        ),
        "games": games,
        "fleet_merge": {
            "sources": [
                {
                    "path": str(raw_source.resolve(strict=True)),
                    "sha256": fleet._sha256(raw_source),  # noqa: SLF001
                }
            ]
        },
    }
    if panel_kind == "external":
        report.update(
            {
                "stratum": "neutral-harness",
                "harness": "catanatron_native_engine",
                "referee_engine": "vendored_python_catanatron",
                "baseline_bot": "catanatron_value",
                "mode": "search",
                "vps_to_win": 10,
                "max_player_trade_offers_per_turn": 0,
                "trained_value_readouts": ["scalar"],
                "engine_identity": {
                    **planned_engine_identity,
                    "native_runtime_sha256": "sha256:" + "f" * 64,
                },
            }
        )
    path = root / f"{arm}-{panel_kind}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path, checkpoint_sha


def _aux_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    panel_kind: str,
):
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"baseline")
    baseline_sha = fleet._sha256(baseline)  # noqa: SLF001
    registry = fleet.ChampionRegistry(tmp_path / "registry.json")
    search_config = {"c_scale": 0.10}
    checkpoint_ref = fleet._checkpoint_ref(baseline)  # noqa: SLF001
    identity = fleet._digest(  # noqa: SLF001
        {
            "schema_version": "a1-deployed-agent-search-config-v1",
            "checkpoint": checkpoint_ref,
            "search_config": search_config,
        }
    )
    registry.set_role(
        "generator_champion",
        baseline,
        version=5,
        provenance={
            "a1_candidate_agent_identity_sha256": identity,
            "a1_candidate_search_config": search_config,
        },
        reason="test fixed panel",
    )
    registry.save()
    evaluation_binding = fleet._evaluation_binding(  # noqa: SLF001
        candidate_parent=baseline,
        baseline=baseline,
        registry=registry,
        comparison_mode="promotion_parent",
        historical_comparison_reason=None,
        champion_c_scale=0.10,
    )
    neutral_engine_identity = fleet._engine_identity(  # noqa: SLF001
        fleet._REPO_ROOT, fleet._git_commit(fleet._REPO_ROOT)  # noqa: SLF001
    )
    planned_engine_identity = neutral_engine_identity
    if panel_kind == "internal":
        # Unit tests exercise the canonical identity comparison without
        # requiring a platform-specific compiled extension in the controller
        # environment. H100 integration tests cover the real runtime hash.
        monkeypatch.setattr(
            fleet,
            "_native_runtime_sha256",
            fleet._sealed_native_runtime_sha256,  # noqa: SLF001
        )
        planned_engine_identity = fleet._internal_engine_identity(  # noqa: SLF001
            repo_commit=neutral_engine_identity["repo_commit"],
            wheel_sha256=neutral_engine_identity["native_wheel_sha256"],
            evaluator_sha256=fleet._tool_hashes(fleet._REPO_ROOT)[  # noqa: SLF001
                "tools/gumbel_search_cross_net_h2h.py"
            ],
        )
    plan = coordinator.canonical_aux_evaluation_plan(
        baseline_checkpoint_sha256=baseline_sha
    )
    cohort = plan[f"{panel_kind}_cohort"]
    arm_reports = {}
    checkpoints = {}
    for index, arm in enumerate(coordinator.ARMS):
        report, checkpoint = _write_report(
            tmp_path,
            arm=arm,
            checkpoint_bytes=f"checkpoint-{arm}".encode(),
            base_seed=cohort["base_seed"],
            pairs=cohort["pairs"],
            panel_kind=panel_kind,
            baseline_path=baseline,
            baseline_sha256=baseline_sha,
            evaluation_binding=evaluation_binding,
            planned_engine_identity=planned_engine_identity,
            invert=bool(index),
        )
        arm_reports[arm] = report
        checkpoints[arm] = checkpoint
    return plan, arm_reports, checkpoints


def _aux_internal_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    return _aux_inputs(tmp_path, monkeypatch, panel_kind="internal")


def _mock_repool(
    monkeypatch: pytest.MonkeyPatch, reports: dict[str, Path]
) -> None:
    def replay(_sources, *, candidate: Path, champion: Path):
        assert champion.is_file()
        result = json.loads(reports[candidate.stem].read_text())
        result.pop("evaluation_binding")
        return result

    monkeypatch.setattr(fleet.evaluation_pool, "pool_internal", replay)


def _mock_neutral_repool(
    monkeypatch: pytest.MonkeyPatch, reports: dict[str, Path]
) -> None:
    def replay(_sources, *, checkpoint: Path):
        result = json.loads(reports[checkpoint.stem].read_text())
        result.pop("evaluation_binding")
        return result

    monkeypatch.setattr(fleet.evaluation_pool, "pool_neutral", replay)


def test_fixed_panel_v2_replays_raw_games_into_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, reports, checkpoints = _aux_internal_inputs(tmp_path, monkeypatch)
    _mock_repool(monkeypatch, reports)
    receipt = fleet.build_fixed_panel_receipt(
        family="AUX",
        panel_kind="internal",
        authority_id="sha256:" + "a" * 64,
        arm_reports=reports,
        arm_checkpoint_sha256=checkpoints,
        baseline_checkpoint_sha256=plan["baseline_checkpoint_sha256"],
        cohort_sha256=plan["internal_cohort_sha256"],
        search_operator_sha256=plan["search_operator_sha256"],
    )
    path = tmp_path / "fixed-panel.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    verified = fleet.verify_fixed_panel_receipt(
        path,
        family="AUX",
        panel_kind="internal",
        authority_id="sha256:" + "a" * 64,
        arms=coordinator.ARMS,
        arm_checkpoint_sha256=checkpoints,
        baseline_checkpoint_sha256=plan["baseline_checkpoint_sha256"],
        cohort_sha256=plan["internal_cohort_sha256"],
        search_operator_sha256=plan["search_operator_sha256"],
    )
    assert verified == receipt
    assert all(
        point in {0, 1000, 2000}
        for points in receipt["points_milli"].values()
        for point in points
    )
    assert len(receipt["points_milli"]["AUX0"]) == 300
    assert coordinator._load_panel_receipt(
        path,
        family="AUX",
        panel_kind="internal",
        authority_id="sha256:" + "a" * 64,
        arms=coordinator.ARMS,
        arm_checkpoint_sha256=checkpoints,
        baseline_checkpoint_sha256=plan["baseline_checkpoint_sha256"],
        cohort_sha256=plan["internal_cohort_sha256"],
        search_operator_sha256=plan["search_operator_sha256"],
        origin_tool_sha256=plan["panel_origin_tool_sha256"],
    ) == receipt


def test_fixed_external_panel_binds_native_runtime_across_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, reports, checkpoints = _aux_inputs(
        tmp_path, monkeypatch, panel_kind="external"
    )
    _mock_neutral_repool(monkeypatch, reports)
    receipt = fleet.build_fixed_panel_receipt(
        family="AUX",
        panel_kind="external",
        authority_id="sha256:" + "a" * 64,
        arm_reports=reports,
        arm_checkpoint_sha256=checkpoints,
        baseline_checkpoint_sha256=plan["baseline_checkpoint_sha256"],
        cohort_sha256=plan["external_cohort_sha256"],
        search_operator_sha256=plan["search_operator_sha256"],
    )
    assert len(receipt["points_milli"]["AUX0"]) == 500
    assert receipt["planned_engine_identity_sha256"] == receipt["source_reports"][
        "AUX0"
    ]["planned_engine_identity_sha256"]

    drifted = json.loads(reports["AUXT"].read_text())
    drifted["engine_identity"]["native_runtime_sha256"] = "sha256:" + "d" * 64
    reports["AUXT"].write_text(json.dumps(drifted, indent=2, sort_keys=True) + "\n")
    with pytest.raises(fleet.FleetError, match="planned engine differs across arms"):
        fleet.build_fixed_panel_receipt(
            family="AUX",
            panel_kind="external",
            authority_id="sha256:" + "a" * 64,
            arm_reports=reports,
            arm_checkpoint_sha256=checkpoints,
            baseline_checkpoint_sha256=plan["baseline_checkpoint_sha256"],
            cohort_sha256=plan["external_cohort_sha256"],
            search_operator_sha256=plan["search_operator_sha256"],
        )


def test_fixed_panel_refuses_resealed_forged_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, reports, checkpoints = _aux_internal_inputs(tmp_path, monkeypatch)
    _mock_repool(monkeypatch, reports)
    receipt = fleet.build_fixed_panel_receipt(
        family="AUX",
        panel_kind="internal",
        authority_id="sha256:" + "a" * 64,
        arm_reports=reports,
        arm_checkpoint_sha256=checkpoints,
        baseline_checkpoint_sha256=plan["baseline_checkpoint_sha256"],
        cohort_sha256=plan["internal_cohort_sha256"],
        search_operator_sha256=plan["search_operator_sha256"],
    )
    forged = copy.deepcopy(receipt)
    forged["points_milli"]["AUXT"][0] = 2000
    forged["points_milli_sha256"] = fleet._digest(forged["points_milli"])  # noqa: SLF001
    unsigned = dict(forged)
    unsigned.pop("state_sha256")
    forged["state_sha256"] = fleet._digest(unsigned)  # noqa: SLF001
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n")
    with pytest.raises(fleet.FleetError, match="raw-game/source replay"):
        fleet.verify_fixed_panel_receipt(
            path,
            family="AUX",
            panel_kind="internal",
            authority_id="sha256:" + "a" * 64,
            arms=coordinator.ARMS,
            arm_checkpoint_sha256=checkpoints,
            baseline_checkpoint_sha256=plan["baseline_checkpoint_sha256"],
            cohort_sha256=plan["internal_cohort_sha256"],
            search_operator_sha256=plan["search_operator_sha256"],
        )


def test_fixed_panel_refuses_source_game_mutation_after_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, reports, checkpoints = _aux_internal_inputs(tmp_path, monkeypatch)
    _mock_repool(monkeypatch, reports)
    receipt = fleet.build_fixed_panel_receipt(
        family="AUX",
        panel_kind="internal",
        authority_id="sha256:" + "a" * 64,
        arm_reports=reports,
        arm_checkpoint_sha256=checkpoints,
        baseline_checkpoint_sha256=plan["baseline_checkpoint_sha256"],
        cohort_sha256=plan["internal_cohort_sha256"],
        search_operator_sha256=plan["search_operator_sha256"],
    )
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    pooled = json.loads(reports["AUX0"].read_text())
    source = Path(pooled["fleet_merge"]["sources"][0]["path"])
    payload = json.loads(source.read_text())
    payload["games"][0]["candidate_won"] = not payload["games"][0]["candidate_won"]
    payload["games"][0]["search_won"] = payload["games"][0]["candidate_won"]
    source.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(fleet.FleetError, match="shard bytes drifted"):
        fleet.verify_fixed_panel_receipt(
            path,
            family="AUX",
            panel_kind="internal",
            authority_id="sha256:" + "a" * 64,
            arms=coordinator.ARMS,
            arm_checkpoint_sha256=checkpoints,
            baseline_checkpoint_sha256=plan["baseline_checkpoint_sha256"],
            cohort_sha256=plan["internal_cohort_sha256"],
            search_operator_sha256=plan["search_operator_sha256"],
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("search", "native_mcts_hot_loop"),
        ("engine", "planned engine differs from canonical source"),
        ("binding", "evaluation binding bytes drifted"),
    ],
)
def test_fixed_panel_refuses_resealed_operator_authority_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    error: str,
) -> None:
    plan, reports, checkpoints = _aux_internal_inputs(tmp_path, monkeypatch)
    report = json.loads(reports["AUXT"].read_text())
    if mutation == "search":
        report["effective_search_config"]["native_mcts_hot_loop"] = False
    elif mutation == "engine":
        report["planned_engine_identity"]["native_wheel_sha256"] = (
            "sha256:" + "d" * 64
        )
    else:
        report["evaluation_binding"]["baseline"]["sha256"] = "sha256:" + "e" * 64
    reports["AUXT"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _mock_repool(monkeypatch, reports)
    with pytest.raises(fleet.FleetError, match=error):
        fleet.build_fixed_panel_receipt(
            family="AUX",
            panel_kind="internal",
            authority_id="sha256:" + "a" * 64,
            arm_reports=reports,
            arm_checkpoint_sha256=checkpoints,
            baseline_checkpoint_sha256=plan["baseline_checkpoint_sha256"],
            cohort_sha256=plan["internal_cohort_sha256"],
            search_operator_sha256=plan["search_operator_sha256"],
        )
