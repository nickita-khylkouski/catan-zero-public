from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import a1_b200_lr_dose_campaign as campaign
from tools import a1_build_post_wave_composite as composite_builder
from tools import a1_one_dose_train as one_dose
from tools import train_bc


def _sha(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _write_native_eval(
    root: Path,
    *,
    arm: str,
    candidate_sha: str,
    f7: tuple[float, float, float],
    v5: tuple[float, float, float],
    baseline_shas: dict[str, str],
) -> Path:
    rows = {}
    for role, (mu, lower, upper) in {"f7": f7, "v5": v5}.items():
        report = root / f"{arm}-vs-{role}.pooled.json"
        payload = {
            "candidate_checkpoint_sha256": candidate_sha,
            "baseline_checkpoint_sha256": baseline_shas[role],
            "paired_score_regularized_mu": mu,
            "paired_score_regularized_95ci": [lower, upper],
            "pairs_requested": 100,
            "complete_pairs": 100,
            "games_requested": 200,
            "games_played": 200,
            "games_with_winner": 200,
            "games_truncated": 0,
            "pairs_truncated_excluded": 0,
            "errors": [],
        }
        report.write_text(json.dumps(payload), encoding="utf-8")
        rows[f"arm-vs-{role}"] = {
            "paired_score_regularized_mu": mu,
            "paired_score_regularized_95ci": [lower, upper],
            "report": str(report),
        }
    summary = root / f"{arm}.native-eval-summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": campaign.NATIVE_EVAL_SUMMARY_SCHEMA,
                "arms": {arm: rows},
            }
        ),
        encoding="utf-8",
    )
    return summary


def _write_matrix_authority(
    root: Path,
    *,
    pairs: int = campaign.MIN_SELECTION_PAIRS,
    f7_sha: str = "sha256:" + "7" * 64,
) -> tuple[Path, dict, dict[str, Path]]:
    campaign_path = root / "campaign.json"
    campaign_payload = {
        "campaign_sha256": "sha256:" + "c" * 64,
        "lineage_contract": {"expected_parent_sha256": f7_sha},
    }
    v5 = root / "v5.pt"
    v5.write_bytes(b"v5")
    v5_sha = campaign._file_sha256(v5)  # noqa: SLF001
    science_contract = root / "science.json"
    science_contract.write_text("{}\n", encoding="utf-8")
    operator_selection = {
        "status": "adopted_teacher_campaign",
        "selected_operator": "base_n128_d6",
    }
    operator_search = {"n_full": 128}
    campaign_path.write_text(json.dumps(campaign_payload), encoding="utf-8")
    rows = [
        {
            "arm": arm,
            "baseline": role,
            "candidate_sha256": "sha256:" + arm.lower() * 64,
            "baseline_sha256": f7_sha if role == "f7" else v5_sha,
        }
        for arm in campaign.ARMS
        for role in campaign.EVALUATION_BASELINE_ROLES
    ]
    matrix = {
        "schema_version": campaign.EVALUATION_MATRIX_SCHEMA,
        "training_campaign": {
            "path": str(campaign_path),
            "file_sha256": campaign._file_sha256(campaign_path),  # noqa: SLF001
            "campaign_sha256": campaign_payload["campaign_sha256"],
        },
        "internal_claim": {"base_seed": 1000, "pairs": pairs},
        "matchups": rows,
        "registry": str(root / "registry.json"),
        "operator_selection": operator_selection,
        "selected_operator": operator_selection["selected_operator"],
        "operator_search": operator_search,
        "science_contract_sha256": campaign._file_sha256(  # noqa: SLF001
            science_contract
        ),
    }
    (root / "registry.json").write_text("{}\n", encoding="utf-8")
    matrix["state_sha256"] = campaign._value_sha256(matrix)  # noqa: SLF001
    (root / "matrix.json").write_text(json.dumps(matrix), encoding="utf-8")
    summary = root / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": campaign.NATIVE_EVAL_SUMMARY_SCHEMA,
                "matrix_state_sha256": matrix["state_sha256"],
            }
        ),
        encoding="utf-8",
    )
    campaign_payload["_test_authority"] = {
        "v5": str(v5),
        "operator_selection": operator_selection,
        "operator_search": operator_search,
        "science_contract": str(science_contract),
    }
    return (
        campaign_path,
        campaign_payload,
        {arm: summary for arm in campaign.ARMS},
    )


def test_lr_selector_matrix_authority_binds_f7_v5_and_minimum_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_path, payload, evaluations = _write_matrix_authority(tmp_path)
    authority_fixture = payload.pop("_test_authority")

    class _Registry:
        def get_role(self, role: str):
            assert role == "generator_champion"
            return type("Pointer", (), {"checkpoint_path": authority_fixture["v5"]})()

    monkeypatch.setattr(campaign.ChampionRegistry, "load", lambda _path: _Registry())
    monkeypatch.setattr(
        campaign.current_science,
        "load",
        lambda: {"operator_selection": authority_fixture["operator_selection"]},
    )
    monkeypatch.setattr(
        campaign.current_science,
        "search",
        lambda: authority_fixture["operator_search"],
    )
    monkeypatch.setattr(
        campaign.current_science,
        "CONTRACT_PATH",
        Path(authority_fixture["science_contract"]),
    )

    authority = campaign._evaluation_matrix_authority(  # noqa: SLF001
        campaign_path=campaign_path,
        campaign=payload,
        evaluation_paths=evaluations,
    )

    assert authority["baseline_sha_by_role"] == {
        "f7": payload["lineage_contract"]["expected_parent_sha256"],
        "v5": campaign._file_sha256(Path(authority_fixture["v5"])),  # noqa: SLF001
    }
    assert authority["claim"]["pairs"] == campaign.MIN_SELECTION_PAIRS

    too_small = tmp_path / "too-small"
    too_small.mkdir()
    campaign_path, payload, evaluations = _write_matrix_authority(
        too_small, pairs=campaign.MIN_SELECTION_PAIRS - 1
    )
    payload.pop("_test_authority")
    with pytest.raises(campaign.CampaignError, match="at least"):
        campaign._evaluation_matrix_authority(  # noqa: SLF001
            campaign_path=campaign_path,
            campaign=payload,
            evaluation_paths=evaluations,
        )


def test_robust_selector_rejects_f7_only_ranking_reversal(tmp_path: Path) -> None:
    baseline_shas = {
        "f7": "sha256:" + "7" * 64,
        "v5": "sha256:" + "5" * 64,
    }
    candidate_shas = {
        arm: "sha256:" + digit * 64
        for arm, digit in zip(campaign.ARMS, "abcd", strict=True)
    }
    # B is the tempting f7-only winner but collapses against v5. C has the
    # strongest worst-baseline lower bound and must win the robust objective.
    scores = {
        "A": ((0.55, 0.50, 0.60), (0.36, 0.34, 0.38)),
        "B": ((0.65, 0.60, 0.70), (0.26, 0.24, 0.28)),
        "C": ((0.68, 0.66, 0.70), (0.64, 0.62, 0.66)),
        "D": ((0.54, 0.49, 0.59), (0.46, 0.44, 0.48)),
    }
    evaluations = {
        arm: _write_native_eval(
            tmp_path,
            arm=arm,
            candidate_sha=candidate_shas[arm],
            f7=scores[arm][0],
            v5=scores[arm][1],
            baseline_shas=baseline_shas,
        )
        for arm in campaign.ARMS
    }
    receipts = {
        arm: {"checkpoint_sha256": candidate_shas[arm]}
        for arm in campaign.ARMS
    }

    winner, ranking, evidence = campaign._rank_authenticated_evaluations(
        receipt_records=receipts,
        evaluation_paths=evaluations,
    )

    assert winner == "C"
    assert [row["arm"] for row in ranking] == ["C", "D", "A", "B"]
    assert evidence["baseline_checkpoint_sha256_by_role"] == baseline_shas
    assert evidence["arms"]["B"]["robust_worst_baseline_95ci_lower"] == 0.24


def test_robust_selector_accepts_real_flat_matchup_summary_shape(
    tmp_path: Path,
) -> None:
    baseline_shas = {
        "f7": "sha256:" + "7" * 64,
        "v5": "sha256:" + "5" * 64,
    }
    candidate_shas = {
        arm: "sha256:" + digit * 64
        for arm, digit in zip(campaign.ARMS, "abcd", strict=True)
    }
    rows = []
    for arm_index, arm in enumerate(campaign.ARMS):
        for role, base_mu in (("f7", 0.55), ("v5", 0.54)):
            mu = base_mu - 0.02 * arm_index
            report = tmp_path / f"{arm}-{role}.pooled.json"
            report.write_text(
                json.dumps(
                    {
                        "candidate_checkpoint_sha256": candidate_shas[arm],
                        "baseline_checkpoint_sha256": baseline_shas[role],
                        "paired_score_regularized_mu": mu,
                        "paired_score_regularized_95ci": [mu - 0.005, mu + 0.005],
                        "pairs_requested": 128,
                        "complete_pairs": 128,
                        "games_played": 256,
                        "games_with_winner": 256,
                        "games_truncated": 0,
                        "pairs_truncated_excluded": 0,
                        "errors": [],
                    }
                ),
                encoding="utf-8",
            )
            rows.append(
                {
                    "matchup": f"{arm.lower()}-vs-{role}",
                    "paired_score_regularized_mu": mu,
                    "paired_score_regularized_95ci": [mu - 0.005, mu + 0.005],
                    "report": str(report),
                }
            )
    summary = tmp_path / "r5-results-summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": campaign.NATIVE_EVAL_SUMMARY_SCHEMA,
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )

    winner, ranking, evidence = campaign._rank_authenticated_evaluations(
        receipt_records={
            arm: {"checkpoint_sha256": candidate_shas[arm]}
            for arm in campaign.ARMS
        },
        evaluation_paths={arm: summary for arm in campaign.ARMS},
    )

    assert winner == "A"
    assert [row["arm"] for row in ranking] == list(campaign.ARMS)
    assert set(evidence["arms"]["A"]["comparisons"]) == {"f7", "v5"}


def test_robust_selector_rejects_incomplete_or_mismatched_evidence(
    tmp_path: Path,
) -> None:
    baseline_shas = {
        "f7": "sha256:" + "7" * 64,
        "v5": "sha256:" + "5" * 64,
    }
    candidate_shas = {
        arm: "sha256:" + digit * 64
        for arm, digit in zip(campaign.ARMS, "abcd", strict=True)
    }
    evaluations = {
        arm: _write_native_eval(
            tmp_path,
            arm=arm,
            candidate_sha=candidate_shas[arm],
            f7=(0.55, 0.50, 0.60),
            v5=(0.54, 0.49, 0.59),
            baseline_shas=baseline_shas,
        )
        for arm in campaign.ARMS
    }
    receipts = {
        arm: {"checkpoint_sha256": candidate_shas[arm]}
        for arm in campaign.ARMS
    }
    bad_report = tmp_path / "A-vs-v5.pooled.json"
    payload = json.loads(bad_report.read_text(encoding="utf-8"))
    payload["games_truncated"] = 1
    payload["games_with_winner"] = 199
    bad_report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(campaign.CampaignError, match="incomplete or truncated"):
        campaign._rank_authenticated_evaluations(
            receipt_records=receipts,
            evaluation_paths=evaluations,
        )


def _paired_metric_report() -> dict:
    games = [
        {
            "pair_id": 0,
            "game_seed": 100,
            "orientation": "candidate_red",
            "candidate_won": True,
            "search_won": True,
        },
        {
            "pair_id": 0,
            "game_seed": 100,
            "orientation": "candidate_blue",
            "candidate_won": False,
            "search_won": False,
        },
        {
            "pair_id": 1,
            "game_seed": 101,
            "orientation": "candidate_red",
            "candidate_won": True,
            "search_won": True,
        },
        {
            "pair_id": 1,
            "game_seed": 101,
            "orientation": "candidate_blue",
            "candidate_won": True,
            "search_won": True,
        },
    ]
    return {
        "base_seed": 100,
        "pairs_requested": 2,
        "complete_pairs": 2,
        "games_played": 4,
        "games_with_winner": 4,
        "candidate_wins": 3,
        "baseline_wins": 1,
        "pair_diagnostics": {
            "ww_pairs": 1,
            "ll_pairs": 0,
            "split_pairs": 1,
            "incomplete_pairs": 0,
        },
        "games": games,
    }


def test_paired_metric_replay_accepts_one_seed_and_both_seats_per_pair() -> None:
    replay = campaign._paired_score_metrics(  # noqa: SLF001
        _paired_metric_report(), where="valid panel"
    )

    assert replay["pair_counts"] == {"ww": 1, "split": 1, "ll": 0}
    assert replay["game_keys"] == [
        (100, "candidate_blue"),
        (100, "candidate_red"),
        (101, "candidate_blue"),
        (101, "candidate_red"),
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report["games"][1].__setitem__("game_seed", 101),
            "base_seed schedule",
        ),
        (
            lambda report: report["games"][1].__setitem__(
                "orientation", "candidate_red"
            ),
            "both seat orientations",
        ),
        (
            lambda report: [
                game.__setitem__("game_seed", 100)
                for game in report["games"][2:]
            ],
            "base_seed schedule",
        ),
        (
            lambda report: [
                game.__setitem__("game_seed", 102)
                for game in report["games"][2:]
            ],
            "base_seed schedule",
        ),
        (
            lambda report: [
                game.__setitem__("pair_id", 2)
                for game in report["games"][2:]
            ],
            "pair_id schedule",
        ),
        (
            lambda report: report["games"][0].__setitem__("search_won", False),
            "candidate_won/search_won alias drift",
        ),
        (
            lambda report: report.__setitem__("candidate_wins", 2),
            "headline candidate_wins",
        ),
    ],
)
def test_paired_metric_replay_rejects_malformed_pairing_and_headlines(
    mutate,
    message: str,
) -> None:
    report = _paired_metric_report()
    mutate(report)

    with pytest.raises(campaign.CampaignError, match=message):
        campaign._paired_score_metrics(report, where="malformed panel")  # noqa: SLF001


def test_winner_argument_is_only_an_assertion() -> None:
    campaign._verify_winner_assertion(None, "C")
    campaign._verify_winner_assertion("C", "C")
    with pytest.raises(campaign.CampaignError, match="disagrees"):
        campaign._verify_winner_assertion("B", "C")


def test_robust_selector_refuses_unresolved_top_tie(tmp_path: Path) -> None:
    baseline_shas = {
        "f7": "sha256:" + "7" * 64,
        "v5": "sha256:" + "5" * 64,
    }
    candidate_shas = {
        arm: "sha256:" + digit * 64
        for arm, digit in zip(campaign.ARMS, "abcd", strict=True)
    }
    evaluations = {}
    for arm in campaign.ARMS:
        tied_top = arm in {"B", "C"}
        evaluations[arm] = _write_native_eval(
            tmp_path,
            arm=arm,
            candidate_sha=candidate_shas[arm],
            f7=(0.60, 0.55, 0.65) if tied_top else (0.50, 0.45, 0.55),
            v5=(0.58, 0.53, 0.63) if tied_top else (0.49, 0.44, 0.54),
            baseline_shas=baseline_shas,
        )
    with pytest.raises(campaign.CampaignError, match="statistically unresolved"):
        campaign._rank_authenticated_evaluations(
            receipt_records={
                arm: {"checkpoint_sha256": candidate_shas[arm]}
                for arm in campaign.ARMS
            },
            evaluation_paths=evaluations,
        )


def test_campaign_seals_independent_parent_arms_and_policy_active_target(
    tmp_path: Path,
) -> None:
    files = {}
    for name in ("lock", "composite", "upgrade", "canary"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"name": name}), encoding="utf-8")
        files[name] = path
    data = tmp_path / "memmap_composite.json"
    data.write_text(json.dumps({"schema_version": "memmap_composite_v2"}))
    executable = tmp_path / "python"
    executable.write_bytes(b"python")
    executable.chmod(0o755)
    args = argparse.Namespace(
        lock=files["lock"],
        data=data,
        composite_build_receipt=files["composite"],
        architecture_upgrade_receipt=files["upgrade"],
        ddp_canary_receipt=files["canary"],
        expected_parent_sha256="ab" * 32,
        reviewed_code_tree_sha256="cd" * 32,
        reviewed_lock_file_sha256=_sha(files["lock"]),
        python=executable,
        output_root=tmp_path / "out",
        observed_base_policy_active_fraction=50_875 / 524_288,
        target_policy_active_rows=524_288,
        policy_aux_active_batch_size=0,
    )

    sealed = campaign._plan(args)

    assert sealed["lineage_contract"]["every_arm_restarts_from_expected_parent"]
    assert sealed["lineage_contract"]["candidate_chaining_forbidden"]
    assert sealed["policy_active_dose"]["policy_aux_active_batch_size"] == 463
    assert set(sealed["arms"]) == {"A", "B", "C", "D"}
    for arm, expected in campaign.ARMS.items():
        command = sealed["commands"][arm]
        override = json.loads(command[command.index("--recipe-overrides-json") + 1])
        assert override["lr"] == expected["lr"]
        assert override["lr_warmup_steps"] == expected["lr_warmup_steps"]
        assert override["max_steps"] == 128
        assert override["policy_aux_active_batch_size"] == 463
        assert override["per_game_policy_surprise_weighting"] is False
        assert override["public_card_lr_mult"] == 1.0
        assert (
            override["forced_row_value_action_type_weights"]
            == "END_TURN=0.1,ROLL=1.0"
        )


class _Composite(dict):
    component_ids = ("current_producer", "historical_replay")

    def component_indices_for_rows(self, rows):
        return np.asarray([0, 0, 1, 1], dtype=np.int64)[np.asarray(rows)]


def test_training_strata_reports_realized_policy_active_dose() -> None:
    data = _Composite(
        legal_action_ids=np.asarray(
            [
                [1, -1, -1, -1],
                [1, 2, 3, -1],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
            ],
            dtype=np.int16,
        ),
        used_full_search=np.asarray([False, True, True, False]),
        simulations_used=np.asarray([0, 128, 256, 64]),
        phase=np.asarray(["opening", "opening", "main", "main"]),
        decision_class=np.asarray(
            ["automatic", "normal_choice", "mandatory_choice", "normal_choice"]
        ),
    )
    dose = train_bc._training_strata_dose_for_batch(
        data,
        np.arange(4, dtype=np.int64),
        policy_weights=np.asarray([0.0, 2.0, 1.0, 0.0]),
        value_weights=np.ones(4, dtype=np.float32),
        value_active_mask=np.asarray([True, True, False, True]),
    )
    report = train_bc._nest_training_strata_dose(
        train_bc._flatten_training_strata_dose(dose)
    )

    assert report["total_row_draws"] == 4
    assert report["policy_active_row_draws"] == 2
    assert report["policy_active_fraction"] == 0.5
    assert report["dimensions"]["fresh_vs_replay"]["fresh"][
        "policy_active_rows"
    ] == 1
    assert report["dimensions"]["simulation_budget"]["256_plus"][
        "sampled_rows"
    ] == 1
    assert report["dimensions"]["decision_class"]["mandatory_choice"][
        "policy_active_rows"
    ] == 1


def test_training_strata_reports_rare_action_exposure_and_explicit_zeroes() -> None:
    legal_tokens = np.zeros(
        (2, 2, 2 + len(train_bc.ACTION_TYPES)), dtype=np.float16
    )
    yop = train_bc.ACTION_TYPES.index("PLAY_YEAR_OF_PLENTY")
    monopoly = train_bc.ACTION_TYPES.index("PLAY_MONOPOLY")
    legal_tokens[0, 1, 2 + yop] = 1.0
    legal_tokens[1, 0, 2 + monopoly] = 1.0
    data = {
        "legal_action_ids": np.asarray([[100, 101], [102, 103]], dtype=np.int16),
        "legal_action_tokens": legal_tokens,
        "action_taken": np.asarray([101, 102], dtype=np.int16),
        "phase": np.asarray(["main", "main"]),
    }

    dose = train_bc._training_strata_dose_for_batch(
        data,
        np.arange(2, dtype=np.int64),
        policy_weights=np.asarray([1.0, 2.0], dtype=np.float32),
        value_weights=np.ones(2, dtype=np.float32),
        value_active_mask=np.asarray([True, True]),
    )
    action_types = dose["action_type"]

    assert action_types["PLAY_YEAR_OF_PLENTY"]["policy_active_rows"] == 1
    assert action_types["PLAY_MONOPOLY"]["policy_weight_sum"] == pytest.approx(2.0)
    assert action_types["BUILD_ROAD"]["sampled_rows"] == 0
    assert set(train_bc.ACTION_TYPES).issubset(action_types)


def test_normalization_preserves_decision_class_and_labels_legacy(tmp_path: Path) -> None:
    base = {
        "obs": np.zeros((2, 806), dtype=np.float16),
        "legal_action_ids": np.asarray([[1, 2], [1, -1]], dtype=np.int16),
        "legal_action_context": np.zeros((2, 2, 1), dtype=np.float16),
        "action_taken": np.asarray([1, 1], dtype=np.int16),
    }
    legacy = train_bc._normalize_teacher_shard(base, tmp_path / "legacy.npz")
    current = train_bc._normalize_teacher_shard(
        {**base, "decision_class": np.asarray(["normal_choice", "automatic"])},
        tmp_path / "current.npz",
    )
    assert legacy["decision_class"].tolist() == ["legacy_unknown"] * 2
    assert current["decision_class"].tolist() == ["normal_choice", "automatic"]


def test_completed_campaign_report_requires_real_policy_and_module_dose(
    tmp_path: Path,
) -> None:
    expected_parent = "sha256:" + "7" * 64
    learner_parent = {
        "schema_version": "a1-learner-lineage-parent-v1",
        "role": "diagnostic_recent_history",
        "checkpoint": {"path": "/parent/f7.pt", "sha256": expected_parent},
        "diagnostic_only": True,
        "promotion_eligible": False,
    }
    dimensions = {
        name: {
            "all": {
                "sampled_rows": 524_288,
                "policy_active_rows": 50_875,
                "policy_weight_sum": 50_875.0,
                "value_active_rows": 524_288,
                "value_weight_sum": 524_288.0,
            }
        }
        for name in (
            "action_type",
            "draw_stream",
            "full_vs_fast",
            "simulation_budget",
            "decision_class",
            "legal_width",
            "phase",
            "fresh_vs_replay",
        )
    }
    report = tmp_path / "train.report.json"
    report.write_text(
        json.dumps(
            {
                "steps_completed": 128,
                "optimizer_restored": False,
                "a1_learner_lineage_parent": learner_parent,
                "a1_lineage_dose": {
                    "declared_producer_sha256": expected_parent,
                },
                "a1_one_dose_input_binding": {
                    "learner_lineage_parent": learner_parent,
                },
                "policy_aux_active_rows": 0,
                "policy_total_active_rows": 50_875,
                "training_strata_dose": {
                    "schema_version": "training-strata-dose-v1",
                    "base_row_draws": 524_288,
                    "policy_aux_row_draws": 0,
                    "policy_active_row_draws": 50_875,
                    "policy_active_fraction": 50_875 / 524_288,
                    "dimensions": dimensions,
                },
                "module_optimizer_observability": {
                    "schema_version": "module-optimizer-observability-v1",
                    "observed_steps": 8,
                    "modules": {"blocks": {"mean_pre_clip_grad_norm": 1.0}},
                },
                "per_game_policy_surprise_weighting": False,
                "public_card_lr_mult": 1.0,
                "forced_row_value_action_type_weights": {
                    "END_TURN": 0.1,
                    "ROLL": 1.0,
                },
                "policy_aux_sampling": {
                    "schema_version": "train-policy-aux-sampling-v1",
                    "enabled": True,
                    "base_measure": "authenticated_component",
                    "exact_per_game_policy_surprise_weighting": False,
                    "preconditioning_weights": {
                        "content_sha256": "sha256:" + "1" * 64,
                    },
                    "final_sampling_weights": {
                        "content_sha256": "sha256:" + "2" * 64,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    sealed = {
        "reporting_contract": {"required_dimensions": list(dimensions)},
        "lineage_contract": {"expected_parent_sha256": expected_parent},
    }
    summary = campaign._verify_training_report(
        sealed,
        arm="A",
        max_steps=128,
        one_dose_plan={"report": str(report)},
    )
    assert summary["policy_active_row_draws"] == 50_875
    assert summary["module_observed_steps"] == 8


def test_lr_dose_profile_is_carried_by_and_replayed_from_descriptor(
    tmp_path: Path, monkeypatch
) -> None:
    base = {
        "schema_version": "memmap_composite_v2",
        "diagnostic_only": False,
        "promotion_eligible": True,
        "learner_recipe_overrides": dict(
            composite_builder.LEARNER_RECIPE_OVERRIDES
        ),
        "learner_recipe_overrides_sha256": one_dose._value_sha256(
            composite_builder.LEARNER_RECIPE_OVERRIDES
        ),
        "policy_distillation_component_ids": list(
            one_dose.FRESH_POLICY_DISTILLATION_COMPONENT_IDS
        ),
        "value_training_component_ids": list(
            one_dose.FRESH_VALUE_TRAINING_COMPONENT_IDS
        ),
    }
    base_path = tmp_path / "production.json"
    base_path.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n")
    effective = {
        **composite_builder.LEARNER_RECIPE_OVERRIDES,
        "epochs": 1,
        "max_steps": 128,
        "lr": 6e-5,
        "lr_warmup_steps": 16,
        "lr_schedule": "flat",
        "policy_aux_active_batch_size": 463,
        "policy_aux_loss_weight": 0.25,
    }
    verified = {
        "data_kind": "production_composite_v2",
        "data_path": base_path.resolve(),
        "corpus_meta_file_sha256": one_dose._file_sha256(base_path),
        "descriptor_fingerprint": one_dose._value_sha256(base),
        "recipe": effective,
        "contract_sha256": "sha256:" + "1" * 64,
        "function_preserving_upgrade": None,
        "learner_ablation": {
            "reporting_contract": {"diagnostic_dose_curve": True}
        },
    }
    derived_path = tmp_path / "arm-c.training-descriptor.json"
    derived = one_dose.bind_diagnostic_training_descriptor(
        verified, descriptor_path=derived_path
    )
    one_dose._materialize_diagnostic_training_descriptor(derived)
    payload = json.loads(derived_path.read_text())
    assert payload["learner_recipe_overrides"]["lr"] == 6e-5
    assert payload["learner_recipe_overrides"]["policy_aux_active_batch_size"] == 463

    monkeypatch.setattr(
        train_bc,
        "_preflight_memmap_composite_descriptor",
        lambda _path: {
            "diagnostic_only": False,
            "promotion_eligible": True,
            "descriptor_file_sha256": one_dose._file_sha256(base_path),
            "descriptor_fingerprint": one_dose._value_sha256(base),
        },
    )
    replayed = train_bc._preflight_flywheel_diagnostic_derivative(
        derived_path.resolve(), payload
    )
    assert replayed is not None
    assert replayed["learner_recipe_overrides"]["lr"] == 6e-5
    assert replayed["learner_recipe_overrides"]["max_steps"] == 128


def test_diagnostic_parent_may_be_exact_sealed_recent_history(
    tmp_path: Path, monkeypatch
) -> None:
    receipt = tmp_path / "upgrade.json"
    receipt.write_text("{}")
    initializer = tmp_path / "init.pt"
    initializer.write_bytes(b"zero-output upgraded f7")
    f7 = {"path": str(tmp_path / "f7.pt"), "sha256": "sha256:" + "7" * 64}
    upgrade = {
        "module": one_dose.architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_FEATURES_V2,
        "source": f7,
        "upgraded_initializer": {
            "path": str(initializer),
            "sha256": one_dose._file_sha256(initializer),
        },
        "receipt_sha256": "sha256:" + "8" * 64,
        "receipt": {"path": str(receipt), "sha256": one_dose._file_sha256(receipt)},
    }
    monkeypatch.setattr(
        one_dose.architecture_upgrade, "verify_receipt", lambda _path: upgrade
    )
    verified = {
        "producer": {
            "path": str(tmp_path / "v5.pt"),
            "sha256": "sha256:" + "5" * 64,
        },
        "contract_sha256": "sha256:" + "1" * 64,
        "data_kind": "production_composite_v2",
        "category_semantics": {"recent_history": {"checkpoint": f7}},
        "category_semantics_sha256": "sha256:" + "2" * 64,
    }
    bound = one_dose.bind_function_preserving_upgrade(
        verified,
        receipt,
        allow_diagnostic_recent_history_source=True,
    )
    assert bound["diagnostic_comparison_source"]["source"] == f7
    assert bound["diagnostic_comparison_source"]["promotion_eligible"] is False
    learner_parent = bound["learner_lineage_parent"]
    assert learner_parent["checkpoint"] == f7
    assert learner_parent["corpus_producer"] == verified["producer"]
    assert learner_parent["diagnostic_only"] is True
    assert learner_parent["promotion_eligible"] is False
    assert one_dose._learner_lineage_parent_sha256(bound) == f7["sha256"]

    missing_parent = dict(bound)
    del missing_parent["learner_lineage_parent"]
    with pytest.raises(
        one_dose.ExecutorError,
        match="explicit learner lineage parent",
    ):
        one_dose._learner_lineage_parent_sha256(missing_parent)
