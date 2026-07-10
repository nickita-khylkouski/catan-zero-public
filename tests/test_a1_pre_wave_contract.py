from __future__ import annotations

import json
import hashlib
import os
import stat
from collections import Counter
from pathlib import Path
from argparse import Namespace

import numpy as np
import pytest
import torch

from catan_zero.rl.entity_token_policy import EntityGraphConfig
from tools import a1_pre_wave_contract as contract
from tools import generate_gumbel_selfplay_data as generator
from tools import legacy_scalar_readout_attestation as legacy_scalar


TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "experiments"
    / "a1_pre_wave_contract.template.json"
)


@pytest.fixture(autouse=True)
def _replay_unit_envelopes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The producers have their own deep tests; contract fixtures stay tiny."""

    monkeypatch.setattr(
        contract.a0_binding,
        "build_binding_verdict",
        lambda **kwargs: json.loads(
            Path(kwargs["result_path"]).with_name("a0.decision.json").read_text()
        ),
    )
    monkeypatch.setattr(
        contract.search_adjudicator,
        "adjudicate",
        lambda manifest: json.loads(
            Path(manifest).with_name(
                Path(manifest).name.replace(".source.json", ".decision.json")
            ).read_text()
        ),
    )


def _checkpoint(path: Path, marker: int) -> None:
    torch.save(
        {
            "marker": marker,
            "mask_hidden_info": True,
            "value_training": {
                "schema_version": "value-training-v1",
                "primary_readout": "scalar",
                "trained_value_readouts": ["scalar"],
                "resolved_scalar_mse_weight": 0.25,
                "resolved_categorical_ce_weight": 0.0,
                "hlgauss_bins": 0,
            },
        },
        path,
    )


def _legacy_scalar_pair(
    tmp_path: Path, *, stem: str = "legacy"
) -> tuple[Path, Path, Path]:
    checkpoint = tmp_path / f"{stem}.pt"
    torch.save(
        {
            "policy_type": "entity_graph",
            "config": EntityGraphConfig(
                action_size=290,
                static_action_feature_size=16,
                value_categorical_bins=0,
            ),
            "mask_hidden_info": True,
            "model": {"value_head.2.bias": torch.tensor([1.0])},
        },
        checkpoint,
    )
    report = tmp_path / f"{stem}.report.json"
    report.write_text(
        json.dumps(
            {
                "arch": "entity_graph",
                "checkpoint": str(checkpoint),
                "mask_hidden_info": True,
                "epochs": 2,
                "steps_completed": 20,
                "value_loss_weight": 0.25,
                "value_head_type": "scalar",
                "value_categorical_loss_weight": 0.0,
                "metrics": [
                    {"epoch": 1, "value_loss": 0.7, "validation": {"value_loss": 0.72}},
                    {"epoch": 2, "value_loss": 0.6, "validation": {"value_loss": 0.64}},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    attestation = tmp_path / f"{stem}.attestation.json"
    legacy_scalar.write_attestation(checkpoint, report, attestation)
    return checkpoint, report, attestation


def _resolved_draft(tmp_path: Path) -> Path:
    payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    search = payload["science"]["search"]
    search.update(
        {
            "c_scale": 0.03,
            "n_full": 128,
            "p_full": 0.4,
            "n_full_wide": 256,
            "n_full_wide_threshold": 40,
            "wide_roots_always_full": True,
            "symmetry_averaged_eval": True,
            "symmetry_averaged_eval_threshold": 20,
            "exact_budget_sh": True,
            "exact_budget_sh_min_n": 48,
            "rescale_noise_floor_c": 1.0,
            "sigma_eval": 0.98,
        }
    )
    payload["science"]["evaluator"]["value_readout"] = "scalar"
    payload["science"]["learner_value_objective"] = {
        "objective": "hlgauss",
        "value_readout": "categorical",
        "value_categorical_bins": 33,
        "hlgauss_sigma_ratio": 0.75,
    }
    selected_search = contract._search_operator(search)
    effective_evaluator = contract._effective_evaluator(payload["science"]["evaluator"])
    producer_checkpoint: Path | None = None
    for index, item in enumerate(payload["checkpoints"]):
        checkpoint = tmp_path / f"checkpoint_{index}.pt"
        _checkpoint(checkpoint, index)
        item["path"] = str(checkpoint)
        if item["role"] == "producer":
            producer_checkpoint = checkpoint
            item["legacy_scalar_readout_attestation"] = None
    assert producer_checkpoint is not None
    prior_decisions: dict[str, Path] = {}
    for item in payload["science"]["evidence"]:
        kind = item["kind"]
        source = tmp_path / f"{kind}.source.json"
        source_payload = (
            {"stage": kind, "raw_result": "locked"}
            if kind == "a0"
            else {
                "schema_version": contract.search_adjudicator.MANIFEST_SCHEMA,
                "stage": kind,
                "checkpoint": {
                    "path": str(producer_checkpoint),
                    "sha256": contract._sha256(producer_checkpoint),
                },
            }
        )
        source.write_text(json.dumps(source_payload) + "\n")
        source_record = {"path": str(source), "sha256": contract._sha256(source)}
        evidence = tmp_path / f"{kind}.decision.json"
        if kind == "a0":
            result = tmp_path / "a0.result.json"
            result.write_text('{"result":"sealed"}\n', encoding="utf-8")
            evidence_payload = {
                "schema_version": "a0-binding-verdict-v1",
                "a0_interpretable": True,
                "a0_stage_complete": True,
                "a0_binding_pass": True,
                "hlgauss_adoption_pass": True,
                "gates": {
                    "scalar_reproduction": True,
                    "hl_training_stability": True,
                    "exact_validation_seeds": True,
                    "categorical_readout_provenance": True,
                    "calibration": True,
                    "policy_drift": True,
                },
                "decision": {
                    "status": "adopt_hlgauss_for_a1",
                    "learner_objective": "hlgauss",
                    "learner_value_readout": "categorical",
                    "mechanism_checkpoint_sha256": "sha256:" + "a" * 64,
                    "mechanism_checkpoint_is_production_candidate": False,
                },
                "sealed_inputs": {
                    "lock": str(source),
                    "lock_sha256": contract._sha256(source).removeprefix("sha256:"),
                    "training_result": str(result),
                    "training_result_sha256": contract._sha256(result).removeprefix(
                        "sha256:"
                    ),
                },
                "calibration_artifacts": {},
                "policy_drift": {},
            }
        else:
            stage_keys = {
                "s1": (
                    "c_scale",
                    "symmetry_averaged_eval",
                    "symmetry_averaged_eval_threshold",
                    "rescale_noise_floor_c",
                    "sigma_eval",
                ),
                "s2": ("n_full", "n_fast", "p_full"),
                "s3": (
                    "n_full_wide",
                    "n_full_wide_threshold",
                    "wide_roots_always_full",
                ),
            }[kind]
            selected = {key: selected_search[key] for key in stage_keys}
            evidence_payload = {
                "schema_version": "rl-rnd-stage-decision-v1",
                "stage": kind,
                "passed": True,
                "decision": "adopt",
                "adjudicator": {
                    "path": str(
                        contract.REPO_ROOT / "tools" / "search_teacher_adjudicator.py"
                    ),
                    "sha256": contract._sha256(
                        contract.REPO_ROOT / "tools" / "search_teacher_adjudicator.py"
                    ),
                },
                "source_artifacts": [
                    source_record,
                    *(
                        [
                            {
                                "path": str(prior_decisions[predecessor]),
                                "sha256": contract._sha256(
                                    prior_decisions[predecessor]
                                ),
                            }
                            for predecessor in (
                                ("s1",) if kind == "s2" else ("s1", "s2") if kind == "s3" else ()
                            )
                        ]
                    ),
                ],
                "selected_fields": selected,
                "selected_fields_sha256": contract._digest_value(selected),
            }
            if kind == "s3":
                evidence_payload.update(
                    {
                        "final_search_operator": selected_search,
                        "final_search_operator_sha256": contract._digest_value(
                            selected_search
                        ),
                        "teacher_evaluator": effective_evaluator,
                        "teacher_evaluator_sha256": contract._digest_value(
                            effective_evaluator
                        ),
                    }
                )
        evidence.write_text(json.dumps(evidence_payload) + "\n", encoding="utf-8")
        item["path"] = str(evidence)
        prior_decisions[kind] = evidence
    payload["generation"]["late_temperature_decisions"] = 180
    payload["generation"]["late_temperature"] = 0.25
    payload["generation"]["workers_per_gpu"] = 1
    payload["fleet"]["seed_base"] = 86_000_000_000
    ledger = tmp_path / "SEED_LEDGER.md"
    ledger.write_text(
        "# seed ledger\n\n| range | owner |\n|---|---|\n", encoding="utf-8"
    )
    payload["fleet"]["seed_ledger"] = str(ledger)
    payload["fleet"]["output_root"] = str(tmp_path / "wave")
    guard = tmp_path / "generate_guard.json"
    guard.write_text(
        json.dumps(
            {
                "guards": [
                    {
                        "name": "cli_flag_lint",
                        "args": {
                            "critical_flags": [
                                "--c-scale",
                                "--c-visit",
                                "--n-full",
                                "--n-fast",
                                "--base-seed",
                                "--games",
                            ],
                            "expected_values": {
                                "--c-scale": 0.03,
                                "--temperature-decisions": 90,
                                "--public-observation": True,
                                "--lazy-interior-chance": True,
                            },
                        },
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload["provenance"] = {
        "guard_config": str(guard),
        "generator_code_files": [
            str(contract.REPO_ROOT / suffix)
            for suffix in sorted(contract.REQUIRED_GENERATOR_CODE_SUFFIXES)
        ],
        "learner_code_files": [
            str(contract.REPO_ROOT / suffix)
            for suffix in sorted(contract.REQUIRED_LEARNER_CODE_SUFFIXES)
        ],
    }
    draft = tmp_path / "draft.json"
    draft.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return draft


def _lock(tmp_path: Path) -> tuple[Path, dict]:
    draft = _resolved_draft(tmp_path)
    payload = contract.build_lock(draft)
    path = tmp_path / "contract.lock.json"
    contract._create_readonly(path, payload)
    return path, payload


def _append_job_claims(lock: dict, jobs: list[dict] | None = None) -> None:
    ledger = Path(lock["fleet"]["seed_ledger"]["path"])
    selected = lock["fleet"]["jobs"] if jobs is None else jobs
    with ledger.open("a", encoding="utf-8") as handle:
        for job in selected:
            handle.write(contract._ledger_claim_row(lock, job) + "\n")


def test_checked_in_template_is_intentionally_unresolved_and_refuses_seal() -> None:
    payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    unresolved = contract._find_unresolved(payload)
    assert "$.science.search.n_full" in unresolved
    assert "$.science.evaluator.value_readout" in unresolved
    assert "$.checkpoints[0].legacy_scalar_readout_attestation" in unresolved
    assert "$.fleet.seed_base" in unresolved
    with pytest.raises(contract.ContractError, match="finish A0/S1-S3"):
        contract.build_lock(TEMPLATE)


def test_checked_in_relative_provenance_paths_canonicalize_to_required_files() -> None:
    payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    provenance = payload["provenance"]
    generator_paths = [
        contract._absolute_ref(raw, base=TEMPLATE.parent)
        for raw in provenance["generator_code_files"]
    ]
    learner_paths = [
        contract._absolute_ref(raw, base=TEMPLATE.parent)
        for raw in provenance["learner_code_files"]
    ]
    assert all(path.is_file() and ".." not in path.parts for path in generator_paths)
    assert all(path.is_file() and ".." not in path.parts for path in learner_paths)
    assert all(
        any(path.as_posix().endswith(suffix) for path in learner_paths)
        for suffix in contract.REQUIRED_LEARNER_CODE_SUFFIXES
    )


def test_seal_expands_exact_category_jobs_and_binds_science_hashes(
    tmp_path: Path,
) -> None:
    draft = _resolved_draft(tmp_path)
    lock = contract.build_lock(draft)

    jobs = lock["fleet"]["jobs"]
    assert len(jobs) == 120
    assert Counter(
        {
            category: sum(job["games"] for job in jobs if job["category"] == category)
            for category in contract.EXPECTED_GAMES
        }
    ) == Counter(
        {"current_producer": 9600, "recent_history": 1800, "hard_negative": 600}
    )
    for worker_id in {job["worker_id"] for job in jobs}:
        per_worker = {
            job["category"]: job["games"]
            for job in jobs
            if job["worker_id"] == worker_id
        }
        assert per_worker == contract.EXPECTED_PER_WORKER
    contract.assert_disjoint_seed_blocks(
        [(job["job_id"], job["base_seed"], job["games"]) for job in jobs]
    )
    assert lock["science"]["search_operator_sha256"].startswith("sha256:")
    assert lock["science"]["effective_search_config_sha256"].startswith("sha256:")
    assert "max_root_candidates" in lock["science"]["effective_search_config"]
    assert "max_root_candidates" not in lock["science"]["search_operator"]
    assert lock["science"]["evaluator_sha256"].startswith("sha256:")
    assert lock["science"]["value_readout"] == "scalar"
    assert (
        lock["science"]["learner_training_recipe"]
        == contract.EXPECTED_LEARNER_TRAINING_RECIPE
    )
    assert lock["science"]["learner_training_recipe_sha256"] == contract._digest_value(
        contract.EXPECTED_LEARNER_TRAINING_RECIPE
    )
    assert lock["fleet"]["seed_ledger"]["sha256"].startswith("sha256:")
    runtime_paths = {
        Path(record["path"]).as_posix()
        for record in lock["provenance"]["runtime_code_tree"]
    }
    assert all(
        any(path.endswith(suffix) for path in runtime_paths)
        for suffix in contract.REQUIRED_RUNTIME_CODE_SUFFIXES
    )
    assert lock["provenance"]["runtime_code_tree_sha256"] == contract._digest_value(
        lock["provenance"]["runtime_code_tree"]
    )
    assert lock["contract_sha256"].startswith("sha256:")


@pytest.mark.parametrize(
    ("key", "drifted", "error"),
    [
        ("batch_size", 8192, "must equal the locked pre-wave value"),
        ("world_size", 1.0, "must have JSON type int"),
        ("resume_optimizer", 0, "must have JSON type bool"),
        ("amp", "none", "must equal the locked pre-wave value"),
    ],
)
def test_learner_training_recipe_rejects_value_or_type_drift(
    key: str,
    drifted: object,
    error: str,
) -> None:
    recipe = dict(contract.EXPECTED_LEARNER_TRAINING_RECIPE)
    recipe[key] = drifted
    with pytest.raises(contract.ContractError, match=error):
        contract._validate_learner_training_recipe(recipe)


def test_learner_training_recipe_rejects_missing_or_extra_fields() -> None:
    missing = dict(contract.EXPECTED_LEARNER_TRAINING_RECIPE)
    missing.pop("ddp_shard_data")
    with pytest.raises(contract.ContractError, match="fields mismatch"):
        contract._validate_learner_training_recipe(missing)

    extra = {**contract.EXPECTED_LEARNER_TRAINING_RECIPE, "unsealed_knob": 1}
    with pytest.raises(contract.ContractError, match="fields mismatch"):
        contract._validate_learner_training_recipe(extra)


def test_seal_rejects_seed_ledger_collision(tmp_path: Path) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text())
    Path(payload["fleet"]["seed_ledger"]).write_text(
        "[86000000000 – 86000000100) | already-used |\n", encoding="utf-8"
    )
    draft.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(contract.ContractError, match="overlaps ledger claim"):
        contract.build_lock(draft)


def test_seal_rejects_val_only_seed_plan(tmp_path: Path) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text())
    payload["fleet"]["seed_base"] = contract.VAL_ONLY_SEED_RANGE[0]
    draft.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(contract.ContractError, match="VAL-ONLY"):
        contract.build_lock(draft)


def test_seal_rejects_a_ledger_range_row_the_shared_parser_skips(
    tmp_path: Path,
) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text())
    # A conventional leading Markdown pipe is range-like but intentionally not
    # accepted by prelaunch_guard.parse_seed_ledger.  The immutable handoff is
    # stricter: it must not mistake this hidden claim for free space.
    Path(payload["fleet"]["seed_ledger"]).write_text(
        "| [86,000,000,000 – 86,000,000,100) | hidden-claim |\n", encoding="utf-8"
    )
    draft.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(contract.ContractError, match="range-like.*parsed"):
        contract.build_lock(draft)


def test_seal_rejects_guard_drift(tmp_path: Path) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text())
    guard_path = Path(payload["provenance"]["guard_config"])
    guard = json.loads(guard_path.read_text())
    guard["guards"][0]["args"]["expected_values"]["--c-scale"] = 0.1
    guard_path.write_text(json.dumps(guard), encoding="utf-8")
    with pytest.raises(contract.ContractError, match="guard drift"):
        contract.build_lock(draft)


def _select_s1_c_scale(draft: Path, selected: float) -> tuple[dict, Path]:
    payload = json.loads(draft.read_text(encoding="utf-8"))
    payload["science"]["search"]["c_scale"] = selected
    s1_path = _evidence_path(payload, "s1")
    s1 = json.loads(s1_path.read_text(encoding="utf-8"))
    s1["selected_fields"]["c_scale"] = selected
    s1["selected_fields_sha256"] = contract._digest_value(s1["selected_fields"])
    s1_path.write_text(json.dumps(s1) + "\n", encoding="utf-8")
    draft.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload, s1_path


def _rebind_downstream_search_evidence(payload: dict, selected: float) -> None:
    """Keep the resolved fixture's S1->S2->S3 artifact chain exact."""

    decisions = {
        stage: _evidence_path(payload, stage) for stage in ("s1", "s2", "s3")
    }
    s2 = json.loads(decisions["s2"].read_text(encoding="utf-8"))
    for record in s2["source_artifacts"]:
        if Path(record["path"]) == decisions["s1"]:
            record["sha256"] = contract._sha256(decisions["s1"])
    decisions["s2"].write_text(json.dumps(s2) + "\n", encoding="utf-8")

    s3 = json.loads(decisions["s3"].read_text(encoding="utf-8"))
    for record in s3["source_artifacts"]:
        for predecessor in ("s1", "s2"):
            if Path(record["path"]) == decisions[predecessor]:
                record["sha256"] = contract._sha256(decisions[predecessor])
    s3["final_search_operator"]["c_scale"] = selected
    s3["final_search_operator_sha256"] = contract._digest_value(
        s3["final_search_operator"]
    )
    decisions["s3"].write_text(json.dumps(s3) + "\n", encoding="utf-8")


def test_sync_generation_guard_is_byte_for_byte_noop_for_default_s1(
    tmp_path: Path,
) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text(encoding="utf-8"))
    guard_path = Path(payload["provenance"]["guard_config"])
    before = guard_path.read_bytes()

    result = contract.sync_generation_guard(draft)

    assert result["status"] == "already_synchronized"
    assert result["changed"] is False
    assert result["selected_c_scale"] == 0.03
    assert result["before_sha256"] == result["after_sha256"]
    assert guard_path.read_bytes() == before
    assert contract.GUARD_SYNC_KEY not in json.loads(before)


def test_sync_generation_guard_embeds_typed_s1_receipt_for_nondefault_selection(
    tmp_path: Path,
) -> None:
    draft = _resolved_draft(tmp_path)
    payload, s1_path = _select_s1_c_scale(draft, 0.1)
    _rebind_downstream_search_evidence(payload, 0.1)
    guard_path = Path(payload["provenance"]["guard_config"])
    before_sha256 = contract._sha256(guard_path)

    result = contract.sync_generation_guard(draft)

    assert result["status"] == "synchronized"
    assert result["changed"] is True
    assert result["before_sha256"] == before_sha256
    assert result["after_sha256"] == contract._sha256(guard_path)
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    expected, _ = contract._guard_expected_values(guard_path)
    assert expected["--c-scale"] == 0.1
    receipt = guard[contract.GUARD_SYNC_KEY]
    assert receipt["schema_version"] == contract.GUARD_SYNC_SCHEMA
    assert receipt["selected_c_scale"] == 0.1
    assert receipt["previous_guard_sha256"] == before_sha256
    assert receipt["source_s1_evidence"] == {
        "path": str(s1_path.resolve(strict=True)),
        "sha256": contract._sha256(s1_path),
    }
    assert receipt["synchronizer"]["path"] == contract.GUARD_SYNC_TOOL
    assert not Path(receipt["synchronizer"]["path"]).is_absolute()
    assert str(contract.REPO_ROOT) not in json.dumps(receipt["synchronizer"])

    # The operation is idempotent only while that exact typed receipt remains
    # valid; reruns do not rewrite or churn the runtime hash.
    synchronized = guard_path.read_bytes()
    second = contract.sync_generation_guard(draft)
    assert second["status"] == "already_synchronized"
    assert second["changed"] is False
    assert guard_path.read_bytes() == synchronized

    lock = contract.build_lock(draft)
    assert lock["provenance"]["guard_config"]["sha256"] == contract._sha256(
        guard_path
    )


def test_sync_generation_guard_rejects_manual_nondefault_edit_without_receipt(
    tmp_path: Path,
) -> None:
    draft = _resolved_draft(tmp_path)
    payload, _ = _select_s1_c_scale(draft, 0.3)
    guard_path = Path(payload["provenance"]["guard_config"])
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    guard["guards"][0]["args"]["expected_values"]["--c-scale"] = 0.3
    guard_path.write_text(json.dumps(guard) + "\n", encoding="utf-8")

    with pytest.raises(contract.ContractError, match="without a .* receipt"):
        contract.sync_generation_guard(draft)


def test_sync_generation_guard_rejects_tampered_s1_receipt(tmp_path: Path) -> None:
    draft = _resolved_draft(tmp_path)
    payload, _ = _select_s1_c_scale(draft, 0.1)
    guard_path = Path(payload["provenance"]["guard_config"])
    contract.sync_generation_guard(draft)
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    guard[contract.GUARD_SYNC_KEY]["source_s1_evidence"]["sha256"] = (
        "sha256:" + "0" * 64
    )
    guard_path.write_text(json.dumps(guard) + "\n", encoding="utf-8")

    with pytest.raises(contract.ContractError, match="S1 provenance drift"):
        contract.sync_generation_guard(draft)


def test_sync_generation_guard_validates_before_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    draft = _resolved_draft(tmp_path)
    payload, _ = _select_s1_c_scale(draft, 0.1)
    guard_path = Path(payload["provenance"]["guard_config"])
    original = guard_path.read_bytes()
    real_validate = contract._validate_guard_payload

    def reject_prospective(payload, **kwargs):
        if contract.GUARD_SYNC_KEY in payload:
            raise contract.ContractError("injected prospective validation failure")
        return real_validate(payload, **kwargs)

    monkeypatch.setattr(contract, "_validate_guard_payload", reject_prospective)
    with pytest.raises(contract.ContractError, match="prospective validation failure"):
        contract.sync_generation_guard(draft)
    assert guard_path.read_bytes() == original


def _evidence_path(payload: dict, kind: str) -> Path:
    return Path(
        next(
            item["path"]
            for item in payload["science"]["evidence"]
            if item["kind"] == kind
        )
    )


def _rebind_search_evidence_checkpoint(payload: dict, checkpoint: Path) -> None:
    decisions: dict[str, Path] = {}
    for stage in ("s1", "s2", "s3"):
        decision_path = _evidence_path(payload, stage)
        source_path = decision_path.with_name(f"{stage}.source.json")
        source = json.loads(source_path.read_text())
        source["checkpoint"] = {
            "path": str(checkpoint),
            "sha256": contract._sha256(checkpoint),
        }
        source_path.write_text(json.dumps(source), encoding="utf-8")
        decision = json.loads(decision_path.read_text())
        decision["source_artifacts"] = [
            {"path": str(source_path), "sha256": contract._sha256(source_path)},
            *[
                {
                    "path": str(decisions[predecessor]),
                    "sha256": contract._sha256(decisions[predecessor]),
                }
                for predecessor in (
                    ("s1",) if stage == "s2" else ("s1", "s2") if stage == "s3" else ()
                )
            ],
        ]
        decision_path.write_text(json.dumps(decision), encoding="utf-8")
        decisions[stage] = decision_path


def test_a0_retain_scalar_is_valid_and_separate_from_teacher_readout(
    tmp_path: Path,
) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text())
    payload["science"]["learner_value_objective"] = {
        "objective": "mse",
        "value_readout": "scalar",
        "value_categorical_bins": None,
        "hlgauss_sigma_ratio": None,
    }
    a0_path = _evidence_path(payload, "a0")
    a0 = json.loads(a0_path.read_text())
    a0["hlgauss_adoption_pass"] = False
    a0["gates"].update(
        {
            "hl_training_stability": False,
            "exact_validation_seeds": None,
            "categorical_readout_provenance": None,
            "calibration": None,
            "policy_drift": None,
        }
    )
    a0["calibration_artifacts"] = None
    a0["policy_drift"] = None
    a0["decision"].update(
        {
            "status": "retain_scalar_for_a1",
            "learner_objective": "mse",
            "learner_value_readout": "scalar",
        }
    )
    a0_path.write_text(json.dumps(a0), encoding="utf-8")
    draft.write_text(json.dumps(payload), encoding="utf-8")

    lock = contract.build_lock(draft)
    assert lock["science"]["learner_value_objective"]["objective"] == "mse"
    assert lock["science"]["value_readout"] == "scalar"


def test_seal_rejects_nonpassing_a0_evidence(tmp_path: Path) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text())
    a0_path = _evidence_path(payload, "a0")
    a0 = json.loads(a0_path.read_text())
    a0["a0_binding_pass"] = False
    a0_path.write_text(json.dumps(a0), encoding="utf-8")
    with pytest.raises(contract.ContractError, match="a0_binding_pass"):
        contract.build_lock(draft)


def test_seal_rejects_fabricated_a0_envelope_even_when_source_hashes_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text())
    a0_path = _evidence_path(payload, "a0")
    replayed = json.loads(a0_path.read_text())
    monkeypatch.setattr(
        contract.a0_binding,
        "build_binding_verdict",
        lambda **_kwargs: replayed,
    )
    fabricated = json.loads(a0_path.read_text())
    fabricated["interpretation"] = "operator-authored arbitrary JSON"
    a0_path.write_text(json.dumps(fabricated), encoding="utf-8")
    with pytest.raises(contract.ContractError, match="replayed binding verdict"):
        contract.build_lock(draft)


def test_seal_rejects_wrong_search_evidence_stage(tmp_path: Path) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text())
    s2_path = _evidence_path(payload, "s2")
    s2 = json.loads(s2_path.read_text())
    s2["stage"] = "s1"
    s2_path.write_text(json.dumps(s2), encoding="utf-8")
    with pytest.raises(contract.ContractError, match="expected 's2'"):
        contract.build_lock(draft)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("passed", False, "passed != true"),
        ("schema_version", "wrong-schema", "schema must be"),
    ],
)
def test_seal_rejects_failed_or_wrong_schema_search_evidence(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text())
    s1_path = _evidence_path(payload, "s1")
    s1 = json.loads(s1_path.read_text())
    s1[field] = value
    s1_path.write_text(json.dumps(s1), encoding="utf-8")
    with pytest.raises(contract.ContractError, match=message):
        contract.build_lock(draft)


def test_seal_rejects_selected_search_config_mismatch(tmp_path: Path) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text())
    s2_path = _evidence_path(payload, "s2")
    s2 = json.loads(s2_path.read_text())
    s2["selected_fields"]["n_full"] = 64
    s2["selected_fields_sha256"] = contract._digest_value(s2["selected_fields"])
    s2_path.write_text(json.dumps(s2), encoding="utf-8")
    with pytest.raises(contract.ContractError, match="selected search fields mismatch"):
        contract.build_lock(draft)


def test_seal_rejects_fabricated_search_envelope_and_swapped_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text())
    s3_path = _evidence_path(payload, "s3")
    replayed = json.loads(s3_path.read_text())
    monkeypatch.setattr(
        contract.search_adjudicator, "adjudicate", lambda _manifest: replayed
    )
    fabricated = json.loads(s3_path.read_text())
    fabricated["metrics"] = {"invented": True}
    s3_path.write_text(json.dumps(fabricated), encoding="utf-8")
    with pytest.raises(contract.ContractError, match="replayed adjudication"):
        contract.build_lock(draft)

    # Restore an exactly replayable S3 envelope but replace its exact S2
    # predecessor with an unrelated typed-looking decision.
    s3_path.write_text(json.dumps(replayed), encoding="utf-8")
    fake_s2 = tmp_path / "fake-s2.decision.json"
    fake_s2.write_text(json.dumps(json.loads(_evidence_path(payload, "s2").read_text())))
    swapped = json.loads(s3_path.read_text())
    swapped["source_artifacts"] = [
        record
        for record in swapped["source_artifacts"]
        if Path(record["path"]) != _evidence_path(payload, "s2")
    ] + [{"path": str(fake_s2), "sha256": contract._sha256(fake_s2)}]
    s3_path.write_text(json.dumps(swapped), encoding="utf-8")
    monkeypatch.setattr(
        contract.search_adjudicator,
        "adjudicate",
        lambda manifest: (
            swapped
            if Path(manifest).name.startswith("s3.")
            else json.loads(
                Path(manifest)
                .with_name(
                    Path(manifest).name.replace(".source.json", ".decision.json")
                )
                .read_text()
            )
        ),
    )
    with pytest.raises(contract.ContractError, match="exact sealed S2"):
        contract.build_lock(draft)


def test_seal_rejects_stringly_typed_science_booleans(tmp_path: Path) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text())
    payload["science"]["search"]["wide_roots_always_full"] = "true"
    draft.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(contract.ContractError, match="JSON boolean"):
        contract.build_lock(draft)


def test_seal_rejects_global_n256(tmp_path: Path) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text())
    payload["science"]["search"]["n_full"] = 256
    draft.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(contract.ContractError, match="never global n256"):
        contract.build_lock(draft)


def test_verify_detects_mutated_artifact_bytes(tmp_path: Path) -> None:
    lock_path, lock = _lock(tmp_path)
    assert contract.verify_lock(lock_path)["contract_sha256"] == lock["contract_sha256"]
    Path(lock["science"]["evidence"][0]["path"]).write_text(
        "changed\n", encoding="utf-8"
    )
    with pytest.raises(contract.ContractError, match="artifact drift"):
        contract.verify_lock(lock_path)


def test_verify_accepts_append_only_own_claim_but_rejects_peer_overlap(
    tmp_path: Path,
) -> None:
    lock_path, lock = _lock(tmp_path)
    job = lock["fleet"]["jobs"][0]
    _append_job_claims(lock, [job])
    assert contract.verify_lock(lock_path)["contract_sha256"] == lock[
        "contract_sha256"
    ]
    with pytest.raises(contract.ContractError, match="missing exact own claim"):
        contract.verify_lock(lock_path, require_all_job_claims=True)

    ledger = Path(lock["fleet"]["seed_ledger"]["path"])
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{job['base_seed']} – {job['seed_end']}) | peer-collision |\n"
        )
    with pytest.raises(contract.ContractError, match="overlaps live ledger claim"):
        contract.verify_lock(lock_path)


def test_verify_rejects_mutation_of_sealed_ledger_prefix(tmp_path: Path) -> None:
    lock_path, lock = _lock(tmp_path)
    ledger = Path(lock["fleet"]["seed_ledger"]["path"])
    ledger.write_text("# rewritten ledger\n", encoding="utf-8")
    with pytest.raises(contract.ContractError, match="append-only extension"):
        contract.verify_lock(lock_path)


def test_verify_rejects_spoofed_or_duplicate_own_claim_rows(tmp_path: Path) -> None:
    lock_path, lock = _lock(tmp_path)
    jobs = lock["fleet"]["jobs"]
    _append_job_claims(lock, jobs[1:])
    first = jobs[0]
    ledger = Path(lock["fleet"]["seed_ledger"]["path"])
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{first['base_seed']} – {first['seed_end']}) | "
            f"nonsense-claim={first['claim_label']} "
            f"contract={'sha256:' + '0' * 64} job=WRONG |\n"
        )
    with pytest.raises(contract.ContractError, match="does not exactly match"):
        contract.verify_lock(lock_path, require_all_job_claims=True)

    # Restore the sealed prefix, then demonstrate that duplicate exact rows are
    # also forbidden: a resume reuses the existing claim instead of appending.
    ledger.write_text(lock["fleet"]["seed_ledger"]["snapshot_text"], encoding="utf-8")
    _append_job_claims(lock)
    _append_job_claims(lock, [first])
    with pytest.raises(contract.ContractError, match="repeats exact own claim"):
        contract.verify_lock(lock_path, require_all_job_claims=True)


def test_verify_rejects_bound_learner_implementation_drift(tmp_path: Path) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text(encoding="utf-8"))
    shadow_paths: list[str] = []
    for suffix in sorted(contract.REQUIRED_LEARNER_CODE_SUFFIXES):
        source = contract.REPO_ROOT / suffix
        shadow = tmp_path / "shadow" / suffix
        shadow.parent.mkdir(parents=True, exist_ok=True)
        shadow.write_bytes(source.read_bytes())
        shadow_paths.append(str(shadow))
    payload["provenance"]["learner_code_files"] = shadow_paths
    draft.write_text(json.dumps(payload), encoding="utf-8")
    lock = contract.build_lock(draft)
    lock_path = tmp_path / "learner.lock.json"
    contract._create_readonly(lock_path, lock)

    train_path = next(path for path in shadow_paths if path.endswith("tools/train_bc.py"))
    Path(train_path).write_text("# drift\n", encoding="utf-8")
    with pytest.raises(contract.ContractError, match="artifact drift"):
        contract.verify_lock(lock_path)


def test_seal_accepts_typed_legacy_scalar_producer_attestation(tmp_path: Path) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text())
    checkpoint, _report, attestation = _legacy_scalar_pair(tmp_path)
    producer = next(
        item for item in payload["checkpoints"] if item["role"] == "producer"
    )
    producer["path"] = str(checkpoint)
    producer["legacy_scalar_readout_attestation"] = str(attestation)
    _rebind_search_evidence_checkpoint(payload, checkpoint)
    draft.write_text(json.dumps(payload), encoding="utf-8")

    lock = contract.build_lock(draft)
    locked_producer = next(
        item for item in lock["checkpoints"] if item["role"] == "producer"
    )
    metadata = locked_producer["metadata"]
    assert metadata["value_training_schema"] == legacy_scalar.SCHEMA_VERSION
    assert (
        metadata["legacy_scalar_readout_attestation"]["checkpoint"]["sha256"]
        == (locked_producer["sha256"])
    )
    lock_path = tmp_path / "legacy.contract.lock.json"
    contract._create_readonly(lock_path, lock)
    assert contract.verify_lock(lock_path)["contract_sha256"] == lock["contract_sha256"]


def test_seal_rejects_legacy_attestation_for_a_different_checkpoint(
    tmp_path: Path,
) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text())
    _wrong_checkpoint, _wrong_report, wrong_attestation = _legacy_scalar_pair(
        tmp_path, stem="wrong"
    )
    producer = next(
        item for item in payload["checkpoints"] if item["role"] == "producer"
    )
    producer["legacy_scalar_readout_attestation"] = str(wrong_attestation)
    draft.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(contract.ContractError, match="wrong checkpoint path"):
        contract.build_lock(draft)


def test_verify_lock_detects_legacy_report_tamper(tmp_path: Path) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text())
    checkpoint, report, attestation = _legacy_scalar_pair(tmp_path)
    producer = next(
        item for item in payload["checkpoints"] if item["role"] == "producer"
    )
    producer.update(
        {"path": str(checkpoint), "legacy_scalar_readout_attestation": str(attestation)}
    )
    _rebind_search_evidence_checkpoint(payload, checkpoint)
    draft.write_text(json.dumps(payload), encoding="utf-8")
    lock = contract.build_lock(draft)
    lock_path = tmp_path / "legacy.contract.lock.json"
    contract._create_readonly(lock_path, lock)

    report_payload = json.loads(report.read_text())
    report_payload["metrics"][0]["value_loss"] = 9.0
    report.write_text(json.dumps(report_payload), encoding="utf-8")
    with pytest.raises(contract.ContractError, match="report hash drift"):
        contract.verify_lock(lock_path)


def test_categorical_contract_rejects_legacy_scalar_attestation(tmp_path: Path) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text())
    checkpoint, _report, attestation = _legacy_scalar_pair(tmp_path)
    producer = next(
        item for item in payload["checkpoints"] if item["role"] == "producer"
    )
    producer.update(
        {"path": str(checkpoint), "legacy_scalar_readout_attestation": str(attestation)}
    )
    _rebind_search_evidence_checkpoint(payload, checkpoint)
    payload["science"]["evaluator"]["value_readout"] = "categorical"
    effective_evaluator = contract._effective_evaluator(payload["science"]["evaluator"])
    s3_path = _evidence_path(payload, "s3")
    s3 = json.loads(s3_path.read_text())
    s3["teacher_evaluator"] = effective_evaluator
    s3["teacher_evaluator_sha256"] = contract._digest_value(effective_evaluator)
    s3_path.write_text(json.dumps(s3), encoding="utf-8")
    draft.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        contract.ContractError, match="cannot authorize categorical readout"
    ):
        contract.build_lock(draft)


def test_render_writes_commands_only_and_never_overwrites(tmp_path: Path) -> None:
    lock_path, lock = _lock(tmp_path)
    rendered = tmp_path / "rendered"
    payload = contract.render(lock_path, rendered)

    assert payload["execution_policy"]["execute"] is False
    assert len(payload["commands"]) == 120
    assert len(list((rendered / "job_attestations").glob("*.json"))) == 120
    assert (
        sum(
            command["category"] == "current_producer" for command in payload["commands"]
        )
        == 40
    )
    current = payload["commands"][0]
    assert current["environment"]["CATAN_A1_CONTRACT_SHA256"] == lock["contract_sha256"]
    assert current["output_attestation"]["destination"].endswith("/a1_contract.json")
    assert current["ledger_claim"] == {
        "path": lock["fleet"]["seed_ledger"]["path"],
        "row": contract._ledger_claim_row(lock, lock["fleet"]["jobs"][0]),
        "row_sha256": contract._digest_value(
            contract._ledger_claim_row(lock, lock["fleet"]["jobs"][0])
        ),
    }
    assert "--n-full" in current["argv"] and "128" in current["argv"]
    assert "--p-full" in current["argv"] and "0.4" in current["argv"]
    assert "--symmetry-averaged-eval" in current["argv"]
    assert "--n-full-wide" in current["argv"] and "256" in current["argv"]
    assert "--value-readout" in current["argv"] and "scalar" in current["argv"]
    history = next(
        command
        for command in payload["commands"]
        if command["category"] == "recent_history"
    )
    assert "--opponent-mix-manifest" in history["argv"]
    parser = generator.build_parser()
    for command in payload["commands"]:
        parsed = parser.parse_args(command["argv"][1:])
        assert parsed.skip_guards is False
        assert parsed.public_observation is True
        assert parsed.base_seed > 0
    assert not os.access(rendered / "commands.json", os.W_OK) or (
        (rendered / "commands.json").stat().st_mode & stat.S_IWUSR == 0
    )
    with pytest.raises(contract.ContractError, match="absent or empty"):
        contract.render(lock_path, rendered)


def test_shard_resolution_rejects_stale_absolute_basename_alias(tmp_path: Path) -> None:
    manifest = tmp_path / "job" / "manifest.json"
    manifest.parent.mkdir()
    (manifest.parent / "shard_00000.npz").write_bytes(b"different-local-bytes")
    with pytest.raises(contract.ContractError, match="missing shard"):
        contract._resolve_shard(manifest, "/stale/other/run/shard_00000.npz")


def test_raw_game_seed_runs_allow_adjacent_split_but_reject_reappearance() -> None:
    closed: set[int] = set()
    active = contract._advance_game_seed_runs(
        np.asarray([11, 11, 12], dtype=np.int64),
        active_seed=None,
        closed_seeds=closed,
        where="shard-0",
    )
    active = contract._advance_game_seed_runs(
        np.asarray([12, 12, 13], dtype=np.int64),
        active_seed=active,
        closed_seeds=closed,
        where="shard-1",
    )

    assert active == 13
    assert closed == {11, 12}
    with pytest.raises(contract.ContractError, match="second non-contiguous raw run"):
        contract._advance_game_seed_runs(
            np.asarray([11], dtype=np.int64),
            active_seed=active,
            closed_seeds=closed,
            where="shard-2",
        )


def _valid_selected_telemetry() -> dict[str, np.ndarray]:
    return {
        "is_forced": np.asarray([False, True, False]),
        "used_full_search": np.asarray([True, True, False]),
        "phase": np.asarray(["MAIN", "MAIN", "ROBBER"]),
        "decision_index": np.asarray([0, 1, 2], dtype=np.int32),
        "target_policy": np.asarray(
            [[0.75, 0.25], [1.0, 0.0], [0.4, 0.6]], dtype=np.float32
        ),
        "target_policy_mask": np.asarray(
            [[True, True], [True, False], [True, True]], dtype=bool
        ),
    }


@pytest.mark.parametrize(
    "missing_column", sorted(contract.REQUIRED_SELECTED_TELEMETRY_COLUMNS)
)
def test_selected_telemetry_rejects_every_missing_report_source(
    missing_column: str,
) -> None:
    payload = _valid_selected_telemetry()
    payload.pop(missing_column)
    with pytest.raises(contract.ContractError, match="missing selected telemetry"):
        contract._selected_telemetry_arrays(
            payload,
            game_seeds=np.asarray([11, 12, 13], dtype=np.int64),
            selected_mask=np.asarray([True, True, True]),
            max_decisions=600,
            where="job",
        )


@pytest.mark.parametrize("invalid", ["empty_mask", "zero_mass"])
def test_selected_telemetry_rejects_empty_policy_evidence(invalid: str) -> None:
    payload = _valid_selected_telemetry()
    if invalid == "empty_mask":
        payload["target_policy_mask"][1] = False
        error = "no active entries"
    else:
        payload["target_policy"][1] = 0.0
        error = "non-positive mass"
    with pytest.raises(contract.ContractError, match=error):
        contract._selected_telemetry_arrays(
            payload,
            game_seeds=np.asarray([11, 12, 13], dtype=np.int64),
            selected_mask=np.asarray([True, True, True]),
            max_decisions=600,
            where="job",
        )


def test_post_wave_audit_fails_closed_when_manifests_are_missing(
    tmp_path: Path,
) -> None:
    lock_path, lock_payload = _lock(tmp_path)
    _append_job_claims(lock_payload)
    report = tmp_path / "audit.json"
    with pytest.raises(contract.ContractError, match="post-wave audit failed"):
        contract.audit_outputs(lock_path, report)
    payload = json.loads(report.read_text())
    assert payload["passed"] is False
    assert payload["total_unique_games"] == 0
    assert any("missing manifest" in error for error in payload["errors"])


def test_post_wave_audit_canonicalizes_symlinked_contract_path(
    tmp_path: Path,
) -> None:
    lock_path, lock_payload = _lock(tmp_path)
    _append_job_claims(lock_payload)
    symlink_path = tmp_path / "contract.alias.json"
    symlink_path.symlink_to(lock_path)
    report = tmp_path / "audit.symlink.json"

    with pytest.raises(contract.ContractError, match="post-wave audit failed"):
        contract.audit_outputs(symlink_path, report)

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["contract_path"] == str(lock_path.resolve(strict=True))


def test_post_wave_audit_accepts_exact_complete_category_corpus(tmp_path: Path) -> None:
    lock_path, lock = _lock(tmp_path)
    _append_job_claims(lock)
    checkpoint_by_id = {record["id"]: record for record in lock["checkpoints"]}
    category_by_name = {item["name"]: item for item in lock["source_categories"]}
    for job in lock["fleet"]["jobs"]:
        out_dir = Path(job["output_dir"])
        out_dir.mkdir(parents=True)
        (out_dir / "a1_contract.json").write_text(
            json.dumps(contract._job_attestation(lock, job)), encoding="utf-8"
        )
        n = int(job["attempts"])
        shard = out_dir / "shard_00000.npz"
        arrays = {
            "game_seed": np.arange(job["base_seed"], job["seed_end"], dtype=np.int64),
            "action_taken": np.zeros(n, dtype=np.int16),
            "legal_action_ids": np.zeros((n, 1), dtype=np.int16),
            "legal_action_mask": np.ones((n, 1), dtype=bool),
            "terminated": np.ones(n, dtype=bool),
            "truncated": np.zeros(n, dtype=bool),
            "is_forced": np.zeros(n, dtype=bool),
            "used_full_search": np.ones(n, dtype=bool),
            "phase": np.full(n, "MAIN", dtype="U8"),
            "decision_index": np.zeros(n, dtype=np.int32),
            "target_policy": np.ones((n, 1), dtype=np.float32),
            "target_policy_mask": np.ones((n, 1), dtype=bool),
        }
        # The bounded reserve is real: the highest-seed attempt truncates and
        # must be excluded before selected metrics/holdout construction.
        arrays["terminated"][-1] = False
        arrays["truncated"][-1] = True
        if job["category"] != "current_producer":
            spec = category_by_name[job["category"]]
            opponent = checkpoint_by_id[spec["checkpoint_ids"][0]]
            arrays["opponent_tag"] = np.full(n, job["category"], dtype="U32")
            arrays["opponent_checkpoint_md5"] = np.full(n, opponent["md5"], dtype="U32")
        np.savez(shard, **arrays)

        cli = contract._expected_cli_fields(lock, job)
        if job["category"] == "current_producer":
            cli["opponent_mix_manifest"] = None
        else:
            mix = out_dir / "opponent_mix.json"
            mix.write_text(
                json.dumps(
                    {
                        "_a1_contract": {
                            "contract_sha256": lock["contract_sha256"],
                            "category": job["category"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            cli["opponent_mix_manifest"] = str(mix)
        worker = out_dir / "worker_000" / "manifest.json"
        worker.parent.mkdir()
        worker.write_text(
            json.dumps(
                {
                    "search_config": {
                        **lock["science"]["effective_search_config"],
                        "seed": 123,
                    },
                    "selfplay_config": contract._expected_selfplay_config(lock),
                }
            ),
            encoding="utf-8",
        )
        config_hash = contract.GenerateConfig.from_namespace(
            Namespace(**cli)
        ).config_hash()
        (out_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "games_requested": n,
                    "games_completed": n,
                    "games_failed": 0,
                    "games_truncated": 1,
                    "errors": [],
                    "base_seed": job["base_seed"],
                    "checkpoint": contract._producer(lock)["path"],
                    "cli_args": cli,
                    "config_hash": config_hash,
                    "worker_summaries": [str(worker)],
                    "shards": [str(shard)],
                }
            ),
            encoding="utf-8",
        )

    report_path = tmp_path / "audit.pass.json"
    report = contract.audit_outputs(lock_path, report_path)
    assert report["passed"] is True
    assert report["games"] == contract.EXPECTED_GAMES
    assert report["total_unique_games"] == 12_000
    assert report["rows"] == 12_000
    assert report["invalid_teacher_actions"] == 0
    assert report["reports"]["full_search_policy_mass"] == 1.0
    assert (
        report["reports"]["truncation"][
            "reserve_truncated_or_incomplete_attempts"
        ]
        == 120
    )
    assert {item["category"] for item in report["shards"]} == set(
        contract.EXPECTED_GAMES
    )
    assert report["source_provenance"]["hard_negative"]["opponent_checkpoint_sha256"]
    validation_path = Path(report["validation_holdout"]["manifest"])
    validation = json.loads(validation_path.read_text())
    assert validation["schema_version"] == "train-validation-game-seeds-v1"
    assert validation["validation_fraction"] == 0.05
    assert validation["validation_seed"] == 17
    assert validation["validation_max_samples"] == 0
    selected_manifest = json.loads(
        Path(report["selected_training_games"]["manifest"]).read_text()
    )
    assert selected_manifest["selected_game_count"] == 12_000
    assert sum(selected_manifest["category_game_counts"].values()) == 12_000
    assert all(
        record["game_seed"]
        < next(
            job["base_seed"] + job["games"]
            for job in lock["fleet"]["jobs"]
            if job["job_id"] == record["job_id"]
        )
        for record in selected_manifest["records"]
    )
    held_out = np.asarray(validation["game_seeds"], dtype="<i8")
    assert validation["validation_game_seed_set_sha256"] == (
        "sha256:" + hashlib.sha256(held_out.tobytes()).hexdigest()
    )


def test_categorical_teacher_requires_positive_hlgauss_provenance(
    tmp_path: Path,
) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text())
    payload["science"]["evaluator"]["value_readout"] = "categorical"
    effective_evaluator = contract._effective_evaluator(payload["science"]["evaluator"])
    s3_path = Path(
        next(
            item["path"]
            for item in payload["science"]["evidence"]
            if item["kind"] == "s3"
        )
    )
    s3 = json.loads(s3_path.read_text())
    s3["teacher_evaluator"] = effective_evaluator
    s3["teacher_evaluator_sha256"] = contract._digest_value(effective_evaluator)
    s3_path.write_text(json.dumps(s3), encoding="utf-8")
    draft.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        contract.ContractError, match="teacher readout 'categorical' was trained"
    ):
        contract.build_lock(draft)
