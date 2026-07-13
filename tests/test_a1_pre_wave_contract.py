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
from tools import search_operator_binding as operator_binding
from tools.fleet import a1_production_executor as production_executor


TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "experiments"
    / "a1_pre_wave_contract.template.json"
)
HISTORICAL_DRAFT = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "experiments"
    / "a1_pre_wave_contract.rnd_draft.json"
)
GENERATION_CAMPAIGN = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "operations"
    / "a1-dual-arm-56gpu-20260710"
    / "contract.json"
)
GENERATION_CAMPAIGN_R2 = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "operations"
    / "a1-dual-arm-56gpu-20260711-r2"
    / "contract.json"
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


def _generation_campaign_copy(tmp_path: Path, mutate) -> Path:
    payload = json.loads(GENERATION_CAMPAIGN.read_text(encoding="utf-8"))
    mutate(payload)
    payload.pop("contract_sha256", None)
    payload["contract_sha256"] = contract._digest_value(payload)  # noqa: SLF001
    path = tmp_path / "generation-campaign.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _historical_db1_campaign(tmp_path: Path) -> Path:
    lines = GENERATION_CAMPAIGN.read_text(encoding="utf-8").splitlines(keepends=True)
    historical: list[str] = []
    for line in lines:
        if '"path": "tools/fleet/a1_lane_supervisor.py"' in line:
            continue
        if '"executor": {"path": "tools/fleet/a1_production_executor.py"' in line:
            line = (
                '    "executor": {"path": "tools/fleet/a1_production_executor.py", '
                f'"sha256": "{contract.HISTORICAL_DB1_EXECUTOR_SHA256}"}},\n'
            )
        if '"contract_sha256":' in line:
            line = f'  "contract_sha256": "{contract.HISTORICAL_DB1_CAMPAIGN_SHA256}"\n'
        historical.append(line)
    path = tmp_path / "historical-db1-campaign.json"
    path.write_text("".join(historical), encoding="utf-8")
    assert contract._sha256(path) == contract.HISTORICAL_DB1_CAMPAIGN_FILE_SHA256  # noqa: SLF001
    return path


def test_exact_db1_campaign_is_accepted_only_as_existing_lock_source(
    tmp_path: Path,
) -> None:
    historical = _historical_db1_campaign(tmp_path)

    with pytest.raises(contract.ContractError, match="provenance file set drift"):
        contract.validate_generation_campaign(historical)
    verified = contract.validate_generation_campaign(
        historical, _allow_historical_lock_source=True
    )
    assert verified["contract_sha256"] == contract.HISTORICAL_DB1_CAMPAIGN_SHA256
    with pytest.raises(contract.ContractError, match="provenance file set drift"):
        contract.materialize_generation_campaign(
            historical,
            promotion_handoff_path=tmp_path / "handoff.json",
            placement_path=tmp_path / "placement.json",
            out_dir=tmp_path / "locks",
        )


@pytest.mark.parametrize("mutation", ["bytes", "provenance"])
def test_db1_lock_source_compatibility_rejects_any_drift(
    tmp_path: Path, mutation: str
) -> None:
    historical = _historical_db1_campaign(tmp_path)
    if mutation == "bytes":
        historical.write_bytes(historical.read_bytes().replace(b'"n_fast": 16', b'"n_fast": 17'))
    else:
        payload = json.loads(historical.read_text(encoding="utf-8"))
        payload["provenance"]["executor"]["sha256"] = "sha256:" + "0" * 64
        payload.pop("contract_sha256")
        payload["contract_sha256"] = contract._digest_value(payload)  # noqa: SLF001
        historical.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(contract.ContractError):
        contract.validate_generation_campaign(
            historical, _allow_historical_lock_source=True
        )


def test_dual_arm_generation_campaign_is_exact_and_fail_closed() -> None:
    payload = contract.validate_generation_campaign(GENERATION_CAMPAIGN)
    arms = {arm["id"]: arm for arm in payload["arms"]}

    assert arms["n256"]["gpu_count"] == arms["n128"]["gpu_count"] == 28
    assert arms["n256"]["games_per_gpu"] == 2_000
    assert arms["n128"]["games_per_gpu"] == 5_000
    assert arms["n256"]["seed_start"] == 300_000_168_192
    assert arms["n256"]["seed_end"] == arms["n128"]["seed_start"]
    assert arms["n256"]["seed_block_size"] == 8_192
    assert arms["n128"]["seed_block_size"] == 8_192
    assert arms["n256"]["selected_per_gpu"] == {
        "current_producer": 1_600,
        "recent_history": 300,
        "hard_negative": 100,
    }
    assert arms["n128"]["selected_per_gpu"] == {
        "current_producer": 4_000,
        "recent_history": 750,
        "hard_negative": 250,
    }
    assert arms["n256"]["total_games"] == 56_000
    assert arms["n128"]["total_games"] == 140_000
    assignments = json.loads(
        (GENERATION_CAMPAIGN.parent / "placement.assignments.json").read_text()
    )["assignments"]
    assert len(assignments) == 56
    assert len({(item["host_alias"], item["gpu"]) for item in assignments}) == 56
    arms_by_host: dict[str, set[str]] = {}
    for item in assignments:
        arms_by_host.setdefault(item["host_alias"], set()).add(
            item["logical_lane"].split("_", 1)[0]
        )
    assert all(len(host_arms) == 1 for host_arms in arms_by_host.values())
    assert payload["common_recipe"]["p_full"] == 0.25
    assert payload["common_recipe"]["n_fast"] == 16
    assert payload["common_recipe"]["c_scale"] == 0.1
    assert payload["common_recipe"]["symmetry_averaged_eval_threshold"] == 20
    _, _, historical_generation = contract._campaign_science(  # noqa: SLF001
        payload, n_full=128
    )
    assert "native_mcts_hot_loop" not in historical_generation
    assert payload["execution_policy"]["launch_authorized"] is False
    with pytest.raises(contract.ContractError, match="not launchable"):
        contract.validate_generation_campaign(GENERATION_CAMPAIGN, require_ready=True)
    with pytest.raises(contract.ContractError, match="draft schema"):
        contract.build_lock(GENERATION_CAMPAIGN)


def test_dual_arm_r2_is_fresh_current_and_lineage_blocked() -> None:
    payload = contract.validate_generation_campaign(GENERATION_CAMPAIGN_R2)
    arms = {arm["id"]: arm for arm in payload["arms"]}

    assert payload["schema_version"] == contract.GENERATION_CAMPAIGN_REVISION_SCHEMA
    assert payload["contract_id"] == contract.GENERATION_CAMPAIGN_R2_CONTRACT_ID
    assert payload["contract_sha256"] == contract.GENERATION_CAMPAIGN_R2_CONTRACT_SHA256
    assert payload["implementation_commit"] == (
        contract.GENERATION_CAMPAIGN_REVISION_IMPLEMENTATION_COMMIT
    )
    assert payload["common_recipe"]["native_mcts_hot_loop"] is True
    assert payload["common_recipe"]["rust_featurize"] is True
    assert arms["n256"]["seed_start"] == contract.GENERATION_CAMPAIGN_R1_NEXT_SEED_FLOOR
    assert arms["n256"]["seed_end"] == arms["n128"]["seed_start"]
    assert payload["fleet"]["next_campaign_seed_floor"] == arms["n128"]["seed_end"]
    assert all("a1-dual-arm-20260711-r2" in arm["output_root"] for arm in arms.values())
    assert payload["supersedes"]["campaign_contract_sha256"] == (
        contract.GENERATION_CAMPAIGN_CONTRACT_SHA256
    )
    assert payload["promotion_handoff"] == {
        "mode": "required_post_promotion",
        "path": None,
        "expected_schema": "a1-post-promotion-producer-handoff-v1",
        "expected_checkpoint_sha256": payload["checkpoints"][0]["sha256"],
    }
    with pytest.raises(contract.ContractError, match="not launchable"):
        contract.validate_generation_campaign(
            GENERATION_CAMPAIGN_R2, require_ready=True
        )


def test_dual_arm_r2_binds_every_current_provenance_file() -> None:
    payload = contract.validate_generation_campaign(GENERATION_CAMPAIGN_R2)
    provenance = payload["provenance"]
    records = [
        *provenance["arm_guards"],
        *provenance["generator_code"],
        provenance["executor"],
        provenance["harvest"],
        provenance["fleet_manifest"],
    ]

    assert len({record["path"] for record in records}) == len(records)
    for record in records:
        assert record["sha256"] == contract._sha256_bytes(  # noqa: SLF001
            contract._git_blob(  # noqa: SLF001
                payload["implementation_commit"], record["path"]
            )
        )


def test_dual_arm_r2_placement_uses_fresh_lane_ids_and_all_56_gpus() -> None:
    payload = contract.validate_generation_campaign(GENERATION_CAMPAIGN_R2)
    assignments = json.loads(
        (GENERATION_CAMPAIGN_R2.parent / "placement.assignments.json").read_text()
    )["assignments"]

    assert len(assignments) == 56
    assert len({item["logical_lane"] for item in assignments}) == 56
    assert len({(item["host_alias"], item["gpu"]) for item in assignments}) == 56
    assert {item["logical_lane"] for item in assignments} == {
        lane for arm in payload["arms"] for lane in arm["logical_lanes"]
    }
    assert all(
        item["logical_lane"].startswith(("n128_gpu", "n256_gpu"))
        for item in assignments
    )


def test_dual_arm_r2_rejects_reusing_a_consumed_seed(tmp_path: Path) -> None:
    payload = json.loads(GENERATION_CAMPAIGN_R2.read_text())
    payload["arms"][0]["seed_start"] = contract.GENERATION_CAMPAIGN_R1_NEXT_SEED_FLOOR - 1
    payload["arms"][0]["seed_end"] = (
        payload["arms"][0]["seed_start"] + 28 * payload["arms"][0]["seed_block_size"]
    )
    payload.pop("contract_sha256")
    payload["contract_sha256"] = contract._digest_value(payload)  # noqa: SLF001
    path = tmp_path / "r2-reused-seed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(contract.ContractError, match="deterministic rebuild"):
        contract.validate_generation_campaign(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["arms"][1].update(
                seed_start=value["arms"][0]["seed_start"]
            ),
            "seed",
        ),
        (
            lambda value: value["arms"][1].update(
                output_root=value["arms"][0]["output_root"]
            ),
            "output roots overlap",
        ),
        (
            lambda value: value["common_recipe"].update(p_full=1.0),
            "common recipe drift",
        ),
        (
            lambda value: value["execution_policy"].update(launch_authorized=True),
            "execution policy drift",
        ),
        (
            lambda value: value["promotion_handoff"].update(
                mode="historical_pre_promotion"
            ),
            "handoff gate drift",
        ),
    ],
)
def test_dual_arm_generation_campaign_rejects_drift(
    tmp_path: Path, mutation, message: str
) -> None:
    path = _generation_campaign_copy(tmp_path, mutation)
    with pytest.raises(contract.ContractError, match=message):
        contract.validate_generation_campaign(path)


def test_dual_arm_generation_campaign_binds_immutable_tooling(
    tmp_path: Path,
) -> None:
    def drift(value: dict) -> None:
        value["provenance"]["executor"]["sha256"] = "sha256:" + "0" * 64

    path = _generation_campaign_copy(tmp_path, drift)
    with pytest.raises(contract.ContractError, match="immutable file drift"):
        contract.validate_generation_campaign(path)


def test_dual_arm_placement_refuses_split_hosts(tmp_path: Path) -> None:
    contract.validate_generation_campaign(GENERATION_CAMPAIGN)
    assignments = json.loads(
        (GENERATION_CAMPAIGN.parent / "placement.assignments.json").read_text()
    )["assignments"]
    n256 = next(item for item in assignments if item["logical_lane"] == "n256_gpu00")
    n128 = next(item for item in assignments if item["logical_lane"] == "n128_gpu00")
    n256["host_alias"], n128["host_alias"] = n128["host_alias"], n256["host_alias"]
    raw = tmp_path / "split-hosts.json"
    raw.write_text(json.dumps(assignments))
    out = tmp_path / "placement.json"

    with pytest.raises(contract.ContractError, match="may not split one host"):
        contract.seal_generation_placement(GENERATION_CAMPAIGN, raw, out)

    assert not out.exists()


@pytest.mark.parametrize("campaign_path", [GENERATION_CAMPAIGN, GENERATION_CAMPAIGN_R2])
def test_dual_arm_materializes_renders_and_replays_in_production_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, campaign_path: Path
) -> None:
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for index, item in enumerate(campaign["checkpoints"]):
        source = checkpoint_dir / f"checkpoint-{index}.pt"
        source.write_bytes(f"checkpoint-{index}".encode())
        item["path"] = str(source)
        item["sha256"] = contract._sha256(source)  # noqa: SLF001
    campaign["promotion_handoff"]["expected_checkpoint_sha256"] = campaign[
        "checkpoints"
    ][0]["sha256"]
    campaign["fleet"]["seed_ledger"] = str(tmp_path / "SEED_LEDGER.md")
    Path(campaign["fleet"]["seed_ledger"]).write_text("# ledger\n")
    for key in ("arm_guards", "generator_code"):
        for record in campaign["provenance"][key]:
            record["sha256"] = contract._sha256(  # noqa: SLF001
                contract.REPO_ROOT / record["path"]
            )
    for key in ("executor", "harvest", "fleet_manifest"):
        record = campaign["provenance"][key]
        record["sha256"] = contract._sha256(  # noqa: SLF001
            contract.REPO_ROOT / record["path"]
        )
    campaign.pop("contract_sha256")
    campaign["contract_sha256"] = contract._digest_value(campaign)  # noqa: SLF001
    monkeypatch.setattr(
        contract, "validate_generation_campaign", lambda _path, **_kwargs: campaign
    )
    if campaign_path == GENERATION_CAMPAIGN:
        monkeypatch.setattr(contract, "_runtime_code_tree_records", lambda: [])
    monkeypatch.setattr(contract, "_validate_against_ledger", lambda *_args: None)
    monkeypatch.setattr(contract, "_verify_live_seed_ledger", lambda *_args, **_kwargs: None)
    handoff = tmp_path / "handoff.json"
    handoff.write_text('{"handoff":"committed"}\n')
    monkeypatch.setattr(
        contract,
        "_promotion_handoff_record",
        lambda *_args, **_kwargs: {
            # This synthetic round-trip does not model a promotion receipt;
            # exact post-promotion identity enforcement has dedicated tests.
            "mode": contract.HISTORICAL_HANDOFF_MODE,
            "path": str(handoff),
            "sha256": contract._sha256(handoff),  # noqa: SLF001
        },
    )
    assignments = json.loads(
        (
            campaign_path.parent / "placement.assignments.json"
        ).read_text()
    )["assignments"]
    assignments_path = tmp_path / "assignments.json"
    assignments_path.write_text(json.dumps(assignments))
    placement_path = tmp_path / "placement.json"
    contract.seal_generation_placement(
        campaign_path, assignments_path, placement_path
    )

    locks = contract.materialize_generation_campaign(
        campaign_path,
        promotion_handoff_path=handoff,
        placement_path=placement_path,
        out_dir=tmp_path / "locks",
    )

    assert len(locks) == 2
    all_outputs: set[str] = set()
    all_ranges: set[tuple[int, int]] = set()
    for lock_path in locks:
        lock = contract.verify_lock(lock_path)
        arm_id = lock["game_contract"]["arm_id"]
        if campaign_path == GENERATION_CAMPAIGN_R2:
            runtime_records = {
                Path(record["path"]).resolve(): record["sha256"]
                for record in lock["provenance"]["runtime_code_tree"]
            }
            for relative in (
                "tools/a1_pre_wave_contract.py",
                "tools/a1_dual_arm_subsets.py",
                "tools/build_memmap_corpus.py",
                "tools/train_bc.py",
            ):
                source = (contract.REPO_ROOT / relative).resolve()
                assert runtime_records[source] == contract._sha256(source)  # noqa: SLF001
            assert lock["provenance"]["harvest"]["sha256"] == contract._sha256(  # noqa: SLF001
                contract.REPO_ROOT / "tools/fleet/a1_harvest_transaction.py"
            )
        render_dir = tmp_path / f"render-{arm_id}"
        rendered = contract.render(lock_path, render_dir)
        replayed_lock, replayed_render, lanes = production_executor.verify_render(
            lock_path, render_dir / "commands.json"
        )
        assert replayed_lock["game_contract"]["arm_id"] == arm_id
        assert replayed_render["render_sha256"] == rendered["render_sha256"]
        assert len(lanes) == 28
        assert len(rendered["commands"]) == 84
        assert lock["game_contract"]["total_complete_games"] == (
            56_000 if arm_id == "n256" else 140_000
        )
        for job in lock["fleet"]["jobs"]:
            assert job["output_dir"] not in all_outputs
            all_outputs.add(job["output_dir"])
            interval = (job["base_seed"], job["seed_end"])
            assert all(not (interval[0] < end and start < interval[1]) for start, end in all_ranges)
            all_ranges.add(interval)
        assert all(command["arm_id"] == arm_id for command in rendered["commands"])
        assert all(
            "--prelaunch-guard-config" in command["argv"]
            and command["argv"][command["argv"].index("--generation-arm-id") + 1]
            == arm_id
            for command in rendered["commands"]
        )
        for command in rendered["commands"]:
            expected_c_scale = (
                "0.1" if command["category"] == "current_producer" else "0.03"
            )
            assert command["argv"][command["argv"].index("--c-scale") + 1] == (
                expected_c_scale
            )
            guard_path = contract.REPO_ROOT / command["argv"][
                command["argv"].index("--prelaunch-guard-config") + 1
            ]
            guard = json.loads(guard_path.read_text())
            expected = guard["guards"][0]["args"]["expected_values"]
            assert str(expected["--c-scale"]) == expected_c_scale
            assert expected["--n-full"] == int(arm_id.removeprefix("n"))
            assert "--information-set-search" in command["argv"]
            assert "--public-observation" in command["argv"]
            assert "--symmetry-averaged-eval" in command["argv"]
            assert command["argv"][
                command["argv"].index("--symmetry-averaged-eval-threshold") + 1
            ] == "20"
        receipt = contract.claim_seed_ledger(
            lock_path,
            render_dir / "commands.json",
            tmp_path / f"{arm_id}.claim-receipt.json",
        )
        assert receipt["claim_count"] == 84
        assert all(
            command["ledger_claim"]["row"].startswith(
                f"[{next(job for job in lock['fleet']['jobs'] if job['job_id'] == command['job_id'])['base_seed']}"
            )
            for command in rendered["commands"]
        )
    assert len(all_outputs) == len(all_ranges) == 168


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
    # The large replay suite intentionally exercises immutable v2 behavior.
    # TEMPLATE is the current v3/64-GPU operator template and has dedicated
    # topology/provenance tests below.
    payload = json.loads(HISTORICAL_DRAFT.read_text(encoding="utf-8"))
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
            "information_set_search": True,
            "determinization_particles": 4,
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
    payload["generation"]["native_mcts_hot_loop"] = True
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
                                "--p-full",
                                "--base-seed",
                                "--games",
                                "--max-depth",
                                "--symmetry-averaged-eval",
                                "--symmetry-averaged-eval-threshold",
                                "--belief-chance-spectra",
                                "--information-set-search",
                                "--native-mcts-hot-loop",
                                "--determinization-particles",
                                "--determinization-min-simulations",
                            ],
                            "expected_values": {
                                "--c-scale": 0.03,
                                "--c-visit": 50.0,
                                "--n-full": 128,
                                "--n-fast": 16,
                                "--p-full": 0.4,
                                "--max-depth": 80,
                                "--temperature-decisions": 90,
                                "--public-observation": True,
                                "--lazy-interior-chance": True,
                                "--symmetry-averaged-eval": True,
                                "--symmetry-averaged-eval-threshold": 20,
                                "--belief-chance-spectra": False,
                                "--information-set-search": True,
                                "--native-mcts-hot-loop": True,
                                "--determinization-particles": 4,
                                "--determinization-min-simulations": 32,
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


def test_realized_search_identity_binds_category_c_scale(tmp_path: Path) -> None:
    _path, lock = _lock(tmp_path)
    base_job = dict(lock["fleet"]["jobs"][0])
    producer_job = {**base_job, "c_scale": 0.1}
    legacy_job = {**base_job, "c_scale": 0.03}
    producer = contract._job_search_identity(lock, producer_job)  # noqa: SLF001
    legacy = contract._job_search_identity(lock, legacy_job)  # noqa: SLF001
    assert producer["search_operator"]["c_scale"] == 0.1
    assert legacy["search_operator"]["c_scale"] == 0.03
    assert producer["search_operator_sha256"] != legacy["search_operator_sha256"]
    assert producer["effective_search_config_sha256"] != legacy["effective_search_config_sha256"]
    assert contract._job_attestation(lock, producer_job)["search_operator_sha256"] == producer["search_operator_sha256"]  # noqa: SLF001


@pytest.mark.parametrize("bad", [True, "0.03", 0.0, -0.1, float("inf"), float("nan")])
def test_realized_search_identity_rejects_invalid_c_scale(
    tmp_path: Path, bad: object
) -> None:
    _path, lock = _lock(tmp_path)
    job = {**lock["fleet"]["jobs"][0], "c_scale": bad}
    with pytest.raises(contract.ContractError, match="c_scale"):
        contract._job_search_identity(lock, job)  # noqa: SLF001


def test_legacy_job_attestation_remains_exact_but_is_not_new_identity(
    tmp_path: Path,
) -> None:
    _path, lock = _lock(tmp_path)
    job = {**lock["fleet"]["jobs"][0], "c_scale": 0.1}
    current = contract._job_attestation(lock, job)  # noqa: SLF001
    legacy = contract._legacy_job_attestation(lock, job)  # noqa: SLF001
    assert current["schema_version"] == "a1-generation-job-attestation-v3"
    assert legacy["schema_version"] == "a1-generation-job-attestation-v2"
    assert current["search_operator_sha256"] != legacy["search_operator_sha256"]
    assert legacy["search_operator_sha256"] == lock["science"]["search_operator_sha256"]


def test_checked_in_template_is_intentionally_unresolved_and_refuses_seal() -> None:
    payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    unresolved = contract._find_unresolved(payload)
    assert payload["science"]["search"]["n_full"] == 128
    assert payload["science"]["search"]["p_full"] == 0.25
    assert payload["science"]["search"]["c_scale"] == 0.1
    recipe = payload["science"]["learner_training_recipe"]
    assert recipe["amp"] == "none"
    assert recipe["forced_action_weight"] == 0.0
    assert recipe["forced_row_value_weight"] == 1.0
    assert recipe["per_game_policy_weight"] is True
    assert recipe["per_game_policy_weight_mode"] == "equal"
    assert recipe["training_rng_rank_offset"] is True
    assert recipe["per_game_value_weight"] is False
    assert "$.promotion_handoff.path" in unresolved
    assert payload["science"]["search"]["n_full_wide"] is None
    assert payload["science"]["search"]["n_full_wide_threshold"] is None
    assert payload["science"]["search"]["wide_roots_always_full"] is False
    assert "$.science.search.n_full_wide" not in unresolved
    assert "$.science.evaluator.value_readout" in unresolved
    assert "$.checkpoints[1].path" in unresolved
    assert "$.checkpoints[2].selection_evidence" in unresolved
    assert "$.fleet.seed_base" in unresolved
    with pytest.raises(contract.ContractError, match="finish A0/S1-S3"):
        contract.build_lock(TEMPLATE)


def test_v3_authoritative_fleet_balances_exact_64_gpu_quotas() -> None:
    workers, record = contract._canonical_workers_from_fleet_manifest(  # noqa: SLF001
        contract.CURRENT_FLEET_MANIFEST
    )
    quotas = contract._balanced_worker_quotas(workers)  # noqa: SLF001

    assert len(workers) == 64
    assert record["sha256"] == contract._sha256(  # noqa: SLF001
        contract.CURRENT_FLEET_MANIFEST
    )
    assert {category: sum(q[category] for q in quotas.values()) for category in contract.EXPECTED_GAMES} == contract.EXPECTED_GAMES
    ordered = [quotas[worker["id"]] for worker in workers]
    assert {quota["current_producer"] for quota in ordered} == {150}
    assert [quota["recent_history"] for quota in ordered].count(29) == 8
    assert [quota["recent_history"] for quota in ordered].count(28) == 56
    assert [quota["hard_negative"] for quota in ordered].count(10) == 24
    assert [quota["hard_negative"] for quota in ordered].count(9) == 40


def test_v3_balanced_jobs_have_exact_selected_and_bounded_attempt_totals(
    tmp_path: Path,
) -> None:
    workers, _ = contract._canonical_workers_from_fleet_manifest(  # noqa: SLF001
        contract.CURRENT_FLEET_MANIFEST
    )
    jobs, quotas = contract._build_balanced_jobs(  # noqa: SLF001
        workers,
        seed_base=90_000_000_000,
        block_size=1_000,
        output_root=str(tmp_path),
        contract_id="v3-test",
    )

    assert len(jobs) == 192
    assert Counter(
        {
            category: sum(job["games"] for job in jobs if job["category"] == category)
            for category in contract.EXPECTED_GAMES
        }
    ) == Counter(contract.EXPECTED_GAMES)
    assert {
        category: sum(job["attempts"] for job in jobs if job["category"] == category)
        for category in contract.EXPECTED_GAMES
    } == {
        category: total + 64 * contract.ATTEMPT_RESERVE_PER_JOB[category]
        for category, total in contract.EXPECTED_GAMES.items()
    }
    assert quotas["c1_gpu0"] == {
        "current_producer": 150,
        "recent_history": 29,
        "hard_negative": 10,
    }
    assert quotas["h100-8d_gpu7"] == {
        "current_producer": 150,
        "recent_history": 28,
        "hard_negative": 9,
    }


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


@pytest.mark.parametrize(
    "marker",
    [
        {"mode": "historical_pre_promotion", "reason": ""},
        {
            "mode": "historical_pre_promotion",
            "reason": "predates promotion",
            "unchecked": True,
        },
    ],
)
def test_verify_lock_refuses_malformed_historical_marker(
    tmp_path: Path, marker: dict
) -> None:
    _, lock = _lock(tmp_path)
    lock["promotion_handoff"] = marker
    unhashed = dict(lock)
    unhashed.pop("contract_sha256")
    lock["contract_sha256"] = contract._digest_value(unhashed)
    mutated = tmp_path / "mutated.lock.json"
    mutated.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    with pytest.raises(contract.ContractError, match="historical promotion_handoff"):
        contract.verify_lock(mutated)


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


def _replace_s2_s3_with_operator_bindings(draft: Path) -> tuple[Path, Path]:
    payload = json.loads(draft.read_text(encoding="utf-8"))
    search = payload["science"]["search"]
    search.update(
        {
            "n_full": 128,
            "n_fast": 16,
            "p_full": 0.25,
            "n_full_wide": None,
            "n_full_wide_threshold": None,
            "wide_roots_always_full": False,
        }
    )
    # This helper changes the winning production recipe, so keep the exact
    # prelaunch guard binding in lockstep.  A stale p_full=0.4 guard must not
    # make a valid p_full=0.25 operator-binding fixture look launchable.
    guard_path = Path(payload["provenance"]["guard_config"])
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    guard["guards"][0]["args"]["expected_values"]["--p-full"] = 0.25
    guard_path.write_text(json.dumps(guard) + "\n", encoding="utf-8")
    s1_path = _evidence_path(payload, "s1")
    s2_path = draft.parent / "s2.operator-binding.json"
    s3_path = draft.parent / "s3.operator-binding.json"
    operator_binding.write_bindings(
        s1_path,
        s2_path,
        s3_path,
        binding_time_utc="2026-07-10T04:10:00Z",
    )
    for item in payload["science"]["evidence"]:
        if item["kind"] == "s2":
            item["path"] = str(s2_path)
        elif item["kind"] == "s3":
            item["path"] = str(s3_path)
    draft.write_text(json.dumps(payload), encoding="utf-8")
    return s2_path, s3_path


def _rewrite_operator_binding(path: Path, mutate) -> None:
    path.chmod(0o600)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    unhashed = dict(payload)
    unhashed.pop("artifact_content_sha256", None)
    payload["artifact_content_sha256"] = operator_binding._digest_value(unhashed)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)


def test_point10_s1_operator_bridge_syncs_and_seals_exact_generation(
    tmp_path: Path,
) -> None:
    """Exercise the exact final-loop S1(.10) -> S2/S3 bridge -> guard path."""

    draft = _resolved_draft(tmp_path)
    _payload, _ = _select_s1_c_scale(draft, 0.1)
    _replace_s2_s3_with_operator_bindings(draft)

    sync = contract.sync_generation_guard(draft)
    assert sync["selected_c_scale"] == 0.1
    assert sync["status"] == "synchronized"

    lock = contract.build_lock(draft)
    assert lock["science"]["effective_search_config"]["c_scale"] == 0.1
    evidence = {row["kind"]: row for row in lock["science"]["evidence"]}
    assert evidence["s1"]["semantic_decision"]["selected_fields"]["c_scale"] == 0.1
    assert evidence["s2"]["semantic_decision"]["evidence_class"] == (
        operator_binding.ARTIFACT_KIND
    )
    assert evidence["s3"]["semantic_decision"]["evidence_class"] == (
        operator_binding.ARTIFACT_KIND
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


def test_seal_accepts_exact_n128_no_adaptive_operator_bindings(tmp_path: Path) -> None:
    draft = _resolved_draft(tmp_path)
    s2_path, s3_path = _replace_s2_s3_with_operator_bindings(draft)

    lock = contract.build_lock(draft)

    by_kind = {item["kind"]: item for item in lock["science"]["evidence"]}
    assert by_kind["s2"]["document_schema"] == operator_binding.SCHEMA
    assert by_kind["s3"]["document_schema"] == operator_binding.SCHEMA
    assert by_kind["s2"]["semantic_decision"]["evidence_class"] == (
        operator_binding.ARTIFACT_KIND
    )
    assert by_kind["s2"]["semantic_decision"]["selected_fields"] == {
        "n_full": 128,
        "n_fast": 16,
        "p_full": 0.25,
    }
    assert by_kind["s3"]["semantic_decision"]["selected_fields"] == {
        "n_full_wide": None,
        "n_full_wide_threshold": None,
        "wide_roots_always_full": False,
    }
    assert stat.S_IMODE(s2_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(s3_path.stat().st_mode) == 0o444


@pytest.mark.parametrize(
    ("stage", "mutation", "message"),
    [
        (
            "s2",
            lambda payload: payload.update(statement="pretend this proves strength"),
            "not strength evidence",
        ),
        (
            "s2",
            lambda payload: payload["selected_fields"].update(n_full=64),
            "must select exactly",
        ),
        (
            "s3",
            lambda payload: payload["selected_fields"].update(
                n_full_wide=256,
                n_full_wide_threshold=40,
                wide_roots_always_full=True,
            ),
            "must select exactly",
        ),
        (
            "s3",
            lambda payload: payload.update(reason="operator-authored free text"),
            "reason mismatch",
        ),
        (
            "s2",
            lambda payload: payload.update(binding_time_utc="2026-07-10T04:10:00-07:00"),
            "explicit UTC offset",
        ),
        (
            "s2",
            lambda payload: payload.update(operator="global_n64"),
            "operator mismatch",
        ),
        (
            "s2",
            lambda payload: payload.update(
                emitter={
                    "path": str(
                        contract.REPO_ROOT / "tools" / "search_teacher_adjudicator.py"
                    ),
                    "sha256": contract._sha256(
                        contract.REPO_ROOT / "tools" / "search_teacher_adjudicator.py"
                    ),
                }
            ),
            "untrusted emitter",
        ),
    ],
)
def test_seal_rejects_mutated_operator_binding_semantics(
    tmp_path: Path, stage: str, mutation, message: str
) -> None:
    draft = _resolved_draft(tmp_path)
    s2_path, s3_path = _replace_s2_s3_with_operator_bindings(draft)
    target = s2_path if stage == "s2" else s3_path
    _rewrite_operator_binding(target, mutation)

    with pytest.raises(contract.ContractError, match=message):
        contract.build_lock(draft)


def test_seal_rejects_operator_binding_self_digest(
    tmp_path: Path,
) -> None:
    draft = _resolved_draft(tmp_path)
    s2_path, _ = _replace_s2_s3_with_operator_bindings(draft)
    s2_path.chmod(0o600)
    s2 = json.loads(s2_path.read_text(encoding="utf-8"))
    s2["artifact_content_sha256"] = "sha256:" + "0" * 64
    s2_path.write_text(json.dumps(s2), encoding="utf-8")
    s2_path.chmod(0o444)
    with pytest.raises(contract.ContractError, match="self digest mismatch"):
        contract.build_lock(draft)


def test_s1_cannot_use_operator_binding_schema(tmp_path: Path) -> None:
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text(encoding="utf-8"))
    s2_path, _ = _replace_s2_s3_with_operator_bindings(draft)
    payload = json.loads(draft.read_text(encoding="utf-8"))
    for item in payload["science"]["evidence"]:
        if item["kind"] == "s1":
            item["path"] = str(s2_path)
    draft.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(contract.ContractError, match="S1 cannot use"):
        contract.build_lock(draft)


def test_seal_rejects_operator_binding_swapped_s1(tmp_path: Path) -> None:
    # Rebuild in a clean directory and point S2 at a byte-identical but
    # different S1 path. It still replays, but must not inherit the exact S1
    # decision named by the A1 contract.
    draft = _resolved_draft(tmp_path)
    payload = json.loads(draft.read_text(encoding="utf-8"))
    s2_path, s3_path = _replace_s2_s3_with_operator_bindings(draft)
    real_s1 = _evidence_path(payload, "s1")
    fake_s1 = tmp_path / "copied-s1.decision.json"
    fake_s1.write_bytes(real_s1.read_bytes())
    _rewrite_operator_binding(
        s2_path,
        lambda binding: binding.update(
            source_s1={"path": str(fake_s1), "sha256": contract._sha256(fake_s1)}
        ),
    )
    _rewrite_operator_binding(
        s3_path,
        lambda binding: binding.update(
            source_s2_binding={
                "path": str(s2_path),
                "sha256": contract._sha256(s2_path),
            }
        ),
    )
    with pytest.raises(contract.ContractError, match="exact S1 lineage"):
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
    assert current["environment"]["CATAN_ZERO_CONFIG_REGISTRY"] == str(
        Path(lock["fleet"]["jobs"][0]["output_dir"]) / "config_registry.jsonl"
    )
    assert current["environment_sha256"] == contract._digest_value(
        current["environment"]
    )
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
    assert "--native-mcts-hot-loop" in current["argv"]
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


def test_claim_transaction_atomically_installs_all_rows_and_is_idempotent(
    tmp_path: Path,
) -> None:
    lock_path, lock = _lock(tmp_path)
    rendered_dir = tmp_path / "rendered"
    rendered = contract.render(lock_path, rendered_dir)
    ledger = Path(lock["fleet"]["seed_ledger"]["path"])
    before = ledger.read_bytes()
    receipt_path = tmp_path / "claims.receipt.json"

    receipt = contract.claim_seed_ledger(
        lock_path, rendered_dir / "commands.json", receipt_path
    )
    after = ledger.read_bytes()
    assert receipt["status"] == "claimed"
    assert receipt["claim_count"] == 120
    assert receipt["render_sha256"] == rendered["render_sha256"]
    assert after.startswith(before) and after != before
    assert contract.verify_lock(
        lock_path, require_all_job_claims=True
    )["contract_sha256"] == lock["contract_sha256"]
    assert receipt_path.stat().st_mode & stat.S_IWUSR == 0

    # Re-entry validates the immutable receipt and exact ledger prefix. It
    # neither appends duplicates nor rewrites the receipt.
    receipt_bytes = receipt_path.read_bytes()
    assert (
        contract.claim_seed_ledger(
            lock_path, rendered_dir / "commands.json", receipt_path
        )
        == receipt
    )
    assert ledger.read_bytes() == after
    assert receipt_path.read_bytes() == receipt_bytes

    # A later disjoint append is allowed; the receipt binds and rechecks the
    # exact post-transaction prefix rather than incorrectly freezing the
    # shared ledger forever.
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write("[900000000000 – 900000000001) | later-disjoint |\n")
    later = ledger.read_bytes()
    assert contract.claim_seed_ledger(
        lock_path, rendered_dir / "commands.json", receipt_path
    ) == receipt
    assert ledger.read_bytes() == later


def test_claim_transaction_recovers_all_exact_rows_without_a_receipt(
    tmp_path: Path,
) -> None:
    lock_path, lock = _lock(tmp_path)
    rendered_dir = tmp_path / "rendered"
    contract.render(lock_path, rendered_dir)
    _append_job_claims(lock)
    ledger = Path(lock["fleet"]["seed_ledger"]["path"])
    before = ledger.read_bytes()
    receipt_path = tmp_path / "recovered.receipt.json"

    receipt = contract.claim_seed_ledger(
        lock_path, rendered_dir / "commands.json", receipt_path
    )
    assert receipt["status"] == "already_claimed"
    assert receipt["claim_count"] == 120
    assert ledger.read_bytes() == before
    assert receipt_path.is_file()


def test_claim_transaction_refuses_partial_own_set_without_mutation(
    tmp_path: Path,
) -> None:
    lock_path, lock = _lock(tmp_path)
    rendered_dir = tmp_path / "rendered"
    contract.render(lock_path, rendered_dir)
    _append_job_claims(lock, [lock["fleet"]["jobs"][0]])
    ledger = Path(lock["fleet"]["seed_ledger"]["path"])
    before = ledger.read_bytes()
    receipt = tmp_path / "claims.receipt.json"

    with pytest.raises(contract.ContractError, match="partial own claim set"):
        contract.claim_seed_ledger(
            lock_path, rendered_dir / "commands.json", receipt
        )
    assert ledger.read_bytes() == before
    assert not receipt.exists()


def test_claim_transaction_rejects_render_drift_and_spoofed_live_claim(
    tmp_path: Path,
) -> None:
    lock_path, lock = _lock(tmp_path)
    rendered_dir = tmp_path / "rendered"
    contract.render(lock_path, rendered_dir)
    commands_path = rendered_dir / "commands.json"
    os.chmod(commands_path, 0o644)
    payload = json.loads(commands_path.read_text(encoding="utf-8"))
    payload["commands"][0]["ledger_claim"]["row"] += " drift"
    commands_path.write_text(json.dumps(payload), encoding="utf-8")
    ledger = Path(lock["fleet"]["seed_ledger"]["path"])
    before = ledger.read_bytes()
    with pytest.raises(contract.ContractError, match="render semantic digest"):
        contract.claim_seed_ledger(
            lock_path, commands_path, tmp_path / "drift.receipt.json"
        )
    assert ledger.read_bytes() == before

    # Restore a valid render, then show that a claim naming this contract with
    # the wrong row is rejected by ordinary live-ledger verification first.
    commands_path.unlink()
    for child in rendered_dir.rglob("*"):
        if child.is_file():
            os.chmod(child, 0o644)
    import shutil

    shutil.rmtree(rendered_dir)
    contract.render(lock_path, rendered_dir)
    first = lock["fleet"]["jobs"][0]
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{first['base_seed']} – {first['seed_end']}) | "
            f"claim={first['claim_label']} contract={'sha256:' + '0' * 64} "
            "job=WRONG |\n"
        )
    spoofed = ledger.read_bytes()
    with pytest.raises(contract.ContractError, match="does not exactly match"):
        contract.claim_seed_ledger(
            lock_path,
            rendered_dir / "commands.json",
            tmp_path / "spoof.receipt.json",
        )
    assert ledger.read_bytes() == spoofed


def test_claim_transaction_replace_failure_leaves_original_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path, lock = _lock(tmp_path)
    rendered_dir = tmp_path / "rendered"
    contract.render(lock_path, rendered_dir)
    ledger = Path(lock["fleet"]["seed_ledger"]["path"])
    before = ledger.read_bytes()
    receipt = tmp_path / "claims.receipt.json"

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(contract.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        contract.claim_seed_ledger(
            lock_path, rendered_dir / "commands.json", receipt
        )
    assert ledger.read_bytes() == before
    assert not receipt.exists()
    assert not list(ledger.parent.glob(f".{ledger.name}.claim-*.tmp"))


def test_claim_transaction_rejects_mutated_existing_receipt(tmp_path: Path) -> None:
    lock_path, _lock_payload = _lock(tmp_path)
    rendered_dir = tmp_path / "rendered"
    contract.render(lock_path, rendered_dir)
    receipt_path = tmp_path / "claims.receipt.json"
    contract.claim_seed_ledger(
        lock_path, rendered_dir / "commands.json", receipt_path
    )
    os.chmod(receipt_path, 0o644)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["status"] = "forged"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(contract.ContractError, match="semantic digest"):
        contract.claim_seed_ledger(
            lock_path, rendered_dir / "commands.json", receipt_path
        )


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


def test_raw_game_decisions_reject_adjacent_duplicate_reset() -> None:
    with pytest.raises(contract.ContractError, match="adjacent duplicate game"):
        contract._advance_game_decision_run(
            np.asarray([17, 17, 17, 17], dtype=np.int64),
            np.asarray([0, 1, 0, 1], dtype=np.int32),
            active_seed=None,
            active_decision_index=None,
            where="shard-0",
        )


def test_raw_game_decisions_track_monotonic_cross_shard_continuation() -> None:
    last_decision = contract._advance_game_decision_run(
        np.asarray([17, 17], dtype=np.int64),
        np.asarray([0, 3], dtype=np.int32),
        active_seed=None,
        active_decision_index=None,
        where="shard-0",
    )
    last_decision = contract._advance_game_decision_run(
        np.asarray([17, 17], dtype=np.int64),
        np.asarray([5, 8], dtype=np.int32),
        active_seed=17,
        active_decision_index=last_decision,
        where="shard-1",
    )
    assert last_decision == 8
    with pytest.raises(contract.ContractError, match="shard boundary"):
        contract._advance_game_decision_run(
            np.asarray([17], dtype=np.int64),
            np.asarray([0], dtype=np.int32),
            active_seed=17,
            active_decision_index=last_decision,
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


def test_selected_opponent_rows_require_only_deterministic_producer_seat(
    tmp_path: Path,
) -> None:
    seeds = np.asarray([100, 100, 101, 101], dtype=np.int64)
    expected_players = np.asarray(
        [
            "RED"
            if contract._pool_champion_plays_first_seat(int(seed) - 100)  # noqa: SLF001
            else "BLUE"
            for seed in seeds
        ],
        dtype="U8",
    )
    arrays = {
        "game_seed": seeds,
        "is_pool_game": np.ones(seeds.size, dtype=bool),
        "opponent_version": np.full(seeds.size, 6, dtype=np.int32),
        "player": expected_players,
        "seat": np.asarray(
            [contract.PLAYER_NAMES.index(str(value)) for value in expected_players],
            dtype=np.int8,
        ),
    }
    good = tmp_path / "good.npz"
    np.savez(good, **arrays)
    with np.load(good, allow_pickle=False) as payload:
        contract._validate_selected_opponent_rows(  # noqa: SLF001
            payload,
            selected_mask=np.ones(seeds.size, dtype=bool),
            game_seeds=seeds,
            job={"base_seed": 100},
            allowed_versions={6},
            colors=("RED", "BLUE"),
        )

    # Simulate the exact ingestion bug this guard closes: rows from both seats
    # carry the same game-level opponent tag/hash, so only player/seat proves
    # the archived opponent's decisions were excluded from policy targets.
    bad = dict(arrays)
    bad["player"] = expected_players.copy()
    bad["player"][0] = "BLUE" if expected_players[0] == "RED" else "RED"
    bad_path = tmp_path / "unfiltered.npz"
    np.savez(bad_path, **bad)
    with np.load(bad_path, allow_pickle=False) as payload:
        with pytest.raises(
            contract.ContractError, match="non-producer-seat policy targets"
        ):
            contract._validate_selected_opponent_rows(  # noqa: SLF001
                payload,
                selected_mask=np.ones(seeds.size, dtype=bool),
                game_seeds=seeds,
                job={"base_seed": 100},
                allowed_versions={6},
                colors=("RED", "BLUE"),
            )


def test_create_or_verify_readonly_reuses_only_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "sidecar.json"
    payload = {"schema_version": "fixture-v1", "value": 7}
    contract._create_or_verify_readonly(path, payload)
    contract._create_or_verify_readonly(path, payload)
    path.chmod(0o644)
    path.write_text('{"different":true}\n', encoding="utf-8")
    with pytest.raises(contract.ContractError, match="differs"):
        contract._create_or_verify_readonly(path, payload)


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


def test_single_read_registry_evidence_rejects_in_place_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "config_registry.jsonl"
    registry.write_bytes(b"x" * 128)
    registry.chmod(0o444)
    original_read = contract.os.read
    changed = False

    def racing_read(descriptor: int, size: int) -> bytes:
        nonlocal changed
        payload = original_read(descriptor, size)
        if payload and not changed:
            changed = True
            registry.chmod(0o644)
            registry.write_bytes(b"y" * 128)
        return payload

    monkeypatch.setattr(contract.os, "read", racing_read)
    with pytest.raises(contract.ContractError, match="mutated during read"):
        contract._read_sealed_regular(registry, where="race-test")


def test_post_wave_audit_accepts_exact_complete_category_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
            "target_information_regime": np.full(
                n, "public_conservation_pimc_v1", dtype="U32"
            ),
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
            arrays["is_pool_game"] = np.ones(n, dtype=bool)
            arrays["opponent_version"] = np.full(
                n, int(opponent.get("version", -1)), dtype=np.int32
            )
            players = np.asarray(
                [
                    "RED"
                    if contract._pool_champion_plays_first_seat(index)  # noqa: SLF001
                    else "BLUE"
                    for index in range(n)
                ],
                dtype="U8",
            )
            arrays["player"] = players
            arrays["seat"] = np.asarray(
                [contract.PLAYER_NAMES.index(str(player)) for player in players],
                dtype=np.int8,
            )
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
        cli["producer_checkpoint_sha256"] = contract._producer(lock)["sha256"]
        worker = out_dir / "worker_000" / "manifest.json"
        worker.parent.mkdir()
        worker.write_text(
            json.dumps(
                {
                    "search_config": {
                        **contract._job_search_identity(lock, job)[  # noqa: SLF001
                            "effective_search_config"
                        ],
                        "seed": 123,
                    },
                    "selfplay_config": contract._expected_selfplay_config(lock),
                    "target_information_regime": "public_conservation_pimc_v1",
                }
            ),
            encoding="utf-8",
        )
        typed_config = contract.GenerateConfig.from_namespace(Namespace(**cli))
        config_hash = typed_config.config_hash()
        registry_path = out_dir / "config_registry.jsonl"
        registry_path.write_text(
            json.dumps(
                {
                    "config_hash": config_hash,
                    "full_config_hash": typed_config.full_config_hash(),
                    "pipeline": "generate",
                    "timestamp": "2026-07-10T00:00:00+00:00",
                    "purpose": "test",
                    "config": typed_config.canonical_payload(),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        registry_path.chmod(0o444)
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
                    "target_information_regime": "public_conservation_pimc_v1",
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
    assert report["target_information_regime"] == {
        "required": "public_conservation_pimc_v1",
        "counts": {
            "public_conservation_pimc_v1": sum(contract.EXPECTED_ATTEMPTS.values())
        },
    }

    original_create = contract._create_or_verify_readonly
    for crash_after in (1, 2, 3):
        crash_report = tmp_path / f"audit.crash-{crash_after}.json"
        calls = 0

        def crash_after_write(path: Path, payload: dict) -> None:
            nonlocal calls
            calls += 1
            original_create(path, payload)
            if calls == crash_after:
                raise RuntimeError(f"injected crash after artifact {crash_after}")

        monkeypatch.setattr(contract, "_create_or_verify_readonly", crash_after_write)
        with pytest.raises(RuntimeError, match="injected crash"):
            contract.audit_outputs(lock_path, crash_report)
        monkeypatch.setattr(contract, "_create_or_verify_readonly", original_create)
        replayed = contract.audit_outputs(lock_path, crash_report)
        assert replayed["passed"] is True

    # The acceptance scanner binds the planner's information regime at the
    # row boundary, not merely via a top-level manifest assertion.  Corrupt a
    # reserve row and prove that even data excluded from training selection is
    # rejected rather than silently carried alongside the accepted corpus.
    first_shard = Path(lock["fleet"]["jobs"][0]["output_dir"]) / "shard_00000.npz"
    with np.load(first_shard, allow_pickle=False) as payload:
        corrupted = {key: np.asarray(payload[key]) for key in payload.files}
    corrupted["target_information_regime"] = corrupted[
        "target_information_regime"
    ].copy()
    corrupted["target_information_regime"][-1] = (
        "authoritative_hidden_state_search_v1"
    )
    np.savez(first_shard, **corrupted)
    with pytest.raises(contract.ContractError, match="post-wave audit failed"):
        contract.audit_outputs(lock_path, tmp_path / "audit.unsafe.json")
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
    registry_evidence = [
        item for item in report["shards"] if item["kind"] == "config_registry"
    ]
    assert len(registry_evidence) == 120
    assert all(item["config_hash"] for item in registry_evidence)
    first_registry = (
        Path(lock["fleet"]["jobs"][0]["output_dir"]) / "config_registry.jsonl"
    )
    original_registry = first_registry.read_text(encoding="utf-8")
    first_registry.chmod(0o644)
    first_registry.write_text(
        '{"config_hash":"sha256:wrong","pipeline":"generate"}\n',
        encoding="utf-8",
    )
    first_registry.chmod(0o444)
    bad_registry_report = tmp_path / "audit.bad-registry.json"
    with pytest.raises(contract.ContractError, match="post-wave audit failed"):
        contract.audit_outputs(lock_path, bad_registry_report)
    assert any(
        "config registry record fields drift" in error
        for error in json.loads(bad_registry_report.read_text())["errors"]
    )
    first_registry.chmod(0o644)
    first_registry.write_text(original_registry, encoding="utf-8")
    first_registry.chmod(0o444)
    first_manifest = json.loads(
        (Path(lock["fleet"]["jobs"][0]["output_dir"]) / "manifest.json").read_text()
    )
    first_worker = Path(first_manifest["worker_summaries"][0])
    worker_payload = json.loads(first_worker.read_text(encoding="utf-8"))
    worker_payload["target_information_regime"] = (
        "authoritative_hidden_state_search_v1"
    )
    first_worker.write_text(json.dumps(worker_payload), encoding="utf-8")
    bad_worker_report = tmp_path / "audit.bad-worker-regime.json"
    with pytest.raises(contract.ContractError, match="post-wave audit failed"):
        contract.audit_outputs(lock_path, bad_worker_report)
    assert any(
        "worker manifest target_information_regime" in error
        for error in json.loads(bad_worker_report.read_text())["errors"]
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


def _promoted_job_identity_fixture(*, deployed_c_scale: float, job_c_scale: float):
    checkpoint = {"role": "producer", "path": "/tmp/producer.pt", "sha256": "sha256:" + "1" * 64}
    deployed = {"c_scale": deployed_c_scale, "n_full": 128}
    lock = {
        "promotion_handoff": {
            "mode": contract.POST_PROMOTION_HANDOFF_MODE,
            "document_schema": contract.promotion_handoff.HANDOFF_SCHEMA,
            "producer_checkpoint": {
                "path": checkpoint["path"],
                "sha256": checkpoint["sha256"],
            },
            "producer_identity_sha256": "sha256:" + "2" * 64,
            "producer_search_config": deployed,
            "producer_search_config_sha256": contract._digest_value(deployed),
        },
        "checkpoints": [checkpoint],
        "science": {"search_operator": {"c_scale": deployed_c_scale, "n_full": 256}},
    }
    return lock, {"job_id": "n256_gpu00__recent_history", "c_scale": job_c_scale}


def test_promoted_producer_job_binds_checkpoint_and_executed_operator() -> None:
    lock, job = _promoted_job_identity_fixture(deployed_c_scale=0.10, job_c_scale=0.10)
    identity = contract._promoted_producer_job_identity(lock, job)
    assert identity is not None
    assert identity["checkpoint"] == {
        "path": "/tmp/producer.pt",
        "sha256": "sha256:" + "1" * 64,
    }
    assert identity["executed_search_operator"] == {"c_scale": 0.10, "n_full": 256}
    assert identity["checkpoint_search_identity_sha256"].startswith("sha256:")


def test_promoted_producer_job_refuses_deployed_c_scale_mismatch() -> None:
    lock, job = _promoted_job_identity_fixture(deployed_c_scale=0.10, job_c_scale=0.03)
    with pytest.raises(
        contract.ContractError,
        match=r"executes c_scale=0\.03.*deployed at c_scale=0\.1",
    ):
        contract._promoted_producer_job_identity(lock, job)
