from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.fleet import a1_n256_lr_eval as trial


def _manifest(tmp_path: Path) -> Path:
    value = {
        "schema_version": "a1-h100-eval-fleet-manifest-v1",
        "ssh_user": "ubuntu",
        "ssh_key": str(tmp_path / "id_ed25519"),
        "strict_host_key_checking": "accept-new",
        "remote_repo": "/home/ubuntu/catan-zero-v1",
        "remote_python": "/home/ubuntu/catan-zero-v1/.venv/bin/python",
        "remote_root": "/home/ubuntu/a1-evaluation",
        "validation_ledger": str(tmp_path / "VAL_ONLY_EVAL_LEDGER.jsonl"),
        "hosts": [
            {
                "alias": alias,
                "address": address,
                "gpu_count": count,
            }
            for alias, (address, count) in trial.fleet.FULL_EXPECTED_HOSTS.items()
        ],
    }
    path = tmp_path / "fleet.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _receipts(tmp_path: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    receipts = {}
    checkpoints = {}
    for label in trial.ARM_SPECS:
        checkpoint = tmp_path / f"{label}.pt"
        checkpoint.write_bytes(label.encode())
        receipt = tmp_path / f"{label}.receipt.json"
        receipt.write_text("{}", encoding="utf-8")
        checkpoints[label] = checkpoint
        receipts[label] = receipt
    return receipts, checkpoints


def _fake_receipts(
    monkeypatch: pytest.MonkeyPatch,
    receipts: dict[str, Path],
    checkpoints: dict[str, Path],
) -> Path:
    by_path = {path.resolve(): label for label, path in receipts.items()}
    parent = next(iter(receipts.values())).parent / "f7.pt"
    parent.write_bytes(b"f7-parent")

    def verify(path: Path) -> dict:
        label = by_path[path.resolve()]
        lr, ablation = trial.ARM_SPECS[label]
        return {
            "arm_id": "n256",
            "subset_id": "full-56k",
            "inputs": {
                "producer": {
                    "path": str(parent.resolve()),
                    "sha256": trial.fleet._sha256(parent),  # noqa: SLF001
                },
                "learner_ablation": {
                    "ablation_id": ablation,
                    "diagnostic_only": True,
                    "promotion_eligible": False,
                    "effective_recipe": {
                        "lr": lr,
                        "loser_sample_weight": 1.0,
                        "epochs": 1,
                    },
                }
            },
            "outputs": {
                "checkpoint": {
                    "path": str(checkpoints[label].resolve()),
                    "sha256": trial.fleet._sha256(checkpoints[label]),  # noqa: SLF001
                }
            },
        }

    monkeypatch.setattr(trial.training, "verify_receipt", verify)
    return parent


def _registry(tmp_path: Path, champion: Path) -> Path:
    path = tmp_path / "registry.json"
    registry = trial.ChampionRegistry(path)
    search_config = {"c_scale": 0.10}
    checkpoint = trial.fleet._checkpoint_ref(champion)  # noqa: SLF001
    identity = trial.fleet._digest(  # noqa: SLF001
        {
            "schema_version": "a1-deployed-agent-search-config-v1",
            "checkpoint": checkpoint,
            "search_config": search_config,
        }
    )
    registry.set_role(
        "generator_champion",
        champion,
        version=4,
        provenance={
            "a1_candidate_agent_identity_sha256": identity,
            "a1_candidate_search_config": search_config,
        },
    )
    registry.save()
    return path


def test_trial_refuses_before_all_three_receipts_exist(tmp_path: Path) -> None:
    receipts, checkpoints = _receipts(tmp_path)
    receipts["lr240u"].unlink()
    champion = tmp_path / "champion.pt"
    champion.write_bytes(b"champion")
    registry = _registry(tmp_path, champion)
    with pytest.raises(FileNotFoundError):
        trial.build_trial(
            manifest_path=_manifest(tmp_path),
            champion=champion,
            registry_path=registry,
            receipts=receipts,
            internal_base_seed=6_190_100_000,
            external_base_seed=6_190_200_000,
            trial_id="n256-lr-micro",
            output_dir=tmp_path / "trial",
            candidate_c_scale=0.10,
            champion_c_scale=0.10,
        )
    assert not (tmp_path / "trial").exists()
    assert checkpoints["lr240u"].is_file()


def test_three_arm_trial_is_exactly_matched_and_diagnostic_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipts, checkpoints = _receipts(tmp_path)
    champion = _fake_receipts(monkeypatch, receipts, checkpoints)
    registry = _registry(tmp_path, champion)
    output = tmp_path / "trial"
    result = trial.build_trial(
        manifest_path=_manifest(tmp_path),
        champion=champion,
        registry_path=registry,
        receipts=receipts,
        internal_base_seed=6_190_100_000,
        external_base_seed=6_190_200_000,
        trial_id="n256-lr-micro",
        output_dir=output,
        candidate_c_scale=0.10,
        champion_c_scale=0.10,
    )

    assert result["diagnostic_only"] is True
    assert result["promotion_eligible"] is False
    assert result["micro_panel"] == {"internal_pairs": 112, "external_pairs": 56}
    assert result["manifest"]["physical_gpus"] == 64
    plans = {
        label: json.loads((output / f"{label}.plan.json").read_text())
        for label in trial.ARM_SPECS
    }
    assert len({plan["seed_cohort_id"] for plan in plans.values()}) == 1
    assert len({json.dumps(plan["pair_claims"], sort_keys=True) for plan in plans.values()}) == 1
    assert len({plan["science_config_hash"] for plan in plans.values()}) == 1
    assert all(plan["pair_claims"]["internal"]["pairs"] == 112 for plan in plans.values())
    assert all(
        plan["pair_claims"]["external_matched"]["pairs"] == 56
        for plan in plans.values()
    )
    for plan in plans.values():
        assert len([job for job in plan["jobs"] if job["phase"] == "internal"]) == 64
        assert len([job for job in plan["jobs"] if job["phase"] == "external"]) == 64
        for job in plan["jobs"]:
            argv = job["argv"]
            assert argv[argv.index("--n-full") + 1] == "128"
            assert "--information-set-search" in argv
            assert "--symmetry-averaged-eval" in argv
            assert "--evaluator-rust-featurize" in argv
            assert "--native-mcts-hot-loop" in argv

    commands = trial.render_commands(output / "trial.json")
    assert commands["diagnostic_only"] is True
    assert [(row["arm"], row["phase"]) for row in commands["commands"]] == [
        (label, phase)
        for label in trial.ARM_SPECS
        for phase in ("internal", "external")
    ]
    assert all(row["launch"][-1] == "--go" for row in commands["commands"])


def test_n256_receipt_parent_f7_rejects_default_gen3_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipts, checkpoints = _receipts(tmp_path)
    f7 = _fake_receipts(monkeypatch, receipts, checkpoints)
    registry = _registry(tmp_path, f7)
    gen3 = tmp_path / "gen3.pt"
    gen3.write_bytes(b"gen3")
    with pytest.raises(
        trial.fleet.FleetError,
        match="promotion baseline differs from candidate parent/init checkpoint",
    ):
        trial.build_trial(
            manifest_path=_manifest(tmp_path),
            champion=gen3,
            registry_path=registry,
            receipts=receipts,
            internal_base_seed=6_190_100_000,
            external_base_seed=6_190_200_000,
            trial_id="n256-wrong-gen3-baseline",
            output_dir=tmp_path / "trial",
            candidate_c_scale=0.10,
            champion_c_scale=0.03,
        )


def test_receipt_recipe_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipts, checkpoints = _receipts(tmp_path)
    champion = _fake_receipts(monkeypatch, receipts, checkpoints)
    original = trial.training.verify_receipt

    def drift(path: Path) -> dict:
        value = original(path)
        if path.resolve() == receipts["lr240u"].resolve():
            value["inputs"]["learner_ablation"]["effective_recipe"]["lr"] = 0.00012
        return value

    monkeypatch.setattr(trial.training, "verify_receipt", drift)
    registry = _registry(tmp_path, champion)
    with pytest.raises(trial.TrialError, match="wrong effective recipe"):
        trial.build_trial(
            manifest_path=_manifest(tmp_path),
            champion=champion,
            registry_path=registry,
            receipts=receipts,
            internal_base_seed=6_190_100_000,
            external_base_seed=6_190_200_000,
            trial_id="n256-lr-micro",
            output_dir=tmp_path / "trial",
            candidate_c_scale=0.10,
            champion_c_scale=0.10,
        )
