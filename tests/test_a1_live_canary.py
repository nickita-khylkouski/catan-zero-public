from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.fleet import a1_live_canary as canary


def _source_argv(out_dir: str, base_seed: int, claim: str) -> list[str]:
    return [
        "tools/generate_gumbel_selfplay_data.py",
        "--out-dir",
        out_dir,
        "--games",
        "245",
        "--workers",
        "16",
        "--checkpoint",
        "/models/champion.pt",
        "--device",
        "cuda",
        "--n-full",
        "128",
        "--n-fast",
        "16",
        "--p-full",
        "0.25",
        "--c-visit",
        "50.0",
        "--c-scale",
        "0.03",
        "--rescale-noise-floor-c",
        "0.0",
        "--sigma-eval",
        "0.98",
        "--wide-candidates-threshold",
        "24",
        "--max-depth",
        "80",
        "--base-seed",
        str(base_seed),
        "--ledger-claim-label",
        claim,
        "--symmetry-averaged-eval-threshold",
        "20",
        "--determinization-particles",
        "4",
        "--determinization-min-simulations",
        "32",
        "--symmetry-averaged-eval",
        "--no-wide-roots-always-full",
        "--lazy-interior-chance",
        "--no-belief-chance-spectra",
        "--information-set-search",
        "--public-observation",
        "--no-eval-server",
        "--seed-claim",
        "--resume",
    ]


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(canary.executor, "_repo_artifacts", lambda *_a, **_k: [])
    lock_path = tmp_path / "lock.json"
    render_path = tmp_path / "render.json"
    hosts_path = tmp_path / "hosts.json"
    receipt_path = tmp_path / "receipt.json"
    for path in (lock_path, render_path, hosts_path):
        path.write_text("{}\n", encoding="utf-8")
    os.chmod(hosts_path, 0o600)
    production_ledger = tmp_path / "PRODUCTION_LEDGER.md"
    production_ledger.write_text("# production\n", encoding="utf-8")
    lock = {
        "contract_sha256": "sha256:" + "a" * 64,
        "science": {"evaluator": {"value_readout": "scalar"}},
        "checkpoints": [
            {
                "role": "producer",
                "metadata": {
                    "mask_hidden_info": True,
                    "legacy_scalar_readout_attestation": {
                        "schema_version": "legacy-scalar-readout-attestation-v1"
                    },
                },
            }
        ],
        "fleet": {"output_root": "/home/ubuntu/catan-zero-production/runs/selfplay"},
    }
    aliases = {
        "c1": 4,
        "c2": 4,
        "c3": 4,
        "c4": 4,
        "c5": 4,
        "c6": 4,
        "h100-8a": 8,
        "h100-8b": 8,
    }
    lanes: dict[str, list[dict]] = {}
    commands: list[dict] = []
    counter = 0
    for alias, count in aliases.items():
        for gpu in range(count):
            worker = f"{alias.replace('h100-', 'h')}_gpu{gpu}"
            lane = []
            previous = None
            for category in canary.CATEGORY_ORDER:
                job = f"{worker}__{category}"
                source_attestation = tmp_path / "attestations" / f"{job}.json"
                source_attestation.parent.mkdir(exist_ok=True)
                payload = {"job_id": job, "contract_sha256": lock["contract_sha256"]}
                source_attestation.write_text(
                    json.dumps(payload) + "\n", encoding="utf-8"
                )
                argv = _source_argv(
                    f"/home/ubuntu/catan-zero-production/runs/selfplay/{job}",
                    300_000_000_000 + counter * 1000,
                    f"production-{counter}",
                )
                command = {
                    "job_id": job,
                    "worker_id": worker,
                    "host_alias": alias,
                    "gpu": gpu,
                    "category": category,
                    "argv": argv,
                    "argv_sha256": canary._digest(argv),
                    "environment": {
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        **canary.executor.CLIENT_ENVIRONMENT,
                        "CATAN_SEED_LEDGER": str(production_ledger),
                        "CATAN_A1_CONTRACT_SHA256": lock["contract_sha256"],
                    },
                    "ledger_claim": {"path": str(production_ledger)},
                    "output_attestation": {
                        "source": str(source_attestation),
                        "source_file_sha256": canary._sha256(source_attestation),
                        "destination": f"/production/{job}/a1_contract.json",
                        "payload_sha256": canary._digest(payload),
                    },
                    "must_run_after": [] if previous is None else [previous],
                }
                lane.append(command)
                commands.append(command)
                previous = job
                counter += 1
            lanes[worker] = lane
    rendered = {
        "render_sha256": "sha256:" + "b" * 64,
        "commands": commands,
        "required_artifacts": {
            "seed_ledger": {
                "path": str(production_ledger),
                "sha256": canary._sha256(production_ledger),
            },
            "checkpoints": [],
            "rendered_opponent_mix": [],
        },
    }
    hosts = {
        "schema_version": canary.executor.HOST_SCHEMA,
        "ssh_user": "ubuntu",
        "ssh_key": str(tmp_path / "key"),
        "remote_root": "/home/ubuntu/a1-production",
        "python": "/home/ubuntu/catan-zero-v1/.venv/bin/python",
        "hosts": {alias: f"192.0.2.{index + 1}" for index, alias in enumerate(aliases)},
    }
    allowed = tmp_path / "gen_out"
    root = allowed / "a1-live-canary-shapecheck"
    plan = canary.derive_canary_plan(
        lock=lock,
        rendered=rendered,
        lanes=lanes,
        hosts=hosts,
        lock_path=lock_path,
        render_path=render_path,
        hosts_path=hosts_path,
        receipt_path=receipt_path,
        canary_id="shapecheck",
        base_seed=canary.contract.VAL_ONLY_SEED_RANGE[0],
        canary_root=root,
        allowed_root=allowed,
    )
    return plan, lock, rendered, lanes, hosts, root, production_ledger


def test_derives_exact_twelve_lane_validation_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _lock, _rendered, _lanes, _hosts, root, production_ledger = _fixture(
        tmp_path, monkeypatch
    )
    assert plan["validation_only"] is True
    assert plan["lane_count"] == 12
    assert plan["job_count"] == 36
    assert plan["claim_count"] == 0
    assert plan["canary_claim_count"] == 36
    assert plan["production_claims_consumed"] == 0
    assert set(plan["_private"]["hosts"]["hosts"]) == {"c1", "h100-8a"}
    assert plan["canary_seed_ledger"] != str(production_ledger)
    assert Path(plan["canary_seed_ledger"]).is_relative_to(root)
    assert production_ledger.read_text() == "# production\n"
    parsed_claims = canary.contract.parse_seed_ledger(plan["canary_seed_ledger"])
    assert len(parsed_claims) == 36
    assert all(
        "VAL-ONLY" in label and "claim=val-" in label for _, _, label in parsed_claims
    )
    assert [
        lane[0]["gpu"]
        for lane in plan["_private"]["lanes"].values()
        if lane[0]["host_alias"] == "c1"
    ] == [0, 1, 2, 3]
    assert sorted(
        lane[0]["gpu"]
        for lane in plan["_private"]["lanes"].values()
        if lane[0]["host_alias"] == "h100-8a"
    ) == list(range(8))
    canary.validate_canary_plan(plan)


def test_only_identity_seed_output_and_attempt_values_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _lock, rendered, _lanes, _hosts, root, _ledger = _fixture(
        tmp_path, monkeypatch
    )
    source = {command["job_id"]: command for command in rendered["commands"]}
    ranges = []
    for lane in plan["_private"]["lanes"].values():
        assert [item["category"] for item in lane] == list(canary.CATEGORY_ORDER)
        for command in lane:
            original = source[command["source_production_job_id"]]
            canary._assert_exact_recipe(original["argv"], command["argv"])
            assert canary._flag_value(command["argv"], "--games") == "16"
            assert Path(canary._flag_value(command["argv"], "--out-dir")).parent == root
            start = int(canary._flag_value(command["argv"], "--base-seed"))
            ranges.append((start, start + canary.GAMES_PER_JOB))
            assert (
                command["environment"]["CATAN_SEED_LEDGER"]
                == plan["canary_seed_ledger"]
            )
            assert command["environment"]["CATAN_A1_CANARY_ID"] == "shapecheck"
            attestation = json.loads(
                Path(command["output_attestation"]["source"]).read_text()
            )
            assert attestation["validation_only"] is True
            assert (
                attestation["target_information_regime"]
                == "public_conservation_pimc_v1"
            )
            assert attestation["source_job"]["job_id"] == original["job_id"]
    assert len(ranges) == len(set(ranges)) == 36
    assert ranges == sorted(ranges)


@pytest.mark.parametrize(
    ("seed_delta", "root_name", "message"),
    [
        (-1, "a1-live-canary-shapecheck", "VAL-ONLY"),
        (0, "wrong-root", "root must be exactly"),
    ],
)
def test_rejects_seed_or_output_boundary_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed_delta: int,
    root_name: str,
    message: str,
) -> None:
    plan, lock, rendered, lanes, hosts, _root, _ledger = _fixture(tmp_path, monkeypatch)
    with pytest.raises(canary.CanaryError, match=message):
        canary.derive_canary_plan(
            lock=lock,
            rendered=rendered,
            lanes=lanes,
            hosts=hosts,
            lock_path=Path(plan["lock"]),
            render_path=tmp_path / "render.json",
            hosts_path=tmp_path / "hosts.json",
            receipt_path=tmp_path / "other.receipt.json",
            canary_id="shapecheck",
            base_seed=canary.contract.VAL_ONLY_SEED_RANGE[0] + seed_delta,
            canary_root=tmp_path / "gen_out" / root_name,
            allowed_root=tmp_path / "gen_out",
        )


def test_rejects_missing_shape_or_scalar_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, lock, rendered, lanes, hosts, root, _ledger = _fixture(tmp_path, monkeypatch)
    broken_lanes = dict(lanes)
    broken_lanes.pop("c1_gpu3")
    with pytest.raises(canary.CanaryError, match="c1 gpu0-3"):
        canary.derive_canary_plan(
            lock=lock,
            rendered=rendered,
            lanes=broken_lanes,
            hosts=hosts,
            lock_path=Path(plan["lock"]),
            render_path=tmp_path / "render.json",
            hosts_path=tmp_path / "hosts.json",
            receipt_path=tmp_path / "missing.receipt.json",
            canary_id="shapecheck",
            base_seed=canary.contract.VAL_ONLY_SEED_RANGE[0],
            canary_root=root,
            allowed_root=tmp_path / "gen_out",
        )


def test_rejects_any_source_science_or_guard_bypass_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, lock, rendered, lanes, hosts, root, _ledger = _fixture(tmp_path, monkeypatch)
    broken_lanes = copy.deepcopy(lanes)
    command = broken_lanes["c1_gpu0"][0]
    n_full_index = command["argv"].index("--n-full") + 1
    command["argv"][n_full_index] = "64"
    with pytest.raises(canary.CanaryError, match="exact recipe requires --n-full=128"):
        canary.derive_canary_plan(
            lock=lock,
            rendered={
                **rendered,
                "commands": [item for lane in broken_lanes.values() for item in lane],
            },
            lanes=broken_lanes,
            hosts=hosts,
            lock_path=Path(plan["lock"]),
            render_path=tmp_path / "render.json",
            hosts_path=tmp_path / "hosts.json",
            receipt_path=tmp_path / "science.receipt.json",
            canary_id="shapecheck",
            base_seed=canary.contract.VAL_ONLY_SEED_RANGE[0],
            canary_root=root,
            allowed_root=tmp_path / "gen_out",
        )

    broken_lanes = copy.deepcopy(lanes)
    command = broken_lanes["c1_gpu0"][0]
    command["argv"].append("--skip-guards")
    with pytest.raises(canary.CanaryError, match="switch drift"):
        canary.derive_canary_plan(
            lock=lock,
            rendered={
                **rendered,
                "commands": [item for lane in broken_lanes.values() for item in lane],
            },
            lanes=broken_lanes,
            hosts=hosts,
            lock_path=Path(plan["lock"]),
            render_path=tmp_path / "render.json",
            hosts_path=tmp_path / "hosts.json",
            receipt_path=tmp_path / "guard.receipt.json",
            canary_id="shapecheck",
            base_seed=canary.contract.VAL_ONLY_SEED_RANGE[0],
            canary_root=root,
            allowed_root=tmp_path / "gen_out",
        )
    broken_lock = copy.deepcopy(lock)
    broken_lock["checkpoints"][0]["metadata"].pop("legacy_scalar_readout_attestation")
    with pytest.raises(canary.CanaryError, match="typed legacy scalar"):
        canary.derive_canary_plan(
            lock=broken_lock,
            rendered=rendered,
            lanes=lanes,
            hosts=hosts,
            lock_path=Path(plan["lock"]),
            render_path=tmp_path / "render.json",
            hosts_path=tmp_path / "hosts.json",
            receipt_path=tmp_path / "scalar.receipt.json",
            canary_id="shapecheck",
            base_seed=canary.contract.VAL_ONLY_SEED_RANGE[0],
            canary_root=root,
            allowed_root=tmp_path / "gen_out",
        )


def test_validation_rejects_any_production_ledger_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, *_ = _fixture(tmp_path, monkeypatch)
    plan["_private"]["rendered"]["required_artifacts"]["seed_ledger"]["path"] = plan[
        "production_seed_ledger"
    ]
    with pytest.raises(canary.CanaryError, match="not the canary ledger"):
        canary.validate_canary_plan(plan)


def test_status_stop_and_execute_reuse_hardened_receipt_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, *_ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(canary, "build_canary_plan", lambda **_kwargs: plan)
    calls = []
    monkeypatch.setattr(
        canary.executor,
        "execute",
        lambda built, *, receipt_path, resume: (
            calls.append(("run", built, receipt_path, resume)) or {"status": "launched"}
        ),
    )
    monkeypatch.setattr(
        canary.executor,
        "status",
        lambda built, *, receipt_path: (
            calls.append(("status", built, receipt_path)) or {"status": "ok"}
        ),
    )
    monkeypatch.setattr(
        canary.executor,
        "stop_execution",
        lambda built, *, receipt_path, go: (
            calls.append(("stop", built, receipt_path, go)) or {"status": "stopped"}
        ),
    )
    common = [
        "--lock",
        str(tmp_path / "lock"),
        "--render",
        str(tmp_path / "render"),
        "--hosts",
        str(tmp_path / "hosts"),
        "--receipt",
        str(tmp_path / "receipt"),
        "--canary-id",
        "shapecheck",
        "--base-seed",
        str(canary.contract.VAL_ONLY_SEED_RANGE[0]),
    ]
    assert canary.main(["run", *common, "--resume", "--go"]) == 0
    assert canary.main(["status", *common]) == 0
    assert canary.main(["stop", *common, "--go"]) == 0
    assert [call[0] for call in calls] == ["run", "status", "stop"]
    assert calls[0][3] is True and calls[2][3] is True


def test_audit_requires_clean_public_information_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, *_ = _fixture(tmp_path, monkeypatch)

    def fake_ssh(_hosts, _alias, command):
        encoded = command.split()[-1]
        # The shell-quoted JSON is easiest to recover from the expected plan,
        # while still proving one result is required per remote command.
        del encoded
        alias_commands = [
            item
            for lane in plan["_private"]["lanes"].values()
            for item in lane
            if item["host_alias"] == _alias
        ]
        payload = [
            {
                "job_id": item["job_id"],
                "rows": 10,
                "simulations": 128,
                "manifest_sha256": "sha256:m",
                "attestation_sha256": item["output_attestation"]["source_file_sha256"],
            }
            for item in alias_commands
        ]
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(canary.executor, "_ssh", fake_ssh)
    report = canary.audit_canary(plan)
    assert report["status"] == "PASS"
    assert report["job_count"] == 36
    assert report["rows"] == 360
    assert report["simulations"] == 36 * 128
    assert report["audit_sha256"].startswith("sha256:")


def test_remote_audit_script_rejects_hidden_truth_target_regime(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    attestation = tmp_path / "attestation.json"
    attestation.write_text('{"validation_only":true}\n', encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "games_requested": 16,
                "games_completed": 16,
                "games_failed": 0,
                "errors": [],
                "base_seed": canary.contract.VAL_ONLY_SEED_RANGE[0],
                "rows": 100,
                "simulations_used_total": 1280,
                "target_information_regime": "authoritative_hidden_state_search_v1",
            }
        ),
        encoding="utf-8",
    )
    expected = [
        {
            "job_id": "unsafe",
            "manifest": str(manifest),
            "attestation": str(attestation),
            "attestation_sha256": canary._sha256(attestation),
            "games": 16,
            "base_seed": canary.contract.VAL_ONLY_SEED_RANGE[0],
        }
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            canary.REMOTE_AUDIT_SCRIPT,
            json.dumps(expected),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "unsafe target regime: unsafe" in (result.stderr or result.stdout)
