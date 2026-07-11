from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.fleet import a1_experimental_opponent_recovery as recovery


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sealed(path: Path, value: dict, field: str) -> dict:
    value[field] = recovery.contract._digest_value(value)  # noqa: SLF001
    _json(path, value)
    return value


def _argv(
    *,
    arm: str,
    worker: str,
    category: str,
    base_seed: int,
    attempts: int,
    checkpoint: Path,
    opponent: Path | None,
) -> list[str]:
    values = [
        "tools/generate_gumbel_selfplay_data.py",
        "--out-dir",
        f"/old/{arm}/{worker}__{category}",
        "--games",
        str(attempts),
        "--workers",
        "16",
        "--checkpoint",
        str(checkpoint),
        "--device",
        "cuda",
        "--n-full",
        arm.removeprefix("n"),
        "--n-fast",
        "16",
        "--p-full",
        "0.25",
        "--c-visit",
        "50.0",
        "--c-scale",
        "0.1" if category == "current_producer" else "0.03",
        "--base-seed",
        str(base_seed),
        "--ledger-claim-label",
        f"r1:{arm}:{worker}__{category}",
        "--symmetry-averaged-eval",
        "--symmetry-averaged-eval-threshold",
        "20",
        "--correct-rust-chance-spectra",
        "--lazy-interior-chance",
        "--no-belief-chance-spectra",
        "--information-set-search",
        "--determinization-particles",
        "4",
        "--determinization-min-simulations",
        "32",
        "--public-observation",
        "--no-rust-featurize",
        "--seed-claim",
        "--resume",
        "--generation-arm-id",
        arm,
    ]
    if opponent is not None:
        values += ["--opponent-mix-manifest", str(opponent)]
    return values


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    runtime = tmp_path / "generate.py"
    runtime.write_text("# runtime\n")
    placement = tmp_path / "placement.json"
    assignments = []
    for arm in ("n128", "n256"):
        for lane in range(28):
            assignments.append(
                {
                    "logical_lane": f"{arm}_gpu{lane:02d}",
                    "host_alias": f"host-{lane // 4}",
                    "gpu": lane % 4,
                }
            )
    _json(placement, {"assignments": assignments})
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    wheel = tmp_path / "catanatron_rs-0.1.5-cp311-cp311-manylinux_2_34_x86_64.whl"
    wheel.write_bytes(b"wheel")
    ledger = tmp_path / "ledger.md"
    ledger.write_text("# ledger\n")
    arms = {}
    for arm in ("n128", "n256"):
        opponent_dir = tmp_path / arm
        recent = opponent_dir / "recent.json"
        hard = opponent_dir / "hard.json"
        _json(recent, {"category": "recent_history"})
        _json(hard, {"category": "hard_negative"})
        jobs, commands = [], []
        for lane in range(28):
            worker = f"{arm}_gpu{lane:02d}"
            cursor = (100_000 if arm == "n128" else 200_000) + lane * 8192
            values = (
                (
                    "current_producer",
                    4080 if arm == "n128" else 1640,
                    4000 if arm == "n128" else 1600,
                    None,
                ),
                (
                    "recent_history",
                    765 if arm == "n128" else 310,
                    750 if arm == "n128" else 300,
                    recent,
                ),
                (
                    "hard_negative",
                    255 if arm == "n128" else 104,
                    250 if arm == "n128" else 100,
                    hard,
                ),
            )
            for category, attempts, selected, opponent in values:
                job_id = f"{worker}__{category}"
                argv = _argv(
                    arm=arm,
                    worker=worker,
                    category=category,
                    base_seed=cursor,
                    attempts=attempts,
                    checkpoint=checkpoint,
                    opponent=opponent,
                )
                jobs.append(
                    {
                        "job_id": job_id,
                        "worker_id": worker,
                        "category": category,
                        "attempts": attempts,
                        "games": selected,
                        "base_seed": cursor,
                        "seed_end": cursor + attempts,
                        "claim_label": f"r1:{arm}:{job_id}",
                    }
                )
                commands.append(
                    {
                        "job_id": job_id,
                        "worker_id": worker,
                        "arm_id": arm,
                        "category": category,
                        "host_alias": f"host-{lane // 4}",
                        "gpu": lane % 4,
                        "argv": argv,
                        "argv_sha256": recovery._digest(argv),  # noqa: SLF001
                        "ledger_claim": {
                            "path": str(ledger),
                            "row": f"[{cursor} - {cursor + attempts})",
                        },
                    }
                )
                cursor += attempts
        lock_path = tmp_path / arm / "lock.json"
        render_path = tmp_path / arm / "render.json"
        lock = _sealed(lock_path, {"fleet": {"jobs": jobs}}, "contract_sha256")
        refs = [
            {"path": str(path.resolve()), "sha256": recovery._sha256(path)}  # noqa: SLF001
            for path in (recent, hard)
        ]
        render = _sealed(
            render_path,
            {
                "commands": commands,
                "required_artifacts": {
                    "rendered_opponent_mix": refs,
                    "checkpoints": [
                        {
                            "path": str(checkpoint),
                            "sha256": recovery._sha256(checkpoint),  # noqa: SLF001
                        }
                    ],
                    "seed_ledger": {
                        "path": str(ledger),
                        "sha256": recovery._sha256(ledger),  # noqa: SLF001
                    },
                },
            },
            "render_sha256",
        )
        arms[arm] = {
            "lock": str(lock_path),
            "lock_file_sha256": recovery._sha256(lock_path),  # noqa: SLF001
            "lock_sha256": lock["contract_sha256"],
            "render": str(render_path),
            "render_file_sha256": recovery._sha256(render_path),  # noqa: SLF001
            "render_sha256": render["render_sha256"],
        }
    config = tmp_path / "config.json"
    _json(
        config,
        {
            "schema_version": recovery.CONFIG_SCHEMA,
            "label": recovery.LABEL,
            "runtime_repo": str(tmp_path),
            "runtime_commit": "a" * 40,
            "native_wheel": {
                "filename": wheel.name,
                "path": str(wheel),
                "sha256": recovery._sha256(wheel),  # noqa: SLF001
            },
            "runtime_files": [
                {"path": str(runtime), "sha256": recovery._sha256(runtime)}  # noqa: SLF001
            ],
            "placement": {
                "path": str(placement),
                "sha256": recovery._sha256(placement),  # noqa: SLF001
            },
            "arms": arms,
            "recovery_root": str(tmp_path / "recovery"),
            "allowed_categories": list(recovery.ALLOWED),
        },
    )
    failed = tmp_path / "failed.json"
    _json(failed, {"status": "failed"})
    return config, failed


def test_plan_is_opponent_only_exact_and_nonpromotable(tmp_path: Path) -> None:
    config, failed = _fixture(tmp_path)
    out = tmp_path / "operator" / "plan.json"
    plan = recovery.build_plan(config_path=config, failed_receipts=[failed], out=out)
    assert plan["label"] == recovery.LABEL
    assert plan["promotable"] is False
    assert len(plan["lanes"]) == 56
    assert plan["native_wheel"]["sha256"] in {
        row["sha256"] for row in plan["required_host_files"]
    }
    assert {lane["arm_id"] for lane in plan["lanes"]} == {"n128", "n256"}
    assert all(len(lane["commands"]) == 2 for lane in plan["lanes"])
    assert all(
        tuple(command["category"] for command in lane["commands"]) == recovery.ALLOWED
        for lane in plan["lanes"]
    )
    for lane in plan["lanes"]:
        for command in lane["commands"]:
            assert "--native-mcts-hot-loop" in command["argv"]
            assert "--rust-featurize" in command["argv"]
            assert "--no-rust-featurize" not in command["argv"]
            assert command["output_dir"].startswith(str(tmp_path / "recovery"))
            assert command["max_attempts"] == int(
                command["argv"][command["argv"].index("--games") + 1]
            )
    assert (
        recovery.build_plan(config_path=config, failed_receipts=[failed], out=out)
        == plan
    )


def test_plan_dry_run_does_not_create_recovery_namespace(tmp_path: Path) -> None:
    config, failed = _fixture(tmp_path)
    recovery.build_plan(
        config_path=config,
        failed_receipts=[failed],
        out=tmp_path / "plan.json",
    )
    assert not (tmp_path / "recovery").exists()


def test_plan_rejects_complete_receipt_and_source_drift(tmp_path: Path) -> None:
    config, failed = _fixture(tmp_path)
    _json(failed, {"status": "complete"})
    with pytest.raises(recovery.RecoveryError, match="complete"):
        recovery.build_plan(
            config_path=config,
            failed_receipts=[failed],
            out=tmp_path / "plan.json",
        )
    _json(failed, {"status": "failed"})
    value = json.loads(config.read_text())
    value["arms"]["n128"]["render_file_sha256"] = "sha256:" + "0" * 64
    _json(config, value)
    with pytest.raises(recovery.RecoveryError, match="render file hash drift"):
        recovery.build_plan(
            config_path=config,
            failed_receipts=[failed],
            out=tmp_path / "other.json",
        )


def test_missing_historical_receipt_requires_explicit_annotation(
    tmp_path: Path,
) -> None:
    config, _failed = _fixture(tmp_path)
    with pytest.raises(recovery.RecoveryError, match="historical annotation"):
        recovery.build_plan(
            config_path=config, failed_receipts=[], out=tmp_path / "refused.json"
        )
    value = json.loads(config.read_text())
    value["historical_parent_receipt"] = "historical_parent_receipt_unavailable"
    _json(config, value)
    plan = recovery.build_plan(
        config_path=config, failed_receipts=[], out=tmp_path / "accepted.json"
    )
    assert plan["failed_receipts"] == []
    assert plan["historical_parent_receipt"] == "historical_parent_receipt_unavailable"


def test_launch_without_go_is_read_only(tmp_path: Path) -> None:
    config, failed = _fixture(tmp_path)
    plan_path = tmp_path / "plan.json"
    recovery.build_plan(config_path=config, failed_receipts=[failed], out=plan_path)
    fleet = tmp_path / "fleet.json"
    hosts = [
        {"alias": f"host-{index}", "address": f"10.0.0.{index + 1}"}
        for index in range(7)
    ] + [
        {"alias": f"extra-{index}", "address": f"10.0.1.{index + 1}"}
        for index in range(3)
    ]
    _json(
        fleet,
        {
            "schema_version": "catan-gpu-fleet-v1",
            "ssh_user": "ubuntu",
            "ssh_key": None,
            "hosts": hosts,
        },
    )
    result = recovery.launch(
        plan_path, fleet_path=fleet, ssh_key=None, go=False, resume=False
    )
    assert result["mode"] == "dry-run"
    assert len(result["targets"]) == 56
    assert not (tmp_path / "recovery").exists()


def test_atomic_plan_refuses_different_existing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    recovery._atomic_exact(path, {"a": 1})  # noqa: SLF001
    recovery._atomic_exact(path, {"a": 1})  # noqa: SLF001
    with pytest.raises(recovery.RecoveryError, match="differs"):
        recovery._atomic_exact(path, {"a": 2})  # noqa: SLF001
