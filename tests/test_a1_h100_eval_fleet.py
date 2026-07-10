from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.fleet import a1_h100_eval_fleet as fleet


def _manifest_file(tmp_path: Path) -> Path:
    hosts = [
        {"alias": alias, "address": f"10.0.0.{index + 10}", "gpu_count": count}
        for index, (alias, count) in enumerate(fleet.EXPECTED_SHAPES.items())
    ]
    value = {
        "schema_version": fleet.MANIFEST_SCHEMA,
        "ssh_user": "ubuntu",
        "ssh_key": str(tmp_path / "id_ed25519"),
        "strict_host_key_checking": "accept-new",
        "remote_repo": "/home/ubuntu/catan-zero-v1",
        "remote_python": "/home/ubuntu/catan-zero-v1/.venv/bin/python",
        "remote_root": "/home/ubuntu/a1-evaluation",
        "validation_ledger": str(tmp_path / "VAL_ONLY_EVAL_LEDGER.jsonl"),
        "ray_head_address": "10.0.0.2",
        "hosts": hosts,
    }
    path = tmp_path / "fleet.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _plan(tmp_path: Path) -> tuple[dict, dict]:
    manifest = fleet.load_manifest(_manifest_file(tmp_path))
    candidate = tmp_path / "candidate.pt"
    champion = tmp_path / "champion.pt"
    candidate.write_bytes(b"candidate")
    champion.write_bytes(b"champion")
    plan = fleet.build_plan(
        manifest,
        candidate=candidate,
        champion=champion,
        internal_pairs=600,
        external_pairs=500,
        internal_base_seed=6_190_000_000,
        external_base_seed=6_191_000_000,
        workers_per_gpu=8,
        repo_commit="a" * 40,
        tool_hashes={
            "tools/gumbel_search_cross_net_h2h.py": "sha256:" + "1" * 64,
            "tools/catanatron_neutral_harness_match.py": "sha256:" + "2" * 64,
            "tools/fleet/launch_detached.sh": "sha256:" + "3" * 64,
        },
    )
    return manifest, plan


def test_manifest_requires_exact_six_four_and_two_eight_gpu_hosts(
    tmp_path: Path,
) -> None:
    path = _manifest_file(tmp_path)
    value = json.loads(path.read_text())
    value["hosts"][-1]["gpu_count"] = 4
    path.write_text(json.dumps(value))
    with pytest.raises(fleet.FleetError, match="exactly six 4xH100 and two 8xH100"):
        fleet.load_manifest(path)


def test_internal_plan_weights_by_physical_gpu_and_conserves_seed_interval(
    tmp_path: Path,
) -> None:
    _manifest, plan = _plan(tmp_path)
    jobs = [job for job in plan["jobs"] if job["phase"] == "internal"]
    assert len(jobs) == 40
    assert sum(job["pairs"] for job in jobs) == 600
    assert {job["pairs"] for job in jobs} == {15}
    by_host = {
        alias: sum(job["pairs"] for job in jobs if job["alias"] == alias)
        for alias in fleet.EXPECTED_SHAPES
    }
    assert by_host["c1"] == 60
    assert by_host["h100-8a"] == 120
    intervals = sorted(
        (job["base_seed"], job["base_seed"] + job["pairs"]) for job in jobs
    )
    assert intervals[0][0] == 6_190_000_000
    assert intervals[-1][1] == 6_190_000_600
    assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))


def test_external_plan_uses_matched_candidate_champion_cohorts(tmp_path: Path) -> None:
    _manifest, plan = _plan(tmp_path)
    jobs = [job for job in plan["jobs"] if job["phase"] == "external"]
    assert len(jobs) == 40
    cohorts: dict[str, list[dict]] = {}
    for job in jobs:
        cohorts.setdefault(job["cohort_id"], []).append(job)
    assert len(cohorts) == 20
    for cohort in cohorts.values():
        assert {job["role"] for job in cohort} == {"candidate", "champion"}
        assert len({(job["base_seed"], job["pairs"]) for job in cohort}) == 1
        assert {job["pairs"] for job in cohort} == {25}
        assert len({job["slot_id"] for job in cohort}) == 2
    by_role = {
        role: sorted(
            (job["base_seed"], job["base_seed"] + job["pairs"])
            for job in jobs
            if job["role"] == role
        )
        for role in ("candidate", "champion")
    }
    assert by_role["candidate"] == by_role["champion"]
    assert by_role["candidate"][0][0] == 6_191_000_000
    assert by_role["candidate"][-1][1] == 6_191_000_500


def test_every_job_is_cuda_pinned_and_has_exact_n128_infoset_d6_recipe(
    tmp_path: Path,
) -> None:
    manifest, plan = _plan(tmp_path)
    required = {
        "--n-full": "128",
        "--c-scale": "0.03",
        "--c-visit": "50.0",
        "--sigma-eval": "0.98",
        "--determinization-particles": "4",
        "--determinization-min-simulations": "32",
        "--symmetry-averaged-eval-threshold": "20",
        "--value-readout": "scalar",
        "--gate-config": "flywheel",
    }
    for job in plan["jobs"]:
        argv = job["argv"]
        for flag, expected in required.items():
            assert argv[argv.index(flag) + 1] == expected
        for flag in (
            "--lazy-interior-chance",
            "--correct-rust-chance-spectra",
            "--public-observation",
            "--information-set-search",
            "--no-belief-chance-spectra",
            "--symmetry-averaged-eval",
        ):
            assert argv.count(flag) == 1
        assert "--device" in argv and argv[argv.index("--device") + 1] == "cuda"
    rendered = fleet.dry_run_commands(manifest, plan, "internal")
    assert len(rendered["hosts"]) == 8
    all_shell = "\n".join(row["ssh_command"][-1] for row in rendered["hosts"])
    for gpu in range(8):
        assert f"CUDA_VISIBLE_DEVICES={gpu}" in all_shell
    assert "B200" not in all_shell


def test_plan_hash_and_checkpoint_bytes_are_replayed_on_load(tmp_path: Path) -> None:
    manifest, plan = _plan(tmp_path)
    path = tmp_path / "plan.json"
    fleet.write_new_readonly(path, plan)
    assert fleet.load_plan(path, manifest)["plan_hash"] == plan["plan_hash"]
    assert plan["candidate"]["remote"] == plan["candidate"]["source"]
    assert plan["champion"]["remote"] == plan["champion"]["source"]
    Path(plan["candidate"]["source"]).chmod(0o644)
    Path(plan["candidate"]["source"]).write_bytes(b"changed")
    with pytest.raises(fleet.FleetError, match="candidate checkpoint bytes drifted"):
        fleet.load_plan(path, manifest)


def test_resume_selects_only_missing_failed_or_stale_jobs(tmp_path: Path) -> None:
    manifest, plan = _plan(tmp_path)
    jobs = [job for job in plan["jobs"] if job["phase"] == "internal"]
    status = {
        "jobs": [
            {
                "job_id": job["job_id"],
                "state": ("done", "active", "failed", "stale", "missing")[index % 5],
            }
            for index, job in enumerate(jobs)
        ]
    }
    selected = fleet.jobs_to_resume(plan, status, "internal")
    expected = {
        row["job_id"]
        for row in status["jobs"]
        if row["state"] in {"failed", "stale", "missing"}
    }
    assert selected == expected
    rendered = fleet.dry_run_commands(
        manifest, plan, "internal", selected_job_ids=selected
    )
    assert sum(row["jobs"] for row in rendered["hosts"]) == len(selected)


def test_ray_spec_advertises_no_b200_gpu_and_all_40_h100_slots(tmp_path: Path) -> None:
    manifest, plan = _plan(tmp_path)
    spec = fleet.ray_cluster_spec(manifest, plan)
    assert spec["head"]["num_gpus"] == 0
    assert sum(worker["num_gpus"] for worker in spec["workers"]) == 40
    assert spec["scheduler_contract"]["actor_resources"] == {
        "num_gpus": 1,
        "resources": {"H100": 1},
    }
    eight = next(worker for worker in spec["workers"] if worker["alias"] == "h100-8a")
    assert eight["resources"] == {"H100": 8}


def test_plan_rejects_overlapping_internal_external_seed_claims(tmp_path: Path) -> None:
    manifest = fleet.load_manifest(_manifest_file(tmp_path))
    candidate = tmp_path / "candidate.pt"
    champion = tmp_path / "champion.pt"
    candidate.write_bytes(b"candidate")
    champion.write_bytes(b"champion")
    with pytest.raises(fleet.FleetError, match="seed intervals overlap"):
        fleet.build_plan(
            manifest,
            candidate=candidate,
            champion=champion,
            internal_pairs=600,
            external_pairs=500,
            internal_base_seed=6_190_000_000,
            external_base_seed=6_190_000_500,
            repo_commit="a" * 40,
            tool_hashes={},
        )


def test_load_plan_rejects_semantic_tamper_even_if_old_hash_remains(
    tmp_path: Path,
) -> None:
    manifest, plan = _plan(tmp_path)
    tampered = copy.deepcopy(plan)
    tampered["jobs"][0]["pairs"] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered))
    with pytest.raises(fleet.FleetError, match="plan hash does not replay"):
        fleet.load_plan(path, manifest)


def test_validation_claim_is_atomic_idempotent_and_journaled(tmp_path: Path) -> None:
    manifest, plan = _plan(tmp_path)
    assert fleet.claim_validation_ranges(manifest, plan) == "claimed"
    assert fleet.claim_validation_ranges(manifest, plan) == "adopted"
    ledger = Path(manifest["validation_ledger"])
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["event"] == "claim"
    assert events[0]["plan_hash"] == plan["plan_hash"]
    claim_files = list(Path(str(ledger) + ".claims").glob("*.json"))
    assert len(claim_files) == 1
    assert (claim_files[0].stat().st_mode & 0o777) == 0o444


def test_validation_claim_rejects_overlap_from_concurrent_plan(tmp_path: Path) -> None:
    manifest, first = _plan(tmp_path)
    assert fleet.claim_validation_ranges(manifest, first) == "claimed"
    candidate = tmp_path / "candidate-2.pt"
    candidate.write_bytes(b"candidate-two")
    second = fleet.build_plan(
        manifest,
        candidate=candidate,
        champion=Path(first["champion"]["source"]),
        internal_pairs=600,
        external_pairs=500,
        internal_base_seed=6_190_000_300,
        external_base_seed=6_192_000_000,
        iteration_id="a2",
        repo_commit="a" * 40,
        tool_hashes=first["tool_hashes"],
    )
    with pytest.raises(fleet.FleetError, match="VAL-only seed overlap"):
        fleet.claim_validation_ranges(manifest, second)


def test_validation_status_adopts_claim_and_appends_event(tmp_path: Path) -> None:
    manifest, plan = _plan(tmp_path)
    fleet.claim_validation_ranges(manifest, plan)
    fleet.record_validation_status(manifest, plan, status="internal_collected")
    events = [
        json.loads(line)
        for line in Path(manifest["validation_ledger"]).read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["claim", "status"]
    assert events[-1]["status"] == "internal_collected"


def test_claim_adoption_repairs_crash_between_claim_file_and_journal(
    tmp_path: Path,
) -> None:
    manifest, plan = _plan(tmp_path)
    ledger = Path(manifest["validation_ledger"])
    claims = Path(str(ledger) + ".claims")
    claims.mkdir(parents=True)
    claim = claims / f"{plan['plan_hash'][7:]}.json"
    claim.write_text(json.dumps(fleet._claim_payload(plan)))  # noqa: SLF001
    assert fleet.claim_validation_ranges(manifest, plan) == "adopted"
    event = json.loads(ledger.read_text().strip())
    assert event["event"] == "claim"
    assert event["recovered"] is True
