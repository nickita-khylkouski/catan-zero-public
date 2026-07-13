from __future__ import annotations

import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tools import a1_pre_wave_contract as contract
from tools.fleet import a1_harvest_transaction as harvest


CATEGORIES = ("current_producer", "recent_history", "hard_negative")


def _current_search_operator() -> dict:
    """Build the fixture operator through the current production validator."""

    return contract._search_operator(  # noqa: SLF001
        {
            "c_scale": 0.03,
            "c_visit": 50.0,
            "n_fast": 16,
            "n_full": 128,
            "p_full": 0.4,
            "max_depth": 600,
            "lazy_interior_chance": True,
            "correct_rust_chance_spectra": True,
            "belief_chance_spectra": True,
            "information_set_search": True,
            "determinization_particles": 4,
            "determinization_min_simulations": 1,
            "rescale_noise_floor_c": 1.0,
            "sigma_eval": 0.98,
            "prior_temperature": 1.0,
            "symmetry_averaged_eval": True,
            "symmetry_averaged_eval_threshold": 20,
            "exact_budget_sh": True,
            "exact_budget_sh_min_n": 48,
            "n_full_wide": 256,
            "n_full_wide_threshold": 40,
            "wide_roots_always_full": True,
            "wide_candidates_threshold": 40,
            "raw_policy_above_width": 80,
        }
    )


def test_bounded_fetch_uses_multiple_direct_host_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    barrier = threading.Barrier(2)
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_fetch(_command, _host, _outputs, _archive):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        barrier.wait(timeout=2)
        with state_lock:
            active -= 1

    monkeypatch.setattr(harvest, "_ssh_fetch", fake_fetch)
    monkeypatch.setattr(harvest, "_ssh_output_bytes", lambda *_args: 1)
    monkeypatch.setattr(harvest, "_preflight_cohort_disk", lambda *_args: None)
    monkeypatch.setattr(harvest, "_install_fetched_batch", lambda *_args, **_kwargs: None)
    batches = [
        harvest._HostFetch(  # noqa: SLF001 - focused transport concurrency test.
            host=f"h{index}",
            missing=(),
            outputs=(Path(f"/sealed/a1/h{index}/job"),),
            archive=tmp_path / f"h{index}.tar",
            extracted=tmp_path / f"h{index}.extract",
        )
        for index in range(2)
    ]

    harvest._fetch_archives(  # noqa: SLF001
        ("ssh",),
        batches,
        workers=2,
        incoming_root=tmp_path,
        lock={},
        jobs_root=tmp_path,
        state_root=tmp_path,
        inventories={},
    )

    assert max_active == 2


def _fixture_contract(tmp_path: Path) -> tuple[dict, dict, Path, Path, Path]:
    remote = tmp_path / "remote"
    jobs = []
    for worker in range(40):
        host = f"h{worker // 5}"
        for category_index, category in enumerate(CATEGORIES):
            job_id = f"w{worker:02d}__{category}"
            output = f"/sealed/a1/{job_id}"
            jobs.append(
                {
                    "job_id": job_id,
                    "worker_id": f"w{worker:02d}",
                    "host_alias": host,
                    "gpu": worker % 5,
                    "category": category,
                    "base_seed": worker * 1000 + category_index * 100,
                    "games": (240, 45, 15)[category_index],
                    "attempts": (245, 47, 16)[category_index],
                    "seed_end": worker * 1000
                    + category_index * 100
                    + (245, 47, 16)[category_index],
                    "output_dir": output,
                }
            )
    search_operator = _current_search_operator()
    lock = {
        "contract_sha256": "sha256:" + "1" * 64,
        "fleet": {
            "seed_plan_sha256": "sha256:" + "2" * 64,
            "jobs": jobs,
        },
        "checkpoints": [
            {"id": "producer", "role": "producer", "sha256": "sha256:" + "3" * 64},
            {"id": "history", "role": "opponent", "sha256": "sha256:" + "4" * 64},
            {"id": "hard", "role": "opponent", "sha256": "sha256:" + "5" * 64},
        ],
        "source_categories": [
            {"name": "current_producer", "checkpoint_ids": ["producer"]},
            {"name": "recent_history", "checkpoint_ids": ["history"]},
            {"name": "hard_negative", "checkpoint_ids": ["hard"]},
        ],
        "science": {
            "search_operator": search_operator,
            "search_operator_sha256": contract._digest_value(search_operator),
            "effective_search_config_sha256": "sha256:" + "7" * 64,
            "evaluator_sha256": "sha256:" + "8" * 64,
            "value_readout": "scalar",
        },
        "provenance": {"runtime_code_tree_sha256": "sha256:" + "9" * 64},
    }
    commands = []
    for job in jobs:
        commands.append(
            {
                **{
                    key: job[key]
                    for key in ("job_id", "worker_id", "host_alias", "gpu", "category")
                },
                "output_attestation": {
                    "destination": f"{job['output_dir']}/a1_contract.json",
                    "payload_sha256": contract._digest_value(
                        contract._job_attestation(lock, job)
                    ),
                },
            }
        )
        local = remote / str(job["host_alias"]) / str(job["job_id"])
        local.mkdir(parents=True)
        (local / "a1_contract.json").write_text(
            json.dumps(contract._job_attestation(lock, job)) + "\n", encoding="utf-8"
        )
        (local / "config_registry.jsonl").write_text("{}\n", encoding="utf-8")
        (local / "shard.npz").write_bytes(f"bytes:{job['job_id']}".encode())
        (local / "worker.json").write_text("{}\n", encoding="utf-8")
        (local / "manifest.json").write_text(
            json.dumps(
                {
                    "base_seed": job["base_seed"],
                    "shards": [f"{job['output_dir']}/shard.npz"],
                    "worker_summaries": [f"{job['output_dir']}/worker.json"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
    rendered = {
        "render_sha256": "sha256:" + "a" * 64,
        "commands": commands,
    }
    lock_path = tmp_path / "lock.json"
    render_path = tmp_path / "render.json"
    lock_path.write_text("{}\n", encoding="utf-8")
    render_path.write_text("{}\n", encoding="utf-8")
    return lock, rendered, lock_path, render_path, remote


def _fake_ssh(tmp_path: Path) -> Path:
    script = tmp_path / "fake_ssh.py"
    script.write_text(
        """#!/usr/bin/env python3
import io, os, shlex, sys, tarfile
from pathlib import Path

host, command = sys.argv[1:3]
if os.environ.get("FAKE_FAIL_HOST") == host:
    print("injected transport failure", file=sys.stderr)
    raise SystemExit(23)
tokens = shlex.split(command)
if tokens[:2] == ["du", "-sb"]:
    root = Path(os.environ["FAKE_REMOTE_ROOT"]) / host
    for raw in tokens[tokens.index("--") + 1:]:
        source = root / Path(raw).name
        size = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
        print(f"{max(size, 1)}\\t{raw}")
    raise SystemExit(0)
with open(os.environ["FAKE_SSH_LOG"], "a", encoding="utf-8") as log:
    log.write(host + "\\n")
names = tokens[tokens.index("--") + 1:]
root = Path(os.environ["FAKE_REMOTE_ROOT"]) / host
mode = os.environ.get("FAKE_MODE", "")
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as bundle:
    for index, name in enumerate(names):
        source = root / name
        if mode == "hostswap" and index == 0:
            source = next(path for path in root.iterdir() if path.name != name)
        bundle.add(source, arcname=name)
        if mode == "duplicate" and index == 0:
            bundle.add(source / "manifest.json", arcname=f"{name}/manifest.json")
    if mode == "traversal":
        info = tarfile.TarInfo("../escape")
        info.size = 1
        bundle.addfile(info, io.BytesIO(b"x"))
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


@pytest.fixture
def fleet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    lock, rendered, lock_path, render_path, remote = _fixture_contract(tmp_path)
    monkeypatch.setattr(harvest.contract, "verify_lock", lambda _path: lock)
    monkeypatch.setattr(
        harvest.contract,
        "_validate_claim_render",
        lambda _lock, _path: (rendered, []),
    )
    log = tmp_path / "ssh.log"
    log.write_text("", encoding="utf-8")
    monkeypatch.setenv("FAKE_REMOTE_ROOT", str(remote))
    monkeypatch.setenv("FAKE_SSH_LOG", str(log))
    return lock, rendered, lock_path, render_path, remote, _fake_ssh(tmp_path), log


def _run(fleet, tmp_path: Path, *, fetch_workers: int = 1):
    _lock, _rendered, lock_path, render_path, _remote, ssh, _log = fleet
    return harvest.harvest(
        lock_path,
        render_path,
        tmp_path / "published",
        ssh_command=(str(ssh),),
        fetch_workers=fetch_workers,
    )


def test_collects_exact_120_jobs_from_eight_hosts_and_resumes_published(
    fleet, tmp_path: Path
) -> None:
    result = _run(fleet, tmp_path)
    assert result["job_count"] == 120
    assert result["host_count"] == 8
    assert len(result["job_identities"]) == 120
    assert len((fleet[-1]).read_text().splitlines()) == 8


def test_parallel_fetch_publishes_the_same_validated_inventory(
    fleet, tmp_path: Path
) -> None:
    result = _run(fleet, tmp_path, fetch_workers=4)

    assert result["job_count"] == 120
    assert result["host_count"] == 8
    assert len(result["job_identities"]) == 120
    assert len(fleet[-1].read_text().splitlines()) == 8
    loaded = contract._load_harvest_relocation(
        tmp_path / "published/relocation_map.json", lock=fleet[0]
    )
    assert loaded.payload["relocation_sha256"] == result["relocation_sha256"]
    assert (tmp_path / "published/harvest_receipt.json").is_file()
    loaded = contract._load_harvest_relocation(
        tmp_path / "published/relocation_map.json", lock=fleet[0]
    )
    assert len(loaded.by_source) == len(result["files"])
    assert _run(fleet, tmp_path) == result
    assert len((fleet[-1]).read_text().splitlines()) == 8


def test_parallel_fetch_bounds_resident_archives_to_worker_count(
    fleet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = harvest._ssh_fetch  # noqa: SLF001
    observed: list[int] = []

    def measured(*args, **kwargs) -> None:
        original(*args, **kwargs)
        incoming = args[3].parent
        observed.append(len(list(incoming.glob("*.tar"))))

    monkeypatch.setattr(harvest, "_ssh_fetch", measured)
    result = _run(fleet, tmp_path, fetch_workers=2)

    assert result["job_count"] == 120
    assert observed and max(observed) <= 2


def test_disk_preflight_refuses_before_archive_fetch(
    fleet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(harvest, "_ssh_output_bytes", lambda *_args: 100_000_000)
    usage_type = type(shutil.disk_usage(tmp_path))
    monkeypatch.setattr(
        harvest.shutil,
        "disk_usage",
        lambda _path: usage_type(total=1_000, used=900, free=100),
    )
    called = False

    def should_not_fetch(*_args, **_kwargs) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(harvest, "_ssh_fetch", should_not_fetch)
    with pytest.raises(harvest.HarvestError, match="insufficient local disk"):
        _run(fleet, tmp_path, fetch_workers=4)
    assert called is False


def test_interrupted_incoming_is_scavenged_on_retry(
    fleet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = harvest._fetch_archives  # noqa: SLF001

    def killed(*_args, incoming_root: Path, **_kwargs) -> None:
        token = "1" * 32
        (incoming_root / f"h0.{token}.tar").write_bytes(b"partial")
        (incoming_root / f"h0.{token}.stderr").write_bytes(b"diagnostic")
        (incoming_root / f"h0.{token}").mkdir()
        raise RuntimeError("simulated hard interruption")

    monkeypatch.setattr(harvest, "_fetch_archives", killed)
    with pytest.raises(RuntimeError, match="hard interruption"):
        _run(fleet, tmp_path, fetch_workers=4)
    stage = next(tmp_path.glob(".published.harvest-*.staging"))
    assert any((stage / "incoming").iterdir())

    monkeypatch.setattr(harvest, "_fetch_archives", original)
    result = _run(fleet, tmp_path, fetch_workers=4)
    assert result["job_count"] == 120


def test_incoming_scavenger_rejects_unknown_artifact(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "unowned-file").write_bytes(b"x")
    with pytest.raises(harvest.HarvestError, match="unsafe unknown artifact"):
        harvest._cleanup_owned_incoming(incoming, hosts={"h0"})  # noqa: SLF001


@pytest.mark.parametrize("attack", ["duplicate", "traversal", "hostswap"])
def test_archive_duplicate_path_attack_and_host_swap_fail_closed(
    fleet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack: str
) -> None:
    monkeypatch.setenv("FAKE_MODE", attack)
    with pytest.raises(harvest.HarvestError):
        _run(fleet, tmp_path)
    assert not (tmp_path / "published").exists()
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize(
    "relative",
    ["../x", "/absolute", "job/../../x", "./job/file", "job//file"],
)
def test_member_path_validator_rejects_noncanonical_names(relative: str) -> None:
    with pytest.raises(harvest.HarvestError):
        harvest._member_relative(relative, {"job"})


def test_missing_and_corrupt_outputs_fail_before_publish(
    fleet, tmp_path: Path
) -> None:
    lock, _rendered, *_rest, remote, _ssh, _log = fleet
    first = lock["fleet"]["jobs"][0]
    (remote / first["host_alias"] / first["job_id"] / "config_registry.jsonl").unlink()
    with pytest.raises(harvest.HarvestError, match="missing required outputs"):
        _run(fleet, tmp_path)
    assert not (tmp_path / "published").exists()


def test_corrupt_attestation_fails_before_publish(fleet, tmp_path: Path) -> None:
    lock, _rendered, *_rest, remote, _ssh, _log = fleet
    first = lock["fleet"]["jobs"][0]
    (remote / first["host_alias"] / first["job_id"] / "a1_contract.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(harvest.HarvestError, match="attestation drift"):
        _run(fleet, tmp_path)
    assert not (tmp_path / "published").exists()


def test_resume_skips_hosts_already_receipted(
    fleet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_FAIL_HOST", "h4")
    with pytest.raises(harvest.HarvestError, match="transport failure"):
        _run(fleet, tmp_path)
    assert not (tmp_path / "published").exists()
    monkeypatch.delenv("FAKE_FAIL_HOST")
    result = _run(fleet, tmp_path)
    assert result["job_count"] == 120
    # h0..h3 succeeded; h4 failed during the size preflight before an archive
    # stream began, then h4..h7 were fetched on resume.
    assert len(fleet[-1].read_text().splitlines()) == 8


def test_resume_recovers_atomic_job_directory_before_state_receipt(
    fleet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_FAIL_HOST", "h1")
    with pytest.raises(harvest.HarvestError):
        _run(fleet, tmp_path)
    stage = next(tmp_path.glob(".published.harvest-*.staging"))
    orphan_state = next((stage / "state").glob("w00__*.json"))
    orphan_state.unlink()
    monkeypatch.delenv("FAKE_FAIL_HOST")

    result = _run(fleet, tmp_path)

    assert result["job_count"] == 120
    assert len(fleet[-1].read_text().splitlines()) == 8


def test_resume_refuses_corrupt_staged_bytes(
    fleet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_FAIL_HOST", "h1")
    with pytest.raises(harvest.HarvestError):
        _run(fleet, tmp_path)
    stage = next(tmp_path.glob(".published.harvest-*.staging"))
    staged_shard = next((stage / "payload/jobs").rglob("shard.npz"))
    staged_shard.chmod(0o644)
    staged_shard.write_bytes(b"changed")
    monkeypatch.delenv("FAKE_FAIL_HOST")
    with pytest.raises(harvest.HarvestError, match="resume state drift"):
        _run(fleet, tmp_path)
    assert not (tmp_path / "published").exists()


def test_relocation_loader_rejects_post_publish_symlink(fleet, tmp_path: Path) -> None:
    result = _run(fleet, tmp_path)
    record = next(item for item in result["files"] if item["source_path"].endswith("shard.npz"))
    victim = tmp_path / "published" / record["relative_path"]
    replacement = tmp_path / "replacement"
    replacement.write_bytes(victim.read_bytes())
    victim.unlink()
    victim.symlink_to(replacement)
    with pytest.raises(contract.ContractError, match="symlink"):
        contract._load_harvest_relocation(
            tmp_path / "published/relocation_map.json", lock=fleet[0]
        )


@pytest.mark.parametrize("input_index", [2, 3], ids=["lock", "render"])
def test_input_inode_drift_during_os_replace_never_publishes_mixed_identity(
    fleet,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_index: int,
) -> None:
    input_path = fleet[input_index]
    original_replace = harvest.os.replace
    injected = False

    def drifting_replace(source, destination, *args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            replacement = input_path.with_suffix(input_path.suffix + ".replacement")
            replacement.write_text('{"drift":true}\n', encoding="utf-8")
            original_replace(replacement, input_path)
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(harvest.os, "replace", drifting_replace)
    with pytest.raises(harvest.HarvestError, match="immutable input .*drifted"):
        _run(fleet, tmp_path)
    assert not (tmp_path / "published").exists()


def test_receipt_write_crash_replays_existing_exact_map(
    fleet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_write = harvest._write_exclusive_json
    crashed = False

    def crash_receipt(path: Path, payload: dict) -> None:
        nonlocal crashed
        if path.name == "harvest_receipt.json" and not crashed:
            crashed = True
            raise RuntimeError("injected receipt crash")
        original_write(path, payload)

    monkeypatch.setattr(harvest, "_write_exclusive_json", crash_receipt)
    with pytest.raises(RuntimeError, match="receipt crash"):
        _run(fleet, tmp_path)
    assert not (tmp_path / "published").exists()
    assert len(fleet[-1].read_text().splitlines()) == 8
    stage = next(tmp_path.glob(".published.harvest-*.staging"))
    assert stage.stat().st_mode & 0o777 == 0o700
    assert all(
        path.stat().st_mode & 0o777 == 0o700
        for path in (stage / "payload", stage / "state", stage / "incoming")
    )
    monkeypatch.setattr(harvest, "_write_exclusive_json", original_write)

    result = _run(fleet, tmp_path)

    assert result["job_count"] == 120
    assert len(fleet[-1].read_text().splitlines()) == 8


def test_staged_mutation_immediately_before_rename_fails_preflight(
    fleet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_publish = harvest._atomic_publish_noreplace

    def mutate_then_publish(source: Path, destination: Path, *, preflight=None) -> None:
        shard = next((source / "jobs").rglob("shard.npz"))
        shard.chmod(0o644)
        shard.write_bytes(b"mutated-before-rename")
        original_publish(source, destination, preflight=preflight)

    monkeypatch.setattr(harvest, "_atomic_publish_noreplace", mutate_then_publish)
    with pytest.raises(
        (harvest.HarvestError, contract.ContractError),
        match="digest|bytes|changed",
    ):
        _run(fleet, tmp_path)
    assert not (tmp_path / "published").exists()


def test_transaction_lock_unlink_before_rename_cannot_publish(
    fleet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_publish = harvest._atomic_publish_noreplace

    def unlink_lock_then_publish(
        source: Path, destination: Path, *, preflight=None
    ) -> None:
        lock_path = next(tmp_path.glob(".a1-harvest-*.lock"))
        lock_path.unlink()
        original_publish(source, destination, preflight=preflight)

    monkeypatch.setattr(harvest, "_atomic_publish_noreplace", unlink_lock_then_publish)
    with pytest.raises(harvest.HarvestError, match="transaction lock"):
        _run(fleet, tmp_path)
    assert not (tmp_path / "published").exists()


def test_concurrent_same_destination_invocations_serialize_on_parent_lock(
    fleet, tmp_path: Path
) -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_run, fleet, tmp_path) for _ in range(2)]
        results = [future.result(timeout=30) for future in futures]

    assert results[0] == results[1]
    assert len(fleet[-1].read_text().splitlines()) == 8
