from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
from threading import Barrier
import time

import pytest

from catan_zero.rl import ppo_distributed as dist
from catan_zero.rl.ppo_run_manifest import PPORunManifest, load_manifest


TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "selfplay"
    / "ppo_2p_no_trade_v2.json"
)


def _bound_manifest(*, seed: int = 1) -> PPORunManifest:
    template = load_manifest(TEMPLATE)
    identity = replace(
        template.spec.identity,
        initializer_sha256="sha256:" + "a" * 64,
    )
    actor = replace(template.spec.actor, seed=seed)
    return replace(
        template,
        status="bound",
        spec=replace(template.spec, identity=identity, actor=actor),
    )


def test_v2_binding_refuses_template_and_historical_v1_root(tmp_path: Path) -> None:
    template = load_manifest(TEMPLATE)
    with pytest.raises(dist.RunManifestError, match="status='bound'"):
        dist.bind_run_manifest(tmp_path / "template", template)

    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"parent")
    legacy_root = tmp_path / "legacy"
    dist.bind_run_contract(
        legacy_root,
        init_checkpoint=checkpoint,
        architecture="entity_graph",
        gamma=1.0,
        gae_lambda=0.95,
        behavior_temperature=1.0,
    )
    with pytest.raises(dist.RunManifestError, match="historical v1 root"):
        dist.bind_run_manifest(legacy_root, _bound_manifest())


def test_historical_v1_binding_refuses_existing_v2_root(tmp_path: Path) -> None:
    root = tmp_path / "v2"
    dist.bind_run_manifest(root, _bound_manifest())
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"parent")

    with pytest.raises(RuntimeError, match="v2 manifest root"):
        dist.bind_run_contract(
            root,
            init_checkpoint=checkpoint,
            architecture="entity_graph",
            gamma=1.0,
            gae_lambda=0.95,
            behavior_temperature=1.0,
        )

    assert dist.run_manifest_path(root).is_file()
    assert not dist.run_contract_path(root).exists()


def test_v2_binding_is_exact_and_idempotent(tmp_path: Path) -> None:
    manifest = _bound_manifest()
    root = tmp_path / "run"

    first = dist.bind_run_manifest(root, manifest)
    second = dist.bind_run_manifest(root, manifest)

    assert second == first
    assert first == {
        "schema": dist.RUN_MANIFEST_BINDING_SCHEMA,
        "manifest_sha256": manifest.sha256(),
        "manifest": json.loads(manifest.canonical_json()),
    }
    assert json.loads(dist.run_manifest_path(root).read_text(encoding="utf-8")) == first


def test_v2_binding_concurrent_same_manifest_converges(tmp_path: Path) -> None:
    manifest = _bound_manifest()
    root = tmp_path / "run"

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _index: dist.bind_run_manifest(root, manifest), range(32)
            )
        )

    assert results == [results[0]] * len(results)
    assert results[0]["manifest_sha256"] == manifest.sha256()


def test_concurrent_weight_publishers_receive_distinct_intact_versions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    publishers = 8
    start = Barrier(publishers)

    def publish(index: int) -> dist.PublishedVersion:
        payload = f"publisher-{index}".encode()
        start.wait()

        def save(path: str) -> None:
            Path(path).write_bytes(payload)
            # Keep the transaction open long enough for every competing caller
            # to reach publication; without shared serialization they all pick
            # the same next version and temporary pathname.
            time.sleep(0.01)

        published = dist.publish_weights(root, save, step=index)
        assert Path(published.path).read_bytes() == payload
        return published

    with ThreadPoolExecutor(max_workers=publishers) as executor:
        results = list(executor.map(publish, range(publishers)))

    assert sorted(result.version for result in results) == list(
        range(1, publishers + 1)
    )
    latest = dist.read_version(root)
    assert latest is not None
    assert latest.version == publishers
    assert Path(latest.path).is_file()


def test_learner_cold_start_reuses_actor_bootstrap_version_and_first_dose(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    actor_weights, actor_published = dist.publish_initial_weights(
        root, lambda path: Path(path).write_bytes(b"initializer")
    )
    shard = dist.write_trajectory_shard(
        root,
        "actor",
        0,
        [{"first_dose": True}],
        policy_version=actor_weights.version,
    )

    learner_weights, learner_published = dist.publish_initial_weights(
        root,
        lambda _path: pytest.fail("learner must not republish actor bootstrap"),
    )

    assert actor_published is True
    assert learner_published is False
    assert learner_weights == actor_weights
    assert list(
        dist.iter_unconsumed_shards(
            root,
            min_policy_version=learner_weights.version,
            max_policy_version=learner_weights.version,
        )
    ) == [shard]


def test_learner_cold_start_refuses_unrecoverable_advanced_publication(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    dist.publish_weights(
        root, lambda path: Path(path).write_bytes(b"learned"), step=4
    )

    with pytest.raises(RuntimeError, match="without a recoverable learner checkpoint"):
        dist.publish_initial_weights(
            root, lambda path: Path(path).write_bytes(b"initializer")
        )


def test_v2_binding_rejects_manifest_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "run"
    dist.bind_run_manifest(root, _bound_manifest(seed=1))

    with pytest.raises(dist.RunManifestError, match="manifest mismatch"):
        dist.bind_run_manifest(root, _bound_manifest(seed=2))


def test_v2_binding_allows_empty_skeleton_but_refuses_runtime_artifacts(
    tmp_path: Path,
) -> None:
    empty_root = tmp_path / "empty"
    dist.ensure_run_dirs(empty_root)
    assert dist.bind_run_manifest(empty_root, _bound_manifest())["manifest_sha256"]

    dirty_root = tmp_path / "dirty"
    dist.ensure_run_dirs(dirty_root)
    dist.version_path(dirty_root).write_text('{"version":1}', encoding="utf-8")
    with pytest.raises(dist.RunManifestError, match="preexisting runtime artifacts"):
        dist.bind_run_manifest(dirty_root, _bound_manifest())
    assert not dist.run_manifest_path(dirty_root).exists()


def test_matching_v2_shard_round_trips_and_iterates(tmp_path: Path) -> None:
    root = tmp_path / "run"
    manifest_sha256 = _bound_manifest().sha256()
    shard = dist.write_trajectory_shard(
        root,
        "worker",
        0,
        [{"trajectory": 1}],
        policy_version=7,
        run_manifest_sha256=manifest_sha256,
    )

    envelope = dist.read_trajectory_shard(
        shard, expected_run_manifest_sha256=manifest_sha256
    )
    assert envelope["run_manifest_sha256"] == manifest_sha256
    assert list(
        dist.iter_unconsumed_shards(root, expected_run_manifest_sha256=manifest_sha256)
    ) == [shard]


@pytest.mark.parametrize("kind", ["missing", "mismatch"])
@pytest.mark.parametrize("newest_first", [False, True])
def test_v2_consumer_rejects_unbound_or_mismatched_shard(
    tmp_path: Path, kind: str, newest_first: bool
) -> None:
    root = tmp_path / kind
    expected = _bound_manifest(seed=1).sha256()
    actual = None if kind == "missing" else _bound_manifest(seed=2).sha256()
    shard = dist.write_trajectory_shard(
        root,
        "worker",
        0,
        [{"trajectory": kind}],
        policy_version=1,
        run_manifest_sha256=actual,
    )

    with pytest.raises(dist.RunManifestError, match="trajectory run manifest mismatch"):
        dist.read_trajectory_shard(shard, expected_run_manifest_sha256=expected)
    with pytest.raises(dist.RunManifestError, match="trajectory run manifest mismatch"):
        list(
            dist.iter_unconsumed_shards(
                root,
                newest_first=newest_first,
                expected_run_manifest_sha256=expected,
            )
        )


def test_legacy_contract_and_shard_calls_remain_unchanged(tmp_path: Path) -> None:
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"legacy-parent")
    root = tmp_path / "legacy"
    contract = dist.bind_run_contract(
        root,
        init_checkpoint=checkpoint,
        architecture="entity_graph",
        gamma=1.0,
        gae_lambda=0.95,
        behavior_temperature=0.7,
    )

    assert contract == {
        "schema": dist.RUN_CONTRACT_SCHEMA,
        "initializer_sha256": dist.checkpoint_sha256(checkpoint),
        "architecture": "entity_graph",
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "behavior_temperature": 0.7,
    }
    shard = dist.write_trajectory_shard(
        root, "legacy-worker", 0, [{"legacy": True}], policy_version=3
    )
    envelope = dist.read_trajectory_shard(shard)
    assert set(envelope) == {
        "worker_id",
        "shard_index",
        "policy_version",
        "created_at",
        "trajectories",
    }
    assert list(dist.iter_unconsumed_shards(root)) == [shard]


def test_consumption_receipt_survives_queue_and_marker_cleanup(tmp_path: Path) -> None:
    root = tmp_path / "durable-completion"
    dist.ensure_run_dirs(root)
    shard = dist.write_trajectory_shard(
        root, "worker", 4, [{"trajectory": 1}], policy_version=2
    )

    dist.mark_consumed(root, shard)
    marker = dist.consumed_dir(root) / "worker__shard_000004.pkl"
    receipt = dist.trajectory_completion_path(root, shard)

    assert not shard.exists()
    assert marker.exists()
    assert receipt.exists()
    assert dist.trajectory_is_complete(root, shard)
    assert dist.prune_consumed_markers(root, older_than_secs=0.0) == 1
    assert not marker.exists()
    assert receipt.exists()
    assert dist.trajectory_is_complete(root, shard)


def test_consumed_marker_is_not_pruned_when_payload_deletion_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "transient-delete-failure"
    shard = dist.write_trajectory_shard(
        root, "worker", 0, [{"trajectory": 1}], policy_version=1
    )
    real_remove = dist.os.remove
    attempts = 0

    def fail_once(path: str | Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("transient storage error")
        real_remove(path)

    monkeypatch.setattr(dist.os, "remove", fail_once)
    with pytest.raises(RuntimeError, match="failed to delete consumed PPO shard"):
        dist.mark_consumed(root, shard)

    marker = dist.consumed_dir(root) / "worker__shard_000000.pkl"
    assert shard.exists()
    assert marker.exists()
    assert dist.prune_consumed_markers(root, older_than_secs=0.0) == 0
    assert list(dist.iter_unconsumed_shards(root)) == []

    dist.mark_consumed(root, shard)
    assert not shard.exists()
    assert dist.prune_consumed_markers(root, older_than_secs=0.0) == 1
    assert list(dist.iter_unconsumed_shards(root)) == []


def test_absolute_shard_finalizes_under_relative_run_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path("relative-run")
    dist.ensure_run_dirs(root)
    shard = dist.write_trajectory_shard(
        root, "worker", 1, [{"trajectory": 1}], policy_version=1
    )

    dist.mark_consumed(root, shard.resolve())

    assert not shard.exists()
    assert (dist.consumed_dir(root) / "worker__shard_000001.pkl").exists()
    assert dist.trajectory_is_complete(root, shard.resolve())
