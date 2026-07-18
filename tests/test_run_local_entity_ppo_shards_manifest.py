from __future__ import annotations

import copy
import errno
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from catan_zero.rl import ppo_distributed as dist
from tools import run_local_entity_ppo_shards as actor


TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "selfplay"
    / "ppo_2p_no_trade_v2.json"
)


def _payload() -> dict:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def _write_manifest(
    tmp_path: Path,
    checkpoint: Path,
    *,
    status: str = "bound",
    opponent_mode: str = "fixed",
    payload: dict | None = None,
) -> Path:
    value = copy.deepcopy(payload if payload is not None else _payload())
    value["status"] = status
    value["spec"]["actor"]["opponent_mode"] = opponent_mode
    if status == "bound":
        value["spec"]["identity"]["initializer_sha256"] = (
            f"sha256:{dist.checkpoint_sha256(checkpoint)}"
        )
    path = tmp_path / "run-manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _base_manifest_argv(manifest: Path, checkpoint: Path) -> list[str]:
    return [
        "--run-manifest",
        str(manifest),
        "--run-name",
        "actor-v2",
        "--checkpoint",
        str(checkpoint),
    ]


def test_manifest_maps_exact_actor_science(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"initializer")
    value = _payload()
    value["spec"]["identity"]["vps_to_win"] = 12
    value["spec"]["actor"].update(
        max_decisions=777,
        games_per_shard=3,
        gamma=0.9,
        gae_lambda=0.8,
        action_temperature=0.75,
        value_shaping_coef=0.2,
        value_shaping_scale=88.0,
        value_shaping_opponent_penalty=0.1,
        seed=12345,
        opponent_mode="fixed",
        opponents=["random", "heuristic"],
        pfsp_mode="pfsp",
    )
    manifest_path = _write_manifest(tmp_path, checkpoint, payload=value)
    monkeypatch.setattr(
        actor,
        "validate_canonical_ppo_actor_contract",
        lambda **_kwargs: None,
    )

    args, manifest = actor.resolve_config(
        _base_manifest_argv(manifest_path, checkpoint)
    )
    spec = manifest.spec

    expected = {
        "architecture": spec.identity.architecture,
        "track": spec.identity.track,
        "vps_to_win": spec.identity.vps_to_win,
        "max_decisions": spec.actor.max_decisions,
        "games_per_shard": spec.actor.games_per_shard,
        "gamma": spec.actor.gamma,
        "gae_lambda": spec.actor.gae_lambda,
        "action_temperature": spec.actor.action_temperature,
        "value_shaping_coef": spec.actor.value_shaping_coef,
        "value_shaping_scale": spec.actor.value_shaping_scale,
        "value_shaping_opponent_penalty": (
            spec.actor.value_shaping_opponent_penalty
        ),
        "seed": spec.actor.seed,
        "opponent_mode": spec.actor.opponent_mode,
        "opponents": ",".join(spec.actor.opponents),
        "pfsp_mode": spec.actor.pfsp_mode,
        "run_manifest_sha256": manifest.sha256(),
    }
    assert {name: getattr(args, name) for name in expected} == expected
    assert args.track == "2p_no_trade"


def test_manifest_rejects_checkpoint_sha_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"expected")
    manifest_path = _write_manifest(tmp_path, checkpoint)
    checkpoint.write_bytes(b"different")

    with pytest.raises(SystemExit):
        actor.resolve_config(_base_manifest_argv(manifest_path, checkpoint))


def test_manifest_rejects_template(tmp_path: Path) -> None:
    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"initializer")
    manifest_path = _write_manifest(
        tmp_path,
        checkpoint,
        status="template",
    )

    with pytest.raises(SystemExit):
        actor.resolve_config(_base_manifest_argv(manifest_path, checkpoint))


def test_manifest_refuses_unsupported_league_pfsp_mode(tmp_path: Path) -> None:
    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"initializer")
    manifest_path = _write_manifest(
        tmp_path,
        checkpoint,
        opponent_mode="league",
    )

    with pytest.raises(SystemExit):
        actor.resolve_config(_base_manifest_argv(manifest_path, checkpoint))


@pytest.mark.parametrize(
    "conflict",
    [
        ["--config", "legacy.json"],
        ["--architecture", "entity_graph"],
        ["--track", "2p_no_trade"],
        ["--vps-to-win", "10"],
        ["--games-per-shard", "8"],
        ["--max-decisions", "1000"],
        ["--opponents", "random"],
        ["--seed", "1"],
        ["--gamma", "1.0"],
        ["--action-temperature", "1.0"],
    ],
)
def test_manifest_rejects_explicit_legacy_science_flags(
    tmp_path: Path,
    conflict: list[str],
) -> None:
    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"initializer")
    manifest_path = _write_manifest(tmp_path, checkpoint)

    with pytest.raises(SystemExit):
        actor.resolve_config(
            [*_base_manifest_argv(manifest_path, checkpoint), *conflict]
        )


def test_manifest_allows_deployment_overrides(tmp_path: Path) -> None:
    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"initializer")
    manifest_path = _write_manifest(tmp_path, checkpoint)

    args, _manifest = actor.resolve_config(
        [
            *_base_manifest_argv(manifest_path, checkpoint),
            "--run-base",
            str(tmp_path / "runs"),
            "--devices",
            "cuda:2,cuda:3",
            "--games",
            "19",
            "--workers",
            "3",
            "--publish",
        ]
    )

    assert args.run_base == str(tmp_path / "runs")
    assert args.run_name == "actor-v2"
    assert args.checkpoint == str(checkpoint)
    assert args.devices == "cuda:2,cuda:3"
    assert args.games == 19
    assert args.workers == 3
    assert args.publish is True


def test_v2_manifest_binder_is_used_instead_of_v1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"initializer")
    manifest_path = _write_manifest(tmp_path, checkpoint)
    args, manifest = actor.resolve_config(
        [
            *_base_manifest_argv(manifest_path, checkpoint),
            "--run-base",
            str(tmp_path),
        ]
    )
    calls: list[tuple] = []
    monkeypatch.setattr(
        dist,
        "bind_run_manifest",
        lambda root, bound: calls.append((root, bound)),
    )
    monkeypatch.setattr(
        dist,
        "bind_run_contract",
        lambda *_args, **_kwargs: pytest.fail("v2 actor invoked v1 binder"),
    )

    root = actor._bind_run_root(args, manifest)  # noqa: SLF001

    assert root == dist.run_root(args.run_base, args.run_name)
    assert calls == [(root, manifest)]


def test_legacy_resolution_and_v1_binding_remain_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"legacy")
    args, manifest = actor.resolve_config(
        [
            "--run-base",
            str(tmp_path),
            "--run-name",
            "legacy",
            "--checkpoint",
            str(checkpoint),
            "--games-per-shard",
            "5",
            "--seed",
            "99",
        ]
    )
    calls: list[dict] = []
    monkeypatch.setattr(
        dist,
        "bind_run_contract",
        lambda *_args, **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        dist,
        "bind_run_manifest",
        lambda *_args, **_kwargs: pytest.fail("legacy actor invoked v2 binder"),
    )

    actor._bind_run_root(args, manifest)  # noqa: SLF001

    assert manifest is None
    assert args.run_manifest_sha256 is None
    assert args.games_per_shard == 5
    assert args.seed == 99
    assert calls[0]["init_checkpoint"] == str(checkpoint)


def test_worker_stamps_manifest_sha_on_every_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[dict] = []
    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"initializer")
    value = _payload()
    value["spec"]["actor"]["games_per_shard"] = 1
    manifest_path = _write_manifest(tmp_path, checkpoint, payload=value)
    args, manifest = actor.resolve_config(
        [
            *_base_manifest_argv(manifest_path, checkpoint),
            "--run-base",
            str(tmp_path),
            "--devices",
            "cpu",
            "--games",
            "2",
            "--workers",
            "1",
        ]
    )
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    published = SimpleNamespace(path=str(weights), version=3)
    launch = actor._prepare_launch(args, published)  # noqa: SLF001
    payloads, _devices, _games = actor._build_worker_payloads(  # noqa: SLF001
        args,
        published,
        launch,
    )

    class _Policy:
        model = SimpleNamespace(eval=lambda: None)

    monkeypatch.setattr(actor, "load_ppo_policy", lambda *_args, **_kwargs: _Policy())
    monkeypatch.setattr(
        actor,
        "parse_track",
        lambda *_args, **_kwargs: SimpleNamespace(players=2),
    )
    monkeypatch.setattr(actor, "make_named_policy", lambda _name: object())
    monkeypatch.setattr(
        actor,
        "collect_ppo_episode",
        lambda *_args, **_kwargs: SimpleNamespace(samples=[1, 2]),
    )
    monkeypatch.setattr(
        dist,
        "write_trajectory_shard",
        lambda *_args, **kwargs: writes.append(kwargs),
    )
    report = actor._worker(payloads[0])  # noqa: SLF001

    assert report["shards"] == 2
    assert len(writes) == 2
    assert all(
        write["run_manifest_sha256"] == manifest.sha256() for write in writes
    )


def test_distinct_launches_use_disjoint_paths_seeds_and_game_offsets(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"initializer")
    args, _manifest = actor.resolve_config(
        [
            "--run-base",
            str(tmp_path),
            "--run-name",
            "integrity",
            "--checkpoint",
            str(checkpoint),
            "--games",
            "4",
            "--workers",
            "2",
        ]
    )
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    published = SimpleNamespace(path=str(weights), version=7)

    first = actor._prepare_launch(args, published)  # noqa: SLF001
    second = actor._prepare_launch(args, published)  # noqa: SLF001
    first_payloads, _devices, _games = actor._build_worker_payloads(  # noqa: SLF001
        args, published, first
    )
    second_payloads, _devices, _games = actor._build_worker_payloads(  # noqa: SLF001
        args, published, second
    )

    assert first["launch_id"] != second["launch_id"]
    assert {item["worker_id"] for item in first_payloads}.isdisjoint(
        item["worker_id"] for item in second_payloads
    )
    assert {item["seed"] for item in first_payloads}.isdisjoint(
        item["seed"] for item in second_payloads
    )
    first_games = {
        item["game_offset"] + game
        for item in first_payloads
        for game in range(item["games"])
    }
    second_games = {
        item["game_offset"] + game
        for item in second_payloads
        for game in range(item["games"])
    }
    assert first_games.isdisjoint(second_games)


def test_named_launch_resumes_exact_identity_and_refuses_drift(tmp_path: Path) -> None:
    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"initializer")
    base_argv = [
        "--run-base",
        str(tmp_path),
        "--run-name",
        "resume",
        "--checkpoint",
        str(checkpoint),
        "--launch-id",
        "retry-001",
        "--games",
        "4",
        "--workers",
        "2",
    ]
    args, _manifest = actor.resolve_config(base_argv)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    published = SimpleNamespace(path=str(weights), version=2)

    first = actor._prepare_launch(args, published)  # noqa: SLF001
    resumed = actor._prepare_launch(args, published)  # noqa: SLF001

    assert resumed == first
    changed, _manifest = actor.resolve_config([*base_argv, "--devices", "cpu"])
    assert actor._prepare_launch(changed, published) == first  # noqa: SLF001
    changed.seed += 1
    with pytest.raises(RuntimeError, match="launch contract mismatch"):
        actor._prepare_launch(changed, published)  # noqa: SLF001


def test_worker_resume_skips_consumed_shards_instead_of_hiding_rewrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = dist.run_root(tmp_path, "resume-worker")
    dist.ensure_run_dirs(root)
    payload = {
        "run_base": str(tmp_path),
        "run_name": "resume-worker",
        "launch_id": "retry-001",
        "worker_id": "local_retry-001_000",
        "checkpoint": str(tmp_path / "weights.pt"),
        "policy_version": 3,
        "architecture": "entity_graph",
        "device": "cpu",
        "track": "2p_no_trade",
        "vps_to_win": 10,
        "games": 2,
        "game_offset": 100,
        "games_per_shard": 1,
        "max_decisions": 10,
        "opponents": "random",
        "opponent_mode": "fixed",
        "pfsp_mode": "pfsp",
        "seed": 123,
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "value_shaping_coef": 0.0,
        "value_shaping_scale": 100.0,
        "value_shaping_opponent_penalty": 0.05,
        "action_temperature": 1.0,
        "run_manifest_sha256": None,
    }
    worker_dir = dist.trajectories_dir(root, payload["worker_id"])
    worker_dir.mkdir(parents=True)
    consumed = dist.consumed_dir(root)
    consumed.mkdir(parents=True, exist_ok=True)
    dist.write_trajectory_shard(
        root,
        payload["worker_id"],
        0,
        [{"published_before_actor_receipt": True}],
        policy_version=payload["policy_version"],
    )
    calls = 0

    class _Policy:
        model = SimpleNamespace(eval=lambda: None)

    def collect(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(samples=[1])

    monkeypatch.setattr(actor, "load_ppo_policy", lambda *_args, **_kwargs: _Policy())
    monkeypatch.setattr(
        actor, "parse_track", lambda *_args, **_kwargs: SimpleNamespace(players=2)
    )
    monkeypatch.setattr(actor, "make_named_policy", lambda _name: object())
    monkeypatch.setattr(actor, "collect_ppo_episode", collect)

    report = actor._worker(payload)  # noqa: SLF001

    assert calls == 1
    assert report["games"] == 1
    assert report["shards"] == 1
    assert (worker_dir / "shard_000000.pkl").exists()
    assert dist.trajectory_is_complete(root, worker_dir / "shard_000000.pkl")
    assert (worker_dir / "shard_000001.pkl").exists()
    dist.mark_consumed(root, worker_dir / "shard_000000.pkl")
    dist.mark_consumed(root, worker_dir / "shard_000001.pkl")
    assert dist.prune_consumed_markers(root, older_than_secs=0.0) == 2
    assert not (consumed / f'{payload["worker_id"]}__shard_000001.pkl').exists()
    assert dist.trajectory_is_complete(root, worker_dir / "shard_000001.pkl")

    resumed = actor._worker(payload)  # noqa: SLF001

    assert calls == 1
    assert resumed["games"] == 0
    assert resumed["shards"] == 0
    assert resumed["resumed_games"] == 2
    assert resumed["resumed_shards"] == 2


def test_publish_flag_cannot_replace_existing_learned_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = dist.run_root(tmp_path, "publish-safe")
    dist.ensure_run_dirs(root)
    existing = SimpleNamespace(version=9, step=42, path=str(tmp_path / "learned.pt"))
    args = SimpleNamespace(
        publish=True,
        checkpoint=str(tmp_path / "initializer.pt"),
        architecture="entity_graph",
    )
    monkeypatch.setattr(dist, "read_version", lambda _root: existing)
    monkeypatch.setattr(
        actor,
        "load_ppo_policy",
        lambda *_args, **_kwargs: pytest.fail("initializer must not be loaded"),
    )
    monkeypatch.setattr(
        dist,
        "publish_weights",
        lambda *_args, **_kwargs: pytest.fail("learned weights must not be replaced"),
    )

    with pytest.raises(RuntimeError, match="bootstrap-only"):
        actor._resolve_published_weights(args, root)  # noqa: SLF001


def test_named_resume_snapshots_mutable_current_weights_before_learner_advances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"initializer")
    args, _manifest = actor.resolve_config(
        [
            "--run-base",
            str(tmp_path),
            "--run-name",
            "advanced-learner",
            "--checkpoint",
            str(checkpoint),
            "--launch-id",
            "partial-launch",
        ]
    )
    root = dist.run_root(args.run_base, args.run_name)
    dist.ensure_run_dirs(root)
    dist.publish_weights(
        root, lambda path: Path(path).write_bytes(b"version two"), step=10
    )
    weights_v2 = dist.current_weights_path(root)
    original = dist.PublishedVersion(
        version=2,
        step=10,
        updated_at=12.5,
        path=str(weights_v2),
    )
    launch = actor._prepare_launch(args, original)  # noqa: SLF001
    dist.publish_weights(
        root, lambda path: Path(path).write_bytes(b"version three"), step=11
    )
    monkeypatch.setattr(
        dist,
        "read_version",
        lambda _root: pytest.fail("resume must not consult mutable current weights"),
    )

    resumed, published = actor._resolve_launch_and_weights(args, root)  # noqa: SLF001

    assert resumed == launch
    assert published.version == 2
    assert published.step == 10
    assert published.updated_at == 12.5
    assert Path(published.path) == Path(launch["checkpoint"])
    assert Path(published.path).read_bytes() == b"version two"
    assert dist.checkpoint_sha256(published.path) == launch["checkpoint_sha256"]


def test_named_resume_fails_closed_when_bound_weights_were_removed(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"initializer")
    args, _manifest = actor.resolve_config(
        [
            "--run-base",
            str(tmp_path),
            "--run-name",
            "missing-policy",
            "--checkpoint",
            str(checkpoint),
            "--launch-id",
            "partial-launch",
        ]
    )
    root = dist.run_root(args.run_base, args.run_name)
    dist.ensure_run_dirs(root)
    weights = dist.policy_dir(root) / "weights_v2.pt"
    weights.write_bytes(b"version two")
    actor._prepare_launch(  # noqa: SLF001
        args,
        dist.PublishedVersion(version=2, step=1, updated_at=2.0, path=str(weights)),
    )
    Path(
        json.loads(
            (
                root
                / "actor_launches"
                / "partial-launch"
                / "launch.json"
            ).read_text(encoding="utf-8")
        )["checkpoint"]
    ).unlink()

    with pytest.raises(RuntimeError, match="immutable policy checkpoint.*unavailable"):
        actor._resolve_launch_and_weights(args, root)  # noqa: SLF001


def test_launch_policy_snapshot_survives_versioned_weight_gc(tmp_path: Path) -> None:
    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"initializer")
    args, _manifest = actor.resolve_config(
        [
            "--run-base",
            str(tmp_path),
            "--run-name",
            "gc-resume",
            "--checkpoint",
            str(checkpoint),
            "--launch-id",
            "long-running-launch",
        ]
    )
    root = dist.run_root(args.run_base, args.run_name)
    dist.ensure_run_dirs(root)
    first = dist.publish_weights(
        root, lambda path: Path(path).write_bytes(b"version one"), step=0
    )
    launch = actor._prepare_launch(args, first)  # noqa: SLF001

    for version in range(2, dist.KEEP_VERSIONED_WEIGHTS + 3):
        dist.publish_weights(
            root,
            lambda path, version=version: Path(path).write_bytes(
                f"version {version}".encode()
            ),
            step=version,
        )

    assert not Path(first.path).exists()
    resumed, published = actor._resolve_launch_and_weights(args, root)  # noqa: SLF001
    assert resumed == launch
    assert published.version == first.version
    assert Path(published.path).read_bytes() == b"version one"


def test_orphan_policy_snapshot_is_recovered_before_launch_binding(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"initializer")
    args, _manifest = actor.resolve_config(
        [
            "--run-base",
            str(tmp_path),
            "--run-name",
            "orphan-recovery",
            "--checkpoint",
            str(checkpoint),
            "--launch-id",
            "crashed-launch",
        ]
    )
    root = dist.run_root(args.run_base, args.run_name)
    dist.ensure_run_dirs(root)
    source = dist.policy_dir(root) / "weights_v4.pt"
    source.write_bytes(b"current policy")
    launch_dir = root / "actor_launches" / "crashed-launch"
    launch_dir.mkdir(parents=True)
    (launch_dir / "policy.pt").write_bytes(b"orphaned stale bytes")

    launch = actor._prepare_launch(  # noqa: SLF001
        args,
        dist.PublishedVersion(version=4, step=20, updated_at=5.0, path=str(source)),
    )

    assert Path(launch["checkpoint"]).read_bytes() == b"current policy"
    assert dist.checkpoint_sha256(launch["checkpoint"]) == launch["checkpoint_sha256"]
    assert (launch_dir / "policy_snapshot.json").is_file()


def test_staged_policy_snapshot_resumes_after_crash_before_launch_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"initializer")
    args, _manifest = actor.resolve_config(
        [
            "--run-base",
            str(tmp_path),
            "--run-name",
            "staged-recovery",
            "--checkpoint",
            str(checkpoint),
            "--launch-id",
            "crashed-launch",
        ]
    )
    root = dist.run_root(args.run_base, args.run_name)
    dist.ensure_run_dirs(root)
    first = dist.publish_weights(
        root, lambda path: Path(path).write_bytes(b"version one"), step=1
    )
    real_bind = actor._atomic_bind_json  # noqa: SLF001

    def crash_before_launch(path, payload):
        if path.name == "launch.json":
            raise RuntimeError("simulated crash")
        return real_bind(path, payload)

    monkeypatch.setattr(actor, "_atomic_bind_json", crash_before_launch)
    with pytest.raises(RuntimeError, match="simulated crash"):
        actor._prepare_launch(args, first)  # noqa: SLF001
    monkeypatch.setattr(actor, "_atomic_bind_json", real_bind)
    latest = dist.publish_weights(
        root, lambda path: Path(path).write_bytes(b"version two"), step=2
    )
    assert latest.version != first.version
    monkeypatch.setattr(
        dist,
        "read_version",
        lambda _root: pytest.fail("staged recovery must not consult mutable current weights"),
    )

    launch, published = actor._resolve_launch_and_weights(args, root)  # noqa: SLF001

    assert launch["policy_version"] == first.version
    assert published.version == first.version
    assert Path(launch["checkpoint"]).read_bytes() == b"version one"


def _completed_launch_fixture(tmp_path: Path):
    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"initializer")
    args, _manifest = actor.resolve_config(
        [
            "--run-base",
            str(tmp_path),
            "--run-name",
            "lifecycle",
            "--checkpoint",
            str(checkpoint),
            "--launch-id",
            "bounded-launch",
            "--games",
            "5",
            "--workers",
            "2",
            "--games-per-shard",
            "2",
        ]
    )
    root = dist.run_root(args.run_base, args.run_name)
    dist.ensure_run_dirs(root)
    weights = dist.policy_dir(root) / "weights_v2.pt"
    weights.write_bytes(b"policy")
    launch = actor._prepare_launch(  # noqa: SLF001
        args,
        dist.PublishedVersion(version=2, step=3, updated_at=4.0, path=str(weights)),
    )
    payloads, _devices, _games = actor._build_worker_payloads(  # noqa: SLF001
        args, actor._published_from_launch(launch), launch  # noqa: SLF001
    )
    return args, root, launch, payloads


def test_completed_launch_cleanup_is_bounded_and_idempotent(tmp_path: Path) -> None:
    _args, root, launch, payloads = _completed_launch_fixture(tmp_path)
    shard_paths = []
    for payload in payloads:
        shard_count = (payload["games"] + payload["games_per_shard"] - 1) // payload[
            "games_per_shard"
        ]
        for shard_index in range(shard_count):
            shard = dist.trajectories_dir(root, payload["worker_id"]) / (
                f"shard_{shard_index:06d}.pkl"
            )
            dist.mark_trajectory_complete(root, shard)
            shard_paths.append(shard)

    first = actor._finalize_launch_if_complete(root, launch)  # noqa: SLF001
    second = actor._finalize_launch_if_complete(root, launch)  # noqa: SLF001

    assert first == second
    assert first is not None
    assert not Path(launch["checkpoint"]).exists()
    assert not (
        root / "actor_launches" / launch["launch_id"] / "policy_snapshot.json"
    ).exists()
    assert all(not dist.trajectory_completion_path(root, path).exists() for path in shard_paths)
    assert actor._load_launch_completion(root, launch) == first  # noqa: SLF001


def test_completed_resume_needs_no_policy_bytes_or_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, root, launch, payloads = _completed_launch_fixture(tmp_path)
    for payload in payloads:
        shard_count = (payload["games"] + payload["games_per_shard"] - 1) // payload[
            "games_per_shard"
        ]
        for shard_index in range(shard_count):
            dist.mark_trajectory_complete(
                root,
                dist.trajectories_dir(root, payload["worker_id"])
                / f"shard_{shard_index:06d}.pkl",
            )
    actor._finalize_launch_if_complete(root, launch)  # noqa: SLF001
    monkeypatch.setattr(
        actor,
        "load_ppo_policy",
        lambda *_args, **_kwargs: pytest.fail("completed launch must not load a model"),
    )
    monkeypatch.setattr(
        dist,
        "read_version",
        lambda _root: pytest.fail("completed launch must not consult current weights"),
    )

    resumed, published = actor._resolve_launch_and_weights(args, root)  # noqa: SLF001

    assert resumed == launch
    assert published is None
    assert actor._build_worker_payloads(args, published, resumed)[0] == []  # noqa: SLF001


def test_cleanup_resumes_after_crash_immediately_after_aggregate_binding(
    tmp_path: Path,
) -> None:
    _args, root, launch, payloads = _completed_launch_fixture(tmp_path)
    shard_paths = []
    for payload in payloads:
        shard_count = (payload["games"] + payload["games_per_shard"] - 1) // payload[
            "games_per_shard"
        ]
        for shard_index in range(shard_count):
            shard = dist.trajectories_dir(root, payload["worker_id"]) / (
                f"shard_{shard_index:06d}.pkl"
            )
            dist.mark_trajectory_complete(root, shard)
            shard_paths.append(shard)
    actor._atomic_bind_json(  # noqa: SLF001
        dist.launch_completion_path(root, launch["launch_id"]),
        dist.launch_completion_payload(launch),
    )
    assert Path(launch["checkpoint"]).exists()
    assert any(dist.trajectory_completion_path(root, path).exists() for path in shard_paths)

    completion = actor._finalize_launch_if_complete(root, launch)  # noqa: SLF001

    assert completion == dist.launch_completion_payload(launch)
    assert not Path(launch["checkpoint"]).exists()
    assert all(not dist.trajectory_completion_path(root, path).exists() for path in shard_paths)


def test_launch_completion_refuses_missing_or_forged_schedule(tmp_path: Path) -> None:
    args, root, launch, _payloads = _completed_launch_fixture(tmp_path)

    assert actor._finalize_launch_if_complete(root, launch) is None  # noqa: SLF001
    assert Path(launch["checkpoint"]).exists()
    completion = root / "actor_launches" / launch["launch_id"] / "launch_complete.json"
    completion.write_text('{"schema":"forged"}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="launch completion"):
        actor._resolve_launch_and_weights(args, root)  # noqa: SLF001


def test_learner_does_not_recreate_shard_receipt_after_aggregate_cleanup(
    tmp_path: Path,
) -> None:
    _args, root, launch, payloads = _completed_launch_fixture(tmp_path)
    shard_paths = []
    for payload in payloads:
        shard_count = (payload["games"] + payload["games_per_shard"] - 1) // payload[
            "games_per_shard"
        ]
        for shard_index in range(shard_count):
            shard = dist.trajectories_dir(root, payload["worker_id"]) / (
                f"shard_{shard_index:06d}.pkl"
            )
            dist.mark_trajectory_complete(root, shard)
            shard_paths.append(shard)
    actor._finalize_launch_if_complete(root, launch)  # noqa: SLF001
    shard_paths[0].parent.mkdir(parents=True, exist_ok=True)
    shard_paths[0].write_bytes(b"already queued")

    dist.mark_consumed(root, shard_paths[0])

    assert not shard_paths[0].exists()
    assert not dist.trajectory_completion_path(root, shard_paths[0]).exists()


def test_snapshot_copy_fallback_handles_link_permission_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.pt"
    destination = tmp_path / "launch" / "policy.pt"
    source.write_bytes(b"policy bytes")
    real_link = actor.os.link

    def permission_denied_once(src, dst):
        if Path(dst) == destination:
            raise OSError(errno.EPERM, "hard links unavailable")
        return real_link(src, dst)

    monkeypatch.setattr(actor.os, "link", permission_denied_once)

    digest = actor._snapshot_policy(source, destination)  # noqa: SLF001

    assert destination.read_bytes() == source.read_bytes()
    assert digest == dist.checkpoint_sha256(destination)
