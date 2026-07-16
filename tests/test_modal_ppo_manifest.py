from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from catan_zero.rl.ppo_run_manifest import PPORunManifest


_REPO = Path(__file__).resolve().parents[1]
_FACTORY_PATH = _REPO / "tools" / "modal_ppo_factory.py"
_TEMPLATE = _REPO / "configs" / "selfplay" / "ppo_2p_no_trade_v2.json"


class _ModalImage:
    @classmethod
    def debian_slim(cls, **_kwargs):
        return cls()

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: self


class _ModalVolume:
    @classmethod
    def from_name(cls, *_args, **_kwargs):
        return cls()

    def reload(self) -> None:
        return None

    def commit(self) -> None:
        return None


class _ModalApp:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def function(self, *_args, **_kwargs):
        return lambda function: function

    def local_entrypoint(self, *_args, **_kwargs):
        return lambda function: function


@pytest.fixture
def factory(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(App=_ModalApp, Image=_ModalImage, Volume=_ModalVolume),
    )
    spec = importlib.util.spec_from_file_location(
        "_modal_ppo_factory_manifest_test", _FACTORY_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bound_manifest(checkpoint: Path, *, mutate=None) -> PPORunManifest:
    raw = json.loads(_TEMPLATE.read_text(encoding="utf-8"))
    raw["status"] = "bound"
    raw["spec"]["identity"]["initializer_sha256"] = (
        "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    if mutate is not None:
        mutate(raw)
    return PPORunManifest.from_dict(raw)


def _write_manifest(path: Path, manifest: PPORunManifest) -> Path:
    path.write_text(manifest.canonical_json(), encoding="utf-8")
    return path


def _actor_payload_kwargs(factory, checkpoint: Path, manifest_path: Path) -> dict:
    return {
        "run_name": "manifest-run",
        "init_checkpoint": str(checkpoint),
        "containers": 2,
        "games_per_container": 3,
        "cpu_workers": 1,
        "games_per_shard": 8,
        "commit_every_shards": 1,
        "commit_min_secs": 0.0,
        "opponent_cache_size": 2,
        "quantize_rollout": False,
        "cold_start_timeout_secs": 0.0,
        "policy_poll_secs": 1.0,
        "max_actor_lag": 2,
        "lag_stall_rounds": 2,
        "lag_stall_sleep": 0.0,
        "seed": 1,
        "architecture": "entity_graph",
        "device": "cpu",
        "track": "2p_no_trade",
        "vps_to_win": 10,
        "max_decisions": 1_000,
        "opponent_mode": "league",
        "opponents": factory.DEFAULT_OPPONENTS,
        "pfsp_mode": "pfsp",
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "value_shaping_coef": 0.0,
        "value_shaping_scale": 100.0,
        "value_shaping_opponent_penalty": 0.05,
        "action_temperature": 1.0,
        "run_manifest": str(manifest_path),
    }


def test_manifest_only_actor_entrypoint_derives_science_and_preserves_order(
    factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"exact-parent")
    manifest = _bound_manifest(
        checkpoint,
        mutate=lambda raw: raw["spec"]["actor"].update(
            {
                "max_decisions": 777,
                "games_per_shard": 5,
                "seed": 123,
                "opponents": ["catanatron_ab3", "random", "heuristic"],
            }
        ),
    )
    manifest_path = _write_manifest(tmp_path / "bound.json", manifest)

    captured: dict = {}
    monkeypatch.setattr(factory, "_launch", lambda **kwargs: captured.update(kwargs))
    factory.launch_ppo_actors_from_manifest(
        str(manifest_path),
        init_checkpoint=str(checkpoint),
        containers=2,
        games_per_container=3,
    )
    payloads = factory._payloads(**captured)

    assert captured["quantize_rollout"] is False
    assert [payload["seed"] for payload in payloads] == [123, 126]
    assert payloads[0]["max_decisions"] == 777
    assert payloads[0]["games_per_shard"] == 5
    assert payloads[0]["opponents"] == "catanatron_ab3,random,heuristic"
    assert payloads[0]["run_manifest_json"] == manifest.canonical_json()
    assert payloads[0]["run_manifest_sha256"] == manifest.sha256()
    assert factory._run_manifest_chunk_fields(payloads[0]) == {
        "run_manifest_sha256": manifest.sha256()
    }
    # Container 1's seed is a deterministic partition of the manifest base seed.
    factory._reject_manifest_science_conflicts(
        payloads[1], factory._actor_container_manifest_science(manifest, payloads[1])
    )


def test_manifest_rejects_template_hash_drift_initializer_drift_and_real_conflict(
    factory, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"exact-parent")
    template = tmp_path / "template.json"
    template.write_text(_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="templates cannot run"):
        factory._manifest_envelope_from_path(
            str(template), init_checkpoint=str(checkpoint)
        )

    manifest = _bound_manifest(checkpoint)
    envelope = {
        "run_manifest_json": manifest.canonical_json(),
        "run_manifest_sha256": "sha256:" + "f" * 64,
    }
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        factory._bound_manifest_from_payload(
            envelope,
            init_checkpoint=str(checkpoint),
            require_initializer=True,
        )

    with pytest.raises(ValueError, match="canonical JSON and SHA-256"):
        factory._bound_manifest_from_payload(
            {"run_manifest_json": manifest.canonical_json()},
            init_checkpoint=str(checkpoint),
            require_initializer=True,
        )

    with pytest.raises(ValueError, match="canonical JSON"):
        factory._bound_manifest_from_payload(
            {
                "run_manifest_json": json.dumps(manifest.to_dict(), indent=2),
                "run_manifest_sha256": manifest.sha256(),
            },
            init_checkpoint=str(checkpoint),
            require_initializer=True,
        )

    wrong_checkpoint = tmp_path / "wrong.pt"
    wrong_checkpoint.write_bytes(b"wrong-parent")
    envelope["run_manifest_sha256"] = manifest.sha256()
    with pytest.raises(ValueError, match="init checkpoint SHA-256"):
        factory._bound_manifest_from_payload(
            envelope,
            init_checkpoint=str(wrong_checkpoint),
            require_initializer=True,
        )

    manifest_path = _write_manifest(tmp_path / "bound.json", manifest)
    kwargs = _actor_payload_kwargs(factory, checkpoint, manifest_path)
    kwargs["max_decisions"] = 999  # conflicts with manifest-owned value 1000
    with pytest.raises(ValueError, match="max_decisions"):
        factory._payloads(**kwargs)


def test_chunk_writer_stamps_every_manifest_shard(
    factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from catan_zero.rl import ppo_distributed as ppd
    from catan_zero.rl import ppo_policy_factory, torch_ppo

    writes: list[dict] = []

    class _Model:
        def eval(self) -> None:
            return None

    class _Resolver:
        effective_mode = "fixed"

        def __init__(self, **_kwargs) -> None:
            pass

        def opponents_for(self, _seats, _rng) -> dict:
            return {}

    monkeypatch.setattr(factory, "VOLUME_ROOT", tmp_path)
    monkeypatch.setattr(factory, "_OpponentResolver", _Resolver)
    monkeypatch.setattr(factory, "_tune_rollout_threads", lambda: None)
    monkeypatch.setattr(
        ppo_policy_factory,
        "load_ppo_policy",
        lambda *_args, **_kwargs: SimpleNamespace(model=_Model()),
    )
    monkeypatch.setattr(
        ppo_policy_factory,
        "validate_canonical_ppo_actor_contract",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        torch_ppo,
        "collect_ppo_episode",
        lambda *_args, **_kwargs: SimpleNamespace(samples=[1]),
    )
    monkeypatch.setitem(
        sys.modules,
        "factory_common",
        SimpleNamespace(
            parse_track=lambda *_args, **_kwargs: SimpleNamespace(players=2)
        ),
    )

    def _write(*_args, **kwargs):
        writes.append(kwargs)
        return tmp_path / f"shard-{len(writes)}.pkl"

    monkeypatch.setattr(ppd, "write_trajectory_shard", _write)
    sha256 = "sha256:" + "a" * 64
    factory._run_actor_chunk(
        {
            "run_name": "run",
            "worker_id": "actor_0",
            "games": 3,
            "game_offset": 0,
            "shard_base": 0,
            "games_per_shard": 2,
            "seed": 1,
            "policy_path": "unused.pt",
            "policy_version": 7,
            "track": "2p_no_trade",
            "vps_to_win": 10,
            "max_decisions": 10,
            "run_manifest_sha256": sha256,
        }
    )

    assert len(writes) == 2
    assert {write["run_manifest_sha256"] for write in writes} == {sha256}


def test_actor_binds_v2_before_artifacts_and_preserves_legacy_binder(
    factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from catan_zero.rl import ppo_distributed as ppd

    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"exact-parent")
    manifest = _bound_manifest(checkpoint)
    manifest_path = _write_manifest(tmp_path / "bound.json", manifest)
    events: list[str] = []
    original_bind_v2 = ppd.bind_run_manifest
    original_bind_v1 = ppd.bind_run_contract
    original_ensure = ppd.ensure_run_dirs

    class _Pool:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(factory, "VOLUME_ROOT", tmp_path / "runs")
    monkeypatch.setattr(factory, "ProcessPoolExecutor", _Pool)
    monkeypatch.setattr(factory.volume, "reload", lambda: events.append("reload"))
    monkeypatch.setattr(factory.volume, "commit", lambda: events.append("commit"))
    monkeypatch.setattr(
        factory,
        "_cold_start_wait",
        lambda *_args, **_kwargs: events.append("cold_wait") or None,
    )
    monkeypatch.setattr(
        ppd,
        "bind_run_manifest",
        lambda *args, **kwargs: (
            events.append("bind_v2") or original_bind_v2(*args, **kwargs)
        ),
    )
    monkeypatch.setattr(
        ppd,
        "bind_run_contract",
        lambda *args, **kwargs: (
            events.append("bind_v1") or original_bind_v1(*args, **kwargs)
        ),
    )
    monkeypatch.setattr(
        ppd,
        "ensure_run_dirs",
        lambda *args, **kwargs: (
            events.append("ensure") or original_ensure(*args, **kwargs)
        ),
    )

    kwargs = _actor_payload_kwargs(factory, checkpoint, manifest_path)
    kwargs["containers"] = 1
    kwargs["games_per_container"] = 0
    v2_payload = factory._payloads(**kwargs)[0]
    factory._run_actor(v2_payload)
    assert (
        events.index("reload")
        < events.index("bind_v2")
        < events.index("ensure")
        < events.index("commit")
        < events.index("cold_wait")
    )
    assert "bind_v1" not in events

    events.clear()
    kwargs.pop("run_manifest")
    kwargs["run_name"] = "legacy-run"
    legacy_payload = factory._payloads(**kwargs)[0]
    factory._run_actor(legacy_payload)
    assert events.index("ensure") < events.index("cold_wait") < events.index("bind_v1")
    assert "bind_v2" not in events

    events.clear()
    kwargs = _actor_payload_kwargs(factory, checkpoint, manifest_path)
    kwargs["containers"] = 1
    kwargs["games_per_container"] = 0
    kwargs["run_name"] = "commit-failure"
    failure_payload = factory._payloads(**kwargs)[0]

    def _fail_commit() -> None:
        events.append("commit_failed")
        raise RuntimeError("commit failed")

    monkeypatch.setattr(factory.volume, "commit", _fail_commit)
    with pytest.raises(RuntimeError, match="commit failed"):
        factory._run_actor(failure_payload)
    assert events.index("bind_v2") < events.index("commit_failed")
    assert "cold_wait" not in events


def test_learner_manifest_uses_resolver_and_temp_outside_run_root(
    factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"exact-parent")
    manifest = _bound_manifest(
        checkpoint,
        mutate=lambda raw: raw["spec"]["learner"].update(
            {"max_steps": 17, "lr": 0.000123, "minibatch_size": 321}
        ),
    )
    manifest_path = _write_manifest(tmp_path / "bound.json", manifest)
    run_base = tmp_path / "runs"
    original_learner = factory.ppo_learner
    spawned: dict[str, object] = {}

    class _LearnerRemote:
        def spawn(self, payload):
            spawned["payload"] = payload
            return SimpleNamespace(object_id="test-object")

    monkeypatch.setattr(factory, "ppo_learner", _LearnerRemote())
    factory.launch_learner_from_manifest(
        str(manifest_path),
        run_name="learner-run",
        init_checkpoint=str(checkpoint),
        run_base=str(run_base),
    )
    payload = spawned["payload"]
    assert isinstance(payload, dict)
    assert payload["max_steps"] == 17
    assert payload["lr"] == 0.000123
    assert payload["minibatch_size"] == 321

    calls: dict[str, object] = {}

    def _resolve(argv):
        calls["argv"] = argv
        path = Path(argv[argv.index("--run-manifest") + 1])
        calls["manifest_path"] = path
        calls["manifest_json"] = path.read_text(encoding="utf-8")
        return SimpleNamespace(run_manifest_sha256=manifest.sha256()), SimpleNamespace()

    def _train(config, **kwargs) -> None:
        calls["config"] = config
        calls["hooks"] = kwargs

    monkeypatch.setitem(
        sys.modules,
        "ppo_distributed_learner",
        SimpleNamespace(resolve_config=_resolve, train=_train),
    )
    monkeypatch.setattr(factory, "REMOTE_ROOT", _REPO)
    monkeypatch.setattr(factory, "ppo_learner", original_learner)

    original_learner(payload)

    argv = calls["argv"]
    assert isinstance(argv, list) and "--run-manifest" in argv
    temporary = calls["manifest_path"]
    assert isinstance(temporary, Path)
    assert temporary.parent == Path("/tmp")
    assert run_base not in temporary.parents
    assert calls["manifest_json"] == manifest.canonical_json()
    assert calls["config"].run_manifest_sha256 == manifest.sha256()


def test_learner_internal_typeerror_propagates_without_retry(
    factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    class _Config:
        __dataclass_fields__ = {
            key: None
            for key in (
                "run_base",
                "run_name",
                "init_checkpoint",
                "architecture",
                "device",
            )
        }

        def __init__(self, **values) -> None:
            self.__dict__.update(values)

    def _train(_config, **hooks) -> None:
        calls.append(hooks)
        raise TypeError("internal learner failure")

    monkeypatch.setitem(
        sys.modules,
        "ppo_distributed_learner",
        SimpleNamespace(LearnerConfig=_Config, train=_train),
    )
    monkeypatch.setattr(factory, "REMOTE_ROOT", _REPO)

    with pytest.raises(TypeError, match="internal learner failure"):
        factory.ppo_learner(
            {
                "run_base": str(tmp_path),
                "run_name": "exactly-once",
                "init_checkpoint": "legacy.pt",
                "architecture": "entity_graph",
                "device": "cpu",
            }
        )

    assert calls == [
        {
            "volume_reload_fn": factory.volume.reload,
            "volume_commit_fn": factory.volume.commit,
        }
    ]


def test_legacy_payload_signature_and_behavior_remain_available(
    factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inspect

    kwargs = _actor_payload_kwargs(factory, Path("missing.pt"), Path("unused.json"))
    kwargs.pop("run_manifest")
    payload = factory._payloads(**kwargs)[0]

    assert "run_manifest_json" not in payload
    assert "run_manifest_sha256" not in payload
    assert payload["max_decisions"] == 1_000
    assert "run_manifest" not in inspect.signature(factory.smoke).parameters
    assert "run_manifest" not in inspect.signature(factory.launch_ppo_actors).parameters
    assert "run_manifest" not in inspect.signature(factory.launch_learner).parameters
    assert "run_manifest" not in inspect.signature(factory.run_learner_blocking).parameters
    actor_manifest_parameters = inspect.signature(
        factory.launch_ppo_actors_from_manifest
    ).parameters
    learner_manifest_parameters = inspect.signature(
        factory.launch_learner_from_manifest
    ).parameters
    assert "run_manifest" in actor_manifest_parameters
    assert "run_manifest" in learner_manifest_parameters
    assert not {
        "seed",
        "gamma",
        "gae_lambda",
        "max_decisions",
        "games_per_shard",
        "opponents",
        "quantize_rollout",
    } & set(actor_manifest_parameters)
    assert not {
        "lr",
        "max_steps",
        "minibatch_size",
        "behavior_temperature",
        "gamma",
    } & set(learner_manifest_parameters)

    trained: dict[str, object] = {}

    class _LegacyConfig:
        __dataclass_fields__ = {
            key: None
            for key in (
                "run_base",
                "run_name",
                "init_checkpoint",
                "architecture",
                "device",
                "lr",
            )
        }

        def __init__(self, **values) -> None:
            self.__dict__.update(values)

    monkeypatch.setitem(
        sys.modules,
        "ppo_distributed_learner",
        SimpleNamespace(
            LearnerConfig=_LegacyConfig,
            resolve_config=lambda _argv: pytest.fail(
                "legacy learner invoked manifest resolver"
            ),
            train=lambda config, **_kwargs: trained.update(config=config),
        ),
    )
    monkeypatch.setattr(factory, "REMOTE_ROOT", _REPO)
    factory.ppo_learner(
        {
            "run_base": str(tmp_path),
            "run_name": "legacy-learner",
            "init_checkpoint": "legacy.pt",
            "architecture": "entity_graph",
            "device": "cpu",
            "lr": 0.0003,
        }
    )
    assert trained["config"].lr == 0.0003
