from __future__ import annotations

import copy
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
    payloads, _devices, _games = actor._build_worker_payloads(  # noqa: SLF001
        args,
        SimpleNamespace(path=str(tmp_path / "weights.pt"), version=3),
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
