from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from catan_zero.rl import ppo_distributed as dist
from catan_zero.rl.ppo_run_manifest import load_manifest
from tools import ppo_distributed_learner as learner


TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "selfplay"
    / "ppo_2p_no_trade_v2.json"
)


def _manifest_payload() -> dict:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def _write_bound_manifest(
    tmp_path: Path,
    initializer: Path,
    *,
    payload: dict | None = None,
) -> Path:
    value = copy.deepcopy(payload if payload is not None else _manifest_payload())
    value["status"] = "bound"
    value["spec"]["identity"]["initializer_sha256"] = (
        f"sha256:{dist.checkpoint_sha256(initializer)}"
    )
    path = tmp_path / "run-manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _resolve_manifest(tmp_path: Path, *, payload: dict | None = None):
    initializer = tmp_path / "initializer.pt"
    initializer.write_bytes(b"canonical initializer bytes")
    manifest_path = _write_bound_manifest(
        tmp_path,
        initializer,
        payload=payload,
    )
    config, args = learner.resolve_config(
        [
            "--run-manifest",
            str(manifest_path),
            "--init-checkpoint",
            str(initializer),
        ]
    )
    return config, args, manifest_path, initializer


def test_bound_manifest_maps_all_learner_science_fields_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _manifest_payload()
    actor = payload["spec"]["actor"]
    actor.update(
        gamma=0.9,
        gae_lambda=0.8,
        action_temperature=0.75,
    )
    learner_spec = payload["spec"]["learner"]
    learner_spec.update(
        shards_per_step=7,
        max_staleness=3,
        max_steps=123,
        resume=False,
        lr=0.0007,
        trunk_lr_mult=0.08,
        clip_ratio=0.2,
        value_coef=0.7,
        value_clip_range=0.15,
        entropy_coef=0.02,
        ppo_epochs=3,
        minibatch_size=4096,
        target_kl=0.02,
        top_advantage_fraction=0.75,
        min_advantage_samples=9,
        advantage_normalization="per_opponent",
        kl_to_bc_init=0.9,
        kl_to_bc_final=0.2,
        kl_to_bc_anneal_steps=321,
        use_vtrace=True,
        vtrace_clip_rho=0.8,
        vtrace_clip_pg_rho=0.7,
        vtrace_use_current_values=False,
        vtrace_forward_chunk=2048,
    )
    learner_spec["advantage_group_weights"] = [
        "catanatron_ab4=1.5",
        "random=0.5",
    ]
    payload["spec"]["checkpoint"].update(
        every_steps=3,
        keep_last=9,
        milestone_every=99,
    )
    payload["spec"]["evaluation"].update(
        dev_games=37,
        opponents=["random", "heuristic"],
        workers=3,
        max_decisions=777,
        device="cuda:2",
        timeout_secs=42.5,
    )
    payload["spec"]["league"].update(
        snapshot_interval=17,
        promote_winrate=0.65,
    )
    # Mapping is tested independently of the narrower canonical W7 science
    # validator; other tests exercise a real production-valid manifest.
    monkeypatch.setattr(learner, "_validate_w7_config", lambda _config: None)
    config, args, manifest_path, _initializer = _resolve_manifest(
        tmp_path,
        payload=payload,
    )
    manifest = load_manifest(manifest_path)
    spec = manifest.spec

    expected = {
        "architecture": spec.identity.architecture,
        "gamma": spec.actor.gamma,
        "gae_lambda": spec.actor.gae_lambda,
        "behavior_temperature": spec.actor.action_temperature,
        "shards_per_step": spec.learner.shards_per_step,
        "max_staleness": spec.learner.max_staleness,
        "max_steps": spec.learner.max_steps,
        "resume": spec.learner.resume,
        "lr": spec.learner.lr,
        "trunk_lr_mult": spec.learner.trunk_lr_mult,
        "clip_ratio": spec.learner.clip_ratio,
        "value_coef": spec.learner.value_coef,
        "value_clip_range": spec.learner.value_clip_range,
        "entropy_coef": spec.learner.entropy_coef,
        "ppo_epochs": spec.learner.ppo_epochs,
        "minibatch_size": spec.learner.minibatch_size,
        "target_kl": spec.learner.target_kl,
        "top_advantage_fraction": spec.learner.top_advantage_fraction,
        "min_advantage_samples": spec.learner.min_advantage_samples,
        "advantage_normalization": spec.learner.advantage_normalization,
        "advantage_group_weights": ",".join(
            spec.learner.advantage_group_weights
        ),
        "kl_to_bc_init": spec.learner.kl_to_bc_init,
        "kl_to_bc_final": spec.learner.kl_to_bc_final,
        "kl_to_bc_anneal_steps": spec.learner.kl_to_bc_anneal_steps,
        "use_vtrace": spec.learner.use_vtrace,
        "vtrace_clip_rho": spec.learner.vtrace_clip_rho,
        "vtrace_clip_pg_rho": spec.learner.vtrace_clip_pg_rho,
        "vtrace_use_current_values": spec.learner.vtrace_use_current_values,
        "vtrace_forward_chunk": spec.learner.vtrace_forward_chunk,
        "checkpoint_every": spec.checkpoint.every_steps,
        "keep_last_checkpoints": spec.checkpoint.keep_last,
        "checkpoint_milestone_every": spec.checkpoint.milestone_every,
        "eval_games": spec.evaluation.dev_games,
        "eval_tracks": ",".join(spec.evaluation.tracks),
        "eval_opponents": ",".join(spec.evaluation.opponents),
        "eval_workers": spec.evaluation.workers,
        "eval_max_decisions": spec.evaluation.max_decisions,
        "eval_timeout_secs": spec.evaluation.timeout_secs,
        "eval_device": spec.evaluation.device,
        "league_snapshot_interval": spec.league.snapshot_interval,
        "league_promote_winrate": spec.league.promote_winrate,
    }

    assert {name: getattr(config, name) for name in expected} == expected
    assert config.run_manifest_sha256 == manifest.sha256()
    assert args.run_manifest == str(manifest_path)


def test_manifest_rejects_initializer_bytes_that_do_not_match_identity(
    tmp_path: Path,
) -> None:
    initializer = tmp_path / "initializer.pt"
    initializer.write_bytes(b"expected bytes")
    manifest_path = _write_bound_manifest(tmp_path, initializer)
    initializer.write_bytes(b"different bytes")

    with pytest.raises(SystemExit):
        learner.resolve_config(
            [
                "--run-manifest",
                str(manifest_path),
                "--init-checkpoint",
                str(initializer),
            ]
        )


def test_manifest_rejects_unbound_template(tmp_path: Path) -> None:
    initializer = tmp_path / "initializer.pt"
    initializer.write_bytes(b"initializer")
    manifest_path = tmp_path / "template.json"
    manifest_path.write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(SystemExit):
        learner.resolve_config(
            [
                "--run-manifest",
                str(manifest_path),
                "--init-checkpoint",
                str(initializer),
            ]
        )


@pytest.mark.parametrize(
    "conflict",
    [
        ["--config", "legacy.json"],
        ["--lr", "0.0002"],
        ["--max-steps", "10"],
        ["--architecture", "entity_graph"],
        ["--no-resume"],
        ["--eval-device", "cpu"],
    ],
)
def test_manifest_rejects_explicit_legacy_science_flags(
    tmp_path: Path,
    conflict: list[str],
) -> None:
    initializer = tmp_path / "initializer.pt"
    initializer.write_bytes(b"initializer")
    manifest_path = _write_bound_manifest(tmp_path, initializer)

    with pytest.raises(SystemExit):
        learner.resolve_config(
            [
                "--run-manifest",
                str(manifest_path),
                "--init-checkpoint",
                str(initializer),
                *conflict,
            ]
        )


def test_manifest_allows_only_explicit_runtime_bindings(tmp_path: Path) -> None:
    initializer = tmp_path / "initializer.pt"
    initializer.write_bytes(b"initializer")
    manifest_path = _write_bound_manifest(tmp_path, initializer)

    config, _args = learner.resolve_config(
        [
            "--run-manifest",
            str(manifest_path),
            "--run-base",
            str(tmp_path / "runs"),
            "--run-name",
            "production-v2",
            "--init-checkpoint",
            str(initializer),
            "--device",
            "cuda:7",
            "--poll-secs",
            "1.25",
            "--stable-secs",
            "2.5",
        ]
    )

    assert config.run_base == str(tmp_path / "runs")
    assert config.run_name == "production-v2"
    assert config.init_checkpoint == str(initializer)
    assert config.device == "cuda:7"
    assert config.poll_secs == 1.25
    assert config.stable_secs == 2.5
    assert config.max_steps == load_manifest(manifest_path).spec.learner.max_steps


def test_legacy_resolution_is_unchanged_without_run_manifest(
    tmp_path: Path,
) -> None:
    initializer = tmp_path / "initializer.pt"
    initializer.write_bytes(b"legacy initializer")

    config, args = learner.resolve_config(
        [
            "--init-checkpoint",
            str(initializer),
            "--lr",
            "0.0003",
            "--max-steps",
            "12",
            "--poll-secs",
            "0.75",
        ]
    )

    assert args.run_manifest is None
    assert config.run_manifest_sha256 is None
    assert config.lr == 0.0003
    assert config.max_steps == 12
    assert config.poll_secs == 0.75
