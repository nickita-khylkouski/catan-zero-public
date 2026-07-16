from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from catan_zero.rl import ppo_distributed as dist
from catan_zero.rl.league import League
from tools import ppo_distributed_learner as learner


class _TinyPolicy:
    def __init__(self, model) -> None:
        self.model = model

    def save(self, path: str) -> None:
        import torch

        torch.save({"model": self.model.state_dict()}, path)


def _adam_with_state():
    import torch

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0125)
    loss = model(torch.ones(4, 3)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def _assert_optimizer_states_equal(expected: dict, actual: dict) -> None:
    import torch

    assert expected["param_groups"] == actual["param_groups"]
    assert expected["state"].keys() == actual["state"].keys()
    for parameter_id, expected_state in expected["state"].items():
        actual_state = actual["state"][parameter_id]
        assert expected_state.keys() == actual_state.keys()
        for name, expected_value in expected_state.items():
            actual_value = actual_state[name]
            if isinstance(expected_value, torch.Tensor):
                torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
            else:
                assert actual_value == expected_value


def test_checkpoint_set_restores_exact_optimizer_state(tmp_path: Path) -> None:
    import torch

    model, optimizer = _adam_with_state()
    expected_optimizer_state = optimizer.state_dict()
    checkpoint, optimizer_path = learner._save_checkpoint_set(  # noqa: SLF001
        policy=_TinyPolicy(model),
        optimizer=optimizer,
        root=tmp_path,
        step=7,
    )

    assert checkpoint.is_file()
    assert optimizer_path.is_file()
    fresh_model = torch.nn.Linear(3, 2)
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=9.0)
    payload = learner._restore_optimizer_checkpoint(  # noqa: SLF001
        optimizer=fresh_optimizer,
        checkpoint_path=checkpoint,
        step=7,
        map_location="cpu",
    )

    assert payload["schema"] == learner.LEARNER_CHECKPOINT_SCHEMA
    assert payload["checkpoint_sha256"] == dist.checkpoint_sha256(checkpoint)
    _assert_optimizer_states_equal(
        expected_optimizer_state,
        fresh_optimizer.state_dict(),
    )


def test_checkpoint_restores_rng_and_finalizes_consumed_shard_frontier(
    tmp_path: Path,
) -> None:
    import torch

    root = tmp_path / "run"
    dist.ensure_run_dirs(root)
    shard = dist.write_trajectory_shard(
        root,
        "worker-A",
        12,
        [{"trajectory": 1}],
        policy_version=4,
    )
    model, optimizer = _adam_with_state()
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(404)
    checkpoint, _optimizer_path = learner._save_checkpoint_set(  # noqa: SLF001
        policy=_TinyPolicy(model),
        optimizer=optimizer,
        root=root,
        step=4,
        consumed_shards=[shard],
    )
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)
    expected_cuda = torch.rand(3, device="cuda") if torch.cuda.is_available() else None

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(999)
    fresh_model = torch.nn.Linear(3, 2)
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters())
    payload = learner._restore_optimizer_checkpoint(  # noqa: SLF001
        optimizer=fresh_optimizer,
        checkpoint_path=checkpoint,
        step=4,
        map_location="cpu",
    )
    learner._finalize_consumed_frontier(root, payload["consumed_shards"])  # noqa: SLF001
    learner._restore_rng_state(payload["rng_state"])  # noqa: SLF001

    assert not shard.exists()
    assert list(dist.iter_unconsumed_shards(root)) == []
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    torch.testing.assert_close(torch.rand(3), expected_torch, rtol=0, atol=0)
    if expected_cuda is not None:
        torch.testing.assert_close(
            torch.rand(3, device="cuda"), expected_cuda, rtol=0, atol=0
        )


def test_resume_refuses_unsafe_consumed_shard_frontier(tmp_path: Path) -> None:
    import torch

    model, optimizer = _adam_with_state()
    checkpoint, optimizer_path = learner._save_checkpoint_set(  # noqa: SLF001
        policy=_TinyPolicy(model),
        optimizer=optimizer,
        root=tmp_path,
        step=9,
        consumed_shards=[],
    )
    payload = torch.load(optimizer_path, map_location="cpu", weights_only=False)
    payload["consumed_shards"] = ["../outside.pkl"]
    torch.save(payload, optimizer_path)
    fresh_model = torch.nn.Linear(3, 2)
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters())

    with pytest.raises(RuntimeError, match="unsafe consumed-shard frontier"):
        learner._restore_optimizer_checkpoint(  # noqa: SLF001
            optimizer=fresh_optimizer,
            checkpoint_path=checkpoint,
            step=9,
            map_location="cpu",
        )


def test_recovery_update_commits_before_publish_consume_and_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    shard = dist.trajectories_dir(tmp_path) / "worker-A" / "shard_1.pkl"
    shard.parent.mkdir(parents=True)
    shard.touch()
    checkpoint = dist.checkpoints_dir(tmp_path) / "step_1.pt"
    optimizer_path = dist.checkpoints_dir(tmp_path) / "step_1.opt.pt"
    monkeypatch.setattr(
        learner,
        "_save_checkpoint_set",
        lambda **_kwargs: (events.append("checkpoint") or (checkpoint, optimizer_path)),
    )
    monkeypatch.setattr(
        dist,
        "publish_weights",
        lambda *_args, **_kwargs: (
            events.append("publish") or SimpleNamespace(version=2)
        ),
    )
    monkeypatch.setattr(
        learner,
        "_finalize_consumed_frontier",
        lambda *_args, **_kwargs: events.append("consume"),
    )

    learner._commit_recovery_update(  # noqa: SLF001
        policy=SimpleNamespace(save=lambda _path: None),
        optimizer=object(),
        root=tmp_path,
        completed_step=1,
        shard_paths=[shard],
        volume_commit_fn=lambda: events.append("commit"),
    )
    events.append("evaluation")

    assert events == [
        "checkpoint",
        "commit",
        "publish",
        "commit",
        "consume",
        "commit",
        "evaluation",
    ]


def test_checkpoint_commit_failure_prevents_publish_and_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = dist.trajectories_dir(tmp_path) / "worker-A" / "shard_1.pkl"
    shard.parent.mkdir(parents=True)
    shard.touch()
    checkpoint = dist.checkpoints_dir(tmp_path) / "step_1.pt"
    optimizer_path = dist.checkpoints_dir(tmp_path) / "step_1.opt.pt"
    monkeypatch.setattr(
        learner,
        "_save_checkpoint_set",
        lambda **_kwargs: (checkpoint, optimizer_path),
    )
    monkeypatch.setattr(
        dist,
        "publish_weights",
        lambda *_args, **_kwargs: pytest.fail("published after failed checkpoint commit"),
    )
    monkeypatch.setattr(
        learner,
        "_finalize_consumed_frontier",
        lambda *_args, **_kwargs: pytest.fail(
            "consumed after failed checkpoint commit"
        ),
    )

    with pytest.raises(RuntimeError, match="commit failed after recovery checkpoint"):
        learner._commit_recovery_update(  # noqa: SLF001
            policy=object(),
            optimizer=object(),
            root=tmp_path,
            completed_step=1,
            shard_paths=[shard],
            volume_commit_fn=lambda: (_ for _ in ()).throw(OSError("volume down")),
        )


def test_resume_refuses_discovered_model_without_optimizer_sidecar(
    tmp_path: Path,
) -> None:
    import torch

    root = tmp_path / "run"
    dist.ensure_run_dirs(root)
    checkpoint = dist.checkpoints_dir(root) / "step_3.pt"
    torch.save({"model": "model-only"}, checkpoint)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())

    assert learner.find_resume_checkpoint(root) == (3, checkpoint)
    with pytest.raises(RuntimeError, match="has no optimizer sidecar"):
        learner._restore_optimizer_checkpoint(  # noqa: SLF001
            optimizer=optimizer,
            checkpoint_path=checkpoint,
            step=3,
            map_location="cpu",
        )


def test_resume_refuses_optimizer_sidecar_bound_to_different_model(
    tmp_path: Path,
) -> None:
    import torch

    model, optimizer = _adam_with_state()
    checkpoint, _optimizer_path = learner._save_checkpoint_set(  # noqa: SLF001
        policy=_TinyPolicy(model),
        optimizer=optimizer,
        root=tmp_path,
        step=5,
    )
    torch.save({"model": "replacement"}, checkpoint)
    fresh_model = torch.nn.Linear(3, 2)
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters())

    with pytest.raises(RuntimeError, match="binds different model bytes"):
        learner._restore_optimizer_checkpoint(  # noqa: SLF001
            optimizer=fresh_optimizer,
            checkpoint_path=checkpoint,
            step=5,
            map_location="cpu",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "unknown", "schema is missing or unsupported"),
        ("step", 88, "step does not match"),
    ],
)
def test_resume_refuses_mismatched_optimizer_sidecar_metadata(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    import torch

    model, optimizer = _adam_with_state()
    checkpoint, optimizer_path = learner._save_checkpoint_set(  # noqa: SLF001
        policy=_TinyPolicy(model),
        optimizer=optimizer,
        root=tmp_path,
        step=6,
    )
    payload = torch.load(optimizer_path, map_location="cpu", weights_only=False)
    payload[field] = value
    torch.save(payload, optimizer_path)
    fresh_model = torch.nn.Linear(3, 2)
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters())

    with pytest.raises(RuntimeError, match=message):
        learner._restore_optimizer_checkpoint(  # noqa: SLF001
            optimizer=fresh_optimizer,
            checkpoint_path=checkpoint,
            step=6,
            map_location="cpu",
        )


def test_checkpoint_refuses_same_step_overwrite(tmp_path: Path) -> None:
    model, optimizer = _adam_with_state()
    learner._save_checkpoint_set(  # noqa: SLF001
        policy=_TinyPolicy(model),
        optimizer=optimizer,
        root=tmp_path,
        step=2,
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        learner._save_checkpoint_set(  # noqa: SLF001
            policy=_TinyPolicy(model),
            optimizer=optimizer,
            root=tmp_path,
            step=2,
        )


def test_rotation_always_keeps_latest_recovery_checkpoint(tmp_path: Path) -> None:
    model, optimizer = _adam_with_state()
    for step in (1, 2):
        learner._save_checkpoint_set(  # noqa: SLF001
            policy=_TinyPolicy(model),
            optimizer=optimizer,
            root=tmp_path,
            step=step,
        )
    config = learner.LearnerConfig(
        run_base=str(tmp_path),
        run_name="rotation",
        init_checkpoint="parent.pt",
        keep_last_checkpoints=0,
        checkpoint_milestone_every=0,
    )

    learner.prune_checkpoints(tmp_path, League(), config)

    latest = dist.checkpoints_dir(tmp_path) / "step_2.pt"
    assert latest.is_file()
    assert learner._opt_path_for(latest).is_file()  # noqa: SLF001


def test_bounded_run_schedules_terminal_checkpoint_off_periodic_cadence() -> None:
    config = learner.LearnerConfig(
        run_base="runs/distributed",
        run_name="terminal-checkpoint-test",
        init_checkpoint="parent.pt",
        max_steps=3,
        checkpoint_every=50,
    )

    assert learner._checkpoint_schedule(config, completed_step=1) == (False, False)  # noqa: SLF001
    assert learner._checkpoint_schedule(config, completed_step=3) == (False, True)  # noqa: SLF001


def test_terminal_only_checkpoint_saves_pair_without_running_scoreboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, optimizer = _adam_with_state()
    policy = _TinyPolicy(model)
    league = League()
    main_id = league.add_main("current.pt").id
    config = learner.LearnerConfig(
        run_base=str(tmp_path),
        run_name="terminal-only",
        init_checkpoint="parent.pt",
    )
    monkeypatch.setattr(
        learner,
        "run_scoreboard_eval",
        lambda *_args, **_kwargs: pytest.fail(
            "terminal-only checkpoint ran evaluation"
        ),
    )

    learner._checkpoint_eval_league(  # noqa: SLF001
        policy=policy,
        optimizer=optimizer,
        league=league,
        main_id=main_id,
        baseline_ids={},
        root=tmp_path,
        step=1,
        config=config,
        run_evaluation=False,
    )

    checkpoint = dist.checkpoints_dir(tmp_path) / "step_1.pt"
    assert checkpoint.is_file()
    assert learner._opt_path_for(checkpoint).is_file()  # noqa: SLF001
