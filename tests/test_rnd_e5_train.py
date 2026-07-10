from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tools.rnd_e5_train as runner


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.upstream = torch.nn.Linear(3, 3)
        self.q_head = torch.nn.Sequential(
            torch.nn.Linear(3, 3),
            torch.nn.GELU(),
            torch.nn.Linear(3, 1),
        )


class _FakePolicy:
    def __init__(self, *, safe: bool = True) -> None:
        self.model = _FakeModel()
        self.trained_with_masked_hidden_info = safe

    def save(self, path, **metadata) -> None:
        torch.save({"model": self.model.state_dict(), "metadata": metadata}, path)


def _write_inputs(tmp_path: Path, *, seeds: list[int]) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "init.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "seeds.json"
    manifest.write_text(
        json.dumps({"schema_version": runner.SEED_SCHEMA_VERSION, "seeds": seeds}),
        encoding="utf-8",
    )
    return checkpoint, manifest


def _config(
    tmp_path: Path,
    *,
    iterations: int = 2,
    games_per_iteration: int = 1,
) -> runner.E5KLENTConfig:
    seeds = list(range(10, 10 + iterations * games_per_iteration))
    checkpoint, manifest = _write_inputs(tmp_path, seeds=seeds)
    return runner.E5KLENTConfig(
        arm=runner.SUPPORTED_ARM,
        init_checkpoint=str(checkpoint),
        init_checkpoint_sha256=runner.sha256_file(checkpoint),
        seed_manifest=str(manifest),
        seed_manifest_sha256=runner.sha256_file(manifest),
        seeds=tuple(seeds),
        run_dir=str(tmp_path / "run"),
        device="cpu",
        iterations=iterations,
        games_per_iteration=games_per_iteration,
        max_decisions=8,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        epochs=1,
        minibatch_size=4,
        gradient_clip_norm=1.0,
        value_loss_weight=0.25,
        entropy_coefficient=0.03,
        reverse_kl_coefficient=0.1,
        trace_horizon=8.0,
        q_loss_weight=1.0,
        q_init="zero-output",
        max_truncation_fraction=1.0,
    )


def _trajectory(seed: int):
    return SimpleNamespace(
        game_seed=seed,
        truncated=False,
        steps=(object(), object()),
    )


def _finite_update(policy, trajectories, optimizer, **_kwargs):
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().sum() for parameter in policy.model.parameters())
    loss.backward()
    optimizer.step()
    return {
        "schema_version": "catan-zero-klent-update/v1",
        "rows": sum(len(trajectory.steps) for trajectory in trajectories),
        "row_passes": sum(len(trajectory.steps) for trajectory in trajectories),
        "updates": 1,
        "epochs": 1,
        "loss": float(loss.detach()),
        "policy_loss": 0.5,
        "q_loss": 0.25,
        "value_loss": 0.125,
    }


def test_seed_manifest_is_explicit_unique_and_exact(tmp_path: Path) -> None:
    _checkpoint, manifest = _write_inputs(tmp_path, seeds=[7, 9])
    assert runner.load_seed_manifest(manifest) == (7, 9)
    manifest.write_text(
        json.dumps({"schema_version": runner.SEED_SCHEMA_VERSION, "seeds": [7, 7]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        runner.load_seed_manifest(manifest)

    config = _config(tmp_path / "exact", iterations=2)
    object.__setattr__(config, "seeds", (10,))
    with pytest.raises(ValueError, match="exactly 2"):
        config.validate()


def test_q_initialization_zeros_only_output_layer() -> None:
    policy = _FakePolicy()
    upstream_before = {
        name: value.detach().clone()
        for name, value in policy.model.q_head[:-1].state_dict().items()
    }
    runner._reset_q_output_head(policy)
    assert torch.count_nonzero(policy.model.q_head[-1].weight) == 0
    assert torch.count_nonzero(policy.model.q_head[-1].bias) == 0
    for name, before in upstream_before.items():
        assert torch.equal(policy.model.q_head[:-1].state_dict()[name], before)


def test_direct_loop_publishes_atomic_outputs_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        runner.EntityGraphPolicy,
        "load",
        lambda *_args, **_kwargs: _FakePolicy(),
    )
    monkeypatch.setattr(
        runner,
        "collect_trajectory",
        lambda _policy, *, seed, **_kwargs: _trajectory(seed),
    )
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return _finite_update(*args, **kwargs)

    monkeypatch.setattr(runner, "update_entity_policy", fail_second)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.train_klent_direct(config)

    run_dir = Path(config.run_dir)
    partial = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert len(partial["iterations"]) == 1
    assert partial["iterations"][0]["seeds"] == [10]
    first = run_dir / "checkpoints" / "iter_0001.pt"
    assert first.is_file()
    assert runner.optimizer_sidecar_path(first).is_file()

    monkeypatch.setattr(runner, "update_entity_policy", _finite_update)
    report = runner.train_klent_direct(config, resume=True)
    assert report["status"] == "complete"
    assert [row["seeds"] for row in report["iterations"]] == [[10], [11]]
    assert report["total_games"] == 2
    final = run_dir / "final.pt"
    assert final.is_file()
    assert runner.optimizer_sidecar_path(final).is_file()
    assert report["final_checkpoint"]["sha256"] == runner.sha256_file(final)

    (run_dir / "checkpoints" / "iter_0002.pt").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="hash"):
        runner.train_klent_direct(config, resume=True)


def test_direct_loop_refuses_checkpoint_without_public_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, iterations=1)
    monkeypatch.setattr(
        runner.EntityGraphPolicy,
        "load",
        lambda *_args, **_kwargs: _FakePolicy(safe=False),
    )
    with pytest.raises(RuntimeError, match="masked hidden"):
        runner.train_klent_direct(config)
    assert not (Path(config.run_dir) / "report.json").exists()


def test_resume_before_first_iteration_restarts_from_initial_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, iterations=1)
    loaded_paths: list[Path] = []

    def load_fake(path, **_kwargs):
        loaded_paths.append(Path(path))
        return _FakePolicy()

    monkeypatch.setattr(runner.EntityGraphPolicy, "load", load_fake)
    calls = 0

    def interrupt_once(_policy, *, seed, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated pre-iteration interruption")
        return _trajectory(seed)

    monkeypatch.setattr(runner, "collect_trajectory", interrupt_once)
    monkeypatch.setattr(runner, "update_entity_policy", _finite_update)
    with pytest.raises(RuntimeError, match="pre-iteration interruption"):
        runner.train_klent_direct(config)
    assert (
        json.loads((Path(config.run_dir) / "report.json").read_text())["iterations"]
        == []
    )

    report = runner.train_klent_direct(config, resume=True)
    assert report["status"] == "complete"
    assert loaded_paths == [Path(config.init_checkpoint), Path(config.init_checkpoint)]


def test_non_direct_arms_remain_blocked(tmp_path: Path) -> None:
    config = _config(tmp_path, iterations=1)
    object.__setattr__(config, "arm", "search-distillation")
    with pytest.raises(ValueError, match="remain blocked"):
        config.validate()


def test_cli_default_refuses_majority_truncated_iterations() -> None:
    args = runner.build_parser().parse_args(
        [
            "--init-checkpoint",
            "init.pt",
            "--seed-manifest",
            "seeds.json",
            "--run-dir",
            "run",
        ]
    )
    assert args.max_truncation_fraction == 0.5


def test_resume_repairs_interrupted_finalization_and_validates_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, iterations=1)
    monkeypatch.setattr(
        runner.EntityGraphPolicy, "load", lambda *_args, **_kwargs: _FakePolicy()
    )
    monkeypatch.setattr(
        runner,
        "collect_trajectory",
        lambda _policy, *, seed, **_kwargs: _trajectory(seed),
    )
    monkeypatch.setattr(runner, "update_entity_policy", _finite_update)
    real_copy = runner._copy_atomic
    failed = False

    def fail_first_final_copy(source, destination):
        nonlocal failed
        if Path(destination).name == "final.pt" and not failed:
            failed = True
            raise RuntimeError("simulated finalization interruption")
        return real_copy(source, destination)

    monkeypatch.setattr(runner, "_copy_atomic", fail_first_final_copy)
    with pytest.raises(RuntimeError, match="finalization interruption"):
        runner.train_klent_direct(config)
    partial = json.loads((Path(config.run_dir) / "report.json").read_text())
    assert partial["status"] == "running"
    assert len(partial["iterations"]) == 1

    monkeypatch.setattr(runner, "_copy_atomic", real_copy)
    report = runner.train_klent_direct(config, resume=True)
    assert report["status"] == "complete"
    final = Path(config.run_dir) / "final.pt"
    final.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="final checkpoint hash"):
        runner.train_klent_direct(config, resume=True)


def test_iteration_rng_makes_interrupted_resume_match_uninterrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def load_fake(path, **_kwargs):
        policy = _FakePolicy()
        with torch.no_grad():
            for parameter in policy.model.parameters():
                parameter.fill_(0.1)
        if Path(path).name.startswith("iter_"):
            blob = torch.load(path, map_location="cpu", weights_only=False)
            policy.model.load_state_dict(blob["model"])
        return policy

    monkeypatch.setattr(runner.EntityGraphPolicy, "load", load_fake)
    monkeypatch.setattr(
        runner,
        "collect_trajectory",
        lambda _policy, *, seed, **_kwargs: _trajectory(seed),
    )

    def stochastic_update(policy, _trajectories, optimizer, **_kwargs):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(
            (parameter * torch.rand_like(parameter)).sum()
            for parameter in policy.model.parameters()
        )
        loss.backward()
        optimizer.step()
        return {
            "updates": 1,
            "loss": float(loss.detach()),
            "policy_loss": 0.5,
            "q_loss": 0.25,
            "value_loss": 0.125,
        }

    uninterrupted = _config(tmp_path / "uninterrupted")
    monkeypatch.setattr(runner, "update_entity_policy", stochastic_update)
    runner.train_klent_direct(uninterrupted)

    interrupted = _config(tmp_path / "interrupted")
    calls = 0

    def interrupt_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("stop")
        return stochastic_update(*args, **kwargs)

    monkeypatch.setattr(runner, "update_entity_policy", interrupt_second)
    with pytest.raises(RuntimeError, match="stop"):
        runner.train_klent_direct(interrupted)
    monkeypatch.setattr(runner, "update_entity_policy", stochastic_update)
    runner.train_klent_direct(interrupted, resume=True)

    left = torch.load(
        Path(uninterrupted.run_dir) / "final.pt", map_location="cpu", weights_only=False
    )["model"]
    right = torch.load(
        Path(interrupted.run_dir) / "final.pt", map_location="cpu", weights_only=False
    )["model"]
    assert left.keys() == right.keys()
    assert all(torch.equal(left[name], right[name]) for name in left)


def test_nonfinite_model_state_is_refused_before_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, iterations=1)
    monkeypatch.setattr(
        runner.EntityGraphPolicy, "load", lambda *_args, **_kwargs: _FakePolicy()
    )
    monkeypatch.setattr(
        runner,
        "collect_trajectory",
        lambda _policy, *, seed, **_kwargs: _trajectory(seed),
    )

    def poison(policy, trajectories, optimizer, **kwargs):
        result = _finite_update(policy, trajectories, optimizer, **kwargs)
        with torch.no_grad():
            next(policy.model.parameters()).flatten()[0] = float("nan")
        return result

    monkeypatch.setattr(runner, "update_entity_policy", poison)
    with pytest.raises(RuntimeError, match="model state is non-finite"):
        runner.train_klent_direct(config)
    assert not (Path(config.run_dir) / "checkpoints" / "iter_0001.pt").exists()


def test_truncation_fraction_gate_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, iterations=1)
    object.__setattr__(config, "max_truncation_fraction", 0.0)
    monkeypatch.setattr(
        runner.EntityGraphPolicy, "load", lambda *_args, **_kwargs: _FakePolicy()
    )
    monkeypatch.setattr(
        runner,
        "collect_trajectory",
        lambda _policy, *, seed, **_kwargs: SimpleNamespace(
            game_seed=seed, truncated=True, steps=(object(),)
        ),
    )
    monkeypatch.setattr(runner, "update_entity_policy", _finite_update)
    with pytest.raises(RuntimeError, match="truncation fraction"):
        runner.train_klent_direct(config)
    assert not (Path(config.run_dir) / "checkpoints" / "iter_0001.pt").exists()
    report = json.loads((Path(config.run_dir) / "report.json").read_text())
    assert report["status"] == "refused"
    assert report["refusal"]["reason"] == "insufficient_terminal_outcome_signal"
    assert report["refusal"]["truncation_fraction"] == 1.0
