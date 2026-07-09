from __future__ import annotations

from tools.ssh_gpu_fleet_controller import (
    active_cpu_count,
    active_cuda_count,
    choose_safe_gpu,
    DEFAULT_CAPACITY,
    plan_refill,
)


def _process(seed: int, checkpoint: str, device: str) -> dict[str, str]:
    return {
        "seed": str(seed),
        "checkpoint": f"runs/self_play/{checkpoint}",
        "device": device,
    }


def _gpu_row(index: int, used: int) -> str:
    return f"{index}, GPU, {used} MiB, 0 %"


def test_refill_plan_noops_when_gpu_fleet_is_full() -> None:
    v100 = {
        "label": "gpu-v100",
        "target": "ubuntu@v100",
        "ok": True,
        "processes": [
            *[
                _process(20000 + idx, f"s2000{idx}_live_v100g{idx}.pt", "cuda")
                for idx in range(8)
            ],
            *[
                _process(20300 + idx, f"s203{idx:02d}_gpu_cpu_selfplay_v100cpu{idx}.pt", "cpu")
                for idx in range(12)
            ],
        ],
        "gpu": [_gpu_row(idx, 1000) for idx in range(8)],
        "files": [],
        "logs": [],
    }
    h100 = {
        "label": "gpu-h100",
        "target": "ubuntu@h100",
        "ok": True,
        "processes": [
            *[
                _process(20150 + idx, f"s2015{idx}_live_h100r{idx}.pt", "cuda")
                for idx in range(4)
            ],
            *[
                _process(20350 + idx, f"s2035{idx}_gpu_cpu_selfplay_h100cpu{idx}.pt", "cpu")
                for idx in range(4)
            ],
        ],
        "gpu": [_gpu_row(0, 78_000)],
        "files": [],
        "logs": [],
    }

    plan = plan_refill(
        {"hosts": [v100, h100], "running_train_processes": 28},
        remote_repo="~/catan-zero-gpu",
        champion="runs/self_play/champions/current_best_s9752_iter0002.pt",
        recipe="auto",
        seed_floor=20400,
        max_launches=2,
        allow_cuda=True,
        allow_cpu=True,
    )

    assert plan["planned_count"] == 0
    assert plan["skipped"]["no_safe_gpu"][0]["host"] == "gpu-v100"
    assert plan["skipped"]["cuda_full"][0]["host"] == "gpu-h100"
    assert plan["skipped"]["cpu_full"] == [
        {"host": "gpu-v100", "active_cpu": 12, "target": 12},
        {"host": "gpu-h100", "active_cpu": 4, "target": 4},
    ]


def test_refill_plan_fills_remote_cpu_slot_without_cuda_env() -> None:
    host = {
        "label": "gpu-v100",
        "target": "ubuntu@v100",
        "ok": True,
        "processes": [
            *[
                _process(20000 + idx, f"s2000{idx}_live_v100g{idx}.pt", "cuda")
                for idx in range(8)
            ],
            *[
                _process(20300 + idx, f"s203{idx:02d}_gpu_cpu_selfplay_v100cpu{idx}.pt", "cpu")
                for idx in range(11)
            ],
        ],
        "gpu": [_gpu_row(idx, 1000) for idx in range(8)],
        "files": [],
        "logs": [],
    }

    plan = plan_refill(
        {"hosts": [host], "running_train_processes": 19},
        remote_repo="~/catan-zero-gpu",
        champion="runs/self_play/champions/current_best_s9752_iter0002.pt",
        recipe="auto",
        seed_floor=20400,
        max_launches=1,
        allow_cuda=True,
        allow_cpu=True,
    )

    assert plan["planned_count"] == 1
    row = plan["planned"][0]
    assert row["device"] == "cpu"
    assert row["recipe"] == "selfplay_vrpo_guard"
    assert '"--device", "cpu"' in row["command"]
    assert "CUDA_VISIBLE_DEVICES" not in row["command"]


def test_large_graph_distill_recipe_is_h100_cuda_only_bootstrap() -> None:
    host = {
        "label": "gpu-h100",
        "target": "ubuntu@h100",
        "ok": True,
        "processes": [
            _process(20150 + idx, f"s2015{idx}_live_h100r{idx}.pt", "cuda")
            for idx in range(3)
        ]
        + [
            _process(20350 + idx, f"s2035{idx}_gpu_cpu_selfplay_h100cpu{idx}.pt", "cpu")
            for idx in range(4)
        ],
        "gpu": [_gpu_row(0, 60_000)],
        "files": [],
        "logs": [],
    }

    assert active_cuda_count(host) == 3
    assert active_cpu_count(host) == 4
    assert choose_safe_gpu(host, DEFAULT_CAPACITY["gpu-h100"]) == 0

    plan = plan_refill(
        {"hosts": [host], "running_train_processes": 7},
        remote_repo="~/catan-zero-gpu",
        champion="runs/self_play/champions/current_best_s9752_iter0002.pt",
        recipe="large_graph_distill",
        seed_floor=20400,
        max_launches=1,
        allow_cuda=True,
        allow_cpu=True,
    )

    row = plan["planned"][0]
    command = row["command"]
    assert row["recipe"] == "large_graph_distill"
    assert row["gpu_index"] == 0
    assert '"CUDA_VISIBLE_DEVICES": "0"' in command
    assert '"--architecture", "graph_history_candidate"' in command
    assert '"--hidden-size", "384"' in command
    assert "--init-checkpoint" not in command
