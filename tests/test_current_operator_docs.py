from __future__ import annotations

import json
from pathlib import Path

from tools import a1_one_dose_train as one_dose
from tools import a1_pre_wave_contract as pre_wave


ROOT = Path(__file__).resolve().parents[1]


def _section(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def test_current_operator_docs_match_executable_a1_v5_contract() -> None:
    handoff = (ROOT / "RL_AGENT_HANDOFF.md").read_text(encoding="utf-8")
    fleet_doc = (ROOT / "FLEET.md").read_text(encoding="utf-8")
    pre_wave_doc = (
        ROOT / "docs/operations/A1_V5_64GPU_PRE_WAVE.md"
    ).read_text(encoding="utf-8")
    eval_doc = (ROOT / "docs/A1_H100_DISTRIBUTED_EVAL.md").read_text(
        encoding="utf-8"
    )
    template = json.loads(
        (ROOT / "configs/experiments/a1_pre_wave_contract.template.json").read_text(
            encoding="utf-8"
        )
    )
    fleet = json.loads(pre_wave.CURRENT_FLEET_MANIFEST.read_text(encoding="utf-8"))

    gpu_count = sum(int(host["gpu_count"]) for host in fleet["hosts"])
    host_count = len(fleet["hosts"])
    job_count = gpu_count * len(pre_wave.EXPECTED_GAMES)
    c_scale = float(template["science"]["search"]["c_scale"])
    topology = one_dose.TRAINING_TOPOLOGIES[one_dose.B200_8GPU_DDP_TOPOLOGY]

    assert gpu_count == pre_wave.CURRENT_WORKER_COUNT == 64
    assert host_count == 12
    assert job_count == 192
    assert c_scale == 0.10
    assert topology == {
        "world_size": 8,
        "local_batch_size": 512,
        "grad_accum_steps": 1,
        "global_batch_size": 4096,
    }

    state = _section(handoff, "## 1. Current handoff state", "## 2.")
    recipe = _section(handoff, "## 6. Production recipe", "## 7.")
    training = _section(handoff, "## 15. Train the canonical 35M control", "## 16.")
    fleet_launch = _section(
        fleet_doc,
        "## 7. Launch / stop / status",
        "### Historical n128 EvalServer throughput lock",
    )

    assert f"{gpu_count} GPUs" in state
    assert f"{job_count}: three category jobs" in state
    assert f"all {host_count} H100 hosts" in handoff
    assert "c-scale .10 on all three source categories" in recipe
    assert "| first 8 | 150 | 29 | 10 |" in pre_wave_doc
    assert "| next 16 | 150 | 28 | 10 |" in pre_wave_doc
    assert "| final 40 | 150 | 28 | 9 |" in pre_wave_doc
    assert "| global | 9,600 | 1,800 | 600 |" in pre_wave_doc
    assert "selected quota plus fixed 5/2/1 reserve" in recipe
    assert "exactly eight local B200 GPUs" in state
    assert "--topology b200-8gpu-ddp" in training
    assert "--nproc_per_node=8" in training
    assert "local batch 512" in training
    assert "global batch 4096" in training
    assert "repo_artifacts_sha256" in handoff
    assert f"exact {job_count}" in fleet_launch
    assert f"{gpu_count}-lane launch" in fleet_launch
    assert "deployed `c_scale=0.10` on every" in fleet_launch
    assert "full 64-H100 fleet" in eval_doc
    assert "both use 0.10" in eval_doc

    for stale in (
        "40 GPUs:",
        "120: three category jobs",
        "one selected B200",
        "c-scale .03",
        "240/45/15 selected games per GPU",
    ):
        assert stale not in state + recipe + training + fleet_launch


def test_current_eval_example_enumerates_the_full_authoritative_fleet() -> None:
    example = json.loads(
        (ROOT / "configs/a1_h100_eval_fleet.example.json").read_text(
            encoding="utf-8"
        )
    )
    authority = json.loads(
        pre_wave.CURRENT_FLEET_MANIFEST.read_text(encoding="utf-8")
    )

    expected = {
        host["alias"]: int(host["gpu_count"]) for host in authority["hosts"]
    }
    actual = {host["alias"]: int(host["gpu_count"]) for host in example["hosts"]}
    assert actual == expected
