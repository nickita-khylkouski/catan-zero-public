#!/usr/bin/env python3
"""Build the immutable 20+20 GPU jobset for the A1 neutral-panel rerun.

This is a frozen historical artifact, not an input to the canonical generic
fleet scheduler. Its eight-host snapshot predates both exact-56 and exact-64;
using a distinct schema prevents it from impersonating either fleet authority.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "catan-gpu-jobset-v1"
HISTORICAL_MANIFEST_SCHEMA = "a1-neutral-panel-fleet-snapshot-v1"
RUN_ID = "a1-neutral-python-panel-n128-20260710-v2"
REPO = "/home/ubuntu/catan-zero-v1"
REMOTE_ROOT = "/home/ubuntu/catan-fleet-jobs"
PYTHON = f"{REPO}/.venv/bin/python"
COMMIT = "f0179eefc927666a4c704f6a3487521765058597"
BASE_SEED = 6_199_000_000
PAIRS_PER_SHARD = 25
WORKERS_PER_GPU = 8

A1 = "/home/ubuntu/catan-zero-production/runs/learner/a1-infoset-n128-20260710-r2/candidate.pt"
A1_SHA256 = "f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4"
GEN3 = "/home/ubuntu/catan-zero/runs/bc/gen3_20260706/checkpoint.pt"
GEN3_SHA256 = "89aa133d629e747021bc725f2ad63e0563f3b76e71f0dd563f056c6de8f77ebb"

HOSTS = (
    ("c1", "192.222.54.251", 4),
    ("c2", "68.209.75.117", 4),
    ("c3", "192.222.53.18", 4),
    ("c4", "68.209.73.252", 4),
    ("c5", "68.209.74.145", 4),
    ("c6", "68.209.74.2", 4),
    ("h100-8a", "192.222.53.119", 8),
    ("h100-8b", "192.222.55.216", 8),
)


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def manifest(*, ssh_key: str) -> dict[str, Any]:
    return {
        "schema_version": HISTORICAL_MANIFEST_SCHEMA,
        "ssh_user": "ubuntu",
        "ssh_key": ssh_key,
        "strict_host_key_checking": "accept-new",
        "remote_repo": REPO,
        "remote_root": REMOTE_ROOT,
        "hosts": [
            {
                "alias": alias,
                "address": address,
                "gpu_count": count,
                "accelerator": "NVIDIA H100 80GB HBM3",
                "repo_commit": COMMIT,
            }
            for alias, address, count in HOSTS
        ],
    }


def _neutral_command(
    *, checkpoint: str, checkpoint_sha256: str, c_scale: str, job_dir: str, seed: int
) -> str:
    argv = [
        PYTHON,
        "tools/catanatron_neutral_harness_match.py",
        "--checkpoint", checkpoint,
        "--opponent", "catanatron_value",
        "--mode", "search",
        "--pairs", str(PAIRS_PER_SHARD),
        "--base-seed", str(seed),
        "--workers", str(WORKERS_PER_GPU),
        "--threads-per-worker", "1",
        "--device", "cuda",
        "--vps-to-win", "10",
        "--max-player-trade-offers-per-turn", "0",
        "--n-full", "128",
        "--c-scale", c_scale,
        "--c-visit", "50.0",
        "--rescale-noise-floor-c", "0.0",
        "--sigma-eval", "0.98",
        "--lazy-interior-chance",
        "--public-observation",
        "--information-set-search",
        "--no-belief-chance-spectra",
        "--determinization-particles", "4",
        "--determinization-min-simulations", "32",
        "--correct-rust-chance-spectra",
        "--max-depth", "80",
        "--max-decisions", "600",
        "--prior-temperature", "1.0",
        "--value-scale", "1.0",
        "--value-squash", "tanh",
        "--value-readout", "scalar",
        "--max-root-candidates", "16",
        "--max-root-candidates-wide", "54",
        "--wide-candidates-threshold", "24",
        "--symmetry-averaged-eval",
        "--symmetry-averaged-eval-threshold", "20",
        "--gate-config", "flywheel",
        "--artifact-dir", f"{job_dir}/games",
        "--resume",
        "--out", f"{job_dir}/report.json",
    ]
    import shlex

    expected = f"{checkpoint_sha256}  {checkpoint}"
    return (
        "set -euo pipefail; "
        f"test \"$(sha256sum {shlex.quote(checkpoint)})\" = {shlex.quote(expected)}; "
        "exec " + " ".join(shlex.quote(part) for part in argv)
    )


def jobset() -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    cohort = 0
    for alias, _address, count in HOSTS:
        for _local_pair in range(count // 2):
            seed = BASE_SEED + cohort * PAIRS_PER_SHARD
            for role, checkpoint, digest, c_scale in (
                ("a1", A1, A1_SHA256, "0.10"),
                ("gen3", GEN3, GEN3_SHA256, "0.03"),
            ):
                job_id = f"neutral-{cohort:02d}-{role}-{alias}"
                job_dir = f"{REMOTE_ROOT}/{RUN_ID}/{job_id}"
                jobs.append(
                    {
                        "job_id": job_id,
                        "gpus": 1,
                        "host": alias,
                        "env": {
                            "PYTHONPATH": f"{REPO}/src:{REPO}/tools",
                            "PYTHONUNBUFFERED": "1",
                        },
                        "argv": [
                            "bash",
                            "-lc",
                            _neutral_command(
                                checkpoint=checkpoint,
                                checkpoint_sha256=digest,
                                c_scale=c_scale,
                                job_dir=job_dir,
                                seed=seed,
                            ),
                        ],
                    }
                )
            cohort += 1
    if cohort != 20 or len(jobs) != 40:
        raise AssertionError("canonical fleet did not resolve 20 paired cohorts")
    return {"schema_version": SCHEMA, "run_id": RUN_ID, "jobs": jobs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ssh-key", default=str(Path.home() / ".ssh/gpu_access_ed25519"))
    args = parser.parse_args()
    _write_new(args.out_dir / "manifest.json", manifest(ssh_key=args.ssh_key))
    _write_new(args.out_dir / "jobset.json", jobset())


if __name__ == "__main__":
    main()
