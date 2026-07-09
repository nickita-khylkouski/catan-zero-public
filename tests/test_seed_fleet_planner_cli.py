"""CLI tests for tools/seed_fleet_planner.py (task #77 part c: a real,
callable disjointness gate any fleet launcher can invoke before firing,
not just an importable library function)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TOOL = str(Path(__file__).resolve().parent.parent / "tools" / "seed_fleet_planner.py")


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, TOOL, *args], capture_output=True, text=True
    )


class TestPlanSubcommand:
    def test_plan_writes_disjoint_table(self, tmp_path):
        out = tmp_path / "seeds.json"
        result = run(
            "plan",
            "--worker-ids", "b200_gpu0,a100a_gpu0,a100a_gpu1,a100b_gpu0,a100b_gpu1",
            "--games-per-worker", "1000",
            "--base", "9000001",
            "--block-size", "2000",
            "--out", str(out),
        )
        assert result.returncode == 0, result.stderr
        table = json.loads(out.read_text())
        assert len(table) == 5
        assert table["b200_gpu0"] == 9_000_001
        assert table["a100a_gpu0"] == 9_002_001


class TestVerifySubcommand:
    def test_verify_passes_on_disjoint_table(self, tmp_path):
        seeds_file = tmp_path / "seeds.json"
        seeds_file.write_text(json.dumps({"w0": 1000, "w1": 5000}))
        result = run("verify", "--seeds-json", str(seeds_file), "--games-per-worker", "1000")
        assert result.returncode == 0, result.stderr

    def test_verify_fails_loudly_on_colliding_table(self, tmp_path):
        seeds_file = tmp_path / "seeds.json"
        # The actual real-world bug: two workers assigned the identical
        # base-seed (314000 collision from the v3b_base confirmation arm).
        seeds_file.write_text(json.dumps({"a100a_gpu4": 314000, "a100b_gpu0": 314000}))
        result = run("verify", "--seeds-json", str(seeds_file), "--games-per-worker", "16")
        assert result.returncode != 0
        assert "collision" in result.stderr.lower() or "collision" in result.stdout.lower()

    def test_verify_fails_on_staged_gen1_scheme(self, tmp_path):
        seeds_file = tmp_path / "seeds.json"
        table = {f"a100a_gpu{i}": 9_100_001 + i * 100_000 for i in range(8)}
        table.update({f"a100b_gpu{i}": 9_200_001 + i * 100_000 for i in range(8)})
        seeds_file.write_text(json.dumps(table))
        result = run("verify", "--seeds-json", str(seeds_file), "--games-per-worker", "100000")
        assert result.returncode != 0
