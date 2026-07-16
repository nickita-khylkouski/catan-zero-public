"""Regression tests for the canonical fleet launcher's resource contract.

The launcher is shell, so these tests deliberately inspect the generated command
template.  They protect the expensive failure modes that are otherwise visible
only after a fleet launch: every selected GPU needs its own generator process,
seed claims must match the games actually emitted, and production speed/training
flags must be explicit rather than inherited from unsafe parser defaults.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools" / "fleet" / "fleet_launch.sh"
DETACHER = ROOT / "tools" / "fleet" / "launch_detached.sh"
STATUS = ROOT / "tools" / "fleet" / "fleet_status.sh"
GATE = ROOT / "scripts" / "gate.sh"
NOOP_CHECK = ROOT / "scripts" / "check_champion_noop.py"


def _source() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


@pytest.fixture
def launcher_env(tmp_path: Path) -> dict[str, str]:
    """Build a complete dry-run host locally; no network or GPU is involved."""

    home = tmp_path / "home"
    tree = home / "tree"
    tree.mkdir(parents=True)
    checkpoint = home / "champion.pt"
    checkpoint.write_bytes(b"contract-test checkpoint placeholder")
    ledger = tree / "runs" / "SEED_LEDGER.md"
    ledger.parent.mkdir()
    ledger.write_text("# contract-test ledger\n")

    fleet_conf = home / ".catan_fleet.conf"
    fleet_conf.write_text(
        "declare -A HOST=( [c1]=127.0.0.1 )\n"
        f"GPU_SSH_KEY={home / 'unused-test-key'}\n"
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text(
        "#!/usr/bin/env bash\n"
        "remote=''\n"
        'for arg in "$@"; do remote="$arg"; done\n'
        'exec bash -c "$remote"\n'
    )
    fake_ssh.chmod(0o755)
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$*\" == *\"--query-gpu=index\"* ]]; then\n"
        "  printf '0\\n1\\n2\\n3\\n4\\n5\\n6\\n7\\n'\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"$*\" == *\"--query-compute-apps=pid,process_name\"* ]]; then\n"
        "  if [ \"${FAKE_BUSY_GPU:-}\" = \"${2:-}\" ]; then\n"
        "    printf '4242, python3\\n'\n"
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n"
    )
    fake_nvidia_smi.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FLEET_CONF": str(fleet_conf),
            "TREE": str(tree),
            "CKPT": str(checkpoint),
            "LEDGER": str(ledger),
            "PY": sys.executable,
            "GEN_PY": sys.executable,
            "FLEET_LAUNCH_HEARTBEAT_WAIT_SECONDS": "0",
            "FLEET_LAUNCH_EARLY_EXIT_SECONDS": "0.05",
        }
    )
    return env


def _run(
    env: dict[str, str], role: str, *args: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), "c1", role, *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_launcher_shell_syntax_is_valid() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    subprocess.run(["bash", "-n", str(DETACHER)], check=True)


def test_launch_detached_publishes_a_live_pid_equal_to_its_sid(tmp_path: Path) -> None:
    if not sys.platform.startswith("linux") or not shutil.which("setsid"):
        pytest.skip("detached PID/SID contract requires Linux setsid")
    rundir = tmp_path / "run"
    result = subprocess.run(
        [
            "bash",
            str(DETACHER),
            str(rundir),
            str(tmp_path / "run.log"),
            "60",
            "--",
            "sleep",
            "30",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    pid = int(result.stdout.strip())
    try:
        assert int((rundir / ".pid").read_text().strip()) == pid
        assert os.getsid(pid) == pid
        os.kill(pid, 0)
    finally:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_launch_detached_rejects_an_immediately_exited_child(tmp_path: Path) -> None:
    if not sys.platform.startswith("linux") or not shutil.which("setsid"):
        pytest.skip("detached PID/SID contract requires Linux setsid")
    rundir = tmp_path / "run"
    result = subprocess.run(
        [
            "bash",
            str(DETACHER),
            str(rundir),
            str(tmp_path / "run.log"),
            "60",
            "--",
            "bash",
            "-c",
            "exit 23",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert not (rundir / ".pid").exists()


def test_launch_detached_reaps_descendants_when_session_leader_exits(
    tmp_path: Path,
) -> None:
    if not sys.platform.startswith("linux") or not shutil.which("setsid"):
        pytest.skip("detached PGID cleanup contract requires Linux setsid")
    rundir = tmp_path / "run"
    child_pid_file = tmp_path / "child.pid"
    program = (
        "import os,time; "
        "pid=os.fork(); "
        f"open({str(child_pid_file)!r},'w').write(str(pid) if pid else str(os.getpid())); "
        "time.sleep(30) if pid == 0 else time.sleep(0.02); "
        "os._exit(0)"
    )
    result = subprocess.run(
        [
            "bash",
            str(DETACHER),
            str(rundir),
            str(tmp_path / "run.log"),
            "60",
            "--",
            sys.executable,
            "-c",
            program,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    if child_pid_file.exists():
        child_pid = int(child_pid_file.read_text())
        for _ in range(50):
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(child_pid)],
                text=True,
                capture_output=True,
                check=False,
            ).stdout.strip()
            if not state or state.startswith("Z"):
                break
            time.sleep(0.01)
        assert not state or state.startswith("Z")


def _generation_runner() -> str:
    source = _source()
    return source.split("<<'GEN_RUNNER_EOF'", 1)[1].split(
        "\nGEN_RUNNER_EOF", 1
    )[0]


def _generation_runner_env(
    tmp_path: Path,
    *,
    pipelines_per_gpu: int,
    gpu_csv: str,
    games: int,
    workers: int,
    base_seed: int,
) -> tuple[dict[str, str], Path, Path]:
    tree = tmp_path / "tree"
    out = tmp_path / "out"
    rundir = tmp_path / "run"
    tree.mkdir()
    out.mkdir()
    rundir.mkdir()
    env = os.environ | {
        "TREE": str(tree),
        "CKPT": str(tmp_path / "checkpoint.pt"),
        "GEN_PY": "/bin/echo",
        "OUT": str(out),
        "RUNDIR": str(rundir),
        "GPU_CSV": gpu_csv,
        "GAMES": str(games),
        "WORKERS": str(workers),
        "BASE_SEED": str(base_seed),
        "NFULL": "128",
        "NFAST": "16",
        "PFULL": "1.0",
        "CSCALE": "0.03",
        "CLAIM_ID": "pipeline-contract",
        "USE_MPS": "0",
        "CPU_AFFINITY": "0",
        "PIPELINES_PER_GPU": str(pipelines_per_gpu),
        "SHARD_SIZE": "",
        "EVAL_SERVER_MAX_BATCH": "96",
        "EVAL_SERVER_REQUEST_COLLECTOR": "1",
        "EVAL_SERVER_MAX_NEURAL_ROWS": "",
        "SYMMETRY_AVERAGED_EVAL": "0",
        "RESCALE_NOISE_FLOOR_C": "",
        "SIGMA_EVAL": "",
        "LATE_TEMPERATURE_DECISIONS": "",
        "LATE_TEMPERATURE": "",
        "OPPONENT_MIX_MANIFEST": "",
        "EXPLOITER_FRACTION": "",
        "RUST_FEATURIZE": "0",
        "EVAL_CACHE_SIZE": "",
        "SYMMETRY_AVERAGED_EVAL_THRESHOLD": "",
        "N_FULL_WIDE": "",
        "N_FULL_WIDE_THRESHOLD": "",
        "WIDE_ROOTS_ALWAYS_FULL": "0",
        "WIDE_CANDIDATES_THRESHOLD": "",
        "VALUE_READOUT": "",
    }
    return env, out, rundir


def test_status_surfaces_live_generator_pipeline_count() -> None:
    source = STATUS.read_text(encoding="utf-8")
    assert 'GEN_PIPELINES=$(grep -c "generate_gumbel_selfplay_data"' in source
    assert "gen_pipelines=%s" in source


def test_status_surfaces_b200_rnd_workloads() -> None:
    source = STATUS.read_text(encoding="utf-8")
    assert 'ROLE="RND(posthoc)"' in source
    assert 'ROLE="RND(learner-probe)"' in source
    assert 'ROLE="RND(nccl)"' in source
    assert "posthoc_teacher_gap_probe" in source
    assert "a1_b200_microbatch_quality" in source


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--pipelines-per-gpu", "3"), "must be 1 or 2"),
        (
            ("--pipelines-per-gpu", "2", "--workers", "3"),
            "must divide evenly",
        ),
        (
            ("--pipelines-per-gpu", "2", "--workers", "4", "--games", "1"),
            "must be >= --pipelines-per-gpu",
        ),
    ],
)
def test_invalid_generation_pipeline_topology_fails_before_ssh(
    launcher_env: dict[str, str], args: tuple[str, ...], message: str
) -> None:
    result = _run(
        launcher_env,
        "teacher",
        "--base-seed",
        "72000000000",
        *args,
    )
    assert result.returncode == 2
    assert message in result.stderr
    assert "===== fleet_launch" not in result.stdout


def test_pipeline_option_is_generation_only(
    launcher_env: dict[str, str], tmp_path: Path
) -> None:
    result = _run(
        launcher_env,
        "train",
        "--data",
        str(tmp_path / "corpus"),
        "--pipelines-per-gpu",
        "1",
    )
    assert result.returncode == 2
    assert "applies only to teacher/volume generation" in result.stderr
    assert "===== fleet_launch" not in result.stdout


def test_dual_pipeline_runner_splits_totals_and_seed_ranges_exactly(
    tmp_path: Path,
) -> None:
    env, out, rundir = _generation_runner_env(
        tmp_path,
        pipelines_per_gpu=2,
        gpu_csv="0,2",
        games=5,
        workers=4,
        base_seed=100,
    )
    subprocess.run(["bash", "-c", _generation_runner()], env=env, check=True)

    expected = {
        "gpu0_pipeline0": (3, 100, 0),
        "gpu0_pipeline1": (2, 103, 1),
        "gpu2_pipeline0": (3, 105, 0),
        "gpu2_pipeline1": (2, 108, 1),
    }
    child_pids: set[str] = set()
    for directory, (pipeline_games, seed, pipeline_index) in expected.items():
        log = (out / directory / "run.log").read_text(encoding="utf-8")
        assert f"--games {pipeline_games}" in log
        assert "--workers 2" in log
        assert f"--base-seed {seed}" in log
        assert "--fleet-pipelines-per-gpu 2" in log
        assert f"--fleet-pipeline-index {pipeline_index}" in log
        assert (
            f"--fleet-pipeline-id pipeline-contract-{directory.replace('_', '-')}"
            in log
        )
        pid = (rundir / f"{directory}.pid").read_text(encoding="utf-8").strip()
        assert pid.isdigit()
        child_pids.add(pid)
    assert len(child_pids) == 4


def test_default_single_pipeline_preserves_layout_and_totals(tmp_path: Path) -> None:
    env, out, rundir = _generation_runner_env(
        tmp_path,
        pipelines_per_gpu=1,
        gpu_csv="4",
        games=5,
        workers=128,
        base_seed=900,
    )
    subprocess.run(["bash", "-c", _generation_runner()], env=env, check=True)

    log = (out / "gpu4" / "run.log").read_text(encoding="utf-8")
    assert "--games 5" in log
    assert "--workers 128" in log
    assert "--base-seed 900" in log
    assert "--fleet-pipelines-per-gpu 1" in log
    assert "--fleet-pipeline-index 0" in log
    assert not (out / "gpu4_pipeline0").exists()
    assert (rundir / "gpu4_pipeline0.pid").read_text().strip().isdigit()


def test_generation_runner_stops_siblings_on_first_pipeline_failure(
    tmp_path: Path,
) -> None:
    if not sys.platform.startswith("linux") or not shutil.which("setsid"):
        pytest.skip("fail-fast sibling PGID cleanup requires Linux setsid")
    env, _out, _rundir = _generation_runner_env(
        tmp_path,
        pipelines_per_gpu=2,
        gpu_csv="0",
        games=4,
        workers=4,
        base_seed=1_000,
    )
    sibling_pid_file = tmp_path / "sibling.pid"
    fake_python = tmp_path / "fake-generator"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "case \" $* \" in\n"
        "  *\" --fleet-pipeline-index 0 \"*) exit 7 ;;\n"
        "esac\n"
        f"printf '%s\\n' \"$$\" > {str(sibling_pid_file)!r}\n"
        "trap 'exit 143' TERM INT\n"
        "sleep 30\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env |= {
        "GEN_PY": str(fake_python),
        "FLEET_CHILD_POLL_SECONDS": "0.01",
    }

    started = time.monotonic()
    result = subprocess.run(
        ["setsid", "bash", "-c", _generation_runner()],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )
    assert result.returncode == 7, result.stdout + result.stderr
    assert time.monotonic() - started < 5
    assert "stopping sibling pipelines" in result.stderr
    if sibling_pid_file.exists():
        sibling_pid = int(sibling_pid_file.read_text())
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(sibling_pid)],
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        assert not state or state.startswith("Z")


def test_gpu_range_expansion_has_no_phantom_trailing_device() -> None:
    source = _source()
    match = re.search(r"expand_gpus\(\) \{.*?^\}", source, flags=re.DOTALL | re.MULTILINE)
    assert match is not None
    output = subprocess.check_output(
        ["bash", "-c", f"{match.group(0)}\nexpand_gpus 0-7"], text=True
    ).strip()
    assert output == "0,1,2,3,4,5,6,7"


def test_gpu_expansion_rejects_duplicate_devices() -> None:
    source = _source()
    match = re.search(r"expand_gpus\(\) \{.*?^\}", source, flags=re.DOTALL | re.MULTILINE)
    assert match is not None
    result = subprocess.run(
        ["bash", "-c", f"{match.group(0)}\nexpand_gpus 0,1,1,2"],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "duplicate" in result.stderr.lower()


def test_generation_fans_out_one_process_per_selected_gpu() -> None:
    source = _source()
    assert 'IFS="," read -r -a GPU_IDS <<< "$GPU_CSV"' in source
    assert 'for GPU in "${GPU_IDS[@]}"' in source
    assert 'CUDA_VISIBLE_DEVICES="$GPU"' in source
    assert '--out-dir "$PIPELINE_OUT"' in source
    assert '--base-seed "$PIPELINE_BASE_SEED"' in source


def test_dual_pipeline_is_explicit_opt_in_and_preserves_claim_total() -> None:
    source = _source()
    assert "PIPELINES_PER_GPU=1" in source
    assert '--pipelines-per-gpu) PIPELINES_PER_GPU="$2"' in source
    assert '[[ "$PIPELINES_PER_GPU" =~ ^[12]$ ]]' in source
    assert "WORKERS % PIPELINES_PER_GPU" in source
    assert "END=$(( BASE_SEED + GAMES * NGPU ))" in source
    assert "GAMES * PIPELINES_PER_GPU" not in source
    assert "pipelines=$PIPELINES_PER_GPU claim=$CLAIM_ID" in source


def test_seed_claim_matches_games_per_gpu_not_worker_count() -> None:
    source = _source()
    assert 'END=$(( BASE_SEED + GAMES * NGPU ))' in source
    assert 'GAMES * WORKERS * NGPU' not in source
    assert 'GPU_BASE_SEED=$(( BASE_SEED + GPU_ORDINAL * GAMES ))' in source


def test_generation_enables_proven_speed_path_explicitly() -> None:
    source = _source()
    generation_runner = source.split("<<'GEN_RUNNER_EOF'", 1)[1].split(
        "\nGEN_RUNNER_EOF", 1
    )[0]
    training_runner = source.split("<<'TRAIN_RUNNER_EOF'", 1)[1].split(
        "\nTRAIN_RUNNER_EOF", 1
    )[0]
    for flag in (
        "--native-mcts-hot-loop",
        "--rust-featurize",
        "--eval-server",
        "--eval-server-max-wait-ms 0.0",
        "--eval-server-matmul-precision highest",
        "--eval-server-transport mp_queue",
        "--eval-server-event-token-limit 0",
        "--no-root-wave-batching",
        "--no-eval-server-cuda-graph",
        "--no-eval-server-local-fallback",
        '--eval-cache-size "${EVAL_CACHE_SIZE:-0}"',
    ):
        assert flag in generation_runner

    for generation_only_control in (
        "--eval-server-transport mp_queue",
        "--eval-server-event-token-limit 0",
        "--no-root-wave-batching",
        "--no-eval-server-cuda-graph",
    ):
        assert generation_only_control not in training_runner


def test_generation_role_recipes_pin_workers_shards_and_eval_server_tuning() -> None:
    source = _source()
    assert (
        "teacher) NFULL=128; PFULL=1.0;  EVAL_SERVER_MAX_BATCH=96; "
        "EVAL_SERVER_REQUEST_COLLECTOR=1;;"
    ) in source
    assert (
        "volume)  NFULL=64;  PFULL=0.25; EVAL_SERVER_MAX_BATCH=64; "
        "EVAL_SERVER_REQUEST_COLLECTOR=0;;"
    ) in source

    defaults_start = source.index("# Measured generation defaults")
    defaults_end = source.index('[[ "$GAMES"', defaults_start)
    defaults = source[defaults_start:defaults_end]
    assert 'if [ -z "$WORKERS" ]; then' in defaults
    assert 'if [ "$ROLE" = "teacher" ]; then' in defaults
    assert 'if [ "$NGPU" -le 4 ]; then WORKERS=128; else WORKERS=64; fi' in defaults
    assert 'if [ "$NGPU" -le 4 ]; then WORKERS=48; else WORKERS=32; fi' in defaults
    assert '--workers)   WORKERS="$2"' in source

    assert 'SHARD_SIZE="${18}"; EVAL_SERVER_MAX_BATCH="${19}"' in source
    assert 'EVAL_SERVER_REQUEST_COLLECTOR="${20}"' in source
    assert 'SCIENCE_ARGS+=(--shard-size "$SHARD_SIZE")' in source
    assert '--eval-server-max-batch "$EVAL_SERVER_MAX_BATCH"' in source
    assert 'EVAL_SERVER_COLLECTOR_FLAG="--no-eval-server-request-collector"' in source
    assert 'EVAL_SERVER_COLLECTOR_FLAG="--eval-server-request-collector"' in source


def test_generation_keeps_mps_opt_in() -> None:
    source = _source()
    assert "USE_MPS=0" in source
    assert "--mps)       USE_MPS=1" in source


def test_training_uses_entity_graph_memmap_and_bf16() -> None:
    source = _source()
    for flag in (
        "--arch entity_graph",
        "--data-format memmap",
        "--amp bf16",
        "--fused-optimizer",
    ):
        assert flag in source


def test_training_requires_only_its_corpus_and_grow_from_artifact() -> None:
    source = _source()
    assert 'if [ "$ROLE" = "train" ]; then\n  OUT="$HOME/train_out/${CLAIM_ID}"' in source
    assert 'OUT="$HOME/gen_out/${CLAIM_ID}"' in source
    checks_start = source.index("FAIL=0")
    checks_end = source.index("# Fail before claiming seeds", checks_start)
    checks = source[checks_start:checks_end]
    train_start = checks.index('if [ "$ROLE" = "train" ]; then')
    generation_else = checks.index("else", train_start)
    assert 'champion $CKPT' not in checks[train_start:generation_else]
    assert 'ledger $LEDGER' not in checks[train_start:generation_else]
    assert 'champion $CKPT' in checks[generation_else:]
    assert 'ledger $LEDGER' in checks[generation_else:]


def test_launcher_defaults_to_canonical_install_tree() -> None:
    source = _source()
    assert 'TREE="${TREE:-$HOME/catan-zero-v1}"' in source
    assert 'TREE="${TREE:-$HOME/catan-zero-runsix}"' not in source


def test_noop_gate_uses_canonical_bundle_checkpoint_or_explicit_override() -> None:
    checker = NOOP_CHECK.read_text(encoding="utf-8")
    gate = GATE.read_text(encoding="utf-8")
    assert 'Path.home() / "bundle" / "champion_v0.pt"' in checker
    assert "NOOP_CHAMPION" in gate
    assert '--champion "$NOOP_CHAMPION"' in gate


def test_generation_defaults_preserve_noop_knobs_and_delegate_shard_size(
    launcher_env: dict[str, str],
) -> None:
    result = _run(launcher_env, "volume", "--base-seed", "73000000000")
    assert result.returncode == 0, result.stdout + result.stderr
    source = _source()
    assert 'volume)  NFULL=64;  PFULL=0.25;' in source
    assert "SYMMETRY_AVERAGED_EVAL=0" in source
    assert 'RESCALE_NOISE_FLOOR_C=""' in source
    assert 'SIGMA_EVAL=""' in source
    assert 'SHARD_SIZE=""' in source


def test_teacher_omits_shard_size_so_cat126_can_auto_scale_n128(
    launcher_env: dict[str, str],
) -> None:
    result = _run(launcher_env, "teacher", "--base-seed", "74000000000")
    assert result.returncode == 0, result.stdout + result.stderr
    source = _source()
    assert 'teacher) NFULL=128; PFULL=1.0;' in source
    assert '[ -z "$SHARD_SIZE" ] || SCIENCE_ARGS+=(--shard-size "$SHARD_SIZE")' in source


def test_generation_science_options_reach_each_generator_argv(
    launcher_env: dict[str, str], tmp_path: Path
) -> None:
    result = _run(
            launcher_env,
            "teacher",
            "--base-seed",
            "75000000000",
            "--symmetry-averaged-eval",
            "--rescale-noise-floor-c",
            "0.75",
            "--sigma-eval",
            "0.42",
            "--late-temperature-decisions",
            "150",
            "--late-temperature",
            "0.25",
            "--rust-featurize",
            "--eval-cache-size",
            "0",
            "--shard-size",
            "777",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    source = _source()
    for required in (
        'SCIENCE_ARGS+=(--symmetry-averaged-eval)',
        'SCIENCE_ARGS+=(--rescale-noise-floor-c "$RESCALE_NOISE_FLOOR_C")',
        'SCIENCE_ARGS+=(--sigma-eval "$SIGMA_EVAL")',
        'SCIENCE_ARGS+=(--late-temperature-decisions "$LATE_TEMPERATURE_DECISIONS")',
        'SCIENCE_ARGS+=(--late-temperature "$LATE_TEMPERATURE")',
        '--eval-cache-size "${EVAL_CACHE_SIZE:-0}"',
        'SCIENCE_ARGS+=(--shard-size "$SHARD_SIZE")',
    ):
        assert required in source


def test_exploiter_fraction_without_mix_manifest_fails_closed(
    launcher_env: dict[str, str],
) -> None:
    result = _run(
        launcher_env,
        "volume",
        "--base-seed",
        "76000000000",
        "--exploiter-fraction",
        "0.03",
    )

    assert result.returncode == 2
    assert "--exploiter-fraction requires --opponent-mix-manifest" in result.stderr


def test_opponent_mix_fails_before_remote_preflight(
    launcher_env: dict[str, str], tmp_path: Path
) -> None:
    missing = tmp_path / "opponent-mix.json"
    missing.write_text('{"categories": []}\n')
    result = _run(
        launcher_env,
        "volume",
        "--base-seed",
        "77000000000",
        "--opponent-mix-manifest",
        str(missing),
    )

    assert result.returncode == 2
    assert "incompatible with the mandatory fleet EvalServer" in result.stderr
    assert "===== fleet_launch" not in result.stdout


def test_generation_science_options_are_rejected_for_train_role(
    launcher_env: dict[str, str], tmp_path: Path
) -> None:
    result = _run(
        launcher_env,
        "train",
        "--data",
        str(tmp_path / "corpus"),
        "--rust-featurize",
    )

    assert result.returncode == 2
    assert "generation science options are not valid for role=train" in result.stderr


def test_typed_s1_s3_search_operator_reaches_each_generator(
    launcher_env: dict[str, str],
) -> None:
    result = _run(
        launcher_env,
        "teacher",
        "--base-seed",
        "78000000000",
        "--n-full",
        "128",
        "--n-fast",
        "16",
        "--p-full",
        "0.25",
        "--c-scale",
        "0.03",
        "--symmetry-averaged-eval",
        "--symmetry-averaged-eval-threshold",
        "20",
        "--n-full-wide",
        "256",
        "--n-full-wide-threshold",
        "40",
        "--wide-roots-always-full",
        "--wide-candidates-threshold",
        "24",
        "--value-readout",
        "scalar",
        "--max-neural-rows",
        "4096",
    )
    assert result.returncode == 0, result.stdout + result.stderr

    source = _source()
    for required in (
        '--n-full "$NFULL" --n-fast "$NFAST" --p-full "$PFULL"',
        '--c-scale "$CSCALE"',
        'SCIENCE_ARGS+=(--symmetry-averaged-eval-threshold "$SYMMETRY_AVERAGED_EVAL_THRESHOLD")',
        'SCIENCE_ARGS+=(--n-full-wide "$N_FULL_WIDE")',
        'SCIENCE_ARGS+=(--n-full-wide-threshold "$N_FULL_WIDE_THRESHOLD")',
        'SCIENCE_ARGS+=(--wide-roots-always-full)',
        'SCIENCE_ARGS+=(--wide-candidates-threshold "$WIDE_CANDIDATES_THRESHOLD")',
        'SCIENCE_ARGS+=(--value-readout "$VALUE_READOUT")',
        'EVAL_SERVER_ROW_CAP_ARGS=(--eval-server-max-neural-rows "$EVAL_SERVER_MAX_NEURAL_ROWS")',
    ):
        assert required in source


def test_noncanonical_c_scale_fails_before_ssh(
    launcher_env: dict[str, str],
) -> None:
    result = _run(
        launcher_env,
        "teacher",
        "--base-seed",
        "78500000000",
        "--c-scale",
        "0.1",
    )
    assert result.returncode == 2
    assert "pinned to 0.03" in result.stderr
    assert "===== fleet_launch" not in result.stdout


def test_busy_requested_gpu_fails_before_seed_claim(
    launcher_env: dict[str, str],
) -> None:
    ledger = Path(launcher_env["LEDGER"])
    before = ledger.read_text()
    env = launcher_env | {"FAKE_BUSY_GPU": "0"}
    result = _run(
        env,
        "teacher",
        "--base-seed",
        "78600000000",
        "--gpus",
        "0",
        "--go",
    )
    assert result.returncode == 3
    assert "requested GPU 0 is busy" in result.stdout
    assert ledger.read_text() == before


def test_real_training_launch_has_bounded_cuda_and_nccl_preflight() -> None:
    source = _source()
    health_start = source.index("# NVML can report an idle")
    state_creation = source.index("# ===== GO path =====")
    health = source[health_start:state_creation]

    assert '[ "$GO" = "1" ] && [ "$ROLE" = "train" ]' in health
    assert 'timeout --signal=TERM --kill-after=5 45' in health
    assert 'CUDA_VISIBLE_DEVICES="$GPU_CSV"' in health
    assert 'TORCH_NCCL_ASYNC_ERROR_HANDLING=1' in health
    assert 'torch.distributed.run --standalone --nproc_per_node="$NGPU"' in health
    assert '"$CUDA_HEALTH_SCRIPT" --expected-devices "$NGPU" --collective' in health


def test_failed_cuda_preflight_refuses_before_training_run_state(
    launcher_env: dict[str, str], tmp_path: Path
) -> None:
    tree = Path(launcher_env["TREE"])
    health_script = tree / "tools" / "cuda_health_preflight.py"
    health_script.parent.mkdir()
    health_script.write_text("# mocked by fake interpreter\n")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    grow_from = tmp_path / "grow.pt"
    grow_from.write_bytes(b"checkpoint")
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 42\n")
    fake_python.chmod(0o755)

    result = _run(
        launcher_env | {"PY": str(fake_python)},
        "train",
        "--data",
        str(corpus),
        "--grow-from",
        str(grow_from),
        "--trust-curated-data",
        "--gpus",
        "0",
        "--go",
    )

    assert result.returncode == 3
    assert "CUDA allocation health preflight failed or timed out (rc=42)" in result.stdout
    assert not (tree.parent / "fleet_runs").exists()


def test_active_mps_client_fails_before_seed_claim(
    launcher_env: dict[str, str], tmp_path: Path
) -> None:
    ledger = Path(launcher_env["LEDGER"])
    before = ledger.read_text()
    fake_bin = Path(launcher_env["PATH"].split(os.pathsep, 1)[0])
    fake_ps = fake_bin / "ps"
    fake_ps.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"$*\" = \"-eo comm=,args=\" ]; then\n"
        "  printf 'nvidia-cuda-mps-server nvidia-cuda-mps-server\\n'\n"
        "  exit 0\n"
        "fi\n"
        "exec /bin/ps \"$@\"\n"
    )
    fake_ps.chmod(0o755)
    fake_mps = fake_bin / "nvidia-cuda-mps-control"
    fake_mps.write_text(
        "#!/usr/bin/env bash\n"
        "read -r command server\n"
        "case \"$command\" in\n"
        "  get_server_list) printf '41001\\n' ;;\n"
        "  get_client_list) printf '41002\\n' ;;\n"
        "esac\n"
    )
    fake_mps.chmod(0o755)
    mps_pipe = tmp_path / "mps"
    mps_pipe.mkdir()
    (mps_pipe / "control").touch()
    env = launcher_env | {"CUDA_MPS_PIPE_DIRECTORY": str(mps_pipe)}
    result = _run(
        env,
        "teacher",
        "--base-seed",
        "78650000000",
        "--gpus",
        "0",
        "--go",
    )
    assert result.returncode == 3
    assert "MPS has active CUDA client PID(s): 41002" in result.stdout
    assert ledger.read_text() == before


def test_missing_detach_library_makes_go_launch_fail_nonzero(
    launcher_env: dict[str, str],
) -> None:
    result = _run(
        launcher_env,
        "teacher",
        "--base-seed",
        "78700000000",
        "--gpus",
        "0",
        "--go",
    )
    assert result.returncode != 0
    assert "launch_detached.sh is missing" in result.stdout


def test_ledger_append_failure_stops_before_detach(
    launcher_env: dict[str, str],
) -> None:
    status_file = Path("/proc/self/status")
    if not status_file.is_file():
        pytest.skip("requires a non-writable Linux procfs regular file")
    env = launcher_env | {"LEDGER": str(status_file)}
    result = _run(
        env,
        "teacher",
        "--base-seed",
        "78800000000",
        "--gpus",
        "0",
        "--go",
    )
    assert result.returncode != 0
    assert "could not append claim to ledger" in result.stdout
    home = Path(launcher_env["HOME"])
    assert not any((home / "fleet_runs").iterdir())
    assert not any((home / "gen_out").iterdir())


@pytest.mark.parametrize(
    "args, message",
    [
        (("--n-full-wide-threshold", "40"), "requires --n-full-wide"),
        (("--wide-roots-always-full",), "requires --n-full-wide"),
        (
            ("--symmetry-averaged-eval-threshold", "20"),
            "requires --symmetry-averaged-eval",
        ),
    ],
)
def test_typed_search_dependencies_fail_closed(
    launcher_env: dict[str, str], args: tuple[str, ...], message: str
) -> None:
    result = _run(
        launcher_env,
        "teacher",
        "--base-seed",
        "79000000000",
        *args,
    )
    assert result.returncode == 2
    assert message in result.stderr


def test_typed_search_overrides_are_generation_only(
    launcher_env: dict[str, str], tmp_path: Path
) -> None:
    result = _run(
        launcher_env,
        "train",
        "--data",
        str(tmp_path / "corpus"),
        "--n-full",
        "128",
    )
    assert result.returncode == 2
    assert "generation science options are not valid for role=train" in result.stderr
