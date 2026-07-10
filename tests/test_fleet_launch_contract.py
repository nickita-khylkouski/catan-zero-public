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
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools" / "fleet" / "fleet_launch.sh"
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
    assert '--out-dir "$GPU_OUT"' in source
    assert '--base-seed "$GPU_BASE_SEED"' in source


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
    manifest = tmp_path / "mix fixtures" / "opponent mix.json"
    manifest.parent.mkdir()
    manifest.write_text('{"categories": []}\n')
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
            "--opponent-mix-manifest",
            str(manifest),
            "--exploiter-fraction",
            "0.03",
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
        'SCIENCE_ARGS+=(--opponent-mix-manifest "$OPPONENT_MIX_MANIFEST")',
        'SCIENCE_ARGS+=(--exploiter-fraction "$EXPLOITER_FRACTION")',
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


def test_missing_remote_mix_manifest_fails_preflight(
    launcher_env: dict[str, str], tmp_path: Path
) -> None:
    missing = tmp_path / "missing-opponent-mix.json"
    result = _run(
        launcher_env,
        "volume",
        "--base-seed",
        "77000000000",
        "--opponent-mix-manifest",
        str(missing),
    )

    assert result.returncode == 3
    assert f"FAIL: --opponent-mix-manifest {missing} missing" in result.stdout


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
        "0.1",
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
