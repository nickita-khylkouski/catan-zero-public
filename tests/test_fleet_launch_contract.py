"""Regression tests for the canonical fleet launcher's resource contract.

The launcher is shell, so these tests deliberately inspect the generated command
template.  They protect the expensive failure modes that are otherwise visible
only after a fleet launch: every selected GPU needs its own generator process,
seed claims must match the games actually emitted, and production speed/training
flags must be explicit rather than inherited from unsafe parser defaults.
"""
from __future__ import annotations

from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools" / "fleet" / "fleet_launch.sh"
GATE = ROOT / "scripts" / "gate.sh"
NOOP_CHECK = ROOT / "scripts" / "check_champion_noop.py"


def _source() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


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
        "--eval-cache-size 0",
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
        "teacher) NFULL=128; PFULL=1.0;  SHARD_SIZE=512;  "
        "EVAL_SERVER_MAX_BATCH=96; EVAL_SERVER_REQUEST_COLLECTOR=1;;"
    ) in source
    assert (
        "volume)  NFULL=64;  PFULL=0.25; SHARD_SIZE=2048; "
        "EVAL_SERVER_MAX_BATCH=64; EVAL_SERVER_REQUEST_COLLECTOR=0;;"
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
    assert '--shard-size "$SHARD_SIZE"' in source
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
