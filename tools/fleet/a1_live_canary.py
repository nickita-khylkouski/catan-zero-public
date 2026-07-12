#!/usr/bin/env python3
"""Run an exact, validation-only 4-GPU + 8-GPU A1 canary transaction.

The production render remains immutable.  This tool verifies all 40 production
lanes, selects only c1/gpu0-3 and h100-8a/gpu0-7, then derives a separately
hashed transaction by changing only job identity, output directory, game count,
seed range, and ledger-claim identity.  Every science flag, checkpoint,
opponent mix, guard, MPS binding, receipt, supervisor, and stop primitive is the
same hardened implementation used by the production executor.

Canary seeds must be wholly inside the repository's VAL-ONLY band.  A private
canary ledger is created below /home/ubuntu/gen_out and is the only ledger the
executor may synchronize; the production ledger is never consumed or mutated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from catan_zero.rl.gumbel_self_play import (  # noqa: E402
    TARGET_INFORMATION_REGIME_PUBLIC,
)
from tools import a1_pre_wave_contract as contract  # noqa: E402
from tools.fleet import a1_exact_canary as static_canary  # noqa: E402
from tools.fleet import a1_production_executor as executor  # noqa: E402

SCHEMA = "a1-live-canary-plan-v1"
RENDER_SCHEMA = "a1-live-canary-render-v1"
ATTESTATION_SCHEMA = "a1-live-canary-job-attestation-v1"
CANARY_ALIASES = {"c1": 4, "h100-8a": 8}
CATEGORY_ORDER = executor.CATEGORY_ORDER
GAMES_PER_JOB = 16  # one game per declared generation worker
JOB_COUNT = sum(CANARY_ALIASES.values()) * len(CATEGORY_ORDER)
CANARY_ROOT = Path("/home/ubuntu/gen_out")
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,47}$")
MUTABLE_VALUE_FLAGS = {
    "--out-dir",
    "--games",
    "--base-seed",
    "--ledger-claim-label",
}
REMOTE_AUDIT_SCRIPT = r"""import hashlib,json,pathlib,stat,sys
items=json.loads(sys.argv[1]);result=[]
digest=lambda b:'sha256:'+hashlib.sha256(b).hexdigest()
sha=lambda p:digest(p.read_bytes())
for item in items:
 mpath=pathlib.Path(item['manifest']);apath=pathlib.Path(item['attestation']);rpath=pathlib.Path(item['config_registry'])
 if not mpath.is_file() or not apath.is_file() or not rpath.is_file() or rpath.is_symlink(): raise SystemExit('missing canary manifest/attestation/registry: '+item['job_id'])
 m=json.loads(mpath.read_text())
 if int(m.get('games_requested',-1))!=item['games'] or int(m.get('games_completed',-1))!=item['games'] or int(m.get('games_failed',-1))!=0 or m.get('errors') not in ([],None): raise SystemExit('unclean canary manifest: '+item['job_id'])
 if int(m.get('base_seed',-1))!=item['base_seed'] or int(m.get('rows',0))<=0 or int(m.get('simulations_used_total',0))<=0: raise SystemExit('empty/drifted canary manifest: '+item['job_id'])
 if m.get('target_information_regime')!='public_conservation_pimc_v1': raise SystemExit('unsafe target regime: '+item['job_id'])
 if sha(apath)!=item['attestation_sha256']: raise SystemExit('canary attestation drift: '+item['job_id'])
 before=rpath.stat();mode=before.st_mode
 if not stat.S_ISREG(mode) or mode&0o222: raise SystemExit('canary config registry is not sealed: '+item['job_id'])
 rbytes=rpath.read_bytes();after=rpath.stat()
 if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) or len(rbytes)!=after.st_size: raise SystemExit('canary config registry mutated during read: '+item['job_id'])
 records=[json.loads(line) for line in rbytes.decode().splitlines() if line.strip()]
 if len(records)!=1 or not isinstance(records[0],dict): raise SystemExit('canary config registry count mismatch: '+item['job_id'])
 record=records[0];expected=item['config_provenance'];config=record.get('config')
 if set(record)!={'config_hash','full_config_hash','pipeline','timestamp','purpose','config'} or not isinstance(config,dict) or not isinstance(record.get('timestamp'),str) or not isinstance(record.get('purpose'),str): raise SystemExit('canary config registry fields mismatch: '+item['job_id'])
 full=digest(json.dumps(config,sort_keys=True,separators=(',',':')).encode());short='sha256:'+full.removeprefix('sha256:')[:16]
 if record.get('pipeline')!='generate' or config.get('pipeline')!='generate' or record.get('full_config_hash')!=full or record.get('config_hash')!=short or m.get('config_hash')!=short or any(record.get(k)!=expected.get(k) for k in ('pipeline','config_hash','full_config_hash','config')): raise SystemExit('canary config registry mismatch: '+item['job_id'])
 result.append({'job_id':item['job_id'],'rows':int(m['rows']),'simulations':int(m['simulations_used_total']),'manifest_sha256':sha(mpath),'attestation_sha256':sha(apath),'config_hash':short,'full_config_hash':full,'config_registry_sha256':digest(rbytes)})
print(json.dumps(result,sort_keys=True))"""
MPS_RUNTIME_ATTESTATION_SCRIPT = r"""import json,pathlib,subprocess,sys,time
required=int(sys.argv[1]);timeout=float(sys.argv[2]);not_before_ns=int(sys.argv[3]);expected=json.loads(sys.argv[4]);deadline=time.monotonic()+timeout
def nofile_soft(pid):
 for line in pathlib.Path(f'/proc/{pid}/limits').read_text().splitlines():
  if line.startswith('Max open files'):
   value=line.split()[3]
   return -1 if value=='unlimited' else int(value)
 raise RuntimeError(f'Max open files is absent for MPS server {pid}')
def positive_progress(item):
 for job in item['jobs']:
  root=pathlib.Path(job['output']);workers={};valid=True
  for index in range(job['workers']):
   worker_id=f'worker_{index:03d}';path=root/worker_id/'progress.json'
   try:
    if not path.is_file() or path.is_symlink(): valid=False;break
    stat=path.stat();payload=json.loads(path.read_text())
   except (OSError,UnicodeError,json.JSONDecodeError): valid=False;break
   rows=payload.get('rows');simulations=payload.get('simulations_used_total');failed=payload.get('games_failed')
   if stat.st_mtime_ns<not_before_ns or type(rows) is not int or rows<=0 or type(simulations) is not int or simulations<=0 or type(failed) is not int or failed!=0:
    valid=False;break
   workers[worker_id]={'progress':str(path),'rows':rows,'simulations':simulations,'games_failed':failed,'mtime_ns':stat.st_mtime_ns}
  if valid and len(workers)==job['workers']:
   return {'worker_id':item['worker_id'],'gpu':item['gpu'],'job_id':job['job_id'],'output':job['output'],'expected_workers':job['workers'],'workers':workers}
 return None
last='no MPS server reported by nvidia-smi'
while time.monotonic()<deadline:
 remaining=max(0.0,deadline-time.monotonic())
 try: response=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True,capture_output=True,check=False,timeout=max(0.1,min(2.0,remaining)))
 except subprocess.TimeoutExpired:
  last='nvidia-smi compute query timed out';continue
 if response.returncode:
  last='nvidia-smi compute query failed: '+response.stderr;time.sleep(min(0.25,max(0.0,deadline-time.monotonic())));continue
 pids=sorted({int(line.split(',',1)[0].strip()) for line in response.stdout.splitlines() if 'nvidia-cuda-mps-server' in line})
 observed={}
 try:
  for pid in pids: observed[str(pid)]=nofile_soft(pid)
 except (FileNotFoundError,ProcessLookupError) as error:
  last='MPS server changed during inspection: '+repr(error);time.sleep(0.25);continue
 if observed:
  low={pid:value for pid,value in observed.items() if value!=-1 and value<required}
  if low: raise SystemExit(f'MPS server RLIMIT_NOFILE below {required}: {low}')
  progress={}
  for item in expected:
   evidence=positive_progress(item)
   if evidence is not None: progress[item['worker_id']]=evidence
  missing=sorted(item['worker_id'] for item in expected if item['worker_id'] not in progress)
  if not missing:
   print(json.dumps({'required_nofile_soft':required,'server_nofile_soft':observed,'canary_lane_progress':progress},sort_keys=True));raise SystemExit(0)
  last='canary lanes without positive progress: '+repr(missing)
 time.sleep(0.25)
raise SystemExit(f'MPS/canary runtime proof incomplete within {timeout}s: {last}')"""


class CanaryError(RuntimeError):
    """A selective canary cannot be proven isolated and recipe-identical."""


class CanaryCleanupError(CanaryError):
    """A post-launch failure occurred and its exact-stop also failed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CanaryError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise CanaryError(f"{path} must contain a JSON object")
    return value


def _create_exact(path: Path, payload: Mapping[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"
    _create_exact_bytes(path, data)


def _create_exact_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise CanaryError(f"immutable canary artifact drift: {path}")
        return
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        return _create_exact_bytes(path, data)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _flag_value(argv: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise CanaryError(f"argv must contain one value for {flag}")
    return str(argv[positions[0] + 1])


def _replace_values(argv: Sequence[str], replacements: Mapping[str, str]) -> list[str]:
    result = list(argv)
    for flag, replacement in replacements.items():
        index = result.index(flag) if result.count(flag) == 1 else -1
        if index < 0 or index + 1 >= len(result) or result[index + 1].startswith("--"):
            raise CanaryError(f"cannot replace missing/non-value flag {flag}")
        result[index + 1] = replacement
    return result


def _assert_exact_recipe(
    original: Sequence[str], derived: Sequence[str], *, native_runtime: bool = False
) -> None:
    original_values, original_switches = static_canary._flag_map(
        list(original), job_id="source"
    )
    derived_values, derived_switches = static_canary._flag_map(
        list(derived), job_id="canary"
    )
    expected_switches = set(original_switches)
    if native_runtime:
        expected_switches -= {"--no-native-mcts-hot-loop", "--no-rust-featurize"}
        expected_switches |= {"--native-mcts-hot-loop", "--rust-featurize"}
    if expected_switches != derived_switches:
        raise CanaryError("canary changed a source recipe switch")
    if set(original_values) != set(derived_values):
        raise CanaryError("canary changed the source recipe flag set")
    for flag in set(original_values) - MUTABLE_VALUE_FLAGS:
        if original_values[flag] != derived_values[flag]:
            raise CanaryError(f"canary changed immutable recipe flag {flag}")
    for flag, expected in static_canary.EXPECTED_VALUE_FLAGS.items():
        if derived_values.get(flag) != expected:
            raise CanaryError(f"canary exact recipe requires {flag}={expected}")
    missing = static_canary.REQUIRED_SWITCHES - derived_switches
    forbidden = static_canary.FORBIDDEN_FLAGS & (set(derived_values) | derived_switches)
    if missing or forbidden:
        raise CanaryError(
            f"canary guard/recipe switch drift: missing={sorted(missing)} "
            f"forbidden={sorted(forbidden)}"
        )


def _validate_scalar_attestation(lock: Mapping[str, Any]) -> None:
    science = lock.get("science")
    if not isinstance(science, Mapping):
        raise CanaryError("lock has no science payload")
    evaluator = science.get("evaluator")
    if not isinstance(evaluator, Mapping) or evaluator.get("value_readout") != "scalar":
        raise CanaryError("live canary requires the sealed scalar teacher readout")
    producers = [
        item for item in lock.get("checkpoints", []) if item.get("role") == "producer"
    ]
    if len(producers) != 1:
        raise CanaryError("live canary requires exactly one producer checkpoint")
    metadata = producers[0].get("metadata")
    if not isinstance(metadata, Mapping):
        try:
            metadata = contract._checkpoint_metadata(  # noqa: SLF001
                Path(str(producers[0]["path"])),
                checkpoint_sha256=str(producers[0]["sha256"]),
                value_readout="scalar",
                require_trained_readout=True,
                legacy_scalar_attestation=None,
            )
        except (contract.ContractError, KeyError) as error:
            raise CanaryError(
                f"producer lacks inspectable masked-checkpoint provenance: {error}"
            ) from error
    if metadata.get("mask_hidden_info") is not True:
        raise CanaryError("producer lacks masked-checkpoint provenance")
    attestation = metadata.get("legacy_scalar_readout_attestation")
    if metadata.get("value_training_schema") == "value-training-v1":
        return
    if not isinstance(attestation, Mapping) or attestation.get(
        "schema_version"
    ) != "legacy-scalar-readout-attestation-v1":
        raise CanaryError("producer lacks trained scalar-readout provenance")


def _selected_lanes(
    lanes: Mapping[str, list[dict[str, Any]]],
    canary_aliases: Mapping[str, int] = CANARY_ALIASES,
) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    placements: dict[str, list[int]] = {alias: [] for alias in canary_aliases}
    for worker_id, lane in lanes.items():
        if not lane:
            continue
        alias = str(lane[0].get("host_alias"))
        if alias not in canary_aliases:
            continue
        gpu = int(lane[0].get("gpu", -1))
        placements[alias].append(gpu)
        selected[worker_id] = lane
    for alias, count in canary_aliases.items():
        if sorted(placements[alias]) != list(range(count)):
            raise CanaryError(
                f"source render must expose {alias} gpu0-{count - 1}, got "
                f"{sorted(placements[alias])}"
            )
    if len(selected) != sum(canary_aliases.values()):
        raise CanaryError("source render does not yield the requested canary lanes")
    return selected


def _canary_ledger_row(start: int, end: int, job_id: str, claim: str) -> str:
    return f"[{start} – {end}) | VAL-ONLY a1-live-canary/{job_id} claim={claim}"


def _canary_config_provenance(
    source: Mapping[str, Any],
    *,
    games: int,
    base_seed: int,
    native_runtime: bool = False,
) -> dict[str, Any]:
    """Derive the validation-only typed config from its verified source command."""

    provenance = source.get("config_provenance")
    if not isinstance(provenance, Mapping) or not isinstance(
        provenance.get("config"), Mapping
    ):
        raise CanaryError("source command lacks typed config provenance")
    config = json.loads(json.dumps(provenance["config"]))
    fields = config.get("fields")
    if not isinstance(fields, dict):
        raise CanaryError("source typed config fields are malformed")
    fields["games"] = int(games)
    fields["base_seed"] = int(base_seed)
    if native_runtime:
        fields["native_mcts_hot_loop"] = True
        fields["rust_featurize"] = True
    full_hash = _digest(config)
    result = {
        "pipeline": "generate",
        "config_hash": "sha256:" + full_hash.removeprefix("sha256:")[:16],
        "full_config_hash": full_hash,
        "config": config,
    }
    result["provenance_sha256"] = _digest(result)
    return result


def derive_canary_plan(
    *,
    lock: dict[str, Any],
    rendered: dict[str, Any],
    lanes: Mapping[str, list[dict[str, Any]]],
    hosts: dict[str, Any],
    lock_path: Path,
    render_path: Path,
    hosts_path: Path,
    receipt_path: Path,
    canary_id: str,
    base_seed: int,
    canary_root: Path,
    canary_aliases: Mapping[str, int] = CANARY_ALIASES,
    games_per_job: int = GAMES_PER_JOB,
    native_runtime: bool = False,
    categories: Sequence[str] = CATEGORY_ORDER,
    allowed_root: Path = CANARY_ROOT,
    repo_root: Path = _REPO_ROOT,
) -> dict[str, Any]:
    if not SAFE_ID.fullmatch(canary_id):
        raise CanaryError("--canary-id must match [a-z0-9][a-z0-9-]{2,47}")
    _validate_scalar_attestation(lock)
    allowed = allowed_root.resolve()
    root = canary_root.resolve()
    if root.parent != allowed or root.name != f"a1-live-canary-{canary_id}":
        raise CanaryError(
            f"canary root must be exactly {allowed}/a1-live-canary-{canary_id}"
        )
    production_root = Path(str(lock["fleet"]["output_root"])).resolve()
    if (
        root == production_root
        or production_root in root.parents
        or root in production_root.parents
    ):
        raise CanaryError("canary output root overlaps the production output tree")

    canary_aliases = dict(canary_aliases)
    categories = tuple(categories)
    if not categories or any(category not in CATEGORY_ORDER for category in categories):
        raise CanaryError("canary categories must be a non-empty production-order subset")
    if tuple(sorted(categories, key=CATEGORY_ORDER.index)) != categories:
        raise CanaryError("canary categories must retain production dependency order")
    if not canary_aliases or any(
        not alias or isinstance(count, bool) or count <= 0
        for alias, count in canary_aliases.items()
    ):
        raise CanaryError("canary aliases must bind positive GPU counts")
    if isinstance(games_per_job, bool) or games_per_job <= 0:
        raise CanaryError("games per job must be positive")
    chosen = _selected_lanes(lanes, canary_aliases)
    lo, hi = contract.VAL_ONLY_SEED_RANGE
    job_count = sum(canary_aliases.values()) * len(categories)
    total_games = job_count * games_per_job
    if base_seed < lo or base_seed + total_games > hi:
        raise CanaryError(
            f"canary [{base_seed},{base_seed + total_games}) must fit VAL-ONLY [{lo},{hi})"
        )

    source_by_job = {item["job_id"]: item for item in rendered["commands"]}
    new_lanes: dict[str, list[dict[str, Any]]] = {}
    new_commands: list[dict[str, Any]] = []
    ledger_rows: list[str] = []
    next_seed = int(base_seed)
    attestation_dir = root / "operator" / "job_attestations"
    canary_ledger = root / "CANARY_LEDGER.md"
    for source_worker, source_lane in sorted(chosen.items()):
        alias = str(source_lane[0]["host_alias"])
        gpu = int(source_lane[0]["gpu"])
        worker_id = f"canary-{canary_id}__{source_worker}"
        transformed: list[dict[str, Any]] = []
        previous: str | None = None
        for source_command in source_lane:
            category = str(source_command["category"])
            if category not in categories:
                continue
            source_job = str(source_command["job_id"])
            if source_by_job.get(source_job) != source_command:
                raise CanaryError(f"source lane command drift for {source_job}")
            job_id = f"{worker_id}__{category}"
            out_dir = root / f"{alias}_gpu{gpu}__{category}"
            claim = f"val-{canary_id}-{alias}-gpu{gpu}-{category}"
            seed_start = next_seed
            seed_end = seed_start + games_per_job
            next_seed = seed_end
            argv = _replace_values(
                source_command["argv"],
                {
                    "--out-dir": str(out_dir),
                    "--games": str(games_per_job),
                    "--base-seed": str(seed_start),
                    "--ledger-claim-label": claim,
                },
            )
            if native_runtime:
                if argv.count("--no-native-mcts-hot-loop") != 1:
                    raise CanaryError("source command lacks explicit native-loop binding")
                if argv.count("--no-rust-featurize") != 1:
                    raise CanaryError("source command lacks explicit Rust-featurizer binding")
                argv[argv.index("--no-native-mcts-hot-loop")] = (
                    "--native-mcts-hot-loop"
                )
                argv[argv.index("--no-rust-featurize")] = "--rust-featurize"
            _assert_exact_recipe(
                source_command["argv"], argv, native_runtime=native_runtime
            )
            environment = {
                **source_command["environment"],
                "CATAN_SEED_LEDGER": str(canary_ledger),
                "CATAN_A1_CANARY_ID": canary_id,
                contract.CONFIG_REGISTRY_ENVIRONMENT_VARIABLE: str(
                    out_dir / contract.CONFIG_REGISTRY_FILENAME
                ),
            }
            source_attestation = source_command["output_attestation"]
            source_path = Path(str(source_attestation["source"]))
            if (
                not source_path.is_file()
                or _sha256(source_path) != source_attestation["source_file_sha256"]
            ):
                raise CanaryError(f"source job attestation drift for {source_job}")
            source_payload = _load(source_path)
            attestation: dict[str, Any] = {
                "schema_version": ATTESTATION_SCHEMA,
                "validation_only": True,
                "target_information_regime": TARGET_INFORMATION_REGIME_PUBLIC,
                "canary_id": canary_id,
                "contract_sha256": lock["contract_sha256"],
                "source_render_sha256": rendered["render_sha256"],
                "source_job": {
                    "job_id": source_job,
                    "attestation_file_sha256": source_attestation["source_file_sha256"],
                    "attestation_payload_sha256": _digest(source_payload),
                },
                "job_id": job_id,
                "worker_id": worker_id,
                "host_alias": alias,
                "gpu": gpu,
                "category": category,
                "base_seed": seed_start,
                "games": games_per_job,
                "seed_end": seed_end,
                "output_dir": str(out_dir),
                "argv_sha256": _digest(argv),
            }
            attestation["attestation_sha256"] = _digest(attestation)
            attestation_path = attestation_dir / f"{job_id}.json"
            _create_exact(attestation_path, attestation)
            row = _canary_ledger_row(seed_start, seed_end, job_id, claim)
            ledger_rows.append(row)
            command = {
                **source_command,
                "job_id": job_id,
                "worker_id": worker_id,
                "argv": argv,
                "argv_sha256": _digest(argv),
                "environment": environment,
                "environment_sha256": _digest(environment),
                "config_provenance": _canary_config_provenance(
                    source_command,
                    games=games_per_job,
                    base_seed=seed_start,
                    native_runtime=native_runtime,
                ),
                "ledger_claim": {
                    "validation_only": True,
                    "path": str(canary_ledger),
                    "row": row,
                    "row_sha256": _digest(row),
                },
                "output_attestation": {
                    "source": str(attestation_path),
                    "source_file_sha256": _sha256(attestation_path),
                    "destination": str(out_dir / "a1_contract.json"),
                    "payload_sha256": _digest(attestation),
                },
                "must_run_after": [] if previous is None else [previous],
                "source_production_job_id": source_job,
            }
            transformed.append(command)
            new_commands.append(command)
            previous = job_id
        new_lanes[worker_id] = transformed

    ledger_data = (
        "# A1 LIVE CANARY LEDGER — VALIDATION ONLY; NEVER MERGE INTO PRODUCTION\n"
        + "\n".join(ledger_rows)
        + "\n"
    ).encode()
    _create_exact_bytes(canary_ledger, ledger_data)
    canary_ledger_record = {
        "path": str(canary_ledger),
        "sha256": _sha256(canary_ledger),
        "validation_only": True,
    }

    required = dict(rendered["required_artifacts"])
    production_ledger_path = str(required["seed_ledger"]["path"])
    required["seed_ledger"] = canary_ledger_record
    canary_render: dict[str, Any] = {
        "schema_version": RENDER_SCHEMA,
        "validation_only": True,
        "canary_id": canary_id,
        "source_contract_sha256": lock["contract_sha256"],
        "source_render": {
            "path": str(render_path.resolve()),
            "sha256": _sha256(render_path),
            "render_sha256": rendered["render_sha256"],
        },
        "production_seed_ledger": {
            "path": production_ledger_path,
            "read_only": True,
            "claims_consumed": 0,
        },
        "canary_seed_ledger": canary_ledger_record,
        "seed_range": [base_seed, next_seed],
        "output_root": str(root),
        "games_per_job": games_per_job,
        "canary_aliases": canary_aliases,
        "native_runtime": native_runtime,
        "categories": list(categories),
        "lane_count": len(new_lanes),
        "job_count": len(new_commands),
        "required_artifacts": required,
        "commands": new_commands,
    }
    canary_render["render_sha256"] = _digest(canary_render)
    canary_render_path = root / "operator" / "commands.canary.json"
    _create_exact(canary_render_path, canary_render)

    repo_artifacts = executor._repo_artifacts(rendered, repo_root=repo_root)
    companion = Path(__file__).resolve()
    relative_companion = companion.relative_to(repo_root.resolve())
    if not any(record["path"] == str(relative_companion) for record in repo_artifacts):
        repo_artifacts.append(
            {
                "path": str(relative_companion),
                "sha256": _sha256(companion),
                "mode": 0o555 if os.access(companion, os.X_OK) else 0o444,
            }
        )
        repo_artifacts.sort(key=lambda record: record["path"])

    subset_hosts = dict(hosts)
    subset_hosts["hosts"] = {
        alias: hosts["hosts"][alias] for alias in sorted(canary_aliases)
    }
    subset_hosts["remote_root"] = (
        str(hosts["remote_root"]).rstrip("/") + f"/live-canary-{canary_id}"
    )
    plan: dict[str, Any] = {
        "schema_version": executor.RECEIPT_SCHEMA,
        "canary_schema_version": SCHEMA,
        "status": "dry_run",
        "validation_only": True,
        "canary_id": canary_id,
        "contract_sha256": lock["contract_sha256"],
        "render_sha256": canary_render["render_sha256"],
        "source_render_sha256": rendered["render_sha256"],
        "lock": str(lock_path.resolve()),
        "render": str(canary_render_path),
        "operator_manifests": {
            "lock": {"sha256": _sha256(lock_path), "remote_name": "contract.lock.json"},
            "render": {
                "sha256": _sha256(canary_render_path),
                "remote_name": "commands.canary.json",
            },
        },
        "hosts_config_sha256": _sha256(hosts_path),
        "remote_root": subset_hosts["remote_root"],
        "lane_count": len(new_lanes),
        "job_count": len(new_commands),
        "canary_aliases": canary_aliases,
        "games_per_job": games_per_job,
        "native_runtime": native_runtime,
        "claim_count": 0,
        "canary_claim_count": len(ledger_rows),
        "production_claims_consumed": 0,
        "category_order": list(categories),
        "client_environment": dict(executor.CLIENT_ENVIRONMENT),
        "repo_artifacts_sha256": _digest(repo_artifacts),
        "live_seed_ledger_sha256": _sha256(canary_ledger),
        "canary_seed_ledger": str(canary_ledger),
        "production_seed_ledger": production_ledger_path,
        "output_root": str(root),
        "receipt": str(receipt_path.resolve()),
        "lanes": [
            {
                "worker_id": worker_id,
                "host_alias": lane[0]["host_alias"],
                "gpu": lane[0]["gpu"],
                "jobs": [item["job_id"] for item in lane],
            }
            for worker_id, lane in sorted(new_lanes.items())
        ],
    }
    plan["plan_sha256"] = _digest(plan)
    plan["_private"] = {
        "hosts": subset_hosts,
        "lanes": new_lanes,
        "rendered": canary_render,
        "repo_artifacts": repo_artifacts,
    }
    validate_canary_plan(plan)
    return plan


def validate_canary_plan(plan: Mapping[str, Any]) -> None:
    executor._verify_plan_digest(plan)
    aliases = plan.get("canary_aliases", CANARY_ALIASES)
    games_per_job = plan.get("games_per_job", GAMES_PER_JOB)
    native_runtime = bool(plan.get("native_runtime", False))
    categories = tuple(plan.get("category_order", CATEGORY_ORDER))
    if not isinstance(aliases, Mapping) or not aliases:
        raise CanaryError("canary host topology is missing")
    expected_lanes = sum(int(count) for count in aliases.values())
    expected_jobs = expected_lanes * len(categories)
    if (
        plan.get("validation_only") is not True
        or plan.get("canary_schema_version") != SCHEMA
        or plan.get("lane_count") != expected_lanes
        or plan.get("job_count") != expected_jobs
        or plan.get("claim_count") != 0
        or plan.get("production_claims_consumed") != 0
    ):
        raise CanaryError("canary plan scope/claim invariants drifted")
    private = plan.get("_private")
    if not isinstance(private, Mapping):
        raise CanaryError("canary plan lacks private execution state")
    rendered = private.get("rendered")
    hosts = private.get("hosts")
    lanes = private.get("lanes")
    if (
        not isinstance(rendered, Mapping)
        or not isinstance(hosts, Mapping)
        or not isinstance(lanes, Mapping)
    ):
        raise CanaryError("canary private execution state is malformed")
    if set(hosts.get("hosts", {})) != set(aliases):
        raise CanaryError("canary host set differs from its sealed topology")
    canary_ledger = str(plan.get("canary_seed_ledger"))
    if rendered["required_artifacts"]["seed_ledger"]["path"] != canary_ledger:
        raise CanaryError("executor seed-ledger stage is not the canary ledger")
    if canary_ledger == str(plan.get("production_seed_ledger")):
        raise CanaryError("canary ledger aliases the production ledger")
    for lane in lanes.values():
        for command in lane:
            out_dir = Path(_flag_value(command["argv"], "--out-dir"))
            if Path(str(plan["output_root"])) not in out_dir.parents:
                raise CanaryError("canary output escapes its validation root")
            expected_environment = {
                **contract.SEALED_RUNTIME_ENVIRONMENT,
                "CUDA_VISIBLE_DEVICES": str(command["gpu"]),
                **executor.CLIENT_ENVIRONMENT,
                "CATAN_SEED_LEDGER": canary_ledger,
                "CATAN_A1_CONTRACT_SHA256": str(plan["contract_sha256"]),
                contract.CONFIG_REGISTRY_ENVIRONMENT_VARIABLE: str(
                    out_dir / contract.CONFIG_REGISTRY_FILENAME
                ),
                "CATAN_A1_CANARY_ID": str(plan["canary_id"]),
            }
            if command.get("environment") != expected_environment:
                raise CanaryError("canary command exact environment drift")
            if command.get("environment_sha256") != _digest(expected_environment):
                raise CanaryError("canary command environment digest drift")
            registry = Path(
                expected_environment[contract.CONFIG_REGISTRY_ENVIRONMENT_VARIABLE]
            )
            if registry.parent != out_dir or not registry.is_relative_to(
                Path(str(plan["output_root"]))
            ):
                raise CanaryError("canary config registry leaked outside validation output")
            provenance = command.get("config_provenance")
            if not isinstance(provenance, Mapping):
                raise CanaryError("canary command lacks typed config provenance")
            unhashed = dict(provenance)
            declared = unhashed.pop("provenance_sha256", None)
            config = unhashed.get("config")
            fields = config.get("fields") if isinstance(config, Mapping) else None
            if (
                declared != _digest(unhashed)
                or not isinstance(fields, Mapping)
                or fields.get("games") != games_per_job
                or bool(fields.get("native_mcts_hot_loop", False)) != native_runtime
                or bool(fields.get("rust_featurize", False)) != native_runtime
                or fields.get("base_seed")
                != int(_flag_value(command["argv"], "--base-seed"))
            ):
                raise CanaryError("canary typed config provenance drift")
            start = int(_flag_value(command["argv"], "--base-seed"))
            games = int(_flag_value(command["argv"], "--games"))
            lo, hi = contract.VAL_ONLY_SEED_RANGE
            if not (lo <= start < start + games <= hi):
                raise CanaryError("canary command escaped the VAL-ONLY seed band")


def _verify_current_producer_source(
    lock_path: Path,
    render_path: Path,
    *,
    verify_lock_fn: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Verify the valid promoted self-play slice of a mixed historical render.

    The issued dual-arm lock contains opponent jobs that predate the promoted
    checkpoint/search-identity guard and execute at c_scale=.03.  Those jobs
    are intentionally unusable.  A corrected-teacher pilot needs only the
    current-producer self-play jobs, all of which execute at the promoted .10
    identity.  Verify every byte and typed field for that slice while refusing
    to reinterpret or execute the invalid opponent rows.
    """

    lock = verify_lock_fn(lock_path, require_all_job_claims=False)
    rendered = _load(render_path)
    unhashed = dict(rendered)
    declared = unhashed.pop("render_sha256", None)
    if declared != _digest(unhashed):
        raise CanaryError("source render semantic digest mismatch")
    if rendered.get("contract_sha256") != lock.get("contract_sha256"):
        raise CanaryError("source render binds a different lock")
    jobs = {str(job["job_id"]): job for job in lock["fleet"]["jobs"]}
    mix_paths = {
        Path(record["path"]).stem: Path(record["path"])
        for record in rendered["required_artifacts"]["rendered_opponent_mix"]
    }
    commands = [
        command
        for command in rendered.get("commands", [])
        if command.get("category") == "current_producer"
    ]
    expected_lanes = int(lock["game_contract"]["worker_count"])
    if len(commands) != expected_lanes:
        raise CanaryError("source render does not contain one current job per lane")
    lanes: dict[str, list[dict[str, Any]]] = {}
    for command in commands:
        job_id = str(command.get("job_id", ""))
        job = jobs.get(job_id)
        if not isinstance(job, Mapping) or job.get("category") != "current_producer":
            raise CanaryError(f"unknown current-producer source job {job_id!r}")
        try:
            identity = contract._promoted_producer_job_identity(lock, job)  # noqa: SLF001
        except contract.ContractError as error:
            raise CanaryError(f"unsafe promoted producer identity for {job_id}: {error}") from error
        if identity is None or float(identity["executed_search_operator"]["c_scale"]) != 0.1:
            raise CanaryError(f"{job_id} is not bound to promoted c_scale=.10")
        expected_argv = contract._generator_argv(lock, job, mix_paths=mix_paths)  # noqa: SLF001
        expected_environment = contract._job_environment(lock, job)  # noqa: SLF001
        if command.get("argv") != expected_argv or command.get("argv_sha256") != _digest(expected_argv):
            raise CanaryError(f"source argv drift for {job_id}")
        if command.get("environment") != expected_environment or command.get(
            "environment_sha256"
        ) != _digest(expected_environment):
            raise CanaryError(f"source environment drift for {job_id}")
        source = Path(str(command["output_attestation"]["source"]))
        if not source.is_file() or _sha256(source) != command["output_attestation"].get(
            "source_file_sha256"
        ):
            raise CanaryError(f"source attestation drift for {job_id}")
        lanes.setdefault(str(command["worker_id"]), []).append(command)
    if len(lanes) != expected_lanes or any(len(lane) != 1 for lane in lanes.values()):
        raise CanaryError("source current-producer lane topology drift")
    return lock, rendered, lanes


def build_canary_plan(
    *,
    lock_path: Path,
    render_path: Path,
    hosts_path: Path,
    receipt_path: Path,
    canary_id: str,
    base_seed: int,
    canary_root: Path,
    canary_aliases: Mapping[str, int] = CANARY_ALIASES,
    games_per_job: int = GAMES_PER_JOB,
    native_runtime: bool = False,
    categories: Sequence[str] = CATEGORY_ORDER,
) -> dict[str, Any]:
    lock_path = lock_path.resolve(strict=True)
    render_path = render_path.resolve(strict=True)

    def verify_without_claims(
        path: Path, *, require_all_job_claims: bool = False
    ) -> dict[str, Any]:
        if path.resolve(strict=True) != lock_path:
            raise CanaryError("source lock path drift")
        # A validation canary proves recipe/runtime shape before production
        # claims exist. It must never require or append those claims.
        return contract.verify_lock(lock_path, require_all_job_claims=False)

    if tuple(categories) == CATEGORY_ORDER:
        lock, rendered, lanes = executor.verify_render(
            lock_path, render_path, verify_lock_fn=verify_without_claims
        )
    elif tuple(categories) == ("current_producer",):
        lock, rendered, lanes = _verify_current_producer_source(
            lock_path, render_path, verify_lock_fn=verify_without_claims
        )
    else:
        raise CanaryError("selective live canary supports current_producer only")
    aliases = {lane[0]["host_alias"] for lane in lanes.values()}
    hosts = executor.load_hosts(hosts_path, aliases)
    return derive_canary_plan(
        lock=lock,
        rendered=rendered,
        lanes=lanes,
        hosts=hosts,
        lock_path=lock_path,
        render_path=render_path,
        hosts_path=hosts_path,
        receipt_path=receipt_path,
        canary_id=canary_id,
        base_seed=base_seed,
        canary_root=canary_root,
        canary_aliases=canary_aliases,
        games_per_job=games_per_job,
        native_runtime=native_runtime,
        categories=categories,
    )


def audit_canary(plan: dict[str, Any]) -> dict[str, Any]:
    validate_canary_plan(plan)
    hosts = plan["_private"]["hosts"]
    by_alias: dict[str, list[dict[str, Any]]] = {}
    for lane in plan["_private"]["lanes"].values():
        by_alias.setdefault(lane[0]["host_alias"], []).extend(lane)
    results: list[dict[str, Any]] = []
    for alias, commands in sorted(by_alias.items()):
        expected = [
            {
                "job_id": command["job_id"],
                "manifest": str(
                    Path(_flag_value(command["argv"], "--out-dir")) / "manifest.json"
                ),
                "attestation": command["output_attestation"]["destination"],
                "attestation_sha256": command["output_attestation"][
                    "source_file_sha256"
                ],
                "config_registry": command["environment"][
                    contract.CONFIG_REGISTRY_ENVIRONMENT_VARIABLE
                ],
                "config_provenance": command["config_provenance"],
                "games": int(_flag_value(command["argv"], "--games")),
                "base_seed": int(_flag_value(command["argv"], "--base-seed")),
            }
            for command in commands
        ]
        command = " ".join(
            shlex.quote(value)
            for value in (
                hosts["python"],
                "-c",
                REMOTE_AUDIT_SCRIPT,
                json.dumps(expected, sort_keys=True, separators=(",", ":")),
            )
        )
        response = executor._ssh(hosts, alias, command)
        if response.returncode != 0:
            raise CanaryError(
                f"canary audit failed on {alias}: {(response.stderr or response.stdout).strip()}"
            )
        try:
            host_results = json.loads(response.stdout)
        except json.JSONDecodeError as error:
            raise CanaryError(
                f"canary audit returned invalid JSON on {alias}"
            ) from error
        if not isinstance(host_results, list) or len(host_results) != len(commands):
            raise CanaryError(f"canary audit result count drift on {alias}")
        expected_by_id = {str(item["job_id"]): item for item in commands}
        if {
            str(item.get("job_id"))
            for item in host_results
            if isinstance(item, Mapping)
        } != set(expected_by_id):
            raise CanaryError(f"canary audit job identity drift on {alias}")
        for item in host_results:
            if not isinstance(item, Mapping):
                raise CanaryError(f"canary audit result shape drift on {alias}")
            source = expected_by_id[str(item["job_id"])]
            provenance = source["config_provenance"]
            if (
                item.get("config_hash") != provenance["config_hash"]
                or item.get("full_config_hash") != provenance["full_config_hash"]
                or item.get("attestation_sha256")
                != source["output_attestation"]["source_file_sha256"]
                or not str(item.get("config_registry_sha256", "")).startswith(
                    "sha256:"
                )
                or type(item.get("rows")) is not int
                or int(item["rows"]) <= 0
                or type(item.get("simulations")) is not int
                or int(item["simulations"]) <= 0
            ):
                raise CanaryError(f"canary audit provenance drift on {alias}")
        results.extend(host_results)
    report = {
        "schema_version": "a1-live-canary-audit-v1",
        "status": "PASS",
        "validation_only": True,
        "canary_id": plan["canary_id"],
        "plan_sha256": plan["plan_sha256"],
        "job_count": len(results),
        "rows": sum(int(item["rows"]) for item in results),
        "simulations": sum(int(item["simulations"]) for item in results),
        "jobs": sorted(results, key=lambda item: item["job_id"]),
    }
    report["audit_sha256"] = _digest(report)
    return report


def _runtime_lane_expectations(
    plan: Mapping[str, Any], alias: str
) -> list[dict[str, Any]]:
    expectations: list[dict[str, Any]] = []
    for worker_id, lane in sorted(plan["_private"]["lanes"].items()):
        if lane[0]["host_alias"] != alias:
            continue
        expectations.append(
            {
                "worker_id": worker_id,
                "gpu": int(lane[0]["gpu"]),
                "jobs": [
                    {
                        "job_id": command["job_id"],
                        "output": _flag_value(command["argv"], "--out-dir"),
                        "workers": min(
                            int(_flag_value(command["argv"], "--workers")),
                            int(_flag_value(command["argv"], "--games")),
                        ),
                    }
                    for command in lane
                ],
            }
        )
    aliases = plan.get("canary_aliases", CANARY_ALIASES)
    expected_gpus = list(range(int(aliases[alias])))
    if sorted(item["gpu"] for item in expectations) != expected_gpus:
        raise CanaryError(f"runtime lane topology drift on {alias}")
    return expectations


def attest_mps_runtime(
    plan: dict[str, Any],
    *,
    not_before_epoch: float,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    """Prove every canary GPU produced rows through a safely-limited MPS server."""

    validate_canary_plan(plan)
    if (
        isinstance(not_before_epoch, bool)
        or not isinstance(not_before_epoch, (int, float))
        or not_before_epoch <= 0
    ):
        raise CanaryError("MPS runtime attestation requires a positive not-before time")
    not_before_ns = int(float(not_before_epoch) * 1_000_000_000)
    hosts = plan["_private"]["hosts"]
    reports: dict[str, Any] = {}
    for alias in sorted(plan.get("canary_aliases", CANARY_ALIASES)):
        expected_lanes = _runtime_lane_expectations(plan, alias)
        command = " ".join(
            shlex.quote(value)
            for value in (
                hosts["python"],
                "-c",
                MPS_RUNTIME_ATTESTATION_SCRIPT,
                str(executor.REQUIRED_NOFILE_SOFT),
                str(timeout_seconds),
                str(not_before_ns),
                json.dumps(expected_lanes, sort_keys=True, separators=(",", ":")),
            )
        )
        try:
            response = executor._ssh(
                hosts,
                alias,
                command,
                timeout_seconds=max(1.0, timeout_seconds + 15.0),
            )
        except subprocess.TimeoutExpired as error:
            raise CanaryError(
                f"MPS runtime attestation transport timed out on {alias}"
            ) from error
        if response.returncode != 0:
            raise CanaryError(
                f"MPS runtime attestation failed on {alias}: "
                f"{(response.stderr or response.stdout).strip()}"
            )
        try:
            report = json.loads(response.stdout)
        except json.JSONDecodeError as error:
            raise CanaryError(
                f"MPS runtime attestation returned invalid JSON on {alias}"
            ) from error
        limits = report.get("server_nofile_soft")
        progress = report.get("canary_lane_progress")
        expected_by_worker = {item["worker_id"]: item for item in expected_lanes}
        if (
            report.get("required_nofile_soft") != executor.REQUIRED_NOFILE_SOFT
            or not isinstance(limits, dict)
            or not limits
            or any(
                type(value) is not int
                or (value != -1 and value < executor.REQUIRED_NOFILE_SOFT)
                for value in limits.values()
            )
        ):
            raise CanaryError(f"unsafe MPS runtime limit report on {alias}")
        if not isinstance(progress, dict) or set(progress) != set(expected_by_worker):
            raise CanaryError(f"incomplete canary lane progress report on {alias}")
        for worker_id, evidence in progress.items():
            expected = expected_by_worker[worker_id]
            expected_jobs = {item["job_id"]: item for item in expected["jobs"]}
            job = expected_jobs.get(evidence.get("job_id")) if isinstance(evidence, dict) else None
            if (
                not isinstance(evidence, dict)
                or evidence.get("worker_id") != worker_id
                or evidence.get("gpu") != expected["gpu"]
                or not isinstance(job, dict)
                or evidence.get("output") != job["output"]
                or evidence.get("expected_workers") != job["workers"]
                or not isinstance(evidence.get("workers"), dict)
                or set(evidence["workers"])
                != {f"worker_{index:03d}" for index in range(job["workers"])}
            ):
                raise CanaryError(
                    f"unsafe canary lane progress evidence for {worker_id} on {alias}"
                )
            for progress_worker_id, worker_evidence in evidence["workers"].items():
                expected_progress = str(
                    Path(job["output"]) / progress_worker_id / "progress.json"
                )
                if (
                    not isinstance(worker_evidence, dict)
                    or worker_evidence.get("progress") != expected_progress
                    or type(worker_evidence.get("rows")) is not int
                    or worker_evidence["rows"] <= 0
                    or type(worker_evidence.get("simulations")) is not int
                    or worker_evidence["simulations"] <= 0
                    or worker_evidence.get("games_failed") != 0
                    or type(worker_evidence.get("mtime_ns")) is not int
                    or worker_evidence["mtime_ns"] < not_before_ns
                ):
                    raise CanaryError(
                        f"stale/unsafe canary worker progress for {worker_id} on {alias}"
                    )
        reports[alias] = report
    return {
        "required_nofile_soft": executor.REQUIRED_NOFILE_SOFT,
        "not_before_epoch": float(not_before_epoch),
        "hosts": reports,
    }


def _public(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key != "_private"}


def _parse_host_shapes(values: Sequence[str] | None) -> dict[str, int]:
    if not values:
        return dict(CANARY_ALIASES)
    result: dict[str, int] = {}
    for value in values:
        alias, separator, raw_count = value.partition("=")
        if not separator or not alias or alias in result:
            raise CanaryError(
                f"invalid/duplicate --host-shape {value!r}; expected ALIAS=GPU_COUNT"
            )
        try:
            count = int(raw_count)
        except ValueError as error:
            raise CanaryError(f"invalid GPU count in --host-shape {value!r}") from error
        if count not in (4, 8):
            raise CanaryError("live canary host shapes must contain 4 or 8 GPUs")
        result[alias] = count
    return result


def _receipt_has_launch_state(plan: Mapping[str, Any], receipt_path: Path) -> bool:
    """Return whether an interrupted execute durably recorded a live lane prefix."""

    if not receipt_path.is_file():
        return False
    try:
        receipt = _load(receipt_path)
    except CanaryError:
        return False
    lane_pids = receipt.get("lane_pids")
    pending_worker = receipt.get("launch_pending_worker_id")
    lanes = plan.get("_private", {}).get("lanes", {})
    return (
        receipt.get("plan_sha256") == plan.get("plan_sha256")
        and receipt.get("status") in {"launching", "launched", "launch_failed"}
        and isinstance(lane_pids, dict)
        and (
            bool(lane_pids)
            or (
                isinstance(pending_worker, str)
                and isinstance(lanes, Mapping)
                and pending_worker in lanes
            )
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status", "stop", "audit"):
        item = sub.add_parser(name)
        item.add_argument("--lock", required=True, type=Path)
        item.add_argument("--render", required=True, type=Path)
        item.add_argument("--hosts", required=True, type=Path)
        item.add_argument("--receipt", required=True, type=Path)
        item.add_argument("--canary-id", required=True)
        item.add_argument("--base-seed", required=True, type=int)
        item.add_argument(
            "--host-shape",
            action="append",
            metavar="ALIAS=GPU_COUNT",
            help="select an exact source-render host; repeat to form the pilot cohort",
        )
        item.add_argument(
            "--games-per-job",
            type=int,
            default=GAMES_PER_JOB,
            help="validation games for each of the three sealed source categories",
        )
        item.add_argument(
            "--native-runtime",
            action="store_true",
            help="use parity-proven Rust featurization and native MCTS hot loop",
        )
        item.add_argument(
            "--current-producer-only",
            action="store_true",
            help="pilot only promoted self-play; refuse legacy mismatched opponent jobs",
        )
        item.add_argument(
            "--canary-root",
            type=Path,
            help="must equal /home/ubuntu/gen_out/a1-live-canary-CANARY_ID",
        )
    sub.choices["run"].add_argument("--resume", action="store_true")
    sub.choices["run"].add_argument("--go", action="store_true")
    sub.choices["stop"].add_argument("--go", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    canary_root = args.canary_root or CANARY_ROOT / f"a1-live-canary-{args.canary_id}"
    try:
        canary_aliases = _parse_host_shapes(args.host_shape)
        plan = build_canary_plan(
            lock_path=args.lock,
            render_path=args.render,
            hosts_path=args.hosts,
            receipt_path=args.receipt,
            canary_id=args.canary_id,
            base_seed=args.base_seed,
            canary_root=canary_root,
            canary_aliases=canary_aliases,
            games_per_job=args.games_per_job,
            native_runtime=bool(args.native_runtime),
            categories=(
                ("current_producer",)
                if args.current_producer_only
                else CATEGORY_ORDER
            ),
        )
        if args.command == "status":
            result = executor.status(plan, receipt_path=args.receipt)
        elif args.command == "stop":
            result = executor.stop_execution(
                plan, receipt_path=args.receipt, go=bool(args.go)
            )
        elif args.command == "audit":
            result = audit_canary(plan)
        elif args.go:
            runtime_not_before = time.time()
            execute_returned = False
            try:
                result = executor.execute(
                    plan, receipt_path=args.receipt, resume=bool(args.resume)
                )
                execute_returned = True
                result["mps_runtime"] = attest_mps_runtime(
                    plan, not_before_epoch=runtime_not_before
                )
                executor._atomic_json(args.receipt, result)
            except BaseException as primary_error:
                if not execute_returned and not _receipt_has_launch_state(
                    plan, args.receipt
                ):
                    raise
                try:
                    executor.stop_execution(plan, receipt_path=args.receipt, go=True)
                except BaseException as cleanup_error:
                    if isinstance(primary_error, (KeyboardInterrupt, SystemExit)):
                        primary_error.add_note(
                            "exact-stop also failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                        raise primary_error from cleanup_error
                    if isinstance(cleanup_error, (KeyboardInterrupt, SystemExit)):
                        cleanup_error.add_note(
                            "post-launch validation/persistence had already failed: "
                            f"{type(primary_error).__name__}: {primary_error}"
                        )
                        raise cleanup_error from primary_error
                    raise CanaryCleanupError(
                        "post-launch validation/persistence failed: "
                        f"{type(primary_error).__name__}: {primary_error}; "
                        "exact-stop also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    ) from primary_error
                raise
        else:
            result = _public(plan)
    except (CanaryError, executor.ExecutorError, OSError) as error:
        print(f"REFUSING: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
