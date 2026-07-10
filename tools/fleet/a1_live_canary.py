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
import sys
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
REMOTE_AUDIT_SCRIPT = r"""import hashlib,json,pathlib,sys
items=json.loads(sys.argv[1]);result=[]
sha=lambda p:'sha256:'+hashlib.sha256(p.read_bytes()).hexdigest()
for item in items:
 mpath=pathlib.Path(item['manifest']);apath=pathlib.Path(item['attestation'])
 if not mpath.is_file() or not apath.is_file(): raise SystemExit('missing canary manifest/attestation: '+item['job_id'])
 m=json.loads(mpath.read_text())
 if int(m.get('games_requested',-1))!=item['games'] or int(m.get('games_completed',-1))!=item['games'] or int(m.get('games_failed',-1))!=0 or m.get('errors') not in ([],None): raise SystemExit('unclean canary manifest: '+item['job_id'])
 if int(m.get('base_seed',-1))!=item['base_seed'] or int(m.get('rows',0))<=0 or int(m.get('simulations_used_total',0))<=0: raise SystemExit('empty/drifted canary manifest: '+item['job_id'])
 if m.get('target_information_regime')!='public_conservation_pimc_v1': raise SystemExit('unsafe target regime: '+item['job_id'])
 if sha(apath)!=item['attestation_sha256']: raise SystemExit('canary attestation drift: '+item['job_id'])
 result.append({'job_id':item['job_id'],'rows':int(m['rows']),'simulations':int(m['simulations_used_total']),'manifest_sha256':sha(mpath),'attestation_sha256':sha(apath)})
print(json.dumps(result,sort_keys=True))"""


class CanaryError(RuntimeError):
    """A selective canary cannot be proven isolated and recipe-identical."""


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


def _assert_exact_recipe(original: Sequence[str], derived: Sequence[str]) -> None:
    original_values, original_switches = static_canary._flag_map(
        list(original), job_id="source"
    )
    derived_values, derived_switches = static_canary._flag_map(
        list(derived), job_id="canary"
    )
    if original_switches != derived_switches:
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
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("mask_hidden_info") is not True
    ):
        raise CanaryError("producer lacks masked-checkpoint provenance")
    attestation = metadata.get("legacy_scalar_readout_attestation")
    if (
        not isinstance(attestation, Mapping)
        or attestation.get("schema_version") != "legacy-scalar-readout-attestation-v1"
    ):
        raise CanaryError("producer lacks the typed legacy scalar attestation")


def _selected_lanes(
    lanes: Mapping[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    placements: dict[str, list[int]] = {alias: [] for alias in CANARY_ALIASES}
    for worker_id, lane in lanes.items():
        if not lane:
            continue
        alias = str(lane[0].get("host_alias"))
        if alias not in CANARY_ALIASES:
            continue
        gpu = int(lane[0].get("gpu", -1))
        placements[alias].append(gpu)
        selected[worker_id] = lane
    for alias, count in CANARY_ALIASES.items():
        if sorted(placements[alias]) != list(range(count)):
            raise CanaryError(
                f"source render must expose {alias} gpu0-{count - 1}, got "
                f"{sorted(placements[alias])}"
            )
    if len(selected) != sum(CANARY_ALIASES.values()):
        raise CanaryError("source render does not yield exactly 12 canary lanes")
    return selected


def _canary_ledger_row(start: int, end: int, job_id: str, claim: str) -> str:
    return f"[{start} – {end}) | VAL-ONLY a1-live-canary/{job_id} claim={claim}"


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

    chosen = _selected_lanes(lanes)
    lo, hi = contract.VAL_ONLY_SEED_RANGE
    total_games = JOB_COUNT * GAMES_PER_JOB
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
            source_job = str(source_command["job_id"])
            if source_by_job.get(source_job) != source_command:
                raise CanaryError(f"source lane command drift for {source_job}")
            job_id = f"{worker_id}__{category}"
            out_dir = root / f"{alias}_gpu{gpu}__{category}"
            claim = f"val-{canary_id}-{alias}-gpu{gpu}-{category}"
            seed_start = next_seed
            seed_end = seed_start + GAMES_PER_JOB
            next_seed = seed_end
            argv = _replace_values(
                source_command["argv"],
                {
                    "--out-dir": str(out_dir),
                    "--games": str(GAMES_PER_JOB),
                    "--base-seed": str(seed_start),
                    "--ledger-claim-label": claim,
                },
            )
            _assert_exact_recipe(source_command["argv"], argv)
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
                "games": GAMES_PER_JOB,
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
                "environment": {
                    **source_command["environment"],
                    "CATAN_SEED_LEDGER": str(canary_ledger),
                    "CATAN_A1_CANARY_ID": canary_id,
                },
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
        "games_per_job": GAMES_PER_JOB,
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
        alias: hosts["hosts"][alias] for alias in sorted(CANARY_ALIASES)
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
        "claim_count": 0,
        "canary_claim_count": len(ledger_rows),
        "production_claims_consumed": 0,
        "category_order": list(CATEGORY_ORDER),
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
    if (
        plan.get("validation_only") is not True
        or plan.get("canary_schema_version") != SCHEMA
        or plan.get("lane_count") != 12
        or plan.get("job_count") != 36
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
    if set(hosts.get("hosts", {})) != set(CANARY_ALIASES):
        raise CanaryError("canary host set must be exactly c1 and h100-8a")
    canary_ledger = str(plan.get("canary_seed_ledger"))
    if rendered["required_artifacts"]["seed_ledger"]["path"] != canary_ledger:
        raise CanaryError("executor seed-ledger stage is not the canary ledger")
    if canary_ledger == str(plan.get("production_seed_ledger")):
        raise CanaryError("canary ledger aliases the production ledger")
    for lane in lanes.values():
        for command in lane:
            if command["environment"].get("CATAN_SEED_LEDGER") != canary_ledger:
                raise CanaryError("canary command references a non-canary ledger")
            out_dir = Path(_flag_value(command["argv"], "--out-dir"))
            if Path(str(plan["output_root"])) not in out_dir.parents:
                raise CanaryError("canary output escapes its validation root")
            start = int(_flag_value(command["argv"], "--base-seed"))
            games = int(_flag_value(command["argv"], "--games"))
            lo, hi = contract.VAL_ONLY_SEED_RANGE
            if not (lo <= start < start + games <= hi):
                raise CanaryError("canary command escaped the VAL-ONLY seed band")


def build_canary_plan(
    *,
    lock_path: Path,
    render_path: Path,
    hosts_path: Path,
    receipt_path: Path,
    canary_id: str,
    base_seed: int,
    canary_root: Path,
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

    lock, rendered, lanes = executor.verify_render(
        lock_path, render_path, verify_lock_fn=verify_without_claims
    )
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


def _public(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key != "_private"}


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
        plan = build_canary_plan(
            lock_path=args.lock,
            render_path=args.render,
            hosts_path=args.hosts,
            receipt_path=args.receipt,
            canary_id=args.canary_id,
            base_seed=args.base_seed,
            canary_root=canary_root,
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
            result = executor.execute(
                plan, receipt_path=args.receipt, resume=bool(args.resume)
            )
        else:
            result = _public(plan)
    except (CanaryError, executor.ExecutorError, OSError) as error:
        print(f"REFUSING: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
