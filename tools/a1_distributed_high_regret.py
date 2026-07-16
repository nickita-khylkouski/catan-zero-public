#!/usr/bin/env python3
"""Shard and merge promotion-grade held-out high-regret evaluation.

The immutable v3 suite remains the sole authority.  ``shard`` emits evaluator-
compatible v3 fragments plus a sealed partition manifest.  ``merge`` replays
that manifest, every fragment, every source-state identity, and every paired
outcome before publishing one report bound to the original suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import a1_promotion_artifacts as artifacts  # noqa: E402
from tools import a1_promotion_transaction as promotion  # noqa: E402
from tools.gumbel_search_cross_net_h2h import (  # noqa: E402
    _load_held_out_high_regret_suite,
)
from tools.regret_common import (  # noqa: E402
    H2H_SEARCH_RNG_CONTRACT,
    validate_h2h_search_rng_report,
)


MANIFEST_SCHEMA = "a1-distributed-high-regret-shards-v1"
PARTITION_ALGORITHM = "stratified-wide-first-stable-round-robin-v1"
PHASES = ("opening", "robber_dev", "chance", "build_trade")
ORIENTATIONS = {
    "legacy": {"candidate_first", "candidate_second"},
    "color": {"candidate_red", "candidate_blue"},
}


class DistributedHighRegretError(RuntimeError):
    """A distributed report cannot be proved equivalent to its source suite."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DistributedHighRegretError(f"cannot read {where}: {error}") from error
    if not isinstance(value, dict):
        raise DistributedHighRegretError(f"{where} must contain one JSON object")
    return value


def _regular(path: Path, *, where: str) -> Path:
    try:
        canonical = path.expanduser().resolve(strict=True)
        info = path.expanduser().lstat()
    except OSError as error:
        raise DistributedHighRegretError(f"cannot resolve {where}: {error}") from error
    if not stat.S_ISREG(info.st_mode) or path.expanduser().is_symlink():
        raise DistributedHighRegretError(f"{where} must be a regular non-symlink file")
    return canonical


def _ref(path: Path, *, where: str) -> dict[str, str]:
    canonical = _regular(path, where=where)
    return {"path": str(canonical), "sha256": _sha256(canonical)}


def _publish_exact(path: Path, value: dict[str, Any]) -> None:
    """O_EXCL publication which is safe to retry after a partial crash."""

    path = Path(os.path.abspath(os.fspath(path.expanduser())))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.resolve(strict=True) != path.parent:
        raise DistributedHighRegretError("output parent must be canonical")
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise DistributedHighRegretError(f"existing output differs: {path}")
        return
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o444)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _phase(state: dict[str, Any]) -> str:
    value = str(state.get("phase", "")).upper()
    if "BUILD_INITIAL_SETTLEMENT" in value or "BUILD_INITIAL_ROAD" in value:
        return "opening"
    if "ROBBER" in value or "KNIGHT" in value or "DEVELOPMENT_CARD" in value:
        return "robber_dev"
    if "DISCARD" in value or "ROLL" in value:
        return "chance"
    return "build_trade"


def _load_fragment(path: Path, *, where: str) -> dict[str, Any]:
    value = _load(path, where=where)
    expected = {
        "schema_version",
        "suite",
        "held_out",
        "source_manifest",
        "selection",
        "states",
        "suite_sha256",
        "validation_seed_manifest",
    }
    if (
        set(value) != expected
        or value.get("schema_version") != artifacts.HIGH_REGRET_SUITE_SCHEMA
        or value.get("suite") != "held_out_high_regret"
        or value.get("held_out") is not True
    ):
        raise DistributedHighRegretError(f"{where} schema/identity drift")
    unhashed = dict(value)
    declared = unhashed.pop("suite_sha256")
    if declared != _digest(unhashed):
        raise DistributedHighRegretError(f"{where} semantic digest mismatch")
    return value


def _partition(states: list[dict[str, Any]], shards: int) -> list[list[dict[str, Any]]]:
    if isinstance(shards, bool) or shards <= 0:
        raise DistributedHighRegretError("shards must be positive")
    by_pair: dict[int, dict[str, Any]] = {}
    for state in states:
        pair_id = state.get("pair_id")
        if isinstance(pair_id, bool) or not isinstance(pair_id, int) or pair_id in by_pair:
            raise DistributedHighRegretError("original suite has invalid pair identities")
        by_pair[pair_id] = state
    ordered = [by_pair[pair_id] for pair_id in sorted(by_pair)]
    phase_counts = {phase: sum(_phase(state) == phase for state in ordered) for phase in PHASES}
    wide_count = sum(int(state.get("legal_count", -1)) >= 41 for state in ordered)
    if any(count < 4 * shards for count in phase_counts.values()) or wide_count < 4 * shards:
        raise DistributedHighRegretError(
            "suite cannot give every fragment the required four states in every stratum"
        )
    result: list[list[dict[str, Any]]] = [[] for _ in range(shards)]
    assigned: set[int] = set()

    def fill(predicate: Any, required: int) -> None:
        for shard in range(shards):
            current = sum(predicate(state) for state in result[shard])
            for state in ordered:
                pair_id = int(state["pair_id"])
                if current >= required:
                    break
                if pair_id not in assigned and predicate(state):
                    result[shard].append(state)
                    assigned.add(pair_id)
                    current += 1
            if current < required:
                raise DistributedHighRegretError("stratified partition is infeasible")

    fill(lambda state: int(state.get("legal_count", -1)) >= 41, 4)
    for phase in PHASES:
        fill(lambda state, phase=phase: _phase(state) == phase, 4)
    for state in ordered:
        if int(state["pair_id"]) not in assigned:
            target = min(range(shards), key=lambda index: (len(result[index]), index))
            result[target].append(state)
            assigned.add(int(state["pair_id"]))
    for fragment in result:
        fragment.sort(key=lambda state: int(state["pair_id"]))
    return result


def shard_suite(*, suite_path: Path, shards: int, out_dir: Path) -> dict[str, Any]:
    suite_path, suite, _pairs = _load_held_out_high_regret_suite(suite_path)
    fragments = _partition(list(suite["states"]), shards)
    out_dir = Path(os.path.abspath(os.fspath(out_dir.expanduser())))
    records: list[dict[str, Any]] = []
    for index, states in enumerate(fragments):
        selection = dict(suite["selection"])
        selection.update(
            {
                "selected_pairs": len(states),
                "selected_unique_games": len(states),
                "stratum_min_pairs": 4,
                "selected_by_stratum": {
                    "phase:opening": 4,
                    "phase:robber_dev": 4,
                    "phase:chance": 4,
                    "phase:build_trade": 4,
                    "41+": 4,
                },
            }
        )
        fragment = {
            **suite,
            "selection": selection,
            "states": states,
        }
        fragment.pop("suite_sha256", None)
        fragment["suite_sha256"] = _digest(fragment)
        path = out_dir / f"fragment-{index:03d}.suite.json"
        _publish_exact(path, fragment)
        # The original suite was replayed once above.  Fragments contain exact
        # subsets, so repeat the cheap schema/digest check here; each evaluator
        # independently replays its fragment before doing GPU work.
        _load_fragment(path, where=f"fragment {index}")
        records.append(
            {
                "index": index,
                "suite": _ref(path, where=f"fragment {index}"),
                "suite_sha256": fragment["suite_sha256"],
                "pair_ids": [int(state["pair_id"]) for state in states],
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "partition_algorithm": PARTITION_ALGORITHM,
        "original_suite": {
            **_ref(suite_path, where="original suite"),
            "suite_sha256": suite["suite_sha256"],
        },
        "shard_count": shards,
        "fragments": records,
    }
    manifest["manifest_sha256"] = _digest(manifest)
    _publish_exact(out_dir / "partition.manifest.json", manifest)
    return manifest


def _verify_manifest(path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    path = _regular(path, where="partition manifest")
    manifest = _load(path, where="partition manifest")
    expected = {
        "schema_version", "partition_algorithm", "original_suite", "shard_count",
        "fragments", "manifest_sha256",
    }
    if set(manifest) != expected or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise DistributedHighRegretError("partition manifest schema drift")
    unhashed = dict(manifest)
    declared = unhashed.pop("manifest_sha256")
    if declared != _digest(unhashed):
        raise DistributedHighRegretError("partition manifest digest mismatch")
    if manifest.get("partition_algorithm") != PARTITION_ALGORITHM:
        raise DistributedHighRegretError("partition algorithm drift")
    original_ref = manifest.get("original_suite")
    if not isinstance(original_ref, dict) or set(original_ref) != {"path", "sha256", "suite_sha256"}:
        raise DistributedHighRegretError("original suite reference is malformed")
    original_path = _regular(Path(str(original_ref["path"])), where="original suite")
    if _sha256(original_path) != original_ref["sha256"]:
        raise DistributedHighRegretError("original suite file hash drift")
    original_path, original, _pairs = _load_held_out_high_regret_suite(original_path)
    if original["suite_sha256"] != original_ref["suite_sha256"]:
        raise DistributedHighRegretError("original suite semantic hash drift")
    return manifest, original_path, original


def _checkpoint_ref(path: Path, *, where: str) -> dict[str, str]:
    return _ref(path, where=where)


def merge_reports(
    *, manifest_path: Path, reports: Sequence[Path], candidate: Path,
    champion: Path, out: Path,
) -> dict[str, Any]:
    manifest, original_path, original = _verify_manifest(manifest_path)
    records = manifest.get("fragments")
    if not isinstance(records, list) or len(records) != manifest.get("shard_count"):
        raise DistributedHighRegretError("partition manifest fragment count drift")
    original_by_pair = {int(state["pair_id"]): state for state in original["states"]}
    fragment_by_hash: dict[str, tuple[dict[str, Any], set[int]]] = {}
    covered: set[int] = set()
    for expected_index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"index", "suite", "suite_sha256", "pair_ids"}:
            raise DistributedHighRegretError("fragment record schema drift")
        if record["index"] != expected_index:
            raise DistributedHighRegretError("fragment indexes are not canonical")
        ref = record["suite"]
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
            raise DistributedHighRegretError("fragment suite reference is malformed")
        path = _regular(Path(str(ref["path"])), where=f"fragment {expected_index}")
        if _sha256(path) != ref["sha256"]:
            raise DistributedHighRegretError("fragment suite file hash drift")
        fragment = _load_fragment(path, where=f"fragment {expected_index}")
        if fragment["suite_sha256"] != record["suite_sha256"]:
            raise DistributedHighRegretError("fragment suite semantic hash drift")
        pair_ids = [int(value) for value in record["pair_ids"]]
        if pair_ids != sorted(pair_ids) or len(pair_ids) != len(set(pair_ids)):
            raise DistributedHighRegretError("fragment pair_ids are not canonical")
        states = {int(state["pair_id"]): state for state in fragment["states"]}
        if set(states) != set(pair_ids):
            raise DistributedHighRegretError("fragment state coverage differs from manifest")
        if any(pair_id not in original_by_pair or _canonical(states[pair_id]) != _canonical(original_by_pair[pair_id]) for pair_id in pair_ids):
            raise DistributedHighRegretError("fragment state differs from original suite")
        if covered.intersection(pair_ids):
            raise DistributedHighRegretError("fragment suites duplicate original pairs")
        covered.update(pair_ids)
        fragment_by_hash[ref["sha256"]] = (record, set(pair_ids))
    if covered != set(original_by_pair):
        raise DistributedHighRegretError("fragment suites do not exactly cover original suite")

    candidate_ref = _checkpoint_ref(candidate, where="candidate")
    champion_ref = _checkpoint_ref(champion, where="champion")
    if len(reports) != len(records):
        raise DistributedHighRegretError("must provide exactly one report per fragment")
    seen_fragments: set[str] = set()
    seen_games: set[tuple[int, str]] = set()
    all_games: list[dict[str, Any]] = []
    common_config: dict[str, Any] | None = None
    common_provenance: dict[str, Any] | None = None
    encoding: str | None = None
    for report_arg in reports:
        report_path = _regular(report_arg, where="fragment report")
        report = _load(report_path, where="fragment report")
        required = {"schema_version", "suite", "held_out", "suite_manifest", "candidate", "champion", "evaluation_config", "planned_engine_identity", "engine_identity", "archived_state_reconstruction", "errors", "games", "search_rng_contract", "pentanomial_sprt", "pair_diagnostics"}
        if set(report) != required or report.get("schema_version") != artifacts.HIGH_REGRET_REPORT_SCHEMA or report.get("suite") != "held_out_high_regret" or report.get("held_out") is not True:
            raise DistributedHighRegretError("fragment report schema/identity drift")
        if report.get("errors") != []:
            raise DistributedHighRegretError("fragment report contains evaluation errors")
        if report.get("candidate") != candidate_ref or report.get("champion") != champion_ref:
            raise DistributedHighRegretError("fragment report checkpoint drift")
        suite_ref = report.get("suite_manifest")
        if not isinstance(suite_ref, dict) or set(suite_ref) != {"path", "sha256"}:
            raise DistributedHighRegretError("fragment report suite reference is malformed")
        suite_path = _regular(Path(str(suite_ref["path"])), where="report fragment suite")
        if _sha256(suite_path) != suite_ref["sha256"] or suite_ref["sha256"] not in fragment_by_hash:
            raise DistributedHighRegretError("fragment report binds an unknown/drifted suite")
        if suite_ref["sha256"] in seen_fragments:
            raise DistributedHighRegretError("multiple reports bind the same fragment")
        seen_fragments.add(suite_ref["sha256"])
        _record, expected_pairs = fragment_by_hash[suite_ref["sha256"]]
        config = dict(report["evaluation_config"])
        config.pop("pairs", None)
        if common_config is None:
            common_config = config
        elif _canonical(config) != _canonical(common_config):
            raise DistributedHighRegretError("fragment evaluation config drift")
        provenance = {
            key: report[key]
            for key in (
                "planned_engine_identity",
                "engine_identity",
                "archived_state_reconstruction",
            )
        }
        if common_provenance is None:
            common_provenance = provenance
        elif _canonical(provenance) != _canonical(common_provenance):
            raise DistributedHighRegretError("fragment engine/replay provenance drift")
        games = report.get("games")
        if not isinstance(games, list) or not games:
            raise DistributedHighRegretError("fragment report has no games")
        try:
            validate_h2h_search_rng_report(
                report.get("search_rng_contract"), games
            )
        except ValueError as error:
            raise DistributedHighRegretError(
                f"fragment report search RNG evidence does not replay: {error}"
            ) from error
        if any(
            not isinstance(game, dict)
            or game.get("truncated") is not False
            or not isinstance(game.get("candidate_won"), bool)
            or game.get("error") not in {None, ""}
            or bool(game.get("engine_divergence", False))
            for game in games
        ):
            raise DistributedHighRegretError(
                "fragment game is errored, divergent, or truncated"
            )
        normalized = [{**game, "search_won": game.get("candidate_won")} for game in games]
        scores, diagnostics = promotion.pair_scores_from_h2h_games(normalized)
        replay = promotion.evaluate_pentanomial_sprt(scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05)
        if diagnostics != report.get("pair_diagnostics") or replay != report.get("pentanomial_sprt"):
            raise DistributedHighRegretError("fragment report statistics do not replay")
        local_pairs: set[int] = set()
        for game in games:
            if not isinstance(game, dict):
                raise DistributedHighRegretError("fragment game is malformed")
            pair_id, orientation = game.get("pair_id"), game.get("orientation")
            if isinstance(pair_id, bool) or not isinstance(pair_id, int) or pair_id not in expected_pairs:
                raise DistributedHighRegretError("fragment game pair is outside its suite")
            game_encoding = "color" if orientation in ORIENTATIONS["color"] else "legacy" if orientation in ORIENTATIONS["legacy"] else None
            if game_encoding is None or (encoding is not None and game_encoding != encoding):
                raise DistributedHighRegretError("fragment reports mix/omit orientation encoding")
            encoding = game_encoding
            identity = (pair_id, str(orientation))
            if identity in seen_games:
                raise DistributedHighRegretError("duplicate paired game across fragments")
            if game.get("truncated") is not False or not isinstance(game.get("candidate_won"), bool) or game.get("error") not in {None, ""} or bool(game.get("engine_divergence", False)):
                raise DistributedHighRegretError("fragment game is errored, divergent, or truncated")
            source = original_by_pair[pair_id]
            if game.get("archived_game_seed") != source.get("game_seed") or game.get("archived_decision_index") != source.get("decision_index"):
                raise DistributedHighRegretError("fragment game archived-state identity drift")
            seen_games.add(identity)
            local_pairs.add(pair_id)
            all_games.append(game)
        if local_pairs != expected_pairs:
            raise DistributedHighRegretError("fragment report omits suite pairs")
    expected_games = {(pair_id, orientation) for pair_id in original_by_pair for orientation in ORIENTATIONS[str(encoding)]}
    if seen_games != expected_games:
        raise DistributedHighRegretError("reports do not exactly cover both orientations")
    all_games.sort(key=lambda game: (int(game["pair_id"]), str(game["orientation"])))
    normalized = [{**game, "search_won": game["candidate_won"]} for game in all_games]
    scores, diagnostics = promotion.pair_scores_from_h2h_games(normalized)
    pentanomial = promotion.evaluate_pentanomial_sprt(scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05)
    final_config = dict(common_config or {})
    final_config["pairs"] = len(original_by_pair)
    result = {
        "schema_version": artifacts.HIGH_REGRET_REPORT_SCHEMA,
        "suite": "held_out_high_regret",
        "held_out": True,
        "suite_manifest": _ref(original_path, where="original suite"),
        "candidate": candidate_ref,
        "champion": champion_ref,
        "evaluation_config": final_config,
        **dict(common_provenance or {}),
        "errors": [],
        "search_rng_contract": H2H_SEARCH_RNG_CONTRACT,
        "games": all_games,
        "pentanomial_sprt": pentanomial,
        "pair_diagnostics": diagnostics,
    }
    _publish_exact(out, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    shard = sub.add_parser("shard")
    shard.add_argument("--suite", type=Path, required=True)
    shard.add_argument("--shards", type=int, required=True)
    shard.add_argument("--out-dir", type=Path, required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--report", action="append", type=Path, required=True)
    merge.add_argument("--candidate", type=Path, required=True)
    merge.add_argument("--champion", type=Path, required=True)
    merge.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "shard":
            result = shard_suite(suite_path=args.suite, shards=args.shards, out_dir=args.out_dir)
        else:
            result = merge_reports(manifest_path=args.manifest, reports=args.report, candidate=args.candidate, champion=args.champion, out=args.out)
    except (DistributedHighRegretError, ValueError, OSError) as error:
        _parser().error(str(error))
    print(json.dumps({"schema_version": result["schema_version"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
