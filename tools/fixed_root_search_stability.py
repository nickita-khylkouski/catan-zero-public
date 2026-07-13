#!/usr/bin/env python3
"""Attributable fixed-root search stability/cost probe for S2/S3.

This tool compares two *named* Gumbel search operators on the exact same
persisted panel of real Rust-engine roots.  Each root is stored as both a
canonical JSON snapshot and a deterministic reconstruction transcript.  A
live run reconstructs every root and fails before search if any snapshot,
legal-action set, checkpoint, evaluator config, or panel hash has drifted.

Unlike whole-game H2H timing, this artifact makes search cost attributable:
every recorded search reports ``SearchResult.simulations_used``, wall time,
logical leaf evaluations, evaluator method calls, and orientation-expanded
evaluation rows (so a D6 root counts as 12 orientation rows).  Repeated runs
use disjoint search seeds within and across roles and report pairwise
Jensen-Shannon divergence and top-1 agreement globally and by phase/legal
width, including the S3-critical ``>=40`` slice.

The root panel is generated with a checkpoint's deterministic raw argmax
policy.  Panel creation is explicit (``--create-root-panel``); otherwise the
tool is read-only with respect to the panel.  Evaluator caching must be zero:
cache hits would make wall/evaluation cost depend on role ordering and would
therefore invalidate the comparison.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
_LOCAL_SRC = _REPO_ROOT / "src"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
# The B200 R&D checkout intentionally shares a venv with the artifact checkout.
# Put the source tree next to this probe first even when PYTHONPATH already
# mentioned it later, so semantics cannot silently come from an older editable
# install in that shared venv.
try:
    sys.path.remove(str(_LOCAL_SRC))
except ValueError:
    pass
sys.path.insert(0, str(_LOCAL_SRC))

from catan_zero.rl.gumbel_self_play import _apply_selected_action  # noqa: E402
from catan_zero.search.gumbel_chance_mcts import (  # noqa: E402
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
)
from catan_zero.search.neural_rust_mcts import (  # noqa: E402
    BatchedEntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)
from catan_zero.search.rust_mcts import _require_rust_module  # noqa: E402
from factory_common import write_json  # noqa: E402

COLORS: tuple[str, ...] = ("RED", "BLUE")
PANEL_SCHEMA = "fixed-root-search-panel-v2"
SEARCH_CONFIG_SCHEMA = "fixed-root-search-config-v1"
EVALUATOR_CONFIG_SCHEMA = "fixed-root-evaluator-config-v1"
REPORT_SCHEMA = "fixed-root-search-stability-v2"
SNAPSHOT_CANONICALIZATION = "rust-snapshot-semantic-node-edge-v1"
CHANCE_SEED_XOR = 0x51A7E
DEFAULT_ROOT_BASE_SEED = 920_001
DEFAULT_SEARCH_SEED_BASE_A = 2_100_001
DEFAULT_SEARCH_SEED_BASE_B = 3_100_001

_PHASE_LABELS = {
    "BUILD_INITIAL_SETTLEMENT": "opening_placement",
    "BUILD_INITIAL_ROAD": "opening_placement",
    "MOVE_ROBBER": "robber",
    "DISCARD": "discard",
    "PLAY_TURN": "play_turn",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def verify_locked_files(locked_hashes: dict[str, str]) -> None:
    """Re-hash every immutable input after the probe; reject mid-run drift."""
    drifted: dict[str, dict[str, str]] = {}
    for path, expected in locked_hashes.items():
        actual = file_sha256(path)
        if actual != expected:
            drifted[path] = {"expected": expected, "actual": actual}
    if drifted:
        raise RuntimeError(f"immutable probe input changed during run: {drifted}")


def _normalize_policy(policy: dict[int, float]) -> dict[int, float]:
    if not policy:
        raise ValueError("policy must not be empty")
    normalized: dict[int, float] = {}
    for action, probability in policy.items():
        value = float(probability)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"policy probability for action {action} must be finite and non-negative"
            )
        normalized[int(action)] = value
    total = sum(normalized.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("policy must have positive finite mass")
    return {action: value / total for action, value in normalized.items()}


def jensen_shannon_divergence(
    first: dict[int, float], second: dict[int, float], *, eps: float = 1.0e-12
) -> float:
    """Symmetric JS divergence in nats over the union of action supports."""
    p = _normalize_policy(first)
    q = _normalize_policy(second)
    keys = set(p) | set(q)
    midpoint = {key: 0.5 * (p.get(key, 0.0) + q.get(key, 0.0)) for key in keys}

    def _kl(left: dict[int, float]) -> float:
        return sum(
            probability
            * math.log((probability + eps) / (midpoint[action] + eps))
            for action, probability in left.items()
            if probability > 0.0
        )

    return max(0.0, 0.5 * (_kl(p) + _kl(q)))


def _policy_top1(policy: dict[int, float]) -> int:
    normalized = _normalize_policy(policy)
    return int(
        max(normalized, key=lambda action: (normalized[action], -int(action)))
    )


def summarize_cross_seed_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize all C(repeats, 2) independent-seed comparisons for one root."""
    if len(runs) < 2:
        raise ValueError("at least two repeated searches are required")
    seeds = [int(run["search_seed"]) for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("repeated searches must use distinct search seeds")

    supports = [set(int(key) for key in run["improved_policy"]) for run in runs]
    if any(support != supports[0] for support in supports[1:]):
        raise ValueError("repeated searches returned incompatible legal-action supports")

    pairwise: list[dict[str, Any]] = []
    policy_top1s = [_policy_top1(run["improved_policy"]) for run in runs]
    selected_actions = [int(run["selected_action"]) for run in runs]
    for selected, support in zip(selected_actions, supports):
        if selected not in support:
            raise ValueError(
                f"selected action {selected} is outside the improved-policy support"
            )
    for left, right in itertools.combinations(range(len(runs)), 2):
        pairwise.append(
            {
                "run_indices": [left, right],
                "search_seeds": [seeds[left], seeds[right]],
                "js_divergence": jensen_shannon_divergence(
                    runs[left]["improved_policy"], runs[right]["improved_policy"]
                ),
                # SearchResult.selected_action is the production T=0 choice
                # emitted by the named operator.  On the normal policy-choice
                # path it includes search's visit/prior tie-breaks, unlike a
                # fresh arbitrary argmax over improved_policy.  This is the
                # binding S3 top-1 stability metric.
                "top1_agreement": bool(
                    selected_actions[left] == selected_actions[right]
                ),
                "policy_argmax_agreement": bool(
                    policy_top1s[left] == policy_top1s[right]
                ),
            }
        )
    js_values = [float(pair["js_divergence"]) for pair in pairwise]
    agreement = [1.0 if pair["top1_agreement"] else 0.0 for pair in pairwise]
    modal_count = max(
        selected_actions.count(action) for action in set(selected_actions)
    )
    return {
        "repeats": len(runs),
        "pair_count": len(pairwise),
        "cross_seed_js_mean": sum(js_values) / len(js_values),
        "cross_seed_js_median": statistics.median(js_values),
        "cross_seed_js_max": max(js_values),
        "top1_pair_agreement": sum(agreement) / len(agreement),
        "top1_modal_fraction": modal_count / len(selected_actions),
        "top1_actions": selected_actions,
        "policy_argmax_actions": policy_top1s,
        "pairwise": pairwise,
    }


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def _median(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.median(materialized) if materialized else None


def _legal_width_bucket(width: int) -> str:
    if width <= 1:
        return "1"
    if width <= 4:
        return "2-4"
    if width <= 10:
        return "5-10"
    if width <= 20:
        return "11-20"
    if width <= 40:
        return "21-40"
    return "41+"


def _phase_label(raw_phase: str) -> str:
    return _PHASE_LABELS.get(str(raw_phase), str(raw_phase).lower() or "unknown")


class CountingEvaluator:
    """Transparent evaluator adapter with exact logical-evaluation counters."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.logical_leaf_evaluations = 0
        self.orientation_evaluation_rows = 0
        self.evaluator_method_calls = 0
        self.evaluate_calls = 0
        self.evaluate_many_calls = 0
        self.symmetry_calls = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "logical_leaf_evaluations": int(self.logical_leaf_evaluations),
            "orientation_evaluation_rows": int(self.orientation_evaluation_rows),
            "evaluator_method_calls": int(self.evaluator_method_calls),
            "evaluate_calls": int(self.evaluate_calls),
            "evaluate_many_calls": int(self.evaluate_many_calls),
            "symmetry_calls": int(self.symmetry_calls),
        }

    @staticmethod
    def delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
        return {key: int(after[key]) - int(before[key]) for key in before}

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        self.logical_leaf_evaluations += 1
        self.orientation_evaluation_rows += 1
        self.evaluator_method_calls += 1
        self.evaluate_calls += 1
        return self.inner.evaluate(*args, **kwargs)

    def evaluate_many(self, requests: list[Any], *args: Any, **kwargs: Any) -> Any:
        count = len(requests)
        self.logical_leaf_evaluations += count
        self.orientation_evaluation_rows += count
        self.evaluator_method_calls += 1
        self.evaluate_many_calls += 1
        return self.inner.evaluate_many(requests, *args, **kwargs)

    def evaluate_symmetry_averaged(self, *args: Any, **kwargs: Any) -> Any:
        # The production D6 helper averages all 12 dihedral orientations.
        self.logical_leaf_evaluations += 1
        self.orientation_evaluation_rows += 12
        self.evaluator_method_calls += 1
        self.symmetry_calls += 1
        return self.inner.evaluate_symmetry_averaged(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _load_json_object(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def load_search_spec(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = _load_json_object(path)
    allowed_top = {"schema_version", "name", "search_config"}
    unknown_top = set(payload) - allowed_top
    if unknown_top:
        raise ValueError(f"{path}: unknown top-level keys: {sorted(unknown_top)}")
    if payload.get("schema_version") != SEARCH_CONFIG_SCHEMA:
        raise ValueError(
            f"{path}: schema_version must be {SEARCH_CONFIG_SCHEMA!r}"
        )
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{path}: non-empty string name is required")
    overrides = payload.get("search_config")
    if not isinstance(overrides, dict):
        raise ValueError(f"{path}: search_config must be an object")

    field_names = {field.name for field in dataclasses.fields(GumbelChanceMCTSConfig)}
    unknown = set(overrides) - field_names
    if unknown:
        raise ValueError(f"{path}: unknown GumbelChanceMCTSConfig keys: {sorted(unknown)}")
    if "seed" in overrides:
        raise ValueError(
            f"{path}: search_config.seed is forbidden; the probe assigns disjoint seeds"
        )
    if "colors" in overrides:
        raise ValueError(f"{path}: search_config.colors is fixed to {COLORS}")

    defaults = dataclasses.asdict(GumbelChanceMCTSConfig())
    defaults.pop("seed", None)
    defaults["colors"] = list(COLORS)
    effective = {**defaults, **overrides}
    effective["colors"] = list(COLORS)
    # Construction catches missing/incompatible field shapes early.  Search
    # seeds are injected only at execution time and are not part of the
    # operator hash.
    constructor = dict(effective)
    constructor["colors"] = COLORS
    GumbelChanceMCTSConfig(seed=0, **constructor)
    if int(effective["n_full"]) <= 0 or int(effective["n_fast"]) <= 0:
        raise ValueError(f"{path}: n_full and n_fast must be positive")
    if float(effective["temperature"]) != 0.0:
        raise ValueError(f"{path}: temperature must be 0 for a stability probe")

    return {
        "name": name.strip(),
        "source_path": str(path.resolve()),
        "source_file_sha256": file_sha256(path),
        "effective_search_config": effective,
        "effective_search_config_sha256": content_sha256(effective),
    }


def load_evaluator_spec(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = _load_json_object(path)
    allowed_top = {"schema_version", "evaluator_config"}
    unknown_top = set(payload) - allowed_top
    if unknown_top:
        raise ValueError(f"{path}: unknown top-level keys: {sorted(unknown_top)}")
    if payload.get("schema_version") != EVALUATOR_CONFIG_SCHEMA:
        raise ValueError(
            f"{path}: schema_version must be {EVALUATOR_CONFIG_SCHEMA!r}"
        )
    overrides = payload.get("evaluator_config")
    if not isinstance(overrides, dict):
        raise ValueError(f"{path}: evaluator_config must be an object")
    field_names = {field.name for field in dataclasses.fields(EntityGraphRustEvaluatorConfig)}
    unknown = set(overrides) - field_names
    if unknown:
        raise ValueError(
            f"{path}: unknown EntityGraphRustEvaluatorConfig keys: {sorted(unknown)}"
        )
    defaults = dataclasses.asdict(EntityGraphRustEvaluatorConfig())
    effective = {**defaults, **overrides}
    EntityGraphRustEvaluatorConfig(**effective)
    if int(effective["cache_size"]) != 0:
        raise ValueError(
            f"{path}: cache_size must be exactly 0 for attributable role cost; "
            "cache reuse would bias whichever role runs second"
        )
    return {
        "source_path": str(path.resolve()),
        "source_file_sha256": file_sha256(path),
        "effective_evaluator_config": effective,
        "effective_evaluator_config_sha256": content_sha256(effective),
    }


def _make_search_config(spec: dict[str, Any], *, seed: int) -> GumbelChanceMCTSConfig:
    values = dict(spec["effective_search_config"])
    values["colors"] = COLORS
    return GumbelChanceMCTSConfig(seed=int(seed), **values)


def validate_search_comparison(
    spec_a: dict[str, Any],
    spec_b: dict[str, Any],
    *,
    allowed_differences: set[str],
) -> dict[str, dict[str, Any]]:
    """Reject accidental operator drift outside the predeclared A/B dose."""
    field_names = {field.name for field in dataclasses.fields(GumbelChanceMCTSConfig)}
    unknown_allowed = set(allowed_differences) - field_names
    if unknown_allowed:
        raise ValueError(
            "unknown --allowed-search-config-differences fields: "
            f"{sorted(unknown_allowed)}"
        )
    first = spec_a["effective_search_config"]
    second = spec_b["effective_search_config"]
    observed = {
        key: {str(spec_a["name"]): first.get(key), str(spec_b["name"]): second.get(key)}
        for key in sorted(set(first) | set(second))
        if first.get(key) != second.get(key)
    }
    undeclared = set(observed) - set(allowed_differences)
    if undeclared:
        raise ValueError(
            "search configs differ outside the predeclared comparison dose: "
            f"{sorted(undeclared)}"
        )
    return observed


def _sorted_json_values(values: Any) -> list[Any]:
    if not isinstance(values, (list, tuple)):
        return []
    return sorted(values, key=_canonical_bytes)


def _canonical_snapshot_nodes(raw_nodes: Any) -> list[dict[str, Any]]:
    """Project shared nodes onto their unique state-semantic fields.

    The Rust serializer visits map tiles and keeps one arbitrary representative
    ``tile_coordinate``/``direction`` for a node shared by several tiles.  That
    representative can differ across same-seed constructions; node id,
    building, and owner color are the actual game state.
    """
    if isinstance(raw_nodes, dict):
        records = list(raw_nodes.values())
    elif isinstance(raw_nodes, (list, tuple)):
        records = list(raw_nodes)
    else:
        return []
    projected = []
    for raw in records:
        if not isinstance(raw, dict):
            continue
        projected.append(
            {
                "id": int(raw["id"]),
                "building": raw.get("building"),
                "color": raw.get("color"),
            }
        )
    return _sorted_json_values(projected)


def _canonical_snapshot_edges(raw_edges: Any) -> list[dict[str, Any]]:
    """Project shared edges onto sorted endpoint id plus owner color."""
    if isinstance(raw_edges, dict):
        records = list(raw_edges.values())
    elif isinstance(raw_edges, (list, tuple)):
        records = list(raw_edges)
    else:
        return []
    projected = []
    for raw in records:
        if not isinstance(raw, dict):
            continue
        raw_id = raw.get("id")
        edge_id: Any
        if isinstance(raw_id, (list, tuple)):
            edge_id = sorted(int(node) for node in raw_id)
        else:
            edge_id = raw_id
        projected.append({"id": edge_id, "color": raw.get("color")})
    return _sorted_json_values(projected)


def _canonicalize_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize Rust snapshot collections whose order is not semantic.

    ``catanatron_rs.json_snapshot()`` is documented to emit map-backed tiles
    and edges in non-canonical order across two same-seed constructions. It
    also chooses arbitrary redundant tile/direction representatives for shared
    nodes and edges. Preserve every order-sensitive sequence (colors,
    player_state, action_records, coordinates/action tuple fields), project
    nodes/edges to their actual state fields, and sort only explicitly
    unordered top-level collections.
    """
    canonical = dict(snapshot)
    for field in ("tiles", "current_playable_actions", "bot_colors"):
        if field in canonical:
            canonical[field] = _sorted_json_values(canonical[field])
    if "nodes" in canonical:
        canonical["nodes"] = _canonical_snapshot_nodes(canonical["nodes"])
    if "edges" in canonical:
        canonical["edges"] = _canonical_snapshot_edges(canonical["edges"])

    # Values in adjacent_tiles are neighbor sets.  Preserve each coordinate's
    # component order while sorting only the outer neighbor collection.
    adjacent = canonical.get("adjacent_tiles")
    if isinstance(adjacent, dict):
        canonical["adjacent_tiles"] = {
            key: _sorted_json_values(value) if isinstance(value, (list, tuple)) else value
            for key, value in adjacent.items()
        }
    return canonical


def _root_material(game: Any) -> dict[str, Any]:
    raw_snapshot = json.loads(game.json_snapshot())
    if not isinstance(raw_snapshot, dict):
        raise ValueError("Rust game json_snapshot() must decode to an object")
    snapshot = _canonicalize_snapshot(raw_snapshot)
    legal = sorted(
        int(action)
        for action in game.playable_action_indices(list(COLORS), None)
    )
    return {
        "snapshot": snapshot,
        "legal_action_ids": legal,
        "current_color": str(game.current_color()),
    }


def _root_hash(material: dict[str, Any]) -> str:
    return content_sha256(material)


def _select_raw_action(evaluator: Any, game: Any, legal: tuple[int, ...]) -> int:
    if len(legal) == 1:
        return int(legal[0])
    priors, *_rest = evaluator.evaluate(
        game,
        legal,
        root_color=str(game.current_color()),
        colors=COLORS,
    )
    return int(max(legal, key=lambda action: (float(priors[action]), -int(action))))


def seal_root_panel(panel_without_hash: dict[str, Any]) -> dict[str, Any]:
    if "panel_content_sha256" in panel_without_hash:
        raise ValueError("panel must be unsealed before seal_root_panel")
    sealed = dict(panel_without_hash)
    sealed["panel_content_sha256"] = content_sha256(panel_without_hash)
    return sealed


def validate_root_panel_payload(
    panel: dict[str, Any],
    *,
    checkpoint_sha256: str,
    evaluator_config_sha256: str,
) -> None:
    if panel.get("schema_version") != PANEL_SCHEMA:
        raise ValueError(f"root panel schema must be {PANEL_SCHEMA!r}")
    recorded_hash = panel.get("panel_content_sha256")
    if not isinstance(recorded_hash, str):
        raise ValueError("root panel is missing panel_content_sha256")
    unhashed = dict(panel)
    unhashed.pop("panel_content_sha256", None)
    actual_hash = content_sha256(unhashed)
    if recorded_hash != actual_hash:
        raise ValueError(
            f"root panel content hash mismatch: recorded={recorded_hash}, actual={actual_hash}"
        )
    provenance = panel.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("root panel provenance must be an object")
    if provenance.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("root panel checkpoint hash is incompatible with requested checkpoint")
    if provenance.get("evaluator_config_sha256") != evaluator_config_sha256:
        raise ValueError("root panel evaluator config hash is incompatible with this run")
    if tuple(provenance.get("colors", ())) != COLORS:
        raise ValueError("root panel colors are incompatible with this probe")
    if int(provenance.get("chance_seed_xor", -1)) != CHANCE_SEED_XOR:
        raise ValueError("root panel chance-seed reconstruction protocol is incompatible")
    if provenance.get("snapshot_canonicalization") != SNAPSHOT_CANONICALIZATION:
        raise ValueError("root panel snapshot canonicalization protocol is incompatible")
    roots = panel.get("roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("root panel must contain at least one root")
    if int(panel.get("root_count", -1)) != len(roots):
        raise ValueError("root panel root_count does not match roots")
    actual_wide_count = sum(
        1 for root in roots if len(root.get("legal_action_ids", ())) >= 40
    )
    if int(panel.get("wide_ge_40_count", -1)) != actual_wide_count:
        raise ValueError("root panel wide_ge_40_count does not match roots")
    expected_indices = list(range(len(roots)))
    if [root.get("root_index") for root in roots] != expected_indices:
        raise ValueError("root panel indices must be consecutive and ordered")
    root_hashes: list[str] = []
    for root in roots:
        prefix = root.get("action_prefix")
        if not isinstance(prefix, list):
            raise ValueError("each root must contain an action_prefix list")
        if int(root.get("decision_index", -1)) != len(prefix):
            raise ValueError("root decision_index must equal action_prefix length")
        material = {
            "snapshot": root.get("snapshot"),
            "legal_action_ids": root.get("legal_action_ids"),
            "current_color": root.get("current_color"),
        }
        if not isinstance(material["snapshot"], dict) or material[
            "snapshot"
        ] != _canonicalize_snapshot(material["snapshot"]):
            raise ValueError(
                f"root {root.get('root_index')} snapshot is not canonical"
            )
        legal_action_ids = material["legal_action_ids"]
        if (
            not isinstance(legal_action_ids, list)
            or legal_action_ids != sorted(set(int(action) for action in legal_action_ids))
        ):
            raise ValueError(
                f"root {root.get('root_index')} legal_action_ids must be sorted and unique"
            )
        if root.get("root_sha256") != _root_hash(material):
            raise ValueError(f"root {root.get('root_index')} content hash mismatch")
        legal_width = len(material["legal_action_ids"] or ())
        if int(root.get("legal_width", -1)) != legal_width:
            raise ValueError(f"root {root.get('root_index')} legal_width mismatch")
        if root.get("legal_width_bucket") != _legal_width_bucket(legal_width):
            raise ValueError(
                f"root {root.get('root_index')} legal_width_bucket mismatch"
            )
        if bool(root.get("wide_ge_40")) != bool(legal_width >= 40):
            raise ValueError(f"root {root.get('root_index')} wide_ge_40 mismatch")
        raw_phase = str((material["snapshot"] or {}).get("current_prompt", ""))
        if root.get("phase_raw") != raw_phase or root.get("phase") != _phase_label(
            raw_phase
        ):
            raise ValueError(f"root {root.get('root_index')} phase metadata mismatch")
        root_hashes.append(str(root["root_sha256"]))
    if len(set(root_hashes)) != len(root_hashes):
        raise ValueError("root panel contains duplicate root hashes")


def build_root_panel(
    evaluator: Any,
    *,
    checkpoint_sha256: str,
    evaluator_config_sha256: str,
    n_roots: int,
    decisions_per_game: tuple[int, ...],
    base_seed: int,
    min_legal_actions: int,
) -> dict[str, Any]:
    if n_roots <= 0:
        raise ValueError("n_roots must be positive")
    if not decisions_per_game or any(index < 0 for index in decisions_per_game):
        raise ValueError("decisions_per_game must contain non-negative indices")
    if min_legal_actions < 1:
        raise ValueError("min_legal_actions must be positive")

    catanatron_rs = _require_rust_module()
    targets = set(int(index) for index in decisions_per_game)
    max_target = max(targets)
    roots: list[dict[str, Any]] = []
    game_index = 0
    max_games = max(100, n_roots * 20)
    while len(roots) < n_roots and game_index < max_games:
        game_seed = int(base_seed) + game_index
        game = catanatron_rs.Game.simple(list(COLORS), seed=game_seed)
        chance_rng = random.Random(game_seed ^ CHANCE_SEED_XOR)
        action_prefix: list[int] = []
        decision_index = 0
        while decision_index <= max_target:
            if game.winning_color() is not None:
                break
            legal = tuple(
                int(action)
                for action in game.playable_action_indices(list(COLORS), None)
            )
            if not legal:
                break
            if decision_index in targets and len(legal) >= min_legal_actions:
                material = _root_material(game)
                raw_phase = str(material["snapshot"].get("current_prompt", ""))
                roots.append(
                    {
                        "root_index": len(roots),
                        "game_seed": game_seed,
                        "decision_index": decision_index,
                        "action_prefix": list(action_prefix),
                        **material,
                        "legal_width": len(material["legal_action_ids"]),
                        "legal_width_bucket": _legal_width_bucket(
                            len(material["legal_action_ids"])
                        ),
                        "wide_ge_40": len(material["legal_action_ids"]) >= 40,
                        "phase_raw": raw_phase,
                        "phase": _phase_label(raw_phase),
                        "root_sha256": _root_hash(material),
                    }
                )
                if len(roots) >= n_roots:
                    break
            selected = _select_raw_action(evaluator, game, legal)
            action_prefix.append(int(selected))
            game = _apply_selected_action(
                game,
                selected,
                colors=COLORS,
                rng=chance_rng,
                correct_rust_chance_spectra=True,
            )
            decision_index += 1
        game_index += 1
    if len(roots) != n_roots:
        raise RuntimeError(
            f"collected {len(roots)}/{n_roots} roots after {game_index} games"
        )
    panel = {
        "schema_version": PANEL_SCHEMA,
        "provenance": {
            "checkpoint_sha256": checkpoint_sha256,
            "evaluator_config_sha256": evaluator_config_sha256,
            "colors": list(COLORS),
            "raw_policy": "checkpoint_argmax_low_action_id_tiebreak",
            "chance_resolution": "_apply_selected_action_corrected_spectra",
            "chance_seed_xor": CHANCE_SEED_XOR,
            "snapshot_canonicalization": SNAPSHOT_CANONICALIZATION,
            "root_base_seed": int(base_seed),
            "decisions_per_game": list(sorted(targets)),
            "min_legal_actions": int(min_legal_actions),
        },
        "root_count": len(roots),
        "wide_ge_40_count": sum(1 for root in roots if root["wide_ge_40"]),
        "roots": roots,
    }
    return seal_root_panel(panel)


def reconstruct_roots(panel: dict[str, Any]) -> list[Any]:
    """Reconstruct and byte-semantically verify every persisted root."""
    catanatron_rs = _require_rust_module()
    states: list[Any] = []
    for record in panel["roots"]:
        game_seed = int(record["game_seed"])
        game = catanatron_rs.Game.simple(list(COLORS), seed=game_seed)
        chance_rng = random.Random(game_seed ^ CHANCE_SEED_XOR)
        for decision_index, action in enumerate(record["action_prefix"]):
            legal = tuple(
                int(item)
                for item in game.playable_action_indices(list(COLORS), None)
            )
            if int(action) not in legal:
                raise ValueError(
                    f"root {record['root_index']} reconstruction drift at decision "
                    f"{decision_index}: stored action {action} is not legal"
                )
            game = _apply_selected_action(
                game,
                int(action),
                colors=COLORS,
                rng=chance_rng,
                correct_rust_chance_spectra=True,
            )
        actual = _root_material(game)
        expected = {
            "snapshot": record["snapshot"],
            "legal_action_ids": record["legal_action_ids"],
            "current_color": record["current_color"],
        }
        if actual != expected or _root_hash(actual) != record["root_sha256"]:
            raise ValueError(
                f"root {record['root_index']} reconstruction hash mismatch; "
                "engine/transcript provenance is incompatible"
            )
        states.append(game)
    return states


def build_seed_manifests(
    *, n_roots: int, repeats: int, base_a: int, base_b: int, name_a: str, name_b: str
) -> dict[str, dict[str, Any]]:
    if n_roots <= 0:
        raise ValueError("n_roots must be positive")
    if repeats < 2:
        raise ValueError("repeats must be at least 2")

    def _role(base: int) -> list[list[int]]:
        return [
            [int(base) + root_index * repeats + repeat for repeat in range(repeats)]
            for root_index in range(n_roots)
        ]

    seeds_a = _role(base_a)
    seeds_b = _role(base_b)
    flat_a = {seed for row in seeds_a for seed in row}
    flat_b = {seed for row in seeds_b for seed in row}
    if flat_a & flat_b:
        raise ValueError("role A and role B search-seed sets must be disjoint")
    return {
        name_a: {
            "base_seed": int(base_a),
            "seeds_by_root": seeds_a,
            "seed_set_sha256": content_sha256(sorted(flat_a)),
        },
        name_b: {
            "base_seed": int(base_b),
            "seeds_by_root": seeds_b,
            "seed_set_sha256": content_sha256(sorted(flat_b)),
        },
    }


def _run_one_search(
    state: Any,
    *,
    evaluator: CountingEvaluator,
    spec: dict[str, Any],
    search_seed: int,
) -> dict[str, Any]:
    config = _make_search_config(spec, seed=search_seed)
    mcts = GumbelChanceMCTS(config, evaluator)
    before = evaluator.snapshot()
    started = time.perf_counter()
    result = mcts.search(state.copy(), force_full=True)
    wall_sec = time.perf_counter() - started
    eval_counts = CountingEvaluator.delta(before, evaluator.snapshot())
    improved = _normalize_policy(
        {int(action): float(value) for action, value in result.improved_policy.items()}
    )
    prior = _normalize_policy(
        {int(action): float(value) for action, value in result.priors.items()}
    )
    q_values = sorted(
        (float(value) for value in result.completed_q_values.values()), reverse=True
    )

    def _entropy(policy: dict[int, float]) -> float:
        return -sum(prob * math.log(prob) for prob in policy.values() if prob > 0.0)

    return {
        "search_seed": int(search_seed),
        "selected_action": int(result.selected_action),
        "improved_policy": improved,
        "prior_policy": prior,
        # Target reliability diagnostics.  These make the opening-road failure
        # visible directly in the immutable replay report: microscopic raw Q
        # margins should not coexist with a near-one-hot improved target.
        "target_top_probability": max(improved.values()),
        "target_entropy": _entropy(improved),
        "prior_top_probability": max(prior.values()),
        "prior_entropy": _entropy(prior),
        "target_prior_js": jensen_shannon_divergence(improved, prior),
        "completed_q_range": (
            q_values[0] - q_values[-1] if len(q_values) >= 2 else 0.0
        ),
        "completed_q_top_margin": (
            q_values[0] - q_values[1] if len(q_values) >= 2 else 0.0
        ),
        "simulations_used": int(result.simulations_used),
        "wall_sec": float(wall_sec),
        **eval_counts,
    }


def _aggregate_role(root_records: list[dict[str, Any]], role_name: str) -> dict[str, Any]:
    runs = [run for root in root_records for run in root["roles"][role_name]["runs"]]
    pairwise = [
        pair
        for root in root_records
        for pair in root["roles"][role_name]["stability"]["pairwise"]
    ]
    simulations = sum(int(run["simulations_used"]) for run in runs)
    wall = sum(float(run["wall_sec"]) for run in runs)
    logical_evals = sum(int(run["logical_leaf_evaluations"]) for run in runs)
    orientation_evals = sum(
        int(run["orientation_evaluation_rows"]) for run in runs
    )
    evaluator_calls = sum(int(run["evaluator_method_calls"]) for run in runs)
    js_values = [float(pair["js_divergence"]) for pair in pairwise]
    agreements = [1.0 if pair["top1_agreement"] else 0.0 for pair in pairwise]

    def _metric_summary(name: str) -> dict[str, float | None]:
        values = [float(run[name]) for run in runs if run.get(name) is not None]
        return {
            "mean": _mean(values),
            "median": _median(values),
            "max": max(values) if values else None,
        }

    return {
        "roots": len(root_records),
        "search_runs": len(runs),
        "cross_seed_pair_count": len(pairwise),
        "simulations_used": simulations,
        "simulations_per_search": simulations / len(runs) if runs else None,
        "wall_sec": wall,
        "wall_sec_per_search": wall / len(runs) if runs else None,
        "logical_leaf_evaluations": logical_evals,
        "logical_leaf_evaluations_per_search": (
            logical_evals / len(runs) if runs else None
        ),
        "orientation_evaluation_rows": orientation_evals,
        "orientation_evaluation_rows_per_search": (
            orientation_evals / len(runs) if runs else None
        ),
        "evaluator_method_calls": evaluator_calls,
        "evaluator_method_calls_per_search": (
            evaluator_calls / len(runs) if runs else None
        ),
        "cross_seed_js_mean": _mean(js_values),
        "cross_seed_js_median": _median(js_values),
        "cross_seed_js_max": max(js_values) if js_values else None,
        "top1_pair_agreement": _mean(agreements),
        "target_top_probability": _metric_summary("target_top_probability"),
        "target_entropy": _metric_summary("target_entropy"),
        "prior_top_probability": _metric_summary("prior_top_probability"),
        "prior_entropy": _metric_summary("prior_entropy"),
        "target_prior_js": _metric_summary("target_prior_js"),
        "completed_q_range": _metric_summary("completed_q_range"),
        "completed_q_top_margin": _metric_summary("completed_q_top_margin"),
    }


def _safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator is None or float(denominator) <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _comparison_slice(
    root_records: list[dict[str, Any]], name_a: str, name_b: str
) -> dict[str, Any]:
    a = _aggregate_role(root_records, name_a)
    b = _aggregate_role(root_records, name_b)
    js_a = a["cross_seed_js_mean"]
    js_b = b["cross_seed_js_mean"]
    agreement_a = a["top1_pair_agreement"]
    agreement_b = b["top1_pair_agreement"]

    def _mean_delta(metric: str) -> float | None:
        first = a[metric]["mean"]
        second = b[metric]["mean"]
        if first is None or second is None:
            return None
        return float(second) - float(first)

    return {
        "roots": len(root_records),
        "by_role": {name_a: a, name_b: b},
        "comparison": {
            "role_b_over_role_a_simulations_ratio": _safe_ratio(
                b["simulations_used"], a["simulations_used"]
            ),
            "role_b_over_role_a_wall_ratio": _safe_ratio(b["wall_sec"], a["wall_sec"]),
            "role_b_over_role_a_logical_evaluations_ratio": _safe_ratio(
                b["logical_leaf_evaluations"], a["logical_leaf_evaluations"]
            ),
            "role_b_over_role_a_orientation_evaluations_ratio": _safe_ratio(
                b["orientation_evaluation_rows"], a["orientation_evaluation_rows"]
            ),
            "role_b_minus_role_a_cross_seed_js": (
                float(js_b) - float(js_a)
                if js_a is not None and js_b is not None
                else None
            ),
            "role_b_relative_js_reduction": (
                (float(js_a) - float(js_b)) / float(js_a)
                if js_a is not None and js_b is not None and float(js_a) > 0.0
                else None
            ),
            "role_b_minus_role_a_top1_agreement": (
                float(agreement_b) - float(agreement_a)
                if agreement_a is not None and agreement_b is not None
                else None
            ),
            "role_b_minus_role_a_target_top_probability": _mean_delta(
                "target_top_probability"
            ),
            "role_b_minus_role_a_target_entropy": _mean_delta("target_entropy"),
            "role_b_minus_role_a_target_prior_js": _mean_delta("target_prior_js"),
            "role_b_minus_role_a_completed_q_range": _mean_delta(
                "completed_q_range"
            ),
            "role_b_minus_role_a_completed_q_top_margin": _mean_delta(
                "completed_q_top_margin"
            ),
        },
    }


def aggregate_report_slices(
    root_records: list[dict[str, Any]], name_a: str, name_b: str
) -> dict[str, Any]:
    def _group(field: str) -> dict[str, Any]:
        keys = sorted({str(root[field]) for root in root_records})
        return {
            key: _comparison_slice(
                [root for root in root_records if str(root[field]) == key],
                name_a,
                name_b,
            )
            for key in keys
        }

    wide = [root for root in root_records if int(root["legal_width"]) >= 40]
    return {
        "global": _comparison_slice(root_records, name_a, name_b),
        "wide_ge_40": _comparison_slice(wide, name_a, name_b),
        "by_phase": _group("phase"),
        "by_raw_phase": _group("phase_raw"),
        "by_legal_width_bucket": _group("legal_width_bucket"),
    }


def run_fixed_root_comparison(
    states: list[Any],
    panel: dict[str, Any],
    *,
    evaluator: CountingEvaluator,
    spec_a: dict[str, Any],
    spec_b: dict[str, Any],
    seed_manifests: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(states) != len(panel["roots"]):
        raise ValueError("reconstructed state count does not match root panel")
    name_a = str(spec_a["name"])
    name_b = str(spec_b["name"])
    if name_a == name_b:
        raise ValueError("search config names must be distinct")

    root_records: list[dict[str, Any]] = []
    for root_index, (state, root) in enumerate(zip(states, panel["roots"])):
        runs_by_role: dict[str, list[dict[str, Any]]] = {name_a: [], name_b: []}
        repeats = len(seed_manifests[name_a]["seeds_by_root"][root_index])
        for repeat in range(repeats):
            ordered = (
                ((spec_a, name_a), (spec_b, name_b))
                if (root_index + repeat) % 2 == 0
                else ((spec_b, name_b), (spec_a, name_a))
            )
            for spec, name in ordered:
                seed = seed_manifests[name]["seeds_by_root"][root_index][repeat]
                runs_by_role[name].append(
                    _run_one_search(
                        state,
                        evaluator=evaluator,
                        spec=spec,
                        search_seed=int(seed),
                    )
                )

        roles: dict[str, Any] = {}
        expected_support = set(int(action) for action in root["legal_action_ids"])
        for name in (name_a, name_b):
            for run in runs_by_role[name]:
                if set(run["improved_policy"]) != expected_support:
                    raise ValueError(
                        f"root {root_index} role {name} returned a policy support "
                        "incompatible with the stored legal-action set"
                    )
            roles[name] = {
                "runs": runs_by_role[name],
                "stability": summarize_cross_seed_runs(runs_by_role[name]),
            }
        root_records.append(
            {
                "root_index": root_index,
                "root_sha256": root["root_sha256"],
                "game_seed": int(root["game_seed"]),
                "decision_index": int(root["decision_index"]),
                "legal_width": int(root["legal_width"]),
                "legal_width_bucket": str(root["legal_width_bucket"]),
                "wide_ge_40": bool(int(root["legal_width"]) >= 40),
                "phase": str(root["phase"]),
                "phase_raw": str(root["phase_raw"]),
                "roles": roles,
            }
        )
    return root_records, aggregate_report_slices(root_records, name_a, name_b)


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("decision indices must be non-negative")
    return values


def _parse_name_set(text: str) -> set[str]:
    values = {item.strip() for item in text.split(",") if item.strip()}
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated field name")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two named Gumbel search configs on an immutable fixed-root "
            "panel with cross-seed JS/top-1 stability and attributable cost."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evaluator-config", required=True)
    parser.add_argument("--config-a", required=True)
    parser.add_argument("--config-b", required=True)
    parser.add_argument(
        "--allowed-search-config-differences",
        type=_parse_name_set,
        required=True,
        help=(
            "Comma-separated GumbelChanceMCTSConfig fields that the two arms are "
            "allowed to differ on; any undeclared operator drift fails closed."
        ),
    )
    parser.add_argument("--root-panel", required=True)
    parser.add_argument(
        "--create-root-panel",
        action="store_true",
        default=False,
        help="Explicitly create --root-panel before running; refuses to overwrite.",
    )
    parser.add_argument("--n-roots", type=int, default=40)
    parser.add_argument(
        "--decisions-per-game",
        type=_parse_int_tuple,
        default=(0, 1, 2, 3, 20, 50, 80, 110),
    )
    parser.add_argument("--root-base-seed", type=int, default=DEFAULT_ROOT_BASE_SEED)
    parser.add_argument("--min-legal-actions", type=int, default=2)
    parser.add_argument(
        "--min-wide-roots",
        type=int,
        default=0,
        help="Fail before search unless the panel contains this many >=40-action roots.",
    )
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument(
        "--search-seed-base-a", type=int, default=DEFAULT_SEARCH_SEED_BASE_A
    )
    parser.add_argument(
        "--search-seed-base-b", type=int, default=DEFAULT_SEARCH_SEED_BASE_B
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--max-wait-ms", type=float, default=3.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    checkpoint_hash = file_sha256(checkpoint)
    spec_a = load_search_spec(args.config_a)
    spec_b = load_search_spec(args.config_b)
    if spec_a["name"] == spec_b["name"]:
        parser.error("--config-a and --config-b must have distinct names")
    config_differences = validate_search_comparison(
        spec_a,
        spec_b,
        allowed_differences=set(args.allowed_search_config_differences),
    )
    evaluator_spec = load_evaluator_spec(args.evaluator_config)
    evaluator_config = EntityGraphRustEvaluatorConfig(
        **evaluator_spec["effective_evaluator_config"]
    )
    evaluator_runtime = {
        "device": str(args.device),
        # BatchedEntityGraphRustEvaluator clamps these values internally. Use
        # and report those exact effective values so timing artifacts remain
        # interpretable even if a caller passes zero or a negative value.
        "max_batch_size": max(1, int(args.max_batch_size)),
        "max_wait_ms": max(0.0, float(args.max_wait_ms)),
    }

    root_panel_path = Path(args.root_panel)
    protected_inputs = {
        checkpoint.resolve(),
        Path(args.evaluator_config).resolve(),
        Path(args.config_a).resolve(),
        Path(args.config_b).resolve(),
        root_panel_path.resolve(),
    }
    if Path(args.out).resolve() in protected_inputs:
        parser.error("--out must not overwrite a checkpoint/config/root-panel input")
    if args.create_root_panel and root_panel_path.exists():
        parser.error(
            f"refusing to overwrite existing root panel: {root_panel_path}; "
            "choose a new path or omit --create-root-panel"
        )
    if not args.create_root_panel and not root_panel_path.is_file():
        parser.error(
            f"root panel does not exist: {root_panel_path}; pass --create-root-panel explicitly"
        )

    inner = BatchedEntityGraphRustEvaluator.from_checkpoint(
        str(checkpoint),
        device=evaluator_runtime["device"],
        config=evaluator_config,
        max_batch_size=evaluator_runtime["max_batch_size"],
        max_wait_ms=evaluator_runtime["max_wait_ms"],
    )
    try:
        if args.create_root_panel:
            panel = build_root_panel(
                inner,
                checkpoint_sha256=checkpoint_hash,
                evaluator_config_sha256=evaluator_spec[
                    "effective_evaluator_config_sha256"
                ],
                n_roots=int(args.n_roots),
                decisions_per_game=tuple(args.decisions_per_game),
                base_seed=int(args.root_base_seed),
                min_legal_actions=int(args.min_legal_actions),
            )
            write_json(root_panel_path, panel)
        else:
            panel = _load_json_object(root_panel_path)

        locked_input_file_hashes = {
            str(checkpoint.resolve()): checkpoint_hash,
            str(Path(args.evaluator_config).resolve()): evaluator_spec[
                "source_file_sha256"
            ],
            str(Path(args.config_a).resolve()): spec_a["source_file_sha256"],
            str(Path(args.config_b).resolve()): spec_b["source_file_sha256"],
            str(root_panel_path.resolve()): file_sha256(root_panel_path),
        }

        validate_root_panel_payload(
            panel,
            checkpoint_sha256=checkpoint_hash,
            evaluator_config_sha256=evaluator_spec[
                "effective_evaluator_config_sha256"
            ],
        )
        if int(panel.get("wide_ge_40_count", 0)) < int(args.min_wide_roots):
            raise ValueError(
                f"root panel has {panel.get('wide_ge_40_count', 0)} >=40-action roots, "
                f"below --min-wide-roots={args.min_wide_roots}"
            )
        states = reconstruct_roots(panel)
        seeds = build_seed_manifests(
            n_roots=len(states),
            repeats=int(args.repeats),
            base_a=int(args.search_seed_base_a),
            base_b=int(args.search_seed_base_b),
            name_a=str(spec_a["name"]),
            name_b=str(spec_b["name"]),
        )
        counting = CountingEvaluator(inner)
        started = time.perf_counter()
        per_root, slices = run_fixed_root_comparison(
            states,
            panel,
            evaluator=counting,
            spec_a=spec_a,
            spec_b=spec_b,
            seed_manifests=seeds,
        )
        elapsed = time.perf_counter() - started
    finally:
        inner.close()

    verify_locked_files(locked_input_file_hashes)

    report = {
        "schema_version": REPORT_SCHEMA,
        "measurement": "fixed_root_search_stability_cost",
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": checkpoint_hash,
        },
        "root_panel": {
            "path": str(root_panel_path.resolve()),
            "file_sha256": file_sha256(root_panel_path),
            "content_sha256": panel["panel_content_sha256"],
            "root_count": len(panel["roots"]),
            "wide_ge_40_count": int(panel.get("wide_ge_40_count", 0)),
            "root_sha256s": [root["root_sha256"] for root in panel["roots"]],
        },
        "evaluator": evaluator_spec,
        "evaluator_runtime": evaluator_runtime,
        "roles": {str(spec_a["name"]): spec_a, str(spec_b["name"]): spec_b},
        "search_config_differences": config_differences,
        "allowed_search_config_differences": sorted(
            args.allowed_search_config_differences
        ),
        "search_seed_manifests": seeds,
        "locked_input_file_hashes": locked_input_file_hashes,
        "protocol": {
            "force_full": True,
            "repeats_per_root_per_role": int(args.repeats),
            "role_order": "alternating_by_root_plus_repeat",
            "cache_size_required": 0,
            "js_log_base": "e_nats",
            "top1_definition": (
                "the actual SearchResult.selected_action emitted by the named "
                "operator at temperature=0; on the normal policy-choice path "
                "this includes the production visit/prior tie-breaks"
            ),
            "policy_argmax_definition": (
                "argmax(improved_policy), with lower action id breaking exact ties; "
                "reported as a non-binding diagnostic"
            ),
            "wide_slice": "legal_width>=40",
            "evaluation_count_definition": (
                "logical_leaf_evaluations counts requested leaf states; "
                "orientation_evaluation_rows expands each D6 root to 12; "
                "evaluator_method_calls counts evaluate/evaluate_many/D6 method invocations"
            ),
        },
        "probe_elapsed_sec": elapsed,
        "slices": slices,
        "per_root": per_root,
    }
    report["report_content_sha256"] = content_sha256(report)
    write_json(args.out, report)
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve()),
                "report_content_sha256": report["report_content_sha256"],
                "global": slices["global"],
                "wide_ge_40": slices["wide_ge_40"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
