#!/usr/bin/env python3
"""Render fixed-root search evidence as a human-readable Catan exam.

``fixed_root_search_stability.py`` deliberately stores exact machine evidence:
action ids, policies, completed-Q margins, and cross-seed stability.  Those
fields are suitable for causal aggregation but awkward for a Catan player to
inspect.  This tool joins the sealed report back to its reconstructed game
roots and explains the actual choices:

* settlement production, resource mix, and port access;
* road endpoints and the production reachable from them;
* robber destination, victim, and settlement/city-weighted blocked pips;
* the acting player's private hand and public opponent summary;
* raw-prior and repeated-search choices, target confidence, Q margins, and
  cross-seed disagreement.

Each exam embeds the sealed replay transcript and binds the checkpoint,
evaluator, and named search-operator hashes.  Programmatic callers may attach
diagnostic forced-action outcomes through ``run_paired_counterfactuals``, which
enforces one common seed manifest across every compared action.

The output is diagnostic evidence, not a promotion verdict.  Reliability
labels are explicitly heuristic and retain the underlying measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
_LOCAL_SRC = _REPO_ROOT / "src"
for path in (_TOOLS_DIR, _LOCAL_SRC):
    try:
        sys.path.remove(str(path))
    except ValueError:
        pass
    sys.path.insert(0, str(path))

from catan_zero.search.neural_rust_mcts import (  # noqa: E402
    _base_ports_by_id,
    _base_tile_topology,
)
from factory_common import write_json  # noqa: E402
from fixed_root_search_stability import (  # noqa: E402
    REPORT_SCHEMA,
    content_sha256,
    validate_root_panel_payload,
)

EXAM_SCHEMA = "strategic-root-exam-v1"
PAIRED_COUNTERFACTUAL_PROTOCOL = "paired-common-seed-v1"

CounterfactualHook = Callable[[Any, int, int], Mapping[str, Any]]


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _dice_pips(number: Any) -> int:
    try:
        parsed = int(number)
    except (TypeError, ValueError):
        return 0
    return max(0, 6 - abs(parsed - 7)) if parsed != 7 else 0


def _coordinate(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return int(value[0]), int(value[1]), int(value[2])
    except (TypeError, ValueError):
        return None


def _resource_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).lower()
    return None if text in {"", "none", "null"} else text


def node_production(snapshot: Mapping[str, Any]) -> dict[int, dict[str, int]]:
    """Return exact dice-pip production by node and resource."""

    topology = _base_tile_topology()
    result: dict[int, dict[str, int]] = {}
    for raw in snapshot.get("tiles", ()):
        if not isinstance(raw, Mapping):
            continue
        tile = raw.get("tile")
        if not isinstance(tile, Mapping) or str(tile.get("type")) != "RESOURCE_TILE":
            continue
        resource = _resource_name(tile.get("resource"))
        pips = _dice_pips(tile.get("number"))
        coordinate = _coordinate(raw.get("coordinate"))
        local = topology.get(coordinate or ())
        if resource is None or pips <= 0 or not isinstance(local, Mapping):
            continue
        for node in dict(local.get("nodes", {})).values():
            node_id = int(node)
            by_resource = result.setdefault(node_id, {})
            by_resource[resource] = by_resource.get(resource, 0) + pips
    return result


def ports_by_node(snapshot: Mapping[str, Any]) -> dict[int, str]:
    """Join live port resources to the base topology's port nodes."""

    base = _base_ports_by_id()
    result: dict[int, str] = {}
    for raw in snapshot.get("tiles", ()):
        if not isinstance(raw, Mapping):
            continue
        tile = raw.get("tile")
        if not isinstance(tile, Mapping) or str(tile.get("type")) != "PORT":
            continue
        try:
            port_id = int(tile["id"])
        except (KeyError, TypeError, ValueError):
            continue
        port = base.get(port_id)
        if not isinstance(port, Mapping):
            continue
        resource = _resource_name(tile.get("resource")) or "3:1"
        for node in port.get("nodes", ()):
            result[int(node)] = resource
    return result


def _buildings(snapshot: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    raw_nodes = snapshot.get("nodes", ())
    values = raw_nodes.values() if isinstance(raw_nodes, Mapping) else raw_nodes
    for raw in values or ():
        if not isinstance(raw, Mapping) or raw.get("building") is None:
            continue
        result[int(raw["id"])] = {
            "color": str(raw.get("color", "")),
            "building": str(raw.get("building", "")),
        }
    return result


def _player_summary(snapshot: Mapping[str, Any], actor: str) -> dict[str, Any]:
    colors = [str(color) for color in snapshot.get("colors", ())]
    states = snapshot.get("player_state", ())
    state_by_color = {
        color: states[index]
        for index, color in enumerate(colors)
        if isinstance(states, (list, tuple))
        and index < len(states)
        and isinstance(states[index], Mapping)
    }
    summary: dict[str, Any] = {}
    for color in colors:
        state = state_by_color.get(color, {})
        private = color == actor
        resources = state.get("resources") if private else None
        dev_cards = state.get("dev_cards") if private else None
        summary[color] = {
            "actor": private,
            "public_vp": int(state.get("victory_points", 0) or 0),
            "resource_card_count": int(
                sum(dict(state.get("resources", {})).values())
                if isinstance(state.get("resources"), Mapping)
                else 0
            ),
            "development_card_count": int(
                sum(dict(state.get("dev_cards", {})).values())
                if isinstance(state.get("dev_cards"), Mapping)
                else 0
            ),
            "resources": dict(resources) if isinstance(resources, Mapping) else None,
            "development_cards": (
                dict(dev_cards) if isinstance(dev_cards, Mapping) else None
            ),
            "played_development_cards": dict(
                state.get("played_dev_cards", {})
                if isinstance(state.get("played_dev_cards"), Mapping)
                else {}
            ),
            "has_longest_road": bool(state.get("has_road", False)),
            "longest_road_length": int(state.get("longest_road_length", 0) or 0),
            "has_largest_army": bool(state.get("has_army", False)),
            "has_played_dev_this_turn": bool(
                state.get("has_played_development_card_in_turn", False)
            ),
            "has_rolled": bool(state.get("has_rolled", False)),
        }
    return summary


def _robber_context(
    snapshot: Mapping[str, Any],
    coordinate: tuple[int, int, int] | None,
    victim: Any,
) -> dict[str, Any]:
    topology = _base_tile_topology()
    local = topology.get(coordinate or (), {})
    adjacent_nodes = {int(node) for node in dict(local.get("nodes", {})).values()}
    buildings = _buildings(snapshot)
    blocked: dict[str, int] = {}
    for node in adjacent_nodes:
        building = buildings.get(node)
        if building is None:
            continue
        multiplier = 2 if building["building"] == "CITY" else 1
        blocked[building["color"]] = blocked.get(building["color"], 0) + multiplier
    tile_summary: dict[str, Any] = {}
    for raw in snapshot.get("tiles", ()):
        if (
            not isinstance(raw, Mapping)
            or _coordinate(raw.get("coordinate")) != coordinate
        ):
            continue
        tile = raw.get("tile")
        if isinstance(tile, Mapping):
            tile_summary = {
                "resource": _resource_name(tile.get("resource")),
                "number": int(tile.get("number", 0) or 0),
                "pips": _dice_pips(tile.get("number")),
            }
        break
    return {
        "coordinate": list(coordinate) if coordinate is not None else None,
        "victim": victim,
        "tile": tile_summary,
        "adjacent_building_units_by_color": blocked,
        "blocked_pip_units_by_color": {
            color: units * int(tile_summary.get("pips", 0))
            for color, units in blocked.items()
        },
    }


def describe_action(
    action_id: int,
    raw: Sequence[Any],
    *,
    snapshot: Mapping[str, Any],
    production: Mapping[int, Mapping[str, int]],
    ports: Mapping[int, str],
) -> dict[str, Any]:
    actor = str(raw[0]) if len(raw) > 0 else ""
    action_type = str(raw[1]) if len(raw) > 1 else ""
    value = raw[2] if len(raw) > 2 else None
    description: dict[str, Any] = {
        "action_id": int(action_id),
        "actor": actor,
        "action_type": action_type,
        "argument": value,
    }
    if action_type in {"BUILD_SETTLEMENT", "BUILD_CITY"}:
        node = int(value)
        resources = dict(production.get(node, {}))
        description["strategic_context"] = {
            "node": node,
            "total_pips": int(sum(resources.values())),
            "resource_pips": resources,
            "resource_diversity": len(resources),
            "port": ports.get(node),
        }
    elif action_type == "BUILD_ROAD" and isinstance(value, (list, tuple)):
        endpoints = [int(node) for node in value]
        endpoint_context = []
        for node in endpoints:
            resources = dict(production.get(node, {}))
            endpoint_context.append(
                {
                    "node": node,
                    "total_pips": int(sum(resources.values())),
                    "resource_pips": resources,
                    "port": ports.get(node),
                }
            )
        description["strategic_context"] = {"endpoints": endpoint_context}
    elif action_type == "MOVE_ROBBER":
        coordinate: tuple[int, int, int] | None = None
        victim = raw[3] if len(raw) > 3 else None
        if isinstance(value, (list, tuple)) and value:
            if isinstance(value[0], (list, tuple)):
                coordinate = _coordinate(value[0])
                if len(value) > 1:
                    victim = value[1]
            else:
                coordinate = _coordinate(value)
        description["strategic_context"] = _robber_context(snapshot, coordinate, victim)
    return description


def _mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else None


def _copy_state(state: Any) -> Any:
    copier = getattr(state, "copy", None)
    return copier() if callable(copier) else state


def run_paired_counterfactuals(
    state: Any,
    *,
    action_ids: Sequence[int],
    seeds: Sequence[int],
    hook: CounterfactualHook,
    action_descriptions: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Run forced actions under one ordered, common downstream seed manifest.

    The hook owns the downstream experiment (raw rollout, search rollout, or a
    deterministic test double).  This layer only enforces the causal pairing
    contract and records aligned outcome deltas.  The result remains diagnostic
    evidence and is never a training target.
    """

    actions = [int(action) for action in action_ids]
    paired_seeds = [int(seed) for seed in seeds]
    if len(actions) < 2 or len(actions) != len(set(actions)):
        raise ValueError("paired counterfactuals require at least two unique actions")
    if not paired_seeds or len(paired_seeds) != len(set(paired_seeds)):
        raise ValueError("paired counterfactual seeds must be non-empty and unique")
    if any(action not in action_descriptions for action in actions):
        raise ValueError("every counterfactual action requires a semantic description")

    action_records: list[dict[str, Any]] = []
    for action in actions:
        trials = []
        for seed in paired_seeds:
            result = dict(hook(_copy_state(state), action, seed))
            if "outcome" not in result:
                raise ValueError("counterfactual hook must return an outcome")
            outcome = float(result["outcome"])
            if not math.isfinite(outcome) or not -1.0 <= outcome <= 1.0:
                raise ValueError("counterfactual outcome must be finite and in [-1, 1]")
            trials.append({"seed": seed, **result, "outcome": outcome})
        outcomes = [float(trial["outcome"]) for trial in trials]
        action_records.append(
            {
                "action": dict(action_descriptions[action]),
                "trials": trials,
                "summary": {
                    "trials": len(trials),
                    "wins": sum(outcome > 0.0 for outcome in outcomes),
                    "losses": sum(outcome < 0.0 for outcome in outcomes),
                    "draws_or_truncations": sum(outcome == 0.0 for outcome in outcomes),
                    "mean_outcome": statistics.fmean(outcomes),
                },
            }
        )

    baseline = action_records[0]
    baseline_outcomes = [float(trial["outcome"]) for trial in baseline["trials"]]
    paired_deltas = []
    for challenger in action_records[1:]:
        challenger_outcomes = [
            float(trial["outcome"]) for trial in challenger["trials"]
        ]
        deltas = [
            challenger_outcome - baseline_outcome
            for baseline_outcome, challenger_outcome in zip(
                baseline_outcomes, challenger_outcomes, strict=True
            )
        ]
        paired_deltas.append(
            {
                "baseline_action": baseline["action"],
                "challenger_action": challenger["action"],
                "by_seed": [
                    {"seed": seed, "outcome_delta": delta}
                    for seed, delta in zip(paired_seeds, deltas, strict=True)
                ],
                "mean_outcome_delta": statistics.fmean(deltas),
            }
        )

    unsealed = {
        "protocol": PAIRED_COUNTERFACTUAL_PROTOCOL,
        "diagnostic_only": True,
        "training_target_eligible": False,
        "seeds": paired_seeds,
        "seed_manifest_sha256": content_sha256(paired_seeds),
        "baseline_action_id": actions[0],
        "actions": action_records,
        "paired_deltas": paired_deltas,
    }
    return {**unsealed, "content_sha256": content_sha256(unsealed)}


def _validate_paired_counterfactuals(
    evidence: Mapping[str, Any], *, legal_action_ids: set[int]
) -> None:
    if evidence.get("protocol") != PAIRED_COUNTERFACTUAL_PROTOCOL:
        raise ValueError("unknown paired counterfactual protocol")
    unsealed = dict(evidence)
    recorded_hash = unsealed.pop("content_sha256", None)
    if recorded_hash != content_sha256(unsealed):
        raise ValueError("paired counterfactual content hash mismatch")
    seeds = evidence.get("seeds")
    if not isinstance(seeds, list) or evidence.get(
        "seed_manifest_sha256"
    ) != content_sha256(seeds):
        raise ValueError("paired counterfactual seed manifest hash mismatch")
    for action in evidence.get("actions", ()):
        action_id = int(action["action"]["action_id"])
        if action_id not in legal_action_ids:
            raise ValueError("paired counterfactual action is not legal at the root")
        if [trial["seed"] for trial in action["trials"]] != seeds:
            raise ValueError("counterfactual trials are not seed-paired")


def _exam_provenance(
    panel: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, Any]:
    panel_provenance = panel.get("provenance")
    checkpoint = report.get("checkpoint")
    evaluator = report.get("evaluator")
    roles = report.get("roles")
    if not isinstance(panel_provenance, Mapping):
        raise ValueError("root panel provenance must be an object")
    if not isinstance(checkpoint, Mapping) or not isinstance(evaluator, Mapping):
        raise ValueError("fixed-root report model provenance is malformed")
    if not isinstance(roles, Mapping):
        raise ValueError("fixed-root report search operator provenance is malformed")

    checkpoint_hash = str(checkpoint.get("sha256", ""))
    evaluator_hash = str(evaluator.get("effective_evaluator_config_sha256", ""))
    if checkpoint_hash != str(panel_provenance.get("checkpoint_sha256", "")):
        raise ValueError("fixed-root report and panel checkpoint hashes differ")
    if evaluator_hash != str(panel_provenance.get("evaluator_config_sha256", "")):
        raise ValueError("fixed-root report and panel evaluator hashes differ")

    operator_hashes = {}
    for name, role in roles.items():
        if not isinstance(role, Mapping) or not role.get(
            "effective_search_config_sha256"
        ):
            raise ValueError(f"fixed-root search role {name!r} has no operator hash")
        operator_hashes[str(name)] = str(role["effective_search_config_sha256"])
    return {
        "root_panel_content_sha256": panel.get("panel_content_sha256"),
        "source_report_content_sha256": report.get("report_content_sha256"),
        "checkpoint_sha256": checkpoint_hash,
        "evaluator_operator_sha256": evaluator_hash,
        "search_operator_sha256_by_role": operator_hashes,
    }


def _role_exam(
    role: Mapping[str, Any],
    *,
    action_descriptions: Mapping[int, Mapping[str, Any]],
    top_k: int,
    js_warning: float,
    q_margin_warning: float,
) -> dict[str, Any]:
    runs = role.get("runs")
    stability = role.get("stability")
    if not isinstance(runs, list) or not runs or not isinstance(stability, Mapping):
        raise ValueError("fixed-root role evidence is malformed")

    def _policy_top(policy: Mapping[str, Any]) -> list[dict[str, Any]]:
        ranked = sorted(
            (
                (int(action), float(probability))
                for action, probability in policy.items()
            ),
            key=lambda item: (item[1], -item[0]),
            reverse=True,
        )[:top_k]
        return [
            {
                "probability": probability,
                "action": dict(action_descriptions[action]),
            }
            for action, probability in ranked
        ]

    selected = [dict(action_descriptions[int(run["selected_action"])]) for run in runs]
    js = float(stability.get("cross_seed_js_mean", 0.0) or 0.0)
    top1_agreement = float(stability.get("top1_pair_agreement", 0.0) or 0.0)
    margins = [float(run.get("completed_q_top_margin", 0.0) or 0.0) for run in runs]
    mean_margin = _mean(margins) or 0.0
    flags = []
    if top1_agreement < 1.0:
        flags.append("selected_action_changes_across_search_seeds")
    if js >= js_warning:
        flags.append("high_cross_seed_policy_disagreement")
    if mean_margin <= q_margin_warning:
        flags.append("top_completed_q_margin_near_noise_scale")
    return {
        "raw_prior_top": _policy_top(runs[0]["prior_policy"]),
        "search_target_top_by_repeat": [
            _policy_top(run["improved_policy"]) for run in runs
        ],
        "selected_actions": selected,
        "cross_seed_js_mean": js,
        "selected_action_pair_agreement": top1_agreement,
        "mean_completed_q_top_margin": mean_margin,
        "mean_target_prior_js": _mean(
            [float(run.get("target_prior_js", 0.0) or 0.0) for run in runs]
        ),
        "mean_target_top_probability": _mean(
            [float(run.get("target_top_probability", 0.0) or 0.0) for run in runs]
        ),
        "diagnostic_flags": flags,
        "diagnostic_only": True,
    }


def build_exam(
    panel: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    top_k: int = 5,
    js_warning: float = 0.1,
    q_margin_warning: float = 0.02,
    counterfactuals_by_root: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError(f"report schema must be {REPORT_SCHEMA!r}")
    roots = panel.get("roots")
    evidence = report.get("per_root")
    if not isinstance(roots, list) or not isinstance(evidence, list):
        raise ValueError("panel/report roots must be lists")
    if len(roots) != len(evidence):
        raise ValueError("panel and report root counts differ")
    report_panel = report.get("root_panel")
    if not isinstance(report_panel, Mapping) or report_panel.get(
        "content_sha256"
    ) != panel.get("panel_content_sha256"):
        raise ValueError("fixed-root report does not bind the supplied root panel")

    provenance = _exam_provenance(panel, report)
    counterfactuals = {
        int(index): value for index, value in (counterfactuals_by_root or {}).items()
    }
    root_exams = []
    for root, root_evidence in zip(roots, evidence, strict=True):
        if root_evidence.get("root_sha256") != root.get("root_sha256"):
            raise ValueError("fixed-root report and panel root hashes differ")
        snapshot = root["snapshot"]
        legal_ids = [int(action) for action in root["legal_action_ids"]]
        raw_actions = snapshot.get("current_playable_actions")
        if not isinstance(raw_actions, list):
            raise ValueError(
                "sealed root snapshot has no current_playable_actions evidence"
            )
        if len(legal_ids) != len(raw_actions):
            raise ValueError(
                "sealed action ids and action descriptions differ in length"
            )
        action_raw = {
            action: raw for action, raw in zip(legal_ids, raw_actions, strict=True)
        }
        production = node_production(snapshot)
        ports = ports_by_node(snapshot)
        descriptions = {
            action: describe_action(
                action,
                raw,
                snapshot=snapshot,
                production=production,
                ports=ports,
            )
            for action, raw in action_raw.items()
        }
        role_evidence = root_evidence.get("roles")
        if not isinstance(role_evidence, Mapping):
            raise ValueError("root role evidence must be an object")
        if set(str(name) for name in role_evidence) != set(
            provenance["search_operator_sha256_by_role"]
        ):
            raise ValueError("root role evidence and search operator hashes differ")
        actor = str(root.get("current_color", ""))
        root_counterfactuals = counterfactuals.get(int(root["root_index"]))
        if root_counterfactuals is not None:
            _validate_paired_counterfactuals(
                root_counterfactuals, legal_action_ids=set(legal_ids)
            )
        root_exams.append(
            {
                "root_index": int(root["root_index"]),
                "game_seed": int(root["game_seed"]),
                "decision_index": int(root["decision_index"]),
                "phase": root["phase"],
                "phase_raw": root["phase_raw"],
                "legal_width": int(root["legal_width"]),
                "actor": actor,
                "replay": {
                    "root_sha256": str(root["root_sha256"]),
                    "game_seed": int(root["game_seed"]),
                    "decision_index": int(root["decision_index"]),
                    "action_prefix": [int(action) for action in root["action_prefix"]],
                    "snapshot": snapshot,
                    "legal_action_ids": legal_ids,
                    "current_color": actor,
                },
                "players": _player_summary(snapshot, actor),
                "recent_public_actions": list(snapshot.get("action_records", ()))[-8:],
                "roles": {
                    str(name): _role_exam(
                        role,
                        action_descriptions=descriptions,
                        top_k=top_k,
                        js_warning=js_warning,
                        q_margin_warning=q_margin_warning,
                    )
                    for name, role in role_evidence.items()
                },
                "paired_counterfactuals": root_counterfactuals,
            }
        )

    unknown_counterfactual_roots = set(counterfactuals) - {
        int(root["root_index"]) for root in roots
    }
    if unknown_counterfactual_roots:
        raise ValueError(
            "paired counterfactual evidence references unknown roots: "
            f"{sorted(unknown_counterfactual_roots)}"
        )

    flag_counts: dict[str, int] = {}
    for root in root_exams:
        for role in root["roles"].values():
            for flag in role["diagnostic_flags"]:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
    unsealed = {
        "schema_version": EXAM_SCHEMA,
        "source_report_schema": REPORT_SCHEMA,
        "provenance": provenance,
        "root_count": len(root_exams),
        "thresholds": {
            "js_warning": float(js_warning),
            "q_margin_warning": float(q_margin_warning),
            "diagnostic_only": True,
        },
        "diagnostic_flag_counts": flag_counts,
        "roots": root_exams,
    }
    return {**unsealed, "content_sha256": content_sha256(unsealed)}


def _action_text(action: Mapping[str, Any]) -> str:
    action_type = str(action["action_type"])
    argument = action.get("argument")
    context = action.get("strategic_context")
    suffix = ""
    if isinstance(context, Mapping) and "total_pips" in context:
        suffix = (
            f" ({context['total_pips']} pips, "
            f"{context.get('resource_pips', {})}, port={context.get('port')})"
        )
    return f"{action_type} {argument!r}{suffix}"


def render_markdown(exam: Mapping[str, Any]) -> str:
    lines = [
        "# Strategic root exam",
        "",
        (
            f"Roots: {exam['root_count']}. Diagnostic thresholds: "
            f"JS ≥ {exam['thresholds']['js_warning']}, completed-Q margin ≤ "
            f"{exam['thresholds']['q_margin_warning']}."
        ),
        "",
    ]
    for root in exam["roots"]:
        lines.extend(
            [
                (
                    f"## Root {root['root_index']} — seed {root['game_seed']} / "
                    f"decision {root['decision_index']}"
                ),
                "",
                (
                    f"{root['phase_raw']}, actor {root['actor']}, "
                    f"{root['legal_width']} legal actions."
                ),
                "",
            ]
        )
        for name, role in root["roles"].items():
            prior = role["raw_prior_top"][0]
            selections = ", ".join(
                _action_text(action) for action in role["selected_actions"]
            )
            lines.extend(
                [
                    f"### {name}",
                    "",
                    (
                        f"- Raw favorite: {_action_text(prior['action'])} "
                        f"({prior['probability']:.3f})"
                    ),
                    f"- Search selections: {selections}",
                    (
                        f"- Cross-seed JS: {role['cross_seed_js_mean']:.4f}; "
                        f"top-action agreement: "
                        f"{role['selected_action_pair_agreement']:.3f}; "
                        f"mean Q margin: {role['mean_completed_q_top_margin']:.4f}"
                    ),
                    (
                        "- Flags: "
                        + (
                            ", ".join(role["diagnostic_flags"])
                            if role["diagnostic_flags"]
                            else "none"
                        )
                    ),
                    "",
                ]
            )
        paired = root.get("paired_counterfactuals")
        if isinstance(paired, Mapping):
            lines.extend(["### Paired counterfactuals", ""])
            for action in paired.get("actions", ()):
                summary = action["summary"]
                lines.append(
                    f"- {_action_text(action['action'])}: mean outcome "
                    f"{float(summary['mean_outcome']):.3f}; wins "
                    f"{int(summary['wins'])}/{int(summary['trials'])}"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-panel", required=True, type=Path)
    parser.add_argument("--search-report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--js-warning", type=float, default=0.1)
    parser.add_argument("--q-margin-warning", type=float, default=0.02)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if not 0.0 <= args.js_warning <= math.log(2.0):
        parser.error("--js-warning must be in [0, ln(2)]")
    if args.q_margin_warning < 0.0:
        parser.error("--q-margin-warning must be non-negative")

    panel = _load_object(args.root_panel)
    report = _load_object(args.search_report)
    provenance = panel.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("root panel provenance must be an object")
    validate_root_panel_payload(
        panel,
        checkpoint_sha256=str(provenance.get("checkpoint_sha256", "")),
        evaluator_config_sha256=str(provenance.get("evaluator_config_sha256", "")),
    )
    exam = build_exam(
        panel,
        report,
        top_k=int(args.top_k),
        js_warning=float(args.js_warning),
        q_margin_warning=float(args.q_margin_warning),
    )
    write_json(args.out, exam)
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(exam), encoding="utf-8")
    print(
        json.dumps(
            {
                "schema_version": exam["schema_version"],
                "root_count": exam["root_count"],
                "diagnostic_flag_counts": exam["diagnostic_flag_counts"],
                "content_sha256": exam["content_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
