#!/usr/bin/env python3
"""Filesystem-mode MLflow logging for evaluation gate runs (task #5).

Replaces (well, indexes) the scattered per-run JSON scoreboards with MLflow
runs under a local `file:` store -- no server, no schema migration. One
experiment per gate type (e.g. "search_vs_raw_h2h", "promotion_gate"); one
MLflow run per gate invocation, carrying the gate's params, metrics, verdict
tags, and the original JSON as an artifact.

OBSERVATION MODE. Every run REQUIRES an `observation_mode` param/tag
("omniscient" or "public"). Our model currently sees opponent hands/dev cards
while catanatron's bots are belief-based, so scoreboards from the two regimes
must never be mixed in one comparison; tagging the mode makes the regime
explicit and filterable in MLflow.

FAIL-OPEN. mlflow is an optional dependency (`pip install mlflow`, or the
`eval` extra). If it is not installed, log_gate_run() logs a warning and
returns None rather than raising -- capturing gate history must never be able
to break a gate run. The param/metric/tag EXTRACTION is pure-dict and import-
free so it can be unit-tested without mlflow.

The extractors tolerate both summary shapes we emit today:
  * gumbel_search_vs_raw_h2h.py  (pentanomial_sprt/pair_sprt/split_rate/...)
  * evaluate_scoreboard.py       (results=[{opponent, win_rate, ...}])
Unknown keys are skipped, not guessed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import warnings
from pathlib import Path
from typing import Any


def git_commit(cwd: str | Path | None = None) -> str | None:
    """Short HEAD commit for provenance; None if not a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _flt(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_params(summary: dict[str, Any], *, observation_mode: str) -> dict[str, Any]:
    """Pull MLflow params (config / provenance, not measurements)."""
    params: dict[str, Any] = {"observation_mode": observation_mode}
    for key in ("checkpoint", "candidate", "n_full", "max_decisions", "seed",
                "pairs_requested", "value_squash", "c_scale", "c_visit",
                "lazy_interior_chance", "correct_rust_chance_spectra", "vps_to_win"):
        if key in summary and summary[key] is not None:
            params[key] = summary[key]
    # elo0/elo1/alpha/beta live inside whichever SPRT block is present.
    for block_key in ("pentanomial_sprt", "pair_sprt", "sprt"):
        block = summary.get(block_key)
        if isinstance(block, dict):
            for k in ("elo0", "elo1", "alpha", "beta"):
                if k in block and f"sprt_{k}" not in params:
                    params[f"sprt_{k}"] = block[k]
    # evaluate_scoreboard.py: record the opponent set as a param.
    results = summary.get("results")
    if isinstance(results, list) and results:
        opponents = [str(r.get("opponent")) for r in results if r.get("opponent") is not None]
        if opponents:
            params["opponents"] = ",".join(opponents)
    commit = git_commit()
    if commit:
        params["git_commit"] = commit
    return params


def extract_metrics(summary: dict[str, Any]) -> dict[str, float]:
    """Pull MLflow metrics (numeric measurements)."""
    metrics: dict[str, float] = {}

    def _put(name: str, value: Any) -> None:
        v = _flt(value)
        if v is not None:
            metrics[name] = v

    # H2H-summary shape.
    _put("search_win_rate", summary.get("search_win_rate"))
    _put("split_rate", summary.get("split_rate"))
    _put("decisive_pair_yield", summary.get("decisive_pair_yield"))
    _put("pairs_decisive", summary.get("pairs_decisive"))
    _put("complete_pairs", summary.get("complete_pairs"))
    games_played = _flt(summary.get("games_played"))
    games_truncated = _flt(summary.get("games_truncated"))
    if games_played and games_played > 0 and games_truncated is not None:
        _put("truncation_rate", games_truncated / games_played)
    for block_key, prefix in (("pentanomial_sprt", "pentanomial"), ("pair_sprt", "concordant")):
        block = summary.get(block_key)
        if isinstance(block, dict):
            _put(f"{prefix}_llr", block.get("llr"))

    # evaluate_scoreboard.py shape: per-opponent + overall win rate.
    results = summary.get("results")
    if isinstance(results, list) and results:
        total_wins = total_games = 0
        for r in results:
            opponent = str(r.get("opponent", "unknown")).replace(" ", "_")
            _put(f"win_rate__{opponent}", r.get("win_rate"))
            total_wins += int(r.get("wins", 0) or 0)
            total_games += int(r.get("games", 0) or 0)
        if total_games > 0:
            _put("overall_win_rate", total_wins / total_games)
    return metrics


def extract_tags(summary: dict[str, Any], *, gate: str, observation_mode: str) -> dict[str, str]:
    """Pull MLflow tags (categorical facets, incl. the verdict)."""
    tags: dict[str, str] = {"gate": gate, "observation_mode": observation_mode}
    for block_key, name in (("pentanomial_sprt", "pentanomial_decision"),
                            ("pair_sprt", "concordant_decision"),
                            ("sprt", "sprt_decision")):
        block = summary.get(block_key)
        if isinstance(block, dict) and block.get("decision") is not None:
            tags[name] = str(block["decision"])
    return tags


def log_gate_run(
    summary: dict[str, Any],
    *,
    gate: str,
    observation_mode: str,
    tracking_uri: str = "file:runs/mlflow",
    artifact_path: str | Path | None = None,
    run_name: str | None = None,
) -> str | None:
    """Log one gate summary as an MLflow run. Returns the run_id, or None if
    mlflow is unavailable (fail-open -- never breaks the caller's gate)."""
    if observation_mode not in {"omniscient", "public"}:
        raise ValueError(
            f"observation_mode must be 'omniscient' or 'public', got {observation_mode!r}"
        )
    try:
        import mlflow
    except Exception:
        warnings.warn(
            "mlflow not installed; skipping gate-run logging "
            "(pip install mlflow, or the project's 'eval' extra). "
            "The gate itself is unaffected.",
            stacklevel=2,
        )
        return None

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(gate)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(extract_params(summary, observation_mode=observation_mode))
        for name, value in extract_metrics(summary).items():
            mlflow.log_metric(name, value)
        mlflow.set_tags(extract_tags(summary, gate=gate, observation_mode=observation_mode))
        if artifact_path and Path(artifact_path).exists():
            mlflow.log_artifact(str(artifact_path))
        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Path to a gate summary/verdict JSON.")
    parser.add_argument("--gate", required=True, help="Experiment name, e.g. search_vs_raw_h2h.")
    parser.add_argument("--observation-mode", required=True, choices=("omniscient", "public"))
    parser.add_argument("--tracking-uri", default="file:runs/mlflow")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the params/metrics/tags that would be logged and exit (no mlflow needed).",
    )
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    if args.dry_run:
        preview = {
            "params": extract_params(summary, observation_mode=args.observation_mode),
            "metrics": extract_metrics(summary),
            "tags": extract_tags(summary, gate=args.gate, observation_mode=args.observation_mode),
        }
        print(json.dumps(preview, indent=2, sort_keys=True, default=str))
        return
    run_id = log_gate_run(
        summary,
        gate=args.gate,
        observation_mode=args.observation_mode,
        tracking_uri=args.tracking_uri,
        artifact_path=args.summary,
        run_name=args.run_name,
    )
    print(json.dumps({"run_id": run_id, "gate": args.gate}, indent=2))


if __name__ == "__main__":
    main()
