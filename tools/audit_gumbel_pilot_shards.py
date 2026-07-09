"""Independent quality audit for Gumbel self-play pilot shards.

Written without touching `catan_zero.rl.gumbel_self_play` /
`catan_zero.search.gumbel_chance_mcts` (the driver under audit) -- this reads
only the on-disk `.npz` shard files + `manifest.json` the driver produces, per
`BASE_KEYS`/`EXTRA_KEYS` in `gumbel_self_play.py` (schema reference only, not
imported). The independence is deliberate: an audit that shares code with the
thing it's auditing can't catch a shared misconception.

Checks (see `--help` for CLI flags):
  (a) label perspective correctness -- winner's final_actual_vps >= vps_to_win,
      exactly one of the two seats crosses the threshold on terminated games
      (a same-game "zero-sum" sanity check for a 2-player win/loss label),
      truncated games carry no outcome labels.
  (b) target sanity -- target_policy sums to 1 over legal actions. NOTE:
      KL(target_policy || prior) is NOT computable from the current shard
      schema: `SearchResult.priors` is not persisted in `BASE_KEYS`/`EXTRA_KEYS`
      (only `improved_policy`, `q_values`, `afterstate_values` are). This
      check is reported as SKIPPED unless a `prior_policy` column (or
      whatever the schema is extended to use) is present.
  (c) mix checks -- is_forced fraction, full-search fraction among non-forced
      rows (vs configured p_full), phase distribution (optionally against a
      reference distribution file).
  (d) afterstate targets present and in [-1, 1] specifically on forced-ROLL
      rows (decoded via the pre-existing, stable `ActionCatalog` -- a
      separate utility from the driver being audited).

Usage:
    .venv/bin/python tools/audit_gumbel_pilot_shards.py --shards-dir runs/gumbel_pilot \
        --vps-to-win 10 --p-full 0.25 --colors RED,BLUE --out runs/gumbel_pilot/audit_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factory_common import write_json  # noqa: E402

from catan_zero.rl.action_mask import ActionCatalog  # noqa: E402

PLAYER_NAMES = ("BLUE", "RED", "ORANGE", "WHITE")


def find_shard_files(shards_dir: Path) -> list[Path]:
    manifest_path = shards_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        shard_paths = [Path(path) for path in manifest.get("shards", [])]
        if shard_paths:
            return [path for path in shard_paths if path.exists()]
    return sorted(shards_dir.rglob("*.npz"))


def find_worker_manifests(shards_dir: Path) -> list[Path]:
    top_manifest_path = shards_dir / "manifest.json"
    if top_manifest_path.exists():
        top_manifest = json.loads(top_manifest_path.read_text())
        paths = [Path(path) for path in top_manifest.get("worker_summaries", [])]
        if paths:
            return [path for path in paths if path.exists()]
    return sorted(shards_dir.glob("worker_*/manifest.json"))


def check_config_provenance(
    shards_dir: Path, *, expected: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Verify search/generation config against what was actually recorded in
    the manifests -- as of commit 982d344, per-worker manifest.json carries
    `selfplay_config`/`search_config` (full dataclasses.asdict dumps of what
    the worker actually constructed) and the top-level manifest carries
    `cli_args`. Older runs (anything generated before that commit) won't have
    either -- this is reported as skipped, not failed, with a pointer to
    whatever out-of-band verification was used instead (see this tool's
    audit report prose for the specific pilot run this applies to).
    """
    expected = expected or {"c_scale": 0.1}
    top_manifest_path = shards_dir / "manifest.json"
    cli_args = None
    if top_manifest_path.exists():
        top_manifest = json.loads(top_manifest_path.read_text())
        cli_args = top_manifest.get("cli_args")

    worker_manifests = find_worker_manifests(shards_dir)
    search_configs = []
    selfplay_configs = []
    for path in worker_manifests:
        manifest = json.loads(path.read_text())
        if "search_config" in manifest:
            search_configs.append(manifest["search_config"])
        if "selfplay_config" in manifest:
            selfplay_configs.append(manifest["selfplay_config"])

    if not search_configs and cli_args is None:
        return {
            "check": "config_provenance",
            "pass": None,
            "skipped": True,
            "reason": (
                "Neither 'cli_args' (top-level manifest) nor 'search_config' "
                "(per-worker manifest) present -- this run predates commit 982d344 "
                "(full config provenance). Verification for such runs must come from "
                "an out-of-band source (e.g. live-process `ps aux` inspection); note "
                "that source explicitly in the audit report instead of relying on this check."
            ),
        }

    failures: list[str] = []
    observed: dict[str, Any] = {}
    for key, expected_value in expected.items():
        values_seen = {
            config[key] for config in search_configs if key in config
        } | {config[key] for config in selfplay_configs if key in config}
        if not values_seen and cli_args is not None and key in cli_args:
            values_seen = {cli_args[key]}
        if not values_seen:
            continue
        observed[key] = sorted(values_seen) if len(values_seen) > 1 else next(iter(values_seen))
        if len(values_seen) > 1:
            failures.append(f"{key}: workers disagree -- values seen {sorted(values_seen)}")
        else:
            actual_value = next(iter(values_seen))
            if actual_value != expected_value:
                failures.append(f"{key}: expected {expected_value!r}, found {actual_value!r}")

    return {
        "check": "config_provenance",
        "pass": len(failures) == 0,
        "skipped": False,
        "workers_checked": len(search_configs),
        "expected": expected,
        "observed": observed,
        "failures": failures,
    }


def load_rows(shard_files: list[Path]) -> dict[str, np.ndarray]:
    """Concatenate all shards' rows into one dict of stacked arrays.

    Shards may have different padded legal-action widths; pad to the global
    max width for the ragged (per-decision) columns before concatenating.
    """
    per_shard: list[dict[str, np.ndarray]] = []
    for path in shard_files:
        with np.load(path, allow_pickle=True) as handle:
            per_shard.append({key: handle[key] for key in handle.files})
    if not per_shard:
        return {}

    all_keys = set()
    for shard in per_shard:
        all_keys.update(shard.keys())

    ragged_keys = {
        "legal_action_ids",
        "target_policy",
        "target_scores",
        "target_policy_mask",
        "target_scores_mask",
        "afterstate_target",
        "afterstate_target_mask",
        "prior_policy",  # forward-compat: not in the schema yet, padded like target_policy once added.
        # Real shards (unlike this test module's narrow synthetic fixtures)
        # also carry `legal_action_context` and the ENTITY_KEYS ragged
        # columns -- all vary by the same per-decision legal-action width as
        # `legal_action_ids`. Missing these caused a real `np.concatenate`
        # dimension-mismatch crash the first time two shards genuinely
        # differed in legal width (e.g. a 54-action placement decision vs a
        # normal ~10-30-action turn) -- 2026-07-04, task #65 raw-selfplay
        # smoke run.
        "legal_action_context",
        "legal_action_tokens",
        "legal_action_target_ids",
        "legal_action_mask",
    }
    width_keys = ragged_keys & all_keys
    max_width = 0
    for shard in per_shard:
        for key in width_keys:
            if key in shard:
                max_width = max(max_width, shard[key].shape[1])

    # Explicit per-key pad fill (not substring-inferred): a probability-like
    # column padded with -1 (legal_action_ids's own "no action here" sentinel)
    # would poison downstream log()/multiply checks at the padded positions
    # even though those positions get masked out afterward -- keep the fill
    # value semantically valid for each column's own dtype/meaning.
    pad_fill_by_key = {
        "legal_action_ids": -1,
        "target_policy": np.nan,
        "target_scores": np.nan,
        "prior_policy": np.nan,
        "target_policy_mask": False,
        "target_scores_mask": False,
        "afterstate_target": np.nan,
        "afterstate_target_mask": False,
        "legal_action_context": 0.0,
        "legal_action_tokens": 0.0,
        "legal_action_target_ids": -1,
        "legal_action_mask": False,
    }

    merged: dict[str, list[np.ndarray]] = defaultdict(list)
    for shard in per_shard:
        n_rows = len(shard.get("game_seed", []))
        for key in all_keys:
            if key not in shard:
                continue
            value = shard[key]
            if key in width_keys and value.ndim >= 2 and value.shape[1] < max_width:
                fill = pad_fill_by_key.get(key, np.nan)
                pad_width = max_width - value.shape[1]
                pad_shape = [(0, 0), (0, pad_width)] + [(0, 0)] * (value.ndim - 2)
                value = np.pad(value, pad_shape, mode="constant", constant_values=fill)
            merged[key].append(value)

    return {key: np.concatenate(values, axis=0) for key, values in merged.items()}


def check_label_perspective(rows: dict[str, np.ndarray], *, vps_to_win: int) -> dict[str, Any]:
    game_seeds = rows["game_seed"]
    winners = rows["winner"]
    terminated = rows["terminated"]
    truncated = rows["truncated"]
    final_actual_vps = rows["final_actual_vps"]
    has_final = rows["has_final_actual_vps"]
    players = rows["player"]

    failures: list[str] = []
    games_checked = 0
    games_terminated = 0
    games_truncated = 0

    unique_seeds = np.unique(game_seeds)
    for seed in unique_seeds:
        mask = game_seeds == seed
        idx = np.nonzero(mask)[0]
        games_checked += 1

        game_terminated = bool(terminated[idx[0]])
        game_truncated = bool(truncated[idx[0]])
        game_winner = str(winners[idx[0]])
        game_has_final = bool(has_final[idx[0]])

        # All rows of one game must agree on outcome fields (set once, post-hoc).
        if not np.all(terminated[idx] == game_terminated):
            failures.append(f"seed {seed}: terminated flag disagrees across rows")
        if not np.all(truncated[idx] == game_truncated):
            failures.append(f"seed {seed}: truncated flag disagrees across rows")
        if not np.all(has_final[idx] == game_has_final):
            failures.append(f"seed {seed}: has_final_actual_vps disagrees across rows")
        if game_terminated == game_truncated:
            failures.append(
                f"seed {seed}: terminated ({game_terminated}) and truncated "
                f"({game_truncated}) are not complementary"
            )

        if game_terminated:
            games_terminated += 1
            if game_winner == "" or game_winner not in PLAYER_NAMES:
                failures.append(f"seed {seed}: terminated but winner={game_winner!r} is invalid")
                continue
            if not game_has_final:
                failures.append(f"seed {seed}: terminated but has_final_actual_vps is False")
                continue
            vps = final_actual_vps[idx[0]]
            winner_seat = PLAYER_NAMES.index(game_winner)
            winner_vps = int(vps[winner_seat])
            if winner_vps < vps_to_win:
                failures.append(
                    f"seed {seed}: winner {game_winner} has final_actual_vps={winner_vps} "
                    f"< vps_to_win={vps_to_win}"
                )
            participants = sorted(set(str(p) for p in players[idx]))
            qualifiers = [
                name
                for name in participants
                if int(vps[PLAYER_NAMES.index(name)]) >= vps_to_win
            ]
            if qualifiers != [game_winner]:
                failures.append(
                    f"seed {seed}: exactly-one-qualifier (zero-sum) check failed -- "
                    f"participants={participants} qualifiers={qualifiers} winner={game_winner}"
                )
        elif game_truncated:
            games_truncated += 1
            if game_winner != "":
                failures.append(f"seed {seed}: truncated but winner={game_winner!r} is non-empty")
            if game_has_final:
                failures.append(f"seed {seed}: truncated but has_final_actual_vps is True")

    return {
        "check": "label_perspective_correctness",
        "pass": len(failures) == 0,
        "games_checked": games_checked,
        "games_terminated": games_terminated,
        "games_truncated": games_truncated,
        "failures": failures[:50],
        "num_failures": len(failures),
    }


def check_target_policy_sums_to_one(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    target_policy = rows["target_policy"].astype(np.float64)
    legal_action_ids = rows["legal_action_ids"]
    valid = legal_action_ids != -1
    sums = np.where(valid, target_policy, 0.0).sum(axis=1)
    tolerance = 5e-3  # target_policy is stored as float16
    bad = np.nonzero(np.abs(sums - 1.0) > tolerance)[0]
    return {
        "check": "target_policy_sums_to_one",
        "pass": len(bad) == 0,
        "rows_checked": int(target_policy.shape[0]),
        "num_failures": int(len(bad)),
        "example_bad_rows": [
            {"row_index": int(i), "sum": float(sums[i])} for i in bad[:10]
        ],
        "sum_distribution": {
            "min": float(sums.min()) if len(sums) else None,
            "max": float(sums.max()) if len(sums) else None,
            "mean": float(sums.mean()) if len(sums) else None,
        },
    }


def check_kl_improved_vs_prior(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    if "prior_policy" not in rows:
        return {
            "check": "kl_improved_policy_vs_prior",
            "pass": None,
            "skipped": True,
            "reason": (
                "'prior_policy' column not present in these shards. As of the "
                "2026-07-03 schema-gap finding, SearchResult.priors was not persisted "
                "in BASE_KEYS/EXTRA_KEYS (only improved_policy/q_values/afterstate_values "
                "were) -- a 'prior_policy' column is being added to close this; rerun once "
                "shards include it."
            ),
        }
    target_policy = rows["target_policy"].astype(np.float64)
    prior_policy = rows["prior_policy"].astype(np.float64)
    legal_action_ids = rows["legal_action_ids"]
    valid = legal_action_ids != -1
    eps = 1e-8
    # Substitute a safe placeholder at padded/invalid positions before log() --
    # np.where still evaluates both branches elementwise, so NaN-padded values
    # would otherwise raise spurious "invalid value in log" warnings even
    # though the result at those positions is discarded anyway.
    safe_target = np.where(valid, target_policy, 1.0)
    safe_prior = np.where(valid, prior_policy, 1.0)
    kl_per_row = np.sum(
        np.where(valid, safe_target * (np.log(safe_target + eps) - np.log(safe_prior + eps)), 0.0),
        axis=1,
    )

    def _distribution(mask: np.ndarray) -> dict[str, Any] | None:
        values = kl_per_row[mask]
        if len(values) == 0:
            return None
        return {
            "rows": int(len(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "near_zero_fraction": float(np.mean(values < 1e-3)),
        }

    result: dict[str, Any] = {
        "check": "kl_improved_policy_vs_prior",
        "pass": None,  # informational: report distribution, no fixed pass/fail threshold
        "skipped": False,
        "kl_distribution": _distribution(np.ones_like(kl_per_row, dtype=bool)),
        "note": (
            "If near_zero_fraction is high, the search isn't adding information over the "
            "prior -- consider raising n_full before the full run. IMPORTANT: forced "
            "(single-legal-action) rows have exactly one choice, so KL(improved||prior) is "
            "trivially 0 for them regardless of search quality -- they dilute the overall "
            "distribution. The 'non_forced' breakdown below is the actually meaningful "
            "signal for whether the search is adding information when there's a real choice."
        ),
    }
    if "is_forced" in rows:
        is_forced = rows["is_forced"].astype(bool)
        result["kl_distribution_non_forced"] = _distribution(~is_forced)
        result["kl_distribution_forced"] = _distribution(is_forced)
    return result


def check_mix(
    rows: dict[str, np.ndarray],
    *,
    p_full: float,
    reference_phase_counts: dict[str, int] | None,
    is_forced_expected_range: tuple[float, float] = (0.40, 0.70),
) -> dict[str, Any]:
    is_forced = rows["is_forced"].astype(bool)
    used_full_search = rows["used_full_search"].astype(bool)
    phases = rows["phase"]

    n = len(is_forced)
    forced_fraction = float(is_forced.mean()) if n else 0.0
    non_forced_mask = ~is_forced
    non_forced_count = int(non_forced_mask.sum())
    full_search_fraction_non_forced = (
        float(used_full_search[non_forced_mask].mean()) if non_forced_count else None
    )

    phase_counts = Counter(str(p) for p in phases)
    total = sum(phase_counts.values()) or 1
    phase_distribution = {phase: count / total for phase, count in phase_counts.items()}

    # NOTE: upper bound raised 0.55 -> 0.70 (2026-07-04, team-lead calibration):
    # post-F1 fix + max_decisions=600, games run shorter and more decisively,
    # so a higher forced-ROLL fraction (every turn has exactly one) is
    # expected and not itself a red flag. The binding correctness check
    # remains `check_weight_multipliers` (policy_weight_multiplier==0 on
    # every forced row) -- this range is a coarse sanity band, not the real
    # guarantee.
    low, high = is_forced_expected_range
    forced_ok = low <= forced_fraction <= high
    full_search_ok = (
        full_search_fraction_non_forced is not None
        and abs(full_search_fraction_non_forced - p_full) <= 0.05
    )

    result: dict[str, Any] = {
        "check": "mix",
        "pass": bool(forced_ok and full_search_ok),
        "is_forced_fraction": forced_fraction,
        "is_forced_expected_range": list(is_forced_expected_range),
        "is_forced_in_range": forced_ok,
        "full_search_fraction_among_non_forced": full_search_fraction_non_forced,
        "p_full_configured": p_full,
        "full_search_matches_p_full": full_search_ok,
        "phase_distribution": phase_distribution,
        "phase_counts": dict(phase_counts),
    }
    if reference_phase_counts:
        ref_total = sum(reference_phase_counts.values()) or 1
        ref_distribution = {k: v / ref_total for k, v in reference_phase_counts.items()}
        result["reference_phase_distribution"] = ref_distribution
        result["phase_distribution_delta"] = {
            phase: phase_distribution.get(phase, 0.0) - ref_distribution.get(phase, 0.0)
            for phase in set(phase_distribution) | set(ref_distribution)
        }
    return result


def check_afterstate_targets_on_forced_roll(
    rows: dict[str, np.ndarray], *, colors: tuple[str, ...]
) -> dict[str, Any]:
    catalog = ActionCatalog(colors)
    is_forced = rows["is_forced"].astype(bool)
    legal_action_ids = rows["legal_action_ids"]
    afterstate_target = rows["afterstate_target"].astype(np.float64)
    afterstate_target_mask = rows["afterstate_target_mask"].astype(bool)

    forced_idx = np.nonzero(is_forced)[0]
    forced_roll_idx = []
    for i in forced_idx:
        valid_ids = legal_action_ids[i][legal_action_ids[i] != -1]
        if len(valid_ids) != 1:
            continue  # not truly single-legal-action at the policy-id level; skip
        action_id = int(valid_ids[0])
        if not (0 <= action_id < catalog.size):
            continue
        if catalog.describe(action_id)["action_type"] == "ROLL":
            forced_roll_idx.append(i)

    failures: list[str] = []
    for i in forced_roll_idx:
        mask_row = afterstate_target_mask[i]
        if not mask_row.any():
            failures.append(f"row {i}: forced-ROLL row has no afterstate targets present")
            continue
        values = afterstate_target[i][mask_row]
        out_of_range = values[(values < -1.0) | (values > 1.0)]
        if len(out_of_range) > 0:
            failures.append(
                f"row {i}: forced-ROLL afterstate values out of [-1,1]: {out_of_range.tolist()[:5]}"
            )

    return {
        "check": "afterstate_targets_on_forced_roll",
        "pass": len(failures) == 0,
        "forced_roll_rows_found": len(forced_roll_idx),
        "num_failures": len(failures),
        "failures": failures[:50],
    }


def check_truncation_rate(rows: dict[str, np.ndarray], *, threshold: float = 0.40) -> dict[str, Any]:
    """Flag (not just describe) an excessive truncated-game fraction.

    Cheap to compute per-game (one row per game_seed suffices, since
    `truncated` is set identically for every row of a game), but this counts
    per-row rather than per-game deliberately: a run with many truncated
    *long* games and few completed *short* games would otherwise understate
    how much of the actual training signal is truncated-derived.
    """
    game_seeds = rows["game_seed"]
    truncated = rows["truncated"].astype(bool)
    unique_seeds = np.unique(game_seeds)
    per_game_truncated = []
    for seed in unique_seeds:
        idx = game_seeds == seed
        per_game_truncated.append(bool(truncated[idx][0]))
    per_game_truncated = np.asarray(per_game_truncated)
    games_total = len(per_game_truncated)
    games_truncated = int(per_game_truncated.sum())
    fraction = games_truncated / games_total if games_total else 0.0
    return {
        "check": "truncation_rate",
        "pass": fraction <= threshold,
        "games_total": games_total,
        "games_truncated": games_truncated,
        "truncated_fraction": fraction,
        "threshold": threshold,
        "rows_from_truncated_games_fraction": float(truncated.mean()) if len(truncated) else 0.0,
    }


def check_weight_multipliers(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    """policy_weight_multiplier must be exactly 0 on every forced row (no
    search signal to imitate when there's no choice); value_weight_multiplier
    must be exactly 1 everywhere (every row -- forced or not -- carries a
    real value target per the driver's own design intent).
    """
    failures: list[str] = []
    is_forced = rows.get("is_forced")
    policy_weight = rows.get("policy_weight_multiplier")
    value_weight = rows.get("value_weight_multiplier")

    if is_forced is None or policy_weight is None or value_weight is None:
        missing = [
            name
            for name, value in (
                ("is_forced", is_forced),
                ("policy_weight_multiplier", policy_weight),
                ("value_weight_multiplier", value_weight),
            )
            if value is None
        ]
        return {
            "check": "weight_multipliers",
            "pass": None,
            "skipped": True,
            "reason": f"missing column(s) in shards: {missing}",
        }

    is_forced = is_forced.astype(bool)
    policy_weight = policy_weight.astype(np.float64)
    value_weight = value_weight.astype(np.float64)

    bad_policy_on_forced = np.nonzero(is_forced & (policy_weight != 0.0))[0]
    bad_value = np.nonzero(value_weight != 1.0)[0]

    if len(bad_policy_on_forced) > 0:
        failures.append(
            f"{len(bad_policy_on_forced)} forced row(s) have nonzero policy_weight_multiplier "
            f"(examples: rows {bad_policy_on_forced[:10].tolist()})"
        )
    if len(bad_value) > 0:
        failures.append(
            f"{len(bad_value)} row(s) have value_weight_multiplier != 1.0 "
            f"(examples: rows {bad_value[:10].tolist()}, values "
            f"{value_weight[bad_value[:10]].tolist()})"
        )

    return {
        "check": "weight_multipliers",
        "pass": len(failures) == 0,
        "skipped": False,
        "forced_rows_checked": int(is_forced.sum()),
        "rows_checked": int(len(is_forced)),
        "num_failures": len(failures),
        "failures": failures,
    }


def check_prior_policy_nondegenerate(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    """Sanity-check the (new) prior_policy column isn't degenerate on real
    (non-forced) decisions -- a collapsed/one-hot prior for a genuine
    multi-way choice would suggest a broken evaluator, not just fp16
    small-probability flushing (which is expected and fine).
    """
    if "prior_policy" not in rows:
        return {
            "check": "prior_policy_nondegenerate",
            "pass": None,
            "skipped": True,
            "reason": "'prior_policy' column not present in these shards.",
        }
    is_forced = rows["is_forced"].astype(bool) if "is_forced" in rows else np.zeros(
        len(rows["prior_policy"]), dtype=bool
    )
    non_forced = ~is_forced
    prior_policy = rows["prior_policy"][non_forced].astype(np.float64)
    legal_action_ids = rows["legal_action_ids"][non_forced]
    valid = legal_action_ids != -1

    eps = 1e-12
    safe_prior = np.where(valid, prior_policy, 0.0)
    entropy_per_row = -np.sum(np.where(valid, safe_prior * np.log(safe_prior + eps), 0.0), axis=1)
    num_legal = valid.sum(axis=1)
    max_entropy = np.log(np.maximum(num_legal, 1))
    normalized_entropy = np.divide(
        entropy_per_row, max_entropy, out=np.zeros_like(entropy_per_row), where=max_entropy > 0
    )

    near_one_hot_fraction = float(np.mean(normalized_entropy < 0.01)) if len(normalized_entropy) else 0.0
    return {
        "check": "prior_policy_nondegenerate",
        "pass": near_one_hot_fraction < 0.5,  # loose guard: majority-collapsed priors are suspicious
        "skipped": False,
        "non_forced_rows_checked": int(non_forced.sum()),
        "normalized_entropy_distribution": {
            "min": float(normalized_entropy.min()) if len(normalized_entropy) else None,
            "max": float(normalized_entropy.max()) if len(normalized_entropy) else None,
            "mean": float(normalized_entropy.mean()) if len(normalized_entropy) else None,
            "median": float(np.median(normalized_entropy)) if len(normalized_entropy) else None,
        },
        "near_one_hot_fraction": near_one_hot_fraction,
        "note": (
            "normalized_entropy is entropy / log(num_legal_actions), so 1.0 = uniform, "
            "0.0 = one-hot. fp16 storage flushing small probabilities to 0 is expected and "
            "not itself a problem; this only flags if collapse-to-one-hot is the norm across "
            "real (non-forced) multi-way decisions, which would suggest a broken evaluator."
        ),
    }


def run_audit(
    shards_dir: Path,
    *,
    vps_to_win: int,
    p_full: float,
    colors: tuple[str, ...],
    reference_phase_counts: dict[str, int] | None,
    truncation_threshold: float = 0.40,
    expected_config: dict[str, Any] | None = None,
    is_forced_expected_range: tuple[float, float] = (0.40, 0.70),
) -> dict[str, Any]:
    shard_files = find_shard_files(shards_dir)
    if not shard_files:
        return {"error": f"no shard files found under {shards_dir}"}
    rows = load_rows(shard_files)
    if not rows:
        return {"error": "shards found but contained no rows"}

    checks = [
        check_label_perspective(rows, vps_to_win=vps_to_win),
        check_target_policy_sums_to_one(rows),
        check_kl_improved_vs_prior(rows),
        check_mix(
            rows,
            p_full=p_full,
            reference_phase_counts=reference_phase_counts,
            is_forced_expected_range=is_forced_expected_range,
        ),
        check_afterstate_targets_on_forced_roll(rows, colors=colors),
        check_truncation_rate(rows, threshold=truncation_threshold),
        check_weight_multipliers(rows),
        check_prior_policy_nondegenerate(rows),
        check_config_provenance(shards_dir, expected=expected_config),
    ]
    overall_pass = all(c["pass"] for c in checks if c["pass"] is not None)
    return {
        "shards_dir": str(shards_dir),
        "shard_files_count": len(shard_files),
        "rows_total": int(len(rows.get("game_seed", []))),
        "games_total": int(len(np.unique(rows["game_seed"]))) if "game_seed" in rows else 0,
        "overall_pass": overall_pass,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--p-full", type=float, default=0.25)
    parser.add_argument("--colors", default="RED,BLUE")
    parser.add_argument(
        "--reference-phase-counts",
        default=None,
        help="Optional JSON file: {\"PLAY_TURN\": 12345, ...} from the teacher corpus, for comparison.",
    )
    parser.add_argument(
        "--truncation-threshold",
        type=float,
        default=0.40,
        help="Flag (fail) the truncation_rate check if more than this fraction of games truncated.",
    )
    parser.add_argument(
        "--expected-config",
        default=None,
        help=(
            "Optional JSON object of expected config values to verify against "
            "manifest.json's 'search_config'/'selfplay_config'/'cli_args' (post commit "
            "982d344 only -- older runs skip this check). Defaults to {\"c_scale\": 0.1}. "
            "e.g. '{\"c_scale\": 0.1, \"n_full\": 64}'"
        ),
    )
    parser.add_argument(
        "--is-forced-expected-range",
        default="0.40,0.70",
        help=(
            "Comma-separated low,high fraction band for is_forced in the mix check. "
            "Raised to 0.40-0.70 (2026-07-04 calibration): shorter, more decisive "
            "post-F1 games legitimately have a higher forced-ROLL fraction; the binding "
            "correctness check remains weight_multipliers, not this coarse band."
        ),
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    reference_phase_counts = None
    if args.reference_phase_counts:
        reference_phase_counts = json.loads(Path(args.reference_phase_counts).read_text())

    expected_config = json.loads(args.expected_config) if args.expected_config else None
    low_str, high_str = args.is_forced_expected_range.split(",")
    is_forced_expected_range = (float(low_str), float(high_str))

    report = run_audit(
        Path(args.shards_dir),
        vps_to_win=args.vps_to_win,
        p_full=args.p_full,
        colors=tuple(c.strip() for c in args.colors.split(",")),
        reference_phase_counts=reference_phase_counts,
        truncation_threshold=args.truncation_threshold,
        expected_config=expected_config,
        is_forced_expected_range=is_forced_expected_range,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if args.out:
        write_json(args.out, report)
        print(f"\nFull report written to {args.out}")

    if report.get("error"):
        raise SystemExit(2)
    if not report.get("overall_pass", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
