from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path

import pytest
import numpy as np

from tools import a1_promotion_transaction as promotion
from tools import a1_promotion_artifacts as artifacts
from tools import a1_one_dose_train as one_dose
from tools.champion_registry import ChampionRegistry
from tools.high_regret_suite_contract import REPLAY_CONTRACT, scope_inventory_sha256


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _contract(
    *,
    n_full: int = 128,
    n_full_wide=None,
    producer: Path | None = None,
    c_scale: float = 0.03,
) -> dict:
    recipe = {
        "world_size": 1,
        "optimizer": "adam",
        "mask_hidden_info": True,
        "epochs": 1,
        "max_steps": 0,
    }
    producer = producer or Path("/producer.pt")
    search = {
        "belief_chance_spectra": False,
        "c_scale": c_scale,
        "c_visit": 50.0,
        "correct_rust_chance_spectra": True,
        "determinization_min_simulations": 32,
        "determinization_particles": 4,
        "exact_budget_sh": False,
        "exact_budget_sh_min_n": 0,
        "information_set_search": True,
        "lazy_interior_chance": True,
        "max_depth": 80,
        "n_fast": 16,
        "n_full": n_full,
        "n_full_wide": n_full_wide,
        "n_full_wide_threshold": None,
        "p_full": 0.25,
        "play_sh_winner": False,
        "policy_target_min_visits": 0,
        "prior_temperature": 1.0,
        "raw_policy_above_width": None,
        "rescale_noise_floor_c": 0.0,
        "root_wave_batching": False,
        "sigma_eval": 0.98,
        "symmetry_averaged_eval": True,
        "symmetry_averaged_eval_threshold": 20,
        "uncertainty_backup_a": 0.25,
        "uncertainty_backup_cap": 1.0,
        "uncertainty_backup_exp": 1.0,
        "uncertainty_backup_weighting": False,
        "use_batch_api": True,
        "variance_aware_closed_form_js": False,
        "variance_aware_k": 1.0,
        "variance_aware_q": False,
        "wide_candidates_threshold": 24,
        "wide_roots_always_full": n_full_wide is not None,
        "max_root_candidates": 16,
        "max_root_candidates_wide": 54,
    }
    evaluator = {
        "cache_size": 0,
        "context_fill": 0.0,
        "emit_uncertainty": False,
        "prior_temperature": 1.0,
        "public_observation": True,
        "rust_featurize": False,
        "value_readout": "scalar",
        "value_scale": 1.0,
        "value_squash": "tanh",
    }
    return {
        "contract_id": promotion.HISTORICAL_MARKERLESS_A1_CONTRACT["contract_id"],
        "contract_sha256": promotion.HISTORICAL_MARKERLESS_A1_CONTRACT[
            "contract_sha256"
        ],
        "science": {
            "search_operator": search,
            "effective_search_config": search,
            "evaluator": evaluator,
            "learner_training_recipe": recipe,
            "learner_training_recipe_sha256": promotion._digest_value(recipe),
            "learner_value_objective": {"value_readout": "scalar"},
        },
        "generation": {"max_decisions": 600},
        "checkpoints": [
            {
                "role": "producer",
                "path": str(producer),
                "sha256": promotion._sha256(producer)
                if producer.is_file()
                else "sha256:" + "f" * 64,
            }
        ],
    }


def _checkpoint_ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": promotion._sha256(path)}


def _write_one_dose_receipt(
    tmp_path: Path,
    *,
    contract_path: Path,
    contract: dict,
    candidate: Path,
    report: Path,
    command: list[str],
    execution_binding: dict,
) -> Path:
    optimizer = Path(str(candidate) + ".optimizer.pt")
    optimizer.write_bytes(b"fresh adam optimizer")
    outputs = {
        "checkpoint": str(candidate),
        "checkpoint_sha256": promotion._sha256(candidate),
        "optimizer_sidecar": str(optimizer),
        "optimizer_sidecar_sha256": promotion._sha256(optimizer),
        "report": str(report),
        "report_sha256": promotion._sha256(report),
        "execution_binding_sha256": promotion._digest_value(execution_binding),
        "steps_completed": 7,
        "corpus_row_count": 100,
        "training_row_count": 95,
        "validation_row_count": 5,
    }
    common = {
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "lock": str(contract_path),
        "lock_file_sha256": promotion._sha256(contract_path),
        "corpus": str(tmp_path / "audited-corpus"),
        "corpus_meta_file_sha256": "sha256:" + "1" * 64,
        "payload_inventory_sha256": "sha256:" + "2" * 64,
        "validation_manifest": str(tmp_path / "validation.json"),
        "validation_manifest_file_sha256": "sha256:" + "3" * 64,
        "producer_checkpoint_sha256": contract["checkpoints"][0]["sha256"],
        "learner_training_recipe_sha256": contract["science"][
            "learner_training_recipe_sha256"
        ],
        "command": command,
        "command_sha256": promotion._digest_value(command),
        "execution_binding": execution_binding,
        "world_size": 1,
        "gpu": 0,
        "gpu_name": "NVIDIA B200",
        "started_unix_ns": 10,
        "finished_unix_ns": 20,
        "returncode": 0,
        "outputs": outputs,
        "failure": None,
    }
    receipt_path = tmp_path / "one-dose.receipt.json"
    claim_path = tmp_path / "one-dose.claim.json"
    claim = {
        "schema_version": one_dose.CLAIM_SCHEMA,
        **common,
        "receipt_target": str(receipt_path),
    }
    claim["state_sha256"] = one_dose._value_sha256(claim)
    _write_json(claim_path, claim)
    receipt = {
        "schema_version": one_dose.RECEIPT_SCHEMA,
        **common,
        "claim": str(claim_path),
        "claim_state_sha256": claim["state_sha256"],
    }
    receipt["receipt_sha256"] = one_dose._value_sha256(receipt)
    _write_json(receipt_path, receipt)
    return receipt_path


def _write_evidence_envelope(
    path: Path,
    *,
    kind: str,
    contract: dict,
    candidate: Path,
    champion: Path,
    sources: list[tuple[str, Path]],
    verdict: str,
    result: dict,
) -> None:
    payload = {
        "schema_version": promotion.EVIDENCE_SCHEMA,
        "kind": kind,
        "passed": True,
        "verdict": verdict,
        "contract_sha256": contract["contract_sha256"],
        "candidate": _checkpoint_ref(candidate),
        "champion": _checkpoint_ref(champion),
        "sources": [
            {"role": role, "path": str(source), "sha256": promotion._sha256(source)}
            for role, source in sources
        ],
        "result": result,
    }
    payload["evidence_sha256"] = promotion._digest_value(payload)
    _write_json(path, payload)


def _fixture(
    tmp_path: Path,
    *,
    promotion_count: int = 0,
    n_full: int = 128,
    champion: Path | None = None,
) -> dict:
    champion = champion or tmp_path / "champion.pt"
    candidate = tmp_path / "candidate.pt"
    champion.parent.mkdir(parents=True, exist_ok=True)
    champion.write_bytes(b"incumbent checkpoint")
    candidate.write_bytes(b"candidate checkpoint")
    registry_path = tmp_path / "registry.json"
    registry = ChampionRegistry(registry_path)
    registry.set_role(
        "generator_champion",
        champion,
        expected_md5=promotion._md5(champion),
        version=4,
        reason="fixture",
    )
    registry.set_role(
        "public_champion",
        champion,
        expected_md5=promotion._md5(champion),
        version=4,
        reason="fixture",
    )
    for _ in range(promotion_count):
        registry.record_promotion()
    registry.save()
    pointer = tmp_path / "CURRENT_CHAMPION"
    pointer.write_text(str(champion.resolve()) + "\n", encoding="utf-8")
    contract_path = tmp_path / "contract.lock.json"
    contract_path.write_text("{}\n", encoding="utf-8")
    contract = _contract(n_full=n_full, producer=champion)
    # The n196 negative fixture must still be constructible so execution can
    # prove the contract is rejected before any mutation/evidence processing.
    evidence_contract = contract if n_full == 128 else _contract(n_full=128, producer=champion)
    evidence_semantics = promotion._sealed_evaluation_semantics(evidence_contract)
    candidate_search_config = promotion._candidate_search_config(evidence_contract)
    champion_search_config = promotion._incumbent_search_config(
        evidence_contract,
        registry=registry,
        champion_path=champion.resolve(),
        champion_sha256=promotion._sha256(champion),
    )
    report_path = tmp_path / "report.json"
    command = ["/usr/bin/python3", "tools/train_bc.py", "--sealed-a1"]
    execution_binding = one_dose._execution_binding(
        command=command, environment=one_dose._child_environment(0)
    )
    _write_json(
        report_path,
        {
            "a1_contract_sha256": contract["contract_sha256"],
            "a1_learner_training_recipe_sha256": contract["science"][
                "learner_training_recipe_sha256"
            ],
            "a1_bound_learner_training_recipe": contract["science"][
                "learner_training_recipe"
            ],
            "arch": "entity_graph",
            "mask_hidden_info": True,
            "track": "2p_no_trade",
            "vps_to_win": 10,
            "steps_completed": 7,
            "epochs": 1,
            "max_steps": 0,
            "checkpoint": str(candidate),
            "init_checkpoint_sha256": contract["checkpoints"][0]["sha256"],
            one_dose.REPORT_EXECUTION_BINDING_FIELD: execution_binding,
        },
    )
    training_receipt = _write_one_dose_receipt(
        tmp_path,
        contract_path=contract_path,
        contract=contract,
        candidate=candidate,
        report=report_path,
        command=command,
        execution_binding=execution_binding,
    )
    calibration_sources = []
    for role, checkpoint, rmse in (
        ("candidate_calibration", candidate, 0.20),
        ("champion_calibration", champion, 0.21),
    ):
        source = tmp_path / f"{role}.json"
        _write_json(
            source,
            {
                "schema_version": "phase-sliced-value-calibration-v2",
                "checkpoint": str(checkpoint),
                "shard_dir": str(tmp_path / "shared_validation_corpus"),
                "value_readout": "scalar",
                "readout_provenance": {
                    "requested_readout": "scalar",
                    "trained_value_readouts": ["scalar"],
                    "optimizer_steps": 7,
                    "completed_epochs": 1,
                },
                "row_selection": {
                    "mode": "validation_seed_manifest",
                    "held_out_filter_applied": True,
                    "validation_fraction": 0.05,
                    "validation_seed": 17,
                    "validation_game_seed_ranges": [],
                    "seed_manifest_sha256": "sha256:" + "9" * 64,
                    "configured_game_seed_count": 256,
                    "observed_game_seed_count": 256,
                    "observed_row_count": 4096,
                },
                "global": {"n": 4096, "value_rmse": rmse},
            },
        )
        calibration_sources.append((role, source))
    internal_games = [
        {"pair_id": pair, "search_won": True, "candidate_won": True}
        for pair in range(200)
        for _orientation in range(2)
    ]
    pair_scores, pair_diagnostics = promotion.pair_scores_from_h2h_games(internal_games)
    pentanomial = promotion.evaluate_pentanomial_sprt(
        pair_scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    assert pentanomial["decision"] == "H1"
    typed_config = {
        "pipeline": "eval",
        "schema_version": 6,
        "fields": {
            **evidence_semantics,
            "mode": "cross_net",
            "candidate": str(candidate),
            "baseline": str(champion),
            "public_observation": True,
            "information_set_search": True,
            "determinization_particles": 4,
            "determinization_min_simulations": 32,
            "candidate_n_full": 128,
            "baseline_n_full": 128,
            "candidate_c_scale": candidate_search_config["c_scale"],
            "baseline_c_scale": champion_search_config["c_scale"],
            "candidate_n_full_wide": None,
            "baseline_n_full_wide": None,
            "candidate_n_full_wide_threshold": None,
            "baseline_n_full_wide_threshold": None,
            "candidate_value_readout": "scalar",
            "baseline_value_readout": "scalar",
        },
    }
    config_digest = hashlib.sha256(promotion._canonical_bytes(typed_config)).hexdigest()
    internal_source = tmp_path / "internal_h2h.raw.json"
    _write_json(
        internal_source,
        {
            "candidate_checkpoint": str(candidate),
            "baseline_checkpoint": str(champion),
            "typed_config": typed_config,
            "config_hash": "sha256:" + config_digest[:16],
            "full_config_hash": "sha256:" + config_digest,
            "candidate_value_readout": "scalar",
            "baseline_value_readout": "scalar",
            "public_observation": True,
            "information_set_search": True,
            "determinization_particles": 4,
            "determinization_min_simulations": 32,
            "search_budgets_by_role": {
                role: {
                    "n_full": 128,
                    "n_full_wide": None,
                    "n_full_wide_threshold": None,
                }
                for role in ("candidate", "baseline")
            },
            "complete_pairs": 200,
            "games_played": 400,
            "games_with_winner": 400,
            "games_truncated": 0,
            "errors": [],
            "games": internal_games,
            "pair_diagnostics": pair_diagnostics,
            "pentanomial_sprt": pentanomial,
            "verdict": "H1",
        },
    )

    external_sources = []
    for role, checkpoint, win_rate, external_search_config in (
        ("candidate_panel", candidate, 0.55, candidate_search_config),
        ("champion_panel", champion, 0.54, champion_search_config),
    ):
        external_games = [
            {
                "pair_id": pair,
                "game_seed": 8_100_000 + pair,
                "orientation": orientation,
                "candidate_won": game_index < int(win_rate * 1_000),
            }
            for game_index, (pair, orientation) in enumerate(
                (pair, orientation)
                for pair in range(500)
                for orientation in ("candidate_first", "candidate_second")
            )
        ]
        normalized_external_games = [
            {**game, "search_won": game["candidate_won"]} for game in external_games
        ]
        external_pair_scores, external_pair_diagnostics = (
            promotion.pair_scores_from_h2h_games(normalized_external_games)
        )
        external_pentanomial = promotion.evaluate_pentanomial_sprt(
            external_pair_scores,
            elo0=-10.0,
            elo1=15.0,
            alpha=0.05,
            beta=0.05,
        )
        candidate_wins = sum(bool(game["candidate_won"]) for game in external_games)
        source = tmp_path / f"{role}.raw.json"
        _write_json(
            source,
            {
                "stratum": "neutral-harness",
                "harness": "catanatron_native_engine",
                "baseline_bot": "catanatron_value",
                "mode": "search",
                "public_observation": True,
                "information_set_search": True,
                "determinization_particles": 4,
                "determinization_min_simulations": 32,
                "candidate_value_readout": "scalar",
                "trained_value_readouts": ["scalar"],
                "n_full": 128,
                "n_full_wide": None,
                "map_kind": "TOURNAMENT",
                "search_config": external_search_config,
                "gate_config": "flywheel",
                "pairs_requested": 500,
                "games_requested": 1000,
                "games_played": 1000,
                "games_with_winner": 1000,
                "games_truncated": 0,
                "games": external_games,
                "candidate_checkpoint": str(checkpoint),
                "candidate_checkpoint_md5": promotion._md5(checkpoint),
                "complete_pairs": 500,
                "candidate_win_rate": win_rate,
                "candidate_wins": candidate_wins,
                "baseline_wins": 1_000 - candidate_wins,
                "pair_diagnostics": external_pair_diagnostics,
                "pentanomial_sprt": external_pentanomial,
                "verdict": external_pentanomial["decision"],
                "errors": [],
                "worker_errors": [],
                "games_engine_divergence": 0,
            },
        )
        external_sources.append((role, source))

    high_regret_scope = tmp_path / "high-regret-worker"
    high_regret_scope.mkdir()
    high_regret_source_shard = high_regret_scope / "source-shard.npz"
    np.savez(
        high_regret_source_shard,
        game_seed=np.arange(7_000_000, 7_000_200, dtype=np.int64),
        decision_index=np.zeros(200, dtype=np.int32),
        action_taken=np.arange(200, dtype=np.int32),
    )
    high_regret_source_manifest = tmp_path / "high_regret.source.npz"
    np.savez(
        high_regret_source_manifest,
        shard_paths=np.asarray([str(high_regret_source_shard)]),
        shard_id=np.zeros(200, dtype=np.int32),
        row_index=np.arange(200, dtype=np.int32),
        game_seed=np.arange(7_000_000, 7_000_200, dtype=np.int64),
        decision_index=np.zeros(200, dtype=np.int32),
    )
    scope_digest, scope_count = scope_inventory_sha256(high_regret_scope)
    high_regret_suite = tmp_path / "high_regret.suite.json"
    high_regret_suite_payload = {
        "schema_version": promotion.HIGH_REGRET_SUITE_SCHEMA,
        "suite": "held_out_high_regret",
        "held_out": True,
        "source_manifest": _checkpoint_ref(high_regret_source_manifest),
        "selection": {
            "algorithm": "stable-hash-holdout-stratified-regret-v1",
            "holdout_fraction": 0.1,
            "holdout_seed": 17,
            "eligible_unique_states": 200,
            "selected_pairs": 200,
            "stratum_min_pairs": 20,
            "selected_by_stratum": {
                "phase:opening": 20,
                "phase:robber_dev": 20,
                "phase:chance": 20,
                "phase:build_trade": 20,
                "41+": 20,
            },
            "replay_preflight": {
                "contract": REPLAY_CONTRACT,
                "candidate_states": 200,
                "replay_complete_states": 200,
                "rejected_bad_source": 0,
                "rejected_noncontiguous": 0,
            },
        },
        "states": [
            {
                "pair_id": pair,
                "shard_path": str(high_regret_source_shard),
                "shard_id": 0,
                "row_index": pair,
                "game_seed": 7_000_000 + pair,
                "decision_index": 0,
                "phase": (
                    "BUILD_INITIAL_SETTLEMENT",
                    "MOVE_ROBBER",
                    "ROLL",
                    "BUILD_ROAD",
                )[pair % 4],
                "legal_count": 54 if pair < 20 else 12,
                "regret_score": 1.0,
                "replay_source": {
                    "contract": REPLAY_CONTRACT,
                    "scope": str(high_regret_scope),
                    "scope_inventory_sha256": scope_digest,
                    "scope_shard_count": scope_count,
                },
            }
            for pair in range(200)
        ],
    }
    high_regret_suite_payload["suite_sha256"] = promotion._digest_value(
        high_regret_suite_payload
    )
    _write_json(high_regret_suite, high_regret_suite_payload)
    high_regret_games = [
        {
            "pair_id": pair,
            "orientation": orientation,
            "candidate_won": True,
            "truncated": False,
            "archived_game_seed": 7_000_000 + pair,
            "archived_decision_index": 0,
            "buckets": ["phase:BUILD", "close"],
        }
        for pair in range(200)
        for orientation in ("candidate_first", "candidate_second")
    ]
    normalized_high_regret_games = [
        {**game, "search_won": game["candidate_won"]} for game in high_regret_games
    ]
    high_pair_scores, high_pair_diagnostics = promotion.pair_scores_from_h2h_games(
        normalized_high_regret_games
    )
    high_pentanomial = promotion.evaluate_pentanomial_sprt(
        high_pair_scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    assert high_pentanomial["decision"] == "H1"
    high_regret_report = tmp_path / "high_regret.report.json"
    _write_json(
        high_regret_report,
        {
            "schema_version": promotion.HIGH_REGRET_REPORT_SCHEMA,
            "suite": "held_out_high_regret",
            "held_out": True,
            "suite_manifest": _checkpoint_ref(high_regret_suite),
            "candidate": _checkpoint_ref(candidate),
            "champion": _checkpoint_ref(champion),
            "evaluation_config": {
                **evidence_semantics,
                "c_scale": candidate_search_config["c_scale"],
                "candidate_c_scale": candidate_search_config["c_scale"],
                "baseline_c_scale": champion_search_config["c_scale"],
                "candidate_n_full": 128,
                "baseline_n_full": 128,
                "candidate_n_full_wide": None,
                "baseline_n_full_wide": None,
                "candidate_n_full_wide_threshold": None,
                "baseline_n_full_wide_threshold": None,
                "candidate_value_readout": "scalar",
                "baseline_value_readout": "scalar",
            },
            "errors": [],
            "games": high_regret_games,
            "pentanomial_sprt": high_pentanomial,
            "pair_diagnostics": high_pair_diagnostics,
        },
    )
    high_regret_source = tmp_path / "high_regret.raw.json"
    _write_json(
        high_regret_source,
        {
            "schema_version": promotion.HIGH_REGRET_SCHEMA,
            "suite": "held_out_high_regret",
            "held_out": True,
            "candidate": _checkpoint_ref(candidate),
            "champion": _checkpoint_ref(champion),
            "passed": True,
            "verdict": "H1",
            "complete_pairs": 200,
            "errors": [],
            "report": _checkpoint_ref(high_regret_report),
            "suite_manifest": _checkpoint_ref(high_regret_suite),
            "pentanomial_sprt": high_pentanomial,
            "pair_diagnostics": high_pair_diagnostics,
        },
    )
    phase_buckets = ("opening", "robber_dev", "chance", "build_trade")

    def fixture_buckets(pair: int) -> list[str]:
        phase = phase_buckets[pair % len(phase_buckets)]
        labels = [f"phase:{phase}", "blowout" if pair % 2 == 0 else "close"]
        if phase == "opening":
            labels.append("opening")
        if pair < 50:
            labels.append("41+")
        return sorted(labels)

    bucket_games = [
        {
            "pair_id": pair,
            "orientation": orientation,
            "candidate_won": True,
            "buckets": fixture_buckets(pair),
        }
        for pair in range(100)
        for orientation in ("candidate_first", "candidate_second")
    ]
    bucket_report = tmp_path / "bucket_veto.report.json"
    _write_json(
        bucket_report,
        {
            "schema_version": promotion.BUCKET_GAME_REPORT_SCHEMA,
            "candidate": _checkpoint_ref(candidate),
            "champion": _checkpoint_ref(champion),
            "errors": [],
            "games": bucket_games,
        },
    )
    bucket_source = tmp_path / "bucket_veto.raw.json"
    _write_json(
        bucket_source,
        {
            "schema_version": promotion.BUCKET_VETO_SCHEMA,
            "candidate": _checkpoint_ref(candidate),
            "champion": _checkpoint_ref(champion),
            "veto": False,
            "veto_buckets": [],
            "per_bucket": {
                "41+": {"status": "pass", "n": 100, "winrate": 1.0},
                "blowout": {"status": "pass", "n": 100, "winrate": 1.0},
                "close": {"status": "pass", "n": 100, "winrate": 1.0},
                "opening": {"status": "pass", "n": 50, "winrate": 1.0},
                "phase:build_trade": {
                    "status": "pass",
                    "n": 50,
                    "winrate": 1.0,
                },
                "phase:chance": {"status": "pass", "n": 50, "winrate": 1.0},
                "phase:opening": {"status": "pass", "n": 50, "winrate": 1.0},
                "phase:robber_dev": {
                    "status": "pass",
                    "n": 50,
                    "winrate": 1.0,
                },
            },
            "report": _checkpoint_ref(bucket_report),
        },
    )
    evidence_specs = {
        "mechanism_calibration": (
            calibration_sources,
            "pass",
            {"value_readout": "scalar", "max_rmse_regression": 0.02},
        ),
        "internal_h2h": ([("internal_h2h", internal_source)], "H1", {}),
        "external_panel": (
            external_sources,
            "pass",
            {"max_win_rate_regression": 0.02},
        ),
        "high_regret": ([("high_regret", high_regret_source)], "pass", {}),
        "bucket_veto": ([("bucket_veto", bucket_source)], "pass", {}),
    }
    evidence = []
    for kind in sorted(promotion.REQUIRED_EVIDENCE_KINDS):
        sources, verdict, result = evidence_specs[kind]
        evidence_path = tmp_path / f"{kind}.json"
        _write_evidence_envelope(
            evidence_path,
            kind=kind,
            contract=contract,
            candidate=candidate,
            champion=champion,
            sources=sources,
            verdict=verdict,
            result=result,
        )
        evidence.append(
            {
                "kind": kind,
                "path": str(evidence_path),
                "sha256": promotion._sha256(evidence_path),
            }
        )
    next_count = promotion_count + 1
    nth_required = next_count % 3 == 0
    adjudication = {
        "schema_version": promotion.ADJUDICATION_SCHEMA,
        "passed": True,
        "decision": "promote",
        "contract_sha256": contract["contract_sha256"],
        "candidate": {
            "path": str(candidate),
            "sha256": promotion._sha256(candidate),
            "version": 5,
            "agent_identity": promotion._agent_identity(
                _checkpoint_ref(candidate), candidate_search_config
            ),
            "training_report": {
                "path": str(report_path),
                "sha256": promotion._sha256(report_path),
            },
        },
        "champion": {
            "path": str(champion),
            "sha256": promotion._sha256(champion),
            "version": 4,
            "agent_identity": promotion._agent_identity(
                _checkpoint_ref(champion), champion_search_config
            ),
        },
        "checks": {name: True for name in promotion.REQUIRED_CHECKS},
        "nth_confirmation_required": nth_required,
        "nth_confirmation_passed": True if nth_required else False,
        "evidence": evidence,
    }
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    adjudication_path = tmp_path / "adjudication.json"
    _write_json(adjudication_path, adjudication)
    return {
        "champion": champion,
        "candidate": candidate,
        "registry": registry_path,
        "pointer": pointer,
        "contract_path": contract_path,
        "contract": contract,
        "adjudication": adjudication_path,
        "report": report_path,
        "training_receipt": training_receipt,
        "receipt": tmp_path / "promotion.receipt.json",
        "lock": registry_path.with_suffix(registry_path.suffix + ".a1.lock"),
    }


def test_promoted_candidate_operator_becomes_next_cycle_incumbent_operator(
    tmp_path: Path,
) -> None:
    """A .10 promotion must never be silently reclassified as champion@.03."""
    old = tmp_path / "gen3.pt"
    promoted = tmp_path / "f7.pt"
    old.write_bytes(b"gen3")
    promoted.write_bytes(b"f7")
    first_contract = _contract(producer=old, c_scale=0.03)
    promoted_config = promotion._candidate_search_config(first_contract)
    assert promoted_config["c_scale"] == 0.10
    promoted_identity = promotion._agent_identity(
        _checkpoint_ref(promoted), promoted_config
    )

    registry = ChampionRegistry(tmp_path / "registry.json")
    registry.set_role(
        "generator_champion",
        promoted,
        expected_md5=promotion._md5(promoted),
        version=5,
        provenance={
            "a1_candidate_agent_identity_sha256": promoted_identity[
                "agent_identity_sha256"
            ],
            "a1_candidate_search_config": promoted_config,
        },
        reason="simulated committed first promotion",
    )
    registry.record_promotion("generator_champion")
    registry.save()

    # The next candidate may tune its operator; that must not reclassify the
    # incumbent, whose exact promoted .10 identity remains authoritative.
    next_contract = _contract(producer=promoted, c_scale=0.15)
    next_contract["contract_id"] = "a1-next-cycle"
    next_contract["contract_sha256"] = "sha256:" + "2" * 64
    incumbent_config = promotion._incumbent_search_config(
        next_contract,
        registry=ChampionRegistry.load(registry.path),
        champion_path=promoted.resolve(),
        champion_sha256=promotion._sha256(promoted),
    )
    assert incumbent_config["c_scale"] == 0.10
    assert promotion._candidate_search_config(next_contract)["c_scale"] == 0.15

    checkpoint_ref = _checkpoint_ref(promoted)
    silently_downgraded = dict(incumbent_config, c_scale=0.03)
    identity = promotion._agent_identity(checkpoint_ref, silently_downgraded)
    with pytest.raises(
        promotion.PromotionError, match="search_config sealed A1 semantic drift"
    ):
        promotion._verify_agent_identity(
            identity,
            expected_search_config=incumbent_config,
            checkpoint_path=promoted.resolve(),
            checkpoint_sha256=checkpoint_ref["sha256"],
            base=tmp_path,
            where="next-cycle champion.agent_identity",
        )


def test_next_cycle_incumbent_without_bound_operator_identity_fails_closed(
    tmp_path: Path,
) -> None:
    """Only the exact historical bootstrap may lack incumbent provenance."""
    promoted = tmp_path / "f7.pt"
    promoted.write_bytes(b"f7")
    registry = ChampionRegistry(tmp_path / "registry.json")
    registry.set_role(
        "generator_champion",
        promoted,
        expected_md5=promotion._md5(promoted),
        version=5,
        provenance={},
        reason="simulated malformed promotion",
    )
    registry.save()

    next_contract = _contract(producer=promoted, c_scale=0.10)
    next_contract["contract_id"] = "a1-next-cycle"
    next_contract["contract_sha256"] = "sha256:" + "2" * 64
    with pytest.raises(
        promotion.PromotionError, match="no bound agent search identity"
    ):
        promotion._incumbent_search_config(
            next_contract,
            registry=ChampionRegistry.load(registry.path),
            champion_path=promoted.resolve(),
            champion_sha256=promotion._sha256(promoted),
        )


def _verify(fixture: dict):
    def verify(path: Path, *, require_all_job_claims: bool = False):
        assert path == fixture["contract_path"]
        assert require_all_job_claims is True
        return fixture["contract"]

    return verify


def _execute(fixture: dict, *, go: bool):
    return promotion.execute_promotion(
        registry_path=fixture["registry"],
        current_pointer=fixture["pointer"],
        contract_lock=fixture["contract_path"],
        adjudication_path=fixture["adjudication"],
        training_receipt=fixture["training_receipt"],
        receipt_path=fixture["receipt"],
        reason="A1 typed promotion",
        lock_path=fixture["lock"],
        go=go,
        verify_lock_fn=_verify(fixture),
    )


def _mutate_training_receipt(fixture: dict, mutate) -> dict:
    path = fixture["training_receipt"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("receipt_sha256")
    mutate(payload)
    payload["receipt_sha256"] = promotion._digest_value(payload)
    _write_json(path, payload)
    return payload


def _convert_training_receipt_to_sealed_retry(fixture: dict) -> None:
    receipt_path = fixture["training_receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    claim_path = Path(receipt["claim"])
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    identity_evidence = {
        "schema_version": one_dose.RETRY_IDENTITY_SCHEMA,
        "repair_kind": one_dose.RETRY_REPAIR_KIND,
        "parent_contract_sha256": fixture["contract"]["contract_sha256"],
        "parent": {
            "claim": "/sealed/failed-r1.claim.json",
            "claim_file_sha256": "sha256:" + "4" * 64,
            "claim_state_sha256": "sha256:" + "5" * 64,
            "receipt": "/sealed/failed-r1.receipt.json",
            "receipt_file_sha256": "sha256:" + "6" * 64,
            "receipt_sha256": "sha256:" + "7" * 64,
            "command_sha256": "sha256:" + "8" * 64,
            "returncode": 1,
            "failure": "ExecutorError: train_bc exited nonzero: 1",
        },
    }
    identity_sha = promotion._digest_value(identity_evidence)
    retry_contract = {
        "schema_version": one_dose.RETRY_CONTRACT_SCHEMA,
        "retry_identity": identity_evidence,
        "retry_identity_sha256": identity_sha,
        "parent": {
            **identity_evidence["parent"],
            "pre_optimizer_proof": {
                "kind": "replayed_init_checkpoint_architecture_preflight",
                "mismatches": ["graph_layers checkpoint=6 cli=4"],
                "optimizer_steps": 0,
                "outputs": None,
            },
        },
        "preserved_bindings": {},
        "retry": {},
    }
    retry_contract["retry_contract_sha256"] = promotion._digest_value(retry_contract)
    retry_contract_path = receipt_path.with_name("learner-retry.contract.json")
    _write_json(retry_contract_path, retry_contract)
    reference = {
        "path": str(retry_contract_path),
        "file_sha256": promotion._sha256(retry_contract_path),
        "retry_contract_sha256": retry_contract["retry_contract_sha256"],
    }

    claim.pop("state_sha256")
    claim["schema_version"] = one_dose.RETRY_CLAIM_SCHEMA
    claim["claim_identity_sha256"] = identity_sha
    claim["retry_contract"] = reference
    claim["state_sha256"] = one_dose._value_sha256(claim)
    _write_json(claim_path, claim)

    receipt.pop("receipt_sha256")
    receipt["schema_version"] = one_dose.RETRY_RECEIPT_SCHEMA
    receipt["claim_identity_sha256"] = identity_sha
    receipt["retry_contract"] = reference
    receipt["claim_state_sha256"] = claim["state_sha256"]
    receipt["receipt_sha256"] = one_dose._value_sha256(receipt)
    _write_json(receipt_path, receipt)


def test_dry_run_binds_v3_one_dose_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    plan = _execute(fixture, go=False)

    assert plan["schema_version"] == promotion.RECEIPT_SCHEMA
    assert plan["training_receipt"] == {
        "path": str(fixture["training_receipt"]),
        "sha256": promotion._sha256(fixture["training_receipt"]),
        "receipt_sha256": json.loads(
            fixture["training_receipt"].read_text(encoding="utf-8")
        )["receipt_sha256"],
        "claim": str(tmp_path / "one-dose.claim.json"),
        "claim_state_sha256": json.loads(
            fixture["training_receipt"].read_text(encoding="utf-8")
        )["claim_state_sha256"],
        "execution_binding_sha256": json.loads(
            fixture["training_receipt"].read_text(encoding="utf-8")
        )["outputs"]["execution_binding_sha256"],
    }


def test_promotion_receipt_binds_deployed_agent_identities(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    plan = _execute(fixture, go=False)

    candidate = plan["candidate"]["agent_identity"]
    champion = plan["champion"]["agent_identity"]
    assert candidate["search_config"]["c_scale"] == 0.10
    assert champion["search_config"]["c_scale"] == 0.03
    assert {
        key: value
        for key, value in candidate["search_config"].items()
        if key != "c_scale"
    } == {
        key: value
        for key, value in champion["search_config"].items()
        if key != "c_scale"
    }
    for identity in (candidate, champion):
        unhashed = dict(identity)
        declared = unhashed.pop("agent_identity_sha256")
        assert declared == promotion._digest_value(unhashed)


def test_adjudication_cannot_rebind_candidate_to_champion_search(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    adjudication = json.loads(fixture["adjudication"].read_text())
    identity = adjudication["candidate"]["agent_identity"]
    identity["search_config"]["c_scale"] = 0.03
    identity.pop("agent_identity_sha256")
    identity["agent_identity_sha256"] = promotion._digest_value(identity)
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(fixture["adjudication"], adjudication)

    with pytest.raises(promotion.PromotionError, match="sealed A1 semantic drift"):
        _execute(fixture, go=False)


def test_dry_run_accepts_schema_separated_sealed_retry_receipt(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _convert_training_receipt_to_sealed_retry(fixture)

    plan = _execute(fixture, go=False)

    assert plan["status"] == "dry_run"
    assert plan["training_receipt"]["path"] == str(fixture["training_receipt"])


def test_direct_promotion_rejects_legacy_one_dose_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _mutate_training_receipt(
        fixture,
        lambda payload: payload.__setitem__(
            "schema_version", "a1-one-dose-training-receipt-v2"
        ),
    )

    with pytest.raises(promotion.PromotionError, match="receipt schema"):
        _execute(fixture, go=False)


def test_direct_promotion_rejects_environment_binding_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def drift_environment(payload: dict) -> None:
        environment = payload["execution_binding"]["environment"]
        environment["PYTHONHASHSEED"] = "123"
        payload["execution_binding"]["environment_sha256"] = promotion._digest_value(
            environment
        )
        payload["outputs"]["execution_binding_sha256"] = promotion._digest_value(
            payload["execution_binding"]
        )

    _mutate_training_receipt(fixture, drift_environment)

    with pytest.raises(promotion.PromotionError, match="exact allowlist"):
        _execute(fixture, go=False)


def test_direct_promotion_rejects_candidate_not_created_by_dose(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    other = tmp_path / "other.pt"
    other.write_bytes(b"other candidate")

    def swap_candidate(payload: dict) -> None:
        payload["outputs"]["checkpoint"] = str(other)
        payload["outputs"]["checkpoint_sha256"] = promotion._sha256(other)

    _mutate_training_receipt(fixture, swap_candidate)

    with pytest.raises(promotion.PromotionError, match="candidate differs"):
        _execute(fixture, go=False)


def test_direct_promotion_rejects_missing_durable_claim(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (tmp_path / "one-dose.claim.json").unlink()

    with pytest.raises(promotion.PromotionError, match="durable claim"):
        _execute(fixture, go=False)


def _mutate_evidence_source(fixture: dict, *, kind: str, role: str, mutate) -> None:
    adjudication = json.loads(fixture["adjudication"].read_text())
    evidence_ref = next(
        item for item in adjudication["evidence"] if item["kind"] == kind
    )
    evidence_path = Path(evidence_ref["path"])
    envelope = json.loads(evidence_path.read_text())
    source_ref = next(item for item in envelope["sources"] if item["role"] == role)
    source_path = Path(source_ref["path"])
    source = json.loads(source_path.read_text())
    mutate(source)
    _write_json(source_path, source)
    source_ref["sha256"] = promotion._sha256(source_path)
    envelope.pop("evidence_sha256")
    envelope["evidence_sha256"] = promotion._digest_value(envelope)
    _write_json(evidence_path, envelope)
    evidence_ref["sha256"] = promotion._sha256(evidence_path)
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(fixture["adjudication"], adjudication)


def test_dry_run_is_read_only_and_attests_global_n128(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    before_registry = fixture["registry"].read_bytes()
    before_pointer = fixture["pointer"].read_bytes()

    result = _execute(fixture, go=False)

    assert result["status"] == "dry_run"
    assert result["contract"]["n_full"] == 128
    assert result["contract"]["n_full_wide"] is None
    assert result["fleet_ckpt_updated"] is False
    assert fixture["registry"].read_bytes() == before_registry
    assert fixture["pointer"].read_bytes() == before_pointer
    assert not fixture["receipt"].exists()


def test_dry_run_accepts_hash_replayed_fleet_pooled_internal_source(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    shard_path = tmp_path / "internal-h2h-shard.json"

    def make_pooled(source: dict) -> None:
        _write_json(shard_path, source)
        typed = source.pop("typed_config")
        short_hash = source.pop("config_hash")
        full_hash = source.pop("full_config_hash")
        effective = dict(typed["fields"])
        effective.pop("candidate")
        effective.pop("baseline")
        source["candidate_checkpoint_sha256"] = promotion._sha256(fixture["candidate"])
        source["baseline_checkpoint_sha256"] = promotion._sha256(fixture["champion"])
        source["effective_search_config"] = effective
        source["fleet_merge"] = {
            "schema_version": promotion.FLEET_EVALUATION_POOL_SCHEMA,
            "kind": "internal_h2h",
            "candidate": _checkpoint_ref(fixture["candidate"]),
            "champion": _checkpoint_ref(fixture["champion"]),
            "sources": [_checkpoint_ref(shard_path)],
            "seed_intervals": [
                {
                    "base_seed": 9_000_000,
                    "end_seed": 9_000_200,
                    "path": str(shard_path.resolve()),
                }
            ],
            "shard_config_hashes": [
                {
                    "path": str(shard_path.resolve()),
                    "config_hash": short_hash,
                    "full_config_hash": full_hash,
                }
            ],
            "effective_search_config_sha256": promotion._digest_value(effective),
        }

    _mutate_evidence_source(
        fixture,
        kind="internal_h2h",
        role="internal_h2h",
        mutate=make_pooled,
    )

    assert _execute(fixture, go=False)["status"] == "dry_run"


def test_fleet_pooled_internal_refuses_effective_config_digest_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    shard_path = tmp_path / "internal-h2h-shard.json"

    def make_bad_pool(source: dict) -> None:
        _write_json(shard_path, source)
        typed = source.pop("typed_config")
        short_hash = source.pop("config_hash")
        full_hash = source.pop("full_config_hash")
        effective = dict(typed["fields"])
        effective.pop("candidate")
        effective.pop("baseline")
        source["candidate_checkpoint_sha256"] = promotion._sha256(fixture["candidate"])
        source["baseline_checkpoint_sha256"] = promotion._sha256(fixture["champion"])
        source["effective_search_config"] = effective
        source["fleet_merge"] = {
            "schema_version": promotion.FLEET_EVALUATION_POOL_SCHEMA,
            "kind": "internal_h2h",
            "candidate": _checkpoint_ref(fixture["candidate"]),
            "champion": _checkpoint_ref(fixture["champion"]),
            "sources": [_checkpoint_ref(shard_path)],
            "seed_intervals": [
                {
                    "base_seed": 9_000_000,
                    "end_seed": 9_000_200,
                    "path": str(shard_path.resolve()),
                }
            ],
            "shard_config_hashes": [
                {
                    "path": str(shard_path.resolve()),
                    "config_hash": short_hash,
                    "full_config_hash": full_hash,
                }
            ],
            "effective_search_config_sha256": "sha256:" + "0" * 64,
        }

    _mutate_evidence_source(
        fixture,
        kind="internal_h2h",
        role="internal_h2h",
        mutate=make_bad_pool,
    )

    with pytest.raises(promotion.PromotionError, match="effective-search config"):
        _execute(fixture, go=False)


def test_go_updates_generator_and_pointer_with_committed_receipt(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    public_before = ChampionRegistry.load(fixture["registry"]).get_role(
        "public_champion"
    )

    receipt = _execute(fixture, go=True)

    assert receipt["status"] == "committed"
    assert receipt["fleet_ckpt_updated"] is False
    registry = ChampionRegistry.load(fixture["registry"])
    generator = registry.get_role("generator_champion")
    assert generator is not None
    assert Path(generator.checkpoint_path).resolve() == fixture["candidate"].resolve()
    assert generator.version == 5
    assert generator.provenance["a1_one_dose_training_receipt"] == str(
        fixture["training_receipt"]
    )
    assert generator.provenance[
        "a1_one_dose_training_receipt_sha256"
    ] == promotion._sha256(fixture["training_receipt"])
    assert (
        generator.provenance["a1_one_dose_execution_binding_sha256"]
        == json.loads(fixture["training_receipt"].read_text(encoding="utf-8"))[
            "outputs"
        ]["execution_binding_sha256"]
    )
    assert registry.promotion_count() == 1
    assert any(
        Path(entry.checkpoint_path).resolve() == fixture["champion"].resolve()
        for entry in registry.opponent_pool()
    )
    assert registry.get_role("public_champion") == public_before
    assert fixture["pointer"].read_text().strip() == str(fixture["candidate"].resolve())
    saved = json.loads(fixture["receipt"].read_text())
    assert saved["status"] == "committed"
    assert Path(saved["rollback"]["registry_backup"]).is_file()
    assert Path(saved["rollback"]["current_backup"]).is_file()


def test_recovery_is_dry_run_then_restores_exact_before_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    registry_before = fixture["registry"].read_bytes()
    pointer_before = fixture["pointer"].read_bytes()
    _execute(fixture, go=True)

    dry = promotion.recover_transaction(
        receipt_path=fixture["receipt"], lock_path=fixture["lock"], go=False
    )
    assert dry["status"] == "recovery_dry_run"
    assert fixture["registry"].read_bytes() != registry_before

    recovered = promotion.recover_transaction(
        receipt_path=fixture["receipt"], lock_path=fixture["lock"], go=True
    )
    assert recovered["status"] == "recovered"
    assert fixture["registry"].read_bytes() == registry_before
    assert fixture["pointer"].read_bytes() == pointer_before
    assert json.loads(fixture["receipt"].read_text())["status"] == "recovered"


def test_recovery_accepts_pre_v2_promotion_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _execute(fixture, go=True)
    receipt = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    receipt.pop("receipt_sha256")
    receipt.pop("training_receipt")
    receipt["schema_version"] = promotion.LEGACY_RECEIPT_SCHEMA
    receipt["receipt_sha256"] = promotion._digest_value(receipt)
    _write_json(fixture["receipt"], receipt)

    result = promotion.recover_transaction(receipt_path=fixture["receipt"], go=False)

    assert result["status"] == "recovery_dry_run"


def test_global_n196_contract_is_rejected_before_mutation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, n_full=196)
    before = fixture["registry"].read_bytes()

    with pytest.raises(promotion.PromotionError, match="n_full=128"):
        _execute(fixture, go=True)

    assert fixture["registry"].read_bytes() == before
    assert not fixture["receipt"].exists()


def test_external_panel_without_information_set_attestation_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    source = tmp_path / "candidate_panel.raw.json"
    payload = json.loads(source.read_text())
    payload["information_set_search"] = False

    with pytest.raises(promotion.PromotionError, match="information-set recipe"):
        promotion._verify_external_panel_source(
            payload,
            checkpoint=fixture["candidate"],
            checkpoint_md5=promotion._md5(fixture["candidate"]),
            where="candidate external panel",
            sealed_semantics=promotion._sealed_evaluation_semantics(
                fixture["contract"]
            ),
            role="candidate",
            deployed_search_config=payload["search_config"],
        )


def test_candidate_hash_drift_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["candidate"].write_bytes(b"mutated after adjudication")

    with pytest.raises(promotion.PromotionError, match="artifact drift"):
        _execute(fixture, go=False)


def test_training_report_must_name_exact_candidate(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    report = json.loads(fixture["report"].read_text())
    report["checkpoint"] = str(fixture["champion"])
    _write_json(fixture["report"], report)
    adjudication = json.loads(fixture["adjudication"].read_text())
    adjudication["candidate"]["training_report"]["sha256"] = promotion._sha256(
        fixture["report"]
    )
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(fixture["adjudication"], adjudication)

    with pytest.raises(promotion.PromotionError, match="report checkpoint differs"):
        _execute(fixture, go=False)


def test_bucket_insufficient_data_is_a_binding_veto(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    adjudication = json.loads(fixture["adjudication"].read_text())
    evidence_ref = next(
        item for item in adjudication["evidence"] if item["kind"] == "bucket_veto"
    )
    evidence_path = Path(evidence_ref["path"])
    envelope = json.loads(evidence_path.read_text())
    source_path = Path(envelope["sources"][0]["path"])
    source = json.loads(source_path.read_text())
    source["per_bucket"]["41+"] = {
        "status": "insufficient_data",
        "n": 4,
        "winrate": 0.75,
    }
    _write_json(source_path, source)
    envelope["sources"][0]["sha256"] = promotion._sha256(source_path)
    envelope.pop("evidence_sha256")
    envelope["evidence_sha256"] = promotion._digest_value(envelope)
    _write_json(evidence_path, envelope)
    evidence_ref["sha256"] = promotion._sha256(evidence_path)
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(fixture["adjudication"], adjudication)

    with pytest.raises(promotion.PromotionError, match="do not replay"):
        _execute(fixture, go=False)


def test_bucket_cannot_launder_regression_with_pass_status(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    adjudication = json.loads(fixture["adjudication"].read_text())
    evidence_ref = next(
        item for item in adjudication["evidence"] if item["kind"] == "bucket_veto"
    )
    evidence_path = Path(evidence_ref["path"])
    envelope = json.loads(evidence_path.read_text())
    source_path = Path(envelope["sources"][0]["path"])
    source = json.loads(source_path.read_text())
    source["per_bucket"]["41+"] = {"status": "pass", "n": 80, "winrate": 0.44}
    _write_json(source_path, source)
    envelope["sources"][0]["sha256"] = promotion._sha256(source_path)
    envelope.pop("evidence_sha256")
    envelope["evidence_sha256"] = promotion._digest_value(envelope)
    _write_json(evidence_path, envelope)
    evidence_ref["sha256"] = promotion._sha256(evidence_path)
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(fixture["adjudication"], adjudication)

    with pytest.raises(promotion.PromotionError, match="do not replay"):
        _execute(fixture, go=False)


@pytest.mark.parametrize(
    ("kind", "field"),
    [
        ("mechanism_calibration", "max_rmse_regression"),
        ("external_panel", "max_win_rate_regression"),
    ],
)
def test_evidence_cannot_launder_regression_with_its_own_tolerance(
    tmp_path: Path, kind: str, field: str
) -> None:
    fixture = _fixture(tmp_path)
    adjudication = json.loads(fixture["adjudication"].read_text())
    evidence_ref = next(
        item for item in adjudication["evidence"] if item["kind"] == kind
    )
    evidence_path = Path(evidence_ref["path"])
    envelope = json.loads(evidence_path.read_text())
    envelope["result"][field] = 1.0
    envelope.pop("evidence_sha256")
    envelope["evidence_sha256"] = promotion._digest_value(envelope)
    _write_json(evidence_path, envelope)
    evidence_ref["sha256"] = promotion._sha256(evidence_path)
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(fixture["adjudication"], adjudication)

    with pytest.raises(promotion.PromotionError, match="fixed policy"):
        _execute(fixture, go=False)


def test_calibration_comparison_rejects_different_validation_seed_cohorts(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def change_seed_manifest(source: dict) -> None:
        source["row_selection"]["seed_manifest_sha256"] = "sha256:" + "8" * 64

    _mutate_evidence_source(
        fixture,
        kind="mechanism_calibration",
        role="candidate_calibration",
        mutate=change_seed_manifest,
    )

    with pytest.raises(promotion.PromotionError, match="different cohorts"):
        _execute(fixture, go=False)


def _install_legacy_incumbent_bridge(fixture: dict, tmp_path: Path) -> Path:
    champion = fixture["champion"]
    historical = tmp_path / "gen3-historical-training-report.json"
    _write_json(
        historical,
        {
            "checkpoint": str(champion.resolve()),
            "checkpoint_sha256": promotion._sha256(champion),
            "steps_completed": 912,
            "epochs": 1,
        },
    )

    def add_bridge(source: dict) -> None:
        source["readout_provenance"]["optimizer_steps"] = None
        source["readout_provenance"]["completed_epochs"] = None
        source["legacy_incumbent_provenance"] = {
            "schema_version": promotion.LEGACY_INCUMBENT_PROVENANCE_SCHEMA,
            "contract_sha256": fixture["contract"]["contract_sha256"],
            "checkpoint_sha256": promotion._sha256(champion),
            "historical_training_report": _checkpoint_ref(historical),
        }

    _mutate_evidence_source(
        fixture,
        kind="mechanism_calibration",
        role="champion_calibration",
        mutate=add_bridge,
    )
    return historical


def test_exact_contract_incumbent_accepts_typed_legacy_calibration_bridge(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _install_legacy_incumbent_bridge(fixture, tmp_path)

    plan = _execute(fixture, go=False)

    assert plan["status"] == "dry_run"


def test_artifact_built_real_gen3_relative_bridge_passes_promotion_from_other_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    champion = tmp_path / "runs" / "bc" / "gen3_20260706" / "checkpoint.pt"
    fixture = _fixture(tmp_path, champion=champion)
    historical = tmp_path / "reports" / "archive" / "gen3-training-report.json"
    historical.parent.mkdir(parents=True)
    _write_json(
        historical,
        {
            "checkpoint": "runs/bc/gen3_20260706/checkpoint.pt",
            "checkpoint_sha256": promotion._sha256(champion),
            "steps_completed": 912,
            "epochs": 1,
        },
    )
    adjudication = json.loads(fixture["adjudication"].read_text())
    evidence_ref = next(
        item
        for item in adjudication["evidence"]
        if item["kind"] == "mechanism_calibration"
    )
    envelope = json.loads(Path(evidence_ref["path"]).read_text())
    source_ref = next(
        item for item in envelope["sources"] if item["role"] == "champion_calibration"
    )
    source_path = Path(source_ref["path"])
    calibration = json.loads(source_path.read_text())
    calibration["readout_provenance"]["optimizer_steps"] = None
    calibration["readout_provenance"]["completed_epochs"] = None
    _write_json(source_path, calibration)
    built = artifacts.build_legacy_incumbent_calibration_source(
        calibration_path=source_path,
        historical_training_report=historical,
        contract=fixture["contract"],
        champion=champion,
    )

    def install_built_bridge(source: dict) -> None:
        source.clear()
        source.update(built)

    _mutate_evidence_source(
        fixture,
        kind="mechanism_calibration",
        role="champion_calibration",
        mutate=install_built_bridge,
    )
    unrelated_cwd = tmp_path / "unrelated-cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    plan = _execute(fixture, go=False)

    assert plan["status"] == "dry_run"


def test_legacy_calibration_bridge_rejects_mutated_historical_report(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    historical = _install_legacy_incumbent_bridge(fixture, tmp_path)
    report = json.loads(historical.read_text())
    report["steps_completed"] = 913
    _write_json(historical, report)

    with pytest.raises(promotion.PromotionError, match="artifact drift"):
        _execute(fixture, go=False)


def test_candidate_cannot_use_legacy_incumbent_calibration_bridge(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    historical = tmp_path / "candidate-historical.json"
    _write_json(
        historical,
        {
            "checkpoint": str(fixture["candidate"].resolve()),
            "steps_completed": 1,
            "epochs": 1,
        },
    )

    def add_bridge(source: dict) -> None:
        source["readout_provenance"]["optimizer_steps"] = None
        source["readout_provenance"]["completed_epochs"] = None
        source["legacy_incumbent_provenance"] = {
            "schema_version": promotion.LEGACY_INCUMBENT_PROVENANCE_SCHEMA,
            "contract_sha256": fixture["contract"]["contract_sha256"],
            "checkpoint_sha256": promotion._sha256(fixture["candidate"]),
            "historical_training_report": _checkpoint_ref(historical),
        }

    _mutate_evidence_source(
        fixture,
        kind="mechanism_calibration",
        role="candidate_calibration",
        mutate=add_bridge,
    )
    with pytest.raises(promotion.PromotionError, match="optimizer_steps"):
        _execute(fixture, go=False)


def test_external_comparison_rejects_different_pair_seed_cohorts(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def change_pair_seed(source: dict) -> None:
        source["games"][0]["game_seed"] += 1_000_000

    _mutate_evidence_source(
        fixture,
        kind="external_panel",
        role="candidate_panel",
        mutate=change_pair_seed,
    )

    with pytest.raises(promotion.PromotionError, match="different cohorts/configs"):
        _execute(fixture, go=False)


def test_external_comparison_rejects_different_search_configs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def change_search_config(source: dict) -> None:
        source["search_config"]["c_scale"] = 0.3

    _mutate_evidence_source(
        fixture,
        kind="external_panel",
        role="candidate_panel",
        mutate=change_search_config,
    )

    with pytest.raises(promotion.PromotionError, match="sealed A1 semantic drift"):
        _execute(fixture, go=False)


def test_external_comparison_rejects_other_cohort_metadata_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    _mutate_evidence_source(
        fixture,
        kind="external_panel",
        role="candidate_panel",
        mutate=lambda source: source.__setitem__("gate_config", "drifted-gate"),
    )

    with pytest.raises(promotion.PromotionError, match="different cohorts/configs"):
        _execute(fixture, go=False)


def test_external_reports_cannot_omit_d6_threshold_attestation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    for role in ("candidate_panel", "champion_panel"):
        _mutate_evidence_source(
            fixture,
            kind="external_panel",
            role=role,
            mutate=lambda source: source["search_config"].pop(
                "symmetry_averaged_eval_threshold"
            ),
        )

    with pytest.raises(promotion.PromotionError, match="keys differ"):
        _execute(fixture, go=False)


def test_high_regret_report_cannot_launder_sealed_search_config(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        report["evaluation_config"]["c_scale"] = 0.3
        _write_json(report_path, report)
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="high_regret", role="high_regret", mutate=mutate
    )

    with pytest.raises(promotion.PromotionError, match="sealed A1 semantic drift"):
        _execute(fixture, go=False)


def test_high_regret_report_binds_candidate_and_incumbent_role_scales(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        report["evaluation_config"]["baseline_c_scale"] = report[
            "evaluation_config"
        ]["candidate_c_scale"]
        _write_json(report_path, report)
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="high_regret", role="high_regret", mutate=mutate
    )

    with pytest.raises(promotion.PromotionError, match="baseline_c_scale"):
        _execute(fixture, go=False)


def test_high_regret_report_rejects_forged_candidate_role_scale(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        report["evaluation_config"]["candidate_c_scale"] = 0.03
        _write_json(report_path, report)
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="high_regret", role="high_regret", mutate=mutate
    )

    with pytest.raises(promotion.PromotionError, match="candidate_c_scale"):
        _execute(fixture, go=False)


def _install_truncated_high_regret_pair(fixture: dict) -> None:
    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        for row in report["games"]:
            if row["orientation"] == "candidate_first":
                row["orientation"] = "candidate_red"
                row["candidate_color"] = "RED"
                row["baseline_color"] = "BLUE"
            else:
                row["orientation"] = "candidate_blue"
                row["candidate_color"] = "BLUE"
                row["baseline_color"] = "RED"
        game = next(
            game
            for game in report["games"]
            if game["pair_id"] == 0
            and game["orientation"] == "candidate_red"
        )
        game["candidate_won"] = None
        game["truncated"] = True
        normalized = [
            {**row, "search_won": row["candidate_won"]}
            for row in report["games"]
        ]
        scores, diagnostics = promotion.pair_scores_from_h2h_games(normalized)
        pentanomial = promotion.evaluate_pentanomial_sprt(
            scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
        )
        report["pair_diagnostics"] = diagnostics
        report["pentanomial_sprt"] = pentanomial
        _write_json(report_path, report)
        source["complete_pairs"] = 199
        source["pair_diagnostics"] = diagnostics
        source["pentanomial_sprt"] = pentanomial
        source["verdict"] = pentanomial["decision"]
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="high_regret", role="high_regret", mutate=mutate
    )


def test_transaction_independently_accepts_legitimate_truncated_high_regret_pair(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _install_truncated_high_regret_pair(fixture)

    plan = _execute(fixture, go=False)

    assert plan["status"] == "dry_run"


def test_transaction_rejects_none_outcome_without_truncation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        report["games"][0]["candidate_won"] = None
        _write_json(report_path, report)
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="high_regret", role="high_regret", mutate=mutate
    )

    with pytest.raises(promotion.PromotionError, match="inconsistent truncation"):
        _execute(fixture, go=False)


def test_transaction_rejects_forged_high_regret_orientation_color(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _install_truncated_high_regret_pair(fixture)

    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        report["games"][0]["candidate_color"] = "BLUE"
        _write_json(report_path, report)
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="high_regret", role="high_regret", mutate=mutate
    )

    with pytest.raises(promotion.PromotionError, match="orientation/color mismatch"):
        _execute(fixture, go=False)


def test_transaction_rejects_mixed_high_regret_orientation_encodings(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _install_truncated_high_regret_pair(fixture)

    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        game = report["games"][2]
        game["orientation"] = "candidate_first"
        game["candidate_color"] = "RED"
        game["baseline_color"] = "BLUE"
        _write_json(report_path, report)
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="high_regret", role="high_regret", mutate=mutate
    )

    with pytest.raises(promotion.PromotionError, match="mixes orientation encodings"):
        _execute(fixture, go=False)


def test_transaction_rejects_half_high_regret_pair(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        report["games"].pop(1)
        normalized = [
            {**row, "search_won": row["candidate_won"]}
            for row in report["games"]
        ]
        scores, diagnostics = promotion.pair_scores_from_h2h_games(normalized)
        pentanomial = promotion.evaluate_pentanomial_sprt(
            scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
        )
        report["pair_diagnostics"] = diagnostics
        report["pentanomial_sprt"] = pentanomial
        _write_json(report_path, report)
        source["complete_pairs"] = 199
        source["pair_diagnostics"] = diagnostics
        source["pentanomial_sprt"] = pentanomial
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="high_regret", role="high_regret", mutate=mutate
    )

    with pytest.raises(promotion.PromotionError, match="cover every suite pair twice"):
        _execute(fixture, go=False)


def test_transaction_rejects_inconsistent_truncated_complete_pair_count(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _install_truncated_high_regret_pair(fixture)
    _mutate_evidence_source(
        fixture,
        kind="high_regret",
        role="high_regret",
        mutate=lambda source: source.__setitem__("complete_pairs", 200),
    )

    with pytest.raises(promotion.PromotionError, match="paired statistics"):
        _execute(fixture, go=False)


def test_transaction_rejects_incomplete_bucket_pair(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        report["games"].pop(0)
        _write_json(report_path, report)
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="bucket_veto", role="bucket_veto", mutate=mutate
    )

    with pytest.raises(promotion.PromotionError, match="incomplete bucket pair"):
        _execute(fixture, go=False)


def test_high_regret_games_must_match_frozen_suite_state_identities(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        suite_path = Path(report["suite_manifest"]["path"])
        suite = json.loads(suite_path.read_text())
        suite["states"][0]["game_seed"] += 1
        suite.pop("suite_sha256")
        suite["suite_sha256"] = promotion._digest_value(suite)
        _write_json(suite_path, suite)
        suite_sha = promotion._sha256(suite_path)
        report["suite_manifest"]["sha256"] = suite_sha
        _write_json(report_path, report)
        source["suite_manifest"]["sha256"] = suite_sha
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="high_regret", role="high_regret", mutate=mutate
    )

    with pytest.raises(promotion.PromotionError, match="not bound to source manifest"):
        _execute(fixture, go=False)


def test_high_regret_promotion_rejects_replaced_replay_scope_bytes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    source_shard = tmp_path / "high-regret-worker/source-shard.npz"
    source_shard.write_bytes(b"forged easier archived trajectory")

    with pytest.raises(promotion.PromotionError, match="scope inventory drifted"):
        _execute(fixture, go=False)


def _set_external_panel_outcomes(
    fixture: dict,
    *,
    role: str,
    wins: int,
    elo0: float = -10.0,
    elo1: float = 15.0,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> None:
    def mutate(source: dict) -> None:
        for index, game in enumerate(source["games"]):
            game["candidate_won"] = index < wins
        source["candidate_wins"] = wins
        source["baseline_wins"] = len(source["games"]) - wins
        source["candidate_win_rate"] = wins / len(source["games"])
        normalized = [
            {**game, "search_won": game["candidate_won"]} for game in source["games"]
        ]
        scores, diagnostics = promotion.pair_scores_from_h2h_games(normalized)
        source["pair_diagnostics"] = diagnostics
        source["pentanomial_sprt"] = promotion.evaluate_pentanomial_sprt(
            scores, elo0=elo0, elo1=elo1, alpha=alpha, beta=beta
        )
        source["verdict"] = source["pentanomial_sprt"]["decision"]

    _mutate_evidence_source(fixture, kind="external_panel", role=role, mutate=mutate)


def test_external_absolute_h0_panels_pass_when_candidate_is_nonregressing(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _set_external_panel_outcomes(fixture, role="candidate_panel", wins=410)
    _set_external_panel_outcomes(fixture, role="champion_panel", wins=420)
    adjudication = json.loads(fixture["adjudication"].read_text())
    envelope_path = Path(
        next(
            item
            for item in adjudication["evidence"]
            if item["kind"] == "external_panel"
        )["path"]
    )
    envelope = json.loads(envelope_path.read_text())
    for source_ref in envelope["sources"]:
        source = json.loads(Path(source_ref["path"]).read_text())
        assert source["verdict"] == "H0"

    assert _execute(fixture, go=False)["status"] == "dry_run"


def test_external_comparative_regression_fails_at_fixed_two_percent_limit(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _set_external_panel_outcomes(fixture, role="candidate_panel", wins=390)
    _set_external_panel_outcomes(fixture, role="champion_panel", wins=420)

    with pytest.raises(promotion.PromotionError, match="allowed regression"):
        _execute(fixture, go=False)


@pytest.mark.parametrize(("candidate_wins", "passes"), [(410, True), (390, False)])
def test_external_sprt_threshold_overrides_cannot_change_comparative_eligibility(
    tmp_path: Path, candidate_wins: int, passes: bool
) -> None:
    fixture = _fixture(tmp_path)
    _set_external_panel_outcomes(
        fixture,
        role="candidate_panel",
        wins=candidate_wins,
        elo0=-400.0,
        elo1=-300.0,
        alpha=0.2,
        beta=0.2,
    )
    _set_external_panel_outcomes(
        fixture,
        role="champion_panel",
        wins=420,
        elo0=300.0,
        elo1=400.0,
        alpha=0.01,
        beta=0.01,
    )

    if passes:
        assert _execute(fixture, go=False)["status"] == "dry_run"
    else:
        with pytest.raises(promotion.PromotionError, match="allowed regression"):
            _execute(fixture, go=False)


def test_external_comparison_accepts_hash_replayed_fleet_pools(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    for role, checkpoint in (
        ("candidate_panel", fixture["candidate"]),
        ("champion_panel", fixture["champion"]),
    ):
        shard_path = tmp_path / f"{role}.shard.json"

        def make_pooled(source: dict, *, _shard=shard_path, _checkpoint=checkpoint):
            _write_json(_shard, source)
            effective = dict(source["search_config"])
            source["candidate_checkpoint_sha256"] = promotion._sha256(_checkpoint)
            source["effective_search_config"] = effective
            source["fleet_merge"] = {
                "schema_version": promotion.FLEET_EVALUATION_POOL_SCHEMA,
                "kind": "external_panel",
                "checkpoint": _checkpoint_ref(_checkpoint),
                "sources": [_checkpoint_ref(_shard)],
                "seed_intervals": [
                    {
                        "base_seed": 8_100_000,
                        "end_seed": 8_100_500,
                        "path": str(_shard.resolve()),
                    }
                ],
                "effective_search_config_sha256": promotion._digest_value(effective),
            }

        _mutate_evidence_source(
            fixture,
            kind="external_panel",
            role=role,
            mutate=make_pooled,
        )

    assert _execute(fixture, go=False)["status"] == "dry_run"


def _different_json_value(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.125
    if value is None:
        return 7
    if isinstance(value, str):
        return value + "_drift"
    raise AssertionError(f"no mutation strategy for {value!r}")


@pytest.mark.parametrize(
    "field",
    sorted(promotion._sealed_evaluation_semantics(_contract()).keys()),
)
def test_internal_h2h_rejects_every_sealed_semantic_drift(
    tmp_path: Path, field: str
) -> None:
    fixture = _fixture(tmp_path)

    def drift_and_rehash(source: dict) -> None:
        fields = source["typed_config"]["fields"]
        fields[field] = _different_json_value(fields[field])
        digest = hashlib.sha256(
            promotion._canonical_bytes(source["typed_config"])
        ).hexdigest()
        source["config_hash"] = "sha256:" + digest[:16]
        source["full_config_hash"] = "sha256:" + digest

    _mutate_evidence_source(
        fixture,
        kind="internal_h2h",
        role="internal_h2h",
        mutate=drift_and_rehash,
    )

    with pytest.raises(promotion.PromotionError, match="sealed A1 semantic drift"):
        _execute(fixture, go=False)


@pytest.mark.parametrize(
    "field",
    sorted(promotion._sealed_evaluation_semantics(_contract()).keys()),
)
def test_external_panels_cannot_jointly_launder_sealed_semantic_drift(
    tmp_path: Path, field: str
) -> None:
    fixture = _fixture(tmp_path)

    def drift(source: dict) -> None:
        config = source["search_config"]
        config[field] = _different_json_value(config[field])

    for role in ("candidate_panel", "champion_panel"):
        _mutate_evidence_source(
            fixture,
            kind="external_panel",
            role=role,
            mutate=drift,
        )

    with pytest.raises(promotion.PromotionError, match="sealed A1 semantic drift"):
        _execute(fixture, go=False)


def test_promotion_fails_closed_when_contract_omits_search_semantic(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    del fixture["contract"]["science"]["effective_search_config"]["c_scale"]

    with pytest.raises(
        promotion.PromotionError,
        match=r"omits effective_search_config\.c_scale",
    ):
        _execute(fixture, go=False)


def test_every_third_confirmation_is_derived_from_registry(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, promotion_count=2)
    payload = json.loads(fixture["adjudication"].read_text())
    payload["nth_confirmation_required"] = False
    payload["nth_confirmation_passed"] = False
    payload.pop("adjudication_sha256")
    payload["adjudication_sha256"] = promotion._digest_value(payload)
    _write_json(fixture["adjudication"], payload)

    with pytest.raises(promotion.PromotionError, match="every-third"):
        _execute(fixture, go=False)


def test_exclusive_lock_refuses_a_second_writer(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    descriptor = os.open(fixture["lock"], os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(promotion.PromotionError, match="already held"):
            _execute(fixture, go=False)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_exclusive_lock_rejects_symlink_and_revalidates_named_inode(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.lock"
    target.write_text("", encoding="utf-8")
    symlink = tmp_path / "symlink.lock"
    symlink.symlink_to(target)
    with pytest.raises(OSError):
        with promotion._exclusive_lock(symlink):
            pass

    lock = tmp_path / "promotion.lock"
    with pytest.raises(promotion.PromotionError, match="identity drifted"):
        with promotion._exclusive_lock(lock):
            replacement = tmp_path / "replacement.lock"
            replacement.write_text("", encoding="utf-8")
            os.replace(replacement, lock)


def test_alternate_lock_path_is_forbidden(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(promotion.PromotionError, match="alternate promotion lock"):
        promotion.execute_promotion(
            registry_path=fixture["registry"],
            current_pointer=fixture["pointer"],
            contract_lock=fixture["contract_path"],
            adjudication_path=fixture["adjudication"],
            training_receipt=fixture["training_receipt"],
            receipt_path=fixture["receipt"],
            reason="A1 typed promotion",
            lock_path=tmp_path / "bypass.lock",
            go=False,
            verify_lock_fn=_verify(fixture),
        )


def test_symlink_registry_is_rejected_before_lock_or_mutation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    alias = tmp_path / "registry.alias.json"
    alias.symlink_to(fixture["registry"])

    with pytest.raises(promotion.PromotionError, match="must not contain symlinks"):
        promotion.execute_promotion(
            registry_path=alias,
            current_pointer=fixture["pointer"],
            contract_lock=fixture["contract_path"],
            adjudication_path=fixture["adjudication"],
            training_receipt=fixture["training_receipt"],
            receipt_path=fixture["receipt"],
            reason="A1 typed promotion",
            go=False,
            verify_lock_fn=_verify(fixture),
        )


def test_failed_second_replace_rolls_registry_and_pointer_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    registry_before = fixture["registry"].read_bytes()
    pointer_before = fixture["pointer"].read_bytes()
    real_write = promotion._atomic_write_bytes
    failed = False

    def fail_once(path: Path, data: bytes) -> None:
        nonlocal failed
        if path == fixture["pointer"] and not failed and data != pointer_before:
            failed = True
            raise OSError("synthetic pointer replace failure")
        real_write(path, data)

    monkeypatch.setattr(promotion, "_atomic_write_bytes", fail_once)
    with pytest.raises(promotion.PromotionError, match="original.*restored"):
        _execute(fixture, go=True)

    assert fixture["registry"].read_bytes() == registry_before
    assert fixture["pointer"].read_bytes() == pointer_before
    assert json.loads(fixture["receipt"].read_text())["status"] == "rolled_back"


def test_recovery_refuses_tampered_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _execute(fixture, go=True)
    receipt = json.loads(fixture["receipt"].read_text())
    receipt["reason"] = "tampered"
    _write_json(fixture["receipt"], receipt)

    with pytest.raises(promotion.PromotionError, match="semantic digest mismatch"):
        promotion.recover_transaction(receipt_path=fixture["receipt"], go=True)


def test_failed_recovery_restores_pre_recovery_committed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _execute(fixture, go=True)
    committed_registry = fixture["registry"].read_bytes()
    committed_pointer = fixture["pointer"].read_bytes()
    before_pointer = Path(str(fixture["receipt"]) + ".current.before").read_bytes()
    real_write = promotion._atomic_write_bytes
    failed = False

    def fail_once(path: Path, data: bytes) -> None:
        nonlocal failed
        if path == fixture["pointer"] and data == before_pointer and not failed:
            failed = True
            raise OSError("synthetic recovery pointer failure")
        real_write(path, data)

    monkeypatch.setattr(promotion, "_atomic_write_bytes", fail_once)
    with pytest.raises(promotion.PromotionError, match="pre-recovery.*restored"):
        promotion.recover_transaction(receipt_path=fixture["receipt"], go=True)

    assert fixture["registry"].read_bytes() == committed_registry
    assert fixture["pointer"].read_bytes() == committed_pointer
    assert json.loads(fixture["receipt"].read_text())["status"] == "committed"


def _legacy_snapshot_for_fixture(fixture: dict, tmp_path: Path):
    source = tmp_path / "legacy-source.json"
    attestation = tmp_path / "legacy-attestation.json"
    _write_json(source, {"source": "pinned"})
    _write_json(attestation, {"attestation": "pinned"})
    return promotion._LegacyPromotionSnapshot(  # noqa: SLF001
        contract_lock=promotion._stable_json_snapshot(  # noqa: SLF001
            fixture["contract_path"], where="test contract"
        ),
        source_draft=promotion._stable_json_snapshot(  # noqa: SLF001
            source, where="test source"
        ),
        training_receipt=promotion._stable_json_snapshot(  # noqa: SLF001
            fixture["training_receipt"], where="test receipt"
        ),
        attestation=promotion._stable_json_snapshot(  # noqa: SLF001
            attestation, where="test attestation"
        ),
    )


def test_legacy_receipt_replacement_during_full_validation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    snapshot = _legacy_snapshot_for_fixture(fixture, tmp_path)
    monkeypatch.setattr(
        promotion,
        "_verify_contract_with_snapshot",
        lambda *_args, **_kwargs: (fixture["contract"], snapshot),
    )
    original = promotion._verify_one_dose_training_receipt  # noqa: SLF001
    replaced = False

    def replace_then_validate(path: Path, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            replacement = tmp_path / "replacement-receipt.json"
            replacement.write_bytes(Path(path).read_bytes())
            replacement.replace(path)
        return original(path, **kwargs)

    monkeypatch.setattr(
        promotion, "_verify_one_dose_training_receipt", replace_then_validate
    )
    with pytest.raises(
        promotion.PromotionError, match="historical training receipt pathname changed"
    ):
        _execute(fixture, go=False)


def test_legacy_attestation_replacement_before_plan_construction_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    snapshot = _legacy_snapshot_for_fixture(fixture, tmp_path)
    monkeypatch.setattr(
        promotion,
        "_verify_contract_with_snapshot",
        lambda *_args, **_kwargs: (fixture["contract"], snapshot),
    )
    original = promotion._verify_adjudication  # noqa: SLF001

    def replace_after_adjudication(*args, **kwargs):
        result = original(*args, **kwargs)
        assert snapshot.attestation is not None
        replacement = tmp_path / "replacement-attestation.json"
        replacement.write_bytes(snapshot.attestation.path.read_bytes())
        replacement.replace(snapshot.attestation.path)
        return result

    monkeypatch.setattr(promotion, "_verify_adjudication", replace_after_adjudication)
    with pytest.raises(
        promotion.PromotionError, match="legacy contract attestation pathname changed"
    ):
        _execute(fixture, go=False)


def test_legacy_snapshot_is_revalidated_at_commit_mutation_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    snapshot = _legacy_snapshot_for_fixture(fixture, tmp_path)
    monkeypatch.setattr(
        promotion,
        "_verify_contract_with_snapshot",
        lambda *_args, **_kwargs: (fixture["contract"], snapshot),
    )
    original_prepare = promotion.prepare_promotion

    def replace_after_prepare(**kwargs):
        plan = original_prepare(**kwargs)
        assert snapshot.attestation is not None
        replacement = tmp_path / "replacement-at-mutation-boundary.json"
        replacement.write_bytes(snapshot.attestation.data)
        replacement.replace(snapshot.attestation.path)
        return plan

    monkeypatch.setattr(promotion, "prepare_promotion", replace_after_prepare)
    registry_before = fixture["registry"].read_bytes()
    pointer_before = fixture["pointer"].read_bytes()
    with pytest.raises(
        promotion.PromotionError, match="legacy contract attestation pathname changed"
    ):
        _execute(fixture, go=True)
    assert fixture["registry"].read_bytes() == registry_before
    assert fixture["pointer"].read_bytes() == pointer_before
    assert not fixture["receipt"].exists()
    assert not fixture["receipt"].with_name(
        fixture["receipt"].name + ".registry.before"
    ).exists()
