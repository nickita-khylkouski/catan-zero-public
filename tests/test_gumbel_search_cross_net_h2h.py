"""Tests for the CAT-25 rollout-doubling extension to
tools/gumbel_search_cross_net_h2h.py: `_build_search_config` accepting an
explicit `n_full` override, and worker_args plumbing for
`candidate_n_full` / `baseline_n_full`. Pure argument-plumbing -- no
GPU/checkpoint/rust dependency needed."""

from __future__ import annotations

import sys
import json
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.rl.pipeline_configs import EvalConfig  # noqa: E402
import gumbel_search_cross_net_h2h as h2h  # type: ignore  # noqa: E402
from high_regret_suite_contract import (  # type: ignore  # noqa: E402
    REPLAY_CONTRACT,
    bind_state_to_manifest,
    scope_inventory_sha256,
)
from gumbel_search_cross_net_h2h import (  # type: ignore  # noqa: E402
    _build_search_config,
    _build_summary,
    _load_held_out_high_regret_suite,
    _new_search_telemetry,
    _resolve_c_scales,
    _resolve_search_budgets,
    play_one_h2h_game,
    _validate_information_set_recipe,
)


def test_held_out_suite_loader_replays_digest_and_source_manifest(
    tmp_path: Path, monkeypatch,
) -> None:
    shard_dir = tmp_path / "worker"
    shard_dir.mkdir()
    shard = shard_dir / "relative-shard.npz"
    np.savez(
        shard,
        game_seed=np.arange(123, 143),
        decision_index=np.zeros(20, dtype=np.int32),
        action_taken=np.arange(20),
    )
    source = tmp_path / "regret.npz"
    np.savez(
        source,
        shard_paths=np.asarray([str(shard)]),
        shard_id=np.zeros(20, dtype=np.int32),
        row_index=np.arange(20, dtype=np.int32),
        game_seed=np.arange(123, 143, dtype=np.int64),
        decision_index=np.zeros(20, dtype=np.int32),
    )
    scope_digest, scope_count = scope_inventory_sha256(shard_dir)
    suite = {
        "schema_version": h2h.SUITE_SCHEMA,
        "suite": "held_out_high_regret",
        "held_out": True,
        "source_manifest": {
            "path": str(source),
            "sha256": h2h._checkpoint_sha256(source),
        },
        "selection": {
            "algorithm": "stable-hash-holdout-stratified-regret-v1",
            "holdout_fraction": 0.10,
            "holdout_seed": 17,
            "eligible_unique_states": 20,
            "selected_pairs": 20,
            "stratum_min_pairs": 4,
            "selected_by_stratum": {
                "phase:opening": 4,
                "phase:robber_dev": 4,
                "phase:chance": 4,
                "phase:build_trade": 4,
                "41+": 4,
            },
            "replay_preflight": {
                "contract": REPLAY_CONTRACT,
                "candidate_states": 20,
                "replay_complete_states": 20,
                "rejected_bad_source": 0,
                "rejected_noncontiguous": 0,
            },
        },
        "states": [
            {
                "pair_id": pair,
                "shard_id": 0,
                "row_index": pair,
                "game_seed": 123 + pair,
                "decision_index": 0,
                "shard_path": "worker/relative-shard.npz",
                "phase": (
                    "BUILD_INITIAL_SETTLEMENT",
                    "MOVE_ROBBER",
                    "ROLL",
                    "BUILD_ROAD",
                )[pair % 4],
                "legal_count": 54 if pair < 4 else 12,
                "replay_source": {
                    "contract": REPLAY_CONTRACT,
                    "scope": str(shard_dir),
                    "scope_inventory_sha256": scope_digest,
                    "scope_shard_count": scope_count,
                },
            }
            for pair in range(20)
        ],
    }
    suite["suite_sha256"] = (
        "sha256:"
        + h2h.hashlib.sha256(
            json.dumps(suite, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite), encoding="utf-8")

    resolved_path, loaded, pairs = _load_held_out_high_regret_suite(path)
    sealed_suite = copy.deepcopy(suite)

    assert resolved_path == path.resolve()
    assert loaded == suite
    assert len(pairs) == 20
    assert pairs[0] == {
        "pair_id": 0,
        "game_seed": 123,
        "archived_state": {
            **suite["states"][0],
            "shard_path": str(shard.resolve()),
        },
    }

    original_shard_bytes = shard.read_bytes()
    h2h._validate_archived_scope_inventory(pairs[0]["archived_state"], {})

    from tools import regret_common

    original_load = regret_common.load_shard
    swapped = False

    def swap_before_load(load_path: Path):
        nonlocal swapped
        if Path(load_path).resolve() == shard.resolve() and not swapped:
            swapped = True
            replacement = shard.with_suffix(".replacement.npz")
            np.savez(
                replacement,
                game_seed=np.arange(123, 143),
                decision_index=np.zeros(20, dtype=np.int32),
                action_taken=np.arange(20) + 100,
            )
            replacement.replace(shard)
        return original_load(load_path)

    monkeypatch.setattr(regret_common, "load_shard", swap_before_load)
    with pytest.raises(ValueError, match="changed while loading source row"):
        _load_held_out_high_regret_suite(path)
    monkeypatch.setattr(regret_common, "load_shard", original_load)
    shard.write_bytes(original_shard_bytes)

    shard.write_bytes(b"replacement trajectory bytes")
    with pytest.raises(ValueError, match="scope inventory drifted"):
        _load_held_out_high_regret_suite(path)
    with pytest.raises(ValueError, match="worker replay scope inventory drifted"):
        h2h._validate_archived_scope_inventory(pairs[0]["archived_state"], {})
    shard.write_bytes(original_shard_bytes)

    injected = shard_dir / "injected.npz"
    np.savez(
        injected,
        game_seed=np.asarray([123]),
        decision_index=np.asarray([0]),
        action_taken=np.asarray([99]),
    )
    with pytest.raises(ValueError, match="scope inventory drifted"):
        _load_held_out_high_regret_suite(path)
    injected.unlink()

    legacy = copy.deepcopy(sealed_suite)
    legacy["schema_version"] = "a1-held-out-high-regret-suite-v2"
    legacy["suite_sha256"] = (
        "sha256:"
        + h2h.hashlib.sha256(
            json.dumps(
                {key: value for key, value in legacy.items() if key != "suite_sha256"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(ValueError, match="identity is invalid"):
        _load_held_out_high_regret_suite(path)
    path.write_text(json.dumps(sealed_suite), encoding="utf-8")

    suite["states"][0]["decision_index"] = 8
    path.write_text(json.dumps(suite), encoding="utf-8")

    with pytest.raises(ValueError, match="semantic digest mismatch"):
        _load_held_out_high_regret_suite(path)

    adversarial = copy.deepcopy(sealed_suite)
    adversarial["states"][0]["shard_path"] = str(tmp_path / "other.npz")
    adversarial["suite_sha256"] = (
        "sha256:"
        + h2h.hashlib.sha256(
            json.dumps(
                {key: value for key, value in adversarial.items() if key != "suite_sha256"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    path.write_text(json.dumps(adversarial), encoding="utf-8")
    with pytest.raises(ValueError, match="shard_path differs"):
        _load_held_out_high_regret_suite(path)

    adversarial = copy.deepcopy(sealed_suite)
    del adversarial["selection"]["replay_preflight"]
    adversarial["suite_sha256"] = (
        "sha256:"
        + h2h.hashlib.sha256(
            json.dumps(
                {key: value for key, value in adversarial.items() if key != "suite_sha256"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    path.write_text(json.dumps(adversarial), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks required replay preflight"):
        _load_held_out_high_regret_suite(path)

    adversarial = copy.deepcopy(sealed_suite)
    adversarial["states"][0]["game_seed"] += 1
    adversarial["suite_sha256"] = (
        "sha256:"
        + h2h.hashlib.sha256(
            json.dumps(
                {key: value for key, value in adversarial.items() if key != "suite_sha256"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    path.write_text(json.dumps(adversarial), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound to source manifest row"):
        _load_held_out_high_regret_suite(path)


def test_manifest_source_must_belong_to_hashed_replay_inventory(
    tmp_path: Path,
) -> None:
    scope = tmp_path / "worker"
    scope.mkdir()
    replay_shard = scope / "trajectory.npz"
    np.savez(
        replay_shard,
        game_seed=np.asarray([123]),
        decision_index=np.asarray([0]),
        action_taken=np.asarray([1]),
    )
    unbound_npz = scope / "unbound.npz"
    np.savez(
        unbound_npz,
        game_seed=np.asarray([123]),
        decision_index=np.asarray([0]),
        action_taken=np.asarray([1]),
    )
    unbound = scope / "unbound.bin"
    unbound_npz.replace(unbound)
    inventory_sha, inventory_count = scope_inventory_sha256(scope)
    state = {
        "shard_id": 0,
        "row_index": 0,
        "game_seed": 123,
        "decision_index": 0,
        "shard_path": str(unbound),
        "replay_source": {
            "contract": REPLAY_CONTRACT,
            "scope": str(scope),
            "scope_inventory_sha256": inventory_sha,
            "scope_shard_count": inventory_count,
        },
    }

    with pytest.raises(ValueError, match="outside replay inventory namespace"):
        bind_state_to_manifest(
            state,
            suite_base=tmp_path,
            manifest_path=tmp_path / "manifest.npz",
            shard_paths=[str(unbound)],
            identities={(0, 0, 123, 0)},
        )


def _base_worker_args(**overrides) -> dict:
    args = {
        "n_full": 64,
        "max_depth": 80,
        "correct_rust_chance_spectra": True,
    }
    args.update(overrides)
    return args


def test_public_observation_requires_information_set_search() -> None:
    args = SimpleNamespace(
        public_observation=True,
        information_set_search=False,
        belief_chance_spectra=False,
        determinization_particles=4,
        determinization_min_simulations=32,
    )
    import pytest

    with pytest.raises(ValueError, match="requires --information-set-search"):
        _validate_information_set_recipe(args)

    args.information_set_search = True
    _validate_information_set_recipe(args)


def test_build_search_config_threads_information_set_recipe() -> None:
    config = _build_search_config(
        _base_worker_args(
            information_set_search=True,
            determinization_particles=4,
            determinization_min_simulations=32,
        ),
        seed=1,
    )
    assert config.information_set_search is True
    assert config.determinization_particles == 4
    assert config.determinization_min_simulations == 32


def test_build_search_config_seals_disabled_experimental_semantics() -> None:
    config = _build_search_config(_base_worker_args(n_full=128), seed=1)
    assert config.exact_budget_sh is False
    assert config.root_wave_batching is False
    assert config.play_sh_winner is False
    assert config.use_batch_api is True
    assert config.policy_target_min_visits == 0
    assert config.uncertainty_backup_weighting is False
    assert config.variance_aware_q is False


def test_build_search_config_defaults_to_shared_n_full_when_no_override():
    worker_args = _base_worker_args()
    config = _build_search_config(worker_args, seed=1)
    assert config.n_full == 64
    assert config.n_fast == 64


def test_build_search_config_explicit_n_full_overrides_shared_value():
    worker_args = _base_worker_args()
    config = _build_search_config(worker_args, seed=1, n_full=128)
    assert config.n_full == 128
    assert config.n_fast == 128


def test_build_search_config_none_n_full_falls_back_to_shared_value():
    worker_args = _base_worker_args()
    config = _build_search_config(worker_args, seed=1, n_full=None)
    assert config.n_full == 64


def test_worker_args_resolution_uses_candidate_and_baseline_keys_when_present():
    """Mirrors the resolution logic _run_worker applies: worker_args.get(
    'candidate_n_full', worker_args['n_full']) / same for baseline."""
    worker_args = _base_worker_args(candidate_n_full=128, baseline_n_full=64)

    candidate_n_full = int(worker_args.get("candidate_n_full", worker_args["n_full"]))
    baseline_n_full = int(worker_args.get("baseline_n_full", worker_args["n_full"]))

    candidate_config = _build_search_config(
        worker_args, seed=1, n_full=candidate_n_full
    )
    baseline_config = _build_search_config(worker_args, seed=1, n_full=baseline_n_full)

    assert candidate_config.n_full == 128
    assert baseline_config.n_full == 64


def test_worker_args_resolution_omits_keys_falls_back_to_shared_n_full():
    """Every existing caller of this tool never sets candidate_n_full/
    baseline_n_full -- both roles must resolve to the shared --n-full,
    byte-identical to pre-extension behavior."""
    worker_args = _base_worker_args(n_full=64)

    candidate_n_full = int(worker_args.get("candidate_n_full", worker_args["n_full"]))
    baseline_n_full = int(worker_args.get("baseline_n_full", worker_args["n_full"]))

    assert candidate_n_full == 64
    assert baseline_n_full == 64

    candidate_config = _build_search_config(
        worker_args, seed=1, n_full=candidate_n_full
    )
    baseline_config = _build_search_config(worker_args, seed=1, n_full=baseline_n_full)
    assert candidate_config.n_full == baseline_config.n_full == 64


def test_build_search_config_preserves_other_fields_regardless_of_n_full_override():
    worker_args = _base_worker_args(
        n_full=64,
        c_scale=0.2,
        c_visit=10.0,
        max_root_candidates=8,
    )
    config = _build_search_config(worker_args, seed=7, n_full=128)
    assert config.c_scale == 0.2
    assert config.c_visit == 10.0
    assert config.max_root_candidates == 8
    assert config.seed == 7


def test_role_specific_c_scales_override_shared_fallback_independently():
    worker_args = _base_worker_args(
        c_scale=0.2,
        candidate_c_scale=0.1,
        baseline_c_scale=0.03,
    )
    resolved = _resolve_c_scales(worker_args)

    candidate = _build_search_config(
        worker_args, seed=1, c_scale=resolved["candidate_c_scale"]
    )
    baseline = _build_search_config(
        worker_args, seed=1, c_scale=resolved["baseline_c_scale"]
    )

    assert candidate.c_scale == 0.1
    assert baseline.c_scale == 0.03


def test_shared_c_scale_is_backward_compatible_role_fallback():
    assert _resolve_c_scales(_base_worker_args(c_scale=0.03)) == {
        "candidate_c_scale": 0.03,
        "baseline_c_scale": 0.03,
    }


def test_worker_constructs_each_role_with_its_effective_c_scale(monkeypatch):
    built_configs = []

    class FakeEvaluator:
        def close(self):
            pass

    class FakeMCTS:
        def __init__(self, config, evaluator):
            built_configs.append(config)

    monkeypatch.setattr(h2h, "_build_evaluator", lambda *args, **kwargs: FakeEvaluator())
    monkeypatch.setattr(h2h, "GumbelChanceMCTS", FakeMCTS)

    result = h2h._run_worker(
        {
            **_base_worker_args(
                c_scale=0.2,
                candidate_c_scale=0.1,
                baseline_c_scale=0.03,
            ),
            "worker_index": 0,
            "worker_seed": 7,
            "candidate_checkpoint": "candidate.pt",
            "baseline_checkpoint": "baseline.pt",
            "pairs": [],
        }
    )

    assert result["error"] is None
    assert [config.c_scale for config in built_configs] == [0.1, 0.03]


def test_build_search_config_threads_d1_noise_floor_calibration():
    config = _build_search_config(
        _base_worker_args(rescale_noise_floor_c=0.25, sigma_eval=0.5),
        seed=1,
    )
    assert config.rescale_noise_floor_c == 0.25
    assert config.sigma_eval == 0.5

    default_config = _build_search_config(_base_worker_args(), seed=1)
    assert default_config.rescale_noise_floor_c == 0.0
    assert default_config.sigma_eval == 0.79


def test_role_specific_wide_budget_overrides_only_candidate():
    worker_args = _base_worker_args(
        n_full_wide=None,
        candidate_n_full=128,
        baseline_n_full=128,
        candidate_n_full_wide=256,
        candidate_n_full_wide_threshold=40,
    )
    budgets = _resolve_search_budgets(worker_args)

    candidate = _build_search_config(
        worker_args,
        seed=1,
        n_full=int(budgets["candidate_n_full"]),
        n_full_wide=budgets["candidate_n_full_wide"],
        n_full_wide_threshold=budgets["candidate_n_full_wide_threshold"],
    )
    baseline = _build_search_config(
        worker_args,
        seed=1,
        n_full=int(budgets["baseline_n_full"]),
        n_full_wide=budgets["baseline_n_full_wide"],
        n_full_wide_threshold=budgets["baseline_n_full_wide_threshold"],
    )

    assert candidate.n_full == baseline.n_full == 128
    assert candidate.n_full_wide == 256
    assert candidate.n_full_wide_threshold == 40
    assert baseline.n_full_wide is None
    assert baseline.n_full_wide_threshold is None


def test_shared_wide_budget_is_backward_compatible_fallback():
    budgets = _resolve_search_budgets(_base_worker_args(n_full_wide=512))
    assert budgets == {
        "candidate_n_full": 64,
        "baseline_n_full": 64,
        "candidate_n_full_wide": 512,
        "baseline_n_full_wide": 512,
        "candidate_n_full_wide_threshold": None,
        "baseline_n_full_wide_threshold": None,
    }


def test_role_specific_wide_budgets_override_shared_fallback_independently():
    budgets = _resolve_search_budgets(
        _base_worker_args(
            n_full_wide=512,
            candidate_n_full_wide=256,
            baseline_n_full_wide=128,
            n_full_wide_threshold=40,
            candidate_n_full_wide_threshold=48,
            baseline_n_full_wide_threshold=32,
        )
    )
    assert budgets["candidate_n_full_wide"] == 256
    assert budgets["baseline_n_full_wide"] == 128
    assert budgets["candidate_n_full_wide_threshold"] == 48
    assert budgets["baseline_n_full_wide_threshold"] == 32


def test_eval_config_hash_distinguishes_adaptive_candidate_from_shared_arm():
    adaptive = EvalConfig(
        mode="cross_net",
        n_full=64,
        candidate_n_full=128,
        baseline_n_full=128,
        candidate_n_full_wide=256,
        baseline_n_full_wide=None,
        candidate_n_full_wide_threshold=40,
        baseline_n_full_wide_threshold=None,
    )
    uniform = EvalConfig(
        mode="cross_net",
        n_full=64,
        candidate_n_full=128,
        baseline_n_full=128,
        candidate_n_full_wide=None,
        baseline_n_full_wide=None,
        candidate_n_full_wide_threshold=None,
        baseline_n_full_wide_threshold=None,
    )
    shared = EvalConfig(
        mode="cross_net",
        n_full=64,
        n_full_wide=256,
        candidate_n_full=128,
        baseline_n_full=128,
        candidate_n_full_wide=256,
        baseline_n_full_wide=256,
        n_full_wide_threshold=40,
        candidate_n_full_wide_threshold=40,
        baseline_n_full_wide_threshold=40,
    )

    assert adaptive.config_hash() != uniform.config_hash()
    assert adaptive.config_hash() != shared.config_hash()
    assert adaptive.full_config_hash().startswith("sha256:")


def test_eval_config_hash_distinguishes_d1_calibration():
    legacy = EvalConfig(mode="cross_net")
    calibrated = EvalConfig(
        mode="cross_net", rescale_noise_floor_c=0.25, sigma_eval=0.5
    )
    assert legacy.config_hash() != calibrated.config_hash()


def test_eval_config_hash_distinguishes_role_specific_c_scales():
    shared = EvalConfig(
        mode="cross_net", candidate_c_scale=0.03, baseline_c_scale=0.03
    )
    tuned = EvalConfig(
        mode="cross_net", candidate_c_scale=0.1, baseline_c_scale=0.03
    )
    assert shared.config_hash() != tuned.config_hash()


def test_h2h_summary_records_resolved_adaptive_budget_by_role():
    args = SimpleNamespace(
        candidate="same.pt",
        baseline="same.pt",
        gate_config="flywheel",
        n_full=64,
        candidate_n_full=128,
        baseline_n_full=128,
        n_full_wide=None,
        candidate_n_full_wide=256,
        baseline_n_full_wide=None,
        n_full_wide_threshold=None,
        candidate_n_full_wide_threshold=40,
        baseline_n_full_wide_threshold=None,
        lazy_interior_chance=True,
        value_squash="tanh",
        value_readout="scalar",
        candidate_value_readout=None,
        baseline_value_readout=None,
        c_scale=0.1,
        candidate_c_scale=0.1,
        baseline_c_scale=0.03,
        c_visit=50.0,
        rescale_noise_floor_c=0.25,
        sigma_eval=0.5,
        max_root_candidates=16,
        max_root_candidates_wide=54,
        correct_rust_chance_spectra=True,
        public_observation=True,
        belief_chance_spectra=False,
        information_set_search=True,
        determinization_particles=4,
        determinization_min_simulations=32,
        raw_policy_above_width=None,
        symmetry_averaged_eval=True,
        symmetry_averaged_eval_threshold=20,
        wide_candidates_threshold=24,
        elo0=-10.0,
        elo1=15.0,
    )
    summary = _build_summary(
        args,
        all_games=[],
        outcomes=[],
        truncated_count=0,
        pairs=[],
        elapsed=0.0,
        workers=1,
        threads_per_worker=1,
        errors=[],
        candidate_checkpoint_sha256="sha256:" + "1" * 64,
        baseline_checkpoint_sha256="sha256:" + "2" * 64,
        search_telemetry={
            "candidate": {
                "search_calls": 10,
                "non_forced_search_calls": 10,
                "search_elapsed_sec": 20.0,
                "simulations_used": 1536,
                "wide_root_calls": 4,
                "wide_root_simulations_used": 1024,
                "selected_vs_prior_disagreement_calls": 3,
                "wide_selected_vs_prior_disagreement_calls": 2,
            },
            "baseline": {
                "search_calls": 10,
                "non_forced_search_calls": 10,
                "search_elapsed_sec": 10.0,
                "simulations_used": 1280,
                "wide_root_calls": 4,
                "wide_root_simulations_used": 512,
                "selected_vs_prior_disagreement_calls": 2,
                "wide_selected_vs_prior_disagreement_calls": 1,
            },
        },
    )
    assert summary["candidate_checkpoint_sha256"] == "sha256:" + "1" * 64
    assert summary["baseline_checkpoint_sha256"] == "sha256:" + "2" * 64

    assert summary["candidate_n_full_wide"] == 256
    assert summary["baseline_n_full_wide"] is None
    assert summary["candidate_n_full_wide_threshold"] == 40
    assert summary["baseline_n_full_wide_threshold"] is None
    assert summary["symmetry_averaged_eval_threshold"] == 20
    assert summary["rescale_noise_floor_c"] == 0.25
    assert summary["sigma_eval"] == 0.5
    assert summary["candidate_c_scale"] == 0.1
    assert summary["baseline_c_scale"] == 0.03
    assert summary["search_parameters_by_role"] == {
        "candidate": {"c_scale": 0.1, "c_visit": 50.0},
        "baseline": {"c_scale": 0.03, "c_visit": 50.0},
    }
    assert (
        summary["comparison_contract"]
        == "paired_same_seed_color_swap_role_specific_search_operators"
    )
    assert summary["search_budgets_by_role"] == {
        "candidate": {
            "n_full": 128,
            "n_full_wide": 256,
            "n_full_wide_threshold": 40,
        },
        "baseline": {
            "n_full": 128,
            "n_full_wide": None,
            "n_full_wide_threshold": None,
        },
    }
    telemetry = summary["search_telemetry"]
    assert telemetry["candidate_over_baseline_elapsed_ratio"] == 2.0
    assert telemetry["candidate_over_baseline_seconds_per_call_ratio"] == 2.0
    assert telemetry["candidate_over_baseline_simulations_ratio"] == 1.2
    assert telemetry["candidate_over_baseline_simulations_per_call_ratio"] == 1.2
    assert telemetry["by_role"]["candidate"]["wide_root_calls"] == 4
    assert telemetry["by_role"]["candidate"]["wide_root_simulations_used"] == 1024
    assert telemetry["by_role"]["candidate"]["wide_root_simulations_per_call"] == 256
    assert (
        telemetry["by_role"]["candidate"]["selected_vs_prior_disagreement_rate"] == 0.3
    )


def test_play_game_records_exact_role_simulations_and_decision_change(monkeypatch):
    class FakeGame:
        won = False

        def winning_color(self):
            return "RED" if self.won else None

        def playable_action_indices(self, _colors, _unused):
            return list(range(40))

        def current_color(self):
            return "RED"

        def player_state_json(self, color):
            return (
                '{"victory_points": 10}' if color == "RED" else '{"victory_points": 2}'
            )

    game = FakeGame()

    class FakeRustGameFactory:
        @staticmethod
        def simple(_colors, *, seed):
            assert seed == 17
            return game

    fake_rust = SimpleNamespace(Game=FakeRustGameFactory)
    monkeypatch.setattr(h2h, "_require_rust_module", lambda: fake_rust)

    def apply_action(current_game, selected, **_kwargs):
        assert selected == 7
        current_game.won = True
        return current_game

    monkeypatch.setattr(h2h, "_apply_selected_action", apply_action)

    class FakeMCTS:
        config = SimpleNamespace(
            n_full_wide_threshold=40,
            wide_candidates_threshold=24,
        )

        def search(self, _game, *, force_full):
            assert force_full is True
            return SimpleNamespace(
                selected_action=7,
                priors={3: 0.9, 7: 0.1},
                simulations_used=256,
            )

    telemetry = _new_search_telemetry()
    record = play_one_h2h_game(
        {"candidate": FakeMCTS(), "baseline": FakeMCTS()},
        role_by_color={"RED": "candidate", "BLUE": "baseline"},
        game_seed=17,
        max_decisions=2,
        correct_rust_chance_spectra=True,
        search_telemetry_by_role=telemetry,
    )

    assert record["candidate_won"] is True
    candidate_telemetry = dict(telemetry["candidate"])
    elapsed = candidate_telemetry.pop("search_elapsed_sec")
    assert candidate_telemetry == {
        "search_calls": 1,
        "non_forced_search_calls": 1,
        "simulations_used": 256,
        "wide_root_calls": 1,
        "wide_root_simulations_used": 256,
        "selected_vs_prior_disagreement_calls": 1,
        "wide_selected_vs_prior_disagreement_calls": 1,
    }
    assert elapsed >= 0.0
