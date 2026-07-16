from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from catan_zero.rl.gumbel_self_play import (
    TARGET_INFORMATION_REGIME_PUBLIC,
    TARGET_INFORMATION_REGIME_PUBLIC_COHERENT,
)
from catan_zero.rl.pipeline_configs import CONFIG_SCHEMA_VERSION, GenerateConfig


REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tools import generate_gumbel_selfplay_data as generator  # noqa: E402
from tools import a1_pre_wave_contract as contract  # noqa: E402
from tools import prelaunch_guard  # noqa: E402
from tools.build_memmap_corpus import (  # noqa: E402
    EVENT_STORAGE_WIDTH,
    _normalize_event_storage_width,
)


CONFIG = (
    REPO
    / "configs/experiments/next_wave/coherent_public_n128_adaptive256.schema13.json"
)
GUARD = REPO / "configs/guards/a1_generation_coherent_public_n128_adaptive256_v1.json"
LEARNER = REPO / "configs/experiments/next_wave/one_dose_public_card_overrides.json"
RUNBOOK = REPO / "configs/operations/a1-next-wave-coherent-public-v1/README.md"


def test_meaningful_history_uses_legacy_compatible_memmap_width() -> None:
    tokens = np.ones((2, 32, 41), dtype=np.float16)
    targets = np.zeros((2, 32, 4), dtype=np.int16)
    mask = np.ones((2, 32), dtype=np.bool_)
    normalized = _normalize_event_storage_width(
        {
            "event_tokens": tokens,
            "event_target_ids": targets,
            "event_mask": mask,
        }
    )
    assert normalized["event_tokens"].shape == (2, EVENT_STORAGE_WIDTH, 41)
    assert np.array_equal(normalized["event_tokens"][:, :32], tokens)
    assert not np.any(normalized["event_tokens"][:, 32:])
    assert not np.any(normalized["event_mask"][:, 32:])
    assert np.all(normalized["event_target_ids"][:, 32:] == -1)


def test_next_wave_typed_generation_config_is_exact_schema_13_recipe() -> None:
    payload = json.loads(CONFIG.read_text())
    assert payload["pipeline"] == GenerateConfig.PIPELINE
    assert payload["schema_version"] == CONFIG_SCHEMA_VERSION == 13

    cfg = GenerateConfig(**payload["fields"])
    assert cfg.canonical_payload() == payload
    assert cfg.public_observation is True
    assert cfg.public_card_count_feature_schema == "public_card_state_v2"
    assert cfg.coherent_public_belief_search is True
    assert cfg.information_set_search is False
    assert cfg.belief_chance_spectra is False
    assert cfg.native_mcts_hot_loop is True
    assert cfg.forced_root_target_mode == "trajectory_only"
    assert (cfg.n_full, cfg.n_fast, cfg.p_full) == (128, 16, 0.25)
    assert (cfg.n_full_wide, cfg.n_full_wide_threshold) == (None, None)
    assert cfg.wide_roots_always_full is False
    assert cfg.symmetry_averaged_eval_threshold == 20
    assert cfg.eval_cache_size == 0
    assert cfg.shard_size == 512
    assert cfg.meaningful_public_history is True
    assert cfg.event_history_limit == 32
    assert cfg.record_automatic_transitions is False
    assert cfg.temperature_clock == "nonforced_choice"
    assert (cfg.temperature_decisions, cfg.late_temperature_decisions) == (40, 100)
    assert cfg.late_temperature == 0.1
    # Operational identity is supplied per lane; this checked-in file cannot
    # accidentally launch a real wave by itself.
    assert cfg.checkpoint is None
    assert cfg.games == 0


def test_next_wave_guard_pins_the_same_science_values() -> None:
    config = json.loads(CONFIG.read_text())["fields"]
    lint = json.loads(GUARD.read_text())["guards"][0]["args"]
    expected = lint["expected_values"]

    mapping = {
        "--belief-chance-spectra": "belief_chance_spectra",
        "--c-scale": "c_scale",
        "--c-visit": "c_visit",
        "--coherent-public-belief-search": "coherent_public_belief_search",
        "--determinization-min-simulations": "determinization_min_simulations",
        "--determinization-particles": "determinization_particles",
        "--eval-cache-size": "eval_cache_size",
        "--forced-root-target-mode": "forced_root_target_mode",
        "--information-set-search": "information_set_search",
        "--late-temperature": "late_temperature",
        "--late-temperature-decisions": "late_temperature_decisions",
        "--meaningful-public-history": "meaningful_public_history",
        "--max-decisions": "max_decisions",
        "--max-depth": "max_depth",
        "--n-fast": "n_fast",
        "--n-full": "n_full",
        "--native-mcts-hot-loop": "native_mcts_hot_loop",
        "--p-full": "p_full",
        "--public-observation": "public_observation",
        "--record-automatic-transitions": "record_automatic_transitions",
        "--rust-featurize": "rust_featurize",
        "--sigma-eval": "sigma_eval",
        "--symmetry-averaged-eval": "symmetry_averaged_eval",
        "--symmetry-averaged-eval-threshold": "symmetry_averaged_eval_threshold",
        "--temperature-clock": "temperature_clock",
        "--temperature-decisions": "temperature_decisions",
        "--temperature-high": "temperature_high",
        "--temperature-low": "temperature_low",
        "--track": "track",
        "--vps-to-win": "vps_to_win",
        "--wide-roots-always-full": "wide_roots_always_full",
        "--event-history-limit": "event_history_limit",
    }
    for flag, field in mapping.items():
        assert expected[flag] == config[field]

    assert config["n_full_wide"] is None
    assert config["n_full_wide_threshold"] is None
    assert "--n-full-wide" not in expected
    assert "--n-full-wide-threshold" not in expected

    critical = set(lint["critical_flags"])
    assert {"--base-seed", "--games"} <= critical


def test_next_wave_guard_is_executable_against_the_real_generator_parser() -> None:
    parser = generator.build_parser()
    lint = json.loads(GUARD.read_text())["guards"][0]["args"]
    argv = [
        "--out-dir",
        "/tmp/next-wave-test",
        "--checkpoint",
        "/tmp/champion.pt",
        "--base-seed",
        "7000000000",
        "--games",
        "1",
    ]
    for flag, expected in lint["expected_values"].items():
        action = parser._option_string_actions[flag]  # noqa: SLF001
        if isinstance(expected, bool):
            if expected:
                argv.append(flag)
            else:
                negative = next(
                    option
                    for option in action.option_strings
                    if option.startswith("--no-")
                )
                argv.append(negative)
        else:
            argv.extend((flag, str(expected)))

    result = prelaunch_guard.guard_cli_flag_lint(
        argv,
        lint["critical_flags"],
        parser=parser,
        expected_values=lint["expected_values"],
    )
    assert result.passed, result.reason


def test_next_wave_learner_preserves_all_forced_value_states() -> None:
    recipe = json.loads(LEARNER.read_text())
    assert recipe == {
        "forced_action_weight": 0.0,
        "forced_row_value_action_type_weights": "END_TURN=1,ROLL=1",
        "forced_row_value_weight": 1.0,
        "max_steps": 128,
        "per_game_policy_surprise_weighting": True,
        "public_card_lr_mult": 4.0,
        "value_loss_weight": 0.25,
    }
    assert "DISCARD_RESOURCE" not in recipe["forced_row_value_action_type_weights"]


def test_next_wave_runbook_closes_generation_training_evaluation_loop() -> None:
    text = RUNBOOK.read_text()
    assert "tools/generate_gumbel_selfplay_data.py" in text
    assert "tools/a1_iteration_orchestrator.py" in text
    assert "tools/gumbel_search_cross_net_h2h.py" in text
    assert "--forced-root-target-mode trajectory_only" in text
    assert "--coherent-public-belief-search" in text
    assert "--no-record-automatic-transitions" in text
    assert "--meaningful-public-history" in text
    assert "--event-history-limit 32" in text
    assert "--flags card_count_v2,meaningful_history" in text
    assert (
        "entity_graph.public_card_count_features+meaningful_public_history.v2"
        in text
    )
    assert "tools/a1_iteration_orchestrator.py initialize-next" in text
    assert "tools/fleet/a1_h100_eval_fleet.py" in text
    assert "action-target gather experiment was neutral" in text


def _contract_fields(*, coherent: bool) -> tuple[dict, dict, dict]:
    fields = json.loads(CONFIG.read_text())["fields"]
    search = {key: fields[key] for key in contract._SEARCH_INPUT_KEYS}  # noqa: SLF001
    generation_fields = {
        **fields,
        "format": fields["fmt"],
        "workers_per_gpu": fields["workers"],
    }
    generation = {
        key: generation_fields[key]
        for key in contract._GENERATION_KEYS  # noqa: SLF001
    }
    if coherent:
        search.update(
            {
                "coherent_public_belief_search": True,
                "forced_root_target_mode": "trajectory_only",
                "information_set_search": False,
                "determinization_particles": 1,
            }
        )
        generation["temperature_clock"] = "nonforced_choice"
        generation.update(
            {
                "record_automatic_transitions": False,
                "meaningful_public_history": True,
                "event_history_limit": 32,
            }
        )
        regime = TARGET_INFORMATION_REGIME_PUBLIC_COHERENT
    else:
        search.update(
            {
                "information_set_search": True,
                "determinization_particles": 4,
            }
        )
        regime = TARGET_INFORMATION_REGIME_PUBLIC
    post_wave = json.loads(
        (REPO / "configs/experiments/a1_pre_wave_contract.rnd_draft.json").read_text()
    )["post_wave_acceptance"]
    post_wave["require_target_information_regime"] = regime
    return search, generation, post_wave


def test_pre_wave_contract_preserves_issued_pimc_shape() -> None:
    search, generation, post_wave = _contract_fields(coherent=False)

    operator = contract._search_operator(search)  # noqa: SLF001
    effective = contract._effective_search(search)  # noqa: SLF001
    assert set(operator) == contract._SEARCH_INPUT_KEYS  # noqa: SLF001
    assert "coherent_public_belief_search" not in operator
    assert "forced_root_target_mode" not in operator
    assert "coherent_public_belief_search" not in effective
    assert "forced_root_target_mode" not in effective
    assert (
        contract._target_information_regime_for_search(operator)  # noqa: SLF001
        == TARGET_INFORMATION_REGIME_PUBLIC
    )
    contract._validate_generation(generation)  # noqa: SLF001
    contract._validate_post_wave(  # noqa: SLF001
        post_wave, expected_target_information_regime=TARGET_INFORMATION_REGIME_PUBLIC
    )


def test_pre_wave_contract_binds_coherent_regime_and_runtime_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search, generation, post_wave = _contract_fields(coherent=True)

    operator = contract._search_operator(search)  # noqa: SLF001
    effective = contract._effective_search(search)  # noqa: SLF001
    assert operator["coherent_public_belief_search"] is True
    assert operator["forced_root_target_mode"] == "trajectory_only"
    assert effective["coherent_public_belief_search"] is True
    assert effective["forced_root_target_mode"] == "trajectory_only"
    assert (
        contract._target_information_regime_for_search(operator)  # noqa: SLF001
        == TARGET_INFORMATION_REGIME_PUBLIC_COHERENT
    )
    contract._validate_generation(generation)  # noqa: SLF001
    contract._validate_post_wave(  # noqa: SLF001
        post_wave,
        expected_target_information_regime=TARGET_INFORMATION_REGIME_PUBLIC_COHERENT,
    )

    fields = json.loads(CONFIG.read_text())["fields"]
    evaluator = json.loads(
        (REPO / "configs/experiments/a1_pre_wave_contract.rnd_draft.json").read_text()
    )["science"]["evaluator"]
    evaluator.update(
        {
            "public_observation": fields["public_observation"],
            "rust_featurize": fields["rust_featurize"],
            "cache_size": fields["eval_cache_size"],
            "value_scale": fields["value_scale"],
            "prior_temperature": fields["prior_temperature"],
            "value_readout": fields["value_readout"],
        }
    )
    lock = {
        "science": {"search_operator": operator, "evaluator": evaluator},
        "generation": generation,
        "checkpoints": [
            {"role": "producer", "path": "/tmp/champion.pt", "sha256": "0" * 64}
        ],
    }
    job = {
        "output_dir": "/tmp/coherent-wave",
        "attempts": 1,
        "base_seed": 7_000_000_000,
        "claim_label": "test-coherent",
        "category": "current_producer",
    }
    argv = contract._generator_argv(lock, job, mix_paths={})  # noqa: SLF001
    assert "--coherent-public-belief-search" in argv
    assert argv[argv.index("--forced-root-target-mode") + 1] == "trajectory_only"
    assert argv[argv.index("--temperature-clock") + 1] == "nonforced_choice"
    assert "--no-record-automatic-transitions" in argv
    assert "--meaningful-public-history" in argv
    assert argv[argv.index("--event-history-limit") + 1] == "32"
    cli = contract._expected_cli_fields(lock, job)  # noqa: SLF001
    assert cli["coherent_public_belief_search"] is True
    assert cli["forced_root_target_mode"] == "trajectory_only"
    assert cli["temperature_clock"] == "nonforced_choice"
    assert cli["record_automatic_transitions"] is False
    assert cli["meaningful_public_history"] is True
    assert cli["event_history_limit"] == 32

    # S1 owns the independent c_scale receipt; this test isolates the guard's
    # exact coherent operator/temperature binding from that evidence fixture.
    monkeypatch.setattr(
        contract, "_validate_guard_sync_provenance", lambda *args, **kwargs: None
    )
    contract._validate_guard_payload(  # noqa: SLF001
        json.loads(GUARD.read_text()),
        path=GUARD,
        search=operator,
        evaluator=evaluator,
        generation=generation,
    )

    mismatched = dict(post_wave)
    mismatched["require_target_information_regime"] = TARGET_INFORMATION_REGIME_PUBLIC
    with pytest.raises(contract.ContractError, match="does not match"):
        contract._validate_post_wave(  # noqa: SLF001
            mismatched,
            expected_target_information_regime=TARGET_INFORMATION_REGIME_PUBLIC_COHERENT,
        )
