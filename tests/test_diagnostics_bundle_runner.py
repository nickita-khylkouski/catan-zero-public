from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from diagnostics_bundle_runner import build_bundle, main  # type: ignore  # noqa: E402


def _search_snr_json() -> dict:
    return {
        "measurement": "search_snr_probe",
        "checkpoints": ["ckpt-old", "ckpt-new"],
        "per_checkpoint": {
            "ckpt-old": {
                "aggregate": {
                    "n_states": 200,
                    "mean_argmax_agreement": 0.9,
                    "mean_kl_pi_vs_prior": 0.20,
                },
                "per_state": [],
            },
            "ckpt-new": {
                "aggregate": {
                    "n_states": 200,
                    "mean_argmax_agreement": 0.6,
                    "mean_kl_pi_vs_prior": 0.21,
                },
                "per_state": [],
            },
        },
    }


def _rollout_doubling_json(win_rate: float = 0.52) -> dict:
    return {
        "measurement": "rollout_doubling_probe",
        "ran": True,
        "rollout_doubling_summary": {
            "candidate_win_rate": win_rate,
            "pentanomial_sprt": {"model": "pentanomial", "decision": "continue"},
            "pair_diagnostics": {"ww_pairs": 40, "ll_pairs": 35, "split_pairs": 100, "incomplete_pairs": 0},
        },
    }


def _diversity_jsons() -> list[dict]:
    return [
        {
            "generation_label": "v3a",
            "unique_state_fraction_cheap": {"unique_fraction": 0.99},
            "unique_state_fraction_content": {"unique_fraction": 0.97},
            "opening_line_concentration": {"herfindahl_index": 0.3, "top1_fraction": 0.4},
            "opening_entropy": {"mean_normalized_entropy": 0.7},
        },
        {
            "generation_label": "gen-1",
            "unique_state_fraction_cheap": {"unique_fraction": 0.95},
            "unique_state_fraction_content": {"unique_fraction": 0.90},
            "opening_line_concentration": {"herfindahl_index": 0.4, "top1_fraction": 0.5},
            "opening_entropy": {"mean_normalized_entropy": 0.6},
        },
    ]


def _noise_spread_json() -> dict:
    return {
        "measurement": "noise_vs_spread_trend",
        "generations_ordered": ["v3a", "gen-1"],
        "trend": {"top5_q_spread_proxy_slope": 0.1, "orientation_noise_std_slope": 0.01},
        "pearson_correlation": {"top5_q_spread_proxy_vs_orientation_noise_std": 0.9},
    }


def test_fully_missing_inputs_leaves_weights_null_with_rationale():
    bundle = build_bundle(
        search_snr_json=None,
        rollout_doubling_json=None,
        diversity_jsons=[],
        noise_spread_json=None,
        search_snr_path="/does/not/exist/search_snr.json",
        rollout_doubling_path="/does/not/exist/rollout.json",
        diversity_paths=[],
        noise_spread_path="/does/not/exist/noise.json",
    )
    conclusion = bundle["mechanism_weight_conclusion"]
    assert conclusion["weight_A_snr_decay"] is None
    assert conclusion["weight_B_exit_fixed_point"] is None
    assert conclusion["weight_C_distribution_narrowing"] is None
    assert "insufficient data" in conclusion["rationale"]
    assert "search_snr_probe" in conclusion["rationale"]
    assert not bundle["inputs_present"]["search_snr_probe"]
    assert not bundle["inputs_present"]["rollout_doubling_probe"]
    assert not bundle["inputs_present"]["corpus_diversity_scan"]
    assert not bundle["inputs_present"]["noise_vs_spread_trend"]


def test_partially_missing_inputs_still_leaves_weights_null():
    bundle = build_bundle(
        search_snr_json=_search_snr_json(),
        rollout_doubling_json=_rollout_doubling_json(),
        diversity_jsons=[],  # missing
        noise_spread_json=_noise_spread_json(),
    )
    conclusion = bundle["mechanism_weight_conclusion"]
    assert conclusion["weight_A_snr_decay"] is None
    assert "corpus_diversity_scan" in conclusion["rationale"]


def test_all_four_inputs_present_populates_weights_summing_to_one():
    bundle = build_bundle(
        search_snr_json=_search_snr_json(),
        rollout_doubling_json=_rollout_doubling_json(win_rate=0.52),
        diversity_jsons=_diversity_jsons(),
        noise_spread_json=_noise_spread_json(),
    )
    conclusion = bundle["mechanism_weight_conclusion"]
    weight_a = conclusion["weight_A_snr_decay"]
    weight_b = conclusion["weight_B_exit_fixed_point"]
    weight_c = conclusion["weight_C_distribution_narrowing"]
    assert weight_a is not None and weight_b is not None and weight_c is not None
    # score_A = clamp(agreement_decay(0.3) - |kl_drift|(0.01)) = 0.29
    # score_B = clamp(0.5 - |0.52-0.5|) = 0.48
    # score_C = mean(herfindahl 0.3, 0.4) = 0.35
    # total = 1.12; weight_A = 0.29/1.12
    assert weight_a == pytest.approx(0.29 / 1.12, abs=1e-6)
    total = weight_a + weight_b + weight_c
    assert total == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= weight_a <= 1.0
    assert 0.0 <= weight_b <= 1.0
    assert 0.0 <= weight_c <= 1.0
    assert "STARTING POINT" in conclusion["rationale"]
    assert bundle["inputs_present"]["search_snr_probe"]
    assert bundle["inputs_present"]["rollout_doubling_probe"]
    assert bundle["inputs_present"]["corpus_diversity_scan"]
    assert bundle["inputs_present"]["noise_vs_spread_trend"]


def test_measurement_1_summary_extracts_checkpoint_aggregates():
    bundle = build_bundle(
        search_snr_json=_search_snr_json(),
        rollout_doubling_json=_rollout_doubling_json(),
        diversity_jsons=_diversity_jsons(),
        noise_spread_json=_noise_spread_json(),
    )
    m1 = bundle["measurement_1_search_snr_probe"]
    assert m1["checkpoints"] == ["ckpt-old", "ckpt-new"]
    assert m1["aggregate_by_checkpoint"]["ckpt-old"]["mean_argmax_agreement"] == pytest.approx(0.9)


def test_measurement_2_summary_extracts_win_rate():
    bundle = build_bundle(
        search_snr_json=_search_snr_json(),
        rollout_doubling_json=_rollout_doubling_json(win_rate=0.55),
        diversity_jsons=_diversity_jsons(),
        noise_spread_json=_noise_spread_json(),
    )
    m2 = bundle["measurement_2_rollout_doubling_probe"]
    assert m2["candidate_win_rate"] == pytest.approx(0.55)


def test_measurement_3_summary_lists_per_generation_diversity():
    bundle = build_bundle(
        search_snr_json=_search_snr_json(),
        rollout_doubling_json=_rollout_doubling_json(),
        diversity_jsons=_diversity_jsons(),
        noise_spread_json=_noise_spread_json(),
    )
    m3 = bundle["measurement_3_corpus_diversity_scan"]
    assert len(m3) == 2
    assert m3[0]["generation_label"] == "v3a"
    assert m3[1]["opening_line_herfindahl"] == pytest.approx(0.4)


def test_cli_end_to_end_with_all_inputs(tmp_path):
    search_snr_path = tmp_path / "search_snr.json"
    rollout_path = tmp_path / "rollout.json"
    diversity_path_1 = tmp_path / "diversity_v3a.json"
    diversity_path_2 = tmp_path / "diversity_gen1.json"
    noise_path = tmp_path / "noise.json"
    out_path = tmp_path / "bundle.json"

    search_snr_path.write_text(json.dumps(_search_snr_json()))
    rollout_path.write_text(json.dumps(_rollout_doubling_json()))
    diversity_jsons = _diversity_jsons()
    diversity_path_1.write_text(json.dumps(diversity_jsons[0]))
    diversity_path_2.write_text(json.dumps(diversity_jsons[1]))
    noise_path.write_text(json.dumps(_noise_spread_json()))

    old_argv = sys.argv
    sys.argv = [
        "diagnostics_bundle_runner.py",
        "--search-snr-json",
        str(search_snr_path),
        "--rollout-doubling-json",
        str(rollout_path),
        "--diversity-json",
        str(diversity_path_1),
        "--diversity-json",
        str(diversity_path_2),
        "--noise-spread-json",
        str(noise_path),
        "--out",
        str(out_path),
    ]
    try:
        main()
    finally:
        sys.argv = old_argv

    data = json.loads(out_path.read_text())
    assert data["bundle"] == "cat25_diagnostics_bundle"
    conclusion = data["mechanism_weight_conclusion"]
    assert conclusion["weight_A_snr_decay"] is not None


def test_cli_end_to_end_with_no_inputs(tmp_path):
    out_path = tmp_path / "bundle_empty.json"
    old_argv = sys.argv
    sys.argv = ["diagnostics_bundle_runner.py", "--out", str(out_path)]
    try:
        main()
    finally:
        sys.argv = old_argv

    data = json.loads(out_path.read_text())
    conclusion = data["mechanism_weight_conclusion"]
    assert conclusion["weight_A_snr_decay"] is None
    assert "insufficient data" in conclusion["rationale"]
