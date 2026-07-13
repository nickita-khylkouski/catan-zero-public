from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = _ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import fixed_root_search_stability as stability_module  # type: ignore  # noqa: E402
from fixed_root_search_stability import (  # type: ignore  # noqa: E402
    CHANCE_SEED_XOR,
    COLORS,
    EVALUATOR_CONFIG_SCHEMA,
    PANEL_SCHEMA,
    SEARCH_CONFIG_SCHEMA,
    SNAPSHOT_CANONICALIZATION,
    CountingEvaluator,
    aggregate_report_slices,
    build_root_panel,
    build_seed_manifests,
    content_sha256,
    jensen_shannon_divergence,
    load_evaluator_spec,
    load_search_spec,
    seal_root_panel,
    reconstruct_roots,
    summarize_cross_seed_runs,
    validate_search_comparison,
    validate_root_panel_payload,
    verify_locked_files,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _search_config(path: Path, *, name: str, **overrides) -> Path:
    return _write_json(
        path,
        {
            "schema_version": SEARCH_CONFIG_SCHEMA,
            "name": name,
            "search_config": {
                "n_full": 64,
                "n_fast": 64,
                "p_full": 1.0,
                "temperature": 0.0,
                **overrides,
            },
        },
    )


def _evaluator_config(path: Path, *, cache_size: int = 0) -> Path:
    return _write_json(
        path,
        {
            "schema_version": EVALUATOR_CONFIG_SCHEMA,
            "evaluator_config": {
                "cache_size": cache_size,
                "public_observation": True,
                "value_readout": "scalar",
            },
        },
    )


def _run(seed: int, policy: dict[int, float], **cost) -> dict:
    selected_action = cost.get(
        "selected_action",
        max(policy, key=lambda action: (float(policy[action]), -int(action))),
    )
    return {
        "search_seed": seed,
        "selected_action": selected_action,
        "improved_policy": policy,
        "simulations_used": cost.get("simulations_used", 64),
        "wall_sec": cost.get("wall_sec", 1.0),
        "logical_leaf_evaluations": cost.get("logical_leaf_evaluations", 100),
        "orientation_evaluation_rows": cost.get("orientation_evaluation_rows", 111),
        "evaluator_method_calls": cost.get("evaluator_method_calls", 90),
        "evaluate_calls": cost.get("evaluate_calls", 80),
        "evaluate_many_calls": cost.get("evaluate_many_calls", 9),
        "symmetry_calls": cost.get("symmetry_calls", 1),
        "target_top_probability": cost.get(
            "target_top_probability", max(policy.values())
        ),
        "target_entropy": cost.get(
            "target_entropy",
            -sum(value * math.log(value) for value in policy.values() if value > 0),
        ),
        "prior_top_probability": cost.get("prior_top_probability", 0.5),
        "prior_entropy": cost.get("prior_entropy", math.log(2.0)),
        "target_prior_js": cost.get("target_prior_js", 0.01),
        "completed_q_range": cost.get("completed_q_range", 0.02),
        "completed_q_top_margin": cost.get("completed_q_top_margin", 1.0e-6),
    }


def _role(runs: list[dict]) -> dict:
    return {"runs": runs, "stability": summarize_cross_seed_runs(runs)}


def _root_record(index: int, width: int, phase: str, a_runs: list[dict], b_runs: list[dict]):
    return {
        "root_index": index,
        "legal_width": width,
        "legal_width_bucket": "21-40" if width <= 40 else "41+",
        "wide_ge_40": width >= 40,
        "phase": phase,
        "phase_raw": phase.upper(),
        "roles": {"n64": _role(a_runs), "n128": _role(b_runs)},
    }


def _panel(checkpoint_hash: str, evaluator_hash: str) -> dict:
    material = {
        "snapshot": {"current_prompt": "PLAY_TURN", "turn": 2},
        "legal_action_ids": [1, 2],
        "current_color": "RED",
    }
    root = {
        "root_index": 0,
        "game_seed": 11,
        "decision_index": 0,
        "action_prefix": [],
        **material,
        "legal_width": 2,
        "legal_width_bucket": "2-4",
        "wide_ge_40": False,
        "phase_raw": "PLAY_TURN",
        "phase": "play_turn",
        "root_sha256": content_sha256(material),
    }
    return seal_root_panel(
        {
            "schema_version": PANEL_SCHEMA,
            "provenance": {
                "checkpoint_sha256": checkpoint_hash,
                "evaluator_config_sha256": evaluator_hash,
                "colors": list(COLORS),
                "chance_seed_xor": CHANCE_SEED_XOR,
                "snapshot_canonicalization": SNAPSHOT_CANONICALIZATION,
            },
            "root_count": 1,
            "wide_ge_40_count": 0,
            "roots": [root],
        }
    )


def test_jensen_shannon_is_symmetric_zero_and_bounded():
    p = {1: 0.8, 2: 0.2}
    q = {1: 0.1, 2: 0.9}
    assert jensen_shannon_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    assert jensen_shannon_divergence(p, q) == pytest.approx(
        jensen_shannon_divergence(q, p)
    )
    assert 0.0 < jensen_shannon_divergence(p, q) <= math.log(2.0)
    assert jensen_shannon_divergence({1: 1.0}, {2: 1.0}) == pytest.approx(
        math.log(2.0), rel=1e-9
    )


def test_jensen_shannon_rejects_invalid_mass():
    with pytest.raises(ValueError, match="non-negative"):
        jensen_shannon_divergence({1: -0.1, 2: 1.1}, {1: 0.5, 2: 0.5})
    with pytest.raises(ValueError, match="positive finite mass"):
        jensen_shannon_divergence({1: 0.0}, {1: 1.0})


def test_cross_seed_summary_uses_all_pairs_and_top1():
    runs = [
        _run(10, {1: 0.8, 2: 0.2}),
        _run(11, {1: 0.7, 2: 0.3}),
        _run(12, {1: 0.2, 2: 0.8}),
    ]
    summary = summarize_cross_seed_runs(runs)
    assert summary["pair_count"] == 3
    assert summary["top1_pair_agreement"] == pytest.approx(1 / 3)
    assert summary["top1_modal_fraction"] == pytest.approx(2 / 3)
    assert summary["cross_seed_js_mean"] > 0.0


def test_cross_seed_top1_uses_production_selected_action_on_policy_ties():
    runs = [
        _run(10, {1: 0.5, 2: 0.5}, selected_action=1),
        _run(11, {1: 0.5, 2: 0.5}, selected_action=2),
    ]
    summary = summarize_cross_seed_runs(runs)
    assert summary["policy_argmax_actions"] == [1, 1]
    assert summary["top1_actions"] == [1, 2]
    assert summary["pairwise"][0]["policy_argmax_agreement"] is True
    assert summary["top1_pair_agreement"] == 0.0


def test_cross_seed_summary_rejects_selected_action_outside_support():
    with pytest.raises(ValueError, match="outside the improved-policy support"):
        summarize_cross_seed_runs(
            [
                _run(10, {1: 1.0}, selected_action=1),
                _run(11, {1: 1.0}, selected_action=2),
            ]
        )


def test_cross_seed_summary_fails_on_seed_or_support_drift():
    with pytest.raises(ValueError, match="distinct search seeds"):
        summarize_cross_seed_runs(
            [_run(10, {1: 1.0}), _run(10, {1: 1.0})]
        )
    with pytest.raises(ValueError, match="incompatible legal-action supports"):
        summarize_cross_seed_runs(
            [_run(10, {1: 1.0}), _run(11, {2: 1.0})]
        )


def test_seed_manifests_are_disjoint_and_hashed():
    manifests = build_seed_manifests(
        n_roots=3,
        repeats=4,
        base_a=100,
        base_b=1000,
        name_a="n64",
        name_b="n128",
    )
    a = {seed for row in manifests["n64"]["seeds_by_root"] for seed in row}
    b = {seed for row in manifests["n128"]["seeds_by_root"] for seed in row}
    assert len(a) == len(b) == 12
    assert not (a & b)
    assert manifests["n64"]["seed_set_sha256"].startswith("sha256:")
    with pytest.raises(ValueError, match="must be disjoint"):
        build_seed_manifests(
            n_roots=2,
            repeats=3,
            base_a=100,
            base_b=102,
            name_a="a",
            name_b="b",
        )


def test_search_config_is_named_hashed_and_seed_is_probe_owned(tmp_path):
    path = _search_config(tmp_path / "n128.json", name="global_n128", n_full=128)
    spec = load_search_spec(path)
    assert spec["name"] == "global_n128"
    assert spec["effective_search_config"]["n_full"] == 128
    assert spec["effective_search_config"]["temperature"] == 0.0
    assert spec["source_file_sha256"].startswith("sha256:")
    assert spec["effective_search_config_sha256"].startswith("sha256:")

    payload = json.loads(path.read_text())
    payload["search_config"]["seed"] = 7
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="seed is forbidden"):
        load_search_spec(path)


def test_s3_search_config_maps_independent_d6_and_adaptive_wide_gates(tmp_path):
    path = _search_config(
        tmp_path / "adaptive_n256.json",
        name="adaptive_n256",
        n_full=128,
        n_fast=16,
        p_full=0.25,
        symmetry_averaged_eval=True,
        symmetry_averaged_eval_threshold=20,
        n_full_wide=256,
        n_full_wide_threshold=40,
        wide_roots_always_full=True,
    )
    spec = load_search_spec(path)
    config = stability_module._make_search_config(spec, seed=91_337)
    assert config.seed == 91_337
    assert config.colors == COLORS
    assert config.n_full == 128
    assert config.n_fast == 16
    assert config.symmetry_averaged_eval is True
    assert config.symmetry_averaged_eval_threshold == 20
    assert config.n_full_wide == 256
    assert config.n_full_wide_threshold == 40
    assert config.wide_roots_always_full is True


def test_checked_in_s3_r3_profiles_bind_only_the_intended_search_dose():
    config_dir = _ROOT / "configs" / "experiments" / "s3_fixed_root_r3"

    n128_p4 = load_search_spec(config_dir / "global_n128_p4.json")
    n256_p4 = load_search_spec(config_dir / "global_n256_p4.json")
    assert validate_search_comparison(
        n128_p4,
        n256_p4,
        allowed_differences={"n_full"},
    ) == {
        "n_full": {"global_n128_p4": 128, "global_n256_p4": 256}
    }
    assert n128_p4["effective_search_config"]["determinization_particles"] == 4
    assert n256_p4["effective_search_config"]["determinization_particles"] == 4

    n256_sigma_matched = load_search_spec(
        config_dir / "global_n256_p4_sigma_matched.json"
    )
    assert validate_search_comparison(
        n128_p4,
        n256_sigma_matched,
        allowed_differences={"c_scale", "n_full"},
    ) == {
        "c_scale": {
            "global_n128_p4": 0.1,
            "global_n256_p4_sigma_matched": 0.08307692307692308,
        },
        "n_full": {
            "global_n128_p4": 128,
            "global_n256_p4_sigma_matched": 256,
        },
    }

    n128_p8 = load_search_spec(config_dir / "global_n128_requested_p8.json")
    adaptive_p8 = load_search_spec(
        config_dir / "adaptive_n256_wide40_requested_p8.json"
    )
    assert validate_search_comparison(
        n128_p8,
        adaptive_p8,
        allowed_differences={
            "n_full_wide",
            "n_full_wide_threshold",
            "wide_roots_always_full",
        },
    ) == {
        "n_full_wide": {
            "global_n128_requested_p8": None,
            "adaptive_n256_wide40_requested_p8": 256,
        },
        "n_full_wide_threshold": {
            "global_n128_requested_p8": None,
            "adaptive_n256_wide40_requested_p8": 40,
        },
        "wide_roots_always_full": {
            "global_n128_requested_p8": False,
            "adaptive_n256_wide40_requested_p8": True,
        },
    }
    assert n128_p8["effective_search_config"]["determinization_particles"] == 8
    assert adaptive_p8["effective_search_config"]["determinization_particles"] == 8

    evaluator = load_evaluator_spec(config_dir / "evaluator_public_scalar.json")
    assert evaluator["effective_evaluator_config"]["cache_size"] == 0
    assert evaluator["effective_evaluator_config"]["public_observation"] is True


def test_search_config_rejects_unknown_or_stochastic_fields(tmp_path):
    path = _search_config(tmp_path / "bad.json", name="bad", mystery_arm=True)
    with pytest.raises(ValueError, match="unknown Gumbel"):
        load_search_spec(path)
    path = _search_config(tmp_path / "temp.json", name="temp", temperature=0.5)
    with pytest.raises(ValueError, match="temperature must be 0"):
        load_search_spec(path)


def test_comparison_rejects_undeclared_operator_drift(tmp_path):
    n64 = load_search_spec(_search_config(tmp_path / "n64.json", name="n64"))
    n128 = load_search_spec(
        _search_config(
            tmp_path / "n128.json",
            name="n128",
            n_full=128,
            n_fast=128,
            c_scale=0.3,
        )
    )
    with pytest.raises(ValueError, match="outside the predeclared"):
        validate_search_comparison(
            n64, n128, allowed_differences={"n_full", "n_fast"}
        )
    differences = validate_search_comparison(
        n64,
        n128,
        allowed_differences={"n_full", "n_fast", "c_scale"},
    )
    assert set(differences) == {"n_full", "n_fast", "c_scale"}


def test_evaluator_config_requires_cache_zero_and_resolves_defaults(tmp_path):
    valid = load_evaluator_spec(_evaluator_config(tmp_path / "eval.json"))
    assert valid["effective_evaluator_config"]["cache_size"] == 0
    assert valid["effective_evaluator_config"]["public_observation"] is True
    assert valid["effective_evaluator_config_sha256"].startswith("sha256:")
    with pytest.raises(ValueError, match="cache_size must be exactly 0"):
        load_evaluator_spec(_evaluator_config(tmp_path / "cached.json", cache_size=1))


def test_locked_input_rehash_detects_mid_run_drift(tmp_path):
    path = tmp_path / "locked.json"
    path.write_text("before", encoding="utf-8")
    locked = {str(path): "sha256:" + hashlib.sha256(b"before").hexdigest()}
    verify_locked_files(locked)
    path.write_text("after", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed during run"):
        verify_locked_files(locked)


def test_panel_hash_checkpoint_and_evaluator_provenance_fail_closed():
    panel = _panel("sha256:checkpoint", "sha256:evaluator")
    validate_root_panel_payload(
        panel,
        checkpoint_sha256="sha256:checkpoint",
        evaluator_config_sha256="sha256:evaluator",
    )
    with pytest.raises(ValueError, match="checkpoint hash"):
        validate_root_panel_payload(
            panel,
            checkpoint_sha256="sha256:other",
            evaluator_config_sha256="sha256:evaluator",
        )
    with pytest.raises(ValueError, match="evaluator config hash"):
        validate_root_panel_payload(
            panel,
            checkpoint_sha256="sha256:checkpoint",
            evaluator_config_sha256="sha256:other",
        )


def test_panel_tampering_is_detected_before_search():
    panel = _panel("sha256:checkpoint", "sha256:evaluator")
    panel["roots"][0]["legal_action_ids"] = [1, 3]
    with pytest.raises(ValueError, match="panel content hash mismatch"):
        validate_root_panel_payload(
            panel,
            checkpoint_sha256="sha256:checkpoint",
            evaluator_config_sha256="sha256:evaluator",
        )


def test_resealed_panel_with_inconsistent_derived_metadata_is_rejected():
    panel = _panel("sha256:checkpoint", "sha256:evaluator")
    panel.pop("panel_content_sha256")
    panel["wide_ge_40_count"] = 1
    panel = seal_root_panel(panel)
    with pytest.raises(ValueError, match="wide_ge_40_count"):
        validate_root_panel_payload(
            panel,
            checkpoint_sha256="sha256:checkpoint",
            evaluator_config_sha256="sha256:evaluator",
        )


class _FakeGame:
    def __init__(self, seed: int, decision: int = 0):
        self.seed = int(seed)
        self.decision = int(decision)

    def copy(self):
        return _FakeGame(self.seed, self.decision)

    def winning_color(self):
        return None

    def current_color(self):
        return "RED" if self.decision % 2 == 0 else "BLUE"

    def playable_action_indices(self, _colors, _map_kind):
        return [10 + self.decision, 20 + self.decision]

    def json_snapshot(self):
        return json.dumps(
            {
                "current_prompt": (
                    "BUILD_INITIAL_SETTLEMENT" if self.decision == 0 else "PLAY_TURN"
                ),
                "seed": self.seed,
                "decision": self.decision,
            }
        )


class _FakeRustModule:
    class Game:
        @staticmethod
        def simple(_colors, *, seed):
            return _FakeGame(seed)


class _FakeRawEvaluator:
    def evaluate(self, _game, legal, **_kwargs):
        return ({int(action): float(index + 1) for index, action in enumerate(legal)}, 0.0)


def test_real_root_panel_transcript_round_trips_before_search(monkeypatch):
    monkeypatch.setattr(stability_module, "_require_rust_module", lambda: _FakeRustModule)

    def _advance(game, action, **_kwargs):
        assert int(action) in game.playable_action_indices(list(COLORS), None)
        return _FakeGame(game.seed, game.decision + 1)

    monkeypatch.setattr(stability_module, "_apply_selected_action", _advance)
    panel = build_root_panel(
        _FakeRawEvaluator(),
        checkpoint_sha256="sha256:checkpoint",
        evaluator_config_sha256="sha256:evaluator",
        n_roots=2,
        decisions_per_game=(0, 1),
        base_seed=50,
        min_legal_actions=2,
    )
    validate_root_panel_payload(
        panel,
        checkpoint_sha256="sha256:checkpoint",
        evaluator_config_sha256="sha256:evaluator",
    )
    roots = reconstruct_roots(panel)
    assert [root.decision for root in roots] == [0, 1]
    assert panel["roots"][1]["action_prefix"] == [20]
    assert panel["roots"][0]["phase"] == "opening_placement"


class _NonCanonicalGame(_FakeGame):
    def __init__(self, seed: int, *, reverse_unordered: bool):
        super().__init__(seed)
        self.reverse_unordered = bool(reverse_unordered)

    def playable_action_indices(self, _colors, _map_kind):
        actions = [10, 20]
        return list(reversed(actions)) if self.reverse_unordered else actions

    def json_snapshot(self):
        tiles = [
            {"coordinate": [-1, 0, 1], "tile": {"id": 0}},
            {"coordinate": [0, 0, 0], "tile": {"id": 1}},
        ]
        nodes = {
            "1": {
                "id": 1,
                "building": "SETTLEMENT",
                "color": "RED",
                "tile_coordinate": [0, 0, 0],
                "direction": "N",
            },
            "2": {
                "id": 2,
                "building": None,
                "color": None,
                "tile_coordinate": [0, 0, 0],
                "direction": "E",
            },
        }
        edges = [
            {
                "id": [2, 1],
                "color": "RED",
                "tile_coordinate": [0, 0, 0],
                "direction": "NE",
            },
            {
                "id": [3, 2],
                "color": None,
                "tile_coordinate": [0, 0, 0],
                "direction": "E",
            },
        ]
        actions = [
            ["RED", "BUILD_SETTLEMENT", 10],
            ["RED", "BUILD_SETTLEMENT", 20],
        ]
        bot_colors = ["RED", "BLUE"]
        neighbors = [[-1, 0, 1], [0, -1, 1]]
        if self.reverse_unordered:
            tiles.reverse()
            edges.reverse()
            actions.reverse()
            bot_colors.reverse()
            neighbors.reverse()
            # Same shared nodes/edges, but Rust's map traversal chose another
            # tile/direction representative for their redundant metadata.
            nodes = dict(reversed(list(nodes.items())))
            nodes["1"]["tile_coordinate"] = [-1, 0, 1]
            nodes["1"]["direction"] = "SW"
            nodes["2"]["tile_coordinate"] = [0, -1, 1]
            nodes["2"]["direction"] = "NW"
            edges[0]["tile_coordinate"] = [0, -1, 1]
            edges[0]["direction"] = "W"
            edges[1]["tile_coordinate"] = [-1, 0, 1]
            edges[1]["direction"] = "SE"
        return json.dumps(
            {
                "current_prompt": "BUILD_INITIAL_SETTLEMENT",
                "seed": self.seed,
                "tiles": tiles,
                "nodes": nodes,
                "edges": edges,
                "current_playable_actions": actions,
                "bot_colors": bot_colors,
                "adjacent_tiles": {"0,0,0": neighbors},
                # These sequences are semantic and must remain untouched.
                "colors": ["RED", "BLUE"],
                "action_records": [["RED", "ROLL", None]],
            }
        )


class _NonCanonicalRustModule:
    construction_count = 0

    class Game:
        @staticmethod
        def simple(_colors, *, seed):
            reverse = bool(_NonCanonicalRustModule.construction_count % 2)
            _NonCanonicalRustModule.construction_count += 1
            return _NonCanonicalGame(seed, reverse_unordered=reverse)


def test_decision_zero_reconstruction_ignores_nonsemantic_rust_json_order(monkeypatch):
    _NonCanonicalRustModule.construction_count = 0
    monkeypatch.setattr(
        stability_module, "_require_rust_module", lambda: _NonCanonicalRustModule
    )
    panel = build_root_panel(
        _FakeRawEvaluator(),
        checkpoint_sha256="sha256:checkpoint",
        evaluator_config_sha256="sha256:evaluator",
        n_roots=1,
        decisions_per_game=(0,),
        base_seed=50,
        min_legal_actions=2,
    )
    validate_root_panel_payload(
        panel,
        checkpoint_sha256="sha256:checkpoint",
        evaluator_config_sha256="sha256:evaluator",
    )
    roots = reconstruct_roots(panel)
    assert len(roots) == 1
    record = panel["roots"][0]
    assert record["legal_action_ids"] == [10, 20]
    assert [tile["tile"]["id"] for tile in record["snapshot"]["tiles"]] == [0, 1]
    assert record["snapshot"]["nodes"] == [
        {"id": 1, "building": "SETTLEMENT", "color": "RED"},
        {"id": 2, "building": None, "color": None},
    ]
    assert record["snapshot"]["edges"] == [
        {"id": [1, 2], "color": "RED"},
        {"id": [2, 3], "color": None},
    ]
    assert record["snapshot"]["colors"] == ["RED", "BLUE"]


class _FakeEvaluator:
    def evaluate(self, *_args, **_kwargs):
        return "one"

    def evaluate_many(self, requests, *_args, **_kwargs):
        return ["many"] * len(requests)

    def evaluate_symmetry_averaged(self, *_args, **_kwargs):
        return "d6"


def test_counting_evaluator_reports_logical_and_orientation_cost():
    evaluator = CountingEvaluator(_FakeEvaluator())
    before = evaluator.snapshot()
    assert evaluator.evaluate(None) == "one"
    assert evaluator.evaluate_many([1, 2, 3]) == ["many"] * 3
    assert evaluator.evaluate_symmetry_averaged(None) == "d6"
    delta = CountingEvaluator.delta(before, evaluator.snapshot())
    assert delta == {
        "logical_leaf_evaluations": 5,
        "orientation_evaluation_rows": 16,
        "evaluator_method_calls": 3,
        "evaluate_calls": 1,
        "evaluate_many_calls": 1,
        "symmetry_calls": 1,
    }


def test_aggregate_slices_include_exact_cost_and_ge40_slice():
    stable_a = [_run(1, {1: 0.8, 2: 0.2}), _run(2, {1: 0.7, 2: 0.3})]
    unstable_b = [
        _run(101, {1: 0.9, 2: 0.1}, simulations_used=128, wall_sec=1.5),
        _run(102, {1: 0.1, 2: 0.9}, simulations_used=128, wall_sec=1.5),
    ]
    records = [
        _root_record(0, 20, "play_turn", stable_a, unstable_b),
        _root_record(1, 40, "opening_placement", stable_a, unstable_b),
        _root_record(2, 54, "opening_placement", stable_a, unstable_b),
    ]
    slices = aggregate_report_slices(records, "n64", "n128")
    assert slices["global"]["roots"] == 3
    assert slices["wide_ge_40"]["roots"] == 2  # inclusive, as S3 specifies.
    assert slices["by_legal_width_bucket"]["41+"]["roots"] == 1
    global_roles = slices["global"]["by_role"]
    assert global_roles["n64"]["simulations_used"] == 3 * 2 * 64
    assert global_roles["n128"]["simulations_used"] == 3 * 2 * 128
    comparison = slices["global"]["comparison"]
    assert comparison["role_b_over_role_a_simulations_ratio"] == pytest.approx(2.0)
    assert comparison["role_b_minus_role_a_top1_agreement"] < 0.0
    assert global_roles["n64"]["target_top_probability"]["mean"] == pytest.approx(
        0.75
    )
    assert global_roles["n64"]["completed_q_top_margin"]["median"] == pytest.approx(
        1.0e-6
    )


def test_cli_help_is_cpu_only_and_imports_the_local_source_tree():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(_TOOLS / "fixed_root_search_stability.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "--create-root-panel" in result.stdout
    assert "--min-wide-roots" in result.stdout
    assert "--allowed-search-config-differences" in result.stdout

    source_result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy,sys; "
                f"sys.argv=[{str(_TOOLS / 'fixed_root_search_stability.py')!r},'--help']; "
                "\ntry: runpy.run_path(sys.argv[0], run_name='__main__')"
                "\nexcept SystemExit: pass"
                "\nimport catan_zero.search.gumbel_chance_mcts as m; print(m.__file__)"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        cwd=_ROOT,
    )
    assert source_result.returncode == 0, source_result.stderr
    assert str((_ROOT / "src").resolve()) in source_result.stdout
