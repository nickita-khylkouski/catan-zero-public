from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catanatron_neutral_harness_match import (  # type: ignore  # noqa: E402
    ORIENTATIONS,
    _checkpoint_digests,
    _create_search,
    _game_semantics,
    _load_game_artifacts,
    _prepare_manifest,
    _run_fingerprint,
    _build_evaluator,
    _search_config,
    _search_recipe,
    _validate_checkpoint_value_readout,
    _write_game_artifact,
    build_parser,
    build_summary,
    play_one_search_game,
)
from catanatron_player_adapter import (  # type: ignore  # noqa: E402
    CatanZeroSearchPlayer,
    SearchEngineBoundaryError,
    apply_native_action_record_to_rust,
)


def test_native_hot_loop_is_explicit_and_fingerprinted(monkeypatch) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            "candidate.pt",
            "--opponent",
            "catanatron_value",
            "--mode",
            "search",
            "--native-mcts-hot-loop",
            "--out",
            "report.json",
        ]
    )
    recipe = _search_recipe(args)
    assert recipe["native_mcts_hot_loop"] is True
    assert recipe["mcts_implementation"] == "rust_native_hot_loop_v1"

    calls = []
    sentinel = object()
    monkeypatch.setattr(
        "catanatron_neutral_harness_match.create_gumbel_search",
        lambda config, evaluator, *, native_hot_loop: (
            calls.append(native_hot_loop) or sentinel
        ),
    )
    assert _create_search(object(), object(), native_mcts_hot_loop=True) is sentinel
    assert calls == [True]


class _Named:
    def __init__(self, name: str) -> None:
        self.name = name


def _native_action(action_type: str, value=None):
    return SimpleNamespace(
        color=_Named("BLUE"), action_type=_Named(action_type), value=value
    )


class _FakeRustGame:
    def __init__(self, raw_action):
        self.raw_action = raw_action
        self.executed: list[tuple[int, list[str], str]] = []
        self.chance_outcomes: list[int] = []

    def playable_action_indices(self, _colors, _map_kind):
        return [17]

    def playable_actions_json(self):
        return json.dumps([self.raw_action])

    def execute_action_index(self, action_id, colors, map_kind):
        self.executed.append((action_id, colors, map_kind))

    def spectrum_json(self, _raw_json):
        return json.dumps([{"probability": 1 / 11}] * 11)

    def apply_chance_outcome(self, _raw_json, outcome_index):
        self.chance_outcomes.append(outcome_index)
        return self


def test_replay_native_deterministic_action_uses_matching_rust_id() -> None:
    rust = _FakeRustGame(["BLUE", "END_TURN", None])
    record = SimpleNamespace(action=_native_action("END_TURN"), result=None)
    returned = apply_native_action_record_to_rust(
        rust,
        record,
        seated_colors=("BLUE", "RED"),
        map_kind="TOURNAMENT",
    )
    assert returned is rust
    assert rust.executed == [(17, ["BLUE", "RED"], "TOURNAMENT")]


def test_replay_native_roll_forces_already_resolved_dice_total() -> None:
    rust = _FakeRustGame(["BLUE", "ROLL", None])
    record = SimpleNamespace(action=_native_action("ROLL"), result=(3, 4))
    returned = apply_native_action_record_to_rust(
        rust,
        record,
        seated_colors=("BLUE", "RED"),
        map_kind="TOURNAMENT",
    )
    assert returned is rust
    assert rust.chance_outcomes == [5]  # totals 2..12 map to spectrum indices 0..10.
    assert rust.executed == []


class _Snapshot:
    def __init__(self, payload):
        self.payload = payload

    def json_snapshot(self):
        return json.dumps(self.payload)


class _SpectrumRustGame(_FakeRustGame):
    def __init__(self, raw_action, before, outcomes):
        super().__init__(raw_action)
        self.before = before
        self.outcomes = [_Snapshot(payload) for payload in outcomes]

    def json_snapshot(self):
        return json.dumps(self.before)

    def spectrum_json(self, _raw_json):
        return json.dumps(
            [{"probability": 1 / len(self.outcomes)}] * len(self.outcomes)
        )

    def apply_chance_outcome(self, _raw_json, outcome_index):
        self.chance_outcomes.append(outcome_index)
        return self.outcomes[outcome_index]


def _player_state(*, resources=None, dev_cards=None):
    return {
        "resources": resources
        or {name: 0 for name in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")},
        "dev_cards": dev_cards
        or {
            name: 0
            for name in (
                "KNIGHT",
                "YEAR_OF_PLENTY",
                "MONOPOLY",
                "ROAD_BUILDING",
                "VICTORY_POINT",
            )
        },
    }


def test_replay_native_robber_forces_recorded_stolen_resource() -> None:
    red_before = {"WOOD": 2, "BRICK": 1, "SHEEP": 0, "WHEAT": 0, "ORE": 0}
    red_after_wood = {**red_before, "WOOD": 1}
    red_after_brick = {**red_before, "BRICK": 0}
    before = {
        "colors": ["BLUE", "RED"],
        "player_state": [_player_state(), _player_state(resources=red_before)],
    }
    outcomes = [
        {
            "colors": before["colors"],
            "player_state": [_player_state(), _player_state(resources=red_after_wood)],
        },
        {
            "colors": before["colors"],
            "player_state": [_player_state(), _player_state(resources=red_after_brick)],
        },
    ]
    raw = ["BLUE", "MOVE_ROBBER", [[0, 0, 0], "RED"]]
    rust = _SpectrumRustGame(raw, before, outcomes)
    native_value = ((0, 0, 0), _Named("RED"))
    record = SimpleNamespace(
        action=_native_action("MOVE_ROBBER", native_value), result="BRICK"
    )
    returned = apply_native_action_record_to_rust(
        rust, record, seated_colors=("BLUE", "RED"), map_kind="TOURNAMENT"
    )
    assert returned is rust.outcomes[1]
    # Materialization checks both outcomes, then applies the exact match.
    assert rust.chance_outcomes == [0, 1, 1]


def test_replay_native_development_card_forces_recorded_card() -> None:
    cards_before = _player_state()["dev_cards"]
    knight = {**cards_before, "KNIGHT": 1}
    vp = {**cards_before, "VICTORY_POINT": 1}
    before = {
        "colors": ["BLUE", "RED"],
        "player_state": [_player_state(dev_cards=cards_before), _player_state()],
        "development_deck_count": 25,
    }
    outcomes = [
        {
            "colors": before["colors"],
            "player_state": [_player_state(dev_cards=knight), _player_state()],
            "development_deck_count": 24,
        },
        {
            "colors": before["colors"],
            "player_state": [_player_state(dev_cards=vp), _player_state()],
            "development_deck_count": 24,
        },
    ]
    rust = _SpectrumRustGame(["BLUE", "BUY_DEVELOPMENT_CARD", None], before, outcomes)
    record = SimpleNamespace(
        action=_native_action("BUY_DEVELOPMENT_CARD"), result="VICTORY_POINT"
    )
    returned = apply_native_action_record_to_rust(
        rust, record, seated_colors=("BLUE", "RED"), map_kind="TOURNAMENT"
    )
    assert returned is rust.outcomes[1]
    assert rust.chance_outcomes == [0, 1, 1]


def test_replay_refuses_native_action_missing_from_rust_legals() -> None:
    rust = _FakeRustGame(["BLUE", "END_TURN", None])
    record = SimpleNamespace(action=_native_action("ROLL"), result=(3, 4))
    with pytest.raises(SearchEngineBoundaryError, match="no unique Rust legal"):
        apply_native_action_record_to_rust(
            rust,
            record,
            seated_colors=("BLUE", "RED"),
            map_kind="TOURNAMENT",
        )


def test_search_player_refuses_unverified_base_map() -> None:
    with pytest.raises(ValueError, match="TOURNAMENT"):
        CatanZeroSearchPlayer(
            _Named("BLUE"),
            rust_game=object(),
            search=object(),
            seated_colors=("BLUE", "RED"),
            map_kind="BASE",
        )


def test_search_player_reuses_boundary_legals_after_search(monkeypatch) -> None:
    """The parity-checked Rust list is enough to map an immutable-root result."""
    native_action = object()
    player = object.__new__(CatanZeroSearchPlayer)
    player._rust_game = object()
    player._search = SimpleNamespace(
        search=lambda _game, *, force_full: SimpleNamespace(
            selected_action=17, simulations_used=128
        )
    )
    player.stats = {
        "decisions": 0,
        "forced_decisions": 0,
        "search_decisions": 0,
        "simulations_used": 0,
        "illegal_policy_picks": 0,
    }
    player.sync_from_native = MethodType(
        lambda _self, _game, **_kwargs: ([17], [["BLUE", "END_TURN", None]]),
        player,
    )
    monkeypatch.setattr(
        "catanatron_player_adapter.rust_legal_actions",
        lambda *_args, **_kwargs: pytest.fail("Rust legals were fetched twice"),
    )
    monkeypatch.setattr(
        "catanatron_player_adapter.raw_action_to_python_action",
        lambda *_args, **_kwargs: native_action,
    )
    monkeypatch.setattr(
        "catanatron_player_adapter.canonical_python_action_key",
        lambda action: id(action),
    )

    assert player.decide(object(), [native_action, object()]) is native_action
    assert player.stats["search_decisions"] == 1
    assert player.stats["simulations_used"] == 128


def test_parser_preserves_raw_smoke_default_and_exposes_search_recipe() -> None:
    args = build_parser().parse_args(
        ["--checkpoint", "checkpoint.pt", "--opponent", "random", "--out", "out.json"]
    )
    assert args.mode == "raw_policy"
    assert args.n_full == 64
    assert args.c_scale == 0.03
    assert args.rescale_noise_floor_c == 0.0
    assert args.sigma_eval == 0.98
    assert args.gameplay_policy_aggregation == "mean_improved_policy"
    assert args.sigma_reference_visits is None
    assert args.lazy_interior_chance is True
    assert args.public_observation is True
    assert args.information_set_search is True
    assert args.determinization_particles == 4
    assert args.determinization_min_simulations == 32
    assert args.value_readout == "scalar"
    assert args.n_full_wide is None
    assert args.n_full_wide_threshold is None
    assert args.wide_roots_always_full is False
    assert args.symmetry_averaged_eval is False
    assert args.symmetry_averaged_eval_threshold is None
    assert args.wide_candidates_threshold == 24
    assert args.evaluator_rust_featurize is False


def test_checkpoint_provenance_digests_share_one_read(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint bytes")
    md5, sha256 = _checkpoint_digests(checkpoint)
    assert md5 == "c0167a0efffa000b43385e00e45658fe"
    assert sha256 == (
        "sha256:469f693d77e177ec6267a25a17f3d1c60a1156bbc63e63d23b2c02eb15d1a38c"
    )


def test_neutral_search_runtime_and_fingerprint_share_d6_adaptive_recipe() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--opponent",
            "random",
            "--mode",
            "search",
            "--out",
            "out.json",
            "--n-full",
            "128",
            "--n-full-wide",
            "256",
            "--n-full-wide-threshold",
            "40",
            "--wide-roots-always-full",
            "--symmetry-averaged-eval",
            "--symmetry-averaged-eval-threshold",
            "20",
            "--evaluator-rust-featurize",
            "--wide-candidates-threshold",
            "24",
        ]
    )
    recipe = _search_recipe(args)
    config = _search_config(recipe, ("BLUE", "RED"), seed=7)

    assert recipe["force_full_every_decision"] is True
    assert config.n_full == 128
    assert config.n_full_wide == 256
    assert config.n_full_wide_threshold == 40
    assert config.wide_roots_always_full is True
    assert config.symmetry_averaged_eval is True
    assert config.symmetry_averaged_eval_threshold == 20
    assert config.wide_candidates_threshold == 24
    assert config.information_set_search is True
    assert config.determinization_particles == 4
    assert config.determinization_min_simulations == 32
    assert recipe["evaluator_rust_featurize"] is True

    semantics = _game_semantics(args, "checkpoint-md5", "sha256:" + "1" * 64)
    assert semantics["checkpoint_sha256"] == "sha256:" + "1" * 64
    assert semantics["search"] == recipe
    default_args = parser.parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--opponent",
            "random",
            "--mode",
            "search",
            "--out",
            "out.json",
        ]
    )
    assert _run_fingerprint(semantics) != _run_fingerprint(
        _game_semantics(default_args, "checkpoint-md5", "sha256:" + "1" * 64)
    )


def test_neutral_search_seals_corrected_belief_gameplay_operator() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--opponent",
            "random",
            "--mode",
            "search",
            "--out",
            "out.json",
            "--gameplay-policy-aggregation",
            "aggregate_q_then_improve",
            "--sigma-reference-visits",
            "8",
            "--rescale-noise-floor-c",
            "1.0",
        ]
    )
    recipe = _search_recipe(args)
    config = _search_config(recipe, ("BLUE", "RED"), seed=7)
    assert config.gameplay_policy_aggregation == "aggregate_q_then_improve"
    assert config.sigma_reference_visits == 8
    assert config.rescale_noise_floor_c == 1.0
    semantics = _game_semantics(args, "checkpoint-md5", "sha256:" + "1" * 64)
    legacy = parser.parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--opponent",
            "random",
            "--mode",
            "search",
            "--out",
            "out.json",
        ]
    )
    assert _run_fingerprint(semantics) != _run_fingerprint(
        _game_semantics(legacy, "checkpoint-md5", "sha256:" + "1" * 64)
    )


def test_retry_fingerprint_changes_across_engine_build_identity() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--opponent",
            "random",
            "--mode",
            "search",
            "--out",
            "out.json",
            "--engine-repo-commit",
            "a" * 40,
            "--native-wheel-sha256",
            "sha256:" + "b" * 64,
            "--python-referee-sha256",
            "sha256:" + "c" * 64,
        ]
    )
    args.native_runtime_sha256 = "sha256:" + "d" * 64
    first = _game_semantics(args, "checkpoint-md5", "sha256:" + "1" * 64)
    args.engine_repo_commit = "e" * 40
    second = _game_semantics(args, "checkpoint-md5", "sha256:" + "1" * 64)
    assert first["engine_identity"]["native_runtime_sha256"] == "sha256:" + "d" * 64
    assert _run_fingerprint(first) != _run_fingerprint(second)


def test_search_evaluator_receives_explicit_categorical_readout(monkeypatch) -> None:
    captured = {}

    def fake_from_checkpoint(checkpoint, *, device, config):
        captured.update(checkpoint=checkpoint, device=device, config=config)
        return object()

    monkeypatch.setattr(
        "catanatron_neutral_harness_match.BatchedEntityGraphRustEvaluator.from_checkpoint",
        fake_from_checkpoint,
    )
    _build_evaluator(
        {
            "checkpoint": "candidate.pt",
            "device": "cuda:0",
            "value_scale": 1.0,
            "prior_temperature": 1.0,
            "value_squash": "tanh",
            "value_readout": "categorical",
            "public_observation": True,
        }
    )
    assert captured["config"].value_readout == "categorical"
    assert captured["config"].cache_size == 0
    assert captured["config"].context_fill == 0.0
    assert captured["config"].rust_featurize is False
    assert captured["config"].emit_uncertainty is False


def test_neutral_harness_categorical_preflight_requires_training_provenance(
    monkeypatch,
) -> None:
    policy = SimpleNamespace(
        model=SimpleNamespace(
            value_categorical_bins=9,
            value_categorical_head=object(),
        ),
        trained_value_readouts=("scalar",),
        _checkpoint_missing_state_keys=(),
        _value_training_provenance_errors=(),
    )
    monkeypatch.setattr(
        "catanatron_neutral_harness_match.EntityGraphPolicy.load",
        lambda *_args, **_kwargs: policy,
    )
    with pytest.raises(ValueError, match="no positive value-training-v1 provenance"):
        _validate_checkpoint_value_readout(
            "config_only_cat_head.pt", value_readout="categorical"
        )

    policy.trained_value_readouts = ("categorical",)
    assert _validate_checkpoint_value_readout(
        "trained_cat_head.pt", value_readout="categorical"
    ) == ("categorical",)


def _record(pair_id: int, orientation: str, won: bool | None, **extra):
    return {
        "pair_id": pair_id,
        "game_seed": 100 + pair_id,
        "orientation": orientation,
        "candidate_won": won,
        "search_won": won,
        "terminated": won is not None,
        "truncated": won is None,
        "error": None,
        "engine_divergence": False,
        "illegal_policy_picks": 0,
        "search_decisions": 2,
        "simulations_used": 128,
        **extra,
    }


def test_per_game_artifacts_round_trip_and_reject_mixed_run(tmp_path: Path) -> None:
    semantics = {"mode": "search", "checkpoint_md5": "abc", "base_seed": 100}
    fingerprint = _run_fingerprint(semantics)
    artifact_dir = tmp_path / "games"
    _prepare_manifest(
        artifact_dir,
        fingerprint=fingerprint,
        semantics=semantics,
        pairs_requested=2,
    )
    _write_game_artifact(
        str(artifact_dir), fingerprint, _record(0, "candidate_first", True)
    )
    loaded = _load_game_artifacts(artifact_dir, fingerprint=fingerprint)
    assert loaded[(0, "candidate_first")]["candidate_won"] is True

    with pytest.raises(SystemExit, match="different run"):
        _prepare_manifest(
            artifact_dir,
            fingerprint="different",
            semantics={"mode": "raw_policy"},
            pairs_requested=2,
        )
    with pytest.raises(SystemExit, match="incompatible"):
        _load_game_artifacts(artifact_dir, fingerprint="different")


def test_summary_stays_gate_and_whr_compatible(tmp_path: Path) -> None:
    args = SimpleNamespace(
        checkpoint="checkpoint.pt",
        opponent="catanatron_value",
        mode="search",
        n_full=64,
        n_full_wide=256,
        n_full_wide_threshold=40,
        wide_roots_always_full=True,
        symmetry_averaged_eval=True,
        symmetry_averaged_eval_threshold=20,
        wide_candidates_threshold=24,
        c_scale=0.03,
        c_visit=50.0,
        lazy_interior_chance=True,
        public_observation=True,
        information_set_search=True,
        determinization_particles=4,
        determinization_min_simulations=32,
        correct_rust_chance_spectra=True,
        max_depth=80,
        max_decisions=600,
        prior_temperature=1.0,
        value_scale=1.0,
        value_squash="tanh",
        value_readout="categorical",
        max_root_candidates=16,
        max_root_candidates_wide=54,
        max_player_trade_offers_per_turn=0,
        vps_to_win=10,
        pairs=2,
        workers=2,
        threads_per_worker=1,
        gate_config="certification",
        elo0=0.0,
        elo1=30.0,
        alpha=0.05,
        beta=0.05,
        resume=True,
    )
    games = [
        _record(0, ORIENTATIONS[0], True),
        _record(0, ORIENTATIONS[1], False),
        _record(1, ORIENTATIONS[0], True),
        _record(1, ORIENTATIONS[1], True),
    ]
    summary = build_summary(
        args,
        games=games,
        checkpoint_md5="abc",
        checkpoint_sha256="sha256:" + "1" * 64,
        run_fingerprint="fingerprint",
        artifact_dir=tmp_path,
        elapsed_sec=1.5,
        games_resumed=2,
        games_run_this_invocation=2,
        worker_errors=[],
        trained_value_readouts=("scalar", "categorical"),
    )
    assert summary["candidate_checkpoint"] == "checkpoint.pt"
    assert summary["candidate_checkpoint_sha256"] == "sha256:" + "1" * 64
    assert summary["baseline_bot"] == "catanatron_value"
    assert len(summary["games"]) == 4
    assert summary["pentanomial_sprt"]["pairs"] == 2
    assert summary["pair_diagnostics"] == {
        "ww_pairs": 1,
        "split_pairs": 1,
        "ll_pairs": 0,
        "incomplete_pairs": 0,
    }
    assert summary["search_config"]["public_observation"] is True
    assert summary["public_observation"] is True
    assert summary["candidate_value_readout"] == "categorical"
    assert summary["trained_value_readouts"] == ["scalar", "categorical"]
    assert summary["search_config"]["value_readout"] == "categorical"
    assert summary["search_config"]["n_full_wide"] == 256
    assert summary["search_config"]["n_full_wide_threshold"] == 40
    assert summary["search_config"]["wide_roots_always_full"] is True
    assert summary["search_config"]["symmetry_averaged_eval"] is True
    assert summary["search_config"]["symmetry_averaged_eval_threshold"] == 20
    assert summary["n_full"] == 64
    assert summary["engine_boundary"].endswith("verified_rust_search_shadow")
    assert summary["resume"] == {
        "enabled": True,
        "games_resumed": 2,
        "games_run_this_invocation": 2,
    }


def test_native_referee_search_shadow_cpu_smoke_when_full_rust_wheel_available() -> (
    None
):
    """No checkpoint/GPU needed: exercise the real two-engine boundary with
    the deterministic heuristic evaluator when the 0.1.2+ Rust API exists.

    Developer laptops with the old minimal wheel skip cleanly; the deploy
    environment's 0.1.4 wheel runs this test rather than masking the boundary
    behind mocks.
    """
    from catan_zero.adapters.engine_equivalence import (
        RustModuleUnavailable,
        require_rust_module,
    )
    from catan_zero.search.rust_mcts import HeuristicRustEvaluator

    try:
        require_rust_module()
    except RustModuleUnavailable as error:
        pytest.skip(str(error))

    record = play_one_search_game(
        evaluator=HeuristicRustEvaluator(score_actions=False),
        search_kwargs={
            "n_full": 1,
            "max_depth": 2,
            "c_visit": 50.0,
            "c_scale": 0.03,
            "lazy_interior_chance": True,
            "correct_rust_chance_spectra": True,
            "max_root_candidates": 1,
            "max_root_candidates_wide": 1,
        },
        opponent="random",
        orientation="candidate_first",
        pair_id=0,
        game_seed=11,
        vps_to_win=10,
        max_decisions=16,
    )
    assert record["error"] is None
    assert record["engine_divergence"] is False
    assert record["decisions"] == 16
    assert record["shadow_records_synced"] == 16
    assert record["search_decisions"] > 0
