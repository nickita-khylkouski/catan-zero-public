from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from catan_zero.search.operator_runner import MeasuredDecision, SearchCounters
from tools.rnd_leaderboard import validate_and_aggregate
from tools.rnd_paired_operator_runner import (
    ArmIdentity,
    BundleRunError,
    FrozenReference,
    SEED_MANIFEST_VERSION,
    TRAINING_MANIFEST_VERSION,
    build_result_bundle,
    load_seed_manifest,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
COMMIT = "c" * 40
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "rnd"
SEED_MANIFEST_PATH = FIXTURE_DIR / "paired-seeds.json"
TRAINING_MANIFEST_PATH = FIXTURE_DIR / "training-manifest.json"


class _FakeGame:
    def __init__(self) -> None:
        self.ply = 0

    def winning_color(self):
        return "RED" if self.ply >= 4 else None

    def current_color(self):
        return ("RED", "BLUE")[self.ply % 2]

    def playable_action_indices(self, _colors, _map_kind):
        return [100 + self.ply] if self.winning_color() is None else []


class _FakeOperator:
    def __init__(self, regime: str, *, leaves: int = 2) -> None:
        self.information_regime = regime
        self.leaves = leaves
        self.require_flags: list[bool] = []
        self.prepared_seeds: list[int] = []

    def prepare_game(self, *, seed: int) -> None:
        self.prepared_seeds.append(seed)

    def run(self, game, *, require_public_information=False):
        self.require_flags.append(bool(require_public_information))
        action = 100 + game.ply
        return MeasuredDecision(
            selected_action=action,
            policy={action: 1.0},
            q_values={action: 0.0},
            root_value=0.0,
            counters=SearchCounters(
                nominal_visits=1,
                scheduled_visits=2,
                logical_leaves=self.leaves,
                orientation_rows=self.leaves,
                evaluator_calls=1,
                wall_time_sec=0.01,
            ),
            information_regime=self.information_regime,
        )


def _apply(game: _FakeGame, action: int, _rng) -> _FakeGame:
    assert action == 100 + game.ply
    game.ply += 1
    return game


def _identity(arm_id: str = "arm-a") -> ArmIdentity:
    return ArmIdentity(
        arm_id=arm_id,
        architecture_id=f"arch-{arm_id}",
        parameter_count=100,
        architecture_config_sha256=SHA_A,
        search_id=f"search-{arm_id}",
        search_config_sha256=SHA_B,
        checkpoint_path=f"/remote/{arm_id}.pt",
        checkpoint_sha256=SHA_A,
    )


def _reference() -> FrozenReference:
    return FrozenReference(
        reference_id="frozen-reference",
        architecture_id="reference-arch",
        parameter_count=200,
        architecture_config_sha256=SHA_B,
        search_id="reference-search",
        search_config_sha256=SHA_A,
        checkpoint_path="/remote/reference.pt",
        checkpoint_sha256=SHA_B,
    )


def _seed_manifest(seed_count: int = 1) -> dict:
    return {
        "path": str(SEED_MANIFEST_PATH),
        "sha256": hashlib.sha256(SEED_MANIFEST_PATH.read_bytes()).hexdigest(),
        "schema_version": SEED_MANIFEST_VERSION,
        "track": "2p_no_trade",
        "seed_count": seed_count,
    }


def _training_manifest() -> dict:
    return {
        "path": str(TRAINING_MANIFEST_PATH),
        "sha256": hashlib.sha256(TRAINING_MANIFEST_PATH.read_bytes()).hexdigest(),
        "schema_version": TRAINING_MANIFEST_VERSION,
    }


def _native_engine() -> dict:
    return {
        "engine_id": "catanatron_rs",
        "version": "0.1.4",
        "path": str(TRAINING_MANIFEST_PATH),
        "sha256": hashlib.sha256(TRAINING_MANIFEST_PATH.read_bytes()).hexdigest(),
    }


def _hardware() -> dict:
    return {
        "device": "cpu",
        "device_type": "cpu",
        "host_fingerprint": SHA_A,
        "machine": "test-machine",
        "accelerator_model": "cpu",
        "accelerator_uuid": None,
        "total_memory_bytes": None,
        "compute_capability": None,
    }


def _campaign() -> dict:
    return {
        "schema_version": "catan-zero-rnd-leaderboard/v1",
        "campaign_id": "campaign",
        "required_information_regime": "public_only",
        "require_same_training_manifest": True,
        "required_arm_ids": ["arm-a", "arm-b"],
        "budget_contracts": {
            "equal_work": {
                "match_metrics": ["logical_leaves", "orientation_rows"],
                "absolute_tolerance": 0.0,
                "relative_tolerance": 0.0,
            },
            "equal_time": {
                "match_metrics": ["wall_time_sec"],
                "absolute_tolerance": 0.0,
                "relative_tolerance": 0.05,
            },
        },
        "arms": [
            {
                "arm_id": arm_id,
                "architecture_id": f"arch-{arm_id}",
                "expected_parameter_count": 100,
                "search_id": f"search-{arm_id}",
                "source_status": "implemented",
                "measurement_adapter_status": "implemented",
                "runnable": True,
            }
            for arm_id in ("arm-a", "arm-b")
        ],
    }


def _build(arm_id: str = "arm-a") -> tuple[dict, _FakeOperator, _FakeOperator]:
    candidate = _FakeOperator("public_conservation_pimc")
    reference = _FakeOperator("public_observation_policy")
    bundle = build_result_bundle(
        candidate,
        reference,
        campaign_id="campaign",
        run_id=f"run-{arm_id}",
        budget_regime="equal_work",
        candidate_identity=_identity(arm_id),
        reference_identity=_reference(),
        seeds=[123],
        seed_manifest=_seed_manifest(),
        training_manifest=_training_manifest(),
        code_provenance={"git_commit": COMMIT, "dirty": False},
        native_engine_provenance=_native_engine(),
        hardware_provenance=_hardware(),
        max_decisions=8,
        game_factory=lambda _seed: _FakeGame(),
        action_applier=_apply,
    )
    return bundle, candidate, reference


def test_complete_fake_engine_bundle_has_exact_swaps_counters_and_regimes() -> None:
    bundle, candidate, reference = _build()

    assert bundle["track"] == "2p_no_trade"
    assert bundle["required_information_regime"] == "public_only"
    assert bundle["seed_manifest"] == _seed_manifest()
    assert bundle["training_manifest"] == _training_manifest()
    assert len(bundle["games"]) == 2
    first, second = bundle["games"]
    assert first["seat_assignment"] == {"candidate": 0, "reference": 1}
    assert second["seat_assignment"] == {"candidate": 1, "reference": 0}
    assert first["seed"] == second["seed"] == 123
    assert first["candidate_score"] == 1.0
    assert second["candidate_score"] == 0.0
    assert first["information_regime"] == "public_conservation_pimc"
    assert first["reference_information_regime"] == "public_observation_policy"
    assert first["counters"] == {
        "nominal_visits": 2,
        "scheduled_visits": 4,
        "logical_leaves": 4,
        "orientation_rows": 4,
        "evaluator_calls": 2,
        "wall_time_sec": 0.02,
    }
    assert first["reference_counters"] == first["counters"]
    assert all(candidate.require_flags)
    assert all(reference.require_flags)
    # Same role-specific search seed is restored before both seat orientations.
    assert candidate.prepared_seeds[0] == candidate.prepared_seeds[1]
    assert reference.prepared_seeds[0] == reference.prepared_seeds[1]


def test_fake_bundles_pass_the_real_aggregate_schema_and_public_gate() -> None:
    first, _candidate, _reference_operator = _build("arm-a")
    second, _candidate, _reference_operator = _build("arm-b")

    report = validate_and_aggregate(_campaign(), [first, second])

    assert report["valid"] is True
    assert report["required_information_regime"] == "public_only"
    assert report["seed_manifest"] == {**_seed_manifest(), "seeds": (123,)}
    assert report["training_manifest"] == _training_manifest()


def test_public_default_rejects_authoritative_operator_before_play() -> None:
    candidate = _FakeOperator("authoritative_hidden_state")
    reference = _FakeOperator("public_conservation_pimc")
    with pytest.raises(BundleRunError, match="both candidate and reference"):
        build_result_bundle(
            candidate,
            reference,
            campaign_id="campaign",
            run_id="run",
            budget_regime="equal_work",
            candidate_identity=_identity(),
            reference_identity=_reference(),
            seeds=[1],
            seed_manifest=_seed_manifest(),
            training_manifest=_training_manifest(),
            code_provenance={"git_commit": COMMIT, "dirty": False},
            native_engine_provenance=_native_engine(),
            hardware_provenance=_hardware(),
            max_decisions=8,
            game_factory=lambda _seed: _FakeGame(),
            action_applier=_apply,
        )


def test_incomplete_game_fails_without_returning_a_partial_bundle() -> None:
    candidate = _FakeOperator("public_conservation_pimc")
    reference = _FakeOperator("public_conservation_pimc")
    with pytest.raises(BundleRunError, match="did not terminate"):
        build_result_bundle(
            candidate,
            reference,
            campaign_id="campaign",
            run_id="run",
            budget_regime="equal_work",
            candidate_identity=_identity(),
            reference_identity=_reference(),
            seeds=[1],
            seed_manifest=_seed_manifest(),
            training_manifest=_training_manifest(),
            code_provenance={"git_commit": COMMIT, "dirty": False},
            native_engine_provenance=_native_engine(),
            hardware_provenance=_hardware(),
            max_decisions=2,
            game_factory=lambda _seed: _FakeGame(),
            action_applier=_apply,
        )


def test_seed_manifest_is_hashed_and_duplicate_seeds_fail(tmp_path: Path) -> None:
    path = tmp_path / "seeds.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": SEED_MANIFEST_VERSION,
                "track": "2p_no_trade",
                "seeds": [7, 9],
            }
        ),
        encoding="utf-8",
    )
    seeds, provenance = load_seed_manifest(path)
    assert seeds == [7, 9]
    assert provenance["seed_count"] == 2
    assert len(provenance["sha256"]) == 64

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["seeds"] = [7, 7]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BundleRunError, match="must be unique"):
        load_seed_manifest(path)
