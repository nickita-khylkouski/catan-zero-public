from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from catan_zero.rl.ppo_run_manifest import ManifestError, PPORunManifest, load_manifest


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "selfplay"
    / "ppo_2p_no_trade_v2.json"
)


def _payload() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_checked_manifest_is_strict_and_deterministic() -> None:
    first = load_manifest(CONFIG)
    second = PPORunManifest.from_json(first.canonical_json())

    assert second == first
    assert second.canonical_json() == first.canonical_json()
    assert second.sha256() == first.sha256()
    assert len(first.sha256()) == len("sha256:") + 64
    assert first.status == "template"
    assert first.spec.identity.track == "2p_no_trade"
    assert first.spec.identity.initializer_sha256.endswith("0" * 64)
    assert first.spec.actor.opponent_mode == "fixed"
    assert first.spec.learner.value_trunk_grad_scale == pytest.approx(0.1)
    assert first.spec.checkpoint.every_steps == 50


def test_shipped_template_actor_science_resolves_once_bound(tmp_path: Path) -> None:
    from catan_zero.rl import ppo_distributed as dist
    from tools import run_local_entity_ppo_shards as actor

    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"initializer")
    payload = _payload()
    payload["status"] = "bound"
    payload["spec"]["identity"]["initializer_sha256"] = (
        f"sha256:{dist.checkpoint_sha256(checkpoint)}"
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    args, manifest = actor.resolve_config(
        [
            "--run-manifest",
            str(manifest_path),
            "--run-name",
            "template-contract-test",
            "--checkpoint",
            str(checkpoint),
        ]
    )

    assert manifest is not None
    assert args.opponent_mode == "fixed"


def test_pre_scale_v2_manifest_preserves_hash_and_legacy_execution_contract() -> None:
    payload = _payload()
    del payload["spec"]["learner"]["value_trunk_grad_scale"]
    expected_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    manifest = PPORunManifest.from_dict(payload)

    assert manifest.spec.learner.value_trunk_grad_scale is None
    assert manifest.canonical_json() == expected_json


def test_manifest_rejects_explicit_null_value_trunk_scale() -> None:
    payload = _payload()
    payload["spec"]["learner"]["value_trunk_grad_scale"] = None

    with pytest.raises(ManifestError, match="floating-point"):
        PPORunManifest.from_dict(payload)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.__setitem__("unknown", 1), "unknown"),
        (lambda value: value["spec"]["actor"].pop("gamma"), "missing"),
        (
            lambda value: value["spec"]["learner"].__setitem__("minibatch_size", True),
            "integer",
        ),
        (
            lambda value: value["spec"]["actor"].__setitem__("gamma", 1),
            "floating-point",
        ),
        (
            lambda value: value["spec"]["identity"].__setitem__(
                "initializer_sha256", "/host/path/checkpoint.pt"
            ),
            "sha256",
        ),
    ],
)
def test_manifest_rejects_unknown_missing_and_wrong_types(mutation, match: str) -> None:
    value = _payload()
    mutation(value)

    with pytest.raises(ManifestError, match=match):
        PPORunManifest.from_dict(value)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_manifest_rejects_nonfinite_json(constant: str) -> None:
    raw = CONFIG.read_text(encoding="utf-8").replace(
        '"gamma": 1.0', f'"gamma": {constant}'
    )

    with pytest.raises(ManifestError, match="non-finite"):
        PPORunManifest.from_json(raw)


def test_manifest_rejects_nonfinite_python_float() -> None:
    value = _payload()
    value["spec"]["learner"]["lr"] = float("nan")

    with pytest.raises(ManifestError, match="finite"):
        PPORunManifest.from_dict(value)


def test_manifest_rejects_temperature_below_executed_clamp() -> None:
    value = _payload()
    value["spec"]["actor"]["action_temperature"] = 1e-7

    with pytest.raises(ManifestError, match=r">= 1e-06"):
        PPORunManifest.from_dict(value)


def test_zero_initializer_is_allowed_only_for_explicit_template() -> None:
    value = _payload()
    value["status"] = "bound"

    with pytest.raises(ManifestError, match="real initializer bytes"):
        PPORunManifest.from_dict(value)


@pytest.mark.parametrize("track", ["2p", "2p_trade", "2p_no_trdae", "4p_no_trade"])
def test_manifest_requires_exact_two_player_no_trade_track(track: str) -> None:
    value = _payload()
    value["spec"]["identity"]["track"] = track

    with pytest.raises(ManifestError, match="2p_no_trade"):
        PPORunManifest.from_dict(value)


def test_every_science_field_change_changes_full_hash() -> None:
    baseline = PPORunManifest.from_dict(_payload())
    changes = [
        ("actor", "max_decisions", 999),
        ("actor", "action_temperature", 0.75),
        ("actor", "value_shaping_coef", 0.1),
        ("learner", "lr", 0.0001),
        ("learner", "value_trunk_grad_scale", 0.25),
        ("learner", "max_staleness", 3),
        ("learner", "target_kl", 0.006),
        ("checkpoint", "keep_last", 19),
        ("evaluation", "dev_games", 1999),
        ("league", "promote_winrate", 0.69),
    ]

    for section, field, replacement in changes:
        value = _payload()
        value["spec"][section][field] = replacement
        assert PPORunManifest.from_dict(value).sha256() != baseline.sha256(), field


def test_opponent_list_order_is_preserved_and_hash_significant() -> None:
    original_payload = _payload()
    reversed_payload = copy.deepcopy(original_payload)
    reversed_payload["spec"]["actor"]["opponents"].reverse()

    original = PPORunManifest.from_dict(original_payload)
    reversed_manifest = PPORunManifest.from_dict(reversed_payload)

    assert original.spec.actor.opponents == tuple(
        original_payload["spec"]["actor"]["opponents"]
    )
    assert reversed_manifest.spec.actor.opponents == tuple(
        reversed_payload["spec"]["actor"]["opponents"]
    )
    assert original.sha256() != reversed_manifest.sha256()


def test_no_vtrace_requires_version_exact_rollouts() -> None:
    value = _payload()
    value["spec"]["learner"]["use_vtrace"] = False

    with pytest.raises(ManifestError, match="max_staleness must be 0"):
        PPORunManifest.from_dict(value)
