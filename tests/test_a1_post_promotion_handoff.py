from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import a1_post_promotion_handoff as handoff
from tools import a1_pre_wave_contract as contract
from tools import a1_promotion_transaction as promotion
from tools.champion_registry import ChampionRegistry


def _identity(checkpoint: Path) -> dict:
    return promotion._agent_identity(  # noqa: SLF001
        {"path": str(checkpoint), "sha256": handoff._sha256(checkpoint)},  # noqa: SLF001
        {"c_scale": 0.1, "n_full": 128},
    )


def _state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate-v4")
    identity = _identity(checkpoint)
    receipt_path = tmp_path / "promotion.receipt.json"
    registry_path = tmp_path / "champions.json"
    registry = ChampionRegistry(registry_path)
    registry.set_role(
        "generator_champion",
        checkpoint,
        version=4,
        provenance={
            "a1_promotion_receipt": str(receipt_path),
            "a1_candidate_agent_identity_sha256": identity[
                "agent_identity_sha256"
            ],
            "a1_candidate_search_config": identity["search_config"],
        },
    )
    registry.save()
    pointer = tmp_path / "CURRENT_CHAMPION"
    pointer.write_bytes((str(checkpoint) + "\n").encode())
    receipt = {
        "status": "committed",
        "transaction_id": "tx-4",
        "receipt_sha256": "sha256:" + "a" * 64,
        "registry": {
            "path": str(registry_path),
            "after_sha256": handoff._sha256(registry_path),  # noqa: SLF001
        },
        "current_pointer": {
            "path": str(pointer),
            "after_sha256": handoff._sha256(pointer),  # noqa: SLF001
        },
        "candidate": {
            "path": str(checkpoint),
            "sha256": handoff._sha256(checkpoint),  # noqa: SLF001
            "version": 4,
            "agent_identity": identity,
        },
    }
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    def load(_: Path):
        return receipt, receipt_path, registry_path, pointer, tmp_path / "rb", tmp_path / "pb"

    monkeypatch.setattr(promotion, "_load_recovery_receipt", load)
    return {
        "checkpoint": checkpoint,
        "registry": registry_path,
        "pointer": pointer,
        "receipt_path": receipt_path,
        "receipt": receipt,
    }


def _write_handoff(state: dict, tmp_path: Path) -> tuple[Path, dict]:
    payload = handoff.build_handoff(state["receipt_path"])
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path, payload


def _science_args(monkeypatch: pytest.MonkeyPatch, *, c_scale: float = 0.1) -> dict:
    monkeypatch.setattr(
        promotion,
        "_sealed_evaluation_semantics",
        lambda _: {"c_scale": c_scale, "n_full": 128},
    )
    return {
        "effective_search": {"c_scale": c_scale},
        "evaluator": {},
        "generation": {},
    }


def test_handoff_binds_committed_registry_pointer_and_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path, monkeypatch)
    path, payload = _write_handoff(state, tmp_path)
    producer = {
        "path": str(state["checkpoint"]),
        "sha256": handoff._sha256(state["checkpoint"]),  # noqa: SLF001
    }
    record = contract._promotion_handoff_record(  # noqa: SLF001
        {"mode": "post_promotion", "path": str(path)},
        base=tmp_path,
        producer=producer,
        **_science_args(monkeypatch),
    )
    assert record["registry_role"] == "generator_champion"
    assert record["registry_version"] == 4
    assert record["producer_identity_sha256"] == payload["producer_identity"][
        "agent_identity_sha256"
    ]


def test_refuses_uncommitted_dry_run_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path, monkeypatch)
    state["receipt"]["status"] = "dry_run"
    with pytest.raises(handoff.HandoffError, match="not committed"):
        handoff.build_handoff(state["receipt_path"])


def test_refuses_swapped_checkpoint_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path, monkeypatch)
    state["checkpoint"].write_bytes(b"swapped-after-promotion")
    with pytest.raises(handoff.HandoffError, match="checkpoint bytes drifted"):
        handoff.build_handoff(state["receipt_path"])


@pytest.mark.parametrize("stale", ["registry", "pointer"])
def test_refuses_stale_registry_or_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stale: str
) -> None:
    state = _state(tmp_path, monkeypatch)
    Path(state[stale]).write_bytes(b"stale\n")
    with pytest.raises(handoff.HandoffError, match=f"live {'CURRENT_CHAMPION' if stale == 'pointer' else 'registry'}"):
        handoff.build_handoff(state["receipt_path"])


@pytest.mark.parametrize("mutation", ["role", "version"])
def test_refuses_wrong_registry_role_or_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    state = _state(tmp_path, monkeypatch)
    raw = json.loads(state["registry"].read_text())
    row = raw["roles"].pop("generator_champion")
    if mutation == "role":
        row["role"] = "public_champion"
        raw["roles"]["public_champion"] = row
    else:
        row["version"] = 3
        raw["roles"]["generator_champion"] = row
    state["registry"].write_text(json.dumps(raw, sort_keys=True))
    state["receipt"]["registry"]["after_sha256"] = handoff._sha256(  # noqa: SLF001
        state["registry"]
    )
    state["receipt_path"].write_text(
        json.dumps(state["receipt"]) + "\n", encoding="utf-8"
    )
    message = "no generator_champion" if mutation == "role" else "role/version"
    with pytest.raises(handoff.HandoffError, match=message):
        handoff.build_handoff(state["receipt_path"])


def test_refuses_registry_provenance_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path, monkeypatch)
    raw = json.loads(state["registry"].read_text())
    raw["roles"]["generator_champion"]["provenance"][
        "a1_candidate_search_config"
    ] = {"c_scale": 0.03, "n_full": 128}
    state["registry"].write_text(json.dumps(raw, sort_keys=True))
    state["receipt"]["registry"]["after_sha256"] = handoff._sha256(  # noqa: SLF001
        state["registry"]
    )
    state["receipt_path"].write_text(
        json.dumps(state["receipt"]) + "\n", encoding="utf-8"
    )
    with pytest.raises(handoff.HandoffError, match="registry generator provenance"):
        handoff.build_handoff(state["receipt_path"])


def test_refuses_atomic_replacement_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path, monkeypatch)
    original = handoff._revalidate_snapshot  # noqa: SLF001
    calls = 0

    def race(snapshot: dict[Path, bytes]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            replacement = tmp_path / "registry.replacement"
            replacement.write_bytes(state["registry"].read_bytes() + b" ")
            replacement.replace(state["registry"])
        original(snapshot)

    monkeypatch.setattr(handoff, "_revalidate_snapshot", race)
    with pytest.raises(handoff.HandoffError, match="replaced before output"):
        handoff.write_handoff(state["receipt_path"], tmp_path / "handoff.json")
    assert not (tmp_path / "handoff.json").exists()


def test_refuses_checkpoint_replacement_after_output_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path, monkeypatch)
    original = handoff._revalidate_snapshot  # noqa: SLF001
    calls = 0

    def race(snapshot: dict[Path, bytes]) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            state["checkpoint"].write_bytes(b"changed-in-final-write-window")
        original(snapshot)

    monkeypatch.setattr(handoff, "_revalidate_snapshot", race)
    out = tmp_path / "handoff.json"
    with pytest.raises(handoff.HandoffError, match="replaced before output"):
        handoff.write_handoff(state["receipt_path"], out)
    assert not out.exists()


def test_refuses_receipt_replacement_between_replay_and_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path, monkeypatch)
    original = promotion._load_recovery_receipt  # noqa: SLF001

    def race(path: Path):
        replayed = original(path)
        changed = dict(state["receipt"])
        changed["transaction_id"] = "different-transaction"
        path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
        return replayed

    monkeypatch.setattr(promotion, "_load_recovery_receipt", race)
    with pytest.raises(handoff.HandoffError, match="semantic replay and byte snapshot"):
        handoff.build_handoff(state["receipt_path"])


def test_refuses_draft_producer_not_equal_to_promoted_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path, monkeypatch)
    path, _ = _write_handoff(state, tmp_path)
    other = tmp_path / "other.pt"
    other.write_bytes(b"other")
    with pytest.raises(contract.ContractError, match="draft producer"):
        contract._promotion_handoff_record(  # noqa: SLF001
            {"mode": "post_promotion", "path": str(path)},
            base=tmp_path,
            producer={"path": str(other), "sha256": handoff._sha256(other)},  # noqa: SLF001
            **_science_args(monkeypatch),
        )


def test_refuses_promoted_point10_identity_with_point03_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path, monkeypatch)
    path, _ = _write_handoff(state, tmp_path)
    producer = {
        "path": str(state["checkpoint"]),
        "sha256": handoff._sha256(state["checkpoint"]),  # noqa: SLF001
    }
    with pytest.raises(contract.ContractError, match="generation c_scale"):
        contract._promotion_handoff_record(  # noqa: SLF001
            {"mode": "post_promotion", "path": str(path)},
            base=tmp_path,
            producer=producer,
            **_science_args(monkeypatch, c_scale=0.03),
        )


def test_missing_handoff_fails_closed_but_historical_is_explicit(
    tmp_path: Path,
) -> None:
    producer = {"path": str(tmp_path / "old.pt"), "sha256": "sha256:" + "1" * 64}
    with pytest.raises(contract.ContractError, match="explicitly select"):
        contract._promotion_handoff_record({}, base=tmp_path, producer=producer)  # noqa: SLF001
    assert contract._promotion_handoff_record(  # noqa: SLF001
        {"mode": "historical_pre_promotion", "reason": "predates transaction"},
        base=tmp_path,
        producer=producer,
    )["mode"] == "historical_pre_promotion"


def test_schema_boundary_allows_history_only_on_legacy_v2(tmp_path: Path) -> None:
    template = (
        Path(__file__).resolve().parents[1]
        / "configs/experiments/a1_pre_wave_contract.template.json"
    )
    payload = json.loads(template.read_text())
    payload["schema_version"] = contract.DRAFT_SCHEMA
    draft = tmp_path / "new-wave.json"
    draft.write_text(json.dumps(payload))
    with pytest.raises(contract.ContractError, match="v3 waves require"):
        contract.build_lock(draft)

    payload["schema_version"] = contract.LEGACY_DRAFT_SCHEMA
    payload["promotion_handoff"] = {
        "mode": "post_promotion",
        "path": str(tmp_path / "handoff.json"),
    }
    draft.write_text(json.dumps(payload))
    with pytest.raises(contract.ContractError, match="legacy v2 drafts"):
        contract.build_lock(draft)
