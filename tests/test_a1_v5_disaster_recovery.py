from __future__ import annotations

import base64
import copy
import fcntl
import json
import os
import stat
from pathlib import Path

import pytest

from tools import a1_post_promotion_handoff
from tools import a1_v5_disaster_recovery as recovery
from tools.champion_registry import ChampionRegistry
from tools.fleet import a1_h100_eval_fleet


def _sha_bytes(value: bytes) -> str:
    return recovery._sha256_bytes(value)  # noqa: SLF001


def _write(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _fixed_smoke(_path: Path) -> dict[str, object]:
    return {
        "schema_version": "a1-v5-recovery-runtime-smoke-v1",
        "load_complete": True,
        "identity": "test-runtime",
    }


def _fixed_tool() -> dict[str, object]:
    return {
        "git_commit": "a" * 40,
        "files": {"tools/a1_v5_disaster_recovery.py": "sha256:" + "b" * 64},
        "files_sha256": "sha256:" + "c" * 64,
    }


def test_runtime_config_scalar_is_normalized_to_canonical_json() -> None:
    class NumPyLikeInt:
        def item(self) -> int:
            return 567

    normalized = recovery._json_runtime_value(  # noqa: SLF001
        {"action_size": NumPyLikeInt(), "shape": (4, 8)},
        where="policy config",
    )
    assert normalized == {"action_size": 567, "shape": [4, 8]}
    assert json.loads(json.dumps(normalized)) == normalized


@pytest.fixture()
def exact_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    producer = _write(tmp_path / "producer-v5.pt", b"exact-v5-checkpoint")
    safety = _write(tmp_path / "f7.pt", b"exact-f7-checkpoint")
    producer_sha = _sha_bytes(producer.read_bytes())
    safety_sha = _sha_bytes(safety.read_bytes())
    identity = {
        "schema_version": "a1-agent-identity-v1",
        "checkpoint": {"path": str(producer), "sha256": producer_sha},
        "search_config": {"c_scale": 0.1, "n_full": 128},
    }
    identity["agent_identity_sha256"] = recovery._digest(identity)  # noqa: SLF001
    lost_root = tmp_path / "lost"
    lost_root.mkdir()
    handoff = {
        "schema_version": "a1-post-promotion-producer-handoff-v1",
        "promotion_receipt": {
            "path": str(lost_root / "promotion.json"),
            "sha256": "sha256:" + "1" * 64,
            "receipt_sha256": "sha256:" + "2" * 64,
            "transaction_id": "lost-transaction",
        },
        "registry_after": {
            "path": str(lost_root / "registry.json"),
            "sha256": "sha256:" + "3" * 64,
            "role": "generator_champion",
            "version": 5,
            "checkpoint": {"path": str(producer), "sha256": producer_sha},
        },
        "current_champion": {
            "path": str(lost_root / "CURRENT_CHAMPION"),
            "sha256": "sha256:" + "4" * 64,
            "bytes_base64": base64.b64encode((str(producer) + "\n").encode()).decode(),
        },
        "producer_identity": identity,
    }
    handoff["handoff_sha256"] = recovery._digest(handoff)  # noqa: SLF001
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(
        recovery,
        "EXPECTED_HANDOFF_FINGERPRINT",
        {
            "checkpoint_sha256": producer_sha,
            "handoff_file_sha256": _sha_bytes(handoff_path.read_bytes()),
            "handoff_sha256": handoff["handoff_sha256"],
            "producer_identity_sha256": identity["agent_identity_sha256"],
            "promotion_receipt_file_sha256": handoff["promotion_receipt"]["sha256"],
            "promotion_receipt_sha256": handoff["promotion_receipt"]["receipt_sha256"],
            "registry_version": 5,
        },
    )
    monkeypatch.setattr(recovery, "EXPECTED_F7_SHA256", safety_sha)
    return {
        "handoff": handoff_path,
        "producer": producer,
        "safety": safety,
        "namespace": tmp_path / recovery.RECOVERY_NAMESPACE_BASENAME,
        "lost_root": lost_root,
    }


def _plan(inputs: dict[str, Path]) -> dict[str, object]:
    return recovery.build_plan(
        handoff_path=inputs["handoff"],
        safety_reference_path=inputs["safety"],
        namespace=inputs["namespace"],
        runtime_smoke_fn=_fixed_smoke,
        tool_identity_fn=_fixed_tool,
    )


def test_dry_run_is_read_only_and_preserves_evidence_loss(
    exact_inputs: dict[str, Path],
) -> None:
    plan = _plan(exact_inputs)
    assert not exact_inputs["namespace"].exists()
    assert plan["mode"] == "dry-run"
    assert plan["lineage"]["promotion_proof_recreated"] is False
    assert plan["lineage"]["verified_promotion_count"] is None
    assert plan["safety_reference"]["relationship"] == recovery.RECOVERY_RELATION
    assert plan["safety_reference"]["causal_parent_proven"] is False
    assert plan["promotion_policy"]["dual_baseline_required"] is True
    for claim in plan["lost_claims_from_surviving_handoff"].values():
        assert claim["present"] is False
        assert claim["replayed"] is False


def test_commit_and_canonical_verifier_expose_only_recovery_authority(
    exact_inputs: dict[str, Path],
) -> None:
    plan = _plan(exact_inputs)
    receipt = recovery.commit(
        plan, runtime_smoke_fn=_fixed_smoke, tool_identity_fn=_fixed_tool
    )
    receipt_path = Path(receipt["destinations"]["receipt"])
    verified = recovery.verify_committed_receipt(
        receipt_path, runtime_smoke_fn=_fixed_smoke, tool_identity_fn=_fixed_tool
    )
    authority = verified["authority"]
    assert authority["recovered_generator"]["sha256"] == plan["recovered_checkpoint"]["sha256"]
    assert authority[recovery.RECOVERY_RELATION]["sha256"] == plan["safety_reference"]["sha256"]
    assert authority["promotion_eligible"] is False
    assert authority["training_proof"] is False
    registry = json.loads(Path(receipt["registry"]["path"]).read_text())
    assert registry["roles"]["generator_champion"]["checkpoint_path"] == str(
        exact_inputs["producer"]
    )
    for role in recovery.SAFETY_ROLES:
        assert registry["roles"][role]["checkpoint_path"] == str(exact_inputs["safety"])
    assert registry["promotion_counts"] == {}
    assert Path(receipt["current_pointer"]["path"]).read_text() == str(
        exact_inputs["producer"]
    ) + "\n"
    with pytest.raises(a1_post_promotion_handoff.HandoffError):
        a1_post_promotion_handoff.build_handoff(receipt_path)


def test_recovered_registry_loads_through_real_registry_and_eval_reader(
    exact_inputs: dict[str, Path],
) -> None:
    receipt = recovery.commit(
        _plan(exact_inputs),
        runtime_smoke_fn=_fixed_smoke,
        tool_identity_fn=_fixed_tool,
    )
    registry_path = Path(receipt["registry"]["path"])
    registry = ChampionRegistry.load(registry_path)
    generator = registry.get_role(recovery.RECOVERED_GENERATOR_ROLE)
    assert generator is not None
    assert generator.checkpoint_path == str(exact_inputs["producer"])
    assert generator.md5 == recovery._md5_bytes(  # noqa: SLF001
        exact_inputs["producer"].read_bytes()
    )
    assert generator.version == recovery.EXPECTED_V5_VERSION
    for role in recovery.SAFETY_ROLES:
        pointer = registry.get_role(role)
        assert pointer is not None
        assert pointer.checkpoint_path == str(exact_inputs["safety"])
        assert pointer.version == recovery.EXPECTED_F7_VERSION

    # Exercise the same deployed-identity reader used by the distributed H100
    # evaluation launcher.  A recovery registry that merely looked plausible
    # as JSON but did not satisfy this binding would strand the first gate.
    binding = a1_h100_eval_fleet._evaluation_binding(  # noqa: SLF001
        candidate_parent=exact_inputs["producer"],
        baseline=exact_inputs["producer"],
        registry=registry,
        comparison_mode="promotion_parent",
        historical_comparison_reason=None,
        champion_c_scale=0.1,
    )
    assert binding["authoritative_incumbent"]["path"] == str(
        exact_inputs["producer"]
    )
    assert binding["authoritative_incumbent"]["search_config"] == {
        "c_scale": 0.1,
        "n_full": 128,
    }
    safety_binding = a1_h100_eval_fleet._evaluation_binding(  # noqa: SLF001
        candidate_parent=exact_inputs["producer"],
        baseline=exact_inputs["safety"],
        registry=registry,
        comparison_mode="recovery_safety_reference",
        historical_comparison_reason=(
            "disaster_recovery_f7_non_regression_veto"
        ),
        champion_c_scale=0.1,
    )
    assert safety_binding["schema_version"] == "a1-evaluation-baseline-binding-v3"
    assert safety_binding["candidate_parent"]["path"] == str(
        exact_inputs["producer"]
    )
    assert safety_binding["baseline"]["path"] == str(exact_inputs["safety"])

    # A recovery receipt is deliberately not shape-compatible with the normal
    # post-promotion bridge even though its registry is runtime-compatible.
    with pytest.raises(a1_post_promotion_handoff.HandoffError):
        a1_post_promotion_handoff.build_handoff(
            Path(receipt["destinations"]["receipt"])
        )


def test_exact_prepared_journal_is_the_only_idempotent_resume(
    exact_inputs: dict[str, Path], tmp_path: Path
) -> None:
    plan = _plan(exact_inputs)
    first = recovery.commit(
        plan, runtime_smoke_fn=_fixed_smoke, tool_identity_fn=_fixed_tool
    )
    resumed = recovery.resume_plan(
        namespace=exact_inputs["namespace"],
        handoff_path=exact_inputs["handoff"],
        safety_reference_path=exact_inputs["safety"],
    )
    assert resumed == plan
    second = recovery.commit(
        resumed, runtime_smoke_fn=_fixed_smoke, tool_identity_fn=_fixed_tool
    )
    assert second == first
    with pytest.raises(recovery.RecoveryError, match="unexpectedly exists"):
        _plan(exact_inputs)
    copied_safety = _write(tmp_path / "copied-f7.pt", exact_inputs["safety"].read_bytes())
    with pytest.raises(recovery.RecoveryError, match="resume inputs differ"):
        recovery.resume_plan(
            namespace=exact_inputs["namespace"],
            handoff_path=exact_inputs["handoff"],
            safety_reference_path=copied_safety,
        )


def test_commit_holds_one_canonical_flock_for_same_and_different_plans(
    exact_inputs: dict[str, Path],
) -> None:
    plan = _plan(exact_inputs)
    competing = copy.deepcopy(plan)
    competing["runtime_smoke"] = {
        "schema_version": "a1-v5-recovery-runtime-smoke-v1",
        "load_complete": True,
        "identity": "different-plan",
    }
    unsigned = dict(competing)
    unsigned.pop("plan_sha256")
    competing["plan_sha256"] = recovery._digest(unsigned)  # noqa: SLF001

    namespace = exact_inputs["namespace"]
    with recovery._publication_lock(namespace) as lock_path:  # noqa: SLF001
        metadata = lock_path.stat(follow_symlinks=False)
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        with pytest.raises(recovery.RecoveryError, match="already held"):
            recovery.commit(
                plan,
                runtime_smoke_fn=_fixed_smoke,
                tool_identity_fn=_fixed_tool,
            )
        with pytest.raises(recovery.RecoveryError, match="already held"):
            recovery.commit(
                competing,
                runtime_smoke_fn=_fixed_smoke,
                tool_identity_fn=_fixed_tool,
            )

    receipt = recovery.commit(
        plan, runtime_smoke_fn=_fixed_smoke, tool_identity_fn=_fixed_tool
    )
    assert Path(receipt["destinations"]["receipt"]).is_file()


def test_foreign_lock_holder_blocks_commit(exact_inputs: dict[str, Path]) -> None:
    plan = _plan(exact_inputs)
    lock_path = exact_inputs["namespace"].parent / recovery.RECOVERY_LOCK_BASENAME
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(recovery.RecoveryError, match="already held"):
            recovery.commit(
                plan,
                runtime_smoke_fn=_fixed_smoke,
                tool_identity_fn=_fixed_tool,
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_commit_refuses_checkpoint_toctou(exact_inputs: dict[str, Path]) -> None:
    plan = _plan(exact_inputs)
    exact_inputs["producer"].write_bytes(b"mutated-after-plan")
    with pytest.raises(recovery.RecoveryError, match="different checkpoint bytes"):
        recovery.commit(
            plan, runtime_smoke_fn=_fixed_smoke, tool_identity_fn=_fixed_tool
        )


def test_recovery_refuses_when_any_claimed_lost_evidence_exists(
    exact_inputs: dict[str, Path],
) -> None:
    _write(exact_inputs["lost_root"] / "promotion.json", b"restored-or-hostile")
    with pytest.raises(recovery.RecoveryError, match="unexpectedly exists"):
        _plan(exact_inputs)


def test_handoff_mutation_is_not_a_shape_based_bypass(
    exact_inputs: dict[str, Path],
) -> None:
    value = json.loads(exact_inputs["handoff"].read_text())
    value["producer_identity"]["search_config"]["c_scale"] = 0.03
    unsigned_identity = dict(value["producer_identity"])
    unsigned_identity.pop("agent_identity_sha256")
    value["producer_identity"]["agent_identity_sha256"] = recovery._digest(  # noqa: SLF001
        unsigned_identity
    )
    unsigned = dict(value)
    unsigned.pop("handoff_sha256")
    value["handoff_sha256"] = recovery._digest(unsigned)  # noqa: SLF001
    exact_inputs["handoff"].write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(recovery.RecoveryError, match="sole allowlisted"):
        _plan(exact_inputs)


def test_verifier_refuses_registry_and_receipt_mutations(
    exact_inputs: dict[str, Path],
) -> None:
    receipt = recovery.commit(
        _plan(exact_inputs), runtime_smoke_fn=_fixed_smoke, tool_identity_fn=_fixed_tool
    )
    receipt_path = Path(receipt["destinations"]["receipt"])
    registry_path = Path(receipt["registry"]["path"])
    registry_path.chmod(0o600)
    registry_path.write_text("{}")
    with pytest.raises(recovery.RecoveryError, match="registry/current-pointer replay drift"):
        recovery.verify_committed_receipt(
            receipt_path, runtime_smoke_fn=_fixed_smoke, tool_identity_fn=_fixed_tool
        )
