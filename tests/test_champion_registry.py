from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.champion_registry import (
    BucketResult,
    ChampionRegistry,
    PanelResult,
    auto_revert_tripwire,
    bucket_veto,
    combine_panels,
    elo_posterior_normal,
    prob_elo_below,
    requires_nth_confirmation,
)


def _write_checkpoint(path: Path, content: bytes = b"checkpoint-bytes") -> Path:
    path.write_bytes(content)
    return path


# =============================================================================
# ChampionRegistry: role CRUD
# =============================================================================
def test_set_role_and_get_role_round_trip(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "gen3.pt")
    reg = ChampionRegistry(tmp_path / "registry.json")

    pointer = reg.set_role("generator_champion", ckpt, version=3, reason="initial seed")
    assert pointer.role == "generator_champion"
    assert pointer.checkpoint_path == str(ckpt)
    assert pointer.version == 3

    reg.save()
    reloaded = ChampionRegistry.load(tmp_path / "registry.json")
    fetched = reloaded.get_role("generator_champion")
    assert fetched is not None
    assert fetched.checkpoint_path == str(ckpt)
    assert fetched.md5 == pointer.md5


def test_set_role_rejects_unknown_role(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "gen3.pt")
    reg = ChampionRegistry(tmp_path / "registry.json")
    with pytest.raises(ValueError):
        reg.set_role("not_a_real_role", ckpt)


def test_set_role_missing_checkpoint_raises(tmp_path: Path) -> None:
    reg = ChampionRegistry(tmp_path / "registry.json")
    with pytest.raises(FileNotFoundError):
        reg.set_role("generator_champion", tmp_path / "does_not_exist.pt")


def test_set_role_rejects_md5_mismatch(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "gen3.pt")
    reg = ChampionRegistry(tmp_path / "registry.json")
    with pytest.raises(ValueError, match="md5 mismatch"):
        reg.set_role("generator_champion", ckpt, expected_md5="0" * 32)


def test_set_role_accepts_matching_md5(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "gen3.pt")
    reg = ChampionRegistry(tmp_path / "registry.json")
    import hashlib

    real_md5 = hashlib.md5(ckpt.read_bytes()).hexdigest()
    pointer = reg.set_role("generator_champion", ckpt, expected_md5=real_md5)
    assert pointer.md5 == real_md5


def test_set_role_reassignment_preserves_from_pointer_in_transition(tmp_path: Path) -> None:
    ckpt_a = _write_checkpoint(tmp_path / "a.pt", b"a")
    ckpt_b = _write_checkpoint(tmp_path / "b.pt", b"b")
    reg = ChampionRegistry(tmp_path / "registry.json")
    reg.set_role("public_champion", ckpt_a, reason="pin gen-3")
    reg.set_role("public_champion", ckpt_b, reason="repin")

    transitions = [t for t in reg.transitions() if t.role == "public_champion"]
    assert len(transitions) == 2
    assert transitions[0].from_pointer is None
    assert transitions[1].from_pointer is not None
    assert transitions[1].from_pointer["checkpoint_path"] == str(ckpt_a)
    assert transitions[1].to_pointer["checkpoint_path"] == str(ckpt_b)


# =============================================================================
# ChampionRegistry: opponent pool is append-only
# =============================================================================
def test_opponent_pool_append_only_and_dedup(tmp_path: Path) -> None:
    ckpt_a = _write_checkpoint(tmp_path / "a.pt", b"a")
    ckpt_b = _write_checkpoint(tmp_path / "b.pt", b"b")
    reg = ChampionRegistry(tmp_path / "registry.json")

    reg.append_pool(ckpt_a, version=0, reason="seed champion")
    assert len(reg.opponent_pool()) == 1

    # re-appending the identical (path, md5) is idempotent, not a duplicate
    reg.append_pool(ckpt_a, version=0, reason="re-run of the same promotion step")
    assert len(reg.opponent_pool()) == 1

    reg.append_pool(ckpt_b, version=1, reason="dethroned champion", status="regressed")
    assert len(reg.opponent_pool()) == 2

    # no removal API exists; the pool can only grow
    assert {e.checkpoint_path for e in reg.opponent_pool()} == {str(ckpt_a), str(ckpt_b)}
    assert not hasattr(reg, "remove_pool_entry")
    assert not hasattr(reg, "delete_from_pool")


def test_transitions_log_append_only_never_shrinks(tmp_path: Path) -> None:
    ckpt_a = _write_checkpoint(tmp_path / "a.pt", b"a")
    ckpt_b = _write_checkpoint(tmp_path / "b.pt", b"b")
    reg = ChampionRegistry(tmp_path / "registry.json")

    reg.set_role("generator_champion", ckpt_a, reason="seed")
    n1 = len(reg.transitions())
    reg.append_pool(ckpt_a, version=0)
    n2 = len(reg.transitions())
    reg.set_role("generator_champion", ckpt_b, reason="promote")
    n3 = len(reg.transitions())
    reg.record_promotion("generator_champion")
    n4 = len(reg.transitions())

    assert n1 < n2 < n3 < n4
    # earlier entries are untouched by later mutations
    first = reg.transitions()[0]
    assert first.role == "generator_champion"
    assert first.to_pointer["checkpoint_path"] == str(ckpt_a)


# =============================================================================
# Promotion counter + every-3rd confirmation flag
# =============================================================================
def test_requires_nth_confirmation_flags_every_third() -> None:
    flags = [requires_nth_confirmation(i, every=3) for i in range(1, 7)]
    assert flags == [False, False, True, False, False, True]


def test_requires_nth_confirmation_rejects_nonpositive_count() -> None:
    with pytest.raises(ValueError):
        requires_nth_confirmation(0)


def test_registry_promotion_counter_drives_confirmation_flag(tmp_path: Path) -> None:
    reg = ChampionRegistry(tmp_path / "registry.json")
    results = []
    for _ in range(6):
        count = reg.record_promotion("generator_champion")
        results.append(requires_nth_confirmation(count, every=3))
    assert results == [False, False, True, False, False, True]
    assert reg.promotion_count("generator_champion") == 6
    # a role with no recorded promotions starts at zero, not an error
    assert reg.promotion_count("public_champion") == 0


# =============================================================================
# Bucket veto
# =============================================================================
def test_bucket_veto_all_pass() -> None:
    buckets = [
        BucketResult("phase_early", wins=40, losses=20),
        BucketResult("phase_late", wins=35, losses=25),
        BucketResult("opening_a", wins=30, losses=20),
    ]
    result = bucket_veto(buckets, min_winrate=0.5, min_n=8)
    assert result.veto is False
    assert result.veto_buckets == ()
    assert all(v["status"] == "pass" for v in result.per_bucket.values())


def test_bucket_veto_single_bucket_vetoes_even_if_aggregate_passes() -> None:
    # Pooled: (60+10) wins / (100 total) = 70% overall, comfortably >= 50%.
    # But the "blowout" bucket alone is 10/30 = 33%, well under 50% -- it must
    # veto even though the aggregate looks fine.
    buckets = [
        BucketResult("phase", wins=60, losses=10),
        BucketResult("blowout", wins=10, losses=20),
    ]
    pooled_wins = sum(b.wins for b in buckets)
    pooled_losses = sum(b.losses for b in buckets)
    assert pooled_wins / (pooled_wins + pooled_losses) >= 0.5  # aggregate would pass

    result = bucket_veto(buckets, min_winrate=0.5, min_n=8)
    assert result.veto is True
    assert result.veto_buckets == ("blowout",)
    assert result.per_bucket["phase"]["status"] == "pass"
    assert result.per_bucket["blowout"]["status"] == "fail"


def test_bucket_veto_insufficient_data_does_not_veto() -> None:
    buckets = [
        BucketResult("phase", wins=40, losses=20),
        BucketResult("rare_opening", wins=1, losses=2),  # n=3, below min_n
    ]
    result = bucket_veto(buckets, min_winrate=0.5, min_n=8)
    assert result.veto is False
    assert result.per_bucket["rare_opening"]["status"] == "insufficient_data"


def test_bucket_veto_zero_game_bucket_is_insufficient_not_a_crash() -> None:
    buckets = [BucketResult("never_hit", wins=0, losses=0)]
    result = bucket_veto(buckets, min_n=8)
    assert result.veto is False
    assert result.per_bucket["never_hit"]["status"] == "insufficient_data"
    assert result.per_bucket["never_hit"]["winrate"] is None


# =============================================================================
# Elo posterior math
# =============================================================================
def test_elo_posterior_sign_matches_winrate() -> None:
    mean_elo, _ = elo_posterior_normal(wins=60, losses=40)
    assert mean_elo > 0.0
    mean_elo_neg, _ = elo_posterior_normal(wins=40, losses=60)
    assert mean_elo_neg < 0.0
    mean_elo_even, se_even = elo_posterior_normal(wins=50, losses=50)
    assert mean_elo_even == pytest.approx(0.0, abs=1e-6)


def test_elo_posterior_se_shrinks_with_more_games() -> None:
    _, se_small = elo_posterior_normal(wins=30, losses=20)
    _, se_large = elo_posterior_normal(wins=300, losses=200)
    assert se_large < se_small


def test_elo_posterior_requires_at_least_one_game() -> None:
    with pytest.raises(ValueError):
        elo_posterior_normal(wins=0, losses=0)


def test_prob_elo_below_monotone_in_threshold() -> None:
    mean_elo, se_elo = elo_posterior_normal(wins=50, losses=50)
    low = prob_elo_below(mean_elo, se_elo, -100.0)
    mid = prob_elo_below(mean_elo, se_elo, 0.0)
    high = prob_elo_below(mean_elo, se_elo, 100.0)
    assert low < mid < high
    assert mid == pytest.approx(0.5, abs=1e-6)  # symmetric panel -> P(elo<0) ~= 0.5


# =============================================================================
# Auto-revert tripwire
# =============================================================================
def test_auto_revert_tripwire_condition_a_catastrophic_single_panel() -> None:
    # A large, lopsided panel (roughly 25% win rate over 400 games) should push
    # the posterior mean well below -25 Elo with high confidence.
    current = PanelResult(wins=100, losses=300, label="panel-1")
    decision = auto_revert_tripwire(current, previous_panel=None)
    assert decision.condition_a is True
    assert decision.should_revert is True
    assert "P(dElo_ext" in decision.reason


def test_auto_revert_tripwire_mild_single_decline_does_not_trigger() -> None:
    # A modest, noisy decline (48% over 100 games) shouldn't hit either leg.
    current = PanelResult(wins=48, losses=52, label="panel-1")
    decision = auto_revert_tripwire(current, previous_panel=None)
    assert decision.condition_a is False
    assert decision.condition_b is False
    assert decision.should_revert is False


def test_auto_revert_tripwire_condition_b_two_consecutive_declines() -> None:
    # Two consecutive, individually-modest-but-real declines (47% over 300
    # games each -> median well below -10, but nowhere near the -25-Elo/0.9
    # single-panel bar) should trip condition (b) via the combined pooled
    # evidence, even though neither alone is catastrophic enough for
    # condition (a).
    previous = PanelResult(wins=141, losses=159, label="panel-1")  # 47%
    current = PanelResult(wins=141, losses=159, label="panel-2")  # 47%
    decision = auto_revert_tripwire(current, previous_panel=previous)
    assert decision.condition_a is False
    assert decision.condition_b is True
    assert decision.should_revert is True
    assert decision.previous_posterior is not None
    assert decision.combined_posterior is not None


def test_auto_revert_tripwire_flapping_does_not_trigger() -> None:
    # A bad panel followed by a clearly GOOD panel must not trip either
    # condition: condition (a) only looks at the current (good) panel, and
    # condition (b) requires BOTH panels to individually show a decline.
    previous = PanelResult(wins=100, losses=300, label="panel-1")  # 25%, bad
    current = PanelResult(wins=220, losses=180, label="panel-2")  # 55%, good
    decision = auto_revert_tripwire(current, previous_panel=previous)
    assert decision.condition_a is False
    assert decision.condition_b is False
    assert decision.should_revert is False


def test_auto_revert_tripwire_flapping_good_then_bad_does_not_trigger() -> None:
    # Symmetric flapping case: a good panel followed by one MILDLY declining
    # panel. Condition (a) needs the CURRENT panel alone to be catastrophic at
    # >0.9 confidence; a single 47%-over-100 panel isn't that extreme. Condition
    # (b) needs BOTH panels declining, but the previous panel was good.
    previous = PanelResult(wins=220, losses=180, label="panel-1")  # 55%, good
    current = PanelResult(wins=47, losses=53, label="panel-2")  # 47%, mild decline
    decision = auto_revert_tripwire(current, previous_panel=previous)
    assert decision.condition_a is False
    assert decision.condition_b is False
    assert decision.should_revert is False


def test_combine_panels_pools_raw_counts() -> None:
    a = PanelResult(wins=10, losses=20, draws=1, label="a")
    b = PanelResult(wins=5, losses=15, draws=0, label="b")
    combined = combine_panels(a, b)
    assert combined.wins == 15
    assert combined.losses == 35
    assert combined.draws == 1


# =============================================================================
# Integration: registry + tripwire
# =============================================================================
def test_registry_auto_revert_writes_pointer_back_and_logs_justification(tmp_path: Path) -> None:
    stable = _write_checkpoint(tmp_path / "stable_gen3.pt", b"stable")
    suspect = _write_checkpoint(tmp_path / "suspect_gen4.pt", b"suspect")
    reg = ChampionRegistry(tmp_path / "registry.json")

    reg.set_role("generator_champion", stable, version=3, reason="gen-3 stable")
    reg.set_role("generator_champion", suspect, version=4, reason="promote gen-4")

    catastrophic_panel = PanelResult(wins=90, losses=310, label="post-gen4-panel")
    decision = auto_revert_tripwire(catastrophic_panel, previous_panel=None)
    assert decision.should_revert is True

    reverted = reg.auto_revert(decision, revert_to_checkpoint=stable, version=3)
    assert reverted.checkpoint_path == str(stable)
    assert reg.get_role("generator_champion").checkpoint_path == str(stable)

    revert_transitions = [t for t in reg.transitions() if t.kind == "auto_revert"]
    assert len(revert_transitions) == 1
    assert "AUTO-REVERT" in revert_transitions[0].reason
    assert decision.reason in revert_transitions[0].reason
    assert revert_transitions[0].to_pointer["checkpoint_path"] == str(stable)


def test_registry_auto_revert_raises_if_decision_did_not_trip(tmp_path: Path) -> None:
    stable = _write_checkpoint(tmp_path / "stable.pt", b"stable")
    reg = ChampionRegistry(tmp_path / "registry.json")
    reg.set_role("generator_champion", stable, version=1)

    fine_panel = PanelResult(wins=52, losses=48, label="fine")
    decision = auto_revert_tripwire(fine_panel, previous_panel=None)
    assert decision.should_revert is False
    with pytest.raises(ValueError):
        reg.auto_revert(decision, revert_to_checkpoint=stable)


def test_registry_json_is_stable_across_save_load_cycles(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "c.pt")
    path = tmp_path / "registry.json"
    reg = ChampionRegistry(path)
    reg.set_role("generator_champion", ckpt, version=1, reason="seed")
    reg.append_pool(ckpt, version=1)
    reg.record_promotion("generator_champion")
    reg.save()

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "roles" in raw and "opponent_pool" in raw and "transitions" in raw and "promotion_counts" in raw

    reloaded = ChampionRegistry.load(path)
    assert reloaded.get_role("generator_champion").checkpoint_path == str(ckpt)
    assert len(reloaded.opponent_pool()) == 1
    assert reloaded.promotion_count("generator_champion") == 1
    assert len(reloaded.transitions()) == len(reg.transitions())
