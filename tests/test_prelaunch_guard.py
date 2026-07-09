"""Unit tests for tools/prelaunch_guard.py (CAT-69): one synthetic
pass/fail pair per guard, reproducing the original incident each guard
encodes. See the module docstring in tools/prelaunch_guard.py for the full
incident-to-guard map.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import prelaunch_guard as guard  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# (a) CLI-default-override trap
# ---------------------------------------------------------------------------


def test_cli_flag_lint_fails_when_critical_flag_is_omitted():
    result = guard.guard_cli_flag_lint(
        argv=["--games", "100", "--base-seed", "1000"],
        critical_flags=["--c-scale", "--games"],
    )
    assert not result.passed
    assert "--c-scale" in result.reason
    assert "--games" not in result.details["missing_flags"]


def test_cli_flag_lint_passes_when_all_critical_flags_explicit():
    result = guard.guard_cli_flag_lint(
        argv=["--c-scale", "1.0", "--games", "100"],
        critical_flags=["--c-scale", "--games"],
    )
    assert result.passed


def test_cli_flag_lint_accepts_equals_form():
    result = guard.guard_cli_flag_lint(
        argv=["--c-scale=1.0"],
        critical_flags=["--c-scale"],
    )
    assert result.passed


def test_discover_configurable_flags_excludes_required_and_help():
    parser = argparse.ArgumentParser()
    parser.add_argument("--required-flag", required=True)
    parser.add_argument("--c-scale", type=float, default=1.0)
    parser.add_argument("--games", type=int, default=100)
    flags = guard.discover_configurable_flags(parser)
    assert "--c-scale" in flags
    assert "--games" in flags
    assert "--required-flag" not in flags
    assert "--help" not in flags


def test_cli_flag_lint_rejects_stale_critical_flag_against_real_parser():
    # The config's critical_flags list still names a flag the target script
    # renamed away from -- a hardcoded flag list drifting out of sync with
    # the script it's supposed to lint, which is exactly the failure mode
    # this guard exists to prevent in the first place.
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-scale-v2", type=float, default=1.0)
    parser.add_argument("--games", type=int, default=100)

    result = guard.guard_cli_flag_lint(
        argv=["--c-scale-v2", "1.0", "--games", "100"],
        critical_flags=["--c-scale", "--games"],  # "--c-scale" no longer exists
        parser=parser,
    )
    assert not result.passed
    assert "--c-scale" in result.details["stale_flags"]


def test_cli_flag_lint_passes_real_parser_cross_check_then_checks_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-scale", type=float, default=1.0)
    parser.add_argument("--games", type=int, default=100)

    result = guard.guard_cli_flag_lint(
        argv=["--c-scale", "1.0", "--games", "100"],
        critical_flags=["--c-scale", "--games"],
        parser=parser,
    )
    assert result.passed


def test_cli_flag_lint_without_parser_trusts_critical_flags_as_given():
    result = guard.guard_cli_flag_lint(
        argv=["--c-scale", "1.0"],
        critical_flags=["--c-scale"],
    )
    assert result.passed


def test_cli_flag_lint_rejects_wrong_value_when_expected_values_given():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-scale", type=float, default=1.0)
    parser.add_argument("--games", type=int, default=100)

    result = guard.guard_cli_flag_lint(
        argv=["--c-scale", "1.0", "--games", "100"],
        critical_flags=["--c-scale", "--games"],
        parser=parser,
        expected_values={"--c-scale": 0.03},
    )
    assert not result.passed
    assert "unsafe value" in result.reason
    assert "0.03" in result.reason


def test_cli_flag_lint_accepts_expected_value_and_rejects_inverted_boolean():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-observation", action=argparse.BooleanOptionalAction, default=False)

    pass_result = guard.guard_cli_flag_lint(
        argv=["--public-observation"],
        critical_flags=["--public-observation"],
        parser=parser,
        expected_values={"--public-observation": True},
    )
    assert pass_result.passed

    fail_result = guard.guard_cli_flag_lint(
        argv=["--no-public-observation"],
        critical_flags=["--public-observation"],
        parser=parser,
        expected_values={"--public-observation": True},
    )
    assert not fail_result.passed
    assert "unsafe value" in fail_result.reason


# ---------------------------------------------------------------------------
# (b) seed-ledger enforcement + VAL-ONLY range
# ---------------------------------------------------------------------------


def test_val_only_never_trains_blocks_overlapping_train_range():
    result = guard.guard_val_only_never_trains((6_195_000_000, 6_195_000_100), purpose="train")
    assert not result.passed
    assert "VAL-ONLY" in result.reason


def test_val_only_never_trains_allows_generation_in_val_only_band():
    result = guard.guard_val_only_never_trains((6_195_000_000, 6_195_000_100), purpose="generate")
    assert result.passed


def test_val_only_never_trains_allows_disjoint_train_range():
    result = guard.guard_val_only_never_trains((1_000, 1_100), purpose="train")
    assert result.passed


def test_val_only_never_trains_allows_train_range_touching_val_only_start_boundary():
    # [., 6_190_000_000) ends exactly where VAL-ONLY begins -- half-open,
    # so this must NOT be treated as an overlap (the same inclusive/
    # half-open boundary trap CAT-30's review caught elsewhere).
    result = guard.guard_val_only_never_trains((6_090_000_000, 6_190_000_000), purpose="train")
    assert result.passed


def test_val_only_never_trains_allows_train_range_touching_val_only_end_boundary():
    # [6_200_000_000, .) starts exactly where VAL-ONLY ends -- half-open,
    # must NOT be treated as an overlap.
    result = guard.guard_val_only_never_trains((6_200_000_000, 6_200_000_100), purpose="train")
    assert result.passed


def test_val_only_never_trains_blocks_train_range_overlapping_by_one_at_start():
    result = guard.guard_val_only_never_trains((6_189_999_999, 6_190_000_001), purpose="train")
    assert not result.passed


def test_val_only_never_trains_blocks_train_range_overlapping_by_one_at_end():
    result = guard.guard_val_only_never_trains((6_199_999_999, 6_200_000_001), purpose="train")
    assert not result.passed


def test_seed_ledger_rejects_overlap_from_different_out_dir(tmp_path):
    out_dir_a = tmp_path / "run_a"
    out_dir_b = tmp_path / "run_b"
    out_dir_a.mkdir()
    out_dir_b.mkdir()
    claims_dir = tmp_path / ".seed_claims"
    claims_dir.mkdir()
    (claims_dir / "run_a.json").write_text(
        json.dumps({"out_dir": str(out_dir_a.resolve()), "base_seed": 1000, "games": 64})
    )

    result = guard.guard_seed_ledger(out_dir_b, base_seed=1032, games=64, claims_dir=claims_dir)
    assert not result.passed
    assert "collides" in result.reason


def test_seed_ledger_allows_disjoint_range(tmp_path):
    out_dir_a = tmp_path / "run_a"
    out_dir_b = tmp_path / "run_b"
    out_dir_a.mkdir()
    out_dir_b.mkdir()
    claims_dir = tmp_path / ".seed_claims"
    claims_dir.mkdir()
    (claims_dir / "run_a.json").write_text(
        json.dumps({"out_dir": str(out_dir_a.resolve()), "base_seed": 1000, "games": 64})
    )

    result = guard.guard_seed_ledger(out_dir_b, base_seed=1064, games=64, claims_dir=claims_dir)
    assert result.passed


def test_seed_ledger_does_not_mutate_claims_dir(tmp_path):
    out_dir = tmp_path / "run_a"
    out_dir.mkdir()
    claims_dir = tmp_path / ".seed_claims"

    result = guard.guard_seed_ledger(out_dir, base_seed=1000, games=64, claims_dir=claims_dir)
    assert result.passed
    # This is a CHECK, not a claim: it must never write a claim file itself.
    assert not (claims_dir / "run_a.json").exists()


def test_seed_ledger_blocks_val_only_range_for_training(tmp_path):
    out_dir = tmp_path / "train_run"
    out_dir.mkdir()
    result = guard.guard_seed_ledger(
        out_dir, base_seed=6_195_000_000, games=1000, claims_dir=tmp_path / ".seed_claims", purpose="train"
    )
    assert not result.passed
    assert "VAL-ONLY" in result.reason


# ---------------------------------------------------------------------------
# (c) masked-checkpoint regime guard
# ---------------------------------------------------------------------------


def _write_fake_checkpoint(path: Path, *, mask_hidden_info: bool) -> None:
    import torch

    torch.save({"mask_hidden_info": mask_hidden_info, "model": {}}, path)


def test_masked_regime_fails_on_mismatch(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    _write_fake_checkpoint(checkpoint, mask_hidden_info=False)

    result = guard.guard_masked_regime(checkpoint, expected_masked=True)
    assert not result.passed
    assert "mask_hidden_info=False" in result.reason


def test_masked_regime_passes_on_match(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    _write_fake_checkpoint(checkpoint, mask_hidden_info=True)

    result = guard.guard_masked_regime(checkpoint, expected_masked=True)
    assert result.passed


def test_masked_regime_defaults_legacy_checkpoint_to_unmasked(tmp_path):
    import torch

    checkpoint = tmp_path / "legacy_ckpt.pt"
    torch.save({"model": {}}, checkpoint)  # predates the mask_hidden_info field entirely

    assert guard.read_checkpoint_mask_hidden_info(checkpoint) is False
    result = guard.guard_masked_regime(checkpoint, expected_masked=True)
    assert not result.passed


def test_masked_regime_fails_closed_on_unreadable_checkpoint(tmp_path):
    checkpoint = tmp_path / "missing.pt"
    result = guard.guard_masked_regime(checkpoint, expected_masked=True)
    assert not result.passed


# ---------------------------------------------------------------------------
# (d) doc-vs-artifact provenance drift detector
# ---------------------------------------------------------------------------


def test_provenance_fails_on_mismatched_claim(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"mask_hidden_info": False, "seed": 42}))

    result = guard.guard_provenance(report_path, claims={"mask_hidden_info": True})
    assert not result.passed
    assert "mask_hidden_info" in result.reason


def test_provenance_fails_on_claim_field_absent_from_artifact(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"seed": 42}))

    result = guard.guard_provenance(report_path, claims={"mask_hidden_info": True})
    assert not result.passed
    assert "no such field" in result.reason


def test_provenance_passes_when_claims_match(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"mask_hidden_info": True, "seed": 42}))

    result = guard.guard_provenance(report_path, claims={"mask_hidden_info": True, "seed": 42})
    assert result.passed


def test_provenance_fails_when_report_missing(tmp_path):
    result = guard.guard_provenance(tmp_path / "does_not_exist.json", claims={"seed": 1})
    assert not result.passed


# ---------------------------------------------------------------------------
# (e) fd limit
# ---------------------------------------------------------------------------


def test_fd_limit_fails_when_soft_limit_too_low(monkeypatch):
    monkeypatch.setattr(guard, "guard_fd_limit", guard.guard_fd_limit)  # keep reference for clarity
    import resource

    monkeypatch.setattr(resource, "getrlimit", lambda _res: (1024, 4096))
    result = guard.guard_fd_limit(minimum=65536)
    assert not result.passed
    assert result.host_only


def test_fd_limit_passes_when_soft_limit_sufficient(monkeypatch):
    import resource

    monkeypatch.setattr(resource, "getrlimit", lambda _res: (65536, 65536))
    result = guard.guard_fd_limit(minimum=65536)
    assert result.passed
    assert result.host_only


# ---------------------------------------------------------------------------
# (f) orphaned multiprocessing children -- explicit-PID-only kill
# ---------------------------------------------------------------------------


def test_explicit_pid_kill_rejects_pkill():
    result = guard.guard_explicit_pid_kill(["pkill", "-f", "generate_gumbel"])
    assert not result.passed


def test_explicit_pid_kill_rejects_command_substitution():
    result = guard.guard_explicit_pid_kill(["kill", "-9", "$(pgrep -f generate_gumbel)"])
    assert not result.passed


def test_explicit_pid_kill_rejects_non_numeric_target():
    result = guard.guard_explicit_pid_kill(["kill", "worker-group"])
    assert not result.passed


def test_explicit_pid_kill_accepts_explicit_numeric_pid():
    result = guard.guard_explicit_pid_kill(["kill", "-9", "42317"])
    assert result.passed


# ---------------------------------------------------------------------------
# (g) pgrep self-match trap + fcntl lock hygiene
# ---------------------------------------------------------------------------


def test_pgrep_self_match_detects_unguarded_pattern():
    result = guard.guard_pgrep_self_match("tools/generate_gumbel_selfplay_data.py")
    assert not result.passed


def test_pgrep_self_match_accepts_bracket_trick():
    result = guard.guard_pgrep_self_match("[t]ools/generate_gumbel_selfplay_data.py")
    assert result.passed


def test_fcntl_lock_present_detects_missing_lock():
    source = "import subprocess\nsubprocess.run(['pgrep', '-af', pattern])\n"
    result = guard.guard_fcntl_lock_present(source, source_name="fake_orchestrator.py")
    assert not result.passed


def test_fcntl_lock_present_detects_real_lock():
    source = "import fcntl\nfcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
    result = guard.guard_fcntl_lock_present(source, source_name="real_orchestrator.py")
    assert result.passed


def test_lock_available_passes_when_free(tmp_path):
    lock_path = tmp_path / "loop.lock"
    result = guard.guard_lock_available(lock_path)
    assert result.passed
    assert result.host_only


def test_lock_available_fails_when_already_held(tmp_path):
    import fcntl

    lock_path = tmp_path / "loop.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    holder = open(lock_path, "a+")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        result = guard.guard_lock_available(lock_path)
        assert not result.passed
        assert result.host_only
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()


# ---------------------------------------------------------------------------
# (h) dup-seed + wave-root quarantine pre-corpus checks
# ---------------------------------------------------------------------------


def test_corpus_no_duplicate_seeds_fails_when_flagged(tmp_path):
    meta_path = tmp_path / "corpus_meta.json"
    meta_path.write_text(
        json.dumps({"stats": {"has_duplicate_game_seeds": True, "duplicate_game_seed_count": 3}})
    )
    result = guard.guard_corpus_no_duplicate_seeds(meta_path)
    assert not result.passed
    assert "3" in result.reason


def test_corpus_no_duplicate_seeds_passes_when_clean(tmp_path):
    meta_path = tmp_path / "corpus_meta.json"
    meta_path.write_text(
        json.dumps({"stats": {"has_duplicate_game_seeds": False, "duplicate_game_seed_count": 0}})
    )
    result = guard.guard_corpus_no_duplicate_seeds(meta_path)
    assert result.passed


def test_corpus_no_duplicate_seeds_fails_when_meta_missing(tmp_path):
    result = guard.guard_corpus_no_duplicate_seeds(tmp_path / "missing_meta.json")
    assert not result.passed


def test_wave_root_quarantine_fails_on_unexpected_extra_source(tmp_path):
    declared = tmp_path / "wave_1"
    extra = tmp_path / "wave_2_accidental"
    declared.mkdir()
    extra.mkdir()

    result = guard.guard_wave_root_quarantine(
        actual_sources=[declared, extra], declared_sources=[declared]
    )
    assert not result.passed
    assert result.details["extra"]


def test_wave_root_quarantine_passes_when_sources_match(tmp_path):
    wave_1 = tmp_path / "wave_1"
    wave_1.mkdir()

    result = guard.guard_wave_root_quarantine(actual_sources=[wave_1], declared_sources=[wave_1])
    assert result.passed


# ---------------------------------------------------------------------------
# Registry + check-runner CLI
# ---------------------------------------------------------------------------


def test_run_guards_reports_unknown_guard_as_failure():
    results = guard.run_guards([{"name": "not_a_real_guard", "args": {}}])
    assert len(results) == 1
    assert not results[0].passed


def test_run_guards_catches_guard_exceptions_without_aborting_the_batch():
    specs = [
        {"name": "explicit_pid_kill", "args": {"command": None}},  # raises inside the guard
        {"name": "explicit_pid_kill", "args": {"command": ["kill", "42317"]}},
    ]
    results = guard.run_guards(specs)
    assert len(results) == 2
    assert not results[0].passed  # the broken spec is reported as FAIL, not an unhandled crash
    assert results[1].passed  # and the rest of the batch still runs


def test_cmd_run_exits_nonzero_when_any_guard_fails(tmp_path, capsys):
    config_path = tmp_path / "guards.json"
    config_path.write_text(
        json.dumps(
            {
                "guards": [
                    {"name": "explicit_pid_kill", "args": {"command": ["pkill", "-f", "x"]}},
                    {"name": "explicit_pid_kill", "args": {"command": ["kill", "42317"]}},
                ]
            }
        )
    )
    exit_code = guard.main(["run", "--config", str(config_path)])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "[FAIL]" in out
    assert "[PASS]" in out


def test_cmd_run_exits_zero_when_all_guards_pass(tmp_path):
    config_path = tmp_path / "guards.json"
    config_path.write_text(
        json.dumps({"guards": [{"name": "explicit_pid_kill", "args": {"command": ["kill", "42317"]}}]})
    )
    exit_code = guard.main(["run", "--config", str(config_path)])
    assert exit_code == 0


def test_cmd_list_prints_every_registered_guard(capsys):
    exit_code = guard.main(["list"])
    assert exit_code == 0
    out = capsys.readouterr().out
    for name in guard.GUARD_REGISTRY:
        assert name in out


# ---------------------------------------------------------------------------
# (b-cross-host) ledger_overlap -- cross-host claimed-range refusal
# ---------------------------------------------------------------------------

_SYNTHETIC_LEDGER = """# CANONICAL SEED LEDGER -- header line, must be skipped
# Format: [start - end) | owner | purpose | date
[0 - 30,000,000)              | historical | gen-1-era                | pre
[30,000,000 - 66,000,000)     | mixed      | gen-2 (en-dash below)    | d
[82,000,000 – 82,012,000)     | Modal      | wave40a (en-dash variant) | d
[9,900,000,000 - 9,900,001,000) | evals    | hyphen variant row        | d
[100,000,000,000+)            | flywheel   | open-ended gate-eval space | d
NEXT SAFE: prose line that must be ignored by the parser entirely.
"""


def _write_ledger(tmp_path):
    p = tmp_path / "SEED_LEDGER.md"
    p.write_text(_SYNTHETIC_LEDGER)
    return p


def test_parse_seed_ledger_handles_dash_variants_and_open_end(tmp_path):
    rows = guard.parse_seed_ledger(_write_ledger(tmp_path))
    ranges = {(s, e) for s, e, _ in rows}
    # both dash variants parsed:
    assert (0, 30_000_000) in ranges
    assert (82_000_000, 82_012_000) in ranges          # en-dash row
    assert (9_900_000_000, 9_900_001_000) in ranges    # hyphen row
    # open-ended "[N+)" row -> sentinel end:
    assert (100_000_000_000, guard._LEDGER_OPEN_END_SENTINEL) in ranges
    # header + NEXT SAFE prose line skipped, not parsed as ranges:
    assert len(rows) == 5


def test_ledger_overlap_refuses_claimed_range(tmp_path):
    ledger = _write_ledger(tmp_path)
    # the historic 60M near-collision: [60M, 61M) overlaps [30M, 66M)
    result = guard.guard_ledger_overlap(60_000_000, 1_000_000, ledger_path=ledger)
    assert not result.passed
    assert "overlaps" in result.reason


def test_ledger_overlap_allows_disjoint_range(tmp_path):
    ledger = _write_ledger(tmp_path)
    # 6.1B sits in the gap between [30M,66M)... and the 1e11 open-ended row.
    result = guard.guard_ledger_overlap(6_100_000_000, 1_000_000, ledger_path=ledger)
    assert result.passed


def test_ledger_overlap_detects_open_ended_row(tmp_path):
    ledger = _write_ledger(tmp_path)
    result = guard.guard_ledger_overlap(100_000_000_005, 1_000, ledger_path=ledger)
    assert not result.passed
    assert "overlaps" in result.reason


def test_ledger_overlap_fails_closed_when_ledger_missing(tmp_path):
    missing = tmp_path / "does_not_exist.md"
    result = guard.guard_ledger_overlap(1_000, 100, ledger_path=missing)
    assert not result.passed
    assert "not found" in result.reason


def test_ledger_overlap_empty_range_is_noop(tmp_path):
    ledger = _write_ledger(tmp_path)
    result = guard.guard_ledger_overlap(60_000_000, 0, ledger_path=ledger)
    assert result.passed  # games=0 -> empty range, nothing to claim


def test_ledger_overlap_excludes_own_claim(tmp_path):
    # CAT-124: canonical order is claim-then-verify, so the launch's OWN row is
    # already in the ledger at guard time. own_claim_label must exclude it.
    ledger = tmp_path / "SEED_LEDGER.md"
    ledger.write_text(
        "# CANONICAL SEED LEDGER -- header\n"
        "[90000000000 - 91000000000) | fleet/H100 | c6 TEACHER claimid=cat122-c6-w2 | 2026-07-09\n"
    )
    # WITHOUT the label: self-collision -> FAIL (the exact trap that forced --skip-guards)
    self_collide = guard.guard_ledger_overlap(90_000_000_000, 1500, ledger_path=ledger)
    assert not self_collide.passed and "overlaps" in self_collide.reason
    # WITH own_claim_label: our own row is excluded -> PASS (no --skip-guards needed)
    ok = guard.guard_ledger_overlap(
        90_000_000_000, 1500, ledger_path=ledger, own_claim_label="cat122-c6-w2"
    )
    assert ok.passed


def test_ledger_overlap_still_collides_with_peer_despite_own_label(tmp_path):
    # own_claim_label must NOT weaken fail-closed behavior against a genuine peer.
    ledger = tmp_path / "SEED_LEDGER.md"
    ledger.write_text(
        "# CANONICAL SEED LEDGER -- header\n"
        "[90000000000 - 91000000000) | fleet/H100 | c6 TEACHER claimid=cat122-c6-w2 | 2026-07-09\n"
        "[90000001000 - 90000002000) | peer | some-other-host job | 2026-07-09\n"
    )
    result = guard.guard_ledger_overlap(
        90_000_000_000, 1500, ledger_path=ledger, own_claim_label="cat122-c6-w2"
    )
    assert not result.passed  # peer [90000001000,90000002000) overlaps [90000000000,90000001500)
    assert "overlaps" in result.reason
