from __future__ import annotations

import pytest

from tools.perf_common import (
    DOCUMENTED_GPU_PER_LEAF_MS,
    HISTORICAL_SH_FLOOR_OVERRUN_MS,
    append_ledger_rows,
    check_gpu_context_thrash,
    check_sh_floor_overrun,
    load_ledger,
    parse_dmon_pucm,
    parse_gate_summary,
    parse_generation_manifest,
    stable_key,
    summarize_latencies,
)


def test_summarize_latencies_basic() -> None:
    stats = summarize_latencies([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats["n"] == 5.0
    assert stats["mean_ms"] == 3.0
    assert stats["total_ms"] == 15.0
    assert stats["p50_ms"] == 3.0


def test_summarize_latencies_empty() -> None:
    assert summarize_latencies([]) == {
        "mean_ms": 0.0,
        "p50_ms": 0.0,
        "p95_ms": 0.0,
        "total_ms": 0.0,
        "n": 0.0,
    }


def test_append_ledger_rows_dedupes_by_key(tmp_path) -> None:
    output = tmp_path / "ledger.jsonl"
    rows = [{"key": "a", "value": 1}, {"key": "b", "value": 2}]
    assert append_ledger_rows(output, rows) == 2
    assert append_ledger_rows(output, rows) == 0
    assert load_ledger(output) == rows


def test_append_ledger_rows_allows_duplicates_when_disabled(tmp_path) -> None:
    output = tmp_path / "ledger.jsonl"
    rows = [{"key": "a", "value": 1}]
    assert append_ledger_rows(output, rows, dedupe_existing=True) == 1
    assert append_ledger_rows(output, rows, dedupe_existing=False) == 1
    assert len(load_ledger(output)) == 2


def test_load_ledger_missing_file_returns_empty(tmp_path) -> None:
    assert load_ledger(tmp_path / "missing.jsonl") == []


def test_load_ledger_skips_malformed_lines(tmp_path) -> None:
    output = tmp_path / "ledger.jsonl"
    output.write_text('{"key": "a"}\nnot json\n{"key": "b"}\n', encoding="utf-8")
    rows = load_ledger(output)
    assert [row["key"] for row in rows] == ["a", "b"]


def test_parse_generation_manifest_computes_rows_per_hr() -> None:
    payload = {
        "rows": 90000,
        "elapsed_sec": 3600.0,
        "workers": 16,
        "out_dir": "runs/self_play/wave1",
        "games_completed": 50,
    }
    row = parse_generation_manifest(payload, hostname="gpu3")
    assert row["kind"] == "generation"
    assert row["hostname"] == "gpu3"
    assert row["rows_per_sec"] == 25.0
    assert row["rows_per_hr"] == 90000.0
    assert row["games_per_hr"] == 50.0


def test_parse_generation_manifest_falls_back_to_computed_rows_per_sec() -> None:
    payload = {"rows": 3600, "elapsed_sec": 3600.0, "out_dir": "x"}
    row = parse_generation_manifest(payload)
    assert row["rows_per_sec"] == 1.0
    assert row["hostname"] is None


def test_parse_gate_summary_extracts_extension_tier() -> None:
    payload = {
        "games": 400,
        "elapsed_sec": 10.0,
        "tiers": [400, 800, 1200],
        "tier_index": 1,
        "decision": "continue",
    }
    row = parse_gate_summary(payload, gate_name="gen2_vs_gen1", summary_path="p.json")
    assert row["extended"] is True
    assert row["games_per_sec"] == 40.0
    assert row["decision"] == "continue"


def test_parse_gate_summary_no_extension_at_tier_zero() -> None:
    payload = {"games": 400, "elapsed_sec": 10.0, "tiers": [400, 800, 1200], "tier_index": 0, "decision": "H1"}
    row = parse_gate_summary(payload, gate_name="g", summary_path="p.json")
    assert row["extended"] is False


def test_parse_gate_summary_wall_clock_override_beats_payload() -> None:
    payload = {"games": 100, "elapsed_sec": 999.0}
    row = parse_gate_summary(payload, gate_name="g", summary_path="p.json", wall_clock_sec=5.0)
    assert row["elapsed_sec"] == 5.0
    assert row["games_per_sec"] == 20.0


def test_parse_gate_summary_falls_back_to_games_completed() -> None:
    payload = {"games_completed": 8, "elapsed_sec": 4.0}
    row = parse_gate_summary(payload, gate_name="g", summary_path="p.json")
    assert row["games"] == 8


def test_parse_gate_summary_records_hostname() -> None:
    # CAT-71 review finding 3: gate rows need a hostname field so
    # tools/perf_report.py can group per-host and not compare wall-clock
    # across different fleet hosts (B200/A100A/A100B).
    payload = {"games": 100, "elapsed_sec": 10.0}
    row = parse_gate_summary(payload, gate_name="g", summary_path="p.json", hostname="a100a")
    assert row["hostname"] == "a100a"


def test_parse_gate_summary_key_changes_with_decision_reversal() -> None:
    # CAT-71 review finding 2: a re-parse of the SAME summary (same
    # games/elapsed) whose decision flipped (CONTINUE -> H1) used to produce
    # an IDENTICAL dedupe key (metrics-only), so `append_ledger_rows` would
    # silently drop the second (correct, reversed) row. `decision` is now
    # part of the key, so the two rows get distinct keys.
    payload_continue = {"games": 400, "elapsed_sec": 10.0, "decision": "CONTINUE"}
    payload_h1 = {"games": 400, "elapsed_sec": 10.0, "decision": "H1"}
    row_continue = parse_gate_summary(payload_continue, gate_name="g", summary_path="p.json")
    row_h1 = parse_gate_summary(payload_h1, gate_name="g", summary_path="p.json")
    assert row_continue["key"] != row_h1["key"]


def test_parse_gate_summary_key_includes_timestamp() -> None:
    # Same games/elapsed/decision, parsed twice (e.g. a monitoring loop
    # re-reading the same on-disk summary), gets distinct keys because a
    # fetch-time timestamp is folded in -- matching the existing
    # `leaf`/`gpu_util` row kinds, which already fold a timestamp into their
    # own dedupe keys for the same reason.
    payload = {"games": 400, "elapsed_sec": 10.0, "decision": "H1"}
    row = parse_gate_summary(payload, gate_name="g", summary_path="p.json")
    assert row["timestamp"]
    assert row["timestamp"] in row["key"]


def test_append_ledger_rows_logs_on_dedupe(capsys, tmp_path) -> None:
    # CAT-71 review finding 2: a dedupe firing must be logged, not silently
    # skipped.
    output = tmp_path / "ledger.jsonl"
    rows = [{"key": "dup-key", "kind": "gate", "value": 1}]
    assert append_ledger_rows(output, rows) == 1
    assert append_ledger_rows(output, rows) == 0
    captured = capsys.readouterr()
    assert "dedupe" in captured.err
    assert "dup-key" in captured.err


def test_parse_gate_summary_real_h2h_writer_schema() -> None:
    # tools/gumbel_search_vs_bot_h2h.py / gumbel_search_vs_raw_h2h.py /
    # gumbel_search_cross_net_h2h.py all write "games_played" as the int
    # count and "games" as the *list* of per-game dicts. Before this fix,
    # the plain `games` fallback treated the list as the count and crashed
    # on `int(list)`.
    payload = {
        "games_played": 400,
        "games_with_winner": 390,
        "elapsed_sec": 120.0,
        "games": [{"candidate_won": True}, {"candidate_won": False}],
    }
    row = parse_gate_summary(payload, gate_name="g", summary_path="p.json")
    assert row["games"] == 400
    assert row["games_per_sec"] == pytest.approx(400 / 120.0)


def test_check_gpu_context_thrash_flags_documented_signature() -> None:
    # docs/plans/CATAN_ZERO_RESEARCH_CHRONICLE.md section 10.1's fleet anomaly.
    assert check_gpu_context_thrash(90.0, 1.0) is True


def test_check_gpu_context_thrash_does_not_flag_genuine_compute_load() -> None:
    assert check_gpu_context_thrash(90.0, 40.0) is False


def test_check_gpu_context_thrash_does_not_flag_idle_gpu() -> None:
    assert check_gpu_context_thrash(5.0, 1.0) is False


def test_check_sh_floor_overrun_flags_historical_incidents() -> None:
    # CAT-71 ground truth: 32ms/105ms/119ms overruns. See
    # tools/perf_common.py's HISTORICAL_SH_FLOOR_OVERRUN_MS docstring for why
    # DOCUMENTED_GPU_PER_LEAF_MS (in-repo, 3.4ms) stands in as an illustrative
    # floor here rather than the incident's own (not-in-this-checkout) floor.
    for observed in HISTORICAL_SH_FLOOR_OVERRUN_MS:
        assert check_sh_floor_overrun(observed, DOCUMENTED_GPU_PER_LEAF_MS) is True


def test_check_sh_floor_overrun_does_not_flag_within_tolerance() -> None:
    assert check_sh_floor_overrun(4.0, 3.4) is False  # 1.18x < default 1.5x ratio


def test_check_sh_floor_overrun_zero_floor_never_flags() -> None:
    assert check_sh_floor_overrun(1000.0, 0.0) is False


def test_parse_dmon_pucm_single_gpu() -> None:
    text = (
        "# gpu   pwr gtemp mtemp    sm   mem   enc   dec   jpg   ofa  mclk  pclk\n"
        "# Idx     W     C     C     %     %     %     %     %     %   MHz   MHz\n"
        "    0   250    45     -    92     1     0     0     0     0  2619  1980\n"
    )
    rows = parse_dmon_pucm(text)
    assert len(rows) == 1
    assert rows[0]["gpu"] == 0
    assert rows[0]["sm"] == 92
    assert rows[0]["mem"] == 1
    assert rows[0]["mtemp"] is None


def test_parse_dmon_pucm_multi_gpu() -> None:
    text = (
        "# gpu   pwr gtemp mtemp    sm   mem\n"
        "# Idx     W     C     C     %     %\n"
        "    0   250    45    60    92     1\n"
        "    1   240    44    58     8    10\n"
    )
    rows = parse_dmon_pucm(text)
    assert [row["gpu"] for row in rows] == [0, 1]
    assert rows[1]["sm"] == 8
    assert rows[1]["mem"] == 10


def test_parse_dmon_pucm_ignores_blank_lines() -> None:
    text = "\n# gpu sm mem\n# Idx % %\n\n  0  10  20\n\n"
    rows = parse_dmon_pucm(text)
    assert len(rows) == 1
    assert rows[0]["sm"] == 10


def test_stable_key_joins_parts() -> None:
    assert stable_key("a", 1, None) == "a|1|None"
