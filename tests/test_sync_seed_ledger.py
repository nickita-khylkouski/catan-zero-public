"""Unit tests for tools/sync_seed_ledger.py (CAT-125): cross-host seed-ledger
dedupe-sync. Each test reproduces a drift/merge scenario the tool must handle
without blind-appending. See the module docstring in tools/sync_seed_ledger.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import sync_seed_ledger as sync  # type: ignore  # noqa: E402

HEADER = (
    "# CANONICAL SEED LEDGER — check + append BEFORE claiming ANY base seed\n"
    "# Format: [start – end) | owner | purpose | date\n"
)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(HEADER + body)
    return path


# ---------------------------------------------------------------------------
# dedupe by claim id -- the shared-claim-across-copies case
# ---------------------------------------------------------------------------


def test_shared_claim_id_appears_once(tmp_path):
    # Both copies carry the same claim (id c2-teacher-w1-100); a third,
    # copy-A-only claim must survive the merge (union).
    a = _write(
        tmp_path,
        "a.md",
        "[71,000,000,000 – 72,000,000,000) | fleet | teacher claim=c2-teacher-w1-100 | d1\n"
        "[80,000,000,000 – 81,000,000,000) | fleet | only-on-a claim=c9-x-w1-1 | d1\n",
    )
    b = _write(
        tmp_path,
        "b.md",
        "[71,000,000,000 – 72,000,000,000) | fleet | teacher claim=c2-teacher-w1-100 | d1\n",
    )
    rows, report = sync.sync_ledgers([a, b])
    assert report.rows_in == 3
    assert report.rows_out == 2
    assert report.duplicates_collapsed == 1
    ids = sorted(r.claim_id for r in rows)
    assert ids == ["c2-teacher-w1-100", "c9-x-w1-1"]
    assert report.ok


def test_shared_claim_id_survives_reformatted_purpose(tmp_path):
    # Same claim id, but box B reworded the purpose/date -- still ONE row.
    a = _write(tmp_path, "a.md", "[10 – 20) | fleet | teacher n64 claim=k1 | 2026-07-08\n")
    b = _write(tmp_path, "b.md", "[10 – 20) | fleet/H100 | TEACHER (reworded) claim=k1 | 2026-07-09\n")
    _, report = sync.sync_ledgers([a, b])
    assert report.rows_out == 1
    assert report.ok


# ---------------------------------------------------------------------------
# claim-id collision -- different range under one id is a hard error
# ---------------------------------------------------------------------------


def test_same_claim_id_different_range_is_conflict(tmp_path):
    a = _write(tmp_path, "a.md", "[10 – 20) | fleet | x claim=dup | d\n")
    b = _write(tmp_path, "b.md", "[30 – 40) | fleet | x claim=dup | d\n")
    _, report = sync.sync_ledgers([a, b])
    assert not report.ok
    assert len(report.id_conflicts) == 1
    assert "dup" in report.id_conflicts[0]


# ---------------------------------------------------------------------------
# legacy (no claim id) rows -- conservative full-row dedup
# ---------------------------------------------------------------------------


def test_identical_legacy_rows_collapse(tmp_path):
    row = "[0 – 30,000,000) | historical | gen-1-era | pre-2026-07-06\n"
    a = _write(tmp_path, "a.md", row)
    b = _write(tmp_path, "b.md", row)
    _, report = sync.sync_ledgers([a, b])
    assert report.rows_out == 1


def test_legacy_rows_differing_in_label_are_both_kept(tmp_path):
    # Same range, DIFFERENT label, no claim id -> conservative: keep both
    # (never silently drop a genuinely different legacy claim).
    a = _write(tmp_path, "a.md", "[10 – 20) | fleet | purpose-A | d\n")
    b = _write(tmp_path, "b.md", "[10 – 20) | fleet | purpose-B | d\n")
    _, report = sync.sync_ledgers([a, b])
    assert report.rows_out == 2
    assert report.ok  # differing legacy labels are NOT an id conflict


def test_legacy_rows_whitespace_only_diff_collapse(tmp_path):
    a = _write(tmp_path, "a.md", "[10 – 20) | fleet | purpose here | d\n")
    b = _write(tmp_path, "b.md", "[10 – 20) | fleet |   purpose    here   | d\n")
    _, report = sync.sync_ledgers([a, b])
    assert report.rows_out == 1


# ---------------------------------------------------------------------------
# rendering + open-ended + sort order
# ---------------------------------------------------------------------------


def test_open_ended_row_round_trips(tmp_path):
    a = _write(tmp_path, "a.md", "[100,000,000,000+) | flywheel | open gate space | d\n")
    rows, _ = sync.sync_ledgers([a])
    assert rows[0].end == sync._OPEN_END
    assert rows[0].render() == "[100,000,000,000+) | flywheel | open gate space | d"


def test_rows_sorted_by_start(tmp_path):
    a = _write(
        tmp_path,
        "a.md",
        "[30 – 40) | f | c claim=c3 | d\n"
        "[10 – 20) | f | a claim=c1 | d\n"
        "[20 – 30) | f | b claim=c2 | d\n",
    )
    rows, _ = sync.sync_ledgers([a])
    assert [r.start for r in rows] == [10, 20, 30]


# ---------------------------------------------------------------------------
# idempotency -- the core CAT-125 guarantee
# ---------------------------------------------------------------------------


def test_sync_is_idempotent(tmp_path):
    a = _write(
        tmp_path,
        "a.md",
        "[71,000,000,000 – 72,000,000,000) | fleet | teacher claim=c2-t-w1-1 | d\n"
        "[0 – 30,000,000) | historical | gen-1 | pre\n",
    )
    b = _write(
        tmp_path,
        "b.md",
        "[72,000,000,000 – 73,000,000,000) | fleet | teacher claim=c3-t-w1-1 | d\n"
        "[0 – 30,000,000) | historical | gen-1 | pre\n"
        "NEXT SAFE: use 6,100,000,000+ for bulk generation\n",
    )
    header = sync._header_lines(a)
    next_safe = sync._next_safe_lines([a, b])
    rows1, _ = sync.sync_ledgers([a, b])
    out1 = sync.render_ledger(rows1, header, next_safe)

    # Re-sync the canonical output: must be byte-identical.
    canonical = tmp_path / "canonical.md"
    canonical.write_text(out1)
    header2 = sync._header_lines(canonical)
    next_safe2 = sync._next_safe_lines([canonical])
    rows2, _ = sync.sync_ledgers([canonical])
    out2 = sync.render_ledger(rows2, header2, next_safe2)
    assert out1 == out2


def test_check_passes_on_canonical_file(tmp_path, capsys):
    a = _write(tmp_path, "a.md", "[10 – 20) | f | x claim=k1 | d\n")
    rows, _ = sync.sync_ledgers([a])
    header = sync._header_lines(a)
    canonical = tmp_path / "canon.md"
    canonical.write_text(sync.render_ledger(rows, header, []))
    assert sync.main([str(canonical), "--check"]) == 0


def test_check_fails_on_drifted_file(tmp_path):
    # Unsorted + duplicated -> not canonical -> --check exits 2.
    a = _write(
        tmp_path,
        "a.md",
        "[30 – 40) | f | b claim=k2 | d\n"
        "[10 – 20) | f | a claim=k1 | d\n"
        "[10 – 20) | f | a claim=k1 | d\n",
    )
    assert sync.main([str(a), "--check"]) == 2


def test_overlaps_are_reported_but_not_fatal(tmp_path):
    a = _write(
        tmp_path,
        "a.md",
        "[30,000,000 – 66,000,000) | mixed | gen-2 claim=m1 | d\n"
        "[63,000,000 – 77,000,000) | fleet | gen2a claim=f1 | d\n",
    )
    _, report = sync.sync_ledgers([a])
    assert len(report.overlaps) == 1
    assert report.ok  # overlaps are informational, not a sync failure


def test_main_writes_output(tmp_path):
    a = _write(tmp_path, "a.md", "[10 – 20) | f | x claim=k1 | d\n")
    out = tmp_path / "out.md"
    assert sync.main([str(a), "-o", str(out)]) == 0
    assert out.exists()
    # Written file is itself canonical (re-check passes).
    assert sync.main([str(out), "--check"]) == 0


def test_missing_input_errors(tmp_path):
    assert sync.main([str(tmp_path / "nope.md")]) == 2
