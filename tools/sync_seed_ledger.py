"""Idempotent cross-host seed-ledger dedupe-sync (CAT-125).

Root cause this fixes: the cross-host ``runs/SEED_LEDGER.md`` is the source of
truth every host/Modal launch must consult before claiming a base seed (see
``tools/prelaunch_guard.py:guard_ledger_overlap``), but each host keeps its own
copy and they DRIFT -- box A appends a claim, box B appends a different claim,
and neither copy has both. The naive "reconcile" is to concatenate the copies,
which BLIND-APPENDS: the rows every copy already shares get duplicated N times,
the file grows without bound, and the overlap guard starts reporting a claim
colliding with its own duplicate. This tool merges N ledger copies into ONE
canonical file by claim identity, so:

  * running it on the same inputs twice produces byte-identical output
    (idempotent -- ``--check`` asserts this in CI);
  * a claim present in several copies appears exactly ONCE in the output;
  * a claim present in only one copy is preserved (union, never drop);
  * two hosts that claimed DIFFERENT ranges under the SAME claim id is a real
    collision and is reported loudly (non-zero exit) rather than silently
    picking one.

Claim identity (the dedupe key):
  * If a row's label carries a ``claim=<id>`` token (the convention the
    canonical launcher writes -- ``claim=<host>-<role>-<wave>-<epoch>``, also
    exported as ``$CATAN_LEDGER_CLAIM_ID`` for the CAT-124 own-claim guard
    exclusion), rows are deduped by that id. This is the robust key: it
    survives trivial reformatting of the purpose/date fields across copies.
  * Otherwise (legacy rows predating the convention) the whole normalized row
    ``(start, end, whitespace-collapsed label)`` is the key -- conservative,
    so a legacy row is only ever collapsed with a byte-equivalent legacy row,
    never with a genuinely different one.

Grammar is NOT re-implemented here: this reuses ``prelaunch_guard`` (the
launch-time overlap guard) as the single source of the row grammar, so the
canonicalizer and the guard can never disagree on what a claimed range is.

CLI::

    # merge every host copy into a canonical file (idempotent)
    python tools/sync_seed_ledger.py copies/*.md -o runs/SEED_LEDGER.md

    # CI / pre-commit: fail if the checked-in ledger is not already canonical
    python tools/sync_seed_ledger.py runs/SEED_LEDGER.md --check

    # dry inspect: print canonical to stdout + report to stderr
    python tools/sync_seed_ledger.py copies/*.md
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import prelaunch_guard  # type: ignore  # noqa: E402

# The launcher embeds a globally-unique claim id in the purpose field as
# ``claim=<token>`` (token is bar/space-free: host-role-wave-epoch). This is the
# same token the CAT-124 own-claim guard exclusion matches on and that the
# launcher exports as $CATAN_LEDGER_CLAIM_ID -- one id, three consumers.
_CLAIM_ID_RE = re.compile(r"claim=([^\s|]+)")

# Rows are rendered with the en-dash the header comment documents
# (``[start - end)``); open-ended claims keep the ``[N+)`` form the guard's
# sentinel round-trips to.
_OPEN_END = prelaunch_guard._LEDGER_OPEN_END_SENTINEL


class ClaimRow(NamedTuple):
    start: int
    end: int
    label: str  # everything after "[range) |": "owner | purpose | date"

    @property
    def claim_id(self) -> str | None:
        match = _CLAIM_ID_RE.search(self.label)
        return match.group(1) if match else None

    def dedupe_key(self) -> tuple:
        """Identity for merging. Prefer the explicit claim id; fall back to the
        whole normalized row so legacy (no-id) rows only collapse with an
        equivalent legacy row, never a genuinely different one."""
        cid = self.claim_id
        if cid is not None:
            return ("id", cid)
        return ("row", self.start, self.end, _norm_ws(self.label))

    def render(self) -> str:
        left = f"[{self.start:,} "
        if self.end == _OPEN_END:
            span = f"{self.start:,}+"
        else:
            span = f"{self.start:,} – {self.end:,}"
        return f"[{span}) | {self.label.strip()}"


def _norm_ws(text: str) -> str:
    return " ".join(text.split())


class SyncReport(NamedTuple):
    inputs: int
    rows_in: int
    rows_out: int
    duplicates_collapsed: int
    dropped_non_rows: int
    id_conflicts: list[str]
    overlaps: list[str]

    @property
    def ok(self) -> bool:
        # Overlaps are informational (the ledger has intentional historical
        # overlaps); only same-id-different-range conflicts are hard errors.
        return not self.id_conflicts


def _parse_copy(path: Path) -> tuple[list[ClaimRow], int]:
    """Parse one ledger copy into ClaimRows via the canonical guard parser.
    Returns (rows, dropped_non_row_line_count)."""
    text = path.read_text()
    total_lines = sum(1 for _ in text.splitlines())
    parsed = prelaunch_guard.parse_seed_ledger(path)
    rows = [ClaimRow(start, end, label) for (start, end, label) in parsed]
    # Non-row lines = header ("#"), NEXT SAFE prose, blanks, malformed. The
    # canonicalizer preserves the header + NEXT SAFE explicitly (below); the
    # remainder is noise the guard already ignores.
    return rows, max(0, total_lines - len(rows))


def _header_lines(path: Path) -> list[str]:
    """Leading ``#`` comment block (the format doc) from a ledger copy."""
    out: list[str] = []
    for raw in path.read_text().splitlines():
        if raw.lstrip().startswith("#"):
            out.append(raw.rstrip())
        elif raw.strip() == "":
            continue
        else:
            break
    return out


def _next_safe_lines(paths: list[Path]) -> list[str]:
    """Collect distinct ``NEXT SAFE:`` guidance lines across all copies,
    sorted for determinism."""
    seen: set[str] = set()
    for path in paths:
        for raw in path.read_text().splitlines():
            if raw.lstrip().startswith("NEXT SAFE:"):
                seen.add(raw.strip())
    return sorted(seen)


def _overlaps(rows: list[ClaimRow]) -> list[str]:
    """Report pairs of DISTINCT claims whose ranges overlap. Informational:
    the ledger intentionally carries historical overlaps (e.g. 'mixed' and
    'fleet' rows over the gen-2 era). The launch-time guard is what refuses a
    NEW overlapping claim; this is a visibility aid for the operator."""
    findings: list[str] = []
    ordered = sorted(rows, key=lambda r: (r.start, r.end))
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            if b.start >= a.end:
                break  # sorted by start; no later row can overlap a
            findings.append(f"{a.render()}  <=>  {b.render()}")
    return findings


def sync_ledgers(paths: list[Path]) -> tuple[list[ClaimRow], SyncReport]:
    """Merge N ledger copies into a canonical, deduped, sorted row list."""
    rows_in = 0
    dropped = 0
    # First-seen wins for a given identity; deterministic because inputs are
    # processed in the caller-supplied (typically sorted-glob) order and rows
    # within a file keep file order.
    by_key: dict[tuple, ClaimRow] = {}
    conflicts: list[str] = []
    for path in paths:
        rows, drop = _parse_copy(path)
        dropped += drop
        for row in rows:
            rows_in += 1
            key = row.dedupe_key()
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = row
                continue
            # Same identity seen again. For claim-id keys, a differing RANGE is
            # a genuine cross-host collision (two boxes claimed different seeds
            # under one id) -- surface it loudly. A differing label under the
            # same id is a benign reformat; keep first-seen.
            if key[0] == "id" and (existing.start, existing.end) != (row.start, row.end):
                conflicts.append(
                    f"claim id {key[1]!r} claims BOTH "
                    f"[{existing.start:,}, {existing.end:,}) and "
                    f"[{row.start:,}, {row.end:,}) across copies"
                )
    canonical = sorted(by_key.values(), key=lambda r: (r.start, r.end))
    report = SyncReport(
        inputs=len(paths),
        rows_in=rows_in,
        rows_out=len(canonical),
        duplicates_collapsed=rows_in - len(canonical),
        dropped_non_rows=dropped,
        id_conflicts=conflicts,
        overlaps=_overlaps(canonical),
    )
    return canonical, report


def render_ledger(rows: list[ClaimRow], header: list[str], next_safe: list[str]) -> str:
    """Assemble the canonical ledger text: header comment block, then sorted
    unique rows, then NEXT SAFE guidance. Deterministic + round-trip stable."""
    parts: list[str] = []
    parts.extend(header)
    parts.extend(row.render() for row in rows)
    parts.extend(next_safe)
    return "\n".join(parts) + "\n"


def _print_report(report: SyncReport, *, stream=sys.stderr) -> None:
    print(
        f"[sync_seed_ledger] {report.inputs} cop(y|ies): {report.rows_in} rows in "
        f"-> {report.rows_out} unique ({report.duplicates_collapsed} duplicate(s) "
        f"collapsed, {report.dropped_non_rows} non-row line(s) dropped)",
        file=stream,
    )
    if report.overlaps:
        print(
            f"[sync_seed_ledger] {len(report.overlaps)} overlapping claim pair(s) "
            "(informational -- launch guard refuses NEW overlaps):",
            file=stream,
        )
        for line in report.overlaps:
            print(f"    {line}", file=stream)
    if report.id_conflicts:
        print(
            f"[sync_seed_ledger] ERROR: {len(report.id_conflicts)} claim-id "
            "collision(s) -- resolve by hand before syncing:",
            file=stream,
        )
        for line in report.id_conflicts:
            print(f"    {line}", file=stream)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Idempotent cross-host seed-ledger dedupe-sync (CAT-125).",
    )
    parser.add_argument(
        "ledgers",
        nargs="+",
        type=Path,
        help="One or more SEED_LEDGER.md copies to merge (host copies scp'd local).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write canonical ledger here (default: stdout).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 2 if OUTPUT (or the sole input) is not already "
        "canonical. Use in CI/pre-commit to enforce a synced ledger.",
    )
    args = parser.parse_args(argv)

    missing = [str(p) for p in args.ledgers if not p.exists()]
    if missing:
        print(f"[sync_seed_ledger] ERROR: input(s) not found: {', '.join(missing)}", file=sys.stderr)
        return 2

    header = _header_lines(args.ledgers[0])
    next_safe = _next_safe_lines(args.ledgers)
    rows, report = sync_ledgers(args.ledgers)
    canonical_text = render_ledger(rows, header, next_safe)

    _print_report(report)
    if not report.ok:
        return 2

    if args.check:
        target = args.output if args.output is not None else args.ledgers[0]
        if not target.exists():
            print(f"[sync_seed_ledger] --check: {target} does not exist", file=sys.stderr)
            return 2
        current = target.read_text()
        if current != canonical_text:
            print(
                f"[sync_seed_ledger] --check FAILED: {target} is not canonical "
                "(run without --check to rewrite it).",
                file=sys.stderr,
            )
            return 2
        print(f"[sync_seed_ledger] --check OK: {target} is canonical.", file=sys.stderr)
        return 0

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(canonical_text)
        print(f"[sync_seed_ledger] wrote {report.rows_out} rows -> {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(canonical_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
