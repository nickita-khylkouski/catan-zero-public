#!/usr/bin/env python3
"""Pre-launch guard runner (CAT-69): every documented incident class the
project has actually hit, encoded as an automated check instead of a
checklist a human is trusted to remember.

Each guard below is a pure function: given the state a launch is about to
use (an argv, a checkpoint path, a seed range, a report.json, ...), it
returns a `GuardResult` (PASS/FAIL + a human-readable reason) and never
mutates anything or exits the process itself. That makes every guard
independently unit-testable with synthetic pass/fail fixtures, and lets a
generation/training/gate launch script decide what to do with a FAIL
(usually: print the reason and refuse to proceed).

A few guards need live host state (open file descriptor limits, whether a
lock file is currently held by another process) rather than being pure
functions of their arguments; those are marked `host_only=True` in their
`GuardResult` and flagged as such in `--help`/the CLI table, since they can
legitimately differ between CI and the real launch host.

Incident-to-guard map (see CATAN_ZERO_ROADMAP.md Sec.1 "standing rules" and
CAT-69 for the citations):

  (a) guard_cli_flag_lint            -- CLI-default-override trap (7+ incidents,
                                         c_scale 1.0-vs-0.1 near-miss)
  (b) guard_seed_ledger /
      guard_val_only_never_trains    -- seed-collision class (task #77);
                                         VAL-ONLY [6.19B, 6.2B) never trains
  (c) guard_masked_regime            -- masked-checkpoint regime mismatch
                                         (task #76 safety net, #71 hidden-info leak)
  (d) guard_provenance               -- doc-vs-artifact drift (2026-07-06 audit)
  (e) guard_fd_limit                 -- round-12 Errno 24 fd-exhaustion crash
  (f) guard_explicit_pid_kill        -- orphaned-children broad-pattern kill
  (g) guard_pgrep_self_match /
      guard_fcntl_lock_present /
      guard_lock_available           -- pgrep self-match trap
  (h) guard_corpus_no_duplicate_seeds /
      guard_wave_root_quarantine     -- dup-seed detector (task #85) +
                                         wave-root quarantine
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from seed_fleet_planner import assert_disjoint_seed_blocks


@dataclass(frozen=True, slots=True)
class GuardResult:
    guard: str
    passed: bool
    reason: str
    host_only: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "guard": self.guard,
            "status": self.status,
            "reason": self.reason,
            "host_only": self.host_only,
            "details": self.details,
        }


def _ok(guard: str, reason: str, *, host_only: bool = False, **details: Any) -> GuardResult:
    return GuardResult(guard=guard, passed=True, reason=reason, host_only=host_only, details=details)


def _fail(guard: str, reason: str, *, host_only: bool = False, **details: Any) -> GuardResult:
    return GuardResult(guard=guard, passed=False, reason=reason, host_only=host_only, details=details)


# ---------------------------------------------------------------------------
# (a) CLI-default-override trap
# ---------------------------------------------------------------------------


def discover_configurable_flags(parser: argparse.ArgumentParser) -> list[str]:
    """Return every optional (non-required) flag's recognized spellings.

    These are exactly the flags a caller CAN silently omit and fall back to
    whatever default the script's author picked -- the class of bug behind
    the c_scale 1.0-vs-0.1 near-miss and 6+ prior incidents. Required flags
    are excluded: argparse already forces those to be explicit.

    All option strings for each action are returned (not just the longest),
    so ``BooleanOptionalAction`` pairs like ``--public-observation`` and
    ``--no-public-observation`` are both recognized by the stale-flag check.
    """
    flags: list[str] = []
    for action in parser._actions:  # noqa: SLF001 -- argparse has no public introspection API
        if not action.option_strings:
            continue  # positional
        if isinstance(action, argparse._HelpAction):  # noqa: SLF001
            continue
        if action.required:
            continue
        flags.extend(action.option_strings)
    return flags


def guard_cli_flag_lint(
    argv: Sequence[str],
    critical_flags: Sequence[str],
    *,
    parser: argparse.ArgumentParser | None = None,
    expected_values: Mapping[str, Any] | None = None,
    forbidden_flags: Sequence[str] = (),
) -> GuardResult:
    """FAIL if any `critical_flags` entry is absent from `argv` (as a bare
    token `--flag` or `--flag=value`), meaning the launch would silently
    rely on that flag's parser default rather than an explicit value.

    `expected_values` (optional) maps a subset of the critical flags to the
    exact resolved value they must have. When a parser is provided, this
    guard parses ``argv`` and compares the resolved value to the expected
    value, so passing ``--c-scale 0.1`` or ``--no-public-observation`` is
    caught as a FAIL even though the token itself is present. This closes
    the "token-only" gap where unsafe values could masquerade as explicit
    flags (CAT-69 follow-up / CAT-88 silent-default class).

    `forbidden_flags` closes the inverse gap for nullable/optional modes.  A
    typed config can deliberately leave (for example) an opponent manifest or
    an adaptive search budget unset, but argparse has no ``--no-...`` spelling
    for a nullable value.  Listing the positive flag here proves that a caller
    did not override the sealed ``None`` with a command-line value.

    `critical_flags` is normally a hand-maintained list, which can itself
    drift out of sync with the target script (a flag gets renamed or
    removed and nobody updates the list, so the guard keeps "passing" on a
    flag that no longer does anything). Pass the target script's real
    `parser` -- typically obtained by importing the script and calling its
    `build_parser()` -- to close that gap: every `critical_flags` entry is
    first cross-checked against `discover_configurable_flags(parser)`, and
    a critical flag that isn't a real optional flag on that parser is
    reported as stale config rather than silently treated as present-or-
    absent. Without a `parser`, `critical_flags` is trusted as given.
    """
    expected_values = dict(expected_values or {})
    all_critical = list(critical_flags)
    for flag in expected_values:
        if flag not in all_critical:
            all_critical.append(flag)

    if parser is not None:
        real_flags = set(discover_configurable_flags(parser))
        stale = [
            flag
            for flag in [*all_critical, *forbidden_flags]
            if flag not in real_flags
        ]
        if stale:
            return _fail(
                "cli_flag_lint",
                f"critical_flags config references flag(s) that are not real optional "
                f"flags on the target parser: {stale} (stale config -- the flag was "
                f"renamed, made required, or removed; a hardcoded flag list drifted out "
                f"of sync with the script it's supposed to lint). Known optional flags: "
                f"{sorted(real_flags)}.",
                stale_flags=stale,
            )

    def _is_present(flag: str) -> bool:
        if flag in argv_set:
            return True
        if any(tok.startswith(flag + "=") for tok in argv_tokens):
            return True
        if parser is not None:
            action = parser._option_string_actions.get(flag)
            if action is not None:
                for option in action.option_strings:
                    if option in argv_set:
                        return True
                    if any(tok.startswith(option + "=") for tok in argv_tokens):
                        return True
        return False

    argv_tokens = list(argv)
    argv_set = set(argv_tokens)
    forbidden_present = [flag for flag in forbidden_flags if _is_present(flag)]
    if forbidden_present:
        return _fail(
            "cli_flag_lint",
            "launch supplies forbidden override(s) for sealed nullable/optional "
            f"fields: {forbidden_present}. Remove these flags; this recipe requires "
            "the typed-config null/off value.",
            forbidden_flags=forbidden_present,
        )
    missing = [flag for flag in all_critical if not _is_present(flag)]
    if missing:
        return _fail(
            "cli_flag_lint",
            f"launch omits explicit value(s) for critical flag(s), silently relying on "
            f"the parser default: {missing}. Pass every flag in {all_critical!r} "
            "explicitly (CLI-default-override trap, 7+ prior incidents).",
            missing_flags=missing,
        )

    if expected_values and parser is not None:
        try:
            parsed = parser.parse_args(argv_tokens)
        except SystemExit as exc:
            # argparse exits on malformed argv; treat as a guard failure so the
            # caller gets a reason instead of the whole process terminating.
            return _fail(
                "cli_flag_lint",
                f"could not parse argv for value-checking: {exc}",
            )
        value_errors: list[str] = []
        for flag, expected in expected_values.items():
            action = parser._option_string_actions.get(flag)
            if action is None:
                # Should have been caught by stale-flag check above; defensive.
                value_errors.append(f"{flag}: not a real flag")
                continue
            actual = getattr(parsed, action.dest, None)
            if actual != expected:
                value_errors.append(f"{flag}={actual!r} (expected {expected!r})")
        if value_errors:
            return _fail(
                "cli_flag_lint",
                f"critical flag(s) resolved to an unsafe value: {value_errors}. "
                "Pass the production value explicitly, or update the guard config.",
                value_errors=value_errors,
            )

    return _ok("cli_flag_lint", f"all {len(all_critical)} critical flag(s) explicit in argv")


# ---------------------------------------------------------------------------
# (b) seed-ledger enforcement + VAL-ONLY range
# ---------------------------------------------------------------------------

# Roadmap Sec.1 standing rule: "Seeds: consult the seed ledger before any
# claim; VAL-ONLY range [6.19B, 6.2B) never trains." Half-open, matching
# assert_disjoint_seed_blocks' convention.
VAL_ONLY_SEED_RANGE: tuple[int, int] = (6_190_000_000, 6_200_000_000)


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def guard_val_only_never_trains(seed_range: tuple[int, int], *, purpose: str) -> GuardResult:
    """Hard-block any `purpose="train"` launch whose [start, end) seed range
    overlaps the VAL-ONLY band. Generation/eval launches may freely use that
    band; only training consumption of it is forbidden.
    """
    if purpose == "train" and _ranges_overlap(seed_range, VAL_ONLY_SEED_RANGE):
        return _fail(
            "val_only_never_trains",
            f"training seed range [{seed_range[0]}, {seed_range[1]}) overlaps the "
            f"VAL-ONLY range [{VAL_ONLY_SEED_RANGE[0]}, {VAL_ONLY_SEED_RANGE[1]}), which "
            "must never be trained on (roadmap Sec.1 standing rule).",
        )
    return _ok(
        "val_only_never_trains",
        f"seed range [{seed_range[0]}, {seed_range[1]}) is compatible with purpose={purpose!r}",
    )


def guard_seed_ledger(
    out_dir: str | Path,
    base_seed: int,
    games: int,
    *,
    claims_dir: str | Path | None = None,
    purpose: str = "generate",
) -> GuardResult:
    """Mandatory, non-mutating pre-launch version of
    generate_gumbel_selfplay_data.py's `_claim_seed_range` guard: checks the
    proposed [base_seed, base_seed+games) range against every OTHER live
    claim in `claims_dir` (default: `out_dir.parent/.seed_claims`) and
    against the VAL-ONLY band, but never writes a claim file itself -- the
    actual claiming remains the generation script's job once this guard
    passes and the launch proceeds.
    """
    val_result = guard_val_only_never_trains((int(base_seed), int(base_seed) + int(games)), purpose=purpose)
    if not val_result.passed:
        return _fail("seed_ledger", val_result.reason)

    out_dir = Path(out_dir)
    resolved_out_dir = str(out_dir.resolve())
    claims_root = Path(claims_dir) if claims_dir is not None else out_dir.parent / ".seed_claims"

    others: list[tuple[str, int, int]] = []
    if claims_root.exists():
        for candidate in sorted(claims_root.glob("*.json")):
            try:
                payload = json.loads(candidate.read_text())
                other_out_dir = str(payload["out_dir"])
                other_base_seed = int(payload["base_seed"])
                other_games = int(payload["games"])
            except (OSError, ValueError, KeyError, TypeError):
                continue  # stale/malformed claim file -- ignore rather than block launches.
            if other_out_dir == resolved_out_dir:
                continue  # same out-dir -- a resume, not a peer.
            others.append((other_out_dir, other_base_seed, other_games))

    try:
        assert_disjoint_seed_blocks([(resolved_out_dir, int(base_seed), int(games))] + others)
    except ValueError as error:
        return _fail(
            "seed_ledger",
            f"seed range collides with an existing claim in {claims_root}: {error}",
        )
    return _ok(
        "seed_ledger",
        f"[{base_seed}, {int(base_seed) + int(games)}) is disjoint from {len(others)} "
        f"existing claim(s) in {claims_root} and outside the VAL-ONLY range",
    )


# ---------------------------------------------------------------------------
# (c) masked-checkpoint regime guard
# ---------------------------------------------------------------------------


def read_checkpoint_mask_hidden_info(checkpoint_path: str | Path) -> bool:
    """Read the `mask_hidden_info` metadata flag directly out of a checkpoint
    file, without reconstructing the full EntityGraphPolicy (config/model
    state), so this guard is cheap enough to run before every gate.
    """
    import torch

    data = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    return bool(data.get("mask_hidden_info", False))


def guard_masked_regime(checkpoint_path: str | Path, expected_masked: bool) -> GuardResult:
    """FAIL if a checkpoint's recorded training regime (`mask_hidden_info`)
    doesn't match what this launch's eval/gen harness is about to request.
    Mirrors `EntityGraphRustEvaluator`'s runtime
    `_assert_public_observation_matches_checkpoint_training` check (task
    #76), but callable before a launch even constructs an evaluator.
    """
    try:
        actual = read_checkpoint_mask_hidden_info(checkpoint_path)
    except Exception as error:  # noqa: BLE001 -- any load failure is a hard guard failure
        return _fail(
            "masked_regime",
            f"could not read mask_hidden_info metadata from checkpoint {checkpoint_path}: {error}",
        )
    if actual != bool(expected_masked):
        return _fail(
            "masked_regime",
            f"checkpoint {checkpoint_path} recorded mask_hidden_info={actual} but this launch "
            f"expects mask_hidden_info={bool(expected_masked)}; masked nets MUST eval masked on "
            "master code (task #76 safety net) -- running mismatched silently regenerates the "
            "#71 hidden-info leak, or feeds an omniscient-trained net inputs it never learned to use.",
        )
    return _ok(
        "masked_regime",
        f"checkpoint mask_hidden_info={actual} matches this launch's expected {bool(expected_masked)}",
    )


# ---------------------------------------------------------------------------
# (d) doc-vs-artifact provenance drift detector
# ---------------------------------------------------------------------------


def guard_provenance(report_path: str | Path, claims: Mapping[str, Any]) -> GuardResult:
    """Cross-check a set of claims (e.g. extracted from a doc or chronicle
    entry) against the actual provenance fields recorded in a training/gate
    `report.json`. FAILs on any claimed field that's either missing from the
    artifact or whose recorded value disagrees with the claim.
    """
    path = Path(report_path)
    if not path.exists():
        return _fail("provenance", f"report artifact not found: {path}")
    try:
        report = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        return _fail("provenance", f"report artifact {path} is not valid JSON: {error}")

    mismatches: list[str] = []
    for key, expected in claims.items():
        if key not in report:
            mismatches.append(f"{key}: claimed {expected!r} but {path} has no such field")
            continue
        actual = report[key]
        if actual != expected:
            mismatches.append(f"{key}: claimed {expected!r} but {path} records {actual!r}")
    if mismatches:
        return _fail(
            "provenance",
            "doc claim(s) do not match artifact provenance -- do not ship this claim in a "
            "report until reconciled: " + "; ".join(mismatches),
            mismatches=mismatches,
        )
    return _ok("provenance", f"all {len(claims)} claim(s) match {path}")


# ---------------------------------------------------------------------------
# (e) fd limit
# ---------------------------------------------------------------------------


def guard_fd_limit(minimum: int = 65536) -> GuardResult:
    """Host-only: the round-12 `Errno 24` crash was a direct fd-exhaustion
    failure. Both generation and training processes must run with a raised
    `RLIMIT_NOFILE` soft limit, baked into the launch script rather than left
    to operator memory.
    """
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < minimum:
        return _fail(
            "fd_limit",
            f"soft RLIMIT_NOFILE={soft} < required {minimum} (hard limit={hard}); "
            f"run `ulimit -n {minimum}` before this launch (round-12 Errno 24 fd-exhaustion crash).",
            host_only=True,
            soft=soft,
            hard=hard,
        )
    return _ok(
        "fd_limit",
        f"soft RLIMIT_NOFILE={soft} >= required {minimum}",
        host_only=True,
        soft=soft,
        hard=hard,
    )


# ---------------------------------------------------------------------------
# (f) orphaned multiprocessing children -- explicit-PID-only kill
# ---------------------------------------------------------------------------

_NUMERIC_TOKEN_RE = re.compile(r"^-?\d+$")


def guard_explicit_pid_kill(command: Sequence[str]) -> GuardResult:
    """FAIL any kill command that targets processes by broad pattern
    (`pkill`, or a `kill $(pgrep ...)` command-substitution) instead of an
    explicit numeric PID list -- a pattern-based kill can hit unrelated
    processes sharing the same match.
    """
    tokens = list(command)
    if not tokens:
        return _fail("explicit_pid_kill", "empty kill command")
    verb = tokens[0]
    if any("$(" in tok or "`" in tok for tok in tokens):
        return _fail(
            "explicit_pid_kill",
            f"{tokens!r} embeds a command substitution instead of an explicit PID; "
            "resolve the PID first and pass it literally.",
        )
    if verb == "pkill":
        return _fail(
            "explicit_pid_kill",
            f"{tokens!r} uses pkill (pattern-based kill); kill by explicit PID only, "
            "never a broad pattern-based kill that can hit unrelated processes.",
        )
    if verb != "kill":
        return _fail("explicit_pid_kill", f"unrecognized kill command verb {verb!r}; expected 'kill'")
    targets = [tok for tok in tokens[1:] if not tok.startswith("-")]
    if not targets:
        return _fail("explicit_pid_kill", f"{tokens!r} has no PID argument")
    if not all(_NUMERIC_TOKEN_RE.match(tok) for tok in targets):
        return _fail(
            "explicit_pid_kill",
            f"{tokens!r} targets a non-numeric token; kill must take explicit numeric PID(s) only.",
        )
    return _ok("explicit_pid_kill", f"kill targets explicit PID(s): {targets}")


# ---------------------------------------------------------------------------
# (g) pgrep self-match trap + fcntl lock hygiene
# ---------------------------------------------------------------------------


def guard_pgrep_self_match(pattern: str) -> GuardResult:
    """FAIL if a `pgrep -f`/`pgrep -af`-style regex `pattern` would match the
    ps listing line produced by pgrep's OWN invocation (the self-match
    trap). The standard fix is the bracket trick (e.g. `[t]ools/foo.py`
    instead of `tools/foo.py`), which this guard verifies actually works for
    a given pattern.
    """
    self_line = f"pgrep -af {pattern}"
    try:
        matched = re.search(pattern, self_line)
    except re.error as error:
        return _fail("pgrep_self_match", f"pattern {pattern!r} is not a valid regex: {error}")
    if matched:
        return _fail(
            "pgrep_self_match",
            f"pattern {pattern!r} matches its own pgrep invocation ({self_line!r}); use the "
            "bracket trick (e.g. '[t]ools/foo.py') or a post-filter so pgrep's own argv can't "
            "match its own search pattern.",
        )
    return _ok("pgrep_self_match", f"pattern {pattern!r} does not match its own pgrep invocation")


def guard_fcntl_lock_present(source_text: str, *, source_name: str = "<script>") -> GuardResult:
    """Static check that a liveness/singleton script actually uses an
    fcntl-based lock rather than relying solely on pgrep/PID pattern
    matching for liveness, which is subject to the self-match trap.
    """
    if "fcntl.flock" in source_text or re.search(r"\bflock\s*\(", source_text):
        return _ok("fcntl_lock_present", f"{source_name} uses an fcntl lock for liveness/singleton checks")
    return _fail(
        "fcntl_lock_present",
        f"{source_name} has no fcntl.flock-based lock; relying on pgrep/PID liveness checks "
        "alone is subject to the pgrep self-match trap -- add an exclusive non-blocking flock.",
    )


def guard_lock_available(lock_path: str | Path) -> GuardResult:
    """Host-only: test-then-release an exclusive non-blocking fcntl lock at
    `lock_path`, providing the same process-level exclusion as the production
    loop coordinator's lock.
    mechanism, to confirm no other live process already holds it before
    this launch proceeds to acquire it for real.
    """
    import fcntl

    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return _fail(
            "lock_available",
            f"{path} is already locked by another live process",
            host_only=True,
        )
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()
    return _ok("lock_available", f"{path} lock is free", host_only=True)


# ---------------------------------------------------------------------------
# (h) dup-seed + wave-root quarantine pre-corpus checks
# ---------------------------------------------------------------------------


def guard_corpus_no_duplicate_seeds(corpus_meta_path: str | Path) -> GuardResult:
    """Wraps build_memmap_corpus.py's `_GameSeedRunTracker` detector (task
    #85): reads an already-built corpus's `corpus_meta.json` and FAILs if
    its `stats.has_duplicate_game_seeds` is set, refusing to admit that
    corpus into the training window.
    """
    path = Path(corpus_meta_path)
    if not path.exists():
        return _fail("corpus_no_duplicate_seeds", f"corpus_meta.json not found at {path}")
    try:
        meta = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        return _fail("corpus_no_duplicate_seeds", f"{path} is not valid JSON: {error}")
    stats = meta.get("stats", {})
    if bool(stats.get("has_duplicate_game_seeds", False)):
        return _fail(
            "corpus_no_duplicate_seeds",
            f"{path} reports {stats.get('duplicate_game_seed_count', '?')} duplicate game_seed "
            "run(s) (task #77 seed-collision class); do not admit this corpus into the training window.",
        )
    return _ok("corpus_no_duplicate_seeds", f"{path} has zero duplicate game_seed runs")


def guard_wave_root_quarantine(
    actual_sources: Sequence[str | Path], declared_sources: Sequence[str | Path]
) -> GuardResult:
    """FAIL if the corpus-build sources actually about to be scanned differ
    from the operator's explicitly declared wave-root set. Catches an
    accidental extra/missing source directory before it silently mixes
    batches into one corpus -- the mixed-batch class that "created absurd
    seed envelopes and false-quarantined 2.9M rows" (research chronicle
    Sec.13.4 Ops).
    """
    actual = sorted(str(Path(item).resolve()) for item in actual_sources)
    declared = sorted(str(Path(item).resolve()) for item in declared_sources)
    if actual != declared:
        extra = sorted(set(actual) - set(declared))
        missing = sorted(set(declared) - set(actual))
        return _fail(
            "wave_root_quarantine",
            f"corpus source list does not match the operator-declared wave-root set "
            f"(extra={extra}, missing={missing}); one wave-root per ingest batch, not a "
            "silently mixed set.",
            extra=extra,
            missing=missing,
        )
    return _ok("wave_root_quarantine", f"{len(actual)} source(s) match the declared wave-root set exactly")


# ---------------------------------------------------------------------------
# (b-cross-host) SEED LEDGER OVERLAP -- cross-host claimed-range refusal
# ---------------------------------------------------------------------------

# guard_seed_ledger (above) is PER-HOST ONLY: it globs *this* host's
# .seed_claims/ directory. The cross-host source of truth is
# runs/SEED_LEDGER.md, a hand-maintained markdown table every host/Modal
# launch is supposed to consult before claiming a base seed. Nothing in code
# ever read it -- the 60M near-collision (task #77) and the silently-poisoned
# staged gen-1 generation (7/8 GPU-idx seed overlap) both slipped past the
# per-host guard for exactly this reason. This guard parses that ledger and
# refuses any launch whose requested [base_seed, base_seed+games) range
# overlaps a claimed row. It fails CLOSED when the ledger cannot be found:
# a launch that cannot be checked against the cross-host ledger is refused
# (operator syncs the ledger, sets $CATAN_SEED_LEDGER, or uses --skip-guards).

# Row grammar tolerates both the en-dash (U+2013) and plain-hyphen variants
# that coexist in the file, plus open-ended "[N+)" rows (no dash).
_LEDGER_ROW_RE = re.compile(
    r"^\s*\[\s*([\d,]+)\s*(?:[\u2013\-]\s*([\d,]+)|(\+))\s*\)\s*\|\s*(.*)$"
)
_LEDGER_OPEN_END_SENTINEL = 1 << 62  # "[N+)" open-ended rows -> [N, +inf)


def parse_seed_ledger(ledger_path: str | Path) -> list[tuple[int, int, str]]:
    """Parse runs/SEED_LEDGER.md into (start, end, label) claimed ranges.

    Handles en-dash and plain-hyphen row variants, comma-separated integers,
    and open-ended "[N+)" rows (end -> a large sentinel). Non-matching lines
    (the `#` header, the `NEXT SAFE:` prose line, blanks) are skipped, and a
    malformed row is skipped rather than raising -- one bad line must not make
    the whole guard crash-fail and block every launch.
    """
    rows: list[tuple[int, int, str]] = []
    for raw in Path(ledger_path).read_text().splitlines():
        match = _LEDGER_ROW_RE.match(raw)
        if not match:
            continue
        start_s, end_s, open_end, label = match.group(1), match.group(2), match.group(3), match.group(4)
        try:
            start = int(start_s.replace(",", ""))
            if open_end is not None or end_s is None:
                end = _LEDGER_OPEN_END_SENTINEL
            else:
                end = int(end_s.replace(",", ""))
        except ValueError:
            continue
        if end <= start:
            continue
        rows.append((start, end, label.strip()))
    return rows


def _resolve_seed_ledger_path(ledger_path: str | Path | None) -> Path:
    """Resolve the ledger path: explicit arg > $CATAN_SEED_LEDGER > repo-local
    runs/SEED_LEDGER.md > ~/catan-zero/runs/SEED_LEDGER.md (the canonical tree)."""
    if ledger_path is not None:
        return Path(ledger_path)
    env = os.environ.get("CATAN_SEED_LEDGER")
    if env:
        return Path(env)
    repo_local = Path(__file__).resolve().parent.parent / "runs" / "SEED_LEDGER.md"
    if repo_local.exists():
        return repo_local
    return Path.home() / "catan-zero" / "runs" / "SEED_LEDGER.md"


def guard_ledger_overlap(
    base_seed: int,
    games: int,
    *,
    ledger_path: str | Path | None = None,
    purpose: str = "generate",
    own_claim_label: str | None = None,
) -> GuardResult:
    """Refuse any launch whose [base_seed, base_seed+games) range overlaps a
    range already claimed in the cross-host SEED_LEDGER.md. Fails CLOSED when
    the ledger is missing (a launch that cannot be verified is refused), EXCEPT
    for seed ranges that live entirely inside the VAL-ONLY band [6.19B, 6.2B),
    which are validation-local and do not need cross-host ledger coverage.

    CAT-124: the canonical launch order is claim-then-verify (append the claim
    row to the ledger, then run guards), so by guard time this launch's OWN
    range is already in the ledger and would otherwise self-collide (the trap
    that forced --skip-guards on generation). Pass ``own_claim_label`` (a unique
    claim id) and only a ledger row carrying the exact whitespace-delimited
    ``claim=<id>`` token is treated as US, not a peer, and excluded from
    collision.  Substring matching is deliberately forbidden: claim ``abc``
    must not exempt a peer carrying ``claim=abc-extra``.
    Peers still collide; the guard stays fail-closed on a genuinely overlapping
    OTHER claim."""
    requested = (int(base_seed), int(base_seed) + int(games))
    if requested[1] <= requested[0]:
        return _ok("ledger_overlap", f"empty seed range [{requested[0]}, {requested[1]}); nothing to claim")
    resolved = _resolve_seed_ledger_path(ledger_path)
    if not resolved.exists():
        if VAL_ONLY_SEED_RANGE[0] <= requested[0] and requested[1] <= VAL_ONLY_SEED_RANGE[1]:
            return _ok(
                "ledger_overlap",
                f"[{requested[0]}, {requested[1]}) is inside the VAL-ONLY band; cross-host "
                "ledger check is not required and the canonical ledger is missing.",
            )
        return _fail(
            "ledger_overlap",
            f"canonical seed ledger not found at {resolved} (tried explicit arg, "
            "$CATAN_SEED_LEDGER, repo-local runs/SEED_LEDGER.md, ~/catan-zero). Cross-host "
            "seed collision cannot be checked; sync the ledger, set $CATAN_SEED_LEDGER, or "
            "pass --skip-guards once you have verified the range by hand.",
        )
    try:
        rows = parse_seed_ledger(resolved)
    except OSError as error:
        return _fail("ledger_overlap", f"could not read seed ledger {resolved}: {error}")
    own_claim_token = f"claim={own_claim_label}" if own_claim_label is not None else None
    collisions = [
        (start, end, label)
        for (start, end, label) in rows
        if _ranges_overlap(requested, (start, end))
        and not (own_claim_token is not None and own_claim_token in label.split())
    ]
    if collisions:
        rendered = "; ".join(
            f"[{s}, {'+inf' if e == _LEDGER_OPEN_END_SENTINEL else e}) {label!r}"
            for (s, e, label) in collisions
        )
        return _fail(
            "ledger_overlap",
            f"requested {purpose} seed range [{requested[0]}, {requested[1]}) overlaps "
            f"{len(collisions)} claimed range(s) in {resolved}: {rendered}. Choose a disjoint "
            "range (see the ledger's NEXT SAFE line) and append your claim before launching.",
        )
    return _ok(
        "ledger_overlap",
        f"[{requested[0]}, {requested[1]}) is disjoint from all {len(rows)} claimed range(s) in {resolved}",
    )


# ---------------------------------------------------------------------------
# Registry + check-runner CLI
# ---------------------------------------------------------------------------

GUARD_REGISTRY: dict[str, Callable[..., GuardResult]] = {
    "cli_flag_lint": guard_cli_flag_lint,
    "val_only_never_trains": guard_val_only_never_trains,
    "seed_ledger": guard_seed_ledger,
    "ledger_overlap": guard_ledger_overlap,
    "masked_regime": guard_masked_regime,
    "provenance": guard_provenance,
    "fd_limit": guard_fd_limit,
    "explicit_pid_kill": guard_explicit_pid_kill,
    "pgrep_self_match": guard_pgrep_self_match,
    "fcntl_lock_present": guard_fcntl_lock_present,
    "lock_available": guard_lock_available,
    "corpus_no_duplicate_seeds": guard_corpus_no_duplicate_seeds,
    "wave_root_quarantine": guard_wave_root_quarantine,
}

# Guards that need live host state rather than being pure functions of their
# arguments -- their result can legitimately differ between CI and the real
# launch host. Kept in sync with each guard's own `host_only=True` results.
HOST_ONLY_GUARDS: frozenset[str] = frozenset({"fd_limit", "lock_available"})


def run_guards(guard_specs: Sequence[Mapping[str, Any]]) -> list[GuardResult]:
    """Run a list of `{"name": <guard_registry_key>, "args": {...}}` specs
    and return one `GuardResult` per spec, in order. A guard that raises
    unexpectedly (bad path, malformed args) is caught and reported as a
    FAIL rather than crashing the whole run, so one broken spec doesn't
    hide the results of every other guard in the batch.
    """
    results: list[GuardResult] = []
    for spec in guard_specs:
        name = spec["name"]
        kwargs = spec.get("args", {})
        func = GUARD_REGISTRY.get(name)
        if func is None:
            results.append(_fail(name, f"unknown guard {name!r}; known guards: {sorted(GUARD_REGISTRY)}"))
            continue
        try:
            results.append(func(**kwargs))
        except Exception as error:  # noqa: BLE001 -- a broken guard spec must not hide the rest
            results.append(_fail(name, f"guard raised {type(error).__name__}: {error}"))
    return results


def _print_results(results: Sequence[GuardResult]) -> int:
    exit_code = 0
    for result in results:
        marker = "host-only, " if result.host_only else ""
        print(f"[{result.status}] ({marker}{result.guard}) {result.reason}")
        if not result.passed:
            exit_code = 1
    return exit_code


def _cmd_run(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.config).read_text())
    guard_specs = payload["guards"] if isinstance(payload, dict) else payload
    results = run_guards(guard_specs)
    return _print_results(results)


def _cmd_list(_args: argparse.Namespace) -> int:
    for name in sorted(GUARD_REGISTRY):
        tag = " (host-only)" if name in HOST_ONLY_GUARDS else ""
        print(f"{name}{tag}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help=(
            "run every guard listed in a JSON config ({\"guards\": [{\"name\": ..., \"args\": "
            "{...}}, ...]}); exits nonzero if any guard FAILs, including host-only guards "
            "unless the config simply omits them for a non-matching host"
        ),
    )
    run_parser.add_argument("--config", required=True, help="path to a JSON guard-spec list/config")
    run_parser.set_defaults(func=_cmd_run)

    list_parser = subparsers.add_parser("list", help="list every registered guard name")
    list_parser.set_defaults(func=_cmd_list)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
