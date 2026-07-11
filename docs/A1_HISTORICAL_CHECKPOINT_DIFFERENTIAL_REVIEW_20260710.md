# A1 Historical Checkpoint Differential Review

## Executive Summary

| Severity | Count |
|---|---:|
| Critical | 0 |
| High | 0 |
| Medium | 0 |
| Low | 0 |

**Overall risk:** Low after the review fix.  
**Recommendation:** Approve `9baa6ee` with the broken-intermediate-symlink fix in the immediately following commit.

The centralized resolver now gives the artifact builder and promotion consumer one fail-closed interpretation of historical checkpoint paths. One blocker was found during review: a broken intermediate symlink at a nearer report ancestor was skipped when a valid checkpoint existed at a higher ancestor. The follow-up fix checks every ancestor candidate lexically before testing final-file existence.

## What Changed

**Reviewed commit:** `9baa6ee` (`f531fd2..9baa6ee`)  
**Original scope:** 160 insertions and 48 deletions across four files.

| File | Risk | Review result |
|---|---|---|
| `tools/a1_promotion_transaction.py` | High | Shared resolver and downstream verification reviewed line by line; one blocker fixed. |
| `tools/a1_promotion_artifacts.py` | High | Delegates to the transaction resolver and translates the typed error; no semantic fork remains. |
| `tests/test_a1_promotion_transaction.py` | Low | Covers builder-to-promotion flow for the real gen3 path from an unrelated cwd. |
| `tests/test_a1_promotion_artifacts.py` | Low | Covers absolute, relative, traversal, basename, ambiguity, and symlink behavior. |

## Findings

No unresolved findings remain.

The review found and fixed a fail-closed gap at `tools/a1_promotion_transaction.py:230-241`. Before the follow-up, `candidate.exists()` returned false when an intermediate path component was a broken symlink, so that candidate was ignored. A higher-ancestor checkpoint could then be accepted. The resolver now compares each lexical candidate with `resolve(strict=False)` before checking existence and rejects any symlinked component. The regression matrix at `tests/test_a1_promotion_artifacts.py:645-671` includes final, live-intermediate, and broken-intermediate symlinks. The shared canonical-file helper also translates `pathlib`'s `RuntimeError` for an absolute symlink loop into the resolver's typed `PromotionError`, allowing the builder to preserve its `ArtifactBuildError` boundary.

Attacker model: an evidence producer able to choose the historical report checkpoint string and arrange files below a report ancestor. The fixed behavior prevents that producer from hiding a symlinked candidate behind a broken target while relying on a higher valid checkpoint.

## Test Coverage Analysis

The changed resolver behavior is covered for:

- The real `runs/bc/gen3_20260706/checkpoint.pt` path.
- Builder-to-promotion acceptance from a different current working directory.
- Exact canonical absolute paths.
- Traversal and bare/basename-only relative paths.
- Zero and multiple ancestor matches.
- Final, live intermediate, broken intermediate, and absolute looping symlinks.
- A mismatched checkpoint and immutable historical-report hash binding.

Verification evidence:

```text
pytest -q tests/test_a1_promotion_artifacts.py tests/test_a1_promotion_transaction.py
180 passed in 3.68s

ruff check tools/a1_promotion_transaction.py tools/a1_promotion_artifacts.py \
  tests/test_a1_promotion_transaction.py tests/test_a1_promotion_artifacts.py
All checks passed!
```

Byte-compilation, `git diff --check`, and an additional adversarial resolver matrix also passed.

## Blast Radius Analysis

The private resolver has two production call sites: the artifact builder and the promotion transaction's legacy calibration verifier. The downstream verifier is reached only for champion mechanism-calibration evidence with the explicitly permitted legacy provenance bridge. Blast radius is low in caller count but high in validation sensitivity.

## Historical Context

The downstream legacy bridge check originated in `50a68d04`. It previously resolved relative paths only against `report_path.parent`, creating semantic drift from the builder's later ancestor-based handling. `9baa6ee` removes that drift by centralizing resolution in the transaction module and making the builder delegate to it.

## Recommendations

- Merge the follow-up broken-symlink fix with `9baa6ee` before using the gen3 legacy bridge.
- Keep future historical-path changes in the shared resolver and retain the builder-to-promotion test as the compatibility contract.

## Analysis Methodology

**Strategy:** Surgical, high-risk validation review.  
**Coverage:** All four changed files, both production call sites, the prior implementation and blame context, and focused one-hop consumers.  
**Techniques:** Line-by-line data-flow analysis, history review, caller search, adversarial filesystem modeling, direct edge-case probes, and full focused-module tests.  
**Limitations:** The review did not run the entire repository test suite or exercise platform-specific Windows path semantics; the production environment and declared report path are POSIX.  
**Confidence:** High for the reviewed resolver and builder-to-promotion flow.
