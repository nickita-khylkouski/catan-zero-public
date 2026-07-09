# CAT-52 audit: is game-level validation splitting the only reachable path?

Date: 2026-07-08. Scope: `tools/train_bc.py` and the memmap corpus loader
(`tools/build_memmap_corpus.py`), which together are the entire train/validation
split surface in this codebase (verified by exhaustive grep for `split`,
`train_test_split`, `random_split`, `holdout` across `tools/` and `src/` —
`split_train_validation_indices` in `train_bc.py` is the only place a
train/validation partition is constructed anywhere in the training pipeline;
`build_memmap_corpus.py` concatenates shards and does not split at all).

## Verdict

**`split_train_validation_indices` (`tools/train_bc.py:5082`) is the only
split path, and it is game-level in both live production paths
(`--validation-game-seed-ranges` explicit, and the `--validation-fraction`
default). It had one practically-live silent corner (a degenerate/all-zero
`game_seed`, silently row-level for corpora under 1000 rows) and one
theoretical-but-worth-guarding corner (a `game_seed` column missing
entirely, only reachable via a direct, non-normalized caller, not through
either production loader). Both are now fixed as of this branch.**

## The three paths, traced

`split_train_validation_indices` is called from exactly one place,
`tools/train_bc.py:930`, once per training run, before the epoch loop. There
is no other split call site, no default-on-omission random re-split, and no
per-round re-splitting of a persistent window anywhere in the current code
(the round-11 mechanism — "per-round random re-splitting of a persistent
window leaked ~95% of validation rows", `docs/plans/CATAN_ZERO_RESEARCH_CHRONICLE.md:263,376`
— has no live equivalent; grepping the whole tree for any re-split-per-round
pattern in `continuous_flywheel.py`/`train_bc.py` turns up nothing).

### Path 1: `--validation-game-seed-ranges` (explicit, deterministic)

`args.validation_game_seed_ranges` defaults to `""` (empty, i.e. NOT the
active path unless the caller passes it explicitly). When non-empty, every
row whose `game_seed` falls in any of the given inclusive ranges becomes
validation; everything else is train. ✓ VERIFIED game-level — it operates on
whole `game_seed` values, so every row of a held-out game is held out
together, never split across train/validation. Existing tests
(`tests/test_train_bc_validation_game_seed_ranges.py`, pre-existing) cover
exact-match, no-leakage, multi-block, and max-samples-cap behavior for this
path.

**Found and fixed (pre-existing bug, both paths, but NOT reachable via either
production loader today — see Path 3 below for the corrected reachability
analysis):** `seeds = np.asarray(data.get("game_seed", np.arange(n, ...)))`
— if the corpus dict has no `"game_seed"` key at all, this silently
substitutes each row's own index as its "seed". Since every row then has a
unique, distinct fake seed, this is *not* caught by the range membership
check; it just silently produces a nonsensical, effectively row-level
selection with no warning. This is now a hard `SystemExit` by default (see
"the fix" below).

### Path 2: `--validation-fraction` (random, grouped by game_seed) — the default

`args.validation_fraction` defaults to `0.05` (**non-zero — this is the live
default path** whenever `--validation-game-seed-ranges` is omitted, which is
the common case for non-value-head-repair runs). When the corpus has more
than one unique `game_seed` value, this groups rows by `game_seed`, shuffles
the *unique seed* list with `np.random.default_rng(validation_seed)`, and
accumulates whole games into the validation set until the target row
fraction is reached. ✓ VERIFIED game-level: the mask is built from
`np.isin(seeds, selected)`, i.e. whole-game membership, never a per-row draw.

### Path 3 (the dangerous corner): degenerate/missing `game_seed`

Two sub-cases of the fallback, both keyed off the same
`data.get("game_seed", np.arange(n, dtype=np.int64))` default. **Correction
after tracing an actual run (not inferring from the code alone), per this
project's claim-verification discipline:** my first pass over this code
assumed sub-case (1) below was live-reachable at any corpus size through the
normal training path. Tracing `load_teacher_data` (line 2857) and
`load_teacher_data_memmap`/`MemmapCorpus` (the only two producers of the
`data` dict `split_train_validation_indices` is called with in production,
`tools/train_bc.py:930`) shows both route every shard through
`_normalize_teacher_shard`, which **unconditionally synthesizes a
`"game_seed"` column** via `_field_or_default(shard, "game_seed",
np.zeros(n, dtype=np.int64), ...)` (`tools/train_bc.py:3244`) when the raw
shard lacks one. `build_memmap_corpus.py`'s `columns` list is derived from
this already-normalized first shard, so a built corpus's `game_seed_present`
stat (`tools/build_memmap_corpus.py:452`) is, in the current code, always
`True` — a raw shard genuinely missing `game_seed` does not produce a
missing column, it produces an all-zero one, which is sub-case (2), not (1).

1. **`game_seed` column absent from `data` entirely.** NOT reachable via
   `load_teacher_data`/`load_teacher_data_memmap` today, for the reason
   above. It IS reachable if `split_train_validation_indices` is called
   directly with a hand-built dict that skips normalization — which is a
   real calling pattern already exercised by this repo's own test suite
   (`tests/test_train_bc_validation_game_seed_ranges.py`'s `_make_data`
   helper builds exactly such a dict), and is plausible for any future
   lightweight caller (e.g. a calibration probe assembling its own dict, in
   the spirit of `tools/value_repair_calibration_probe.py`) that forgets to
   carry `game_seed` through. Worth guarding as defense-in-depth even though
   it is not live through today's two production loaders: if it ever
   silently defaulted to `np.arange(n)`, every row would get a distinct fake
   seed (`unique_seeds.size == n > 1`), which takes the "group by game_seed"
   branch but, since every group has exactly one row, degenerates to a plain
   row-level shuffle that **bypasses even the existing `n >= 1000` guard**
   (that guard only fires when `unique_seeds.size <= 1`, which this case
   never hits) — the exact round-11 mechanism, and at any corpus size.

2. **`game_seed` present but degenerate** (`unique_seeds.size <= 1`, i.e. one
   game, or every row zero-filled because the raw source never had
   `game_seed` at all — **this is the practically-live version of the
   round-11 hazard**, not sub-case 1): guarded by a `SystemExit` for
   `n >= 1000`, but for `n < 1000` it silently executes a genuine
   `rng.permutation(all_indices)` row-level split with **zero warning**. This
   corner is intentionally tolerated for tiny synthetic/smoke corpora (per
   the existing code comment: "requires non-degenerate game_seed values for
   *large* teacher datasets"), but was indistinguishable from a safe split
   from the outside.

## The fix (this branch)

In `tools/train_bc.py`:

- `split_train_validation_indices` now takes `allow_missing_game_seed: bool = False`
  and raises `SystemExit` immediately if `"game_seed" not in data` and the
  caller hasn't explicitly opted in — closing sub-case (1) in **both** the
  explicit-ranges and fraction-based branches (the check runs before either
  branch). Wired to a new CLI flag, `--allow-missing-game-seed-validation-split`
  (default off), recorded in `report.json` for provenance.
- Sub-case (2) (`n < 1000`, single/degenerate `game_seed`) now prints a loud
  `WARNING: ... ROW-LEVEL ...` to stderr before executing, instead of running
  silently. Not hard-refused, since it's a real, load-bearing path for tiny
  test corpora, but no longer invisible.

## Regression tests (`tests/test_train_bc_validation_game_seed_ranges.py`)

Added, all passing (`.venv/bin/python -m pytest tests/test_train_bc_validation_game_seed_ranges.py -v`, 15/15):

- `test_missing_game_seed_column_raises_by_default` — proves sub-case (1) is
  now refused via the fraction-based (default) path.
- `test_missing_game_seed_column_raises_even_with_explicit_ranges` — same,
  via the explicit-ranges path (the "safe" path had the identical bug).
- `test_missing_game_seed_allowed_via_explicit_opt_in` — the opt-out still
  functions when explicitly requested (not a silent no-op).
- `test_degenerate_single_game_seed_small_corpus_warns_on_stderr` — exercises
  sub-case (2) directly (per the ticket's instruction to test the guarded
  path, not just assume it's absent) and asserts the warning fires.
- `test_degenerate_single_game_seed_large_corpus_still_raises` — regression
  guard that the pre-existing `n >= 1000` raise is unchanged.
- `test_synthetic_overlapping_windows_game_seed_ranges_yield_zero_row_overlap`
  — the ticket's prescribed smoke test: two synthetic "windows" with a
  deliberately overlapping game-seed band; splitting on that band via
  `--validation-game-seed-ranges` yields disjoint train/validation game-seed
  sets.

## Does not need re-auditing before future value-head experiments

This file is the audit artifact the ticket asks for. As long as
`split_train_validation_indices` remains the sole split call site (verify with
`grep -rn "split_train_validation_indices\|train_test_split\|random_split" tools/ src/`
if this changes) and its default posture (refuse on missing `game_seed`, warn
on degenerate-small) is unchanged, no further audit is required before
trusting a value-head tournament's validation metrics.
