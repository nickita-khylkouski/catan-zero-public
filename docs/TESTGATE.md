# CAT-129 — explicit local/cluster acceptance

One command turns the v1.0-freeze clean run into a **repeatable** acceptance
check. It is intentionally invoked by an operator rather than by hosted CI. This is
the mandatory **pre-deploy** check, the **post-deploy** check (CAT-130) run on
each box, and the **re-gate** consolidator runs on every Wave-2 merge.

## Run it

```bash
make test          # or: bash scripts/gate.sh
```

Exit 0 **iff all four stages pass**:

| stage | what | how |
|---|---|---|
| `noop` | champion_v0 forward is **BIT-IDENTICAL** with the new heads' flags OFF (whole-model, max_diff 0.0 over 64 rows; 16-row = first 16) | `scripts/check_champion_noop.py` vs a committed fixture+reference |
| `parity` | featurizer **parity 19/19** (rust_featurize + action_context + symmetry) | 3 pytest files |
| `goldens` | **CAT-75 CLI** option-string goldens + CLI-drift/default guards | 4 pytest files |
| `suite` | **full pytest suite** green — GPU tests self-skip, CAT-94 + modal collection-quarantined (this stage also contains parity+goldens; the explicit stages give fast standalone PASS lines) | `pytest tests/` |

Cheap stages run first, the ~17-min full suite last, so a regression fails fast.

## CPU-first / off-GPU

Runs green on any box **without a GPU** (validated: 1798 passed / 8 skipped / 0
failed, `CUDA_VISIBLE_DEVICES=""`). GPU-requiring tests self-skip. Override with
`GATE_ALLOW_GPU=1` to let them run on a GPU box (still self-skip if CUDA absent).

## Quarantine (why it's 0-error, not "0-fail-N-error")

`conftest.py` `collect_ignore`s two modules that ERROR at collection for reasons
unrelated to the code under test (documented in that file):
- `tests/test_concat_memmap_corpus.py` — 6 errors; `ConcatMemmapCorpus` is CAT-94
  window-feed work absent from this tree. Unquarantine when CAT-94 lands.
- `tests/test_modal_gumbel_factory_legacy_guard.py` — `modal.Image` API drift;
  `modal` is an optional cloud dep. Unquarantine when modal is pinned / the
  module-level modal call is made lazy.

## Config

- `PY=<python>` — interpreter (default `$REPO/.venv/bin/python`, the layout
  `tools/install_v1_freeze.sh` produces). E.g. `PY=~/cz-e2e/.venv/bin/python make test`.
- `REPO=<root>` — repo root (default: auto).

## No-op reference — rebanking

The no-op fixture + reference are committed under `tests/fixtures/`
(`noop_input_64.npz`, `noop_ref.npz`) and were banked on the **v1.0-freeze** tag
(the certified-good state). The forward is CPU-deterministic so it reproduces on
any box. If a DELIBERATE, reviewed change to the champion forward lands, rebank:

```bash
PY=… python scripts/check_champion_noop.py --bank --champion <champion_v0.pt>
```

(Banked here as value_sha1 `cb932795` / logits_sha1 `26b043eb` — a self-consistent
CPU construction; distinct from the consolidator's `aba0012a`/`a6bba3f3`, which
use a different input/device. Both assert the same property: flags-OFF forward is
unchanged. See the note to consolidator if a single canonical hash is desired.)
