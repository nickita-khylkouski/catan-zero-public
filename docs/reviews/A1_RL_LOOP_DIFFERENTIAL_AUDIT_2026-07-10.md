# A1 RL loop differential audit — 2026-07-10

## Executive summary

| Severity | Count |
|---|---:|
| Critical | 3 |
| High | 3 |
| Medium | 2 |

**Overall risk:** high. **Recommendation:** continue read-only evaluation work,
but do not describe the current tree as a reproducible train-to-promote loop and
do not mutate the champion registry yet.

The sealed one-B200 learner dose is correct and complete. The live retry receipt
is `a1-one-dose-training-receipt-v4`, candidate SHA-256 is
`f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4`,
and it records 680 optimizer steps over 2,781,407 training rows. The remaining
risk is not learner throughput; it is the inability to turn the real candidate
and incumbent into all evidence required by promotion without bypasses or
hand-authored substitutes.

## Findings

### Critical: the orchestrator cannot adopt the completed retry-v4 dose

`tools/a1_iteration_orchestrator.py:465-520` accepts only
`one_dose.RECEIPT_SCHEMA` (v3), compares the receipt command to the original dry
plan, derives only the contract-keyed direct claim, and loads that claim without
a retry identity. `initialize()` also refuses the already-consumed parent
contract. The completed production candidate is a v4 derived retry with claim
identity `sha256:a44c5056...`; no iteration state exists on the B200 host.

Impact: the authoritative state machine cannot advance the real candidate to
`dose_complete`, so `verify-evaluation` and `promote` cannot be reached through
the purported durable loop. Direct promotion accepts v4, but that bypasses the
orchestrator the new code says should be authoritative.

Required fix: add an explicit, fail-closed retry adoption transition that binds
the parent failed claim, retry contract, v4 plan/receipt, derived claim identity,
fresh outputs, and exact execution binding. Do not coerce v4 into v3.

Required tests:

- adopt a valid v4 receipt and derived claim after a v3 zero-step failure;
- reject v4 with retry-contract, identity, parent, command, or output drift;
- crash after receipt publication and resume without rerunning training;
- retain the existing direct-v3 path unchanged.

### Critical: incumbent mechanism calibration is impossible under the verifier

`tools/a1_promotion_transaction.py:679-740` requires positive
`readout_provenance.optimizer_steps` and `completed_epochs` for both calibration
roles. The real gen-3 incumbent checkpoint has no `value_training` or
`trained_value_readouts` fields, although its historical report records 912
steps and one epoch. `phase_sliced_value_calibration.py` therefore emits null
step/epoch provenance for that legacy scalar checkpoint, and promotion rejects
it before comparing RMSE.

Impact: no valid mechanism-calibration envelope can be produced for the actual
incumbent.

Required fix: define a typed legacy-incumbent provenance bridge whose authority
comes from the sealed contract's producer hash plus the immutable historical
training report, or make the verifier role-aware and permit legacy scalar
provenance only for the exact contract-bound incumbent. Never relax candidate
provenance.

Required test: the exact contract producer with its bound historical report is
accepted, while any other provenance-free checkpoint or mutated report is
rejected.

### Critical: the external-panel absolute SPRT is both blocking and mutable

`tools/a1_promotion_transaction.py:1038-1042` rejects each candidate/incumbent
panel when its individual verdict is H0. The real policy objective at
`tools/a1_promotion_transaction.py:1329-1362` is comparative non-regression
(candidate win rate no more than 0.02 below incumbent). Historically both nets
can score below 50% versus `catanatron_value`, so both honest individual panels
can be H0 while the candidate is non-regressing.

Conversely, the verifier does not replay fixed external-panel SPRT thresholds.
The CLI accepts explicit `--elo0/--elo1`, so an operator can choose permissive
thresholds to turn the same games into `continue`/H1 and bypass the H0 check.

Impact: honest evidence may be impossible, while threshold manipulation can
make it pass.

Required fix: either remove the individual absolute-SPRT condition and make the
fixed paired-cohort 0.02 differential the sole external tripwire, or bind and
replay an explicitly approved absolute threshold. The former matches the
documented policy.

Required tests:

- candidate 41%, incumbent 42% on the same cohort passes the comparative tripwire
  regardless of both absolute H0 verdicts;
- candidate 39%, incumbent 42% fails;
- CLI threshold overrides cannot change promotion eligibility.

### High: high-regret and bucket-veto source evidence still has no authoritative producer

The promotion consumer at `tools/a1_promotion_transaction.py:1094-1191` checks
only compact summaries. The in-progress `tools/a1_promotion_artifacts.py`
converts a raw high-regret report and bucket counts into those summaries, but it
does not run or replay the underlying matches. No canonical command currently
produces `a1-held-out-high-regret-report-v1` or extracts approved bucket counts
from retained games. The ordinary H2H report lacks phase/opening/blowout labels.

Impact: an operator can only stop, or hand-author the raw inputs. The latter is
not acceptable strength evidence.

Required fix: implement source producers that retain game/state identities,
fixed held-out cohort selection, checkpoint/config hashes, outcomes, and replay
their statistics. Evidence builders should hash and retain those raw sources.

### High: no authoritative champion registry or current pointer is present

Read-only searches on both B200 hosts found no `champion_registry*.json` and no
`CURRENT_CHAMPION`. Promotion intentionally refuses an empty/new registry, and
the every-third counter cannot be reconstructed from a new file without a
separate migration artifact.

Impact: even perfect evaluation evidence cannot be committed.

Required fix: locate and stage the existing authoritative registry/pointer, or
perform a separately audited history migration. Do not silently create an empty
registry.

### High: canonical operator documentation describes incompatible loops

- `RL_AGENT_HANDOFF.md:742-873` still prescribes the legacy n16 gate and
  `gumbel_search_vs_bot_h2h.py`, while the hardened A1 promotion verifier requires
  global n128 and `catanatron_neutral_harness_match.py --mode search`.
- The handoff says the neutral harness is raw-only even though the uncommitted
  tool now implements searched, information-set-safe panels.
- `docs/A1_ONE_DOSE_TRAINING.md:37-54` says no retry is allowed, while the live
  candidate used the new sealed v4 retry.
- `docs/A1_PROMOTION_TRANSACTION.md:26-30,130-132` documents only v3 training
  receipts although promotion accepts v4.
- Older strategy/chronicle documents still present `continuous_flywheel.py` as
  the loop, while current A1 docs say only the one-dose executor is authoritative.

Impact: two competent operators can follow repository documentation and produce
mutually incompatible, non-promotable evidence.

Required fix: one current A1 runbook must name the authoritative entry points,
schemas, exact evaluation semantics, and legacy/dead tools. Historical docs
should be marked historical rather than treated as live instructions.

### Medium: evaluation is not yet a crash-resumable, ledgered transaction

`gumbel_search_cross_net_h2h.py` writes progress but retains the full result only
at the end; it has no per-game resume manifest. A 600-pair n128+D6+PIMC run must
be rerun from the beginning after interruption. Neither H2H tool owns a durable
validation-seed claim, and there is no dedicated VAL-only ledger on the B200
hosts.

Required fix: use the neutral harness's per-game artifact pattern for cross-net
H2H and add an atomic validation-ledger claim transaction before launch.

### Medium: eight-GPU DDP is supported by the trainer but is not this A1 dose

`train_bc.py:9659-9722` correctly partitions memmap indices under DDP. An
8x512 run can preserve nominal global batch 4096. However the sealed contract,
executor, output verifier, and receipt all require `world_size=1`, and the batch
probe showed batch 4096 already saturates one B200 without a material throughput
gain above it. Dropout/RNG and optimizer execution would also make 8-GPU DDP a
new experiment, not a transparent replay.

Decision: keep the completed A1 dose at one B200. If multi-GPU learner training
is desired later, seal it as a new contract and parity-test equal-global-batch
semantics before use.

## Exact post-checkpoint evaluation topology

Do not run calibration on the 8-B200 learner host unless the validation raw NPZ
rows are first staged there. They already exist on the 2-B200 hub, so the data-
local schedule is:

- hub GPU0/GPU1: candidate/incumbent phase-sliced calibration on the identical
  raw shard root and validation manifest (after the incumbent provenance fix);
- 8-B200 GPU1-GPU5: one cross-net n128 internal H2H;
- 8-B200 GPU6/GPU7: candidate/incumbent searched native-Catanatron panels on the
  identical seed cohort (after the external-SPRT fix).

The following is the exact sealed internal search recipe. `<BASE_SEED>` must be
claimed first in a new canonical VAL-only ledger; the source tree must also be
frozen and hashed before launch.

```bash
ROOT=/home/ubuntu/catan-zero-production/gate-worktrees/rl-loop-739cbfb
PY=/home/ubuntu/catan-zero-v1/.venv/bin/python
CANDIDATE=/home/ubuntu/catan-zero-production/runs/learner/a1-infoset-n128-20260710-r2/candidate.pt
INCUMBENT=/home/ubuntu/catan-zero/runs/bc/gen3_20260706/checkpoint.pt
OUT=/home/ubuntu/catan-zero-production/runs/eval/a1-r2/internal-n128
mkdir -p "$OUT"
cd "$ROOT"
PYTHONPATH="$ROOT/src" "$PY" tools/gumbel_search_cross_net_h2h.py \
  --candidate "$CANDIDATE" --baseline "$INCUMBENT" \
  --pairs 600 --base-seed <BASE_SEED> \
  --workers 40 --devices cuda:1,cuda:2,cuda:3,cuda:4,cuda:5 \
  --threads-per-worker 3 \
  --n-full 128 --max-depth 80 --max-decisions 600 \
  --prior-temperature 1.0 --value-scale 1.0 --value-squash tanh \
  --value-readout scalar --c-visit 50.0 --c-scale 0.03 \
  --rescale-noise-floor-c 0.0 --sigma-eval 0.98 \
  --max-root-candidates 16 --max-root-candidates-wide 54 \
  --wide-candidates-threshold 24 \
  --correct-rust-chance-spectra --lazy-interior-chance \
  --public-observation --information-set-search \
  --determinization-particles 4 --determinization-min-simulations 32 \
  --no-belief-chance-spectra \
  --symmetry-averaged-eval --symmetry-averaged-eval-threshold 20 \
  --gate-config flywheel \
  --dump-config "$OUT/config.json" --config-hash \
  --config-purpose a1-r2-internal-n128 \
  --out "$OUT/result.json"
```

External candidate and incumbent commands must differ only by checkpoint,
device/output path, and run fingerprint; use the same `<PANEL_SEED>` and pair
count. Do not treat the current H0 rule as acceptable—fix it first.

```bash
PYTHONPATH="$ROOT/src" "$PY" tools/catanatron_neutral_harness_match.py \
  --checkpoint "$CANDIDATE" --opponent catanatron_value --mode search \
  --pairs 500 --base-seed <PANEL_SEED> \
  --workers 8 --device cuda:6 --threads-per-worker 3 \
  --n-full 128 --max-depth 80 --max-decisions 600 \
  --prior-temperature 1.0 --value-scale 1.0 --value-squash tanh \
  --value-readout scalar --c-visit 50.0 --c-scale 0.03 \
  --rescale-noise-floor-c 0.0 --sigma-eval 0.98 \
  --max-root-candidates 16 --max-root-candidates-wide 54 \
  --wide-candidates-threshold 24 \
  --correct-rust-chance-spectra --lazy-interior-chance \
  --public-observation --information-set-search \
  --determinization-particles 4 --determinization-min-simulations 32 \
  --no-belief-chance-spectra \
  --symmetry-averaged-eval --symmetry-averaged-eval-threshold 20 \
  --gate-config flywheel \
  --artifact-dir /home/ubuntu/catan-zero-production/runs/eval/a1-r2/external-candidate.games \
  --out /home/ubuntu/catan-zero-production/runs/eval/a1-r2/external-candidate.json

PYTHONPATH="$ROOT/src" "$PY" tools/catanatron_neutral_harness_match.py \
  --checkpoint "$INCUMBENT" --opponent catanatron_value --mode search \
  --pairs 500 --base-seed <PANEL_SEED> \
  --workers 8 --device cuda:7 --threads-per-worker 3 \
  --n-full 128 --max-depth 80 --max-decisions 600 \
  --prior-temperature 1.0 --value-scale 1.0 --value-squash tanh \
  --value-readout scalar --c-visit 50.0 --c-scale 0.03 \
  --rescale-noise-floor-c 0.0 --sigma-eval 0.98 \
  --max-root-candidates 16 --max-root-candidates-wide 54 \
  --wide-candidates-threshold 24 \
  --correct-rust-chance-spectra --lazy-interior-chance \
  --public-observation --information-set-search \
  --determinization-particles 4 --determinization-min-simulations 32 \
  --no-belief-chance-spectra \
  --symmetry-averaged-eval --symmetry-averaged-eval-threshold 20 \
  --gate-config flywheel \
  --artifact-dir /home/ubuntu/catan-zero-production/runs/eval/a1-r2/external-incumbent.games \
  --out /home/ubuntu/catan-zero-production/runs/eval/a1-r2/external-incumbent.json
```

Calibration should run data-local on the 2-B200 hub, using the same exact raw
root and manifest for both roles:

```bash
RAW=/home/ubuntu/catan-zero-production/runs/selfplay/a1-infoset-n128-p4-12000games-20260710-r1
VAL=$RAW/a1_post_wave.audit.validation_seeds.json
PY=/home/ubuntu/catan-zero-v1/.venv/bin/python
PYTHONPATH=/home/ubuntu/catan-zero-v1/src "$PY" tools/phase_sliced_value_calibration.py \
  --shard-dir "$RAW" --validation-seed-manifest "$VAL" --require-held-out \
  --checkpoint /immutable/a1-r2/candidate.pt --device cuda:0 \
  --value-readout scalar --out /immutable/a1-r2/candidate-calibration.json
PYTHONPATH=/home/ubuntu/catan-zero-v1/src "$PY" tools/phase_sliced_value_calibration.py \
  --shard-dir "$RAW" --validation-seed-manifest "$VAL" --require-held-out \
  --checkpoint /immutable/a1-r2/incumbent.pt --device cuda:1 \
  --value-readout scalar --out /immutable/a1-r2/incumbent-calibration.json
```

## Verification performed

- `git diff --check`: pass.
- Python compilation of the five changed runtime entry points: pass.
- Focused local suite excluding the locally missing `networkx` dependency:
  `190 passed, 8 skipped`.
- The neutral-harness test could not collect in the local macOS venv because
  vendored Catanatron imports `networkx`; the B200 gate previously ran it in the
  full environment.
- Read-only B200 receipt/claim/output replay confirmed the v4 training result
  and all eight GPUs idle.

## Methodology and limits

Focused differential review of the uncommitted RL-loop delta against commit
`739cbfb94f38b754d71d7e50cd90f20d186f9621`, plus one-hop consumers in
`train_bc.py`, evaluation tools, the sealed contract, and live B200 artifacts.
The tree was changing concurrently while the promotion-artifact builder was
being implemented; that builder must receive its own final review after its
tests settle. Confidence is high for the blocking findings above and medium for
the final post-checkpoint wall-time allocation, which still needs a short
real-checkpoint worker-count canary.
