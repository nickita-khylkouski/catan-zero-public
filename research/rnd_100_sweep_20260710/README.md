# Catan-Zero 100+ Trial R&D Sweep

This directory is an isolated, promotion-ineligible research workspace. Nothing
here authorizes production generation, training, evaluation, promotion, merge,
or push operations.

## Frozen baseline

- Source commit: `c807874940fd5b3e4c51775f33a64279786504da`
- Production target hardware: NVIDIA H100 80GB
- Screening hardware: A100 40GB and GH200, with finalists reproduced on H100
- Incumbent model: dense h640/L6, 8 heads, 35,041,353 parameters

## Trial families

| Family | Minimum | Screening host |
|---|---:|---|
| Architecture and representation | 36 | `129.213.28.15` (8x A100) |
| Learner, objectives, and data efficiency | 36 | `150.136.90.111` (8x A100) |
| Search, simulator, and systems | 34 | `192.222.51.39` (GH200), H100 validation on `68.209.74.159` |

The minimum registered total is 106 trials. Failed and invalid trials count only
when their attempted configuration and failure reason are retained; they do not
count as evidence for a scientific conclusion.

## Required controls

1. Every trial records source commit, host GPU, seed, command/config digest,
   input identity, wall time, status, and metrics.
2. Quality comparisons use a fixed train/validation split and fixed seeds within
   each family.
3. Performance comparisons hold logical work constant and report both absolute
   throughput and hardware identity.
4. Cross-GPU throughput is never treated as an H100 production conclusion.
5. Short proxy trials rank hypotheses; they do not establish full-training
   strength. Finalists require H100 reproduction and later searched H2H.
6. No experiment may touch fleet services, production outputs, registered
   contracts, promotion artifacts, or remote Git branches.

## Result record

Each JSONL record should contain at least:

```json
{
  "trial_id": "family-000",
  "family": "architecture|learner|systems",
  "status": "passed|failed|invalid",
  "source_commit": "c807874940fd5b3e4c51775f33a64279786504da",
  "host": "hostname",
  "gpu": "GPU model",
  "seed": 0,
  "config": {},
  "input_id": "fixture or corpus digest",
  "wall_seconds": 0.0,
  "metrics": {},
  "failure": null
}
```

## Decision hierarchy

1. Correctness and public-information safety.
2. Held-out policy/value quality and calibration.
3. Stability across seeds.
4. H100-normalized throughput and memory.
5. Parameter or implementation complexity.

The sweep produces hypotheses and a ranked shortlist. Only a later, controlled
multi-seed learner run followed by searched paired H2H can select a production
model.
