# Production opponent mix

The existing opponent-mix sampler remains deterministic and default-off. This
runbook only makes its inputs reproducible: registry pointers are resolved once,
every checkpoint is verified, and workers receive the same frozen config.

## 1. Populate distinct registry slots

Set the public/previous champion role, then append separate older and hard
checkpoints with explicit provenance tags. `ChampionRegistry` computes each md5
when the entry is written.

```bash
python -m tools.champion_registry --registry runs/champion_registry.json \
  set-role --role public_champion --checkpoint runs/public/checkpoint.pt --version 3

python -m tools.champion_registry --registry runs/champion_registry.json \
  append-pool --checkpoint runs/archive/gen2/checkpoint.pt --version 2 \
  --provenance '{"tag":"older_champion"}'

python -m tools.champion_registry --registry runs/champion_registry.json \
  append-pool --checkpoint runs/hard_negative/checkpoint.pt \
  --provenance '{"tag":"hard_negative"}'
```

The producer, public/previous, older, and hard checkpoints must contain distinct
bytes. A copied or aliased producer checkpoint is not a real opponent and is
rejected. A checkpoint appearing in two categories is also rejected rather than
silently receiving double probability.

## 2. Resolve and freeze the 75/10/5/5 + 3% external mix

Use the checked-in registry-backed template, override its registry path, bind it
to the exact producer, and set the external lane to exactly 3% of the effective
mix (rather than the template's approximate raw weight):

```bash
python -m tools.opponent_mix_registry \
  --manifest configs/opponent_mix/opponent_mix_r9_exploiter.json \
  --registry runs/champion_registry.json \
  --producer-checkpoint runs/generator/checkpoint.pt \
  --external-fraction 0.03 \
  --freeze-output runs/gen5/opponent_mix.resolved.json
```

This command fails before generation if:

- a required role or pool category resolves empty;
- any checkpoint is missing or its registry/manifest md5 is stale;
- the same checkpoint bytes occur twice or match the producer;
- the external fraction exceeds the existing 5% cap; or
- the output already exists.

The output contains only concrete absolute checkpoint paths, computed md5s,
effective weights, a producer binding, and a SHA-256 config digest. It is created
once with mode `0444`; rerunning the command cannot overwrite it. Loading it also
rechecks the digest and all checkpoint bytes.

## 3. Generate from the frozen manifest

Pass the resolved file, not the live registry-backed template:

```bash
python tools/generate_gumbel_selfplay_data.py \
  --checkpoint runs/generator/checkpoint.pt \
  --opponent-mix-manifest runs/gen5/opponent_mix.resolved.json \
  ...
```

The generator verifies and binds this config in the main process before spawning
workers. The verified `OpponentMixConfig` is then passed directly to every worker;
workers do not reread the registry, so a promotion during a run cannot split the
fleet across different opponents.

Omitting `--opponent-mix-manifest` preserves the prior pure-self-play behavior.
