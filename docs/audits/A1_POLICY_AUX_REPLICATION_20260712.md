# A1 policy-active dose replication (2026-07-12)

## Question

Does one additional policy-active learner dose improve a full model when the
treatment and control independently start from the same f7 parent and consume
the same value/data dose?

## Sealed comparison

- Candidate: `L1_POLICY_AUX`, SHA-256
  `0b27b75daf93fecea08eb044f525479bb53d59deafc533c0dbb995c9449b9086`
- Matched control: `L1_CONTROL`, SHA-256
  `d7e1720fa0ea3ee8b2aacce27826367beadf64c139ce269a46366730c34f418a`
- Common parent: f7, SHA-256
  `f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4`
- Plan hash:
  `sha256:539592746d93a385400bb9ddbac08664c10b506595bb9d3089b3d33ae53b97cf`
- Run ID: `a1-eval-5c91300936142533`
- Operator: paired same-seed/color-swap, n128, P4, D6 at width 20,
  public-observation information-set search, native MCTS hot loop.

## Result

- 600 complete pairs / 1,200 games
- Candidate 596, control 604 (49.6667%)
- Pair outcomes: 122 candidate sweeps, 352 splits, 126 control sweeps
- Zero errors and zero truncations
- Pentanomial regression-protection SPRT (-10/+15 Elo): LLR -1.4421,
  `continue`
- Pentanomial superiority SPRT (0/+15 Elo): LLR -1.7651, `continue`
- Approximate unpaired score interval: 46.84% to 52.50% (about -22 to +17 Elo)

The treatment did not demonstrate improvement over its actual initializer-
matched control. It is not promotion eligible and should not receive another
policy-only dose. This also confirms why earlier candidate-chained results
against an older gen3 baseline were not evidence that the learner improved
from its own starting point.

## Consequence

The next learner experiment changes the representation/objective rather than
repeating policy exposure: independently initialized matched-dose AUX0 versus
the existing CAT-100 dense auxiliary-subgoal heads. Separately, PIMC search
target work must test aggregate-before-transform because averaging each
particle's nonlinear improved policy can stabilize a biased strategy-fusion
mixture without increasing playing strength.
