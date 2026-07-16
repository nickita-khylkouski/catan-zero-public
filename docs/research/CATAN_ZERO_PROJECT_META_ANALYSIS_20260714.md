# Catan Zero: a forensic meta-analysis of the project, its learning failures, and the path to a reliable expert-iteration system

**Draft date:** 2026-07-14  
**Repository scope:** the canonical Catan Zero public repository, preserved historical branches, plans, audits, experiment manifests, and read-only surviving GPU artifacts  
**Primary research track:** two-player, no-trade, 10-VP Catan  
**Ultimate product target:** full four-player Catan with trading  
**Status:** internal research paper and decision record; not a peer-reviewed strength claim

> **2026-07-15 research addendum:** direct source tracing after this draft found
> a more proximal representation/value failure: the neural state omits
> development-card playability and other public multi-step rule state, the
> value readout cannot recover those distinctions from legal actions, and
> policy learning can damage the shared value representation. The exact
> diagnosis and repair program are
> `../audits/A1_RL_SOFTWARE_DIAGNOSIS_20260715.md` and
> `../plans/A1_REPRESENTATION_VALUE_RECOVERY_PLAN_20260715.md`.

## Abstract

Catan Zero began as a broad behavior-cloning and PPO effort, evolved into a structured 35-million-parameter entity-token policy, and then pivoted to AlphaZero-style expert iteration with Gumbel MCTS. The project achieved three credible internal policy-improvement generations and large systems gains, including information-set-safe search, lazy chance expansion, Rust feature construction, a native MCTS hot loop, durable multi-host generation, eight-rank B200 training, and typed promotion transactions. It nevertheless failed to turn a large n128/n256 search corpus into an externally stronger successor.

The failure was initially attributed to bad data, too few epochs, insufficient model capacity, or weak MCTS. The combined evidence does not support those explanations as the proximal cause. The dominant failed learners were not independent candidate tests: they chained candidate checkpoints, accumulated approximately 44.7 million sampled-row exposures, and were judged against an older checkpoint under the wrong search operator rather than against their actual initializer. The learner also optimized source policies with incompatible entropy scales, historically outcome-conditioned loser weights, stale or semantically ambiguous optional search-value fields, and incomplete behavioral rehearsal. These errors allowed held-out imitation and some internal matchups to improve while external playing strength regressed.

A later independent, producer-started 4.19-million-sample TEMP arm corrected the target-temperature mismatch and beat its exact parent 670–530 over 1,200 games. A 524,288-sample, 128-step checkpoint achieved much of the teacher closure with far less parameter drift and first scored 75–53 over 128 diagnostic games. Decisive matched follow-ups now localize the gain to that short controlled learner dose. Selected-dose checkpoint 9dd1 beat exact f7 187–133 over 320 internal games. A no-symmetry exact short-dose arm beat f7 150–106, while D6 and no-symmetry were 149–150 on the same games with matched p=1.0. Pure exact-f7 target gather was 157–163, no effect. Current-policy replay scope was externally indistinguishable from selected dose, p=.452. The winning mechanism is therefore not D6, target gather, or replay-policy scope; it is the 524,288-draw controlled update.

The result transfers externally. On one fixed 768-game catanatron_value cohort, selected dose scored 376–392 while f7 scored 292–476, an improvement of 10.94 percentage points with matched p=1.73e-5. Current-policy scored 365–403 on the same cohort, a raw +9.505-point lift over that f7 reference and not distinguishable from selected dose, p=.452. The final old-gather selector scored 403–365 while its own f7 rerun scored 311–457, a controlled +11.979-point lift. Run-to-run nondeterminism moved the f7 absolute score, so the correct gather-versus-selected comparison is the difference in controlled lifts: only +1.04 points, not the raw candidate-score gap. The older two-stage gather lineage separately scored 414–354 while f7 scored 305–463 on its matched cohort, p=2.36e-9 with no seat artifact. Because pure gather had no internal effect, the inherited gather tied its selected-dose base directly, and the common-cohort difference-in-differences was small, these external wins establish the short-dose lineage but do not isolate an adapter benefit. None of these diagnostic checkpoints is yet a promoted champion: the binding full gate is incomplete. Conversely, an independently trained n256 specialist showed an internal win but a statistically significant external regression, demonstrating competitive overfitting rather than data corruption.

The project’s central lesson is causal: more search, more rows, more epochs, more GPUs, and more architecture do not improve a policy if learner dose, target semantics, agent identity, and evaluation parent are uncontrolled. The first mechanism with both internal and matched external support is now precise: one independent 524,288-draw update from the exact parent. The highest-value immediate work is to reproduce that minimal recipe through the independent FINAL/full gate and make the repaired checkpoint/data semantics canonical. Architecture is no longer the proximal explanation for the recovered gain.

## 1. Epistemic rules

This paper distinguishes four evidence classes:

| Label | Meaning |
|---|---|
| **Verified** | Replayed source or immutable artifact supports the claim, and the experiment has no known causal invalidation. |
| **Diagnostic** | The run is mechanically valid and scientifically informative, but was not authorized or complete enough for promotion. |
| **Invalidated** | A later audit found a parent, operator, dose, provenance, target, or adjudication confound that prevents the original conclusion. |
| **Planned** | The code, manifest, or plan exists, but no accepted result proves the hypothesis. |

Offline loss, target closure, parameter drift, and calibration are mechanisms or tripwires. They are not playing-strength evidence. A candidate is stronger only when a fixed, correctly paired playing evaluation establishes that conclusion. A generator checkpoint, a public/tournament checkpoint, and a checkpoint-plus-search configuration are separate identities unless an artifact proves otherwise.

No repository artifact uses the exact label “PP-zero.” This paper interprets the requested “PP-zero origin” as the project’s pre-AlphaZero behavior-cloning plus PPO era. It does not invent an undocumented named system.

## 2. Evidence base and method

The reconstruction used:

- the project’s technical handoffs, teacher-data freeze, architecture plans, system paper, research chronicle, roadmap, and master plan;
- all dated learner, search, topology, target-semantics, integration, and flywheel audits;
- Git history across preserved branches and the canonical line;
- experiment plans, manifests, reports, checkpoint metadata, and known checkpoint SHA-256 identities;
- a read-only snapshot of the current H100 and B200 fleet;
- read-only checks of surviving B200 experiment directories and reports.

The repository’s visible public history begins on 2026-07-09 with the canonical import, although dated documents preserve the June program. Across all local refs the repository contains 856 commits: 32 dated July 9, 105 July 10, 172 July 11, 326 July 12, and 221 July 13. The high commit velocity explains why dated prose often describes an older loop than the current code. Live code and immutable artifacts take precedence over narrative documents when their scopes overlap.

Primary local sources include:

- [early technical handoff](../CATAN_AI_TECHNICAL_HANDOFF_2026-06-26.md)
- [teacher RNG/root-fix plan](../35M_TEACHER_ROOT_FIX_AND_AB3_PLAN_2026-06-28.md)
- [teacher-data freeze](../TEACHER_DATA_FREEZE_2026-06-29.md)
- [system paper](../CATAN_ZERO_SYSTEM_PAPER_2026-07-06.md)
- [research chronicle](../plans/CATAN_ZERO_RESEARCH_CHRONICLE.md)
- [master plan](../plans/CATAN_ZERO_MASTER_PLAN.md)
- [roadmap](../plans/CATAN_ZERO_ROADMAP.md)
- [A1 learner end-to-end forensics](../audits/A1_LEARNER_END_TO_END_FORENSICS_20260713.md)
- [learner recovery plan](../plans/A1_LEARNER_RECOVERY_PLAN_20260712.md)
- [stored-policy temperature result](../audits/A1_STORED_POLICY_TEMPERATURE_WIN_20260712.md)
- [policy/AUX replication](../audits/A1_POLICY_AUX_REPLICATION_20260712.md)
- [topology/gather causal audit](../audits/A1_TOPOLOGY_GATHER_CAUSAL_AUDIT_20260713.md)
- [paid-search target audit](../audits/A1_PAID_SEARCH_TARGET_SEMANTICS_20260712.md)
- [evaluation flamegraph](../profiling/EVAL_FLAMEGRAPH_2026-07-11.md)
- [flywheel invariant audit](../reviews/A1_FLYWHEEL_INVARIANT_AUDIT_2026-07-13.md)
- [integration differential review](../reviews/CATAN_ZERO_INTEGRATION_DIFFERENTIAL_REVIEW_2026-07-13.md)
- [v5 disaster-recovery boundary](../A1_V5_DISASTER_RECOVERY.md)
- [current operator handoff](../../RL_AGENT_HANDOFF.md)
- [fleet inventory](../../FLEET.md)

External literature claims in the master plan and reviews were not independently re-reviewed for this draft. They are treated as project hypotheses, not as newly verified literature conclusions.

## 3. Executive causal thesis

The project did not train badly for one reason. It crossed several failure boundaries at once:

1. **The experiment unit was wrong.** A candidate should be one fresh parent plus one declared optimizer dose. The failed n128/n256 line instead chained candidates and accumulated multiple doses.
2. **The comparator was wrong.** Some “wins” were measured against older gen3 at c-scale 0.03, not the f7 initializer/deployed agent at c-scale 0.10.
3. **The policy targets were not on one scale.** Search sources with different entropy and sharpness were mixed without per-source temperature calibration.
4. **The objective did not preserve incumbent behavior.** Historical loser weight 0.3, inadequate authenticated replay anchoring, and shared-trunk updates made behavior forgetting easy.
5. **Optional search-value targets were unsafe.** Root values were stale self-estimates and target scores were raw visited-action Q or generic preference scores, not one completed-Q return-scale contract.
6. **The architecture has real blind surfaces, but target gather did not cause the recovered strength.** The incumbent lacks direct action-to-target-token joins, player-seat identity, and a live longest-road signal. Long-dose topology/gather replicas tied; pure exact-f7 gather scored 157–163; inherited gather tied its selected-dose base. Architecture remains future R&D, not the proximal learner fix.
7. **Evaluation originally conflated model and operator.** A neural checkpoint is not the full agent; c-scale, MCTS budget, PIMC particle count, symmetry, masking, and runtime all affect strength.
8. **The loop’s evidence boundary lagged its compute boundary.** The project could generate and train at scale before it could reliably prove parentage, one-dose consumption, cohort freshness, and promotion eligibility.

The result looked paradoxical—better imitation, more MCTS, more rows, and worse play—only because the original experiment did not isolate the causal learner update.

## 4. Project evolution

### 4.1 Phase I: behavior cloning and PPO prototype, through 2026-06-26

The earliest documented system used:

- a flat 1,002-dimensional observation;
- a 607-action policy space plus 12 action-context features;
- a 1,024-wide MLP;
- behavior cloning from heuristic, value, and search teachers;
- PPO self-play with KL regularization, anchor imitation, Q auxiliaries, DAgger, and reanalysis;
- a four-player environment as the nominal product target.

The checkpoint named in the early handoff was a weak internal prototype. It scored 18/64 against the heuristic bot and 11/64 against the value bot. Three longer PPO branches were killed without useful checkpoints. The early diagnosis was already directionally correct: the representation was too flat, dense reanalysis was expensive, experiments were operationally fragile, and evaluation samples were too small.

PPO code remains in the repository as historical and experimental infrastructure. It is not the current champion lineage. The current learning program is search distillation/expert iteration.

### 4.2 Phase II: corrected teachers and the first 35M structured model, 2026-06-28 to 2026-06-29

A major teacher bug was found in the AlphaBeta path: teacher search consumed the global Python random stream, changing subsequent dice and resource outcomes. Search was therefore changing the trajectory it was meant to label. Saving and restoring RNG state around teacher calls produced zero replay mismatches in the post-fix test.

The audit also discovered that the advertised xdim_graph representation was not a real graph/history model. It split a flat vector into arbitrary chunks. Opening placement and robber play were weak, mixed teachers averaged incompatible policies, and forced/roll decisions could dominate training without carrying useful policy choice.

The replacement was an entity-token transformer:

- 19 hex tokens;
- 54 vertex tokens;
- 72 edge tokens;
- four padded player tokens;
- one global token;
- optional event/history tokens;
- sparse legal-action features;
- hidden size 640, six state layers, eight attention heads, dropout 0.05;
- 35,041,353 parameters in the incumbent configuration.

The frozen teacher bank contained 72,904 games and 22,915,911 rows:

| Corpus | Rows | Games/seeds | Track |
|---|---:|---:|---|
| Entity 2p | 14,933,075 | 64,127 | two-player |
| Entity 4p | 8,080,346 | 9,766 | four-player |
| Total | 23,013,421 converted rows in the two listed stores; 22,915,911 raw observed samples in the freeze summary | 72,904 raw games | mixed |

The slight difference between raw observed samples and converted listed stores reflects freeze/conversion accounting and should not be collapsed into one number. The QA reported no active illegal-action, zero-policy, or nonfinite target failures.

This phase established a durable lesson: data can be byte-valid and still be scientifically invalid if the teacher mutates the game RNG, mixes incompatible policy semantics, or trains on a mislabeled architecture.

### 4.3 Phase III: pivot to Gumbel-MCTS expert iteration, early July

The project moved from “train a policy and then PPO it” to:

~~~text
current model
  -> public-information self-play
  -> Gumbel MCTS improvement targets
  -> audited shards and memmap corpus
  -> one supervised learner dose
  -> paired candidate-versus-parent and external panels
  -> explicit promotion
  -> next generation
~~~

The first ladder produced meaningful internal improvements:

| Generation | Comparison | Result | Evidence status |
|---|---|---:|---|
| gen1 | versus v3a | 57.0% over 400 | Verified internal |
| gen2A | versus gen1 | 57.0%, LLR 3.57 | Verified internal |
| gen2 variant | versus gen1 | 59% | Diagnostic variant |
| gen3 | versus gen2A | 54.71% over 700 | Verified internal |
| turn 4 | mixed n8/n16 evidence | approximately 52–54% | Suggestive, not a clean promotion |

Historical external results against catanatron_value rose from approximately 35.5% to 37.0% to 45.7%. Those panels showed real transfer but did not prove parity at the sample sizes used.

### 4.4 Search repair: why more MCTS first made the policy worse

The first Gate-A search lost badly to the raw policy, at roughly 19–22%. At wide roots, many actions received only one or two visits. Noisy Q min-max rescaling then stretched tiny random differences across the full logit range. Search amplified noise rather than information.

Two repairs reversed the result:

1. lower c-scale to 0.03 in the then-current operator;
2. repair true self-play outcome/value semantics.

After repair, search beat raw policy at approximately 67–71%. This established a project-wide principle: MCTS simulation count is not a monotonic strength knob when the value noise, root width, rescaling, and prior calibration change.

Lazy chance expansion then reduced the cost of a search from roughly 5,400 expanded leaves to approximately 47–63 by enumerating root chance outcomes and sampling interior chance. This was a 13–19× structural reduction, not a micro-optimization.

D6 board symmetry exposed another signal-to-noise problem. Orientation-dependent value noise had standard deviation about 0.175 nats while candidate spread was only around 0.06. Twelve-way averaging reduced noise by about 3.3×. This led to the plan’s enduring order of operations:

1. denoise;
2. re-tune search calibration;
3. only then raise simulation budget.

### 4.5 Hidden-information leak and information-set-safe search

The network input was publicly masked, but historical MCTS cloned the authoritative game object, which contained hidden resources and cards. The search tree could therefore condition on hidden truth even though its neural evaluator could not. This invalidated the claim that old searched play represented a public-information agent.

The July 10 A1 work introduced public-observation, information-set-safe PIMC search with public conservation and sealed particle settings. The current operator uses four determinization particles with a minimum of 32 simulations per particle, D6 root averaging from legal width 20, corrected chance spectra, and no belief-chance spectra. Old leaked runs remain historical evidence only; they do not authorize current production.

### 4.6 Systems stabilization and canonical fleet software, July 9–11

The public repository line begins with:

- canonical v1 deployment;
- atomic file-write and SSH/path repairs;
- one generator process per physical GPU;
- corrected seed accounting;
- guard value checks;
- fleet/data integration;
- n128 A1 canaries;
- a one-dose executor;
- atomic promotion scaffolding;
- information-set-safe generation;
- a reproducible Rust wheel;
- native H100 evaluation;
- dual-arm n128/n256 campaign tooling;
- a native MCTS hot loop.

One fleet bug illustrates why systems correctness belongs in the research record. The launcher exposed four GPUs to one process, but every worker selected logical cuda:0. It also claimed seed space as games times workers times GPUs even though the generator consumed one seed per game. A nominal four-GPU job could use one GPU and overclaim seeds by roughly 64×. The fix launches one pinned generator per physical GPU and assigns one disjoint games-wide seed block per process.

### 4.7 The n128/n256 campaign and failed learner line, July 11–12

The dual-arm campaign planned:

- n128: 28 GPUs times 5,000 games = 140,000 games;
- n256: 28 GPUs times 2,000 games = 56,000 games;
- total = 196,000 games;
- a nominal 80/15/5 current/history/hard-negative mix;
- 16 workers per GPU;
- D6 at legal width 20;
- public-information native Rust search.

Authenticated n128 and n256 corpora survived and were consumed by several learner runs. The critical lineage was:

~~~text
f7
  -> n256 early: 2,962 steps, 12.13M sampled rows
      -> combined-196k: +7,403 steps, +30.32M sampled rows

f7
  -> corrective n256: 2,962 steps, 12.13M sampled rows
      -> corrective n128: +7,403 steps, +30.32M sampled rows
~~~

Each terminal chained candidate therefore accumulated about 44,692,523 sampled rows of candidate-lineage exposure. This was not an independent “train on n128” or “train on n256” experiment.

The reported 52–55% internal results were also compared with older gen3 using shared c-scale 0.03 rather than with the actual f7 initializer and its deployed c-scale 0.10 operator. A zero-training re-evaluation later showed the n256 LR 1.2e-4 checkpoint beating its actual f7 initializer 360–240 over 600 matched-operator games. The original label and the corrected label could differ because checkpoint plus search settings define the agent.

### 4.8 Learner forensics and controlled repairs, July 12–13

The end-to-end forensic audit rejected the convenient explanations:

- no NaN or Inf corruption was found;
- logits and values were not saturated;
- checkpoint architectures loaded consistently;
- the generator/teacher distribution matched a fresh native-runtime pilot closely;
- forced policy rows were already zero-weighted;
- the same 35M family had produced both gen3 and the stronger f7 agent.

Instead it measured progressive shared-trunk deformation:

| Checkpoint | Global parameter drift from f7 |
|---|---:|
| 524,288-sample midpoint | 0.691% |
| TEMP full dose | 2.598% |
| replay-anchor arm | 2.652% |
| n256 early | 5.167% |
| combined-196k | 9.763% |
| corrective n256 | 15.313% |
| corrective n128 | 34.129% |

The corrective n128 candidate’s shared trunk was severely altered while its value head moved less. Across LR studies, 96–98% of update energy landed in the shared trunk. The problem was behavior forgetting and objective imbalance, not simply a bad value head.

Two controlled results changed the diagnosis:

1. **TEMP arm.** An independent f7 initialization, fresh Adam, 4.194M samples, LR 3e-5, and per-source policy temperatures produced 670–530 over 1,200 exact64 games, 55.833%, with positive ordinary, pentanomial, and superiority evidence. This is a diagnostic win, not a promotion.
2. **Dose midpoint.** The 524,288-sample, 128-step checkpoint achieved teacher closure 0.1023 with 0.691% drift and scored 75–53 over 128 games. The 4.194M-sample dose achieved closure 0.1358 with 2.595% drift and scored 65–63 on the same-size diagnostic cohort. Eight times the samples and 12.41 times the integrated LR area bought only 1.33 times closure and 3.75 times drift.

Later causal panels strengthened the second result. The selected-dose checkpoint beat exact f7 187–133 internally and improved the matched external catanatron_value score by 10.94 points. A no-symmetry replication also beat f7 150–106, while D6 versus no-symmetry was 149–150 on identical games. Pure gather was 157–163. The recoverable learner signal is therefore the 128-step dose itself, not a symmetry or architecture add-on.

The selected experimental dose is therefore 524,288 samples, not “one epoch” and not “train until the loss looks smooth.”

### 4.9 Recovery and central coordination, July 13–14

The exact recovered v5 generator checkpoint survived, as did the f7 safety-reference bytes, but the original v5 promotion receipt, registry, and current pointer did not. The disaster-recovery transaction explicitly refuses to recreate that missing proof:

- recovered v5 is generator champion only;
- f7 remains the public/tournament safety reference;
- f7 is an unproven predecessor, not a proven causal parent;
- no promotion count is inferred;
- a recovery candidate must pass strict H1 against recovered v5 and a separate fixed f7 veto;
- auto-promotion remains false.

The central learner coordinator now models P1 recipe selection, AUX0/AUXT diagnosis, an independent FINAL replication, and the full recovery gate. Recent committed repairs bind exact recovery semantics and require the final candidate to contain learned, finite, nonzero public longest-road signal before entering the gate.

At the time of this paper, concurrent integration work has modified the central coordinator, stage executor, trainer, policy, tests, and auxiliary experiment code. The worktree is not a pristine release state. This paper therefore describes the intended current boundary and the last reviewed committed invariants, not a declaration that the dirty integration tip is production-ready.

## 5. Current system architecture

### 5.1 End-to-end control flow

~~~text
recovered/promoted generator identity
  -> sealed search/operator decision
  -> content-addressed generation contract
  -> one generator per physical H100
  -> Rust game + information-set-safe Gumbel MCTS
  -> immutable NPZ shards and per-job manifests
  -> reconciliation, duplicate and semantic audits
  -> selected train/validation game manifests
  -> duplicate-safe memmap corpus
  -> exact one-dose 8-rank B200 learner
  -> immutable checkpoint/report/optimizer evidence
  -> fixed internal and external common-random-number panels
  -> calibration, high-regret and bucket tripwires
  -> typed promotion transaction
  -> registry/current pointer mutation
  -> exact post-promotion handoff for next turn
~~~

The canonical system deliberately separates generation, training, evaluation, and promotion. A continuous controller is retained for experimentation but is explicitly noncanonical.

### 5.2 State and action model

The incumbent entity-token policy is a 35.041M-parameter Transformer with hidden width 640, six layers, eight heads, and 0.05 dropout. It consumes typed token banks and scores only currently legal actions. Its strengths are:

- player-count-padded public state;
- compact entity structure instead of a 1,002-flat-vector bottleneck;
- sparse action scoring;
- stable checkpoint compatibility;
- enough demonstrated capacity to produce multiple improving generations.

Its known blind surfaces are:

1. **No direct target-token join in the incumbent.** Adjacency may be computed, but the legacy action head scores legal-action features against a pooled state rather than directly gathering the board tokens an action affects.
2. **No board coordinates/IDs on vertex and edge tokens.** A pooled CLS vector cannot bind an absolute settlement class to a permuted vertex token.
3. **No explicit nonactor seat identity.** Player rows expose public attributes, but the model cannot always join “which public opponent” to fixed-color board ownership.
4. **Longest-road public signal absent historically.** Player slot 12 was always zero in historical corpora even though road ownership was visible. The model could count edges but could not recover connected chain length from a permutation-insensitive trunk without topology.
5. **Settlement auxiliary aliasing.** The original 54-class next-settlement head predicted an absolute vertex from CLS even though vertices had no canonical identity. A shared per-vertex pointer head fixes the semantics.

These are real architecture deficiencies. The first controlled experiments showed they were not sufficient to rescue the long-dose learner:

- topology plus gather, using the exact 4.19M-sample TEMP learner recipe, scored 601–599 over 1,200;
- gather-only fresh reproduction scored 591–609;
- the paths trained and changed, so the tie was not a dead flag or fallback.

A newer short-dose screen initially appeared to weaken that conclusion: an inherited target-gather checkpoint scored 63.02% against exact f7, with static action features at 60.94% and topology at 58.33%. The historical 45-column static action table and serialized target IDs truly were dead in the incumbent scorer, so the implementation defects remain real. Causal follow-up nevertheless rejected gather as the strength mechanism. The leader inherited 524,288 selected-dose draws before its 65,536 adapter stage, tied the selected-dose base 128–128, and pure exact-f7 gather scored 157–163. Architecture repairs remain valid function-preserving experiments, but the demonstrated improvement comes from the short learner dose. There is no evidence yet that the 35M model needs replacement.

### 5.3 Search operator

The current production A1 operator is:

| Field | Value |
|---|---|
| Track | 2-player, no-trade, 10 VP |
| Full simulations | n128 |
| Fast simulations | n16 |
| Full-search fraction | 0.25 |
| c-visit | 50.0 |
| Deployed producer c-scale | 0.10 |
| Sigma eval | 0.98 |
| Information sets | public observation plus PIMC |
| Particles | 4 |
| Minimum simulations/particle | 32 |
| Symmetry | D6 average at legal width at least 20 |
| Chance | corrected spectra, root enumeration, lazy interior sampling |
| Precision | strict FP32 |
| Adaptive n256 | disabled for current production |

The old n256 campaign used a different question: more total simulations. The proposed adaptive n256 mechanism keeps per-particle dose fixed—base n128 as P4x32 and wide roots n256 as P8x32—so it varies belief coverage rather than confounding belief coverage with per-particle search depth. That mechanism remains planned, not authorized production.

### 5.4 Stored row semantics

A valid self-play row can contain:

- public entity-token state;
- legal action IDs and action features;
- improved Gumbel policy target;
- selected action;
- terminal outcome/value target;
- root value;
- visited-action target scores;
- forced/full/fast search class;
- source category and producer/operator provenance;
- game seed and decision identity;
- optional future-event/AUX labels.

The learner’s trustworthy paid-search policy signal is the improved target policy on active full-search rows. Fast-PCR and single-action forced rows carry policy multiplier zero. They remain real value rows.

The optional value-like fields are not interchangeable:

- root value is a stale self/search estimate and can create self-distillation;
- target scores are raw visited-action Q in Gumbel shards, not the completed-Q values that form the improved policy;
- other teacher sources may put preference scores in the same generic field;
- the existing optional Q loss row-standardizes scores, which is incompatible with a return-scale Q head.

Therefore q-loss remains off until one typed completed-Q semantic contract exists.

## 6. Data history and what “the dataset” actually means

The project has several distinct datasets:

| Dataset | Purpose | Scale | Current interpretation |
|---|---|---:|---|
| June teacher freeze | BC warm-start, 2p and 4p | 72,904 games; 22.916M raw samples | Corrected historical teacher bank |
| Banked gen3-era window | expert-iteration states/targets | 32.6M rows; approximately 417GB | States useful, many targets stale |
| n128 corpus | stronger fresh search teacher | planned 140k games | Authenticated corpus exists; exact selected subset is contract-bound |
| n256 corpus | higher-search specialist | planned 56k games | Authenticated corpus exists; specialist overfit externally |
| Mixed replay | incumbent/gen3/gen4 rehearsal | nominal 20% in recovery experiments | Must be authenticated and anchor-eligible |
| 524k short dose | causal learner experiment | 524,288 sampler draws | Selected Pareto diagnostic dose |
| 4.19M sentinel | longer matched learner experiment | 4,194,304 sampler draws | Useful mechanism comparison, not automatic default |

The exact composite used by the newest short-dose screens contains 208,000 games and 47,620,447 rows:

| Component | Sampling share | Physical rows |
|---|---:|---:|
| n128 current | 57.1429% | 31,919,276 |
| n256 current | 22.8571% | 12,773,247 |
| gen3 replay | 20.0000% | 2,927,924 |
| Total | 100% | 47,620,447 |

The reports bind the exact f7 baseline SHA-256 as f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4. A 128-step eight-rank run at global batch 4,096 consumes 524,288 global sampler draws; per-rank reports can show 65,536 local draws and must not be mistaken for the global dose.

Only 5,414,299 of the 47,620,447 physical rows have positive policy weight in the audited composite, approximately 11.37%. A dedicated policy-active sampler can increase useful policy-row exposure by about 11.2× for a fixed base-draw budget. That is a systems/statistical efficiency gain, not strength evidence: the earlier extra policy-active-dose replication was 596–604.

The gen3 replay component’s stored policy is stale relative to the current teacher. In the corrected composite it is not distilled as a fresh search target; it supplies value coverage and, when authenticated, a behavior-anchor distribution. This distinction prevents replay from becoming a second incompatible policy teacher.

All 47,620,447 rows also authenticate an empty native event-history axis. The old model still paid event-encoder compute, memory, dropout-stream, and optimizer costs for no signal. The corrected learner scans the immutable payload, crops the axis to width zero, and freezes the disconnected encoder. This is an efficiency and causal-RNG repair, not a new source of game strength.

The n128/n256 data were not rejected as bad. A fresh 576-game native-runtime pilot matched a historical 576-game current-producer sample:

- target/prior KL differed by 0.43%;
- forced fraction by 0.14 percentage points;
- full-search fraction by 0.13 points;
- each phase fraction by less than 0.18 points;
- both had zero failures and truncations.

The early claim that “20% of the data was missing” conflated provenance completeness, quota contracts, and scientific usability. The important later conclusion is narrower: authenticated corpora exist and can support diagnostic training, but only a contract-selected game set with exact producer, source mix, seed identity, validation exclusions, feature semantics, and no duplicate paths is eligible for the canonical loop.

### 6.1 Forced rows

Forced moves were repeatedly suspected because they can be roughly half of all decisions. The code already sets policy weight multiplier zero for single-legal-action and fast-PCR rows. They contribute neither policy numerator nor denominator. They remain value examples because the state and eventual outcome are real.

Dropping forced rows is therefore a value-data ablation, not a policy bug fix. The recovery plan tests forced value weight 1.0 versus 0.25 only after the learner and search-value recipe stabilize.

### 6.2 Source-policy temperature

The TEMP experiment used per-source target temperatures:

- n128: 1.00;
- n256: 1.11;
- replay: 0.52.

This corrected a hidden objective mismatch: equal policy-loss weights did not mean equal teacher sharpness or equal gradient pressure. The independent TEMP win makes this one of the strongest established learner-side mechanisms.

### 6.3 Winner/loser weighting

The historical learner downweighted losing-game policy rows to 0.3. That made the search-policy objective outcome-conditioned and allocated only 18.14% of policy mass to losing trajectories. A corrected L1 recipe uses winner and loser policy weights of 1.0 and won both a direct incumbent gate and matched external panel in the recovery evidence summarized by the plan.

The corrected default is uniform winner/loser policy weight, not because all game states are equally informative, but because outcome-conditioned suppression was an unprincipled bias on the search teacher.

### 6.4 Sampling measure

The composite sampler selects component, then uniform game, then uniform row. Applying another inverse-game-length loss correction would double-correct long games. Data weighting belongs in one explicit sampling measure, not partly in the sampler and partly in losses.

## 7. Why the learners failed

### 7.1 Candidate chaining

Candidate chaining makes a dose look like a data comparison when it is actually a curriculum:

~~~text
parent -> n256 update -> combined update
~~~

The second model carries all prior parameter drift, optimizer-history effects if restored, and changed behavior into the next experiment. Even with fresh Adam, its lineage exposure remains cumulative. Comparing that terminal checkpoint with a fresh-parent arm does not isolate the second corpus.

Every scientific arm must independently reload the same parent bytes unless a typed curriculum declaration explicitly makes cumulative lineage the treatment.

### 7.2 Oversized dose

An epoch is not a portable training unit. Corpus size, world size, batch, accumulation, and sampling scheme change its meaning. The failed lines saw tens of millions of draws. The controlled curve shows that a short 524k dose captured most useful closure before drift dominated.

The correct independent variable is samples seen plus integrated optimizer schedule, not epochs or steps alone.

### 7.3 Wrong parent and operator

The original internal evaluation used older gen3 and c-scale 0.03. The candidate initialized from f7, whose deployed operator used c-scale 0.10. The result answered:

> Does this new checkpoint under operator X beat an older checkpoint under operator X?

It did not answer:

> Did the learner improve its actual parent agent?

The corrected panel fixes both candidate and parent to matched role-specific deployed operators and common random numbers.

### 7.4 Shared-trunk forgetting

Policy, value, and auxiliary gradients all pass through the shared entity trunk. Better teacher imitation on the new distribution can overwrite features that support external play. This is standard competitive overfitting expressed in a long-horizon imperfect-information game.

Evidence:

- increasing LR increased trunk drift strongly;
- value-head drift remained comparatively small;
- held-out imitation improved while external strength fell;
- n256 specialist improved internal same-lineage play but regressed sharply against catanatron_value.

The P1 recovery sweep tests authenticated incumbent-era replay plus forward KL/cross-entropy anchoring over eligible multi-action replay rows only:

| Arm | Conditional anchor weight | Global-mass equivalent |
|---|---:|---:|
| K0 | 0.000 | 0.00 |
| K3 | 0.006 | 0.03 |
| K10 | 0.020 | 0.10 |

If all full-update arms forget, a trunk-frozen head-only arm localizes whether trunk change is causal. Only a positive localization justifies a trunk LR sweep.

### 7.5 Policy-anchor implementation defects

The old anchor averaged reverse KL over all rows carrying priors. Forced single-action rows have exactly zero KL but entered the denominator, diluting the configured coefficient. The repaired behavior-preservation objective uses forward KL, equivalently old-policy cross-entropy, and only authenticated replay rows with more than one legal action.

Sparse DDP objectives also had a serious collective bug: one rank could return early when it lacked an eligible KL or Q row while another entered an all-reduce, causing deadlock or rank-divergent behavior. The fix makes all ranks enter the reduction, contributing graph-connected zero numerators when locally empty.

### 7.6 Stale search-value self-distillation

Historical continuous training used blends such as:

~~~text
target = lambda * terminal outcome + (1 - lambda) * stored root value
~~~

At convergence, the stored root value is largely the network teaching itself. Repeated reuse can slow fresh value learning or amplify error. Current recovery arms hold lambda at 1.0 until a controlled V100 versus V75 comparison establishes that a 0.75 outcome plus 0.25 search-value blend helps. HL-Gauss is conditional on that target comparison.

### 7.7 Categorical-value misdiagnosis

The early 33-bin HL-Gauss replication failed badly: scalar loss progressed approximately 0.665 to 0.809 to 0.842, while HL-Gauss was approximately 1.198 to 1.533 to 1.710. Later audit found that the nominal 0.25 categorical coefficient did not equal the scalar head’s initial gradient budget. Ninety-six of 128 categorical arms clipped versus one scalar arm.

This rules out that particular recipe. It does not prove categorical values are bad. A fair test needs head-only warmup or an analytically matched coefficient, with no scalar auxiliary and identical sample dose.

### 7.8 Architecture was a real defect but a false proximal explanation

The 35M model has real representational limits, but:

- the same family produced the improving gen ladder and f7;
- the first 4.19M-sample topology/gather controlled replicas tied;
- 47.8M v3b did not beat 35M v3a;
- a 91M probe undertrained at epoch one and its value head destabilized at epoch two;
- current RRT and graph alternatives were far slower.

The 192-pair CRN screen initially looked like evidence that a function-preserving action-local repair mattered: inherited gather 63.02%, static action residual 60.94%, and topology 58.33% against exact f7. The incumbent serialized both action target IDs and a 45-column static action table but did not use them in its scoring path. Some expensive data representation was literally dead at the learner boundary.

The lineage correction and pure-parent replication overturn that interpretation. Checkpoint 03886c did not start at exact f7: it started from selected-dose checkpoint 9dd1 after 524,288 draws, then trained only its adapter stage for another 65,536 draws. Its total lineage dose was 589,824 draws. It tied current-policy 337–335, tied its selected-dose base 128–128, and pure exact-f7 gather 2227ae… scored 157–163 against f7. Gather contributed no measurable strength.

The old two-stage gather still transferred externally: 414–354 against catanatron_value while f7 scored 305–463 on the same 768-game cohort, paired p=2.36e-9 with no seat artifact. That is important strength evidence for the lineage, not the adapter. The selected-dose base independently transferred on another fixed cohort, and pure gather did not improve internally. The causal conclusion is therefore:

1. fix independent initialization, dose, objective, and comparator;
2. use the minimal 524,288-draw recipe as the recovered learner mechanism;
3. complete the independent FINAL/full gate; the common-cohort selector is now complete and shows only a +1.04-point difference-in-differences for gather over selected dose;
4. treat gather/static/topology as lower-priority architecture R&D;
5. scale model capacity only if a stable 35M learner underfits active targets without external regression.

### 7.9 Optional-module checkpoint completeness was fail-open

The checkpoint loader historically used non-strict state loading and broadly allowed parameters for optional modules to be absent. If a checkpoint config enabled target gather, action cross, topology, static-action residual, categorical value, uncertainty, or AUX modules but its state dictionary lacked those tensors, ordinary inference could instantiate fresh zero/random parameters and continue. The config therefore claimed one function while the checkpoint bytes supplied another. This could contaminate both evaluation and architecture comparisons without an obvious crash.

The corrected loader makes all config-enabled optional modules complete by default. `EntityGraphPolicy.load` now rejects missing enabled-module tensors, while a deliberately named `allow_missing_optional_parameters` escape exists only for function-preserving warm-start construction. Rust evaluator factories and the shared evaluation server call the strict default. This is the correct enforcement layer because every checkpoint-backed native or Python MCTS path ultimately loads the same policy. The historical missing `q_head` remains a narrow legacy exception and is not consumed by production search.

## 8. Experiment ledger

### 8.1 Results that can guide decisions

| Result | Outcome | Status | What it establishes |
|---|---|---|---|
| gen1/gen2A/gen3 internal ladder | roughly 54.7–57% generation-over-generation | Verified internal | Expert iteration can improve this model family |
| Gate-A repaired search | approximately 67–71% search over raw | Verified diagnostic | Search calibration/value semantics mattered more than raw sims |
| D6 averaging | about 3.3× value-noise reduction | Verified mechanism | Symmetry is a high-value denoiser |
| lazy interior chance | about 13–19× fewer leaf expansions | Verified systems/search | Chance handling was a dominant cost |
| TEMP | 670–530/1200, 55.833% | Diagnostic | Source target-temperature matching can recover an independent learner |
| midpoint dose | 75–53/128, 0.691% drift | Diagnostic | Short dose is on a better closure/drift frontier |
| selected 524k dose, internal | 187–133/320 | Verified diagnostic | Short controlled update improves exact f7 |
| selected 524k dose, external | 376–392 versus bot; f7 292–476 on same cohort | Verified diagnostic | +10.94pp, paired p=1.73e-5 |
| full dose | 65–63/128, 2.595% drift | Diagnostic | More samples need not improve play |
| n256 specialist internal | two cohorts totaling 54.08%, about +28 Elo | Diagnostic | Stronger search can specialize internally |
| n256 specialist external | 18–46 versus f7 33–31 on same 32 paired seeds; delta −23.44pp, p=.00592 | Verified diagnostic regression | Internal improvement did not transfer |
| topology+gather | 601–599/1200 | Diagnostic tie | Missing topology was not proximal |
| gather-only | 591–609 | Diagnostic tie | Direct gather alone did not fix learner |
| AUX replication | 596–604/1200 | Diagnostic inconclusive | Repeating policy-active exposure did not help |

### 8.2 New short-dose CRN screen against exact f7

Every result below used 192 common-random-number pairs, 384 games, the same exact f7 baseline, matched c-scale 0.10, n128, public PIMC, D6, and the native hot loop. These are internal diagnostic panels. H1 refers to the internal regression-protection SPRT and does not make a checkpoint promotion-eligible.

| Arm | Candidate wins | f7 wins | Score | Internal status |
|---|---:|---:|---:|---|
| inherited D6 plus target-gather adapter stage | 242 | 142 | 63.02% | H1 / superiority H1; two-stage lineage |
| current-policy scope, 128 steps | 240 | 144 | 62.50% | H1 / superiority H1 |
| static action residual, 128 steps | 234 | 150 | 60.94% | H1 / superiority H1 |
| target gather, 256 steps | 230 | 154 | 59.90% | H1 / superiority H1 |
| pure search plus deployed tanh, 128 steps | 226 | 158 | 58.85% | H1 / superiority H1 |
| topology only, 128 steps | 224 | 160 | 58.33% | H1 / superiority continue |
| AUX64 gather | 218 | 166 | 56.77% | Rejected below base gather |
| deployed tanh, 128 steps | 217 | 167 | 56.51% | H1 / superiority continue |
| pure soft, 128 steps | 209 | 175 | 54.43% | continue |

The matched 256-step variants of gather, current-policy, and pure-search-plus-tanh were each approximately 3.1 percentage points below their shorter counterpart. This repeated dose penalty supports the short-dose frontier and indicts the historical 4.19M production-gather dose as an over-dose, but it does not repair the gather parentage confound.

A subsequent direct panel compared the inherited gather candidate with current-policy over 336 paired seeds and 672 games. Gather won 337 and current-policy won 335: a tie. Another direct comparison with the selected-dose base was exactly 128–128. Pure exact-f7 gather then scored 157–163. The initial screen ranked lineages, not architecture effects.

### 8.3 Decisive causal and external localization

| Comparison | Result | Interpretation |
|---|---:|---|
| selected-dose D6 versus exact f7, internal | 187–133/320, 58.44% | The 524k lineage improves internally |
| no-symmetry exact short dose versus f7 | 150–106/256, 58.59% | Short dose works without D6 |
| D6 versus no-symmetry on identical games | 149–150, matched p=1.0 | D6 is not the learner-strength mechanism |
| pure exact-f7 gather versus f7 | 157–163/320, 49.06% | Gather alone has no measurable effect |
| inherited gather versus selected-dose base | 128–128 | Adapter adds no direct strength |
| selected dose versus catanatron_value | 376–392/768, 48.958% | Candidate score on fixed external cohort |
| exact f7 versus catanatron_value, same cohort | 292–476/768, 38.021% | Selected dose gains 10.94pp; paired p=1.73e-5 |
| current-policy versus catanatron_value, same cohort | 365–403/768, 47.526% | Above f7; indistinguishable from selected, p=.452 |
| current-policy raw lift over the same f7 reference | 365 versus 292 wins/768 | +9.505pp; the earlier +8.72pp figure was arithmetic error |
| final old-gather selector versus catanatron_value | 403–365/768, 52.474% | Common-cohort candidate result |
| exact f7 rerun for final old-gather selector | 311–457/768, 40.495% | Controlled gather-lineage lift +11.979pp |
| gather versus selected controlled-lift difference | +11.979pp versus +10.938pp | Only +1.04pp difference-in-differences; use this because the f7 rerun moved |
| old two-stage gather versus catanatron_value | 414–354/768, 53.906% | Strong external lineage result on its cohort |
| exact f7 versus catanatron_value, old-gather cohort | 305–463/768, 39.714% | Matched discordant p=2.36e-9; no seat artifact |
| canonical no-symmetry gather gate, current aggregate | 268–244/512, 52.344% | `continue`; insufficient for a promotion decision |

The external runs must not be compared through raw candidate win rates alone. Each candidate is compared with the f7 execution paired to its own run. Even on the nominally common cohort, the f7 rerun moved from 292 to 311 wins, exposing engine/runtime nondeterminism at the absolute-score level. Controlled lifts are therefore the valid comparison: selected +10.938 points, current-policy +9.505 points against the original f7 execution, and gather +11.979 points against its own f7 rerun. Gather exceeds selected by only +1.04 points under difference-in-differences. That does not reverse the causal finding that pure gather had no effect.

The simplest supported learner recipe is now the selected 524,288-draw update. D6 remains a verified search-denoising mechanism, but it did not cause this learner gain. Current-policy scope is optional because it did not improve over selected dose. Architecture adapters remain experimental.

### 8.4 Results invalidated as causal evidence

| Original claim | Why invalidated |
|---|---|
| combined-196k proves mixed n128/n256 improvement | initialized from an already-trained n256 candidate; cumulative 44.7M-sample lineage |
| corrective n128 proves n128 fixes n256 | initialized from corrective n256; same chaining/oversized-dose failure |
| 52–55% internal means parent improvement | compared with older gen3 under wrong operator, not actual f7 initializer |
| n256 is globally better because it wins internally | contradicted by matched external regression |
| forced moves diluted policy loss | policy multiplier already zero; they only remain value rows |
| bad H100 data caused the regression | fresh native pilot closely matched historical producer sample |
| 35M is obviously too small | same family produced f7; controlled architecture additions tied |
| HL-Gauss is universally worse | tested with an unmatched/clipping gradient budget |
| more HBM use or more epochs means a better learner | no strength relation; longer dose increased drift |

### 8.5 Remaining live frontier

Read-only inspection of the second B200 host found mechanically completed checkpoints and retained reports for the fast learner arms. Internal causal localization and the selected/current/gather external panels are complete. The final gather selector produced only a +1.04-point controlled-lift advantage over selected dose. What remains unadjudicated is the binding full gate and several lower-priority composition comparisons:

- current-policy pure-search;
- deployed-tanh value;
- pure-search plus deployed tanh;
- double-dose pure-search tanh;
- gather current-policy;
- gather plus static current-policy;
- double-dose current-policy;
- independent FINAL replication of the selected short-dose recipe.

Their process return codes were zero and checkpoint files existed. The collected panel artifacts bind exact f7, common seeds, matched search settings, and 384 games per reported arm. They are meaningful internal evidence, but remain nonpromotable until independent external and full-gate evidence exists.

## 9. Software and scientific bugs found and repaired

### 9.1 Teacher and environment

- global RNG consumed by AlphaBeta teacher search, changing future game chance;
- hidden truth available to MCTS through authoritative state clones;
- mislabeled “graph” representation that was arbitrary flat-vector chunks;
- mixed teacher policies with incompatible semantics;
- unmasked featurization in at least one historical path;
- longest-road public feature slot permanently zero;
- settlement auxiliary target unidentifiable from permutation-invariant CLS.

### 9.2 Search and generation

- Q min-max rescaling amplified low-visit noise at wide roots;
- true self-play outcome/value semantics were wrong in early Gate-A;
- eager chance expansion wasted thousands of leaves;
- all visible GPUs routed into one logical cuda:0 process;
- seed claims multiplied by workers and GPUs despite one seed/game;
- atomic writes called flush/fileno on Path objects rather than open handles;
- ad hoc MPS daemons died or left stale locks; systemd foreground management fixed persistence;
- public masking did not initially imply information-set-safe tree search;
- D6 originally rotated board/target IDs without remapping the action’s global spatial catalog identity, so the incumbent could evaluate a rotated board against the original node/edge action feature;
- adaptive n256 historically confounded particle count with per-particle dose.

### 9.3 Corpus and provenance

- adjacent-game duplicate logic incorrectly carried an open seed run across independent source roots;
- repeated canonical shard paths could multiply data weight;
- composite seeds were not always source-namespaced;
- selected versus validation versus optimizer-excluded seed identities were incompletely bound;
- memmap resume/config identities omitted feature semantics;
- optional event crops needed byte-backed proof that all omitted masks were empty;
- the full 47.62M-row current composite carries authenticated empty event history despite the model historically paying for that path;
- only about 11.37% of physical rows carry positive policy weight, so unconditioned row sampling wastes most policy-update capacity;
- stale replay policy targets needed to be separated from replay value/behavior-anchor use;
- teacher rows and memmap corpora bind the Rust entity-adapter version and reject mixed known/unknown semantics, but inference checkpoints historically did not bind the adapter version that produced their training features;
- legacy and authoritative public-award rows needed row-level routing and transition evidence;
- same corpus could be at risk of repeated consumption without an immutable turn/state claim.

### 9.4 Learner numerics and objectives

- candidate chaining and cumulative dose not surfaced as lineage exposure;
- historical loser policy weight 0.3 outcome-conditioned imitation;
- source target temperatures mismatched;
- the historical 4.19M production-gather dose obscured the real 524k-dose gain; later pure-parent evidence showed gather itself was neutral;
- policy anchor direction and denominator wrong;
- optional zero-weight heads still received AdamW decay or advanced dropout RNG;
- zero-signal optimizer steps could occur;
- nonfinite gradients needed fail-closed consensus;
- training diagnostics performed extra autograd passes and parameter clones;
- train-value-only did not freeze every intended adapter;
- target ID padding used zero instead of negative sentinel;
- accumulation used unsafe mean-of-means aggregation;
- sampler weighting could be corrected twice;
- validation aggregation and sparse objective denominators were wrong;
- prefetch dropped replay-KL scope;
- sparse DDP KL/Q objectives could deadlock;
- rank-zero-only RNG state made sharded resume incomplete;
- rank-divergent zero-step decisions needed collective consensus;
- resume identity omitted later feature/holdout semantics.

### 9.5 Architecture semantics

- adjacency existed but incumbent trunk did not consume it;
- action logits lacked direct action-to-board-target joins;
- the serialized 45-column static action table was dead in the incumbent; its nonredundant columns 19–41 were never consumed;
- vertex/edge tokens lacked canonical coordinates/identity;
- nonactor player tokens lacked explicit seat identity;
- public longest-road signal was absent;
- absolute settlement classifier could not equivary with vertex permutations;
- legacy event modules could decay even when authenticated event width was zero;
- the checkpoint loader could accept config-enabled optional modules with missing tensors and silently evaluate fresh zero/random parameters; ordinary loads now fail closed and reserve the escape hatch for explicit warm starts.

### 9.6 Evaluation and promotion

- checkpoint evaluated against wrong parent;
- a checkpoint could be evaluated through the live Rust entity adapter without proving that its training adapter version matched, allowing featurizer semantic drift to masquerade as a model change;
- checkpoint and search operator treated as separable after the fact;
- mixed simulation budgets pooled into suggestive claims;
- external absolute-SPRT rule could block honest comparative non-regression and could be threshold-manipulated;
- high-regret and bucket summaries lacked authoritative raw-game producers;
- cross-net evaluation lacked crash-resumable per-game receipts;
- no canonical validation ledger existed in an early audit;
- installed native wheel version/capabilities were checked without verifying exact wheel bytes;
- retired continuous controller treated disabled gate or generic truthy pass as positive evidence;
- orchestrator initially failed to forward promotion cohort-exclusion evidence;
- next-turn initialization could strand a corpus after a crash;
- a bare `EntityGraphPolicy.load(...); policy.save(...)` round trip was not metadata-idempotent: unless every caller re-supplied them, it reset `mask_hidden_info` to false and `soft_target_source` to empty, and dropped value-readout and training-information-surface attestations;
- snapshot EMA required equality for architecture, masking, action catalog, entity adapter, public-award contract, and static features, but not for soft-target source, trained readout set, or training information surface; it then copied only the newest checkpoint's metadata onto averaged weights;
- original v5 registry/pointer/promotion receipt was lost.

The adapter-version gap is distinct from byte identity. Existing row-level checks preserved the runtime adapter through NPZ and memmap data, but checkpoints did not prove which adapter semantics their weights learned. The repaired design gives a dependency-free `entity_adapter_contract.py` ownership of the current adapter name and an explicit `legacy_or_unknown` sentinel. New policies bind the current adapter; ordinary and distributed checkpoint writers persist top-level `entity_feature_adapter`; loading preserves the exact stored string, maps missing/blank legacy checkpoints to the sentinel, and never guesses current semantics during re-save. `EntityGraphRustEvaluator.__init__` centrally rejects a bound adapter that differs from the live runtime, covering local, batched, native Gumbel, reanalysis, and H2H paths. EvalServer validates its server checkpoint, includes the adapter in its handshake, and validates local fallback. No legacy evaluator override was added. Focused evidence was 89 passed and four skipped, including direct/distributed save, legacy persistence, exact-string preservation, evaluator mismatch rejection, EvalServer lifecycle/payload, teacher-adapter, and checkpoint-serialization tests.

The metadata-round-trip issue has a narrower current blast radius than its API suggests. The production behavior-cloning writer explicitly supplies masking, soft-target, value-training/readout, and information-surface fields on single-GPU, DDP, and FSDP paths. The function-preserving architecture upgrader reopens its output and restores every source top-level field except the intentionally changed model and config; that repair was introduced after the upgrader once relabeled a masked checkpoint as unmasked. The remaining sharp edge is the public `save` default itself: it reconstructs a payload rather than inheriting the loaded policy's semantic attestations. A future transform can therefore repeat the historical failure unless the API becomes tri-state/inheriting or requires explicit metadata.

EMA is the more immediate promotion risk. Two checkpoints can have the same tensor schema and inference contracts while differing in whether a categorical readout was actually trained, which soft policy target produced their updates, or which event/information surface reached the optimizer. Averaging their model tensors and copying the newest attestation does not prove the averaged readouts were trained under that attestation. EMA must compare normalized semantic training contracts, at minimum `soft_target_source`, the trained-readout set, and the relevant information-surface identity, or mark the output diagnostic-only until separately retrained and gated. Diagnostic interpolation already carries `diagnostic_only=true` and `promotion_eligible=false`; ordinary EMA does not have that containment.

### 9.7 Operations

- divergent host trees and undocumented branch state;
- personal paths and fragile SSH defaults;
- incorrect assumptions about GPU process counts because idle MPS servers looked like compute;
- file descriptor exhaustion with many memmap parts;
- unstable ad hoc MPS startup;
- slow many-small-file staging;
- documentation described incompatible legacy and current loops;
- release capability names did not prove installed binary identity.

## 10. Performance and compute

### 10.1 Current fleet

The documented fleet contains:

- 64 H100 GPUs: eight four-GPU hosts and four eight-GPU hosts;
- 16 B200 GPUs: two eight-GPU hosts;
- total: 80 accelerators.

The second B200 host has an eight-GPU full NVLink/NVSwitch mesh, approximately 183GB HBM per GPU, 208 CPU threads, about 2.8TB RAM, and measured NCCL average bus bandwidth of 485.834GB/s. It is accepted for eight-rank local DDP.

At the 2026-07-14 01:29 UTC read-only snapshot:

- every H100 reported 0% utilization and only a small persistent MPS footprint;
- every B200 reported 0% utilization and no compute application;
- H100 MPS services were active;
- the main B200 MPS service was active; the R&D B200 MPS service was inactive.

Thus no generation, training, or evaluation workload was running at that snapshot. Historical throughput must not be reported as a current run.

### 10.2 Generation efficiency

Historical measured and synthetic capacity figures include:

- approximately 121–128k rows/hour/H100 in a prior 24-H100 MPS run;
- approximately 81.93k rows/hour/GPU for one documented 96-worker evidence point;
- approximately 91.85k rows/hour/GPU at a synthetic 128-worker frontier;
- approximately 2.20M rows/hour over 24 H100s in a synthetic-checkpoint EvalServer recipe.

These are recipe-specific measurements, not a universal capacity model for the sealed A1 n128 information-set-safe runtime.

### 10.3 Learner geometry

On B200:

| Geometry | Throughput | Relative |
|---|---:|---:|
| 8 ranks × local batch 512 | 2,643 rows/s | 100% |
| 4 ranks × local batch 1,024 | 1,953 rows/s | 73.9% |

Both preserve global batch 4,096. Batch 4,096 was also the safe single-device ceiling for the 35M model; 16,384 OOMed a 178GB B200. GPU memory occupancy is not the objective. The selected topology is eight-rank FP32 because it is faster at matched global batch and its learning parity was tested.

### 10.4 Evaluation flamegraph

Before the native hot-loop work, mutually exclusive wall time was approximately:

| Component | Wall time |
|---|---:|
| Python traversal/bookkeeping | 34.31% |
| feature construction plus FFI | 34.57% |
| waiting on asynchronous evaluator | 20.39% |
| active neural inference | 7.96% |

The external Python-Catanatron bridge was only about 6.55%, so replacing the bridge alone could not explain or fix the slow evaluation.

Rust featurization improved median total leaf processing from 6.354ms to about 4.5ms and feature work from 0.785ms to 0.0969ms. The native MCTS hot loop delivered about 1.383× internal simulations/second and 1.263× neutral-harness elapsed improvement in canaries. After the port, feature/leaf FFI remained about 44.15%, native traversal 26.62%, and Python orchestration 3.73%. Packing 16 workers instead of eight on B200 improved simulations/second by 19.4% and games/hour by 14.8%.

Optional AUX heads were also being computed during inference even though search discarded their outputs. Adding an explicit inference skip was bit-exact and improved a B200 forward benchmark by 5.2% at batch one and 1.8% at batch 48. The larger single-row gain matters because much of search still presents small effective batches; it is a safe systems win, not a strength claim.

The remaining performance target is batched feature/inference work and lower leaf-boundary overhead, not merely “use more GPU.”

### 10.5 Alternative architecture economics

Measured H100 probes:

| Architecture | Parameters | Rows/s | Relative to incumbent |
|---|---:|---:|---:|
| incumbent entity transformer | about 35.04M | 1,500.4 | 1.0× |
| relational recurrent transformer | 20.07M | 203.1 | 0.135× |
| residual relational GCN | 20.94M | 6.68 | 0.0045× |
| think-RRT K4 | not primary | 197.9 | 0.132× |
| MoE RRT | not primary | 133.6 | 0.089× |

These implementations are not viable production replacements without major systems redesign. Their slowness does not prove the ideas are weak; it proves their present implementations fail the Bitter Lesson’s compute-efficiency constraint.

## 11. The Bitter Lesson applied correctly

The project’s version of the Bitter Lesson should not mean “always make the model larger” or “always increase n.” It means prefer scalable search, data, learning, and evaluation mechanisms over handcrafted strategy patches.

The highest-leverage scalable mechanisms demonstrated here are:

- native game/search execution;
- information-set-safe parallel MCTS;
- playout-cap randomization;
- symmetry averaging;
- large independent search corpora;
- short repeated learner doses from an exact parent;
- population and external evaluation;
- typed, replayable promotion transactions.

Handcrafted Catan strategy features may still be useful as diagnostics or public state inputs, but the learning loop should discover strategy from scalable supervision. Conversely, compute does not excuse causal sloppiness. Eight B200s can reproduce a confounded objective faster; 64 H100s can generate more stale or inbred targets faster. The Bitter Lesson requires a correct improvement operator, not just more FLOPs.

## 12. Current canonical learner program

### 12.1 P1: select anti-forgetting recipe

From the exact recovered current parent, every K arm independently:

- applies the authorized zero-step public-award transition;
- uses fresh Adam;
- runs FP32 on 8×512 B200 geometry;
- consumes 4.19M samples for the initial P1 sweep as specified by the recovery plan, with centrally fixed data routing;
- uses winner/loser weights 1.0;
- holds scalar MSE, lambda 1.0, and forced value weight 1.0;
- differs only in replay-anchor coefficient;
- is evaluated on fixed internal and external common-random-number cohorts.

The plan then selects the Pareto recipe, not necessarily the lowest validation loss.

This is the documented recovery specification, but its 4.19M-sample starting dose is superseded by the causal evidence. The 128-step selected dose improved internally and externally; extending matched families to 256 steps lost about 3.1 percentage points; historical 4.19M gather tied. Current-policy replay scope was externally indistinguishable from selected dose, p=.452. The provisional P1 recipe should therefore be the minimal selected 524,288-draw update. A K0/K3/K10 sweep is conditional R&D if an independent FINAL shows forgetting, not a prerequisite before reproducing the demonstrated recipe.

### 12.2 P2: localize trunk damage

Reuse the selected full-update checkpoint as control. Independently train a trunk-frozen head-only arm. If head-only restores external performance, run a small trunk LR multiplier sweep. If it does not, do not assume the trunk is the problem.

### 12.3 P3: test trustworthy search value

Compare:

- V100: pure terminal outcome;
- V75: 0.75 outcome plus 0.25 authenticated root value on masked eligible rows;
- VH75: the same blend with fairly budgeted 33-bin HL-Gauss, only if V75 helps.

Do not enable generic q-loss.

### 12.4 P4: forced-value and independent dose curve

The recovery plan specifies forced value weight 1.0 versus 0.25 followed by fresh-parent independent doses at 4.19M, 8.39M, and 16.78M samples. That grid is now too coarse and begins beyond the observed short-dose frontier. Preserve its causal structure—fresh parent for every point—but move the curve down to include 128, 256, and intermediate/long sentinels before spending on multi-million-sample arms. Early stopping cannot be learned from three points that all lie after the likely optimum.

### 12.5 P5: architecture after stability

Use a three-way diagnostic:

1. selected P1 full-update checkpoint;
2. independent head-only checkpoint;
3. independent head-only plus zero-initialized action gather/cross path.

Only if action-local capacity adds strength over head-only should the project commission a larger action-local architecture. Only if the stable 35M learner underfits active search targets without external regression should it scale model size.

### 12.6 AUX commissioning and FINAL

The auxiliary program uses:

- a zero-step public-award transition;
- a function-preserving pointer upgrade;
- 128 head-only warmup steps;
- five fixed 512-row gradient-geometry probes;
- exact sufficient statistics for main/AUX norms and dot product;
- a mechanically selected AUX coefficient capped by norm ratio and opposing projection;
- matched AUX0 and AUXT 524,288-draw arms from the same warmed bytes;
- fixed internal and external panels;
- an independent FINAL run from the raw causal lineage, never from an AUX diagnostic candidate.

FINAL must use a fresh sampler seed and physical row set, fresh Adam, and the selected 128-step dose. It must begin with exact-zero public-award parameters and end with finite nonzero learned slot-12 signal. Gate eligibility is not promotion.

## 13. Promotion model

The desired promotion rule is not “candidate trained successfully.” It is a transaction:

1. replay exact training parent, initializer transformation, corpus, code, runtime, command, environment, sample order, and completed dose;
2. replay fixed candidate-versus-parent games under typed agent identities;
3. replay matched external candidate/incumbent panels on the same cohort;
4. verify calibration, high-regret, population, and bucket tripwires from retained raw evidence;
5. verify cohort freshness and disjointness;
6. for recovery, require strict H1 against recovered v5 plus the independent fixed f7 veto;
7. preflight registry and pointer mutation;
8. atomically write receipt, registry, pointer, and post-promotion handoff;
9. let the next turn consume only that exact handoff.

The current code has repaired several historical fail-open paths:

- disabled gate is a hold, never a pass;
- promotion requires exact positive verdict fields;
- cohort exclusions are hash-bound and replayed;
- one consumed corpus cannot initialize two learner turns;
- exact crash recovery may adopt identical bytes but cannot rewrite history;
- a recovered v5 receipt cannot masquerade as a normal promotion.

Outstanding review work remains around the concurrent central-stage integration and authoritative production of all raw high-regret/bucket evidence.

## 14. Readiness matrix as of 2026-07-14

| Layer | State | Evidence | Remaining condition |
|---|---|---|---|
| Public game masking | Implemented | code/tests/audits | Preserve in every producer/evaluator |
| Information-set-safe MCTS | Implemented and historically canaried | A1 runtime artifacts | Final combined parity gate |
| n128 production operator | Sealed in handoff | lock/render/docs | Recovery no-wave prerequisites |
| n256 | Historical diagnostic only | corpus and specialist results | Do not use as uniform production default |
| H100 generation fleet | Provisioned, idle at snapshot | read-only fleet snapshot | Launch only sealed contract |
| B200 8-rank learner | Measured and supported | DDP probe and reports | Final clean integration/no-op parity |
| n128/n256 data | Authenticated corpora survived | forensic audit/artifacts | Use exact contract-selected subsets |
| 35M learner | Mechanically mature with a supported short-dose recipe | trainer/tests/matched panels | Independent FINAL and full gate |
| Short dose | Internal and matched-external gain | 187–133 internal; +10.94pp external | Independent FINAL plus full gate |
| TEMP recipe | Diagnostic win | 1,200-game report | Production replication required |
| Architecture upgrade | Not proximal | pure gather 157–163; inherited gather/base 128–128 | Revisit only after stable learner gate |
| Entity-adapter binding | Repaired in integration tree | central evaluator/loader contract; 89 passed/4 skipped | Consolidate with canonical tip |
| External evaluator | Native hot loop with matched selected/current panels | fixed 768-game cohorts | Finish selector and retain raw receipts |
| Promotion transaction | Hardened | flywheel audit | Current recovery dual-baseline evidence |
| Champion lineage | Recovery-only | disaster-recovery receipt design | Cannot recreate missing v5 promotion |
| Current new champion | None established | no accepted full gate | Complete FINAL plus full gate |
| Current workloads | Final old-gather same-cohort external selector running | latest experiment handoff | Do not infer selector outcome before retained report |

## 15. Pareto-ranked next actions

### Priority 1: freeze and verify the current integration tip

The shared worktree contains concurrent edits across the model, trainer, coordinator, stage executor, and tests. Before GPU use:

- settle one canonical diff;
- run focused central-stage, public-award, DDP, sample-order, AUX, and promotion tests;
- run repository gate;
- run eight-rank no-op/parity canary on the exact source/runtime bytes;
- seal the executor authority and command.

This is not bureaucratic over-verification. It protects the causal identity of the next expensive experiment.

### Priority 2: reproduce the minimal short-dose recipe as independent FINAL

Run the selected 524,288-draw recipe independently from the exact causal initializer with a fresh sampler order, fresh Adam, and the fixed internal/external cohorts. The mechanism now has internal and matched-external support; the remaining question is reproducibility under the canonical FINAL transaction. The old-gather selector is complete and does not justify delaying FINAL for architecture experiments.

### Priority 3: run the binding full gate

Evaluate FINAL against the current causal parent, recovered-v5 requirement, fixed f7 veto, and matched external panel. Select by:

1. external panel;
2. internal parent panel;
3. active teacher closure;
4. phase calibration;
5. drift/clipping;
6. throughput.

Promote only through the typed atomic transaction. Do not chain a diagnostic checkpoint into FINAL.

### Priority 4: make P1 replay anchoring conditional

Current-policy scope did not beat selected dose externally. Do not spend the critical path on K3/K10. If independent FINAL fails by forgetting, then run K0/K3/K10 from the same parent and one head-only localization; this asks whether replay anchoring or shared-trunk change is the remaining cause.

### Priority 5: preserve architecture work as R&D

Keep target gather, static residual, topology, and AUX improvements function-preserving and benchmarkable, but do not attribute the recovered strength to them. Pure gather was neutral and AUX64 underperformed the inherited base.

### Priority 6: start the next n128 wave only after promotion

The next wave should use the newly promoted generator if and only if the gate succeeds. If the candidate holds, diagnose and adjust the learner; do not generate a massive same-parent corpus merely to keep GPUs busy.

### Priority 7: conditional R&D

- If anti-forgetting wins: test root-value blend.
- If head-only wins: tune trunk LR.
- If a future independent gather/head arm wins externally: commission action-local architecture.
- If stable 35M underfits: scale model.
- If external regression persists despite preserved parent behavior: add opponent-pool/high-regret distribution work.
- If n128 search gain is weak after denoising: test adaptive n256 as P8x32 at wide roots, not global n256.

## 16. Open hypotheses

### H1: extra current-policy rehearsal is unnecessary at the selected dose

Current-policy scope and selected dose were indistinguishable externally, p=.452, while both exceeded f7 on the matched cohort. At 524k draws the short dose itself appears to control forgetting. Replay anchoring remains a fallback for longer or future-distribution updates, not part of the minimal supported recipe.

### H2: the shared trunk is the destructive surface

If a head-only arm externally recovers while full update fails, representation forgetting is causal. If both fail, the objective or labels remain wrong.

### H3: source entropy mismatch was the main policy-target defect

TEMP’s diagnostic win is strong evidence. Independent production replication must show the gain survives fresh row order and full external gates.

### H4: short doses dominate at the current target quality

The 524k point lies on a better closure/drift frontier and now improves on matched internal and external panels. The independent FINAL asks whether the result reproduces under a fresh row order; longer-dose expansion should require evidence rather than being the default.

### H5: root search value can improve long-horizon credit

A small, authenticated 25% root-value blend may reduce terminal-return noise. It may also reintroduce self-distillation. V100/V75 isolates this.

### H6: target gather does not improve the current short-dose learner

Pure exact-f7 gather scored 157–163, inherited gather tied its selected-dose base 128–128, and AUX64 gather underperformed the inherited base. The dead-input repair remains technically correct, but it is not a current strength lever. Static residual and broader action-local capacity remain separate hypotheses.

### H7: global n256 over-specializes

The n256 internal/external split suggests higher search can narrow style rather than improve general play. Adaptive wide-root particle coverage is the more precise test.

### H8: population data is necessary for transfer

If a stable, anchored n128 learner still improves internally but not externally, the remaining failure is likely distributional inbreeding. Then old champions, external-loss starts, and high-regret restarts become first-class data sources.

### H9: the 2p research track will not transfer automatically to 4p full trade

The current token layout is partly four-player-ready, but search backup, action catalog, native trade prompts, and league dynamics remain materially 2p/no-trade. Success on the research track is a method result, not a full-Catan product claim.

## 17. What should not be done

- Do not launch a full 64-H100 wave merely because the fleet is idle.
- Do not chain diagnostic candidates.
- Do not select by validation loss alone.
- Do not compare a candidate with the wrong parent or search operator.
- Do not turn n256 on globally because it has more simulations.
- Do not drop forced rows under the belief that they still affect policy loss.
- Do not enable generic q-loss.
- Do not revive PPO as the default champion path.
- Do not scale to 91M before the stable 35M learner is characterized.
- Do not treat HBM occupancy as a training objective.
- Do not report diagnostic TEMP, midpoint, topology, AUX, or fast-arm checkpoints as promotions.
- Do not recreate the missing v5 promotion record from circumstantial evidence.
- Do not let a mutable checkout or version label substitute for exact source and wheel bytes.
- Do not use the ultimate four-player benchmark to describe current two-player evidence.

## 18. Reproducibility and limitations

### 18.1 Surviving identities

Important surviving identities include:

- recovered v5 generator SHA-256 beginning 6817…;
- f7 safety-reference SHA-256 f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4;
- midpoint evaluation artifact SHA-256 beginning f52ea3…;
- TEMP evaluation artifact SHA-256 beginning 9074d9…;
- topology-plus-gather candidate SHA-256 beginning 63a560….

Full digests should be copied from sealed artifacts, not from this abbreviated narrative.

### 18.2 Evidence loss

The v5 promotion receipt, champion registry, and pointer did not survive. That loss permanently limits causal ancestry claims. The disaster-recovery design contains rather than hides the limitation.

### 18.3 Branch and documentation drift

Historical branches preserve useful experiments but can disagree with the canonical line. Several handoffs were correct when written and obsolete days later. A result should cite its source commit, runtime, contract, and artifact hashes.

### 18.4 Current dirty tree

The current integration worktree contains uncommitted concurrent changes. It cannot be cited as a released executable state until consolidated and tested.

### 18.5 External strength

Internal two-player paired games and catanatron_value panels do not establish state-of-the-art four-player Catan. No claim in this paper should be read as such.

### 18.6 Historical throughput

Throughput depends on model, search settings, particles, symmetry, CPU topology, workers, MPS, and evaluator architecture. Historical rates are reference points, not current capacity guarantees.

## 19. Conclusions

Catan Zero has already solved many hard engineering problems: a structured public-state policy, corrected search teachers, information-set-safe Gumbel MCTS, a native engine path, scalable H100 generation, B200 DDP training, duplicate-safe corpora, and replayable promotion transactions. Its failure to produce a stronger n128/n256-trained champion was not evidence that expert iteration failed or that the data were useless.

The learner experiments failed causally. They chained candidates, overshot the useful dose, mixed target sharpness, forgot incumbent behavior, and evaluated the resulting checkpoint against the wrong parent/operator. Once those variables were controlled, a 524,288-draw update improved both internally and on matched external cohorts. Causal subtraction was decisive: removing D6 did not remove the gain; adding pure gather did not create one; current-policy scope did not improve it. The project has finally isolated a learner mechanism rather than another correlated candidate.

The project is now at two narrow, answerable questions:

> Does an independent FINAL reproduction of the minimal 524,288-draw recipe retain the internal and external gain under a fresh sampler order?

> Does that independently reproduced candidate clear the recovered-v5 requirement, fixed-f7 veto, calibration, high-regret, bucket, and promotion transaction gates?

Both questions are answerable quickly on the existing fleet. The completed old-gather common-cohort selector does not change the causal recipe. If FINAL yields a fully gated win, the flywheel can compound again. If it fails, head-only localization and conditional replay anchoring identify the next bottleneck. The route to the strongest Catan agent is therefore not another uncontrolled full run. It is a sequence of small, exact, independently initialized improvement tests whose winner alone earns the next data distribution.

## Appendix A: compressed chronology

| Date | Milestone | Main lesson |
|---|---|---|
| 2026-06-26 | BC/PPO prototype handoff | Flat policy and fragile PPO were weak |
| 2026-06-28 | AlphaBeta RNG root fix | Teacher calls must not mutate trajectory chance |
| 2026-06-29 | Entity transformer and teacher freeze | Structured representation and audited data bank |
| early July | Gumbel expert-iteration pivot | Search distillation produced three internal improvements |
| early July | Gate-A repair | Search calibration can reverse MCTS strength |
| early July | lazy chance and D6 | Structural search/denoising gains dominate raw sims |
| before July 10 | hidden-state audit | Masked net did not imply information-set-safe tree |
| July 9 | canonical public/fleet import | One source of truth and deployable stack |
| July 10 | A1 PIMC/search sealing | Public-information production path |
| July 10–11 | Rust featurizer/native MCTS | Evaluation bottleneck moved out of Python traversal |
| July 11 | n128/n256 196k campaign | Large fresh search corpus |
| July 11–12 | chained learner runs | Internal metrics rose while external strength regressed |
| July 12 | TEMP and dose experiments | Target temperature and short dose recovered signal |
| July 12–13 | forensics and bug sweep | Learner semantics and adjudication were proximal |
| July 13 | recovery/flywheel invariants | Missing promotion evidence contained, not recreated |
| July 14 | central P1/AUX/FINAL integration | Mechanically controlled next learner path |
| July 14 | short-dose causal localization | Pure gather was neutral; no-symmetry retained the gain; selected dose transferred externally |
| July 14 | inference completeness hardening | Config-enabled optional modules may no longer load with missing tensors during ordinary evaluation |
| July 14 | entity-adapter contract | Checkpoints and every Rust evaluator now fail closed on feature-adapter semantic mismatch |

## Appendix B: current decision table

| Question | Current answer |
|---|---|
| Was n128/n256 data corrupt? | No proximal corruption found. |
| Did more search automatically help? | No; n256 specialized internally and regressed externally. |
| Was forced policy trained? | No; forced and fast rows already carry zero policy multiplier. |
| Are forced rows useless? | Unknown for value; test later as a value-weight ablation. |
| Is 35M too small? | Not established. |
| Did direct topology/gather fix strength? | No for target gather at the tested dose: pure gather was 157–163 and inherited gather tied its selected-dose base 128–128. |
| Did source target temperature matter? | Yes, strongly in a diagnostic independent arm. |
| Did more epochs help? | Not under the failed design; longer dose increased drift. |
| Is 8×B200 training useful? | Yes, at fixed global batch it was faster and matched learning. |
| Should production use n256 now? | No; current sealed wave is uniform n128. |
| Is there a new champion? | No accepted full-gate result establishes one. |
| Is the loop conceptually complete? | Mostly; current integration still needs consolidation and final parity/gate evidence. |
| What is the next best experiment? | Reproduce the minimal 524k recipe as independent FINAL and run the binding full gate. |

## Appendix C: repository implementation map

| Responsibility | Canonical implementation family |
|---|---|
| Entity policy | ../../src/catan_zero/rl/entity_token_policy.py |
| Entity features | ../../src/catan_zero/rl/entity_token_features.py |
| Gumbel self-play | ../../src/catan_zero/rl/gumbel_self_play.py |
| Learner | ../../tools/train_bc.py |
| Memmap corpus | ../../tools/build_memmap_corpus.py |
| One-dose execution | ../../tools/a1_one_dose_train.py |
| Iteration state machine | ../../tools/a1_iteration_orchestrator.py |
| Flywheel turn binding | ../../tools/a1_flywheel_turn.py |
| Promotion transaction | ../../tools/a1_promotion_transaction.py |
| v5 disaster recovery | ../../tools/a1_v5_disaster_recovery.py |
| Central P1/AUX/FINAL coordination | ../../tools/a1_aux_pair_coordinator.py |
| H100 generation orchestration | ../../tools/fleet/a1_production_executor.py |
| H100 evaluation orchestration | ../../tools/fleet/a1_h100_eval_fleet.py |
| Native engine release | ../../tools/build_catanatron_rs_wheel.sh |
| Retired experimental continuous loop | ../../tools/continuous_flywheel.py |

## Appendix D: July 15 coherent-loop addendum

This addendum records the next day of integration and live experimentation.
It supersedes the paper wherever the older text describes the current loop as
only conceptual or still PIMC-based. Results explicitly marked in progress are
not promotion evidence.

### D.1 One executable science contract

Canonical GitHub `main` reached commit
`206c2be05d8b837d4f234838110254adbe40dbb0` with one machine-readable
coherent-public contract for generation, learning, evaluation, and promotion:

- `configs/operations/a1-next-wave-coherent-public-v1/science.contract.json`
- `tools/a1_current_science_contract.py`

The contract makes the agent identity a checkpoint plus an exact search
operator. The current operator searches one coherent public-belief tree; it
does not search a clone containing authoritative hidden cards, and it does not
silently fall back to the older PIMC operator. Production sealing remains
closed until the teacher-budget campaign is adopted into this contract.

The learner contract now distinguishes mechanical forced rows from strategic
policy rows. Forced rows have zero policy mass. Their realised-outcome value
signal is retained, but `END_TURN` receives 0.1x and `ROLL` 0.25x value weight;
unlisted forced types retain 1.0x. Policy-active rows use capped, within-game
search-surprise redistribution so a long game cannot steal total mass from
other games. The public-card residual has a separate 4x LR group.

### D.2 Teacher-budget result: n256 has not earned production

The whole-game common-random-number comparisons used the exact v5 checkpoint
for both agents and changed only the wide-root simulation rule:

| Arm | Adaptive wins | Base n128 wins | Adaptive rate |
|---|---:|---:|---:|
| n256 when legal width >=20 | 97 | 103 | 48.5% |
| n256 when legal width >=40 | 99 | 101 | 49.5% |

Neither adaptive arm improved playing strength. The corrected fixed-root audit
also exposed an important Catan-specific semantic fact: in roughly 300 real
champion trajectories it observed more than 3,700 `PLAY_TURN` roots, including
hundreds at width >=20, but none at width >=40; the observed play-turn maximum
was 39. Therefore a width-40 trigger is effectively an opening-placement-only
operator in this distribution. Earlier panels accidentally populated their
"wide" bucket mostly with opening placements and could not support a generic
hard-turn claim.

The replacement panel is stratified by reachable phase and width:
`PLAY_TURN` 2--19, 20--31, and 32--39, plus
`OPENING_PLACEMENT` 40+. Until that panel is aggregated and adopted, the formal
selection is pending. The current causal evidence favors base n128+D6; it does
not justify spending the production wave on global or adaptive n256.

### D.3 Learner-dose audit found two more causal bugs

The 8xB200 campaign independently reloads the exact f7 parent with fresh Adam
for every arm. It compares LR/warmup recipes at 128 optimizer steps and a fixed
global batch of 4,096; no arm may initialize from another candidate. The old
base sampler delivered only about 50,875 policy-active rows in 524,288 draws,
because most Catan rows are mechanical or otherwise carry zero policy mass.
The new auxiliary active sampler targets about 524,987 policy-active rows so
the LR comparison measures an actual policy update rather than mostly value
rehearsal.

Live command inspection stopped the first sweep before accepting a result. It
was still inheriting three old-composite defaults instead of the current
learner semantics:

1. the typed `ROLL`/`END_TURN` forced-value map was absent;
2. the public-card LR multiplier was 1x rather than 4x;
3. exact per-game surprise weighting was disabled.

A deeper sampler inspection found that the auxiliary policy-active batch also
conditioned only on component mix and positive policy mass. Because that
auxiliary batch supplies roughly 90% of the intended policy-active dose, it
would have ignored the hard/search-disagreement prioritization for most policy
updates. The repair conditions the already-composed authenticated
component-plus-surprise measure on policy-active rows, then applies any sealed
phase allocation. This is a substantive learner fix, not an observability
change.

The old composite has authenticated-empty meaningful-event histories.
Consequently this pre-wave diagnostic correctly uses the receipt-backed
card-only function-preserving initializer; pretending to train the history
gate on those rows would be false. Fresh coherent-wave data will use the
combined bias-free public-card-count v2 plus meaningful-public-history v2
initializer.

As of this addendum the corrected four-arm learner sweep is being rebound; no
checkpoint from the stopped mismatched sweep is accepted as evidence.

### D.4 Evaluation and next execution boundary

A 64-H100 evaluation matrix is implemented for the four independent learner
arms. It assigns every GPU exactly once and runs eight concurrent, seat-swapped
comparisons: each arm versus f7 and versus the recovered v5 incumbent, using
one common-random-number cohort and the teacher operator adopted above. The
matrix deliberately refuses to launch while teacher selection is provisional.

The immediate sequence is therefore:

1. finish the reachable fixed-root panel and adopt base n128 or an adaptive
   rule only if its preregistered evidence passes;
2. run all four corrected 8xB200 learner arms from f7 with equal realised
   policy-active dose;
3. evaluate all arms concurrently against both f7 and v5 on 64 H100s;
4. replay only the playing-strength winner for the longer 256-step dose from
   the original parent, never from its short-dose candidate;
5. promote only after the matched coherent-public gate passes;
6. generate the next fresh wave with the promoted checkpoint and the combined
   card-count/history v2 input surface.

The project is no longer asking whether "more MCTS" or "more epochs" helps in
the abstract. It is testing one teacher operator, one independently initialized
learner dose, and one matched agent identity at a time. That is the minimum
causal unit required for a reliable self-improvement loop.
