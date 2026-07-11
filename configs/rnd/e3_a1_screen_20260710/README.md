# E3 A1 fixed-K screen

This is an isolated, non-production 15-run screen: `K0/K1/K2/K4/K8 × seeds
11/29/47`. K1, K2, K4, and K8 have 22,146,068 trainable parameters and use
one shared recurrent block. K0 has 20,070,932 parameters and is a compute-only,
capacity-unmatched control. The preregistered primary comparisons are K2 versus
K1 and K4 versus K1; K0 cannot support a promotion claim.

The checked-in `experiment.template.json` is intentionally non-runnable. The
required sequence is:

1. `tools/rnd_e3_a1_admission.py initialize --repo-root "$CHECKOUT"` creates a
   separate K0 checkpoint for each training seed, then uses
   `tools/rnd_latent_upgrade_checkpoint.py` to create the four function-
   preserving expansions. It refuses overwrite and requires identical expanded
   model-state fingerprints across K1/K2/K4/K8 for each seed.
2. `register` receives the 15 `ARM@SEED=PATH` checkpoints, identity report,
   A1 corpus, selected-game/validation manifests, four relocated artifacts, and
   source root. It hashes every payload file and exclusively publishes
   `configs/rnd/e3_a1_screen_20260710/experiment.registered.json`.
3. Prefer `admit-all`: it authenticates the 40.69GB corpus and all other inputs
   once, preflights every destination, then transactionally publishes all 15
   manifests. Repeating `--run ARM@SEED` restricts publication to one host's
   registered subset. A link failure rolls back every manifest created by that
   invocation. Single-run `admit` remains available. Both paths publish only
   `runs/rnd_e3_a1_screen_20260710/{arm}/seed_{seed}/admission.json`; each emitted
   `train_argv` is the exact 250-step, microbatch-1024, accumulation-4 A1 recipe.

Registration or admission fails on missing identity evidence, source drift,
corpus inventory drift, an unregistered arm/seed, a wrong output directory, or
an existing destination. No command changes production defaults or promotes a
checkpoint.

`learning_gate.v1.json` is a separate pre-outcome scoring contract bound to the
immutable registered experiment's semantic and file hashes. The scorer requires
all 15 runs on identical holdout decisions. Its primary comparisons are K2/K4
versus K1 using nonforced game-macro soft-target CE and a paired crossed
seed/game bootstrap. K8 is secondary; K0 is descriptive only. The overall
safety gate is nonforced decision-micro CE within seed, preserving the weight of
large games. Bootstrap count and RNG seed are frozen in the gate contract.
