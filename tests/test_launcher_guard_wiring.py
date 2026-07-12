"""CAT-75 regression + integration tests: build_parser() extraction from the
three real launchers (tools/generate_gumbel_selfplay_data.py, tools/train_bc.py,
tools/continuous_flywheel.py) is behavior-identical to their old inline-main()
parsers, and tools/prelaunch_guard.py's guard library (CAT-69) is actually
wired into each launcher's main() rather than build-and-shelved.

Part 1 (regression): each launcher's build_parser() produces the exact same
set of option strings the pre-refactor inline parser produced (golden lists
below, captured from the pre-refactor code via the same
argparse.ArgumentParser.parse_args-intercept trick tests/test_train_bc_cli_
defaults.py already uses), plus exactly the one new --skip-guards flag. Sample
argvs are parsed into the same Namespace values the pre-refactor parser would
have produced.

Part 2 (wiring): each launcher's guard-spec-building helper is exercised
end-to-end (without running a real generation/training/gate launch) to prove
an intentionally-bad launch config is refused (SystemExit) and a good one is
not, and that --skip-guards logs a warning and bypasses rather than silently
no-oping.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path
import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import continuous_flywheel  # type: ignore  # noqa: E402
import generate_gumbel_selfplay_data as gen_cli  # type: ignore  # noqa: E402
import launcher_guards  # type: ignore  # noqa: E402
import prelaunch_guard  # type: ignore  # noqa: E402
import train_bc  # type: ignore  # noqa: E402

VAL_ONLY_SEED = 6_195_000_000  # inside prelaunch_guard.VAL_ONLY_SEED_RANGE

# ---------------------------------------------------------------------------
# Part 1: build_parser() regression -- golden option-string sets captured from
# the pre-refactor inline-main() parsers (tests/test_cli_config_drift.py's
# sibling: that file guards *defaults*, this one guards the *flag set* itself
# across the CAT-75 extraction).
# ---------------------------------------------------------------------------

GOLDEN_OPTION_STRINGS = {
    "generate_gumbel_selfplay_data": {
        ("--base-seed",),
        ("--belief-chance-spectra", "--no-belief-chance-spectra"),
        ("--information-set-search", "--no-information-set-search"),
        ("--native-mcts-hot-loop", "--no-native-mcts-hot-loop"),
        ("--determinization-particles",),
        ("--determinization-min-simulations",),
        ("--c-scale",),
        ("--c-visit",),
        ("--checkpoint",),
        ("--config",),
        ("--config-hash",),
        ("--config-purpose",),
        ("--correct-rust-chance-spectra", "--no-correct-rust-chance-spectra"),
        ("--device",),
        ("--dump-config",),
        ("--eval-cache-size",),
        ("--eval-server", "--no-eval-server"),
        ("--eval-server-batch-timeout-sec",),
        ("--eval-server-cuda-graph", "--no-eval-server-cuda-graph"),
        ("--eval-server-cuda-graph-batch-buckets",),
        ("--eval-server-cuda-graph-warmup-iterations",),
        ("--eval-server-event-token-limit",),
        ("--eval-server-local-fallback", "--no-eval-server-local-fallback"),
        ("--eval-server-max-batch",),
        ("--eval-server-max-neural-rows",),
        ("--eval-server-max-wait-ms",),
        ("--eval-server-matmul-precision",),
        ("--eval-server-request-collector", "--no-eval-server-request-collector"),
        ("--eval-server-shared-memory-slot-bytes",),
        ("--eval-server-timeout-ms",),
        ("--eval-server-transport",),
        ("--exact-budget-sh", "--no-exact-budget-sh"),
        ("--exact-budget-sh-min-n",),
        ("--exploiter-fraction",),
        ("--format",),
        ("--fleet-pipeline-id",),
        ("--fleet-pipeline-index",),
        ("--fleet-pipelines-per-gpu",),
        ("--games",),
        ("--generation-arm-id",),
        ("--help", "-h"),
        ("--late-temperature",),
        ("--late-temperature-decisions",),
        ("--lazy-interior-chance", "--no-lazy-interior-chance"),
        ("--ledger-claim-label",),
        ("--max-decisions",),
        ("--max-depth",),
        ("--n-fast",),
        ("--n-full",),
        ("--n-full-wide",),
        ("--n-full-wide-threshold",),
        ("--no-public-observation", "--public-observation"),
        (
            "--evaluator-rust-featurize",
            "--no-evaluator-rust-featurize",
            "--no-rust-featurize",
            "--rust-featurize",
        ),
        ("--no-score-actions", "--score-actions"),
        ("--no-resume", "--resume"),
        ("--no-seed-claim", "--seed-claim"),
        ("--obs-width",),
        ("--opponent-mix-manifest",),
        ("--opponent-pool-manifest",),
        ("--out-dir",),
        ("--p-full",),
        ("--prior-temperature",),
        ("--prelaunch-guard-config",),
        ("--raw-policy-above-width",),
        ("--rescale-noise-floor-c",),
        ("--no-root-wave-batching", "--root-wave-batching"),
        ("--shard-size",),
        ("--sigma-eval",),
        ("--skip-guards",),
        ("--no-symmetry-averaged-eval", "--symmetry-averaged-eval"),
        ("--symmetry-averaged-eval-threshold",),
        ("--temperature-decisions",),
        ("--temperature-high",),
        ("--temperature-low",),
        ("--temperature-move-fraction",),
        ("--track",),
        ("--value-scale",),
        ("--value-readout",),
        ("--vps-to-win",),
        ("--wide-candidates-threshold",),
        ("--no-wide-roots-always-full", "--wide-roots-always-full"),
        ("--workers",),
    },
    "train_bc": {
        ("--a1-ablation-code-binding-json",),
        ("--a1-ablation-code-tree-sha256",),
        ("--a1-batch-probe-plan",),
        ("--a1-batch-probe-run-id",),
        ("--a1-curriculum-parent-receipt",),
        ("--a1-dual-learner-lock",),
        ("--a1-dual-reviewed-lock-file-sha256",),
        ("--a1-effective-learner-recipe-json",),
        ("--a1-effective-learner-recipe-sha256",),
        ("--a1-learner-ablation-id",),
        ("--a1-reviewed-lock-file-sha256",),
        ("--advantage-policy-weighting",),
        ("--advantage-temperature",),
        ("--advantage-weight-cap",),
        ("--advantage-weight-floor",),
        ("--allow-concurrent-bc",),
        ("--allow-legacy-action-mask-upgrade",),
        ("--allow-missing-game-seed-validation-split",),
        ("--allow-teacher-score-q-loss",),
        ("--action-module-lr-mult",),
        ("--amp",),
        ("--arch",),
        ("--attention-heads",),
        ("--aux-subgoal-heads",),
        ("--aux-subgoal-loss-weight",),
        ("--batch-size",),
        ("--checkpoint",),
        ("--config",),
        ("--config-hash",),
        ("--config-purpose",),
        ("--data",),
        ("--data-format",),
        ("--data-loader-prefetch",),
        ("--data-loader-workers",),
        ("--ddp-find-unused-parameters", "--no-ddp-find-unused-parameters"),
        ("--ddp-shard-data",),
        ("--device",),
        ("--dump-config",),
        ("--edge-policy-head",),
        ("--entity-state-trunk",),
        ("--epochs",),
        ("--final-vp-loss-weight",),
        ("--forced-action-weight",),
        ("--forced-row-value-weight",),
        ("--freeze-modules",),
        ("--fsdp",),
        ("--fused-optimizer", "--no-fused-optimizer"),
        ("--grad-accum-steps",),
        ("--graph-dropout",),
        ("--graph-history-features",),
        ("--graph-layers",),
        ("--graph-tokens",),
        ("--grow-from-checkpoint",),
        ("--help", "-h"),
        ("--hlgauss-scalar-aux-loss-weight",),
        ("--hidden-size",),
        ("--host-lock-file",),
        ("--init-checkpoint",),
        ("--loser-sample-weight",),
        ("--latent-deliberation-slots",),
        ("--latent-deliberation-steps",),
        ("--lr",),
        ("--lr-schedule",),
        ("--lr-warmup-steps",),
        ("--mask-hidden-info",),
        ("--max-35m-params",),
        ("--max-steps",),
        ("--min-35m-params",),
        ("--moe-expert-ff-size",),
        ("--moe-balance-loss-weight",),
        ("--moe-routed-experts",),
        ("--moe-top-k",),
        ("--no-symmetry-augment", "--symmetry-augment"),
        ("--no-symmetry-augment-events", "--symmetry-augment-events"),
        ("--optimizer",),
        ("--per-game-policy-weight",),
        ("--per-game-policy-weight-mode",),
        ("--per-game-value-weight",),
        ("--per-game-value-weight-mode",),
        ("--phase-weights",),
        ("--policy-kl-anchor-direction",),
        ("--policy-kl-anchor-weight",),
        ("--policy-loss-weight",),
        ("--policy-surprise-cap",),
        ("--policy-surprise-weight",),
        ("--progress-every-batches",),
        ("--q-loss-weight",),
        ("--q-skip-teacher-prefixes",),
        ("--relational-action-cross-layers",),
        ("--relational-bases",),
        ("--relational-block-pattern",),
        ("--relational-ff-size",),
        ("--no-relational-edge-policy-head", "--relational-edge-policy-head"),
        ("--report",),
        ("--no-resume-optimizer", "--resume-optimizer"),
        ("--require-35m-model",),
        ("--require-production-35m-teacher",),
        ("--require-strict-35m-teacher",),
        ("--save-each-epoch",),
        ("--seed",),
        ("--skip-guards",),
        ("--skip-teacher-quality-gate",),
        ("--soft-target-min-legal-coverage",),
        ("--soft-target-source",),
        ("--soft-target-temperature",),
        ("--soft-target-weight",),
        ("--teacher-weights",),
        ("--track",),
        ("--train-diagnostics-every-batches",),
        ("--train-value-only",),
        ("--truncated-vp-margin-value-weight",),
        ("--trust-curated-data-quality",),
        ("--validation-fraction",),
        ("--validation-game-seed-manifest",),
        ("--validation-game-seed-ranges",),
        ("--validation-max-samples",),
        ("--validation-seed",),
        ("--value-categorical-loss-weight",),
        ("--value-categorical-bins",),
        ("--value-head-type",),
        ("--value-hlgauss-sigma-ratio",),
        ("--value-loss-weight",),
        ("--value-lr-mult",),
        ("--value-phase-weights",),
        ("--value-target-lambda",),
        ("--value-uncertainty-head",),
        ("--value-uncertainty-loss-weight",),
        ("--vp-margin-weight",),
        ("--vps-to-win",),
        ("--weight-decay",),
        ("--winner-sample-weight",),
    },
    "continuous_flywheel": {
        ("--anchor-corpus",),
        ("--anchor-drift-alert-threshold",),
        ("--anchor-eval-every-rounds",),
        ("--anchor-holdout-ranges",),
        ("--base-seed",),
        ("--batch-size",),
        ("--champion-registry",),
        ("--device",),
        ("--dry-run",),
        ("--evict-grace-seconds",),
        ("--evict-stale-shards",),
        ("--games-per-round",),
        ("--gate-games",),
        ("--gate-baseline-value-readout",),
        ("--gate-candidate-value-readout",),
        ("--gen-c-scale",),
        ("--gen-c-visit",),
        ("--gen-correct-rust-chance-spectra", "--no-gen-correct-rust-chance-spectra"),
        ("--gen-determinization-min-simulations",),
        ("--gen-determinization-particles",),
        ("--gen-information-set-search", "--no-gen-information-set-search"),
        ("--gen-lazy-interior-chance", "--no-gen-lazy-interior-chance"),
        ("--gen-max-decisions",),
        ("--gen-max-depth",),
        ("--gen-n-fast",),
        ("--gen-n-full",),
        ("--gen-n-full-wide",),
        ("--gen-n-full-wide-threshold",),
        ("--gen-out-root",),
        ("--gen-p-full",),
        ("--gen-symmetry-averaged-eval", "--no-gen-symmetry-averaged-eval"),
        ("--gen-symmetry-averaged-eval-threshold",),
        ("--gen-temperature-decisions",),
        ("--gen-wide-candidates-threshold",),
        ("--gen-wide-roots-always-full", "--no-gen-wide-roots-always-full"),
        ("--help", "-h"),
        ("--loop-dir",),
        ("--max-rounds",),
        ("--opponent-pool-fraction",),
        ("--regime",),
        ("--seed-checkpoint",),
        ("--skip-guards",),
        ("--window-c-rows",),
        ("--workers",),
    },
}


@pytest.fixture(autouse=True)
def _stable_host_fd_limit(monkeypatch):
    """Keep launcher wiring tests independent of the pytest shell's ulimit.

    The fd-limit guard's pass/fail behavior is covered directly in
    test_prelaunch_guard.py.  This module verifies launcher wiring, so its
    allowed-path cases should not fail merely because CI starts pytest with a
    lower host-only soft limit.
    """
    original_getrlimit = resource.getrlimit

    def getrlimit(limit: int) -> tuple[int, int]:
        if limit == resource.RLIMIT_NOFILE:
            return 65536, 65536
        return original_getrlimit(limit)

    monkeypatch.setattr(resource, "getrlimit", getrlimit)


def _option_strings(parser: argparse.ArgumentParser) -> set[tuple[str, ...]]:
    return {
        tuple(sorted(action.option_strings))
        for action in parser._actions  # noqa: SLF001
        if action.option_strings
    }


@pytest.mark.parametrize(
    "module, launcher",
    [
        (gen_cli, "generate_gumbel_selfplay_data"),
        (train_bc, "train_bc"),
        (continuous_flywheel, "continuous_flywheel"),
    ],
)
def test_build_parser_option_strings_match_golden_set(module, launcher):
    parser = module.build_parser()
    assert _option_strings(parser) == GOLDEN_OPTION_STRINGS[launcher]


@pytest.mark.parametrize(
    "module",
    [gen_cli, train_bc, continuous_flywheel],
)
def test_build_parser_is_import_safe_and_side_effect_free(module):
    """Calling build_parser() twice must not raise or leak state -- pure
    add_argument calls, safe to import/introspect without a real launch."""
    parser_a = module.build_parser()
    parser_b = module.build_parser()
    assert parser_a is not parser_b
    assert _option_strings(parser_a) == _option_strings(parser_b)


def test_generate_gumbel_sample_argv_produces_expected_namespace():
    parser = gen_cli.build_parser()
    args = parser.parse_args(
        [
            "--out-dir",
            "/tmp/does-not-matter",
            "--base-seed",
            "1234",
            "--games",
            "8",
            "--c-scale",
            "0.1",
        ]
    )
    assert args.out_dir == "/tmp/does-not-matter"
    assert args.base_seed == 1234
    assert args.games == 8
    assert args.c_scale == pytest.approx(0.1)
    # Untouched defaults must be unaffected by the extraction.
    assert args.n_full == 64
    assert args.n_fast == 16
    assert args.seed_claim is True
    assert args.skip_guards is False


def test_train_bc_sample_argv_produces_expected_namespace(tmp_path):
    parser = train_bc.build_parser()
    args = parser.parse_args(
        [
            "--data",
            str(tmp_path / "data"),
            "--checkpoint",
            str(tmp_path / "ckpt.pt"),
            "--report",
            str(tmp_path / "report.json"),
            "--optimizer",
            "adamw",
            "--weight-decay",
            "0.01",
        ]
    )
    assert args.data == str(tmp_path / "data")
    assert args.optimizer == "adamw"
    assert args.weight_decay == pytest.approx(0.01)
    # Untouched defaults must be unaffected by the extraction.
    assert args.epochs == 2
    assert args.batch_size == 65536
    assert args.truncated_vp_margin_value_weight == pytest.approx(0.25)
    assert args.skip_guards is False


def test_continuous_flywheel_sample_argv_produces_expected_namespace(tmp_path):
    parser = continuous_flywheel.build_parser()
    args = parser.parse_args(
        [
            "--loop-dir",
            str(tmp_path / "loop"),
            "--seed-checkpoint",
            str(tmp_path / "seed.pt"),
            "--dry-run",
        ]
    )
    assert args.loop_dir == str(tmp_path / "loop")
    assert args.dry_run is True
    # Untouched defaults must be unaffected by the extraction.
    assert args.regime == "continuous"
    assert args.batch_size == 65536
    assert args.gen_information_set_search is None
    assert args.gen_determinization_particles is None
    assert args.gen_determinization_min_simulations is None
    assert args.skip_guards is False


@pytest.mark.parametrize(
    "module, launcher",
    [
        (gen_cli, "generate_gumbel_selfplay_data"),
        (train_bc, "train_bc"),
        (continuous_flywheel, "continuous_flywheel"),
    ],
)
def test_static_guard_config_critical_flags_are_not_stale(module, launcher):
    """The committed configs/guards/<launcher>.json critical_flags and any
    expected_values keys must all be real optional flags on the CURRENT parser
    -- this is exactly the "stale hardcoded flag list" failure mode
    guard_cli_flag_lint's parser cross-check exists to catch (CAT-69 review
    comment bd5b777)."""
    parser = module.build_parser()
    for spec in launcher_guards.load_static_guard_specs(launcher):
        if spec["name"] != "cli_flag_lint":
            continue
        critical_flags = list(spec["args"]["critical_flags"])
        for flag in spec["args"].get("expected_values", {}):
            if flag not in critical_flags:
                critical_flags.append(flag)
        result = prelaunch_guard.guard_cli_flag_lint(
            argv=critical_flags,  # every critical flag present -> nothing "missing"
            critical_flags=critical_flags,
            parser=parser,
        )
        assert result.passed, result.reason


# ---------------------------------------------------------------------------
# Part 2: guards are actually wired -- an intentionally-bad launch config is
# refused end-to-end (SystemExit) via each launcher's guard-spec-building
# helper + launcher_guards.run_or_refuse, without ever running a real
# generation/training/gate launch.
# ---------------------------------------------------------------------------


def _write_fake_checkpoint(path: Path, *, mask_hidden_info: bool) -> None:
    import torch

    torch.save({"mask_hidden_info": mask_hidden_info, "model": {}}, path)


def _write_generation_manifest(data_dir: Path, *, base_seed: int, games: int) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "manifest.json").write_text(
        json.dumps({"base_seed": base_seed, "games_requested": games})
    )


class TestGenerateGumbelSelfplayGuardWiring:
    # Production recipe flags that the generate guard config now value-checks.
    # Each test that should pass cli_flag_lint must include these (or skip them
    # intentionally to test a guard failure).
    _GEN_RECIPE_FLAGS = [
        "--c-scale",
        "0.03",
        "--c-visit",
        "50.0",
        "--n-full",
        "128",
        "--n-fast",
        "16",
        "--p-full",
        "0.25",
        "--max-depth",
        "80",
        "--temperature-decisions",
        "90",
        "--public-observation",
        "--lazy-interior-chance",
        "--symmetry-averaged-eval",
        "--symmetry-averaged-eval-threshold",
        "20",
        "--no-belief-chance-spectra",
        "--information-set-search",
        "--native-mcts-hot-loop",
        "--determinization-particles",
        "4",
        "--determinization-min-simulations",
        "32",
    ]

    def test_seed_ledger_collision_refuses_launch(self, tmp_path):
        claims_dir = tmp_path / ".seed_claims"
        claims_dir.mkdir()
        out_dir_a = tmp_path / "run_a"
        out_dir_a.mkdir()
        (claims_dir / "run_a.json").write_text(
            json.dumps(
                {"out_dir": str(out_dir_a.resolve()), "base_seed": 1000, "games": 64}
            )
        )
        out_dir_b = tmp_path / "run_b"
        parser = gen_cli.build_parser()
        argv = [
            "--out-dir",
            str(out_dir_b),
            "--base-seed",
            "1032",
            "--games",
            "64",
        ] + self._GEN_RECIPE_FLAGS
        args = parser.parse_args(argv)
        specs = gen_cli._build_guard_specs(args, argv, parser)
        # Point seed_ledger at the fixture claims dir, not the real default location.
        for spec in specs:
            if spec["name"] == "seed_ledger":
                spec["args"]["claims_dir"] = claims_dir
        with pytest.raises(SystemExit, match="seed_ledger"):
            launcher_guards.run_or_refuse(
                specs, launcher="generate_gumbel_selfplay_data", skip=False
            )

    def test_val_only_range_is_allowed_for_generation(self, tmp_path, monkeypatch):
        # This test exercises the documented VAL-ONLY exception when no canonical
        # cross-host ledger is available.  Isolate it from any operator ledger in
        # the runner's home directory; a real, overlapping ledger must still win
        # and refuse the launch.
        monkeypatch.setenv("CATAN_SEED_LEDGER", str(tmp_path / "missing-ledger.md"))
        parser = gen_cli.build_parser()
        argv = [
            "--out-dir",
            str(tmp_path / "run"),
            "--base-seed",
            str(VAL_ONLY_SEED),
            "--games",
            "8",
        ] + self._GEN_RECIPE_FLAGS
        args = parser.parse_args(argv)
        specs = gen_cli._build_guard_specs(args, argv, parser)
        launcher_guards.run_or_refuse(
            specs, launcher="generate_gumbel_selfplay_data", skip=False
        )

    def test_missing_critical_flag_refuses_launch(self, tmp_path):
        parser = gen_cli.build_parser()
        # --c-scale omitted -- the CLI-default-override trap this guard exists for.
        argv = ["--out-dir", str(tmp_path / "run"), "--base-seed", "1", "--games", "8"]
        args = parser.parse_args(argv)
        specs = gen_cli._build_guard_specs(args, argv, parser)
        with pytest.raises(SystemExit, match="cli_flag_lint"):
            launcher_guards.run_or_refuse(
                specs, launcher="generate_gumbel_selfplay_data", skip=False
            )

    def test_skip_guards_bypasses_refusal_with_warning(self, tmp_path, capsys):
        parser = gen_cli.build_parser()
        argv = ["--out-dir", str(tmp_path / "run"), "--base-seed", "1", "--games", "8"]
        args = parser.parse_args(argv)
        specs = gen_cli._build_guard_specs(args, argv, parser)
        launcher_guards.run_or_refuse(
            specs, launcher="generate_gumbel_selfplay_data", skip=True
        )
        assert "WARNING" in capsys.readouterr().err

    def _ledger_overlap_own_claim(self, tmp_path, argv_extra, env_claim, monkeypatch):
        """Helper: build specs and return the ledger_overlap guard's own_claim_label."""
        if env_claim is None:
            monkeypatch.delenv("CATAN_LEDGER_CLAIM_ID", raising=False)
        else:
            monkeypatch.setenv("CATAN_LEDGER_CLAIM_ID", env_claim)
        parser = gen_cli.build_parser()
        argv = (
            [
                "--out-dir",
                str(tmp_path / "run"),
                "--base-seed",
                "1",
                "--games",
                "8",
            ]
            + self._GEN_RECIPE_FLAGS
            + argv_extra
        )
        args = parser.parse_args(argv)
        specs = gen_cli._build_guard_specs(args, argv, parser)
        overlap = next(s for s in specs if s["name"] == "ledger_overlap")
        return overlap["args"]["own_claim_label"]

    def test_ledger_claim_label_flag_threads_into_overlap_guard(
        self, tmp_path, monkeypatch
    ):
        # CAT-124: --ledger-claim-label reaches the ledger_overlap guard so it can
        # exclude this launch's own just-written claim row.
        got = self._ledger_overlap_own_claim(
            tmp_path, ["--ledger-claim-label", "c2-teacher-w1-42"], None, monkeypatch
        )
        assert got == "c2-teacher-w1-42"

    def test_ledger_claim_label_flag_overrides_env(self, tmp_path, monkeypatch):
        # Explicit flag wins over the launcher's $CATAN_LEDGER_CLAIM_ID export.
        got = self._ledger_overlap_own_claim(
            tmp_path, ["--ledger-claim-label", "flag-id"], "env-id", monkeypatch
        )
        assert got == "flag-id"

    def test_ledger_claim_label_falls_back_to_env(self, tmp_path, monkeypatch):
        # No flag -> use the launcher's exported id.
        got = self._ledger_overlap_own_claim(tmp_path, [], "env-id", monkeypatch)
        assert got == "env-id"

    def test_ledger_claim_label_unset_is_none(self, tmp_path, monkeypatch):
        # Neither flag nor env -> None -> prior fail-closed-on-any-overlap behavior.
        got = self._ledger_overlap_own_claim(tmp_path, [], None, monkeypatch)
        assert got is None


class TestTrainBcGuardWiring:
    def test_val_only_seed_range_in_train_purpose_is_refused_end_to_end(self, tmp_path):
        """The exact scenario named in CAT-75's verification section: a train
        invocation whose corpus was generated from a VAL-ONLY seed range must
        be refused, discovered purely from the generation manifest.json --
        train_bc.py itself never takes a --base-seed."""
        data_dir = tmp_path / "corpus"
        _write_generation_manifest(data_dir, base_seed=VAL_ONLY_SEED, games=1000)

        parser = train_bc.build_parser()
        argv = [
            "--data",
            str(data_dir),
            "--checkpoint",
            str(tmp_path / "ckpt.pt"),
            "--report",
            str(tmp_path / "report.json"),
            "--mask-hidden-info",
            "--optimizer",
            "adam",
            "--weight-decay",
            "0.0",
            "--truncated-vp-margin-value-weight",
            "0.25",
            "--lr-schedule",
            "flat",
        ]
        args = parser.parse_args(argv)
        specs = train_bc._build_guard_specs(args, argv, parser)
        with pytest.raises(SystemExit, match="val_only_never_trains"):
            launcher_guards.run_or_refuse(specs, launcher="train_bc", skip=False)

    def test_disjoint_seed_range_corpus_is_not_refused(self, tmp_path):
        data_dir = tmp_path / "corpus"
        _write_generation_manifest(data_dir, base_seed=1000, games=1000)

        parser = train_bc.build_parser()
        argv = [
            "--data",
            str(data_dir),
            "--checkpoint",
            str(tmp_path / "ckpt.pt"),
            "--report",
            str(tmp_path / "report.json"),
            "--mask-hidden-info",
            "--optimizer",
            "adam",
            "--weight-decay",
            "0.0",
            "--truncated-vp-margin-value-weight",
            "0.25",
            "--lr-schedule",
            "flat",
        ]
        args = parser.parse_args(argv)
        specs = train_bc._build_guard_specs(args, argv, parser)
        launcher_guards.run_or_refuse(specs, launcher="train_bc", skip=False)

    def test_init_checkpoint_masked_regime_mismatch_is_refused(self, tmp_path):
        init_checkpoint = tmp_path / "init.pt"
        _write_fake_checkpoint(init_checkpoint, mask_hidden_info=False)
        data_dir = tmp_path / "corpus"
        _write_generation_manifest(data_dir, base_seed=1000, games=1000)

        parser = train_bc.build_parser()
        argv = [
            "--data",
            str(data_dir),
            "--checkpoint",
            str(tmp_path / "ckpt.pt"),
            "--report",
            str(tmp_path / "report.json"),
            "--init-checkpoint",
            str(init_checkpoint),
            "--mask-hidden-info",
            "--optimizer",
            "adam",
            "--weight-decay",
            "0.0",
            "--truncated-vp-margin-value-weight",
            "0.25",
            "--lr-schedule",
            "flat",
        ]
        args = parser.parse_args(argv)
        specs = train_bc._build_guard_specs(args, argv, parser)
        with pytest.raises(SystemExit, match="masked_regime"):
            launcher_guards.run_or_refuse(specs, launcher="train_bc", skip=False)

    def test_missing_critical_flag_refuses_launch(self, tmp_path):
        data_dir = tmp_path / "corpus"
        _write_generation_manifest(data_dir, base_seed=1000, games=1000)
        parser = train_bc.build_parser()
        # --weight-decay/--optimizer/etc all omitted (silently defaulted).
        argv = [
            "--data",
            str(data_dir),
            "--checkpoint",
            str(tmp_path / "ckpt.pt"),
            "--report",
            str(tmp_path / "report.json"),
        ]
        args = parser.parse_args(argv)
        specs = train_bc._build_guard_specs(args, argv, parser)
        with pytest.raises(SystemExit, match="cli_flag_lint"):
            launcher_guards.run_or_refuse(specs, launcher="train_bc", skip=False)

    def test_skip_guards_bypasses_refusal_with_warning(self, tmp_path, capsys):
        data_dir = tmp_path / "corpus"
        _write_generation_manifest(data_dir, base_seed=VAL_ONLY_SEED, games=1000)
        parser = train_bc.build_parser()
        argv = [
            "--data",
            str(data_dir),
            "--checkpoint",
            str(tmp_path / "ckpt.pt"),
            "--report",
            str(tmp_path / "report.json"),
        ]
        args = parser.parse_args(argv)
        specs = train_bc._build_guard_specs(args, argv, parser)
        launcher_guards.run_or_refuse(specs, launcher="train_bc", skip=True)
        assert "WARNING" in capsys.readouterr().err


class TestContinuousFlywheelGuardWiring:
    def test_seed_checkpoint_masked_regime_mismatch_is_refused(self, tmp_path):
        seed_checkpoint = tmp_path / "seed.pt"
        _write_fake_checkpoint(seed_checkpoint, mask_hidden_info=False)
        loop_dir = tmp_path / "loop"

        parser = continuous_flywheel.build_parser()
        argv = [
            "--loop-dir",
            str(loop_dir),
            "--seed-checkpoint",
            str(seed_checkpoint),
            "--batch-size",
            "65536",
            "--games-per-round",
            "2000",
            "--gate-games",
            "150",
        ]
        args = parser.parse_args(argv)
        loop_dir.mkdir(parents=True, exist_ok=True)
        specs = continuous_flywheel._build_guard_specs(args, argv, parser, loop_dir)
        with pytest.raises(SystemExit, match="masked_regime"):
            launcher_guards.run_or_refuse(
                specs, launcher="continuous_flywheel", skip=False
            )

    def test_matching_masked_regime_is_not_refused(self, tmp_path):
        seed_checkpoint = tmp_path / "seed.pt"
        _write_fake_checkpoint(seed_checkpoint, mask_hidden_info=True)
        loop_dir = tmp_path / "loop"

        parser = continuous_flywheel.build_parser()
        argv = [
            "--loop-dir",
            str(loop_dir),
            "--seed-checkpoint",
            str(seed_checkpoint),
            "--batch-size",
            "65536",
            "--games-per-round",
            "2000",
            "--gate-games",
            "150",
        ]
        args = parser.parse_args(argv)
        loop_dir.mkdir(parents=True, exist_ok=True)
        specs = continuous_flywheel._build_guard_specs(args, argv, parser, loop_dir)
        launcher_guards.run_or_refuse(specs, launcher="continuous_flywheel", skip=False)

    def test_resumed_config_provenance_drift_is_refused(self, tmp_path):
        seed_checkpoint = tmp_path / "seed.pt"
        _write_fake_checkpoint(seed_checkpoint, mask_hidden_info=True)
        loop_dir = tmp_path / "loop"
        loop_dir.mkdir(parents=True, exist_ok=True)
        # Simulate a prior run that recorded regime="continuous".
        (loop_dir / "flywheel_config.json").write_text(
            json.dumps({"regime": "continuous"})
        )

        parser = continuous_flywheel.build_parser()
        argv = [
            "--loop-dir",
            str(loop_dir),
            "--seed-checkpoint",
            str(seed_checkpoint),
            "--regime",
            "discrete",  # drifts from the recorded "continuous"
            "--batch-size",
            "65536",
            "--games-per-round",
            "2000",
            "--gate-games",
            "150",
        ]
        args = parser.parse_args(argv)
        specs = continuous_flywheel._build_guard_specs(args, argv, parser, loop_dir)
        with pytest.raises(SystemExit, match="provenance"):
            launcher_guards.run_or_refuse(
                specs, launcher="continuous_flywheel", skip=False
            )

    def test_fresh_loop_has_no_provenance_guard(self, tmp_path):
        """A brand-new --loop-dir has no flywheel_config.json yet -- the
        provenance guard must not be added at all (it would otherwise FAIL on
        a merely-absent report, which is the wrong verdict for a fresh start)."""
        seed_checkpoint = tmp_path / "seed.pt"
        _write_fake_checkpoint(seed_checkpoint, mask_hidden_info=True)
        loop_dir = tmp_path / "loop"
        loop_dir.mkdir(parents=True, exist_ok=True)

        parser = continuous_flywheel.build_parser()
        argv = ["--loop-dir", str(loop_dir), "--seed-checkpoint", str(seed_checkpoint)]
        args = parser.parse_args(argv)
        specs = continuous_flywheel._build_guard_specs(args, argv, parser, loop_dir)
        assert not any(spec["name"] == "provenance" for spec in specs)

    def test_dry_run_never_reaches_guard_wiring(self, tmp_path):
        """--dry-run must not construct/run any guard -- it legitimately points
        --seed-checkpoint at a fixture path a real masked_regime guard would
        correctly refuse (here: not even a valid torch checkpoint, just bytes;
        --dry-run's seed_champion only copies the file, never torch.loads it)."""
        loop_dir = tmp_path / "loop"
        fake_checkpoint = tmp_path / "not_a_real_checkpoint.pt"
        fake_checkpoint.write_bytes(b"not a real torch checkpoint")
        argv = [
            "--loop-dir",
            str(loop_dir),
            "--seed-checkpoint",
            str(fake_checkpoint),
            "--dry-run",
            "--max-rounds",
            "0",
        ]
        exit_code = continuous_flywheel.main(argv)
        assert exit_code == 0

    def test_skip_guards_bypasses_refusal_with_warning(self, tmp_path, capsys):
        seed_checkpoint = tmp_path / "seed.pt"
        _write_fake_checkpoint(seed_checkpoint, mask_hidden_info=False)
        loop_dir = tmp_path / "loop"
        loop_dir.mkdir(parents=True, exist_ok=True)

        parser = continuous_flywheel.build_parser()
        argv = ["--loop-dir", str(loop_dir), "--seed-checkpoint", str(seed_checkpoint)]
        args = parser.parse_args(argv)
        specs = continuous_flywheel._build_guard_specs(args, argv, parser, loop_dir)
        launcher_guards.run_or_refuse(specs, launcher="continuous_flywheel", skip=True)
        assert "WARNING" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# --help must never trigger a guard (argparse exits during parse_args itself,
# before any launcher's main() body -- including the guard wiring -- runs).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module, extra_argv",
    [
        (gen_cli, ["--out-dir", "/tmp/x"]),
        (
            train_bc,
            [
                "--data",
                "/tmp/x",
                "--checkpoint",
                "/tmp/y.pt",
                "--report",
                "/tmp/z.json",
            ],
        ),
        (
            continuous_flywheel,
            ["--loop-dir", "/tmp/x", "--seed-checkpoint", "/tmp/y.pt"],
        ),
    ],
)
def test_help_exits_before_any_guard_runs(module, extra_argv, monkeypatch):
    calls = []
    monkeypatch.setattr(
        launcher_guards, "run_or_refuse", lambda *a, **k: calls.append((a, k))
    )
    with pytest.raises(SystemExit) as exc_info:
        module.main(extra_argv + ["--help"])
    assert exc_info.value.code == 0
    assert calls == []


def test_train_bc_guards_run_before_expensive_memmap_preflight(monkeypatch):
    """A cheap launch refusal must not first scan the entire corpus payload."""

    class GuardRefused(RuntimeError):
        pass

    preflight_calls: list[object] = []
    monkeypatch.setattr(
        train_bc,
        "_coordinated_a1_memmap_preflight",
        lambda *args, **kwargs: preflight_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        launcher_guards,
        "run_or_refuse",
        lambda *args, **kwargs: (_ for _ in ()).throw(GuardRefused()),
    )

    with pytest.raises(GuardRefused):
        train_bc.main(
            [
                "--data",
                "/does/not/need/to/exist",
                "--data-format",
                "memmap",
                "--checkpoint",
                "/does/not/need/to/exist.pt",
                "--report",
                "/does/not/need/to/exist.json",
            ]
        )

    assert preflight_calls == []


def test_train_bc_checkpoint_topology_runs_before_memmap_preflight(monkeypatch):
    """A warm-start topology mismatch must not first hash the data payload."""

    class TopologyRefused(RuntimeError):
        pass

    preflight_calls: list[object] = []
    monkeypatch.setattr(launcher_guards, "run_or_refuse", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_bc, "_resolve_effective_value_categorical_bins", lambda args: 0
    )
    monkeypatch.setattr(
        train_bc,
        "_preflight_init_checkpoint_architecture",
        lambda *args, **kwargs: (_ for _ in ()).throw(TopologyRefused()),
    )
    monkeypatch.setattr(
        train_bc,
        "_coordinated_a1_memmap_preflight",
        lambda *args, **kwargs: preflight_calls.append((args, kwargs)),
    )

    with pytest.raises(TopologyRefused):
        train_bc.main(
            [
                "--arch",
                "entity_graph",
                "--data",
                "/does/not/need/to/exist",
                "--data-format",
                "memmap",
                "--init-checkpoint",
                "/does/not/need/to/exist.pt",
                "--checkpoint",
                "/does/not/need/to/exist-output.pt",
                "--report",
                "/does/not/need/to/exist.json",
            ]
        )

    assert preflight_calls == []
