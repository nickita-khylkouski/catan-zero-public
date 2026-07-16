# Local branch archive — 2026-07-15

The local repository contained 57 branch tips whose commits were not reachable
from any pre-existing `origin` ref. They were preserved on GitHub under:

```text
archive/20260715/<original-local-branch-name>
```

Examples:

```text
archive/20260715/agent/learner-dose-telemetry
archive/20260715/agent/target-reliability-20260715
archive/20260715/codex/static-action-residual
archive/20260715/codex/opening-road-d1
```

List the complete archive with:

```bash
git ls-remote --heads origin 'refs/heads/archive/20260715/*'
```

Before publication, `gitleaks` scanned 202 commits reachable from local refs
but not the former remote refs. It reported zero findings.

## Interpretation

These refs are backups and research provenance, not approved integrations.
They contain contradictory and superseded treatments. Do not merge the archive
namespace wholesale.

The canonical collaboration line is:

```text
agent/rl-system-repair-20260715
```

That branch starts from `agent/integrate-safe-rl-fixes-latest` and contains the
curated current diagnosis, repair plan, machine-readable findings, preserved
evaluator work, and selectively salvaged evidence/code.

## Curated salvage decisions

Integrated or being ported into the canonical collaboration branch:

- current-parent Stage-C replication;
- native H2H evaluator integration;
- July-15 observation/value/PPO diagnosis and work packages;
- opponent provenance through memmap/training;
- distributed high-regret evaluation;
- short-dose commissioning and topology-cost evidence;
- compact 100-experiment R&D summaries.

Archived but not made canonical:

- older candidate-chained learner campaigns;
- stale PIMC/belief grids;
- disabled native uncertainty-backup experiments;
- old n256 replication paths;
- obsolete large `train_bc` worktrees;
- raw R&D logs and generated artifacts.

When salvaging from an archive ref, port the logic onto the current branch
rather than cherry-picking a large historical commit range.
