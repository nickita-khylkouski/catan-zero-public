"""Pure, Modal-free resume-decision logic for the Gumbel Modal factories.

Split out of `modal_gumbel_factory_gpu.py` (which pulls in the `modal` SDK at
import time, making it unimportable in a plain local Python environment)
specifically so this logic is unit-testable without Modal, CUDA, or the
compiled `catanatron_rs` Rust engine -- see `tests/test_gumbel_resume.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_part_resume_action(
    *,
    part_dir: Path,
    manifest_path: Path,
    marker_path: Path,
    run_id: str,
    resume: bool,
) -> tuple[str, dict[str, Any] | None]:
    """Decide what `gpu_part_worker` should do about an existing `part_dir`.

    Pure/side-effect-free apart from reading the two small marker files, so
    it is unit-testable without the Modal runtime (no volume, no GPU, no
    `modal` app context). Returns `(action, complete_manifest)`:

      - "return_complete": the part already finished; `complete_manifest`
        is the parsed `manifest.json` to hand back as-is (idempotent
        no-op, unchanged from before).
      - "incremental_resume": a SAME-run_id reinvocation after a Modal
        preemption/retry. MUST NOT wipe `part_dir` -- the caller instead
        resumes via `run_worker_games(resume=True, run_id=run_id)`, which
        only ever replays games not yet confirmed durably flushed (see
        `catan_zero.rl.gumbel_self_play.WorkerProgress`). This is the fix:
        the old code took the SAME `shutil.rmtree(part_dir)` path here as
        "wipe_and_restart" below, which is what erased every completed
        game on every preemption retry.
      - "wipe_and_restart": an operator explicitly passed `resume=True`
        (the CLI's documented "fill failed/missing parts" flag) to
        force-redo an INCOMPLETE part left by a DIFFERENT (stale/foreign)
        run_id. This is a deliberate, explicit operator override -- not an
        automatic same-run retry -- so wiping remains correct here: there
        is no guarantee a foreign run's partial output used a compatible
        payload/config to resume into.
      - "fresh": `part_dir` doesn't exist yet (first launch of this part).

    Raises `RuntimeError` (byte-for-byte the pre-existing message) if
    `part_dir` holds output from a DIFFERENT run_id and `resume=False` --
    the duplicate-launch guard that caught the prior 40-container
    seed-overlap incident. This guard is completely untouched by the
    preemption fix: it only fires when `own_partial` is False and the
    explicit `resume` override is also False.
    """
    if not part_dir.exists():
        return "fresh", None

    if manifest_path.exists():
        complete = json.loads(manifest_path.read_text(encoding="utf-8"))
        if resume or (run_id and str(complete.get("run_id", "")) == run_id):
            return "return_complete", complete

    own_partial = marker_path.exists() and marker_path.read_text(
        encoding="utf-8"
    ).strip() == run_id
    if own_partial:
        return "incremental_resume", None
    if resume:
        return "wipe_and_restart", None
    if any(part_dir.iterdir()):
        raise RuntimeError(
            f"{part_dir} already contains output from a different run_id; "
            "use a fresh run_name or pass resume=True."
        )
    return "fresh", None
