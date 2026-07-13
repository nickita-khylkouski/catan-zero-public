"""CAT-134 regression guard for the Modal L4 GPU factory container cap.

Two invariants, checked at the SOURCE level so this needs no `modal` SDK and
runs in every venv / CI:

1. `tools/modal_gumbel_factory_gpu.py` must cap `max_containers` at or under the
   hard GPU cap (44 = a 1-GPU safety margin under the standing "Modal must STAY
   UNDER 45 L4s ALWAYS" user constraint). A `max_containers=100` regressed this
   and, combined with a second concurrent app, caused a 50-container breach on
   2026-07-06; the fix had been stranded in an orphaned `tools/modal/` copy.
2. The `tools/modal/` package must stay deleted: besides stranding the fixed cap,
   it shadowed the real `modal` SDK (`import modal` → AttributeError at import)
   whenever `tools/` was placed at the front of `sys.path`.

CAVEAT (documented, not enforceable in source): `max_containers` is a PER-app /
per-Function limit. Modal queues excess inputs rather than exceeding it, so ONE
app holds at <=44 GPUs. Running TWO concurrent ephemeral apps = two pools = up to
2x the cap (~88), which breaches 45 — that combo caused the 2026-07-06 incident.
The operational rule is "run ONE wave (one app) at a time"; a hard cross-app guard
(if ever wanted) needs external coordination and is out of CAT-134 scope.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_GPU_FACTORY = _REPO / "tools" / "modal_gumbel_factory_gpu.py"

MODAL_L4_HARD_CAP = 44  # 1-GPU margin under the <45 constraint


def test_gpu_factory_caps_containers_at_or_under_hard_cap():
    src = _GPU_FACTORY.read_text(encoding="utf-8")
    caps = [int(n) for n in re.findall(r"max_containers\s*=\s*(\d+)", src)]
    assert caps, "no `max_containers=<int>` found in the GPU factory"
    assert all(c <= MODAL_L4_HARD_CAP for c in caps), (
        f"max_containers exceeds the {MODAL_L4_HARD_CAP} hard cap "
        f"(Modal <45 L4s; 2026-07-06 breach): found {caps}"
    )
    # the live decorator must set exactly the hard cap, not just something smaller
    assert MODAL_L4_HARD_CAP in caps, (
        f"expected the decorator to set max_containers={MODAL_L4_HARD_CAP}; found {caps}"
    )


def test_tools_modal_shadow_package_stays_deleted():
    shadow = _REPO / "tools" / "modal"
    assert not shadow.exists(), (
        f"{shadow} re-introduces the Modal-SDK-shadowing package removed in CAT-134"
    )


def test_gpu_factory_resume_contract_binds_checkpoint_bytes_and_recipe():
    src = _GPU_FACTORY.read_text(encoding="utf-8")
    assert 'science_payload["producer_checkpoint_sha256"] = _file_sha256(checkpoint)' in src
    assert '"resume_semantics_sha256": _resume_semantics_sha256(' in src
    assert 'resume_semantics_sha256=str(worker_args["resume_semantics_sha256"])' in src
