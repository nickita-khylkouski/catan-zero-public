# CAT-129 test gate — collection quarantine.
#
# This test module errors at collection time on a clean canonical env
# (they'd interrupt the whole run), for reasons unrelated to the code under
# test. They are quarantined here — in ONE documented place — so `pytest tests/`
# is truly 0-error (not "0-fail-but-N-collection-errors"). Everything else runs.
#
# 1. tests/test_concat_memmap_corpus.py  (6 errors)
#    Exercises `ConcatMemmapCorpus`, which lives in the QUARANTINED CAT-94
#    window-feed work and is absent from the runsix/run6-consolidated tree.
#    Pre-existed on 2a17d84. UNQUARANTINE when CAT-94 ConcatMemmapCorpus is
#    wired into the canonical tree.
#
# It is also passed as explicit --ignore by scripts/gate.sh so the gate is
# self-contained even if run without this conftest on PYTHONPATH.

import pytest


collect_ignore = [
    "tests/test_concat_memmap_corpus.py",
]


def _reset_train_bc_process_globals() -> None:
    """Keep in-process trainer tests from inheriting prior CLI state."""
    import sys

    for name in ("tools.train_bc", "train_bc"):
        module = sys.modules.get(name)
        if module is None:
            continue
        module._MASK_HIDDEN_INFO_PLAYER_TOKENS = False
        module._CROP_AUTHENTICATED_EMPTY_EVENT_HISTORY = False


@pytest.fixture(autouse=True)
def _isolate_train_bc_process_globals():
    """Restore CLI-owned module state around every in-process test."""
    _reset_train_bc_process_globals()
    try:
        yield
    finally:
        _reset_train_bc_process_globals()
