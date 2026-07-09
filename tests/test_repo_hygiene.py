"""Repository hygiene guards (task #75).

Born from a real incident (2026-07-05): src/catan_zero/data was an untracked,
gitignored package, so every fresh worktree/clone failed test collection on
`import catan_zero.data`. Engineers worked around it with worktree-local
symlinks, and because the ignore rules were trailing-slash DIRECTORY patterns
(which do not match a symlink), `git add -A` swept a self-referential
absolute-path symlink into two feature branches; merging them clobbered the
real package on master twice (fixed in b3493e5 / 981dce8). These tests keep
both failure modes dead.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_colonist_package_is_tracked():
    tracked = _git("ls-files", "src/catan_zero/data").splitlines()
    assert "src/catan_zero/data/__init__.py" in tracked
    assert "src/catan_zero/data/colonist.py" in tracked


def test_colonist_package_imports():
    from catan_zero.data import colonist  # noqa: F401


def test_no_tracked_symlinks_anywhere():
    # Mode 120000 is git's symlink object mode. This repo has NO legitimate
    # tracked symlinks (verified 2026-07-05); any tracked symlink -- under
    # src/ or anywhere else, including merge-conflict redirect paths like
    # "src/catan_zero/data~<sha> (<subject>)" -- is a leak that breaks other
    # checkouts. Repo-wide on purpose: the f79 cherry-pick redirect would
    # have been caught by a src/ scope, but root-level redirects would not.
    entries = _git("ls-files", "-s").splitlines()
    symlinks = [line for line in entries if line.startswith("120000")]
    assert not symlinks, f"tracked symlinks: {symlinks}"


def test_no_symlinks_on_disk_under_src():
    # The index check above cannot see a symlink that is sitting UNTRACKED in
    # the working tree -- which is exactly the pre-commit state of the
    # 2026-07-05 incident (the worktree-hack symlink existed on disk, tests
    # passed, and a later `git add -A` swept it into the branch). Catch it on
    # disk, before any add. src/ contains only Python packages; there is no
    # legitimate symlink under it.
    on_disk = [
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "src").rglob("*")
        if path.is_symlink()
    ]
    assert not on_disk, f"symlinks on disk under src/: {on_disk}"
