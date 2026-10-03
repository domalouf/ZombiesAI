"""Which code wrote a file: the commit, and whether the tree had uncommitted changes.

Every clip and every fleet hello carries it, so data can be traced to the code that made it and a learner can
refuse a worker on a different commit (rl/fleet.py).
"""

import functools
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


@functools.cache
def git_provenance() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"sha": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": "unknown", "dirty": None}
