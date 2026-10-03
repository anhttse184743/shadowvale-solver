"""Which code produced a result: the package version plus the git commit, if there is one.

Every benchmark row, tuning report and replay record carries this, so a result can always be
traced to the code that made it — and after a change it is clear which results must be re-run
(see ShadowVale Docs/Experiments/protocol.md).
"""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path

from . import __version__


@functools.lru_cache(maxsize=1)
def code_version() -> str:
    """'0.1.0+e7ccd83' (clean checkout), '0.1.0+e7ccd83.dirty' (uncommitted changes), or '0.1.0'
    when there is no git (the packaged sidecar)."""
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                                text=True, timeout=5, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return __version__
    return f"{__version__}+{commit}{'.dirty' if dirty else ''}" if commit else __version__
