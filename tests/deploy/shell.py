"""Run functions of scripts/lib/platform.sh in bash, as scripts/vast/run.sh does (set -euo pipefail)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "scripts" / "lib" / "platform.sh"
BASH = shutil.which("bash")

needs_bash = pytest.mark.skipif(BASH is None, reason="bash not found")


def run_lib(snippet: str, *, path: str | None = None) -> subprocess.CompletedProcess[str]:
    """``snippet`` after sourcing the library, in the repo root; ``path`` replaces PATH."""
    env = dict(os.environ) if path is None else {"PATH": path}
    assert BASH is not None
    return subprocess.run(
        [BASH, "-c", f'set -euo pipefail; . "$1"; {snippet}', "bash", str(LIB)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def fake_tools(bin_dir: Path, *, os_name: str, arch: str, **tools: int) -> str:
    """A PATH holding only a fake ``uname`` and the GPU tools ``tools`` (name -> exit code).

    Underscores in a tool name stand for dashes (``nvidia_smi`` -> ``nvidia-smi``).
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    scripts = {"uname": f'case "$1" in -m) echo {arch} ;; *) echo {os_name} ;; esac\n'}
    scripts.update({name.replace("_", "-"): f"exit {code}\n" for name, code in tools.items()})
    for name, body in scripts.items():
        tool = bin_dir / name
        tool.write_text("#!/bin/sh\n" + body)
        tool.chmod(0o755)
    return str(bin_dir)
