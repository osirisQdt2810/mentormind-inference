"""Which ``requirements/*.txt`` installs the gateway's libraries, for the "not installed" errors.

The gateway runs on CPU by design, so this is the CPU flavour of the OS: the same choice as
``default_gateway_requirements`` and ``uv_flags_for`` in ``scripts/lib/platform.sh`` (a test keeps
the two in step).
"""

from __future__ import annotations

import platform

#: ``platform.system()`` -> (requirements file, the uv flags it needs).
GATEWAY_REQUIREMENTS: dict[str, tuple[str, str]] = {
    "Linux": ("requirements/gateway-linux-cpu.txt", "--torch-backend=cpu"),
    "Darwin": ("requirements/gateway-macos.txt", ""),
}


def install_hint(system: str | None = None) -> str:
    """The install command for this OS (``system`` defaults to ``platform.system()``)."""
    system = system or platform.system()
    if system not in GATEWAY_REQUIREMENTS:
        return f"install the gateway's requirements/gateway-*.txt (none is made for {system})"
    path, flags = GATEWAY_REQUIREMENTS[system]
    return f"uv pip install -r {path} {flags}".rstrip()
