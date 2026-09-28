"""AMD GPUs (MI250 and friends) through ``amd-smi`` or the older ``rocm-smi``.

On an MI250 each of the two GCDs is its own device (64 GB each): "one GPU" here means one GCD,
exactly what HIP enumerates. Both tools print JSON whose shape varies across ROCm releases, so the
parsers accept the known shapes and raise :class:`GpuQueryError` on anything else — the launcher
then asks for an explicit ``VLM_SERVER_GPU`` instead of guessing.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Any

from vlm_server.gpu.models import GpuStatus

_MIB = 1024 * 1024


class GpuQueryError(RuntimeError):
    """The vendor tool is missing or printed something this parser does not know."""


def parse_rocm_smi(json_text: str) -> list[GpuStatus]:
    """``rocm-smi --showmeminfo vram --showuse --json``: ``{"card0": {"VRAM Total Memory (B)"...}}``."""
    data = json.loads(json_text)
    gpus = []
    for card, fields in data.items():
        match = re.fullmatch(r"card(\d+)", card)
        if not match or not isinstance(fields, dict):
            continue
        total = _field(fields, "VRAM Total Memory (B)")
        used = _field(fields, "VRAM Total Used Memory (B)")
        busy = _field(fields, "GPU use (%)", default=0)
        gpus.append(
            GpuStatus(
                index=int(match.group(1)),
                used_mib=used // _MIB,
                total_mib=max(total // _MIB, 1),
                utilization_pct=min(busy, 100),
            )
        )
    if not gpus:
        raise GpuQueryError("rocm-smi printed no card")
    return sorted(gpus, key=lambda g: g.index)


def parse_amd_smi(json_text: str) -> list[GpuStatus]:
    """``amd-smi metric --mem-usage --json``: a list of ``{"gpu": i, "mem_usage": {...}}``."""
    data = json.loads(json_text)
    items = data.get("gpu_data", data) if isinstance(data, dict) else data
    gpus = []
    for item in items:
        usage = item.get("mem_usage") or {}
        total = _mib(usage.get("total_vram"))
        used = _mib(usage.get("used_vram"))
        if total is None or used is None:
            raise GpuQueryError(f"amd-smi output without total/used VRAM: {str(item)[:200]}")
        gpus.append(GpuStatus(index=int(item["gpu"]), used_mib=used, total_mib=max(total, 1)))
    if not gpus:
        raise GpuQueryError("amd-smi printed no GPU")
    return sorted(gpus, key=lambda g: g.index)


def query_rocm() -> list[GpuStatus]:
    """Ask ``amd-smi`` first (ROCm 6+), then ``rocm-smi``."""
    attempts = [
        ("amd-smi", ["amd-smi", "metric", "--mem-usage", "--json"], parse_amd_smi),
        ("rocm-smi", ["rocm-smi", "--showmeminfo", "vram", "--showuse", "--json"], parse_rocm_smi),
    ]
    errors = []
    for tool, argv, parse in attempts:
        if shutil.which(tool) is None:
            continue
        try:
            out = subprocess.run(argv, check=True, capture_output=True, text=True).stdout
            return parse(out)
        except (subprocess.CalledProcessError, ValueError, KeyError, GpuQueryError) as exc:
            errors.append(f"{tool}: {exc}")
    raise GpuQueryError(
        "Không đọc được danh sách GPU AMD ("
        + ("; ".join(errors) or "thiếu amd-smi/rocm-smi")
        + "). Đặt VLM_SERVER_GPU=<index> để chọn thẳng một GCD."
    )


def _field(fields: dict[str, Any], name: str, default: int | None = None) -> int:
    value = fields.get(name)
    if value is None:
        if default is not None:
            return default
        raise GpuQueryError(f"rocm-smi output without {name!r}")
    return int(str(value).strip().rstrip("%") or 0)


def _mib(value: Any) -> int | None:
    """``{"value": 65536, "unit": "MB"}`` (amd-smi) -> MiB; plain numbers are MiB already."""
    if value is None:
        return None
    if isinstance(value, dict):
        amount, unit = float(value.get("value", 0)), str(value.get("unit", "MB")).upper()
        scale = {"B": 1 / _MIB, "KB": 1 / 1024, "MB": 1.0, "MIB": 1.0, "GB": 1024.0, "GIB": 1024.0}
        return int(amount * scale.get(unit, 1.0))
    return int(value)
