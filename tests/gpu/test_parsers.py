"""Vendor tool output -> GpuStatus (NVIDIA nvidia-smi, AMD rocm-smi / amd-smi on MI250)."""

from __future__ import annotations

import json

import pytest

from vlm_server.gpu.nvidia import parse_nvidia_smi
from vlm_server.gpu.rocm import GpuQueryError, parse_amd_smi, parse_rocm_smi

NVIDIA_SMI = """\
0, 52, 24564, 0
1, 15, 24564, 3
7, 13827, 24564, [N/A]
"""

# One MI250 = two GCDs of 64 GB, enumerated as card0/card1.
ROCM_SMI = json.dumps(
    {
        "card0": {
            "VRAM Total Memory (B)": "68702699520",
            "VRAM Total Used Memory (B)": "10485760",
            "GPU use (%)": "0",
        },
        "card1": {
            "VRAM Total Memory (B)": "68702699520",
            "VRAM Total Used Memory (B)": "34359738368",
            "GPU use (%)": "97",
        },
        "system": {"Driver version": "6.8.5"},
    }
)

AMD_SMI = json.dumps(
    [
        {
            "gpu": 0,
            "mem_usage": {
                "total_vram": {"value": 65520, "unit": "MB"},
                "used_vram": {"value": 10, "unit": "MB"},
            },
        },
        {
            "gpu": 1,
            "mem_usage": {
                "total_vram": {"value": 65520, "unit": "MB"},
                "used_vram": {"value": 32768, "unit": "MB"},
            },
        },
    ]
)


def test_nvidia_smi_csv() -> None:
    gpus = parse_nvidia_smi(NVIDIA_SMI)
    assert [(g.index, g.used_mib, g.total_mib, g.utilization_pct) for g in gpus] == [
        (0, 52, 24564, 0),
        (1, 15, 24564, 3),
        (7, 13827, 24564, 0),
    ]


def test_rocm_smi_json_sees_each_mi250_gcd() -> None:
    gpus = parse_rocm_smi(ROCM_SMI)
    assert [g.index for g in gpus] == [0, 1]
    assert gpus[0].total_gib == pytest.approx(64.0, rel=0.01)
    assert gpus[1].used_mib == 32768 and gpus[1].utilization_pct == 97


def test_amd_smi_json() -> None:
    gpus = parse_amd_smi(AMD_SMI)
    assert [(g.index, g.used_mib, g.total_mib) for g in gpus] == [(0, 10, 65520), (1, 32768, 65520)]


def test_amd_smi_wrapped_in_gpu_data() -> None:
    assert len(parse_amd_smi(json.dumps({"gpu_data": json.loads(AMD_SMI)}))) == 2


@pytest.mark.parametrize("payload", ["{}", '{"system": {}}'])
def test_rocm_smi_without_cards_is_an_error(payload: str) -> None:
    with pytest.raises(GpuQueryError):
        parse_rocm_smi(payload)


def test_amd_smi_without_vram_is_an_error() -> None:
    with pytest.raises(GpuQueryError):
        parse_amd_smi(json.dumps([{"gpu": 0, "mem_usage": {}}]))
