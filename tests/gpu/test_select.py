"""spec 04 AC-14 — exactly one GPU is chosen, on CUDA or ROCm."""

from __future__ import annotations

import pytest

from vlm_server.gpu import GpuStatus, detect_platform, pick_gpu
from vlm_server.gpu import select as select_module

GPUS = [
    GpuStatus(index=0, used_mib=52, total_mib=24564),
    GpuStatus(index=1, used_mib=15, total_mib=24564),
    GpuStatus(index=2, used_mib=539, total_mib=24564, utilization_pct=2),
    GpuStatus(index=7, used_mib=13827, total_mib=24564),
]


def test_auto_picks_the_freest_gpu() -> None:
    assert pick_gpu(GPUS, requested="auto", min_free_gib=21).index == 1


def test_explicit_gpu_is_honoured_even_if_busy() -> None:
    assert pick_gpu(GPUS, requested="7", min_free_gib=21).index == 7


def test_auto_refuses_when_no_gpu_is_free_enough() -> None:
    busy = [GpuStatus(index=0, used_mib=20000, total_mib=24564, utilization_pct=90)]
    with pytest.raises(RuntimeError, match="No GPU"):
        pick_gpu(busy, requested="auto", min_free_gib=21)


def test_unknown_explicit_gpu_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="GPU 5 not found"):
        pick_gpu(GPUS, requested="5", min_free_gib=21)


@pytest.mark.parametrize(
    ("tools", "expected"),
    [({"nvidia-smi"}, "cuda"), ({"amd-smi"}, "rocm"), ({"rocm-smi"}, "rocm")],
)
def test_platform_is_detected_from_the_vendor_tool(
    tools: set[str], expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(select_module.shutil, "which", lambda name: name if name in tools else None)
    assert detect_platform("auto") == expected


def test_configured_platform_wins_and_missing_tools_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(select_module.shutil, "which", lambda name: None)
    assert detect_platform("rocm") == "rocm"
    with pytest.raises(RuntimeError, match="VLM_SERVER_PLATFORM"):
        detect_platform("auto")
