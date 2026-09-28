"""spec 04 AC-14 — one-GPU vLLM command and environment, CUDA and ROCm (MI250)."""

from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

from vlm_server.config import ServerConfig
from vlm_server.gpu import GpuStatus
from vlm_server.serve.command import build_command, build_env

A5000 = GpuStatus(index=3, used_mib=18, total_mib=24564)
MI250_GCD = GpuStatus(index=1, used_mib=10, total_mib=65520)


def config(**values: object) -> ServerConfig:
    return ServerConfig(_env_file=None, **values)  # type: ignore[arg-type]


def flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


@pytest.mark.parametrize("value", ["0,1", "0 1", "all", "-1", "1-3"])
def test_ac_14_multi_gpu_values_are_rejected(value: str) -> None:
    with pytest.raises(ValidationError, match="exactly one GPU"):
        config(gpu=value)


def test_ac_14_cuda_env_exposes_exactly_one_gpu() -> None:
    env = build_env(config(), "cuda", 3, {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "PATH": "/bin"})
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["PATH"] == "/bin"


def test_ac_14_rocm_env_exposes_exactly_one_gcd() -> None:
    env = build_env(config(), "rocm", 1, {"HIP_VISIBLE_DEVICES": "0,1"})
    assert env["HIP_VISIBLE_DEVICES"] == env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["VLLM_ROCM_USE_AITER"] == "0"  # AITER kernels need gfx942+, an MI250 is gfx90a


def test_extra_env_overrides_platform_defaults() -> None:
    env = build_env(config(extra_env={"VLLM_ROCM_USE_AITER": "1"}), "rocm", 0, {})
    assert env["VLLM_ROCM_USE_AITER"] == "1"


def test_ac_14_command_pins_single_gpu_parallelism() -> None:
    argv = build_command(config(), A5000)
    assert flag(argv, "--tensor-parallel-size") == "1"
    assert flag(argv, "--pipeline-parallel-size") == "1"
    assert argv[:4] == [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve"]
    assert argv[4] == "Qwen/Qwen3-VL-8B-Instruct"


def test_ac_14_api_key_goes_to_env_not_argv() -> None:
    cfg = config(api_key="s3cret")
    assert "s3cret" not in " ".join(build_command(cfg, A5000))
    assert build_env(cfg, "cuda", 0, {})["VLLM_API_KEY"] == "s3cret"
    assert "VLLM_API_KEY" not in build_env(config(), "cuda", 0, {"VLLM_API_KEY": "stale"})


def test_context_length_follows_gpu_size_unless_configured() -> None:
    assert flag(build_command(config(), A5000), "--max-model-len") == "16384"
    assert flag(build_command(config(), MI250_GCD), "--max-model-len") == "32768"
    assert flag(build_command(config(max_model_len=8192), MI250_GCD), "--max-model-len") == "8192"


def test_defaults_bind_loopback_and_leave_port_8000_to_the_backend() -> None:
    argv = build_command(config(), A5000)
    assert flag(argv, "--host") == "127.0.0.1"
    assert flag(argv, "--port") == "8100"
    assert flag(argv, "--limit-mm-per-prompt") == '{"image":64,"video":0}'
