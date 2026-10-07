"""requirements/*.txt: one gateway file per platform on top of gateway-common.txt, plus vLLM's."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.deploy.shell import REPO, needs_bash, run_lib

REQUIREMENTS = REPO / "requirements"
COMMON = REQUIREMENTS / "gateway-common.txt"
PLATFORM_FILES = sorted(p for p in REQUIREMENTS.glob("gateway-*.txt") if p != COMMON)
ALL_FILES = sorted(REQUIREMENTS.glob("*.txt"))
EXPECTED = {
    "gateway-common.txt",
    "gateway-linux-cpu.txt",
    "gateway-macos.txt",
    "gateway-linux-cuda.txt",
    "gateway-linux-rocm.txt",
    "vllm-linux-cuda.txt",
}


def header(path: Path) -> str:
    """The leading comment block."""
    lines = []
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            break
        lines.append(line)
    return "\n".join(lines)


def entries(path: Path) -> list[str]:
    """Requirement and option lines, comments stripped."""
    lines = (line.split(" #")[0].strip() for line in path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def includes(path: Path) -> list[str]:
    return [line.split()[1] for line in entries(path) if line.split()[0] in ("-r", "--requirement")]


def test_the_expected_files_exist_and_the_old_single_file_is_gone() -> None:
    assert {p.name for p in ALL_FILES} == EXPECTED
    assert not (REPO / "requirements-gateway.txt").exists()


@pytest.mark.parametrize("path", PLATFORM_FILES, ids=lambda p: p.name)
def test_each_gateway_file_includes_the_common_file_and_names_torch(path: Path) -> None:
    assert includes(path) == ["gateway-common.txt"]
    assert any(re.match(r"torch\b", line) for line in entries(path))


def test_the_pyav_pin_is_in_the_common_file_once_with_its_reason() -> None:
    text = COMMON.read_text()
    assert [line for line in entries(COMMON) if line.startswith("av")] == ["av>=11,<19"]
    assert text.count("<19") == 1 and "metadata_errors" in text
    for path in ALL_FILES:
        if path != COMMON:
            assert not any(line.startswith("av") for line in entries(path)), path.name


def test_the_common_file_leaves_torch_to_the_platform_files() -> None:
    assert not includes(COMMON)
    assert not any(re.match(r"torch\b", line) for line in entries(COMMON))


@pytest.mark.parametrize("name", ["gateway-linux-cpu.txt", "gateway-macos.txt"])
def test_cpu_and_macos_files_pin_no_cuda_wheel(name: str) -> None:
    for path in (REQUIREMENTS / name, COMMON):
        text = path.read_text()
        assert not re.search(r"download\.pytorch\.org/whl/(cu|rocm)", text), path.name
        for line in entries(path):
            assert not re.search(r"\+(cu|rocm)\d|https?://|--(extra-)?index-url|--find-links", line)


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.name)
def test_each_file_opens_with_platform_install_command_and_test_status(path: Path) -> None:
    text = header(path)
    assert text, f"{path.name} has no header comment"
    for label in ("# Platform:", "# Install", "# Tested:"):
        assert label in text, f"{path.name}: no {label!r}"
    if path != COMMON:
        assert f"-r requirements/{path.name}" in text


@needs_bash
@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.name)
def test_the_header_command_carries_the_flags_run_sh_uses(path: Path) -> None:
    rel = f"requirements/{path.name}"
    result = run_lib(f'uv_flags_for "{rel}"')
    assert result.returncode == 0, result.stderr
    flags = result.stdout.strip()
    if path == COMMON:
        assert flags == ""
        return
    commands = [line for line in header(path).splitlines() if f"-r {rel}" in line]
    assert any(
        line.rstrip().endswith(f"-r {rel}" + (f" {flags}" if flags else "")) for line in commands
    )
