"""scripts/lib/platform.sh: platform detection and its requirements files, as run.sh uses them."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.deploy.shell import REPO, fake_tools, needs_bash, run_lib
from vlm_server.gateway.deps import GATEWAY_REQUIREMENTS

pytestmark = needs_bash

REPORT = (
    'detect_platform; echo "$PLATFORM_OS $PLATFORM_ARCH $PLATFORM_ACCEL";'
    " default_gateway_requirements || echo none;"
    " default_vllm_requirements 2>/dev/null || echo none"
)


def detect(tmp_path: Path, os_name: str, arch: str, **tools: int) -> list[str]:
    result = run_lib(REPORT, path=fake_tools(tmp_path / "bin", os_name=os_name, arch=arch, **tools))
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def test_vast_box_cuda_gets_the_cpu_gateway_and_the_cuda_vllm(tmp_path: Path) -> None:
    assert detect(tmp_path, "Linux", "x86_64", nvidia_smi=0) == [
        "Linux x86_64 cuda",
        "requirements/gateway-linux-cpu.txt",
        "requirements/vllm-linux-cuda.txt",
    ]


def test_a_broken_nvidia_smi_is_not_cuda(tmp_path: Path) -> None:
    lines = detect(tmp_path, "Linux", "x86_64", nvidia_smi=9)
    assert lines == ["Linux x86_64 cpu", "requirements/gateway-linux-cpu.txt", "none"]


@pytest.mark.parametrize("tool", ["amd_smi", "rocm_smi"])
def test_amd_tools_mean_rocm_and_vllm_comes_from_amds_image(tmp_path: Path, tool: str) -> None:
    lines = detect(tmp_path, "Linux", "x86_64", **{tool: 0})
    assert lines == ["Linux x86_64 rocm", "requirements/gateway-linux-cpu.txt", "none"]


def test_cuda_wins_over_rocm_tools(tmp_path: Path) -> None:
    assert detect(tmp_path, "Linux", "aarch64", nvidia_smi=0, rocm_smi=0)[0] == "Linux aarch64 cuda"


def test_apple_silicon_gets_the_macos_gateway_and_no_vllm(tmp_path: Path) -> None:
    lines = detect(tmp_path, "Darwin", "arm64")
    assert lines == ["Darwin arm64 cpu", "requirements/gateway-macos.txt", "none"]


def test_an_unknown_os_has_no_gateway_file(tmp_path: Path) -> None:
    assert detect(tmp_path, "FreeBSD", "amd64")[1] == "none"


@pytest.mark.parametrize("os_name", sorted(GATEWAY_REQUIREMENTS))
def test_the_python_install_hint_agrees_with_the_shell(tmp_path: Path, os_name: str) -> None:
    snippet = 'detect_platform; f=$(default_gateway_requirements); echo "$f"; uv_flags_for "$f"'
    result = run_lib(snippet, path=fake_tools(tmp_path / "bin", os_name=os_name, arch="x86_64"))
    assert result.returncode == 0, result.stderr
    assert tuple(result.stdout.splitlines()) == GATEWAY_REQUIREMENTS[os_name]


def test_requirements_files_follows_the_includes() -> None:
    result = run_lib("requirements_files requirements/gateway-linux-cpu.txt")
    assert result.stdout.splitlines() == [
        "requirements/gateway-linux-cpu.txt",
        "requirements/gateway-common.txt",
    ]


def test_the_hash_changes_with_an_included_file(tmp_path: Path) -> None:
    (tmp_path / "common.txt").write_text("fastapi>=0.115\n")
    (tmp_path / "flavour.txt").write_text("# header\n-r common.txt\ntorch>=2.2\n")
    hash_of = f'requirements_hash "{tmp_path}/flavour.txt"'
    first = run_lib(hash_of).stdout.strip()
    assert len(first) == 64 and run_lib(hash_of).stdout.strip() == first
    (tmp_path / "common.txt").write_text("fastapi>=0.115\nav>=11,<19\n")
    assert run_lib(hash_of).stdout.strip() != first


def test_a_missing_include_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "flavour.txt").write_text("-r gone.txt\n")
    result = run_lib(f'requirements_hash "{tmp_path}/flavour.txt"')
    assert result.returncode != 0 and "gone.txt" in result.stderr


def test_install_command_is_the_header_command() -> None:
    result = run_lib("install_command /opt/inference-cpu requirements/gateway-linux-cpu.txt")
    assert result.stdout.strip() == (
        "uv pip install --python /opt/inference-cpu/bin/python"
        " -r requirements/gateway-linux-cpu.txt --torch-backend=cpu"
    )
    assert result.stdout.strip() in (REPO / "requirements/gateway-linux-cpu.txt").read_text()


def test_the_hash_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    """A header edit (e.g. the "Tested:" line) must not reinstall a venv or restart vLLM."""
    (tmp_path / "common.txt").write_text("fastapi>=0.115\n")
    flavour = tmp_path / "flavour.txt"
    flavour.write_text("# Tested: no\n-r common.txt\ntorch>=2.2\n")
    hash_of = f'requirements_hash "{flavour}"'
    first = run_lib(hash_of).stdout.strip()
    flavour.write_text(
        "# Tested: yes, on Vast\n#   since today\n\n-r common.txt  \ntorch>=2.2  # CPU\n"
    )
    (tmp_path / "common.txt").write_text("# shared\nfastapi>=0.115 # api\n\n")
    assert run_lib(hash_of).stdout.strip() == first
    flavour.write_text("# Tested: yes, on Vast\n-r common.txt\ntorch>=2.3\n")
    assert run_lib(hash_of).stdout.strip() != first


def test_an_absolute_include_is_followed(tmp_path: Path) -> None:
    common = tmp_path / "shared" / "common.txt"
    common.parent.mkdir()
    common.write_text("fastapi>=0.115\n")
    (tmp_path / "flavour.txt").write_text(f"-r {common}\ntorch>=2.2\n")
    result = run_lib(f'requirements_files "{tmp_path}/flavour.txt"')
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [f"{tmp_path}/flavour.txt", str(common)]


def test_install_command_runs_word_for_word_under_eval(tmp_path: Path) -> None:
    """run.sh logs install_command's output and evals that same string: paths must survive it."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text('#!/bin/sh\nfor arg in "$@"; do echo "$arg"; done\n')
    (bin_dir / "uv").chmod(0o755)
    snippet = 'eval "$(install_command "/opt/my venv" "req dir/gateway-linux-cpu.txt" -q)"'
    result = run_lib(snippet, path=f"{bin_dir}:/usr/bin:/bin")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "pip",
        "install",
        "--python",
        "/opt/my venv/bin/python",
        "-r",
        "req dir/gateway-linux-cpu.txt",
        "--torch-backend=cpu",
        "-q",
    ]


def test_run_sh_installs_only_through_the_logged_command() -> None:
    """No second, unlogged "uv pip install" in run.sh: what the log shows is what runs."""
    lines = (REPO / "scripts/vast/run.sh").read_text().splitlines()
    code = [line for line in lines if not line.lstrip().startswith("#")]
    assert not [line for line in code if "uv pip install" in line]
    assert any('(eval "$install")' in line for line in code)


def test_run_sh_reexecs_itself_when_the_checkout_changes_it() -> None:
    """bash parses run.sh before the git checkout: a changed script must run as the new version, once."""
    script = (REPO / "scripts" / "vast" / "run.sh").read_text()
    copy_at = script.index('cp "${BASH_SOURCE[0]}" "$RUNNING_COPY"')
    checkout_at = script.index("checkout -q --detach")
    reexec_at = script.index('RUN_SH_REEXEC=1 exec bash "$ENGINE/scripts/vast/run.sh" "$@"')
    assert copy_at < checkout_at < reexec_at
    assert '[ -z "${RUN_SH_REEXEC:-}" ]' in script
