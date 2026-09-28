"""``python -m vlm_server <command>``.

serve [--dry-run]   pick ONE GPU (CUDA or ROCm) and exec ``vllm serve`` on it
smoke [--frames N]  health + model list + one multi-frame request against a running server
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    command, rest = (args[0], args[1:]) if args else ("", [])
    if command == "serve":
        from vlm_server.serve.launch import main as serve

        serve(rest)
    elif command == "smoke":
        from vlm_server.tools.smoke import main as smoke

        smoke(rest)
    else:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
