"""``python -m vlm_server gateway``: the CPU gateway (ASR, embeddings, documents + VLM proxy).

Listens on INFERENCE_HOST:INFERENCE_PORT (default 127.0.0.1:18080) and forwards other /v1/* paths
to INFERENCE_VLM_UPSTREAM (default http://127.0.0.1:18000). Runs from the venv of the platform's
requirements/gateway-*.txt (CPU flavour by default); models load on their first request.
"""

from __future__ import annotations

import argparse

from vlm_server.gateway.config import GatewayConfig
from vlm_server.gateway.deps import install_hint


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m vlm_server gateway", description=__doc__)
    parser.parse_args(args)

    config = GatewayConfig()
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(f"uvicorn is not installed: {install_hint()}") from exc
    from vlm_server.gateway.app import create_app

    print(
        f"[gateway] http://{config.host}:{config.port} | VLM upstream {config.vlm_upstream}"
        f" | ASR {config.asr_model} ({config.asr_device}, {config.asr_compute_type})"
        f" | embeddings {config.embed_model} | documents docling",
        flush=True,
    )
    uvicorn.run(create_app(config), host=config.host, port=config.port, log_level="info")
