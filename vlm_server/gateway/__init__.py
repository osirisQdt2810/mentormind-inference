"""CPU gateway: one HTTP server in front of everything MentorMind runs remotely.

``/v1/audio/transcriptions`` (faster-whisper), ``/v1/embeddings`` (bge-m3) and
``/v1/documents/convert`` (Docling) run here on the CPU; every other ``/v1/*`` path is
reverse-proxied to the VLM (vLLM on the GPU). One public URL (Caddy token edge + ngrok) then
serves all of it. The heavy libraries are imported on first use, so the app and its tests run
without them.

``config`` (``INFERENCE_*`` settings), ``app`` (``create_app``), ``asr``, ``embeddings``,
``documents`` (engines), ``proxy`` (VLM pass-through), ``cli`` (``python -m vlm_server gateway``).
"""
