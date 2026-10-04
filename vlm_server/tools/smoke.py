"""End-to-end smoke test of a running server: health, model list, one multi-frame request.

    uv run python -m vlm_server smoke                    # synthetic 8-frame clip
    uv run python -m vlm_server smoke --frames 48        # worst case of spec 04 (48 frames)
    uv run python -m vlm_server smoke --images a.jpg b.jpg c.jpg

The synthetic clip shows a red square sliding left -> right into a blue box, so a correct answer
needs the frames' ORDER, not just their content: it checks video understanding, not captioning.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw

from vlm_server.config import ServerConfig

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "moving_object_color": {"type": "string"},
        "direction": {"enum": ["left_to_right", "right_to_left", "static", "other"]},
        "ends_inside_box": {"type": "boolean"},
        "description": {"type": "string"},
    },
    "required": ["moving_object_color", "direction", "ends_inside_box", "description"],
}
PROMPT = (
    "The following {n} images are consecutive frames of one short video, in time order. "
    "Describe the motion you observe. Answer ONLY with JSON: "
    '{{"moving_object_color": str, "direction": "left_to_right|right_to_left|static|other", '
    '"ends_inside_box": bool, "description": str}}'
)


def synthetic_frames(count: int, size: tuple[int, int] = (448, 252)) -> list[bytes]:
    """JPEG frames of a red square moving left -> right into a blue open box."""
    width = size[0]
    box_left = width - 130
    frames = []
    for i in range(count):
        image = Image.new("RGB", size, (200, 200, 200))
        draw = ImageDraw.Draw(image)
        draw.rectangle([box_left, 90, width - 20, 230], outline=(30, 60, 220), width=6)
        x = 20 + round((box_left + 25 - 20) * i / max(count - 1, 1))
        draw.rectangle([x, 150, x + 50, 200], fill=(220, 30, 30))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        frames.append(buffer.getvalue())
    return frames


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m vlm_server smoke", description=__doc__)
    parser.add_argument("--base-url", help="default: http://<VLM_SERVER_HOST>:<PORT>/v1")
    parser.add_argument("--frames", type=int, default=8, help="synthetic frame count")
    parser.add_argument("--images", nargs="*", type=Path, help="send these JPEGs instead")
    opts = parser.parse_args(args)

    config = ServerConfig()
    base_url = (opts.base_url or f"http://{config.host}:{config.serve_port}/v1").rstrip("/")
    headers = {}
    if config.api_key is not None and config.api_key.get_secret_value():
        headers["Authorization"] = f"Bearer {config.api_key.get_secret_value()}"
    client = httpx.Client(timeout=120, headers=headers)

    health = client.get(base_url.removesuffix("/v1") + "/health")
    print(f"health: HTTP {health.status_code}")
    models = [m["id"] for m in client.get(f"{base_url}/models").json()["data"]]
    print(f"models: {models}")

    jpegs = [p.read_bytes() for p in opts.images] if opts.images else synthetic_frames(opts.frames)
    content: list[dict[str, Any]] = [{"type": "text", "text": PROMPT.format(n=len(jpegs))}]
    content += [{"type": "image_url", "image_url": {"url": _data_url(j)}} for j in jpegs]
    payload = {
        "model": config.ollama_model
        if config.backend == "ollama"
        else config.served_model_names[0],
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": 300,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "smoke", "schema": ANSWER_SCHEMA},
        },
    }
    started = time.perf_counter()
    response = client.post(f"{base_url}/chat/completions", json=payload)
    latency = time.perf_counter() - started
    response.raise_for_status()
    body = response.json()
    answer = json.loads(body["choices"][0]["message"]["content"])
    print(f"frames: {len(jpegs)} | latency: {latency:.2f}s | usage: {body.get('usage')}")
    print(json.dumps(answer, ensure_ascii=False, indent=2))
    if not opts.images:
        ok = answer["direction"] == "left_to_right" and answer["ends_inside_box"] is True
        print("RESULT:", "PASS" if ok else "FAIL (unexpected motion description)")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
