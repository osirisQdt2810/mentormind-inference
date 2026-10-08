"""Speed of one server config on the same 6 LASI segments: pass A one at a time and N at once.

usage: cd $MENTORMIND && uv run python speed.py <work_dir with the LASI segments> <out_dir> <label> [modes]
modes: comma-separated numbers of requests in flight, 1 = one at a time (default "1,4"). N > 1 uses
describe_segments(executor=...) of MentorMind's KNOWHOW_VLM_CONCURRENCY (spec 04 AC-30).
One JSON line per mode: wall time, seconds per call, output tokens, total output tokens per second.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median

from mentormind.capture.video import store
from mentormind.core.config import get_settings
from mentormind.core.providers.vlm import get_vlm_provider
from mentormind.processing.vlm.passes.describe import describe_segments

work, out, label = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
modes = [int(m) for m in (sys.argv[4] if len(sys.argv) > 4 else "1,4").split(",")]
vid = next((work / "video").iterdir()).name
segments = store.read_segments(store.segments_path(work, vid))
pick = [segments[i] for i in (2, 6, 9, 12, 16, 22)]  # people, robot, conveyor, packing
settings = get_settings()
vlm = get_vlm_provider(settings)

for n in modes:
    run_dir = out / label / f"c{n}"
    started = time.time()
    with ThreadPoolExecutor(n) as pool:
        result = describe_segments(
            pick,
            vlm,
            process_id="PACK01",
            work_dir=run_dir,
            default_fps=settings.vlm_fps,
            max_frames=settings.vlm_max_frames,
            executor=pool if n > 1 else None,
        )
    wall = time.time() - started
    raws = [json.loads(f.read_text()) for f in sorted((run_dir / "pass_a").glob("*.json"))]
    latency = [r["response"]["latency_s"] for r in raws]
    tokens = [(r["response"].get("usage") or {}).get("completion_tokens", 0) for r in raws]
    print(
        json.dumps(
            {
                "label": label,
                "model": settings.vlm_model,
                "concurrency": n,
                "wall_s": round(wall),
                "calls": len(raws),
                "per_call_median_s": round(median(latency), 1),
                "out_tokens_total": sum(tokens),
                "out_tokens_median": int(median(tokens)),
                "tok_per_s_total": round(sum(tokens) / wall, 1),
                "steps": len(result.steps),
                "failed": len(result.failed),
            }
        ),
        flush=True,
    )
