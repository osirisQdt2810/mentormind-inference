"""Speed of one server config on the same 6 LASI segments: pass A, sequential then N concurrent.

usage: cd $MENTORMIND && uv run python speed.py <work_dir with the LASI segments> <out_dir> <label> [concurrency=3]
One JSON line: per-call latency, output tokens/s per stream, wall time sequential vs concurrent.
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
conc = int(sys.argv[4]) if len(sys.argv) > 4 else 3
vid = next((work / "video").iterdir()).name
segments = store.read_segments(store.segments_path(work, vid))
pick = [segments[i] for i in (2, 6, 9, 12, 16, 22)]
s = get_settings()
vlm = get_vlm_provider(s)


def one(seg, tag):
    t = time.time()
    r = describe_segments(
        [seg],
        vlm,
        process_id="PACK01",
        work_dir=out / label / tag / seg.segment_id,
        default_fps=s.vlm_fps,
        max_frames=s.vlm_max_frames,
    )
    return time.time() - t, r


t0 = time.time()
seq = [one(seg, "seq") for seg in pick]
seq_wall = time.time() - t0
par, par_wall = [], 0.0
if conc > 1:  # 0/1 = sequential only
    t0 = time.time()
    with ThreadPoolExecutor(conc) as pool:
        par = list(pool.map(lambda seg: one(seg, f"par{conc}"), pick))
    par_wall = time.time() - t0

lat = [w for w, _ in seq]
out_tok = [r.completion_tokens for _, r in seq]
print(
    json.dumps(
        {
            "label": label,
            "model": s.vlm_model,
            "seq_per_call_median_s": round(median(lat), 1),
            "seq_per_call_mean_s": round(sum(lat) / len(lat), 1),
            "seq_wall_s": round(seq_wall),
            "out_tokens_mean": sum(out_tok) // len(out_tok),
            "out_tok_per_s_single_stream": round(sum(out_tok) / sum(lat), 1),
            f"par{conc}_wall_s": round(par_wall),
            "speedup_concurrent": round(seq_wall / par_wall, 2) if par_wall else None,
            "steps_seq": sum(len(r.steps) for _, r in seq),
            "steps_par": sum(len(r.steps) for _, r in par),
            "failed_seq": sum(len(r.failed) for _, r in seq),
            "failed_par": sum(len(r.failed) for _, r in par),
        }
    ),
    flush=True,
)
