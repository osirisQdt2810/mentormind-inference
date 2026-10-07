"""Score UI runs of the LASI copies against data/video/groundtruth.json with the repo's sweep metrics.

usage: cd $MENTORMIND && uv run python score.py <work_dir> <label>=<video_id> [...] > scores.json

Per run: the final steps the UI shows (predictions.json: pass A, B, C) and pass A run 1 alone (rebuilt
from the raw answers exactly as describe.py parses them). Metrics are spec 04 §6's: a prediction
matches a truth step when token_set_ratio(step) >= 70 AND the times overlap (one-to-one, best first).
"""

import json
import os
import sys
from pathlib import Path
from statistics import median

from mentormind.capture.video import store
from mentormind.core.providers.errors import InvalidJSONError
from mentormind.core.providers.parsing import parse_json_object
from mentormind.processing.vlm.passes.language import english_fields
from mentormind.processing.vlm.passes.models import PassAOutput, PassAResult, TimedStep
from mentormind.processing.vlm.passes.step_times import get_step_timer
from mentormind.processing.vlm.sweep import evaluate, load_truth, match_steps
from pydantic import ValidationError

REPO = Path(os.environ.get("MENTORMIND", "."))
truth = load_truth(REPO / "data/video/groundtruth.json")
work = Path(sys.argv[1])


def raw_records(folder: Path) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted(folder.glob("*.json"))]


def rebuild_pass_a(folder: Path, segments: dict, timer) -> PassAResult:
    """Pass A from the raw answers as describe.py keeps them: the first answer unless it was invalid
    or in English, then the repair answer (an English answer is kept if the repair fails)."""
    result = PassAResult()
    for seg_id in sorted({f.name.split(".")[0] for f in folder.glob("*.json")}):
        output = english = None
        attempts = [folder / f"{seg_id}.json", folder / f"{seg_id}.repair.json"]
        recs = [json.loads(f.read_text()) for f in attempts if f.exists()]
        for rec in recs:
            usage = rec["response"].get("usage") or {}
            result.prompt_tokens += usage.get("prompt_tokens", 0)
            result.completion_tokens += usage.get("completion_tokens", 0)
            result.latency_s += rec["response"].get("latency_s", 0.0)
            result.calls += 1
            try:
                cand = PassAOutput.model_validate(parse_json_object(rec["response"]["text"]))
            except (InvalidJSONError, ValidationError):
                continue  # as describe.py: an invalid answer counts as no answer
            if english_fields(cand) and rec is recs[0]:
                english = cand
                continue
            output = cand
            break
        output = output or english
        if output is None:
            continue
        for step, start, end in timer.place(output.steps, segments[seg_id]):
            result.steps.append(
                TimedStep(
                    segment_id=seg_id,
                    t_start=start,
                    t_end=end,
                    step=step.step.strip(),
                    actor=step.actor,
                    hand_action=step.hand_action,
                    objects=step.objects,
                    key_point_observed=step.key_point_observed,
                    confidence=step.confidence,
                    fps=recs[0].get("fps", 4.0),
                )
            )
    return result


def latency_stats(folder: Path) -> dict:
    recs = raw_records(folder)
    lat = [r["response"].get("latency_s", 0.0) for r in recs]
    out = [(r["response"].get("usage") or {}).get("completion_tokens", 0) for r in recs]
    return {
        "calls": len(recs),
        "median_s": round(median(lat), 1) if lat else None,
        "mean_s": round(sum(lat) / len(lat), 1) if lat else None,
        "total_s": round(sum(lat)),
        "out_tokens_median": int(median(out)) if out else None,
        "out_tok_per_s": round(sum(out) / sum(lat), 1) if lat and sum(lat) else None,
    }


rows = {}
timer = get_step_timer("segment")
for arg in sys.argv[2:]:
    # <label>=<video_id>[@<describe dir>]: a describe dir kept elsewhere (e.g. a job that failed after
    # passes A and B) is scored with the video's segments; without predictions.json only pass A counts.
    label, spec = arg.split("=", 1)
    vid, _, kept = spec.partition("@")
    segments = {s.segment_id: s for s in store.read_segments(store.segments_path(work, vid))}
    run = Path(kept) if kept else next((work / "vlm" / vid).glob("describe-*"))
    passes = {"pass_a": rebuild_pass_a(run / "pass_a", segments, timer)}
    if (run / "pass_b" / "pass_a").exists():
        passes["pass_b_run"] = rebuild_pass_a(run / "pass_b" / "pass_a", segments, timer)
    if (run / "predictions.json").exists():
        passes["final"] = PassAResult.model_validate_json((run / "predictions.json").read_text())
    row = {}
    for name, res in passes.items():
        r = evaluate(
            truth, res, fps=4.0, provider="remote", model="", step_times="segment"
        ).model_dump()
        matches = match_steps(truth, res.steps)
        r["matched"] = [
            {
                "truth": truth[m.truth_index].step,
                "pred": res.steps[m.predicted_index].step,
                "score": round(m.score),
            }
            for m in matches
        ]
        row[name] = r
        row[f"{name}_steps"] = [
            {
                "t": [s.t_start, s.t_end],
                "step": s.step,
                "actor": s.actor,
                "key_point": s.key_point_observed,
            }
            for s in res.steps
        ]
    row["latency_pass_a"] = latency_stats(run / "pass_a")
    if (run / "pass_b" / "pass_a").exists():
        row["latency_pass_b"] = latency_stats(run / "pass_b" / "pass_a")
    row["segments"] = len(segments)
    rows[label] = row
print(json.dumps(rows, ensure_ascii=False, indent=1))
