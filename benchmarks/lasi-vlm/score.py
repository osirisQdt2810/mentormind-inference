"""Score UI runs of the LASI copies against data/video/groundtruth.json with the repo's sweep metrics.

usage: cd $MENTORMIND && uv run python score.py <work_dir> <label>=<video_id>[@<describe dir>] [...] > scores.json

Per run: the final steps the UI shows (predictions.json: passes A, B, C) and each pass-A run alone (pass A
and pass B's own pass A), rebuilt from the raw answers the way describe.py keeps them. Steps are placed by
BENCH_STEP_TIMES (segment | model, default segment): it must be the KNOWHOW_VLM_STEP_TIMES of the run.
A segment with no valid answer (or no answer saved) counts in failed_segments. Metrics are spec 04 §6's:
a prediction matches a truth step when token_set_ratio(step) >= 70 AND the times overlap (one-to-one,
best first).
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
from mentormind.processing.vlm.passes.models import (
    FailedSegment,
    PassAOutput,
    PassAResult,
    TimedStep,
)
from mentormind.processing.vlm.passes.step_times import get_step_timer
from mentormind.processing.vlm.sweep import evaluate, load_truth, match_steps
from pydantic import ValidationError

REPO = Path(os.environ.get("MENTORMIND", "."))
STEP_TIMES = os.environ.get("BENCH_STEP_TIMES", "segment")
truth = load_truth(REPO / "data/video/groundtruth.json")
work = Path(sys.argv[1])


def parsed(rec: dict) -> PassAOutput | None:
    try:
        return PassAOutput.model_validate(parse_json_object(rec["response"]["text"]))
    except (InvalidJSONError, ValidationError):
        return None  # as describe.py: an invalid answer counts as no answer


def attempts(folder: Path, seg_id: str) -> list[Path]:
    """The answer files of one segment in this run. describe.py writes a repair only when the first
    answer is invalid or in English, and then always (over)writes it: a ``.repair.json`` next to a valid
    Vietnamese first answer is left from an earlier run of the same video and does not count."""
    first = folder / f"{seg_id}.json"
    repair = folder / f"{seg_id}.repair.json"
    if not first.exists():
        return []
    answer = parsed(json.loads(first.read_text()))
    needed_repair = answer is None or bool(english_fields(answer))
    return [first, repair] if needed_repair and repair.exists() else [first]


def answer_files(folder: Path) -> list[Path]:
    ids = sorted({f.name.split(".")[0] for f in folder.glob("*.json")})
    return [f for seg_id in ids for f in attempts(folder, seg_id)]


def rebuild_pass_a(folder: Path, segments: dict, timer) -> PassAResult:
    """Pass A from the raw answers as describe.py keeps them: the first answer unless it was invalid
    or in English, then the repair answer (an English answer is kept if the repair fails)."""
    result = PassAResult()
    for seg_id, segment in sorted(segments.items()):
        output = english = None
        recs = [json.loads(f.read_text()) for f in attempts(folder, seg_id)]
        for rec in recs:
            usage = rec["response"].get("usage") or {}
            result.prompt_tokens += usage.get("prompt_tokens", 0)
            result.completion_tokens += usage.get("completion_tokens", 0)
            result.latency_s += rec["response"].get("latency_s", 0.0)
            result.calls += 1
            cand = parsed(rec)
            if cand is None:
                continue
            if english_fields(cand) and rec is recs[0]:
                english = cand
                continue
            output = cand
            break
        output = output or english
        if output is None:
            error = "invalid answer" if recs else "no answer saved"
            raw = recs[-1]["response"].get("text", "") if recs else ""
            result.failed.append(FailedSegment(segment_id=seg_id, error=error, raw_text=raw))
            continue
        for step, start, end in timer.place(output.steps, segment):
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
    recs = [json.loads(f.read_text()) for f in answer_files(folder)]
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
timer = get_step_timer(STEP_TIMES)
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
            truth, res, fps=4.0, provider="remote", model="", step_times=STEP_TIMES
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
