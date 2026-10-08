"""The overview table of every run: accuracy (repo metrics + blind judges) and speed, as Markdown.

usage: MENTORMIND=<checkout> python3 overview.py <runs.json> [--all-lists]

By default only the final list of each run (what the UI shows after passes A, B, C); --all-lists adds
the two pass-A runs.

runs.json lists the runs in table order:
  [{"model": "8B Instruct", "weights": "BF16", "scores": "score_8b-instruct.json", "label": "8b-instruct",
    "verdicts": "judge/", "job_s": 558}, ...]
Paths are relative to runs.json. Verdict files are <verdicts>/verdict_<label>_<list>_<n>.json, list in
pass_a | pass_b_run | final. A list without verdicts shows only the repo metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

from judge import metrics  # same folder

ALL_LISTS = ("pass_a", "pass_b_run", "final")


def fmt(value: float | None, digits: int = 2) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def judged(base: Path, run: dict, which: str) -> dict | None:
    files = sorted((base / run["verdicts"]).glob(f"verdict_{run['label']}_{which}_*.json"))
    if not files:
        return None
    return metrics(str(base / run["scores"]), run["label"], which, [str(f) for f in files])


def main() -> None:
    spec = Path(sys.argv[1])
    lists = ALL_LISTS if "--all-lists" in sys.argv[2:] else ("final",)
    base = spec.parent
    runs = json.loads(spec.read_text())
    acc = [
        (
            "| Model | Trọng số | Danh sách | Số bước | Tìm ra (recall) | Tìm ra bước chính | Chính xác (precision) | "
            "Bịa | Đúng người làm | Khớp chữ chặt (repo) | Phủ thời gian |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    speed = [
        (
            "| Model | Trọng số | Giây mỗi lượt gọi (trung vị) | Token sinh ra mỗi lượt (trung vị) | "
            "Tốc độ sinh (token/giây) | Cả job trên UI |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for run in runs:
        row = json.loads((base / run["scores"]).read_text())[run["label"]]
        for which in lists:
            if which not in row:
                continue
            repo = row[which]
            sem = judged(base, run, which)
            name = {"pass_a": "pass A lượt 1", "pass_b_run": "pass A lượt 2", "final": "cuối (UI)"}[
                which
            ]
            acc.append(
                f"| {run['model']} | {run['weights']} | {name} | {repo['predicted']} | "
                f"{fmt(sem and sem['sem_recall'])} | {fmt(sem and sem['sem_recall_essential'])} | "
                f"{fmt(sem and sem['sem_precision'])} | {fmt(sem and sem['wrong_rate'])} | "
                f"{fmt(sem and sem['actor_acc'])} | {fmt(repo['recall'])} | {fmt(repo['recall_time_only'])} |"
            )
        lat = [row[k] for k in ("latency_pass_a", "latency_pass_b") if k in row]
        job = run.get("job_s")
        speed.append(
            f"| {run['model']} | {run['weights']} | {fmt(mean(x['median_s'] for x in lat), 0)} | "
            f"{fmt(mean(x['out_tokens_median'] for x in lat), 0)} | {fmt(mean(x['out_tok_per_s'] for x in lat), 1)} | "
            f"{'–' if job is None else f'{job / 60:.0f} phút'} |"
        )
    print("\n".join(acc))
    print()
    print("\n".join(speed))


if __name__ == "__main__":
    main()
