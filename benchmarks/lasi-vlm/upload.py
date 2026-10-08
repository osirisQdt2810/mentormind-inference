"""Upload one LASI copy through the UI (Playwright), wait for the job, screenshot it.

usage: upload.py <video> <process_id> <shots_dir> <label>
Upload with cut method "scene" and label "public", as a user would; screenshots <label>-progress/-done/
-review/-step.png. Prints one JSON line {label, video_id, process_id, wall_s, failed, steps} at the end:
video_id from the upload's own answer, steps = the drafts of that video. wall_s uses a monotonic clock
(time the machine sleeps does not count); BENCH_MAX_HOURS (default 6) stops the wait.
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

UI = os.environ.get("BENCH_UI", "http://localhost:5175")
MAX_S = float(os.environ.get("BENCH_MAX_HOURS", "6")) * 3600
API = os.environ.get("BENCH_API", "http://localhost:8010")
video, process, shots, label = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
shots.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def api(path: str) -> object:
    with urllib.request.urlopen(API + path, timeout=30) as r:
        return json.load(r)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.goto(f"{UI}/?tab=upload")
    page.locator("#input-video").set_input_files(str(video))
    page.locator("#input-video-process").fill(process)
    page.locator("#select-video-method").select_option("scene")
    page.locator("#select-video-classification").select_option("public")
    t0 = time.monotonic()
    with page.expect_response(
        lambda r: r.request.method == "POST" and r.url.endswith("/videos")
    ) as uploaded:
        page.locator("#btn-upload-video").click()
    vid = uploaded.value.json()["video_id"]
    log(f"{label}: uploaded as {vid}")
    done = page.locator("#btn-review-video").or_(page.get_by_text("Phân tích thất bại")).first
    mid_shot = False
    while True:
        try:
            done.wait_for(timeout=60_000)
            break
        except Exception:
            elapsed = time.monotonic() - t0
            log(f"{label}: running {elapsed / 60:.0f} min")
            if not mid_shot and elapsed > 300:
                page.screenshot(path=str(shots / f"{label}-progress.png"), full_page=True)
                mid_shot = True
            if elapsed > MAX_S:
                raise
    wall = time.monotonic() - t0
    failed = page.get_by_text("Phân tích thất bại").count() > 0
    page.screenshot(path=str(shots / f"{label}-done.png"), full_page=True)
    profiles = [
        p for p in api(f"/profiles?process_id={process}") if p["evidence"].get("video_id") == vid
    ]
    if not failed:
        page.locator("#btn-review-video").click()
        page.locator(".profile-item").first.wait_for(timeout=60_000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(shots / f"{label}-review.png"))
        page.locator(".profile-item").first.click()
        page.wait_for_timeout(2000)
        page.screenshot(path=str(shots / f"{label}-step.png"))
    browser.close()

print(
    json.dumps(
        {
            "label": label,
            "video_id": vid,
            "process_id": process,
            "wall_s": round(wall, 1),
            "failed": failed,
            "steps": len(profiles),
        },
        ensure_ascii=False,
    ),
    flush=True,
)
