"""Upload one LASI copy through the UI (Playwright), wait for the job, screenshot it.

usage: cmp_upload.py <video> <process_id> <shots_dir> <label>
Prints a JSON line {label, video_id, process_id, wall_s, summary} at the end.
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

UI = os.environ.get("BENCH_UI", "http://localhost:5175")
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
    t0 = time.time()
    page.locator("#btn-upload-video").click()
    done = page.locator("#btn-review-video").or_(page.get_by_text("Phân tích thất bại")).first
    mid_shot = False
    while True:
        try:
            done.wait_for(timeout=60_000)
            break
        except Exception:
            elapsed = time.time() - t0
            log(f"{label}: running {elapsed / 60:.0f} min")
            if not mid_shot and elapsed > 300:
                page.screenshot(path=str(shots / f"{label}-progress.png"), full_page=True)
                mid_shot = True
            if elapsed > 4 * 3600:
                raise
    wall = time.time() - t0
    failed = page.get_by_text("Phân tích thất bại").count() > 0
    page.screenshot(path=str(shots / f"{label}-done.png"), full_page=True)
    profiles = api(f"/profiles?process_id={process}")
    vid = profiles[0]["evidence"]["video_id"] if profiles else None
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
