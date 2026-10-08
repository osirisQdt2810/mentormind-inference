#!/bin/bash
# The MentorMind API the comparison runs go through: the checkout's own code on :8010, data in the checkout's
# data/work (so the videos stay in the UI afterwards). usage: MENTORMIND=<checkout> api.sh <model id> <answer tokens>
# The UI: VITE_API_BASE_URL=http://localhost:8010 npm --prefix $MENTORMIND/mentormind/frontend run dev -- --port 5175
set -euo pipefail
M=${MENTORMIND:?path of a mentormind-knowhow-ai checkout}
cd "$M"
export KNOWHOW_WORK_DIR=$M/data/work KNOWHOW_PROFILE_DB=$M/data/work/profiles.sqlite
export KNOWHOW_TRACE_DIR=$M/data/work/traces KNOWHOW_GUARD_AUDIT_DIR=$M/data/work/audit
export KNOWHOW_CORS_ORIGINS='["http://localhost:5175"]'
export KNOWHOW_VLM_MODEL=$1 KNOWHOW_VLM_MAX_TOKENS=$2 KNOWHOW_LLM_MAX_TOKENS=$2
exec uv run uvicorn mentormind.backend.app.main:app --host 127.0.0.1 --port 8010
