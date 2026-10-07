#!/bin/bash
# Vast.ai deploy from a clone of this repo: bash run.sh  (the real script: scripts/vast/run.sh)
exec bash "$(dirname "$0")/scripts/vast/run.sh" "$@"
