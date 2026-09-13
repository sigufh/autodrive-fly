#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export AUTODRIVE_FLY_ROOT="${PROJECT_ROOT}"

exec "${PROJECT_ROOT}/.venv/bin/uvicorn" fly_emotion_api.main:app \
  --app-dir "${PROJECT_ROOT}/apps/api/src" \
  --host 127.0.0.1 \
  --port 8000
