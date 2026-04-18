#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_BIN="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PY_BIN" ]]; then
  echo "Missing virtual environment python at $PY_BIN"
  echo "Create it with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

exec "$PY_BIN" "$ROOT_DIR/run_daily.py"
