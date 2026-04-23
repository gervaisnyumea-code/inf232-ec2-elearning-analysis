#!/usr/bin/env bash
set -euo pipefail

# Activate venv if present
if [ -f venv/bin/activate ]; then
  . venv/bin/activate
fi

# Export .env variables into this shell so subprocesses (Streamlit) inherit them
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . .env
  set +a
fi

PORT="${STREAMLIT_SERVER_PORT:-8501}"
exec streamlit run app/main.py --server.port "$PORT"
