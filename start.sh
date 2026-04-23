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

# Ensure logs directory
mkdir -p logs

# Start data simulator in background if not already running
if ! ss -ltnp 2>/dev/null | grep -q ":454"; then
  echo "Starting data simulator in background..."
  nohup python3 scripts/data_stream_simulator.py > logs/streamer.log 2> logs/streamer.err &
  echo $! > logs/streamer.pid || true
fi

# Start Streamlit in background (nohup) and open browser when ready
nohup ./venv/bin/streamlit run app/main.py --server.port "$PORT" > logs/streamlit.out 2> logs/streamlit.err < /dev/null &
echo $! > logs/streamlit.pid || true

# wait for server and open default browser
for i in {1..30}; do
  if curl -sS -m 2 "http://localhost:$PORT" >/dev/null 2>&1; then
    if command -v xdg-open >/dev/null 2>&1; then
      xdg-open "http://localhost:$PORT" >/dev/null 2>&1 || true
    elif command -v open >/dev/null 2>&1; then
      open "http://localhost:$PORT" >/dev/null 2>&1 || true
    fi
    echo "Streamlit started at http://localhost:$PORT"
    exit 0
  fi
  sleep 1
done

echo "Streamlit did not respond in time; check logs/logs/streamlit.out or logs/streamlit.err"
