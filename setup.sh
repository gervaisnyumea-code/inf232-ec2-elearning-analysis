#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python not found ($PYTHON). Install Python 3.8+ or set PYTHON env var."
  exit 1
fi

# Create virtualenv if missing
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  "$PYTHON" -m venv venv
fi

# Activate and install
. venv/bin/activate
pip install --upgrade pip
if [ -f requirements.txt ]; then
  echo "Installing requirements..."
  pip install -r requirements.txt
else
  echo "No requirements.txt found; skipping pip install."
fi

# Create .env from example if missing
if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo "Created .env from .env.example. Edit .env to add secrets."
fi

# Prepare directories
mkdir -p logs reports data/stream data/models app/static/icons

echo "Setup complete. Activate the environment with: source venv/bin/activate"
