#!/usr/bin/env python3
"""Cross-platform start script for INF232 app.

- Loads .env into environment
- Starts Streamlit using venv python if present
- Waits for server to be reachable then opens the browser
"""
import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path
import socket

ROOT = Path(__file__).parent
VENV_DIR = ROOT / 'venv'
ENV_FILE = ROOT / '.env'


def load_env(path: Path = None):
    try:
        from src.env_loader import load_dotenv
        load_dotenv(str(path or ENV_FILE), override=False)
    except Exception:
        # simple fallback: parse key=val lines
        p = path or ENV_FILE
        if p.exists():
            for line in p.read_text().splitlines():
                line=line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k,v = line.split('=',1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


def is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except Exception:
        return False


def start_streamlit(port: int = 8501):
    # choose python executable from venv if present
    python = VENV_DIR / ('Scripts' if os.name == 'nt' else 'bin') / 'python'
    if not python.exists():
        python = shutil_which('python3') or shutil_which('python')
    cmd = [str(python), '-m', 'streamlit', 'run', 'app/main.py', '--server.port', str(port)]
    print('Running:', ' '.join(cmd))
    # Start in background
    if os.name == 'nt':
        # Windows: create new window
        subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        subprocess.Popen(cmd, preexec_fn=os.setsid)


def shutil_which(name):
    from shutil import which
    return which(name)


def main():
    load_env()
    port = int(os.getenv('STREAMLIT_SERVER_PORT', '8501'))
    start_streamlit(port)
    # wait for server
    url = f'http://localhost:{port}'
    for i in range(30):
        if is_port_open('127.0.0.1', port):
            print('Server up, opening browser at', url)
            webbrowser.open(url)
            return
        time.sleep(1)
    print('Server did not respond in time; opening browser anyway')
    webbrowser.open(url)

if __name__ == '__main__':
    main()
