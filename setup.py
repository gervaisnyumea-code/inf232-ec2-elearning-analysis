#!/usr/bin/env python3
"""Cross-platform setup script for INF232 app.

This script attempts to prepare a machine with the project requirements:
- ensures Python 3.8+ is available (attempts to install via package managers when possible)
- creates a virtual environment at ./venv
- installs pip requirements from requirements.txt
- copies .env.example to .env if missing
- creates required directories
- optionally starts the app and opens the browser

Run as: python3 setup.py
"""
import sys
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
import webbrowser

ROOT = Path(__file__).parent
VENV_DIR = ROOT / 'venv'
REQUIREMENTS = ROOT / 'requirements.txt'
ENV_EXAMPLE = ROOT / '.env.example'
ENV_FILE = ROOT / '.env'

MIN_PY = (3, 8)


def run(cmd, check=True, shell=False):
    print('> ', ' '.join(cmd) if isinstance(cmd, (list, tuple)) else cmd)
    return subprocess.run(cmd, check=check, shell=shell)


def find_executable(names):
    for name in names:
        p = shutil.which(name)
        if p:
            return p
    return None


def ensure_system_python():
    """Attempt to ensure a system python3 executable exists and meets the minimum version.
    Returns path to python executable.
    """
    py = find_executable(['python3', 'python'])
    if py:
        try:
            out = subprocess.check_output([py, '--version'], stderr=subprocess.STDOUT).decode()
            ver = out.strip().split()[-1]
            parts = tuple(int(x) for x in ver.split('.')[:2])
            if parts >= MIN_PY:
                print(f'Found python at {py} version {ver}')
                return py
            else:
                print(f'Python {ver} found at {py} but < {MIN_PY}.')
        except Exception:
            pass

    system = platform.system()
    print('No suitable Python found. Attempting to install via package manager (may require sudo).')
    try:
        if system == 'Linux':
            if find_executable(['apt-get']):
                run(['sudo', 'apt-get', 'update'])
                run(['sudo', 'apt-get', 'install', '-y', 'python3', 'python3-venv', 'python3-pip'])
                return find_executable(['python3', 'python'])
            if find_executable(['dnf']):
                run(['sudo', 'dnf', 'install', '-y', 'python3'])
                return find_executable(['python3', 'python'])
            if find_executable(['pacman']):
                run(['sudo', 'pacman', '-Syu', '--noconfirm', 'python'])
                return find_executable(['python3', 'python'])
        elif system == 'Darwin':
            if find_executable(['brew']):
                run(['brew', 'install', 'python'])
                return find_executable(['python3', 'python'])
        elif system == 'Windows':
            # try winget or choco
            if find_executable(['winget']):
                run(['winget', 'install', '--silent', '--accept-package-agreements', 'Python.Python.3'])
                return find_executable(['python', 'python3'])
            if find_executable(['choco']):
                run(['choco', 'install', 'python', '-y'])
                return find_executable(['python', 'python3'])
    except subprocess.CalledProcessError as e:
        print('Package manager install failed:', e)

    raise RuntimeError('Python 3.8+ is required but could not be installed automatically. Please install Python and re-run setup.')


def create_venv(python_exe: str = None):
    if VENV_DIR.exists():
        print('Virtualenv already exists at', VENV_DIR)
        return
    py = python_exe or find_executable(['python3', 'python'])
    if not py:
        py = ensure_system_python()
    print('Creating virtual environment with', py)
    run([py, '-m', 'venv', str(VENV_DIR)])


def install_requirements():
    if not REQUIREMENTS.exists():
        print('No requirements.txt found, skipping pip install.')
        return
    pip = VENV_DIR / ('Scripts' if platform.system() == 'Windows' else 'bin') / 'pip'
    if not pip.exists():
        print('pip not found in venv, attempting to use python -m pip')
        python = VENV_DIR / ('Scripts' if platform.system() == 'Windows' else 'bin') / 'python'
        run([str(python), '-m', 'pip', 'install', '--upgrade', 'pip'])
        run([str(python), '-m', 'pip', 'install', '-r', str(REQUIREMENTS)])
        return
    run([str(pip), 'install', '--upgrade', 'pip'])
    run([str(pip), 'install', '-r', str(REQUIREMENTS)])


def copy_env():
    if not ENV_FILE.exists() and ENV_EXAMPLE.exists():
        shutil.copy(ENV_EXAMPLE, ENV_FILE)
        print('.env created from .env.example — please edit to add secrets (API keys)')


def prepare_dirs():
    for d in ['logs', 'reports', 'data/stream', 'data/models', 'app/static/icons']:
        Path(d).mkdir(parents=True, exist_ok=True)


def start_and_open(port=None):
    port = port or os.getenv('STREAMLIT_SERVER_PORT', '8501')
    # Use venv python to run streamlit
    python = VENV_DIR / ('Scripts' if platform.system() == 'Windows' else 'bin') / 'python'
    if not python.exists():
        python = find_executable(['python3', 'python'])
    cmd = [str(python), '-m', 'streamlit', 'run', 'app/main.py', '--server.port', str(port)]
    print('Starting streamlit server: ', ' '.join(cmd))
    # start detached process
    if platform.system() == 'Windows':
        subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        subprocess.Popen(cmd, preexec_fn=os.setsid)
    # wait a bit and open browser
    time.sleep(3)
    url = f'http://localhost:{port}'
    try:
        webbrowser.open(url)
        print('Opened browser at', url)
    except Exception as e:
        print('Failed to open browser:', e)


def main():
    try:
        py = find_executable(['python3', 'python'])
        if not py:
            py = ensure_system_python()
    except Exception as e:
        print('Error ensuring python:', e)
        sys.exit(1)

    create_venv(py)
    install_requirements()
    copy_env()
    prepare_dirs()
    print('Setup complete.')

    # Start server and open browser
    start_and_open()


if __name__ == '__main__':
    main()
