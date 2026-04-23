"""Utilities to load and persist .env files into os.environ.

Functions:
- load_dotenv(path='.env', override=False): load variables into os.environ and return dict of loaded values.
- persist_env(updates: dict, path='.env'): persist updates to .env file (preserve comments) and update os.environ.

This is intentionally small and dependency-free to avoid adding python-dotenv.
"""
from pathlib import Path
import os
import shlex


def _parse_line(line: str):
    if '=' not in line:
        return None
    key, val = line.split('=', 1)
    key = key.strip()
    val = val.strip()
    # remove surrounding quotes
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        val = val[1:-1]
    # expand environment variables in value
    try:
        val = os.path.expandvars(val)
    except Exception:
        pass
    return key, val


def load_dotenv(path: str = None, override: bool = False):
    """Load key=value pairs from a .env file into os.environ.

    Returns a dict of keys that were set.
    """
    path = Path(path or os.getenv('ENV_PATH', '.env'))
    loaded = {}
    if not path.exists():
        return loaded
    try:
        text = path.read_text()
    except Exception:
        return loaded

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parsed = _parse_line(line)
        if not parsed:
            continue
        k, v = parsed
        if override or k not in os.environ:
            os.environ[k] = v
            loaded[k] = v
    return loaded


def persist_env(updates: dict, path: str = None):
    """Persist updates into a .env file (create if missing). Also update os.environ.

    updates: dict of str->str
    path: file path to .env
    Returns True on success, False otherwise.
    """
    path = Path(path or os.getenv('ENV_PATH', '.env'))
    try:
        if path.exists():
            lines = path.read_text().splitlines()
        else:
            lines = []
        new_lines = []
        found = set()
        for line in lines:
            if line.strip().startswith('#') or '=' not in line:
                new_lines.append(line)
                continue
            key = line.split('=', 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                found.add(key)
            else:
                new_lines.append(line)
        for k, v in updates.items():
            if k not in found:
                new_lines.append(f"{k}={v}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(new_lines) + "\n")
        # update os.environ
        for k, v in updates.items():
            os.environ[k] = str(v)
        return True
    except Exception:
        return False
