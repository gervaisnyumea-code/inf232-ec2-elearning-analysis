"""LLM usage logging and simple token/cost estimation utilities.

This module provides a single-file utility to estimate token usage and record
per-call usage to logs/llm_usage.json. It's intentionally small and local-only
(does not call external services).
"""
import json
import math
import time
import os
from pathlib import Path
from typing import Any, Dict

USAGE_FILE = Path(os.getenv('LLM_USAGE_FILE', 'logs/llm_usage.json'))


def ensure_usage_file():
    """Ensure the usage file and parent directory exist."""
    try:
        USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not USAGE_FILE.exists():
            USAGE_FILE.write_text(json.dumps([]))
    except Exception:
        # best-effort, callers should catch if needed
        pass


def estimate_tokens(s: str) -> int:
    """Rudimentary token estimator: ~4 chars per token (approx).

    This is only an estimate for cost-tracking; replace with a proper
    tokenizer if exact billing is required.
    """
    if not s:
        return 0
    return max(1, math.ceil(len(s) / 4.0))


def get_cost_per_1k(provider: str) -> float:
    """Return the configured cost (USD) per 1000 tokens for a provider.

    Environment variables used (if present):
      - MISTRAL_COST_PER_1K
      - GEMINI_COST_PER_1K
      - GROQ_COST_PER_1K

    Defaults are conservative placeholders; update your .env for accurate pricing.
    """
    defaults = {
        'mistral': float(os.getenv('MISTRAL_COST_PER_1K', '0.002')),
        'gemini': float(os.getenv('GEMINI_COST_PER_1K', '0.03')),
        'groq': float(os.getenv('GROQ_COST_PER_1K', '0.02')),
    }
    return defaults.get(provider, float(os.getenv(f'{provider.upper()}_COST_PER_1K', '0.01')))


def _read_all():
    try:
        raw = USAGE_FILE.read_text()
        return json.loads(raw) if raw.strip() else []
    except Exception:
        return []


def log_usage(provider: str, method: str, prompt_tokens: int, completion_tokens: int, cost: float, extra: Dict[str, Any] | None = None):
    """Append a usage entry (best-effort).

    Entry fields: ts, provider, method, prompt_tokens, completion_tokens, cost, extra
    """
    try:
        ensure_usage_file()
        entries = _read_all()
        entry = {
            'ts': int(time.time()),
            'provider': provider,
            'method': method,
            'prompt_tokens': int(prompt_tokens or 0),
            'completion_tokens': int(completion_tokens or 0),
            'cost': float(cost or 0.0),
            'extra': extra or {}
        }
        entries.append(entry)
        USAGE_FILE.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    except Exception:
        # do not crash the caller over logging failures
        pass


def summary():
    entries = _read_all()
    total_calls = len(entries)
    total_cost = sum(float(e.get('cost', 0.0)) for e in entries)
    by_provider = {}
    for e in entries:
        p = e.get('provider', 'unknown')
        by_provider[p] = by_provider.get(p, 0) + 1
    return {
        'total_calls': total_calls,
        'total_cost': total_cost,
        'by_provider': by_provider,
        'last': entries[-10:]
    }


def clear():
    try:
        ensure_usage_file()
        USAGE_FILE.write_text(json.dumps([]))
    except Exception:
        pass
