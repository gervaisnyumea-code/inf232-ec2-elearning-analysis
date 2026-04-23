#!/usr/bin/env python3
"""Periodic runner for LLM pipeline: loads live data snapshot and runs the orchestration.
Produces JSON outputs in the reports/ directory. Respects quota configured in LLMClient.
"""
import os
import time
import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Charger .env si présent
try:
    from src.env_loader import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.llm_orchestrator import LLMOrchestrator


def load_latest_snapshot(path: str = None, rows: int = 500) -> pd.DataFrame:
    """Load the latest snapshot from live_data.csv.
    
    Uses absolute path based on project root by default.
    """
    if path is None:
        path = str(project_root / 'data' / 'stream' / 'live_data.csv')
    
    p = Path(path)
    if not p.exists():
        logger = None
        try:
            import logging
            logger = logging.getLogger(__name__)
        except Exception:
            pass
        if logger:
            logger.warning(f"Live data file not found: {path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(p)
        if rows and len(df) > rows:
            df = df.tail(rows)
        return df
    except Exception as e:
        if logger:
            logger.error(f"Error reading {path}: {e}")
        return pd.DataFrame()


def main():
    interval = int(os.getenv('LLM_PERIODIC_INTERVAL_SEC', '3600'))
    output_dir = Path(os.getenv('EXPORT_DIR', 'reports'))
    output_dir.mkdir(parents=True, exist_ok=True)
    orch = LLMOrchestrator()

    question = os.getenv('LLM_PERIODIC_QUESTION', 'Please provide a consolidated analysis of the latest live data.')
    rounds = int(os.getenv('LLM_PERIODIC_ROUNDS', '1'))
    window = int(os.getenv('LLM_PERIODIC_WINDOW', '3600'))

    while True:
        df = load_latest_snapshot()
        if df.empty:
            print('No live data yet.')
        else:
            try:
                res = orch.concert_and_merge(question, rounds=rounds, include_data=True, data_window_sec=window, force_real=(os.getenv('LLM_CALLS_ENABLED','false').lower() in ('1','true','yes')))
                timestamp = int(time.time())
                out_path = output_dir / f"llm_output_{timestamp}.json"
                out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
                print(f'Wrote {out_path}')
            except Exception as e:
                print('Error running LLM pipeline:', e)
        time.sleep(interval)


if __name__ == '__main__':
    main()
