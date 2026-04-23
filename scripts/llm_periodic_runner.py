#!/usr/bin/env python3
"""Periodic runner for LLM pipeline: loads live data snapshot and runs the orchestration.
Produces JSON outputs in the reports/ directory. Respects quota configured in LLMClient.
"""
import os
import time
import json
from pathlib import Path

import pandas as pd

# Charger .env si présent
try:
    from src.env_loader import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.llm_orchestrator import LLMOrchestrator


def load_latest_snapshot(path: str = 'data/stream/live_data.csv', rows: int = 500) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if rows and len(df) > rows:
        df = df.tail(rows)
    return df


def main():
    interval = int(os.getenv('LLM_PERIODIC_INTERVAL_SEC', '3600'))
    output_dir = Path(os.getenv('EXPORT_DIR', 'reports'))
    output_dir.mkdir(parents=True, exist_ok=True)
    orch = LLMOrchestrator()

    while True:
        df = load_latest_snapshot()
        if df.empty:
            print('No live data yet.')
        else:
            try:
                res = orch.run_full_pipeline(df)
                timestamp = int(time.time())
                out_path = output_dir / f"llm_output_{timestamp}.json"
                out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
                print(f'Wrote {out_path}')
            except Exception as e:
                print('Error running LLM pipeline:', e)
        time.sleep(interval)


if __name__ == '__main__':
    main()
