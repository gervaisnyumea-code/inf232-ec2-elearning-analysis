#!/usr/bin/env python3
"""Run BrainNet ensemble on live stream, save predictions CSV and generate a report ZIP.

Usage: python scripts/run_ensemble_live_and_report.py
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration import BrainNet
from src.data_streaming import read_live_data
from src.reporting import generate_report


def main(weights=None):
    bn = BrainNet(auto_load=True)
    if weights:
        bn.set_weights(weights, persist=True)
    df_live = read_live_data(limit=1000)
    if df_live.empty:
        print('No live data found, exiting')
        return 1

    out_csv = Path('reports') / f'ensemble_preds_{int(time.time())}.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    saved_path = bn.save_ensemble_predictions(df_live, out_csv=str(out_csv))
    print('Saved ensemble predictions to', saved_path)

    rpt_zip = generate_report(df_live, out_dir='reports')
    print('Generated report:', rpt_zip)
    return 0


if __name__ == '__main__':
    # Example weights - adjust as needed
    weights = {
        'classifier_model_meta.pkl': 2.0,
        'classifier_model.pkl': 1.0,
        'regression_model_meta.pkl': 0.5,
        'regression_model.pkl': 0.5,
    }
    sys.exit(main(weights=weights))
