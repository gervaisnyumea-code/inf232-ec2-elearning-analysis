#!/usr/bin/env python3
"""Wrap existing trained models with metadata (scaler + feature_names) and save as *_meta.pkl
Usage: python scripts/save_model_metadata.py
"""
import os
from pathlib import Path
import joblib
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_cleaning import get_feature_matrix


def main():
    model_dir = Path('data/models')
    if not model_dir.exists():
        print('data/models not found')
        return
    data_path = Path('data/processed/elearning_clean.csv')
    if not data_path.exists():
        print('Processed data not found: run full pipeline first')
        return
    import pandas as pd
    df = pd.read_csv(data_path)
    X, _, _ = get_feature_matrix(df)
    feature_names = list(X.columns)

    for name in ['classifier_model.pkl', 'regression_model.pkl']:
        src = model_dir / name
        if not src.exists():
            print(f'{name} not found — skipping')
            continue
        meta_name = src.stem + '_meta.pkl'
        dst = model_dir / meta_name
        if dst.exists():
            print(f'{dst} already exists — skipping')
            continue
        print('Wrapping', src)
        obj = joblib.load(src)
        scaler = None
        scaler_path = model_dir / 'scaler.pkl'
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        meta = {'model': obj, 'scaler': scaler, 'feature_names': feature_names}
        joblib.dump(meta, dst)
        print('Saved', dst)


if __name__ == '__main__':
    main()
