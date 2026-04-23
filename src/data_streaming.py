"""Helpers to read live streaming CSV produced by scripts/data_stream_simulator.py"""
import os
import pandas as pd


def read_live_data(limit=500):
    path = os.path.join(os.getcwd(), 'data', 'stream', 'live_data.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=['timestamp'])
        return df.tail(limit)
    except Exception:
        return pd.DataFrame()
