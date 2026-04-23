"""Helpers to read live streaming CSV produced by scripts/data_stream_simulator.py

Provides helpers to slice chronologically and to read recent windows.
"""
import os
from typing import Optional
import pandas as pd


def read_live_data(limit: int = 500, last_seconds: Optional[int] = None, since_t: Optional[float] = None, path: Optional[str] = None):
    """Read the live CSV stream and optionally filter by time window.

    Parameters
    ----------
    limit : int
        Maximum number of rows to return (tail selection).
    last_seconds : Optional[int]
        If provided and the CSV contains a 'timestamp' column, only rows newer
        than now - last_seconds will be returned.
    since_t : Optional[float]
        If provided and the CSV contains a numeric 't' column, only rows with
        t >= since_t will be returned.
    path : Optional[str]
        Path to the CSV file (defaults to data/stream/live_data.csv).
    """
    path = path or os.path.join(os.getcwd(), 'data', 'stream', 'live_data.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=['timestamp'])
        if last_seconds is not None and 'timestamp' in df.columns:
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(seconds=int(last_seconds))
            df = df[df['timestamp'] >= cutoff]
        if since_t is not None and 't' in df.columns:
            df = df[df['t'] >= since_t]
        if limit:
            df = df.tail(limit)
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def slice_chronological(df: pd.DataFrame, start_ts: Optional[pd.Timestamp] = None, end_ts: Optional[pd.Timestamp] = None, last_n: Optional[int] = None):
    """Return a chronological slice of the live stream DataFrame.

    Use start_ts/end_ts (pandas-compatible) or last_n rows as a fallback.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if start_ts is not None:
        out = out[out['timestamp'] >= pd.to_datetime(start_ts)]
    if end_ts is not None:
        out = out[out['timestamp'] <= pd.to_datetime(end_ts)]
    if last_n is not None:
        out = out.tail(last_n)
    return out.reset_index(drop=True)
