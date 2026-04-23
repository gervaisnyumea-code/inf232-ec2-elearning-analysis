"""Helpers to read live streaming CSV produced by scripts/data_stream_simulator.py

Provides helpers to slice chronologically and to read recent windows.
"""
import os
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Get project root - tries to find .env or setup.py to determine root
_PROJECT_ROOT = None

def _get_project_root():
    """Get the project root directory."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT
    # Try to find .env or setup.py in parent directories
    cwd = Path(os.getcwd())
    for parent in [cwd] + list(cwd.parents):
        if (parent / '.env').exists() or (parent / 'setup.py').exists():
            _PROJECT_ROOT = parent
            return parent
    # Fallback to cwd
    _PROJECT_ROOT = cwd
    return cwd


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
        Path to the CSV file (defaults to <project_root>/data/stream/live_data.csv).
    """
    if path is None:
        project_root = _get_project_root()
        default_path = project_root / 'data' / 'stream' / 'live_data.csv'
        path = str(default_path)
    
    if not os.path.exists(path):
        logger.warning(f"Live data file not found: {path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(path)
        
        # Handle timestamp parsing - if there are future timestamps, adjust them to present
        if 'timestamp' in df.columns:
            # Parse timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Check if timestamps are in the future (common issue with simulated data)
            now = pd.Timestamp.now()
            max_ts = df['timestamp'].max()
            
            if max_ts > now:
                # Calculate time difference
                time_diff = max_ts - now
                
                # If the difference is more than 1 day, it's likely simulated data with wrong year
                # Replace the year with current year
                if time_diff > pd.Timedelta(days=1):
                    current_year = now.year
                    # Replace year in all timestamps while keeping month, day, time
                    df['timestamp'] = df['timestamp'].map(
                        lambda ts: ts.replace(year=current_year) if hasattr(ts, 'replace') else ts
                    )
                    adjusted_max = df['timestamp'].max()
                    # If still in future after year adjustment, shift back
                    if adjusted_max > now:
                        shift = adjusted_max - now
                        df['timestamp'] = df['timestamp'] - shift
                    logger.info(f"Adjusted timestamps year to {current_year} (data was from year {max_ts.year})")
                else:
                    # Just shift back if difference is small
                    if time_diff > pd.Timedelta(0):
                        df['timestamp'] = df['timestamp'] - time_diff
                        logger.info(f"Adjusted timestamps back by {time_diff} (data was slightly in the future)")
        
        # ensure chronological ordering by timestamp (ascending)
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp', ascending=True)
        
        if last_seconds is not None and 'timestamp' in df.columns:
            import datetime
            now = datetime.datetime.now()
            now = pd.Timestamp(now)
            cutoff = now - pd.Timedelta(seconds=int(last_seconds))
            df = df[df['timestamp'] >= cutoff]
            # FALLBACK: si pas de données après filter (data old), lire quand même sans filter
            if len(df) == 0:
                # Re-read without time filter
                df_full = pd.read_csv(path)
                if 'timestamp' in df_full.columns:
                    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
                    df_full = df_full.sort_values('timestamp', ascending=True)
                    df = df_full.tail(limit) if limit else df_full
                else:
                    df = df_full.tail(limit) if limit else df_full
            logger.debug(f"Filtered to last {last_seconds}s. Cutoff: {cutoff}, remaining rows: {len(df)}")
        
        if since_t is not None and 't' in df.columns:
            df = df[df['t'] >= since_t]
        
        if limit:
            df = df.tail(limit)
        
        # after tail, make sure ordering remains chronological
        try:
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp', ascending=True)
        except Exception:
            pass
        
        return df.reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error reading live data from {path}: {e}")
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
