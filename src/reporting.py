"""Reporting helpers: detect periodic points (simple peaks) and generate a zipped report (CSV + PNG).
"""
import os
import zipfile
import time
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_periodic_points(df, time_col='t', value_col='value1', order=1):
    """Return DataFrame rows corresponding to simple local peaks.

    This is a lightweight peak detector: a point is a peak if it's the maximum
    within a small neighbourhood (order). Returns an empty DataFrame if none.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    vals = df[value_col].values
    idx = []
    for i in range(order, len(vals) - order):
        window = vals[i - order: i + order + 1]
        # peak if central value strictly max and larger than local mean
        if vals[i] == window.max() and vals[i] > (np.mean(np.delete(window, order)) if len(window) > 1 else window.max()):
            idx.append(i)
    if not idx:
        return pd.DataFrame()
    return df.iloc[idx].reset_index(drop=True)


def generate_report(df, value_col='value1', out_dir='reports'):
    """Generate a simple report from a live-data snapshot.

    Saves:
    - CSV snapshot
    - CSV peaks
    - PNG plot with peaks annotated
    - ZIP archive containing the above

    Returns the path to the ZIP archive (str).
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time())
    csv_path = Path(out_dir) / f"report_{ts}.csv"
    peaks_path = Path(out_dir) / f"report_{ts}_peaks.csv"
    fig_path = Path(out_dir) / f"report_{ts}.png"
    zip_path = Path(out_dir) / f"report_{ts}.zip"

    # snapshot
    try:
        df.to_csv(csv_path, index=False)
    except Exception:
        df.copy().to_csv(csv_path, index=False)

    # peaks
    peaks = find_periodic_points(df, time_col='t', value_col=value_col, order=1)
    peaks.to_csv(peaks_path, index=False)

    # plot
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(df['t'], df[value_col], label=value_col)
        if not peaks.empty:
            plt.scatter(peaks['t'], peaks[value_col], color='red', label='peaks')
        plt.legend()
        plt.xlabel('t')
        plt.title(f'Report snapshot {ts}')
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
    except Exception:
        # fallback: create an empty file
        Path(fig_path).write_text('')

    # zip
    with zipfile.ZipFile(zip_path, 'w') as z:
        if csv_path.exists():
            z.write(csv_path, arcname=csv_path.name)
        if peaks_path.exists():
            z.write(peaks_path, arcname=peaks_path.name)
        if fig_path.exists():
            z.write(fig_path, arcname=fig_path.name)

    return str(zip_path)
