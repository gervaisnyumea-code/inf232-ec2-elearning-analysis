#!/usr/bin/env python3
"""
Simple CSV data streamer that appends oscillating values to data/stream/live_data.csv
Usage: python scripts/data_stream_simulator.py [interval_seconds]
"""
import os, time, math, random, csv
from datetime import datetime

ROOT = os.getcwd()
OUT_DIR = os.path.join(ROOT, 'data', 'stream')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, 'live_data.csv')


def append_row(t, value1, value2):
    write_header = not os.path.exists(OUT_FILE)
    with open(OUT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['timestamp','t','value1','value2'])
        writer.writerow([datetime.utcnow().isoformat(), t, value1, value2])
        f.flush()


def run(interval=0.5):
    t0 = 0.0
    i = 0
    while True:
        t = i * interval
        value1 = math.sin(t * 0.5) + 0.2 * math.sin(t * 3.0) + 0.05 * (random.random()-0.5)
        value2 = math.cos(t * 0.25) + 0.05 * (random.random()-0.5)
        append_row(t, value1, value2)
        time.sleep(interval)
        i += 1


if __name__ == '__main__':
    import sys
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    print('Starting data stream simulator, writing to', OUT_FILE, 'interval=', interval)
    run(interval)
