#!/usr/bin/env python3
# Simple repo-wide emoji -> svg replacement script
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

emoji_map = {
    "✅": "check",
    "❌": "cross",
    "📊": "chart",
    "📈": "up",
    "📉": "down",
    "🧠": "brain",
    "🔁": "loop",
    "📌": "pin",
    "⚠️": "warning",
    "⚠": "warning",
    "🎯": "target",
    "🔍": "search",
    "🌡️": "thermometer",
    "🟢": "circle_green",
    "🔴": "circle_red",
}

EXTS = {'.md', '.rst', '.txt', '.markdown'}

TEMPLATE = "<img src=app/static/icons/{fname}.svg alt={name} width=18/>"

count = 0
for root, dirs, files in os.walk(REPO_ROOT):
    # skip large or virtualenv directories
    if any(skip in root for skip in ['.git', 'venv', 'node_modules', '__pycache__']):
        continue
    for f in files:
        _, ext = os.path.splitext(f)
        if ext.lower() not in EXTS:
            continue
        path = os.path.join(root, f)
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                txt = fh.read()
        except Exception:
            # skip binary or unreadable
            continue
        new_txt = txt
        for emoji, name in emoji_map.items():
            if emoji in new_txt:
                new_txt = new_txt.replace(emoji, TEMPLATE.format(fname=name, name=name))
        if new_txt != txt:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(new_txt)
            print('Updated', path)
            count += 1

print('Done. Files updated:', count)
