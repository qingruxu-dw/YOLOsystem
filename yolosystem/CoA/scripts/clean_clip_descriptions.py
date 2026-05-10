#!/usr/bin/env python3
"""Clean CLIP descriptions file by deduplicating labels and keeping top-K per image."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "outputs" / "clip_dehaze" / "clip_descriptions.txt"
OUT = ROOT / "outputs" / "clip_dehaze" / "clip_descriptions.cleaned.txt"
K = 3


def parse_line(line):
    if '->' not in line:
        return None, None
    name, rest = line.split('->', 1)
    name = name.strip()
    pairs = re.findall(r"\(\s*'([^']+)'\s*,\s*([0-9.eE+-]+)\s*\)", rest)
    return name, [(lbl, float(score)) for lbl, score in pairs]


def clean_pairs(pairs):
    d = {}
    for lbl, sc in pairs:
        d[lbl] = max(d.get(lbl, -float('inf')), sc)
    items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return items[:K]


def main():
    if not IN.exists():
        print(f"Input not found: {IN}")
        return
    out_lines = []
    with IN.open('r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            name, pairs = parse_line(ln)
            if name is None:
                out_lines.append(ln)
                continue
            cleaned = clean_pairs(pairs)
            out_lines.append(f"{name} -> {cleaned}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as f:
        for ln in out_lines:
            f.write(ln + "\n")

    print(f"Wrote cleaned file: {OUT}")


if __name__ == '__main__':
    main()
