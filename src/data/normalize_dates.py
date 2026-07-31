#!/usr/bin/env python3
"""
normalize_dates.py — Normalize date fields to YYYY-MM-DD across all CSV and
JSONL files in a directory tree.

Walks the input directory, and for every .csv/.jsonl file found, rewrites any
column/key whose name contains "date" (case-insensitive) into ISO YYYY-MM-DD
format. Files are processed one at a time. The output directory mirrors the
input directory's structure; the input directory itself is never modified.

Usage:
    python3 normalize_dates.py [input_dir] [output_dir]

Defaults:
    input_dir  = current directory
    output_dir = ./normalized
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

DATE_KEY_MARKER = "date"

# Candidate input formats, tried in order until one matches.
DATE_FORMATS = [
    "%Y-%m-%d",   # 2015-01-08 (already ISO)
    "%d %b %Y",   # 3 Feb 2010
    "%Y%m%d",     # 20150131
    "%d-%b-%Y",   # 3-Feb-2010
    "%m/%d/%Y",   # 02/03/2010
    "%d/%m/%Y",   # 03/02/2010
]


def is_date_key(key: str) -> bool:
    return key is not None and DATE_KEY_MARKER in key.lower()


def normalize_value(value: str, warnings: list) -> str:
    stripped = value.strip()
    if not stripped:
        return value
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(stripped, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    warnings.append(stripped)
    return value


def process_csv(src: Path, dst: Path) -> int:
    warnings = []
    with src.open(newline="", encoding="utf-8-sig") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        date_keys = [k for k in fieldnames if is_date_key(k)]
        rows = []
        for row in reader:
            row.pop(None, None)  # drop any ragged extra fields from malformed lines
            for k in date_keys:
                row[k] = normalize_value(row[k], warnings)
            rows.append(row)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    for w in warnings:
        print(f"  warning: could not parse date value {w!r} in {src}", file=sys.stderr)
    return len(rows)


def process_jsonl(src: Path, dst: Path) -> int:
    warnings = []
    count = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8-sig") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for k in list(record.keys()):
                if is_date_key(k) and isinstance(record[k], str):
                    record[k] = normalize_value(record[k], warnings)
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    for w in warnings:
        print(f"  warning: could not parse date value {w!r} in {src}", file=sys.stderr)
    return count


def main():
    args = sys.argv[1:]
    input_dir = Path(args[0]) if len(args) > 0 else Path(".")
    output_dir = Path(args[1]) if len(args) > 1 else Path("./normalized")

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in (".csv", ".jsonl")
        and output_dir not in p.resolve().parents
    )

    if not files:
        print(f"No .csv/.jsonl files found under {input_dir}")
        return

    total = 0
    for src in files:
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        if src.suffix.lower() == ".csv":
            n = process_csv(src, dst)
        else:
            n = process_jsonl(src, dst)
        print(f"{rel}: {n} rows -> {dst}")
        total += 1

    print(f"\nProcessed {total} files. Output written to {output_dir}")


if __name__ == "__main__":
    main()
