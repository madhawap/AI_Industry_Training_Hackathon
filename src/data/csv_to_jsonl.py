#!/usr/bin/env python3
"""
csv_to_jsonl.py — Convert a CSV file to JSONL format.

Usage:
    python3 csv_to_jsonl.py input.csv [output.jsonl] [--ticker BHP.AX]

Arguments:
    input.csv       Path to the source CSV file.
    output.jsonl    Path for the output JSONL file (optional).
                    Defaults to the input filename with .jsonl extension.
    --ticker VALUE  Add a 'ticker' field with VALUE to every record (optional).

Examples:
    python3 csv_to_jsonl.py data.csv
    python3 csv_to_jsonl.py data.csv out.jsonl
    python3 csv_to_jsonl.py data.csv --ticker BHP.AX
    python3 csv_to_jsonl.py data.csv out.jsonl --ticker CBA.AX
"""

import csv
import json
import sys
from pathlib import Path


def infer_value(value: str):
    """Try to parse a string as int, then float; fall back to string."""
    stripped = value.strip()
    if stripped == "":
        return None
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return stripped


def convert(input_path: str, output_path: str | None = None, ticker: str | None = None) -> int:
    src = Path(input_path)
    if not src.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 1

    dst = Path(output_path) if output_path else src.with_suffix(".jsonl")

    with src.open(newline="", encoding="utf-8-sig") as f_in, \
         dst.open("w", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        count = 0
        for row in reader:
            record = {k.strip(): infer_value(v) for k, v in row.items() if k is not None}
            if ticker is not None:
                record["ticker"] = ticker
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Converted {count} rows → {dst}")
    return 0


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    input_path = args[0]
    output_path = None
    ticker = None

    i = 1
    while i < len(args):
        if args[i] == "--ticker" and i + 1 < len(args):
            ticker = args[i + 1]
            i += 2
        elif not args[i].startswith("--") and output_path is None:
            output_path = args[i]
            i += 1
        else:
            print(f"Unknown argument: {args[i]}", file=sys.stderr)
            sys.exit(1)

    sys.exit(convert(input_path, output_path, ticker))


if __name__ == "__main__":
    main()
