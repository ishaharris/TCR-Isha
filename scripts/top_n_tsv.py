#!/usr/bin/env python3
"""
Filter top N rows by frequency from TSV files.

Usage:
    python top_n_tsv.py --input /path/to/input_dir --output /path/to/output_dir --pattern "*.tsv" --n_top 10000
"""

import argparse
from pathlib import Path
import pandas as pd


def process_files(input_dir: Path, output_dir: Path, pattern: str, n_top: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = []
    skipped = []

    for tsv_path in sorted(input_dir.glob(pattern)):
        try:
            df = pd.read_csv(tsv_path, sep="\t")

            if "amino_acid" in df.columns:
                df = df.dropna(subset=["amino_acid"])
            else:
                raise KeyError("Column 'amino_acid' not found in file")

            df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce").fillna(0)
            df_top = df.sort_values(by="frequency", ascending=False).head(n_top)

            out_path = output_dir / tsv_path.name
            df_top.to_csv(out_path, sep="\t", index=False)

            processed.append((tsv_path.name, len(df), len(df_top), out_path.name))
            print(f"✔ {tsv_path.name} → {out_path.name} (rows {len(df)} → {len(df_top)})")

            if len(df_top) == 0:
                print(f"⚠ WARNING: Output file {out_path.name} has 0 rows!")

        except Exception as e:
            skipped.append((tsv_path.name, str(e)))
            print(f"✖ Skipped {tsv_path.name}: {e}")

    print("\nSummary")
    print(f"Processed: {len(processed)} file(s)")
    for name, total, kept, outname in processed:
        print(f"  - {name}: kept {kept}/{total} → {outname}")
        if kept == 0:
            print(f"    ⚠ WARNING: {outname} has 0 rows!")
    if skipped:
        print(f"Skipped: {len(skipped)} file(s)")
        for name, err in skipped:
            print(f"  - {name}: {err}")


def main():
    parser = argparse.ArgumentParser(description="Filter top N rows by frequency from TSV files.")
    parser.add_argument("--input", required=True, type=Path, help="Input directory containing TSV files")
    parser.add_argument("--output", required=False, type=Path, help="Output directory")
    parser.add_argument("--pattern", default="*.tsv", help="File pattern to match (default: *.tsv)")
    parser.add_argument("--n_top", default=10000, type=int, help="Number of top rows to keep (default: 10000)")

    args = parser.parse_args()
    output_dir = args.output or (args.input / f"top{args.n_top}")

    process_files(args.input, output_dir, args.pattern, args.n_top)


if __name__ == "__main__":
    main()
