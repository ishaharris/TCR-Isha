#!/usr/bin/env python3
"""
Debulk large TSVs by selecting specific columns.

- Reads TSVs from INPUT_DIR (external disk).
- Skips files whose debulked output already exists (resume-safe).
- Writes processed TSVs to OUTPUT_DIR (internal disk).
- Streams with chunks (doesn't load entire files into memory).
- Parallelised across 4 processes (files run concurrently).

Dependencies: pandas, tqdm
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple
import pandas as pd
from tqdm import tqdm

# ───────────────────────── CONFIG ─────────────────────────
INPUT_DIR   = Path("/Volumes/IshaVerbat/Isha/TCR/All_Emerson_Cohort01")
OUTPUT_DIR  = Path("/Users/ishaharris/Projects/TCR/TCR-Isha/data/All_Debulked_Emerson")
FILENAME_PREFIX = "P"  # only process files starting with this

SELECTED_COLS = [
    "rearrangement", "amino_acid", "seq_reads", "frequency",
    "productive_frequency", "v_gene","v_allele", "d_gene","d_allele",
    "j_gene", "j_allele"
]

# TSV parsing / writing options
SEP         = "\t"
CHUNKSIZE   = 250_000      # tune for your machine/files
ENCODING    = "utf-8"
ON_BAD_LINES= "skip"       # or "warn" (pandas >= 2.0)
WORKERS     = 4            # number of parallel processes
# ──────────────────────── /CONFIG ────────────────────────


def already_processed(out_path: Path) -> bool:
    """Return True if an output file exists and is non-empty (simple resume-safe check)."""
    try:
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def process_one_file(
    in_path: Path,
    out_path: Path,
    selected_cols: List[str],
    sep: str,
    chunksize: int,
    encoding: str,
    on_bad_lines: str,
) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single TSV: select columns and write streamed TSV to out_path.
    Returns (filename, success, message_or_error).
    """
    try:
        # Peek header to decide which cols are available
        header_df = pd.read_csv(
            in_path, sep=sep, nrows=0, encoding=encoding, on_bad_lines=on_bad_lines, dtype=str
        )
        available_cols = list(header_df.columns)
        usecols = [c for c in selected_cols if c in available_cols]

        if not usecols:
            # Mark as processed by writing an empty header-only file (so resume will skip next time)
            pd.DataFrame(columns=[]).to_csv(out_path, sep=sep, index=False, encoding=encoding)
            return (in_path.name, True, "no requested columns present; wrote empty header")

        missing = [c for c in selected_cols if c not in available_cols]
        missing_msg = f" (missing omitted: {missing})" if missing else ""

        wrote_any = False
        for chunk in pd.read_csv(
            in_path,
            sep=sep,
            usecols=usecols,
            chunksize=chunksize,
            encoding=encoding,
            on_bad_lines=on_bad_lines,
            dtype=str,
        ):
            mode = "w" if not wrote_any else "a"
            header = not wrote_any
            chunk.to_csv(out_path, sep=sep, index=False, mode=mode, header=header, encoding=encoding)
            wrote_any = True

        if not wrote_any:
            # Input had only header/no rows; still create an empty file with header
            pd.DataFrame(columns=usecols).to_csv(out_path, sep=sep, index=False, encoding=encoding)

        return (in_path.name, True, f"wrote cols: {usecols}{missing_msg}")

    except Exception as e:
        # Remove partial output so resume logic isn't confused later
        try:
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass
        return (in_path.name, False, str(e))


def main():
    input_dir = INPUT_DIR.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Gather files
    all_files = [p for p in input_dir.iterdir() if p.is_file()]
    matching  = sorted([p for p in all_files if p.name.startswith(FILENAME_PREFIX)])

    if not matching:
        print(f"No files starting with '{FILENAME_PREFIX}' found in {input_dir}")
        return

    # Build worklist with resume-safe skip
    tasks = []
    skipped = 0
    for in_path in matching:
        out_path = OUTPUT_DIR / in_path.name
        if already_processed(out_path):
            skipped += 1
            continue
        tasks.append((in_path, out_path))

    print(f"Found {len(all_files)} file(s) in folder; {len(matching)} match prefix '{FILENAME_PREFIX}'.")
    print(f"Resume check → Skipped (existing): {skipped}, To process: {len(tasks)}.\n")

    if not tasks:
        print("Nothing to do. All matching files already processed.")
        return

    # Submit tasks to a 4-process pool
    results_ok = 0
    results_err = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = [
            ex.submit(
                process_one_file,
                in_path,
                out_path,
                SELECTED_COLS,
                SEP,
                CHUNKSIZE,
                ENCODING,
                ON_BAD_LINES,
            )
            for (in_path, out_path) in tasks
        ]

        # Overall progress bar (one tick per completed file)
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing files", unit="file"):
            fname, ok, msg = fut.result()
            if ok:
                if msg:
                    print(f"✓ {fname}: {msg}")
                else:
                    print(f"✓ {fname}")
                results_ok += 1
            else:
                print(f"✖ {fname}: {msg}")
                results_err += 1

    print(f"\nDone. Succeeded: {results_ok}, Failed: {results_err}, Skipped pre-existing: {skipped}")


if __name__ == "__main__":
    main()
