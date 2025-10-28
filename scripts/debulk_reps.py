#!/usr/bin/env python3
"""
Debulk large TSVs by selecting specific columns, with metadata-based file filtering.

- Reads TSVs from INPUT_DIR.
- Filters files based on metadata in METADATA_FILE and specified filter rules.
- Skips files whose debulked output already exists (resume-safe).
- Writes processed TSVs to OUTPUT_DIR.
- Streams in chunks to avoid loading whole files.
- Parallelised across WORKERS processes.

Dependencies: pandas, tqdm, PyYAML
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
from tqdm import tqdm
import yaml

# ───────────────────────── CONFIG ─────────────────────────
CONFIG_PATH = Path("yaml/config.yaml")  # specify config YAML path
# ──────────────────────── /CONFIG ────────────────────────


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def already_processed(out_path: Path) -> bool:
    try:
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def filter_files_by_metadata(
    metadata_path: Path, input_dir: Path, filter_columns: List[str], filter_values: Dict[str, List[Any]]
) -> List[Path]:
    """Return list of repertoire files whose metadata rows match the filter criteria."""
    meta = pd.read_csv(metadata_path, sep="\t", dtype=str)
    meta["sample_name"] = meta["sample_name"].astype(str) + ".tsv"

    for col, vals in filter_values.items():
        if col not in meta.columns:
            raise ValueError(f"Column '{col}' not in metadata file.")
        meta = meta[meta[col].isin(vals)]

    matched_filenames = set(meta["sample_name"].astype(str).tolist())
    all_files = [p for p in input_dir.iterdir() if p.is_file()]
    return [p for p in all_files if p.name in matched_filenames]


def process_one_file(
    in_path: Path,
    out_path: Path,
    selected_cols: List[str],
    sep: str,
    chunksize: int,
    encoding: str,
    on_bad_lines: str,
) -> Tuple[str, bool, Optional[str]]:
    try:
        header_df = pd.read_csv(in_path, sep=sep, nrows=0, encoding=encoding, on_bad_lines=on_bad_lines, dtype=str)
        available_cols = list(header_df.columns)
        usecols = [c for c in selected_cols if c in available_cols]

        if not usecols:
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
            pd.DataFrame(columns=usecols).to_csv(out_path, sep=sep, index=False, encoding=encoding)

        return (in_path.name, True, f"wrote cols: {usecols}{missing_msg}")

    except Exception as e:
        try:
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass
        return (in_path.name, False, str(e))


def main():
    cfg = load_config(CONFIG_PATH)

    INPUT_DIR = Path(cfg["input_dir"]).resolve()
    OUTPUT_DIR = Path(cfg["output_dir"]).resolve()
    METADATA_FILE = Path(cfg["metadata_file"]).resolve()
    SELECTED_COLS = cfg["selected_cols"]
    SEP = cfg.get("sep", "\t")
    CHUNKSIZE = cfg.get("chunksize", 250_000)
    ENCODING = cfg.get("encoding", "utf-8")
    ON_BAD_LINES = cfg.get("on_bad_lines", "skip")
    WORKERS = cfg.get("workers", 4)

    FILTER_COLUMNS = cfg.get("filter_columns", [])
    FILTER_VALUES = cfg.get("filter_values", {})

    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Filter files based on metadata
    matching = filter_files_by_metadata(METADATA_FILE, INPUT_DIR, FILTER_COLUMNS, FILTER_VALUES)

    if not matching:
        print("No files match metadata criteria.")
        return

    tasks = []
    skipped = 0
    for in_path in matching:
        out_path = OUTPUT_DIR / in_path.name
        if already_processed(out_path):
            skipped += 1
            continue
        tasks.append((in_path, out_path))

    print(f"Filtered {len(matching)} files by metadata.")
    print(f"Resume check → Skipped (existing): {skipped}, To process: {len(tasks)}.\n")

    if not tasks:
        print("Nothing to do. All matching files already processed.")
        return

    results_ok = 0
    results_err = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = [
            ex.submit(process_one_file, in_path, out_path, SELECTED_COLS, SEP, CHUNKSIZE, ENCODING, ON_BAD_LINES)
            for (in_path, out_path) in tasks
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing files", unit="file"):
            fname, ok, msg = fut.result()
            if ok:
                print(f"✓ {fname}: {msg}")
                results_ok += 1
            else:
                print(f"✖ {fname}: {msg}")
                results_err += 1

    print(f"\nDone. Succeeded: {results_ok}, Failed: {results_err}, Skipped pre-existing: {skipped}")


if __name__ == "__main__":
    main()
