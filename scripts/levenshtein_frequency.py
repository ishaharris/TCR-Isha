#!/usr/bin/env python3
"""
Parallel processing of TCR files to compute frequency sums by Levenshtein distance
to a set of high-confidence sequences.
"""

import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein as L
import pickle
import sys

# ——————————————————————————————————————————————————————
# PARAMETERS (edit as needed)
max_distance = 4
output_file = '/Users/ishaharris/Projects/TCR/TCR-Isha/data/Processed/freqs_leven_withoutEBV.csv'
input_folder = '/Volumes/IshaVerbat/Isha/TCR/filtered_for_tcrdist'
pickle_dir = '/Users/ishaharris/Projects/TCR/TCR-Isha/data/Levenshtein distance/pickles'
# You must define these two lists before running:


# highconf_seqs = [...]  # list of your high-confidence sequences
with open(os.path.join(pickle_dir, f'highconf_seqs'), 'rb') as f:
    highconf_seqs = pickle.load(f)


# file_names    = [...]  # list of filenames (e.g. ['sample1.txt', 'sample2.txt', ...])
with open(os.path.join(pickle_dir, f'hla02_file_names'), 'rb') as f:
    file_names = pickle.load(f)

# Column names for the output CSV
col_names = ['patient_id'] + [f'dist_{d}' for d in range(max_distance + 1)]

# Build length‐buckets for high‐confidence sequences
hc_buckets = defaultdict(list)
for seq in highconf_seqs:
    hc_buckets[len(seq)].append(seq)

def min_distance_to_highconf(seq, max_d=max_distance, buckets=hc_buckets):
    """
    Return the smallest Levenshtein distance from `seq` to any highconf seq ≤ max_d;
    else return max_d+1.
    """
    L_seq = len(seq)
    best = max_d + 1

    for target_len in range(L_seq - max_d, L_seq + max_d + 1):
        for hc in buckets.get(target_len, []):
            d = L.distance(seq, hc, score_cutoff=best)
            if d is None:
                continue
            if d < best:
                best = d
                if best == 0:
                    return 0
    return best

def process_file(file_name):
    """
    Read one TCR file, compute freq sums by distance, and return a dict row.
    """
    path = os.path.join(input_folder, file_name)
    try:
        rep = (
            pd.read_csv(path, sep='\t',
                        usecols=['cdr3_b_aa', 'productive_frequency'])
            .dropna()
        )
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        return None

    seqs  = rep['cdr3_b_aa'].to_numpy()
    freqs = rep['productive_frequency'].to_numpy()

    # compute min distances
    min_ds = [min_distance_to_highconf(s) for s in seqs]

    # aggregate
    mask = np.array(min_ds) <= max_distance
    freq_sums = np.bincount(
        np.array(min_ds)[mask],
        weights=freqs[mask],
        minlength=max_distance + 1
    )

    row = {'patient_id': file_name}
    for d in range(max_distance + 1):
        row[f'dist_{d}'] = freq_sums[d]
    return row

def main():
    # Prepare or resume output
    if os.path.exists(output_file):
        freq_df  = pd.read_csv(output_file)
        completed = set(freq_df['patient_id'])
    else:
        freq_df  = pd.DataFrame(columns=col_names)
        completed = set()

    # Filter out already-done files
    remaining = [f for f in file_names if f not in completed]
    total = len(remaining)
    print(f"Processing {total} files with 2 parallel workers...")

    with ProcessPoolExecutor(max_workers=3) as executor:
        for idx, result in enumerate(executor.map(process_file, remaining), start=1):
            if result is None:
                continue
            freq_df = pd.concat([freq_df, pd.DataFrame([result])],
                                ignore_index=True)
            # save after each file
            freq_df.to_csv(output_file, index=False)
            print(f"[{idx}/{total}] Saved {result['patient_id']}")

if __name__ == '__main__':
    main()
