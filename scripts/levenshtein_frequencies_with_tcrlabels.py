import os
import random
import pickle  # only if you still want to checkpoint freq_df
from functools import partial
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein as L

# -------------------------------------------------------------------------
# Parameters (example values; replace as needed)
# -------------------------------------------------------------------------

hla = 'A*02'
hla_name = hla.replace('*', '')

donor = '5'

input_dir = f'/Volumes/IshaVerbat/Isha/TCR/Filtered_HLA-{hla_name}'
highconf_dir = '/Users/ishaharris/Projects/TCR/TCR-Isha/data/IDH/highconf'
highconf_file_name = f'donor{donor}_IDH_highconf.tsv'
metadata_file = "/Users/ishaharris/Projects/TCR/TCR-Isha/data/Repertoires/Cohort01_whole_metadata.tsv"
output_file = f'/Users/ishaharris/Projects/TCR/TCR-Isha/data/IDH/Levenshtein/donor_{donor}-{hla_name}.csv'


max_distance = 4
n_workers = 3
sample_n = 292
random_seed = 42
sep = '\t'
seq_col = 'cdr3_b_aa'
freq_col = 'productive_frequency'



# -------------------------------------------------------------------------
# 1) Load metadata & select files
# -------------------------------------------------------------------------
metadata = pd.read_csv(metadata_file, sep=sep)
metadata = metadata[
    metadata['sample_tags'].str.contains(f'HLA-{hla}', case=False, regex=False)
]
metadata = metadata[
    metadata['sample_tags'].str.contains(r'\bcytomegalovirus|CMV\b', case=False, na=False)
].reset_index(drop=True)

file_names = [f"{name}.slim.tsv" for name in metadata['sample_name']]

# -------------------------------------------------------------------------
# 2) Load high-confidence sequences & bucket by length
# -------------------------------------------------------------------------
highconf = pd.read_csv(os.path.join(highconf_dir, highconf_file_name), sep=sep)

top_n = highconf.shape[0]

# order highconf by magnitude of 'count' column
count_label = 'frequency_t2'
highconf = highconf.sort_values(by=count_label, ascending=False)
highconf_top = highconf.head(top_n)
# reset index
highconf_top = highconf_top.reset_index(drop=True)

aa_colname = 'amino_acid'
hc_seqs = highconf_top.loc[:,aa_colname].tolist()

hc_buckets: dict[int, list[str]] = defaultdict(list)
for seq in hc_seqs:
    hc_buckets[len(seq)].append(seq)



# -------------------------------------------------------------------------
# 3) Per-file worker: compute per-distance frequency sums
# -------------------------------------------------------------------------
def process_patient(
    fname: str,
    input_dir: str,
    sep: str,
    seq_col: str,
    freq_col: str,
    max_distance: int,
    hc_buckets: dict[int, list[str]],
) -> dict[str, float]:
    path = os.path.join(input_dir, fname)
    rep = pd.read_csv(path, sep=sep, usecols=[seq_col, freq_col]).dropna()
    seqs = rep[seq_col].to_numpy()
    freqs = rep[freq_col].to_numpy()

    total_hc = sum(len(v) for v in hc_buckets.values())
    print(f"{fname}: Searching {len(seqs)} patient sequences against {total_hc} high-confidence sequences")

    def min_distance_to_highconf(seq: str) -> int:
        best = max_distance + 1
        L_seq = len(seq)
        for tgt_len in range(L_seq - max_distance, L_seq + max_distance + 1):
            for hc in hc_buckets.get(tgt_len, []):
                d = L.distance(seq, hc, score_cutoff=best)
                if d is not None and d < best:
                    best = d
                    if best == 0:
                        return 0
        return best

    # compute min distances
    min_ds = np.array([min_distance_to_highconf(s) for s in seqs])

    # sum up frequencies for each exact distance ≤ max_distance
    mask = min_ds <= max_distance
    freq_sums = np.bincount(min_ds[mask], weights=freqs[mask], minlength=max_distance + 1)

    # build output row
    row = {'patient_id': fname}
    for d in range(max_distance + 1):
        row[f'dist_{d}'] = float(freq_sums[d])
    return row

# -------------------------------------------------------------------------
# 4) Orchestrator: run across all files, parallelized
# -------------------------------------------------------------------------
def compute_freqs(
    file_names: list[str],
    max_distance: int,
    input_dir: str,
    output_file: str,
    sep: str = '\t',
    seq_col: str = 'cdr3_b_aa',
    freq_col: str = 'productive_frequency',
    n_workers: int = None,
    sample_n: int = None,
    random_seed: int = None,
    process_all: bool = False
) -> pd.DataFrame:
    # load existing results if present
    if os.path.exists(output_file):
        freq_df = pd.read_csv(output_file)
        completed = set(freq_df['patient_id'])
    else:
        cols = ['patient_id'] + [f'dist_{d}' for d in range(max_distance + 1)]
        freq_df = pd.DataFrame(columns=cols)
        completed = set()

    # select pending files
    pending = [fn for fn in file_names if fn not in completed]
    if not process_all and sample_n is not None and sample_n < len(pending):
        random.seed(random_seed)
        pending = random.sample(pending, sample_n)
        print(f"Sampling {len(pending)} files from {len(file_names)} total files.")
    else:
        print(f"Processing all {len(pending)} files.")

    if not pending:
        print("No new files to process.")
        return freq_df

    worker = partial(
        process_patient,
        input_dir=input_dir,
        sep=sep,
        seq_col=seq_col,
        freq_col=freq_col,
        max_distance=max_distance,
        hc_buckets=hc_buckets,
    )

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for row in executor.map(worker, pending):
            freq_df = pd.concat([freq_df, pd.DataFrame([row])], ignore_index=True)
            # checkpoint after each file
            freq_df.to_csv(output_file, index=False)

    return freq_df

# -------------------------------------------------------------------------
# 5) Run if main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    df = compute_freqs(
        file_names=file_names,
        max_distance=max_distance,
        input_dir=input_dir,
        output_file=output_file,
        sep=sep,
        seq_col=seq_col,
        freq_col=freq_col,
        n_workers=n_workers,
        sample_n=sample_n,
        random_seed=random_seed,
    )
    print("Done. Frequencies saved to:", output_file)
