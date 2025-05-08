import os
import random
from functools import partial
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein as L

# -------------------------------------------------------------------------
# Parameters (example values; replace as needed)
# -------------------------------------------------------------------------

donor = '2'
hla = 'A*02'
hla_name = hla.replace('*', '')


input_dir = f'/Volumes/IshaVerbat/Isha/TCR/Filtered_HLA-{hla_name}'
highconf_dir = '/Users/ishaharris/Projects/TCR/TCR-Isha/data/High_Confidence_CMV_TCR'
highconf_file_name = f'WITHOUT_EBV_highconf.tsv'
metadata_file = "/Users/ishaharris/Projects/TCR/TCR-Isha/data/Repertoires/Cohort01_whole_metadata.tsv"
output_file = f'/Users/ishaharris/Projects/TCR/TCR-Isha/data/CMV_Heatmaps/withoutEBV_{hla_name}.csv'

max_distance = 4
n_workers = 4
sample_n = None
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
count_label = 'count'

highconf = highconf.sort_values(by=count_label, ascending=False).reset_index(drop=True)

aa_colname = 'amino_acid'

hc_seqs = highconf[aa_colname].tolist()

hc_buckets: dict[int, list[str]] = defaultdict(list)
for seq in hc_seqs:
    hc_buckets[len(seq)].append(seq)

# -------------------------------------------------------------------------
# 3) Per-file worker: compute per-distance frequency sums
# -------------------------------------------------------------------------
def make_inner_dict():
    return defaultdict(float)

def process_patient(
    fname: str,
    input_dir: str,
    sep: str,
    seq_col: str,
    freq_col: str,
    max_distance: int,
    hc_buckets: dict[int, list[str]],
) -> tuple[str, dict[int, dict[str, float]]]:
    
    """
    Return:
        (patient_id, {distance -> {hc_seq -> total_weight}})
    Every repertoire sequence is compared to *all* high-confidence sequences whose
    length is within ±max_distance.  If the distance ≤ max_distance, its frequency
    is added to that (hc_seq, distance) bucket.  A repertoire sequence may
    contribute to multiple hc_seqs.
    """
    
    path = os.path.join(input_dir, fname)
    rep = pd.read_csv(path, sep=sep, usecols=[seq_col, freq_col]).dropna()
    seqs = rep[seq_col].to_numpy()
    freqs = rep[freq_col].to_numpy()

    hc_by_dist = {d: defaultdict(float) for d in range(max_distance + 1)}

    for seq, w in zip(seqs, freqs):
        L_seq = len(seq)
        # loop over HC seqs whose length is within max dist
        for tgt_len in range(L_seq - max_distance, L_seq + max_distance + 1):
            for hc in hc_buckets.get(tgt_len, []):
                d = L.distance(seq, hc, score_cutoff = max_distance)
                if d is not None and d <= max_distance:
                    hc_by_dist[d][hc] += w

    return fname, hc_by_dist

# -------------------------------------------------------------------------
# 4) Orchestrator: run across all files, parallelized, with per-distance CSVs
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
) -> None:
    # determine which patients are already done (if CSV exists). need to check for every output file
    out_files = {
        d: output_file.replace('.csv', f'_dist_{d}.csv')
        for d in range(max_distance + 1)
    }
    completed_per_dist = {}
    for d, fn in out_files.items():
        if os.path.exists(fn):
            df_done = pd.read_csv(fn, usecols=['patient_id'])
            completed_per_dist[d] = set(df_done['patient_id'])
        else:
            completed_per_dist[d] = set()


    
    pending = [
        fn for fn in file_names
        if not all(fn in completed_per_dist[d] for d in range(max_distance + 1))
    ]
    if sample_n and sample_n < len(pending):
        random.seed(random_seed)
        pending = random.sample(pending, sample_n)
        print(f"Sampling {len(pending)} of {len(file_names)} total files.")

    total = len(pending)
    print(f"Pending files: {total}.")

    # prepare per-distance output filenames
    out_files = {
        d: output_file.replace('.csv', f'_dist_{d}.csv')
        for d in range(max_distance + 1)
    }

    worker = partial(
        process_patient,
        input_dir=input_dir,
        sep=sep,
        seq_col=seq_col,
        freq_col=freq_col,
        max_distance=max_distance,
        hc_buckets=hc_buckets,
    )

    # process in parallel, checkpoint per patient and per distance
    with ProcessPoolExecutor(max_workers=n_workers) as exe:

        for idx, (patient_id, hc_by_dist) in enumerate(exe.map(worker, pending), start=1):
            print(f"Processed {idx}/{total}: {patient_id}")

            # append each distance-specific tall CSV
            for d, hc_dict in hc_by_dist.items():
                # skip if already written for this distance
                if patient_id in completed_per_dist[d]:
                    continue

                # build one record per highconf tcr, defaulting to 0 if no match
                records = []
                for hc in hc_seqs:
                    records.append({
                        'patient_id': patient_id,
                        'hc_seq': hc,
                        'weight': hc_dict.get(hc, 0.0),
                    })

                df_d = pd.DataFrame(records)
                out_csv = out_files[d]
                header = not os.path.exists(out_csv)
                os.makedirs(os.path.dirname(out_csv), exist_ok=True)
                df_d.to_csv(out_csv, mode='a', index=False, header=header)

    print("All done! Per-distance tall CSVs saved.")

    # ---------------------------------------------------------------------
    # 5) Pivot each tall CSV into a wide patient×hc_seq matrix
    # ---------------------------------------------------------------------
    for d, out_csv in out_files.items():
        df = pd.read_csv(out_csv)
        mat = df.pivot(index='patient_id', columns='hc_seq', values='weight').fillna(0)
        matrix_file = out_csv.replace('.csv', '_matrix.csv')
        mat.to_csv(matrix_file)
        print(f"→ matrix for distance {d} saved to {matrix_file}")


# -------------------------------------------------------------------------
# 6) Run if main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    compute_freqs(
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
