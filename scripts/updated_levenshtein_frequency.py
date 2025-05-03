import os
import pickle
import random
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein as L  # your C-backed implementation


# ------------------------------------------------------------------------------
# Load file names
# ------------------------------------------------------------------------------
# Set HLA of interest - change dynamically
hla = 'A*02'
hla_name = hla.replace('*','')

metadata = pd.read_csv("/Users/ishaharris/Projects/TCR/TCR-Isha/data/Repertoires/Cohort01_whole_metadata.tsv", sep="\t") 
metadata_annotat = metadata[metadata['sample_tags'].str.contains(f'HLA-{hla}',case=False, regex = False)]
metadata_annotat = metadata_annotat[metadata_annotat['sample_tags'].str.contains(r'\bcytomegalovirus|CMV\b', case=False, na=False)]
metadata_annotat = metadata_annotat.reset_index(drop=True)

file_names = [name + '.slim.tsv' for name in metadata_annotat['sample_name'].tolist()]

# ------------------------------------------------------------------------------
# Load high confidence sequences
# ------------------------------------------------------------------------------
# Load high conf data

highconf_dir = '/Users/ishaharris/Projects/TCR/TCR-Isha/data/vdjdb/'
highconf_file_name = 'vdjdb_one_epitope.tsv'
highconf = pd.read_csv(highconf_dir + highconf_file_name, sep='\t')

aa_colname = 'CDR3'

highconf_seqs = highconf.loc[:,aa_colname].tolist()

len(highconf_seqs)

# ------------------------------------------------------------------------------
# 1) Define your per-file worker at module scope
# ------------------------------------------------------------------------------
def process_patient(
    fname: str,
    input_dir: str,
    sep: str,
    seq_col: str,
    freq_col: str,
    max_distance: int,
    hc_buckets: dict[int, list[str]],
    contrib_top_n: int,
) -> tuple[str, dict[str, float], dict[str, float]]:
    """
    Reads one patient file, computes min-Levenshtein distances to high-conf sequences,
    sums up frequencies by distance, and picks the top contributors.
    """
    path = os.path.join(input_dir, fname)
    rep = pd.read_csv(path, sep=sep, usecols=[seq_col, freq_col]).dropna()
    seqs, freqs = rep[seq_col].to_numpy(), rep[freq_col].to_numpy()

    def min_distance_to_highconf(seq: str) -> int:
        best = max_distance + 1
        L_seq = len(seq)
        for tgt_len in range(L_seq - max_distance, L_seq + max_distance + 1):
            for hc in hc_buckets.get(tgt_len, []):
                # rapidfuzz’s C-level .distance() with cutoff
                d = L.distance(seq, hc, score_cutoff=best)
                if d is not None and d < best:
                    best = d
                    if best == 0:
                        return 0
        return best

    min_ds_list = []
    for s in tqdm(seqs,
              desc=f"Processing {fname}",
              unit="seq",
              total=len(seqs),
              miniters=1000):          # update display at least every 1000 iters
        min_ds_list.append(min_distance_to_highconf(s))
    min_ds = np.array(min_ds_list)

    # sum up frequencies for each exact distance
    mask = min_ds <= max_distance
    freq_sums = np.bincount(min_ds[mask], weights=freqs[mask],
                            minlength=max_distance + 1)

    # aggregate per-sequence frequencies across all close distances
    contrib_sums = defaultdict(float)
    for d, s, f in zip(min_ds, seqs, freqs):
        if d <= max_distance:
            contrib_sums[s] += f

    top_contrib = dict(
        sorted(contrib_sums.items(), key=lambda kv: kv[1], reverse=True)[:contrib_top_n]
    )

    row = {'patient_id': fname}
    for d in range(max_distance + 1):
        row[f'dist_{d}'] = float(freq_sums[d])

    return fname, row, top_contrib


def compute_freqs_and_contributors(
    highconf_seqs: list[str],
    file_names: list[str],
    max_distance: int,
    input_dir: str,
    output_file: str,
    contributors_output: str,
    sep: str = '\t',
    seq_col: str = 'cdr3_b_aa',
    freq_col: str = 'productive_frequency',
    contrib_top_n: int = 100,
    n_workers: int = None,
    sample_n: int = None,
    random_seed: int = None,
) -> tuple[pd.DataFrame, dict]:
    # bucket your high-confidence sequences by length
    hc_buckets: dict[int, list[str]] = defaultdict(list)
    for seq in highconf_seqs:
        hc_buckets[len(seq)].append(seq)

    # load or initialize the outputs
    if os.path.exists(output_file):
        freq_df = pd.read_csv(output_file)
        completed = set(freq_df['patient_id'])

        # 🔍 Debug print
        print("First 5 entries in completed:")
        print(list(completed)[:5])
        print("Type of first entry:", type(list(completed)[0]))


    else:
        cols = ['patient_id'] + [f'dist_{d}' for d in range(max_distance + 1)]
        freq_df = pd.DataFrame(columns=cols)
        completed = set()

    if os.path.exists(contributors_output):
        with open(contributors_output, 'rb') as f:
            contributors = pickle.load(f)
    else:
        contributors = {}

    pending = [fn for fn in file_names if fn not in completed]
    if sample_n is not None and sample_n < len(pending):
        if random_seed is not None:
            random.seed(random_seed)
        pending = random.sample(pending, sample_n)

    if not pending:
        return freq_df, contributors

    # bind all constant params into a single‐arg function
    from functools import partial
    worker = partial(
        process_patient,
        input_dir=input_dir,
        sep=sep,
        seq_col=seq_col,
        freq_col=freq_col,
        max_distance=max_distance,
        hc_buckets=hc_buckets,
        contrib_top_n=contrib_top_n,
    )

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(worker, fn): fn for fn in pending}
        for i, fut in enumerate(as_completed(futures), 1):
            fname = futures[fut]
            try:
                _, row, top_contrib = fut.result()
                print(f"[{i}/{len(pending)}] Processed {fname}")

                freq_df = pd.concat([freq_df, pd.DataFrame([row])],
                                     ignore_index=True)
                contributors[fname] = top_contrib

                # save intermediate results
                freq_df.to_csv(output_file, index=False)
                with open(contributors_output, 'wb') as f:
                    pickle.dump(contributors, f)

            except Exception as e:
                print(f"[{i}/{len(pending)}] ERROR {fname}: {e}")

    return freq_df, contributors

############### Prepare to run #################

directory = '/Users/ishaharris/Projects/TCR/TCR-Isha/data/Levenshtein/'

if __name__ == "__main__":
    # Example values — replace with real values or argparse
    hc_seqs = highconf_seqs  # list of strings
    fnames = file_names     # list of file names
    input_dir = '/Volumes/IshaVerbat/Isha/TCR/filtered_for_tcrdist'
    output_file = f'{directory}/output/VDJdb_freqs.csv'
    contributors_output = "contributors.pkl"
    max_distance = 3
    n_workers = 4
    random_seed = 42
    sample_n = 292  # or some integer

    compute_freqs_and_contributors(
        highconf_seqs=hc_seqs,
        file_names=fnames,
        max_distance=3,
        input_dir=input_dir,
        output_file=output_file,
        contributors_output=contributors_output,
        n_workers=n_workers,
        random_seed=random_seed,
        sample_n=sample_n
    )

