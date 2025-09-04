import os
import argparse
import ast
import random
from functools import partial
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from rapidfuzz.distance import Levenshtein as L


# -------------------------------------------------------------------------
# 1) Argument parsing
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Compute per-distance frequency sums against high-confidence TCRs."
    )
    p.add_argument("--highconf", choices=["cmv","invitro","random_vdjdb","random_olga","vatcr"], default="cmv",
                   help="highconf: 'cmv', 'invitro', 'random_vdjdb', 'random_olga', 'vatcr' or 'mair'")
    p.add_argument("--rep", choices=["emerson","mair"], default="emerson",
                   help="repertoire choice: 'emerson' or 'mair'")
    p.add_argument("--hla", default="all",
                   help="HLA allele to stratify by (e.g. b35), or 'all' to process every repertoire")
    p.add_argument("--cmv_status", choices=["Positive","Negative"], default="all",
                   help="CMV status to stratify by (Positive, Negative, or all)")
    p.add_argument("--donor", default="all",
                   help="Donor ID (or 'all' to use All_IDH_highconf.tsv or skip subfolders)")
    p.add_argument("--max_distance", type=int, default=4,
                   help="Maximum Levenshtein distance to consider")
    p.add_argument("--n_workers", type=int, default=4,
                   help="Number of parallel workers")
    p.add_argument("--sample_n", type=int, default=None,
                   help="Randomly sample this many repertoires if given")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--sep", default="\t",
                   help="Column separator for CSV/TSV files")
                   
    return p.parse_args()


# -------------------------------------------------------------------------
# 2) Worker function: compare each repertoire to high-conf seqs
# -------------------------------------------------------------------------
def process_patient(
    fname: str,
    input_dir: str,
    sep: str,
    seq_col: str,
    freq_col: str,
    max_distance: int,
    hc_buckets: dict[int, list[str]],
) -> tuple[str, dict[int, dict[str, float]]]:
    
    path = os.path.join(input_dir, fname)
    print(f"🔍 About to read: {path}")        # ← log exactly which file
    try:
        rep = pd.read_csv(
            path,
            sep=sep,
            usecols=[seq_col, freq_col],
            encoding='latin-1',    # or try utf-8 first then latin-1
        ).dropna()
    except Exception as e:
        print(f"❌ Failed on {path}: {e}")
        raise

    seqs = rep[seq_col].to_numpy()
    freqs = rep[freq_col].to_numpy()

    hc_by_dist = {d: defaultdict(float) for d in range(max_distance + 1)}
    for seq, w in zip(seqs, freqs):
        L_seq = len(seq)
        for tgt_len in range(L_seq - max_distance, L_seq + max_distance + 1):
            for hc in hc_buckets.get(tgt_len, []):
                d = L.distance(seq, hc, score_cutoff=max_distance)
                if d is not None and d <= max_distance:
                    hc_by_dist[d][hc] += w

    return fname, hc_by_dist


# -------------------------------------------------------------------------
# 3) Orchestrator: parallelize, save per-distance tall CSVs, then pivot to matrices
# -------------------------------------------------------------------------
def compute_freqs(
    file_names: list[str],
    max_distance: int,
    input_dir: str,
    output_file: str,
    sep: str,
    seq_col: str,
    freq_col: str,
    n_workers: int,
    sample_n: int,
    random_seed: int,
    hc_seqs: list[str],
    hc_buckets: dict[int, list[str]],
):
    # prepare output filenames
    out_files = {d: output_file.replace('.csv', f'_dist_{d}.csv')
                 for d in range(max_distance + 1)}

    # find completed patients
    completed = {}
    for d, fn in out_files.items():
        if os.path.exists(fn):
            df_done = pd.read_csv(fn, usecols=['patient_id'])
            completed[d] = set(df_done['patient_id'])
        else:
            completed[d] = set()

    pending = [fn for fn in file_names
               if not all(fn in completed[d] for d in range(max_distance + 1))]

    if sample_n and sample_n < len(pending):
        random.seed(random_seed)
        pending = random.sample(pending, sample_n)
        print(f"Sampling {len(pending)} of {len(file_names)} total files.")

    print(f"Pending files: {len(pending)}.")

    worker = partial(
        process_patient,
        input_dir=input_dir,
        sep=sep,
        seq_col=seq_col,
        freq_col=freq_col,
        max_distance=max_distance,
        hc_buckets=hc_buckets,
    )

    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        for idx, (patient_id, hc_by_dist) in enumerate(exe.map(worker, pending), start=1):
            print(f"Processed {idx}/{len(pending)}: {patient_id}")
            for d, hc_dict in hc_by_dist.items():
                if patient_id in completed[d]:
                    continue
                records = [
                    {'patient_id': patient_id, 'hc_seq': hc, 'weight': hc_dict.get(hc, 0.0)}
                    for hc in hc_seqs
                ]
                df_d = pd.DataFrame(records)
                out_csv = out_files[d]
                header = not os.path.exists(out_csv)
                os.makedirs(os.path.dirname(out_csv), exist_ok=True)
                df_d.to_csv(out_csv, mode='a', index=False, header=header)

    print("All done! Per-distance tall CSVs saved.")

    # pivot to wide matrices
    for d, out_csv in out_files.items():
        df = pd.read_csv(out_csv)
        mat = df.pivot(index='patient_id', columns='hc_seq', values='weight').fillna(0)
        matrix_file = out_csv.replace('.csv', '_matrix.csv')
        mat.to_csv(matrix_file)
        print(f"→ matrix for distance {d} saved to {matrix_file}")


# -------------------------------------------------------------------------
# 4) Main: glue everything together
# -------------------------------------------------------------------------
def main():

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

    args = parse_args()

    # single mode name
    mode_name = f"{args.highconf}_{args.rep}"

    # donor subdir logic 
    donor = args.donor.lower()
    donor_subdir = '' if donor == 'all' else f'donor{args.donor}'

    # set highconf parameters
    if args.highconf == 'cmv':
        base_output_dir = os.path.join(PROJECT_ROOT, 'data/heatmap_output', mode_name, args.cmv_status)
        highconf_dir    = "/Users/ishaharris/Projects/TCR/TCR-Isha/data/High_Confidence_CMV_TCR"
        highconf_file   = 'WITHOUT_EBV_highconf.tsv'
        count_label     = 'count'
        # input_dir       = '/Volumes/IshaVerbat/Isha/TCR/All_Emerson_Cohort01'
        # seq_col         = 'cdr3_amino_acid'
        # freq_col        = 'productive_frequency'

    elif args.highconf == 'invitro':
        base_output_dir = os.path.join(PROJECT_ROOT,'data/heatmap_output', mode_name, args.cmv_status, donor_subdir)
        highconf_dir    =  "/Users/ishaharris/Projects/TCR/TCR-Isha/data/IDH_Heatmaps/highconf"
        highconf_file   = 'All_IDH_highconf.tsv' if donor == 'all' else f'donor{args.donor}_IDH_highconf.tsv'
        count_label     = 'freq_t2'
        # input_dir       = '/Volumes/IshaVerbat/Isha/TCR/All_Emerson_Cohort01'
        # seq_col         = 'cdr3_amino_acid'
        # freq_col        = 'productive_frequency'

    # elif args.mode == 'mair':
    #     base_output_dir = os.path.join(PROJECT_ROOT, 'data/heatmap_output', 'mair', donor_subdir)
    #     highconf_dir    = "/Users/ishaharris/Projects/TCR/TCR-Isha/data/IDH_Heatmaps/highconf"
    #     highconf_file   = 'All_IDH_highconf.tsv' if donor == 'all' else f'donor{args.donor}_IDH_highconf.tsv'
    #     count_label     = 'freq_t2'
    #     input_dir       = '/Volumes/IshaVerbat/Isha/TCR/mair_glioma_batch_1'
    #     seq_col         = 'aaSeqCDR3'
    #     freq_col        = 'readFraction'
    
    elif args.highconf == 'random_vdjdb':
        base_output_dir = os.path.join(PROJECT_ROOT, 'data/heatmap_output', mode_name)
        highconf_dir    = "/Users/ishaharris/Projects/TCR/TCR-Isha/data/vdjdb"
        highconf_file   = 'vdjdb_sample_100_epitopes.tsv'
        count_label     = 'count'
        # input_dir       = '/Volumes/IshaVerbat/Isha/TCR/All_Emerson_Cohort01'
        # seq_col         = 'cdr3_amino_acid'
        # freq_col        = 'productive_frequency'

    elif args.highconf == 'random_olga':
        base_output_dir = os.path.join(PROJECT_ROOT, 'data/heatmap_output', mode_name)
        highconf_dir    = "/Users/ishaharris/Projects/TCR/TCR-Isha/data/highconf"
        highconf_file   = 'highconf_olga_random.tsv'
        count_label     = 'count'
        # input_dir       = '/Volumes/IshaVerbat/Isha/TCR/All_Emerson_Cohort01'
        # seq_col         = 'cdr3_amino_acid'
        # freq_col        = 'productive_frequency'
        

    elif args.highconf == 'vatcr':
        base_output_dir = os.path.join(PROJECT_ROOT, 'data/heatmap_output', mode_name)
        highconf_dir    = "/Users/ishaharris/Projects/TCR/TCR-Isha/data/vatcr_heatmaps"
        highconf_file   = 'highconf_vatcr.tsv'
        count_label     = 'freq_t2'
        # input_dir       = '/Volumes/IshaVerbat/Isha/TCR/collapsed_mair_glioma_batch_1'
        # seq_col         = 'aaSeqCDR3'
        # freq_col        = 'readFraction'

    

    if args.rep == 'mair':
        input_dir       = '/Volumes/IshaVerbat/Isha/TCR/collapsed_mair_glioma_batch_1'
        seq_col         = 'aaSeqCDR3'
        freq_col        = 'readFraction'

    elif args.rep == 'emerson':
        input_dir       = '/Volumes/IshaVerbat/Isha/TCR/All_Emerson_Cohort01'
        seq_col         = 'cdr3_amino_acid'
        freq_col        = 'productive_frequency'




    os.makedirs(base_output_dir, exist_ok=True)
    output_file = os.path.join(base_output_dir, f"{args.hla if args.hla!='all' else 'all'}.csv")

    # build file list
    if args.hla.lower() == 'all':
        if args.rep == 'mair':
            metadata = pd.read_csv("/Users/ishaharris/Projects/TCR/TCR-Isha/data/Repertoires/mair/mair_glioma_batch_1_metadata.tsv", sep="\t")
            file_names = metadata[metadata['repID'] == 'R1']['subjectID'].apply(
                lambda x: f"collapsed_mair_{x}.tsv"
            ).unique().tolist()

        else:
            # Load metadata and apply CMV filter for Emerson files starting with 'P'
            metadata = pd.read_csv("/Users/ishaharris/Projects/TCR/TCR-Isha/data/Repertoires/Cohort01_whole_metadata.tsv", sep="\t")
            filtered = metadata[
                metadata['sample_name'].str.startswith('P') &
                metadata['sample_tags'].str.contains(r'Cytomegalovirus\s*[+-]', case=False, na=False) 
            ]
            file_names = [name + '.tsv' for name in filtered['sample_name']]


    else:
        key = f"{args.hla}_cmv_{args.cmv_status}"
        pl = pd.read_csv('data/Repertoires/CMV_HLA_groups.tsv', sep=args.sep).iloc[0].to_dict()
        file_names = ast.literal_eval(pl[key])

    # ----- Check file names --------

    # Absolute path for each file
    full_paths = [os.path.join(input_dir, fname) for fname in file_names]
    # Filter out files that don’t exist
    valid_file_names = [
        fname for fname, full_path in zip(file_names, full_paths)
        if os.path.exists(full_path)
    ]
    # Report how many were removed
    n_removed = len(file_names) - len(valid_file_names)
    if n_removed > 0:
        print(f"⚠️ Skipped {n_removed} non-existent files.")
    file_names = valid_file_names  # use this filtered list going forward

    # ----------- load highconf buckets ---------
    if highconf_file:
        highconf = pd.read_csv(os.path.join(highconf_dir, highconf_file), sep=args.sep)
        aa_col = highconf.columns[highconf.columns.str.contains('amino|aa', case = False)][0]
        highconf = highconf.groupby(aa_col, as_index = False)[count_label].sum()
        highconf = highconf.sort_values(by=count_label, ascending=False).reset_index(drop=True)
        hc_seqs = highconf[aa_col].tolist()

        hc_buckets = defaultdict(list)
        for seq in hc_seqs:
            hc_buckets[len(seq)].append(seq)

    else:
        hc_seqs = []
        hc_buckets = {}

    # run
    compute_freqs(
        file_names=file_names,
        max_distance=args.max_distance,
        input_dir=input_dir,
        output_file=output_file,
        sep=args.sep,
        seq_col=seq_col,
        freq_col=freq_col,
        n_workers=args.n_workers,
        sample_n=args.sample_n,
        random_seed=args.random_seed,
        hc_seqs=hc_seqs,
        hc_buckets=hc_buckets,
    )

if __name__ == '__main__':
    
    main()
