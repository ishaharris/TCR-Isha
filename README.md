# TCR-Isha

TCR-Isha is a notebook-first exploratory research repository focused on T-cell receptor repertoire analysis, CMV-associated sequence discovery, similarity-based matching, and related classification and visualisation ideas.

The repo is intended as a workspace for trying analyses, testing hypotheses, and iterating on research questions rather than as a polished software package.

Most of the work happens in Jupyter notebooks, with a small set of Python scripts for preprocessing and batch sequence-comparison tasks.

## What it contains

- `notebooks/exploratory_data_analysis.ipynb` and `notebooks/repertoire_processing.ipynb` for initial repertoire exploration, cleaning, and summary work.
- `notebooks/immudex_clean.ipynb`, `notebooks/matches_clean.ipynb`, `notebooks/mair_repertoires.ipynb`, and `notebooks/new_reps.ipynb` for harmonising input repertoires and reference tables.
- `notebooks/Heatmaps.ipynb`, `notebooks/Heatmaps_cmv.ipynb`, `notebooks/Heatmaps_idh.ipynb`, `notebooks/CLEAN_HEATMAPS.ipynb`, and `notebooks/top10000_heatmaps.ipynb` for heatmap generation and comparison experiments.
- `notebooks/levenshtein_clean.ipynb`, `notebooks/fuzzy.ipynb`, and `notebooks/tcrdist_rough.ipynb` for sequence-similarity analyses using Levenshtein-style and TCRdist-style approaches.
- `notebooks/pgen.ipynb` and `notebooks/olga.ipynb` for generation-probability and OLGA-related experiments.
- `notebooks/classifier.ipynb` for early classification experiments on repertoire-derived features.
- `notebooks/vdjdb.ipynb`, `notebooks/covid_highconfs.ipynb`, `notebooks/idh1_exploratory.ipynb`, `notebooks/special_seqs.ipynb`, and `notebooks/AGE.ipynb` for antigen-, cohort-, or question-specific exploratory analyses.
- `scripts/levenshtein_frequency.py`, `scripts/updated_levenshtein_frequency.py`, `scripts/levenshtein_distances_with_tcrlabels.py`, and related `one_to_many` scripts for batch matching, frequency counting, and distance calculations.

## Notes

- Some notebooks and scripts use hard-coded local paths and assumptions about available input files.
- The repository includes rough variants, intermediate outputs, and analysis artifacts that reflect an active research workflow rather than a curated release.
- The code is best read as exploratory analysis code, not a stable or reusable library.
- Expect notebooks, scripts, and outputs to evolve as the research questions change.
