# Stratified subject-grouped cross-validation overstates person-level generalization in EEG-based drowsiness detection

Code and result tables for the paper:

> **Stratified Subject-Grouped Cross-Validation Overstates Person-Level Generalization in EEG-Based Drowsiness Detection: A Multi-Objective, Explainable Model-Selection Study**
> Gyanendra Chaubey and Sweta Kaman, School of AI and Data Science, Indian Institute of Technology Jodhpur.

## Summary

Subject-grouped cross-validation keeps any one person out of both train and test. When classes are imbalanced, practitioners usually use a label-stratified version such as scikit-learn's `StratifiedGroupKFold`. This repository shows that when labels are clustered by subject, stratification is itself a large source of optimism. It picks which subjects share a fold by looking at their labels, so train and test class priors match in a way a new user never will.

On the DROZY database (14 subjects, 720 thirty-second EEG/EOG/EMG/ECG epochs, psychomotor-vigilance lapse labels), a three-objective NSGA-II search selected 20 models. The objectives were MCC, SHAP explanation faithfulness and lapse recall. The table shows those 20 models under each evaluation protocol:

| Protocol | MCC (mean ± s.d.) | Noise-feature null |
|---|---|---|
| P0 Search score (selected on these folds) | 0.105 ± 0.024 | – |
| P1 Stratified subject-grouped 3-fold, new seeds | 0.094 ± 0.015 | −0.008 |
| P2 Unstratified subject-grouped 3-fold | 0.025 ± 0.012 | −0.002 |
| P3 Leave-one-subject-out (per-subject mean) | −0.013 ± 0.034 | −0.004 |
| P4 LOSO + CORAL alignment | 0.029 ± 0.050 | – |
| P5 LOSO + EM class-prior adjustment | −0.022 ± 0.032 | – |
| P6 Rested → sleep-deprived transfer (seen subjects) | 0.179 ± 0.035 | −0.008 |

## Repository layout

```
src/
  config.py              paths (data, results, figures), overridable by env vars
  features.py            epoching, PVT labels, band-power / ratio / ECG features,
                         per-subject baseline normalization
  xai_core.py            search space, subject-grouped CV, SHAP faithfulness (Phi),
                         lapse recall, sensor-noise stress test
  navigation.py          Pareto front, navigation rules, Monte Carlo hypervolume
  run_search.py          NSGA-II / random / TPE search (100 trials x 5 seeds)
  aggregate_results.py   per-seed summary, navigation tables, hypervolume
  run_transfer_eval.py   P0, P3, P4 (CORAL), P6 and the noise stress test
  diagnose_gap.py        P1 (fresh stratified folds), within-subject MCC, P3' (pooled LOSO)
  null_baseline.py       noise-feature nulls for P1, P3, P3', P6
  label_shift_eval.py    P2 (unstratified folds), P5 (EM prior adjustment)
  paper_stats.py         P2 null, prior divergence, bootstrap CI, power, Wilcoxon tests
  mi_diagnostic.py       mutual-information check of baseline normalization
  make_eeg_figure.py     Figure 1: spectra, session effect, class effect, mutual information
  make_figures.py        Figures 2-4: Pareto fronts, hypervolume, protocol decomposition
results/                 CSV outputs of every step, as used in the paper
```

## Setup

Python 3.11 was used.

```bash
git clone https://github.com/GyanendraChaubey/DROZY.git
cd DROZY
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The DROZY database is **not** included in this repository. It is distributed by the University of Liège under a licence agreement; see http://www.drozy.ulg.ac.be for access and contact details.

Only the `psg/`, `pvt-rt/` and `timestamps/` folders and `KSS.txt` are needed. Place them as follows:

```
data/DROZY/
  psg/          1-1.edf, 1-2.edf, ...
  pvt-rt/
  timestamps/
  KSS.txt
```

To keep the data elsewhere, set `export DROZY_DATA_DIR=/path/to/DROZY`.

## Reproducing the results

The `results/` folder already contains every CSV used in the paper. You can regenerate the figures and statistics without re-running the search, or re-run everything from scratch.

**Figures and statistics only** (about 2 minutes). `make_figures.py` needs only `results/`; `paper_stats.py`, `mi_diagnostic.py` and `make_eeg_figure.py` also need the data:

```bash
python src/paper_stats.py      # decomposition table, TV, bootstrap CI, power, Wilcoxon
python src/mi_diagnostic.py    # mutual-information diagnostic
python src/make_eeg_figure.py  # Figure 1 (reads the EDF files)
python src/make_figures.py     # Figures 2-4, writes figures/*.pdf
```

**Full pipeline** (run in this order):

```bash
python src/run_search.py            # ~65 min on 2 CPU cores; writes {nsga2,random,tpe}_seed*.csv
python src/aggregate_results.py     # Table 2, hypervolume (Fig. 3)
python src/run_transfer_eval.py     # P0, P3, P4, P6, noise stress test
python src/diagnose_gap.py          # P1, P3', within-subject MCC
python src/null_baseline.py         # noise-feature nulls (~3 min)
python src/label_shift_eval.py      # P2, P5
python src/paper_stats.py
python src/mi_diagnostic.py
python src/make_eeg_figure.py
python src/make_figures.py
```

Every step after the search reads the search CSVs and uses fixed seeds, so it reproduces the paper's numbers exactly. The search itself is seeded too, but its results can vary slightly across library versions and hardware. For exact reproduction of the downstream analyses, use the provided search CSVs.

### Notes

- `run_search.py` sets `population_size=10` for Optuna's `NSGAIISampler`. With the default of 50 and a small trial budget, no generation completes, and NSGA-II silently behaves exactly like random search.
- The search uses `StratifiedGroupKFold` (3 folds, group = subject), as a careful practitioner would. Showing how optimistic that estimate is is the point of the paper.
- Per-subject baseline normalization uses each subject's first five unlabelled epochs. Every person-level result therefore assumes a short unlabelled calibration recording per user.

## Citation

If you use this code, please cite the paper (citation details will be added on publication).

## Licence

Code: MIT (see `LICENSE`). The DROZY data are subject to their own licence and are not redistributed here.
