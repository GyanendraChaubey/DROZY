"""
Summary statistics reported in the paper's decomposition analysis
(Table "Evaluation-protocol decomposition" and the surrounding text).

Reads the CSVs produced by run_transfer_eval.py, diagnose_gap.py,
null_baseline.py and label_shift_eval.py, and additionally computes:
  * the noise-feature null for unstratified subject-grouped 3-fold CV (P2)
  * mean train-test class-prior total-variation distance per protocol
  * subject-level bootstrap 95% CI and minimum detectable effect for LOSO
  * descriptive Wilcoxon signed-rank tests across configurations

Usage:  python src/paper_stats.py      (about 1 minute)
"""
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from config import RESULTS_DIR
from features import build_dataset
from navigation import pareto_mask, build_navigation_table
from xai_core import subject_grouped_cv_splits
from label_shift_eval import unstratified_3fold
from run_transfer_eval import OBJ_COLS, N_SEEDS


def fmt(s):
    return f"{s.mean():+.3f} +/- {s.std():.3f}"


def selected_configs():
    cfgs = []
    for seed in range(N_SEEDS):
        df = pd.read_csv(RESULTS_DIR / f"nsga2_seed{seed}.csv")
        front = df[pareto_mask(df[list(OBJ_COLS)].values)].reset_index(drop=True)
        for nrow in build_navigation_table(front, OBJ_COLS):
            cfgs.append(df[df["trial"] == nrow["trial"]].iloc[0].to_dict())
    return cfgs


def prior_tv(y, groups):
    prior = lambda v: np.array([(v == c).mean() for c in range(3)])
    tv = lambda a, b: 0.5 * np.abs(prior(a) - prior(b)).sum()
    subjects = np.unique(groups)
    strat = [tv(y[tr], y[te]) for fs in range(100, 110)
             for tr, te in subject_grouped_cv_splits(y, groups, 3, fs)]
    unstrat = []
    for rep in range(10):
        perm = np.random.RandomState(rep).permutation(subjects)
        for part in np.array_split(perm, 3):
            te = np.isin(groups, part)
            unstrat.append(tv(y[~te], y[te]))
    loso = [tv(y[groups != s], y[groups == s]) for s in subjects]
    return np.mean(strat), np.mean(unstrat), np.mean(loso)


def main():
    X_df, y_s, meta = build_dataset()
    X, y = X_df.values, y_s.values
    groups = meta["subject"].values

    te = pd.read_csv(RESULTS_DIR / "transfer_eval.csv")
    gd = pd.read_csv(RESULTS_DIR / "gap_decomposition.csv")
    ls = pd.read_csv(RESULTS_DIR / "label_shift_eval.csv")
    nb = pd.read_csv(RESULTS_DIR / "null_baseline.csv")

    # Null for P2 (unstratified 3-fold), 3 noise seeds
    cfgs = selected_configs()
    unstrat_null = []
    for ns in range(3):
        Xn = np.random.RandomState(1000 + ns).normal(size=X.shape)
        for row in cfgs:
            unstrat_null.append(unstratified_3fold(row, Xn, y, groups, reps=5)["unstrat3fold_mcc"])
    unstrat_null = pd.Series(unstrat_null)

    tv1, tv2, tv3 = prior_tv(y, groups)

    print("Protocol                          MCC                 null")
    print(f"P0 search score                   {fmt(te['search_mcc'])}")
    print(f"P1 stratified 3-fold, new seeds   {fmt(gd['fresh3fold_mcc'])}   {fmt(nb['fresh3fold_mcc'])}")
    print(f"P2 unstratified 3-fold            {fmt(ls['unstrat3fold_mcc'])}   {fmt(unstrat_null)}")
    print(f"P3 LOSO per-subject               {fmt(te['loso_mcc_mean'])}   {fmt(nb['loso_per_subject_mcc'])}")
    print(f"P3' LOSO pooled predictions       {fmt(gd['loso_pooled_mcc'])}   {fmt(nb['loso_pooled_mcc'])}")
    print(f"P4 LOSO + CORAL                   {fmt(te['loso_coral_mcc_mean'])}")
    print(f"P5 LOSO + EM prior                {fmt(ls['loso_em_per_subject_mcc'])}")
    print(f"P6 sleep-deprivation transfer     {fmt(te['sdt_test3_mcc'])}   {fmt(nb['sdt_mcc'])}")
    print()
    print(f"Within-subject MCC: P1 {fmt(gd['fresh3fold_per_subject_mcc'])}, "
          f"P2 {fmt(ls['unstrat3fold_per_subject_mcc'])}")
    print(f"Train-test prior TV: P1 {tv1:.3f}, P2 {tv2:.3f}, P3 {tv3:.3f}")
    print()

    p0p1 = (gd["search_mcc"] > gd["fresh3fold_mcc"]).sum()
    p1p2 = (gd["fresh3fold_mcc"].values > ls["unstrat3fold_mcc"].values).sum()
    w12 = stats.wilcoxon(gd["fresh3fold_mcc"].values, ls["unstrat3fold_mcc"].values).pvalue
    coral_up = (te["loso_coral_mcc_mean"] > te["loso_mcc_mean"]).sum()
    wc = stats.wilcoxon(te["loso_coral_mcc_mean"], te["loso_mcc_mean"]).pvalue
    em_up = (ls["loso_em_per_subject_mcc"].values > te["loso_mcc_mean"].values).sum()
    print(f"P0 > P1 for {p0p1}/20 configs")
    print(f"P1 > P2 for {p1p2}/20 configs (Wilcoxon p = {w12:.2e}, descriptive)")
    print(f"CORAL > LOSO for {coral_up}/20 configs (Wilcoxon p = {wc:.4f}, descriptive)")
    print(f"EM > LOSO for {em_up}/20 configs")
    print(f"LOSO lapse recall: {te['loso_lapse_recall_mean'].mean():.3f} -> "
          f"CORAL {te['loso_coral_lapse_recall_mean'].mean():.3f}")
    print()

    ps = pd.read_csv(RESULTS_DIR / "loso_per_subject.csv").dropna(subset=["mcc"])
    subj = ps.groupby("subject")["mcc"].mean()
    rng = np.random.RandomState(0)
    boot = [rng.choice(subj.values, len(subj)).mean() for _ in range(10_000)]
    sd, n = subj.std(ddof=1), len(subj)
    mde = (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.8, n - 1)) * sd / np.sqrt(n)
    print(f"LOSO evaluable subjects: {n}; per-subject range "
          f"[{subj.min():+.3f}, {subj.max():+.3f}]")
    print(f"LOSO mean {subj.mean():+.3f}, subject-bootstrap 95% CI "
          f"[{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]")
    print(f"Between-subject SD {sd:.3f}; minimum detectable mean MCC "
          f"(alpha=0.05, power=0.8): {mde:.3f}")


if __name__ == "__main__":
    main()
