"""
Decompose the "pooled search MCC (0.105) vs LOSO MCC (-0.013)" gap.

The search CV is ALREADY subject-held-out (StratifiedGroupKFold, group =
subject), so the gap cannot be a subject-leakage effect. Candidate
explanations, each isolated here for the same 20 navigation-rule configs:

  A. Selection optimism: search MCC is the score of the trial that WON a
     100-trial search on those exact folds. -> re-evaluate each config on
     fresh subject-grouped 3-fold splits (10 new fold seeds).
  B. Estimand: search MCC pools predictions over the ~4-5 subjects in each
     test fold (between-subject + within-subject discrimination); reported
     LOSO MCC is the mean of per-subject MCCs (within-subject only, and
     undefined for single-class subjects). -> compute LOSO MCC both ways
     from the same LOSO predictions.
  C. Fold-size / training-set-size: 3-fold trains on ~9-10 subjects, LOSO
     on 13 -- goes the "wrong" way for explaining a drop, noted only.

Also saves per-subject LOSO MCCs for a subject-level bootstrap CI.
"""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from features import build_dataset
from navigation import pareto_mask, build_navigation_table
from xai_core import subject_grouped_cv_splits
from run_transfer_eval import rebuild_pipeline, RESULTS_DIR, OBJ_COLS, N_SEEDS

FRESH_FOLD_SEEDS = range(100, 110)


def fit_predict(row, X, y, tr, te):
    pipe, _ = rebuild_pipeline(row, y_tr=y[tr])
    pipe.fit(X[tr], y[tr])
    return pipe.predict(X[te])


def main():
    t0 = time.time()
    X_df, y_s, meta = build_dataset()
    X, y = X_df.values, y_s.values
    groups = meta["subject"].values
    subjects = np.unique(groups)

    cfg_rows, subj_rows = [], []
    for seed in range(N_SEEDS):
        df = pd.read_csv(f"{RESULTS_DIR}/nsga2_seed{seed}.csv")
        front = df[pareto_mask(df[list(OBJ_COLS)].values)].reset_index(drop=True)
        for nrow in build_navigation_table(front, OBJ_COLS):
            row = df[df["trial"] == nrow["trial"]].iloc[0].to_dict()

            # A. fresh subject-grouped 3-fold CV, same estimand as search
            fresh_fold_mccs, fresh_subj_mccs = [], []
            for fs in FRESH_FOLD_SEEDS:
                for tr, te in subject_grouped_cv_splits(y, groups, n_splits=3, seed=fs):
                    if len(np.unique(y[tr])) < 2:
                        continue
                    pred = fit_predict(row, X, y, tr, te)
                    fresh_fold_mccs.append(matthews_corrcoef(y[te], pred))
                    # per-subject MCC inside the same test fold (estimand B)
                    for s in np.unique(groups[te]):
                        m = groups[te] == s
                        if len(np.unique(y[te][m])) > 1:
                            fresh_subj_mccs.append(matthews_corrcoef(y[te][m], pred[m]))

            # B. LOSO, keeping all predictions
            y_all, p_all, g_all = [], [], []
            for s in subjects:
                te = np.where(groups == s)[0]
                tr = np.where(groups != s)[0]
                pred = fit_predict(row, X, y, tr, te)
                y_all.append(y[te]); p_all.append(pred); g_all.append(groups[te])
                ys = y[te]
                subj_rows.append({
                    "seed": seed, "rule": nrow["rule"], "subject": s,
                    "n": len(te), "n_classes": len(np.unique(ys)),
                    "mcc": matthews_corrcoef(ys, pred) if len(np.unique(ys)) > 1 else np.nan,
                    "acc": float((ys == pred).mean()),
                })
            y_all = np.concatenate(y_all); p_all = np.concatenate(p_all)
            per_subj = [r["mcc"] for r in subj_rows
                        if r["seed"] == seed and r["rule"] == nrow["rule"]]

            cfg_rows.append({
                "seed": seed, "rule": nrow["rule"], "trial": nrow["trial"],
                "model_family": row["model_family"], "search_mcc": row["mcc"],
                "fresh3fold_mcc": float(np.mean(fresh_fold_mccs)),
                "fresh3fold_per_subject_mcc": float(np.nanmean(fresh_subj_mccs)),
                "loso_pooled_mcc": float(matthews_corrcoef(y_all, p_all)),
                "loso_per_subject_mcc": float(np.nanmean(per_subj)),
            })
            print(f"seed{seed} {nrow['rule']:>17}: search={row['mcc']:.3f} "
                  f"fresh3f={cfg_rows[-1]['fresh3fold_mcc']:.3f} "
                  f"fresh3f_subj={cfg_rows[-1]['fresh3fold_per_subject_mcc']:.3f} "
                  f"loso_pooled={cfg_rows[-1]['loso_pooled_mcc']:.3f} "
                  f"loso_subj={cfg_rows[-1]['loso_per_subject_mcc']:.3f}", flush=True)

    cfg = pd.DataFrame(cfg_rows)
    cfg.to_csv(f"{RESULTS_DIR}/gap_decomposition.csv", index=False)
    pd.DataFrame(subj_rows).to_csv(f"{RESULTS_DIR}/loso_per_subject.csv", index=False)
    print("\nMEANS (std) over 20 configs:")
    for c in ["search_mcc", "fresh3fold_mcc", "fresh3fold_per_subject_mcc",
              "loso_pooled_mcc", "loso_per_subject_mcc"]:
        print(f"  {c:>28}: {cfg[c].mean():+.3f} ({cfg[c].std():.3f})")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
