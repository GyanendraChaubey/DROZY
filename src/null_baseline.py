"""
Null (pure-noise-feature) baseline for every evaluation protocol.

Motivation: diagnose_gap.py showed subject-grouped 3-fold CV (also
person-held-out) gives MCC ~ +0.09 on fresh folds while LOSO gives
-0.013 (per-subject mean) / -0.068 (pooled predictions). Both protocols
hold out whole subjects, so the difference is not leakage. A candidate
mechanism is leave-one-group-out class-prior shift: labels here are
heavily subject-clustered (e.g. subject 1 holds 44/68 'optimal' epochs),
so holding one subject out shifts the TRAINING prior away from that
subject's own label mix -- anti-correlating predictions with truth.

Test: replace the 43 features with i.i.d. Gaussian noise (same shape,
same labels, same subject grouping) and run each protocol with the same
20 configurations. Whatever MCC a protocol produces on noise is its
chance level; real-feature results must be read relative to it.
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

NOISE_SEEDS = range(5)
FOLD_SEEDS = range(100, 103)


def fit_predict(row, X, y, tr, te):
    pipe, _ = rebuild_pipeline(row, y_tr=y[tr])
    pipe.fit(X[tr], y[tr])
    return pipe.predict(X[te])


def protocols(row, X, y, groups, meta):
    out = {}
    f = []
    for fs in FOLD_SEEDS:
        for tr, te in subject_grouped_cv_splits(y, groups, n_splits=3, seed=fs):
            f.append(matthews_corrcoef(y[te], fit_predict(row, X, y, tr, te)))
    out["fresh3fold_mcc"] = float(np.mean(f))

    ya, pa, ps = [], [], []
    for s in np.unique(groups):
        te = np.where(groups == s)[0]; tr = np.where(groups != s)[0]
        p = fit_predict(row, X, y, tr, te)
        ya.append(y[te]); pa.append(p)
        if len(np.unique(y[te])) > 1:
            ps.append(matthews_corrcoef(y[te], p))
    out["loso_pooled_mcc"] = float(matthews_corrcoef(np.concatenate(ya), np.concatenate(pa)))
    out["loso_per_subject_mcc"] = float(np.mean(ps))

    tr = np.where(meta["test"].values == 1)[0]; te = np.where(meta["test"].values == 3)[0]
    out["sdt_mcc"] = float(matthews_corrcoef(y[te], fit_predict(row, X, y, tr, te)))
    return out


def main():
    t0 = time.time()
    X_df, y_s, meta = build_dataset()
    X, y = X_df.values, y_s.values
    groups = meta["subject"].values

    configs = []
    for seed in range(N_SEEDS):
        df = pd.read_csv(f"{RESULTS_DIR}/nsga2_seed{seed}.csv")
        front = df[pareto_mask(df[list(OBJ_COLS)].values)].reset_index(drop=True)
        for nrow in build_navigation_table(front, OBJ_COLS):
            configs.append((seed, nrow["rule"],
                            df[df["trial"] == nrow["trial"]].iloc[0].to_dict()))

    rows = []
    for ns in NOISE_SEEDS:
        Xn = np.random.RandomState(1000 + ns).normal(size=X.shape)
        for seed, rule, row in configs:
            r = {"noise_seed": ns, "seed": seed, "rule": rule}
            r.update(protocols(row, Xn, y, groups, meta))
            rows.append(r)
        print(f"noise seed {ns} done ({time.time()-t0:.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS_DIR}/null_baseline.csv", index=False)
    print("\nNULL (noise features) means (std) over 5 noise seeds x 20 configs:")
    for c in ["fresh3fold_mcc", "loso_pooled_mcc", "loso_per_subject_mcc", "sdt_mcc"]:
        print(f"  {c:>22}: {out[c].mean():+.3f} ({out[c].std():.3f})")


if __name__ == "__main__":
    main()
