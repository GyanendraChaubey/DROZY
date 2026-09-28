"""
Is the LOSO collapse a LABEL-shift (class-prior) problem rather than a
covariate-shift one?

Evidence so far (diagnose_gap.py, null_baseline.py, strat check):
  * stratified subject-grouped 3-fold CV (StratifiedGroupKFold, as in the
    search): MCC ~ +0.09 on 10 fresh fold seeds -- NOT a selection artifact.
  * the SAME person-held-out 3-fold design with random (unstratified)
    subject partitions: MCC collapses toward chance.
  * LOSO (unstratified by construction): chance.
  * pure-noise features: every protocol ~0 (LOSO pooled slightly negative).
StratifiedGroupKFold chooses which subjects share a fold USING THEIR
LABELS so train/test class priors match. A deployment never gets that:
a new person's lapse rate is unknown. Labels here are strongly
subject-clustered, so random subject hold-out induces large prior shift.

This script, for all 20 navigation-rule configs:
  1. unstratified subject-grouped 3-fold (10 random subject partitions)
  2. LOSO + EM prior adaptation (Saerens, Latinne & Decaestecker 2002):
     re-estimate the held-out subject's class prior from the model's own
     unlabeled posteriors and re-weight -- unsupervised, like CORAL, but
     targets label shift instead of covariate shift.
  3. per-subject LOSO MCCs for a subject-level bootstrap CI.
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from features import build_dataset
from navigation import pareto_mask, build_navigation_table
from run_transfer_eval import (build_preprocessing, build_smote, build_model,
                               rebuild_pipeline, RESULTS_DIR, OBJ_COLS, N_SEEDS)

CLASSES = np.array([0, 1, 2])


def em_prior_adjust(P, prior_tr, n_iter=100, tol=1e-6):
    """Saerens et al. (2002) EM: P = source posteriors (n, K), prior_tr = the
    class prior the classifier was fit under. Returns adjusted posteriors."""
    prior_tr = np.clip(prior_tr, 1e-6, None)
    prior = prior_tr.copy()
    Q = P
    for _ in range(n_iter):
        Q = P * (prior / prior_tr)
        Q /= np.clip(Q.sum(axis=1, keepdims=True), 1e-12, None)
        new = Q.mean(axis=0)
        if np.abs(new - prior).max() < tol:
            prior = new
            break
        prior = new
    return Q


def full_proba(model, Xte):
    P = np.zeros((len(Xte), len(CLASSES)))
    P[:, np.searchsorted(CLASSES, model.classes_)] = model.predict_proba(Xte)
    return P


def loso_label_shift(row, X, y, groups):
    ya, p_plain, p_em, subj = [], [], [], []
    for s in np.unique(groups):
        te = groups == s; tr = ~te
        pre = build_preprocessing(row)
        Xtr = pre.fit_transform(X[tr]); Xte = pre.transform(X[te])
        smote = build_smote(row, y[tr])
        Xf, yf = smote.fit_resample(Xtr, y[tr]) if smote is not None else (Xtr, y[tr])
        _, model = build_model(row)
        model.fit(Xf, yf)
        P = full_proba(model, Xte)
        prior_tr = np.array([(yf == c).mean() for c in CLASSES])
        Q = em_prior_adjust(P, prior_tr)
        pp, pe = CLASSES[P.argmax(1)], CLASSES[Q.argmax(1)]
        ya.append(y[te]); p_plain.append(pp); p_em.append(pe)
        subj.append((s, y[te], pp, pe))
    per = lambda k: [matthews_corrcoef(t, p) for (_, t, *ps) in subj
                     for p in [ps[k]] if len(np.unique(t)) > 1]
    return {
        "loso_per_subject_mcc_proba": float(np.mean(per(0))),
        "loso_em_per_subject_mcc": float(np.mean(per(1))),
        "loso_em_pooled_mcc": float(matthews_corrcoef(np.concatenate(ya), np.concatenate(p_em))),
    }, subj


def unstratified_3fold(row, X, y, groups, reps=10):
    S = np.unique(groups)
    fold, per = [], []
    for rep in range(reps):
        perm = np.random.RandomState(rep).permutation(S)
        for part in np.array_split(perm, 3):
            te = np.isin(groups, part); tr = ~te
            if len(np.unique(y[tr])) < 3:
                continue
            pipe, _ = rebuild_pipeline(row, y_tr=y[tr])
            pipe.fit(X[tr], y[tr]); p = pipe.predict(X[te])
            fold.append(matthews_corrcoef(y[te], p))
            for s in part:
                m = groups[te] == s
                if len(np.unique(y[te][m])) > 1:
                    per.append(matthews_corrcoef(y[te][m], p[m]))
    return {"unstrat3fold_mcc": float(np.mean(fold)),
            "unstrat3fold_per_subject_mcc": float(np.mean(per))}


def main():
    t0 = time.time()
    X_df, y_s, meta = build_dataset()
    X, y = X_df.values, y_s.values
    groups = meta["subject"].values
    rows, subj_rows = [], []
    for seed in range(N_SEEDS):
        df = pd.read_csv(f"{RESULTS_DIR}/nsga2_seed{seed}.csv")
        front = df[pareto_mask(df[list(OBJ_COLS)].values)].reset_index(drop=True)
        for nrow in build_navigation_table(front, OBJ_COLS):
            row = df[df["trial"] == nrow["trial"]].iloc[0].to_dict()
            r = {"seed": seed, "rule": nrow["rule"], "model_family": row["model_family"]}
            r.update(unstratified_3fold(row, X, y, groups))
            ls, subj = loso_label_shift(row, X, y, groups)
            r.update(ls)
            for s, t, pp, pe in subj:
                ok = len(np.unique(t)) > 1
                subj_rows.append({"seed": seed, "rule": nrow["rule"], "subject": s,
                                  "mcc": matthews_corrcoef(t, pp) if ok else np.nan,
                                  "mcc_em": matthews_corrcoef(t, pe) if ok else np.nan})
            rows.append(r)
            print(f"seed{seed} {nrow['rule']:>17} {row['model_family']:>10}: "
                  + " ".join(f"{k}={v:+.3f}" for k, v in r.items()
                             if isinstance(v, float)), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS_DIR}/label_shift_eval.csv", index=False)
    pd.DataFrame(subj_rows).to_csv(f"{RESULTS_DIR}/loso_per_subject_em.csv", index=False)
    print("\nMEANS (std):")
    for c in [c for c in out.columns if c.endswith("mcc")]:
        print(f"  {c:>28}: {out[c].mean():+.3f} ({out[c].std():.3f})")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
