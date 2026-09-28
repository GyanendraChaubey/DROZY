"""
Post-hoc evaluation of the NSGA-II navigation-rule points from the 100-trial
x 5-seed search (see aggregate_results.py), applied only to the
navigation-rule points, not every trial.

Three evaluations per (seed, rule) config, all refit from the trial's
stored hyperparameters (search used CV-averaged scores, not a saved fitted
model):

  1. Sensor-noise stress test (inject_sensor_noise + faithfulness rank
     stability), reusing xai_core.noise_stress_test verbatim.
  2. Leave-one-subject-out (LOSO) transfer, with and without CORAL
     covariance alignment: train on 13 subjects, predict the held-out
     subject, repeated for all 14.
  3. Sleep-deprivation transfer: train on rested epochs (test==1),
     evaluate on sleep-deprived epochs (test==3). Uses only the subjects
     that HAVE a test==1 file (12/14 -- subjects 7 and 9 are missing it).

Scope: NSGA-II front only (the method the paper's claims are about), all 5
seeds, all 4 navigation rules = 20 configs. Random/TPE fronts are not
re-evaluated here -- their role was already served by the hypervolume
comparison in aggregate_results.py.
"""
import sys, time, traceback
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from features import build_dataset, feature_groups
from navigation import pareto_mask, build_navigation_table
from xai_core import (subject_grouped_cv_splits, lapse_recall, noise_stress_test,
                       MLP_ARCHS)

from config import RESULTS_DIR
OBJ_COLS = ("mcc", "phi", "lapse_recall")
N_SEEDS = 5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------- rebuild pipeline
def _trial_params(row):
    return {k[len("param_"):]: v for k, v in row.items() if k.startswith("param_")}


def build_preprocessing(row):
    """Impute + (optional) scaler only -- SMOTE is deliberately excluded
    here so the CORAL path (build_coral_pipeline_parts below) can insert
    covariance alignment between preprocessing and resampling/classifier
    fit, which a monolithic sklearn Pipeline can't express (CORAL needs
    joint access to source AND target features at 'fit' time, not just
    source)."""
    p = _trial_params(row)
    steps = [("impute", SimpleImputer(strategy="median"))]
    scaler = p.get("scaler", "none")
    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    elif scaler == "robust":
        steps.append(("scaler", RobustScaler()))
    from sklearn.pipeline import Pipeline as SkPipeline
    return SkPipeline(steps)


def build_smote(row, y_tr):
    """SMOTE object for this row's search-selected imbalance setting, or
    None if the row didn't select SMOTE or the split can't support it.
    k_neighbors is capped to what the smallest training class can hold
    (see rebuild_pipeline's original docstring note -- post-hoc splits
    like LOSO/sleep-deprivation-transfer/CORAL are not as evenly balanced
    as the search-phase 3-fold splits)."""
    p = _trial_params(row)
    if p.get("imbalance") != "smote":
        return None
    k_neighbors = 3
    if y_tr is not None:
        min_class_n = pd.Series(y_tr).value_counts().min()
        k_neighbors = min(3, int(min_class_n) - 1)
    if k_neighbors >= 1:
        return SMOTE(random_state=0, k_neighbors=k_neighbors)
    return None


def build_model(row, random_state=0):
    p = _trial_params(row)
    family = row["model_family"]
    cw_flag = p.get(f"{family}_class_weight", False)
    cw = "balanced" if cw_flag else None

    if family == "logreg":
        model = LogisticRegression(C=p["lr_C"], penalty=p["lr_penalty"], solver="saga",
                                    max_iter=3000, random_state=random_state, class_weight=cw)
    elif family == "dtree":
        model = DecisionTreeClassifier(max_depth=int(p["dt_max_depth"]),
                                        min_samples_split=int(p["dt_min_samples_split"]),
                                        random_state=random_state, class_weight=cw)
    elif family == "rforest":
        model = RandomForestClassifier(n_estimators=int(p["rf_n_estimators"]),
                                        max_depth=int(p["rf_max_depth"]),
                                        n_jobs=-1, random_state=random_state, class_weight=cw)
    elif family == "extratrees":
        model = ExtraTreesClassifier(n_estimators=int(p["et_n_estimators"]),
                                      max_depth=int(p["et_max_depth"]),
                                      n_jobs=-1, random_state=random_state, class_weight=cw)
    elif family == "xgboost":
        model = xgb.XGBClassifier(n_estimators=int(p["xgb_n_estimators"]),
                                   max_depth=int(p["xgb_max_depth"]), learning_rate=p["xgb_lr"],
                                   objective="multi:softprob", num_class=3,
                                   eval_metric="mlogloss", n_jobs=-1, random_state=random_state)
    elif family == "lightgbm":
        model = lgb.LGBMClassifier(n_estimators=int(p["lgb_n_estimators"]),
                                    num_leaves=int(p["lgb_num_leaves"]), learning_rate=p["lgb_lr"],
                                    n_jobs=-1, random_state=random_state, verbosity=-1)
    elif family == "nb":
        model = GaussianNB()
    elif family == "mlp":
        model = MLPClassifier(hidden_layer_sizes=MLP_ARCHS[p["mlp_units"]], alpha=p["mlp_alpha"],
                               max_iter=400, random_state=random_state)
    else:
        raise ValueError(f"unknown family {family}")
    return family, model


def rebuild_pipeline(row, random_state=0, y_tr=None):
    """Reconstruct the (preprocessing + model) pipeline from a trials_to_df
    row's stored param_* columns -- no fitted model was persisted from the
    search (only CV-averaged scores), so every post-hoc eval refits from
    the winning hyperparameters.

    y_tr (optional): the actual training labels this pipeline is about to
    be fit on -- see build_smote's docstring for why it affects
    k_neighbors."""
    pre = build_preprocessing(row)
    steps = list(pre.steps)
    smote = build_smote(row, y_tr)
    if smote is not None:
        steps.append(("smote", smote))
    family, model = build_model(row, random_state=random_state)
    return ImbPipeline(steps + [("clf", model)]), family


# ---------------------------------------------------------- CORAL align --
def coral_align(Xs, Xt, reg=1e-6):
    """CORrelation ALignment (Sun, Feng & Saenko 2016): recolor the SOURCE
    feature covariance to match the TARGET feature covariance by a single
    linear transform, using only unlabeled target features (no target
    labels involved -- this is the standard unsupervised-domain-adaptation
    setting, and is legitimate for LOSO since the held-out subject's
    epochs, unlabeled, stand in for 'unlabeled target data' a real
    deployment would have from a new user before any lapse is scored).

    Xs: (n_s, d) source (training-subjects) features, already
    imputed/scaled. Xt: (n_t, d) target (held-out subject) features,
    imputed/scaled with the SAME fitted preprocessing. Returns Xs
    recolored to Xt's second-order statistics; Xt itself is returned
    unchanged by the caller, since it is already in its own distribution."""
    d = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + reg * np.eye(d)
    Ct = np.cov(Xt, rowvar=False) + reg * np.eye(d)

    def sqrtm_inv_and_sqrt(C):
        # symmetric C -> (C^{-1/2}, C^{1/2}) via eigendecomposition
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-12, None)
        inv_sqrt = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
        sqrt = V @ np.diag(np.sqrt(w)) @ V.T
        return inv_sqrt, sqrt

    Cs_inv_sqrt, _ = sqrtm_inv_and_sqrt(Cs)
    _, Ct_sqrt = sqrtm_inv_and_sqrt(Ct)
    Xs_mean, Xt_mean = Xs.mean(axis=0), Xt.mean(axis=0)
    Xs_white = (Xs - Xs_mean) @ Cs_inv_sqrt
    Xs_aligned = Xs_white @ Ct_sqrt + Xt_mean
    return Xs_aligned


def loso_eval_coral(row, X, y, groups):
    """Same protocol as loso_eval, but with CORAL covariance alignment
    inserted between preprocessing and classifier fit: the 13-subject
    source pool is recolored to match the held-out subject's (unlabeled)
    feature covariance before the classifier ever sees it. Tests whether
    a standard, cheap unsupervised domain-adaptation step closes any of
    the LOSO gap that per-subject baseline normalization (which has no
    access to a genuinely new subject's data at all) structurally
    cannot."""
    subjects = np.unique(groups)
    mccs, recalls, n_ok = [], [], 0
    for s in subjects:
        te_mask = groups == s
        tr_mask = ~te_mask
        y_tr, y_te = y[tr_mask], y[te_mask]
        if len(np.unique(y_tr)) < 2 or te_mask.sum() < 3:
            continue
        try:
            pre = build_preprocessing(row)
            Xtr_pre = pre.fit_transform(X[tr_mask])
            Xte_pre = pre.transform(X[te_mask])
            Xtr_aligned = coral_align(Xtr_pre, Xte_pre)

            smote = build_smote(row, y_tr)
            if smote is not None:
                Xtr_fit, y_tr_fit = smote.fit_resample(Xtr_aligned, y_tr)
            else:
                Xtr_fit, y_tr_fit = Xtr_aligned, y_tr

            _, model = build_model(row)
            model.fit(Xtr_fit, y_tr_fit)
            pred = model.predict(Xte_pre)
        except Exception:
            continue
        mccs.append(matthews_corrcoef(y_te, pred) if len(np.unique(y_te)) > 1 else np.nan)
        recalls.append(lapse_recall(y_te, pred))
        n_ok += 1
    mccs = np.array(mccs, dtype=float)
    return {
        "loso_coral_mcc_mean": float(np.nanmean(mccs)),
        "loso_coral_mcc_std": float(np.nanstd(mccs)),
        "loso_coral_lapse_recall_mean": float(np.mean(recalls)) if recalls else np.nan,
        "loso_coral_n_subjects_evaluable": n_ok,
    }


# --------------------------------------------------------------- LOSO -----
def loso_eval(row, X, y, groups):
    subjects = np.unique(groups)
    mccs, recalls, n_ok = [], [], 0
    for s in subjects:
        te_mask = groups == s
        tr_mask = ~te_mask
        y_tr = y[tr_mask]
        if len(np.unique(y_tr)) < 2 or te_mask.sum() < 3:
            continue
        pipe, _ = rebuild_pipeline(row, y_tr=y_tr)
        try:
            pipe.fit(X[tr_mask], y_tr)
        except Exception:
            continue
        pred = pipe.predict(X[te_mask])
        y_te = y[te_mask]
        mccs.append(matthews_corrcoef(y_te, pred) if len(np.unique(y_te)) > 1 else np.nan)
        recalls.append(lapse_recall(y_te, pred))
        n_ok += 1
    mccs = np.array(mccs, dtype=float)
    return {
        "loso_mcc_mean": float(np.nanmean(mccs)),
        "loso_mcc_std": float(np.nanstd(mccs)),
        "loso_lapse_recall_mean": float(np.mean(recalls)) if recalls else np.nan,
        "loso_n_subjects_evaluable": n_ok,
    }


# ---------------------------------------------- sleep-deprivation transfer
def sleep_deprivation_transfer(row, X, y, meta):
    tr_mask = (meta["test"] == 1).values
    te_mask = (meta["test"] == 3).values
    y_tr, y_te = y[tr_mask], y[te_mask]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return {"sdt_test3_mcc": np.nan, "sdt_test3_lapse_recall": np.nan,
                "sdt_n_train": int(tr_mask.sum()), "sdt_n_test": int(te_mask.sum())}
    pipe, _ = rebuild_pipeline(row, y_tr=y_tr)
    pipe.fit(X[tr_mask], y_tr)
    pred = pipe.predict(X[te_mask])
    return {
        "sdt_test3_mcc": float(matthews_corrcoef(y_te, pred)),
        "sdt_test3_lapse_recall": lapse_recall(y_te, pred),
        "sdt_n_train": int(tr_mask.sum()), "sdt_n_test": int(te_mask.sum()),
    }


# ------------------------------------------------------------ noise stress
def noise_eval(row, X, y, groups, feature_idx_by_group):
    folds = subject_grouped_cv_splits(y, groups, n_splits=3, seed=0)
    tr_idx, te_idx = folds[0]
    pipe, family = rebuild_pipeline(row, y_tr=y[tr_idx])
    pipe.fit(X[tr_idx], y[tr_idx])

    def pre_t(Xr):
        Xp = Xr
        for name, step in pipe.steps[:-1]:
            if name != "smote":
                Xp = step.transform(Xp)
        return Xp

    X_tr_pre = pre_t(X[tr_idx])
    n_f = min(40, len(te_idx))
    idx = np.random.RandomState(0).choice(len(te_idx), n_f, replace=False)
    X_explain_raw_full = X[te_idx][idx]
    try:
        res = noise_stress_test(pipe, family, X_tr_pre, X_explain_raw_full,
                                 feature_idx_by_group, n_reps=5, seed=0)
    except Exception as e:
        res = {"phi_clean": np.nan, "phi_noisy_mean": np.nan, "phi_drop": np.nan,
               "rank_stability_mean": np.nan, "error": str(e)}
    return res


def main():
    t0 = time.time()
    X_df, y_s, meta = build_dataset()
    X, y = X_df.values, y_s.values
    groups = meta["subject"].values

    col_groups = feature_groups(X_df.columns.tolist())
    col_index = {c: i for i, c in enumerate(X_df.columns)}
    feature_idx_by_group = {g: [col_index[c] for c in cols] for g, cols in col_groups.items()}

    all_rows = []
    for seed in range(N_SEEDS):
        df = pd.read_csv(f"{RESULTS_DIR}/nsga2_seed{seed}.csv")
        vals = df[list(OBJ_COLS)].values
        front = df[pareto_mask(vals)].reset_index(drop=True)
        nav = build_navigation_table(front, OBJ_COLS)

        for nrow in nav:
            trial_num = nrow["trial"]
            row = df[df["trial"] == trial_num].iloc[0].to_dict()
            log(f"seed{seed} / {nrow['rule']}: trial={trial_num} family={row['model_family']} "
                f"mcc={row['mcc']:.4f} phi={row['phi']:.4f} lapse_recall={row['lapse_recall']:.4f}")
            out = {"seed": seed, "rule": nrow["rule"], "trial": trial_num,
                   "model_family": row["model_family"],
                   "search_mcc": row["mcc"], "search_phi": row["phi"],
                   "search_lapse_recall": row["lapse_recall"]}
            try:
                out.update(loso_eval(row, X, y, groups))
            except Exception as e:
                log(f"  LOSO FAILED: {e}\n{traceback.format_exc()}")
            try:
                out.update(loso_eval_coral(row, X, y, groups))
            except Exception as e:
                log(f"  LOSO+CORAL FAILED: {e}\n{traceback.format_exc()}")
            try:
                out.update(sleep_deprivation_transfer(row, X, y, meta))
            except Exception as e:
                log(f"  sleep-deprivation transfer FAILED: {e}\n{traceback.format_exc()}")
            try:
                out.update(noise_eval(row, X, y, groups, feature_idx_by_group))
            except Exception as e:
                log(f"  noise stress FAILED: {e}\n{traceback.format_exc()}")
            all_rows.append(out)

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(f"{RESULTS_DIR}/transfer_eval.csv", index=False)
    log(f"DONE in {time.time()-t0:.0f}s -- saved {RESULTS_DIR}/transfer_eval.csv "
        f"({len(out_df)} rows)")


if __name__ == "__main__":
    main()
