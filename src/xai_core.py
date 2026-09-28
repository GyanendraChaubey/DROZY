"""
Core search and explanation-evaluation library: model families,
preprocessing search space, SHAP dispatch, insertion/deletion faithfulness
(Phi), lapse recall, and the sensor-noise stress test.

  1. CV IS SUBJECT-GROUPED. Adjacent 30 s epochs within one subject's
     session are near-identical in physiological state, so the only split
     that prevents identity leakage keeps a whole subject's epochs on one
     side. StratifiedGroupKFold (group=subject) does that and keeps all
     three classes on both sides of every fold, even though 13 of 14
     subjects lack at least one class. (The paper shows that this label
     stratification is itself optimistic when labels cluster by subject.)
  2. TWO IMBALANCE LEVERS. The lapse class is a real minority (72/720,
     ~10%), so the search can choose class_weight='balanced', SMOTE, both
     or neither, per trial.

KernelExplainer calls are reseeded per call (nsamples=300) so that Phi
comparisons are not confounded by SHAP sampling noise.
"""
import numpy as np
import pandas as pd
import shap
import optuna
from optuna.samplers import NSGAIISampler, TPESampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import matthews_corrcoef, recall_score
import xgboost as xgb
import lightgbm as lgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import spearmanr

optuna.logging.set_verbosity(optuna.logging.WARNING)

MLP_ARCHS = {"32": (32,), "64": (64,), "64_32": (64, 32)}
# kNN is not included: it has no native class_weight lever
# and no SHAP explainer path other than the already-well-represented
# KernelExplainer branch (shared with nb/mlp) -- keeping it added a 10th
# family with no new information about the class_weight/explainer-type
# question this study is asking.
MODEL_FAMILIES = ["logreg", "dtree", "rforest", "extratrees", "xgboost",
                   "lightgbm", "nb", "mlp"]
CLASS_WEIGHT_FAMILIES = {"logreg", "dtree", "rforest", "extratrees"}


# ---------------------------------------------------------------- CV -----
def subject_grouped_cv_splits(y, groups, n_splits=3, seed=0):
    """StratifiedGroupKFold: every fold keeps a whole subject's epochs on
    one side, and stratifies by class as well as it can given the
    grouping constraint. Puts all 3 classes in both train and test for
    every fold at n_splits=3, despite 13/14 subjects individually lacking
    at least one class."""
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y, groups))


# ------------------------------------------------------- preprocessing ---
def build_preprocessing_steps(trial, n_features):
    steps = [("impute", SimpleImputer(strategy="median"))]
    # Handles occasional NaN from failed ECG peak detection (see
    # features._ecg_features) -- rare but not absent.

    scaler_choice = trial.suggest_categorical("scaler", ["none", "standard", "minmax", "robust"])
    if scaler_choice == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler_choice == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    elif scaler_choice == "robust":
        steps.append(("scaler", RobustScaler()))

    imb_choice = trial.suggest_categorical("imbalance", ["none", "smote"])
    if imb_choice == "smote":
        # k_neighbors=3: the lapse class has as few as a handful of
        # examples in some CV folds' training split; SMOTE's default
        # k_neighbors=5 would error out on those folds.
        steps.append(("smote", SMOTE(random_state=0, k_neighbors=3)))

    return steps


def lapse_recall(y_true, y_pred, lapse_class=2):
    """Recall on the minority lapse class alone. The third search
    objective (alongside MCC and Phi): does the search catch drowsiness
    lapses specifically, or trade them away for overall correctness?
    0.0 if the lapse class isn't present in y_true for this fold (should
    not happen with subject_grouped_cv_splits at n_splits=3, but guarded
    defensively)."""
    if lapse_class not in np.unique(y_true):
        return 0.0
    return float(recall_score(y_true, y_pred, labels=[lapse_class], average="macro", zero_division=0))


def suggest_model(trial, n_classes, random_state=0):
    family = trial.suggest_categorical("model_family", MODEL_FAMILIES)
    use_cw = (family in CLASS_WEIGHT_FAMILIES and
              trial.suggest_categorical(f"{family}_class_weight", [False, True]))
    cw = "balanced" if use_cw else None

    if family == "logreg":
        C = trial.suggest_float("lr_C", 1e-3, 1e2, log=True)
        penalty = trial.suggest_categorical("lr_penalty", ["l1", "l2"])
        model = LogisticRegression(C=C, penalty=penalty, solver="saga",
                                    max_iter=3000, random_state=random_state,
                                    class_weight=cw)
    elif family == "dtree":
        depth = trial.suggest_int("dt_max_depth", 2, 20)
        min_split = trial.suggest_int("dt_min_samples_split", 2, 20)
        model = DecisionTreeClassifier(max_depth=depth, min_samples_split=min_split,
                                        random_state=random_state, class_weight=cw)
    elif family == "rforest":
        n_est = trial.suggest_int("rf_n_estimators", 20, 200)
        depth = trial.suggest_int("rf_max_depth", 3, 18)
        model = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                        n_jobs=-1, random_state=random_state,
                                        class_weight=cw)
    elif family == "extratrees":
        n_est = trial.suggest_int("et_n_estimators", 20, 200)
        depth = trial.suggest_int("et_max_depth", 3, 18)
        model = ExtraTreesClassifier(n_estimators=n_est, max_depth=depth,
                                      n_jobs=-1, random_state=random_state,
                                      class_weight=cw)
    elif family == "xgboost":
        n_est = trial.suggest_int("xgb_n_estimators", 20, 200)
        depth = trial.suggest_int("xgb_max_depth", 2, 10)
        lr = trial.suggest_float("xgb_lr", 1e-3, 0.5, log=True)
        model = xgb.XGBClassifier(n_estimators=n_est, max_depth=depth, learning_rate=lr,
                                   objective="multi:softprob", num_class=n_classes,
                                   eval_metric="mlogloss", n_jobs=-1,
                                   random_state=random_state)
    elif family == "lightgbm":
        n_est = trial.suggest_int("lgb_n_estimators", 20, 200)
        leaves = trial.suggest_int("lgb_num_leaves", 7, 100)
        lr = trial.suggest_float("lgb_lr", 1e-3, 0.5, log=True)
        model = lgb.LGBMClassifier(n_estimators=n_est, num_leaves=leaves, learning_rate=lr,
                                    n_jobs=-1, random_state=random_state, verbosity=-1)
    elif family == "nb":
        model = GaussianNB()
    elif family == "mlp":
        arch_key = trial.suggest_categorical("mlp_units", list(MLP_ARCHS.keys()))
        alpha = trial.suggest_float("mlp_alpha", 1e-5, 1e-1, log=True)
        model = MLPClassifier(hidden_layer_sizes=MLP_ARCHS[arch_key], alpha=alpha,
                               max_iter=400, random_state=random_state)
    return family, model


# ----------------------------------------------------------- SHAP/Phi ----
TREE_FAMILIES = {"dtree", "rforest", "extratrees", "xgboost", "lightgbm"}


def get_shap_values(model, family, X_background, X_explain, n_classes, seed=0):
    if family in TREE_FAMILIES:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_explain, check_additivity=False)
    elif family == "logreg":
        explainer = shap.LinearExplainer(model, X_background)
        sv = explainer.shap_values(X_explain)
    else:
        # Reseeded per call + nsamples=300 so repeated Phi evaluations are
        # not confounded by KernelExplainer's sampling noise.
        np.random.seed(seed)
        bg = shap.kmeans(X_background, min(25, len(X_background)))
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        sv = explainer.shap_values(X_explain, nsamples=300, silent=True)

    if isinstance(sv, list):
        return np.array(sv)
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return np.transpose(sv, (2, 0, 1))
    if n_classes == 2:
        return np.stack([-sv, sv])
    return np.stack([sv] * n_classes)


def faithfulness(model, family, X_background, X_explain, k_steps=8, feature_groups=None, seed=0):
    """Phi = InsertionAUC - DeletionAUC, per-instance on the predicted
    class, grouped by physiological block (EEG region / EOG / EMG / ECG)
    when feature_groups is given."""
    n, d = X_explain.shape
    background_vec = np.median(X_background, axis=0)
    proba_full = model.predict_proba(X_explain)
    target_cls = proba_full.argmax(axis=1)
    n_classes_local = proba_full.shape[1]

    sv_all = get_shap_values(model, family, X_background, X_explain, n_classes_local, seed=seed)
    sv = np.stack([sv_all[target_cls[i], i, :] for i in range(n)])

    if feature_groups is None:
        units = [[j] for j in range(d)]
    else:
        units = list(feature_groups.values())
    n_units = len(units)
    unit_importance = np.stack([np.abs(sv[:, u]).sum(axis=1) for u in units], axis=1)
    order = np.argsort(-unit_importance, axis=1)

    steps = np.unique(np.linspace(0, n_units, k_steps, dtype=int))
    x = steps / n_units

    del_curves = np.zeros((n, len(steps)))
    ins_curves = np.zeros((n, len(steps)))
    for si, k in enumerate(steps):
        Xdel = X_explain.copy()
        Xins = np.tile(background_vec, (n, 1))
        for i in range(n):
            unit_idx = order[i, :k]
            cols = np.concatenate([units[u] for u in unit_idx]) if k > 0 else np.array([], dtype=int)
            if len(cols):
                Xdel[i, cols] = background_vec[cols]
                Xins[i, cols] = X_explain[i, cols]
        del_curves[:, si] = model.predict_proba(Xdel)[np.arange(n), target_cls]
        ins_curves[:, si] = model.predict_proba(Xins)[np.arange(n), target_cls]

    del_auc = np.array([np.trapezoid(del_curves[i], x) for i in range(n)])
    ins_auc = np.array([np.trapezoid(ins_curves[i], x) for i in range(n)])
    phi = float((ins_auc - del_auc).mean())
    return phi, sv, order


# --------------------------------------------------- sensor-noise stress -
def inject_sensor_noise(X, rng, rel_noise_std=0.15, missing_frac=0.15, noisy_cols=None):
    """Simulates degraded acquisition: multiplicative Gaussian noise
    (rel_noise_std, a literature-typical relative-error magnitude) plus
    MCAR missingness. All 43 features are physiologically sensor-derived,
    so noisy_cols defaults to every feature column."""
    Xn = X.copy()
    cols = noisy_cols if noisy_cols is not None else np.arange(X.shape[1])
    noise = rng.normal(1.0, rel_noise_std, size=(X.shape[0], len(cols)))
    Xn[:, cols] = Xn[:, cols] * noise
    miss_mask = rng.random(size=(X.shape[0], len(cols))) < missing_frac
    for j_local, j in enumerate(cols):
        Xn[miss_mask[:, j_local], j] = np.nan
    return Xn


def noise_stress_test(pipe, family, X_background_pre, X_explain_raw, feature_idx_by_group,
                       n_reps=5, seed=0, k_steps=8, sensor_cols=None):
    rng = np.random.RandomState(seed)
    fitted_model = pipe.named_steps["clf"]

    def transform_through_pre(Xraw):
        Xp = Xraw
        for name, step in pipe.steps[:-1]:
            if name != "smote":
                Xp = step.transform(Xp)
        return Xp

    X_explain_pre = transform_through_pre(X_explain_raw)
    phi_clean, sv_clean, _ = faithfulness(fitted_model, family, X_background_pre, X_explain_pre,
                                           k_steps=k_steps, feature_groups=feature_idx_by_group,
                                           seed=seed)
    units = list(feature_idx_by_group.values())
    clean_rank = -np.stack([np.abs(sv_clean[:, u]).sum(axis=1) for u in units], axis=1).mean(0)
    clean_order = np.argsort(clean_rank)

    phis, rhos = [], []
    for r in range(n_reps):
        Xn_raw = inject_sensor_noise(X_explain_raw, np.random.RandomState(seed * 100 + r),
                                      noisy_cols=sensor_cols)
        Xn_pre = transform_through_pre(Xn_raw)
        phi_n, sv_n, _ = faithfulness(fitted_model, family, X_background_pre, Xn_pre,
                                       k_steps=k_steps, feature_groups=feature_idx_by_group,
                                       seed=seed * 100 + r + 1)
        noisy_rank = -np.stack([np.abs(sv_n[:, u]).sum(axis=1) for u in units], axis=1).mean(0)
        noisy_order = np.argsort(noisy_rank)
        rho, _ = spearmanr(clean_order, noisy_order)
        phis.append(phi_n)
        rhos.append(rho if not np.isnan(rho) else 0.0)

    return {
        "phi_clean": phi_clean,
        "phi_noisy_mean": float(np.mean(phis)),
        "phi_noisy_std": float(np.std(phis)),
        "phi_drop": phi_clean - float(np.mean(phis)),
        "rank_stability_mean": float(np.mean(rhos)),
        "rank_stability_std": float(np.std(rhos)),
    }
