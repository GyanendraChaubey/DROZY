"""
Main search driver. A single cross-subject search (not per-subject: 13 of
14 subjects individually lack at least one of the 3 classes, so a
per-subject search is not viable) across all 720 epochs, per seed:

  - NSGA-II, 3-objective (MCC x Phi x lapse-recall), N_SEEDS x N_TRIALS
  - TPE, single-objective (MCC only) baseline -- same budget
  - Random search, 3-objective -- same budget

all under subject-grouped (StratifiedGroupKFold) CV. Search-phase Phi
uses UNGROUPED faithfulness (comparable across all 3 methods); the
grouped/noise-stress/transfer evaluations apply post-hoc to just the 4
navigation-rule points in run_transfer_eval.py.

Resumable: every (method, seed) is its own Optuna study in a shared
sqlite DB; re-running skips studies that already have enough trials.
"""
import sys, time, argparse, traceback
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import NSGAIISampler, TPESampler, RandomSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import matthews_corrcoef

from features import build_dataset
from xai_core import (subject_grouped_cv_splits, build_preprocessing_steps,
                       suggest_model, faithfulness, lapse_recall)

optuna.logging.set_verbosity(optuna.logging.WARNING)

from config import RESULTS_DIR
LOG_PATH = RESULTS_DIR / "run_search.log"
STUDY_DB = f"sqlite:///{RESULTS_DIR / 'studies.db'}"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def make_objective(X, y, groups, seed, cv_splits=3, faith_n=40, faith_steps=6, single_obj=False):
    n_classes = len(np.unique(y))
    folds = subject_grouped_cv_splits(y, groups, n_splits=cv_splits, seed=seed)

    def objective(trial):
        pre_steps = build_preprocessing_steps(trial, X.shape[1])
        family, model = suggest_model(trial, n_classes, random_state=seed)
        mccs, phis, recalls = [], [], []
        for tr_idx, te_idx in folds:
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            if len(np.unique(y_tr)) < 2:
                raise optuna.TrialPruned()
            pipe = ImbPipeline(pre_steps + [("clf", model)])
            try:
                pipe.fit(X_tr, y_tr)
            except Exception:
                raise optuna.TrialPruned()
            pred = pipe.predict(X_te)
            mccs.append(matthews_corrcoef(y_te, pred))
            recalls.append(lapse_recall(y_te, pred))

            if not single_obj:
                fitted = pipe.named_steps["clf"]
                def pre_t(Xr):
                    Xp = Xr
                    for name, step in pipe.steps[:-1]:
                        if name != "smote":
                            Xp = step.transform(Xp)
                    return Xp
                X_tr_pre, X_te_pre = pre_t(X_tr), pre_t(X_te)
                n_f = min(faith_n, len(X_te_pre))
                idx = np.random.RandomState(seed).choice(len(X_te_pre), n_f, replace=False)
                try:
                    phi, _, _ = faithfulness(fitted, family, X_tr_pre, X_te_pre[idx], k_steps=faith_steps)
                except Exception:
                    phi = 0.0
                phis.append(phi)

        trial.set_user_attr("model_family", family)
        if single_obj:
            return float(np.mean(mccs))
        return float(np.mean(mccs)), float(np.mean(phis)), float(np.mean(recalls))

    return objective


def run_study(name, sampler, directions, X, y, groups, cv_splits, seed, n_trials, single_obj=False):
    study = optuna.create_study(study_name=name, directions=directions, sampler=sampler,
                                 storage=STUDY_DB, load_if_exists=True)
    remaining = max(0, n_trials - len(study.trials))
    if remaining > 0:
        log(f"  {name}: running {remaining} new trials ({len(study.trials)} done)")
        study.optimize(make_objective(X, y, groups, seed, cv_splits, single_obj=single_obj),
                        n_trials=remaining, show_progress_bar=False,
                        catch=(Exception,))
    else:
        log(f"  {name}: already complete ({len(study.trials)} trials)")
    return study


def trials_to_df(study, single_obj=False):
    rows = []
    for t in study.trials:
        if t.values is None:
            continue
        row = {"trial": t.number, "model_family": t.user_attrs.get("model_family", "unknown"),
               **{f"param_{k}": v for k, v in t.params.items()}}
        if single_obj:
            row["mcc"] = t.values[0]
        else:
            row["mcc"], row["phi"], row["lapse_recall"] = t.values
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--cv-splits", type=int, default=3)
    args = ap.parse_args()

    log(f"Starting search: n_trials={args.n_trials} n_seeds={args.n_seeds} cv_splits={args.cv_splits}")
    t0 = time.time()
    X_df, y_s, meta = build_dataset()
    X, y = X_df.values, y_s.values
    groups = meta["subject"].values
    log(f"n={len(X)}, class dist={pd.Series(y).value_counts().sort_index().to_dict()}, "
        f"n_subjects={len(np.unique(groups))}")

    for seed in range(args.n_seeds):
        try:
            # population_size deliberately << n_trials: NSGAIISampler
            # defaults to population_size=50, which for n_trials=40 means
            # the run never leaves its randomly-sampled initial
            # population -- NSGA-II never actually evolves (no selection/
            # crossover/mutation ever fires), and degrades to exactly
            # RandomSampler's behavior. Caught by nsga2_seed0.csv and
            # random_seed0.csv coming back byte-identical on the first
            # run. population_size=10 gives 4 real generations across 40
            # trials instead.
            s = run_study(f"nsga2_seed{seed}", NSGAIISampler(seed=seed, population_size=10),
                           ["maximize", "maximize", "maximize"], X, y, groups,
                           args.cv_splits, seed, args.n_trials)
            trials_to_df(s).to_csv(f"{RESULTS_DIR}/nsga2_seed{seed}.csv", index=False)

            s = run_study(f"random_seed{seed}", RandomSampler(seed=seed),
                           ["maximize", "maximize", "maximize"], X, y, groups,
                           args.cv_splits, seed, args.n_trials)
            trials_to_df(s).to_csv(f"{RESULTS_DIR}/random_seed{seed}.csv", index=False)

            s = run_study(f"tpe_seed{seed}", TPESampler(seed=seed),
                           ["maximize"], X, y, groups, args.cv_splits, seed,
                           args.n_trials, single_obj=True)
            trials_to_df(s, single_obj=True).to_csv(f"{RESULTS_DIR}/tpe_seed{seed}.csv", index=False)
        except Exception as e:
            log(f"SEED {seed} FAILED: {e}\n{traceback.format_exc()}")
    log(f"ALL DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
