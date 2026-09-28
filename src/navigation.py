"""N-dimensional Pareto front + navigation rules (knee-point / max-MCC /
max-Phi / max-lapse-recall) and a Monte Carlo hypervolume estimator.
Works for any number of objectives (here MCC x Phi x lapse recall);
hypervolume is estimated by Monte Carlo dominance counting (see
hypervolume_nd)."""
import numpy as np


def pareto_mask(points):
    """points: (n, d), maximize all d objectives. Returns boolean mask of
    non-dominated points. Works for any number of objectives."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i]):
                mask[i] = False
                break
    return mask


def hypervolume_nd(points, ref=None, n_samples=200_000, seed=0):
    """Monte Carlo estimate of the dominated hypervolume for d>=2
    objectives (all maximized), relative to `ref` (default: all zeros).
    Samples uniformly in the axis-aligned box [ref, max(points)] and
    reports the fraction dominated by the front, times the box volume.
    Converges to the exact value as n_samples grows; used for simplicity
    instead of an exact 3-objective algorithm."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return 0.0, 0.0
    d = pts.shape[1]
    ref = np.zeros(d) if ref is None else np.asarray(ref, dtype=float)
    hi = pts.max(axis=0)
    hi = np.maximum(hi, ref + 1e-9)
    box_vol = float(np.prod(hi - ref))
    if box_vol <= 0:
        return 0.0, 0.0

    rng = np.random.RandomState(seed)
    samples = rng.uniform(ref, hi, size=(n_samples, d))
    # a sample is dominated (counts toward HV) if some front point
    # weakly dominates it in every dimension
    dominated = np.zeros(n_samples, dtype=bool)
    for p in pts:
        dominated |= np.all(samples <= p, axis=1)
    frac = dominated.mean()
    hv = frac * box_vol
    se = box_vol * np.sqrt(frac * (1 - frac) / n_samples)  # Monte Carlo std error
    return float(hv), float(se)


def knee_point(df, obj_cols=("mcc", "phi"), mcc_col="mcc", mcc_floor=0.0):
    """Point on the front closest (Euclidean, normalized 0-1 per
    objective) to the ideal corner (all objectives at their front-wise
    max) -- restricted to points with mcc >= mcc_floor first.

    Without this floor, the knee-point rule can pick a model with
    NEGATIVE MCC (worse than chance) purely because it scores well on
    the other, normalized objectives -- happened on real DROZY search
    output (seed0: knee-point picked a naive-Bayes model at mcc=-0.035).
    A model worse than chance is never an acceptable "balanced" choice
    regardless of how good its faithfulness or lapse-recall look, so the
    candidate set is filtered to non-degenerate predictors before the
    distance-to-ideal-corner computation runs. Falls back to the
    front-wide max-MCC point (with a printed warning) if literally no
    front point clears the floor -- this should be rare but is not
    silently swallowed if it happens."""
    candidates = df[df[mcc_col] >= mcc_floor]
    if len(candidates) == 0:
        print(f"WARNING: knee_point found no front points with {mcc_col} >= {mcc_floor}; "
              f"falling back to max-{mcc_col} point (front-wide, may itself be weak).")
        candidates = df
    vals = candidates[list(obj_cols)].values.astype(float)
    lo, hi = vals.min(axis=0), vals.max(axis=0)
    norm = (vals - lo) / (hi - lo + 1e-12)
    dist = np.sqrt(((1 - norm) ** 2).sum(axis=1))
    return candidates.iloc[[int(np.argmin(dist))]]


def max_col_point(df, col, mcc_col=None, mcc_floor=0.0):
    """argmax of `col`, optionally restricted to mcc >= mcc_floor first --
    same rationale as knee_point's floor: max_lapse_recall (and in
    principle max_phi) can otherwise land on a near-random classifier that
    happens to score well on the OTHER axis alone. Observed on real DROZY
    search output (100-trial/5-seed run, seed2: max_lapse_recall picked a
    logreg trial at mcc=0.0, phi=0.0, lapse_recall=0.667 -- a degenerate
    model, not a genuine recall/precision trade-off). mcc_col=None (the
    default, used for max_mcc itself) skips the floor entirely."""
    if mcc_col is not None:
        candidates = df[df[mcc_col] >= mcc_floor]
        if len(candidates) == 0:
            print(f"WARNING: max_col_point({col}) found no rows with {mcc_col} >= {mcc_floor}; "
                  f"falling back to front-wide argmax (may itself be degenerate).")
            candidates = df
    else:
        candidates = df
    return candidates.iloc[[int(candidates[col].values.argmax())]]


def build_navigation_table(front_df, obj_cols=("mcc", "phi", "lapse_recall"), mcc_floor=1e-6):
    # mcc_floor defaults just above 0.0, not exactly 0.0: an mcc==0.0
    # model (no better than always predicting the majority class) is
    # exactly as degenerate as a negative-mcc one and should also be
    # excluded, but ">= 0.0" would let it through.
    rules = {
        "knee_point": knee_point(front_df, obj_cols=obj_cols, mcc_floor=mcc_floor),
        "max_mcc": max_col_point(front_df, "mcc"),
        "max_phi": max_col_point(front_df, "phi", mcc_col="mcc", mcc_floor=mcc_floor),
        "max_lapse_recall": max_col_point(front_df, "lapse_recall", mcc_col="mcc", mcc_floor=mcc_floor),
    }
    rows = []
    for name, sub in rules.items():
        r = sub.iloc[0]
        row = {"rule": name, "trial": r["trial"], "model_family": r["model_family"]}
        for c in obj_cols:
            row[c] = r[c]
        rows.append(row)
    return rows
