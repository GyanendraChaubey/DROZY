"""
Aggregate the 100-trial x 5-seed x {nsga2, random, tpe} search output.

Reports, per seed:
  - Pareto front size (nsga2, random -- 3-objective) / best MCC (tpe -- 1-obj)
  - navigation table (knee_point / max_mcc / max_phi / max_lapse_recall) for
    both nsga2 and random fronts
  - hypervolume (Monte Carlo, nsga2 vs random) with SE

Then aggregates across the 5 seeds:
  - max-MCC-ever-seen per seed, per method (nsga2/random/tpe) -> distribution
  - hypervolume(nsga2) vs hypervolume(random): paired diff across 5 seeds,
    sign test / matched-pairs summary (n=5 is still small, report honestly,
    do not overclaim significance)
  - compares to the earlier 40-trial/2-seed/raw-feature run's numbers
    (0.091/0.087 seed0/1 pre-fix; 0.079/0.141 post-normalization@40 trials)
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from navigation import pareto_mask, hypervolume_nd, build_navigation_table

from config import RESULTS_DIR
N_SEEDS = 5
OBJ_COLS = ("mcc", "phi", "lapse_recall")

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def load(method, seed):
    return pd.read_csv(f"{RESULTS_DIR}/{method}_seed{seed}.csv")


def front_of(df):
    vals = df[list(OBJ_COLS)].values
    mask = pareto_mask(vals)
    return df[mask].reset_index(drop=True)


summary_rows = []
nav_rows = []
hv_rows = []

for seed in range(N_SEEDS):
    nsga2 = load("nsga2", seed)
    random_ = load("random", seed)
    tpe = load("tpe", seed)

    front_n = front_of(nsga2)
    front_r = front_of(random_)

    hv_n, se_n = hypervolume_nd(front_n[list(OBJ_COLS)].values)
    hv_r, se_r = hypervolume_nd(front_r[list(OBJ_COLS)].values)

    summary_rows.append({
        "seed": seed,
        "nsga2_front_size": len(front_n),
        "nsga2_max_mcc": nsga2["mcc"].max(),
        "random_front_size": len(front_r),
        "random_max_mcc": random_["mcc"].max(),
        "tpe_max_mcc": tpe["mcc"].max(),
        "hv_nsga2": hv_n, "hv_nsga2_se": se_n,
        "hv_random": hv_r, "hv_random_se": se_r,
        "hv_diff (nsga2-random)": hv_n - hv_r,
    })

    for row in build_navigation_table(front_n, OBJ_COLS):
        row = dict(row)
        row["seed"] = seed
        row["method"] = "nsga2"
        nav_rows.append(row)
    for row in build_navigation_table(front_r, OBJ_COLS):
        row = dict(row)
        row["seed"] = seed
        row["method"] = "random"
        nav_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
nav_df = pd.DataFrame(nav_rows)

print("=" * 100)
print("PER-SEED SUMMARY (100 trials/method, 3-obj MCC x Phi x lapse_recall)")
print("=" * 100)
print(summary_df.to_string(index=False))

print()
print("=" * 100)
print("NAVIGATION TABLES (all 5 seeds, nsga2 + random)")
print("=" * 100)
print(nav_df[["seed", "method", "rule", "model_family", "mcc", "phi", "lapse_recall"]]
      .sort_values(["seed", "method", "rule"]).to_string(index=False))

print()
print("=" * 100)
print("AGGREGATE ACROSS 5 SEEDS")
print("=" * 100)
print(f"NSGA-II max MCC ever seen: mean={summary_df.nsga2_max_mcc.mean():.4f} "
      f"std={summary_df.nsga2_max_mcc.std():.4f} "
      f"range=[{summary_df.nsga2_max_mcc.min():.4f}, {summary_df.nsga2_max_mcc.max():.4f}]")
print(f"Random   max MCC ever seen: mean={summary_df.random_max_mcc.mean():.4f} "
      f"std={summary_df.random_max_mcc.std():.4f} "
      f"range=[{summary_df.random_max_mcc.min():.4f}, {summary_df.random_max_mcc.max():.4f}]")
print(f"TPE      max MCC ever seen: mean={summary_df.tpe_max_mcc.mean():.4f} "
      f"std={summary_df.tpe_max_mcc.std():.4f} "
      f"range=[{summary_df.tpe_max_mcc.min():.4f}, {summary_df.tpe_max_mcc.max():.4f}]")

print()
diffs = summary_df["hv_diff (nsga2-random)"].values
n_pos = int((diffs > 0).sum())
n_neg = int((diffs < 0).sum())
print(f"Hypervolume(nsga2) - Hypervolume(random), per seed: {np.round(diffs, 5).tolist()}")
print(f"  sign: {n_pos} positive / {n_neg} negative / {len(diffs)-n_pos-n_neg} tied  (n=5 -- descriptive only, "
      f"not enough for a sign-test p<0.05 either direction)")
print(f"  mean diff = {diffs.mean():.5f}, std = {diffs.std():.5f}")

print()
print("=" * 100)
print("COMPARISON TO EARLIER 40-trial/2-seed RUNS")
print("=" * 100)
print("  Pre-fix (raw features, buggy population_size), TPE max MCC:      seed0=0.091  seed1=0.087")
print("  Post-normalization+ratio-features @ 40 trials, TPE max MCC:      seed0=0.079  seed1=0.141")
print(f"  This run (normalized+ratio features, 100 trials x 5 seeds), TPE max MCC per seed: "
      f"{summary_df.tpe_max_mcc.round(4).tolist()}")
print(f"    -> mean={summary_df.tpe_max_mcc.mean():.4f}, "
      f"vs earlier 2-seed mean of ({0.079+0.141:.4f})/2 = {(0.079+0.141)/2:.4f}")

summary_df.to_csv(f"{RESULTS_DIR}/aggregate_summary.csv", index=False)
nav_df.to_csv(f"{RESULTS_DIR}/aggregate_navigation.csv", index=False)
print()
print(f"Saved: {RESULTS_DIR}/aggregate_summary.csv, {RESULTS_DIR}/aggregate_navigation.csv")
