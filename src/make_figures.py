"""
Generate the three paper figures from results/*.csv, following the
dataviz skill's static-figure adaptation of its method (form -> validated
categorical color -> thin marks -> direct labels -> legend for >=2 series).
Palette: reference instance in the dataviz skill (light mode), validated
via scripts/validate_palette.js for the 2- and 3-slot combinations used
here (both PASS, one WARN on aqua/white contrast at slot 3 -- mitigated
below with direct value labels, matching the skill's "relief" rule).
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from navigation import pareto_mask, build_navigation_table

from config import RESULTS_DIR as RESULTS, FIG_DIR as OUT
OBJ_COLS = ("mcc", "phi", "lapse_recall")

# ---- palette (dataviz skill reference instance, light mode) --------------
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.facecolor": SURFACE,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def style_axes(ax, y_zero_line=False):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
        ax.spines[spine].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    if y_zero_line:
        ax.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=1)


# ============================================================ Figure 1 ===
# Pareto front shape + navigation-rule points (marker shape = rule, not
# color -- keeps this a 1-hue chart since "rule" is a role/identity
# encoded better by shape here, not a magnitude comparison).
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

# Hollow markers of decreasing size, so a model picked by several rules
# shows as concentric outlines instead of one marker hiding another.
MARKERS = {"knee_point": ("*", 230, "knee point"),
           "max_mcc": ("o", 120, "max MCC"),
           "max_phi": ("^", 70, "max $\\Phi$"),
           "max_lapse_recall": ("s", 28, "max lapse recall")}

all_front_pts = []
all_nav_rows = []
for seed in range(5):
    df = pd.read_csv(f"{RESULTS}/nsga2_seed{seed}.csv")
    vals = df[list(OBJ_COLS)].values
    front = df[pareto_mask(vals)].reset_index(drop=True)
    all_front_pts.append(front)
    for row in build_navigation_table(front, OBJ_COLS):
        row["seed"] = seed
        all_nav_rows.append(row)
front_all = pd.concat(all_front_pts, ignore_index=True)
nav_all = pd.DataFrame(all_nav_rows)

for ax, ycol, ylabel in [(axes[0], "phi", "Explanation faithfulness ($\\Phi$)"),
                          (axes[1], "lapse_recall", "Lapse recall")]:
    ax.scatter(front_all["mcc"], front_all[ycol], s=12, color=INK_MUTED,
               alpha=0.45, linewidths=0, zorder=2, label="Pareto-front points\n(5 seeds combined)")
    for rule, (marker, size, label) in MARKERS.items():
        sub = nav_all[nav_all["rule"] == rule]
        ax.scatter(sub["mcc"], sub[ycol], s=size, marker=marker,
                   facecolor="none", edgecolor=BLUE, linewidths=1.3,
                   zorder=3, label=label)
    ax.set_xlabel("MCC")
    ax.set_ylabel(ylabel)
    style_axes(ax)

handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, -0.06), fontsize=7.5, labelcolor=INK_SECONDARY)
fig.suptitle("NSGA-II Pareto fronts and navigation-rule selections (5 seeds)",
             fontsize=10, color=INK, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_pareto_fronts.pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("wrote fig_pareto_fronts.pdf")

# ============================================================ Figure 2 ===
# Hypervolume comparison, NSGA-II vs random, per seed -- grouped bars.
summ = pd.read_csv(f"{RESULTS}/aggregate_summary.csv")
seeds = summ["seed"].values
hv_n = summ["hv_nsga2"].values
hv_r = summ["hv_random"].values

fig, ax = plt.subplots(figsize=(5.4, 3.3))
x = np.arange(len(seeds))
w = 0.36
b1 = ax.bar(x - w / 2, hv_n, width=w, color=BLUE, label="NSGA-II", zorder=2)
b2 = ax.bar(x + w / 2, hv_r, width=w, color=ORANGE, label="Random search", zorder=2)
for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", (rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5, color=INK_SECONDARY)
ax.set_xticks(x)
ax.set_xticklabels([f"seed {s}" for s in seeds])
ax.set_ylabel("Dominated hypervolume (Monte Carlo)")
ax.set_ylim(0, max(hv_n.max(), hv_r.max()) * 1.12)
ax.set_title("NSGA-II dominates random search on 4 of 5 seeds",
              fontsize=10, color=INK, pad=22)
style_axes(ax)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          ncol=2, fontsize=8, labelcolor=INK_SECONDARY, borderaxespad=0.2)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_hypervolume.pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("wrote fig_hypervolume.pdf")

# ============================================================ Figure 3 ===
# THE central result: decomposition of the apparent person-level MCC.
# Horizontal bars (long protocol labels), one row per protocol, colored by
# protocol family; black diamond = the same protocol run on pure-noise
# features (its empirical chance level), where computed.
te = pd.read_csv(f"{RESULTS}/transfer_eval.csv")
gd = pd.read_csv(f"{RESULTS}/gap_decomposition.csv")
ls = pd.read_csv(f"{RESULTS}/label_shift_eval.csv")
nb = pd.read_csv(f"{RESULTS}/null_baseline.csv")
UNSTRAT_NULL = -0.002  # null for unstratified 3-fold (15 configs x 3 noise seeds), see paper
rows = [
    ("Search score of selected model\n(stratified subject-grouped 3-fold)", te["search_mcc"], BLUE, None),
    ("Same models, fresh stratified folds\n(10 new fold seeds)", gd["fresh3fold_mcc"], BLUE, nb["fresh3fold_mcc"].mean()),
    ("Unstratified subject-grouped 3-fold\n(random subject partitions)", ls["unstrat3fold_mcc"], ORANGE, UNSTRAT_NULL),
    ("Leave-one-subject-out\n(per-subject mean)", te["loso_mcc_mean"], ORANGE, nb["loso_per_subject_mcc"].mean()),
    ("LOSO + CORAL alignment", te["loso_coral_mcc_mean"], YELLOW, None),
    ("Sleep-deprivation transfer\n(rested $\\to$ sleep-deprived, seen subjects)", te["sdt_test3_mcc"], AQUA, nb["sdt_mcc"].mean()),
]
fig, ax = plt.subplots(figsize=(6.8, 4.6))
y = np.arange(len(rows))[::-1]
for yi, (lab, s, col, null) in zip(y, rows):
    m, sd = s.mean(), s.std()
    ax.barh(yi, m, xerr=sd, height=0.62, color=col, zorder=2, capsize=3,
            error_kw={"ecolor": INK_SECONDARY, "linewidth": 1.0})
    xe = max(m + sd, 0.0)
    txt = f"{m:+.3f}" + (f"   (null {null:+.3f})" if null is not None else "")
    ax.annotate(txt, (xe, yi), xytext=(8, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8.5, color=INK)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=7.8)
ax.set_xlabel("MCC (mean $\\pm$ std over 20 navigation-rule configurations)")
ax.set_xlim(-0.08, 0.36)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(BASELINE)
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(length=0)
ax.axvline(0, color=INK_MUTED, linewidth=1.0, zorder=1)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, YELLOW, AQUA)]
ax.legend(handles, ["label-stratified subject hold-out", "unstratified subject hold-out",
                    "unsupervised adaptation", "state transfer (seen subjects)"],
          frameon=False, fontsize=7.5, loc="upper center", ncol=2,
          bbox_to_anchor=(0.4, -0.13), labelcolor=INK_SECONDARY)
ax.set_title("Apparent person-level skill depends on label-stratified fold assignment",
             fontsize=9.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_generalization_gap.pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("wrote fig_generalization_gap.pdf")
