"""
EEG / physiological feature analysis figure (paper Fig. 1).

  a  Mean relative Welch PSD at Cz, rested (session 1) vs sleep-deprived
     (session 3), for subjects with both sessions.
  b  Per-subject median relative theta power at Cz, session 1 -> 3.
  c  Baseline-normalized Cz theta/alpha ratio by PVT class.
  d  Per-feature mutual information with the label, raw vs baseline-normalized
     (top 12 features by normalized MI; reads results/mi_diagnostic.csv).

Prints the statistics quoted in the paper. Usage:  python src/make_eeg_figure.py
"""
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from scipy.signal import welch

from config import RESULTS_DIR, FIG_DIR
from features import build_dataset, list_tests, PSG_DIR, EPOCH_S, BANDS

BLUE, ORANGE = "#2a78d6", "#eb6834"
CLASS_COLORS = ["#86b6ef", "#2a78d6", "#104281"]   # ordinal blue ramp (validated)
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CHANNEL = "Cz"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
    "axes.facecolor": SURFACE, "figure.facecolor": "white",
})


def style(ax, grid_axis="y"):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASELINE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def panel_label(ax, s):
    ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


def relative_psd(subject, test):
    raw = mne.io.read_raw_edf(PSG_DIR / f"{subject}-{test}.edf", preload=True, verbose=False)
    sf = raw.info["sfreq"]
    sig = raw.get_data(picks=[CHANNEL])[0]
    n_ep = int(len(sig) // (sf * EPOCH_S))
    psds = []
    for k in range(n_ep):
        seg = sig[int(k * sf * EPOCH_S): int((k + 1) * sf * EPOCH_S)]
        f, p = welch(seg, fs=sf, nperseg=int(sf * 4))
        band = (f >= 0.5) & (f <= 45)
        psds.append(p / np.trapezoid(p[band], f[band]))
    return f, np.mean(psds, axis=0)


def main():
    X_raw, y, meta = build_dataset(normalize=False)
    X_norm, _, _ = build_dataset(normalize=True)
    tests = set(list_tests())
    both = sorted(s for s in {s for s, _ in tests} if (s, 1) in tests and (s, 3) in tests)

    # ---- a: PSD, per-subject mean then across subjects
    curves = {1: [], 3: []}
    for s in both:
        for t in (1, 3):
            f, p = relative_psd(s, t)
            curves[t].append(p)
    fmask = (f >= 1) & (f <= 30)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    ax = axes[0, 0]
    for lo, hi in [BANDS["theta"], BANDS["beta"]]:
        ax.axvspan(lo, hi, color=GRID, alpha=0.45, zorder=0, linewidth=0)
    for name, (lo, hi) in [("δ", BANDS["delta"]), ("θ", BANDS["theta"]),
                           ("α", BANDS["alpha"]), ("β", (13, 30))]:
        ax.text((max(lo, 1) + hi) / 2, 1.02, name, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
    for t, col, lab in [(1, BLUE, "Rested (session 1)"), (3, ORANGE, "Sleep-deprived (session 3)")]:
        c = np.array(curves[t])
        m, se = c.mean(0), c.std(0, ddof=1) / np.sqrt(len(c))
        ax.fill_between(f[fmask], (m - se)[fmask], (m + se)[fmask], color=col, alpha=0.18, linewidth=0)
        ax.plot(f[fmask], m[fmask], color=col, linewidth=1.6, label=lab)
    ax.set_yscale("log")
    ax.set_xlim(1, 30)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"Relative power at {CHANNEL} (1/Hz)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=INK_SECONDARY)
    style(ax)
    panel_label(ax, "a")

    # ---- b: per-subject theta at Cz, session 1 -> 3
    ax = axes[0, 1]
    feat = f"eeg_{CHANNEL}_theta"
    d = pd.DataFrame({"v": X_raw[feat], "s": meta["subject"], "t": meta["test"]})
    p = d.groupby(["s", "t"])["v"].median().unstack().loc[both]
    for s_, row in p.iterrows():
        up = row[3] > row[1]
        ax.plot([0, 1], [row[1], row[3]], color=ORANGE if up else INK_MUTED,
                linewidth=1.2, alpha=0.9, zorder=2)
        ax.scatter([0, 1], [row[1], row[3]], s=18, color=ORANGE if up else INK_MUTED,
                   edgecolor="white", linewidth=0.8, zorder=3)
    n_up = int((p[3] > p[1]).sum())
    w_p = stats.wilcoxon(p[3], p[1]).pvalue
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Rested\n(session 1)", "Sleep-deprived\n(session 3)"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel(f"Median relative θ power at {CHANNEL}")
    ymin, ymax = np.nanmin(p[[1, 3]].values), np.nanmax(p[[1, 3]].values)
    ax.set_ylim(ymin - 0.01, ymax + 0.25 * (ymax - ymin))
    ax.text(0.5, 0.97, f"{n_up} of {len(p)} subjects increase\nWilcoxon p = {w_p:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=INK_SECONDARY)
    style(ax)
    panel_label(ax, "b")

    # ---- c: normalized theta/alpha by class
    ax = axes[1, 0]
    feat = f"eeg_{CHANNEL}_theta_alpha_ratio"
    groups = [X_norm[feat][y == c].dropna().values for c in range(3)]
    lo, hi = np.percentile(np.concatenate(groups), [2, 98])
    bp = ax.boxplot(groups, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops={"color": "white", "linewidth": 1.6},
                    whiskerprops={"color": INK_SECONDARY}, capprops={"color": INK_SECONDARY})
    for patch, col in zip(bp["boxes"], CLASS_COLORS):
        patch.set_facecolor(col)
        patch.set_edgecolor(col)
    rng = np.random.RandomState(0)
    for i, g in enumerate(groups):
        gg = g[(g >= lo) & (g <= hi)]
        ax.scatter(i + 1 + rng.uniform(-0.18, 0.18, len(gg)), gg, s=4, color=INK_MUTED,
                   alpha=0.35, linewidth=0, zorder=1)
    kw_p = stats.kruskal(*groups).pvalue
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"Optimal\n(n={len(groups[0])})", f"Normal\n(n={len(groups[1])})",
                        f"Lapse\n(n={len(groups[2])})"])
    ax.set_ylim(lo, hi)
    ax.axhline(0, color=INK_MUTED, linewidth=0.8, zorder=1)
    ax.set_ylabel(f"θ/α ratio at {CHANNEL}\n(change from own baseline)")
    ax.text(0.03, 0.97, f"Kruskal–Wallis p = {kw_p:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5, color=INK_SECONDARY)
    style(ax)
    panel_label(ax, "c")

    # ---- d: MI raw vs normalized, top 12
    ax = axes[1, 1]
    mi = pd.read_csv(RESULTS_DIR / "mi_diagnostic.csv")
    top = mi.sort_values("mi_normalized", ascending=False).head(12).iloc[::-1]

    def pretty(n):
        n = n.replace("eeg_", "EEG ").replace("eogv_", "EOG-V ").replace("eogh_", "EOG-H ")
        n = n.replace("emg_", "EMG ").replace("ecg_", "ECG ")
        n = n.replace("theta_alpha_ratio", "θ/α").replace("slow_fast_ratio", "(θ+α)/β")
        n = n.replace("_", " ").replace("hr bpm", "heart rate").replace("rr sdnn ms", "SDNN")
        for a, b in [("theta", "θ"), ("alpha", "α"), ("beta", "β"), ("gamma", "γ"), ("delta", "δ")]:
            n = n.replace(a, b)
        return n.replace("rms", "RMS").replace("std", "s.d.").replace("zcr", "zero-crossings")

    yy = np.arange(len(top))
    for i, (_, r) in enumerate(top.iterrows()):
        ax.plot([r.mi_raw, r.mi_normalized], [i, i], color=BASELINE, linewidth=1.4, zorder=1)
    ax.scatter(top.mi_raw, yy, s=24, facecolor="white", edgecolor=INK_MUTED, linewidth=1.2,
               zorder=2, label="Raw")
    ax.scatter(top.mi_normalized, yy, s=24, color=BLUE, edgecolor="white", linewidth=0.8,
               zorder=3, label="Baseline-normalized")
    ax.set_yticks(yy)
    ax.set_yticklabels([pretty(n) for n in top.feature], fontsize=7.2)
    ax.set_xlabel("Mutual information with PVT class (nats)")
    ax.legend(frameon=False, fontsize=7.2, loc="lower center", bbox_to_anchor=(0.45, 1.0),
              ncol=2, labelcolor=INK_SECONDARY, borderaxespad=0.1)
    style(ax, grid_axis="x")
    panel_label(ax, "d")

    fig.tight_layout(h_pad=2.2, w_pad=1.6)
    out = FIG_DIR / "fig_eeg_features.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig_eeg_features.png", bbox_inches="tight", dpi=200)
    print(f"wrote {out}")
    print(f"subjects with sessions 1 and 3: {len(both)}")
    print(f"Cz theta: {n_up}/{len(p)} increase, Wilcoxon p={w_p:.4f}, "
          f"median {p[1].median():.3f} -> {p[3].median():.3f}")
    print(f"Cz theta/alpha (normalized) medians by class: "
          f"{[round(float(np.median(g)), 3) for g in groups]}, Kruskal-Wallis p={kw_p:.4f}")


if __name__ == "__main__":
    main()
