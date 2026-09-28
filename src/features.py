"""Feature engineering for the DROZY-based drowsiness model-selection study.

  - Subjects are the unit of generalization (leave-one-subject-out).
  - Sleep-deprivation transfer uses DROZY's protocol directly
    (test 1 = rested -> test 3 = ~28-30 h sleep-deprived).
  - Features are grouped by physiological mechanism (EEG scalp regions,
    EOG, EMG, ECG) for grouped SHAP faithfulness.
  - The target is PVT performance using the standard Basner & Dinges
    (2011) lapse criterion (RT >= 500 ms) rather than a quantile split:
    optimal (< 300 ms), normal (300-500 ms), lapse (>= 500 ms).

Each 10-minute (600s) PSG recording is split into 30-second epochs (20
epochs/test). Each epoch gets: EEG relative band power per channel,
EOG/EMG/ECG summary statistics, and a PVT-derived ordinal performance
label from the trials whose stimulus time falls inside that epoch window.
"""
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks, butter, filtfilt

mne.set_log_level("ERROR")

from config import DATA_DIR
PSG_DIR = DATA_DIR / "psg"
PVT_DIR = DATA_DIR / "pvt-rt"
KSS_PATH = DATA_DIR / "KSS.txt"

EPOCH_S = 30.0
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
EEG_CHANNELS = ["Fz", "Cz", "C3", "C4", "Pz"]
# 34/36 tests have exactly this 5-channel montage (Fz, Cz, C3, C4, Pz),
# matching the DROZY README. 2/36 (subjects 1 & 2, test 1 only) also
# recorded Oz; Oz is dropped for a consistent feature schema across all
# tests, which keeps feature identity fixed for grouped SHAP indexing.
# Grouped by scalp region / physiological mechanism -- used for grouped SHAP.
EEG_REGION_GROUPS = {
    "eeg_frontal": ["Fz"],
    "eeg_central": ["Cz", "C3", "C4"],
    "eeg_posterior": ["Pz"],
}
PVT_LAPSE_MS = 500.0  # Basner & Dinges (2011) standard PVT lapse criterion


def list_tests():
    """(subject:int, test:int) pairs actually present as EDF files."""
    tests = []
    for f in sorted(PSG_DIR.glob("*.edf")):
        m = re.match(r"(\d+)-(\d+)\.edf", f.name)
        if m:
            tests.append((int(m.group(1)), int(m.group(2))))
    return tests


def load_kss():
    """KSS.txt -> DataFrame(subject, test, kss). 14 lines x 3 cols."""
    rows = []
    with open(KSS_PATH) as f:
        for subj, line in enumerate(f, start=1):
            vals = [int(x) for x in line.split()]
            for test, kss in enumerate(vals, start=1):
                rows.append({"subject": subj, "test": test, "kss": kss})
    return pd.DataFrame(rows)


def _bandpower_features(sig, sfreq, prefix):
    """Welch PSD relative band power for one channel's epoch signal."""
    freqs, psd = welch(sig, fs=sfreq, nperseg=min(len(sig), int(sfreq * 4)))
    total = np.trapezoid(psd[(freqs >= 0.5) & (freqs <= 45)],
                          freqs[(freqs >= 0.5) & (freqs <= 45)])
    out = {}
    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        p = np.trapezoid(psd[mask], freqs[mask]) if mask.any() else 0.0
        out[f"{prefix}_{band}"] = p / total if total > 0 else np.nan
    return out


def _ecg_features(sig, sfreq):
    """Simple peak-detected heart rate + RR-interval SDNN from raw ECG."""
    try:
        b, a = butter(2, [5 / (sfreq / 2), 25 / (sfreq / 2)], btype="band")
        filt = filtfilt(b, a, sig)
        peaks, _ = find_peaks(filt, distance=int(sfreq * 0.4),
                               height=np.std(filt) * 1.2)
        if len(peaks) < 3:
            return {"ecg_hr_bpm": np.nan, "ecg_rr_sdnn_ms": np.nan}
        rr = np.diff(peaks) / sfreq * 1000.0  # ms
        hr = 60000.0 / np.mean(rr)
        return {"ecg_hr_bpm": float(hr), "ecg_rr_sdnn_ms": float(np.std(rr))}
    except Exception:
        return {"ecg_hr_bpm": np.nan, "ecg_rr_sdnn_ms": np.nan}


def _epoch_features(raw, start_s, end_s):
    sfreq = raw.info["sfreq"]
    feats = {}
    for ch in EEG_CHANNELS:
        sig = raw.get_data(picks=[ch], tmin=start_s, tmax=end_s)[0]
        bp = _bandpower_features(sig, sfreq, prefix=f"eeg_{ch}")
        feats.update(bp)
        # Literature-validated drowsiness ratios, not just raw band
        # power: theta/alpha and (theta+alpha)/beta are the most-cited
        # single EEG drowsiness markers (rising theta/alpha and rising
        # slow-to-fast ratio track sleepiness). Raw relative band power
        # alone leaves the model to rediscover these from scratch.
        theta, alpha, beta = bp[f"eeg_{ch}_theta"], bp[f"eeg_{ch}_alpha"], bp[f"eeg_{ch}_beta"]
        feats[f"eeg_{ch}_theta_alpha_ratio"] = theta / alpha if alpha > 1e-9 else np.nan
        feats[f"eeg_{ch}_slow_fast_ratio"] = (theta + alpha) / beta if beta > 1e-9 else np.nan
    for ch, prefix in [("EOG-V", "eogv"), ("EOG-H", "eogh")]:
        sig = raw.get_data(picks=[ch], tmin=start_s, tmax=end_s)[0]
        feats[f"{prefix}_std"] = float(np.std(sig))
        # zero-crossing rate as a cheap blink/saccade-rate proxy
        feats[f"{prefix}_zcr"] = float(np.mean(np.diff(np.sign(sig)) != 0))
    emg = raw.get_data(picks=["EMG"], tmin=start_s, tmax=end_s)[0]
    feats["emg_rms"] = float(np.sqrt(np.mean(emg ** 2)))
    feats["emg_std"] = float(np.std(emg))
    ecg = raw.get_data(picks=["ECG"], tmin=start_s, tmax=end_s)[0]
    feats.update(_ecg_features(ecg, sfreq))
    return feats


def _pvt_trials(subject, test):
    """(stimulus_time, response_time) pairs, seconds relative to the first
    (start-marker) line in the file."""
    path = PVT_DIR / f"{subject}-{test}.csv"
    if not path.exists():
        return None
    lines = path.read_text().strip().splitlines()
    if not lines:
        return None

    def parse_ts(s):
        # "2014-11-26_10.08.39.274" -> seconds-of-day float
        date, clock = s.split("_")
        h, m, sec, ms = clock.split(".")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0

    t0 = parse_ts(lines[0])
    trials = []
    for line in lines[1:]:
        if ";" not in line:
            continue
        stim, resp = line.split(";")
        stim_t = parse_ts(stim) - t0
        resp_t = parse_ts(resp) - t0
        rt_ms = (resp_t - stim_t) * 1000.0
        if 0 < rt_ms < 30000:  # guard against parsing/wrap artifacts
            trials.append((stim_t, rt_ms))
    return trials


def _lapse_class(median_rt_ms):
    """Ordinal PVT performance class from the Basner & Dinges lapse
    criterion (RT > 500ms = lapse), not an arbitrary quantile split.
    A 4-class version (splitting lapses at 1000ms) was tried first but
    left only 2 epochs in the top class across the whole dataset --
    unusable for CV -- so lapses are kept as one class, matching the
    standard PVT alert/lapse binary distinction, with "optimal" broken
    out as a third, faster-than-typical band."""
    if median_rt_ms < 300:
        return 0  # optimal
    elif median_rt_ms < PVT_LAPSE_MS:
        return 1  # normal
    else:
        return 2  # lapse (RT >= 500ms, Basner & Dinges 2011)


def build_dataset(min_trials_per_epoch=2, normalize=True):
    """Returns (X: DataFrame, y: Series[int ordinal class], meta: DataFrame
    with subject/test/epoch_index/median_rt_ms/n_trials)."""
    kss = load_kss()
    rows, targets, metas = [], [], []
    skipped_epochs = 0

    for subject, test in list_tests():
        edf_path = PSG_DIR / f"{subject}-{test}.edf"
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        duration = raw.n_times / raw.info["sfreq"]
        trials = _pvt_trials(subject, test)
        if trials is None:
            continue
        n_epochs = int(duration // EPOCH_S)

        for ei in range(n_epochs):
            start_s, end_s = ei * EPOCH_S, (ei + 1) * EPOCH_S
            epoch_trials = [rt for (stim, rt) in trials if start_s <= stim < end_s]
            if len(epoch_trials) < min_trials_per_epoch:
                skipped_epochs += 1
                continue
            median_rt = float(np.median(epoch_trials))

            feats = _epoch_features(raw, start_s, end_s)
            rows.append(feats)
            targets.append(_lapse_class(median_rt))
            metas.append({
                "subject": subject, "test": test, "epoch_index": ei,
                "median_rt_ms": median_rt, "n_trials": len(epoch_trials),
                "kss": kss[(kss.subject == subject) & (kss.test == test)]["kss"].iloc[0],
            })

    X = pd.DataFrame(rows)
    y = pd.Series(targets, name="lapse_class")
    meta = pd.DataFrame(metas)
    print(f"Built {len(X)} epoch-rows from {len(list_tests())} tests "
          f"({skipped_epochs} epochs skipped for <{min_trials_per_epoch} PVT trials)")

    if normalize:
        X = baseline_normalize(X, meta)
    return X, y, meta


def baseline_normalize(X, meta, baseline_n_epochs=5):
    """Re-express every feature relative to each subject's OWN early
    baseline, replacing absolute values.

    Pooling raw band power / EOG / EMG / ECG values across 14 different
    people confounds individual anatomy (skull thickness, electrode
    impedance, resting cortical activity) with the actual drowsiness
    signal -- a well-known problem in cross-subject EEG/BCI modeling.
    Centering each subject's epochs on their own baseline (mean of their
    first `baseline_n_epochs` epochs from their EARLIEST available test --
    not hardcoded to test 1, since tests 7-1 and 9-1 are missing from the
    raw data) removes the between-subject offset and leaves only the
    within-subject deviation, which is what should actually carry the
    lapse signal.

    Deliberately REPLACES raw values rather than adding baseline-relative
    columns alongside them: keeping both would let a model fall back on
    absolute-value shortcuts, defeating the point of this fix."""
    X = X.reset_index(drop=True)
    meta = meta.reset_index(drop=True)
    Xn = X.copy()
    for subject, sub_meta in meta.groupby("subject"):
        earliest_test = sub_meta["test"].min()
        baseline_rows = sub_meta[sub_meta["test"] == earliest_test].sort_values("epoch_index").index[:baseline_n_epochs]
        baseline_mean = X.loc[baseline_rows].mean()
        subj_rows = sub_meta.index
        Xn.loc[subj_rows] = X.loc[subj_rows] - baseline_mean
    return Xn


FEATURE_GROUPS_TEMPLATE = None  # built after knowing X's exact columns


def feature_groups(columns):
    """Physiologically-grouped feature blocks for grouped SHAP faithfulness."""
    groups = {}
    for gname, chans in EEG_REGION_GROUPS.items():
        cols = [c for c in columns if any(c.startswith(f"eeg_{ch}_") for ch in chans)]
        groups[gname] = cols
    groups["eog"] = [c for c in columns if c.startswith("eogv_") or c.startswith("eogh_")]
    groups["emg"] = [c for c in columns if c.startswith("emg_")]
    groups["ecg"] = [c for c in columns if c.startswith("ecg_")]
    return groups


# Columns that are real sensor/derived-signal readings (candidates for the
# sensor-noise stress test), vs. context/engineered columns that are never
# noise-injected. All features in this dataset are sensor-derived.
SENSOR_DERIVED_PREFIXES = ("eeg_", "eogv_", "eogh_", "emg_", "ecg_")

if __name__ == "__main__":
    X, y, meta = build_dataset()
    print(X.shape, y.value_counts().sort_index().to_dict())
    print(meta.groupby("test")["subject"].nunique())
    groups = feature_groups(X.columns.tolist())
    for g, cols in groups.items():
        print(g, len(cols), cols[:3])
