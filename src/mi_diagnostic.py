"""
Search-independent check that per-subject baseline normalization increases
feature-label information (paper: Results, "Explanation robustness and
feature-signal diagnostic"; Methods, "Sensor-noise stress test and mutual
information").

Computes per-feature mutual information I(X_f; Y) with scikit-learn's
k-nearest-neighbour estimator for raw and baseline-normalized features.

Usage:  python src/mi_diagnostic.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from config import RESULTS_DIR
from features import build_dataset


def mi(X, y, seed=0):
    Xf = X.fillna(X.median())
    return mutual_info_classif(Xf.values, y.values, discrete_features=False,
                               random_state=seed)


def main():
    X_raw, y, _ = build_dataset(normalize=False)
    X_norm, _, _ = build_dataset(normalize=True)
    mi_raw, mi_norm = mi(X_raw, y), mi(X_norm, y)
    out = pd.DataFrame({"feature": X_raw.columns, "mi_raw": mi_raw, "mi_normalized": mi_norm})
    out.to_csv(RESULTS_DIR / "mi_diagnostic.csv", index=False)
    print(f"mean MI raw        : {mi_raw.mean():.4f}")
    print(f"mean MI normalized : {mi_norm.mean():.4f}")
    print(f"features improved  : {(mi_norm > mi_raw).sum()} / {len(mi_raw)}")


if __name__ == "__main__":
    main()
