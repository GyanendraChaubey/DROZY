"""Central paths. Override with environment variables if your layout differs:

    DROZY_DATA_DIR     folder containing psg/, pvt-rt/, timestamps/, KSS.txt
                       (default: <repo>/data/DROZY)
    DROZY_RESULTS_DIR  where CSV results are read/written (default: <repo>/results)
    DROZY_FIG_DIR      where figures are written (default: <repo>/figures)
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DROZY_DATA_DIR", REPO_ROOT / "data" / "DROZY"))
RESULTS_DIR = Path(os.environ.get("DROZY_RESULTS_DIR", REPO_ROOT / "results"))
FIG_DIR = Path(os.environ.get("DROZY_FIG_DIR", REPO_ROOT / "figures"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
