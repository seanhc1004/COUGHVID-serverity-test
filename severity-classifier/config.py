# severity_classifier/config.py

from pathlib import Path

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
SEGMENTS_DIR = PROJECT_ROOT / "segments"
METADATA_CSV = PROJECT_ROOT / "data" / "processed" / "segment_features.csv"

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# FEATURES_CSV = PROCESSED_DIR / "features_severity.csv"
FEATURES_CSV = PROCESSED_DIR / "features_yamnet.csv"

# ---- Severity binning ----
# Continuous severity (0–100) -> 3 classes: 0=low, 1=moderate, 2=high
LOW_MAX = 0.5
MID_MAX = 1.5
