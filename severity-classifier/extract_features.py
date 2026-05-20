# severity-classifier/extract_features.py
#
# Reads segment_severity.csv (segment_filename + severity label),
# finds each WAV segment on disk, extracts MFCC + spectral features,
# and saves everything into features_severity.csv for training.

from config import SEGMENTS_DIR, METADATA_CSV, FEATURES_CSV
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import warnings
from typing import Optional
warnings.filterwarnings("ignore")


def extract_features_for_file(filepath: Path, sr_target: int = 16000) -> dict:
    """
    Extract MFCCs + simple spectral features from one cough segment.
    Returns a dict: feature_name -> value.
    """
    # load audio as mono at fixed sampling rate
    y, sr = librosa.load(str(filepath), sr=sr_target, mono=True)

    if len(y) < 10:
        raise ValueError(f"Audio too short: {filepath}")

    # normalize amplitude
    y = y / (np.max(np.abs(y)) + 1e-9)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    # spectral features
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spec_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, roll_percent=0.85
    ).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rms = librosa.feature.rms(y=y).mean()

    feats = {}
    for i, (m, s) in enumerate(zip(mfcc_mean, mfcc_std), start=1):
        feats[f"mfcc{i}_mean"] = float(m)
        feats[f"mfcc{i}_std"] = float(s)

    feats["spec_centroid"] = float(spec_centroid)
    feats["spec_rolloff"] = float(spec_rolloff)
    feats["zcr"] = float(zcr)
    feats["rms"] = float(rms)

    return feats


def find_segment_path(seg_name: str) -> Optional[Path]:
    """
    Search SEGMENTS_DIR recursively for a file with this name.
    This is robust to per-UUID subfolders like clips/<uuid>/<uuid>_cough_01.wav
    """
    for p in SEGMENTS_DIR.rglob(seg_name):
        if p.is_file():
            return p
    return None


def main():
    print(f"Loading segment labels from: {METADATA_CSV}")
    df_meta = pd.read_csv(METADATA_CSV)

    # from make_segment_severity.py we expect: segment_filename, severity
    required_cols = {"segment_filename", "severity"}
    if not required_cols.issubset(df_meta.columns):
        raise ValueError(f"Metadata CSV must contain columns: {required_cols}")

    print(f"Found {len(df_meta)} labeled segments in metadata.")

    records = []

    for idx, row in df_meta.iterrows():
        seg_name = row["segment_filename"]
        label = int(row["severity"])  # 0=healthy, 1=symptomatic, 2=COVID-19

        seg_path = find_segment_path(seg_name)
        if seg_path is None:
            print(f"[WARN] Could not find audio file for segment: {seg_name}")
            continue

        try:
            feats = extract_features_for_file(seg_path)
        except Exception as e:
            print(f"[ERROR] Failed to extract features for {seg_path}: {e}")
            continue

        feats["segment_filename"] = seg_name
        # for training
        feats["severity_class"] = label

        records.append(feats)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} segments...")

    if not records:
        raise RuntimeError(
            "No features extracted. "
            "Check that SEGMENTS_DIR and METADATA_CSV paths are correct, "
            "and that segment_severity.csv actually has usable rows."
        )

    df_features = pd.DataFrame(records)
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(FEATURES_CSV, index=False)

    print(f"\nSaved features to: {FEATURES_CSV}")
    print(f"Total segments with features: {len(df_features)}")


if __name__ == "__main__":
    main()
