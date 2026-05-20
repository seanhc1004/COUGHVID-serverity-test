# severity-classifier/make_segment_severity.py

import pandas as pd
from pathlib import Path

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_CSV = PROJECT_ROOT / "segments" / "segments_manifest.csv"
COUGHVID_META_CSV = PROJECT_ROOT / "data" / "metadata" / "metadata_compiled.csv"
OUT_CSV = PROJECT_ROOT / "data" / "metadata" / "segment_severity.csv"


def encode_status(status: str):
    """
    Map COUGHVID 'status' to severity class:
      0 = healthy
      1 = symptomatic
      2 = COVID-19
    Return None if not usable.
    """
    if not isinstance(status, str):
        return None

    s = status.strip().lower()

    if s == "healthy":
        return 0
    elif s == "symptomatic":
        return 1
    elif s in {"covid-19", "covid19", "covid"}:
        return 2
    else:
        return None   # ignore any other categories


def main():
    print(f"Loading segments manifest from: {MANIFEST_CSV}")
    df_seg = pd.read_csv(MANIFEST_CSV)

    if df_seg.empty:
        raise RuntimeError(
            "segments_manifest.csv is empty. Run segmentation first.")

    print(f"Loaded {len(df_seg)} segment rows.")

    print(f"Loading COUGHVID metadata from: {COUGHVID_META_CSV}")
    df_meta = pd.read_csv(COUGHVID_META_CSV)

    # Confirm required columns
    if "uuid" not in df_meta.columns:
        raise ValueError("metadata_compiled.csv must contain a 'uuid' column.")

    if "status" not in df_meta.columns:
        raise ValueError(
            "metadata_compiled.csv must contain a 'status' column.")

    # Build uuid -> class mapping
    uuid_to_class = {}
    for _, row in df_meta.iterrows():
        uid = str(row["uuid"]).strip()
        cls = encode_status(row["status"])
        if cls is not None:
            uuid_to_class[uid] = cls

    print(f"Usable labeled recordings: {len(uuid_to_class)}")

    records = []

    for idx, row in df_seg.iterrows():
        orig_path = str(row["file"])      # original wav16 path
        seg_path = str(row["out_wav"])    # segmented file path

        # UUID = stem of original filename
        orig_uuid = Path(orig_path).stem.strip()

        if orig_uuid not in uuid_to_class:
            print(
                f"[WARN] No valid label for UUID={orig_uuid}, skipping {seg_path}")
            continue

        label_class = uuid_to_class[orig_uuid]
        seg_filename = Path(seg_path).name

        records.append({
            "segment_filename": seg_filename,
            "severity": label_class
        })

    if not records:
        raise RuntimeError(
            "No segments matched valid labels. Check metadata and filenames.")

    df_out = pd.DataFrame(records)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)

    print(f"\nSaved labeled segment CSV to: {OUT_CSV}")
    print(f"Total labeled segments: {len(df_out)}")


if __name__ == "__main__":
    main()
