from pathlib import Path
WAV_ROOT = Path("segments/clips")
fname = "00039425-7f3a-42aa-ac13-834aaa2b6b92_cough_01.wav"
folder = fname.split("_cough")[0]
path = WAV_ROOT / folder / fname
print(path, path.exists())


def load_dataset():
    print(f"Loading features from {FEATURES_CSV}")
    df = pd.read_csv(FEATURES_CSV)
    print("Columns:", len(df.columns))
    print("Head:")
    print(df.head())

    # ----- OPTIONAL: BALANCE CLASSES BY DOWNSAMPLING -----
    # Count how many samples per class
    print("Class counts before balancing:")
    print(df["severity_class"].value_counts())

    # choose how many per class to keep
    N = 200  # or min count across classes

    dfs = []
    for cls in sorted(df["severity_class"].unique()):
        df_cls = df[df["severity_class"] == cls]
        # if class has fewer than N samples, keep all
        if len(df_cls) > N:
            df_cls = df_cls.sample(n=N, random_state=42)
        dfs.append(df_cls)

    df_balanced = pd.concat(dfs, axis=0).sample(
        frac=1.0, random_state=42)  # shuffle

    print("Class counts after balancing:")
    print(df_balanced["severity_class"].value_counts())
    # ------------------------------------------------------

    # Drop non-feature columns
    non_features = {"segment_filename", "severity_score", "severity_class"}
    feature_cols = [c for c in df_balanced.columns if c not in non_features]

    X = df_balanced[feature_cols].values
    y = df_balanced["severity_class"].values

    return X, y, feature_cols
