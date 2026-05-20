# severity_classifier/train_severity_models.py

from config import FEATURES_CSV, PROCESSED_DIR
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# XGBoost is optional but recommended
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARNING] xgboost is not installed. Only kNN will be trained.")


def load_dataset():

    print(f"Loading features from {FEATURES_CSV}")
    df = pd.read_csv(FEATURES_CSV)
    print("Columns:", len(df.columns))
    print("Head:")
    print(df.head())

    if "severity_class" in df.columns:
        target_col = "severity_class"
    elif "label" in df.columns:
        target_col = "label"
    else:
        raise ValueError(
            f"Could not find label column in {FEATURES_CSV}. "
            "Expected one of: 'severity_class' or 'label'."
        )

    print("Class counts before balancing:")
    print(df[target_col].value_counts())

    N = 200  # max samples per class (tune as you like)

    dfs = []
    for cls in sorted(df[target_col].unique()):
        df_cls = df[df[target_col] == cls]
        if len(df_cls) > N:
            df_cls = df_cls.sample(n=N, random_state=42)
        dfs.append(df_cls)

    # shuffle
    df_balanced = pd.concat(dfs, axis=0).sample(
        frac=1.0, random_state=42
    )

    print("Class counts after balancing:")
    print(df_balanced[target_col].value_counts())

    non_features = {"segment_filename",
                    "severity_score", "severity_class", "label"}
    feature_cols = [c for c in df_balanced.columns if c not in non_features]

    X = df_balanced[feature_cols].values
    y = df_balanced[target_col].values

    return X, y, feature_cols


def train_and_evaluate_knn(X_train, X_test, y_train, y_test):
    print("\n=== Training kNN (baseline) ===")
    knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski")
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"kNN Accuracy: {acc:.3f}")
    print("kNN Classification Report:")
    print(classification_report(y_test, y_pred))
    print("kNN Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(knn, PROCESSED_DIR / "knn_severity_model.joblib")
    print(f"Saved kNN model to {PROCESSED_DIR / 'knn_severity_model.joblib'}")


def train_and_evaluate_xgb(X_train, X_test, y_train, y_test, num_classes: int):
    if not HAS_XGB:
        return

    print("\n=== Training XGBoost (final model) ===")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist",
    )

    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"XGBoost Accuracy: {acc:.3f}")
    print("XGBoost Classification Report:")
    print(classification_report(y_test, y_pred))
    print("XGBoost Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    xgb.save_model(str(PROCESSED_DIR / "xgb_severity_model.json"))
    print(
        f"Saved XGBoost model to {PROCESSED_DIR / 'xgb_severity_model.json'}")


def main():
    X, y, feature_cols = load_dataset()
    num_classes = len(np.unique(y))
    print(
        f"Number of samples: {len(y)}, features: {len(feature_cols)}, classes: {num_classes}")

    # Train:80% /test split:20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (important for kNN, helpful for XGB too)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for inference
    joblib.dump(scaler, PROCESSED_DIR / "severity_scaler.joblib")
    print(f"Saved scaler to {PROCESSED_DIR / 'severity_scaler.joblib'}")

    # Train models
    train_and_evaluate_knn(X_train_scaled, X_test_scaled, y_train, y_test)
    train_and_evaluate_xgb(X_train_scaled, X_test_scaled,
                           y_train, y_test, num_classes)


if __name__ == "__main__":
    main()
