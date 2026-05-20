# severity-classifier/predict_severity.py

from extract_features import extract_features_for_file  # reuse same fn
from config import PROCESSED_DIR
import numpy as np
import librosa
from pathlib import Path
import joblib
import warnings

warnings.filterwarnings("ignore")


LABEL_MAP = {
    0: "healthy",
    1: "symptomatic",
    2: "COVID-19",
}


def predict_file(wav_path: str, model_type: str = "xgb"):
    wav_path = Path(wav_path)

    if not wav_path.exists():
        raise FileNotFoundError(wav_path)

    # 1) extract features same way as training
    feats = extract_features_for_file(wav_path)
    feature_vec = np.array([list(feats.values())])  # shape (1, D)

    # 2) load scaler
    scaler_path = PROCESSED_DIR / "severity_scaler.joblib"
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(feature_vec)

    # 3) load model
    model_type = model_type.lower()
    if model_type == "knn":
        model_path = PROCESSED_DIR / "knn_severity_model.joblib"
        model = joblib.load(model_path)
    else:
        from xgboost import XGBClassifier
        model_path = PROCESSED_DIR / "xgb_severity_model.json"
        model = XGBClassifier()
        model.load_model(str(model_path))

    # 4) predict
    y_pred = model.predict(X_scaled)
    cls = int(y_pred[0])
    print(f"Predicted class: {cls} ({LABEL_MAP.get(cls, 'unknown')})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, required=True,
                    help="Path to a single-cough WAV file (16k mono recommended)")
    ap.add_argument("--model", type=str, default="xgb",
                    help="'xgb' (default) or 'knn'")
    args = ap.parse_args()

    predict_file(args.wav, model_type=args.model)
