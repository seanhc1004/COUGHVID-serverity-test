#!/usr/bin/env python
"""
yamnet_baseline.py
- Zero-shot cough scoring with YAMNet
- Optional linear-probe classifier (cough vs no-cough) using YAMNet embeddings

Usage:
  # Zero-shot on a single clip:
  python yamnet_baseline.py zeroshot --wav path/to/clip.wav

  # Linear probe (train/test) on folders:
  python yamnet_baseline.py probe --cough_dir data/cough --nocough_dir data/nocough

Requires:
  pip install tensorflow==2.15 tensorflow_hub numpy scipy scikit-learn librosa soundfile
"""
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
CLASS_MAP_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"


def load_yamnet():
    model = hub.load(YAMNET_URL)
    return model


def wav_to_mono16(y, sr):
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return y.astype(np.float32), 16000


def yamnet_embeddings(model, wav_path: Path):
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    y, sr = wav_to_mono16(y, sr)
    scores, embeddings, spectrogram = model(y)
    return embeddings.numpy(), scores.numpy()


def zeroshot_cough_score(model, wav_path: Path):
    import pandas as pd
    import requests
    import io
    csv_text = requests.get(CLASS_MAP_URL, timeout=10).text
    df = pd.read_csv(io.StringIO(csv_text))
    cough_idx = int(df[df["display_name"] == "Cough"]["index"].iloc[0])
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    y, sr = wav_to_mono16(y, sr)
    scores, embeddings, spectrogram = model(y)
    probs = tf.reduce_mean(scores, axis=0).numpy()
    return float(probs[cough_idx])


def collect_embeddings(model, cough_dir: Path, nocough_dir: Path):
    X, y = [], []
    for p in cough_dir.glob("*.wav"):
        emb, _ = yamnet_embeddings(model, p)
        X.append(emb.mean(axis=0))
        y.append(1)
    for p in nocough_dir.glob("*.wav"):
        emb, _ = yamnet_embeddings(model, p)
        X.append(emb.mean(axis=0))
        y.append(0)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s0 = sub.add_parser("zeroshot")
    s0.add_argument("--wav", type=Path, required=True)

    s1 = sub.add_parser("probe")
    s1.add_argument("--cough_dir", type=Path, required=True)
    s1.add_argument("--nocough_dir", type=Path, required=True)
    s1.add_argument("--test_size", type=float, default=0.2)

    args = ap.parse_args()
    model = load_yamnet()

    if args.cmd == "zeroshot":
        score = zeroshot_cough_score(model, args.wav)
        print(f"Zero-shot cough probability (YAMNet): {score:.3f}")

    elif args.cmd == "probe":
        X, y = collect_embeddings(model, args.cough_dir, args.nocough_dir)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
        yhat = clf.predict(Xte)
        print(classification_report(yte, yhat, digits=3))
