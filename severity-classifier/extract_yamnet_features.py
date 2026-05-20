import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_CSV = PROJECT_ROOT / "data/metadata/segment_severity.csv"
WAV_ROOT = PROJECT_ROOT / "segments/clips"
OUT_CSV = PROJECT_ROOT / "data/processed/features_yamnet.csv"

YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(YAMNET_MODEL_HANDLE)


def load_wav_16k_mono(wav_path: Path) -> tf.Tensor:
    file_contents = tf.io.read_file(str(wav_path))
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sr = int(sample_rate.numpy())
    if sr != 16000:
        print(f"Warning: {wav_path} has sample rate {sr}, not 16000")

    return wav


def yamnet_embedding(wav: tf.Tensor) -> np.ndarray:
    scores, embeddings, spectrogram = yamnet_model(wav)
    embedding_mean = tf.reduce_mean(embeddings, axis=0)
    return embedding_mean.numpy()


def main():
    meta = pd.read_csv(METADATA_CSV)
    meta["segment_filename"] = meta["segment_filename"].str.strip()

    rows = []
    for idx, row in meta.iterrows():
        fname = row["segment_filename"]
        label = row["severity"]

        # subfolder = part before "_cough"
        folder = fname.split("_cough")[0]
        wav_path = WAV_ROOT / folder / fname

        if not wav_path.exists():
            print(f"Missing file: {wav_path}")
            continue

        wav = load_wav_16k_mono(wav_path)
        emb = yamnet_embedding(wav)
        rows.append(np.concatenate([[label], emb]))

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{len(meta)}")

    if not rows:
        raise RuntimeError(
            "No features were extracted – check paths/filenames.")

    rows = np.stack(rows, axis=0)
    n_features = rows.shape[1] - 1
    columns = ["label"] + [f"f{i}" for i in range(n_features)]

    df = pd.DataFrame(rows, columns=columns)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved YAMNet features to {OUT_CSV}")


if __name__ == "__main__":
    main()
