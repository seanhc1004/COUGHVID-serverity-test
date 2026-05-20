#!/usr/bin/env python
"""
convert_and_preview.py
- Batch convert audio to 16 kHz mono WAV
- Export a tiny metadata CSV (filename, duration_s)
- Save a waveform and mel-spectrogram PNG for one example file

Usage:
  python convert_and_preview.py --src "path/to/raw" --dst "path/to/wav16" [--example example.wav]

Requires:
  pip install librosa soundfile matplotlib numpy pandas
"""
import argparse
from pathlib import Path
import librosa
import soundfile as sf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


def convert_folder(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    exts = {".wav", ".webm", ".mp3", ".m4a", ".ogg"}
    converted = []
    for p in src.rglob("*"):
        if p.suffix.lower() in exts:
            y, sr = librosa.load(str(p), sr=16000, mono=True)
            out = dst / (p.stem + ".wav")
            sf.write(str(out), y, 16000)
            converted.append(out)
    return converted


def make_metadata_csv(dst: Path, out_csv: Path, nrows: int = 8):
    rows = []
    for i, wav in enumerate(sorted(dst.glob("*.wav"))):
        if i >= nrows:
            break
        y, sr = librosa.load(str(wav), sr=None, mono=True)
        dur = librosa.get_duration(y=y, sr=sr)
        rows.append({"filename": wav.name, "duration_s": round(dur, 2)})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def save_waveform_and_melspec(example_wav: Path, out_prefix: Path):
    y, sr = librosa.load(str(example_wav), sr=None, mono=True)
    # Waveform
    plt.figure(figsize=(8, 2.5))
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel("Time (s)")
    plt.title(f"Waveform: {example_wav.name}")
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".waveform.png"), dpi=200)
    plt.close()
    # Mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(8, 2.8))
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram: {example_wav.name}")
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".melspec.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="Folder with raw audio (.wav/.webm/.mp3/...)")
    ap.add_argument("--dst", type=Path, required=True,
                    help="Output folder for 16kHz mono WAVs")
    ap.add_argument("--example", type=str, default="",
                    help="Example WAV (name only) to plot; defaults to first converted file")
    args = ap.parse_args()

    converted = convert_folder(args.src, args.dst)
    if not converted:
        raise SystemExit("No audio files found to convert. Check --src.")

    # Choose example
    example = None
    if args.example:
        cand = args.dst / args.example
        if cand.exists():
            example = cand
    if example is None:
        example = converted[0]

    # Metadata sample CSV
    meta_csv = args.dst / "slide_metadata_sample.csv"
    make_metadata_csv(args.dst, meta_csv)

    # Waveform & mel-spectrogram
    out_prefix = args.dst / f"{example.stem}.preview"
    save_waveform_and_melspec(example, out_prefix)

    print("Done.")
    print(f"- Converted WAVs in: {args.dst}")
    print(f"- Metadata CSV: {meta_csv}")
    print(f"- Waveform PNG: {out_prefix.with_suffix('.waveform.png')}")
    print(f"- Mel-spectrogram PNG: {out_prefix.with_suffix('.melspec.png')}")
