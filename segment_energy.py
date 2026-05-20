#!/usr/bin/env python
"""
segment_energy.py
- Short-time energy (STE) cough segmentation.
- Works with a single WAV or an input directory (recurses).
- Saves segments as WAVs and an optional PNG of spans.
- Writes a CSV manifest of segments for later training.

Usage:
  # single file
  python segment_energy.py --wav "C:\\Users\\seanb\\coughvid\\wav16\\123.wav" --outdir segments

  # directory (recurses)
  python segment_energy.py --wav "C:\\Users\\seanb\\coughvid\\wav16" --outdir segments --plot

Requires:
  pip install librosa soundfile matplotlib numpy
"""

import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import csv
import warnings

import numpy as np
import soundfile as sf
import librosa

# Use headless backend so plotting works on servers/CI
import matplotlib
matplotlib.use("Agg")


def short_time_energy(y, frame_length=1024, hop_length=256):
    """Return STE safely; fallback if clip is shorter than one frame."""
    if len(y) < frame_length:
        # pad a tiny bit so framing works, then trim back logically
        pad = frame_length - len(y)
        y = np.pad(y, (0, pad), mode="constant")
    frames = librosa.util.frame(
        y, frame_length=frame_length, hop_length=hop_length)
    ste = (frames**2).sum(axis=0) / frame_length
    return ste


def detect_regions(ste, hop_length, sr, k=3.0, min_dur=0.15, max_sil_gap=0.25):
    """Median + k*MAD threshold; merge short gaps; drop short segments."""
    med = np.median(ste)
    mad = np.median(np.abs(ste - med)) + 1e-12
    thr = med + k * mad

    above = ste > thr
    segs = []
    start = None
    gap = 0
    max_gap_frames = int(max_sil_gap * sr / hop_length)

    for i, flag in enumerate(above):
        if flag:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap_frames:
                end = i - gap
                if end >= start:
                    segs.append((start, end))
                start, gap = None, 0
    if start is not None:
        segs.append((start, len(ste) - 1))

    tsegs = []
    for s, e in segs:
        t0 = s * hop_length / sr
        t1 = (e + 1) * hop_length / sr
        if (t1 - t0) >= min_dur:
            tsegs.append((t0, t1))
    return tsegs, thr


def save_segments(y, sr, segments, outdir: Path, stem: str, pad=0.05):
    outdir.mkdir(parents=True, exist_ok=True)
    saved = []
    for idx, (t0, t1) in enumerate(segments, 1):
        a = max(0, int((t0 - pad) * sr))
        b = min(len(y), int((t1 + pad) * sr))
        if b <= a:  # guard
            continue
        clip = y[a:b]
        out = outdir / f"{stem}_cough_{idx:02d}.wav"
        sf.write(str(out), clip, sr)
        saved.append(out)
    return saved


def plot_segments(y, sr, segments, out_png: Path, title="Detected cough segments"):
    times = np.arange(len(y)) / sr
    plt.figure(figsize=(9, 3))
    plt.plot(times, y, linewidth=0.8)
    for (t0, t1) in segments:
        plt.axvspan(t0, t1, alpha=0.3)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def process_file(wav_path: Path, outdir: Path, frame: int, hop: int, k: float,
                 min_dur: float, max_gap: float, make_plot: bool, csv_writer):
    try:
        # Strictly load as mono; keep native sr to respect 16k if already converted
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    except Exception as e:
        print(f"[WARN] Could not read {wav_path}: {e}")
        return 0

    if y.size == 0 or sr is None:
        print(f"[WARN] Empty audio {wav_path}")
        return 0

    ste = short_time_energy(y, frame_length=frame, hop_length=hop)
    segments, thr = detect_regions(
        ste, hop_length=hop, sr=sr, k=k, min_dur=min_dur, max_sil_gap=max_gap
    )

    stem = wav_path.stem
    seg_dir = outdir / "clips" / stem
    saved = save_segments(y, sr, segments, seg_dir, stem=stem, pad=0.05)

    # CSV rows
    for idx, (t0, t1) in enumerate(segments, 1):
        csv_writer.writerow({
            "file": str(wav_path),
            "sr": sr,
            "seg_idx": idx,
            "t0": f"{t0:.3f}",
            "t1": f"{t1:.3f}",
            "thr": f"{thr:.6e}",
            "out_wav": str(seg_dir / f"{stem}_cough_{idx:02d}.wav")
        })

    if make_plot:
        png_out = outdir / "plots" / f"{stem}.segments.png"
        try:
            plot_segments(y, sr, segments, png_out)
        except Exception as e:
            print(f"[WARN] Plot failed for {wav_path}: {e}")

    print(f"[OK] {wav_path.name}: {len(segments)} segment(s)")
    return len(segments)


def iter_wavs(path: Path):
    if path.is_file():
        if path.suffix.lower() == ".wav":
            yield path
        return
    # directory: recurse
    for p in path.rglob("*.wav"):
        if p.is_file():
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=Path, required=True,
                    help="Path to a 16k mono WAV file or a directory of WAVs")
    ap.add_argument("--outdir", type=Path, required=True,
                    help="Output directory for segments/plots/csv")
    ap.add_argument("--frame", type=int, default=1024,
                    help="Frame length (samples)")
    ap.add_argument("--hop", type=int, default=256,
                    help="Hop length (samples)")
    ap.add_argument("--k", type=float, default=3.0,
                    help="Threshold factor (higher = fewer segments)")
    ap.add_argument("--min_dur", type=float, default=0.15,
                    help="Minimum segment duration (seconds)")
    ap.add_argument("--max_gap", type=float, default=0.25,
                    help="Maximum silence gap to merge (seconds)")
    ap.add_argument("--plot", action="store_true",
                    help="Also export per-file PNG plots with highlighted segments")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "segments_manifest.csv"
    total_files = 0
    total_segments = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "sr", "seg_idx",
                           "t0", "t1", "thr", "out_wav"]
        )
        writer.writeheader()

        for wav_path in iter_wavs(args.wav):
            total_files += 1
            total_segments += process_file(
                wav_path=wav_path,
                outdir=args.outdir,
                frame=args.frame,
                hop=args.hop,
                k=args.k,
                min_dur=args.min_dur,
                max_gap=args.max_gap,
                make_plot=args.plot,
                csv_writer=writer,
            )

    print(f"\nProcessed files: {total_files}")
    print(f"Total segments:  {total_segments}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
