# Cough Project Scripts

## Midterm (Detection & Segmentation)

Python files:

- convert_and_preview.py — convert to 16 kHz mono WAV, export small metadata CSV, and save waveform + mel-spectrogram PNGs.
- segment_energy.py — simple cough segmentation using short-time energy; saves extracted segments and a slide-ready plot.
- extract_features.py — builds your features from COUGHVID audio and saves them into a training-ready CSV.

CSV data files:

- metadata_compiled.csv — the cleaned and merged COUGHVID metadata that links audio files to labels and subject information.
- segment_severity.csv — records how each audio file is segmented into cough clips and assigns a severity label to each segment.

## Finals (Classification)

Python files:

- yamnet_baseline.py: zero-shot cough score with YAMNet , or train a light linear-probe classifier on YAMNet embeddings.
- extract_yamnet_features.py — extracts YAMNet features from audio clips and saves them for training/testing.
- make_segment_severity.py — converts raw COUGHVID audio (and/or metadata) into labeled segments for the severity task.
- predict_severity.py — loads the trained model + scaler and runs inference on new audio segments to output predicted severity.
- train_severity_models.py — trains your severity classifiers from the saved features, evaluates, and exports the final model (model + scaler + result).

CSV data files:

- segments_manifest — segmented files of every audios labelled with local path, uuid and segmented clips.
- features_severity.csv — extracted hand-crafted features for each cough segment with its severity label, used to train and test models.
- features_yamnet.csv — YAMNet embedding features for each cough segment, used as an alternative feature set for modeling.

## Versions and packages

Quick start (Windows PowerShell):
py -3.10 -m venv venv
.\venv\Scripts\activate
Preinstall packages:
'pip install librosa soundfile matplotlib numpy pandas scikit-learn tensorflow==2.15 tensorflow_hub'

## Open souce references and datasets

Note: Trained models, scalers, segmented audio, and visualization outputs are omitted as they are reproducible artifacts generated from the provided scripts and CSV files.

Open source segmentation code:
Paper and audio dataset: 'https://www.nature.com/articles/s41597-021-00937-4' /'https://zenodo.org/records/7024894'
Open-source repo: 'https://github.com/bagustris/detect-segment-cough'
