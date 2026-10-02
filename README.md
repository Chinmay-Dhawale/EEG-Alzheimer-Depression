# EEG Alzheimer's & Depression AI

An academic EEG machine-learning project for classification of:

- Healthy
- Alzheimer's disease
- Depression

## Project Pipeline

EEG Upload
→ Preprocessing
→ Channel Selection
→ Resampling
→ Filtering
→ 10-second Segmentation
→ Feature Extraction
→ SVM Classification
→ EEG Analysis Report

## Models

The project contains three trained models:

1. Alzheimer's vs Healthy
2. Depression vs Healthy
3. Three-class experimental model

## EEG Processing

The system uses:

- 19 common EEG channels
- 256 Hz sampling rate
- 1–45 Hz bandpass filtering
- 50 Hz notch filtering
- Average reference
- 10-second EEG windows
- Artifact rejection
- EEG frequency-band features
- SVM machine-learning models

## Important

This is an academic EEG machine-learning project.

The model outputs are classification results and should not be interpreted as a medical diagnosis.
