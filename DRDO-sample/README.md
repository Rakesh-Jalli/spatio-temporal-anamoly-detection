# Spatio-Temporal Anomaly Detection in High-Resolution Satellite Imagery

This project implements an advanced AI-based system for detecting spatio-temporal anomalies in satellite imagery for defence and strategic surveillance applications.

## Features

- **Oriented Object Detection**: Uses YOLO-OBB to detect vehicles, ships, aircraft, and infrastructure with arbitrary orientations.
- **Multi-Object Tracking**: Employs ByteTrack for maintaining object identity across time frames.
- **Semantic Understanding**: Utilizes CLIP embeddings for contextual representation.
- **Vector Database**: Stores embeddings in ChromaDB for similarity retrieval.
- **Anomaly Scoring**: Combines Kernel Density Estimation and cosine similarity for anomaly detection.
- **Visualization Dashboard**: Provides interpretable outputs with heatmaps and overlays.

## Installation

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt` or `pip install -e .`
3. Configure Python environment.

## Usage

Run the main script: `python -m src.main`

For training: `python scripts/train.py`

For evaluation: `python scripts/evaluate.py`

Launch dashboard: `streamlit run src/visualization/dashboard.py`

## Datasets

Supports SpaceNet, xView, DOTA, Sentinel imagery.

## Evaluation

Uses mAP for detection and F1-score for anomaly detection.