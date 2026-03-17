"""
Configuration module for the Spatio-Temporal Movement Anomaly Detection System.
Contains all hyperparameters, paths, and system settings.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


# ─────────────────────────── Path Configuration ───────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


# ─────────────────────────── Detector Config ──────────────────────────────

@dataclass
class DetectorConfig:
    """YOLOv8-OBB detector configuration."""
    model_variant: str = "yolov8s-obb"          # pretrained DOTA
    confidence_threshold: float = 0.10
    iou_threshold: float = 0.45
    max_detections: int = 800
    input_size: Tuple[int, int] = (640, 640)
    classes: List[str] = field(default_factory=lambda: [
        "plane", "ship", "storage tank", "baseball diamond",
        "tennis court", "basketball court", "ground track field",
        "harbor", "bridge", "large vehicle", "small vehicle",
        "helicopter", "roundabout", "soccer ball field", "swimming pool",
    ])
    num_classes: int = 15
    use_obb: bool = True


# ─────────────────────────── Tracker Config ───────────────────────────────

@dataclass
class TrackerConfig:
    """ByteTrack multi-object tracker configuration."""
    track_thresh: float = 0.5                    # High-score detection threshold
    track_buffer: int = 30                       # Frames to keep lost tracks
    match_thresh: float = 0.8                    # IoU matching threshold
    min_box_area: int = 10                       # Minimum bounding box area
    frame_rate: int = 1                          # Satellite revisit rate (fps)
    max_time_lost: int = 30                      # Max frames before track deletion


# ─────────────────────────── Transformer Config ──────────────────────────

@dataclass
class TransformerConfig:
    """Temporal Transformer configuration for sequence modeling."""
    d_model: int = 128                           # Model dimension
    nhead: int = 8                               # Number of attention heads
    num_encoder_layers: int = 4                  # Encoder depth
    num_decoder_layers: int = 2                  # Decoder depth
    dim_feedforward: int = 512                   # FFN inner dimension
    dropout: float = 0.1
    max_seq_length: int = 64                     # Maximum temporal sequence
    input_feature_dim: int = 7                   # [x, y, w, h, angle, vx, vy]
    positional_encoding: str = "sinusoidal"      # or "learnable"


# ─────────────────────────── Anomaly Config ───────────────────────────────

@dataclass
class AnomalyConfig:
    """Anomaly detection and risk scoring configuration."""
    risk_threshold_low: float = 0.3
    risk_threshold_medium: float = 0.6
    risk_threshold_high: float = 0.85
    speed_anomaly_sigma: float = 2.5             # σ for speed outlier
    direction_anomaly_deg: float = 90.0          # Sudden direction change (°)
    clustering_eps: float = 50.0                 # DBSCAN eps in pixels
    clustering_min_samples: int = 3
    temporal_window: int = 10                    # Frames for rolling analysis
    spatial_grid_size: Tuple[int, int] = (32, 32)  # Anomaly heatmap resolution
    convergence_radius: float = 100.0            # Pixels for convergence detect


# ─────────────────────────── Pipeline Config ──────────────────────────────

@dataclass
class PipelineConfig:
    """End-to-end pipeline configuration."""
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)

    # General
    device: str = "cpu"                          # "cuda" or "cpu"
    num_workers: int = 4
    batch_size: int = 1
    seed: int = 42
    verbose: bool = True
    demo_mode: bool = False                      # Real YOLOv8-OBB inference

    # Image settings
    image_size: Tuple[int, int] = (1024, 1024)   # Satellite image size
    num_demo_frames: int = 20                    # Frames for demo sequence
    num_temporal_scenes: int = 20                # Real temporal scenes per region

    # Output
    save_visualizations: bool = True
    output_dir: str = OUTPUT_DIR


def get_config() -> PipelineConfig:
    """Return default pipeline configuration."""
    cfg = PipelineConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    return cfg
