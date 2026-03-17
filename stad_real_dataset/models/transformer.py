"""
Temporal Transformer Module for Spatio-Temporal Reasoning.

Encodes trajectory sequences using self-attention to capture long-range
temporal dependencies in object motion. Outputs per-timestep feature
embeddings used for anomaly classification.

Architecture:
  - Sinusoidal Positional Encoding
  - Multi-head Self-Attention Encoder (captures global motion context)
  - Feedforward Decoder (projects to anomaly feature space)
  - Sequence-level classification head
"""

import torch
import torch.nn as nn
import numpy as np
import math
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Positional Encoding
# ═══════════════════════════════════════════════════════════════════════════

class SinusoidalPositionalEncoding(nn.Module):
    """
    Injects temporal position information using sinusoidal waves.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ═══════════════════════════════════════════════════════════════════════════
# Temporal Transformer Model
# ═══════════════════════════════════════════════════════════════════════════

class TemporalTransformer(nn.Module):
    """
    Transformer-based temporal model for motion pattern analysis.
    
    Takes trajectory features (position, velocity, size, angle) as input
    and produces:
      1. Per-timestep motion embeddings
      2. Sequence-level anomaly logits
      3. Attention weights for interpretability
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model

        # Input projection: raw features → d_model
        self.input_projection = nn.Sequential(
            nn.Linear(config.input_feature_dim, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=d,
            max_len=config.max_seq_length,
            dropout=config.dropout
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers,
            enable_nested_tensor=False,
        )

        # Output heads
        self.motion_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, d // 4),
        )

        # Anomaly classification head (sequence-level)
        self.anomaly_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(d // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()
        logger.info(f"TemporalTransformer: d_model={d}, heads={config.nhead}, "
                    f"layers={config.num_encoder_layers}, "
                    f"params={sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self):
        """Xavier initialization for stable training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x:    (batch, seq_len, input_feature_dim) trajectory features
            mask: (batch, seq_len) boolean mask (True = padded/ignore)
            
        Returns:
            dict with:
              'motion_embeddings': (batch, seq_len, d_model//4)
              'anomaly_scores':    (batch, 1) per-sequence anomaly probability
              'encoded':           (batch, seq_len, d_model) full encoder output
        """
        # Project input features
        h = self.input_projection(x)                      # (B, T, d_model)

        # Add positional encoding
        h = self.pos_encoder(h)                            # (B, T, d_model)

        # Transformer encoder
        if mask is not None:
            src_key_padding_mask = mask
        else:
            src_key_padding_mask = None

        encoded = self.transformer_encoder(
            h, src_key_padding_mask=src_key_padding_mask
        )  # (B, T, d_model)

        # Motion embeddings per timestep
        motion_emb = self.motion_head(encoded)             # (B, T, d_model//4)

        # Sequence-level anomaly score (mean pooling → classification)
        if mask is not None:
            # Masked mean pooling
            mask_expanded = (~mask).unsqueeze(-1).float()
            pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)                   # (B, d_model)

        anomaly_scores = self.anomaly_head(pooled)          # (B, 1)

        return {
            "motion_embeddings": motion_emb,
            "anomaly_scores": anomaly_scores,
            "encoded": encoded,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory Feature Extractor
# ═══════════════════════════════════════════════════════════════════════════

class TrajectoryFeatureExtractor:
    """
    Converts raw trajectory data from ByteTrack into feature tensors
    suitable for the Temporal Transformer.
    
    Features per timestep: [cx, cy, w, h, angle, vx, vy]  (7-dim)
    """

    def __init__(self, config, image_size: Tuple[int, int] = (2048, 2048)):
        self.config = config
        self.image_size = image_size
        self.max_seq_len = config.max_seq_length

    def extract(self, trajectories: Dict) -> Tuple[torch.Tensor, torch.Tensor,
                                                     List[int]]:
        """
        Convert trajectory dict to padded tensor batch.
        
        Args:
            trajectories: {track_id: {"trajectory": [...]}} from ByteTracker
            
        Returns:
            features: (N_tracks, max_seq_len, 7) float tensor
            mask:     (N_tracks, max_seq_len) bool tensor (True = padded)
            track_ids: list of track IDs in order
        """
        all_features = []
        all_masks = []
        track_ids = []

        w, h = self.image_size

        for tid, tdata in trajectories.items():
            traj = tdata["trajectory"]
            seq_len = min(len(traj), self.max_seq_len)

            # Extract and normalize features
            feats = np.zeros((self.max_seq_len, 7), dtype=np.float32)
            mask = np.ones(self.max_seq_len, dtype=bool)  # True = padded

            for t in range(seq_len):
                point = traj[t]
                feats[t, 0] = point["cx"] / w        # Normalized cx
                feats[t, 1] = point["cy"] / h        # Normalized cy
                feats[t, 2] = point["w"] / w          # Normalized width
                feats[t, 3] = point["h"] / h          # Normalized height
                feats[t, 4] = point["angle"] / 360.0  # Normalized angle
                feats[t, 5] = point["vx"] / 50.0      # Normalized vx
                feats[t, 6] = point["vy"] / 50.0      # Normalized vy
                mask[t] = False

            all_features.append(feats)
            all_masks.append(mask)
            track_ids.append(tid)

        if not all_features:
            return (torch.zeros(1, self.max_seq_len, 7),
                    torch.ones(1, self.max_seq_len, dtype=torch.bool),
                    [])

        features = torch.tensor(np.stack(all_features))
        masks = torch.tensor(np.stack(all_masks))

        return features, masks, track_ids


# ═══════════════════════════════════════════════════════════════════════════
# Inference Wrapper (no training needed for demo)
# ═══════════════════════════════════════════════════════════════════════════

class TemporalAnalyzer:
    """
    High-level wrapper that runs the Temporal Transformer on tracked
    trajectories and returns per-track anomaly scores.
    """

    def __init__(self, config, device: str = "cpu"):
        self.config = config
        self.device = device
        self.model = TemporalTransformer(config).to(device)
        self.model.eval()  # Use in inference mode (random weights for demo)
        self.feature_extractor = TrajectoryFeatureExtractor(config)

        logger.info(f"TemporalAnalyzer ready on {device}")

    @torch.no_grad()
    def analyze(self, trajectories: Dict) -> Dict[int, Dict]:
        """
        Analyze trajectories and return anomaly scores.
        
        Args:
            trajectories: Output from ByteTracker.get_all_trajectories()
            
        Returns:
            {track_id: {"anomaly_score": float, "motion_embedding": ndarray}}
        """
        features, masks, track_ids = self.feature_extractor.extract(trajectories)
        features = features.to(self.device)
        masks = masks.to(self.device)

        output = self.model(features, masks)

        results = {}
        anomaly_scores = output["anomaly_scores"].cpu().numpy()
        motion_embs = output["motion_embeddings"].cpu().numpy()

        for i, tid in enumerate(track_ids):
            results[tid] = {
                "transformer_anomaly_score": float(anomaly_scores[i, 0]),
                "motion_embedding": motion_embs[i],
            }

        return results
