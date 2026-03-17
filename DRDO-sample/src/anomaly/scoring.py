"""
Spatio-Temporal Anomaly Scoring.
"""

import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import cosine_similarity

class AnomalyScorer:
    def __init__(self):
        self.kde = KernelDensity(bandwidth=1.0)

    def fit_spatial(self, positions):
        """
        Fit KDE for spatial distribution.
        Args:
            positions: array of (x, y) positions
        """
        self.kde.fit(positions)

    def score_spatial_anomaly(self, position):
        """
        Score spatial anomaly using KDE.
        """
        log_density = self.kde.score_samples([position])
        return -log_density[0]  # higher score for lower density

    def score_semantic_anomaly(self, embedding, historical_embeddings):
        """
        Score semantic anomaly using cosine similarity to historical.
        """
        similarities = cosine_similarity([embedding], historical_embeddings)
        max_sim = np.max(similarities)
        return 1 - max_sim  # higher score for lower similarity