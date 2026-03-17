"""
Anomaly Detection and Risk Scoring Module.

Combines multiple anomaly indicators from trajectory analysis and
transformer embeddings to produce:
  1. Per-track risk scores (0-1)
  2. Spatial anomaly heatmap
  3. Categorical threat classification

Anomaly Indicators:
  - Speed anomaly:        Sudden acceleration / deceleration
  - Direction anomaly:    Abrupt heading changes (evasive maneuvers)
  - Convergence anomaly:  Multiple objects converging to a point
  - Loitering anomaly:    Circular / stationary patterns in sensitive areas
  - Formation anomaly:    Coordinated group movement
  - Transformer score:    Learned temporal pattern deviation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Individual Anomaly Detectors
# ═══════════════════════════════════════════════════════════════════════════

class SpeedAnomalyDetector:
    """Detects abnormal speed patterns using statistical thresholds."""

    def __init__(self, sigma_threshold: float = 2.5):
        self.sigma = sigma_threshold

    def compute(self, trajectory: List[Dict]) -> float:
        """
        Returns anomaly score [0,1] based on speed variations.
        High score = sudden speed bursts or stops.
        """
        if len(trajectory) < 3:
            return 0.0

        speeds = []
        for pt in trajectory:
            vx, vy = pt.get("vx", 0), pt.get("vy", 0)
            speeds.append(np.sqrt(vx**2 + vy**2))

        speeds = np.array(speeds)
        mean_speed = speeds.mean()
        std_speed = speeds.std() + 1e-6

        # Count how many timesteps exceed sigma threshold
        anomalous = np.sum(np.abs(speeds - mean_speed) > self.sigma * std_speed)
        ratio = anomalous / len(speeds)

        # Also check max acceleration
        accels = np.abs(np.diff(speeds))
        max_accel = accels.max() if len(accels) > 0 else 0
        accel_score = min(max_accel / 15.0, 1.0)

        return float(np.clip(0.5 * ratio + 0.5 * accel_score, 0, 1))


class DirectionAnomalyDetector:
    """Detects abrupt changes in movement direction."""

    def __init__(self, angle_threshold_deg: float = 90.0):
        self.threshold = angle_threshold_deg

    def compute(self, trajectory: List[Dict]) -> float:
        """
        Returns anomaly score [0,1] based on direction changes.
        High score = sharp turns, zigzag evasion patterns.
        """
        if len(trajectory) < 3:
            return 0.0

        angles = []
        for pt in trajectory:
            vx, vy = pt.get("vx", 0), pt.get("vy", 0)
            if abs(vx) > 0.01 or abs(vy) > 0.01:
                angles.append(np.degrees(np.arctan2(vy, vx)))

        if len(angles) < 3:
            return 0.0

        # Compute angular differences
        angle_changes = []
        for i in range(1, len(angles)):
            diff = abs(angles[i] - angles[i-1])
            if diff > 180:
                diff = 360 - diff
            angle_changes.append(diff)

        angle_changes = np.array(angle_changes)
        sharp_turns = np.sum(angle_changes > self.threshold)
        ratio = sharp_turns / len(angle_changes)

        max_turn = angle_changes.max() if len(angle_changes) > 0 else 0
        turn_score = min(max_turn / 180.0, 1.0)

        return float(np.clip(0.4 * ratio + 0.6 * turn_score, 0, 1))


class ConvergenceDetector:
    """Detects multiple objects converging toward a common point."""

    def __init__(self, radius: float = 100.0, min_objects: int = 3):
        self.radius = radius
        self.min_objects = min_objects

    def compute(self, all_tracks: List[Dict]) -> Tuple[float, Optional[Tuple[float, float]]]:
        """
        Analyze current positions of all tracks for convergence.
        
        Returns:
            score:  [0,1] convergence anomaly score
            center: (x, y) convergence point or None
        """
        if len(all_tracks) < self.min_objects:
            return 0.0, None

        positions = np.array([[t["cx"], t["cy"]] for t in all_tracks])

        # Try each track position as potential convergence center
        best_score = 0.0
        best_center = None

        for pos in positions:
            dists = np.linalg.norm(positions - pos, axis=1)
            nearby = np.sum(dists < self.radius)
            if nearby >= self.min_objects:
                score = nearby / len(all_tracks)
                if score > best_score:
                    best_score = score
                    best_center = (float(pos[0]), float(pos[1]))

        return float(np.clip(best_score, 0, 1)), best_center


class LoiteringDetector:
    """Detects circular or stationary patterns indicating surveillance/loitering."""

    def __init__(self, displacement_threshold: float = 50.0):
        self.disp_thresh = displacement_threshold

    def compute(self, trajectory: List[Dict]) -> float:
        """
        Returns anomaly score [0,1]. High = object staying in small area
        despite high cumulative distance traveled (circular motion).
        """
        if len(trajectory) < 5:
            return 0.0

        positions = np.array([[pt["cx"], pt["cy"]] for pt in trajectory])

        # Net displacement (start to end)
        net_disp = np.linalg.norm(positions[-1] - positions[0])

        # Cumulative distance traveled
        diffs = np.diff(positions, axis=0)
        cum_dist = np.sum(np.linalg.norm(diffs, axis=1))

        if cum_dist < 1.0:
            return 0.0

        # Loitering ratio: high cumulative distance but low net displacement
        loiter_ratio = 1.0 - (net_disp / (cum_dist + 1e-6))

        # Also check spatial spread (std of positions)
        spread = np.std(positions, axis=0).mean()
        spread_score = max(0, 1.0 - spread / self.disp_thresh)

        return float(np.clip(0.6 * loiter_ratio + 0.4 * spread_score, 0, 1))


# ═══════════════════════════════════════════════════════════════════════════
# Risk Score Aggregator
# ═══════════════════════════════════════════════════════════════════════════

class RiskScorer:
    """
    Aggregates multiple anomaly indicators into a unified risk score
    for each tracked object.
    
    Risk Levels:
      LOW    (0.0 - 0.3):  Normal behavior
      MEDIUM (0.3 - 0.6):  Unusual but possibly benign
      HIGH   (0.6 - 0.85): Suspicious, requires attention
      CRITICAL (0.85-1.0): Highly anomalous, likely threat
    """

    def __init__(self, config):
        self.config = config
        self.speed_detector = SpeedAnomalyDetector(config.speed_anomaly_sigma)
        self.direction_detector = DirectionAnomalyDetector(config.direction_anomaly_deg)
        self.convergence_detector = ConvergenceDetector(
            config.convergence_radius, config.clustering_min_samples
        )
        self.loitering_detector = LoiteringDetector()

        # Weights for combining anomaly scores
        self.weights = {
            "speed": 0.20,
            "direction": 0.20,
            "loitering": 0.15,
            "convergence": 0.15,
            "transformer": 0.30,
        }

        logger.info("RiskScorer initialized with multi-indicator fusion")

    def compute_risk_scores(self, trajectories: Dict,
                            transformer_scores: Dict[int, Dict],
                            current_tracks: List[Dict]) -> Dict[int, Dict]:
        """
        Compute comprehensive risk scores for all tracks.
        
        Args:
            trajectories:       From ByteTracker.get_all_trajectories()
            transformer_scores: From TemporalAnalyzer.analyze()
            current_tracks:     Current frame active track dicts
            
        Returns:
            {track_id: {
                "risk_score": float,
                "risk_level": str,
                "anomaly_breakdown": {...},
                ...
            }}
        """
        # ── Convergence (global, across all tracks) ──
        conv_score, conv_center = self.convergence_detector.compute(current_tracks)

        results = {}

        for tid, tdata in trajectories.items():
            traj = tdata["trajectory"]

            # Per-track anomaly indicators
            speed_score = self.speed_detector.compute(traj)
            direction_score = self.direction_detector.compute(traj)
            loiter_score = self.loitering_detector.compute(traj)

            # Transformer score (if available)
            t_score = 0.5  # default
            if tid in transformer_scores:
                t_score = transformer_scores[tid].get("transformer_anomaly_score", 0.5)

            # Weighted fusion
            risk_score = (
                self.weights["speed"] * speed_score +
                self.weights["direction"] * direction_score +
                self.weights["loitering"] * loiter_score +
                self.weights["convergence"] * conv_score +
                self.weights["transformer"] * t_score
            )
            risk_score = float(np.clip(risk_score, 0, 1))

            # Classify risk level
            if risk_score >= self.config.risk_threshold_high:
                risk_level = "CRITICAL"
            elif risk_score >= self.config.risk_threshold_medium:
                risk_level = "HIGH"
            elif risk_score >= self.config.risk_threshold_low:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            results[tid] = {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "anomaly_breakdown": {
                    "speed": round(speed_score, 3),
                    "direction": round(direction_score, 3),
                    "loitering": round(loiter_score, 3),
                    "convergence": round(conv_score, 3),
                    "transformer": round(t_score, 3),
                },
                "class_name": tdata.get("class_name", "unknown"),
                "trajectory_length": len(traj),
                "convergence_center": conv_center,
            }

        return results


# ═══════════════════════════════════════════════════════════════════════════
# Spatial Anomaly Map Generator
# ═══════════════════════════════════════════════════════════════════════════

class SpatialAnomalyMapper:
    """
    Generates a 2D heatmap of anomaly intensity across the spatial domain.
    
    Each cell in the grid accumulates risk scores from nearby tracks,
    producing a visual overview of where suspicious activity is concentrated.
    """

    def __init__(self, image_size: Tuple[int, int] = (2048, 2048),
                 grid_size: Tuple[int, int] = (32, 32)):
        self.image_size = image_size
        self.grid_size = grid_size
        self.cell_w = image_size[0] / grid_size[0]
        self.cell_h = image_size[1] / grid_size[1]

    def generate_heatmap(self, risk_results: Dict[int, Dict],
                          trajectories: Dict) -> np.ndarray:
        """
        Generate spatial anomaly heatmap.
        
        Args:
            risk_results:  Output from RiskScorer.compute_risk_scores()
            trajectories:  Trajectory data from tracker
            
        Returns:
            heatmap: (grid_h, grid_w) float array normalized [0, 1]
        """
        heatmap = np.zeros(self.grid_size, dtype=np.float64)

        for tid, risk_data in risk_results.items():
            if tid not in trajectories:
                continue

            traj = trajectories[tid]["trajectory"]
            risk_score = risk_data["risk_score"]

            # Deposit risk score along the trajectory
            for pt in traj:
                gx = int(np.clip(pt["cx"] / self.cell_w, 0, self.grid_size[0] - 1))
                gy = int(np.clip(pt["cy"] / self.cell_h, 0, self.grid_size[1] - 1))

                # Gaussian-like spread to neighboring cells
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                            dist = abs(dx) + abs(dy)
                            weight = [1.0, 0.5, 0.25][dist]
                            heatmap[ny, nx] += risk_score * weight

        # Normalize
        max_val = heatmap.max()
        if max_val > 0:
            heatmap /= max_val

        return heatmap.astype(np.float32)

    def get_hotspots(self, heatmap: np.ndarray,
                     threshold: float = 0.7) -> List[Dict]:
        """
        Extract hotspot regions from the anomaly heatmap.
        
        Returns list of hotspot dicts with grid coordinates and scores.
        """
        hotspots = []
        for gy in range(self.grid_size[1]):
            for gx in range(self.grid_size[0]):
                if heatmap[gy, gx] >= threshold:
                    hotspots.append({
                        "grid_x": gx,
                        "grid_y": gy,
                        "center_x": (gx + 0.5) * self.cell_w,
                        "center_y": (gy + 0.5) * self.cell_h,
                        "intensity": float(heatmap[gy, gx]),
                    })

        # Sort by intensity (highest first)
        hotspots.sort(key=lambda h: h["intensity"], reverse=True)
        return hotspots
