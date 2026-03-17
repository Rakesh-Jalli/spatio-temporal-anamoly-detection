"""
ByteTrack Multi-Object Tracker Module.

Implements the BYTE association strategy for robust multi-object tracking
in satellite imagery sequences. Handles track lifecycle management
(creation, update, deletion) and provides trajectory history for
downstream temporal analysis.

Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
Every Detection Box," ECCV 2022.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Kalman Filter for State Estimation
# ═══════════════════════════════════════════════════════════════════════════

class KalmanFilter:
    """
    Lightweight Kalman filter for 2D bounding box tracking.
    State: [cx, cy, w, h, vx, vy, vw, vh]
    """

    def __init__(self):
        self.dt = 1.0
        # State transition (8x8)
        self.F = np.eye(8)
        self.F[0, 4] = self.dt
        self.F[1, 5] = self.dt
        self.F[2, 6] = self.dt
        self.F[3, 7] = self.dt

        # Observation matrix (4x8)
        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1

        # Process noise
        self.Q = np.eye(8) * 1.0
        self.Q[4:, 4:] *= 0.1

        # Measurement noise
        self.R = np.eye(4) * 1.0

        self.x = np.zeros(8)        # State
        self.P = np.eye(8) * 10.0   # Covariance

    def init_state(self, cx, cy, w, h):
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)
        self.P = np.eye(8) * 10.0

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]

    def update(self, z: np.ndarray) -> np.ndarray:
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self.x[:4]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[4:6]


# ═══════════════════════════════════════════════════════════════════════════
# Track Object
# ═══════════════════════════════════════════════════════════════════════════

class STrack:
    """Single object track with Kalman-filtered state estimation."""

    _id_counter = 0

    def __init__(self, detection_dict: Dict):
        STrack._id_counter += 1
        self.track_id = STrack._id_counter
        self.kalman = KalmanFilter()

        cx = detection_dict["cx"]
        cy = detection_dict["cy"]
        w = detection_dict["width"]
        h = detection_dict["height"]
        self.kalman.init_state(cx, cy, w, h)

        self.class_id = detection_dict.get("class_id", 0)
        self.class_name = detection_dict.get("class_name", "unknown")
        self.angle = detection_dict.get("angle", 0)
        self.confidence = detection_dict.get("confidence", 0.5)

        self.is_activated = True
        self.frame_id = 0
        self.start_frame = 0
        self.time_since_update = 0

        # Trajectory history: list of (frame_id, cx, cy, vx, vy, w, h, angle)
        self.trajectory: List[Dict] = []
        self._record_state(0)

    def _record_state(self, frame_id: int):
        """Record current state in trajectory."""
        state = self.kalman.x
        self.trajectory.append({
            "frame_id": frame_id,
            "cx": float(state[0]),
            "cy": float(state[1]),
            "w": float(state[2]),
            "h": float(state[3]),
            "vx": float(state[4]),
            "vy": float(state[5]),
            "angle": float(self.angle),
        })

    def predict(self):
        """Predict next state."""
        self.kalman.predict()

    def update(self, detection_dict: Dict, frame_id: int):
        """Update track with new detection."""
        z = np.array([detection_dict["cx"], detection_dict["cy"],
                       detection_dict["width"], detection_dict["height"]])
        self.kalman.update(z)
        self.angle = detection_dict.get("angle", self.angle)
        self.confidence = detection_dict.get("confidence", self.confidence)
        self.frame_id = frame_id
        self.time_since_update = 0
        self._record_state(frame_id)

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.kalman.x[0]), float(self.kalman.x[1])

    @property
    def velocity(self) -> Tuple[float, float]:
        v = self.kalman.velocity
        return float(v[0]), float(v[1])

    @property
    def speed(self) -> float:
        vx, vy = self.velocity
        return float(np.sqrt(vx**2 + vy**2))

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        s = self.kalman.x
        return float(s[0]), float(s[1]), float(s[2]), float(s[3])

    def to_dict(self) -> Dict:
        cx, cy = self.position
        vx, vy = self.velocity
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "cx": cx, "cy": cy,
            "width": float(self.kalman.x[2]),
            "height": float(self.kalman.x[3]),
            "vx": vx, "vy": vy,
            "speed": self.speed,
            "angle": self.angle,
            "confidence": self.confidence,
            "age": len(self.trajectory),
        }

    @classmethod
    def reset_id_counter(cls):
        cls._id_counter = 0


# ═══════════════════════════════════════════════════════════════════════════
# IoU Computation for Axis-Aligned Boxes
# ═══════════════════════════════════════════════════════════════════════════

def _compute_iou_matrix(tracks: List[STrack], detections: List[Dict]) -> np.ndarray:
    """Compute IoU cost matrix between track predictions and detections."""
    n_tracks = len(tracks)
    n_dets = len(detections)
    iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)

    for i, track in enumerate(tracks):
        tcx, tcy, tw, th = track.bbox
        tx1, ty1 = tcx - tw / 2, tcy - th / 2
        tx2, ty2 = tcx + tw / 2, tcy + th / 2

        for j, det in enumerate(detections):
            dcx, dcy = det["cx"], det["cy"]
            dw, dh = det["width"], det["height"]
            dx1, dy1 = dcx - dw / 2, dcy - dh / 2
            dx2, dy2 = dcx + dw / 2, dcy + dh / 2

            inter_x1 = max(tx1, dx1)
            inter_y1 = max(ty1, dy1)
            inter_x2 = min(tx2, dx2)
            inter_y2 = min(ty2, dy2)

            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            union_area = tw * th + dw * dh - inter_area + 1e-6

            iou_matrix[i, j] = inter_area / union_area

    return iou_matrix


# ═══════════════════════════════════════════════════════════════════════════
# ByteTrack Tracker
# ═══════════════════════════════════════════════════════════════════════════

class ByteTracker:
    """
    ByteTrack multi-object tracker.
    
    Key innovation: Uses ALL detection boxes (high + low confidence) in a
    two-stage association process, recovering occluded or partially visible targets.
    
    Stage 1: Match high-confidence detections to tracks using IoU
    Stage 2: Match remaining low-confidence detections to unmatched tracks
    """

    def __init__(self, config):
        self.config = config
        self.active_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self.removed_tracks: List[STrack] = []
        self.frame_id = 0

        STrack.reset_id_counter()
        logger.info(f"ByteTracker initialized: thresh={config.track_thresh}, "
                    f"buffer={config.track_buffer}")

    def update(self, detections: List[Dict]) -> List[STrack]:
        """
        Process detections for one frame and return active tracks.
        
        Args:
            detections: List of detection dicts with keys
                        {cx, cy, width, height, angle, confidence, class_id, class_name}
        
        Returns:
            List of active STrack objects
        """
        self.frame_id += 1

        # ── Split detections by confidence ──
        high_dets = [d for d in detections if d["confidence"] >= self.config.track_thresh]
        low_dets = [d for d in detections if d["confidence"] < self.config.track_thresh]

        # ── Predict existing tracks ──
        for track in self.active_tracks:
            track.predict()

        # ══════════════ STAGE 1: High-confidence association ══════════════
        matched_tracks, unmatched_tracks, unmatched_dets = self._associate(
            self.active_tracks, high_dets, self.config.match_thresh
        )

        # Update matched tracks
        for track, det in matched_tracks:
            track.update(det, self.frame_id)

        # ══════════════ STAGE 2: Low-confidence association ══════════════
        if low_dets and unmatched_tracks:
            matched_low, still_unmatched_tracks, _ = self._associate(
                unmatched_tracks, low_dets, 0.5
            )
            for track, det in matched_low:
                track.update(det, self.frame_id)
            unmatched_tracks = still_unmatched_tracks

        # ── Handle unmatched tracks → lost ──
        for track in unmatched_tracks:
            track.time_since_update += 1
            if track.time_since_update > self.config.max_time_lost:
                self.removed_tracks.append(track)
            else:
                self.lost_tracks.append(track)

        # ── Remove lost tracks from active ──
        self.active_tracks = [t for t in self.active_tracks
                               if t.time_since_update == 0]

        # ── Try to re-activate lost tracks ──
        if unmatched_dets:
            reactivated = []
            still_unmatched_dets = []

            for det in unmatched_dets:
                best_iou = 0
                best_lost = None
                for lt in self.lost_tracks:
                    iou = self._single_iou(lt, det)
                    if iou > best_iou:
                        best_iou = iou
                        best_lost = lt
                if best_iou > 0.3 and best_lost is not None:
                    best_lost.update(det, self.frame_id)
                    best_lost.is_activated = True
                    reactivated.append(best_lost)
                    self.lost_tracks.remove(best_lost)
                else:
                    still_unmatched_dets.append(det)

            self.active_tracks.extend(reactivated)
            unmatched_dets = still_unmatched_dets

        # ── Create new tracks for remaining detections ──
        for det in unmatched_dets:
            if det.get("confidence", 1.0) >= self.config.track_thresh:
                new_track = STrack(det)
                new_track.start_frame = self.frame_id
                new_track.frame_id = self.frame_id
                self.active_tracks.append(new_track)

        # ── Cleanup old lost tracks ──
        self.lost_tracks = [t for t in self.lost_tracks
                            if t.time_since_update <= self.config.max_time_lost]

        logger.debug(f"Frame {self.frame_id}: {len(self.active_tracks)} active, "
                     f"{len(self.lost_tracks)} lost, "
                     f"{len(high_dets)} high-conf, {len(low_dets)} low-conf dets")

        return self.active_tracks

    def _associate(self, tracks: List[STrack], detections: List[Dict],
                   iou_threshold: float):
        """Hungarian-free greedy association using IoU."""
        if not tracks or not detections:
            return [], list(tracks), list(detections)

        iou_matrix = _compute_iou_matrix(tracks, detections)

        matched_pairs = []
        matched_track_idx = set()
        matched_det_idx = set()

        # Greedy matching: highest IoU first
        while True:
            if iou_matrix.size == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < iou_threshold:
                break
            idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            ti, di = idx[0], idx[1]

            if ti not in matched_track_idx and di not in matched_det_idx:
                matched_pairs.append((tracks[ti], detections[di]))
                matched_track_idx.add(ti)
                matched_det_idx.add(di)

            iou_matrix[ti, di] = 0  # Mark as used

        unmatched_tracks = [t for i, t in enumerate(tracks) if i not in matched_track_idx]
        unmatched_dets = [d for i, d in enumerate(detections) if i not in matched_det_idx]

        return matched_pairs, unmatched_tracks, unmatched_dets

    def _single_iou(self, track: STrack, det: Dict) -> float:
        """Compute IoU between one track and one detection."""
        tcx, tcy, tw, th = track.bbox
        tx1, ty1 = tcx - tw / 2, tcy - th / 2
        tx2, ty2 = tcx + tw / 2, tcy + th / 2

        dcx, dcy = det["cx"], det["cy"]
        dw, dh = det["width"], det["height"]
        dx1, dy1 = dcx - dw / 2, dcy - dh / 2
        dx2, dy2 = dcx + dw / 2, dcy + dh / 2

        inter_x1 = max(tx1, dx1)
        inter_y1 = max(ty1, dy1)
        inter_x2 = min(tx2, dx2)
        inter_y2 = min(ty2, dy2)

        inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union = tw * th + dw * dh - inter + 1e-6
        return inter / union

    def get_all_trajectories(self) -> Dict[int, List[Dict]]:
        """Return trajectory history for all tracks (active + removed)."""
        all_tracks = self.active_tracks + self.lost_tracks + self.removed_tracks
        trajectories = {}
        for t in all_tracks:
            if len(t.trajectory) >= 2:
                trajectories[t.track_id] = {
                    "class_name": t.class_name,
                    "class_id": t.class_id,
                    "trajectory": t.trajectory,
                }
        return trajectories

    def __repr__(self):
        return (f"ByteTracker(active={len(self.active_tracks)}, "
                f"lost={len(self.lost_tracks)}, frame={self.frame_id})")
