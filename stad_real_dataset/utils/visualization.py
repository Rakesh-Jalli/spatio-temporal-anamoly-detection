"""
Visualization Module for the Anomaly Detection System.

Generates comprehensive visual outputs:
  1. Detection overlay (OBB boxes on frames)
  2. Trajectory visualization with risk coloring
  3. Spatial anomaly heatmap
  4. Dashboard summary with risk statistics
"""

import numpy as np
import cv2
import os
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Color Schemes
# ═══════════════════════════════════════════════════════════════════════════

RISK_COLORS = {
    "LOW":      (0, 200, 0),      # Green
    "MEDIUM":   (0, 200, 255),    # Yellow-Orange
    "HIGH":     (0, 100, 255),    # Orange
    "CRITICAL": (0, 0, 255),      # Red
}

CLASS_COLORS = [
    (0, 255, 255),     # plane - cyan
    (0, 0, 255),       # ship - red
    (255, 165, 0),     # storage tank - orange
    (0, 255, 0),       # baseball diamond - green
    (255, 255, 0),     # tennis court - yellow
    (255, 0, 255),     # basketball court - magenta
    (128, 128, 0),     # ground track field - olive
    (0, 128, 255),     # harbor - blue
    (255, 0, 128),     # bridge - pink
    (0, 200, 0),       # large vehicle - dark green
    (200, 200, 0),     # small vehicle - dark yellow
    (0, 100, 255),     # helicopter - orange-blue
    (200, 0, 200),     # roundabout - purple
    (0, 200, 200),     # soccer ball field - teal
    (255, 100, 100),   # swimming pool - light blue
]


# ═══════════════════════════════════════════════════════════════════════════
# Detection Overlay
# ═══════════════════════════════════════════════════════════════════════════

def _scale_factor(image: np.ndarray) -> float:
    """Compute annotation scale factor based on image resolution."""
    h = image.shape[0] if len(image.shape) >= 2 else 1024
    return max(h / 1024.0, 1.0)


def draw_detections(image: np.ndarray, detections: List,
                    show_labels: bool = True) -> np.ndarray:
    """
    Draw oriented bounding boxes on image with resolution-aware sizing.
    """
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    sf = _scale_factor(vis)
    line_w = max(2, int(3 * sf))
    font_scale = 0.5 * sf
    font_thick = max(1, int(1.5 * sf))

    for det in detections:
        color = CLASS_COLORS[det.class_id % len(CLASS_COLORS)]
        corners = det.corners.astype(np.int32)

        # Glow effect: draw a wider semi-transparent outline behind the box
        cv2.polylines(vis, [corners], True, (255, 255, 255), line_w + 4)
        cv2.polylines(vis, [corners], True, color, line_w)

        # Corner markers for emphasis
        for ci in range(4):
            cv2.circle(vis, tuple(corners[ci]), max(3, int(4 * sf)), color, -1)

        if show_labels:
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                           font_scale, font_thick)
            cx, cy = int(det.cx), int(det.cy)
            pad = int(4 * sf)
            cv2.rectangle(vis, (cx - tw//2 - pad, cy - th - pad*2),
                         (cx + tw//2 + pad, cy - pad//2), color, -1)
            cv2.putText(vis, label, (cx - tw//2, cy - pad),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       (255, 255, 255), font_thick)

    return vis


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory Visualization
# ═══════════════════════════════════════════════════════════════════════════

def draw_trajectories(image: np.ndarray, trajectories: Dict,
                      risk_results: Optional[Dict] = None,
                      max_trail: int = 50) -> np.ndarray:
    """
    Draw object trajectories with risk-based coloring.
    
    Green trails = normal, Red trails = anomalous.
    """
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for tid, tdata in trajectories.items():
        traj = tdata["trajectory"]
        if len(traj) < 2:
            continue

        # Determine color based on risk
        if risk_results and tid in risk_results:
            risk_level = risk_results[tid]["risk_level"]
            color = RISK_COLORS.get(risk_level, (200, 200, 200))
        else:
            color = (200, 200, 200)

        sf = _scale_factor(vis)
        trail_thick_max = max(3, int(4 * sf))
        dot_r = max(6, int(10 * sf))
        font_scale = 0.5 * sf
        font_thick = max(1, int(1.5 * sf))

        # Draw trail
        points = [(int(pt["cx"]), int(pt["cy"])) for pt in traj[-max_trail:]]
        for i in range(1, len(points)):
            alpha = i / len(points)
            thickness = max(1, int(alpha * trail_thick_max))
            faded = tuple(int(c * alpha) for c in color)
            cv2.line(vis, points[i-1], points[i], faded, thickness)

        # Draw current position
        if points:
            cx, cy = points[-1]
            cv2.circle(vis, (cx, cy), dot_r, color, -1)
            cv2.circle(vis, (cx, cy), dot_r + 3, (255, 255, 255), max(1, int(sf)))

            # Label with track ID and risk
            label = f"T{tid}"
            if risk_results and tid in risk_results:
                rs = risk_results[tid]["risk_score"]
                label += f" [{rs:.2f}]"

            cv2.putText(vis, label, (cx + int(12 * sf), cy - int(5 * sf)),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thick)

    return vis


# ═══════════════════════════════════════════════════════════════════════════
# Spatial Anomaly Heatmap Visualization
# ═══════════════════════════════════════════════════════════════════════════

def draw_anomaly_heatmap(heatmap: np.ndarray,
                          image_size: Tuple[int, int] = (1024, 1024),
                          background: Optional[np.ndarray] = None,
                          alpha: float = 0.6) -> np.ndarray:
    """
    Render spatial anomaly heatmap as a colored overlay.
    
    Args:
        heatmap:     (grid_h, grid_w) normalized [0,1] anomaly map
        image_size:  Target output size (w, h)
        background:  Optional background image to overlay onto
        alpha:       Overlay transparency
        
    Returns:
        Visualization image (H, W, 3) uint8
    """
    # Upscale heatmap to image size
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap_uint8, image_size,
                                  interpolation=cv2.INTER_CUBIC)

    # Apply colormap (blue=low, red=high)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    if background is not None:
        bg = background.copy()
        if len(bg.shape) == 2:
            bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        if bg.shape[:2] != (image_size[1], image_size[0]):
            bg = cv2.resize(bg, image_size)

        # Blend: only overlay where heatmap is significant
        mask = (heatmap_resized > 20).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (15, 15), 5)
        mask = np.stack([mask] * 3, axis=-1)

        vis = (bg * (1 - mask * alpha) + heatmap_colored * mask * alpha).astype(np.uint8)
    else:
        vis = heatmap_colored

    return vis


# ═══════════════════════════════════════════════════════════════════════════
# Dashboard / Summary Panel
# ═══════════════════════════════════════════════════════════════════════════

def create_dashboard(frame: np.ndarray,
                     risk_results: Dict[int, Dict],
                     heatmap_vis: np.ndarray,
                     frame_id: int,
                     total_tracks: int) -> np.ndarray:
    """
    Create a multi-panel dashboard image.
    
    Layout:
    ┌───────────────┬───────────────┐
    │  Tracked      │   Anomaly     │
    │  Scene        │   Heatmap     │
    ├───────────────┴───────────────┤
    │       Risk Summary Panel      │
    └───────────────────────────────┘
    """
    panel_w, panel_h = 1024, 1024
    summary_h = 300

    # Resize panels
    scene_panel = cv2.resize(frame, (panel_w, panel_h))
    heatmap_panel = cv2.resize(heatmap_vis, (panel_w, panel_h))

    # Top row
    top_row = np.hstack([scene_panel, heatmap_panel])

    # Summary panel (dark background)
    summary = np.zeros((summary_h, panel_w * 2, 3), dtype=np.uint8)
    summary[:] = (30, 30, 40)

    # Title
    cv2.putText(summary, "SPATIO-TEMPORAL ANOMALY DETECTION SYSTEM",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
    cv2.putText(summary, f"Frame: {frame_id}  |  Active Tracks: {total_tracks}",
                (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # Risk statistics
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for _, data in risk_results.items():
        level = data.get("risk_level", "LOW")
        risk_counts[level] = risk_counts.get(level, 0) + 1

    y_offset = 120
    for level, count in risk_counts.items():
        color = RISK_COLORS[level]
        bar_width = min(count * 60, 500)
        cv2.rectangle(summary, (30, y_offset), (30 + bar_width, y_offset + 28),
                     color, -1)
        cv2.putText(summary, f"{level}: {count}", (35, y_offset + 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 38

    # Top threats
    top_threats = sorted(risk_results.items(),
                         key=lambda x: x[1]["risk_score"], reverse=True)[:3]
    x_threats = 1100
    cv2.putText(summary, "TOP THREATS:", (x_threats, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    for i, (tid, data) in enumerate(top_threats):
        text = (f"Track {tid} ({data['class_name']}): "
                f"Risk={data['risk_score']:.3f} [{data['risk_level']}]")
        color = RISK_COLORS[data["risk_level"]]
        cv2.putText(summary, text, (x_threats, 160 + i * 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # DRDO watermark
    cv2.putText(summary, "Defense-Grade Surveillance Analytics",
                (x_threats, summary_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 120), 2)

    # Combine
    dashboard = np.vstack([top_row, summary])
    return dashboard


# ═══════════════════════════════════════════════════════════════════════════
# Save Utilities
# ═══════════════════════════════════════════════════════════════════════════

def save_visualization(image: np.ndarray, output_dir: str,
                       filename: str) -> str:
    """Save visualization image to disk."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)
    return filepath


def save_all_outputs(scene_vis: np.ndarray, heatmap_vis: np.ndarray,
                     dashboard: np.ndarray, output_dir: str,
                     frame_id: int) -> Dict[str, str]:
    """Save all visualization outputs for a frame."""
    paths = {}
    paths["scene"] = save_visualization(
        scene_vis, output_dir, f"scene_frame_{frame_id:04d}.png")
    paths["heatmap"] = save_visualization(
        heatmap_vis, output_dir, f"heatmap_frame_{frame_id:04d}.png")
    paths["dashboard"] = save_visualization(
        dashboard, output_dir, f"dashboard_frame_{frame_id:04d}.png")
    return paths
