"""
Change Detection Module
========================
Computes pixel-level and statistical change maps between consecutive
SAR frames in a multi-temporal satellite stack.

Used to identify:
- New construction / vehicle build-up
- Troop movement corridors
- Vegetation clearance (possible camp sites)
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple


def compute_change_maps(scenes: List[Dict]) -> List[Dict]:
    """
    Given a sorted list of scene dicts (oldest to newest),
    compute change detection between every consecutive pair.

    Returns list of change_result dicts:
        {
            "date_from":   str,
            "date_to":     str,
            "change_map":  np.ndarray (H×W uint8, 0=no change, 255=max change),
            "change_rgb":  np.ndarray (H×W×3 BGR heatmap),
            "change_pct":  float  (% pixels significantly changed),
            "hotspot_mask": np.ndarray (bool H×W),
        }
    """
    results = []
    for i in range(len(scenes) - 1):
        a = scenes[i]
        b = scenes[i + 1]
        res = _compute_single_change(a, b)
        results.append(res)
    return results


def _compute_single_change(scene_a: Dict, scene_b: Dict) -> Dict:
    sar_a = _ensure_gray(scene_a["sar"])
    sar_b = _ensure_gray(scene_b["sar"])

    # Align sizes
    if sar_a.shape != sar_b.shape:
        sar_b = cv2.resize(sar_b, (sar_a.shape[1], sar_a.shape[0]))

    # 1. Log-ratio change detector (standard for SAR)
    a_f = sar_a.astype(np.float32) + 1.0
    b_f = sar_b.astype(np.float32) + 1.0
    log_ratio = np.abs(np.log(b_f / a_f))

    # Normalise to 0-255
    lo, hi = np.percentile(log_ratio, 2), np.percentile(log_ratio, 98)
    change_map = np.clip((log_ratio - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)

    # Smooth to remove speckle noise
    change_map = cv2.GaussianBlur(change_map, (9, 9), 3)

    # Threshold for significant change (top 15%)
    threshold = np.percentile(change_map, 85)
    hotspot_mask = change_map > threshold

    change_pct = float(hotspot_mask.sum()) / hotspot_mask.size * 100.0

    # 2. Colour the change map (blue=low, yellow=medium, red=high)
    change_rgb = _colorize_change(change_map, hotspot_mask)

    return {
        "date_from":    scene_a["date"],
        "date_to":      scene_b["date"],
        "change_map":   change_map,
        "change_rgb":   change_rgb,
        "change_pct":   round(change_pct, 2),
        "hotspot_mask": hotspot_mask,
    }


def _colorize_change(change_map: np.ndarray,
                      hotspot_mask: np.ndarray) -> np.ndarray:
    """Apply a traffic-light colour scheme to the change map."""
    norm = change_map.astype(np.float32) / 255.0
    h, w = norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Low change → dark blue
    low  = (norm < 0.33)
    mid  = (norm >= 0.33) & (norm < 0.66)
    high = (norm >= 0.66)

    rgb[low,  0] = norm[low]  * 80
    rgb[low,  2] = norm[low]  * 220

    rgb[mid,  0] = norm[mid]  * 50
    rgb[mid,  1] = norm[mid]  * 200
    rgb[mid,  2] = norm[mid]  * 100

    rgb[high, 0] = norm[high] * 50
    rgb[high, 1] = norm[high] * 80
    rgb[high, 2] = norm[high] * 255     # red channel in BGR

    # Boost hotspot outline
    contours, _ = cv2.findContours(
        hotspot_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mask_img = rgb.astype(np.uint8)
    cv2.drawContours(mask_img, contours, -1, (0, 0, 255), 2)
    return mask_img


def compute_temporal_stats(scenes: List[Dict],
                            change_maps: List[Dict]) -> Dict:
    """
    Compute summary statistics across the full temporal stack.
    Returns a dict suitable for display/PDF reporting.
    """
    if not scenes:
        return {}

    dates     = [s["date"] for s in scenes]
    platforms = list({s["platform"] for s in scenes})
    sources   = list({s["source"]   for s in scenes})

    change_pcts = [c["change_pct"] for c in change_maps] if change_maps else [0]
    max_change_idx = int(np.argmax(change_pcts)) if change_pcts else 0
    max_change_pair = (
        f"{change_maps[max_change_idx]['date_from']} → "
        f"{change_maps[max_change_idx]['date_to']}"
    ) if change_maps else "N/A"

    return {
        "n_scenes":        len(scenes),
        "date_range_start": dates[0],
        "date_range_end":   dates[-1],
        "platforms":        platforms,
        "sources":          sources,
        "mean_change_pct":  round(float(np.mean(change_pcts)), 2),
        "max_change_pct":   round(float(np.max(change_pcts)), 2),
        "max_change_pair":  max_change_pair,
        "total_days":       (
            scenes[-1]["date_obj"] - scenes[0]["date_obj"]
        ).days if len(scenes) > 1 else 0,
    }


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
