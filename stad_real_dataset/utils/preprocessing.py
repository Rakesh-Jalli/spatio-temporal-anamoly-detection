"""
Preprocessing module for SAR and Optical satellite imagery.

Handles:
  - Geography-aware region classification
  - SAR speckle filtering
  - Image normalization & resizing
  - Multi-modal fusion (SAR + Optical)
"""

import numpy as np
import cv2
import os
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data preprocessing — geography classification and image fusion
# ═══════════════════════════════════════════════════════════════════════════

# ── Geography-aware class pools ──────────────────────────────────────────
# Ships only appear when there is water; inland regions never spawn ships.

_MARITIME_KEYWORDS = [
    "spratly", "paracel",
    "open ocean", "mid-ocean", "deep sea",
]
_COASTAL_KEYWORDS = [
    "mumbai", "chennai", "kolkata", "visakhapatnam", "vizag",
    "kochi", "goa", "karachi", "colombo", "dhaka", "chittagong",
    "shanghai", "hong kong", "singapore", "istanbul", "cyprus",
    "hormuz", "aden", "suez", "malacca",
    "coastal", "littoral", "beach", "shore", "delta",
    "gujarat", "canal", "egypt", "uae", "persian", "gulf",
    "sea", "ocean", "strait", "bay", "channel", "harbor",
    "harbour", "port", "island", "andaman", "taiwan", "creek",
    "mediterranean", "mandeb", "red sea", "naval",
]
_MOUNTAIN_KEYWORDS = [
    "ladakh", "siachen", "glacier", "doklam", "sikkim",
    "arunachal", "himalaya", "karakoram", "mountain", "tawang",
]
_LAND_BORDER_KEYWORDS = [
    "dmz", "korea", "donbas", "ukraine", "russia",
    "border", "bihar", "manipur", "bangladesh", "nepal",
    "pakistan", "punjab",
]
_DESERT_KEYWORDS = [
    "rajasthan", "thar", "desert",
]

ALL_CLASS_NAMES = [
    "vehicle", "aircraft", "ship", "armored_vehicle",
    "personnel", "helicopter", "radar_installation", "missile_launcher",
    "uav_drone", "submarine", "artillery", "supply_depot",
    "bunker", "airstrip", "naval_vessel", "tank",
]

# Class pools per region type
# NOTE: "ship" and "naval_vessel" appear ONLY in the maritime pool.
# Coastal regions have predominantly land units with limited naval presence.
_CLASS_POOLS = {
    "land":     ["vehicle", "aircraft", "armored_vehicle", "personnel",
                 "helicopter", "radar_installation", "missile_launcher",
                 "uav_drone", "artillery", "supply_depot", "tank"],
    "coastal":  ["ship", "naval_vessel", "vehicle", "aircraft", "armored_vehicle",
                 "personnel", "helicopter", "radar_installation", "missile_launcher",
                 "uav_drone", "supply_depot", "bunker"],
    "maritime": ["ship", "aircraft", "helicopter",
                 "submarine", "naval_vessel", "uav_drone"],
    "mountainous": ["vehicle", "armored_vehicle", "personnel",
                     "helicopter", "radar_installation", "missile_launcher",
                     "artillery", "bunker", "supply_depot"],
    "desert":   ["vehicle", "aircraft", "armored_vehicle", "personnel",
                 "helicopter", "missile_launcher", "tank",
                 "uav_drone", "artillery", "airstrip"],
}


def classify_region_type(region_name: str = "",
                         center_lat: float = 0.0,
                         center_lon: float = 0.0) -> str:
    """
    Classify a region as 'land', 'coastal', 'maritime', 'mountainous',
    or 'desert' using its name and coordinates so that only
    geographically plausible object classes are generated.

    Returns one of: 'land', 'coastal', 'maritime', 'mountainous', 'desert'.
    """
    name_lower = region_name.lower()

    # Strong maritime signal (open ocean only)
    if any(kw in name_lower for kw in _MARITIME_KEYWORDS):
        return "maritime"

    # Mountain/glacier signal (checked before coastal — mountains override)
    if any(kw in name_lower for kw in _MOUNTAIN_KEYWORDS):
        return "mountainous"

    # Coastal signal — ports, naval bases, harbours always get ships
    if any(kw in name_lower for kw in _COASTAL_KEYWORDS):
        return "coastal"

    # Land border regions (only if no coastal/mountain match)
    if any(kw in name_lower for kw in _LAND_BORDER_KEYWORDS):
        # Desert borders
        if any(kw in name_lower for kw in _DESERT_KEYWORDS):
            return "desert"
        return "land"

    # Desert signal
    if any(kw in name_lower for kw in _DESERT_KEYWORDS):
        return "desert"

    # Heuristic – regions whose bbox is mostly water are rare in our
    # predefined list, so default to 'land' for any inland search.
    return "land"


# ═══════════════════════════════════════════════════════════════════════════
# Image Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def apply_lee_filter(sar_image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Lee speckle filter for SAR imagery.
    Reduces multiplicative speckle while preserving edges.
    """
    img = sar_image.astype(np.float64)
    mean = cv2.blur(img, (kernel_size, kernel_size))
    sq_mean = cv2.blur(img ** 2, (kernel_size, kernel_size))
    variance = sq_mean - mean ** 2
    overall_var = np.var(img)

    weight = np.clip(variance / (variance + overall_var + 1e-10), 0, 1)
    filtered = mean + weight * (img - mean)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1] range."""
    img = image.astype(np.float32)
    min_val, max_val = img.min(), img.max()
    if max_val - min_val < 1e-6:
        return np.zeros_like(img)
    return (img - min_val) / (max_val - min_val)


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target (width, height)."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def enhance_optical(image: np.ndarray) -> np.ndarray:
    """Enhance optical satellite image for clear object visibility.
    Applies CLAHE contrast enhancement + unsharp mask sharpening."""
    if image is None or image.size == 0:
        return image
    img = image.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # CLAHE on L channel (preserves color, boosts local contrast)
    if len(img.shape) == 3 and img.shape[2] == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

    # Unsharp mask: sharpen edges to make objects pop
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return img


def enhance_sar(image: np.ndarray) -> np.ndarray:
    """Enhance SAR image — speckle reduction + contrast stretch."""
    if image is None or image.size == 0:
        return image
    img = image.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Lee filter for speckle
    img = apply_lee_filter(img, kernel_size=5)

    # CLAHE contrast stretch to highlight bright targets (ships, vehicles)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    img = clahe.apply(img)

    # Light bilateral filter — smooths speckle while preserving edges
    img = cv2.bilateralFilter(img, d=5, sigmaColor=40, sigmaSpace=40)
    return img


def fuse_sar_optical(sar: np.ndarray, optical: np.ndarray,
                     alpha: float = 0.5) -> np.ndarray:
    """
    Multi-modal fusion: blend SAR and Optical imagery.
    
    Converts SAR to 3-channel and performs weighted blending.
    """
    # ── Normalise SAR to 3-channel BGR ──────────────────────────────────────
    if len(sar.shape) == 2:
        sar_3ch = cv2.cvtColor(sar, cv2.COLOR_GRAY2BGR)
    elif sar.shape[2] == 1:
        sar_3ch = cv2.cvtColor(sar[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        sar_3ch = sar.copy()

    # ── Normalise optical to 3-channel BGR ──────────────────────────────────
    if len(optical.shape) == 2:
        optical = cv2.cvtColor(optical, cv2.COLOR_GRAY2BGR)
    elif optical.shape[2] == 1:
        optical = cv2.cvtColor(optical[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif optical.shape[2] == 4:
        optical = cv2.cvtColor(optical, cv2.COLOR_BGRA2BGR)

    # ── Align spatial dimensions ─────────────────────────────────────────────
    if sar_3ch.shape[:2] != optical.shape[:2]:
        optical = cv2.resize(optical, (sar_3ch.shape[1], sar_3ch.shape[0]))

    # ── Align dtype (addWeighted requires identical dtype) ───────────────────
    if sar_3ch.dtype != optical.dtype:
        optical = optical.astype(sar_3ch.dtype)

    fused = cv2.addWeighted(sar_3ch, alpha, optical, 1 - alpha, 0)
    return fused


def preprocess_frame(sar: np.ndarray, optical: np.ndarray,
                     target_size: Tuple[int, int] = (640, 640)
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for a single frame.
    
    Returns:
        sar_processed:  Enhanced SAR
        optical_resized: Enhanced optical
        fused:          Fused multi-modal image
    """
    sar_enhanced = enhance_sar(sar)
    sar_resized = resize_image(sar_enhanced, target_size)
    optical_enhanced = enhance_optical(optical)
    optical_resized = resize_image(optical_enhanced, target_size)
    fused = fuse_sar_optical(sar_resized, optical_resized)
    return sar_resized, optical_resized, fused
