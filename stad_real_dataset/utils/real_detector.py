"""
Real Object Detector — Pixel-Based
====================================
Detects REAL objects in satellite imagery using image processing:

 - **SAR (Sentinel-1):** Ships appear as bright radar targets against
   dark ocean.  Uses CFAR-like adaptive thresholding to find them.
 - **Optical (Sentinel-2):** Detects bright/distinct objects on water
   (ships) and structured objects on land (infrastructure, vehicles)
   using contrast analysis and morphological operations.

No neural network needed — pure OpenCV image processing tuned for
10m-resolution Sentinel imagery where ships are 10-40 pixels long.
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def detect_real_objects_sar(
    sar_img: np.ndarray,
    water_mask: Optional[np.ndarray] = None,
    min_area: int = 8,
    max_area: int = 2000,
    cfar_guard: int = 5,
    cfar_bg: int = 20,
    cfar_threshold: float = 2.5,
) -> List[Dict]:
    """Detect bright targets in SAR imagery using CFAR-like detection.

    Ships in Sentinel-1 SAR appear as very bright scatterers against
    the dark ocean background.  This uses local adaptive thresholding
    to find pixels significantly brighter than their surroundings.

    Parameters
    ----------
    sar_img : ndarray (H, W) uint8 grayscale SAR image.
    water_mask : ndarray (H, W) bool, optional.  If given, only detect on water.
    min_area, max_area : pixel area bounds for valid detections.
    cfar_guard : guard band radius (pixels to skip around test cell).
    cfar_bg : background window radius for computing local stats.
    cfar_threshold : number of std-devs above local mean to flag a target.

    Returns
    -------
    list of dict with keys: cx, cy, width, height, angle, class_name,
    confidence, class_id, source.
    """
    if sar_img is None or sar_img.size == 0:
        return []

    gray = sar_img.copy()
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    img_f = gray.astype(np.float32)

    # Compute local mean & std using box filters (fast CFAR approximation)
    ksize_bg = 2 * cfar_bg + 1
    ksize_guard = 2 * cfar_guard + 1

    mean_bg = cv2.blur(img_f, (ksize_bg, ksize_bg))
    mean_sq_bg = cv2.blur(img_f ** 2, (ksize_bg, ksize_bg))
    std_bg = np.sqrt(np.maximum(mean_sq_bg - mean_bg ** 2, 0) + 1e-6)

    # Adaptive threshold: pixel > local_mean + threshold * local_std
    target_map = img_f > (mean_bg + cfar_threshold * std_bg)

    # Also require the pixel to be reasonably bright in absolute terms
    target_map &= (gray > 40)

    # Mask to water-only if provided — erode to stay away from coastline
    if water_mask is not None:
        if water_mask.shape != target_map.shape:
            water_mask = cv2.resize(
                water_mask.astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        # Erode water mask by 25px to avoid coastline artifacts
        coast_buffer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        safe_water = cv2.erode(
            water_mask.astype(np.uint8) * 255, coast_buffer
        ) > 127
        target_map &= safe_water

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    target_u8 = target_map.astype(np.uint8) * 255
    target_u8 = cv2.morphologyEx(target_u8, cv2.MORPH_CLOSE, kernel)
    target_u8 = cv2.morphologyEx(target_u8, cv2.MORPH_OPEN, kernel)

    # Find connected components
    contours, _ = cv2.findContours(
        target_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # Fit oriented bounding box
        if len(cnt) < 5:
            rect = cv2.minAreaRect(cnt)
        else:
            rect = cv2.fitEllipse(cnt)
            # Convert ellipse to rect-like format
            rect = cv2.minAreaRect(cnt)

        (cx, cy), (bw, bh), angle = rect

        # Skip if too small or too large
        if bw < 3 or bh < 3:
            continue
        if bw > 200 or bh > 200:
            continue

        # Compute confidence from how bright relative to background
        roi_mask = np.zeros_like(gray)
        cv2.drawContours(roi_mask, [cnt], -1, 255, -1)
        mean_target = cv2.mean(img_f, mask=roi_mask)[0]
        local_cx, local_cy = int(np.clip(cx, 0, w-1)), int(np.clip(cy, 0, h-1))
        local_bg = mean_bg[local_cy, local_cx]
        local_std = std_bg[local_cy, local_cx]

        if local_std > 0:
            snr = (mean_target - local_bg) / local_std
            confidence = float(np.clip(0.3 + 0.1 * snr, 0.3, 0.98))
        else:
            confidence = 0.5

        # Determine class based on size and aspect ratio
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        major = max(bw, bh)

        if major > 15 and aspect > 2.0:
            cls_name = "ship"
            cls_id = 2
        elif major > 10:
            cls_name = "naval_vessel"
            cls_id = 14
        else:
            cls_name = "ship"
            cls_id = 2

        detections.append({
            "cx": float(cx),
            "cy": float(cy),
            "width": float(max(bw, bh)),
            "height": float(min(bw, bh)),
            "angle": float(angle),
            "class_name": cls_name,
            "class_id": cls_id,
            "confidence": confidence,
            "area": float(area),
            "source": "sar_cfar",
        })

    # Sort by confidence
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # Remove detections too close to image borders (edge artifacts)
    margin = 20
    detections = [d for d in detections
                  if margin < d["cx"] < w - margin
                  and margin < d["cy"] < h - margin]

    # STRICT: SAR detections (ships) MUST be on water
    if water_mask is not None:
        wm_check = water_mask.copy()
        if wm_check.shape != (h, w):
            wm_check = cv2.resize(wm_check.astype(np.uint8), (w, h),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
        detections = [d for d in detections
                      if wm_check[int(np.clip(d["cy"], 0, h-1)),
                                  int(np.clip(d["cx"], 0, w-1))]]

    logger.info(f"SAR CFAR: detected {len(detections)} real targets")
    return detections


def detect_real_objects_optical(
    optical_img: np.ndarray,
    water_mask: Optional[np.ndarray] = None,
    min_area: int = 10,
    max_area: int = 3000,
) -> List[Dict]:
    """Detect real objects in optical Sentinel-2 imagery.

    On water: finds bright/white objects (ships, wakes) against dark sea.
    On land: finds high-contrast structures (buildings, vehicles, infrastructure).

    Parameters
    ----------
    optical_img : ndarray (H, W, 3) BGR optical image.
    water_mask : ndarray (H, W) bool.
    min_area, max_area : pixel area bounds.

    Returns
    -------
    list of detection dicts.
    """
    if optical_img is None or optical_img.size == 0:
        return []

    img = optical_img.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    detections = []

    # ── Erode masks to create coastline buffer zone ──
    # This prevents false detections at the water/land boundary
    coast_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))

    # ── Water-based detection (ships) ──
    if water_mask is not None and np.any(water_mask):
        wm = water_mask.copy()
        if wm.shape != (h, w):
            wm = cv2.resize(wm.astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST).astype(bool)

        # Erode water mask — only detect ships well INSIDE water, not at coast
        safe_water = cv2.erode(
            wm.astype(np.uint8) * 255, coast_kernel
        ) > 127

        # Within safe water, find bright objects
        water_gray = gray.copy()
        water_gray[~safe_water] = 0

        # Adaptive threshold — objects brighter than local water background
        blur = cv2.GaussianBlur(water_gray.astype(np.float32), (51, 51), 0)
        bright = (water_gray.astype(np.float32) - blur) > 15
        bright &= safe_water
        bright &= (gray > 50)  # minimum brightness

        bright_u8 = bright.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright_u8 = cv2.morphologyEx(bright_u8, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            bright_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            rect = cv2.minAreaRect(cnt)
            (cx, cy), (bw, bh), angle = rect
            if bw < 3 or bh < 3:
                continue

            # Confidence from brightness contrast
            roi_m = np.zeros_like(gray)
            cv2.drawContours(roi_m, [cnt], -1, 255, -1)
            mean_obj = cv2.mean(gray, mask=roi_m)[0]
            lcx, lcy = int(np.clip(cx, 0, w-1)), int(np.clip(cy, 0, h-1))
            bg_val = blur[lcy, lcx]
            contrast = (mean_obj - bg_val) / (bg_val + 1e-6)
            conf = float(np.clip(0.4 + 0.5 * contrast, 0.3, 0.95))

            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            major = max(bw, bh)

            if major > 12 and aspect > 1.8:
                cls_name, cls_id = "ship", 2
            else:
                cls_name, cls_id = "naval_vessel", 14

            detections.append({
                "cx": float(cx), "cy": float(cy),
                "width": float(max(bw, bh)),
                "height": float(min(bw, bh)),
                "angle": float(angle),
                "class_name": cls_name, "class_id": cls_id,
                "confidence": conf, "area": float(area),
                "source": "optical_water",
            })

    # ── Land-based detection (infrastructure, vehicles) ──
    if water_mask is not None:
        land_mask = ~water_mask
        if land_mask.shape != (h, w):
            land_mask = cv2.resize(land_mask.astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
        # Erode land mask — only detect on land well AWAY from coastline
        safe_land = cv2.erode(
            land_mask.astype(np.uint8) * 255, coast_kernel
        ) > 127
    else:
        land_mask = np.ones((h, w), dtype=bool)
        safe_land = land_mask

    if np.any(safe_land):
        # Detect high-contrast edges on land (structured objects)
        edges = cv2.Canny(gray, 50, 150)
        edges[~safe_land] = 0

        # Dilate to connect nearby edge fragments
        kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges_d = cv2.dilate(edges, kernel_d, iterations=1)

        contours_land, _ = cv2.findContours(
            edges_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        land_dets = []
        for cnt in contours_land:
            area = cv2.contourArea(cnt)
            if area < 15 or area > 1500:
                continue

            rect = cv2.minAreaRect(cnt)
            (cx, cy), (bw, bh), angle = rect
            if bw < 4 or bh < 4 or max(bw, bh) > 100:
                continue

            # Check it's actually on safe land (away from coast)
            px, py = int(np.clip(cx, 0, w-1)), int(np.clip(cy, 0, h-1))
            if not safe_land[py, px]:
                continue

            # Compute local contrast as confidence proxy
            roi_m = np.zeros_like(gray)
            cv2.drawContours(roi_m, [cnt], -1, 255, -1)
            obj_std = cv2.meanStdDev(gray, mask=roi_m)[1][0][0]
            conf = float(np.clip(0.3 + obj_std / 80.0, 0.25, 0.85))

            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            major = max(bw, bh)

            # Classify by shape
            if aspect > 3.0 and major > 20:
                cls_name, cls_id = "airstrip", 13
            elif major > 15 and aspect < 2.0:
                cls_name, cls_id = "supply_depot", 11
            elif major > 8:
                cls_name, cls_id = "vehicle", 0
            else:
                cls_name, cls_id = "vehicle", 0

            land_dets.append({
                "cx": float(cx), "cy": float(cy),
                "width": float(max(bw, bh)),
                "height": float(min(bw, bh)),
                "angle": float(angle),
                "class_name": cls_name, "class_id": cls_id,
                "confidence": conf, "area": float(area),
                "source": "optical_land",
            })

        # Keep top detections by confidence (avoid flooding with edge noise)
        land_dets.sort(key=lambda d: d["confidence"], reverse=True)
        detections.extend(land_dets[:20])

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # Remove detections too close to image borders (edge artifacts)
    margin = 20
    detections = [d for d in detections
                  if margin < d["cx"] < w - margin
                  and margin < d["cy"] < h - margin]

    # STRICT terrain verification — ships MUST be on water, land objects on land
    if water_mask is not None:
        wm_check = water_mask.copy()
        if wm_check.shape != (h, w):
            wm_check = cv2.resize(wm_check.astype(np.uint8), (w, h),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
        verified = []
        for d in detections:
            px = int(np.clip(d["cx"], 0, w - 1))
            py = int(np.clip(d["cy"], 0, h - 1))
            is_water = wm_check[py, px]
            if d["source"] == "optical_water":
                # Ship/naval_vessel MUST be on water
                if is_water:
                    verified.append(d)
            else:
                # Land object MUST be on land
                if not is_water:
                    verified.append(d)
        detections = verified

    logger.info(f"Optical: detected {len(detections)} real targets")
    return detections


def merge_sar_optical_detections(
    sar_dets: List[Dict],
    opt_dets: List[Dict],
    iou_merge_threshold: float = 0.3,
    max_total: int = 30,
) -> List[Dict]:
    """Merge SAR and optical detections, removing duplicates.

    When a SAR detection overlaps an optical one (same real object
    seen in both sensors), keep the higher-confidence one.
    """
    if not sar_dets and not opt_dets:
        return []

    all_dets = []
    used_opt = set()

    for sd in sar_dets:
        merged = False
        for j, od in enumerate(opt_dets):
            if j in used_opt:
                continue
            dist = np.sqrt((sd["cx"] - od["cx"])**2 + (sd["cy"] - od["cy"])**2)
            max_dim = max(sd["width"], sd["height"], od["width"], od["height"])
            if dist < max_dim * 1.5:
                # Same object — keep higher confidence, note dual-sensor
                best = sd if sd["confidence"] > od["confidence"] else od
                best = dict(best)
                best["confidence"] = min(best["confidence"] + 0.1, 0.99)
                best["source"] = "sar+optical"
                all_dets.append(best)
                used_opt.add(j)
                merged = True
                break
        if not merged:
            all_dets.append(sd)

    for j, od in enumerate(opt_dets):
        if j not in used_opt:
            all_dets.append(od)

    all_dets.sort(key=lambda d: d["confidence"], reverse=True)
    return all_dets[:max_total]


def detect_all_real_objects(
    sar_img: np.ndarray,
    optical_img: np.ndarray,
    water_mask: Optional[np.ndarray] = None,
    max_total: int = 30,
) -> List[Dict]:
    """Run full real object detection on both SAR and optical.

    This is the main entry point. Returns merged, deduplicated
    detections of real objects found in the satellite imagery.
    """
    sar_dets = detect_real_objects_sar(sar_img, water_mask)
    opt_dets = detect_real_objects_optical(optical_img, water_mask)
    merged = merge_sar_optical_detections(sar_dets, opt_dets, max_total=max_total)

    logger.info(f"Real detection total: {len(merged)} objects "
                f"(SAR:{len(sar_dets)}, Optical:{len(opt_dets)})")
    return merged
