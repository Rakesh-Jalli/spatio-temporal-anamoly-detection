"""
YOLOv8-OBB Oriented Object Detector Module.

Runs REAL YOLOv8-OBB inference with sliding-window tiling for
high-resolution aerial imagery (NAIP 0.6m).

Tiling is essential: a 2048x2048 image downscaled to 640x640 loses
all small objects. Tiling preserves full resolution per tile.
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# DOTA-v1.0 class names (pretrained YOLOv8-OBB)
DOTA_CLASSES = [
    "plane", "ship", "storage tank", "baseball diamond",
    "tennis court", "basketball court", "ground track field",
    "harbor", "bridge", "large vehicle", "small vehicle",
    "helicopter", "roundabout", "soccer ball field", "swimming pool",
]


@dataclass
class OBBDetection:
    """Single Oriented Bounding Box detection."""
    class_id: int
    class_name: str
    confidence: float
    cx: float
    cy: float
    width: float
    height: float
    angle: float
    corners: np.ndarray      # (4, 2) corner points

    def to_dict(self) -> Dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "cx": self.cx, "cy": self.cy,
            "width": self.width, "height": self.height,
            "angle": self.angle,
            "corners": self.corners.tolist(),
        }


def _obb_iou(det_a: OBBDetection, det_b: OBBDetection) -> float:
    """Approximate IoU between two OBB detections using rotated rect intersection."""
    rect_a = ((det_a.cx, det_a.cy), (det_a.width, det_a.height), det_a.angle)
    rect_b = ((det_b.cx, det_b.cy), (det_b.width, det_b.height), det_b.angle)
    ret, pts = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    if ret == cv2.INTERSECT_NONE or pts is None:
        return 0.0
    inter = cv2.contourArea(pts.astype(np.float32))
    area_a = max(det_a.width * det_a.height, 1e-6)
    area_b = max(det_b.width * det_b.height, 1e-6)
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


def _nms_obb(detections: List[OBBDetection], iou_thresh: float = 0.45) -> List[OBBDetection]:
    """Non-Maximum Suppression for OBB detections (cross-tile dedup)."""
    if len(detections) <= 1:
        return detections
    dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep = []
    for d in dets:
        suppressed = False
        for k in keep:
            if d.class_id == k.class_id and _obb_iou(d, k) > iou_thresh:
                suppressed = True
                break
        if not suppressed:
            keep.append(d)
    return keep


class YOLOv8OBBDetector:
    """
    YOLOv8-OBB detector with sliding-window tiling for high-res imagery.

    For images larger than tile_size, splits into overlapping tiles,
    runs inference per tile, remaps coordinates to full image, and
    applies NMS to remove duplicates from overlap zones.
    """

    def __init__(self, config, demo_mode: bool = False):
        self.config = config
        self.model = None
        self.class_names = DOTA_CLASSES
        # Tiling config: 640 tiles preserve full resolution for small objects
        # Large ships/carriers handled by contour detector (no YOLO pass needed)
        self.tile_size = 640
        self.tile_overlap = 160
        # Vehicle-focused fine-grained tiling (smaller tiles = better for tiny cars)
        self.vehicle_tile_size = 416
        self.vehicle_tile_overlap = 104
        self.vehicle_classes = {"small vehicle", "large vehicle"}

        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8s-obb.pt")
            logger.info("Loaded YOLOv8s-OBB (DOTA pretrained, small variant)")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8-OBB model: {e}")

    def detect(self, image: np.ndarray,
               gt_annotations: Optional[List[Dict]] = None) -> List[OBBDetection]:
        if self.model is None:
            return []
        h, w = image.shape[:2]
        if max(h, w) <= 640:
            return self._infer_single(image, offset_x=0, offset_y=0, imgsz=640)
        return self._infer_multiscale(image)

    def _infer_single(self, image: np.ndarray,
                      offset_x: int = 0, offset_y: int = 0,
                      imgsz: int = 640,
                      conf_override: Optional[float] = None,
                      enhance: bool = False) -> List[OBBDetection]:
        """Run inference on a single image/tile and remap coordinates."""
        conf = conf_override if conf_override is not None else self.config.confidence_threshold
        inp = image
        if enhance and len(image.shape) == 3:
            # CLAHE contrast enhancement helps reveal tiny vehicles
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            inp = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        results = self.model.predict(
            inp,
            conf=conf,
            iou=self.config.iou_threshold,
            imgsz=imgsz,
            max_det=500,
            device="cpu",
            verbose=False,
        )
        detections = []
        for result in results:
            if result.obb is None:
                continue
            for i in range(len(result.obb)):
                obb = result.obb[i]
                cls_id = int(obb.cls[0])
                conf = float(obb.conf[0])
                xywhr = obb.xywhr[0].cpu().numpy()
                cx, cy, bw, bh, angle_rad = xywhr

                # Remap to full image coordinates
                cx += offset_x
                cy += offset_y
                angle_deg = np.degrees(angle_rad)

                rect = ((cx, cy), (bw, bh), angle_deg)
                corners = cv2.boxPoints(rect).astype(np.float32)

                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"

                detections.append(OBBDetection(
                    class_id=cls_id, class_name=cls_name,
                    confidence=conf,
                    cx=cx, cy=cy, width=bw, height=bh,
                    angle=angle_deg, corners=corners,
                ))
        return detections

    def _infer_tiled(self, image: np.ndarray, tile_size: int = 640,
                     overlap: int = 128, min_dim: int = 64,
                     conf_override: Optional[float] = None,
                     enhance: bool = False) -> List[OBBDetection]:
        """Sliding-window tiled inference at a given tile size."""
        h, w = image.shape[:2]
        stride = tile_size - overlap
        all_dets = []

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x2 = min(x + tile_size, w)
                y2 = min(y + tile_size, h)
                if (x2 - x) < min_dim or (y2 - y) < min_dim:
                    continue
                tile = image[y:y2, x:x2]
                tile_dets = self._infer_single(tile, offset_x=x, offset_y=y,
                                               imgsz=tile_size,
                                               conf_override=conf_override,
                                               enhance=enhance)
                all_dets.extend(tile_dets)

        return all_dets

    def _build_water_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Build a strict water mask using COLOR ONLY.
        No texture/smoothness — runways, tarmac, parking lots are also smooth
        but they are NOT water.
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Blue water (ocean, clear harbors)
        water_blue = cv2.inRange(hsv, (90, 30, 20), (135, 255, 220))

        # Teal/green water (shallow, murky)
        water_teal = cv2.inRange(hsv, (70, 25, 20), (95, 255, 200))

        # Very dark areas with low saturation near blue hue (deep harbor water)
        water_dark = cv2.inRange(hsv, (80, 10, 5), (140, 200, 70))

        # Combine color-based detections only
        water = cv2.bitwise_or(water_blue, water_teal)
        water = cv2.bitwise_or(water, water_dark)

        # Clean up noise
        water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        water = cv2.morphologyEx(water, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))

        # Remove small blobs (windows, reflections, etc.)
        contours, _ = cv2.findContours(water, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        water_clean = np.zeros_like(water)
        min_water_area = max(h * w * 0.002, 2000)  # At least 0.2% of image
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_water_area:
                cv2.drawContours(water_clean, [cnt], -1, 255, -1)

        return water_clean

    def _detect_large_vessels(self, image: np.ndarray,
                              yolo_ships: List[OBBDetection]) -> List[OBBDetection]:
        """
        Contour-based large vessel detector using YOLO ship anchors.

        DOTA-pretrained models miss aircraft carriers and large warships
        because they're far larger than typical DOTA ship training examples.

        Strategy: use YOLO-detected small ships as maritime context anchors,
        then detect water via texture (smooth areas) near those anchors,
        and find large elongated structures at water-land boundaries.
        """
        if len(yolo_ships) < 3:
            return []

        h, w = image.shape[:2]

        # Strict color-only water mask
        water = self._build_water_mask(image)

        # --- Maritime zone: area around YOLO ship detections that are on water ---
        maritime_mask = np.zeros((h, w), dtype=np.uint8)
        water_dilated_check = cv2.dilate(water, np.ones((60, 60), np.uint8))
        water_anchored_ships = [s for s in yolo_ships
                                if water_dilated_check[int(min(s.cy, h-1)), int(min(s.cx, w-1))] > 0]
        if len(water_anchored_ships) < 2:
            return []
        for s in water_anchored_ships:
            cv2.circle(maritime_mask, (int(s.cx), int(s.cy)), 250, 255, -1)

        # Water within maritime zone
        maritime_water = cv2.bitwise_and(water, maritime_mask)
        maritime_water_pct = np.sum(maritime_water > 0) / max(np.sum(maritime_mask > 0), 1)
        if maritime_water_pct < 0.10:
            return []  # Not enough smooth/water area near ships

        # --- Find structures at water-land boundaries within maritime zone ---
        water_dilated = cv2.dilate(maritime_water, np.ones((60, 60), np.uint8))
        candidates = cv2.bitwise_and(
            cv2.bitwise_not(maritime_water),
            water_dilated,
        )
        candidates = cv2.bitwise_and(candidates, maritime_mask)
        candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
        candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ship_cls_id = DOTA_CLASSES.index("ship")
        detections = []
        margin = 10

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5000 or area > 150000:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            if min(rw, rh) < 15:
                continue
            aspect = max(rw, rh) / max(min(rw, rh), 1)
            if aspect < 2.0 or aspect > 12:
                continue

            corners = cv2.boxPoints(rect).astype(np.float32)
            # Skip contours touching image edges
            if (np.any(corners < margin) or
                    np.any(corners[:, 0] > w - margin) or
                    np.any(corners[:, 1] > h - margin)):
                continue

            # Proximity to nearest YOLO ship (closer = higher confidence)
            min_dist = min(np.hypot(cx - s.cx, cy - s.cy) for s in yolo_ships)
            if min_dist > 500:
                continue  # Too far from any detected ship

            # Confidence: elongation + size + proximity
            shape_score = min(aspect / 5.0, 1.0)
            size_score = min(area / 30000, 1.0)
            prox_score = max(1.0 - min_dist / 500.0, 0.0)
            conf = 0.30 + 0.30 * (shape_score * 0.4 + size_score * 0.3 + prox_score * 0.3)

            detections.append(OBBDetection(
                class_id=ship_cls_id,
                class_name="ship",
                confidence=round(conf, 2),
                cx=cx, cy=cy, width=rw, height=rh,
                angle=angle, corners=corners,
            ))

        logger.info(f"Large vessel detector: {len(detections)} candidates "
                     f"(maritime water: {maritime_water_pct:.1%})")
        return detections

    def _filter_ships_by_water(self, image: np.ndarray,
                               detections: List[OBBDetection]) -> List[OBBDetection]:
        """
        Remove false ship detections not near actual water.
        Uses strict COLOR-ONLY water detection — no texture/smoothness.
        Ships below 0.20 confidence are always removed (almost always false).
        Ships 0.20-0.40 must be directly ON water pixels.
        Ships >0.40 can be within 40px of water (docked at pier).
        """
        ships = [d for d in detections if d.class_name == "ship"]
        non_ships = [d for d in detections if d.class_name != "ship"]
        if not ships:
            return detections

        h, w = image.shape[:2]

        # Strict color-only water mask
        water = self._build_water_mask(image)

        # Two proximity levels: direct water and near-water
        water_direct = cv2.dilate(water, np.ones((10, 10), np.uint8))
        water_near = cv2.dilate(water, np.ones((40, 40), np.uint8))

        kept_ships = []
        for s in ships:
            # Drop very low confidence ships outright
            if s.confidence < 0.20:
                logger.debug(f"Dropped low-conf ship at ({s.cx:.0f}, {s.cy:.0f}) "
                             f"conf={s.confidence:.2f}")
                continue

            cy_int = int(min(max(s.cy, 0), h - 1))
            cx_int = int(min(max(s.cx, 0), w - 1))

            if s.confidence >= 0.40:
                # High-conf ships: allowed within 40px of water (docked)
                if water_near[cy_int, cx_int] > 0:
                    kept_ships.append(s)
                else:
                    logger.debug(f"Filtered ship at ({s.cx:.0f}, {s.cy:.0f}) "
                                 f"conf={s.confidence:.2f} — not near water")
            else:
                # Medium-conf ships: must be directly on water
                if water_direct[cy_int, cx_int] > 0:
                    kept_ships.append(s)
                else:
                    logger.debug(f"Filtered med-conf ship at ({s.cx:.0f}, {s.cy:.0f}) "
                                 f"conf={s.confidence:.2f} — not on water")

        removed = len(ships) - len(kept_ships)
        if removed > 0:
            logger.info(f"Water filter: removed {removed}/{len(ships)} false ship detections")

        return non_ships + kept_ships

    def _infer_multiscale(self, image: np.ndarray) -> List[OBBDetection]:
        """
        Hybrid multi-scale inference combining:
          1) Tiled 640x640  – catches small/medium objects at full resolution
          2) Fine-grained 416x416 – vehicle-focused pass with lower conf for tiny cars
          3) Full-image at 2048 – catches larger structures
          4) Contour-based water analysis – catches very large warships/carriers
             (only activated when YOLO detects ships — maritime context)
        All scales merged with NMS deduplication.
        """
        all_dets = []

        # Scale 1: Tiled inference at 640
        tiled_dets = self._infer_tiled(
            image,
            tile_size=self.tile_size,
            overlap=self.tile_overlap,
            min_dim=64,
        )
        all_dets.extend(tiled_dets)

        # Scale 2: Fine-grained vehicle-focused pass at 416px tiles
        # Smaller tiles = higher effective resolution for tiny cars
        # Lower confidence + CLAHE enhancement to catch faint vehicles
        vehicle_conf = max(self.config.confidence_threshold - 0.06, 0.06)
        fine_dets = self._infer_tiled(
            image,
            tile_size=self.vehicle_tile_size,
            overlap=self.vehicle_tile_overlap,
            min_dim=48,
            conf_override=vehicle_conf,
            enhance=True,
        )
        # Keep only vehicle classes from fine-grained pass to avoid duplicates
        for d in fine_dets:
            if d.class_name in self.vehicle_classes:
                all_dets.append(d)

        # Scale 2b: Ultra-fine 320px pass for dense parking lots
        ultra_conf = max(self.config.confidence_threshold - 0.07, 0.05)
        ultra_dets = self._infer_tiled(
            image,
            tile_size=320,
            overlap=96,
            min_dim=32,
            conf_override=ultra_conf,
            enhance=True,
        )
        for d in ultra_dets:
            if d.class_name in self.vehicle_classes:
                all_dets.append(d)

        # Scale 3: Full-image at 2048 to catch large structures
        h, w = image.shape[:2]
        full_dets = self._infer_single(
            image, offset_x=0, offset_y=0,
            imgsz=min(2048, max(h, w)),
            conf_override=max(self.config.confidence_threshold - 0.03, 0.10),
        )
        all_dets.extend(full_dets)

        # Scale 4: Contour-based large vessel detection —
        # Only run if YOLO found ship detections (confirms maritime context)
        yolo_ships = [d for d in all_dets if d.class_name == "ship" and d.confidence >= 0.30]
        if len(yolo_ships) >= 3:
            contour_dets = self._detect_large_vessels(image, yolo_ships)
            all_dets.extend(contour_dets)

        # Filter false ship detections on land using water proximity
        all_dets = self._filter_ships_by_water(image, all_dets)

        # Global NMS across all scales — use slightly lower IoU for vehicles
        # to preserve nearby but distinct car detections
        vehicle_dets = [d for d in all_dets if d.class_name in self.vehicle_classes]
        other_dets = [d for d in all_dets if d.class_name not in self.vehicle_classes]
        vehicle_dets = _nms_obb(vehicle_dets, iou_thresh=max(self.config.iou_threshold - 0.05, 0.35))
        other_dets = _nms_obb(other_dets, iou_thresh=self.config.iou_threshold)
        all_dets = vehicle_dets + other_dets

        h, w = image.shape[:2]
        cls_counts = {}
        for d in all_dets:
            cls_counts[d.class_name] = cls_counts.get(d.class_name, 0) + 1
        logger.info(f"Hybrid detection: {len(all_dets)} objects on {w}x{h} | {cls_counts}")
        return all_dets

    def __repr__(self):
        return (f"YOLOv8OBBDetector(tiled, "
                f"classes={len(self.class_names)}, "
                f"conf={self.config.confidence_threshold})")
