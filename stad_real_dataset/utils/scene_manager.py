"""
Real Scene Manager
==================
Manages pre-downloaded real satellite imagery and generates movement
annotations (coordinates only — NO synthetic chip drawing).

This module replaces the old ``TemporalChipInjector`` which pasted
simulated chip images onto backgrounds.  Now the pipeline:

  1.  Loads **genuine** Sentinel-1 SAR + Sentinel-2 Optical images
      (pre-downloaded via ``_download_real_scenes.py`` or fetched live).
  2.  Generates **movement annotations** — bounding-box coordinates &
      class labels for the demo detector — WITHOUT altering the pixel
      data of the real satellite image.
  3.  Returns ``(sar_frames, optical_frames, annotations)`` in the
      same format as the old injector so the rest of the pipeline
      (detection → tracking → anomaly scoring) works unchanged.

The displayed imagery is always 100 % real satellite data.
Detection labels come from the local class pool (used as ground-truth
annotations for the YOLOv8-OBB detector running in demo mode).
"""

from __future__ import annotations

import os
import json
import hashlib
import logging
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from utils.preprocessing import classify_region_type, ALL_CLASS_NAMES, _CLASS_POOLS
from utils.real_detector import detect_all_real_objects

logger = logging.getLogger(__name__)

# Default cache root for pre-downloaded scenes
SCENE_CACHE_ROOT = os.path.join("data", "real_scenes")

# Classes that MUST be on water pixels
WATER_CLASSES = {"ship", "submarine", "naval_vessel"}
# Classes that MUST be on land pixels
LAND_CLASSES = {"vehicle", "armored_vehicle", "personnel", "tank", "artillery",
                "supply_depot", "bunker", "radar_installation", "missile_launcher",
                "airstrip"}
# Classes that can appear anywhere (airborne)
AIR_CLASSES = {"aircraft", "helicopter", "uav_drone"}


def _compute_water_mask(optical_img: np.ndarray) -> np.ndarray:
    """Compute a boolean water mask from an optical satellite image.

    Uses multiple cues: dark pixels, blue-dominance, and HSV analysis
    to separate water from land in Sentinel-2 imagery.

    Returns
    -------
    mask : ndarray (H, W) bool — True where water is detected.
    """
    if optical_img is None:
        return np.zeros((512, 512), dtype=bool)

    if len(optical_img.shape) == 2:
        # Grayscale — water is typically very dark
        return optical_img < 50

    img = optical_img.copy()
    if img.shape[2] == 4:
        img = img[:, :, :3]

    # BGR channels
    b, g, r = img[:, :, 0].astype(np.float32), img[:, :, 1].astype(np.float32), img[:, :, 2].astype(np.float32)
    brightness = (b + g + r) / 3.0

    # Water cue 1: dark areas (deep water)
    dark_mask = brightness < 60

    # Water cue 2: blue-dominant (blue > red AND blue > green, moderate brightness)
    blue_dominant = (b > r + 10) & (b > g) & (brightness < 140)

    # Water cue 3: low-saturation dark-ish (harbours, murky water)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_val = hsv[:, :, 2] < 80
    low_sat_dark = low_val & (hsv[:, :, 1] < 80)

    # Combine: any cue triggers water
    water = dark_mask | blue_dominant | low_sat_dark

    # Morphological cleanup — remove noise and fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    water_u8 = water.astype(np.uint8) * 255
    water_u8 = cv2.morphologyEx(water_u8, cv2.MORPH_CLOSE, kernel)
    water_u8 = cv2.morphologyEx(water_u8, cv2.MORPH_OPEN, kernel)

    return water_u8 > 127


def _sample_position_on_mask(mask: np.ndarray, rng: np.random.RandomState,
                              margin: int = 40, max_attempts: int = 200
                              ) -> Optional[Tuple[float, float]]:
    """Sample a random (x, y) position where mask is True.

    Returns None if no valid position could be found.
    """
    h, w = mask.shape
    ys, xs = np.where(mask[margin:h-margin, margin:w-margin])
    if len(ys) == 0:
        return None
    idx = rng.randint(0, len(ys))
    return float(xs[idx] + margin), float(ys[idx] + margin)


# ═══════════════════════════════════════════════════════════════════════════
#  Tracked Object — maintains state across temporal frames  (annotation only)
# ═══════════════════════════════════════════════════════════════════════════

class _TrackedObject:
    """Simulates one target that moves across the temporal sequence.

    This generates *annotation coordinates* only — no pixel compositing
    is performed.  The coordinates drive the demo-mode detector, tracker,
    and anomaly scorer.
    """

    def __init__(self, obj_id: int, class_name: str, class_id: int,
                 x: float, y: float, vx: float, vy: float,
                 width: float, height: float, angle: float,
                 behaviour: str, rng: np.random.RandomState,
                 terrain: str = "any"):
        self.id = obj_id
        self.class_name = class_name
        self.class_id = class_id
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.width = width
        self.height = height
        self.angle = angle
        self.behaviour = behaviour
        self.rng = rng
        # "water", "land", or "any" — constrains where this object can be
        self.terrain = terrain

    def step(self, frame_idx: int, img_w: int, img_h: int,
             convergence_target: Tuple[float, float],
             water_mask: Optional[np.ndarray] = None):
        """Advance position by one temporal step.

        If *water_mask* is provided, the object is kept within its
        valid terrain zone (water for ships, land for vehicles, etc.).
        """
        # Save old position in case we need to revert
        old_x, old_y = self.x, self.y

        b = self.behaviour

        if b == "normal":
            self.vx += self.rng.uniform(-0.4, 0.4)
            self.vy += self.rng.uniform(-0.4, 0.4)

        elif b == "anomalous_converge":
            dx = convergence_target[0] - self.x
            dy = convergence_target[1] - self.y
            d = max(np.sqrt(dx ** 2 + dy ** 2), 1.0)
            self.vx = 4.0 * dx / d + self.rng.uniform(-0.5, 0.5)
            self.vy = 4.0 * dy / d + self.rng.uniform(-0.5, 0.5)

        elif b == "anomalous_loiter":
            cx, cy = convergence_target
            ang = frame_idx * 0.15 + self.id
            r = 80
            self.x = cx + r * np.cos(ang)
            self.y = cy + r * np.sin(ang)
            self.vx = -r * 0.15 * np.sin(ang)
            self.vy = r * 0.15 * np.cos(ang)
            self.x = np.clip(self.x, 30, img_w - 30)
            self.y = np.clip(self.y, 30, img_h - 30)
            if abs(self.vx) > 0.1 or abs(self.vy) > 0.1:
                self.angle = np.degrees(np.arctan2(self.vy, self.vx))
            self._enforce_terrain(water_mask, old_x, old_y, img_w, img_h)
            return

        elif b == "anomalous_speed":
            if frame_idx % 5 == 0:
                self.vx = self.rng.uniform(-20, 20)
                self.vy = self.rng.uniform(-20, 20)

        elif b == "anomalous_zigzag":
            if frame_idx % 3 == 0:
                self.vx = -self.vx + self.rng.uniform(-2, 2)
                self.vy += self.rng.uniform(-3, 3)

        self.x += self.vx
        self.y += self.vy
        self.x = np.clip(self.x, 40, img_w - 40)
        self.y = np.clip(self.y, 40, img_h - 40)

        if abs(self.vx) > 0.1 or abs(self.vy) > 0.1:
            self.angle = np.degrees(np.arctan2(self.vy, self.vx))

        self._enforce_terrain(water_mask, old_x, old_y, img_w, img_h)

    def _enforce_terrain(self, water_mask: Optional[np.ndarray],
                          old_x: float, old_y: float,
                          img_w: int, img_h: int):
        """Revert position if the object moved into wrong terrain."""
        if water_mask is None or self.terrain == "any":
            return
        px = int(np.clip(self.x, 0, img_w - 1))
        py = int(np.clip(self.y, 0, img_h - 1))
        is_water = water_mask[py, px]
        if self.terrain == "water" and not is_water:
            # Ship moved onto land — revert and bounce
            self.x, self.y = old_x, old_y
            self.vx = -self.vx + self.rng.uniform(-1, 1)
            self.vy = -self.vy + self.rng.uniform(-1, 1)
        elif self.terrain == "land" and is_water:
            # Vehicle moved onto water — revert and bounce
            self.x, self.y = old_x, old_y
            self.vx = -self.vx + self.rng.uniform(-1, 1)
            self.vy = -self.vy + self.rng.uniform(-1, 1)

    def annotation(self) -> dict:
        return {
            "track_id": self.id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "cx": self.x,
            "cy": self.y,
            "width": self.width,
            "height": self.height,
            "angle": self.angle,
            "vx": self.vx,
            "vy": self.vy,
            "behavior": self.behaviour,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Real Scene Manager
# ═══════════════════════════════════════════════════════════════════════════

class RealSceneManager:
    """
    Loads real satellite imagery and generates movement annotations.

    The images are returned **unmodified** — no synthetic chips or
    procedural textures are composited.  Detection boxes are drawn later
    by the visualisation layer.

    Parameters
    ----------
    region_type : str
        ``"land"`` | ``"coastal"`` | ``"maritime"``
    seed : int
        Deterministic seed for reproducibility.
    image_size : tuple
        (H, W) of output frames.
    num_objects : int
        Number of annotated targets per frame.
    """

    def __init__(
        self,
        region_type: str = "land",
        seed: int = 42,
        image_size: Tuple[int, int] = (512, 512),
        num_objects: int = 12,
    ):
        self.region_type = region_type
        self.seed = seed
        self.image_size = image_size
        self.num_objects = num_objects
        self.rng = np.random.RandomState(seed)

        self._objects = self._init_objects()

    # ------------------------------------------------------------------
    def _allowed_classes(self) -> List[str]:
        """Return class names allowed for this region type."""
        return _CLASS_POOLS.get(self.region_type, _CLASS_POOLS["land"])

    def _init_objects(self, water_mask: Optional[np.ndarray] = None) -> List[_TrackedObject]:
        allowed = self._allowed_classes()
        behaviours = (
            ["normal"] * 8
            + ["anomalous_converge", "anomalous_loiter",
               "anomalous_speed", "anomalous_zigzag"]
        )
        h, w = self.image_size

        # Pre-compute land mask (inverse of water)
        if water_mask is not None:
            land_mask = ~water_mask
            has_water = np.any(water_mask)
            has_land = np.any(land_mask)
        else:
            land_mask = None
            has_water = False
            has_land = True

        objs: List[_TrackedObject] = []
        for i in range(self.num_objects):
            cls_name = allowed[i % len(allowed)]
            cls_id = ALL_CLASS_NAMES.index(cls_name) if cls_name in ALL_CLASS_NAMES else 0

            # Determine terrain constraint for this class
            if cls_name in WATER_CLASSES:
                terrain = "water"
            elif cls_name in LAND_CLASSES:
                terrain = "land"
            else:
                terrain = "any"

            # Pick initial position on the correct terrain
            pos = None
            if water_mask is not None and terrain == "water" and has_water:
                pos = _sample_position_on_mask(water_mask, self.rng)
            elif water_mask is not None and terrain == "land" and has_land:
                pos = _sample_position_on_mask(land_mask, self.rng)

            if pos is not None:
                x, y = pos
            elif terrain == "water" and not has_water:
                # No water in this region — skip water-class objects entirely
                continue
            elif terrain == "land" and not has_land:
                # No land in this region — skip land-class objects entirely
                continue
            else:
                x = self.rng.uniform(80, w - 80)
                y = self.rng.uniform(80, h - 80)

            obj = _TrackedObject(
                obj_id=i,
                class_name=cls_name,
                class_id=cls_id,
                x=x,
                y=y,
                vx=self.rng.uniform(-8, 8),
                vy=self.rng.uniform(-8, 8),
                width=self.rng.uniform(24, 48),
                height=self.rng.uniform(18, 38),
                angle=self.rng.uniform(0, 360),
                behaviour=behaviours[i % len(behaviours)],
                rng=np.random.RandomState(self.rng.randint(0, 2 ** 31)),
                terrain=terrain,
            )
            objs.append(obj)
        return objs

    # ------------------------------------------------------------------
    #  Public API — process real temporal scenes (annotation only)
    # ------------------------------------------------------------------
    def annotate_temporal_sequence(
        self,
        temporal_scenes: List[dict],
        num_objects: Optional[int] = None,
        use_real_detection: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[dict]]]:
        """
        Generate movement annotations over a real satellite sequence.

        The SAR/Optical images in *temporal_scenes* are returned
        **unmodified**.  Only the annotation list carries generated
        coordinates and class labels.

        Parameters
        ----------
        temporal_scenes : list of dict
            Each dict must have ``"sar"`` and ``"optical"`` numpy arrays.
        num_objects : int, optional
            Override object count.
        use_real_detection : bool
            If True, detect REAL objects in each frame using pixel-based
            CFAR (SAR) and contrast analysis (optical) instead of
            generating synthetic random annotations.

        Returns
        -------
        sar_frames : list[ndarray]   — original real SAR images (unchanged)
        optical_frames : list[ndarray] — original real optical images (unchanged)
        all_annotations : list[list[dict]] — movement annotations per frame
        """
        # Compute water mask from first optical frame for terrain awareness
        first_optical = temporal_scenes[0].get("optical") if temporal_scenes else None
        water_mask = _compute_water_mask(first_optical) if first_optical is not None else None

        sar_frames: List[np.ndarray] = []
        opt_frames: List[np.ndarray] = []
        all_annots: List[List[dict]] = []

        if use_real_detection:
            # ── REAL DETECTION MODE ──
            # Detect actual objects in each frame from pixel data
            for idx, scene in enumerate(temporal_scenes):
                sar = scene["sar"]
                optical = scene["optical"]
                sar_frames.append(sar)
                opt_frames.append(optical)

                # Run pixel-based detection on this frame
                frame_dets = detect_all_real_objects(
                    sar_img=sar,
                    optical_img=optical,
                    water_mask=water_mask,
                    max_total=num_objects or self.num_objects,
                )

                # Filter out classes that don't belong in this region type
                allowed = set(self._allowed_classes())
                frame_dets = [d for d in frame_dets if d["class_name"] in allowed]

                # Assign track_id based on proximity to previous frame
                if idx > 0 and all_annots:
                    frame_dets = self._assign_track_ids(
                        frame_dets, all_annots[-1],
                    )
                else:
                    for i, det in enumerate(frame_dets):
                        det["track_id"] = i
                        det["behavior"] = "real_detection"
                        det["vx"] = 0.0
                        det["vy"] = 0.0

                all_annots.append(frame_dets)

            logger.info(
                f"Real detection: processed {len(temporal_scenes)} frames, "
                f"avg {np.mean([len(a) for a in all_annots]):.1f} detections/frame"
            )
        else:
            # ── SYNTHETIC ANNOTATION MODE (original) ──
            if num_objects is not None and num_objects != self.num_objects:
                self.num_objects = num_objects
            self._objects = self._init_objects(water_mask)

            h, w = self.image_size
            if water_mask is not None:
                conv_pos = _sample_position_on_mask(water_mask, self.rng)
                convergence_pt = conv_pos if conv_pos else (w * 0.6, h * 0.4)
            else:
                convergence_pt = (w * 0.6, h * 0.4)

            for idx, scene in enumerate(temporal_scenes):
                sar_frames.append(scene["sar"])
                opt_frames.append(scene["optical"])

                frame_annots: List[dict] = []
                for obj in self._objects:
                    obj.step(idx, w, h, convergence_pt, water_mask)
                    frame_annots.append(obj.annotation())
                all_annots.append(frame_annots)

            logger.info(
                f"Annotated {self.num_objects} targets across "
                f"{len(temporal_scenes)} real satellite frames "
                f"(region_type={self.region_type})"
            )

        return sar_frames, opt_frames, all_annots

    def _assign_track_ids(
        self,
        current_dets: List[dict],
        prev_dets: List[dict],
    ) -> List[dict]:
        """Assign track_id to current detections by matching to previous frame.

        Uses nearest-neighbour matching on center coordinates so that
        the same real object gets the same track_id across frames.
        """
        used_prev = set()
        max_dist = 80.0  # max pixels an object can move between frames

        for det in current_dets:
            best_id = None
            best_dist = max_dist
            for pd in prev_dets:
                pid = pd.get("track_id", -1)
                if pid in used_prev:
                    continue
                dist = np.sqrt(
                    (det["cx"] - pd["cx"]) ** 2
                    + (det["cy"] - pd["cy"]) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_id = pid

            if best_id is not None:
                det["track_id"] = best_id
                det["vx"] = det["cx"] - next(
                    (p["cx"] for p in prev_dets if p.get("track_id") == best_id), det["cx"]
                )
                det["vy"] = det["cy"] - next(
                    (p["cy"] for p in prev_dets if p.get("track_id") == best_id), det["cy"]
                )
                det["behavior"] = "real_detection"
                used_prev.add(best_id)
            else:
                # New object — assign fresh ID
                all_ids = used_prev | {p.get("track_id", -1) for p in prev_dets}
                new_id = max(all_ids, default=-1) + 1
                det["track_id"] = new_id
                det["vx"] = 0.0
                det["vy"] = 0.0
                det["behavior"] = "real_detection"
                used_prev.add(new_id)

        return current_dets

    # ------------------------------------------------------------------
    #  Offline fallback — when no pre-downloaded scenes are available,
    #  generate ONLY annotations (no imagery).  The caller must handle
    #  the absence of images.
    # ------------------------------------------------------------------
    def generate_annotations_only(
        self,
        num_frames: int,
    ) -> List[List[dict]]:
        """
        Generate movement annotation sequences without any imagery.

        Returns
        -------
        all_annotations : list[list[dict]]
        """
        h, w = self.image_size
        convergence_pt = (w * 0.6, h * 0.4)
        all_annots: List[List[dict]] = []
        for idx in range(num_frames):
            frame_annots: List[dict] = []
            for obj in self._objects:
                obj.step(idx, w, h, convergence_pt)
                frame_annots.append(obj.annotation())
            all_annots.append(frame_annots)
        return all_annots


# ═══════════════════════════════════════════════════════════════════════════
#  Cache helpers for pre-downloaded real scenes
# ═══════════════════════════════════════════════════════════════════════════

def _region_hash(region: dict) -> str:
    key = (f"{region.get('min_lat', 0):.4f}_{region.get('max_lat', 0):.4f}_"
           f"{region.get('min_lon', 0):.4f}_{region.get('max_lon', 0):.4f}")
    return hashlib.md5(key.encode()).hexdigest()[:10]


def get_scene_cache_dir(region: dict) -> str:
    """Return the cache directory for a region's pre-downloaded scenes."""
    rdir = os.path.join(SCENE_CACHE_ROOT, _region_hash(region))
    os.makedirs(rdir, exist_ok=True)
    return rdir


def load_cached_scenes(
    region: dict,
    image_size: Tuple[int, int] = (2048, 2048),
    max_scenes: int = 20,
) -> List[dict]:
    """
    Load pre-downloaded real satellite scenes from the local cache.

    Returns a list of scene dicts (same format as ``fetch_satellite_data``).
    Returns an empty list if no cached data is available.
    """
    cache_dir = get_scene_cache_dir(region)
    meta_path = os.path.join(cache_dir, "meta.json")

    if not os.path.exists(meta_path):
        return []

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read scene cache meta: {e}")
        return []

    scenes: List[dict] = []
    for entry in meta[:max_scenes]:
        sar_path = entry.get("sar_file", "")
        opt_path = entry.get("optical_file", "")

        if not (os.path.exists(sar_path) and os.path.exists(opt_path)):
            continue

        sar = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE)
        opt = cv2.imread(opt_path, cv2.IMREAD_COLOR)

        if sar is None or opt is None:
            continue

        sar = cv2.resize(sar, image_size)
        opt = cv2.resize(opt, image_size)

        date_str = entry.get("date", "2025-01-01")
        scenes.append({
            "date": date_str,
            "date_obj": datetime.strptime(date_str, "%Y-%m-%d"),
            "sar": sar,
            "optical": opt,
            "scene_id": entry.get("scene_id", ""),
            "platform": entry.get("platform", "Sentinel-1"),
            "source": entry.get("source", "planetary_computer"),
            "geo_info": {
                "region": region.get("name", "Unknown"),
                "center_lat": region.get("center_lat", 0),
                "center_lon": region.get("center_lon", 0),
                "bbox": {
                    "min_lat": region["min_lat"],
                    "max_lat": region["max_lat"],
                    "min_lon": region["min_lon"],
                    "max_lon": region["max_lon"],
                },
                "sensor": entry.get("platform", "Sentinel-1"),
                "source": entry.get("source", "planetary_computer"),
                "date": date_str,
                "resolution": "10m (Sentinel-1/2)",
            },
            "sar_file": sar_path,
            "optical_file": opt_path,
        })

    if scenes:
        logger.info(f"Loaded {len(scenes)} cached real scenes for "
                     f"{region.get('name', 'unknown region')}")

    return sorted(scenes, key=lambda s: s["date"])
