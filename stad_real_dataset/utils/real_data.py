"""
Real Multi-Temporal Satellite Data Loader
===========================================
Downloads REAL Sentinel-1 SAR + Sentinel-2 Optical acquisitions
from MULTIPLE DATES over user-selected strategic regions.

Data source: Microsoft Planetary Computer STAC API (free, no login)
Sentinel-1 revisit time: ~6 days => 20 scenes = ~120 days of real data.

Each scene = a genuine satellite acquisition on a different date,
making the temporal analysis truly time-series based.
"""

import os
import logging
import numpy as np
import cv2
import requests
import json
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)

# Default region — overridden at runtime via set_active_region()
REGION = {
    "name": "India-China Border (Ladakh)",
    "min_lon": 77.0,
    "max_lon": 79.5,
    "min_lat": 33.0,
    "max_lat": 35.0,
    "center_lat": 34.0,
    "center_lon": 78.25,
}

PC_STAC_URL  = "https://planetarycomputer.microsoft.com/api/stac/v1"
TILE_SIZE    = 512
OUTPUT_SIZE  = 2048
DEG_PER_PIXEL_LAT = (REGION["max_lat"] - REGION["min_lat"]) / OUTPUT_SIZE
DEG_PER_PIXEL_LON = (REGION["max_lon"] - REGION["min_lon"]) / OUTPUT_SIZE


def set_active_region(region_dict: dict):
    """Update the module-level REGION used for coordinate conversions.

    Parameters
    ----------
    region_dict : dict with keys
        name, min_lon, max_lon, min_lat, max_lat, center_lat, center_lon
    """
    global REGION, DEG_PER_PIXEL_LAT, DEG_PER_PIXEL_LON
    REGION.update(region_dict)
    DEG_PER_PIXEL_LAT = (REGION["max_lat"] - REGION["min_lat"]) / OUTPUT_SIZE
    DEG_PER_PIXEL_LON = (REGION["max_lon"] - REGION["min_lon"]) / OUTPUT_SIZE
    logger.info(f"Active region set to: {REGION['name']} "
                f"({REGION['center_lat']:.4f}°N, {REGION['center_lon']:.4f}°E)")


def get_active_region() -> dict:
    """Return a copy of the current active REGION dict."""
    return dict(REGION)


def pixel_to_latlon(px: float, py: float) -> Tuple[float, float]:
    lat = REGION["max_lat"] - py * DEG_PER_PIXEL_LAT
    lon = REGION["min_lon"] + px * DEG_PER_PIXEL_LON
    return round(lat, 6), round(lon, 6)


def latlon_to_pixel(lat: float, lon: float) -> Tuple[int, int]:
    px = int((lon - REGION["min_lon"]) / DEG_PER_PIXEL_LON)
    py = int((REGION["max_lat"] - lat) / DEG_PER_PIXEL_LAT)
    return px, py


def _region_cache_subdir(data_dir: str) -> str:
    """Return a region-specific cache subdirectory based on current active REGION."""
    import hashlib
    key = f"{REGION.get('min_lat',0):.4f}_{REGION.get('max_lat',0):.4f}_{REGION.get('min_lon',0):.4f}_{REGION.get('max_lon',0):.4f}"
    rhash = hashlib.md5(key.encode()).hexdigest()[:10]
    rdir = os.path.join(data_dir, "temporal", rhash)
    os.makedirs(rdir, exist_ok=True)
    return rdir


def load_multitemporal_data(
    n_scenes: int = 20,
    image_size: Tuple[int, int] = (OUTPUT_SIZE, OUTPUT_SIZE),
    data_dir: str = "data",
    force_refresh: bool = False,
) -> List[Dict]:
    """
    Download n_scenes real Sentinel-1 SAR acquisitions from different dates.
    Returns list of scene dicts sorted oldest to newest.
    """
    # Region-specific cache directory
    region_dir = _region_cache_subdir(data_dir)
    cache_meta = os.path.join(region_dir, "meta.json")

    if not force_refresh and os.path.exists(cache_meta):
        try:
            with open(cache_meta) as f:
                meta = json.load(f)
            scenes = _load_scenes_from_cache(meta, region_dir, image_size)
            if len(scenes) >= n_scenes:
                logger.info(f"Loaded {len(scenes)} scenes from cache.")
                return sorted(scenes, key=lambda s: s["date"])[:n_scenes]
        except Exception as e:
            logger.warning(f"Cache load failed ({e}), re-downloading.")

    logger.info("Querying Planetary Computer for Sentinel-1 scenes...")
    scenes = _download_multitemporal_scenes(n_scenes, region_dir, image_size)

    if not scenes:
        logger.warning("No real satellite data could be downloaded. "
                       "Run _download_real_scenes.py to pre-populate the cache, "
                       "or check internet connectivity.")

    # Save metadata for cache
    try:
        meta = [
            {
                "date": s["date"], "scene_id": s["scene_id"],
                "platform": s["platform"], "source": s["source"],
                "sar_file": s.get("sar_file", ""),
                "optical_file": s.get("optical_file", ""),
            }
            for s in scenes
        ]
        with open(cache_meta, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        logger.warning(f"Cache save failed: {e}")

    return sorted(scenes, key=lambda s: s["date"])


def _download_multitemporal_scenes(
    n_scenes: int, data_dir: str, image_size: Tuple[int, int]
) -> List[Dict]:
    bbox = [REGION["min_lon"], REGION["min_lat"],
            REGION["max_lon"], REGION["max_lat"]]
    try:
        resp = requests.post(
            f"{PC_STAC_URL}/search",
            json={
                "collections": ["sentinel-1-grd"],
                "bbox": bbox,
                "limit": min(n_scenes * 5, 100),
                "sortby": [{"field": "datetime", "direction": "desc"}],
            },
            timeout=15,
        )
        resp.raise_for_status()
        s1_items = resp.json().get("features", [])
    except Exception as e:
        logger.warning(f"Sentinel-1 STAC search failed: {e}")
        return []

    if not s1_items:
        return []

    selected = _select_spaced_items(s1_items, n_scenes, min_gap_days=3)
    logger.info(f"Selected {len(selected)} Sentinel-1 scenes.")

    scenes = []
    for item in selected:
        date_str = item["properties"].get("datetime", "")[:10]
        scene_id = item["id"]
        platform = item["properties"].get("platform", "Sentinel-1")

        sar_path = os.path.join(data_dir,
                                f"sar_{date_str}_{scene_id[:8]}.png")
        sar_img  = _download_item_thumbnail(item, sar_path, image_size, grayscale=True)

        opt_path = os.path.join(data_dir,
                                f"opt_{date_str}_{scene_id[:8]}.png")
        opt_img  = _get_paired_optical(date_str, bbox, opt_path, image_size)

        # Skip scenes where BOTH modalities failed to download
        if sar_img is None and opt_img is None:
            logger.debug(f"Skipping scene {scene_id} — no imagery available")
            continue

        # Cross-modal fill from real data (format conversion, not synthesis)
        if sar_img is None and opt_img is not None:
            sar_img = cv2.cvtColor(opt_img, cv2.COLOR_BGR2GRAY) if len(opt_img.shape) == 3 else opt_img
            sar_img = cv2.resize(sar_img, image_size)
        if opt_img is None and sar_img is not None:
            opt_img = cv2.cvtColor(sar_img, cv2.COLOR_GRAY2BGR) if len(sar_img.shape) == 2 else sar_img.copy()
            opt_img = cv2.resize(opt_img, image_size)

        geo_info = _build_geo_info(date_str, platform, "planetary_computer")
        scenes.append({
            "date": date_str,
            "date_obj": datetime.strptime(date_str, "%Y-%m-%d"),
            "sar": sar_img, "optical": opt_img,
            "scene_id": scene_id, "platform": platform,
            "source": "planetary_computer",
            "geo_info": geo_info,
            "sar_file": sar_path, "optical_file": opt_path,
        })

    return scenes


def _select_spaced_items(items: List[dict], n: int,
                          min_gap_days: int) -> List[dict]:
    selected, last_date = [], None
    for item in items:
        dt_str = item["properties"].get("datetime", "")[:10]
        try:
            dt = datetime.strptime(dt_str, "%Y-%m-%d")
        except ValueError:
            continue
        if last_date is None or abs((dt - last_date).days) >= min_gap_days:
            selected.append(item)
            last_date = dt
        if len(selected) >= n:
            break
    return selected


def _download_item_thumbnail(
    item: dict, save_path: str,
    image_size: Tuple[int, int], grayscale: bool = False
) -> Optional[np.ndarray]:
    if os.path.exists(save_path):
        img = cv2.imread(save_path,
                         cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.resize(img, image_size)

    assets = item.get("assets", {})
    for key in ("thumbnail", "preview", "overview"):
        url = assets.get(key, {}).get("href", "")
        if not url:
            continue
        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 200:
                arr  = np.frombuffer(r.content, dtype=np.uint8)
                flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
                img  = cv2.imdecode(arr, flags)
                if img is not None:
                    img_r = cv2.resize(img, image_size)
                    cv2.imwrite(save_path, img_r)
                    return img_r
        except Exception as e:
            logger.debug(f"Thumbnail download failed ({url}): {e}")
    return None


def _get_paired_optical(
    sar_date: str, bbox: list,
    save_path: str, image_size: Tuple[int, int]
) -> Optional[np.ndarray]:
    if os.path.exists(save_path):
        img = cv2.imread(save_path)
        if img is not None:
            return cv2.resize(img, image_size)
    try:
        dt    = datetime.strptime(sar_date, "%Y-%m-%d")
        start = (dt - timedelta(days=15)).strftime("%Y-%m-%dT00:00:00Z")
        end   = (dt + timedelta(days=15)).strftime("%Y-%m-%dT23:59:59Z")
        resp  = requests.post(
            f"{PC_STAC_URL}/search",
            json={
                "collections": ["sentinel-2-l2a"],
                "bbox": bbox,
                "datetime": f"{start}/{end}",
                "limit": 3,
                "sortby": [{"field": "datetime", "direction": "desc"}],
                "query": {"eo:cloud_cover": {"lt": 30}},
            },
            timeout=20,
        )
        resp.raise_for_status()
        items = resp.json().get("features", [])
        if items:
            return _download_item_thumbnail(items[0], save_path,
                                             image_size, grayscale=False)
    except Exception as e:
        logger.debug(f"Sentinel-2 search failed for {sar_date}: {e}")
    return None


def _load_scenes_from_cache(
    meta: List[Dict], data_dir: str, image_size: Tuple[int, int]
) -> List[Dict]:
    scenes = []
    for m in meta:
        sar = cv2.imread(m.get("sar_file", ""), cv2.IMREAD_GRAYSCALE)
        opt = cv2.imread(m.get("optical_file", ""))
        if sar is None or opt is None:
            continue
        date_str = m["date"]
        scenes.append({
            "date": date_str,
            "date_obj": datetime.strptime(date_str, "%Y-%m-%d"),
            "sar": cv2.resize(sar, image_size),
            "optical": cv2.resize(opt, image_size),
            "scene_id": m.get("scene_id", ""),
            "platform": m.get("platform", "Sentinel-1"),
            "source": m.get("source", "cache"),
            "geo_info": _build_geo_info(date_str,
                                         m.get("platform", "Sentinel-1"),
                                         m.get("source", "cache")),
            "sar_file": m.get("sar_file", ""),
            "optical_file": m.get("optical_file", ""),
        })
    return scenes


def _build_geo_info(date_str: str, platform: str, source: str) -> Dict:
    return {
        "region":     REGION["name"],
        "center_lat": REGION["center_lat"],
        "center_lon": REGION["center_lon"],
        "bbox": {
            "min_lat": REGION["min_lat"], "max_lat": REGION["max_lat"],
            "min_lon": REGION["min_lon"], "max_lon": REGION["max_lon"],
        },
        "sensor":     platform,
        "source":     source,
        "date":       date_str,
        "resolution": "10m (Sentinel-1/2)",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Legacy single-tile API (backward-compat)
# ═══════════════════════════════════════════════════════════════════════════

def load_real_satellite_data(
    image_size=(OUTPUT_SIZE, OUTPUT_SIZE), data_dir="data",
):
    """Load a single real satellite scene (latest available)."""
    scenes = load_multitemporal_data(n_scenes=1,
                                      image_size=image_size,
                                      data_dir=data_dir)
    if scenes:
        s = scenes[-1]
        return s["sar"], s["optical"], s["geo_info"]
    # No data available
    logger.warning("No real satellite data available.")
    return None, None, _build_geo_info("2025-01-01", "N/A", "unavailable")


# ═══════════════════════════════════════════════════════════════════════════
#  (Synthetic terrain generators and legacy generate_georeferenced_sequence
#   have been REMOVED.  All imagery is now real satellite data only.
#   Run _download_real_scenes.py to pre-populate the offline cache.)
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
#  High-Resolution Scene Loader (full COG data with clear objects)
# ═══════════════════════════════════════════════════════════════════════════

HIGHRES_ROOT = os.path.join("data", "highres_scenes")


def load_highres_scenes(
    region: dict,
    image_size: Tuple[int, int] = (OUTPUT_SIZE, OUTPUT_SIZE),
    max_scenes: int = 20,
) -> List[Dict]:
    """Load pre-downloaded high-resolution scenes with clear visible objects.

    These scenes use full COG data (not thumbnails) so ships, vehicles,
    and infrastructure are clearly recognizable at 10m resolution.

    Automatically maps wide regions to focused port AOIs if available.

    Parameters
    ----------
    region : dict
        Region dict with min_lat, max_lat, min_lon, max_lon, name keys.
    image_size : tuple
        Target (width, height) for loaded images.
    max_scenes : int
        Maximum number of scenes to return.

    Returns
    -------
    list of scene dicts, sorted by date (oldest first).
    """
    import hashlib

    # Try the direct region hash first
    key = f"{region.get('min_lat',0):.4f}_{region.get('max_lat',0):.4f}_{region.get('min_lon',0):.4f}_{region.get('max_lon',0):.4f}"
    rhash = hashlib.md5(key.encode()).hexdigest()[:10]
    region_dir = os.path.join(HIGHRES_ROOT, rhash)
    meta_path = os.path.join(region_dir, "meta.json")

    # If not found, try to map wide region to focused port AOI
    if not os.path.exists(meta_path):
        try:
            from _download_highres_cogdata import get_highres_region_for
            region_name = region.get("name", "")
            hr_region = get_highres_region_for(region_name)
            if hr_region:
                key2 = f"{hr_region.get('min_lat',0):.4f}_{hr_region.get('max_lat',0):.4f}_{hr_region.get('min_lon',0):.4f}_{hr_region.get('max_lon',0):.4f}"
                rhash = hashlib.md5(key2.encode()).hexdigest()[:10]
                region_dir = os.path.join(HIGHRES_ROOT, rhash)
                meta_path = os.path.join(region_dir, "meta.json")
                if os.path.exists(meta_path):
                    logger.info(f"Mapped '{region_name}' → focused port AOI ({rhash})")
        except ImportError:
            pass

    if not os.path.exists(meta_path):
        # Last resort: scan all highres dirs for any data
        if os.path.isdir(HIGHRES_ROOT):
            for entry in os.listdir(HIGHRES_ROOT):
                candidate = os.path.join(HIGHRES_ROOT, entry, "meta.json")
                if os.path.isfile(candidate):
                    meta_path = candidate
                    region_dir = os.path.join(HIGHRES_ROOT, entry)
                    rhash = entry
                    logger.info(f"Using available high-res data from {entry}")
                    break

    if not os.path.exists(meta_path):
        logger.debug(f"No high-res data for region hash {rhash}")
        return []

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load high-res meta: {e}")
        return []

    optical_by_date = {}
    for sc in meta.get("optical_scenes", []):
        if sc.get("image_ok") and os.path.exists(sc.get("file", "")):
            optical_by_date[sc["date"]] = sc["file"]

    sar_by_date = {}
    for sc in meta.get("sar_scenes", []):
        if sc.get("image_ok") and os.path.exists(sc.get("file", "")):
            sar_by_date[sc["date"]] = sc["file"]

    # Pair by nearest date
    all_dates = sorted(set(list(optical_by_date.keys()) + list(sar_by_date.keys())))

    scenes = []
    for date_str in all_dates:
        opt_file = optical_by_date.get(date_str)
        sar_file = sar_by_date.get(date_str)

        # If missing one modality, try nearest date
        if opt_file is None and optical_by_date:
            nearest = min(optical_by_date.keys(), key=lambda d: abs(
                (datetime.strptime(d, "%Y-%m-%d") - datetime.strptime(date_str, "%Y-%m-%d")).days))
            opt_file = optical_by_date[nearest]
        if sar_file is None and sar_by_date:
            nearest = min(sar_by_date.keys(), key=lambda d: abs(
                (datetime.strptime(d, "%Y-%m-%d") - datetime.strptime(date_str, "%Y-%m-%d")).days))
            sar_file = sar_by_date[nearest]

        if opt_file is None and sar_file is None:
            continue

        # Load images
        sar_img = cv2.imread(sar_file, cv2.IMREAD_GRAYSCALE) if sar_file else None
        opt_img = cv2.imread(opt_file) if opt_file else None

        # Cross-fill if one is missing
        if sar_img is None and opt_img is not None:
            sar_img = cv2.cvtColor(opt_img, cv2.COLOR_BGR2GRAY)
        if opt_img is None and sar_img is not None:
            opt_img = cv2.cvtColor(sar_img, cv2.COLOR_GRAY2BGR)

        if sar_img is None or opt_img is None:
            continue

        sar_img = cv2.resize(sar_img, image_size)
        opt_img = cv2.resize(opt_img, image_size)

        geo_info = {
            "region": meta.get("region", region.get("name", "Unknown")),
            "center_lat": meta.get("center_lat", region.get("center_lat", 0)),
            "center_lon": meta.get("center_lon", region.get("center_lon", 0)),
            "bbox": {
                "min_lat": region.get("min_lat", 0),
                "max_lat": region.get("max_lat", 0),
                "min_lon": region.get("min_lon", 0),
                "max_lon": region.get("max_lon", 0),
            },
            "sensor": "Sentinel-1/2 (High-Res COG)",
            "source": "planetary_computer_highres",
            "date": date_str,
            "resolution": "10m native",
            "highres": True,
        }

        scenes.append({
            "date": date_str,
            "date_obj": datetime.strptime(date_str, "%Y-%m-%d"),
            "sar": sar_img,
            "optical": opt_img,
            "scene_id": f"highres_{rhash}_{date_str}",
            "platform": "Sentinel-1/2",
            "source": "planetary_computer_highres",
            "geo_info": geo_info,
            "sar_file": sar_file or "",
            "optical_file": opt_file or "",
            "highres": True,
        })

    scenes.sort(key=lambda s: s["date"])
    if len(scenes) > max_scenes:
        scenes = scenes[:max_scenes]

    if scenes:
        logger.info(f"Loaded {len(scenes)} HIGH-RES scenes (clear objects) "
                     f"for {meta.get('region', 'unknown')}")
    return scenes


def has_highres_data(region: dict) -> bool:
    """Check if high-res data exists for a given region."""
    import hashlib
    key = f"{region.get('min_lat',0):.4f}_{region.get('max_lat',0):.4f}_{region.get('min_lon',0):.4f}_{region.get('max_lon',0):.4f}"
    rhash = hashlib.md5(key.encode()).hexdigest()[:10]
    return os.path.exists(os.path.join(HIGHRES_ROOT, rhash, "meta.json"))
