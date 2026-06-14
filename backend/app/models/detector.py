"""
YOLOv8/v11 Multi-Model Hybrid Detector
========================================
Handles inference for aerial/satellite/drone imagery using:
  1. Pre-trained YOLO: Buildings (HuggingFace)
  2. Pre-trained YOLO: Vehicles (COCO)
  3. Pre-trained YOLO: Trees (forest-guardian)
  4. Pre-trained YOLO: Roads (lane-markings)
  5. Pre-trained YOLO: Waste (trash-detection)
  6. HSV Color Segmentation: Water, Parks, Drains (fallback)
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

from app.models.color_segmenter import ColorSegmenter
from app.utils.geo_utils import build_precise_geometry

# ═══════════════════════════════════════════════════════════════
#  MULTI-MODEL CONFIGURATION — All 8 Asset Categories
# ═══════════════════════════════════════════════════════════════

# 1. Properties & Buildings — HuggingFace pre-trained
BUILDING_MAP = {0: "Properties & Buildings"}

# 2. Trees & Green Cover — forest-guardian (Acacia from satellite/drone)
TREES_MAP = {0: "Trees & Green Cover"}

# 3. Roads & Footpaths — Lane markings segmentation
ROADS_MAP = {
    0: "Roads & Footpaths",  # lm_solid
    1: "Roads & Footpaths",  # lm_dashed
}

# 4. Waste Dumps — Trash/garbage segmentation
WASTE_MAP = {
    0: "Waste Dumps",  # Glass
    1: "Waste Dumps",  # Metal
    2: "Waste Dumps",  # Paper
    3: "Waste Dumps",  # Plastic
    4: "Waste Dumps",  # Waste
}

# 5. Vehicles & Parking — COCO YOLOv8 classes
VEHICLE_MAP = {
    2: "Vehicles & Parking",   # car
    3: "Vehicles & Parking",   # motorcycle
    5: "Vehicles & Parking",   # bus
    7: "Vehicles & Parking",   # truck
}

# 6. Land Cover (from Colab training — overrides Trees, Parks, Water, Roads if present)
LANDCOVER_MAP = {
    0: "Trees & Green Cover",
    1: "Parks & Open Spaces",
    2: "Water Bodies",
    3: "Roads & Footpaths",
}

# 7. Trains Model (Pre-trained YOLOv8 for COCO class 6)
TRAIN_MAP = {
    6: "Trains & Rolling Stock",
}

# 8. Track Defects Model (HuggingFace: velocitatem/railway-image-processing)
TRACK_DEFECTS_MAP = {
    0: "Track Components & Defects", # clip
    1: "Track Components & Defects", # sleeper
}

# --- COLOR-BASED DETECTION (HSV fallback for categories without ML models) ---
CATEGORY_COLORS = {
    "Properties & Buildings": "#E74C3C",
    "Trees & Green Cover": "#2ECC71",
    "Parks & Open Spaces": "#27AE60",
    "Water Bodies": "#3498DB",
    "Roads & Footpaths": "#95A5A6",
    "Drains & Sewage": "#8E44AD",
    "Vehicles & Parking": "#F39C12",
    "Waste Dumps": "#7F8C8D",
    "Railway Tracks": "#D35400",
    "Station Platforms": "#8E44AD",
    "Trains & Rolling Stock": "#C0392B",
    "Track Components & Defects": "#F1C40F",
}


def _bbox_metrics(bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    min_side = min(width, height)
    max_side = max(width, height)
    area = max(1.0, width * height)
    return {
        "width": width,
        "height": height,
        "area": area,
        "aspect": max_side / max(1.0, min_side),
        "min_side": min_side,
        "max_side": max_side,
        "area_ratio": 1.0,
    }


def _mask_polygon_to_contour(mask_polygon):
    if not mask_polygon or len(mask_polygon) < 3:
        return None
    return np.array(mask_polygon, dtype=np.float32).reshape((-1, 1, 2))


def _region_stats(image_rgb, bbox, mask_polygon=None):
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    px1 = max(0, min(width - 1, int(round(x1))))
    py1 = max(0, min(height - 1, int(round(y1))))
    px2 = max(0, min(width, int(round(x2))))
    py2 = max(0, min(height, int(round(y2))))
    if px2 <= px1 or py2 <= py1:
        return {
            "saturation_mean": 0,
            "value_mean": 0,
            "hue_mean": 0,
            "is_blue": False,
            "is_green": False,
            "is_grey": True,
        }

    mask = np.zeros((py2 - py1, px2 - px1), dtype=np.uint8)
    contour = _mask_polygon_to_contour(mask_polygon)
    if contour is not None:
        shifted = contour.copy()
        shifted[:, 0, 0] -= px1
        shifted[:, 0, 1] -= py1
        cv2.fillPoly(mask, [shifted.astype(np.int32)], 255)
    else:
        cv2.rectangle(mask, (0, 0), (px2 - px1, py2 - py1), 255, -1)

    roi_rgb = image_rgb[py1:py2, px1:px2]
    roi_mask = mask > 0
    roi_pixels = roi_rgb[roi_mask]
    if roi_pixels.size == 0:
        return {
            "saturation_mean": 0,
            "value_mean": 0,
            "hue_mean": 0,
            "is_blue": False,
            "is_green": False,
            "is_grey": True,
        }

    roi_hsv = cv2.cvtColor(roi_pixels.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    r, g, b = roi_pixels.mean(axis=0)
    saturation_mean = float(roi_hsv[:, 1].mean())
    value_mean = float(roi_hsv[:, 2].mean())
    hue_mean = float(roi_hsv[:, 0].mean())

    return {
        "saturation_mean": saturation_mean,
        "value_mean": value_mean,
        "hue_mean": hue_mean,
        "is_blue": bool(b > r + 15 and b > g + 5),
        "is_green": bool(g > r + 15 and g > b + 5),
        "is_grey": bool(saturation_mean < 55),
    }


def _intersection_area(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a["bbox_pixels"]]
    bx1, by1, bx2, by2 = [float(v) for v in b["bbox_pixels"]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _overlap_ratio(a, b):
    inter = _intersection_area(a, b)
    if inter <= 0:
        return 0.0
    a_area = _bbox_metrics(a["bbox_pixels"])["area"]
    b_area = _bbox_metrics(b["bbox_pixels"])["area"]
    return inter / max(1.0, min(a_area, b_area))


def _conflicting_categories(a, b):
    if a == b:
        return False
    railway = {
        "Railway Tracks",
        "Trains & Rolling Stock",
        "Station Platforms",
        "Track Components & Defects",
    }
    land_cover_noise = {"Water Bodies", "Drains & Sewage", "Roads & Footpaths"}
    if (a in railway and b in land_cover_noise) or (b in railway and a in land_cover_noise):
        return True
    if {a, b} in [
        {"Water Bodies", "Drains & Sewage"},
        {"Water Bodies", "Roads & Footpaths"},
        {"Drains & Sewage", "Roads & Footpaths"},
    ]:
        return True
    return False


def _overlap_threshold(a, b):
    pair = {a, b}
    if "Railway Tracks" in pair and ("Water Bodies" in pair or "Drains & Sewage" in pair):
        return 0.35
    if "Station Platforms" in pair and ("Properties & Buildings" in pair or "Roads & Footpaths" in pair):
        return 0.20 # Reject overlapping platforms aggressively
    if "Railway Tracks" in pair:
        return 0.45
    if "Water Bodies" in pair and ("Roads & Footpaths" in pair or "Drains & Sewage" in pair):
        return 0.55
    if "Drains & Sewage" in pair and "Roads & Footpaths" in pair:
        return 0.65
    return 0.75


class AssetDetector:
    """
    Multi-model detector using 5 YOLO models + HSV color segmentation
    to detect all 8 spatial asset categories from aerial/satellite imagery.
    """

    def __init__(self, model_path: str):
        print("═" * 60)
        print("  🚀 INITIALIZING MULTI-ASSET DETECTION SYSTEM")
        print("═" * 60)

        # ── ML Models ──
        self.models = {}
        building_model_path = Path(model_path).expanduser()
        models_dir = building_model_path.parent

        # Buildings
        if building_model_path.exists():
            self.models["buildings"] = {
                "model": YOLO(str(building_model_path)),
                "map": BUILDING_MAP,
                "label": "Properties & Buildings",
                "path": building_model_path,
            }
            print(f"  ✅ [ML] Properties & Buildings  → {building_model_path}")
        else:
            print(f"  ⚠️  [ML] Buildings model not found at {building_model_path}")

        # Trees
        trees_model_path = models_dir / "trees.pt"
        if trees_model_path.exists():
            self.models["trees"] = {
                "model": YOLO(str(trees_model_path)),
                "map": TREES_MAP,
                "label": "Trees & Green Cover",
                "path": trees_model_path,
            }
            print(f"  ✅ [ML] Trees & Green Cover     → {trees_model_path}")

        # Roads
        roads_model_path = models_dir / "roads.pt"
        if roads_model_path.exists():
            self.models["roads"] = {
                "model": YOLO(str(roads_model_path)),
                "map": ROADS_MAP,
                "label": "Roads & Footpaths",
                "path": roads_model_path,
            }
            print(f"  ✅ [ML] Roads & Footpaths        → {roads_model_path}")

        # Waste
        waste_model_path = models_dir / "waste.pt"
        if waste_model_path.exists():
            self.models["waste"] = {
                "model": YOLO(str(waste_model_path)),
                "map": WASTE_MAP,
                "label": "Waste Dumps",
                "path": waste_model_path,
            }
            print(f"  ✅ [ML] Waste Dumps              → {waste_model_path}")

        # Vehicles
        vehicles_model_path = models_dir / "vehicles.pt"
        if vehicles_model_path.exists():
            self.models["vehicles"] = {
                "model": YOLO(str(vehicles_model_path)),
                "map": VEHICLE_MAP,
                "label": "Vehicles & Parking",
                "path": vehicles_model_path,
            }
            print(f"  ✅ [ML] Vehicles & Parking       → {vehicles_model_path}")

        # Land Cover (from Colab training — overrides Trees, Parks, Water, Roads if present)
        landcover_model_path = models_dir / "landcover.pt"
        if landcover_model_path.exists():
            self.models["landcover"] = {
                "model": YOLO(str(landcover_model_path)),
                "map": LANDCOVER_MAP,
                "label": "Land Cover (4 Classes)",
                "path": landcover_model_path,
            }
            print(f"  ✅ [ML] Land Cover (4 Classes)   → {landcover_model_path}")

        # Trains (using highly-accurate official YOLOv8 segmentation)
        trains_model_path = models_dir / "yolov8m-seg.pt"
        try:
            model = YOLO(str(trains_model_path)) if trains_model_path.exists() else YOLO("yolov8m-seg.pt")
            self.models["trains"] = {
                "model": model,
                "map": TRAIN_MAP,
                "label": "Trains & Rolling Stock",
                "path": trains_model_path,
            }
            print(f"  ✅ [ML] Trains & Rolling Stock   → Auto-downloading/Loading yolov8m-seg.pt")
        except Exception as e:
            print(f"  ⚠️  [ML] Failed to load official trains model: {e}")

        # Track Defects
        track_defects_model_path = models_dir / "track_defects.pt"
        if track_defects_model_path.exists():
            self.models["track_defects"] = {
                "model": YOLO(str(track_defects_model_path)),
                "map": TRACK_DEFECTS_MAP,
                "label": "Track Components & Defects",
                "path": track_defects_model_path,
            }
            print(f"  ✅ [ML] Track Components         → {track_defects_model_path}")

        # ── Color Segmenter ──
        self.color_segmenter = ColorSegmenter()

        # ALWAYS run HSV for land-cover categories (supplementary to ML)
        # ML models for trees/roads were trained on ground-level data, not aerial,
        # so HSV color analysis is essential for satellite imagery detection.
        self.hsv_categories = [
            "Railway Tracks",
            "Station Platforms",
            "Water Bodies",
            "Trees & Green Cover",
        ]

        for cat in self.hsv_categories:
            print(f"  ✅ [CV] {cat:<24} → HSV Color Analysis")

        # ── SAHI tiled inference ──
        self.sahi_models = {}
        try:
            from sahi import AutoDetectionModel

            for key, info in self.models.items():
                model_path_for_sahi = info.get("path")
                if not model_path_for_sahi:
                    continue

                self.sahi_models[key] = AutoDetectionModel.from_pretrained(
                    model_type="yolov8",
                    model_path=str(model_path_for_sahi),
                    confidence_threshold=0.25,
                    device="cpu",
                )
        except Exception as exc:
            print(f"  ⚠️  SAHI init failed (using direct YOLO): {exc}")

        ml_count = len(self.models)
        cv_count = len(self.hsv_categories)
        print("═" * 60)
        print(f"  🎯 {ml_count} ML MODELS + {cv_count} CV ANALYZERS = ALL 8 CATEGORIES")
        print("═" * 60)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(
        self,
        image: Image.Image,
        confidence: float = 0.35,
        use_sahi: bool = True,
        geo_transform: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run detection on a PIL image using all available detectors.
        """
        all_detections = []
        image_rgb = np.array(image)

        # ── Run each ML model ──
        for key, info in self.models.items():
            if use_sahi and key in self.sahi_models:
                all_detections.extend(
                    self._run_sahi(image, self.sahi_models[key], info["map"], geo_transform)
                )
            else:
                all_detections.extend(
                    self._run_yolo(
                        image, info["model"], info["map"], confidence, geo_transform
                    )
                )

        # ── Run HSV color segmentation for remaining categories ──
        if self.hsv_categories:
            color_detections = self.color_segmenter.detect(
                image=image,
                categories=self.hsv_categories,
                geo_transform=geo_transform,
            )
            all_detections.extend(color_detections)

        filtered_detections = [
            det for det in all_detections
            if self._passes_detection_filter(det, image_rgb)
        ]
        final_detections = self._deduplicate_overlapping_noise(filtered_detections)

        return self._build_response(final_detections, image.size)

    def _passes_detection_filter(self, det: Dict[str, Any], image_rgb: np.ndarray) -> bool:
        category = det.get("category")
        bbox = det.get("bbox_pixels")
        if not bbox:
            return True

        metrics = _bbox_metrics(bbox)
        contour = _mask_polygon_to_contour(det.get("mask_polygon"))
        if contour is not None:
            contour_area = float(cv2.contourArea(contour))
            metrics["area_ratio"] = min(1.0, contour_area / max(1.0, metrics["area"]))
        stats = _region_stats(image_rgb, bbox, det.get("mask_polygon"))

        if category == "Railway Tracks":
            if metrics["aspect"] < 2.0:
                return False
            if metrics["min_side"] < 12:
                return False
            if metrics["area_ratio"] > 0.85 and metrics["aspect"] < 8.0:
                return False
            if stats["is_blue"] or stats["is_green"]:
                return False
            if stats["value_mean"] > 160 and stats["saturation_mean"] < 30:
                # Reject sky/haze misclassified as tracks
                return False
            return True

        if category == "Trains & Rolling Stock":
            if det.get("confidence", 1.0) < 0.65:
                # Reject low confidence YOLO detections to stop shadow/pole false positives
                return False
            # Reject poles/shadows which are vertical (aspect < 0.6) or very dark
            if metrics["aspect"] < 0.8:
                return False
            if metrics["area"] < 1500:
                # Small noise likely not a train from high altitude
                return False
            if stats["value_mean"] < 40:
                # Very dark pixels (shadows)
                return False
            return True

        if category == "Water Bodies":
            if metrics["area"] < 600:
                return False
            if not stats["is_blue"] and not (
                85 <= stats["hue_mean"] <= 140 and stats["saturation_mean"] > 40
            ):
                return False
            if metrics["aspect"] > 12 and metrics["area_ratio"] < 0.18:
                return False
            if metrics["area_ratio"] < 0.08 and metrics["aspect"] > 5:
                return False
            return True

        if category == "Drains & Sewage":
            if metrics["aspect"] < 3.5:
                return False
            if metrics["area_ratio"] > 0.55:
                return False
            if stats["is_blue"] or stats["is_green"]:
                return False
            if stats["value_mean"] > 215:
                return False
            return True

        if category == "Roads & Footpaths":
            if (
                metrics["aspect"] > 8.0
                and metrics["area_ratio"] < 0.28
                and stats["saturation_mean"] < 45
                and stats["value_mean"] < 195
            ):
                det["confidence"] = max(0.0, det.get("confidence", 0) - 0.20)
            return det.get("confidence", 0) >= 0.35

        if category == "Station Platforms":
            if metrics["aspect"] > 6.0 and metrics["area_ratio"] < 0.35:
                return False
            # Check if platform is too dark or pure green/blue (likely ground/noise)
            if stats["is_green"] or stats["is_blue"]:
                return False
            if stats["value_mean"] < 60:
                return False
            return True

        return True

    def _deduplicate_overlapping_noise(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        priority = {
            "Railway Tracks": 0,
            "Track Components & Defects": 1,
            "Trains & Rolling Stock": 2,
            "Station Platforms": 3,
            "Water Bodies": 4,
            "Drains & Sewage": 5,
            "Roads & Footpaths": 6,
        }
        ordered = sorted(
            detections,
            key=lambda det: (priority.get(det.get("category"), 99), -det.get("confidence", 0)),
        )
        kept: List[Dict[str, Any]] = []

        for det in ordered:
            category = det.get("category")
            skip = False
            for existing in kept:
                existing_category = existing.get("category")
                if not _conflicting_categories(category, existing_category):
                    continue
                if _overlap_ratio(det, existing) > _overlap_threshold(category, existing_category):
                    skip = True
                    break
            if not skip:
                kept.append(det)

        return kept

    # ------------------------------------------------------------------
    # SAHI inference
    # ------------------------------------------------------------------
    def _run_sahi(
        self, image: Image.Image, sahi_model, category_map: dict, geo_transform: Optional[Dict[str, Any]]
    ) -> List[Dict]:
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(
            image=image,
            detection_model=sahi_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            perform_standard_pred=True,
            postprocess_type="GREEDYNMM",
            postprocess_match_threshold=0.5,
        )

        detections: List[Dict] = []
        for obj in result.object_prediction_list:
            cat_id = obj.category.id
            if cat_id not in category_map:
                continue

            category = category_map[cat_id]
            bbox = obj.bbox
            conf = obj.score.value
            pixel_area = (bbox.maxx - bbox.minx) * (bbox.maxy - bbox.miny)

            bbox_pixels = [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy]
            geo_bbox, geo_polygon, geo_area_sqm = build_precise_geometry(
                bbox_pixels=bbox_pixels,
                mask_polygon=None,
                geo_reference=geo_transform,
            )

            detections.append({
                "category": category,
                "confidence": round(conf, 3),
                "bbox_pixels": bbox_pixels,
                "bbox_geo": geo_bbox,
                "geo_polygon": geo_polygon,
                "area_sqm": geo_area_sqm,
                "pixel_area": pixel_area,
                "color": CATEGORY_COLORS.get(category, "#FFFFFF"),
                "mask_polygon": None,
                "source": "ml",
            })
        return detections

    # ------------------------------------------------------------------
    # Direct YOLO inference
    # ------------------------------------------------------------------
    def _run_yolo(
        self,
        image: Image.Image,
        model,
        category_map: dict,
        confidence: float,
        geo_transform: Optional[Dict[str, Any]],
    ) -> List[Dict]:
        results = model(image, conf=confidence, iou=0.45, verbose=False)
        result = results[0]
        detections: List[Dict] = []
        has_masks = result.masks is not None

        for i, box in enumerate(result.boxes):
            cat_id = int(box.cls[0])
            if cat_id not in category_map:
                continue

            category = category_map[cat_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            pixel_area = (x2 - x1) * (y2 - y1)

            mask_polygon = None
            if has_masks and i < len(result.masks.xy):
                mask_polygon = result.masks.xy[i].tolist()

            bbox_pixels = [x1, y1, x2, y2]
            geo_bbox, geo_polygon, geo_area_sqm = build_precise_geometry(
                bbox_pixels=bbox_pixels,
                mask_polygon=mask_polygon,
                geo_reference=geo_transform,
            )

            detections.append({
                "category": category,
                "confidence": round(conf, 3),
                "bbox_pixels": bbox_pixels,
                "bbox_geo": geo_bbox,
                "geo_polygon": geo_polygon,
                "area_sqm": geo_area_sqm,
                "pixel_area": pixel_area,
                "color": CATEGORY_COLORS.get(category, "#FFFFFF"),
                "mask_polygon": mask_polygon,
                "source": "ml",
            })
        return detections

    # ------------------------------------------------------------------
    # Build structured response
    # ------------------------------------------------------------------
    @staticmethod
    def _build_response(
        detections: List[Dict], image_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        summary: Dict[str, Dict] = {}
        for det in detections:
            cat = det["category"]
            if cat not in summary:
                summary[cat] = {"count": 0, "total_area_sqm": 0, "avg_confidence": 0}
            summary[cat]["count"] += 1
            summary[cat]["avg_confidence"] += det["confidence"]
            if det["area_sqm"]:
                summary[cat]["total_area_sqm"] += det["area_sqm"]

        for cat in summary:
            if summary[cat]["count"] > 0:
                summary[cat]["avg_confidence"] = round(
                    summary[cat]["avg_confidence"] / summary[cat]["count"], 3
                )
            summary[cat]["total_area_sqm"] = round(summary[cat]["total_area_sqm"], 2)

        return {
            "total_detections": len(detections),
            "detections": detections,
            "summary": summary,
            "image_size": {"width": image_size[0], "height": image_size[1]},
        }
