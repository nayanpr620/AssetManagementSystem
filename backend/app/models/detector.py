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
    "Railway Tracks": "#D35400",          # Dark Orange
    "Station Platforms": "#8E44AD",       # Purple
    "Trains & Rolling Stock": "#C0392B",  # Dark Red
    "Track Components & Defects": "#F1C40F", # Yellow
}


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
            self.models["trains"] = {
                "model": YOLO("yolov8m-seg.pt"),  # Ultralytics will auto-download this
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

        # ── SPATIAL EXCLUSION LOGIC (Resolve Overlaps) ──
        # Do not let generic land cover (Water, Roads, Trees, Waste, Parks)
        # overlap significantly with Railway Infrastructure (Tracks, Trains, Platforms, Defects)
        railway_categories = {
            "Railway Tracks", "Trains & Rolling Stock",
            "Station Platforms", "Track Components & Defects"
        }
        
        final_detections = []
        railway_dets = [d for d in all_detections if d["category"] in railway_categories]
        generic_dets = [d for d in all_detections if d["category"] not in railway_categories]
        
        final_detections.extend(railway_dets)
        
        for g_det in generic_dets:
            gx1, gy1, gx2, gy2 = g_det["bbox_pixels"]
            g_area = max(1, (gx2 - gx1) * (gy2 - gy1))
            
            overlap_violation = False
            for r_det in railway_dets:
                rx1, ry1, rx2, ry2 = r_det["bbox_pixels"]
                
                # Intersection area
                ix1 = max(gx1, rx1)
                iy1 = max(gy1, ry1)
                ix2 = min(gx2, rx2)
                iy2 = min(gy2, ry2)
                
                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                overlap_ratio = inter_area / g_area
                
                # If generic box is overlapping a railway box by more than 20% of its own area
                if overlap_ratio > 0.20:
                    overlap_violation = True
                    break
            
            if not overlap_violation:
                final_detections.append(g_det)

        return self._build_response(final_detections, image.size)

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
