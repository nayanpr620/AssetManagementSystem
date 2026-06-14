"""
HSV Color-Space Land Cover Segmenter for Satellite/Aerial Imagery.

This module implements computer-vision-based segmentation for land cover
categories where color analysis on satellite imagery outperforms generic
ML models (Trees, Water, Roads, Parks). This is the same principle used
by the DeepGlobe Land Cover Classification benchmark.

Each category is detected by filtering the image in HSV color space,
finding contours, and converting them to normalized polygon coordinates.
"""
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from app.utils.geo_utils import build_precise_geometry


# HSV ranges tuned for satellite/aerial imagery
# Format: (H_low, S_low, V_low, H_high, S_high, V_high)
COLOR_PROFILES = {
    "Trees & Green Cover": {
        "ranges": [
            (30, 45, 45, 85, 255, 220),   # Raised saturation & value to avoid dark/desaturated gravel shadows
        ],
        "min_area": 300,       # Smaller to catch vegetation patches
        "color": "#2ECC71",
    },
    "Water Bodies": {
        "ranges": [
            (90, 80, 60, 130, 255, 255),  # Blue - high saturation, raised value to avoid shadows
            (85, 80, 50, 140, 255, 200),  # Dark blue - raised sat and value
        ],
        "min_area": 800,      # Water bodies
        "color": "#3498DB",
    },
    "Railway Tracks": {
        "ranges": [
            (0, 0, 15, 20, 80, 240),      # Dark brown/rust
            (15, 0, 15, 40, 90, 230),    # Brown ballast
            (0, 0, 40, 20, 50, 180),     # Dark grey
            (160, 0, 40, 180, 60, 200),   # Cool grey (wraps)
        ],
        "min_area": 400,      # Smaller to catch track segments
        "color": "#E67E22",
    },
    "Station Platforms": {
        "ranges": [
            (15, 5, 180, 40, 60, 255),   # Light concrete/beige (warm tint only, avoids pure grey roads)
        ],
        "min_area": 5000,     # Larger - platforms are substantial
        "color": "#8E44AD",   # Fixed UX mismatch: changed from Orange to Purple to match UI
    },
}


class ColorSegmenter:
    """
    Performs HSV color-space segmentation on satellite/aerial images
    to detect land cover categories (Trees, Water, Roads, Parks, etc.).
    """

    def detect(
        self,
        image: Image.Image,
        categories: Optional[List[str]] = None,
        geo_transform: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run color-based segmentation on a PIL image.

        Args:
            image: PIL Image (RGB).
            categories: List of category names to detect. None = all.
            geo_transform: (origin_lon, origin_lat, width_deg, height_deg).

        Returns:
            List of detection dicts compatible with AssetDetector output.
        """
        # Convert PIL → OpenCV BGR → HSV
        img_rgb = np.array(image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, w = img_hsv.shape[:2]

        if categories is None:
            categories = list(COLOR_PROFILES.keys())

        all_detections = []

        for category in categories:
            if category not in COLOR_PROFILES:
                continue

            profile = COLOR_PROFILES[category]
            combined_mask = np.zeros((h, w), dtype=np.uint8)

            # Apply all HSV ranges for this category
            for hsv_range in profile["ranges"]:
                lower = np.array(hsv_range[:3])
                upper = np.array(hsv_range[3:])
                mask = cv2.inRange(img_hsv, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Morphological cleanup: remove noise, fill gaps
            k_size = 11 if category == "Station Platforms" else 5
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            if category == "Station Platforms":
                # Extra closing for platforms to merge fragmented concrete blocks
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            else:
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            
            # Skip MORPH_OPEN for tracks because it erases thin steel rails and long continuous ballast
            if category != "Railway Tracks":
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(
                combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < profile["min_area"]:
                    continue

                # Bounding box for aspect check
                x, y, bw, bh = cv2.boundingRect(contour)
                aspect = max(bw, bh) / max(1, min(bw, bh))

                # Platforms should be compact, not elongated like tracks
                if category == "Station Platforms" and aspect > 6.0:
                    continue

                # Simplify polygon
                epsilon = 0.003 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) < 3:
                    continue

                # Bounding box
                x, y, bw, bh = cv2.boundingRect(contour)
                x1, y1, x2, y2 = float(x), float(y), float(x + bw), float(y + bh)
                pixel_area = float(bw * bh)

                # Normalized polygon for segmentation mask
                points = approx.reshape(-1, 2)
                mask_polygon = points.tolist()

                # Geo conversion
                geo_bbox, geo_polygon, geo_area_sqm = build_precise_geometry(
                    bbox_pixels=[x1, y1, x2, y2],
                    mask_polygon=mask_polygon,
                    geo_reference=geo_transform,
                )

                # Confidence based on color purity of the region
                roi_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(roi_mask, [contour], -1, 255, -1)
                matching_pixels = cv2.countNonZero(cv2.bitwise_and(combined_mask, roi_mask))
                total_pixels = cv2.countNonZero(roi_mask)
                confidence = round(matching_pixels / max(total_pixels, 1), 3)
                confidence = min(max(confidence, 0.4), 0.95)  # Clamp to realistic range

                all_detections.append({
                    "category": category,
                    "confidence": confidence,
                    "bbox_pixels": [x1, y1, x2, y2],
                    "bbox_geo": geo_bbox,
                    "geo_polygon": geo_polygon,
                    "area_sqm": geo_area_sqm,
                    "pixel_area": pixel_area,
                    "color": profile["color"],
                    "mask_polygon": mask_polygon,
                    "source": "color_segmentation",
                })

        return all_detections
