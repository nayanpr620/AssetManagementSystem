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
            (30, 45, 45, 85, 255, 220),   # Greens with raised sat/val to avoid dark shadows
            (35, 50, 40, 80, 255, 180),   # Darker greens
        ],
        "min_area": 400,      # Small tree clusters
        "color": "#2ECC71",
    },
    "Water Bodies": {
        "ranges": [
            (90, 80, 60, 130, 255, 255),  # Blue - high saturation, raised value to avoid shadows
            (85, 80, 50, 140, 255, 200),  # Dark blue - raised sat and value
        ],
        "min_area": 800,      # Water bodies
        "color": "#3498DB",
        "require_smooth": True, # Water should have very few internal edges compared to tracks
    },
    "Railway Tracks": {
        "ranges": [
            (0, 0, 50, 180, 40, 200),      # Metallic Grey / Steel 
            (10, 10, 50, 30, 80, 150),     # Brownish rust on tracks
        ],
        "min_area": 200,      
        "color": "#95A5A6",
        "require_lines": True, # CRITICAL: Validate with Hough Lines to avoid detecting random roads/land
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
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
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

                # --- STRUCTURAL VALIDATION ---
                y1_i, y2_i = int(y), int(y + bh)
                x1_i, x2_i = int(x), int(x + bw)
                
                # Check line density for tracks vs water
                if profile.get("require_lines") or profile.get("require_smooth"):
                    roi_bgr = img_bgr[y1_i:y2_i, x1_i:x2_i]
                    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                    # Use Canny edge detection
                    edges = cv2.Canny(roi_gray, 50, 150, apertureSize=3)
                    
                    if profile.get("require_lines"):
                        # Railway tracks must have long straight lines (rails)
                        min_line_len = max(20, min(bw, bh) * 0.3)
                        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=min_line_len, maxLineGap=10)
                        if lines is None or len(lines) < 2:
                            continue  # Reject: No straight parallel rails found
                            
                    if profile.get("require_smooth"):
                        # Water bodies shouldn't have dense structured edges (unlike tracks)
                        edge_density = np.sum(edges > 0) / (bw * bh)
                        if edge_density > 0.05:  # Too many edges, probably a building or steel track
                            continue

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
