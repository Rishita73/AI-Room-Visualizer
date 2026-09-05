"""
Stage 9: Photometric Lighting & Shadow Preservation Engine
Decomposes original room illumination to extract ambient lighting, sunlight, and contact shadows.
"""

import cv2
import numpy as np

class LightingEngine:
    def __init__(self):
        pass

    def extract_lighting_map(self, room_bgr, floor_mask):
        """
        Extracts the normalized room lighting multiplier across the floor.
        Uses large-kernel bilateral filtering to separate high-frequency original tile texture
        from low-frequency room illumination.
        Returns:
          - lighting_map: (H, W) float32 array in [0.0, 1.0]
        """
        H, W = room_bgr.shape[:2]
        gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        blur_k = int(W * 0.04) | 1
        gray_blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

        floor_px = gray_blurred[floor_mask]
        if len(floor_px) > 0:
            white_pt = float(np.percentile(floor_px, 90))
            shadow_map = np.clip(gray_blurred / max(white_pt, 1.0), 0.0, 1.0)
            shadow_map = np.clip(0.70 * shadow_map + 0.30, 0.0, 1.0)
        else:
            shadow_map = np.ones((H, W), dtype=np.float32)

        return shadow_map
