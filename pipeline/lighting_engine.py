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
        Extracts the smooth, normalized room lighting multiplier across the floor or active surface.
        Uses large-kernel Gaussian blurring to filter out high-frequency objects, stains,
        hanging items, and clutter, capturing pure ambient illumination and natural room gradients.
        Returns:
          - lighting_map: (H, W) float32 array in [0.0, 1.0]
        """
        H, W = room_bgr.shape[:2]
        gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Large-kernel Gaussian blur to eliminate all surface clutter and object texture
        blur_k = int(W * 0.06) | 1
        gray_blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

        floor_px = gray_blurred[floor_mask > 0] if np.any(floor_mask > 0) else []
        if len(floor_px) > 0:
            white_pt = float(np.percentile(floor_px, 92))
            shadow_map = np.clip(gray_blurred / max(white_pt, 1.0), 0.0, 1.0)
            shadow_map = np.clip(0.72 * shadow_map + 0.28, 0.0, 1.0)
        else:
            shadow_map = np.ones((H, W), dtype=np.float32)

        return shadow_map

    def compute_contact_ao(self, active_mask, obstacle_mask, radius_px=18, intensity=0.45):
        """
        Computes Contact Ambient Occlusion (AO) shadows under objects, furniture bases, and baseboards.
        Gives realistic physical grounding so tiles don't look 'flat' or 'floating'.
        """
        H, W = active_mask.shape[:2]
        
        # Source of occlusion shadows: obstacle borders and boundaries
        obs_u8 = (obstacle_mask > 0).astype(np.uint8)
        if not np.any(obs_u8 > 0):
            return np.ones((H, W), dtype=np.float32)

        # Distance transform from obstacles
        non_obs = (1 - obs_u8) * 255
        dist = cv2.distanceTransform(non_obs.astype(np.uint8), cv2.DIST_L2, 5)

        # Normalize distance within shadow radius
        rad = max(6.0, float(radius_px))
        ao_norm = np.clip(dist / rad, 0.0, 1.0)

        # Falloff curve (quadratic for soft, natural penumbra)
        ao = 1.0 - (1.0 - ao_norm) ** 1.8 * float(intensity)
        ao = np.clip(ao, 1.0 - intensity, 1.0).astype(np.float32)

        # Restrict AO to active floor/wall regions
        final_ao = np.ones((H, W), dtype=np.float32)
        final_ao[active_mask > 0] = ao[active_mask > 0]

        return final_ao
