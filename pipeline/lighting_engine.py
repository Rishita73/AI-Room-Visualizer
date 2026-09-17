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
        Uses large-kernel morphological opening and Gaussian blurring to strip out high-frequency
        textures, seams, and clutter while preserving natural window illumination and room shadows.
        """
        H, W = room_bgr.shape[:2]
        gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Morphological opening to strip out old tile grout lines and high-frequency seams
        k_open = max(15, (int(W * 0.025) | 1))
        open_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        gray_opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, open_elem)

        # Wide Gaussian blur to capture pure ambient room light field
        blur_k = max(31, (int(W * 0.08) | 1))
        gray_blurred = cv2.GaussianBlur(gray_opened, (blur_k, blur_k), 0)

        floor_px = gray_blurred[floor_mask > 0] if np.any(floor_mask > 0) else []
        if len(floor_px) > 0:
            white_pt = float(np.percentile(floor_px, 94))
            shadow_map = np.clip(gray_blurred / max(white_pt, 1.0), 0.0, 1.0)
            shadow_map = np.clip(0.68 * shadow_map + 0.32, 0.32, 1.0)
        else:
            shadow_map = np.ones((H, W), dtype=np.float32)

        return shadow_map

    def extract_specular_map(self, room_bgr, active_mask):
        """
        Extracts natural specular reflection highlights from windows and ambient light sources
        present on glossy/polished surfaces in the original room.
        """
        H, W = room_bgr.shape[:2]
        gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        active_px = gray[active_mask > 0] if np.any(active_mask > 0) else []

        if len(active_px) == 0:
            return np.zeros((H, W), dtype=np.float32)

        p80 = float(np.percentile(active_px, 80))
        p98 = max(float(np.percentile(active_px, 98)), p80 + 10.0)

        # Specular sheen ramp
        specular = np.clip((gray - p80) / max(1.0, p98 - p80), 0.0, 1.0)
        specular = cv2.GaussianBlur(specular, (7, 7), 0)
        specular[active_mask == 0] = 0.0

        return specular

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
