"""
Stage 3: Surface Continuity & Texture Analyzer
Detects tile grout lines, plank seams, color consistency, and surface continuity.
"""

import cv2
import numpy as np

class SurfaceTextureAnalyzer:
    def __init__(self):
        pass

    def analyze_floor_surface(self, image_bgr, rough_floor_bool, depth_map=None, obstacle_mask=None):
        """
        Analyzes the appearance and texture of the original floor surface.
        Expands the floor mask across continuous texture/color regions and grout lines,
        while strictly preventing bleed onto furniture or walls.
        Returns:
          - surface_mask: (H, W) boolean mask
        """
        H, W = image_bgr.shape[:2]
        
        # LAB Color Space Analysis for Floor Surface Homogeneity
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        # Extract color statistics inside the confident floor seed (lower portion of rough floor)
        ys_seed, xs_seed = np.where(rough_floor_bool)
        if len(ys_seed) == 0:
            return rough_floor_bool.copy()

        y_cutoff = np.percentile(ys_seed, 50)
        seed_idx = ys_seed >= y_cutoff
        if np.count_nonzero(seed_idx) > 50:
            seed_L = L[ys_seed[seed_idx], xs_seed[seed_idx]]
            seed_A = A[ys_seed[seed_idx], xs_seed[seed_idx]]
            seed_B = B[ys_seed[seed_idx], xs_seed[seed_idx]]

            med_L, std_L = np.median(seed_L), np.std(seed_L)
            med_A, std_A = np.median(seed_A), np.std(seed_A)
            med_B, std_B = np.median(seed_B), np.std(seed_B)

            # Color distance from floor profile
            dL = (L.astype(np.float32) - med_L) / max(8.0, std_L * 2.0)
            dA = (A.astype(np.float32) - med_A) / max(3.5, std_A * 2.0)
            dB = (B.astype(np.float32) - med_B) / max(3.5, std_B * 2.0)
            color_dist = np.sqrt(dL**2 + dA**2 + dB**2)
            
            # Confine color support strictly within the lower 65% of the room and proximate to floor seeds
            color_support = (color_dist < 2.0) & (np.arange(H)[:, None] > H * 0.40)
        else:
            color_support = rough_floor_bool.copy()

        # Morphological Texture Bridge (bridges grout joints & tile seams)
        combined = (rough_floor_bool | color_support).astype(np.uint8) * 255
        if obstacle_mask is not None:
            combined[obstacle_mask > 0] = 0

        bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        bridged = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, bridge_kernel)

        if obstacle_mask is not None:
            bridged[obstacle_mask > 0] = 0

        return bridged > 0

    def visualize_surface(self, image_bgr, surface_mask):
        """Visualizes surface continuity support mask."""
        vis = image_bgr.copy()
        vis[surface_mask] = (
            vis[surface_mask].astype(np.float32) * 0.5 + np.array([255, 128, 0], dtype=np.float32) * 0.5
        ).astype(np.uint8)
        return vis
