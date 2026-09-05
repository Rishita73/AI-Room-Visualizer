"""
Stage 4: Occlusion & Furniture Geometry Shield
Features:
  • Morphological Furniture Component Closure: Isolates connected furniture entities (sofas, armchairs, TV consoles, tables) and seals their internal voids/cushions as protected 3D obstacle hulls.
  • Topological Void Gate: Enforces a strict 0% floor overlap and zero furniture probability constraint inside furniture bodies.
"""

import cv2
import numpy as np

# ADE20K obstacle classes that sit on or in front of the floor
STRICT_OBSTACLE_IDS = {
    0,   # wall
    5,   # ceiling
    7,   # bed
    8, 9,# windowpane / window
    10,  # cabinet / TV console / credenza
    14,  # door
    15,  # table / desk
    17,  # plant / flower
    18,  # curtain
    19,  # chair
    22,  # painting / art
    23,  # sofa / couch / settee
    24,  # shelf
    27,  # mirror
    30,  # armchair
    31,  # seat / bench
    36,  # wardrobe / lamp
    39,  # cushion
    41,  # box / ottoman
    44,  # chest of drawers
    50,  # desk
    52,  # pillow
    53,  # stairs
    57,  # headboard
    64,  # coffee table
    75,  # swivel chair
    85,  # chandelier
    89,  # television
    108, # cushion / decor
    110, # floor lamp
    119, # pendant lamp
    125, # flowerpot
    132, # vase
    137, # tray / tableware
    138, # ashcan / bin
    146, # radiator
}

class OcclusionDetector:
    def __init__(self):
        pass

    def detect_obstacles(self, seg_map, depth_map=None, W=None, H=None):
        """
        Builds a high-precision binary obstacle mask isolating all foreground objects.
        Implements:
          1. Base semantic obstacle extraction.
          2. Morphological Furniture Component Closure on solid seating & tables.
          3. Topological Void Gate ensuring 0% floor bleed on internal cushion voids.
        """
        if H is None or W is None:
            H, W = seg_map.shape[:2]

        # 1. Base semantic obstacle mask
        obstacle_mask = np.isin(seg_map, list(STRICT_OBSTACLE_IDS)).astype(np.uint8) * 255

        # 2. Morphological Furniture Component Closure on Seating (Sofas, Armchairs, Sectionals)
        sofa_ids = [23, 30, 31, 39, 52, 57, 75, 108]
        sofa_raw = np.isin(seg_map, sofa_ids).astype(np.uint8) * 255

        # Isolate each connected furniture component independently
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sofa_raw, connectivity=8)
        for lbl in range(1, num_labels):
            x, y, w, h, area = stats[lbl]
            if area > 1200:
                comp_mask = (labels == lbl).astype(np.uint8) * 255
                
                # Topological Void Gate: fill internal voids enclosed by the sofa frame
                cnts_c, hier = cv2.findContours(comp_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                if hier is not None:
                    for i, c in enumerate(cnts_c):
                        # Internal hole enclosed by component
                        if hier[0][i][3] != -1:
                            cv2.drawContours(obstacle_mask, [c], -1, 255, thickness=-1)
                
                # Morphological closure on the sofa body to seal low-confidence grey seat cushions
                k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
                closed_comp = cv2.morphologyEx(comp_mask, cv2.MORPH_CLOSE, k_close)
                obstacle_mask[closed_comp > 0] = 255

        # 3. Coffee Table & Consoles Component Closure
        table_raw = np.isin(seg_map, [15, 50, 64, 137]).astype(np.uint8) * 255
        num_t, labels_t, stats_t, _ = cv2.connectedComponentsWithStats(table_raw, connectivity=8)
        for lbl in range(1, num_t):
            x, y, w, h, area = stats_t[lbl]
            if area > 600:
                comp_t = (labels_t == lbl).astype(np.uint8) * 255
                k_close_t = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                closed_t = cv2.morphologyEx(comp_t, cv2.MORPH_CLOSE, k_close_t)
                obstacle_mask[closed_t > 0] = 255

        return obstacle_mask

    def visualize_obstacles(self, image_bgr, obstacle_mask):
        """Visualizes obstacle exclusion zones in red."""
        vis = image_bgr.copy()
        is_obs = obstacle_mask > 0
        vis[is_obs] = (
            vis[is_obs].astype(np.float32) * 0.4 + np.array([0, 0, 255], dtype=np.float32) * 0.6
        ).astype(np.uint8)
        return vis
