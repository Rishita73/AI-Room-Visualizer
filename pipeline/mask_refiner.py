r"""
Stage 6: Multi-Signal Mask Fusion & Edge Refiner
Fuses: (SegFormer Floor ∪ Surface Continuity) ∩ Ground Geometry \ Strict Obstacles
"""

import cv2
import numpy as np

def _box_filter(img, r):
    return cv2.boxFilter(img, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)

def _guided_filter(guide_f32, src_f32, radius=2, eps=1e-4):
    """Sub-pixel edge-preserving guided filter."""
    mean_I = _box_filter(guide_f32, radius)
    mean_p = _box_filter(src_f32, radius)
    mean_Ip = _box_filter(guide_f32 * src_f32, radius)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = _box_filter(guide_f32 * guide_f32, radius)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)

    return mean_a * guide_f32 + mean_b

class MaskRefiner:
    def __init__(self):
        pass

    def refine_mask(self, rough_floor, surface_mask, obstacle_mask, ground_plane_info, image_bgr, depth_map=None, depth_engine=None):
        """
        Refines and outputs the final sharp, occlusion-free floor mask.
        Combines 3D Depth-Plane Support + Surface Continuity + Semantic Priors.
        Preserves floor under chair legs and throughout narrow furniture corridors.
        """
        H, W = image_bgr.shape[:2]

        # 1. 3D Geometric Ground Plane Support
        plane_support = np.zeros((H, W), dtype=bool)
        if depth_engine is not None and ground_plane_info is not None and depth_map is not None:
            plane_support = depth_engine.compute_dense_ground_support(ground_plane_info, depth_map, W, H)

        # 2. Multi-signal floor candidate (Rough SegFormer Floor ∪ Surface Support ∪ 3D Plane Support)
        floor_candidate = (rough_floor | surface_mask | plane_support).astype(np.uint8) * 255
        
        # 3. Strict Obstacle Subtraction (Protects furniture, walls, and objects)
        floor_candidate[obstacle_mask > 0] = 0

        # 4. Morphological Grout & Aisle Continuity Bridge
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        closed_floor = cv2.morphologyEx(floor_candidate, cv2.MORPH_CLOSE, kernel)
        closed_floor[obstacle_mask > 0] = 0

        # 5. Ground-Connected Component Extraction
        num, labels, stats, _ = cv2.connectedComponentsWithStats(closed_floor, connectivity=8)
        connected_floor = np.zeros((H, W), dtype=np.uint8)

        for lbl in range(1, num):
            x, y, w, h, a = stats[lbl]
            # Retain all components on the ground plane (including floor under chairs and in aisles)
            if a > (W * H * 0.0002) and (y + h) > H * 0.35:
                connected_floor[labels == lbl] = 255

        # 6. Sub-pixel Edge Guidance with Original Image
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        guide = gray.astype(np.float32) / 255.0
        src = (connected_floor > 0).astype(np.float32)

        refined_alpha = np.clip(_guided_filter(guide, src, radius=2, eps=1e-4), 0.0, 1.0)
        final_mask = (refined_alpha > 0.40) & (obstacle_mask == 0)

        return final_mask

    def visualize_refined(self, image_bgr, final_mask):
        """Visualizes final refined mask with bright green overlay and white contour line."""
        vis = image_bgr.copy()
        vis[final_mask] = (
            vis[final_mask].astype(np.float32) * 0.35 + np.array([0, 255, 0], dtype=np.float32) * 0.65
        ).astype(np.uint8)
        
        # Draw contour boundary line
        cnts, _ = cv2.findContours(final_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (255, 255, 255), 2)
        return vis
