"""
Stage 10: Physically Plausible Visualizer Compositor & 9-Stage Debug Visualizer
"""

import cv2
import numpy as np

class VisualizerCompositor:
    def __init__(self):
        pass

    def composite(self, room_bgr, warped_material, floor_mask, lighting_map, shadow_strength=0.55, finish="satin"):
        """
        Blends the perspective-warped material into the room photograph with physically plausible lighting.
        """
        H, W = room_bgr.shape[:2]

        # 1. Apply Shadow Map to Warped Material
        # shadow_factor modulates how deeply original shadows darken the tile
        effective_shadow = (1.0 - shadow_strength) + shadow_strength * lighting_map
        material_lit = warped_material.astype(np.float32) * effective_shadow[:, :, None]

        # 2. Finish Specular & Reflection Modulation
        if finish == "glossy":
            specular = np.clip((lighting_map - 0.75) / 0.25, 0.0, 1.0)
            material_lit = material_lit * (1.0 - 0.18 * specular[:, :, None]) + room_bgr.astype(np.float32) * (0.18 * specular[:, :, None])
        elif finish == "satin":
            specular = np.clip((lighting_map - 0.82) / 0.18, 0.0, 1.0)
            material_lit = material_lit * (1.0 - 0.08 * specular[:, :, None]) + room_bgr.astype(np.float32) * (0.08 * specular[:, :, None])

        # 3. Sub-pixel Alpha Blend
        alpha_mask = cv2.GaussianBlur(floor_mask.astype(np.float32), (3, 3), 0)[:, :, None]
        composite = material_lit * alpha_mask + room_bgr.astype(np.float32) * (1.0 - alpha_mask)

        return np.clip(composite, 0, 255).astype(np.uint8)

    def build_debug_montage(self, debug_stages_dict):
        """
        Arranges all 9 debug stages into a clean 3x3 grid montage with headers.
        """
        stage_keys = [
            ("1_original", "1. Original Room Photo"),
            ("2_raw_segformer", "2. Raw SegFormer Candidate"),
            ("3_surface_analysis", "3. Surface & Texture Analysis"),
            ("4_depth_map", "4. Monocular Depth Map"),
            ("5_floor_plane", "5. 3D RANSAC Ground Plane"),
            ("6_occlusion_mask", "6. Occlusion & Furniture Barrier"),
            ("7_refined_mask", "7. Final Refined Floor Mask"),
            ("8_warped_flooring", "8. Perspective Warped Material"),
            ("9_final_composite", "9. Final Physical Composite")
        ]

        # Find target dimension
        first_img = next(iter(debug_stages_dict.values()))
        th, tw = 360, 480

        panels = []
        for key, title in stage_keys:
            if key in debug_stages_dict and debug_stages_dict[key] is not None:
                img = debug_stages_dict[key]
                resized = cv2.resize(img, (tw, th))
            else:
                resized = np.zeros((th, tw, 3), dtype=np.uint8)

            # Add title banner
            banner = np.zeros((40, tw, 3), dtype=np.uint8)
            cv2.putText(banner, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            panel = np.vstack([banner, resized])
            panels.append(panel)

        # 3x3 Grid
        row1 = np.hstack(panels[0:3])
        row2 = np.hstack(panels[3:6])
        row3 = np.hstack(panels[6:9])
        montage = np.vstack([row1, row2, row3])

        return montage
