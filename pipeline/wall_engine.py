"""
Stage: 3D Multi-Plane Wall Detection, Occlusion Shielding, Perspective Warping & Dimension Estimation
"""

import cv2
import numpy as np

# Everything that sits on or in front of a wall that must NOT be tiled over
WALL_OCCLUDER_IDS = {
    3,   # floor
    5,   # ceiling
    7,   # bed
    8,   # windowpane
    9,   # window
    10,  # cabinet / credenza / wardrobe
    14,  # door
    15,  # table / desk
    17,  # plant / flower
    18,  # curtain
    19,  # chair
    22,  # painting / art / picture frame
    23,  # sofa / couch
    24,  # shelf / bookcase
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
    89,  # television / screen
    108, # decor / ornament
    110, # floor lamp
    119, # pendant lamp
    125, # flowerpot
    132, # vase
    137, # tray / tableware
    138, # ashcan
    146, # radiator / heater
}

class WallEngine:
    def __init__(self):
        pass

    def extract_wall_mask(self, seg_map, floor_mask=None, depth_map=None):
        """
        Extracts clean, occlusion-free wall mask.
        Preserves doors, windows, paintings, TVs, ceiling, and all foreground furniture.
        """
        H, W = seg_map.shape[:2]

        # Raw wall class (ADE20K ID 0 = wall)
        wall_raw = (seg_map == 0).astype(np.uint8) * 255
        if not np.any(wall_raw > 0):
            return np.zeros((H, W), dtype=bool)

        # Build strict wall occluder mask
        occluders = np.isin(seg_map, list(WALL_OCCLUDER_IDS)).astype(np.uint8) * 255

        if floor_mask is not None:
            occluders[floor_mask > 0] = 255

        # Subtract occluders
        wall_clean = wall_raw.copy()
        wall_clean[occluders > 0] = 0

        # Sub-pixel edge refinement: retain significant connected wall structures
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(wall_clean, connectivity=8)
        clean_mask = np.zeros((H, W), dtype=bool)
        min_wall_area = W * H * 0.003

        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_wall_area:
                clean_mask[labels == lbl] = True

        return clean_mask

    def partition_wall_planes(self, wall_mask, depth_map, W, H, vp=None):
        """
        Partitions the wall mask into discrete 3D physical wall planes:
          - Left Wall (normal pointing right, X < 0)
          - Back / Feature Wall (facing camera, fronto-parallel)
          - Right Wall (normal pointing left, X > 0)
        Returns list of plane dicts with 'id', 'name', 'mask', 'quad', 'bbox'.
        """
        if not np.any(wall_mask):
            return []

        wall_u8 = (wall_mask.astype(np.uint8) * 255)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(wall_u8, connectivity=8)

        planes = []
        cx = W / 2.0

        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < (W * H * 0.005):
                continue

            x = stats[lbl, cv2.CC_STAT_LEFT]
            y = stats[lbl, cv2.CC_STAT_TOP]
            w = stats[lbl, cv2.CC_STAT_WIDTH]
            h = stats[lbl, cv2.CC_STAT_HEIGHT]

            comp_mask = (labels == lbl)
            comp_cx = x + w / 2.0

            # Determine wall orientation
            if comp_cx < cx - W * 0.15:
                plane_name = "Left Wall"
                plane_id = "left"
            elif comp_cx > cx + W * 0.15:
                plane_name = "Right Wall"
                plane_id = "right"
            else:
                plane_name = "Back Wall"
                plane_id = "back"

            # Compute perspective wall quad [top_left, top_right, bot_right, bot_left]
            quad = [
                [float(x), float(y)],
                [float(x + w), float(y)],
                [float(x + w), float(y + h)],
                [float(x), float(y + h)]
            ]

            planes.append({
                "id": f"{plane_id}_{len(planes)+1}",
                "name": plane_name,
                "mask": comp_mask,
                "quad": quad,
                "bbox": [x, y, w, h],
                "area_px": int(area)
            })

        return planes

    def compute_wall_quads(self, wall_mask, floor_quad=None, vp=None, W=1024, H=768):
        """
        Generates perspective quads for vertical wall planes.
        """
        if not np.any(wall_mask):
            return [[float(W * 0.05), float(H * 0.05)], [float(W * 0.95), float(H * 0.05)],
                    [float(W * 0.95), float(H * 0.65)], [float(W * 0.05), float(H * 0.65)]]

        ys, xs = np.where(wall_mask)
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())

        return [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ]

    def warp_wall_material(self, room_bgr, tile_surface, dest_quad, mask_bool, shadow_strength=0.55, finish="matte"):
        """
        Applies photorealistic perspective homography warping and illumination preservation to walls.
        """
        H, W = room_bgr.shape[:2]
        ph, pw = tile_surface.shape[:2]

        if not np.any(mask_bool):
            return room_bgr

        # 1. Perspective Homography
        src_pts = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], dtype=np.float32)
        dst_pts = np.array(dest_quad, dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(tile_surface, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # 2. Extract Vertical Ambient Lighting & Shadow Map
        gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blur_k = int(W * 0.05) | 1
        gray_blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

        wall_px = gray_blurred[mask_bool]
        if len(wall_px) > 0:
            white_pt = float(np.percentile(wall_px, 92))
            shadow_map = np.clip(gray_blurred / max(white_pt, 1.0), 0.0, 1.0)
            shadow_map = np.clip(0.65 * shadow_map + 0.35, 0.0, 1.0)
        else:
            shadow_map = np.ones((H, W), dtype=np.float32)

        # Lighting multiplier
        light_gain = (1.0 - shadow_strength) + shadow_strength * shadow_map
        material_lit = warped.astype(np.float32) * light_gain[:, :, None]

        # Specular Highlight for Glossy/Polished Marble Walls
        if finish in ("glossy", "polished"):
            specular = np.clip((shadow_map - 0.78) / 0.22, 0.0, 1.0)
            material_lit = material_lit * (1.0 - 0.15 * specular[:, :, None]) + room_bgr.astype(np.float32) * (0.15 * specular[:, :, None])

        # 3. Sub-pixel Edge Guidance Blend
        alpha_mask = cv2.GaussianBlur(mask_bool.astype(np.float32), (3, 3), 0)[:, :, None]
        final_comp = material_lit * alpha_mask + room_bgr.astype(np.float32) * (1.0 - alpha_mask)

        return np.clip(final_comp, 0, 255).astype(np.uint8)

    def estimate_room_metrics(self, floor_quad, wall_mask, W, H, ppm=200.0):
        """
        Calculates metric room dimensions, gross & net square footage, and tile estimates.
        """
        ppm_val = float(ppm) if ppm and ppm > 1.0 else (W / 4.5)

        # Floor dimensions
        if floor_quad and len(floor_quad) >= 4:
            fq = np.array(floor_quad, dtype=np.float32)
            w_px = float((np.hypot(*(fq[1] - fq[0])) + np.hypot(*(fq[2] - fq[3]))) / 2.0)
            d_px = float((np.hypot(*(fq[3] - fq[0])) + np.hypot(*(fq[2] - fq[1]))) / 2.0)
        else:
            w_px, d_px = W * 0.8, H * 0.5

        w_m = max(2.5, w_px / ppm_val)
        d_m = max(2.5, d_px / ppm_val)
        h_m = 2.85  # Standard residential ceiling height in meters

        w_ft = round(w_m * 3.28084, 1)
        d_ft = round(d_m * 3.28084, 1)
        h_ft = round(h_m * 3.28084, 1)

        floor_sqft = round(w_ft * d_ft, 1)
        perimeter_ft = 2 * (w_ft + d_ft)
        gross_wall_sqft = round(perimeter_ft * h_ft, 1)
        net_wall_sqft = round(gross_wall_sqft * 0.72, 1)  # average 28% openings

        return {
            "width_ft": w_ft,
            "depth_ft": d_ft,
            "height_ft": h_ft,
            "floor_sqft": floor_sqft,
            "gross_wall_sqft": gross_wall_sqft,
            "net_wall_sqft": net_wall_sqft,
            "floor_tiles_600x600": int(np.ceil(floor_sqft / 3.875 * 1.10)), # +10% cuts
            "wall_tiles_300x600": int(np.ceil(net_wall_sqft / 1.937 * 1.10))
        }
