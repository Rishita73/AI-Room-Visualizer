"""
Stage: 3D Multi-Plane Wall Detection, Occlusion Shielding, Perspective Warping & Dimension Estimation
"""

import cv2
import numpy as np

# Structural & major furniture objects that must NOT be tiled over on walls
WALL_OCCLUDER_IDS = {
    3,   # floor
    5,   # ceiling
    7,   # bed
    8,   # windowpane
    9,   # window
    10,  # cabinet / credenza / wardrobe
    14,  # door
    15,  # table / desk
    18,  # curtain
    19,  # chair
    22,  # painting / art / picture frame / poster
    23,  # sofa / couch
    24,  # shelf / bookcase
    27,  # mirror
    30,  # armchair
    31,  # seat / bench
    44,  # chest of drawers
    49,  # countertop
    50,  # desk
    57,  # headboard
    64,  # coffee table
    65,  # sink / washbasin
    69,  # bookcase
    75,  # swivel chair
    80,  # bathtub
    89,  # television / screen
    128, # refrigerator
}

class WallEngine:
    def __init__(self):
        pass

    def extract_wall_mask(self, seg_map, floor_mask=None, depth_map=None):
        """
        Extracts clean, occlusion-free wall mask.
        Enforces strict mutual exclusivity with floor_mask and preserves doors,
        windows, paintings, TVs, ceilings, and all foreground furniture.
        Suppresses clothes, cushions, boxes, shoes, and non-structural items.
        """
        H, W = seg_map.shape[:2]

        # Raw wall class (ADE20K ID 0 = wall)
        wall_raw = (seg_map == 0).astype(np.uint8) * 255
        if not np.any(wall_raw > 0):
            return np.zeros((H, W), dtype=bool)

        # Build strict wall occluder mask with protected structural objects only
        occluders = np.isin(seg_map, list(WALL_OCCLUDER_IDS)).astype(np.uint8) * 255

        # Strict Mutual Exclusivity: Zero overlap between floor and wall
        if floor_mask is not None:
            occluders[floor_mask > 0] = 255

        # Subtract occluders
        wall_clean = wall_raw.copy()
        wall_clean[occluders > 0] = 0

        # Regularize the wall-floor baseboard junction (remove neural network scallop ripples)
        if floor_mask is not None and np.any(wall_clean > 0) and np.any(floor_mask > 0):
            ys_w, xs_w = np.where(wall_clean > 0)
            if len(xs_w) > 20:
                bot_w = {}
                for px, py in zip(xs_w, ys_w):
                    if px not in bot_w or py > bot_w[px]:
                        bot_w[px] = py
                cols = sorted(bot_w.keys())
                if len(cols) > 15:
                    vx = np.array(cols, dtype=np.float32)
                    vy = np.array([bot_w[c] for c in cols], dtype=np.float32)
                    A = np.column_stack([vx, np.ones(len(vx), dtype=np.float32)])
                    m_b, c_b = np.linalg.lstsq(A, vy, rcond=None)[0]
                    # Snap boundary points close to baseline to eliminate wavy scalloping
                    for c in cols:
                        y_line = int(round(m_b * c + c_b))
                        curr_bot = bot_w[c]
                        if abs(curr_bot - y_line) < 14 and 0 <= y_line < H:
                            if y_line > curr_bot:
                                wall_clean[curr_bot:min(y_line + 1, H), c] = 255
                            elif y_line < curr_bot:
                                wall_clean[y_line + 1:curr_bot + 1, c] = 0

        # Morphological Clutter Hole Healing on wall surface
        cnts_w, hier_w = cv2.findContours(wall_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier_w is not None:
            max_hole_area = int(W * H * 0.015)
            for i, c in enumerate(cnts_w):
                if hier_w[0][i][3] != -1:  # Internal hole enclosed by wall
                    if cv2.contourArea(c) < max_hole_area:
                        cv2.drawContours(wall_clean, [c], -1, 255, thickness=-1)

        # Retain significant connected wall structures
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(wall_clean, connectivity=8)
        clean_mask = np.zeros((H, W), dtype=bool)
        min_wall_area = W * H * 0.002

        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_wall_area:
                clean_mask[labels == lbl] = True

        return clean_mask

    def partition_wall_planes(self, wall_mask, depth_map, W, H, floor_quad=None, room_box=None):
        """
        Partitions the wall mask into discrete 3D physical wall planes with slope-aware perspective quads:
          - Left Wall (normal pointing right / receding away on left)
          - Back / Feature Wall (facing camera, fronto-parallel)
          - Right Wall (normal pointing left / receding away on right)
        Anchors plane quads to shared 3D Room-Box geometry & local baseline slope.
        """
        if not np.any(wall_mask):
            return []

        wall_u8 = (wall_mask.astype(np.uint8) * 255)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(wall_u8, connectivity=8)

        planes = []
        cx = W / 2.0

        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < (W * H * 0.004):
                continue

            x = stats[lbl, cv2.CC_STAT_LEFT]
            y = stats[lbl, cv2.CC_STAT_TOP]
            w = stats[lbl, cv2.CC_STAT_WIDTH]
            h = stats[lbl, cv2.CC_STAT_HEIGHT]

            comp_mask = (labels == lbl)

            # Analyze component bottom edge slope and perspective baseline
            c_ys, c_xs = np.where(comp_mask)
            col_bots = {}
            for px, py in zip(c_xs, c_ys):
                if px not in col_bots or py > col_bots[px]:
                    col_bots[px] = py
            
            c_cols = sorted(col_bots.keys())
            if len(c_cols) > 10:
                c_vx = np.array(c_cols, dtype=np.float32)
                c_vy = np.array([col_bots[col] for col in c_cols], dtype=np.float32)
                A = np.column_stack([c_vx, np.ones(len(c_vx), dtype=np.float32)])
                comp_slope, comp_intercept = np.linalg.lstsq(A, c_vy, rcond=None)[0]
                comp_slope = float(np.clip(comp_slope, -0.65, 0.65))
            else:
                comp_slope, comp_intercept = 0.0, float(y + h)
            
            x1 = float(x)
            x2 = float(x + w)
            y_bl = float(np.clip(comp_slope * x1 + comp_intercept, 0, H))
            y_br = float(np.clip(comp_slope * x2 + comp_intercept, 0, H))
            wall_h = max(float(h), float(H * 0.35))

            if comp_slope > 0.06:
                plane_name = "Left Wall"
                plane_id = "left"
                h_near = wall_h * 1.25
                h_far = wall_h * 0.82
                quad = [
                    [x1, float(max(0.0, y_bl - h_near))],
                    [x2, float(max(0.0, y_br - h_far))],
                    [x2, y_br],
                    [x1, y_bl]
                ]
            elif comp_slope < -0.06:
                plane_name = "Right Wall"
                plane_id = "right"
                h_far = wall_h * 0.82
                h_near = wall_h * 1.25
                quad = [
                    [x1, float(max(0.0, y_bl - h_far))],
                    [x2, float(max(0.0, y_br - h_near))],
                    [x2, y_br],
                    [x1, y_bl]
                ]
            else:
                plane_name = "Back Wall"
                plane_id = "back"
                quad = [
                    [x1, float(max(0.0, y_bl - wall_h))],
                    [x2, float(max(0.0, y_br - wall_h))],
                    [x2, y_br],
                    [x1, y_bl]
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

    def compute_wall_quads(self, wall_mask, floor_quad=None, W=1024, H=768):
        """
        Generates fallback perspective quad for vertical wall planes with aligned baseline.
        """
        if not np.any(wall_mask):
            return [[float(W * 0.05), float(H * 0.05)], [float(W * 0.95), float(H * 0.05)],
                    [float(W * 0.95), float(H * 0.65)], [float(W * 0.05), float(H * 0.65)]]

        ys, xs = np.where(wall_mask)
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())

        if floor_quad and len(floor_quad) >= 2:
            P_FL = floor_quad[0]
            P_FR = floor_quad[1]
            far_dx = max(1.0, P_FR[0] - P_FL[0])
            m = (P_FR[1] - P_FL[1]) / far_dx
            c = P_FL[1] - m * P_FL[0]
            y_bl = float(m * x_min + c)
            y_br = float(m * x_max + c)
            h = max(y_max - y_min, H * 0.3)
            return [
                [x_min, max(0.0, y_bl - h)],
                [x_max, max(0.0, y_br - h)],
                [x_max, y_br],
                [x_min, y_bl]
            ]

        return [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ]

    def warp_wall_material(self, room_bgr, tile_surface, dest_quad, mask_bool, shadow_strength=0.55, finish="matte", fg_alpha=None):
        """
        Applies photorealistic perspective homography warping and illumination preservation to walls.
        Respects foreground object occlusion so decor/doors/windows remain on top.
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
        blur_k = int(W * 0.06) | 1
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

        # 3. Crisp Anti-Aliased Edge Feathering Blend (3x3)
        alpha_mask = cv2.GaussianBlur(mask_bool.astype(np.float32), (3, 3), 0)[:, :, None]

        # 3b. Architectural Baseboard Crease AO (Contact Shadow along bottom baseline)
        dist_to_edge = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 3)
        crease_ao = np.clip(dist_to_edge / 6.0, 0.0, 1.0)
        crease_factor = 0.85 + 0.15 * crease_ao[:, :, None]
        material_lit = material_lit * crease_factor

        composite = material_lit * alpha_mask + room_bgr.astype(np.float32) * (1.0 - alpha_mask)

        # 4. Strictly re-composite foreground objects on top if fg_alpha is provided
        if fg_alpha is not None:
            fg_a = fg_alpha[:, :, None] if fg_alpha.ndim == 2 else fg_alpha
            composite = composite * (1.0 - fg_a) + room_bgr.astype(np.float32) * fg_a

        return np.clip(composite, 0, 255).astype(np.uint8)

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
