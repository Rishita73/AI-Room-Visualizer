"""
Stage 4: Occlusion & Furniture Geometry Shield
Features:
  • Morphological Furniture Component Closure: Isolates connected furniture entities (sofas, armchairs, TV consoles, tables) and seals their internal voids/cushions as protected 3D obstacle hulls.
  • Topological Void Gate: Enforces a strict 0% floor overlap and zero furniture probability constraint inside furniture bodies.
"""

import cv2
import numpy as np

# Primary structural & major furniture entities that are strictly preserved on top of tiles:
PROTECTED_STRUCTURAL_IDS = {
    5,   # ceiling
    7,   # bed
    8,   # windowpane
    9,   # window
    10,  # cabinet / wardrobe / credenza
    14,  # door
    15,  # table
    17,  # plant
    18,  # curtain
    19,  # chair
    22,  # painting / art / poster / picture frame
    23,  # sofa / couch
    24,  # shelf
    27,  # mirror
    30,  # armchair
    31,  # seat / bench
    33,  # desk
    35,  # wardrobe / closet
    36,  # lamp (floor lamp, desk lamp)
    37,  # bathtub
    39,  # cushion
    44,  # chest of drawers
    47,  # sink / washbasin
    50,  # refrigerator
    57,  # pillow / headboard
    58,  # screen door / shower door
    62,  # bookcase
    64,  # coffee table
    65,  # toilet
    70,  # countertop
    75,  # swivel chair
    80,  # bus (retained for backward compatibility)
    41,  # box / storage crate / trunk
    81,  # towel
    89,  # television / screen
    110, # lamp alias
    115, # bag / handbag / backpack
    119, # pendant lamp / chandelier
    125, # flowerpot / potted plant
    128, # refrigerator alias
    130, # screen
    132, # vase / pitcher
    135, # vase
    141, # crt screen
}

STRICT_OBSTACLE_IDS = PROTECTED_STRUCTURAL_IDS.union({
    5,   # ceiling
})

# Loose small non-structural clutter classes that may be suppressed on floor/table surfaces
LOOSE_CLUTTER_IDS = {108, 131, 137, 138, 146}

class OcclusionDetector:
    def __init__(self):
        pass

    def detect_obstacles(self, seg_map, depth_map=None, W=None, H=None):
        """
        Builds a high-precision binary obstacle mask isolating ALL major furniture and structural objects
        (Sofas, Armchairs, Chairs, Beds, Tables, Cabinets, Paintings, Doors, Windows, Ceilings, Boxes, Bags).
        Seals seat cushions and internal frames so tiles never bleed into seating.
        Suppresses only small loose clutter items (loose clothes, small boxes, wires, trash).
        """
        if H is None or W is None:
            H, W = seg_map.shape[:2]

        # 1. Base semantic obstacle mask using protected structural and furniture classes + ceiling
        base_obstacles = np.isin(seg_map, list(STRICT_OBSTACLE_IDS)).astype(np.uint8) * 255

        # Wall (ID 0) Topological Ground-Exclusion Analysis:
        # True physical walls originate in the upper room (y < H * 0.40) and extend down to baseboards.
        # Connected wall components rooted in the upper room are classified as true walls (strict obstacles for floor).
        # Small isolated patches of 'wall' located deep on the floor plane without connection to the upper room
        # are ground shadows/carpet misclassifications and are excluded from obstacles.
        wall_raw = (seg_map == 0).astype(np.uint8)
        num_w, labels_w, stats_w, _ = cv2.connectedComponentsWithStats(wall_raw, connectivity=8)
        true_wall_mask = np.zeros((H, W), dtype=bool)
        min_wall_area = max(100, int(W * H * 0.005))
        for lbl in range(1, num_w):
            y_top = stats_w[lbl, cv2.CC_STAT_TOP]
            area = stats_w[lbl, cv2.CC_STAT_AREA]
            if y_top < (H * 0.40) and area > min_wall_area:
                true_wall_mask |= (labels_w == lbl)

        obstacle_mask = base_obstacles | (true_wall_mask.astype(np.uint8) * 255)

        # 2. Structural Component Closure on Seating (Sofas, Armchairs, Chairs, Beds, Cushions)
        seating_ids = [7, 19, 23, 30, 31, 39, 52, 57, 75]
        seating_raw = np.isin(seg_map, seating_ids).astype(np.uint8) * 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seating_raw, connectivity=8)
        for lbl in range(1, num_labels):
            x, y, w, h, area = stats[lbl]
            if area > 300:
                comp_mask = (labels == lbl).astype(np.uint8) * 255
                cnts_c, hier = cv2.findContours(comp_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                if hier is not None:
                    for i, c in enumerate(cnts_c):
                        if hier[0][i][3] != -1:
                            cv2.drawContours(obstacle_mask, [c], -1, 255, thickness=-1)
                k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
                closed_comp = cv2.morphologyEx(comp_mask, cv2.MORPH_CLOSE, k_close)
                obstacle_mask[closed_comp > 0] = 255

        # 3. Tables, Desks, Countertops, Cabinets, Sinks & Consoles Component Closure
        table_ids = [10, 15, 24, 33, 35, 37, 44, 47, 50, 58, 62, 64, 65, 70, 89, 128]
        table_raw = np.isin(seg_map, table_ids).astype(np.uint8) * 255
        num_t, labels_t, stats_t, _ = cv2.connectedComponentsWithStats(table_raw, connectivity=8)
        for lbl in range(1, num_t):
            x, y, w, h, area = stats_t[lbl]
            if area > 250:
                comp_t = (labels_t == lbl).astype(np.uint8) * 255
                k_close_t = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
                closed_t = cv2.morphologyEx(comp_t, cv2.MORPH_CLOSE, k_close_t)
                obstacle_mask[closed_t > 0] = 255

        # 4. Small Object Suppression: eliminate only tiny micro-noise
        num_obs, labels_obs, stats_obs, _ = cv2.connectedComponentsWithStats(obstacle_mask, connectivity=8)
        min_structural_area = max(120, int(W * H * 0.001))
        for lbl in range(1, num_obs):
            w = stats_obs[lbl, cv2.CC_STAT_WIDTH]
            h = stats_obs[lbl, cv2.CC_STAT_HEIGHT]
            area = stats_obs[lbl, cv2.CC_STAT_AREA]
            if area < min_structural_area and (w < 25 and h < 25):
                obstacle_mask[labels_obs == lbl] = 0

        return obstacle_mask

    def suppress_and_blur_clutter(self, image_bgr, seg_map, floor_mask=None, wall_mask=None):
        """
        Suppresses and smoothly blurs small, loose, non-structural clutter items
        (loose clothes, shoes, wires, small boxes, toys, trash, scattered debris)
        on the floor and wall surfaces while NEVER blurring sofas, armchairs, chairs, or furniture.
        """
        H, W = image_bgr.shape[:2]

        # Target loose clutter classes
        clutter_raw = np.isin(seg_map, list(LOOSE_CLUTTER_IDS)).astype(np.uint8) * 255

        # Find isolated loose cushions/pillows not attached to seating
        loose_cushions = np.isin(seg_map, [39, 52]).astype(np.uint8) * 255
        seating_bodies = np.isin(seg_map, [7, 19, 23, 30, 31, 57, 75]).astype(np.uint8) * 255
        seating_dil = cv2.dilate(seating_bodies, np.ones((15, 15), np.uint8))
        # Cushions not touching any sofa/armchair/chair are loose clutter on floor
        isolated_cushions = (loose_cushions > 0) & (seating_dil == 0)
        clutter_raw[isolated_cushions] = 255

        # Do NOT include any major furniture in clutter
        furniture_mask = np.isin(seg_map, list(PROTECTED_STRUCTURAL_IDS)).astype(np.uint8) * 255
        clutter_raw[furniture_mask > 0] = 0

        active_zone = np.zeros((H, W), dtype=bool)
        if floor_mask is not None:
            active_zone |= (floor_mask > 0)
        if wall_mask is not None:
            active_zone |= (wall_mask > 0)

        clutter_mask = clutter_raw.copy()

        if np.any(active_zone):
            active_dil = cv2.dilate(active_zone.astype(np.uint8) * 255, np.ones((11, 11), np.uint8)) > 0
            clutter_mask[~active_dil] = 0

        if not np.any(clutter_mask > 0):
            return image_bgr.copy(), clutter_mask

        # Dilate clutter mask for smooth seamless blending
        clutter_dil = cv2.dilate(clutter_mask, np.ones((9, 9), np.uint8))
        k_blur = max(25, int(W * 0.035) | 1)
        blurred_bg = cv2.GaussianBlur(image_bgr, (k_blur, k_blur), 0)
        
        alpha_clutter = cv2.GaussianBlur((clutter_dil > 0).astype(np.float32), (9, 9), 0)[:, :, None]
        cleaned_image = (blurred_bg.astype(np.float32) * alpha_clutter + image_bgr.astype(np.float32) * (1.0 - alpha_clutter)).astype(np.uint8)

        # Inpaint solid cores to eliminate dark shadows/ghosting
        core_clutter = (clutter_mask > 0).astype(np.uint8)
        if np.any(core_clutter > 0):
            try:
                cleaned_image = cv2.inpaint(cleaned_image, core_clutter, 5, cv2.INPAINT_TELEA)
            except Exception:
                pass

        return cleaned_image, clutter_mask

    def extract_foreground_matte(self, image_bgr, obstacle_mask, floor_mask=None, wall_mask=None, seg_map=None):
        """
        Generates a sub-pixel edge-guided alpha matte for foreground objects.
        This ensures objects (chairs, sofas, table legs, plants, lamps, shelves) are cleanly preserved on top of tiles.
        """
        H, W = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        guide = gray.astype(np.float32) / 255.0

        # Create ROI where foreground intersects or borders active tiled zones
        active_surface = np.zeros((H, W), dtype=bool)
        if floor_mask is not None:
            active_surface |= (floor_mask > 0)
        if wall_mask is not None:
            active_surface |= (wall_mask > 0)

        # Dilate active surface to capture contacting boundary
        active_dilated = cv2.dilate(active_surface.astype(np.uint8) * 255, np.ones((15, 15), np.uint8)) > 0
        
        if seg_map is not None:
            fg_entities = np.isin(seg_map, list(PROTECTED_STRUCTURAL_IDS)).astype(np.uint8) * 255
        else:
            fg_entities = obstacle_mask

        fg_roi = (fg_entities > 0) & active_dilated

        fg_src = fg_roi.astype(np.float32)

        # Fast Box & Guided Filter implementation for matte refinement
        def box_filt(img, r):
            return cv2.boxFilter(img, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)

        r = 3
        eps = 1e-4
        mean_I = box_filt(guide, r)
        mean_p = box_filt(fg_src, r)
        mean_Ip = box_filt(guide * fg_src, r)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = box_filt(guide * guide, r)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = box_filt(a, r)
        mean_b = box_filt(b, r)

        fg_alpha = np.clip(mean_a * guide + mean_b, 0.0, 1.0)
        
        # Smooth anti-aliasing: solid foreground obstacle core is fully opaque, outer boundary is feathered
        core_fg = cv2.erode((fg_entities > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        penumbra_fg = cv2.dilate((fg_entities > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        fg_alpha[core_fg] = 1.0
        fg_alpha[~penumbra_fg] = 0.0

        return fg_alpha

    def visualize_obstacles(self, image_bgr, obstacle_mask):
        """Visualizes obstacle exclusion zones in red."""
        vis = image_bgr.copy()
        is_obs = obstacle_mask > 0
        vis[is_obs] = (
            vis[is_obs].astype(np.float32) * 0.4 + np.array([0, 0, 255], dtype=np.float32) * 0.6
        ).astype(np.uint8)
        return vis
