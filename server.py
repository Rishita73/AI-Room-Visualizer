import os
import io
import base64
import json
import numpy as np
from PIL import Image
import cv2

# FastAPI Imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI(title="AI Room Visualizer API", version="2.1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ADE20K Label Map (relevant labels for room analysis)
# ============================================================
ADE20K_LABELS = {
    0:  "wall",
    3:  "floor",
    7:  "bed",
    9:  "window",
    10: "cabinet",
    14: "door",
    15: "table",
    19: "chair",
    23: "sofa",
    28: "rug",
    53: "stairs",
    64: "coffee table",
    75: "swivel chair",
}

# Labels excluded from floor net area calculation (furniture on floor)
FLOOR_OBSTACLE_IDS = {7, 10, 15, 19, 23, 53, 64, 75}

# Labels excluded from wall net area calculation (openings on wall)
WALL_OBSTACLE_IDS = {9, 14}

# Standard real-world reference sizes (in meters)
REAL_WORLD_REF = {
    14: {"h": 2.05, "w": 0.85},  # door: 6.7ft x 2.8ft
    9:  {"h": 1.20, "w": 1.00},  # window: ~4ft x 3.3ft
}

# ============================================================
# Global variables for AI model
# ============================================================
MODEL_AVAILABLE = False
processor = None
seg_model = None

try:
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    print("Loading SegFormer model (nvidia/segformer-b2-finetuned-ade-512-512)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
        seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    except Exception as fetch_err:
        print(f"HuggingFace fetch failed ({fetch_err}). Using local cache...")
        processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512", local_files_only=True)
        seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512", local_files_only=True)

    seg_model.to(device)
    seg_model.eval()
    MODEL_AVAILABLE = True
    print(f"SegFormer loaded on {device.upper()}!")
except Exception as e:
    print(f"Warning: SegFormer unavailable. Simulation mode active. ({e})")
    MODEL_AVAILABLE = False


# ============================================================
# HELPER: Robust Full Floor Mask Extraction
# ============================================================
def extract_clean_floor_mask(seg_map: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    Extracts a continuous, complete floor mask from the semantic segmentation map.
    - Includes floor (Label 3) and rugs/carpets (Label 28).
    - Bridges gaps across chair/table legs with morphological closing.
    - Retains ALL significant floor components (not just the largest).
    - Fills small/medium enclosed interior holes (gaps under tables, between chair legs).
    - Subtracts large foreground furniture (sofas, beds, cabinets).
    """
    # 1. Combine floor and rug/carpet labels
    floor_raw = ((seg_map == 3) | (seg_map == 28)).astype(np.uint8) * 255

    # 2. Morphological closing — bridges chair/table leg gaps
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    floor_closed = cv2.morphologyEx(floor_raw, cv2.MORPH_CLOSE, close_k)

    # 3. Keep all substantial floor components (> 0.2% of total image area)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(floor_closed, connectivity=8)
    floor_mask = np.zeros_like(floor_closed)
    min_comp_area = W * H * 0.002
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_comp_area:
            floor_mask[labels == lbl] = 255

    # 4. Fill small/medium interior holes geometrically enclosed in the floor
    contours, hierarchy = cv2.findContours(floor_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    floor_filled = floor_mask.copy()
    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            # h[3] != -1 means this contour is inside another contour (an internal hole)
            if h[3] != -1:
                area = cv2.contourArea(contours[i])
                # Only fill holes smaller than 6% of the room (e.g. gaps under chairs/tables)
                if area < (W * H * 0.06):
                    cv2.drawContours(floor_filled, [contours[i]], -1, 255, -1)

    # 5. Subtract heavy foreground objects that sit in front of the new floor
    for obj_label in [7, 10, 23]:   # bed, cabinet, sofa
        obj_mask = (seg_map == obj_label).astype(np.uint8) * 255
        if np.any(obj_mask > 0):
            obj_dilated = cv2.dilate(obj_mask, np.ones((5, 5), np.uint8))
            floor_filled[obj_dilated > 0] = 0

    return floor_filled




# ============================================================
# HELPER: Robust Wall Mask Extraction
# ============================================================
def extract_clean_wall_mask(seg_map: np.ndarray, floor_mask: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    Extracts contiguous wall segments. Floor pixels take precedence over walls.
    """
    wall_raw = (seg_map == 0).astype(np.uint8) * 255
    # Remove floor overlap
    wall_clean = cv2.bitwise_and(wall_raw, cv2.bitwise_not(floor_mask))
    
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    wall_closed = cv2.morphologyEx(wall_clean, cv2.MORPH_CLOSE, close_k)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(wall_closed, connectivity=8)
    wall_mask = np.zeros_like(wall_closed)
    min_comp_area = W * H * 0.005
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_comp_area:
            wall_mask[labels == lbl] = 255

    return wall_mask


# ============================================================
# HELPER: Extract simplified polygon from mask
# ============================================================
def mask_to_polygon(mask_u8: np.ndarray, epsilon_frac: float = 0.005) -> list:
    """
    Returns the simplified polygon boundary of the largest contour.
    """
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    epsilon = epsilon_frac * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    pts = approx.squeeze()
    if pts.ndim == 2:
        return pts.tolist()
    return []


# ============================================================
# HELPER: Mask → RGBA base64 PNG
# ============================================================
def mask_to_rgba_b64(mask_u8: np.ndarray, color: tuple = (255, 255, 255, 255)) -> str:
    h, w = mask_u8.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask_u8 > 0] = list(color)
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ============================================================
# HELPER: Robust Mask Upload Parser
# ============================================================
def parse_mask_upload(file_bytes: bytes) -> np.ndarray:
    if not file_bytes:
        return None
    pil_img = Image.open(io.BytesIO(file_bytes))
    if pil_img.mode == "RGBA":
        arr = np.array(pil_img)
        alpha = arr[:, :, 3]
        rgb_sum = arr[:, :, :3].sum(axis=-1)
        mask = (alpha > 20) | (rgb_sum > 20)
        return mask.astype(np.uint8) * 255
    else:
        gray = np.array(pil_img.convert("L"))
        return (gray > 20).astype(np.uint8) * 255


# ============================================================
# HELPER: Extended Perspective Floor Ground Plane Quad
# ============================================================
def compute_floor_quad(mask_u8: np.ndarray) -> list:
    """
    Notebook 'ROBUST FLOOR QUAD INFERENCE - VERSION 2' Implementation:
    - Extracts top envelope of floor mask across columns.
    - Fits robust line y = m*x + b using RANSAC on left and right visible floor regions.
    - Projects far boundary with true perspective slope (far_left_y, far_right_y).
    - Extends corners past canvas borders to guarantee 100% ground coverage.
    """
    H, W = mask_u8.shape
    ys, xs = np.where(mask_u8 > 0)
    if len(ys) == 0:
        return [
            [-int(W * 0.20), int(H * 0.40)],
            [int(W * 1.20),  int(H * 0.40)],
            [int(W * 1.35),  int(H * 1.05)],
            [-int(W * 0.35), int(H * 1.05)],
        ]

    # 1. Morphological closing to clean mask
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    clean_mask = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)

    # 2. Extract Top Envelope (first floor pixel in each x column)
    margin = int(W * 0.02)
    top_points = []
    for x in range(margin, W - margin):
        col_ys = np.where(clean_mask[:, x] > 0)[0]
        if len(col_ys) > 0:
            top_points.append([x, col_ys[0]])

    if len(top_points) < 10:
        far_y = int(np.percentile(ys, 5))
        return [
            [-int(W * 0.20), far_y],
            [int(W * 1.20),  far_y],
            [int(W * 1.35),  int(H * 1.05)],
            [-int(W * 0.35), int(H * 1.05)],
        ]

    top_pts_arr = np.array(top_points, dtype=np.float64)

    # 3. Split into Left and Right Envelope regions
    left_limit = W * 0.35
    right_limit = W * 0.65
    left_env = top_pts_arr[top_pts_arr[:, 0] <= left_limit]
    right_env = top_pts_arr[top_pts_arr[:, 0] >= right_limit]

    # 4. RANSAC Line Fit (y = m*x + b)
    def robust_line_fit(pts):
        if len(pts) < 8:
            return None
        px = pts[:, 0]
        py = pts[:, 1]
        best_count = 0
        best_model = None
        rng = np.random.default_rng(42)
        iterations = min(1000, len(pts) * 3)

        for _ in range(iterations):
            idx = rng.choice(len(pts), 2, replace=False)
            x1, x2 = px[idx]
            y1, y2 = py[idx]
            if abs(x2 - x1) < 1e-5:
                continue
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            residuals = np.abs(py - (m * px + b))
            inliers = residuals < 25
            count = np.count_nonzero(inliers)
            if count > best_count:
                best_count = count
                best_model = (m, b, inliers)

        if best_model is None:
            return None

        m, b, inliers = best_model
        x_in = px[inliers]
        y_in = py[inliers]
        if len(x_in) >= 2:
            A = np.column_stack([x_in, np.ones(len(x_in))])
            m, b = np.linalg.lstsq(A, y_in, rcond=None)[0]

        return {"m": float(m), "b": float(b), "count": best_count}

    left_fit = robust_line_fit(left_env)
    right_fit = robust_line_fit(right_env)

    # Combine fits or pick dominant slope
    m, b = 0.0, float(np.percentile(ys, 5))
    if left_fit and right_fit:
        if abs(left_fit["m"] - right_fit["m"]) < 0.35:
            m = (left_fit["m"] + right_fit["m"]) / 2.0
            b = (left_fit["b"] + right_fit["b"]) / 2.0
        elif left_fit["count"] >= right_fit["count"]:
            m, b = left_fit["m"], left_fit["b"]
        else:
            m, b = right_fit["m"], right_fit["b"]
    elif left_fit:
        m, b = left_fit["m"], left_fit["b"]
    elif right_fit:
        m, b = right_fit["m"], right_fit["b"]

    # Limit slope to realistic perspective bounds (-0.45 to +0.45)
    m = max(-0.45, min(0.45, m))

    # Calculate far-left and far-right y coordinates using perspective line equation
    far_left_y  = max(0, min(int(H * 0.80), int(m * (-W * 0.20) + b)))
    far_right_y = max(0, min(int(H * 0.80), int(m * (W * 1.20)  + b)))

    return [
        [-int(W * 0.20), far_left_y],        # P1: far left
        [int(W * 1.20),  far_right_y],       # P2: far right
        [int(W * 1.35),  int(H * 1.05)],     # P3: near right
        [-int(W * 0.35), int(H * 1.05)],     # P4: near left
    ]



# ============================================================
# HELPER: Per-segment wall quads with perspective inference
# ============================================================
def compute_wall_quads(wall_mask_u8: np.ndarray, min_area_frac: float = 0.003) -> list:
    """
    Computes true 4-corner perspective quads [TL, TR, BR, BL] for each wall segment
    in the room, accounting for left side-wall recession, right side-wall recession,
    and back-wall perspective.
    """
    H, W = wall_mask_u8.shape
    min_area = H * W * min_area_frac
    contours, _ = cv2.findContours(wall_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.03 * peri, True)
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w / 2.0

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            sorted_by_y = pts[np.argsort(pts[:, 1])]
            top_pts = sorted_by_y[:2]
            bot_pts = sorted_by_y[2:]
            tl = top_pts[np.argmin(top_pts[:, 0])].tolist()
            tr = top_pts[np.argmax(top_pts[:, 0])].tolist()
            bl = bot_pts[np.argmin(bot_pts[:, 0])].tolist()
            br = bot_pts[np.argmax(bot_pts[:, 0])].tolist()
            quad = [tl, tr, br, bl]
        else:
            if cx < W * 0.40:
                top_slope = int(h * 0.12)
                bot_slope = int(h * 0.10)
                quad = [
                    [x, max(0, y - top_slope)],
                    [x + w, y + top_slope],
                    [x + w, y + h - bot_slope],
                    [x, min(H, y + h + bot_slope)],
                ]
            elif cx > W * 0.60:
                top_slope = int(h * 0.12)
                bot_slope = int(h * 0.10)
                quad = [
                    [x, y + top_slope],
                    [x + w, max(0, y - top_slope)],
                    [x + w, min(H, y + h + bot_slope)],
                    [x, y + h - bot_slope],
                ]
            else:
                quad = [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h],
                ]

        quads.append({"quad": quad, "bbox": [x, y, w, h]})

    return quads


# ============================================================
# HELPER: Estimate real-world area (sq ft) from mask + pixel scale
# ============================================================
def estimate_area_sqft(
    mask_u8: np.ndarray,
    obstacle_ids: set,
    seg_map: np.ndarray,
    pixels_per_meter: float
) -> dict:
    total_px = int(np.count_nonzero(mask_u8))
    obstacle_detail = {}
    excluded_px = 0

    for label_id in obstacle_ids:
        obs_mask = (seg_map == label_id).astype(np.uint8) * 255
        k = np.ones((5, 5), np.uint8)
        obs_mask = cv2.morphologyEx(obs_mask, cv2.MORPH_CLOSE, k)
        overlap = cv2.bitwise_and(mask_u8, obs_mask)
        px = int(np.count_nonzero(overlap))
        if px > 0:
            label_name = ADE20K_LABELS.get(label_id, f"label_{label_id}")
            obstacle_detail[label_name] = round(px / (pixels_per_meter ** 2) * 10.764, 2)
            excluded_px += px

    net_px = max(0, total_px - excluded_px)
    sqm_per_px = 1.0 / (pixels_per_meter ** 2)

    return {
        "total_sqft":     round(total_px * sqm_per_px * 10.764, 2),
        "net_sqft":       round(net_px  * sqm_per_px * 10.764, 2),
        "obstacles":      obstacle_detail,
        "pixels_per_meter": round(pixels_per_meter, 2),
    }


# ============================================================
# HELPER: Derive pixels-per-meter from door/window in seg map
# ============================================================
def derive_pixel_scale(seg_map: np.ndarray, W: int, H: int) -> float:
    for label_id, ref in REAL_WORLD_REF.items():
        lmask = (seg_map == label_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(lmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        _, _, _, bbox_h = cv2.boundingRect(largest)
        if bbox_h < 20:
            continue
        ppm = bbox_h / ref["h"]
        print(f"[Scale] Using label {label_id} ({ADE20K_LABELS.get(label_id)}) => {ppm:.1f} px/m")
        return ppm

    ppm_fallback = (H * 0.40) / 1.0
    print(f"[Scale] Fallback heuristic => {ppm_fallback:.1f} px/m")
    return ppm_fallback


# ============================================================
# HELPER: Build perspective-correct tiled surface (flat)
# ============================================================
def build_tile_surface(
    tile_np: np.ndarray,
    plane_w: int,
    plane_h: int,
    tiles_x: int = 6,
    tiles_y: int = 6,
    pattern: str = "grid",  # "grid" | "brick"
    grout_width: int = 0,
    grout_color: tuple = (210, 210, 210),
    rotation_deg: float = 0,
    brightness: float = 1.0,
) -> np.ndarray:
    cell_w = max(4, plane_w // max(1, tiles_x))
    cell_h = max(4, plane_h // max(1, tiles_y))
    surface = np.zeros((plane_h, plane_w, 3), dtype=np.uint8)

    tile_bgr = cv2.cvtColor(tile_np, cv2.COLOR_RGB2BGR)

    if rotation_deg != 0:
        cx, cy = tile_bgr.shape[1] // 2, tile_bgr.shape[0] // 2
        M = cv2.getRotationMatrix2D((cx, cy), rotation_deg, 1.0)
        tile_bgr = cv2.warpAffine(tile_bgr, M, (tile_bgr.shape[1], tile_bgr.shape[0]),
                                   borderMode=cv2.BORDER_REFLECT)

    resized_tile = cv2.resize(tile_bgr, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

    # Shape-preserving rotation/flip variants for natural stone/marble realism
    rotations = [
        resized_tile,
        cv2.rotate(resized_tile, cv2.ROTATE_180),
        cv2.flip(resized_tile, 1),
        cv2.flip(resized_tile, 0),
    ]

    # Deterministic RNG seeded by tile position for organic natural stone variation
    rng = np.random.default_rng(seed=101)

    for row in range(tiles_y):
        for col in range(tiles_x):
            x_offset = 0
            if pattern == "brick" and row % 2 == 1:
                x_offset = cell_w // 2

            x1 = (col * cell_w + x_offset) % plane_w
            y1 = row * cell_h
            x2 = min(x1 + cell_w, plane_w)
            y2 = min(y1 + cell_h, plane_h)
            tw = x2 - x1
            th = y2 - y1

            # Select rotation variant for natural organic tiling (prevents wallpaper grid look)
            rot_idx = (row * 3 + col * 7) % len(rotations)
            base_var = rotations[rot_idx]

            # Micro-variation: ±1.5% subtle brightness jitter per tile instance
            jitter = 1.0 + rng.uniform(-0.015, 0.015)
            tile_instance = np.clip(base_var[:th, :tw].astype(np.float32) * jitter, 0, 255).astype(np.uint8)
            surface[y1:y2, x1:x2] = tile_instance


            if grout_width > 0:
                gc = grout_color
                gx2 = min(x2, plane_w - 1)
                surface[y1:y2, max(0, gx2 - grout_width):gx2] = gc
                gy2 = min(y2, plane_h - 1)
                surface[max(0, gy2 - grout_width):gy2, x1:x2] = gc

    if brightness != 1.0:
        surface = np.clip(surface.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    return surface


# ============================================================
# HELPER: Apply photorealistic homography warp
# ============================================================
def apply_homography_warp(
    room_bgr: np.ndarray,
    tile_surface: np.ndarray,
    dest_quad: list,
    mask_u8: np.ndarray,
    shadow_strength: float = 0.55,
    finish: str = "matte",
    feather_px: int = 4,
) -> np.ndarray:
    H, W = room_bgr.shape[:2]
    ph, pw = tile_surface.shape[:2]

    # Ensure mask matches room dimensions (may differ if segmented at different res)
    if mask_u8.shape[0] != H or mask_u8.shape[1] != W:
        mask_u8 = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)

    src_pts = np.array([
        [0,      0],
        [pw - 1, 0],
        [pw - 1, ph - 1],
        [0,      ph - 1],
    ], dtype=np.float32)

    dst_pts = np.array(dest_quad, dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(tile_surface, M, (W, H),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)

    # ── 1. SMOOTH ILLUMINATION MAP (Zero Original Texture Bleed) ────────
    # Extract macroscopic ambient light field — morphological opening completely
    # strips out any old floor seams, wood grain, rug patterns, or previous tile lines
    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k_size = max(21, (int(max(W, H) * 0.05) | 1))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    gray_opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, open_kernel)

    # Wide-band Gaussian blur captures only macroscopic room light gradients (zero local patterns)
    blur_k = max(51, (int(max(W, H) * 0.20) | 1))
    light_map = cv2.GaussianBlur(gray_opened, (blur_k, blur_k), 0)

    # Normalize against 95th percentile of the masked surface
    masked_vals = light_map[mask_u8 > 0]
    if len(masked_vals) > 0:
        wp = float(np.percentile(masked_vals, 95))
        wp = max(wp, 1.0)
        light_norm = np.clip(light_map / wp, 0.0, 1.0)
    else:
        light_norm = np.clip(light_map / 255.0, 0.0, 1.0)

    # ── 2. AMBIENT OCCLUSION (Grounding Furniture Contact Shadows) ─────
    dist_transform = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    max_dist = dist_transform.max()
    if max_dist > 0:
        # Subtle contact shadow at perimeter (outermost 6% of floor)
        ao_map = np.clip(dist_transform / (max_dist * 0.06), 0.0, 1.0).astype(np.float32)
    else:
        ao_map = np.ones_like(gray, dtype=np.float32)
    ao_map = cv2.GaussianBlur(ao_map, (7, 7), 0)

    # ── 3. SHADOW BLEND ──────────────────────────────────────
    shadow_blend = light_norm * shadow_strength + (1.0 - shadow_strength)

    # ── 4. COLOR TEMPERATURE HARMONIZATION (15% Ambient Match) ────────
    if np.any(mask_u8 > 0):
        erode_kernel = np.ones((7, 7), np.uint8)
        mask_interior = cv2.erode(mask_u8, erode_kernel, iterations=2)

        room_lab = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        warped_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2Lab).astype(np.float32)

        sample_mask = mask_interior if np.any(mask_interior > 0) else mask_u8
        room_a_mean = float(np.mean(room_lab[sample_mask > 0, 1]))
        room_b_mean = float(np.mean(room_lab[sample_mask > 0, 2]))
        tile_a_mean = float(np.mean(warped_lab[mask_u8 > 0, 1]))
        tile_b_mean = float(np.mean(warped_lab[mask_u8 > 0, 2]))

        # Harmonize tile color temperature with the room's ambient light
        da = (room_a_mean - tile_a_mean) * 0.15
        db = (room_b_mean - tile_b_mean) * 0.15
        warped_lab[:, :, 1] = np.clip(warped_lab[:, :, 1] + da, 0, 255)
        warped_lab[:, :, 2] = np.clip(warped_lab[:, :, 2] + db, 0, 255)
        warped = cv2.cvtColor(warped_lab.astype(np.uint8), cv2.COLOR_Lab2BGR)

    # ── 5. APPLY LIGHTING & CONTACT OCCLUSION ────────────────
    warped_f = warped.astype(np.float32)
    lm3 = np.stack([shadow_blend] * 3, axis=-1)
    lit_warped = warped_f * lm3

    # Ground furniture legs with gentle 8% AO contact shadow
    ao3 = np.stack([ao_map] * 3, axis=-1)
    lit_warped = lit_warped * (0.92 + 0.08 * ao3)

    # ── 6. REALISTIC CLEAN FINISH RESPONSE ───────────────────
    mask_binary_f = (mask_u8 > 0).astype(np.float32)
    mb3 = np.stack([mask_binary_f] * 3, axis=-1)

    if finish == "glossy":
        highlight = np.clip((light_norm - 0.80) / 0.20, 0.0, 1.0)
        hl3 = np.stack([highlight] * 3, axis=-1) * mb3
        lit_warped = np.clip(lit_warped + hl3 * 40, 0, 255)

    elif finish == "satin":
        highlight = np.clip((light_norm - 0.85) / 0.15, 0.0, 1.0)
        hl3 = np.stack([highlight] * 3, axis=-1) * mb3
        lit_warped = np.clip(lit_warped + hl3 * 20, 0, 255)

    # ── 7. TWO-LAYER ANTI-ALIASED MASK (Zero Old Tile Bleed) ──
    feather_px = max(1, feather_px)
    erode_k = np.ones((feather_px * 2 + 1, feather_px * 2 + 1), np.uint8)
    mask_eroded = cv2.erode(mask_u8, erode_k, iterations=1)
    mask_feathered = cv2.GaussianBlur(
        mask_u8.astype(np.float32) / 255.0,
        (feather_px * 2 + 1, feather_px * 2 + 1),
        float(feather_px) * 0.4
    )
    # Guaranteed 100% opacity in mask interior, smooth anti-aliasing at boundary
    mask_eroded_f = mask_eroded.astype(np.float32) / 255.0

    mask_f32 = np.maximum(mask_feathered, mask_eroded_f)

    # ── 8. PURE COMPOSITE (Zero Old Pixel Blendback) ─────────
    m3 = np.stack([mask_f32] * 3, axis=-1)
    result = room_bgr.astype(np.float32)
    composited = result * (1.0 - m3) + lit_warped * m3

    return np.clip(composited, 0, 255).astype(np.uint8)



# ============================================================
# HELPER: Mock masks (when model unavailable)
# ============================================================
def build_mock_masks(width: int, height: int):
    floor_mock = np.zeros((height, width), dtype=np.uint8)
    floor_pts = np.array([
        [0, height],
        [int(width * 0.28), int(height * 0.52)],
        [int(width * 0.72), int(height * 0.52)],
        [width, height],
    ], np.int32)
    cv2.fillPoly(floor_mock, [floor_pts], 255)

    wall_mock = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(wall_mock, (0, 0), (width, int(height * 0.55)), 255, -1)
    cv2.fillPoly(wall_mock, [floor_pts], 0)

    seg_mock = np.zeros((height, width), dtype=np.int32)
    seg_mock[wall_mock > 0] = 0
    seg_mock[floor_mock > 0] = 3

    return floor_mock, wall_mock, seg_mock


# ============================================================
# ENDPOINT: /api/segment
# ============================================================
@app.post("/api/segment")
async def segment_room(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = original_image.size

        is_mock = False
        seg_map_full = None

        floor_mask_u8 = np.zeros((height, width), dtype=np.uint8)
        wall_mask_u8  = np.zeros((height, width), dtype=np.uint8)

        if MODEL_AVAILABLE:
            try:
                inference_img = original_image.resize((1024, 768))
                inputs = processor(images=inference_img, return_tensors="pt")
                with __import__("torch").no_grad():
                    outputs = seg_model(**inputs)
                seg_small = processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[(768, 1024)]
                )[0].cpu().numpy()

                seg_map_full = cv2.resize(seg_small.astype(np.int32), (width, height),
                                          interpolation=cv2.INTER_NEAREST)

                # Extract complete continuous floor and wall masks
                floor_mask_u8 = extract_clean_floor_mask(seg_map_full, width, height)
                wall_mask_u8  = extract_clean_wall_mask(seg_map_full, floor_mask_u8, width, height)

            except Exception as model_err:
                print(f"Model inference error: {model_err}. Falling back to simulation.")
                is_mock = True
        else:
            is_mock = True

        if is_mock:
            floor_mask_u8, wall_mask_u8, seg_map_full = build_mock_masks(width, height)
        
        if seg_map_full is None:
            _, _, seg_map_full = build_mock_masks(width, height)

        # ── Compute quads ──────────────────────────────────────
        floor_quad  = compute_floor_quad(floor_mask_u8)
        wall_quads  = compute_wall_quads(wall_mask_u8)

        # ── Obstacle detection ─────────────────────────────────
        detected_obstacles = {}
        for lid, lname in ADE20K_LABELS.items():
            if lid in (0, 3):
                continue
            count = int(np.count_nonzero(seg_map_full == lid))
            if count > (width * height * 0.003):
                detected_obstacles[lname] = count

        # ── Real-world scale & Area estimation ─────────────────
        ppm = derive_pixel_scale(seg_map_full, width, height)
        floor_area = estimate_area_sqft(floor_mask_u8, FLOOR_OBSTACLE_IDS, seg_map_full, ppm)
        wall_area  = estimate_area_sqft(wall_mask_u8,  WALL_OBSTACLE_IDS,  seg_map_full, ppm)

        floor_poly = mask_to_polygon(floor_mask_u8)
        wall_poly  = mask_to_polygon(wall_mask_u8)

        return JSONResponse({
            "status": "success",
            "is_mock": is_mock,
            "width": width,
            "height": height,
            "floor_polygon": floor_poly,
            "wall_polygon": wall_poly,
            "floor_quad": floor_quad,
            "wall_quads": wall_quads,
            "detected_obstacles": detected_obstacles,
            "pixels_per_meter": ppm,
            "floor_area": floor_area,
            "wall_area": wall_area,
            "floor_mask": mask_to_rgba_b64(floor_mask_u8, color=(255, 255, 255, 255)),
            "wall_mask": mask_to_rgba_b64(wall_mask_u8, color=(255, 255, 255, 255)),
            "floor_mask_b64": mask_to_rgba_b64(floor_mask_u8, color=(255, 255, 255, 255)),
            "wall_mask_b64": mask_to_rgba_b64(wall_mask_u8, color=(255, 255, 255, 255)),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ============================================================
# ENDPOINT: /api/visualize
# ============================================================
@app.post("/api/visualize")
async def visualize_tiles(
    room:                  UploadFile = File(...),
    floor_tile:            UploadFile = File(None),
    wall_tile:             UploadFile = File(None),
    tile:                  UploadFile = File(None),  # Fallback for single-tile requests
    target:                str = Form("both"),       # "floor", "wall", or "both"
    floor_quad:            str = Form("[]"),
    wall_quads:            str = Form("[]"),
    floor_mask:            UploadFile = File(None),
    wall_mask:             UploadFile = File(None),
    # Floor params
    floor_scale:           float = Form(0.18),
    floor_rotation:        float = Form(0.0),
    floor_brightness:      float = Form(1.0),
    floor_pattern:         str   = Form("grid"),
    floor_grout_width:     int   = Form(0),
    floor_grout_color:     str   = Form("#cccccc"),
    floor_finish:          str   = Form("matte"),
    floor_shadow_strength: float = Form(0.55),
    floor_tiles_x:         int   = Form(6),
    floor_tiles_y:         int   = Form(6),
    # Wall params
    wall_scale:            float = Form(0.18),
    wall_rotation:         float = Form(0.0),
    wall_brightness:       float = Form(1.0),
    wall_pattern:          str   = Form("grid"),
    wall_grout_width:      int   = Form(0),
    wall_grout_color:      str   = Form("#cccccc"),
    wall_finish:           str   = Form("matte"),
    wall_shadow_strength:  float = Form(0.55),
    wall_tiles_x:          int   = Form(5),
    wall_tiles_y:          int   = Form(7),
    # Generic fallback params
    scale:                 float = Form(0.18),
    rotation:              float = Form(0.0),
    brightness:            float = Form(1.0),
    pattern:               str   = Form("grid"),
    grout_width:           int   = Form(0),
    grout_color:           str   = Form("#cccccc"),
    finish:                str   = Form("matte"),
    shadow_strength:       float = Form(0.55),
    tiles_x:               int   = Form(6),
    tiles_y:               int   = Form(6),
):
    try:
        room_bytes = await room.read()
        room_pil = Image.open(io.BytesIO(room_bytes)).convert("RGB")
        W, H = room_pil.size
        room_bgr = cv2.cvtColor(np.array(room_pil), cv2.COLOR_RGB2BGR)

        # Helper to parse hex colors
        def parse_grout_color(hex_str):
            h = hex_str.lstrip("#")
            if len(h) == 6:
                r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                return (b, g, r)
            return (210, 210, 210)

        # ── Parse Quads ────────────────────────────────────────
        fq = json.loads(floor_quad) if floor_quad else []
        wqs = json.loads(wall_quads) if wall_quads else []

        # ── Load Masks ─────────────────────────────────────────
        floor_mask_u8 = None
        wall_mask_u8  = None
        if floor_mask is not None:
            fm_bytes = await floor_mask.read()
            floor_mask_u8 = parse_mask_upload(fm_bytes)
        if wall_mask is not None:
            wm_bytes = await wall_mask.read()
            wall_mask_u8 = parse_mask_upload(wm_bytes)


        if floor_mask_u8 is None or wall_mask_u8 is None:
            f_mock, w_mock, _ = build_mock_masks(W, H)
            if floor_mask_u8 is None:
                floor_mask_u8 = f_mock
            if wall_mask_u8 is None:
                wall_mask_u8 = w_mock

        composite = room_bgr.copy()

        # ── 1. Apply FLOOR Tiling (only if floor_tile is provided or fallback tile with floor target) ──
        active_floor_file = floor_tile if floor_tile is not None else (tile if target in ("floor", "both") else None)
        if active_floor_file is not None:
            f_bytes = await active_floor_file.read()
            if f_bytes:
                f_pil = Image.open(io.BytesIO(f_bytes)).convert("RGB")
                f_tile_np = np.array(f_pil)

                if len(fq) < 4:
                    fq = compute_floor_quad(floor_mask_u8)

                plane_w, plane_h = 1600, 1200
                f_tx = max(3, floor_tiles_x if floor_tiles_x else tiles_x)
                f_ty = max(3, floor_tiles_y if floor_tiles_y else tiles_y)
                f_gc = parse_grout_color(floor_grout_color if floor_grout_color else grout_color)

                flat_floor = build_tile_surface(
                    f_tile_np, plane_w, plane_h, f_tx, f_ty,
                    pattern=floor_pattern or pattern,
                    grout_width=floor_grout_width if floor_grout_width is not None else grout_width,
                    grout_color=f_gc,
                    rotation_deg=floor_rotation if floor_rotation is not None else rotation,
                    brightness=floor_brightness if floor_brightness is not None else brightness,
                )
                composite = apply_homography_warp(
                    composite, flat_floor, fq, floor_mask_u8,
                    shadow_strength=floor_shadow_strength if floor_shadow_strength is not None else shadow_strength,
                    finish=floor_finish or finish,
                )

        # ── 2. Apply WALL Tiling (only if wall_tile is provided or fallback tile with wall target) ──
        active_wall_file = wall_tile if wall_tile is not None else (tile if target in ("wall", "both") and floor_tile is None else None)
        if active_wall_file is not None and wqs:
            w_bytes = await active_wall_file.read()
            if w_bytes:
                w_pil = Image.open(io.BytesIO(w_bytes)).convert("RGB")
                w_tile_np = np.array(w_pil)

                w_tx = max(3, wall_tiles_x if wall_tiles_x else tiles_x)
                w_ty = max(3, wall_tiles_y if wall_tiles_y else tiles_y)
                w_gc = parse_grout_color(wall_grout_color if wall_grout_color else grout_color)

                for seg in wqs:
                    seg_quad = seg.get("quad", [])
                    if len(seg_quad) < 4:
                        continue
                    bbox = seg.get("bbox", [0, 0, W, H])
                    bx, by, bw, bh = bbox

                    seg_mask = np.zeros((H, W), dtype=np.uint8)
                    seg_pts = np.array(seg_quad, np.int32)
                    cv2.fillPoly(seg_mask, [seg_pts], 255)
                    seg_mask = cv2.bitwise_and(seg_mask, wall_mask_u8)

                    if not np.any(seg_mask > 0):
                        continue

                    pw = max(200, min(bw * 2, 1600))
                    ph = max(200, min(bh * 2, 1200))
                    flat_wall = build_tile_surface(
                        w_tile_np, pw, ph, w_tx, w_ty,
                        pattern=wall_pattern or pattern,
                        grout_width=wall_grout_width if wall_grout_width is not None else grout_width,
                        grout_color=w_gc,
                        rotation_deg=wall_rotation if wall_rotation is not None else rotation,
                        brightness=wall_brightness if wall_brightness is not None else brightness,
                    )
                    composite = apply_homography_warp(
                        composite, flat_wall, seg_quad, seg_mask,
                        shadow_strength=wall_shadow_strength if wall_shadow_strength is not None else shadow_strength,
                        finish=wall_finish or finish,
                    )

        result_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_rgb)
        buf = io.BytesIO()
        result_pil.save(buf, format="JPEG", quality=93)
        b64 = base64.b64encode(buf.getvalue()).decode()

        return JSONResponse(content={
            "image": "data:image/jpeg;base64," + b64,
            "width": W,
            "height": H,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


# ============================================================
# Static frontend
# ============================================================
frontend_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(frontend_dir, "index.html")):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    @app.get("/")
    def read_root():
        return {"message": "AI Room Visualizer API v2.1 — Frontend not found."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
