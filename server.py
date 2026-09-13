import os
import io
import base64
import json
import logging
import numpy as np
from PIL import Image
import cv2

# Suppress HuggingFace hub unauthenticated notice
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# FastAPI Imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
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
    5:  "ceiling",
    7:  "bed",
    8:  "windowpane",
    9:  "window",
    10: "cabinet",
    14: "door",
    15: "table",
    17: "plant",
    18: "curtain",
    19: "chair",
    22: "painting",
    23: "sofa",
    24: "shelf",
    27: "mirror",
    28: "rug",
    30: "armchair",
    31: "seat",
    36: "wardrobe",
    39: "shelf",
    44: "chest of drawers",
    50: "desk",
    52: "pillow",
    53: "stairs",
    57: "headboard",
    64: "coffee table",
    75: "swivel chair",
    108: "cushion",
    110: "lamp",
    119: "pendant lamp",
    125: "flowerpot",
    132: "vase",
    137: "tableware",
}

# Labels for protected major furniture & structural entities (Bed, Table, Sofa, Armchair, Chair, Cabinet, Door, Painting, Bookcase, Countertop)
FLOOR_OBSTACLE_IDS = {7, 10, 14, 15, 18, 19, 22, 23, 24, 27, 30, 31, 44, 49, 50, 57, 64, 65, 69, 75, 80, 89, 128}

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
device = "cpu"

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
# OPTIONAL: Monocular depth (Depth-Anything-V2-Small)
# ============================================================
DEPTH_AVAILABLE = False
depth_processor = None
depth_model = None
try:
    if MODEL_AVAILABLE:
        from transformers import AutoImageProcessor as _DepthAIP, AutoModelForDepthEstimation
        _DEPTH_ID = "depth-anything/Depth-Anything-V2-Small-hf"
        try:
            depth_processor = _DepthAIP.from_pretrained(_DEPTH_ID)
            depth_model = AutoModelForDepthEstimation.from_pretrained(_DEPTH_ID)
        except Exception:
            depth_processor = _DepthAIP.from_pretrained(_DEPTH_ID, local_files_only=True)
            depth_model = AutoModelForDepthEstimation.from_pretrained(_DEPTH_ID, local_files_only=True)
        depth_model.to(device)
        depth_model.eval()
        DEPTH_AVAILABLE = True
        print("Depth-Anything-V2-Small loaded — depth-assisted wall analysis active.")
except Exception as e:
    print(f"Depth model unavailable — classical wall-plane fallback active. ({e})")

# Initialize Hybrid Multi-Stage Architectural Pipeline
try:
    from pipeline import RoomVisualizerPipeline
    visualizer_pipeline = RoomVisualizerPipeline(
        seg_model=seg_model,
        seg_processor=processor,
        depth_model=depth_model,
        depth_processor=depth_processor,
        device=device
    )
    print("RoomVisualizerPipeline initialized successfully!")
except Exception as pipe_err:
    print(f"Pipeline init error: {pipe_err}")
    visualizer_pipeline = None


# ============================================================
# Core Utilities & Mathematical Helpers
# ============================================================

def _guided_filter(guide01: np.ndarray, src01: np.ndarray, radius: int = 2, eps: float = 1e-4) -> np.ndarray:
    """Sub-pixel edge-aware guided filter for foreground extraction."""
    guide01 = guide01.astype(np.float32)
    src01 = src01.astype(np.float32)
    d = int(radius) * 2 + 1
    mean_I = cv2.boxFilter(guide01, -1, (d, d), borderType=cv2.BORDER_REFLECT)
    mean_p = cv2.boxFilter(src01, -1, (d, d), borderType=cv2.BORDER_REFLECT)
    mean_Ip = cv2.boxFilter(guide01 * src01, -1, (d, d), borderType=cv2.BORDER_REFLECT)
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = cv2.boxFilter(guide01 * guide01, -1, (d, d), borderType=cv2.BORDER_REFLECT) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return cv2.boxFilter(a, -1, (d, d), borderType=cv2.BORDER_REFLECT) * guide01 + cv2.boxFilter(b, -1, (d, d), borderType=cv2.BORDER_REFLECT)


def mask_to_polygon(mask_u8: np.ndarray, epsilon_frac: float = 0.005) -> list:
    """Returns the simplified polygon boundary of the largest contour."""
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    epsilon = epsilon_frac * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    pts = approx.squeeze()
    if pts.ndim == 2:
        return pts.tolist()
    return []


def mask_to_rgba_b64(mask_u8: np.ndarray, color: tuple = (255, 255, 255, 255), soft: bool = False) -> str:
    """Converts a binary or grayscale mask to base64 PNG data URL."""
    h, w = mask_u8.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if soft:
        rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = 255
        rgba[..., 3] = mask_u8
    else:
        r, g, b, a = color
        rgba[mask_u8 > 0] = [r, g, b, a]
    pil_img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _ransac_edge(pts, thresh: float, iters: int = 500):
    """Robust fit of x = m*y + k (y-parameterized, stable for near-vertical room edges)."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 6:
        return None
    xs, ys = pts[:, 0], pts[:, 1]
    n = len(pts)
    rng = np.random.default_rng(7)
    best_cnt, best = 0, None
    for _ in range(min(iters, n * 4)):
        i, j = rng.choice(n, 2, replace=False)
        if abs(ys[j] - ys[i]) < 1e-5:
            continue
        m = (xs[j] - xs[i]) / (ys[j] - ys[i])
        k = xs[i] - m * ys[i]
        res = np.abs(xs - (m * ys + k))
        cnt = int(np.count_nonzero(res < thresh))
        if cnt > best_cnt:
            best_cnt, best = cnt, (m, k, res < thresh)
    if best is None:
        return None
    m, k, inl = best
    if np.count_nonzero(inl) >= 2:
        A = np.column_stack([ys[inl], np.ones(int(np.count_nonzero(inl)))])
        m, k = np.linalg.lstsq(A, xs[inl], rcond=None)[0]
    return float(m), float(k), best_cnt


def estimate_vanishing_point(floor_mask_u8: np.ndarray, wall_mask_u8: np.ndarray, W: int, H: int) -> dict:
    """Recover dominant vanishing point from floor receding boundaries."""
    default = {"vp": [W * 0.5, H * 0.32], "horizon": H * 0.32, "confidence": 0.0}
    if floor_mask_u8 is None or not np.any(floor_mask_u8 > 0):
        return default

    clean = cv2.morphologyEx(
        floor_mask_u8, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )
    ys, xs = np.where(clean > 0)
    top, bot = int(ys.min()), int(ys.max())
    span = max(1, bot - top)

    left_pts, right_pts = [], []
    y0 = top + int(span * 0.04)
    y1 = bot - int(span * 0.28)
    for y in range(y0, max(y0 + 2, y1)):
        row = np.where(clean[y] > 0)[0]
        if len(row) < 5:
            continue
        xl, xr = int(row.min()), int(row.max())
        if xl > W * 0.02:
            left_pts.append([xl, y])
        if xr < W * 0.98:
            right_pts.append([xr, y])

    tol = max(6.0, W * 0.012)
    lfit = _ransac_edge(left_pts, tol)
    rfit = _ransac_edge(right_pts, tol)

    vp = None
    if lfit and rfit and lfit[2] >= 10 and rfit[2] >= 10 and abs(lfit[0] - rfit[0]) > 1e-3:
        vy = (rfit[1] - lfit[1]) / (lfit[0] - rfit[0])
        vx = lfit[0] * vy + lfit[1]
        if 0.05 * W <= vx <= 0.95 * W and 0.05 * H <= vy <= 0.50 * H:
            vp = [float(vx), float(vy)]

    conf = 0.9 if vp is not None else 0.3
    if vp is None:
        vx = float(np.clip(np.median(xs), W * 0.30, W * 0.70))
        vy = float(np.clip(top - span * 0.03, H * 0.12, H * 0.42))
        vp = [vx, vy]

    vp[1] = float(np.clip(vp[1], H * 0.05, H * 0.50))
    vp[0] = float(np.clip(vp[0], W * 0.05, W * 0.95))
    return {"vp": vp, "horizon": vp[1], "confidence": conf}


def compute_floor_quad(mask_u8: np.ndarray, vp=None, W=None, H=None, depth_map=None) -> list:
    """Computes perspective ground-plane quad [P1_far_left, P2_far_right, P3_near_right, P4_near_left]."""
    if H is None or W is None:
        H, W = mask_u8.shape

    ys, xs = np.where(mask_u8 > 0)
    if len(ys) == 0:
        return [[-int(W * 0.15), int(H * 0.5)], [int(W * 1.15), int(H * 0.5)],
                [int(W * 1.15), int(H * 1.02)], [-int(W * 0.15), int(H * 1.02)]]

    clean = cv2.morphologyEx(
        mask_u8, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(11, W // 45) | 1, max(11, W // 45) | 1))
    )
    col_has = clean.max(axis=0) > 0
    cols = np.where(col_has)[0]
    if len(cols) < 4:
        cols = np.arange(W)
    ftop = (clean > 0).argmax(axis=0).astype(np.float32)
    fbot = (H - 1 - (clean[::-1] > 0).argmax(axis=0)).astype(np.float32)
    top = float(ys.min())

    near_y = float(np.clip(np.percentile(fbot[cols], 90), top + H * 0.15, H - 1))
    near_band = clean[int(near_y - H * 0.07):int(near_y) + 1, :].max(axis=0) > 0
    nb = np.where(near_band)[0]
    near_lx = float(nb.min()) if len(nb) else 0.0
    near_rx = float(nb.max()) if len(nb) else float(W - 1)
    if near_lx <= W * 0.05:
        near_lx = -W * 0.10
    if near_rx >= W * 0.95:
        near_rx = W * 1.10

    far_y = float(np.percentile(ftop[cols], 10))
    far_y = float(np.clip(far_y, top, near_y - H * 0.12))
    far_band = clean[int(far_y):int(far_y + H * 0.08) + 1, :].max(axis=0) > 0
    fb = np.where(far_band)[0]
    if len(fb) > 6:
        far_lx = float(np.percentile(fb, 3))
        far_rx = float(np.percentile(fb, 97))
    else:
        far_lx, far_rx = W * 0.35, W * 0.65

    near_w = max(1.0, near_rx - near_lx)
    near_cx = 0.5 * (near_lx + near_rx)
    far_cx = 0.5 * (far_lx + far_rx)
    far_w = float(np.clip(far_rx - far_lx, 0.16 * near_w, 0.88 * near_w))
    far_cx += (near_cx - far_cx) * 0.4
    far_lx, far_rx = far_cx - far_w / 2.0, far_cx + far_w / 2.0

    near_y_ext = H * 1.02

    # Fit angled top baseline
    valid_x = cols[(cols >= far_lx) & (cols <= far_rx)]
    if len(valid_x) > 10:
        vx = valid_x.astype(np.float32)
        vy = ftop[valid_x]
        p_thresh = np.percentile(vy, 75)
        keep = vy <= (p_thresh + H * 0.1)
        if np.count_nonzero(keep) >= 5:
            vx, vy = vx[keep], vy[keep]
        if len(vx) >= 2 and (vx.max() - vx.min()) > 10:
            A = np.column_stack([vx, np.ones(len(vx), dtype=np.float32)])
            m_top, c_top = np.linalg.lstsq(A, vy, rcond=None)[0]
            m_top = float(np.clip(m_top, -0.45, 0.45))
            c_top = float(np.median(vy - m_top * vx))
            y_far_l = float(np.clip(m_top * far_lx + c_top, top, near_y - H * 0.1))
            y_far_r = float(np.clip(m_top * far_rx + c_top, top, near_y - H * 0.1))
        else:
            y_far_l = y_far_r = far_y
    else:
        y_far_l = y_far_r = far_y

    def extrap(xf, xn, yf, yn, yt):
        if abs(yn - yf) < 1e-3:
            return xn
        return xf + (yt - yf) / (yn - yf) * (xn - xf)

    nl_x = extrap(far_lx, near_lx, y_far_l, near_y, near_y_ext)
    nr_x = extrap(far_rx, near_rx, y_far_r, near_y, near_y_ext)

    return [
        [int(round(far_lx)), int(round(y_far_l))],
        [int(round(far_rx)), int(round(y_far_r))],
        [int(round(nr_x)),   int(round(near_y_ext))],
        [int(round(nl_x)),   int(round(near_y_ext))],
    ]


def derive_pixel_scale(seg_map: np.ndarray, W: int, H: int) -> float:
    """Derives pixels-per-meter scale from known architectural objects (doors/windows)."""
    for label_id, ref in REAL_WORLD_REF.items():
        if np.any(seg_map == label_id):
            ys, xs = np.where(seg_map == label_id)
            h_px = ys.max() - ys.min()
            if h_px > 40:
                return float(h_px / ref["h"])
    return float(W / 4.5)


def estimate_area_sqft(
    mask_u8: np.ndarray,
    obstacle_ids: set,
    seg_map: np.ndarray,
    pixels_per_meter: float
) -> dict:
    """Estimates gross and net square footage excluding obstacle objects."""
    if pixels_per_meter <= 0:
        pixels_per_meter = 200.0

    total_px = int(np.count_nonzero(mask_u8))
    sqm_per_px = 1.0 / (pixels_per_meter ** 2)
    total_sqft = round(total_px * sqm_per_px * 10.7639, 1)

    obstacles = {}
    if seg_map is not None:
        for lid in obstacle_ids:
            obs_px = int(np.count_nonzero((seg_map == lid) & (mask_u8 > 0)))
            if obs_px > (pixels_per_meter ** 2 * 0.05):
                obs_sqft = round((obs_px * sqm_per_px) * 10.7639, 1)
                lname = ADE20K_LABELS.get(lid, f"object_{lid}")
                obstacles[lname] = obs_sqft

    net_sqft = max(0.0, round(total_sqft - sum(obstacles.values()), 1))
    return {
        "total_sqft": total_sqft,
        "net_sqft": net_sqft,
        "obstacles": obstacles,
        "obstacle_sqft": round(sum(obstacles.values()), 1),
        "pixels_per_meter": round(pixels_per_meter, 2),
    }


def estimate_room_dims_ft(floor_quad: list, W: int, H: int, ppm: float) -> list:
    """Estimates physical room dimensions [width, depth] in feet."""
    if not floor_quad or len(floor_quad) < 4:
        return [14.0, 12.0]
    fq = np.array(floor_quad, dtype=np.float32)
    w_px = float((np.hypot(*(fq[1] - fq[0])) + np.hypot(*(fq[2] - fq[3]))) / 2.0)
    d_px = float((np.hypot(*(fq[3] - fq[0])) + np.hypot(*(fq[2] - fq[1]))) / 2.0)
    ppm_val = ppm if ppm > 1.0 else (W / 4.5)
    w_ft = max(6.0, round((w_px / ppm_val) * 3.28084, 1))
    d_ft = max(6.0, round((d_px / ppm_val) * 3.28084, 1))
    return [w_ft, d_ft]


def parse_grout_color(gc_str: str) -> tuple:
    """Parses hex color string into BGR tuple."""
    if not gc_str:
        return (210, 210, 210)
    try:
        s = gc_str.strip().lstrip("#")
        if len(s) == 6:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
            return (b, g, r)  # BGR
    except Exception:
        pass
    return (210, 210, 210)


# ============================================================
# Architectural Tile Pattern Synthesizers
# ============================================================

def _bevel_seam(surface: np.ndarray, x0: int, y0: int, x1: int, y1: int, gw: int, gcol: tuple):
    if gw <= 0:
        return
    h, w = surface.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    gcol_arr = np.array(gcol, dtype=np.float32)
    seg = surface[y0:y1, x0:x1].astype(np.float32)
    seg[:] = gcol_arr * 0.9
    surface[y0:y1, x0:x1] = np.clip(seg, 0, 255).astype(np.uint8)


def _apply_grid_grout(surface: np.ndarray, cell_w: int, cell_h: int, gw: int, gcol: tuple):
    h, w = surface.shape[:2]
    y = 0
    while y < h:
        _bevel_seam(surface, 0, y, w, y + gw, gw, gcol)
        y += cell_h
    x = 0
    while x < w + cell_w:
        _bevel_seam(surface, x, 0, x + gw, h, gw, gcol)
        x += cell_w


def _tile_variants(cell_bgr: np.ndarray) -> list:
    return [
        cell_bgr,
        cv2.rotate(cell_bgr, cv2.ROTATE_180),
        cv2.flip(cell_bgr, 1),
        cv2.flip(cell_bgr, 0),
    ]


def _render_bond(tile_bgr: np.ndarray, W: int, H: int, cell_w: int, cell_h: int, pattern: str, gw: int, gcol: tuple) -> np.ndarray:
    surface = np.zeros((H, W, 3), dtype=np.uint8)
    cell = cv2.resize(tile_bgr, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
    variants = _tile_variants(cell)
    rng = np.random.default_rng(101)

    n_rows = int(np.ceil(H / cell_h)) + 2
    n_cols = int(np.ceil(W / cell_w)) + 3
    vertical = pattern == "vertical"

    for row in range(-1, n_rows):
        for col in range(-1, n_cols):
            ox, oy = 0, 0
            if pattern == "brick" and row % 2:
                ox = cell_w // 2
            elif pattern == "brick_third" and row % 3:
                ox = (cell_w // 3) * (row % 3)
            elif vertical and col % 2:
                oy = cell_h // 2

            x1, y1 = col * cell_w + ox, row * cell_h + oy
            x1c, y1c = max(0, x1), max(0, y1)
            x2, y2 = min(x1 + cell_w, W), min(y1 + cell_h, H)
            if x2 <= x1c or y2 <= y1c:
                continue
            v = variants[(row * 3 + (col + 5) * 7) % 4]
            patch = v[y1c - y1:y2 - y1, x1c - x1:x2 - x1]
            if patch.shape[0] != y2 - y1c or patch.shape[1] != x2 - x1c:
                continue
            jit = 1.0 + rng.uniform(-0.02, 0.02)
            surface[y1c:y2, x1c:x2] = np.clip(patch.astype(np.float32) * jit, 0, 255).astype(np.uint8)

    if gw > 0:
        _apply_grid_grout(surface, cell_w, cell_h, gw, gcol)
    return surface


def _render_slab(tile_bgr: np.ndarray, W: int, H: int) -> np.ndarray:
    top = np.hstack([tile_bgr, cv2.flip(tile_bgr, 1)])
    block = np.vstack([top, cv2.flip(top, 0)])
    bh, bw = block.shape[:2]
    nx = max(1, int(np.ceil(W / (bw * 1.3))))
    ny = max(1, int(np.ceil(H / (bh * 1.3))))
    if nx > 1 or ny > 1:
        block = np.tile(block, (ny, nx, 1))
    interp = cv2.INTER_AREA if (block.shape[1] > W or block.shape[0] > H) else cv2.INTER_CUBIC
    return cv2.resize(block, (W, H), interpolation=interp)


def _paste(surface: np.ndarray, sprite: np.ndarray, x: int, y: int):
    H, W = surface.shape[:2]
    sh, sw = sprite.shape[:2]
    x0, y0 = int(x), int(y)
    xd0, yd0 = max(0, x0), max(0, y0)
    xd1, yd1 = min(W, x0 + sw), min(H, y0 + sh)
    if xd1 <= xd0 or yd1 <= yd0:
        return
    surface[yd0:yd1, xd0:xd1] = sprite[yd0 - y0:yd1 - y0, xd0 - x0:xd1 - x0]


def _render_basketweave(tile_bgr: np.ndarray, W: int, H: int, unit: int, gw: int, gcol: tuple) -> np.ndarray:
    surface = np.zeros((H, W, 3), dtype=np.uint8)
    u = max(8, int(unit))
    blk = u * 2
    hp = cv2.resize(tile_bgr, (blk, u), interpolation=cv2.INTER_AREA)
    vp = cv2.resize(cv2.rotate(tile_bgr, cv2.ROTATE_90_CLOCKWISE), (u, blk), interpolation=cv2.INTER_AREA)
    for bj, y in enumerate(range(-blk, H + blk, blk)):
        for bi, x in enumerate(range(-blk, W + blk, blk)):
            horiz = (bi + bj) % 2 == 0
            for p in (0, 1):
                if horiz:
                    _paste(surface, hp, x, y + p * u)
                else:
                    _paste(surface, vp, x + p * u, y)
    if gw > 0:
        _apply_grid_grout(surface, blk, blk, gw, gcol)
        _apply_grid_grout(surface, u, u, max(1, gw - 1), gcol)
    return surface


def _render_herringbone(tile_bgr: np.ndarray, W: int, H: int, unit: int, gw: int, gcol: tuple, k: int = 2) -> np.ndarray:
    b = max(6, int(unit))
    Hs = cv2.resize(tile_bgr, (k * b, b), interpolation=cv2.INTER_AREA)
    Vs = cv2.rotate(cv2.resize(tile_bgr, (k * b, b), interpolation=cv2.INTER_AREA), cv2.ROTATE_90_CLOCKWISE)

    yy, xx = np.mgrid[0:H, 0:W]
    i = xx // b
    j = yy // b
    r = (i + j) % (2 * k)
    horiz = r < k

    plank_i0 = i - r
    plank_j0 = j - (r - k)
    HmapX = np.clip(xx - plank_i0 * b, 0, k * b - 1).astype(np.intp)
    HmapY = np.clip(yy - j * b, 0, b - 1).astype(np.intp)
    VmapX = np.clip(xx - i * b, 0, b - 1).astype(np.intp)
    VmapY = np.clip(yy - plank_j0 * b, 0, k * b - 1).astype(np.intp)

    surface = np.where(horiz[..., None], Hs[HmapY, HmapX], Vs[VmapY, VmapX]).astype(np.uint8)

    if gw > 0:
        pid = np.where(horiz, (plank_i0 * 8192 + j) * 2, (i * 8192 + plank_j0) * 2 + 1)
        seam = np.zeros((H, W), np.uint8)
        seam[:, 1:] |= (pid[:, 1:] != pid[:, :-1]).astype(np.uint8)
        seam[1:, :] |= (pid[1:, :] != pid[:-1, :]).astype(np.uint8)
        if gw > 1:
            seam = cv2.dilate(seam, np.ones((gw, gw), np.uint8))
        surface[seam > 0] = np.clip(np.array(gcol, np.float32) * 0.72, 0, 255).astype(np.uint8)
    return surface


def _render_diagonal(tile_bgr: np.ndarray, W: int, H: int, cell_w: int, cell_h: int, base_pattern: str, gw: int, gcol: tuple) -> np.ndarray:
    d = int(np.hypot(W, H)) + 2 * max(cell_w, cell_h) + 4
    base = _render_bond(tile_bgr, d, d, cell_w, cell_h, base_pattern, gw, gcol)
    M = cv2.getRotationMatrix2D((d / 2.0, d / 2.0), 45, 1.0)
    rot = cv2.warpAffine(base, M, (d, d), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    y0, x0 = (d - H) // 2, (d - W) // 2
    return rot[y0:y0 + H, x0:x0 + W].copy()


def _render_chevron(tile_bgr: np.ndarray, W: int, H: int, unit: int, gw: int, gcol: tuple, ratio: int = 5) -> np.ndarray:
    w = max(6, int(unit))
    l = w * ratio
    plank = cv2.resize(tile_bgr, (l, w), interpolation=cv2.INTER_AREA)
    if gw > 0:
        plank[:max(1, gw), :] = np.clip(np.array(gcol, np.float32) * 0.72, 0, 255).astype(np.uint8)
    surface = np.zeros((H, W, 3), dtype=np.uint8)
    strip_h = H + 2 * l + 2 * w
    strip = np.zeros((strip_h, l, 3), dtype=np.uint8)
    yy = 0
    while yy < strip_h:
        _paste(strip, plank, 0, yy)
        yy += w
    nb = W // l + 2
    for bi in range(-1, nb):
        x0 = bi * l
        dirn = 1.0 if bi % 2 == 0 else -1.0
        Msh = np.float32([[1, 0, 0], [dirn, 1, -l if dirn > 0 else 0]])
        sheared = cv2.warpAffine(strip, Msh, (l, strip_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        seg = sheared[l:l + H]
        _paste(surface, seg, x0, 0)
        if gw > 0:
            _bevel_seam(surface, max(0, x0), 0, max(0, x0) + gw, H, gw, gcol)
    return surface


def _render_windmill(tile_bgr: np.ndarray, W: int, H: int, unit: int, gw: int, gcol: tuple) -> np.ndarray:
    u = max(8, int(unit))
    L = u * 2
    blk = L + u
    hp = cv2.resize(tile_bgr, (L, u), interpolation=cv2.INTER_AREA)
    vp = cv2.resize(cv2.rotate(tile_bgr, cv2.ROTATE_90_CLOCKWISE), (u, L), interpolation=cv2.INTER_AREA)
    ctr = cv2.resize(tile_bgr, (u, u), interpolation=cv2.INTER_AREA)
    surface = np.zeros((H, W, 3), dtype=np.uint8)
    for by in range(-blk, H + blk, blk):
        for bx in range(-blk, W + blk, blk):
            _paste(surface, hp, bx, by)
            _paste(surface, vp, bx + L, by)
            _paste(surface, hp, bx + u, by + L)
            _paste(surface, vp, bx, by + u)
            _paste(surface, ctr, bx + u, by + u)
    if gw > 0:
        _apply_grid_grout(surface, blk, blk, gw, gcol)
        _apply_grid_grout(surface, u, u, max(1, gw - 1), gcol)
    return surface


def build_tile_surface(
    tile_np: np.ndarray,
    plane_w: int,
    plane_h: int,
    tiles_x: float = 6,
    tiles_y: float = 6,
    pattern: str = "grid",
    grout_width: int = 0,
    grout_color: tuple = (210, 210, 210),
    rotation_deg: float = 0,
    brightness: float = 1.0,
    far_fade: float = 0.0,
    far_blur: bool = False,
    slab: bool = False,
) -> np.ndarray:
    """Renders a fronto-parallel tiled material surface."""
    tiles_x = max(1.0, float(tiles_x))
    tiles_y = max(1.0, float(tiles_y))
    cell_w = max(4, int(round(plane_w / tiles_x)))
    cell_h = max(4, int(round(plane_h / tiles_y)))

    tile_bgr = cv2.cvtColor(tile_np, cv2.COLOR_RGB2BGR)
    
    # Auto-crop outer margins
    th, tw = tile_bgr.shape[:2]
    if th > 100 and tw > 100:
        crop_top = int(th * 0.085)
        crop_bottom = int(th * 0.025)
        crop_left = int(tw * 0.03)
        crop_right = int(tw * 0.03)
        tile_bgr = tile_bgr[crop_top:th - crop_bottom, crop_left:tw - crop_right]

    if rotation_deg:
        cx, cy = tile_bgr.shape[1] // 2, tile_bgr.shape[0] // 2
        M = cv2.getRotationMatrix2D((cx, cy), rotation_deg, 1.0)
        tile_bgr = cv2.warpAffine(tile_bgr, M, (tile_bgr.shape[1], tile_bgr.shape[0]), borderMode=cv2.BORDER_REFLECT)

    gw = int(max(0, grout_width))
    gcol = tuple(int(c) for c in grout_color)
    small_unit = max(6, int(min(cell_w, cell_h) / 2.4))

    if slab or pattern == "slab":
        surface = _render_slab(tile_bgr, plane_w, plane_h)
    elif pattern == "herringbone":
        surface = _render_herringbone(tile_bgr, plane_w, plane_h, small_unit, max(2, gw), gcol)
    elif pattern == "basketweave":
        surface = _render_basketweave(tile_bgr, plane_w, plane_h, small_unit, max(2, gw), gcol)
    elif pattern == "chevron":
        surface = _render_chevron(tile_bgr, plane_w, plane_h, small_unit, max(2, gw), gcol)
    elif pattern == "windmill":
        surface = _render_windmill(tile_bgr, plane_w, plane_h, small_unit, max(2, gw), gcol)
    elif pattern in ("diagonal", "diagonal_brick"):
        surface = _render_diagonal(tile_bgr, plane_w, plane_h, cell_w, cell_h,
                                   "brick" if pattern == "diagonal_brick" else "grid", gw, gcol)
    else:
        surface = _render_bond(tile_bgr, plane_w, plane_h, cell_w, cell_h, pattern, gw, gcol)

    if brightness != 1.0:
        surface = np.clip(surface.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    if far_blur and plane_h > 40:
        blurred = cv2.GaussianBlur(surface, (0, 0), sigmaX=max(1.0, plane_w * 0.0011))
        wv = np.linspace(1.0, 0.0, plane_h, dtype=np.float32) ** 2.6
        surface = np.clip(surface * (1.0 - wv[:, None, None]) + blurred * wv[:, None, None],
                          0, 255).astype(np.uint8)

    if far_fade > 0.0:
        g = np.linspace(1.0 - float(np.clip(far_fade, 0, 0.45)), 1.0, plane_h, dtype=np.float32)
        surface = np.clip(surface.astype(np.float32) * g[:, None, None], 0, 255).astype(np.uint8)

    return surface


def plan_floor_tiling(quad: list, W: int, H: int, ppm: float, tile_wmm: float = 600.0, tile_hmm: float = 600.0, size_mult: float = 1.0):
    """Calculates grid dimensions and resolution for ground-plane homography projection."""
    q = np.asarray(quad, dtype=np.float64)
    far_w = float(np.hypot(*(q[1] - q[0])))
    near_w = float(np.hypot(*(q[2] - q[3])))
    near_w = float(np.clip(near_w, W * 0.6, W * 1.35))

    ppm = float(ppm) if ppm else 0.0
    if W / 9.0 <= ppm <= W / 2.5:
        room_w_m = float(np.clip(near_w / ppm, 2.6, 9.0))
    else:
        room_w_m = 6.0

    fk = float(np.clip(near_w / max(far_w, 1.0), 1.05, 6.0))
    room_d_m = float(np.clip(room_w_m * (0.32 * fk), 2.4, 12.0))

    tw = max(0.25, (float(tile_wmm) / 1000.0) * size_mult * 1.8)
    th = max(0.25, (float(tile_hmm) / 1000.0) * size_mult * 1.8)

    tiles_x = max(2.0, room_w_m / tw)
    tiles_y = max(2.0, room_d_m / th)

    if tiles_x > 12.0:
        s = 12.0 / tiles_x
        tiles_x, tiles_y = 12.0, max(2.0, tiles_y * s)
    if tiles_y > 14.0:
        s = 14.0 / tiles_y
        tiles_y, tiles_x = 14.0, max(2.0, tiles_x * s)

    plane_w = 2048
    plane_h = int(np.clip(round(plane_w * room_d_m / room_w_m), 512, 4096))
    return tiles_x, tiles_y, plane_w, plane_h


def apply_homography_warp(
    room_bgr: np.ndarray,
    tile_surface: np.ndarray,
    dest_quad: list,
    mask_u8: np.ndarray,
    shadow_strength: float = 0.55,
    finish: str = "satin",
    feather_px: int = 3,
    fg_alpha: np.ndarray = None,
) -> np.ndarray:
    """Applies perspective homography warping, contact AO, and lighting modulation to floor surface."""
    H, W = room_bgr.shape[:2]
    ph, pw = tile_surface.shape[:2]

    if not np.any(mask_u8 > 0):
        return room_bgr

    if mask_u8.shape[0] != H or mask_u8.shape[1] != W:
        mask_u8 = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)

    # 1. Perspective Homography
    src_pts = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], dtype=np.float32)
    dst_pts = np.array(dest_quad, dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(tile_surface, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 2. Extract Room Illumination Field
    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur_k = int(W * 0.06) | 1
    gray_blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    floor_px = gray_blurred[mask_u8 > 0]
    if len(floor_px) > 0:
        white_pt = float(np.percentile(floor_px, 90))
        shadow_map = np.clip(gray_blurred / max(white_pt, 1.0), 0.0, 1.0)
        shadow_map = np.clip(0.70 * shadow_map + 0.30, 0.0, 1.0)
    else:
        shadow_map = np.ones((H, W), dtype=np.float32)

    # Compute Contact AO from foreground objects and floor perimeter
    ao_map = np.ones((H, W), dtype=np.float32)
    if fg_alpha is not None and np.any(fg_alpha > 0.1):
        fg_binary = (fg_alpha > 0.2).astype(np.uint8)
        non_fg = (1 - fg_binary) * 255
        dist = cv2.distanceTransform(non_fg.astype(np.uint8), cv2.DIST_L2, 5)
        ao_norm = np.clip(dist / 18.0, 0.0, 1.0)
        ao_map = 1.0 - (1.0 - ao_norm) ** 1.8 * 0.42

    combined_lighting = shadow_map * ao_map
    light_gain = (1.0 - shadow_strength) + shadow_strength * combined_lighting
    material_lit = warped.astype(np.float32) * light_gain[:, :, None]

    # Specular response
    if finish == "glossy":
        specular = np.clip((combined_lighting - 0.75) / 0.25, 0.0, 1.0)
        material_lit = material_lit * (1.0 - 0.18 * specular[:, :, None]) + room_bgr.astype(np.float32) * (0.18 * specular[:, :, None])
    elif finish == "satin":
        specular = np.clip((combined_lighting - 0.82) / 0.18, 0.0, 1.0)
        material_lit = material_lit * (1.0 - 0.08 * specular[:, :, None]) + room_bgr.astype(np.float32) * (0.08 * specular[:, :, None])

    # 3. Crisp Anti-Aliased Edge Feathering Blend (3x3)
    alpha_mask = cv2.GaussianBlur((mask_u8 > 0).astype(np.float32), (3, 3), 0)[:, :, None]
    final_comp = material_lit * alpha_mask + room_bgr.astype(np.float32) * (1.0 - alpha_mask)

    # 4. Strict Foreground Re-composition: Objects on top of tiles
    if fg_alpha is not None:
        fg_a = fg_alpha[:, :, None] if fg_alpha.ndim == 2 else fg_alpha
        if fg_a.shape[0] != H or fg_a.shape[1] != W:
            fg_a = cv2.resize(fg_a, (W, H))
            if fg_a.ndim == 2:
                fg_a = fg_a[:, :, None]
        final_comp = final_comp * (1.0 - fg_a) + room_bgr.astype(np.float32) * fg_a

    return np.clip(final_comp, 0, 255).astype(np.uint8)


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
        floor_quad = []

        room_bgr = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

        global visualizer_pipeline
        if visualizer_pipeline is None:
            from pipeline import RoomVisualizerPipeline
            visualizer_pipeline = RoomVisualizerPipeline(
                seg_model=seg_model,
                seg_processor=processor,
                depth_model=depth_model,
                depth_processor=depth_processor,
                device="cpu"
            )

        clean_wall_mask = np.zeros((height, width), dtype=bool)
        wall_quads = []
        wall_planes = []
        room_metrics = {}

        if MODEL_AVAILABLE and visualizer_pipeline is not None:
            try:
                pipe_out = visualizer_pipeline.process_room(image_bgr=room_bgr)
                seg_map_full = pipe_out["seg_map"]
                clean_mask = pipe_out["clean_mask"]
                floor_mask_u8 = (clean_mask.astype(np.uint8) * 255)
                floor_quad = pipe_out["floor_quad"]
                clean_wall_mask = pipe_out["clean_wall_mask"]
                wall_quads = pipe_out["wall_quads"]
                wall_planes = pipe_out["wall_planes"]
                room_metrics = pipe_out["room_metrics"]
            except Exception as model_err:
                print(f"Model inference notice: {model_err}")
                is_mock = True

        if is_mock or seg_map_full is None:
            floor_mask_u8 = np.zeros((height, width), dtype=np.uint8)
            pts = np.array([[0, int(height * 0.55)], [width, int(height * 0.55)], [width, height], [0, height]], dtype=np.int32)
            cv2.fillPoly(floor_mask_u8, [pts], 255)
            seg_map_full = np.zeros((height, width), dtype=np.int32)
            floor_quad = [[-int(width * 0.15), int(height * 0.5)], [int(width * 1.15), int(height * 0.5)],
                          [int(width * 1.15), int(height * 1.02)], [-int(width * 0.15), int(height * 1.02)]]
            clean_wall_mask = np.zeros((height, width), dtype=bool)
            clean_wall_mask[:int(height * 0.55), :] = True
            wall_quads = [[[0, 0], [width, 0], [width, int(height * 0.55)], [0, int(height * 0.55)]]]

        wall_mask_u8 = (clean_wall_mask.astype(np.uint8) * 255)
        persp = estimate_vanishing_point(floor_mask_u8, wall_mask_u8, width, height)
        vp = persp["vp"]

        detected_obstacles = {}
        for lid, lname in ADE20K_LABELS.items():
            if lid in (0, 3, 5):
                continue
            count = int(np.count_nonzero(seg_map_full == lid))
            if count > (width * height * 0.003):
                detected_obstacles[lname] = count

        ppm = derive_pixel_scale(seg_map_full, width, height)
        floor_area = estimate_area_sqft(floor_mask_u8, FLOOR_OBSTACLE_IDS, seg_map_full, ppm)
        floor_poly = mask_to_polygon(floor_mask_u8)

        # Extract Guided Foreground Matte for Protected Major Objects on Floor
        fg_raw = np.zeros((height, width), dtype=np.uint8)
        for lid in FLOOR_OBSTACLE_IDS:
            fg_raw[seg_map_full == lid] = 255
        
        # Suppress small non-structural clutter (clothes, shoes, bags, boxes, toys)
        num_fg, labels_fg, stats_fg, _ = cv2.connectedComponentsWithStats(fg_raw, connectivity=8)
        min_fg_area = max(450, int(width * height * 0.006))
        for lbl in range(1, num_fg):
            w = stats_fg[lbl, cv2.CC_STAT_WIDTH]
            h = stats_fg[lbl, cv2.CC_STAT_HEIGHT]
            area = stats_fg[lbl, cv2.CC_STAT_AREA]
            if area < min_fg_area or (w < 60 and h < 60):
                fg_raw[labels_fg == lbl] = 0

        fg_floor = cv2.bitwise_and(fg_raw, cv2.dilate(floor_mask_u8, np.ones((9, 9), np.uint8)))
        gray = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY)
        guide = gray.astype(np.float32) / 255.0
        fg_f32 = fg_floor.astype(np.float32) / 255.0
        fg_alpha = np.clip(_guided_filter(guide, fg_f32, radius=2, eps=1e-4), 0.0, 1.0)
        fg_alpha = np.where(fg_alpha > 0.40, 1.0, fg_alpha * 0.35)
        floor_fg_u8 = (np.clip(fg_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)

        wall_planes_serializable = []
        for p in wall_planes:
            wall_planes_serializable.append({
                "id": str(p.get("id", "")),
                "name": str(p.get("name", "")),
                "quad": [[float(pt[0]), float(pt[1])] for pt in p.get("quad", [])],
                "bbox": [int(x) for x in p.get("bbox", [])],
                "area_px": int(p.get("area_px", 0)),
            })

        room_metrics_serializable = {}
        for k, v in room_metrics.items():
            if isinstance(v, (np.floating, float)):
                room_metrics_serializable[k] = round(float(v), 2)
            else:
                room_metrics_serializable[k] = int(v)

        gross_w_sqft = float(room_metrics_serializable.get("gross_wall_sqft", 0.0))
        net_w_sqft = float(room_metrics_serializable.get("net_wall_sqft", 0.0))
        wall_area = {
            "total_sqft": gross_w_sqft,
            "net_sqft": net_w_sqft,
            "obstacles": {
                "Openings (Doors & Windows)": round(gross_w_sqft - net_w_sqft, 1)
            } if gross_w_sqft > net_w_sqft else {}
        }

        return JSONResponse({
            "status": "success",
            "is_mock": is_mock,
            "width": width,
            "height": height,
            "floor_polygon": floor_poly,
            "floor_quad": floor_quad,
            "wall_quads": wall_quads,
            "wall_planes": wall_planes_serializable,
            "room_metrics": room_metrics_serializable,
            "perspective": {
                "vanishing_point": [round(vp[0], 1), round(vp[1], 1)],
                "horizon_y": round(vp[1], 1),
                "confidence": round(persp["confidence"], 2),
                "room_dims_ft": [float(room_metrics_serializable.get("width_ft", 14.0)), float(room_metrics_serializable.get("depth_ft", 12.0)), float(room_metrics_serializable.get("height_ft", 9.3))],
            },
            "detected_obstacles": detected_obstacles,
            "pixels_per_meter": ppm,
            "floor_area": floor_area,
            "wall_area": wall_area,
            "floor_mask": mask_to_rgba_b64(floor_mask_u8, color=(255, 255, 255, 255)),
            "floor_mask_b64": mask_to_rgba_b64(floor_mask_u8, color=(255, 255, 255, 255)),
            "floor_fg_mask": mask_to_rgba_b64(floor_fg_u8, color=(255, 255, 255, 255)),
            "wall_mask": mask_to_rgba_b64(wall_mask_u8, color=(255, 255, 255, 255)),
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
async def visualize_room(
    room:             UploadFile = File(...),
    tile:             UploadFile = File(None),
    floor_tile:       UploadFile = File(None),
    wall_tile:        UploadFile = File(None),
    floor_mask:       UploadFile = File(None),
    wall_mask:        UploadFile = File(None),
    floor_fg_mask:    UploadFile = File(None),
    floor_quad:       str = Form(None),
    floor_scale:      float = Form(0.18),
    floor_rotation:   float = Form(0.0),
    floor_pattern:    str = Form("grid"),
    floor_grout_width: int = Form(0),
    floor_grout_color: str = Form("#d2d2d2"),
    floor_brightness: float = Form(1.0),
    floor_shadow_strength: float = Form(0.55),
    floor_finish:     str = Form("satin"),
    floor_slab:       int = Form(0),
    wall_scale:       float = Form(0.18),
    wall_rotation:    float = Form(0.0),
    wall_pattern:     str = Form("grid"),
    wall_grout_width: int = Form(0),
    wall_grout_color: str = Form("#d2d2d2"),
    wall_brightness:  float = Form(1.0),
    wall_shadow_strength: float = Form(0.55),
    wall_finish:      str = Form("satin"),
    wall_slab:        int = Form(0),
    scale:            float = Form(0.18),
    rotation:         float = Form(0.0),
    pattern:          str = Form("grid"),
    grout_width:      int = Form(0),
    grout_color:      str = Form("#d2d2d2"),
    brightness:       float = Form(1.0),
    shadow_strength:  float = Form(0.55),
    finish:           str = Form("satin"),
    pixels_per_meter: float = Form(0.0),
):
    try:
        room_bytes = await room.read()
        room_pil = Image.open(io.BytesIO(room_bytes)).convert("RGB")
        W, H = room_pil.size
        room_bgr = cv2.cvtColor(np.array(room_pil), cv2.COLOR_RGB2BGR)

        # Floor Mask
        floor_mask_u8 = None
        if floor_mask is not None:
            fm_bytes = await floor_mask.read()
            if fm_bytes:
                fm_pil = Image.open(io.BytesIO(fm_bytes))
                if fm_pil.mode in ("RGBA", "LA") or (fm_pil.mode == "P" and "transparency" in fm_pil.info):
                    fm_np = np.array(fm_pil.convert("RGBA"))
                    floor_mask_u8 = (fm_np[..., 3] > 10).astype(np.uint8) * 255
                else:
                    fm_np = np.array(fm_pil.convert("L"))
                    floor_mask_u8 = (fm_np > 10).astype(np.uint8) * 255

        if floor_mask_u8 is None or not np.any(floor_mask_u8 > 0):
            floor_mask_u8 = np.zeros((H, W), dtype=np.uint8)
            pts = np.array([[0, int(H * 0.55)], [W, int(H * 0.55)], [W, H], [0, H]], dtype=np.int32)
            cv2.fillPoly(floor_mask_u8, [pts], 255)

        # Floor Foreground Alpha Matte
        floor_fg_alpha = None
        if floor_fg_mask is not None:
            ffg_bytes = await floor_fg_mask.read()
            if ffg_bytes:
                ffg_pil = Image.open(io.BytesIO(ffg_bytes))
                if ffg_pil.mode in ("RGBA", "LA") or (ffg_pil.mode == "P" and "transparency" in ffg_pil.info):
                    ffg_np = np.array(ffg_pil.convert("RGBA"))
                    floor_fg_alpha = ffg_np[..., 3].astype(np.float32) / 255.0
                else:
                    ffg_np = np.array(ffg_pil.convert("L"))
                    floor_fg_alpha = ffg_np.astype(np.float32) / 255.0

        # Floor Quad
        fq = []
        if floor_quad:
            try:
                fq = json.loads(floor_quad)
            except Exception:
                fq = []
        if len(fq) < 4:
            fq = compute_floor_quad(floor_mask_u8)

        if visualizer_pipeline is not None:
            clean_bg, _ = visualizer_pipeline.occlusion_detector.suppress_and_blur_clutter(
                image_bgr=room_bgr,
                seg_map=np.zeros((H, W), dtype=np.int32),
                floor_mask=(floor_mask_u8 > 0) if floor_mask_u8 is not None else None,
                wall_mask=None
            )
            composite = clean_bg.copy()
        else:
            composite = room_bgr.copy()

        # Apply Floor Tiling
        active_tile_file = floor_tile if floor_tile is not None else (tile if not wall_tile else None)
        if active_tile_file is not None:
            t_bytes = await active_tile_file.read()
            if t_bytes:
                t_pil = Image.open(io.BytesIO(t_bytes)).convert("RGB")
                t_np = np.array(t_pil)

                size_mult = float(np.clip((floor_scale or scale or 0.18) / 0.18, 0.4, 3.0))
                f_tx, f_ty, plane_w, plane_h = plan_floor_tiling(
                    fq, W, H, pixels_per_meter,
                    600.0, 600.0, size_mult
                )
                f_gc = parse_grout_color(floor_grout_color or grout_color)

                flat_floor = build_tile_surface(
                    t_np, plane_w, plane_h, f_tx, f_ty,
                    pattern=floor_pattern or pattern,
                    grout_width=floor_grout_width if floor_grout_width is not None else grout_width,
                    grout_color=f_gc,
                    rotation_deg=floor_rotation if floor_rotation is not None else rotation,
                    brightness=floor_brightness if floor_brightness is not None else brightness,
                    slab=bool(floor_slab),
                )
                composite = apply_homography_warp(
                    composite, flat_floor, fq, floor_mask_u8,
                    shadow_strength=floor_shadow_strength if floor_shadow_strength is not None else shadow_strength,
                    finish=floor_finish or finish,
                    fg_alpha=floor_fg_alpha,
                )

        # Apply Wall Tiling
        if wall_tile is not None:
            wt_bytes = await wall_tile.read()
            if wt_bytes:
                wt_pil = Image.open(io.BytesIO(wt_bytes)).convert("RGB")
                wt_np = np.array(wt_pil)
                if visualizer_pipeline is not None:
                    wm_np = None
                    if wall_mask is not None:
                        wm_bytes = await wall_mask.read()
                        if wm_bytes:
                            wm_pil = Image.open(io.BytesIO(wm_bytes)).convert("L")
                            wm_np = np.array(wm_pil) > 10
                    
                    if wm_np is None or not np.any(wm_np):
                        seg_out = visualizer_pipeline.process_room(room_bgr)
                        wm_np = seg_out["clean_wall_mask"]
                        w_quads = seg_out["wall_quads"]
                        w_planes = seg_out["wall_planes"]
                    else:
                        # Enforce strict mutual exclusivity with floor mask
                        wm_np = wm_np & (floor_mask_u8 == 0)
                        room_box = visualizer_pipeline.perspective_engine.compute_room_box_geometry(fq, wm_np, W, H)
                        w_planes = visualizer_pipeline.wall_engine.partition_wall_planes(wm_np, None, W, H, floor_quad=fq, room_box=room_box)
                        w_quads = [p["quad"] for p in w_planes] if w_planes else [room_box["back_wall_quad"]]

                    if np.any(wm_np):
                        w_mult = float(np.clip((wall_scale or scale or 0.18) / 0.18, 0.4, 3.0))
                        w_planes_list = w_planes if w_planes else [{"mask": wm_np, "quad": w_quads[0]}]
                        w_gc = parse_grout_color(wall_grout_color or grout_color)
                        for plane in w_planes_list:
                            p_mask = plane["mask"]
                            p_quad = plane["quad"]
                            w_tx, w_ty, w_pw, w_ph = visualizer_pipeline.perspective_engine.plan_tiling(
                                p_quad, W, H, pixels_per_meter, 600.0, 600.0, w_mult
                            )
                            w_flat = build_tile_surface(
                                wt_np, w_pw, w_ph, w_tx, w_ty,
                                pattern=wall_pattern or pattern,
                                grout_width=wall_grout_width if wall_grout_width is not None else grout_width,
                                grout_color=w_gc,
                                rotation_deg=wall_rotation if wall_rotation is not None else rotation,
                                brightness=wall_brightness if wall_brightness is not None else brightness,
                                slab=bool(wall_slab),
                            )
                            composite = visualizer_pipeline.wall_engine.warp_wall_material(
                                composite, w_flat, p_quad, p_mask,
                                shadow_strength=wall_shadow_strength if wall_shadow_strength is not None else shadow_strength,
                                finish=wall_finish or finish,
                                fg_alpha=floor_fg_alpha
                            )

        result_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_rgb)
        buf = io.BytesIO()
        result_pil.save(buf, format="JPEG", quality=98, subsampling=0)
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
# ENDPOINT: /api/debug_pipeline (9-Stage Inspection Engine)
# ============================================================
@app.post("/api/debug_pipeline")
async def debug_pipeline_endpoint(
    room: UploadFile = File(...),
    tile: UploadFile = File(None),
):
    try:
        room_bytes = await room.read()
        room_pil = Image.open(io.BytesIO(room_bytes)).convert("RGB")
        room_bgr = cv2.cvtColor(np.array(room_pil), cv2.COLOR_RGB2BGR)

        tile_bgr = None
        if tile is not None:
            t_bytes = await tile.read()
            if t_bytes:
                t_pil = Image.open(io.BytesIO(t_bytes)).convert("RGB")
                tile_bgr = cv2.cvtColor(np.array(t_pil), cv2.COLOR_RGB2BGR)
        
        if tile_bgr is None:
            tile_bgr = np.full((600, 600, 3), (245, 245, 248), dtype=np.uint8)
            cv2.line(tile_bgr, (0, 80), (600, 520), (195, 198, 208), 2)
            cv2.line(tile_bgr, (600, 150), (0, 480), (210, 212, 222), 1)

        global visualizer_pipeline
        if visualizer_pipeline is None:
            from pipeline import RoomVisualizerPipeline
            visualizer_pipeline = RoomVisualizerPipeline(
                seg_model=seg_model,
                seg_processor=processor,
                depth_model=depth_model,
                depth_processor=depth_processor,
                device="cpu"
            )

        out = visualizer_pipeline.process_room(
            image_bgr=room_bgr,
            tile_bgr=tile_bgr,
            tile_config={"pattern": "grid", "finish": "satin", "shadow_strength": 0.55},
            generate_debug=True
        )

        stages_b64 = {}
        for key, img in out["debug_stages"].items():
            if img is not None:
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                stages_b64[key] = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

        montage = visualizer_pipeline.compositor.build_debug_montage(out["debug_stages"])
        _, m_buf = cv2.imencode(".jpg", montage, [cv2.IMWRITE_JPEG_QUALITY, 95])
        montage_b64 = "data:image/jpeg;base64," + base64.b64encode(m_buf.tobytes()).decode()

        return JSONResponse({
            "status": "success",
            "montage": montage_b64,
            "stages": stages_b64,
            "floor_quad": out["floor_quad"]
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ============================================================
# ENDPOINT: /api/catalog and /api/catalog/sync (Automated Drive Sync)
# ============================================================
@app.get("/api/catalog")
@app.get("/api/catalog/sync")
@app.post("/api/catalog/sync")
async def sync_drive_catalog(request: Request):
    try:
        from pipeline.drive_sync import DriveCatalogSync
        syncer = DriveCatalogSync()
        
        synced_tiles = syncer.scan_local_catalog()
        if not synced_tiles or request.method == "POST":
            try:
                synced_tiles = syncer.sync_catalog()
            except Exception as se:
                print(f"[DriveSync] Live fetch notice: {se}")
                synced_tiles = syncer.scan_local_catalog()

        return JSONResponse({
            "status": "success",
            "count": len(synced_tiles),
            "tiles": synced_tiles
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ============================================================
# Static frontend mount
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
