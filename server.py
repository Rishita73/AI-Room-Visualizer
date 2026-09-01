import os
import io
import base64
import json
import numpy as np
from PIL import Image
import cv2

# Suppress HuggingFace hub unauthenticated notice
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

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
    53: "stairs",
    64: "coffee table",
    75: "swivel chair",
}

# Labels excluded from floor net area & mask (furniture/decor on floor)
FLOOR_OBSTACLE_IDS = {7, 10, 14, 15, 17, 18, 19, 22, 23, 24, 27, 30, 31, 36, 39, 44, 50, 53, 64, 75}

# Labels excluded from wall net area calculation (openings/fixtures on wall)
WALL_OBSTACLE_IDS = {8, 9, 14, 22, 27}

# Everything that visually sits IN FRONT of a wall and must not be tiled over
WALL_OCCLUDER_IDS = {7, 8, 9, 10, 14, 15, 17, 18, 19, 22, 23, 24, 27, 30, 31, 36, 39, 44, 50, 53, 64, 75}

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
# OPTIONAL: Monocular depth (Depth-Anything-V2-Small) — graceful fallback
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


def estimate_depth_map(pil_img, W, H):
    """
    Relative depth as float32 HxW in [0, 1] where 0 = nearest, 1 = farthest.
    Returns None when the depth model is not available.
    """
    if not DEPTH_AVAILABLE:
        return None
    try:
        import torch as _t
        inp = depth_processor(images=pil_img, return_tensors="pt")
        with _t.inference_mode():
            pred = depth_model(**inp).predicted_depth[0].cpu().numpy().astype(np.float32)
        pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_CUBIC)
        lo, hi = float(np.percentile(pred, 1)), float(np.percentile(pred, 99))
        norm = np.clip((pred - lo) / max(1e-6, hi - lo), 0.0, 1.0)
        return 1.0 - norm            # Depth-Anything outputs disparity (near = large)
    except Exception as ex:
        print(f"[depth] inference failed: {ex}")
        return None


# ============================================================
# Heavy solid ground-resting floor obstacles ONLY
SOLID_FLOOR_OBSTACLES = {7, 10, 14, 23, 27, 30, 31, 36}

# ============================================================
# HELPER: Robust Full Floor Mask Extraction
# ============================================================
def extract_clean_floor_mask(seg_map: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    Continuous floor mask that covers the WHOLE visible floor.

    Strategy: the floor of a room is a roughly-convex trapezoid. Take the raw
    floor/rug pixels, convex-fill them so gaps under furniture legs / rugs / dark
    corners are bridged, then carve back out the solid furniture footprints and
    anything that is clearly wall / ceiling / window so the floor never climbs a
    vertical surface.
    """
    floor_raw = ((seg_map == 3) | (seg_map == 28)).astype(np.uint8) * 255
    if not floor_raw.any():
        return np.zeros((H, W), np.uint8)

    # ---- convex fill of the significant floor blobs -------------------------
    filled = floor_raw.copy()
    cnts, _ = cv2.findContours(floor_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in cnts if cv2.contourArea(c) > W * H * 0.0015]
    if big:
        hull = cv2.convexHull(np.vstack(big))
        cv2.fillPoly(filled, [hull], 255)
    filled = cv2.morphologyEx(
        filled, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(9, W // 40) | 1, max(9, W // 40) | 1))
    )

    # ---- carve out everything that is NOT walkable floor -------------------
    carve = np.zeros((H, W), np.uint8)
    for fid in (SOLID_FLOOR_OBSTACLES
                | {15, 17, 18, 19, 24, 30, 31, 39, 44, 50, 64, 75}):   # + open furniture footprints
        carve[seg_map == fid] = 255
    carve = cv2.dilate(carve, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    filled[carve > 0] = 0
    for vid in (0, 5, 8, 9, 14):                       # wall / ceiling / window / door
        filled[seg_map == vid] = 0

    # ---- keep the dominant component that reaches the lower frame ----------
    num, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    best, best_area = 0, 0
    for lbl in range(1, num):
        x, y, w, h, a = stats[lbl]
        if a > best_area and (y + h) > H * 0.55 and a > W * H * 0.01:
            best, best_area = lbl, a
    floor_mask = np.zeros_like(filled)
    if best:
        floor_mask[labels == best] = 255
    else:
        floor_mask = filled

    # gentle close to smooth the boundary, re-carve furniture so it stays out
    floor_mask = cv2.morphologyEx(
        floor_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    floor_mask[carve > 0] = 0
    for vid in (0, 5, 8, 9, 14):
        floor_mask[seg_map == vid] = 0
    return floor_mask


# All furniture, decor, openings, and ceiling/floor that sit in front of or border walls
ALL_WALL_OCCLUDER_IDS = {
    3,   # floor
    5,   # ceiling
    7,   # bed
    8,   # windowpane
    9,   # window
    10,  # cabinet
    14,  # door
    15,  # table
    17,  # plant
    18,  # curtain
    19,  # chair
    22,  # painting
    23,  # sofa
    24,  # shelf
    27,  # mirror
    28,  # rug
    30,  # armchair
    31,  # seat
    36,  # wardrobe
    39,  # shelf
    44,  # chest of drawers
    50,  # desk
    53,  # stairs
    64,  # coffee table
    75,  # swivel chair
}

# ============================================================
# HELPER: Precise Wall Mask Extraction (Window Column Protection)
# ============================================================
def extract_clean_wall_mask(seg_map: np.ndarray, floor_mask: np.ndarray, W: int, H: int, floor_quad=None) -> np.ndarray:
    """
    Strict SegFormer wall pixels (Label 0) minus openings and objects in front of
    the wall. This is only a *fallback* / sanity check — `build_wall_region`
    provides the real paintable area because SegFormer routinely drops large
    patches of plain wall (behind sofas, near windows, in shadow).
    """
    wall_raw = (seg_map == 0).astype(np.uint8) * 255

    hard = np.zeros((H, W), np.uint8)     # never wall: openings, doors, ceiling
    for oid in (5, 8, 9, 14):
        hard[seg_map == oid] = 255
    hard = cv2.dilate(hard, np.ones((5, 5), np.uint8))

    soft = np.zeros((H, W), np.uint8)     # objects in front of the wall
    for oid in (7, 10, 15, 17, 18, 19, 22, 23, 24, 27, 30, 31, 36, 39, 44, 50, 64, 75):
        soft[seg_map == oid] = 255
    soft = cv2.dilate(soft, np.ones((3, 3), np.uint8))

    wall_clean = cv2.bitwise_and(wall_raw, cv2.bitwise_not(cv2.bitwise_or(hard, soft)))
    wall_clean[floor_mask > 0] = 0
    wall_clean = cv2.morphologyEx(wall_clean, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(wall_clean, connectivity=8)
    wall_mask = np.zeros_like(wall_clean)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= W * H * 0.0008:
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
def mask_to_rgba_b64(mask_u8: np.ndarray, color: tuple = (255, 255, 255, 255), soft: bool = False) -> str:
    h, w = mask_u8.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if soft:
        rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = 255
        rgba[..., 3] = mask_u8                       # preserve the alpha gradient
    else:
        rgba[mask_u8 > 0] = list(color)
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ============================================================
# HELPER: Robust Mask Upload Parser
# ============================================================
def parse_mask_upload(file_bytes: bytes, soft: bool = False) -> np.ndarray:
    if not file_bytes:
        return None
    pil_img = Image.open(io.BytesIO(file_bytes))
    if pil_img.mode == "RGBA":
        arr = np.array(pil_img)
        alpha = arr[:, :, 3]
        if soft:
            return alpha.astype(np.uint8)                 # keep the gradient
        rgb_sum = arr[:, :, :3].sum(axis=-1)
        return ((alpha > 20) | (rgb_sum > 20)).astype(np.uint8) * 255
    else:
        gray = np.array(pil_img.convert("L"))
        return gray.astype(np.uint8) if soft else (gray > 20).astype(np.uint8) * 255


# ============================================================
# HELPER: Projective geometry primitives
# ============================================================
def _line_through(p1, p2):
    """Homogeneous line a*x + b*y + c = 0 through two points."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = -(a * x1 + b * y1)
    return (a, b, c)


def _intersect(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    d = a1 * b2 - a2 * b1
    if abs(d) < 1e-9:
        return None
    return [(b1 * c2 - b2 * c1) / d, (a2 * c1 - a1 * c2) / d]


def _ray_x_at_y(vp, through, y):
    """X coordinate where the ray (vp -> through) crosses horizontal line `y`."""
    vx, vy = vp
    tx, ty = through
    if abs(ty - vy) < 1e-6:
        return tx
    t = (y - vy) / (ty - vy)
    return vx + t * (tx - vx)


def _ransac_edge(pts, thresh, iters=500):
    """
    Robust fit of x = m*y + k (y-parameterised, stable for near-vertical room
    edges). Returns (m, k, inlier_count) or None.
    """
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


# ============================================================
# HELPER: Vanishing point / horizon estimation
# ============================================================
def estimate_vanishing_point(floor_mask_u8, wall_mask_u8, W, H):
    """
    Recover the dominant (depth-axis) vanishing point from the floor's left and
    right receding boundaries. Floor side-edges and side-wall horizontals all
    share this VP, so it anchors a single consistent room perspective.
    """
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
        if xl > W * 0.02:              # genuine interior edge, not the frame
            left_pts.append([xl, y])
        if xr < W * 0.98:
            right_pts.append([xr, y])

    tol = max(6.0, W * 0.012)
    lfit = _ransac_edge(left_pts, tol)
    rfit = _ransac_edge(right_pts, tol)

    vp = None
    if lfit and rfit and lfit[2] >= 10 and rfit[2] >= 10 and abs(lfit[0] - rfit[0]) > 1e-3:
        vy = (rfit[1] - lfit[1]) / (lfit[0] - rfit[0])   # x=m*y+k equated
        vx = lfit[0] * vy + lfit[1]
        # a wall-facing room shot has its VP roughly central and above the floor
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


# ============================================================
# HELPER: Perspective floor ground-plane quad
# ============================================================
def compute_floor_quad(mask_u8: np.ndarray, vp=None, W=None, H=None) -> list:
    """
    A trapezoid that hugs the actual floor mask: near edge = where the floor
    reaches the bottom of the frame, far edge = the back-wall contact line.
    Built straight from the mask so it stays sane even when the vanishing-point
    estimate is unreliable.

    Returned corner order: [far-left, far-right, near-right, near-left].
    """
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

    # --- near edge (front of the room) ------------------------------------
    near_y = float(np.clip(np.percentile(fbot[cols], 90), top + H * 0.15, H - 1))
    near_band = clean[int(near_y - H * 0.07):int(near_y) + 1, :].max(axis=0) > 0
    nb = np.where(near_band)[0]
    near_lx = float(nb.min()) if len(nb) else 0.0
    near_rx = float(nb.max()) if len(nb) else float(W - 1)
    if near_lx <= W * 0.05:
        near_lx = -W * 0.10
    if near_rx >= W * 0.95:
        near_rx = W * 1.10

    # --- far edge (back-wall contact) -----------------------------------
    # robust: 10th-percentile of per-column tops over floor-bearing columns
    far_y = float(np.percentile(ftop[cols], 10))
    far_y = float(np.clip(far_y, top, near_y - H * 0.12))
    far_band = clean[int(far_y):int(far_y + H * 0.08) + 1, :].max(axis=0) > 0
    fb = np.where(far_band)[0]
    if len(fb) > 6:
        far_lx = float(np.percentile(fb, 3))
        far_rx = float(np.percentile(fb, 97))
    else:
        far_lx, far_rx = W * 0.35, W * 0.65

    # --- perspective sanity: far edge must be narrower & roughly centred ---
    near_w = max(1.0, near_rx - near_lx)
    near_cx = 0.5 * (near_lx + near_rx)
    far_cx = 0.5 * (far_lx + far_rx)
    far_w = float(np.clip(far_rx - far_lx, 0.16 * near_w, 0.88 * near_w))
    far_cx += (near_cx - far_cx) * 0.4          # pull the far edge toward centre
    far_lx, far_rx = far_cx - far_w / 2.0, far_cx + far_w / 2.0

    # --- extrapolate near corners below the frame along each side edge ----
    near_y_ext = H * 1.02

    def extrap(xf, xn, yf, yn, yt):
        if abs(yn - yf) < 1e-3:
            return xn
        return xf + (yt - yf) / (yn - yf) * (xn - xf)

    nl_x = extrap(far_lx, near_lx, far_y, near_y, near_y_ext)
    nr_x = extrap(far_rx, near_rx, far_y, near_y, near_y_ext)

    return [
        [int(round(far_lx)), int(round(far_y))],
        [int(round(far_rx)), int(round(far_y))],
        [int(round(nr_x)),   int(round(near_y_ext))],
        [int(round(nl_x)),   int(round(near_y_ext))],
    ]


# ============================================================
# HELPER: Geometric wall paint region (fills what SegFormer misses)
# ============================================================
def build_wall_region(seg_map, floor_mask_u8, W, H, floor_quad=None, room_bgr=None):
    """
    The paintable wall area = the band between the ceiling line and the floor's
    top edge, minus openings (windows/doors) and objects in front of the wall
    (furniture, curtains, art, mirrors, TVs, plants).

    This is the PRIMARY wall mask — SegFormer's `wall` label routinely drops
    huge patches of plain wall, so we reconstruct the surface geometrically.
    """
    raw_wall = (seg_map == 0).astype(np.uint8)
    ceiling = (seg_map == 5).astype(np.uint8)

    # ---- lower bound: floor's top edge, per column ------------------------
    fm = cv2.morphologyEx(floor_mask_u8, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    floor_top = np.full(W, float(H), np.float32)
    has_floor = fm.max(axis=0) > 0
    if has_floor.any():
        floor_top[has_floor] = (fm > 0).argmax(axis=0)[has_floor].astype(np.float32)
    if floor_quad and len(floor_quad) == 4:
        far_y = float((floor_quad[0][1] + floor_quad[1][1]) / 2.0)
    else:
        far_y = float(np.median(floor_top[has_floor])) if has_floor.any() else H * 0.6
    floor_top[~has_floor] = far_y
    floor_top = cv2.GaussianBlur(floor_top.reshape(1, -1), (0, 0), max(1.0, W * 0.02)).reshape(-1)

    # ---- upper bound: ceiling line, per column --------------------------
    ceil_line = np.zeros(W, np.float32)
    has_ceil = ceiling.max(axis=0) > 0
    if has_ceil.any():
        ceil_bottom = (H - 1 - (ceiling[::-1] > 0).argmax(axis=0)).astype(np.float32)
        ceil_line[has_ceil] = ceil_bottom[has_ceil]
    hw = (raw_wall.max(axis=0) > 0) & (~has_ceil)
    if hw.any():
        ceil_line[hw] = raw_wall.argmax(axis=0)[hw].astype(np.float32)
    ceil_line = cv2.GaussianBlur(ceil_line.reshape(1, -1), (0, 0), max(1.0, W * 0.03)).reshape(-1)
    # the ceiling MUST stay well above the floor edge (keeps the wall band tall)
    ceil_line = np.minimum(ceil_line, floor_top - H * 0.12)
    ceil_line = np.clip(ceil_line, 0.0, H * 0.60)

    ys = np.arange(H, dtype=np.float32)[:, None]
    region = ((ys >= ceil_line[None, :]) & (ys < floor_top[None, :])).astype(np.uint8) * 255

    # ---- subtract openings / doors / mirrors (dilated wide) -------------
    hard = np.zeros((H, W), np.uint8)
    for lid in (8, 9, 14, 27):
        hard |= (seg_map == lid).astype(np.uint8)
    if hard.any():
        hard = cv2.dilate(hard * 255,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(7, int(W * 0.03)) | 1,) * 2))
        region[hard > 0] = 0

    # ---- subtract foreground objects & lower furniture bodies (without splitting wall plane) ----
    occ = np.zeros((H, W), np.uint8)
    for lid in (10, 15, 17, 18, 19, 22, 24, 30, 31, 36, 39, 44, 49, 50, 53, 64, 75):
        occ |= (seg_map == lid).astype(np.uint8)

    # Bed (7) and Sofa (23): subtract lower body near floor, preserve upper wall plane behind headboard/backrest
    for lid in (7, 23):
        m = (seg_map == lid).astype(np.uint8) * 255
        if m.any():
            ys_idx, _ = np.where(m > 0)
            mid_y = int(np.percentile(ys_idx, 40))
            lower_body = np.zeros((H, W), dtype=np.uint8)
            lower_body[ys_idx[ys_idx >= mid_y], :] = 255
            occ |= (cv2.bitwise_and(lower_body, m) > 0).astype(np.uint8)

    if occ.any():
        occ = cv2.dilate(occ * 255, np.ones((5, 5), np.uint8))
        region[occ > 0] = 0
    region[floor_mask_u8 > 0] = 0

    # ---- drop dark recesses / curtains SegFormer mis-labelled as wall ---
    if room_bgr is not None:
        g = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)
        wall_vals = g[(raw_wall > 0)]
        if wall_vals.size > 200:
            thr = max(18.0, float(np.percentile(wall_vals, 12)) - 22.0)
            dark = cv2.morphologyEx((g < thr).astype(np.uint8) * 255, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            region[dark > 0] = 0

    # Close horizontal gaps across back wall behind headboards / sofa backs
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9)))
    region = cv2.morphologyEx(region, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    num, lab, st, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    out = np.zeros((H, W), np.uint8)
    for i in range(1, num):
        if st[i, cv2.CC_STAT_AREA] > W * H * 0.002:
            out[lab == i] = 255
    return out


# ============================================================
# HELPER: Room-box wall quads (share the floor vanishing point)
# ============================================================
def compute_wall_quads(wall_mask_u8: np.ndarray, floor_quad=None, vp=None,
                       min_area_frac: float = 0.004) -> list:
    """
    Room-box wall planes derived from the floor quad's far corners:
      - BACK wall spans between the two side walls (or the whole frame),
      - LEFT / RIGHT walls recede from the frame edge to the room corner,
        their floor edge following the floor quad's side edge.
    Each quad is later intersected with the real wall mask, so overhang is fine.
    """
    H, W = wall_mask_u8.shape
    quads = []
    if not np.any(wall_mask_u8 > 0):
        return quads

    def _mk(tl, tr, br, bl, bbox):
        return {
            "quad": [[int(round(p[0])), int(round(p[1]))] for p in (tl, tr, br, bl)],
            "bbox": [int(bbox[0]), int(bbox[1]), int(max(1, bbox[2])), int(max(1, bbox[3]))],
        }

    if floor_quad and len(floor_quad) == 4:
        fl_far  = [float(floor_quad[0][0]), float(floor_quad[0][1])]
        fr_far  = [float(floor_quad[1][0]), float(floor_quad[1][1])]
        fr_near = [float(floor_quad[2][0]), float(floor_quad[2][1])]
        fl_near = [float(floor_quad[3][0]), float(floor_quad[3][1])]
    else:
        fl_far, fr_far = [W * 0.18, H * 0.42], [W * 0.82, H * 0.42]
        fr_near, fl_near = [W * 1.15, H * 1.05], [-W * 0.15, H * 1.05]

    wys, wxs = np.where(wall_mask_u8 > 0)
    ceil_y = float(np.clip(np.percentile(wys, 1), 0.0, H * 0.45))
    far_y = float((fl_far[1] + fr_far[1]) / 2.0)
    wall_lx, wall_rx = float(wxs.min()), float(wxs.max())

    flx = float(np.clip(fl_far[0], 0.0, W))
    frx = float(np.clip(fr_far[0], 0.0, W))

    has_left  = (fl_far[0] > W * 0.06) and int((wxs < fl_far[0]).sum()) > W * H * 0.002
    has_right = (fr_far[0] < W * 0.94) and int((wxs > fr_far[0]).sum()) > W * H * 0.002

    # ---- BACK WALL -------------------------------------------------------
    bx0 = flx if has_left else -W * 0.03
    bx1 = frx if has_right else W * 1.03
    quads.append(_mk([bx0, ceil_y], [bx1, ceil_y], [bx1, far_y], [bx0, far_y],
                     [max(0.0, bx0), ceil_y, bx1 - bx0, far_y - ceil_y]))

    # ---- LEFT WALL (bottom edge follows the floor's left edge) ----------
    if has_left:
        lx = min(wall_lx, fl_near[0])
        quads.append(_mk([lx, ceil_y], [flx, ceil_y], [flx, fl_far[1]], [lx, fl_near[1]],
                         [0.0, ceil_y, max(1.0, flx), H - ceil_y]))

    # ---- RIGHT WALL ---------------------------------------------------
    if has_right:
        rx = max(wall_rx, fr_near[0])
        quads.append(_mk([frx, ceil_y], [rx, ceil_y], [rx, fr_near[1]], [frx, fr_far[1]],
                         [frx, ceil_y, W - frx, H - ceil_y]))

    return quads


# ============================================================
# ============================================================
#  WALL PIPELINE v2  (Phases A–C: detection · masking · planes)
#  Floor pipeline is untouched.
# ============================================================
# ============================================================

def _guided_filter(guide01, src01, radius=6, eps=1e-4):
    """Edge-aware smoothing of `src01` guided by `guide01` (both float32 [0,1])."""
    guide01 = guide01.astype(np.float32)
    src01 = src01.astype(np.float32)
    d = int(radius) * 2 + 1
    mean_I = cv2.boxFilter(guide01, -1, (d, d))
    mean_p = cv2.boxFilter(src01, -1, (d, d))
    mean_Ip = cv2.boxFilter(guide01 * src01, -1, (d, d))
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = cv2.boxFilter(guide01 * guide01, -1, (d, d)) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return cv2.boxFilter(a, -1, (d, d)) * guide01 + cv2.boxFilter(b, -1, (d, d))


def wall_band_bounds(seg_map, floor_mask, W, H, floor_quad=None):
    """Per-column ceiling_y (top of the paintable wall) and floor_y (bottom)."""
    ceiling = (seg_map == 5).astype(np.uint8)
    wall_raw = (seg_map == 0).astype(np.uint8)
    fm = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))

    floor_y = np.full(W, float(H), np.float32)
    hf = fm.max(axis=0) > 0
    if hf.any():
        floor_y[hf] = (fm > 0).argmax(axis=0)[hf].astype(np.float32)
    if floor_quad and len(floor_quad) == 4:
        far_y = float((floor_quad[0][1] + floor_quad[1][1]) / 2.0)
    else:
        far_y = float(np.median(floor_y[hf])) if hf.any() else H * 0.62
    floor_y[~hf] = far_y
    floor_y = cv2.GaussianBlur(floor_y.reshape(1, -1), (0, 0), max(1.0, W * 0.02)).reshape(-1)

    ceil_y = np.zeros(W, np.float32)
    hc = ceiling.max(axis=0) > 0
    if hc.any():
        ceil_y[hc] = (H - 1 - (ceiling[::-1] > 0).argmax(axis=0)[hc]).astype(np.float32)
    hw = (wall_raw.max(axis=0) > 0) & (~hc)
    if hw.any():
        ceil_y[hw] = wall_raw.argmax(axis=0)[hw].astype(np.float32)
    ceil_y = cv2.GaussianBlur(ceil_y.reshape(1, -1), (0, 0), max(1.0, W * 0.03)).reshape(-1)
    ceil_y = np.minimum(ceil_y, floor_y - H * 0.12)
    ceil_y = np.clip(ceil_y, 0.0, H * 0.62)
    return ceil_y, floor_y


# ADE labels that are definitely NOT paintable wall (objects / openings in front)
_WALL_FG_LABELS = (7, 8, 9, 10, 14, 15, 17, 18, 19, 22, 23, 24, 30, 31, 36, 39, 44, 49, 50, 53, 64, 75)


def detect_wall_foreground(seg_map, room_bgr, band_u8, depth, W, H):
    """
    Union of everything that sits in front of / instead of the wall, combining
    semantic labels + brightness + rectangle detection + (optional) depth.
    Returns (fg_u8, parts_dict) — parts kept for the debug panel.
    """
    parts = {}
    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)
    fg = np.zeros((H, W), np.uint8)

    # 1. semantic objects SegFormer does label
    sem = np.zeros((H, W), np.uint8)
    for lid in _WALL_FG_LABELS:
        sem[seg_map == lid] = 255
    parts["semantic"] = sem.copy()
    fg |= sem

    # 2. bright blobs inside the band = windows / mirrors SegFormer missed
    bright = cv2.bitwise_and((gray > 236).astype(np.uint8) * 255, band_u8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nb, lb, sb, _ = cv2.connectedComponentsWithStats(bright, 8)
    bmask = np.zeros((H, W), np.uint8)
    for i in range(1, nb):
        if sb[i, cv2.CC_STAT_AREA] > W * H * 0.0012:
            bmask[lb == i] = 255
    parts["bright"] = bmask
    fg |= bmask

    # 3. framed art / TV / mirror = strong closed rectangles inside the band
    interior_band = cv2.erode(band_u8, np.ones((11, 11), np.uint8))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
    edges = cv2.bitwise_and(edges, interior_band)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = np.zeros((H, W), np.uint8)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < W * H * 0.004 or area > W * H * 0.28:
            continue
        x, y, w, h = cv2.boundingRect(c)
        rectangularity = area / (w * h + 1e-6)
        ar = w / (h + 1e-6)
        if rectangularity > 0.60 and 0.22 < ar < 5.0:
            cv2.drawContours(rects, [cv2.convexHull(c)], -1, 255, -1)
    rects = cv2.bitwise_and(rects, band_u8)
    parts["rects"] = rects
    fg |= rects

    # 4. depth: pixels clearly nearer than the wall plane AND not SegFormer-wall
    if depth is not None:
        hint = cv2.bitwise_and((seg_map == 0).astype(np.uint8) * 255, band_u8)
        hint[fg > 0] = 0
        if np.count_nonzero(hint) > W * H * 0.01:
            wd = float(np.percentile(depth[hint > 0], 45))
            sd = float(np.std(depth[hint > 0])) + 1e-3
            nearer = (depth < (wd - max(0.10, sd * 1.8))).astype(np.uint8) * 255
            nearer = cv2.bitwise_and(nearer, band_u8)
            nearer[seg_map == 0] = 0                     # keep anything SegFormer calls wall
            nearer = cv2.morphologyEx(nearer, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
            nn, ln, sn, _ = cv2.connectedComponentsWithStats(nearer, 8)
            dmask = np.zeros((H, W), np.uint8)
            for i in range(1, nn):
                if sn[i, cv2.CC_STAT_AREA] > W * H * 0.002:
                    dmask[ln == i] = 255
            parts["depth_fg"] = dmask
            fg |= dmask

    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    parts["union"] = fg.copy()
    return fg, parts


def build_wall_layers(seg_map, floor_mask, W, H, room_bgr, depth=None, floor_quad=None):
    """
    Two layers instead of one perforated mask:

      • band_u8     — the wall SURFACE: one solid, continuous, hole-free region
                      between the ceiling and floor lines. This is where the new
                      material goes. No object cut-outs -> no swiss-cheese.

      • fg_alpha_u8 — the FOREGROUND: everything that sits in front of the wall
                      (furniture, art, TV, mirror, curtain, plant, lamp, window,
                      door, cabinet, stairs …) as a SOFT, feathered alpha. It is
                      composited back on TOP of the material, so the material can
                      never leak onto an object and the edges stay clean.

    Robust because over-including the foreground just shows a little more of the
    original photo; under-including just shows the wall behind an object edge.
    """
    ceil_y, floor_y = wall_band_bounds(seg_map, floor_mask, W, H, floor_quad)
    ys = np.arange(H, dtype=np.float32)[:, None]
    band = ((ys >= ceil_y[None, :]) & (ys < floor_y[None, :])).astype(np.uint8) * 255

    # ---- carve ONLY hard non-wall: floor, ceiling, full-height openings ---
    hard = np.zeros((H, W), np.uint8)
    hard[seg_map == 5] = 255
    hard[floor_mask > 0] = 255
    band_h = np.maximum(1.0, floor_y - ceil_y)
    opening = ((seg_map == 8) | (seg_map == 9) | (seg_map == 14)).astype(np.uint8) * 255
    if opening.any():
        op_h = (cv2.bitwise_and(opening, band) > 0).sum(axis=0).astype(np.float32)
        full_cols = np.where((op_h / band_h) > 0.72)[0]
        if len(full_cols):
            hard[:, full_cols] = 255
        hard |= cv2.dilate(opening, np.ones((5, 5), np.uint8))

    # Glazed wall / large window SegFormer missed: BRIGHT *and*, per depth,
    # clearly farther than the wall plane (a bright white wall is at wall depth).
    if depth is not None:
        gray0 = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)
        wall_hint0 = cv2.bitwise_and((seg_map == 0).astype(np.uint8) * 255, band)
        if np.count_nonzero(wall_hint0) > W * H * 0.01:
            wdz = float(np.percentile(depth[wall_hint0 > 0], 55))
            glaze = ((gray0 > 224) & (depth > wdz + 0.12)).astype(np.uint8) * 255
            glaze = cv2.bitwise_and(glaze, band)
            glaze = cv2.morphologyEx(glaze, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
            ng, lg, sg, _ = cv2.connectedComponentsWithStats(glaze, 8)
            for i in range(1, ng):
                if sg[i, cv2.CC_STAT_AREA] > W * H * 0.003:
                    hard[lg == i] = 255
    band[cv2.dilate(hard, np.ones((3, 3), np.uint8)) > 0] = 0

    # ---- make the surface SOLID, CONTINUOUS and EVEN --------------------
    band = cv2.morphologyEx(
        band, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(21, W // 22) | 1, max(11, H // 34) | 1)))
    # flood-fill every interior hole (objects get re-composited on top anyway)
    ff = band.copy()
    cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 255)
    band |= cv2.bitwise_not(ff)
    # keep the substantial wall component(s)
    nB, labB, stB, _ = cv2.connectedComponentsWithStats(band, 8)
    solid = np.zeros((H, W), np.uint8)
    for i in range(1, nB):
        if stB[i, cv2.CC_STAT_AREA] > W * H * 0.012:
            solid[labB == i] = 255
    band = solid if solid.any() else band
    band[hard > 0] = 0                                   # never tile a window / floor / ceiling
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))

    # ---- foreground preserve layer (soft, feathered) ------------------
    fg, parts = detect_wall_foreground(seg_map, room_bgr, band, depth, W, H)
    fg = cv2.bitwise_and(fg, band)
    guide = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    fg_soft = np.clip(_guided_filter(guide, fg.astype(np.float32) / 255.0,
                                     radius=max(4, W // 160), eps=1e-3), 0.0, 1.0)
    fg_in = cv2.erode(fg, np.ones((7, 7), np.uint8)).astype(np.float32) / 255.0
    fg_alpha = np.clip(np.maximum(fg_in, fg_soft), 0.0, 1.0)
    fg_alpha = cv2.GaussianBlur(fg_alpha, (0, 0), max(1.0, W * 0.003))
    fg_alpha = np.clip(fg_alpha, 0.0, 1.0)

    # ---- metrics -------------------------------------------------------
    denom = max(1, int(np.count_nonzero(band)))
    visible_material = int(np.count_nonzero((band > 0) & (fg_alpha < 0.5)))
    seg_wall_in = int(np.count_nonzero((seg_map == 0) & (band > 0)))
    # "solidity" = how hole-free the surface is (1.0 == no swiss-cheese)
    bcnt, blab, bst, _ = cv2.connectedComponentsWithStats(band, 8)
    solidity = 1.0
    if bcnt > 1:
        big = 1 + int(np.argmax(bst[1:, cv2.CC_STAT_AREA]))
        ys2, xs2 = np.where(blab == big)
        if len(xs2) > 10:
            hull = cv2.convexHull(np.column_stack([xs2, ys2]))
            hull_area = cv2.contourArea(hull)
            solidity = float(np.clip(bst[big, cv2.CC_STAT_AREA] / max(1.0, hull_area), 0.0, 1.0))
    meta = {
        "coverage": round(solidity, 3),
        "visible_material": round(visible_material / denom, 3),
        "wall_confidence": round(float(np.clip(seg_wall_in / denom, 0.0, 1.0)), 3),
        "object_confidence": round(float(np.clip(np.count_nonzero(parts["semantic"]) /
                                                 max(1, np.count_nonzero(parts["union"])), 0.0, 1.0)), 3),
        "depth_used": bool(depth is not None),
        "ceil_y": ceil_y, "floor_y": floor_y, "band": band,
        "parts": parts, "fg": fg, "fg_alpha": fg_alpha,
    }
    return band, (fg_alpha * 255).astype(np.uint8), meta


def detect_wall_planes(wall_alpha, seg_map, room_bgr, W, H, vp, ceil_y, floor_y, depth=None):
    """
    Split the wall alpha into ≤3 planar surfaces bounded by the detected
    ceiling / floor lines and vertical wall–wall seams. Each plane's quad is the
    true image of a rectangle on that wall, so a homography onto it foreshortens
    the texture correctly.
    """
    a = (wall_alpha > 127).astype(np.uint8) * 255
    ys, xs = np.where(a > 0)
    if len(xs) < W * H * 0.004:
        return []
    x_lo, x_hi = int(xs.min()), int(xs.max())

    seams = []
    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)
    band_gray = cv2.bitwise_and(gray, cv2.erode(a, np.ones((15, 15), np.uint8)))
    try:
        lsd = cv2.createLineSegmentDetector()
        det = lsd.detect(band_gray)[0]
    except Exception:
        det = None
    if det is not None:
        for l in det.reshape(-1, 4):
            x1, y1, x2, y2 = [float(v) for v in l]
            length = np.hypot(x2 - x1, y2 - y1)
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if length > H * 0.28 and ang > 62:
                seams.append(0.5 * (x1 + x2))

    if depth is not None:
        dband = np.where(a > 0, depth.astype(np.float32), np.nan)
        valid_cols = np.isfinite(dband).any(axis=0)
        col_d = np.full(W, 0.5, np.float32)
        if valid_cols.any():
            with np.errstate(all="ignore"):
                col_d[valid_cols] = np.nanmedian(dband[:, valid_cols], axis=0)
            col_d = np.nan_to_num(col_d, nan=float(np.median(col_d[valid_cols])))
        col_d = cv2.GaussianBlur(col_d.reshape(1, -1), (0, 0), max(1.0, W * 0.012)).reshape(-1)
        g = np.abs(np.gradient(col_d))
        thr = max(float(np.percentile(g, 95)), 0.004)
        seams += [float(x) for x in np.where(g > thr)[0]]

    seams = sorted(s for s in seams if x_lo + W * 0.09 < s < x_hi - W * 0.09)
    clustered = []
    for s in seams:
        if clustered and s - clustered[-1] < W * 0.07:
            clustered[-1] = 0.5 * (clustered[-1] + s)
        else:
            clustered.append(s)
    clustered = clustered[:2]
    xbreaks = [float(x_lo)] + clustered + [float(x_hi)]

    planes = []
    for i in range(len(xbreaks) - 1):
        xl, xr = int(round(xbreaks[i])), int(round(xbreaks[i + 1]))
        if xr - xl < W * 0.05:
            continue
        col_any = a[:, xl:xr].max(axis=0) > 0
        if not col_any.any():
            continue
        xl2 = xl + int(np.argmax(col_any))
        xr2 = xr - int(np.argmax(col_any[::-1]))
        cxl, cxr = np.clip(xl2, 0, W - 1), np.clip(xr2, 0, W - 1)
        tl = [xl2, float(ceil_y[cxl])]
        tr = [xr2, float(ceil_y[cxr])]
        br = [xr2, float(floor_y[cxr])]
        bl = [xl2, float(floor_y[cxl])]
        pm = np.zeros((H, W), np.uint8)
        cv2.fillPoly(pm, [np.array([tl, tr, br, bl], np.int32)], 255)
        pm = cv2.bitwise_and(pm, a)
        if int(np.count_nonzero(pm)) < W * H * 0.004:
            continue
        cx = 0.5 * (xl2 + xr2)
        ptype = "back" if 0.24 * W < cx < 0.76 * W else ("left" if cx <= 0.5 * W else "right")
        conf = float(np.clip(0.55 + 2.5 * (int(np.count_nonzero(pm)) / (W * H)), 0.3, 0.97))
        planes.append({
            "quad": [[int(round(p[0])), int(round(p[1]))] for p in (tl, tr, br, bl)],
            "bbox": [int(xl2), int(min(tl[1], tr[1])), int(xr2 - xl2),
                     int(max(bl[1], br[1]) - min(tl[1], tr[1]))],
            "plane": ptype, "confidence": round(conf, 2),
        })

    if not planes:
        cxl, cxr = np.clip(x_lo, 0, W - 1), np.clip(x_hi, 0, W - 1)
        planes = [{
            "quad": [[x_lo, int(ceil_y[cxl])], [x_hi, int(ceil_y[cxr])],
                     [x_hi, int(floor_y[cxr])], [x_lo, int(floor_y[cxl])]],
            "bbox": [x_lo, int(ceil_y[cxl]), x_hi - x_lo, int(floor_y[cxl] - ceil_y[cxl])],
            "plane": "back", "confidence": 0.5,
        }]
    return planes


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
# HELPER: draw a recessed grout seam (dark valley + thin highlight)
# ============================================================
def _bevel_seam(surface, x0, y0, x1, y1, gw, gcol):
    """Paint a grout line as a soft recessed groove rather than a flat stroke."""
    if gw <= 0:
        return
    h, w = surface.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    gcol = np.array(gcol, dtype=np.float32)
    seg = surface[y0:y1, x0:x1].astype(np.float32)
    # core grout colour
    seg[:] = gcol
    # darken the centre a touch for depth
    seg *= 0.9
    surface[y0:y1, x0:x1] = np.clip(seg, 0, 255).astype(np.uint8)


def _apply_grid_grout(surface, cell_w, cell_h, gw, gcol, x_off_fn=None):
    h, w = surface.shape[:2]
    y = 0
    row = 0
    while y < h:
        _bevel_seam(surface, 0, y, w, y + gw, gw, gcol)
        y += cell_h
        row += 1
    col = 0
    x = 0
    while x < w + cell_w:
        _bevel_seam(surface, x, 0, x + gw, h, gw, gcol)
        x += cell_w
        col += 1


# ============================================================
# HELPER: pattern renderers (all fronto-parallel, top edge = far)
# ============================================================
def _tile_variants(cell_bgr):
    return [
        cell_bgr,
        cv2.rotate(cell_bgr, cv2.ROTATE_180),
        cv2.flip(cell_bgr, 1),
        cv2.flip(cell_bgr, 0),
    ]


def _render_bond(tile_bgr, W, H, cell_w, cell_h, pattern, gw, gcol):
    """grid / brick (1/2) / brick_third (1/3) / vertical (1/2 column offset)."""
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


def _render_slab(tile_bgr, W, H):
    """Large-format book-matched slab. Tiles the mirrored block so we never
    upscale the source more than ~1.3x (keeps veining crisp, not 'pasted')."""
    top = np.hstack([tile_bgr, cv2.flip(tile_bgr, 1)])
    block = np.vstack([top, cv2.flip(top, 0)])          # book-matched 2x2
    bh, bw = block.shape[:2]
    nx = max(1, int(np.ceil(W / (bw * 1.3))))
    ny = max(1, int(np.ceil(H / (bh * 1.3))))
    if nx > 1 or ny > 1:
        block = np.tile(block, (ny, nx, 1))
    interp = cv2.INTER_AREA if (block.shape[1] > W or block.shape[0] > H) else cv2.INTER_CUBIC
    return cv2.resize(block, (W, H), interpolation=interp)


def _render_basketweave(tile_bgr, W, H, unit, gw, gcol):
    """Pairs of planks alternating horizontal/vertical in a checkerboard."""
    surface = np.zeros((H, W, 3), dtype=np.uint8)
    u = max(8, int(unit))            # single plank short side
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


def _render_herringbone(tile_bgr, W, H, unit, gw, gcol, k=2):
    """
    True interlocking herringbone via per-pixel classification: square sub-cells
    of side ``b``; every plank is ``k`` sub-cells long, alternating H/V in a
    staircase governed by ``(i + j) mod 2k``.
    """
    b = max(6, int(unit))
    Hs = cv2.resize(tile_bgr, (k * b, b), interpolation=cv2.INTER_AREA)            # (b, k*b)
    Vs = cv2.rotate(cv2.resize(tile_bgr, (k * b, b), interpolation=cv2.INTER_AREA),
                    cv2.ROTATE_90_CLOCKWISE)                                        # (k*b, b)

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
        # unique id per plank so grout only lands on real plank boundaries
        pid = np.where(horiz, (plank_i0 * 8192 + j) * 2, (i * 8192 + plank_j0) * 2 + 1)
        seam = np.zeros((H, W), np.uint8)
        seam[:, 1:] |= (pid[:, 1:] != pid[:, :-1]).astype(np.uint8)
        seam[1:, :] |= (pid[1:, :] != pid[:-1, :]).astype(np.uint8)
        if gw > 1:
            seam = cv2.dilate(seam, np.ones((gw, gw), np.uint8))
        surface[seam > 0] = np.clip(np.array(gcol, np.float32) * 0.72, 0, 255).astype(np.uint8)
    return surface


def _paste(surface, sprite, x, y):
    H, W = surface.shape[:2]
    sh, sw = sprite.shape[:2]
    x0, y0 = int(x), int(y)
    xd0, yd0 = max(0, x0), max(0, y0)
    xd1, yd1 = min(W, x0 + sw), min(H, y0 + sh)
    if xd1 <= xd0 or yd1 <= yd0:
        return
    surface[yd0:yd1, xd0:xd1] = sprite[yd0 - y0:yd1 - y0, xd0 - x0:xd1 - x0]


def _render_diagonal(tile_bgr, W, H, cell_w, cell_h, base_pattern, gw, gcol):
    """Grid / brick rotated 45° (diamond set)."""
    d = int(np.hypot(W, H)) + 2 * max(cell_w, cell_h) + 4
    base = _render_bond(tile_bgr, d, d, cell_w, cell_h, base_pattern, gw, gcol)
    M = cv2.getRotationMatrix2D((d / 2.0, d / 2.0), 45, 1.0)
    rot = cv2.warpAffine(base, M, (d, d), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    y0, x0 = (d - H) // 2, (d - W) // 2
    return rot[y0:y0 + H, x0:x0 + W].copy()


def _render_chevron(tile_bgr, W, H, unit, gw, gcol, ratio=5):
    """Planks sheared ±45° in alternating vertical bands, meeting in a V."""
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
        sheared = cv2.warpAffine(strip, Msh, (l, strip_h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        seg = sheared[l:l + H]
        _paste(surface, seg, x0, 0)
        if gw > 0:                       # seam at band join
            _bevel_seam(surface, max(0, x0), 0, max(0, x0) + gw, H, gw, gcol)
    return surface


def _render_windmill(tile_bgr, W, H, unit, gw, gcol):
    """Pinwheel: four planks rotating around a centre square."""
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


# ============================================================
# HELPER: Build perspective-correct tiled surface (flat)
# ============================================================
def build_tile_surface(
    tile_np: np.ndarray,
    plane_w: int,
    plane_h: int,
    tiles_x: float = 6,
    tiles_y: float = 6,
    pattern: str = "grid",  # grid|brick|brick_third|vertical|basketweave|herringbone|slab
    grout_width: int = 0,
    grout_color: tuple = (210, 210, 210),
    rotation_deg: float = 0,
    brightness: float = 1.0,
    far_fade: float = 0.0,     # 0..0.4 atmospheric darkening toward the far (top) edge
    far_blur: bool = False,    # progressive blur toward the far edge (anti-moire)
    slab: bool = False,        # force continuous slab regardless of `pattern`
) -> np.ndarray:
    """
    Render a fronto-parallel tiled plane. ``tiles_x`` / ``tiles_y`` are the real
    tile counts that span the plane (may be fractional) — the caller derives
    them from metric room size so that a single homography warp of this surface
    produces physically-plausible perspective and tile scale.

    The plane's TOP edge is the far edge (back of the room); ``far_fade`` and
    ``far_blur`` add depth cues there before the warp.
    """
    tiles_x = max(1.0, float(tiles_x))
    tiles_y = max(1.0, float(tiles_y))
    cell_w = max(4, int(round(plane_w / tiles_x)))
    cell_h = max(4, int(round(plane_h / tiles_y)))

    tile_bgr = cv2.cvtColor(tile_np, cv2.COLOR_RGB2BGR)
    if rotation_deg:
        cx, cy = tile_bgr.shape[1] // 2, tile_bgr.shape[0] // 2
        M = cv2.getRotationMatrix2D((cx, cy), rotation_deg, 1.0)
        tile_bgr = cv2.warpAffine(tile_bgr, M, (tile_bgr.shape[1], tile_bgr.shape[0]),
                                  borderMode=cv2.BORDER_REFLECT)

    gw = int(max(0, grout_width))
    gcol = tuple(int(c) for c in grout_color)

    # herringbone / basketweave use small planks — subdivide the metric cell
    small_unit = max(6, int(min(cell_w, cell_h) / 2.4))

    if slab or pattern == "slab":
        surface = _render_slab(tile_bgr, plane_w, plane_h)
    elif pattern == "herringbone":
        # plank patterns need a visible seam to read as a pattern
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

    # --- progressive blur confined to the extreme far edge (anti-moire) -----
    if far_blur and plane_h > 40:
        blurred = cv2.GaussianBlur(surface, (0, 0), sigmaX=max(1.0, plane_w * 0.0011))
        wv = np.linspace(1.0, 0.0, plane_h, dtype=np.float32) ** 2.6
        surface = np.clip(surface * (1.0 - wv[:, None, None]) + blurred * wv[:, None, None],
                          0, 255).astype(np.uint8)

    # --- atmospheric fade toward the far edge -------------------------------
    if far_fade > 0.0:
        g = np.linspace(1.0 - float(np.clip(far_fade, 0, 0.45)), 1.0,
                        plane_h, dtype=np.float32)
        surface = np.clip(surface.astype(np.float32) * g[:, None, None], 0, 255).astype(np.uint8)

    return surface


# ============================================================
# HELPER: Metric tiling plan for a ground-plane quad
# ============================================================
def plan_floor_tiling(quad, W, H, ppm, tile_wmm, tile_hmm, size_mult=1.0):
    """
    Turn the floor quad + real-world scale into (tiles_x, tiles_y, plane_w,
    plane_h) so ``build_tile_surface`` lays tiles at a believable physical size
    and the homography warp gives true perspective foreshortening.
    """
    q = np.asarray(quad, dtype=np.float64)
    far_w = float(np.hypot(*(q[1] - q[0])))
    near_w = float(np.hypot(*(q[2] - q[3])))
    near_w = float(np.clip(near_w, W * 0.6, W * 1.35))

    # Real-world scale sanity band — `derive_pixel_scale` can be far off when a
    # door/window is mis-detected, so only trust ppm inside a plausible range.
    ppm = float(ppm) if ppm else 0.0
    if W / 9.0 <= ppm <= W / 2.5:
        room_w_m = float(np.clip(near_w / ppm, 2.6, 9.0))
    else:
        room_w_m = 6.0

    # Depth from the near/far foreshortening ratio (projective proxy).
    fk = float(np.clip(near_w / max(far_w, 1.0), 1.05, 6.0))
    room_d_m = float(np.clip(room_w_m * (0.32 * fk), 2.4, 12.0))

    tw = max(0.05, (float(tile_wmm) / 1000.0) * size_mult)
    th = max(0.05, (float(tile_hmm) / 1000.0) * size_mult)

    tiles_x = max(2.0, room_w_m / tw)
    tiles_y = max(2.0, room_d_m / th)

    # Cap density but keep the tile aspect ratio intact (planks stay long).
    if tiles_x > 22.0:
        s = 22.0 / tiles_x
        tiles_x, tiles_y = 22.0, max(2.0, tiles_y * s)
    if tiles_y > 26.0:
        s = 26.0 / tiles_y
        tiles_y, tiles_x = 26.0, max(2.0, tiles_x * s)

    plane_w = 2048
    plane_h = int(np.clip(round(plane_w * room_d_m / room_w_m), 512, 4096))
    return tiles_x, tiles_y, plane_w, plane_h


# ============================================================
# HELPER: Apply photorealistic homography warp
# ============================================================
def apply_homography_warp(
    room_bgr: np.ndarray,
    tile_surface: np.ndarray,
    dest_quad: list,
    mask_u8: np.ndarray,
    shadow_strength: float = 0.55,
    finish: str = "satin",
    feather_px: int = 3,
) -> np.ndarray:
    H, W = room_bgr.shape[:2]
    ph, pw = tile_surface.shape[:2]

    if mask_u8.shape[0] != H or mask_u8.shape[1] != W:
        mask_u8 = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)

    src_pts = np.array([
        [0,      0],
        [pw - 1, 0],
        [pw - 1, ph - 1],
        [0,      ph - 1],
    ], dtype=np.float32)

    dst_pts = np.array(dest_quad, dtype=np.float32)

    # Pre-shrink the flat surface with area-averaging when it is much bigger
    # than the region it will occupy — warpPerspective's own interpolation
    # aliases badly on a large downscale (the "blurry / pasted" look).
    dq = np.array(dest_quad, dtype=np.float64)
    dst_w = max(1.0, np.hypot(*(dq[1] - dq[0])), np.hypot(*(dq[2] - dq[3])))
    dst_h = max(1.0, np.hypot(*(dq[3] - dq[0])), np.hypot(*(dq[2] - dq[1])))
    sx = min(1.0, (dst_w * 1.6) / pw)
    sy = min(1.0, (dst_h * 1.6) / ph)
    if sx < 0.85 or sy < 0.85:
        new_w, new_h = max(8, int(pw * sx)), max(8, int(ph * sy))
        tile_surface = cv2.resize(tile_surface, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ph, pw = tile_surface.shape[:2]
        src_pts = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(tile_surface, M, (W, H),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)

    wsharp = cv2.GaussianBlur(warped, (0, 0), 1.0)
    warped = cv2.addWeighted(warped, 1.5, wsharp, -0.5, 0)

    # ── 1. DUAL-RADIUS LIGHT MAP (7% broad blur + 2% sharp local blur) ────
    scale_f = min(1.0, 640.0 / max(W, H))
    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if scale_f < 1.0:
        low_w, low_h = max(1, int(W * scale_f)), max(1, int(H * scale_f))
        gray_low = cv2.resize(gray, (low_w, low_h), interpolation=cv2.INTER_AREA)
        k_broad = max(15, (int(max(low_w, low_h) * 0.07) | 1))
        k_sharp = max(5, (int(max(low_w, low_h) * 0.02) | 1))
        broad_blur = cv2.GaussianBlur(gray_low, (k_broad, k_broad), 0)
        sharp_blur = cv2.GaussianBlur(gray_low, (k_sharp, k_sharp), 0)
        light_low = 0.70 * broad_blur + 0.30 * sharp_blur
        light_map = cv2.resize(light_low, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        k_broad = max(51, (int(max(W, H) * 0.07) | 1))
        k_sharp = max(15, (int(max(W, H) * 0.02) | 1))
        broad_blur = cv2.GaussianBlur(gray, (k_broad, k_broad), 0)
        sharp_blur = cv2.GaussianBlur(gray, (k_sharp, k_sharp), 0)
        light_map = 0.70 * broad_blur + 0.30 * sharp_blur

    masked_vals = light_map[mask_u8 > 0]
    if len(masked_vals) > 0:
        wp = float(np.percentile(masked_vals, 95))
        wp = max(wp, 1.0)
        light_norm = np.clip(light_map / wp, 0.0, 1.0)
    else:
        light_norm = np.clip(light_map / 255.0, 0.0, 1.0)

    light_norm = np.clip(0.5 + (light_norm - 0.5) * 1.35, 0.0, 1.0)

    # ── 2. AMBIENT OCCLUSION (15% contact shadow at mask boundary) ────────
    if scale_f < 1.0:
        low_w, low_h = max(1, int(W * scale_f)), max(1, int(H * scale_f))
        mask_low = cv2.resize(mask_u8, (low_w, low_h), interpolation=cv2.INTER_NEAREST)
        dist_transform_low = cv2.distanceTransform(mask_low, cv2.DIST_L2, 5)
        max_dist_low = dist_transform_low.max()
        if max_dist_low > 0:
            ao_low = np.clip(dist_transform_low / (max_dist_low * 0.15), 0.0, 1.0).astype(np.float32)
        else:
            ao_low = np.ones_like(gray_low, dtype=np.float32)
        ao_map = cv2.resize(ao_low, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        dist_transform = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
        max_dist = dist_transform.max()
        if max_dist > 0:
            ao_map = np.clip(dist_transform / (max_dist * 0.15), 0.0, 1.0).astype(np.float32)
        else:
            ao_map = np.ones_like(gray, dtype=np.float32)
    ao_map = cv2.GaussianBlur(ao_map, (0, 0), max(3.0, W * 0.008))

    # ── 3. SHADOW BLEND ──────────────────────────────────────
    shadow_blend = light_norm * shadow_strength + (1.0 - shadow_strength)

    # ── 4. COLOR TEMPERATURE HARMONIZATION (30% Lab a/b shift) ────────
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

        da = (room_a_mean - tile_a_mean) * 0.30
        db = (room_b_mean - tile_b_mean) * 0.30
        warped_lab[:, :, 1] = np.clip(warped_lab[:, :, 1] + da, 0, 255)
        warped_lab[:, :, 2] = np.clip(warped_lab[:, :, 2] + db, 0, 255)
        warped = cv2.cvtColor(warped_lab.astype(np.uint8), cv2.COLOR_Lab2BGR)

    # ── 5. APPLY LIGHTING & CONTACT OCCLUSION ────────────────
    warped_f = warped.astype(np.float32)
    lm3 = np.stack([shadow_blend] * 3, axis=-1)
    lit_warped = warped_f * lm3

    # Ground furniture legs with 15% contact shadow
    ao3 = np.stack([ao_map] * 3, axis=-1)
    lit_warped = lit_warped * (0.85 + 0.15 * ao3)

    mask_binary_f = (mask_u8 > 0).astype(np.float32)
    mb3 = np.stack([mask_binary_f] * 3, axis=-1)

    # ── 5b. DETAIL TRANSFER — borrow original photo micro-texture / grain ──
    orig_gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail = orig_gray - cv2.GaussianBlur(orig_gray, (0, 0), 1.1)
    detail = np.clip(detail, -14, 14)[..., None]
    lit_warped = np.clip(lit_warped + detail * 0.5 * mb3, 0, 255)

    # ── 6. REALISTIC CLEAN FINISH RESPONSE ───────────────────
    glare = np.clip((orig_gray - 210.0) / 45.0, 0.0, 1.0)
    glare = cv2.GaussianBlur(glare, (0, 0), max(2.0, W * 0.004))[..., None] * mb3
    glare_amt = 0.38 if finish == "glossy" else (0.24 if finish == "satin" else 0.12)
    lit_warped = 255.0 - (255.0 - lit_warped) * (1.0 - glare * glare_amt)

    if finish in ("glossy", "satin"):
        refl_a = 0.18 if finish == "glossy" else 0.09
        ys = np.arange(H, dtype=np.int32)[:, None]
        xs = np.arange(W, dtype=np.int32)[None, :]
        col_any = mask_u8.max(axis=0) > 0
        edge = (mask_u8 > 0).argmax(axis=0).astype(np.int32)
        edge[~col_any] = H
        src_y = np.clip(2 * edge[None, :] - ys, 0, H - 1)
        reflection = room_bgr[src_y, np.broadcast_to(xs, (H, W))].astype(np.float32)
        reflection = cv2.GaussianBlur(reflection, (0, 0), max(3.0, W * 0.012))
        depth = np.clip((ys - edge[None, :]) / (H * 0.28), 0.0, 1.0)
        refl_w = ((1.0 - depth) ** 1.8)[..., None] * mb3 * refl_a
        lit_warped = lit_warped * (1.0 - refl_w) + reflection * refl_w

    if finish == "glossy":
        highlight = np.clip((light_norm - 0.78) / 0.22, 0.0, 1.0)
        hl3 = np.stack([highlight] * 3, axis=-1) * mb3
        lit_warped = np.clip(lit_warped + hl3 * 30, 0, 255)
    elif finish == "satin":
        highlight = np.clip((light_norm - 0.85) / 0.15, 0.0, 1.0)
        hl3 = np.stack([highlight] * 3, axis=-1) * mb3
        lit_warped = np.clip(lit_warped + hl3 * 15, 0, 255)

    lit_warped = np.clip(lit_warped, 0, 255)

    # ── 7. TWO-LAYER ANTI-ALIASED MASK (feather_px = 5) ────────
    feather_px = max(1, feather_px)
    erode_k = np.ones((feather_px * 2 + 1, feather_px * 2 + 1), np.uint8)
    mask_eroded = cv2.erode(mask_u8, erode_k, iterations=1)
    mask_feathered = cv2.GaussianBlur(
        mask_u8.astype(np.float32) / 255.0,
        (feather_px * 2 + 1, feather_px * 2 + 1),
        float(feather_px) * 0.4
    )
    mask_f32 = np.maximum(mask_feathered, mask_eroded.astype(np.float32) / 255.0)

    # ── 8. PURE COMPOSITE ────────────────────────────────────
    m3 = np.stack([mask_f32] * 3, axis=-1)
    result = room_bgr.astype(np.float32)
    composited = result * (1.0 - m3) + lit_warped * m3
    return np.clip(composited, 0, 255).astype(np.uint8)


# ============================================================
# HELPER: Apply photorealistic wall homography warp (vertical surface model)
# ============================================================
def apply_wall_photorealism_warp(
    room_bgr: np.ndarray,
    tile_surface: np.ndarray,
    dest_quad: list,
    mask_u8: np.ndarray,
    shadow_strength: float = 0.55,
    finish: str = "satin",
    feather_px: int = 5,
    fg_alpha: np.ndarray = None,
    material_kind: str = "tile",
) -> np.ndarray:
    """
    PHASE F — linear-light illumination-decomposition compositing.

        final = material  x  recovered_shading(original photo)

    instead of a flat colour overlay. The wall's own shading (window-light
    gradient, corner falloff, cast shadows from furniture/plants) is recovered
    from the ORIGINAL photo and used to modulate the new material in LINEAR
    light, so a patch that was dark because of a shadow stays dark on the new
    material too — the lighting is preserved, only the surface changes.
    Foreground objects are then composited back on top (soft alpha), and the
    whole thing is feathered into the room at the wall-band edge.
    """
    H, W = room_bgr.shape[:2]
    ph, pw = tile_surface.shape[:2]

    if mask_u8.shape[0] != H or mask_u8.shape[1] != W:
        mask_u8 = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)
    if not np.any(mask_u8 > 0):
        return room_bgr

    # ── 1. Perspective warp with anti-aliased pre-shrink (unchanged — good) ──
    src_pts = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], dtype=np.float32)
    dst_pts = np.array(dest_quad, dtype=np.float32)

    dq = np.array(dest_quad, dtype=np.float64)
    dst_w = max(1.0, np.hypot(*(dq[1] - dq[0])), np.hypot(*(dq[2] - dq[3])))
    dst_h = max(1.0, np.hypot(*(dq[3] - dq[0])), np.hypot(*(dq[2] - dq[1])))
    sx = min(1.0, (dst_w * 1.6) / pw)
    sy = min(1.0, (dst_h * 1.6) / ph)
    if sx < 0.85 or sy < 0.85:
        new_w, new_h = max(8, int(pw * sx)), max(8, int(ph * sy))
        tile_surface = cv2.resize(tile_surface, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ph, pw = tile_surface.shape[:2]
        src_pts = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(tile_surface, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    wsharp = cv2.GaussianBlur(warped, (0, 0), 1.0)
    warped = cv2.addWeighted(warped, 1.5, wsharp, -0.5, 0)

    # ── 2. Recover the ORIGINAL wall's shading layer (STEP 8/9/15/16) ──────
    mask01 = mask_u8.astype(np.float32) / 255.0
    shading, light_cast = recover_wall_shading(room_bgr, mask01, W, H)

    # material-aware shading fidelity: flat paint/large slabs should track the
    # room's light gradient closely; busy multi-tile patterns get it slightly
    # damped so grout/pattern contrast doesn't get crushed by strong shadow
    fidelity = {"paint": 1.15, "slab": 1.05, "tile": 0.90, "textured": 0.80}.get(material_kind, 0.90)
    shading_eff = 1.0 + (shading - 1.0) * float(np.clip(fidelity, 0.5, 1.3))

    # contact AO — a touch darker right at furniture / ceiling / floor joins
    dist_transform = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    max_dist = dist_transform.max()
    ao_map = (np.clip(dist_transform / (max_dist * 0.16), 0.0, 1.0).astype(np.float32)
             if max_dist > 0 else np.ones((H, W), np.float32))
    ao_map = cv2.GaussianBlur(ao_map, (0, 0), max(3.0, W * 0.008))
    ao_gain = 0.90 + 0.10 * ao_map

    gain = np.clip(shading_eff * ao_gain, 0.25, 2.3)
    gain = gain * shadow_strength + (1.0 - shadow_strength)     # user's "ambient light blend" slider

    # ── 3. LINEAR-LIGHT compositing: material x shading, in linear space ───
    material_lin = _srgb_to_linear(warped.astype(np.float32)[..., ::-1] / 255.0)   # BGR -> RGB linear
    lit_lin = material_lin * gain[..., None] * light_cast[None, None, :]

    # finish-dependent specular response (physically it belongs post-tonemap,
    # kept simple: a soft additive highlight where the recovered shading is bright)
    if finish in ("glossy", "satin"):
        spec_amt = 0.16 if finish == "glossy" else 0.07
        spec = np.clip((shading - 1.06) / 0.35, 0.0, 1.0) ** 1.3
        lit_lin = lit_lin + spec[..., None] * spec_amt

    lit_lin = np.clip(lit_lin, 0.0, 1.0)
    lit_warped = _linear_to_srgb(lit_lin)[..., ::-1] * 255.0                       # RGB -> BGR sRGB

    # ── 4. Micro-texture / grain transfer (small, keeps it from looking CG-flat) ──
    mask_binary_f = (mask_u8 > 0).astype(np.float32)
    mb3 = np.stack([mask_binary_f] * 3, axis=-1)
    orig_gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail = np.clip(orig_gray - cv2.GaussianBlur(orig_gray, (0, 0), 1.1), -12, 12)[..., None]
    lit_warped = np.clip(lit_warped + detail * 0.35 * mb3, 0, 255)

    # ── 5. FOREGROUND layer — objects preserved on TOP (STEP 3/4/9) ────────
    if fg_alpha is not None:
        if fg_alpha.shape[:2] != (H, W):
            fg_alpha = cv2.resize(fg_alpha, (W, H), interpolation=cv2.INTER_LINEAR)
        fga = np.clip(fg_alpha.astype(np.float32) / 255.0, 0.0, 1.0)[..., None]
        lit_warped = lit_warped * (1.0 - fga) + room_bgr.astype(np.float32) * fga

    # ── 6. Feathered band-edge composite into the room (STEP 11) ───────────
    feather_px = max(1, feather_px)
    erode_k = np.ones((feather_px * 2 + 1, feather_px * 2 + 1), np.uint8)
    mask_eroded = cv2.erode(mask_u8, erode_k, iterations=1)
    mask_feathered = cv2.GaussianBlur(mask_u8.astype(np.float32) / 255.0, (feather_px * 2 + 1, feather_px * 2 + 1), float(feather_px) * 0.4)
    mask_f32 = np.maximum(mask_feathered, mask_eroded.astype(np.float32) / 255.0)

    m3 = np.stack([mask_f32] * 3, axis=-1)
    result = room_bgr.astype(np.float32)
    composited = result * (1.0 - m3) + lit_warped * m3

    return np.clip(composited, 0, 255).astype(np.uint8)



# ============================================================
# HELPER: Approximate real room L x B (feet) from the floor quad
# ============================================================
def estimate_room_dims_ft(floor_quad, W, H, ppm):
    q = np.asarray(floor_quad, dtype=np.float64)
    near_w = float(np.hypot(*(q[2] - q[3])))
    far_w = float(np.hypot(*(q[1] - q[0])))
    ppm = float(ppm) if ppm else 0.0
    if W / 9.0 <= ppm <= W / 2.5:
        width_m = float(np.clip(min(near_w, W * 1.3) / ppm, 2.6, 9.0))
    else:
        width_m = 6.0
    fk = float(np.clip(near_w / max(far_w, 1.0), 1.05, 6.0))
    depth_m = float(np.clip(width_m * (0.32 * fk), 2.4, 12.0))
    return [round(depth_m * 3.28084, 1), round(width_m * 3.28084, 1)]   # [length, breadth]


# ============================================================
#  WALL DEBUG / PREVIEW  (Phase A visual inspection)
# ============================================================
def _srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1 / 2.4)) - 0.055)


def recover_wall_shading(room_bgr, alpha01, W, H):
    """
    Approximate the wall's shading layer (broad light gradient + local cast
    shadows + AO) with the wall's own colour/texture removed. 1.0 = neutrally
    lit; <1 in shadow; >1 in a highlight. Also returns the light colour cast.
    """
    lin = _srgb_to_linear(room_bgr.astype(np.float32)[..., ::-1] / 255.0)      # -> RGB linear
    lum = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    m = alpha01 > 0.5
    fill_val = float(np.median(lum[m])) if m.sum() > 100 else float(np.median(lum))
    filled = np.where(m, lum, fill_val).astype(np.float32)
    base = cv2.GaussianBlur(filled, (0, 0), max(3.0, W * 0.045))
    ref = float(np.median(base[m])) if m.sum() > 100 else float(np.median(base))
    shading = np.clip(base / max(ref, 1e-4), 0.35, 2.2).astype(np.float32)
    cast = np.ones(3, np.float32)
    if m.sum() > 100:
        wm = lin[m].reshape(-1, 3).mean(0)
        cast = (wm / max(float(wm.mean()), 1e-4)).astype(np.float32)
        cast = np.clip(cast, 0.85, 1.15)
    return shading, cast


def _wall_plane_texture(tile_bgr, quad, W, H, ppm, tile_wmm, tile_hmm, pattern, gw, gcol,
                        rotation, brightness, size_mult, slab):
    """Metric texture for one wall plane, warped to its quad (no lighting)."""
    q = np.asarray(quad, np.float64)
    plane_w_px = float(max(np.hypot(*(q[1] - q[0])), np.hypot(*(q[2] - q[3]))))
    plane_h_px = float(max(np.hypot(*(q[3] - q[0])), np.hypot(*(q[2] - q[1]))))
    ppm_eff = ppm if ppm and ppm > 1 else H * 0.42
    wall_w_m = float(np.clip(plane_w_px / ppm_eff, 0.8, 20.0))
    wall_h_m = float(np.clip(plane_h_px / ppm_eff, 1.6, 5.0))
    tw = max(0.03, (tile_wmm / 1000.0) * size_mult)
    th = max(0.03, (tile_hmm / 1000.0) * size_mult)
    tiles_x = float(np.clip(wall_w_m / tw, 1.0, 90.0))
    tiles_y = float(np.clip(wall_h_m / th, 1.0, 60.0))
    surf_w = 1600
    surf_h = int(np.clip(round(surf_w * plane_h_px / max(1.0, plane_w_px)), 400, 4096))
    surface = build_tile_surface(tile_bgr, surf_w, surf_h, tiles_x, tiles_y,
                                 pattern=pattern, grout_width=gw, grout_color=gcol,
                                 rotation_deg=rotation, brightness=brightness, slab=slab)
    ph, pw = surface.shape[:2]
    src = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], np.float32)
    sx = min(1.0, (plane_w_px * 1.7) / pw)
    sy = min(1.0, (plane_h_px * 1.7) / ph)
    if sx < 0.9 or sy < 0.9:
        surface = cv2.resize(surface, (max(8, int(pw * sx)), max(8, int(ph * sy))), interpolation=cv2.INTER_AREA)
        ph, pw = surface.shape[:2]
        src = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], np.float32)
    Mh = cv2.getPerspectiveTransform(src, np.array(quad, np.float32))
    return cv2.warpPerspective(surface, Mh, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _flat_wall_surface(tile_bgr, quad, W, H, ppm, tile_wmm, tile_hmm, pattern, slab, size_mult=1.0):
    """A FRONTO-PARALLEL metric tile surface sized for one wall plane's quad."""
    q = np.asarray(quad, np.float64)
    plane_w_px = float(max(np.hypot(*(q[1] - q[0])), np.hypot(*(q[2] - q[3]))))
    plane_h_px = float(max(np.hypot(*(q[3] - q[0])), np.hypot(*(q[2] - q[1]))))
    ppm_eff = ppm if ppm and ppm > 1 else H * 0.42
    wall_w_m = float(np.clip(plane_w_px / ppm_eff, 0.8, 20.0))
    wall_h_m = float(np.clip(plane_h_px / ppm_eff, 1.6, 5.0))
    tw = max(0.03, (tile_wmm / 1000.0) * size_mult)
    th = max(0.03, (tile_hmm / 1000.0) * size_mult)
    tiles_x = float(np.clip(wall_w_m / tw, 1.0, 90.0))
    tiles_y = float(np.clip(wall_h_m / th, 1.0, 60.0))
    sw = 1800
    sh = int(np.clip(round(sw * plane_h_px / max(1.0, plane_w_px)), 400, 4096))
    return build_tile_surface(tile_bgr, sw, sh, tiles_x, tiles_y, pattern=pattern,
                              grout_width=0, grout_color=(210, 210, 210), slab=slab)


def wall_preview_composite(room_bgr, planes, band_u8, fg_u8, tile_bgr, W, H, ppm,
                           tile_wmm=300.0, tile_hmm=600.0, pattern="grid",
                           finish="matte", size_mult=1.0):
    """
    Preview that runs the SAME production compositor (`apply_wall_photorealism_warp`)
    per plane, so the debug panel matches what /api/visualize produces.
    """
    slab = (tile_wmm >= 900 or tile_hmm >= 1100)
    m_kind = "slab" if slab else ("paint" if pattern == "grid" else
             ("textured" if pattern in ("herringbone", "chevron", "basketweave",
                                        "brick", "brick_third", "windmill") else "tile"))
    tex = np.zeros((H, W, 3), np.uint8)
    comp = room_bgr.copy()
    for pl in planes:
        pm = np.zeros((H, W), np.uint8)
        cv2.fillPoly(pm, [np.array(pl["quad"], np.int32)], 255)
        seg_mask = cv2.bitwise_and(pm, band_u8)
        if not np.any(seg_mask > 0):
            continue
        flat = _flat_wall_surface(tile_bgr, pl["quad"], W, H, ppm, tile_wmm, tile_hmm,
                                  pattern, slab, size_mult)
        tvis = _wall_plane_texture(tile_bgr, pl["quad"], W, H, ppm, tile_wmm, tile_hmm,
                                   pattern, 0, (210, 210, 210), 0.0, 1.0, size_mult, slab=slab)
        tex[seg_mask > 0] = tvis[seg_mask > 0]
        seg_fg = None
        if fg_u8 is not None:
            seg_fg = fg_u8.copy()
            seg_fg[seg_mask == 0] = 0
        comp = apply_wall_photorealism_warp(comp, flat, pl["quad"], seg_mask,
                                            shadow_strength=0.55, finish=finish,
                                            fg_alpha=seg_fg, material_kind=m_kind)

    shading, _ = recover_wall_shading(room_bgr, band_u8.astype(np.float32) / 255.0, W, H)
    return np.clip(comp, 0, 255).astype(np.uint8), tex, shading


def _label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 20), (24, 24, 24), -1)
    cv2.putText(img, text, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def build_wall_debug_panel(room_bgr, seg_map, floor_mask, W, H, depth, floor_quad,
                           tile_bgr=None, tile_wmm=300.0, tile_hmm=600.0,
                           pattern="grid", finish="matte"):
    """10-image inspection panel + metrics (STEP 21)."""
    def gray3(m):
        return cv2.cvtColor(m if m.dtype == np.uint8 else m.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    band_u8, fg_u8, meta = build_wall_layers(seg_map, floor_mask, W, H, room_bgr, depth, floor_quad)
    planes = detect_wall_planes(band_u8, seg_map, room_bgr, W, H, None,
                                meta["ceil_y"], meta["floor_y"], depth)

    imgs, names = [], []
    imgs.append(room_bgr);                                     names.append("01 original")
    imgs.append(gray3((seg_map == 0).astype(np.uint8) * 255)); names.append("02 raw wall seg")

    objv = room_bgr.copy() // 2
    pal = {"semantic": (0, 165, 255), "bright": (255, 255, 0), "rects": (255, 0, 255), "depth_fg": (0, 0, 255)}
    for k, col in pal.items():
        if k in meta["parts"]:
            objv[meta["parts"][k] > 0] = col
    imgs.append(objv);                                         names.append("03 foreground cues")

    ov = room_bgr.copy()
    ov[band_u8 > 127] = (255, 90, 0)                               # solid wall surface -> blue
    imgs.append(cv2.addWeighted(room_bgr, 0.42, ov, 0.58, 0))
    names.append(f'04 wall surface  cov={meta["coverage"]} visible={meta.get("visible_material")}')

    fgv = room_bgr.copy().astype(np.float32)
    fg3 = (fg_u8.astype(np.float32) / 255.0)[..., None]
    fgv = fgv * (0.35 + 0.65 * fg3)
    imgs.append(fgv.astype(np.uint8));                         names.append("05 foreground alpha (kept on top)")

    pv = room_bgr.copy()
    pcols = [(0, 200, 255), (0, 255, 120), (255, 120, 0)]
    for i, pl in enumerate(planes):
        cv2.polylines(pv, [np.array(pl["quad"], np.int32)], True, pcols[i % 3], 2)
        cv2.putText(pv, f'{pl["plane"]} {pl["confidence"]}', tuple(pl["quad"][0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, pcols[i % 3], 2)
    imgs.append(pv);                                           names.append(f"06 wall planes (n={len(planes)})")

    if tile_bgr is not None:
        comp, tex, shading = wall_preview_composite(room_bgr, planes, band_u8, fg_u8, tile_bgr,
                                                    W, H, None, tile_wmm, tile_hmm, pattern, finish)
        imgs.append(tex);                                      names.append("07 texture projection")
        sh = np.clip((shading - 0.5) / 1.4, 0, 1) * 255
        imgs.append(gray3(sh.astype(np.uint8)));               names.append("08 lighting / shading map")
        b3 = (band_u8.astype(np.float32) / 255.0)[..., None]
        litwall = (comp.astype(np.float32) * b3 + 30 * (1 - b3)).astype(np.uint8)
        imgs.append(litwall);                                  names.append("09 final wall (material x light)")
        imgs.append(comp);                                     names.append("10 final composite")
    else:
        for nm in ("07 texture", "08 shading", "09 final wall", "10 composite"):
            imgs.append(np.full((H, W, 3), 40, np.uint8));      names.append(nm + " (no tile)")

    cell_h = 260
    cells = []
    for im, nm in zip(imgs, names):
        r = cell_h / im.shape[0]
        cell = cv2.resize(im, (int(im.shape[1] * r), cell_h))
        cells.append(_label(cell, nm))
    cw = max(c.shape[1] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, 0, 0, cw - c.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20)) for c in cells]
    rows = [np.hstack(cells[i:i + 3]) for i in range(0, len(cells), 3)]
    rw = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, rw - r.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20)) for r in rows]
    panel = np.vstack(rows)

    metrics = {k: v for k, v in meta.items() if k in
               ("coverage", "visible_material", "wall_confidence", "object_confidence", "depth_used")}
    metrics["wall_planes"] = [{"plane": p["plane"], "confidence": p["confidence"]} for p in planes]
    return panel, metrics, band_u8, planes


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
                with __import__("torch").inference_mode():
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

        # ── Room perspective (shared vanishing point) ──────────
        persp = estimate_vanishing_point(floor_mask_u8, wall_mask_u8, width, height)
        vp = persp["vp"]

        # Nothing physical sits above the horizon — trim stray floor/wall bleed
        horizon_cut = int(max(0, min(vp[1] - height * 0.02, height * 0.55)))
        if horizon_cut > 0:
            floor_mask_u8[:horizon_cut] = 0

        # ── Compute quads (VP-consistent) ──────────────────────
        floor_quad  = compute_floor_quad(floor_mask_u8, vp=vp, W=width, H=height)
        room_bgr_full = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

        # ── WALL v2: solid surface band + soft foreground layer ────
        wall_meta = {}
        wall_planes_meta = []
        wall_fg_u8 = np.zeros((height, width), np.uint8)
        try:
            depth_map = estimate_depth_map(original_image, width, height) if not is_mock else None
            wall_band_u8, wall_fg_u8, wall_meta = build_wall_layers(
                seg_map_full, floor_mask_u8, width, height, room_bgr_full, depth_map, floor_quad)
            wall_planes = detect_wall_planes(
                wall_band_u8, seg_map_full, room_bgr_full, width, height, vp,
                wall_meta["ceil_y"], wall_meta["floor_y"], depth_map)
            if int(np.count_nonzero(wall_band_u8)) > width * height * 0.01 and wall_planes:
                wall_mask_u8 = wall_band_u8
                wall_quads = wall_planes
                wall_planes_meta = [{"plane": p["plane"], "confidence": p["confidence"]} for p in wall_planes]
            else:
                raise RuntimeError("wall v2 produced empty result")
        except Exception as _we:
            print(f"[wall v2] fallback to legacy region: {_we}")
            wall_region = build_wall_region(seg_map_full, floor_mask_u8, width, height, floor_quad, room_bgr=room_bgr_full)
            wall_strict = extract_clean_wall_mask(seg_map_full, floor_mask_u8, width, height, floor_quad=floor_quad)
            rn, sn = int(np.count_nonzero(wall_region)), int(np.count_nonzero(wall_strict))
            wall_mask_u8 = wall_region if rn >= sn * 0.8 and rn > width * height * 0.01 else wall_strict
            if int(np.count_nonzero(wall_mask_u8)) < width * height * 0.008:
                wall_mask_u8 = cv2.bitwise_or(wall_region, wall_strict)
            wall_quads = compute_wall_quads(wall_mask_u8, floor_quad=floor_quad, vp=vp)

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
            "perspective": {
                "vanishing_point": [round(vp[0], 1), round(vp[1], 1)],
                "horizon_y": round(vp[1], 1),
                "confidence": round(persp["confidence"], 2),
                "room_dims_ft": estimate_room_dims_ft(floor_quad, width, height, ppm),
            },
            "detected_obstacles": detected_obstacles,
            "pixels_per_meter": ppm,
            "floor_area": floor_area,
            "wall_area": wall_area,
            "wall_analysis": {
                "coverage": wall_meta.get("coverage"),
                "visible_material": wall_meta.get("visible_material"),
                "wall_confidence": wall_meta.get("wall_confidence"),
                "object_confidence": wall_meta.get("object_confidence"),
                "depth_used": wall_meta.get("depth_used", False),
                "planes": wall_planes_meta,
            },
            "floor_mask": mask_to_rgba_b64(floor_mask_u8, color=(255, 255, 255, 255)),
            "wall_mask": mask_to_rgba_b64(wall_mask_u8, color=(255, 255, 255, 255)),
            "wall_fg_mask": mask_to_rgba_b64(wall_fg_u8, soft=True),
            "floor_mask_b64": mask_to_rgba_b64(floor_mask_u8, color=(255, 255, 255, 255)),
            "wall_mask_b64": mask_to_rgba_b64(wall_mask_u8, color=(255, 255, 255, 255)),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ============================================================
# ENDPOINT: /api/wall-debug  (Phase A — visual inspection)
# ============================================================
@app.post("/api/wall-debug")
async def wall_debug(
    room:        UploadFile = File(...),
    wall_tile:   UploadFile = File(None),
    wall_pattern: str = Form("grid"),
    wall_finish:  str = Form("matte"),
    wall_tile_wmm: float = Form(300.0),
    wall_tile_hmm: float = Form(600.0),
):
    try:
        img = Image.open(io.BytesIO(await room.read())).convert("RGB")
        W, H = img.size
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if MODEL_AVAILABLE:
            inf = img.resize((1024, 768))
            with __import__("torch").inference_mode():
                out = seg_model(**processor(images=inf, return_tensors="pt"))
            ss = processor.post_process_semantic_segmentation(out, target_sizes=[(768, 1024)])[0].cpu().numpy()
            seg_map = cv2.resize(ss.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
            floor_mask = extract_clean_floor_mask(seg_map, W, H)
            depth_map = estimate_depth_map(img, W, H)
        else:
            floor_mask, _, seg_map = build_mock_masks(W, H)
            depth_map = None

        vp = estimate_vanishing_point(floor_mask, None, W, H)["vp"]
        hc = int(max(0, min(vp[1] - H * 0.02, H * 0.55)))
        if hc > 0:
            floor_mask[:hc] = 0
        fq = compute_floor_quad(floor_mask, vp=vp, W=W, H=H)

        tile_bgr = None
        if wall_tile is not None:
            tb = await wall_tile.read()
            if tb:
                tile_bgr = cv2.cvtColor(np.array(Image.open(io.BytesIO(tb)).convert("RGB")), cv2.COLOR_RGB2BGR)

        panel, metrics, _, _ = build_wall_debug_panel(
            bgr, seg_map, floor_mask, W, H, depth_map, fq,
            tile_bgr=tile_bgr, tile_wmm=wall_tile_wmm, tile_hmm=wall_tile_hmm,
            pattern=wall_pattern, finish=wall_finish)

        ok, enc = cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return JSONResponse({
            "status": "success",
            "panel": "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode(),
            "metrics": metrics,
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
    wall_fg_mask:          UploadFile = File(None),
    wall_add_mask:         UploadFile = File(None),   # manual: paint MORE wall
    wall_remove_mask:      UploadFile = File(None),   # manual: keep original (no material)
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
    # Real-world scale (metric perspective tiling)
    pixels_per_meter:      float = Form(0.0),
    floor_tile_wmm:        float = Form(600.0),
    floor_tile_hmm:        float = Form(600.0),
    wall_tile_wmm:         float = Form(300.0),
    wall_tile_hmm:         float = Form(600.0),
    floor_slab:            int   = Form(0),   # 1 = force continuous slab
    wall_slab:             int   = Form(0),
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
        wall_fg_u8    = None
        if floor_mask is not None:
            fm_bytes = await floor_mask.read()
            floor_mask_u8 = parse_mask_upload(fm_bytes)
        if wall_mask is not None:
            wm_bytes = await wall_mask.read()
            wall_mask_u8 = parse_mask_upload(wm_bytes)
        if wall_fg_mask is not None:
            wfg_bytes = await wall_fg_mask.read()
            wall_fg_u8 = parse_mask_upload(wfg_bytes, soft=True)

        # ── Manual wall-mask corrections (Phase H brush tool) ──────────
        manual_planes = []
        if wall_mask_u8 is not None:
            def _fit(m):
                if m is None:
                    return None
                if m.shape[:2] != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                return m
            add_m = _fit(parse_mask_upload(await wall_add_mask.read())) if wall_add_mask is not None else None
            rem_m = _fit(parse_mask_upload(await wall_remove_mask.read())) if wall_remove_mask is not None else None
            if add_m is not None and np.any(add_m > 0):
                new_area = cv2.bitwise_and(add_m, cv2.bitwise_not(wall_mask_u8))
                wall_mask_u8 = cv2.bitwise_or(wall_mask_u8, add_m)
                if wall_fg_u8 is not None:
                    wall_fg_u8 = cv2.bitwise_and(wall_fg_u8, cv2.bitwise_not(add_m))  # painted = wall, not fg
                na, la, sa, _ = cv2.connectedComponentsWithStats(new_area, 8)
                for i in range(1, na):
                    x, y, w, h, a = sa[i]
                    if a > W * H * 0.0008:
                        manual_planes.append({
                            "quad": [[int(x), int(y)], [int(x + w), int(y)],
                                     [int(x + w), int(y + h)], [int(x), int(y + h)]],
                            "bbox": [int(x), int(y), int(w), int(h)],
                            "plane": "manual", "confidence": 1.0,
                        })
            if rem_m is not None and np.any(rem_m > 0):
                remd = cv2.dilate(rem_m, np.ones((3, 3), np.uint8))
                wall_mask_u8[remd > 0] = 0
                if wall_fg_u8 is not None:
                    soft = cv2.GaussianBlur(rem_m.astype(np.float32) / 255.0, (0, 0), max(1.0, W * 0.004))
                    wall_fg_u8 = np.maximum(wall_fg_u8, (np.clip(soft, 0, 1) * 255).astype(np.uint8))
        if manual_planes:
            wqs = (wqs or []) + manual_planes

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

                # Metric tiling plan — real tile size + perspective foreshortening
                size_mult = float(np.clip((floor_scale or 0.18) / 0.18, 0.4, 3.0))
                f_tx, f_ty, plane_w, plane_h = plan_floor_tiling(
                    fq, W, H, pixels_per_meter,
                    floor_tile_wmm or 600.0, floor_tile_hmm or 600.0, size_mult
                )
                f_gc = parse_grout_color(floor_grout_color if floor_grout_color else grout_color)

                flat_floor = build_tile_surface(
                    f_tile_np, plane_w, plane_h, f_tx, f_ty,
                    pattern=floor_pattern or pattern,
                    grout_width=floor_grout_width if floor_grout_width is not None else grout_width,
                    grout_color=f_gc,
                    rotation_deg=floor_rotation if floor_rotation is not None else rotation,
                    brightness=floor_brightness if floor_brightness is not None else brightness,
                    far_fade=0.07, far_blur=True,
                    slab=bool(floor_slab),
                )
                composite = apply_homography_warp(
                    composite, flat_floor, fq, floor_mask_u8,
                    shadow_strength=floor_shadow_strength if floor_shadow_strength is not None else shadow_strength,
                    finish=floor_finish or finish,
                )

        # ── 2. Apply WALL Tiling (only if wall_tile is provided or fallback tile with wall/both target) ──
        active_wall_file = wall_tile if wall_tile is not None else (tile if target in ("wall", "both") else None)
        if active_wall_file is not None:
            if not wqs and wall_mask_u8 is not None and np.any(wall_mask_u8 > 0):
                wqs = compute_wall_quads(wall_mask_u8)
            if not wqs:
                wqs = [{
                    "quad": [[0, 0], [W, 0], [W, int(H * 0.6)], [0, int(H * 0.6)]],
                    "bbox": [0, 0, W, int(H * 0.6)]
                }]

        if active_wall_file is not None and wqs:
            w_bytes = await active_wall_file.read()
            if w_bytes:
                w_pil = Image.open(io.BytesIO(w_bytes)).convert("RGB")
                w_tile_np = np.array(w_pil)

                w_gc = parse_grout_color(wall_grout_color if wall_grout_color else grout_color)
                w_size_mult = float(np.clip((wall_scale or 0.18) / 0.18, 0.4, 3.0))
                ppm_eff = float(pixels_per_meter) if pixels_per_meter and pixels_per_meter > 1 else H * 0.40
                wall_h_m = 2.6
                # Large-format wall tiles (>= 800 mm) read best as a book-matched slab
                wall_is_slab = bool(wall_slab) or (wall_tile_wmm or 0) >= 800 or (wall_tile_hmm or 0) >= 1000
                tw_m = max(0.05, (wall_tile_wmm or 300.0) / 1000.0 * w_size_mult)
                th_m = max(0.05, (wall_tile_hmm or 600.0) / 1000.0 * w_size_mult)

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

                    seg_fg = None
                    if wall_fg_u8 is not None:
                        seg_fg = wall_fg_u8.copy()
                        seg_fg[seg_mask == 0] = 0

                    seg_w_m = float(np.clip(bw / ppm_eff, 1.0, 16.0))
                    w_tx = float(np.clip(seg_w_m / tw_m, 2.0, 70.0))
                    w_ty = float(np.clip(wall_h_m / th_m, 2.0, 45.0))
                    pw = 2000
                    ph = int(np.clip(round(pw * wall_h_m / seg_w_m), 500, 3600))
                    flat_wall = build_tile_surface(
                        w_tile_np, pw, ph, w_tx, w_ty,
                        pattern=wall_pattern or pattern,
                        grout_width=wall_grout_width if wall_grout_width is not None else grout_width,
                        grout_color=w_gc,
                        rotation_deg=wall_rotation if wall_rotation is not None else rotation,
                        brightness=wall_brightness if wall_brightness is not None else brightness,
                        far_fade=0.05, far_blur=True,
                        slab=wall_is_slab,
                    )
                    eff_pattern = wall_pattern or pattern
                    eff_grout = wall_grout_width if wall_grout_width is not None else grout_width
                    if wall_is_slab:
                        m_kind = "slab"
                    elif not eff_grout and eff_pattern == "grid":
                        m_kind = "paint"
                    elif eff_pattern in ("herringbone", "chevron", "basketweave", "brick", "brick_third", "windmill"):
                        m_kind = "textured"
                    else:
                        m_kind = "tile"

                    composite = apply_wall_photorealism_warp(
                        composite, flat_wall, seg_quad, seg_mask,
                        shadow_strength=wall_shadow_strength if wall_shadow_strength is not None else shadow_strength,
                        finish=wall_finish or finish,
                        fg_alpha=seg_fg,
                        material_kind=m_kind,
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
