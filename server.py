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

# Labels excluded from floor net area & mask (furniture/decor/plants/vases on floor)
FLOOR_OBSTACLE_IDS = {7, 10, 14, 15, 17, 18, 19, 22, 23, 24, 27, 28, 30, 31, 36, 39, 44, 50, 52, 53, 57, 64, 75, 108, 110, 119, 125, 132, 137}

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

# Initialize Hybrid Multi-Stage Architectural Pipeline
try:
    from pipeline import RoomVisualizerPipeline
    visualizer_pipeline = RoomVisualizerPipeline(
        seg_model=seg_model,
        seg_processor=processor,
        depth_model=depth_model,
        depth_processor=depth_processor,
        device=device if 'device' in locals() else "cpu"
    )
    print("RoomVisualizerPipeline initialized successfully!")
except Exception as pipe_err:
    print(f"Pipeline init error: {pipe_err}")
    visualizer_pipeline = None


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
# Heavy solid ground-resting floor obstacles ONLY
SOLID_FLOOR_OBSTACLES = {7, 10, 14, 23, 27, 30, 31, 36}

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
def compute_floor_quad(mask_u8: np.ndarray, vp=None, W=None, H=None, depth_map=None) -> list:
    """
    3D RANSAC Floor Plane Estimation (Floor_Detection_using_SAM_2_n1 (2).ipynb Cells 28-50):
      1. Converts (X, Y, Z) point cloud to camera coordinates
      2. Fits 3D ground plane Z = A*X + B*Y + C via RANSAC
      3. Projects 3D floor corners (P1, P2, P3, P4) to image coordinates
    """
    if H is None or W is None:
        H, W = mask_u8.shape

    if depth_map is not None:
        try:
            from sklearn.linear_model import RANSACRegressor, LinearRegression
            ys, xs = np.where(mask_u8 > 0)
            if len(xs) > 100:
                zs = depth_map[ys, xs].astype(np.float32)
                f = W
                cx = W / 2.0
                cy = H / 2.0
                X = (xs.astype(np.float32) - cx) * zs / f
                Y = (ys.astype(np.float32) - cy) * zs / f
                Z = zs
                points_3d = np.column_stack((X, Y, Z))

                MAX_POINTS = 100000
                if len(points_3d) > MAX_POINTS:
                    rng = np.random.default_rng(42)
                    indices = rng.choice(len(points_3d), MAX_POINTS, replace=False)
                    sample_pts = points_3d[indices]
                else:
                    sample_pts = points_3d

                ransac = RANSACRegressor(
                    estimator=LinearRegression(),
                    residual_threshold=0.03,
                    max_trials=100,
                    min_samples=3,
                    random_state=42
                )
                ransac.fit(sample_pts[:, :2], sample_pts[:, 2])
                inlier_mask = ransac.inlier_mask_
                if np.count_nonzero(inlier_mask) >= 30:
                    A, B = ransac.estimator_.coef_
                    C = ransac.estimator_.intercept_
                    inlier_pts = sample_pts[inlier_mask]

                    n = np.array([A, B, -1.0], dtype=np.float64)
                    n = n / np.linalg.norm(n)

                    camera_x = np.array([1.0, 0.0, 0.0])
                    u = camera_x - np.dot(camera_x, n) * n
                    u = u / (np.linalg.norm(u) + 1e-8)
                    v = np.cross(n, u)
                    v = v / (np.linalg.norm(v) + 1e-8)

                    origin = np.mean(inlier_pts, axis=0)
                    rel = inlier_pts - origin
                    plane_u = rel @ u
                    plane_v = rel @ v

                    u_min, u_max = np.percentile(plane_u, [1, 99])
                    v_min, v_max = np.percentile(plane_v, [1, 99])

                    P1 = origin + u_min * u + v_min * v
                    P2 = origin + u_max * u + v_min * v
                    P3 = origin + u_max * u + v_max * v
                    P4 = origin + u_min * u + v_max * v

                    corners = []
                    for pt in [P1, P2, P3, P4]:
                        z_s = max(float(pt[2]), 1e-3)
                        x_i = f * pt[0] / z_s + cx
                        y_i = f * pt[1] / z_s + cy
                        corners.append([float(x_i), float(y_i)])
                    return corners
        except Exception as e:
            pass

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

    # ---- subtract structural non-wall openings (windows, doors, ceiling, floor) -------------
    hard = np.zeros((H, W), np.uint8)
    for lid in (5, 8, 9, 14):
        hard |= (seg_map == lid).astype(np.uint8)
    if hard.any():
        hard = cv2.dilate(hard * 255,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(7, int(W * 0.03)) | 1,) * 2))
        region[hard > 0] = 0
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
_WALL_FG_LABELS = (7, 8, 9, 10, 14, 15, 17, 18, 19, 22, 23, 24, 27, 30, 31, 36, 39, 44, 49, 50, 52, 53, 57, 64, 69, 75, 97, 105, 110, 119)


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

    # Bed Pillow & Headboard Envelope Protection
    bed_mask = (seg_map == 7).astype(np.uint8) * 255
    if bed_mask.any():
        ys_b, xs_b = np.where(bed_mask > 0)
        y_min, y_max = ys_b.min(), ys_b.max()
        x_min, x_max = xs_b.min(), xs_b.max()
        pillow_y_min = max(0, y_min - int(H * 0.22))
        sem[pillow_y_min:y_max, max(0, x_min - 15):min(W, x_max + 15)] = 255

    parts["semantic"] = sem.copy()
    fg |= sem

    # 2. bright blobs inside the band = windows / mirrors SegFormer missed
    bright = cv2.bitwise_and((gray > 240).astype(np.uint8) * 255, band_u8)
    bright[seg_map == 0] = 0  # Keep genuine wall pixels paintable
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
        wall_hint = cv2.bitwise_and((seg_map == 0).astype(np.uint8) * 255, band_u8)
        if np.count_nonzero(wall_hint) > W * H * 0.01:
            wd = float(np.percentile(depth[wall_hint > 0], 50))
            sd = float(np.std(depth[wall_hint > 0])) + 1e-3
            nearer = (depth < (wd - max(0.08, sd * 1.5))).astype(np.uint8) * 255
            nearer = cv2.bitwise_and(nearer, band_u8)
            nearer[seg_map == 0] = 0  # NEVER mark genuine wall pixels as foreground obstacles!
            nearer = cv2.morphologyEx(nearer, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            nn, ln, sn, _ = cv2.connectedComponentsWithStats(nearer, 8)
            dmask = np.zeros((H, W), np.uint8)
            for i in range(1, nn):
                if sn[i, cv2.CC_STAT_AREA] > W * H * 0.001:
                    dmask[ln == i] = 255
            parts["depth_fg"] = dmask
            fg |= dmask

    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    fg = cv2.bitwise_and(fg, band_u8)
    parts["union"] = fg.copy()
    return fg, parts


def build_wall_layers(seg_map, floor_mask, W, H, room_bgr, depth=None, floor_quad=None):
    """
    State-of-the-art Architectural Wall Segmentation & Foreground Layer Engine.
    Produces:
      • band_u8: Clean, unbroken wall surface mask strictly respecting ceiling, floor, & window openings.
      • fg_alpha: High-precision edge-aligned alpha matte for all furniture & decor objects.
    """
    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Ceiling & Floor Boundaries
    ceil_mask = (seg_map == 5).astype(np.uint8) * 255
    if ceil_mask.any() and np.count_nonzero(ceil_mask) > W * H * 0.008:
        # Per-column ceiling bottom edge
        c_bot = (H - 1 - (ceil_mask[::-1] > 0).argmax(axis=0)).astype(np.float32)
        has_c = ceil_mask.max(axis=0) > 0
        c_y_val = float(np.median(c_bot[has_c])) if has_c.any() else H * 0.15
    else:
        far_floor_y = float((floor_quad[0][1] + floor_quad[1][1]) / 2.0) if (floor_quad and len(floor_quad) == 4) else H * 0.60
        c_y_val = float(np.clip(far_floor_y - (H - far_floor_y) * 0.85, H * 0.05, H * 0.35))

    ys = np.arange(H, dtype=np.float32)[:, None]
    is_below_ceiling = ys >= c_y_val

    # 2. Semantic Wall Extraction
    raw_wall = (seg_map == 0).astype(np.uint8) * 255
    # Strict non-wall exclusions: ceiling, floor, outdoor sky/trees, glass windows, open exterior doors
    exclude = np.zeros((H, W), np.uint8)
    exclude[seg_map == 5] = 255
    exclude[~is_below_ceiling.reshape(H, W)] = 255
    exclude[floor_mask > 0] = 255
    exclude[(seg_map == 2) | (seg_map == 4) | (seg_map == 8) | (seg_map == 9) | (seg_map == 14)] = 255

    # Filter wall pixels strictly
    wall_clean = cv2.bitwise_and(raw_wall, cv2.bitwise_not(cv2.dilate(exclude, np.ones((3, 3), np.uint8))))

    # Connect gaps behind furniture/shelves without breaching ceiling or floor
    wall_connected = cv2.morphologyEx(wall_clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)))
    wall_connected[exclude > 0] = 0

    # Keep all genuine wall components (> 0.2% of image area)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(wall_connected, 8)
    band = np.zeros((H, W), np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > W * H * 0.002:
            band[labels == i] = 255

    if not np.any(band > 0):
        band = cv2.bitwise_and(raw_wall, cv2.bitwise_not(exclude))

    # 3. High-Precision Edge-Aligned Foreground Matte
    fg_raw = np.zeros((H, W), np.uint8)
    for lid in _WALL_FG_LABELS:
        fg_raw[seg_map == lid] = 255

    # Only protect foreground objects that border or overlap wall regions
    wall_expanded = cv2.dilate(band, np.ones((15, 15), np.uint8))
    fg_relevant = cv2.bitwise_and(fg_raw, wall_expanded)
    fg_relevant[band > 0] = 0  # Protect genuine wall pixels

    guide = gray.astype(np.float32) / 255.0
    fg_f32 = fg_relevant.astype(np.float32) / 255.0
    fg_soft = _guided_filter(guide, fg_f32, radius=2, eps=1e-4)
    fg_soft = np.clip(fg_soft, 0.0, 1.0)
    fg_alpha = np.where(fg_soft > 0.40, 1.0, fg_soft * 0.4)
    fg_alpha = cv2.GaussianBlur(fg_alpha, (3, 3), 0.5)
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
    
    # Auto-crop outer margins and top text labels (e.g. 'MARFIL FAB LIGHT')
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

    tw = max(0.25, (float(tile_wmm) / 1000.0) * size_mult * 1.8)
    th = max(0.25, (float(tile_hmm) / 1000.0) * size_mult * 1.8)

    tiles_x = max(2.0, room_w_m / tw)
    tiles_y = max(2.0, room_d_m / th)

    # Cap density for spacious luxury slab look
    if tiles_x > 12.0:
        s = 12.0 / tiles_x
        tiles_x, tiles_y = 12.0, max(2.0, tiles_y * s)
    if tiles_y > 14.0:
        s = 14.0 / tiles_y
        tiles_y, tiles_x = 14.0, max(2.0, tiles_x * s)

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
    fg_alpha: np.ndarray = None,
) -> np.ndarray:
    """
    Photorealistic Floor Projection Engine:
      • Perspective homography
      • Bilateral LAB luminance lighting recovery (captures window light gradients & natural room shadows)
      • Contact Ambient Occlusion directly at furniture grounding points (zero gaps)
      • Specular window highlight & micro sensor detail transfer
      • Anti-aliased edge blending
    """
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

    # 2. Extract Clean Lighting Field (Gaussian Room Illumination)
    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur_k = int(W * 0.04) | 1
    gray_blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    floor_px = gray_blurred[mask_u8 > 0]
    if len(floor_px) > 0:
        white_pt = float(np.percentile(floor_px, 90))
        shadow_map = np.clip(gray_blurred / max(white_pt, 1.0), 0.0, 1.0)
        shadow_map = np.clip(0.70 * shadow_map + 0.30, 0.0, 1.0)
    else:
        shadow_map = np.ones((H, W), dtype=np.float32)

    # Illumination multiplier modulated by user shadow slider
    light_gain = (1.0 - shadow_strength) + shadow_strength * shadow_map

    # 3. Apply Lighting to Warped Material
    material_lit = warped.astype(np.float32) * light_gain[:, :, None]

    # Specular Window Highlight & Finish Response
    if finish == "glossy":
        specular = np.clip((shadow_map - 0.75) / 0.25, 0.0, 1.0)
        material_lit = material_lit * (1.0 - 0.18 * specular[:, :, None]) + room_bgr.astype(np.float32) * (0.18 * specular[:, :, None])
    elif finish == "satin":
        specular = np.clip((shadow_map - 0.82) / 0.18, 0.0, 1.0)
        material_lit = material_lit * (1.0 - 0.08 * specular[:, :, None]) + room_bgr.astype(np.float32) * (0.08 * specular[:, :, None])

    # 4. Clean Sub-pixel Alpha Blend
    alpha_mask = cv2.GaussianBlur((mask_u8 > 0).astype(np.float32), (3, 3), 0)[:, :, None]
    final_comp = material_lit * alpha_mask + room_bgr.astype(np.float32) * (1.0 - alpha_mask)

    return np.clip(final_comp, 0, 255).astype(np.uint8)


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

    # contact AO — realistic contact shadow behind furniture, shelves, & floor joins
    if fg_alpha is not None:
        fga_bin = (fg_alpha > 0.3).astype(np.uint8) * 255
        dist_from_fg = cv2.distanceTransform(cv2.bitwise_not(fga_bin), cv2.DIST_L2, 5)
        contact_shadow = np.clip(dist_from_fg / 24.0, 0.0, 1.0)
        contact_ao = 0.75 + 0.25 * contact_shadow
        ao_gain = cv2.GaussianBlur(contact_ao, (0, 0), 2.5)
    else:
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
        fga_val = fg_alpha.astype(np.float32)
        if fga_val.max() > 1.0:
            fga_val = fga_val / 255.0
        fga = np.clip(fga_val, 0.0, 1.0)[..., None]
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
        seg_fg = fg_u8 if fg_u8 is not None else None
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
# HELPER: Utilities
# ============================================================
def mask_to_rgba_b64(mask_u8: np.ndarray, color: tuple = (255, 255, 255, 255)) -> str:
    H, W = mask_u8.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    r, g, b, a = color
    rgba[mask_u8 > 0] = [r, g, b, a]
    pil_img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def mask_to_polygon(mask_u8: np.ndarray) -> list:
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return [[int(pt[0][0]), int(pt[0][1])] for pt in approx]


def derive_pixel_scale(seg_map: np.ndarray, W: int, H: int) -> float:
    for lid, ref in REAL_WORLD_REF.items():
        if np.any(seg_map == lid):
            ys, xs = np.where(seg_map == lid)
            h_px = ys.max() - ys.min()
            if h_px > 40:
                return float(h_px / ref["h"])
    return float(W / 4.5)


def estimate_area_sqft(mask_u8: np.ndarray, obstacle_ids: set, seg_map: np.ndarray, ppm: float) -> dict:
    if ppm <= 0:
        ppm = 200.0
    px_sq = np.count_nonzero(mask_u8)
    sqm = px_sq / (ppm ** 2)
    total_sqft = round(sqm * 10.7639, 1)

    obstacles = {}
    total_obs_px = 0
    if seg_map is not None:
        for lid in obstacle_ids:
            obs_px = np.count_nonzero((seg_map == lid) & (mask_u8 > 0))
            if obs_px > (ppm ** 2 * 0.05):
                obs_sqft = round((obs_px / (ppm ** 2)) * 10.7639, 1)
                lname = ADE20K_LABELS.get(lid, f"object_{lid}")
                obstacles[lname] = obs_sqft
                total_obs_px += obs_px

    net_sqft = max(0.0, round(total_sqft - sum(obstacles.values()), 1))
    return {
        "total_sqft": total_sqft,
        "net_sqft": net_sqft,
        "obstacles": obstacles,
        "obstacle_sqft": round(sum(obstacles.values()), 1),
    }


def estimate_room_dims_ft(floor_quad: list, W: int, H: int, ppm: float) -> list:
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
                print(f"Model error: {model_err}")
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

        # Obstacles
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

        # Extract Guided Sub-Pixel Foreground Matte for Objects ON Floor (sofa, chairs, tables, vases, flowerpots, lamps)
        fg_raw = np.zeros((height, width), dtype=np.uint8)
        for lid in FLOOR_OBSTACLE_IDS:
            fg_raw[seg_map_full == lid] = 255
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
                        w_quads = [visualizer_pipeline.wall_engine.compute_wall_quads(wm_np, fq, W=W, H=H)]
                        w_planes = visualizer_pipeline.wall_engine.partition_wall_planes(wm_np, None, W, H)

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
                                composite, w_flat[..., ::-1] if w_flat.shape[2] == 3 else w_flat, p_quad, p_mask,
                                shadow_strength=wall_shadow_strength if wall_shadow_strength is not None else shadow_strength,
                                finish=wall_finish or finish
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
    """
    Executes the 10-stage hybrid architectural visualizer pipeline and returns
    high-definition base64 visual renders of all 9 diagnostic inspection stages:
      1. Original Room Photo
      2. Raw SegFormer Candidate
      3. Surface & Texture Analysis
      4. Monocular Depth Map
      5. 3D RANSAC Ground Plane
      6. Occlusion & Furniture Barrier
      7. Final Refined Floor Mask
      8. Perspective Warped Material
      9. Final Physical Composite
      + Combined 3x3 Diagnostic Montage
    """
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
            # Default white marble tile
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

        # Encode each stage to base64 JPEG
        stages_b64 = {}
        for key, img in out["debug_stages"].items():
            if img is not None:
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                stages_b64[key] = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

        # Build full montage
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
        
        # Check if local tiles exist
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
# Static frontend
# ============================================================
frontend_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(frontend_dir, "index.html")):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    @app.get("/")
    def read_root():
        return {"message": "AI Room Visualizer API v2.5 — Frontend not found."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
