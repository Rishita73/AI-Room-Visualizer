"""
Stage 7 & Stage 8: 3D Perspective Ground Quad & Tile Surface Synthesizer
"""

import cv2
import numpy as np

class PerspectiveEngine:
    def __init__(self):
        pass

    def compute_ground_quad(self, final_mask, ground_plane_info, W, H):
        """
        Computes the 4-corner perspective quad [P1_far_left, P2_far_right, P3_near_right, P4_near_left].
        Prioritizes 3D RANSAC projected plane corners if valid, with robust mask trapezoid fallback.
        """
        if ground_plane_info is not None and ground_plane_info.get("corners_2d") is not None:
            corners = ground_plane_info["corners_2d"]
            if isinstance(corners, (list, np.ndarray)) and len(corners) == 4:
                # Validate that near corners are strictly lower than far corners and outward flaring
                c_fl, c_fr, c_nr, c_nl = corners
                if (c_nr[1] > c_fl[1] + H * 0.1 and c_nl[1] > c_fr[1] + H * 0.1 and 
                    (c_nr[0] - c_nl[0]) >= (c_fr[0] - c_fl[0]) * 0.85):
                    return corners

        ys, xs = np.where(final_mask)
        if len(ys) == 0:
            return [[-int(W * 0.15), int(H * 0.5)], [int(W * 1.15), int(H * 0.5)],
                    [int(W * 1.15), int(H * 1.02)], [-int(W * 0.15), int(H * 1.02)]]

        top_y = float(ys.min())
        bot_y = float(ys.max())
        
        far_y = max(top_y, H * 0.30)
        near_y = min(bot_y + H * 0.05, H * 1.02)

        far_band = final_mask[int(far_y):int(min(H - 1, far_y + H * 0.08)), :].max(axis=0) if int(far_y) < H else []
        fb = np.where(far_band)[0] if len(far_band) > 0 else []
        if len(fb) > 10:
            far_lx = float(np.percentile(fb, 5))
            far_rx = float(np.percentile(fb, 95))
        else:
            far_lx, far_rx = float(W * 0.25), float(W * 0.75)

        near_band = final_mask[int(max(0, near_y - H * 0.08)):int(min(H, near_y)), :].max(axis=0) if int(near_y) > 0 else []
        nb = np.where(near_band)[0] if len(near_band) > 0 else []
        if len(nb) > 10:
            near_lx = float(np.percentile(nb, 2))
            near_rx = float(np.percentile(nb, 98))
        else:
            near_lx, near_rx = float(-W * 0.15), float(W * 1.15)

        # Enforce minimum architectural span for far corners (prevent collapsed tips)
        min_far_w = max(float(W * 0.45), 320.0)
        if (far_rx - far_lx) < min_far_w:
            mid_far = (far_lx + far_rx) / 2.0
            far_lx = max(-float(W * 0.1), mid_far - min_far_w / 2.0)
            far_rx = min(float(W * 1.1), mid_far + min_far_w / 2.0)

        # Enforce trapezoidal outward flare
        if near_rx - near_lx < (far_rx - far_lx) * 1.1:
            cx = (near_lx + near_rx) / 2.0
            hw = max((far_rx - far_lx) * 0.75, W * 0.55)
            near_lx = cx - hw
            near_rx = cx + hw

        # Extract per-column top floor boundary to compute true baseline angle
        col_has = final_mask.max(axis=0) > 0
        cols = np.where(col_has)[0]
        if len(cols) > 10:
            ftop = final_mask.argmax(axis=0).astype(np.float32)
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
                    y_far_l = float(np.clip(m_top * far_lx + c_top, top_y, near_y - H * 0.1))
                    y_far_r = float(np.clip(m_top * far_rx + c_top, top_y, near_y - H * 0.1))
                else:
                    y_far_l = y_far_r = far_y
            else:
                y_far_l = y_far_r = far_y
        else:
            y_far_l = y_far_r = far_y

        return [
            [float(far_lx), float(y_far_l)],
            [float(far_rx), float(y_far_r)],
            [float(near_rx), float(near_y)],
            [float(near_lx), float(near_y)]
        ]

    def compute_room_box_geometry(self, floor_quad, wall_mask, W, H):
        P_FL = np.array(floor_quad[0], dtype=np.float32)
        P_FR = np.array(floor_quad[1], dtype=np.float32)
        P_NR = np.array(floor_quad[2], dtype=np.float32)
        P_NL = np.array(floor_quad[3], dtype=np.float32)

        # Wall top bounds
        ys, xs = np.where(wall_mask) if wall_mask is not None and np.any(wall_mask) else ([], [])
        if len(ys) > 0:
            top_wall_y = float(np.percentile(ys, 5))
            top_wall_y = float(np.clip(top_wall_y, 0, min(P_FL[1], P_FR[1]) - H * 0.15))
        else:
            top_wall_y = float(H * 0.05)

        # Far wall height
        far_wall_h = max(float(min(P_FL[1], P_FR[1]) - top_wall_y), float(H * 0.35))

        C_FL = np.array([P_FL[0], max(0.0, P_FL[1] - far_wall_h)], dtype=np.float32)
        C_FR = np.array([P_FR[0], max(0.0, P_FR[1] - far_wall_h)], dtype=np.float32)

        # Perspective expansion for near corners
        near_wall_h = far_wall_h * 1.45
        C_NL = np.array([P_NL[0], max(-H * 0.1, P_NL[1] - near_wall_h)], dtype=np.float32)
        C_NR = np.array([P_NR[0], max(-H * 0.1, P_NR[1] - near_wall_h)], dtype=np.float32)

        # Shared 3D Wall Quads
        back_wall_quad = [C_FL.tolist(), C_FR.tolist(), P_FR.tolist(), P_FL.tolist()]
        left_wall_quad = [C_NL.tolist(), C_FL.tolist(), P_FL.tolist(), P_NL.tolist()]
        right_wall_quad = [C_FR.tolist(), C_NR.tolist(), P_NR.tolist(), P_FR.tolist()]

        return {
            "floor_quad": floor_quad,
            "back_wall_quad": back_wall_quad,
            "left_wall_quad": left_wall_quad,
            "right_wall_quad": right_wall_quad,
            "corners": {
                "P_FL": P_FL.tolist(), "P_FR": P_FR.tolist(),
                "P_NR": P_NR.tolist(), "P_NL": P_NL.tolist(),
                "C_FL": C_FL.tolist(), "C_FR": C_FR.tolist(),
                "C_NL": C_NL.tolist(), "C_NR": C_NR.tolist(),
            }
        }

    def plan_tiling(self, floor_quad, W, H, ppm=200.0, tile_w_mm=600.0, tile_h_mm=600.0, tile_scale=1.0, is_wall=False):
        """
        Plans tile surface grid dimensions and resolution matching true architectural scale.
        Produces 4-7 tiles across visible room by default (not 40 micro-tiles).
        """
        pw = int(np.clip(W * 1.5, 1400, 2400))
        ph = int(np.clip(H * 1.5, 1000, 2000))

        # Normalize UI scale multiplier (slider default is 0.18)
        raw_scale = float(tile_scale) if tile_scale else 1.0
        if raw_scale <= 0.40:
            size_mult = float(np.clip(raw_scale / 0.18, 0.4, 3.0))
        else:
            size_mult = float(np.clip(raw_scale, 0.4, 3.0))

        tw_mm = float(tile_w_mm) if tile_w_mm else 600.0
        th_mm = float(tile_h_mm) if tile_h_mm else 600.0
        aspect = max(tw_mm, th_mm) / max(1.0, min(tw_mm, th_mm))

        if is_wall:
            if aspect > 2.5: # Wood planks (e.g. 150x900 mm)
                tx = max(1, int(round(3.0 / size_mult)))
                ty = max(2, int(round(2.0 / size_mult)))
            else: # Standard wall tiles / slabs
                tx = max(2, int(round(4.0 / size_mult)))
                ty = max(2, int(round(3.0 / size_mult)))
        else:
            if tw_mm >= 1200.0 or th_mm >= 1200.0: # Large format marble slabs
                tx = max(2, int(round(3.0 / size_mult)))
                ty = max(2, int(round(2.0 / size_mult)))
            elif aspect > 2.5: # Wood planks on floor
                tx = max(2, int(round(4.0 / size_mult)))
                ty = max(3, int(round(6.0 / size_mult)))
            else: # Standard 600x600 or 800x800 floor tiles
                tx = max(3, int(round(5.0 / size_mult)))
                ty = max(2, int(round(4.0 / size_mult)))

        return tx, ty, pw, ph

    def build_surface(self, tile_bgr, pw, ph, tx, ty, pattern="grid", grout_w=0, grout_col=(200, 200, 200), slab=False):
        """
        Builds a high-fidelity tiled surface with seamless random tile rotation/flip
        variants to prevent repetitive stamping patterns in natural stone/wood.
        """
        if slab:
            # Bookmatched slab mode
            top = np.hstack([tile_bgr, cv2.flip(tile_bgr, 1)])
            block = np.vstack([top, cv2.flip(top, 0)])
            return cv2.resize(block, (pw, ph), interpolation=cv2.INTER_AREA)

        surface = np.zeros((ph, pw, 3), dtype=np.uint8)
        cell_w = max(16, pw // max(1, tx))
        cell_h = max(16, ph // max(1, ty))
        cell = cv2.resize(tile_bgr, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

        # Subtle natural variants: 0 deg, 180 deg, flip X, flip Y
        variants = [
            cell,
            cv2.flip(cell, 1),
            cv2.flip(cell, 0),
            cv2.rotate(cell, cv2.ROTATE_180)
        ]

        n_rows = int(np.ceil(ph / cell_h)) + 1
        n_cols = int(np.ceil(pw / cell_w)) + 1

        for r in range(n_rows):
            for c in range(n_cols):
                ox = 0
                if pattern == "brick" and (r % 2):
                    ox = cell_w // 2
                x1, y1 = c * cell_w + ox, r * cell_h
                x2, y2 = min(x1 + cell_w, pw), min(y1 + cell_h, ph)
                if x1 < pw and y1 < ph and x2 > x1 and y2 > y1:
                    v = variants[(r * 3 + c * 7) % 4]
                    surface[y1:y2, x1:x2] = v[:y2 - y1, :x2 - x1]

        # Apply realistic grout lines
        gw = int(max(0, grout_w))
        if gw > 0:
            g_col = tuple(int(c) for c in grout_col)
            for r in range(n_rows + 1):
                y = min(ph - 1, r * cell_h)
                surface[max(0, y - gw // 2):min(ph, y + (gw + 1) // 2), :] = g_col
            for c in range(n_cols + 1):
                x = min(pw - 1, c * cell_w)
                surface[:, max(0, x - gw // 2):min(pw, x + (gw + 1) // 2)] = g_col

        return surface

    def warp_to_quad(self, surface, floor_quad, W, H):
        """Applies perspective homography warping to project the flat tile surface into room perspective."""
        ph, pw = surface.shape[:2]
        src_pts = np.array([
            [0, 0],
            [pw - 1, 0],
            [pw - 1, ph - 1],
            [0, ph - 1]
        ], dtype=np.float32)
        dst_pts = np.array(floor_quad, dtype=np.float32)

        H_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(surface, H_mat, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return warped
