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
        if ground_plane_info is not None and "corners_2d" in ground_plane_info:
            corners = ground_plane_info["corners_2d"]
            if len(corners) == 4:
                return corners

        ys, xs = np.where(final_mask)
        if len(ys) == 0:
            return [[-int(W * 0.15), int(H * 0.5)], [int(W * 1.15), int(H * 0.5)],
                    [int(W * 1.15), int(H * 1.02)], [-int(W * 0.15), int(H * 1.02)]]

        top_y = float(ys.min())
        bot_y = float(ys.max())
        
        far_y = max(top_y, H * 0.40)
        near_y = min(bot_y + H * 0.05, H * 1.02)

        far_lx, far_rx = float(W * 0.25), float(W * 0.75)
        near_lx, near_rx = float(-W * 0.15), float(W * 1.15)

        return [
            [far_lx, far_y],
            [far_rx, far_y],
            [near_rx, near_y],
            [near_lx, near_y]
        ]

    def plan_tiling(self, floor_quad, W, H, ppm=200.0, tile_w_mm=600.0, tile_h_mm=600.0, tile_scale=1.0):
        """Plans the tile surface grid dimensions and resolution."""
        tw_px = max(16.0, (tile_w_mm / 1000.0) * ppm * tile_scale)
        th_px = max(16.0, (tile_h_mm / 1000.0) * ppm * tile_scale)

        pw = int(np.clip(W * 1.5, 1200, 2400))
        ph = int(np.clip(H * 1.5, 1200, 2400))

        tx = max(2, int(np.ceil(pw / tw_px)))
        ty = max(2, int(np.ceil(ph / th_px)))

        return tx, ty, pw, ph

    def build_surface(self, tile_bgr, pw, ph, tx, ty, pattern="grid", grout_w=0, grout_col=(200, 200, 200)):
        """Builds a seamless fronto-parallel tiled surface without artificial lines unless requested."""
        surface = np.zeros((ph, pw, 3), dtype=np.uint8)
        cell_w = max(8, pw // tx)
        cell_h = max(8, ph // ty)
        cell = cv2.resize(tile_bgr, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

        for r in range(ty + 1):
            for c in range(tx + 1):
                ox = 0
                if pattern == "brick" and (r % 2):
                    ox = cell_w // 2
                x1, y1 = c * cell_w + ox, r * cell_h
                x2, y2 = min(x1 + cell_w, pw), min(y1 + cell_h, ph)
                if x1 < pw and y1 < ph and x2 > x1 and y2 > y1:
                    patch = cell[:y2 - y1, :x2 - x1]
                    surface[y1:y2, x1:x2] = patch

        # Apply grout lines
        if grout_w > 0:
            gw = max(1, int(grout_w))
            for r in range(ty + 1):
                y = min(ph - 1, r * cell_h)
                surface[max(0, y - gw // 2):min(ph, y + gw // 2 + 1), :] = grout_col
            for c in range(tx + 1):
                x = min(pw - 1, c * cell_w)
                surface[:, max(0, x - gw // 2):min(pw, x + gw // 2 + 1)] = grout_col

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
