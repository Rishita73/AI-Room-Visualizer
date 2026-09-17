"""
Stage 2 & Stage 5: Monocular 3D Depth Engine & Ground Plane RANSAC Fitting
"""

import cv2
import numpy as np
from PIL import Image
from sklearn.linear_model import RANSACRegressor, LinearRegression

class DepthGeometryEngine:
    def __init__(self, model=None, processor=None, device="cpu"):
        self.model = model
        self.processor = processor
        self.device = device

    def infer_depth(self, image_bgr):
        """
        Runs Depth-Anything-V2 to extract a metric-relative depth map.
        Returns:
          - depth_map: (H, W) float32 array
        """
        H, W = image_bgr.shape[:2]
        if self.model is None or self.processor is None:
            # Fallback linear gradient depth map (near=1.0, far=5.0)
            yy, xx = np.mgrid[0:H, 0:W]
            return (1.0 + 4.0 * (1.0 - yy / max(1, H - 1))).astype(np.float32)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        inf_img = pil_img.resize((1024, 768))

        inputs = self.processor(images=inf_img, return_tensors="pt")
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with __import__("torch").inference_mode():
            outputs = self.model(**inputs)

        pred_depth = outputs.predicted_depth.squeeze().cpu().numpy()
        depth_map = cv2.resize(pred_depth.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        return depth_map

    def fit_ground_plane(self, rough_floor_bool, depth_map, W, H):
        """
        Fits a 3D ground plane Z = A*X + B*Y + C to the floor points via RANSAC.
        Returns dictionary with plane parameters and basis vectors.
        """
        ys, xs = np.where(rough_floor_bool)
        if len(xs) < 100:
            return None

        f = float(W)
        cx, cy = float(W) / 2.0, float(H) / 2.0
        zs = depth_map[ys, xs].astype(np.float32)

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

        try:
            ransac = RANSACRegressor(
                estimator=LinearRegression(),
                residual_threshold=0.03,
                max_trials=100,
                min_samples=3,
                random_state=42
            )
            ransac.fit(sample_pts[:, :2], sample_pts[:, 2])
            inlier_mask = ransac.inlier_mask_
            if np.count_nonzero(inlier_mask) < 30:
                return None

            A, B = ransac.estimator_.coef_
            C = ransac.estimator_.intercept_
            inlier_pts = sample_pts[inlier_mask]

            # Plane normal: (A, B, -1) normalized
            n = np.array([A, B, -1.0], dtype=np.float64)
            n = n / (np.linalg.norm(n) + 1e-8)

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

            raw_corners = []
            for pt in [P1, P2, P3, P4]:
                z_s = max(float(pt[2]), 1e-3)
                x_i = f * pt[0] / z_s + cx
                y_i = f * pt[1] / z_s + cy
                raw_corners.append([float(x_i), float(y_i)])

            # Sort corners geometrically into [far_left, far_right, near_right, near_left]
            sorted_by_y = sorted(raw_corners, key=lambda p: p[1])
            top_two = sorted(sorted_by_y[:2], key=lambda p: p[0])
            bot_two = sorted(sorted_by_y[2:], key=lambda p: p[0])
            
            p_far_left = top_two[0]
            p_far_right = top_two[1]
            p_near_left = bot_two[0]
            p_near_right = bot_two[1]

            # Enforce physical perspective checks:
            # 1. Near y must be significantly lower in image than far y
            # 2. Near width must flare outward or match far width (not narrow down)
            # 3. Minimum vertical span
            span_y = min(p_near_left[1], p_near_right[1]) - max(p_far_left[1], p_far_right[1])
            near_w = p_near_right[0] - p_near_left[0]
            far_w = p_far_right[0] - p_far_left[0]

            corners_2d = None
            if span_y > H * 0.15 and near_w > 0 and far_w > 0 and near_w >= far_w * 0.95:
                corners_2d = [p_far_left, p_far_right, p_near_right, p_near_left]

            return {
                "A": A, "B": B, "C": C,
                "normal": n,
                "u": u, "v": v,
                "origin": origin,
                "corners_3d": [P1, P2, P3, P4],
                "corners_2d": corners_2d,
                "inlier_ratio": float(np.count_nonzero(inlier_mask)) / len(sample_pts)
            }
        except Exception:
            return None

    def compute_dense_ground_support(self, plane_info, depth_map, W, H, max_dist=0.08):
        """
        Calculates the per-pixel 3D geometric ground support mask.
        A pixel belongs to the ground plane if its depth Z(x, y) is consistent
        with the parametric 3D ground plane equation:
          Z_plane(x, y) = C / (1 - A*(x - cx)/f - B*(y - cy)/f)
        """
        if plane_info is None:
            return np.zeros((H, W), dtype=bool)

        A, B, C = plane_info["A"], plane_info["B"], plane_info["C"]
        f = float(W)
        cx, cy = float(W) / 2.0, float(H) / 2.0

        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        denom = 1.0 - A * (xx.astype(np.float32) - cx) / f - B * (yy.astype(np.float32) - cy) / f
        denom = np.where(np.abs(denom) < 1e-4, 1e-4, denom)
        expected_Z = C / denom

        actual_Z = depth_map.astype(np.float32)
        diff = np.abs(actual_Z - expected_Z)

        support = (diff < max_dist) & (expected_Z > 0) & (yy > int(H * 0.35))
        return support

    def visualize_depth(self, depth_map):
        """Visualizes depth map as an inferno color map."""
        d_min, d_max = depth_map.min(), depth_map.max()
        norm_d = ((depth_map - d_min) / (d_max - d_min + 1e-6) * 255.0).astype(np.uint8)
        return cv2.applyColorMap(norm_d, cv2.COLORMAP_INFERNO)

    def visualize_plane(self, image_bgr, plane_info, rough_floor):
        """Visualizes estimated 3D floor plane inliers and bounding quad."""
        vis = image_bgr.copy()
        if plane_info is not None and "corners_2d" in plane_info:
            quad = np.array(plane_info["corners_2d"], dtype=np.int32)
            cv2.polylines(vis, [quad], isClosed=True, color=(255, 0, 0), thickness=3)
            for idx, pt in enumerate(quad):
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 8, (0, 255, 255), -1)
                cv2.putText(vis, f"P{idx+1}", (int(pt[0]) + 10, int(pt[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return vis
