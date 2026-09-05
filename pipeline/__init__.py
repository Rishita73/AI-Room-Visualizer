"""
Room Visualizer Pipeline Package
A hybrid multi-stage architectural visualizer combining semantic priors,
monocular 3D depth geometry, surface texture continuity, and photometric relighting
for both Floors and Walls.
"""

import cv2
import numpy as np
from .segmentation import SemanticSegmenter
from .depth_engine import DepthGeometryEngine
from .surface_analyzer import SurfaceTextureAnalyzer
from .occlusion_detector import OcclusionDetector
from .mask_refiner import MaskRefiner
from .perspective_engine import PerspectiveEngine
from .lighting_engine import LightingEngine
from .compositor import VisualizerCompositor
from .wall_engine import WallEngine

class RoomVisualizerPipeline:
    """
    Unified 10-stage architectural visualizer orchestrator for floors and walls.
    """
    def __init__(self, seg_model=None, seg_processor=None, depth_model=None, depth_processor=None, device="cpu"):
        self.device = device
        self.segmenter = SemanticSegmenter(model=seg_model, processor=seg_processor, device=device)
        self.depth_engine = DepthGeometryEngine(model=depth_model, processor=depth_processor, device=device)
        self.surface_analyzer = SurfaceTextureAnalyzer()
        self.occlusion_detector = OcclusionDetector()
        self.mask_refiner = MaskRefiner()
        self.perspective_engine = PerspectiveEngine()
        self.lighting_engine = LightingEngine()
        self.compositor = VisualizerCompositor()
        self.wall_engine = WallEngine()

    def process_room(self, image_bgr, tile_bgr=None, tile_config=None, generate_debug=False):
        """
        Executes the complete multi-stage pipeline on a room image.
        Returns:
          - 'clean_mask': Final refined boolean floor mask (H, W)
          - 'clean_wall_mask': Final refined boolean wall mask (H, W)
          - 'floor_quad': 4-corner perspective ground quad
          - 'wall_quads': List of 4-corner perspective wall quads
          - 'wall_planes': Partitioned physical wall planes
          - 'room_metrics': Estimated dimensions, sqft, and tile count
          - 'depth_map': Monocular depth map
          - 'result_bgr': Composite result if tile_bgr is provided
          - 'debug_stages': Dictionary of visual debug stage images
        """
        H, W = image_bgr.shape[:2]
        debug_stages = {}

        # Stage 1: Semantic Priors (SegFormer)
        seg_map, rough_floor = self.segmenter.infer_semantic_map(image_bgr)
        if generate_debug:
            debug_stages["1_original"] = image_bgr.copy()
            debug_stages["2_raw_segformer"] = self.segmenter.visualize_mask(image_bgr, rough_floor)

        # Stage 2: Monocular 3D Depth
        depth_map = self.depth_engine.infer_depth(image_bgr)
        if generate_debug:
            debug_stages["4_depth_map"] = self.depth_engine.visualize_depth(depth_map)

        # Stage 3: Occlusion & Furniture Geometry Shielding (Strict Barrier)
        obstacle_mask = self.occlusion_detector.detect_obstacles(seg_map, depth_map, W, H)
        if generate_debug:
            debug_stages["6_occlusion_mask"] = self.occlusion_detector.visualize_obstacles(image_bgr, obstacle_mask)

        # Stage 4: Surface Continuity & Texture Analysis (Bounded by Obstacle Shield)
        surface_mask = self.surface_analyzer.analyze_floor_surface(image_bgr, rough_floor, depth_map, obstacle_mask)
        if generate_debug:
            debug_stages["3_surface_analysis"] = self.surface_analyzer.visualize_surface(image_bgr, surface_mask)

        # Stage 5: 3D Ground Plane Estimation
        ground_plane_info = self.depth_engine.fit_ground_plane(rough_floor, depth_map, W, H)
        if generate_debug:
            debug_stages["5_floor_plane"] = self.depth_engine.visualize_plane(image_bgr, ground_plane_info, rough_floor)

        # Stage 6: Multi-Signal Mask Fusion & Edge Refinement (Floor)
        final_floor_mask = self.mask_refiner.refine_mask(
            rough_floor=rough_floor,
            surface_mask=surface_mask,
            obstacle_mask=obstacle_mask,
            ground_plane_info=ground_plane_info,
            image_bgr=image_bgr,
            depth_map=depth_map,
            depth_engine=self.depth_engine
        )
        if generate_debug:
            debug_stages["7_refined_mask"] = self.mask_refiner.visualize_refined(image_bgr, final_floor_mask)

        # Stage 7: Perspective Ground Quad
        floor_quad = self.perspective_engine.compute_ground_quad(
            final_mask=final_floor_mask,
            ground_plane_info=ground_plane_info,
            W=W, H=H
        )

        # Stage 8: 3D Multi-Plane Wall Segmentation & Geometry
        clean_wall_mask = self.wall_engine.extract_wall_mask(seg_map, floor_mask=final_floor_mask, depth_map=depth_map)
        wall_planes = self.wall_engine.partition_wall_planes(clean_wall_mask, depth_map, W, H)
        wall_quads = [p["quad"] for p in wall_planes] if wall_planes else [self.wall_engine.compute_wall_quads(clean_wall_mask, floor_quad, W=W, H=H)]

        # Room Dimension & Material Estimation
        room_metrics = self.wall_engine.estimate_room_metrics(floor_quad, clean_wall_mask, W, H)

        result_bgr = None
        if tile_bgr is not None:
            cfg = tile_config or {}
            target_surface = cfg.get("target_surface", "floor") # 'floor' or 'wall'
            pattern = cfg.get("pattern", "grid")
            grout_w = cfg.get("grout_width", 0)
            grout_col = cfg.get("grout_color", (200, 200, 200))
            tile_scale = cfg.get("scale", 1.0)
            tw_mm = cfg.get("tile_width_mm", 600.0)
            th_mm = cfg.get("tile_height_mm", 600.0)
            ppm = cfg.get("pixels_per_meter", 200.0)
            shadow_strength = cfg.get("shadow_strength", 0.55)
            finish = cfg.get("finish", "satin")

            if target_surface == "wall" and np.any(clean_wall_mask):
                # Apply wall material across wall planes
                comp_wall = image_bgr.copy()
                for plane in (wall_planes or [{"mask": clean_wall_mask, "quad": wall_quads[0]}]):
                    p_mask = plane["mask"]
                    p_quad = plane["quad"]
                    tx, ty, pw, ph = self.perspective_engine.plan_tiling(
                        floor_quad=p_quad, W=W, H=H, ppm=ppm,
                        tile_w_mm=tw_mm, tile_h_mm=th_mm, tile_scale=tile_scale
                    )
                    flat_surf = self.perspective_engine.build_surface(
                        tile_bgr=tile_bgr, pw=pw, ph=ph, tx=tx, ty=ty,
                        pattern=pattern, grout_w=grout_w, grout_col=grout_col
                    )
                    comp_wall = self.wall_engine.warp_wall_material(
                        room_bgr=comp_wall,
                        tile_surface=flat_surf,
                        dest_quad=p_quad,
                        mask_bool=p_mask,
                        shadow_strength=shadow_strength,
                        finish=finish
                    )
                result_bgr = comp_wall
            else:
                # Apply floor material
                tx, ty, pw, ph = self.perspective_engine.plan_tiling(
                    floor_quad=floor_quad, W=W, H=H, ppm=ppm,
                    tile_w_mm=tw_mm, tile_h_mm=th_mm, tile_scale=tile_scale
                )
                flat_surface = self.perspective_engine.build_surface(
                    tile_bgr=tile_bgr, pw=pw, ph=ph, tx=tx, ty=ty,
                    pattern=pattern, grout_w=grout_w, grout_col=grout_col
                )
                warped_material = self.perspective_engine.warp_to_quad(flat_surface, floor_quad, W, H)
                if generate_debug:
                    debug_stages["8_warped_flooring"] = warped_material.copy()

                lighting_map = self.lighting_engine.extract_lighting_map(image_bgr, final_floor_mask)
                result_bgr = self.compositor.composite(
                    room_bgr=image_bgr,
                    warped_material=warped_material,
                    floor_mask=final_floor_mask,
                    lighting_map=lighting_map,
                    shadow_strength=shadow_strength,
                    finish=finish
                )
                if generate_debug:
                    debug_stages["9_final_composite"] = result_bgr.copy()

        return {
            "seg_map": seg_map,
            "clean_mask": final_floor_mask,
            "clean_wall_mask": clean_wall_mask,
            "floor_quad": floor_quad,
            "wall_quads": wall_quads,
            "wall_planes": wall_planes,
            "room_metrics": room_metrics,
            "depth_map": depth_map,
            "result_bgr": result_bgr,
            "debug_stages": debug_stages
        }
