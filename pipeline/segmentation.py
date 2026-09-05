"""
Stage 1: Semantic Prior Engine (SegFormer)
"""

import cv2
import numpy as np
from PIL import Image

class SemanticSegmenter:
    def __init__(self, model=None, processor=None, device="cpu"):
        self.model = model
        self.processor = processor
        self.device = device

    def infer_semantic_map(self, image_bgr):
        """
        Runs SegFormer on the input image.
        Returns:
          - seg_map: (H, W) int32 map of ADE20K semantic IDs
          - rough_floor: (H, W) boolean mask of initial floor candidates (ID 3 = floor, 28 = rug)
        """
        H, W = image_bgr.shape[:2]
        if self.model is None or self.processor is None:
            # Fallback mock semantic map
            rough_floor = np.zeros((H, W), dtype=bool)
            rough_floor[int(H * 0.55):, :] = True
            seg_map = np.zeros((H, W), dtype=np.int32)
            seg_map[rough_floor] = 3
            return seg_map, rough_floor

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        inf_img = pil_img.resize((1024, 768))

        inputs = self.processor(images=inf_img, return_tensors="pt")
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with __import__("torch").inference_mode():
            outputs = self.model(**inputs)

        seg_small = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(768, 1024)]
        )[0].cpu().numpy()

        seg_map = cv2.resize(seg_small.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
        rough_floor = (seg_map == 3) | (seg_map == 28)
        return seg_map, rough_floor

    def visualize_mask(self, image_bgr, mask_bool):
        """Creates a green overlay visualization of a mask on the room image."""
        vis = image_bgr.copy()
        vis[mask_bool] = (
            vis[mask_bool].astype(np.float32) * 0.4 + np.array([0, 255, 0], dtype=np.float32) * 0.6
        ).astype(np.uint8)
        return vis
