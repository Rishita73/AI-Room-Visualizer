import os
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

def apply_perspective_tiling(original_image, floor_mask, tile_image, horizon_pct=53, vpx_pct=50, scale=0.8, rotation=0):
    """
    Applies perspective correct tiling to the floor region of the image.
    """
    # Convert inputs to numpy arrays
    room = np.array(original_image)
    tile = np.array(tile_image)
    
    H, W = floor_mask.shape[:2]
    th, tw = tile.shape[:2]
    
    # 1. Find indices of floor pixels
    ys, xs = np.where(floor_mask > 0)
    
    # 2. Convert horizon and vanishing point percentages to pixels
    horizon = int(H * (horizon_pct / 100.0))
    vpx = int(W * (vpx_pct / 100.0))
    
    # 3. Filter out points at or above the horizon to prevent division by zero/negative depth
    valid = ys > horizon
    ys_v = ys[valid]
    xs_v = xs[valid]
    dy_v = ys_v - horizon
    
    # 4. Map each floor pixel to 3D perspective space (Z coordinates)
    # The depth Z is proportional to 1 / dy
    # X coordinate on ground plane is proportional to (x - vpx) / dy
    final_scale = scale * 150.0  # multiplier to get reasonable tile sizing
    z = final_scale / dy_v
    px = (xs_v - vpx) * z
    py = z
    
    # 5. Apply rotation
    rad = np.radians(rotation)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    rx = px * cos_a - py * sin_a
    ry = px * sin_a + py * cos_a
    
    # 6. Wrap to tile width and height (tessellation)
    tx = np.mod(np.floor(rx), tw).astype(int)
    ty = np.mod(np.floor(ry), th).astype(int)
    
    # 7. Sample tile colors and construct the perspective-tiled floor image
    tiled_floor = np.zeros_like(room)
    tiled_floor[ys_v, xs_v] = tile[ty, tx]
    
    # 8. Blending: Multiply tile texture with original room image to preserve lighting and shadows
    room_float = room.astype(float)
    # Shading factor from the original room image
    shading = room_float / 255.0
    blended_floor = (tiled_floor.astype(float) * shading).astype(np.uint8)
    
    # 9. Composite the floor back onto the original room image
    result = room.copy()
    # Mask indices where valid points exist
    mask_indices = (ys_v, xs_v)
    result[mask_indices] = blended_floor[mask_indices]
    
    return result

def main():
    print("Loading test room image...")
    room_img = Image.open("assets/living_room_before.png").convert("RGB")
    tile_img = Image.open("assets/tile_1.png").convert("RGB")
    
    print("Loading SegFormer model for floor segmentation...")
    processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    model.eval()
    
    print("Running floor segmentation model...")
    inference_img = room_img.resize((1024, 768))
    inputs = processor(images=inference_img, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    segmentation = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[inference_img.size[::-1]]
    )[0]
    
    # ADE20K floor label is 3
    floor_mask_1024 = (segmentation == 3).cpu().numpy().astype(np.uint8) * 255
    
    # Resize mask to original image size
    floor_mask = cv2.resize(floor_mask_1024, room_img.size, interpolation=cv2.INTER_NEAREST)
    
    print("Applying perspective correct tiling...")
    # Default values for living_room_before: horizon at 53%, vanishing point at 50%
    result_img_np = apply_perspective_tiling(
        original_image=room_img,
        floor_mask=floor_mask,
        tile_image=tile_img,
        horizon_pct=53,
        vpx_pct=50,
        scale=0.8,
        rotation=0
    )
    
    print("Saving test output image...")
    result_img = Image.fromarray(result_img_np)
    result_img.save("assets/test_warped_output.png")
    print("Done! Warped image saved to assets/test_warped_output.png")

if __name__ == "__main__":
    main()
