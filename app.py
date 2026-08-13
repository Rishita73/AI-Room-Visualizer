import os
import time
import io
import base64
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFont
import gradio as gr

# Try loading PyTorch and SegFormer, otherwise fall back to Simulation Mode
MODEL_AVAILABLE = False
processor = None
model = None

try:
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    
    print("Loading SegFormer model for Gradio (nvidia/segformer-b2-finetuned-ade-512-512)...")
    device = "cpu"
    try:
        processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    except Exception as fetch_err:
        print(f"Hugging Face fetch failed ({fetch_err}). Retrying with local cached files...")
        processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512", local_files_only=True)
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512", local_files_only=True)
        
    model.eval()
    MODEL_AVAILABLE = True
    print("SegFormer Model Loaded Successfully on CPU!")
except Exception as e:
    print(f"Warning: SegFormer not loaded, running Gradio in simulation mode. Error: {e}")
    MODEL_AVAILABLE = False

# Scanned room templates database injected dynamically
ROOMS_DATABASE = [
    {
        "id": "room-1",
        "name": "Living Room 1",
        "sub": "Curated Living Room space template",
        "img": "images_templates/images_living_room/floor-tiles-ideas-for-living-room.jpg",
        "cardImg": "images_templates/images_living_room/floor-tiles-ideas-for-living-room.jpg",
        "baseScale": 1.0,
        "type": "living_room"
    },
    {
        "id": "room-2",
        "name": "Living Room 2",
        "sub": "Curated Living Room space template",
        "img": "images_templates/images_living_room/images (1).jpg",
        "cardImg": "images_templates/images_living_room/images (1).jpg",
        "baseScale": 1.0,
        "type": "living_room"
    },
    {
        "id": "room-3",
        "name": "Living Room 3",
        "sub": "Curated Living Room space template",
        "img": "images_templates/images_living_room/images (2).jpg",
        "cardImg": "images_templates/images_living_room/images (2).jpg",
        "baseScale": 1.0,
        "type": "living_room"
    },
    {
        "id": "room-4",
        "name": "Living Room 4",
        "sub": "Curated Living Room space template",
        "img": "images_templates/images_living_room/images (3).jpg",
        "cardImg": "images_templates/images_living_room/images (3).jpg",
        "baseScale": 1.0,
        "type": "living_room"
    },
    {
        "id": "room-5",
        "name": "Living Room 5",
        "sub": "Curated Living Room space template",
        "img": "images_templates/images_living_room/images (4).jpg",
        "cardImg": "images_templates/images_living_room/images (4).jpg",
        "baseScale": 1.0,
        "type": "living_room"
    },
    {
        "id": "room-6",
        "name": "Living Room 6",
        "sub": "Curated Living Room space template",
        "img": "images_templates/images_living_room/images (5).jpg",
        "cardImg": "images_templates/images_living_room/images (5).jpg",
        "baseScale": 1.0,
        "type": "living_room"
    },
    {
        "id": "room-7",
        "name": "Living Room 7",
        "sub": "Curated Living Room space template",
        "img": "images_templates/images_living_room/images.jpg",
        "cardImg": "images_templates/images_living_room/images.jpg",
        "baseScale": 1.0,
        "type": "living_room"
    },
    {
        "id": "room-8",
        "name": "Living Room 8",
        "sub": "Curated Living Room space template",
        "img": "images_templates/images_living_room/WhatsApp-Image-2026-05-05-at-7.02.40-PM.jpg",
        "cardImg": "images_templates/images_living_room/WhatsApp-Image-2026-05-05-at-7.02.40-PM.jpg",
        "baseScale": 1.0,
        "type": "living_room"
    },
    {
        "id": "room-9",
        "name": "Bedroom 1",
        "sub": "Curated Bedroom space template",
        "img": "images_templates/images_bedroom/download.jpg",
        "cardImg": "images_templates/images_bedroom/download.jpg",
        "baseScale": 1.0,
        "type": "bedroom"
    },
    {
        "id": "room-10",
        "name": "Bedroom 2",
        "sub": "Curated Bedroom space template",
        "img": "images_templates/images_bedroom/images (1).jpg",
        "cardImg": "images_templates/images_bedroom/images (1).jpg",
        "baseScale": 1.0,
        "type": "bedroom"
    },
    {
        "id": "room-11",
        "name": "Bedroom 3",
        "sub": "Curated Bedroom space template",
        "img": "images_templates/images_bedroom/images (2).jpg",
        "cardImg": "images_templates/images_bedroom/images (2).jpg",
        "baseScale": 1.0,
        "type": "bedroom"
    },
    {
        "id": "room-12",
        "name": "Bedroom 4",
        "sub": "Curated Bedroom space template",
        "img": "images_templates/images_bedroom/images (3).jpg",
        "cardImg": "images_templates/images_bedroom/images (3).jpg",
        "baseScale": 1.0,
        "type": "bedroom"
    },
    {
        "id": "room-13",
        "name": "Bedroom 5",
        "sub": "Curated Bedroom space template",
        "img": "images_templates/images_bedroom/images (4).jpg",
        "cardImg": "images_templates/images_bedroom/images (4).jpg",
        "baseScale": 1.0,
        "type": "bedroom"
    },
    {
        "id": "room-14",
        "name": "Bedroom 6",
        "sub": "Curated Bedroom space template",
        "img": "images_templates/images_bedroom/images (5).jpg",
        "cardImg": "images_templates/images_bedroom/images (5).jpg",
        "baseScale": 1.0,
        "type": "bedroom"
    },
    {
        "id": "room-15",
        "name": "Bedroom 7",
        "sub": "Curated Bedroom space template",
        "img": "images_templates/images_bedroom/images.jpg",
        "cardImg": "images_templates/images_bedroom/images.jpg",
        "baseScale": 1.0,
        "type": "bedroom"
    },
    {
        "id": "room-16",
        "name": "Bedroom 8",
        "sub": "Curated Bedroom space template",
        "img": "images_templates/images_bedroom/tiles-for-bedroom.jpg",
        "cardImg": "images_templates/images_bedroom/tiles-for-bedroom.jpg",
        "baseScale": 1.0,
        "type": "bedroom"
    },
    {
        "id": "room-17",
        "name": "Kitchen 1",
        "sub": "Curated Kitchen space template",
        "img": "images_templates/images_kitchen/images (1).jpg",
        "cardImg": "images_templates/images_kitchen/images (1).jpg",
        "baseScale": 1.0,
        "type": "kitchen"
    },
    {
        "id": "room-18",
        "name": "Kitchen 2",
        "sub": "Curated Kitchen space template",
        "img": "images_templates/images_kitchen/images (2).jpg",
        "cardImg": "images_templates/images_kitchen/images (2).jpg",
        "baseScale": 1.0,
        "type": "kitchen"
    },
    {
        "id": "room-19",
        "name": "Kitchen 3",
        "sub": "Curated Kitchen space template",
        "img": "images_templates/images_kitchen/images (3).jpg",
        "cardImg": "images_templates/images_kitchen/images (3).jpg",
        "baseScale": 1.0,
        "type": "kitchen"
    },
    {
        "id": "room-20",
        "name": "Kitchen 4",
        "sub": "Curated Kitchen space template",
        "img": "images_templates/images_kitchen/images (4).jpg",
        "cardImg": "images_templates/images_kitchen/images (4).jpg",
        "baseScale": 1.0,
        "type": "kitchen"
    },
    {
        "id": "room-21",
        "name": "Kitchen 5",
        "sub": "Curated Kitchen space template",
        "img": "images_templates/images_kitchen/images (5).jpg",
        "cardImg": "images_templates/images_kitchen/images (5).jpg",
        "baseScale": 1.0,
        "type": "kitchen"
    },
    {
        "id": "room-22",
        "name": "Kitchen 6",
        "sub": "Curated Kitchen space template",
        "img": "images_templates/images_kitchen/images (6).jpg",
        "cardImg": "images_templates/images_kitchen/images (6).jpg",
        "baseScale": 1.0,
        "type": "kitchen"
    },
    {
        "id": "room-23",
        "name": "Kitchen 7",
        "sub": "Curated Kitchen space template",
        "img": "images_templates/images_kitchen/images (7).jpg",
        "cardImg": "images_templates/images_kitchen/images (7).jpg",
        "baseScale": 1.0,
        "type": "kitchen"
    },
    {
        "id": "room-24",
        "name": "Kitchen 8",
        "sub": "Curated Kitchen space template",
        "img": "images_templates/images_kitchen/images (8).jpg",
        "cardImg": "images_templates/images_kitchen/images (8).jpg",
        "baseScale": 1.0,
        "type": "kitchen"
    },
    {
        "id": "room-25",
        "name": "Kitchen 9",
        "sub": "Curated Kitchen space template",
        "img": "images_templates/images_kitchen/images.jpg",
        "cardImg": "images_templates/images_kitchen/images.jpg",
        "baseScale": 1.0,
        "type": "kitchen"
    },
    {
        "id": "room-26",
        "name": "Bathroom 1",
        "sub": "Curated Bathroom space template",
        "img": "images_templates/images_bathroom/bathroom-floortiles-homesquare.webp",
        "cardImg": "images_templates/images_bathroom/bathroom-floortiles-homesquare.webp",
        "baseScale": 1.0,
        "type": "bathroom"
    },
    {
        "id": "room-27",
        "name": "Bathroom 2",
        "sub": "Curated Bathroom space template",
        "img": "images_templates/images_bathroom/images (1).jpg",
        "cardImg": "images_templates/images_bathroom/images (1).jpg",
        "baseScale": 1.0,
        "type": "bathroom"
    },
    {
        "id": "room-28",
        "name": "Bathroom 3",
        "sub": "Curated Bathroom space template",
        "img": "images_templates/images_bathroom/images (2).jpg",
        "cardImg": "images_templates/images_bathroom/images (2).jpg",
        "baseScale": 1.0,
        "type": "bathroom"
    },
    {
        "id": "room-29",
        "name": "Bathroom 4",
        "sub": "Curated Bathroom space template",
        "img": "images_templates/images_bathroom/images (3).jpg",
        "cardImg": "images_templates/images_bathroom/images (3).jpg",
        "baseScale": 1.0,
        "type": "bathroom"
    },
    {
        "id": "room-30",
        "name": "Bathroom 5",
        "sub": "Curated Bathroom space template",
        "img": "images_templates/images_bathroom/images (4).jpg",
        "cardImg": "images_templates/images_bathroom/images (4).jpg",
        "baseScale": 1.0,
        "type": "bathroom"
    },
    {
        "id": "room-31",
        "name": "Bathroom 6",
        "sub": "Curated Bathroom space template",
        "img": "images_templates/images_bathroom/images (5).jpg",
        "cardImg": "images_templates/images_bathroom/images (5).jpg",
        "baseScale": 1.0,
        "type": "bathroom"
    },
    {
        "id": "room-32",
        "name": "Bathroom 7",
        "sub": "Curated Bathroom space template",
        "img": "images_templates/images_bathroom/images (6).jpg",
        "cardImg": "images_templates/images_bathroom/images (6).jpg",
        "baseScale": 1.0,
        "type": "bathroom"
    },
    {
        "id": "room-33",
        "name": "Bathroom 8",
        "sub": "Curated Bathroom space template",
        "img": "images_templates/images_bathroom/images (7).jpg",
        "cardImg": "images_templates/images_bathroom/images (7).jpg",
        "baseScale": 1.0,
        "type": "bathroom"
    },
    {
        "id": "room-34",
        "name": "Bathroom 9",
        "sub": "Curated Bathroom space template",
        "img": "images_templates/images_bathroom/images (8).jpg",
        "cardImg": "images_templates/images_bathroom/images (8).jpg",
        "baseScale": 1.0,
        "type": "bathroom"
    },
    {
        "id": "room-35",
        "name": "Bathroom 10",
        "sub": "Curated Bathroom space template",
        "img": "images_templates/images_bathroom/images (9).jpg",
        "cardImg": "images_templates/images_bathroom/images (9).jpg",
        "baseScale": 1.0,
        "type": "bathroom"
    },
    {
        "id": "room-36",
        "name": "Bathroom 11",
        "sub": "Curated Bathroom space template",
        "img": "images_templates/images_bathroom/images.jpg",
        "cardImg": "images_templates/images_bathroom/images.jpg",
        "baseScale": 1.0,
        "type": "bathroom"
    }
]

# Preset tiles definitions
PRESET_TILES_DATABASE = [
    {"name": "White Statuario Marble (₹75/sq.ft)", "img": "assets/tile_1.png", "price": 75},
    {"name": "Grey Veined Marble (₹65/sq.ft)", "img": "assets/tile_2.png", "price": 65},
    {"name": "Classic Beige Marble (₹62/sq.ft)", "img": "assets/tile_3.png", "price": 62},
    {"name": "Oak Brown Wood Planks (₹58/sq.ft)", "img": "assets/tile_4.png", "price": 58},
    {"name": "Walnut Wood Planks (₹60/sq.ft)", "img": "assets/tile_5.png", "price": 60},
    {"name": "Light Grey Terrazzo (₹52/sq.ft)", "img": "assets/tile_6.png", "price": 52},
    {"name": "Dark Charcoal Granite (₹80/sq.ft)", "img": "assets/tile_7.png", "price": 80}
]


def clean_and_extract_mask(mask, width, height):
    """
    Cleans binary mask using morphological operations and resizes it.
    """
    kernel = np.ones((7, 7), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_resized = cv2.resize(mask_cleaned, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask_resized


def get_floor_mask(room_image):
    """
    Detects floor area using SegFormer (Label index 3).
    Falls back to trapezoid simulation if model is not loaded.
    """
    width, height = room_image.size
    
    if MODEL_AVAILABLE:
        try:
            inference_img = room_image.resize((1024, 768))
            inputs = processor(images=inference_img, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            segmentation = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[inference_img.size[::-1]]
            )[0]
            
            floor_mask = (segmentation == 3).cpu().numpy().astype(np.uint8) * 255
            mask_cleaned = clean_and_extract_mask(floor_mask, width, height)
            return mask_cleaned
            
        except Exception as err:
            print(f"Error in model inference: {err}. Falling back to simulation.")
            
    # Simulation Fallback (Trapezoid floor outline)
    mask_mock = np.zeros((height, width), dtype=np.uint8)
    pts = np.array([
        [0, int(height * 0.95)],
        [int(width * 0.35), int(height * 0.58)],
        [int(width * 0.65), int(height * 0.58)],
        [width, int(height * 0.95)]
    ], dtype=np.int32)
    cv2.fillPoly(mask_mock, [pts], 255)
    return clean_and_extract_mask(mask_mock, width, height)


def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


def generate_tiled_texture(tile_image, W, H, scale, rotation, pattern, grout_width, grout_color_hex, brightness):
    """
    Python-based high-fidelity tiling renderer.
    """
    tile_w = int(tile_image.width * scale)
    tile_h = int(tile_image.height * scale)
    
    if tile_w < 5 or tile_h < 5:
        tile_w, tile_h = 5, 5
        
    tile_scaled = tile_image.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
    
    if grout_width > 0:
        grout_color = hex_to_rgb(grout_color_hex)
        tile_scaled = ImageOps.expand(tile_scaled, border=grout_width, fill=grout_color)
        tile_w = tile_scaled.width
        tile_h = tile_scaled.height

    if rotation != 0:
        tile_rotated = tile_scaled.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
    else:
        tile_rotated = tile_scaled

    max_range = int(max(W, H) * 2.2)
    tiled_canvas = Image.new("RGB", (max_range, max_range))
    
    tr_w, tr_h = tile_rotated.width, tile_rotated.height
    
    for y in range(0, max_range, tr_h):
        row_offset = 0
        if pattern == "Brick Offset" and (y // tr_h) % 2 != 0:
            row_offset = tr_w // 2
            
        for x in range(-tr_w, max_range, tr_w):
            tiled_canvas.paste(tile_rotated, (x + row_offset, y))
            
    left = (max_range - W) // 2
    top = (max_range - H) // 2
    tiled_crop = tiled_canvas.crop((left, top, left + W, top + H))
    
    if brightness != 100:
        enhancer = ImageEnhance.Brightness(tiled_crop)
        tiled_crop = enhancer.enhance(brightness / 100.0)
        
    return tiled_crop


def resolve_file_path(file_input):
    if file_input is None:
        return None
    # If list of files (e.g. from multiple file uploads or Gradio lists)
    if isinstance(file_input, list):
        if len(file_input) == 0:
            return None
        file_input = file_input[0]
    
    # If dict (e.g. from older Gradio or serialization)
    if isinstance(file_input, dict):
        return file_input.get("name", None)
        
    # If file object or tempfile wrapper
    if hasattr(file_input, "name"):
        return file_input.name
        
    # If string path
    if isinstance(file_input, str):
        return file_input
        
    return None


def process_visualization(
    selected_room_path,
    custom_room_file,
    selected_tile_path,
    custom_tile_file,
    scale,
    rotation,
    brightness,
    pattern,
    grout_width,
    grout_color_hex,
    finish,
    calc_length,
    calc_width,
    horizon_pct,
    vpx_pct
):
    print(f"\n[DEBUG] process_visualization called:")
    print(f"  - selected_room_path: {selected_room_path}")
    print(f"  - custom_room_file: {custom_room_file} (type: {type(custom_room_file)})")
    print(f"  - selected_tile_path: {selected_tile_path}")
    print(f"  - custom_tile_file: {custom_tile_file} (type: {type(custom_tile_file)})")
    print(f"  - scale: {scale}, rotation: {rotation}, brightness: {brightness}")
    print(f"  - pattern: {pattern}, grout_width: {grout_width}, finish: {finish}")
    print(f"  - horizon_pct: {horizon_pct}, vpx_pct: {vpx_pct}\n")

    # 1. Resolve room image source
    custom_room_path = resolve_file_path(custom_room_file)
    if custom_room_path is not None and os.path.exists(custom_room_path):
        room_image = Image.open(custom_room_path).convert("RGB")
    else:
        room_path = resolve_file_path(selected_room_path)
        if room_path is not None and os.path.exists(room_path):
            room_image = Image.open(room_path).convert("RGB")
        else:
            room_image = Image.new("RGB", (800, 600), (200, 200, 200))

    W, H = room_image.size

    # --- QUALITY FIX: Auto-Upscale Room Image to fit nice resolution (e.g. 1200px width) ---
    target_w = 1200
    if W < target_w:
        ratio = target_w / W
        target_h = int(H * ratio)
        room_image = room_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        W, H = room_image.size

    # 2. Resolve tile image source
    tile_unit_price = 60
    custom_tile_path = resolve_file_path(custom_tile_file)
    if custom_tile_path is not None and os.path.exists(custom_tile_path):
        tile_image = Image.open(custom_tile_path).convert("RGB")
    else:
        tile_path = resolve_file_path(selected_tile_path)
        if tile_path is not None and os.path.exists(tile_path):
            tile_image = Image.open(tile_path).convert("RGB")
            # Get tile price
            for t in PRESET_TILES_DATABASE:
                if t["img"] == tile_path:
                    tile_unit_price = t["price"]
                    break
        else:
            tile_image = Image.new("RGB", (200, 200), (255, 255, 255))

    # 3. Detect / generate floor mask
    floor_mask_np = get_floor_mask(room_image)
    floor_mask_pil = Image.fromarray(floor_mask_np).convert("L")

    # 4. Generate flat tiled floor as implemented in the notebook (with rotation, offset, and grout support)
    tiled_floor = generate_tiled_texture(
        tile_image, W, H, scale * (W / 1024), rotation, pattern, grout_width, grout_color_hex, brightness
    )

    # 5. Composite blend using normalized grayscale shadow map with detail suppression
    room_np = np.array(room_image)
    
    # Extract luminance (convert to grayscale)
    gray = cv2.cvtColor(room_np, cv2.COLOR_RGB2GRAY).astype(float)
    
    # Apply Gaussian blur to suppress old grout lines, joints, and tile patterns
    # A kernel size of ~4% of image width works beautifully to wash out sharp features
    blur_kernel = int(W * 0.04) | 1  # Ensure odd number
    gray_blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    
    # Calculate shadow map based on the 95th percentile of brightness inside the blurred floor area
    floor_pixels = gray_blurred[floor_mask_np > 0]
    if len(floor_pixels) > 0:
        white_point = np.percentile(floor_pixels, 95)
        if white_point < 50.0:
            white_point = 50.0
        # Normalize: 1.0 means no shadow (fully lit), < 1.0 means shadow
        shadow_map = np.clip(gray_blurred / white_point, 0.0, 1.0)
        # Dampen the shadow map to prevent high-contrast old tile lines/features from bleeding through
        shadow_map = np.clip(0.65 * shadow_map + 0.35, 0.0, 1.0)
    else:
        shadow_map = np.ones_like(gray_blurred)
        
    # Scale each channel of the tiled floor by the shadow map
    tiled_np = np.array(tiled_floor).astype(float)
    blended_np = np.zeros_like(tiled_np)
    for c in range(3):
        blended_np[:, :, c] = tiled_np[:, :, c] * shadow_map
        
    blended_image = Image.fromarray(np.clip(blended_np, 0, 255).astype(np.uint8))

    # 6. Apply floor mask clipping
    result_image = Image.composite(blended_image, room_image, floor_mask_pil)

    # 7. Apply Glossy reflection highlight simulation
    if finish == "Glossy":
        result_image = Image.blend(result_image, Image.composite(room_image, result_image, floor_mask_pil), 0.12)

    # 8. Compute cost calculations
    area = calc_length * calc_width
    material_cost = area * tile_unit_price
    gst = material_cost * 0.18
    shipping = 1500 if area > 0 else 0
    total_cost = material_cost + gst + shipping

    calculator_report = (
        f"📊 ESTIMATED PROJECT COST BREAKDOWN:\n"
        f"----------------------------------------\n"
        f"📏 Calculated Area: {area:,} sq.ft\n"
        f"📦 Material Unit Price: ₹{tile_unit_price}/sq.ft\n"
        f"💵 Raw Material Cost: ₹{int(material_cost):,}\n"
        f"🏛️ GST Tax (18%): ₹{int(gst):,}\n"
        f"🚚 Shipping & Freight: ₹{shipping:,}\n"
        f"----------------------------------------\n"
        f"💰 TOTAL ESTIMATE: ₹{int(total_cost):,}\n\n"
        f"Note: Recommended buffer is 10% extra for cutting losses."
    )

    return result_image, calculator_report, room_image, result_image


def get_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()


def generate_client_pdf(
    selected_room_path,
    custom_room_file,
    selected_tile_path,
    custom_tile_file,
    scale,
    rotation,
    brightness,
    pattern,
    grout_width,
    grout_color_hex,
    finish,
    calc_length,
    calc_width,
    horizon_pct,
    vpx_pct
):
    # 1. Generate visualizer result
    res_img, report, room_img, res_img_comp = process_visualization(
        selected_room_path,
        custom_room_file,
        selected_tile_path,
        custom_tile_file,
        scale,
        rotation,
        brightness,
        pattern,
        grout_width,
        grout_color_hex,
        finish,
        calc_length,
        calc_width,
        horizon_pct,
        vpx_pct
    )
    
    # 2. Get base room name & swatch details
    room_name = "Custom Uploaded Room"
    if custom_room_file is None:
        room_path = resolve_file_path(selected_room_path)
        for r in ROOMS_DATABASE:
            if r["img"] == room_path:
                room_name = r["name"]
                break
                
    tile_name = "Custom Uploaded Tile Swatch"
    tile_unit_price = 60
    if custom_tile_file is None:
        tile_path = resolve_file_path(selected_tile_path)
        for t in PRESET_TILES_DATABASE:
            if t["img"] == tile_path:
                tile_name = t["name"]
                tile_unit_price = t["price"]
                break
                
    # 3. Create PDF canvas sheet (1200 x 1700 px)
    sheet = Image.new("RGB", (1200, 1700), "#0f172a")
    draw = ImageDraw.Draw(sheet)
    
    # Draw Header Block
    draw.rectangle([(0, 0), (1200, 180)], fill="#1e1b4b")
    draw.text((100, 50), "VisionRoom AI", fill="#6366f1", font=get_font(48))
    draw.text((100, 110), f"CLIENT DESIGN SPECIFICATION - {room_name.upper()}", fill="#94a3b8", font=get_font(20))
    
    # Paste Rendered Workspace (resize to fit 1000 x 760 centered)
    render_resized = res_img.resize((1000, 760), Image.Resampling.LANCZOS)
    draw.rectangle([(98, 218), (1102, 982)], outline="#6366f1", width=4)
    sheet.paste(render_resized, (100, 220))
    
    # Draw Columns
    # Left Column: Material details
    draw.text((100, 1030), "MATERIAL SPECIFICATIONS", fill="#6366f1", font=get_font(24))
    
    # Resolve tile image for swatch
    if custom_tile_file is not None:
        tile_path = resolve_file_path(custom_tile_file)
    else:
        tile_path = resolve_file_path(selected_tile_path)
        
    if tile_path and os.path.exists(tile_path):
        try:
            swatch_img = Image.open(tile_path).convert("RGB")
        except:
            swatch_img = Image.new("RGB", (200, 200), (255, 255, 255))
    else:
        swatch_img = Image.new("RGB", (200, 200), (255, 255, 255))
        
    swatch_resized = swatch_img.resize((200, 200), Image.Resampling.LANCZOS)
    draw.rectangle([(98, 1078), (302, 1282)], outline="#475569", width=2)
    sheet.paste(swatch_resized, (100, 1080))
    
    # Text info next to swatch
    draw.text((330, 1080), f"Material: {tile_name}", fill="#f8fafc", font=get_font(18))
    draw.text((330, 1115), f"Layout Pattern: {pattern}", fill="#94a3b8", font=get_font(16))
    draw.text((330, 1150), f"Surface Finish: {finish} Finish", fill="#94a3b8", font=get_font(16))
    draw.text((330, 1185), f"Grout Configuration: {grout_width}px Width ({grout_color_hex})", fill="#94a3b8", font=get_font(16))
    draw.text((330, 1220), f"Scale Modifier: {scale}x | Rotation: {rotation} deg", fill="#94a3b8", font=get_font(16))

    # Right Column: Pricing Breakdown
    draw.text((700, 1030), "PROJECT COSTING BREAKDOWN", fill="#6366f1", font=get_font(24))
    
    area = int(calc_length * calc_width)
    material_cost = int(area * tile_unit_price)
    gst = int(material_cost * 0.18)
    shipping = 1500 if area > 0 else 0
    total_cost = material_cost + gst + shipping
    
    costs = [
        ("Room Dimensions:", f"{calc_length} ft x {calc_width} ft"),
        ("Total Surface Area:", f"{area:,} sq.ft"),
        ("Material Unit Price:", f"₹{tile_unit_price}/sq.ft"),
        ("Raw Material Cost:", f"₹{material_cost:,}"),
        ("GST Tax (18%):", f"₹{gst:,}"),
        ("Shipping & Freight:", f"₹{shipping:,}"),
        ("----------------------------------", "------------------"),
        ("TOTAL PROJECT ESTIMATE:", f"₹{total_cost:,}")
    ]
    
    y_off = 1080
    for label, val in costs:
        is_bold = "TOTAL" in label
        color = "#ffffff" if is_bold else "#f8fafc" if label.startswith("Raw") or label.startswith("Total") else "#94a3b8"
        draw.text((700, y_off), label, fill=color, font=get_font(18 if is_bold else 16))
        draw.text((950, y_off), val, fill=color, font=get_font(18 if is_bold else 16))
        y_off += 30

    # Draw Footer
    draw.rectangle([(0, 1620), (1200, 1700)], fill="#020617")
    draw.text((600, 1645), "Generated by VisionRoom AI Engine - All Rights Reserved", fill="#475569", font=get_font(14), anchor="ms")
    
    # Save as PDF file with unique timestamp to prevent Windows lock conflicts
    pdf_path = f"design_specification_sheet_{int(time.time())}.pdf"
    sheet.save(pdf_path, "PDF")
    
    return sheet, pdf_path


# Helper to get rooms list for a category
def get_rooms_by_category(category):
    if category == "All":
        return ROOMS_DATABASE
    cat_type = category.lower().replace(" ", "_")
    return [r for r in ROOMS_DATABASE if r["type"] == cat_type]


# Helper to format list of tuples for gr.Gallery
def get_gallery_items(category):
    rooms_list = get_rooms_by_category(category)
    # Return list of tuples: (image_path, name)
    return [(r["img"], r["name"]) for r in rooms_list]


# Setup Gradio theme
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="teal",
    neutral_hue="slate"
).set(
    body_background_fill="#090d16",
    body_text_color="#f1f5f9",
    block_background_fill="rgba(17, 24, 39, 0.75)",
    block_border_color="rgba(255, 255, 255, 0.08)",
    button_primary_background_fill="#6366f1",
    button_primary_background_fill_hover="#4f46e5"
)

custom_css = """
#left-panel {
    position: -webkit-sticky !important;
    position: sticky !important;
    top: 20px;
    align-self: start;
}

#right-panel {
    max-height: 80vh;
    overflow-y: auto;
    padding-right: 8px;
}

#right-panel::-webkit-scrollbar {
    width: 6px;
}
#right-panel::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 4px;
}
#right-panel::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.3);
    border-radius: 4px;
}
#right-panel::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.6);
}
"""

with gr.Blocks(theme=theme, css=custom_css, title="VisionRoom AI - Gradio Model Visualizer") as demo:
    status_badge = f"""
        <div style="display: inline-block; margin-top: 8px; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600; background: {('rgba(16, 185, 129, 0.15)' if MODEL_AVAILABLE else 'rgba(245, 158, 11, 0.15)')}; color: {('#10b981' if MODEL_AVAILABLE else '#f59e0b')}; border: 1px solid {('rgba(16, 185, 129, 0.3)' if MODEL_AVAILABLE else 'rgba(245, 158, 11, 0.3)')};">
            {('⚡ SegFormer AI Engine: Active' if MODEL_AVAILABLE else '⚠️ Simulation Mode (Generic Floor Mask)')}
        </div>
    """
    
    gr.HTML(f"""
        <div style="text-align: center; margin-bottom: 24px; padding-top: 10px;">
            <h1 style="font-size: 32px; font-weight: 800; color: #6366f1; margin-bottom: 4px;">VisionRoom AI</h1>
            <p style="color: #94a3b8; font-size: 16px; margin-bottom: 4px;">Premium Room & Flooring AI Model Visualizer (Gradio Interface)</p>
            {status_badge}
        </div>
    """)

    # State variables
    category_state = gr.State("All")
    selected_room_path = gr.Textbox(value=ROOMS_DATABASE[0]["img"], visible=False)
    selected_tile_path = gr.Textbox(value=PRESET_TILES_DATABASE[0]["img"], visible=False)

    with gr.Tabs():
        with gr.Tab("🎨 Interactive Workspace"):
            with gr.Row():
                # Left Panel - Input presets & Output viewer
                with gr.Column(scale=1.2, elem_id="left-panel"):
                    output_image = gr.Image(label="Visualization Workspace Render", type="pil", height=500)
                    cost_report = gr.Textbox(label="Pricing Calculator Breakdown", lines=8, interactive=False)

                # Right Panel - Customizers and layout controls
                with gr.Column(scale=1, elem_id="right-panel"):
            
                    with gr.Accordion("Step 1: Choose Room Space (Curated Gallery)", open=True):
                        room_category = gr.Radio(
                            choices=["All", "Living Room", "Bedroom", "Kitchen", "Bathroom"],
                            value="All",
                            label="Filter Room Type"
                        )
                
                        # Visual Rooms selection Grid
                        room_gallery = gr.Gallery(
                            value=get_gallery_items("All"),
                            label="Select Room Template",
                            columns=3,
                            rows=2,
                            height=280,
                            allow_preview=False,
                            object_fit="contain"
                        )
                
                        custom_room = gr.File(
                            label="Or Upload Your Room Image (PNG, JPG)",
                            file_types=["image"]
                        )

                    with gr.Accordion("Step 2: Select Tile Material (Visual Catalog)", open=True):
                        # Visual Tiles selection Grid
                        tile_gallery = gr.Gallery(
                            value=[(t["img"], t["name"]) for t in PRESET_TILES_DATABASE],
                            label="Select Tile from Catalog",
                            columns=3,
                            height=160,
                            allow_preview=False,
                            object_fit="contain"
                        )
                
                        custom_tile = gr.File(
                            label="Or Upload Custom Tile Swatch",
                            file_types=["image"]
                        )

                    with gr.Accordion("Step 3: Tiling Layout Customization", open=True):
                        pattern = gr.Radio(
                            choices=["Grid Standard", "Brick Offset"],
                            value="Grid Standard",
                            label="Layout Pattern"
                        )
                        scale = gr.Slider(
                            minimum=0.1,
                            maximum=2.5,
                            step=0.05,
                            value=1.0,
                            label="Tile Size Scale"
                        )
                        rotation = gr.Slider(
                            minimum=0,
                            maximum=360,
                            step=5,
                            value=0,
                            label="Tiling Angle (Rotation)"
                        )
                        brightness = gr.Slider(
                            minimum=60,
                            maximum=140,
                            step=5,
                            value=100,
                            label="Exposure (Brightness)"
                        )
                        grout_width = gr.Slider(
                            minimum=0,
                            maximum=8,
                            step=1,
                            value=0,
                            label="Grout Width (pixels)"
                        )
                        grout_color = gr.ColorPicker(
                            value="#ffffff",
                            label="Grout Line Color"
                        )
                        finish = gr.Radio(
                            choices=["Matte", "Glossy"],
                            value="Matte",
                            label="Surface Finish Style"
                        )
                        horizon_pct = gr.Slider(
                            minimum=10,
                            maximum=90,
                            step=1,
                            value=50,
                            label="Perspective Horizon Height (%)"
                        )
                        vpx_pct = gr.Slider(
                            minimum=10,
                            maximum=90,
                            step=1,
                            value=50,
                            label="Perspective Vanishing Point X (%)"
                        )

                    with gr.Accordion("Step 4: Project Area", open=False):
                        with gr.Row():
                            calc_length = gr.Number(value=15, label="Room Length (ft)")
                            calc_width = gr.Number(value=20, label="Room Width (ft)")

                    # Submit Actions
                    with gr.Row():
                        reset_btn = gr.Button("Reset Parameters")
                        submit_btn = gr.Button("Generate Render", variant="primary")

        with gr.Tab("↔️ Before / After Comparison"):
            gr.Markdown("### Compare Original Room vs. Newly Tiled Flooring Render")
            with gr.Row():
                original_viewer = gr.Image(label="Original Room Layout", type="pil")
                comparison_viewer = gr.Image(label="Visualization Render", type="pil")
                
        with gr.Tab("📋 Client Design Catalog Summary"):
            gr.Markdown("### Generate Client Specification PDF Sheet")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("Generate a formal, download-ready A4 design specification sheet compiling your layout specs, grout configurations, material swatch, and final costing estimate.")
                    gen_spec_btn = gr.Button("Generate Design Specification Catalog", variant="primary")
                    spec_download_file = gr.File(label="Download Generated PDF Spec Sheet")
                with gr.Column(scale=1.5):
                    spec_preview = gr.Image(label="Spec Sheet Layout Preview", type="pil")

    # Connect Submit Logic
    inputs_list = [
        selected_room_path,
        custom_room,
        selected_tile_path,
        custom_tile,
        scale,
        rotation,
        brightness,
        pattern,
        grout_width,
        grout_color,
        finish,
        calc_length,
        calc_width,
        horizon_pct,
        vpx_pct
    ]

    # Render triggers
    submit_btn.click(
        fn=process_visualization,
        inputs=inputs_list,
        outputs=[output_image, cost_report, original_viewer, comparison_viewer]
    )

    # Helper function when Room Category Tab changes
    def on_category_change(category):
        items = get_gallery_items(category)
        default_path = ROOMS_DATABASE[0]["img"]
        filtered_rooms = get_rooms_by_category(category)
        if filtered_rooms:
            default_path = filtered_rooms[0]["img"]
        
        # Return updated gallery list, updated state, and set default path
        return items, category, default_path

    room_category.change(
        fn=on_category_change,
        inputs=[room_category],
        outputs=[room_gallery, category_state, selected_room_path]
    )

    # Click Select Listener for Room Gallery (automatically clears custom upload)
    def on_room_select(evt: gr.SelectData, category):
        filtered_rooms = get_rooms_by_category(category)
        if evt.index < len(filtered_rooms):
            selected_r = filtered_rooms[evt.index]
            return selected_r["img"], None
        return ROOMS_DATABASE[0]["img"], None

    room_gallery.select(
        fn=on_room_select,
        inputs=[category_state],
        outputs=[selected_room_path, custom_room]
    )

    # Click Select Listener for Tile Gallery (automatically clears custom upload)
    def on_tile_select(evt: gr.SelectData):
        if evt.index < len(PRESET_TILES_DATABASE):
            selected_t = PRESET_TILES_DATABASE[evt.index]
            return selected_t["img"], None
        return PRESET_TILES_DATABASE[0]["img"], None

    tile_gallery.select(
        fn=on_tile_select,
        inputs=[],
        outputs=[selected_tile_path, custom_tile]
    )

    # Trigger automatic recalculation/render when inputs change
    selected_room_path.change(
        fn=process_visualization,
        inputs=inputs_list,
        outputs=[output_image, cost_report, original_viewer, comparison_viewer]
    )
    selected_tile_path.change(
        fn=process_visualization,
        inputs=inputs_list,
        outputs=[output_image, cost_report, original_viewer, comparison_viewer]
    )
    custom_room.change(
        fn=process_visualization,
        inputs=inputs_list,
        outputs=[output_image, cost_report, original_viewer, comparison_viewer]
    )
    custom_tile.change(
        fn=process_visualization,
        inputs=inputs_list,
        outputs=[output_image, cost_report, original_viewer, comparison_viewer]
    )

    # Client PDF generator trigger
    gen_spec_btn.click(
        fn=generate_client_pdf,
        inputs=inputs_list,
        outputs=[spec_preview, spec_download_file]
    )

    # Reset Action
    def reset_controls():
        return (
            ROOMS_DATABASE[0]["img"], None,
            PRESET_TILES_DATABASE[0]["img"], None,
            1.0, 0, 100,
            "Grid Standard", 0, "#ffffff",
            "Matte", 15, 20,
            50, 50,
            "All"
        )
    
    reset_btn.click(
        fn=reset_controls,
        inputs=[],
        outputs=[
            selected_room_path, custom_room,
            selected_tile_path, custom_tile,
            scale, rotation, brightness,
            pattern, grout_width, grout_color,
            finish, calc_length, calc_width,
            horizon_pct, vpx_pct,
            room_category
        ]
    )

    # Initial Run
    demo.load(
        fn=process_visualization,
        inputs=inputs_list,
        outputs=[output_image, cost_report, original_viewer, comparison_viewer]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
