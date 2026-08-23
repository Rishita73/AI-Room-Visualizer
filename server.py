import os
import io
import base64
import numpy as np
from PIL import Image
import cv2

# FastAPI Imports
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI(title="AI Room Visualizer API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for AI model
MODEL_AVAILABLE = False
processor = None
model = None

# Attempt to load model dependencies
try:
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    
    print("Loading SegFormer model (nvidia/segformer-b2-finetuned-ade-512-512)...")
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
    print(f"Warning: Could not load SegFormer model. Running in simulation mode. Error: {e}")
    MODEL_AVAILABLE = False


def clean_and_extract_contour(mask, width, height):
    """
    Cleans up a binary mask using morphological operations and extracts the largest contour.
    """
    # 1. Clean mask using Morphological Closing (removes small holes)
    kernel = np.ones((7, 7), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 2. Resize mask to match original image size
    mask_resized = cv2.resize(mask_cleaned, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # 3. Find contours
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    polygon_points = []
    if contours:
        largest = max(contours, key=cv2.contourArea)
        # Simplify contour to reduce payload size and make it smooth
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        # approx shape is (N, 1, 2). Squeeze and convert to list of [x, y]
        pts = approx.squeeze()
        if pts.ndim == 2:
            polygon_points = pts.tolist()
        elif pts.ndim == 1 and len(pts) == 2:
            # Handle edge case where polygon has only 1 point
            polygon_points = [pts.tolist()]
            
    return mask_resized, polygon_points


def create_rgba_mask_image(mask_binary):
    """
    Converts a binary mask (0 or 255) to a transparent RGBA PNG.
    The mask area is filled with solid white (255, 255, 255, 255)
    and the rest is fully transparent (0, 0, 0, 0).
    """
    h, w = mask_binary.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Set white color where mask is active
    rgba[mask_binary > 0] = [255, 255, 255, 255]
    
    img = Image.fromarray(rgba, mode="RGBA")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    mask_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{mask_base64}"


@app.post("/api/segment")
async def segment_room(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_bytes = await file.read()
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = original_image.size
        
        # Initialize output structures
        floor_mask_base64 = ""
        floor_polygon = []
        wall_mask_base64 = ""
        wall_polygon = []
        is_mock = False
        
        if MODEL_AVAILABLE:
            try:
                # Resize image for SegFormer model
                inference_img = original_image.resize((1024, 768))
                
                # Preprocess image
                inputs = processor(images=inference_img, return_tensors="pt")
                
                # Run inference
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Post-process segmentation map
                segmentation = processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[inference_img.size[::-1]]
                )[0]
                
                # ADE20K floor label ID is 3, wall label ID is 0
                FLOOR_LABEL_ID = 3
                WALL_LABEL_ID = 0
                
                # Extract floor
                floor_mask_raw = (segmentation == FLOOR_LABEL_ID).cpu().numpy().astype(np.uint8) * 255
                floor_cleaned, floor_polygon = clean_and_extract_contour(floor_mask_raw, width, height)
                floor_mask_base64 = create_rgba_mask_image(floor_cleaned)
                
                # Extract wall
                wall_mask_raw = (segmentation == WALL_LABEL_ID).cpu().numpy().astype(np.uint8) * 255
                wall_cleaned, wall_polygon = clean_and_extract_contour(wall_mask_raw, width, height)
                wall_mask_base64 = create_rgba_mask_image(wall_cleaned)
                
            except Exception as model_err:
                print(f"Error during model inference: {model_err}. Falling back to simulation.")
                is_mock = True
        else:
            is_mock = True
            
        if is_mock:
            # 1. Mock Floor (Trapezoid at the bottom third)
            floor_mock = np.zeros((height, width), dtype=np.uint8)
            floor_pts = np.array([
                [0, int(height * 0.95)],
                [int(width * 0.35), int(height * 0.58)],
                [int(width * 0.65), int(height * 0.58)],
                [width, int(height * 0.95)]
            ], dtype=np.int32)
            cv2.fillPoly(floor_mock, [floor_pts], 255)
            floor_cleaned, floor_polygon = clean_and_extract_contour(floor_mock, width, height)
            floor_mask_base64 = create_rgba_mask_image(floor_cleaned)
            
            # 2. Mock Wall (Two rectangles on the left and right sides)
            wall_mock = np.zeros((height, width), dtype=np.uint8)
            # Left wall rect
            cv2.rectangle(wall_mock, (0, 0), (int(width * 0.35), int(height * 0.95)), 255, -1)
            # Right wall rect
            cv2.rectangle(wall_mock, (int(width * 0.65), 0), (width, int(height * 0.95)), 255, -1)
            # Exclude floor overlap from wall mock
            cv2.fillPoly(wall_mock, [floor_pts], 0)
            wall_cleaned, wall_polygon = clean_and_extract_contour(wall_mock, width, height)
            wall_mask_base64 = create_rgba_mask_image(wall_cleaned)
            
        return JSONResponse(content={
            "floor_mask": floor_mask_base64,
            "floor_polygon": floor_polygon,
            "wall_mask": wall_mask_base64,
            "wall_polygon": wall_polygon,
            "width": width,
            "height": height,
            "simulated": is_mock
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")


# Serve Frontend files
frontend_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(frontend_dir, "index.html")):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    @app.get("/")
    def read_root():
        return {"message": "AI Room Visualizer API is active. Frontend files not found in the current directory."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
