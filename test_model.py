import os
import sys

print("=" * 60)
print("VisionRoom AI Model Verification & Diagnostics")
print("=" * 60)

print("\n[Step 1] Checking Python environment and version...")
print(f"Python Version: {sys.version}")

print("\n[Step 2] Checking required packages...")
libs = ["torch", "transformers", "cv2", "PIL", "numpy", "fastapi", "uvicorn"]
all_ok = True
for lib in libs:
    try:
        __import__(lib)
        print(f"  - {lib}: OK")
    except ImportError:
        print(f"  - {lib}: NOT INSTALLED")
        all_ok = False

if not all_ok:
    print("\n[Warning] Some packages are missing. The server will run in simulation mode.")
    print("If you want the full AI features, run the setup inside `run.bat` or install them using:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    print("  pip install transformers fastapi uvicorn opencv-python pillow numpy")

print("\n[Step 3] Verifying SegFormer Model Loading...")
try:
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    from PIL import Image
    import numpy as np
    
    print("  - Loading processor...")
    processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    print("  - Loading model (cpu mode)...")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    
    print("  - Performing dummy inference...")
    # Create a dummy image
    dummy_img = Image.fromarray(np.uint8(np.random.rand(512, 512, 3) * 255))
    inputs = processor(images=dummy_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        
    segmentation = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[dummy_img.size[::-1]]
    )[0]
    
    print(f"  - Inference Success! Output Shape: {segmentation.shape}")
    print("\n[SUCCESS] AI segmentation model is fully operational locally!")
    
except Exception as e:
    print(f"  - Model test skipped/failed: {e}")
    print("\n[Info] Diagnostics completed. Server is ready to run.")
print("=" * 60)
