# VisionRoom AI — Premium Floor & Wall Visualizer

An AI-powered, interactive room visualizer application that allows users to instantly visualize premium tiling (marble, wood planks, terrazzo, granite) on both the **floor** and **walls** of any room photo with **realistic perspective homography warping**, **ambient light/shadow blending**, and **automatic obstacle-aware area estimation**.

---

## 🌟 Key Features

1. **Dual Floor & Wall AI Segmentation**:
   * Leverages the **SegFormer** semantic segmentation network (`nvidia/segformer-b2-finetuned-ade-512-512`) to simultaneously identify and separate the floor (Label 3) and walls (Label 0) in one inference pass.
   * Auto-closes small contour gaps to handle obstacle boundaries cleanly.

2. **Perspective-Correct Homography Engine**:
   * Auto-detects 4-point projective vanishing quads for the floor and individual wall planes.
   * Warps repeating tile textures using perspective homography (`cv2.warpPerspective` & `cv2.findHomography`), ensuring tiles naturally recede and foreshorten with distance.
   * Grout lines and brick/grid pattern offsets are embedded before projective warping so that seams also foreshorten accurately with depth.

3. **Obstacle-Aware Area & Dimension Deduction**:
   * Automatically isolates indoor obstacles using ADE20K semantic classes:
     * **Wall Obstacles**: Windows (Label 9), Doors (Label 14).
     * **Floor Obstacles**: Sofas (Label 23), Tables (Label 15), Beds (Label 7), Chairs (Label 19), Rugs (Label 58), Stairs (Label 53).
   * Calculates **Total Area** vs. **Net Tile Area (sq.ft)** by subtracting obstacle footprints.
   * Derives real-world scale using standard architectural reference objects (doors @ 2.05m / windows @ 1.2m).

4. **Ambient Light & Shadow Mapping**:
   * Dynamically extracts grayscale lighting and shadow maps from the original photo.
   * Blends this lighting map on top of the tile textures to preserve natural window light, corner shading, and furniture shadows.
   * New **Ambient Light Blend** slider (10%–100%) for fine-tuning lighting realism.

5. **Interactive Multi-Target Control Panel**:
   * Toggle between **Floor Tiling** and **Wall Tiling** in the materials panel.
   * Adjust tile size, rotation angle, brightness, grout size/color, surface finish (Matte / Satin / Glossy), and shadow blend with debounced live updates.
   * Visual shimmer indicator during AI rendering passes.

6. **Original Image Restore (Reset)**:
   * Global reset button restores all sliders, clears active tile selections on both layers, and immediately displays the clean original room photo.

7. **Compare Mode, Estimator & PDF Spec sheets**:
   * Swipable split-screen compare viewer.
   * Cost and tile count calculator with real-time tax/shipping calculations.
   * Professional PDF spec sheet export with cost breakdowns and design specifications.

---

## 💻 Tech Stack

- **Frontend**: HTML5 Canvas (2D Composite Engine), CSS3 (Dark Glassmorphism Layout), Vanilla JS (ES6 State Machine).
- **Backend**: Python 3.13, FastAPI (HTTP API hosting), Uvicorn (ASGI web server).
- **Deep Learning & CV**: PyTorch, HuggingFace Transformers (SegFormer ADE20K), OpenCV, NumPy, Pillow.

---

## 🚀 Setup & Installation (Windows)

1. Clone or download this repository.
2. Double-click the **`run.bat`** file in the root folder. The script will automatically:
   * Detect or create a local Python virtual environment (`venv`).
   * Download and install PyTorch and all backend dependencies.
   * Run the FastAPI application.
3. Open your web browser and navigate to:
   👉 **[http://localhost:8000](http://localhost:8000)**

---

## 📓 Advanced Auto-Perspective & Area Notebook
The project includes the Jupyter Notebook: **`Floor_Detection_using_SAM_2_n1 (1).ipynb`**
* **Cells 10–15**: Identifies walls using SegFormer.
* **Cells 51–54**: Auto-derives floor perspective vanishing quads from extreme points.
* **Cells 55–56**: Wall perspective quad detection and homography warping prototype.
* **Cell 57**: Obstacle-aware mask extraction (excluding doors, windows, and furniture footprints).
* **Cell 58**: Real-world scale derivation (using door height) & net tile area calculation in sq.ft.

---

## 📂 Project Structure

```bash
├── assets/                    # Seamless tile textures and interface media
├── images_templates/          # Preset curated room templates (living room, bedroom, kitchen, bathroom)
├── venv/                      # Local Python virtual environment
├── index.html                 # Main interface structure & panels
├── style.css                  # UI layout, toggles, shimmer animations & responsive styling
├── app.js                     # Core frontend compositor, slider synchronization & resets
├── server.py                  # FastAPI server with SegFormer inference & homography rendering
├── run.bat                    # One-click startup script for Windows
├── .gitignore                 # Configured git ignore definitions
└── README.md                  # Project documentation & presentation guide
```
