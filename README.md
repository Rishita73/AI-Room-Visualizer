# VisionRoom AI — Premium Floor & Wall Visualizer

An AI-powered, interactive room visualizer application that allows users to instantly visualize premium tiling (marble, wood planks, terrazzo, granite) on both the **floor** and **walls** of any room photo. 

It leverages semantic segmentation models to isolate room structures, calculates natural shadows, and overlays realistic textures directly in the browser. It also includes an advanced auto-perspective Jupyter Notebook for prototyping corner-based vanishing wall tiles.

---

## 🌟 Key Features

1. **Dual Floor & Wall AI Segmentation**:
   * Leverages the **SegFormer** semantic segmentation network (`nvidia/segformer-b2-finetuned-ade-512-512`) to simultaneously identify and separate the floor (Label 3) and walls (Label 0) in one inference pass.
   * Auto-closes small contour gaps to handle obstacle boundaries like baseboards, sofa edges, and mounted televisions.

2. **Simultaneous Composite Rendering**:
   * Renders **both floor and wall tiling layers at the same time** on the HTML5 canvas.
   * Applies custom tiling rules (grid/brick patterns, scale, rotation angles, brightness adjustments) separately for each target.

3. **Ambient Light & Shadow Mapping**:
   * Dynamically extracts grayscale light/shadow maps from the original photo.
   * Blends this lighting map on top of the tile textures to preserve natural window light, corner shading, and furniture shadows, ensuring the result looks realistic rather than flat.

4. **Interactive Target Control Panel**:
   * Toggle between **Floor Tiling** and **Wall Tiling** in the materials panel.
   * Catalog swatches and layout adjustment sliders (scale, rotation, brightness, grout size, grout color, matte/glossy finish) automatically synchronize and bind to the active target's state.

5. **Original Image Restore (Reset)**:
   * A global reset button resets all sliders, clears active tile selections on both layers, and immediately displays the original room photo.

6. **Compare Mode, Estimator & PDF Spec sheets**:
   * Swipable split-screen compare viewer.
   * Cost and tile count calculator.
   * Professional PDF spec sheet export with cost breakdowns and designs.

---

## 💻 Tech Stack

- **Frontend**: HTML5 Canvas (2D Composite Engine), CSS3 (Dark Glassmorphism Layout), Vanilla JS (ES6 State Machine).
- **Backend**: Python 3.13, FastAPI (HTTP API hosting), Uvicorn (ASGI web server).
- **Deep Learning & CV**: PyTorch, HuggingFace Transformers (SegFormer), OpenCV, NumPy.

---

## 🚀 Setup & Installation (Windows)

1. Clone or download this repository.
2. Double-click the **`run.bat`** file in the root folder. The script will automatically:
   * Detect or create a local Python virtual environment (`venv`).
   * Download and install CPU-optimized PyTorch and all backend dependencies.
   * Run the FastAPI application.
3. Open your web browser and navigate to:
   👉 **[http://localhost:8000](http://localhost:8000)**

---

## 📓 Advanced Auto-Perspective Notebook
The directory contains a Jupyter Notebook: **`Floor_Detection_using_SAM_2_n1 (1).ipynb`**
* **Cells 10–15**: Identifies walls using SegFormer.
* **Cells 55–56**: Contains the advanced Python prototype that automatically isolates distinct wall sections, detects their 4 perspective corners (vanishing quad), and uses `cv2.getPerspectiveTransform` (homography) to skew tile textures realistically to match room geometry.

---

## 📂 Project Structure

```bash
├── assets/                    # Seamless tile textures and interface media
├── images_templates/          # Preset curated room templates (bedroom, bathroom, kitchen)
├── venv/                      # Local Python virtual environment
├── index.html                 # Main interface structure & panels
├── style.css                  # UI layout, toggles, and responsive styling
├── app.js                     # Core frontend compositor, slider synchronization & resets
├── server.py                  # FastAPI server & SegFormer segment endpoint
├── run.bat                    # One-click startup script for Windows
├── .gitignore                 # Configured git ignore definitions
└── README.md                  # Project documentation & presentation guide
```
