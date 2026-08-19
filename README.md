# VisionRoom AI — Premium Floor Visualizer

An AI-powered, interactive web application that allows users to instantly visualize custom flooring (marble, wood planks, terrazzo, granite) inside any room photo. It segments the floor using a semantic segmentation deep learning model and overlays new tiles with realistic lighting and furniture shadows.

---

## 🌟 Key Features

1. **AI-Powered Floor Segmentation**:
   * Leverages the **SegFormer** semantic segmentation network (fine-tuned on ADE20K) to instantly identify and segment the floor area in any photo.
   * Smart masking automatically excludes obstacle outlines such as sofa bases, table legs, and chairs.
2. **Realistic Blending & Lighting**:
   * Uses **dynamic shadow mapping** to extract room shadows, window glare, and soft lighting details.
   * Blends these elements on top of the new floor tiles, preventing a "flat" or artificial look.
3. **Interactive Layout Customizer**:
   * Rotate tiles/planks to any angle (e.g., 45-degree herringbone layouts).
   * Adjust tile size/scale interactively.
   * Toggle between Standard Grid or Brick Offset layouts.
   * Customize grout line width and grout color.
   * Toggle finish styles between **Matte** or **Glossy** (adds subtle reflection highlights).
4. **Compare Mode**:
   * Direct side-by-side swipe slider comparing the original room floor with the newly visualized design.
5. **Instant Cost Calculator**:
   * Enter room dimensions to estimate tile count, base material cost, GST, and shipping.
6. **PDF Spec Catalog Export**:
   * Download a professional client-facing quote sheet detailing layout settings, cost breakdowns, and the final visualization.

---

## 💻 Tech Stack

### Frontend (Client-side)
* **HTML5 & Vanilla CSS3**: Highly optimized, modern dark-themed glassmorphism visual layout.
* **Vanilla JavaScript (ES6+)**: Handles UI routing, dynamic interactive event listeners, and backend API communication.
* **HTML5 2D Canvas API**: High-performance, GPU-accelerated rendering engine that performs tile scaling, rotation, pattern offsets, and shadow composites directly in the browser.

### Backend (AI Server)
* **Python 3**: Core language powering the backend data pipeline and model execution.
* **FastAPI**: Modern, high-performance web framework for Python serving static web files and the AI `/api/segment` endpoint.
* **Uvicorn**: Lightweight ASGI web server for local execution.
* **PyTorch & Hugging Face Transformers**: Loads and runs the SegFormer model (`nvidia/segformer-b2-finetuned-ade-512-512`).
* **OpenCV & NumPy**: Cleans the binary AI segmentation mask (morphological closing) and simplifies the polygon borders.

---

## 🚀 Setup & Installation (Windows)

1. Clone or download this project folder.
2. Run the **`run.bat`** file. The script will automatically:
   * Detect or create a local Python virtual environment (`venv`).
   * Download and install CPU-optimized PyTorch and all backend dependencies.
   * Start the FastAPI application.
3. Open your web browser and navigate to:
   👉 **[http://localhost:8000](http://localhost:8000)**

---

## 📂 Project Structure

```bash
├── assets/                    # Seamless tile textures and interface media
├── images_templates/          # Preset curated room space photos
├── venv/                      # Local Python virtual environment
├── index.html                 # Main interface structure
├── style.css                  # UI layout and interactive styling rules
├── app.js                     # Core frontend visualizer & calculator logic
├── server.py                  # FastAPI server and SegFormer AI endpoint
├── run.bat                    # One-click startup script for Windows
├── .gitignore                 # Configured git ignore definitions
└── README.md                  # Project documentation & presentation guide
```
