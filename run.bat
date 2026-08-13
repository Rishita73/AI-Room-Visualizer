@echo off
title VisionRoom AI - Premium Web Visualizer
cls
echo ===================================================
echo     VisionRoom AI Premium Web Interface (FastAPI)
echo ===================================================
echo.
echo Setting up local python environment.

REM 1. Check if virtual environment exists, if not create it
if not exist venv (
    echo Creating python virtual environment.
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv. Ensure Python is installed and in path.
        pause
        exit /b 1
    )
)

REM 2. Activate Virtual Environment
echo Activating virtual environment.
call venv\Scripts\activate

REM 3. Check and install dependencies
echo Checking dependencies.
python -c "import torch, transformers, cv2, PIL, numpy, fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [INFO] Installing required libraries. This may take a few minutes.
    echo.
    echo Installing CPU-optimized PyTorch.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-warn-script-location
    
    echo Installing Transformers, OpenCV, Pillow, FastAPI, Uvicorn.
    pip install transformers opencv-python pillow numpy fastapi uvicorn --no-warn-script-location
    
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies. Check your internet connection.
        pause
        exit /b 1
    )
) else (
    echo All dependencies are already installed.
)

REM 4. Launch FastAPI Web app
echo.
echo Starting FastAPI server and serving Premium Web Interface.
echo Open your browser and navigate to: http://localhost:8000
echo.
echo Press Ctrl+C in this terminal to stop the server.
echo ===================================================
echo.

venv\Scripts\python server.py

pause
