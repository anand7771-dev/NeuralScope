@echo off
title NeuralScope Backend (FastAPI + PyTorch)
color 0e

cd /d "%~dp0backend"

echo Checking PyTorch installation...
python -c "import torch; print('PyTorch version:', torch.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo.
    echo [NOTICE] PyTorch is not yet installed in your Python environment.
    echo Installing CPU version of PyTorch now...
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install transformers accelerate psutil
)

echo.
echo Starting FastAPI Backend on http://localhost:8000 ...
python run.py
pause

