# Windows Setup Script for NeuralScope
# Run this in PowerShell from the neural-scope directory

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  NeuralScope Setup Script" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Python not found. Install Python 3.10+ from https://python.org" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] $pythonVersion" -ForegroundColor Green

# Check Node
$nodeVersion = node --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Node.js not found. Install from https://nodejs.org" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Node.js $nodeVersion" -ForegroundColor Green

Write-Host ""
Write-Host "--- Backend Setup ---" -ForegroundColor Yellow

Set-Location backend

# Check for CUDA
Write-Host "Checking for CUDA GPU..." -ForegroundColor Gray
$cudaAvailable = python -c "
try:
    import torch
    print('cuda' if torch.cuda.is_available() else 'cpu')
except:
    print('no_torch')
" 2>&1

if ($cudaAvailable -eq "no_torch") {
    Write-Host "PyTorch not installed. Installing CPU version..." -ForegroundColor Yellow
    Write-Host "If you have an NVIDIA GPU, cancel and install CUDA version instead:" -ForegroundColor Gray
    Write-Host "  pip install torch --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor Gray
    Write-Host ""
    pip install torch --index-url https://download.pytorch.org/whl/cpu
} elseif ($cudaAvailable -eq "cuda") {
    Write-Host "[OK] CUDA GPU detected" -ForegroundColor Green
} else {
    Write-Host "[OK] Running on CPU" -ForegroundColor Yellow
}

Write-Host "Installing backend dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "--- Frontend Setup ---" -ForegroundColor Yellow
Set-Location ../frontend
npm install

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start NeuralScope:" -ForegroundColor White
Write-Host ""
Write-Host "  Terminal 1 (Backend):" -ForegroundColor Gray
Write-Host "    cd neural-scope/backend" -ForegroundColor White
Write-Host "    python run.py" -ForegroundColor White
Write-Host ""
Write-Host "  Terminal 2 (Frontend):" -ForegroundColor Gray
Write-Host "    cd neural-scope/frontend" -ForegroundColor White
Write-Host "    npm run dev" -ForegroundColor White
Write-Host ""
Write-Host "  Then open: http://localhost:5173" -ForegroundColor Cyan
Write-Host ""
Write-Host "NOTE: First run downloads GPT-2 weights (~500 MB)" -ForegroundColor Yellow

