@echo off
title NeuralScope - Starting All Services
color 0b

echo =======================================================
echo          NeuralScope Service Orchestrator
echo =======================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Frontend Dependencies...
cd frontend
if not exist "node_modules\" (
    echo Installing frontend packages (one-time setup)...
    call npm install
    if %errorlevel% neq 0 (
        echo [ERROR] npm install encountered an issue. Make sure Node.js is installed.
        pause
        exit /b
    )
) else (
    echo Frontend dependencies are already installed.
)

echo.
echo [2/3] Launching Frontend Development Server...
start "NeuralScope Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo [3/3] Launching Backend FastAPI Server...
start "NeuralScope Backend" cmd /k "cd /d %~dp0backend && python run.py"

echo.
echo =======================================================
echo Both servers are launching!
echo Frontend: http://localhost:5173
echo Backend:  http://localhost:8000 (Docs: /docs)
echo =======================================================
echo.

timeout /t 3 /nobreak >nul
start http://localhost:5173

exit

