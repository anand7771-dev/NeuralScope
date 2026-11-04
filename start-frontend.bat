@echo off
title NeuralScope Frontend (Vite)
color 0a

cd /d "%~dp0frontend"

if not exist "node_modules\" (
    echo Installing frontend dependencies...
    call npm install
)

echo Starting Vite Frontend Server on http://localhost:5173 ...
call npm run dev
pause

