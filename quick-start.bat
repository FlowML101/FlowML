@echo off
echo.
echo  ========================================
echo     FlowML Studio - Quick Start
echo  ========================================
echo.

:: Start Docker containers first
echo  [1/3] Starting Docker (Postgres + Redis)...
cd /d %~dp0deploy
docker-compose up -d postgres redis >nul 2>&1
cd /d %~dp0
timeout /t 2 /nobreak > nul

:: Start Backend in new window
echo  [2/3] Starting Backend...
start "FlowML Backend" cmd /k "cd /d %~dp0backend && .venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

:: Brief pause
timeout /t 2 /nobreak > nul

:: Start Frontend in new window  
echo  [3/3] Starting Frontend...
start "FlowML Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo  ========================================
echo   Services are starting in new windows
echo  ========================================
echo.
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo   API Docs: http://localhost:8000/docs
echo.
echo   Close this window when done.
echo  ========================================

:: Wait a moment then open browser
timeout /t 4 /nobreak > nul
start http://localhost:5173
