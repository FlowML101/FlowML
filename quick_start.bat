@echo off
echo.
echo  ========================================
echo     FlowML Studio - Complete Setup
echo  ========================================
echo.
echo  This will install everything and start all services.
echo  First run takes 5-10 minutes.
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found! Install Python 3.10+ from python.org
    echo  Make sure to check "Add Python to PATH" during installation
    pause
    exit
)
echo  [OK] Python detected

:: Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Node.js not found! Install Node.js 18+ from nodejs.org
    pause
    exit
)
echo  [OK] Node.js detected

:: Check Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo  [WARNING] Docker not found - will use SQLite instead of PostgreSQL
    set SKIP_DOCKER=1
) else (
    echo  [OK] Docker detected
    set SKIP_DOCKER=0
)

echo.
echo  ========================================
echo   Starting Installation
echo  ========================================
echo.

:: Create Python venv
echo  [1/5] Setting up Python environment...
cd /d %~dp0backend
if not exist .venv (
    python -m venv .venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create virtual environment
        pause
        exit
    )
)
echo  [OK] Virtual environment ready

:: Install Python packages
echo  [2/5] Installing Python packages (this takes 3-5 minutes)...
call .venv\Scripts\activate.bat

:: Check if packages are already installed
python -c "import fastapi, uvicorn, celery" >nul 2>&1
if errorlevel 1 (
    echo  Upgrading pip...
    python -m pip install --upgrade pip setuptools wheel
    echo.
    echo  Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo  [ERROR] Failed to install Python packages
        pause
        exit
    )
    echo  [OK] Python packages installed
) else (
    echo  [OK] Python packages already installed (skipping)
)

:: Install Node packages
echo  [3/5] Installing Node.js packages (this takes 2-3 minutes)...
cd /d %~dp0frontend
if not exist node_modules (
    echo  Installing Node modules...
    call npm install
    if errorlevel 1 (
        echo  [ERROR] Failed to install Node packages
        pause
        exit
    )
    echo  [OK] Node packages installed
) else (
    echo  [OK] Node packages already installed (skipping)
)

:: Create directories
echo  [4/5] Creating directories...
cd /d %~dp0backend
if not exist uploads mkdir uploads
if not exist trained_models mkdir trained_models
if not exist logs mkdir logs
echo  [OK] Directories created

:: Start Docker
if "%SKIP_DOCKER%"=="0" (
    echo  [5/5] Starting Docker services...
    cd /d %~dp0deploy
    docker-compose up -d postgres redis minio minio-setup >nul 2>&1
    timeout /t 5 /nobreak > nul
    echo  [OK] Docker services started
) else (
    echo  [5/5] Skipping Docker (using SQLite)
)

echo.
echo  ========================================
echo   Launching Services
echo  ========================================
echo.

:: N Word
:: Start Backend
echo  Starting Backend API...
start "FlowML Backend" cmd /k "cd /d %~dp0backend && .venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"
timeout /t 2 /nobreak > nul

:: Start Worker
echo  Starting Celery Worker...
start "FlowML Worker" cmd /k "cd /d %~dp0backend && .venv\Scripts\celery.exe -A worker.celery_app worker --loglevel=INFO --pool=solo -Q cpu,gpu --hostname=flowml-worker@%%h"
timeout /t 2 /nobreak > nul

:: Start Frontend
echo  Starting Frontend...
start "FlowML Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo  ========================================
echo   Services are starting!
echo  ========================================
echo.
echo   Frontend:  http://localhost:5173
echo   Backend:   http://localhost:8000
echo   API Docs:  http://localhost:8000/docs
echo.
echo   Wait 10 seconds for services to start...
echo   Browser will open automatically.
echo.
echo   Close this window when done.
echo  ========================================
echo.

:: Wait and open browser
timeout /t 10 /nobreak > nul
start http://localhost:5173

echo  Setup complete!
echo.
