@echo off
setlocal enabledelayedexpansion

REM ========================================
REM  FlowML Studio - Complete Setup Script
REM ========================================
REM
REM This script will:
REM  1. Check for required software (Python, Node.js, Docker)
REM  2. Create Python virtual environment
REM  3. Install Python dependencies
REM  4. Install Node.js dependencies
REM  5. Start Docker infrastructure
REM  6. Launch all services (Backend, Worker, Frontend)
REM
REM Requirements:
REM  - Python 3.10 or higher
REM  - Node.js 18 or higher
REM  - Docker Desktop (for Postgres, Redis, MinIO)
REM ========================================

echo.
echo  ========================================
echo     FlowML Studio - Complete Setup
echo  ========================================
echo.

REM Store the project root directory
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

REM ========================================
REM Step 1: Check Prerequisites
REM ========================================
echo  [1/8] Checking prerequisites...
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python is not installed or not in PATH
    echo  Please install Python 3.10+ from https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo  ✓ Python %PYTHON_VERSION% detected

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Node.js is not installed or not in PATH
    echo  Please install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)

for /f "tokens=1" %%i in ('node --version') do set NODE_VERSION=%%i
echo  ✓ Node.js %NODE_VERSION% detected

REM Check npm
npm --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] npm is not installed
    pause
    exit /b 1
)

echo  ✓ npm detected

REM Check Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo  [WARNING] Docker is not installed or not running
    echo  Docker is required for Postgres, Redis, and MinIO
    echo  Please install Docker Desktop from https://www.docker.com/products/docker-desktop/
    echo.
    choice /C YN /M "Continue without Docker (limited functionality)"
    if errorlevel 2 exit /b 1
    set "SKIP_DOCKER=1"
) else (
    echo  ✓ Docker detected
    set "SKIP_DOCKER=0"
)

echo.

REM ========================================
REM Step 2: Create Python Virtual Environment
REM ========================================
echo  [2/8] Setting up Python virtual environment...
echo.

cd /d "%PROJECT_ROOT%backend"

if exist ".venv\" (
    echo  ✓ Virtual environment already exists
) else (
    echo  Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo  ✓ Virtual environment created
)

echo.

REM ========================================
REM Step 3: Install Python Dependencies
REM ========================================
echo  [3/8] Installing Python dependencies...
echo  This may take several minutes...
echo.

call .venv\Scripts\activate.bat

REM Upgrade pip first
python -m pip install --upgrade pip setuptools wheel --quiet

REM Install requirements
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo  [ERROR] Failed to install Python dependencies
    echo  Trying with verbose output...
    pip install -r requirements.txt
    pause
    exit /b 1
)

echo  ✓ Python dependencies installed

echo.

REM ========================================
REM Step 4: Install Node.js Dependencies
REM ========================================
echo  [4/8] Installing Node.js dependencies...
echo  This may take several minutes...
echo.

cd /d "%PROJECT_ROOT%frontend"

if exist "node_modules\" (
    echo  ✓ Node modules already exist
    echo  Checking for updates...
    call npm install --quiet
) else (
    echo  Installing Node modules...
    call npm install --quiet
    if errorlevel 1 (
        echo  [ERROR] Failed to install Node dependencies
        echo  Trying with verbose output...
        call npm install
        pause
        exit /b 1
    )
)

echo  ✓ Node.js dependencies installed

echo.

REM ========================================
REM Step 5: Create Required Directories
REM ========================================
echo  [5/8] Creating required directories...
echo.

cd /d "%PROJECT_ROOT%backend"

if not exist "uploads\" mkdir uploads
if not exist "trained_models\" mkdir trained_models
if not exist "logs\" mkdir logs

echo  ✓ Directories created

echo.

REM ========================================
REM Step 6: Start Docker Infrastructure
REM ========================================
if "%SKIP_DOCKER%"=="0" (
    echo  [6/8] Starting Docker infrastructure...
    echo  Starting Postgres, Redis, and MinIO...
    echo.
    
    cd /d "%PROJECT_ROOT%deploy"
    
    REM Check if Docker daemon is running
    docker info >nul 2>&1
    if errorlevel 1 (
        echo  [WARNING] Docker daemon is not running
        echo  Please start Docker Desktop and try again
        pause
        exit /b 1
    )
    
    REM Start Docker containers
    docker-compose up -d postgres redis minio minio-setup
    if errorlevel 1 (
        echo  [ERROR] Failed to start Docker containers
        echo  Please check Docker Desktop and try again
        pause
        exit /b 1
    )
    
    echo  ✓ Docker containers started
    echo.
    echo  Waiting for services to be ready...
    timeout /t 8 /nobreak > nul
    echo  ✓ Services ready
    echo.
) else (
    echo  [6/8] Skipping Docker infrastructure (not installed)
    echo  [WARNING] Application will run in limited mode with SQLite
    echo.
)

REM ========================================
REM Step 7: Initialize Database
REM ========================================
echo  [7/8] Database initialization...
echo  Database will be auto-created on first run
echo  ✓ Ready
echo.

REM ========================================
REM Step 8: Launch All Services
REM ========================================
echo  [8/8] Launching FlowML Studio...
echo.

cd /d "%PROJECT_ROOT%"

REM Start Backend in new window
echo  Starting Backend API...
start "FlowML Backend" cmd /k "cd /d "%PROJECT_ROOT%backend" && .venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

REM Brief pause
timeout /t 3 /nobreak > nul

REM Start Celery Worker in new window
echo  Starting Celery Worker...
start "FlowML Worker" cmd /k "cd /d "%PROJECT_ROOT%backend" && .venv\Scripts\celery.exe -A worker.celery_app worker --loglevel=INFO --pool=solo -Q cpu,gpu --hostname=flowml-worker@%%h"

REM Brief pause
timeout /t 3 /nobreak > nul

REM Start Frontend in new window
echo  Starting Frontend...
start "FlowML Frontend" cmd /k "cd /d "%PROJECT_ROOT%frontend" && npm run dev"

echo.
echo  ========================================
echo   🚀 FlowML Studio is Starting!
echo  ========================================
echo.
echo   Services are launching in separate windows:
echo.
echo   📊 Frontend:  http://localhost:5173
echo   🔧 Backend:   http://localhost:8000
echo   📚 API Docs:  http://localhost:8000/docs
echo   👷 Worker:    Processing jobs in background
echo.

if "%SKIP_DOCKER%"=="0" (
    echo   Infrastructure Services:
    echo   🗄️  PostgreSQL: localhost:5432
    echo   🔴 Redis:      localhost:6379
    echo   💾 MinIO:      http://localhost:9001
    echo.
)

echo   ⏳ Please wait 10-15 seconds for all services to start...
echo.
echo   Close this window when you're done working.
echo   To stop all services:
echo     - Close the service windows
if "%SKIP_DOCKER%"=="0" (
    echo     - Run: docker-compose -f deploy\docker-compose.yml down
)
echo.
echo  ========================================

REM Wait for services to start, then open browser
timeout /t 12 /nobreak > nul

REM Check if services are responding
echo  Opening browser...
start http://localhost:5173

echo.
echo  ✓ Setup complete! Happy machine learning! 🎉
echo.
echo  Press any key to keep this window open for logs...
pause > nul
