@echo off
REM FlowML Development Startup Script (Windows)
REM Usage: start-dev.bat [command]
REM Commands: infra, backend, frontend, all, stop

setlocal enabledelayedexpansion

set COMPOSE_FILE=deploy\docker-compose.yml

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="infra" goto infra
if "%1"=="backend" goto backend
if "%1"=="frontend" goto frontend
if "%1"=="all" goto all
if "%1"=="stop" goto stop
if "%1"=="logs" goto logs
if "%1"=="status" goto status
goto help

:help
echo.
echo FlowML Development Startup
echo ==========================
echo.
echo Usage: start-dev.bat [command]
echo.
echo Commands:
echo   infra     Start Docker infrastructure (Postgres, Redis, MinIO)
echo   backend   Start FastAPI backend server
echo   frontend  Start Vite frontend dev server
echo   all       Start everything (infra + backend + frontend)
echo   stop      Stop all Docker containers
echo   logs      Show Docker container logs
echo   status    Show status of all services
echo.
goto end

:infra
echo.
echo Starting FlowML Infrastructure...
echo =================================
echo.
docker-compose -f %COMPOSE_FILE% up -d postgres redis minio minio-setup
echo.
echo Waiting for services to be healthy...
timeout /t 5 /nobreak > nul
echo.
echo Infrastructure started! Services:
echo   - PostgreSQL: localhost:5432
echo   - Redis:      localhost:6379
echo   - MinIO:      localhost:9000 (Console: localhost:9001)
echo.
goto end

:backend
echo.
echo Starting FlowML Backend...
echo ==========================
echo.
cd backend
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo Installing dependencies...
pip install -q -r requirements.txt
echo.
echo Starting FastAPI server...
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
goto end

:frontend
echo.
echo Starting FlowML Frontend...
echo ===========================
echo.
cd frontend
echo Installing dependencies...
call npm install
echo.
echo Starting Vite dev server...
call npm run dev
goto end

:all
echo.
echo Starting FlowML Full Stack...
echo =============================
echo.
call :infra
echo.
echo Starting backend in new terminal...
start "FlowML Backend" cmd /k "cd backend && call .venv\Scripts\activate.bat && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"
timeout /t 3 /nobreak > nul
echo Starting frontend in new terminal...
start "FlowML Frontend" cmd /k "cd frontend && npm run dev"
echo.
echo All services starting! Check the new terminal windows.
echo.
echo Access points:
echo   - Frontend:    http://localhost:5173
echo   - Backend API: http://localhost:8000
echo   - API Docs:    http://localhost:8000/docs
echo   - MinIO:       http://localhost:9001 (admin / flowml-secret)
echo.
goto end

:stop
echo.
echo Stopping FlowML Infrastructure...
echo =================================
docker-compose -f %COMPOSE_FILE% down
echo Done.
goto end

:logs
echo.
echo Docker Container Logs
echo =====================
docker-compose -f %COMPOSE_FILE% logs -f --tail=100
goto end

:status
echo.
echo FlowML Service Status
echo =====================
echo.
echo Docker containers:
docker-compose -f %COMPOSE_FILE% ps
echo.
echo Checking ports...
netstat -an | findstr ":8000 :5173 :5432 :6379 :9000" 2>nul
echo.
goto end

:end
endlocal
