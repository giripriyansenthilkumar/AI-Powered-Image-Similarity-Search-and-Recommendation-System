@echo off
setlocal

:: Root of the project (this file sits in the project root)
set "ROOT=%~dp0"

:: Change this to your Conda env name if different
set "ENV_NAME=env"

:: Backend settings (reload only backend dir for faster reloads)
set "BACKEND_CMD=uvicorn backend.main:app --reload --reload-dir backend --host 0.0.0.0 --port 8000"

:: Frontend settings (simple static server)
set "FRONTEND_CMD=python -m http.server 8001"

:: Start backend in a new window (run from root so package imports resolve)
start "Backend" cmd /k "cd /d %ROOT% && conda activate %ENV_NAME% && set PYTHONPATH=%ROOT% && %BACKEND_CMD%"

:: Small pause to avoid concurrent temp-file contention
timeout /t 2 /nobreak >nul

:: Start frontend in a new window (no conda activation required)
start "Frontend" cmd /k "cd /d %ROOT%frontend && %FRONTEND_CMD%"

:: Optional: open the app in the browser after a short delay
:: timeout /t 2 >nul
:: start http://localhost:8001

endlocal
