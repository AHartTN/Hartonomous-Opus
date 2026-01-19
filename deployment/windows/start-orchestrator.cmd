@echo off
REM Hartonomous Orchestrator Startup Script for Windows
REM Run this after completing installation steps in INSTALL.md

echo ============================================
echo Hartonomous Orchestrator Startup
echo ============================================
echo.

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
set BIN_DIR=%SCRIPT_DIR%bin
set ORCHESTRATOR_DIR=%SCRIPT_DIR%orchestrator

REM Add DLLs to PATH
echo Setting PATH to include DLLs...
set PATH=%BIN_DIR%;%PATH%
echo PATH: %BIN_DIR%
echo.

REM Check if DLLs exist
if not exist "%BIN_DIR%\embedding_c.dll" (
    echo ERROR: embedding_c.dll not found in %BIN_DIR%
    echo Please ensure DLLs are copied to bin directory
    pause
    exit /b 1
)

if not exist "%BIN_DIR%\generative_c.dll" (
    echo ERROR: generative_c.dll not found in %BIN_DIR%
    pause
    exit /b 1
)

if not exist "%BIN_DIR%\hypercube_c.dll" (
    echo ERROR: hypercube_c.dll not found in %BIN_DIR%
    pause
    exit /b 1
)

echo DLLs found: OK
echo.

REM Check if virtual environment exists
if not exist "%ORCHESTRATOR_DIR%\venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found
    echo Please run: cd orchestrator && python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check if .env exists
if not exist "%ORCHESTRATOR_DIR%\.env" (
    echo WARNING: .env file not found in orchestrator directory
    echo Please copy .env.template and configure:
    echo   copy ..\shared\config\.env.template orchestrator\.env
    echo   notepad orchestrator\.env
    pause
)

REM Activate virtual environment
echo Activating virtual environment...
call "%ORCHESTRATOR_DIR%\venv\Scripts\activate.bat"
echo.

REM Change to orchestrator directory
cd /d "%ORCHESTRATOR_DIR%"

REM Start the orchestrator
echo Starting Hartonomous Orchestrator...
echo.
echo ============================================
echo Server will start on http://0.0.0.0:8700
echo Press Ctrl+C to stop
echo ============================================
echo.

python openai_gateway.py

REM If we get here, server stopped
echo.
echo Orchestrator stopped.
pause
