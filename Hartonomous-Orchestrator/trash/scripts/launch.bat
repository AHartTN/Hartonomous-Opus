@echo off
REM Hartonomous Orchestrator Launch Script for Windows
REM This script provides easy launching and management of the RAG orchestrator

setlocal enabledelayedexpansion

REM Configuration
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
set "CONFIG_FILE=%PROJECT_DIR%\config.yaml"
set "LOG_FILE=%PROJECT_DIR%\logs\orchestrator.log"
set "PID_FILE=%PROJECT_DIR%\orchestrator.pid"

REM Colors for output (Windows)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Logging functions
:log_info
echo [INFO] %~1
goto :eof

:log_success
echo [SUCCESS] %~1
goto :eof

:log_warning
echo [WARNING] %~1
goto :eof

:log_error
echo [ERROR] %~1
goto :eof

REM Check if process is running
:is_running
if exist "%PID_FILE%" (
    set /p PID=<"%PID_FILE%"
    tasklist /FI "PID eq %PID%" 2>NUL | find /I /N "hartonomous-orchestrator">NUL
    if !errorlevel! equ 0 (
        exit /b 0
    ) else (
        del "%PID_FILE%" 2>NUL
        exit /b 1
    )
)
exit /b 1

REM Check dependencies
:check_dependencies
call :log_info "Checking dependencies..."

REM Check if binary exists
if not exist "%PROJECT_DIR%\build\bin\hartonomous-orchestrator.exe" (
    call :log_error "Binary not found. Run 'build.bat' first."
    exit /b 1
)

REM Check config file
if not exist "%CONFIG_FILE%" (
    call :log_error "Config file not found: %CONFIG_FILE%"
    exit /b 1
)

call :log_success "Dependencies check passed"
goto :eof

REM Pre-launch health checks
:health_check
call :log_info "Performing pre-launch health checks..."

REM Check if port is available (simplified check)
for /f "tokens=2 delims=:" %%a in ('findstr "port:" "%CONFIG_FILE%" ^| findstr /n "^" ^| findstr "^1:"') do (
    set "PORT_LINE=%%a"
)
for /f "tokens=2" %%a in ("%PORT_LINE%") do set "PORT=%%a"
set "PORT=%PORT: =%"

netstat -an | find ":%PORT% " >nul 2>&1
if %errorlevel% equ 0 (
    call :log_error "Port %PORT% is already in use"
    exit /b 1
)

REM Check log directory
if not exist "%PROJECT_DIR%\logs" (
    mkdir "%PROJECT_DIR%\logs" 2>NUL
    call :log_info "Created log directory: %PROJECT_DIR%\logs"
)

call :log_success "Health checks passed"
goto :eof

REM Start the orchestrator
:start_service
call :log_info "Starting Hartonomous Orchestrator..."

call :is_running
if !errorlevel! equ 0 (
    call :log_warning "Orchestrator is already running"
    exit /b 1
)

call :check_dependencies
call :health_check

REM Build the command
set "CMD=%PROJECT_DIR%\build\bin\hartonomous-orchestrator.exe"
set "ARGS="

REM Add config file argument if not default
if not "%CONFIG_FILE%"=="%PROJECT_DIR%\config.yaml" (
    set "ARGS=%ARGS% --config %CONFIG_FILE%"
)

call :log_info "Launching: !CMD! !ARGS!"

REM Start in background (Windows equivalent)
start /B "Hartonomous Orchestrator" "!CMD!" !ARGS! > "%LOG_FILE%" 2>&1
timeout /t 2 /nobreak >nul

REM Get PID (simplified - Windows doesn't have easy PID tracking)
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq hartonomous-orchestrator.exe" /FO CSV ^| find "hartonomous-orchestrator.exe"') do (
    set "PID=%%a"
    echo !PID! > "%PID_FILE%"
    goto pid_found
)

:pid_found
if defined PID (
    call :log_success "Orchestrator started successfully (PID: !PID!)"
    call :log_info "Logs: %LOG_FILE%"
    call :log_info "Monitor: type %LOG_FILE%"
) else (
    call :log_error "Failed to start orchestrator"
    if exist "%LOG_FILE%" (
        call :log_info "Check logs for details:"
        type "%LOG_FILE%" | findstr /tail 20
    )
    exit /b 1
)
goto :eof

REM Stop the orchestrator
:stop_service
call :log_info "Stopping Hartonomous Orchestrator..."

call :is_running
if !errorlevel! neq 0 (
    call :log_warning "Orchestrator is not running"
    goto :eof
)

set /p PID=<"%PID_FILE%"

REM Send termination signal
taskkill /PID %PID% /T >nul 2>&1
if !errorlevel! equ 0 (
    call :log_info "Sent termination signal to process %PID%"
) else (
    call :log_warning "Failed to terminate process gracefully"
)

REM Wait for process to end
timeout /t 5 /nobreak >nul

tasklist /FI "PID eq %PID%" 2>NUL | find /I /N "hartonomous-orchestrator">NUL
if !errorlevel! neq 0 (
    del "%PID_FILE%" 2>NUL
    call :log_success "Orchestrator stopped successfully"
) else (
    call :log_error "Failed to stop orchestrator"
    exit /b 1
)
goto :eof

REM Check status
:status_service
call :is_running
if !errorlevel! equ 0 (
    set /p PID=<"%PID_FILE%"
    call :log_success "Orchestrator is running (PID: %PID%)"
    call :log_info "Logs: %LOG_FILE%"
) else (
    call :log_info "Orchestrator is not running"
    if exist "%LOG_FILE%" (
        call :log_info "Last logs:"
        type "%LOG_FILE%" | findstr /tail 5
    )
)
goto :eof

REM Show usage
:usage
echo Hartonomous Orchestrator Management Script for Windows
echo.
echo USAGE:
echo     %0 command [options]
echo.
echo COMMANDS:
echo     start           Start the orchestrator
echo     stop            Stop the orchestrator
echo     restart         Restart the orchestrator
echo     status          Show orchestrator status
echo     logs            Show logs
echo     health          Run health checks
echo.
echo OPTIONS:
echo     -c, --config FILE    Use specific config file
echo     -h, --help           Show this help message
echo.
echo EXAMPLES:
echo     %0 start
echo     %0 stop
echo     %0 status
echo     %0 -c C:\path\to\config.yaml start
goto :eof

REM Main script logic
set "COMMAND=%1"
if "%COMMAND%"=="" set "COMMAND=status"

REM Parse arguments
if "%1"=="-c" (
    set "CONFIG_FILE=%2"
    shift
    shift
    set "COMMAND=%1"
)
if "%1"=="--config" (
    set "CONFIG_FILE=%2"
    shift
    shift
    set "COMMAND=%1"
)
if "%1"=="-h" goto usage
if "%1"=="--help" goto usage

REM Execute command
if "%COMMAND%"=="start" goto start_service
if "%COMMAND%"=="stop" goto stop_service
if "%COMMAND%"=="restart" (
    call :stop_service
    timeout /t 2 /nobreak >nul
    goto start_service
)
if "%COMMAND%"=="status" goto status_service
if "%COMMAND%"=="health" (
    call :check_dependencies
    call :health_check
    goto :eof
)
if "%COMMAND%"=="logs" (
    if exist "%LOG_FILE%" (
        type "%LOG_FILE%"
    ) else (
        call :log_warning "Log file not found: %LOG_FILE%"
    )
    goto :eof
)

call :log_error "Unknown command: %COMMAND%"
goto usage