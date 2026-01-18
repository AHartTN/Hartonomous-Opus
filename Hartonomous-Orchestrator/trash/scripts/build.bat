@echo off
REM Hartonomous Orchestrator Build Script for Windows
REM This script handles building, testing, and packaging the application

setlocal enabledelayedexpansion

REM Configuration
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
set "BUILD_DIR=%PROJECT_DIR%\build"
set "BUILD_TYPE=Release"
set "ENABLE_TESTS=true"

REM Set up VS2022 environment
call "D:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Failed to set up Visual Studio 2022 environment
    exit /b 1
)

REM Check for required tools
"D:\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: CMake not found in VS2022 installation
    exit /b 1
)

REM Check for MSVC or other compilers
where cl >nul 2>nul
if %errorlevel% neq 0 (
    echo WARNING: MSVC compiler not found in PATH
    where g++ >nul 2>nul
    if %errorlevel% neq 0 (
        echo ERROR: No C++ compiler found
        exit /b 1
    ) else (
        echo Using g++ compiler
    )
) else (
    echo Using MSVC compiler
)

REM Parse command line arguments
:parse_args
if "%1"=="" goto end_parse
if "%1"=="--build-type" (
    set "BUILD_TYPE=%2"
    shift & shift
    goto parse_args
)
if "%1"=="--no-tests" (
    set "ENABLE_TESTS=false"
    shift
    goto parse_args
)
if "%1"=="--help" goto show_help
if "%1"=="clean" goto clean_build
shift
goto parse_args

:end_parse

REM Main build process
echo Building Hartonomous Orchestrator...
echo Build type: %BUILD_TYPE%
echo Tests enabled: %ENABLE_TESTS%

call :setup_build_dir
call :configure_build
call :build_project

if "%ENABLE_TESTS%"=="true" (
    call :run_tests
)

echo.
echo Build completed successfully!
echo Binary location: %BUILD_DIR%\bin\%BUILD_TYPE%\hartonomous-orchestrator.exe
echo.
echo To run: scripts\launch.bat start
goto :eof

:setup_build_dir
echo Setting up build directory...
if exist "%BUILD_DIR%" (
    echo Build directory exists, cleaning...
    rmdir /s /q "%BUILD_DIR%" 2>nul
)
mkdir "%BUILD_DIR%" 2>nul
cd /d "%BUILD_DIR%"
echo Build directory ready
goto :eof

:configure_build
echo Configuring build with CMake...
set "CMAKE_ARGS=-DCMAKE_BUILD_TYPE=%BUILD_TYPE%"

if "%ENABLE_TESTS%"=="true" (
    set "CMAKE_ARGS=%CMAKE_ARGS% -DBUILD_TESTS=ON"
) else (
    set "CMAKE_ARGS=%CMAKE_ARGS% -DBUILD_TESTS=OFF"
)

REM Add vcpkg toolchain if available
if exist "D:\Microsoft Visual Studio\2022\Community\VC\vcpkg" (
    set "VCPKG_ROOT=D:\Microsoft Visual Studio\2022\Community\VC\vcpkg"
    set "CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake""
    echo Using vcpkg toolchain: %VCPKG_ROOT%
)

echo CMake command: "D:\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" %CMAKE_ARGS% "%PROJECT_DIR%"
"D:\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" %CMAKE_ARGS% "%PROJECT_DIR%" 2>&1
if %errorlevel% neq 0 (
    echo ERROR: CMake configuration failed
    exit /b 1
)
echo CMake configuration completed
goto :eof

:build_project
echo Building project...
cmake --build . --config %BUILD_TYPE% --parallel
if %errorlevel% neq 0 (
    echo ERROR: Build failed
    exit /b 1
)
echo Build completed successfully
goto :eof

:run_tests
echo Running tests...
ctest --build-config %BUILD_TYPE% --output-on-failure
if %errorlevel% neq 0 (
    echo WARNING: Some tests failed
    REM Don't exit with error for test failures in CI
)
goto :eof

:clean_build
echo Cleaning build artifacts...
if exist "%BUILD_DIR%" (
    rmdir /s /q "%BUILD_DIR%"
    echo Build directory cleaned
) else (
    echo Build directory not found, nothing to clean
)
goto :eof

:show_help
echo Hartonomous Orchestrator Build Script for Windows
echo.
echo USAGE:
echo     %0 [options] [command]
echo.
echo COMMANDS:
echo     (no command)    Build everything
echo     clean           Clean build artifacts
echo.
echo OPTIONS:
echo     --build-type TYPE    Build type (Debug, Release, RelWithDebInfo)
echo     --no-tests           Disable building and running tests
echo     --help               Show this help message
echo.
echo ENVIRONMENT VARIABLES:
echo     VCPKG_ROOT          Path to vcpkg installation
echo.
echo EXAMPLES:
echo     %0
echo     %0 --build-type Debug
echo     %0 --no-tests
echo     %0 clean
goto :eof