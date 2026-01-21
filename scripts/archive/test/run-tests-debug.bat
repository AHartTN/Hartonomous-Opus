@echo off
REM Run CTest with Debug configuration
REM Usage: run-tests-debug.bat [additional ctest args]

cd /d "%~dp0build"
if not exist "CTestTestfile.cmake" (
    echo Error: Build directory not found or not configured
    echo Please run cmake first: cmake -B build -S .
    exit /b 1
)

echo Running tests in Debug configuration...
ctest -C Debug --output-on-failure %*
