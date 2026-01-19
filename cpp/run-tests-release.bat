@echo off
REM Run CTest with Release configuration
REM Usage: run-tests-release.bat [additional ctest args]

cd /d "%~dp0build"
if not exist "CTestTestfile.cmake" (
    echo Error: Build directory not found or not configured
    echo Please run cmake first: cmake -B build -S .
    exit /b 1
)

echo Running tests in Release configuration...
ctest -C Release --output-on-failure %*
