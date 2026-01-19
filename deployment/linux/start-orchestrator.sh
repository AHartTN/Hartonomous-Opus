#!/bin/bash
# Hartonomous Orchestrator Startup Script for Linux
# Run this after completing installation steps in INSTALL.md

set -e

echo "============================================"
echo "Hartonomous Orchestrator Startup"
echo "============================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$SCRIPT_DIR/bin"
ORCHESTRATOR_DIR="$SCRIPT_DIR/orchestrator"

# Set LD_LIBRARY_PATH
echo "Setting LD_LIBRARY_PATH to include shared libraries..."
export LD_LIBRARY_PATH="$BIN_DIR:$LD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Check if shared libraries exist
if [ ! -f "$BIN_DIR/libembedding_c.so" ]; then
    echo "ERROR: libembedding_c.so not found in $BIN_DIR"
    echo "Please ensure shared libraries are built and copied to bin directory"
    exit 1
fi

if [ ! -f "$BIN_DIR/libgenerative_c.so" ]; then
    echo "ERROR: libgenerative_c.so not found in $BIN_DIR"
    exit 1
fi

if [ ! -f "$BIN_DIR/libhypercube_c.so" ]; then
    echo "ERROR: libhypercube_c.so not found in $BIN_DIR"
    exit 1
fi

echo "Shared libraries found: OK"
echo ""

# Check if virtual environment exists
if [ ! -f "$ORCHESTRATOR_DIR/venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found"
    echo "Please run:"
    echo "  cd orchestrator"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if .env exists
if [ ! -f "$ORCHESTRATOR_DIR/.env" ]; then
    echo "WARNING: .env file not found in orchestrator directory"
    echo "Please copy .env.template and configure:"
    echo "  cp ../shared/config/.env.template orchestrator/.env"
    echo "  nano orchestrator/.env"
    read -p "Press Enter to continue or Ctrl+C to abort..."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$ORCHESTRATOR_DIR/venv/bin/activate"
echo ""

# Change to orchestrator directory
cd "$ORCHESTRATOR_DIR"

# Start the orchestrator
echo "Starting Hartonomous Orchestrator..."
echo ""
echo "============================================"
echo "Server will start on http://0.0.0.0:8700"
echo "Press Ctrl+C to stop"
echo "============================================"
echo ""

python3 openai_gateway.py

# If we get here, server stopped
echo ""
echo "Orchestrator stopped."
