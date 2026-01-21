#!/bin/bash

# ============================================================================
# Hartonomous-Opus - Orchestrator Deployment Script
# ============================================================================
# Deploys and starts the FastAPI orchestrator with OpenAI-compatible API
#
# Usage:
#   ./scripts/deploy-orchestrator.sh          # Start orchestrator
#   ./scripts/deploy-orchestrator.sh --stop   # Stop orchestrator
#   ./scripts/deploy-orchestrator.sh --restart # Restart orchestrator
#   ./scripts/deploy-orchestrator.sh --status # Check status
# ============================================================================

set -e

# Get script directory and load utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/shared/detect-platform.sh"
source "$SCRIPT_DIR/shared/logging.sh"

# Orchestrator configuration
PLATFORM=$(detect_os)
ORCHESTRATOR_DIR="$PROJECT_ROOT/Hartonomous-Orchestrator"
PORT="${ORCHESTRATOR_PORT:-8700}"
HOST="${ORCHESTRATOR_HOST:-0.0.0.0}"
COMMAND="start"
PID_FILE="$PROJECT_ROOT/orchestrator.pid"
LOG_FILE="$PROJECT_ROOT/logs/orchestrator.log"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --start) COMMAND="start"; shift ;;
        --stop) COMMAND="stop"; shift ;;
        --restart) COMMAND="restart"; shift ;;
        --status) COMMAND="status"; shift ;;
        --port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [command] [options]"
            echo "Commands:"
            echo "  --start     Start orchestrator (default)"
            echo "  --stop      Stop orchestrator"
            echo "  --restart   Restart orchestrator"
            echo "  --status    Check orchestrator status"
            echo "Options:"
            echo "  --port      Port to bind to (default: 8700)"
            echo "  --host      Host to bind to (default: 0.0.0.0)"
            echo "  --help, -h  Show this help"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# Check if orchestrator directory exists
if [ ! -d "$ORCHESTRATOR_DIR" ]; then
    log_error "Orchestrator directory not found: $ORCHESTRATOR_DIR"
    exit 1
fi

cd "$ORCHESTRATOR_DIR"

log_section "Orchestrator Management ($PLATFORM - $COMMAND)"

echo "  Directory:   $ORCHESTRATOR_DIR"
echo "  Port:        $PORT"
echo "  Host:        $HOST"
echo "  PID File:    $PID_FILE"
echo "  Log File:    $LOG_FILE"
echo

# Create logs directory
mkdir -p "$(dirname "$LOG_FILE")"

# ============================================================================
# STATUS CHECK
# ============================================================================

check_status() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            log_info "Orchestrator is running (PID: $pid)"
            return 0
        else
            log_warning "PID file exists but process not running, cleaning up"
            rm -f "$PID_FILE"
        fi
    fi

    log_info "Orchestrator is not running"
    return 1
}

# ============================================================================
# STOP ORCHESTRATOR
# ============================================================================

stop_orchestrator() {
    log_info "Stopping orchestrator..."

    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -TERM "$pid" 2>/dev/null; then
            log_info "Sent SIGTERM to process $pid"

            # Wait for graceful shutdown
            local count=0
            while [ $count -lt 30 ] && kill -0 "$pid" 2>/dev/null; do
                sleep 1
                ((count++))
            done

            if kill -0 "$pid" 2>/dev/null; then
                log_warning "Process still running, sending SIGKILL"
                kill -KILL "$pid" 2>/dev/null || true
                sleep 1
            fi
        else
            log_warning "Could not send signal to process $pid"
        fi

        rm -f "$PID_FILE"
    else
        log_warning "No PID file found"
    fi

    # Also try to kill by port if available
    if command -v lsof &> /dev/null; then
        local port_pid
        port_pid=$(lsof -ti ":$PORT" 2>/dev/null)
        if [ -n "$port_pid" ]; then
            log_info "Killing process using port $PORT (PID: $port_pid)"
            kill -TERM "$port_pid" 2>/dev/null || true
        fi
    fi

    log_success "Orchestrator stopped"
}

# ============================================================================
# START ORCHESTRATOR
# ============================================================================

start_orchestrator() {
    log_info "Starting orchestrator..."

    # Check if already running
    if check_status 2>/dev/null; then
        log_error "Orchestrator is already running"
        exit 1
    fi

    # Check Python availability
    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found. Please install Python 3.9+"
        exit 1
    fi

    # Check requirements
    if [ ! -f "requirements.txt" ]; then
        log_error "requirements.txt not found"
        exit 1
    fi

    # Install/update dependencies
    log_info "Installing Python dependencies..."
    pip3 install -r requirements.txt --quiet

    # Set environment variables
    export ORCHESTRATOR_PORT="$PORT"
    export ORCHESTRATOR_HOST="$HOST"

    # Start orchestrator in background
    log_info "Starting FastAPI server on $HOST:$PORT..."

    if [ "$PLATFORM" = "windows" ]; then
        # Windows-specific startup
        python3 openai_gateway.py > "$LOG_FILE" 2>&1 &
        local pid=$!
    else
        # Unix-like systems
        nohup python3 openai_gateway.py > "$LOG_FILE" 2>&1 &
        local pid=$!
    fi

    # Save PID
    echo $pid > "$PID_FILE"

    # Wait a moment for startup
    sleep 2

    # Check if process is still running
    if kill -0 "$pid" 2>/dev/null; then
        log_success "Orchestrator started successfully (PID: $pid)"
        echo
        echo "API Endpoint: http://localhost:$PORT"
        echo "OpenAI-compatible: http://localhost:$PORT/v1/chat/completions"
        echo "Health check: http://localhost:$PORT/health"
        echo "Logs: $LOG_FILE"
    else
        log_error "Orchestrator failed to start"
        log_info "Check logs: $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
}

# ============================================================================
# RESTART ORCHESTRATOR
# ============================================================================

restart_orchestrator() {
    log_info "Restarting orchestrator..."
    stop_orchestrator
    sleep 2
    start_orchestrator
}

# ============================================================================
# COMMAND EXECUTION
# ============================================================================

case "$COMMAND" in
    start)
        start_orchestrator
        ;;
    stop)
        stop_orchestrator
        ;;
    restart)
        restart_orchestrator
        ;;
    status)
        check_status
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        exit 1
        ;;
esac

log_success "Orchestrator $COMMAND operation completed"