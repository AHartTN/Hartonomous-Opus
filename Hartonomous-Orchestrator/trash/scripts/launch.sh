#!/bin/bash

# Hartonomous Orchestrator Launch Script
# This script provides easy launching and management of the RAG orchestrator

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_DIR/config.yaml"
LOG_FILE="$PROJECT_DIR/logs/orchestrator.log"
PID_FILE="$PROJECT_DIR/orchestrator.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if process is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            # PID file exists but process is dead
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."

    # Check if binary exists
    if [ ! -f "$PROJECT_DIR/build/bin/hartonomous-orchestrator" ]; then
        log_error "Binary not found. Run 'make build' first."
        exit 1
    fi

    # Check config file
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi

    log_success "Dependencies check passed"
}

# Pre-launch health checks
health_check() {
    log_info "Performing pre-launch health checks..."

    # Check if port is available
    local port=$(grep "port:" "$CONFIG_FILE" | head -1 | awk '{print $2}' | tr -d ' ')
    if lsof -i :"$port" > /dev/null 2>&1; then
        log_error "Port $port is already in use"
        exit 1
    fi

    # Check log directory
    local log_dir=$(dirname "$LOG_FILE")
    if [ ! -d "$log_dir" ]; then
        mkdir -p "$log_dir"
        log_info "Created log directory: $log_dir"
    fi

    log_success "Health checks passed"
}

# Start the orchestrator
start() {
    log_info "Starting Hartonomous Orchestrator..."

    if is_running; then
        log_warning "Orchestrator is already running (PID: $(cat "$PID_FILE"))"
        exit 1
    fi

    check_dependencies
    health_check

    # Build the command
    local cmd="$PROJECT_DIR/build/bin/hartonomous-orchestrator"
    local args=""

    # Add config file argument if not default
    if [ "$CONFIG_FILE" != "$PROJECT_DIR/config.yaml" ]; then
        args="$args --config $CONFIG_FILE"
    fi

    log_info "Launching: $cmd $args"

    # Start in background
    nohup "$cmd" $args > "$LOG_FILE" 2>&1 &
    local pid=$!
    echo $pid > "$PID_FILE"

    # Wait a moment for startup
    sleep 2

    if is_running; then
        log_success "Orchestrator started successfully (PID: $pid)"
        log_info "Logs: $LOG_FILE"
        log_info "Monitor: tail -f $LOG_FILE"
    else
        log_error "Failed to start orchestrator"
        if [ -f "$LOG_FILE" ]; then
            log_info "Check logs for details:"
            tail -20 "$LOG_FILE"
        fi
        exit 1
    fi
}

# Stop the orchestrator
stop() {
    log_info "Stopping Hartonomous Orchestrator..."

    if ! is_running; then
        log_warning "Orchestrator is not running"
        return 0
    fi

    local pid=$(cat "$PID_FILE")

    # Send SIGTERM first
    kill "$pid"
    log_info "Sent SIGTERM to process $pid"

    # Wait for graceful shutdown
    local count=0
    while is_running && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
    done

    if is_running; then
        log_warning "Process didn't stop gracefully, sending SIGKILL..."
        kill -9 "$pid"
        sleep 1
    fi

    if ! is_running; then
        rm -f "$PID_FILE"
        log_success "Orchestrator stopped successfully"
    else
        log_error "Failed to stop orchestrator"
        exit 1
    fi
}

# Restart the orchestrator
restart() {
    log_info "Restarting Hartonomous Orchestrator..."
    stop
    sleep 2
    start
}

# Check status
status() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        local uptime=$(ps -o etime= -p "$pid" | tr -d ' ')
        log_success "Orchestrator is running (PID: $pid, Uptime: $uptime)"
        log_info "Logs: $LOG_FILE"
    else
        log_info "Orchestrator is not running"
        if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
            log_info "Last logs:"
            tail -5 "$LOG_FILE"
        fi
    fi
}

# Show logs
logs() {
    if [ -f "$LOG_FILE" ]; then
        if [ "$1" = "-f" ] || [ "$1" = "--follow" ]; then
            tail -f "$LOG_FILE"
        else
            local lines=${1:-50}
            tail -"$lines" "$LOG_FILE"
        fi
    else
        log_warning "Log file not found: $LOG_FILE"
    fi
}

# Clean up
cleanup() {
    log_info "Cleaning up..."
    if [ -f "$PID_FILE" ]; then
        rm -f "$PID_FILE"
        log_info "Removed PID file"
    fi
    log_success "Cleanup completed"
}

# Show usage
usage() {
    cat << EOF
Hartonomous Orchestrator Management Script

USAGE:
    $0 <command> [options]

COMMANDS:
    start           Start the orchestrator
    stop            Stop the orchestrator
    restart         Restart the orchestrator
    status          Show orchestrator status
    logs [lines]    Show last logs (default: 50 lines)
    logs -f         Follow logs in real-time
    health          Run health checks
    cleanup         Clean up temporary files

OPTIONS:
    -c, --config FILE    Use specific config file
    -h, --help           Show this help message

EXAMPLES:
    $0 start
    $0 stop
    $0 restart
    $0 status
    $0 logs 100
    $0 logs -f
    $0 -c /path/to/config.yaml start

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

# Main command handling
case ${1:-status} in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        shift
        logs "$@"
        ;;
    health)
        check_dependencies
        health_check
        ;;
    cleanup)
        cleanup
        ;;
    *)
        log_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac