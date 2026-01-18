#!/bin/bash

# Hartonomous Orchestrator Health Check Script
# Performs comprehensive health checks on the orchestrator and its dependencies

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_DIR/config.yaml}"
VERBOSE="${VERBOSE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Global health status
OVERALL_HEALTH=true

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
    OVERALL_HEALTH=false
}

log_verbose() {
    if [ "$VERBOSE" = "true" ]; then
        echo -e "${BLUE}[VERBOSE]${NC} $1"
    fi
}

# Check if a port is open
check_port() {
    local host=$1
    local port=$2
    local timeout=${3:-5}

    if timeout "$timeout" bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Check HTTP endpoint
check_http_endpoint() {
    local url=$1
    local expected_code=${2:-200}
    local timeout=${3:-10}

    local response
    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$timeout" "$url" 2>/dev/null || echo "000")

    if [ "$response" = "$expected_code" ]; then
        return 0
    else
        return 1
    fi
}

# Check orchestrator health
check_orchestrator() {
    log_info "Checking orchestrator health..."

    # Check if process is running
    local pid_file="$PROJECT_DIR/orchestrator.pid"
    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            log_success "Orchestrator process is running (PID: $pid)"
        else
            log_error "Orchestrator PID file exists but process is not running"
            return 1
        fi
    else
        log_warning "Orchestrator PID file not found - process may not be running"
        return 1
    fi

    # Check HTTP endpoint
    local port
    port=$(grep "port:" "$CONFIG_FILE" | head -1 | awk '{print $2}' | tr -d ' ')
    local host="localhost"

    if check_port "$host" "$port"; then
        log_verbose "Orchestrator port $port is open"
    else
        log_error "Orchestrator port $port is not accessible"
        return 1
    fi

    # Check health endpoint
    local health_url="http://$host:$port/health"
    if check_http_endpoint "$health_url" 200; then
        log_success "Orchestrator health endpoint responding"
    else
        log_error "Orchestrator health endpoint not responding"
        return 1
    fi

    # Check metrics endpoint if configured
    local metrics_port
    metrics_port=$(grep "metrics:" -A 10 "$CONFIG_FILE" | grep "port:" | head -1 | awk '{print $2}' | tr -d ' ' || echo "")
    if [ -n "$metrics_port" ] && [ "$metrics_port" != "null" ]; then
        local metrics_url="http://$host:$metrics_port/metrics"
        if check_http_endpoint "$metrics_url" 200; then
            log_success "Metrics endpoint responding"
        else
            log_warning "Metrics endpoint not responding"
        fi
    fi

    return 0
}

# Check service dependencies
check_services() {
    log_info "Checking service dependencies..."

    # Parse service configurations from config
    local services=("embedding" "reranking" "generative" "vector_db")

    for service in "${services[@]}"; do
        log_verbose "Checking $service service..."

        # Extract endpoint from config
        local endpoint
        endpoint=$(grep "$service:" -A 5 "$CONFIG_FILE" | grep "endpoint:" | head -1 | awk '{print $2}' | tr -d ' ' || echo "")

        if [ -z "$endpoint" ] || [ "$endpoint" = "null" ]; then
            log_warning "$service service endpoint not configured"
            continue
        fi

        # Parse host and port from endpoint
        local host port
        if [[ $endpoint =~ http://([^:]+):([0-9]+) ]]; then
            host="${BASH_REMATCH[1]}"
            port="${BASH_REMATCH[2]}"
        else
            log_warning "Could not parse endpoint for $service: $endpoint"
            continue
        fi

        # Check if service is reachable
        if check_port "$host" "$port" 3; then
            log_success "$service service is reachable at $endpoint"

            # Try to check service health if it has a health endpoint
            local health_url="$endpoint/health"
            if check_http_endpoint "$health_url" 200 3; then
                log_verbose "$service service health check passed"
            else
                log_verbose "$service service health check failed or not available"
            fi
        else
            log_error "$service service is not reachable at $endpoint"
        fi
    done
}

# Check system resources
check_system_resources() {
    log_info "Checking system resources..."

    # Check CPU usage
    local cpu_usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    if (( $(echo "$cpu_usage < 90" | bc -l) )); then
        log_success "CPU usage: ${cpu_usage}%"
    else
        log_warning "High CPU usage: ${cpu_usage}%"
    fi

    # Check memory usage
    local mem_info
    mem_info=$(free | grep Mem)
    local mem_used_percent
    mem_used_percent=$(echo "$mem_info" | awk '{printf "%.1f", $3/$2 * 100.0}')

    if (( $(echo "$mem_used_percent < 90" | bc -l) )); then
        log_success "Memory usage: ${mem_used_percent}%"
    else
        log_warning "High memory usage: ${mem_used_percent}%"
    fi

    # Check disk space
    local disk_usage
    disk_usage=$(df "$PROJECT_DIR" | tail -1 | awk '{print $5}' | sed 's/%//')

    if [ "$disk_usage" -lt 90 ]; then
        log_success "Disk usage: ${disk_usage}%"
    else
        log_warning "High disk usage: ${disk_usage}%"
    fi
}

# Check configuration
check_configuration() {
    log_info "Checking configuration..."

    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        return 1
    fi

    log_success "Configuration file exists"

    # Basic YAML syntax check
    if command -v python3 &> /dev/null; then
        if python3 -c "import yaml; yaml.safe_load(open('$CONFIG_FILE'))" 2>/dev/null; then
            log_success "Configuration file has valid YAML syntax"
        else
            log_error "Configuration file has invalid YAML syntax"
            return 1
        fi
    else
        log_verbose "Python3 not available for YAML validation"
    fi

    # Check required configuration sections
    local required_sections=("server" "services")
    for section in "${required_sections[@]}"; do
        if grep -q "^$section:" "$CONFIG_FILE"; then
            log_verbose "Required section '$section' found"
        else
            log_error "Required section '$section' missing from configuration"
        fi
    done
}

# Generate health report
generate_report() {
    log_info "Generating health report..."

    echo "========================================"
    echo "Hartonomous Orchestrator Health Report"
    echo "========================================"
    echo "Timestamp: $(date)"
    echo "Configuration: $CONFIG_FILE"
    echo ""

    if [ "$OVERALL_HEALTH" = "true" ]; then
        echo -e "${GREEN}OVERALL STATUS: HEALTHY${NC}"
    else
        echo -e "${RED}OVERALL STATUS: UNHEALTHY${NC}"
    fi

    echo ""
    echo "========================================"
}

# Show usage
usage() {
    cat << EOF
Hartonomous Orchestrator Health Check Script

USAGE:
    $0 [options]

OPTIONS:
    -c, --config FILE    Configuration file path [default: ../config.yaml]
    -v, --verbose        Enable verbose output
    -h, --help           Show this help message

EXAMPLES:
    $0
    $0 --config /path/to/config.yaml
    $0 --verbose

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
log_info "Starting comprehensive health check..."

check_configuration
echo ""
check_system_resources
echo ""
check_services
echo ""
check_orchestrator
echo ""

generate_report

# Exit with appropriate code
if [ "$OVERALL_HEALTH" = "true" ]; then
    log_success "All health checks passed"
    exit 0
else
    log_error "Some health checks failed"
    exit 1
fi