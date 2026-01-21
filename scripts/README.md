# Hartonomous-Opus Scripts

This directory contains organized, cross-platform scripts for building, testing, deploying, and managing the Hartonomous-Opus project.

## Directory Structure

```
scripts/
├── build.sh              # Unified cross-platform build script
├── deploy.sh             # Unified deployment script (database + orchestrator)
├── test.sh               # Unified test runner with system validation
├── clean.sh              # Unified cleanup script
├── benchmark.sh          # Performance benchmarking
├── data/                 # Data processing scripts
│   ├── catalog_models.py
│   ├── shape_based_extraction.py
│   └── ...
├── shared/               # Shared utilities
│   ├── detect-platform.sh
│   ├── logging.sh
│   ├── requirements-check.sh
│   └── ...
├── archive/              # Archived legacy scripts
│   ├── README.md
│   ├── platforms/
│   ├── deploy/
│   └── test/
└── README.md            # This file
```

## Quick Start

### Build the Project
```bash
# Build C++ components and install PostgreSQL extensions
./scripts/build.sh

# Debug build
./scripts/build.sh --debug

# Build without installing extensions
./scripts/build.sh --no-install
```

### Run Tests
```bash
# Run all tests
./scripts/test.sh

# Run only unit tests
./scripts/test.sh --unit

# Run system validation only
./scripts/test.sh --system

# Run only C++ tests
./scripts/test.sh --cpp
```

### Deploy Database and Orchestrator
```bash
# Setup database only
./scripts/deploy.sh database

# Manage orchestrator
./scripts/deploy.sh orchestrator start
./scripts/deploy.sh orchestrator status
./scripts/deploy.sh orchestrator stop

# Full deployment (database + orchestrator)
./scripts/deploy.sh all

# Install PostgreSQL extensions only
./scripts/deploy.sh extensions
```

### Clean Project
```bash
# Safe cleanup (preserves config)
./scripts/clean.sh

# Clean build artifacts only
./scripts/clean.sh --build

# Clean everything (with confirmation)
./scripts/clean.sh --all
```

### Run Benchmarks
```bash
# Run performance benchmarks
./scripts/benchmark.sh

# Quick benchmark run
./scripts/benchmark.sh --quick
```

## Configuration

Most scripts use environment variables for configuration. Copy and modify `scripts/config.env.example` to set your preferences:

```bash
# Database connection
HC_DB_HOST=your-postgres-server
HC_DB_NAME=hypercube
HC_DB_USER=postgres

# Build configuration
HC_BUILD_TYPE=Release

# Orchestrator settings
ORCHESTRATOR_PORT=8700
```

## Platform Support

Scripts automatically detect the platform and load appropriate platform-specific implementations:

- **Linux**: Uses GCC/Clang, system package managers, standard Unix tools
- **macOS**: Uses Clang, Homebrew, Unix tools
- **Windows**: Uses MSVC (via Visual Studio), PowerShell, Windows tools

Platform-specific code is isolated in `platforms/{platform}/` directories.

## Development Utilities

Additional development utilities are available in subdirectories:

- **utils/**: Code formatting, linting, documentation generation
- **shared/**: Common functions used by multiple scripts

## Troubleshooting

### Common Issues

1. **"Permission denied" on Linux/macOS**
   ```bash
   chmod +x scripts/*.sh
   ```

2. **PostgreSQL connection failed**
   - Ensure PostgreSQL is running
   - Check `HC_DB_*` environment variables
   - Verify firewall settings

3. **Build fails on Windows**
   - Run from Developer Command Prompt
   - Ensure Visual Studio is installed
   - Check Intel MKL installation

4. **Python dependencies missing**
   ```bash
   pip3 install -r requirements.txt
   ```

### Debug Mode

Run scripts with verbose logging:

```bash
# Enable debug logging
export HC_LOG_LEVEL=DEBUG

# Run with verbose output
./scripts/build.sh --verbose
```

### Manual Execution

If automatic platform detection fails, you can run platform-specific scripts directly:

```bash
# Linux-specific build
./scripts/platforms/linux/build.sh

# Windows-specific test
./scripts/platforms/windows/test.sh
```

## Contributing

When adding new scripts:

1. Use the shared utilities in `shared/` for common functionality
2. Add platform-specific code to appropriate `platforms/{platform}/` subdirectories
3. Follow naming conventions: `action-description.sh` or `action-description.py`
4. Include help text with `--help` option
5. Test on all supported platforms

## Legacy Scripts

The old scattered scripts in various directories are deprecated but remain for backward compatibility. New development should use the organized scripts in this directory.