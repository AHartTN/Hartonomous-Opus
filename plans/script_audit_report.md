# Hartonomous-Opus Script Ecosystem Audit Report

## Executive Summary

This comprehensive audit analyzed **85+ scripts** across the Hartonomous-Opus project, identifying **37 core operational scripts** that require refactoring into singular perfect scripts. The current script ecosystem suffers from significant fragmentation, duplication, and platform-specific inconsistencies.

**Key Findings:**
- **Core operational scripts**: 37 scripts need consolidation into 4 perfect scripts (build, deploy, test, clean)
- **Platform fragmentation**: Windows/Linux/Mac scripts are duplicated with inconsistent behavior
- **Duplication**: Multiple scripts perform similar functions (e.g., 6 different benchmark runners)
- **Dependency issues**: Scripts have inconsistent error handling and environment assumptions
- **Maintenance burden**: 85+ scripts create significant maintenance overhead

## Audit Methodology

Scripts were analyzed from these locations:
- `scripts/` (45+ scripts)
- `deployment/` (15+ scripts)
- `Hartonomous-Orchestrator/` (3 scripts)
- `Hartonomous-Benchmark/scripts/` (6 scripts)
- `sql/deploy/` (5 scripts)
- `cpp/` (2 scripts)

For each script, I documented: purpose, arguments, dependencies, issues, and recommendations.

## Core Operational Scripts Requiring Refactoring

### Build Scripts (Target: 1 unified build script)

| Script | Purpose | Issues | Recommendation |
|--------|---------|--------|----------------|
| `scripts/platforms/linux/build.sh` | Linux-specific C++ build | Hardcoded paths, no error recovery | **MERGE** |
| `scripts/platforms/windows/build.sh` | Windows-specific build | MSVC assumptions, no cross-platform | **MERGE** |
| `scripts/build.sh` | Cross-platform build wrapper | Outdated logic, missing extensions | **PRESERVE as base** |
| `deployment/linux/deploy-postgresql-extensions-hardened.sh` | PostgreSQL extension build | Complex staging, rollback needed | **MERGE** |
| `deployment/build-extensions-with-pg-dev.sh` | Extension build utility | Depends on external packages | **MERGE** |

### Deploy Scripts (Target: 1 unified deploy script)

| Script | Purpose | Issues | Recommendation |
|--------|---------|--------|----------------|
| `scripts/deploy-database.sh` | Database setup and seeding | Complex environment handling | **MERGE** |
| `scripts/deploy-orchestrator.sh` | Orchestrator management | PID file handling, Windows support missing | **MERGE** |
| `deployment/linux/deploy-postgresql-extensions.sh` | Extension deployment | Manual steps, no rollback | **MERGE** |
| `deployment/linux/start-orchestrator.sh` | Orchestrator startup | Environment assumptions | **MERGE** |
| `sql/deploy/build-schema.sh` | Schema deployment | Simple but duplicated | **MERGE** |
| `sql/deploy/setup-database.sh` | Database initialization | Basic functionality | **MERGE** |

### Test Scripts (Target: 1 unified test script)

| Script | Purpose | Issues | Recommendation |
|--------|---------|--------|----------------|
| `scripts/test.sh` | Cross-platform test runner | Limited coverage, no reporting | **PRESERVE as base** |
| `scripts/validate.sh` | Full system validation | PostgreSQL-specific, long runtime | **MERGE** |
| `cpp/run-tests-debug.bat` | Debug test runner | Windows-only, no cross-platform | **MERGE** |
| `cpp/run-tests-release.bat` | Release test runner | Duplicates debug version | **MERGE** |
| `Hartonomous-Orchestrator/test_gateway.py` | Gateway testing | Python-only, no integration | **MERGE** |

### Benchmark Scripts (Related to testing)

| Script | Purpose | Issues | Recommendation |
|--------|---------|--------|----------------|
| `scripts/benchmark.sh` | Unified benchmark runner | Good architecture, needs cleanup | **PRESERVE** |
| `Hartonomous-Benchmark/scripts/run_benchmarks.*` | Platform-specific runners | Triplicated, inconsistent | **ARCHIVE** |
| `Hartonomous-Benchmark/scripts/analyze_results.*` | Result analysis | Triplicated | **MERGE** |

## Detailed Script Analysis

### scripts/ Directory

#### Core Scripts
1. **`benchmark.sh`** - Unified benchmark runner
   - **Purpose**: Cross-platform benchmark execution with platform detection
   - **Args**: --quick, --cpp, --python, --output, --help
   - **Deps**: Hartonomous-Benchmark directory, C++ build
   - **Issues**: Assumes specific build paths, no cleanup
   - **Recommendation**: **PRESERVE** - Good foundation for unified testing

2. **`deploy-database.sh`** - Database deployment
   - **Purpose**: PostgreSQL setup, schema deployment, atom seeding
   - **Args**: --clean, --test, database connection options
   - **Deps**: psql, PostgreSQL extensions, HC_BIN_DIR
   - **Issues**: Complex environment variable handling, assumes specific paths
   - **Recommendation**: **MERGE** into unified deploy script

3. **`deploy-orchestrator.sh`** - Orchestrator management
   - **Purpose**: Start/stop/restart orchestrator with PID management
   - **Args**: --start/--stop/--restart/--status, --port/--host
   - **Deps**: python3, FastAPI requirements, orchestrator directory
   - **Issues**: Windows support incomplete, no daemon mode
   - **Recommendation**: **MERGE** into unified deploy script

4. **`test.sh`** - Test runner
   - **Purpose**: Cross-platform test execution
   - **Args**: --unit/--integration/--cpp/--python/--benchmark
   - **Deps**: C++ build, Python, orchestrator components
   - **Issues**: Limited test discovery, basic reporting
   - **Recommendation**: **PRESERVE as base** for unified testing

5. **`validate.sh`** - System validation
   - **Purpose**: Comprehensive system health check
   - **Args**: --quick
   - **Deps**: PostgreSQL, C++ build, all extensions
   - **Issues**: Very PostgreSQL-specific, long runtime
   - **Recommendation**: **MERGE** validation into unified test script

#### Platform-Specific Scripts
6. **`linux/env.sh`** - Linux environment setup
   - **Purpose**: Configure PostgreSQL, Intel MKL, project paths
   - **Args**: None
   - **Deps**: pg_config, oneAPI paths
   - **Issues**: Assumes specific installation paths
   - **Recommendation**: **MERGE** into unified environment setup

7. **`linux/ingest-models.sh`** - Model ingestion
   - **Purpose**: Scan and ingest ML models from HuggingFace cache
   - **Args**: --path, --list, --model, --type, --threshold, --force
   - **Deps**: Python, C++ tools, database connection
   - **Issues**: Complex path detection, no error recovery
   - **Recommendation**: **PRESERVE** - Specialized functionality

#### Data Processing Scripts
8. **`data/analyze_uncategorized.py`** - Tensor analysis
   - **Purpose**: Identify unknown tensors in safetensor files
   - **Args**: model_directory
   - **Deps**: Python, safetensor files
   - **Issues**: Research tool, not operational
   - **Recommendation**: **ARCHIVE** or move to tools/

9. **Other data scripts** - Various ML data processing tools
   - **Recommendation**: **PRESERVE** - Specialized ML tooling

### deployment/ Directory

#### PostgreSQL Extension Scripts
10. **`build-extensions-with-pg-dev.sh`** - Extension building
    - **Purpose**: Build PostgreSQL extensions from source
    - **Deps**: CMake, PostgreSQL dev package
    - **Issues**: Depends on external package-pg-dev-from-server.sh
    - **Recommendation**: **MERGE** into unified build

11. **`linux/deploy-postgresql-extensions-hardened.sh`** - Hardened deployment
    - **Purpose**: Production-ready extension installation with rollback
    - **Args**: --dry-run, --verbose, --skip-build, staging options
    - **Deps**: sudo, pg_config, PostgreSQL directories
    - **Issues**: Complex staging, assumes specific paths
    - **Recommendation**: **MERGE** into unified deploy

12. **`linux/deploy-postgresql-extensions.sh`** - Basic deployment
    - **Purpose**: Install extensions to PostgreSQL
    - **Deps**: pg_config, extension files
    - **Issues**: No backup/rollback, manual steps
    - **Recommendation**: **ARCHIVE** - replaced by hardened version

13. **`linux/start-orchestrator.sh`** - Orchestrator startup
    - **Purpose**: Linux-specific orchestrator launch
    - **Deps**: Virtual environment, shared libraries
    - **Issues**: Environment assumptions, no daemon mode
    - **Recommendation**: **MERGE** into unified deploy

### Hartonomous-Orchestrator/ Directory

14. **`start.sh`** / **`start.bat`** - Docker orchestration
    - **Purpose**: Launch orchestrator via Docker Compose
    - **Args**: None
    - **Deps**: Docker, docker-compose
    - **Issues**: Simple wrappers, no error handling
    - **Recommendation**: **MERGE** into unified deploy

15. **`test_gateway.py`** - Gateway testing
    - **Purpose**: Comprehensive API testing suite
    - **Deps**: httpx, running orchestrator
    - **Issues**: Requires external services (Llama.cpp, Qdrant)
    - **Recommendation**: **MERGE** into unified test script

### Hartonomous-Benchmark/scripts/ Directory

16. **`run_benchmarks.*`** (3 files) - Benchmark execution
    - **Purpose**: Platform-specific benchmark runners
    - **Issues**: Triplicated code, inconsistent options
    - **Recommendation**: **ARCHIVE** - functionality in scripts/benchmark.sh

17. **`analyze_results.*`** (3 files) - Result analysis
    - **Purpose**: Process benchmark results
    - **Issues**: Triplicated, basic functionality
    - **Recommendation**: **MERGE** into unified benchmark processing

### sql/deploy/ Directory

18. **`build-schema.sh`** / **`build-schema.ps1`** - Schema deployment
    - **Purpose**: Apply database schema files
    - **Args**: Basic database connection options
    - **Deps**: psql/SQL Server
    - **Issues**: Platform-specific, basic error handling
    - **Recommendation**: **MERGE** into unified deploy

19. **`setup-database.sh`** - Database initialization
    - **Purpose**: Create database and enable extensions
    - **Deps**: Database client
    - **Issues**: Simple but duplicated functionality
    - **Recommendation**: **MERGE** into unified deploy

### cpp/ Directory

20. **`run-tests-debug.bat`** / **`run-tests-release.bat`** - Test runners
    - **Purpose**: Windows-specific C++ test execution
    - **Deps**: CTest, Visual Studio build
    - **Issues**: Windows-only, duplicates functionality
    - **Recommendation**: **MERGE** into unified test script

## Script Ecosystem Issues

### 1. Platform Fragmentation
- **Issue**: Scripts duplicated for Windows/Linux/Mac with inconsistent behavior
- **Impact**: Maintenance burden, user confusion, bugs in one platform not fixed in others
- **Solution**: Unified scripts with platform detection

### 2. Inconsistent Dependencies
- **Issue**: Scripts assume different environment setups
- **Impact**: Fragile execution, unclear requirements
- **Solution**: Standardized dependency checking and environment setup

### 3. Poor Error Handling
- **Issue**: Many scripts lack proper error recovery and rollback
- **Impact**: Failed operations leave system in inconsistent state
- **Solution**: Transactional operations with rollback capabilities

### 4. Duplication of Functionality
- **Issue**: Similar operations implemented multiple times
- **Impact**: Code drift, inconsistent behavior
- **Solution**: Single source of truth for each operation

### 5. Missing Documentation
- **Issue**: Many scripts lack usage documentation
- **Impact**: Poor developer experience, incorrect usage
- **Solution**: Comprehensive help systems and documentation

## Recommended Refactoring Plan

### Phase 1: Core Consolidation (Priority: High)
1. **Create unified build script** (`scripts/build.sh`)
   - Merge platform-specific build logic
   - Add PostgreSQL extension building
   - Implement dependency checking

2. **Create unified deploy script** (`scripts/deploy.sh`)
   - Database deployment and seeding
   - Orchestrator management
   - Extension installation with rollback

3. **Create unified test script** (`scripts/test.sh`)
   - Cross-platform test execution
   - System validation
   - Benchmark integration

### Phase 2: Cleanup (Priority: Medium)
4. **Archive redundant scripts**
   - Platform-specific duplicates
   - Basic functionality replaced by unified scripts
   - Research/development tools

5. **Standardize shared utilities**
   - Consistent logging and error handling
   - Platform detection and environment setup

### Phase 3: Enhancement (Priority: Low)
6. **Add comprehensive help system**
   - Usage documentation in all scripts
   - Interactive help and suggestions

7. **Implement rollback capabilities**
   - Database migration rollback
   - File system changes rollback
   - Service management rollback

## Success Criteria

- **Reduced script count**: From 85+ to ~25 core scripts
- **Cross-platform consistency**: Single scripts work on all supported platforms
- **Improved reliability**: Proper error handling and rollback
- **Better maintainability**: Centralized logic, reduced duplication
- **Enhanced usability**: Comprehensive help and clear error messages

## Next Steps

This audit provides the foundation for refactoring. The user should review these findings and approve proceeding to the implementation phase, where I'll create the singular perfect scripts for build, deploy, test, and clean operations.