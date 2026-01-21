# Archived Scripts

This directory contains scripts that have been archived as part of the script ecosystem refactoring. These scripts are no longer maintained but preserved for reference.

## Archived Scripts by Category

### Platform-Specific Build Scripts (`platforms/`)
- `build-windows.sh` - Windows-specific build logic, now handled by unified `../build.sh`

### Deployment Scripts (`deploy/`)
- `deploy-database.sh` - Database deployment functionality, merged into `../deploy.sh database`
- `deploy-orchestrator.sh` - Orchestrator management, merged into `../deploy.sh orchestrator`

### Test Scripts (`test/`)
- `validate.sh` - System validation functionality, merged into `../test.sh --system`
- `run-tests-debug.bat` - Windows debug test runner, functionality in `../test.sh`
- `run-tests-release.bat` - Windows release test runner, functionality in `../test.sh`

## Why These Scripts Were Archived

1. **Duplication**: Multiple scripts performed similar functions with inconsistent behavior
2. **Platform Fragmentation**: Scripts were duplicated for Windows/Linux/Mac with maintenance burden
3. **Unified Approach**: New unified scripts provide consistent cross-platform functionality
4. **Better Error Handling**: Unified scripts include rollback and better error recovery

## Migration Guide

| Old Script | New Unified Script | Command |
|------------|-------------------|---------|
| `./scripts/deploy-database.sh` | `./scripts/deploy.sh` | `database` |
| `./scripts/deploy-orchestrator.sh --start` | `./scripts/deploy.sh` | `orchestrator start` |
| `./scripts/validate.sh` | `./scripts/test.sh` | `--system` |
| Platform-specific build scripts | `./scripts/build.sh` | (automatic platform detection) |

## If You Need These Scripts

If you need functionality from an archived script, please:

1. Check if the unified script provides the functionality you need
2. File an issue requesting the feature be added to the unified script
3. As a last resort, you can copy and modify the archived script locally

## Future Cleanup

These archived scripts may be removed entirely in a future major version after confirming no active usage.