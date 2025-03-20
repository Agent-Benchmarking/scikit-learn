# CI Migration: Azure Pipelines to GitHub Actions

This document describes the migration of CI jobs from Azure Pipelines to GitHub Actions.

## Migrated Workflows

### 1. Ubuntu Jammy Jellyfish Build

- **Original**: Azure Pipelines `Ubuntu_Jammy_Jellyfish` job
- **New**: GitHub Actions workflow in `.github/workflows/jammy-build.yml`
- **Description**: Builds and tests scikit-learn on Ubuntu 22.04 with Python minimum version and OpenBLAS

### 2. ARM Unit Tests

- **Original**: Previously independent GitHub Actions workflow
- **Updated**: Now depends on the Jammy build
- **Description**: Runs scikit-learn tests on ARM64 architecture

### 3. Wheel Building

- **Original**: Independent GitHub Actions workflow
- **Updated**: Now depends on the Jammy build
- **Description**: Builds wheels for various platforms and Python versions

### 4. CUDA GPU Tests

- **Original**: Independent GitHub Actions workflow
- **Updated**: Now depends on the Jammy build
- **Description**: Runs tests on CUDA GPU hardware

## Dependency Graph

The new dependency graph is as follows:

```
                       ┌─────────────────┐
                       │ Jammy Build     │
                       └─────┬─────┬─────┘
                             │     │
                 ┌───────────┘     └───────────┐
                 │                             │
     ┌───────────▼───────────┐     ┌──────────▼────────────┐
     │ ARM Unit Tests        │     │ Wheel Building        │
     └───────────────────────┘     └──────────┬────────────┘
                                               │
                                   ┌──────────▼────────────┐
                                   │ CUDA GPU Tests        │
                                   └───────────────────────┘
```

## Configuration Details

- The Jammy build uses the same conda lock file as the original Azure Pipelines job
- All workflows maintain the same environment settings as their original counterparts
- The ARM unit tests use the same build script and conda lock file as before
- All workflows include proper skip controls based on commit messages

## Future Work

- Consider migrating more CI jobs from Azure Pipelines to GitHub Actions
- Optimize the build dependencies to reduce redundant work
- Consolidate environment configuration files
