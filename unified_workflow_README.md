# Unified GitHub Actions Workflows

## Overview

This repository contains updated GitHub Actions workflows for scikit-learn's CI process. The key change is combining the previously separate Ubuntu Jammy and Ubuntu Atlas workflows into a single unified workflow, simplifying dependencies and improving reliability during the migration from Azure Pipelines to GitHub Actions.

## Workflow Files

### 1. `.github/workflows/ubuntu-builds.yml`

This new workflow file combines the functionality previously split between:
- `ubuntu-jammy.yml`
- `ubuntu-atlas.yml`

Key features:
- Explicit job-to-job dependencies using the `needs` keyword
- Job output parameters to share information between jobs
- Clearer workflow commands using `::notice::` and `::group::` syntax
- Consistent environment variable handling

### 2. `.github/workflows/wheels.yml`

This updated workflow now depends on the unified `ubuntu-builds.yml` workflow:
- Simplified dependency checking
- Improved error handling to continue even if Ubuntu builds are still in progress
- Clearer status reporting
- Uses modern cibuildwheel practices

## Dependency Management

The previous approach had these issues:
- Cross-workflow dependencies using GitHub API status checks were unreliable
- Race conditions could occur when workflows started at similar times
- Hard failures when dependent workflows weren't found or hadn't completed

The new approach has these benefits:
- Direct job-to-job dependencies within a single workflow
- Information sharing via job outputs
- Clear status reporting using workflow commands
- Graceful handling of failures in dependent jobs

## Testing Locally

Both workflows can be tested manually via the workflow_dispatch trigger in the GitHub UI.

## Migration Context

These changes are part of the ongoing effort to migrate scikit-learn's CI from Azure Pipelines to GitHub Actions. The unified workflow approach reduces complexity during this transition period and provides a more maintainable solution going forward. 