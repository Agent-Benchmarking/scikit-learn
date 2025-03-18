# Fork Testing Configuration

This document describes the changes made to enable testing the CI workflows on a fork.

## Updates to CI Workflows

To enable testing in the `Agent-Benchmarking/scikit-learn` fork, the following changes have been made:

### Repository Checks

All repository checks have been updated from `scikit-learn/scikit-learn` to `Agent-Benchmarking/scikit-learn` in the following files:

1. `.github/workflows/arm-unit-tests.yml`
2. `.github/workflows/wheels.yml`
3. `.github/workflows/check-sdist.yml`
4. `.github/workflows/labeler-module.yml`
5. `.github/workflows/artifact-redirector.yml`
6. `.github/workflows/update-lock-files.yml`
7. `.github/workflows/update_tracking_issue.yml`

### API URLs

- Updated the GitHub API URL in `update-lock-files.yml` to point to `Agent-Benchmarking/scikit-learn`
- Updated the fork reference for creating pull requests to `Agent-Benchmarking-bot/scikit-learn`

## Testing Instructions

1. Push these changes to your fork
2. Verify that GitHub Actions workflows are enabled in your repository settings
3. Trigger the workflows manually or through a push
4. Monitor the workflow runs in the "Actions" tab

## Reverting Changes

If you need to submit a pull request to the main scikit-learn repository, you'll need to revert these changes to use the original repository name.

## Notes

- The ARM and CUDA tests may require specific hardware that might not be available in your GitHub Actions account
- Some workflows require secrets which may not be configured in your fork
