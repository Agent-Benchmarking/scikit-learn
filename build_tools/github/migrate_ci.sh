#!/bin/bash
# Script to help with the migration from Azure Pipelines to GitHub Actions

set -e
set -x

echo "Migrating CI from Azure Pipelines to GitHub Actions"

# Enable workflows
for workflow in .github/workflows/jammy-build.yml .github/workflows/arm-unit-tests.yml .github/workflows/wheels.yml .github/workflows/cuda-ci.yml; do
  echo "Enabling workflow: $workflow"
  git add "$workflow"
done

# Add README and documentation
git add .github/MIGRATED_CI.md
git add .github/FORK_TESTING.md

echo "=========================="
echo "Migration complete!"
echo "Please review the changes and commit them."
echo "=========================="
echo ""
echo "Migration includes:"
echo "1. New jammy-build.yml workflow (replaces Ubuntu_Jammy_Jellyfish)"
echo "2. Updated arm-unit-tests.yml to depend on jammy-build"
echo "3. Updated wheels.yml to depend on jammy-build"
echo "4. Updated cuda-ci.yml to depend on jammy-build"
echo "5. Added documentation in MIGRATED_CI.md"
echo "6. Updated repository checks to Agent-Benchmarking/scikit-learn (see FORK_TESTING.md)"
echo ""
echo "To test changes locally, use 'act -j jammy_build' (requires act: https://github.com/nektos/act)"
