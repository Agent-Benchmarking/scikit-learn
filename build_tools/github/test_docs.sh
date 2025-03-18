#!/bin/bash

set -e

# Defines the show_installed_libraries and activate_environment functions.
source build_tools/github/shared.sh

activate_environment

cd $TEST_DIR

if [[ "$DISTRIB" == "conda" ]]; then
    # conda install sphinx numpydoc
    echo "sphinx and numpydoc already installed"
else
    # Check that we have sphinx and numpydoc installed
    pip install sphinx numpydoc
fi

# Test that we can build the docs
cd $GITHUB_WORKSPACE/doc
make html
