#!/bin/bash

set -e
set -x

# Defines the show_installed_libraries and activate_environment functions.
source build_tools/github/shared.sh

activate_environment

# Install codecov
pip install codecov

# Upload coverage report
codecov
