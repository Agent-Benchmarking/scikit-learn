#!/bin/bash

set -e

# We need to install miniforge on GitHub Actions runners
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
wget ${MINIFORGE_URL} -O miniforge.sh
bash miniforge.sh -b -u -p $CONDA_PATH

# Set conda path for subsequent steps
export PATH=$CONDA_PATH/bin:$PATH

# Initialize conda for bash shell
source $CONDA_PATH/etc/profile.d/conda.sh

# Create a test environment
conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y

# Activate the test environment
conda activate $CONDA_ENV_NAME

# Install build dependencies
conda install -y numpy scipy cython joblib threadpoolctl pytest pytest-xdist pytest-timeout

# Install additional dependencies using pip
pip install -e .

# Show the conda info and list of packages for debugging
conda info
conda list 