#!/bin/bash

set -e

# Install system packages
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    libatlas-base-dev \
    gfortran \
    libopenblas-dev

# Install Python dependencies
pip3 install numpy scipy cython joblib threadpoolctl
pip3 install pytest pytest-xdist pytest-timeout

# Install scikit-learn
pip3 install -e .

# Show the installed packages for debugging
pip3 list 