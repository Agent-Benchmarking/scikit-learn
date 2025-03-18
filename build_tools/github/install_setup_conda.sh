#!/bin/bash

set -e
set -x

# Install Miniconda
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
wget $MINICONDA_URL -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Configure conda
conda config --set always_yes yes --set changeps1 no
conda update -q conda

# Make conda activate command available
echo "conda activate" > ~/.bashrc
