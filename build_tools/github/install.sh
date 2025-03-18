#!/bin/bash

set -e
set -x

# defines the get_dep and show_installed_libraries functions
source build_tools/github/shared.sh

UNAMESTR=`uname`
CCACHE_LINKS_DIR="/tmp/ccache"

setup_ccache() {
    CCACHE_BIN=`which ccache || echo ""`
    if [[ "${CCACHE_BIN}" == "" ]]; then
        echo "ccache not found, skipping..."
    elif [[ -d "${CCACHE_LINKS_DIR}" ]]; then
        echo "ccache already configured, skipping..."
    else
        echo "Setting up ccache with CCACHE_DIR=${CCACHE_DIR}"
        mkdir ${CCACHE_LINKS_DIR}
        which ccache
        for name in gcc g++ cc c++ clang clang++ i686-linux-gnu-gcc i686-linux-gnu-c++ x86_64-linux-gnu-gcc x86_64-linux-gnu-c++ x86_64-apple-darwin13.4.0-clang x86_64-apple-darwin13.4.0-clang++; do
        ln -s ${CCACHE_BIN} "${CCACHE_LINKS_DIR}/${name}"
        done
        export PATH="${CCACHE_LINKS_DIR}:${PATH}"
        ccache -M 256M

        # Zeroing statistics so that ccache statistics are shown only for this build
        ccache -z
    fi
}

pre_python_environment_install() {
    if [[ "$DISTRIB" == "ubuntu" ]]; then
        sudo apt-get update
        sudo apt-get install python3-scipy python3-matplotlib \
             libatlas3-base libatlas-base-dev python3-virtualenv ccache

    elif [[ "$DISTRIB" == "debian-32" ]]; then
        apt-get update
        apt-get install -y python3-dev python3-numpy python3-scipy \
                python3-matplotlib libopenblas-dev \
                python3-virtualenv python3-pandas ccache git
    fi
}

check_packages_dev_version() {
    for package in $@; do
        package_version=$(python -c "import $package; print($package.__version__)")
        if [[ $package_version =~ "^[.0-9]+$" ]]; then
            echo "$package is not a development version: $package_version"
            exit 1
        fi
    done
}

python_environment_install_and_activate() {
    if [[ "$DISTRIB" == "conda"* ]]; then
        # GitHub Actions requires a different conda activation method
        eval "$(conda shell.bash hook)"
        create_conda_environment_from_lock_file $VIRTUALENV $LOCK_FILE
        conda activate $VIRTUALENV

    elif [[ "$DISTRIB" == "ubuntu" || "$DISTRIB" == "debian-32" ]]; then
        python3 -m virtualenv --system-site-packages --python=python3 $VIRTUALENV
        activate_environment
        pip install -r "${LOCK_FILE}"

    fi

    # Install additional packages on top of the lock-file in specific cases
    if [[ "$DISTRIB" == "conda-free-threaded" ]]; then
        # TODO: we install scipy with pip. When there is a conda-forge package,
        # we can update build_tools/update_environments_and_lock_files.py and
        # remove the line below
        pip install scipy --only-binary :all:
        # TODO: we install cython 3.1 alpha from pip. When there is a conda-forge package,
        # we can update build_tools/update_environments_and_lock_files.py and
        # remove the line below
        pip install --pre cython --only-binary :all:

    elif [[ "$DISTRIB" == "conda-pip-scipy-dev" ]]; then
        echo "Installing development dependency wheels"
        dev_anaconda_url=https://pypi.anaconda.org/scientific-python-nightly-wheels/simple
        pip install --pre --upgrade --timeout=60 --extra-index $dev_anaconda_url scipy
        # Check that we have indeed installed a development version of scipy
        check_packages_dev_version scipy
    fi
}

scikit_learn_install() {
    if [[ "$DISTRIB" == "conda-pip-scipy-dev" ]]; then
        # Need to compile scikit-learn with the same OpenMP version as scipy
        pip install --no-build-isolation -e .
    elif [[ "$DISTRIB" == "conda-free-threaded" ]]; then
        # dependencies specified in pyproject.toml using an isolated build
        # environment:
        pip install --verbose .
    else
        if [[ "$UNAMESTR" == "MINGW64"* ]]; then
           # Needed on Windows CI to compile with Visual Studio compiler
           # otherwise Meson detects a MINGW64 platform and use MINGW64
           # toolchain
           ADDITIONAL_PIP_OPTIONS='-Csetup-args=--vsenv'
        fi
        # Use the pre-installed build dependencies and build directly in the
        # current environment.
        pip install --verbose --no-build-isolation --editable . $ADDITIONAL_PIP_OPTIONS
    fi

    ccache -s || echo "ccache not installed, skipping ccache statistics"
}

main() {
    pre_python_environment_install
    python_environment_install_and_activate
    scikit_learn_install
}

main
