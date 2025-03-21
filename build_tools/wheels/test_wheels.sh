#!/bin/bash

set -e
set -x

PROJECT_DIR="$1"

echo "Testing wheel at $(pwd) with PROJECT_DIR=${PROJECT_DIR}"
echo "Environment variables:"
env | sort

echo "Testing license:"
python $PROJECT_DIR/build_tools/wheels/check_license.py
if [ $? -ne 0 ]; then
    echo "License check failed!"
    exit 1
fi
echo "License check passed!"

python -c "import joblib; print(f'Number of cores (physical): \
{joblib.cpu_count()} ({joblib.cpu_count(only_physical_cores=True)})')"

FREE_THREADED_BUILD="$(python -c"import sysconfig; print(bool(sysconfig.get_config_var('Py_GIL_DISABLED')))")"
if [[ $FREE_THREADED_BUILD == "True" ]]; then
    # TODO: delete when importing numpy no longer enables the GIL
    # setting to zero ensures the GIL is disabled while running the
    # tests under free-threaded python
    export PYTHON_GIL=0
fi

# Test that there are no links to system libraries in the
# threadpoolctl output section of the show_versions output:
echo "Testing sklearn import and show_versions:"
python -c "import sklearn; sklearn.show_versions()"
if [ $? -ne 0 ]; then
    echo "sklearn.show_versions() failed!"
    exit 1
fi
echo "sklearn import test passed!"

echo "Running pytest:"
if pip show -qq pytest-xdist; then
    XDIST_WORKERS=$(python -c "import joblib; print(joblib.cpu_count(only_physical_cores=True))")
    pytest --pyargs sklearn -n $XDIST_WORKERS
else
    pytest --pyargs sklearn
fi
