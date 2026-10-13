#!/bin/sh
# Build the C++ kernels (system OpenBLAS provides BLAS+LAPACK).
set -e
cd "$(dirname "$0")"
g++ -O3 -march=native -fopenmp -shared -fPIC -std=c++17 \
    $(python3 -m pybind11 --includes) src/lrft_cpp.cpp \
    -o lrft/_lrft_cpp$(python3-config --extension-suffix) \
    -L/usr/lib/x86_64-linux-gnu -lopenblas
