import os

# benchmarks/* pin these to 1 at import time; doing it here first avoids a Numba
# "cannot change NUMBA_NUM_THREADS after launch" error when a test imports them.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"
