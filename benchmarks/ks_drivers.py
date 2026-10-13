"""Kohn-Sham reference: which LAPACK path is fastest for these matrices? (referee request)

For 3D H2 crystals (128-1024 electrons), time one KS density build with
  gvx = dsygvx (bisection + inverse iteration, lowest N/2+1 pairs),
  gvd = dsygvd (divide and conquer, all pairs),
  evr = Cholesky + dsygst + dsyevr (MRRR, lowest N/2+1 pairs),
each with the orbital-sum density and with the density-matrix projection.
Eigensolver and density times are reported separately; best of 3 (best of 2 at 1024).
Also: 1D tridiagonal KS with stebz (bisection + inverse iteration) vs stemr (MRRR).
Writes results/ks_drivers.json.
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"

import json
import sys
import time
import warnings

import numpy as np
from scipy import linalg

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lrft.lcao3d import KSLCAOCpp2, LCAOBasis
from benchmarks.three_d import crystal, guess_potential, BASIS
from lrft.core import Grid, tridiag
from benchmarks.accuracy import dimer_chain
from benchmarks.scaling import model_potential


def main():
    rows3 = []
    for n in (4, 6, 8):
        cr = crystal(n)
        b = LCAOBasis(cr, **BASIS)
        v = guess_potential(cr)
        reps = 2 if n == 8 else 3
        for drv in ("gvx", "gvd", "evr"):
            for dm in (False, True):
                K = KSLCAOCpp2(b, drv, dm)
                best = None
                for _ in range(reps):
                    t = time.perf_counter()
                    K.density(v)
                    tt = time.perf_counter() - t
                    if best is None or tt < best[0]:
                        best = (tt, K.t_eig, K.t_dens)
                rows3.append(dict(N_e=cr.n_elec, N_b=b.nb, driver=drv, density="DM" if dm else "orbitals",
                                  t_total=best[0], t_eig=best[1], t_density=best[2],
                                  load=os.getloadavg()[0]))
                print(json.dumps(rows3[-1]), flush=True)
    rows1 = []
    for nd in (128, 256, 512):
        s = dimer_chain(nd)
        g = Grid.around(s, 0.1, 8)
        d, e = tridiag(model_potential(s, g), g.dx)
        nocc = s.n_elec // 2
        row = dict(N_e=s.n_elec, N_g=g.n)
        for drv in ("stebz", "stemr"):
            best = 1e9
            for _ in range(2):
                t = time.perf_counter()
                linalg.eigh_tridiagonal(d, e, select="i", select_range=(0, nocc - 1), lapack_driver=drv)
                best = min(best, time.perf_counter() - t)
            row[drv] = best
        rows1.append(row)
        print(json.dumps(row), flush=True)
    json.dump(dict(three_d=rows3, one_d=rows1), open(os.path.join(ROOT, "results", "ks_drivers.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
