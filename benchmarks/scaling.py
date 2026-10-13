"""Computational-cost validation (single thread for every method, so the
comparison is algorithmic, not about parallelism).

System: (H2)_n dimer chains, N_e = 2n electrons, grid spacing 0.1 bohr
(N_g ~ 44 n). For each size we time ONE density build n[v_eff] (the step
repeated in every SCF iteration) for:
  LRFT-exact     : O(M N_g)  two-sided continued fractions, incl. the mu search
  LRFT-near      : O(M N_g) blocked nearsighted divide-and-conquer (d-dim form)
  KS-tridiag     : LAPACK stebz+stein, O(N_g N_occ) (best case, 1D-only trick)
  KS-dense       : LAPACK evr on the full matrix, O(N_g^3) (cost class of KS in
                   general 3D bases)
and the full SCF wall time for LRFT-exact vs KS-tridiag at a few sizes.
Writes results/scaling.json.
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"

import json
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lrft.core import (Grid, Hartree, exchange, external_potential, ks_density, lrft_fixed_N,
                       lrft_density, scf)
from benchmarks.accuracy import dimer_chain

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
M = 64          # density error ~1e-9 per electron for this system class (accuracy.py, part C)
BLOCK = (200, 480)  # (block, buffer) = (20, 48) bohr; fixed once for all sizes


def best_time(f, repeat=3):
    best, out = np.inf, None
    for _ in range(repeat):
        t = time.perf_counter()
        out = f()
        best = min(best, time.perf_counter() - t)
    return best, out


def model_potential(s, g):
    dens = np.zeros(g.n)
    for X in s.X:
        dens += np.exp(-((g.x - X) ** 2))
    dens *= s.n_elec / (dens.sum() * g.dx)
    return external_potential(s, g) + Hartree(g).potential(dens) + exchange(dens)[1]


def main():
    sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    rows = []
    # warm up JIT
    s = dimer_chain(4)
    g = Grid.around(s, 0.1, 8)
    v = model_potential(s, g)
    lrft_fixed_N(v, g.dx, s.n_elec, M)
    lrft_fixed_N(v, g.dx, s.n_elec, M, window=BLOCK)
    for n in sizes:
        s = dimer_chain(n)
        g = Grid.around(s, 0.1, 8)
        v = model_potential(s, g)
        N = s.n_elec
        row = dict(n_dimers=n, N_e=N, N_g=g.n)
        t_lr, (res, mu, e_lo) = best_time(lambda: lrft_fixed_N(v, g.dx, N, M))
        t_one, _ = best_time(lambda: lrft_density(v, g.dx, mu, e_lo, M))
        row.update(t_lrft=t_lr, t_lrft_single_mu=t_one, mu_evals=round(t_lr / t_one, 1))
        if n <= 2048:
            t_near, (rn, _, _) = best_time(lambda: lrft_fixed_N(v, g.dx, N, M, window=BLOCK), 1)
            row.update(t_lrft_near=t_near,
                       near_err_per_e=float(np.sum(np.abs(rn.density - res.density)) * g.dx / N))
        if n <= 1024:
            t_kt, (nks, ebks, _) = best_time(lambda: ks_density(v, g.dx, N, "tridiag"), 1 if n > 256 else 3)
            row.update(t_ks_tridiag=t_kt,
                       lrft_vs_ks_err_per_e=float(np.sum(np.abs(res.density - nks)) * g.dx / N),
                       lrft_vs_ks_dEband_per_e=float(abs(res.band_energy - ebks) / N))
        if g.n <= 6000:
            t_kd, _ = best_time(lambda: ks_density(v, g.dx, N, "dense"), 1)
            row.update(t_ks_dense=t_kd)
        rows.append(row)
        print(json.dumps({k: (f"{x:.3e}" if isinstance(x, float) else x) for k, x in row.items()}), flush=True)

    # full SCF comparison
    scf_rows = []
    for n in [16, 64, 256]:
        s = dimer_chain(n)
        g = Grid.around(s, 0.1, 8)
        lr = scf(s, g, "lrft", M=M, tol=1e-8)
        ks = scf(s, g, "ks", tol=1e-8)
        scf_rows.append(dict(n_dimers=n, N_e=s.n_elec, t_lrft=lr.time, it_lrft=lr.iterations,
                             t_ks=ks.time, it_ks=ks.iterations, dE_per_e=(lr.energy - ks.energy) / s.n_elec))
        print("SCF", scf_rows[-1], flush=True)

    # empirical exponents (log-log fit over the largest sizes available)
    def slope(key, xkey="N_e"):
        pts = [(r[xkey], r[key]) for r in rows if key in r and r[key] > 1e-3]
        if len(pts) < 3:
            return None
        x, y = np.log([p[0] for p in pts[-4:]]), np.log([p[1] for p in pts[-4:]])
        return float(np.polyfit(x, y, 1)[0])

    fits = {k: slope(k) for k in ("t_lrft", "t_lrft_near", "t_ks_tridiag", "t_ks_dense")}
    print("empirical exponents  t ~ N_e^p :", fits)
    json.dump(dict(M=M, window=BLOCK, rows=rows, scf=scf_rows, exponents=fits),
              open(os.path.join(OUT, "scaling.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
