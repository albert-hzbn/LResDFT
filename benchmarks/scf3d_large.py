"""Self-consistent 3D calculations in the linear-scaling regime (referee request).

For H2 crystals with 432, 686 and 1024 electrons, run full SCF with
  * KS   : dense generalized eigensolver (C++, dsygvx; the fastest of dsygvx/dsygvd/dsyevr
           for these matrices, see benchmarks/referee_checks.py and the SM),
  * LRFT : nearsighted windows (C++), R_w = 12.1 bohr, M = 64,
and record total energy, density difference, iterations, total wall time, and the time of
every SCF step (so cold first step and warm later steps are both reported).
At 432 electrons, LRFT is also run at R_w = 8.5 and 13.5 bohr (SCF error vs window size).
Single thread; the 1-minute load average of the (shared) machine is recorded.
Writes results/scf3d_large.json.
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"

import json
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lrft.three_d import h2_crystal, hartree, ion_potential, lda_xc
from lrft.lcao3d import KSLCAOCpp, LCAOBasis, LRFTLCAOCpp, cell_centres

OUT = os.path.join(ROOT, "results", "scf3d_large.json")
BASIS = dict(rcut=4.5, s_exps=(1.2, 0.4, 0.15), p_exps=(0.5,))


class Timed:
    """Wrap a solver and record the wall time of every density build."""

    def __init__(self, solver):
        self.s, self.times = solver, []

    def density(self, v):
        t = time.perf_counter()
        out = self.s.density(v)
        self.times.append(time.perf_counter() - t)
        return out


def run_scf(cr, solver, tol=1e-6, maxiter=60, mix=0.4, ndiis=6):
    """Same Pulay SCF as lrft.lcao3d.scf_lcao, but returning per-step times."""
    from lrft.lcao3d import scf_lcao
    w = Timed(solver)
    t = time.perf_counter()
    r = scf_lcao(cr, w, tol=tol, maxiter=maxiter, mix=mix, ndiis=ndiis)
    r["wall"] = time.perf_counter() - t
    r["step_times"] = w.times
    return r


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for n in (6, 7, 8):
        key = f"n{n}"
        if key in res:
            continue
        cr = h2_crystal((n, n, n), a=6.0, pts_per_cell=10, rc=0.6)
        b = LCAOBasis(cr, **BASIS)
        N = cr.n_elec
        load0 = os.getloadavg()[0]
        ks = run_scf(cr, KSLCAOCpp(b))
        lr = run_scf(cr, LRFTLCAOCpp(b, cell_centres((n, n, n), 6.0), R_w=12.1, M=64))
        row = dict(N_e=N, N_b=b.nb, load_avg_start=load0, load_avg_end=os.getloadavg()[0],
                   E_KS=ks["energy"], E_LRFT=lr["energy"], dE_per_e=(lr["energy"] - ks["energy"]) / N,
                   dn_per_e=float(np.abs(lr["density"] - ks["density"]).sum() * cr.dV / N),
                   it_KS=ks["iterations"], it_LRFT=lr["iterations"],
                   wall_KS=ks["wall"], wall_LRFT=lr["wall"],
                   steps_KS=ks["step_times"], steps_LRFT=lr["step_times"])
        if n == 6:
            row["Rw_series"] = []
            for Rw in (8.5, 13.5):
                l2 = run_scf(cr, LRFTLCAOCpp(b, cell_centres((n, n, n), 6.0), R_w=Rw, M=64))
                row["Rw_series"].append(dict(R_w=Rw, dE_per_e=(l2["energy"] - ks["energy"]) / N,
                                             dn_per_e=float(np.abs(l2["density"] - ks["density"]).sum()
                                                            * cr.dV / N),
                                             it=l2["iterations"], wall=l2["wall"]))
        res[key] = row
        json.dump(res, open(OUT, "w"), indent=1)
        print(json.dumps({k: v for k, v in row.items() if not k.startswith("steps")}), flush=True)


if __name__ == "__main__":
    main()
