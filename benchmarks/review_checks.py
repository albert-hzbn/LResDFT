"""Checks prompted by a strict referee report (see docs/REFEREE_REPORT.md).  Single thread, C++ kernels.
Usage: taskset -c K python3 benchmarks/review_checks.py <tol|hop|force|pareto> ; writes results/review_<name>.json
  tol    : KS-LRes energy difference versus the SCF convergence threshold
  hop    : energy jump when atoms change block (Voronoi cell) under a uniform translation of the crystal
  force  : finite-difference forces from E_LRes and E_KS (truncated windows) for one displaced atom
  pareto : density-build time and error versus window radius, at several gaps (self-consistent KS potential)
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
from lrft.lcao3d import KSLCAOCpp, LCAOBasis, LRFTLCAOCpp, cell_centres, scf_lcao  # noqa: E402
from lrft.three_d import h2_crystal, hartree, ion_potential, lda_xc  # noqa: E402

BASIS = dict(rcut=4.5, s_exps=(1.2, 0.4, 0.15), p_exps=(0.5,))
which = sys.argv[1]
RW_ARG = float(sys.argv[2]) if len(sys.argv) > 2 else None
OUT = os.path.join(ROOT, "results", f"review_{which}" + (f"_{RW_ARG}" if RW_ARG else "") + ".json")


def system(n, a=6.0, shift=None, move=None):
    cr = h2_crystal((n, n, n), a=a, pts_per_cell=10, rc=0.6)
    if shift is not None:
        cr.pos = (cr.pos + np.asarray(shift)) % cr.cell
    if move is not None:
        i, d = move
        cr.pos = cr.pos.copy()
        cr.pos[i] = cr.pos[i] + np.asarray(d)
    return cr, LCAOBasis(cr, **BASIS)


def both(cr, b, n, a, Rw, tol):
    ks = scf_lcao(cr, KSLCAOCpp(b), tol=tol, maxiter=120)
    lr = scf_lcao(cr, LRFTLCAOCpp(b, cell_centres((n, n, n), a), R_w=Rw, M=64), tol=tol, maxiter=120)
    return ks, lr


def dump(res):
    json.dump(res, open(OUT, "w"), indent=1)


res = []
if which == "tol":
    for n in (4, 5):
        cr, b = system(n)
        for tol in (1e-5, 1e-6, 1e-8, 1e-10):
            for Rw in (8.5, 12.1):
                ks, lr = both(cr, b, n, 6.0, Rw, tol)
                row = dict(n=n, N_e=cr.n_elec, tol=tol, R_w=Rw, dE_per_e=(lr["energy"] - ks["energy"]) / cr.n_elec,
                           it_ks=ks["iterations"], it_lr=lr["iterations"], conv=[bool(ks["converged"]), bool(lr["converged"])])
                res.append(row)
                print(row, flush=True)
                dump(res)
elif which == "hop":
    n, Rw = 4, (RW_ARG or 8.5)
    for t in np.arange(1.95, 2.45, 0.025 if RW_ARG is None else 0.05):
        cr, b = system(n, shift=[t, 0, 0])
        ks, lr = both(cr, b, n, 6.0, Rw, 1e-9)
        row = dict(t=float(t), dE_per_e=(lr["energy"] - ks["energy"]) / cr.n_elec, E_ks=ks["energy"], E_lr=lr["energy"])
        res.append(row)
        print(row, flush=True)
        dump(res)
elif which == "force":
    n = 4
    h = 0.02
    for Rw in ((RW_ARG,) if RW_ARG else (8.5, 12.1)):
        for atom in (0, 5):
            for ax in range(3):
                E = {}
                for sgn in (+1, -1):
                    d = np.zeros(3)
                    d[ax] = sgn * h
                    cr, b = system(n, move=(atom, d))
                    ks, lr = both(cr, b, n, 6.0, Rw, 1e-8 if RW_ARG else 1e-10)
                    E[sgn] = (ks["energy"], lr["energy"])
                F_ks = -(E[1][0] - E[-1][0]) / (2 * h)
                F_lr = -(E[1][1] - E[-1][1]) / (2 * h)
                row = dict(R_w=Rw, atom=atom, axis=ax, F_ks=F_ks, F_lr=F_lr, dF=F_lr - F_ks)
                res.append(row)
                print(row, flush=True)
                dump(res)
elif which == "pareto":
    n = 6
    for a in (6.0, 5.5, 5.0):
        cr = h2_crystal((n, n, n), a=a, pts_per_cell=int(round(10 * a / 6.0)) if False else 10, rc=0.6)
        b = LCAOBasis(cr, **BASIS)
        ks = KSLCAOCpp(b)
        r = scf_lcao(cr, ks, tol=1e-8, maxiter=120)
        dens = r["density"]
        exc, vxc = lda_xc(dens)
        v = ion_potential(cr) + hartree(cr, dens) + vxc
        tks = []
        for _ in range(2):
            t = time.perf_counter()
            o = ks.density(v)
            tks.append(time.perf_counter() - t)
        nks, w = o[0], np.asarray(o[2])
        nocc = cr.n_elec // 2
        gap = float(w[nocc] - w[nocc - 1])
        for Rw in (6.1, 8.5, 10.4, 12.1, 13.5, 15.0):
            L = LRFTLCAOCpp(b, cell_centres((n, n, n), a), R_w=Rw, M=64)
            t = time.perf_counter()
            o1 = L.density(v)
            tc = time.perf_counter() - t
            t = time.perf_counter()
            o2 = L.density(v)
            tw = time.perf_counter() - t
            row = dict(a=a, N_e=cr.n_elec, gap=gap, R_w=Rw, n_W=float(L.nW.mean()), t_ks=min(tks), t_cold=tc, t_warm=tw,
                       dn=float(np.abs(o2[0] - nks).sum() * cr.dV / cr.n_elec),
                       dEb=float(abs(o2[1] - o[1]) / cr.n_elec))
            res.append(row)
            print({k: (round(v_, 6) if isinstance(v_, float) else v_) for k, v_ in row.items()}, flush=True)
            dump(res)
