"""3D benchmark: LRFT vs Kohn-Sham for H2 molecular crystals (single thread).

(A) self-consistent accuracy: LCAO-KS vs LCAO-LRFT (identical basis/Hamiltonian),
    plus the basis-set error of the LCAO model vs real-space-grid KS (LOBPCG).
(B) window convergence (nearsightedness) on a 5x5x5 crystal.
(C) cost of one density build n[v] vs system size, n^3 cells, n = 2..8.
(D) shared-memory parallel scaling at N = 432.
All one-body solvers run in the C++ back end (src/lrft_cpp.cpp, system OpenBLAS).
Writes results/three_d.json.
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"

import gc
import json
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lrft.three_d import KSSolver, h2_crystal, hartree, ion_potential, lda_xc, scf3d
from lrft.lcao3d import KSLCAOCpp, LCAOBasis, LRFTLCAOCpp, cell_centres, scf_lcao

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
A, PTS, RC = 6.0, 10, 0.6
BASIS = dict(rcut=4.5, s_exps=(1.2, 0.4, 0.15), p_exps=(0.5,))
RW = 12.1
M = 64


def crystal(n):
    return h2_crystal((n, n, n), a=A, pts_per_cell=PTS, rc=RC)


def guess_potential(cr):
    d0 = np.full(cr.shape, cr.n_elec / (cr.ng * cr.dV))
    return ion_potential(cr) + hartree(cr, d0) + lda_xc(d0)[1]


def part_A():
    rows = []
    for n in (2, 3, 4):
        cr = crystal(n)
        b = LCAOBasis(cr, **BASIS)
        ks = scf_lcao(cr, KSLCAOCpp(b), tol=1e-7)
        lr = scf_lcao(cr, LRFTLCAOCpp(b, cell_centres((n, n, n), A), R_w=RW, M=M), tol=1e-7)
        N = cr.n_elec
        row = dict(n=n, N_e=N, E_KS=ks["energy"], E_LRFT=lr["energy"],
                   dE_per_e=(lr["energy"] - ks["energy"]) / N,
                   dn_per_e=float(np.abs(lr["density"] - ks["density"]).sum() * cr.dV / N),
                   it_KS=ks["iterations"], it_LRFT=lr["iterations"],
                   t_scf_KS=ks["time"], t_scf_LRFT=lr["time"])
        if n <= 2:   # real-space-grid KS (LOBPCG) is too slow beyond this on a shared machine
            g = scf3d(cr, KSSolver(cr, tol=1e-8, maxiter=400), tol=1e-7)
            row.update(E_KS_grid=g.energy, basis_err_per_e=(ks["energy"] - g.energy) / N, t_scf_KS_grid=g.time)
        rows.append(row)
        print("A", json.dumps(row), flush=True)
    return rows


def part_B():
    n = 5
    cr = crystal(n)
    b = LCAOBasis(cr, **BASIS)
    N = cr.n_elec
    ks = scf_lcao(cr, KSLCAOCpp(b), tol=1e-8)
    v = ion_potential(cr) + hartree(cr, ks["density"]) + lda_xc(ks["density"])[1]
    nk, ebk, w = KSLCAOCpp(b).density(v)
    rows = []
    for Rw in (6.1, 8.5, 12.1, 13.5):
        L = LRFTLCAOCpp(b, cell_centres((n, n, n), A), R_w=Rw, M=M)
        nn, eb, mu = L.density(v)
        rows.append(dict(R_w=Rw, n_W=float(L.nW.mean()), dEband_per_e=(eb - ebk) / N,
                         dn_per_e=float(np.abs(nn - nk).sum() * cr.dV / N)))
        print("B", json.dumps(rows[-1]), flush=True)
    return dict(gap=float(w[N // 2] - w[N // 2 - 1]), rows=rows)


def part_C(sizes=(2, 3, 4, 5, 6, 7, 8, 10), ks_max=8):
    """Cost of one density build. KS and the LR-DFT cold step (fresh object) and warm step
    (second call) are each the best of `reps` repetitions; the load average is recorded."""
    rows = []
    for n in sizes:
        cr = crystal(n)
        t = time.perf_counter()
        b = LCAOBasis(cr, **BASIS)
        t_basis = time.perf_counter() - t
        v = guess_potential(cr)
        N = cr.n_elec
        reps = 3 if n <= 7 else 2
        cold, warm = [], []
        for _ in range(reps):
            L = LRFTLCAOCpp(b, cell_centres((n, n, n), A), R_w=RW, M=M)
            t = time.perf_counter()
            L.density(v)
            cold.append(time.perf_counter() - t)
            t = time.perf_counter()
            nl, ebl, mu = L.density(v)
            warm.append(time.perf_counter() - t)
        row = dict(n=n, N_e=N, N_b=b.nb, N_g=cr.ng, n_W=float(L.nW.mean()), t_basis=t_basis,
                   t_lrft_cold=min(cold), t_lrft=min(warm), reps=reps, load=os.getloadavg()[0])
        del L
        gc.collect()
        if n <= ks_max:
            tk = []
            for _ in range(reps):
                t = time.perf_counter()
                nk, ebk, w = KSLCAOCpp(b).density(v)
                tk.append(time.perf_counter() - t)
            row.update(t_ks=min(tk), dEband_per_e=(ebl - ebk) / N,
                       dn_per_e=float(np.abs(nl - nk).sum() * cr.dV / N),
                       gap=float(w[N // 2] - w[N // 2 - 1]))
            del nk
        rows.append(row)
        print("C", json.dumps(row), flush=True)
        del b, v, nl
        gc.collect()
    return rows


def part_D(n=6, threads=(1, 2, 4, 8)):
    """Shared-memory parallel scaling (OpenMP over windows vs threaded LAPACK)."""
    cr = crystal(n)
    b = LCAOBasis(cr, **BASIS)
    v = guess_potential(cr)
    rows = []
    for nt in threads:
        L = LRFTLCAOCpp(b, cell_centres((n, n, n), A), R_w=RW, M=M, nthreads=nt)
        L.density(v)
        t = time.perf_counter()
        L.density(v)
        tl = time.perf_counter() - t
        K = KSLCAOCpp(b, nthreads=nt)
        t = time.perf_counter()
        K.density(v)
        tk = time.perf_counter() - t
        rows.append(dict(threads=nt, t_lrft=tl, t_ks=tk))
        print("D", json.dumps(rows[-1]), flush=True)
    return dict(N_e=cr.n_elec, rows=rows)


def exponent(rows, key):
    pts = [(r["N_e"], r[key]) for r in rows if key in r]
    x, y = np.log([p[0] for p in pts[-4:]]), np.log([p[1] for p in pts[-4:]])
    return float(np.polyfit(x, y, 1)[0])


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ABC"
    res = {}
    path = os.path.join(OUT, "three_d.json")
    if os.path.exists(path):
        res = json.load(open(path))
    if "A" in which:
        res["scf"] = part_A()
    if "B" in which:
        res["window"] = part_B()
    if "C" in which:
        rows = part_C()
        res["cost"] = rows
        res["exponents"] = {k: exponent(rows, k) for k in ("t_lrft", "t_ks")}
        print("exponents", res["exponents"])
    if "D" in which:
        res["threads"] = part_D()
    res["settings"] = dict(a=A, pts_per_cell=PTS, rc=RC, basis=BASIS, R_w=RW, M=M)
    json.dump(res, open(path, "w"), indent=1)
