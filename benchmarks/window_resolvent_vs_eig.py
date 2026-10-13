"""Cost of one window: eigendecomposition (as used) versus M/2 = 32 complex shifted solves (the 'resolvent' route).
3D H2 crystal, 6x6x6 cells, self-consistent potential; window of the central block at several radii.
Both routes give the same owned-block density matrix (checked).  Single thread.  Writes results/window_resolvent.json."""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"
import json
import sys
import time
import warnings

import numpy as np
import scipy.sparse.linalg as spla
from scipy import linalg, sparse

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lrft.lcao3d import KSLCAOCpp, LCAOBasis, LRFTLCAO, cell_centres, scf_lcao  # noqa: E402
from lrft.three_d import h2_crystal, hartree, ion_potential, lda_xc  # noqa: E402

n, a, M = 6, 6.0, 64
cr = h2_crystal((n, n, n), a=a, pts_per_cell=10, rc=0.6)
b = LCAOBasis(cr, rcut=4.5, s_exps=(1.2, 0.4, 0.15), p_exps=(0.5,))
r = scf_lcao(cr, KSLCAOCpp(b), tol=1e-7, maxiter=60)
dens = r["density"]
v = ion_potential(cr) + hartree(cr, dens) + lda_xc(dens)[1]
H, S = b.H(v).tocsr(), b.S.tocsr()
w = np.sort(linalg.eigh(H.toarray(), S.toarray(), eigvals_only=True, subset_by_index=[0, cr.n_elec // 2]))
nocc = cr.n_elec // 2
mu = 0.5 * (w[nocc - 1] + w[nocc])
e_lo = w[0] - 0.3
c, R = 0.5 * (mu + e_lo), 0.5 * (mu - e_lo)
zeta = np.exp(1j * np.pi * (2 * np.arange(M // 2) + 1) / M)         # M/2 nodes in the upper half plane
zs, ws = c + R * zeta, R * zeta / M
res = []
for Rw in (6.1, 8.5, 12.1, 13.5):
    L = LRFTLCAO(b, cell_centres((n, n, n), a), R_w=Rw, M=M)
    B = n ** 3 // 2 + n * n // 2 + n // 2                           # a block near the centre
    B = int(np.argmin(np.linalg.norm(cell_centres((n, n, n), a) - 0.5 * a * n, axis=1)))
    W = L.windows[B]
    cell_of_fn = None
    own_fn = np.where(np.isin(W, np.where(b.atom_of_fn >= 0)[0]))[0]   # placeholder, replaced below
    d = np.linalg.norm(cr.pos[b.atom_of_fn[W]] - cell_centres((n, n, n), a)[B], axis=1)
    own_fn = np.where(d < 0.5 * a)[0]                                # functions of the atoms in the block
    HW, SW = H[W][:, W].toarray(), S[W][:, W].toarray()
    HWs, SWs = sparse.csc_matrix(H[W][:, W]), sparse.csc_matrix(S[W][:, W])
    nW = len(W)
    rep = 3

    def eig():
        lam, U = linalg.eigh(HW, SW, subset_by_value=(-np.inf, c + 1.5 * R), driver="gvx")
        occ = 1.0 / (1.0 + ((lam - c) / R) ** M)
        Uo = U[own_fn]
        return (Uo * occ) @ Uo.T

    def dense():
        P = np.zeros((len(own_fn),) * 2, complex)
        E = np.zeros((nW, len(own_fn)))
        E[own_fn, np.arange(len(own_fn))] = 1.0
        for z, wt in zip(zs, ws):
            X = linalg.solve(z * SW - HW, E)
            P += 2 * wt * X[own_fn]                                  # + conjugate half
        return P.real

    def sp():
        P = np.zeros((len(own_fn),) * 2, complex)
        E = np.zeros((nW, len(own_fn)))
        E[own_fn, np.arange(len(own_fn))] = 1.0
        for z, wt in zip(zs, ws):
            lu = spla.splu((z * SWs - HWs).tocsc())
            P += 2 * wt * lu.solve(E.astype(complex))[own_fn]
        return P.real

    row = dict(R_w=Rw, n_W=nW, n_own=len(own_fn), fill=float(HWs.nnz / nW ** 2))
    outs = {}
    for name, f in (("eig", eig), ("dense_solves", dense), ("sparse_solves", sp)):
        ts = []
        for _ in range(rep):
            t = time.perf_counter()
            outs[name] = f()
            ts.append(time.perf_counter() - t)
        row["t_" + name] = min(ts)
    row["max_diff_dense_vs_eig"] = float(np.abs(outs["dense_solves"] - outs["eig"]).max())
    row["max_diff_sparse_vs_eig"] = float(np.abs(outs["sparse_solves"] - outs["eig"]).max())
    res.append(row)
    print(row, flush=True)
    json.dump(res, open(os.path.join(ROOT, "results", "window_resolvent.json"), "w"), indent=1)
