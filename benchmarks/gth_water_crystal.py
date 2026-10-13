"""Realistic-Hamiltonian check: periodic H2O molecular crystal (one randomly oriented molecule per a^3 cell),
GTH-PADE nonlocal pseudopotentials + GTH-TZV2P basis, LDA.  KS (dense) versus LRes-DFT windows, identical H.
The grid (dx = a/30) is coarser than the validated 0.2 bohr because of memory: this tests KS versus LRes on the
same Hamiltonian, not agreement with all-electron DFT.  Usage: taskset -c K python3 benchmarks/gth_water_crystal.py n [a]
Writes results/gth_water_n<n>.json."""
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
from scipy.spatial.transform import Rotation  # noqa: E402

from lrft.gth import PseudoBasis, load_gth  # noqa: E402
from lrft.lcao3d import KSLCAOCpp, LRFTLCAOCpp, cell_centres, scf_lcao  # noqa: E402
from lrft.molecules import ANG, MOLECULES  # noqa: E402
from lrft.three_d import Crystal  # noqa: E402

n = int(sys.argv[1])
a = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
pts = 30
rng = np.random.default_rng(1)
_, h2o = MOLECULES["H2O"]
geo = np.array([[x, y, z] for _, x, y, z in h2o]) * ANG
geo -= geo[0]
syms = [s for s, *_ in h2o]
pos, sy = [], []
for i in range(n):
    for j in range(n):
        for k in range(n):
            c = (np.array([i, j, k]) + 0.5) * a + rng.normal(scale=0.3, size=3)
            Rm = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
            pos += list(c + geo @ Rm.T)
            sy += syms
pps = [load_gth(s) for s in sy]
Z = np.array([p["zion"] for p in pps])
cr = Crystal(np.array(pos), Z, np.full(3, a * n), (pts * n,) * 3, rc=1.0, rcs=np.array([2 ** 0.5 * p["rloc"] for p in pps]), pseudo=pps)
t = time.perf_counter()
b = PseudoBasis(cr, basis="gth-tzv2p", rcut=6.0, analytic=False)
print(f"{len(sy)} atoms, {cr.n_elec} electrons, N_b={b.nb}, grid {cr.shape[0]}^3, dx={cr.dx:.3f}, basis {time.perf_counter() - t:.0f}s", flush=True)
res = dict(n=n, a=a, N_e=cr.n_elec, N_b=b.nb, dx=cr.dx, runs=[])
ks = scf_lcao(cr, KSLCAOCpp(b), tol=1e-8 if len(sys.argv) > 3 else 1e-6, maxiter=80)
w = np.asarray(ks["last"][2])
no = cr.n_elec // 2
res["KS"] = dict(E=ks["energy"], wall=ks["time"], it=ks["iterations"], conv=bool(ks["converged"]), gap=float(w[no] - w[no - 1]))
print(res["KS"], flush=True)
cells = cell_centres((n, n, n), a)
if len(sys.argv) > 3 and sys.argv[3] == "M":      # occupation-function check: window = whole system, vary M
    res["M_series"] = []
    for M in (64, 128, 256, 512, 1024):
        lr = scf_lcao(cr, LRFTLCAOCpp(b, cells, R_w=a * 2.05, M=M), tol=1e-8, maxiter=80)
        row = dict(M=M, dE_per_e=(lr["energy"] - ks["energy"]) / cr.n_elec, it=lr["iterations"], conv=bool(lr["converged"]))
        res["M_series"].append(row)
        print(row, flush=True)
        json.dump(res, open(os.path.join(ROOT, "results", f"gth_water_n{n}_M.json"), "w"), indent=1)
    sys.exit()
for Rw in (a * 1.05, a * 1.5, a * 2.05):
    lr = scf_lcao(cr, LRFTLCAOCpp(b, cells, R_w=Rw, M=64), tol=1e-6, maxiter=80)
    row = dict(R_w=Rw, E=lr["energy"], dE_per_e=(lr["energy"] - ks["energy"]) / cr.n_elec, wall=lr["time"], it=lr["iterations"],
               conv=bool(lr["converged"]), dn=float(np.abs(lr["density"] - ks["density"]).sum() * cr.dV / cr.n_elec))
    res["runs"].append(row)
    print(row, flush=True)
    json.dump(res, open(os.path.join(ROOT, "results", f"gth_water_n{n}.json"), "w"), indent=1)
