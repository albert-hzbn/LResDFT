"""Repeated, interleaved SCF timings of dense KS vs LRes-DFT for H2 crystals (n x n x n cells, 2n^3 electrons).

For each size and each repeat both solvers are run on a freshly built system, with the order alternating between
repeats; wall time, process CPU time and the load average are recorded for every run, so that machine noise can
be separated from the method.  Single thread.  Writes results/scaling_rigorous.json (appended per run).
Usage: taskset -c 3 python3 benchmarks/scaling_rigorous.py [repeats]
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"

import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
from scf3d_large import BASIS, run_scf  # noqa: E402
from lrft.lcao3d import KSLCAOCpp, LCAOBasis, LRFTLCAOCpp, cell_centres  # noqa: E402
from lrft.three_d import h2_crystal  # noqa: E402

OUT = os.path.join(ROOT, "results", "scaling_rigorous.json")
REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SIZES = (3, 4, 5, 6, 7, 8)
res = json.load(open(OUT)) if os.path.exists(OUT) else []


def one(n, which):
    cr = h2_crystal((n, n, n), a=6.0, pts_per_cell=10, rc=0.6)
    b = LCAOBasis(cr, **BASIS)
    t0, c0, l0 = time.perf_counter(), time.process_time(), os.getloadavg()[0]
    s = KSLCAOCpp(b) if which == "KS" else LRFTLCAOCpp(b, cell_centres((n, n, n), 6.0), R_w=12.1, M=64)
    r = run_scf(cr, s)
    row = dict(n=n, N_e=cr.n_elec, N_b=b.nb, solver=which, wall=time.perf_counter() - t0, cpu=time.process_time() - c0,
               scf_wall=r["wall"], iters=r["iterations"], energy=r["energy"], converged=bool(r["converged"]),
               step_times=r["step_times"], load_start=l0, load_end=os.getloadavg()[0])
    return row


for rep in range(REPEATS):
    for n in SIZES:
        order = ("KS", "LRes") if (rep + n) % 2 == 0 else ("LRes", "KS")
        for which in order:
            if any(x["n"] == n and x["solver"] == which and x["rep"] == rep for x in res):
                continue
            row = one(n, which)
            row["rep"] = rep
            res.append(row)
            json.dump(res, open(OUT, "w"), indent=1)
            print(rep, n, row["N_e"], which, f"wall {row['wall']:.1f} cpu {row['cpu']:.1f} it {row['iters']} "
                  f"load {row['load_start']:.2f}/{row['load_end']:.2f}", flush=True)
if not any(x["n"] == 10 for x in res):                       # LRes-DFT only: dense KS does not fit in time/memory
    row = one(10, "LRes")
    row["rep"] = 0
    res.append(row)
    json.dump(res, open(OUT, "w"), indent=1)
    print(10, row["N_e"], "LRes", f"wall {row['wall']:.1f} cpu {row['cpu']:.1f}", flush=True)
