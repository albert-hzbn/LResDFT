"""Additional validations anticipating referee questions (Supplemental Material).

R1  3D: LRFT occupation r_M vs divide-and-conquer with Fermi-Dirac occupation, same windows
R2  3D: dependence on the band gap (compressed H2 crystals)
R3  3D: disordered (amorphous) H2 arrangement
R4  3D: basis-set trade-off (basis error vs window convergence)
R5  1D: sensitivity to the margin delta (compare with Theorem 3)
R6  SCF convergence histories, 1D and 3D
R7  3D timing repeatability
R8  peak memory, KS vs LRFT (3D)
R9  1D: closing gap (uniform H chains): M needed vs gap
Results are appended to results/referee_checks.json (one key per part).
Usage: python3 benchmarks/referee_checks.py R1 R2 ...
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"

import contextlib
import io
import json
import re
import subprocess
import sys
import time
import warnings

import numpy as np
from scipy import linalg
from scipy.optimize import brentq

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lrft.core import Grid, Hartree, System, exchange, external_potential, scf, tridiag, nodes
from lrft.three_d import Crystal, h2_crystal, hartree, ion_potential, lda_xc, KSSolver, scf3d, r_M
from lrft.lcao3d import (KSLCAOCpp, LCAOBasis, LRFTLCAO, LRFTLCAOCpp, cell_centres, scf_lcao,
                         _min_image)
from benchmarks.accuracy import dimer_chain

OUT = os.path.join(ROOT, "results", "referee_checks.json")
BASIS = dict(rcut=4.5, s_exps=(1.2, 0.4, 0.15), p_exps=(0.5,))


def save(key, val):
    d = json.load(open(OUT)) if os.path.exists(OUT) else {}
    d[key] = val
    json.dump(d, open(OUT, "w"), indent=1)
    print(key, "saved", flush=True)


def scf_potential(cr, b, tol=1e-8):
    ks = scf_lcao(cr, KSLCAOCpp(b), tol=tol)
    v = ion_potential(cr) + hartree(cr, ks["density"]) + lda_xc(ks["density"])[1]
    return v, ks


def window_density(L, v, occ_fn, N):
    """Density from LRFT windows with an arbitrary occupation function occ_fn(lam, mu)."""
    b = L.b
    H, S = b.H(v), b.S
    spec = [L._window(H, S, B) for B in range(len(L.windows))]
    lam = np.concatenate([s[0] for s in spec])
    c = np.concatenate([np.sum(s[1] ** 2, axis=0) * b.cr.dV for s in spec])
    count = lambda m: 2 * np.dot(c, occ_fn(lam, m)) - N
    lo, hi = lam.min() + 1e-6, lam.min() + 2.0
    mu = brentq(count, lo, hi, xtol=1e-13)
    dens = np.zeros(b.cr.ng)
    for B, (l, Y) in enumerate(spec):
        dens[L.own[B]] = (Y * Y) @ occ_fn(l, mu)
    return 2 * dens.reshape(b.cr.shape), 2 * np.dot(c, lam * occ_fn(lam, mu))


def R1():
    """r_M (LRFT) vs Fermi-Dirac (divide-and-conquer) occupations in identical windows."""
    n = (5, 5, 5)
    cr = h2_crystal(n, a=6.0, pts_per_cell=10, rc=0.6)
    b = LCAOBasis(cr, **BASIS)
    v, _ = scf_potential(cr, b)
    N = cr.n_elec
    nk, ebk, w = KSLCAOCpp(b).density(v)
    gap = w[N // 2] - w[N // 2 - 1]
    rows = []
    for Rw in (6.1, 8.5, 12.1, 13.5):
        L = LRFTLCAO(b, cell_centres(n, 6.0), R_w=Rw)
        e_lo = None
        # LRFT occupation (circle below spectrum, M = 64)
        H, S = b.H(v), b.S
        lam_min = min(L._window(H, S, B)[0][0] for B in range(len(L.windows)))
        e_lo = lam_min - 0.3
        res = {"R_w": Rw, "n_W": float(L.nW.mean())}
        for name, fn in [("LRFT_M64", lambda l, m: r_M(l, m, e_lo, 64)),
                         ("DC_FD_kT0.01", lambda l, m: 0.5 * (1 - np.tanh((l - m) / 0.02))),
                         ("DC_FD_kT0.005", lambda l, m: 0.5 * (1 - np.tanh((l - m) / 0.01)))]:
            nn, eb = window_density(L, v, fn, N)
            res[name + "_dn"] = float(np.abs(nn - nk).sum() * cr.dV / N)
            res[name + "_dE"] = float(abs(eb - ebk) / N)
        rows.append(res)
        print("R1", json.dumps(res), flush=True)
    save("R1_dc_comparison", dict(gap=float(gap), rows=rows))


def R2():
    """Gap dependence: compress the H2 crystal."""
    rows = []
    for a in (6.0, 5.5, 5.0, 4.5):
        n = (6, 6, 6) if a < 5.6 else (5, 5, 5)
        cr = h2_crystal(n, a=a, pts_per_cell=int(round(a / 0.6)), rc=0.6)
        b = LCAOBasis(cr, **BASIS)
        v, ks = scf_potential(cr, b)
        N = cr.n_elec
        nk, ebk, w = KSLCAOCpp(b).density(v)
        gap = float(w[N // 2] - w[N // 2 - 1])
        for Rw in (8.5, 12.1):
            for M in (32, 64, 128):
                L = LRFTLCAOCpp(b, cell_centres(n, a), R_w=Rw, M=M)
                t = time.perf_counter()
                nn, eb, mu = L.density(v)
                row = dict(a=a, N_e=N, gap=gap, R_w=Rw, M=M, n_W=float(L.nW.mean()),
                           dn_per_e=float(np.abs(nn - nk).sum() * cr.dV / N),
                           dE_per_e=float(abs(eb - ebk) / N), t=time.perf_counter() - t)
                rows.append(row)
                print("R2", json.dumps(row), flush=True)
    save("R2_gap_dependence", rows)


def amorphous_h2(nmol, box, dmin=4.2, bond=1.4, seed=7, pts=10, rc=0.6):
    rng = np.random.default_rng(seed)
    centres = []
    while len(centres) < nmol:
        c = rng.uniform(0, box, 3)
        if all(np.linalg.norm(_min_image(c - q, np.full(3, box))) >= dmin for q in centres):
            centres.append(c)
    pos = []
    for c in centres:
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        pos += [c - 0.5 * bond * u, c + 0.5 * bond * u]
    m = int(round(box / 0.6))
    return Crystal(np.array(pos), np.ones(2 * nmol), np.full(3, box), (m, m, m), rc), np.array(centres)


def R3():
    """Disordered arrangement: same density as the a=6 crystal (125 molecules in 30^3 bohr^3)."""
    cr, centres = amorphous_h2(125, 30.0)
    b = LCAOBasis(cr, **BASIS)
    v, ks = scf_potential(cr, b)
    N = cr.n_elec
    nk, ebk, w = KSLCAOCpp(b).density(v)
    rows = []
    for Rw in (8.5, 12.1, 13.5):
        L = LRFTLCAOCpp(b, centres, R_w=Rw)
        nn, eb, mu = L.density(v)
        rows.append(dict(R_w=Rw, n_W_mean=float(L.nW.mean()), n_W_max=int(L.nW.max()),
                         dn_per_e=float(np.abs(nn - nk).sum() * cr.dV / N), dE_per_e=float(abs(eb - ebk) / N)))
        print("R3", json.dumps(rows[-1]), flush=True)
    lr = scf_lcao(cr, LRFTLCAOCpp(b, centres, R_w=12.1), tol=1e-7)
    save("R3_amorphous", dict(N_e=N, gap=float(w[N // 2] - w[N // 2 - 1]), rows=rows,
                              scf_dE_per_e=(lr["energy"] - ks["energy"]) / N,
                              scf_it_KS=ks["iterations"], scf_it_LRFT=lr["iterations"]))


def R4():
    """Basis-set trade-off: basis error (vs grid KS) and window convergence."""
    bases = {"compact (5/atom, $r_c=4.5$)": dict(rcut=4.5, s_exps=(1.0, 0.25), p_exps=(0.6,)),
             "used (6/atom, $r_c=4.5$)": BASIS,
             "diffuse (6/atom, $r_c=5.5$)": dict(rcut=5.5, s_exps=(1.0, 0.3, 0.12), p_exps=(0.5,))}
    cr2 = h2_crystal((2, 2, 2), a=6.0, pts_per_cell=10, rc=0.6)
    Eg = scf3d(cr2, KSSolver(cr2, tol=1e-8, maxiter=400), tol=1e-7).energy
    rows = []
    for name, bp in bases.items():
        b2 = LCAOBasis(cr2, **bp)
        e2 = scf_lcao(cr2, KSLCAOCpp(b2), tol=1e-7)["energy"]
        n = (5, 5, 5)
        cr = h2_crystal(n, a=6.0, pts_per_cell=10, rc=0.6)
        b = LCAOBasis(cr, **bp)
        v, _ = scf_potential(cr, b)
        nk, ebk, _ = KSLCAOCpp(b).density(v)
        smin = float(np.linalg.eigvalsh(b2.S.toarray())[0])
        for Rw in (8.5, 12.1):
            L = LRFTLCAOCpp(b, cell_centres(n, 6.0), R_w=Rw)
            nn, eb, _ = L.density(v)
            rows.append(dict(basis=name, basis_err_per_e=(e2 - Eg) / 16, min_overlap_eig=smin, R_w=Rw,
                             n_W=float(L.nW.mean()),
                             dn_per_e=float(np.abs(nn - nk).sum() * cr.dV / cr.n_elec),
                             dE_per_e=float(abs(eb - ebk) / cr.n_elec)))
            print("R4", json.dumps(rows[-1]), flush=True)
    save("R4_basis", rows)


def R5():
    """Margin delta: measured error vs Theorem-3 prediction (1D (H2)_16, M = 48)."""
    s = dimer_chain(16)
    g = Grid.around(s, dx=0.1, pad=8)
    ks = scf(s, g, "ks", tol=1e-11)
    v = external_potential(s, g) + Hartree(g).potential(ks.density) + exchange(ks.density)[1]
    d, e = tridiag(v, g.dx)
    eps, U = linalg.eigh_tridiagonal(d, e)
    N = s.n_elec
    nocc = N // 2
    n_ks = 2 * (U[:, :nocc] ** 2).sum(1) / g.dx
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])
    rows = []
    for delta in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0):
        e_lo = eps[0] - delta
        c, R = 0.5 * (mu + e_lo), 0.5 * (mu - e_lo)
        u = (eps - c) / R
        rho = max(np.abs(u[:nocc]).max(), (1 / np.abs(u[nocc:])).max())
        for M in (32, 48):
            occ = 1 / (1 + u**M)
            err = np.sum(np.abs(2 * (U**2) @ occ / g.dx - n_ks)) * g.dx / N
            rows.append(dict(delta=delta, M=M, rho=float(rho), err_per_e=float(err),
                             bound_per_e=float((rho**M / (1 + rho**M)) * 2 * (nocc + 5) / N)))
            print("R5", json.dumps(rows[-1]), flush=True)
    save("R5_margin", dict(gap=float(eps[nocc] - eps[nocc - 1]), occupied_width=float(eps[nocc - 1] - eps[0]),
                           rows=rows))


def capture(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn()
    hist = [float(x) for x in re.findall(r"\|dn\|(?:/N)? = ([0-9.eE+-]+)", buf.getvalue())]
    return out, hist


def R6():
    res = {}
    s = dimer_chain(16)
    g = Grid.around(s, dx=0.1, pad=8)
    _, h = capture(lambda: scf(s, g, "ks", tol=1e-10, verbose=True))
    res["1D_KS"] = [x / s.n_elec for x in h]
    _, h = capture(lambda: scf(s, g, "lrft", M=64, tol=1e-10, verbose=True))
    res["1D_LRFT"] = [x / s.n_elec for x in h]
    n = (4, 4, 4)
    cr = h2_crystal(n, a=6.0, pts_per_cell=10, rc=0.6)
    b = LCAOBasis(cr, **BASIS)
    _, res["3D_KS"] = capture(lambda: scf_lcao(cr, KSLCAOCpp(b), tol=1e-9, verbose=True))
    _, res["3D_LRFT"] = capture(lambda: scf_lcao(cr, LRFTLCAOCpp(b, cell_centres(n, 6.0), R_w=12.1),
                                                 tol=1e-9, verbose=True))
    print("R6", {k: len(v) for k, v in res.items()}, flush=True)
    save("R6_scf_histories", res)


def R7(reps=5):
    rows = []
    for n in (5, 6, 7):
        cr = h2_crystal((n, n, n), a=6.0, pts_per_cell=10, rc=0.6)
        b = LCAOBasis(cr, **BASIS)
        d0 = np.full(cr.shape, cr.n_elec / (cr.ng * cr.dV))
        v = ion_potential(cr) + hartree(cr, d0) + lda_xc(d0)[1]
        L = LRFTLCAOCpp(b, cell_centres((n, n, n), 6.0), R_w=12.1)
        L.density(v)
        K = KSLCAOCpp(b)
        tl, tk = [], []
        for _ in range(reps):
            t = time.perf_counter(); L.density(v); tl.append(time.perf_counter() - t)
            t = time.perf_counter(); K.density(v); tk.append(time.perf_counter() - t)
        rows.append(dict(N_e=cr.n_elec, lrft_mean=float(np.mean(tl)), lrft_std=float(np.std(tl)),
                         lrft_min=float(np.min(tl)), ks_mean=float(np.mean(tk)), ks_std=float(np.std(tk)),
                         ks_min=float(np.min(tk))))
        print("R7", json.dumps(rows[-1]), flush=True)
    save("R7_timing_repeatability", rows)


MEM_CHILD = r'''
import os, sys, json, warnings
warnings.filterwarnings("ignore")
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0, %r)
import numpy as np
from lrft.three_d import h2_crystal, ion_potential, hartree, lda_xc
from lrft.lcao3d import LCAOBasis, KSLCAOCpp, LRFTLCAOCpp, cell_centres
def status(key):
    for line in open("/proc/self/status"):
        if line.startswith(key):
            return int(line.split()[1]) / 1024.0
n, which = int(sys.argv[1]), sys.argv[2]
cr = h2_crystal((n,n,n), a=6.0, pts_per_cell=10, rc=0.6)
b = LCAOBasis(cr, rcut=4.5, s_exps=(1.2,0.4,0.15), p_exps=(0.5,))
d0 = np.full(cr.shape, cr.n_elec/(cr.ng*cr.dV)); v = ion_potential(cr)+hartree(cr,d0)+lda_xc(d0)[1]
if which == "lrft":
    L = LRFTLCAOCpp(b, cell_centres((n,n,n),6.0), R_w=12.1)
open("/proc/self/clear_refs","w").write("5")      # reset the peak-RSS counter
base = status("VmRSS:")
if which == "ks": KSLCAOCpp(b).density(v)
elif which == "lrft": L.density(v); L.density(v)
peak = status("VmHWM:")
print(json.dumps(dict(n=n, N_e=cr.n_elec, N_b=b.nb, which=which, rss_before_MB=base, peak_MB=peak,
                      extra_MB=peak - base)))
''' % ROOT


def R8():
    rows = []
    for n in (4, 5, 6, 7, 8):
        for which in ("lrft", "ks"):
            out = subprocess.run([sys.executable, "-c", MEM_CHILD, str(n), which], capture_output=True, text=True)
            r = json.loads(out.stdout.strip().splitlines()[-1])
            rows.append(r)
            print("R8", json.dumps(r), flush=True)
    save("R8_memory", rows)


def R9():
    """Closing gap: uniform H_n chains (spacing 2 bohr) in 1D."""
    rows = []
    for nat in (10, 20, 40, 80):
        s = System(np.ones(nat), 2.0 * (np.arange(nat) - (nat - 1) / 2), nat)
        g = Grid.around(s, dx=0.1, pad=8)
        ks = scf(s, g, "ks", tol=1e-10)
        v = external_potential(s, g) + Hartree(g).potential(ks.density) + exchange(ks.density)[1]
        d, e = tridiag(v, g.dx)
        eps, U = linalg.eigh_tridiagonal(d, e, select="i", select_range=(0, nat // 2 + 20))
        nocc = nat // 2
        n_ks = 2 * (U[:, :nocc] ** 2).sum(1) / g.dx
        mu = 0.5 * (eps[nocc - 1] + eps[nocc])
        e_lo = eps[0] - 0.3
        c, R = 0.5 * (mu + e_lo), 0.5 * (mu - e_lo)
        u = (eps - c) / R
        need = {}
        for tol in (1e-4, 1e-6):
            need[tol] = next((M for M in range(4, 4000, 2)
                              if np.sum(np.abs(2 * (U**2) @ (1 / (1 + u**M)) / g.dx - n_ks)) * g.dx / nat < tol), None)
        rows.append(dict(N_atoms=nat, gap=float(eps[nocc] - eps[nocc - 1]), M_for_1e4=need[1e-4], M_for_1e6=need[1e-6]))
        print("R9", json.dumps(rows[-1]), flush=True)
    save("R9_closing_gap", rows)


# ----------------------------------------------------------------------------
# R10: number of complex solves for three pole rules (same Hamiltonians)
# ----------------------------------------------------------------------------
from scipy import special


def zolotarev_sign(x, ell, r):
    """Zolotarev's best type-(2r+1,2r) rational approximation of sign(x) on [-1,-ell]U[ell,1].
    Poles at x = +-i sqrt(c_{2j-1}), j = 1..r  ->  r conjugate pairs of solves."""
    lp = np.sqrt(1 - ell**2)
    Kp = special.ellipk(lp**2)
    c = []
    for i in range(1, 2 * r + 1):
        sn, cn, dn, ph = special.ellipj(i * Kp / (2 * r + 1), lp**2)
        c.append(ell**2 * sn**2 / cn**2)
    def raw(t):
        out = t.copy()
        for j in range(r):
            out = out * (t**2 + c[2 * j + 1]) / (t**2 + c[2 * j])
        return out
    tt = np.geomspace(ell, 1, 4000)
    vals = raw(tt)
    Mh = 2.0 / (vals.max() + vals.min())
    return Mh * raw(x)


def gl_circle(eps, c, R, n):
    """Gauss-Legendre in the angle on the same circle (FEAST-type); n conjugate pairs of solves."""
    xg, wg = np.polynomial.legendre.leggauss(n)
    th = np.pi * (xg + 1) / 2
    z = c + R * np.exp(1j * th)
    val = (wg[:, None] * 0.5 * R * np.exp(1j * th)[:, None] / (z[:, None] - eps[None, :])).sum(0)
    return np.real(val)


def solves_needed(eps, U, nocc, dx, N, tol=1e-6, delta=0.3):
    n_ks = 2 * (U[:, :nocc] ** 2).sum(1) / dx
    err = lambda occ: np.sum(np.abs(2 * (U**2) @ occ / dx - n_ks)) * dx / N
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])
    e_lo = eps[0] - delta
    c, R = 0.5 * (mu + e_lo), 0.5 * (mu - e_lo)
    u = (eps - c) / R
    out = {}
    out["half_step_trapezoid"] = next((M // 2 for M in range(4, 8000, 2) if err(1 / (1 + u**M)) < tol), None)
    out["gauss_legendre_circle"] = next((k for k in range(2, 4000) if err(gl_circle(eps, c, R, k)) < tol), None)
    W = np.max(np.abs(eps - mu))
    ell = (0.5 * (eps[nocc] - eps[nocc - 1])) / W
    x = (eps - mu) / W
    out["zolotarev"] = next((r for r in range(1, 200) if err(0.5 * (1 - zolotarev_sign(x, ell, r))) < tol), None)
    return out, float(W), float(eps[nocc] - eps[nocc - 1]), float(eps[nocc - 1] - eps[0])


def full_spectrum(s, g):
    ks = scf(s, g, "ks", tol=1e-11)
    v = external_potential(s, g) + Hartree(g).potential(ks.density) + exchange(ks.density)[1]
    d, e = tridiag(v, g.dx)
    return linalg.eigh_tridiagonal(d, e)


def R10():
    rows = []
    s = dimer_chain(4)
    for dx in (0.2, 0.1, 0.05, 0.025):
        g = Grid.around(s, dx=dx, pad=6)
        eps, U = full_spectrum(s, g)
        out, W, gap, wocc = solves_needed(eps, U, s.n_elec // 2, dx, s.n_elec)
        rows.append(dict(series="grid", dx=dx, spectral_width=W, gap=gap, occ_width=wocc, **out))
        print("R10", json.dumps(rows[-1]), flush=True)
    for nat in (10, 20, 40, 80):
        s = System(np.ones(nat), 2.0 * (np.arange(nat) - (nat - 1) / 2), nat)
        g = Grid.around(s, dx=0.1, pad=8)
        eps, U = full_spectrum(s, g)
        out, W, gap, wocc = solves_needed(eps, U, nat // 2, g.dx, nat)
        rows.append(dict(series="gap", N_atoms=nat, spectral_width=W, gap=gap, occ_width=wocc, **out))
        print("R10", json.dumps(rows[-1]), flush=True)
    for Z in (2, 4, 6, 8):
        base = dimer_chain(4)
        X = np.concatenate([base.X, [base.X.max() + 3.5]])
        s = System(np.concatenate([base.Z, [float(Z)]]), X, base.n_elec + Z)
        g = Grid.around(s, dx=0.1, pad=6)
        eps, U = full_spectrum(s, g)
        out, W, gap, wocc = solves_needed(eps, U, s.n_elec // 2, g.dx, s.n_elec)
        rows.append(dict(series="deep_core", Z_heavy=Z, spectral_width=W, gap=gap, occ_width=wocc, **out))
        print("R10", json.dumps(rows[-1]), flush=True)
    save("R10_pole_rules", rows)


# ----------------------------------------------------------------------------
# R11: forces -- finite-difference step study, grand-canonical and fixed-N
# ----------------------------------------------------------------------------

def R11():
    from benchmarks.extra_checks import grand_scf
    base = System([1.0, 1.0], [-0.8, 0.8], 2)
    g = Grid.around(base, dx=0.1, pad=8)
    ks = scf(base, g, "ks", tol=1e-12)
    mu = ks.mu + 0.15
    v0 = external_potential(base, g)
    e_lo = linalg.eigh_tridiagonal(*tridiag(v0, g.dx), eigvals_only=True, select="i",
                                   select_range=(0, 0))[0] - 0.3
    X2 = 0.8
    dvdX = -(g.x - X2) / ((g.x - X2) ** 2 + 1) ** 1.5
    dEnn = -(X2 + 0.8) / ((X2 + 0.8) ** 2 + 1) ** 1.5
    rows = []
    for M in (8, 16, 64):
        # grand canonical
        Phi0, n0, _ = grand_scf(base, g, mu, e_lo, M)
        F_hf = -(np.sum(n0 * dvdX) * g.dx + dEnn)
        fd = {}
        for h in (1e-2, 1e-3, 1e-4):
            P = [grand_scf(System([1.0, 1.0], [-0.8, 0.8 + sg * h], 2), g, mu, e_lo, M)[0] for sg in (-1, 1)]
            fd[h] = -(P[1] - P[0]) / (2 * h)
        rich = (4 * fd[1e-3] - fd[1e-2]) / 3          # Richardson on h=1e-2,1e-3 is not standard;
        rich = fd[1e-3] + (fd[1e-3] - fd[1e-2]) / 99   # h ratio 10 -> extrapolate O(h^2)
        # fixed N (mu adjusted; circle moves with mu)
        r0 = scf(base, g, "lrft", M=M, tol=1e-12, maxiter=400)
        F_hf_N = -(np.sum(r0.density * dvdX) * g.dx + dEnn)
        EN = [scf(System([1.0, 1.0], [-0.8, 0.8 + sg * 1e-3], 2), g, "lrft", M=M, tol=1e-12, maxiter=400).energy
              for sg in (-1, 1)]
        F_fd_N = -(EN[1] - EN[0]) / 2e-3
        rows.append(dict(M=M, F_hf_grand=F_hf, F_fd_grand={str(k): v for k, v in fd.items()},
                         F_fd_grand_extrap=rich, diff_grand_extrap=abs(rich - F_hf),
                         F_hf_fixedN=F_hf_N, F_fd_fixedN=F_fd_N, diff_fixedN=abs(F_fd_N - F_hf_N)))
        print("R11", json.dumps(rows[-1]), flush=True)
    ks_force = None
    save("R11_forces", rows)


# ----------------------------------------------------------------------------
# R12: window diagnostics -- spurious in-gap / sub-E_lo window states
# ----------------------------------------------------------------------------

def R12():
    out = {}
    for name in ("crystal", "amorphous"):
        if name == "crystal":
            n = (5, 5, 5)
            cr = h2_crystal(n, a=6.0, pts_per_cell=10, rc=0.6)
            centres = cell_centres(n, 6.0)
        else:
            cr, centres = amorphous_h2(125, 30.0)
        b = LCAOBasis(cr, **BASIS)
        v, _ = scf_potential(cr, b)
        N = cr.n_elec
        nk, ebk, w = KSLCAOCpp(b).density(v)
        homo, lumo = w[N // 2 - 1], w[N // 2]
        res = {}
        for Rw in (8.5, 12.1):
            L = LRFTLCAO(b, centres, R_w=Rw)
            H, S = b.H(v), b.S
            lam, cw = [], []
            for B in range(len(L.windows)):
                l, Y = L._window(H, S, B)
                lam.append(l)
                cw.append(np.sum(Y**2, 0) * cr.dV)
            lam, cw = np.concatenate(lam), np.concatenate(cw)
            ingap = (lam > homo + 1e-4) & (lam < lumo - 1e-4)
            res[str(Rw)] = dict(homo=float(homo), lumo=float(lumo), n_states_in_gap=int(ingap.sum()),
                                weight_in_gap_per_e=float(2 * cw[ingap].sum() / N),
                                min_window_eig=float(lam.min()), ks_min_eig=float(w[0]),
                                hist_counts=np.histogram(lam, bins=40, range=(w[0] - 0.2, lumo + 0.3))[0].tolist(),
                                hist_edges=np.histogram(lam, bins=40, range=(w[0] - 0.2, lumo + 0.3))[1].tolist())
            print("R12", name, Rw, {k: v for k, v in res[str(Rw)].items() if not k.startswith("hist")}, flush=True)
        out[name] = res
    save("R12_window_diagnostics", out)


# ----------------------------------------------------------------------------
# R13: Theorem-3 bound computed exactly as stated (1D (H2)_16, delta = 0.3)
# ----------------------------------------------------------------------------

def R13():
    s = dimer_chain(16)
    g = Grid.around(s, dx=0.1, pad=8)
    eps, U = full_spectrum(s, g)
    N, nocc, Ng = s.n_elec, s.n_elec // 2, g.n
    n_ks = 2 * (U[:, :nocc] ** 2).sum(1) / g.dx
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])
    delta = 0.3
    e_lo = eps[0] - delta
    c, R = 0.5 * (mu + e_lo), 0.5 * (mu - e_lo)
    u = (eps - c) / R
    rho = max(abs(1 - delta / R), 1 - (mu - eps[nocc - 1]) / R, 1 / (1 + (eps[nocc] - mu) / R))
    n_near = int(np.sum((np.abs(u) > 1) & (np.abs(u) < 2)))
    rows = []
    for M in (8, 16, 24, 32, 48, 64, 96, 128):
        occ = 1 / (1 + u**M)
        err = float(np.sum(np.abs(2 * (U**2) @ occ / g.dx - n_ks)) * g.dx)
        bound = 2 * (nocc + n_near) * rho**M + 2 * Ng * 2.0 ** (-M)
        rows.append(dict(M=M, err_L1_total=err, err_per_e=err / N, bound_total=float(bound)))
        print("R13", json.dumps(rows[-1]), flush=True)
    save("R13_theorem1_bound", dict(rho=float(rho), R=float(R), delta=delta, N_near=n_near, N_g=Ng,
                                    N=N, rows=rows))



# ----------------------------------------------------------------------------
# R17: fixed-N forces with a rigidly moving circle (R fixed, c = mu - R):
#      A_M = Omega_M(mu) + mu N - sum v_Hxc n + E_H + E_x + E_nn is stationary at fixed N
# ----------------------------------------------------------------------------

def rigid_scf(s, g, R, M, N, tol=1e-12, maxiter=400):
    from lrft.core import lrft_density
    dx = g.dx
    vext = external_potential(s, g)
    hart = Hartree(g, s.soft)
    dens = np.zeros(g.n)
    for X in s.X:
        dens += np.exp(-((g.x - X) ** 2))
    dens *= N / (dens.sum() * dx)
    mu = None
    hist_n, hist_r = [], []
    for it in range(maxiter):
        v = vext + hart.potential(dens) + exchange(dens)[1]
        cnt = lambda m: lrft_density(v, dx, m, m - 2 * R, M).n_elec - N
        lo = (mu - 0.2) if mu is not None else -1.5
        hi = (mu + 0.2) if mu is not None else 0.5
        while cnt(lo) > 0:
            lo -= 0.3
        while cnt(hi) < 0:
            hi += 0.3
        mu = brentq(cnt, lo, hi, xtol=1e-14)
        nout = lrft_density(v, dx, mu, mu - 2 * R, M).density
        r = nout - dens
        if np.sum(np.abs(r)) * dx < tol:
            dens = nout
            break
        hist_n.append(dens.copy()); hist_r.append(r.copy())
        hist_n, hist_r = hist_n[-8:], hist_r[-8:]
        k = len(hist_n)
        B = np.ones((k + 1, k + 1)); B[:k, :k] = [[a @ b_ for b_ in hist_r] for a in hist_r]; B[k, k] = 0
        rhs = np.zeros(k + 1); rhs[k] = 1
        cf = np.linalg.lstsq(B, rhs, rcond=None)[0][:k]
        dens = sum(c * h for c, h in zip(cf, hist_n)) + 0.3 * sum(c * h for c, h in zip(cf, hist_r))
    v_hxc = hart.potential(dens) + exchange(dens)[1]
    d, e = tridiag(vext + v_hxc, dx)
    eps = linalg.eigh_tridiagonal(d, e, eigvals_only=True)
    zs, ws = nodes(M, mu, mu - 2 * R)
    Omega = -4.0 * np.real(np.sum(ws[:, None] * np.log(zs[:, None] - eps[None, :])))
    E_H = 0.5 * np.sum(dens * hart.potential(dens)) * dx
    A = Omega + mu * N - np.sum(v_hxc * dens) * dx + E_H + np.sum(exchange(dens)[0]) * dx + s.nuclear_repulsion()
    return A, dens, mu


def R17():
    base = System([1.0, 1.0], [-0.8, 0.8], 2)
    g = Grid.around(base, dx=0.1, pad=8)
    X2 = 0.8
    dvdX = -(g.x - X2) / ((g.x - X2) ** 2 + 1) ** 1.5
    dEnn = -(X2 + 0.8) / ((X2 + 0.8) ** 2 + 1) ** 1.5
    R = 0.5
    rows = []
    for M in (8, 16, 64):
        A0, n0, mu0 = rigid_scf(base, g, R, M, 2)
        F_hf = -(np.sum(n0 * dvdX) * g.dx + dEnn)
        fd = {}
        for h in (1e-2, 1e-3):
            P = [rigid_scf(System([1.0, 1.0], [-0.8, 0.8 + sg * h], 2), g, R, M, 2)[0] for sg in (-1, 1)]
            fd[h] = -(P[1] - P[0]) / (2 * h)
        extrap = fd[1e-3] + (fd[1e-3] - fd[1e-2]) / 99
        # moving-edge fixed-N energy (Eq. 2 of the paper), extrapolated in h
        r0 = scf(base, g, "lrft", M=M, tol=1e-12, maxiter=400)
        F_hf_m = -(np.sum(r0.density * dvdX) * g.dx + dEnn)
        fdm = {}
        for h in (1e-2, 1e-3):
            E = [scf(System([1.0, 1.0], [-0.8, 0.8 + sg * h], 2), g, "lrft", M=M, tol=1e-12, maxiter=400).energy
                 for sg in (-1, 1)]
            fdm[h] = -(E[1] - E[0]) / (2 * h)
        extrap_m = fdm[1e-3] + (fdm[1e-3] - fdm[1e-2]) / 99
        rows.append(dict(M=M, N_electrons=float(n0.sum() * g.dx), mu=mu0, F_hf_rigid=F_hf,
                         F_fd_rigid_extrap=extrap, diff_rigid=abs(extrap - F_hf),
                         F_hf_moving=F_hf_m, F_fd_moving_extrap=extrap_m, diff_moving=abs(extrap_m - F_hf_m)))
        print("R17", json.dumps(rows[-1]), flush=True)
    save("R17_fixedN_forces", rows)


# ----------------------------------------------------------------------------
# R14: occupation range of the three pole rules at the order needed for 1e-6
# ----------------------------------------------------------------------------

def R14():
    rows = []
    cases = []
    for nat in (10, 20, 40, 80):
        s_ = System(np.ones(nat), 2.0 * (np.arange(nat) - (nat - 1) / 2), nat)
        cases.append((f"H{nat} chain", s_, Grid.around(s_, dx=0.1, pad=8)))
    for Z in (4, 8):
        base = dimer_chain(4)
        s_ = System(np.concatenate([base.Z, [float(Z)]]), np.concatenate([base.X, [base.X.max() + 3.5]]),
                    base.n_elec + Z)
        cases.append((f"core Z={Z}", s_, Grid.around(s_, dx=0.1, pad=6)))
    for name, s_, g in cases:
        eps, U = full_spectrum(s_, g)
        nocc = s_.n_elec // 2
        out, W, gap, wocc = solves_needed(eps, U, nocc, g.dx, s_.n_elec)
        mu = 0.5 * (eps[nocc - 1] + eps[nocc])
        e_lo = eps[0] - 0.3
        c, R = 0.5 * (mu + e_lo), 0.5 * (mu - e_lo)
        hs = 1 / (1 + ((eps - c) / R) ** (2 * out["half_step_trapezoid"]))
        gl = gl_circle(eps, c, R, out["gauss_legendre_circle"])
        ell = 0.5 * gap / W
        zo = 0.5 * (1 - zolotarev_sign((eps - mu) / W, ell, out["zolotarev"]))
        rows.append(dict(case=name, gap=gap, occ_over_gap=wocc / gap,
                         half_step=dict(min=float(hs.min()), max=float(hs.max())),
                         gauss_legendre=dict(min=float(gl.min()), max=float(gl.max())),
                         zolotarev=dict(min=float(zo.min()), max=float(zo.max()))))
        print("R14", json.dumps(rows[-1]), flush=True)
    save("R14_occupation_range", rows)


# ----------------------------------------------------------------------------
# R15: pointwise probing on the real-space grid (the abandoned 3D route)
# ----------------------------------------------------------------------------

def R15():
    from lrft.three_d import hamiltonian, probe_vectors
    cr = h2_crystal((1, 1, 2), a=6.0, pts_per_cell=10, rc=0.6)
    d0 = np.full(cr.shape, cr.n_elec / (cr.ng * cr.dV))
    v = ion_potential(cr) + hartree(cr, d0) + lda_xc(d0)[1]
    Hd = hamiltonian(cr, v).toarray()
    w, U = np.linalg.eigh(Hd)
    nocc = cr.n_elec // 2
    P = U[:, :nocc] @ U[:, :nocc].T
    rows = []
    for d in (2, 5, 10):
        col = probe_vectors(cr, d)
        est = np.array([P[i, col == col[i]].sum() for i in range(cr.ng)])
        rows.append(dict(spacing_points=d, spacing_bohr=d * cr.dx, n_probes=d**3,
                         rel_error=float(np.abs(est - np.diag(P)).sum() / np.diag(P).sum())))
        print("R15", json.dumps(rows[-1]), flush=True)
    save("R15_probing", rows)


# ----------------------------------------------------------------------------
# R16: warm-started LOBPCG KS (complete SCF, 128 and 432 electrons)
# ----------------------------------------------------------------------------

def R16():
    from lrft.lcao3d import KSLOBPCG
    from benchmarks.scf3d_large import run_scf
    d = json.load(open(OUT)) if os.path.exists(OUT) else {}
    prev = d.get("R16_lobpcg_1024")
    rows = []
    for n in (4, 6):
        cr = h2_crystal((n, n, n), a=6.0, pts_per_cell=10, rc=0.6)
        b = LCAOBasis(cr, **BASIS)
        rd = run_scf(cr, KSLCAOCpp(b))
        for tol in (1e-5, 1e-7):
            K = KSLOBPCG(b, cell_centres((n, n, n), 6.0), tol=tol)
            r = run_scf(cr, K)
            rows.append(dict(N_e=cr.n_elec, tol=tol, dE_per_e=(r["energy"] - rd["energy"]) / cr.n_elec,
                             it=r["iterations"], wall_lobpcg=r["wall"], wall_dense=rd["wall"],
                             lobpcg_iters=K.iters, step_times=r["step_times"]))
            print("R16", json.dumps({k: v for k, v in rows[-1].items() if k != "step_times"}), flush=True)
    save("R16_lobpcg", dict(scf=rows, n1024_partial=prev))


# ----------------------------------------------------------------------------
# R18: shared-memory threads (432 electrons): LR-DFT OpenMP over windows, KS threaded LAPACK
# ----------------------------------------------------------------------------

def R18():
    from lrft.lcao3d import KSLCAOCpp2
    cr = h2_crystal((6, 6, 6), a=6.0, pts_per_cell=10, rc=0.6)
    b = LCAOBasis(cr, **BASIS)
    d0 = np.full(cr.shape, cr.n_elec / (cr.ng * cr.dV))
    v = ion_potential(cr) + hartree(cr, d0) + lda_xc(d0)[1]
    rows = []
    for nt in (1, 2, 4, 8):
        L = LRFTLCAOCpp(b, cell_centres((6, 6, 6), 6.0), R_w=12.1, nthreads=nt)
        L.density(v)
        tl = []
        for _ in range(3):
            t = time.perf_counter(); L.density(v); tl.append(time.perf_counter() - t)
        tk = []
        for _ in range(3):
            K = KSLCAOCpp2(b, "gvx", False, nthreads=nt)
            t = time.perf_counter(); K.density(v); tk.append(time.perf_counter() - t)
        rows.append(dict(threads=nt, lrft_warm=min(tl), ks=min(tk), load=os.getloadavg()[0]))
        print("R18", json.dumps(rows[-1]), flush=True)
    save("R18_threads", rows)


if __name__ == "__main__":
    for part in sys.argv[1:]:
        t = time.time()
        globals()[part]()
        print(part, "done in", round(time.time() - t, 1), "s", flush=True)
