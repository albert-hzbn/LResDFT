"""Two additional verifications requested for the manuscript.

(A) Grid-refinement test (Theorem 3): the number of complex energies M needed
    for a fixed density accuracy should NOT grow when the grid is refined,
    whereas the degree of a Chebyshev polynomial expansion of the occupation
    function (Fermi-operator expansion) grows with the spectral width ~ 1/dx^2.
    Both are evaluated exactly on the eigen-decomposition of the same KS
    Hamiltonian, so only the approximation of the occupation function differs.

(B) Force consistency (Theorem 2): in the grand-canonical form (fixed filter
    parameters c, R), the self-consistent LRFT grand potential
        Phi_M = Omega_M[v] - int v_Hxc n + E_H + E_xc + E_nn
    is stationary, so the Hellmann-Feynman force -int n dv_ext/dX - dE_nn/dX
    must equal the finite-difference derivative -dPhi_M/dX at ANY finite M.

Writes results/extra_checks.json.
"""
import json
import os
import sys

import numpy as np
from scipy import fft, linalg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lrft.core import (Grid, Hartree, System, exchange, external_potential, lrft_density,
                       nodes, scf, tridiag)
from benchmarks.accuracy import dimer_chain

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
TOL = 1e-6          # target density error per electron


def occ_density(U, occ, dx):
    return 2.0 * (U**2) @ occ / dx


def cheb_coeffs(f, K):
    k = np.arange(K)
    x = np.cos(np.pi * (k + 0.5) / K)
    c = fft.dct(f(x), type=2) / K
    c[0] /= 2
    return c


def part_A():
    rows = []
    s = dimer_chain(4)
    for dx in (0.2, 0.1, 0.05, 0.025):
        g = Grid.around(s, dx=dx, pad=6)
        ks = scf(s, g, "ks", tol=1e-11)
        v = external_potential(s, g) + Hartree(g).potential(ks.density) + exchange(ks.density)[1]
        d, e = tridiag(v, dx)
        eps, U = linalg.eigh_tridiagonal(d, e)
        N = s.n_elec
        nocc = N // 2
        n_ks = occ_density(U, (np.arange(len(eps)) < nocc).astype(float), dx)
        gap = eps[nocc] - eps[nocc - 1]
        mu = 0.5 * (eps[nocc] + eps[nocc - 1])
        e_lo = eps[0] - 0.3
        c, R = 0.5 * (mu + e_lo), 0.5 * (mu - e_lo)

        def err(occ):
            return np.sum(np.abs(occ_density(U, occ, dx) - n_ks)) * dx / N

        # LRFT: smallest even M reaching TOL
        M_need = next(M for M in range(4, 2000, 2)
                      if err(1.0 / (1.0 + ((eps - c) / R) ** M)) < TOL)
        # Chebyshev expansion of a Fermi function on the full spectrum [eps_min, eps_max]
        a, b = eps[0], eps[-1]
        kT = gap / (2 * np.log(2 * N / TOL))          # smearing error below TOL
        fermi = lambda x: 0.5 * (1 - np.tanh((0.5 * (b - a) * x + 0.5 * (b + a) - mu) / (2 * kT)))
        xs = (2 * eps - (a + b)) / (b - a)

        def cheb_err(K):
            return err(np.polynomial.chebyshev.chebval(xs, cheb_coeffs(fermi, K)))

        lo, hi = 8, 8
        while cheb_err(hi) >= TOL:
            lo, hi = hi, hi * 2
        while hi - lo > max(2, lo // 50):
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if cheb_err(mid) < TOL else (mid, hi)
        row = dict(dx=dx, N_g=g.n, spectral_width=float(eps[-1] - eps[0]), gap=float(gap),
                   occupied_width=float(eps[nocc - 1] - eps[0]), M_needed=M_need, cheb_degree_needed=hi)
        rows.append(row)
        print("A", json.dumps(row), flush=True)
    return rows


def grand_scf(s, g, mu, e_lo, M, tol=1e-12, maxiter=400):
    """Self-consistent LRFT at fixed filter parameters (grand canonical)."""
    dx = g.dx
    vext = external_potential(s, g)
    hart = Hartree(g, s.soft)
    dens = np.zeros(g.n)
    for Z, X in zip(s.Z, s.X):
        dens += np.exp(-((g.x - X) ** 2))
    dens *= s.n_elec / (dens.sum() * dx)
    hist_n, hist_r = [], []
    for it in range(maxiter):
        v = vext + hart.potential(dens) + exchange(dens)[1]
        nout = lrft_density(v, dx, mu, e_lo, M).density
        r = nout - dens
        if np.sum(np.abs(r)) * dx < tol:
            dens = nout
            break
        hist_n.append(dens.copy())
        hist_r.append(r.copy())
        hist_n, hist_r = hist_n[-8:], hist_r[-8:]
        k = len(hist_n)
        B = np.ones((k + 1, k + 1))
        B[:k, :k] = [[a @ b_ for b_ in hist_r] for a in hist_r]
        B[k, k] = 0
        rhs = np.zeros(k + 1)
        rhs[k] = 1
        cf = np.linalg.lstsq(B, rhs, rcond=None)[0][:k]
        dens = sum(c * h for c, h in zip(cf, hist_n)) + 0.3 * sum(c * h for c, h in zip(cf, hist_r))
    v_hxc = hart.potential(dens) + exchange(dens)[1]
    v = vext + v_hxc
    d, e = tridiag(v, dx)
    eps = linalg.eigh_tridiagonal(d, e, eigvals_only=True)
    zs, ws = nodes(M, mu, e_lo)
    Omega = -4.0 * np.real(np.sum(ws[:, None] * np.log(zs[:, None] - eps[None, :])))
    E_H = 0.5 * np.sum(dens * hart.potential(dens)) * dx
    E_x = np.sum(exchange(dens)[0]) * dx
    Phi = Omega - np.sum(v_hxc * dens) * dx + E_H + E_x + s.nuclear_repulsion()
    return Phi, dens, it


def part_B():
    rows = []
    base = System([1.0, 1.0], [-0.8, 0.8], 2)
    g = Grid.around(base, dx=0.1, pad=8)
    ks = scf(base, g, "ks", tol=1e-11)
    mu = ks.mu + 0.15                            # inside the gap
    v0 = external_potential(base, g)
    e_lo = linalg.eigh_tridiagonal(*tridiag(v0, g.dx), eigvals_only=True, select="i",
                                   select_range=(0, 0))[0] - 0.3
    h = 1e-3
    for M in (8, 16, 64):
        Phi = {}
        for sgn in (-1, 0, 1):
            s = System([1.0, 1.0], [-0.8, 0.8 + sgn * h], 2)
            Phi[sgn], dens, it = grand_scf(s, g, mu, e_lo, M)
            if sgn == 0:
                n0 = dens
        F_fd = -(Phi[1] - Phi[-1]) / (2 * h)
        X2 = 0.8
        dvdX = -(g.x - X2) / ((g.x - X2) ** 2 + 1) ** 1.5     # d/dX of -1/sqrt((x-X)^2+1)
        dEnn = -1.0 * (X2 + 0.8) / ((X2 + 0.8) ** 2 + 1) ** 1.5
        F_hf = -(np.sum(n0 * dvdX) * g.dx + dEnn)
        rows.append(dict(M=M, N_electrons=float(n0.sum() * g.dx), F_finite_difference=F_fd,
                         F_hellmann_feynman=F_hf, abs_diff=abs(F_fd - F_hf)))
        print("B", json.dumps(rows[-1]), flush=True)
    return rows


if __name__ == "__main__":
    res = dict(grid_refinement=part_A(), forces=part_B())
    json.dump(res, open(os.path.join(OUT, "extra_checks.json"), "w"), indent=1)
