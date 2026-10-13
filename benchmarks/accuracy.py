"""Accuracy validation.

(A) Against the EXACT interacting many-body solution (2 electrons, full
    configuration-space diagonalisation on the same grid/Hamiltonian):
    He, H2 binding curve.  Shows LRFT is an n-body approximation of the same
    quality as KS-DFT with the same exchange functional.
(B) LRFT vs KS on larger systems (agreement to the convergence threshold).
(C) Error vs number of poles M against the a-priori bound of Lemma 2.
Writes results/accuracy.json.
"""
import json
import os
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lrft.core import (Grid, System, external_potential, lrft_fixed_N, scf, tridiag, ks_density,
                       exchange, Hartree)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)


def exact_two_electron(system, grid):
    """Exact singlet ground state for 2 electrons: (h x 1 + 1 x h + w12) Psi = E Psi."""
    d, e = tridiag(external_potential(system, grid), grid.dx)
    h = sparse.diags([e, d, e], [-1, 0, 1], format="csr")
    I = sparse.identity(grid.n, format="csr")
    x = grid.x
    w12 = 1.0 / np.sqrt((x[:, None] - x[None, :]) ** 2 + system.soft**2)
    H = sparse.kron(h, I) + sparse.kron(I, h) + sparse.diags(w12.ravel())
    E, _ = eigsh(H, k=1, which="SA", tol=1e-10)
    return E[0] + system.nuclear_repulsion()


def part_A():
    rows = []
    cases = [("He", System([2.0], [0.0], 2))]
    for R in [1.0, 1.6, 2.0, 3.0, 4.0, 6.0]:
        cases.append((f"H2 R={R}", System([1.0, 1.0], [-R / 2, R / 2], 2)))
    for name, s in cases:
        g = Grid.around(s, dx=0.1, pad=8)
        Eex = exact_two_electron(s, g)
        ks = scf(s, g, "ks", tol=1e-10)
        lr = scf(s, g, "lrft", M=96, tol=1e-10)
        rows.append(dict(system=name, E_exact=Eex, E_KS=ks.energy, E_LRFT=lr.energy,
                         err_KS=ks.energy - Eex, err_LRFT=lr.energy - Eex,
                         LRFT_minus_KS=lr.energy - ks.energy))
        print(f"{name:10s} exact {Eex: .6f}  KS-LDAx {ks.energy: .6f}  LRFT-LDAx {lr.energy: .6f}"
              f"  LRFT-KS {lr.energy - ks.energy: .1e}")
    # binding-curve shape: errors relative to the dissociation limit R=6
    return rows


def dimer_chain(n_dimers, b=1.4, a=4.4):
    X = []
    for k in range(n_dimers):
        X += [k * a - b / 2, k * a + b / 2]
    X = np.array(X) - np.mean(X)
    return System(np.ones(2 * n_dimers), X, 2 * n_dimers)


def part_B():
    rows = []
    systems = [("LiH", System([3.0, 1.0], [-1.5, 1.5], 4)),
               ("Be", System([4.0], [0.0], 4)),
               ("(H2)_8 chain", dimer_chain(8)),
               ("(H2)_32 chain", dimer_chain(32)),
               ("H_20 uniform chain (small gap)", System(np.ones(20), 2.0 * (np.arange(20) - 9.5), 20))]
    for name, s in systems:
        g = Grid.around(s, dx=0.1, pad=8)
        ks = scf(s, g, "ks", tol=1e-10)
        lr = scf(s, g, "lrft", M=128, tol=1e-10)
        dn = np.sum(np.abs(lr.density - ks.density)) * g.dx / s.n_elec
        rows.append(dict(system=name, N=s.n_elec, E_KS=ks.energy, E_LRFT=lr.energy,
                         dE=lr.energy - ks.energy, dE_per_e=(lr.energy - ks.energy) / s.n_elec,
                         density_L1_per_e=dn))
        print(f"{name:32s} N={s.n_elec:3d}  E_KS {ks.energy: .8f}  E_LRFT {lr.energy: .8f}"
              f"  dE/N {(lr.energy - ks.energy) / s.n_elec: .1e}  |dn|/N {dn:.1e}")
    return rows


def part_C():
    """Error vs M at the converged KS potential, compared with the bound of Lemma 2."""
    s = dimer_chain(16)
    g = Grid.around(s, dx=0.1, pad=8)
    ks = scf(s, g, "ks", tol=1e-11)
    vext = external_potential(s, g)
    vH = Hartree(g).potential(ks.density)
    veff = vext + vH + exchange(ks.density)[1]
    n_ks, eb_ks, eps_occ = ks_density(veff, g.dx, s.n_elec)
    d, e = tridiag(veff, g.dx)
    from scipy import linalg
    eps = linalg.eigh_tridiagonal(d, e, eigvals_only=True, select="i",
                                  select_range=(0, s.n_elec // 2 + 5))
    homo, lumo = eps[s.n_elec // 2 - 1], eps[s.n_elec // 2]
    rows = []
    for M in [8, 16, 24, 32, 48, 64, 96, 128, 192]:
        res, mu, e_lo = lrft_fixed_N(veff, g.dx, s.n_elec, M)
        c, R = 0.5 * (mu + e_lo), 0.5 * (mu - e_lo)
        u = np.abs((eps - c) / R)
        rho = max(np.max(u[u < 1]), np.max(1 / u[u > 1]))
        bound = s.n_elec * rho**M / (1 + rho**M)     # Lemma 2 per-state bound x N
        err_n = np.sum(np.abs(res.density - n_ks)) * g.dx
        err_E = abs(res.band_energy - eb_ks)
        rows.append(dict(M=M, err_density_L1=err_n, err_band_energy=err_E, rho=rho, bound=bound))
        print(f"M={M:4d}  |dn|_1 {err_n:.2e}  |dE_band| {err_E:.2e}  rho {rho:.4f}  bound N*rho^M {bound:.2e}")
    return dict(gap=lumo - homo, rows=rows)


if __name__ == "__main__":
    print("== (A) vs exact many-body ==")
    A = part_A()
    print("== (B) LRFT vs KS ==")
    B = part_B()
    print("== (C) convergence in M ==")
    C = part_C()
    json.dump(dict(exact=A, ks=B, convergence=C), open(os.path.join(OUT, "accuracy.json"), "w"), indent=1)
