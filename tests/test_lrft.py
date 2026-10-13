"""Correctness tests for LRFT: each test checks one statement of THEORY.md
on the actual implementation (not on toy algebra)."""
import numpy as np
import pytest
from scipy import linalg

from lrft.core import (Grid, System, external_potential, exchange, lrft_density,
                       lrft_fixed_N, nodes, resolvent_diag, resolvent_diag_windowed,
                       scf, tridiag, ks_density)

rng = np.random.default_rng(0)


def random_h(n=60, dx=0.2):
    v = rng.normal(size=n) - 1.0
    d, e = tridiag(v, dx)
    H = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    return v, d, e, H


def test_resolvent_diag_exact():
    """Theorem 3: O(N) recursion equals the diagonal of the dense inverse."""
    v, d, e, H = random_h()
    zs = np.array([-1.3 + 0.2j, 0.5 + 1e-3j, 7.0 + 3j])
    g = resolvent_diag(d, e, zs)
    for j, z in enumerate(zs):
        ref = np.diag(np.linalg.inv(z * np.eye(len(d)) - H))
        assert np.max(np.abs(g[j] - ref)) < 1e-10 * np.max(np.abs(ref))


def test_windowed_resolvent_converges_exponentially():
    """Theorem 4: window truncation error decays exponentially in w."""
    v, d, e, H = random_h(200)
    zs = np.array([-0.5 + 0.3j])
    exact = resolvent_diag(d, e, zs)
    # z inside the band: decay length ~ v_group / Im z, so test the asymptotic regime
    errs = [np.max(np.abs(resolvent_diag_windowed(d, e, zs, w) - exact)) for w in (16, 32, 64, 96)]
    assert all(b < a for a, b in zip(errs, errs[1:]))
    assert errs[-1] < 1e-4 * errs[0]


def test_projector_is_butterworth_filter_and_N_representable():
    """Lemma 1+2: sum_j w_j (z_j-h)^-1 = r_M((h-c)/R) with occupations in [0,1]."""
    v, d, e, H = random_h(40)
    eps, U = np.linalg.eigh(H)
    M, mu, e_lo = 24, eps[10] + 0.5 * (eps[11] - eps[10]), eps[0] - 0.3
    zs, ws = nodes(M, mu, e_lo)
    P = np.zeros_like(H, dtype=complex)
    for z, w in zip(zs, ws):
        P += w * np.linalg.inv(z * np.eye(len(d)) - H)
    P = 2 * P.real  # add conjugate half
    c, R = 0.5 * (mu + e_lo), 0.5 * (mu - e_lo)
    r = 1 / (1 + ((eps - c) / R) ** M)
    assert np.allclose(P, (U * r) @ U.T, atol=1e-12)
    occ = np.linalg.eigvalsh(P)
    assert occ.min() > -1e-12 and occ.max() < 1 + 1e-12


def test_hellmann_feynman_consistency_at_finite_M():
    """Theorem 2: n_M(x) dx = dOmega_M/dv(x) exactly, Omega_M = sum_i W_M(eps_i)."""
    v, d, e, H = random_h(30, dx=0.3)
    eps0 = np.linalg.eigvalsh(H)
    M, mu, e_lo = 12, 0.5 * (eps0[5] + eps0[6]), eps0[0] - 0.4   # deliberately small M
    zs, ws = nodes(M, mu, e_lo)

    def omega(vv):
        dd, ee = tridiag(vv, 0.3)
        ev = linalg.eigh_tridiagonal(dd, ee, eigvals_only=True)
        # W_M(eps) = -2 sum_{all j} w_j log(z_j - eps); conjugate half -> 2 Re
        return -2 * 2 * np.real(np.sum(ws[:, None] * np.log(zs[:, None] - ev[None, :])))

    n = lrft_density(v, 0.3, mu, e_lo, M).density * 0.3
    h = 1e-6
    for x in (0, 7, 15, 29):
        vp, vm = v.copy(), v.copy()
        vp[x] += h
        vm[x] -= h
        fd = (omega(vp) - omega(vm)) / (2 * h)
        assert abs(fd - n[x]) < 1e-7


def test_constructive_inversion_recovers_potential():
    """Theorem 1(b): v(x) = (1/2 pi i) oint z g(x,z) dz - 1/dx^2 (contour > spectrum)."""
    v, d, e, H = random_h(50, dx=0.25)
    Rbig, K = 2.0 * np.max(np.abs(np.linalg.eigvalsh(H))), 128
    zs = Rbig * np.exp(2j * np.pi * (np.arange(K) + 0.5) / K)
    g = resolvent_diag(d, e, zs)
    hxx = np.real(np.sum((zs**2)[:, None] * g, axis=0) / K)   # dz = i z dtheta
    v_rec = hxx - 1 / 0.25**2
    assert np.max(np.abs(v_rec - v)) < 1e-10


def test_band_energy_linear_in_g_equals_sum_of_occupied_eigenvalues():
    v, d, e, H = random_h(80, dx=0.2)
    eps = np.linalg.eigvalsh(H)
    mu, e_lo = 0.5 * (eps[7] + eps[8]), eps[0] - 0.3
    res = lrft_density(v, 0.2, mu, e_lo, M=400)
    assert abs(res.band_energy - 2 * eps[:8].sum()) < 1e-8
    assert abs(res.n_elec - 16) < 1e-8


def test_harmonic_oscillator_noninteracting():
    """Physical check: 8 electrons in v = x^2/2 -> E = 2*(0.5+1.5+2.5+3.5) = 16."""
    x = np.arange(-12, 12, 0.02)
    v = 0.5 * x**2
    res, mu, _ = lrft_fixed_N(v, 0.02, 8, M=200)
    assert abs(res.band_energy - 16.0) < 2e-3          # FD discretisation error only
    n_ks, eb, _ = ks_density(v, 0.02, 8)
    assert abs(res.band_energy - eb) < 1e-9
    assert np.max(np.abs(res.density - n_ks)) < 1e-9


def test_exchange_derivative():
    n = np.linspace(0.01, 3, 50)
    e, vx = exchange(n)
    h = 1e-5  # balances FD truncation (log curvature) and iti0k0 round-off
    fd = (exchange(n + h)[0] - exchange(n - h)[0]) / (2 * h)
    assert np.max(np.abs(fd - vx) / np.abs(vx)) < 1e-6


@pytest.mark.parametrize("Z,X,N", [([2.0], [0.0], 2),            # He
                                   ([1.0, 1.0], [-0.8, 0.8], 2),  # H2
                                   ([3.0, 1.0], [-1.5, 1.5], 4),  # LiH
                                   ([1.0] * 6, [-7.5, -4.5, -1.5, 1.5, 4.5, 7.5], 6)])
def test_scf_lrft_equals_kohn_sham(Z, X, N):
    """Theorem 5: interacting self-consistent LRFT converges to the KS solution."""
    s = System(Z, X, N)
    g = Grid.around(s, dx=0.1, pad=10)
    ks = scf(s, g, "ks", tol=1e-10)
    lr = scf(s, g, "lrft", M=96, tol=1e-10)
    assert lr.converged and ks.converged
    assert abs(lr.energy - ks.energy) < 1e-7
    assert np.sum(np.abs(lr.density - ks.density)) * g.dx < 1e-6


def test_blocked_nearsighted_matches_exact_and_is_size_independent():
    """Theorem 4b in divide-and-conquer form: fixed (block, buffer) gives the
    same per-electron error for a 4x larger system."""
    from benchmarks.accuracy import dimer_chain
    from benchmarks.scaling import model_potential
    errs = []
    for n in (24, 96):
        s = dimer_chain(n)
        g = Grid.around(s, 0.1, 8)
        v = model_potential(s, g)
        res, mu, e_lo = lrft_fixed_N(v, g.dx, s.n_elec, 64)
        rb = lrft_density(v, g.dx, mu, e_lo, 64, window=(200, 480))
        errs.append(np.sum(np.abs(rb.density - res.density)) * g.dx / s.n_elec)
    assert max(errs) < 1e-6
    assert errs[1] < 3 * errs[0]
