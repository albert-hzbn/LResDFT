"""
Local Resolvent Field Theory (LRFT) -- reference implementation (1D).

Fundamental variable: the local resolvent field g(x, z) = <x|(z - h)^-1|x>,
sampled at M complex energies z_j on a circle through the chemical potential.

    n(x)    = (2/dx) * sum_j w_j g(x, z_j)             (density, linear in g)
    E_band  = 2 * sum_j w_j z_j sum_x g(x, z_j)        (band energy, linear in g)
    v(x)    = lim z^2 (g - 1/z) - 1/dx^2               (explicit inversion)

with z_j = c + R zeta_j, zeta_j^M = -1, w_j = R zeta_j / M  (THEORY.md, Lemma 1).
Each g(., z_j) is computed EXACTLY in O(N_g) by a two-sided continued-fraction
recursion (Lemma 3 of the manuscript) or, for the dimension-agnostic nearsighted variant,
from local windows (nearsightedness, Sec. II F).

Also contains the Kohn-Sham reference solvers on the *identical* Hamiltonian
(same grid, same Hartree, same exchange) so that every difference measured is
due only to the LRFT treatment of the one-body problem.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange
from scipy import linalg, special
from scipy.optimize import brentq

# ----------------------------------------------------------------------------
# System definition
# ----------------------------------------------------------------------------


@dataclass
class System:
    """1D soft-Coulomb molecule/chain: nuclei with charges Z at positions X."""

    Z: np.ndarray
    X: np.ndarray
    n_elec: int
    soft: float = 1.0

    def __post_init__(self):
        self.Z = np.asarray(self.Z, float)
        self.X = np.asarray(self.X, float)

    def nuclear_repulsion(self) -> float:
        e = 0.0
        for i in range(len(self.Z)):
            for j in range(i):
                e += self.Z[i] * self.Z[j] / np.sqrt((self.X[i] - self.X[j]) ** 2 + self.soft**2)
        return e


@dataclass
class Grid:
    x: np.ndarray
    dx: float

    @classmethod
    def around(cls, system: System, dx: float = 0.1, pad: float = 10.0) -> "Grid":
        lo, hi = system.X.min() - pad, system.X.max() + pad
        n = int(np.ceil((hi - lo) / dx)) + 1
        return cls(lo + dx * np.arange(n), dx)

    @property
    def n(self) -> int:
        return self.x.size


def external_potential(system: System, grid: Grid) -> np.ndarray:
    v = np.zeros(grid.n)
    for Z, X in zip(system.Z, system.X):
        v -= Z / np.sqrt((grid.x - X) ** 2 + system.soft**2)
    return v


# ----------------------------------------------------------------------------
# Hartree (soft-Coulomb, open boundaries via zero-padded FFT convolution)
# ----------------------------------------------------------------------------


class Hartree:
    def __init__(self, grid: Grid, soft: float = 1.0):
        n = grid.n
        self.n = n
        self.m = 1 << int(np.ceil(np.log2(2 * n)))
        k = np.arange(self.m)
        d = np.minimum(k, self.m - k) * grid.dx
        self.kern_hat = np.fft.rfft(1.0 / np.sqrt(d**2 + soft**2))
        self.dx = grid.dx

    def potential(self, dens: np.ndarray) -> np.ndarray:
        nh = np.fft.rfft(dens, self.m)
        return np.fft.irfft(nh * self.kern_hat, self.m)[: self.n] * self.dx


# ----------------------------------------------------------------------------
# Exchange: closed form for the 1D soft-Coulomb (a=1) uniform gas (CAS-verified)
#   e_x(n) = -(1/pi^2) [b IK0(b) + b K1(b) - 1],  v_x = -(1/pi) IK0(b),  b = pi n
# ----------------------------------------------------------------------------


def exchange(dens: np.ndarray):
    b = np.pi * np.maximum(dens, 1e-14)
    ik0 = special.iti0k0(b)[1]
    bk1 = np.where(b < 1e-6, 1.0 + 0.5 * b**2 * np.log(b / 2), b * special.k1(b))
    e_density = -(b * ik0 + bk1 - 1.0) / np.pi**2  # energy per unit length
    v = -ik0 / np.pi
    return e_density, v


# ----------------------------------------------------------------------------
# One-body Hamiltonian (3-point finite differences, hard walls)
# ----------------------------------------------------------------------------


def tridiag(v: np.ndarray, dx: float):
    d = v + 1.0 / dx**2
    e = np.full(v.size - 1, -0.5 / dx**2)
    return d, e


# ----------------------------------------------------------------------------
# LRFT kernels
# ----------------------------------------------------------------------------


def nodes(M: int, mu: float, e_lo: float):
    """Upper-half-plane nodes/weights of the Butterworth-trapezoid rule.

    The circle passes through e_lo (below the spectrum) and mu.
    Only j < M/2 are returned: the lower half are complex conjugates.
    """
    assert M % 2 == 0
    c, R = 0.5 * (mu + e_lo), 0.5 * (mu - e_lo)
    zeta = np.exp(1j * np.pi * (2 * np.arange(M // 2) + 1) / M)
    return c + R * zeta, R * zeta / M


@njit(cache=True, parallel=True)
def resolvent_diag(d, e, zs):
    """g[j, i] = ((z_j - h)^-1)_ii for tridiagonal h=(d, e). Exact, O(N) per z.

    Two-sided continued fraction (manuscript Lemma 3):
      L_i = a_i - e_{i-1}^2 / L_{i-1},  U_i = a_i - e_i^2 / U_{i+1},
      G_ii = 1 / (L_i + U_i - a_i),     a_i = z - d_i.
    """
    nz, n = zs.size, d.size
    out = np.empty((nz, n), np.complex128)
    for j in prange(nz):
        z = zs[j]
        L = np.empty(n, np.complex128)
        L[0] = z - d[0]
        for i in range(1, n):
            L[i] = (z - d[i]) - e[i - 1] * e[i - 1] / L[i - 1]
        U = z - d[n - 1]
        out[j, n - 1] = 1.0 / L[n - 1]
        for i in range(n - 2, -1, -1):
            U = (z - d[i]) - e[i] * e[i] / U
            out[j, i] = 1.0 / (L[i] + U - (z - d[i]))
    return out


@njit(cache=True, parallel=True)
def resolvent_diag_windowed(d, e, zs, w):
    """Nearsighted variant: G_ii from the window [i-w, i+w] only.

    Uses the same continued fractions but truncated at distance w
    (equivalent to Dirichlet-truncating h to the window). The error decays
    like |lambda|^(2w) (Remark 3 of cas_verification.py), independent of system size, and in d
    dimensions the identical construction costs O(N_g * w^d).
    """
    nz, n = zs.size, d.size
    out = np.empty((nz, n), np.complex128)
    for j in prange(nz):
        z = zs[j]
        for i in range(n):
            lo = max(0, i - w)
            hi = min(n - 1, i + w)
            L = z - d[lo]
            for k in range(lo + 1, i + 1):
                L = (z - d[k]) - e[k - 1] * e[k - 1] / L
            U = z - d[hi]
            for k in range(hi - 1, i - 1, -1):
                U = (z - d[k]) - e[k] * e[k] / U
            out[j, i] = 1.0 / (L + U - (z - d[i]))
    return out


@njit(cache=True, parallel=True)
def resolvent_diag_blocked(d, e, zs, block, buf):
    """Nearsighted divide-and-conquer form: the grid is cut into blocks of
    `block` points; each block's diagonal is computed EXACTLY on the local
    window [block - buf, block + buf] (Dirichlet-truncated h), and only the
    central block is kept. Error O(exp(-gamma * buf)) (nearsightedness), cost
    O(N_g (block + 2 buf)/block) per pole -- linear, and in d dimensions the
    same construction costs O(N_g ((block+2buf)/block)^d * local solve).
    """
    nz, n = zs.size, d.size
    nb = (n + block - 1) // block
    out = np.empty((nz, n), np.complex128)
    for t in prange(nz * nb):
        j = t // nb
        kb = t % nb
        z = zs[j]
        b0 = kb * block
        b1 = min(n, b0 + block)
        lo = max(0, b0 - buf)
        hi = min(n, b1 + buf)
        m = hi - lo
        L = np.empty(m, np.complex128)
        L[0] = z - d[lo]
        for i in range(1, m):
            L[i] = (z - d[lo + i]) - e[lo + i - 1] * e[lo + i - 1] / L[i - 1]
        U = z - d[hi - 1]
        if hi - 1 < b1:
            out[j, hi - 1] = 1.0 / L[m - 1]
        for i in range(m - 2, b0 - lo - 1, -1):
            U = (z - d[lo + i]) - e[lo + i] * e[lo + i] / U
            if lo + i < b1:
                out[j, lo + i] = 1.0 / (L[i] + U - (z - d[lo + i]))
    return out


def lowest_eigenvalue(d, e) -> float:
    return float(linalg.eigh_tridiagonal(d, e, eigvals_only=True, select="i",
                                         select_range=(0, 0), lapack_driver="stebz")[0])


@dataclass
class LRFTResult:
    density: np.ndarray
    band_energy: float
    n_elec: float


def lrft_density(v, dx, mu, e_lo, M, window=None):
    """window: None (exact O(N) recursion), int w (per-point windows), or
    (block, buf) tuple (blocked nearsighted divide-and-conquer)."""
    d, e = tridiag(v, dx)
    zs, ws = nodes(M, mu, e_lo)
    if window is None:
        g = resolvent_diag(d, e, zs)
    elif isinstance(window, tuple):
        g = resolvent_diag_blocked(d, e, zs, window[0], window[1])
    else:
        g = resolvent_diag_windowed(d, e, zs, window)
    # factor 2 (spin) x 2 (conjugate pairs) -> 4 Re(...)
    occ = 4.0 * np.real(ws @ g)                     # occupation per grid site
    dens = occ / dx
    e_band = 4.0 * np.real(np.sum((ws * zs) @ g))
    return LRFTResult(dens, e_band, occ.sum())


def lrft_fixed_N(v, dx, N, M, mu_guess=None, margin=0.3, window=None, tol_N=1e-10):
    """Find mu with LRFT electron count = N (|dN| < tol_N * N).

    Secant iterations warm-started from mu_guess (the previous SCF step);
    falls back to a bracketing Brent search when no guess is available or
    the secant step leaves the physical range.
    """
    d, e = tridiag(v, dx)
    e0 = lowest_eigenvalue(d, e)
    e_lo = e0 - margin
    tol = tol_N * max(N, 1)
    cache = {}

    def count(mu):
        r = lrft_density(v, dx, mu, e_lo, M, window)
        cache[mu] = r
        return r.n_elec - N

    if mu_guess is not None and mu_guess > e0:
        x0, x1 = mu_guess, mu_guess + 1e-3
        f0, f1 = count(x0), count(x1)
        for _ in range(30):
            if abs(f1) < tol:
                return cache[x1], x1, e_lo
            if f1 == f0:
                break
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            if not (e0 < x2 < x1 + 2.0):
                break
            x0, f0, x1, f1 = x1, f1, x2, count(x2)
    a, b = e0 + 1e-3, e0 + 1.0
    fa, fb = count(a), count(b)
    while fa > 0:
        a = e0 + 0.5 * (a - e0)
        fa = count(a)
    while fb < 0:
        b = b + 2 * (b - a)
        fb = count(b)
    mu = brentq(count, a, b, xtol=1e-13, rtol=1e-15, maxiter=200)
    return cache.get(mu) or lrft_density(v, dx, mu, e_lo, M, window), mu, e_lo


# ----------------------------------------------------------------------------
# Kohn-Sham reference solvers (identical Hamiltonian)
# ----------------------------------------------------------------------------


def ks_density(v, dx, N, method="tridiag"):
    """Closed-shell KS density. method: 'tridiag' (1D-specialised LAPACK
    bisection+inverse iteration, O(N_g N_occ)) or 'dense' (generic O(N_g^3)
    diagonalisation, the cost class of KS in 2D/3D bases)."""
    nocc = N // 2
    d, e = tridiag(v, dx)
    if method == "tridiag":
        w, V = linalg.eigh_tridiagonal(d, e, select="i", select_range=(0, nocc - 1))
    elif method == "dense":
        H = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        w, V = linalg.eigh(H, subset_by_index=(0, nocc - 1), driver="evr")
    else:
        raise ValueError(method)
    dens = 2.0 * np.sum(V**2, axis=1) / dx
    return dens, 2.0 * w.sum(), w


# ----------------------------------------------------------------------------
# Self-consistent field driver (shared by LRFT and KS)
# ----------------------------------------------------------------------------


@dataclass
class SCFResult:
    energy: float
    density: np.ndarray
    mu: float
    iterations: int
    time: float
    components: dict = field(default_factory=dict)
    converged: bool = True


def scf(system: System, grid: Grid, solver="lrft", M=64, method="tridiag", window=None,
        tol=1e-9, maxiter=200, mix=0.5, ndiis=6, margin=0.3, verbose=False):
    t0 = time.perf_counter()
    dx, N = grid.dx, system.n_elec
    vext = external_potential(system, grid)
    hart = Hartree(grid, system.soft)
    # initial guess: superposition of normalised Gaussians
    dens = np.zeros(grid.n)
    for Z, X in zip(system.Z, system.X):
        dens += np.exp(-((grid.x - X) ** 2))
    dens *= N / (dens.sum() * dx)
    mu = None
    hist_n, hist_r = [], []
    converged = False
    for it in range(1, maxiter + 1):
        vH = hart.potential(dens)
        ex, vx = exchange(dens)
        veff = vext + vH + vx
        if solver == "lrft":
            res, mu, _ = lrft_fixed_N(veff, dx, N, M, mu_guess=mu, margin=margin, window=window)
            nout, eband = res.density, res.band_energy
        else:
            nout, eband, eps = ks_density(veff, dx, N, method)
            mu = eps[-1]
        resid = nout - dens
        err = np.sum(np.abs(resid)) * dx
        if verbose:
            print(f"  it {it:3d}  |dn| = {err:.3e}")
        if err < tol:
            converged = True
            dens = nout
            break
        # Pulay/DIIS density mixing
        hist_n.append(dens.copy())
        hist_r.append(resid.copy())
        if len(hist_n) > ndiis:
            hist_n.pop(0)
            hist_r.pop(0)
        k = len(hist_n)
        if k > 1:
            B = np.empty((k + 1, k + 1))
            B[:k, :k] = np.array([[ri @ rj for rj in hist_r] for ri in hist_r]) * dx
            B[k, :k] = B[:k, k] = 1.0
            B[k, k] = 0.0
            rhs = np.zeros(k + 1)
            rhs[k] = 1.0
            try:
                coef = np.linalg.solve(B, rhs)[:k]
                nbar = sum(c * h for c, h in zip(coef, hist_n))
                rbar = sum(c * r for c, r in zip(coef, hist_r))
                dens = nbar + mix * rbar
            except np.linalg.LinAlgError:
                dens = dens + mix * resid
        else:
            dens = dens + mix * resid
        dens = np.maximum(dens, 0.0)
        dens *= N / (dens.sum() * dx)
    # energy at the final (output) density, evaluated with the input potential
    # consistently (Harris-Foulkes form is variational to 2nd order)
    vH = hart.potential(dens)
    ex, vx = exchange(dens)
    E_H = 0.5 * np.sum(dens * vH) * dx
    E_x = np.sum(ex) * dx
    E_nn = system.nuclear_repulsion()
    veff = vext + vH + vx
    if solver == "lrft":
        res, mu, _ = lrft_fixed_N(veff, dx, N, M, mu_guess=mu, margin=margin, window=window)
        eband = res.band_energy
    else:
        _, eband, eps = ks_density(veff, dx, N, method)
    E = eband - np.sum((vH + vx) * dens) * dx + E_H + E_x + E_nn
    Ts = eband - np.sum(veff * dens) * dx
    comps = dict(T_s=Ts, E_ext=np.sum(vext * dens) * dx, E_H=E_H, E_x=E_x, E_nn=E_nn, E_band=eband)
    return SCFResult(E, dens, mu, it, time.perf_counter() - t0, comps, converged)
