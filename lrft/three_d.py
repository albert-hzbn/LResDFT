"""
LRFT in three dimensions, and the Kohn-Sham reference on the identical Hamiltonian.

Model: periodic supercell, real-space grid (spacing dx), 7-point finite-difference
kinetic operator, smooth local pseudopotentials  v_I(r) = -Z erf(r/rc)/r,
FFT Hartree (G=0 removed: neutralising background), LDA exchange-correlation
(Slater exchange + Perdew-Zunger 1981 correlation).

LRFT 3D algorithm (the 1D exact recursion has no 3D analogue):
  diag P_M = diag r_M((h-c)/R)  is obtained by
  * probing: the grid is coloured with spacing d in each direction, C = d^3
    probe vectors b_c (indicator of colour c).  diag(P)_i = (P b_c(i))_i + sum of
    P_ik with k != i in the same colour, which decays like the density matrix
    (nearsightedness, Sec. II F of the manuscript)  -> error exp(-gamma d dx), C intensive.
  * f(h) b via batched Lanczos (Gauss quadrature): P b ~ |b| V_K r_M(T_K) e1.
    The electron count and band energy follow from the K x K matrices T_K
    alone (N(mu) = sum_c |b_c|^2 e1' r_M(T_K^c) e1), so mu is found at no
    extra matvec cost; one more Lanczos pass assembles the density.
  Cost per SCF step: O(C K N_g), with C, K independent of system size.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.fft
from scipy import sparse, special
from scipy.optimize import brentq
from scipy.sparse.linalg import LinearOperator, lobpcg


FFT_WORKERS = 1   # threads used by the FFTs (set by the command-line tool)


# ----------------------------------------------------------------------------
# Geometry, grid, operators
# ----------------------------------------------------------------------------


@dataclass
class Crystal:
    pos: np.ndarray        # (natoms, 3) bohr
    Z: np.ndarray
    cell: np.ndarray       # (3,) orthorhombic box lengths
    shape: tuple           # grid points per axis
    rc: float = 1.0
    rcs: np.ndarray = None  # optional per-atom softening radii (default: rc for all)
    charge: int = 0         # net charge (electrons removed); neutralised by the G=0 background
    pseudo: list = None     # optional per-atom GTH pseudopotentials (lrft.gth); then Z = ionic valence charge
                            # and rcs = sqrt(2) r_loc, so the erf part below is the GTH Coulomb tail

    @property
    def atom_rc(self):
        return np.full(len(self.Z), self.rc) if self.rcs is None else np.asarray(self.rcs, float)

    @property
    def dx(self):
        return self.cell[0] / self.shape[0]

    @property
    def n_elec(self):
        return int(round(self.Z.sum())) - int(self.charge)

    @property
    def ng(self):
        return int(np.prod(self.shape))

    @property
    def dV(self):
        return float(np.prod(self.cell / np.array(self.shape)))


def h2_crystal(n=(2, 2, 2), a=4.5, bond=1.4, pts_per_cell=8, jitter=0.15, seed=0, rc=1.0):
    """n1 x n2 x n3 cells, one H2 per cell, random orientation + displacement."""
    rng = np.random.default_rng(seed)
    pos = []
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                c = (np.array([i, j, k]) + 0.5) * a + rng.normal(scale=jitter, size=3)
                u = rng.normal(size=3)
                u /= np.linalg.norm(u)
                pos += [c - 0.5 * bond * u, c + 0.5 * bond * u]
    cell = a * np.array(n, float)
    shape = tuple(int(pts_per_cell * m) for m in n)
    return Crystal(np.array(pos), np.ones(len(pos)), cell, shape, rc)


def gvectors(cr: Crystal):
    ks = [2 * np.pi * np.fft.fftfreq(m, d=L / m) for m, L in zip(cr.shape, cr.cell)]
    return np.meshgrid(*ks, indexing="ij")


def fd_kinetic_symbol(cr: Crystal):
    """Fourier symbol of the 7-point FD kinetic operator, -1/2 Lap_FD."""
    dx = cr.dx
    kx, ky, kz = gvectors(cr)
    return (3 - np.cos(kx * dx) - np.cos(ky * dx) - np.cos(kz * dx)) / dx**2


def ion_potential(cr: Crystal):
    """Periodic sum of -Z erf(r/rc)/r via FFT (G=0 dropped)."""
    kx, ky, kz = gvectors(cr)
    G2 = kx**2 + ky**2 + kz**2
    vol = float(np.prod(cr.cell))
    S = np.zeros(cr.shape, complex)
    for (x, y, z), Z, rc in zip(cr.pos, cr.Z, cr.atom_rc):
        S += Z * np.exp(-1j * (kx * x + ky * y + kz * z) - G2 * rc**2 / 4)
    with np.errstate(divide="ignore", invalid="ignore"):
        vG = -4 * np.pi / G2 * S / vol
    vG[0, 0, 0] = 0.0
    if cr.pseudo is not None:                       # GTH short-range Gaussian-polynomial part
        from .gth import local_short_range_G
        Sg = np.zeros(cr.shape, complex)
        for (x, y, z), pp in zip(cr.pos, cr.pseudo):
            Sg += np.exp(-1j * (kx * x + ky * y + kz * z)) * local_short_range_G(pp, G2)
        vG = vG + Sg / vol
    return np.real(scipy.fft.ifftn(vG, workers=FFT_WORKERS)) * cr.ng


def hartree(cr: Crystal, n):
    kx, ky, kz = gvectors(cr)
    G2 = kx**2 + ky**2 + kz**2
    nG = scipy.fft.fftn(n, workers=FFT_WORKERS)
    with np.errstate(divide="ignore", invalid="ignore"):
        vG = 4 * np.pi * nG / G2
    vG[0, 0, 0] = 0.0
    return np.real(scipy.fft.ifftn(vG, workers=FFT_WORKERS))


def lda_xc(n):
    """Slater exchange + PZ81 correlation (unpolarised). Returns (eps_xc, v_xc)."""
    n = np.maximum(n, 1e-14)
    rs = (3 / (4 * np.pi * n)) ** (1 / 3)
    ex = -0.75 * (3 / np.pi) ** (1 / 3) * n ** (1 / 3)
    vx = 4.0 / 3.0 * ex
    g, b1, b2 = -0.1423, 1.0529, 0.3334
    A, B, C, D = 0.0311, -0.048, 0.0020, -0.0116
    sr = np.sqrt(rs)
    den = 1 + b1 * sr + b2 * rs
    ec_hi = g / den
    vc_hi = ec_hi * (1 + 7 / 6 * b1 * sr + 4 / 3 * b2 * rs) / den
    lr = np.log(rs)
    ec_lo = A * lr + B + C * rs * lr + D * rs
    vc_lo = A * lr + (B - A / 3) + 2 / 3 * C * rs * lr + (2 * D - C) / 3 * rs
    hi = rs >= 1
    return ex + np.where(hi, ec_hi, ec_lo), vx + np.where(hi, vc_hi, vc_lo)


def hamiltonian(cr: Crystal, v):
    """Sparse 7-point FD Hamiltonian, periodic."""
    n1, n2, n3 = cr.shape
    dx = cr.dx
    idx = np.arange(cr.ng).reshape(cr.shape)
    rows, cols = [], []
    for ax in range(3):
        for s in (1, -1):
            rows.append(idx.ravel())
            cols.append(np.roll(idx, s, axis=ax).ravel())
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    off = sparse.csr_matrix((np.full(rows.size, -0.5 / dx**2), (rows, cols)), shape=(cr.ng, cr.ng))
    return (off + sparse.diags(v.ravel() + 3.0 / dx**2)).tocsr()


# ----------------------------------------------------------------------------
# Kohn-Sham reference: LOBPCG with the exact FD-kinetic preconditioner
# ----------------------------------------------------------------------------


class KSSolver:
    def __init__(self, cr: Crystal, extra=4, tol=1e-6, maxiter=200):
        self.cr, self.extra, self.tol, self.maxiter = cr, extra, tol, maxiter
        self.X = None
        self.tsym = fd_kinetic_symbol(cr)

    def density(self, v):
        cr = self.cr
        nocc = cr.n_elec // 2
        k = nocc + self.extra
        H = hamiltonian(cr, v)
        shift = 0.5
        vmin = v.min()
        sym = self.tsym + shift

        def prec(R):
            R = np.asarray(R).reshape(cr.shape + (-1,))
            out = np.fft.ifftn(np.fft.fftn(R, axes=(0, 1, 2)) / sym[..., None], axes=(0, 1, 2)).real
            return out.reshape(cr.ng, -1)

        if self.X is None:
            rng = np.random.default_rng(1)
            self.X = rng.normal(size=(cr.ng, k))
        Hs = H - sparse.identity(cr.ng) * (vmin - shift)   # shift: lobpcg-friendly
        w, X = lobpcg(LinearOperator(H.shape, matvec=lambda x: Hs @ x, matmat=lambda X: Hs @ X,
                                     dtype=float),
                      self.X, M=LinearOperator(H.shape, matvec=prec, matmat=prec, dtype=float),
                      largest=False, tol=self.tol, maxiter=self.maxiter)
        order = np.argsort(w)
        w, X = w[order] + (vmin - shift), X[:, order]
        self.X = X
        Xo = X[:, :nocc]
        dens = 2.0 * np.sum(Xo**2, axis=1).reshape(cr.shape) / cr.dV
        return dens, 2.0 * w[:nocc].sum(), w


def ks_dense_density(cr: Crystal, v):
    H = hamiltonian(cr, v).toarray()
    w, U = np.linalg.eigh(H)
    nocc = cr.n_elec // 2
    dens = 2.0 * np.sum(U[:, :nocc] ** 2, axis=1).reshape(cr.shape) / cr.dV
    return dens, 2.0 * w[:nocc].sum(), w


# ----------------------------------------------------------------------------
# LRFT 3D: probing + batched Lanczos Gauss quadrature of the Butterworth filter
# ----------------------------------------------------------------------------


def r_M(lam, mu, e_lo, M):
    c, R = 0.5 * (mu + e_lo), 0.5 * (mu - e_lo)
    u = (lam - c) / R
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + u**M)


def probe_vectors(cr: Crystal, d):
    """Indicator vectors of the d^3 colour classes (grid dims must be multiples of d)."""
    assert all(m % d == 0 for m in cr.shape), (cr.shape, d)
    I, J, K = np.meshgrid(*[np.arange(m) for m in cr.shape], indexing="ij")
    col = ((I % d) * d + (J % d)) * d + (K % d)
    return col.ravel()


class LRFT3D:
    def __init__(self, cr: Crystal, M=32, d=4, K=120, margin=0.3, chunk=64):
        self.cr, self.M, self.d, self.K, self.margin, self.chunk = cr, M, d, K, margin, chunk
        self.colour = probe_vectors(cr, d)
        self.C = d**3

    def _lanczos(self, H, B, K, coef=None):
        """Batched Lanczos on the columns of B. Pass 1 (coef=None): returns
        (alpha, beta) of shape (K, nb). Pass 2: returns sum_k V_k * coef[k]."""
        nb = B.shape[1]
        nrm = np.linalg.norm(B, axis=0)
        V = B / nrm
        Vp = np.zeros_like(V)
        al = np.zeros((K, nb))
        be = np.zeros((K, nb))
        acc = None if coef is None else V * coef[0]
        bprev = np.zeros(nb)
        for k in range(K):
            W = H @ V
            a = np.einsum("ij,ij->j", V, W)
            W -= V * a + Vp * bprev
            b = np.linalg.norm(W, axis=0)
            al[k], be[k] = a, b
            if k == K - 1:
                break
            Vp, V = V, W / np.where(b > 0, b, 1.0)
            bprev = b
            if coef is not None:
                acc += V * coef[k + 1]
        return (al, be, nrm) if coef is None else acc

    def density(self, v, mu_guess=None, N=None):
        cr = self.cr
        N = cr.n_elec if N is None else N
        H = hamiltonian(cr, v)
        cols = np.arange(self.C)
        chunks = [cols[i:i + self.chunk] for i in range(0, self.C, self.chunk)]
        # pass 1: tridiagonals for every probe
        tri = []
        for ch in chunks:
            B = (self.colour[:, None] == ch[None, :]).astype(float)
            tri.append(self._lanczos(H, B, self.K))
        theta, U1, Bn = [], [], []
        for al, be, nrm in tri:
            for j in range(al.shape[1]):
                T = np.diag(al[:, j]) + np.diag(be[:-1, j], 1) + np.diag(be[:-1, j], -1)
                th, U = np.linalg.eigh(T)
                theta.append(th)
                U1.append(U)
                Bn.append(nrm[j])
        e_lo = min(t[0] for t in theta) - self.margin
        w1 = [U[0] ** 2 for U in U1]

        def count(mu):
            return sum(2 * b**2 * np.dot(w, r_M(th, mu, e_lo, self.M))
                       for b, w, th in zip(Bn, w1, theta)) - N

        a, b = e_lo, e_lo + self.margin + 2.0
        while count(a) > 0:
            a -= 1.0
        while count(b) < 0:
            b += 2.0
        mu = brentq(count, a, b, xtol=1e-12)
        e_band = sum(2 * b_**2 * np.dot(w, th * r_M(th, mu, e_lo, self.M))
                     for b_, w, th in zip(Bn, w1, theta))
        # pass 2: density  n_i = 2/dV * (P b_c(i))_i
        dens = np.zeros(cr.ng)
        jglob = 0
        for ch in chunks:
            B = (self.colour[:, None] == ch[None, :]).astype(float)
            coefs = np.empty((self.K, len(ch)))
            for jj in range(len(ch)):
                U, th, bn = U1[jglob + jj], theta[jglob + jj], Bn[jglob + jj]
                coefs[:, jj] = bn * U @ (r_M(th, mu, e_lo, self.M) * U[0])
            Y = self._lanczos(H, B, self.K, coef=coefs)
            dens += np.sum(Y * B, axis=1)
            jglob += len(ch)
        return 2.0 * dens.reshape(cr.shape) / cr.dV, e_band, mu


# ----------------------------------------------------------------------------
# SCF driver
# ----------------------------------------------------------------------------


@dataclass
class SCF3DResult:
    energy: float
    density: np.ndarray
    iterations: int
    time: float
    t_solver: float
    converged: bool
    comps: dict = field(default_factory=dict)


def scf3d(cr: Crystal, solver, tol=1e-6, maxiter=60, mix=0.4, ndiis=6, verbose=False):
    t0 = time.perf_counter()
    vion = ion_potential(cr)
    dens = np.full(cr.shape, cr.n_elec / (cr.ng * cr.dV))
    hist_n, hist_r = [], []
    t_solver, converged = 0.0, False
    for it in range(1, maxiter + 1):
        exc, vxc = lda_xc(dens)
        v = vion + hartree(cr, dens) + vxc
        ts = time.perf_counter()
        out = solver.density(v)
        t_solver += time.perf_counter() - ts
        nout, eband = out[0], out[1]
        resid = nout - dens
        err = np.sum(np.abs(resid)) * cr.dV / cr.n_elec
        if verbose:
            print(f"  it {it:2d} |dn|/N = {err:.2e}  ({time.perf_counter() - ts:.1f}s)", flush=True)
        if err < tol:
            converged = True
            dens_in = dens
            break
        hist_n.append(dens.copy())
        hist_r.append(resid.copy())
        if len(hist_n) > ndiis:
            hist_n.pop(0)
            hist_r.pop(0)
        k = len(hist_n)
        if k > 1:
            Bm = np.empty((k + 1, k + 1))
            Bm[:k, :k] = [[np.vdot(ri, rj) for rj in hist_r] for ri in hist_r]
            Bm[k, :k] = Bm[:k, k] = 1.0
            Bm[k, k] = 0.0
            rhs = np.zeros(k + 1)
            rhs[k] = 1.0
            coef = np.linalg.lstsq(Bm, rhs, rcond=None)[0][:k]
            dens = sum(c * h for c, h in zip(coef, hist_n)) + mix * sum(c * r for c, r in zip(coef, hist_r))
        else:
            dens = dens + mix * resid
        dens = np.maximum(dens, 1e-12)
        dens *= cr.n_elec / (dens.sum() * cr.dV)
    else:
        dens_in = dens
    # Harris-Foulkes energy at the input density of the last step
    vh = hartree(cr, dens_in)
    exc, vxc = lda_xc(dens_in)
    E_H = 0.5 * np.sum(dens_in * vh) * cr.dV
    E_xc = np.sum(dens_in * exc) * cr.dV
    E = eband - np.sum((vh + vxc) * dens_in) * cr.dV + E_H + E_xc
    return SCF3DResult(E, nout, it, time.perf_counter() - t0, t_solver, converged,
                       dict(E_band=eband, E_H=E_H, E_xc=E_xc))
