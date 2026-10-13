"""
3D LRFT and Kohn-Sham in a localized-orbital (LCAO-on-grid) discretisation.

Why this discretisation: on a fine real-space grid the density matrix has
~10^3 grid points per localisation volume, so any grid-pointwise O(N) method
(probing, windows) carries a huge prefactor (measured; see benchmarks/three_d.py
notes). All production linear-scaling DFT codes (SIESTA, CONQUEST, ONETEP,
CP2K) therefore use localized basis functions; we do the same.  KS and LRFT
use the IDENTICAL basis, Hamiltonian, Hartree and LDA, so all differences are
due to the one-body solver only.

Basis: per H atom, s-Gaussians (alpha = 1.0, 0.3, 0.12) and a p-shell (alpha = 0.5),
multiplied by a smooth compact cutoff (1-(r/rcut)^2)^2, tabulated on the grid.
Matrices: S = X^T X dV,  H = X^T (T_FD + v) X dV  (Galerkin projection of the
grid Hamiltonian of lrft.three_d).

LRFT-3D:  P_M = r_M(S^-1 H) S^-1 = sum_j w_j (z_j S - H)^-1   (Lemma 1, generalised).
Nearsightedness (manuscript Sec. II F): rows of P for the functions of molecule-cell B
are taken from the window W(B) = all cells within R_w of B:
      P[B, W] ~ U_B r_M(Lambda) U_W^T,  (H_W, S_W) U = S_W U Lambda.
r_M is evaluated in closed form on the window spectrum (identical to the
pole sum by Lemma 1), so N(mu) and E_band are explicit for every mu:
      N(mu)  = 2 sum_B sum_k a_Bk r_M(lambda_Bk),  a_Bk = sum_{m in B} U_mk (S_W U)_mk
      E_band = 2 sum_B sum_k a_Bk lambda_Bk r_M(lambda_Bk)
Cost: N_cells x (one n_W^3 dense window problem), n_W independent of size -> O(N).
"""
from __future__ import annotations

import time

import numpy as np
from scipy import linalg, sparse
from scipy.optimize import brentq

from .three_d import Crystal, hamiltonian, hartree, ion_potential, lda_xc, r_M

S_EXPS = (1.0, 0.3, 0.12)
P_EXPS = (0.5,)


class _StaleMu(RuntimeError):
    """The warm-start chemical potential was too far off for the truncated window spectra."""


def _min_image(d, L):
    return d - L * np.round(d / L)


class LCAOBasis:
    def __init__(self, cr: Crystal, rcut=5.5, s_exps=None, p_exps=None, atom_exps=None):
        self.cr, self.rcut = cr, rcut
        s_exps = S_EXPS if s_exps is None else s_exps
        p_exps = P_EXPS if p_exps is None else p_exps
        dx = cr.dx
        n1, n2, n3 = cr.shape
        rad = int(np.ceil(rcut / dx))
        off = np.arange(-rad, rad + 1)
        rows, cols, vals = [], [], []
        self.atom_of_fn, self.kind = [], []
        fn = 0
        for a, R in enumerate(cr.pos):
            if atom_exps is not None:          # per-atom (s_exps, p_exps), e.g. one entry per element
                s_exps, p_exps = atom_exps[a]
            c = np.round(R / dx).astype(int)
            I, J, K = np.meshgrid(c[0] + off, c[1] + off, c[2] + off, indexing="ij")
            P = np.stack([I, J, K], -1).reshape(-1, 3)
            d = P * dx - R
            d = _min_image(d, cr.cell)
            r = np.linalg.norm(d, axis=1)
            m = r < rcut
            P, d, r = P[m], d[m], r[m]
            idx = np.ravel_multi_index((P[:, 0] % n1, P[:, 1] % n2, P[:, 2] % n3), cr.shape)
            cut = (1 - (r / rcut) ** 2) ** 2
            funcs = [np.exp(-al * r**2) * cut for al in s_exps]
            for al in p_exps:
                g = np.exp(-al * r**2) * cut
                funcs += [d[:, 0] * g, d[:, 1] * g, d[:, 2] * g]
            for f in funcs:
                f = f / np.sqrt(np.sum(f**2) * cr.dV)
                rows.append(idx)
                cols.append(np.full(idx.size, fn))
                vals.append(f)
                self.atom_of_fn.append(a)
                fn += 1
        self.nb = fn
        self.atom_of_fn = np.array(self.atom_of_fn)
        self.X = sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                   shape=(cr.ng, fn))
        self.Xr = self.X.tocsr()
        self._matrices()

    def _matrices(self):
        dV = self.cr.dV
        self.S = (self.X.T @ self.X).tocsr() * dV
        T = hamiltonian(self.cr, np.zeros(self.cr.shape))     # FD kinetic only
        self.T = (self.X.T @ (T @ self.X)).tocsr() * dV

    def H(self, v):
        Xv = self.Xr.multiply(v.reshape(-1, 1)).tocsc()
        return (self.T + (self.X.T @ Xv) * self.cr.dV).tocsr()

    def density_from_orbitals(self, C, chunk=256):
        n = np.zeros(self.cr.ng)
        for i in range(0, C.shape[1], chunk):
            Y = self.Xr @ C[:, i:i + chunk]
            n += np.sum(Y * Y, axis=1)
        return 2.0 * n.reshape(self.cr.shape)


def ks_lcao(basis: LCAOBasis, v):
    """Standard LCAO Kohn-Sham: dense generalised eigenproblem, O(N_b^3)."""
    nocc = basis.cr.n_elec // 2
    H = basis.H(v).toarray()
    S = basis.S.toarray()
    w, C = linalg.eigh(H, S, subset_by_index=(0, nocc), driver="gvx")
    dens = basis.density_from_orbitals(C[:, :nocc])
    return dens, 2.0 * w[:nocc].sum(), w


class LRFTLCAO:
    """Nearsighted LRFT with real-space ownership.

    Each molecule-cell B owns the grid points of its Voronoi cell. The density
    there is evaluated from the window problem W(B) (all cells within R_w):
        n(r) = 2 sum_k r_M(lambda_k) |phi_k^W(r)|^2,   r in own(B),
    i.e. n(r) depends only on the Hamiltonian within R_w of r (nearsightedness,
    manuscript Sec. II F). Electron count and band energy are closed-form in mu:
        N(mu)  = 2 sum_B sum_k c_Bk r_M(lambda_Bk),   c_Bk = int_own(B) |phi_k|^2
        E_band = 2 sum_B sum_k c_Bk lambda_Bk r_M(lambda_Bk)
    """

    def __init__(self, basis: LCAOBasis, cells, R_w=9.0, M=64, margin=0.3):
        self.b, self.M, self.margin = basis, M, margin
        cr = basis.cr
        d = _min_image(cr.pos[:, None, :] - cells[None, :, :], cr.cell)
        cell_of_atom = np.argmin(np.linalg.norm(d, axis=2), axis=1)
        cell_of_fn = cell_of_atom[basis.atom_of_fn]
        dc = np.linalg.norm(_min_image(cells[:, None, :] - cells[None, :, :], cr.cell), axis=2)
        # grid ownership: Voronoi cells of the block centres, computed with a
        # nearest-centre search over the block centres in chunks (O(N_g) memory)
        I, J, K = np.meshgrid(*[np.arange(m) for m in cr.shape], indexing="ij")
        r = np.stack([I, J, K], -1).reshape(-1, 3) * cr.dx
        owner = np.empty(cr.ng, int)
        best = np.full(cr.ng, np.inf)
        for B, cB in enumerate(cells):
            dd = _min_image(r - cB, cr.cell)
            d2 = np.einsum("ij,ij->i", dd, dd)
            m = d2 < best
            best[m] = d2[m]
            owner[m] = B
        del r, best
        self.windows, self.own, self.Xow = [], [], []
        Xr = basis.Xr
        for B in range(len(cells)):
            W = np.where(np.isin(cell_of_fn, np.where(dc[B] <= R_w)[0]))[0]
            own = np.where(owner == B)[0]
            self.windows.append(W)
            self.own.append(own)
            self.Xow.append(Xr[own][:, W].tocsr())
        self.nW = np.array([len(W) for W in self.windows])
        self.mu = None

    def _window(self, H, S, B, upper=None):
        """Window spectrum. If `upper` is given, only eigenpairs below it are
        computed (states above c + 1.5 R have r_M < 1.5^-M, negligible)."""
        W = self.windows[B]
        HW, SW = H[W][:, W].toarray(), S[W][:, W].toarray()
        if upper is None:
            lam, U = linalg.eigh(HW, SW)
        else:
            lam, U = linalg.eigh(HW, SW, subset_by_value=(-np.inf, upper), driver="gvx")
        Y = self.Xow[B] @ U                                    # phi_k on owned points
        return lam, Y

    def density(self, v, N=None):
        b = self.b
        cr = b.cr
        N = cr.n_elec if N is None else N
        H, S = b.H(v), b.S
        nB = len(self.windows)
        lams, cs, dens = [], [], np.zeros(cr.ng)
        e_lo = None
        mu = self.mu
        Ys = []
        upper = None
        if mu is not None:
            c, R = 0.5 * (mu + self.e_lo), 0.5 * (mu - self.e_lo)
            upper = c + 1.5 * R
        for B in range(nB):
            lam, Y = self._window(H, S, B, upper)
            lams.append(lam)
            cs.append(np.sum(Y * Y, axis=0) * cr.dV)
            Ys.append(None)
            if mu is not None:
                # single-pass path: accumulate with the previous mu (in-gap mu
                # gives the same density up to exp(-M) terms)
                e_lo_prev = self.e_lo
                dens[self.own[B]] = (Y * Y) @ r_M(lam, mu, e_lo_prev, self.M)
        if not any(len(l) for l in lams):          # stale mu (early SCF): truncation removed every state
            self.mu = None
            return self.density(v, N)
        e_lo = min(l[0] for l in lams if len(l)) - self.margin
        lam_all, c_all = np.concatenate(lams), np.concatenate(cs)

        def count(m):
            return 2.0 * np.dot(c_all, r_M(lam_all, m, e_lo, self.M)) - N

        ok = mu is not None and abs(e_lo - self.e_lo) < 1e-3 and abs(count(mu)) < 1e-8 * N
        if not ok:
            lo, hi = e_lo + self.margin, e_lo + self.margin + 1.0
            while count(lo) > 0:
                lo -= 0.5
            while count(hi) < 0:
                hi += 0.5
                if hi > lam_all.max() + 10.0 and upper is not None:   # truncated spectra hold < N electrons: stale mu
                    self.mu = None
                    return self.density(v, N)
            mu = brentq(count, lo, hi, xtol=1e-12)
            for B in range(nB):                                 # second pass
                lam, Y = self._window(H, S, B)
                dens[self.own[B]] = (Y * Y) @ r_M(lam, mu, e_lo, self.M)
        else:
            e_lo = self.e_lo
        self.mu, self.e_lo = mu, e_lo
        e_band = 2.0 * np.dot(c_all, lam_all * r_M(lam_all, mu, e_lo, self.M))
        return 2.0 * dens.reshape(cr.shape), e_band, mu


class KSLCAO:
    def __init__(self, basis):
        self.basis = basis

    def density(self, v):
        return ks_lcao(self.basis, v)


def scf_lcao(cr: Crystal, solver, tol=1e-6, maxiter=60, mix=0.4, ndiis=6, verbose=False, dens0=None):
    t0 = time.perf_counter()
    vion = ion_potential(cr)
    dens = np.full(cr.shape, cr.n_elec / (cr.ng * cr.dV)) if dens0 is None else dens0.copy()
    hist_n, hist_r = [], []
    t_solver, converged = 0.0, False
    for it in range(1, maxiter + 1):
        exc, vxc = lda_xc(dens)
        v = vion + hartree(cr, dens) + vxc
        ts = time.perf_counter()
        out = solver.density(v)
        nout, eband = out[0], out[1]
        t_solver += time.perf_counter() - ts
        resid = nout - dens
        err = np.sum(np.abs(resid)) * cr.dV / cr.n_elec
        if verbose:
            print(f"  it {it:2d} |dn|/N = {err:.2e}  ({time.perf_counter() - ts:.2f}s)", flush=True)
        if err < tol:
            converged = True
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
    # Harris-Foulkes energy at the input density of the last step
    vh = hartree(cr, dens)
    exc, vxc = lda_xc(dens)
    E_H = 0.5 * np.sum(dens * vh) * cr.dV
    E_xc = np.sum(dens * exc) * cr.dV
    E = eband - np.sum((vh + vxc) * dens) * cr.dV + E_H + E_xc
    return dict(energy=E, density=nout, iterations=it, time=time.perf_counter() - t0,
                t_solver=t_solver, converged=converged, dens_in=dens, eband=eband, last=out,
                E_H=E_H, E_xc=E_xc)


def cell_centres(n, a):
    g = np.array([[i, j, k] for i in range(n[0]) for j in range(n[1]) for k in range(n[2])], float)
    return (g + 0.5) * a


# ----------------------------------------------------------------------------
# C++ back end (src/lrft_cpp.cpp): identical algorithms, compiled kernels
# ----------------------------------------------------------------------------


def _csr64(A):
    A = A.tocsr()
    A.sort_indices()
    return (A.indptr.astype(np.int64), A.indices.astype(np.int64), np.ascontiguousarray(A.data))


class KSLCAOCpp:
    """KS density build in C++: dense dsygvx (lowest nocc+1 pairs) + orbital density."""

    def __init__(self, basis: LCAOBasis, nthreads=1):
        from . import _lrft_cpp
        self.cpp, self.b, self.nthreads = _lrft_cpp, basis, nthreads
        self.S = _csr64(basis.S)
        self.X = _csr64(basis.Xr)

    def density(self, v):
        b = self.b
        nocc = b.cr.n_elec // 2
        d, w = self.cpp.ks_density(*_csr64(b.H(v)), *self.S, *self.X, b.nb, nocc, self.nthreads)
        return d.reshape(b.cr.shape), 2.0 * w[:nocc].sum(), w


class KSLCAOCpp2(KSLCAOCpp):
    """KS reference with selectable LAPACK driver and density construction.
    driver: 'gvx' (bisection + inverse iteration), 'gvd' (divide and conquer, all pairs),
    'evr' (Cholesky + MRRR, lowest pairs); dm=True builds P = 2CC^T and projects it."""

    DRIVERS = {"gvx": 0, "gvd": 1, "evr": 2}

    def __init__(self, basis, driver="evr", dm=True, nthreads=1):
        super().__init__(basis, nthreads)
        self.driver, self.dm = self.DRIVERS[driver], dm
        self.t_eig = self.t_dens = 0.0

    def density(self, v):
        b = self.b
        nocc = b.cr.n_elec // 2
        d, w, te, td = self.cpp.ks_density2(*_csr64(b.H(v)), *self.S, *self.X, b.nb, nocc,
                                            self.driver, self.dm, self.nthreads)
        self.t_eig, self.t_dens = te, td
        return d.reshape(b.cr.shape), 2.0 * w[:nocc].sum(), w


class LRFTLCAOCpp(LRFTLCAO):
    """Nearsighted LRFT with the window kernel in C++ (OpenMP over windows)."""

    def __init__(self, basis: LCAOBasis, cells, R_w=12.1, M=64, margin=0.3, nthreads=1, lag_mu=False, reuse_vectors=True):
        super().__init__(basis, cells, R_w=R_w, M=M, margin=margin)
        self.lag_mu = lag_mu
        self.reuse_vectors = reuse_vectors
        from . import _lrft_cpp
        self.cpp, self.nthreads = _lrft_cpp, nthreads
        self.Wflat = np.concatenate(self.windows).astype(np.int64)
        self.Woff = np.concatenate([[0], np.cumsum([len(w) for w in self.windows])]).astype(np.int64)
        self.Oflat = np.concatenate(self.own).astype(np.int64)
        self.Ooff = np.concatenate([[0], np.cumsum([len(o) for o in self.own])]).astype(np.int64)
        self.S = _csr64(basis.S)
        self.X = _csr64(basis.Xr)
        self.Xow = None          # not needed by the C++ path
        self.e_lo = None
        self.passes = 0

    def _pass(self, Hc, mu, e_lo, upper, want_density):
        self.passes += 1
        b = self.b
        out = self.cpp.lrft_windows(*Hc, *self.S, *self.X, self.Wflat, self.Woff, self.Oflat,
                                    self.Ooff, b.nb, b.cr.dV, mu, e_lo, self.M, upper,
                                    want_density, self.nthreads)
        self._y2, self._y2off, self._lamoff = out[4], out[5], out[3]
        return out[:4]

    def _redensity(self, lam, mu, e_lo):
        """Density for a new mu from the stored squared window eigenvectors (no eigensolves)."""
        dens = np.zeros(self.b.cr.ng)
        occ = r_M(lam, mu, e_lo, self.M)
        for w in range(len(self.windows)):
            l0, l1 = self._lamoff[w], self._lamoff[w + 1]
            m = l1 - l0
            own = self.Oflat[self.Ooff[w]:self.Ooff[w + 1]]
            if m == 0:
                dens[own] = 0.0
                continue
            Y2 = self._y2[self._y2off[w]:self._y2off[w + 1]].reshape(len(own), m)
            dens[own] = Y2.astype(np.float64) @ occ[l0:l1]
        return dens

    def _upper(self, mu, e_lo):
        c, R = 0.5 * (mu + e_lo), 0.5 * (mu - e_lo)
        return c + 1.5 * R

    def _solve_mu(self, lam, c, e_lo, N, guess=None):
        def count(m):
            return 2.0 * np.dot(c, r_M(lam, m, e_lo, self.M)) - N
        lo = max(lam.min(), e_lo + 1e-6)          # mu below the spectrum -> count < 0
        if count(lo) > 0:
            raise RuntimeError("electron count positive at the bottom of the spectrum")
        hi = lo + 1.0
        while count(hi) < 0:
            hi += 0.5
            if hi > lam.max() + 10.0:
                raise _StaleMu("truncated window spectra hold fewer than N electrons")
        return brentq(count, lo, hi, xtol=1e-12)

    def density(self, v, N=None):
        try:
            return self._density(v, N)
        except _StaleMu:
            self.mu = None
            return self._density(v, N)

    def _density(self, v, N=None):
        b = self.b
        cr = b.cr
        N = cr.n_elec if N is None else N
        Hc = _csr64(b.H(v))
        tol = 1e-8 * N
        if self.mu is not None:
            # warm path: keep the filter edge e_lo (a free parameter of r_M, it only
            # has to stay below the spectrum) and the previous mu
            mu, e_lo = self.mu, self.e_lo
            dens, lam, c, _ = self._pass(Hc, mu, e_lo, self._upper(mu, e_lo), True)
            # keep the filter edge while it stays between margin/3 and 2*margin below
            # the spectrum (a looser edge only lowers rho, see Theorem 1 of the manuscript)
            if lam.size and e_lo + self.margin / 3 < lam.min() < e_lo + 2 * self.margin:   # empty: stale mu, go cold
                cnt = 2.0 * np.dot(c, r_M(lam, mu, e_lo, self.M)) - N
                e_band = 2.0 * np.dot(c, lam * r_M(lam, mu, e_lo, self.M))
                if self.lag_mu and abs(cnt) < 0.25:
                    # mu is in the gap: the window count differs from N only by the window
                    # error, so keep mu (one pass per step); the SCF mixer renormalizes to N
                    self.mu = mu
                    return 2.0 * dens.reshape(cr.shape), e_band, mu
                if abs(cnt) > tol and self.reuse_vectors:
                    # re-solve mu from the closed-form count and rebuild the density from the
                    # stored window vectors: no second eigensolver pass
                    mu = self._solve_mu(lam, c, e_lo, N)
                    dens = self._redensity(lam, mu, e_lo)
                    self.mu = mu
                    e_band = 2.0 * np.dot(c, lam * r_M(lam, mu, e_lo, self.M))
                    return 2.0 * dens.reshape(cr.shape), e_band, mu
                if abs(cnt) > tol:
                    mu_new = self._solve_mu(lam, c, e_lo, N)
                    if False:
                        # lagged update: the density of this step uses the previous mu
                        # (the SCF mixer renormalizes it to N); the new mu is used from the
                        # next step on. At SCF convergence mu and the density are consistent.
                        self.mu = mu_new
                        return 2.0 * dens.reshape(cr.shape), e_band, mu
                    mu = mu_new
                    dens, lam, c, _ = self._pass(Hc, mu, e_lo, self._upper(mu, e_lo), True)
                    e_band = 2.0 * np.dot(c, lam * r_M(lam, mu, e_lo, self.M))
                self.mu = mu
                return 2.0 * dens.reshape(cr.shape), e_band, mu
        # cold path: full window spectra, then mu, then density
        _, lam, c, _ = self._pass(Hc, 0.0, -1.0, np.inf, False)
        e_lo = lam.min() - self.margin
        mu = self._solve_mu(lam, c, e_lo, N)
        dens, lam, c, _ = self._pass(Hc, mu, e_lo, self._upper(mu, e_lo), True)
        self.mu, self.e_lo = mu, e_lo
        e_band = 2.0 * np.dot(c, lam * r_M(lam, mu, e_lo, self.M))
        return 2.0 * dens.reshape(cr.shape), e_band, mu


class KSLOBPCG:
    """Warm-started iterative KS reference (LOBPCG on the pencil (H, S)).

    The first call (cold) uses the dense C++ solver to obtain starting vectors; later calls
    start from the previous eigenvectors, the standard practice in SCF codes. Preconditioner:
    block-Jacobi inverse of (H - sigma S) with one block per molecule cell, sigma below the
    spectrum. The density is the orbital sum over the grid."""

    def __init__(self, basis: LCAOBasis, cells, extra=24, tol=1e-6, maxiter=60):
        from scipy.sparse.linalg import lobpcg  # noqa: F401
        self.b, self.extra, self.tol, self.maxiter = basis, extra, tol, maxiter
        cr = basis.cr
        d = _min_image(cr.pos[:, None, :] - cells[None, :, :], cr.cell)
        cell_of_fn = np.argmin(np.linalg.norm(d, axis=2), axis=1)[basis.atom_of_fn]
        self.blocks = [np.where(cell_of_fn == B)[0] for B in range(len(cells))]
        self.X, self.w = None, None
        self.dense = KSLCAOCpp(basis)
        self.iters = []

    def density(self, v):
        from scipy.sparse.linalg import lobpcg, LinearOperator
        b = self.b
        nocc = b.cr.n_elec // 2
        if self.X is None:
            dens, eb, w = self.dense.density(v)
            # starting vectors for the next step: lowest nocc+extra pairs, dense once
            H, S = b.H(v).toarray(), b.S.toarray()
            ww, X = linalg.eigh(H, S, subset_by_index=(0, nocc + self.extra - 1), driver="gvx")
            self.X, self.w = X, ww
            self.iters.append(0)
            return dens, eb, w
        H, S = b.H(v), b.S
        sigma = self.w[0] - 0.5
        A = (H - sigma * S).tocsr()
        inv = np.empty(b.nb, dtype=object)
        blocks = [(fB, np.linalg.inv(A[fB][:, fB].toarray())) for fB in self.blocks]

        def prec(R):
            R = np.asarray(R)
            out = np.empty_like(R)
            for fB, Ai in blocks:
                out[fB] = Ai @ R[fB]
            return out
        Mop = LinearOperator(H.shape, matvec=prec, matmat=prec, dtype=float)
        w, X, hist = lobpcg(H, self.X, B=S, M=Mop, largest=False, tol=self.tol, maxiter=self.maxiter,
                            retResidualNormsHistory=True)
        o = np.argsort(w)
        self.w, self.X = w[o], X[:, o]
        self.iters.append(len(hist))
        dens = b.density_from_orbitals(self.X[:, :nocc])
        return dens, 2.0 * self.w[:nocc].sum(), self.w
