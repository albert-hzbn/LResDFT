"""Goedecker-Teter-Hutter (GTH/HGH) norm-conserving pseudopotentials and a GTH-matched Gaussian
basis on the real-space grid.

V_loc(r) = -Z erf(r/(sqrt2 r_loc))/r + exp(-(r/r_loc)^2/2) [C1 + C2 (r/r_loc)^2 + C3 (r/r_loc)^4 + C4 (r/r_loc)^6]
V_nl     = sum_{l,m} sum_{ij} |p_i^lm> h^l_ij <p_j^lm|,
p_i^lm(r) = sqrt2 r^{l+2(i-1)} exp(-r^2/(2 r_l^2)) / (r_l^{l+(4i-1)/2} sqrt(Gamma(l+(4i-1)/2))) Y_lm(r^)
(Goedecker, Teter, Hutter, PRB 54, 1703 (1996); Hartwigsen, Goedecker, Hutter, PRB 58, 3641 (1998)).
Parameters and the valence basis sets (MOLOPT/cc-type GTH bases) are read from PySCF's data files.
The erf part of V_loc is the existing Gaussian-ion term of lrft.three_d (r_c = sqrt2 r_loc); only the
short-range polynomial-Gaussian part is added here, analytically in reciprocal space.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.special import gamma

from .three_d import Crystal, hamiltonian  # noqa: F401
from .lcao3d import LCAOBasis, _min_image


def load_gth(sym, name="gth-pade", q=None):
    """Parameters of one element.  q = number of valence electrons; default: the largest q listed
    (semi-core included, e.g. Li q3, Be q4), which is the more transferable choice."""
    import os
    import re
    from pyscf.pbc.gto import pseudo
    text = open(os.path.join(os.path.dirname(pseudo.__file__), name + ".dat")).read()
    segs, cur = [], None
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if re.match(r"^[A-Z][a-z]?\s+GTH", line):
            cur = [line]
            segs.append(cur)
        elif cur is not None:
            cur.append(line)
    cand = [(int(re.search(r"-q(\d+)", g[0]).group(1)), g) for g in segs if g[0].split()[0] == sym]
    if not cand:
        raise ValueError(f"no {name} pseudopotential for {sym}")
    qs = {k: g for k, g in cand}
    seg = qs[q] if q is not None else qs[max(qs)]
    nel, rloc, nC, C, nch, *ch = _parse_seg(seg)
    channels = []
    for l, c in enumerate(ch):
        rl, n, h = c
        if n == 0:
            continue
        H = np.zeros((n, n))
        for i, row in enumerate(h):
            for j, v in enumerate(row):
                H[i, i + j] = H[i + j, i] = v
        channels.append((l, rl, H))
    return dict(sym=sym, zion=float(sum(nel)), rloc=rloc, C=list(C), channels=channels, q=max(qs) if q is None else q)


def _parse_seg(seg):
    """Parse one CP2K-format GTH segment -> (nelec per l, rloc, nC, C, nchannels, *channels)."""
    toks = [l.split() for l in seg[1:] if l.strip()]
    nel = [int(x) for x in toks[0]]
    rloc, nC = float(toks[1][0]), int(toks[1][1])
    C = [float(x) for x in toks[1][2:2 + nC]]
    nch = int(toks[2][0])
    ch, k = [], 3
    for _l in range(nch):
        rl, n = float(toks[k][0]), int(toks[k][1])
        h = []
        first = [float(x) for x in toks[k][2:]]
        if n:
            h.append(first)
            for _i in range(1, n):
                k += 1
                h.append([float(x) for x in toks[k]])
        k += 1
        ch.append([rl, n, h])
    return nel, rloc, nC, C, nch, *ch


def local_short_range_G(pp, G2):
    """FT of the Gaussian-polynomial part of V_loc, plus the alpha-Z G=0 term of the erf part."""
    rl, C = pp["rloc"], list(pp["C"]) + [0.0] * (4 - len(pp["C"]))
    x2 = G2 * rl**2
    P = (C[0] + C[1] * (3 - x2) + C[2] * (15 - 10 * x2 + x2**2) + C[3] * (105 - 105 * x2 + 21 * x2**2 - x2**3))
    out = np.sqrt(8 * np.pi**3) * rl**3 * np.exp(-x2 / 2) * P
    out = np.where(G2 == 0, out + 2 * np.pi * pp["zion"] * rl**2, out)   # G=0: alpha Z term
    return out


def _real_ylm_poly(l, d):
    """r^l Y_lm(r^) for l=0,1 (real, orthonormal on the sphere) -> list of arrays."""
    if l == 0:
        return [np.full(len(d), 0.5 / np.sqrt(np.pi))]
    if l == 1:
        c = np.sqrt(3 / (4 * np.pi))
        return [c * d[:, 1], c * d[:, 2], c * d[:, 0]]
    raise NotImplementedError("GTH projectors with l >= 2 (elements beyond Ne) are not implemented")


def projector_radial(l, i, rl, r):
    """Radial part of p_i^l divided by r^l (so that r^l Y_lm is supplied separately)."""
    e = l + (4 * i - 1) / 2.0                     # i = 1,2,...
    return np.sqrt(2.0) * r ** (2 * (i - 1)) * np.exp(-r**2 / (2 * rl**2)) / (rl**e * np.sqrt(gamma(e)))


class PseudoBasis(LCAOBasis):
    """GTH-matched Gaussian basis tabulated on the grid + separable nonlocal pseudopotential.

    All matrices (S, T, V_nl) are grid quadratures of the same tabulated functions, so the grid
    spacing must resolve the tightest basis functions (check with `overlap_error`)."""

    def __init__(self, cr: Crystal, basis="gth-tzv2p", rcut=7.0, taper=1.5, prune=1e-7, analytic=True):
        from pyscf import gto
        self.cr, self.rcut, self.basis_name, self.analytic = cr, rcut, basis, analytic
        dx = cr.dx
        n1, n2, n3 = cr.shape
        rad = int(np.ceil(rcut / dx))
        off = np.arange(-rad, rad + 1)
        rows, cols, vals = [], [], []
        gr = [[], [], []]                                            # gradient components
        self.atom_of_fn, fn = [], 0
        for a, (R, pp) in enumerate(zip(cr.pos, cr.pseudo)):
            sym = pp["sym"]
            mol = gto.M(atom=[(sym, tuple(R))], basis={sym: gto.basis.load(basis, sym)}, unit="Bohr", spin=None, verbose=0)
            c = np.round(R / dx).astype(int)
            I, J, K = np.meshgrid(c[0] + off, c[1] + off, c[2] + off, indexing="ij")
            P = np.stack([I, J, K], -1).reshape(-1, 3)
            d = _min_image(P * dx - R, cr.cell)
            r = np.linalg.norm(d, axis=1)
            m = r < rcut
            P, d, r = P[m], d[m], r[m]
            idx = np.ravel_multi_index((P[:, 0] % n1, P[:, 1] % n2, P[:, 2] % n3), cr.shape)
            t = np.clip((r - (rcut - taper)) / taper, 0, 1)
            sw = 1 - t**3 * (10 - 15 * t + 6 * t**2)                 # C2 switch, 1 -> 0
            dsw = np.where((t > 0) & (t < 1), -30 * t**2 * (1 - t) ** 2 / taper, 0.0)
            coords = R + d                                           # un-wrapped coordinates
            nao = mol.nao
            acc = [[[], [], [], [], []] for _ in range(nao)]         # per function: idx, f, gx, gy, gz
            for c0 in range(0, len(r), 50000):
                sl = slice(c0, c0 + 50000)
                ao = mol.eval_gto("GTOval_sph_deriv1", coords[sl])  # (4, n, nao): f, df/dx, df/dy, df/dz
                swc, dswc, dc, rc_ = sw[sl], dsw[sl], d[sl], np.maximum(r[sl], 1e-12)
                for k in range(nao):
                    f = ao[0, :, k] * swc
                    g = [ao[1 + q, :, k] * swc + ao[0, :, k] * dswc * dc[:, q] / rc_ for q in range(3)]
                    keep = np.abs(f) > prune
                    keep |= np.max(np.abs(np.array(g)), axis=0) > prune
                    acc[k][0].append(idx[sl][keep])
                    acc[k][1].append(f[keep])
                    for q in range(3):
                        acc[k][2 + q].append(g[q][keep])
            for k in range(nao):
                ii = np.concatenate(acc[k][0])
                rows.append(ii)
                cols.append(np.full(ii.size, fn))
                vals.append(np.concatenate(acc[k][1]))
                for q in range(3):
                    gr[q].append(np.concatenate(acc[k][2 + q]))
                self.atom_of_fn.append(a)
                fn += 1
        self.nb = fn
        self.atom_of_fn = np.array(self.atom_of_fn)
        rr, cc = np.concatenate(rows), np.concatenate(cols)
        mk = lambda v: sparse.csc_matrix((np.concatenate(v), (rr, cc)), shape=(cr.ng, fn))
        self.X = mk(vals)
        self.Xr = self.X.tocsr()
        self.Gr = [mk(g).tocsr() for g in gr]
        self._matrices()

    def _matrices(self):
        cr, dV = self.cr, self.cr.dV
        self.S = (self.X.T @ self.X).tocsr() * dV
        self.T = sum((g.T @ g).tocsr() for g in self.Gr) * (0.5 * dV)
        if self.analytic:       # exact Gaussian integrals for S and T; the grid only enters through potentials
            from pyscf import gto
            mol = gto.M(atom=[(pp["sym"], tuple(R)) for R, pp in zip(cr.pos, cr.pseudo)],
                        basis={pp["sym"]: gto.basis.load(self.basis_name, pp["sym"]) for pp in cr.pseudo},
                        unit="Bohr", spin=None, verbose=0)
            assert mol.nao == self.nb
            thr = lambda M: sparse.csr_matrix(np.where(np.abs(M) > 1e-9, M, 0.0))
            self.S, self.T = thr(mol.intor("int1e_ovlp")), thr(mol.intor("int1e_kin"))
        self.Vnl = self._nonlocal()
        self.T = (self.T + self.Vnl).tocsr()            # fold V_nl into the v-independent part
        del self.Gr

    def overlap_error(self):
        """Max |S_grid - S_analytic| over the basis (quadrature check of the grid spacing)."""
        from pyscf import gto
        errs = []
        S = self.S.toarray()
        i0 = 0
        for a, pp in enumerate(self.cr.pseudo):
            sym = pp["sym"]
            mol = gto.M(atom=[(sym, tuple(self.cr.pos[a]))], basis={sym: gto.basis.load(self.basis_name, sym)},
                        unit="Bohr", spin=None, verbose=0) if hasattr(self, "basis_name") else None
            if mol is None:
                return None
            n = mol.nao
            errs.append(np.abs(S[i0:i0 + n, i0:i0 + n] - mol.intor("int1e_ovlp")).max())
            i0 += n
        return max(errs)

    def _nonlocal(self):
        cr = self.cr
        dx = cr.dx
        n1, n2, n3 = cr.shape
        blocks = []                                        # (B matrix nb x nproj, h)
        for R, pp in zip(cr.pos, cr.pseudo):
            for l, rl, h in pp["channels"]:
                rc = 9.0 * rl
                rad = int(np.ceil(rc / dx))
                off = np.arange(-rad, rad + 1)
                c = np.round(R / dx).astype(int)
                I, J, K = np.meshgrid(c[0] + off, c[1] + off, c[2] + off, indexing="ij")
                P = np.stack([I, J, K], -1).reshape(-1, 3)
                d = _min_image(P * dx - R, cr.cell)
                r = np.linalg.norm(d, axis=1)
                m = r < rc
                P, d, r = P[m], d[m], r[m]
                idx = np.ravel_multi_index((P[:, 0] % n1, P[:, 1] % n2, P[:, 2] % n3), cr.shape)
                ylm = _real_ylm_poly(l, d)
                for mm in range(2 * l + 1):
                    cols = []
                    for i in range(1, h.shape[0] + 1):
                        p = projector_radial(l, i, rl, r) * ylm[mm]
                        cols.append(sparse.csr_matrix((p, (idx, np.zeros(idx.size, int))), shape=(cr.ng, 1)))
                    Pm = sparse.hstack(cols).tocsr()
                    B = (self.X.T @ Pm).toarray() * cr.dV          # (nb, nproj)
                    blocks.append(B @ h @ B.T)
        Vnl = np.zeros((self.nb, self.nb))
        for b in blocks:
            Vnl += b
        return sparse.csr_matrix(Vnl)
