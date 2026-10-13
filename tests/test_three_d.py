"""3D tests: grid KS, LCAO KS and 3D LRFT on the identical Hamiltonian."""
import warnings

import numpy as np

from lrft.three_d import (KSSolver, h2_crystal, hartree, ion_potential, ks_dense_density, lda_xc)
from lrft.lcao3d import LCAOBasis, LRFTLCAO, KSLCAO, cell_centres, ks_lcao, scf_lcao

warnings.filterwarnings("ignore")
BASIS = dict(rcut=4.5, s_exps=(1.2, 0.4, 0.15), p_exps=(0.5,))


def guess_v(cr):
    d0 = np.full(cr.shape, cr.n_elec / (cr.ng * cr.dV))
    return ion_potential(cr) + hartree(cr, d0) + lda_xc(d0)[1]


def test_lda_potential_is_derivative():
    n = np.logspace(-4, 1, 40)
    e, v = lda_xc(n)
    h = 1e-6 * n
    fd = ((n + h) * lda_xc(n + h)[0] - (n - h) * lda_xc(n - h)[0]) / (2 * h)
    assert np.max(np.abs(fd - v) / np.abs(v)) < 1e-6


def test_ion_potential_is_neutral_and_attractive():
    cr = h2_crystal((1, 1, 1), a=6.0, pts_per_cell=10, rc=0.6)
    v = ion_potential(cr)
    assert abs(v.mean()) < 1e-10                    # G = 0 removed
    i = np.unravel_index(np.argmin(v), v.shape)
    r = np.array(i) * cr.dx
    assert min(np.linalg.norm(r - p) for p in cr.pos) < 1.0   # minimum at a nucleus


def test_lobpcg_ks_matches_dense():
    cr = h2_crystal((1, 1, 1), a=6.0, pts_per_cell=8, rc=0.8)
    v = guess_v(cr)
    nd, ebd, _ = ks_dense_density(cr, v)
    nl, ebl, _ = KSSolver(cr, tol=1e-9, maxiter=500).density(v)
    assert abs(ebl - ebd) < 1e-8
    assert np.sum(np.abs(nl - nd)) * cr.dV < 1e-5


def test_lcao_ks_density_normalised():
    cr = h2_crystal((2, 2, 2), a=6.0, pts_per_cell=10, rc=0.6)
    b = LCAOBasis(cr, **BASIS)
    n, eb, w = ks_lcao(b, guess_v(cr))
    assert abs(n.sum() * cr.dV - cr.n_elec) < 1e-8


def test_lrft_exact_when_window_is_whole_cell():
    """With R_w covering the whole periodic cell LRFT must equal KS (to e^-M)."""
    n = (2, 2, 2)
    cr = h2_crystal(n, a=6.0, pts_per_cell=10, rc=0.6)
    b = LCAOBasis(cr, **BASIS)
    v = guess_v(cr)
    nk, ebk, _ = ks_lcao(b, v)
    nl, ebl, _ = LRFTLCAO(b, cell_centres(n, 6.0), R_w=20.0, M=64).density(v)
    assert abs(ebl - ebk) < 1e-10
    assert np.sum(np.abs(nl - nk)) * cr.dV < 1e-9


def test_lrft_window_error_decays():
    n = (4, 4, 4)
    cr = h2_crystal(n, a=6.0, pts_per_cell=10, rc=0.6)
    b = LCAOBasis(cr, **BASIS)
    v = guess_v(cr)
    nk, ebk, _ = ks_lcao(b, v)
    errs = []
    for Rw in (6.1, 12.1):
        nl, ebl, _ = LRFTLCAO(b, cell_centres(n, 6.0), R_w=Rw, M=64).density(v)
        errs.append(np.sum(np.abs(nl - nk)) * cr.dV / cr.n_elec)
    assert errs[1] < 1e-2 * errs[0] and errs[1] < 1e-5


def test_scf_lrft_matches_ks_3d():
    n = (2, 2, 2)
    cr = h2_crystal(n, a=6.0, pts_per_cell=10, rc=0.6)
    b = LCAOBasis(cr, **BASIS)
    ks = scf_lcao(cr, KSLCAO(b), tol=1e-8)
    lr = scf_lcao(cr, LRFTLCAO(b, cell_centres(n, 6.0), R_w=12.1, M=64), tol=1e-8)
    assert ks["converged"] and lr["converged"]
    assert abs(lr["energy"] - ks["energy"]) / cr.n_elec < 1e-8


def test_cpp_kernels_reproduce_python():
    """The C++ back end must give the same numbers as the reference Python code."""
    from lrft.lcao3d import KSLCAOCpp, LRFTLCAOCpp
    n = (3, 3, 3)
    cr = h2_crystal(n, a=6.0, pts_per_cell=10, rc=0.6)
    b = LCAOBasis(cr, **BASIS)
    v = guess_v(cr)
    nk, ebk, _ = ks_lcao(b, v)
    nkc, ebkc, _ = KSLCAOCpp(b).density(v)
    assert abs(ebkc - ebk) < 1e-9 and np.max(np.abs(nkc - nk)) < 1e-10
    npy, epy, _ = LRFTLCAO(b, cell_centres(n, 6.0), R_w=8.5).density(v)
    C = LRFTLCAOCpp(b, cell_centres(n, 6.0), R_w=8.5)
    for _ in range(2):                                  # cold, then warm path
        ncp, ecp, _ = C.density(v)
        assert abs(ecp - epy) < 1e-9 and np.max(np.abs(ncp - npy)) < 1e-10


def test_cpp_threads_give_identical_density():
    from lrft.lcao3d import LRFTLCAOCpp
    n = (3, 3, 3)
    cr = h2_crystal(n, a=6.0, pts_per_cell=10, rc=0.6)
    b = LCAOBasis(cr, **BASIS)
    v = guess_v(cr)
    n1, e1, _ = LRFTLCAOCpp(b, cell_centres(n, 6.0), R_w=8.5, nthreads=1).density(v)
    n4, e4, _ = LRFTLCAOCpp(b, cell_centres(n, 6.0), R_w=8.5, nthreads=4).density(v)
    assert abs(e1 - e4) < 1e-12 and np.max(np.abs(n1 - n4)) < 1e-12


def test_ks_driver_variants_agree():
    """All LAPACK paths and both density constructions give the same KS density."""
    from lrft.lcao3d import KSLCAOCpp2
    cr = h2_crystal((2, 2, 2), a=6.0, pts_per_cell=10, rc=0.6)
    b = LCAOBasis(cr, **BASIS)
    v = guess_v(cr)
    ref, ebr, _ = ks_lcao(b, v)
    for drv in ("gvx", "gvd", "evr"):
        for dm in (False, True):
            d, eb, _ = KSLCAOCpp2(b, drv, dm).density(v)
            assert np.max(np.abs(d - ref)) < 1e-10 and abs(eb - ebr) < 1e-9


def test_window_spectra_bounded_below_by_global_spectrum():
    """Lemma 4 on actual windows: lowest window eigenvalue >= lowest KS eigenvalue."""
    from scipy import linalg
    n = (3, 3, 3)
    cr = h2_crystal(n, a=6.0, pts_per_cell=10, rc=0.6)
    b = LCAOBasis(cr, **BASIS)
    v = guess_v(cr)
    H, S = b.H(v), b.S
    lam_glob = linalg.eigh(H.toarray(), S.toarray(), eigvals_only=True)[0]
    L = LRFTLCAO(b, cell_centres(n, 6.0), R_w=8.5)
    for W in L.windows:
        lw = linalg.eigh(H[W][:, W].toarray(), S[W][:, W].toarray(), eigvals_only=True)[0]
        assert lw >= lam_glob - 1e-10


def test_stored_window_vectors_reproduce_second_pass():
    """Re-solving mu from stored window vectors equals a full second eigensolver pass."""
    from lrft.lcao3d import LRFTLCAOCpp
    n = (3, 3, 3)
    cr = h2_crystal(n, a=6.0, pts_per_cell=10, rc=0.6)
    b = LCAOBasis(cr, **BASIS)
    v = guess_v(cr)
    out = []
    for reuse in (True, False):
        C = LRFTLCAOCpp(b, cell_centres(n, 6.0), R_w=8.5, reuse_vectors=reuse)
        C.density(v)                       # cold step sets mu
        C.mu += 0.02                       # force a mu re-solve in the warm step
        out.append(C.density(0.999 * v))
    assert abs(out[0][1] - out[1][1]) < 1e-9
    assert np.max(np.abs(out[0][0] - out[1][0])) < 1e-6   # float32 storage of |phi|^2
