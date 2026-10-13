// C++ kernels for 3D LRFT and the Kohn-Sham reference (LCAO-on-grid).
//
// Both one-body solvers are implemented here with the SAME LAPACK (system
// OpenBLAS) so that the comparison is between algorithms, not languages.
//
//  lrft_windows : for every nearsighted window W(B)
//                   - gather dense H_W, S_W from the global CSR matrices
//                   - generalised eigenproblem (dsygvd full, or dsygvx subset
//                     below `upper` when mu is known: r_M < 1.5^-M above it)
//                   - Y = X[own(B), W] U   (window eigenstates on owned points)
//                   - c_k = int_own |phi_k|^2, density on owned points with r_M
//                 windows are independent -> OpenMP over windows.
//  ks_density   : dense generalised eigenproblem for the lowest nocc(+1)
//                 states (dsygvx), then n(r) = 2 sum_i |sum_mu X_r,mu C_mu,i|^2.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace py = pybind11;
using i64 = std::int64_t;
using darr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using iarr = py::array_t<i64, py::array::c_style | py::array::forcecast>;

extern "C" {
void dsygvx_(const int* itype, const char* jobz, const char* range, const char* uplo,
             const int* n, double* a, const int* lda, double* b, const int* ldb,
             const double* vl, const double* vu, const int* il, const int* iu,
             const double* abstol, int* m, double* w, double* z, const int* ldz,
             double* work, const int* lwork, int* iwork, int* ifail, int* info,
             size_t, size_t, size_t);
void dsygvd_(const int* itype, const char* jobz, const char* uplo, const int* n, double* a,
             const int* lda, double* b, const int* ldb, double* w, double* work,
             const int* lwork, int* iwork, const int* liwork, int* info, size_t, size_t);
void openblas_set_num_threads(int);
void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info, size_t);
void dsygst_(const int* itype, const char* uplo, const int* n, double* a, const int* lda,
             const double* b, const int* ldb, int* info, size_t);
void dsyevr_(const char* jobz, const char* range, const char* uplo, const int* n, double* a,
             const int* lda, const double* vl, const double* vu, const int* il, const int* iu,
             const double* abstol, int* m, double* w, double* z, const int* ldz, int* isuppz,
             double* work, const int* lwork, int* iwork, const int* liwork, int* info,
             size_t, size_t, size_t);
void dtrsm_(const char* side, const char* uplo, const char* transa, const char* diag, const int* m,
            const int* n, const double* alpha, const double* a, const int* lda, double* b,
            const int* ldb, size_t, size_t, size_t, size_t);
void dsyrk_(const char* uplo, const char* trans, const int* n, const int* k, const double* alpha,
            const double* a, const int* lda, const double* beta, double* c, const int* ldc,
            size_t, size_t);
}

struct CSR {
    const i64* ptr;
    const i64* idx;
    const double* val;
    i64 nrow;
};

static CSR csr(const iarr& p, const iarr& i, const darr& v) {
    return CSR{p.data(), i.data(), v.data(), (i64)p.size() - 1};
}

static inline double r_M(double lam, double mu, double e_lo, int M) {
    const double c = 0.5 * (mu + e_lo), R = 0.5 * (mu - e_lo);
    const double u = std::fabs((lam - c) / R);
    const double p = std::pow(u, M);
    return std::isfinite(p) ? 1.0 / (1.0 + p) : 0.0;
}

// Generalised symmetric eigenproblem A z = lam B z (column-major, n x n).
// upper = +inf -> all pairs (dsygvd); else pairs with lam <= upper (dsygvx).
// Returns number of pairs; eigenvectors written to Z (n x m, column-major).
static int geneig(int n, std::vector<double>& A, std::vector<double>& B, double upper,
                  std::vector<double>& lam, std::vector<double>& Z) {
    int itype = 1, info = 0, m = 0;
    lam.resize(n);
    if (!std::isfinite(upper)) {
        int lwork = -1, liwork = -1, iwq = 0;
        double wq = 0;
        dsygvd_(&itype, "V", "U", &n, A.data(), &n, B.data(), &n, lam.data(), &wq, &lwork, &iwq,
                &liwork, &info, 1, 1);
        lwork = (int)wq;
        liwork = iwq;
        std::vector<double> work(lwork);
        std::vector<int> iwork(liwork);
        dsygvd_(&itype, "V", "U", &n, A.data(), &n, B.data(), &n, lam.data(), work.data(), &lwork,
                iwork.data(), &liwork, &info, 1, 1);
        if (info) throw std::runtime_error("dsygvd failed");
        Z.swap(A);
        m = n;
    } else {
        double vl = -1e300, vu = upper, abstol = 0.0;
        int il = 0, iu = 0, lwork = -1;
        double wq = 0;
        Z.assign((size_t)n * n, 0.0);
        std::vector<int> iwork(5 * n), ifail(n);
        dsygvx_(&itype, "V", "V", "U", &n, A.data(), &n, B.data(), &n, &vl, &vu, &il, &iu, &abstol,
                &m, lam.data(), Z.data(), &n, &wq, &lwork, iwork.data(), ifail.data(), &info, 1, 1, 1);
        lwork = (int)wq;
        std::vector<double> work(lwork);
        dsygvx_(&itype, "V", "V", "U", &n, A.data(), &n, B.data(), &n, &vl, &vu, &il, &iu, &abstol,
                &m, lam.data(), Z.data(), &n, work.data(), &lwork, iwork.data(), ifail.data(), &info,
                1, 1, 1);
        if (info < 0) throw std::runtime_error("dsygvx failed");
    }
    lam.resize(m);
    return m;
}

// Gather dense (column-major) submatrix M[W, W] of a CSR matrix; pos maps global->local.
static void gather(const CSR& Mx, const i64* W, int n, const std::vector<int>& pos,
                   std::vector<double>& out) {
    out.assign((size_t)n * n, 0.0);
    for (int a = 0; a < n; ++a) {
        const i64 row = W[a];
        for (i64 p = Mx.ptr[row]; p < Mx.ptr[row + 1]; ++p) {
            const int b = pos[Mx.idx[p]];
            if (b >= 0) out[(size_t)b * n + a] = Mx.val[p];
        }
    }
}

py::tuple lrft_windows(iarr Hp, iarr Hi, darr Hv, iarr Sp, iarr Si, darr Sv, iarr Xp, iarr Xi,
                       darr Xv, iarr Wflat, iarr Woff, iarr Oflat, iarr Ooff, i64 nb, double dV,
                       double mu, double e_lo, int M, double upper, bool want_density,
                       int nthreads) {
    const CSR H = csr(Hp, Hi, Hv), S = csr(Sp, Si, Sv), X = csr(Xp, Xi, Xv);
    const i64 nwin = Woff.size() - 1;
    const i64 ng = X.nrow;
    const i64* Wf = Wflat.data();
    const i64* Wo = Woff.data();
    const i64* Of = Oflat.data();
    const i64* Oo = Ooff.data();
    std::vector<double> dens(want_density ? ng : 0, 0.0);
    std::vector<std::vector<double>> lam_w(nwin), c_w(nwin);
    std::vector<std::vector<float>> y2_w(want_density ? nwin : 0);
    // OpenMP over windows; when there are fewer windows than threads (e.g. one molecule), the
    // spare threads go to the threaded BLAS/LAPACK inside each window problem.
    const int nomp = (int)std::max<i64>(1, std::min<i64>(nthreads, nwin));
    openblas_set_num_threads(std::max(1, nthreads / nomp));
    omp_set_num_threads(nomp);
#pragma omp parallel
    {
        std::vector<int> pos(nb, -1);
        std::vector<double> A, B, Z, Zt, lam, Y;
#pragma omp for schedule(dynamic)
        for (i64 w = 0; w < nwin; ++w) {
            const i64* W = Wf + Wo[w];
            const int n = (int)(Wo[w + 1] - Wo[w]);
            for (int a = 0; a < n; ++a) pos[W[a]] = a;
            gather(H, W, n, pos, A);
            gather(S, W, n, pos, B);
            const int m = geneig(n, A, B, upper, lam, Z);
            // row-major copy Zt (n x m) for contiguous inner loops
            Zt.resize((size_t)n * m);
            for (int k = 0; k < m; ++k)
                for (int a = 0; a < n; ++a) Zt[(size_t)a * m + k] = Z[(size_t)k * n + a];
            // Y (nown x m, row-major) = X[own, W] Z
            const i64* own = Of + Oo[w];
            const i64 nown = Oo[w + 1] - Oo[w];
            Y.assign((size_t)nown * m, 0.0);
            for (i64 r = 0; r < nown; ++r) {
                const i64 g = own[r];
                double* y = &Y[(size_t)r * m];
                for (i64 p = X.ptr[g]; p < X.ptr[g + 1]; ++p) {
                    const int a = pos[X.idx[p]];
                    if (a < 0) continue;
                    const double xv = X.val[p];
                    const double* z = &Zt[(size_t)a * m];
                    for (int k = 0; k < m; ++k) y[k] += xv * z[k];
                }
            }
            std::vector<double> c(m, 0.0), occ(m, 0.0);
            for (i64 r = 0; r < nown; ++r)
                for (int k = 0; k < m; ++k) c[k] += Y[(size_t)r * m + k] * Y[(size_t)r * m + k];
            for (int k = 0; k < m; ++k) c[k] *= dV;
            if (want_density) {
                y2_w[w].resize((size_t)nown * m);
                for (size_t q = 0; q < (size_t)nown * m; ++q) y2_w[w][q] = (float)(Y[q] * Y[q]);
                for (int k = 0; k < m; ++k) occ[k] = r_M(lam[k], mu, e_lo, M);
                for (i64 r = 0; r < nown; ++r) {
                    double s = 0.0;
                    for (int k = 0; k < m; ++k) s += occ[k] * Y[(size_t)r * m + k] * Y[(size_t)r * m + k];
                    dens[own[r]] = s;   // owned sets are disjoint -> no race
                }
            }
            lam_w[w] = lam;
            c_w[w] = c;
            for (int a = 0; a < n; ++a) pos[W[a]] = -1;
        }
    }
    i64 tot = 0;
    for (auto& v : lam_w) tot += v.size();
    py::array_t<double> lam_out(tot), c_out(tot);
    py::array_t<i64> off_out(nwin + 1);
    auto lo = lam_out.mutable_data();
    auto co = c_out.mutable_data();
    auto oo = off_out.mutable_data();
    i64 k = 0;
    oo[0] = 0;
    for (i64 w = 0; w < nwin; ++w) {
        std::copy(lam_w[w].begin(), lam_w[w].end(), lo + k);
        std::copy(c_w[w].begin(), c_w[w].end(), co + k);
        k += lam_w[w].size();
        oo[w + 1] = k;
    }
    py::array_t<double> d_out(want_density ? ng : 0);
    if (want_density) std::copy(dens.begin(), dens.end(), d_out.mutable_data());
    // squared window eigenvectors on owned points (row-major nown x m per window), so that the
    // density for a different mu can be formed without repeating the eigensolves
    i64 ytot = 0;
    for (auto& v : y2_w) ytot += v.size();
    py::array_t<float> y_out(ytot);
    py::array_t<i64> yoff(want_density ? nwin + 1 : 1);
    auto yo = y_out.mutable_data();
    auto yf = yoff.mutable_data();
    yf[0] = 0;
    if (want_density) {
        i64 q = 0;
        for (i64 w = 0; w < nwin; ++w) {
            std::copy(y2_w[w].begin(), y2_w[w].end(), yo + q);
            q += y2_w[w].size();
            yf[w + 1] = q;
        }
    }
    return py::make_tuple(d_out, lam_out, c_out, off_out, y_out, yoff);
}

py::tuple ks_density(iarr Hp, iarr Hi, darr Hv, iarr Sp, iarr Si, darr Sv, iarr Xp, iarr Xi, darr Xv,
                     i64 nb, int nocc, int nthreads) {
    const CSR H = csr(Hp, Hi, Hv), S = csr(Sp, Si, Sv), X = csr(Xp, Xi, Xv);
    const int n = (int)nb;
    std::vector<double> A((size_t)n * n, 0.0), B((size_t)n * n, 0.0);
    for (int a = 0; a < n; ++a) {
        for (i64 p = H.ptr[a]; p < H.ptr[a + 1]; ++p) A[(size_t)H.idx[p] * n + a] = H.val[p];
        for (i64 p = S.ptr[a]; p < S.ptr[a + 1]; ++p) B[(size_t)S.idx[p] * n + a] = S.val[p];
    }
    openblas_set_num_threads(nthreads);
    int itype = 1, info = 0, m = 0, il = 1, iu = nocc + 1, lwork = -1;
    double vl = 0, vu = 0, abstol = 0, wq = 0;
    std::vector<double> lam(n), Z((size_t)n * (nocc + 1));
    std::vector<int> iwork(5 * n), ifail(n);
    dsygvx_(&itype, "V", "I", "U", &n, A.data(), &n, B.data(), &n, &vl, &vu, &il, &iu, &abstol, &m,
            lam.data(), Z.data(), &n, &wq, &lwork, iwork.data(), ifail.data(), &info, 1, 1, 1);
    lwork = (int)wq;
    std::vector<double> work(lwork);
    dsygvx_(&itype, "V", "I", "U", &n, A.data(), &n, B.data(), &n, &vl, &vu, &il, &iu, &abstol, &m,
            lam.data(), Z.data(), &n, work.data(), &lwork, iwork.data(), ifail.data(), &info, 1, 1, 1);
    if (info < 0) throw std::runtime_error("dsygvx failed");
    // density n(r) = 2 sum_{i<nocc} |sum_mu X[r,mu] C[mu,i]|^2  (row-parallel)
    std::vector<double> Ct((size_t)n * nocc);           // row-major C (n x nocc)
    for (int i = 0; i < nocc; ++i)
        for (int a = 0; a < n; ++a) Ct[(size_t)a * nocc + i] = Z[(size_t)i * n + a];
    const i64 ng = X.nrow;
    py::array_t<double> d_out(ng);
    double* d = d_out.mutable_data();
    omp_set_num_threads(nthreads);
#pragma omp parallel
    {
        std::vector<double> t(nocc);
#pragma omp for schedule(static)
        for (i64 r = 0; r < ng; ++r) {
            std::fill(t.begin(), t.end(), 0.0);
            for (i64 p = X.ptr[r]; p < X.ptr[r + 1]; ++p) {
                const double xv = X.val[p];
                const i64 mu = X.idx[p];
                const double* cr = &Ct[(size_t)mu * nocc];
                for (int i = 0; i < nocc; ++i) t[i] += xv * cr[i];
            }
            double s = 0.0;
            for (int i = 0; i < nocc; ++i) s += t[i] * t[i];
            d[r] = 2.0 * s;
        }
    }
    py::array_t<double> w_out(m);
    std::copy(lam.begin(), lam.begin() + m, w_out.mutable_data());
    return py::make_tuple(d_out, w_out);
}


// KS density with a choice of eigensolver and of density construction, so that
// the reference is not handicapped by one particular LAPACK path.
//   driver: 0 = dsygvx (bisection + inverse iteration), 1 = dsygvd (full, divide and
//           conquer), 2 = Cholesky + dsygst + dsyevr (MRRR, lowest nocc+1 pairs)
//   dm:     false = orbital sum on the grid, O(N_g N_occ nnz_row) ~ O(N^2)
//           true  = density matrix P = 2 C C^T (dsyrk) projected with the sparse basis,
//                   O(N_b^2 N_occ) BLAS3 + O(N_g nnz_row^2)
// returns (density, eigenvalues, t_eig, t_density)
py::tuple ks_density2(iarr Hp, iarr Hi, darr Hv, iarr Sp, iarr Si, darr Sv, iarr Xp, iarr Xi,
                      darr Xv, i64 nb, int nocc, int driver, bool dm, int nthreads) {
    const CSR H = csr(Hp, Hi, Hv), S = csr(Sp, Si, Sv), X = csr(Xp, Xi, Xv);
    const int n = (int)nb;
    std::vector<double> A((size_t)n * n, 0.0), B((size_t)n * n, 0.0);
    for (int a = 0; a < n; ++a) {
        for (i64 p = H.ptr[a]; p < H.ptr[a + 1]; ++p) A[(size_t)H.idx[p] * n + a] = H.val[p];
        for (i64 p = S.ptr[a]; p < S.ptr[a + 1]; ++p) B[(size_t)S.idx[p] * n + a] = S.val[p];
    }
    openblas_set_num_threads(nthreads);
    const double t0 = omp_get_wtime();
    int itype = 1, info = 0, m = 0, il = 1, iu = nocc + 1, lwork = -1;
    double vl = 0, vu = 0, abstol = 0, wq = 0;
    std::vector<double> lam(n), Z;
    if (driver == 0) {
        Z.assign((size_t)n * (nocc + 1), 0.0);
        std::vector<int> iwork(5 * n), ifail(n);
        dsygvx_(&itype, "V", "I", "U", &n, A.data(), &n, B.data(), &n, &vl, &vu, &il, &iu, &abstol,
                &m, lam.data(), Z.data(), &n, &wq, &lwork, iwork.data(), ifail.data(), &info, 1, 1, 1);
        lwork = (int)wq;
        std::vector<double> work(lwork);
        dsygvx_(&itype, "V", "I", "U", &n, A.data(), &n, B.data(), &n, &vl, &vu, &il, &iu, &abstol,
                &m, lam.data(), Z.data(), &n, work.data(), &lwork, iwork.data(), ifail.data(), &info,
                1, 1, 1);
    } else if (driver == 1) {
        int liwork = -1, iwq = 0;
        dsygvd_(&itype, "V", "U", &n, A.data(), &n, B.data(), &n, lam.data(), &wq, &lwork, &iwq,
                &liwork, &info, 1, 1);
        lwork = (int)wq;
        liwork = iwq;
        std::vector<double> work(lwork);
        std::vector<int> iwork(liwork);
        dsygvd_(&itype, "V", "U", &n, A.data(), &n, B.data(), &n, lam.data(), work.data(), &lwork,
                iwork.data(), &liwork, &info, 1, 1);
        Z.assign(A.begin(), A.begin() + (size_t)n * (nocc + 1));
        m = nocc + 1;
    } else {
        dpotrf_("U", &n, B.data(), &n, &info, 1);
        if (info) throw std::runtime_error("dpotrf failed");
        dsygst_(&itype, "U", &n, A.data(), &n, B.data(), &n, &info, 1);
        Z.assign((size_t)n * (nocc + 1), 0.0);
        std::vector<int> isuppz(2 * (nocc + 1));
        int liwork = -1, iwq = 0;
        dsyevr_("V", "I", "U", &n, A.data(), &n, &vl, &vu, &il, &iu, &abstol, &m, lam.data(),
                Z.data(), &n, isuppz.data(), &wq, &lwork, &iwq, &liwork, &info, 1, 1, 1);
        lwork = (int)wq;
        liwork = iwq;
        std::vector<double> work(lwork);
        std::vector<int> iwork(liwork);
        dsyevr_("V", "I", "U", &n, A.data(), &n, &vl, &vu, &il, &iu, &abstol, &m, lam.data(),
                Z.data(), &n, isuppz.data(), work.data(), &lwork, iwork.data(), &liwork, &info,
                1, 1, 1);
        const double one = 1.0;
        dtrsm_("L", "U", "N", "N", &n, &m, &one, B.data(), &n, Z.data(), &n, 1, 1, 1, 1);
    }
    if (info < 0) throw std::runtime_error("eigensolver failed");
    const double t1 = omp_get_wtime();
    const i64 ng = X.nrow;
    py::array_t<double> d_out(ng);
    double* d = d_out.mutable_data();
    omp_set_num_threads(nthreads);
    if (!dm) {
        std::vector<double> Ct((size_t)n * nocc);
        for (int i = 0; i < nocc; ++i)
            for (int a = 0; a < n; ++a) Ct[(size_t)a * nocc + i] = Z[(size_t)i * n + a];
#pragma omp parallel
        {
            std::vector<double> t(nocc);
#pragma omp for schedule(static)
            for (i64 r = 0; r < ng; ++r) {
                std::fill(t.begin(), t.end(), 0.0);
                for (i64 p = X.ptr[r]; p < X.ptr[r + 1]; ++p) {
                    const double* cr = &Ct[(size_t)X.idx[p] * nocc];
                    for (int i = 0; i < nocc; ++i) t[i] += X.val[p] * cr[i];
                }
                double s = 0.0;
                for (int i = 0; i < nocc; ++i) s += t[i] * t[i];
                d[r] = 2.0 * s;
            }
        }
    } else {
        std::vector<double> P((size_t)n * n, 0.0);   // column-major, upper triangle
        const double two = 2.0, zero = 0.0;
        dsyrk_("U", "N", &n, &nocc, &two, Z.data(), &n, &zero, P.data(), &n, 1, 1);
#pragma omp parallel for schedule(static)
        for (i64 r = 0; r < ng; ++r) {
            double s = 0.0;
            for (i64 p = X.ptr[r]; p < X.ptr[r + 1]; ++p) {
                const i64 a = X.idx[p];
                for (i64 q = X.ptr[r]; q < X.ptr[r + 1]; ++q) {
                    const i64 b = X.idx[q];
                    const double pab = a <= b ? P[(size_t)b * n + a] : P[(size_t)a * n + b];
                    s += X.val[p] * X.val[q] * pab;
                }
            }
            d[r] = s;
        }
    }
    const double t2 = omp_get_wtime();
    py::array_t<double> w_out(m);
    std::copy(lam.begin(), lam.begin() + m, w_out.mutable_data());
    return py::make_tuple(d_out, w_out, t1 - t0, t2 - t1);
}

PYBIND11_MODULE(_lrft_cpp, mod) {
    mod.doc() = "C++ kernels for 3D LRFT and the KS reference";
    mod.def("lrft_windows", &lrft_windows);
    mod.def("ks_density", &ks_density);
    mod.def("ks_density2", &ks_density2);
}
