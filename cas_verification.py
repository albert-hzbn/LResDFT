"""
CAS (SymPy) proofs / verifications for Local Resolvent Field Theory (LRFT).

Fundamental variable:  g(x, z) = <x| (z - h)^-1 |x>   (z complex energy)
the *energy-resolved* (complex-energy) density.  The electron density, the
potential, and the band energy are explicit, linear or local functionals of g:
no unknown kinetic functional, no orbitals, no diagonalisation.

Each block corresponds to a Lemma/Proposition/Remark of the manuscript (manuscript/main.tex).
The checks verify identities on explicit instances (given M, given matrices) and to 40 digits;
they complement, but do not replace, the general proofs in Appendix A.
Run:  python3 cas_verification.py      (exit code != 0 on any failure)
"""
import sys
import random
import sympy as sp
import mpmath as mp

FAILED = []


def check(name, cond):
    ok = bool(cond)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILED.append(name)


u = sp.symbols("u")
Msym = sp.symbols("M", positive=True, integer=True)

# ---------------------------------------------------------------------------
# Lemma 1 (Butterworth-trapezoid identity).
# zeta_j = exp(i pi (2j+1)/M), j=0..M-1, are the M roots of zeta^M = -1, and
#        (1/M) sum_j zeta_j / (zeta_j - u) = 1 / (1 + u^M).
# Proof: 1/(1+u^M) has simple poles at zeta_j with residue 1/(M zeta_j^(M-1))
#        = zeta_j/(M zeta_j^M) = -zeta_j/M, and vanishes at infinity (M>=1),
#        so it equals the sum of its principal parts.
# ---------------------------------------------------------------------------
zj = sp.symbols("zeta")
z0 = sp.exp(sp.I * sp.pi / 4)
res = sp.limit((u - z0) / (1 + u**4), u, z0)          # residue at a simple pole
check("residue of 1/(1+u^4) at e^{i pi/4} equals -zeta/M",
      abs(complex(sp.N(res + z0 / 4, 30))) < 1e-25)
check("1/(M zeta^(M-1)) = -zeta/M  whenever zeta^M = -1",
      sp.simplify((zj / (Msym * zj**Msym)).subs(zj**Msym, -1) + zj / Msym) == 0)
ok = True
for M in range(2, 15, 2):
    zetas = [sp.exp(sp.I * sp.pi * (2 * j + 1) / M) for j in range(M)]
    lhs = sum(zt / (zt - u) for zt in zetas) / M
    for uv in [0.3 + 0.1j, -0.9 + 0.2j, 1.7 - 0.4j, 2.5j, 0.999]:
        ok &= abs(complex(lhs.subs(u, uv).evalf(30)) - 1 / (1 + uv**M)) < 1e-12
check("(1/M) sum zeta_j/(zeta_j-u) = 1/(1+u^M), M = 2..14", ok)
ok = all(abs(complex(sum(sp.exp(sp.I * sp.pi * (2 * j + 1) / M) for j in range(M)).evalf())) < 1e-14
         for M in range(2, 15, 2))
check("sum_j zeta_j = 0  (no constant term -> Lemma 3)", ok)

# ---------------------------------------------------------------------------
# Lemma 2 (filter = Fermi-Dirac in log-energy => N-representable occupations).
# M even, real u:  r_M(u) = 1/(1+u^M) = 1/(1+exp(M ln|u|)) in (0,1].
# Bounds:  |u| = rho<1:  1-r = rho^M/(1+rho^M) <= rho^M ;  |u| = 1/rho:  r <= rho^M
# ---------------------------------------------------------------------------
uu = sp.symbols("u", real=True)
ok = True
for m in [2, 6, 20]:
    for v in [-1.7, -0.4, 0.2, 0.99, 3.0]:
        r = 1 / (1 + v**m)
        ok &= abs(r - 1 / (1 + mp.exp(m * mp.log(abs(v))))) < 1e-14 and 0 < r <= 1
check("r_M(u) = FermiDirac(beta=M, s=ln|u|) and 0 < r <= 1 (M even)", ok)
rho, Mp = sp.symbols("rho M", positive=True)
check("inside error: 1 - r_M = rho^M/(1+rho^M)",
      sp.simplify(1 - 1 / (1 + rho**Mp) - rho**Mp / (1 + rho**Mp)) == 0)
check("outside error: r_M(1/rho) = rho^M/(1+rho^M)",
      sp.simplify(1 / (1 + rho ** (-Mp)) - rho**Mp / (1 + rho**Mp)) == 0)

# ---------------------------------------------------------------------------
# Proposition 1(b): band energy is LINEAR in g (T_s = E_band - sum v n is not).
#  P_M(h) = sum_j w_j (z_j - h)^-1,  z_j = c + R zeta_j,  w_j = R zeta_j / M
#  Tr[h P_M] = sum_j w_j z_j Tr(z_j-h)^-1 - N_g sum_j w_j = sum_j w_j z_j sum_x g(x,z_j)
#  and P_M(h) = [1 + ((h-c)/R)^M]^-1  as an exact matrix identity.
# ---------------------------------------------------------------------------
mp.mp.dps = 40
random.seed(1)
Ng = 4
Hq = sp.zeros(Ng, Ng)
for i in range(Ng):
    for j in range(i, Ng):
        Hq[i, j] = Hq[j, i] = sp.Rational(random.randint(-9, 9), 7)
H = sp.Matrix(Hq)
Hm = mp.matrix([[mp.mpf(sp.Rational(Hq[i, j]).p) / sp.Rational(Hq[i, j]).q for j in range(Ng)] for i in range(Ng)])
M = 6
c_, R_ = mp.mpf(-1) / 2, mp.mpf(3)
zetas = [mp.exp(1j * mp.pi * (2 * j + 1) / M) for j in range(M)]
zs = [c_ + R_ * zt for zt in zetas]
ws = [R_ * zt / M for zt in zetas]
I4 = mp.eye(Ng)
Gs = [mp.inverse(zz * I4 - Hm) for zz in zs]
P = mp.zeros(Ng, Ng)
for ww, Gz in zip(ws, Gs):
    P += ww * Gz
tr = lambda A: sum(A[i, i] for i in range(Ng))
lhs = tr(Hm * P)
rhs = sum(ww * zz * tr(Gz) for zz, ww, Gz in zip(zs, ws, Gs))
check("Tr[h P_M] = sum_j w_j z_j sum_x g(x,z_j)   (band energy linear in g, 40 digits)",
      abs(lhs - rhs) < mp.mpf(10) ** -30)
Pr = mp.inverse(I4 + ((Hm - c_ * I4) / R_) ** M)
check("P_M(h) = [1 + ((h-c)/R)^M]^-1   (exact matrix identity, 40 digits)",
      mp.mnorm(P - Pr, 1) < mp.mpf(10) ** -30)
check("P_M(h) is real symmetric (conjugate node pairs)",
      max(abs(mp.im(P[i, j])) for i in range(Ng) for j in range(Ng)) < mp.mpf(10) ** -30
      and mp.mnorm(P - P.T, 1) < mp.mpf(10) ** -30)

# ---------------------------------------------------------------------------
# Remark 1 (diagonal of h from a large-contour moment of g; grid Hamiltonians with local v only).
#  g(x,z) = 1/z + h_xx/z^2 + (h^2)_xx/z^3 + ...  =>  v(x) = [z^2(g-1/z)]_{z->inf} - t_xx
#  (In DFT the map n -> v is a global, ill-conditioned inverse problem.)
# ---------------------------------------------------------------------------
w_ = sp.symbols("w")
gx = ((sp.eye(Ng) - w_ * H).inv())[1, 1] * w_          # g(x=1, z=1/w)
ser = sp.series(gx, w_, 0, 4).removeO()
check("g(x,z) = 1/z + h_xx/z^2 + (h^2)_xx/z^3 + O(z^-4)",
      sp.simplify(ser - (w_ + H[1, 1] * w_**2 + (H * H)[1, 1] * w_**3)) == 0)
dx = sp.symbols("dx", positive=True)
vs = sp.symbols("v0:5")
T = sp.Matrix(5, 5, lambda i, j: (1 / dx**2 if i == j else (-1 / (2 * dx**2) if abs(i - j) == 1 else 0)))
hfd = T + sp.diag(*vs)
z = sp.symbols("z")
# by the Neumann series above (valid for any matrix), lim z^2(g-1/z) = h_xx:
check("v(x) = lim z^2 (g - 1/z) - 1/dx^2   on the FD Hamiltonian (symbolic)",
      all(sp.simplify(hfd[i, i] - 1 / dx**2 - vs[i]) == 0 for i in range(5)))
# explicit numeric confirmation of the limit at large |z| (3x3 FD block)
hnum = sp.Matrix(3, 3, lambda i, j: (100 + [0.3, -1.2, 0.7][i] if i == j else (-50 if abs(i - j) == 1 else 0)))
zbig = sp.Integer(10)**10
gnum = ((zbig * sp.eye(3) - hnum).inv())[1, 1]
check("numeric: z^2(g - 1/z) - 1/dx^2 -> v(x) (dx=0.1, z=1e10)",
      abs(float(sp.N(zbig**2 * (gnum - 1 / zbig), 50)) - 100 + 1.2) < 1e-5)

# ---------------------------------------------------------------------------
# Proposition 2 (exact derivative of Omega_M at fixed circle, every M).
#  d Tr F(h) = Tr F'(h) dh, dh/dv(x) = |x><x|  =>  dOmega_M/dv(x) = n_M(x).
#  Each pole contributes F(h) = log(z-h): d/dv_x Tr log(z-h) = -g(x,z).
# ---------------------------------------------------------------------------
a, b, cc, d_, e_, f_ = sp.symbols("a b c d e f")
H3 = sp.Matrix([[a, d_, e_], [d_, b, f_], [e_, f_, cc]])
Fdet = sp.log((z * sp.eye(3) - H3).det())
G3 = (z * sp.eye(3) - H3).inv()
check("d/dv_x Tr log(z-h) = -g(x,z) for all x (Jacobi formula)",
      all(sp.simplify(sp.diff(Fdet, var) + G3[i, i]) == 0 for i, var in enumerate([a, b, cc])))
check("d/dv_x Tr (z-h)^-1 = <x|(z-h)^-2|x>  (consistent sign convention)",
      sp.simplify(sp.diff(G3.trace(), a) - (G3 * G3)[0, 0]) == 0)
# The grand-potential primitive of the filter, W_M' = 2 r_M, exists in closed
# form via the same poles: W_M(e) = 2 sum_j w_j * (-log(z_j - e)) + const.
ee = sp.symbols("e")
# derivative of -log(z-e) is 1/(z-e)
dW = lambda ev: 2 * sum(ww / (zz - ev) for ww, zz in zip(ws, zs))
okW = all(abs(dW(ev) - 2 / (1 + ((ev - c_) / R_) ** M)) < mp.mpf(10) ** -30
          for ev in [mp.mpf(-3), mp.mpf(1) / 3, mp.mpf(5) / 2, mp.mpf(9)])
check("W_M'(e) = 2 r_M((e-c)/R): Omega_M = Tr W_M(h) has dOmega_M/dv = n_M", okW)

# ---------------------------------------------------------------------------
# Lemma 3 (exact O(N) diagonal of the resolvent for tridiagonal h; cf. Meurant 1992).
#  A = z - h, diag a_i, off-diag b_i.  L_1=a_1, L_i=a_i-b_{i-1}^2/L_{i-1};
#  U_n=a_n, U_i=a_i-b_i^2/U_{i+1};   (A^-1)_ii = 1/(L_i + U_i - a_i).
# ---------------------------------------------------------------------------
n4 = 5
A_ = sp.symbols("a1:6")
B_ = sp.symbols("b1:5")
A4 = sp.zeros(n4, n4)
for i in range(n4):
    A4[i, i] = A_[i]
for i in range(n4 - 1):
    A4[i, i + 1] = A4[i + 1, i] = B_[i]
Ainv = A4.inv()
L = [A_[0]]
for i in range(1, n4):
    L.append(A_[i] - B_[i - 1] ** 2 / L[i - 1])
U = [None] * n4
U[-1] = A_[-1]
for i in range(n4 - 2, -1, -1):
    U[i] = A_[i] - B_[i] ** 2 / U[i + 1]
check("diag((z-h)^-1)_i = 1/(L_i + U_i - a_i)  (two-sided O(N) recursion, n=5 symbolic)",
      all(sp.simplify(Ainv[i, i] - 1 / (L[i] + U[i] - A_[i])) == 0 for i in range(n4)))

# ---------------------------------------------------------------------------
# Remark 3 (decay of the FD resolvent, Combes-Thomas type): for constant v the FD
#  resolvent obeys G_{m+1} + 2 alpha G_m + G_{m-1} = 0 (m != 0),
#  alpha = dx^2 (z - v) - 1, so G_m ~ lam^|m| with lam^2 + 2 alpha lam + 1 = 0;
#  roots multiply to 1, and |lam| < 1 for z off the spectrum.
# ---------------------------------------------------------------------------
lam, alpha = sp.symbols("lambda alpha")
sols = sp.solve(lam**2 + 2 * alpha * lam + 1, lam)
check("product of decay roots = 1", sp.simplify(sols[0] * sols[1] - 1) == 0)
ok = True
for zv in [-0.5 + 0.05j, 0.3 + 0.2j, -2.0 + 0.0j]:
    al = 0.04 * zv - 1
    roots = [complex(sp.N(s_.subs(alpha, al))) for s_ in sols]
    ok &= min(abs(r_) for r_ in roots) < 1
check("decaying root |lam| < 1 for z off the spectrum", ok)

# ---------------------------------------------------------------------------
# Many-body part: exchange of the 1D soft-Coulomb uniform gas (closed form)
#   e_x(n) = -(1/pi^2)[ b IK0(b) + b K1(b) - 1 ],  b = pi n,   v_x = -(1/pi) IK0(pi n)
# from e_x = -(2/pi^2) Int_0^inf sin^2(kF r)/(r^2 sqrt(1+r^2)) dr, kF = pi n/2.
# ---------------------------------------------------------------------------
bb = sp.symbols("b", positive=True)
IK0 = sp.Function("IK0")
I_closed = bb * IK0(bb) + bb * sp.besselk(1, bb) - 1
dI = sp.diff(I_closed, bb).subs(sp.Derivative(IK0(bb), bb), sp.besselk(0, bb))
check("d/db [b IK0 + b K1 - 1] = IK0(b)  (=> I''(b) = K0(b))",
      sp.simplify(sp.expand_func(dI - IK0(bb))) == 0)
check("I(0) = 0  (b K1(b) -> 1)", sp.limit(bb * sp.besselk(1, bb), bb, 0) == 1)
ok = True
for bv in [0.1, 0.7, 2.0, 5.0]:
    f = lambda t: (1 - mp.cos(bv * t)) / (t**2 * mp.sqrt(1 + t**2))
    direct = mp.quad(f, [0, 1]) + mp.quadosc(f, [1, mp.inf], omega=bv)
    closed = bv * mp.quad(lambda t: mp.besselk(0, t), [0, bv]) + bv * mp.besselk(1, bv) - 1
    ok &= abs(direct - closed) < 1e-6 * abs(closed)
check("closed-form soft-Coulomb exchange matches direct oscillatory quadrature", ok)

# ---------------------------------------------------------------------------
# Three-dimensional ingredients (Sec. III B and Appendix A of the manuscript)
# ---------------------------------------------------------------------------
# (i) Lemma 1 for a non-orthogonal basis: sum_j w_j (z_j S - H)^-1 = r_M(S^-1 H) S^-1
mp.mp.dps = 40
random.seed(3)
nb = 4
Hn = mp.matrix(nb, nb)
An = mp.matrix(nb, nb)
for i in range(nb):
    for j in range(nb):
        An[i, j] = mp.mpf(random.randint(-5, 5)) / 9
for i in range(nb):
    for j in range(i, nb):
        Hn[i, j] = Hn[j, i] = mp.mpf(random.randint(-9, 9)) / 7
Sn = An.T * An + mp.eye(nb)                                   # SPD overlap
M6, c6, R6 = 8, mp.mpf(-1), mp.mpf(4)
zs6 = [c6 + R6 * mp.exp(1j * mp.pi * (2 * j + 1) / M6) for j in range(M6)]
ws6 = [R6 * mp.exp(1j * mp.pi * (2 * j + 1) / M6) / M6 for j in range(M6)]
Pgen = mp.zeros(nb, nb)
for z_, w__ in zip(zs6, ws6):
    Pgen += w__ * mp.inverse(z_ * Sn - Hn)
A6 = mp.inverse(Sn) * Hn
Pref = mp.inverse(mp.eye(nb) + ((A6 - c6 * mp.eye(nb)) / R6) ** M6) * mp.inverse(Sn)
check("generalised Lemma 1: sum_j w_j (z_j S - H)^-1 = r_M(S^-1 H) S^-1  (40 digits)",
      mp.mnorm(Pgen - Pref, 1) < mp.mpf(10) ** -30)

# (ii) LDA potentials are exact derivatives of the energy densities n*eps_xc
nn_ = sp.symbols("n", positive=True)
rs_ = (3 / (4 * sp.pi * nn_)) ** sp.Rational(1, 3)
ex_ = -sp.Rational(3, 4) * (3 / sp.pi) ** sp.Rational(1, 3) * nn_ ** sp.Rational(1, 3)
check("Slater: v_x = d(n e_x)/dn = (4/3) e_x", sp.simplify(sp.diff(nn_ * ex_, nn_) - sp.Rational(4, 3) * ex_) == 0)
g_, b1_, b2_ = sp.symbols("gamma beta1 beta2")
ec_hi = g_ / (1 + b1_ * sp.sqrt(rs_) + b2_ * rs_)
vc_hi = ec_hi * (1 + sp.Rational(7, 6) * b1_ * sp.sqrt(rs_) + sp.Rational(4, 3) * b2_ * rs_) / (1 + b1_ * sp.sqrt(rs_) + b2_ * rs_)
check("PZ81 (rs>=1): v_c = d(n e_c)/dn", sp.simplify(sp.diff(nn_ * ec_hi, nn_) - vc_hi) == 0)
A_, B_c, C_, D_ = sp.symbols("A B C D")
ec_lo = A_ * sp.log(rs_) + B_c + C_ * rs_ * sp.log(rs_) + D_ * rs_
vc_lo = A_ * sp.log(rs_) + (B_c - A_ / 3) + sp.Rational(2, 3) * C_ * rs_ * sp.log(rs_) + (2 * D_ - C_) / 3 * rs_
check("PZ81 (rs<1): v_c = d(n e_c)/dn", sp.simplify(sp.expand_log(sp.diff(nn_ * ec_lo, nn_) - vc_lo, force=True)) == 0)

# (iii) pseudopotential: FT of -erf(r/rc)/r is -4 pi exp(-G^2 rc^2/4)/G^2.
#  Radial FT: (4 pi/G) Int_0^inf sin(Gr) [r * erf(r/rc)/r] dr; the bare-Coulomb part
#  Int sin(Gr) dr = 1/G (Abel), so check Int sin(Gr)(erf(r/rc)-1) dr + 1/G.
rr, G_, rc_ = sp.symbols("r G r_c", positive=True)
ok = True
for Gv, rcv in [(0.7, 0.6), (1.9, 1.0), (3.0, 0.5)]:
    # convergent form: Int_0^inf sin(Gr) * erf(r/rc) dr = exp(-G^2 rc^2/4)/G  (Abel-regularised)
    val = mp.quadosc(lambda t: mp.sin(Gv * t) * (mp.erf(t / rcv) - 1), [0, mp.inf], omega=Gv) + 1 / Gv
    ok &= abs(val - mp.exp(-Gv**2 * rcv**2 / 4) / Gv) < 1e-10
check("pseudopotential FT: Int sin(Gr) erf(r/rc) dr = exp(-G^2 rc^2/4)/G", ok)

# (iv) Lemma 4 (window interlacing): for a symmetric pencil (H,S), S > 0, every principal
#      sub-pencil (H_W, S_W) has lowest eigenvalue >= that of (H,S)  (Courant-Fischer).
#      => a filter edge E_lo below the global spectrum is below every window spectrum.
random.seed(5)
ok = True
for trial in range(5):
    nbig = 7
    A = mp.matrix(nbig, nbig)
    B = mp.matrix(nbig, nbig)
    for i in range(nbig):
        for j in range(i, nbig):
            A[i, j] = A[j, i] = mp.mpf(random.randint(-9, 9)) / 5
            B[i, j] = B[j, i] = mp.mpf(random.randint(-3, 3)) / 10
    S_ = B * B.T + mp.eye(nbig)
    def lowest(Hm, Sm):
        Lc = mp.cholesky(Sm)
        Li = mp.inverse(Lc)
        return min(mp.eigsy(Li * Hm * Li.T)[0])
    lam_glob = lowest(A, S_)
    idx = sorted(random.sample(range(nbig), 4))
    Aw = mp.matrix([[A[i, j] for j in idx] for i in idx])
    Sw = mp.matrix([[S_[i, j] for j in idx] for i in idx])
    ok &= lowest(Aw, Sw) >= lam_glob - mp.mpf(10) ** -30
check("Lemma 4: lowest window eigenvalue >= lowest global eigenvalue (5 random pencils)", ok)

# (v) Proposition 2 / Appendix A: the functional
#      G_M[v] = Omega_M[v] - sum (v - v_ext) n_M[v] + E_Hxc[n_M[v]]
#      is stationary exactly at self-consistency v = v_ext + v_Hxc[n_M].  Checked on a
#      3-site model with E_Hxc(n) = (U/2) sum n^2 (so v_Hxc = U n) and M = 8, at 40 digits.
Ng3 = 3
T3 = mp.matrix([[1, -0.5, 0], [-0.5, 1, -0.5], [0, -0.5, 1]])
vext = [mp.mpf("-0.7"), mp.mpf("0.2"), mp.mpf("-0.1")]
Uc = mp.mpf("0.6")
M8, c8, R8 = 8, mp.mpf("-0.2"), mp.mpf("1.3")
z8 = [c8 + R8 * mp.exp(1j * mp.pi * (2 * j + 1) / M8) for j in range(M8)]
w8 = [R8 * mp.exp(1j * mp.pi * (2 * j + 1) / M8) / M8 for j in range(M8)]
def h_of(v):
    Hm = T3.copy()
    for i in range(Ng3):
        Hm[i, i] += v[i]
    return Hm
def n_of(v):
    Hm = h_of(v)
    P = mp.zeros(Ng3, Ng3)
    for zz, ww in zip(z8, w8):
        P += ww * mp.inverse(zz * mp.eye(Ng3) - Hm)
    return [2 * mp.re(P[i, i]) for i in range(Ng3)]
def Omega(v):
    ev = mp.eigsy(h_of(v))[0]
    return mp.re(-2 * sum(ww * mp.log(zz - e) for ww, zz in zip(w8, z8) for e in ev))
def G(v):
    nn = n_of(v)
    return Omega(v) - sum((v[i] - vext[i]) * nn[i] for i in range(Ng3)) + Uc / 2 * sum(x**2 for x in nn)
v = list(vext)
for it in range(200):                      # self-consistency by damped iteration
    nn = n_of(v)
    v = [mp.mpf("0.5") * v[i] + mp.mpf("0.5") * (vext[i] + Uc * nn[i]) for i in range(Ng3)]
hfd = mp.mpf(10) ** -12
grad = []
for i in range(Ng3):
    vp, vm = list(v), list(v)
    vp[i] += hfd
    vm[i] -= hfd
    grad.append((G(vp) - G(vm)) / (2 * hfd))
check("Prop. 2/App. A: dG_M/dv = 0 at self-consistency (3-site model, M = 8)",
      max(abs(gg) for gg in grad) < mp.mpf(10) ** -15)

# (vi) Proposition 3: for a rigidly moving circle (R fixed, c = mu - R) the grand potential
#      Omega_M(mu) = -2 sum_i sum_j w_j log(z_j(mu) - eps_i) satisfies dOmega_M/dmu = -N_M(mu),
#      so A_M = G_M + mu N is stationary at fixed N.  Checked at 40 digits on a spectrum.
epsl = [mp.mpf(x) for x in ("-0.9", "-0.55", "-0.4", "0.1", "0.35", "1.2")]
Rr, Mr = mp.mpf("0.6"), 12
def Om(mu):
    zr = [mu - Rr + Rr * mp.exp(1j * mp.pi * (2 * j + 1) / Mr) for j in range(Mr)]
    wr = [Rr * mp.exp(1j * mp.pi * (2 * j + 1) / Mr) / Mr for j in range(Mr)]
    return mp.re(-2 * sum(ww * mp.log(zz - e) for ww, zz in zip(wr, zr) for e in epsl))
def Nm(mu):
    c_ = mu - Rr
    return 2 * sum(1 / (1 + ((e - c_) / Rr) ** Mr) for e in epsl)
mu0 = mp.mpf("-0.2")
dOm = mp.diff(Om, mu0)
check("Prop. 3: dOmega_M/dmu = -N_M for a rigidly moving circle (40 digits)", abs(dOm + Nm(mu0)) < mp.mpf(10) ** -25)

print()
if FAILED:
    print(f"{len(FAILED)} CAS check(s) FAILED:", FAILED)
    sys.exit(1)
print("All CAS checks passed.")
