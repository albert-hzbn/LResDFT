"""
Computer-algebra (SymPy) derivation and verification of every analytic
ingredient of Response-Matched Orbital-Free DFT (RM-OFDFT) in 1D.

Run:  python3 cas_verification.py
Every check prints PASS/FAIL; the script exits non-zero on any failure.

Conventions: Hartree atomic units, spin-unpolarised electrons (2 per orbital),
kF = pi*n/2 for the 1D homogeneous electron gas (HEG).
"""
import sys
import sympy as sp

FAILED = []


def check(name, cond):
    ok = bool(cond)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILED.append(name)


x, k, q, kF, n0, r = sp.symbols("x k q k_F n_0 r", positive=True)
eta = sp.symbols("eta", positive=True)

# ---------------------------------------------------------------------------
# 1. Homogeneous gas: density, kinetic energy density -> Thomas-Fermi constant
# ---------------------------------------------------------------------------
dens = 2 * sp.integrate(1 / (2 * sp.pi), (k, -kF, kF))                # n(kF)
check("1D HEG density n = 2 kF / pi", sp.simplify(dens - 2 * kF / sp.pi) == 0)
tdens = 2 * sp.integrate(k**2 / 2 / (2 * sp.pi), (k, -kF, kF))        # t(kF)
n = sp.symbols("n", positive=True)
t_of_n = sp.simplify(tdens.subs(kF, sp.pi * n / 2))
C_TF = sp.simplify(t_of_n / n**3)
check("T_TF = (pi^2/24) * Int n^3", sp.simplify(C_TF - sp.pi**2 / 24) == 0)

# ---------------------------------------------------------------------------
# 2. 1D Lindhard function (static, spin-summed), principal value, q < 2 kF
#    chi0(q) = 2 * Int_{-kF}^{kF} dk/(2pi) [1/(e_k - e_{k+q}) + 1/(e_k - e_{k-q})]
# ---------------------------------------------------------------------------
# PV antiderivative of 1/(a k + b) is log|a k + b|/a; evaluate explicitly.
def pv_int(a, b, lo, hi):
    return (sp.log(sp.Abs(a * hi + b)) - sp.log(sp.Abs(a * lo + b))) / a

I1 = pv_int(-q, -q**2 / 2, -kF, kF)   # e_k - e_{k+q} = -(k q + q^2/2)
I2 = pv_int(q, -q**2 / 2, -kF, kF)    # e_k - e_{k-q} =  (k q - q^2/2)
chi0 = 2 / (2 * sp.pi) * (I1 + I2)
chi0_closed = -2 / (sp.pi * q) * sp.log(sp.Abs((2 * kF + q) / (2 * kF - q)))
# compare numerically at several (q,kF) on both sides of 2kF
ok = True
for qv, kv in [(0.3, 1.0), (1.7, 1.0), (2.5, 1.0), (7.0, 1.3), (0.01, 2.0)]:
    a = complex(chi0.subs({q: qv, kF: kv}).evalf())
    b = complex(chi0_closed.subs({q: qv, kF: kv}).evalf())
    ok &= abs(a - b) < 1e-12
check("Lindhard chi0(q) = -(2/(pi q)) ln|(2kF+q)/(2kF-q)|", ok)
chi0_eta = -1 / (sp.pi * kF * eta) * sp.log((1 + eta) / (1 - eta))   # eta=q/2kF<1
check("chi0 in eta=q/(2kF) form",
      sp.simplify(chi0_closed.subs(q, 2 * kF * eta).rewrite(sp.log)
                  .subs(sp.Abs((2 * kF + 2 * kF * eta) / (2 * kF - 2 * kF * eta)),
                        (1 + eta) / (1 - eta)) - chi0_eta) == 0)
# compressibility sum rule: chi0(0) = -g(eF) = -2/(pi kF)
check("chi0(q->0) = -2/(pi kF)  (density of states)",
      sp.simplify(sp.limit(chi0_eta, eta, 0) + 2 / (sp.pi * kF)) == 0)

# ---------------------------------------------------------------------------
# 3. Second functional derivatives (uniform-gas response) of each functional
#    For F[n] = Int f(n, n') dx, delta^2 F / delta n delta n (q) at n=n0 is
#    f_nn - q^2 * f_{n'n'}  (from the quadratic expansion).
# ---------------------------------------------------------------------------
n_, np_ = sp.symbols("n n'", positive=True)
f_TF = sp.pi**2 / 24 * n_**3
f_vW = np_**2 / (8 * n_)
def resp(f):
    return sp.simplify(sp.diff(f, n_, 2).subs(np_, 0) + q**2 * sp.diff(f, np_, 2).subs(np_, 0)).subs(n_, n0)
R_TF = resp(f_TF)
R_vW = resp(f_vW)
check("TF response = pi^2 n0/4", sp.simplify(R_TF - sp.pi**2 * n0 / 4) == 0)
check("vW response = q^2/(4 n0)", sp.simplify(R_vW - q**2 / (4 * n0)) == 0)
check("TF response = -1/chi0(0)",
      sp.simplify(R_TF.subs(n0, 2 * kF / sp.pi) + 1 / sp.limit(chi0_eta, eta, 0)) == 0)

# ---------------------------------------------------------------------------
# 4. The response-matched kernel.
#    Require  R_TF + R_vW + R_NL(q) = -1/chi0(q)   for all q.
#    R_NL for T_NL = Int Int n^a(x) w(x-x') n^a(x') is 2 a^2 n0^(2a-2) w(q).
#    With a = 3/2 the prefactor is (9/2) n0, so w is a pure function of eta
#    times a constant -> scale-invariant (dimensionally consistent) kernel.
# ---------------------------------------------------------------------------
a = sp.Rational(3, 2)
minv_chi = -1 / chi0_eta                                  # = pi kF eta / L(eta)
K_eta = minv_chi - (R_TF + R_vW).subs({n0: 2 * kF / sp.pi, q: 2 * kF * eta})
K_eta = sp.simplify(K_eta)
w_eta = sp.simplify(K_eta / (2 * a**2 * (2 * kF / sp.pi) ** (2 * a - 2)))
print("   Kernel  w(eta) =", w_eta)
check("w(eta) independent of kF (a=3/2 => scale invariant)",
      sp.simplify(sp.diff(w_eta, kF)) == 0)
# Small-eta: exact T_s -> T_TF - (1/3) T_vW (2nd-order gradient expansion), so the
# kernel on top of the *full* vW term must behave as -(4/3) x vW response.
ser = sp.series(K_eta, eta, 0, 4).removeO()
vW_eta = (q**2 / (4 * n0)).subs({n0: 2 * kF / sp.pi, q: 2 * kF * eta})
check("K(q->0) = -(4/3) vW response  (=> TF + vW + NL -> TF - vW/3, the GEA)",
      sp.simplify(ser - (-sp.Rational(4, 3) * vW_eta)) == 0)
check("K(0) = 0  => T_NL vanishes for the uniform gas", sp.limit(K_eta, eta, 0) == 0)
# Large-eta (eta>1 branch uses log((eta+1)/(eta-1)))
K_big = sp.pi * kF * eta / sp.log((eta + 1) / (eta - 1)) - (R_TF + R_vW).subs(
    {n0: 2 * kF / sp.pi, q: 2 * kF * eta})
check("K(q->inf) -> -(4/3) * TF response = -2 pi kF/3",
      sp.simplify(sp.limit(K_big, eta, sp.oo) + 2 * sp.pi * kF / 3) == 0)
check("K(q = 2kF) = -pi kF (log singularity of chi0 handled)",
      sp.simplify(sp.limit(K_eta, eta, 1, "-") + sp.pi * kF) == 0)

# ---------------------------------------------------------------------------
# 5. Euler-Lagrange (functional derivative) formulas used in the code
# ---------------------------------------------------------------------------
nf = sp.Function("n")(x)
L_vW = sp.diff(nf, x) ** 2 / (8 * nf)
EL = sp.euler_equations(L_vW, nf, x)[0].lhs
# code uses v_vW = -psi''/(2 psi) with psi = sqrt(n)
psi = sp.sqrt(nf)
v_code = -sp.diff(psi, x, 2) / (2 * psi)
check("delta T_vW/delta n = -psi''/(2 psi)", sp.simplify(EL - v_code) == 0)
check("delta T_TF/delta n = pi^2 n^2/8",
      sp.simplify(sp.diff(sp.pi**2 / 24 * nf**3, nf) - sp.pi**2 * nf**2 / 8) == 0)
# Nonlocal term: d/d eps T_NL[n + eps h] for a discrete symmetric kernel matrix
N = 3
nv = sp.symbols("n0:%d" % N, positive=True)
W = sp.Matrix(N, N, lambda i, j: sp.Symbol(f"w{min(i, j)}{max(i, j)}"))
p = sp.Matrix([v**a for v in nv])
T_NL = (p.T * W * p)[0]
grad = [sp.diff(T_NL, v) for v in nv]
grad_code = [2 * a * nv[i] ** (a - 1) * (W * p)[i] for i in range(N)]
check("delta T_NL/delta n = 2 a n^(a-1) (w * n^a)",
      all(sp.simplify(g1 - g2) == 0 for g1, g2 in zip(grad, grad_code)))

# ---------------------------------------------------------------------------
# 6. Scaling: T_s[n_l](n_l(x) = l n(l x)) = l^2 T_s[n]. Check each term.
# ---------------------------------------------------------------------------
lam = sp.symbols("lambda", positive=True)
# TF: Int (l n(lx))^3 dx = l^2 Int n^3 ;  vW: likewise ; NL with scale-free w
# expressed via substitution u = l x  (symbolic check on a Gaussian density)
g = sp.exp(-x**2)
def TF(dfun):
    return sp.integrate(sp.pi**2 / 24 * dfun**3, (x, -sp.oo, sp.oo))
def VW(dfun):
    return sp.integrate(sp.diff(dfun, x) ** 2 / (8 * dfun), (x, -sp.oo, sp.oo))
gl = lam * g.subs(x, lam * x)
check("T_TF[n_l] = l^2 T_TF[n]", sp.simplify(TF(gl) - lam**2 * TF(g)) == 0)
check("T_vW[n_l] = l^2 T_vW[n]", sp.simplify(VW(gl) - lam**2 * VW(g)) == 0)

# ---------------------------------------------------------------------------
# 7. Exchange of the soft-Coulomb 1D gas: e_x(n) = -(2/pi^2) Int_0^inf
#    sin^2(kF r) / (r^2 sqrt(1+r^2)) dr   (derived from the HEG density matrix)
# ---------------------------------------------------------------------------
gamma = 2 * sp.sin(kF * r) / (sp.pi * r)            # spin-summed 1-RDM
ex_integrand = -sp.Rational(1, 4) * gamma**2 / sp.sqrt(1 + r**2) * 2   # both signs of r
check("exchange integrand form",
      sp.simplify(ex_integrand + 2 / sp.pi**2 * sp.sin(kF * r) ** 2 / (r**2 * sp.sqrt(1 + r**2))) == 0)
# Closed form.  With b = 2 kF:  Int_0^inf (1-cos br)/(r^2 sqrt(1+r^2)) dr = I(b),
# I''(b) = Int cos(br)/sqrt(1+r^2) = K0(b),  I(0)=I'(0)=0  =>
# I(b) = b*IK0(b) + b K1(b) - 1,   IK0(b) = Int_0^b K0.
# e_x(n) = -(1/pi^2) I(pi n),   v_x(n) = d e_x/dn = -(1/pi) IK0(pi n).
bb = sp.symbols("b", positive=True)
IK0 = sp.Function("IK0")
I_closed = bb * IK0(bb) + bb * sp.besselk(1, bb) - 1
dI = sp.diff(I_closed, bb).subs(sp.Derivative(IK0(bb), bb), sp.besselk(0, bb))
check("d/db [b IK0 + b K1 - 1] = IK0(b)  (so I'' = K0)",
      sp.simplify(sp.expand_func(dI - IK0(bb))) == 0)
check("I(0) = 0 (b K1(b) -> 1)", sp.limit(bb * sp.besselk(1, bb), bb, 0) == 1)
import mpmath as mp
ok = True
for bv in [0.1, 0.7, 2.0, 5.0]:
    direct = mp.quad(lambda t: (1 - mp.cos(bv * t)) / (t**2 * mp.sqrt(1 + t**2)), [0, 1, 10, mp.inf])
    ik0 = mp.quad(lambda t: mp.besselk(0, t), [0, bv])
    closed = bv * ik0 + bv * mp.besselk(1, bv) - 1
    ok &= abs(direct - closed) < 1e-8
check("closed-form exchange integral matches direct quadrature", ok)
check("1-RDM diagonal gamma(0) = n",
      sp.simplify(sp.limit(gamma, r, 0) - 2 * kF / sp.pi) == 0)

print()
if FAILED:
    print(f"{len(FAILED)} CAS check(s) FAILED:", FAILED)
    sys.exit(1)
print("All CAS checks passed.")
