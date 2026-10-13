> **Superseded.** These are early working notes. Several statements here were later corrected (e.g. the kinetic energy is *not* linear in g; the potential-inversion identity is not a Hohenberg–Kohn result; Theorem 4(b,c) are remarks, not theorems; the 3D algorithm is divide-and-conquer with the r_M occupation). The authoritative statements are in `manuscript/main.tex`.

# Local Resolvent Field Theory (LRFT)

*A density-based, orbital-free and kinetic-functional-free reformulation of the
electronic many-body problem whose cost is linear in system size.*

All CAS-checkable statements below are verified in `cas_verification.py`.
The implementation is checked against the theorems in `tests/test_lrft.py`.

---

## 0. Why a new variable?

DFT replaces the N-body wavefunction by the density n(x). That is a great
reduction, but it creates two problems:

* **Kohn–Sham (KS) DFT** puts back N/2 orbitals to get the kinetic energy.
  Storing them costs O(N²) and orthogonalising them costs O(N³). Above about 10³ atoms this cost dominates.
* **Orbital-free DFT** keeps n alone. Then the kinetic energy T_s[n] is an
  unknown, highly non-local functional, and every approximation to it is
  uncontrolled. Shell structure, bonding and Friedel oscillations are lost.

Both problems have the same cause. **The density is the energy-integrated
local density of states, ρ(x,ε) integrated up to μ. The energy resolution has
been thrown away, and it is exactly what the kinetic energy needs.**

LRFT keeps that information without orbitals. It uses a quantity that is
density-like, being local in x, but more fundamental than n.

## 1. The fundamental variable

For the effective one-body Hamiltonian ĥ = −½∇² + v(x), define the
**local resolvent field**

    g(x, z) = ⟨x|(z − ĥ)⁻¹|x⟩ = Σᵢ |φᵢ(x)|² / (z − εᵢ) = ∫ dε ρ(x,ε) / (z − ε),   z ∈ ℂ \ σ(ĥ).

g(x,z) is the Stieltjes transform of the local density of states. It is a scalar
field in x, like the density, labelled by a complex energy z.
The density is a single contour moment of g:

    n(x) = 2 · (1/2πi) ∮_C g(x,z) dz,     C encloses the occupied spectrum.

So g contains n, together with the energy resolution that n discards.

## 2. Exact representation theorems (consistency)

**Theorem 1 (representation).** For any local ĥ:

* **(a)** n is a *linear* functional of g (contour formula above).
* **(b)** v is an *explicit, local* functional of g:
  v(x) = (1/2πi)∮_{|z|=R>‖ĥ‖} z·g(x,z) dz − t_xx, where t_xx = 1/dx² is the kinetic diagonal.
  *Proof:* the Neumann series is g = 1/z + h_xx/z² + (h²)_xx/z³ + …, and the residue of z·g at ∞ is h_xx. *(CAS ✓; test ✓ to 1e-10.)*
  Compare DFT, where n → v is a global, ill-conditioned inverse problem.
* **(c)** The band energy, and hence the KS kinetic energy, is *linear* in g:
  E_band = 2·(1/2πi)∮ z Σₓ g(x,z) dz and T_s = E_band − ∫ v n. No kinetic functional has to be approximated. *(CAS ✓)*
* **(d)** The map v ↦ g is injective, by (b). Together with Hohenberg–Kohn,
  every ground-state observable is therefore a functional of g. The
  interacting energy is

      E[g] = T_s[g] + ∫ v_ext n[g] + E_H[n[g]] + E_xc[n[g]] + E_nn.

  Its only non-exact ingredient is E_xc, exactly as in KS-DFT. Given the exact
  E_xc, LRFT is exact (Theorem 5).

## 3. Discretisation in energy: the Butterworth–trapezoid ensemble

Take the circle through E_lo (below the spectrum) and μ: centre c = (μ+E_lo)/2,
radius R = (μ−E_lo)/2. Sample it at the M half-step nodes
z_j = c + R ζ_j, where ζ_j = e^{iπ(2j+1)/M} are the M roots of ζ^M = −1, with weights w_j = Rζ_j/M.

**Lemma 1 (closed form).** (1/M) Σ_j ζ_j/(ζ_j − u) = 1/(1+u^M).
Hence P_M := Σ_j w_j (z_j − ĥ)⁻¹ = [1 + ((ĥ−c)/R)^M]⁻¹ *exactly*.
*Proof:* 1/(1+u^M) has simple poles at ζ_j with residues 1/(Mζ_j^{M−1}) = −ζ_j/M
and vanishes at infinity. *(CAS ✓ for M = 2…14; exact matrix identity to 40 digits ✓.)*

The conventional trapezoid rule (nodes ζ^M = +1) gives instead 1/(1−u^M). Its
occupations exceed 1 or go negative, so it is not N-representable.

**Lemma 2 (N-representability and error bound).** For even M and real
u = (ε−c)/R,

    r_M(u) = 1/(1+u^M) = 1/(1 + e^{M·ln|u|}),

so r_M is a Fermi–Dirac distribution with inverse temperature M in the
log-energy variable s = ln|u|. Therefore:

* every occupation lies in (0,1], so P_M is an *ensemble N-representable* one-body
  density matrix (Coleman), the same class as Mermin finite-temperature DFT;
* if |u| ≤ ρ < 1, then 1 − r_M ≤ ρ^M/(1+ρ^M);
* if |u| ≥ 1/ρ, then r_M ≤ ρ^M/(1+ρ^M).

*(CAS ✓; test: eigenvalues of P_M lie in [0,1] ✓.)*

**Theorem 2 (variational consistency at every finite M).** Define
Ω_M[v] = Tr W_M(ĥ[v]), with W_M(ε) = −2 Σ_j w_j log(z_j − ε), so that W_M′ = 2r_M. Then

    ∂Ω_M / ∂v(x) = n_M(x)    exactly,

because d Tr F(ĥ) = Tr F′(ĥ) dĥ by cyclicity, and ∂ĥ/∂v(x) = |x⟩⟨x|.
*(CAS: Jacobi formula ✓; test: finite difference of Ω_M equals n_M to 1e-7 at M = 12 ✓.)*

Consequences:

* the LRFT self-consistent field is the stationary point of a well-defined
  Mermin-type functional F_M[n] = sup_v {Ω_M[v] − ∫vn} + E_ext + E_Hxc, by the standard Legendre argument;
* energy errors are second order in density errors;
* Hellmann–Feynman forces are exact at every M, not only as M → ∞.

**Theorem 3 (exponential convergence; M is intensive).** Let E_lo = E_min − δ,
let Δ₋ = μ − ε_HOMO and Δ₊ = ε_LUMO − μ, and set
ρ = max(1 − δ/R, 1 − Δ₋/R, 1/(1+Δ₊/R)) < 1. Then

    ‖n_M − n‖₁ ≤ Σᵢ 2|r_M(uᵢ) − θᵢ| ≤ 2(N_occ + N_near) ρ^M + 2 N_g 2^{−M},

since Σₓ|A_xx| ≤ ‖A‖_trace, and Lemma 2 applies state by state. Here N_near counts the states with 1 < |u| < 2.
R is set by the occupied band width, not by the grid's spectral width of order 1/dx².
So M = O((W_occ/Δ) · log(N/ε)) depends on the system size only through a logarithm.
*(Benchmark C: the measured error is below the bound for every M until machine precision; ρ ≈ 0.72 for the dimer chain.)*

## 4. Cost theorems

**Theorem 4a (exact linear-time evaluation in 1D).** For tridiagonal ĥ, compute the
left pivots L_i = a_i − b_{i−1}²/L_{i−1} and the right pivots U_i = a_i − b_i²/U_{i+1}. Then

    G_ii(z) = 1/(L_i + U_i − a_i),

so every g(·, z_j) costs O(N_g) operations and O(N_g) memory, with no approximation.
*(CAS ✓ symbolic for n = 5; test ✓ against the dense inverse.)*

**Theorem 4b (nearsightedness in any dimension).** For z ∉ σ(ĥ), the
off-diagonal resolvent decays exponentially: |G_xy(z)| ≤ C e^{−γ(z)|x−y|}, with
γ > 0 increasing with dist(z, σ). This is a Combes–Thomas bound. For a constant potential,
G_m ∝ λ^{|m|} with λ² + 2αλ + 1 = 0; the two roots multiply to 1, so one root has |λ| < 1 *(CAS ✓)*.
G_xx can therefore be computed from a local window of radius r with error O(e^{−γr}).
The total cost is

    C_LRFT = O(N_g · M · r^d) = O(N)        (M and r are intensive).

*(Test: window error decays exponentially ✓. Benchmark: a fixed window gives the
same error at every system size.)*

**Theorem 4c (lower bound for any orbital method).** Any method that outputs
N_occ orthonormal orbitals on N_g grid points must write N_occ·N_g numbers, so it
costs Ω(N²) in time and memory. Generic KS solvers also orthogonalise, which costs O(N_occ²·N_g) = O(N³).
Hence

    speed-up  S(N) = C_KS / C_LRFT ≥ Ω(N)   (vs. any orbital method),    Θ(N²) vs. standard O(N³) KS.

The bound holds for any orbital-based algorithm, not just a weak implementation. 1D LAPACK bisection with
inverse iteration actually reaches the Ω(N²) lower bound, which makes it the toughest possible baseline.

## 5. Interacting problem and equivalence with KS

**Theorem 5.** Let Φ_KS and Φ_M be the KS and LRFT self-consistent maps
n ↦ n_out. They share the same Hartree and xc terms and differ only in the
one-body step. By Theorem 3, ‖Φ_M − Φ_KS‖ ≤ Cρ^M. If Φ_KS is a contraction near its fixed
point n*, the LRFT fixed point n*_M satisfies ‖n*_M − n*‖ = O(ρ^M), and the
energy error is O(ρ^{2M}) by the stationarity in Theorem 2.
*Tests:* He, H₂, LiH, and an H₆ chain all agree with KS to 1e-7 Ha or better; the benchmarks reach 1e-12 to 1e-15 Ha.

The approximation to the N-body problem is therefore exactly that of DFT with
the chosen E_xc. LRFT removes the cubic cost and adds no kinetic-functional error.

## 6. Relation to prior work (honest positioning)

The ingredients have precursors:

* Green's-function and LDOS representations (Haydock recursion; the Stieltjes transform);
* Fermi-operator and pole expansions (Goedecker FOE; Ozaki; PEXSI by Lin, Lu, Ying and E);
* nearsightedness (Kohn 1996).

To my knowledge, LRFT's specific contributions are:

1. treating the local resolvent field itself as the fundamental variable, with
   explicit inversion (Thm 1b) and a kinetic energy that is explicit and linear (Thm 1c);
2. the half-step "Butterworth" trapezoid rule, with closed form 1/(1+u^M). It is
   exactly a Fermi ensemble in log-energy, so it is N-representable and variational at
   every M (Lemma 1, Lemma 2, Thm 2), which the standard trapezoid rule is not;
3. pole count governed by W_occ/Δ and independent of the grid's spectral width,
   unlike Chebyshev FOE, where the order grows like 1/dx².

## 7. Limitations

* **Small gaps and metals:** ρ → 1 as Δ → 0, so M grows like W_occ/Δ.
  Stretched H₂ at R = 6 needs more poles; a uniform H chain gives 1e-7 Ha per electron at M = 128.
  A finite electronic temperature is the standard remedy, because r_M is already a Fermi function.
* **Dimension:** the demonstration is 1D. In 3D the prefactor M·r³ is large, so the
  crossover with KS will occur at larger N than in 1D.
* **E_xc:** E_xc is still approximated, as in every DFT. Here the 1D soft-Coulomb LDA exchange is used,
  and its error against the exact many-body energy is about 0.08 Ha for He.

---

## 8. Extension to three dimensions

The theorems in §§2–5 do not depend on dimension. Lemma 1 holds unchanged in a
non-orthogonal basis {χ_μ} with overlap S:

    Σ_j w_j (z_j S − H)⁻¹ = r_M(S⁻¹H) S⁻¹            (CAS ✓, 40 digits)

The only thing that changes in 3D is **how g is evaluated**. The exact two-sided
recursion of Theorem 4a is specific to 1D.

**What does not work: pointwise probing on a fine grid.** I implemented this
(`lrft/three_d.py`: colour probing plus batched Lanczos Gauss quadrature). On a
0.6-bohr grid the density matrix of an H₂ crystal stays comparable to its diagonal over
about 3 bohr, which is around 10³ grid points per localisation volume. Probes with spacing
d then pick up O(1) contamination (measured), and controlling it needs thousands of probes.
This is the well-known reason why every production linear-scaling DFT code (SIESTA, CONQUEST,
ONETEP, CP2K) uses localized basis functions.

**What works: nearsighted windows in a localized basis** (`lrft/lcao3d.py`):

* Basis: 6 contracted Gaussians per H atom, tabulated on the real-space grid.
  H = Xᵀ(T_FD + v)X dV is the Galerkin projection of the grid Hamiltonian.
  KS and LRFT use this identical H, S, Hartree and LDA (Slater exchange + PZ81 correlation).
* Each molecular cell B owns its Voronoi region of grid points. Its window W(B) contains all cells
  within R_w, and the density on the owned points is

      n(r) = 2 Σ_k r_M(λ_k) |φ_k^{W}(r)|²,   (H_W, S_W)φ_k = λ_k S_W φ_k.

  This is Lemma 1 evaluated on the window spectrum, which is identical to the pole sum.
  It is exact as R_w → ∞, and by nearsightedness (Theorem 4b) the error decays exponentially in R_w.
* N(μ) and E_band are explicit functions of μ through the window spectra, so finding μ costs nothing.
  Inside the gap, μ from the previous SCF step gives the same density to O(ρ^M).
  In that case only eigenpairs below c + 1.5R are needed (LAPACK `sygvx`), because r_M < 1.5^{−M} above it.
* Cost: N_cells × O(n_W³), where n_W does not depend on system size. **That is O(N)**, against
  O(N_b³) for KS with a dense eigensolver. Memory is O(N) against O(N_b²).

**Measured (H₂ molecular crystal, 1 thread, both solvers in C++ with the same OpenBLAS):**

* Window convergence on a 5×5×5 crystal: the error decays exponentially once the window
  reaches the second neighbour shell. The density error per electron is 3e-4, 1e-4, 3.6e-6 and 3e-8
  at R_w = 6.1, 8.5, 12.1 and 13.5 bohr.
* Linear scaling: t_LRFT ∝ N^0.94, against t_KS ∝ N^3.04. These match the cost theorems, which predict N^1 and N^3.
* Crossover with dense LCAO-KS at about 430 electrons. LRFT is **6.2× faster at N = 1024**, and about 25× at
  N = 2000, where KS is extrapolated as N³ because its O(N²) memory no longer fits.
* Across all sizes, LRFT matches KS to about 1e-8 Ha per electron in band energy and about 8e-7 per electron in density.
  The self-consistent energies agree to 1e-13 to 2e-7 Ha per electron.
* See `README.md` §4 and `results/three_d.json`.

**3D limitations:**

* The crossover lies at hundreds of electrons, not tens. This is typical of linear-scaling methods.
* Windows must be about 12 bohr for 1e-6 density accuracy.
* Results depend on the gap. All tests are on an insulator; metals would need a finite electronic temperature.
