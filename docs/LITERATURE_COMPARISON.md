# Comparison with literature values

Script: `benchmarks/literature.py` (data: `results/literature.json`), KS solver, default settings (`dx = 0.3`).
Reference values are experimental unless stated and were entered from memory of standard tables (NIST CCCBDB),
not re-checked against a source in this session; the all-electron LDA numbers are approximate.

| Quantity | This tool (defaults) | Reference | Verdict |
|---|---|---|---|
| H₂ bond length r_e | 0.976 Å | 0.741 Å (exp.), ≈0.765 Å (LDA) | 30 % too long |
| N₂ bond length r_e | 1.448 Å | 1.098 Å (exp.) | 32 % too long |
| He total energy | −1.821 Ha | −2.835 Ha (LDA, all-electron) | 1.0 Ha too high |
| H₂O dipole | 0.84 D | 1.85 D (exp.) | far too small |
| NH₃ dipole | 0.99 D | 1.47 D (exp.) | too small |
| HF dipole | 0.43 D | 1.83 D (exp.) | far too small |
| LiH dipole | 1.00 D | 5.88 D (exp.) | far too small |

**Conclusion: with default settings the tool is not quantitatively accurate against literature.**

## Why

The model replaces each nucleus by the softened potential `-Z erf(r/rc)/r` (a Gaussian charge of width `rc`) and uses a
7-point finite-difference kinetic operator on a 0.3 bohr grid. A softened core is weaker than the true Coulomb
potential, so atoms are too weakly bound, bonds too long and densities too diffuse. For Li–Ne the valence charge and `rc`
are in addition unfitted (`lrft/molecules.py`).

Sensitivity test (`rc` and `dx` reduced together, larger basis; one-off script, not part of the repository):

| dx, rc (bohr) | He energy (Ha) | H₂ r_e (Å) |
|---|---|---|
| 0.30, 0.5 / 0.6 | −1.832 | 0.979 |
| 0.20, 0.3 | −2.216 | 0.840 |
| 0.15, 0.2 | −2.439 | 0.801 |

Results move monotonically toward the references but have not reached them at dx = 0.15 (cost grows as dx⁻³).
Reaching literature accuracy needs a real pseudopotential (or true all-electron treatment) rather than a softened
Coulomb core; that is outside the present model.

## What the tool *is* validated for

The CLI is a faithful implementation of the manuscript's comparison: KS and LRes-DFT on the **identical** model
Hamiltonian. That comparison is what the examples test (`examples/run_all.sh`): 14 systems from 2 to 30 electrons,
all SCF-converged; KS–LRes energy difference per electron is ≤ 7×10⁻⁴ Ha (≈10⁻¹⁴ when windows cover the molecule).
Use the absolute energies and geometries for qualitative work only.
