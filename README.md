# Local-resolvent DFT (LRes-DFT)

The Kohn–Sham density is written in terms of the diagonal of the one-electron resolvent,
g(r,z) = ⟨r|(z − h)⁻¹|r⟩, sampled at M complex energies on a circle. A half-step trapezoid rule gives the occupation
1/(1+u^M). With local windows the method scales linearly; in 3D the window construction is divide-and-conquer DFT with
this occupation function.

## Main results (all reproducible from `results/*.json`)

**Formulation**
- The density and the band energy are linear in g.
- Occupations lie in [0,1] for every M.
- With a circle that moves rigidly with μ, fixed-N forces are exact at every M.
- The a priori error bound does not depend on the grid spacing.
- 30 computer-algebra checks and 25 automated tests pass.

**Pole rules (1D, same Hamiltonians)**
- The half-step rule needs more poles than Gauss–Legendre on the same circle.
- It needs far more than Zolotarev for small gaps and deep core states: 213 vs 16 solves at a gap of 0.014 Ha.
- It is the only one of the three whose occupations stay exactly in [0,1].

**3D H₂ crystals, LCAO basis, identical Hamiltonian for KS and LRes-DFT**
- Self-consistent energies agree with KS to 2×10⁻⁷ Ha per electron up to 1024 electrons.
- Cost (single thread, idle machine, 3 interleaved repeats, spread <7%; `results/scaling_rigorous*.json`): complete SCF wall time scales as N^2.92±0.02 (dense KS) and N^1.15±0.06 (LRes). LRes is slower below ~500 electrons (0.17× at 54, 0.75× at 432), 2.0× faster at 686 and 3.5× faster at 1024. Dense KS was not run at 2000 electrons. No comparison with ELPA/parallel solvers. For GTH molecules (≤10 atoms) KS and LRes take the same time.
- LRes-DFT uses less memory than dense KS.
- The KS reference uses the fastest of three LAPACK paths; warm-started LOBPCG was slower than dense in this basis.

**Equivalence with divide-and-conquer**
- In 3D, the occupation 1/(1+u^M) and Fermi–Dirac occupations in the same windows give the same accuracy.

**Limitations (measured)**
- Window error is ten times larger for an amorphous arrangement.
- Window error grows strongly under compression: 9.5×10⁻⁴ per electron at a gap of 0.05 Ha.
- Window error grows with basis diffuseness.
- Metals, nonlocal pseudopotentials and 3D forces were not treated.

## Command-line tool (atoms and molecules)

```
./build_cpp.sh                                   # optional: C++ kernels (else pure Python is used)
python3 -m lrft list                             # built-in molecules and elements
python3 -m lrft run H2O                          # Kohn-Sham single point
python3 -m lrft run mol.xyz --solver both --rw 10 --json out.json --cube dens.cube
python3 -m lrft run HeH+ --solver lres -M 64 --dx 0.25 -v
```

`run` places the system (built-in name or `.xyz` in angstrom; `charge=<n>` on the comment line) in a cubic vacuum box and does a
self-consistent closed-shell LDA calculation with either dense Kohn-Sham (`--solver ks`), the local-resolvent windows
(`--solver lres`, one window of radius `--rw` per atom, `-M` complex energies) or both with a comparison (`both`). It reports the
total energy (including the Gaussian-ion repulsion), HOMO/LUMO (KS) or the chemical potential (LRes), dipole, and optionally
writes JSON and a cube file. Exit status is 0 only if every SCF converged.

Full tutorial and option reference: `docs/CLI_TUTORIAL.md`.

Scope and limits: the Hamiltonian is the model of the manuscript (local pseudopotentials, no spin polarisation, no forces).
The model is not quantitatively accurate against literature (H2 bond length 0.98 vs 0.74 A, He energy -1.82 vs -2.83 Ha, dipoles far too small; see `docs/LITERATURE_COMPARISON.md`); Li-Ne additionally use an *unfitted* valence parametrisation (`lrft/molecules.py`). Use it for KS vs LRes comparisons and qualitative work. With `--rw` larger than the molecule, LRes equals KS exactly (e.g. H2, HeH+ agree to 1e-14 Ha); smaller
windows converge toward KS as `--rw` grows (H12 chain: 4.8e-3, 1.5e-4, 9.5e-5 Ha/electron at 5, 8, 12 bohr).

## Layout

| Path | Content |
|---|---|
| `lrft/core.py` | 1D model, exact two-sided recursion, KS reference |
| `lrft/three_d.py` | 3D grid model (pseudopotentials, LDA, FFT Hartree), grid KS (LOBPCG), grid-probing attempt |
| `lrft/lcao3d.py`, `src/lrft_cpp.cpp` | 3D LCAO: KS (three LAPACK paths) and LRes-DFT windows; Python and C++ back ends |
| `lrft/cli.py`, `lrft/molecules.py`, `lrft/__main__.py` | command-line tool for atoms and molecules (`python3 -m lrft`) |
| `cas_verification.py` | 30 computer-algebra checks |
| `tests/` | 25 automated tests |
| `benchmarks/` | accuracy, scaling, 3D, referee checks (R1–R13), large SCF, KS solver comparison |
| `results/` | all data (JSON) and logs |

## Reproduce

```
./build_cpp.sh
python3 cas_verification.py && python3 -m pytest -q tests
python3 benchmarks/accuracy.py && python3 benchmarks/scaling.py && python3 benchmarks/extra_checks.py
python3 benchmarks/three_d.py ABC
python3 benchmarks/referee_checks.py R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18
python3 benchmarks/scf3d_large.py && python3 benchmarks/ks_drivers.py
```
