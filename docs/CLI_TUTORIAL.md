# Command-line tool: documentation and tutorial

`python3 -m lrft` runs self-consistent calculations on atoms and molecules with either dense Kohn–Sham (KS) or the
local-resolvent (LRes-DFT) window solver, on the identical Hamiltonian, so the two can be compared directly.

## 1. Setup

Requirements: Python ≥ 3.10, `numpy`, `scipy`. Optional (faster): the C++ kernels.

```
cd LResDFT
pip install numpy scipy pybind11      # pybind11 only for the C++ build
./build_cpp.sh                        # optional; needs g++, OpenBLAS, OpenMP
python3 -m lrft list
```

Without the compiled extension the tool falls back to pure Python (`--backend auto`); `--backend cpp` makes a missing
extension an error. Always run from the `LResDFT` directory (or put it on `PYTHONPATH`).

Example inputs for 14 systems are in `examples/` (`examples/run_all.sh` runs them all with both solvers).

## 2. Tutorial

### 2.1 First calculation

```
python3 -m lrft run H2
```

Prints the grid, basis size, SCF result (total energy in Ha and eV, iterations, wall time), HOMO/LUMO and gap, and the dipole.
Total energy includes the ion–ion repulsion of the Gaussian ion charges.

### 2.2 A larger molecule, both solvers

```
python3 -m lrft run H2O --solver both
```

Output (C++ backend):

```
[ks]   E = -13.22147387 Ha  iterations 14   HOMO -0.511 eV  LUMO 4.048 eV  gap 4.559 eV  dipole 0.8433 Debye
[lres] E = -13.22067894 Ha  iterations 14   chemical potential 1.653 eV              dipole 0.8427 Debye
LRes - KS: dE = 7.949e-04 Ha (9.94e-05 per electron), density diff 7.41e-04 per electron
```

LRes places one window of radius `--rw` (default 10 bohr) around each atom. The difference to KS is the *window error*.

### 2.3 Controlling the window error

Window error falls as `--rw` grows; once a window covers the whole molecule LRes equals KS to machine precision
(H₂, HeH⁺: ~1e-14 Ha). Example on a 12-atom hydrogen chain (alternating 0.75/1.65 Å bonds, file `chain.xyz`):

```
12
H12 chain
H 0 0 0.00
H 0 0 0.75
H 0 0 2.40
H 0 0 3.15
...            (continue, alternating +1.65 / +0.75 Å)
```

```
for rw in 5 8 12; do python3 -m lrft run chain.xyz --solver both --rw $rw | grep "LRes -"; done
```

gives 4.8e-3, 1.5e-4, 9.5e-5 Ha per electron. Use `-M` (number of complex energies, default 64) to trade pole count for
occupation sharpness; small-gap systems need larger M.

### 2.4 Your own geometry

An `.xyz` file in **angstrom**; the second line is a free comment, and may contain `charge=<n>`:

```
2
charge=1 HeH+
He 0 0 0
H  0 0 0.774
```

```
python3 -m lrft run heh.xyz --solver both
python3 -m lrft run H2O --charge 0 --dx 0.25      # override charge, finer grid
```

### 2.5 Output files

```
python3 -m lrft run CH4 --json ch4.json --cube ch4.cube
```

* `ch4.json`: system, grid, and per-solver results (`energy_Ha`, `energy_electronic_Ha`, `E_ion_ion_Ha`, `E_hartree_Ha`,
  `E_xc_Ha`, `iterations`, `converged`, `time_s`, `n_basis`, `homo_Ha`/`lumo_Ha`/`gap_Ha` (KS), `chemical_potential_Ha`
  (LRes), `dipole_au`), plus `comparison` when `--solver both`.
* `ch4.cube`: final density (e/bohr³) in Gaussian cube format for VMD, VESTA, etc.

### 2.6 Convergence checklist

Converge these one at a time and compare energies only at identical settings:

1. `--dx` (grid spacing): 0.3 default; try 0.25, 0.2. Heavier elements need finer grids.
2. `--padding` (vacuum): 6 bohr default. Increase for anions or diffuse systems (periodic images and the neutralising
   background otherwise matter).
3. `--rcut` (basis cutoff): 4.5 default.
4. `--tol` (SCF `|dn|/N`): 1e-6 default; `--mix` lower (0.2) and `--maxiter` higher if the SCF oscillates, `-v` shows every step.
5. For LRes: `--rw` and `-M`, as in 2.3.

## 3. Option reference

| Option | Default | Meaning |
|---|---|---|
| `system` | – | built-in name (`list`) or `.xyz` path |
| `--solver ks\|lres\|both` | `ks` | dense KS, LRes windows, or both with comparison |
| `--charge n` | file/built-in | net charge; electron count must be even |
| `--dx` | 0.3 | grid spacing (bohr) |
| `--padding` | 6.0 | vacuum between molecule and box edge (bohr) |
| `--rcut` | 4.5 | basis cutoff radius (bohr) |
| `--rw` | 10.0 | LRes window radius around each atom (bohr) |
| `-M`, `--poles` | 64 | complex energies in the LRes occupation |
| `--tol`, `--maxiter`, `--mix` | 1e-6, 80, 0.4 | SCF controls (Pulay/DIIS mixing) |
| `--backend` | `auto` | `python`, `cpp`, or `auto` |
| `--threads` | 1 | BLAS/OpenMP/FFT threads |
| `--json FILE`, `--cube FILE` | – | write results / density |
| `-v` | off | per-iteration SCF output |

Exit status: 0 success, 1 input error, 2 SCF not converged.

## 4. What it computes (and does not)

* Periodic box with vacuum, 7-point finite-difference kinetic energy, local pseudopotentials `-Z erf(r/rc)/r`, FFT Hartree
  with a neutralising G=0 background, Slater + PZ81 LDA, Gaussian-type localized basis (s and p) on the grid.
* KS: dense generalized eigenproblem. LRes: occupation `1/(1+u^M)` evaluated on windowed spectra (see the manuscript).
* Closed shell only, no spin polarisation, no forces or geometry optimisation, no nonlocal pseudopotentials, no metals.
* Absolute accuracy against literature is poor with default settings (softened cores; see `LITERATURE_COMPARISON.md`):
  the tool is meant for KS-vs-LRes comparisons and qualitative work. Li–Ne use unfitted valence parameters in `lrft/molecules.py`. Edit the `ELEMENTS` table (valence charge, softening radius `rc`, s/p exponents) to experiment.
* Absolute eigenvalues are defined only up to the G=0 convention; use gaps and differences.

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `C++ extension not built` | run `./build_cpp.sh` or use `--backend python` |
| `only closed-shell ... supported` | choose a charge giving an even electron count |
| `NOT CONVERGED` | lower `--mix`, raise `--maxiter`, refine `--dx`; check the gap with `-v` |
| LRes differs from KS | increase `--rw`; for small gaps also increase `-M` |
| Slow / large memory | coarser `--dx`, smaller `--padding`; grid size grows as (L/dx)³ |
