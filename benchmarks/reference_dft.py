"""All-electron LDA (Slater + PZ81) reference with PySCF, large basis, for validating the CLI Hamiltonian.

Diatomics: bond length from a 7-point energy scan (parabola+cubic fit).  Polyatomics: dipole and
HOMO-LUMO gap at the fixed built-in geometries.  Writes results/reference_dft.json.
Usage: python3 benchmarks/reference_dft.py [basis]
"""
import json
import os
import sys

import numpy as np
from pyscf import dft, gto

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lrft.molecules import MOLECULES  # noqa: E402

BASIS = sys.argv[1] if len(sys.argv) > 1 else "aug-cc-pvqz"
XC = "LDA_X,LDA_C_PZ"


def scf(atoms, charge=0):
    mol = gto.M(atom=[(s, (x, y, z)) for s, x, y, z in atoms], basis=BASIS, charge=charge, verbose=0, unit="Angstrom")
    mf = dft.RKS(mol)
    mf.xc = XC
    mf.grids.level = 5
    mf.conv_tol = 1e-9
    e = mf.kernel()
    nocc = mol.nelectron // 2
    gap = (mf.mo_energy[nocc] - mf.mo_energy[nocc - 1]) * 27.211386
    return e, mf.dip_moment(unit="Debye", verbose=0), gap, mf.mo_energy[nocc - 1] * 27.211386


def dimer(s1, s2, d):
    return [(s1, 0, 0, 0), (s2, 0, 0, d)]


res = {"basis": BASIS, "xc": XC}
for name, (s1, s2, d0) in {"H2": ("H", "H", 0.75), "LiH": ("Li", "H", 1.6), "HF": ("F", "H", 0.92),
                            "N2": ("N", "N", 1.10), "CO": ("C", "O", 1.13)}.items():
    ds = d0 + np.linspace(-0.06, 0.06, 7)
    E = [scf(dimer(s1, s2, d))[0] for d in ds]
    c = np.polyfit(ds, E, 4)
    xs = np.linspace(ds[0], ds[-1], 4001)
    re = float(xs[np.argmin(np.polyval(c, xs))])
    e, dip, gap, homo = scf(dimer(s1, s2, re))
    res[name] = dict(re=re, dipole_D=float(np.linalg.norm(dip)), gap_eV=gap, homo_eV=homo, E_Ha=e)
    print(name, res[name], flush=True)
for name in ("H2O", "NH3", "CH4", "He", "Ne"):
    q, at = MOLECULES[name]
    e, dip, gap, homo = scf(at, q)
    res[name + "_fixed_geom"] = dict(dipole_D=float(np.linalg.norm(dip)), gap_eV=gap, homo_eV=homo, E_Ha=e)
    print(name, res[name + "_fixed_geom"], flush=True)
json.dump(res, open(os.path.join(ROOT, "results", "reference_dft.json"), "w"), indent=1)
