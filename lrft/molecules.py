"""Elements, built-in molecules and geometry I/O for the command-line tool.

The Hamiltonian is the model of lrft.three_d: smooth local pseudopotentials
v(r) = -Z_val erf(r/rc)/r, LDA (Slater + PZ81), closed shell.  The element table
below is a *model* parametrisation (valence charge, softening radius, Gaussian
basis exponents); it is not fitted to any reference and gives qualitative
chemistry only.  H and He are the cases the model is built for.
"""
from __future__ import annotations

import numpy as np

ANG = 1.8897261246257702   # bohr per angstrom

# symbol: (Z_val, rc [bohr], s exponents, p exponents)   (p empty for H, He)
ELEMENTS = {
    "H":  (1, 0.60, (1.2, 0.4, 0.15), (0.5,)),
    "He": (2, 0.50, (2.5, 0.8, 0.3), (1.0,)),
    "Li": (1, 1.10, (0.9, 0.3, 0.1), (0.4, 0.12)),
    "Be": (2, 1.00, (1.1, 0.4, 0.15), (0.5, 0.15)),
    "B":  (3, 0.95, (2.0, 0.7, 0.25, 0.09), (1.2, 0.4, 0.14)),
    "C":  (4, 0.90, (2.4, 0.8, 0.28, 0.1), (1.4, 0.5, 0.17)),
    "N":  (5, 0.85, (2.8, 0.9, 0.3, 0.11), (1.6, 0.55, 0.19)),
    "O":  (6, 0.80, (3.2, 1.0, 0.35, 0.12), (2.0, 0.65, 0.22)),
    "F":  (7, 0.75, (3.6, 1.1, 0.38, 0.13), (2.3, 0.75, 0.25)),
    "Ne": (8, 0.70, (4.0, 1.2, 0.4, 0.14), (2.6, 0.85, 0.28)),
}

# built-in geometries, angstrom: (charge, [(symbol, x, y, z), ...])
_a = 0.9572 * np.sin(np.radians(104.52 / 2)), 0.9572 * np.cos(np.radians(104.52 / 2))
_c = 1.089 / np.sqrt(3)
MOLECULES = {
    "H2":   (0, [("H", 0, 0, -0.3708), ("H", 0, 0, 0.3708)]),
    "He":   (0, [("He", 0, 0, 0)]),
    "HeH+": (1, [("He", 0, 0, 0), ("H", 0, 0, 0.774)]),
    "LiH":  (0, [("Li", 0, 0, 0), ("H", 0, 0, 1.595)]),
    "Ne":   (0, [("Ne", 0, 0, 0)]),
    "HF":   (0, [("F", 0, 0, 0), ("H", 0, 0, 0.917)]),
    "H2O":  (0, [("O", 0, 0, 0), ("H", _a[0], 0, _a[1]), ("H", -_a[0], 0, _a[1])]),
    "NH3":  (0, [("N", 0, 0, 0.1165)] + [("H", 0.9377 * np.cos(t), 0.9377 * np.sin(t), -0.2) for t in (0, 2 * np.pi / 3, 4 * np.pi / 3)]),
    "CH4":  (0, [("C", 0, 0, 0)] + [("H", sx * _c, sy * _c, sx * sy * _c) for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1))]),
    "N2":   (0, [("N", 0, 0, -0.5488), ("N", 0, 0, 0.5488)]),
}


def parse_xyz(text):
    """XYZ text (angstrom) -> list of (symbol, x, y, z).  An optional charge may be
    given as 'charge=<int>' on the comment line."""
    lines = [l for l in text.strip().splitlines()]
    n = int(lines[0].split()[0])
    charge = 0
    for tok in lines[1].replace(",", " ").split():
        if tok.lower().startswith("charge="):
            charge = int(tok.split("=")[1])
    atoms = []
    for l in lines[2:2 + n]:
        f = l.split()
        sym = f[0].capitalize()
        atoms.append((sym, *map(float, f[1:4])))
    if len(atoms) != n:
        raise ValueError(f"XYZ header says {n} atoms, found {len(atoms)}")
    return charge, atoms


def load_molecule(spec):
    """spec: built-in name (case-insensitive) or path to an .xyz file."""
    import os
    for k, v in MOLECULES.items():
        if k.lower() == spec.lower():
            return v
    if os.path.exists(spec):
        return parse_xyz(open(spec).read())
    raise ValueError(f"unknown molecule '{spec}' (not a built-in name or an existing file); see 'list'")


def check_elements(atoms):
    for s, *_ in atoms:
        if s not in ELEMENTS:
            raise ValueError(f"element '{s}' not supported (available: {' '.join(ELEMENTS)})")


def to_xyz(atoms, comment=""):
    return f"{len(atoms)}\n{comment}\n" + "\n".join(f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, x, y, z in atoms) + "\n"
